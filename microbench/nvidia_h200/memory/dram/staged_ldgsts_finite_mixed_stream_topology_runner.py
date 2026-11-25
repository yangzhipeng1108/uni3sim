#!/usr/bin/env python3
"""Sweep and publish the H200 finite mixed-tier LDGSTS stream startup.

Owner copy of
microbench/nvidia_a800/memory/dram/staged_ldgsts_finite_mixed_stream_topology_runner.py.
Per-(grid, slots, rounds) probe processes, the five L2 fractions, row
validation and per-cell medians are unchanged.  The A800 reference derived
its startup by hand; ``publish`` scripts exactly its stated rule:

* finite residual = median duration at 32 rounds - same-grid zero-round
  median - 32 x (256-to-512 steady-round secant);
* the parameter is the residual at one full launch-local cohort (one
  resident CTA on every SM), the midpoint mixed tier (50% L2) and the
  canonical three-complete-slot ring;
* the A800 single device becomes two independent capture processes on GPU 2;
  the per-cell median pools the trials of both captures.

Hardware coupling: the A800 grids 64/96/108/216/304 (108 = one wave,
216 = two waves, 304 = 2.81 waves) become 64/96/132/264/371 on the 132-SM
H200, the five-slot depth check runs at 132/264 instead of 108/216, H200 GPU 2
only, CUDA 12.9 sm_90 build, 50-ms SM-clock gate per probe process (helpers
from ../l2/staged_ldgsts_l2_runner.py), and the probe pads its dynamic
reservation to keep one resident CTA per SM.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l2"))
from staged_ldgsts_l2_runner import (  # noqa: E402
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


SOURCE = Path(__file__).with_name("staged_ldgsts_finite_mixed_stream.cu")
RUNNER = Path(__file__).resolve()
PANEL_BYTES = 16 * 1024
ROUNDS = (0, 32, 256, 512)
FRACTIONS = (("l2_0pct", 0), ("l2_25pct", 25), ("l2_50pct", 50),
             ("l2_75pct", 75), ("l2_100pct", 100))
# (grid, slots) cells: 3-slot wave scaling plus the 5-slot ring-depth check.
CELLS = ((64, 3), (96, 3), (132, 3), (264, 3), (371, 3), (132, 5), (264, 5))
ONE_WAVE_GRID = SM_COUNT


def _parse(output, trials, rounds, grid):
    rows = list(csv.DictReader(output.splitlines()))
    known_fractions = dict(FRACTIONS)
    parsed = []
    for row in rows:
        fraction_name = row["l2_fraction"]
        trial = int(row["trial"])
        if fraction_name not in known_fractions or trial not in range(trials):
            raise ValueError(f"unexpected probe row: {row}")
        if int(row["rounds"]) != rounds:
            raise ValueError(f"probe rounds mismatch: {row}")
        if int(row["total_bytes"]) != grid * rounds * PANEL_BYTES:
            raise ValueError(f"probe traffic mismatch: {row}")
        if int(row["resident_ctas_per_sm"]) != 1:
            raise ValueError(f"probe residency mismatch: {row}")
        parsed.append({
            "l2_fraction_pct": known_fractions[fraction_name],
            "trial": trial,
            "rounds": rounds,
            "duration_ms": float(row["duration_ms"]),
        })
    expected = {(fraction, t) for _, fraction in FRACTIONS
                for t in range(trials)}
    if {(r["l2_fraction_pct"], r["trial"]) for r in parsed} != expected:
        raise ValueError("probe capture matrix is incomplete")
    return parsed


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    rows = []
    command_records = []
    binaries = {}
    for grid, slots in CELLS:
        binary = output / f"probe_g{grid}_s{slots}_k0"
        compile_command, compiler_stderr = _compile(
            SOURCE, binary, f"-DPROBE_GRID={grid}", f"-DPROBE_SLOTS={slots}",
            "-DPROBE_KSYNC=0")
        sass_path = output / f"probe_g{grid}_s{slots}_k0.sass.txt"
        sass = _dump_sass(binary, sass_path)
        required = ("LDGSTS.E.BYPASS.128", "LDGDEPBAR", "DEPBAR.LE SB0, 0x1")
        missing = [opcode for opcode in required if opcode not in sass]
        if missing or "LDL" in sass or "STL" in sass:
            raise ValueError(f"compiled SASS violates contract: {missing}")
        binaries[f"grid{grid}_slots{slots}"] = {
            "compile_command": compile_command,
            "compiler_stderr": compiler_stderr,
            "binary_sha256": _sha256(binary),
            "sass_sha256": _sha256(sass_path),
        }
        for rounds in ROUNDS:
            command = [str(binary), "0", str(args.trials), str(rounds)]
            captured, attempts = _gated_run(
                command, device=args.device,
                log_stem=output / f"probe_g{grid}_s{slots}_r{rounds}")
            for row in _parse(captured.stdout, args.trials, rounds, grid):
                row["grid"] = grid
                row["slots"] = slots
                rows.append(row)
            command_records.append({
                "grid": grid, "slots": slots, "rounds": rounds,
                "attempts": attempts, "stderr": captured.stderr,
            })
    (output / "raw.csv").write_text(
        "grid,slots,l2_fraction_pct,trial,rounds,duration_ms\n"
        + "".join(
            f'{r["grid"]},{r["slots"]},{r["l2_fraction_pct"]},{r["trial"]},'
            f'{r["rounds"]},{r["duration_ms"]:.9f}\n'
            for r in rows), encoding="utf-8")
    device_after = _wait_for_idle(args.device)
    report = {
        "schema": "nvidia_h200_finite_mixed_stream_topology_candidate_v1",
        "hardware_id": "nvidia_h200",
        "mechanism": "finite_async_operand_stream_completion_topology",
        "production_or_evaluation_duration_consumed": False,
        "scope": {
            "threads_per_cta": 128, "resident_ctas_per_sm": 1,
            "operand_pair_bytes": PANEL_BYTES,
            "k_synchronized_order": False,
            "cells": [list(cell) for cell in CELLS],
            "rounds": list(ROUNDS),
            "l2_fraction_pct": [fraction for _, fraction in FRACTIONS],
        },
        "provenance": {
            "owner": "h200-port-wp3_ldgsts",
            "source_commit": args.source_commit,
            "device_before": device_before, "device_after": device_after,
            "binaries": binaries,
            "commands": command_records,
            "clock_gate": _clock_summary(
                [record["attempts"] for record in command_records]),
            "pid": os.getpid(),
            "source_sha256": _sha256(SOURCE), "runner_sha256": _sha256(RUNNER),
        },
    }
    candidate = output / "candidate.json"
    candidate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(candidate)


def _read_raw(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "grid": int(row["grid"]), "slots": int(row["slots"]),
                "l2_fraction_pct": int(row["l2_fraction_pct"]),
                "rounds": int(row["rounds"]),
                "duration_ns": float(row["duration_ms"]) * 1.0e6,
            })
    return rows


def _residual(rows, grid, slots, fraction):
    def median(rounds):
        values = [
            row["duration_ns"] for row in rows
            if row["grid"] == grid and row["slots"] == slots
            and row["l2_fraction_pct"] == fraction and row["rounds"] == rounds]
        if not values:
            raise ValueError(f"missing cell {grid}/{slots}/{fraction}/{rounds}")
        return statistics.median(values)
    secant = (median(512) - median(256)) / (512 - 256)
    return median(32) - median(0) - 32 * secant


def publish(args):
    captures = []
    rows = []
    clock_attempts = []
    for path in args.capture:
        capture_dir, candidate_path, candidate = _load_capture(
            path, "candidate.json")
        clock_attempts.extend(
            record["attempts"]
            for record in candidate["provenance"]["commands"])
        rows.extend(_read_raw(capture_dir / "raw.csv"))
        captures.append({
            "path": _relative(capture_dir),
            "device_index": candidate["provenance"]["device_before"]["index"],
            "raw_sha256": _sha256(capture_dir / "raw.csv"),
            "candidate_sha256": _sha256(candidate_path),
            "source_commit": candidate["provenance"]["source_commit"],
            "grids": sorted({grid for grid, _ in CELLS}),
            "complete_operand_pair_slots": sorted({slots for _, slots in CELLS}),
        })
    if len(captures) < 2:
        raise ValueError("publication needs two independent captures")
    fractions = [fraction for _, fraction in FRACTIONS]
    slots3 = {
        str(grid): {
            str(fraction): _residual(rows, grid, 3, fraction)
            for fraction in fractions}
        for grid, slots in CELLS if slots == 3}
    depth_check = {
        f"{grid}_slots5": {
            str(fraction): _residual(rows, grid, 5, fraction)
            for fraction in fractions}
        for grid, slots in CELLS if slots == 5}
    startup = slots3[str(ONE_WAVE_GRID)]["50"]
    reference = {
        "schema": "finite_mixed_operand_stream_startup_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": DEVICE_NAME,
        "mechanism": "finite_mixed_operand_stream_startup",
        "production_or_evaluation_duration_consumed": False,
        "source": _relative(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "runner": _relative(RUNNER),
        "runner_sha256": _sha256(RUNNER),
        "captures": captures,
        "probe_contract": {
            "description": (
                "Copy-only finite staged-LDGSTS operand stream. A 128-thread "
                "CTA moves private 16-KiB two-byte-element operand panels "
                "through a complete-pair ring with LDGSTS.E.BYPASS.128 / "
                "LDGDEPBAR / DEPBAR, replicating the launch geometry and ring "
                "structure of the sm80 128-thread K64 staged GEMM source class "
                "without any matrix instruction, output epilogue, production "
                "shape, or production duration. Rounds 0 gives the same-grid "
                "empty launch plus ring prologue; the 256-to-512 steady-round "
                "secant gives the per-tier steady service; the finite residual "
                "follows the methodology of the accepted A800 "
                "staged_ldgsts_depth3_wave_scaling_sm80 reference."),
            "timer_domain": "cuda_event_gpu_timeline",
            "threads_per_cta": 128,
            "resident_ctas_per_sm": 1,
            "operand_pair_bytes": PANEL_BYTES,
            "rounds": list(ROUNDS),
            "l2_fraction_pct": fractions,
            "methodology_reference": (
                "microbench/nvidia_a800/references/"
                "staged_ldgsts_depth3_wave_scaling_sm80.json"),
        },
        "derived_model": {
            "finite_residual_definition": (
                "median duration at 32 rounds minus the same-grid zero-round "
                "launch and the 256-to-512 steady-round secant"),
            "residual_ns_by_grid_and_l2_fraction_slots3": slots3,
            "residual_ns_ring_depth_check": depth_check,
            "selection_rule": (
                "The staged GEMM operand stream is intrinsically mixed-tier: "
                "the shared A panel is L2-resident while each CTA's B panel "
                "strip is DRAM-unique. The parameter is the finite residual at "
                f"one full launch-local cohort ({ONE_WAVE_GRID} CTAs = one "
                f"resident CTA on each of {SM_COUNT} H200 SMs), the midpoint "
                "mixed tier (50 pct L2), and the canonical three-complete-slot "
                "ring, with per-cell medians pooled over two independent "
                "clock-accepted capture processes."),
            "one_wave_mixed_tier_startup_ns": startup,
            "boundedness": (
                "Same-cell (50 pct L2, three slots) residuals on H200: "
                + ", ".join(
                    f"grid {grid}: {slots3[str(grid)]['50']:.1f} ns"
                    for grid, slots in CELLS if slots == 3)
                + "; five-slot ring: "
                + ", ".join(
                    f"grid {key.split('_')[0]}: {value['50']:.1f} ns"
                    for key, value in depth_check.items())
                + "; pure-DRAM (0 pct L2) three-slot residuals: "
                + ", ".join(
                    f"grid {grid}: {slots3[str(grid)]['0']:.1f} ns"
                    for grid, slots in CELLS if slots == 3)
                + "."),
            "formula": (
                "prediction = common_terms + one_wave_mixed_tier_startup_ns, "
                "once per launch"),
        },
        "scope": {
            "dtype_element_bytes": 2,
            "threads_per_cta": 128,
            "tile_k": 64,
            "minimum_pipeline_depth": 4,
            "maximum_pipeline_depth": 6,
            "minimum_rounds": 16,
            "maximum_rounds": 512,
            "one_wave_grid_ctas": ONE_WAVE_GRID,
        },
        "allowed_use": (
            "One launch-local additive startup for a staged K64-tile "
            "128-thread two-byte-element GEMM source program whose operand "
            "read stream carries both L2-hit and DRAM-miss bytes, composed "
            "once per launch. The probe is byte-level, so the two-byte scope "
            "covers fp16 and bf16 element encodings of the same panel "
            "geometry."),
        "forbidden_use": (
            "A production GEMM duration, a per-shape or per-bucket correction, "
            "a generic DRAM or L2 capacity replacement, a per-grid or per-wave "
            "latency lookup, or a second serial charge of work already inside "
            "the common K-tile pipeline. Pure-DRAM operand streams are outside "
            "scope."),
        "provenance": {
            "clock_gate": _clock_summary(clock_attempts),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference(
        "finite_mixed_operand_stream_startup_sm90.json", reference)


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
        if not 3 <= args.trials <= 11:
            raise ValueError("trials must be in [3, 11]")
        capture(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
