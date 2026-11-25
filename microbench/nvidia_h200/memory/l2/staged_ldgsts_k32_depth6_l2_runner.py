#!/usr/bin/env python3
"""Build, capture and publish the H200 K32 depth-6 staged LDGSTS L2 probe.

Owner copy of microbench/nvidia_a800/memory/l2/staged_ldgsts_k32_depth6_l2_runner.py.
Capture contract (seven event-timed trials, 10% stability gate, median trial
per capture, minimum over two independent captures) is unchanged.  Hardware
coupling: grid 264 (two resident CTAs on 132 SMs), H200 GPU 2 only, CUDA 12.9
sm_90 build, 50-ms SM-clock gate (see staged_ldgsts_l2_runner.py), per-clock
value at the measured 1.98-GHz SM clock.  The 77,824-B padded reservation
keeps the A800 two-CTA residency on the 228-KiB H200 SM (see the probe
header); the runner still rejects any row that does not report two.
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
    SM_CLOCK_HZ,
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


SOURCE = Path(__file__).with_name("staged_ldgsts_k32_depth6_l2.cu")
RUNNER = Path(__file__).resolve()
GRID = 264
LOGICAL_BYTES = GRID * 4096 * (12 * 1024)


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "staged_ldgsts_k32_depth6_l2"
    compile_command, compiler_stderr = _compile(SOURCE, binary)
    sass = _dump_sass(binary, output / "probe.sass.txt")
    if "LDGSTS" not in sass or "LDL" in sass or "STL" in sass:
        raise ValueError("compiled SASS violates the LDGSTS contract")
    command = [str(binary), "0", str(args.trials)]
    result, attempts = _gated_run(
        command, device=args.device, log_stem=output / "probe")
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != args.trials:
        raise ValueError("probe trial count changed")
    durations = []
    for trial, row in enumerate(rows):
        if (int(row["trial"]) != trial
                or int(row["logical_bytes"]) != LOGICAL_BYTES
                or int(row["resident_ctas_per_sm"]) != 2):
            raise ValueError("probe row violates the frozen contract")
        duration_ms = float(row["duration_ms"])
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError("probe duration must be finite and positive")
        durations.append(duration_ms)
    center = statistics.median(durations)
    spread = (max(durations) - min(durations)) / center
    if spread > 0.10:
        raise ValueError(f"probe stability gate failed: {spread}")
    (output / "raw.csv").write_text(result.stdout, encoding="utf-8")
    device_after = _wait_for_idle(args.device)
    provenance = {
        "schema": "nvidia_h200_staged_ldgsts_k32_depth6_l2_service_v1",
        "hardware": "nvidia_h200",
        "owner": "h200-port-wp3_ldgsts",
        "device_before": device_before,
        "device_after": device_after,
        "source_commit": args.source_commit,
        "source_sha256": _sha256(SOURCE),
        "runner_sha256": _sha256(RUNNER),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(output / "probe.sass.txt"),
        "compile_command": compile_command,
        "compiler_stderr": compiler_stderr,
        "attempts": attempts,
        "clock_gate": _clock_summary([attempts]),
        "pid": os.getpid(),
        "trials": args.trials,
        "logical_bytes_per_trial": LOGICAL_BYTES,
        "duration_ms": durations,
        "relative_range": spread,
        "logical_byte_capacity_per_s": LOGICAL_BYTES / (center * 1e-3),
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output / "provenance.json")


def publish(args):
    captures = []
    clock_attempts = []
    commits = set()
    for path in args.capture:
        capture_dir, provenance_path, provenance = _load_capture(
            path, "provenance.json")
        commits.add(provenance["source_commit"])
        clock_attempts.append(provenance["attempts"])
        captures.append({
            "path": _relative(capture_dir),
            "device_index": provenance["device_before"]["index"],
            "device_uuid": provenance["device_before"]["uuid"],
            "provenance_sha256": _sha256(provenance_path),
            "raw_sha256": _sha256(capture_dir / "raw.csv"),
            "sass_sha256": provenance["sass_sha256"],
            "logical_byte_capacity_per_s": provenance[
                "logical_byte_capacity_per_s"],
            "relative_trial_spread": provenance["relative_range"],
        })
    if len(captures) < 2 or len(commits) != 1:
        raise ValueError("publication needs two captures of one source commit")
    selected = min(item["logical_byte_capacity_per_s"] for item in captures)
    reference = {
        "schema": "nvidia_h200_staged_ldgsts_k32_depth6_l2_service_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": DEVICE_NAME,
        "compute_capability": "sm_90",
        "source_baseline_commit": commits.pop(),
        "source": _relative(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "runner": _relative(RUNNER),
        "runner_sha256": _sha256(RUNNER),
        "captures": captures,
        "probe_contract": {
            "dtype": "fp16",
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 32,
            "threads_per_cta": 128,
            "pipeline_slots": 6,
            "resident_ctas_per_sm": 2,
            "physical_sm_count": SM_COUNT,
            "grid_ctas": GRID,
            "panel_bytes": 12288,
            "shared_bytes_per_cta": 77824,
            "backing_bytes": 33554432,
            "rounds": 4096,
            "load_instruction": "LDGSTS",
            "cache_operator": "cg",
        },
        "selected_service": {
            "selection_rule": (
                "minimum complete-capture capacity across two independent "
                "clock-accepted capture processes on H200 GPU 2"),
            "logical_byte_capacity_per_s": selected,
            "throughput_per_sm_clock": selected / SM_CLOCK_HZ,
            "scope": "full_chip",
        },
        "applicability": {
            "dtype": "fp16",
            "tile_shape": [64, 128, 32],
            "threads_per_cta": 128,
            "pipeline_slots": 6,
            "minimum_ctas_per_sm": 2,
            "required_launch_algorithm": "observed_gemm:gemm",
            "required_operand_streams": 2,
            "required_source_evidence": (
                "six-slot CUTLASS SM80 A/B operand staging with LDGSTS"),
        },
        "decision": {
            "status": "accepted_mechanism_capacity",
            "forward_eligible": True,
            "reason": (
                "The workload-independent staged copy is stable in two "
                "independent clock-gated H200 captures at the A800 source "
                "class's two-resident-CTA topology and provides only the exact "
                "source-class L2 read service ceiling."),
        },
        "excluded_conclusions": [
            "a complete production GEMM duration",
            "a generic L2 capacity replacement",
            "write or read/write-duplex capacity",
            "a per-shape timing correction",
            "an H200 three-resident-CTA topology (the H200 SM could hold three "
            "74,752-B CTAs; the probe pins two by construction)",
        ],
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "sm_clock_hz_for_per_clock_values": SM_CLOCK_HZ,
            "clock_gate": _clock_summary(clock_attempts),
            "residency_note": (
                "dynamic shared reservation padded 74,752 -> 77,824 B so the "
                "occupancy query reports exactly two CTAs per H200 SM, as on "
                "A800; the six 12-KiB ring slots are unchanged"),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("staged_ldgsts_k32_depth6_l2_service_sm90.json", reference)


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
