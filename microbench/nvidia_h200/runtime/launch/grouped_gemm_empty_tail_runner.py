#!/usr/bin/env python3
"""Run the H200 source-ordered empty-tail probe and publish its reference.

The A800 probe ``microbench/nvidia_a800/runtime/launch/grouped_gemm_empty_tail.cu``
had no runner; its reference ``bm16_grouped_gemm_empty_tail_sm80.json``
states the procedure, which this runner scripts unchanged:
  - zero-work cells are the ``active_rows = 0`` launches (every CTA takes the
    early return); per capture and grid the completion is the median over all
    trials and hold settings;
  - the envelope is the maximum across two independent captures (A800: two
    idle devices; H200: two clock-accepted processes on the owned GPU);
  - ``empty_tail_round_ns`` is the envelope slope between the frozen
    1280-CTA and 32768-CTA training endpoints per ``SM_COUNT``-wide round,
    i.e. (E[32768] - E[1280]) / (ceil(32768/132) - ceil(1280/132));
  - the 8192-CTA cell is held out and only evaluated.
Hardware coupling: SM_COUNT 132 (was 108), sm_90 build with the H200-host
CUDA 12.9 nvcc, and a 50-ms nvidia-smi clock log per capture; a capture with
any busy sample below 1980 MHz is rejected.

Subcommands:
  run     (H200 host) compile, then clock-gated captures into --output-dir
  publish (local)     derive the reference from the copied run directory
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


SOURCE = Path(__file__).with_name("grouped_gemm_empty_tail.cu")
RUNNER = Path(__file__)
ROOT = Path(__file__).resolve().parents[4]
SM_COUNT = 132
BOOST_MHZ = 1980
TRIALS = 9
REQUIRED_ACCEPTED = 2
MAXIMUM_ATTEMPTS = 6
TRAINING_GRIDS = (1280, 32768)
VALIDATION_GRID = 8192
GRIDS = (1280, 8192, 32768)
HOLD_CYCLES = (0, 7000, 14000, 28000)
REFERENCE = "microbench/nvidia_h200/references/bm16_grouped_gemm_empty_tail_sm90.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clocked(command, environment, device, clock_path):
    with clock_path.open("w", encoding="utf-8") as clock_stream:
        sampler = subprocess.Popen(
            ["nvidia-smi", "-i", str(device),
             "--query-gpu=clocks.sm,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=clock_stream, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.3)
            result = subprocess.run(
                command, text=True, capture_output=True, env=environment)
        finally:
            time.sleep(0.2)
            sampler.terminate()
            sampler.wait()
    busy = []
    samples = 0
    for line in clock_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            clock, utilization = (int(item) for item in line.split(","))
            samples += 1
            if utilization > 0:
                busy.append(clock)
    return result, {
        "samples": samples, "busy_samples": len(busy),
        "busy_min_mhz": min(busy) if busy else None,
        "busy_median_mhz": statistics.median(busy) if busy else None,
        "busy_max_mhz": max(busy) if busy else None,
        "accepted": bool(busy) and min(busy) >= BOOST_MHZ,
    }


def _parse(text):
    meta = {}
    rows = []
    for fields in csv.reader(text.splitlines()):
        if fields[0] == "meta":
            meta[fields[1]] = fields[2]
        elif fields[0] == "data" and fields[1] != "trial":
            rows.append({
                "trial": int(fields[1]), "grid": int(fields[2]),
                "active_rows": int(fields[3]), "num_pid_n": int(fields[4]),
                "hold_cycles": int(fields[5]),
                "event_duration_ns": float(fields[6]),
            })
        elif fields[0] != "data":
            raise ValueError(f"unexpected probe row: {fields}")
    if (int(meta["sm_count"]) != SM_COUNT
            or int(meta["resident_ctas_per_sm"]) != 10):
        raise ValueError(f"probe contract changed: {meta}")
    return meta, rows


def run(arguments):
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = output / "grouped_gemm_empty_tail"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_90",
        str(SOURCE), "-o", str(binary)]
    subprocess.run(compile_command, check=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device)
    command = [str(binary), "0", str(TRIALS), "wp2_moe_cuda",
               arguments.baseline]
    captures = []
    accepted = 0
    for attempt in range(MAXIMUM_ATTEMPTS):
        if accepted == REQUIRED_ACCEPTED:
            break
        result, clock = _clocked(
            command, environment, arguments.device,
            output / f"capture{attempt}.clock.csv")
        if result.returncode != 0 or result.stderr.strip():
            raise RuntimeError(f"probe failed: {result.stderr}")
        raw = output / f"capture{attempt}.raw.csv"
        raw.write_text(result.stdout, encoding="utf-8")
        _parse(result.stdout)
        status = "accepted" if clock["accepted"] else "rejected_clock"
        accepted += status == "accepted"
        captures.append({"capture": attempt, "status": status,
                         "clock": clock, "raw_sha256": _sha256(raw)})
    (output / "summary.json").write_text(json.dumps({
        "compile_command": compile_command, "command": command,
        "device": arguments.device, "binary_sha256": _sha256(binary),
        "source_sha256": _sha256(SOURCE), "captures": captures,
        "accepted_captures": accepted,
        "rejected_captures": len(captures) - accepted,
    }, indent=2) + "\n", encoding="utf-8")
    if accepted < REQUIRED_ACCEPTED:
        raise RuntimeError("fewer than two clock-accepted captures")


def _rounds(grid):
    return math.ceil(grid / SM_COUNT)


def publish(arguments):
    run_dir = arguments.run_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text())
    accepted = [c for c in summary["captures"] if c["status"] == "accepted"]
    if len(accepted) != REQUIRED_ACCEPTED:
        raise ValueError("publish requires exactly two accepted captures")
    zero_work = []
    mixed_ok = []
    capture_entries = []
    for capture in accepted:
        raw = run_dir / f"capture{capture['capture']}.raw.csv"
        if _sha256(raw) != capture["raw_sha256"]:
            raise ValueError("raw capture changed after the run")
        meta, rows = _parse(raw.read_text())
        medians = {}
        for grid in GRIDS:
            values = [r["event_duration_ns"] for r in rows
                      if r["grid"] == grid and r["active_rows"] == 0]
            if len(values) != TRIALS * len(HOLD_CYCLES):
                raise ValueError("zero-work matrix incomplete")
            medians[grid] = statistics.median(values)
        zero_work.append(medians)

        def active(grid, hold):
            values = [r["event_duration_ns"] for r in rows
                      if r["grid"] == grid and r["active_rows"] == 320
                      and r["hold_cycles"] == hold]
            if len(values) != TRIALS:
                raise ValueError("mixed-prefix matrix incomplete")
            return statistics.median(values)
        # Partial overlap: with a busy 320-row prefix the extra completion
        # of a larger grid is smaller than the zero-work tail growth.
        for grid in (VALIDATION_GRID, TRAINING_GRIDS[1]):
            for hold in HOLD_CYCLES[1:]:
                mixed_ok.append(
                    active(grid, hold) - active(TRAINING_GRIDS[0], hold)
                    < medians[grid] - medians[TRAINING_GRIDS[0]])
        capture_entries.append({
            "path": str((run_dir / raw.name).relative_to(ROOT)),
            "remote_path": f"{arguments.remote_run_dir}/{raw.name}",
            "capture": capture["capture"],
            "device_index": summary["device"],
            "device_name": meta["device_name"],
            "raw_sha256": capture["raw_sha256"],
            "binary_sha256": summary["binary_sha256"],
            "clock": capture["clock"],
            "zero_work_median_ns": {str(g): medians[g] for g in GRIDS},
        })
    envelope = {grid: max(m[grid] for m in zero_work) for grid in GRIDS}
    low, high = TRAINING_GRIDS
    slope = (envelope[high] - envelope[low]) / (_rounds(high) - _rounds(low))
    predicted = envelope[low] + slope * (
        _rounds(VALIDATION_GRID) - _rounds(low))
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        text=True, capture_output=True).stdout.strip()
    reference = {
        "schema": "nvidia_h200_bm16_grouped_gemm_empty_tail_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "software_revision": "vllm_0.24.0_triton_3.6_sm90",
        "source_baseline_commit": commit,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": summary["source_sha256"],
        "production_source": arguments.production_source,
        "production_source_sha256": arguments.production_source_sha256,
        "admission_order_reference": None,
        "admission_order_reference_sha256": None,
        "captures": capture_entries,
        "probe_contract": {
            "threads_per_cta": 128,
            "resident_ctas_per_sm": 10,
            "physical_sm_count": SM_COUNT,
            "active_prefix_rows": 320,
            "tile_m": 16,
            "num_pid_n": 64,
            "training_grid_ctas": list(TRAINING_GRIDS),
            "validation_grid_ctas": [VALIDATION_GRID],
            "hold_cycles": list(HOLD_CYCLES),
            "production_routing_sort_reduce_absent": True,
            "production_or_evaluation_duration_consumed": False,
        },
        "zero_work_cross_device_upper_envelope_ns": {
            str(g): envelope[g] for g in GRIDS},
        "service_parameters": {
            "empty_tail_round_ns": slope,
            "round_width_ctas": SM_COUNT,
            "derivation": (
                "Slope of the cross-capture maximum zero-work completion "
                "envelope between the frozen 1280-CTA and 32768-CTA training "
                "endpoints per 132-CTA round: (E[32768]-E[1280]) / "
                "(ceil(32768/132)-ceil(1280/132)); zero-work cell = median "
                "over trials and hold settings of the active_rows=0 launch; "
                "the 8192-CTA cell is held out."),
        },
        "validation": {
            "grid_ctas": VALIDATION_GRID,
            "observed_cross_device_upper_envelope_ns":
                envelope[VALIDATION_GRID],
            "predicted_ns": predicted,
            "absolute_relative_error": abs(
                predicted - envelope[VALIDATION_GRID])
                / envelope[VALIDATION_GRID],
            "mixed_prefix_sweep_confirms_partial_overlap": all(mixed_ok),
        },
        "decision": {
            "status": "accepted_independent_scheduler_mechanism",
            "forward_eligible": True,
            "allowed_use": (
                "Price only the source-ordered early-return CTA tail of the "
                "exact BF16 BM16/BN32/BK64/GSM1 grouped-GEMM pipeline after "
                "one full resident cohort has entered the active K loop."),
            "excluded_conclusions": [
                "a complete FusedMoE operator duration",
                "a production-shape latency lookup",
                "a residual correction",
                "tile signatures other than BF16 BM16/BN32/BK64/GSM1",
                "resources or devices outside the recorded H200 contract",
            ],
        },
        "h200_capture_provenance": {
            "git_commit": commit,
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha256(RUNNER),
            "compile_command": summary["compile_command"],
            "command": summary["command"],
            "raw_dir": str(run_dir.relative_to(ROOT)),
            "remote_raw_dir": arguments.remote_run_dir,
            "clock_gate": (
                "nvidia-smi clocks.sm/utilization.gpu every 50 ms; capture "
                f"rejected when any busy sample < {BOOST_MHZ} MHz"),
            "captures": summary["captures"],
            "accepted_captures": summary["accepted_captures"],
            "rejected_captures": summary["rejected_captures"],
            "admission_order_reference_note": (
                "the A800 heterogeneous CTA admission reference has no H200 "
                "counterpart in this work package; recorded as null"),
            "occupancy_note": (
                "dynamic shared memory 15 KiB -> 21 KiB keeps 10 resident "
                "CTAs/SM on the 228-KiB H200 SM, matching the H200 Triton 3.6 "
                "W2 BM16 compile facts (45 registers -> 10 CTAs/SM)"),
        },
    }
    (ROOT / REFERENCE).write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reference["service_parameters"], indent=2))
    print(json.dumps(reference["validation"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--device", type=int, required=True)
    run_parser.add_argument("--nvcc", type=Path, required=True)
    run_parser.add_argument("--baseline", required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument("--remote-run-dir", required=True)
    publish_parser.add_argument("--production-source", required=True)
    publish_parser.add_argument("--production-source-sha256", required=True)
    arguments = parser.parse_args()
    if arguments.command == "run":
        run(arguments)
    else:
        publish(arguments)


if __name__ == "__main__":
    main()
