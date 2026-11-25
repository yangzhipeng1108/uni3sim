#!/usr/bin/env python3
"""Capture the H200 L2 operand-residency curve for a repeated 2R1W stream.

Owner copy of microbench/nvidia_a800/memory/l2/l2_residency_knee_runner.py.
Trials, spread gate and candidate schema are unchanged.  Hardware coupling
changed for H200 only: nvcc 12.9 at the capture host's toolkit path and
-arch=sm_90 (cuobjdump is not installed on the H200 host, so the SASS dump is
replaced by the binary sha256); device name "NVIDIA H200"; resident CTAs
132 * 8; the footprint sweep is redrawn around the 60 MiB H200 L2 with the
same relative density as the A800 sweep around 40 MiB (multiples of 3 MiB,
from 0.1x to 9.6x the L2 capacity); the probe process runs under the SM-clock
gate of ../clock_gated_capture.py and its accepted/rejected statistics are
recorded; the device is selected with CUDA_DEVICE_ORDER=PCI_BUS_ID.

Deviation from the A800 runner (not a hardware constant): a footprint whose
trial spread exceeds MAXIMUM_RELATIVE_SPREAD is recorded in the candidate as
``spread_gate_exceeded`` instead of aborting the capture.  On H200 the
45..54 MiB transition (0.75-0.9x L2) drifts from trial to trial by 9-17
percent in every capture, including captures with 256 instead of 5 warm-up
passes (retained as l2_residency_knee_20260925_{a,b}_warmup5 and
l2_residency_knee_20260925_{a,b}_warmup256), so the A800 abort would leave
the curve unmeasurable; the median still selects the per-footprint rate and
the derivation reports which footprints exceeded the gate.

The probe re-executes one single-pass two-read / one-write stream over the
same three buffers and reports the achieved logical byte rate at each total
footprint.  Sweeping the footprint across the GA100 40 MiB L2 capacity
measures where a repeated same-operand stream stops being served by L2.  The
runner compiles the source, refuses a busy device, records the device state
around every trial, and writes an auditable candidate.
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import clock_gated_capture as gate  # noqa: E402

NVCC = gate.NVCC
DEVICE_NAME = "NVIDIA H200"
SOURCE = Path(__file__).with_name("l2_residency_knee.cu")
RUNNER = Path(__file__).resolve()
FOOTPRINTS_MIB = (
    6, 12, 18, 27, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72,
    81, 90, 108, 144, 288, 576,
)
MAXIMUM_RELATIVE_SPREAD = 0.08


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment, timeout=1800)


def _rate_curve(rows):
    curve = []
    for footprint in FOOTPRINTS_MIB:
        cell = [row for row in rows
                if int(row["total_footprint_mib"]) == footprint]
        if not cell:
            raise ValueError(f"missing footprint {footprint}")
        durations_ns = [float(row["duration_ms"]) * 1.0e6 for row in cell]
        median_ns = statistics.median(durations_ns)
        spread = (max(durations_ns) - min(durations_ns)) / median_ns
        logical_bytes = int(cell[0]["timed_logical_bytes"])
        curve.append({
            "total_footprint_mib": footprint,
            "in_kernel_passes": int(cell[0]["in_kernel_passes"]),
            "timed_logical_bytes": logical_bytes,
            "median_duration_ns": median_ns,
            "relative_spread": spread,
            "spread_gate_exceeded": spread > MAXIMUM_RELATIVE_SPREAD,
            "logical_byte_rate_per_s": logical_bytes / (median_ns * 1.0e-9),
        })
    return curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 11:
        raise ValueError("trials must be within 3..11")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    binary = arguments.output_dir / "l2_residency_knee"
    _capture([
        str(NVCC), "-O3", "-arch=sm_90", "-cudart", "static", "-o", str(binary), str(SOURCE)])

    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stdout, accepted, rejected = gate.run_clock_gated(
        [str(binary), str(arguments.trials),
         ",".join(str(value) for value in FOOTPRINTS_MIB)],
        device=arguments.device,
        log_stem=arguments.output_dir / "l2_residency_knee")
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)

    measurements_path = arguments.output_dir / "measurements.csv"
    measurements_path.write_text(stdout, encoding="utf-8")
    rows = list(csv.DictReader(stdout.splitlines()))
    curve = _rate_curve(rows)

    candidate = {
        "schema": "nvidia_h200_l2_residency_knee_candidate_v1",
        "hardware_id": "nvidia_h200",
        "mechanism": "l2_operand_residency_curve",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {
            "stream": "two_read_one_write_vec4",
            "read_instruction": "ld.global.cg.v4.u32",
            "write_policy": "default_write_back",
            "threads_per_cta": 128,
            "resident_ctas": 132 * 8,
            "warmup_passes": 5,
            "timed_traffic_budget_bytes": 2 * 1024 ** 3,
            "footprints_mib": list(FOOTPRINTS_MIB),
            "timer_domain": "cuda_event_gpu_timeline",
        },
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "trials": arguments.trials,
        "source_sha256": _sha256(SOURCE),
        "runner_sha256": _sha256(RUNNER),
        "binary_sha256": _sha256(binary),
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": [accepted],
                       "rejected": rejected},
        "measurements_sha256": _sha256(measurements_path),
        "curve": curve,
    }
    candidate_path = arguments.output_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {candidate_path}")
    for entry in curve:
        print(f"{entry['total_footprint_mib']:4d} MiB "
              f"{entry['logical_byte_rate_per_s'] / 1e9:8.1f} GB/s "
              f"spread={entry['relative_spread'] * 100:.2f}%")


if __name__ == "__main__":
    main()
