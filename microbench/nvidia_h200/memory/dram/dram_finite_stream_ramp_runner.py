#!/usr/bin/env python3
"""Capture the H200 finite DRAM read-stream bandwidth ramp.

Owner copy of microbench/nvidia_a800/memory/dram/dram_finite_stream_ramp_runner.py.
Sweep, trials, spread gate and candidate schema are unchanged.  Hardware
coupling changed for H200 only: nvcc 12.9 at the capture host's toolkit path
and -arch=sm_90 (cuobjdump is not installed on the H200 host, so the SASS
dump is replaced by the binary sha256); device name "NVIDIA H200"; the
full-chip grid 108 becomes 132 (the other absolute grids are kept);
DRAM_READ_CAPACITY_BYTES_PER_S is the measured H200 saturated read capacity
(dram_read_bw_runner.py capture dram_read_bw_20260925: 4512.6 GB/s, median
of five clock-gated processes at four 512-thread CTAs per SM) instead of the
A800 1124 B/clk x 1.593 GHz; every probe process runs under the SM-clock gate
of ../clock_gated_capture.py and its accepted/rejected statistics are
recorded; the device is selected with CUDA_DEVICE_ORDER=PCI_BUS_ID.

A finite DRAM read stream reaches only part of its asymptotic rate; the
achieved fraction rises with how many bytes each CTA streams.  The probe
measures that ramp at a fixed request depth per CTA over a grid and per-CTA
length sweep.  The runner compiles every configuration from the committed
source, refuses a busy device, records the device state around the sweep and
writes an auditable candidate.  No production shape, operator or duration is
consumed.
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
SOURCE = Path(__file__).with_name("dram_finite_stream_ramp.cu")
RUNNER = Path(__file__).resolve()
GRIDS = (56, 64, 80, 86, 96, 100, 112, 128, 132, 172, 224, 320, 448)
KIB_PER_CTA = (
    256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584,
    4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384,
)
# Bytes in flight per CTA = threads * batch * 16.  Batch 32 gives 64 KiB, the
# depth at which the stream is request-limited rather than issue-limited.
BATCH = 32
THREADS = 128
MAXIMUM_RELATIVE_SPREAD = 0.10
DRAM_READ_CAPACITY_BYTES_PER_S = 4512.6e9


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment, timeout=3600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=9)
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 15:
        raise ValueError("trials must be within 3..15")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    before = gate.require_idle(arguments.device)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    header = None
    lines = []
    binary_digests = {}
    clocks = []
    rejected = []
    for grid in GRIDS:
        binary = arguments.output_dir / f"ramp_grid{grid}"
        _capture([
            str(NVCC), "-O3", "-arch=sm_90", "-cudart", "static",
            f"-DPROBE_GRID={grid}",
            f"-DPROBE_BATCH={BATCH}", "-o", str(binary), str(SOURCE)])
        binary_digests[binary.name] = _sha256(binary)
        stdout, accepted, rejects = gate.run_clock_gated(
            [str(binary), "0", str(arguments.trials),
             ",".join(str(value) for value in KIB_PER_CTA)],
            device=arguments.device,
            log_stem=arguments.output_dir / f"ramp_grid{grid}")
        clocks.append(dict(accepted, grid=grid))
        rejected.extend(dict(r, grid=grid) for r in rejects)
        output = stdout.splitlines()
        if header is None:
            header = output[0]
        elif output[0] != header:
            raise ValueError("probe changed its output schema mid-sweep")
        lines.extend(output[1:])
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = gate.require_idle(arguments.device, settle_seconds=30.0)

    measurements_path = arguments.output_dir / "measurements.csv"
    measurements_path.write_text(
        "\n".join([header] + lines) + "\n", encoding="utf-8")
    rows = list(csv.DictReader([header] + lines))

    ramp = []
    excluded = []
    for grid in GRIDS:
        for kib in KIB_PER_CTA:
            cell = [
                row for row in rows
                if int(row["grid"]) == grid and int(row["kib_per_cta"]) == kib
            ]
            if not cell:
                raise ValueError(f"missing cell grid={grid} kib={kib}")
            durations_ns = [float(row["duration_ms"]) * 1.0e6 for row in cell]
            median = statistics.median(durations_ns)
            spread = (max(durations_ns) - min(durations_ns)) / median
            logical_bytes = int(cell[0]["logical_bytes"])
            if spread > MAXIMUM_RELATIVE_SPREAD:
                excluded.append({
                    "grid_ctas": grid, "kib_per_cta": kib,
                    "relative_spread": spread})
                continue
            achieved = logical_bytes / (median * 1.0e-9)
            ramp.append({
                "grid_ctas": grid,
                "bytes_per_cta": kib * 1024,
                "logical_bytes": logical_bytes,
                "median_duration_ns": median,
                "relative_spread": spread,
                "achieved_bytes_per_s": achieved,
                "achieved_fraction_of_dram_read_capacity": (
                    achieved / DRAM_READ_CAPACITY_BYTES_PER_S),
            })

    candidate = {
        "schema": "nvidia_h200_dram_finite_stream_ramp_candidate_v1",
        "hardware_id": "nvidia_h200",
        "mechanism": "dram_finite_read_stream_ramp",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {
            "threads_per_cta": THREADS,
            "outstanding_loads_per_thread": BATCH,
            "in_flight_bytes_per_cta": THREADS * BATCH * 16,
            "load_instruction": "ld.global.cg.v4.u32",
            "grids": list(GRIDS),
            "kib_per_cta": list(KIB_PER_CTA),
            "dram_read_capacity_bytes_per_s": DRAM_READ_CAPACITY_BYTES_PER_S,
            "timer_domain": "cuda_event_gpu_timeline",
        },
        "device_state": {"before": before, "after": after},
        "timing": {"started_at": started_at, "completed_at": completed_at},
        "trials": arguments.trials,
        "source_sha256": _sha256(SOURCE),
        "runner_sha256": _sha256(RUNNER),
        "binary_sha256": binary_digests,
        "clock_gate": {"boost_mhz": gate.BOOST_MHZ, "accepted": clocks,
                       "rejected": rejected},
        "measurements_sha256": _sha256(measurements_path),
        "ramp": ramp,
        "excluded_unstable_cells": excluded,
    }
    candidate_path = arguments.output_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {candidate_path} ramp={len(ramp)} excluded={len(excluded)}")


if __name__ == "__main__":
    main()
