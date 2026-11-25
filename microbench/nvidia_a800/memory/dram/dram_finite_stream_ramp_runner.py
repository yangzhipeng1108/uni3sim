#!/usr/bin/env python3
"""Capture the A800 finite DRAM read-stream bandwidth ramp.

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
import time
from pathlib import Path

NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
SOURCE = Path(__file__).with_name("dram_finite_stream_ramp.cu")
RUNNER = Path(__file__).resolve()
GRIDS = (56, 64, 80, 86, 96, 100, 108, 112, 128, 172, 224, 320, 448)
KIB_PER_CTA = (
    256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584,
    4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336, 16384,
)
# Bytes in flight per CTA = threads * batch * 16.  Batch 32 gives 64 KiB, the
# depth at which the stream is request-limited rather than issue-limited.
BATCH = 32
THREADS = 128
MAXIMUM_RELATIVE_SPREAD = 0.10
DRAM_READ_CAPACITY_BYTES_PER_S = 1124 * 1.593e9


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


def _device_state(device):
    result = _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,"
        "clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ])
    values = tuple(item.strip() for item in result.stdout.split(","))
    if len(values) != 7:
        raise ValueError("unexpected nvidia-smi device row")
    state = {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "memory_used_mib": int(values[3]),
        "gpu_utilization_pct": int(values[4]),
        "sm_clock_mhz": int(values[5]), "memory_clock_mhz": int(values[6]),
    }
    if state["name"] != DEVICE_NAME:
        raise RuntimeError(f"device {device} is not {DEVICE_NAME}")
    return state


def _require_idle(device, *, settle_seconds=0.0):
    deadline = time.monotonic() + settle_seconds
    while True:
        state = _device_state(device)
        if state["gpu_utilization_pct"] == 0 and state["memory_used_mib"] <= 64:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"device {device} is not idle: {state}")
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=9)
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 15:
        raise ValueError("trials must be within 3..15")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    before = _require_idle(arguments.device)
    environment = {
        "CUDA_VISIBLE_DEVICES": str(arguments.device),
        "PATH": "/usr/bin:/bin",
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    header = None
    lines = []
    sass_digests = {}
    for grid in GRIDS:
        binary = arguments.output_dir / f"ramp_grid{grid}"
        _capture([
            str(NVCC), "-O3", "-arch=sm_80", f"-DPROBE_GRID={grid}",
            f"-DPROBE_BATCH={BATCH}", "-o", str(binary), str(SOURCE)])
        sass_path = arguments.output_dir / f"ramp_grid{grid}.sass"
        sass_path.write_text(
            _capture([str(CUOBJDUMP), "-sass", str(binary)]).stdout,
            encoding="utf-8")
        sass_digests[binary.name] = _sha256(sass_path)
        result = _capture(
            [str(binary), "0", str(arguments.trials),
             ",".join(str(value) for value in KIB_PER_CTA)],
            environment=environment)
        output = result.stdout.splitlines()
        if header is None:
            header = output[0]
        elif output[0] != header:
            raise ValueError("probe changed its output schema mid-sweep")
        lines.extend(output[1:])
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = _require_idle(arguments.device, settle_seconds=30.0)

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
        "schema": "nvidia_a800_dram_finite_stream_ramp_candidate_v1",
        "hardware_id": "nvidia_a800",
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
        "sass_sha256": sass_digests,
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
