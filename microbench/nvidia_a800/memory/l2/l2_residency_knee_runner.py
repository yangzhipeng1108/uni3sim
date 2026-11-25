#!/usr/bin/env python3
"""Capture the A800 L2 operand-residency curve for a repeated 2R1W stream.

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
import time
from pathlib import Path

NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
SOURCE = Path(__file__).with_name("l2_residency_knee.cu")
RUNNER = Path(__file__).resolve()
FOOTPRINTS_MIB = (
    6, 12, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 54, 60, 72, 96,
    192, 384,
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


def _device_state(device):
    result = _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,"
        "utilization.memory,clocks.current.sm,clocks.current.memory",
        "--format=csv,noheader,nounits",
    ])
    values = tuple(item.strip() for item in result.stdout.split(","))
    if len(values) != 8:
        raise ValueError("unexpected nvidia-smi device row")
    state = {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "memory_used_mib": int(values[3]),
        "gpu_utilization_pct": int(values[4]),
        "memory_utilization_pct": int(values[5]),
        "sm_clock_mhz": int(values[6]), "memory_clock_mhz": int(values[7]),
    }
    if state["name"] != DEVICE_NAME:
        raise RuntimeError(f"device {device} is not {DEVICE_NAME}")
    return state


def _require_idle(device, *, settle_seconds=0.0):
    """Return the device state once it is idle, optionally waiting first.

    nvidia-smi keeps reporting the last sampled utilization for a short window
    after a probe exits, so the post-run check settles before it asserts.
    """

    deadline = time.monotonic() + settle_seconds
    while True:
        state = _device_state(device)
        if state["gpu_utilization_pct"] == 0 and state["memory_used_mib"] <= 64:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"device {device} is not idle: {state}")
        time.sleep(0.5)


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
        if spread > MAXIMUM_RELATIVE_SPREAD:
            raise ValueError(
                f"unstable footprint {footprint} MiB: spread={spread}")
        logical_bytes = int(cell[0]["timed_logical_bytes"])
        curve.append({
            "total_footprint_mib": footprint,
            "in_kernel_passes": int(cell[0]["in_kernel_passes"]),
            "timed_logical_bytes": logical_bytes,
            "median_duration_ns": median_ns,
            "relative_spread": spread,
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
        str(NVCC), "-O3", "-arch=sm_80", "-o", str(binary), str(SOURCE)])
    sass = _capture([str(CUOBJDUMP), "-sass", str(binary)]).stdout
    sass_path = arguments.output_dir / "l2_residency_knee.sass"
    sass_path.write_text(sass, encoding="utf-8")

    before = _require_idle(arguments.device)
    environment = {
        "CUDA_VISIBLE_DEVICES": str(arguments.device),
        "PATH": "/usr/bin:/bin",
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result = _capture(
        [str(binary), str(arguments.trials),
         ",".join(str(value) for value in FOOTPRINTS_MIB)],
        environment=environment)
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    after = _require_idle(arguments.device, settle_seconds=30.0)

    measurements_path = arguments.output_dir / "measurements.csv"
    measurements_path.write_text(result.stdout, encoding="utf-8")
    rows = list(csv.DictReader(result.stdout.splitlines()))
    curve = _rate_curve(rows)

    candidate = {
        "schema": "nvidia_a800_l2_residency_knee_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "l2_operand_residency_curve",
        "production_or_evaluation_duration_consumed": False,
        "probe_contract": {
            "stream": "two_read_one_write_vec4",
            "read_instruction": "ld.global.cg.v4.u32",
            "write_policy": "default_write_back",
            "threads_per_cta": 128,
            "resident_ctas": 108 * 8,
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
        "sass_sha256": _sha256(sass_path),
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
