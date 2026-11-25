#!/usr/bin/env python3
"""Capture an isolated one-CTA/SM 256-thread CTA-barrier service on H200.

H200 owner copy of microbench/nvidia_a800/runtime/synchronization/
cta_barrier_service_runner.py.  Trials, modes, repeats, samples, the per-trial
median(barrier) - median(baseline) / repeats rule, the median-over-trials
publication and the report keys are the A800 ones.  Hardware coupling changed:
  - GRID 108 -> 132 and the expected device name NVIDIA H200;
  - runs on the H200 host with the system python3: the binary and its SASS
    dump are built off-host (CUDA 12.2 nvcc -O3 -std=c++17 -arch=sm_90
    -cudart static; cuobjdump --dump-sass) and passed in with the source /
    runner SHA256 and baseline commit, instead of compiling in-process;
  - the idle check accepts the H200 idle framebuffer reservation (<= 16 MiB
    used, no compute process) instead of exactly 0 MiB;
  - SM clock/utilization are sampled every 50 ms during every trial x mode
    process; the clock gate rejects a process with no busy sample or a busy
    sample below 1980 MHz (the clock cannot be locked in this container);
  - output root is the run directory given on the command line.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


SOURCE_PATH = "microbench/nvidia_h200/runtime/synchronization/cta_barrier_service.cu"
EXPECTED_DEVICE_NAME = "NVIDIA H200"
BOOST_MHZ = 1980
GRID = 132
REPEATS = 1024
SAMPLES = 32
TRIALS = 3
THREADS_PER_CTA = 256
TOTAL_SHARED_BYTES = 147456
STATIC_SHARED_BYTES = 2 * THREADS_PER_CTA * 8
DYNAMIC_SHARED_BYTES = TOTAL_SHARED_BYTES - STATIC_SHARED_BYTES


def _run(command, *, env=None):
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, env=env)
    if completed.stderr.strip():
        raise RuntimeError(
            f"command wrote stderr: {command!r}: {completed.stderr.strip()}")
    return completed.stdout


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_row(device_id, *, require_idle):
    output = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,pstate,utilization.gpu,memory.used,"
        "clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ]).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 9:
        raise ValueError(f"unexpected nvidia-smi device row: {output!r}")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "pstate": fields[3], "utilization_gpu_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]), "sm_clock_mhz": int(fields[6]),
        "memory_clock_mhz": int(fields[7]), "driver_version": fields[8],
    }
    if row["index"] != device_id or row["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    if (require_idle and (row["utilization_gpu_pct"] != 0
                          or row["memory_used_mib"] > 16)):
        raise RuntimeError(f"target GPU is not idle: {row}")
    applications = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    for line in applications.splitlines():
        if line.strip() and line.split(",", 1)[0].strip() == row["uuid"]:
            raise RuntimeError(f"target GPU has a compute process: {line.strip()}")
    return row


def _parse_rows(text, *, trial, mode_name, before, after):
    reader = csv.DictReader(io.StringIO(text))
    expected = (
        "sample", "mode", "grid", "repeats", "block",
        "dynamic_shared_bytes", "sm_id", "cycles")
    if tuple(reader.fieldnames or ()) != expected:
        raise ValueError("benchmark CSV fields changed")
    rows = []
    for raw in reader:
        if int(raw["grid"]) != GRID or int(raw["repeats"]) != REPEATS:
            raise ValueError("benchmark launch geometry changed")
        if int(raw["dynamic_shared_bytes"]) != DYNAMIC_SHARED_BYTES:
            raise ValueError("benchmark shared-memory contract changed")
        rows.append({
            "trial": trial, "mode_name": mode_name,
            **{field: int(raw[field]) for field in expected},
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
            "memory_clock_before_mhz": before["memory_clock_mhz"],
            "memory_clock_after_mhz": after["memory_clock_mhz"],
        })
    if len(rows) != SAMPLES * GRID:
        raise ValueError("benchmark did not emit one row per CTA envelope")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--baseline-commit", required=True)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if arguments.device_id < 0:
        raise ValueError("device id must be non-negative")
    device = _device_row(arguments.device_id, require_idle=True)
    output_dir.mkdir(parents=True)
    binary = arguments.binary.resolve()
    compile_command = ["/usr/local/cuda-12.2/bin/nvcc", "-O3", "-std=c++17",
                       "-arch=sm_90", "-cudart", "static", SOURCE_PATH,
                       "-o", "cta_barrier_service"]
    sass_path = arguments.sass.resolve()
    sass = sass_path.read_text(encoding="utf-8")
    if "BAR.SYNC" not in sass:
        raise ValueError("barrier microbenchmark SASS lacks BAR.SYNC")
    if "HMMA" in sass or "IMMA" in sass:
        raise ValueError("barrier microbenchmark SASS contains matrix work")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    clock_gate = []
    for trial in range(TRIALS):
        for mode, mode_name in ((1, "timer_baseline"), (2, "cta_barrier")):
            before = _device_row(arguments.device_id, require_idle=False)
            clock_path = output_dir / f"trial{trial}_{mode_name}.clock.csv"
            with clock_path.open("w") as clock_file:
                sampler = subprocess.Popen(
                    ["nvidia-smi", "-i", str(arguments.device_id),
                     "--query-gpu=clocks.sm,utilization.gpu",
                     "--format=csv,noheader,nounits", "-lms", "50"],
                    stdout=clock_file)
                try:
                    text = _run([str(binary), str(mode), str(GRID), str(REPEATS),
                                 str(SAMPLES)], env=environment)
                finally:
                    time.sleep(0.2)
                    sampler.terminate()
                    sampler.wait()
            busy = [int(line.split(",")[0])
                    for line in clock_path.read_text().splitlines()
                    if line.strip() and int(line.split(",")[1]) > 0]
            clock_gate.append({
                "trial": trial, "mode_name": mode_name, "busy_samples": len(busy),
                "busy_clock_mhz_min": min(busy) if busy else None,
                "busy_clock_mhz_max": max(busy) if busy else None,
                "accepted": bool(busy) and min(busy) >= BOOST_MHZ})
            rows.extend(_parse_rows(
                text, trial=trial, mode_name=mode_name, before=before,
                after=_device_row(arguments.device_id, require_idle=False)))
    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "nvidia_h200_cta_barrier_service_capture_v1",
        "baseline_commit": arguments.baseline_commit,
        "owner": "h200-port-wp7_launch",
        "source_path": SOURCE_PATH,
        "source_sha256": arguments.source_sha256,
        "runner_sha256": arguments.runner_sha256,
        "clock_gate": clock_gate,
        "host_loadavg_after": Path("/proc/loadavg").read_text().split()[:3],
        "binary_sha256": _sha256(binary), "sass_sha256": _sha256(sass_path),
        "raw_sha256": _sha256(raw_path), "compile_command": compile_command,
        "device": device, "grid": GRID, "repeats": REPEATS,
        "samples_per_trial": SAMPLES, "trials": TRIALS,
        "threads_per_cta": THREADS_PER_CTA,
        "total_shared_bytes": TOTAL_SHARED_BYTES,
        "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "production_or_evaluation_duration_consumed": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    samples_by_trial_mode = {}
    for trial in range(TRIALS):
        for mode_name in ("timer_baseline", "cta_barrier"):
            selected = [
                row for row in rows
                if row["trial"] == trial and row["mode_name"] == mode_name]
            sm_ids = {row["sm_id"] for row in selected}
            if len(selected) != SAMPLES * GRID or len(sm_ids) != GRID:
                raise ValueError("CTA residency envelope did not cover all SMs")
            samples_by_trial_mode[(trial, mode_name)] = statistics.median(
                row["cycles"] for row in selected)
    trial_cycles = []
    for trial in range(TRIALS):
        baseline = samples_by_trial_mode[(trial, "timer_baseline")]
        barrier = samples_by_trial_mode[(trial, "cta_barrier")]
        trial_cycles.append((barrier - baseline) / REPEATS)
    cycle_spread = (max(trial_cycles) - min(trial_cycles)) / min(trial_cycles)
    report = {
        "schema": "nvidia_h200_cta_barrier_service_report_v1",
        "hardware_id": "nvidia_h200",
        "mechanism": "isolated_256_thread_single_cta_sm_barrier",
        "production_or_evaluation_duration_consumed": False,
        "capture_manifest_sha256": _sha256(manifest_path),
        "raw_sha256": _sha256(raw_path),
        "source_sha256": arguments.source_sha256,
        "runner_sha256": arguments.runner_sha256,
        "sass_sha256": _sha256(sass_path),
        "clock_gate": clock_gate,
        "threads_per_cta": THREADS_PER_CTA,
        "total_shared_bytes": TOTAL_SHARED_BYTES,
        "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
        "resident_ctas_per_sm": 1,
        "grid": GRID,
        "repeats": REPEATS,
        "samples_per_trial": SAMPLES,
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "trial_barrier_cycles": trial_cycles,
        "barrier_cycles": statistics.median(trial_cycles),
        "trial_relative_spread": cycle_spread,
        "gates": {
            "all_trials_cover_each_sm_once_per_sample": True,
            "clock_stable": all(
                row["sm_clock_before_mhz"] == row["sm_clock_after_mhz"]
                for row in rows),
            "busy_clock_at_least_1980_mhz": all(
                entry["accepted"] for entry in clock_gate),
            "relative_spread_at_most_0_01": cycle_spread <= 0.01,
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path),
                      "report": str(report_path), "row_count": len(rows)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
