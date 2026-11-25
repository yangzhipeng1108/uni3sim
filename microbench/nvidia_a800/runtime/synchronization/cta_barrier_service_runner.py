#!/usr/bin/env python3
"""Capture an isolated one-CTA/SM 256-thread CTA-barrier service on A800."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("cta_barrier_service.cu")
OUTPUT_ROOT = (REPOSITORY_ROOT / "profiler_output" / "microbench"
               / "nvidia_a800")
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
GRID = 108
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
                          or row["memory_used_mib"] != 0)):
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
    parser.add_argument("--nvcc", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument("--cuobjdump", type=Path,
                        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if OUTPUT_ROOT.resolve() not in output_dir.parents:
        raise ValueError("output directory must be below the A800 microbench root")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if (arguments.device_id < 0 or not arguments.nvcc.is_file()
            or not arguments.cuobjdump.is_file()):
        raise ValueError(
            "device id must be non-negative and CUDA tools must exist")
    device = _device_row(arguments.device_id, require_idle=True)
    output_dir.mkdir(parents=True)
    binary = output_dir / "cta_barrier_service"
    compile_command = [str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
                       str(SOURCE_PATH), "-o", str(binary)]
    _run(compile_command)
    sass_path = output_dir / "sass.txt"
    sass_path.write_text(
        _run([str(arguments.cuobjdump), "--dump-sass", str(binary)]),
        encoding="utf-8")
    sass = sass_path.read_text(encoding="utf-8")
    if "BAR.SYNC" not in sass:
        raise ValueError("barrier microbenchmark SASS lacks BAR.SYNC")
    if "HMMA" in sass or "IMMA" in sass:
        raise ValueError("barrier microbenchmark SASS contains matrix work")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    for trial in range(TRIALS):
        for mode, mode_name in ((1, "timer_baseline"), (2, "cta_barrier")):
            before = _device_row(arguments.device_id, require_idle=False)
            rows.extend(_parse_rows(
                _run([str(binary), str(mode), str(GRID), str(REPEATS),
                      str(SAMPLES)], env=environment),
                trial=trial, mode_name=mode_name, before=before,
                after=_device_row(arguments.device_id, require_idle=False)))
    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "nvidia_a800_cta_barrier_service_capture_v1",
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).strip(),
        "owner": "capture-owner-3",
        "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(Path(__file__)),
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
        "schema": "nvidia_a800_cta_barrier_service_report_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "isolated_256_thread_single_cta_sm_barrier",
        "production_or_evaluation_duration_consumed": False,
        "capture_manifest_sha256": _sha256(manifest_path),
        "raw_sha256": _sha256(raw_path),
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "sass_sha256": _sha256(sass_path),
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
