#!/usr/bin/env python3
"""Capture the H200 two-resident-CTA finite LDGSTS service.

Owner copy of microbench/nvidia_a800/memory/l1/fa2_resident_ldgsts_finite_runner.py.
Repeat set (1..64 panels), seven measured launches after the probe's ten
warm-up launches, and the compile / SASS snapshot are unchanged.  Hardware
coupling:

* Nsight Compute is not installed on the H200 host.  The A800 runner timed
  each launch with NCU ``gpu__time_duration.sum`` and converted it with the
  NCU-measured ``sm__cycles_elapsed.avg.per_second``.  Here each launch's
  GPU duration comes from the Nsight Systems ``cuda_gpu_trace`` report (the
  same per-kernel GPU start/end timeline) and is converted with the measured
  1.98-GHz SM clock, which the 50-ms clock gate proves for every accepted
  process (helpers from ../l2/staged_ldgsts_l2_runner.py);
* the NCU dynamic LDGSTS instruction/byte counter closure cannot be
  captured; the static SASS contract (32 LDGSTS.E.BYPASS.128, no spill) is
  still enforced by the report;
* H200 GPU 2 only, CUDA 12.9 sm_90 build, grid 264 (two resident CTAs on
  each of 132 SMs), output under profiler_output/microbench/nvidia_h200.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "l2"))
from staged_ldgsts_l2_runner import (  # noqa: E402
    _capture,
    _clock_summary,
    _compile,
    _device,
    _dump_sass,
    _gated_run,
    _new_output,
    _sha256,
    _wait_for_idle,
)


SOURCE_PATH = Path(__file__).with_name("fa2_resident_ldgsts_finite.cu")
REPORT_PATH = Path(__file__).with_name("fa2_resident_ldgsts_finite_report.py")
NSYS = Path(shutil.which("nsys") or "nsys")
NSYS_HOME = (
    Path(os.environ["MIDLAYER_H200_HOME"])
    if "MIDLAYER_H200_HOME" in os.environ else Path.home()
)
REPEATS = (1, 2, 4, 8, 16, 32, 64)
WARMUP_LAUNCHES = 10
SAMPLES = 7
GRID_CTAS = 264
KERNEL_NAME = "fa2_resident_ldgsts_finite"


def _kernel_durations(trace_csv):
    rows = list(csv.DictReader(Path(trace_csv).read_text(
        encoding="utf-8").splitlines()))
    durations = []
    for row in rows:
        if KERNEL_NAME not in row["Name"]:
            continue
        if int(row["GrdX"]) != GRID_CTAS or int(row["BlkX"]) != 128:
            raise ValueError(f"unexpected launch geometry: {row}")
        durations.append(int(row["Duration (ns)"]))
    if len(durations) != WARMUP_LAUNCHES + SAMPLES:
        raise ValueError(f"{trace_csv}: {len(durations)} probe launches")
    return durations[WARMUP_LAUNCHES:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args()
    output_dir = _new_output(arguments.output_dir)
    device = _device(arguments.device)
    binary_path = output_dir / "fa2_resident_ldgsts_finite"
    compile_command, ptxas = _compile(SOURCE_PATH, binary_path)
    ptxas_path = output_dir / "ptxas.txt"
    ptxas_path.write_text(ptxas, encoding="utf-8")
    sass_path = output_dir / "probe.sass"
    _dump_sass(binary_path, sass_path)
    shutil.copy2(SOURCE_PATH, output_dir / "source_snapshot.cu")

    NSYS_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(NSYS_HOME)
    cells = []
    for repeats in REPEATS:
        stem = output_dir / f"final_r{repeats}"
        command = [
            str(NSYS), "profile", "-t", "cuda", "-o", str(stem),
            "--force-overwrite", "true",
            str(binary_path), str(repeats), str(SAMPLES),
        ]
        result, attempts = _gated_run(
            command, device=arguments.device, log_stem=stem)
        (output_dir / f"final_r{repeats}.stdout.txt").write_text(
            result.stdout, encoding="utf-8")
        _capture([
            str(NSYS), "stats", "--report", "cuda_gpu_trace",
            "--format", "csv", "--output", str(stem), "--force-export", "true",
            f"{stem}.nsys-rep"])
        trace_csv = Path(f"{stem}_cuda_gpu_trace.csv")
        durations = _kernel_durations(trace_csv)
        (output_dir / f"final_r{repeats}.durations.json").write_text(
            json.dumps({"repeats": repeats, "duration_ns": durations}) + "\n",
            encoding="utf-8")
        cells.append({
            "repeats": repeats, "command": command, "attempts": attempts,
            "duration_ns": durations,
            "trace_csv_sha256": _sha256(trace_csv),
        })
    device_after = _wait_for_idle(arguments.device)
    manifest = {
        "schema": "h200_fa2_resident_ldgsts_finite_capture_v1",
        "hardware_id": "nvidia_h200",
        "owner": "h200-port-wp3_ldgsts",
        "device": device,
        "device_after": device_after,
        "source_commit": arguments.source_commit,
        "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        "pid": os.getpid(),
        "compile_command": compile_command,
        "cells": cells,
        "clock_gate": _clock_summary([cell["attempts"] for cell in cells]),
        "timer": "nsys 2025.6.1 cuda_gpu_trace kernel duration",
        "source_sha256": _sha256(SOURCE_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "reporter_sha256": _sha256(REPORT_PATH),
        "binary_sha256": _sha256(binary_path),
        "ptxas_sha256": _sha256(ptxas_path),
        "sass_sha256": _sha256(sass_path),
        "repeats": list(REPEATS),
        "samples_per_cell": SAMPLES,
        "threads_per_cta": 128,
        "grid_ctas": GRID_CTAS,
        "dynamic_shared_bytes": 77824,
        "panel_bytes_per_cta": 65536,
        "production_or_evaluation_duration_consumed": False,
        "cleanup": "complete",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
