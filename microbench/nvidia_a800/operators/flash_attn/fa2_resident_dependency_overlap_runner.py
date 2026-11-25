#!/usr/bin/env python3
"""Capture one- versus two-resident-CTA FA2 scalar dependency spans."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name(
    "fa2_resident_dependency_overlap.cu")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_a800")
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
OWNER = "kernel-accuracy-fa-resident-dependency"
MODES = {
    0: "timer_baseline",
    1: "fp32_max_quad4",
    3: "cta_barrier",
    4: "fp32_quad4_sum",
    5: "fp32_scale_exp2_sum",
    6: "integer_column_predicate",
    7: "fp32_to_bf16_conversion",
}
GRIDS = (108, 216)
GEOMETRIES = ((2, 32), (4, 16), (4, 32))
REPEATS = 64
SAMPLES = 100
TRIALS = 3


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
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
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
        "index": int(fields[0]),
        "name": fields[1],
        "uuid": fields[2],
        "pstate": fields[3],
        "utilization_gpu_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]),
        "sm_clock_mhz": int(fields[6]),
        "memory_clock_mhz": int(fields[7]),
        "driver_version": fields[8],
    }
    if row["index"] != device_id or row["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    if (require_idle
            and (row["utilization_gpu_pct"] != 0
                 or row["memory_used_mib"] != 0)):
        raise RuntimeError(f"target GPU is not idle: {row}")
    applications = _run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    for line in applications.splitlines():
        if line.strip() and line.split(",", 1)[0].strip() == row["uuid"]:
            raise RuntimeError(
                f"target GPU has a compute process: {line.strip()}")
    return row


def _parse_csv(text, *, trial, mode_name, before, after):
    reader = csv.DictReader(io.StringIO(text))
    expected = (
        "sample", "mode", "grid", "repeats",
        "lane_local_rows_per_thread", "score_values_per_lane_local_row",
        "resident_ctas_per_sm", "dynamic_shared_bytes", "duration_ns",
        "median_cta_cycles",
    )
    if tuple(reader.fieldnames or ()) != expected:
        raise ValueError("benchmark CSV fields changed")
    rows = []
    for raw in reader:
        rows.append({
            "trial": trial,
            "mode_name": mode_name,
            "sample": int(raw["sample"]),
            "mode": int(raw["mode"]),
            "grid": int(raw["grid"]),
            "repeats": int(raw["repeats"]),
            "lane_local_rows_per_thread": int(
                raw["lane_local_rows_per_thread"]),
            "score_values_per_lane_local_row": int(
                raw["score_values_per_lane_local_row"]),
            "resident_ctas_per_sm": int(raw["resident_ctas_per_sm"]),
            "dynamic_shared_bytes": int(raw["dynamic_shared_bytes"]),
            "duration_ns": float(raw["duration_ns"]),
            "median_cta_cycles": int(raw["median_cta_cycles"]),
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
            "memory_clock_before_mhz": before["memory_clock_mhz"],
            "memory_clock_after_mhz": after["memory_clock_mhz"],
        })
    if len(rows) != SAMPLES:
        raise ValueError("benchmark emitted the wrong sample count")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--device-id", type=int, choices=tuple(range(8)), default=1)
    parser.add_argument(
        "--nvcc", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if output_dir == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in (
            output_dir.parents):
        raise ValueError("output directory must be below the A800 microbench root")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not arguments.nvcc.is_file():
        raise FileNotFoundError(f"nvcc does not exist: {arguments.nvcc}")

    device = _device_row(arguments.device_id, require_idle=True)
    output_dir.mkdir(parents=True)
    binary_path = output_dir / "fa2_resident_dependency_overlap"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE_PATH), "-o", str(binary_path),
    ]
    _run(compile_command)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    conditioning_command = [
        str(binary_path), str(5 + 1), str(216), str(REPEATS),
        str(10), str(4), str(32),
    ]
    _run(conditioning_command, env=environment)
    schedule = [
        (mode, grid, lane_rows, score_values)
        for lane_rows, score_values in GEOMETRIES
        for mode in MODES
        for grid in GRIDS
    ]
    rows = []
    commands = []
    for trial in range(TRIALS):
        offset = trial * 5 % len(schedule)
        for mode, grid, lane_rows, score_values in (
                schedule[offset:] + schedule[:offset]):
            command = [
                str(binary_path), str(mode + 1), str(grid), str(REPEATS),
                str(SAMPLES), str(lane_rows), str(score_values),
            ]
            before = _device_row(arguments.device_id, require_idle=False)
            output = _run(command, env=environment)
            after = _device_row(arguments.device_id, require_idle=False)
            if before["uuid"] != after["uuid"]:
                raise RuntimeError("GPU identity changed during capture")
            if (before["sm_clock_mhz"] != after["sm_clock_mhz"]
                    or before["memory_clock_mhz"]
                    != after["memory_clock_mhz"]):
                raise RuntimeError(
                    "GPU clocks changed during a captured cell: "
                    f"before={before}, after={after}")
            commands.append(command)
            rows.extend(_parse_csv(
                output, trial=trial, mode_name=MODES[mode],
                before=before, after=after))

    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "a800_fa2_resident_dependency_overlap_capture_v1",
        "hardware_id": "nvidia_a800",
        "owner": OWNER,
        "device": device,
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).strip(),
        "command": [str(Path(__file__).resolve()), *os.sys.argv[1:]],
        "pid": os.getpid(),
        "tmux": None,
        "compile_command": compile_command,
        "conditioning_command": conditioning_command,
        "cell_commands": commands,
        "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "binary_sha256": _sha256(binary_path),
        "raw_sha256": _sha256(raw_path),
        "nvcc_path": str(arguments.nvcc),
        "nvcc_version": _run([str(arguments.nvcc), "--version"]).strip(),
        "modes": MODES,
        "grids": list(GRIDS),
        "geometries": [list(value) for value in GEOMETRIES],
        "repeats": REPEATS,
        "samples_per_cell": SAMPLES,
        "trials": TRIALS,
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": "cuda_event_grid_span_and_clock64_cta_span",
        "cleanup": "complete",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "manifest": str(manifest_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
