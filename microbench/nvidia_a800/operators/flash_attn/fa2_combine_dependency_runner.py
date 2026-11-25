#!/usr/bin/env python3
"""Capture isolated FA2 split-combine dependency primitives on an A800."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_combine_dependency.cu")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_a800")
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
MODES = {
    0: "timer_baseline",
    1: "fp32_max_allreduce",
    2: "fp32_sum_allreduce",
    3: "output_conversion_baseline",
    4: "fp32_to_bf16_output_conversion",
    5: "slse_shared_store_barrier_transpose_load",
    6: "post_max_expf_local_sum",
    7: "post_sum_logf",
    8: "scale_expf_shared_store_barrier",
    9: "lse_global_load_to_shared_transpose_ready",
}
GRIDS = (1, 32, 108)
ALLREDUCE_WIDTHS = (2, 4, 8, 16, 32)
CONVERSION_GEOMETRIES = ((4, 128), (8, 64))
# (block_m, head_dim, actual splits, source kMaxSplits).  These cover every
# split geometry presently emitted by the A800 D64/D128 FlashAttention set.
SOURCE_FRAGMENT_GEOMETRIES = (
    (8, 64, 2, 2),
    (8, 64, 4, 4),
    (8, 64, 8, 8),
    (4, 128, 2, 2),
    (4, 128, 3, 4),
    (4, 128, 4, 4),
    (4, 128, 5, 8),
    (4, 128, 6, 8),
    (4, 128, 7, 8),
    (4, 128, 8, 8),
    (4, 128, 9, 16),
    (4, 128, 11, 16),
    (4, 128, 16, 16),
    (4, 128, 22, 32),
)
REPEATS = 64
SAMPLES = 50
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


def _parse_binary_csv(text, *, trial, mode_name, before, after):
    reader = csv.DictReader(io.StringIO(text))
    expected = (
        "sample", "mode", "grid", "repeats", "allreduce_width",
        "num_splits", "max_splits", "block_m", "head_dim",
        "output_values_per_lane", "block", "sm_id", "cycles", "checksum",
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
            "allreduce_width": int(raw["allreduce_width"]),
            "num_splits": int(raw["num_splits"]),
            "max_splits": int(raw["max_splits"]),
            "block_m": int(raw["block_m"]),
            "head_dim": int(raw["head_dim"]),
            "output_values_per_lane": int(raw["output_values_per_lane"]),
            "block": int(raw["block"]),
            "sm_id": int(raw["sm_id"]),
            "cycles": int(raw["cycles"]),
            "checksum": int(raw["checksum"]),
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
            "memory_clock_before_mhz": before["memory_clock_mhz"],
            "memory_clock_after_mhz": after["memory_clock_mhz"],
        })
    if not rows:
        raise ValueError("benchmark emitted no rows")
    return rows


def _schedule():
    cells = []
    for width in ALLREDUCE_WIDTHS:
        for mode in (0, 1, 2):
            for grid in GRIDS:
                cells.append((mode, grid, width, width, 4, 128))
    for block_m, head_dim in CONVERSION_GEOMETRIES:
        for mode in (3, 4):
            for grid in GRIDS:
                cells.append((mode, grid, 0, 0, block_m, head_dim))
    for block_m, head_dim, num_splits, _ in SOURCE_FRAGMENT_GEOMETRIES:
        for mode in (0, 5, 6, 7, 8, 9):
            for grid in GRIDS:
                cells.append((
                    mode, grid, 0, num_splits, block_m, head_dim))
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=3)
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
    if arguments.device_id < 0:
        raise ValueError("device id must be non-negative")
    if not arguments.nvcc.is_file():
        raise FileNotFoundError(f"nvcc does not exist: {arguments.nvcc}")

    output_dir.mkdir(parents=True)
    binary_path = output_dir / "fa2_combine_dependency"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE_PATH), "-o", str(binary_path),
    ]
    _run(compile_command)
    device = _device_row(arguments.device_id, require_idle=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    schedule = _schedule()
    for trial in range(TRIALS):
        offset = trial * 7 % len(schedule)
        order = schedule[offset:] + schedule[:offset]
        for mode, grid, width, num_splits, block_m, head_dim in order:
            before = _device_row(arguments.device_id, require_idle=False)
            text = _run([
                str(binary_path), str(mode + 1), str(grid), str(REPEATS),
                str(SAMPLES), str(width), str(num_splits), str(block_m),
                str(head_dim),
            ], env=environment)
            after = _device_row(arguments.device_id, require_idle=False)
            if before["uuid"] != after["uuid"]:
                raise RuntimeError("GPU identity changed during a cell")
            rows.extend(_parse_binary_csv(
                text, trial=trial, mode_name=MODES[mode],
                before=before, after=after))

    raw_path = output_dir / "raw.csv"
    fieldnames = tuple(rows[0])
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "a800_fa2_combine_dependency_capture_v3",
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).strip(),
        "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "binary_sha256": _sha256(binary_path),
        "raw_sha256": _sha256(raw_path),
        "upstream_flash_attn_commit": (
            "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"),
        "upstream_combine_source_sha256": (
            "765dd3ef217bc9d79c9c0494ba52ea63767099be737c14604bec748d85f0dde3"),
        "upstream_allreduce_source_sha256": (
            "abc6d4c73522af35f6c31be3938830b9474f620d7c8083168d80dab226a0fdc9"),
        "nvcc_path": str(arguments.nvcc),
        "nvcc_version": _run([str(arguments.nvcc), "--version"]).strip(),
        "compile_command": compile_command,
        "device": device,
        "modes": MODES,
        "grids": list(GRIDS),
        "allreduce_widths": list(ALLREDUCE_WIDTHS),
        "conversion_geometries": [
            {"block_m": block_m, "head_dim": head_dim,
             "output_values_per_lane": 4}
            for block_m, head_dim in CONVERSION_GEOMETRIES
        ],
        "source_fragment_geometries": [
            {"block_m": block_m, "head_dim": head_dim,
             "num_splits": num_splits, "max_splits": max_splits}
            for block_m, head_dim, num_splits, max_splits
            in SOURCE_FRAGMENT_GEOMETRIES
        ],
        "repeats": REPEATS,
        "samples_per_cell": SAMPLES,
        "trials": TRIALS,
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": "clock64_same_cta_envelope_cycles",
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
