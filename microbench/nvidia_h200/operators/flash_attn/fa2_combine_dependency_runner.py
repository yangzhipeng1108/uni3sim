#!/usr/bin/env python3
"""Capture isolated FA2 split-combine dependency primitives on an H200.

H200 owner copy of microbench/nvidia_a800/operators/flash_attn/
fa2_combine_dependency_runner.py.  Modes, allreduce widths, conversion
geometries, repeats, samples, trials, rotation and CSV schema are unchanged.
Hardware-coupling changes only:
  - device contract "NVIDIA A800-SXM4-80GB" -> "NVIDIA H200", -arch=sm_80 ->
    -arch=sm_90 (same CUDA 12.2 nvcc, plus -cudart static);
  - grids 1/32/108 -> 1/32/132 (full-device grid = SM count);
  - output root profiler_output/microbench/nvidia_h200;
  - idle contract: memory.used <= 8 MiB (H200 driver reports 4 MiB idle);
  - ``build`` (local) / ``capture`` (remote) stages, explicit baseline commit;
  - 50 ms ClockGate (unlocked SM clock) with bounded re-runs of rejected
    cells;
  - ``--fragment-set``: ``a800_source`` is the unchanged A800 split table;
    ``h200_heuristic_extension`` adds every other split count that the
    FA2 num_splits_heuristic can choose with 132 SMs (launch slots 264,
    max_splits 128) and that this probe can represent without changing its
    kernel: D128/kBlockM=4 splits 2..32 and D64/kBlockM=8 splits 2..16
    (the probe's static MaxSplits <= 32 and its mode-9 LSE buffer of
    grid*128 floats bound the representable set).
"""

import argparse
import atexit
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_combine_dependency.cu")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_h200")
EXPECTED_DEVICE_NAME = "NVIDIA H200"
IDLE_MEMORY_MIB = 8
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
GRIDS = (1, 32, 132)
ALLREDUCE_WIDTHS = (2, 4, 8, 16, 32)
CONVERSION_GEOMETRIES = ((4, 128), (8, 64))
# (block_m, head_dim, actual splits, source kMaxSplits).  These cover every
# split geometry emitted by the A800 D64/D128 FlashAttention set.
A800_SOURCE_FRAGMENT_GEOMETRIES = (
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


def _ceil_power_of_two(value):
    result = 2
    while result < value:
        result *= 2
    return result


H200_HEURISTIC_EXTENSION_GEOMETRIES = tuple(
    (block_m, head_dim, splits, _ceil_power_of_two(splits))
    for block_m, head_dim, maximum in ((8, 64, 16), (4, 128, 32))
    for splits in range(2, maximum + 1)
    if (block_m, head_dim, splits, _ceil_power_of_two(splits))
    not in A800_SOURCE_FRAGMENT_GEOMETRIES)
FRAGMENT_SETS = {
    "a800_source": A800_SOURCE_FRAGMENT_GEOMETRIES,
    "h200_heuristic_extension": H200_HEURISTIC_EXTENSION_GEOMETRIES,
}
REPEATS = 64
SAMPLES = 50
TRIALS = 3
BINARY_NAME = "fa2_combine_dependency"

# --- H200 clock gate (hardware coupling: the H200 SM clock cannot be locked
# in this container).  A 50 ms nvidia-smi logger runs for the whole capture;
# every cell's wall-clock window is judged afterwards.  A cell is rejected
# when any busy sample (utilization > 0) inside its window, or any sample
# while its CUDA context is alive, reports an SM clock below the 1980 MHz
# boost clock; rejected cells are re-run (bounded passes) and counted.
CLOCK_TARGET_MHZ = 1980
CLOCK_MAX_PASSES = 4


class ClockGate:
    def __init__(self, device_id, log_path):
        self.log_path = Path(log_path)
        # nvidia-smi -f block-buffers its file; line-buffered stdout keeps
        # the log current so cells can be judged during the capture.
        self._stream = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen([
            "stdbuf", "-oL", "nvidia-smi", "-i", str(device_id),
            "--query-gpu=timestamp,clocks.sm,utilization.gpu",
            "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=self._stream, stderr=subprocess.DEVNULL)
        self.pid = self.process.pid
        # Stop the logger on every interpreter exit, including failures.
        atexit.register(self.stop)
        deadline = time.time() + 10.0
        while not self._samples_after(time.time() - 0.5):
            if time.time() > deadline:
                raise RuntimeError("clock logger produced no samples")
            time.sleep(0.1)

    def _samples(self):
        samples = []
        with self.log_path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3 or not fields[1].isdigit():
                    continue
                stamp = datetime.datetime.strptime(
                    fields[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                samples.append((stamp, int(fields[1]), int(fields[2])))
        return samples

    def _samples_after(self, stamp):
        if not self.log_path.exists():
            return []
        return [sample for sample in self._samples() if sample[0] > stamp]

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=30)
        self._stream.close()

    def judge(self, windows):
        """windows: {key: (start_s, end_s)} -> ({key: verdict}, samples)."""
        deadline = time.time() + 10.0
        last_end = max(end for _, end in windows.values())
        while not self._samples_after(last_end + 0.3):
            if time.time() > deadline:
                raise RuntimeError("clock logger stopped advancing")
            time.sleep(0.1)
        samples = self._samples()
        verdicts = {}
        for key, (start, end) in windows.items():
            inside = [s for s in samples if start <= s[0] <= end]
            busy = [s for s in samples
                    if start <= s[0] <= end + 0.25 and s[2] > 0]
            context = [s for s in inside if s[1] >= CLOCK_TARGET_MHZ]
            # Samples before the context raises the clock (process start-up)
            # are not busy; after the first boosted sample every sample in
            # the window belongs to the live context.
            live = ([s for s in inside if s[0] >= context[0][0]]
                    if context else [])
            low = [s for s in busy + live if s[1] < CLOCK_TARGET_MHZ]
            verdicts[key] = {
                "accepted": bool(context) and not low,
                "samples_in_window": len(inside),
                "busy_samples": len(busy),
                "min_live_or_busy_clock_mhz": (
                    min(s[1] for s in busy + live) if busy + live else None),
            }
        return verdicts


def _clock_summary(gate, windows):
    samples = gate._samples()
    live = [s[1] for s in samples if s[2] > 0]
    return {
        "logger": "nvidia-smi -lms 50 timestamp,clocks.sm,utilization.gpu",
        "logger_pid": gate.pid,
        "log_path": str(gate.log_path.name),
        "target_sm_clock_mhz": CLOCK_TARGET_MHZ,
        "total_samples": len(samples),
        "busy_samples": len(live),
        "busy_sm_clock_mhz_min": min(live) if live else None,
        "busy_sm_clock_mhz_max": max(live) if live else None,
        "busy_sm_clock_mhz_mean": (
            sum(live) / len(live) if live else None),
    }
# --- end H200 clock gate

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
                 or row["memory_used_mib"] > IDLE_MEMORY_MIB)):
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


def _schedule(fragment_geometries):
    cells = []
    for width in ALLREDUCE_WIDTHS:
        for mode in (0, 1, 2):
            for grid in GRIDS:
                cells.append((mode, grid, width, width, 4, 128))
    for block_m, head_dim in CONVERSION_GEOMETRIES:
        for mode in (3, 4):
            for grid in GRIDS:
                cells.append((mode, grid, 0, 0, block_m, head_dim))
    for block_m, head_dim, num_splits, _ in fragment_geometries:
        for mode in (0, 5, 6, 7, 8, 9):
            for grid in GRIDS:
                cells.append((
                    mode, grid, 0, num_splits, block_m, head_dim))
    return cells


def _validated_run_dir(path, *, must_exist):
    run_dir = path.resolve()
    if run_dir == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in (
            run_dir.parents):
        raise ValueError("run directory must be below the H200 microbench root")
    if must_exist != run_dir.exists():
        raise FileExistsError(
            f"run directory existence must be {must_exist}: {run_dir}")
    return run_dir


def build(arguments):
    run_dir = _validated_run_dir(arguments.output_dir, must_exist=False)
    if not arguments.nvcc.is_file():
        raise FileNotFoundError(f"nvcc does not exist: {arguments.nvcc}")
    (run_dir / "build").mkdir(parents=True)
    binary_path = run_dir / "build" / BINARY_NAME
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_90",
        "-cudart", "static", str(SOURCE_PATH), "-o", str(binary_path),
    ]
    _run(compile_command)
    (run_dir / "build" / "build_manifest.json").write_text(json.dumps({
        "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "binary_sha256": _sha256(binary_path),
        "nvcc_path": str(arguments.nvcc),
        "nvcc_version": _run([str(arguments.nvcc), "--version"]).strip(),
        "compile_command": compile_command,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture(arguments):
    run_dir = _validated_run_dir(arguments.output_dir, must_exist=True)
    build_manifest = json.loads(
        (run_dir / "build" / "build_manifest.json").read_text(
            encoding="utf-8"))
    binary_path = run_dir / "build" / BINARY_NAME
    if build_manifest["source_sha256"] != _sha256(SOURCE_PATH):
        raise ValueError("build source differs from current source")
    if build_manifest["binary_sha256"] != _sha256(binary_path):
        raise ValueError("binary differs from its build manifest")
    if (run_dir / "raw.csv").exists():
        raise FileExistsError("capture already exists")
    device = _device_row(arguments.device_id, require_idle=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    fragment_geometries = FRAGMENT_SETS[arguments.fragment_set]
    schedule = _schedule(fragment_geometries)
    ordered_keys = []
    for trial in range(TRIALS):
        offset = trial * 7 % len(schedule)
        order = schedule[offset:] + schedule[:offset]
        ordered_keys.extend((trial, *cell) for cell in order)
    gate = ClockGate(arguments.device_id, run_dir / "clock_log.csv")
    results = {}
    passes = []
    pending = list(ordered_keys)
    all_windows = {}
    for pass_index in range(CLOCK_MAX_PASSES):
        windows = {}
        for key in pending:
            trial, mode, grid, width, num_splits, block_m, head_dim = key
            start = time.time()
            before = _device_row(arguments.device_id, require_idle=False)
            text = _run([
                str(binary_path), str(mode + 1), str(grid), str(REPEATS),
                str(SAMPLES), str(width), str(num_splits), str(block_m),
                str(head_dim),
            ], env=environment)
            after = _device_row(arguments.device_id, require_idle=False)
            windows[key] = (start, time.time())
            if before["uuid"] != after["uuid"]:
                raise RuntimeError("GPU identity changed during a cell")
            results[key] = _parse_binary_csv(
                text, trial=trial, mode_name=MODES[mode],
                before=before, after=after)
        verdicts = gate.judge(windows)
        all_windows.update(windows)
        rejected = [key for key in pending if not verdicts[key]["accepted"]]
        passes.append({
            "pass": pass_index, "cells": len(pending),
            "accepted": len(pending) - len(rejected),
            "rejected": len(rejected),
            "rejected_cells": [
                {"key": list(key), **verdicts[key]} for key in rejected],
        })
        pending = rejected
        if not pending:
            break
    clock = _clock_summary(gate, all_windows)
    gate.stop()
    if pending:
        raise RuntimeError(f"{len(pending)} cells never passed the clock gate")
    rows = [row for key in ordered_keys for row in results[key]]
    raw_path = run_dir / "raw.csv"
    fieldnames = tuple(rows[0])
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "h200_fa2_combine_dependency_capture_v3",
        "fragment_set": arguments.fragment_set,
        "baseline_commit": arguments.baseline_commit,
        "source_path": build_manifest["source_path"],
        "source_sha256": build_manifest["source_sha256"],
        "binary_sha256": build_manifest["binary_sha256"],
        "raw_sha256": _sha256(raw_path),
        "upstream_flash_attn_commit": (
            "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"),
        "upstream_combine_source_sha256": (
            "765dd3ef217bc9d79c9c0494ba52ea63767099be737c14604bec748d85f0dde3"),
        "upstream_allreduce_source_sha256": (
            "abc6d4c73522af35f6c31be3938830b9474f620d7c8083168d80dab226a0fdc9"),
        "nvcc_path": build_manifest["nvcc_path"],
        "nvcc_version": build_manifest["nvcc_version"],
        "compile_command": build_manifest["compile_command"],
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
            in fragment_geometries
        ],
        "repeats": REPEATS,
        "samples_per_cell": SAMPLES,
        "trials": TRIALS,
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": "clock64_same_cta_envelope_cycles",
        "clock_gate": {**clock, "passes": passes,
                       "accepted_cells": len(ordered_keys),
                       "rejected_cell_runs": sum(
                           item["rejected"] for item in passes)},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(run_dir),
        "row_count": len(rows),
        "manifest": str(manifest_path),
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "capture"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--baseline-commit")
    parser.add_argument(
        "--fragment-set", choices=tuple(FRAGMENT_SETS),
        default="a800_source")
    parser.add_argument(
        "--nvcc", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    arguments = parser.parse_args()
    if arguments.stage == "build":
        build(arguments)
    else:
        if arguments.baseline_commit is None:
            raise ValueError("capture requires --baseline-commit")
        if arguments.device_id < 0:
            raise ValueError("device id must be non-negative")
        capture(arguments)


if __name__ == "__main__":
    main()
