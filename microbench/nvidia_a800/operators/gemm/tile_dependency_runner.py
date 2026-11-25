#!/usr/bin/env python3
"""Capture the independent A800 BF16 tile load-to-MMA dependency probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from statistics import median


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("tile_dependency.cu")
RUNNER_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800")
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
MAXIMUM_IDLE_MEMORY_MIB = 8
MINIMUM_TRIALS = 3
MAXIMUM_TRIALS = 11
GRIDS = (1, 64, 80, 108)
ROUNDS = (16, 64, 128, 224)
BLOCK_THREADS = 128
WARPS_PER_CTA = 4
TILE_SHAPE = "64x64x64"
PIPELINE_STAGES = 6
PANEL_BYTES = 2 * 64 * 64 * 2
DYNAMIC_SHARED_BYTES = 98304
STORED_BYTES_PER_CTA = 64 * 64 * 4
TENSOR_FMA_PER_CTA_ROUND = 64 * 64 * 64
MAXIMUM_TRIAL_SPREAD = 0.10
MAXIMUM_STEADY_SLOPE_SPREAD = 0.15
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
RAW_FIELDS = (
    "grid", "rounds", "trial", "active_sms", "median_cta_cycles",
    "maximum_cta_cycles", "loaded_bytes", "stored_bytes", "tensor_fma",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, capture_output: bool = True):
    return subprocess.run(
        command, cwd=cwd, check=True, text=True,
        capture_output=capture_output, timeout=600)


def _git_output(*arguments: str) -> str:
    return _run(
        ["git", *arguments], cwd=REPOSITORY_ROOT).stdout.strip()


def _device_rows() -> list[dict[str, object]]:
    query = "index,name,uuid,pci.bus_id,memory.used,utilization.gpu"
    result = _run([
        "nvidia-smi", f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("unexpected nvidia-smi device row")
        rows.append({
            "index": int(row[0]),
            "name": row[1],
            "uuid": row[2],
            "pci_bus_id": row[3],
            "memory_used_mib": int(row[4]),
            "utilization_percent": int(row[5]),
        })
    return rows


def _compute_process_rows() -> list[dict[str, object]]:
    result = _run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT)
    rows = []
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 3:
            raise ValueError("unexpected nvidia-smi compute-process row")
        rows.append({
            "uuid": row[0], "pid": int(row[1]),
            "memory_mib": int(row[2]),
        })
    return rows


def _require_idle_device(device: int) -> dict[str, object]:
    matching = [row for row in _device_rows() if row["index"] == device]
    if len(matching) != 1:
        raise ValueError(f"device {device} does not resolve exactly once")
    row = matching[0]
    if row["name"] != EXPECTED_DEVICE:
        raise ValueError(f"device {device} is not {EXPECTED_DEVICE}")
    if row["memory_used_mib"] > MAXIMUM_IDLE_MEMORY_MIB:
        raise RuntimeError(
            f"device {device} has {row['memory_used_mib']} MiB allocated")
    if row["utilization_percent"] != 0:
        raise RuntimeError(f"device {device} has nonzero utilization")
    processes = [
        process for process in _compute_process_rows()
        if process["uuid"] == row["uuid"]
    ]
    if processes:
        raise RuntimeError(
            f"device {device} has compute processes: {processes}")
    return row


def _wait_for_idle_device(device: int) -> dict[str, object]:
    last_error = None
    for _ in range(10):
        try:
            return _require_idle_device(device)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(
        f"device {device} did not return idle after probe: {last_error}")


def _parse_native_output(text: str, trials: int) -> dict[str, object]:
    meta: dict[str, str] = {}
    rows: list[dict[str, int]] = []
    reader = csv.reader(io.StringIO(text))
    header_seen = False
    for line_number, columns in enumerate(reader, start=1):
        if not columns:
            continue
        if columns[0] == "meta":
            if len(columns) != 3 or columns[1] in meta:
                raise ValueError(f"invalid meta row {line_number}")
            meta[columns[1]] = columns[2]
            continue
        if columns[0] == "raw" and tuple(columns[1:]) == RAW_FIELDS:
            if header_seen:
                raise ValueError("native output repeats raw header")
            header_seen = True
            continue
        if columns[0] != "raw" or len(columns) != len(RAW_FIELDS) + 1:
            raise ValueError(f"unexpected native row {line_number}")
        rows.append({
            field: int(value)
            for field, value in zip(RAW_FIELDS, columns[1:], strict=True)
        })
    expected_meta = {
        "schema": "nvidia_a800_bf16_staged_tile_dependency_native_v1",
        "device_index": "0",
        "device_name": EXPECTED_DEVICE,
        "compute_capability": "sm_80",
        "sm_count": str(EXPECTED_SM_COUNT),
        "block_threads": str(BLOCK_THREADS),
        "warps_per_cta": str(WARPS_PER_CTA),
        "tile_shape": TILE_SHAPE,
        "pipeline_stages": str(PIPELINE_STAGES),
        "panel_bytes": str(PANEL_BYTES),
        "dynamic_shared_bytes_per_cta": str(DYNAMIC_SHARED_BYTES),
        "resident_ctas_per_sm": "1",
        "dependency": "ldgsts_shared_hmma_loop_carried_accumulator",
        "timer_domain": "clock64_sm_local_cta_cycles",
        "production_or_evaluation_duration_consumed": "false",
    }
    if meta != expected_meta:
        raise ValueError(
            f"native meta differs: expected={expected_meta}, actual={meta}")
    expected_cells = {(grid, rounds) for grid in GRIDS for rounds in ROUNDS}
    if len(rows) != len(expected_cells) * trials:
        raise ValueError("native output has an unexpected row count")
    seen = set()
    for row in rows:
        key = (row["grid"], row["rounds"])
        trial_key = (*key, row["trial"])
        if key not in expected_cells or trial_key in seen:
            raise ValueError(f"invalid or repeated native cell {trial_key}")
        seen.add(trial_key)
        if row["trial"] not in range(trials):
            raise ValueError(f"invalid trial index in {trial_key}")
        expected_active = 1 if row["grid"] == 1 else row["grid"]
        if row["active_sms"] != expected_active:
            raise ValueError(f"cell {trial_key} does not use one CTA per SM")
        if min(row["median_cta_cycles"], row["maximum_cta_cycles"]) <= 0:
            raise ValueError(f"cell {trial_key} has nonpositive cycles")
        if row["maximum_cta_cycles"] < row["median_cta_cycles"]:
            raise ValueError(f"cell {trial_key} has inverted cycle order")
        expected_loaded = row["grid"] * row["rounds"] * PANEL_BYTES
        expected_stored = row["grid"] * STORED_BYTES_PER_CTA
        expected_fma = (
            row["grid"] * row["rounds"] * TENSOR_FMA_PER_CTA_ROUND)
        if (row["loaded_bytes"] != expected_loaded
                or row["stored_bytes"] != expected_stored
                or row["tensor_fma"] != expected_fma):
            raise ValueError(f"cell {trial_key} work accounting changed")
    return {"meta": meta, "rows": rows}


def _derive_candidate(parsed: dict[str, object]) -> dict[str, object]:
    rows = parsed["rows"]
    if not isinstance(rows, list):
        raise TypeError("parsed rows must be a list")
    cells = []
    medians = {}
    all_cells_stable = True
    for grid in GRIDS:
        for rounds in ROUNDS:
            cell = [
                row for row in rows
                if row["grid"] == grid and row["rounds"] == rounds
            ]
            maximum_cycles = [row["maximum_cta_cycles"] for row in cell]
            median_cycles = [row["median_cta_cycles"] for row in cell]
            center = float(median(maximum_cycles))
            relative_spread = (max(maximum_cycles) - min(maximum_cycles)) / center
            stable = relative_spread <= MAXIMUM_TRIAL_SPREAD
            all_cells_stable = all_cells_stable and stable
            medians[(grid, rounds)] = float(median(median_cycles))
            cells.append({
                "grid": grid,
                "rounds": rounds,
                "median_of_median_cta_cycles": medians[(grid, rounds)],
                "median_maximum_cta_cycles": center,
                "maximum_cycle_relative_spread": relative_spread,
                "stable_within_10_percent": stable,
            })
    grid_one_slopes = []
    for left, right in zip(ROUNDS, ROUNDS[1:]):
        slope = (
            (medians[(1, right)] - medians[(1, left)])
            / (right - left))
        if slope <= 0.0:
            raise ValueError("grid-one steady slope must be positive")
        grid_one_slopes.append({
            "rounds": [left, right], "cycles_per_round": slope,
        })
    slope_values = [cell["cycles_per_round"] for cell in grid_one_slopes]
    slope_center = float(median(slope_values))
    slope_spread = (max(slope_values) - min(slope_values)) / slope_center
    slope_stable = slope_spread <= MAXIMUM_STEADY_SLOPE_SPREAD
    concurrency_rounds = 128
    concurrency = [
        {
            "grid": grid,
            "median_cta_cycles": medians[(grid, concurrency_rounds)],
            "ratio_to_grid108": (
                medians[(grid, concurrency_rounds)]
                / medians[(108, concurrency_rounds)]),
        }
        for grid in (64, 80, 108)
    ]
    return {
        "schema": "nvidia_a800_bf16_staged_tile_dependency_candidate_v1",
        "hardware_id": "nvidia_a800",
        "mechanism": "bf16_tile_ldgsts_shared_hmma_handoff",
        "scope": {
            "tile_shape": TILE_SHAPE,
            "warps_per_cta": WARPS_PER_CTA,
            "block_threads": BLOCK_THREADS,
            "pipeline_stages": PIPELINE_STAGES,
            "panel_bytes": PANEL_BYTES,
            "resident_ctas_per_sm": 1,
            "dynamic_shared_bytes_per_cta": DYNAMIC_SHARED_BYTES,
            "grid_points": list(GRIDS),
            "round_points": list(ROUNDS),
        },
        "cell_summaries": cells,
        "grid_one_incremental_slopes": grid_one_slopes,
        "median_grid_one_steady_cycles_per_round": slope_center,
        "grid_one_steady_slope_relative_spread": slope_spread,
        "grid_one_steady_service_stable_within_15_percent": slope_stable,
        "concurrency_at_128_rounds": concurrency,
        "all_cells_stable_within_10_percent": all_cells_stable,
        "probe_success": all_cells_stable and slope_stable,
    }


def _validate_sass(text: str) -> dict[str, object]:
    required = (
        "LDGSTS.E.BYPASS.128", " LDS ", "HMMA.16816.F32.BF16",
        "STG.E", "BAR.SYNC")
    missing = [opcode for opcode in required if opcode not in text]
    if missing:
        raise ValueError(f"SASS is missing required opcodes: {missing}")
    forbidden = ("LDL", "STL")
    present = [opcode for opcode in forbidden if opcode in text]
    if present:
        raise ValueError(f"SASS contains local-memory opcodes: {present}")
    return {
        "required_opcodes": list(required),
        "forbidden_opcodes_absent": list(forbidden),
    }


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner", required=True)
    arguments = parser.parse_args()
    if arguments.device != 0:
        raise ValueError("this versioned probe is pinned to physical device 0")
    if arguments.trials not in range(MINIMUM_TRIALS, MAXIMUM_TRIALS + 1):
        raise ValueError(
            f"trials must be in [{MINIMUM_TRIALS}, {MAXIMUM_TRIALS}]")
    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("run-id has invalid syntax")
    if not arguments.owner.strip():
        raise ValueError("owner must be non-empty")
    if not SOURCE_PATH.is_file() or not NVCC.is_file() or not CUOBJDUMP.is_file():
        raise FileNotFoundError("source or pinned CUDA tool is missing")
    output = (OUTPUT_ROOT / arguments.run_id).resolve()
    if output == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output escaped the profiler output root")
    if output.exists():
        raise FileExistsError(output)
    device = _require_idle_device(arguments.device)
    output.mkdir(parents=True)
    command = [
        str(RUNNER_PATH), "--device", str(arguments.device),
        "--trials", str(arguments.trials), "--run-id", arguments.run_id,
        "--owner", arguments.owner,
    ]
    with tempfile.TemporaryDirectory(prefix="a800_tile_dependency_") as temp:
        executable = Path(temp) / "tile_dependency"
        compile_result = _run([
            str(NVCC), "-std=c++17", "-O3", "-arch=sm_80", "-lineinfo",
            "-Xptxas=-v", str(SOURCE_PATH), "-o", str(executable),
        ], cwd=REPOSITORY_ROOT)
        sass_result = _run([
            str(CUOBJDUMP), "--dump-sass", str(executable),
        ], cwd=REPOSITORY_ROOT)
        sass_validation = _validate_sass(sass_result.stdout)
        native = _run([
            str(executable), str(arguments.device), str(arguments.trials),
        ], cwd=REPOSITORY_ROOT)
        (output / "native_stdout.csv").write_text(
            native.stdout, encoding="utf-8")
        (output / "compile_stderr.txt").write_text(
            compile_result.stderr, encoding="utf-8")
        (output / "sass.txt").write_text(
            sass_result.stdout, encoding="utf-8")
    parsed = _parse_native_output(native.stdout, arguments.trials)
    candidate = _derive_candidate(parsed)
    candidate["provenance"] = {
        "owner": arguments.owner,
        "runner_pid": os.getpid(),
        "command": shlex.join(command),
        "baseline_commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "source": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "runner": str(RUNNER_PATH.relative_to(REPOSITORY_ROOT)),
        "runner_sha256": _sha256(RUNNER_PATH),
        "device": device,
        "trials": arguments.trials,
        "production_or_evaluation_duration_consumed": False,
    }
    candidate["sass"] = sass_validation
    _write_json(output / "candidate.json", candidate)
    postflight = _wait_for_idle_device(arguments.device)
    _write_json(output / "postflight_device.json", postflight)
    print(output / "candidate.json")


if __name__ == "__main__":
    main()
