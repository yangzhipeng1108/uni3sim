#!/usr/bin/env python3
"""Capture the bounded A800 indirect-row gather mechanism experiment."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fused_moe_gather_service.cu")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / "nvidia_a800")
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
CASES = (
    (0, "scatter_bm16_bk64_l2", 16, 64, 4096, True, "publishable_l2"),
    (1, "scatter_bm64_bk32_l2", 64, 32, 4096, True, "publishable_l2"),
    (2, "contiguous_bm16_bk64_l2", 16, 64, 4096, False,
     "contiguous_control"),
    (3, "scatter_bm16_bk64_gt_l2", 16, 64, 32768, True,
     "greater_than_l2_control"),
)
TRIALS = 3
SAMPLES_PER_TRIAL = 5


def _run(command, *, environment=None):
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment)
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
        "--query-gpu=index,name,uuid,pstate,utilization.gpu,"
        "utilization.memory,memory.used,clocks.current.sm,"
        "clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ]).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 10:
        raise ValueError(f"unexpected nvidia-smi device row: {output!r}")
    row = {
        "index": int(fields[0]),
        "name": fields[1],
        "uuid": fields[2],
        "pstate": fields[3],
        "utilization_gpu_pct": int(fields[4]),
        "utilization_memory_pct": int(fields[5]),
        "memory_used_mib": int(fields[6]),
        "sm_clock_mhz": int(fields[7]),
        "memory_clock_mhz": int(fields[8]),
        "driver_version": fields[9],
    }
    if row["index"] != device_id or row["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    if (require_idle
            and (row["utilization_gpu_pct"] != 0
                 or row["utilization_memory_pct"] != 0
                 or row["memory_used_mib"] > 8)):
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


def _wait_for_idle(device_id):
    observations = []
    for _ in range(10):
        row = _device_row(device_id, require_idle=False)
        observations.append(row)
        if (row["utilization_gpu_pct"] == 0
                and row["utilization_memory_pct"] == 0
                and row["memory_used_mib"] <= 8):
            return row
        time.sleep(0.5)
    raise RuntimeError(
        f"target GPU did not become idle after ten checks: {observations}")


def _execute_case(binary_path, case_id, environment):
    command = [str(binary_path), str(case_id), str(SAMPLES_PER_TRIAL)]
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr)
    validation_lines = tuple(
        line for line in stderr.splitlines() if line.strip())
    if len(validation_lines) != 1 or not validation_lines[0].startswith(
            "validated case="):
        raise ValueError(f"unexpected validation output: {stderr!r}")
    return command, process.pid, stdout, validation_lines[0]


def _parse_rows(text, *, trial, case_definition, process_id, before, after):
    case_id, name, bm, bk, tokens, scattered, purpose = case_definition
    reader = csv.reader(io.StringIO(text))
    rows = []
    for fields in reader:
        if len(fields) != 9:
            raise ValueError(f"unexpected benchmark row: {fields!r}")
        if fields[0] != name:
            raise ValueError("benchmark case name changed")
        row = {
            "trial": trial,
            "case_id": case_id,
            "case_name": name,
            "purpose": purpose,
            "sample": int(fields[1]),
            "tile_m": int(fields[2]),
            "tile_k": int(fields[3]),
            "tokens": int(fields[4]),
            "scattered": int(fields[5]),
            "grid": int(fields[6]),
            "milliseconds": float(fields[7]),
            "logical_bandwidth_gb_s": float(fields[8]),
            "process_id": process_id,
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
            "memory_clock_before_mhz": before["memory_clock_mhz"],
            "memory_clock_after_mhz": after["memory_clock_mhz"],
            "gpu_utilization_before_pct": before["utilization_gpu_pct"],
            "gpu_utilization_after_pct": after["utilization_gpu_pct"],
        }
        if (row["tile_m"] != bm or row["tile_k"] != bk
                or row["tokens"] != tokens
                or bool(row["scattered"]) != scattered
                or not 0 <= row["sample"] < SAMPLES_PER_TRIAL
                or row["milliseconds"] <= 0.0
                or row["logical_bandwidth_gb_s"] <= 0.0):
            raise ValueError(f"benchmark row violates the frozen case: {row}")
        rows.append(row)
    if len(rows) != SAMPLES_PER_TRIAL:
        raise ValueError("benchmark emitted the wrong sample count")
    return rows


def _summaries(rows):
    summaries = []
    for case_definition in CASES:
        case_id, name, bm, bk, tokens, scattered, purpose = case_definition
        values = [
            row["logical_bandwidth_gb_s"] for row in rows
            if row["case_id"] == case_id]
        if len(values) != TRIALS * SAMPLES_PER_TRIAL:
            raise ValueError("capture matrix is incomplete")
        ordered = sorted(values)
        summaries.append({
            "case_id": case_id,
            "case_name": name,
            "purpose": purpose,
            "tile_m": bm,
            "tile_k": bk,
            "logical_row_bytes": bk * 2,
            "tokens": tokens,
            "activation_footprint_mib": tokens * 2048 * 2 / 1048576,
            "scattered": scattered,
            "sample_count": len(values),
            "median_logical_bandwidth_gb_s": statistics.median(values),
            "minimum_logical_bandwidth_gb_s": ordered[0],
            "maximum_logical_bandwidth_gb_s": ordered[-1],
            "relative_range_pct": (
                (ordered[-1] - ordered[0]) / statistics.median(values)
                * 100.0),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--nvcc", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if (output_dir == OUTPUT_ROOT.resolve()
            or OUTPUT_ROOT.resolve() not in output_dir.parents):
        raise ValueError("output directory must be below the A800 root")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not arguments.nvcc.is_file():
        raise FileNotFoundError(f"nvcc does not exist: {arguments.nvcc}")
    output_dir.mkdir(parents=True)
    binary_path = output_dir / "fused_moe_gather_service"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE_PATH), "-o", str(binary_path),
    ]
    _run(compile_command)
    campaign_before = _wait_for_idle(arguments.device_id)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    executions = []
    for trial in range(TRIALS):
        order = CASES[trial:] + CASES[:trial]
        for case_definition in order:
            before = _wait_for_idle(arguments.device_id)
            command, process_id, stdout, validation = _execute_case(
                binary_path, case_definition[0], environment)
            after = _device_row(arguments.device_id, require_idle=False)
            rows.extend(_parse_rows(
                stdout, trial=trial, case_definition=case_definition,
                process_id=process_id, before=before, after=after))
            executions.append({
                "trial": trial,
                "case_id": case_definition[0],
                "command": command,
                "process_id": process_id,
                "validation": validation,
                "device_before": before,
                "device_after": after,
            })
    campaign_after = _wait_for_idle(arguments.device_id)
    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = _summaries(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest = {
        "schema": "a800_fused_moe_gather_service_capture_v1",
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).strip(),
        "source_path": str(SOURCE_PATH.relative_to(REPOSITORY_ROOT)),
        "source_sha256": _sha256(SOURCE_PATH),
        "binary_sha256": _sha256(binary_path),
        "raw_sha256": _sha256(raw_path),
        "summary_sha256": _sha256(summary_path),
        "nvcc_path": str(arguments.nvcc),
        "nvcc_version": _run([str(arguments.nvcc), "--version"]).strip(),
        "compile_command": compile_command,
        "invocation_command": sys.argv,
        "campaign_device_before": campaign_before,
        "campaign_device_after": campaign_after,
        "executions": executions,
        "cases": [
            {
                "case_id": item[0],
                "case_name": item[1],
                "tile_m": item[2],
                "tile_k": item[3],
                "tokens": item[4],
                "scattered": item[5],
                "purpose": item[6],
            }
            for item in CASES
        ],
        "trials": TRIALS,
        "samples_per_trial": SAMPLES_PER_TRIAL,
        "logical_traffic_definition": "routed_rows * K * sizeof(bf16)",
        "production_or_evaluation_duration_consumed": False,
        "timing_domain": "CUDA event around one gather-only kernel launch",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "summaries": summaries,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
