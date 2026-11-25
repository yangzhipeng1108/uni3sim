#!/usr/bin/env python3
"""Capture the A800 BM64 grouped-MoE masked-row operand mechanism."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("fused_moe_operand_service.cu")
RUNNER = Path(__file__)
OUTPUT_ROOT = ROOT / "profiler_output" / "microbench" / "nvidia_a800"
VLLM_SOURCE = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
VLLM_SOURCE_SHA256 = (
    "8be8477a653894194ef5c5095afc287b0409a51d980590a36dd43c8471f6740e")
VLLM_SOURCE_CONTRACT = (
    "token_mask = offs_token < num_valid_tokens",
    "mask=token_mask[:, None] &",
    "other=0.0",
    '"BLOCK_SIZE_M": 64',
    '"BLOCK_SIZE_N": 64',
    '"BLOCK_SIZE_K": 32',
    '"GROUP_SIZE_M": 8',
)
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
CASE_ID = 1
CASE_NAME = "operand_bm64_bn64_bk32"
TILE_M = 64
TILE_N = 64
TILE_K = 32
GRID = 4096
BLOCK_THREADS = 128
ALLOCATION_ACTIVATION_BYTES = 4096 * 2048 * 2
ALLOCATION_WEIGHT_BYTES = 64 * 2048 * 64 * 2
ACTIVE_ROW_SWEEP = tuple(range(1, TILE_M + 1))
RUNS = 3
SAMPLES_PER_RUN = 5
STABILITY_RELATIVE_LIMIT = 0.20
FIELD_UNITS = {
    "run": "ordinal",
    "sample": "ordinal",
    "tile_m": "row",
    "tile_n": "column",
    "tile_k": "element",
    "grid": "CTA",
    "milliseconds": "ms",
    "logical_bandwidth_gb_s": "GB/s (10^9 byte/s)",
    "allocation_activation_bytes": "byte",
    "allocation_weight_bytes": "byte",
    "logical_a_read_bytes": "byte",
    "logical_b_read_bytes": "byte",
    "active_rows": "row/CTA",
    "sm_clock_before_mhz": "MHz",
    "sm_clock_after_mhz": "MHz",
    "gpu_utilization_before_pct": "percent",
    "gpu_utilization_after_pct": "percent",
    "run_median_milliseconds": "ms",
    "median_milliseconds": "ms",
    "increment_over_one_active_row_ms": "ms",
    "cross_run_relative_range": "ratio",
}
EXACT_DOMAIN = {
    "hardware": "nvidia_a800",
    "architecture": "sm_80",
    "operand_dtype": "bfloat16",
    "tile_m": TILE_M,
    "tile_n": TILE_N,
    "tile_k": TILE_K,
    "group_size_m": 8,
    "grid_ctas": GRID,
    "threads_per_cta": BLOCK_THREADS,
    "activation_allocation_bytes": ALLOCATION_ACTIVATION_BYTES,
    "weight_allocation_bytes": ALLOCATION_WEIGHT_BYTES,
    "active_rows": list(ACTIVE_ROW_SWEEP),
    "a_operand_mask": "row < active_rows; inactive shared-memory words are zero",
    "b_operand": "unmasked and identical for every active-row case",
    "consumer": "shared-memory uint4 XOR checksum",
}


def _run(command, *, environment=None):
    result = subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment)
    if result.stderr.strip():
        raise RuntimeError(f"command wrote stderr: {result.stderr.strip()}")
    return result.stdout


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1048576)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path):
    resolved = path.resolve(strict=True)
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_vllm_source():
    if _sha256(VLLM_SOURCE) != VLLM_SOURCE_SHA256:
        raise ValueError("vLLM fused_moe.py SHA256 changed")
    source = VLLM_SOURCE.read_text(encoding="utf-8")
    missing = tuple(item for item in VLLM_SOURCE_CONTRACT if item not in source)
    if missing:
        raise ValueError(f"vLLM masked-row contract changed: {missing}")


def _device(device_id):
    output = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,pci.bus_id,pstate,utilization.gpu,"
        "utilization.memory,memory.used,clocks.current.sm,"
        "clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ]).strip()
    fields = [field.strip() for field in output.split(",")]
    if len(fields) != 11:
        raise ValueError(f"unexpected device row: {output!r}")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "pci_bus_id": fields[3], "pstate": fields[4],
        "gpu_utilization_pct": int(fields[5]),
        "memory_utilization_pct": int(fields[6]),
        "memory_used_mib": int(fields[7]), "sm_clock_mhz": int(fields[8]),
        "memory_clock_mhz": int(fields[9]), "driver_version": fields[10],
    }
    if row["index"] != device_id or row["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _run([
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    if any(
            line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
            for line in applications.splitlines()):
        raise RuntimeError("target GPU has a compute process")
    return row


def _wait_idle(device_id):
    rows = []
    for _ in range(10):
        row = _device(device_id)
        rows.append(row)
        if (row["gpu_utilization_pct"] == 0
                and row["memory_utilization_pct"] == 0
                and row["memory_used_mib"] <= 8):
            return row
        time.sleep(0.5)
    raise RuntimeError(f"GPU did not become idle: {rows}")


def _execute(binary, active_rows, environment):
    command = [
        str(binary), str(CASE_ID), str(active_rows), str(SAMPLES_PER_RUN)]
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr)
    validation = tuple(line for line in stderr.splitlines() if line.strip())
    expected = f"validated case={CASE_NAME} active_rows={active_rows} "
    if len(validation) != 1 or not validation[0].startswith(expected):
        raise ValueError(f"unexpected validation output: {stderr!r}")
    return command, process.pid, stdout, validation[0]


def _parse(text, *, run, active_rows, process_id, before, after):
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if len(fields) != 13 or fields[0] != CASE_NAME:
            raise ValueError(f"unexpected output row: {fields!r}")
        row = {
            "run": run, "case_id": CASE_ID, "case_name": fields[0],
            "sample": int(fields[1]), "tile_m": int(fields[2]),
            "tile_n": int(fields[3]), "tile_k": int(fields[4]),
            "grid": int(fields[5]), "active_rows": int(fields[6]),
            "allocation_activation_bytes": int(fields[7]),
            "allocation_weight_bytes": int(fields[8]),
            "logical_a_read_bytes": int(fields[9]),
            "logical_b_read_bytes": int(fields[10]),
            "milliseconds": float(fields[11]),
            "logical_bandwidth_gb_s": float(fields[12]),
            "process_id": process_id,
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
            "gpu_utilization_before_pct": before["gpu_utilization_pct"],
            "gpu_utilization_after_pct": after["gpu_utilization_pct"],
        }
        expected_a_bytes = GRID * 2048 * active_rows * 2
        expected_b_bytes = GRID * 2048 * TILE_N * 2
        if ((row["tile_m"], row["tile_n"], row["tile_k"])
                != (TILE_M, TILE_N, TILE_K)
                or row["grid"] != GRID or row["active_rows"] != active_rows
                or row["allocation_activation_bytes"]
                != ALLOCATION_ACTIVATION_BYTES
                or row["allocation_weight_bytes"] != ALLOCATION_WEIGHT_BYTES
                or row["logical_a_read_bytes"] != expected_a_bytes
                or row["logical_b_read_bytes"] != expected_b_bytes
                or not 0 <= row["sample"] < SAMPLES_PER_RUN
                or row["milliseconds"] <= 0
                or row["logical_bandwidth_gb_s"] <= 0):
            raise ValueError(f"row violates the frozen case: {row}")
        rows.append(row)
    if len(rows) != SAMPLES_PER_RUN:
        raise ValueError("wrong sample count")
    if {row["sample"] for row in rows} != set(range(SAMPLES_PER_RUN)):
        raise ValueError("sample indices are incomplete or duplicated")
    return rows


def _relative_range(values):
    median = statistics.median(values)
    return (max(values) - min(values)) / median


def _build_report(rows, capture_identity):
    cells = []
    medians = {}
    stability_pass = True
    for active_rows in ACTIVE_ROW_SWEEP:
        run_medians_ms = []
        for run in range(RUNS):
            values = [
                row["milliseconds"] for row in rows
                if row["active_rows"] == active_rows and row["run"] == run]
            if len(values) != SAMPLES_PER_RUN:
                raise ValueError(
                    f"active_rows={active_rows} run={run} is incomplete")
            run_medians_ms.append(statistics.median(values))
        spread = _relative_range(run_medians_ms)
        cell_pass = spread <= STABILITY_RELATIVE_LIMIT
        stability_pass = stability_pass and cell_pass
        median_ms = statistics.median(run_medians_ms)
        medians[active_rows] = median_ms
        cells.append({
            "active_rows": active_rows,
            "run_median_milliseconds": run_medians_ms,
            "median_milliseconds": median_ms,
            "cross_run_relative_range": spread,
            "cross_run_stability_gate": cell_pass,
        })
    anchor_ms = medians[ACTIVE_ROW_SWEEP[0]]
    signals = [{
        "active_rows": active_rows,
        "increment_over_one_active_row_ms": medians[active_rows] - anchor_ms,
    } for active_rows in ACTIVE_ROW_SWEEP]
    nonnegative = all(
        item["increment_over_one_active_row_ms"] >= 0 for item in signals)
    monotonic_observation = all(
        medians[right] >= medians[left]
        for left, right in zip(ACTIVE_ROW_SWEEP, ACTIVE_ROW_SWEEP[1:]))
    full_row_increment_positive = (
        medians[ACTIVE_ROW_SWEEP[-1]] > anchor_ms)
    publishable = (
        stability_pass and nonnegative and full_row_increment_positive)
    return {
        "schema": "a800_fused_moe_masked_row_operand_report_v1",
        "exact_domain": EXACT_DOMAIN,
        "field_units": FIELD_UNITS,
        "capture_identity": capture_identity,
        "cells": cells,
        "signals": signals,
        "gates": {
            "three_run_stability_relative_limit": STABILITY_RELATIVE_LIMIT,
            "all_cells_cross_run_stability_at_most_20_percent": stability_pass,
            "increment_over_one_row_nonnegative": nonnegative,
            "full_row_increment_positive": full_row_increment_positive,
        },
        "observations": {
            "median_duration_monotonic_with_active_rows": (
                monotonic_observation),
        },
        "publication": {
            "exact_domain_evidence_publishable": publishable,
            "production_parameter_published": False,
            "rule": (
                "Evidence may describe only the exact BM64/BN64/BK32 BF16 "
                "masked-A-row operand-read mechanism when every three-run "
                "stability and endpoint-signal gate passes. Each row count "
                "must be consumed as an exact measured service point: CUDA "
                "warp scheduling does not require adjacent points to be "
                "monotonic. It cannot publish a production duration, fitted "
                "parameter, full-GEMM result, routing result, or another "
                "launch domain."),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--nvcc", required=True, type=Path)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if (output_dir == OUTPUT_ROOT.resolve()
            or OUTPUT_ROOT.resolve() not in output_dir.parents):
        raise ValueError("output directory must be below the A800 root")
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    _validate_vllm_source()
    output_dir.mkdir(parents=True)
    binary = output_dir / "fused_moe_operand_service"
    sass = output_dir / "fused_moe_operand_service.sass"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE), "-o", str(binary),
    ]
    _run(compile_command)
    sass_command = [str(arguments.cuobjdump), "--dump-sass", str(binary)]
    sass_text = _run(sass_command)
    if "grouped_operand_read" not in sass_text:
        raise ValueError("SASS does not contain grouped_operand_read")
    sass.write_text(sass_text, encoding="utf-8")
    campaign_before = _wait_idle(arguments.device_id)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    executions = []
    for run in range(RUNS):
        order = ACTIVE_ROW_SWEEP[run:] + ACTIVE_ROW_SWEEP[:run]
        for active_rows in order:
            before = _wait_idle(arguments.device_id)
            command, process_id, stdout, validation = _execute(
                binary, active_rows, environment)
            after = _device(arguments.device_id)
            rows.extend(_parse(
                stdout, run=run, active_rows=active_rows,
                process_id=process_id, before=before, after=after))
            executions.append({
                "run": run, "active_rows": active_rows, "command": command,
                "process_id": process_id, "validation": validation,
                "device_before": before, "device_after": after,
            })
    campaign_after = _wait_idle(arguments.device_id)
    raw = output_dir / "raw.csv"
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    capture_identity = {
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).strip(),
        "source": _identity(SOURCE),
        "runner": _identity(RUNNER),
        "vllm_source": _identity(VLLM_SOURCE),
        "binary": _identity(binary),
        "sass": _identity(sass),
        "compile_command": compile_command,
        "sass_command": sass_command,
        "nvcc": {
            "identity": _identity(arguments.nvcc),
            "version": _run([str(arguments.nvcc), "--version"]).strip(),
        },
        "cuobjdump": {
            "identity": _identity(arguments.cuobjdump),
            "version": _run([str(arguments.cuobjdump), "--version"]).strip(),
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "cuda_visible_devices": str(arguments.device_id),
            "campaign_device_before": campaign_before,
            "campaign_device_after": campaign_after,
        },
    }
    report = _build_report(rows, capture_identity)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest = {
        "schema": "a800_fused_moe_masked_row_operand_capture_v1",
        "capture_identity": capture_identity,
        "field_units": FIELD_UNITS,
        "invocation_command": sys.argv,
        "executions": executions,
        "runs": RUNS,
        "samples_per_run": SAMPLES_PER_RUN,
        "active_row_sweep": list(ACTIVE_ROW_SWEEP),
        "raw": _identity(raw),
        "report": _identity(report_path),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
        "timing_domain": "CUDA event around one operand-read-only kernel",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "row_count": len(rows),
        "gates": report["gates"],
        "publication": report["publication"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
