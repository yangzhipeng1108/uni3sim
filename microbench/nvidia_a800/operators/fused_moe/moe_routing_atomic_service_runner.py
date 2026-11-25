#!/usr/bin/env python3
"""Capture independent A800 shared/global MoE-routing atomic services."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("moe_routing_atomic_service.cu")
RUNNER = Path(__file__)
OUTPUT_ROOT = ROOT / "profiler_output" / "microbench" / "nvidia_a800"
VLLM_MOE_BINARY = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"]) / "_moe_C.abi3.so"
)
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
CASES = (
    (0, "shared_cta_e128_r262144", 0, 262144, 128, 1, 1024),
    (1, "global_grid1024_e128_r262144", 1, 262144, 128, 1024, 256),
    (2, "global_grid128_e32_r32768", 1, 32768, 32, 128, 256),
    (3, "global_grid256_e8_r65536", 1, 65536, 8, 256, 256),
)
RUNS = 3
SAMPLES_PER_RUN = 7


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment)


def _device(device_id):
    output = _run([
        "nvidia-smi", f"--id={device_id}",
        "--query-gpu=index,name,uuid,utilization.gpu,utilization.memory,"
        "memory.used,clocks.current.sm,driver_version",
        "--format=csv,noheader,nounits",
    ]).stdout.strip()
    fields = tuple(item.strip() for item in output.split(","))
    if len(fields) != 8:
        raise ValueError(f"unexpected device row: {output!r}")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "gpu_utilization_pct": int(fields[3]),
        "memory_utilization_pct": int(fields[4]),
        "memory_used_mib": int(fields[5]), "sm_clock_mhz": int(fields[6]),
        "driver_version": fields[7],
    }
    if row["index"] != device_id or row["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    processes = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if any(line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
           for line in processes.splitlines()):
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


def _parse(stdout, case, run, before, after):
    case_id, case_name, mode, items, experts, grid, block = case
    rows = []
    for values in csv.reader(stdout.splitlines()):
        if len(values) != 9:
            raise ValueError(f"unexpected native row: {values!r}")
        row = {
            "case_id": case_id, "case_name": values[0],
            "sample": int(values[1]), "mode": int(values[2]),
            "items": int(values[3]), "experts": int(values[4]),
            "grid": int(values[5]), "block": int(values[6]),
            "milliseconds": float(values[7]),
            "atomic_ops_per_s": float(values[8]), "run": run,
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
        }
        expected = (case_name, mode, items, experts, grid, block)
        actual = (row["case_name"], row["mode"], row["items"],
                  row["experts"], row["grid"], row["block"])
        if (actual != expected or not 0 <= row["sample"] < SAMPLES_PER_RUN
                or row["milliseconds"] <= 0 or row["atomic_ops_per_s"] <= 0):
            raise ValueError(f"native row violates frozen case: {row}")
        rows.append(row)
    if len(rows) != SAMPLES_PER_RUN:
        raise ValueError("native sample count is incomplete")
    if {row["sample"] for row in rows} != set(range(SAMPLES_PER_RUN)):
        raise ValueError("native sample indices are incomplete")
    return rows


def _summary(rows):
    summaries = []
    for case in CASES:
        case_id, case_name, mode, items, experts, grid, block = case
        run_medians = []
        for run in range(RUNS):
            samples = [row["milliseconds"] for row in rows
                       if row["case_id"] == case_id and row["run"] == run]
            if len(samples) != SAMPLES_PER_RUN:
                raise ValueError("atomic capture is incomplete")
            run_medians.append(statistics.median(samples))
        median_ms = statistics.median(run_medians)
        summaries.append({
            "case_id": case_id, "case_name": case_name, "mode": mode,
            "items": items, "experts": experts, "grid": grid, "block": block,
            "run_median_milliseconds": run_medians,
            "median_milliseconds": median_ms,
            "median_atomic_ops_per_s": items / (median_ms * 1.0e-3),
            "cross_run_relative_range": (
                (max(run_medians) - min(run_medians)) / median_ms),
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    arguments = parser.parse_args()
    output_dir = arguments.output_dir.resolve()
    if output_dir == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output_dir.parents:
        raise ValueError("output directory must be below the A800 profiler root")
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    if not arguments.nvcc.is_file() or not os.access(arguments.nvcc, os.X_OK):
        raise ValueError(f"nvcc is not executable: {arguments.nvcc}")
    output_dir.mkdir(parents=True)
    binary = output_dir / "moe_routing_atomic_service"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        str(SOURCE), "-o", str(binary),
    ]
    _run(compile_command)
    campaign_before = _wait_idle(arguments.device_id)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    executions = []
    for run in range(RUNS):
        for case in CASES:
            before = _wait_idle(arguments.device_id)
            command = [str(binary), str(case[0]), str(SAMPLES_PER_RUN)]
            result = _run(command, environment=environment)
            after = _device(arguments.device_id)
            rows.extend(_parse(result.stdout, case, run, before, after))
            executions.append({
                "run": run, "case_id": case[0], "command": command,
                "stdout": result.stdout, "stderr": result.stderr,
                "device_before": before, "device_after": after,
            })
    campaign_after = _wait_idle(arguments.device_id)
    raw_path = output_dir / "raw.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(_summary(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest = {
        "schema": "a800_moe_routing_atomic_service_capture_v1",
        "baseline_commit": _run(["git", "rev-parse", "HEAD"]).stdout.strip(),
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha256(SOURCE),
        "runner_path": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER), "binary_sha256": _sha256(binary),
        "vllm_moe_binary": str(VLLM_MOE_BINARY),
        "vllm_moe_binary_sha256": _sha256(VLLM_MOE_BINARY),
        "compile_command": compile_command, "invocation_command": os.sys.argv,
        "campaign_device_before": campaign_before,
        "campaign_device_after": campaign_after, "executions": executions,
        "mechanism_scope": {
            "shared": "one 1024-thread CTA, local counter atomicAdd stream",
            "global": "fixed 256-thread CTA grid, global counter atomicAdd stream",
            "histogram": "deterministic uniform synthetic counters",
        },
        "production_or_evaluation_duration_consumed": False,
        "production_kernel_reproduced": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": _summary(rows)},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
