#!/usr/bin/env python3
"""Capture independent H200 shared/global MoE-routing atomic services.

H200 owner copy of
microbench/nvidia_a800/operators/fused_moe/moe_routing_atomic_service_runner.py.
Cases, 3 runs x 7 samples, per-execution idle wait, parsing and the
median-of-run-medians summary are unchanged.  Hardware coupling only:
  - ``-arch=sm_90``; nvcc is the H200-host CUDA 12.9 toolkit (passed with
    ``--nvcc``); device name NVIDIA H200; the output directory is any new
    directory (the remote run directory), copied back under
    profiler_output/microbench/nvidia_h200;
  - vLLM MoE binary identity: the H200 vLLM 0.24
    ``_moe_C_stable_libtorch.abi3.so``, passed with ``--vllm-moe-binary``
    (0.24 no longer ships ``_moe_C``);
  - clock gate: each execution runs under a 50-ms nvidia-smi SM-clock log.
    These kernels last 10-80 us, so utilization-based busy samples are
    usually absent; the gate uses the process window instead: from the first
    sample at >= 1980 MHz (context boost) to the last such sample no sample
    may fall below 1980 MHz, and any busy sample must be >= 1980 MHz.  A
    rejected execution is repeated (at most 3 attempts) and counted;
  - ``publish`` derives the yaml per-subcore rates
    median ops/s / (132 SMs x 4 subcores) into
    microbench/nvidia_h200/references/moe_routing_atomic_service_sm90.json.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("moe_routing_atomic_service.cu")
RUNNER = Path(__file__)
DEVICE_NAME = "NVIDIA H200"
BOOST_MHZ = 1980
SM_COUNT = 132
SUBCORES_PER_SM = 4
MAXIMUM_ATTEMPTS = 3
REFERENCE = "microbench/nvidia_h200/references/moe_routing_atomic_service_sm90.json"
YAML_KEYS = {
    "global_grid256_e8_r65536": "moe_routing_global_atomic_e8",
    "global_grid128_e32_r32768": "moe_routing_global_atomic_e32",
    "global_grid1024_e128_r262144": "moe_routing_global_atomic_e128",
}
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


def _clocked(command, environment, device_id, clock_path):
    with clock_path.open("w", encoding="utf-8") as stream:
        sampler = subprocess.Popen(
            ["nvidia-smi", "-i", str(device_id),
             "--query-gpu=clocks.sm,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=stream, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.3)
            result = _run(command, environment=environment)
        finally:
            time.sleep(0.2)
            sampler.terminate()
            sampler.wait()
    samples = [tuple(int(v) for v in line.split(","))
               for line in clock_path.read_text().splitlines() if line.strip()]
    boosted = [index for index, (clock, _) in enumerate(samples)
               if clock >= BOOST_MHZ]
    window = (samples[boosted[0]:boosted[-1] + 1] if boosted else [])
    busy = [clock for clock, utilization in samples if utilization > 0]
    clock = {
        "samples": len(samples), "window_samples": len(window),
        "window_min_mhz": min(c for c, _ in window) if window else None,
        "busy_samples": len(busy),
        "busy_min_mhz": min(busy) if busy else None,
        "accepted": (bool(window)
                     and min(c for c, _ in window) >= BOOST_MHZ
                     and all(c >= BOOST_MHZ for c in busy)),
    }
    return result, clock


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


def publish(run_dir, remote_run_dir):
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    if _sha256(run_dir / "raw.csv") != manifest["raw_sha256"]:
        raise ValueError("raw capture changed after the run")
    divisor = SM_COUNT * SUBCORES_PER_SM
    commit = _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
    reference = {
        "schema": "nvidia_h200_moe_routing_atomic_service_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": "NVIDIA H200",
        "compute_capability": "sm_90",
        "source": manifest["source_path"],
        "source_sha256": manifest["source_sha256"],
        "runner": str(RUNNER.relative_to(ROOT)),
        "runner_sha256": _sha256(RUNNER),
        "cases": summary,
        "per_subcore_rate_derivation": (
            "yaml moe_routing_global_atomic_eN = median_atomic_ops_per_s of "
            "the exact global case / (132 SMs x 4 compute_alu subcores); "
            "median over three run medians of seven samples, as on A800"),
        "yaml_values_ops_per_s_per_subcore": {
            YAML_KEYS[row["case_name"]]:
                row["median_atomic_ops_per_s"] / divisor
            for row in summary if row["case_name"] in YAML_KEYS},
        "production_or_evaluation_duration_consumed": False,
        "production_kernel_reproduced": False,
        "h200_capture_provenance": {
            "git_commit": commit,
            "capture_baseline_commit": manifest["baseline_commit"],
            "raw_dir": str(run_dir.relative_to(ROOT)),
            "remote_raw_dir": remote_run_dir,
            "raw_sha256": manifest["raw_sha256"],
            "binary_sha256": manifest["binary_sha256"],
            "compile_command": manifest["compile_command"],
            "vllm_moe_binary": manifest["vllm_moe_binary"],
            "vllm_moe_binary_sha256": manifest["vllm_moe_binary_sha256"],
            "clock_gate": manifest["clock_gate"],
            "accepted_executions": manifest["accepted_executions"],
            "rejected_executions": manifest["rejected_executions"],
            "execution_clocks": manifest["execution_clocks"],
        },
    }
    (ROOT / REFERENCE).write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reference["yaml_values_ops_per_s_per_subcore"], indent=2))


def main():
    if sys.argv[1:2] == ["publish"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--run-dir", required=True, type=Path)
        parser.add_argument("--remote-run-dir", required=True)
        arguments = parser.parse_args()
        publish(arguments.run_dir, arguments.remote_run_dir)
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--vllm-moe-binary", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    arguments = parser.parse_args()
    if not arguments.vllm_moe_binary.is_file():
        raise ValueError("--vllm-moe-binary must identify the captured runtime library")
    output_dir = arguments.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory exists: {output_dir}")
    if not arguments.nvcc.is_file() or not os.access(arguments.nvcc, os.X_OK):
        raise ValueError(f"nvcc is not executable: {arguments.nvcc}")
    output_dir.mkdir(parents=True)
    binary = output_dir / "moe_routing_atomic_service"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_90",
        str(SOURCE), "-o", str(binary),
    ]
    _run(compile_command)
    campaign_before = _wait_idle(arguments.device_id)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    executions = []
    rejected = []
    for run in range(RUNS):
        for case in CASES:
            for attempt in range(MAXIMUM_ATTEMPTS):
                before = _wait_idle(arguments.device_id)
                command = [str(binary), str(case[0]), str(SAMPLES_PER_RUN)]
                result, clock = _clocked(
                    command, environment, arguments.device_id,
                    output_dir / f"run{run}_case{case[0]}_a{attempt}.clock.csv")
                after = _device(arguments.device_id)
                record = {
                    "run": run, "case_id": case[0], "attempt": attempt,
                    "command": command, "stdout": result.stdout,
                    "stderr": result.stderr, "device_before": before,
                    "device_after": after, "clock": clock,
                }
                if clock["accepted"]:
                    break
                rejected.append(record)
            else:
                raise RuntimeError(f"clock gate failed 3 times: {case}")
            rows.extend(_parse(result.stdout, case, run, before, after))
            executions.append(record)
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
        "schema": "h200_moe_routing_atomic_service_capture_v1",
        "baseline_commit": arguments.baseline,
        "source_path": "microbench/nvidia_h200/operators/fused_moe/"
                       "moe_routing_atomic_service.cu",
        "source_sha256": _sha256(SOURCE),
        "runner_path": "microbench/nvidia_h200/operators/fused_moe/"
                       "moe_routing_atomic_service_runner.py",
        "raw_sha256": _sha256(raw_path),
        "clock_gate": (
            "per execution: no 50-ms sample below 1980 MHz between the first "
            "and last boosted sample, and every busy sample >= 1980 MHz"),
        "accepted_executions": len(executions),
        "rejected_executions": len(rejected),
        "execution_clocks": [
            {"run": e["run"], "case_id": e["case_id"],
             "attempt": e["attempt"], "clock": e["clock"]}
            for e in executions + rejected],
        "runner_sha256": _sha256(RUNNER), "binary_sha256": _sha256(binary),
        "vllm_moe_binary": str(arguments.vllm_moe_binary),
        "vllm_moe_binary_sha256": _sha256(arguments.vllm_moe_binary),
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
