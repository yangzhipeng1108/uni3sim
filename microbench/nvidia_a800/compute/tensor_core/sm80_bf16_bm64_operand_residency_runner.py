#!/usr/bin/env python3
"""Capture A800 BM64 two-slot hit and grouped-weight miss services."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name(
    "sm80_bf16_bm64_operand_residency.cu")
RUNNER = Path(__file__)
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_a800"
ALLOWED_DEVICES = (6, 7)
DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
NVCC = Path("/usr/local/cuda-12.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-12.2/bin/cuobjdump")
TRIAL_LIMIT = 0.10
POST_RUN_IDLE_TIMEOUT_S = 15.0
POST_RUN_IDLE_POLL_S = 0.25
LOGICAL_BYTES = 1_132_462_080
EXPECTED_CASES = (
    "l2_resident",
    "dram_stream",
    "grouped_fanout_1_n4096",
    "grouped_fanout_1_n7168",
    "grouped_fanout_8_n6144",
    "source_group_multiwave_k2048_n7168_fanout1",
    "source_group_multiwave_k2048_n7168_fanout2",
    "source_group_multiwave_k2048_n7168_fanout4",
    "source_group_multiwave_k2048_n7168",
    "source_group_multiwave_k2880_n5760_fanout1",
    "source_group_multiwave_k2880_n5760_fanout2",
    "source_group_multiwave_k2880_n5760_fanout4",
    "source_group_multiwave_k2880_n5760",
)
EXPECTED_LOGICAL_BYTES = {
    "l2_resident": LOGICAL_BYTES,
    "dram_stream": LOGICAL_BYTES,
    "grouped_fanout_1_n4096": LOGICAL_BYTES,
    "grouped_fanout_1_n7168": LOGICAL_BYTES,
    "grouped_fanout_8_n6144": LOGICAL_BYTES,
    "source_group_multiwave_k2048_n7168_fanout1": 469_762_048,
    "source_group_multiwave_k2048_n7168_fanout2": 469_762_048,
    "source_group_multiwave_k2048_n7168_fanout4": 469_762_048,
    "source_group_multiwave_k2048_n7168": 469_762_048,
    "source_group_multiwave_k2880_n5760_fanout1": 530_841_600,
    "source_group_multiwave_k2880_n5760_fanout2": 530_841_600,
    "source_group_multiwave_k2880_n5760_fanout4": 530_841_600,
    "source_group_multiwave_k2880_n5760": 530_841_600,
}
SOURCE_GROUP_CASES_BY_WEIGHT_STREAMS = {
    1: (
        "source_group_multiwave_k2048_n7168",
        "source_group_multiwave_k2880_n5760",
    ),
    2: (
        "source_group_multiwave_k2048_n7168_fanout4",
        "source_group_multiwave_k2880_n5760_fanout4",
    ),
    4: (
        "source_group_multiwave_k2048_n7168_fanout2",
        "source_group_multiwave_k2880_n5760_fanout2",
    ),
    8: (
        "source_group_multiwave_k2048_n7168_fanout1",
        "source_group_multiwave_k2880_n5760_fanout1",
    ),
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command, *, environment=None):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment)


def _device(device):
    if device not in ALLOWED_DEVICES:
        raise ValueError(f"device must be one of {ALLOWED_DEVICES}")
    result = _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.used,"
        "utilization.gpu,utilization.memory,clocks.current.sm,"
        "clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ])
    fields = tuple(value.strip() for value in result.stdout.split(","))
    if len(fields) != 10:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
        "pci_bus_id": fields[3], "memory_used_mib": int(fields[4]),
        "gpu_utilization_pct": int(fields[5]),
        "memory_utilization_pct": int(fields[6]),
        "sm_clock_mhz": int(fields[7]),
        "memory_clock_mhz": int(fields[8]), "driver_version": fields[9],
    }
    if row["index"] != device or row["name"] != DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    applications = _capture([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if any(
            line.strip() and line.split(",", 1)[0].strip() == row["uuid"]
            for line in applications.splitlines()):
        raise RuntimeError("target GPU has a compute process")
    if (row["memory_used_mib"] > 8 or row["gpu_utilization_pct"] != 0
            or row["memory_utilization_pct"] != 0):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _wait_for_post_run_idle(device):
    deadline = time.monotonic() + POST_RUN_IDLE_TIMEOUT_S
    while True:
        try:
            return _device(device)
        except RuntimeError as error:
            if not str(error).startswith("target GPU is not idle:"):
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "target GPU did not report idle after the probe") from error
            time.sleep(POST_RUN_IDLE_POLL_S)


def _parse(stdout, trials):
    attributes = None
    pipeline_attributes = None
    rows = []
    for line in stdout.splitlines():
        fields = next(csv.reader([line]))
        if fields[0] == "ATTRIBUTE":
            if len(fields) != 7 or attributes is not None:
                raise ValueError("invalid operand attribute row")
            attributes = tuple(int(value) for value in fields[1:])
        elif fields[0] == "PIPELINE_ATTRIBUTE":
            if len(fields) != 7 or pipeline_attributes is not None:
                raise ValueError("invalid pipeline attribute row")
            pipeline_attributes = tuple(int(value) for value in fields[1:])
        elif fields[:2] in (
                ["DATA", "trial"], ["PIPELINE_DATA", "trial"]):
            continue
        elif fields[0] in ("DATA", "PIPELINE_DATA"):
            if len(fields) != 7:
                raise ValueError("invalid measurement row")
            trial = int(fields[1])
            residency = fields[2]
            backing_bytes = int(fields[3])
            logical_bytes = int(fields[4])
            duration_ms = float(fields[5])
            checksum = float(fields[6])
            if (trial not in range(trials)
                    or residency not in EXPECTED_CASES
                    or backing_bytes <= 0
                    or logical_bytes != EXPECTED_LOGICAL_BYTES[residency]
                    or duration_ms <= 0 or not math.isfinite(checksum)):
                raise ValueError(f"measurement violates contract: {fields}")
            rows.append({
                "trial": trial, "residency": residency,
                "backing_bytes": backing_bytes,
                "logical_bytes": logical_bytes,
                "duration_ms": duration_ms,
                "checksum": checksum,
            })
        elif line:
            raise ValueError(f"unexpected probe output: {line!r}")
    if attributes != (86, 0, 0, 128, 80, 5):
        raise ValueError(f"operand resource envelope changed: {attributes}")
    if pipeline_attributes != (93, 0, 0, 128, 80, 5):
        raise ValueError(
            f"pipeline resource envelope changed: {pipeline_attributes}")
    expected = len(EXPECTED_CASES) * trials
    if len(rows) != expected or len({
            (row["trial"], row["residency"]) for row in rows}) != expected:
        raise ValueError("probe matrix is incomplete or duplicated")
    return attributes, pipeline_attributes, rows


def _relative_range(values):
    median = statistics.median(values)
    return (max(values) - min(values)) / median


def _report(rows, device):
    cases = {}
    for name in EXPECTED_CASES:
        selected = [row for row in rows if row["residency"] == name]
        durations = [row["duration_ms"] for row in selected]
        median_ms = statistics.median(durations)
        numerator = selected[0]["logical_bytes"]
        byte_meaning = "combined_a_plus_b_logical_bytes"
        if name.startswith("grouped_"):
            numerator //= 2
            byte_meaning = "strided_b_weight_bytes"
        capacity = numerator / (median_ms * 1.0e-3)
        spread = _relative_range(durations)
        cases[name] = {
            "median_duration_ms": median_ms,
            "trial_duration_ms": durations,
            "relative_range": spread,
            "stable": spread <= TRIAL_LIMIT,
            "service_bytes": numerator,
            "service_byte_meaning": byte_meaning,
            "logical_byte_capacity_per_s": capacity,
        }
    if not all(case["stable"] for case in cases.values()):
        raise ValueError("probe stability gate failed")
    miss_names = (
        "grouped_fanout_1_n4096", "grouped_fanout_1_n7168")
    miss_capacity = min(
        cases[name]["logical_byte_capacity_per_s"] for name in miss_names)
    source_group_capacity_by_weight_streams = {
        str(streams): min(
            cases[name]["logical_byte_capacity_per_s"]
            for name in names)
        for streams, names in SOURCE_GROUP_CASES_BY_WEIGHT_STREAMS.items()
    }
    return {
        "schema": "nvidia_a800_bm64_operand_residency_service_v1",
        "production_or_evaluation_duration_consumed": False,
        "device": device,
        "scope": {
            "tile_m": 64, "tile_n": 64, "tile_k": 32,
            "group_size_m": 8, "threads_per_cta": 128,
            "resident_ctas_per_sm": 5, "sm_count": 108,
            "pipeline_slots": 2, "dynamic_shared_bytes": 16384,
            "rounds": 256, "grid_ctas": 540,
        },
        "cases": cases,
        "candidates": {
            "grouped_operand_l2_hit_bytes_per_s":
                cases["l2_resident"]["logical_byte_capacity_per_s"],
            "grouped_weight_dram_miss_bytes_per_s": miss_capacity,
            "complete_source_group_pipeline_bytes_per_s": (
                source_group_capacity_by_weight_streams["1"]),
            "complete_source_group_pipeline_bytes_per_s_by_weight_streams": (
                source_group_capacity_by_weight_streams),
            "dram_miss_rule": (
                "minimum of the independently measured N=4096 and N=7168 "
                "source-stride services; no production duration selected it"),
            "source_group_rule": (
                "for each exact 1/2/4/8 weight-stream topology, minimum of "
                "the static K2048/N7168 and K2880/N5760 complete-GSM8 "
                "source-group services"),
        },
        "allowed_use": (
            "BM64/BN64/BK32 BF16 grouped operand L2-hit and strided-weight "
            "DRAM-miss service under the five-CTA, GSM8 envelope"),
        "forbidden_use": (
            "production duration, per-shape residual, or non-BM64 launch"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    if args.trials < 3 or args.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    output = args.output_dir.resolve()
    if output == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output-dir must be an exclusive A800 profiler child")
    if output.exists():
        raise FileExistsError(output)
    device_before = _device(args.device)
    output.mkdir(parents=True)
    binary = output / "sm80_bf16_bm64_operand_residency"
    compile_command = [
        str(NVCC), "-std=c++17", "-O3", "--use_fast_math",
        "--generate-code=arch=compute_80,code=sm_80", "--disable-warnings",
        "-o", str(binary), str(SOURCE),
    ]
    _capture(compile_command)
    sass = _capture([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    if ("LDGSTS" not in sass or "HMMA.16816.F32.BF16" not in sass
            or "LDL" in sass or "STL" in sass):
        raise ValueError("compiled SASS violates the tile pipeline contract")
    sass_path = output / "probe.sass.txt"
    sass_path.write_text(sass, encoding="utf-8")
    environment = dict(os.environ)
    for name in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if name in environment:
            raise RuntimeError(f"{name} must be unset")
    command = [str(binary), str(args.device), str(args.trials)]
    started = time.time()
    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr)
    if stderr.strip():
        raise RuntimeError(f"probe wrote stderr: {stderr.strip()}")
    completed = time.time()
    attributes, pipeline_attributes, rows = _parse(
        stdout, args.trials)
    raw_path = output / "raw.csv"
    raw_path.write_text(stdout, encoding="utf-8")
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(_report(rows, device_before), indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
    device_after = _wait_for_post_run_idle(args.device)
    manifest = {
        "schema": "nvidia_a800_bm64_operand_residency_capture_v1",
        "hardware_task": {
            "hardware": "nvidia_a800", "device": args.device,
            "owner": "kernel-accuracy", "baseline_commit":
                _capture(["git", "rev-parse", "HEAD"]).stdout.strip(),
            "command": command, "pid": process.pid,
            "started_epoch_s": started, "completed_epoch_s": completed,
            "output": str(output),
        },
        "device_before": device_before, "device_after": device_after,
        "compile_command": compile_command,
        "resource_contract": {
            "operand": attributes, "pipeline": pipeline_attributes},
        "source": str(SOURCE.relative_to(ROOT)),
        "runner": str(RUNNER.relative_to(ROOT)),
        "sha256": {
            "source": _sha256(SOURCE), "runner": _sha256(RUNNER),
            "binary": _sha256(binary), "sass": _sha256(sass_path),
            "raw": _sha256(raw_path), "report": _sha256(report_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
