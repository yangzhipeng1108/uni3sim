#!/usr/bin/env python3
"""Capture the independent A800 LDG128-to-FMA readiness mechanism."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


HARDWARE_ID = "nvidia_a800"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_oaccum_ldg128_load_use.cu")
REPORT_PATH = Path(__file__).with_name("fa2_oaccum_ldg128_load_use_report.py")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / HARDWARE_ID)
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
THREADS_PER_CTA = 128
GRIDS = (1, 32, 108)
REPEATS = (1, 2, 3, 4, 6, 8, 11, 16, 22)
MODES = ("empty", "fma_only", "ldg128_fma")
TIERS = ("l2_warm_cg", "dram_cg")
TRIALS = 3
SAMPLES = 9
NCU_METRICS = (
    "smsp__sass_inst_executed_op_global_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "lts__t_sectors_aperture_device_op_read.sum",
    "dram__sectors_read.sum",
)
NCU_METRIC_UNITS = {
    "smsp__sass_inst_executed_op_global_ld.sum": "inst",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "sector",
    "lts__t_sectors_aperture_device_op_read.sum": "sector",
    "dram__sectors_read.sum": "sector",
}


def _require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, env=None, stdout_path=None, stderr_path=None):
    if not isinstance(command, list) or not command:
        raise TypeError("command must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in command):
        raise ValueError("command entries must be non-empty strings")
    opened = []
    stdout = subprocess.PIPE
    stderr = subprocess.PIPE
    if stdout_path is not None:
        stdout = open(stdout_path, "w", encoding="utf-8")
        opened.append(stdout)
    if stderr_path is not None:
        stderr = open(stderr_path, "w", encoding="utf-8")
        opened.append(stderr)
    try:
        return subprocess.run(
            command, cwd=cwd, env=env, text=True, stdout=stdout,
            stderr=stderr, check=True)
    finally:
        for stream in opened:
            stream.close()


def _git_output(repo_root, *arguments):
    return _run(["git", *arguments], cwd=repo_root).stdout.strip()


def _parse_int(text, name):
    normalized = _require_text(text, name).replace(",", "")
    if not re.fullmatch(r"[0-9]+", normalized):
        raise ValueError(f"{name} must be a non-negative integer: {text!r}")
    return int(normalized)


def _device_row(nvidia_smi, device_id, repo_root, *, require_idle):
    result = _run([
        str(nvidia_smi), f"--id={device_id}",
        "--query-gpu=index,name,uuid,pci.bus_id,memory.used,utilization.gpu,"
        "clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ], cwd=repo_root)
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError("nvidia-smi returned an unexpected device row")
    values = tuple(item.strip() for item in rows[0])
    row = {
        "index": _parse_int(values[0], "GPU index"),
        "name": values[1],
        "uuid": _require_text(values[2], "GPU UUID"),
        "pci_bus_id": _require_text(values[3], "GPU PCI bus ID"),
        "memory_used_mib": _parse_int(values[4], "GPU memory used"),
        "utilization_gpu_pct": _parse_int(values[5], "GPU utilization"),
        "sm_clock_mhz": _parse_int(values[6], "GPU SM clock"),
        "memory_clock_mhz": _parse_int(values[7], "GPU memory clock"),
        "driver_version": _require_text(values[8], "GPU driver version"),
    }
    if row["index"] != device_id or row["name"] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(f"unexpected target GPU: {row}")
    if row["sm_clock_mhz"] <= 0 or row["memory_clock_mhz"] <= 0:
        raise RuntimeError(f"GPU clock is unavailable: {row}")
    applications = _run([
        str(nvidia_smi), "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ], cwd=repo_root).stdout
    for line in applications.splitlines():
        values = next(csv.reader([line], skipinitialspace=True))
        if values and values[0].strip() == row["uuid"]:
            raise RuntimeError(f"target GPU has a compute process: {line}")
    if require_idle and (
            row["memory_used_mib"] != 0 or row["utilization_gpu_pct"] != 0):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _binary_kernel_name(mode, tier):
    if mode == "ldg128_fma":
        return f"fa2_oaccum_ldg128_fma_{tier}"
    if mode == "fma_only":
        return "fa2_oaccum_fma_only"
    if mode == "empty":
        return "fa2_oaccum_empty"
    raise ValueError(f"unknown mode {mode!r}")


def _schedule():
    return tuple(
        (tier, mode, grid, repeats)
        for tier in TIERS
        for mode in MODES
        for grid in GRIDS
        for repeats in REPEATS)


def _parse_binary_csv(text, *, trial, expected_cell, before, after):
    tier, mode, grid, repeats = expected_cell
    reader = csv.DictReader(io.StringIO(text))
    expected_fields = (
        "sample", "mode", "tier", "kernel", "grid", "repeats",
        "block", "sm_id", "cycles", "checksum", "event_ns")
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise ValueError("native benchmark CSV schema changed")
    rows = []
    identities = set()
    sm_ids_by_sample = {}
    for ordinal, raw in enumerate(reader, start=2):
        row = {
            "trial": trial,
            "tier": raw["tier"],
            "mode": raw["mode"],
            "kernel": raw["kernel"],
            "grid": _parse_int(raw["grid"], "native grid"),
            "repeats": _parse_int(raw["repeats"], "native repeats"),
            "sample": _parse_int(raw["sample"], "native sample"),
            "block": _parse_int(raw["block"], "native block"),
            "sm_id": _parse_int(raw["sm_id"], "native SM ID"),
            "cycles": _parse_int(raw["cycles"], "native cycles"),
            "checksum": _parse_int(raw["checksum"], "native checksum"),
            "event_ns": float(raw["event_ns"]),
            "sm_clock_before_mhz": before["sm_clock_mhz"],
            "sm_clock_after_mhz": after["sm_clock_mhz"],
        }
        if (row["tier"], row["mode"], row["grid"], row["repeats"]) != (
                tier, mode, grid, repeats):
            raise ValueError(f"native row {ordinal} disagrees with its cell")
        if row["kernel"] != _binary_kernel_name(mode, tier):
            raise ValueError(f"native row {ordinal} has an unexpected kernel")
        if not (0 <= row["sample"] < SAMPLES and 0 <= row["block"] < grid):
            raise ValueError(f"native row {ordinal} index is outside matrix")
        if row["sm_id"] >= EXPECTED_SM_COUNT or row["cycles"] <= 0:
            raise ValueError(f"native row {ordinal} has an invalid clock64 result")
        if not math.isfinite(row["event_ns"]) or row["event_ns"] <= 0:
            raise ValueError(f"native row {ordinal} has an invalid CUDA event")
        identity = (row["sample"], row["block"])
        if identity in identities:
            raise ValueError(f"native row {ordinal} repeats CTA result {identity}")
        identities.add(identity)
        if row["sample"] not in sm_ids_by_sample:
            sm_ids_by_sample[row["sample"]] = set()
        if row["sm_id"] in sm_ids_by_sample[row["sample"]]:
            raise ValueError(f"native row {ordinal} has duplicate timed SM ID")
        sm_ids_by_sample[row["sample"]].add(row["sm_id"])
        rows.append(row)
    if len(rows) != SAMPLES * grid:
        raise ValueError("native benchmark emitted an incomplete sample matrix")
    return rows


def _parse_ncu_dimension(text, name):
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        _require_text(text, name))
    if match is None:
        raise ValueError(f"{name} has invalid CUDA dimension syntax")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _parse_ncu_csv(text):
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        fields = tuple(next(csv.reader([line])))
        field_set = set(fields)
        launch_fields = {"Kernel Name", "Grid Size", "Block Size"}
        tall_fields = launch_fields | {
            "Metric Name", "Metric Unit", "Metric Value"}
        wide_fields = launch_fields | set(NCU_METRICS)
        if tall_fields <= field_set or wide_fields <= field_set:
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU CSV has no recognized tall/wide metric header")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)):
        raise ValueError("NCU CSV header is absent or contains duplicates")
    fields = set(reader.fieldnames)
    launch_fields = {"Kernel Name", "Grid Size", "Block Size"}
    tall_fields = launch_fields | {
        "Metric Name", "Metric Unit", "Metric Value"}
    wide_fields = launch_fields | set(NCU_METRICS)
    rows = tuple(reader)
    if not rows:
        raise ValueError("NCU CSV has no metric rows")

    def validate_launch(row):
        grid = _parse_ncu_dimension(row["Grid Size"], "NCU grid size")
        block = _parse_ncu_dimension(row["Block Size"], "NCU block size")
        if grid != (108, 1, 1) or block != (THREADS_PER_CTA, 1, 1):
            raise ValueError("NCU profile launch geometry changed")
        kernel = _require_text(row["Kernel Name"], "NCU kernel name")
        if "fa2_oaccum_ldg128_load_use_kernel" not in kernel:
            raise ValueError("NCU did not select the LDG128 mechanism kernel")
        return kernel

    if wide_fields <= fields:
        unit_rows = tuple(row for row in rows if not row["Kernel Name"])
        data_rows = tuple(row for row in rows if row["Kernel Name"])
        if len(unit_rows) != 1 or len(data_rows) != 1:
            raise ValueError(
                "wide NCU CSV requires exactly one unit row and one data row")
        unit_row = unit_rows[0]
        data_row = data_rows[0]
        kernel_name = validate_launch(data_row)
        metrics = {}
        for metric in NCU_METRICS:
            if unit_row[metric] != NCU_METRIC_UNITS[metric]:
                raise ValueError(f"NCU unit changed for {metric!r}")
            metrics[metric] = _parse_int(data_row[metric], f"NCU {metric}")
    elif tall_fields <= fields:
        metrics = {}
        kernel_name = None
        for raw in rows:
            observed_kernel = validate_launch(raw)
            if kernel_name is None:
                kernel_name = observed_kernel
            elif kernel_name != observed_kernel:
                raise ValueError("NCU metrics span more than one kernel")
            metric = raw["Metric Name"]
            if metric not in NCU_METRIC_UNITS:
                raise ValueError(f"NCU emitted an unknown metric {metric!r}")
            if metric in metrics:
                raise ValueError(f"NCU repeats requested metric {metric!r}")
            if raw["Metric Unit"] != NCU_METRIC_UNITS[metric]:
                raise ValueError(f"NCU unit changed for {metric!r}")
            metrics[metric] = _parse_int(raw["Metric Value"], f"NCU {metric}")
    else:
        raise ValueError("NCU CSV lacks either the tall or wide metric schema")
    if set(metrics) != set(NCU_METRICS):
        raise ValueError(
            "NCU metrics are incomplete: "
            f"missing={sorted(set(NCU_METRICS) - set(metrics))}")
    return {"kernel_name": kernel_name, "metrics": metrics}


def _validate_ncu_audit(tier, audit):
    metrics = audit["metrics"]
    expected_sectors = 108 * 22 * (THREADS_PER_CTA * 16 // 32)
    # Nsight Compute reports this SASS counter per executed warp instruction,
    # while the sector counters below count the four 32-byte sectors produced
    # by every lane's 16-byte LDG.128 request.
    expected_warp_load_instructions = (
        108 * 22 * (THREADS_PER_CTA // 32))
    load_instructions = metrics["smsp__sass_inst_executed_op_global_ld.sum"]
    l1_sectors = metrics["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]
    l2_sectors = metrics["lts__t_sectors_aperture_device_op_read.sum"]
    dram_sectors = metrics["dram__sectors_read.sum"]
    if load_instructions != expected_warp_load_instructions:
        raise ValueError(
            "NCU LDG128 warp-instruction count disagrees with source "
            "geometry")
    if l1_sectors < expected_sectors or l2_sectors < expected_sectors:
        raise ValueError("NCU traffic is below the LDG128 source geometry")
    if tier == "l2_warm_cg":
        if dram_sectors > 64:
            raise ValueError("L2-warm target produced unexpected DRAM traffic")
    elif tier == "dram_cg":
        if dram_sectors < math.floor(expected_sectors * 0.75):
            raise ValueError("DRAM target did not leave the cache hierarchy")
    else:
        raise ValueError(f"unknown NCU cache tier {tier!r}")


def _validate_sass(text):
    vector_loads = re.findall(r"\bLDG(?:\.[A-Z0-9_]+)*\.128\b", text)
    fmas = re.findall(r"\bFFMA\b", text)
    if not vector_loads:
        raise ValueError("SASS contains no 128-bit global load")
    if len(fmas) < 4:
        raise ValueError("SASS contains fewer than four FP32 FMA instructions")
    return {
        "ldg128_instruction_sites": len(vector_loads),
        "ffma_instruction_sites": len(fmas),
        "model_parameter_from_instruction_counts": False,
    }


def _validate_ptxas(text):
    matches = re.findall(r"([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads", text)
    if not matches:
        raise ValueError("ptxas output contains no spill summary")
    if any(int(stores) != 0 or int(loads) != 0 for stores, loads in matches):
        raise ValueError("LDG128 mechanism binary has register spills")


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_sha_manifest(output_dir):
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            entries.append(f"{_sha256(path)}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(entries) + "\n", encoding="utf-8")


def _capture(arguments):
    repo_root = arguments.repo_root.resolve()
    output_dir = arguments.output_dir.resolve()
    if not repo_root.is_dir():
        raise NotADirectoryError(repo_root)
    if not SOURCE_PATH.is_file() or not REPORT_PATH.is_file():
        raise FileNotFoundError("mechanism source or report does not exist")
    for path in (arguments.nvcc, arguments.cuobjdump, arguments.ncu,
                 arguments.nvidia_smi):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    root = OUTPUT_ROOT.resolve()
    if root not in output_dir.parents:
        raise ValueError("output directory must be below the A800 microbench root")
    preflight = _device_row(
        arguments.nvidia_smi, arguments.device_id, repo_root, require_idle=True)
    baseline = _git_output(repo_root, "rev-parse", "HEAD")
    output_dir.mkdir(parents=True)
    build_dir = output_dir / "build"
    native_dir = output_dir / "native"
    ncu_dir = output_dir / "ncu"
    snapshot_dir = output_dir / "source_snapshot"
    for path in (build_dir, native_dir, ncu_dir, snapshot_dir):
        path.mkdir()
    for path in (SOURCE_PATH, Path(__file__).resolve(), REPORT_PATH):
        shutil.copy2(path, snapshot_dir / path.name)
    _write_json(output_dir / "capture_identity.json", {
        "schema": "a800_fa2_oaccum_ldg128_load_use_capture_v1",
        "baseline_commit": baseline,
        "run_id": arguments.run_id,
        "hardware_id": HARDWARE_ID,
        "device": preflight,
        "command": sys.argv,
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    binary = build_dir / "fa2_oaccum_ldg128_load_use"
    _run([
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_80",
        "-Xptxas=-v", str(snapshot_dir / SOURCE_PATH.name), "-o", str(binary),
    ], cwd=repo_root, stdout_path=build_dir / "compile.stdout",
       stderr_path=build_dir / "ptxas.stderr")
    _validate_ptxas((build_dir / "ptxas.stderr").read_text(encoding="utf-8"))
    sass_path = build_dir / "fa2_oaccum_ldg128_load_use.sass"
    _run([str(arguments.cuobjdump), "--dump-sass", str(binary)], cwd=repo_root,
         stdout_path=sass_path, stderr_path=build_dir / "cuobjdump.stderr")
    _write_json(build_dir / "sass_audit.json", _validate_sass(
        sass_path.read_text(encoding="utf-8")))

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    rows = []
    schedule = _schedule()
    for trial in range(TRIALS):
        offset = (trial * 37) % len(schedule)
        ordered = schedule[offset:] + schedule[:offset]
        for tier, mode, grid, repeats in ordered:
            before = _device_row(
                arguments.nvidia_smi, arguments.device_id, repo_root,
                require_idle=False)
            output = _run([
                str(binary), mode, str(grid), str(repeats), str(SAMPLES), tier,
            ], cwd=repo_root, env=environment)
            after = _device_row(
                arguments.nvidia_smi, arguments.device_id, repo_root,
                require_idle=False)
            if before["uuid"] != after["uuid"]:
                raise RuntimeError("GPU identity changed during a mechanism cell")
            rows.extend(_parse_binary_csv(
                output.stdout, trial=trial, expected_cell=(tier, mode, grid, repeats),
                before=before, after=after))
    raw_path = native_dir / "raw.csv"
    fields = tuple(rows[0])
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_argument = ",".join(NCU_METRICS)
    ncu_audit = {}
    for tier in TIERS:
        name = f"{tier}_grid108_repeat22"
        csv_path = ncu_dir / f"{name}.csv"
        _run([
            str(arguments.ncu), "--kernel-name-base", "demangled",
            "--kernel-name", "regex:.*fa2_oaccum_ldg128_load_use_kernel.*",
            # Kernel replay invalidates the cache state established by the
            # immediately preceding warm_l2 launch on later metric passes.
            # Application replay reruns that source-defined warmup sequence
            # before every pass, preserving the cache-tier contract.
            "--replay-mode", "application",
            "--cache-control", "none",
            "--launch-skip", "5", "--launch-count", "1", "--metrics",
            metric_argument, "--page", "raw", "--csv", "--print-units",
            "base", "--log-file", str(csv_path), "--force-overwrite",
            str(binary), "ldg128_fma", "108", "22", "1", tier,
        ], cwd=repo_root, env=environment,
           stdout_path=ncu_dir / f"{name}.stdout",
           stderr_path=ncu_dir / f"{name}.stderr")
        parsed = _parse_ncu_csv(csv_path.read_text(encoding="utf-8"))
        _validate_ncu_audit(tier, parsed)
        ncu_audit[tier] = parsed
    _write_json(ncu_dir / "audit.json", ncu_audit)

    postflight = _device_row(
        arguments.nvidia_smi, arguments.device_id, repo_root, require_idle=True)
    if postflight["uuid"] != preflight["uuid"]:
        raise RuntimeError("GPU identity changed after capture")
    _write_json(output_dir / "manifest.json", {
        "schema": "a800_fa2_oaccum_ldg128_load_use_capture_v1",
        "baseline_commit": baseline,
        "source_sha256": _sha256(snapshot_dir / SOURCE_PATH.name),
        "runner_sha256": _sha256(snapshot_dir / Path(__file__).name),
        "report_sha256": _sha256(snapshot_dir / REPORT_PATH.name),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(sass_path),
        "raw_sha256": _sha256(raw_path),
        "ncu_audit_sha256": _sha256(ncu_dir / "audit.json"),
        "device": preflight,
        "grids": list(GRIDS),
        "repeats": list(REPEATS),
        "modes": list(MODES),
        "tiers": list(TIERS),
        "trials": TRIALS,
        "samples_per_cell": SAMPLES,
        "timer_domain": "clock64_same_cta_envelope_cycles",
        "cuda_event_role": "validation_only_not_forward_parameter",
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    _write_json(output_dir / "postflight_device.json", postflight)
    _write_sha_manifest(output_dir)


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    parser.add_argument(
        "--ncu", type=Path, default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument(
        "--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument("--print-schedule", action="store_true")
    arguments = parser.parse_args(argv)
    arguments.run_id = _require_text(arguments.run_id, "run_id")
    if arguments.device_id < 0:
        raise ValueError("device-id must be non-negative")
    if not arguments.print_schedule and arguments.output_dir is None:
        raise ValueError("output-dir is required for a capture")
    return arguments


def main(argv=None):
    arguments = _arguments(argv)
    if arguments.print_schedule:
        print(json.dumps({
            "cells_per_trial": len(_schedule()), "grids": GRIDS,
            "repeats": REPEATS, "modes": MODES, "tiers": TIERS,
            "trials": TRIALS, "samples_per_cell": SAMPLES,
        }, indent=2, sort_keys=True))
        return
    _capture(arguments)


if __name__ == "__main__":
    main()
