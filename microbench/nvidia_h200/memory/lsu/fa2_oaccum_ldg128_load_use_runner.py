#!/usr/bin/env python3
"""Capture the independent LDG128-to-FMA readiness mechanism on H200.

H200 owner copy of microbench/nvidia_a800/memory/lsu/
fa2_oaccum_ldg128_load_use_runner.py.  Modes, tiers, repeats, samples,
trials, rotation, CSV schema and the non-NCU clock64 timing path are
unchanged.  Hardware-coupling changes only:
  - device contract "NVIDIA A800-SXM4-80GB" -> "NVIDIA H200", -arch=sm_80 ->
    -arch=sm_90 (same CUDA 12.2 nvcc/cuobjdump, plus -cudart static);
  - grids 1/32/108 -> 1/32/132; valid %smid bound 108 -> 132;
  - output root profiler_output/microbench/nvidia_h200;
  - idle contract: memory.used <= 8 MiB (H200 driver reports 4 MiB idle);
  - ``build`` (local: compile, ptxas spill check, SASS audit) and
    ``capture`` (remote H200) stages, explicit baseline commit;
  - Nsight Compute is not installed on the H200 host, so the NCU cache-tier
    traffic audit (a validation channel, never a parameter source) is
    recorded as blocked in ncu/audit.json.  The clock64 timing path that
    defines the parameter does not use NCU and runs unchanged;
  - 50 ms ClockGate (unlocked SM clock) with bounded re-runs of rejected
    cells.
"""

import argparse
import atexit
import datetime
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


HARDWARE_ID = "nvidia_h200"
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_oaccum_ldg128_load_use.cu")
REPORT_PATH = Path(__file__).with_name("fa2_oaccum_ldg128_load_use_report.py")
OUTPUT_ROOT = (
    REPOSITORY_ROOT / "profiler_output" / "microbench" / HARDWARE_ID)
EXPECTED_DEVICE_NAME = "NVIDIA H200"
EXPECTED_SM_COUNT = 132
IDLE_MEMORY_MIB = 8
THREADS_PER_CTA = 128
GRIDS = (1, 32, 132)
REPEATS = (1, 2, 3, 4, 6, 8, 11, 16, 22)
MODES = ("empty", "fma_only", "ldg128_fma")
TIERS = ("l2_warm_cg", "dram_cg")
TRIALS = 3
SAMPLES = 9
# A800 NCU audit request, recorded in the blocked H200 audit record.
NCU_METRICS = (
    "smsp__sass_inst_executed_op_global_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "lts__t_sectors_aperture_device_op_read.sum",
    "dram__sectors_read.sum",
)


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
            row["memory_used_mib"] > IDLE_MEMORY_MIB
            or row["utilization_gpu_pct"] != 0):
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


def _validated_output_dir(arguments):
    output_dir = arguments.output_dir.resolve()
    root = OUTPUT_ROOT.resolve()
    if root not in output_dir.parents:
        raise ValueError("output directory must be below the H200 microbench root")
    return output_dir


def _build(arguments):
    repo_root = arguments.repo_root.resolve()
    output_dir = _validated_output_dir(arguments)
    if not SOURCE_PATH.is_file() or not REPORT_PATH.is_file():
        raise FileNotFoundError("mechanism source or report does not exist")
    for path in (arguments.nvcc, arguments.cuobjdump):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    build_dir = output_dir / "build"
    snapshot_dir = output_dir / "source_snapshot"
    for path in (build_dir, snapshot_dir):
        path.mkdir()
    for path in (SOURCE_PATH, Path(__file__).resolve(), REPORT_PATH):
        shutil.copy2(path, snapshot_dir / path.name)
    binary = build_dir / "fa2_oaccum_ldg128_load_use"
    compile_command = [
        str(arguments.nvcc), "-O3", "-std=c++17", "-arch=sm_90",
        "-cudart", "static", "-Xptxas=-v",
        str(snapshot_dir / SOURCE_PATH.name), "-o", str(binary),
    ]
    _run(compile_command, cwd=repo_root,
         stdout_path=build_dir / "compile.stdout",
         stderr_path=build_dir / "ptxas.stderr")
    _validate_ptxas((build_dir / "ptxas.stderr").read_text(encoding="utf-8"))
    sass_path = build_dir / "fa2_oaccum_ldg128_load_use.sass"
    _run([str(arguments.cuobjdump), "--dump-sass", str(binary)], cwd=repo_root,
         stdout_path=sass_path, stderr_path=build_dir / "cuobjdump.stderr")
    _write_json(build_dir / "sass_audit.json", _validate_sass(
        sass_path.read_text(encoding="utf-8")))
    _write_json(build_dir / "build_manifest.json", {
        "compile_command": compile_command,
        "nvcc_version": _run(
            [str(arguments.nvcc), "--version"], cwd=repo_root).stdout.strip(),
        "source_sha256": _sha256(snapshot_dir / SOURCE_PATH.name),
        "runner_sha256": _sha256(snapshot_dir / Path(__file__).name),
        "report_sha256": _sha256(snapshot_dir / REPORT_PATH.name),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(sass_path),
    })


def _capture(arguments):
    repo_root = arguments.repo_root.resolve()
    output_dir = _validated_output_dir(arguments)
    build_dir = output_dir / "build"
    snapshot_dir = output_dir / "source_snapshot"
    native_dir = output_dir / "native"
    ncu_dir = output_dir / "ncu"
    build_manifest = json.loads(
        (build_dir / "build_manifest.json").read_text(encoding="utf-8"))
    binary = build_dir / "fa2_oaccum_ldg128_load_use"
    sass_path = build_dir / "fa2_oaccum_ldg128_load_use.sass"
    if (_sha256(binary) != build_manifest["binary_sha256"]
            or _sha256(snapshot_dir / SOURCE_PATH.name)
            != build_manifest["source_sha256"]
            or _sha256(SOURCE_PATH) != build_manifest["source_sha256"]):
        raise ValueError("build artifacts differ from the build manifest")
    if native_dir.exists():
        raise FileExistsError(native_dir)
    if not arguments.nvidia_smi.is_file():
        raise FileNotFoundError(arguments.nvidia_smi)
    preflight = _device_row(
        arguments.nvidia_smi, arguments.device_id, repo_root, require_idle=True)
    native_dir.mkdir()
    ncu_dir.mkdir()
    _write_json(output_dir / "capture_identity.json", {
        "schema": "h200_fa2_oaccum_ldg128_load_use_capture_v1",
        "baseline_commit": arguments.baseline_commit,
        "run_id": arguments.run_id,
        "hardware_id": HARDWARE_ID,
        "device": preflight,
        "command": sys.argv,
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(arguments.device_id)
    schedule = _schedule()
    ordered_keys = []
    for trial in range(TRIALS):
        offset = (trial * 37) % len(schedule)
        ordered_keys.extend(
            (trial, *cell) for cell in schedule[offset:] + schedule[:offset])
    gate = ClockGate(arguments.device_id, output_dir / "clock_log.csv")
    results = {}
    passes = []
    pending = list(ordered_keys)
    for pass_index in range(CLOCK_MAX_PASSES):
        windows = {}
        for key in pending:
            trial, tier, mode, grid, repeats = key
            start = time.time()
            before = _device_row(
                arguments.nvidia_smi, arguments.device_id, repo_root,
                require_idle=False)
            output = _run([
                str(binary), mode, str(grid), str(repeats), str(SAMPLES), tier,
            ], cwd=repo_root, env=environment)
            after = _device_row(
                arguments.nvidia_smi, arguments.device_id, repo_root,
                require_idle=False)
            windows[key] = (start, time.time())
            if before["uuid"] != after["uuid"]:
                raise RuntimeError("GPU identity changed during a mechanism cell")
            results[key] = _parse_binary_csv(
                output.stdout, trial=trial,
                expected_cell=(tier, mode, grid, repeats),
                before=before, after=after)
        verdicts = gate.judge(windows)
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
    clock = _clock_summary(gate, {})
    gate.stop()
    if pending:
        raise RuntimeError(f"{len(pending)} cells never passed the clock gate")
    rows = [row for key in ordered_keys for row in results[key]]
    raw_path = native_dir / "raw.csv"
    fields = tuple(rows[0])
    with raw_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(ncu_dir / "audit.json", {
        "status": "blocked",
        "reason": (
            'Nsight Compute (ncu) is not installed on the recorded capture host; the A800 NCU cache-tier traffic audit cannot be re-executed.  It is a validation channel only and never a parameter source.'),
        "requested_metrics": list(NCU_METRICS),
        "requested_profile": "ldg128_fma grid132 repeats22 per tier",
        "alternative": (
            "cache-tier separation is checked from the same clock64 data: "
            "the dram_cg first-load readiness must exceed the l2_warm_cg "
            "first-load readiness (report gate)"),
    })
    postflight = _device_row(
        arguments.nvidia_smi, arguments.device_id, repo_root, require_idle=True)
    if postflight["uuid"] != preflight["uuid"]:
        raise RuntimeError("GPU identity changed after capture")
    _write_json(output_dir / "manifest.json", {
        "schema": "h200_fa2_oaccum_ldg128_load_use_capture_v1",
        "baseline_commit": arguments.baseline_commit,
        "source_sha256": build_manifest["source_sha256"],
        "runner_sha256": build_manifest["runner_sha256"],
        "report_sha256": build_manifest["report_sha256"],
        "binary_sha256": build_manifest["binary_sha256"],
        "sass_sha256": build_manifest["sass_sha256"],
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
        "clock_gate": {**clock, "passes": passes,
                       "accepted_cells": len(ordered_keys),
                       "rejected_cell_runs": sum(
                           item["rejected"] for item in passes)},
    })
    _write_json(output_dir / "postflight_device.json", postflight)
    _write_sha_manifest(output_dir)


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "capture"), nargs="?")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    parser.add_argument(
        "--nvidia-smi", type=Path, default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument("--print-schedule", action="store_true")
    arguments = parser.parse_args(argv)
    arguments.run_id = _require_text(arguments.run_id, "run_id")
    if arguments.device_id < 0:
        raise ValueError("device-id must be non-negative")
    if not arguments.print_schedule and (
            arguments.output_dir is None or arguments.stage is None):
        raise ValueError("stage and output-dir are required")
    if arguments.stage == "capture" and arguments.baseline_commit is None:
        raise ValueError("capture requires --baseline-commit")
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
    if arguments.stage == "build":
        _build(arguments)
    else:
        _capture(arguments)


if __name__ == "__main__":
    main()
