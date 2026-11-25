#!/usr/bin/env python3
"""Build, capture and publish the H200 staged LDGSTS L2 service probe.

Owner copy of microbench/nvidia_a800/memory/l2/staged_ldgsts_l2_runner.py.
The capture contract (seven event-timed trials, 10% stability gate, median
trial as the capture capacity, minimum over two independent captures as the
published service) is unchanged.  Hardware coupling changes only:

* the probe runs on the H200 host with CUDA 12.9 nvcc (sm_90) and the
  cuobjdump shipped with the remote Triton wheel;
* the only allowed device is the owned GPU 2, exposed as
  CUDA_VISIBLE_DEVICES=2 (the probe then addresses device 0);
* the A800 "two idle devices" become two independent capture processes on
  GPU 2 (``--run-id`` r1/r2);
* the SM clock cannot be locked in this container, so every probe process
  runs under a 50-ms nvidia-smi SM-clock log; a process whose busy samples
  fall below 1980 MHz (or that shows no busy sample) is rejected and re-run,
  and accepted/rejected counts are recorded;
* per-clock values use the measured 1.98-GHz H200 SM clock, never 1.14 GHz;
* the grid is 132 (one CTA per H200 SM).

This module also holds the small capture helpers that the other H200 WP3
LDGSTS runners import (the A800 runners likewise import this module).

Usage on the H200 host:
    python3 staged_ldgsts_l2_runner.py capture --device 2 \
        --output-dir <repo>/profiler_output/microbench/nvidia_h200/<run> \
        --source-commit <sha>
Locally, after copying the small capture files back:
    python3 staged_ldgsts_l2_runner.py publish --capture <dir1> --capture <dir2>
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import shutil
import time


ROOT = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("staged_ldgsts_l2.cu")
RUNNER = Path(__file__).resolve()
OUTPUT_ROOT = ROOT / "profiler_output/microbench/nvidia_h200"
REFERENCE_ROOT = ROOT / "microbench/nvidia_h200/references"
NVCC = Path(shutil.which("nvcc") or "nvcc")
CUOBJDUMP = Path(shutil.which("cuobjdump") or "cuobjdump")
DEVICE_NAME = "NVIDIA H200"
ALLOWED_DEVICES = (2,)
BOOST_SM_CLOCK_MHZ = 1980
SM_CLOCK_HZ = 1.98e9
SM_COUNT = 132
MAXIMUM_CLOCK_ATTEMPTS = 5
GRID = 132
LOGICAL_BYTES = GRID * 4096 * 16384


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(command, *, environment=None, timeout=3600):
    return subprocess.run(
        command, check=True, text=True, capture_output=True,
        env=environment, timeout=timeout)


def _device(device):
    if device not in ALLOWED_DEVICES:
        raise ValueError(f"device must be one of {ALLOWED_DEVICES}")
    result = _capture([
        "nvidia-smi", f"--id={device}",
        "--query-gpu=index,name,uuid,memory.used,utilization.gpu,"
        "clocks.current.sm,clocks.current.memory,driver_version",
        "--format=csv,noheader,nounits",
    ])
    values = tuple(item.strip() for item in result.stdout.split(","))
    if len(values) != 8:
        raise ValueError("unexpected nvidia-smi device row")
    row = {
        "index": int(values[0]), "name": values[1], "uuid": values[2],
        "memory_used_mib": int(values[3]),
        "gpu_utilization_pct": int(values[4]),
        "sm_clock_mhz": int(values[5]), "memory_clock_mhz": int(values[6]),
        "driver_version": values[7],
    }
    applications = _capture([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ]).stdout
    if (row["index"] != device or row["name"] != DEVICE_NAME
            or row["memory_used_mib"] > 64
            or row["gpu_utilization_pct"] != 0
            or any(line.strip()
                   and line.split(",", 1)[0].strip() == row["uuid"]
                   for line in applications.splitlines())):
        raise RuntimeError(f"target GPU is not idle: {row}")
    return row


def _wait_for_idle(device):
    deadline = time.monotonic() + 30.0
    while True:
        try:
            return _device(device)
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "target GPU did not return idle after probe") from error
            time.sleep(0.5)


def _probe_environment(device):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(device)
    if "CUDA_DEVICE_ORDER" in environment:
        del environment["CUDA_DEVICE_ORDER"]
    return environment


def _new_output(path):
    output = Path(path).resolve()
    if output == OUTPUT_ROOT.resolve() or OUTPUT_ROOT.resolve() not in output.parents:
        raise ValueError("output-dir must be an exclusive H200 profiler child")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    return output


def _compile(source, binary, *extra):
    command = [
        str(NVCC), "-std=c++17", "-O3", "-arch=sm_90",
        "--ptxas-options=-v", *extra, "-o", str(binary), str(source),
    ]
    result = _capture(command)
    return command, result.stderr


def _dump_sass(binary, path):
    sass = _capture([str(CUOBJDUMP), "--dump-sass", str(binary)]).stdout
    Path(path).write_text(sass, encoding="utf-8")
    return sass


def _clock_stats(log_path):
    samples = []
    for line in Path(log_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clock_text, utilization_text = (item.strip() for item in line.split(","))
        samples.append((int(clock_text), int(utilization_text)))
    busy = [clock for clock, utilization in samples if utilization > 0]
    return {
        "samples": len(samples),
        "busy_samples": len(busy),
        "busy_sm_clock_min_mhz": min(busy) if busy else None,
        "busy_sm_clock_max_mhz": max(busy) if busy else None,
        "busy_sm_clock_median_mhz": statistics.median(busy) if busy else None,
        "accepted": bool(busy) and min(busy) >= BOOST_SM_CLOCK_MHZ,
    }


def _gated_run(command, *, device, log_stem, timeout=3600):
    """Run one probe process under a 50-ms SM-clock log; retry on rejection."""

    attempts = []
    for attempt in range(MAXIMUM_CLOCK_ATTEMPTS):
        log_path = Path(f"{log_stem}.a{attempt}.clock.csv")
        with log_path.open("w", encoding="utf-8") as log:
            sampler = subprocess.Popen(
                ["nvidia-smi", "-i", str(device),
                 "--query-gpu=clocks.sm,utilization.gpu",
                 "--format=csv,noheader,nounits", "-lms", "50"],
                stdout=log, stderr=subprocess.DEVNULL)
            try:
                time.sleep(0.2)
                started = time.time()
                result = _capture(
                    command, environment=_probe_environment(device),
                    timeout=timeout)
                completed = time.time()
                time.sleep(0.3)
            finally:
                sampler.terminate()
                sampler.wait()
        stats = _clock_stats(log_path)
        record = {
            "attempt": attempt, "command": command,
            "started_at_unix_s": started, "completed_at_unix_s": completed,
            "clock_log": log_path.name, "clock_log_sha256": _sha256(log_path),
            "clock": stats,
        }
        attempts.append(record)
        if stats["accepted"]:
            return result, attempts
    raise RuntimeError(f"no clock-accepted capture for {command}: {attempts}")


def _clock_summary(attempt_lists):
    accepted = [
        attempt for attempts in attempt_lists for attempt in attempts
        if attempt["clock"]["accepted"]]
    rejected = [
        attempt for attempts in attempt_lists for attempt in attempts
        if not attempt["clock"]["accepted"]]
    return {
        "rule": (
            "every probe process runs under nvidia-smi -lms 50; a process is "
            f"accepted only if it shows a busy sample and every busy sample is "
            f">= {BOOST_SM_CLOCK_MHZ} MHz, otherwise it is rejected and re-run"),
        "accepted_captures": len(accepted),
        "rejected_captures": len(rejected),
        "accepted_busy_sm_clock_min_mhz": min(
            attempt["clock"]["busy_sm_clock_min_mhz"] for attempt in accepted),
        "accepted_busy_sm_clock_max_mhz": max(
            attempt["clock"]["busy_sm_clock_max_mhz"] for attempt in accepted),
        "accepted_busy_samples": sum(
            attempt["clock"]["busy_samples"] for attempt in accepted),
        "rejected_busy_sm_clock_min_mhz": [
            attempt["clock"]["busy_sm_clock_min_mhz"] for attempt in rejected],
    }


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _load_capture(path, candidate_name):
    capture_dir = Path(path).resolve()
    candidate_path = capture_dir / candidate_name
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    return capture_dir, candidate_path, candidate


def _write_reference(name, reference):
    path = REFERENCE_ROOT / name
    path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(path)


def capture(args):
    output = _new_output(args.output_dir)
    device_before = _device(args.device)
    binary = output / "staged_ldgsts_l2"
    compile_command, compiler_stderr = _compile(SOURCE, binary)
    sass = _dump_sass(binary, output / "probe.sass.txt")
    if "LDGSTS" not in sass or "LDL" in sass or "STL" in sass:
        raise ValueError("compiled SASS violates the LDGSTS contract")
    command = [str(binary), "0", str(args.trials)]
    result, attempts = _gated_run(
        command, device=args.device, log_stem=output / "probe")
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != args.trials:
        raise ValueError("probe trial count changed")
    durations = []
    for trial, row in enumerate(rows):
        if (int(row["trial"]) != trial
                or int(row["logical_bytes"]) != LOGICAL_BYTES
                or int(row["resident_ctas_per_sm"]) != 2):
            raise ValueError("probe row violates the frozen contract")
        duration_ms = float(row["duration_ms"])
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError("probe duration must be finite and positive")
        durations.append(duration_ms)
    center = statistics.median(durations)
    spread = (max(durations) - min(durations)) / center
    if spread > 0.10:
        raise ValueError(f"probe stability gate failed: {spread}")
    (output / "raw.csv").write_text(result.stdout, encoding="utf-8")
    device_after = _wait_for_idle(args.device)
    provenance = {
        "schema": "nvidia_h200_staged_ldgsts_l2_service_v1",
        "hardware": "nvidia_h200",
        "owner": "h200-port-wp3_ldgsts",
        "device_before": device_before,
        "device_after": device_after,
        "source_commit": args.source_commit,
        "source_sha256": _sha256(SOURCE),
        "runner_sha256": _sha256(RUNNER),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(output / "probe.sass.txt"),
        "compile_command": compile_command,
        "compiler_stderr": compiler_stderr,
        "attempts": attempts,
        "clock_gate": _clock_summary([attempts]),
        "pid": os.getpid(),
        "trials": args.trials,
        "logical_bytes_per_trial": LOGICAL_BYTES,
        "duration_ms": durations,
        "relative_range": spread,
        "logical_byte_capacity_per_s": LOGICAL_BYTES / (center * 1e-3),
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output / "provenance.json")


def publish(args):
    captures = []
    clock_attempts = []
    commits = set()
    for path in args.capture:
        capture_dir, provenance_path, provenance = _load_capture(
            path, "provenance.json")
        commits.add(provenance["source_commit"])
        clock_attempts.append(provenance["attempts"])
        captures.append({
            "path": _relative(capture_dir),
            "device_index": provenance["device_before"]["index"],
            "device_uuid": provenance["device_before"]["uuid"],
            "provenance_sha256": _sha256(provenance_path),
            "raw_sha256": _sha256(capture_dir / "raw.csv"),
            "sass_sha256": provenance["sass_sha256"],
            "logical_byte_capacity_per_s": provenance[
                "logical_byte_capacity_per_s"],
            "relative_trial_spread": provenance["relative_range"],
        })
    if len(captures) < 2 or len(commits) != 1:
        raise ValueError("publication needs two captures of one source commit")
    selected = min(item["logical_byte_capacity_per_s"] for item in captures)
    reference = {
        "schema": "nvidia_h200_staged_ldgsts_l2_service_reference_v1",
        "hardware_id": "nvidia_h200",
        "device_name": DEVICE_NAME,
        "compute_capability": "sm_90",
        "source_baseline_commit": commits.pop(),
        "source": _relative(SOURCE),
        "runner": _relative(RUNNER),
        "captures": captures,
        "probe_contract": {
            "dtype": "fp16",
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 32,
            "threads_per_cta": 128,
            "pipeline_slots": 5,
            "resident_ctas_per_sm": 2,
            "physical_sm_count": SM_COUNT,
            "grid_ctas": GRID,
            "panel_bytes": 16384,
            "backing_bytes": 33554432,
            "rounds": 4096,
            "load_instruction": "LDGSTS",
            "cache_operator": "cg",
        },
        "selected_service": {
            "selection_rule": (
                "minimum complete-capture capacity across two independent "
                "clock-accepted capture processes on H200 GPU 2"),
            "logical_byte_capacity_per_s": selected,
            "throughput_per_sm_clock": selected / SM_CLOCK_HZ,
            "scope": "full_chip",
        },
        "applicability": {
            "dtype": "fp16",
            "tile_shape": [128, 128, 32],
            "threads_per_cta": 128,
            "required_launch_algorithm": "observed_gemm:gemm",
            "required_operand_streams": 2,
            "required_source_evidence": (
                "five-slot CUTLASS SM80 A/B operand staging with LDGSTS"),
        },
        "decision": {
            "status": "accepted_mechanism_capacity",
            "forward_eligible": True,
            "reason": (
                "The workload-independent staged copy is stable in two "
                "independent clock-gated H200 captures and provides only a "
                "source-class L2 read service ceiling."),
        },
        "excluded_conclusions": [
            "a complete production GEMM duration",
            "a generic L2 capacity replacement",
            "write or read/write-duplex capacity",
            "a per-shape timing correction",
        ],
        "production_or_evaluation_duration_consumed": False,
        "provenance": {
            "source_sha256": _sha256(SOURCE),
            "runner_sha256": _sha256(RUNNER),
            "sm_clock_hz_for_per_clock_values": SM_CLOCK_HZ,
            "clock_gate": _clock_summary(clock_attempts),
            "remote_raw_root": (
                'profiler_output/microbench/nvidia_h200'),
        },
    }
    _write_reference("staged_ldgsts_l2_service_sm90.json", reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--device", type=int, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--source-commit", required=True)
    capture_parser.add_argument("--trials", type=int, default=7)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--capture", type=Path, action="append",
                                required=True)
    args = parser.parse_args()
    if args.action == "capture":
        if args.trials < 3 or args.trials > 11:
            raise ValueError("trials must be in [3, 11]")
        capture(args)
    else:
        publish(args)


if __name__ == "__main__":
    main()
