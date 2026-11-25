#!/usr/bin/env python3
"""Capture three balanced runs of the independent staged CTA-overlap probe."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
OUTPUT_ROOT = REPOSITORY_ROOT / "profiler_output/microbench/nvidia_a800"
SOURCE_PATH = Path(__file__).with_name("silu_cta_overlap.cu")
RUNNER_PATH = Path(__file__).resolve()
REPORT_PATH = Path(__file__).with_name("silu_cta_overlap_report.py")
NVCC = "/usr/local/cuda-12.2/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda-12.2/bin/cuobjdump"
EXPECTED_DEVICE = "NVIDIA A800-SXM4-80GB"
EXPECTED_COMPUTE_CAPABILITY = "8.0"
EXPECTED_SM_COUNT = 108
MAXIMUM_IDLE_MEMORY_MIB = 8
RUNS = 3
TRIALS = 3
OWNER_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
PROCESS_GROUP_TERM_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_KILL_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_POLL_SECONDS = 0.01
_ACTIVE_PROCESS = None
_PROCESS_GROUP_RESERVATION_FDS = {}
_REGISTRATION_IN_PROGRESS = False
_DEFERRED_SIGNAL = None
_SIGNAL_CLEANUP_IN_PROGRESS = False

REPORT_SPEC = importlib.util.spec_from_file_location(
    "silu_cta_overlap_report", REPORT_PATH)
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(REPORT)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, stdout=None, stderr=None, capture_output=False):
    return subprocess.run(
        [str(item) for item in command], cwd=cwd, check=True, text=True,
        stdout=stdout, stderr=stderr, capture_output=capture_output,
        timeout=600)


def _timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _process_group_exists(process_group):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_group_identity_is_safe(process_group):
    reservation_fd = _PROCESS_GROUP_RESERVATION_FDS[process_group]
    try:
        current = os.stat(f"/proc/{process_group}")
    except FileNotFoundError:
        return True
    reserved = os.fstat(reservation_fd)
    return (current.st_dev, current.st_ino) == (reserved.st_dev, reserved.st_ino)


def _wait_for_process_group_exit(process, process_group, timeout):
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_exists(process_group):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_GROUP_POLL_SECONDS)


def _terminate_process_group(process):
    process_group = process.pid
    if not _process_group_exists(process_group):
        process.wait()
        return
    if not _process_group_identity_is_safe(process_group):
        raise RuntimeError(
            f"refusing to signal reused child process group {process_group}")
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    if not _wait_for_process_group_exit(
            process, process_group, PROCESS_GROUP_TERM_TIMEOUT_SECONDS):
        if not _process_group_identity_is_safe(process_group):
            raise RuntimeError(
                f"refusing to signal reused child process group {process_group}")
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            process.wait()
        if not _wait_for_process_group_exit(
                process, process_group, PROCESS_GROUP_KILL_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"child process group {process_group} survived SIGKILL")
    process.wait()


def _handle_termination(signum, _frame):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _SIGNAL_CLEANUP_IN_PROGRESS

    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, TERMINATION_SIGNALS)
    if _REGISTRATION_IN_PROGRESS:
        try:
            if _DEFERRED_SIGNAL is None:
                _DEFERRED_SIGNAL = signum
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        return
    if _SIGNAL_CLEANUP_IN_PROGRESS:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        return
    process = _ACTIVE_PROCESS
    _ACTIVE_PROCESS = None
    _SIGNAL_CLEANUP_IN_PROGRESS = True
    try:
        if process is not None:
            try:
                _terminate_process_group(process)
            finally:
                _unregister_process(process)
    finally:
        # Keep the guard set while restoring the caller's mask.  A pending
        # second termination signal can run during restoration, and must
        # observe cleanup in progress rather than re-enter it.
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        finally:
            _SIGNAL_CLEANUP_IN_PROGRESS = False
    raise SystemExit(128 + signum)


def _install_signal_handlers():
    for signum in TERMINATION_SIGNALS:
        signal.signal(signum, _handle_termination)


def _spawn_managed_process(command, stdout, stderr):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _REGISTRATION_IN_PROGRESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    _REGISTRATION_IN_PROGRESS = True
    try:
        def restore_child_signal_mask():
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        process = subprocess.Popen(
            [str(item) for item in command], cwd=REPOSITORY_ROOT, text=True,
            stdout=stdout, stderr=stderr, start_new_session=True,
            preexec_fn=restore_child_signal_mask)
        try:
            reservation_fd = os.open(
                f"/proc/{process.pid}", os.O_RDONLY | os.O_DIRECTORY)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.wait()
            process.wait(timeout=PROCESS_GROUP_KILL_TIMEOUT_SECONDS)
            raise
        _PROCESS_GROUP_RESERVATION_FDS[process.pid] = reservation_fd
        _ACTIVE_PROCESS = process
    finally:
        _REGISTRATION_IN_PROGRESS = False
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    deferred_signal = _DEFERRED_SIGNAL
    _DEFERRED_SIGNAL = None
    if deferred_signal is not None:
        _handle_termination(deferred_signal, None)
    return process


def _unregister_process(process):
    global _ACTIVE_PROCESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    try:
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
        if process.pid in _PROCESS_GROUP_RESERVATION_FDS:
            reservation_fd = _PROCESS_GROUP_RESERVATION_FDS.pop(process.pid)
            os.close(reservation_fd)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _run_native(command, stdout, stderr, *, timeout=600):
    started_at = _timestamp()
    process = _spawn_managed_process(command, stdout, stderr)
    try:
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise RuntimeError(
                f"native PID {process.pid} exceeded {timeout} seconds") from error
        except BaseException:
            _terminate_process_group(process)
            raise
        else:
            _terminate_process_group(process)
    finally:
        _unregister_process(process)
    completed_at = _timestamp()
    if returncode != 0:
        raise RuntimeError(
            f"native PID {process.pid} failed with return code {returncode}")
    return {
        "command": shlex.join(str(item) for item in command),
        "pid": process.pid,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "returncode": returncode,
    }


def _validated_output(path):
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a child of {root}")
    if resolved.exists():
        raise FileExistsError(resolved)
    return resolved


def _device_rows():
    query = (
        "index,name,uuid,pci.bus_id,memory.used,utilization.gpu,compute_cap")
    result = _run([
        "nvidia-smi", f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(fields) != 7:
            raise ValueError("unexpected nvidia-smi device row")
        rows.append({
            "index": int(fields[0]),
            "name": fields[1],
            "uuid": fields[2],
            "pci_bus_id": fields[3],
            "memory_used_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
            "compute_capability": fields[6],
        })
    return rows


def _compute_process_rows():
    result = _run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], cwd=REPOSITORY_ROOT, capture_output=True)
    rows = []
    for fields in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError("unexpected nvidia-smi compute-process row")
        rows.append({
            "uuid": fields[0], "pid": int(fields[1]),
            "memory_mib": int(fields[2]),
        })
    return rows


def _require_idle_device(device):
    matches = [row for row in _device_rows() if row["index"] == device]
    if len(matches) != 1:
        raise ValueError(f"device {device} does not resolve exactly once")
    row = matches[0]
    if (row["name"] != EXPECTED_DEVICE
            or row["compute_capability"] != EXPECTED_COMPUTE_CAPABILITY):
        raise ValueError(f"device {device} is not the expected SM80 A800: {row}")
    if (row["memory_used_mib"] > MAXIMUM_IDLE_MEMORY_MIB
            or row["utilization_percent"] != 0):
        raise RuntimeError(f"device {device} is not idle: {row}")
    processes = [
        process for process in _compute_process_rows()
        if process["uuid"] == row["uuid"]]
    if processes:
        raise RuntimeError(f"device {device} has compute processes: {processes}")
    return row


def _wait_for_idle_device(device):
    last_error = None
    for _ in range(20):
        try:
            return _require_idle_device(device)
        except RuntimeError as error:
            last_error = error
            time.sleep(0.5)
    raise RuntimeError(
        f"device {device} did not return idle within ten seconds") from last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--owner", required=True)
    args = parser.parse_args()
    if args.device < 0:
        raise ValueError("device must be nonnegative")
    if OWNER_PATTERN.fullmatch(args.owner) is None:
        raise ValueError("owner must match [A-Za-z0-9._-]{1,64}")
    _install_signal_handlers()
    output = _validated_output(args.output)
    output.mkdir(parents=True)
    for directory in (
            output / "build", output / "raw", output / "source_snapshot"):
        directory.mkdir()
    snapshots = {}
    for path in (SOURCE_PATH, RUNNER_PATH, REPORT_PATH):
        destination = output / "source_snapshot" / path.name
        shutil.copy2(path, destination)
        snapshots[path.name] = destination

    commit = _run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout.strip()
    status = _run(
        ["git", "status", "--short"], cwd=REPOSITORY_ROOT,
        capture_output=True).stdout
    (output / "baseline_commit.txt").write_text(commit + "\n")
    (output / "tracked_worktree_status.txt").write_text(status)

    binary = output / "build/silu_cta_overlap"
    compile_command = [
        NVCC, "-O3", "-std=c++17", "-arch=sm_80", "-lineinfo",
        "--maxrregcount=24", "-Xptxas=-v", snapshots[SOURCE_PATH.name],
        "-o", binary,
    ]
    with (output / "build/compile.stdout").open("w") as stdout, (
            output / "build/compile.stderr").open("w") as stderr:
        _run(compile_command, cwd=REPOSITORY_ROOT, stdout=stdout, stderr=stderr)
    resource_result = _run(
        [CUOBJDUMP, "--dump-resource-usage", binary], cwd=REPOSITORY_ROOT,
        capture_output=True)
    resource_path = output / "build/resource_usage.txt"
    resource_path.write_text(resource_result.stdout)
    resources = REPORT.parse_resource_usage(resource_result.stdout)
    sass_result = _run(
        [CUOBJDUMP, "--dump-sass", binary], cwd=REPOSITORY_ROOT,
        capture_output=True)
    sass_path = output / "build/silu_cta_overlap.sass"
    sass_path.write_text(sass_result.stdout)
    sass_contract = REPORT.validate_sass(sass_result.stdout)

    device_before = _require_idle_device(args.device)
    native_runs = []
    for rotation in range(RUNS):
        command = [binary, args.device, TRIALS, rotation]
        with (output / f"raw/run_{rotation}.csv").open("w") as stdout, (
                output / f"raw/run_{rotation}.stderr").open("w") as stderr:
            lifecycle = _run_native(command, stdout, stderr)
        native_runs.append({"rotation": rotation, **lifecycle})
        REPORT.parse_raw_text(
            (output / f"raw/run_{rotation}.csv").read_text(), rotation)
    device_after = _wait_for_idle_device(args.device)
    candidate = REPORT.build_report(output)
    (output / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n")

    provenance = {
        "schema": "nvidia_a800_silu_cta_overlap_capture_v3",
        "owner": args.owner,
        "runner_pid": os.getpid(),
        "baseline_commit": commit,
        "command": shlex.join([sys.executable, *sys.argv]),
        "compile_command": shlex.join(str(item) for item in compile_command),
        "native_runs": native_runs,
        "device_before": device_before,
        "device_after": device_after,
        "runs": RUNS,
        "trials_per_run": TRIALS,
        "source_sha256": _sha256(snapshots[SOURCE_PATH.name]),
        "runner_sha256": _sha256(snapshots[RUNNER_PATH.name]),
        "report_sha256": _sha256(snapshots[REPORT_PATH.name]),
        "binary_sha256": _sha256(binary),
        "resource_usage_sha256": _sha256(resource_path),
        "sass_sha256": _sha256(sass_path),
        "resources": resources,
        "sass_contract": sass_contract,
        "production_or_evaluation_duration_consumed": False,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
