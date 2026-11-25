#!/usr/bin/env python3
"""Run the frozen RMSNorm key-0 warmed repeated-row edge campaign."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


RMS_NORM_SOURCE_SHA256 = (
    "d6f83f716070a583367c95c6f215cf1a4f94631953730aa50268e446b45f28f8")
RMS_NORM_BASE_SOURCE_SHA256 = (
    "a2ece498a72530c9163330d3631676ed5879af99ad16ea3e8f37a2d8838dae93")
BLOCK_DIMS = (32, 43, 47)
MODE_NAMES = (
    "mte2_steady", "mte2_terminal", "mte3_steady", "mte3_terminal")
MODE_REQUEST_COUNTS = (8, 1, 8, 1)
PAYLOAD_BYTES = (4096, 5120, 10240)
PAIR_ORDERS = (0, 1)
BASE_CELLS = tuple(
    (block_dim, mode, payload_bytes)
    for block_dim in BLOCK_DIMS
    for mode in (0, 2)
    for payload_bytes in PAYLOAD_BYTES)
CELLS = tuple(
    (*base_cell, pair_order)
    for base_cell in BASE_CELLS
    for pair_order in PAIR_ORDERS)
DEFAULT_TRIALS = 7 * len(CELLS)
REQUIRED_INDEPENDENT_RUNS = 3
STRIDE_BYTES = 131072
SYSTEM_COUNTER_TICK_NS = 20.0
CHILD_CSV_FIELDS = 14
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
PROCESS_GROUP_TERM_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_KILL_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_POLL_SECONDS = 0.01
_ACTIVE_PROCESS = None
_PROCESS_GROUP_RESERVATION_FDS = {}
_REGISTRATION_IN_PROGRESS = False
_DEFERRED_SIGNAL = None
_SIGNAL_CLEANUP_IN_PROGRESS = False


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _cell_index(trial_index, order_position):
    return (order_position + 5 * trial_index) % len(CELLS)


def _validate_schedule(trials):
    if trials <= 0 or trials % len(CELLS) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {len(CELLS)}")
    for position in range(len(CELLS)):
        counts = {
            cell: sum(
                CELLS[_cell_index(trial, position)] == cell
                for trial in range(trials))
            for cell in CELLS
        }
        if set(counts.values()) != {trials // len(CELLS)}:
            raise ValueError("campaign schedule is not position-balanced")


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != CHILD_CSV_FIELDS:
        raise RuntimeError("child emitted an invalid schema")
    try:
        integers = tuple(int(fields[index]) for index in range(9))
        control_ns = float(fields[9])
        target_ns = float(fields[10])
        sink = int(fields[11])
        destination_head = int(fields[12])
    except ValueError as exc:
        raise RuntimeError("child emitted non-numeric data") from exc
    (block_dim, actual_worker, mode, payload_bytes, request_count, pair_order,
     stride_bytes, control_cycles, target_cycles) = integers
    expected_block, expected_mode, expected_payload, expected_order = expected
    if ((block_dim, mode, payload_bytes, pair_order)
            != (expected_block, expected_mode, expected_payload,
                expected_order)):
        raise RuntimeError("child changed requested cell")
    if (actual_worker != worker
            or request_count != MODE_REQUEST_COUNTS[mode]
            or stride_bytes != STRIDE_BYTES
            or control_cycles < 0 or target_cycles <= 0
            or not math.isfinite(control_ns) or control_ns < 0
            or not math.isfinite(target_ns) or target_ns <= 0
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or target_ns != target_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child changed the frozen value contract")
    expected_sink = request_count + 1 if mode < 2 else request_count
    expected_destination = 0 if mode < 2 else 0x3C00
    if (sink != expected_sink or destination_head != expected_destination
            or fields[13] != "pass"):
        raise RuntimeError("child data validation failed")
    return line


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


def _wait_for_process_group_exit(process, timeout):
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_exists(process.pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(PROCESS_GROUP_POLL_SECONDS)


def _terminate_process_group(process):
    if not _process_group_exists(process.pid):
        process.wait()
        return
    if not _process_group_identity_is_safe(process.pid):
        raise RuntimeError(f"refusing to signal reused process {process.pid}")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    if not _wait_for_process_group_exit(
            process, PROCESS_GROUP_TERM_TIMEOUT_SECONDS):
        if not _process_group_identity_is_safe(process.pid):
            raise RuntimeError(
                f"refusing to signal reused process {process.pid}")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            process.wait()
        if not _wait_for_process_group_exit(
                process, PROCESS_GROUP_KILL_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"process group {process.pid} survived SIGKILL")
    process.wait()


def _unregister_process(process):
    global _ACTIVE_PROCESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    try:
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
        if process.pid in _PROCESS_GROUP_RESERVATION_FDS:
            os.close(_PROCESS_GROUP_RESERVATION_FDS.pop(process.pid))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _handle_signal(signum, _frame):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _SIGNAL_CLEANUP_IN_PROGRESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
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
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        finally:
            _SIGNAL_CLEANUP_IN_PROGRESS = False
    raise SystemExit(128 + signum)


def _spawn_process(command, environment):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _REGISTRATION_IN_PROGRESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    _REGISTRATION_IN_PROGRESS = True
    try:
        def restore_child_mask():
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True,
            preexec_fn=restore_child_mask)
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
        _handle_signal(deferred_signal, None)
    return process


def _communicate(command, environment, timeout):
    process = _spawn_process(command, environment)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(f"child PID {process.pid} timed out") from exc
        except BaseException:
            _terminate_process_group(process)
            raise
        else:
            _terminate_process_group(process)
    finally:
        _unregister_process(process)
    return process.returncode, process.pid, stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--base-source-file", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    binary = args.binary.resolve()
    source_file = args.source_file.resolve()
    base_source_file = args.base_source_file.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if not source_file.is_file() or not base_source_file.is_file():
        parser.error("source files must name the frozen CANN sources")
    if (_sha256(source_file) != RMS_NORM_SOURCE_SHA256
            or _sha256(base_source_file) != RMS_NORM_BASE_SOURCE_SHA256):
        parser.error("CANN RMSNorm source identity changed")
    trials = DEFAULT_TRIALS if args.trials is None else args.trials
    try:
        _validate_schedule(trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} owner=ascend_rmsnorm_key0_steady_edges "
        f"physical_device_id={args.device_id} run_id={args.run_id} "
        f"trials={trials} schedule_cycles={trials // len(CELLS)} "
        f"required_independent_runs={REQUIRED_INDEPENDENT_RUNS} "
        f"binary={binary} binary_sha256={_sha256(binary)} "
        f"source_file={source_file} source_sha256={_sha256(source_file)} "
        f"base_source_file={base_source_file} "
        f"base_source_sha256={_sha256(base_source_file)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,mode,"
        "payload_bytes,request_count,pair_order,stride_bytes,control_cycles,"
        "target_cycles,control_ns,target_ns,sink,destination_head,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for managed_signal in TERMINATION_SIGNALS:
        signal.signal(managed_signal, _handle_signal)
    for trial_index in range(trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [
            str(binary), "--batch",
            *(f"{block_dim}:{mode}:{payload_bytes}:{pair_order}"
              for block_dim, mode, payload_bytes, pair_order
              in ordered_cells),
        ]
        returncode, process_pid, stdout, stderr = _communicate(
            command, environment, args.child_timeout_seconds)
        if stderr:
            print(
                f"child_pid={process_pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
        if returncode != 0:
            raise RuntimeError(
                f"child PID {process_pid} failed with exit {returncode}")
        lines = stdout.strip().splitlines()
        expected_lines = sum(cell[0] for cell in ordered_cells)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child emitted {len(lines)} rows; expected {expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            for worker in range(expected[0]):
                line = _parse_child_row(lines[line_index], expected, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process_pid},{line}", flush=True)


if __name__ == "__main__":
    main()
