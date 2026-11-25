#!/usr/bin/env python3
"""Run balanced RMSNorm key-1 partial-sum handoff captures."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


SOURCE_SHA256 = "4e308c07445c14510322c2c6fe7126a6f2763ed5356235fef66eabf1ee830292"
EXISTING_BLOCK_DIMS = (1, 32, 47)
EXISTING_PARTIAL_COUNTS = (8, 32, 64)
PARTIAL_ONE_BLOCK_DIMS = (1, 2, 4, 8, 16, 32, 43, 47)
PAIR_ORDERS = (0, 1)
ITERATIONS = 512
BASE_CELLS = (
    tuple((block_dim, 1) for block_dim in PARTIAL_ONE_BLOCK_DIMS)
    + tuple(
        (block_dim, partial_count)
        for block_dim in EXISTING_BLOCK_DIMS
        for partial_count in EXISTING_PARTIAL_COUNTS)
)
CELLS = tuple(
    (*base_cell, pair_order)
    for base_cell in BASE_CELLS
    for pair_order in PAIR_ORDERS
)
DEFAULT_SCHEDULE_CYCLES = 1
REQUIRED_INDEPENDENT_RUNS = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
EXPECTED_FIELDS = 13
MANAGED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
PROCESS_GROUP_TERM_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_KILL_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_POLL_SECONDS = 0.01
_ACTIVE_PROCESS = None
_PROCESS_GROUP_RESERVATION_FDS = {}
_REGISTRATION_IN_PROGRESS = False
_DEFERRED_SIGNAL = None
_SIGNAL_CLEANUP_IN_PROGRESS = False


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_index(trial, position):
    # Five is coprime to the 34-cell campaign, so one complete cycle rotates
    # every cell through every position exactly once.
    return (position + 5 * trial) % len(CELLS)


def _validate_schedule(trials):
    if trials % len(CELLS) != 0:
        raise ValueError(f"trials must be a multiple of {len(CELLS)}")
    for position in range(len(CELLS)):
        counts = {
            cell: sum(
                CELLS[_cell_index(trial, position)] == cell
                for trial in range(trials))
            for cell in CELLS
        }
        if len(set(counts.values())) != 1:
            raise ValueError("campaign schedule is not position-balanced")


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != EXPECTED_FIELDS:
        raise RuntimeError("child row has an invalid field count")
    integers = tuple(int(fields[index]) for index in range(9))
    block_dim, observed_worker, partial_count, pair_order = integers[:4]
    partial_stride_bytes, group_payload_bytes, iterations = integers[4:7]
    control_cycles, target_cycles = integers[7:9]
    control_ns = float(fields[9])
    target_ns = float(fields[10])
    sink = int(fields[11])
    if ((block_dim, partial_count, pair_order) != expected
            or observed_worker != worker
            or partial_stride_bytes != 32
            or group_payload_bytes != partial_count * partial_stride_bytes
            or iterations != ITERATIONS
            or control_cycles <= 0 or target_cycles <= 0
            or control_ns != control_cycles * 20.0
            or target_ns != target_cycles * 20.0
            or sink != 4 * partial_count or fields[12] != "pass"):
        raise RuntimeError("child row failed strict identity validation")
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
    # The open /proc inode reserves the original leader identity even after
    # exit.  A different live inode proves numeric PID/PGID reuse.
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


def _handle_signal(signum, _frame):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _SIGNAL_CLEANUP_IN_PROGRESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        if _REGISTRATION_IN_PROGRESS or _SIGNAL_CLEANUP_IN_PROGRESS:
            if _DEFERRED_SIGNAL is None:
                _DEFERRED_SIGNAL = signum
            return
        process = _ACTIVE_PROCESS
        _ACTIVE_PROCESS = None
        _SIGNAL_CLEANUP_IN_PROGRESS = True
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    try:
        if process is not None:
            _terminate_process_group(process)
    finally:
        previous_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, MANAGED_SIGNALS)
        try:
            _SIGNAL_CLEANUP_IN_PROGRESS = False
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    raise SystemExit(128 + signum)


def _install_signal_handlers():
    for managed_signal in MANAGED_SIGNALS:
        signal.signal(managed_signal, _handle_signal)


def _close_process_streams(process):
    for stream in (process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _cleanup_registered_spawn(process):
    # A pending managed signal can be delivered by the mask restoration at
    # the end of registration, before _communicate_managed has entered its
    # own try/finally.  Keep a complete cleanup boundary inside spawn for
    # exactly that interval.
    try:
        _terminate_process_group(process)
    finally:
        _unregister_process(process)


def _spawn_managed_process(command, *, environment=None):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _REGISTRATION_IN_PROGRESS

    process = None
    registered = False
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
    _REGISTRATION_IN_PROGRESS = True
    try:
        def restore_child_signal_mask():
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True,
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
            _close_process_streams(process)
            raise
        _PROCESS_GROUP_RESERVATION_FDS[process.pid] = reservation_fd
        _ACTIVE_PROCESS = process
        registered = True
    finally:
        _REGISTRATION_IN_PROGRESS = False
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException:
            if registered:
                _cleanup_registered_spawn(process)
            raise
    try:
        deferred_signal = _DEFERRED_SIGNAL
        _DEFERRED_SIGNAL = None
        if deferred_signal is not None:
            _handle_signal(deferred_signal, None)
    except BaseException:
        if registered:
            _cleanup_registered_spawn(process)
        raise
    return process


def _unregister_process(process):
    global _ACTIVE_PROCESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
        reservation_fd = _PROCESS_GROUP_RESERVATION_FDS.pop(process.pid)
        try:
            os.close(reservation_fd)
        finally:
            _close_process_streams(process)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _communicate_managed(command, *, environment=None, timeout):
    process = _spawn_managed_process(command, environment=environment)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out after {timeout} seconds"
            ) from error
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
    parser.add_argument("--device-id", required=True, type=_positive_integer)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    binary = args.binary.resolve()
    source_file = args.source_file.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if not source_file.is_file():
        parser.error("--source-file must name the frozen CANN source")
    source_sha256 = _sha256(source_file)
    if source_sha256 != SOURCE_SHA256:
        parser.error("CANN RMSNorm key-1 source hash changed")
    trials = (
        DEFAULT_SCHEDULE_CYCLES * len(CELLS)
        if args.trials is None else args.trials)
    try:
        _validate_schedule(trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} owner=ascend_rmsnorm_key1_reduction "
        f"physical_device_id={args.device_id} run_id={args.run_id} "
        f"trials={trials} schedule_cycles={trials // len(CELLS)} "
        f"required_independent_runs={REQUIRED_INDEPENDENT_RUNS} "
        f"binary={binary} binary_sha256={_sha256(binary)} "
        f"source_file={source_file} source_sha256={source_sha256}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "partial_count,pair_order,partial_stride_bytes,group_payload_bytes,"
        "iterations,control_cycles,target_cycles,control_ns,target_ns,"
        "sink,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    _install_signal_handlers()
    for trial_index in range(trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [
            str(binary), "--batch",
            *(f"{block_dim}:{partial_count}:{pair_order}:{ITERATIONS}"
              for block_dim, partial_count, pair_order in ordered_cells),
        ]
        returncode, process_pid, stdout, stderr = _communicate_managed(
            command, environment=environment,
            timeout=args.child_timeout_seconds)
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
                f"child PID {process_pid} emitted {len(lines)} rows; "
                f"expected {expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            for worker in range(expected[0]):
                line = _parse_child_row(lines[line_index], expected, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process_pid},{line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
