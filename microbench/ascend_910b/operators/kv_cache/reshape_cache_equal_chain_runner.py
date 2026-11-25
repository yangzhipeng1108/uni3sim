#!/usr/bin/env python3
"""Run the frozen key-200 hard-event-chain mechanism campaign."""

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


DEFAULT_CAMPAIGN = "edge_workers"
CAMPAIGN_GEOMETRIES = {
    DEFAULT_CAMPAIGN: ((1, 1024), (1, 2048), (48, 1024), (48, 2048)),
    "equal_midworkers": (
        (2, 1024), (2, 2048),
        (4, 1024), (4, 2048),
        (8, 1024), (8, 2048),
        (16, 1024), (16, 2048),
    ),
    "equal_midworkers_warm": (
        (2, 1024), (2, 2048),
        (4, 1024), (4, 2048),
        (8, 1024), (8, 2048),
        (16, 1024), (16, 2048),
    ),
}
CAMPAIGN_OWNERS = {
    DEFAULT_CAMPAIGN: "ascend_kvcache_residual",
    "equal_midworkers": "ascend_kvcache_equal_midworkers",
    "equal_midworkers_warm": "capture-owner-4c0f2645f650",
}
GEOMETRIES = CAMPAIGN_GEOMETRIES[DEFAULT_CAMPAIGN]
PAIR_ORDERS = (0, 1)
CAMPAIGN_CELLS = {
    campaign: tuple(
        (block_dim, payload_bytes, pair_order)
        for block_dim, payload_bytes in geometries
        for pair_order in PAIR_ORDERS)
    for campaign, geometries in CAMPAIGN_GEOMETRIES.items()
}
CELLS = CAMPAIGN_CELLS[DEFAULT_CAMPAIGN]
DEFAULT_SCHEDULE_CYCLES = 8
STRIDE_BYTES = 4096
SYSTEM_COUNTER_TICK_NS = 20.0
CSV_FIELDS = 13
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
MANAGED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
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


def _cell_index(trial_index, order_position, cells=CELLS):
    return (trial_index + order_position) % len(cells)


def _validate_schedule(trials, cells=CELLS):
    if trials <= 0 or trials % len(cells) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {len(cells)}")
    counts = [[0 for _ in cells] for _ in cells]
    for trial_index in range(trials):
        seen = set()
        for order_position in range(len(cells)):
            cell_index = _cell_index(trial_index, order_position, cells)
            if cell_index in seen:
                raise AssertionError("one trial contains a duplicate cell")
            seen.add(cell_index)
            counts[order_position][cell_index] += 1
    expected = trials // len(cells)
    if any(count != expected for row in counts for count in row):
        raise AssertionError("cell/order schedule is not balanced")


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != CSV_FIELDS:
        raise RuntimeError("child emitted an invalid schema")
    try:
        integers = tuple(int(fields[index]) for index in range(7))
        control_ns = float(fields[7])
        request_ns = float(fields[8])
        sink = int(fields[9])
        key_head = int(fields[10])
        value_head = int(fields[11])
    except ValueError as exc:
        raise RuntimeError("child emitted a non-numeric value") from exc
    (block_dim, actual_worker, payload_bytes, pair_order, stride_bytes,
     control_cycles, request_cycles) = integers
    if (block_dim, payload_bytes, pair_order) != expected:
        raise RuntimeError("child changed requested cell")
    if actual_worker != worker or stride_bytes != STRIDE_BYTES:
        raise RuntimeError("child changed worker or stride")
    if control_cycles < 0 or request_cycles <= 0:
        raise RuntimeError("child emitted invalid cycle counts")
    if (not math.isfinite(control_ns) or not math.isfinite(request_ns)
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or request_ns != request_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child emitted inconsistent elapsed time")
    if (sink, key_head, value_head, fields[12]) != (
            2, 0x3c00, 0x4000, "pass"):
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
    # Keep the original /proc inode open until cleanup.  A live path with a
    # different inode proves that the numeric leader PID/PGID was reused.
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
                f"refusing to signal reused child process group "
                f"{process_group}")
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

    if _REGISTRATION_IN_PROGRESS or _SIGNAL_CLEANUP_IN_PROGRESS:
        if _DEFERRED_SIGNAL is None:
            _DEFERRED_SIGNAL = signum
        return
    process = _ACTIVE_PROCESS
    _ACTIVE_PROCESS = None
    _SIGNAL_CLEANUP_IN_PROGRESS = True
    try:
        if process is not None:
            _terminate_process_group(process)
    finally:
        _SIGNAL_CLEANUP_IN_PROGRESS = False
    raise SystemExit(128 + signum)


def _install_signal_handlers():
    for managed_signal in MANAGED_SIGNALS:
        signal.signal(managed_signal, _handle_signal)


def _spawn_managed_process(command, *, environment=None):
    global _ACTIVE_PROCESS
    global _DEFERRED_SIGNAL
    global _REGISTRATION_IN_PROGRESS

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


def _unregister_process(process):
    global _ACTIVE_PROCESS

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, MANAGED_SIGNALS)
    try:
        if _ACTIVE_PROCESS is process:
            _ACTIVE_PROCESS = None
        reservation_fd = _PROCESS_GROUP_RESERVATION_FDS.pop(process.pid)
        os.close(reservation_fd)
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
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--campaign", choices=tuple(CAMPAIGN_CELLS),
        default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--warmup-launches", type=int, choices=(0, 4, 8, 16), default=0)
    parser.add_argument("--trials", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    cells = CAMPAIGN_CELLS[args.campaign]
    if args.campaign == "equal_midworkers_warm":
        if args.warmup_launches == 0:
            parser.error(
                "equal_midworkers_warm requires --warmup-launches 4, 8, or 16")
    elif args.warmup_launches != 0:
        parser.error("--warmup-launches is only valid for the warm campaign")
    trials = (
        DEFAULT_SCHEDULE_CYCLES * len(cells)
        if args.trials is None else args.trials)
    try:
        _validate_schedule(trials, cells)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} owner={CAMPAIGN_OWNERS[args.campaign]} "
        f"device_id={args.device_id} run_id={args.run_id} "
        f"campaign={args.campaign} trials={trials} "
        f"warmup_launches={args.warmup_launches} "
        f"schedule_cycles={trials // len(cells)} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "payload_bytes,pair_order,stride_bytes,control_cycles,request_cycles,"
        "control_ns,request_ns,sink,key_destination_head,"
        "value_destination_head,validation,warmup_launches",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    _install_signal_handlers()
    for trial_index in range(trials):
        ordered_cells = tuple(
            cells[_cell_index(trial_index, position, cells)]
            for position in range(len(cells)))
        command = [
            str(args.binary), "--batch",
            *(f"{block_dim}:{payload_bytes}:{pair_order}:"
              f"{args.warmup_launches}"
              for block_dim, payload_bytes, pair_order in ordered_cells),
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
                f"child PID {process_pid} failed with exit "
                f"{returncode}: {' '.join(command)}")
        lines = stdout.strip().splitlines()
        expected_lines = sum(cell[0] for cell in ordered_cells)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child PID {process_pid} emitted {len(lines)} data lines; "
                f"expected {expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            for worker in range(expected[0]):
                line = lines[line_index]
                line_index += 1
                validated = _parse_child_row(line, expected, worker)
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process_pid},{validated},{args.warmup_launches}",
                    flush=True)


if __name__ == "__main__":
    main()
