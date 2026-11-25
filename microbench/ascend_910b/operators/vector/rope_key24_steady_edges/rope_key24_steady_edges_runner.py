#!/usr/bin/env python3
"""Run the Ascend 910B RoPE key-24 double-buffer edge campaign."""

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


BLOCK_DIM = 47
MODE_NAMES = (
    "mte2_first", "mte2_steady", "mte3_completion", "event_init_drain")
BASE_CELLS = (
    (0, 4096, 1), (0, 2048, 1), (0, 128, 1),
    (1, 4096, 1), (1, 4096, 2), (1, 4096, 4), (1, 4096, 8),
    (1, 4096, 20), (1, 4096, 21),
    (2, 4096, 1), (2, 2048, 1), (2, 128, 1),
    (3, 0, 16),
)
PAIR_ORDERS = (0, 1)
CELLS = tuple(
    (*base_cell, pair_order)
    for base_cell in BASE_CELLS
    for pair_order in PAIR_ORDERS)
DEFAULT_SCHEDULE_CYCLES = 2
STRIDE_BYTES = 131072
SYSTEM_COUNTER_TICK_NS = 20.0
CHILD_CSV_FIELDS = 13
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
PROCESS_GROUP_TERM_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_POLL_SECONDS = 0.01
_ACTIVE_PROCESS = None


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
    return (trial_index + order_position) % len(CELLS)


def _validate_schedule(trials):
    if trials <= 0 or trials % len(CELLS) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {len(CELLS)}")
    counts = [[0 for _ in CELLS] for _ in CELLS]
    for trial_index in range(trials):
        seen = set()
        for order_position in range(len(CELLS)):
            cell_index = _cell_index(trial_index, order_position)
            if cell_index in seen:
                raise AssertionError("one trial contains a duplicate cell")
            seen.add(cell_index)
            counts[order_position][cell_index] += 1
    expected = trials // len(CELLS)
    if any(count != expected for row in counts for count in row):
        raise AssertionError("cell/order schedule is not balanced")


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != CHILD_CSV_FIELDS:
        raise RuntimeError("child emitted an invalid schema")
    try:
        integers = tuple(int(fields[index]) for index in range(9))
        control_ns = float(fields[9])
        request_ns = float(fields[10])
        sink = int(fields[11])
    except ValueError as exc:
        raise RuntimeError("child emitted a non-numeric value") from exc
    (block_dim, actual_worker, mode, payload_bytes, request_count, pair_order,
     stride_bytes, control_cycles, request_cycles) = integers
    if (mode, payload_bytes, request_count, pair_order) != expected:
        raise RuntimeError("child changed requested cell")
    if (block_dim != BLOCK_DIM or actual_worker != worker
            or stride_bytes != STRIDE_BYTES):
        raise RuntimeError("child changed frozen launch geometry")
    if control_cycles < 0 or request_cycles <= 0:
        raise RuntimeError("child emitted invalid cycle counts")
    if (not math.isfinite(control_ns) or not math.isfinite(request_ns)
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or request_ns != request_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child emitted inconsistent elapsed time")
    if sink <= 0 or fields[12] != "pass":
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


def _terminate_process_group(process):
    if not _process_group_exists(process.pid):
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + PROCESS_GROUP_TERM_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process.pid):
            process.wait()
            return
        time.sleep(PROCESS_GROUP_POLL_SECONDS)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.wait()
    process.wait()


def _handle_signal(signum, _frame):
    process = _ACTIVE_PROCESS
    if process is not None:
        _terminate_process_group(process)
    raise SystemExit(128 + signum)


def _communicate(command, *, environment, timeout):
    global _ACTIVE_PROCESS

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out after {timeout} seconds"
            ) from exc
        except BaseException:
            _terminate_process_group(process)
            raise
        else:
            _terminate_process_group(process)
    finally:
        _ACTIVE_PROCESS = None
    return process.returncode, process.pid, stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
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
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    trials = (
        DEFAULT_SCHEDULE_CYCLES * len(CELLS)
        if args.trials is None else args.trials)
    try:
        _validate_schedule(trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} owner=ascend_rope_key24_steady_edges "
        f"device_id={args.device_id} run_id={args.run_id} "
        f"block_dim={BLOCK_DIM} trials={trials} "
        f"schedule_cycles={trials // len(CELLS)} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,mode,"
        "payload_bytes,request_count,pair_order,stride_bytes,control_cycles,"
        "request_cycles,control_ns,request_ns,sink,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for managed_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(managed_signal, _handle_signal)
    for trial_index in range(trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [
            str(args.binary), "--batch",
            *(f"{mode}:{payload}:{count}:{pair_order}"
              for mode, payload, count, pair_order in ordered_cells),
        ]
        returncode, process_pid, stdout, stderr = _communicate(
            command, environment=environment,
            timeout=args.child_timeout_seconds)
        if stderr:
            print(
                f"child_pid={process_pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
        if returncode != 0:
            raise RuntimeError(
                f"child PID {process_pid} failed with exit {returncode}: "
                f"{' '.join(command)}")
        lines = stdout.strip().splitlines()
        expected_lines = BLOCK_DIM * len(CELLS)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child PID {process_pid} emitted {len(lines)} data lines; "
                f"expected {expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            for worker in range(BLOCK_DIM):
                validated = _parse_child_row(
                    lines[line_index], expected, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process_pid},{validated}", flush=True)


if __name__ == "__main__":
    main()
