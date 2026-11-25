#!/usr/bin/env python3
"""Capture the independent FIA AIV output-init plus SyncAll phase."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


AIV_BLOCK_DIMS = (2, 8, 16, 24, 32, 48)
ELEMENTS_PER_AIV = (16, 64, 128, 256, 1024, 4096, 16384)
TRIALS = 8
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_cells(trial_index):
    cells = tuple(
        (block_dim, elements)
        for block_dim in AIV_BLOCK_DIMS
        for elements in ELEMENTS_PER_AIV)
    rotation = trial_index % len(cells)
    rotated = cells[rotation:] + cells[:rotation]
    return rotated if trial_index % 2 == 0 else tuple(reversed(rotated))


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_signal(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")

    schedule = tuple(
        (trial + 1, position, block_dim, elements)
        for trial in range(TRIALS)
        for position, (block_dim, elements) in enumerate(
            _ordered_cells(trial)))
    command = [str(args.binary), "--batch"]
    for _, _, block_dim, elements in schedule:
        command.extend((str(block_dim), str(elements)))
    environment = os.environ.copy()
    environment["ASCEND_DEVICE_ID"] = str(args.device_id)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        stdout, stderr = process.communicate(
            timeout=args.child_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise RuntimeError(f"child PID {process.pid} timed out") from error
    finally:
        if process.poll() is None:
            _terminate_process_group(process)
        _ACTIVE_PROCESS = None
    if stderr:
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"child PID {process.pid} failed with {process.returncode}")
    lines = stdout.strip().splitlines()
    if len(lines) != len(schedule) + 1:
        raise RuntimeError("child emitted an invalid number of raw rows")
    expected_header = (
        "aiv_block_dim,elements_per_aiv,block_cycles,tick_ns,"
        "phase_time_ns,device_elapsed_ns,validation")
    if lines[0] != expected_header:
        raise RuntimeError("child emitted an invalid header")

    print(
        "run_id,trial,order_position,process_pid," + expected_header,
        flush=True)
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    for expected, line in zip(schedule, lines[1:]):
        trial, position, block_dim, elements = expected
        fields = line.split(",")
        if (len(fields) != 7 or fields[-1] != "pass"
                or int(fields[0]) != block_dim
                or int(fields[1]) != elements):
            raise RuntimeError("child changed the requested cell")
        print(
            f"{args.run_id},{trial},{position},{process.pid},{line}",
            flush=True)


if __name__ == "__main__":
    main()
