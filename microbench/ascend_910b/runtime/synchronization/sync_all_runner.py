#!/usr/bin/env python3
"""Capture the complete Ascend 910B AIV SyncAll participant domain."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


BLOCK_DIMS = tuple(range(2, 49, 2))
TRIALS = 12
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _ordered_block_dims(trial_index):
    if trial_index < 0 or trial_index >= TRIALS:
        raise ValueError("trial index is outside the frozen campaign")
    rotation = trial_index % len(BLOCK_DIMS)
    rotated = BLOCK_DIMS[rotation:] + BLOCK_DIMS[:rotation]
    return rotated if trial_index % 2 == 0 else tuple(reversed(rotated))


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child-timeout-seconds", type=int, default=120)
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
        (trial + 1, order_position, block_dim)
        for trial in range(TRIALS)
        for order_position, block_dim in enumerate(
            _ordered_block_dims(trial)))
    command = [
        str(args.binary), "--batch",
        *(str(block_dim) for _, _, block_dim in schedule),
    ]
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
    if lines[0] != "block_dim,block_cycles,host_wall_ns,validation":
        raise RuntimeError("child emitted an invalid header")

    print(
        "run_id,trial,order_position,process_pid,block_dim,block_cycles,"
        "host_wall_ns,validation",
        flush=True,
    )
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    for (trial, order_position, block_dim), line in zip(schedule, lines[1:]):
        fields = line.split(",")
        if len(fields) != 4 or fields[-1] != "pass":
            raise RuntimeError("child emitted a failed raw row")
        if int(fields[0]) != block_dim:
            raise RuntimeError("child changed the requested cell")
        print(
            f"{args.run_id},{trial},{order_position},{process.pid},{line}",
            flush=True,
        )


if __name__ == "__main__":
    main()
