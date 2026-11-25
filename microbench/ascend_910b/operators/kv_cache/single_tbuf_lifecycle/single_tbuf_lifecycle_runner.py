#!/usr/bin/env python3
"""Run balanced Ascend 910B single-TBuf lifecycle trials."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


BLOCK_DIMS = (1, 2, 4, 8, 16, 48)
# The first two cells are the CANN key-200 token buffers.  110080 B is the
# exact BroadcastOneDimSch allocation for the BF16 Mul key-210020000
# two-tile class: ubFormer=5504 elements and the source DAG's compile-time
# footprint is 20 bytes per element.
PAYLOAD_BYTES = (1024, 2048, 110080)
ITERATIONS = 64
CELLS = tuple(
    (block_dim, payload_bytes)
    for payload_bytes in PAYLOAD_BYTES
    for block_dim in BLOCK_DIMS)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_index(trial_index, order_position):
    rotation = trial_index % len(CELLS)
    phase = trial_index // len(CELLS)
    if phase % 2 == 0:
        return (rotation + order_position) % len(CELLS)
    return (rotation - order_position) % len(CELLS)


def _terminate(process):
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
        _terminate(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=2 * len(CELLS))
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if args.trials != 2 * len(CELLS):
        parser.error(f"--trials must equal {2 * len(CELLS)}")
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,payload_bytes,"
        "iterations,short_block_cycles,long_block_cycles,tick_ns,"
        "service_ns,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(args.trials):
        ordered = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [str(args.binary), "--batch"]
        command.extend(
            f"{block_dim}:{payload_bytes}:{ITERATIONS}"
            for block_dim, payload_bytes in ordered)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True)
        _ACTIVE_PROCESS = process
        try:
            stdout, stderr = process.communicate(timeout=900)
        except subprocess.TimeoutExpired as error:
            _terminate(process)
            raise RuntimeError(f"child PID {process.pid} timed out") from error
        finally:
            if process.poll() is None:
                _terminate(process)
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with {process.returncode}")
        lines = stdout.strip().splitlines()
        if len(lines) != len(ordered):
            raise RuntimeError(
                f"child PID {process.pid} emitted {len(lines)} rows")
        for position, (line, cell) in enumerate(zip(lines, ordered)):
            fields = line.split(",")
            if (len(fields) != 8
                    or (int(fields[0]), int(fields[1])) != cell
                    or int(fields[2]) != ITERATIONS
                    or fields[-1] != "pass"):
                raise RuntimeError("child changed the requested cell order")
            print(
                f"{args.run_id},{trial_index + 1},{position},"
                f"{process.pid},{line}", flush=True)


if __name__ == "__main__":
    main()
