#!/usr/bin/env python3
"""Run balanced raw trials for the CANN AIV tiling-data mechanism."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


BLOCK_DIMS = (1, 24, 48)
SUPPORTED_PAYLOAD_BYTES = (28, 48, 56, 80, 192, 1872)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
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
    parser.add_argument(
        "--payload-bytes", type=int, choices=SUPPORTED_PAYLOAD_BYTES,
        default=28)
    parser.add_argument("--trials", type=_positive_integer, default=6)
    parser.add_argument("--iterations", type=_positive_integer, default=4096)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=120)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.trials % (2 * len(BLOCK_DIMS)) != 0:
        parser.error("--trials must be a positive multiple of 6")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")

    print(
        "run_id,trial,order_position,process_pid,payload_bytes,block_dim,"
        "iterations,short_cycles,long_cycles,tick_ns,service_time_ns,"
        "validation",
        flush=True,
    )
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    environment = os.environ.copy()
    # The child passes this physical ID directly to aclrtSetDevice.  Do not
    # also remap visibility, because a singleton visible set renumbers it to 0.
    if "ASCEND_RT_VISIBLE_DEVICES" in environment:
        del environment["ASCEND_RT_VISIBLE_DEVICES"]
    environment["ASCEND_DEVICE_ID"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    schedule = []
    for trial in range(args.trials):
        rotation = trial % len(BLOCK_DIMS)
        rotated = BLOCK_DIMS[rotation:] + BLOCK_DIMS[:rotation]
        ordered = rotated if (trial // len(BLOCK_DIMS)) % 2 == 0 else tuple(
            reversed(rotated))
        schedule.extend(
            (trial + 1, order_position, block_dim)
            for order_position, block_dim in enumerate(ordered))
    command = [
        str(args.binary), "--batch", str(args.payload_bytes),
        str(args.iterations),
        *(str(block_dim) for _, _, block_dim in schedule),
    ]
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
    for (trial, order_position, block_dim), line in zip(
            schedule, lines[1:]):
        if not line.endswith(",pass"):
            raise RuntimeError("child emitted a failed raw row")
        fields = line.split(",")
        if (int(fields[0]) != args.payload_bytes
                or int(fields[1]) != block_dim):
            raise RuntimeError("child changed the requested cell")
        print(
            f"{args.run_id},{trial},{order_position},"
            f"{process.pid},{line}",
            flush=True,
        )


if __name__ == "__main__":
    main()
