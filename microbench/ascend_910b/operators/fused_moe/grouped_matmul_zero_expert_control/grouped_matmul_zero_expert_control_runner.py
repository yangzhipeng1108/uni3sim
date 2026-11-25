#!/usr/bin/env python3
"""Run the frozen AIC GroupedMatmul zero-expert controller sweep."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


CELLS = (8, 32, 64, 128, 256, 512)
REPEATS = 256
CSV_FIELDS = 7
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


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id not in (6, 7):
        parser.error("--device-id must be 6 or 7")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.trials <= 0 or args.trials % 2:
        parser.error("--trials must be a positive even integer")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")

    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_position,process_pid,group_count,repeats,"
        "min_cycles,max_cycles,service_ns_per_scan,checksum,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial in range(args.trials):
        cells = CELLS if trial % 2 == 0 else tuple(reversed(CELLS))
        command = [str(args.binary), "--batch"]
        for group_count in cells:
            command.extend((str(group_count), str(REPEATS)))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        _ACTIVE_PROCESS = process
        try:
            stdout, stderr = process.communicate(
                timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out") from error
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial + 1} "
                f"stderr={stderr.rstrip()}",
                file=sys.stderr,
                flush=True,
            )
        if process.returncode != 0:
            if stdout:
                print(
                    f"child_pid={process.pid} trial={trial + 1} "
                    f"stdout={stdout.rstrip()}",
                    file=sys.stderr,
                    flush=True,
                )
            raise RuntimeError(
                f"child PID {process.pid} failed with {process.returncode}")
        rows = list(csv.reader(stdout.strip().splitlines()))
        if len(rows) != len(CELLS) + 1:
            raise RuntimeError("child emitted an incomplete cell matrix")
        for order_position, (fields, expected_group_count) in enumerate(
                zip(rows[1:], cells)):
            if (len(fields) != CSV_FIELDS
                    or int(fields[0]) != expected_group_count
                    or int(fields[1]) != REPEATS
                    or fields[-1] != "pass"):
                raise RuntimeError(f"invalid child row: {fields}")
            print(
                f"{args.run_id},{trial + 1},{order_position},"
                f"{process.pid},{','.join(fields)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
