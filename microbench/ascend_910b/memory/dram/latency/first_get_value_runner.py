#!/usr/bin/env python3
"""Run the frozen Ascend 910B first-GetValue dependency curve."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


BLOCK_DIMS = (1, 2, 4, 8, 16, 24, 48)
REQUEST_COUNTS = (0, 1)
CELLS = tuple(
    (block_dim, request_count)
    for block_dim in BLOCK_DIMS
    for request_count in REQUEST_COUNTS)
CSV_FIELDS = 10
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _cell_index(trial_index, order_position):
    cell_count = len(CELLS)
    return (trial_index + order_position) % cell_count


def _validate_schedule(trials):
    cycle = len(CELLS)
    if trials <= 0 or trials % cycle != 0:
        raise ValueError(f"trials must be a positive multiple of {cycle}")
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
    parser.add_argument("--trials", type=_positive_integer, default=28)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    try:
        _validate_schedule(args.trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "request_count,buffer_bytes,"
        "stride_bytes,node_count,system_cycles,elapsed_ns,sink,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(args.trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [
            str(args.binary), "--batch",
            *(f"{block_dim}:{request_count}"
              for block_dim, request_count in ordered_cells),
        ]
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
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out after "
                f"{args.child_timeout_seconds} seconds") from exc
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}",
                file=sys.stderr,
                flush=True,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with exit "
                f"{process.returncode}: {' '.join(command)}")
        lines = stdout.strip().splitlines()
        expected_lines = sum(block_dim for block_dim, _ in ordered_cells)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child PID {process.pid} emitted {len(lines)} data lines")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            block_dim, request_count = expected
            for worker in range(block_dim):
                line = lines[line_index]
                line_index += 1
                fields = line.split(",")
                if len(fields) != CSV_FIELDS:
                    raise RuntimeError(
                        f"child PID {process.pid} emitted an invalid raw schema")
                if (int(fields[0]) != block_dim
                        or int(fields[1]) != worker
                        or int(fields[2]) != request_count):
                    raise RuntimeError(
                        f"child PID {process.pid} changed the requested order")
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process.pid},{line}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
