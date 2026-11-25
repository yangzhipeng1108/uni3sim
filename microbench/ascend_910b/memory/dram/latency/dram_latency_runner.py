#!/usr/bin/env python3
"""Run the frozen Ascend 910B AIV scalar-memory latency sweep."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


CELLS = (
    ("l2", 1, 64, 131_072),
    ("dram", 512, 4096, 262_144),
)
CSV_FIELDS = 17
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _cell_index(trial_index, order_position):
    if trial_index % 2 == 0:
        return order_position
    return len(CELLS) - 1 - order_position


def _validate_schedule(trials):
    if trials <= 0 or trials % (2 * len(CELLS)) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {2 * len(CELLS)}")
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
    parser.add_argument("--trials", type=_positive_integer, default=24)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=900)
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
        f"binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_position,process_pid,working_set,buffer_mb,"
        "buffer_bytes,stride_bytes,node_count,short_hops,long_hops,short_cycles,"
        "short_wall_ns,long_cycles,long_wall_ns,tick_ns,latency_cycles,"
        "latency_ns,short_sink,long_sink,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(args.trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, order_position)]
            for order_position in range(len(CELLS)))
        command = [str(args.binary), "--batch"]
        for working_set, buffer_mb, stride_bytes, hops in ordered_cells:
            command.extend((
                working_set, str(buffer_mb), str(stride_bytes), str(hops)))
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
        if len(lines) != len(CELLS):
            raise RuntimeError(
                f"child PID {process.pid} emitted {len(lines)} data lines")
        for order_position, (line, expected) in enumerate(
                zip(lines, ordered_cells)):
            fields = line.split(",")
            if len(fields) != CSV_FIELDS:
                raise RuntimeError(
                    f"child PID {process.pid} emitted an invalid raw schema")
            working_set, buffer_mb, stride_bytes, hops = expected
            if (fields[0] != working_set
                    or int(fields[1]) != buffer_mb
                    or int(fields[3]) != stride_bytes
                    or int(fields[5]) != hops):
                raise RuntimeError(
                    f"child PID {process.pid} changed the requested order")
            print(
                f"{args.run_id},{trial_index + 1},{order_position},"
                f"{process.pid},{line}",
                flush=True,
            )


if __name__ == "__main__":
    main()
