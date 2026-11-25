#!/usr/bin/env python3
"""Run the frozen 48-AIV repeated key-200 chain campaign."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


BLOCK_DIM = 48
SUPPORTED_PAYLOAD_BYTES = (1024, 2048)
DEFAULT_PAYLOAD_BYTES = 2048
REPEAT_COUNTS = (1, 2, 6, 22)
PAIR_ORDERS = (0, 1)
CELLS = tuple(
    (repeat_count, pair_order)
    for repeat_count in REPEAT_COUNTS
    for pair_order in PAIR_ORDERS)
DEFAULT_SCHEDULE_CYCLES = 8
STRIDE_BYTES = 4096
SYSTEM_COUNTER_TICK_NS = 20.0
CSV_FIELDS = 14
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")


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
        for order_position in range(len(CELLS)):
            counts[order_position][
                _cell_index(trial_index, order_position)] += 1
    expected = trials // len(CELLS)
    if any(count != expected for row in counts for count in row):
        raise AssertionError("cell/order schedule is not balanced")


def _parse_child_row(line, expected, worker, expected_payload_bytes):
    fields = line.split(",")
    if len(fields) != CSV_FIELDS:
        raise RuntimeError("child emitted an invalid schema")
    try:
        integers = tuple(int(fields[index]) for index in range(8))
        control_ns = float(fields[8])
        request_ns = float(fields[9])
        sink = int(fields[10])
        key_head = int(fields[11])
        value_head = int(fields[12])
    except ValueError as exc:
        raise RuntimeError("child emitted a non-numeric value") from exc
    (block_dim, actual_worker, payload_bytes, repeat_count, pair_order,
     stride_bytes, control_cycles, request_cycles) = integers
    if (repeat_count, pair_order) != expected:
        raise RuntimeError("child changed requested cell")
    if (block_dim != BLOCK_DIM or payload_bytes != expected_payload_bytes
            or actual_worker != worker or stride_bytes != STRIDE_BYTES):
        raise RuntimeError("child changed frozen geometry")
    if control_cycles < 0 or request_cycles <= 0:
        raise RuntimeError("child emitted invalid cycle counts")
    if (not math.isfinite(control_ns) or not math.isfinite(request_ns)
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or request_ns != request_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child emitted inconsistent elapsed time")
    if (sink, key_head, value_head, fields[13]) != (
            3, 0x3C00, 0x4000, "pass"):
        raise RuntimeError("child data validation failed")
    return line


def _run_child(command, environment, timeout):
    process = subprocess.Popen(
        command, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    return process.returncode, process.pid, stdout, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--payload-bytes", type=int, choices=SUPPORTED_PAYLOAD_BYTES,
        default=DEFAULT_PAYLOAD_BYTES)
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
        f"runner_pid={os.getpid()} owner=capture-owner-4c0f2645f650 "
        f"device_id={args.device_id} run_id={args.run_id} "
        f"payload_bytes={args.payload_bytes} trials={trials} "
        f"schedule_cycles={trials // len(CELLS)} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "payload_bytes,repeat_count,pair_order,stride_bytes,control_cycles,"
        "request_cycles,control_ns,request_ns,sink,key_destination_head,"
        "value_destination_head,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for trial_index in range(trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [
            str(args.binary), "--payload-bytes", str(args.payload_bytes),
            "--batch",
            *(f"{repeat_count}:{pair_order}"
              for repeat_count, pair_order in ordered_cells),
        ]
        returncode, process_pid, stdout, stderr = _run_child(
            command, environment, args.child_timeout_seconds)
        if stderr:
            print(
                f"child_pid={process_pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}",
                file=sys.stderr, flush=True)
        if returncode != 0:
            raise RuntimeError(
                f"child PID {process_pid} failed with exit {returncode}: "
                f"{' '.join(command)}")
        lines = stdout.strip().splitlines()
        expected_lines = len(CELLS) * BLOCK_DIM
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child emitted {len(lines)} data lines; expected "
                f"{expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            for worker in range(BLOCK_DIM):
                validated = _parse_child_row(
                    lines[line_index], expected, worker,
                    args.payload_bytes)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process_pid},{validated}", flush=True)


if __name__ == "__main__":
    main()
