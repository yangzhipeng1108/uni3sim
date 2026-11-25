#!/usr/bin/env python3
"""Run the source-ordered CANN SwiGLU input-pair completion campaign."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


# Cells are selected from source-derived production tilings already covered by
# independently published single-input completion controls. The 24-worker cell
# is a representative admission control; the 40/48-worker cells are production
# domains. Queue depth is a discrete host-tiler fact.
BASE_CELLS = (
    (24, 19456, 2),
    (40, 17920, 2),
    (48, 18432, 2),
    (48, 19456, 2),
    (48, 26624, 1),
)
PAIR_ORDERS = (0, 1)
CELLS = tuple(
    (*cell, pair_order)
    for cell in BASE_CELLS
    for pair_order in PAIR_ORDERS
)
DEFAULT_SCHEDULE_CYCLES = 8
REQUIRED_INDEPENDENT_RUNS = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
EXPECTED_FIELDS = 12
EXPECTED_SINK = 0xB4
_ACTIVE_PROCESS = None


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_index(trial, position):
    # Three is coprime to the ten-cell schedule, so complete schedule cycles
    # balance every cell across every process-local order position.
    return (position + 3 * trial) % len(CELLS)


def _validate_schedule(trials):
    if trials % len(CELLS) != 0:
        raise ValueError(f"trials must be a multiple of {len(CELLS)}")
    for position in range(len(CELLS)):
        counts = {
            cell: sum(
                CELLS[_cell_index(trial, position)] == cell
                for trial in range(trials))
            for cell in CELLS
        }
        if len(set(counts.values())) != 1:
            raise ValueError("campaign schedule is not position-balanced")


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != EXPECTED_FIELDS:
        raise RuntimeError("child row has an invalid field count")
    integers = tuple(int(fields[index]) for index in range(8))
    block_dim, observed_worker, payload_bytes, queue_depth, pair_order = (
        integers[:5])
    stride_bytes, control_cycles, request_cycles = integers[5:]
    control_ns = float(fields[8])
    request_ns = float(fields[9])
    sink = int(fields[10])
    if ((block_dim, payload_bytes, queue_depth, pair_order) != expected
            or observed_worker != worker
            or stride_bytes < payload_bytes or stride_bytes % 32 != 0
            or control_cycles < 0 or request_cycles <= 0
            or control_ns != control_cycles * 20.0
            or request_ns != request_cycles * 20.0
            or sink != EXPECTED_SINK or fields[11] != "pass"):
        raise RuntimeError("child row failed strict identity validation")
    return line


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
    parser.add_argument("--trials", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an executable file")
    trials = (
        DEFAULT_SCHEDULE_CYCLES * len(CELLS)
        if args.trials is None else args.trials)
    try:
        _validate_schedule(trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} owner=vector_swiglu_input_pair "
        f"physical_device_id={args.device_id} run_id={args.run_id} "
        f"trials={trials} schedule_cycles={trials // len(CELLS)} "
        f"required_independent_runs={REQUIRED_INDEPENDENT_RUNS} "
        f"binary={binary} binary_sha256={_sha256(binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "payload_bytes,queue_depth,pair_order,stride_bytes,control_cycles,"
        "request_cycles,control_ns,request_ns,sink,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(trials):
        ordered_cells = tuple(
            CELLS[_cell_index(trial_index, position)]
            for position in range(len(CELLS)))
        command = [
            str(binary), "--batch",
            *(f"{block_dim}:{payload_bytes}:{queue_depth}:{pair_order}"
              for block_dim, payload_bytes, queue_depth, pair_order
              in ordered_cells),
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True)
        _ACTIVE_PROCESS = process
        try:
            stdout, stderr = process.communicate(
                timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out") from exc
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with "
                f"exit {process.returncode}")
        lines = stdout.strip().splitlines()
        expected_lines = sum(cell[0] for cell in ordered_cells)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child emitted {len(lines)} rows; expected {expected_lines}")
        line_index = 0
        for order_position, expected in enumerate(ordered_cells):
            for worker in range(expected[0]):
                line = _parse_child_row(
                    lines[line_index], expected, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process.pid},{line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
