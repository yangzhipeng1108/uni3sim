#!/usr/bin/env python3
"""Capture a source-independent q2 A/B row-stride geometry matrix.

This is a hardware mechanism sweep, not a production-operator benchmark.  Its
fixed Cartesian grid excludes case IDs, tiling keys and production durations;
raw cells cannot be used as Forward services or lookups.
"""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


ACTIVE_WORKERS = (1, 4, 16, 32, 48)
ROW_COUNTS = (1, 2, 4)
ROW_BYTES = (2048, 8192)
STRIDE_RATIOS = (1, 4, 16, 32)
BASE_PHASE_BYTES = (0, 512)
PAIR_ORDERS = (0, 1)
QUEUE_DEPTH = 2
MINIMUM_SAMPLES_PER_CELL = 8
REQUIRED_INDEPENDENT_RUNS = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
EXPECTED_FIELDS = 13
EXPECTED_SINK = 0xB4
_ACTIVE_PROCESS = None

BASE_CELLS = tuple(
    (workers, row_count, row_bytes, stride_ratio, base_phase)
    for workers in ACTIVE_WORKERS
    for row_count in ROW_COUNTS
    for row_bytes in ROW_BYTES
    for stride_ratio in STRIDE_RATIOS
    for base_phase in BASE_PHASE_BYTES
)
CELLS = tuple(
    (*cell, pair_order)
    for cell in BASE_CELLS
    for pair_order in PAIR_ORDERS
)


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
    # 137 is coprime to the 480 ordered cells.  Each cell rotates through a
    # different host position on every required replication trial.
    return (position + 137 * (trial - 1)) % len(CELLS)


def _validate_schedule(trials):
    if trials < MINIMUM_SAMPLES_PER_CELL:
        raise ValueError(
            f"trials must provide at least {MINIMUM_SAMPLES_PER_CELL} "
            "samples per fixed geometry")
    for cell in CELLS:
        positions = {
            position
            for trial in range(1, trials + 1)
            for position in range(len(CELLS))
            if CELLS[_cell_index(trial, position)] == cell
        }
        if len(positions) != trials:
            raise AssertionError("fixed geometry did not rotate by host order")


def _parse_child_row(line, expected, worker):
    fields = line.split(",")
    if len(fields) != EXPECTED_FIELDS:
        raise RuntimeError("child row has an invalid field count")
    integers = tuple(int(fields[index]) for index in range(9))
    (block_dim, observed_worker, row_count, row_bytes, stride_bytes,
     base_phase, pair_order, control_cycles, request_cycles) = integers
    control_ns = float(fields[9])
    request_ns = float(fields[10])
    sink = int(fields[11])
    (expected_workers, expected_rows, expected_row_bytes,
     expected_stride_ratio, expected_phase, expected_order) = expected
    if (block_dim != expected_workers or observed_worker != worker
            or row_count != expected_rows or row_bytes != expected_row_bytes
            or stride_bytes != expected_row_bytes * expected_stride_ratio
            or base_phase != expected_phase or pair_order != expected_order
            or control_cycles < 0 or request_cycles <= 0
            or control_ns != control_cycles * 20.0
            or request_ns != request_cycles * 20.0
            or sink != EXPECTED_SINK or fields[12] != "pass"):
        raise RuntimeError("child row failed strict geometry validation")
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
        "--child-timeout-seconds", type=_positive_integer, default=1800)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an executable file")
    trials = MINIMUM_SAMPLES_PER_CELL if args.trials is None else args.trials
    try:
        _validate_schedule(trials)
    except ValueError as error:
        parser.error(str(error))

    print(
        f"runner_pid={os.getpid()} owner=vector_swiglu_q2_geometry "
        f"physical_device_id={args.device_id} run_id={args.run_id} "
        f"trials={trials} samples_per_cell={trials} "
        f"required_independent_runs={REQUIRED_INDEPENDENT_RUNS} "
        f"active_workers={ACTIVE_WORKERS} row_counts={ROW_COUNTS} "
        f"row_bytes={ROW_BYTES} stride_ratios={STRIDE_RATIOS} "
        f"base_phase_bytes={BASE_PHASE_BYTES} queue_depth={QUEUE_DEPTH} "
        f"binary={binary} binary_sha256={_sha256(binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "row_count,row_bytes,stride_bytes,base_phase_bytes,pair_order,"
        "control_cycles,request_cycles,control_ns,request_ns,sink,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial in range(1, trials + 1):
        ordered_cells = tuple(
            CELLS[_cell_index(trial, position)]
            for position in range(len(CELLS)))
        command = [
            str(binary), "--batch",
            *(f"{workers}:{row_count}:{row_bytes}:{stride_ratio}:"
              f"{base_phase}:{pair_order}"
              for (workers, row_count, row_bytes, stride_ratio,
                   base_phase, pair_order) in ordered_cells),
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
            raise RuntimeError(
                f"child PID {process.pid} timed out") from error
        finally:
            if process.poll() is None:
                _terminate_process_group(process)
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} trial={trial} "
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
        for position, expected in enumerate(ordered_cells):
            for worker in range(expected[0]):
                line = _parse_child_row(lines[line_index], expected, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial},{position},{process.pid},{line}",
                    flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
