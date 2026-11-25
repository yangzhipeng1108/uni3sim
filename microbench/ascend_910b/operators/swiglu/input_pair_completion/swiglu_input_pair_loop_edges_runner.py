#!/usr/bin/env python3
"""Capture the independent 48-AIV q2 CopyIn first/steady edge primitive."""

import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys


BLOCK_DIM = 48
PAYLOAD_BYTES = 18432
REPETITIONS = (1, 2)
PAIR_ORDERS = (0, 1)
CELLS = tuple(
    (repetitions, pair_order)
    for repetitions in REPETITIONS
    for pair_order in PAIR_ORDERS)
DEFAULT_TRIALS = 32
EXPECTED_SINK_PER_PAIR = 0xB4
EXPECTED_CHILD_FIELDS = 14
ACTIVE_PROCESS = None


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


def _cell_index(trial_index, position):
    return (position + trial_index) % len(CELLS)


def _terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_signal(_signum, _frame):
    if ACTIVE_PROCESS is not None:
        _terminate(ACTIVE_PROCESS)
    raise SystemExit(130)


def _parse_row(line, repetitions, pair_order, worker):
    fields = line.split(",")
    if len(fields) != EXPECTED_CHILD_FIELDS:
        raise RuntimeError("child row has an invalid field count")
    values = tuple(int(field) for field in fields[:8])
    (
        block_dim, observed_worker, observed_repetitions, observed_order,
        control_first_cycles, control_steady_cycles, request_first_cycles,
        request_steady_cycles,
    ) = values
    timing_ns = tuple(float(field) for field in fields[8:12])
    sink = int(fields[12])
    if (
            block_dim != BLOCK_DIM
            or observed_worker != worker
            or observed_repetitions != repetitions
            or observed_order != pair_order
            or control_first_cycles < 0
            or request_first_cycles <= 0
            or (repetitions == 1
                and (control_steady_cycles != 0 or request_steady_cycles != 0))
            or (repetitions == 2
                and (control_steady_cycles < 0 or request_steady_cycles <= 0))
            or sink != repetitions * EXPECTED_SINK_PER_PAIR
            or fields[13] != "pass"):
        raise RuntimeError("child row failed device identity validation")
    expected_ns = tuple(
        cycles * 20.0
        for cycles in (
            control_first_cycles, control_steady_cycles,
            request_first_cycles, request_steady_cycles))
    if timing_ns != expected_ns:
        raise RuntimeError("child row has inconsistent system-counter units")
    return line


def main():
    global ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=_positive_integer, default=DEFAULT_TRIALS)
    parser.add_argument("--child-timeout-seconds", type=_positive_integer,
                        default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if args.trials % len(CELLS) != 0:
        parser.error("--trials must be a multiple of four")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an executable file")

    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "payload_bytes,repetitions,pair_order,control_first_cycles,"
        "control_steady_cycles,request_first_cycles,request_steady_cycles,"
        "control_first_ns,control_steady_ns,request_first_ns,"
        "request_steady_ns,sink,validation",
        flush=True)
    print(
        f"owner=capture-owner-4c0f2645f650 physical_device={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"binary={binary} binary_sha256={_sha256(binary)}",
        file=sys.stderr, flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(args.trials):
        for position in range(len(CELLS)):
            repetitions, pair_order = CELLS[_cell_index(trial_index, position)]
            process = subprocess.Popen(
                [str(binary), str(BLOCK_DIM), str(repetitions),
                 str(pair_order + 1)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=environment, start_new_session=True)
            ACTIVE_PROCESS = process
            try:
                stdout, stderr = process.communicate(
                    timeout=args.child_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate(process)
                raise RuntimeError(
                    f"child timed out in trial {trial_index + 1}") from exc
            finally:
                if process.poll() is None:
                    _terminate(process)
                ACTIVE_PROCESS = None
            if process.returncode != 0:
                raise RuntimeError(
                    f"child failed trial={trial_index + 1} position={position}: "
                    f"{stderr.rstrip()}")
            rows = tuple(line for line in stdout.splitlines() if line)
            if len(rows) != BLOCK_DIM:
                raise RuntimeError("child did not return one row per AIV")
            for worker, row in enumerate(rows):
                _parse_row(row, repetitions, pair_order, worker)
                child_fields = row.split(",")
                print(
                    f"{args.run_id},{trial_index + 1},{position},"
                    f"{process.pid},{BLOCK_DIM},{worker},{PAYLOAD_BYTES},"
                    f"{repetitions},{pair_order},"
                    f"{','.join(child_fields[4:])}",
                    flush=True)


if __name__ == "__main__":
    main()
