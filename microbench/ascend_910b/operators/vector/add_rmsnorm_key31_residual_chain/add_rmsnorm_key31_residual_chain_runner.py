#!/usr/bin/env python3
"""Run the exact AddRMSNorm key-31 residual-reload chain probe."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


BLOCK_DIM = 47
BLOCK_FACTOR = 22
LAST_BLOCK_FACTOR = 12
TILES_PER_ROW = 2
REQUEST_BYTES = 16384
MAXIMUM_REQUESTS = 44
PAIR_ORDERS = (0, 1)
DEFAULT_TRIALS = 120
REQUIRED_INDEPENDENT_RUNS = 3
SYSTEM_COUNTER_TICK_NS = 20.0
SOURCE_SHA256 = "9ff6a5cde40fbaa7948de058a5172ebc80824ab985531af8103eb34ed4b9e64a"
BASE_SOURCE_SHA256 = "a2ece498a72530c9163330d3631676ed5879af99ad16ea3e8f37a2d8838dae93"
TQUE_INTERFACE_SHA256 = "f4e856e057f1ce3e3268184d17bf27c3cd4f23b8a8a1e167c5902d5e94adec56"
TQUE_IMPL_SHA256 = "6d084e33a64889a54586785eba80768cd21f038212924384c7e67c9d9f194b73"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
CHILD_FIELDS = 12
TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
PROCESS_GROUP_TERM_TIMEOUT_SECONDS = 10.0
PROCESS_GROUP_POLL_SECONDS = 0.01
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


def _ordered_pair_orders(trial_index):
    return PAIR_ORDERS if trial_index % 2 == 0 else tuple(reversed(PAIR_ORDERS))


def _request_offsets_elements(row_count):
    return tuple(
        row * 16384 + tile * 8192
        for tile in range(TILES_PER_ROW)
        for row in range(row_count))


def _process_group_exists(process_group):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_child(process):
    if _process_group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError as error:
            if _process_group_exists(process.pid):
                raise RuntimeError(
                    "process group remained after SIGTERM lookup failure") \
                    from error
        deadline = time.monotonic() + PROCESS_GROUP_TERM_TIMEOUT_SECONDS
        while (_process_group_exists(process.pid)
               and time.monotonic() < deadline):
            process.poll()
            time.sleep(PROCESS_GROUP_POLL_SECONDS)
        if _process_group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError as error:
                if _process_group_exists(process.pid):
                    raise RuntimeError(
                        "process group remained after SIGKILL lookup failure") \
                        from error
    process.wait()


def _handle_signal(signum, _frame):
    process = _ACTIVE_PROCESS
    if process is not None:
        _terminate_child(process)
    raise SystemExit(128 + signum)


def _run_child(command, environment, timeout):
    global _ACTIVE_PROCESS

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=environment, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, process.pid, stdout, stderr
    finally:
        try:
            if process.poll() is None or _process_group_exists(process.pid):
                _terminate_child(process)
            else:
                process.wait()
        finally:
            _ACTIVE_PROCESS = None


def _validate_trials(trials):
    if trials < DEFAULT_TRIALS or trials % 2 != 0:
        raise ValueError("trials must be an even integer of at least 120")


def _parse_child_row(line, expected_order, worker):
    fields = line.split(",")
    if len(fields) != CHILD_FIELDS:
        raise RuntimeError("child emitted an invalid schema")
    try:
        integers = tuple(int(fields[index]) for index in range(7))
        control_ns = float(fields[7])
        request_ns = float(fields[8])
        control_sink = int(fields[9])
        request_sink = int(fields[10])
    except ValueError as exc:
        raise RuntimeError("child emitted non-numeric data") from exc
    (block_dim, actual_worker, pair_order, request_count, request_bytes,
     control_cycles, request_cycles) = integers
    expected_count = LAST_BLOCK_FACTOR * TILES_PER_ROW if worker == 46 else MAXIMUM_REQUESTS
    if (block_dim != BLOCK_DIM or actual_worker != worker
            or pair_order != expected_order or request_count != expected_count
            or request_bytes != REQUEST_BYTES or control_cycles <= 0
            or request_cycles <= 0 or control_sink != 1 or request_sink != 2
            or fields[11] != "pass" or not math.isfinite(control_ns)
            or not math.isfinite(request_ns)
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or request_ns != request_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child changed the frozen exact-cell contract")
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--base-source-file", required=True, type=Path)
    parser.add_argument("--tque-interface", required=True, type=Path)
    parser.add_argument("--tque-impl", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int, choices=(6, 7))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=_positive_integer, default=DEFAULT_TRIALS)
    parser.add_argument("--child-timeout-seconds", type=_positive_integer, default=300)
    args = parser.parse_args()
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("invalid --run-id")
    try:
        _validate_trials(args.trials)
    except ValueError as exc:
        parser.error(str(exc))
    identities = (
        (args.source_file, SOURCE_SHA256),
        (args.base_source_file, BASE_SOURCE_SHA256),
        (args.tque_interface, TQUE_INTERFACE_SHA256),
        (args.tque_impl, TQUE_IMPL_SHA256),
    )
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if any(not path.is_file() or _sha256(path) != expected for path, expected in identities):
        parser.error("fixed CANN 8.5.1 source identity changed")
    print(
        f"runner_pid={os.getpid()} owner=add_rmsnorm_key31_residual_chain "
        f"physical_device_id={args.device_id} run_id={args.run_id} "
        f"baseline=1aac4ffdb463f4a5462abf99321aa3ac53a248c8 "
        f"trials={args.trials} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,pair_order,"
        "request_count,request_bytes,control_cycles,request_cycles,control_ns,"
        "request_ns,control_sink,request_sink,validation", flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    schedule = tuple(
        order
        for trial_index in range(args.trials)
        for order in _ordered_pair_orders(trial_index))
    command = [str(args.binary), "--batch", *(str(order) for order in schedule)]
    returncode, process_pid, stdout, stderr = _run_child(
        command, environment, args.child_timeout_seconds)
    if stderr:
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    if returncode != 0:
        raise RuntimeError(f"child failed with exit {returncode}")
    lines = stdout.strip().splitlines()
    expected_lines = args.trials * BLOCK_DIM * len(PAIR_ORDERS)
    if len(lines) != expected_lines:
        raise RuntimeError(
            f"child emitted {len(lines)} rows; expected {expected_lines}")
    line_index = 0
    for trial_index in range(args.trials):
        orders = _ordered_pair_orders(trial_index)
        for position, order in enumerate(orders):
            for worker in range(BLOCK_DIM):
                line = _parse_child_row(lines[line_index], order, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{position},"
                    f"{process_pid},{line}", flush=True)


if __name__ == "__main__":
    for managed_signal in TERMINATION_SIGNALS:
        signal.signal(managed_signal, _handle_signal)
    main()
