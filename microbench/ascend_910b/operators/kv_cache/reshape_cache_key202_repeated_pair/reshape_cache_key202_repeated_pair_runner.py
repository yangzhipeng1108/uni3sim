#!/usr/bin/env python3
"""Run the frozen key-202 repeated K/V queue-pair campaign."""

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


BLOCK_DIM = 48
PAYLOAD_BYTES = 2048
REPEAT_COUNT = 22
PAIR_ORDERS = (0, 1)
DEFAULT_SCHEDULE_CYCLES = 8
STRIDE_BYTES = 2048
SYSTEM_COUNTER_TICK_NS = 20.0
CHILD_CSV_FIELDS = 14
SOURCE_REVISION = "cann_8.5.1"
SOURCE_CPP_SHA256 = (
    "dda9a4f36ec658ceec13b54dfd26388218f119d2dca42abf533b7d81d38185b8")
SOURCE_BASE_SHA256 = (
    "e80c4a4ca36b6b2a949f72fbfc0f275b938ce8adf90645e288481fe94301da33")
REFERENCE_PATH = (
    Path(__file__).resolve().parents[3] / "references"
    / "reshape_and_cache_cann_8_5_1.json")
REFERENCE_SHA256 = (
    "6bae8fdd0cf09a68c364fe6034d545d8f8d72627e49044aa93dadda98723ba60")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
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


def _validate_schedule(trials):
    if trials <= 0 or trials % len(PAIR_ORDERS) != 0:
        raise ValueError(
            f"trials must be a positive multiple of {len(PAIR_ORDERS)}")


def _parse_child_row(line, expected_pair_order, worker):
    fields = line.split(",")
    if len(fields) != CHILD_CSV_FIELDS:
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
    if (block_dim != BLOCK_DIM or actual_worker != worker
            or payload_bytes != PAYLOAD_BYTES
            or repeat_count != REPEAT_COUNT
            or pair_order != expected_pair_order
            or stride_bytes != STRIDE_BYTES):
        raise RuntimeError("child changed the frozen request geometry")
    if control_cycles < 0 or request_cycles <= 0:
        raise RuntimeError("child emitted invalid cycle counts")
    if (not math.isfinite(control_ns) or not math.isfinite(request_ns)
            or control_ns != control_cycles * SYSTEM_COUNTER_TICK_NS
            or request_ns != request_cycles * SYSTEM_COUNTER_TICK_NS):
        raise RuntimeError("child emitted inconsistent elapsed time")
    if (sink, key_head, value_head, fields[13]) != (
            3, 0x3c00, 0x4000, "pass"):
        raise RuntimeError("child data validation failed")
    return line


def _process_group_exists(process_group):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process):
    if not _process_group_exists(process.pid):
        process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + PROCESS_GROUP_TERM_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process.pid):
            process.wait()
            return
        time.sleep(PROCESS_GROUP_POLL_SECONDS)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        process.wait()
    process.wait()


def _handle_signal(signum, _frame):
    process = _ACTIVE_PROCESS
    if process is not None:
        _terminate_process_group(process)
    raise SystemExit(128 + signum)


def _communicate(command, *, environment, timeout):
    global _ACTIVE_PROCESS

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment, start_new_session=True)
    _ACTIVE_PROCESS = process
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            raise RuntimeError(
                f"child PID {process.pid} timed out after {timeout} seconds"
            ) from exc
        except BaseException:
            _terminate_process_group(process)
            raise
        else:
            _terminate_process_group(process)
    finally:
        _ACTIVE_PROCESS = None
    return process.returncode, process.pid, stdout, stderr


def main():
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
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if not REFERENCE_PATH.is_file() or _sha256(REFERENCE_PATH) != (
            REFERENCE_SHA256):
        parser.error("frozen CANN source reference is missing or changed")
    trials = (
        DEFAULT_SCHEDULE_CYCLES * len(PAIR_ORDERS)
        if args.trials is None else args.trials)
    try:
        _validate_schedule(trials)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"runner_pid={os.getpid()} owner=ascend_kvcache_key202_repeated_pair "
        f"device_id={args.device_id} run_id={args.run_id} "
        f"block_dim={BLOCK_DIM} payload_bytes={PAYLOAD_BYTES} "
        f"repeat_count={REPEAT_COUNT} trials={trials} "
        f"schedule_cycles={trials // len(PAIR_ORDERS)} "
        f"source_revision={SOURCE_REVISION} "
        f"source_cpp_sha256={SOURCE_CPP_SHA256} "
        f"source_base_sha256={SOURCE_BASE_SHA256} "
        f"reference_sha256={REFERENCE_SHA256} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,order_position,process_pid,block_dim,worker,"
        "payload_bytes,repeat_count,pair_order,stride_bytes,control_cycles,"
        "request_cycles,control_ns,request_ns,sink,key_head,value_head,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for managed_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(managed_signal, _handle_signal)
    for trial_index in range(trials):
        ordered = tuple(
            PAIR_ORDERS[(trial_index + position) % len(PAIR_ORDERS)]
            for position in range(len(PAIR_ORDERS)))
        command = [str(args.binary), "--batch", *(str(cell) for cell in ordered)]
        returncode, process_pid, stdout, stderr = _communicate(
            command, environment=environment,
            timeout=args.child_timeout_seconds)
        if stderr:
            print(
                f"child_pid={process_pid} trial={trial_index + 1} "
                f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
        if returncode != 0:
            raise RuntimeError(
                f"child PID {process_pid} failed with exit {returncode}")
        lines = stdout.strip().splitlines()
        expected_lines = BLOCK_DIM * len(PAIR_ORDERS)
        if len(lines) != expected_lines:
            raise RuntimeError(
                f"child PID {process_pid} emitted {len(lines)} data lines; "
                f"expected {expected_lines}")
        line_index = 0
        for order_position, pair_order in enumerate(ordered):
            for worker in range(BLOCK_DIM):
                validated = _parse_child_row(
                    lines[line_index], pair_order, worker)
                line_index += 1
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process_pid},{validated}", flush=True)


if __name__ == "__main__":
    main()
