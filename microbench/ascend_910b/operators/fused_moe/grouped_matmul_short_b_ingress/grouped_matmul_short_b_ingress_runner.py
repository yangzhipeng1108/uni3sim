#!/usr/bin/env python3
"""Run fresh-process short-GroupedMatmul B-ingress source-ring captures."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


_ACTIVE_PROCESS = None
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ROW_FIELDS = 7


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


def _handle_signal(_signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128)


def _parse_counts(path):
    with path.open() as stream:
        counts = json.load(stream)
    if (not isinstance(counts, list) or not counts or len(counts) > 512
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value < 0 for value in counts)
            or sum(counts) <= 0):
        raise ValueError("counts JSON must be a non-empty non-negative list")
    return tuple(counts)


def main():
    global _ACTIVE_PROCESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--counts-json", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id not in (2, 6, 7):
        parser.error("--device-id must be one of 2, 6, 7")
    if _RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.trials < 3:
        parser.error("--trials must be at least three fresh processes")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    counts = _parse_counts(args.counts_json)
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"counts_sha256={_sha256(args.counts_json)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,process_pid,worker,completion_cycles,completion_ns,"
        "issued_b_groups,ring_offset,group_count,validation",
        flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    command = [
        str(args.binary), "--case", str(len(counts)),
        *(str(value) for value in counts)]
    for trial in range(1, args.trials + 1):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=environment, start_new_session=True)
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
            print(
                f"child_pid={process.pid} trial={trial} stderr={stderr.rstrip()}",
                file=sys.stderr, flush=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with {process.returncode}")
        rows = list(csv.reader(stdout.strip().splitlines()))
        if len(rows) != 25:
            raise RuntimeError("child emitted an incomplete 24-AIC matrix")
        for worker, fields in enumerate(rows[1:]):
            if (len(fields) != _ROW_FIELDS or int(fields[0]) != worker
                    or int(fields[1]) <= 0 or int(fields[3]) < 0
                    or fields[-1] != "pass"):
                raise RuntimeError(f"invalid child row: {fields}")
            print(
                f"{args.run_id},{trial},{process.pid},{','.join(fields)}",
                flush=True)


if __name__ == "__main__":
    main()
