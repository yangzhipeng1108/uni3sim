#!/usr/bin/env python3
"""Run fresh-process CANN GroupedMatmul controller-first-issue captures."""

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


ROW_FIELDS = 7
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


def _handle_signal(_signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128)


def _parse_counts(path):
    with path.open() as stream:
        counts = json.load(stream)
    if (not isinstance(counts, list) or not counts
            or len(counts) > 512
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
    parser.add_argument("--projection-k", required=True, type=int)
    parser.add_argument("--projection-n", required=True, type=int)
    parser.add_argument("--base-m", required=True, type=int)
    parser.add_argument("--base-n", required=True, type=int)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id not in (2, 6, 7):
        parser.error("--device-id must be one of 2, 6, 7")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.trials < 3:
        parser.error("--trials must be at least three fresh processes")
    if args.repeats != 1:
        parser.error("--repeats must be exactly one for a source-pass probe")
    if any(value <= 0 for value in (
            args.projection_k, args.projection_n, args.base_m, args.base_n)):
        parser.error("the source dimensions must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    counts = _parse_counts(args.counts_json)
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} trials={args.trials} repeats={args.repeats} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"counts_sha256={_sha256(args.counts_json)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,process_pid,projection_k,projection_n,base_m,base_n,"
        "worker,first_issue_cycles,completion_cycles,first_issue_ns,"
        "completion_ns,checksum,validation", flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    command = [
        str(args.binary), "--case", str(len(counts)), str(args.projection_k),
        str(args.projection_n), str(args.base_m), str(args.base_n),
        str(args.repeats), *(str(value) for value in counts)]
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
        for expected_worker, fields in enumerate(rows[1:]):
            if (len(fields) != ROW_FIELDS or int(fields[0]) != expected_worker
                    or int(fields[1]) <= 0 or int(fields[2]) < int(fields[1])
                    or fields[-1] != "pass"):
                raise RuntimeError(f"invalid child row: {fields}")
            print(
                f"{args.run_id},{trial},{process.pid},{args.projection_k},"
                f"{args.projection_n},{args.base_m},{args.base_n},"
                f"{','.join(fields)}", flush=True)


if __name__ == "__main__":
    main()
