#!/usr/bin/env python3
"""Run exact M16/M64 GroupedMatmul admission captures in fresh processes."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


_RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ISSUES = (1, 2, 4, 8, 16, 32)
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


def main():
    global _ACTIVE_PROCESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--program", required=True, choices=("m16", "m64"))
    parser.add_argument("--projection", required=True, choices=("w1", "w2"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id not in (2, 6, 7):
        parser.error("--device-id must be one of 2, 6, 7")
    if _RUN_ID.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.trials < 3:
        parser.error("--trials must be at least three fresh processes")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"program={args.program} projection={args.projection} run_id={args.run_id} trials={args.trials} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,trial,process_pid,worker,program,projection,issue_index,post_cycles,"
        "post_ns,base_m,base_n,base_k,depth_a1,depth_b1,step_ka,step_kb,"
        "db_l0a,db_l0b,db_l0c,validation", flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for issues in _ISSUES:
        for trial in range(1, args.trials + 1):
            command = [str(args.binary), "--program", args.program,
                       "--projection", args.projection,
                       "--issues", str(issues)]
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
                    f"child PID {process.pid} timed out at issues={issues}") from error
            finally:
                if process.poll() is None:
                    _terminate_process_group(process)
                _ACTIVE_PROCESS = None
            if stderr:
                print(
                    f"child_pid={process.pid} trial={trial} issues={issues} "
                    f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
            if process.returncode != 0:
                raise RuntimeError(
                    f"child PID {process.pid} failed with {process.returncode}")
            rows = list(csv.reader(stdout.strip().splitlines()))
            expected_rows = 1 + 24 * issues
            if len(rows) != expected_rows:
                raise RuntimeError(
                    f"issues={issues} emitted {len(rows)} rows, expected {expected_rows}")
            for expected_worker in range(24):
                for issue in range(issues):
                    fields = rows[1 + expected_worker * issues + issue]
                    expected_program = (["16", "256", "64", "120", "8",
                                         "60", "4", "2", "2", "1"]
                                        if args.program == "m16" else
                                        ["64", "256", "64", "24", "8",
                                         "12", "4", "2", "2", "1"])
                    if (len(fields) != 17 or int(fields[0]) != expected_worker
                            or fields[1] != args.program
                            or fields[2] != args.projection
                            or int(fields[3]) != issue or int(fields[4]) <= 0
                            or fields[6:16] != expected_program
                            or fields[16] != "pass"):
                        raise RuntimeError(f"invalid child row: {fields}")
                    print(
                        f"{args.run_id},{trial},{process.pid},{','.join(fields)}",
                        flush=True)


if __name__ == "__main__":
    main()
