#!/usr/bin/env python3
"""Capture the independent key-98513 finite A1/B1 program."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


RUN_COUNT = 3
TRIALS = 16
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_small_m_program_kernel.cpp",
    "aic_mte2_small_m_program_main.cpp",
    "aic_mte2_small_m_program_runner.py",
)
_ACTIVE_PROCESS = None


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        payload = (root / filename).read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _stop_child(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        try:
            os.killpg(_ACTIVE_PROCESS.pid, signal.SIGTERM)
        except ProcessLookupError as error:
            raise SystemExit(128 + signum) from error
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")

    root = Path(__file__).resolve().parent
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} runs={RUN_COUNT} trials={TRIALS} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(root)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,run_index,process_pid,trial,max_cycles,duration_ns,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    for run_index in range(RUN_COUNT):
        process = subprocess.Popen(
            [str(args.binary), "--trials", str(TRIALS)],
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
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            raise RuntimeError(
                f"child PID {process.pid} timed out after "
                f"{args.child_timeout_seconds} seconds") from error
        finally:
            _ACTIVE_PROCESS = None
        if stderr:
            print(
                f"child_pid={process.pid} run_index={run_index} "
                f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with {process.returncode}")
        lines = stdout.strip().splitlines()
        if len(lines) != TRIALS + 1 or lines[0] != (
                "trial,max_cycles,duration_ns,validation"):
            raise RuntimeError("small-M program child emitted invalid output")
        for line in lines[1:]:
            fields = line.split(",")
            if len(fields) != 4 or fields[-1] != "pass":
                raise RuntimeError("small-M program child row is invalid")
            print(
                f"{args.run_id},{run_index},{process.pid},{line}", flush=True)


if __name__ == "__main__":
    main()
