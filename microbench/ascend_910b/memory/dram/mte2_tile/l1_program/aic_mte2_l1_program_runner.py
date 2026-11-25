#!/usr/bin/env python3
"""Run one independent 910B source-shaped A1/B1 MTE2 program capture."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_l1_program_kernel.cpp",
    "aic_mte2_l1_program_main.cpp",
    "aic_mte2_l1_program_runner.py",
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
RAW_FIELDS = (
    "block_dim", "short_programs", "long_programs", "a_descriptors",
    "b_descriptors", "a_descriptor_bytes", "b_descriptor_bytes",
    "l1_bytes", "program_bytes", "short_cycles", "short_wall_ns",
    "long_cycles", "long_wall_ns", "tick_ns", "ns_per_program",
    "bandwidth_gbs", "validation",
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
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {path}")
        payload = path.read_bytes()
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
        except ProcessLookupError:
            raise SystemExit(128 + signum)
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--short-programs", type=int, default=8)
    parser.add_argument("--long-programs", type=int, default=32)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.short_programs <= 0 or args.long_programs <= args.short_programs:
        parser.error("program counts must be positive and increasing")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")

    benchmark_root = Path(__file__).resolve().parent
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(benchmark_root)}",
        file=sys.stderr,
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    process = subprocess.Popen(
        [
            str(args.binary), "--short-programs", str(args.short_programs),
            "--long-programs", str(args.long_programs),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    _ACTIVE_PROCESS = process
    try:
        stdout, stderr = process.communicate(timeout=args.child_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        raise RuntimeError(
            f"child PID {process.pid} timed out after "
            f"{args.child_timeout_seconds} seconds") from error
    finally:
        _ACTIVE_PROCESS = None
    if stderr:
        print(f"child_pid={process.pid} stderr={stderr.rstrip()}",
              file=sys.stderr, flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"child PID {process.pid} failed with exit {process.returncode}")
    lines = stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError(
            f"child PID {process.pid} emitted {len(lines)} data lines")
    fields = lines[0].split(",")
    if len(fields) != len(RAW_FIELDS) or fields[-1] != "pass":
        raise RuntimeError(f"child PID {process.pid} emitted an invalid raw schema")
    integers = [int(value) for value in fields[:-4]]
    if integers[0] != 24 or integers[1:3] != [args.short_programs, args.long_programs]:
        raise RuntimeError(f"child PID {process.pid} changed requested geometry")
    print(",".join((args.run_id, *fields)), flush=True)


if __name__ == "__main__":
    main()
