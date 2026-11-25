#!/usr/bin/env python3
"""Measure finite BF16 FixPipe streams used by CANN GEMM tile programs."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


RUN_COUNT = 3
FINITE_SHORT_INVOCATIONS = 32
FINITE_LONG_INVOCATIONS = 96
LONG_STREAM_SHORT_INVOCATIONS = 96
LONG_STREAM_LONG_INVOCATIONS = 1195
CELLS = (
    (0, 16, 176, 22),
    (0, 32, 256, 16),
    (0, 32, 256, 20),
    (0, 32, 256, 22),
    (0, 48, 256, 22),
    (0, 64, 256, 20),
    (0, 64, 256, 22),
    (0, 128, 256, 24),
)
LONG_STREAM_CELLS = ((0, 128, 256, 24),)
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "gemm_v3_store_contract.h",
    "gemm_v3_store_kernel.cpp",
    "gemm_v3_store_main.cpp",
    "gemm_v3_store_finite_runner.py",
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OWNER_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
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
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    parser.add_argument("--long-stream", action="store_true")
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if OWNER_PATTERN.fullmatch(args.owner) is None:
        parser.error("--owner must match [A-Za-z0-9._-]{1,64}")
    if BASELINE_PATTERN.fullmatch(args.baseline) is None:
        parser.error("--baseline must be a lowercase 40-digit commit")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")

    root = Path(__file__).resolve().parent
    cells = LONG_STREAM_CELLS if args.long_stream else CELLS
    short_invocations = (
        LONG_STREAM_SHORT_INVOCATIONS
        if args.long_stream else FINITE_SHORT_INVOCATIONS)
    long_invocations = (
        LONG_STREAM_LONG_INVOCATIONS
        if args.long_stream else FINITE_LONG_INVOCATIONS)
    binary_mode = "--long-stream-batch" if args.long_stream else "--finite-batch"
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} runs={RUN_COUNT} cells={cells} "
        f"short_invocations={short_invocations} "
        f"long_invocations={long_invocations} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(root)}",
        file=sys.stderr,
        flush=True,
    )
    output_fields = (
        "dtype", "m", "n", "block_dim", "output_bytes", "l0c_bytes",
        "short_invocations", "long_invocations", "short_cycles",
        "long_cycles", "validation",
    )
    print("run_id,run_index,process_pid," + ",".join(output_fields), flush=True)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    for run_index in range(RUN_COUNT):
        ordered_cells = (
            cells if run_index % 2 == 0 else tuple(reversed(cells)))
        command = [str(args.binary), binary_mode]
        for cell in ordered_cells:
            command.extend(str(value) for value in cell)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment, start_new_session=True)
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
        rows = list(csv.DictReader(stdout.splitlines()))
        if len(rows) != len(ordered_cells):
            raise RuntimeError("finite FixPipe child emitted wrong row count")
        for row, (_, m, n, block_dim) in zip(rows, ordered_cells):
            if (tuple(row) != output_fields
                    or row["dtype"] != "bf16"
                    or int(row["m"]) != m
                    or int(row["n"]) != n
                    or int(row["block_dim"]) != block_dim
                    or int(row["short_invocations"])
                        != short_invocations
                    or int(row["long_invocations"])
                        != long_invocations
                    or row["validation"] != "exact"):
                raise RuntimeError("finite FixPipe row violated its contract")
            for name in ("short_cycles", "long_cycles"):
                values = tuple(int(value) for value in row[name].split(";"))
                if len(values) != block_dim or any(value <= 0 for value in values):
                    raise RuntimeError("finite FixPipe cycle vector is invalid")
            print(
                f"{args.run_id},{run_index},{process.pid},"
                + ",".join(row[name] for name in output_fields),
                flush=True,
            )


if __name__ == "__main__":
    main()
