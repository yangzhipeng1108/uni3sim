#!/usr/bin/env python3
"""Capture finite GroupedMatmul GM-to-L1 completion-span cells."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


CAMPAIGNS = {
    "long": ((256, 1024, 4096, 8192), 16),
    "m128_extension": ((8192, 12288, 16384, 24576), 128),
    "m64_tail16": ((512, 1024, 3072, 6144), 16),
    "m64_tail32": ((512, 1024, 3072, 6144), 32),
    "m128_tail16": ((512, 1024, 3072, 6144), 16),
    "m128_tail32": ((512, 1024, 3072, 6144), 32),
    "m64_tail16_k2880_stride": ((512, 1024, 3072, 6144), 16),
    "m64_tail32_k2880_stride": ((512, 1024, 3072, 6144), 32),
    "m128_tail16_k2880_stride": ((512, 1024, 3072, 6144), 16),
    "m128_tail32_k2880_stride": ((512, 1024, 3072, 6144), 32),
    "m128_tail16_k2880_tile": ((32, 44, 48, 64), 16),
    "m128_tail32_k2880_tile": ((32, 44, 48, 64), 32),
    "m128_tail48_k2880_tile": ((32, 44, 48, 64), 48),
}
RUN_INDEX_COUNT = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OWNER_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "README.md",
    "grouped_pair_completion_kernel.cpp",
    "grouped_pair_completion_main.cpp",
    "grouped_pair_completion_runner.py",
)
CHILD_FIELDS = (
    "pairs", "active_aics", "a_rows", "b_columns", "base_k",
    "b_source_row_stride", "b_source_tiles_per_worker",
    "a_residency", "b_residency", "drain_interval", "eviction_bytes",
    "min_cycles", "max_cycles", "tick_ns", "completion_span_ns",
    "wall_ns", "validation",
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


def _ordered_pairs(run_index, pair_counts):
    rotation = run_index % len(pair_counts)
    cells = pair_counts[rotation:] + pair_counts[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


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
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument(
        "--campaign", choices=tuple(CAMPAIGNS), default="long")
    parser.add_argument("--child-timeout-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if OWNER_PATTERN.fullmatch(args.owner) is None:
        parser.error("--owner must match [A-Za-z0-9._-]{1,64}")
    if BASELINE_PATTERN.fullmatch(args.baseline) is None:
        parser.error("--baseline must be a lowercase 40-digit commit")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.run_index < 0 or args.run_index >= RUN_INDEX_COUNT:
        parser.error(f"--run-index must be in [0,{RUN_INDEX_COUNT - 1}]")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    pair_counts, expected_a_rows = CAMPAIGNS[args.campaign]
    expected_b_source_row_stride = (
        2880 if "_k2880_" in args.campaign else 256)

    benchmark_root = Path(__file__).resolve().parent
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} run_index={args.run_index} "
        f"campaign={args.campaign} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(benchmark_root)}",
        file=sys.stderr,
        flush=True,
    )
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow((
        "run_id", "run_index", "order_position", "process_pid",
        *CHILD_FIELDS,
    ))
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    for position, pairs in enumerate(
            _ordered_pairs(args.run_index, pair_counts)):
        process = subprocess.Popen(
            [str(args.binary), "--pairs", str(pairs)],
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
                f"child_pid={process.pid} stderr={stderr.rstrip()}",
                file=sys.stderr,
                flush=True,
            )
        if process.returncode != 0:
            raise RuntimeError(
                f"child PID {process.pid} failed with exit "
                f"{process.returncode}")
        rows = list(csv.reader(stdout.splitlines()))
        if len(rows) != 2 or tuple(rows[0]) != CHILD_FIELDS:
            raise RuntimeError(
                f"child PID {process.pid} emitted an invalid schema")
        if len(rows[1]) != len(CHILD_FIELDS):
            raise RuntimeError(
                f"child PID {process.pid} emitted an invalid row")
        if (int(rows[1][0]) != pairs
                or int(rows[1][2]) != expected_a_rows
                or int(rows[1][5]) != expected_b_source_row_stride
                or rows[1][-1] != "pass"):
            raise RuntimeError(
                f"child PID {process.pid} changed the requested cell")
        writer.writerow((
            args.run_id, args.run_index, position, process.pid, *rows[1],
        ))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
