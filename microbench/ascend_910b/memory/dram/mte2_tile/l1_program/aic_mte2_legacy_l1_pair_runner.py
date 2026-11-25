#!/usr/bin/env python3
"""Capture exact legacy MatMulV2 cold A1/B1 pair-program cells."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


FOOTPRINT_MB_PER_OPERAND = 16384
SHORT_PROGRAMS = 1
LONG_PROGRAMS = 4
RUN_INDEX_COUNT = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_legacy_l1_pair_kernel.cpp",
    "aic_mte2_legacy_l1_pair_main.cpp",
    "aic_mte2_legacy_l1_pair_runner.py",
)
CELLS = (
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (4, 1),
    (5, 0),
    (5, 1),
)
EXPECTED_GEOMETRIES = {
    0: (22, 128, 256, 14336, 96, 256, 14336, 112),
    1: (24, 256, 256, 5120, 112, 256, 5120, 80),
    2: (22, 16, 256, 14336, 96, 256, 14336, 112),
    3: (22, 32, 256, 14336, 96, 256, 14336, 112),
    4: (22, 48, 256, 14336, 96, 256, 14336, 112),
    5: (22, 64, 256, 14336, 96, 256, 14336, 112),
}
RAW_FIELDS = (
    "geometry_id", "drain_after_each_pair", "footprint_mb_per_operand",
    "block_dim", "a_rows", "a_columns", "a_source_row_stride",
    "b_rows", "b_columns", "b_source_row_stride", "pairs_per_program",
    "a_local_bytes", "b_local_bytes", "pair_bytes", "short_programs",
    "long_programs", "short_cycles", "short_wall_ns", "long_cycles",
    "long_wall_ns", "tick_ns", "cycles_per_program", "ns_per_program",
    "ns_per_pair", "bandwidth_gbs", "validation",
)
_ACTIVE_PROCESS = None


def _ordered_cells(run_index):
    rotation = run_index % len(CELLS)
    cells = CELLS[rotation:] + CELLS[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


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
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if args.run_index < 0 or args.run_index >= RUN_INDEX_COUNT:
        parser.error(f"--run-index must be in [0,{RUN_INDEX_COUNT - 1}]")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")

    benchmark_root = Path(__file__).resolve().parent
    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} run_index={args.run_index} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(benchmark_root)}",
        file=sys.stderr,
        flush=True,
    )
    cells = _ordered_cells(args.run_index)
    command = [
        str(args.binary), "--batch", str(FOOTPRINT_MB_PER_OPERAND),
        str(SHORT_PROGRAMS), str(LONG_PROGRAMS),
    ]
    for geometry_id, drain_after_each_pair in cells:
        command.extend((str(geometry_id), str(drain_after_each_pair)))

    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    process = subprocess.Popen(
        command,
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
        print(
            f"child_pid={process.pid} stderr={stderr.rstrip()}",
            file=sys.stderr,
            flush=True,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"child PID {process.pid} failed with exit {process.returncode}")

    output_lines = stdout.strip().splitlines()
    if len(output_lines) != len(cells):
        raise RuntimeError(
            f"child PID {process.pid} emitted {len(output_lines)} data lines")
    print(
        ",".join(("run_id", "run_index", "order_position", "process_pid",
                  *RAW_FIELDS)),
        flush=True,
    )
    for order_position, output_line in enumerate(output_lines):
        fields = output_line.split(",")
        if len(fields) != len(RAW_FIELDS) or fields[-1] != "pass":
            raise RuntimeError(
                f"child PID {process.pid} emitted an invalid raw schema")
        geometry_id, drain_after_each_pair = cells[order_position]
        observed_geometry = tuple(int(fields[index]) for index in (
            3, 4, 5, 6, 7, 8, 9, 10))
        if (int(fields[0]), int(fields[1])) != (
                geometry_id, drain_after_each_pair):
            raise RuntimeError(
                f"child PID {process.pid} changed the requested cell order")
        if observed_geometry != EXPECTED_GEOMETRIES[geometry_id]:
            raise RuntimeError(
                f"child PID {process.pid} changed the static geometry")
        if tuple(int(fields[index]) for index in (2, 14, 15, 20)) != (
                FOOTPRINT_MB_PER_OPERAND, SHORT_PROGRAMS, LONG_PROGRAMS, 20):
            raise RuntimeError(
                f"child PID {process.pid} changed the capture constants")
        print(
            f"{args.run_id},{args.run_index},{order_position},{process.pid},"
            f"{output_line}",
            flush=True,
        )


if __name__ == "__main__":
    main()
