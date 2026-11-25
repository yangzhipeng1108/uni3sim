#!/usr/bin/env python3
"""Capture independent mixed-residency grouped-MatMul MTE2 pair cells."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


ACTIVE_AICS = 24
GROUP_COUNT_EXPANSION_AICS = (22, 24)
GROUP_COUNT_EXPANSION_ITERATIONS = 512
RESIDENT_FINITE_PAIR_COUNTS = (
    16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
MIXED_RING_LONG_PAIR_COUNTS = (
    512, 768, 1024, 1152, 1280, 1536,
    2048, 3072, 4096, 6144, 8192, 12288,
)
COLD_PAIR_COUNT_ITERATIONS = (512, 1024, 2048)
COLD_M16_ACTIVE_AICS = 23
COLD_FOOTPRINT_BYTES = 512 * 1024 * 1024
HOT_SOURCE_TILES = ACTIVE_AICS
MINIMUM_ITERATIONS = 1024
RUN_INDEX_COUNT = 3
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_grouped_pair_kernel.cpp",
    "aic_mte2_grouped_pair_main.cpp",
    "aic_mte2_grouped_pair_runner.py",
)
CHILD_FIELDS = 18
_ACTIVE_PROCESS = None


def _source_tiles(rows, residency):
    tile_bytes = rows * 64 * 2
    if residency == "resident":
        return HOT_SOURCE_TILES
    if residency == "cold":
        return COLD_FOOTPRINT_BYTES // tile_bytes
    raise ValueError(f"unsupported residency: {residency}")


def _b_source_tiles(columns, residency):
    if residency == "resident":
        return HOT_SOURCE_TILES
    if residency == "cold":
        return COLD_FOOTPRINT_BYTES // (64 * columns * 2)
    raise ValueError(f"unsupported residency: {residency}")


def _iterations(a_source_tiles, b_source_tiles):
    minimum = max(
        MINIMUM_ITERATIONS,
        math.ceil(a_source_tiles / ACTIVE_AICS),
        math.ceil(b_source_tiles / ACTIVE_AICS),
    )
    return math.ceil(minimum / 4) * 4


def _cold_source_tiles(active_aics, iterations):
    return 4 * active_aics * iterations


CELLS = tuple(
    (a_residency, b_residency, rows, drain_interval)
    for a_residency in ("resident", "cold")
    for b_residency in ("resident", "cold")
    for rows in (32, 128)
    for drain_interval in (0, 4)
)

CAMPAIGNS = (
    "mixed-residency", "group-count-22x512", "cold-pair-count",
    "cold-m16-pair-count", "resident-finite-pair-count",
    "mixed-ring-long-pair-count")


def _ordered_cells(run_index):
    rotation = run_index * 5 % len(CELLS)
    cells = CELLS[rotation:] + CELLS[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _ordered_group_count_expansion_cells(run_index):
    cells = tuple(
        ("resident", "resident", 128, 4, active_aics,
         GROUP_COUNT_EXPANSION_ITERATIONS)
        for active_aics in GROUP_COUNT_EXPANSION_AICS
    )
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _ordered_cold_pair_count_cells(run_index):
    cells = tuple(
        ("cold", "cold", 128, 4, ACTIVE_AICS, iterations)
        for iterations in COLD_PAIR_COUNT_ITERATIONS
    )
    rotation = run_index % len(cells)
    cells = cells[rotation:] + cells[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _ordered_cold_m16_pair_count_cells(run_index):
    cells = tuple(
        ("cold", "cold", 16, 4, COLD_M16_ACTIVE_AICS, iterations)
        for iterations in COLD_PAIR_COUNT_ITERATIONS
    )
    rotation = run_index % len(cells)
    cells = cells[rotation:] + cells[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _ordered_resident_finite_pair_count_cells(run_index):
    cells = tuple(
        ("resident", "resident", 128, 4, ACTIVE_AICS, iterations)
        for iterations in RESIDENT_FINITE_PAIR_COUNTS
    )
    rotation = run_index * 3 % len(cells)
    cells = cells[rotation:] + cells[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _ordered_mixed_ring_long_pair_count_cells(run_index):
    cells = tuple(
        (a_residency, b_residency, 128, 4, ACTIVE_AICS, iterations)
        for a_residency in ("resident", "cold")
        for b_residency in ("resident", "cold")
        for iterations in MIXED_RING_LONG_PAIR_COUNTS
    )
    rotation = run_index % len(cells)
    cells = cells[rotation:] + cells[:rotation]
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
    parser.add_argument("--b-columns", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument(
        "--campaign", choices=CAMPAIGNS, default="mixed-residency")
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if args.b_columns not in (240, 256):
        parser.error("--b-columns must be one of 240, 256")
    if args.campaign == "group-count-22x512" and args.b_columns != 256:
        parser.error("group-count-22x512 requires --b-columns 256")
    if args.campaign == "cold-pair-count" and args.b_columns != 256:
        parser.error("cold-pair-count requires --b-columns 256")
    if args.campaign == "cold-m16-pair-count" and args.b_columns != 256:
        parser.error("cold-m16-pair-count requires --b-columns 256")
    if (args.campaign == "resident-finite-pair-count"
            and args.b_columns != 256):
        parser.error("resident-finite-pair-count requires --b-columns 256")
    if (args.campaign == "mixed-ring-long-pair-count"
            and args.b_columns != 256):
        parser.error("mixed-ring-long-pair-count requires --b-columns 256")
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
        f"campaign={args.campaign} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(benchmark_root)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,run_index,order_position,process_pid,a_residency,b_residency,"
        "a_rows,b_columns,a_tile_bytes,b_tile_bytes,a_source_tiles,b_source_tiles,"
        "drain_interval,block_dim,iterations,short_cycles,short_wall_ns,"
        "long_cycles,long_wall_ns,tick_ns,cycles_per_pair,ns_per_pair,"
        "bandwidth_gbs,validation",
        flush=True,
    )
    if args.campaign == "mixed-residency":
        cells = _ordered_cells(args.run_index)
    elif args.campaign == "group-count-22x512":
        cells = _ordered_group_count_expansion_cells(args.run_index)
    elif args.campaign == "cold-pair-count":
        cells = _ordered_cold_pair_count_cells(args.run_index)
    elif args.campaign == "cold-m16-pair-count":
        cells = _ordered_cold_m16_pair_count_cells(args.run_index)
    elif args.campaign == "resident-finite-pair-count":
        cells = _ordered_resident_finite_pair_count_cells(args.run_index)
    elif args.campaign == "mixed-ring-long-pair-count":
        cells = _ordered_mixed_ring_long_pair_count_cells(args.run_index)
    else:
        raise ValueError(f"unsupported campaign: {args.campaign}")
    command = [str(args.binary), "--batch"]
    expected = []
    for cell in cells:
        a_residency, b_residency, rows, drain_interval = cell[:4]
        if args.campaign == "mixed-residency":
            a_tiles = _source_tiles(rows, a_residency)
            b_tiles = _b_source_tiles(args.b_columns, b_residency)
            active_aics = ACTIVE_AICS
            iterations = _iterations(a_tiles, b_tiles)
        elif args.campaign == "group-count-22x512":
            active_aics, iterations = cell[4:]
            a_tiles = active_aics
            b_tiles = active_aics
        elif args.campaign in ("cold-pair-count", "cold-m16-pair-count"):
            active_aics, iterations = cell[4:]
            source_tiles = _cold_source_tiles(active_aics, iterations)
            a_tiles = source_tiles
            b_tiles = source_tiles
        elif args.campaign == "resident-finite-pair-count":
            active_aics, iterations = cell[4:]
            a_tiles = active_aics
            b_tiles = active_aics
        elif args.campaign == "mixed-ring-long-pair-count":
            active_aics, iterations = cell[4:]
            a_tiles = _source_tiles(rows, a_residency)
            b_tiles = _b_source_tiles(args.b_columns, b_residency)
        else:
            raise ValueError(f"unsupported campaign: {args.campaign}")
        child_cell = (
            rows, args.b_columns, a_tiles, b_tiles, drain_interval,
            active_aics, iterations)
        expected.append(child_cell)
        command.extend(str(value) for value in child_cell)

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
        print(f"child_pid={process.pid} stderr={stderr.rstrip()}",
              file=sys.stderr, flush=True)
    if process.returncode != 0:
        raise RuntimeError(
            f"child PID {process.pid} failed with exit {process.returncode}")
    output_lines = stdout.strip().splitlines()
    if len(output_lines) != len(cells):
        raise RuntimeError(
            f"child PID {process.pid} emitted {len(output_lines)} data lines")
    for order_position, output_line in enumerate(output_lines):
        fields = output_line.split(",")
        if len(fields) != CHILD_FIELDS or fields[-1] != "pass":
            raise RuntimeError(
                f"child PID {process.pid} emitted an invalid raw schema")
        observed = tuple(
            int(fields[index]) for index in (0, 1, 4, 5, 6, 7, 8))
        if observed != expected[order_position]:
            raise RuntimeError(
                f"child PID {process.pid} changed the requested order")
        a_residency, b_residency = cells[order_position][:2]
        print(
            f"{args.run_id},{args.run_index},{order_position},{process.pid},"
            f"{a_residency},{b_residency},{output_line}",
            flush=True,
        )


if __name__ == "__main__":
    main()
