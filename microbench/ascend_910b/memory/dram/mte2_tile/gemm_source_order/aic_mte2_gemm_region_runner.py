#!/usr/bin/env python3
"""Measure pre-registered MatMulV3 ingress-region programs."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


RUN_COUNT = 3
TRIALS = 8
REFERENCE_REGION_CELLS = (
    (32, 32, 4, 2, 8, 16, 16),
    (32, 54, 4, 2, 8, 27, 10),
)
TAIL_REGION_CELLS = (
    (16, 20, 1, 1, 16, 20, 10),
    (16, 54, 2, 2, 8, 27, 10),
    (32, 16, 4, 2, 8, 8, 28),
    (32, 20, 4, 2, 8, 10, 27),
    (256, 16, 36, 2, 8, 8, 28),
)
TAIL_256X16_REGION_CELLS = (
    (256, 16, 36, 2, 8, 8, 28),
)
LONG_REGION_CELLS = (
    (256, 112, 32, 8, 8, 14, 16),
)
K5120_LONG_REGION_CELLS = (
    (256, 54, 32, 2, 8, 27, 10),
)
K13824_LONG_REGION_CELLS = (
    (256, 20, 32, 2, 8, 10, 27),
)
SINGLE_REGION_CELLS = (
    (16, 16, 1, 1, 16, 16, 8),
)
TWO_BY_TWO_REGION_CELLS = (
    (16, 32, 2, 2, 8, 16, 16),
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OWNER_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_gemm_region_schedule.h",
    "aic_mte2_gemm_region_source_order_kernel.cpp",
    "aic_mte2_gemm_region_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_region_pipeline_completion_kernel.cpp",
    "aic_mte2_gemm_region_main.cpp",
    "aic_mte2_gemm_region_runner.py",
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


def _worker_tiles(
        m_tiles, n_tiles, m_regions, n_regions, m_block, n_block):
    if m_tiles > m_regions * m_block or n_tiles > n_regions * n_block:
        raise ValueError("logical tile extent exceeds its region schedule")
    workers = []
    skew_period = m_block // math.gcd(m_block, n_block) * n_block
    tail_aics_per_region = m_block * n_block % 24
    rounds = (m_block * n_block + 23) // 24
    for block in range(24):
        tiles = []
        for region in range(m_regions * n_regions):
            block_start = region * tail_aics_per_region % 24
            relative_block = (block + 24 - block_start) % 24
            worker_rounds = rounds if (
                tail_aics_per_region == 0
                or relative_block < tail_aics_per_region
            ) else rounds - 1
            for round_index in range(worker_rounds):
                local_block = relative_block + round_index * 24
                local_m = local_block % m_block
                local_n = (
                    local_block + local_block // skew_period) % n_block
                m_region = region // n_regions
                n_position = region % n_regions
                n_region = (
                    n_regions - n_position - 1
                    if m_region & 1 else n_position)
                tile = (
                    m_region * m_block + local_m,
                    n_region * n_block + local_n,
                )
                if tile[0] < m_tiles and tile[1] < n_tiles:
                    tiles.append(tile)
        workers.append(tuple(tiles))
    return tuple(workers)


def _worker_tile_counts(
        m_tiles, n_tiles, m_regions, n_regions, m_block, n_block):
    return tuple(len(tiles) for tiles in _worker_tiles(
        m_tiles, n_tiles, m_regions, n_regions, m_block, n_block))


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile", required=True,
        choices=(
            "reference", "tail", "long", "single-region",
            "two-by-two-region", "tail-256x16", "k5120-long",
            "k13824-long"))
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
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
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")

    cells_by_profile = {
        "reference": REFERENCE_REGION_CELLS,
        "tail": TAIL_REGION_CELLS,
        "long": LONG_REGION_CELLS,
        "single-region": SINGLE_REGION_CELLS,
        "two-by-two-region": TWO_BY_TWO_REGION_CELLS,
        "tail-256x16": TAIL_256X16_REGION_CELLS,
        "k5120-long": K5120_LONG_REGION_CELLS,
        "k13824-long": K13824_LONG_REGION_CELLS,
    }
    cells = cells_by_profile[args.profile]
    cell_summaries = []
    for (m_tiles, n_tiles, m_regions, n_regions,
            m_block, n_block, k_slabs) in cells:
        counts = _worker_tile_counts(
            m_tiles, n_tiles, m_regions, n_regions, m_block, n_block)
        aggregate_tiles = m_tiles * n_tiles
        if sum(counts) != aggregate_tiles:
            raise RuntimeError("invalid compact L2-region worker schedule")
        cell_summaries.append((
            m_tiles, n_tiles, m_regions, n_regions,
            m_block, n_block, k_slabs, counts))
    root = Path(__file__).resolve().parent
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} runs={RUN_COUNT} trials={TRIALS} "
        f"region_cells={cell_summaries} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(root)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,run_index,process_pid,trial,mode,max_cycles,duration_ns,"
        "m_tiles,n_tiles,k_slabs,aggregate_tiles,aggregate_checksum,"
        "validation",
        flush=True,
    )
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for (m_tiles, n_tiles, m_regions, n_regions,
            m_block, n_block, k_slabs) in cells:
        aggregate_tiles = m_tiles * n_tiles
        aggregate_checksum = aggregate_tiles * (aggregate_tiles + 1) // 2
        for run_index in range(RUN_COUNT):
            process = subprocess.Popen(
                [str(args.binary), "--trials", str(TRIALS),
                 "--k-slabs", str(k_slabs),
                 "--m-regions", str(m_regions),
                 "--n-regions", str(n_regions),
                 "--m-block", str(m_block),
                 "--n-block", str(n_block),
                 "--m-tiles", str(m_tiles),
                 "--n-tiles", str(n_tiles)],
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
                    f"k_slabs={k_slabs} stderr={stderr.rstrip()}",
                    file=sys.stderr, flush=True)
            if process.returncode != 0:
                raise RuntimeError(
                    f"child PID {process.pid} failed with "
                    f"{process.returncode}")
            lines = stdout.strip().splitlines()
            if len(lines) != 3 * TRIALS + 1 or lines[0] != (
                    "trial,mode,max_cycles,duration_ns,m_tiles,n_tiles,"
                    "k_slabs,aggregate_tiles,aggregate_checksum,validation"):
                raise RuntimeError("region child emitted invalid output")
            for line in lines[1:]:
                fields = line.split(",")
                if (len(fields) != 10
                        or fields[1] not in {
                            "mte2_only", "mte2_mte1_completion",
                            "mte2_mte1_cube_completion"}
                        or fields[4:6] != [str(m_tiles), str(n_tiles)]
                        or fields[6:9] != [
                            str(k_slabs), str(aggregate_tiles),
                            str(aggregate_checksum)]
                        or fields[-1] != "pass"):
                    raise RuntimeError("region child emitted invalid row")
                print(
                    f"{args.run_id},{run_index},{process.pid},{line}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
