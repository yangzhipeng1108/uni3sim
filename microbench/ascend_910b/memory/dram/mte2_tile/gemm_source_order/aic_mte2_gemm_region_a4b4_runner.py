#!/usr/bin/env python3
"""Run the independent 24-AIC A4/B4 MatMulV3 region ingress probe."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess


RUNS = 3
TRIALS = 8
_RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")
_OWNER = re.compile(r"[A-Za-z0-9._-]{1,64}")
_BASELINE = re.compile(r"[0-9a-f]{40}")
_SOURCES = (
    "CMakeLists.txt",
    "aic_mte2_gemm_region_schedule.h",
    "aic_mte2_gemm_region_a4b4_source_order_kernel.cpp",
    "aic_mte2_gemm_region_a4b4_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_region_a4b4_main.cpp",
    "aic_mte2_gemm_region_a4b4_runner.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for name in _SOURCES:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if _OWNER.fullmatch(args.owner) is None:
        parser.error("--owner is invalid")
    if _BASELINE.fullmatch(args.baseline) is None:
        parser.error("--baseline must be a lowercase 40-digit commit")
    if _RUN_ID.fullmatch(args.run_id) is None:
        parser.error("--run-id is invalid")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must be an executable file")
    root = Path(__file__).resolve().parent
    cell = (64, 20, 54, 8, 2, 8, 10)
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} run_id={args.run_id} runs={RUNS} "
        f"trials={TRIALS} source_program=V3_A4_B4_depth2 "
        f"cell=(m_tiles={cell[0]},n_tiles={cell[1]},k_groups={cell[2]},"
        f"m_regions={cell[3]},n_regions={cell[4]},m_block={cell[5]},"
        f"n_block={cell[6]}) active_aics=24 "
        f"binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_snapshot_sha256(root)}",
        flush=True)
    print("run_id,run_index,trial,mode,max_cycles,duration_ns,m_tiles,n_tiles,k_groups,aggregate_tiles,aggregate_checksum,validation", flush=True)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for run_index in range(RUNS):
        result = subprocess.run(
            [str(args.binary), "--trials", str(TRIALS), "--k-groups", "54",
             "--m-regions", "8", "--n-regions", "2", "--m-block", "8",
             "--n-block", "10", "--m-tiles", "64", "--n-tiles", "20"],
            check=False, capture_output=True, text=True, env=environment,
            timeout=900)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.rstrip())
        lines = result.stdout.strip().splitlines()
        if len(lines) != 2 * TRIALS + 1 or lines[0] != (
                "trial,mode,max_cycles,duration_ns,m_tiles,n_tiles,k_groups,"
                "aggregate_tiles,aggregate_checksum,validation"):
            raise RuntimeError("A4/B4 probe emitted invalid CSV")
        for line in lines[1:]:
            fields = line.split(",")
            if len(fields) != 10 or fields[-1] != "pass":
                raise RuntimeError("A4/B4 probe emitted malformed row")
            print(f"{args.run_id},{run_index},{line}", flush=True)


if __name__ == "__main__":
    main()
