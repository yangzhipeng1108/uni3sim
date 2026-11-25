#!/usr/bin/env python3
"""Run the independent full-A MatMulV2 16x96x128 MTE2 probe."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


RUNS = 3
TRIALS = 8
_RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")
_OWNER = re.compile(r"[A-Za-z0-9._-]{1,64}")
_BASELINE = re.compile(r"[0-9a-f]{40}")
_SOURCES = (
    "CMakeLists.txt",
    "aic_mte2_matmul_v2_full_a_k4096_n96_kernel.cpp",
    "aic_mte2_matmul_v2_full_a_k4096_n96_main.cpp",
    "aic_mte2_matmul_v2_full_a_k4096_n96_runner.py",
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
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} run_id={args.run_id} runs={RUNS} "
        f"trials={TRIALS} active_aics=23 a_descriptor=16x4096 "
        f"b_descriptor=96x512 a1_depth=1 b1_depth=2 "
        f"output_tiles_per_worker=5 b_descriptors_per_worker=40 "
        f"logical_n_tiles=115 source_residency=dram_cold_after_clean_276MiB_mte2_eviction "
        f"binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_snapshot_sha256(root)}",
        file=sys.stderr, flush=True)
    print("run_id,run_index,trial,max_cycles,duration_ns,validation")
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    # ACL renumbers the single selected physical device to logical zero.
    environment["ASCEND_DEVICE_ID"] = "0"
    for run_index in range(RUNS):
        result = subprocess.run(
            [str(args.binary), "--trials", str(TRIALS), "--device-id", "0"],
            check=False, capture_output=True, text=True, env=environment,
            timeout=900)
        if result.stderr:
            print(
                f"run_index={run_index} stderr={result.stderr.rstrip()}",
                file=sys.stderr, flush=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"probe run {run_index} failed with {result.returncode}")
        lines = result.stdout.strip().splitlines()
        if (not lines or lines[0] != "trial,max_cycles,duration_ns,validation"
                or len(lines) != TRIALS + 1):
            raise RuntimeError(
                "probe emitted an invalid CSV: "
                f"{result.stdout!r}")
        for line in lines[1:]:
            trial, cycles, duration_ns, validation = line.split(",")
            if validation != "pass":
                raise RuntimeError("probe validation failed")
            print(
                f"{args.run_id},{run_index},{trial},{cycles},"
                f"{duration_ns},{validation}", flush=True)


if __name__ == "__main__":
    main()
