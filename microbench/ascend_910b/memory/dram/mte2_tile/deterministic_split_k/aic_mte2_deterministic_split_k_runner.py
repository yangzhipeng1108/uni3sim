#!/usr/bin/env python3
"""Run the independent CANN deterministic Split-K MTE2 topology probe."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


RUNS = 3
TRIALS = 8
A_ROW_CELLS = 8
_BASELINE = re.compile(r"[0-9a-f]{40}")
_OWNER = re.compile(r"[A-Za-z0-9._-]{1,64}")
_RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")
_SOURCES = (
    "CMakeLists.txt",
    "aic_mte2_deterministic_split_k_kernel.cpp",
    "aic_mte2_deterministic_split_k_main.cpp",
    "aic_mte2_deterministic_split_k_runner.py",
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
        f"trials={TRIALS} active_aics=24 n_slabs=2 "
        "full_workers=3 full_k_chunks=4 regular_workers=21 "
        "regular_k_chunks=3 k_chunk=384 k_tail=256 "
        "a_descriptor_rows=16:16:128 a_descriptor_columns=384 "
        "a1_slots=1 b1_slots=2 l0a_slots=2 l0b_slots=2 "
        "b_descriptor=128x384 "
        "b_source_stride=28672 b_descriptors_per_chunk=32 "
        "source_preload_commands=4 aiv_wait_before_second_slab=false "
        "mechanism=mte2_mte1_source_queue_no_cube_fixpipe_aiv "
        f"binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_snapshot_sha256(root)}",
        file=sys.stderr, flush=True)
    print(
        "run_id,run_index,a_rows,trial,max_cycles,duration_ns,validation")
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for run_index in range(RUNS):
        result = subprocess.run(
            [str(args.binary), "--trials", str(TRIALS), "--device-id",
             str(args.device_id)],
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
        if (not lines
                or lines[0] != (
                    "a_rows,trial,max_cycles,duration_ns,validation")
                or len(lines) != A_ROW_CELLS * TRIALS + 1):
            raise RuntimeError("probe emitted an invalid CSV")
        for line in lines[1:]:
            a_rows, trial, cycles, duration_ns, validation = line.split(",")
            if validation != "pass":
                raise RuntimeError("probe validation failed")
            print(
                f"{args.run_id},{run_index},{a_rows},{trial},{cycles},"
                f"{duration_ns},{validation}", flush=True)


if __name__ == "__main__":
    main()
