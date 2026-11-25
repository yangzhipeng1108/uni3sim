#!/usr/bin/env python3
"""Capture five-N full-A narrow MatMulV2 Scalar-control endpoints."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


CELLS = ((16, 2048, 1), (16, 4096, 1), (1, 2048, 0), (1, 4096, 0))
RUNS = 3
TRIALS = 8
SOURCE_FILES = (
    "CMakeLists.txt",
    "matmul_v2_full_a_five_n_scalar_control_kernel.cpp",
    "matmul_v2_full_a_five_n_scalar_control_main.cpp",
    "matmul_v2_full_a_five_n_scalar_control_runner.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        payload = (root / name).read_bytes()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(len(payload)).encode())
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-snapshot", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if re.fullmatch(r"[0-9a-f]{40}", args.baseline) is None:
        parser.error("--baseline must be a lowercase commit")
    if re.fullmatch(r"[A-Za-z0-9._-]{1,96}", args.run_id) is None:
        parser.error("--run-id is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_snapshot) is None:
        parser.error("--expected-snapshot must be a SHA256")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must be executable")
    root = Path(__file__).resolve().parent
    snapshot = _snapshot(root)
    if snapshot != args.expected_snapshot:
        raise RuntimeError("deployed source differs from the audited snapshot")
    print(
        f"hardware_id=ascend_910b owner=capture-owner-3 "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} run_id={args.run_id} runs={RUNS} "
        f"trials={TRIALS} aic_workers=23 base=16x256x64 "
        "worker_n_tiles=5 a1_slots=1 b1_slots=2 static_m=16 dynamic_m=1 "
        f"binary_sha256={_sha256(args.binary)} source_snapshot_sha256={snapshot}",
        file=sys.stderr, flush=True)
    print("run_id,run_index,trial,worker,m,k,static_no_tail,depth_a1,"
          "depth_b1,step_ka,step_kb,base_n,worker_n_tiles,program_cycles,"
          "program_ns,validation")
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    header = ("trial,worker,m,k,static_no_tail,depth_a1,depth_b1,step_ka,"
              "step_kb,base_n,worker_n_tiles,program_cycles,program_ns,validation")
    for run_index in range(RUNS):
        cells = CELLS if run_index % 2 == 0 else tuple(reversed(CELLS))
        for logical_m, projection_k, static_no_tail in cells:
            command = [
                str(args.binary), "--m", str(logical_m), "--k",
                str(projection_k), "--static-no-tail", str(static_no_tail),
                "--trials", str(TRIALS)]
            completed = subprocess.run(command, env=environment,
                capture_output=True, text=True, timeout=180, check=False)
            if completed.stderr:
                print(completed.stderr.rstrip(), file=sys.stderr, flush=True)
            if completed.returncode != 0:
                raise RuntimeError("five-N full-A Scalar endpoint failed")
            lines = completed.stdout.strip().splitlines()
            if not lines or lines[0] != header or len(lines) != 1 + 23 * TRIALS:
                raise RuntimeError("five-N full-A Scalar endpoint emitted an invalid CSV")
            for line in lines[1:]:
                fields = line.split(",")
                if len(fields) != 14 or fields[-1] != "pass":
                    raise RuntimeError("five-N full-A Scalar endpoint emitted a failed row")
                if (int(fields[2]), int(fields[3]), int(fields[4])) != (
                        logical_m, projection_k, static_no_tail):
                    raise RuntimeError("five-N full-A Scalar endpoint changed its cell")
                if (int(fields[9]), int(fields[10])) != (256, 5):
                    raise RuntimeError("five-N full-A Scalar endpoint changed its topology")
                print(f"{args.run_id},{run_index},{line}")


if __name__ == "__main__":
    main()
