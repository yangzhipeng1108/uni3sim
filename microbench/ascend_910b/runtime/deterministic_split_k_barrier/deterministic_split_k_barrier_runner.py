#!/usr/bin/env python3
"""Run the source-matched deterministic Split-K control-lifecycle probe."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys


CELLS = ((1, 0), (2, 0), (2, 1))
RUNS = 3
TRIALS = 8
SOURCE_FILES = (
    "CMakeLists.txt",
    "deterministic_split_k_barrier_kernel.cpp",
    "deterministic_split_k_barrier_main.cpp",
    "deterministic_split_k_barrier_runner.py",
)
RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expected-snapshot", required=True)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID.fullmatch(args.run_id) is None:
        parser.error("--run-id is invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", args.baseline):
        parser.error("--baseline must be a lowercase commit")
    if SHA256.fullmatch(args.expected_snapshot) is None:
        parser.error("--expected-snapshot must be a SHA256")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must be executable")
    root = Path(__file__).resolve().parent
    snapshot = _snapshot(root)
    if snapshot != args.expected_snapshot:
        raise RuntimeError("deployed source differs from its audited snapshot")
    print(
        f"hardware_id=ascend_910b owner=capture-owner-3 "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} run_id={args.run_id} runs={RUNS} trials={TRIALS} "
        "aic_groups=24 aiv_groups=48 source_n_slabs=2 "
        "source_sequence=AIC_to_AIV_wait_syncall_pipe_to_AIV_to_AIC_wait_syncall_pipe "
        f"binary_sha256={_sha256(args.binary)} source_snapshot_sha256={snapshot}",
        file=sys.stderr, flush=True)
    print("run_id,run_index,trial,slab_count,mode,max_actor_cycles,"
          "max_aic_reverse_wait_cycles,max_aic_sync_cycles,"
          "max_aiv_first_sync_cycles,max_aiv_second_sync_cycles,"
          "host_wall_ns,validation")
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    environment["ASCEND_DEVICE_ID"] = "0"
    command = [str(args.binary), "--batch", *(str(field) for cell in CELLS for field in cell)]
    expected_header = ("slab_count,mode,max_actor_cycles,max_aic_reverse_wait_cycles,"
                       "max_aic_sync_cycles,max_aiv_first_sync_cycles,"
                       "max_aiv_second_sync_cycles,host_wall_ns,validation")
    for run_index in range(RUNS):
        for trial in range(TRIALS):
            completed = subprocess.run(command, capture_output=True, text=True,
                env=environment, timeout=120, check=False)
            if completed.stderr:
                print(f"run_index={run_index} trial={trial} stderr="
                      f"{completed.stderr.rstrip()}", file=sys.stderr, flush=True)
            if completed.returncode != 0:
                raise RuntimeError(f"probe run={run_index} trial={trial} failed")
            lines = completed.stdout.strip().splitlines()
            if not lines or lines[0] != expected_header or len(lines) != len(CELLS) + 1:
                raise RuntimeError("probe emitted an invalid CSV")
            for expected, line in zip(CELLS, lines[1:], strict=True):
                fields = line.split(",")
                if len(fields) != 9 or fields[-1] != "pass":
                    raise RuntimeError("probe emitted a failed row")
                if (int(fields[0]), int(fields[1])) != expected:
                    raise RuntimeError("probe changed the requested source cell")
                if any(int(value) < 0 for value in fields[2:-1]):
                    raise RuntimeError("probe emitted a negative control span")
                print(f"{args.run_id},{run_index},{trial},{line}")


if __name__ == "__main__":
    main()
