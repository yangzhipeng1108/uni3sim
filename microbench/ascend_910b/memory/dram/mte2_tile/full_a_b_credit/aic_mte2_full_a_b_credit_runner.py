#!/usr/bin/env python3
"""Run the full-A/B-credit MTE2 probe for CANN 8.5.1 MatMulV2 decode launches."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


RUNS = 3
TRIALS = 8
FOOTPRINT_MB = 4096
CANN_SOURCE_SHA256 = "7007bca9cc8d2e34ce3f0ba641382f52821be688c4911ad7c7ba3c4389230ec5"
# (a_columns, b_rows, b_columns, b_source_stride, n_slabs, block_dim).
# Geometries are the qwen3-4b decode MatMulV2 full-A launches from the
# canonical launch_data payload decode: A slab is 16xK bf16 (a_columns=K,
# contiguous source), each B1 slab is (b_l1_k*64) x tile_N with the physical
# K source stride, and one output tile (wave) issues K/(b_l1_k*64) slabs.
# The down K9728 cell uses b_l1_k=4 (slab 256x112) instead of the production
# b_l1_k=8: the 304-KiB A slab plus two 112-KiB b_l1_k=8 slots exceeds the
# 512-KiB L1 budget, so the production slot depth cannot be replicated; this
# is a deliberate, recorded deviation, not a silent fallback.
GEOMETRIES = (
    (2560, 256, 256, 2560, 10, 24),   # qkv M1-48 tile 16x256x64 b_l1_k=4
    (2560, 256, 256, 2560, 10, 19),   # gate_up M1-48, 19 AICs
    (2560, 256, 256, 2560, 10, 24),   # lm_head M1-48, same tile as qkv
    (4096, 512, 112, 4096, 8, 23),    # o M1-15 tile 16x112x64 b_l1_k=8
    (9728, 256, 112, 9728, 38, 23),   # down K9728, b_l1_k=4 capacity variant
)
WAVES = (1, 2, 4, 8, 16, 24)
CELLS = tuple(
    (16, a_columns, b_rows, b_columns, b_source_stride, n_slabs, waves,
     block_dim)
    for a_columns, b_rows, b_columns, b_source_stride, n_slabs, block_dim
    in GEOMETRIES
    for waves in WAVES
)
CHILD_FIELDS = 19
RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,96}")
OWNER = re.compile(r"[A-Za-z0-9._-]{1,64}")
BASELINE = re.compile(r"[0-9a-f]{40}")
SOURCES = (
    "CMakeLists.txt",
    "aic_mte2_full_a_b_credit_kernel.cpp",
    "aic_mte2_full_a_b_credit_main.cpp",
    "aic_mte2_full_a_b_credit_runner.py",
)
ACTIVE_PROCESS = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for filename in SOURCES:
        payload = (root / filename).read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _stop_child(signum, _frame) -> None:
    if ACTIVE_PROCESS is not None:
        try:
            os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


def main() -> None:
    global ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cann-source", required=True, type=Path)
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if OWNER.fullmatch(args.owner) is None:
        parser.error("--owner must match [A-Za-z0-9._-]{1,64}")
    if BASELINE.fullmatch(args.baseline) is None:
        parser.error("--baseline must be a lowercase 40-digit commit")
    if RUN_ID.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    if not args.cann_source.is_file():
        parser.error("--cann-source must name the CANN AL1-full-load source")
    if _sha256(args.cann_source) != CANN_SOURCE_SHA256:
        parser.error("--cann-source SHA256 does not match CANN 8.5.1 "
                     "mat_mul_l1_full_load.h")
    if args.child_timeout_seconds <= 0:
        parser.error("--child-timeout-seconds must be positive")

    root = Path(__file__).resolve().parent
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} run_id={args.run_id} runs={RUNS} "
        f"trials={TRIALS} footprint_mb={FOOTPRINT_MB} "
        "source_program=mat_mul_l1_full_load(AL1FullLoad) "
        "a_slab=16xK_full_load_resident b1_slots=2 "
        f"cann_source={args.cann_source} "
        f"cann_source_sha256={CANN_SOURCE_SHA256} cells={CELLS} "
        f"binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(root)}",
        file=sys.stderr,
        flush=True,
    )
    child_header = (
        "trial,cell_index,mode,wave_index,waves,n_slabs,"
        "wave_max_cycles,wave_ns,wave_bytes_per_aic,gbps_per_aic,"
        "a_rows,a_columns,b_rows,b_columns,b_source_stride,block_dim,"
        "total_max_cycles,a_load_max_cycles,validation")
    print(f"run_id,run_index,{child_header}", flush=True)
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    expected_rows = sum(2 * TRIALS * cell[6] for cell in CELLS)
    for run_index in range(RUNS):
        command = [
            str(args.binary), "--trials", str(TRIALS),
            "--footprint-mb", str(FOOTPRINT_MB),
        ]
        for cell in CELLS:
            command.extend(str(value) for value in cell)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        ACTIVE_PROCESS = process
        try:
            stdout, stderr = process.communicate(
                timeout=args.child_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            stdout, stderr = process.communicate()
            raise RuntimeError(
                f"probe timed out in run {run_index}: {stderr.rstrip()}"
            ) from error
        finally:
            ACTIVE_PROCESS = None
        if process.returncode != 0:
            raise RuntimeError(
                f"probe failed in run {run_index}: {stderr.rstrip()}"
            )
        lines = stdout.strip().splitlines()
        if len(lines) != expected_rows + 1 or lines[0] != child_header:
            raise RuntimeError(
                f"probe emitted an invalid CSV in run {run_index}"
            )
        for line in lines[1:]:
            fields = line.split(",")
            if len(fields) != CHILD_FIELDS or fields[-1] != "pass":
                raise RuntimeError(
                    f"probe emitted a malformed row in run {run_index}"
                )
            print(f"{args.run_id},{run_index},{line}", flush=True)


if __name__ == "__main__":
    main()
