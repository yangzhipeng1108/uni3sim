#!/usr/bin/env python3
"""Compare exact MatMulV2 MTE2, MTE1 and dependent-Cube spans."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


RUN_COUNT = 3
TRIALS = 8
# (active AICs, M groups, output tiles per worker, 512-column K slabs).  The
# cells vary source reuse topology and finite program length independently
# while retaining
# the exact 128x256x64 tile, A1 K=512/B1 K=256 grouping, two L1 slots, and two
# L0A/L0B slots.  They are mechanism cells rather than production cases.
CELLS = (
    (22, 1, 2, 8),
    (22, 1, 2, 16),
    (23, 1, 3, 8),
    (23, 1, 3, 16),
    (23, 1, 5, 8),
    (23, 1, 5, 16),
    (24, 1, 3, 10),
    (24, 2, 3, 10),
    (24, 4, 1, 10),
    (24, 4, 3, 10),
    (24, 4, 5, 10),
    (24, 4, 9, 6),
    (24, 4, 9, 10),
    (24, 4, 9, 16),
)
K13824_CELLS = ((24, 4, 4, 27, 224),)
K11008_SMALL_M_CELLS = (
    (24, 1, 1, 22, 176, 11008),
)
# Exact M=2048/K=11008/N=4096 normalized source program.  The probe retains
# only the 24-AIC MTE2-to-MTE1 path: no Cube work or production duration is
# reproduced or consumed.  The final N worker is shifted inward exactly as in
# the normalized persistent launch.
K11008_M2048_CELLS = (
    (24, 8, 12, 22, 256, 11008),
)
K14336_SMALL_M_CELLS = (
    (22, 1, 1, 14, 96, 3584),
    (22, 1, 1, 28, 96, 7168),
    (22, 1, 1, 56, 96, 14336),
    (22, 1, 2, 14, 96, 3584),
    (22, 1, 2, 28, 96, 7168),
    # Exact production launch geometry, retained as the final holdout.
    (22, 1, 2, 56, 96, 14336),
)
# Exact 128x128x128 MatMulV2 descriptor geometry.  The shorter programs vary
# only finite source length; the final 3x56 cell is the production-sized
# holdout for the 22-AIC K14336/N4096 launch.
K14336_M128_N128_CELLS = (
    (22, 2, 1, 7, 128),
    (22, 2, 1, 14, 128),
    (22, 2, 2, 28, 128),
    (22, 2, 3, 56, 128),
)
# Exact source geometries for the finite-to-steady transition.  The first two
# cells keep the 128x128x128 MatMulV2 program fixed while changing only its
# worker topology and finite length.  The last cell keeps the corresponding
# 128x256x64 program fixed.  All three consume disjoint cold sources.
K5120_M128_N128_CELLS = (
    (22, 1, 5, 20, 128, 5120),
    (24, 2, 9, 20, 128, 5120),
)
K5120_M128_N256_CELLS = (
    (24, 4, 9, 10, 256, 5120),
)
K4096_M128_N256_CELLS = (
    (24, 4, 8, 8, 256, 4096),
)
K5120_M256_N112_CELLS = (
    (24, 2, 4, 20, 112, 5120),
)
# Exact normalized programs for the K8192 transition cases.  Keep the two
# descriptor geometries separate because they are compiled with different A1
# slab widths; within each profile only source topology and finite length vary.
K8192_M128_N128_CELLS = (
    (22, 1, 3, 32, 128, 8192),
    (22, 2, 6, 32, 128, 8192),
    (23, 1, 10, 32, 128, 8192),
    (24, 2, 19, 32, 128, 8192),
)
K8192_M128_N256_CELLS = (
    (24, 4, 6, 16, 256, 8192),
    # Exact longest-worker source program for the normalized
    # B512/M1/K8192/N28672 launch: a 4x6 worker grid covers 112 N tiles, so
    # four workers in each M group own the 19-tile tail-balanced program.
    (24, 4, 19, 16, 256, 8192),
)
K13824_M128_N112_CELLS = (
    (24, 2, 4, 54, 112, 13824),
)
K13824_M16_N112_CELLS = (
    # Exact narrow-M key-99025 source program: one M group, two persistent
    # N112 output tiles per worker, and 27 A1/B1 K512 slabs.
    (23, 1, 2, 27, 112, 13824),
)
# Same descriptor geometry as the normalized 128x112x128 MatMulV2 program.
# The cells independently vary the number of output tiles and K slabs; the
# production-sized 2x6 cell is retained as a holdout rather than a fit point.
K1536_N5120_FINITE_CELLS = (
    (23, 1, 1, 3, 112),
    (23, 1, 1, 6, 112),
    (23, 1, 2, 3, 112),
    (23, 1, 2, 6, 112),
    (23, 1, 3, 6, 112),
    (23, 1, 2, 12, 112),
)
K1536_M128_N256_CELLS = (
    # Finite output-tile sweep at the exact A1/B1 descriptor geometry and
    # physical K extent.  The final cell is the normalized
    # B512/M1/K1536/N5120 program; shorter cells isolate fill/drain while
    # preserving the real three-slab K program.
    (20, 4, 1, 3, 256, 1536),
    (20, 4, 2, 3, 256, 1536),
    (20, 4, 3, 3, 256, 1536),
    (20, 4, 4, 3, 256, 1536),
)
K1536_M16_N224_CELLS = (
    (23, 1, 1, 1, 224, 1536),
)
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
OWNER_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")
BASELINE_PATTERN = re.compile(r"[0-9a-f]{40}")
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_gemm_source_order_kernel.cpp",
    "aic_mte2_gemm_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_pipeline_completion_kernel.cpp",
    "aic_mte2_gemm_source_order_main.cpp",
    "aic_mte2_gemm_k1536_n5120_source_order_kernel.cpp",
    "aic_mte2_gemm_k1536_n5120_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_k1536_n5120_source_order_main.cpp",
    "aic_mte2_gemm_k14336_source_order_kernel.cpp",
    "aic_mte2_gemm_k14336_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_k14336_source_order_main.cpp",
    "aic_mte2_gemm_k13824_m16_n112_source_order_kernel.cpp",
    "aic_mte2_gemm_k13824_m16_n112_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_k13824_m16_n112_source_order_main.cpp",
    "aic_mte2_gemm_k1536_m16_n224_source_order_kernel.cpp",
    "aic_mte2_gemm_k1536_m16_n224_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_k1536_m16_n224_pipeline_completion_kernel.cpp",
    "aic_mte2_gemm_k1536_m16_n224_source_order_main.cpp",
    "aic_mte2_gemm_k14336_m128_n128_source_order_kernel.cpp",
    "aic_mte2_gemm_k14336_m128_n128_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_k14336_m128_n128_source_order_main.cpp",
    "aic_mte2_gemm_m256_n112_source_order_kernel.cpp",
    "aic_mte2_gemm_m256_n112_ingress_completion_kernel.cpp",
    "aic_mte2_gemm_m256_n112_source_order_main.cpp",
    "aiv_l2_dirty_preamble_kernel.cpp",
    "aic_mte2_gemm_source_order_runner.py",
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


def main():
    global _ACTIVE_PROCESS

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile", choices=(
            "generic", "k13824", "k11008_small_m", "k11008_m2048",
            "k14336_small_m",
            "k14336_m128_n128", "k1536_n5120_finite",
            "k1536_m128_n256", "k1536_m16_n224",
            "k5120_m128_n128", "k5120_m128_n256",
            "k4096_m128_n256", "k5120_m256_n112",
            "k8192_m128_n128", "k8192_m128_n256",
            "k13824_m128_n112", "k13824_m16_n112"),
        default="generic")
    parser.add_argument("--child-timeout-seconds", type=int, default=900)
    parser.add_argument("--dirty-preamble-mib", type=int, default=0)
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
    if args.dirty_preamble_mib < 0 or args.dirty_preamble_mib > 192:
        parser.error("--dirty-preamble-mib must be in [0,192]")

    root = Path(__file__).resolve().parent
    expected_binary_name = "aic_mte2_gemm_source_order"
    if args.profile == "generic":
        cells = tuple((*cell, 256, cell[3] * 512) for cell in CELLS)
    elif args.profile == "k13824":
        cells = tuple((*cell, 13824) for cell in K13824_CELLS)
    elif args.profile == "k11008_small_m":
        cells = K11008_SMALL_M_CELLS
        expected_binary_name = "aic_mte2_gemm_k14336_source_order"
    elif args.profile == "k11008_m2048":
        cells = K11008_M2048_CELLS
    elif args.profile == "k14336_small_m":
        cells = K14336_SMALL_M_CELLS
        expected_binary_name = "aic_mte2_gemm_k14336_source_order"
    elif args.profile == "k14336_m128_n128":
        cells = tuple((*cell, cell[3] * 256)
                      for cell in K14336_M128_N128_CELLS)
        expected_binary_name = (
            "aic_mte2_gemm_k14336_m128_n128_source_order")
    elif args.profile == "k1536_n5120_finite":
        cells = tuple((*cell, cell[3] * 256)
                      for cell in K1536_N5120_FINITE_CELLS)
        expected_binary_name = "aic_mte2_gemm_k1536_n5120_source_order"
    elif args.profile == "k1536_m128_n256":
        cells = K1536_M128_N256_CELLS
    elif args.profile == "k1536_m16_n224":
        cells = K1536_M16_N224_CELLS
        expected_binary_name = (
            "aic_mte2_gemm_k1536_m16_n224_source_order")
    elif args.profile == "k5120_m128_n128":
        cells = K5120_M128_N128_CELLS
        expected_binary_name = (
            "aic_mte2_gemm_k14336_m128_n128_source_order")
    elif args.profile == "k5120_m128_n256":
        cells = K5120_M128_N256_CELLS
    elif args.profile == "k4096_m128_n256":
        cells = K4096_M128_N256_CELLS
    elif args.profile == "k5120_m256_n112":
        cells = K5120_M256_N112_CELLS
        expected_binary_name = "aic_mte2_gemm_m256_n112_source_order"
    elif args.profile == "k8192_m128_n128":
        cells = K8192_M128_N128_CELLS
        expected_binary_name = (
            "aic_mte2_gemm_k14336_m128_n128_source_order")
    elif args.profile == "k8192_m128_n256":
        cells = K8192_M128_N256_CELLS
    elif args.profile == "k13824_m128_n112":
        cells = K13824_M128_N112_CELLS
        expected_binary_name = "aic_mte2_gemm_k1536_n5120_source_order"
    else:
        cells = K13824_M16_N112_CELLS
        expected_binary_name = (
            "aic_mte2_gemm_k13824_m16_n112_source_order")
    if args.binary.name != expected_binary_name:
        parser.error(
            f"--profile {args.profile} requires binary "
            f"{expected_binary_name!r}")
    mode_count = 3 if expected_binary_name in {
        "aic_mte2_gemm_source_order",
        "aic_mte2_gemm_k1536_m16_n224_source_order",
    } else 2
    if (args.dirty_preamble_mib > 0
            and args.profile not in {
                "k11008_m2048", "k14336_small_m", "k14336_m128_n128",
                "k4096_m128_n256", "k5120_m256_n112"}):
        parser.error(
            "--dirty-preamble-mib requires a dirty-preamble profile")
    print(
        f"hardware_id=ascend_910b owner={args.owner} "
        f"physical_device_id={args.device_id} baseline={args.baseline} "
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} runs={RUN_COUNT} trials={TRIALS} "
        f"dirty_preamble_mib={args.dirty_preamble_mib} "
        f"profile={args.profile} cells={cells} "
        f"binary={args.binary.resolve()} binary_sha256={_sha256(args.binary)} "
        f"source_snapshot_sha256={_source_snapshot_sha256(root)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,run_index,process_pid,active_aics,m_groups,n_groups,"
        "worker_m_tiles,worker_n_tiles,logical_n_tiles,"
        "output_tiles_per_worker,k_slabs,b_rows,mte2_programs_per_worker,"
        "source_row_stride,completion_pairs_per_worker,trial,mode,"
        "max_cycles,duration_ns,"
        "launch_envelope_cycles,launch_envelope_ns,validation",
        flush=True,
    )
    signal.signal(signal.SIGINT, _stop_child)
    signal.signal(signal.SIGTERM, _stop_child)
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    for (active_aics, m_groups, output_tiles, k_slabs, b_rows,
         source_row_stride) in cells:
        n_groups = active_aics // m_groups
        if args.profile == "k11008_m2048":
            worker_m_tiles, worker_n_tiles, logical_n_tiles = 2, 6, 16
        else:
            worker_m_tiles = 1
            worker_n_tiles = output_tiles
            logical_n_tiles = n_groups * worker_n_tiles
        mte2_programs = output_tiles * k_slabs
        completion_pairs = output_tiles * source_row_stride // 64
        for run_index in range(RUN_COUNT):
            command = [
                str(args.binary), "--trials", str(TRIALS),
                "--active-aics", str(active_aics),
                "--m-groups", str(m_groups),
                "--worker-m-tiles", str(worker_m_tiles),
                "--worker-n-tiles", str(worker_n_tiles),
                "--logical-n-tiles", str(logical_n_tiles),
                "--output-tiles-per-worker", str(output_tiles),
                "--k-slabs", str(k_slabs), "--b-rows", str(b_rows),
                "--source-row-stride", str(source_row_stride),
            ]
            if args.dirty_preamble_mib > 0:
                command.extend((
                    "--dirty-preamble-mib",
                    str(args.dirty_preamble_mib),
                ))
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
                print(
                    f"child_pid={process.pid} run_index={run_index} "
                    f"stderr={stderr.rstrip()}", file=sys.stderr, flush=True)
            if process.returncode != 0:
                raise RuntimeError(
                    f"child PID {process.pid} failed with "
                    f"{process.returncode}")
            lines = stdout.strip().splitlines()
            if len(lines) != mode_count * TRIALS + 1 or lines[0] != (
                    "trial,mode,max_cycles,duration_ns,launch_envelope_cycles,"
                    "launch_envelope_ns,validation"):
                raise RuntimeError("source-order child emitted invalid output")
            for line in lines[1:]:
                fields = line.split(",")
                if (len(fields) != 7
                        or fields[1] not in {
                            "mte2_only", "mte2_mte1_completion",
                            "mte2_mte1_cube_completion"}
                        or fields[-1] != "pass"):
                    raise RuntimeError("source-order child emitted invalid row")
                print(
                    f"{args.run_id},{run_index},{process.pid},{active_aics},"
                    f"{m_groups},"
                    f"{n_groups},{worker_m_tiles},{worker_n_tiles},"
                    f"{logical_n_tiles},{output_tiles},{k_slabs},{b_rows},"
                    f"{mte2_programs},{source_row_stride},"
                    f"{completion_pairs},{line}", flush=True)


if __name__ == "__main__":
    main()
