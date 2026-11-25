#!/usr/bin/env python3
"""Capture the frozen 24-AIC cold N-major GEMM B descriptor cell."""

from pathlib import Path

import aic_mte2_nd2nz_stride_runner as runner


PROFILE = "gemm_n_major_b_cold"
CELLS = (
    (64, 256, 28672, 0, 24),
    (64, 256, 28672, 1, 24),
)
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_nd2nz_stride_kernel.cpp",
    "aic_mte2_nd2nz_stride_main.cpp",
    "aic_mte2_nd2nz_stride_runner.py",
    Path(__file__).name,
)


def _cells(profile):
    if profile != PROFILE:
        raise ValueError(f"unknown benchmark profile: {profile!r}")
    return CELLS


def _execution_parameters(profile, footprint_mb, iterations):
    if profile != PROFILE:
        raise ValueError(f"unknown benchmark profile: {profile!r}")
    return (
        runner.GEMM_N_MAJOR_B_COLD_FOOTPRINT_MB
        if footprint_mb is None else footprint_mb,
        runner.GEMM_N_MAJOR_B_COLD_ITERATIONS
        if iterations is None else iterations,
    )


def main():
    runner.PROFILE_NAMES = (PROFILE,)
    runner.SOURCE_FILENAMES = SOURCE_FILENAMES
    runner._cells = _cells
    runner._execution_parameters = _execution_parameters
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
