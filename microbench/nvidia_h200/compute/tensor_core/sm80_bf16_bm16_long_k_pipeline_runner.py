#!/usr/bin/env python3
"""Measure the long-K steady interval of the isolated SM80-class BM16 pipeline.

H200 owner copy of
microbench/nvidia_a800/compute/tensor_core/sm80_bf16_bm16_long_k_pipeline_runner.py.
Kernel (imported from the H200 sparse-row runner copy), 2 streams x N4096
(256 CTAs), K values, warm-up, polluter and timing are unchanged.  Hardware
coupling only: device check NVIDIA H200 / 9.0 / 132 SMs, Triton 3.6, and the
50-ms ClockLog over the timed section (capture rejected downstream when any
busy sample is below 1980 MHz).  256 CTAs remain one wave on 132 SMs.
"""

import argparse
from pathlib import Path
import statistics

import torch

from sm80_bf16_bm16_sparse_row_pipeline_runner import (
    ACTIVE_ROWS,
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    ClockLog,
    POLLUTER_BYTES,
    sparse_row_pipeline,
)


STREAMS = 2
N = 4096
K_VALUES = (2048, 4096, 7168, 14336)


def _run_case(k, trials, polluter):
    grid = STREAMS * (N // BLOCK_N)
    activation = torch.randn(
        STREAMS * BLOCK_M * k, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        STREAMS * N * k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(
        grid * BLOCK_M * BLOCK_N, device="cuda", dtype=torch.bfloat16)
    kernel = sparse_row_pipeline[(grid,)]
    for _ in range(3):
        torch.sum(polluter)
        kernel(
            activation, weights, output, K=k, N=N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            ACTIVE_ROWS=ACTIVE_ROWS, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    logical_bytes = grid * (k // BLOCK_K) * (
        ACTIVE_ROWS * BLOCK_K * 2 + BLOCK_K * BLOCK_N * 2)
    durations_ms = []
    for trial in range(trials):
        torch.sum(polluter)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel(
            activation, weights, output, K=k, N=N,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            ACTIVE_ROWS=ACTIVE_ROWS, num_warps=4, num_stages=2)
        stop.record()
        stop.synchronize()
        duration_ms = start.elapsed_time(stop)
        durations_ms.append(duration_ms)
        print(
            f"DATA,{trial},{k},{grid},{logical_bytes},"
            f"{duration_ms:.9f}")
    return statistics.median(durations_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--clock-log", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.trials < 3 or arguments.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != "NVIDIA H200"
            or properties.major != 9 or properties.minor != 0
            or properties.multi_processor_count != 132):
        raise RuntimeError("probe requires one visible 132-SM H200")
    torch.manual_seed(0x13198A2E)
    polluter = torch.randint(
        0, 256, (POLLUTER_BYTES,), device="cuda", dtype=torch.uint8)
    print("DATA,trial,k,grid,logical_bytes,event_duration_ms")
    clock = ClockLog(arguments.clock_log)
    clock.start()
    medians = {
        k: _run_case(k, arguments.trials, polluter) for k in K_VALUES}
    print("CLOCK,device,samples,busy_samples,busy_min_mhz,busy_median_mhz,"
          "busy_max_mhz,accepted")
    clock.stop()
    print("SUMMARY,k,median_duration_ms")
    for k in K_VALUES:
        print(f"SUMMARY,{k},{medians[k]:.9f}")
    for lower, upper in zip(K_VALUES, K_VALUES[1:]):
        rounds = (upper - lower) // BLOCK_K
        interval_ns = (medians[upper] - medians[lower]) * 1.0e6 / rounds
        print(f"INTERVAL,{lower},{upper},{interval_ns:.9f}")


if __name__ == "__main__":
    main()
