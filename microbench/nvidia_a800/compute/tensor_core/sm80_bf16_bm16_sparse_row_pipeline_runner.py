#!/usr/bin/env python3
"""Independent SM80 BM16 sparse-row load-to-HMMA pipeline probe."""

import argparse
import statistics

import torch
import triton
import triton.language as tl


STREAMS = 8
BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64
ACTIVE_ROWS = 1
POLLUTER_BYTES = 128 * 1024 * 1024


@triton.jit
def sparse_row_pipeline(
        activation, weights, output, K: tl.constexpr, N: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr, ACTIVE_ROWS: tl.constexpr):
    program = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    stream = program // n_tiles
    panel = program % n_tiles
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        a = tl.load(
            activation + stream * K
            + rows[:, None] * K + k_start + reduction[None, :],
            mask=(rows[:, None] < ACTIVE_ROWS), other=0.0)
        b = tl.load(
            weights + stream * N * K
            + (panel * BLOCK_N + columns[None, :]) * K
            + k_start + reduction[:, None])
        accumulator = tl.dot(a, b, acc=accumulator)
    tl.store(
        output + program * BLOCK_M * BLOCK_N
        + rows[:, None] * BLOCK_N + columns[None, :],
        accumulator.to(tl.bfloat16),
        mask=(rows[:, None] < ACTIVE_ROWS))


def run_case(name, k, n, trials, polluter):
    grid = STREAMS * (n // BLOCK_N)
    activation = torch.randn(
        STREAMS * BLOCK_M * k, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(
        STREAMS * n * k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(
        grid * BLOCK_M * BLOCK_N, device="cuda", dtype=torch.bfloat16)
    kernel = sparse_row_pipeline[(grid,)]
    for _ in range(3):
        torch.sum(polluter)
        kernel(
            activation, weights, output, K=k, N=n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            ACTIVE_ROWS=ACTIVE_ROWS,
            num_warps=4, num_stages=2)
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
            activation, weights, output, K=k, N=n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            ACTIVE_ROWS=ACTIVE_ROWS,
            num_warps=4, num_stages=2)
        stop.record()
        stop.synchronize()
        duration_ms = start.elapsed_time(stop)
        durations_ms.append(duration_ms)
        print(
            f"DATA,{trial},{name},{grid},{logical_bytes},"
            f"{duration_ms:.9f},{logical_bytes / duration_ms / 1.0e9:.9f}")
    return statistics.median(durations_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.trials < 3 or arguments.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != "NVIDIA A800-SXM4-80GB"
            or properties.major != 8 or properties.minor != 0
            or properties.multi_processor_count != 108):
        raise RuntimeError("probe requires one visible 108-SM A800")
    torch.manual_seed(0x243F6A88)
    polluter = torch.randint(
        0, 256, (POLLUTER_BYTES,), device="cuda", dtype=torch.uint8)
    print("DATA,trial,projection,grid,logical_bytes,event_duration_ms,logical_tb_s")
    for k, n in ((512, 1024), (2048, 1024),
                 (512, 2048), (2048, 2048)):
        run_case(f"k{k}_n{n}", k, n, arguments.trials, polluter)


if __name__ == "__main__":
    main()
