#!/usr/bin/env python3
"""Independent H200 FP8 block grouped-MoE multi-wave refill probe.

Companion of sm90_fp8_block_tile_pipeline_runner.py.  That probe measures the
fixed span and K-round interval of one sub-wave launch (64/128 CTAs).  This
probe measures what each additional resident wave of the same tile program
costs when its operands are L2-resident: the launch is a whole number W of
full resident waves (W x 132 SMs x compiled residency), and every CTA runs
the vLLM 0.24 FP8 block-scaled K loop of the pipeline runner statement for
statement.  The only change is the operand stream index, wrapped modulo
eight (N = 1024, so B = 8 x 1024 x K and A = 8 x BM x K FP8 bytes stay
L2-resident); every CTA still stores its own output tile.  Cells: W in
{1, 2, 4, 8} x K in {512, 1024, 2048} (K1024 is a held-out validation cell),
three warm launches then nine CUDA-event trials per cell, no polluter; a
``torch.cuda._sleep`` spin precedes each timed launch so that host submission
latency stays outside the event interval.  The per-wave cost at each K is the
least-squares slope of the median duration over W.

Clock gate as the pipeline runner: every 50-ms sample in the timed section
must read >= 1980 MHz.  No routing, production shape or production duration.
"""

import argparse
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl

from sm90_fp8_block_tile_pipeline_runner import GROUP, TILE_PROGRAMS, ClockLog


WRAP_STREAMS = 8
N = 1024
WAVES = (1, 2, 4, 8)
K_VALUES = (512, 1024, 2048)
SM_COUNT = 132
HOST_COVER_CYCLES = 400000
# Compiled residency of each probe program (registers/shared memory of the
# pipeline runner's COMPILE rows on sm_90): CTAs per SM.
RESIDENT_CTAS_PER_SM = {
    "bm16_bn128_bk128_w4_s3": 4,
    "bm16_bn128_bk128_w8_s4": 3,
    "bm16_bn128_bk128_w4_s4": 4,
    "bm16_bn128_bk128_w4_s5": 3,
    "bm32_bn128_bk128_w4_s3": 3,
    "bm64_bn128_bk128_w4_s4": 2,
    "bm64_bn128_bk128_w4_s3": 2,
}


@triton.jit
def fp8_block_tile_wave(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
        N, K,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        group_n: tl.constexpr, group_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, ACTIVE_ROWS: tl.constexpr,
        WRAP: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    group = pid // num_pid_n
    stream = group % WRAP
    pid_n = pid % num_pid_n
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = stream * BLOCK_SIZE_M + offs
    offs_out = group * BLOCK_SIZE_M + offs
    token_mask = offs < ACTIVE_ROWS
    off_experts = stream.to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))
    a_scale_ptrs = a_scale_ptr + offs_token * stride_asm
    offs_bsn = offs_bn // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                          mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    accumulator = accumulator.to(tl.bfloat16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_out[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_cell(program, waves, k, trials, phase, polluter=None):
    block_m, block_n, block_k, warps, stages, active_rows = TILE_PROGRAMS[program]
    grid = waves * SM_COUNT * RESIDENT_CTAS_PER_SM[program]
    panels = N // block_n
    groups = -(-grid // panels)
    # DRAM mode: every stream group owns its operands (no wrap) and the
    # polluter is summed before each launch, so operands stream from DRAM.
    wrap = WRAP_STREAMS if polluter is None else groups
    a = torch.randn(wrap * block_m, k, device="cuda").to(
        torch.float8_e4m3fn)
    b = torch.randn(wrap, N, k, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.rand(
        wrap * block_m, k // GROUP, device="cuda", dtype=torch.float32)
    b_scale = torch.rand(
        wrap, N // GROUP, k // GROUP, device="cuda",
        dtype=torch.float32)
    c = torch.empty(groups * block_m, N, device="cuda", dtype=torch.bfloat16)

    def launch():
        return fp8_block_tile_wave[(grid,)](
            a, b, c, a_scale, b_scale, N, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
            c.stride(0), c.stride(1), a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),
            group_n=GROUP, group_k=GROUP,
            BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
            ACTIVE_ROWS=active_rows, WRAP=wrap,
            num_warps=warps, num_stages=stages)

    compiled = None
    for _ in range(3):
        compiled = launch()
    torch.cuda.synchronize()
    if phase == "timed":
        print(f"COMPILE,{program},w{waves}_k{k},{compiled.n_regs},"
              f"{compiled.n_spills},{compiled.metadata.shared},"
              f"{compiled.metadata.num_warps},{compiled.metadata.num_stages}")
    for trial in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        # Keep the stream busy while the host submits the launch, so the
        # event interval excludes host launch latency (the pipeline runner
        # achieves the same with its polluter sum).
        if polluter is not None:
            torch.sum(polluter)
        torch.cuda._sleep(HOST_COVER_CYCLES)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        if phase == "timed":
            print(f"DATA,{program},{trial},w{waves}_k{k},{waves},{grid},{k},"
                  f"{start.elapsed_time(stop):.9f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", choices=sorted(TILE_PROGRAMS),
                        required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--clock-log", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(K_VALUES))
    parser.add_argument("--dram", action="store_true",
                        help="unwrapped operands + 128-MiB polluter before each launch")
    arguments = parser.parse_args()
    k_values = tuple(arguments.k_values)
    polluter = (torch.randint(0, 256, (128 * 1024 * 1024,), device="cuda", dtype=torch.uint8)
                if arguments.dram else None)
    if arguments.trials < 3 or arguments.trials > 11:
        raise ValueError("trials must be in [3, 11]")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != "NVIDIA H200"
            or properties.major != 9 or properties.minor != 0
            or properties.multi_processor_count != SM_COUNT):
        raise RuntimeError("probe requires one visible 132-SM H200")
    torch.manual_seed(0x243F6A88)
    print("META,triton," + triton.__version__ + ",torch," + torch.__version__)
    print("COMPILE,program,cell,registers_per_thread,spilled_registers,"
          "shared_memory_bytes,num_warps,num_stages")
    print("DATA,program,trial,cell,waves,grid,k,event_duration_ms")
    for waves in WAVES:
        for k in k_values:
            run_cell(arguments.program, waves, k, 3, "warm", polluter)
    clock = ClockLog(arguments.clock_log)
    clock.start()
    for waves in WAVES:
        for k in k_values:
            run_cell(arguments.program, waves, k, arguments.trials, "timed",
                     polluter)
    print("CLOCK,device,samples,busy_samples,busy_min_mhz,busy_median_mhz,"
          "busy_max_mhz,window_min_mhz,accepted")
    clock.stop()


if __name__ == "__main__":
    main()
