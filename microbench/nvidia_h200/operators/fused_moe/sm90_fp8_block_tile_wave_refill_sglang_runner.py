#!/usr/bin/env python3
"""Independent H200 FP8 block grouped-MoE multi-wave refill probe, SGLang 0.5.12 K loop.

The SGLang counterpart of sm90_fp8_block_tile_wave_refill_runner.py (same
contract: a whole number W of full resident waves, W x 132 SMs x compiled
residency CTAs; operand streams wrapped modulo eight so A and B stay
L2-resident; W in {1, 2, 4, 8} x K cells, three warm launches, nine
CUDA-event trials per cell behind a ``torch.cuda._sleep`` host cover; the
per-wave cost at each K is the slope of the median duration over W; with
``--dram`` every stream group owns its operands and a 128-MiB polluter is
summed before each launch).  Every CTA runs the SGLang 0.5.12 FP8
block-scaled K loop of sm90_fp8_block_tile_pipeline_sglang_runner.py
(even_Ks, swap_ab, one B-scale block for BLOCK_N <= 128, TMA descriptors for
the ``_down`` programs) statement for statement.  No routing, production shape
or production duration.  Runs in the SGLang 0.5.12 environment.
"""

import argparse
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from sm90_fp8_block_tile_pipeline_runner import GROUP, ClockLog
from sm90_fp8_block_tile_pipeline_sglang_runner import TILE_PROGRAMS


WRAP_STREAMS = 8
N = 1024
WAVES = (1, 2, 4, 8)
K_VALUES = (512, 1024, 2048)
SM_COUNT = 132
HOST_COVER_CYCLES = 400000
# Compiled residency of each probe program on sm_90 (COMPILE rows of the
# SGLang pipeline runner: 228 KiB shared memory per SM with 1 KiB reserved per
# CTA, 64 Ki registers per SM): CTAs per SM.
RESIDENT_CTAS_PER_SM = {
    "sgl_bm16_bn64_bk128_w4_s5": 4,
    "sgl_bm16_bn64_bk128_w4_s3": 7,
    "sgl_bm16_bn64_bk128_w4_s4": 5,
    "sgl_bm16_bn128_bk128_w4_s3": 4,
    "sgl_bm16_bn128_bk128_w4_s4": 3,
    "sgl_tma_bm16_bn128_bk128_w4_s2": 6,
    "sgl_tma_bm16_bn128_bk64_w4_s3": 8,
    "sgl_tma_bm16_bn256_bk64_w4_s3": 4,
}


def compiled_residency(registers, shared_bytes, warps):
    """CTAs per SM of a compiled program (sm_90 register/shared/thread caps)."""
    per_warp = -(-registers * 32 // 256) * 256
    return min(65536 // (per_warp * warps), 233472 // (shared_bytes + 1024),
               2048 // (32 * warps))


@triton.jit
def sglang_fp8_block_tile_wave(
        a_ptr, a_desc, b_ptr, b_desc, c_ptr, a_scale_ptr, b_scale_ptr,
        N, K,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        group_n: tl.constexpr, group_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, ACTIVE_ROWS: tl.constexpr,
        USE_TMA: tl.constexpr, swap_ab: tl.constexpr, WRAP: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    group = pid // num_pid_n
    stream = group % WRAP
    pid_n = pid % num_pid_n
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = stream * BLOCK_SIZE_M + offs
    offs_out = group * BLOCK_SIZE_M + offs
    token_mask = offs < ACTIVE_ROWS
    off_experts_i32 = stream
    off_experts = off_experts_i32.to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if USE_TMA:
        start_offs_m = stream * BLOCK_SIZE_M
        start_offs_n = pid_n * BLOCK_SIZE_N
    else:
        a_ptrs = a_ptr + (offs_token[:, None] * stride_am
                          + offs_k[None, :] * stride_ak)
        b_ptrs = (b_ptr + off_experts * stride_be
                  + (offs_k[:, None] * stride_bk
                     + offs_bn[None, :] * stride_bn))
    a_scale_ptrs = a_scale_ptr + offs_token * stride_asm
    if BLOCK_SIZE_N > group_n:
        offs_bsn = offs_bn // group_n
    else:
        offs_bsn = pid_n * BLOCK_SIZE_N // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
    if swap_ab:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if USE_TMA:
            a = a_desc.load([start_offs_m, k_start])
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        if USE_TMA:
            b = (b_desc.load([off_experts_i32, start_offs_n, k_start])
                 .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K).T)
        else:
            b = tl.load(b_ptrs)
        offs_ks = k_start // group_k
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                          mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
        if swap_ab:
            a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
            a_scale, b_scale = b_scale, a_scale
        if BLOCK_SIZE_N > group_n:
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
        else:
            accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
        if not USE_TMA:
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    if swap_ab:
        accumulator = tl.trans(accumulator, (1, 0))
    accumulator = accumulator.to(tl.bfloat16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_out[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_cell(program, waves, k, trials, phase, polluter=None,
             resident=None):
    (block_m, block_n, block_k, warps, stages, active_rows,
     use_tma) = TILE_PROGRAMS[program]
    swap_ab = block_m < 64 and block_n >= 64
    grid = waves * SM_COUNT * (resident if resident is not None else 1)
    panels = N // block_n
    groups = -(-grid // panels)
    wrap = WRAP_STREAMS if polluter is None else groups
    a = torch.randn(wrap * block_m, k, device="cuda").to(
        torch.float8_e4m3fn)
    b = torch.randn(wrap, N, k, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.rand(
        wrap * block_m, k // GROUP, device="cuda", dtype=torch.float32)
    b_scale = torch.rand(
        wrap, N // GROUP, k // GROUP, device="cuda", dtype=torch.float32)
    c = torch.empty(groups * block_m, N, device="cuda", dtype=torch.bfloat16)
    if use_tma:
        a_desc = TensorDescriptor(a, a.shape, a.stride(), [block_m, block_k])
        b_desc = TensorDescriptor(
            b, b.shape, b.stride(), [1, block_n, block_k])
    else:
        a_desc = b_desc = None

    def launch():
        return sglang_fp8_block_tile_wave[(grid,)](
            a, a_desc, b, b_desc, c, a_scale, b_scale, N, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
            c.stride(0), c.stride(1), a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),
            group_n=GROUP, group_k=GROUP,
            BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
            ACTIVE_ROWS=active_rows, USE_TMA=use_tma, swap_ab=swap_ab,
            WRAP=wrap, num_warps=warps, num_stages=stages)

    compiled = None
    for _ in range(3):
        compiled = launch()
    torch.cuda.synchronize()
    if phase == "residency":
        return compiled_residency(compiled.n_regs, compiled.metadata.shared,
                                  warps)
    if phase == "timed":
        print(f"COMPILE,{program},w{waves}_k{k},{compiled.n_regs},"
              f"{compiled.n_spills},{compiled.metadata.shared},"
              f"{compiled.metadata.num_warps},{compiled.metadata.num_stages}")
    for trial in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
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
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=list(K_VALUES))
    parser.add_argument("--dram", action="store_true",
                        help="unwrapped operands + 128-MiB polluter before each launch")
    arguments = parser.parse_args()
    k_values = tuple(arguments.k_values)
    polluter = (torch.randint(0, 256, (128 * 1024 * 1024,), device="cuda",
                              dtype=torch.uint8)
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
    # Residency from the compiled program itself (one-CTA-per-SM launch);
    # the decode-range programs must reproduce their recorded residency.
    resident = run_cell(arguments.program, 1, k_values[0], 0, "residency")
    if (arguments.program in RESIDENT_CTAS_PER_SM
            and RESIDENT_CTAS_PER_SM[arguments.program] != resident):
        raise RuntimeError("compiled residency disagrees with the table")
    print(f"RESIDENCY,{arguments.program},{resident}")
    for waves in WAVES:
        for k in k_values:
            run_cell(arguments.program, waves, k, 3, "warm", polluter,
                     resident)
    clock = ClockLog(arguments.clock_log)
    clock.start()
    for waves in WAVES:
        for k in k_values:
            run_cell(arguments.program, waves, k, arguments.trials, "timed",
                     polluter, resident)
    print("CLOCK,device,samples,busy_samples,busy_min_mhz,busy_median_mhz,"
          "busy_max_mhz,window_min_mhz,accepted")
    clock.stop()


if __name__ == "__main__":
    main()
