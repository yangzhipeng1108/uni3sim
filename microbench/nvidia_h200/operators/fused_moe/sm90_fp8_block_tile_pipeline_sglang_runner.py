#!/usr/bin/env python3
"""Independent H200 FP8 block-scaled grouped-MoE tile K-pipeline probe, SGLang 0.5.12 K loop.

The SGLang counterpart of sm90_fp8_block_tile_pipeline_runner.py (same probe
contract: eight synthetic weight streams, ACTIVE_ROWS valid A rows per M
tile, a 128-MiB polluter summed before every launch so every weight panel
streams from DRAM, K512/K2048 x N1024/N2048 cells, nine CUDA-event trials per
cell, 50-ms nvidia-smi clock gate).  The tile program reproduces the K loop
of sglang 0.5.12 ``moe_runner/triton_utils/fused_moe_triton_kernels.py``
``fused_moe_kernel`` for ``use_fp8_w8a8`` with a [128, 128] block, statement
for statement, with SGLang's launch-time specialisations:

* ``even_Ks`` (K % BLOCK_SIZE_K == 0): A loads masked by token only, B
  unmasked;
* ``swap_ab`` (sm90, BLOCK_SIZE_M < 64 and BLOCK_SIZE_N >= 64): the dot runs
  as B^T A^T and the scales swap;
* ``BLOCK_SIZE_N <= group_n``: one B-scale block per program
  (``offs_bsn = pid_n * BLOCK_SIZE_N // group_n``);
* ``USE_TMA`` (the ``_down`` configs): A and B are read through TMA tensor
  descriptors (``a_desc.load([start_offs_m, k_start])``,
  ``b_desc.load([expert, start_offs_n, k_start]).reshape(BN, BK).T``), A
  from rows sorted per expert (``c_sorted`` layout).

Tile/warp/stage constants are the SGLang triton_3_5_1 H200 configs
``E=257,N=256,...,fp8_w8a8,block_shape=[128, 128].json`` and ``_down.json``.
No routing, sorting, gather, router weight, activation or reduction and no
production shape; the probe measures the launch-wide fixed span and the
K-round interval of the source-shaped tile loop only.  Must run in the
SGLang 0.5.12 environment (Triton 3.5.1).
"""

import argparse
from pathlib import Path
import statistics

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from sm90_fp8_block_tile_pipeline_runner import (
    CELLS, GROUP, POLLUTER_BYTES, STREAMS, ClockLog)

# name -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, active_rows,
#          use_tma).  Decode-range (M <= 256) programs of the SGLang H200
# E=257, N=256 configs; swap_ab follows from BM < 64 and BN >= 64.
TILE_PROGRAMS = {
    # up: M 1, 2, 4, 8, 32, 48, 96
    "sgl_bm16_bn64_bk128_w4_s5": (16, 64, 128, 4, 5, 1, False),
    # up: M 16, 128
    "sgl_bm16_bn64_bk128_w4_s3": (16, 64, 128, 4, 3, 1, False),
    # up: M 64
    "sgl_bm16_bn64_bk128_w4_s4": (16, 64, 128, 4, 4, 1, False),
    # up: M 24
    "sgl_bm16_bn128_bk128_w4_s3": (16, 128, 128, 4, 3, 1, False),
    # up: M 256
    "sgl_bm16_bn128_bk128_w4_s4": (16, 128, 128, 4, 4, 1, False),
    # down: M 1, 4, 8, 16, 64, 96, 128, 256
    "sgl_tma_bm16_bn128_bk128_w4_s2": (16, 128, 128, 4, 2, 1, True),
    # down: M 2, 24, 48
    "sgl_tma_bm16_bn128_bk64_w4_s3": (16, 128, 64, 4, 3, 1, True),
    # down: M 32
    "sgl_tma_bm16_bn256_bk64_w4_s3": (16, 256, 64, 4, 3, 1, True),
    # prefill range (full tiles): up M 512 / 1024, down M 512 / 1024
    "sgl_bm32_bn128_bk128_w4_s5": (32, 128, 128, 4, 5, 32, False),
    "sgl_bm64_bn128_bk128_w4_s4": (64, 128, 128, 4, 4, 64, False),
    "sgl_tma_bm32_bn128_bk128_w4_s2": (32, 128, 128, 4, 2, 32, True),
    "sgl_tma_bm64_bn128_bk128_w4_s3": (64, 128, 128, 4, 3, 64, True),
}


@triton.jit
def sglang_fp8_block_tile_pipeline(
        a_ptr, a_desc, b_ptr, b_desc, c_ptr, a_scale_ptr, b_scale_ptr,
        N, K,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        group_n: tl.constexpr, group_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, ACTIVE_ROWS: tl.constexpr,
        USE_TMA: tl.constexpr, swap_ab: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    stream = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = stream * BLOCK_SIZE_M + offs
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
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_cell(program, k, n, trials, polluter, phase):
    (block_m, block_n, block_k, warps, stages, active_rows,
     use_tma) = TILE_PROGRAMS[program]
    swap_ab = block_m < 64 and block_n >= 64
    grid = STREAMS * (n // block_n)
    rows = STREAMS * block_m
    a = torch.randn(rows, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(STREAMS, n, k, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.rand(rows, k // GROUP, device="cuda", dtype=torch.float32)
    b_scale = torch.rand(
        STREAMS, n // GROUP, k // GROUP, device="cuda", dtype=torch.float32)
    c = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
    if use_tma:
        a_desc = TensorDescriptor(a, a.shape, a.stride(), [block_m, block_k])
        b_desc = TensorDescriptor(
            b, b.shape, b.stride(), [1, block_n, block_k])
    else:
        a_desc = b_desc = None

    def launch():
        return sglang_fp8_block_tile_pipeline[(grid,)](
            a, a_desc, b, b_desc, c, a_scale, b_scale, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
            c.stride(0), c.stride(1), a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),
            group_n=GROUP, group_k=GROUP,
            BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
            ACTIVE_ROWS=active_rows, USE_TMA=use_tma, swap_ab=swap_ab,
            num_warps=warps, num_stages=stages)

    compiled = None
    for _ in range(3):
        torch.sum(polluter)
        compiled = launch()
    torch.cuda.synchronize()
    if phase == "timed":
        print(f"COMPILE,{program},k{k}_n{n},{compiled.n_regs},"
              f"{compiled.n_spills},{compiled.metadata.shared},"
              f"{compiled.metadata.num_warps},{compiled.metadata.num_stages}")
    k_rounds = k // block_k
    logical_bytes = grid * k_rounds * (
        active_rows * block_k + block_k * block_n
        + 4 * active_rows + 4)
    durations_ms = []
    for trial in range(trials):
        torch.sum(polluter)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        duration_ms = start.elapsed_time(stop)
        durations_ms.append(duration_ms)
        print(f"DATA,{phase},{program},{trial},k{k}_n{n},{grid},{k_rounds},"
              f"{logical_bytes},{duration_ms:.9f}")
    return statistics.median(durations_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", choices=sorted(TILE_PROGRAMS),
                        required=True)
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
    torch.manual_seed(0x243F6A88)
    polluter = torch.randint(
        0, 256, (POLLUTER_BYTES,), device="cuda", dtype=torch.uint8)
    print("META,triton," + triton.__version__ + ",torch," + torch.__version__)
    print("COMPILE,program,cell,registers_per_thread,spilled_registers,"
          "shared_memory_bytes,num_warps,num_stages")
    print("DATA,phase,program,trial,cell,grid,k_rounds,logical_bytes,"
          "event_duration_ms")
    for k, n in CELLS:
        run_cell(arguments.program, k, n, 3, polluter, "warm")
    clock = ClockLog(arguments.clock_log)
    clock.start()
    for k, n in CELLS:
        run_cell(arguments.program, k, n, arguments.trials, polluter,
                 "timed")
    print("CLOCK,device,samples,busy_samples,busy_min_mhz,busy_median_mhz,"
          "busy_max_mhz,window_min_mhz,accepted")
    clock.stop()


if __name__ == "__main__":
    main()
