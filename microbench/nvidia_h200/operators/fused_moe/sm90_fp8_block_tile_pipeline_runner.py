#!/usr/bin/env python3
"""Independent H200 FP8 block-scaled grouped-MoE tile K-pipeline probe.

The FP8 counterpart of
microbench/nvidia_h200/compute/tensor_core/sm80_bf16_bm16_sparse_row_pipeline_runner.py
(the H200 BF16 BM16 ``grouped_gemm_tile_pipeline`` reference).  The probe
contract is unchanged: a synthetic sparse-row program with eight independent
weight streams, ``ACTIVE_ROWS`` valid A rows per M tile, a 128-MiB cache
polluter summed before every launch (so every weight panel streams from
DRAM), K512/K2048 x N1024/N2048 cells, CUDA-event timing, nine trials per
cell, and a 50-ms nvidia-smi clock gate that accepts a process only when every
sample in the timed section reads >= 1980 MHz.

What changes is the tile program, which now reproduces the vLLM 0.24.0
``fused_moe_kernel`` K loop for ``use_fp8_w8a8`` with a [128, 128] block
shape, statement for statement:

    a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k*BK))
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k*BK)
    a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask)
    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
    accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

with FP8 e4m3 A [rows, K] and B [E, N, K] (stride_bk = 1, stride_bn = K, the
vLLM weight layout), FP32 per-token-group A scales [rows, K/128] and FP32
block B scales [E, N/128, K/128], int64 offsets, and a masked BF16 epilogue
store.  Tile/warp/stage constants are the bundled H200 tuned configurations
the DeepSeek-V3 TP4 launches resolve to (WP0 reference
``fused_moe_tuned_config_vllm_0_24_0_h200.json``).  There is no routing,
sorting, token gather, expert lookup, router weight, activation or reduction,
and no production shape: the probe measures the launch-wide fixed span and
the K-round interval of the source-shaped tile loop only.

Every launch prints one DATA row; compile metadata (registers, shared memory)
of each cell is printed as a COMPILE row so the reference can compare it with
the production compile metadata.  No production operator or evaluation
duration is read or produced.
"""

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import triton
import triton.language as tl


STREAMS = 8
GROUP = 128
POLLUTER_BYTES = 128 * 1024 * 1024
BOOST_MHZ = 1980

# name -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, active_rows)
# Every distinct compiled tile program of the DeepSeek-V3 TP4 (N=512) and
# TP8 (N=256) bundled H200 FP8 block configs; GROUP_SIZE_M only reorders
# programs and is not part of the tile loop.  BM16 tiles keep one valid row
# (the decode routing of <= 1 row per expert tile), BM32/BM64 tiles are full.
TILE_PROGRAMS = {
    # cb47dc9924056440, 0e0b351e3086258f, 96f97a9b394c141b,
    # 63d65789df08a862, 50e8c1da81dc66d8, 65c0c8e5705f60f2
    "bm16_bn128_bk128_w4_s3": (16, 128, 128, 4, 3, 1),
    # 32e1b6206e8cd459
    "bm16_bn128_bk128_w8_s4": (16, 128, 128, 8, 4, 1),
    # c707ac95391bd157, 12d9d4bdf00fb2ff, 35591cb2c794b91b
    "bm16_bn128_bk128_w4_s4": (16, 128, 128, 4, 4, 1),
    # b336fde8d25c71c0
    "bm16_bn128_bk128_w4_s5": (16, 128, 128, 4, 5, 1),
    # a2b5ff407c7333d6
    "bm32_bn128_bk128_w4_s3": (32, 128, 128, 4, 3, 32),
    # 57bc196faf4fc400, d26bcbdc1237b406, bc3c0c641b7ec1b3
    "bm64_bn128_bk128_w4_s4": (64, 128, 128, 4, 4, 64),
    # 12974297821ec13c
    "bm64_bn128_bk128_w4_s3": (64, 128, 128, 4, 3, 64),
}
CELLS = ((512, 1024), (2048, 1024), (512, 2048), (2048, 2048))


class ClockLog:
    """50-ms nvidia-smi SM-clock sampler for the visible physical GPU."""

    def __init__(self, path):
        visible = os.environ["CUDA_VISIBLE_DEVICES"]
        if not visible.isdigit():
            raise RuntimeError("CUDA_VISIBLE_DEVICES must name one GPU index")
        self.device = int(visible)
        self.path = Path(path)
        self.process = None
        self.stream = None
        self.minimum = None

    def start(self):
        self.stream = self.path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            ["nvidia-smi", "-i", str(self.device),
             "--query-gpu=clocks.sm,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "50"],
            stdout=self.stream, stderr=subprocess.DEVNULL)
        time.sleep(0.3)

    def stop(self):
        time.sleep(0.2)
        self.process.terminate()
        self.process.wait()
        self.stream.close()
        busy = []
        samples = 0
        self.minimum = BOOST_MHZ
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                clock, utilization = (int(v) for v in line.split(","))
                samples += 1
                self.minimum = min(self.minimum, clock)
                if utilization > 0:
                    busy.append(clock)
        accepted = samples > 0 and self.minimum >= BOOST_MHZ
        print(
            f"CLOCK,{self.device},{samples},{len(busy)},"
            f"{min(busy) if busy else -1},"
            f"{statistics.median(busy) if busy else -1},"
            f"{max(busy) if busy else -1},{self.minimum},{int(accepted)}")
        return accepted


@triton.jit
def fp8_block_tile_pipeline(
        a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
        N, K,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn,
        stride_cm, stride_cn, stride_asm, stride_ask,
        stride_bse, stride_bsk, stride_bsn,
        group_n: tl.constexpr, group_k: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, ACTIVE_ROWS: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    stream = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = stream * BLOCK_SIZE_M + offs
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
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def run_cell(program, k, n, trials, polluter, phase):
    block_m, block_n, block_k, warps, stages, active_rows = TILE_PROGRAMS[program]
    grid = STREAMS * (n // block_n)
    rows = STREAMS * block_m
    a = torch.randn(rows, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(STREAMS, n, k, device="cuda").to(torch.float8_e4m3fn)
    a_scale = torch.rand(rows, k // GROUP, device="cuda", dtype=torch.float32)
    b_scale = torch.rand(
        STREAMS, n // GROUP, k // GROUP, device="cuda", dtype=torch.float32)
    c = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)

    def launch():
        return fp8_block_tile_pipeline[(grid,)](
            a, b, c, a_scale, b_scale, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(2), b.stride(1),
            c.stride(0), c.stride(1), a_scale.stride(0), a_scale.stride(1),
            b_scale.stride(0), b_scale.stride(2), b_scale.stride(1),
            group_n=GROUP, group_k=GROUP,
            BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
            ACTIVE_ROWS=active_rows, num_warps=warps, num_stages=stages)

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
    # Compile every cell before the clock-gated section.
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
