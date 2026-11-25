#!/usr/bin/env python3
"""Measure sparse BM16 grouped-MatMul source envelopes (H200, vLLM 0.11 kernel).

H200 owner copy of
microbench/nvidia_a800/operators/fused_moe/sm80_bf16_bm16_sparse_w1_source_runner.py
for the WP1 re-measurement of ``bm16_grouped_gemm_h2880_source_depth``.  The
same-subpath H200 file ``sm80_bf16_bm16_sparse_w1_source_runner.py`` already
belongs to WP2 (sparse second wave) and runs the vLLM 0.24 kernel, so this
copy carries a ``_vllm011`` suffix instead of overwriting it.  Probe cases,
synthetic routing, workspace clamp, warm-up, polluter, timing and output rows
are unchanged.  Hardware/stack coupling only:
  - device check ``NVIDIA H200`` / 9.0 / 132 SMs (A800: A800 / 8.0 / 108);
  - kernel source: the byte-identical vLLM 0.11 ``invoke_fused_moe_kernel``
    / ``moe_align_block_size`` segments from ``vllm_0_11_fused_moe_kernel.py``
    (audited against ``--vllm-011-source-dir``), compiled by Triton 3.6 over
    the vLLM 0.24 ``_moe_C`` sort op;
  - clock gate: ``nvidia-smi -lms 50`` samples the whole process; after each
    case's three warm-ups a 0.8 s L2-resident FP32 add spin raises the SM clock and the
    case prints ``clock,...,accepted`` (1 only if every sample in
    [first timed launch - 0.1 s, last timed launch + 0.1 s] reads >= 1980
    MHz).  The polluter still runs before every timed launch.
"""

import argparse
import math
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_moe_triton_operand_stream_runner import (  # noqa: E402
    ClockGate, _prewarm)


BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64
GROUP_SIZE_M = 1
POLLUTER_BYTES = 128 * 1024 * 1024
SOURCE_SNIPPETS = (
    "num_pid_in_group = GROUP_SIZE_M * num_pid_n",
    "offs_token[:, None] // top_k * stride_am",
    "off_experts * stride_be",
    "accumulator += tl.dot(a, b)",
)
PROBE_CASES = (
    (2, 256, 8, 16, 2048, (992, 1056)),
    (4, 128, 8, 16, 2048, (1504, 1568)),
    (4, 128, 8, 31, 2048, (1504, 1568)),
    (16, 256, 1, 16, 512, (1984, 2112)),
    (16, 128, 4, 53, 2816, (5696,)),
    (16, 128, 4, 53, 2944, (5824,)),
    (64, 128, 1, 53, 2816, (2816,)),
    (64, 128, 1, 53, 2944, (2944,)),
    (8, 128, 4, 31, 2816, (5696,)),
    (8, 128, 4, 31, 2944, (5824,)),
    (32, 128, 1, 31, 2816, (2816,)),
    (32, 128, 1, 31, 2944, (2944,)),
)


def _audit_source(source_dir):
    vllm_source = source_dir / "fused_moe.py"
    source = vllm_source.read_text(encoding="utf-8")
    missing = tuple(fragment for fragment in SOURCE_SNIPPETS
                    if fragment not in source)
    if missing:
        raise ValueError(f"vLLM grouped-MoE source changed: {missing}")
    from vllm_0_11_fused_moe_kernel import audit_against_source
    return audit_against_source(
        vllm_source, source_dir / "moe_align_block_size.py")[
            "fused_moe_sha256"]


def measure(tokens, experts, topk, active_experts, k, n, trials, polluter,
            clock_windows):
    from vllm_0_11_fused_moe_kernel import (
        invoke_fused_moe_kernel, moe_align_block_size)

    routed_rows = tokens * topk
    if not 1 <= active_experts <= min(routed_rows, experts):
        raise ValueError("active expert count is outside the routing domain")
    route_ids = [index % active_experts for index in range(routed_rows)]
    topk_ids = torch.tensor(
        route_ids, dtype=torch.int32, device="cuda").reshape(tokens, topk)
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, experts, None)
    active_m_tiles = int(post_padded.item()) // BLOCK_M
    if active_m_tiles != active_experts:
        raise RuntimeError("synthetic routing changed the active M-tile count")
    source_em = sorted_ids.numel()
    if tokens < BLOCK_M:
        source_em = min(source_em, tokens * topk * BLOCK_M)
    maximum_m_tiles = math.ceil(source_em / BLOCK_M)

    activation = torch.ones((tokens, k), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((experts, n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(
        (tokens, topk, n), dtype=torch.bfloat16, device="cuda")
    config = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
    }

    def operation():
        invoke_fused_moe_kernel(
            activation, weights, output, None, None, None, None,
            sorted_ids, expert_ids, post_padded, False, topk, config,
            tl.bfloat16, False, False, False, False, False)

    for _ in range(3):
        torch.sum(polluter)
        operation()
    torch.cuda.synchronize()
    _prewarm(torch)
    samples = []
    window_begin = time.time()
    for trial in range(trials):
        torch.sum(polluter)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        duration_ns = start.elapsed_time(stop) * 1.0e6
        samples.append(duration_ns)
        print(
            "data", trial, tokens, experts, topk, active_experts, k, n,
            active_m_tiles, maximum_m_tiles,
            maximum_m_tiles * math.ceil(n / BLOCK_N), f"{duration_ns:.3f}",
            sep=",")
    clock_windows.append((
        (tokens, experts, topk, active_experts, k, n),
        window_begin, time.time()))
    print(
        "summary", tokens, experts, topk, active_experts, k, n,
        active_m_tiles, maximum_m_tiles,
        maximum_m_tiles * math.ceil(n / BLOCK_N),
        f"{statistics.median(samples):.3f}", sep=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--vllm-011-source-dir", required=True, type=Path)
    parser.add_argument("--clock-log", required=True, type=Path)
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 11:
        raise ValueError("trials must be in [3, 11]")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != "NVIDIA H200"
            or properties.major != 9 or properties.minor != 0
            or properties.multi_processor_count != 132):
        raise RuntimeError("probe requires one visible 132-SM H200")
    visible = os.environ["CUDA_VISIBLE_DEVICES"]
    if visible not in ("0", "3"):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be H200 GPU 0 or 3")
    source_sha256 = _audit_source(arguments.vllm_011_source_dir.resolve())
    clock = ClockGate(int(visible), arguments.clock_log)
    clock.start()
    clock_windows = []
    torch.manual_seed(0x299F31D0)
    polluter = torch.randint(
        0, 256, (POLLUTER_BYTES,), device="cuda", dtype=torch.uint8)
    print("meta,device_name", properties.name, sep=",")
    print("meta,vllm_source_sha256", source_sha256, sep=",")
    print("meta,production_or_evaluation_duration_consumed,false")
    print(
        "data,trial,tokens,experts,topk,active_experts,k,n,active_m_tiles,"
        "maximum_m_tiles,grid,"
        "event_duration_ns")
    for tokens, experts, topk, active_experts, k, widths in PROBE_CASES:
        for n in widths:
            measure(
                tokens, experts, topk, active_experts, k, n,
                arguments.trials, polluter, clock_windows)
    clock.finish()
    print("clock,tokens,experts,topk,active_experts,k,n,window_samples,"
          "window_clock_mhz_min,accepted")
    for case, begin, end in clock_windows:
        verdict = clock.verdict(begin, end)
        print("clock", *case, verdict["window_samples"],
              verdict["window_clock_mhz_min"], int(verdict["accepted"]),
              sep=",")


if __name__ == "__main__":
    main()
