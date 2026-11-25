#!/usr/bin/env python3
"""Measure sparse BM16 grouped-MatMul source envelopes (H200).

H200 owner copy of
microbench/nvidia_a800/operators/fused_moe/sm80_bf16_bm16_sparse_w1_source_runner.py.
Probe cases, synthetic routing, source audit snippets, warm-up, polluter and
timing are unchanged.  Hardware/software coupling only:
  - device check NVIDIA H200 / 9.0 / 132 SMs (was A800 / 8.0 / 108);
  - vLLM source: the H200 environment's vLLM 0.24 ``fused_moe.py`` (the A800
    copy under vllm_src_3ver/a800 is vLLM 0.11).  vLLM 0.24 renamed
    ``invoke_fused_moe_kernel`` to ``invoke_fused_moe_triton_kernel`` and
    dropped its ``B_zp`` argument; the call passes the same values otherwise;
  - ClockLog (from the H200 sparse-row runner copy) samples the SM clock
    every 50 ms over the timed section; the capture is accepted only when
    every sample is >= 1980 MHz;
  - H200-scaled two-wave pair: the A800 N1504/N1568 pair (grid 1504/1568)
    straddles the A800 W1 resident capacity 108 x 12 = 1296 but fits one
    wave of 132 x 12 = 1584 on H200.  Two extra cases with the same source
    (4 tokens, E128, top-8, K2048, 16 or 31 active experts) at N1824/N1952
    (grid 1824/1952, midpoint 1888 = 59 N tiles; 31 x 59 = 1829 active CTAs)
    exceed 1584 and so re-exercise the two-resident-wave refill on H200.
"""

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys

import torch
import triton.language as tl


BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64
GROUP_SIZE_M = 1
POLLUTER_BYTES = 128 * 1024 * 1024
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "compute/tensor_core"))
from sm80_bf16_bm16_sparse_row_pipeline_runner import ClockLog  # noqa: E402
from vllm.model_executor.layers.fused_moe import (  # noqa: E402
    fused_moe as vllm_fused_moe,
)

VLLM_SOURCE = Path(vllm_fused_moe.__file__)
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
    (4, 128, 8, 16, 2048, (1824, 1952)),
    (4, 128, 8, 31, 2048, (1824, 1952)),
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


def _audit_source():
    source = VLLM_SOURCE.read_text(encoding="utf-8")
    missing = tuple(fragment for fragment in SOURCE_SNIPPETS
                    if fragment not in source)
    if missing:
        raise ValueError(f"vLLM grouped-MoE source changed: {missing}")
    return hashlib.sha256(VLLM_SOURCE.read_bytes()).hexdigest()


def measure(tokens, experts, topk, active_experts, k, n, trials, polluter):
    invoke_fused_moe_kernel = vllm_fused_moe.invoke_fused_moe_triton_kernel
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

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
            activation, weights, output, None, None, None,
            sorted_ids, expert_ids, post_padded, False, topk, config,
            tl.bfloat16, False, False, False, False, False)

    for _ in range(3):
        torch.sum(polluter)
        operation()
    torch.cuda.synchronize()
    samples = []
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
    print(
        "summary", tokens, experts, topk, active_experts, k, n,
        active_m_tiles, maximum_m_tiles,
        maximum_m_tiles * math.ceil(n / BLOCK_N),
        f"{statistics.median(samples):.3f}", sep=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=9)
    parser.add_argument("--clock-log", type=Path, required=True)
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
    source_sha256 = _audit_source()
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
    clock = ClockLog(arguments.clock_log)
    clock.start()
    for tokens, experts, topk, active_experts, k, widths in PROBE_CASES:
        for n in widths:
            measure(
                tokens, experts, topk, active_experts, k, n,
                arguments.trials, polluter)
    print("CLOCK,device,samples,busy_samples,busy_min_mhz,busy_median_mhz,"
          "busy_max_mhz,accepted")
    clock.stop()


if __name__ == "__main__":
    main()
