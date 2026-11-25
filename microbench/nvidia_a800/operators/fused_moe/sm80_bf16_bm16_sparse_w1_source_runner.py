#!/usr/bin/env python3
"""Measure sparse BM16 grouped-MatMul source envelopes."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import statistics

import torch
import triton.language as tl


BLOCK_M = 16
BLOCK_N = 32
BLOCK_K = 64
GROUP_SIZE_M = 1
POLLUTER_BYTES = 128 * 1024 * 1024
VLLM_SOURCE = (
    Path(os.environ["MIDLAYER_A800_VLLM_SOURCE"])
    / "model_executor/layers/fused_moe/fused_moe.py"
)
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


def _audit_source():
    source = VLLM_SOURCE.read_text(encoding="utf-8")
    missing = tuple(fragment for fragment in SOURCE_SNIPPETS
                    if fragment not in source)
    if missing:
        raise ValueError(f"vLLM grouped-MoE source changed: {missing}")
    return hashlib.sha256(VLLM_SOURCE.read_bytes()).hexdigest()


def measure(tokens, experts, topk, active_experts, k, n, trials, polluter):
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_kernel,
    )
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
            activation, weights, output, None, None, None, None,
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
    arguments = parser.parse_args()
    if not 3 <= arguments.trials <= 11:
        raise ValueError("trials must be in [3, 11]")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one GPU must be visible")
    properties = torch.cuda.get_device_properties(0)
    if (properties.name != "NVIDIA A800-SXM4-80GB"
            or properties.major != 8 or properties.minor != 0
            or properties.multi_processor_count != 108):
        raise RuntimeError("probe requires one visible 108-SM A800")
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
    for tokens, experts, topk, active_experts, k, widths in PROBE_CASES:
        for n in widths:
            measure(
                tokens, experts, topk, active_experts, k, n,
                arguments.trials, polluter)


if __name__ == "__main__":
    main()
