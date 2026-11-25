#!/usr/bin/env python3
"""Compile-time launch facts of the BM16 grouped-GEMM tile pipeline on H200.

H200 counterpart of the A800 kernel_dataset launch observation behind the
``grouped_gemm_tile_pipeline`` launch facts (registers per thread, dynamic
shared memory, resident CTAs per SM).  No duration is measured: the script
compiles the unchanged vLLM ``fused_moe_kernel`` (W1: top_k=8, no routed
weight; W2: top_k=1, routed weight) and the isolated
``sparse_row_pipeline`` probe kernel with the BF16 BM16/BN32/BK64/GSM1
configuration on the visible H200, then reads the Triton compiled metadata
and the driver occupancy calculator.  The BM64/BN64/BK32/GSM8 W1 launch is
recorded the same way as the residency fact behind the BM64 operand probe.

Hardware coupling: device check NVIDIA H200 / sm_90 / 132 SMs; Triton 3.6
from the H200 vLLM 0.24 environment (A800 used Triton 3.4 / vLLM 0.11).
"""

import ctypes
import hashlib
import json
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "compute/tensor_core"))
from sm80_bf16_bm16_sparse_row_pipeline_runner import (  # noqa: E402
    ACTIVE_ROWS, BLOCK_K, BLOCK_M, BLOCK_N, sparse_row_pipeline,
)
from vllm.model_executor.layers.fused_moe import fused_moe as vllm_fused_moe  # noqa: E402
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)


def _occupancy(function_handle, threads, dynamic_shared):
    library = ctypes.CDLL("libcuda.so.1")
    blocks = ctypes.c_int(0)
    status = library.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks), ctypes.c_void_p(function_handle),
        ctypes.c_int(threads), ctypes.c_size_t(dynamic_shared))
    if status != 0:
        raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor={status}")
    return blocks.value


def _facts(kernel):
    compiled = [value for cache in kernel.device_caches.values()
                for value in cache[0].values()]
    if len(compiled) != 1:
        raise RuntimeError(f"expected one compiled variant, got {len(compiled)}")
    binary = compiled[0]
    binary._init_handles()
    threads = binary.metadata.num_warps * 32
    shared = binary.metadata.shared
    return {
        "registers_per_thread": binary.n_regs,
        "spilled_registers": binary.n_spills,
        "num_warps": binary.metadata.num_warps,
        "threads_per_cta": threads,
        "num_stages": binary.metadata.num_stages,
        "dynamic_shared_memory_bytes_per_cta": shared,
        "resident_ctas_per_sm": _occupancy(binary.function, threads, shared),
        "target": str(binary.metadata.target),
    }


def _fused_moe(tokens, topk, experts, k, n, config, mul_routed_weight):
    kernel = vllm_fused_moe.fused_moe_kernel
    kernel.device_caches.clear()
    route_ids = [index % experts for index in range(tokens * topk)]
    topk_ids = torch.tensor(route_ids, dtype=torch.int32,
                            device="cuda").reshape(tokens, topk)
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], experts, None)
    activation = torch.ones((tokens, k), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((experts, n, k), dtype=torch.bfloat16, device="cuda")
    output = torch.zeros((tokens, topk, n), dtype=torch.bfloat16, device="cuda")
    topk_weights = (torch.ones((tokens, topk), dtype=torch.float32,
                               device="cuda") if mul_routed_weight else None)
    vllm_fused_moe.invoke_fused_moe_triton_kernel(
        activation, weights, output, None, None, topk_weights, sorted_ids,
        expert_ids, post_padded, mul_routed_weight, topk, config,
        tl.bfloat16, False, False, False, False, False)
    torch.cuda.synchronize()
    return _facts(kernel)


def main():
    properties = torch.cuda.get_device_properties(0)
    if (torch.cuda.device_count() != 1 or properties.name != "NVIDIA H200"
            or properties.major != 9 or properties.minor != 0
            or properties.multi_processor_count != 132):
        raise RuntimeError("requires exactly one visible 132-SM H200")
    bm16 = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1}
    bm64 = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8}
    facts = {
        "bm16_w1": _fused_moe(4, 8, 128, 2048, 1536, bm16, False),
        "bm16_w2": _fused_moe(32, 1, 128, 768, 2048, bm16, True),
        "bm64_w1": _fused_moe(256, 8, 128, 2048, 1536, bm64, False),
    }
    probe = sparse_row_pipeline
    probe.device_caches.clear()
    grid = 8 * (1024 // BLOCK_N)
    probe[(grid,)](
        torch.zeros(8 * BLOCK_M * 512, device="cuda", dtype=torch.bfloat16),
        torch.zeros(8 * 1024 * 512, device="cuda", dtype=torch.bfloat16),
        torch.empty(grid * BLOCK_M * BLOCK_N, device="cuda",
                    dtype=torch.bfloat16),
        K=512, N=1024, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        ACTIVE_ROWS=ACTIVE_ROWS, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    facts["bm16_isolated_probe"] = _facts(probe)
    source = Path(vllm_fused_moe.__file__)
    print(json.dumps({
        "device_name": properties.name,
        "sm_count": properties.multi_processor_count,
        "torch": torch.__version__, "triton": triton.__version__,
        "vllm_fused_moe_source": str(source),
        "vllm_fused_moe_source_sha256": hashlib.sha256(
            source.read_bytes()).hexdigest(),
        "duration_measured": False,
        "facts": facts,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
