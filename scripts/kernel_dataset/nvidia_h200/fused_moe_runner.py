"""NVIDIA H200 owner copy of ``scripts/kernel_dataset/nvidia_a800/fused_moe_runner.py``.

Executes the H200 deployment stack (vLLM 0.24.0 / torch 2.11 / triton 3.6 in
the caller-selected runtime) under the A800 canonical NSYS NVTX
timing protocol.  Changes versus the A800 runner, all hardware/stack coupling:

* vLLM 0.11 -> 0.24 op mapping: ``fused_experts(..., inplace=False,
  activation="silu")`` becomes ``fused_experts(..., activation=MoEActivation.SILU)``
  (0.24 removed ``inplace`` and takes the ``MoEActivation`` enum); the
  deployment's unquantized BF16 MoE path is the same Triton ``fused_experts``.
  ``try_get_optimal_moe_config`` and ``moe_align_block_size`` keep their
  0.11 call forms; on H200 the config lookup reads vLLM's
  ``device_name=NVIDIA_H200`` tuned tables where they exist.
* Only the NVTX (normal) mode is kept: it is the mode the A800 canonical
  timing (``nsys_nvtx_kernel_median``) used.  The A800 device-Event mode is
  not ported.
* ``H200CaptureSession`` (``dataset_pipeline/kernel_dataset/nvidia_h200_capture.py``):
  GPU lock before the first CUDA call, device identity check (NVIDIA H200,
  132 SMs, 60 MiB L2), BF16 GEMM power warm-up before the runner's own
  warm-up, SM-clock query before NVTX push / after NVTX pop.
* Optional pre-NVTX cache sweep minimum is 480 MiB (8 x 60 MiB L2) instead of
  320 MiB (8 x 40 MiB); the canonical protocol leaves it at 0.
* The tiling side file is written into the H200 telemetry record instead of a
  separate path.

vLLM 0.30 (``installed_vllm_stack() == "vllm030"``): the unquantized MoE
oracle still ranks FLASHINFER_TRTLLM / FLASHINFER_CUTLASS behind TRITON on
SM90, so ``TritonExperts`` / ``fused_experts`` remains the deployment path,
with the same ``fused_experts(..., activation=MoEActivation.SILU)``,
``try_get_optimal_moe_config`` and ``moe_align_block_size`` signatures.  The
0.30 Triton kernel gains ``SWAP_AB`` (FP8 W8A8 only) and ``USE_TD``
(``VLLM_TRITON_USE_TD``, default on XPU only), both off for BF16 on H200; the
recorded ``moe_tiling.config`` is whatever the installed stack's
``try_get_optimal_moe_config`` returns (H200 tuned JSON or default).

Unchanged: fixed seed 42, deterministic canonical routing from
``tok_per_expert`` (``build_nvidia_a800_moe_topk_ids``), input construction,
the pre-NVTX warm-up count (0 in the canonical job), one exact public-op call
inside the NVTX range, and the shared-config tiling derivation.
"""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_h200_capture import (
    H200_MINIMUM_CACHE_EVICTION_MIB,
    H200_HARDWARE_ID,
    add_h200_capture_arguments,
    installed_vllm_stack,
    open_h200_capture_session,
)

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts, try_get_optimal_moe_config)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)

from evaluation.kernel.device_event import (
    prepare_bf16_cache_eviction_sweep,
    validate_cache_eviction_mib,
)
from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    build_nvidia_a800_moe_topk_ids,
)


# Deployment call per supported vLLM stack (identical on both).
VLLM_CALL_PATH = {
    "vllm024": ("vLLM 0.24 UnquantizedMoeBackend.TRITON -> fused_experts("
                "activation=MoEActivation.SILU)"),
    "vllm030": ("vLLM 0.30 UnquantizedMoeBackend.TRITON -> fused_experts("
                "activation=MoEActivation.SILU); SWAP_AB=False, USE_TD=False"),
}

DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def run_fused_moe(
    num_tokens,
    num_experts,
    top_k,
    hidden_size,
    intermediate_size,
    nvtx_range_name,
    capture,
    dtype="fp16",
    device="cuda",
    expected_token_counts=None,
    pre_nvtx_cache_eviction_mib=0,
    pre_nvtx_warmup_iterations=0,
):
    if (isinstance(pre_nvtx_warmup_iterations, bool)
            or not isinstance(pre_nvtx_warmup_iterations, int)
            or pre_nvtx_warmup_iterations < 0):
        raise ValueError(
            "H200 FusedMoE pre-NVTX warmup iterations must be a non-negative integer")
    if not isinstance(nvtx_range_name, str) or not nvtx_range_name:
        raise ValueError("H200 FusedMoE runner requires an NVTX name")
    vllm_stack = installed_vllm_stack()
    torch_dtype = DTYPE_MAP[dtype.lower()]
    canonical_topk_ids = None
    if expected_token_counts is not None:
        canonical_topk_ids = build_nvidia_a800_moe_topk_ids(
            token_counts=expected_token_counts,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            context="H200 FusedMoE canonical routing",
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch_dtype, device=device)
    router_logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device)
    routing_weights = torch.softmax(router_logits, dim=-1)
    topk_weights, random_topk_ids = torch.topk(
        routing_weights, top_k, dim=-1)
    if canonical_topk_ids is None:
        topk_ids = random_topk_ids
    else:
        topk_ids = torch.tensor(
            canonical_topk_ids, dtype=torch.int32, device=device)
    topk_weights = topk_weights.to(torch_dtype)
    topk_ids = topk_ids.to(torch.int32)

    w1 = torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=torch_dtype, device=device)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, dtype=torch_dtype, device=device)

    tiling_info = _extract_tiling_info(
        w1, w2, topk_ids, num_tokens, num_experts, top_k,
        hidden_size, intermediate_size)
    if expected_token_counts is not None:
        if tiling_info["tok_per_expert"] != expected_token_counts:
            raise ValueError(
                "H200 FusedMoE actual routing disagrees with canonical Attr")
    capture.add_launch_evidence("moe_tiling", tiling_info)

    def operation():
        with torch.no_grad():
            return fused_experts(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation=MoEActivation.SILU,
            )

    validate_cache_eviction_mib(
        pre_nvtx_cache_eviction_mib,
        minimum_nonzero_mib=H200_MINIMUM_CACHE_EVICTION_MIB,
        hardware_id=H200_HARDWARE_ID)
    cache_eviction_operation, _ = prepare_bf16_cache_eviction_sweep(
        torch_module=torch, device="cuda",
        cache_eviction_mib=pre_nvtx_cache_eviction_mib)
    torch.cuda.synchronize()
    capture.power_warmup(torch)
    for _ in range(pre_nvtx_warmup_iterations):
        operation()
    torch.cuda.synchronize()
    if cache_eviction_operation is not None:
        cache_eviction_operation()
        torch.cuda.current_stream().synchronize()
    capture.before_nvtx()
    torch.cuda.nvtx.range_push(nvtx_range_name)
    output = operation()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    capture.after_nvtx()
    capture.add_launch_evidence("environment", {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": importlib.metadata.version("vllm"),
        "triton_version": importlib.metadata.version("triton"),
        "input_seed": 42,
        "vllm_stack": vllm_stack,
        "vllm_call_path": VLLM_CALL_PATH[vllm_stack],
    })
    return output


def _extract_tiling_info(w1, w2, topk_ids, num_tokens, num_experts, top_k,
                         hidden_size, intermediate_size):
    """Shared Triton config + padded EM for both projections (A800 logic).

    vLLM 0.24 ``invoke_fused_moe_triton_kernel`` keeps the 0.11 small-batch
    EM reduction keyed on ``A.size(0)`` and the single shared config.
    """
    N_w1 = 2 * intermediate_size
    N_w2 = hidden_size

    config = try_get_optimal_moe_config(
        w1.size(), w2.size(), top_k, dtype=None, M=num_tokens)
    BM = config["BLOCK_SIZE_M"]
    BN = config["BLOCK_SIZE_N"]
    BK = config["BLOCK_SIZE_K"]
    GSM = config["GROUP_SIZE_M"]

    sorted_ids, _, _ = moe_align_block_size(topk_ids, BM, num_experts, None)
    sorted_sz = sorted_ids.size(0)

    EM_w1 = sorted_sz
    if num_tokens < BM:
        EM_w1 = min(sorted_sz, num_tokens * top_k * BM)

    EM_w2 = sorted_sz
    a_size_w2 = num_tokens * top_k
    if a_size_w2 < BM:
        EM_w2 = min(sorted_sz, num_tokens * top_k * BM)

    grid_w1 = math.ceil(EM_w1 / BM) * math.ceil(N_w1 / BN)
    grid_w2 = math.ceil(EM_w2 / BM) * math.ceil(N_w2 / BN)

    tok_per_expert = []
    for e in range(num_experts):
        tok_per_expert.append(int((topk_ids == e).sum().item()))

    return {
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "config": {name: config[name] for name in sorted(config)},
        "w1": {
            "N": N_w1, "K": hidden_size,
            "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GSM,
            "EM": EM_w1, "grid": grid_w1,
        },
        "w2": {
            "N": N_w2, "K": intermediate_size,
            "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN,
            "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GSM,
            "EM": EM_w2, "grid": grid_w2,
        },
        "tok_per_expert": tok_per_expert,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--num_experts", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--intermediate_size", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--nvtx_range_name", type=str, required=True)
    parser.add_argument("--pre-nvtx-cache-eviction-mib", type=int, default=0)
    parser.add_argument("--pre-nvtx-warmup-iterations", type=int, default=0)
    parser.add_argument("--expected-tok-per-expert-json")
    add_h200_capture_arguments(parser)

    args = parser.parse_args()
    expected_token_counts = (
        None
        if args.expected_tok_per_expert_json is None
        else json.loads(args.expected_tok_per_expert_json))
    capture = open_h200_capture_session(args, args.nvtx_range_name)
    torch.cuda.set_device(0)
    capture.validate_device(torch)

    run_fused_moe(
        num_tokens=args.num_tokens,
        num_experts=args.num_experts,
        top_k=args.top_k,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        nvtx_range_name=args.nvtx_range_name,
        capture=capture,
        dtype=args.dtype,
        expected_token_counts=expected_token_counts,
        pre_nvtx_cache_eviction_mib=args.pre_nvtx_cache_eviction_mib,
        pre_nvtx_warmup_iterations=args.pre_nvtx_warmup_iterations,
    )
    capture.finish()


if __name__ == "__main__":
    main()
