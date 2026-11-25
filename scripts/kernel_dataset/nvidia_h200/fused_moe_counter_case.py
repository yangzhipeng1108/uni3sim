"""One synthetic complete BF16 FusedMoE call for H200 traffic counters.

No canonical routing, model weights, or duration is imported into this input.
The production vLLM 0.30 fused_experts entry remains the measured program.
"""

import hashlib
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path

import torch
import triton
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts, try_get_optimal_moe_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(8, 4096, 8192, 32768), required=True)
    parser.add_argument("--topk", type=int, choices=(2, 8, 16), required=True)
    parser.add_argument("--profile", choices=("e128_h2048_i768", "e256_h6144_i256", "e8_h6144_i16384"),
                        default="e128_h2048_i768")
    args = parser.parse_args()
    tokens = args.tokens
    topk = args.topk
    if args.profile == "e128_h2048_i768":
        experts, hidden_size, intermediate = 128, 2048, 768
        expected_tiling = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256,
                           "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16}
    elif args.profile == "e8_h6144_i16384":
        if tokens not in (8,4096) or topk != 2:
            raise ValueError("E8/H6144/I16384 counter requires tokens8or4096/topk2")
        experts, hidden_size, intermediate = 8, 6144, 16384
        expected_tiling = ({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,
                            "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1}
                           if tokens==8 else
                           {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128,
                            "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16})
    else:
        if tokens != 32768 or topk != 8:
            raise ValueError("E256/H6144/I256 probe requires tokens=32768 and topk=8")
        experts, hidden_size, intermediate = 256, 6144, 256
        expected_tiling = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128,
                           "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}
    fused_moe_module = importlib.import_module(
        "vllm.model_executor.layers.fused_moe.fused_moe")
    version = importlib.metadata.version("vllm")
    if version != "0.30.0":
        raise RuntimeError(f"expected vLLM 0.30.0, got {version}")
    properties = torch.cuda.get_device_properties(0)
    if torch.cuda.device_count() != 1 or "H200" not in properties.name or properties.multi_processor_count != 132:
        raise RuntimeError("requires one 132-SM H200")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    hidden = torch.randn(tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    logits = torch.randn(tokens, experts, dtype=torch.float32, device="cuda")
    weights, ids = torch.topk(torch.softmax(logits, dim=-1), topk, dim=-1)
    weights = weights.to(torch.bfloat16)
    ids = ids.to(torch.int32)
    up = torch.randn(experts, 2 * intermediate, hidden_size,
                     dtype=torch.bfloat16, device="cuda")
    down = torch.randn(experts, hidden_size, intermediate,
                       dtype=torch.bfloat16, device="cuda")
    config = try_get_optimal_moe_config(
        up.size(), down.size(), topk, dtype=None, M=tokens)
    tiling = {key: config[key] for key in (
        "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")}
    if tiling != expected_tiling:
        raise RuntimeError(f"unexpected FusedMoE tile for this probe: {tiling}")
    histogram = torch.bincount(ids.flatten(), minlength=experts).cpu().tolist()
    if len(histogram) != experts or sum(histogram) != tokens * topk:
        raise RuntimeError("synthetic FusedMoE routing is invalid")

    def operation():
        with torch.no_grad():
            return fused_experts(
                hidden, up, down, weights, ids,
                activation=MoEActivation.SILU)

    operation()
    torch.cuda.synchronize()
    output = operation()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("FusedMoE output is nonfinite")
    source_path = Path(fused_moe_module.__file__).resolve()
    print("MIDLAYE_COUNTER_CASE=" + json.dumps({
        "vllm_version": version,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_sm_count": properties.multi_processor_count,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "profile": args.profile,
        "shape": {"tokens": tokens, "experts": experts, "topk": topk,
                  "hidden": hidden_size, "intermediate": intermediate},
        "tiling": tiling,
        "tokens_per_expert": histogram,
        "output_shape": list(output.shape),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
