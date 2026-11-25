#!/usr/bin/env python3
"""vLLM 0.24.0 FUSED_MOE config selection + Triton 3.6 sm90 compile facts on H200.

H200 counterpart of the A800 record
``microbench/nvidia_a800/references/fused_moe_tuned_config_vllm_0_11_0_a800.json``
(vLLM 0.11 bundled tuned configs used by
``src/kernel/launch_config/nvidia_a800/fused_moe_launch.py``).  No duration is
measured.  The script

1. resolves, through vLLM's own ``try_get_optimal_moe_config`` (which calls
   ``get_config`` / ``get_moe_configs`` / ``get_default_config``) on the
   visible H200, the config that the deployed vLLM 0.24.0 chooses for every
   token count M = 1..32768 of the DeepSeek-V3 routed-expert GEMMs
   (E = 256 global experts, top-8, hidden 7168, moe_intermediate 2048 split
   over TP4 -> N = 512 and TP8 -> N = 256, no EP), for fp8_w8a8 with block
   [128, 128] and for BF16, and compresses the per-M choice into contiguous
   M ranges;
2. compiles the unchanged ``fused_moe_kernel`` once for every distinct
   (dtype, N, config, W1/W2) through ``invoke_fused_moe_triton_kernel`` with
   the production tensor shapes (W1: top_k = 8, no routed weight; W2:
   top_k = 1, routed weight), and records Triton's compiled metadata
   (registers, spills, shared memory, num_warps, num_stages) plus the driver
   occupancy calculator's resident CTAs/SM.

Hardware coupling: device check NVIDIA H200 / sm_90 / 132 SMs; vLLM 0.24.0 /
Triton 3.6 (A800: vLLM 0.11.0 / Triton 3.4).  Run with
``CUDA_VISIBLE_DEVICES=<one owned GPU>``.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
import vllm
from vllm.model_executor.layers.fused_moe import fused_moe as vllm_fused_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

GLOBAL_EXPERTS = 256
TOP_K = 8
HIDDEN = 7168
MOE_INTERMEDIATE = 2048
TENSOR_PARALLEL = (4, 8)
M_MAX = 32768
BLOCK_SHAPE = [128, 128]
DTYPES = ("fp8_w8a8", "bf16")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _occupancy(function_handle: int, threads: int, dynamic_shared: int) -> int:
    library = ctypes.CDLL("libcuda.so.1")
    blocks = ctypes.c_int(0)
    status = library.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks), ctypes.c_void_p(function_handle),
        ctypes.c_int(threads), ctypes.c_size_t(dynamic_shared))
    if status != 0:
        raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor={status}")
    return blocks.value


def _compiled_variants(kernel) -> dict[object, object]:
    variants = {}
    for device, cache in kernel.device_caches.items():
        for key, value in cache[0].items():
            variants[(device, key)] = value
    return variants


def _facts(binary) -> dict[str, object]:
    binary._init_handles()
    threads = binary.metadata.num_warps * 32
    shared = binary.metadata.shared
    return {
        "registers_per_thread": binary.n_regs,
        "spilled_registers": binary.n_spills,
        "num_warps": binary.metadata.num_warps,
        "threads_per_cta": threads,
        "num_stages": binary.metadata.num_stages,
        "shared_memory_bytes_per_cta": shared,
        "resident_ctas_per_sm": _occupancy(binary.function, threads, shared),
        "target": str(binary.metadata.target),
    }


def _config_source(n: int, dtype: str) -> dict[str, object]:
    config_dtype = "fp8_w8a8" if dtype == "fp8_w8a8" else None
    block_shape = BLOCK_SHAPE if dtype == "fp8_w8a8" else None
    name = vllm_fused_moe.get_config_file_name(
        GLOBAL_EXPERTS, n, config_dtype, block_shape)
    path = Path(vllm_fused_moe.__file__).parent / "configs" / name
    record = {"file_name": name, "bundled_file_exists": path.exists()}
    if path.exists():
        record["path"] = str(path.relative_to(Path(vllm_fused_moe.__file__).parent))
        record["sha256"] = _sha256(path)
        record["tuned_batch_keys"] = sorted(
            int(key) for key in json.loads(path.read_text())
            if key != "triton_version")
    return record


def _bundled_h200_configs() -> list[dict[str, object]]:
    """Every bundled config file vLLM would resolve for device NVIDIA_H200."""
    directory = Path(vllm_fused_moe.__file__).parent / "configs"
    records = []
    for path in sorted(directory.glob("E=*,N=*,device_name=NVIDIA_H200*.json")):
        fields = dict(item.split("=", 1) for item in path.stem.split(",", 3)[:3])
        tail = path.stem.split(",", 3)[3] if path.stem.count(",") >= 3 else ""
        records.append({
            "local_experts": int(fields["E"]), "hidden": int(fields["N"]),
            "selector_suffix": tail, "bf16": tail == "",
            "path": f"configs/{path.name}", "sha256": _sha256(path),
        })
    return records


def _selection_ranges(n: int, dtype: str) -> list[dict[str, object]]:
    config_dtype = "fp8_w8a8" if dtype == "fp8_w8a8" else None
    block_shape = BLOCK_SHAPE if dtype == "fp8_w8a8" else None
    w1_shape = (GLOBAL_EXPERTS, 2 * n, HIDDEN)
    w2_shape = (GLOBAL_EXPERTS, HIDDEN, n)
    ranges: list[dict[str, object]] = []
    for m in range(1, M_MAX + 1):
        config = vllm_fused_moe.try_get_optimal_moe_config(
            w1_shape, w2_shape, TOP_K, config_dtype, m, block_shape)
        config = dict(sorted(config.items()))
        if ranges and ranges[-1]["config"] == config:
            ranges[-1]["m_max"] = m
        else:
            ranges.append({"m_min": m, "m_max": m, "config": config})
    return ranges


def _effective_kernel_config(config: dict[str, int], dtype: str) -> dict[str, int]:
    """Constexpr config ``invoke_fused_moe_triton_kernel`` launches with."""
    effective = dict(config)
    effective["SPLIT_K"] = 1
    if dtype == "fp8_w8a8":
        effective["BLOCK_SIZE_K"] = min(effective["BLOCK_SIZE_K"], min(BLOCK_SHAPE))
    return effective


def _run_gemm(dtype: str, n: int, config: dict[str, int], stage: str,
              tokens: int) -> dict[str, object]:
    """Launch one production-shape fused_moe_kernel and return compile facts."""
    kernel = vllm_fused_moe.fused_moe_kernel
    device = "cuda"
    if stage == "w1":
        rows, width, top_k, mul_routed = 2 * n, HIDDEN, TOP_K, False
        a_rows = tokens
    else:
        rows, width, top_k, mul_routed = HIDDEN, n, 1, True
        a_rows = tokens * TOP_K
    route = torch.arange(a_rows * top_k, device=device, dtype=torch.int32)
    topk_ids = (route % GLOBAL_EXPERTS).reshape(a_rows, top_k)
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], GLOBAL_EXPERTS, None)
    output = torch.empty((a_rows, top_k, rows), dtype=torch.bfloat16, device=device)
    topk_weights = (torch.ones((a_rows, top_k), dtype=torch.float32, device=device)
                    if mul_routed else None)
    if dtype == "fp8_w8a8":
        a = torch.ones((a_rows, width), device=device).to(torch.float8_e4m3fn)
        b = torch.ones((GLOBAL_EXPERTS, rows, width), device=device,
                       dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        a_scale = torch.ones((a_rows, triton.cdiv(width, BLOCK_SHAPE[1])),
                             device=device, dtype=torch.float32)
        b_scale = torch.ones((GLOBAL_EXPERTS, triton.cdiv(rows, BLOCK_SHAPE[0]),
                              triton.cdiv(width, BLOCK_SHAPE[1])),
                             device=device, dtype=torch.float32)
        use_fp8, block_shape = True, BLOCK_SHAPE
    else:
        a = torch.ones((a_rows, width), device=device, dtype=torch.bfloat16)
        b = torch.ones((GLOBAL_EXPERTS, rows, width), device=device,
                       dtype=torch.bfloat16)
        a_scale = b_scale = None
        use_fp8, block_shape = False, None
    # Each (dtype, N, config, stage) is compiled from an empty cache so the
    # recorded binary is exactly the specialization of this launch.
    kernel.device_caches.clear()
    before = set(_compiled_variants(kernel))
    vllm_fused_moe.invoke_fused_moe_triton_kernel(
        a, b, output, a_scale, b_scale, topk_weights, sorted_ids, expert_ids,
        post_padded, mul_routed, top_k, dict(config), tl.bfloat16, use_fp8,
        False, False, False, False, block_shape)
    torch.cuda.synchronize()
    variants = _compiled_variants(kernel)
    new = [variants[key] for key in variants if key not in before]
    if len(new) != 1:
        raise RuntimeError(
            f"{dtype} N={n} {stage} {config}: expected one new compiled "
            f"variant, got {len(new)}")
    facts = _facts(new[0])
    facts["representative_tokens"] = tokens
    facts["top_k"] = top_k
    facts["mul_routed_weight"] = mul_routed
    del a, b, output
    torch.cuda.empty_cache()
    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    properties = torch.cuda.get_device_properties(0)
    if (torch.cuda.device_count() != 1 or properties.name != "NVIDIA H200"
            or (properties.major, properties.minor) != (9, 0)
            or properties.multi_processor_count != 132):
        raise RuntimeError("requires exactly one visible 132-SM NVIDIA H200")
    from vllm.model_executor.layers.fused_moe import get_config
    if get_config() is not None:
        raise RuntimeError("a get_config override is active")
    if "VLLM_TUNED_CONFIG_FOLDER" in os.environ:
        raise RuntimeError("VLLM_TUNED_CONFIG_FOLDER must be unset")

    shapes = []
    distinct: dict[str, dict[str, object]] = {}
    for tp in TENSOR_PARALLEL:
        n = MOE_INTERMEDIATE // tp
        for dtype in DTYPES:
            ranges = _selection_ranges(n, dtype)
            source = _config_source(n, dtype)
            for item in ranges:
                effective = _effective_kernel_config(item["config"], dtype)
                config_id = json.dumps(
                    {"dtype": dtype, "N": n, "config": effective},
                    sort_keys=True, separators=(",", ":"))
                item["kernel_config"] = effective
                item["config_id"] = hashlib.sha256(
                    config_id.encode()).hexdigest()[:16]
                if config_id not in distinct:
                    distinct[config_id] = {
                        "config_id": item["config_id"], "dtype": dtype,
                        "N": n, "tensor_parallel_size": tp,
                        "kernel_config": effective,
                        "selector_config": item["config"],
                        "representative_tokens": item["m_min"],
                    }
            shapes.append({
                "tensor_parallel_size": tp, "expert_parallel_size": 1,
                "global_experts": GLOBAL_EXPERTS, "local_experts": GLOBAL_EXPERTS,
                "top_k": TOP_K, "hidden": HIDDEN,
                "intermediate_per_rank_N": n, "dtype": dtype,
                "block_shape": BLOCK_SHAPE if dtype == "fp8_w8a8" else None,
                "config_resolution": (
                    "bundled_tuned_config_nearest_key"
                    if source["bundled_file_exists"] else
                    "get_default_config"),
                "config_file": source,
                "m_domain": [1, M_MAX],
                "selection_ranges": ranges,
            })
    compile_facts = []
    for record in distinct.values():
        for stage in ("w1", "w2"):
            facts = _run_gemm(record["dtype"], record["N"],
                              record["selector_config"], stage,
                              record["representative_tokens"])
            compile_facts.append({
                "config_id": record["config_id"], "dtype": record["dtype"],
                "N": record["N"], "stage": stage,
                "kernel_config": record["kernel_config"], **facts})
    source = Path(vllm_fused_moe.__file__)
    payload = {
        "device_name": properties.name,
        "sm_count": properties.multi_processor_count,
        "torch": torch.__version__, "triton": triton.__version__,
        "vllm": vllm.__version__,
        "vllm_fused_moe_source": str(source),
        "vllm_fused_moe_source_sha256": _sha256(source),
        "bundled_h200_configs": _bundled_h200_configs(),
        "shapes": shapes,
        "distinct_kernel_config_count": len(distinct),
        "compile_metadata": compile_facts,
        "duration_measured": False,
    }
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output),
                      "distinct_configs": len(distinct),
                      "compiled": len(compile_facts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
