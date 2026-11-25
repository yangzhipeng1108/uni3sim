#!/usr/bin/env python3
"""Emit a reproducible projected H200 PD per-rank memory ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.serving.kv_cache.model_residency import model_rank_residencies
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.workload.model_config import load_model_config
from src.kernel.hardware_spec import HWSpec

H200_CONFIG = Path("configs/hardware/nvidia_h200.yaml")


def ledger(model_path: Path, output: Path) -> None:
    model = load_model_config(model_path)
    hardware = HWSpec(H200_CONFIG)
    h200_bytes = hardware.resource_by_role("dram").size_bytes
    parallelism = ParallelConfig(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
    )
    residencies = model_rank_residencies(
        model,
        parallelism,
        worker_role="prefill",
        include_mtp=model.num_mtp_layers > 0,
    )
    if not residencies:
        raise ValueError("model residency produced no EP8 rank")
    weight_bytes = residencies[0].weight_resident_bytes
    kv_bytes = model.kv_bytes_per_token_per_parallel_rank(8, 1)
    weight_gib = weight_bytes / 1024**3
    budget_gib = h200_bytes * 0.9 / 1024**3 - weight_gib - 2.0
    if budget_gib <= 0:
        raise ValueError("model does not fit the projected H200 rank budget")
    quantization = (
        "FP8 Linear storage with BF16 compute"
        if model.weight_dtype == "fp8"
        else "INT4 group32 Linear storage with BF16 ignored modules"
        if model.weight_dtype == "mixed_int4_bf16"
        else f"unquantized {model.weight_dtype} weights")
    payload = {
        "schema_version": 2,
        "model": str(model_path.resolve()),
        "hardware": "nvidia_h200 (141 GiB per rank; H200 public spec, A800-proxy mechanisms)",
        "deployment": "P pool 8xH200 + D pool 8xH200; attention DP8/MoE EP8",
        "parallelism": {
            "tensor_parallel_size": 8,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 8,
        },
        "model_config_sha256": model.config_sha256,
        "model_type": model.model_type,
        "compute_dtype": model.compute_dtype,
        "weight_dtype": model.weight_dtype,
        "quantization_semantics": quantization,
        "weight_resident_bytes_per_ep8_rank": weight_bytes,
        "weight_gib_per_ep8_rank_estimate": weight_gib,
        "kv_bytes_per_token_estimate": kv_bytes,
        "h200_rank_capacity_gib": h200_bytes / 1024**3,
        "gpu_memory_utilization": 0.9,
        "non_kv_reserve_gib": 2.0,
        "kv_budget_gib_at_90pct": budget_gib,
        "assumptions": [
            "Rank weights and KV bytes are computed from the validated ModelConfig and model residency contract.",
            "The GLM FP8 variant binds the byte-identical official architecture config to the sidecar FP8 deployment provenance.",
            "A800 mechanism tree is reused for launch and predictor timing.",
            "KV budget uses 90% HBM utilization and a 2 GiB aggregate framework reserve.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ledger(args.model, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
