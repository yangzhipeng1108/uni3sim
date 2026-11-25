#!/usr/bin/env python3
"""Derive the H200 runtime memory policies from real vLLM startup logs.

For every ShareGPT model, the vLLM server startup logs of the H200 capture
(``<capture-root>/<model>/<case>/trace_off/logs/vllm_server.stdout.log``, all
cases and ranks of the model) report vLLM's own pre-KV accounting::

    Model loading took W GiB ...
    Actual usage is C GiB for consumed memory (weights + non-torch),
    A GiB for peak activation, and G GiB for CUDAGraph memory.
    Available KV cache memory: K GiB

with ``K = utilization * total - C - A`` (checked for every log).  The policy
therefore reserves ``activation = A`` and ``framework = C - W`` (non-torch
bytes); CUDA Graph memory is allocated after the KV cache is sized, so it is
no additional pre-KV reserve, and communication/allocator/fragmentation are
already inside ``C``.  Each reserve takes the largest value over the model's
cases and ranks.  The logs are rounded to 0.01 GiB by vLLM.

Writes ``configs/runtime_memory_policy/nvidia_h200/<model>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.serving.kv_cache.kv_capacity import load_runtime_memory_policy  # noqa: E402

HARDWARE_CONFIG = "configs/hardware/nvidia_h200.yaml"
GIB = 1 << 30
_MODELS = {
    # model id: (model config path, TP, EP)
    "qwen3-4b": ("configs/model/qwen3-4b.json", 1, 1),
    "qwen3-8b": ("configs/model/qwen3-8b.json", 1, 1),
    "llama-3.1-8b": ("configs/model/llama-3.1-8b-instruct.json", 1, 1),
    "qwen3-32b": ("configs/model/qwen3-32b.json", 2, 1),
    "qwen3-30b-a3b": ("configs/model/qwen3-30b-a3b.json", 2, 2),
}
_WEIGHTS = re.compile(r"\((?P<rank>[^)]*) pid=\d+\).*Model loading took (?P<gib>[0-9.]+) GiB")
_USAGE = re.compile(
    r"\((?P<rank>[^)]*) pid=\d+\).*Desired GPU memory utilization is "
    r"\((?P<util>[0-9.]+), (?P<desired>[0-9.]+) GiB\)\. Actual usage is "
    r"(?P<consumed>[0-9.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"(?P<activation>[0-9.]+) GiB for peak activation, and (?P<graph>[0-9.]+) GiB "
    r"for CUDAGraph memory")
_KV = re.compile(r"\((?P<rank>[^)]*) pid=\d+\).*Available KV cache memory: (?P<gib>[0-9.]+) GiB")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_by_rank(pattern: re.Pattern, text: str, label: str) -> dict[str, re.Match]:
    matches = {}
    for match in pattern.finditer(text):
        rank = re.sub(r"\x1b\[[0-9;]*m", "", match.group("rank"))
        if rank in matches:
            raise ValueError(f"{label} appears twice for rank {rank}")
        matches[rank] = match
    if not matches:
        raise ValueError(f"no {label} line")
    return matches


def _log_observations(log: Path) -> list[dict[str, object]]:
    text = log.read_text(errors="replace")
    weights = _one_by_rank(_WEIGHTS, text, "model loading")
    usage = _one_by_rank(_USAGE, text, "memory usage")
    kv = _one_by_rank(_KV, text, "available KV cache")
    if set(weights) != set(usage):
        raise ValueError(f"{log}: rank sets differ between weights and usage")
    observations = []
    for rank in sorted(usage):
        u = usage[rank]
        consumed = float(u.group("consumed"))
        activation = float(u.group("activation"))
        desired = float(u.group("desired"))
        observations.append({
            "log": log, "rank": rank,
            "weights_gib": float(weights[rank].group("gib")),
            "consumed_gib": consumed, "activation_gib": activation,
            "cuda_graph_gib": float(u.group("graph")),
            "utilization": float(u.group("util")), "desired_gib": desired,
        })
    # vLLM prints the KV size once (rank 0); check its accounting identity.
    for match in kv.values():
        kv_gib = float(match.group("gib"))
        if not any(abs(item["desired_gib"] - item["consumed_gib"]
                       - item["activation_gib"] - kv_gib) <= 0.011
                   for item in observations):
            raise ValueError(f"{log}: KV {kv_gib} GiB != desired - consumed - activation")
    return observations


def _reserve(kind: str, value_bytes: int, source_kind: str, reference: str,
             digest: str) -> dict[str, object]:
    return {
        "kind": kind,
        "bytes": value_bytes,
        "provenance": {
            "source_kind": source_kind,
            "source_reference": reference,
            "source_sha256": digest,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capture-root", type=Path, required=True,
                        help="<mirror>/<vLLM version>/nvidia_h200")
    args = parser.parse_args()
    capture_root = args.capture_root.resolve(strict=True)
    hardware_sha = _sha256(ROOT / HARDWARE_CONFIG)
    for model_id, (model_path, tp, ep) in _MODELS.items():
        # vLLM writes its INFO accounting to the server's stdout log.
        logs = sorted((capture_root / model_id).glob("*/trace_off/logs/vllm_server.stdout.log"))
        observations = [item for log in logs for item in _log_observations(log)]
        if not observations:
            raise ValueError(f"no startup accounting for {model_id}")
        utilizations = {item["utilization"] for item in observations}
        if utilizations != {0.8}:
            raise ValueError(f"{model_id}: unexpected utilization {utilizations}")
        activation = max(observations, key=lambda item: item["activation_gib"])
        framework = max(observations,
                        key=lambda item: item["consumed_gib"] - item["weights_gib"])
        summary = "; ".join(
            f"{item['log'].relative_to(capture_root)}[{item['rank']}]: W={item['weights_gib']} "
            f"C={item['consumed_gib']} A={item['activation_gib']} G={item['cuda_graph_gib']}"
            for item in observations)
        header = (f"vLLM H200 startup memory accounting, max over the model's "
                  f"cases and ranks ({summary})")
        activation_bytes = round(activation["activation_gib"] * GIB)
        framework_bytes = round((framework["consumed_gib"] - framework["weights_gib"]) * GIB)
        no_reserve = ("vLLM sizes the KV cache as utilization*total - consumed - peak "
                      "activation (checked in every log), so this category adds no "
                      "independent pre-KV reserve")
        policy = {
            "schema_version": 2,
            "scheduler_pools": [{
                "pool_name": "main",
                "binding": {
                    "hardware_id": "nvidia_h200",
                    "hardware_config_sha256": hardware_sha,
                    "model_config_sha256": _sha256(ROOT / model_path),
                    "deployment": "colocated",
                    "worker_role": "colocated",
                    "tensor_parallel_size": tp,
                    "pipeline_parallel_size": 1,
                    "expert_parallel_size": ep,
                    "max_model_len": 8192,
                    "max_batched_tokens": 8192,
                    "max_requests": 256,
                    "kv_block_size_tokens": 16,
                },
                "gpu_memory_utilization": {
                    "numerator": 4,
                    "denominator": 5,
                    "provenance": {
                        "source_kind": "runtime_configuration",
                        "source_reference": (
                            f"{activation['log'].relative_to(capture_root)}: vLLM "
                            "'Desired GPU memory utilization is (0.8, ...)' of the H200 capture"),
                        "source_sha256": _sha256(activation["log"]),
                    },
                },
                "non_kv_reserves": [
                    _reserve("activation", activation_bytes, "runtime_configuration",
                             f"{header}: peak activation "
                             f"{activation['activation_gib']} GiB from "
                             f"{activation['log'].relative_to(capture_root)}[{activation['rank']}]",
                             _sha256(activation["log"])),
                    *(_reserve(kind, 0, "framework_source", f"{header}: {no_reserve}",
                               _sha256(activation["log"]))
                      for kind in ("cuda_graph", "communication", "allocator", "fragmentation")),
                    _reserve("framework", framework_bytes, "runtime_configuration",
                             f"{header}: non-torch = consumed {framework['consumed_gib']} GiB "
                             f"- weights {framework['weights_gib']} GiB from "
                             f"{framework['log'].relative_to(capture_root)}[{framework['rank']}]",
                             _sha256(framework["log"])),
                ],
                "reserve_scope": "verified_uniform_across_pool_ranks",
                "completeness_assertion": "all_non_kv_reserves_for_bound_runtime",
            }],
        }
        destination = ROOT / "configs/runtime_memory_policy/nvidia_h200" / f"{model_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(policy, indent=2) + "\n")
        load_runtime_memory_policy(destination)
        print(destination.relative_to(ROOT), activation_bytes, framework_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
