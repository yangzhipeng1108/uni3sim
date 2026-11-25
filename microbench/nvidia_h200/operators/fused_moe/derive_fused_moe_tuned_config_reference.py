#!/usr/bin/env python3
"""Derive the H200 FUSED_MOE tuned-config reference from the runner output.

Input: the JSON written on target host by
``fused_moe_tuned_config_vllm_0_24_0_runner.py`` (copied back into
``profiler_output/microbench/nvidia_h200/<run>/``).  Output:
``microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json``
with the keys of the A800 record
``microbench/nvidia_a800/references/fused_moe_tuned_config_vllm_0_11_0_a800.json``
(schema_version, hardware_id, evidence_kind, the duration/hardware-execution
flags, deployment, source, bundled_bf16_config_domains, forward_scope) plus the
H200 sections the port needs: the per-M config selection for the DeepSeek-V3
shapes, the Triton sm90 compile metadata of every distinct config, and
provenance.  Pure re-organisation: no value is computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).resolve().with_name(
    "fused_moe_tuned_config_vllm_0_24_0_runner.py")
OUTPUT = (ROOT / "microbench" / "nvidia_h200" / "references"
          / "fused_moe_tuned_config_vllm_0_24_0_h200.json")
A800_REFERENCE = (ROOT / "microbench" / "nvidia_a800" / "references"
                  / "fused_moe_tuned_config_vllm_0_11_0_a800.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--remote-raw", required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    if (raw["device_name"] != "NVIDIA H200" or raw["sm_count"] != 132
            or raw["vllm"] != "0.24.0" or raw["duration_measured"] is not False):
        raise ValueError("raw output is not an H200 vLLM 0.24.0 no-duration run")
    a800_keys = set(json.loads(A800_REFERENCE.read_text()))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    bf16 = [
        {key: record[key] for key in ("local_experts", "hidden", "path", "sha256")}
        for record in raw["bundled_h200_configs"] if record["bf16"]]
    compile_by_id: dict[str, dict[str, object]] = {}
    for fact in raw["compile_metadata"]:
        if fact["config_id"] not in compile_by_id:
            compile_by_id[fact["config_id"]] = {
                "config_id": fact["config_id"], "dtype": fact["dtype"],
                "N": fact["N"], "kernel_config": fact["kernel_config"],
                "stages": {}}
        entry = compile_by_id[fact["config_id"]]
        entry["stages"][fact["stage"]] = {
            key: fact[key] for key in (
                "registers_per_thread", "spilled_registers", "num_warps",
                "threads_per_cta", "num_stages", "shared_memory_bytes_per_cta",
                "resident_ctas_per_sm", "top_k", "mul_routed_weight",
                "representative_tokens", "target")}
    reference = {
        "schema_version": 1,
        "hardware_id": "nvidia_h200",
        "evidence_kind": (
            "installed_runtime_source_and_bundled_tuned_config_with_"
            "triton_compile_metadata"),
        "production_or_evaluation_duration_consumed": False,
        "hardware_execution_performed": True,
        "deployment": {
            "python_environment": 'vLLM 0.24 Python environment',
            "vllm_version": raw["vllm"],
            "torch_version": raw["torch"],
            "triton_version": raw["triton"],
            "device_name": raw["device_name"],
            "vllm_tuned_config_folder": None,
        },
        "source": {
            "path": raw["vllm_fused_moe_source"],
            "sha256": raw["vllm_fused_moe_source_sha256"],
            "resolution_precedence": (
                "get_config override, VLLM_TUNED_CONFIG_FOLDER candidate, "
                "bundled config candidate (nearest tuned batch key), "
                "get_default_config"),
            "device_name_rule": (
                "get_config_file_name maps any device whose name contains the "
                "token H200 to NVIDIA_H200"),
        },
        "bundled_bf16_config_domains": bf16,
        "bundled_h200_configs_all_dtypes": raw["bundled_h200_configs"],
        "forward_scope": {
            "status": "supported_for_listed_deepseek_v3_shapes",
            "supported": (
                "DeepSeek-V3 routed experts E=256 top-8 hidden 7168 with "
                "N=2048/TP (TP4 N=512, TP8 N=256, no EP), fp8_w8a8 block "
                "[128,128] via the bundled NVIDIA_H200 tuned config and BF16 "
                "via get_default_config (no bundled E=256 BF16 file), "
                "M = 1..32768 tokens"),
            "unsupported": (
                "other (E, N, dtype) domains are not enumerated here; any "
                "BF16 domain listed in bundled_bf16_config_domains selects a "
                "tuned file instead of get_default_config"),
            "also_unrepresented": (
                "explicit get_config override and any VLLM_TUNED_CONFIG_FOLDER "
                "candidate"),
        },
        "deepseek_v3_selection": raw["shapes"],
        "compile_metadata": sorted(
            compile_by_id.values(), key=lambda item: (
                item["dtype"], item["N"], item["config_id"])),
        "distinct_kernel_config_count": raw["distinct_kernel_config_count"],
        "provenance": {
            "runner": str(RUNNER.relative_to(ROOT)),
            "runner_sha256": _sha256(RUNNER),
            "derivation": str(Path(__file__).resolve().relative_to(ROOT)),
            "derivation_sha256": _sha256(Path(__file__).resolve()),
            "git_commit": commit,
            "raw_local": str(args.raw.resolve().relative_to(ROOT)),
            "raw_sha256": _sha256(args.raw),
            "raw_remote": args.remote_raw,
            "device": 'GPU 0 (CUDA_VISIBLE_DEVICES=0)',
            "timing_captured": False,
            "clock_stats": "not applicable: no timed capture",
        },
    }
    missing = a800_keys - set(reference)
    if missing:
        raise ValueError(f"reference lacks A800 keys {sorted(missing)}")
    OUTPUT.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)),
                      "distinct_configs": len(compile_by_id)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
