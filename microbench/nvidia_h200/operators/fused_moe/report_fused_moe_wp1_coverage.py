#!/usr/bin/env python3
"""Report which H200 DeepSeek-V3 fused-MoE launches match a published service.

Generality finding for the H200 port (WP1); it adds no service.  Inputs are
committed launch facts only (no duration):
  - ``microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json``
    (vLLM 0.24 config resolution for DeepSeek-V3 E256/top-8/H7168 at TP4 and
    TP8, fp8_w8a8 block [128,128] and BF16, M = 1..32768, with Triton 3.6
    compile metadata per config);
  - ``microbench/nvidia_h200/references/vllm_0_24_0_dsv3_l32_tp4_launch_geometry_sm90.json``
    (the ``fused_moe_kernel`` launch geometries observed in a real DeepSeek-V3
    32-layer TP4 decode run);
  - the fused-MoE ``memory_subsystem.memory_access_class_services`` rows of
    ``configs/hardware/nvidia_h200.yaml`` whose signature carries a Triton
    tile (read only).
A launch "matches" a service when dtype, BLOCK_SIZE_M/N/K and GROUP_SIZE_M
all equal the service signature (dtype, tile_m, tile_n, tile_k,
group_size_m); the per-service topology keys (active/maximum M tiles,
projection class) are a further restriction that is not evaluated here.
"""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
TUNED = ROOT / "microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json"
GEOMETRY = ROOT / "microbench/nvidia_h200/references/vllm_0_24_0_dsv3_l32_tp4_launch_geometry_sm90.json"
CONFIG = ROOT / "configs/hardware/nvidia_h200.yaml"
DTYPE = {"bf16": "bf16", "fp8_w8a8": "fp8"}


def _services():
    document = yaml.safe_load(CONFIG.read_text())
    rows = document["memory_subsystem"]["memory_access_class_services"]
    services = []
    for row in rows:
        signature = row["signature"]
        if "tile_m" not in signature or "group_size_m" not in signature:
            continue
        services.append({
            "service": row["service"], "dtype": signature["dtype"],
            "tile": (signature["tile_m"], signature["tile_n"],
                     signature["tile_k"], signature["group_size_m"])})
    return services


def _tile(config):
    return (config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
            config["BLOCK_SIZE_K"], config["GROUP_SIZE_M"])


def main():
    services = _services()
    signatures = sorted({(item["dtype"], item["tile"]) for item in services})
    tuned = json.loads(TUNED.read_text())
    metadata = {item["config_id"]: item for item in tuned["compile_metadata"]}
    print("Published fused-MoE service signatures (dtype, BM/BN/BK/GSM): "
          + "; ".join(f"{dtype} {tile[0]}/{tile[1]}/{tile[2]}/{tile[3]} "
                      f"({sum(1 for s in services if (s['dtype'], s['tile']) == (dtype, tile))} services)"
                      for dtype, tile in signatures))
    print()
    print("| deployment | dtype | M range | kernel config BM/BN/BK/GSM, warps, stages | "
          "regs w1/w2 | smem B/CTA | CTAs/SM | matching services |")
    print("|---|---|---|---|---|---|---|---|")
    compiled_configs = []
    for selection in tuned["deepseek_v3_selection"]:
        deployment = (f"TP{selection['tensor_parallel_size']} "
                      f"N={selection['intermediate_per_rank_N']}")
        dtype = DTYPE[selection["dtype"]]
        for item in selection["selection_ranges"]:
            config = item["kernel_config"]
            compiled = metadata[item["config_id"]]["stages"]
            tile = _tile(config)
            matches = [s["service"] for s in services
                       if (s["dtype"], s["tile"]) == (dtype, tile)]
            key = (compiled["w1"]["registers_per_thread"],
                   compiled["w2"]["registers_per_thread"])
            compiled_configs.append((key, dtype, selection["dtype"], tile))
            print(f"| {deployment} | {selection['dtype']} | "
                  f"{item['m_min']}-{item['m_max']} | "
                  f"{'/'.join(map(str, tile))}, w{config['num_warps']}, "
                  f"s{config['num_stages']} | "
                  f"{key[0]}/{key[1]} | "
                  f"{compiled['w1']['shared_memory_bytes_per_cta']} | "
                  f"{compiled['w1']['resident_ctas_per_sm']} | "
                  f"{', '.join(matches) if matches else 'none'} |")
    print()
    geometry = json.loads(GEOMETRY.read_text())
    kernels = [kernel for kernel in geometry["kernels"]
               if kernel["kernel_name"] == "fused_moe_kernel"]
    print("| observed TP4 decode fused_moe_kernel launch (grid CTAs, regs, "
          "dynamic smem MB) | launches | compiled configs with these registers "
          "(TP4/TP8 tuned table) | matching services |")
    print("|---|---|---|---|")
    for kernel in kernels:
        for item in kernel["geometries"]:
            registers = item["registers_per_thread"]
            candidates = sorted({
                (dtype, label, tile) for key, dtype, label, tile
                in compiled_configs if registers in key})
            matches = sorted({
                service["service"] for service in services
                for dtype, _, tile in candidates
                if (service["dtype"], service["tile"]) == (dtype, tile)})
            launches = sum(item["launch_count_by_window"].values())
            names = "; ".join(f"{label} {'/'.join(map(str, tile))}"
                              for _, label, tile in candidates)
            print(f"| {item['ctas']}, {registers}, "
                  f"{item['dynamic_shared_memory_mb_nsys']} | {launches} | "
                  f"{names if names else 'none'} | "
                  f"{', '.join(matches) if matches else 'none'} |")


if __name__ == "__main__":
    main()
