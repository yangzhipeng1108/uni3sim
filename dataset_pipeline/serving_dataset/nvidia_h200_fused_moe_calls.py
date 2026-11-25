#!/usr/bin/env python3
"""Extract H200 Triton fused-MoE launch geometry and per-call durations.

Evaluation only.  Reads one or more nsys ``cuda_gpu_trace`` CSV exports of
the vLLM 0.24 DSV3-L32 TP4 server, keeps the ``fused_moe_kernel`` rows of one
device, classifies each launch as W1 or W2 from its source-order pair (the
source launches W1 then W2 per MoE layer with the same M tiling, so the W2
grid is the W1 grid times ceil(hidden/BN)/ceil(2*intermediate/BN)), and
reports, per distinct (variant, grid, block, registers, shared memory)
geometry, the launch count and the duration median/p10/p90.  Durations are
the evaluation truth for WP_REPORT tables; nothing here feeds a model
parameter.  Standard library only (runs with the remote system python3).

Usage:
    python3 extract_nvidia_h200_fused_moe_trace_calls.py --device 0 \
        --w2-to-w1-grid-ratio 7 --window C=1::<csv> [--window ...] \
        --output <json>
"""
import argparse
import csv
import hashlib
import json
import statistics


KERNEL_NAME = "fused_moe_kernel"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _rows(path, device_label):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["Name"] != KERNEL_NAME or row["Device"] != device_label:
                continue
            yield {
                "start_ns": int(row["Start (ns)"]),
                "duration_ns": int(row["Duration (ns)"]),
                "grid": [int(row["GrdX"]), int(row["GrdY"]), int(row["GrdZ"])],
                "block": [int(row["BlkX"]), int(row["BlkY"]), int(row["BlkZ"])],
                "registers_per_thread": int(row["Reg/Trd"]),
                "static_shared_memory_mb": row["StcSMem (MB)"],
                "dynamic_shared_memory_mb": row["DymSMem (MB)"],
            }


def _classify(launches, ratio):
    launches.sort(key=lambda row: row["start_ns"])
    classified = []
    index = 0
    unpaired = 0
    while index < len(launches):
        first = launches[index]
        if (index + 1 < len(launches)
                and launches[index + 1]["grid"][0] == ratio * first["grid"][0]):
            classified.append(("w1", first))
            classified.append(("w2", launches[index + 1]))
            index += 2
        else:
            unpaired += 1
            index += 1
    return classified, unpaired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--w2-to-w1-grid-ratio", type=int, required=True)
    parser.add_argument("--window", action="append", required=True,
                        help="KEY::PATH, e.g. C=1::/path/window_1_cuda_gpu_trace.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device_label = f"NVIDIA H200 ({args.device})"
    report = {
        "artifact": "h200_fused_moe_trace_calls_v1",
        "kernel_name": KERNEL_NAME,
        "device": device_label,
        "w2_to_w1_grid_ratio": args.w2_to_w1_grid_ratio,
        "evaluation_only": True,
        "production_or_evaluation_duration_consumed_by_model": False,
        "windows": {},
    }
    for spec in args.window:
        key, path = spec.split("::", 1)
        launches = list(_rows(path, device_label))
        classified, unpaired = _classify(launches, args.w2_to_w1_grid_ratio)
        groups = {}
        for variant, row in classified:
            geometry = (
                variant, tuple(row["grid"]), tuple(row["block"]),
                row["registers_per_thread"], row["static_shared_memory_mb"],
                row["dynamic_shared_memory_mb"])
            if geometry not in groups:
                groups[geometry] = []
            groups[geometry].append(row["duration_ns"])
        report["windows"][key] = {
            "source_csv": path,
            "source_csv_sha256": _sha256(path),
            "fused_moe_launches": len(launches),
            "unpaired_launches": unpaired,
            "geometries": [
                {
                    "variant": variant,
                    "grid": list(grid),
                    "block": list(block),
                    "registers_per_thread": registers,
                    "static_shared_memory_mb_nsys": static_mb,
                    "dynamic_shared_memory_mb_nsys": dynamic_mb,
                    "launch_count": len(durations),
                    "duration_ns_median": statistics.median(durations),
                    "duration_ns_p10": _quantile(durations, 0.10),
                    "duration_ns_p90": _quantile(durations, 0.90),
                    "duration_ns_min": min(durations),
                    "duration_ns_max": max(durations),
                }
                for (variant, grid, block, registers, static_mb, dynamic_mb),
                durations in sorted(groups.items())
            ],
        }
    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
