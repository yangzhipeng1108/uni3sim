"""Prepare pinned H200 NCU case batches for A800's kernel case IDs.

This only derives launch-matched capture plans. It never runs a profiler or
changes canonical data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dataset_pipeline.kernel_dataset.nvidia_h200 import (
    H200_FAMILIES, load_campaign_plan)
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX, load_kernel_dataset_document)


TRAFFIC_METRICS = [
    "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
    "dram__bytes_read.sum", "dram__bytes_write.sum",
]
SERVICE_METRICS = [
    "smsp__warps_eligible.avg.per_cycle_active",
    "smsp__warps_active.avg.per_cycle_active",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_active.avg", "sm__cycles_elapsed.avg",
]
BATCH_SIZES = {"gemm": 10, "fused_moe": 4, "flash_attn": 6, "vector": 12}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dimensions(text: str) -> list[int]:
    result = [int(value.strip()) for value in text.strip("()").split(",")]
    if len(result) != 3 or any(value <= 0 for value in result):
        raise ValueError(f"invalid H200 launch dimensions: {text}")
    return result


def _source_plans(directories: list[Path]) -> dict[str, dict]:
    indexed = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            document = load_campaign_plan(path)
            for case in document["plans"]:
                case_id = case["case_id"]
                if case_id in indexed:
                    if indexed[case_id]["plan"] != case:
                        raise ValueError(f"conflicting H200 runner plan: {case_id}")
                    continue
                indexed[case_id] = {
                    "plan": case, "source_plan_path": str(path),
                    "source_plan_sha256": _sha(path),
                }
    return indexed


def prepare(*, a800_dir: Path, h200_dir: Path,
            plan_directories: list[Path], output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite NCU plan: {output_dir}")
    sources = _source_plans(plan_directories)
    by_family = {}
    all_ids = []
    existing_sidecars = []
    missing_nsys = {}
    canonical_sha = {}
    for family in H200_FAMILIES:
        a800 = load_kernel_dataset_document(
            a800_dir / f"{family}{KERNEL_DATASET_SUFFIX}")
        h200_path = h200_dir / f"{family}{KERNEL_DATASET_SUFFIX}"
        h200 = load_kernel_dataset_document(h200_path)
        canonical_sha[family] = _sha(h200_path)
        missing_nsys[family] = sorted(set(a800) - set(h200))
        cases = []
        for case_id in sorted(set(a800) & set(h200)):
            all_ids.append(case_id)
            entry = h200[case_id]
            if all("NCU Counters" in kernel["Sections"]
                   for kernel in entry["kernels"]):
                existing_sidecars.append(case_id)
                continue
            if any("NCU Counters" in kernel["Sections"]
                   for kernel in entry["kernels"]):
                raise ValueError(f"partial existing NCU sidecar: {case_id}")
            if case_id not in sources:
                raise ValueError(f"missing H200 source runner plan: {case_id}")
            source = sources[case_id]
            plan = source["plan"]
            if plan["family"] != family:
                raise ValueError(f"H200 plan family differs: {case_id}")
            signature = [[kernel["Metadata"]["Kernel Name"],
                          _dimensions(kernel["Metadata"]["Grid Size"]),
                          _dimensions(kernel["Metadata"]["Block Size"])]
                         for kernel in entry["kernels"]]
            cases.append({
                "family": family, "case_id": case_id, "plan": plan,
                "canonical_launch_signature": signature,
                "canonical_family_sha256": canonical_sha[family],
                "source_plan_path": source["source_plan_path"],
                "source_plan_sha256": source["source_plan_sha256"],
            })
        by_family[family] = cases
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A800/H200 case IDs repeat across families")

    pilot = []
    for family in H200_FAMILIES:
        cases = by_family[family]
        if len(cases) < 3:
            raise ValueError(f"too few {family} cases for an NCU pilot")
        for index in sorted({len(cases) // 3, 2 * len(cases) // 3}, reverse=True):
            pilot.append(cases.pop(index))
    batches = {"ncu_pilot_01": pilot}
    for family in H200_FAMILIES:
        cases = by_family[family]
        size = BATCH_SIZES[family]
        for offset in range(0, len(cases), size):
            name = f"ncu_{family}_{offset // size + 1:03d}"
            batches[name] = cases[offset:offset + size]
    manifest = {
        "schema_version": 1,
        "purpose": "H200 NCU non-duration counters for A800 kernel case IDs",
        "status": "prepared_not_executed",
        "hardware_id": "nvidia_h200",
        "software_gate": {"vllm": "0.30.0+cu129", "torch": "2.13.0+cu129",
                          "torch_cuda": "12.9"},
        "gpu": "H200", "gpu_count": 1, "budget_usd": 20,
        "metrics": {"traffic_metrics": TRAFFIC_METRICS,
                    "service_metrics": SERVICE_METRICS},
        "source_plan_directories": [str(path) for path in plan_directories],
        "canonical_family_sha256": canonical_sha,
        "a800_case_count": sum(len(load_kernel_dataset_document(
            a800_dir / f"{family}{KERNEL_DATASET_SUFFIX}"))
            for family in H200_FAMILIES),
        "matched_case_count": len(all_ids),
        "existing_sidecar_case_ids": sorted(existing_sidecars),
        "missing_nsys_case_ids": missing_nsys,
        "pending_case_count": sum(map(len, batches.values())),
        "pending_kernel_count": sum(
            len(case["canonical_launch_signature"])
            for batch in batches.values() for case in batch),
        "batch_order": list(batches),
        "batches": {name: [case["case_id"] for case in cases]
                    for name, cases in batches.items()},
    }
    output_dir.mkdir(parents=True)
    for name, cases in batches.items():
        (output_dir / f"{name}.json").write_text(json.dumps({
            "schema_version": 1, "batch_name": name,
            "traffic_metrics": TRAFFIC_METRICS,
            "service_metrics": SERVICE_METRICS, "cases": cases,
        }, indent=2, allow_nan=False) + "\n")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a800-dir", type=Path, required=True)
    parser.add_argument("--h200-dir", type=Path, required=True)
    parser.add_argument("--plan-directory", type=Path, action="append",
                        required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(a800_dir=args.a800_dir, h200_dir=args.h200_dir,
                     plan_directories=args.plan_directory,
                     output_dir=args.output_dir)
    print(json.dumps({name: result[name] for name in (
        "a800_case_count", "matched_case_count", "pending_case_count",
        "pending_kernel_count", "missing_nsys_case_ids")}, indent=2))


if __name__ == "__main__":
    main()
