#!/usr/bin/env python3
"""Summarize H200 kernel-dataset manifests for WP_REPORT.md (read-only)."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.kernel.dataset_storage import load_kernel_dataset_document  # noqa: E402


def main():
    output_dir = ROOT / "profiler_output/kernel_dataset/nvidia_h200"
    rows = []
    for family in ("fused_moe", "gemm", "vector", "flash_attn"):
        manifest = json.loads(
            (output_dir / f"{family}.manifest.json").read_text(encoding="utf-8"))
        a800 = load_kernel_dataset_document(
            ROOT / f"kernel_dataset/nvidia_a800/{family}.json.zst")
        h200 = load_kernel_dataset_document(output_dir / f"{family}.json.zst")
        ratios = sorted(
            h200[key]["op_info"]["evaluation"]["timing"]["total_duration_ns"]
            / a800[key]["op_info"]["evaluation"]["timing"]["total_duration_ns"]
            for key in h200)
        retried = sum(1 for case in manifest["cases"]
                      if case["status"] == "pass" and case["window_count"] > 3)
        excluded = [
            {"case_id": case["case_id"], "status": case["status"],
             "window_count": case["window_count"],
             "maximum_relative_difference": case.get(
                 "maximum_relative_difference"),
             "clock_rejected_windows": len(case["clock_rejected_windows"])}
            for case in manifest["cases"] if case["status"] != "pass"]
        rows.append({
            "family": family,
            "a800_cases": manifest["a800_case_count"],
            "h200_cases": manifest["h200_case_count"],
            "retried_passing_cases": retried,
            "clock": manifest["clock_stats"],
            "max_spread_retained": max(
                case["maximum_relative_difference"]
                for case in manifest["cases"] if case["status"] == "pass"),
            "h200_over_a800_ratio_p5_p50_p95": [
                round(ratios[int(q * (len(ratios) - 1))], 3)
                for q in (0.05, 0.5, 0.95)],
            "excluded": excluded,
        })
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
