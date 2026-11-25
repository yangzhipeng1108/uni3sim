#!/usr/bin/env python3
"""Write the observed-versus-injected GEMM launch comparison report."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.gemm_launch_comparison import (  # noqa: E402
    compare_all_gemm_launch_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare_all_gemm_launch_sources()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for hardware_id, result in report.items():
        summary = result["summary"]
        physical = summary["physical_launch"]
        execution_plan = summary["execution_plan"]
        print(
            f"{hardware_id}: GEMM={summary['gemm_main_kernels']} "
            f"plan_covered={summary['injected_execution_plan_covered']} "
            f"physical_delta={physical['prediction_max_abs_delta_percent']:.6f}% "
            f"plan_delta_mean={execution_plan['prediction_mean_abs_delta_percent']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
