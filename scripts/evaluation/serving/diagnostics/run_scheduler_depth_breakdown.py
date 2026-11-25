#!/usr/bin/env python3
"""Run the exact Scheduler-depth TTFT/TPOT signed error breakdown."""
import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.diagnostics.scheduler_depth_breakdown import (  # noqa: E402
    run_full_only_serving_evaluation,
    run_scheduler_depth_final_serving_quantiles,
    run_scheduler_depth_breakdown,
)
from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("serving_dataset/sharegpt"),
        help="validated canonical ShareGPT dataset root",
    )
    parser.add_argument(
        "--full-results",
        type=Path,
        default=Path("experiment_logs/serving/scheduler_depth/"
                     "scheduler_depth_20260902_v6/results.json"),
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--reuse-results",
        action="append",
        type=Path,
        default=[],
        help="completed exact-cell results to reuse in --full-serving-report-only",
    )
    parser.add_argument(
        "--nvidia-h200-vllm-version", choices=("0.30.0", "0.11.0"),
        help="--full-serving-report-only: add the 15 H200 cells of this vLLM version")
    parser.add_argument(
        "--nvidia-h200-dataset-root", type=Path,
        help=("non-canonical H200 candidate root of --nvidia-h200-vllm-version "
              "(default: --dataset-root)"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--final-serving-quantiles-only", action="store_true")
    parser.add_argument("--full-serving-report-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    return parser.parse_args()

def main() -> None:
    arguments = _arguments()
    if (arguments.final_serving_quantiles_only
            and arguments.full_serving_report_only):
        raise ValueError("--final-serving-quantiles-only and "
                         "--full-serving-report-only are mutually exclusive")
    if (not arguments.full_serving_report_only
            and arguments.nvidia_h200_vllm_version is not None):
        raise ValueError("H200 cells are available only with --full-serving-report-only")
    if arguments.full_serving_report_only:
        result = run_full_only_serving_evaluation(
            REPOSITORY_ROOT,
            arguments.dataset_root,
            arguments.run_id,
            jobs=arguments.jobs,
            selected_cell_ids=arguments.case,
            reuse_results_paths=arguments.reuse_results,
            load_validated_case=load_compact_case,
            nvidia_h200_vllm_version=arguments.nvidia_h200_vllm_version,
            nvidia_h200_dataset_root=arguments.nvidia_h200_dataset_root,
        )
    elif arguments.final_serving_quantiles_only:
        result = run_scheduler_depth_final_serving_quantiles(
            REPOSITORY_ROOT,
            arguments.dataset_root,
            arguments.full_results,
            arguments.run_id,
            jobs=arguments.jobs,
            selected_cell_ids=arguments.case,
            load_validated_case=load_compact_case,
        )
    else:
        result = run_scheduler_depth_breakdown(
            REPOSITORY_ROOT,
            arguments.dataset_root,
            arguments.full_results,
            arguments.run_id,
            jobs=arguments.jobs,
            selected_cell_ids=arguments.case,
            resume=arguments.resume,
            load_validated_case=load_compact_case,
        )
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
