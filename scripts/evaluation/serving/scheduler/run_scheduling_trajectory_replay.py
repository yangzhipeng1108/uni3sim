#!/usr/bin/env python3
"""Run Scheduling Trajectory Replay for one canonical observed Scheduler trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import (  # noqa: E402
    load_compact_case,
)
from evaluation.serving.traces import scheduling_trajectory_replay  # noqa: E402
from src.serving.host_timing import load_host_launch_profile  # noqa: E402


DEFAULT_CONFIG = Path("configs/evaluation/experiments/sharegpt_phase1.json")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help=(
        "exact cell ID, for example ascend_910b/qwen3-4b/tp1_qps16"))
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scheduler-trace", type=Path, help=(
        "instrumented canonical JSONL; defaults to the selected canonical case"))
    parser.add_argument("--scheduler-schema-version", type=int, help=(
        "exact trace schema; defaults to the schema_version stored in the trace"))
    parser.add_argument("--host-launch-profile", type=Path)
    parser.add_argument("--execution-mode",
                        choices=("accelerated", "reference"),
                        default="accelerated")
    parser.add_argument("--counterfactual-pipeline-depth", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; stdout is used when omitted",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    root = arguments.repository_root.resolve()
    profile = (
        None
        if arguments.host_launch_profile is None
        else load_host_launch_profile(arguments.host_launch_profile.resolve())
    )
    result = scheduling_trajectory_replay.run_scheduling_trajectory_replay_case(
        root,
        arguments.config,
        arguments.case,
        load_validated_case=load_compact_case,
        scheduler_trace=arguments.scheduler_trace,
        scheduler_schema_version=arguments.scheduler_schema_version,
        host_launch_profile=profile,
        execution_mode=arguments.execution_mode,
        counterfactual_pipeline_depth=arguments.counterfactual_pipeline_depth,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(rendered, end="")
        return
    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(json.dumps({"output": str(output_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
