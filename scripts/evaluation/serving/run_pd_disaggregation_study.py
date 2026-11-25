#!/usr/bin/env python3
"""Run fair colocated-versus-PD ShareGPT simulation experiments."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset import (  # noqa: E402
    load_compact_case,
    load_pd_disaggregated_case,
)
from evaluation.serving.metrics import pd_disaggregation_study as study  # noqa: E402
from evaluation.serving.simulation import simulation_runner  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "repeat to select cells across existing ShareGPT suite configs; "
            "defaults to sharegpt_phase1.json"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="hardware/model/case_id; defaults to qwen3-4b QPS16 on all hardware",
    )
    parser.add_argument(
        "--topology",
        action="append",
        choices=tuple(study.TOPOLOGIES),
        default=[],
        help="repeat to evaluate multiple P:D layouts; defaults to 1p1d",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config_arguments = args.config or [
        "configs/evaluation/experiments/sharegpt_phase1.json"]
    requested = args.case or [
        "nvidia_a800/qwen3-4b/tp1_qps16",
        "metax_c500/qwen3-4b/tp1_qps16",
        "ascend_910b/qwen3-4b/tp1_qps16",
    ]
    selected_cells = study.select_study_cells(
        ROOT, config_arguments, requested)
    summary = study.run_study(
        ROOT,
        selected_cells,
        args.topology or ["1p1d"],
        args.run_id,
        load_compact_case,
        load_pd_disaggregated_case,
        simulation_runner.run_cli_args,
        jobs=args.jobs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
