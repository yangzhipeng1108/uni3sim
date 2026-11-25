#!/usr/bin/env python3
"""Analyze a same-run 910B selected-step msprof capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset.ascend_910b_selected_step_msprof import (
    analyze_ascend_910b_selected_step_dispatches,
    analyze_ascend_910b_selected_step_msprof,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scheduler-trace", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--selected-step-id", required=True)
    parser.add_argument("--expected-rank-count", type=int, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replay is None:
        if args.scheduler_trace is None:
            parser.error("--scheduler-trace is required without --replay")
        result = analyze_ascend_910b_selected_step_msprof(
            args.run_dir,
            args.scheduler_trace,
            args.validation,
            args.selected_step_id,
            args.expected_rank_count,
        )
    else:
        result = analyze_ascend_910b_selected_step_dispatches(
            args.run_dir,
            args.validation,
            args.replay,
            args.selected_step_id,
            args.expected_rank_count,
        )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
