#!/usr/bin/env python3
"""Run the 45-case native-vs-depth-one Scheduler capacity study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.diagnostics.scheduler_pipeline_depth_study import (  # noqa: E402
    materialize_scheduler_pipeline_depth_report,
    run_scheduler_pipeline_depth_study,
)
from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=Path("profiler_output/serving_dataset_compaction"),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
    )
    parser.add_argument(
        "--from-results",
        type=Path,
        help="materialize a paired report without rerunning simulation",
    )
    args = parser.parse_args()
    candidate_root = (
        args.candidate_root
        if args.candidate_root.is_absolute()
        else ROOT / args.candidate_root
    )
    if args.from_results is None:
        result = run_scheduler_pipeline_depth_study(
            ROOT,
            candidate_root,
            args.run_id,
            jobs=args.jobs,
            load_validated_case=load_compact_case,
        )
    else:
        source_results = (
            args.from_results
            if args.from_results.is_absolute()
            else ROOT / args.from_results
        )
        result = materialize_scheduler_pipeline_depth_report(
            ROOT, source_results, args.run_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
