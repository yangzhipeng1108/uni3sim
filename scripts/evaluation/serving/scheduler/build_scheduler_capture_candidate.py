#!/usr/bin/env python3
"""Build one schema-v3 Serving case from an accepted Scheduler capture."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.scheduler_capture_candidate import (  # noqa: E402
    build_scheduler_capture_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-case", required=True, type=Path)
    parser.add_argument("--trace-on-result", required=True, type=Path)
    parser.add_argument("--scheduler-trace", required=True, type=Path)
    parser.add_argument("--observer-effect", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--converter-source-commit", required=True)
    parser.add_argument(
        "--observer-validation-status",
        choices=(
            "measured_for_this_case",
            "inherited_from_representative_case",
        ),
        default="measured_for_this_case",
    )
    args = parser.parse_args()
    output = build_scheduler_capture_candidate(
        canonical_case_dir=args.canonical_case,
        trace_on_result_path=args.trace_on_result,
        scheduler_trace_path=args.scheduler_trace,
        observer_effect_path=args.observer_effect,
        candidate_root=args.candidate_root,
        converter_source_commit=args.converter_source_commit,
        observer_validation_status=args.observer_validation_status,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
