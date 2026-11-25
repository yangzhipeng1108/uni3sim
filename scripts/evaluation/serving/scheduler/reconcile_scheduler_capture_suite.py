#!/usr/bin/env python3
"""Validate and summarize a complete Scheduler/Graph Serving capture suite."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.scheduler_capture_suite import (  # noqa: E402
    reconcile_scheduler_capture_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-hardware-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--canary-run", required=True, type=Path)
    parser.add_argument("--expected-case-count", required=True, type=int)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--execution-snapshot", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    report = reconcile_scheduler_capture_suite(
        canonical_hardware_root=args.canonical_hardware_root,
        run_root=args.run_root,
        candidate_root=args.candidate_root,
        canary_run=args.canary_run,
        expected_case_count=args.expected_case_count,
        baseline_commit=args.baseline_commit,
        execution_snapshot=args.execution_snapshot,
        container_name=args.container_name,
        container_image_id=args.container_image_id,
        container_image=args.container_image,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
