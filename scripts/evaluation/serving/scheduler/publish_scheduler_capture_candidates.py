#!/usr/bin/env python3
"""Publish fully validated Scheduler capture candidates to one dataset root."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.scheduler_capture_publisher import (  # noqa: E402
    publish_scheduler_capture_candidates,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument(
        "--dataset-root", type=Path, default=ROOT / "serving_dataset/sharegpt"
    )
    parser.add_argument(
        "--case", action="append", required=True,
        help="exact hardware_id/model_id/case_id; repeat for every case",
    )
    parser.add_argument(
        "--new-case-reference-hardware",
        help=("publish cases with no canonical predecessor, requiring this "
              "hardware's canonical case to hold the same requests"),
    )
    args = parser.parse_args()
    for path in publish_scheduler_capture_candidates(
        candidate_root=args.candidate_root,
        dataset_root=args.dataset_root,
        cases=args.case,
        new_case_reference_hardware=args.new_case_reference_hardware,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
