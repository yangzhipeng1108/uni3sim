#!/usr/bin/env python3
"""Create a complete Scheduler-depth metadata candidate for one hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.scheduler_depth_migration import (  # noqa: E402
    build_scheduler_depth_metadata_candidates,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--raw-result-name", required=True)
    parser.add_argument("--raw-mapping", type=Path)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--executor-class", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--converter-source-commit", required=True)
    parser.add_argument("--expected-case-count", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    result = build_scheduler_depth_metadata_candidates(
        canonical_root=args.canonical_root,
        raw_root=args.raw_root,
        raw_result_name=args.raw_result_name,
        raw_mapping_path=args.raw_mapping,
        hardware_id=args.hardware_id,
        executor_class=args.executor_class,
        source_identity=args.source_identity,
        candidate_root=args.candidate_root,
        converter_source_commit=args.converter_source_commit,
        expected_case_count=args.expected_case_count,
    )
    print(json.dumps({
        "status": "passed",
        "hardware_id": result["hardware_id"],
        "case_count": result["case_count"],
        "report_path": result["report_path"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
