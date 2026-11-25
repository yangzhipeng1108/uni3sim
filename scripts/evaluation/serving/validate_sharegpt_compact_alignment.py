#!/usr/bin/env python3
"""Validate all compact ShareGPT cases and run structural alignment."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics.compact_alignment_suite import (  # noqa: E402
    run_compact_alignment_suite,
)
from evaluation.serving.metrics.sharegpt_compact_evaluation import (  # noqa: E402
    NVIDIA_H200_SUITE_SELECTIONS, repository_source_commit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvidia-h200-vllm-version", choices=tuple(NVIDIA_H200_SUITE_SELECTIONS),
        help="include the existing 15 H200 cells of this captured version")
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="case-level worker processes (default: available CPUs)",
    )
    args = parser.parse_args()
    result = run_compact_alignment_suite(
        REPOSITORY_ROOT,
        load_compact_case,
        source_commit=repository_source_commit(REPOSITORY_ROOT),
        jobs=args.jobs,
        nvidia_h200_vllm_version=args.nvidia_h200_vllm_version,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if (
        result["alignment_checked"] is True
        and result["formal_conclusion_allowed"] is True
        and result["mismatched_count"] == 0
        and result["unsupported_count"] == 0
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
