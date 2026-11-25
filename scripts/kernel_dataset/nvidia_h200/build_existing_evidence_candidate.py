"""Build a review-only H200 kernel dataset from already pinned evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_pipeline.kernel_dataset.nvidia_h200_existing_evidence_candidate import (
    build_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--helper-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_candidate(
        root=args.root.resolve(),
        canonical_dir=args.canonical_dir.resolve(),
        helper_path=args.helper_path.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "case_counts": result["case_counts"],
        "ncu_target_available_count": result["ncu_target_available_count"],
        "ncu_attached_case_count": result["ncu_attached_case_count"],
        "ncu_held_case_count": result["ncu_held_case_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
