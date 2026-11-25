"""Publish the validated H200 missing-case and NCU candidate from local raw."""

import argparse
import json
from pathlib import Path

from dataset_pipeline.kernel_dataset.nvidia_h200_missing_publication import (
    publish_missing_h200_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--publication-dir", type=Path, required=True)
    args = parser.parse_args()
    result = publish_missing_h200_evidence(
        campaign_root=args.campaign_root,
        canonical_dir=args.canonical_dir,
        publication_dir=args.publication_dir)
    print(json.dumps({
        "accepted_counts": {family: len(cases) for family, cases in
                            result["accepted_added_cases"].items()},
        "unaccepted_counts": {family: len(cases) for family, cases in
                              result["unaccepted_case_ids"].items()},
        "ncu_case_count": len(result["ncu_sources"]),
    }, indent=2))


if __name__ == "__main__":
    main()
