#!/usr/bin/env python3
"""Audit the complete Scheduler-depth metadata-only migration candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.scheduler_depth_migration import (  # noqa: E402
    audit_scheduler_depth_metadata_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--migration-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_scheduler_depth_metadata_candidates(
        canonical_root=args.canonical_root,
        candidate_root=args.candidate_root,
        migration_reports=args.migration_report,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite migration audit: {output}")
    output.write_text(json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    print(json.dumps({
        "status": "passed",
        "case_count": result["case_count"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
