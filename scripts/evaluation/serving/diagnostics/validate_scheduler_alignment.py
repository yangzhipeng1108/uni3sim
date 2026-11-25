#!/usr/bin/env python3
"""Run aggregate-only Scheduler alignment for an explicit suite subset."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.traces.scheduler_alignment import (  # noqa: E402
    run_scheduler_alignment_cli,
)


def run_cli(argv: list[str] | None = None) -> dict[str, object]:
    return run_scheduler_alignment_cli(
        argv,
        repository_root=REPOSITORY_ROOT,
        load_validated_case=load_compact_case,
    )


def main(argv: list[str] | None = None) -> int:
    payload = run_cli(argv)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if payload["status_counts"]["runtime_mismatch"]:
        return 1
    policy = payload["policy_diagnostic_status_counts"]
    if "mismatched" in policy and policy["mismatched"]:
        return 1
    if payload["status_counts"]["inconclusive_clock_alignment"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
