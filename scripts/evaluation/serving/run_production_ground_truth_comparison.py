#!/usr/bin/env python3
"""Compare Serving simulation aggregates with production ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.production_ground_truth_comparison import (  # noqa: E402
    compare_production_ground_truth,
    write_production_ground_truth_comparison,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare analytical Serving simulation JSON with production "
            "ground_truth.json files"
        )
    )
    parser.add_argument(
        "--ground-truth",
        action="append",
        required=True,
        help="ground_truth.json file or recursively searched directory; repeatable",
    )
    parser.add_argument(
        "--simulation",
        action="append",
        required=True,
        help="simulation JSON file or recursively searched directory; repeatable",
    )
    parser.add_argument(
        "--scheduler-policy-attestation",
        help=(
            "structured per-case attestation for legacy ground_truth.json files "
            "that do not record server.engine.scheduler_policy"
        ),
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="new directory for JSON, CSV, and Markdown aggregate reports",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = compare_production_ground_truth(
        args.ground_truth,
        args.simulation,
        scheduler_policy_attestation=args.scheduler_policy_attestation,
    )
    paths = write_production_ground_truth_comparison(
        report, args.output_directory
    )
    print(f"Compared {report['cell_count']} cells")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
