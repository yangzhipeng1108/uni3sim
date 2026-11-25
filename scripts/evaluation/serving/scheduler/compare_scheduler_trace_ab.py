#!/usr/bin/env python3
"""Compare trace-off and trace-on Serving ground-truth runs."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.diagnostics.scheduler_trace_observer_effect import (  # noqa: E402
    compare_scheduler_trace_observer_effect,
    require_observer_effect_passed,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-off", required=True, type=Path)
    parser.add_argument("--trace-on", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold-percent", type=float, default=3.0)
    parser.add_argument("--classification-only", action="store_true",
                        help="write every verdict without requiring a pass")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    comparison = compare_scheduler_trace_observer_effect(
        args.trace_off, args.trace_on, threshold_percent=args.threshold_percent
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    if not args.classification_only:
        require_observer_effect_passed(comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
