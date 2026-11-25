#!/usr/bin/env python3
"""Execute one materialized real 1P1D PD capture plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.pd_capture_executor import (  # noqa: E402
    execute_pd_capture_plan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one real 1P1D PD capture plan")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--readiness-timeout-s", type=float, default=900.0)
    parser.add_argument("--benchmark-timeout-s", type=float, default=7200.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    execute_pd_capture_plan(
        args.plan,
        readiness_timeout_s=args.readiness_timeout_s,
        benchmark_timeout_s=args.benchmark_timeout_s,
        poll_interval_s=args.poll_interval_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
