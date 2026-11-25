#!/usr/bin/env python3
"""CLI for serving-metric evaluation and aggregation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.metric_evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
