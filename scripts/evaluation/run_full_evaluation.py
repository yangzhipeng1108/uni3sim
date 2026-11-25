#!/usr/bin/env python3
"""CLI for the cross-hardware full evaluation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.reporting.full_evaluation_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
