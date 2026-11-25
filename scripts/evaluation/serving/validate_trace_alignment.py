#!/usr/bin/env python3
"""CLI for validating serving-trace alignment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.traces.trace_alignment_validation import main


if __name__ == "__main__":
    raise SystemExit(main())
