#!/usr/bin/env python3
"""CLI for converting msprof operator summaries."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.conversion.msprof_summary import main


if __name__ == "__main__":
    raise SystemExit(main())
