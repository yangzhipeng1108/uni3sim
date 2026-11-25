#!/usr/bin/env python3
"""CLI for building compact ShareGPT Serving candidates."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.sharegpt_candidate_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
