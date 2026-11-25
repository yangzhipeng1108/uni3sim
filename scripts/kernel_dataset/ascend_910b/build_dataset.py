#!/usr/bin/env python3
"""CLI for building the Ascend 910B kernel dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.ascend_910b import main


if __name__ == "__main__":
    raise SystemExit(main())
