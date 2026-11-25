#!/usr/bin/env python3
"""CLI for validating or generating Ascend 910B launch data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.launch_data.ascend_910b import main


if __name__ == "__main__":
    raise SystemExit(main())
