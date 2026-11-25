#!/usr/bin/env python3
"""CLI for validating or generating NVIDIA H200 launch-selection CSV."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.launch_data.nvidia_h200 import main


if __name__ == "__main__":
    raise SystemExit(main())
