#!/usr/bin/env python3
"""CLI for validating or generating NVIDIA A800 launch-selection CSV."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.launch_data.nvidia_a800 import main


if __name__ == "__main__":
    raise SystemExit(main())
