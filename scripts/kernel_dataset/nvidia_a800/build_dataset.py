#!/usr/bin/env python3
"""CLI for building the NVIDIA A800 kernel dataset."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_a800 import main


if __name__ == "__main__":
    raise SystemExit(main())
