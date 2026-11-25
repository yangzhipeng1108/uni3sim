#!/usr/bin/env python3
"""CLI for converting mcTracer observations to the Nsight schema."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.conversion.mctracer_to_nsys import main


if __name__ == "__main__":
    raise SystemExit(main())
