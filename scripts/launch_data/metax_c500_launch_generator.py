#!/usr/bin/env python3
"""CLI for validating or generating MetaX C500 launch-selection CSV."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.launch_data.metax_c500 import main


if __name__ == "__main__":
    raise SystemExit(main())
