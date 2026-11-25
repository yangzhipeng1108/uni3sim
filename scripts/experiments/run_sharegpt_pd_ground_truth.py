#!/usr/bin/env python3
"""CLI for one frozen ShareGPT 1P1D Serving ground-truth benchmark."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.pd_sharegpt_ground_truth import main


if __name__ == "__main__":
    raise SystemExit(main())
