#!/usr/bin/env python3
"""CLI for manifest-driven serving experiments."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.experiment_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
