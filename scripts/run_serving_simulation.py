#!/usr/bin/env python3
"""CLI for the production no-GPU serving simulation."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.simulation_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
