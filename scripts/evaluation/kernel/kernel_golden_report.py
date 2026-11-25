#!/usr/bin/env python3
"""CLI for deterministic kernel-golden sampling and Event comparison."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.kernel_golden import main


if __name__ == "__main__":
    raise SystemExit(main())
