#!/usr/bin/env python3
"""CLI for the cross-hardware kernel mechanism-validation report."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.mechanism_validation_report import main


if __name__ == "__main__":
    raise SystemExit(main())
