#!/usr/bin/env python3
"""CLI for comparing a simulated trace with Nsight Systems."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.traces.nsys_trace_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
