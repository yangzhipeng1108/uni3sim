#!/usr/bin/env python3
"""CLI for the OpenAI-compatible evaluation request router."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.request_router import main


if __name__ == "__main__":
    raise SystemExit(main())
