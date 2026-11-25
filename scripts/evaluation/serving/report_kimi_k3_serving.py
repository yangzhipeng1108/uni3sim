#!/usr/bin/env python3
"""Run ordinary Kimi-K3 text Serving or its native H200 capacity plan."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.kimi_k3_serving import main


if __name__ == "__main__":
    main()
