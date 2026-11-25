#!/usr/bin/env python3
"""Run the audited 910B matched-trajectory timing reduction."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.ascend_910b_same_trajectory_reduction import main  # noqa: E402


if __name__ == "__main__":
    main()
