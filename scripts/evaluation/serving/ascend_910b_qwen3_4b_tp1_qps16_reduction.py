#!/usr/bin/env python3
"""Run the audited 910B Qwen3-4B QPS16 timing reduction."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.ascend_910b_qwen3_4b_tp1_qps16_reduction import main  # noqa: E402


if __name__ == "__main__":
    main()
