#!/usr/bin/env python3
"""Run the audited C500 Qwen3-4B QPS16 timing reduction."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics.metax_c500_qwen3_4b_tp1_qps16_reduction import main  # noqa: E402


if __name__ == "__main__":
    main(load_compact_case)
