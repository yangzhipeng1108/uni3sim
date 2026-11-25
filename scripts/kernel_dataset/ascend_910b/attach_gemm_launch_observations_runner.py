#!/usr/bin/env python3
"""Attach duration-free CANN launch observations to a GEMM Dataset."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.ascend_910b_gemm_launch import main


if __name__ == "__main__":
    raise SystemExit(main())
