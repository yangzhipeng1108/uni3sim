"""Prepare raw-only H200 NCU plans for the 15 missing NSYS shapes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.profiler.nvidia_h200_ncu_plan import (
    prepare_missing15_batch_plans,
)


CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"


if __name__ == "__main__":
    for path in prepare_missing15_batch_plans(CAMPAIGN):
        print(path)
