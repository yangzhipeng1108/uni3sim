"""Prepare one H200 FA NCU retry after adding pre-NVTX clock settling."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.profiler.nvidia_h200_ncu_plan import (  # noqa: E402
    prepare_clock_settle_retry_plan,
)

CAMPAIGN = ROOT / "profiler_output/kernel_dataset/nvidia_h200/ncu_1027_20261031"


if __name__ == "__main__":
    print(prepare_clock_settle_retry_plan(CAMPAIGN))
