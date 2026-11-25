"""Compatibility imports for H200 capture runners; orchestration belongs to profiler."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.profiler.nvidia_h200_serving_capture import (
    B, ModelProfile, OffloadArm, OffloadGroup, main,
)
