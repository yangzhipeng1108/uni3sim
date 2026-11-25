#!/usr/bin/env python3
"""Compare exact Dataset and Serving predictor inputs for three hardware."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.reporting.dataset_serving_input_comparison import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main(load_compact_case)
