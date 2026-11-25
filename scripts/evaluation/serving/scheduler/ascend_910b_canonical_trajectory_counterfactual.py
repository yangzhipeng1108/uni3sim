#!/usr/bin/env python3
"""Run the formal 910B canonical-trajectory counterfactual."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics.ascend_910b_canonical_trajectory_counterfactual import main  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--standard-results", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    main(
        load_compact_case,
        repository_root=arguments.repository_root,
        standard_results_path=arguments.standard_results,
    )
