#!/usr/bin/env python3
"""Execute one virtual Serving case and retain its identity-checked comparison."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dataset_pipeline.serving_dataset import load_compact_case
from evaluation.serving.metrics.symbolic_model import run_paired_virtual_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-path", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("simulation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.simulation_args or args.simulation_args[0] != "--":
        parser.error("simulation arguments must follow --")
    run_paired_virtual_case(args.simulation_args[1:], load_compact_case(args.case_path),
                            args.comparison_output)


if __name__ == "__main__":
    main()
