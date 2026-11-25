#!/usr/bin/env python3
"""Compare official GLM-5.2-FP8/5.3 source graphs, capacity terms and math."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.model_operator_probe import write_glm_official_fp8_family_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 8), required=True)
    parser.add_argument("--mtp-mode", choices=("off", "native"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_glm_official_fp8_family_comparison(
        args.output, tensor_parallel_size=args.tensor_parallel_size, mtp_mode=args.mtp_mode)
    print(f"{args.output}: failed={result['failed']}; source-only, normal/runtime=false")
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
