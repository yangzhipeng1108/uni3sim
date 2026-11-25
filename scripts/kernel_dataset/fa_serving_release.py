#!/usr/bin/env python3
"""Publish one hardware's FA serving capture evidence dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_release import build_release_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--timing-root", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(build_release_dataset(
        capture_manifest=args.capture_manifest,
        timing_root=args.timing_root,
        candidate_root=args.candidate_root,
        hardware_id=args.hardware,
        output_path=args.output,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
