#!/usr/bin/env python3
"""Merge one published FA serving campaign into the canonical FA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_release import (
    merge_release_into_canonical,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--release", required=True, type=Path)
    args = parser.parse_args()
    print(merge_release_into_canonical(
        canonical_path=args.canonical,
        release_path=args.release,
        hardware_id=args.hardware,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
