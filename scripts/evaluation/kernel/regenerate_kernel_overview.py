#!/usr/bin/env python3
"""Rebuild a Kernel HTML report from its saved sidecars, without simulation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.reporting.kernel_overview import regenerate_kernel_overview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original report .html path")
    parser.add_argument("--output", required=True, type=Path, help="New, separate .html path")
    args = parser.parse_args()
    print(json.dumps(regenerate_kernel_overview(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
