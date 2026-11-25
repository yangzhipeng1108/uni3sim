#!/usr/bin/env python3
"""Freeze the current 15-case A800 ShareGPT source contract for PD capture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.pd_capture_plan import (  # noqa: E402
    freeze_a800_pd_source_suite,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze A800 ShareGPT PD source suite")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-suite", required=True, action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    suite = freeze_a800_pd_source_suite(
        dataset_root=args.dataset_root,
        source_suite_paths=args.source_suite,
        output_path=args.output,
    )
    print(len(suite["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
