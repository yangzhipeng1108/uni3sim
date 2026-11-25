#!/usr/bin/env python3
"""Write final candidate-only FA serving publication-audit artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture_report import (
    _parse_hardware_paths,
    build_publication_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", required=True, type=Path)
    parser.add_argument("--timing-root", required=True, action="append")
    parser.add_argument("--candidate-root", required=True, action="append")
    parser.add_argument("--canonical-dataset", required=True, action="append")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    outputs = build_publication_audit(
        capture_manifest=args.capture_manifest,
        timing_roots=_parse_hardware_paths(args.timing_root, "--timing-root"),
        candidate_roots=_parse_hardware_paths(args.candidate_root, "--candidate-root"),
        canonical_paths=_parse_hardware_paths(
            args.canonical_dataset, "--canonical-dataset"),
        output_root=args.output_root.resolve(),
    )
    print(outputs["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
