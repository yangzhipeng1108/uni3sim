#!/usr/bin/env python3
"""Run Forward accuracy on the published FA serving capture datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.kernel.fa_serving_coverage_report import build_report, write_report


def _parse_release(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(f"--release must be HARDWARE=PATH: {value!r}")
        hardware_id, raw_path = value.split("=", 1)
        if hardware_id in result or not raw_path:
            raise ValueError(f"invalid --release: {value!r}")
        result[hardware_id] = Path(raw_path).resolve()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, action="append")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(_parse_release(args.release))
    outputs = write_report(report, args.output_root.resolve())
    print(outputs["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
