#!/usr/bin/env python3
"""Write strict paired trace-off/trace-on PD observer-effect evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.pd_observer_effect import (  # noqa: E402
    write_pd_observer_effect,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare paired real PD runs")
    parser.add_argument("--trace-off", required=True, type=Path)
    parser.add_argument("--trace-on", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    write_pd_observer_effect(args.output, args.trace_off, args.trace_on)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
