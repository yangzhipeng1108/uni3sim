#!/usr/bin/env python3
"""Validate one MCCL mcTracer run and write normalized selection facts."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.collective_selection.mccl_mctracer import (  # noqa: E402
    write_selection_observations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    arguments = parser.parse_args(argv)
    print(write_selection_observations(arguments.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
