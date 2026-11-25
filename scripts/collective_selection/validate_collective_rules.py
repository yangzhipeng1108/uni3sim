#!/usr/bin/env python3
"""Validate normalized collective observations against selection rules."""

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.collective_selection.rule_audit import (  # noqa: E402
    audit_selection_observations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "selection_observations",
        type=Path,
        nargs="+",
        help="normalized selection_observations.json paths",
    )
    arguments = parser.parse_args(argv)
    summary = audit_selection_observations(arguments.selection_observations)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
