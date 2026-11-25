#!/usr/bin/env python3
"""Reconstruct client TTFT from an absolute-clock Scheduler capture."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.traces.oracle_ttft_reconstruction import (  # noqa: E402
    run_oracle_ttft_reconstruction_cli,
)


if __name__ == "__main__":
    run_oracle_ttft_reconstruction_cli()
