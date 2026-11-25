#!/usr/bin/env python3
"""Extract correlated A800 selected-step CUDA Graph intervals from NSYS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset.nvidia_a800_selected_step_nsys import (  # noqa: E402
    analyze_a800_selected_step_nsys,
)
from evaluation.serving.metrics.diagnostics.selected_step_injection import (  # noqa: E402
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys-sqlite", type=Path, required=True)
    parser.add_argument("--scheduler-trace", type=Path, required=True)
    parser.add_argument("--step-id", action="append", required=True)
    parser.add_argument("--expected-rank-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = analyze_a800_selected_step_nsys(
        arguments.nsys_sqlite,
        arguments.scheduler_trace,
        arguments.step_id,
        arguments.expected_rank_count,
    )
    write_json(arguments.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
