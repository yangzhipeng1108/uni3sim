#!/usr/bin/env python3
"""Measure how dynamic Scheduling propagates synthetic step-duration errors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset.compact_case import (  # noqa: E402
    load_compact_case,
)
from evaluation.serving.metrics.diagnostics.scheduler_error_propagation import (  # noqa: E402
    CELL_IDS,
    SCENARIOS,
    run_scheduler_error_propagation_suite,
    run_scheduler_error_propagation_worker,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--cell-id", choices=CELL_IDS, default=CELL_IDS[1])
    parser.add_argument("--output-dir")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seeds", default="11,29,47,71,101")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.worker:
        if args.scenario is None or args.seed is None or args.output is None:
            parser.error("--worker requires --scenario, --seed, and --output")
    elif args.output_dir is None:
        parser.error("top-level run requires --output-dir")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository_root = Path(args.repository_root).resolve(strict=True)
    if args.worker:
        run_scheduler_error_propagation_worker(
            repository_root,
            args.cell_id,
            args.scenario,
            args.seed,
            Path(args.output),
            load_compact_case,
        )
        return 0
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique comma-separated integers")
    run_scheduler_error_propagation_suite(
        repository_root,
        args.cell_id,
        Path(args.output_dir).resolve(),
        seeds,
        args.jobs,
        (sys.executable, str(Path(__file__).resolve())),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
