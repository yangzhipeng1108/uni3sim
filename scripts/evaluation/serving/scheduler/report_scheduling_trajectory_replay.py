#!/usr/bin/env python3
"""Plot Scheduling Trajectory Replay and optional A/B/C/D attribution results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.metrics.diagnostics import scheduling_trajectory_replay_report  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dynamic-results",
        action="append",
        default=[],
        type=Path,
        help="aggregate dynamic-scheduler results.json; repeat for split suites",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    cells = scheduling_trajectory_replay_report.load_scheduling_trajectory_replay_results(
        arguments.input_dir
    )
    paths = scheduling_trajectory_replay_report.write_scheduling_trajectory_replay_quantile_plots(
        arguments.output_dir,
        cells,
    )
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    if arguments.dynamic_results:
        dynamic = scheduling_trajectory_replay_report.load_dynamic_evaluation_results(
            arguments.dynamic_results
        )
        attribution_paths = scheduling_trajectory_replay_report.write_latency_source_attribution_plots(
            arguments.output_dir.parent / f"{arguments.output_dir.name}_latency_sources",
            cells,
            dynamic,
        )
        for name, path in sorted(attribution_paths.items()):
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
