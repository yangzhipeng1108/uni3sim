#!/usr/bin/env python3
"""Price one case's observed steps under exact and real graph capture sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics import (  # noqa: E402
    sharegpt_compact_evaluation as compact,
)
from evaluation.serving.metrics.diagnostics import (  # noqa: E402
    graph_padding_paired_steps as paired,
)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/evaluation/experiments/sharegpt_phase1.json")
    parser.add_argument("--case", default="ascend_910b/qwen3-4b/tp1_qps16")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step-limit", type=int, default=None)
    return parser.parse_args(argv)


def run_cli(argv=None) -> dict[str, object]:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = compact.load_phase1_config(
        compact.resolve_repository_input(
            REPOSITORY_ROOT, args.config, "experiment config"))
    cell = compact.select_phase1_cells(config, [args.case], all_cases=False)[0]
    case = load_compact_case(
        compact.compact_case_path(REPOSITORY_ROOT, config, cell))
    report = paired.run_paired_steps(
        REPOSITORY_ROOT, config, cell, case, step_limit=args.step_limit)
    output_root = (
        REPOSITORY_ROOT / "experiment_logs" / "serving" / args.run_id
        / "graph_padding_paired_steps")
    path = paired.write_report(report, output_root)
    return {
        "report_path": str(path),
        "cell_id": report["cell_id"],
        "summary": {
            name: value for name, value in report["summary"].items()
            if name != "kernel_delta_ms"
        },
    }


def main(argv=None) -> int:
    print(json.dumps(run_cli(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
