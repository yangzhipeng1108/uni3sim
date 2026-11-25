#!/usr/bin/env python3
"""Build a Serving-to-Kernel coverage census for one hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics.diagnostics import (  # noqa: E402
    metax_c500_kernel_coverage as coverage,
)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = coverage.parse_coverage_arguments(argv)
    return parser


def run_cli(argv=None) -> dict[str, object]:
    raw = list(sys.argv[1:] if argv is None else argv)
    hardware_parser = argparse.ArgumentParser(add_help=False)
    hardware_parser.add_argument(
        "--hardware-id",
        required=True,
        choices=("nvidia_a800", "metax_c500", "ascend_910b"),
    )
    hardware_args, remaining = hardware_parser.parse_known_args(raw)
    args = _parse_args(remaining)
    configs = (
        ("configs/evaluation/experiments/sharegpt_phase1.json",)
        if args.config is None else tuple(args.config)
    )
    selections = coverage.select_coverage_cells(
        REPOSITORY_ROOT,
        args,
        configs,
        hardware_id=hardware_args.hardware_id,
    )
    candidate_root = (
        None if args.candidate_root is None
        else coverage.resolve_candidate_root(
            REPOSITORY_ROOT, args.candidate_root)
    )
    report = coverage.run_coverage_census(
        REPOSITORY_ROOT,
        selections,
        input_mode=args.input_mode,
        execution_mode=args.execution_mode,
        candidate_root=candidate_root,
        jobs=args.jobs,
        compact_case_loader=load_compact_case,
    )
    output_root = (
        REPOSITORY_ROOT / "experiment_logs" / "serving" / args.run_id
        / hardware_args.hardware_id / "kernel_coverage"
    )
    coverage.write_report(report, output_root)
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "hardware_id": hardware_args.hardware_id,
        "output_dir": str(output_root.resolve()),
        "input_mode": args.input_mode,
        "hard_gate_pass": report["hard_gate_pass"],
        "case_counts": report["case_counts"],
        "reconciliation": report["reconciliation"],
    }


def main(argv=None) -> int:
    summary = run_cli(argv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(not summary["hard_gate_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
