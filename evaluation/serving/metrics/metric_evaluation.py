#!/usr/bin/env python3
"""Generate comparison reports for manifest-driven real vLLM experiments."""

from __future__ import annotations

import argparse

from evaluation.serving.metrics.serving_metric_comparison import (
    aggregate_run_directory,
    compare_cell,
    generate_final_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate real vLLM E2E comparison reports")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cell = subparsers.add_parser("cell", help="Generate one cell comparison report")
    cell.add_argument("--command-manifest", required=True)
    cell.add_argument("--ground-truth", default=None)
    cell.add_argument("--prediction", default=None)
    cell.add_argument("--output", default=None)
    cell.add_argument("--markdown", default=None)
    cell.add_argument("--diagnostic-mode", choices=("colocated", "pd"), default=None)

    aggregate = subparsers.add_parser(
        "aggregate", help="Aggregate persisted cell comparison reports")
    aggregate.add_argument("--run-dir", required=True)
    aggregate.add_argument("--output-dir", default=None)

    final = subparsers.add_parser(
        "final", help="Generate final modeling status report from a run directory")
    final.add_argument("--run-dir", required=True)
    final.add_argument("--output-dir", default=None)

    args = parser.parse_args(argv)
    if args.command == "cell":
        compare_cell(
            args.command_manifest,
            ground_truth_path=args.ground_truth,
            prediction_path=args.prediction,
            output_path=args.output,
            markdown_path=args.markdown,
            diagnostic_mode=args.diagnostic_mode,
        )
    elif args.command == "aggregate":
        aggregate_run_directory(args.run_dir, output_dir=args.output_dir)
    elif args.command == "final":
        generate_final_report(args.run_dir, output_dir=args.output_dir)
    else:
        raise ValueError(f"unknown metric evaluation command: {args.command!r}")
    return 0
