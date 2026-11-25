#!/usr/bin/env python3
"""Regenerate the seven-model Goal checkpoint from static/source artifacts."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.simulation.model_operator_probe import (
    write_operator_support_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tests-log", type=Path)
    parser.add_argument("--recheck-log", type=Path)
    parser.add_argument("--kernel-results", type=Path)
    parser.add_argument("--current-probe", type=Path, action="append", default=[],
                        help="additional snapshot-bound static evidence; historical artifacts stay unchanged")
    parser.add_argument("--source-storage-probe", type=Path, action="append", default=[],
                        help="unpriced storage subset evidence; excluded from complete model-call coverage")
    parser.add_argument("--weight-diagnostic", type=Path, action="append", default=[],
                        help="snapshot-bound V4 or Kimi outer constructor parameters; excluded from normal capacity acceptance")
    parser.add_argument("--source-graph-probe", type=Path, action="append", default=[],
                        help="V4 provenance graph evidence; contains no lowering or pricing results")
    args = parser.parse_args()
    if args.reference is None or args.tests_log is None or args.kernel_results is None:
        parser.error("checkpoint requires --reference, --tests-log and --kernel-results")
    write_operator_support_checkpoint(
        args.output, args.reference, args.tests_log, args.recheck_log,
        args.kernel_results, current_probes=tuple(args.current_probe),
        source_storage_probes=tuple(args.source_storage_probe),
        weight_diagnostics=tuple(args.weight_diagnostic),
        source_graph_probes=tuple(args.source_graph_probe))
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
