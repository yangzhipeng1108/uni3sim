#!/usr/bin/env python3
"""Compare baseline, detailed, and fast standard Serving result files."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.reporting.serving_execution_mode_comparison import compare_result_files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--detailed", type=Path, required=True)
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-command")
    parser.add_argument("--detailed-command")
    parser.add_argument("--fast-command")
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--supplement", type=Path,
        help="JSON with boolean S4/S6/S7 gates, evidence, and optimization notes")
    args = parser.parse_args(argv)
    commands = None
    if any(value is not None for value in (args.baseline_command, args.detailed_command, args.fast_command)):
        if any(value is None for value in (args.baseline_command, args.detailed_command, args.fast_command)):
            parser.error("all three mode commands must be supplied together")
        commands = {"baseline": args.baseline_command, "detailed": args.detailed_command, "fast": args.fast_command}
    environment = None
    if args.environment is not None:
        environment = json.loads(args.environment.read_text())
    supplement = None
    if args.supplement is not None:
        supplement = json.loads(args.supplement.read_text())
    _, report, comparison = compare_result_files(
        args.baseline, args.detailed, args.fast, args.output_dir,
        commands=commands, environment=environment, supplement=supplement)
    print(report)
    print(json.dumps(comparison["gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
