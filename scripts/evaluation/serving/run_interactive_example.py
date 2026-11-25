#!/usr/bin/env python3
"""Run a CPU-only Serving example and generate its offline report."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.serving.simulation.interactive_examples import run_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("single", "tp2", "pd", "four"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sdk-python", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_example(args.mode, args.output, args.sdk_python, args.source_root, record=not args.no_record)))


if __name__ == "__main__":
    main()
