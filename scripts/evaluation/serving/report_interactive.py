#!/usr/bin/env python3
"""Regenerate an offline Serving report from retained visualization evidence."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    from evaluation.reporting.serving_interactive import regenerate_report
    report = regenerate_report(payload, args.output)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
