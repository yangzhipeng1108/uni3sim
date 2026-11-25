#!/usr/bin/env python3
"""CLI for building the audited MetaX C500 Device Event selection."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.metax_c500_full_event_selection import (
    _write_json,
    build_full_event_selection,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the 149-case Vector warning set and combine it with "
            "the 344-case C500 completion manifest."))
    parser.add_argument("--completion-manifest", required=True, type=Path)
    parser.add_argument("--vector-raw-root", required=True, type=Path)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_full_event_selection(
        args.completion_manifest, args.vector_raw_root,
        args.canonical_root)
    _write_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "selected_case_count": payload["selected_case_count"],
        "group_counts": {
            group: entry["case_count"]
            for group, entry in payload["groups"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
