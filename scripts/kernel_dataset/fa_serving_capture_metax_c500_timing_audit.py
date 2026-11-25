#!/usr/bin/env python3
"""Emit C500 FA Event/trace timing evidence when counters are unavailable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture_metax_c500 import (
    build_c500_timing_without_counter,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--workload-digest", required=True)
    parser.add_argument("--logical-identity-sha256", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--event", required=True, type=Path)
    parser.add_argument("--trace", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"timing audit already exists: {output}")
    value = build_c500_timing_without_counter(
        case_id=args.case_id,
        workload_digest=args.workload_digest,
        logical_identity_sha256=args.logical_identity_sha256,
        source_snapshot_sha256=args.source_snapshot_sha256,
        event_path=args.event,
        trace_paths=args.trace,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
