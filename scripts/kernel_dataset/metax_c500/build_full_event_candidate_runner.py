#!/usr/bin/env python3
"""CLI for the candidate-only MetaX C500 987-case Event dataset builder."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.kernel_dataset.metax_c500_event_candidate import (
    build_full_event_candidate,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build four candidate-only C500 datasets from an accepted full "
            "OP Device Event audit."))
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--full-event-audit", required=True, type=Path)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_full_event_candidate(
        selection_path=args.selection_manifest,
        audit_path=args.full_event_audit,
        canonical_root=args.canonical_root,
        output_root=args.output_root,
    )
    print(json.dumps({
        "output": str((args.output_root / "source_manifest.json").resolve()),
        "status": manifest["status"],
        "strict_event_case_count": manifest["event_timing"][
            "strict_case_count"],
        "trace_binding_status": manifest["trace_binding"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
