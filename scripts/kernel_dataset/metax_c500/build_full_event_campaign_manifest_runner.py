#!/usr/bin/env python3
"""CLI for assembling one immutable MetaX C500 Event campaign manifest."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.kernel_dataset.metax_c500_event_campaign_manifest import (
    build_full_event_campaign_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a full C500 Event campaign manifest for audit.")
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    manifest = build_full_event_campaign_manifest(
        selection_path=args.selection_manifest,
        campaign_root=args.campaign_root,
        source_registry_path=args.source_registry,
        output_root=args.output_root,
    )
    print(json.dumps({
        "output": str((args.output_root / "campaign_manifest.json").resolve()),
        "attempt_count": len(manifest["attempts"]),
        "selection_rule": manifest["selection_rule"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
