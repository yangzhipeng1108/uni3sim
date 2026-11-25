#!/usr/bin/env python3
"""CLI for the MetaX C500 full OP Event campaign audit."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.kernel_dataset.metax_c500_event_audit import (
    audit_full_event_campaign,
    write_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed audit of complete MetaX C500 OP Device Event evidence."))
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_full_event_campaign(args.campaign_manifest, args.output)
    write_audit_report(args.output, report)
    print(json.dumps({
        "output": report["output"],
        "status": report["status"],
        "strict_event_case_count": report["strict_event_case_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
