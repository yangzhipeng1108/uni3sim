#!/usr/bin/env python3
"""CLI for collecting one complete MetaX C500 host process snapshot."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.kernel_dataset.metax_c500_process_isolation import (
    collect_full_system_process_snapshot,
    write_process_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect all host processes and C500 device-file owners.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mx-smi", type=Path, default=Path("/opt/mxdriver/bin/mx-smi"))
    args = parser.parse_args()
    snapshot = collect_full_system_process_snapshot(mx_smi_path=args.mx_smi)
    write_process_snapshot(args.output, snapshot)
    print(json.dumps({
        "device_process_count": len(snapshot["device_processes"]),
        "output": str(args.output.resolve()),
        "process_count": len(snapshot["processes"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
