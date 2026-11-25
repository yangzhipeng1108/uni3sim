#!/usr/bin/env python3
"""Run one auditable C500 Event attempt with explicit container/device group."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.kernel_dataset.metax_c500_event_campaign import (
    run_event_attempt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--ordinal", required=True, type=int)
    parser.add_argument(
        "--cohort", required=True, choices=("new", "existing_strict"))
    parser.add_argument("--source-id", required=True)
    parser.add_argument(
        "--family", required=True,
        choices=("vector", "flash_attn", "gemm", "fused_moe"))
    parser.add_argument("--physical-device", required=True, type=int)
    parser.add_argument("--container", required=True,
                        help="Docker container name or ID for this capture")
    parser.add_argument("--device-gid", required=True, type=int,
                        help="Host group ID granting access to the selected accelerator")
    parser.add_argument("--case-id", required=True, action="append")
    parser.add_argument("--snapshot-interval-seconds", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    result = run_event_attempt(
        attempt_root=args.attempt_root, attempt_id=args.attempt_id,
        ordinal=args.ordinal, cohort=args.cohort, source_id=args.source_id,
        family=args.family, physical_device=args.physical_device,
        container=args.container, device_gid=args.device_gid,
        planned_case_ids=args.case_id, command_argv=command,
        snapshot_interval_seconds=args.snapshot_interval_seconds,
        wrapper_path=Path(__file__))
    print(json.dumps(result, sort_keys=True))
    return_code_by_status = {"completed": 0, "failed": 1}
    return return_code_by_status[result["attempt"]["terminal_status"]]


if __name__ == "__main__":
    raise SystemExit(main())
