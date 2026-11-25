#!/usr/bin/env python3
"""Create fail-closed candidate audit artifacts for FA serving capture."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture_audit import (
    build_a800_counter_candidate,
    build_ascend_counter_candidate,
    build_canary_closure,
    write_json_new,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware", required=True, choices=("nvidia_a800", "ascend_910b"))
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--workload-digest", required=True)
    parser.add_argument("--source-snapshot-sha256")
    parser.add_argument("--raw-counter-csv", type=Path)
    parser.add_argument("--nsys-sqlite", action="append", type=Path)
    parser.add_argument("--counter-op-basic", action="append", type=Path)
    parser.add_argument("--counter-pipe", action="append", type=Path)
    parser.add_argument("--counter-memory", action="append", type=Path)
    parser.add_argument("--timeline-op-summary", action="append", type=Path)
    parser.add_argument("--timeline-program-hash", action="append", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-input", type=Path)
    parser.add_argument("--timing-audit", type=Path)
    parser.add_argument("--closure-output", type=Path)
    args = parser.parse_args()
    if args.candidate_input is not None:
        if (args.timing_audit is None or args.closure_output is None
                or args.output is not None
                or any(value is not None for value in (
                    args.raw_counter_csv, args.nsys_sqlite, args.counter_op_basic,
                    args.counter_pipe, args.counter_memory, args.timeline_op_summary,
                    args.timeline_program_hash))):
            raise ValueError("canary closure requires only candidate-input, timing-audit, and closure-output")
        closure = build_canary_closure(
            hardware_id=args.hardware,
            case_id=args.case_id,
            workload_digest=args.workload_digest,
            timing_audit_path=args.timing_audit,
            counter_candidate_path=args.candidate_input,
        )
        write_json_new(args.closure_output, closure)
        print(args.closure_output.resolve())
        return
    if args.output is None:
        raise ValueError("counter candidate requires output")
    if args.source_snapshot_sha256 is None:
        raise ValueError("counter candidate requires source-snapshot-sha256")
    if args.hardware == "nvidia_a800":
        if (args.raw_counter_csv is None or args.nsys_sqlite is None
                or any(value is not None for value in (
                    args.counter_op_basic, args.counter_pipe, args.counter_memory,
                    args.timeline_op_summary, args.timeline_program_hash))):
            raise ValueError("A800 requires only raw-counter-csv and nsys-sqlite")
        candidate = build_a800_counter_candidate(
            case_id=args.case_id,
            workload_digest=args.workload_digest,
            raw_csv_path=args.raw_counter_csv,
            nsys_sqlite_paths=args.nsys_sqlite,
            source_snapshot_sha256=args.source_snapshot_sha256,
        )
    elif args.hardware == "ascend_910b":
        if (args.raw_counter_csv is not None or args.nsys_sqlite is not None
                or any(value is None for value in (
                    args.counter_op_basic, args.counter_pipe, args.counter_memory,
                    args.timeline_op_summary, args.timeline_program_hash))):
            raise ValueError("Ascend requires only the five counter/timing path groups")
        candidate = build_ascend_counter_candidate(
            case_id=args.case_id,
            workload_digest=args.workload_digest,
            source_snapshot_sha256=args.source_snapshot_sha256,
            counter_op_basic_paths=args.counter_op_basic,
            counter_pipe_paths=args.counter_pipe,
            counter_memory_paths=args.counter_memory,
            timeline_op_summary_paths=args.timeline_op_summary,
            timeline_program_hash_paths=args.timeline_program_hash,
        )
    else:
        raise AssertionError(f"unhandled hardware: {args.hardware!r}")
    write_json_new(args.output, candidate)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
