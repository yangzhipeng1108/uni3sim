#!/usr/bin/env python3
"""Build one immutable 1P1D Serving compact candidate from measured raw data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.pd_compact_candidate import (  # noqa: E402
    PDRunArtifacts,
    build_pd_disaggregated_candidate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one schema-v5 measured ShareGPT 1P1D candidate"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--source-case", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--frozen-manifest", required=True, type=Path)
    parser.add_argument("--frozen-requests", required=True, type=Path)
    parser.add_argument("--run-contract", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--observer-effect-report", required=True, type=Path)
    parser.add_argument("--prefill-scheduler", required=True, type=Path)
    parser.add_argument("--decode-scheduler", required=True, type=Path)
    parser.add_argument("--router-handoffs", required=True, type=Path)
    parser.add_argument("--connector-events", required=True, nargs="+", type=Path)
    args = parser.parse_args(argv)
    output = build_pd_disaggregated_candidate(
        args.output,
        candidate_root=args.candidate_root,
        artifacts=PDRunArtifacts(
            source_case_dir=args.source_case,
            frozen_manifest_path=args.frozen_manifest,
            frozen_requests_path=args.frozen_requests,
            run_contract_path=args.run_contract,
            ground_truth_path=args.ground_truth,
            observer_effect_report_path=args.observer_effect_report,
            prefill_scheduler_path=args.prefill_scheduler,
            decode_scheduler_path=args.decode_scheduler,
            router_handoffs_path=args.router_handoffs,
            connector_event_paths=tuple(args.connector_events),
            raw_root=args.raw_root,
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
