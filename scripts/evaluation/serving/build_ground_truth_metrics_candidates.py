#!/usr/bin/env python3
"""Build candidate ShareGPT metrics from production ground-truth results."""
import argparse
from pathlib import Path
import sys
from typing import Sequence
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.serving_dataset.ground_truth_metrics_converter import (  # noqa: E402
    build_ground_truth_metrics_candidates,
)
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Validate production ground_truth.json files and build candidate-only "
        "compact-schema-v2 ShareGPT metrics.json files"
    ))
    parser.add_argument(
        "--ground-truth",
        action="append", required=True, type=Path,
        help="ground_truth.json file or recursively searched directory; repeatable",
    )
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "serving_dataset/sharegpt")
    parser.add_argument("--suite-manifest", type=Path, default=(
        ROOT / "configs/evaluation/experiments/sharegpt_phase1.json"))
    parser.add_argument(
        "--capture-suite-manifest",
        action="append", required=True, type=Path,
        help="immutable suite snapshot recorded by ground truth; repeatable",
    )
    parser.add_argument(
        "--capture-dataset-root",
        type=Path, help=(
            "dataset root containing the immutable case manifests used by the "
            "capture; defaults to --dataset-root for legacy in-place captures"
        ),
    )
    parser.add_argument(
        "--template-hardware-id",
        help=("create absent targets from this canonical hardware's matching "
              "model_id/case_id; existing targets are rejected"),
    )
    parser.add_argument(
        "--converter-source-commit",
        help="exact converter commit; required only with --template-hardware-id",
    )
    parser.add_argument(
        "--scheduler-policy-attestation",
        type=Path, help=(
            "structured per-case attestation required when old ground truth "
            "does not record scheduler_policy"
        ),
    )
    parser.add_argument(
        "--capture-source-snapshot-manifest",
        action="append", default=[], type=Path,
        help="snapshot manifest required for dirty ground-truth captures; repeatable",
    )
    parser.add_argument("--candidate-root", required=True, type=Path,
                        help="new directory outside the canonical Serving dataset")
    args = parser.parse_args(argv)
    outputs = build_ground_truth_metrics_candidates(
        args.ground_truth,
        dataset_root=args.dataset_root,
        suite_manifest_path=args.suite_manifest,
        capture_suite_manifest_paths=args.capture_suite_manifest,
        scheduler_policy_attestation_path=args.scheduler_policy_attestation,
        capture_source_snapshot_manifest_paths=args.capture_source_snapshot_manifest,
        candidate_root=args.candidate_root,
        capture_dataset_root=args.capture_dataset_root,
        template_hardware_id=args.template_hardware_id,
        converter_source_commit=args.converter_source_commit,
    )
    print(f"wrote {len(outputs)} ground-truth metrics candidates")
    print(args.candidate_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
