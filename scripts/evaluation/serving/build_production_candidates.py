#!/usr/bin/env python3
"""Build strict local Production request or target-alignment candidates."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))
from dataset_pipeline.serving_dataset import build_production_compact_candidate  # noqa: E402
_CANDIDATE_ROOT = REPOSITORY_ROOT / "profiler_output" / "serving" / "production"
_ALIGNMENT_ROOT = REPOSITORY_ROOT / "profiler_output" / "target_model_alignment"
_MODEL_IDS = ("glm-5.2", "kimi-k2.6")
_SCHEMA_PROFILES = ("glm_5_2_debug_csv_v1", "kimi_k2_6_debug_csv_v1")
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", type=Path)
    sources.add_argument("--target-model-validation-run", type=Path)
    parser.add_argument("--model-id", choices=_MODEL_IDS)
    parser.add_argument("--case-id")
    parser.add_argument("--schema-profile", choices=_SCHEMA_PROFILES)
    parser.add_argument("--upstream-success-provenance")
    parser.add_argument("--converter-source-commit")
    parser.add_argument("--target-model-source-root", type=Path,
                        help="caller-supplied canonical cases, organized as model_id/debug_workload_1s")
    parser.add_argument("--target-model-candidate-root", type=Path,
                        default=_ALIGNMENT_ROOT)
    return parser.parse_args(argv)
def _build_target_alignment(args) -> tuple[Path, ...]:
    from dataset_pipeline.serving_dataset.target_model_alignment import write_target_model_alignment_candidate
    from evaluation.serving.simulation.target_model_alignment import build_target_model_alignment, load_target_model_validation_run
    loaded = load_target_model_validation_run(args.target_model_validation_run)
    outputs = []
    for model_id in _MODEL_IDS:
        payload = build_target_model_alignment(
            model_id, loaded["payload"], validation_run_sha256=loaded["sha256"],
            repository_root=REPOSITORY_ROOT)
        source = args.target_model_source_root / model_id / "debug_workload_1s"
        destination = args.target_model_candidate_root / model_id / "debug_workload_1s"
        outputs.append(write_target_model_alignment_candidate(
            source, destination, payload,
            candidate_root=args.target_model_candidate_root))
    return tuple(outputs)
def run_cli(argv=None) -> Path | tuple[Path, ...]:
    args = _parse_args(argv)
    raw_values = (args.model_id, args.case_id, args.schema_profile,
                  args.upstream_success_provenance, args.converter_source_commit)
    if args.target_model_validation_run is not None:
        if any(value is not None for value in raw_values):
            raise ValueError("target alignment mode rejects raw candidate arguments")
        if args.target_model_source_root is None:
            raise ValueError("target alignment mode requires --target-model-source-root")
        if not args.target_model_source_root.is_dir():
            raise ValueError("target-model source root must be an existing directory")
        return _build_target_alignment(args)
    if args.target_model_source_root is not None:
        raise ValueError("raw candidate mode rejects --target-model-source-root")
    if any(value is None for value in raw_values):
        raise ValueError("raw candidate mode requires all raw candidate arguments")
    if not args.source.is_file():
        raise ValueError(f"production raw CSV is not a file: {args.source}")
    source_bytes = args.source.read_bytes()
    destination = _CANDIDATE_ROOT / args.model_id / args.case_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    return build_production_compact_candidate(
        source_bytes, destination, source_name=args.source.name,
        model_id=args.model_id, case_id=args.case_id,
        schema_profile=args.schema_profile,
        upstream_success_provenance=args.upstream_success_provenance,
        converter_source_commit=args.converter_source_commit)
def main(argv=None) -> int:
    output = run_cli(argv)
    if isinstance(output, Path):
        outputs = (output,)
    elif isinstance(output, tuple):
        outputs = output
    else:
        raise TypeError("candidate CLI returned an unsupported output type")
    for path in outputs:
        print(path)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
