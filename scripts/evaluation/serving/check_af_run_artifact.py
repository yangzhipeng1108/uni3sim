#!/usr/bin/env python3
"""Validate a finished standalone AF run-output artifact (Gate A2 checks)."""
import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.serving.simulation.af_run_invariants import (  # noqa: E402
    validate_af_run_artifact_file,
)

DEFAULT_MODEL = "configs/model/qwen3-4b.json"
DEFAULT_REQUESTS = (
    "serving_dataset/sharegpt/nvidia_a800/qwen3-4b/tp1_qps16/requests.csv")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", required=True,
                        help="run-output JSON from run_serving_simulation.py "
                             "--deployment af_disagg")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--requests", default=DEFAULT_REQUESTS,
                        help="canonical request CSV the run consumed")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    summary = validate_af_run_artifact_file(
        args.run_output, model_path=args.model, requests_path=args.requests)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
