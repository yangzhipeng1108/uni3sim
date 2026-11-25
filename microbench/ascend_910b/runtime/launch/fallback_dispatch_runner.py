#!/usr/bin/env python3
"""Measure the independent compiled-FX fallback-dispatch metadata path.

The probe deliberately contains only a synthetic NPU tensor and repeated
``aten.reshape`` metadata calls.  It does not replay a model, Serving shape,
graph bucket, or production duration.  The operation implementation is shared
with the broader metadata probe; this runner owns the fallback-dispatch-only
selection and result contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import torch_npu

from compiled_fx_metadata_runner import (
    CHAIN_LENGTHS,
    FIT_CHAIN_LENGTHS,
    _fit,
    _measure,
    _sha256,
    _validate_output_dir,
    _write_json,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--baseline-commit", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if isinstance(arguments.device_id, bool) or arguments.device_id < 0:
        raise ValueError("--device-id must be a non-negative integer")
    if (len(arguments.baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in arguments.baseline_commit)):
        raise ValueError("--baseline-commit must be a lowercase full Git SHA")
    output_dir = _validate_output_dir(arguments.output_dir)
    output_dir.mkdir(parents=True)
    trials_path = output_dir / "trials.jsonl"
    source = torch.empty((256, 256), dtype=torch.float32,
                         device=f"npu:{arguments.device_id}")
    torch.npu.synchronize()
    cells = [
        _measure("fx_aten_reshape_metadata", source, count, trials_path)
        for count in CHAIN_LENGTHS
    ]
    runner_path = Path(__file__).resolve()
    summary = {
        "schema_version": 1,
        "kind": "ascend_fallback_dispatch_microbenchmark",
        "hardware_id": "ascend_910b",
        "baseline_commit": arguments.baseline_commit,
        "environment": {"device_id": arguments.device_id},
        "runner": {
            "path": str(runner_path),
            "sha256": _sha256(runner_path),
            "operation_source_path": str(
                (runner_path.parent / "compiled_fx_metadata_runner.py").resolve()),
            "operation_source_sha256": _sha256(
                runner_path.parent / "compiled_fx_metadata_runner.py"),
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_contract": {
            "source_shape": [256, 256],
            "mechanism": "fx_aten_reshape_metadata",
            "chain_lengths": list(CHAIN_LENGTHS),
            "fit_chain_lengths": sorted(FIT_CHAIN_LENGTHS),
            "holdout_chain_length": 512,
            "production_or_evaluation_duration_consumed": False,
        },
        "cells": cells,
        "fit": _fit(cells, "fx_aten_reshape_metadata"),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
