#!/usr/bin/env python3
"""Sweep a controlled admission shift across selected Serving cells."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics import admission_sensitivity_sweep as sweep  # noqa: E402
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact  # noqa: E402

DEFAULT_CONFIG = Path("configs/evaluation/experiments/sharegpt_phase1.json")


def _evaluate(payload):
    config, cell, source_commit, shift_ms, scratch, shift_field = payload
    point = sweep.evaluate_shift(
        REPOSITORY_ROOT, config, cell, source_commit, shift_ms, Path(scratch),
        shift_field, load_validated_case=load_compact_case)
    return {
        "cell_id": point.cell_id,
        "shift_ms": point.shift_ms,
        "shift_field": point.shift_field,
        "predicted_ttft_mean_ms": point.predicted_ttft_mean_ms,
        "observed_ttft_mean_ms": point.observed_ttft_mean_ms,
        "paired_mape_percent": point.paired_mape_percent,
        "signed_error_percent": point.signed_error_percent,
        "predicted_wait_mean_ms": point.predicted_wait_mean_ms,
        "predicted_processing_mean_ms": point.predicted_processing_mean_ms,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--case", action="append", required=True,
                        help="hardware/model/case_id; repeat for more cells")
    parser.add_argument("--shift-ms", type=float, action="append", required=True,
                        help="admission shift in ms; repeat for a sweep")
    parser.add_argument("--scratch", required=True,
                        help="directory for the shifted profile copies")
    parser.add_argument("--shift-field", default="api_entry",
                        choices=sweep.SHIFTABLE_FIELDS,
                        help="which request path service time the shift moves")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    config_path = compact.resolve_repository_input(
        REPOSITORY_ROOT, args.config, "experiment config")
    config = compact.load_phase1_config(config_path)
    cells = compact.select_phase1_cells(config, args.case, all_cases=False)
    source_commit = compact.repository_source_commit(REPOSITORY_ROOT)
    shifts = [0.0] + [value for value in args.shift_ms if value != 0.0]

    work = [
        (config, cell, source_commit, shift, args.scratch, args.shift_field)
        for cell in cells
        for shift in shifts
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(_evaluate, work))

    points = [
        sweep.SweepPoint(
            cell_id=record["cell_id"],
            shift_ms=record["shift_ms"],
            shift_field=record["shift_field"],
            predicted_ttft_mean_ms=record["predicted_ttft_mean_ms"],
            observed_ttft_mean_ms=record["observed_ttft_mean_ms"],
            paired_mape_percent=record["paired_mape_percent"],
            signed_error_percent=record["signed_error_percent"],
            predicted_wait_mean_ms=record["predicted_wait_mean_ms"],
            predicted_processing_mean_ms=record["predicted_processing_mean_ms"],
        )
        for record in records
    ]
    summary = {
        "schema_version": 1,
        "kind": "admission_sensitivity_sweep",
        "shift_field": args.shift_field,
        "source_commit": source_commit,
        "points": records,
        "response_ratios": sweep.response_ratios(points),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["response_ratios"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
