#!/usr/bin/env python3
"""Audit an Ascend 910B AIV SyncAll mechanism capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from sync_all_runner import BLOCK_DIMS, TRIALS, _ordered_block_dims


SYSTEM_COUNTER_TICK_NS = 20.0
MAXIMUM_HALF_MEDIAN_DELTA = 0.20
MAXIMUM_MODEL_MAPE = 0.10
MAXIMUM_MODEL_APE = 0.20
FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "block_cycles", "host_wall_ns", "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_affine(points):
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        raise ValueError("SyncAll affine fit needs distinct worker counts")
    per_worker_ns = sum(
        (x - x_mean) * (y - y_mean) for x, y in points) / denominator
    startup_ns = y_mean - per_worker_ns * x_mean
    if startup_ns <= 0 or per_worker_ns <= 0:
        raise ValueError("SyncAll affine coefficients must be positive")
    return startup_ns, per_worker_ns


def build_report(raw_path, source_commit, source_snapshot_sha256):
    with raw_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("SyncAll raw header changed")
        source_rows = list(reader)
    if len(source_rows) != TRIALS * len(BLOCK_DIMS):
        raise ValueError("SyncAll raw campaign is incomplete")

    rows = []
    run_ids = set()
    for raw in source_rows:
        run_ids.add(raw["run_id"])
        trial = int(raw["trial"])
        order_position = int(raw["order_position"])
        block_dim = int(raw["block_dim"])
        if trial < 1 or trial > TRIALS:
            raise ValueError("SyncAll trial is outside the frozen campaign")
        ordered = _ordered_block_dims(trial - 1)
        if order_position < 0 or order_position >= len(ordered):
            raise ValueError("SyncAll order position is invalid")
        if ordered[order_position] != block_dim:
            raise ValueError("SyncAll row order changed")
        cycles = tuple(int(value) for value in raw["block_cycles"].split(";"))
        if len(cycles) != block_dim or any(value <= 0 for value in cycles):
            raise ValueError("SyncAll block-cycle vector is invalid")
        if int(raw["process_pid"]) <= 0 or int(raw["host_wall_ns"]) <= 0:
            raise ValueError("SyncAll process health fields are invalid")
        if raw["validation"] != "pass":
            raise ValueError("SyncAll device validation failed")
        rows.append({
            "trial": trial,
            "block_dim": block_dim,
            "completion_ns": max(cycles) * SYSTEM_COUNTER_TICK_NS,
        })
    if len(run_ids) != 1:
        raise ValueError("SyncAll raw must contain exactly one run ID")
    medians = {
        block_dim: statistics.median(
            row["completion_ns"] for row in rows
            if row["block_dim"] == block_dim)
        for block_dim in BLOCK_DIMS
    }
    spans = {
        block_dim: (
            max(row["completion_ns"] for row in rows
                if row["block_dim"] == block_dim)
            - min(row["completion_ns"] for row in rows
                  if row["block_dim"] == block_dim)
        ) / medians[block_dim]
        for block_dim in BLOCK_DIMS
    }
    if any(not math.isfinite(value) or value <= 0 for value in medians.values()):
        raise ValueError("SyncAll median must be positive and finite")
    half = TRIALS // 2
    if TRIALS % 2 != 0:
        raise ValueError("SyncAll trials must split into two balanced halves")
    half_median_deltas = {}
    for block_dim in BLOCK_DIMS:
        first = statistics.median(
            row["completion_ns"] for row in rows
            if row["block_dim"] == block_dim and row["trial"] <= half)
        second = statistics.median(
            row["completion_ns"] for row in rows
            if row["block_dim"] == block_dim and row["trial"] > half)
        half_median_deltas[block_dim] = (
            abs(first - second) / medians[block_dim])
    startup_ns, per_worker_ns = _fit_affine(tuple(medians.items()))
    fit_apes = {
        block_dim: abs(
            startup_ns + per_worker_ns * block_dim - duration_ns
        ) / duration_ns
        for block_dim, duration_ns in medians.items()
    }
    maximum_span = max(spans.values())
    maximum_half_median_delta = max(half_median_deltas.values())
    model_mape = statistics.mean(fit_apes.values())
    maximum_model_ape = max(fit_apes.values())
    mechanism_publishable = (
        maximum_half_median_delta <= MAXIMUM_HALF_MEDIAN_DELTA
        and model_mape <= MAXIMUM_MODEL_MAPE
        and maximum_model_ape <= MAXIMUM_MODEL_APE
    )
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aiv_sync_all",
        "model": "aiv_sync_all_affine_v1",
        "request_size_unit": "active_aiv_workers",
        "duration_unit": "ns_per_complete_sync_all",
        "active_aiv_workers": list(BLOCK_DIMS),
        "observed_median_duration_ns": [
            medians[value] for value in BLOCK_DIMS],
        "parameters_ns": {
            "startup": startup_ns,
            "per_active_aiv_worker": per_worker_ns,
        },
        "model_mape": model_mape,
        "maximum_model_ape": maximum_model_ape,
        "maximum_allowed_model_mape": MAXIMUM_MODEL_MAPE,
        "maximum_allowed_model_ape": MAXIMUM_MODEL_APE,
        "model_ape_by_active_aiv_workers": {
            str(key): value for key, value in fit_apes.items()
        },
        "relative_span_by_active_aiv_workers": {
            str(key): value for key, value in spans.items()
        },
        "maximum_relative_span": maximum_span,
        "relative_half_median_delta_by_active_aiv_workers": {
            str(key): value for key, value in half_median_deltas.items()
        },
        "maximum_relative_half_median_delta": maximum_half_median_delta,
        "maximum_allowed_relative_half_median_delta": (
            MAXIMUM_HALF_MEDIAN_DELTA),
        "trials": TRIALS,
        "clock": {
            "source": "GetSystemCycle",
            "frequency_hz": 50_000_000,
            "tick_ns": SYSTEM_COUNTER_TICK_NS,
        },
        "mechanism": "one AIV SyncAll with no arithmetic or tensor traffic",
        "composition": "one source-proven global AIV barrier before FIA main tasks",
        "source_commit": source_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": mechanism_publishable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw, args.source_commit, args.source_snapshot_sha256)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["mechanism_publishable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
