#!/usr/bin/env python3
"""Validate and summarize the A800 FA2 scalar-dependency capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_scalar_dependency.cu")
RUNNER_PATH = Path(__file__).with_name("fa2_scalar_dependency_runner.py")
MODES = {
    0: "timer_baseline",
    1: "fp32_max_quad4",
    2: "fp32_local_sum",
    3: "cta_barrier",
    4: "fp32_quad4_sum",
    5: "fp32_scale_exp2_sum",
    6: "integer_column_predicate",
    7: "fp32_to_bf16_conversion",
}
GRIDS = (1, 32, 108)
GEOMETRIES = ((2, 32), (2, 64), (4, 16), (4, 32))
REPEATS = 64
SAMPLES = 100
TRIALS = 3


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path, raw_path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a mapping")
    if manifest["schema"] != "a800_fa2_scalar_dependency_capture_v2":
        raise ValueError("manifest schema changed")
    if manifest["source_sha256"] != _sha256(SOURCE_PATH):
        raise ValueError("capture source differs from current source")
    if manifest["raw_sha256"] != _sha256(raw_path):
        raise ValueError("capture raw SHA256 mismatch")
    if manifest["production_or_evaluation_duration_consumed"]:
        raise ValueError("mechanism capture consumed a production duration")
    if tuple(manifest["grids"]) != GRIDS:
        raise ValueError("capture grid sweep changed")
    geometries = tuple(
        (item["lane_local_rows_per_thread"],
         item["score_values_per_lane_local_row"])
        for item in manifest["geometries"])
    if geometries != GEOMETRIES:
        raise ValueError("capture geometry sweep changed")
    if manifest["repeats"] != REPEATS:
        raise ValueError("capture repeat count changed")
    if manifest["samples_per_cell"] != SAMPLES:
        raise ValueError("capture sample count changed")
    if manifest["trials"] != TRIALS:
        raise ValueError("capture trial count changed")
    return manifest


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = (
            "trial", "mode_name", "sample", "mode", "grid", "repeats",
            "lane_local_rows_per_thread",
            "score_values_per_lane_local_row", "block", "sm_id", "cycles",
            "checksum",
            "sm_clock_before_mhz", "sm_clock_after_mhz",
            "memory_clock_before_mhz", "memory_clock_after_mhz",
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("raw fields changed")
        rows = []
        identities = set()
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": int(raw["trial"]),
                "sample": int(raw["sample"]),
                "mode": int(raw["mode"]),
                "grid": int(raw["grid"]),
                "repeats": int(raw["repeats"]),
                "lane_local_rows_per_thread": int(
                    raw["lane_local_rows_per_thread"]),
                "score_values_per_lane_local_row": int(
                    raw["score_values_per_lane_local_row"]),
                "block": int(raw["block"]),
                "sm_id": int(raw["sm_id"]),
                "cycles": int(raw["cycles"]),
                "sm_clock_before_mhz": int(raw["sm_clock_before_mhz"]),
                "sm_clock_after_mhz": int(raw["sm_clock_after_mhz"]),
                "memory_clock_before_mhz": int(
                    raw["memory_clock_before_mhz"]),
                "memory_clock_after_mhz": int(
                    raw["memory_clock_after_mhz"]),
            }
            if row["mode"] not in MODES:
                raise ValueError(f"line {ordinal} has an unknown mode")
            if raw["mode_name"] != MODES[row["mode"]]:
                raise ValueError(f"line {ordinal} mode name changed")
            if row["grid"] not in GRIDS or row["repeats"] != REPEATS:
                raise ValueError(f"line {ordinal} geometry changed")
            geometry = (
                row["lane_local_rows_per_thread"],
                row["score_values_per_lane_local_row"],
            )
            if geometry not in GEOMETRIES:
                raise ValueError(f"line {ordinal} scalar geometry changed")
            if not 0 <= row["trial"] < TRIALS:
                raise ValueError(f"line {ordinal} trial is out of range")
            if not 0 <= row["sample"] < SAMPLES:
                raise ValueError(f"line {ordinal} sample is out of range")
            if not 0 <= row["block"] < row["grid"]:
                raise ValueError(f"line {ordinal} block is out of range")
            if not 0 <= row["sm_id"] < 108 or row["cycles"] <= 0:
                raise ValueError(f"line {ordinal} device result is invalid")
            identity = (
                row["trial"], row["mode"], row["grid"],
                row["lane_local_rows_per_thread"],
                row["score_values_per_lane_local_row"], row["sample"],
                row["block"],
            )
            if identity in identities:
                raise ValueError(f"duplicate row identity: {identity}")
            identities.add(identity)
            rows.append(row)
    expected = {
        (trial, mode, grid, rows, score_values, sample, block)
        for trial in range(TRIALS)
        for mode in MODES
        for grid in GRIDS
        for rows, score_values in GEOMETRIES
        for sample in range(SAMPLES)
        for block in range(grid)
    }
    if identities != expected:
        raise ValueError("raw capture matrix is incomplete")
    return tuple(rows)


def _cell_medians(rows):
    cells = []
    for trial in range(TRIALS):
        for lane_rows, score_values in GEOMETRIES:
          for mode in MODES:
            for grid in GRIDS:
                values = [
                    row["cycles"] / row["repeats"]
                    for row in rows
                    if row["trial"] == trial
                    and row["mode"] == mode
                    and row["grid"] == grid
                    and row["lane_local_rows_per_thread"] == lane_rows
                    and row["score_values_per_lane_local_row"] == score_values
                ]
                if len(values) != SAMPLES * grid:
                    raise ValueError("capture cell has the wrong row count")
                cells.append({
                    "trial": trial,
                    "mode": mode,
                    "mode_name": MODES[mode],
                    "lane_local_rows_per_thread": lane_rows,
                    "score_values_per_lane_local_row": score_values,
                    "grid": grid,
                    "median_cycles_per_repeat": statistics.median(values),
                    "p10_cycles_per_repeat": statistics.quantiles(
                        values, n=10, method="inclusive")[0],
                    "p90_cycles_per_repeat": statistics.quantiles(
                        values, n=10, method="inclusive")[8],
                })
    return tuple(cells)


def _parameters(cells):
    parameters = []
    for lane_rows, score_values in GEOMETRIES:
      for mode in range(1, len(MODES)):
        training = [
            cell for cell in cells
            if cell["trial"] in (0, 1)
            and cell["mode"] == mode
            and cell["lane_local_rows_per_thread"] == lane_rows
            and cell["score_values_per_lane_local_row"] == score_values
        ]
        baseline = {
            (cell["trial"], cell["grid"],
             cell["lane_local_rows_per_thread"],
             cell["score_values_per_lane_local_row"]):
                cell["median_cycles_per_repeat"]
            for cell in cells if cell["mode"] == 0
        }
        corrected = [
            cell["median_cycles_per_repeat"]
            - baseline[(
                cell["trial"], cell["grid"], lane_rows, score_values)]
            for cell in training
        ]
        if any(value <= 0 or not math.isfinite(value) for value in corrected):
            raise ValueError("mechanism cycles must be finite and positive")
        value = statistics.median(corrected)
        validation = []
        for cell in cells:
            if (cell["trial"] != 2 or cell["mode"] != mode
                    or cell["lane_local_rows_per_thread"] != lane_rows
                    or cell["score_values_per_lane_local_row"]
                    != score_values):
                continue
            observed = (
                cell["median_cycles_per_repeat"]
                - baseline[(
                    cell["trial"], cell["grid"], lane_rows, score_values)])
            validation.append(abs(value - observed) / observed * 100.0)
        parameters.append({
            "mode": mode,
            "mode_name": MODES[mode],
            "lane_local_rows_per_thread": lane_rows,
            "score_values_per_lane_local_row": score_values,
            "cycles_per_128_thread_cta_stage": value,
            "training_cells": len(corrected),
            "validation_cells": len(validation),
            "validation_mape_pct": statistics.fmean(validation),
            "validation_max_ape_pct": max(validation),
        })
    return tuple(parameters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = _read_manifest(arguments.manifest, arguments.input)
    rows = _read_rows(arguments.input)
    cells = _cell_medians(rows)
    parameters = _parameters(cells)
    result = {
        "schema": "a800_fa2_scalar_dependency_report_v2",
        "status": "candidate",
        "hardware_id": "nvidia_a800",
        "mechanism": "fa2_register_reduction_and_cta_barrier",
        "production_or_evaluation_duration_consumed": False,
        "source_sha256": manifest["source_sha256"],
        "runner_sha256": _sha256(RUNNER_PATH),
        "raw_sha256": manifest["raw_sha256"],
        "baseline_commit": manifest["baseline_commit"],
        "timing_domain": manifest["timing_domain"],
        "source_geometry": {
            "threads_per_cta": 128,
            "warps_per_cta": 4,
            "measured_fragment_geometries": [
                {
                    "lane_local_rows_per_thread": rows,
                    "score_values_per_lane_local_row": score_values,
                }
                for rows, score_values in GEOMETRIES
            ],
            "quad_reduction_width": 4,
        },
        "derivation": (
            "training trials 0-1 take the median clock64 CTA-envelope cycles "
            "per repeated primitive across grids 1/32/108 after subtracting "
            "the matching timer baseline; trial 2 is frozen validation"),
        "parameters": list(parameters),
        "cells": list(cells),
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result["parameters"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
