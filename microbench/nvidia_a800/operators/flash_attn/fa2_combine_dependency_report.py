#!/usr/bin/env python3
"""Validate and summarize the A800 FA2 split-combine dependency capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = Path(__file__).with_name("fa2_combine_dependency.cu")
RUNNER_PATH = Path(__file__).with_name("fa2_combine_dependency_runner.py")
MODES = {
    0: "timer_baseline",
    1: "fp32_max_allreduce",
    2: "fp32_sum_allreduce",
    3: "output_conversion_baseline",
    4: "fp32_to_bf16_output_conversion",
    5: "slse_shared_store_barrier_transpose_load",
    6: "post_max_expf_local_sum",
    7: "post_sum_logf",
    8: "scale_expf_shared_store_barrier",
    9: "lse_global_load_to_shared_transpose_ready",
}
GRIDS = (1, 32, 108)
ALLREDUCE_WIDTHS = (2, 4, 8, 16, 32)
CONVERSION_GEOMETRIES = ((4, 128), (8, 64))
SOURCE_FRAGMENT_MODES = (5, 6, 7, 8, 9)
SOURCE_FRAGMENT_GEOMETRIES = (
    (8, 64, 2, 2),
    (8, 64, 4, 4),
    (8, 64, 8, 8),
    (4, 128, 2, 2),
    (4, 128, 3, 4),
    (4, 128, 4, 4),
    (4, 128, 5, 8),
    (4, 128, 6, 8),
    (4, 128, 7, 8),
    (4, 128, 8, 8),
    (4, 128, 9, 16),
    (4, 128, 11, 16),
    (4, 128, 16, 16),
    (4, 128, 22, 32),
)
REPEATS = 64
SAMPLES = 50
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
    if manifest["schema"] != "a800_fa2_combine_dependency_capture_v3":
        raise ValueError("manifest schema changed")
    if manifest["source_sha256"] != _sha256(SOURCE_PATH):
        raise ValueError("capture source differs from current source")
    if manifest["raw_sha256"] != _sha256(raw_path):
        raise ValueError("capture raw SHA256 mismatch")
    if manifest["production_or_evaluation_duration_consumed"]:
        raise ValueError("mechanism capture consumed a production duration")
    if tuple(manifest["grids"]) != GRIDS:
        raise ValueError("capture grid sweep changed")
    if tuple(manifest["allreduce_widths"]) != ALLREDUCE_WIDTHS:
        raise ValueError("capture allreduce widths changed")
    conversion_geometries = tuple(
        (entry["block_m"], entry["head_dim"])
        for entry in manifest["conversion_geometries"])
    if conversion_geometries != CONVERSION_GEOMETRIES:
        raise ValueError("capture conversion geometries changed")
    source_fragment_geometries = tuple(
        (entry["block_m"], entry["head_dim"], entry["num_splits"],
         entry["max_splits"])
        for entry in manifest["source_fragment_geometries"])
    if source_fragment_geometries != SOURCE_FRAGMENT_GEOMETRIES:
        raise ValueError("capture source-fragment geometries changed")
    if manifest["repeats"] != REPEATS:
        raise ValueError("capture repeat count changed")
    if manifest["samples_per_cell"] != SAMPLES:
        raise ValueError("capture sample count changed")
    if manifest["trials"] != TRIALS:
        raise ValueError("capture trial count changed")
    return manifest


def _expected_cells():
    cells = set()
    for trial in range(TRIALS):
        for width in ALLREDUCE_WIDTHS:
            for mode in (0, 1, 2):
                for grid in GRIDS:
                    cells.add((
                        trial, mode, grid, width, width, width, 4, 128))
        for block_m, head_dim in CONVERSION_GEOMETRIES:
            for mode in (3, 4):
                for grid in GRIDS:
                    cells.add((
                        trial, mode, grid, 0, 0, 0, block_m, head_dim))
        for block_m, head_dim, num_splits, max_splits in (
                SOURCE_FRAGMENT_GEOMETRIES):
            for mode in (0,) + SOURCE_FRAGMENT_MODES:
                for grid in GRIDS:
                    cells.add((
                        trial, mode, grid, 0, num_splits, max_splits,
                        block_m, head_dim))
    return cells


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = (
            "trial", "mode_name", "sample", "mode", "grid", "repeats",
            "allreduce_width", "num_splits", "max_splits", "block_m",
            "head_dim", "output_values_per_lane", "block", "sm_id",
            "cycles", "checksum", "sm_clock_before_mhz",
            "sm_clock_after_mhz", "memory_clock_before_mhz",
            "memory_clock_after_mhz",
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError("raw fields changed")
        rows = []
        identities = set()
        observed_cells = set()
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": int(raw["trial"]),
                "mode": int(raw["mode"]),
                "grid": int(raw["grid"]),
                "repeats": int(raw["repeats"]),
                "allreduce_width": int(raw["allreduce_width"]),
                "num_splits": int(raw["num_splits"]),
                "max_splits": int(raw["max_splits"]),
                "block_m": int(raw["block_m"]),
                "head_dim": int(raw["head_dim"]),
                "output_values_per_lane": int(
                    raw["output_values_per_lane"]),
                "sample": int(raw["sample"]),
                "block": int(raw["block"]),
                "sm_id": int(raw["sm_id"]),
                "cycles": int(raw["cycles"]),
            }
            if row["mode"] not in MODES:
                raise ValueError(f"line {ordinal} has an unknown mode")
            if raw["mode_name"] != MODES[row["mode"]]:
                raise ValueError(f"line {ordinal} mode name changed")
            if row["repeats"] != REPEATS or row["grid"] not in GRIDS:
                raise ValueError(f"line {ordinal} launch geometry changed")
            if row["output_values_per_lane"] != 4:
                raise ValueError(f"line {ordinal} conversion width changed")
            if row["mode"] in (1, 2) or (
                    row["mode"] == 0 and row["allreduce_width"] > 0):
                if not (
                        row["allreduce_width"] in ALLREDUCE_WIDTHS
                        and row["num_splits"] == row["allreduce_width"]
                        and row["max_splits"] == row["allreduce_width"]
                        and (row["block_m"], row["head_dim"]) == (4, 128)):
                    raise ValueError(
                        f"line {ordinal} allreduce geometry changed")
            elif row["mode"] in (3, 4):
                if not (
                        row["allreduce_width"] == 0
                        and row["num_splits"] == 0
                        and row["max_splits"] == 0
                        and (row["block_m"], row["head_dim"])
                        in CONVERSION_GEOMETRIES):
                    raise ValueError(
                        f"line {ordinal} conversion geometry changed")
            elif row["mode"] in SOURCE_FRAGMENT_MODES or (
                    row["mode"] == 0 and row["allreduce_width"] == 0):
                if (
                        row["allreduce_width"] != 0
                        or (row["block_m"], row["head_dim"],
                            row["num_splits"], row["max_splits"])
                        not in SOURCE_FRAGMENT_GEOMETRIES):
                    raise ValueError(
                        f"line {ordinal} source-fragment geometry changed")
            else:
                raise ValueError(f"line {ordinal} has an invalid mode geometry")
            if not 0 <= row["trial"] < TRIALS:
                raise ValueError(f"line {ordinal} trial is out of range")
            if not 0 <= row["sample"] < SAMPLES:
                raise ValueError(f"line {ordinal} sample is out of range")
            if not 0 <= row["block"] < row["grid"]:
                raise ValueError(f"line {ordinal} block is out of range")
            if not 0 <= row["sm_id"] < 108 or row["cycles"] <= 0:
                raise ValueError(f"line {ordinal} device result is invalid")
            cell = (
                row["trial"], row["mode"], row["grid"],
                row["allreduce_width"], row["num_splits"],
                row["max_splits"], row["block_m"], row["head_dim"])
            identity = cell + (row["sample"], row["block"])
            if identity in identities:
                raise ValueError(f"duplicate row identity: {identity}")
            identities.add(identity)
            observed_cells.add(cell)
            rows.append(row)
    expected_cells = _expected_cells()
    if observed_cells != expected_cells:
        raise ValueError("raw capture cell matrix is incomplete")
    expected_row_count = sum(
        SAMPLES * cell[2] for cell in expected_cells)
    if len(rows) != expected_row_count:
        raise ValueError("raw capture row matrix is incomplete")
    return tuple(rows)


def _cell_medians(rows):
    values_by_identity = {}
    for row in rows:
        identity = (
            row["trial"], row["mode"], row["grid"],
            row["allreduce_width"], row["num_splits"],
            row["max_splits"], row["block_m"], row["head_dim"])
        if identity not in values_by_identity:
            values_by_identity[identity] = []
        values_by_identity[identity].append(row["cycles"] / row["repeats"])
    cells = []
    for identity in sorted(_expected_cells()):
        (trial, mode, grid, width, num_splits, max_splits, block_m,
         head_dim) = identity
        values = values_by_identity[identity]
        if len(values) != SAMPLES * grid:
            raise ValueError("capture cell has the wrong row count")
        cells.append({
            "trial": trial,
            "mode": mode,
            "mode_name": MODES[mode],
            "grid": grid,
            "allreduce_width": width,
            "num_splits": num_splits,
            "max_splits": max_splits,
            "block_m": block_m,
            "head_dim": head_dim,
            "output_values_per_lane": 4,
            "median_cycles_per_repeat": statistics.median(values),
            "p10_cycles_per_repeat": statistics.quantiles(
                values, n=10, method="inclusive")[0],
            "p90_cycles_per_repeat": statistics.quantiles(
                values, n=10, method="inclusive")[8],
        })
    return tuple(cells)


def _corrected(cells, cell, baseline_mode):
    matches = [
        baseline for baseline in cells
        if baseline["trial"] == cell["trial"]
        and baseline["grid"] == cell["grid"]
        and baseline["allreduce_width"] == cell["allreduce_width"]
        and baseline["num_splits"] == cell["num_splits"]
        and baseline["max_splits"] == cell["max_splits"]
        and baseline["block_m"] == cell["block_m"]
        and baseline["head_dim"] == cell["head_dim"]
        and baseline["mode"] == baseline_mode
    ]
    if len(matches) != 1:
        raise ValueError("mechanism cell has no unique matched baseline")
    value = (
        cell["median_cycles_per_repeat"]
        - matches[0]["median_cycles_per_repeat"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "mechanism cycles must be finite and positive: "
            f"cell={cell}, baseline={matches[0]}, corrected={value}")
    return value


def _parameter(cells, *, mode, width, num_splits, max_splits, block_m,
               head_dim, baseline_mode):
    selected = [
        cell for cell in cells
        if cell["mode"] == mode
        and cell["allreduce_width"] == width
        and cell["num_splits"] == num_splits
        and cell["max_splits"] == max_splits
        and cell["block_m"] == block_m
        and cell["head_dim"] == head_dim
    ]
    training = [
        _corrected(cells, cell, baseline_mode)
        for cell in selected if cell["trial"] in (0, 1)
    ]
    value = statistics.median(training)
    validation = [
        abs(value - observed) / observed * 100.0
        for observed in (
            _corrected(cells, cell, baseline_mode)
            for cell in selected if cell["trial"] == 2)
    ]
    return {
        "mechanism": MODES[mode],
        "allreduce_width": width if mode in (1, 2) else None,
        "block_m": (
            block_m if mode == 4 or mode in SOURCE_FRAGMENT_MODES else None),
        "head_dim": (
            head_dim if mode == 4 or mode in SOURCE_FRAGMENT_MODES else None),
        "num_splits": num_splits if mode in SOURCE_FRAGMENT_MODES else None,
        "max_splits": max_splits if mode in SOURCE_FRAGMENT_MODES else None,
        "output_values_per_lane": 4 if mode == 4 else None,
        "cycles_per_128_thread_cta_stage": value,
        "training_cells": len(training),
        "validation_cells": len(validation),
        "validation_mape_pct": statistics.fmean(validation),
        "validation_max_ape_pct": max(validation),
    }


def _parameters(cells):
    reduction_entries = []
    for width in ALLREDUCE_WIDTHS:
        reduction_entries.append(_parameter(
            cells, mode=1, width=width, num_splits=width,
            max_splits=width, block_m=4, head_dim=128,
            baseline_mode=0))
        reduction_entries.append(_parameter(
            cells, mode=2, width=width, num_splits=width,
            max_splits=width, block_m=4, head_dim=128,
            baseline_mode=0))
    conversion_entries = []
    for block_m, head_dim in CONVERSION_GEOMETRIES:
        conversion_entries.append(_parameter(
            cells, mode=4, width=0, num_splits=0, max_splits=0,
            block_m=block_m, head_dim=head_dim,
            baseline_mode=3))
    source_fragment_entries = []
    for block_m, head_dim, num_splits, max_splits in (
            SOURCE_FRAGMENT_GEOMETRIES):
        for mode in SOURCE_FRAGMENT_MODES:
            source_fragment_entries.append(_parameter(
                cells, mode=mode, width=0, num_splits=num_splits,
                max_splits=max_splits, block_m=block_m, head_dim=head_dim,
                baseline_mode=(5 if mode == 9 else 0)))
    return (
        tuple(reduction_entries), tuple(conversion_entries),
        tuple(source_fragment_entries))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = _read_manifest(arguments.manifest, arguments.input)
    rows = _read_rows(arguments.input)
    cells = _cell_medians(rows)
    (reduction_entries, conversion_entries,
     source_fragment_entries) = _parameters(cells)
    parameters = (
        reduction_entries + conversion_entries + source_fragment_entries)
    result = {
        "schema": "a800_fa2_combine_dependency_report_v3",
        "status": "candidate",
        "hardware_id": "nvidia_a800",
        "mechanism": "fa2_split_combine_dependency_stages",
        "production_or_evaluation_duration_consumed": False,
        "source_sha256": manifest["source_sha256"],
        "runner_sha256": _sha256(RUNNER_PATH),
        "raw_sha256": manifest["raw_sha256"],
        "baseline_commit": manifest["baseline_commit"],
        "upstream_flash_attn_commit": manifest[
            "upstream_flash_attn_commit"],
        "upstream_combine_source_sha256": manifest[
            "upstream_combine_source_sha256"],
        "upstream_allreduce_source_sha256": manifest[
            "upstream_allreduce_source_sha256"],
        "timing_domain": manifest["timing_domain"],
        "source_geometry": {
            "threads_per_cta": 128,
            "allreduce_widths": list(ALLREDUCE_WIDTHS),
            "conversion_geometries": [
                {"block_m": block_m, "head_dim": head_dim,
                 "output_values_per_lane": 4}
                for block_m, head_dim in CONVERSION_GEOMETRIES
            ],
            "source_fragment_geometries": [
                {"block_m": block_m, "head_dim": head_dim,
                 "num_splits": num_splits, "max_splits": max_splits}
                for block_m, head_dim, num_splits, max_splits
                in SOURCE_FRAGMENT_GEOMETRIES
            ],
        },
        "dataset_geometry_mapping": [
            {"block_m": block_m, "head_dim": head_dim,
             "num_splits": num_splits, "max_splits": max_splits,
             "allreduce_width": min(128 // block_m, max_splits),
             "output_values_per_lane": 4}
            for block_m, head_dim, num_splits, max_splits
            in SOURCE_FRAGMENT_GEOMETRIES
        ],
        "derivation": (
            "trials 0-1 take the median clock64 CTA-envelope cycles per "
            "source fragment across grids 1/32/108 after subtracting the "
            "matching timer or conversion-loop baseline; trial 2 is frozen "
            "validation"),
        "reduction_entries": list(reduction_entries),
        "conversion_entries": list(conversion_entries),
        "source_fragment_entries": list(source_fragment_entries),
        "parameters": list(parameters),
        "cells": list(cells),
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result["parameters"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
