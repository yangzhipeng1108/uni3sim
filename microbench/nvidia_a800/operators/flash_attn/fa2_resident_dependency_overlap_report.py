#!/usr/bin/env python3
"""Validate the two-resident-CTA FA2 dependency overlap capture."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SOURCE_PATH = Path(__file__).with_name(
    "fa2_resident_dependency_overlap.cu")
RUNNER_PATH = Path(__file__).with_name(
    "fa2_resident_dependency_overlap_runner.py")
MODES = {
    0: "timer_baseline",
    1: "fp32_max_quad4",
    3: "cta_barrier",
    4: "fp32_quad4_sum",
    5: "fp32_scale_exp2_sum",
    6: "integer_column_predicate",
    7: "fp32_to_bf16_conversion",
}
GRIDS = (108, 216)
GEOMETRIES = ((2, 32), (4, 16), (4, 32))
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
    if manifest["schema"] != (
            "a800_fa2_resident_dependency_overlap_capture_v1"):
        raise ValueError("manifest schema changed")
    if manifest["hardware_id"] != "nvidia_a800":
        raise ValueError("capture hardware changed")
    if manifest["source_sha256"] != _sha256(SOURCE_PATH):
        raise ValueError("capture source differs from current source")
    if manifest["raw_sha256"] != _sha256(raw_path):
        raise ValueError("capture raw SHA256 mismatch")
    if manifest["production_or_evaluation_duration_consumed"]:
        raise ValueError("capture consumed a production duration")
    if tuple(manifest["grids"]) != GRIDS:
        raise ValueError("capture grid sweep changed")
    if tuple(tuple(value) for value in manifest["geometries"]) != GEOMETRIES:
        raise ValueError("capture geometry sweep changed")
    if (manifest["repeats"] != REPEATS
            or manifest["samples_per_cell"] != SAMPLES
            or manifest["trials"] != TRIALS):
        raise ValueError("capture repetition contract changed")
    return manifest


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_fields = (
            "trial", "mode_name", "sample", "mode", "grid", "repeats",
            "lane_local_rows_per_thread",
            "score_values_per_lane_local_row", "resident_ctas_per_sm",
            "dynamic_shared_bytes", "duration_ns", "median_cta_cycles",
            "sm_clock_before_mhz", "sm_clock_after_mhz",
            "memory_clock_before_mhz", "memory_clock_after_mhz",
        )
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError("raw fields changed")
        rows = []
        identities = set()
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": int(raw["trial"]),
                "mode_name": raw["mode_name"],
                "sample": int(raw["sample"]),
                "mode": int(raw["mode"]),
                "grid": int(raw["grid"]),
                "repeats": int(raw["repeats"]),
                "lane_local_rows_per_thread": int(
                    raw["lane_local_rows_per_thread"]),
                "score_values_per_lane_local_row": int(
                    raw["score_values_per_lane_local_row"]),
                "resident_ctas_per_sm": int(raw["resident_ctas_per_sm"]),
                "dynamic_shared_bytes": int(raw["dynamic_shared_bytes"]),
                "duration_ns": float(raw["duration_ns"]),
                "median_cta_cycles": int(raw["median_cta_cycles"]),
                "sm_clock_before_mhz": int(raw["sm_clock_before_mhz"]),
                "sm_clock_after_mhz": int(raw["sm_clock_after_mhz"]),
                "memory_clock_before_mhz": int(
                    raw["memory_clock_before_mhz"]),
                "memory_clock_after_mhz": int(
                    raw["memory_clock_after_mhz"]),
            }
            if row["mode"] not in MODES:
                raise ValueError(f"line {ordinal} has an unknown mode")
            if row["mode_name"] != MODES[row["mode"]]:
                raise ValueError(f"line {ordinal} mode name changed")
            if (row["grid"] not in GRIDS
                    or row["repeats"] != REPEATS
                    or (row["lane_local_rows_per_thread"],
                        row["score_values_per_lane_local_row"])
                    not in GEOMETRIES):
                raise ValueError(f"line {ordinal} geometry changed")
            if (row["resident_ctas_per_sm"] != 2
                    or row["dynamic_shared_bytes"] != 77824):
                raise ValueError(f"line {ordinal} residency changed")
            if (row["sm_clock_before_mhz"]
                    != row["sm_clock_after_mhz"]
                    or row["memory_clock_before_mhz"]
                    != row["memory_clock_after_mhz"]):
                raise ValueError(f"line {ordinal} clocks changed")
            if (not math.isfinite(row["duration_ns"])
                    or row["duration_ns"] <= 0
                    or row["median_cta_cycles"] <= 0):
                raise ValueError(f"line {ordinal} timing is invalid")
            identity = (
                row["trial"], row["mode"], row["grid"],
                row["lane_local_rows_per_thread"],
                row["score_values_per_lane_local_row"], row["sample"],
            )
            if identity in identities:
                raise ValueError(f"duplicate row identity: {identity}")
            identities.add(identity)
            rows.append(row)
    expected = {
        (trial, mode, grid, lane_rows, score_values, sample)
        for trial in range(TRIALS)
        for mode in MODES
        for grid in GRIDS
        for lane_rows, score_values in GEOMETRIES
        for sample in range(SAMPLES)
    }
    if identities != expected:
        raise ValueError("raw capture matrix is incomplete")
    return tuple(rows)


def _cell(rows, trial, mode, grid, geometry):
    selected = tuple(
        row for row in rows
        if row["trial"] == trial
        and row["mode"] == mode
        and row["grid"] == grid
        and (row["lane_local_rows_per_thread"],
             row["score_values_per_lane_local_row"]) == geometry)
    if len(selected) != SAMPLES:
        raise ValueError("capture cell has the wrong sample count")
    durations = tuple(row["duration_ns"] for row in selected)
    cycles = tuple(
        row["median_cta_cycles"] / REPEATS for row in selected)
    return {
        "duration_ns_median": statistics.median(durations),
        "duration_ns_p10": statistics.quantiles(
            durations, n=10, method="inclusive")[0],
        "duration_ns_p90": statistics.quantiles(
            durations, n=10, method="inclusive")[8],
        "cta_cycles_per_repeat_median": statistics.median(cycles),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    manifest = _read_manifest(arguments.manifest, arguments.input)
    rows = _read_rows(arguments.input)
    cells = []
    ratios = []
    for geometry in GEOMETRIES:
        for mode, mode_name in MODES.items():
            for trial in range(TRIALS):
                one = _cell(rows, trial, mode, 108, geometry)
                two = _cell(rows, trial, mode, 216, geometry)
                span_ratio = (
                    two["duration_ns_median"]
                    / one["duration_ns_median"])
                cta_ratio = (
                    two["cta_cycles_per_repeat_median"]
                    / one["cta_cycles_per_repeat_median"])
                ratios.append({
                    "trial": trial,
                    "mode": mode,
                    "mode_name": mode_name,
                    "lane_local_rows_per_thread": geometry[0],
                    "score_values_per_lane_local_row": geometry[1],
                    "grid_span_ratio_216_to_108": span_ratio,
                    "cta_cycle_ratio_216_to_108": cta_ratio,
                })
                cells.extend((
                    {"trial": trial, "mode": mode,
                     "mode_name": mode_name,
                     "lane_local_rows_per_thread": geometry[0],
                     "score_values_per_lane_local_row": geometry[1],
                     "grid": 108, **one},
                    {"trial": trial, "mode": mode,
                     "mode_name": mode_name,
                     "lane_local_rows_per_thread": geometry[0],
                     "score_values_per_lane_local_row": geometry[1],
                     "grid": 216, **two},
                ))

    contention_modes = tuple(mode for mode in MODES if mode != 0)
    contention = tuple(
        row for row in ratios if row["mode"] in contention_modes)
    contention_bounds = []
    contention_spreads = []
    for geometry in GEOMETRIES:
        for mode in contention_modes:
            selected = tuple(
                row["grid_span_ratio_216_to_108"]
                for row in contention
                if row["mode"] == mode
                and (row["lane_local_rows_per_thread"],
                     row["score_values_per_lane_local_row"]) == geometry)
            upper_bound = max(selected)
            spread = (upper_bound - min(selected)) / min(selected)
            contention_spreads.append(spread)
            contention_bounds.append({
                "mode": mode,
                "mode_name": MODES[mode],
                "lane_local_rows_per_thread": geometry[0],
                "score_values_per_lane_local_row": geometry[1],
                "maximum_grid_span_ratio_216_to_108": upper_bound,
                "per_cta_demand_factor_upper_bound": upper_bound / 2.0,
                "cross_trial_ratio_spread": spread,
            })

    overlap_modes = (3, 4)
    overlap = tuple(row for row in ratios if row["mode"] in overlap_modes)
    overlap_span_max = max(
        row["grid_span_ratio_216_to_108"] for row in overlap)
    overlap_cta_max = max(
        row["cta_cycle_ratio_216_to_108"] for row in overlap)
    overlap_spreads = []
    for geometry in GEOMETRIES:
        for mode in overlap_modes:
            values = tuple(
                row["grid_span_ratio_216_to_108"] for row in overlap
                if row["mode"] == mode
                and (row["lane_local_rows_per_thread"],
                     row["score_values_per_lane_local_row"]) == geometry)
            overlap_spreads.append((max(values) - min(values)) / min(values))
    gates = {
        "all_cells_have_constant_clocks": True,
        "barrier_and_quad_grid_span_ratio_at_most_1_10": (
            overlap_span_max <= 1.10),
        "barrier_and_quad_cta_cycle_ratio_at_most_1_10": (
            overlap_cta_max <= 1.10),
        "barrier_and_quad_cross_trial_ratio_spread_at_most_0_10": (
            max(overlap_spreads) <= 0.10),
        "all_contention_grid_span_ratios_finite_and_at_most_2_00": all(
            math.isfinite(row["grid_span_ratio_216_to_108"])
            and 0.0 < row["grid_span_ratio_216_to_108"] <= 2.0
            for row in contention),
        "all_contention_cross_trial_ratio_spreads_at_most_0_10": (
            max(contention_spreads) <= 0.10),
    }
    status = "pass" if all(gates.values()) else "fail"
    result = {
        "schema": "a800_fa2_resident_dependency_overlap_report_v1",
        "status": status,
        "hardware_id": "nvidia_a800",
        "mechanism": "two_resident_cta_scalar_dependency_overlap",
        "production_or_evaluation_duration_consumed": False,
        "source_sha256": manifest["source_sha256"],
        "runner_sha256": _sha256(RUNNER_PATH),
        "raw_sha256": manifest["raw_sha256"],
        "baseline_commit": manifest["baseline_commit"],
        "device": manifest["device"],
        "cells": cells,
        "ratios": ratios,
        "gates": gates,
        "maximum_overlap_grid_span_ratio": overlap_span_max,
        "maximum_overlap_cta_cycle_ratio": overlap_cta_max,
        "maximum_overlap_cross_trial_ratio_spread": max(overlap_spreads),
        "contention_span_upper_bounds": contention_bounds,
        "allowed_conclusion": (
            "For the isolated source-shaped SM80 FA2 fragments with exactly "
            "two resident 128-thread CTAs per SM, each measured dependency's "
            "two-CTA aggregate span can be bounded by the maximum 216/108 "
            "grid-span ratio across three trials. Dividing that ratio by two "
            "gives the corresponding conservative per-CTA demand factor. "
            "CTA barriers and FP32 quad reductions additionally satisfy the "
            "frozen near-complete-overlap gates."),
        "excluded_conclusions": [
            "a complete production FlashAttention duration",
            "resident counts other than two",
            "source geometries outside the captured matrix",
        ],
    }
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "status": status,
        "maximum_overlap_grid_span_ratio": overlap_span_max,
        "maximum_overlap_cta_cycle_ratio": overlap_cta_max,
        "maximum_overlap_cross_trial_ratio_spread": max(overlap_spreads),
        "output": str(arguments.output.resolve()),
    }, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
