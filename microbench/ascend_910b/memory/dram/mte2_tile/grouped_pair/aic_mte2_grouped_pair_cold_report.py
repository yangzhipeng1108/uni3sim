#!/usr/bin/env python3
"""Audit the pre-registered 24-AIC cold operand-pair count campaign."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


PAIR_COUNTS = (512, 1024, 2048)
ACTIVE_AICS = 24
RUN_INDEX_COUNT = 3
TICK_NS = 20.0
MAX_WITHIN_COUNT_RELATIVE_SPAN = 0.10
MAX_COUNT_MEDIAN_RELATIVE_SPAN = 0.10
MAX_PROGRAM_SPAN_RESIDUAL = 0.10
RAW_FIELDS = (
    "run_id", "run_index", "order_position", "process_pid",
    "a_residency", "b_residency", "a_rows", "b_columns",
    "a_tile_bytes", "b_tile_bytes", "a_source_tiles", "b_source_tiles",
    "drain_interval", "block_dim", "iterations", "short_cycles",
    "short_wall_ns", "long_cycles", "long_wall_ns", "tick_ns",
    "cycles_per_pair", "ns_per_pair", "bandwidth_gbs", "validation",
)
CAMPAIGN_GEOMETRIES = {
    "cold-pair-count": (128, 24),
    "cold-m16-pair-count": (16, 23),
}


def _relative_span(values):
    center = statistics.median(values)
    return (max(values) - min(values)) / center


def _ordered_pair_counts(run_index):
    rotation = run_index % len(PAIR_COUNTS)
    counts = PAIR_COUNTS[rotation:] + PAIR_COUNTS[:rotation]
    return counts if run_index % 2 == 0 else tuple(reversed(counts))


def _load_run(path, expected_run_index, a_rows, active_aics):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        raw_rows = list(reader)
    expected_counts = _ordered_pair_counts(expected_run_index)
    if len(raw_rows) != len(expected_counts):
        raise ValueError(f"{path} must contain exactly three cold cells")
    rows = []
    for position, (raw, pair_count) in enumerate(
            zip(raw_rows, expected_counts)):
        context = f"{path} row {position + 2}"
        if (int(raw["run_index"]) != expected_run_index
                or int(raw["order_position"]) != position
                or int(raw["process_pid"]) <= 0
                or raw["a_residency"] != "cold"
                or raw["b_residency"] != "cold"
                or raw["validation"] != "pass"):
            raise ValueError(f"{context} has invalid identity")
        source_tiles = 4 * active_aics * pair_count
        pair_bytes = (a_rows * 64 + 64 * 256) * 2
        expected_integers = {
            "a_rows": a_rows,
            "b_columns": 256,
            "a_tile_bytes": a_rows * 64 * 2,
            "b_tile_bytes": 64 * 256 * 2,
            "a_source_tiles": source_tiles,
            "b_source_tiles": source_tiles,
            "drain_interval": 4,
            "block_dim": active_aics,
            "iterations": pair_count,
        }
        observed_integers = {
            name: int(raw[name]) for name in expected_integers
        }
        if observed_integers != expected_integers:
            raise ValueError(f"{context} changed the frozen mechanism cell")
        short_cycles = int(raw["short_cycles"])
        short_wall_ns = int(raw["short_wall_ns"])
        long_cycles = int(raw["long_cycles"])
        long_wall_ns = int(raw["long_wall_ns"])
        if (short_cycles <= 0 or short_wall_ns <= 0
                or long_cycles <= short_cycles
                or long_wall_ns <= short_wall_ns):
            raise ValueError(f"{context} has invalid measured intervals")
        tick_ns = float(raw["tick_ns"])
        cycles_per_pair = float(raw["cycles_per_pair"])
        ns_per_pair = float(raw["ns_per_pair"])
        bandwidth_gbs = float(raw["bandwidth_gbs"])
        expected_cycles_per_pair = (
            (long_cycles - short_cycles) / (pair_count * 3))
        expected_ns_per_pair = expected_cycles_per_pair * TICK_NS
        expected_bandwidth = pair_bytes / expected_ns_per_pair
        derived_values = (
            (tick_ns, TICK_NS),
            (cycles_per_pair, expected_cycles_per_pair),
            (ns_per_pair, expected_ns_per_pair),
            (bandwidth_gbs, expected_bandwidth),
        )
        if any(not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
               for actual, expected in derived_values):
            raise ValueError(f"{context} has inconsistent derivation")
        short_span_ns = short_cycles * TICK_NS
        steady_span_ns = pair_count * ns_per_pair
        rows.append({
            "run_id": raw["run_id"],
            "run_index": expected_run_index,
            "pair_count": pair_count,
            "ns_per_pair": ns_per_pair,
            "program_span_residual": abs(
                short_span_ns - steady_span_ns) / short_span_ns,
        })
    return rows


def build_report(paths, campaign="cold-pair-count"):
    if len(paths) != RUN_INDEX_COUNT:
        raise ValueError("the campaign requires exactly three raw runs")
    if campaign not in CAMPAIGN_GEOMETRIES:
        raise ValueError(f"unsupported cold pair campaign: {campaign}")
    a_rows, active_aics = CAMPAIGN_GEOMETRIES[campaign]
    pair_bytes = (a_rows * 64 + 64 * 256) * 2
    rows = []
    for run_index, path in enumerate(paths):
        rows.extend(_load_run(path, run_index, a_rows, active_aics))
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != RUN_INDEX_COUNT:
        raise ValueError("run IDs must be unique per independent run")

    medians = {
        str(pair_count): statistics.median([
            row["ns_per_pair"] for row in rows
            if row["pair_count"] == pair_count
        ])
        for pair_count in PAIR_COUNTS
    }
    within_count_spans = {
        str(pair_count): _relative_span([
            row["ns_per_pair"] for row in rows
            if row["pair_count"] == pair_count
        ])
        for pair_count in PAIR_COUNTS
    }
    count_median_span = _relative_span(list(medians.values()))
    maximum_program_span_residual = max(
        row["program_span_residual"] for row in rows)
    stability_gate = max(within_count_spans.values()) <= (
        MAX_WITHIN_COUNT_RELATIVE_SPAN)
    constant_rate_gate = count_median_span <= MAX_COUNT_MEDIAN_RELATIVE_SPAN
    program_span_gate = maximum_program_span_residual <= (
        MAX_PROGRAM_SPAN_RESIDUAL)
    return {
        "campaign": campaign,
        "hardware_id": "ascend_910b",
        "physical_device_id": 4,
        "pair_geometry": {
            "a": [a_rows, 64],
            "b": [64, 256],
            "pair_bytes": pair_bytes,
            "drain_interval": 4,
            "active_aic_groups": active_aics,
            "measured_pair_counts_per_worker": list(PAIR_COUNTS),
            "source_tiles_per_operand": {
                str(count): 4 * active_aics * count for count in PAIR_COUNTS
            },
        },
        "medians_ns_per_pair": medians,
        "published_cold_pair_service_ns": statistics.median([
            row["ns_per_pair"] for row in rows]),
        "within_count_relative_spans": within_count_spans,
        "count_median_relative_span": count_median_span,
        "maximum_program_span_residual": maximum_program_span_residual,
        "stability_gate": stability_gate,
        "constant_rate_gate": constant_rate_gate,
        "program_span_gate": program_span_gate,
        "mechanism_gate_passed": (
            stability_gate and constant_rate_gate and program_span_gate),
        "production_or_evaluation_duration_consumed": False,
        "publication_scope": (
            f"{active_aics}-AIC cold A1/B1 {a_rows}x64+64x256 pair streams "
            "with a four-pair drain and 512--2048 pairs per worker"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs=3, required=True, type=Path)
    parser.add_argument(
        "--campaign", choices=tuple(CAMPAIGN_GEOMETRIES),
        default="cold-pair-count")
    args = parser.parse_args()
    print(json.dumps(
        build_report(args.raw, campaign=args.campaign),
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
