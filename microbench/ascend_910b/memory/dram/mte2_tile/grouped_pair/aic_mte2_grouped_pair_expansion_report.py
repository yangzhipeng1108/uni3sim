#!/usr/bin/env python3
"""Audit the pre-registered 22/24-AIC, 512-pair expansion campaign."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


GROUP_COUNTS = (22, 24)
ITERATIONS = 512
RUN_INDEX_COUNT = 3
PAIR_BYTES = (128 * 64 + 64 * 256) * 2
TICK_NS = 20.0
MAX_WITHIN_DEVICE_RELATIVE_SPAN = 0.20
MAX_CROSS_DEVICE_RELATIVE_MEDIAN_DELTA = 0.10
MAX_GROUP_SCALING_APE = 0.10
MAX_PROGRAM_SPAN_RESIDUAL = 0.10
RAW_FIELDS = (
    "run_id", "run_index", "order_position", "process_pid",
    "a_residency", "b_residency", "a_rows", "b_columns",
    "a_tile_bytes", "b_tile_bytes", "a_source_tiles", "b_source_tiles",
    "drain_interval", "block_dim", "iterations", "short_cycles",
    "short_wall_ns", "long_cycles", "long_wall_ns", "tick_ns",
    "cycles_per_pair", "ns_per_pair", "bandwidth_gbs", "validation",
)


def _relative_span(values):
    center = statistics.median(values)
    return (max(values) - min(values)) / center


def _ordered_group_counts(run_index):
    return GROUP_COUNTS if run_index % 2 == 0 else tuple(reversed(GROUP_COUNTS))


def _load_run(path, expected_run_index, device_label):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        raw_rows = list(reader)
    expected_groups = _ordered_group_counts(expected_run_index)
    if len(raw_rows) != len(expected_groups):
        raise ValueError(f"{path} must contain exactly two expansion cells")
    rows = []
    for position, (raw, group_count) in enumerate(zip(raw_rows, expected_groups)):
        context = f"{path} row {position + 2}"
        if (int(raw["run_index"]) != expected_run_index
                or int(raw["order_position"]) != position
                or int(raw["process_pid"]) <= 0
                or raw["a_residency"] != "resident"
                or raw["b_residency"] != "resident"
                or raw["validation"] != "pass"):
            raise ValueError(f"{context} has invalid identity")
        expected_integers = {
            "a_rows": 128,
            "b_columns": 256,
            "a_tile_bytes": 128 * 64 * 2,
            "b_tile_bytes": 64 * 256 * 2,
            "a_source_tiles": group_count,
            "b_source_tiles": group_count,
            "drain_interval": 4,
            "block_dim": group_count,
            "iterations": ITERATIONS,
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
            (long_cycles - short_cycles) / (ITERATIONS * 3))
        expected_ns_per_pair = expected_cycles_per_pair * TICK_NS
        expected_bandwidth = PAIR_BYTES / expected_ns_per_pair
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
        steady_span_ns = ITERATIONS * ns_per_pair
        program_span_residual = abs(
            short_span_ns - steady_span_ns) / short_span_ns
        rows.append({
            "device": device_label,
            "run_id": raw["run_id"],
            "run_index": expected_run_index,
            "group_count": group_count,
            "ns_per_pair": ns_per_pair,
            "short_span_ns": short_span_ns,
            "program_span_residual": program_span_residual,
        })
    return rows


def build_report(device6_paths, device7_paths):
    if len(device6_paths) != RUN_INDEX_COUNT or len(device7_paths) != RUN_INDEX_COUNT:
        raise ValueError("each device requires exactly three independent raw runs")
    rows = []
    for device_label, paths in (("device6", device6_paths),
                                ("device7", device7_paths)):
        for run_index, path in enumerate(paths):
            rows.extend(_load_run(path, run_index, device_label))
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != 2 * RUN_INDEX_COUNT:
        raise ValueError("run IDs must be unique per device run")

    medians = {
        device: {
            str(group_count): statistics.median([
                row["ns_per_pair"] for row in rows
                if row["device"] == device
                and row["group_count"] == group_count
            ])
            for group_count in GROUP_COUNTS
        }
        for device in ("device6", "device7")
    }
    within_device_spans = {
        device: {
            str(group_count): _relative_span([
                row["ns_per_pair"] for row in rows
                if row["device"] == device
                and row["group_count"] == group_count
            ])
            for group_count in GROUP_COUNTS
        }
        for device in ("device6", "device7")
    }
    cross_device_deltas = {
        str(group_count): abs(
            medians["device6"][str(group_count)]
            - medians["device7"][str(group_count)]) / statistics.median((
                medians["device6"][str(group_count)],
                medians["device7"][str(group_count)],
            ))
        for group_count in GROUP_COUNTS
    }
    group_scaling_ape = {
        device: abs(
            medians[device]["22"]
            - medians[device]["24"] * 22 / 24
        ) / medians[device]["22"]
        for device in ("device6", "device7")
    }
    maximum_program_span_residual = max(
        row["program_span_residual"] for row in rows)
    stability_gate = max(
        span for device in within_device_spans.values()
        for span in device.values()) <= MAX_WITHIN_DEVICE_RELATIVE_SPAN
    cross_device_gate = max(cross_device_deltas.values()) <= (
        MAX_CROSS_DEVICE_RELATIVE_MEDIAN_DELTA)
    group_scaling_gate = max(group_scaling_ape.values()) <= MAX_GROUP_SCALING_APE
    program_span_gate = maximum_program_span_residual <= (
        MAX_PROGRAM_SPAN_RESIDUAL)
    return {
        "campaign": "group-count-22x512",
        "hardware_id": "ascend_910b",
        "devices": [6, 7],
        "pair_geometry": {
            "a": [128, 64],
            "b": [64, 256],
            "pair_bytes": PAIR_BYTES,
            "drain_interval": 4,
            "iterations_per_worker": ITERATIONS,
        },
        "medians_ns_per_pair": medians,
        "within_device_relative_spans": within_device_spans,
        "cross_device_relative_median_deltas": cross_device_deltas,
        "group_scaling_ape": group_scaling_ape,
        "maximum_program_span_residual": maximum_program_span_residual,
        "stability_gate": stability_gate,
        "cross_device_gate": cross_device_gate,
        "group_scaling_gate": group_scaling_gate,
        "program_span_gate": program_span_gate,
        "mechanism_gate_passed": (
            stability_gate and cross_device_gate and group_scaling_gate
            and program_span_gate),
        "production_or_evaluation_duration_consumed": False,
        "publication_scope": (
            "exact resident A1/B1 128x64+64x256 pair stream at 22 AIC; "
            "24 AIC is the pre-registered group-count control"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device6-raw", nargs=3, required=True, type=Path)
    parser.add_argument("--device7-raw", nargs=3, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(
        build_report(args.device6_raw, args.device7_raw),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
