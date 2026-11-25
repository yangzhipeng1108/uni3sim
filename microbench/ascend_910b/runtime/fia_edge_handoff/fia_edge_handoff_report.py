#!/usr/bin/env python3
"""Audit three captures of FIA's individual cross-domain handoff edges."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from fia_edge_handoff_runner import (
    BLOCK_DIMS,
    CELLS,
    EDGE_FIELDS,
    EXPECTED_CANN,
    RAW_FIELDS,
    SOURCE_PATHS,
    SYSTEM_COUNTER_TICK_NS,
    TASK_COUNTS,
    TRIALS,
    _ordered_cells,
    _sha256,
    _snapshot_sha256,
)


REPORT_RUN_COUNT = 3
PARAMETER_RUN_COUNT = 2
EDGE_NAMES = ("c1_to_v1", "v1_to_c2", "c2_to_v2")
COMPONENT_NAMES = ("first", "second", "third", "steady")
CROSS_RUN_SPAN_LIMIT = 0.20
TASK_COUNT_SPAN_LIMIT = 0.25
ORDER_P95_LIMIT = 0.20
VALIDATION_MAX_APE_LIMIT = 0.25
RING_LOWER_BOUND_TOLERANCE = 0.25


def _require_mapping(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} has an invalid schema")


def _relative_span(values):
    center = statistics.median(values)
    if center <= 0:
        raise ValueError("relative-span center must be positive")
    return (max(values) - min(values)) / center


def _relative_error(actual, expected):
    if actual <= 0 or expected <= 0:
        raise ValueError("relative-error inputs must be positive")
    return abs(actual - expected) / expected


def _nearest_rank(values, probability):
    if not values or probability <= 0 or probability > 1:
        raise ValueError("percentile inputs are invalid")
    return sorted(values)[math.ceil(probability * len(values)) - 1]


def _load_run(run_dir):
    with (run_dir / "provenance.json").open(encoding="utf-8") as stream:
        provenance = json.load(stream)
    expected_fields = {
        "schema", "run_id", "captured_at_utc", "source_commit",
        "source_snapshot_sha256", "source_files", "binary", "cann",
        "device", "idle_telemetry", "clock", "measurement_scope",
        "production_or_evaluation_duration_consumed", "campaign", "capture",
    }
    _require_mapping(provenance, expected_fields, "provenance")
    if provenance["schema"] != "ascend_910b_fia_edge_handoff_provenance_v2":
        raise ValueError("unsupported FIA edge provenance schema")
    if provenance["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("FIA edge capture consumed forbidden duration")
    if provenance["source_snapshot_sha256"] != _snapshot_sha256():
        raise ValueError("capture source snapshot differs from report source")
    if provenance["source_files"] != {
            label: _sha256(path) for label, path in SOURCE_PATHS.items()}:
        raise ValueError("capture source file identity changed")
    if provenance["cann"]["identity"] != EXPECTED_CANN:
        raise ValueError("capture CANN identity changed")
    if (provenance["device"]["model"] != "910B2"
            or provenance["device"]["logical_device_id"] != 0):
        raise ValueError("capture device identity changed")
    for endpoint in ("pre", "post"):
        if any(value != 0 for value in provenance["idle_telemetry"][endpoint].values()):
            raise ValueError("capture endpoint telemetry is not idle")
    expected_campaign = {
        "block_dims": list(BLOCK_DIMS),
        "task_counts": list(TASK_COUNTS),
        "trials": TRIALS,
        "order": "balanced_forward_reverse",
        "untimed_preamble": "one B1/N4 then one B24/N128 launch",
        "exact_cell_warmups": 1,
        "prefix_depth": 3,
    }
    if provenance["campaign"] != expected_campaign:
        raise ValueError("capture campaign changed")
    raw_path = run_dir / provenance["capture"]["raw"]
    if _sha256(raw_path) != provenance["capture"]["raw_sha256"]:
        raise ValueError("capture raw SHA256 changed")

    rows = []
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("FIA edge raw schema changed")
        for raw in reader:
            trial = int(raw["trial"])
            position = int(raw["order_position"])
            direction, ordered = _ordered_cells(trial - 1)
            profile_id, block_dim, task_count = ordered[position]
            if (raw["run_id"] != provenance["run_id"]
                    or raw["order_direction"] != direction
                    or raw["profile_id"] != profile_id
                    or int(raw["block_dim"]) != block_dim
                    or int(raw["task_count"]) != task_count
                    or raw["validation"] != "ok"):
                raise ValueError("raw row identity changed")
            totals = {
                "c1_to_v1": int(raw["c1v1_total_cycles"]),
                "v1_to_c2": int(raw["v1c2_total_cycles"]),
                "c2_to_v2": int(raw["c2v2_total_cycles"]),
            }
            components = {}
            for short, edge in (
                    ("c1v1", "c1_to_v1"),
                    ("v1c2", "v1_to_c2"),
                    ("c2v2", "c2_to_v2")):
                prefix = tuple(
                    int(raw[f"{short}_prefix{depth}_cycles"])
                    for depth in (1, 2, 3))
                increments = (
                    prefix[0], prefix[1] - prefix[0],
                    prefix[2] - prefix[1],
                    (totals[edge] - prefix[2]) / (task_count - 3),
                )
                if any(value <= 0 for value in increments):
                    raise ValueError("edge completion component is non-positive")
                components[edge] = {
                    name: value * SYSTEM_COUNTER_TICK_NS
                    for name, value in zip(COMPONENT_NAMES, increments)}
            rows.append({
                "run_id": provenance["run_id"],
                "trial": trial,
                "direction": direction,
                "profile_id": profile_id,
                "block_dim": block_dim,
                "task_count": task_count,
                "components_ns": components,
                "lane_skew": max(
                    int(raw["c1v1_lane_skew_ppm"]),
                    int(raw["c2v2_lane_skew_ppm"])) / 1_000_000.0,
                "ring_ns_per_task": (
                    int(raw["ring_elapsed_cycles"])
                    * SYSTEM_COUNTER_TICK_NS / task_count),
            })
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError("FIA edge raw campaign is incomplete")
    return provenance, rows


def _cell_medians(rows):
    result = {}
    for run_id in sorted({row["run_id"] for row in rows}):
        for profile_id, _, _ in CELLS:
            selected = [
                row for row in rows
                if row["run_id"] == run_id
                and row["profile_id"] == profile_id]
            if len(selected) != TRIALS:
                raise ValueError("FIA edge cell lost trials")
            for edge in EDGE_NAMES:
                for component in COMPONENT_NAMES:
                    result[(run_id, profile_id, edge, component)] = (
                        statistics.median(
                            row["components_ns"][edge][component]
                            for row in selected))
            result[(run_id, profile_id, "ring", "steady")] = (
                statistics.median(
                    row["ring_ns_per_task"] for row in selected))
    return result


def _build_report(run_dirs):
    provenances = []
    rows = []
    for run_dir in run_dirs:
        provenance, run_rows = _load_run(run_dir)
        provenances.append(provenance)
        rows.extend(run_rows)
    run_ids = [provenance["run_id"] for provenance in provenances]
    if len(run_ids) != REPORT_RUN_COUNT or len(set(run_ids)) != REPORT_RUN_COUNT:
        raise ValueError("report requires three distinct ordered runs")
    frozen_fields = (
        "source_commit", "source_snapshot_sha256", "source_files",
        "binary", "cann", "device", "clock", "measurement_scope",
        "campaign")
    for field in frozen_fields:
        if any(
                provenance[field] != provenances[0][field]
                for provenance in provenances[1:]):
            raise ValueError(f"independent runs disagree on {field}")

    medians = _cell_medians(rows)
    parameter_runs = run_ids[:PARAMETER_RUN_COUNT]
    validation_run = run_ids[PARAMETER_RUN_COUNT]
    latency_lut = {}
    cross_run_spans = {}
    task_count_spans = {}
    for block_dim in BLOCK_DIMS:
        latency_lut[str(block_dim)] = {}
        cross_run_spans[str(block_dim)] = {}
        task_count_spans[str(block_dim)] = {}
        for edge in EDGE_NAMES:
            latency_lut[str(block_dim)][edge] = {}
            cross_run_spans[str(block_dim)][edge] = {}
            task_count_spans[str(block_dim)][edge] = {}
            for component in COMPONENT_NAMES:
                by_run = [
                    statistics.median(
                        medians[(
                            run_id, f"uniform_b{block_dim}_n{task_count}",
                            edge, component)]
                        for task_count in TASK_COUNTS)
                    for run_id in parameter_runs]
                latency_lut[str(block_dim)][edge][component] = (
                    statistics.median(by_run))
                cross_run_spans[str(block_dim)][edge][component] = (
                    _relative_span(by_run))
                task_count_spans[str(block_dim)][edge][component] = max(
                    _relative_span([
                        medians[(
                            run_id,
                            f"uniform_b{block_dim}_n{task_count}",
                            edge,
                            component)]
                        for task_count in TASK_COUNTS])
                    for run_id in parameter_runs)

    validation_errors = []
    validation_worst = None
    for _, block_dim, task_count in CELLS:
        profile_id = f"uniform_b{block_dim}_n{task_count}"
        for edge in EDGE_NAMES:
            for component in COMPONENT_NAMES:
                expected = latency_lut[str(block_dim)][edge][component]
                actual = medians[(
                    validation_run, profile_id, edge, component)]
                error = _relative_error(actual, expected)
                validation_errors.append(error)
                if validation_worst is None or error > validation_worst["ape"]:
                    validation_worst = {
                        "profile_id": profile_id,
                        "edge": edge,
                        "component": component,
                        "observed_ns": actual,
                        "reference_ns": expected,
                        "ape": error,
                    }

    order_errors = []
    for run_id in run_ids:
        for profile_id, _, _ in CELLS:
            for edge in EDGE_NAMES:
                for component in COMPONENT_NAMES:
                    forward = statistics.median(
                        row["components_ns"][edge][component]
                        for row in rows
                        if row["run_id"] == run_id
                        and row["profile_id"] == profile_id
                        and row["direction"] == "forward")
                    reverse = statistics.median(
                        row["components_ns"][edge][component]
                        for row in rows
                        if row["run_id"] == run_id
                        and row["profile_id"] == profile_id
                        and row["direction"] == "reverse")
                    order_errors.append(_relative_error(forward, reverse))
    lane_skews = [row["lane_skew"] for row in rows]
    ring_lower_bound_failures = 0
    for row in rows:
        critical_edge = max(
            row["components_ns"][edge]["steady"] for edge in EDGE_NAMES)
        if row["ring_ns_per_task"] < (
                1.0 - RING_LOWER_BOUND_TOLERANCE) * critical_edge:
            ring_lower_bound_failures += 1

    gates = {
        "provenance_and_raw_complete": True,
        "all_parameter_cross_run_spans_at_most_20pct": all(
            span <= CROSS_RUN_SPAN_LIMIT
            for by_edge in cross_run_spans.values()
            for by_component in by_edge.values()
            for span in by_component.values()),
        "all_parameter_task_count_spans_at_most_25pct": all(
            span <= TASK_COUNT_SPAN_LIMIT
            for by_edge in task_count_spans.values()
            for by_component in by_edge.values()
            for span in by_component.values()),
        "validation_max_ape_at_most_25pct": (
            max(validation_errors) <= VALIDATION_MAX_APE_LIMIT),
        "balanced_order_p95_at_most_20pct": (
            _nearest_rank(order_errors, 0.95) <= ORDER_P95_LIMIT),
        "critical_edge_not_longer_than_ring": ring_lower_bound_failures == 0,
    }
    publishable = all(gates.values())
    return {
        "schema_version": 2,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "cann_fia_cross_domain_edge_completion_v2",
        "scope": (
            "empty source-ordered C1-to-V1, V1-to-C2, and C2-to-V2 "
            "dependency completion in KERNEL_TYPE_MIX_AIC_1_2"),
        "unit": "ns_per_dependency_completion",
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": publishable,
        "forward_eligible": publishable,
        "model": {
            "latency_lut_ns": latency_lut if publishable else None,
            "components": list(COMPONENT_NAMES),
            "active_aic_groups": list(BLOCK_DIMS),
            "measurement_task_counts": list(TASK_COUNTS),
            "formula": (
                "first + second + third + max(0, task_count-3) * steady"),
            "interpretation": (
                "the first three source-ring dependency completions remain "
                "explicit; only later repetitions use the measured steady "
                "completion service"),
        },
        "gates": gates,
        "thresholds": {
            "parameter_cross_run_relative_span": CROSS_RUN_SPAN_LIMIT,
            "parameter_task_count_relative_span": TASK_COUNT_SPAN_LIMIT,
            "validation_maximum_absolute_percentage_error": (
                VALIDATION_MAX_APE_LIMIT),
            "balanced_order_p95_relative_error": ORDER_P95_LIMIT,
            "ring_lower_bound_relative_tolerance": (
                RING_LOWER_BOUND_TOLERANCE),
        },
        "diagnostics": {
            "parameter_cross_run_spans": cross_run_spans,
            "parameter_task_count_spans": task_count_spans,
            "validation_maximum_absolute_percentage_error": max(validation_errors),
            "validation_worst": validation_worst,
            "balanced_order_median_relative_error": statistics.median(order_errors),
            "balanced_order_p95_relative_error": _nearest_rank(order_errors, 0.95),
            "balanced_order_maximum_relative_error": max(order_errors),
            "c1_c2_aiv_lane_skew_median": statistics.median(lane_skews),
            "c1_c2_aiv_lane_skew_p95": _nearest_rank(lane_skews, 0.95),
            "c1_c2_aiv_lane_skew_maximum": max(lane_skews),
            "ring_lower_bound_failure_count": ring_lower_bound_failures,
        },
        "provenance": {
            "parameter_run_ids": parameter_runs,
            "validation_run_id": validation_run,
            "source_commit": provenances[0]["source_commit"],
            "source_snapshot_sha256": provenances[0]["source_snapshot_sha256"],
            "binary_sha256": provenances[0]["binary"]["sha256"],
            "raw_sha256": [
                provenance["capture"]["raw_sha256"]
                for provenance in provenances],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs=REPORT_RUN_COUNT, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"report output already exists: {args.output}")
    report = _build_report(args.run_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
