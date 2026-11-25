#!/usr/bin/env python3
"""Audit raw Ascend 910B MTE2 tile-service measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from mte2_tile_runner import CELLS


REQUIRED_FIELDS = (
    "run_id",
    "trial",
    "order_position",
    "process_pid",
    "tile_bytes",
    "footprint_bytes",
    "mode",
    "n_iters",
    "short_cycles",
    "short_wall_ns",
    "long_cycles",
    "long_wall_ns",
    "tick_ns",
    "per_tile_cycles",
    "per_tile_ns",
    "bandwidth_gbs",
    "validation",
)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each MTE2 cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _load_rows(paths):
    rows = []
    identities = set()
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            for raw in reader:
                if raw["validation"] != "pass":
                    raise ValueError(f"{path} contains failed validation")
                integers = {
                    field: int(raw[field])
                    for field in (
                        "trial", "order_position", "process_pid",
                        "tile_bytes", "footprint_bytes", "n_iters",
                        "short_cycles", "short_wall_ns", "long_cycles",
                        "long_wall_ns",
                    )
                }
                if integers["trial"] <= 0 or integers["process_pid"] <= 0:
                    raise ValueError(f"{path} contains invalid process identity")
                if (integers["order_position"] < 0
                        or integers["order_position"] >= len(CELLS)):
                    raise ValueError(f"{path} contains invalid order position")
                if integers["footprint_bytes"] != 512 * 1024 * 1024:
                    raise ValueError(f"{path} changed the frozen footprint")
                if raw["mode"] not in (
                        "hard_event_completion", "pipeline",
                        "dynamic_params_completion"):
                    raise ValueError(f"{path} contains an invalid MTE2 mode")
                if (integers["tile_bytes"] <= 0
                        or integers["footprint_bytes"]
                        % integers["tile_bytes"] != 0):
                    raise ValueError(f"{path} contains invalid tile geometry")
                minimum_iterations = (
                    integers["footprint_bytes"] // integers["tile_bytes"])
                if integers["n_iters"] < minimum_iterations:
                    raise ValueError(f"{path} has incomplete footprint coverage")
                if (integers["short_cycles"] <= 0
                        or integers["short_wall_ns"] <= 0
                        or integers["long_cycles"]
                        <= integers["short_cycles"]
                        or integers["long_wall_ns"]
                        <= integers["short_wall_ns"]):
                    raise ValueError(f"{path} contains invalid raw intervals")
                identity = (
                    raw["run_id"], integers["trial"],
                    integers["order_position"])
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                row = {
                    "run_id": raw["run_id"],
                    "trial": integers["trial"],
                    "order_position": integers["order_position"],
                    "tile_bytes": integers["tile_bytes"],
                    "mode": raw["mode"],
                    "tick_ns": float(raw["tick_ns"]),
                    "per_tile_cycles": float(raw["per_tile_cycles"]),
                    "per_tile_ns": float(raw["per_tile_ns"]),
                    "bandwidth_gbs": float(raw["bandwidth_gbs"]),
                }
                if any(
                        not math.isfinite(row[name]) or row[name] <= 0
                        for name in (
                            "tick_ns", "per_tile_cycles", "per_tile_ns",
                            "bandwidth_gbs")):
                    raise ValueError(f"{path} contains non-positive measurement")
                delta_cycles = (
                    integers["long_cycles"] - integers["short_cycles"])
                delta_wall_ns = (
                    integers["long_wall_ns"] - integers["short_wall_ns"])
                expected_tick_ns = delta_wall_ns / delta_cycles
                expected_per_tile_cycles = delta_cycles / (
                    integers["n_iters"] * 3)
                expected_per_tile_ns = (
                    expected_tick_ns * expected_per_tile_cycles)
                expected_bandwidth_gbs = (
                    integers["tile_bytes"] / expected_per_tile_ns)
                checks = (
                    (row["tick_ns"], expected_tick_ns),
                    (row["per_tile_cycles"], expected_per_tile_cycles),
                    (row["per_tile_ns"], expected_per_tile_ns),
                    (row["bandwidth_gbs"], expected_bandwidth_gbs),
                )
                if any(
                        not math.isclose(
                            observed, expected, rel_tol=1e-8, abs_tol=1e-8)
                        for observed, expected in checks):
                    raise ValueError(f"{path} contains inconsistent derived values")
                rows.append(row)
    if not rows:
        raise ValueError("no MTE2 rows were provided")
    return rows


def _audit(rows):
    mode_names = {
        0: "hard_event_completion",
        1: "pipeline",
        2: "dynamic_params_completion",
    }
    expected_cells = {
        (tile_bytes, mode_names[mode])
        for tile_bytes, mode in CELLS
    }
    by_cell = {cell: [] for cell in expected_cells}
    by_trial = {}
    positions_by_trial = {}
    for row in rows:
        cell = (row["tile_bytes"], row["mode"])
        if cell not in by_cell:
            raise ValueError(f"unexpected MTE2 cell {cell}")
        by_cell[cell].append(row)
        trial_key = (row["run_id"], row["trial"])
        if trial_key not in by_trial:
            by_trial[trial_key] = {}
            positions_by_trial[trial_key] = set()
        if cell in by_trial[trial_key]:
            raise ValueError(f"trial {trial_key} duplicates cell {cell}")
        if row["order_position"] in positions_by_trial[trial_key]:
            raise ValueError(
                f"trial {trial_key} duplicates order position "
                f"{row['order_position']}")
        by_trial[trial_key][cell] = row
        positions_by_trial[trial_key].add(row["order_position"])
    counts = {len(cell_rows) for cell_rows in by_cell.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("MTE2 cells do not have equal non-zero sample counts")
    for trial_key, trial_rows in by_trial.items():
        if set(trial_rows) != expected_cells:
            raise ValueError(f"trial {trial_key} is incomplete")
        if positions_by_trial[trial_key] != set(range(len(CELLS))):
            raise ValueError(f"trial {trial_key} has incomplete order positions")

    summaries = {
        cell: _quartiles(
            [row["bandwidth_gbs"] for row in cell_rows])
        for cell, cell_rows in by_cell.items()
    }
    pipeline_lower = max(
        summary[0] for cell, summary in summaries.items()
        if cell[1] == "pipeline")
    pipeline_upper = min(
        summary[2] for cell, summary in summaries.items()
        if cell[1] == "pipeline")
    pipeline_common_capacity = pipeline_lower <= pipeline_upper
    pipeline_values = [
        row["bandwidth_gbs"]
        for cell, cell_rows in by_cell.items()
        if cell[1] == "pipeline"
        for row in cell_rows
    ]

    residual_by_tile = {
        tile_bytes: []
        for tile_bytes in sorted({cell[0] for cell in expected_cells})
    }
    for trial_rows in by_trial.values():
        for tile_bytes in residual_by_tile:
            residual_by_tile[tile_bytes].append(
                trial_rows[(tile_bytes, "hard_event_completion")][
                    "per_tile_ns"]
                - trial_rows[(tile_bytes, "pipeline")]["per_tile_ns"])
    residual_summary = {
        tile_bytes: _quartiles(values)
        for tile_bytes, values in residual_by_tile.items()
    }
    residual_lower = max(summary[0] for summary in residual_summary.values())
    residual_upper = min(summary[2] for summary in residual_summary.values())
    fixed_residual = residual_lower <= residual_upper and residual_lower > 0
    residual_values = [
        value for values in residual_by_tile.values() for value in values
    ]
    descriptor_residual_by_tile = {
        tile_bytes: []
        for tile_bytes in sorted({cell[0] for cell in expected_cells})
    }
    for trial_rows in by_trial.values():
        for tile_bytes in descriptor_residual_by_tile:
            descriptor_residual_by_tile[tile_bytes].append(
                trial_rows[(tile_bytes, "dynamic_params_completion")][
                    "per_tile_ns"]
                - trial_rows[(tile_bytes, "hard_event_completion")][
                    "per_tile_ns"])
    descriptor_residual_summary = {
        tile_bytes: _quartiles(values)
        for tile_bytes, values in descriptor_residual_by_tile.items()
    }
    descriptor_residual_lower = max(
        summary[0] for summary in descriptor_residual_summary.values())
    descriptor_residual_upper = min(
        summary[2] for summary in descriptor_residual_summary.values())
    fixed_descriptor_residual = (
        descriptor_residual_lower <= descriptor_residual_upper
        and descriptor_residual_lower > 0)
    descriptor_residual_values = [
        value
        for values in descriptor_residual_by_tile.values()
        for value in values
    ]
    return {
        "samples_per_cell": counts.pop(),
        "summaries": summaries,
        "pipeline_common_capacity": pipeline_common_capacity,
        "pipeline_capacity_candidate": (
            statistics.median(pipeline_values)
            if pipeline_common_capacity else None),
        "pipeline_intersection": (pipeline_lower, pipeline_upper),
        "residual_summary": residual_summary,
        "fixed_residual": fixed_residual,
        "fixed_residual_candidate": (
            statistics.median(residual_values) if fixed_residual else None),
        "residual_intersection": (residual_lower, residual_upper),
        "descriptor_residual_summary": descriptor_residual_summary,
        "fixed_descriptor_residual": fixed_descriptor_residual,
        "fixed_descriptor_residual_candidate": (
            statistics.median(descriptor_residual_values)
            if fixed_descriptor_residual else None),
        "descriptor_residual_intersection": (
            descriptor_residual_lower, descriptor_residual_upper),
    }


def _print_report(audit):
    print("# Ascend 910B MTE2 mechanism audit")
    print(f"samples_per_cell={audit['samples_per_cell']}")
    print("tile_bytes,mode,q1_gbs,median_gbs,q3_gbs")
    for cell in sorted(audit["summaries"]):
        q1, median, q3 = audit["summaries"][cell]
        print(f"{cell[0]},{cell[1]},{q1:.9f},{median:.9f},{q3:.9f}")
    print(
        "pipeline_common_capacity_gate="
        f"{'pass' if audit['pipeline_common_capacity'] else 'fail'}")
    lower, upper = audit["pipeline_intersection"]
    print(f"pipeline_iqr_intersection_gbs={lower:.9f},{upper:.9f}")
    capacity = audit["pipeline_capacity_candidate"]
    print(
        f"pipeline_capacity_pooled_median_gbs={capacity:.9f}"
        if capacity is not None
        else "pipeline_capacity_pooled_median_gbs=unavailable")
    print("tile_bytes,serial_minus_pipeline_q1_ns,median_ns,q3_ns")
    for tile_bytes in sorted(audit["residual_summary"]):
        q1, median, q3 = audit["residual_summary"][tile_bytes]
        print(f"{tile_bytes},{q1:.9f},{median:.9f},{q3:.9f}")
    print(
        "fixed_residual_gate="
        f"{'pass' if audit['fixed_residual'] else 'fail'}")
    lower, upper = audit["residual_intersection"]
    print(f"fixed_residual_iqr_intersection_ns={lower:.9f},{upper:.9f}")
    residual = audit["fixed_residual_candidate"]
    print(
        f"fixed_residual_pooled_median_ns={residual:.9f}"
        if residual is not None
        else "fixed_residual_pooled_median_ns=unavailable")
    print(
        "tile_bytes,dynamic_params_minus_simple_completion_q1_ns,"
        "median_ns,q3_ns")
    for tile_bytes in sorted(audit["descriptor_residual_summary"]):
        q1, median, q3 = audit["descriptor_residual_summary"][tile_bytes]
        print(f"{tile_bytes},{q1:.9f},{median:.9f},{q3:.9f}")
    print(
        "fixed_dynamic_params_residual_gate="
        f"{'pass' if audit['fixed_descriptor_residual'] else 'fail'}")
    lower, upper = audit["descriptor_residual_intersection"]
    print(
        "fixed_dynamic_params_residual_iqr_intersection_ns="
        f"{lower:.9f},{upper:.9f}")
    descriptor_residual = audit["fixed_descriptor_residual_candidate"]
    print(
        "fixed_dynamic_params_residual_pooled_median_ns="
        f"{descriptor_residual:.9f}"
        if descriptor_residual is not None
        else "fixed_dynamic_params_residual_pooled_median_ns=unavailable")
    print(
        "production_eligibility=candidate_only_requires_source_and_"
        "provenance_audit_and_does_not_prove_production_buffer_lifecycle")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    _print_report(_audit(_load_rows(args.csv)))


if __name__ == "__main__":
    main()
