#!/usr/bin/env python3
"""Audit raw MTE3 CSVs using predeclared, evaluation-independent gates."""

import argparse
import csv
from pathlib import Path
import math
import statistics

from mte3_tile_runner import CELLS, TILE_BYTES


AFFINE_VALIDATION_MAPE_LIMIT_PCT = 10.0
AFFINE_VALIDATION_MAX_APE_LIMIT_PCT = 20.0
AFFINE_COEFFICIENT_RELATIVE_SPAN_LIMIT_PCT = 15.0


REQUIRED_FIELDS = (
    "run_id",
    "trial",
    "order_position",
    "process_pid",
    "tile_bytes",
    "footprint_bytes",
    "mode",
    "source_buffers",
    "barrier_window",
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
        raise ValueError("each MTE3 cell requires at least four samples")
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
                        "tile_bytes", "footprint_bytes", "source_buffers",
                        "barrier_window", "n_iters", "short_cycles", "short_wall_ns",
                        "long_cycles", "long_wall_ns",
                    )
                }
                if integers["trial"] <= 0 or integers["process_pid"] <= 0:
                    raise ValueError(f"{path} contains invalid process identity")
                if integers["order_position"] < 0 \
                        or integers["order_position"] >= len(CELLS):
                    raise ValueError(f"{path} contains invalid order position")
                if integers["footprint_bytes"] != 512 * 1024 * 1024:
                    raise ValueError(f"{path} changed the frozen footprint")
                valid_completion = (
                    raw["mode"] == "hard_event_completion"
                    and integers["barrier_window"] == 1
                )
                valid_pipeline = (
                    raw["mode"] == "pipeline"
                    and integers["barrier_window"] == 0
                )
                valid_dynamic_completion = (
                    raw["mode"] == "dynamic_params_completion"
                    and integers["barrier_window"] == 1
                )
                if (not valid_completion and not valid_pipeline
                        and not valid_dynamic_completion) \
                        or integers["source_buffers"] != 1:
                    raise ValueError(f"{path} contains invalid mode geometry")
                if integers["tile_bytes"] <= 0 \
                        or integers["footprint_bytes"] % integers["tile_bytes"] != 0:
                    raise ValueError(f"{path} contains invalid tile geometry")
                minimum_iterations = (
                    integers["footprint_bytes"] // integers["tile_bytes"]
                )
                divisor = max(
                    integers["source_buffers"], integers["barrier_window"], 1
                )
                if integers["n_iters"] < minimum_iterations \
                        or integers["n_iters"] % divisor != 0:
                    raise ValueError(f"{path} contains invalid iteration coverage")
                if integers["short_cycles"] <= 0 \
                        or integers["short_wall_ns"] <= 0 \
                        or integers["long_cycles"] <= integers["short_cycles"] \
                        or integers["long_wall_ns"] <= integers["short_wall_ns"]:
                    raise ValueError(f"{path} contains invalid raw intervals")
                identity = (
                    raw["run_id"], integers["trial"],
                    integers["order_position"],
                )
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                row = {
                    "run_id": raw["run_id"],
                    "trial": integers["trial"],
                    "order_position": integers["order_position"],
                    "tile_bytes": integers["tile_bytes"],
                    "mode": raw["mode"],
                    "source_buffers": integers["source_buffers"],
                    "barrier_window": integers["barrier_window"],
                    "tick_ns": float(raw["tick_ns"]),
                    "per_tile_cycles": float(raw["per_tile_cycles"]),
                    "per_tile_ns": float(raw["per_tile_ns"]),
                    "bandwidth_gbs": float(raw["bandwidth_gbs"]),
                }
                if row["tick_ns"] <= 0 or row["per_tile_cycles"] <= 0 \
                        or row["per_tile_ns"] <= 0 \
                        or row["bandwidth_gbs"] <= 0:
                    raise ValueError(f"{path} contains non-positive measurement")
                delta_cycles = (
                    integers["long_cycles"] - integers["short_cycles"]
                )
                delta_wall_ns = (
                    integers["long_wall_ns"] - integers["short_wall_ns"]
                )
                expected_tick_ns = delta_wall_ns / delta_cycles
                expected_per_tile_cycles = delta_cycles / (
                    integers["n_iters"] * 3
                )
                expected_per_tile_ns = (
                    expected_tick_ns * expected_per_tile_cycles
                )
                expected_bandwidth_gbs = (
                    integers["tile_bytes"] / expected_per_tile_ns
                )
                checks = (
                    (row["tick_ns"], expected_tick_ns),
                    (row["per_tile_cycles"], expected_per_tile_cycles),
                    (row["per_tile_ns"], expected_per_tile_ns),
                    (row["bandwidth_gbs"], expected_bandwidth_gbs),
                )
                if any(
                    not math.isclose(observed, expected, rel_tol=1e-8, abs_tol=1e-8)
                    for observed, expected in checks
                ):
                    raise ValueError(f"{path} contains inconsistent derived values")
                rows.append(row)
    if not rows:
        raise ValueError("no MTE3 rows were provided")
    return rows


def _fit_affine(rows):
    if len(rows) < 2:
        raise ValueError("MTE3 affine fit requires at least two rows")
    mean_bytes = statistics.fmean(row["tile_bytes"] for row in rows)
    mean_time = statistics.fmean(row["per_tile_ns"] for row in rows)
    denominator = math.fsum(
        (row["tile_bytes"] - mean_bytes) ** 2 for row in rows)
    if denominator <= 0:
        raise ValueError("MTE3 affine fit requires more than one tile size")
    ns_per_byte = math.fsum(
        (row["tile_bytes"] - mean_bytes)
        * (row["per_tile_ns"] - mean_time)
        for row in rows) / denominator
    startup_ns = mean_time - ns_per_byte * mean_bytes
    if startup_ns <= 0 or ns_per_byte <= 0:
        raise ValueError("MTE3 affine coefficients must be positive")
    return startup_ns, ns_per_byte


def _affine_errors(rows, coefficients):
    startup_ns, ns_per_byte = coefficients
    errors = tuple(
        abs(startup_ns + ns_per_byte * row["tile_bytes"]
            - row["per_tile_ns"]) / row["per_tile_ns"] * 100.0
        for row in rows)
    if not errors:
        raise ValueError("MTE3 affine validation requires rows")
    return statistics.fmean(errors), max(errors)


def _relative_span(values):
    center = statistics.fmean(values)
    if center <= 0:
        raise ValueError("MTE3 affine coefficient center must be positive")
    return (max(values) - min(values)) / center * 100.0


def _audit(rows):
    mode_names = {
        0: "hard_event_completion",
        1: "pipeline",
        2: "dynamic_params_completion",
    }
    expected_cells = {
        (
            tile_bytes,
            mode_names[mode],
            source_buffers,
            barrier_window,
        )
        for tile_bytes, mode, source_buffers, barrier_window in CELLS
    }
    by_cell = {cell: [] for cell in expected_cells}
    by_trial = {}
    positions_by_trial = {}
    for row in rows:
        cell = (
            row["tile_bytes"], row["mode"], row["source_buffers"],
            row["barrier_window"],
        )
        if cell not in by_cell:
            raise ValueError(f"unexpected MTE3 cell {cell}")
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
                f"{row['order_position']}"
            )
        by_trial[trial_key][cell] = row
        positions_by_trial[trial_key].add(row["order_position"])
    counts = {len(cell_rows) for cell_rows in by_cell.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("MTE3 cells do not have equal non-zero sample counts")
    for trial_key, trial_rows in by_trial.items():
        if set(trial_rows) != expected_cells:
            raise ValueError(f"trial {trial_key} is incomplete")
        if positions_by_trial[trial_key] != set(range(len(CELLS))):
            raise ValueError(f"trial {trial_key} has incomplete order positions")

    summaries = {}
    for cell, cell_rows in by_cell.items():
        summaries[cell] = _quartiles(
            [row["bandwidth_gbs"] for row in cell_rows]
        )

    pipeline_intersection_lower = max(
        summary[0]
        for cell, summary in summaries.items()
        if cell[1] == "pipeline"
    )
    pipeline_intersection_upper = min(
        summary[2]
        for cell, summary in summaries.items()
        if cell[1] == "pipeline"
    )
    pipeline_common_capacity = (
        pipeline_intersection_lower <= pipeline_intersection_upper
    )
    pipeline_values = [
        row["bandwidth_gbs"]
        for cell, cell_rows in by_cell.items()
        if cell[1] == "pipeline"
        for row in cell_rows
    ]
    pipeline_capacity_candidate = (
        statistics.median(pipeline_values)
        if pipeline_common_capacity else None
    )

    completion_rows = tuple(
        row for row in rows if row["mode"] == "hard_event_completion")
    odd_rows = tuple(row for row in completion_rows if row["trial"] % 2 == 1)
    even_rows = tuple(row for row in completion_rows if row["trial"] % 2 == 0)
    odd_fit = _fit_affine(odd_rows)
    even_fit = _fit_affine(even_rows)
    odd_to_even = _affine_errors(even_rows, odd_fit)
    even_to_odd = _affine_errors(odd_rows, even_fit)
    startup_span_pct = _relative_span((odd_fit[0], even_fit[0]))
    slope_span_pct = _relative_span((odd_fit[1], even_fit[1]))
    affine_gate = (
        odd_to_even[0] <= AFFINE_VALIDATION_MAPE_LIMIT_PCT
        and odd_to_even[1] <= AFFINE_VALIDATION_MAX_APE_LIMIT_PCT
        and even_to_odd[0] <= AFFINE_VALIDATION_MAPE_LIMIT_PCT
        and even_to_odd[1] <= AFFINE_VALIDATION_MAX_APE_LIMIT_PCT
        and startup_span_pct <= AFFINE_COEFFICIENT_RELATIVE_SPAN_LIMIT_PCT
        and slope_span_pct <= AFFINE_COEFFICIENT_RELATIVE_SPAN_LIMIT_PCT
    )
    pooled_fit = _fit_affine(completion_rows) if affine_gate else None

    descriptor_residual_by_tile = {tile_bytes: [] for tile_bytes in TILE_BYTES}
    for trial_rows in by_trial.values():
        for tile_bytes in TILE_BYTES:
            descriptor_residual_by_tile[tile_bytes].append(
                trial_rows[(
                    tile_bytes, "dynamic_params_completion", 1, 1
                )]["per_tile_ns"]
                - trial_rows[(
                    tile_bytes, "hard_event_completion", 1, 1
                )]["per_tile_ns"])
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
    descriptor_residual_values = tuple(
        value
        for values in descriptor_residual_by_tile.values()
        for value in values)

    return {
        "samples_per_cell": counts.pop(),
        "summaries": summaries,
        "pipeline_intersection_lower": pipeline_intersection_lower,
        "pipeline_intersection_upper": pipeline_intersection_upper,
        "pipeline_common_capacity": pipeline_common_capacity,
        "pipeline_capacity_candidate": pipeline_capacity_candidate,
        "odd_fit": odd_fit,
        "even_fit": even_fit,
        "odd_to_even": odd_to_even,
        "even_to_odd": even_to_odd,
        "startup_span_pct": startup_span_pct,
        "slope_span_pct": slope_span_pct,
        "affine_gate": affine_gate,
        "pooled_fit": pooled_fit,
        "descriptor_residual_summary": descriptor_residual_summary,
        "fixed_descriptor_residual": fixed_descriptor_residual,
        "fixed_descriptor_residual_candidate": (
            statistics.median(descriptor_residual_values)
            if fixed_descriptor_residual else None),
        "descriptor_residual_intersection": (
            descriptor_residual_lower, descriptor_residual_upper),
    }


def _print_report(audit):
    print("# Ascend 910B MTE3 mechanism audit")
    print(f"samples_per_cell={audit['samples_per_cell']}")
    print(
        "tile_bytes,mode,source_buffers,barrier_window,"
        "q1_gbs,median_gbs,q3_gbs"
    )
    for cell in sorted(audit["summaries"]):
        q1, median, q3 = audit["summaries"][cell]
        print(
            f"{cell[0]},{cell[1]},{cell[2]},{cell[3]},"
            f"{q1:.9f},{median:.9f},{q3:.9f}"
        )
    print(
        "pipeline_common_capacity_gate="
        f"{'pass' if audit['pipeline_common_capacity'] else 'fail'}"
    )
    print(
        "pipeline_iqr_intersection_gbs="
        f"{audit['pipeline_intersection_lower']:.9f},"
        f"{audit['pipeline_intersection_upper']:.9f}"
    )
    capacity_candidate = audit["pipeline_capacity_candidate"]
    print(
        "pipeline_capacity_pooled_median_gbs="
        f"{capacity_candidate:.9f}"
        if capacity_candidate is not None
        else "pipeline_capacity_pooled_median_gbs=unavailable"
    )
    for name in ("odd_fit", "even_fit"):
        startup_ns, ns_per_byte = audit[name]
        print(
            f"hard_event_{name}_startup_ns={startup_ns:.9f} "
            f"ns_per_byte={ns_per_byte:.12f} "
            f"bandwidth_gbs={1.0 / ns_per_byte:.9f}")
    for name in ("odd_to_even", "even_to_odd"):
        mape, maximum = audit[name]
        print(
            f"hard_event_{name}_mape_pct={mape:.9f} "
            f"max_ape_pct={maximum:.9f}")
    print(
        "hard_event_affine_coefficient_relative_span_pct="
        f"{audit['startup_span_pct']:.9f},{audit['slope_span_pct']:.9f}")
    print(
        "hard_event_affine_gate="
        f"{'pass' if audit['affine_gate'] else 'fail'}")
    pooled_fit = audit["pooled_fit"]
    if pooled_fit is None:
        print("hard_event_affine_candidate=unavailable")
    else:
        startup_ns, ns_per_byte = pooled_fit
        print(
            "hard_event_affine_candidate="
            f"startup_ns={startup_ns:.9f},"
            f"ns_per_byte={ns_per_byte:.12f},"
            f"bandwidth_gbs={1.0 / ns_per_byte:.9f},"
            f"minimum_bytes={min(TILE_BYTES)},"
            f"maximum_bytes={max(TILE_BYTES)}")
    print(
        "tile_bytes,dynamic_params_minus_simple_completion_q1_ns,"
        "median_ns,q3_ns")
    for tile_bytes in TILE_BYTES:
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
        "production_eligibility=typed_candidate_only_for_source_proven_"
        "mte3_mte2_completion_edges_within_measured_byte_range"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    _print_report(_audit(_load_rows(args.csv)))


if __name__ == "__main__":
    main()
