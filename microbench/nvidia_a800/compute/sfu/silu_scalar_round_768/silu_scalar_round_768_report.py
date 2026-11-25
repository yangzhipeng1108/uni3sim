#!/usr/bin/env python3
"""Report scalar-SiLU round evidence without duration fitting or LUTs."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import median


THREADS = 768
ROUNDS = 64
SM_COUNT = 108
GRIDS = (108, 216)
ROTATIONS = 3
TRIALS = 3
MAXIMUM_SPREAD = 0.20
RAW_FIELDS = ("run_rotation", "trial", "order_position", "grid_blocks", "block_threads", "rounds", "resident_cta_limit", "active_sms", "minimum_ctas_per_sm", "maximum_ctas_per_sm", "maximum_sm_span_cycles", "median_cta_cycles", "checksum")


def _spread(values):
    center = median(values)
    if center <= 0:
        raise ValueError("spread values must have positive median")
    return (max(values) - min(values)) / center


def parse_raw(text, rotation):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != RAW_FIELDS:
        raise ValueError("raw schema changed")
    rows = []
    seen = set()
    for row in reader:
        values = {}
        for key, value in row.items():
            if key == "checksum":
                values[key] = float(value)
            elif key in RAW_FIELDS:
                values[key] = int(value)
            else:
                raise ValueError(f"unknown raw field {key!r}")
        identity = (values["trial"], values["grid_blocks"])
        expected_grid = GRIDS[(values["order_position"] + values["trial"] + rotation) % 2]
        expected_min = values["grid_blocks"] // SM_COUNT
        expected_max = math.ceil(values["grid_blocks"] / SM_COUNT)
        if (values["run_rotation"] != rotation or values["trial"] not in range(TRIALS)
                or values["order_position"] not in range(2) or values["grid_blocks"] != expected_grid
                or values["grid_blocks"] not in GRIDS or identity in seen
                or values["block_threads"] != THREADS or values["rounds"] != ROUNDS
                or values["resident_cta_limit"] != 2 or values["active_sms"] != SM_COUNT
                or values["minimum_ctas_per_sm"] != expected_min
                or values["maximum_ctas_per_sm"] != expected_max
                or values["maximum_sm_span_cycles"] <= 0 or values["median_cta_cycles"] <= 0
                or not math.isfinite(values["checksum"])):
            raise ValueError(f"invalid raw record {identity}")
        seen.add(identity)
        rows.append(values)
    if len(rows) != len(GRIDS) * TRIALS:
        raise ValueError("raw row count changed")
    return tuple(rows)


def derive(runs, provenance):
    if set(runs) != set(range(ROTATIONS)):
        raise ValueError("exactly rotations 0, 1, and 2 are required")
    resources = provenance["resources"]
    static = {"probe_production_math_spine_identical": provenance["probe_sass_contract"]["math_spine"] == provenance["production_sass_contract"], "no_local_spill_or_atomic": provenance["probe_sass_contract"]["local_spill_atomic_absent"], "single_recurrent_silu_math_spine": provenance["probe_sass_contract"]["single_recurrent_silu_math_spine"], "exact_768_thread_24_register_1kib_smem_contract": resources == {"registers_per_thread": 24, "static_shared_bytes": 1024, "local_bytes": 0}, "no_production_duration_consumed": provenance["production_or_evaluation_duration_consumed"] is False}
    cells = {}
    assignment = True
    within_spreads = []
    cross_spreads = []
    for grid in GRIDS:
        medians = []
        for rotation in range(ROTATIONS):
            records = [row for row in runs[rotation] if row["grid_blocks"] == grid]
            cta_rounds = [row["median_cta_cycles"] / ROUNDS for row in records]
            spans = [row["maximum_sm_span_cycles"] for row in records]
            within = _spread(cta_rounds)
            within_spreads.append(within)
            medians.append(median(cta_rounds))
            expected = grid // SM_COUNT
            assignment = assignment and all(row["minimum_ctas_per_sm"] == expected and row["maximum_ctas_per_sm"] == expected for row in records)
            cells[f"grid{grid}:rotation{rotation}"] = {"samples": TRIALS, "sm_clock_cycles_per_768_thread_cta_round": median(cta_rounds), "within_run_relative_spread": within}
        cross = _spread(medians)
        cross_spreads.append(cross)
        cells[f"grid{grid}:cross_run"] = {"samples": ROTATIONS, "sm_clock_cycles_per_768_thread_cta_round": median(medians), "cross_run_relative_spread": cross}
    cycles_108 = [row["median_cta_cycles"] / ROUNDS for rotation in range(ROTATIONS) for row in runs[rotation] if row["grid_blocks"] == 108]
    spans_108 = [row["maximum_sm_span_cycles"] / ROUNDS for rotation in range(ROTATIONS) for row in runs[rotation] if row["grid_blocks"] == 108]
    spans_216 = [row["maximum_sm_span_cycles"] / ROUNDS for rotation in range(ROTATIONS) for row in runs[rotation] if row["grid_blocks"] == 216]
    capacity_samples = [2.0 * baseline / loaded for baseline, loaded in zip(spans_108, spans_216)]
    capacity = median(capacity_samples)
    capacity_spread = _spread(capacity_samples)
    gates = {**static, "exact_grid_assignment": assignment, "within_cell_spread_at_most_20_percent": max(within_spreads) <= MAXIMUM_SPREAD, "cross_run_cell_spread_at_most_20_percent": max(cross_spreads) <= MAXIMUM_SPREAD, "overlap_capacity_spread_at_most_20_percent": capacity_spread <= MAXIMUM_SPREAD, "effective_overlap_capacity_within_one_and_two_ctas_per_sm": 1.0 <= capacity <= 2.0}
    failed = sorted(name for name, passed in gates.items() if not passed)
    allowed = not failed
    return {"schema": "nvidia_a800_silu_scalar_round_768_candidate_v1", "hardware_id": "nvidia_a800", "mechanism": "independent_register_only_recurrent_scalar_silu", "block_threads": THREADS, "rounds_per_cta": ROUNDS, "run_count": ROTATIONS, "trials_per_run": TRIALS, "cell_diagnostics": cells, "maximum_observed_relative_spreads": {"within_cell": max(within_spreads), "cross_run_cell": max(cross_spreads), "overlap_capacity": capacity_spread}, "gates": gates, "publication": {"allowed": allowed, "failed_gates": failed, "sm_clock_cycles_per_768_thread_cta_round": median(cycles_108) if allowed else None, "effective_overlap_capacity_ctas_per_sm": capacity if allowed else None, "capacity_unit": "ctas_per_sm"}, "production_or_evaluation_duration_consumed": False, "excluded_conclusions": ["per-grid lookup table", "stage-time addition", "duration fitting or correction", "direct Forward consumption"]}


def failure_report(provenance, reason):
    return {"schema": "nvidia_a800_silu_scalar_round_768_failure_v1", "hardware_id": "nvidia_a800", "mechanism": "independent_register_only_recurrent_scalar_silu", "capture_schema_error": reason, "gates": {"raw_capture_schema_valid": False, "no_rerun_required": True, "no_production_or_evaluation_duration_consumed": provenance["production_or_evaluation_duration_consumed"] is False}, "publication": {"allowed": False, "failed_gates": ["raw_capture_schema_valid"], "sm_clock_cycles_per_768_thread_cta_round": None, "effective_overlap_capacity_ctas_per_sm": None}, "excluded_conclusions": ["candidate publication", "per-grid lookup table", "stage-time addition", "duration fitting or correction", "direct Forward consumption"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-reason")
    args = parser.parse_args()
    capture = args.capture.resolve()
    provenance = json.loads((capture / "provenance.json").read_text())
    if (provenance["schema"] != "nvidia_a800_silu_scalar_round_768_capture_v1"
            or provenance["physical_device"] != 7
            or provenance["production_parameter_published"]):
        raise ValueError("capture provenance contract changed")
    if args.failure_reason is not None:
        result = failure_report(provenance, args.failure_reason)
    else:
        runs = {rotation: parse_raw((capture / "raw" / f"run_{rotation}.csv").read_text(), rotation) for rotation in range(ROTATIONS)}
        result = derive(runs, provenance)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
