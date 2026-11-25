#!/usr/bin/env python3
"""Validate staged A800 SiLU-adjacent mechanism captures without fitting."""

import argparse
import csv
import io
import json
from pathlib import Path
import re
from statistics import median


EXPECTED_SM_COUNT = 108
EXPECTED_THREADS = 1024
EXPECTED_ROUNDS = 512
EXPECTED_RESIDENT_CTAS = 2
EXPECTED_RUNS = 3
EXPECTED_TRIALS = 3
GRIDS = (108, 216, 324, 1024)
STAGES = ("cached_two_u16_load", "register_ex2_rcp", "u16_store")
CELLS = tuple((stage, grid) for stage in STAGES for grid in GRIDS)
RAW_FIELDS = (
    "run_rotation", "trial", "order_position", "stage", "grid_blocks",
    "block_threads", "rounds", "resident_cta_limit", "active_sms",
    "minimum_ctas_per_sm", "maximum_ctas_per_sm", "maximum_sm_span_ns",
    "median_block_ns", "stage_units", "sm_cta_counts", "sm_spans_ns",
    "checksum",
)
FUNCTION_MARKER = re.compile(r"^\s*Function\s*:\s*(\S+)\s*$", re.MULTILINE)
SYMBOL_PATTERN = re.compile(r"Function(?:\s*:\s*|\s+)(\S+?):?$")
REGISTER_PATTERN = re.compile(r"REG:(\d+)")
SHARED_PATTERN = re.compile(r"SHARED:(\d+)")
LOCAL_PATTERN = re.compile(r"LOCAL:(\d+)")
STAGE_TOKENS = {
    "Stage0": "cached_two_u16_load",
    "Stage1": "register_ex2_rcp",
    "Stage2": "u16_store",
}
EXPECTED_REGISTERS = {
    "cached_two_u16_load": 24,
    "register_ex2_rcp": 24,
    "u16_store": 23,
}
MAXIMUM_RELATIVE_SPREAD = 0.20


def _relative_spread(values):
    center = median(values)
    if center <= 0:
        raise ValueError("relative-spread inputs must have a positive median")
    return (max(values) - min(values)) / center


def parse_resource_usage(text):
    resources = {}
    current = None
    for line in text.splitlines():
        match = SYMBOL_PATTERN.search(line)
        if match:
            current = match.group(1)
            continue
        if current is None or "silu_cta_overlap_stage" not in current:
            continue
        registers = REGISTER_PATTERN.search(line)
        shared = SHARED_PATTERN.search(line)
        local = LOCAL_PATTERN.search(line)
        if registers and shared and local:
            tokens = [token for token in STAGE_TOKENS if token in current]
            if len(tokens) != 1:
                raise ValueError(f"stage specialization changed: {current}")
            stage = STAGE_TOKENS[tokens[0]]
            if stage in resources:
                raise ValueError(f"duplicate resource record for {stage}")
            resources[stage] = {
                "registers_per_thread": int(registers.group(1)),
                "static_shared_bytes": int(shared.group(1)),
                "local_bytes": int(local.group(1)),
            }
    if set(resources) != set(STAGES):
        raise ValueError(
            f"expected exactly three staged specializations, got {sorted(resources)}")
    for stage, values in resources.items():
        expected = {
            "registers_per_thread": EXPECTED_REGISTERS[stage],
            "static_shared_bytes": 0,
            "local_bytes": 0,
        }
        if values != expected:
            raise ValueError(
                f"24-register lifetime/resource contract changed for {stage}: {values}")
    return resources


def _function_sections(sass):
    matches = tuple(FUNCTION_MARKER.finditer(sass))
    sections = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(sass)
        sections[match.group(1)] = sass[match.start():end]
    return sections


def validate_sass(sass):
    sections = _function_sections(sass)
    staged = {}
    for token, stage in STAGE_TOKENS.items():
        matches = tuple(
            section for symbol, section in sections.items()
            if "silu_cta_overlap_stage" in symbol and token in symbol)
        if len(matches) != 1:
            raise ValueError(
                f"expected one SASS specialization for {stage}, got {len(matches)}")
        staged[stage] = matches[0]
    for stage, section in staged.items():
        forbidden = [opcode for opcode in ("LDL", "STL", "ATOM") if opcode in section]
        if forbidden:
            raise ValueError(f"{stage} contains forbidden opcodes: {forbidden}")
        if "BAR.SYNC" not in section:
            raise ValueError(f"{stage} is missing BAR.SYNC timing boundaries")
    load = staged["cached_two_u16_load"]
    if load.count("LDG.E.U16") != 2:
        raise ValueError("cached load stage must contain exactly two LDG.E.U16 instructions")
    if "MUFU.EX2" in load or "MUFU.RCP" in load or "STG.E.U16" in load:
        raise ValueError("cached load stage contains another measured stage")
    sfu = staged["register_ex2_rcp"]
    if sfu.count("MUFU.EX2") != 1 or sfu.count("MUFU.RCP") != 1:
        raise ValueError("register SFU stage must contain one EX2 and one RCP")
    if "LDG.E.U16" in sfu or "STG.E.U16" in sfu:
        raise ValueError("register SFU stage contains a measured memory stage")
    ex2_index = sfu.index("MUFU.EX2")
    rcp_index = sfu.index("MUFU.RCP")
    if ex2_index >= rcp_index:
        raise ValueError("register SFU dependency order changed")
    store = staged["u16_store"]
    if store.count("STG.E.U16") != 1:
        raise ValueError("store stage must contain exactly one STG.E.U16 instruction")
    if "LDG.E.U16" in store or "MUFU.EX2" in store or "MUFU.RCP" in store:
        raise ValueError("store stage contains another measured stage")
    return {
        "cached_two_u16_load_instruction_count": 2,
        "register_ex2_instruction_count": 1,
        "register_rcp_instruction_count": 1,
        "u16_store_instruction_count": 1,
        "register_ex2_precedes_rcp": True,
        "forbidden_local_or_atomic_opcodes_absent": True,
    }


def parse_raw_text(text, expected_rotation):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != RAW_FIELDS:
        raise ValueError("raw staged-probe header changed")
    rows = list(reader)
    if len(rows) != EXPECTED_TRIALS * len(CELLS):
        raise ValueError("raw staged-probe row count changed")
    normalized = []
    seen = set()
    for row in rows:
        stage = row["stage"]
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        integers = {
            name: int(row[name]) for name in RAW_FIELDS
            if name not in ("stage", "sm_cta_counts", "sm_spans_ns")
        }
        sm_cta_counts = tuple(
            int(value) for value in row["sm_cta_counts"].split(";"))
        sm_spans_ns = tuple(
            int(value) for value in row["sm_spans_ns"].split(";"))
        rotation = integers["run_rotation"]
        trial = integers["trial"]
        position = integers["order_position"]
        grid = integers["grid_blocks"]
        identity = (trial, stage, grid)
        if (rotation != expected_rotation or trial not in range(EXPECTED_TRIALS)
                or position not in range(len(CELLS)) or grid not in GRIDS
                or identity in seen):
            raise ValueError(f"invalid raw staged-probe identity {identity}")
        seen.add(identity)
        expected_index = (position + 5 * trial + 4 * rotation) % len(CELLS)
        if CELLS[expected_index] != (stage, grid):
            raise ValueError(f"cell order changed for {identity}")
        if (integers["block_threads"] != EXPECTED_THREADS
                or integers["rounds"] != EXPECTED_ROUNDS
                or integers["resident_cta_limit"] != EXPECTED_RESIDENT_CTAS
                or integers["active_sms"] != EXPECTED_SM_COUNT):
            raise ValueError(f"launch contract changed for {identity}")
        if (integers["minimum_ctas_per_sm"] <= 0
                or integers["maximum_ctas_per_sm"]
                < integers["minimum_ctas_per_sm"]):
            raise ValueError(f"CTA assignment record is invalid for {identity}")
        if (len(sm_cta_counts) != EXPECTED_SM_COUNT
                or len(sm_spans_ns) != EXPECTED_SM_COUNT
                or any(count < 0 for count in sm_cta_counts)
                or any(span < 0 for span in sm_spans_ns)
                or sum(sm_cta_counts) != grid
                or sum(count > 0 for count in sm_cta_counts)
                != integers["active_sms"]
                or min(sm_cta_counts) != integers["minimum_ctas_per_sm"]
                or max(sm_cta_counts) != integers["maximum_ctas_per_sm"]
                or max(sm_spans_ns) != integers["maximum_sm_span_ns"]
                or any((count > 0) != (span > 0)
                       for count, span in zip(sm_cta_counts, sm_spans_ns))):
            raise ValueError(f"per-SM assignment record is invalid for {identity}")
        expected_units = grid * EXPECTED_THREADS * EXPECTED_ROUNDS
        if integers["stage_units"] != expected_units:
            raise ValueError(f"stage-unit contract changed for {identity}")
        if (integers["maximum_sm_span_ns"] <= 0
                or integers["median_block_ns"] <= 0):
            raise ValueError(f"timing record is invalid for {identity}")
        normalized.append({
            "stage": stage,
            "sm_cta_counts": sm_cta_counts,
            "sm_spans_ns": sm_spans_ns,
            **integers,
        })
    return tuple(normalized)


def _cell_rate(row):
    units_per_cta = EXPECTED_THREADS * EXPECTED_ROUNDS
    per_sm_rates = [
        count * units_per_cta * 1e9 / span
        for count, span in zip(row["sm_cta_counts"], row["sm_spans_ns"])
        if count > 0
    ]
    return median(per_sm_rates)


def derive_candidate(runs, resources, sass_contract):
    if set(runs) != set(range(EXPECTED_RUNS)):
        raise ValueError("exactly run rotations 0, 1, and 2 are required")
    rows = tuple(row for rotation in range(EXPECTED_RUNS) for row in runs[rotation])
    order_balance = {}
    for stage, grid in CELLS:
        positions = [
            row["order_position"] for row in rows
            if row["stage"] == stage and row["grid_blocks"] == grid]
        thirds = tuple(
            sum(1 for position in positions if lower <= position < lower + 4)
            for lower in (0, 4, 8))
        order_balance[f"{stage}:grid{grid}"] = list(thirds)
    order_gate = all(counts == [3, 3, 3] for counts in order_balance.values())

    cell_summaries = {}
    within_run_spreads = []
    assignment_gate = True
    cross_run_spreads = []
    per_run_rates = {}
    for stage, grid in CELLS:
        run_medians = []
        for rotation in range(EXPECTED_RUNS):
            records = [
                row for row in runs[rotation]
                if row["stage"] == stage and row["grid_blocks"] == grid]
            spans = [row["maximum_sm_span_ns"] for row in records]
            rates = [_cell_rate(row) for row in records]
            spread = _relative_spread(spans)
            within_run_spreads.append(spread)
            run_median = median(rates)
            run_medians.append(run_median)
            per_run_rates[(rotation, stage, grid)] = run_median
            assignment_gate = assignment_gate and all(
                sum(row["sm_cta_counts"]) == grid
                and max(row["sm_spans_ns"]) == row["maximum_sm_span_ns"]
                for row in records)
        cross_spread = _relative_spread(run_medians)
        cross_run_spreads.append(cross_spread)
        cell_summaries[f"{stage}:grid{grid}"] = {
            "sample_count": EXPECTED_RUNS * EXPECTED_TRIALS,
            "cross_run_relative_spread": cross_spread,
            "within_run_spread_gate": all(
                _relative_spread([
                    row["maximum_sm_span_ns"] for row in runs[rotation]
                    if row["stage"] == stage
                    and row["grid_blocks"] == grid
                ]) <= MAXIMUM_RELATIVE_SPREAD
                for rotation in range(EXPECTED_RUNS)),
        }

    service_values = {}
    overlap_values = {}
    multi_grid_spreads = []
    overlap_spreads = []
    for stage in STAGES:
        services_by_run = []
        capacities_by_run = []
        for rotation in range(EXPECTED_RUNS):
            multi_grid = [
                per_run_rates[(rotation, stage, grid)]
                for grid in (216, 324, 1024)]
            multi_grid_spreads.append(_relative_spread(multi_grid))
            services_by_run.append(median(multi_grid))
            capacities_by_run.append(
                per_run_rates[(rotation, stage, 216)]
                / per_run_rates[(rotation, stage, 108)])
        service_spread = _relative_spread(services_by_run)
        capacity_spread = _relative_spread(capacities_by_run)
        cross_run_spreads.append(service_spread)
        overlap_spreads.append(capacity_spread)
        capacity = median(capacities_by_run)
        service_values[stage] = median(services_by_run)
        overlap_values[stage] = capacity

    gates = {
        "three_independent_runs": len(runs) == EXPECTED_RUNS,
        "position_balanced_early_middle_late": order_gate,
        "exact_per_sm_grid_assignment_accounted": assignment_gate,
        "within_run_cell_spread_at_most_20_percent": (
            max(within_run_spreads) <= MAXIMUM_RELATIVE_SPREAD),
        "cross_run_cell_and_service_spread_at_most_20_percent": (
            max(cross_run_spreads) <= MAXIMUM_RELATIVE_SPREAD),
        "multi_grid_service_spread_at_most_20_percent": (
            max(multi_grid_spreads) <= MAXIMUM_RELATIVE_SPREAD),
        "overlap_capacity_spread_at_most_20_percent": (
            max(overlap_spreads) <= MAXIMUM_RELATIVE_SPREAD),
        "two_cta_aggregate_service_ratio_positive": all(
            value > 0 for value in overlap_values.values()),
        "all_services_positive": all(value > 0 for value in service_values.values()),
        "static_resource_contract": all(
            values == {
                "registers_per_thread": EXPECTED_REGISTERS[stage],
                "static_shared_bytes": 0,
                "local_bytes": 0,
            }
            for stage, values in resources.items()),
        "production_24_register_residency_class": (
            max(values["registers_per_thread"] for values in resources.values())
            == 24
            and all(values["registers_per_thread"] <= 24
                    for values in resources.values())),
        "static_sass_contract": all(bool(value) for value in sass_contract.values()),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    publication_allowed = not failed
    publication = {
        "allowed": publication_allowed,
        "failed_gates": failed,
        "stage_service_units_per_sm_s": (
            service_values if publication_allowed else {}),
        "two_cta_aggregate_service_ratio": (
            overlap_values if publication_allowed else {}),
    }
    return {
        "schema": "nvidia_a800_silu_cta_overlap_candidate_v3",
        "hardware_id": "nvidia_a800",
        "mechanism": "independent_staged_cta_overlap",
        "grid_cells": list(GRIDS),
        "block_threads": EXPECTED_THREADS,
        "resident_cta_limit_per_sm": EXPECTED_RESIDENT_CTAS,
        "production_register_lifetime_ceiling": 24,
        "run_count": EXPECTED_RUNS,
        "trials_per_run": EXPECTED_TRIALS,
        "order_balance_counts": order_balance,
        "cell_diagnostics": cell_summaries,
        "maximum_observed_relative_spreads": {
            "within_run_cell": max(within_run_spreads),
            "cross_run_cell_or_service": max(cross_run_spreads),
            "multi_grid_service": max(multi_grid_spreads),
            "overlap_capacity": max(overlap_spreads),
        },
        "gates": gates,
        "publication": publication,
        "resources": resources,
        "sass_contract": sass_contract,
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "Only stable per-SM stage-unit service and the measured two-CTA "
            "aggregate-service ratio for the three independent probe stages may "
            "be published."),
        "excluded_conclusions": [
            "a complete SiLU or activation duration",
            "a production-kernel correction factor",
            "a per-grid or per-shape lookup table",
            "direct Forward consumption before an independently reviewed composition",
        ],
    }


def build_report(capture):
    resources = parse_resource_usage(
        (capture / "build/resource_usage.txt").read_text())
    sass_contract = validate_sass(
        (capture / "build/silu_cta_overlap.sass").read_text())
    runs = {
        rotation: parse_raw_text(
            (capture / f"raw/run_{rotation}.csv").read_text(), rotation)
        for rotation in range(EXPECTED_RUNS)
    }
    return derive_candidate(runs, resources, sass_contract)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.capture.resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
