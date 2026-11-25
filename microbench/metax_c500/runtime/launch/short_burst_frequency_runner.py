#!/usr/bin/env python3
"""Generate and audit C500 register-only and memory-coupled frequency runs."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics


SCHEMA_VERSION = 3
TRIALS = 5
PATTERNS = (
    "add", "exp", "rcp", "silu_mix",
    "memory_only", "memory_add", "memory_silu_mix",
)
GRIDS = (104, 416, 1664, 3328, 6656, 9984, 13312)
PROGRAM_REPEATS = (1, 4, 16)
BLOCK = 512
CONDITIONS = (
    ("cold", 2000, 0),
    ("idle100", 100, 0),
    ("warm", 0, 1),
    ("warm3self", 0, 2),
)
DPM_LEVELS_MHZ = (429, 600, 686, 750, 800, 900, 960, 1000, 1067, 1125)
PLAN_FIELDS = (
    "sample_id", "condition", "pattern", "trial", "order_position",
    "grid", "program_repeats", "idle_ms", "prewarm",
)
SAMPLE_FIELDS = (
    "sample_id", "condition", "pattern", "trial", "order_position",
    "grid", "block", "program_repeats", "instructions_per_wave",
    "idle_ms", "prewarm", "host_start_ns", "host_end_ns", "event_ns",
    "clock64_min", "clock64_median", "clock64_max", "realtime_min_ns",
    "realtime_median_ns", "realtime_max_ns", "grid_realtime_span_ns",
    "frequency_min_mhz", "frequency_median_mhz", "frequency_max_mhz",
    "frequency_first_decile_mhz", "frequency_last_decile_mhz",
)


def _matrix(campaign):
    if campaign == "smoke":
        return (
            (1,), ("add", "silu_mix"), (104, 7168, 13312),
            (1, 4), CONDITIONS,
        )
    if campaign == "full":
        return (
            tuple(range(1, TRIALS + 1)), PATTERNS, GRIDS,
            PROGRAM_REPEATS, CONDITIONS,
        )
    if campaign == "memory":
        return (
            (1, 2, 3),
            ("silu_mix", "memory_only", "memory_add", "memory_silu_mix"),
            GRIDS,
            (1,),
            (
                ("cold", 2000, 0),
                ("warm", 0, 1),
                ("warm3self", 0, 2),
            ),
        )
    raise ValueError(f"unknown short-burst campaign: {campaign!r}")


def _instructions_per_wave(pattern, program_repeats):
    if pattern in ("add", "exp", "rcp", "silu_mix"):
        return 256 * program_repeats
    if pattern == "memory_only":
        return 3
    if pattern in ("memory_add", "memory_silu_mix"):
        return 3 + 256 * program_repeats
    raise ValueError(f"unknown short-burst pattern: {pattern!r}")


def build_plan(campaign):
    trials, patterns, grids, repeats, conditions = _matrix(campaign)
    rows = []
    for trial in trials:
        scheduled = [
            (condition, idle_ms, prewarm, pattern, grid, program_repeats)
            for condition, idle_ms, prewarm in conditions
            for pattern in patterns
            for grid in grids
            for program_repeats in repeats
        ]
        random.Random(500823 + 7919 * trial).shuffle(scheduled)
        for order_position, item in enumerate(scheduled):
            condition, idle_ms, prewarm, pattern, grid, program_repeats = item
            rows.append({
                "sample_id": (
                    f"{condition}_{pattern}_g{grid}_r{program_repeats}_"
                    f"t{trial}"),
                "condition": condition,
                "pattern": pattern,
                "trial": trial,
                "order_position": order_position,
                "grid": grid,
                "program_repeats": program_repeats,
                "idle_ms": idle_ms,
                "prewarm": prewarm,
            })
    return tuple(rows)


def _plan_text(rows):
    return "".join(
        " ".join(str(row[field]) for field in PLAN_FIELDS) + "\n"
        for row in rows
    )


def write_plan(path, campaign):
    if path.exists():
        raise FileExistsError(f"plan path already exists: {path}")
    rows = build_plan(campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_plan_text(rows), encoding="utf-8")
    return len(rows)


def read_plan(path, campaign):
    actual = path.read_text(encoding="utf-8")
    expected_rows = build_plan(campaign)
    if actual != _plan_text(expected_rows):
        raise ValueError("short-burst plan differs from the frozen generator")
    return expected_rows


def _parse_int(text, label, *, positive=False):
    value = int(text)
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")
    if not positive and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _parse_float(text, label):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def read_samples(path, plan):
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != len(plan) + 2:
        raise ValueError("short-burst stdout row count is invalid")
    metadata = rows[0]
    if (len(metadata) != 14 or metadata[0] != "metadata"
            or metadata[1] != str(SCHEMA_VERSION)
            or metadata[2] != "MetaX C500"
            or metadata[4].lower() != "0000:2c:00.0"
            or metadata[5:7] != ["104", "64"]
            or int(metadata[7]) <= 0
            or metadata[8:] != [
                "512", "416", "256", "8", "3", "100000000"]):
        raise ValueError("short-burst metadata is invalid")
    expected_header = ("sample_header",) + SAMPLE_FIELDS
    if tuple(rows[1]) != expected_header:
        raise ValueError("short-burst sample header is invalid")
    samples = []
    for raw, planned in zip(rows[2:], plan):
        if len(raw) != len(SAMPLE_FIELDS) + 1 or raw[0] != "sample":
            raise ValueError("short-burst sample row schema is invalid")
        values = dict(zip(SAMPLE_FIELDS, raw[1:]))
        for field in PLAN_FIELDS:
            if str(planned[field]) != values[field]:
                raise ValueError(
                    "short-burst sample differs from plan: "
                    f"{planned['sample_id']} {field}")
        sample = {
            "sample_id": values["sample_id"],
            "condition": values["condition"],
            "pattern": values["pattern"],
            "trial": _parse_int(values["trial"], "trial", positive=True),
            "order_position": _parse_int(
                values["order_position"], "order_position"),
            "grid": _parse_int(values["grid"], "grid", positive=True),
            "block": _parse_int(values["block"], "block", positive=True),
            "program_repeats": _parse_int(
                values["program_repeats"], "program_repeats", positive=True),
            "instructions_per_wave": _parse_int(
                values["instructions_per_wave"],
                "instructions_per_wave", positive=True),
            "idle_ms": _parse_int(values["idle_ms"], "idle_ms"),
            "prewarm": _parse_int(values["prewarm"], "prewarm"),
            "host_start_ns": _parse_int(
                values["host_start_ns"], "host_start_ns", positive=True),
            "host_end_ns": _parse_int(
                values["host_end_ns"], "host_end_ns", positive=True),
        }
        for field in SAMPLE_FIELDS[13:]:
            sample[field] = _parse_float(values[field], field)
        expected_instructions = _instructions_per_wave(
            sample["pattern"], sample["program_repeats"])
        if (sample["block"] != BLOCK
                or sample["instructions_per_wave"] != expected_instructions
                or sample["host_end_ns"] <= sample["host_start_ns"]
                or not (sample["clock64_min"]
                        <= sample["clock64_median"]
                        <= sample["clock64_max"])
                or not (sample["realtime_min_ns"]
                        <= sample["realtime_median_ns"]
                        <= sample["realtime_max_ns"])
                or not (sample["frequency_min_mhz"]
                        <= sample["frequency_median_mhz"]
                        <= sample["frequency_max_mhz"])):
            raise ValueError("short-burst sample invariant failed")
        samples.append(sample)
    return metadata, tuple(samples)


def _nearest_dpm_level(frequency_mhz):
    return min(DPM_LEVELS_MHZ, key=lambda level: abs(level - frequency_mhz))


def summarize(metadata, samples, campaign):
    trials, patterns, grids, repeats, conditions = _matrix(campaign)
    cells = []
    for condition, _, _ in conditions:
        for pattern in patterns:
            for grid in grids:
                for program_repeats in repeats:
                    selected = [
                        sample for sample in samples
                        if sample["condition"] == condition
                        and sample["pattern"] == pattern
                        and sample["grid"] == grid
                        and sample["program_repeats"] == program_repeats
                    ]
                    if len(selected) != len(trials):
                        raise ValueError("short-burst cell is incomplete")
                    median_frequency = statistics.median(
                        sample["frequency_median_mhz"]
                        for sample in selected)
                    cells.append({
                        "condition": condition,
                        "pattern": pattern,
                        "grid": grid,
                        "block": BLOCK,
                        "program_repeats": program_repeats,
                        "instructions_per_wave": _instructions_per_wave(
                            pattern, program_repeats),
                        "sample_count": len(selected),
                        "median_event_ns": statistics.median(
                            sample["event_ns"] for sample in selected),
                        "median_grid_realtime_span_ns": statistics.median(
                            sample["grid_realtime_span_ns"]
                            for sample in selected),
                        "median_clock64_span": statistics.median(
                            sample["clock64_median"] for sample in selected),
                        "median_realtime_span_ns": statistics.median(
                            sample["realtime_median_ns"]
                            for sample in selected),
                        "median_frequency_mhz": median_frequency,
                        "nearest_dpm_level_mhz": _nearest_dpm_level(
                            median_frequency),
                        "median_first_decile_frequency_mhz": statistics.median(
                            sample["frequency_first_decile_mhz"]
                            for sample in selected),
                        "median_last_decile_frequency_mhz": statistics.median(
                            sample["frequency_last_decile_mhz"]
                            for sample in selected),
                    })
    by_identity = {
        (
            cell["condition"], cell["pattern"], cell["grid"],
            cell["program_repeats"],
        ): cell
        for cell in cells
    }
    comparisons = []
    for pattern in patterns:
        for grid in grids:
            for program_repeats in repeats:
                warm = by_identity[(
                    "warm", pattern, grid, program_repeats)]
                for condition, _, _ in conditions:
                    if condition == "warm":
                        continue
                    cell = by_identity[(
                        condition, pattern, grid, program_repeats)]
                    comparisons.append({
                        "condition": condition,
                        "pattern": pattern,
                        "grid": grid,
                        "block": BLOCK,
                        "program_repeats": program_repeats,
                        "warm_median_event_ns": warm["median_event_ns"],
                        "condition_median_event_ns": cell["median_event_ns"],
                        "event_time_ratio_to_warm": (
                            cell["median_event_ns"]
                            / warm["median_event_ns"]),
                        "warm_median_frequency_mhz": (
                            warm["median_frequency_mhz"]),
                        "condition_median_frequency_mhz": (
                            cell["median_frequency_mhz"]),
                        "frequency_ratio_to_warm": (
                            cell["median_frequency_mhz"]
                            / warm["median_frequency_mhz"]),
                    })
    return {
        "schema_version": SCHEMA_VERSION,
        "mechanism": "short_burst_xcore_frequency_state",
        "campaign": campaign,
        "timer_contract": {
            "clock64": "AP-local XCORE cycles",
            "get_realtime": "100-MHz device-global ticks",
            "derived_frequency_mhz": "clock64_span / realtime_span * 100",
        },
        "instruction_patterns": {
            "add": "256 add_f32 instructions per repeat and wave",
            "exp": "256 exp_f32 instructions per repeat and wave",
            "rcp": "256 rcp_f32 instructions per repeat and wave",
            "silu_mix": (
                "240 add_f32 plus 8 exp_f32 plus 8 rcp_f32 instructions "
                "per repeat and wave"),
            "memory_only": (
                "two coalesced 16-byte-per-thread global loads and one "
                "coalesced 16-byte-per-thread global store per wave"),
            "memory_add": (
                "the memory-only body plus 256 add_f32 instructions per "
                "repeat and wave"),
            "memory_silu_mix": (
                "the memory-only body plus 240 add_f32, 8 exp_f32 and 8 "
                "rcp_f32 instructions per repeat and wave"),
        },
        "metadata": metadata,
        "sample_count": len(samples),
        "cells": cells,
        "comparisons": comparisons,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("plan", "report"))
    parser.add_argument(
        "--campaign", default="full", choices=("smoke", "full", "memory"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stdout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        if args.stdout is not None or args.output is not None:
            parser.error("plan mode accepts only --plan and --campaign")
        print(json.dumps({
            "campaign": args.campaign,
            "sample_count": write_plan(args.plan, args.campaign),
        }))
        return 0
    if args.stdout is None or args.output is None:
        parser.error("report mode requires --stdout and --output")
    if args.output.exists():
        parser.error("--output must not already exist")
    plan = read_plan(args.plan, args.campaign)
    metadata, samples = read_samples(args.stdout, plan)
    report = summarize(metadata, samples, args.campaign)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "campaign": args.campaign,
        "sample_count": report["sample_count"],
        "cell_count": len(report["cells"]),
        "comparison_count": len(report["comparisons"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
