#!/usr/bin/env python3
"""Audit C500 L2-request / DRAM-fill interference trials."""

import argparse
import csv
import math
from pathlib import Path
import statistics


EXPECTED_FIELDS = (
    "mode",
    "cold_fraction",
    "grid",
    "block",
    "iterations",
    "logical_bytes_per_iteration",
    "cold_bytes_per_iteration",
    "trial",
    "event_ms",
    "logical_gbs",
    "cold_gbs",
)
MODE_FRACTIONS = {
    "cold0": 0.0,
    "cold25": 0.25,
    "cold50": 0.5,
    "cold75": 0.75,
    "cold100": 1.0,
}
EXPECTED_GRID = 1664
EXPECTED_BLOCK = 256
EXPECTED_ITERATIONS = 32
EXPECTED_LOGICAL_BYTES = 256 * 1024 * 1024
EXPECTED_TRIALS = 7
MINIMUM_INDEPENDENT_RUNS = 3
DISCRIMINATING_FRACTIONS = (0.25, 0.5)


def _positive_float(text, name):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each fraction requires at least four samples")
    q1, _, q3 = statistics.quantiles(
        values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _parse_run(run_id, path):
    if not run_id or not run_id.strip():
        raise ValueError("run ID must be non-empty")
    rows = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        identities = set()
        for raw in reader:
            mode = raw["mode"]
            if mode not in MODE_FRACTIONS:
                raise ValueError(f"{path} has unknown mode {mode!r}")
            fraction = float(raw["cold_fraction"])
            grid = int(raw["grid"])
            block = int(raw["block"])
            iterations = int(raw["iterations"])
            logical_bytes = int(raw["logical_bytes_per_iteration"])
            cold_bytes = int(raw["cold_bytes_per_iteration"])
            trial = int(raw["trial"])
            event_ms = _positive_float(raw["event_ms"], "event_ms")
            logical_gbs = _positive_float(
                raw["logical_gbs"], "logical_gbs")
            cold_gbs = float(raw["cold_gbs"])
            if not math.isfinite(cold_gbs) or cold_gbs < 0:
                raise ValueError("cold_gbs must be finite and non-negative")
            if not math.isclose(
                    fraction, MODE_FRACTIONS[mode],
                    rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{path} mode/fraction mismatch")
            if (grid != EXPECTED_GRID or block != EXPECTED_BLOCK
                    or iterations != EXPECTED_ITERATIONS
                    or logical_bytes != EXPECTED_LOGICAL_BYTES):
                raise ValueError(f"{path} changed frozen launch/work")
            expected_cold_bytes = round(logical_bytes * fraction)
            if cold_bytes != expected_cold_bytes:
                raise ValueError(f"{path} changed cold-byte fraction")
            if trial < 1 or trial > EXPECTED_TRIALS:
                raise ValueError(f"{path} has invalid trial {trial}")
            identity = (mode, trial)
            if identity in identities:
                raise ValueError(f"{path} duplicates {identity}")
            identities.add(identity)
            total_bytes = logical_bytes * iterations
            expected_logical_gbs = total_bytes / (event_ms * 1.0e6)
            expected_cold_gbs = expected_logical_gbs * fraction
            if (not math.isclose(
                    logical_gbs, expected_logical_gbs,
                    rel_tol=1e-6, abs_tol=1e-6)
                    or not math.isclose(
                        cold_gbs, expected_cold_gbs,
                        rel_tol=1e-6, abs_tol=1e-6)):
                raise ValueError(f"{path} has inconsistent derived rates")
            rows.append({
                "run_id": run_id.strip(),
                "mode": mode,
                "fraction": fraction,
                "event_ms": event_ms,
            })
    expected_identities = {
        (mode, trial)
        for mode in MODE_FRACTIONS
        for trial in range(1, EXPECTED_TRIALS + 1)
    }
    if identities != expected_identities:
        raise ValueError(f"{path} is incomplete")
    return rows


def load_runs(run_specs):
    if len(run_specs) < MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            f"at least {MINIMUM_INDEPENDENT_RUNS} runs are required")
    run_ids = tuple(run_id for run_id, _path in run_specs)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("run IDs must be unique")
    rows = []
    for run_id, path in run_specs:
        rows.extend(_parse_run(run_id, path))
    return rows


def audit(rows, l2_gbs, dram_gbs):
    l2_gbs = _positive_float(l2_gbs, "l2_gbs")
    dram_gbs = _positive_float(dram_gbs, "dram_gbs")
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) < MINIMUM_INDEPENDENT_RUNS:
        raise ValueError("rows do not contain three independent runs")
    summaries = {}
    for mode, fraction in MODE_FRACTIONS.items():
        values = [
            row["event_ms"] for row in rows if row["mode"] == mode]
        expected_count = len(run_ids) * EXPECTED_TRIALS
        if len(values) != expected_count:
            raise ValueError(f"fraction {fraction} has incomplete samples")
        summaries[fraction] = _quartiles(values)

    baseline = summaries[0.0]
    capacity_ratio = l2_gbs / dram_gbs
    comparisons = {}
    classifications = []
    for fraction, summary in summaries.items():
        observed_interval = (
            summary[0] / baseline[2],
            summary[2] / baseline[0],
        )
        observed_ratio = summary[1] / baseline[1]
        independent_ratio = max(1.0, fraction * capacity_ratio)
        shared_fill_ratio = max(1.0 + fraction, fraction * capacity_ratio)
        midpoint = (independent_ratio + shared_fill_ratio) / 2.0
        classification = "not_discriminating"
        if fraction in DISCRIMINATING_FRACTIONS:
            if observed_interval[0] > midpoint:
                classification = "shared_fill_preferred"
            elif observed_interval[1] < midpoint:
                classification = "shared_fill_rejected"
            else:
                classification = "ambiguous"
            classifications.append(classification)
        comparisons[fraction] = {
            "event_ms_iqr": summary,
            "observed_ratio": observed_ratio,
            "observed_ratio_iqr": observed_interval,
            "independent_ratio": independent_ratio,
            "shared_fill_ratio": shared_fill_ratio,
            "decision_midpoint": midpoint,
            "classification": classification,
        }
    if all(
            value == "shared_fill_preferred"
            for value in classifications):
        topology = "shared_fill_preferred"
    elif all(
            value == "shared_fill_rejected"
            for value in classifications):
        topology = "shared_fill_rejected"
    else:
        topology = "ambiguous"
    return {
        "independent_runs": len(run_ids),
        "samples_per_fraction": len(run_ids) * EXPECTED_TRIALS,
        "l2_gbs": l2_gbs,
        "dram_gbs": dram_gbs,
        "comparisons": comparisons,
        "topology_candidate": topology,
    }


def _run_spec(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("run must be RUN_ID=PATH")
    run_id, path_text = text.split("=", 1)
    if not run_id.strip() or not path_text:
        raise argparse.ArgumentTypeError("run must be RUN_ID=PATH")
    path = Path(path_text)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"raw CSV does not exist: {path}")
    return run_id.strip(), path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        type=_run_spec)
    parser.add_argument("--l2-gbs", required=True)
    parser.add_argument("--dram-gbs", required=True)
    args = parser.parse_args()
    result = audit(
        load_runs(tuple(args.run)), args.l2_gbs, args.dram_gbs)
    print(f"independent_runs={result['independent_runs']}")
    print(f"samples_per_fraction={result['samples_per_fraction']}")
    print(f"l2_gbs={result['l2_gbs']}")
    print(f"dram_gbs={result['dram_gbs']}")
    for fraction in sorted(result["comparisons"]):
        print(f"fraction_{fraction}={result['comparisons'][fraction]}")
    print(f"topology_candidate={result['topology_candidate']}")
    if result["topology_candidate"] == "ambiguous":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
