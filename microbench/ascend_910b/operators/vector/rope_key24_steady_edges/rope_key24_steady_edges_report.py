#!/usr/bin/env python3
"""Audit Ascend 910B RoPE key-24 double-buffer edge measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from rope_key24_steady_edges_runner import (
    BASE_CELLS,
    BLOCK_DIM,
    CELLS,
    DEFAULT_SCHEDULE_CYCLES,
    MODE_NAMES,
    STRIDE_BYTES,
    SYSTEM_COUNTER_TICK_NS,
)


RELATIVE_SPREAD_LIMIT = 0.20
REQUIRED_INDEPENDENT_RUNS = 3
PUBLISHABLE_CELLS = (
    (1, 4096, 20),
    (1, 4096, 21),
)
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "mode", "payload_bytes", "request_count", "pair_order",
    "stride_bytes", "control_cycles", "request_cycles", "control_ns",
    "request_ns", "sink", "validation",
)


def _relative_spread(values):
    low = min(values)
    high = max(values)
    return 0.0 if high == 0 else (high - low) / high


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
                try:
                    values = {
                        name: int(raw[name])
                        for name in REQUIRED_FIELDS[1:13]
                    }
                    control_ns = float(raw["control_ns"])
                    request_ns = float(raw["request_ns"])
                    sink = int(raw["sink"])
                except ValueError as exc:
                    raise ValueError(f"{path} contains non-numeric data") from exc
                cell = (
                    values["mode"], values["payload_bytes"],
                    values["request_count"], values["pair_order"])
                if (values["trial"] <= 0 or values["order_position"] < 0
                        or values["process_pid"] <= 0
                        or values["block_dim"] != BLOCK_DIM
                        or values["worker"] not in range(BLOCK_DIM)
                        or cell not in CELLS
                        or values["stride_bytes"] != STRIDE_BYTES
                        or values["control_cycles"] < 0
                        or values["request_cycles"] <= 0 or sink <= 0):
                    raise ValueError(f"{path} contains invalid values")
                if (not math.isfinite(control_ns) or control_ns < 0
                        or not math.isfinite(request_ns) or request_ns <= 0
                        or not math.isclose(
                            control_ns,
                            values["control_cycles"] * SYSTEM_COUNTER_TICK_NS,
                            rel_tol=0.0, abs_tol=1e-9)
                        or not math.isclose(
                            request_ns,
                            values["request_cycles"] * SYSTEM_COUNTER_TICK_NS,
                            rel_tol=0.0, abs_tol=1e-9)):
                    raise ValueError(f"{path} has inconsistent elapsed time")
                identity = (
                    raw["run_id"], values["trial"], *cell,
                    values["worker"])
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                rows.append({
                    "run_id": raw["run_id"],
                    **values,
                    "control_ns": control_ns,
                    "request_ns": request_ns,
                })
    if not rows:
        raise ValueError("no steady-edge rows were provided")
    return rows


def _audit_structure(rows):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one raw file must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    expected_trials = DEFAULT_SCHEDULE_CYCLES * len(CELLS)
    if trials != list(range(1, expected_trials + 1)):
        raise ValueError(
            f"run must contain the frozen {expected_trials}-trial schedule")
    rows_by_trial = {trial: [] for trial in trials}
    for row in rows:
        rows_by_trial[row["trial"]].append(row)
    position_counts = [dict() for _ in CELLS]
    by_trial = {}
    for trial in trials:
        trial_rows = rows_by_trial[trial]
        rows_by_cell = {cell: [] for cell in CELLS}
        for row in trial_rows:
            cell = (
                row["mode"], row["payload_bytes"], row["request_count"],
                row["pair_order"])
            if cell not in rows_by_cell:
                raise ValueError(f"trial contains unknown cell {cell!r}")
            rows_by_cell[cell].append(row)
            if row["worker"] == 0:
                if row["order_position"] not in range(len(CELLS)):
                    raise ValueError(
                        "each trial must contain every order position")
                counts = position_counts[row["order_position"]]
                counts[cell] = counts[cell] + 1 if cell in counts else 1
        observed_cells = {
            cell for cell, cell_rows in rows_by_cell.items() if cell_rows}
        if observed_cells != set(CELLS):
            raise ValueError("each trial changed the frozen cell set")
        if ({row["order_position"] for row in trial_rows}
                != set(range(len(CELLS)))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        intervals = {}
        for cell in CELLS:
            cell_rows = rows_by_cell[cell]
            if (len(cell_rows) != BLOCK_DIM
                    or {row["worker"] for row in cell_rows}
                    != set(range(BLOCK_DIM))
                    or len({row["order_position"] for row in cell_rows}) != 1):
                raise ValueError("each cell must contain every worker once")
            intervals[cell] = {
                row["worker"]: row["request_ns"] - row["control_ns"]
                for row in cell_rows
            }
        by_trial[trial] = intervals
    expected_per_position = len(trials) // len(CELLS)
    for counts in position_counts:
        if (set(counts) != set(CELLS)
                or set(counts.values()) != {expected_per_position}):
            raise ValueError("cell/order schedule is not balanced")
    return next(iter(run_ids)), trials, by_trial


def _base_samples(trials, by_trial):
    samples = {}
    for base_cell in BASE_CELLS:
        samples[base_cell] = tuple(
            max(
                by_trial[trial][(*base_cell, pair_order)][worker]
                for pair_order in (0, 1)
                for worker in range(BLOCK_DIM))
            for trial in trials)
    return samples


def _audit(rows):
    run_id, trials, by_trial = _audit_structure(rows)
    samples = _base_samples(trials, by_trial)
    medians = {
        cell: statistics.median(values) for cell, values in samples.items()}
    event_cell = (3, 0, 16)
    medians[event_cell] /= 8
    parity_spreads = {}
    for cell, values in samples.items():
        first = tuple(
            value for index, value in enumerate(values)
            if (index // len(CELLS)) % 2 == 0)
        second = tuple(
            value for index, value in enumerate(values)
            if (index // len(CELLS)) % 2 == 1)
        parity_spreads[cell] = _relative_spread((
            statistics.median(first), statistics.median(second)))
    positive = all(medians[cell] > 0 for cell in PUBLISHABLE_CELLS)
    parity_gate = all(
        parity_spreads[cell] <= RELATIVE_SPREAD_LIMIT
        for cell in PUBLISHABLE_CELLS)
    return {
        "run_id": run_id,
        "samples_per_cell": len(trials),
        "medians": medians,
        "maximum_parity_spread": max(parity_spreads.values()),
        "mechanism_gate": positive and parity_gate,
    }


def _audit_independent_runs(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} raw files")
    audits = tuple(_audit(_load_rows((path,))) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("raw files must have distinct run_id values")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one run failed its mechanism gate")
    services = {}
    for key in BASE_CELLS:
        values = tuple(
            audit["medians"][key]
            for audit in audits)
        spread = _relative_spread(values)
        services[key] = {
            "service_time_ns": statistics.median(values),
            "maximum_run_relative_spread": spread,
            "stability_gate": spread <= RELATIVE_SPREAD_LIMIT,
        }
    return {
        "run_ids": run_ids,
        "samples_per_cell_per_run": audits[0]["samples_per_cell"],
        "services": services,
        "mechanism_gate": all(
            services[cell]["stability_gate"]
            for cell in PUBLISHABLE_CELLS),
    }


def _print_report(audit):
    print("# Ascend 910B RoPE key-24 steady-edge audit")
    print("publication_scope=grid47_4096b_mte2_steady_counts_20_21")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(f"samples_per_cell_per_run={audit['samples_per_cell_per_run']}")
    print(
        "service,mode,payload_bytes,request_count,service_time_ns,"
        "maximum_run_relative_spread,stability,publication")
    for key, service in audit["services"].items():
        mode, payload_bytes, request_count = key
        name = MODE_NAMES[mode]
        stability = "pass" if service["stability_gate"] else "fail"
        publication = "publish" if key in PUBLISHABLE_CELLS else "diagnostic_only"
        print(
            f"{name},{mode},{payload_bytes},{request_count},"
            f"{service['service_time_ns']:.9f},"
            f"{service['maximum_run_relative_spread']:.9f},{stability},"
            f"{publication}")
    print(f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    audit = _audit_independent_runs(args.raw_csv)
    _print_report(audit)
    if not audit["mechanism_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
