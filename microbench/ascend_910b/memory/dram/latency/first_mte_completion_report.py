#!/usr/bin/env python3
"""Audit paired first MTE request-to-completion measurements."""

import argparse
import csv
import math
from pathlib import Path
import statistics

from first_mte_completion_runner import (
    CAMPAIGN_CELLS,
    CAMPAIGN_REQUIRED_INDEPENDENT_RUNS,
    CELLS,
    PAIR_ORDERS,
)


SYSTEM_COUNTER_TICK_NS = 20.0
REPLICATION_RELATIVE_LIMIT = 0.20
INDEPENDENT_RUN_RELATIVE_LIMIT = 0.20
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "direction", "tile_bytes", "pair_order", "stride_bytes",
    "control_cycles", "request_cycles", "control_ns", "request_ns",
    "sink", "destination_head", "validation",
)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _load_rows(paths, cells=CELLS):
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
                    name: int(raw[name])
                    for name in (
                        "trial", "order_position", "process_pid",
                        "block_dim", "worker", "direction", "tile_bytes",
                        "pair_order", "stride_bytes", "control_cycles",
                        "request_cycles", "sink", "destination_head",
                    )
                }
                cell = (
                    integers["block_dim"], integers["direction"],
                    integers["tile_bytes"], integers["pair_order"])
                if (integers["trial"] <= 0 or integers["process_pid"] <= 0
                        or cell not in cells or integers["worker"] < 0
                        or integers["worker"] >= integers["block_dim"]
                        or integers["stride_bytes"] < integers["tile_bytes"]
                        or integers["stride_bytes"] % 32 != 0
                        or integers["control_cycles"] < 0
                        or integers["request_cycles"] <= 0):
                    raise ValueError(
                        f"{path} contains invalid geometry or values")
                control_ns = float(raw["control_ns"])
                request_ns = float(raw["request_ns"])
                expected_control_ns = (
                    integers["control_cycles"] * SYSTEM_COUNTER_TICK_NS)
                expected_request_ns = (
                    integers["request_cycles"] * SYSTEM_COUNTER_TICK_NS)
                if (not math.isfinite(control_ns) or control_ns < 0
                        or not math.isfinite(request_ns) or request_ns <= 0
                        or not math.isclose(
                            control_ns, expected_control_ns,
                            rel_tol=0.0, abs_tol=1e-9)
                        or not math.isclose(
                            request_ns, expected_request_ns,
                            rel_tol=0.0, abs_tol=1e-9)):
                    raise ValueError(f"{path} contains inconsistent elapsed time")
                identity = (
                    raw["run_id"], integers["trial"], *cell,
                    integers["worker"])
                if identity in identities:
                    raise ValueError(f"duplicate row identity {identity}")
                identities.add(identity)
                rows.append({
                    "run_id": raw["run_id"],
                    "trial": integers["trial"],
                    "order_position": integers["order_position"],
                    "process_pid": integers["process_pid"],
                    "block_dim": integers["block_dim"],
                    "worker": integers["worker"],
                    "direction": integers["direction"],
                    "tile_bytes": integers["tile_bytes"],
                    "pair_order": integers["pair_order"],
                    "control_ns": control_ns,
                    "request_ns": request_ns,
                })
    if not rows:
        raise ValueError("no first-MTE rows were provided")
    return rows


def _relative_difference(left, right):
    scale = max(abs(left), abs(right))
    return 0.0 if scale == 0 else abs(left - right) / scale


def _audit(rows, cells=CELLS):
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("one report must contain exactly one run_id")
    trials = sorted({row["trial"] for row in rows})
    if trials != list(range(1, len(trials) + 1)):
        raise ValueError("trial identities must be contiguous from one")
    by_trial = {}
    for trial in trials:
        trial_rows = [row for row in rows if row["trial"] == trial]
        observed_cells = {
            (row["block_dim"], row["direction"], row["tile_bytes"],
             row["pair_order"])
            for row in trial_rows}
        if observed_cells != set(cells):
            raise ValueError("each trial changed the frozen cell set")
        if {row["order_position"] for row in trial_rows} != set(
                range(len(cells))):
            raise ValueError("each trial must contain every order position")
        if len({row["process_pid"] for row in trial_rows}) != 1:
            raise ValueError("one trial must come from one child process")
        intervals = {}
        for cell in cells:
            block_dim, direction, tile_bytes, pair_order = cell
            cell_rows = [
                row for row in trial_rows
                if (row["block_dim"], row["direction"], row["tile_bytes"],
                    row["pair_order"]) == cell]
            if (len(cell_rows) != block_dim
                    or {row["worker"] for row in cell_rows}
                    != set(range(block_dim))
                    or len({row["order_position"] for row in cell_rows}) != 1):
                raise ValueError(
                    "each cell must contain every AIV worker exactly once")
            intervals[cell] = {
                row["worker"]: {
                    "control_ns": row["control_ns"],
                    "request_ns": row["request_ns"],
                }
                for row in cell_rows
            }
        by_trial[trial] = intervals
    expected_per_position = len(trials) // len(cells)
    for position in range(len(cells)):
        counts = {
            cell: sum(
                row["order_position"] == position
                and row["worker"] == 0
                and (row["block_dim"], row["direction"],
                     row["tile_bytes"], row["pair_order"]) == cell
                for row in rows)
            for cell in cells
        }
        if set(counts.values()) != {expected_per_position}:
            raise ValueError("cell/order schedule is not balanced")

    complete_schedule_cycles = len(trials) // len(cells)
    use_balanced_cycle_replicas = complete_schedule_cycles >= 2
    services = {}
    base_cells = sorted({cell[:3] for cell in cells})
    if any(
            (block_dim, direction, tile_bytes, pair_order) not in cells
            for block_dim, direction, tile_bytes in base_cells
            for pair_order in PAIR_ORDERS):
        raise ValueError("each request cell requires both pair orders")
    for block_dim, direction, tile_bytes in base_cells:
        samples = tuple(
            max(
                intervals[(block_dim, direction, tile_bytes, 1)]
                [worker]["request_ns"]
                - intervals[(block_dim, direction, tile_bytes, 0)]
                [worker]["control_ns"]
                for worker in range(block_dim))
            for intervals in by_trial.values())
        summary = _quartiles(samples)
        if use_balanced_cycle_replicas:
            first_replica = tuple(
                sample for index, sample in enumerate(samples)
                if (index // len(cells)) % 2 == 0)
            second_replica = tuple(
                sample for index, sample in enumerate(samples)
                if (index // len(cells)) % 2 == 1)
        else:
            first_replica = samples[::2]
            second_replica = samples[1::2]
        odd = statistics.median(first_replica)
        even = statistics.median(second_replica)
        reproducible = (
            _relative_difference(odd, even)
            <= REPLICATION_RELATIVE_LIMIT)
        services[(block_dim, direction, tile_bytes)] = {
            "samples": samples,
            "summary": summary,
            "odd": odd,
            "even": even,
            "reproducible": reproducible,
            "mechanism_gate": reproducible and summary[0] > 0,
        }
    unloaded_block_dim = min(cell[0] for cell in cells)
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_cell": len(trials),
        "replication_basis": (
            "balanced_schedule_cycle_parity"
            if use_balanced_cycle_replicas else "trial_parity"),
        "services": services,
        "unloaded_mechanism_gate": all(
            result["mechanism_gate"]
            for (block_dim, _, _), result in services.items()
            if block_dim == unloaded_block_dim),
        "mechanism_gate": all(
            result["mechanism_gate"] for result in services.values()),
    }


def _audit_independent_runs(paths, cells, required_runs):
    if len(paths) != required_runs:
        raise ValueError(
            f"campaign requires exactly {required_runs} independent raw files")
    audits = tuple(_audit(_load_rows((path,), cells), cells) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != required_runs:
        raise ValueError("independent raw files must have distinct run_id values")
    if not all(audit["mechanism_gate"] for audit in audits):
        raise ValueError("at least one independent run failed its mechanism gate")
    samples_per_cell = {audit["samples_per_cell"] for audit in audits}
    replication_bases = {audit["replication_basis"] for audit in audits}
    if len(samples_per_cell) != 1 or len(replication_bases) != 1:
        raise ValueError("independent runs changed the frozen schedule")
    services = {}
    for cell in sorted(audits[0]["services"]):
        medians = tuple(
            audit["services"][cell]["summary"][1] for audit in audits)
        spread = _relative_difference(min(medians), max(medians))
        services[cell] = {
            "medians_by_run": dict(zip(run_ids, medians)),
            "service_time_ns": statistics.median(medians),
            "maximum_run_relative_median_spread": spread,
            "stability_gate": spread <= INDEPENDENT_RUN_RELATIVE_LIMIT,
        }
    return {
        "run_ids": run_ids,
        "samples_per_cell_per_run": next(iter(samples_per_cell)),
        "replication_basis": next(iter(replication_bases)),
        "services": services,
        "mechanism_gate": all(
            service["stability_gate"] for service in services.values()),
    }


def _print_report(audit):
    print("# Ascend 910B first-MTE completion audit")
    print(f"samples_per_cell={audit['samples_per_cell']}")
    print(f"replication_basis={audit['replication_basis']}")
    print("block_dim,direction,tile_bytes,q1_ns,median_ns,q3_ns,replication")
    for cell in sorted(audit["services"]):
        block_dim, direction, tile_bytes = cell
        result = audit["services"][cell]
        q1, median, q3 = result["summary"]
        direction_name = {
            0: "mte2_to_mte3",
            1: "mte3_to_mte2",
            2: "mte2_to_vector",
            3: "mte3_to_vector",
            4: "tque_mte2_to_vector",
            5: "tque_vector_to_mte3",
            6: "tque_datacopypad_mte2_to_vector",
            7: "tque_datacopypad_vector_to_mte3",
            8: "gather_index_mte2_to_scalar_get_value",
        }[direction]
        replication = "pass" if result["reproducible"] else "fail"
        print(
            f"{block_dim},{direction_name},{tile_bytes},"
            f"{q1:.9f},{median:.9f},{q3:.9f},{replication}")
    print(
        "unloaded_mechanism_gate="
        f"{'pass' if audit['unloaded_mechanism_gate'] else 'fail'}")
    print(
        f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")


def _print_independent_report(audit):
    print("# Ascend 910B first-MTE independent-run audit")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(f"samples_per_cell_per_run={audit['samples_per_cell_per_run']}")
    print(f"replication_basis={audit['replication_basis']}")
    print(
        "block_dim,direction,tile_bytes,service_time_ns,"
        "maximum_run_relative_median_spread,stability")
    direction_names = {
        0: "mte2_to_mte3",
        1: "mte3_to_mte2",
        2: "mte2_to_vector",
        3: "mte3_to_vector",
        4: "tque_mte2_to_vector",
        5: "tque_vector_to_mte3",
        6: "tque_datacopypad_mte2_to_vector",
        7: "tque_datacopypad_vector_to_mte3",
        8: "gather_index_mte2_to_scalar_get_value",
    }
    for cell, result in audit["services"].items():
        block_dim, direction, tile_bytes = cell
        stability = "pass" if result["stability_gate"] else "fail"
        print(
            f"{block_dim},{direction_names[direction]},{tile_bytes},"
            f"{result['service_time_ns']:.9f},"
            f"{result['maximum_run_relative_median_spread']:.9f},"
            f"{stability}")
    print(
        f"mechanism_gate={'pass' if audit['mechanism_gate'] else 'fail'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", choices=tuple(CAMPAIGN_CELLS),
                        default="baseline")
    parser.add_argument("raw_csv", nargs="+", type=Path)
    args = parser.parse_args()
    cells = CAMPAIGN_CELLS[args.campaign]
    required_runs = CAMPAIGN_REQUIRED_INDEPENDENT_RUNS[args.campaign]
    if required_runs == 1:
        _print_report(_audit(_load_rows(args.raw_csv, cells), cells))
    else:
        audit = _audit_independent_runs(args.raw_csv, cells, required_runs)
        _print_independent_report(audit)
        if not audit["mechanism_gate"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
