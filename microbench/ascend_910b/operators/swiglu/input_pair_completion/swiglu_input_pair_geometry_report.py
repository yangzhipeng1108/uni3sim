#!/usr/bin/env python3
"""Audit the generic q2 A/B row-stride geometry experiment.

The experiment measures a source-independent hardware matrix.  Its raw cells
are never Forward services: a later analytical candidate must first derive
one physical formula and validate it on a held-out geometry stratum.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from swiglu_input_pair_geometry_runner import (
    ACTIVE_WORKERS,
    BASE_CELLS,
    BASE_PHASE_BYTES,
    CELLS,
    MINIMUM_SAMPLES_PER_CELL,
    REQUIRED_INDEPENDENT_RUNS,
    ROW_BYTES,
    ROW_COUNTS,
    STRIDE_RATIOS,
    _cell_index,
)


SYSTEM_COUNTER_TICK_NS = 20.0
MAXIMUM_RUN_MEDIAN_SPREAD = 0.20
REQUIRED_FIELDS = (
    "run_id", "trial", "order_position", "process_pid", "block_dim",
    "worker", "row_count", "row_bytes", "stride_bytes",
    "base_phase_bytes", "pair_order", "control_cycles", "request_cycles",
    "control_ns", "request_ns", "sink", "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_span(values):
    center = statistics.median(values)
    if center <= 0:
        return math.inf
    return (max(values) - min(values)) / center


def _cell_key(cell):
    return ":".join(str(value) for value in cell)


def _load_run(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        raw_rows = tuple(reader)
    if not raw_rows:
        raise ValueError(f"{path} is empty")

    rows_by_trial_position = {}
    run_ids = set()
    identities = set()
    for raw in raw_rows:
        if raw["validation"] != "pass":
            raise ValueError(f"{path} includes a failed device validation")
        values = {
            name: int(raw[name])
            for name in (
                "trial", "order_position", "process_pid", "block_dim",
                "worker", "row_count", "row_bytes", "stride_bytes",
                "base_phase_bytes", "pair_order", "control_cycles",
                "request_cycles", "sink",
            )
        }
        control_ns = float(raw["control_ns"])
        request_ns = float(raw["request_ns"])
        if values["row_bytes"] <= 0:
            raise ValueError(f"{path} has a non-positive row size")
        stride_ratio, stride_remainder = divmod(
            values["stride_bytes"], values["row_bytes"])
        cell = (
            values["block_dim"], values["row_count"],
            values["row_bytes"], stride_ratio,
            values["base_phase_bytes"], values["pair_order"])
        if (cell not in CELLS or values["trial"] <= 0
                or not 0 <= values["order_position"] < len(CELLS)
                or values["process_pid"] <= 0
                or not 0 <= values["worker"] < values["block_dim"]
                or stride_remainder != 0
                or values["stride_bytes"] != values["row_bytes"] * cell[3]
                or values["base_phase_bytes"] not in BASE_PHASE_BYTES
                or values["control_cycles"] < 0
                or values["request_cycles"] <= 0
                or values["sink"] != 0xB4
                or not math.isclose(
                    control_ns, values["control_cycles"]
                    * SYSTEM_COUNTER_TICK_NS, rel_tol=0.0, abs_tol=1e-9)
                or not math.isclose(
                    request_ns, values["request_cycles"]
                    * SYSTEM_COUNTER_TICK_NS, rel_tol=0.0, abs_tol=1e-9)):
            raise ValueError(f"{path} contains an invalid row")
        identity = (
            values["trial"], values["order_position"], values["worker"])
        if identity in identities:
            raise ValueError(f"{path} repeats raw identity {identity}")
        identities.add(identity)
        rows_by_trial_position.setdefault(
            (values["trial"], values["order_position"]), []).append({
                "cell": cell,
                **values,
                "control_ns": control_ns,
                "request_ns": request_ns,
            })
        run_ids.add(raw["run_id"])

    if len(run_ids) != 1:
        raise ValueError(f"{path} must contain exactly one run ID")
    trials = tuple(sorted({row["trial"] for rows in rows_by_trial_position.values()
                           for row in rows}))
    if trials != tuple(range(1, len(trials) + 1)):
        raise ValueError(f"{path} trial IDs must be contiguous")
    if len(trials) < MINIMUM_SAMPLES_PER_CELL:
        raise ValueError(f"{path} has too few samples per geometry")

    per_trial_completion = {}
    for trial in trials:
        source_pairs = {}
        for position in range(len(CELLS)):
            expected = CELLS[_cell_index(trial, position)]
            rows = rows_by_trial_position.get((trial, position))
            if rows is None or len(rows) != expected[0]:
                raise ValueError(f"{path} lost scheduled workers")
            if {row["cell"] for row in rows} != {expected}:
                raise ValueError(f"{path} lost its position-balanced grid")
            if {row["worker"] for row in rows} != set(range(expected[0])):
                raise ValueError(f"{path} lost worker identities")
            for row in rows:
                base_cell = row["cell"][:-1]
                source_pairs.setdefault((base_cell, row["worker"]), {})[
                    row["pair_order"]] = row
        for base_cell in BASE_CELLS:
            completions = []
            for worker in range(base_cell[0]):
                orders = source_pairs.get((base_cell, worker))
                if orders is None or set(orders) != {0, 1}:
                    raise ValueError(f"{path} lost an A/B source-order pair")
                completion_ns = (
                    orders[1]["request_ns"] - orders[0]["control_ns"])
                if completion_ns <= 0:
                    raise ValueError(f"{path} has non-positive completion")
                completions.append(completion_ns)
            per_trial_completion.setdefault(base_cell, []).append(
                max(completions))
    if set(per_trial_completion) != set(BASE_CELLS):
        raise ValueError(f"{path} lost a base geometry")
    return {
        "run_id": next(iter(run_ids)),
        "samples_per_cell": len(trials),
        "median_maximum_worker_completion_ns": {
            cell: statistics.median(values)
            for cell, values in per_trial_completion.items()
        },
    }


def _axis_medians(cells, axis):
    groups = {}
    for cell in cells.values():
        groups.setdefault(cell[axis], []).append(cell["median_ns"])
    return {
        str(value): statistics.median(values)
        for value, values in sorted(groups.items())
    }


def build_report(paths):
    if len(paths) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"exactly {REQUIRED_INDEPENDENT_RUNS} raw runs are required")
    audits = tuple(_load_run(path) for path in paths)
    run_ids = tuple(audit["run_id"] for audit in audits)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("raw files must have independent run IDs")

    cells = {}
    for cell in BASE_CELLS:
        workers, row_count, row_bytes, stride_ratio, base_phase = cell
        medians = tuple(
            audit["median_maximum_worker_completion_ns"][cell]
            for audit in audits)
        cells[_cell_key(cell)] = {
            "active_aiv_workers": workers,
            "row_count": row_count,
            "row_bytes": row_bytes,
            "payload_bytes_per_input": row_count * row_bytes,
            "source_stride_bytes": row_bytes * stride_ratio,
            "source_stride_ratio": stride_ratio,
            "base_phase_bytes": base_phase,
            "medians_ns_by_run": dict(zip(run_ids, medians)),
            "median_ns": statistics.median(medians),
            "relative_median_spread": _relative_span(medians),
        }
    stable = all(
        cell["relative_median_spread"] <= MAXIMUM_RUN_MEDIAN_SPREAD
        for cell in cells.values())
    unstable_cell_keys = tuple(
        key for key, cell in cells.items()
        if cell["relative_median_spread"] > MAXIMUM_RUN_MEDIAN_SPREAD)
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": (
            "generic source-ordered q2 A/B GM-to-UB row-stride "
            "cohort completion"),
        "formula_status": "unfitted" if stable else "rejected_unstable",
        "forward_policy": (
            "Raw geometry cells are not Forward services. A future candidate "
            "must use one source-independent physical formula with active "
            "admission, row-request, byte and row-gap terms, then pass a "
            "predeclared held-out geometry validation."),
        "active_aiv_workers": list(ACTIVE_WORKERS),
        "row_counts": list(ROW_COUNTS),
        "row_bytes": list(ROW_BYTES),
        "stride_ratios": list(STRIDE_RATIOS),
        "base_phase_bytes": list(BASE_PHASE_BYTES),
        "queue_depth": 2,
        "independent_runs": len(audits),
        "samples_per_cell_per_run": audits[0]["samples_per_cell"],
        "maximum_relative_median_spread": MAXIMUM_RUN_MEDIAN_SPREAD,
        "stability_gate_passed": stable,
        "stability_failure_cell_keys": unstable_cell_keys,
        "axis_median_completion_ns": {
            "active_aiv_workers": _axis_medians(
                cells, "active_aiv_workers"),
            "row_count": _axis_medians(cells, "row_count"),
            "row_bytes": _axis_medians(cells, "row_bytes"),
            "source_stride_ratio": _axis_medians(
                cells, "source_stride_ratio"),
            "base_phase_bytes": _axis_medians(cells, "base_phase_bytes"),
        },
        "cells": cells,
        "mechanism_publishable": False,
        "production_or_evaluation_duration_consumed": False,
        "raw_artifacts": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in paths
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.raw)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
