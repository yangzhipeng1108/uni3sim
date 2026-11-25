#!/usr/bin/env python3
"""Validate the finite 4-pair-drained AIC MTE2 overlap domain."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


ACTIVE_AICS = 24
PAIR_BYTES = (128 * 64 + 64 * 256) * 2
DRAIN_INTERVAL = 4
TICK_NS = 20.0
FIT_PAIR_COUNTS = (16, 32, 64, 128, 256, 512)
HELD_OUT_PAIR_COUNTS = (24, 48, 96, 192, 384)
PAIR_COUNTS = tuple(sorted(FIT_PAIR_COUNTS + HELD_OUT_PAIR_COUNTS))
RUN_INDEX_COUNT = 3
MAX_WITHIN_COUNT_RELATIVE_SPAN = 0.10
MAX_VALIDATION_MAPE_PERCENT = 10.0
MAX_VALIDATION_APE_PERCENT = 20.0
RAW_FIELDS = (
    "run_id", "run_index", "order_position", "process_pid",
    "a_residency", "b_residency", "a_rows", "b_columns",
    "a_tile_bytes", "b_tile_bytes", "a_source_tiles", "b_source_tiles",
    "drain_interval", "block_dim", "iterations", "short_cycles",
    "short_wall_ns", "long_cycles", "long_wall_ns", "tick_ns",
    "cycles_per_pair", "ns_per_pair", "bandwidth_gbs", "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_order(run_index):
    cells = list(PAIR_COUNTS)
    rotation = run_index * 3 % len(cells)
    cells = cells[rotation:] + cells[:rotation]
    return cells if run_index % 2 == 0 else list(reversed(cells))


def _load_run(path, expected_run_index):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        raw_rows = list(reader)
    expected_counts = _expected_order(expected_run_index)
    if len(raw_rows) != len(expected_counts):
        raise ValueError(
            f"{path} has {len(raw_rows)} rows, expected {len(expected_counts)}")
    rows = []
    for position, (raw, pair_count) in enumerate(
            zip(raw_rows, expected_counts)):
        context = f"{path} row {position + 2}"
        expected = {
            "run_index": expected_run_index,
            "order_position": position,
            "a_rows": 128,
            "b_columns": 256,
            "a_tile_bytes": 128 * 64 * 2,
            "b_tile_bytes": 64 * 256 * 2,
            "a_source_tiles": ACTIVE_AICS,
            "b_source_tiles": ACTIVE_AICS,
            "drain_interval": DRAIN_INTERVAL,
            "block_dim": ACTIVE_AICS,
            "iterations": pair_count,
        }
        observed = {name: int(raw[name]) for name in expected}
        if (observed != expected or raw["a_residency"] != "resident"
                or raw["b_residency"] != "resident"
                or raw["validation"] != "pass"
                or int(raw["process_pid"]) <= 0):
            raise ValueError(f"{context} changed the frozen mechanism cell")
        short_cycles = int(raw["short_cycles"])
        long_cycles = int(raw["long_cycles"])
        short_wall_ns = int(raw["short_wall_ns"])
        long_wall_ns = int(raw["long_wall_ns"])
        tick_ns = float(raw["tick_ns"])
        ns_per_pair = float(raw["ns_per_pair"])
        bandwidth_gbs = float(raw["bandwidth_gbs"])
        if (short_cycles <= 0 or long_cycles <= short_cycles
                or short_wall_ns <= 0 or long_wall_ns <= short_wall_ns
                or not math.isclose(tick_ns, TICK_NS)):
            raise ValueError(f"{context} has invalid measured intervals")
        expected_ns_per_pair = (
            (long_cycles - short_cycles) * TICK_NS / (3 * pair_count))
        if (not math.isclose(
                    ns_per_pair, expected_ns_per_pair,
                    rel_tol=1e-9, abs_tol=1e-9)
                or not math.isclose(
                    bandwidth_gbs, PAIR_BYTES / expected_ns_per_pair,
                    rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"{context} has inconsistent derivation")
        rows.append({
            "run_id": raw["run_id"],
            "run_index": expected_run_index,
            "pair_count": pair_count,
            "program_span_ns": short_cycles * TICK_NS,
            "steady_ns_per_pair": ns_per_pair,
        })
    return rows


def _metrics(rows):
    apes = [
        abs(row["predicted_ns"] - row["measured_ns"])
        / row["measured_ns"] * 100.0
        for row in rows
    ]
    return {
        "count": len(apes),
        "mape_percent": sum(apes) / len(apes),
        "max_ape_percent": max(apes),
    }


def _require_digest(value, name, length):
    if (not isinstance(value, str) or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} must be {length}-character lowercase hex")


def build_report(raw_paths, source_commit, source_snapshot_sha256,
                 binary_sha256, device_id, *, host, container_name):
    if not host.strip() or not container_name.strip():
        raise ValueError("capture host and container identity must be nonempty")
    if len(raw_paths) != RUN_INDEX_COUNT:
        raise ValueError(f"exactly {RUN_INDEX_COUNT} raw runs are required")
    rows = []
    for run_index, path in enumerate(raw_paths):
        rows.extend(_load_run(path, run_index))
    run_ids = [row["run_id"] for row in rows]
    if len(set(run_ids)) != RUN_INDEX_COUNT:
        raise ValueError("run IDs must be unique")
    program_spans = {
        pair_count: statistics.median(
            row["program_span_ns"] for row in rows
            if row["pair_count"] == pair_count)
        for pair_count in PAIR_COUNTS
    }
    pair_services = {
        pair_count: statistics.median(
            row["steady_ns_per_pair"] for row in rows
            if row["pair_count"] == pair_count)
        for pair_count in PAIR_COUNTS
    }
    within_count_spans = {
        str(pair_count): (
            max(row["steady_ns_per_pair"] for row in rows
                if row["pair_count"] == pair_count)
            - min(row["steady_ns_per_pair"] for row in rows
                  if row["pair_count"] == pair_count)
        ) / pair_services[pair_count]
        for pair_count in PAIR_COUNTS
    }
    validation_service_ns = statistics.median(
        pair_services[pair_count] for pair_count in FIT_PAIR_COUNTS)
    if validation_service_ns <= 0.0:
        raise ValueError("finite pair service must be positive")
    validation_rows = [
        {
            "pair_count": pair_count,
            "measured_ns": pair_services[pair_count],
            "predicted_ns": validation_service_ns,
        }
        for pair_count in HELD_OUT_PAIR_COUNTS
    ]
    validation = _metrics(validation_rows)
    stability_gate = max(within_count_spans.values()) <= (
        MAX_WITHIN_COUNT_RELATIVE_SPAN)
    validation_gate = (
        validation["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and validation["max_ape_percent"] <= MAX_VALIDATION_APE_PERCENT)
    _require_digest(source_commit, "source_commit", 40)
    _require_digest(source_snapshot_sha256, "source_snapshot_sha256", 64)
    _require_digest(binary_sha256, "binary_sha256", 64)
    if isinstance(device_id, bool) or not isinstance(device_id, int):
        raise TypeError("device_id must be an integer")
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "service": "aic_mte2_operand_pair_finite_overlap_domain",
        "forward_change": {
            "parameter_fit_from_this_validation": False,
            "duration_parameter_published": False,
            "overlap_pair_count_range": [min(PAIR_COUNTS), max(PAIR_COUNTS)],
            "interpretation": (
                "The source-defined four-pair MTE2 drain remains stable over "
                "the measured finite program interval. Forward may use this "
                "evidence only to select the existing analytical L1-slab "
                "pipeline; measured service duration is not published."),
        },
        "publication_scope": {
            "dtype": "bf16",
            "source_residency": "l2_resident",
            "a_rows": 128,
            "a_columns": 64,
            "b_rows": 64,
            "b_columns": 256,
            "active_aic_groups": ACTIVE_AICS,
            "drain_interval_pairs": DRAIN_INTERVAL,
            "minimum_pairs_per_worker": min(PAIR_COUNTS),
            "maximum_pairs_per_worker": max(PAIR_COUNTS),
        },
        "measurement": {
            "fit_pair_counts": list(FIT_PAIR_COUNTS),
            "held_out_pair_counts": list(HELD_OUT_PAIR_COUNTS),
            "independent_runs": RUN_INDEX_COUNT,
            "median_program_span_ns": {
                str(key): value for key, value in program_spans.items()},
            "median_steady_ns_per_pair": {
                str(key): value for key, value in pair_services.items()},
            "validation_service_ns_per_pair": validation_service_ns,
            "within_count_relative_spans": within_count_spans,
            "validation": validation,
            "validation_rows": validation_rows,
            "stability_gate": stability_gate,
            "validation_gate": validation_gate,
            "mechanism_gate_passed": stability_gate and validation_gate,
        },
        "provenance": {
            "source_commit": source_commit,
            "source_snapshot_sha256": source_snapshot_sha256,
            "binary_sha256": binary_sha256,
            "host": host,
            "container": container_name,
            "physical_device_id": device_id,
            "raw_files": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in raw_paths
            ],
        },
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": stability_gate and validation_gate,
        "forward_eligible": stability_gate and validation_gate,
        "forward_eligibility_constraint": (
            "source-derived 128x64 A plus 64x256 B L2-resident stream with "
            "one PIPE_MTE2 drain after every four pairs, 24 active AICs, "
            "and a finite pair count inside the measured closed interval"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--raw", nargs=RUN_INDEX_COUNT, required=True,
                        type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        args.raw, args.source_commit, args.source_snapshot_sha256,
        args.binary_sha256, args.device_id, host=args.host, container_name=args.container_name)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    if not report["mechanism_publishable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
