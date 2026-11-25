#!/usr/bin/env python3
"""Validate and summarize FIA BlockReduceSum mechanism captures."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from block_reduce_sum_runner import (
    CELLS,
    COLUMN_ALIGNMENTS,
    COMMIT_PATTERN,
    EXPECTED_PRODUCTION_SOURCE_SHA256,
    ITERATIONS,
    MAX_RELATIVE_TRIAL_SPREAD,
    RAW_FIELDS,
    ROW_CHUNKS,
    RUN_ID_PATTERN,
    SHA256_PATTERN,
    SOURCE_FILENAMES,
    SPECIAL_COLUMNS,
    SYSTEM_COUNTER_TICK_NS,
    TAIL_COLUMNS,
    TRIALS,
    _ordered_cells,
    _sha256,
)


PROVENANCE_FIELDS = {
    "schema", "run_id", "captured_at_utc", "source_commit",
    "source_identity_mode", "source_snapshot_sha256", "source_files",
    "binary", "cann", "static_instruction_evidence", "device",
    "idle_telemetry", "clock", "measurement_scope",
    "production_or_evaluation_duration_consumed", "campaign", "capture",
}


def _load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _require_exact_mapping(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{label} has an invalid schema")


def _validate_provenance(capture_dir):
    path = capture_dir / "provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"capture provenance is missing: {path}")
    provenance = _load_json(path)
    _require_exact_mapping(provenance, PROVENANCE_FIELDS, "provenance")
    if provenance["schema"] != "ascend_910b_fia_block_reduce_sum_provenance_v1":
        raise ValueError("unsupported block-reduce-sum provenance schema")
    if provenance["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError("capture consumed forbidden production duration")
    if RUN_ID_PATTERN.fullmatch(provenance["run_id"]) is None:
        raise ValueError("capture run ID is invalid")
    if COMMIT_PATTERN.fullmatch(provenance["source_commit"]) is None:
        raise ValueError("capture source commit is invalid")
    if SHA256_PATTERN.fullmatch(
            provenance["source_snapshot_sha256"]) is None:
        raise ValueError("capture source snapshot identity is invalid")
    captured_at = datetime.fromisoformat(provenance["captured_at_utc"])
    if captured_at.tzinfo is None or captured_at.utcoffset().total_seconds() != 0:
        raise ValueError("capture timestamp must be timezone-aware UTC")
    if set(provenance["source_files"]) != set(SOURCE_FILENAMES):
        raise ValueError("capture source-file set changed")
    if any(
            SHA256_PATTERN.fullmatch(value) is None
            for value in provenance["source_files"].values()):
        raise ValueError("capture contains an invalid source-file identity")
    if provenance["clock"] != {
            "source": "GetSystemCycle",
            "frequency_hz": 50_000_000,
            "tick_ns": SYSTEM_COUNTER_TICK_NS}:
        raise ValueError("capture clock contract changed")
    campaign = provenance["campaign"]
    expected_campaign = {
        "row_chunks": list(ROW_CHUNKS),
        "tail_columns": list(TAIL_COLUMNS),
        "special_columns": list(SPECIAL_COLUMNS),
        "column_alignments": list(COLUMN_ALIGNMENTS),
        "iterations": ITERATIONS,
        "trials": TRIALS,
        "cell_count": len(CELLS),
        "order": "rotated_balanced_forward_reverse",
        "maximum_relative_trial_spread": MAX_RELATIVE_TRIAL_SPREAD,
    }
    if campaign != expected_campaign:
        raise ValueError("capture campaign dimensions changed")
    production_source = provenance["cann"]["production_source"]
    if production_source["sha256"] != EXPECTED_PRODUCTION_SOURCE_SHA256:
        raise ValueError("capture production source identity changed")
    if provenance["static_instruction_evidence"]["disassembly_claimed"] is not False:
        raise ValueError("capture makes an unsupported disassembly claim")
    device = provenance["device"]
    if (device["physical_device_id"] not in (0, 1)
            or device["logical_device_id"] != 0):
        raise ValueError("capture used the wrong physical/logical NPU mapping")
    for moment in ("pre", "post"):
        values = provenance["idle_telemetry"][moment]["values"]
        if set(values) != {
                "aic_percent", "aiv_percent", "hbm_bandwidth_percent"}:
            raise ValueError("capture idle telemetry schema changed")
        if any(value != 0 for value in values.values()):
            raise ValueError("capture idle telemetry is nonzero")
    raw_path = capture_dir / provenance["capture"]["raw"]
    if not raw_path.is_file():
        raise FileNotFoundError(f"capture raw CSV is missing: {raw_path}")
    if _sha256(raw_path) != provenance["capture"]["raw_sha256"]:
        raise ValueError("capture raw CSV identity changed")
    if provenance["capture"]["raw_rows"] != TRIALS * len(CELLS):
        raise ValueError("capture raw row count contract changed")
    return provenance, raw_path


def _decode_row(raw, path):
    if raw["validation"] != "pass":
        raise ValueError(f"{path} contains failed validation")
    integers = {
        field: int(raw[field])
        for field in (
            "trial", "order_position", "process_pid", "row_chunk_round",
            "column_count", "column_count_round", "column_alignment",
            "iterations", "system_cycles",
        )
    }
    floats = {
        field: float(raw[field])
        for field in (
            "per_iteration_ns", "first_row_sum", "last_row_sum")
    }
    if any(not math.isfinite(value) for value in floats.values()):
        raise ValueError(f"{path} contains non-finite values")
    if integers["process_pid"] <= 0 or integers["system_cycles"] <= 0:
        raise ValueError(f"{path} contains invalid positive values")
    expected_ns = (
        integers["system_cycles"] * SYSTEM_COUNTER_TICK_NS
        / integers["iterations"])
    if not math.isclose(
            floats["per_iteration_ns"], expected_ns,
            rel_tol=0.0, abs_tol=5.1e-10):
        raise ValueError(f"{path} contains inconsistent derived time")
    if (floats["first_row_sum"] != integers["column_count"]
            or floats["last_row_sum"] != integers["column_count"]):
        raise ValueError(f"{path} contains inconsistent row sums")
    cell = (
        raw["branch"], integers["row_chunk_round"],
        integers["column_count"], integers["column_count_round"],
        integers["column_alignment"],
    )
    if cell not in CELLS:
        raise ValueError(f"{path} contains an unknown campaign cell")
    return {
        "run_id": raw["run_id"],
        "trial": integers["trial"],
        "order_direction": raw["order_direction"],
        "order_position": integers["order_position"],
        "cell": cell,
        "per_iteration_ns": floats["per_iteration_ns"],
    }


def _load_capture(capture_dir):
    provenance, raw_path = _validate_provenance(capture_dir)
    rows = []
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{raw_path} has an invalid raw schema")
        for raw in reader:
            rows.append(_decode_row(raw, raw_path))
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError(f"{raw_path} has an incomplete row set")
    if {row["run_id"] for row in rows} != {provenance["run_id"]}:
        raise ValueError("raw run identity does not match provenance")
    by_trial = {}
    for row in rows:
        trial = row["trial"]
        if trial not in by_trial:
            by_trial[trial] = []
        by_trial[trial].append(row)
    if set(by_trial) != set(range(1, TRIALS + 1)):
        raise ValueError("capture does not contain the frozen trial set")
    for trial in range(1, TRIALS + 1):
        direction, expected_cells = _ordered_cells(trial - 1)
        ordered_rows = sorted(
            by_trial[trial], key=lambda row: row["order_position"])
        if [row["order_position"] for row in ordered_rows] != list(
                range(len(CELLS))):
            raise ValueError(f"trial {trial} has invalid order positions")
        if any(row["order_direction"] != direction for row in ordered_rows):
            raise ValueError(f"trial {trial} changed order direction")
        if tuple(row["cell"] for row in ordered_rows) != expected_cells:
            raise ValueError(f"trial {trial} changed frozen cell order")
    return provenance, rows


def _summaries(rows):
    by_cell = {cell: [] for cell in CELLS}
    for row in rows:
        by_cell[row["cell"]].append(row["per_iteration_ns"])
    summaries = []
    for cell in CELLS:
        values = by_cell[cell]
        if not values:
            raise ValueError("one campaign cell has no samples")
        median = statistics.median(values)
        if median <= 0:
            raise ValueError("one campaign cell has non-positive service")
        relative_spread = (max(values) - min(values)) / median
        branch, rows_round, columns, columns_round, alignment = cell
        summaries.append({
            "branch": branch,
            "row_chunk_round": rows_round,
            "column_count": columns,
            "column_count_round": columns_round,
            "column_alignment": alignment,
            "sample_count": len(values),
            "median_service_ns": median,
            "minimum_service_ns": min(values),
            "maximum_service_ns": max(values),
            "relative_trial_spread": relative_spread,
            "stable": relative_spread <= MAX_RELATIVE_TRIAL_SPREAD,
        })
    return summaries


def _write_json_exclusive(path, payload):
    if path.exists():
        raise FileExistsError(f"report output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary report output exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenances = []
    rows = []
    run_ids = set()
    source_snapshots = set()
    production_sources = set()
    for capture in args.capture:
        resolved_capture = capture.resolve()
        provenance, capture_rows = _load_capture(resolved_capture)
        if provenance["run_id"] in run_ids:
            raise ValueError("capture list contains a duplicate run ID")
        run_ids.add(provenance["run_id"])
        source_snapshots.add(provenance["source_snapshot_sha256"])
        production_sources.add(
            provenance["cann"]["production_source"]["sha256"])
        provenances.append({
            "directory": str(resolved_capture),
            "run_id": provenance["run_id"],
            "provenance_sha256": _sha256(
                resolved_capture / "provenance.json"),
            "raw_sha256": provenance["capture"]["raw_sha256"],
        })
        rows.extend(capture_rows)
    if len(source_snapshots) != 1:
        raise ValueError("captures use different benchmark snapshots")
    if len(production_sources) != 1:
        raise ValueError("captures use different production source revisions")
    summaries = _summaries(rows)
    unstable = [summary for summary in summaries if not summary["stable"]]
    report = {
        "schema": "ascend_910b_fia_block_reduce_sum_report_v1",
        "mechanism": (
            "CANN 8.5.1 FIA CalcLocalRowSum BlockReduceSum/Add service"
        ),
        "capture_count": len(provenances),
        "captures": provenances,
        "source_snapshot_sha256": next(iter(source_snapshots)),
        "production_source_sha256": next(iter(production_sources)),
        "production_or_evaluation_duration_consumed": False,
        "gate": {
            "maximum_relative_trial_spread": MAX_RELATIVE_TRIAL_SPREAD,
            "unstable_cell_count": len(unstable),
            "mechanism_publishable": not unstable,
        },
        "coverage": {
            "special_512": sum(
                summary["branch"] == "special_512" for summary in summaries),
            "special_256": sum(
                summary["branch"] == "special_256" for summary in summaries),
            "tail": sum(
                summary["branch"] == "tail" for summary in summaries),
            "row_chunks": sorted({
                summary["row_chunk_round"] for summary in summaries}),
            "column_alignments": sorted({
                summary["column_alignment"] for summary in summaries}),
        },
        "services": summaries,
    }
    _write_json_exclusive(args.output.resolve(), report)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
