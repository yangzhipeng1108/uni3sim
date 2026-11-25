#!/usr/bin/env python3
"""Validate and summarize the frozen MatMulV3 BF16 store campaign."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from gemm_v3_store_runner import (
    CELLS,
    LONG_INVOCATIONS,
    RAW_FIELDS,
    SHORT_INVOCATIONS,
)


STABILITY_LIMIT = 0.10
SYSTEM_COUNTER_TICK_NS = 20.0
DTYPE_BYTES = {"bf16": 2, "fp32": 4}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_key(row):
    return (row["dtype"], int(row["m"]), int(row["n"]), int(row["block_dim"]))


def _cycles(text, block_dim):
    values = tuple(int(value) for value in text.split(";"))
    if len(values) != block_dim or any(value <= 0 for value in values):
        raise ValueError("cycle vector violates the block-dimension contract")
    return values


def _load(provenance_path, raw_path):
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {
        "schema", "run_id", "captured_at_utc", "owner", "hardware",
        "physical_device_id", "logical_device_id", "baseline_commit",
        "source_snapshot_sha256", "source_files", "cann_version",
        "cann_identities", "v3_tiling_key", "v3_kernel_type", "v3_tile",
        "measurement_scope", "production_or_evaluation_duration_consumed",
        "idle_command", "idle_telemetry", "idle_telemetry_sha256", "build",
        "commands", "trials", "cells", "raw_output", "raw_sha256", "raw_rows",
    }
    if set(provenance) != required:
        raise ValueError("provenance schema drifted")
    if (
        provenance["schema"] != "ascend_910b_gemm_v3_store_provenance_v1"
        or provenance["hardware"] != "ascend_910b"
        or provenance["logical_device_id"] != 0
        or provenance["v3_tiling_key"] != 65536
        or provenance["v3_kernel_type"] != "AIC_ONLY"
        or provenance["v3_tile"] != {"m": 128, "n": 256, "k": 64}
        or provenance["production_or_evaluation_duration_consumed"] is not False
        or provenance["trials"] != 3
        or provenance["cells"] != [list(cell) for cell in CELLS]
        or provenance["raw_rows"] != 3 * len(CELLS)
        or provenance["raw_sha256"] != _sha256(raw_path)
    ):
        raise ValueError("provenance violates the frozen campaign contract")
    with raw_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames) != RAW_FIELDS:
            raise ValueError("raw CSV schema drifted")
        rows = list(reader)
    if len(rows) != 3 * len(CELLS):
        raise ValueError("raw CSV is incomplete")
    return provenance, rows


def _summarize(provenance, rows):
    grouped = {}
    for row in rows:
        if row["run_id"] != provenance["run_id"] or row["validation"] != "exact":
            raise ValueError("raw row identity or validation drifted")
        trial = int(row["trial"])
        if trial not in (1, 2, 3):
            raise ValueError("raw row has an invalid trial")
        expected_direction = "forward" if trial % 2 == 1 else "reverse"
        if row["order_direction"] != expected_direction:
            raise ValueError("trial order direction drifted")
        key = _cell_key(row)
        if key not in {(dtype, m, n, block) for dtype, _id, m, n, block in CELLS}:
            raise ValueError("raw row contains an unregistered cell")
        dtype, m, n, block_dim = key
        expected_bytes = m * n * DTYPE_BYTES[dtype]
        if (
            int(row["output_bytes"]) != expected_bytes
            or int(row["l0c_bytes"]) != m * n * 4
            or int(row["short_invocations"]) != SHORT_INVOCATIONS
            or int(row["long_invocations"]) != LONG_INVOCATIONS
        ):
            raise ValueError("raw shape/byte/interval contract drifted")
        short = _cycles(row["short_cycles"], block_dim)
        long = _cycles(row["long_cycles"], block_dim)
        service = tuple(
            (long_value - short_value)
            / (LONG_INVOCATIONS - SHORT_INVOCATIONS)
            for short_value, long_value in zip(short, long)
        )
        if any(value <= 0 for value in service):
            raise ValueError("differenced store service must be positive")
        if key not in grouped:
            grouped[key] = {}
        if trial in grouped[key]:
            raise ValueError("duplicate trial for cell")
        grouped[key][trial] = {
            "median_cycles_per_tile": statistics.median(service),
            "maximum_cycles_per_tile": max(service),
        }
    if len(grouped) != len(CELLS):
        raise ValueError("campaign cell set is incomplete")

    summaries = []
    for dtype, _dtype_id, m, n, block_dim in CELLS:
        key = (dtype, m, n, block_dim)
        if set(grouped[key]) != {1, 2, 3}:
            raise ValueError("campaign cell lacks exactly three trials")
        trial_cycles = [
            grouped[key][trial]["median_cycles_per_tile"]
            for trial in (1, 2, 3)
        ]
        median_cycles = statistics.median(trial_cycles)
        relative_span = (max(trial_cycles) - min(trial_cycles)) / median_cycles
        if relative_span > STABILITY_LIMIT:
            raise ValueError(
                f"unstable cell {key}: relative span {relative_span:.6f}")
        output_bytes = m * n * DTYPE_BYTES[dtype]
        summaries.append({
            "dtype": dtype,
            "m": m,
            "n": n,
            "block_dim": block_dim,
            "output_bytes_per_tile": output_bytes,
            "l0c_source_bytes_per_tile": m * n * 4,
            "trial_median_cycles_per_tile": trial_cycles,
            "median_cycles_per_tile": median_cycles,
            "median_ns_per_tile": median_cycles * SYSTEM_COUNTER_TICK_NS,
            "output_bytes_per_cycle": output_bytes / median_cycles,
            "relative_trial_span": relative_span,
        })
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenance, rows = _load(args.provenance, args.raw)
    cells = _summarize(provenance, rows)
    target = [
        cell for cell in cells
        if cell["dtype"] == "bf16" and cell["m"] == 128
        and cell["n"] == 256 and cell["block_dim"] == 24
    ]
    if len(target) != 1:
        raise AssertionError("missing active-AIC target cell")
    reference = {
        "schema": "ascend_910b_gemm_v3_bf16_store_reference_v1",
        "hardware": "ascend_910b",
        "cann_version": provenance["cann_version"],
        "source_snapshot_sha256": provenance["source_snapshot_sha256"],
        "source_files": provenance["source_files"],
        "cann_identities": provenance["cann_identities"],
        "benchmark_binary_sha256": provenance["build"]["binary_sha256"],
        "benchmark_archive_sha256": provenance["build"]["archive_sha256"],
        "run_id": provenance["run_id"],
        "physical_device_id": provenance["physical_device_id"],
        "measurement_scope": provenance["measurement_scope"],
        "production_or_evaluation_duration_consumed": False,
        "system_counter_tick_ns": SYSTEM_COUNTER_TICK_NS,
        "stability_limit": STABILITY_LIMIT,
        "cells": cells,
        "result": {
            "status": "pass",
            "target_active_aic_median_cycles_per_tile": target[0][
                "median_cycles_per_tile"],
            "target_active_aic_median_ns_per_tile": target[0][
                "median_ns_per_tile"],
        },
        "limitations": [
            "This is an independent mechanism measurement, not a production MatMul duration.",
            "It measures one initialized L0C tile repeatedly stored to distinct GM ranges.",
            "The producer-to-Fixpipe M_FIX handoff is established before timing; the timed region includes CANN's public pre-issue PIPE_FIX barrier and one final completion barrier.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(reference, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
