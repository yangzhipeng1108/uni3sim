#!/usr/bin/env python3
"""Summarize the independent deterministic Split-K collective probe."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


CELLS = ((1, 0), (2, 0), (2, 1))
RUNS = 3
TRIALS = 8
SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_FILES = (
    "CMakeLists.txt",
    "deterministic_split_k_barrier_kernel.cpp",
    "deterministic_split_k_barrier_main.cpp",
    "deterministic_split_k_barrier_runner.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(root: Path) -> str:
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        payload = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _integer(row: dict[str, str], name: str) -> int:
    value = row[name]
    if re.fullmatch(r"-?[0-9]+", value) is None:
        raise ValueError(f"raw {name} must be an integer")
    return int(value)


def _cell_summary(rows: list[dict[str, str]], slabs: int, mode: int):
    selected = [
        row for row in rows
        if _integer(row, "slab_count") == slabs and _integer(row, "mode") == mode
    ]
    if len(selected) != RUNS * TRIALS:
        raise ValueError("raw cell count differs from the frozen campaign")
    per_run = []
    for run in range(RUNS):
        run_rows = [row for row in selected if _integer(row, "run_index") == run]
        if len(run_rows) != TRIALS:
            raise ValueError("raw run count differs from the frozen campaign")
        if {_integer(row, "trial") for row in run_rows} != set(range(TRIALS)):
            raise ValueError("raw trial axis differs from the frozen campaign")
        per_run.append({
            name: statistics.median(_integer(row, name) for row in run_rows)
            for name in (
                "max_actor_cycles",
                "max_aic_reverse_wait_cycles",
                "max_aic_sync_cycles",
                "max_aiv_first_sync_cycles",
                "max_aiv_second_sync_cycles",
            )
        })
    median_cycles = {
        name: statistics.median(run[name] for run in per_run)
        for name in per_run[0]
    }
    return {
        "slab_count": slabs,
        "mode": "source_collective" if mode == 0 else "flags_and_pipe_only",
        "per_run_median_cycles": per_run,
        "median_cycles": median_cycles,
        "median_ns": {name: value * 20.0 for name, value in median_cycles.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--runner-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--cann-source-sha256", required=True)
    args = parser.parse_args()
    if not args.container_name.strip():
        parser.error("--container-name must be nonempty")
    if re.fullmatch(r"[0-9a-f]{40}", args.baseline) is None:
        parser.error("--baseline must be a lowercase commit")
    for name, value in (
            ("binary SHA256", args.binary_sha256),
            ("CANN source SHA256", args.cann_source_sha256)):
        if SHA256.fullmatch(value) is None:
            parser.error(f"{name} is invalid")
    if args.output.exists():
        raise FileExistsError("reference output already exists")
    with args.raw.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_columns = {
        "run_id", "run_index", "trial", "slab_count", "mode",
        "max_actor_cycles", "max_aic_reverse_wait_cycles",
        "max_aic_sync_cycles", "max_aiv_first_sync_cycles",
        "max_aiv_second_sync_cycles", "host_wall_ns", "validation",
    }
    if not rows or set(rows[0]) != expected_columns:
        raise ValueError("raw CSV schema changed")
    if len(rows) != len(CELLS) * RUNS * TRIALS:
        raise ValueError("raw row count differs from the frozen campaign")
    run_ids = {row["run_id"] for row in rows}
    if len(run_ids) != 1:
        raise ValueError("raw capture must contain one run ID")
    for row in rows:
        if row["validation"] != "pass":
            raise ValueError("raw capture contains a failed cell")
        if _integer(row, "host_wall_ns") <= 0:
            raise ValueError("raw host interval must be positive")
        if (_integer(row, "slab_count"), _integer(row, "mode")) not in CELLS:
            raise ValueError("raw capture contains an unsupported cell")
    summaries = {
        "one_slab_source_collective": _cell_summary(rows, 1, 0),
        "two_slab_source_collective": _cell_summary(rows, 2, 0),
        "two_slab_flags_and_pipe_only": _cell_summary(rows, 2, 1),
    }
    complete = summaries["two_slab_source_collective"]["median_ns"][
        "max_actor_cycles"]
    control = summaries["two_slab_flags_and_pipe_only"]["median_ns"][
        "max_actor_cycles"]
    collective_excess = complete - control
    if collective_excess <= 0.0:
        raise ValueError("collective completion must exceed the flag-only control")
    root = Path(__file__).resolve().parent
    payload = {
        "schema_version": (
            "midlayer.ascend_910b."
            "deterministic_split_k_collective_barrier.v1"),
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "cann_deterministic_split_k_collective_barrier",
        "duration_unit": "ns_per_two_slab_24aic_48aiv_control_sequence",
        "production_or_evaluation_duration_consumed": False,
        "source_program": {
            "dtype": "bf16",
            "active_aic_groups": 24,
            "active_aiv_workers": 48,
            "n_slabs": 2,
            "scheduler_policy": "l2_split_matmul",
            "sequence": (
                "AIC_PIPE_FIX_flag_to_paired_AIV; all_AIV_wait_then_SyncAll;"
                "PIPE_ALL; all_AIV_PIPE_MTE3_flag_back; next_AIC_wait_then_"
                "SyncAll; PIPE_ALL"),
        },
        "control_comparison": {
            "source_collective_complete_ns": complete,
            "same_flag_and_pipe_control_ns": control,
            "collective_lifecycle_excess_ns": collective_excess,
            "interpretation": (
                "The excess measures only the source-mandatory all-AIV and "
                "all-AIC SyncAll lifecycle for the second N slab. The probe "
                "does not contain production GEMM arithmetic or traffic."),
        },
        "measurement_quality": {
            "independent_runs": RUNS,
            "trials_per_cell": TRIALS,
            "statistic": "median_of_run_medians",
            "validated_rows": len(rows),
            "failed_rows": 0,
            "system_counter": {
                "frequency_hz": 50_000_000,
                "tick_ns": 20.0,
            },
            "cells": summaries,
        },
        "source_binding": {
            "cann_source_path": (
                "opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/"
                "ascendc/3rd/mat_mul_v3/op_kernel/"
                "mat_mul_deterministic_splitk_kernel.h"),
            "cann_source_sha256": args.cann_source_sha256,
            "aic_symbols": ["Mc2MatMulMultiCoreSplitKDivideL2cache"],
            "aiv_symbols": ["ReduceKInUbL2cache"],
        },
        "exclusions": [
            "operand traffic", "cube", "fixpipe", "workspace",
            "aiv reduction body", "output storage", "production duration",
        ],
        "provenance": {
            "measurement_baseline_commit": args.baseline,
            "run_id": next(iter(run_ids)),
            "source_snapshot_sha256": _source_snapshot(root),
            "source_sha256": {name: _sha256(root / name) for name in SOURCE_FILES},
            "binary_sha256": args.binary_sha256,
            "raw_path": (
                "profiler_output/microbench/ascend_910b/"
                "deterministic_splitk_barrier_d1d843589_20260916_d4/raw.csv"),
            "raw_sha256": _sha256(args.raw),
            "runner_log_path": (
                "profiler_output/microbench/ascend_910b/"
                "deterministic_splitk_barrier_d1d843589_20260916_d4/run.log"),
            "runner_log_sha256": _sha256(args.runner_log),
            "physical_device_id": 4,
            "container": args.container_name,
        },
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
