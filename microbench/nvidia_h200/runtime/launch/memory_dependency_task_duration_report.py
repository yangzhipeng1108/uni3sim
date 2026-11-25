#!/usr/bin/env python3
"""Freeze the H200 warm default-cache memory-dependency task floor.

H200 owner copy of the report of commit 9eb2bba7b
(microbench/nvidia_a800/runtime/launch/memory_dependency_task_duration_report.py
at that commit), which produced the ``isolated_warm_default_cache_memory_
dependency_task_v1`` model.  Kept identical: the summary schema, the fixed
feature form ``base + fanout(grid >= 3) + grid * cta + ceil(block/32) * warp``,
the relative least-squares fit on trials 1-2, the lower-envelope rule (subtract
the largest training-cell excess over any observed sample from the intercept
only), trial-3 validation metrics and the four publication gates, and the
report keys.

Changed for H200 (hardware coupling / capture tooling only):
  - grids 107, 108 -> 131, 132 and the maximum grid 108 -> 132 (132 SMs);
  - the A800 frozen-evidence audit (its SHA256SUMS manifest, telemetry files
    and SQLite re-join against a 1410 MHz locked clock) is replaced by the H200
    capture of ``nvtx_kernel_sweep_runner.py``: the NVTX->launch->kernel join
    and label/geometry check happen in that runner, the per-cell minima come
    from its ``cell_minimum_by_trial.csv``, and the clock gate is its
    per-trial busy-clock check (unlocked clock; every busy sample >= 1980 MHz);
  - the SASS contract is checked on the sm_90 dump (``LDG.E Rx, desc[..]``
    syntax) of the binary that was run;
  - a failed publication gate is recorded (``publishable: false``) instead of
    aborting, so the H200 outcome is reported either way.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


EXPECTED_SUMMARY_FIELDS = (
    "trial", "grid", "block", "sample_count", "median_duration_ns")
EXPECTED_MINIMUM_FIELDS = ("trial", "grid", "block", "minimum_duration_ns")
EXPECTED_TRIALS = (1, 2, 3)
TRAINING_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
EXPECTED_GRIDS = (1, 2, 3, 4, 5, 8, 16, 32, 64, 131, 132)
EXPECTED_BLOCKS = (1, 16, 31, 32, 33, 64, 65, 128, 256, 512, 1024)
EXPECTED_SAMPLES_PER_CELL = 200
PARALLEL_WIDTH = 32
FANOUT_GRID_THRESHOLD = 3
BOOST_CLOCK_MHZ = 1980
EXPECTED_HARDWARE_NAME = "NVIDIA H200"
MAXIMUM_GRID = 132
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SOURCE = "microbench/nvidia_h200/runtime/launch/memory_dependency_task_duration.cu"
RUNNER = "microbench/nvidia_h200/runtime/launch/nvtx_kernel_sweep_runner.py"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value, name):
    parsed = int(value)
    if str(parsed) != value or parsed <= 0:
        raise ValueError(f"{name} must be a canonical positive integer")
    return parsed


def _positive_float(value, name):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _expected_cells():
    return {
        (trial, grid, block)
        for trial in EXPECTED_TRIALS
        for grid in EXPECTED_GRIDS
        for block in EXPECTED_BLOCKS
    }


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_SUMMARY_FIELDS:
            raise ValueError("H200 memory-task summary fields changed")
        rows = []
        identities = set()
        for ordinal, raw in enumerate(reader, start=2):
            row = {
                "trial": _positive_int(raw["trial"], f"line {ordinal}.trial"),
                "grid": _positive_int(raw["grid"], f"line {ordinal}.grid"),
                "block": _positive_int(raw["block"], f"line {ordinal}.block"),
                "sample_count": _positive_int(
                    raw["sample_count"], f"line {ordinal}.sample_count"),
                "median_duration_ns": _positive_float(
                    raw["median_duration_ns"],
                    f"line {ordinal}.median_duration_ns"),
            }
            identity = (row["trial"], row["grid"], row["block"])
            if identity in identities:
                raise ValueError(f"duplicate H200 memory-task cell: {identity}")
            identities.add(identity)
            rows.append(row)
    if identities != _expected_cells():
        raise ValueError("H200 memory-task matrix is incomplete")
    if any(row["sample_count"] != EXPECTED_SAMPLES_PER_CELL for row in rows):
        raise ValueError("every H200 memory-task cell must contain 200 samples")
    return tuple(rows)


def read_minima(path):
    minima = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_MINIMUM_FIELDS:
            raise ValueError("H200 memory-task minimum fields changed")
        for raw in reader:
            identity = (int(raw["trial"]), int(raw["grid"]), int(raw["block"]))
            if identity in minima:
                raise ValueError(f"duplicate minimum cell {identity}")
            minima[identity] = _positive_float(
                raw["minimum_duration_ns"], "minimum_duration_ns")
    if set(minima) != _expected_cells():
        raise ValueError("H200 memory-task minimum matrix is incomplete")
    return minima


def _features(row):
    return (
        1.0,
        float(row["grid"] >= FANOUT_GRID_THRESHOLD),
        float(row["grid"]),
        float(math.ceil(row["block"] / PARALLEL_WIDTH)),
    )


def _predict(row, model):
    return (
        model["base_ns"]
        + (model["fanout_activation_ns"]
           if row["grid"] >= FANOUT_GRID_THRESHOLD else 0.0)
        + row["grid"] * model["cta_ns"]
        + math.ceil(row["block"] / PARALLEL_WIDTH) * model["warp_ns"]
    )


def fit_floor_model(rows, minima):
    training = tuple(
        row for row in rows if row["trial"] in TRAINING_TRIALS)
    design = np.asarray([_features(row) for row in training], dtype=float)
    training_medians = np.asarray(
        [row["median_duration_ns"] for row in training], dtype=float)
    relative_design = design / training_medians[:, None]
    coefficients, _, _, _ = np.linalg.lstsq(
        relative_design, np.ones(len(training)), rcond=None)
    if np.any(coefficients <= 0) or not np.all(np.isfinite(coefficients)):
        raise ValueError(
            "H200 memory-task model requires finite positive services: "
            f"{coefficients.tolist()}")
    unconstrained = dict(zip(
        ("base_ns", "fanout_activation_ns", "cta_ns", "warp_ns"),
        map(float, coefficients),
    ))
    guard_ns = max(
        _predict(row, unconstrained)
        - minima[(row["trial"], row["grid"], row["block"])]
        for row in training
    )
    floor = dict(unconstrained)
    floor["base_ns"] -= guard_ns
    if floor["base_ns"] <= 0:
        raise ValueError("H200 memory-task lower-envelope base is not positive")
    return floor, unconstrained, float(guard_ns)


def _metrics(rows, model):
    observed = np.asarray(
        [row["median_duration_ns"] for row in rows], dtype=float)
    predicted = np.asarray([_predict(row, model) for row in rows])
    error = predicted - observed
    absolute_percentage = np.abs(error) / observed
    return {
        "cell_count": len(rows),
        "mape_percent": float(np.mean(absolute_percentage) * 100.0),
        "wmae_percent": float(
            np.sum(np.abs(error)) / np.sum(observed) * 100.0),
        "signed_mean_percentage_error": float(
            np.mean(error / observed) * 100.0),
        "p95_absolute_percentage_error": float(
            np.percentile(absolute_percentage, 95) * 100.0),
        "maximum_absolute_percentage_error": float(
            np.max(absolute_percentage) * 100.0),
    }


def _lower_envelope_metrics(rows, minima, model):
    overages = np.asarray([
        _predict(row, model) - minima[(row["trial"], row["grid"], row["block"])]
        for row in rows
    ])
    return {
        "cell_count": len(rows),
        "cells_above_observed_minimum": int(np.sum(overages > 1e-9)),
        "maximum_above_observed_minimum_ns": float(max(0.0, np.max(overages))),
        "minimum_margin_below_observed_minimum_ns": float(-np.max(overages)),
    }


def _audit_sass(sass_path):
    lines = sass_path.read_text(encoding="utf-8").splitlines()
    markers = [
        offset for offset, line in enumerate(lines)
        if "Function :" in line and "memory_dependency_taskEPKjPj" in line
    ]
    if len(markers) != 1:
        raise ValueError("SASS must contain exactly one target memory-task symbol")
    start = markers[0]
    end = next(
        (offset for offset in range(start + 1, len(lines))
         if ("Function :" in lines[offset]
             or lines[offset].startswith("Fatbin ptx code:"))),
        len(lines),
    )
    target = "\n".join(lines[start:end])
    if "EF_CUDA_SM90" not in target:
        raise ValueError("target SASS is not sm_90")
    load = re.findall(r"LDG\.E (R\d+), desc\[UR\d+\]\[R\d+\.64\]", target)
    if target.count("LDG.") != 1 or target.count("STG.") != 1 or len(load) != 1:
        raise ValueError("target SASS must hold exactly one LDG.E and one STG.E")
    dependent = re.findall(rf"IADD3 (R\d+), {load[0]}, 0x1, RZ", target)
    if len(dependent) != 1 or len(re.findall(
            rf"STG\.E desc\[UR\d+\]\[R\d+\.64\], {dependent[0]}", target)) != 1:
        raise ValueError("target SASS lacks the LDG.E -> IADD3 -> STG.E chain")
    if any(instruction in target for instruction in (
            "LDG.E.STRONG", "MEMBAR", "BAR.SYNC", "ATOM.", "RED.")):
        raise ValueError("target SASS has an unsupported mechanism")


def build_report(run_dir, sass_path, binary_sha256, baseline_commit, compiler):
    summary_path = run_dir / "summary_by_trial.csv"
    minimum_path = run_dir / "cell_minimum_by_trial.csv"
    capture_path = run_dir / "capture.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    if not all(trial["accepted"] for trial in capture["trials"]):
        raise ValueError("a capture trial failed the 1980 MHz clock gate")
    name, uuid, driver = (field.strip() for field in capture["device"].split(","))
    if name != EXPECTED_HARDWARE_NAME:
        raise ValueError(f"unexpected device {name}")
    _audit_sass(sass_path)
    rows = read_summary(summary_path)
    minima = read_minima(minimum_path)
    model, unconstrained, guard_ns = fit_floor_model(rows, minima)
    training = tuple(row for row in rows if row["trial"] in TRAINING_TRIALS)
    validation = tuple(row for row in rows if row["trial"] in VALIDATION_TRIALS)
    training_metrics = _metrics(training, model)
    validation_metrics = _metrics(validation, model)
    training_envelope = _lower_envelope_metrics(training, minima, model)
    validation_envelope = _lower_envelope_metrics(validation, minima, model)
    gates = {
        "training_never_above_observed_minimum": (
            training_envelope["cells_above_observed_minimum"] == 0),
        "validation_never_above_observed_minimum": (
            validation_envelope["cells_above_observed_minimum"] == 0),
        "validation_mape_at_most_5_percent": (
            validation_metrics["mape_percent"] <= 5.0),
        "validation_max_at_most_15_percent": (
            validation_metrics["maximum_absolute_percentage_error"] <= 15.0),
    }
    artifacts = {
        "summary_by_trial.csv": _sha256(summary_path),
        "cell_minimum_by_trial.csv": _sha256(minimum_path),
        "capture.json": _sha256(capture_path),
        "memory_dependency_task_duration.sass": _sha256(sass_path),
    }
    manifest_json = json.dumps(
        artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    maximum_warps = MAXIMUM_GRID * (1024 // PARALLEL_WIDTH)
    return {
        "schema_version": 2,
        "hardware_id": "nvidia_h200",
        "mechanism": "warm_default_cache_memory_dependency_task_floor",
        "duration_source": "nsys_gpu_kernel",
        "production_or_evaluation_duration_consumed": False,
        "input": {
            "summary_path": str(summary_path),
            "summary_sha256": artifacts["summary_by_trial.csv"],
            "measurement_path": str(minimum_path),
            "measurement_sha256": artifacts["cell_minimum_by_trial.csv"],
            "training_trials": list(TRAINING_TRIALS),
            "validation_trials": list(VALIDATION_TRIALS),
            "samples_per_cell": EXPECTED_SAMPLES_PER_CELL,
        },
        "evidence": {
            "baseline_commit": baseline_commit,
            "device_uuid": uuid,
            "device_name": EXPECTED_HARDWARE_NAME,
            "sm_clock_mhz": BOOST_CLOCK_MHZ,
            "nsys_version": "2025.6.1.190",
            "compiler_version": compiler,
            "driver_version": driver,
            "artifact_sha256": artifacts,
            "manifest_sha256": hashlib.sha256(manifest_json).hexdigest(),
            "audited_report_generator_sha256": _sha256(Path(__file__).resolve()),
            "sass_contract": (
                "one default-cache LDG.E -> dependent IADD3 -> STG.E chain "
                "per active warp, with no barrier or atomic"),
            "source": SOURCE,
            "source_sha256": _sha256(REPOSITORY_ROOT / SOURCE),
            "runner": RUNNER,
            "runner_sha256": _sha256(REPOSITORY_ROOT / RUNNER),
            "binary_sha256": binary_sha256,
            "capture": capture,
            "clock_gate": (
                "SM clock not lockable in the container; every nvidia-smi -lms 50 "
                "sample with nonzero utilization during each trial must be >= "
                "1980 MHz, else the trial is rejected"),
            "host_contention": (
                "GPUs 1-3 of the same host were used concurrently by other workers"),
        },
        "support": {
            "launch_rank": 1,
            "minimum_grid": 1,
            "maximum_grid": MAXIMUM_GRID,
            "minimum_block": 1,
            "maximum_block": 1024,
            "cache_policy": "warm_default_cache",
            "measured_sm_clock_mhz": BOOST_CLOCK_MHZ,
            "maximum_supported_sm_clock_mhz": BOOST_CLOCK_MHZ,
            "warm_input_footprint_bytes": maximum_warps * 4,
            "warm_output_footprint_bytes": maximum_warps * 4,
            "address_contract": "separate_contiguous_uint32_arrays",
            "active_lane_contract": "lane_zero_only_per_physical_warp",
            "memory_contract": (
                "one active lane per warp executes a 32-bit default-cache "
                "global load -> integer dependency -> global store"),
        },
        "formula": (
            "base_ns + (fanout_activation_ns if grid >= 3 else 0) + "
            "grid * cta_ns + ceil(block/32) * warp_ns"),
        "parameters": model,
        "lower_envelope_derivation": {
            "unconstrained_relative_least_squares_parameters": unconstrained,
            "training_observed_minimum_guard_ns": guard_ns,
            "rule": (
                "subtract the largest training-cell excess over any observed "
                "sample from the intercept only"),
        },
        "fit_diagnostics": {
            "training_median": training_metrics,
            "validation_median": validation_metrics,
            "training_lower_envelope": training_envelope,
            "validation_lower_envelope": validation_envelope,
        },
        "publication_gates": gates,
        "publishable": all(gates.values()),
        "limitations": [
            "ordinary LDG.E/STG.E plus repeated warm-up proves warm default-cache behavior, not exclusive L2 residency",
            "the grid-three fanout transition has no public internal scheduler-topology label",
            "the floor applies only to statically proven default-cache load-dependency-store typed paths",
            "the measured nanosecond floor was captured at a 1980 MHz busy SM clock (unlocked, boost) and is not rescaled at lower clocks",
            "the model must not be extrapolated to Dim3, grid greater than 132, or multi-wave launch",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.run_dir, args.sass, args.binary_sha256,
                          args.baseline_commit, args.compiler)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = sorted(name for name, passed in report["publication_gates"].items()
                    if not passed)
    print(json.dumps({"parameters": report["parameters"],
                      "validation": report["fit_diagnostics"]["validation_median"],
                      "failed_gates": failed}, indent=1))


if __name__ == "__main__":
    main()
