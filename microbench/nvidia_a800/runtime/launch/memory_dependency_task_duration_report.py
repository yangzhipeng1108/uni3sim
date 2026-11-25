#!/usr/bin/env python3
"""Freeze the A800 warm default-cache memory-dependency task floor."""

import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_SUMMARY_FIELDS = (
    "trial", "grid", "block", "sample_count", "median_duration_ns")
EXPECTED_MEASUREMENT_FIELDS = (
    "trial", "seq", "nvtx_range", "grid", "block", "duration_ns",
    "correlationId", "range_start", "range_end", "launch_start",
    "launch_end", "kernel_start", "kernel_end")
EXPECTED_TRIALS = (1, 2, 3)
EXPECTED_GRIDS = (256, 512, 1024, 2048, 4096)
EXPECTED_BLOCKS = (128, 256)
TRAINING_GRIDS = (256, 1024, 4096)
HELD_OUT_GRIDS = (512, 2048)
EXPECTED_SAMPLES_PER_CELL = 200
SM_COUNT = 108
MAXIMUM_THREADS_PER_SM = 2048
MAXIMUM_CTAS_PER_SM = 32
EXPECTED_CLOCK_MHZ = 1410
EXPECTED_HARDWARE_NAME = "NVIDIA A800-SXM4-80GB"
MAXIMUM_CROSS_RUN_SPREAD_PERCENT = 10.0
MAXIMUM_HELD_OUT_MAPE_PERCENT = 5.0
MAXIMUM_HELD_OUT_APE_PERCENT = 15.0
SOURCE_PATH = Path(__file__).with_name("memory_dependency_task_duration.cu")
RUNNER_PATH = Path(__file__).with_name("memory_dependency_task_duration_runner.py")
REPORT_PATH = Path(__file__).resolve()
REQUIRED_EVIDENCE_PATHS = (
    "SHA256SUMS",
    "aggregate.sql",
    "baseline_commit.txt",
    "provenance.json",
    "build/memory_dependency_task_duration",
    "build/memory_dependency_task_duration.resources",
    "build/memory_dependency_task_duration.sass",
    "run/device_identity.txt",
    "run/completed_at.txt",
    "run/measurements_3trials.csv",
    "run/memory_dependency_trial1.nsys-rep",
    "run/memory_dependency_trial1.sqlite",
    "run/trial1.stdout",
    "run/memory_dependency_trial2.nsys-rep",
    "run/memory_dependency_trial2.sqlite",
    "run/trial2.stdout",
    "run/memory_dependency_trial3.nsys-rep",
    "run/memory_dependency_trial3.sqlite",
    "run/trial3.stdout",
    "run/summary_by_trial.csv",
    "run/started_at.txt",
    "run/validation_3trials.txt",
    "source_snapshot/memory_dependency_task_duration.cu",
    "source_snapshot/memory_dependency_task_duration_report.py",
    "source_snapshot/memory_dependency_task_duration_runner.py",
    "telemetry/trial1_live_clock.csv",
    "telemetry/trial1_preflight_compute_processes.csv",
    "telemetry/trial1_postflight_compute_processes.csv",
    "telemetry/trial1_postflight.csv",
    "telemetry/trial2_live_clock.csv",
    "telemetry/trial2_preflight_compute_processes.csv",
    "telemetry/trial2_postflight_compute_processes.csv",
    "telemetry/trial2_postflight.csv",
    "telemetry/trial3_live_clock.csv",
    "telemetry/trial3_preflight_compute_processes.csv",
    "telemetry/trial3_postflight_compute_processes.csv",
    "telemetry/trial3_postflight.csv",
    "telemetry/preflight.csv",
    "telemetry/preflight_compute_processes.csv",
)
REQUIRED_PROVENANCE_FIELDS = (
    "hardware_id", "owner", "baseline_commit", "device_index",
    "device_uuid", "device_pci_bus_id", "timer_domain", "owner_shell_pid",
    "nsys_version", "compiler_version", "cuobjdump_version",
    "nvidia_smi_version", "driver_version", "nvcc_path", "cuobjdump_path",
    "nsys_path", "nvidia_smi_path", "source_sha256",
    "report_source_sha256", "runner_source_sha256", "binary_sha256",
    "resource_sha256", "sass_sha256", "summary_sha256", "measurement_sha256",
    "nvcc_sha256", "cuobjdump_sha256", "nsys_sha256", "nvidia_smi_sha256",
    "production_or_evaluation_duration_consumed", "cleanup")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value, name):
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if str(parsed) != value or parsed <= 0:
        raise ValueError(f"{name} must be a canonical positive integer")
    return parsed


def _positive_float(value, name):
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_SUMMARY_FIELDS:
            raise ValueError(
                "A800 memory-task summary fields disagree with the frozen "
                "schema")
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
                raise ValueError(f"duplicate A800 memory-task cell: {identity}")
            identities.add(identity)
            rows.append(row)
    expected = {
        (trial, grid, block)
        for trial in EXPECTED_TRIALS
        for grid in EXPECTED_GRIDS
        for block in EXPECTED_BLOCKS
    }
    if identities != expected:
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ValueError(
            "A800 memory-task matrix is incomplete: "
            f"missing={missing[:4]}, extra={extra[:4]}")
    if any(row["sample_count"] != EXPECTED_SAMPLES_PER_CELL for row in rows):
        raise ValueError(
            "every A800 memory-task cell must contain exactly 200 samples")
    return tuple(rows)


def read_measurements(path):
    durations = defaultdict(list)
    sequences = set()
    correlations = set()
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_MEASUREMENT_FIELDS:
            raise ValueError(
                "A800 memory-task measurement fields disagree with the "
                "frozen schema")
        for ordinal, raw in enumerate(reader, start=2):
            trial = _positive_int(raw["trial"], f"line {ordinal}.trial")
            sequence = _positive_int(raw["seq"], f"line {ordinal}.seq")
            grid = _positive_int(raw["grid"], f"line {ordinal}.grid")
            block = _positive_int(raw["block"], f"line {ordinal}.block")
            duration = _positive_float(
                raw["duration_ns"], f"line {ordinal}.duration_ns")
            correlation = _positive_int(
                raw["correlationId"], f"line {ordinal}.correlationId")
            range_start = _positive_int(
                raw["range_start"], f"line {ordinal}.range_start")
            range_end = _positive_int(
                raw["range_end"], f"line {ordinal}.range_end")
            launch_start = _positive_int(
                raw["launch_start"], f"line {ordinal}.launch_start")
            launch_end = _positive_int(
                raw["launch_end"], f"line {ordinal}.launch_end")
            kernel_start = _positive_int(
                raw["kernel_start"], f"line {ordinal}.kernel_start")
            kernel_end = _positive_int(
                raw["kernel_end"], f"line {ordinal}.kernel_end")
            identity = (trial, sequence)
            if identity in sequences:
                raise ValueError(
                    f"duplicate A800 memory-task sequence: {identity}")
            sequences.add(identity)
            correlation_identity = (trial, correlation)
            if correlation_identity in correlations:
                raise ValueError(
                    "duplicate A800 memory-task correlation: "
                    f"{correlation_identity}")
            correlations.add(correlation_identity)
            if trial not in EXPECTED_TRIALS:
                raise ValueError(f"unexpected trial in measurements: {trial}")
            if grid not in EXPECTED_GRIDS or block not in EXPECTED_BLOCKS:
                raise ValueError(
                    "measurement geometry is outside the frozen matrix: "
                    f"trial={trial}, grid={grid}, block={block}")
            expected_label = f"memory_dependency_grid_{grid}_block_{block}"
            if raw["nvtx_range"] != expected_label:
                raise ValueError(
                    "measurement NVTX label disagrees with geometry: "
                    f"{raw['nvtx_range']!r}")
            if not (
                    range_start < range_end
                    and launch_start < launch_end
                    and kernel_start < kernel_end):
                raise ValueError(
                    f"measurement timestamps are not ordered at line {ordinal}")
            if kernel_end - kernel_start != duration:
                raise ValueError(
                    "measurement duration disagrees with kernel timestamps "
                    f"at line {ordinal}")
            if not (
                    range_start <= launch_start
                    and launch_end <= range_end):
                raise ValueError(
                    f"launch is outside its NVTX range at line {ordinal}")
            row = {
                "trial": trial,
                "sequence": sequence,
                "nvtx_range": raw["nvtx_range"],
                "grid": grid,
                "block": block,
                "duration_ns": duration,
                "correlation_id": correlation,
                "range_start": range_start,
                "range_end": range_end,
                "launch_start": launch_start,
                "launch_end": launch_end,
                "kernel_start": kernel_start,
                "kernel_end": kernel_end,
            }
            rows.append(row)
            durations[(trial, grid, block)].append(duration)
    expected = {
        (trial, grid, block)
        for trial in EXPECTED_TRIALS
        for grid in EXPECTED_GRIDS
        for block in EXPECTED_BLOCKS
    }
    if set(durations) != expected:
        raise ValueError("A800 raw measurement matrix is incomplete")
    if any(
            len(cell) != EXPECTED_SAMPLES_PER_CELL
            for cell in durations.values()):
        raise ValueError(
            "every A800 raw measurement cell must contain exactly 200 samples")
    return {
        "durations_by_cell": {
            identity: tuple(values) for identity, values in durations.items()
        },
        "rows": tuple(rows),
    }


def validate_summary_against_measurements(rows, measurements):
    durations = measurements["durations_by_cell"]
    for row in rows:
        identity = (row["trial"], row["grid"], row["block"])
        raw = durations[identity]
        if row["sample_count"] != len(raw):
            raise ValueError(
                f"summary sample count disagrees with raw cell {identity}")
        raw_median = float(statistics.median(raw))
        if row["median_duration_ns"] != raw_median:
            raise ValueError(
                "summary median disagrees with raw measurements for cell "
                f"{identity}: summary={row['median_duration_ns']}, "
                f"raw={raw_median}")


def _resident_ctas_per_sm(block):
    if block not in EXPECTED_BLOCKS:
        raise ValueError(f"unsupported A800 memory-task block: {block}")
    return min(MAXIMUM_CTAS_PER_SM, MAXIMUM_THREADS_PER_SM // block)


def _features(row):
    ctas_per_sm = math.ceil(row["grid"] / SM_COUNT)
    resident_ctas = _resident_ctas_per_sm(row["block"])
    queued_ctas = max(0, ctas_per_sm - resident_ctas)
    return (
        1.0,
        float(ctas_per_sm),
        float(queued_ctas),
    )


PARAMETER_NAMES = (
    "launch_base_ns",
    "cta_wave_service_ns",
    "queued_cta_wave_penalty_ns",
)


def _nonnegative_relative_least_squares(design, observed):
    relative_design = design / observed[:, None]
    target = np.ones(len(observed), dtype=float)
    best = None
    for active_mask in range(1, 1 << design.shape[1]):
        active = tuple(
            index for index in range(design.shape[1])
            if active_mask & (1 << index))
        candidate = np.zeros(design.shape[1], dtype=float)
        values, _, _, _ = np.linalg.lstsq(
            relative_design[:, active], target, rcond=None)
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            continue
        candidate[list(active)] = values
        residual = relative_design @ candidate - target
        objective = float(residual @ residual)
        if best is None or objective < best[0]:
            best = (objective, candidate)
    if best is None:
        raise ValueError(
            "A800 multi-wave model has no finite nonnegative solution")
    return best[1]


def fit_floor_model(rows, measurements):
    durations_by_cell = measurements["durations_by_cell"]
    training = tuple(
        row for row in rows if row["grid"] in TRAINING_GRIDS)
    design = np.asarray([_features(row) for row in training], dtype=float)
    training_medians = np.asarray(
        [row["median_duration_ns"] for row in training], dtype=float)
    coefficients = _nonnegative_relative_least_squares(
        design, training_medians)
    unconstrained = dict(zip(PARAMETER_NAMES, map(float, coefficients)))
    guard_ns = max(
        _predict(row, unconstrained)
        - min(durations_by_cell[
            (row["trial"], row["grid"], row["block"])])
        for row in training
    )
    floor = dict(unconstrained)
    floor["launch_base_ns"] -= guard_ns
    if floor["launch_base_ns"] < 0:
        raise ValueError(
            "A800 memory-task lower-envelope base is negative")
    if any(value < 0 or not math.isfinite(value) for value in floor.values()):
        raise ValueError(
            "A800 memory-task floor requires finite nonnegative services")
    return floor, unconstrained, float(guard_ns)


def _predict(row, model):
    return sum(
        model[name] * feature
        for name, feature in zip(PARAMETER_NAMES, _features(row)))


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


def _lower_envelope_metrics(rows, measurements, model):
    durations = measurements["durations_by_cell"]
    overages = np.asarray([
        _predict(row, model)
        - min(durations[(row["trial"], row["grid"], row["block"])])
        for row in rows
    ])
    return {
        "cell_count": len(rows),
        "cells_above_observed_minimum": int(np.sum(overages > 1e-9)),
        "maximum_above_observed_minimum_ns": float(max(0.0, np.max(overages))),
        "minimum_margin_below_observed_minimum_ns": float(-np.max(overages)),
    }


def _cross_run_stability(rows):
    cells = {}
    for grid in EXPECTED_GRIDS:
        for block in EXPECTED_BLOCKS:
            medians = tuple(
                row["median_duration_ns"] for row in rows
                if row["grid"] == grid and row["block"] == block)
            if len(medians) != len(EXPECTED_TRIALS):
                raise ValueError(
                    f"A800 memory-task stability cell is incomplete: "
                    f"grid={grid}, block={block}")
            pooled_median = float(statistics.median(medians))
            spread_percent = (
                (max(medians) - min(medians)) / pooled_median * 100.0)
            cells[f"grid{grid}_block{block}"] = {
                "trial_medians_ns": list(medians),
                "pooled_median_ns": pooled_median,
                "relative_spread_percent": spread_percent,
            }
    maximum = max(
        cell["relative_spread_percent"] for cell in cells.values())
    return {
        "maximum_relative_spread_percent": maximum,
        "limit_percent": MAXIMUM_CROSS_RUN_SPREAD_PERCENT,
        "cells": cells,
    }


def _read_provenance(evidence_root):
    path = evidence_root / "provenance.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(REQUIRED_PROVENANCE_FIELDS):
        raise ValueError(
            "A800 memory-task provenance fields disagree with the frozen "
            "schema")
    if raw["hardware_id"] != "nvidia_a800":
        raise ValueError("A800 memory-task provenance has wrong hardware_id")
    if raw["timer_domain"] != "nsys_gpu_kernel":
        raise ValueError("A800 memory-task provenance has wrong timer domain")
    if raw["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            "A800 memory-task evidence consumed production/evaluation duration")
    for name in (
            "owner", "baseline_commit", "device_uuid", "device_pci_bus_id",
            "nsys_version", "compiler_version", "cuobjdump_version",
            "nvidia_smi_version", "driver_version", "nvcc_path",
            "cuobjdump_path", "nsys_path", "nvidia_smi_path", "cleanup"):
        if not isinstance(raw[name], str) or not raw[name]:
            raise ValueError(f"A800 memory-task provenance {name} is empty")
    for name in ("nvcc_path", "cuobjdump_path", "nsys_path", "nvidia_smi_path"):
        if not Path(raw[name]).is_absolute():
            raise ValueError(
                f"A800 memory-task provenance {name} is not absolute")
    if raw["cleanup"] != "complete":
        raise ValueError(
            "A800 memory-task provenance cleanup must be exactly complete")
    for name in (
            "source_sha256", "report_source_sha256", "runner_source_sha256",
            "binary_sha256", "resource_sha256", "sass_sha256", "summary_sha256",
            "measurement_sha256", "nvcc_sha256", "cuobjdump_sha256",
            "nsys_sha256", "nvidia_smi_sha256"):
        if (not isinstance(raw[name], str) or len(raw[name]) != 64
                or any(character not in "0123456789abcdef" for character in raw[name])):
            raise ValueError(
                f"A800 memory-task provenance {name} is not SHA256")
    if not isinstance(raw["device_index"], int) or raw["device_index"] < 0:
        raise ValueError("A800 memory-task device_index is invalid")
    if not isinstance(raw["owner_shell_pid"], int) or raw["owner_shell_pid"] <= 0:
        raise ValueError("A800 memory-task owner_shell_pid is invalid")
    return raw


def _audit_device_identity(path, provenance):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        for ordinal, raw in enumerate(reader, start=1):
            if len(raw) != 6:
                raise ValueError(
                    f"device identity line {ordinal} must contain six fields")
            (label, visible_index, physical_index, uuid_suffix, pci_bus_id,
             name) = raw
            rows.append((
                label, int(visible_index), int(physical_index), uuid_suffix,
                pci_bus_id, name))
    if tuple(row[0] for row in rows) != ("trial1", "trial2", "trial3"):
        raise ValueError("device identity does not cover the three frozen trials")
    expected_suffix = provenance["device_uuid"].removeprefix("GPU-")
    if any(
            visible != 0 or physical != provenance["device_index"]
            or suffix != expected_suffix
            or pci_bus_id != provenance["device_pci_bus_id"]
            or name != EXPECTED_HARDWARE_NAME
            for _, visible, physical, suffix, pci_bus_id, name in rows):
        raise ValueError("device identity drifted across A800 trials")


def _measurement_window(evidence_root, trial, timezone):
    stdout = (evidence_root / f"run/trial{trial}.stdout").read_text(
        encoding="utf-8")
    starts = re.findall(r"^measurement_started_unix_ns=(\d+)$", stdout,
                        re.MULTILINE)
    completions = re.findall(
        r"^measurement_completed_unix_ns=(\d+)$", stdout, re.MULTILINE)
    if len(starts) != 1 or len(completions) != 1:
        raise ValueError(
            f"A800 trial {trial} lacks one exact measurement window")
    started_ns = int(starts[0])
    completed_ns = int(completions[0])
    if completed_ns <= started_ns:
        raise ValueError(
            f"A800 trial {trial} measurement window is not ordered")
    return (
        datetime.fromtimestamp(started_ns / 1.0e9, tz=timezone),
        datetime.fromtimestamp(completed_ns / 1.0e9, tz=timezone),
    )


def _audit_no_compute_processes(evidence_root):
    paths = [evidence_root / "telemetry/preflight_compute_processes.csv"]
    for trial in EXPECTED_TRIALS:
        paths.extend((
            evidence_root
            / f"telemetry/trial{trial}_preflight_compute_processes.csv",
            evidence_root
            / f"telemetry/trial{trial}_postflight_compute_processes.csv",
        ))
    if any(path.read_text(encoding="utf-8").strip() for path in paths):
        raise ValueError(
            "A800 preflight or postflight has an active compute process")


def _audit_live_clocks(evidence_root, provenance):
    expected_uuid = provenance["device_uuid"]
    started = datetime.fromisoformat(
        (evidence_root / "run/started_at.txt").read_text(
            encoding="utf-8").strip())
    completed = datetime.fromisoformat(
        (evidence_root / "run/completed_at.txt").read_text(
            encoding="utf-8").strip())
    if completed <= started:
        raise ValueError("A800 capture completion must follow its start")
    with (evidence_root / "telemetry/preflight.csv").open(
            newline="", encoding="utf-8") as handle:
        preflight = list(csv.reader(handle))
    if len(preflight) != 1 or len(preflight[0]) != 8:
        raise ValueError("A800 preflight must contain one eight-field record")
    index, uuid, name, memory_used, utilization, memory_clock, clock, pstate = (
        field.strip() for field in preflight[0])
    idle_memory_used_mib = int(memory_used)
    if (int(index) != provenance["device_index"]
            or uuid != expected_uuid
            or name != EXPECTED_HARDWARE_NAME
            or idle_memory_used_mib < 0
            or int(utilization) != 0
            or int(memory_clock) <= 0
            or int(clock) <= 0
            or pstate != "P0"):
        raise ValueError("A800 preflight does not prove an idle target device")
    previous_timestamp = None
    for trial in EXPECTED_TRIALS:
        measurement_started, measurement_completed = _measurement_window(
            evidence_root, trial, started.tzinfo)
        if (measurement_started < started
                or measurement_completed >= completed + timedelta(seconds=1)):
            raise ValueError(
                f"A800 trial {trial} measurement window is outside capture")
        path = evidence_root / f"telemetry/trial{trial}_live_clock.csv"
        row_count = 0
        measurement_row_count = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for ordinal, raw in enumerate(csv.reader(handle), start=1):
                if len(raw) != 8:
                    raise ValueError(
                        f"live-clock line {ordinal} must contain eight fields")
                timestamp, index, uuid, memory_used, utilization, memory_clock, clock, pstate = (
                    field.strip() for field in raw)
                observed_at = datetime.strptime(
                    timestamp, "%Y/%m/%d %H:%M:%S.%f").replace(
                        tzinfo=started.tzinfo)
                if not started <= observed_at < completed + timedelta(seconds=1):
                    raise ValueError(
                        f"A800 trial {trial} telemetry is outside capture time")
                if (previous_timestamp is not None
                        and observed_at <= previous_timestamp):
                    raise ValueError("A800 live-clock telemetry is not ordered")
                previous_timestamp = observed_at
                memory_value = int(memory_used)
                utilization_value = int(utilization)
                memory_clock_value = int(memory_clock)
                clock_value = int(clock)
                in_measurement = (
                    measurement_started - timedelta(milliseconds=1)
                    <= observed_at
                    <= measurement_completed + timedelta(milliseconds=1))
                if (int(index) != provenance["device_index"]
                        or uuid != expected_uuid
                        or memory_value < 0
                        or not 0 <= utilization_value <= 100
                        or memory_clock_value <= 0
                        or clock_value <= 0
                        or pstate != "P0"):
                    raise ValueError(
                        f"A800 trial {trial} device or clock telemetry drifted")
                if in_measurement:
                    if clock_value != EXPECTED_CLOCK_MHZ:
                        raise ValueError(
                            f"A800 trial {trial} measurement clock is not "
                            f"exactly {EXPECTED_CLOCK_MHZ} MHz")
                if in_measurement:
                    measurement_row_count += 1
                row_count += 1
        if row_count == 0:
            raise ValueError(f"A800 trial {trial} live-clock telemetry is empty")
        if measurement_row_count == 0:
            raise ValueError(
                f"A800 trial {trial} has no measurement-clock telemetry")
    for trial in EXPECTED_TRIALS:
        path = evidence_root / f"telemetry/trial{trial}_postflight.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            postflight = list(csv.reader(handle))
        if len(postflight) != 1 or len(postflight[0]) != 7:
            raise ValueError(
                f"A800 trial {trial} postflight must contain one record")
        index, uuid, memory_used, utilization, memory_clock, clock, pstate = (
            field.strip() for field in postflight[0])
        if (int(index) != provenance["device_index"]
                or uuid != expected_uuid
                or int(memory_used) != idle_memory_used_mib
                or int(utilization) != 0
                or int(memory_clock) <= 0
                or int(clock) <= 0
                or pstate != "P0"):
            raise ValueError(
                f"A800 trial {trial} postflight did not return to idle")
    _audit_no_compute_processes(evidence_root)


def _audit_validation_summary(path):
    expected_count = (
        len(EXPECTED_GRIDS)
        * len(EXPECTED_BLOCKS)
        * EXPECTED_SAMPLES_PER_CELL)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        for ordinal, raw in enumerate(reader, start=1):
            if len(raw) != 7:
                raise ValueError(
                    f"validation line {ordinal} must contain seven fields")
            rows.append(tuple(int(field) for field in raw))
    expected = tuple(
        (trial, expected_count, len(EXPECTED_GRIDS) * len(EXPECTED_BLOCKS),
         0, 0, 0, 0)
        for trial in EXPECTED_TRIALS)
    if tuple(rows) != expected:
        raise ValueError(
            "frozen A800 validation summary reports incomplete or mismatched "
            "kernel records")


def _audit_sqlite_records(evidence_root, measurements):
    rows_by_trial = {
        trial: tuple(
            row for row in measurements["rows"] if row["trial"] == trial)
        for trial in EXPECTED_TRIALS
    }
    for trial in EXPECTED_TRIALS:
        path = evidence_root / f"run/memory_dependency_trial{trial}.sqlite"
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            kernel_rows = connection.execute(
                "SELECT kernel.correlationId, kernel.deviceId, "
                "kernel.gridX, kernel.gridY, kernel.gridZ, "
                "kernel.blockX, kernel.blockY, kernel.blockZ, "
                "kernel.start, kernel.end, strings.value "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel "
                "JOIN StringIds AS strings ON strings.id = kernel.shortName "
                "WHERE strings.value LIKE '%memory_dependency_task%' "
                "ORDER BY kernel.start"
            ).fetchall()
            runtime_rows = connection.execute(
                "SELECT runtime.correlationId, runtime.start, runtime.end, "
                "strings.value "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime "
                "JOIN StringIds AS strings ON strings.id = runtime.nameId "
                "WHERE strings.value LIKE 'cudaLaunchKernel%'"
            ).fetchall()
            nvtx_rows = connection.execute(
                "SELECT text, start, end FROM NVTX_EVENTS "
                "WHERE text LIKE 'memory_dependency_grid_%' ORDER BY start"
            ).fetchall()
        finally:
            connection.close()
        raw_rows = rows_by_trial[trial]
        if len(nvtx_rows) != len(raw_rows):
            raise ValueError(
                f"A800 trial {trial} SQLite record count disagrees with raw CSV")
        kernel_by_correlation = {}
        for kernel in kernel_rows:
            correlation = kernel[0]
            if correlation in kernel_by_correlation:
                raise ValueError(
                    f"A800 trial {trial} has duplicate kernel correlation "
                    f"{correlation}")
            kernel_by_correlation[correlation] = kernel
        raw_correlations = {row["correlation_id"] for row in raw_rows}
        warmup_rows = [
            kernel for correlation, kernel in kernel_by_correlation.items()
            if correlation not in raw_correlations
        ]
        first_sweep_start = nvtx_rows[0][1]
        if (len(warmup_rows) != 200
                or any(
                    kernel[1] != 0
                    or tuple(kernel[2:5]) != (4096, 1, 1)
                    or tuple(kernel[5:8]) != (256, 1, 1)
                    or kernel[9] > first_sweep_start
                    or "memory_dependency_task" not in kernel[10]
                    for kernel in warmup_rows)):
            raise ValueError(
                f"A800 trial {trial} lacks exactly 200 pre-sweep "
                "4096x256 default-cache warm-up kernels")
        runtime_by_correlation = {}
        for correlation, start, end, name in runtime_rows:
            if correlation is None:
                continue
            if correlation in runtime_by_correlation:
                raise ValueError(
                    f"A800 trial {trial} has duplicate runtime correlation "
                    f"{correlation}")
            runtime_by_correlation[correlation] = (start, end, name)
        for raw, nvtx in zip(raw_rows, nvtx_rows):
            if raw["correlation_id"] not in kernel_by_correlation:
                raise ValueError(
                    f"A800 trial {trial} lacks kernel correlation "
                    f"{raw['correlation_id']}")
            kernel = kernel_by_correlation[raw["correlation_id"]]
            (correlation, device, grid_x, grid_y, grid_z, block_x,
             block_y, block_z, kernel_start, kernel_end,
             kernel_name) = kernel
            if (
                    correlation != raw["correlation_id"]
                    or device != 0
                    or (grid_x, grid_y, grid_z) != (raw["grid"], 1, 1)
                    or (block_x, block_y, block_z) != (raw["block"], 1, 1)
                    or kernel_start != raw["kernel_start"]
                    or kernel_end != raw["kernel_end"]
                    or "memory_dependency_task" not in kernel_name):
                raise ValueError(
                    f"A800 trial {trial} SQLite kernel record disagrees with "
                    f"raw sequence {raw['sequence']}")
            if correlation not in runtime_by_correlation:
                raise ValueError(
                    f"A800 trial {trial} lacks runtime correlation {correlation}")
            runtime_start, runtime_end, runtime_name = (
                runtime_by_correlation[correlation])
            if (
                    runtime_start != raw["launch_start"]
                    or runtime_end != raw["launch_end"]
                    or not runtime_name.startswith("cudaLaunchKernel")):
                raise ValueError(
                    f"A800 trial {trial} runtime record disagrees with raw "
                    f"sequence {raw['sequence']}")
            nvtx_text, nvtx_start, nvtx_end = nvtx
            if (
                    nvtx_text != raw["nvtx_range"]
                    or nvtx_start != raw["range_start"]
                    or nvtx_end != raw["range_end"]):
                raise ValueError(
                    f"A800 trial {trial} NVTX record disagrees with raw "
                    f"sequence {raw['sequence']}")


def _audit_target_source_and_sass(evidence_root):
    source_path = (
        evidence_root / "source_snapshot/memory_dependency_task_duration.cu")
    source = source_path.read_text(encoding="utf-8")
    required_source = (
        "constexpr int kWarmupLaunches = 200;",
        "constexpr int kMaximumGrid = 4096;",
        "constexpr int kMaximumBlock = 256;",
        "for (int grid : {256, 512, 1024, 2048, 4096})",
        "for (int block : {128, 256})",
        "__global__ void memory_dependency_task(",
        "if (lane != 0)",
        "output[index] = input[index] + 1U;",
        "emit_measurement_marker(\"measurement_started_unix_ns\");",
        "emit_measurement_marker(\"measurement_completed_unix_ns\");",
    )
    if any(fragment not in source for fragment in required_source):
        raise ValueError(
            "A800 memory-task source lacks its frozen warm-up or dependency "
            "contract")
    sass_path = evidence_root / "build/memory_dependency_task_duration.sass"
    sass = sass_path.read_text(encoding="utf-8")
    markers = [
        offset for offset, line in enumerate(sass.splitlines())
        if "Function :" in line and "memory_dependency_taskEPKjPj" in line
    ]
    if len(markers) != 1:
        raise ValueError(
            "A800 SASS must contain exactly one target memory-task symbol")
    lines = sass.splitlines()
    start = markers[0]
    end = next(
        (offset for offset in range(start + 1, len(lines))
         if ("Function :" in lines[offset]
             or lines[offset].startswith("Fatbin ptx code:"))),
        len(lines),
    )
    target_sass = "\n".join(lines[start:end])
    required_sass = (
        "LDG.E R2, [R2.64]", "IADD3 R7, R2, 0x1, RZ",
        "STG.E [R4.64], R7")
    if any(target_sass.count(instruction) != 1 for instruction in required_sass):
        raise ValueError(
            "A800 target memory-task SASS lacks one exact "
            "load/dependency/store chain")
    if target_sass.count("LDG.") != 1 or target_sass.count("STG.") != 1:
        raise ValueError(
            "A800 target memory-task SASS must contain exactly one global "
            "load and one global store")
    forbidden_sass = (
        "LDG.E.STRONG", "MEMBAR", "BAR.SYNC", "ATOM.", "RED.")
    if any(instruction in target_sass for instruction in forbidden_sass):
        raise ValueError("A800 target memory-task SASS has an unsupported mechanism")
    source_identifier = f"identifier = {source_path.resolve()}"
    if source_identifier not in sass or "arch = sm_80" not in sass:
        raise ValueError(
            "A800 SASS is not bound to the frozen source path and sm_80 arch")
    resource_text = (
        evidence_root / "build/memory_dependency_task_duration.resources"
    ).read_text(encoding="utf-8")
    current_symbol = None
    target_rows = []
    for line in resource_text.splitlines():
        symbol = re.match(r"\s*Function\s+([^:]+):\s*$", line)
        if symbol is not None:
            current_symbol = symbol.group(1)
            continue
        if (current_symbol is None
                or "memory_dependency_taskEPKjPj" not in current_symbol):
            continue
        usage = re.search(
            r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)",
            line)
        if usage is not None:
            target_rows.append(tuple(int(value) for value in usage.groups()))
    if len(target_rows) != 1:
        raise ValueError(
            "A800 resource evidence must contain one target kernel row")
    registers, stack_bytes, shared_bytes, local_bytes = target_rows[0]
    if (registers <= 0 or registers > 32 or stack_bytes != 0
            or shared_bytes != 0 or local_bytes != 0):
        raise ValueError(
            "A800 target resources do not prove thread-limited occupancy")
    return {
        "registers_per_thread": registers,
        "stack_bytes_per_thread": stack_bytes,
        "static_shared_bytes_per_cta": shared_bytes,
        "local_bytes_per_thread": local_bytes,
        "resident_ctas_per_sm_by_block": {
            str(block): _resident_ctas_per_sm(block)
            for block in EXPECTED_BLOCKS
        },
    }


def _audit_capture_manifest(evidence_root):
    manifest_path = evidence_root / "SHA256SUMS"
    recorded = {}
    for ordinal, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(
                f"capture manifest line {ordinal} must contain hash and path")
        digest, path_text = fields
        if (len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(
                f"capture manifest line {ordinal} has invalid SHA256")
        path = Path(path_text).resolve()
        if path in recorded:
            raise ValueError(f"capture manifest repeats path {path}")
        recorded[path] = digest
    for path, expected in recorded.items():
        try:
            path.relative_to(evidence_root)
        except ValueError as error:
            raise ValueError(
                f"capture manifest path escapes evidence root: {path}") from error
        if not path.is_file():
            raise ValueError(f"capture manifest path is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"capture manifest hash mismatch for {path}: {actual}")
    for relative in REQUIRED_EVIDENCE_PATHS:
        if relative == "SHA256SUMS":
            continue
        path = (evidence_root / relative).resolve()
        if path not in recorded:
            raise ValueError(
                f"capture manifest does not freeze required evidence: {relative}")


def audit_evidence(evidence_root, summary_path, measurement_path):
    evidence_root = evidence_root.resolve()
    summary_path = summary_path.resolve()
    measurement_path = measurement_path.resolve()
    expected_summary = (evidence_root / "run/summary_by_trial.csv").resolve()
    expected_measurements = (
        evidence_root / "run/measurements_3trials.csv").resolve()
    if summary_path != expected_summary or measurement_path != expected_measurements:
        raise ValueError(
            "report inputs must be the summary and raw measurements from the "
            "same frozen evidence root")
    for relative in REQUIRED_EVIDENCE_PATHS:
        path = evidence_root / relative
        if not path.is_file():
            raise ValueError(f"required A800 evidence is missing: {relative}")
    _audit_capture_manifest(evidence_root)
    provenance = _read_provenance(evidence_root)
    exact_sources = {
        "source_sha256": SOURCE_PATH,
        "report_source_sha256": REPORT_PATH,
        "runner_source_sha256": RUNNER_PATH,
    }
    for field, current_path in exact_sources.items():
        if provenance[field] != _sha256(current_path):
            raise ValueError(
                f"A800 capture {field} differs from the current campaign")
    baseline_commit = (evidence_root / "baseline_commit.txt").read_text(
        encoding="utf-8").strip()
    if (len(baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline_commit)
            or not baseline_commit.startswith(provenance["baseline_commit"])):
        raise ValueError(
            "A800 frozen baseline commit must be a full SHA matching "
            "provenance")
    expected_hashes = {
        "source_snapshot/memory_dependency_task_duration.cu": (
            provenance["source_sha256"]),
        "source_snapshot/memory_dependency_task_duration_report.py": (
            provenance["report_source_sha256"]),
        "source_snapshot/memory_dependency_task_duration_runner.py": (
            provenance["runner_source_sha256"]),
        "build/memory_dependency_task_duration": provenance["binary_sha256"],
        "build/memory_dependency_task_duration.resources": (
            provenance["resource_sha256"]),
        "build/memory_dependency_task_duration.sass": provenance["sass_sha256"],
        "run/summary_by_trial.csv": provenance["summary_sha256"],
        "run/measurements_3trials.csv": provenance["measurement_sha256"],
    }
    for relative, expected_hash in expected_hashes.items():
        actual_hash = _sha256(evidence_root / relative)
        if actual_hash != expected_hash:
            raise ValueError(
                f"A800 evidence hash mismatch for {relative}: {actual_hash}")
    _audit_device_identity(
        evidence_root / "run/device_identity.txt", provenance)
    _audit_live_clocks(evidence_root, provenance)
    _audit_validation_summary(
        evidence_root / "run/validation_3trials.txt")
    resource_contract = _audit_target_source_and_sass(evidence_root)
    manifest = {
        relative: _sha256(evidence_root / relative)
        for relative in REQUIRED_EVIDENCE_PATHS
    }
    manifest_json = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "baseline_commit": baseline_commit,
        "device_uuid": provenance["device_uuid"],
        "device_pci_bus_id": provenance["device_pci_bus_id"],
        "device_name": EXPECTED_HARDWARE_NAME,
        "sm_clock_mhz": EXPECTED_CLOCK_MHZ,
        "nsys_version": provenance["nsys_version"],
        "compiler_version": provenance["compiler_version"],
        "cuobjdump_version": provenance["cuobjdump_version"],
        "nvidia_smi_version": provenance["nvidia_smi_version"],
        "driver_version": provenance["driver_version"],
        "toolchain": {
            "nvcc": {
                "path": provenance["nvcc_path"],
                "sha256": provenance["nvcc_sha256"],
            },
            "cuobjdump": {
                "path": provenance["cuobjdump_path"],
                "sha256": provenance["cuobjdump_sha256"],
            },
            "nsys": {
                "path": provenance["nsys_path"],
                "sha256": provenance["nsys_sha256"],
            },
            "nvidia_smi": {
                "path": provenance["nvidia_smi_path"],
                "sha256": provenance["nvidia_smi_sha256"],
            },
        },
        "resource_contract": resource_contract,
        "artifact_sha256": manifest,
        "manifest_sha256": hashlib.sha256(manifest_json).hexdigest(),
        "audited_report_generator_sha256": _sha256(
            REPORT_PATH),
        "sass_contract": (
            "one default-cache LDG.E -> dependent IADD3 -> STG.E chain "
            "per active warp, with no barrier or atomic"),
    }


def build_report(summary_path, measurement_path, evidence_root):
    evidence = audit_evidence(
        evidence_root, summary_path, measurement_path)
    rows = read_summary(summary_path)
    measurements = read_measurements(measurement_path)
    validate_summary_against_measurements(rows, measurements)
    _audit_sqlite_records(evidence_root.resolve(), measurements)
    model, unconstrained, guard_ns = fit_floor_model(rows, measurements)
    training = tuple(
        row for row in rows if row["grid"] in TRAINING_GRIDS)
    held_out = tuple(
        row for row in rows if row["grid"] in HELD_OUT_GRIDS)
    training_metrics = _metrics(training, model)
    held_out_metrics = _metrics(held_out, model)
    training_envelope = _lower_envelope_metrics(
        training, measurements, model)
    held_out_envelope = _lower_envelope_metrics(
        held_out, measurements, model)
    stability = _cross_run_stability(rows)
    gates = {
        "parameters_are_nonnegative": all(
            math.isfinite(value) and value >= 0.0
            for value in model.values()),
        "cross_run_stability_at_most_10_percent": (
            stability["maximum_relative_spread_percent"]
            <= MAXIMUM_CROSS_RUN_SPREAD_PERCENT),
        "training_never_above_observed_minimum": (
            training_envelope["cells_above_observed_minimum"] == 0),
        "held_out_never_above_observed_minimum": (
            held_out_envelope["cells_above_observed_minimum"] == 0),
        "held_out_mape_at_most_5_percent": (
            held_out_metrics["mape_percent"]
            <= MAXIMUM_HELD_OUT_MAPE_PERCENT),
        "held_out_max_at_most_15_percent": (
            held_out_metrics["maximum_absolute_percentage_error"]
            <= MAXIMUM_HELD_OUT_APE_PERCENT),
    }
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    if failed_gates:
        raise ValueError(
            "A800 memory-task publication gates failed: "
            f"{failed_gates}")
    return {
        "schema_version": 5,
        "hardware_id": "nvidia_a800",
        "mechanism": "warm_default_cache_memory_dependency_task_floor",
        "duration_source": "nsys_gpu_kernel",
        "production_or_evaluation_duration_consumed": False,
        "input": {
            "summary_path": str(summary_path),
            "summary_sha256": _sha256(summary_path),
            "measurement_path": str(measurement_path),
            "measurement_sha256": _sha256(measurement_path),
            "trials": list(EXPECTED_TRIALS),
            "training_grids": list(TRAINING_GRIDS),
            "held_out_grids": list(HELD_OUT_GRIDS),
            "split_rule": (
                "fixed grid identity frozen before capture; all three trials "
                "of a grid remain on the same side of the split"),
            "samples_per_cell": EXPECTED_SAMPLES_PER_CELL,
        },
        "evidence": evidence,
        "support": {
            "launch_rank": 1,
            "minimum_grid": 256,
            "maximum_grid": 4096,
            "minimum_block": 128,
            "maximum_block": 256,
            "sm_count": SM_COUNT,
            "maximum_threads_per_sm": MAXIMUM_THREADS_PER_SM,
            "maximum_ctas_per_sm": MAXIMUM_CTAS_PER_SM,
            "resident_ctas_per_sm_by_block": {
                str(block): _resident_ctas_per_sm(block)
                for block in EXPECTED_BLOCKS
            },
            "cache_policy": "warm_default_cache",
            "measured_sm_clock_mhz": EXPECTED_CLOCK_MHZ,
            "maximum_supported_sm_clock_mhz": EXPECTED_CLOCK_MHZ,
            "warm_input_footprint_bytes": 131072,
            "warm_output_footprint_bytes": 131072,
            "address_contract": "separate_contiguous_uint32_arrays",
            "active_lane_contract": "lane_zero_only_per_physical_warp",
            "memory_contract": (
                "one active lane per warp executes a 32-bit default-cache "
                "global load -> integer dependency -> global store"),
        },
        "formula": (
            "launch_base_ns + ceil(grid/108) * cta_wave_service_ns + "
            "max(0, ceil(grid/108) - resident_ctas_per_sm[block]) "
            "* queued_cta_wave_penalty_ns"),
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
            "held_out_median": held_out_metrics,
            "training_lower_envelope": training_envelope,
            "held_out_lower_envelope": held_out_envelope,
            "cross_run_stability": stability,
        },
        "publication_gates": gates,
        "publishable": all(gates.values()),
        "limitations": [
            "ordinary LDG.E/STG.E plus repeated warm-up proves warm default-cache behavior, not exclusive L2 residency",
            "the floor applies only to statically proven default-cache load-dependency-store typed paths",
            "the measured nanosecond floor is conservative only at SM clocks no faster than 1410 MHz and is not rescaled at lower clocks",
            "the deterministic geometry holdout was frozen before capture and uses grids 512 and 2048",
            "the model must not be extrapolated outside one-dimensional grids 256 through 4096 or blocks 128 and 256",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        args.summary, args.measurements, args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
