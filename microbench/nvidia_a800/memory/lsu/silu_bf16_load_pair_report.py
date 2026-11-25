#!/usr/bin/env python3
"""Audit three independent A800 SiLU BF16 load-pair captures."""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from silu_bf16_load_pair_runner import (
    CELLS,
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_DEVICE_NAME,
    EXPECTED_SM_COUNT,
    MAXIMUM_IDLE_DRIVER_MEMORY_MIB,
    REQUIRED_INDEPENDENT_RUNS,
    SCHEDULE_NAME,
    TRIALS,
    VARIANTS,
    _parse_probe_resources,
    _parse_probe_sass,
    _validate_production_sass,
    cell_index_for,
)


STABILITY_RELATIVE_LIMIT = 0.20
READ_MULTIPLIERS = {
    "control": 0,
    "single_bf16_load": 1,
    "paired_bf16_load": 2,
}
RAW_FIELDS = (
    "run_id", "trial", "order_position", "cell_index", "variant", "grid",
    "elements_per_cta", "read_bytes_per_cta", "duration_ns")
MANIFEST_FIELDS = {
    "schema", "hardware_id", "owner", "run_id", "run_rotation",
    "runner_pid", "native_pid", "baseline_commit", "tracked_worktree_status",
    "command", "native_command", "device_preflight", "device_postflight",
    "schedule", "trials", "source_sha256", "runner_sha256",
    "report_sha256", "probe_binary_sha256", "probe_resource_sha256",
    "probe_sass_sha256", "probe_resources", "probe_sass_contract",
    "production_binary_path", "production_binary_sha256",
    "production_cubin_sha256", "production_sass_sha256",
    "production_sass_contract", "raw_sha256",
    "production_or_evaluation_duration_consumed",
    "production_parameter_published",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_spread(values):
    maximum = max(values)
    return 0.0 if maximum == 0 else (maximum - min(values)) / maximum


def _validated_output(path):
    resolved = path.resolve()
    root = (
        Path(__file__).resolve().parents[4]
        / "profiler_output/microbench/nvidia_a800").resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output must be a child of {root}")
    if resolved.exists():
        raise FileExistsError(f"output already exists: {resolved}")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(
            f"output parent does not exist: {resolved.parent}")
    return resolved


def _read_native_csv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = {}
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("meta,"):
            fields = next(csv.reader([line]))
            if len(fields) != 3 or fields[1] in metadata:
                raise ValueError(f"{path} has invalid metadata")
            metadata[fields[1]] = fields[2]
        elif line.startswith("raw,"):
            fields = next(csv.reader([line]))
            if tuple(fields[1:]) != RAW_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            header_index = index
            break
        else:
            raise ValueError(f"{path} has an unexpected preamble row")
    if header_index is None:
        raise ValueError(f"{path} has no raw header")
    reader = csv.DictReader(
        io.StringIO("\n".join(lines[header_index:])),
        fieldnames=("record", *RAW_FIELDS))
    next(reader)
    rows = []
    for raw in reader:
        if raw["record"] != "raw":
            raise ValueError(f"{path} contains a non-raw data row")
        integers = {
            name: int(raw[name]) for name in (
                "trial", "order_position", "cell_index", "grid",
                "elements_per_cta", "read_bytes_per_cta")}
        duration_ns = float(raw["duration_ns"])
        if not math.isfinite(duration_ns) or duration_ns <= 0:
            raise ValueError(f"{path} contains invalid duration")
        rows.append({
            "run_id": raw["run_id"], "variant": raw["variant"],
            **integers, "duration_ns": duration_ns})
    return metadata, rows


def _validate_device(manifest, name):
    row = manifest[name]
    if set(row) != {
            "index", "name", "uuid", "memory_used_mib",
            "utilization_gpu_pct", "compute_capability", "sm_clock_mhz",
            "memory_clock_mhz", "driver_version"}:
        raise ValueError(f"{name} device schema changed")
    if (row["name"] != EXPECTED_DEVICE_NAME
            or row["compute_capability"] != EXPECTED_COMPUTE_CAPABILITY
            or row["memory_used_mib"] > MAXIMUM_IDLE_DRIVER_MEMORY_MIB
            or row["utilization_gpu_pct"] != 0):
        raise ValueError(f"{name} does not identify an idle A800")
    return row


def _audit_capture(capture):
    capture = capture.resolve()
    manifest_path = capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError(f"{capture} manifest schema changed")
    if (manifest["schema"]
            != "nvidia_a800_silu_bf16_load_pair_capture_v1"
            or manifest["hardware_id"] != "nvidia_a800"
            or manifest["schedule"] != SCHEDULE_NAME
            or manifest["trials"] != TRIALS
            or manifest["production_or_evaluation_duration_consumed"]
            or manifest["production_parameter_published"]):
        raise ValueError(f"{capture} manifest contract changed")
    rotation = manifest["run_rotation"]
    if rotation not in range(REQUIRED_INDEPENDENT_RUNS):
        raise ValueError(f"{capture} has an invalid run rotation")
    preflight = _validate_device(manifest, "device_preflight")
    postflight = _validate_device(manifest, "device_postflight")
    if preflight["uuid"] != postflight["uuid"]:
        raise ValueError(f"{capture} changed devices during execution")
    expected_hash_paths = {
        "source_sha256": "source_snapshot/silu_bf16_load_pair.cu",
        "runner_sha256": "source_snapshot/silu_bf16_load_pair_runner.py",
        "report_sha256": "source_snapshot/silu_bf16_load_pair_report.py",
        "probe_binary_sha256": "build/silu_bf16_load_pair",
        "probe_resource_sha256": "build/probe_resource_usage.txt",
        "probe_sass_sha256": "build/probe.sass",
        "production_cubin_sha256": "build/production_silu_sm80.cubin",
        "production_sass_sha256": "build/production_silu_bf16.sass",
        "raw_sha256": "raw.csv",
    }
    for field, relative in expected_hash_paths.items():
        if manifest[field] != _sha256(capture / relative):
            raise ValueError(f"{capture} has a stale {field}")
    if manifest["probe_resources"] != {
            "silu_load_control": 11,
            "silu_paired_bf16_load": 20,
            "silu_single_bf16_load": 13}:
        raise ValueError(f"{capture} probe resource contract changed")
    parsed_resources = _parse_probe_resources(
        (capture / "build/probe_resource_usage.txt").read_text(
            encoding="utf-8"))
    if parsed_resources != manifest["probe_resources"]:
        raise ValueError(f"{capture} probe resource evidence disagrees")
    if manifest["probe_sass_contract"] != {
            "bf16_load_instruction_counts": {
                "silu_load_control": 0,
                "silu_paired_bf16_load": 2,
                "silu_single_bf16_load": 1},
            "paired_loads_are_adjacent": True}:
        raise ValueError(f"{capture} probe SASS contract changed")
    parsed_probe_sass = _parse_probe_sass(
        (capture / "build/probe.sass").read_text(encoding="utf-8"))
    if parsed_probe_sass != manifest["probe_sass_contract"]:
        raise ValueError(f"{capture} probe SASS evidence disagrees")
    production_contract = manifest["production_sass_contract"]
    if (production_contract["symbol"] == ""
            or production_contract["bf16_load_instruction_count"] != 2
            or production_contract["bf16_load_offsets"] != ["0x170", "0x180"]
            or not production_contract["loads_are_adjacent"]
            or production_contract["loop_back_offset"] != "0xe0"):
        raise ValueError(f"{capture} production SASS contract changed")
    parsed_production_sass = _validate_production_sass(
        (capture / "build/production_silu_bf16.sass").read_text(
            encoding="utf-8"))
    if parsed_production_sass != production_contract:
        raise ValueError(f"{capture} production SASS evidence disagrees")

    metadata, rows = _read_native_csv(capture / "raw.csv")
    expected_metadata = {
        "device_index": "0",
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": EXPECTED_COMPUTE_CAPABILITY,
        "sm_count": str(EXPECTED_SM_COUNT),
        "threads_per_cta": "1024",
        "eviction_bytes": str(96 * 1024 * 1024),
        "clock_warmup_launches": "2048",
        "event_samples_per_trial": "8",
        "run_id": manifest["run_id"],
        "run_rotation": str(rotation),
        "schedule": SCHEDULE_NAME,
        "production_or_evaluation_duration_consumed": "false",
        "production_parameter_published": "false",
    }
    if metadata != expected_metadata:
        raise ValueError(f"{capture} native metadata changed")
    if len(rows) != TRIALS * len(CELLS):
        raise ValueError(f"{capture} raw row count changed")
    by_cell = {cell: [] for cell in CELLS}
    seen = set()
    for row in rows:
        trial_index = row["trial"] - 1
        order_position = row["order_position"]
        if trial_index not in range(TRIALS):
            raise ValueError(f"{capture} trial is outside the schedule")
        if order_position not in range(len(CELLS)):
            raise ValueError(f"{capture} order position is outside the schedule")
        expected_index = cell_index_for(
            trial_index, order_position, rotation)
        cell = (row["variant"], row["grid"], row["elements_per_cta"])
        expected_cell = CELLS[expected_index]
        identity = (trial_index, order_position)
        if (row["run_id"] != manifest["run_id"]
                or row["cell_index"] != expected_index
                or cell != expected_cell or identity in seen):
            raise ValueError(f"{capture} violates its cyclic schedule")
        seen.add(identity)
        expected_bytes = (
            row["elements_per_cta"] * 2
            * READ_MULTIPLIERS[row["variant"]])
        if row["read_bytes_per_cta"] != expected_bytes:
            raise ValueError(f"{capture} read-byte contract changed")
        by_cell[cell].append(row["duration_ns"])
    if any(len(values) != TRIALS for values in by_cell.values()):
        raise ValueError(f"{capture} does not replicate every cell")

    replication = {}
    for cell, values in by_cell.items():
        q1, _, q3 = statistics.quantiles(
            values, n=4, method="inclusive")
        spread = _relative_spread((q1, q3))
        replication[cell] = {
            "median_ns": statistics.median(values),
            "q1_ns": q1,
            "q3_ns": q3,
            "interquartile_relative_spread": spread,
            "gate": spread <= STABILITY_RELATIVE_LIMIT,
        }
    if not all(result["gate"] for result in replication.values()):
        raise ValueError(f"{capture} failed within-run replication stability")
    return {
        "capture": str(capture),
        "manifest": manifest,
        "device_uuid": preflight["uuid"],
        "replication": replication,
    }


def _audit_runs(captures):
    if len(captures) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError(
            f"campaign requires exactly {REQUIRED_INDEPENDENT_RUNS} captures")
    audits = tuple(_audit_capture(capture) for capture in captures)
    manifests = tuple(audit["manifest"] for audit in audits)
    run_ids = tuple(manifest["run_id"] for manifest in manifests)
    if len(set(run_ids)) != REQUIRED_INDEPENDENT_RUNS:
        raise ValueError("capture run IDs must be distinct")
    if {manifest["run_rotation"] for manifest in manifests} != {0, 1, 2}:
        raise ValueError("captures must cover rotations 0, 1, and 2")
    identity_fields = (
        "baseline_commit", "source_sha256", "runner_sha256", "report_sha256",
        "probe_resource_sha256", "probe_sass_sha256",
        "production_binary_sha256", "production_cubin_sha256",
        "production_sass_sha256", "production_sass_contract")
    for field in identity_fields:
        if len({json.dumps(manifest[field], sort_keys=True)
                for manifest in manifests}) != 1:
            raise ValueError(f"captures disagree on {field}")
    if len({audit["device_uuid"] for audit in audits}) != 1:
        raise ValueError("captures disagree on device UUID")

    cells = {}
    for cell in CELLS:
        medians = tuple(
            audit["replication"][cell]["median_ns"] for audit in audits)
        spread = _relative_spread(medians)
        cells[cell] = {
            "medians_by_run_ns": dict(zip(run_ids, medians)),
            "cross_run_relative_spread": spread,
            "stability_gate": spread <= STABILITY_RELATIVE_LIMIT,
        }
    if not all(result["stability_gate"] for result in cells.values()):
        raise ValueError("campaign failed cross-run stability")
    return {
        "schema": "nvidia_a800_silu_bf16_load_pair_audit_v1",
        "run_ids": run_ids,
        "capture_paths": tuple(audit["capture"] for audit in audits),
        "device_uuid": audits[0]["device_uuid"],
        "baseline_commit": manifests[0]["baseline_commit"],
        "identity": {field: manifests[0][field] for field in identity_fields},
        "probe_binary_sha256_by_run": {
            manifest["run_id"]: manifest["probe_binary_sha256"]
            for manifest in manifests},
        "trials_per_cell_per_run": TRIALS,
        "within_run_replication_gate": True,
        "cross_run_stability_limit": STABILITY_RELATIVE_LIMIT,
        "cross_run_stability_gate": True,
        "cells": {
            f"{cell[0]}/grid{cell[1]}/width{cell[2]}": result
            for cell, result in cells.items()},
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit = _audit_runs(args.capture)
    output = _validated_output(args.output)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"run_ids={','.join(audit['run_ids'])}")
    print(f"cells={len(audit['cells'])}")
    print("within_run_replication_gate=pass")
    print("cross_run_stability_gate=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
