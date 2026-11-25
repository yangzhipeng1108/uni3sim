#!/usr/bin/env python3
"""Fit and validate the source-shaped TensorIterator copy task mechanism."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROGRAMS = {
    "bf16_rank4_offset": {
        "contract": "torch_tensor_iterator_strided_copy_bf16_rank4_v1",
        "dtype": "bf16",
        "tensor_iterator_dimensions": 4,
        "values_per_thread": 4,
    },
    "fp32_rank3_offset": {
        "contract": "torch_tensor_iterator_strided_copy_fp32_rank3_v1",
        "dtype": "float32",
        "tensor_iterator_dimensions": 3,
        "values_per_thread": 2,
    },
}
EXPECTED_GRIDS = (
    1, 2, 4, 8, 16, 32, 64, 108, 216, 432, 864, 1024,
    1728, 2048, 3456, 4096,
)
TRAIN_TRIALS = (0, 1, 2)
VALIDATION_TRIALS = (3, 4)
FANOUT_ACTIVATION_GRID = 4


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    metadata = {}
    raw_lines = []
    for line in path.read_text().splitlines():
        if line.startswith("meta,"):
            fields = next(csv.reader([line]))
            metadata[fields[1]] = fields[2:]
        else:
            raw_lines.append(line)
    rows = list(csv.DictReader(raw_lines))
    return metadata, rows


def _fit(rows, reference_clock_hz):
    design = np.asarray([
        [1.0, float(row["grid"] >= FANOUT_ACTIVATION_GRID), row["grid"]]
        for row in rows
    ])
    cycles = np.asarray([
        row["duration_ns"] * reference_clock_hz * 1.0e-9
        for row in rows
    ])
    coefficients = np.linalg.lstsq(design, cycles, rcond=None)[0]
    if np.any(coefficients <= 0.0):
        raise ValueError("copy-task mechanism coefficients must be positive")
    return tuple(float(value) for value in coefficients)


def _accuracy(rows, coefficients, reference_clock_hz):
    errors = []
    for row in rows:
        predicted_cycles = (
            coefficients[0]
            + coefficients[1] * (row["grid"] >= FANOUT_ACTIVATION_GRID)
            + coefficients[2] * row["grid"])
        actual_cycles = row["duration_ns"] * reference_clock_hz * 1.0e-9
        errors.append(abs(predicted_cycles - actual_cycles) / actual_cycles)
    return {
        "row_count": len(errors),
        "mape_pct": 100.0 * sum(errors) / len(errors),
        "max_ape_pct": 100.0 * max(errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    capture = Path(args.capture).resolve()
    metadata, raw_rows = _load(capture / "raw.csv")
    required_metadata = {
        "device", "device_name", "sm_count", "reference_sm_clock_hz",
        "threads", "resident_ctas_per_sm_rank4",
        "resident_ctas_per_sm_rank3", "launches_per_sample",
    }
    if set(metadata) < required_metadata:
        raise ValueError(
            f"capture metadata is incomplete: {sorted(required_metadata - set(metadata))}")
    if metadata["device_name"] != ["NVIDIA A800-SXM4-80GB"]:
        raise ValueError("capture device changed")
    if metadata["sm_count"] != ["108"] or metadata["threads"] != ["128"]:
        raise ValueError("capture launch geometry changed")
    if (metadata["resident_ctas_per_sm_rank4"] != ["16"]
            or metadata["resident_ctas_per_sm_rank3"] != ["16"]):
        raise ValueError("TensorIterator probe residency changed")
    reference_clock_hz = float(metadata["reference_sm_clock_hz"][0])
    if not math.isfinite(reference_clock_hz) or reference_clock_hz <= 0.0:
        raise ValueError("reference SM clock must be finite and positive")

    rows = []
    for raw in raw_rows:
        if raw["program"] not in PROGRAMS:
            continue
        row = {
            "trial": int(raw["trial"]),
            "program": raw["program"],
            "grid": int(raw["grid"]),
            "bytes_read": int(raw["bytes_read"]),
            "bytes_written": int(raw["bytes_written"]),
            "duration_ns": float(raw["duration_ns"]),
        }
        if row["grid"] not in EXPECTED_GRIDS:
            raise ValueError("capture contains an unexpected grid")
        expected_bytes = row["grid"] * 128 * 8
        if (row["bytes_read"] != expected_bytes
                or row["bytes_written"] != expected_bytes):
            raise ValueError("copy-task bytes disagree with source geometry")
        if not math.isfinite(row["duration_ns"]) or row["duration_ns"] <= 0:
            raise ValueError("copy-task duration must be finite and positive")
        rows.append(row)
    expected_trials = set(TRAIN_TRIALS + VALIDATION_TRIALS)
    for program in PROGRAMS:
        program_rows = [row for row in rows if row["program"] == program]
        identities = {(row["trial"], row["grid"]) for row in program_rows}
        expected = {
            (trial, grid) for trial in expected_trials for grid in EXPECTED_GRIDS
        }
        if identities != expected:
            raise ValueError(f"capture is incomplete for {program}")

    entries = []
    for program, static in PROGRAMS.items():
        program_rows = [row for row in rows if row["program"] == program]
        train = [row for row in program_rows if row["trial"] in TRAIN_TRIALS]
        validation = [
            row for row in program_rows if row["trial"] in VALIDATION_TRIALS
        ]
        coefficients = _fit(train, reference_clock_hz)
        train_accuracy = _accuracy(train, coefficients, reference_clock_hz)
        validation_accuracy = _accuracy(
            validation, coefficients, reference_clock_hz)
        if validation_accuracy["mape_pct"] > 5.0:
            raise ValueError(
                f"{program} validation MAPE exceeds 5%: {validation_accuracy}")
        entries.append({
            **static,
            "base_cycles": coefficients[0],
            "fanout_activation_cycles": coefficients[1],
            "cta_cycles": coefficients[2],
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        })
    provenance = json.loads((capture / "provenance.json").read_text())
    candidate = {
        "schema": "nvidia_a800_tensor_iterator_copy_task_candidate_v1",
        "hardware_id": "nvidia_a800",
        "model": "isolated_tensor_iterator_strided_copy_task_v1",
        "cycle_unit": "sm_cycles_per_complete_task",
        "timer_domain": "cuda_event_gpu_timeline",
        "software_revision": "pytorch_2.8.0_cuda_12.8_sm80_source_semantics",
        "cache_policy": "warm_default_cache",
        "reference_sm_clock_hz": reference_clock_hz,
        "launch_rank": 1,
        "threads_per_cta": 128,
        "minimum_grid_size": min(EXPECTED_GRIDS),
        "maximum_grid_size": max(EXPECTED_GRIDS),
        "fanout_activation_grid": FANOUT_ACTIVATION_GRID,
        "resident_ctas_per_sm": 16,
        "train_trials": list(TRAIN_TRIALS),
        "validation_trials": list(VALIDATION_TRIALS),
        "entries": entries,
        "capture_source_sha256": provenance["source_sha256"],
        "capture_runner_sha256": provenance["runner_sha256"],
        "production_or_evaluation_duration_consumed": False,
        "allowed_conclusion": (
            "A source-shaped non-contiguous TensorIterator offset program has "
            "a fixed launch/fanout cost plus per-CTA service, expressed in SM "
            "cycles for rank-4 BF16 vt4 and rank-3 FP32 vt2 copies."),
        "excluded_conclusions": [
            "production FlashAttention duration",
            "a per-shape or per-kernel-name latency correction",
            "a limit of eight simultaneously outstanding requests per warp",
        ],
    }
    (capture / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    (capture / "report_identity.json").write_text(
        json.dumps({
            "report_sha256": _sha256(Path(__file__).resolve()),
            "raw_sha256": _sha256(capture / "raw.csv"),
            "provenance_sha256": _sha256(capture / "provenance.json"),
        }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
