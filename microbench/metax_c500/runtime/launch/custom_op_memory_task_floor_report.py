#!/usr/bin/env python3
"""Publish a validated geometry-aware PyTorch custom-op task model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPECTED_FIELDS = {
    "schema_version", "hardware_id", "software_revision", "torch_version",
    "device_id", "device_name", "multi_processor_count", "trial", "pid",
    "timer", "profiler_attached", "mechanism",
    "production_or_evaluation_duration_consumed", "target_operator_used",
    "probe_operators", "grid_sizes", "block_sizes", "wave_size",
    "samples_per_cell", "event_bootstrap_rounds", "warmups_per_cell",
    "source_sha256", "samples_ns", "medians_ns",
}
EXPECTED_TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
EXPECTED_GRID_SIZES = (1, 16, 64, 104, 256, 1024, 4096, 16384)
EXPECTED_BLOCK_SIZES = (64, 256, 512, 1024)
EXPECTED_AP_COUNT = 104
EXPECTED_WAVE_SIZE = 64
MAXIMUM_VALIDATION_MAPE_PERCENT = 20.0
MAXIMUM_VALIDATION_WMAE_PERCENT = 20.0


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trial(path, expected_trial):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_FIELDS:
        raise ValueError(f"custom-op task trial schema changed: {path}")
    expected_identity = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "trial": expected_trial,
        "timer": "torch_cuda_event",
        "profiler_attached": False,
        "mechanism": "warm_pytorch_custom_op_memory_task_floor",
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "probe_operators": [
            "midlayer_microbench.custom_op_empty_task",
            "midlayer_microbench.custom_op_memory_dependency_task",
        ],
        "grid_sizes": list(EXPECTED_GRID_SIZES),
        "block_sizes": list(EXPECTED_BLOCK_SIZES),
        "wave_size": EXPECTED_WAVE_SIZE,
        "multi_processor_count": EXPECTED_AP_COUNT,
    }
    for name, expected in expected_identity.items():
        if payload[name] != expected:
            raise ValueError(
                f"custom-op task trial {name} must be {expected!r}")
    sample_count = payload["samples_per_cell"]
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count <= 0):
        raise ValueError("custom-op samples_per_cell must be positive")
    expected_keys = {
        f"{kind}:grid{grid}:block{block}"
        for kind in ("empty", "memory")
        for grid in EXPECTED_GRID_SIZES
        for block in EXPECTED_BLOCK_SIZES
    }
    if (set(payload["samples_ns"]) != expected_keys
            or set(payload["medians_ns"]) != expected_keys):
        raise ValueError("custom-op task sample cells changed")
    for key in expected_keys:
        samples = payload["samples_ns"][key]
        if not isinstance(samples, list) or len(samples) != sample_count:
            raise ValueError(f"custom-op sample count changed for {key}")
        if any(isinstance(value, bool)
               or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0
               for value in samples):
            raise ValueError(f"custom-op samples must be positive for {key}")
        median = statistics.median(float(value) for value in samples)
        if median != float(payload["medians_ns"][key]):
            raise ValueError(f"custom-op median disagrees for {key}")
    return payload


def _solve_three_by_three(matrix, vector):
    augmented = [
        [float(value) for value in row] + [float(rhs)]
        for row, rhs in zip(matrix, vector)
    ]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(
            augmented[row][column]))
        if augmented[pivot][column] == 0:
            raise ValueError("custom-op geometry fit is singular")
        augmented[column], augmented[pivot] = (
            augmented[pivot], augmented[column])
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            multiplier = augmented[row][column]
            augmented[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(
                    augmented[row], augmented[column])
            ]
    return tuple(augmented[row][3] for row in range(3))


def _fit_scheduler(rows):
    # The fixed model is one custom-op submission, one descriptor per CTA,
    # and one wave64 service per full-chip scheduling round. Only the three
    # mechanism coefficients are derived from the independent benchmark.
    normal_matrix = [[0.0] * 3 for _ in range(3)]
    normal_vector = [0.0] * 3
    for row in rows:
        features = (
            1.0,
            float(row["grid_size"]),
            float(math.ceil(row["grid_size"] / EXPECTED_AP_COUNT)
                  * math.ceil(row["block_size"] / EXPECTED_WAVE_SIZE)),
        )
        for left in range(3):
            normal_vector[left] += features[left] * row["duration_ns"]
            for right in range(3):
                normal_matrix[left][right] += (
                    features[left] * features[right])
    coefficients = _solve_three_by_three(normal_matrix, normal_vector)
    if any(not math.isfinite(value) or value <= 0
           for value in coefficients):
        raise ValueError(
            "custom-op scheduler coefficients must be finite and positive")
    return coefficients


def _predict(coefficients, grid_size, block_size):
    submit_ns, cta_descriptor_ns, wave_round_ns = coefficients
    waves_per_cta = math.ceil(block_size / EXPECTED_WAVE_SIZE)
    scheduling_rounds = math.ceil(grid_size / EXPECTED_AP_COUNT)
    return (
        submit_ns
        + grid_size * cta_descriptor_ns
        + scheduling_rounds * waves_per_cta * wave_round_ns
    )


def build_report(raw_dir, source_path, runner_path):
    raw_root = Path(raw_dir)
    paths = {
        trial: raw_root / f"trial_{trial}.json"
        for trial in EXPECTED_TRIALS
    }
    trials = {
        trial: _load_trial(path, trial)
        for trial, path in paths.items()
    }
    identity_fields = (
        "hardware_id", "software_revision", "torch_version", "device_id",
        "device_name", "multi_processor_count", "timer", "mechanism",
        "probe_operators", "grid_sizes", "block_sizes", "wave_size",
        "samples_per_cell", "event_bootstrap_rounds", "warmups_per_cell",
        "source_sha256",
    )
    reference = trials[EXPECTED_TRIALS[0]]
    for trial, payload in trials.items():
        disagreements = [
            field for field in identity_fields
            if payload[field] != reference[field]
        ]
        if disagreements:
            raise ValueError(
                f"custom-op trial {trial} identity differs: {disagreements}")
    if reference["source_sha256"] != _sha256(source_path):
        raise ValueError("custom-op source does not match the captured source")

    derivation_rows = []
    for grid_size in EXPECTED_GRID_SIZES:
        for block_size in EXPECTED_BLOCK_SIZES:
            key = f"empty:grid{grid_size}:block{block_size}"
            values = [
                float(value)
                for trial in DERIVATION_TRIALS
                for value in trials[trial]["samples_ns"][key]
            ]
            derivation_rows.append({
                "grid_size": grid_size,
                "block_size": block_size,
                "duration_ns": statistics.median(values),
            })
    coefficients = _fit_scheduler(derivation_rows)
    validation = []
    for grid_size in EXPECTED_GRID_SIZES:
        for block_size in EXPECTED_BLOCK_SIZES:
            key = f"empty:grid{grid_size}:block{block_size}"
            measured = float(
                trials[VALIDATION_TRIAL]["medians_ns"][key])
            predicted = _predict(coefficients, grid_size, block_size)
            validation.append({
                "grid_size": grid_size,
                "block_size": block_size,
                "measured_ns": measured,
                "predicted_ns": predicted,
                "ape_percent": abs(predicted - measured) / measured * 100.0,
            })
    validation_mape = statistics.fmean(
        row["ape_percent"] for row in validation)
    validation_wmae = (
        math.fsum(abs(row["predicted_ns"] - row["measured_ns"])
                  for row in validation)
        / math.fsum(row["measured_ns"] for row in validation)
        * 100.0)
    validation_maximum_ape = max(
        row["ape_percent"] for row in validation)
    if (validation_mape >= MAXIMUM_VALIDATION_MAPE_PERCENT
            or validation_wmae >= MAXIMUM_VALIDATION_WMAE_PERCENT):
        raise ValueError(
            "custom-op scheduler model failed frozen validation: "
            f"MAPE={validation_mape}, WMAE={validation_wmae}")

    submit_ns, cta_descriptor_ns, wave_round_ns = coefficients
    return {
        "schema_version": 1,
        "hardware_id": reference["hardware_id"],
        "software_revision": reference["software_revision"],
        "torch_version": reference["torch_version"],
        "device_name": reference["device_name"],
        "timer_domain": "torch_cuda_event",
        "model": "warm_pytorch_custom_op_scheduler_v1",
        "task_class": (
            "pytorch_cpp_extension_independent_cta_memory_task"),
        "duration_unit": "ns_per_task",
        "derivation_rule": "least_squares_physical_scheduler_geometry",
        "submit_ns": submit_ns,
        "cta_descriptor_ns": cta_descriptor_ns,
        "wave_round_ns": wave_round_ns,
        "minimum_grid_size": min(EXPECTED_GRID_SIZES),
        "maximum_grid_size": max(EXPECTED_GRID_SIZES),
        "minimum_block_size": min(EXPECTED_BLOCK_SIZES),
        "maximum_block_size": max(EXPECTED_BLOCK_SIZES),
        "ap_count": EXPECTED_AP_COUNT,
        "wave_size": EXPECTED_WAVE_SIZE,
        "derivation_trials": list(DERIVATION_TRIALS),
        "validation_trial": VALIDATION_TRIAL,
        "validation": validation,
        "validation_mape_percent": validation_mape,
        "validation_wmae_percent": validation_wmae,
        "validation_maximum_ape_percent": validation_maximum_ape,
        "maximum_allowed_validation_mape_percent": (
            MAXIMUM_VALIDATION_MAPE_PERCENT),
        "maximum_allowed_validation_wmae_percent": (
            MAXIMUM_VALIDATION_WMAE_PERCENT),
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "profiler_attached": False,
        "source_sha256": _sha256(source_path),
        "runner_sha256": _sha256(runner_path),
        "raw_sha256": {
            f"trial_{trial}.json": _sha256(path)
            for trial, path in paths.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.raw_dir, args.source, args.runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
