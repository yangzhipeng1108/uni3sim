#!/usr/bin/env python3
"""Publish a validated graph-replay BF16 TensorIterator task lower bound."""

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
    "probe_operator", "probe_elements", "repetition_counts",
    "samples_per_cell", "warmups_per_cell", "results",
}
EXPECTED_TRIALS = (1, 2, 3)
DERIVATION_TRIALS = (1, 2)
VALIDATION_TRIAL = 3
MAXIMUM_FLOOR_REGIME_MEDIAN_SPREAD = 0.10


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_trial(path, expected_trial):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != EXPECTED_FIELDS:
        raise ValueError(f"task-floor trial schema changed: {path}")
    if payload["schema_version"] != 2:
        raise ValueError("task-floor trial schema_version must be two")
    if payload["hardware_id"] != "metax_c500":
        raise ValueError("task-floor trial hardware_id must be metax_c500")
    if payload["trial"] != expected_trial:
        raise ValueError(
            f"task-floor trial identity mismatch: expected {expected_trial}")
    if payload["timer"] != (
            "paired_differential_events_around_cuda_graph_replay"):
        raise ValueError("task-floor trial timer contract changed")
    for name in (
            "profiler_attached", "production_or_evaluation_duration_consumed",
            "target_operator_used"):
        if payload[name] is not False:
            raise ValueError(f"task-floor {name} must be false")
    if payload["mechanism"] != (
            "warm_bf16_binary_tensor_iterator_task_floor"):
        raise ValueError("task-floor mechanism identity changed")
    if payload["probe_operator"] != "torch.maximum":
        raise ValueError("task-floor probe operator must be torch.maximum")
    if payload["probe_elements"] != [1, 1024, 16384, 1048576]:
        raise ValueError("task-floor probe element sweep changed")
    if payload["repetition_counts"] != [100, 500]:
        raise ValueError("task-floor repetition counts changed")
    expected_keys = {str(value) for value in payload["probe_elements"]}
    if set(payload["results"]) != expected_keys:
        raise ValueError("task-floor result cells changed")
    sample_count = payload["samples_per_cell"]
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count <= 0):
        raise ValueError("task-floor samples_per_cell must be positive")
    for key, cell in payload["results"].items():
        if set(cell) != {"samples", "minimum_ns", "median_ns", "maximum_ns"}:
            raise ValueError(f"task-floor cell schema changed for {key}")
        samples = cell["samples"]
        if not isinstance(samples, list) or len(samples) != sample_count:
            raise ValueError(f"task-floor sample count changed for {key}")
        values = []
        for sample, row in enumerate(samples):
            if set(row) != {
                    "sample", "order", "lower_replay_ns",
                    "higher_replay_ns", "differential_ns"}:
                raise ValueError(f"task-floor sample schema changed for {key}")
            if row["sample"] != sample or sorted(row["order"]) != [0, 1]:
                raise ValueError(f"task-floor sample identity changed for {key}")
            value = row["differential_ns"]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"task-floor differential must be positive for {key}")
            values.append(float(value))
        if float(cell["minimum_ns"]) != min(values):
            raise ValueError(f"task-floor minimum disagrees for {key}")
        if float(cell["median_ns"]) != statistics.median(values):
            raise ValueError(f"task-floor median disagrees for {key}")
        if float(cell["maximum_ns"]) != max(values):
            raise ValueError(f"task-floor maximum disagrees for {key}")
    return payload


def build_report(raw_dir, runner_path):
    raw_root = Path(raw_dir)
    paths = {trial: raw_root / f"trial_{trial}.json"
             for trial in EXPECTED_TRIALS}
    trials = {trial: _load_trial(path, trial)
              for trial, path in paths.items()}
    identity_fields = (
        "hardware_id", "software_revision", "torch_version", "device_id",
        "device_name", "multi_processor_count", "timer", "mechanism",
        "probe_operator", "probe_elements", "repetition_counts",
        "samples_per_cell", "warmups_per_cell",
    )
    reference = trials[EXPECTED_TRIALS[0]]
    for trial, payload in trials.items():
        disagreements = [field for field in identity_fields
                         if payload[field] != reference[field]]
        if disagreements:
            raise ValueError(
                f"task-floor trial {trial} identity differs: {disagreements}")
    keys = tuple(str(value) for value in reference["probe_elements"])
    derivation_samples = [
        float(row["differential_ns"])
        for trial in DERIVATION_TRIALS
        for key in keys
        for row in trials[trial]["results"][key]["samples"]
    ]
    floor_ns = min(derivation_samples)
    validation_samples = [
        float(row["differential_ns"])
        for key in keys
        for row in trials[VALIDATION_TRIAL]["results"][key]["samples"]
    ]
    below_floor = [value for value in validation_samples if value < floor_ns]
    if below_floor:
        raise ValueError(
            "validation contains a BF16 binary TensorIterator task below the "
            f"derived floor: floor={floor_ns}, minimum={min(below_floor)}")
    trial_median_spreads = []
    for key in keys:
        medians = [float(trials[trial]["results"][key]["median_ns"])
                   for trial in EXPECTED_TRIALS]
        trial_median_spreads.append(max(medians) / min(medians) - 1.0)
    maximum_trial_median_spread = max(trial_median_spreads)
    if maximum_trial_median_spread > 0.05:
        raise ValueError(
            "task-floor trial medians vary by more than 5% for one size")
    smallest_probe_median = min(
        float(trials[trial]["results"][key]["median_ns"])
        for trial in EXPECTED_TRIALS
        for key in keys)
    floor_regime = []
    for key in keys:
        maximum_median = max(
            float(trials[trial]["results"][key]["median_ns"])
            for trial in EXPECTED_TRIALS)
        if (maximum_median / smallest_probe_median - 1.0
                <= MAXIMUM_FLOOR_REGIME_MEDIAN_SPREAD):
            floor_regime.append(int(key))
        else:
            break
    if not floor_regime:
        raise ValueError("task-floor probe has no validated floor regime")
    return {
        "schema_version": 2,
        "hardware_id": reference["hardware_id"],
        "software_revision": reference["software_revision"],
        "torch_version": reference["torch_version"],
        "device_name": reference["device_name"],
        "timer_domain": reference["timer"],
        "model": "warm_bf16_binary_tensor_iterator_task_floor_v1",
        "task_class": "pytorch_bf16_binary_tensor_iterator",
        "duration_unit": "ns_per_task",
        "duration_ns": floor_ns,
        "derivation_trials": list(DERIVATION_TRIALS),
        "validation_trial": VALIDATION_TRIAL,
        "derivation_rule": "minimum_all_binary_graph_replay_differentials",
        "validation_sample_count": len(validation_samples),
        "validation_below_floor_count": len(below_floor),
        "maximum_cross_trial_median_spread_fraction": maximum_trial_median_spread,
        "maximum_floor_regime_elements": floor_regime[-1],
        "maximum_floor_regime_median_spread_fraction": (
            MAXIMUM_FLOOR_REGIME_MEDIAN_SPREAD),
        "probe_elements": reference["probe_elements"],
        "repetition_counts": reference["repetition_counts"],
        "production_or_evaluation_duration_consumed": False,
        "target_operator_used": False,
        "profiler_attached": False,
        "runner_sha256": _sha256(runner_path),
        "raw_sha256": {f"trial_{trial}.json": _sha256(path)
                       for trial, path in paths.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(args.raw_dir, args.runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
