#!/usr/bin/env python3
"""Publish the independent CANN FIA regular-GQA score-copy service."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


_SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_nd2nz_stride_kernel.cpp",
    "aic_mte2_nd2nz_stride_main.cpp",
    "aic_mte2_nd2nz_stride_runner.py",
)
_ROWS = (4, 5, 8)
_COLUMNS = 256
_SOURCE_STRIDES = (512, 1024, 2048, 4096, 8192)
_ACTIVE_AIC_GROUPS = (8, 16, 22, 24)
_MODES = (0, 1)
_TRIALS = (1, 2, 3, 4, 5)
_EXPECTED_HEADER = (
    "run_id", "trial", "order_direction", "order_position", "process_pid",
    "rows", "columns", "source_row_stride", "mode", "source_tile_span_bytes",
    "copied_bytes", "active_aic_groups", "iterations", "short_cycles",
    "short_wall_ns", "long_cycles", "long_wall_ns", "tick_ns",
    "cycles_per_copy", "ns_per_copy", "bandwidth_gbs", "validation",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(source_root):
    digest = hashlib.sha256()
    for filename in _SOURCE_FILENAMES:
        path = source_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _integer(row, name):
    value = int(row[name])
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_positive(row, name):
    value = float(row[name])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _expected_cells():
    return {
        (rows, _COLUMNS, source_stride, active_aic_groups, mode)
        for rows in _ROWS
        for source_stride in _SOURCE_STRIDES
        for active_aic_groups in _ACTIVE_AIC_GROUPS
        for mode in _MODES
    }


def _load_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _EXPECTED_HEADER:
            raise ValueError("raw CSV header differs from the frozen runner schema")
        grouped = {}
        for row in reader:
            if row["validation"] != "pass":
                raise ValueError("raw CSV contains a failed validation record")
            rows = _integer(row, "rows")
            columns = _integer(row, "columns")
            source_stride = _integer(row, "source_row_stride")
            active_aic_groups = _integer(row, "active_aic_groups")
            mode = int(row["mode"])
            if mode not in _MODES:
                raise ValueError("raw CSV contains an unknown completion mode")
            trial = _integer(row, "trial")
            if trial not in _TRIALS:
                raise ValueError("raw CSV contains an unexpected trial number")
            copied_bytes = _integer(row, "copied_bytes")
            if copied_bytes != rows * columns * 2:
                raise ValueError("raw CSV copied byte count disagrees with BF16 geometry")
            _finite_positive(row, "ns_per_copy")
            key = (rows, columns, source_stride, active_aic_groups, mode)
            if key not in _expected_cells():
                raise ValueError("raw CSV contains a geometry outside the frozen domain")
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((trial, _finite_positive(row, "ns_per_copy")))
    if set(grouped) != _expected_cells():
        raise ValueError("raw CSV does not contain the full frozen rectangular domain")
    normalized = {}
    for key, samples in grouped.items():
        if len(samples) != len(_TRIALS):
            raise ValueError("each source descriptor/mode must have five fresh samples")
        trial_numbers = tuple(sorted(sample[0] for sample in samples))
        if trial_numbers != _TRIALS:
            raise ValueError("each source descriptor/mode must contain trials one through five")
        values = tuple(value for _, value in sorted(samples))
        median = float(statistics.median(values))
        relative_span_pct = (max(values) - min(values)) / median * 100.0
        if relative_span_pct > 5.0:
            raise ValueError("source descriptor/mode exceeds the five percent stability gate")
        normalized[key] = {
            "process_ns_per_copy": values,
            "median_ns_per_copy": median,
            "max_relative_span_pct": relative_span_pct,
        }
    return normalized


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-repository-path", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--physical-device-id", type=int, required=True)
    parser.add_argument("--binary-sha256", required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.physical_device_id < 0:
        raise ValueError("physical device id must be non-negative")
    if len(args.binary_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in args.binary_sha256):
        raise ValueError("binary SHA256 must be lowercase hexadecimal")
    if (not args.raw.is_file()
            or not args.source_root.is_dir()
            or not args.raw_repository_path.startswith(
                "profiler_output/microbench/ascend_910b/")):
        raise ValueError("raw and source inputs are outside the evidence contract")
    records = _load_rows(args.raw)
    services = []
    for rows in _ROWS:
        for source_stride in _SOURCE_STRIDES:
            for active_aic_groups in _ACTIVE_AIC_GROUPS:
                serial = records[(rows, _COLUMNS, source_stride,
                                  active_aic_groups, 0)]
                stream = records[(rows, _COLUMNS, source_stride,
                                  active_aic_groups, 1)]
                services.append({
                    "rows": rows,
                    "columns": _COLUMNS,
                    "source_row_stride_elements": source_stride,
                    "active_aic_groups": active_aic_groups,
                    "serial_completion_ns": serial["median_ns_per_copy"],
                    "two_buffer_stream_ns": stream["median_ns_per_copy"],
                    "serial_process_ns": list(serial["process_ns_per_copy"]),
                    "stream_process_ns": list(stream["process_ns_per_copy"]),
                    "serial_max_relative_span_pct": serial[
                        "max_relative_span_pct"],
                    "stream_max_relative_span_pct": stream[
                        "max_relative_span_pct"],
                })
    result = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism": "aic_mte2_fia_regular_gqa_score_panel",
        "duration_unit": "ns_per_nd2nz_descriptor_per_aic",
        "publication_scope": {
            "dtype": "bf16",
            "source_residency": "disjoint_512mb_dram_stream",
            "source_operation": "regular_gqa_CopyPToL1_Nd2NzParams",
            "completion_edges": {
                "serial": "MTE2_completion_after_each_descriptor",
                "two_buffer_stream": "source_ordered_two_buffer_stream",
            },
            "estimator": "exact_measured_geometry_and_aic_topology_only",
            "services": services,
        },
        "measurement": {
            "mode": "independent_aic_nd2nz_descriptor_microbenchmark",
            "measurement_baseline_commit": args.baseline_commit,
            "source_commit": args.source_commit,
            "physical_device_id": args.physical_device_id,
            "binary_sha256": args.binary_sha256,
            "source_snapshot_sha256": _source_snapshot_sha256(
                args.source_root),
            "source_files_sha256": {
                filename: _sha256(args.source_root / filename)
                for filename in _SOURCE_FILENAMES
            },
            "raw_path": args.raw_repository_path,
            "raw_sha256": _sha256(args.raw),
            "raw_row_count": sum(len(value["process_ns_per_copy"])
                                 for value in records.values()),
            "fresh_processes_per_cell": len(_TRIALS),
            "maximum_observed_relative_span_pct": max(
                value["max_relative_span_pct"] for value in records.values()),
            "mechanism_gate_passed": True,
        },
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": True,
        "forward_eligible": False,
        "forward_eligibility_constraint": (
            "not eligible for Forward: the independent descriptor uses a "
            "disjoint DRAM stream, while the source P-panel residency and "
            "completion edge have not been independently matched; no direct "
            "replacement, interpolation, or extrapolation is permitted"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
