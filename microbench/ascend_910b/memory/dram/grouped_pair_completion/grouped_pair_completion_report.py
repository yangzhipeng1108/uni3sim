#!/usr/bin/env python3
"""Validate finite GroupedMatmul GM-to-L1 completion-span captures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


PAIR_COUNTS = (256, 1024, 4096, 8192)
RUN_INDEX_COUNT = 3
ACTIVE_AICS = 24
TICK_NS = 20.0
MAX_WITHIN_COUNT_RELATIVE_SPAN = 0.10
MAX_RATE_RELATIVE_SPAN = 0.10
RAW_FIELDS = (
    "run_id", "run_index", "order_position", "process_pid", "pairs",
    "active_aics", "a_rows", "b_columns", "base_k", "a_residency",
    "b_residency", "drain_interval", "eviction_bytes", "min_cycles",
    "max_cycles", "tick_ns", "completion_span_ns", "wall_ns",
    "validation",
)
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "README.md",
    "grouped_pair_completion_kernel.cpp",
    "grouped_pair_completion_main.cpp",
    "grouped_pair_completion_runner.py",
)
REGISTRATION_FIELDS = {
    "hardware_id", "owner", "physical_device_id", "baseline",
    "runner_pid", "device_id", "run_id", "run_index", "binary",
    "binary_sha256", "source_snapshot_sha256",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source is missing: {path}")
        payload = path.read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _ordered_pairs(run_index):
    rotation = run_index % len(PAIR_COUNTS)
    cells = PAIR_COUNTS[rotation:] + PAIR_COUNTS[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _relative_span(values):
    center = statistics.median(values)
    return (max(values) - min(values)) / center


def _load_registration(path, expected_run_index):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ValueError(f"{path} must contain exactly one registration line")
    fields = {}
    for token in lines[0].split():
        name, separator, value = token.partition("=")
        if separator != "=" or not name or not value or name in fields:
            raise ValueError(f"{path} contains an invalid registration token")
        fields[name] = value
    if set(fields) != REGISTRATION_FIELDS:
        raise ValueError(f"{path} has an invalid registration schema")
    expected = {
        "hardware_id": "ascend_910b",
        "owner": "fused_moe_worker",
        "physical_device_id": "2",
        "baseline": "f1f021d3aac254992b16fd92b35534e016b196f5",
        "device_id": "2",
        "run_id": f"completion-long-r1-{expected_run_index}",
        "run_index": str(expected_run_index),
    }
    if any(fields[name] != value for name, value in expected.items()):
        raise ValueError(f"{path} changed the registered hardware run")
    if int(fields["runner_pid"]) <= 0:
        raise ValueError(f"{path} has an invalid runner PID")
    for name in ("binary_sha256", "source_snapshot_sha256"):
        value = fields[name]
        if len(value) != 64 or any(character not in "0123456789abcdef"
                                   for character in value):
            raise ValueError(f"{path} has an invalid {name}")
    return fields


def _load_run(path, expected_run_index):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        raw_rows = list(reader)
    expected_pairs = _ordered_pairs(expected_run_index)
    if len(raw_rows) != len(expected_pairs):
        raise ValueError(f"{path} must contain exactly four cells")
    rows = []
    for position, (raw, pairs) in enumerate(zip(raw_rows, expected_pairs)):
        context = f"{path} row {position + 2}"
        expected_text = {
            "run_id": f"completion-long-r1-{expected_run_index}",
            "a_residency": "resident",
            "b_residency": "cold",
            "validation": "pass",
        }
        if any(raw[name] != value for name, value in expected_text.items()):
            raise ValueError(f"{context} changed the frozen identity")
        expected_integers = {
            "run_index": expected_run_index,
            "order_position": position,
            "pairs": pairs,
            "active_aics": ACTIVE_AICS,
            "a_rows": 16,
            "b_columns": 256,
            "base_k": 64,
            "drain_interval": 4,
            "eviction_bytes": 537919488,
        }
        if any(int(raw[name]) != value
               for name, value in expected_integers.items()):
            raise ValueError(f"{context} changed the frozen mechanism cell")
        process_pid = int(raw["process_pid"])
        minimum_cycles = int(raw["min_cycles"])
        maximum_cycles = int(raw["max_cycles"])
        wall_ns = int(raw["wall_ns"])
        tick_ns = float(raw["tick_ns"])
        completion_span_ns = float(raw["completion_span_ns"])
        if (process_pid <= 0 or minimum_cycles <= 0
                or maximum_cycles < minimum_cycles or wall_ns <= 0):
            raise ValueError(f"{context} has invalid measured intervals")
        if (not math.isclose(tick_ns, TICK_NS)
                or not math.isclose(
                    completion_span_ns, maximum_cycles * TICK_NS)
                or completion_span_ns >= wall_ns):
            raise ValueError(f"{context} has inconsistent derivation")
        rows.append({
            "run_index": expected_run_index,
            "pairs": pairs,
            "completion_span_ns": completion_span_ns,
        })
    return rows


def build_report(raw_paths, stderr_paths, source_root, *, host, container_name, snapshot_path):
    if not host.strip() or not container_name.strip():
        raise ValueError("capture host and container identity must be nonempty")
    if not Path(snapshot_path).is_absolute():
        raise ValueError("capture snapshot path must be absolute")
    if len(raw_paths) != RUN_INDEX_COUNT or len(stderr_paths) != RUN_INDEX_COUNT:
        raise ValueError("the campaign requires exactly three raw and stderr files")
    rows = []
    registrations = []
    for run_index, (raw_path, stderr_path) in enumerate(
            zip(raw_paths, stderr_paths)):
        registrations.append(_load_registration(stderr_path, run_index))
        rows.extend(_load_run(raw_path, run_index))
    binary_hashes = {item["binary_sha256"] for item in registrations}
    source_hashes = {item["source_snapshot_sha256"] for item in registrations}
    if len(binary_hashes) != 1 or len(source_hashes) != 1:
        raise ValueError("independent runs changed the binary or source snapshot")
    source_snapshot_sha256 = _source_snapshot_sha256(source_root)
    if source_hashes != {source_snapshot_sha256}:
        raise ValueError("captured source snapshot differs from the report source")
    medians = {
        str(pairs): statistics.median([
            row["completion_span_ns"] for row in rows
            if row["pairs"] == pairs
        ])
        for pairs in PAIR_COUNTS
    }
    within_count_spans = {
        str(pairs): _relative_span([
            row["completion_span_ns"] for row in rows
            if row["pairs"] == pairs
        ])
        for pairs in PAIR_COUNTS
    }
    median_rates = {
        str(pairs): medians[str(pairs)] / pairs for pairs in PAIR_COUNTS
    }
    rate_relative_span = _relative_span(list(median_rates.values()))
    stability_gate = max(within_count_spans.values()) <= (
        MAX_WITHIN_COUNT_RELATIVE_SPAN)
    constant_rate_gate = rate_relative_span <= MAX_RATE_RELATIVE_SPAN
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "aic_grouped_matmul_long_stream_completion_v1",
        "duration_unit": "ns_per_completed_M16_N256_K64_operand_pair_per_aic",
        "publication_scope": {
            "a_residency": "l2_resident",
            "b_residency": "dram_cold",
            "active_aic_groups": ACTIVE_AICS,
            "a_rows": 16,
            "a_columns": 64,
            "b_rows": 64,
            "b_columns": 256,
            "pair_bytes": 34816,
            "drain_interval_pairs": 4,
            "minimum_pairs_per_worker": min(PAIR_COUNTS),
            "maximum_pairs_per_worker": max(PAIR_COUNTS),
            "measured_pairs_per_worker": list(PAIR_COUNTS),
            "eviction_bytes": 537919488,
        },
        "measurement": {
            "median_completion_span_ns": medians,
            "median_ns_per_pair": median_rates,
            "published_completion_ns_per_pair": max(median_rates.values()),
            "within_count_relative_spans": within_count_spans,
            "rate_relative_span": rate_relative_span,
            "maximum_allowed_within_count_relative_span": (
                MAX_WITHIN_COUNT_RELATIVE_SPAN),
            "maximum_allowed_rate_relative_span": MAX_RATE_RELATIVE_SPAN,
            "stability_gate": stability_gate,
            "constant_rate_gate": constant_rate_gate,
            "mechanism_gate_passed": stability_gate and constant_rate_gate,
        },
        "provenance": {
            "measurement_baseline_commit": (
                "f1f021d3aac254992b16fd92b35534e016b196f5"),
            "snapshot_path": snapshot_path,
            "raw_root": (
                "profiler_output/microbench/ascend_910b/"
                "grouped_pair_completion_c39cc350_20260906_long_r1"),
            "host": host,
            "container": container_name,
            "physical_device_id": 2,
            "logical_device_id": 0,
            "owner": "fused_moe_worker",
            "source_snapshot_sha256": source_snapshot_sha256,
            "source_sha256": {
                name: _sha256(source_root / name)
                for name in SOURCE_FILENAMES
            },
            "binary_sha256": next(iter(binary_hashes)),
            "raw_sha256": [_sha256(path) for path in raw_paths],
            "stderr_sha256": [_sha256(path) for path in stderr_paths],
            "command": (
                "python grouped_pair_completion_runner.py --binary "
                "build/grouped_pair_completion --device-id 2 --owner "
                "fused_moe_worker --baseline "
                "f1f021d3aac254992b16fd92b35534e016b196f5 --run-id "
                "completion-long-r1-${run_index} --run-index ${run_index}"),
        },
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": stability_gate and constant_rate_gate,
        "forward_eligible": stability_gate and constant_rate_gate,
        "forward_eligibility_constraint": (
            "use only as a lower bound for 24-AIC BF16 M16/N256/K64 "
            "operand streams with 256--8192 pairs per worker, four-pair drain "
            "semantics, resident-or-slower A, and DRAM-cold B"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--raw", nargs=3, required=True, type=Path)
    parser.add_argument("--stderr", nargs=3, required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
                 args.raw,
                 args.stderr,
                 Path(__file__).resolve().parent,
                 host=args.host,
                 container_name=args.container_name,
                 snapshot_path=args.snapshot_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["mechanism_publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
