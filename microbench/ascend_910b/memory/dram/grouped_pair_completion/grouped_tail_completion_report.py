#!/usr/bin/env python3
"""Validate wide-base GroupedMatmul tail-stream completion captures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


CAMPAIGNS = {
    "m64_tail16": (64, 16),
    "m64_tail32": (64, 32),
    "m128_tail16": (128, 16),
    "m128_tail32": (128, 32),
}
PAIR_COUNTS = (512, 1024, 3072, 6144)
RUN_INDEX_COUNT = 3
ACTIVE_AICS = 24
TICK_NS = 20.0
MAX_WITHIN_COUNT_RELATIVE_SPAN = 0.10
MAX_RATE_RELATIVE_SPAN = 0.10
MAX_CROSS_GEOMETRY_RATE_RELATIVE_SPAN = 0.10
BASELINE = "e83efef7291ed8cb91706594a40598bf2dda6958"
RAW_ROOT = (
    "profiler_output/microbench/ascend_910b/"
    "grouped_pair_tail_e83efef7_20260908_r2")
RAW_FIELDS = (
    "run_id", "run_index", "order_position", "process_pid", "pairs",
    "active_aics", "a_rows", "b_columns", "base_k", "a_residency",
    "b_residency", "drain_interval", "eviction_bytes", "min_cycles",
    "max_cycles", "tick_ns", "completion_span_ns", "wall_ns",
    "validation",
)
REGISTRATION_FIELDS = {
    "hardware_id", "owner", "physical_device_id", "baseline",
    "runner_pid", "device_id", "run_id", "run_index", "campaign",
    "binary", "binary_sha256", "source_snapshot_sha256",
}
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "README.md",
    "grouped_pair_completion_kernel.cpp",
    "grouped_pair_completion_main.cpp",
    "grouped_pair_completion_runner.py",
)


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


def _load_registration(path, campaign, expected_run_index, *, container_snapshot_path):
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
        "baseline": BASELINE,
        "device_id": "2",
        "run_id": f"{campaign}-r1-{expected_run_index}",
        "run_index": str(expected_run_index),
        "campaign": campaign,
    }
    if any(fields[name] != value for name, value in expected.items()):
        raise ValueError(f"{path} changed the registered hardware run")
    if not PurePosixPath(container_snapshot_path).is_absolute():
        raise ValueError("container snapshot path must be absolute")
    expected_binary = PurePosixPath(container_snapshot_path) / (
            "microbench/ascend_910b/memory/dram/grouped_pair_completion/"
            f"build_{campaign}/grouped_pair_completion")
    if PurePosixPath(fields["binary"]) != expected_binary:
        raise ValueError(f"{path} changed the registered binary path")
    if int(fields["runner_pid"]) <= 0:
        raise ValueError(f"{path} has an invalid runner PID")
    for name in ("binary_sha256", "source_snapshot_sha256"):
        value = fields[name]
        if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{path} has an invalid {name}")
    return fields


def _load_build_cache(path, expected_base_m, expected_active_m):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GROUPED_PAIR_BASE_M:"):
            values["base_m"] = int(line.partition("=")[2])
        elif line.startswith("GROUPED_PAIR_ACTIVE_M:"):
            values["active_m"] = int(line.partition("=")[2])
    if values != {"base_m": expected_base_m, "active_m": expected_active_m}:
        raise ValueError(f"{path} changed the compiled tail geometry")


def _load_run(path, campaign, expected_run_index, expected_active_m):
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
            "run_id": f"{campaign}-r1-{expected_run_index}",
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
            "a_rows": expected_active_m,
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


def build_report(raw_root, source_root, *, host, container_name, snapshot_path, container_snapshot_path):
    if not host.strip() or not container_name.strip():
        raise ValueError("capture host and container identity must be nonempty")
    if not Path(snapshot_path).is_absolute():
        raise ValueError("capture snapshot path must be absolute")
    raw_root = Path(raw_root)
    source_root = Path(source_root)
    campaign_reports = {}
    all_rates = []
    provenance = {}
    source_hashes = set()
    for campaign, (base_m, active_m) in CAMPAIGNS.items():
        cache_path = raw_root / f"{campaign}_CMakeCache.txt"
        _load_build_cache(cache_path, base_m, active_m)
        rows = []
        registrations = []
        raw_paths = []
        stderr_paths = []
        for run_index in range(RUN_INDEX_COUNT):
            raw_path = raw_root / f"{campaign}_run{run_index}.csv"
            stderr_path = raw_root / f"{campaign}_run{run_index}.stderr"
            raw_paths.append(raw_path)
            stderr_paths.append(stderr_path)
            registrations.append(
                _load_registration(
                    stderr_path,
                    campaign,
                    run_index,
                    container_snapshot_path=container_snapshot_path))
            rows.extend(_load_run(
                raw_path, campaign, run_index, active_m))
        binary_hashes = {item["binary_sha256"] for item in registrations}
        if len(binary_hashes) != 1:
            raise ValueError(f"{campaign} changed binary between runs")
        source_hashes.update(
            item["source_snapshot_sha256"] for item in registrations)
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
            str(pairs): medians[str(pairs)] / pairs
            for pairs in PAIR_COUNTS
        }
        rate_relative_span = _relative_span(list(median_rates.values()))
        stability_gate = max(within_count_spans.values()) <= (
            MAX_WITHIN_COUNT_RELATIVE_SPAN)
        constant_rate_gate = rate_relative_span <= MAX_RATE_RELATIVE_SPAN
        all_rates.extend(median_rates.values())
        campaign_reports[campaign] = {
            "base_m": base_m,
            "active_m": active_m,
            "median_completion_span_ns": medians,
            "median_ns_per_pair": median_rates,
            "within_count_relative_spans": within_count_spans,
            "rate_relative_span": rate_relative_span,
            "stability_gate": stability_gate,
            "constant_rate_gate": constant_rate_gate,
        }
        provenance[campaign] = {
            "binary_sha256": next(iter(binary_hashes)),
            "cmake_cache_sha256": _sha256(cache_path),
            "raw_sha256": [_sha256(path) for path in raw_paths],
            "stderr_sha256": [_sha256(path) for path in stderr_paths],
        }
    source_snapshot_sha256 = _source_snapshot_sha256(source_root)
    if source_hashes != {source_snapshot_sha256}:
        raise ValueError("captured source snapshot differs from report source")
    cross_geometry_rate_span = _relative_span(all_rates)
    mechanism_gate_passed = (
        all(item["stability_gate"] and item["constant_rate_gate"]
            for item in campaign_reports.values())
        and cross_geometry_rate_span <= MAX_CROSS_GEOMETRY_RATE_RELATIVE_SPAN)
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "software_revision": "cann_8.5.1",
        "mechanism_id": "aic_grouped_matmul_wide_base_tail_completion_v1",
        "duration_unit": (
            "ns_per_completed_tail_N256_K64_operand_pair_per_aic"),
        "publication_scope": {
            "base_m_values": [64, 128],
            "active_m_values": [16, 32],
            "a_residency": "l2_resident",
            "b_residency": "dram_cold",
            "active_aic_groups": ACTIVE_AICS,
            "a_columns": 64,
            "b_rows": 64,
            "b_columns": 256,
            "drain_interval_pairs": 4,
            "minimum_pairs_per_worker": min(PAIR_COUNTS),
            "maximum_pairs_per_worker": max(PAIR_COUNTS),
            "measured_pairs_per_worker": list(PAIR_COUNTS),
            "eviction_bytes": 537919488,
        },
        "measurement": {
            "campaigns": campaign_reports,
            "published_completion_ns_per_pair": max(all_rates),
            "cross_geometry_rate_relative_span": cross_geometry_rate_span,
            "maximum_allowed_within_count_relative_span": (
                MAX_WITHIN_COUNT_RELATIVE_SPAN),
            "maximum_allowed_rate_relative_span": MAX_RATE_RELATIVE_SPAN,
            "maximum_allowed_cross_geometry_rate_relative_span": (
                MAX_CROSS_GEOMETRY_RATE_RELATIVE_SPAN),
            "mechanism_gate_passed": mechanism_gate_passed,
        },
        "provenance": {
            "measurement_baseline_commit": BASELINE,
            "snapshot_path": snapshot_path,
            "raw_root": RAW_ROOT,
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
            "campaigns": provenance,
            "build_command": (
                "cmake -S <source> -B build_<campaign> "
                "-DSOC_VERSION=Ascend910B2 "
                "-DGROUPED_PAIR_BASE_M=<64|128> "
                "-DGROUPED_PAIR_ACTIVE_M=<16|32> && "
                "cmake --build build_<campaign> -j 8"),
            "run_command": (
                "python grouped_pair_completion_runner.py --binary "
                "build_<campaign>/grouped_pair_completion --device-id 2 "
                "--owner fused_moe_worker --baseline "
                f"{BASELINE} --run-id <campaign>-r1-${{run_index}} "
                "--run-index ${run_index} --campaign <campaign>"),
        },
        "production_or_evaluation_duration_consumed": False,
        "parameter_fit_from_production_duration": False,
        "mechanism_publishable": mechanism_gate_passed,
        "forward_eligible": mechanism_gate_passed,
        "forward_eligibility_constraint": (
            "use only as a lower bound for 24-AIC BF16 M16/M32 tail "
            "operand streams under source-selected base-M64/base-M128, "
            "N256/K64, 512--6144 pairs per worker, four-pair drain "
            "semantics, resident-or-slower A, and DRAM-cold B"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", required=True)
    parser.add_argument("--container-snapshot-path", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
                 args.raw_root,
                 Path(__file__).resolve().parent,
                 host=args.host,
                 container_name=args.container_name,
                 snapshot_path=args.snapshot_path,
                 container_snapshot_path=args.container_snapshot_path)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0 if report["mechanism_publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
