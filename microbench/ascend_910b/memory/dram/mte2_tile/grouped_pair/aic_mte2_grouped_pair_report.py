#!/usr/bin/env python3
"""Audit provenance and raw results for the grouped-MatMul MTE2 pair probe."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = "midlayer.ascend_910b.grouped_mte2_stepk_capture.v1"
RAW_FIELDS = (
    "rows", "columns", "tile_bytes", "mode", "block_dim", "iterations",
    "short_cycles", "short_wall_ns", "long_cycles", "long_wall_ns",
    "tick_ns", "cycles_per_pair", "ns_per_pair", "bandwidth_gbs",
    "validation",
)
EXPECTED_ROWS = (32, 128)
EXPECTED_MODES = (3, 4)
EXPECTED_BLOCK_DIM = 24
EXPECTED_TICK_NS = 20.0
MAX_RELATIVE_SPAN = 0.20


def _require_keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{context} must contain exactly {sorted(expected)}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path):
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    _require_keys(
        manifest,
        ("schema_version", "hardware_id", "device", "software", "snapshot",
         "build", "capture", "limitations"),
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported grouped-pair capture schema")
    if manifest["hardware_id"] != "ascend_910b":
        raise ValueError("capture hardware_id must be ascend_910b")
    _require_keys(
        manifest["device"],
        ("host", "container", "physical_device_id", "logical_device_id",
         "product_name", "serial_number"),
        "device",
    )
    if (manifest["device"]["physical_device_id"] < 0
            or manifest["device"]["logical_device_id"] != 0):
        raise ValueError("capture device mapping is invalid")
    _require_keys(
        manifest["software"],
        ("cann", "container_image", "container_image_id"),
        "software",
    )
    _require_keys(
        manifest["snapshot"],
        ("commit", "path", "source_sha256"),
        "snapshot",
    )
    if len(manifest["snapshot"]["commit"]) != 40:
        raise ValueError("snapshot commit must be a full Git object ID")
    _require_keys(
        manifest["snapshot"]["source_sha256"],
        ("CMakeLists.txt", "aic_mte2_nd2nz_kernel.cpp",
         "aic_mte2_nd2nz_main.cpp"),
        "snapshot source hashes",
    )
    _require_keys(
        manifest["build"], ("binary_path", "binary_sha256"), "build")
    _require_keys(
        manifest["capture"], ("environment", "invocations"), "capture")
    _require_keys(
        manifest["capture"]["environment"],
        ("ASCEND_RT_VISIBLE_DEVICES",),
        "capture environment",
    )
    if (manifest["capture"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"]
            != str(manifest["device"]["physical_device_id"])):
        raise ValueError("capture environment does not select the physical device")
    _require_keys(
        manifest["limitations"],
        ("independent_run_ids", "mixed_residency_cells"),
        "limitations",
    )
    if (manifest["limitations"]["independent_run_ids"] is not False
            or manifest["limitations"]["mixed_residency_cells"] is not False):
        raise ValueError("legacy capture limitations must remain explicit")
    return manifest


def _parse_argv(argv):
    if (not isinstance(argv, list) or len(argv) != 23
            or argv[1] != "--batch"):
        raise ValueError("capture argv must contain one four-cell batch")
    footprint_mb = int(argv[2])
    cells = []
    for offset in range(3, len(argv), 5):
        cells.append(tuple(int(item) for item in argv[offset:offset + 5]))
    expected_cells = {
        (rows, 64, mode, EXPECTED_BLOCK_DIM)
        for rows in EXPECTED_ROWS for mode in EXPECTED_MODES
    }
    observed_cells = {
        (rows, columns, mode, block_dim)
        for rows, columns, mode, block_dim, _ in cells
    }
    if observed_cells != expected_cells:
        raise ValueError("capture argv changed the frozen grouped-pair cells")
    if any(mode == 4 and iterations % 4 != 0
           for _, _, mode, _, iterations in cells):
        raise ValueError("stepK4 cells require iteration counts divisible by four")
    return footprint_mb, cells


def _read_raw(path, invocation):
    _require_keys(
        invocation,
        ("footprint_mb", "argv", "repetitions", "raw_file", "raw_sha256"),
        "capture invocation",
    )
    footprint_mb, cells = _parse_argv(invocation["argv"])
    if footprint_mb != invocation["footprint_mb"]:
        raise ValueError("manifest footprint disagrees with capture argv")
    if invocation["repetitions"] < 4:
        raise ValueError("each grouped-pair cell requires at least four samples")
    if _sha256(path) != invocation["raw_sha256"]:
        raise ValueError(f"raw capture hash mismatch: {path}")
    with path.open(newline="") as stream:
        raw_rows = list(csv.reader(stream))
    expected_count = invocation["repetitions"] * len(cells)
    if len(raw_rows) != expected_count:
        raise ValueError(f"{path} must contain exactly {expected_count} rows")

    decoded = []
    for index, raw in enumerate(raw_rows):
        if len(raw) != len(RAW_FIELDS) or raw[-1] != "pass":
            raise ValueError(f"{path} row {index + 1} has an invalid schema")
        integers = [int(raw[position]) for position in range(10)]
        floats = [float(raw[position]) for position in range(10, 14)]
        (rows, columns, tile_bytes, mode, block_dim, iterations,
         short_cycles, short_wall_ns, long_cycles, long_wall_ns) = integers
        tick_ns, cycles_per_pair, ns_per_pair, bandwidth_gbs = floats
        expected_cell = cells[index % len(cells)]
        if (rows, columns, mode, block_dim, iterations) != expected_cell:
            raise ValueError(f"{path} row {index + 1} changed capture order")
        if tile_bytes != rows * columns * 2:
            raise ValueError(f"{path} row {index + 1} has invalid A bytes")
        if (short_cycles <= 0 or short_wall_ns <= 0
                or long_cycles <= short_cycles or long_wall_ns <= short_wall_ns):
            raise ValueError(f"{path} row {index + 1} has invalid intervals")
        expected_cycles = (long_cycles - short_cycles) / (iterations * 3)
        expected_ns = expected_cycles * EXPECTED_TICK_NS
        pair_bytes = tile_bytes + 64 * 256 * 2
        expected_bandwidth = pair_bytes / expected_ns
        checks = (
            (tick_ns, EXPECTED_TICK_NS),
            (cycles_per_pair, expected_cycles),
            (ns_per_pair, expected_ns),
            (bandwidth_gbs, expected_bandwidth),
        )
        if any(not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
               for actual, expected in checks):
            raise ValueError(f"{path} row {index + 1} has inconsistent derivation")
        decoded.append({
            "footprint_mb": footprint_mb,
            "trial": index // len(cells) + 1,
            "rows": rows,
            "mode": mode,
            "ns_per_pair": ns_per_pair,
        })
    return decoded


def _audit(manifest_path, capture_root, source_root):
    manifest = _load_manifest(manifest_path)
    for filename, expected_hash in manifest["snapshot"]["source_sha256"].items():
        source_path = source_root / filename
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"source identity mismatch: {source_path}")

    rows = []
    footprints = set()
    invocations = manifest["capture"]["invocations"]
    if not isinstance(invocations, list) or len(invocations) != 2:
        raise ValueError("capture must contain exactly two footprint invocations")
    for invocation in invocations:
        raw_path = capture_root / invocation["raw_file"]
        rows.extend(_read_raw(raw_path, invocation))
        footprints.add(invocation["footprint_mb"])
    if footprints != {64, 512}:
        raise ValueError("capture must contain the resident and cold footprints")

    summaries = []
    all_stable = True
    for footprint_mb in sorted(footprints):
        for rows_count in EXPECTED_ROWS:
            values = {
                mode: [
                    row["ns_per_pair"] for row in rows
                    if row["footprint_mb"] == footprint_mb
                    and row["rows"] == rows_count and row["mode"] == mode
                ]
                for mode in EXPECTED_MODES
            }
            continuous = statistics.median(values[3])
            stepk4 = statistics.median(values[4])
            spans = {
                mode: (max(samples) - min(samples)) / statistics.median(samples)
                for mode, samples in values.items()
            }
            stable = max(spans.values()) <= MAX_RELATIVE_SPAN
            all_stable = all_stable and stable
            summaries.append({
                "footprint_mb": footprint_mb,
                "a_rows": rows_count,
                "samples_per_mode": len(values[3]),
                "continuous_median_ns": continuous,
                "stepk4_median_ns": stepk4,
                "stepk4_increase_percent": (stepk4 / continuous - 1.0) * 100.0,
                "continuous_relative_span": spans[3],
                "stepk4_relative_span": spans[4],
                "stability_gate": stable,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_identity_verified": True,
        "stability_gate": all_stable,
        "cells": summaries,
        "production_publishable": False,
        "production_blockers": [
            "capture has no independent run identifiers",
            "capture has no A-resident/B-cold or A-cold/B-resident cells",
            "aggregate pair overhead cannot be allocated to Forward A/B services",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    report = _audit(args.manifest, args.capture_root, args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
