#!/usr/bin/env python3
"""Audit the three-run grouped-MatMul mixed-residency MTE2 campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = "midlayer.ascend_910b.grouped_mte2_mixed_capture.v1"
ACTIVE_AICS = 24
COLD_FOOTPRINT_BYTES = 512 * 1024 * 1024
HOT_SOURCE_TILES = ACTIVE_AICS
MINIMUM_ITERATIONS = 1024
RUN_INDEX_COUNT = 3
MAX_RELATIVE_SPAN = 0.20
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_grouped_pair_kernel.cpp",
    "aic_mte2_grouped_pair_main.cpp",
    "aic_mte2_grouped_pair_runner.py",
)
RAW_FIELDS = (
    "run_id", "run_index", "order_position", "process_pid",
    "a_residency", "b_residency", "a_rows", "b_columns",
    "a_tile_bytes", "b_tile_bytes", "a_source_tiles", "b_source_tiles",
    "drain_interval", "block_dim", "iterations", "short_cycles", "short_wall_ns",
    "long_cycles", "long_wall_ns", "tick_ns", "cycles_per_pair",
    "ns_per_pair", "bandwidth_gbs", "validation",
)
CELLS = tuple(
    (a_residency, b_residency, rows, drain_interval)
    for a_residency in ("resident", "cold")
    for b_residency in ("resident", "cold")
    for rows in (32, 128)
    for drain_interval in (0, 4)
)


def _require_keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{context} must contain exactly {sorted(expected)}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_sha256(root):
    digest = hashlib.sha256()
    for filename in SOURCE_FILENAMES:
        payload = (root / filename).read_bytes()
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _ordered_cells(run_index):
    rotation = run_index * 5 % len(CELLS)
    cells = CELLS[rotation:] + CELLS[:rotation]
    return cells if run_index % 2 == 0 else tuple(reversed(cells))


def _source_tiles(rows, residency):
    if residency == "resident":
        return HOT_SOURCE_TILES
    if residency == "cold":
        return COLD_FOOTPRINT_BYTES // (rows * 64 * 2)
    raise ValueError(f"unsupported A residency: {residency}")


def _b_source_tiles(columns, residency):
    if residency == "resident":
        return HOT_SOURCE_TILES
    if residency == "cold":
        return COLD_FOOTPRINT_BYTES // (64 * columns * 2)
    raise ValueError(f"unsupported B residency: {residency}")


def _iterations(a_tiles, b_tiles):
    minimum = max(
        MINIMUM_ITERATIONS,
        math.ceil(a_tiles / ACTIVE_AICS),
        math.ceil(b_tiles / ACTIVE_AICS),
    )
    return math.ceil(minimum / 4) * 4


def _load_manifest(path):
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    _require_keys(
        manifest,
        ("schema_version", "hardware_id", "device", "software", "snapshot",
         "build", "capture", "publication"),
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported mixed-residency capture schema")
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
        ("commit", "path", "source_snapshot_sha256", "source_sha256"),
        "snapshot",
    )
    if len(manifest["snapshot"]["commit"]) != 40:
        raise ValueError("snapshot commit must be a full Git object ID")
    _require_keys(
        manifest["snapshot"]["source_sha256"], SOURCE_FILENAMES,
        "snapshot source hashes")
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
        manifest["publication"],
        ("production_duration_used", "exact_points_only"),
        "publication",
    )
    if (manifest["publication"]["production_duration_used"] is not False
            or manifest["publication"]["exact_points_only"] is not True):
        raise ValueError("mixed-residency publication policy is invalid")
    return manifest


def _validate_invocation(invocation, manifest, capture_root, b_columns):
    _require_keys(
        invocation,
        ("run_id", "run_index", "argv", "raw_file", "raw_sha256",
         "stderr_file", "stderr_sha256"),
        "capture invocation",
    )
    run_id = invocation["run_id"]
    run_index = invocation["run_index"]
    expected_argv = [
        "python3", "aic_mte2_grouped_pair_runner.py",
        "--binary", "build/aic_mte2_grouped_pair",
        "--device-id", str(manifest["device"]["physical_device_id"]),
        "--b-columns", str(b_columns),
        "--run-id", run_id, "--run-index", str(run_index),
    ]
    if invocation["argv"] != expected_argv:
        raise ValueError(f"run {run_id} changed the capture command")
    raw_path = capture_root / invocation["raw_file"]
    stderr_path = capture_root / invocation["stderr_file"]
    if _sha256(raw_path) != invocation["raw_sha256"]:
        raise ValueError(f"raw capture hash mismatch: {raw_path}")
    if _sha256(stderr_path) != invocation["stderr_sha256"]:
        raise ValueError(f"stderr capture hash mismatch: {stderr_path}")
    stderr = stderr_path.read_text(encoding="utf-8")
    required_tokens = (
        f"device_id={manifest['device']['physical_device_id']}",
        f"run_id={run_id}",
        f"run_index={run_index}",
        f"binary_sha256={manifest['build']['binary_sha256']}",
        "source_snapshot_sha256="
        f"{manifest['snapshot']['source_snapshot_sha256']}",
    )
    if any(token not in stderr for token in required_tokens):
        raise ValueError(f"run {run_id} stderr provenance is incomplete")
    return raw_path


def _load_run(path, run_id, run_index, b_columns):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        raw_rows = list(reader)
    expected_cells = _ordered_cells(run_index)
    if len(raw_rows) != len(expected_cells):
        raise ValueError(f"{path} must contain exactly {len(expected_cells)} cells")
    rows = []
    for position, (raw, cell) in enumerate(zip(raw_rows, expected_cells)):
        if (raw["run_id"] != run_id or int(raw["run_index"]) != run_index
                or int(raw["order_position"]) != position
                or int(raw["process_pid"]) <= 0
                or raw["validation"] != "pass"):
            raise ValueError(f"{path} row {position + 2} has invalid identity")
        a_residency, b_residency, rows_count, drain_interval = cell
        if (raw["a_residency"], raw["b_residency"]) != (
                a_residency, b_residency):
            raise ValueError(f"{path} row {position + 2} changed residency")
        integer_fields = (
            "a_rows", "b_columns", "a_tile_bytes", "b_tile_bytes",
            "a_source_tiles", "b_source_tiles", "drain_interval",
            "block_dim", "iterations", "short_cycles", "short_wall_ns",
            "long_cycles", "long_wall_ns",
        )
        values = {name: int(raw[name]) for name in integer_fields}
        a_tiles = _source_tiles(rows_count, a_residency)
        b_tiles = _b_source_tiles(b_columns, b_residency)
        expected = {
            "a_rows": rows_count,
            "b_columns": b_columns,
            "a_tile_bytes": rows_count * 64 * 2,
            "b_tile_bytes": 64 * b_columns * 2,
            "a_source_tiles": a_tiles,
            "b_source_tiles": b_tiles,
            "drain_interval": drain_interval,
            "block_dim": ACTIVE_AICS,
            "iterations": _iterations(a_tiles, b_tiles),
        }
        if any(values[name] != value for name, value in expected.items()):
            raise ValueError(f"{path} row {position + 2} changed its cell")
        if (values["short_cycles"] <= 0 or values["short_wall_ns"] <= 0
                or values["long_cycles"] <= values["short_cycles"]
                or values["long_wall_ns"] <= values["short_wall_ns"]):
            raise ValueError(f"{path} row {position + 2} has invalid intervals")
        tick_ns = float(raw["tick_ns"])
        cycles_per_pair = float(raw["cycles_per_pair"])
        ns_per_pair = float(raw["ns_per_pair"])
        bandwidth_gbs = float(raw["bandwidth_gbs"])
        expected_cycles = (
            (values["long_cycles"] - values["short_cycles"])
            / (values["iterations"] * 3))
        expected_ns = expected_cycles * 20.0
        expected_bandwidth = (
            expected["a_tile_bytes"] + expected["b_tile_bytes"]
        ) / expected_ns
        checks = (
            (tick_ns, 20.0),
            (cycles_per_pair, expected_cycles),
            (ns_per_pair, expected_ns),
            (bandwidth_gbs, expected_bandwidth),
        )
        if any(not math.isclose(actual, target, rel_tol=1e-9, abs_tol=1e-9)
               for actual, target in checks):
            raise ValueError(f"{path} row {position + 2} has inconsistent derivation")
        rows.append({
            "run_id": run_id,
            "a_residency": a_residency,
            "b_residency": b_residency,
            "a_rows": rows_count,
            "drain_interval": drain_interval,
            "ns_per_pair": ns_per_pair,
        })
    return rows


def _audit(manifest_path, capture_root, source_root, b_columns):
    if b_columns not in (240, 256):
        raise ValueError("b_columns must be one of 240, 256")
    manifest = _load_manifest(manifest_path)
    for filename, expected_hash in manifest["snapshot"]["source_sha256"].items():
        source_path = source_root / filename
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"source identity mismatch: {source_path}")
    if (_source_snapshot_sha256(source_root)
            != manifest["snapshot"]["source_snapshot_sha256"]):
        raise ValueError("source snapshot hash mismatch")
    binary_path = capture_root / "aic_mte2_grouped_pair"
    if (_sha256(binary_path) != manifest["build"]["binary_sha256"]):
        raise ValueError("captured binary hash mismatch")

    invocations = manifest["capture"]["invocations"]
    if not isinstance(invocations, list) or len(invocations) != RUN_INDEX_COUNT:
        raise ValueError("capture requires exactly three independent runs")
    rows = []
    identities = set()
    indexes = set()
    for invocation in invocations:
        run_id = invocation["run_id"]
        run_index = invocation["run_index"]
        if run_id in identities or run_index in indexes:
            raise ValueError("capture repeats a run identity")
        identities.add(run_id)
        indexes.add(run_index)
        raw_path = _validate_invocation(
            invocation, manifest, capture_root, b_columns)
        rows.extend(_load_run(raw_path, run_id, run_index, b_columns))
    if indexes != set(range(RUN_INDEX_COUNT)):
        raise ValueError("capture run indexes must be exactly 0, 1 and 2")

    cells = []
    all_stable = True
    all_positive = True
    for a_residency, b_residency in (
            ("resident", "resident"), ("resident", "cold"),
            ("cold", "resident"), ("cold", "cold")):
        for rows_count in (32, 128):
            samples = {
                drain: [
                    row["ns_per_pair"] for row in rows
                    if row["a_residency"] == a_residency
                    and row["b_residency"] == b_residency
                    and row["a_rows"] == rows_count
                    and row["drain_interval"] == drain
                ]
                for drain in (0, 4)
            }
            if any(len(values) != RUN_INDEX_COUNT for values in samples.values()):
                raise ValueError("mixed-residency cell lacks three run samples")
            continuous = statistics.median(samples[0])
            stepk4 = statistics.median(samples[4])
            spans = {
                drain: (max(values) - min(values)) / statistics.median(values)
                for drain, values in samples.items()
            }
            stable = max(spans.values()) <= MAX_RELATIVE_SPAN
            positive = stepk4 > continuous
            all_stable = all_stable and stable
            all_positive = all_positive and positive
            cells.append({
                "a_residency": a_residency,
                "b_residency": b_residency,
                "a_rows": rows_count,
                "samples_per_mode": RUN_INDEX_COUNT,
                "continuous_median_ns": continuous,
                "stepk4_median_ns": stepk4,
                "stepk4_increment_ns": stepk4 - continuous,
                "stepk4_increase_percent": (stepk4 / continuous - 1.0) * 100.0,
                "continuous_relative_span": spans[0],
                "stepk4_relative_span": spans[4],
                "stability_gate": stable,
                "positive_increment_gate": positive,
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_identity_verified": True,
        "independent_run_count": len(identities),
        "stability_gate": all_stable,
        "positive_increment_gate": all_positive,
        "cells": cells,
        "mechanism_publishable": all_stable and all_positive,
        "publication_scope": (
            "exact A rows, B columns, and A/B residency pair only; no interpolation"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--b-columns", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(
        _audit(args.manifest, args.capture_root, args.source_root,
               args.b_columns),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
