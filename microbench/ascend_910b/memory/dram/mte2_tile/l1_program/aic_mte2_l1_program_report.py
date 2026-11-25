#!/usr/bin/env python3
"""Audit three independent source-shaped 910B MTE2 L1-program captures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


SCHEMA_VERSION = "midlayer.ascend_910b.aic_mte2_l1_program_capture.v1"
SOURCE_FILENAMES = (
    "CMakeLists.txt",
    "aic_mte2_l1_program_kernel.cpp",
    "aic_mte2_l1_program_main.cpp",
    "aic_mte2_l1_program_runner.py",
)
RAW_FIELDS = (
    "run_id", "block_dim", "short_programs", "long_programs",
    "a_descriptors", "b_descriptors", "a_descriptor_bytes",
    "b_descriptor_bytes", "l1_bytes", "program_bytes", "short_cycles",
    "short_wall_ns", "long_cycles", "long_wall_ns", "tick_ns",
    "ns_per_program", "bandwidth_gbs", "validation",
)
EXPECTED_INTEGERS = {
    "block_dim": 24,
    "short_programs": 8,
    "long_programs": 32,
    "a_descriptors": 112,
    "b_descriptors": 168,
    "a_descriptor_bytes": 131072,
    "b_descriptor_bytes": 122880,
    "l1_bytes": 507904,
    "program_bytes": 35323904,
}
RUN_COUNT = 3
MAX_RELATIVE_SPAN = 0.05


def _require_exact_keys(value, expected, context):
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


def _load_manifest(path):
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    _require_exact_keys(
        value,
        ("schema_version", "hardware_id", "device", "software", "snapshot",
         "build", "capture", "publication"),
        "manifest",
    )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported L1-program capture schema")
    if value["hardware_id"] != "ascend_910b":
        raise ValueError("capture hardware_id must be ascend_910b")
    _require_exact_keys(
        value["device"],
        ("host", "container", "physical_device_id", "logical_device_id",
         "product_name", "serial_number"),
        "device",
    )
    if (value["device"]["physical_device_id"] not in (6, 7)
            or value["device"]["logical_device_id"] != 0):
        raise ValueError("capture device mapping is outside the authorized cards")
    _require_exact_keys(
        value["software"], ("cann", "container_image", "container_image_id"),
        "software")
    if value["software"]["cann"] != "8.5.1":
        raise ValueError("capture must use CANN 8.5.1")
    _require_exact_keys(
        value["snapshot"],
        ("commit", "path", "source_snapshot_sha256", "source_sha256"),
        "snapshot")
    if len(value["snapshot"]["commit"]) != 40:
        raise ValueError("snapshot commit must be a full Git object ID")
    _require_exact_keys(value["snapshot"]["source_sha256"], SOURCE_FILENAMES,
                        "snapshot source hashes")
    _require_exact_keys(value["build"], ("binary_path", "binary_sha256"),
                        "build")
    _require_exact_keys(value["capture"], ("environment", "invocations"),
                        "capture")
    _require_exact_keys(value["capture"]["environment"],
                        ("ASCEND_RT_VISIBLE_DEVICES",), "capture environment")
    if value["capture"]["environment"]["ASCEND_RT_VISIBLE_DEVICES"] != str(
            value["device"]["physical_device_id"]):
        raise ValueError("capture environment does not select the physical device")
    _require_exact_keys(value["publication"],
                        ("production_duration_used", "exact_points_only"),
                        "publication")
    if (value["publication"]["production_duration_used"] is not False
            or value["publication"]["exact_points_only"] is not True):
        raise ValueError("L1-program publication policy is invalid")
    return value


def _validate_invocation(invocation, manifest, capture_root):
    _require_exact_keys(
        invocation,
        ("run_id", "argv", "raw_file", "raw_sha256", "stderr_file",
         "stderr_sha256"),
        "capture invocation")
    run_id = invocation["run_id"]
    expected_argv = [
        "python3", "aic_mte2_l1_program_runner.py", "--binary",
        "build/aic_mte2_l1_program", "--device-id",
        str(manifest["device"]["physical_device_id"]), "--run-id", run_id,
        "--short-programs", "8", "--long-programs", "32",
    ]
    if invocation["argv"] != expected_argv:
        raise ValueError(f"run {run_id} changed the capture command")
    raw_path = capture_root / invocation["raw_file"]
    stderr_path = capture_root / invocation["stderr_file"]
    if _sha256(raw_path) != invocation["raw_sha256"]:
        raise ValueError(f"raw capture hash mismatch: {raw_path}")
    if _sha256(stderr_path) != invocation["stderr_sha256"]:
        raise ValueError(f"stderr capture hash mismatch: {stderr_path}")
    required_tokens = (
        f"device_id={manifest['device']['physical_device_id']}",
        f"run_id={run_id}",
        f"binary_sha256={manifest['build']['binary_sha256']}",
        "source_snapshot_sha256="
        f"{manifest['snapshot']['source_snapshot_sha256']}",
    )
    stderr = stderr_path.read_text(encoding="utf-8")
    if any(token not in stderr for token in required_tokens):
        raise ValueError(f"run {run_id} stderr provenance is incomplete")
    return raw_path


def _load_run(path, run_id):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError(f"{path} has an invalid raw schema")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one raw row")
    row = rows[0]
    if row["run_id"] != run_id or row["validation"] != "pass":
        raise ValueError(f"{path} has invalid identity or validation")
    values = {name: int(row[name]) for name in EXPECTED_INTEGERS}
    if values != EXPECTED_INTEGERS:
        raise ValueError(f"{path} changed the source-shaped program geometry")
    cycle_values = {name: int(row[name]) for name in (
        "short_cycles", "short_wall_ns", "long_cycles", "long_wall_ns")}
    if (cycle_values["short_cycles"] <= 0 or cycle_values["short_wall_ns"] <= 0
            or cycle_values["long_cycles"] <= cycle_values["short_cycles"]
            or cycle_values["long_wall_ns"] <= cycle_values["short_wall_ns"]):
        raise ValueError(f"{path} has an invalid differential interval")
    tick_ns = float(row["tick_ns"])
    ns_per_program = float(row["ns_per_program"])
    bandwidth_gbs = float(row["bandwidth_gbs"])
    expected_ns = ((cycle_values["long_cycles"] - cycle_values["short_cycles"])
                   / (EXPECTED_INTEGERS["long_programs"]
                      - EXPECTED_INTEGERS["short_programs"])) * 20.0
    expected_bandwidth = EXPECTED_INTEGERS["program_bytes"] / expected_ns
    if any(not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
           for actual, expected in (
               (tick_ns, 20.0), (ns_per_program, expected_ns),
               (bandwidth_gbs, expected_bandwidth))):
        raise ValueError(f"{path} has inconsistent derived metrics")
    return ns_per_program


def _audit(manifest_path, capture_root, source_root):
    manifest = _load_manifest(manifest_path)
    for filename, expected_hash in manifest["snapshot"]["source_sha256"].items():
        source_path = source_root / filename
        if not source_path.is_file() or _sha256(source_path) != expected_hash:
            raise ValueError(f"source identity mismatch: {source_path}")
    if (_source_snapshot_sha256(source_root)
            != manifest["snapshot"]["source_snapshot_sha256"]):
        raise ValueError("source snapshot hash mismatch")
    binary_path = capture_root / "aic_mte2_l1_program"
    if _sha256(binary_path) != manifest["build"]["binary_sha256"]:
        raise ValueError("captured binary hash mismatch")
    invocations = manifest["capture"]["invocations"]
    if not isinstance(invocations, list) or len(invocations) != RUN_COUNT:
        raise ValueError("capture requires exactly three independent runs")
    identities = set()
    samples = []
    for invocation in invocations:
        run_id = invocation["run_id"]
        if run_id in identities:
            raise ValueError("capture repeats a run identity")
        identities.add(run_id)
        samples.append(_load_run(
            _validate_invocation(invocation, manifest, capture_root), run_id))
    median = statistics.median(samples)
    relative_span = (max(samples) - min(samples)) / median
    stable = relative_span <= MAX_RELATIVE_SPAN
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_identity_verified": True,
        "independent_run_count": len(identities),
        "descriptor_program": {
            "a_descriptor_count": EXPECTED_INTEGERS["a_descriptors"],
            "a_descriptor_geometry": [128, 512],
            "b_descriptor_count": EXPECTED_INTEGERS["b_descriptors"],
            "b_descriptor_geometry": [256, 240],
            "active_aic_groups": EXPECTED_INTEGERS["block_dim"],
            "l1_ping_pong_bytes": EXPECTED_INTEGERS["l1_bytes"],
        },
        "ns_per_program_samples": samples,
        "continuous_median_ns_per_program": median,
        "relative_span": relative_span,
        "stability_gate": stable,
        "mechanism_publishable": stable,
        "publication_scope": (
            "exact descriptor counts, geometry, active AIC count, and "
            "L1 double-buffer capacity only; no interpolation"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(
        _audit(args.manifest, args.capture_root, args.source_root), indent=2,
        sort_keys=True))


if __name__ == "__main__":
    main()
