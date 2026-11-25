#!/usr/bin/env python3
"""Freeze C500 production launch coordinates without consuming timing data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
FAMILIES = ("flash_attn", "fused_moe", "gemm", "vector")
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from src.kernel.dataset_storage import load_kernel_dataset_document

DEFAULT_OUTPUT = Path(__file__).with_name(
    "production_launch_geometry_manifest.json"
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_dim3(value, label):
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a comma-separated string")
    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"{label} must contain exactly three dimensions")
    dimensions = []
    for part in parts:
        stripped = part.strip()
        if not stripped or not stripped.isdecimal():
            raise ValueError(f"{label} contains a non-integer dimension")
        dimension = int(stripped)
        if dimension <= 0:
            raise ValueError(f"{label} dimensions must be positive")
        dimensions.append(dimension)
    return tuple(dimensions)


def build_manifest(repo_root=REPO_ROOT):
    repo_root = Path(repo_root).resolve()
    source_files = []
    geometries = set()
    total_case_count = 0
    total_kernel_launch_count = 0
    seen_case_ids = set()
    for family in FAMILIES:
        relative_path = (
            Path("kernel_dataset") / HARDWARE_ID / f"{family}.json.zst")
        path = repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"canonical dataset is absent: {path}")
        payload = load_kernel_dataset_document(path)
        if not isinstance(payload, dict) or not payload:
            raise TypeError(f"canonical dataset must be a non-empty mapping: {path}")
        duplicate_cases = seen_case_ids.intersection(payload)
        if duplicate_cases:
            raise ValueError(
                f"canonical C500 case IDs overlap across families: "
                f"{sorted(duplicate_cases)}"
            )
        seen_case_ids.update(payload)
        family_kernel_launch_count = 0
        for case_id, entry in payload.items():
            if not isinstance(case_id, str) or not case_id:
                raise TypeError("canonical C500 case IDs must be non-empty strings")
            if not isinstance(entry, dict):
                raise TypeError(f"canonical case {case_id!r} must be a mapping")
            kernels = entry["kernels"]
            if not isinstance(kernels, list) or not kernels:
                raise TypeError(
                    f"canonical case {case_id!r} must contain kernel launches"
                )
            for kernel_index, kernel in enumerate(kernels):
                if not isinstance(kernel, dict):
                    raise TypeError(
                        f"canonical case {case_id!r} kernel {kernel_index} "
                        "must be a mapping"
                    )
                metadata = kernel["Metadata"]
                if not isinstance(metadata, dict):
                    raise TypeError(
                        f"canonical case {case_id!r} kernel {kernel_index} "
                        "Metadata must be a mapping"
                    )
                grid = _parse_dim3(
                    metadata["Grid Size"],
                    f"{case_id} kernel {kernel_index} Grid Size",
                )
                block = _parse_dim3(
                    metadata["Block Size"],
                    f"{case_id} kernel {kernel_index} Block Size",
                )
                geometries.add((grid, block))
                family_kernel_launch_count += 1
        source_files.append({
            "family": family,
            "path": relative_path.as_posix(),
            "sha256": _sha256(path),
            "case_count": len(payload),
            "kernel_launch_count": family_kernel_launch_count,
        })
        total_case_count += len(payload)
        total_kernel_launch_count += family_kernel_launch_count

    entries = [
        {"grid": list(grid), "block": list(block)}
        for grid, block in sorted(geometries)
    ]
    if total_case_count != len(seen_case_ids):
        raise AssertionError("canonical C500 case count is inconsistent")
    if len(entries) != len(geometries):
        raise AssertionError("production geometry manifest contains duplicates")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "source_contract": (
            "canonical kernel_dataset Metadata Grid Size and Block Size only"
        ),
        "duration_fields_consumed": False,
        "source_files": source_files,
        "total_case_count": total_case_count,
        "total_kernel_launch_count": total_kernel_launch_count,
        "geometry_count": len(entries),
        "geometry_policy": "exact_grid_and_block_only",
        "geometries": entries,
    }


def _write_json(path, payload):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.repo_root)
    _write_json(args.output, manifest)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "geometry_count": manifest["geometry_count"],
        "total_case_count": manifest["total_case_count"],
        "total_kernel_launch_count": manifest["total_kernel_launch_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
