#!/usr/bin/env python3
"""Validate and materialize frozen cross-hardware FA serving-capture inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.fa_serving_capture import (
    build_capture_plan,
    load_capture_manifest,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"candidate capture input already exists: {path}")
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8")


def _materialize(plan, output_root):
    materialized = []
    for item in plan["plans"]:
        case_root = output_root / "capture_inputs" / item["case_id"]
        metadata_path = case_root / "capture_metadata.json"
        workload_path = case_root / "runner_workload.json"
        _write_json(metadata_path, item["capture_metadata"])
        _write_json(workload_path, item["runner_workload"])
        arguments = []
        for argument in item["runner_arguments"]:
            if argument == "{capture_metadata_json}":
                arguments.append(str(metadata_path.resolve()))
            elif argument == "{runner_workload_json}":
                arguments.append(str(workload_path.resolve()))
            else:
                arguments.append(argument)
        materialized.append({
            "case_id": item["case_id"],
            "workload_digest": item["workload_digest"],
            "capture_metadata": str(metadata_path.resolve()),
            "runner_workload": str(workload_path.resolve()),
            "runner": item["runner"],
            "runner_arguments": arguments,
        })
    return materialized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hardware", required=True, choices=(
        "nvidia_a800", "metax_c500", "ascend_910b"))
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.dry_run == (args.output_root is not None):
        raise ValueError("specify exactly one of --dry-run or --output-root")
    loaded_manifest = load_capture_manifest(args.manifest)
    plan = build_capture_plan(
        loaded_manifest, hardware_id=args.hardware, case_ids=args.case_id)
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        return 0
    output_root = args.output_root.resolve()
    materialized = _materialize(plan, output_root)
    result = {
        "schema_version": 1,
        "capture_kind": "fa_serving_coverage_candidate_materialized_inputs",
        "manifest_path": plan["manifest_path"],
        "manifest_sha256": plan["manifest_sha256"],
        "hardware_id": args.hardware,
        "case_count": plan["case_count"],
        "cases": materialized,
    }
    _write_json(output_root / "capture_input_manifest.json", result)
    print(json.dumps({
        "hardware_id": args.hardware,
        "case_count": plan["case_count"],
        "output": str((output_root / "capture_input_manifest.json").resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
