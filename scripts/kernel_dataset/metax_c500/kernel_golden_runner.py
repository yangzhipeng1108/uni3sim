#!/usr/bin/env python3
"""Run the fixed no-profiler C500 kernel-golden sample serially."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.metax_c500_event_capture import (
    build_metax_c500_event_plans,
    build_metax_c500_family_event_plans,
    capture_metax_c500_case,
)
from dataset_pipeline.kernel_dataset.config import (
    get_kernel_dataset_path,
    load_pipeline_config,
)
from evaluation.kernel.device_event import (
    file_sha256,
    require_lower_hex,
    validate_cache_eviction_mib,
    write_kernel_golden_json,
)
from evaluation.kernel.kernel_golden import (
    DATASET_FAMILIES,
    _validate_event_measurement,
    load_sample_manifest,
)
from src.kernel.dataset_storage import load_kernel_dataset_document


def _selected_cases(manifest):
    families = manifest["hardware"]["metax_c500"]["families"]
    selected = {}
    for family in DATASET_FAMILIES:
        case_ids = []
        for rows in families[family]["buckets"].values():
            case_ids.extend(row["case_id"] for row in rows)
        selected[family] = case_ids
    return selected


def _explicit_cases(args):
    if args.family is None or args.baseline_commit is None:
        raise ValueError(
            "explicit selection requires family and baseline-commit")
    if args.all_cases == (args.case_id is not None):
        raise ValueError(
            "explicit selection requires exactly one of all-cases or case-id")
    config = load_pipeline_config(str(args.config))
    dataset_path = Path(get_kernel_dataset_path(config, args.family))
    dataset = load_kernel_dataset_document(dataset_path)
    case_ids = list(dataset) if args.all_cases else list(args.case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case-id values must be unique")
    return case_ids


def _load_resumed_result(output, plan, baseline_commit,
                         source_snapshot_sha256, timer_sanity_sha256,
                         cache_eviction_mib, warmup_iterations, sample_count):
    with output.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    for field, expected in (
            ("case_id", plan["case_id"]),
            ("baseline_commit", baseline_commit),
            ("source_snapshot_sha256", source_snapshot_sha256),
            ("timer_sanity_sha256", timer_sanity_sha256),
            ("hardware_id", "metax_c500"),
            ("family", plan["family"]),
            ("canonical_attributes_sha256",
             plan["canonical_attributes_sha256"]),
            ("runner_sha256", file_sha256(plan["runner_path"]))):
        if artifact[field] != expected:
            raise ValueError(
                f"resumed C500 artifact {field} mismatch: {output}")
    if artifact["status"] != "measured":
        raise ValueError(f"resumed C500 artifact is not measured: {output}")
    expected_eviction_bytes = cache_eviction_mib * 1024 * 1024
    measurement = artifact["measurement"]
    _validate_event_measurement(measurement, plan["case_id"])
    if (measurement["warmup_iterations"] != warmup_iterations
            or measurement["sample_count"] != sample_count):
        raise ValueError(f"resumed C500 sampling contract mismatch: {output}")
    if measurement["schema_version"] != 3:
        raise ValueError(
            f"C500 resume requires queued Event schema 3: {output}")
    else:
        if (measurement["cache_eviction_requested_bytes"]
                != expected_eviction_bytes
                or measurement["cache_eviction_bytes_before_sample"]
                != expected_eviction_bytes):
            raise ValueError(
                f"resumed C500 artifact cache eviction mismatch: {output}")
        if artifact["workload"][
                "cache_eviction_bytes_before_sample"] != expected_eviction_bytes:
            raise ValueError(
                f"resumed C500 workload cache eviction mismatch: {output}")
        expected_condition = (
            "queued_pre_sample_device_sweep"
            if expected_eviction_bytes else "ordinary_warm_state")
        if artifact["timer"]["cache_condition"] != expected_condition:
            raise ValueError(
                f"resumed C500 timer cache condition mismatch: {output}")
    return {
        "case_id": plan["case_id"],
        "status": "measured",
        "command": None,
        "stdout": "",
        "stderr": "",
        "output": str(output),
        "resumed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument(
        "--family", choices=("gemm", "flash_attn", "fused_moe", "vector"))
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--timer-sanity-sha256", required=True)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--cache-eviction-mib", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.warmup <= 0 or args.samples <= 0:
        raise ValueError("warmup and samples must be positive")
    source_snapshot_sha256 = require_lower_hex(
        args.source_snapshot_sha256, 64, "source_snapshot_sha256")
    timer_sanity_sha256 = require_lower_hex(
        args.timer_sanity_sha256, 64, "timer_sanity_sha256")
    validate_cache_eviction_mib(
        args.cache_eviction_mib, minimum_nonzero_mib=64,
        hardware_id="metax_c500")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    if args.sample_manifest is not None:
        if (args.baseline_commit is not None or args.family is not None
                or args.all_cases or args.case_id is not None):
            raise ValueError(
                "sample-manifest cannot be mixed with explicit selection")
        manifest = load_sample_manifest(args.sample_manifest)
        baseline_commit = manifest["baseline_commit"]
        plans = build_metax_c500_event_plans(
            config_path=args.config,
            selected_case_ids=_selected_cases(manifest),
        )
    else:
        case_ids = _explicit_cases(args)
        baseline_commit = require_lower_hex(
            args.baseline_commit, 40, "baseline_commit")
        plans = build_metax_c500_family_event_plans(
            config_path=args.config,
            family=args.family,
            selected_case_ids=case_ids,
        )
    if any(plan["family"] == "fused_moe" for plan in plans):
        raise ValueError(
            "C500 direct kernel-golden runner does not support FusedMoE; "
            "use kernel_golden_graph_runner.py")
    unsharded_case_count = len(plans)
    plans = [
        plan for index, plan in enumerate(plans)
        if index % args.shard_count == args.shard_index
    ]
    results = []
    for plan in plans:
        output = args.output_dir / "cases" / f"{plan['case_id']}.json"
        if args.resume and output.is_file():
            result = _load_resumed_result(
                output, plan, baseline_commit, source_snapshot_sha256,
                timer_sanity_sha256, args.cache_eviction_mib,
                args.warmup, args.samples)
        else:
            result = capture_metax_c500_case(
                plan=plan,
                output=output,
                baseline_commit=baseline_commit,
                source_snapshot_sha256=source_snapshot_sha256,
                timer_sanity_sha256=timer_sanity_sha256,
                warmup_iterations=args.warmup,
                sample_count=args.samples,
                cache_eviction_mib=args.cache_eviction_mib,
            )
        results.append(result)
        print(json.dumps({
            "case_id": plan["case_id"],
            "completed": len(results),
            "expected": len(plans),
            "status": result["status"],
        }, sort_keys=True), flush=True)
    summary = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
        "cache_eviction_mib": args.cache_eviction_mib,
        "unsharded_case_count": unsharded_case_count,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "selected_case_count": len(plans),
        "measured_case_count": sum(
            row["status"] == "measured" for row in results),
        "unsupported_case_count": sum(
            row["status"] == "unsupported" for row in results),
        "results": results,
    }
    write_kernel_golden_json(args.output_dir / "capture_summary.json", summary)
    print(json.dumps({
        "measured": summary["measured_case_count"],
        "unsupported": summary["unsupported_case_count"],
        "output": str((args.output_dir / "capture_summary.json").resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
