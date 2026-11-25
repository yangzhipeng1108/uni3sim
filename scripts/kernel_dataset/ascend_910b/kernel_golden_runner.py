#!/usr/bin/env python3
"""Run the fixed no-profiler Ascend 910B kernel-golden sample serially."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.ascend_910b_event_capture import (
    ASCEND_VECTOR_DIRECT_MODE,
    ASCEND_VECTOR_NPU_GRAPH_MODE,
    build_ascend_910b_event_plans,
    capture_ascend_910b_supported_case,
    validate_ascend_vector_capture_contract as _validate_vector_capture_contract,
)
from dataset_pipeline.kernel_dataset.config import (
    get_kernel_dataset_path,
    load_pipeline_config,
)
from evaluation.kernel.device_event import (
    DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE,
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
    families = manifest["hardware"]["ascend_910b"]["families"]
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
    return {args.family: case_ids}


def _persistent_vector_runner_arguments(plan):
    command = plan["command"]
    if len(command) < 4 or command[0] != sys.executable:
        raise ValueError(
            f"invalid persistent Vector runner command: {plan['case_id']}")
    arguments = command[2:]
    if len(arguments) % 2 != 0:
        raise ValueError(
            f"persistent Vector runner command has valueless option: "
            f"{plan['case_id']}")
    mapping = {}
    for index in range(0, len(arguments), 2):
        option = arguments[index]
        value = arguments[index + 1]
        if option in mapping:
            raise ValueError(
                f"persistent Vector runner repeats option {option}: "
                f"{plan['case_id']}")
        mapping[option] = value
    expected_options = {
        "--op", "--tok", "--hidden", "--num_heads", "--head_dim",
        "--intermediate", "--num_kv_heads", "--batch", "--dtype", "--op2",
        "--event-vector-mode",
    }
    if set(mapping) != expected_options:
        raise ValueError(
            f"persistent Vector runner options mismatch: {plan['case_id']}")
    return mapping


def _run_persistent_vector_capture(
        *, plans, output_dir, baseline_commit, source_snapshot_sha256,
        timer_sanity_sha256, warmup_iterations, sample_count,
        vector_input_ring_mib, vector_power_warmup_gemm_iterations,
        vector_operation_repetitions_per_sample):
    if not plans:
        raise ValueError("persistent Vector capture requires non-empty plans")
    if any(plan["family"] != "vector" for plan in plans):
        raise ValueError("persistent Vector capture only accepts Vector plans")
    if (any(plan["canonical_attributes"]["op"] in {
            "add_rmsnorm", "elementwise", "kvcache"} for plan in plans)
            and vector_input_ring_mib < 2048):
        raise ValueError(
            "persistent paired-input Vector capture requires at least a "
            "2048 MiB input ring")
    runner_paths = {str(plan["runner_path"]) for plan in plans}
    if len(runner_paths) != 1:
        raise ValueError("persistent Vector plans disagree on runner path")
    manifest = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "family": "vector",
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
        "warmup_iterations": warmup_iterations,
        "sample_count": sample_count,
        "event_input_ring_mib": vector_input_ring_mib,
        "event_power_warmup_gemm_iterations": (
            vector_power_warmup_gemm_iterations),
        "event_operation_repetitions_per_sample": (
            vector_operation_repetitions_per_sample),
        "cases": [{
            "case_id": plan["case_id"],
            "canonical_attributes_sha256": (
                plan["canonical_attributes_sha256"]),
            "runner_arguments": _persistent_vector_runner_arguments(plan),
        } for plan in plans],
    }
    manifest_path = output_dir / "persistent_vector_event_manifest.json"
    write_kernel_golden_json(manifest_path, manifest)
    runner_path = Path(next(iter(runner_paths)))
    completed = subprocess.run(
        [sys.executable, str(runner_path), "--event-batch-manifest",
         str(manifest_path), "--event-batch-output-dir", str(output_dir)],
        text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "persistent Ascend Vector Event runner failed: "
            f"stderr={completed.stderr!r}")
    summary_path = output_dir / "capture_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(
            "persistent Ascend Vector Event runner returned success without "
            "capture summary")
    return {
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _load_resumed_result(output, plan, baseline_commit,
                         source_snapshot_sha256, timer_sanity_sha256,
                         cache_eviction_mib, warmup_iterations, sample_count,
                         vector_input_ring_mib=0,
                         vector_power_warmup_gemm_iterations=0):
    with output.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    for field, expected in (
            ("case_id", plan["case_id"]),
            ("baseline_commit", baseline_commit),
            ("source_snapshot_sha256", source_snapshot_sha256),
            ("timer_sanity_sha256", timer_sanity_sha256),
            ("hardware_id", "ascend_910b"),
            ("family", plan["family"]),
            ("canonical_attributes_sha256",
             plan["canonical_attributes_sha256"]),
            ("runner_sha256", file_sha256(plan["runner_path"]))):
        if artifact[field] != expected:
            raise ValueError(
                f"resumed 910B artifact {field} mismatch: {output}")
    if artifact["status"] != "measured":
        raise ValueError(f"resumed 910B artifact is not measured: {output}")
    expected_eviction_bytes = cache_eviction_mib * 1024 * 1024
    measurement = artifact["measurement"]
    _validate_event_measurement(measurement, plan["case_id"])
    if (measurement["warmup_iterations"] != warmup_iterations
            or measurement["sample_count"] != sample_count):
        raise ValueError(f"resumed 910B sampling contract mismatch: {output}")
    if measurement["schema_version"] != 3:
        raise ValueError(
            f"910B resume requires queued Event schema 3: {output}")
    else:
        vector_graph_mode = (
            plan["family"] == "vector"
            and plan["event_measurement_mode"]
            == ASCEND_VECTOR_NPU_GRAPH_MODE)
        if vector_graph_mode:
            graph_contract = plan["graph_contract"]
            expected_counts = [
                graph_contract["lower_repetition_count"],
                graph_contract["higher_repetition_count"],
            ]
            event_graph = artifact["workload"]["event_graph"]
            if (measurement["mode"]
                    != DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE
                    or measurement["repetition_counts"] != expected_counts
                    or measurement["operation_cycle_length"]
                    != graph_contract["higher_repetition_count"]
                    or measurement["cache_eviction_requested_bytes"]
                    != expected_eviction_bytes
                    or measurement["cache_eviction_bytes_before_replay"]
                    != expected_eviction_bytes
                    or artifact["workload"][
                        "cache_eviction_bytes_before_replay"]
                    != expected_eviction_bytes
                    or event_graph["lower_repetition_count"]
                    != expected_counts[0]
                    or event_graph["higher_repetition_count"]
                    != expected_counts[1]
                    or event_graph["preallocated_operation_count"]
                    != expected_counts[1]):
                raise ValueError(
                    f"resumed 910B NPUGraph contract mismatch: {output}")
            if artifact["timer"]["cache_condition"] != (
                    "same_stream_2_gib_bf16_sweep_before_every_graph_replay"):
                raise ValueError(
                    f"resumed 910B NPUGraph cache condition mismatch: {output}")
            if artifact["workload"]["event_measurement_mode"] != (
                    ASCEND_VECTOR_NPU_GRAPH_MODE):
                raise ValueError(
                    f"resumed 910B workload measurement mode mismatch: {output}")
            return {
                "case_id": plan["case_id"],
                "status": "measured",
                "command": None,
                "stdout": "",
                "stderr": "",
                "output": str(output),
                "resumed": True,
            }
        if (plan["family"] == "vector"
                and plan["event_measurement_mode"]
                != ASCEND_VECTOR_DIRECT_MODE):
            raise ValueError(
                f"unknown resumed 910B measurement mode: {output}")
        if plan["family"] == "vector" and "mode" in measurement:
            raise ValueError(
                f"resumed 910B Vector artifact is not direct: {output}")
        measurement_cache_field = "cache_eviction_bytes_before_sample"
        workload_cache_field = "cache_eviction_bytes_before_sample"
        expected_condition = (
            "rotating_dense_input_backing_ring"
            if vector_input_ring_mib else (
                "queued_pre_sample_device_sweep"
                if expected_eviction_bytes else "ordinary_warm_state"))
        if (measurement["cache_eviction_requested_bytes"]
                != expected_eviction_bytes
                or measurement[measurement_cache_field]
                != expected_eviction_bytes):
            raise ValueError(
                f"resumed 910B artifact cache eviction mismatch: {output}")
        if artifact["workload"][workload_cache_field] != expected_eviction_bytes:
            raise ValueError(
                f"resumed 910B workload cache eviction mismatch: {output}")
        if (plan["family"] == "vector"
                and artifact["workload"]["event_measurement_mode"]
                != plan["event_measurement_mode"]):
            raise ValueError(
                f"resumed 910B workload measurement mode mismatch: {output}")
        if artifact["timer"]["cache_condition"] != expected_condition:
            raise ValueError(
                f"resumed 910B timer cache condition mismatch: {output}")
        if vector_input_ring_mib:
            if plan["family"] != "vector":
                raise ValueError("resumed Ascend input ring is not a Vector case")
            input_ring = artifact["workload"]["input_ring"]
            if input_ring["requested_bytes"] != vector_input_ring_mib * 1024 * 1024:
                raise ValueError(
                    f"resumed 910B input-ring size mismatch: {output}")
            if vector_power_warmup_gemm_iterations:
                power_warmup = artifact["workload"]["power_warmup"]
                if power_warmup["gemm_iterations"] != (
                        vector_power_warmup_gemm_iterations):
                    raise ValueError(
                        f"resumed 910B power warmup mismatch: {output}")
                if artifact["timer"]["power_state_condition"] not in {
                        "sustained_bf16_gemm_before_event_samples",
                        "sustained_bf16_gemm_before_each_event_interval"}:
                    raise ValueError(
                        f"resumed 910B power-state condition mismatch: {output}")
            elif "power_warmup" in artifact["workload"]:
                raise ValueError(
                    f"resumed 910B artifact unexpectedly has power warmup: {output}")
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
    parser.add_argument("--vector-input-ring-mib", type=int, default=0)
    parser.add_argument(
        "--vector-power-warmup-gemm-iterations", type=int, default=0)
    parser.add_argument(
        "--vector-operation-repetitions-per-sample", type=int, default=1)
    parser.add_argument(
        "--vector-event-mode",
        choices=(ASCEND_VECTOR_DIRECT_MODE, ASCEND_VECTOR_NPU_GRAPH_MODE),
        default=ASCEND_VECTOR_DIRECT_MODE)
    parser.add_argument("--persistent-vector-event", action="store_true")
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
        args.cache_eviction_mib, minimum_nonzero_mib=1536,
        hardware_id="ascend_910b")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    if args.sample_manifest is not None:
        if (args.baseline_commit is not None or args.family is not None
                or args.all_cases or args.case_id is not None):
            raise ValueError(
                "sample-manifest cannot be mixed with explicit selection")
        manifest = load_sample_manifest(args.sample_manifest)
        selected_cases = _selected_cases(manifest)
        baseline_commit = manifest["baseline_commit"]
    else:
        selected_cases = _explicit_cases(args)
        baseline_commit = require_lower_hex(
            args.baseline_commit, 40, "baseline_commit")
    _validate_vector_capture_contract(
        selected_cases, args.cache_eviction_mib, args.samples,
        args.sample_manifest is None and args.family == "vector"
        and args.all_cases, args.vector_input_ring_mib,
        args.vector_power_warmup_gemm_iterations,
        args.vector_operation_repetitions_per_sample,
        args.vector_event_mode)
    plans = build_ascend_910b_event_plans(
        config_path=args.config,
        selected_case_ids=selected_cases,
        vector_event_mode=args.vector_event_mode,
    )
    if args.vector_input_ring_mib:
        unsupported_ring_cases = [
            plan["case_id"] for plan in plans
            if (plan["family"] != "vector"
                or plan["canonical_attributes"]["op"]
                not in {"rmsnorm", "add_rmsnorm", "rope", "silu",
                        "elementwise", "kvcache"})]
        if unsupported_ring_cases:
            raise ValueError(
                "Ascend input ring selection contains unsupported cases: "
                f"{unsupported_ring_cases[:8]}")
    if (args.vector_operation_repetitions_per_sample != 1
            and not args.persistent_vector_event):
        raise ValueError(
            "batched Ascend Vector Event timing requires persistent capture")
    if args.persistent_vector_event:
        if args.vector_event_mode != ASCEND_VECTOR_DIRECT_MODE:
            raise ValueError(
                "persistent Vector Event capture supports only direct mode")
        if args.resume:
            raise ValueError("persistent Vector Event capture does not support resume")
        if set(selected_cases) != {"vector"}:
            raise ValueError(
                "persistent Vector Event capture requires only Vector cases")
        if args.vector_operation_repetitions_per_sample not in {
                32, 512, 2048, 8192}:
            raise ValueError(
                "persistent Vector Event capture requires 32, 512, 2048, "
                "or 8192 calls per sample")
        if args.shard_count != 1 or args.shard_index != 0:
            raise ValueError(
                "persistent Vector Event capture does not support sharding")
        result = _run_persistent_vector_capture(
            plans=plans, output_dir=args.output_dir,
            baseline_commit=baseline_commit,
            source_snapshot_sha256=source_snapshot_sha256,
            timer_sanity_sha256=timer_sanity_sha256,
            warmup_iterations=args.warmup, sample_count=args.samples,
            vector_input_ring_mib=args.vector_input_ring_mib,
            vector_power_warmup_gemm_iterations=(
                args.vector_power_warmup_gemm_iterations),
            vector_operation_repetitions_per_sample=(
                args.vector_operation_repetitions_per_sample))
        print(json.dumps({
            "output": result["summary"],
            "persistent_manifest": result["manifest"],
        }, sort_keys=True))
        return 0
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
                args.warmup, args.samples, args.vector_input_ring_mib,
                args.vector_power_warmup_gemm_iterations)
        else:
            result = capture_ascend_910b_supported_case(
                plan=plan,
                output=output,
                baseline_commit=baseline_commit,
                source_snapshot_sha256=source_snapshot_sha256,
                timer_sanity_sha256=timer_sanity_sha256,
                warmup_iterations=args.warmup,
                sample_count=args.samples,
                cache_eviction_mib=args.cache_eviction_mib,
                vector_input_ring_mib=args.vector_input_ring_mib,
                vector_power_warmup_gemm_iterations=(
                    args.vector_power_warmup_gemm_iterations),
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
        "hardware_id": "ascend_910b",
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
        "cache_eviction_mib": args.cache_eviction_mib,
        "vector_input_ring_mib": args.vector_input_ring_mib,
        "vector_power_warmup_gemm_iterations": (
            args.vector_power_warmup_gemm_iterations),
        "vector_event_mode": args.vector_event_mode,
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
