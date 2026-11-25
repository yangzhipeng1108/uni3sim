#!/usr/bin/env python3
"""Run the fixed no-profiler NVIDIA A800 kernel-golden sample serially."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.kernel_dataset.nvidia_a800_event_capture import (
    A800_VECTOR_EVENT_BATCH_PUBLIC_CALL_COUNTS,
    A800_NONVECTOR_EVENT_STEADY_STATE_PUBLIC_CALL_COUNTS,
    A800_VECTOR_DIFFERENTIAL_GRAPH_PROTOCOL_VERSION,
    A800_VECTOR_EVENT_RING_PROTOCOL_VERSION,
    A800_VECTOR_EVENT_TIMING_MODES,
    build_nvidia_a800_vector_differential_graph_contract,
    build_nvidia_a800_vector_differential_graph_provenance,
    build_nvidia_a800_vector_event_batch_contract,
    build_nvidia_a800_vector_event_batch_provenance,
    build_nvidia_a800_vector_graph_contract,
    build_nvidia_a800_vector_graph_provenance,
    build_nvidia_a800_vector_power_warmup_provenance,
    build_nvidia_a800_event_plans,
    capture_nvidia_a800_supported_case,
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
    families = manifest["hardware"]["nvidia_a800"]["families"]
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


def _load_resumed_result(output, plan, baseline_commit,
                         source_snapshot_sha256, timer_sanity_sha256,
                         cache_eviction_mib, warmup_iterations, sample_count,
                         vector_input_ring_mib=0,
                         vector_operation_repetitions_per_sample=1,
                         vector_event_timing_mode="direct_batched_event",
                         telemetry_physical_gpu_index=None):
    with output.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    for field, expected in (
            ("case_id", plan["case_id"]),
            ("baseline_commit", baseline_commit),
            ("source_snapshot_sha256", source_snapshot_sha256),
            ("timer_sanity_sha256", timer_sanity_sha256),
            ("hardware_id", "nvidia_a800"),
            ("family", plan["family"]),
            ("canonical_attributes_sha256",
             plan["canonical_attributes_sha256"]),
            ("runner_sha256", file_sha256(plan["runner_path"]))):
        if artifact[field] != expected:
            raise ValueError(
                f"resumed A800 artifact {field} mismatch: {output}")
    if artifact["status"] != "measured":
        raise ValueError(f"resumed A800 artifact is not measured: {output}")
    expected_eviction_bytes = cache_eviction_mib * 1024 * 1024
    measurement = artifact["measurement"]
    _validate_event_measurement(measurement, plan["case_id"])
    if (measurement["warmup_iterations"] != warmup_iterations
            or measurement["sample_count"] != sample_count):
        raise ValueError(f"resumed A800 sampling contract mismatch: {output}")
    if telemetry_physical_gpu_index is not None:
        if artifact["environment"]["telemetry_physical_gpu_index"] != (
                telemetry_physical_gpu_index):
            raise ValueError(
                f"resumed A800 telemetry GPU mismatch: {output}")
        for sample in measurement["samples"]:
            telemetry = sample["telemetry_before_event"]
            if telemetry["physical_gpu_index"] != telemetry_physical_gpu_index:
                raise ValueError(
                    f"resumed A800 telemetry sample GPU mismatch: {output}")
    is_vector_graph = (
        plan["family"] == "vector"
        and vector_event_timing_mode in {
            "cuda_graph_full_ring_replay", "cuda_graph_differential"})
    is_vector_differential_graph = (
        plan["family"] == "vector"
        and vector_event_timing_mode == "cuda_graph_differential")
    is_vector_batch = (
        plan["family"] == "vector"
        and vector_operation_repetitions_per_sample != 1
        and not is_vector_graph)
    if is_vector_graph:
        if is_vector_differential_graph:
            expected_contract = (
                build_nvidia_a800_vector_differential_graph_contract(
                    plan["canonical_attributes"],
                    minimum_distinct_input_mib=vector_input_ring_mib))
        else:
            expected_contract = build_nvidia_a800_vector_graph_contract(
                plan["canonical_attributes"],
                input_ring_mib=vector_input_ring_mib)
        if measurement["schema_version"] != 3:
            raise ValueError(
                f"A800 Vector CUDA-Graph resume requires Event schema 3: {output}")
        expected_measurement_mode = (
            "paired_differential_events_around_cuda_graph_replay"
            if is_vector_differential_graph
            else "external_events_around_cuda_graph_replay")
        if measurement["mode"] != expected_measurement_mode:
            raise ValueError(
                f"A800 Vector CUDA-Graph replay mode mismatch: {output}")
        if (measurement["operation_cycle_length"]
                != expected_contract["slot_count"]
                or measurement["repetition_counts"]
                != expected_contract["graph_repetition_counts"]):
            raise ValueError(
                f"A800 Vector CUDA-Graph replay plan mismatch: {output}")
        if is_vector_differential_graph:
            distinct_input = measurement["distinct_input_contract"]
            if (distinct_input["hardware_l2_bytes"]
                    != expected_contract["hardware_l2_bytes"]
                    or distinct_input["configured_minimum_bytes"]
                    != expected_contract["minimum_distinct_input_bytes"]
                    or distinct_input["per_call_bytes"]
                    != expected_contract["input_bytes_per_public_call"]
                    or artifact["workload"]["distinct_input_contract"]
                    != distinct_input):
                raise ValueError(
                    "resumed A800 Vector differential distinct-input "
                    f"contract mismatch: {output}")
        if artifact["workload"]["event_measurement_mode"] != (
                vector_event_timing_mode):
            raise ValueError(
                f"A800 Vector CUDA-Graph workload mode mismatch: {output}")
        if (artifact["workload"]["cache_eviction_bytes_before_replay"] != 0
                or measurement["cache_eviction_requested_bytes"] != 0
                or measurement["cache_eviction_bytes_before_replay"] != 0):
            raise ValueError(
                f"A800 Vector CUDA-Graph replay must not use a sweep: {output}")
        if artifact["workload"]["input_ring"] != expected_contract:
            raise ValueError(
                f"resumed A800 Vector CUDA-Graph input ring mismatch: {output}")
        expected_batch = (
            build_nvidia_a800_vector_differential_graph_provenance(
                expected_contract)
            if is_vector_differential_graph
            else build_nvidia_a800_vector_graph_provenance(expected_contract))
        if artifact["workload"]["event_operation_batch"] != expected_batch:
            raise ValueError(
                f"resumed A800 Vector CUDA-Graph provenance mismatch: {output}")
        if is_vector_differential_graph:
            if artifact["workload"]["power_warmup"] != (
                    build_nvidia_a800_vector_power_warmup_provenance()):
                raise ValueError(
                    "resumed A800 Vector differential power-warmup mismatch: "
                    f"{output}")
            if artifact["timer"]["power_state_condition"] != (
                    "sustained_bf16_gemm_before_each_graph_replay"):
                raise ValueError(
                    "resumed A800 Vector differential power-state mismatch: "
                    f"{output}")
            if measurement[
                    "pre_event_operation_executions_per_replay"] != 1:
                raise ValueError(
                    "resumed A800 Vector differential replay-maintenance "
                    f"mismatch: {output}")
        if artifact["timer"]["cache_condition"] != expected_contract["mode"]:
            raise ValueError(
                f"resumed A800 Vector CUDA-Graph cache condition mismatch: {output}")
        expected_timer_mode = (
            "paired_differential_events_around_cuda_graph_replay"
            if is_vector_differential_graph else (
                "external_cuda_events_around_full_ring_cuda_graph_replay_"
                "reported_per_public_call"))
        if artifact["timer"]["mode"] != expected_timer_mode:
            raise ValueError(
                f"resumed A800 Vector CUDA-Graph timer mode mismatch: {output}")
    elif is_vector_batch:
        expected_contract = build_nvidia_a800_vector_event_batch_contract(
            plan["canonical_attributes"],
            input_ring_mib=vector_input_ring_mib,
            public_call_count=vector_operation_repetitions_per_sample,
        )
        if measurement["schema_version"] != 4:
            raise ValueError(
                f"A800 Vector batch resume requires Event schema 4: {output}")
        if measurement["operation_repetitions_per_sample"] != (
                vector_operation_repetitions_per_sample):
            raise ValueError(
                f"resumed A800 Vector batch count mismatch: {output}")
        if (measurement["cache_eviction_requested_bytes"] != 0
                or measurement["cache_eviction_bytes_before_sample"] != 0
                or artifact["workload"][
                    "cache_eviction_bytes_before_sample"] != 0):
            raise ValueError(
                f"resumed A800 Vector batch must not use a sweep: {output}")
        if artifact["workload"]["input_ring"] != expected_contract:
            raise ValueError(
                f"resumed A800 Vector input ring mismatch: {output}")
        expected_batch = build_nvidia_a800_vector_event_batch_provenance(
            vector_operation_repetitions_per_sample)
        if artifact["workload"]["event_operation_batch"] != expected_batch:
            raise ValueError(
                f"resumed A800 Vector operation batch mismatch: {output}")
        if artifact["timer"]["cache_condition"] != expected_contract["mode"]:
            raise ValueError(
                f"resumed A800 Vector cache condition mismatch: {output}")
        if artifact["timer"]["mode"] != (
                "fixed_count_public_ops_over_full_ring_cycle_per_event_interval_"
                "reported_per_call"):
            raise ValueError(
                f"resumed A800 Vector timer mode mismatch: {output}")
    elif measurement["schema_version"] != 3:
        raise ValueError(
            f"A800 resume requires queued Event schema 3: {output}")
    else:
        if (measurement["cache_eviction_requested_bytes"]
                != expected_eviction_bytes
                or measurement["cache_eviction_bytes_before_sample"]
                != expected_eviction_bytes):
            raise ValueError(
                f"resumed A800 artifact cache eviction mismatch: {output}")
        if artifact["workload"][
                "cache_eviction_bytes_before_sample"] != expected_eviction_bytes:
            raise ValueError(
                f"resumed A800 workload cache eviction mismatch: {output}")
        expected_condition = (
            "queued_pre_sample_device_sweep"
            if expected_eviction_bytes else "ordinary_warm_state")
        if artifact["timer"]["cache_condition"] != expected_condition:
            raise ValueError(
                f"resumed A800 timer cache condition mismatch: {output}")
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
    parser.add_argument("--telemetry-physical-gpu", type=int)
    parser.add_argument(
        "--nonvector-event-public-op-repetitions", type=int, default=1)
    parser.add_argument("--vector-input-ring-mib", type=int, default=0)
    parser.add_argument(
        "--vector-operation-repetitions-per-sample", type=int, default=1)
    parser.add_argument(
        "--vector-event-timing-mode",
        choices=sorted(A800_VECTOR_EVENT_TIMING_MODES),
        default="direct_batched_event")
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
        args.cache_eviction_mib, minimum_nonzero_mib=320,
        hardware_id="nvidia_a800")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if (args.telemetry_physical_gpu is not None
            and args.telemetry_physical_gpu < 0):
        raise ValueError("telemetry-physical-gpu must be nonnegative")
    if args.nonvector_event_public_op_repetitions not in (
            A800_NONVECTOR_EVENT_STEADY_STATE_PUBLIC_CALL_COUNTS):
        raise ValueError(
            "nonvector-event-public-op-repetitions must be one of 1, 8, 32")

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
    plans = build_nvidia_a800_event_plans(
        config_path=args.config,
        selected_case_ids=selected_cases,
    )
    if (args.nonvector_event_public_op_repetitions != 1
            and any(plan["family"] == "vector" for plan in plans)):
        raise ValueError(
            "non-Vector Event repetitions cannot be used for Vector cases")
    uses_vector_ring = args.vector_input_ring_mib != 0
    uses_vector_batch = args.vector_operation_repetitions_per_sample != 1
    uses_vector_graph = args.vector_event_timing_mode in {
        "cuda_graph_full_ring_replay", "cuda_graph_differential"}
    uses_vector_differential_graph = (
        args.vector_event_timing_mode == "cuda_graph_differential")
    if uses_vector_graph:
        if not uses_vector_ring or uses_vector_batch:
            raise ValueError(
                "A800 Vector CUDA-Graph replay requires an input ring and "
                "exactly one direct public call")
        if not any(plan["family"] == "vector" for plan in plans):
            raise ValueError(
                "A800 Vector CUDA-Graph options require selected Vector cases")
        if args.cache_eviction_mib != 0:
            raise ValueError(
                "A800 Vector CUDA-Graph replay requires cache eviction 0")
    elif uses_vector_ring != uses_vector_batch:
        raise ValueError(
            "A800 Vector capture requires ring and batching together")
    if uses_vector_batch:
        if args.vector_operation_repetitions_per_sample not in (
                A800_VECTOR_EVENT_BATCH_PUBLIC_CALL_COUNTS):
            raise ValueError(
                "A800 Vector Event batches require an explicitly selected "
                "public call count in {32, 128, 512, 2048}")
        if not any(plan["family"] == "vector" for plan in plans):
            raise ValueError(
                "A800 Vector batch options require selected Vector cases")
        if args.cache_eviction_mib != 0:
            raise ValueError(
                "A800 Vector batched ring capture requires cache eviction 0")
    unsharded_case_count = len(plans)
    plans = [
        plan for index, plan in enumerate(plans)
        if index % args.shard_count == args.shard_index
    ]
    results = []
    for plan in plans:
        output = args.output_dir / "cases" / f"{plan['case_id']}.json"
        if args.resume and output.is_file():
            if args.nonvector_event_public_op_repetitions != 1:
                raise ValueError(
                    "batched non-Vector Event artifacts must not be resumed; "
                    "use a new output directory")
            result = _load_resumed_result(
                output, plan, baseline_commit, source_snapshot_sha256,
                timer_sanity_sha256, args.cache_eviction_mib,
                args.warmup, args.samples, args.vector_input_ring_mib,
                args.vector_operation_repetitions_per_sample,
                args.vector_event_timing_mode,
                args.telemetry_physical_gpu)
        else:
            result = capture_nvidia_a800_supported_case(
                plan=plan,
                output=output,
                baseline_commit=baseline_commit,
                source_snapshot_sha256=source_snapshot_sha256,
                timer_sanity_sha256=timer_sanity_sha256,
                warmup_iterations=args.warmup,
                sample_count=args.samples,
                cache_eviction_mib=args.cache_eviction_mib,
                vector_input_ring_mib=args.vector_input_ring_mib,
                vector_operation_repetitions_per_sample=(
                    args.vector_operation_repetitions_per_sample),
                vector_event_timing_mode=args.vector_event_timing_mode,
                telemetry_physical_gpu_index=args.telemetry_physical_gpu,
                nonvector_operation_repetitions_per_sample=(
                    args.nonvector_event_public_op_repetitions),
            )
        results.append(result)
        print(json.dumps({
            "case_id": plan["case_id"],
            "completed": len(results),
            "expected": len(plans),
            "status": result["status"],
        }, sort_keys=True), flush=True)
    if uses_vector_batch:
        vector_event_batch_protocol_version = (
            A800_VECTOR_EVENT_RING_PROTOCOL_VERSION)
    elif uses_vector_differential_graph:
        vector_event_batch_protocol_version = (
            A800_VECTOR_DIFFERENTIAL_GRAPH_PROTOCOL_VERSION)
    elif uses_vector_graph:
        vector_event_batch_protocol_version = 3
    else:
        vector_event_batch_protocol_version = None
    summary = {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
        "cache_eviction_mib": args.cache_eviction_mib,
        "telemetry_physical_gpu_index": args.telemetry_physical_gpu,
        "nonvector_event_public_op_repetitions": (
            args.nonvector_event_public_op_repetitions),
        "vector_input_ring_mib": args.vector_input_ring_mib,
        "vector_operation_repetitions_per_sample": (
            args.vector_operation_repetitions_per_sample),
        "vector_event_timing_mode": args.vector_event_timing_mode,
        "vector_event_batch_protocol_version": (
            vector_event_batch_protocol_version),
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
