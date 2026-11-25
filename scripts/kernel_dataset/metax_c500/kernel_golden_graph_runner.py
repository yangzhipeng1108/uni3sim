#!/usr/bin/env python3
"""Measure canonical C500 cases with graph-amortized device Events."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dataset_pipeline.kernel_dataset.config import (
    get_kernel_dataset_path,
    load_pipeline_config,
)
from dataset_pipeline.kernel_dataset.metax_c500_event_capture import (
    MINIMUM_DISTINCT_INPUT_MIB,
    build_vector_distinct_input_contract,
    build_metax_c500_family_event_plans,
    capture_metax_c500_case,
)
import dataset_pipeline.kernel_dataset.metax_c500 as metax_c500
from evaluation.kernel.device_event import (
    KERNEL_GOLDEN_OUTPUT_ROOT,
    file_sha256,
    require_lower_hex,
    validate_cache_eviction_mib,
    write_kernel_golden_json,
)
from evaluation.kernel.kernel_golden import _validate_event_measurement
from src.kernel.dataset_storage import load_kernel_dataset_document


TABLE_FIELDS = (
    "family", "case_id", "attributes_json", "executed_workload_json",
    "op", "op2", "tok",
    "hidden", "intermediate",
    "num_heads", "num_kv_heads", "head_dim", "batch",
    "B", "M", "K", "N", "S", "Q_LEN", "HQ", "HKV", "D",
    "SP_REQUESTED", "kv_layout", "causal",
    "TOK", "E", "TOPK", "H", "INT",
    "canonical_duration_us", "lower_repetitions", "higher_repetitions",
    "lower_median_us", "lower_p10_us", "lower_p90_us", "lower_cv",
    "higher_median_us", "higher_p10_us", "higher_p90_us", "higher_cv",
    "selected_duration_us", "absolute_error_us", "relative_error",
    "signed_relative_error", "convergence_status", "status", "error",
)

def _repetition_counts(canonical_duration_ns):
    if canonical_duration_ns <= 20_000:
        return 100, 500
    if canonical_duration_ns <= 100_000:
        return 50, 200
    if canonical_duration_ns <= 500_000:
        return 10, 50
    if canonical_duration_ns <= 2_000_000:
        return 5, 20
    if canonical_duration_ns <= 10_000_000:
        return 2, 10
    return 1, 3


def _resolve_buffer_count(*, family, fixed_buffer_count,
                          unique_buffers_per_replay, repetitions):
    if unique_buffers_per_replay:
        if family != "vector":
            raise ValueError(
                "unique_buffers_per_replay is supported only for vector")
        if fixed_buffer_count != 1:
            raise ValueError(
                "unique_buffers_per_replay cannot be mixed with a fixed "
                "buffer_count")
        return repetitions[1]
    return fixed_buffer_count


def _base_row(plan, repetitions):
    attributes = plan["canonical_attributes"]
    optional_fields = (
        "op", "op2", "tok", "hidden", "intermediate", "num_heads",
        "num_kv_heads", "head_dim", "batch", "B", "M", "K", "N", "S",
        "Q_LEN", "HQ", "HKV", "D", "SP_REQUESTED", "kv_layout", "causal",
        "TOK", "E", "TOPK", "H", "INT",
    )
    attribute_columns = dict.fromkeys(optional_fields)
    for field, value in attributes.items():
        if field in attribute_columns:
            attribute_columns[field] = value
    return {
        "family": plan["family"],
        "case_id": plan["case_id"],
        "attributes_json": json.dumps(
            attributes, separators=(",", ":"), sort_keys=True),
        "executed_workload_json": None,
        **attribute_columns,
        "canonical_duration_us": plan["canonical_duration_ns"] / 1_000.0,
        "lower_repetitions": repetitions[0],
        "higher_repetitions": repetitions[1],
    }


def _measured_row(plan, repetitions, artifact):
    row = _base_row(plan, repetitions)
    measurement = artifact["measurement"]
    runs = measurement["repetition_runs"]
    lower = runs[0]["statistics"]
    higher = runs[1]["statistics"]
    if measurement["mode"] == (
            "paired_differential_events_around_cuda_graph_replay"):
        selected_ns = measurement["statistics"]["median_ns"]
        convergence_status = "paired_differential"
    elif measurement["mode"] == (
            "external_events_around_cuda_graph_replay"):
        selected_ns = higher["median_ns"]
        convergence_status = measurement["convergence"]["status"]
    else:
        raise ValueError(
            f"unsupported graph measurement mode: {measurement['mode']!r}")
    canonical_ns = plan["canonical_duration_ns"]
    signed_error_ns = selected_ns - canonical_ns
    row.update({
        "executed_workload_json": json.dumps(
            artifact["workload"], separators=(",", ":"), sort_keys=True),
        "lower_median_us": lower["median_ns"] / 1_000.0,
        "lower_p10_us": lower["p10_ns"] / 1_000.0,
        "lower_p90_us": lower["p90_ns"] / 1_000.0,
        "lower_cv": lower["coefficient_of_variation"],
        "higher_median_us": higher["median_ns"] / 1_000.0,
        "higher_p10_us": higher["p10_ns"] / 1_000.0,
        "higher_p90_us": higher["p90_ns"] / 1_000.0,
        "higher_cv": higher["coefficient_of_variation"],
        "selected_duration_us": selected_ns / 1_000.0,
        "absolute_error_us": abs(signed_error_ns) / 1_000.0,
        "relative_error": abs(signed_error_ns) / canonical_ns,
        "signed_relative_error": signed_error_ns / canonical_ns,
        "convergence_status": convergence_status,
        "status": "measured",
        "error": None,
    })
    return row


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=path.parent,
                prefix=".kernel-golden-table-", delete=False) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_artifact(path, plan, metadata, repetitions):
    with path.open(encoding="utf-8") as stream:
        artifact = json.load(stream)
    if artifact["status"] != "measured":
        raise ValueError(f"case artifact is not measured: {path}")
    for field, expected in (
            ("case_id", plan["case_id"]),
            ("family", plan["family"]),
            ("baseline_commit", metadata["baseline_commit"]),
            ("source_snapshot_sha256", metadata["source_snapshot_sha256"]),
            ("timer_sanity_sha256", metadata["timer_sanity_sha256"]),
            ("canonical_attributes_sha256",
             plan["canonical_attributes_sha256"]),
            ("hardware_id", "metax_c500"),
            ("runner_sha256", file_sha256(plan["runner_path"]))):
        if artifact[field] != expected:
            raise ValueError(
                f"case artifact {field} mismatch: "
                f"expected={expected!r}, actual={artifact[field]!r}")
    expected_mode = (
        "paired_differential_events_around_cuda_graph_replay"
        if metadata["distinct_input_contract"] is not None
        else "external_events_around_cuda_graph_replay")
    if artifact["measurement"]["mode"] != expected_mode:
        raise ValueError(f"case artifact has wrong measurement mode: {path}")
    measurement = artifact["measurement"]
    _validate_event_measurement(measurement, plan["case_id"])
    if (measurement["warmup_iterations"] != metadata["warmup_iterations"]
            or measurement["sample_count"]
            != metadata["sample_count_per_repetition"]):
        raise ValueError(f"graph artifact sampling contract mismatch: {path}")
    if measurement["repetition_counts"] != list(repetitions):
        raise ValueError(f"graph artifact repetition counts mismatch: {path}")
    if measurement["operation_cycle_length"] != metadata["buffer_count"]:
        raise ValueError(f"graph artifact buffer count mismatch: {path}")
    if metadata["distinct_input_contract"] is not None:
        expected_contract = dict(metadata["distinct_input_contract"])
        del expected_contract["repetition_counts"]
        if measurement["distinct_input_contract"] != expected_contract:
            raise ValueError(
                f"graph measurement distinct-input contract mismatch: {path}")
        if artifact["workload"]["distinct_input_contract"] != expected_contract:
            raise ValueError(
                f"graph workload distinct-input contract mismatch: {path}")
        if artifact["workload"]["per_call_distinct_input_bytes"] != (
                expected_contract["per_call_bytes"]):
            raise ValueError(
                f"graph workload per-call input bytes mismatch: {path}")
    expected_eviction_bytes = metadata["cache_eviction_mib"] * 1024 * 1024
    if measurement["schema_version"] != 3:
        raise ValueError(f"graph resume requires queued Event schema 3: {path}")
    else:
        if (measurement["cache_eviction_requested_bytes"]
                != expected_eviction_bytes
                or measurement["cache_eviction_bytes_before_replay"]
                != expected_eviction_bytes):
            raise ValueError(
                f"graph artifact cache eviction mismatch: {path}")
        if artifact["workload"][
                "cache_eviction_bytes_before_replay"] != expected_eviction_bytes:
            raise ValueError(
                f"graph workload cache eviction mismatch: {path}")
        if plan["family"] == "vector" and metadata["buffer_count"] > 1:
            if (expected_eviction_bytes
                    and repetitions[1] <= metadata["buffer_count"]):
                expected_condition = (
                    "queued_pre_replay_device_sweep_without_intra_replay_"
                    "buffer_reuse")
            elif expected_eviction_bytes:
                expected_condition = (
                    "queued_pre_replay_device_sweep_with_intra_replay_reuse")
            else:
                expected_condition = (
                    "rotating_buffer_ring_without_explicit_sweep")
        elif expected_eviction_bytes:
            expected_condition = (
                "queued_pre_replay_device_sweep_with_intra_replay_reuse")
        else:
            expected_condition = "ordinary_warm_state"
        if artifact["timer"]["cache_condition"] != expected_condition:
            raise ValueError(f"graph timer cache condition mismatch: {path}")
    return artifact


def _write_tables(output_dir, family, rows, metadata):
    payload = dict(metadata)
    payload["rows"] = rows
    write_kernel_golden_json(output_dir / f"{family}_table.json", payload)
    _write_csv(output_dir / f"{family}_table.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--timer-sanity-sha256", required=True)
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trace-batch-manifest", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--buffer-count", type=int, default=1)
    parser.add_argument("--unique-buffers-per-replay", action="store_true")
    parser.add_argument("--cache-eviction-mib", type=int)
    parser.add_argument(
        "--minimum-distinct-input-mib", type=int,
        default=MINIMUM_DISTINCT_INPUT_MIB)
    parser.add_argument("--graph-repetitions", nargs=2, type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--family", choices=(
            "gemm", "flash_attn", "fused_moe", "vector"),
        default="vector")
    args = parser.parse_args()
    if args.cache_eviction_mib is None:
        args.cache_eviction_mib = (
            256 if args.unique_buffers_per_replay else 2048)

    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(KERNEL_GOLDEN_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "output_dir must stay under experiment_logs/kernel_golden") from error
    baseline_commit = require_lower_hex(
        args.baseline_commit, 40, "baseline_commit")
    source_snapshot_sha256 = require_lower_hex(
        args.source_snapshot_sha256, 64, "source_snapshot_sha256")
    timer_sanity_sha256 = require_lower_hex(
        args.timer_sanity_sha256, 64, "timer_sanity_sha256")
    if args.buffer_count <= 0:
        raise ValueError("buffer_count must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    validate_cache_eviction_mib(
        args.cache_eviction_mib, minimum_nonzero_mib=64,
        hardware_id="metax_c500")
    if args.minimum_distinct_input_mib < MINIMUM_DISTINCT_INPUT_MIB:
        raise ValueError(
            "minimum-distinct-input-mib must be at least 64 (8x C500 L2)")
    if args.minimum_distinct_input_mib % 8 != 0:
        raise ValueError(
            "minimum-distinct-input-mib must be an integer multiple of the "
            "8 MiB C500 L2")
    if args.buffer_count != 1 and args.family != "vector":
        raise ValueError("buffer_count greater than one requires vector")
    if args.unique_buffers_per_replay and args.family != "vector":
        raise ValueError(
            "unique_buffers_per_replay is supported only for vector")
    if args.unique_buffers_per_replay and args.buffer_count != 1:
        raise ValueError(
            "unique_buffers_per_replay cannot be mixed with buffer_count")
    if args.unique_buffers_per_replay and args.cache_eviction_mib != 256:
        raise ValueError(
            "C500 unique-buffer Vector measurement requires exactly 256 MiB "
            "of queued cache sweep before each replay")
    if args.buffer_count != 1 and args.graph_repetitions is None:
        raise ValueError(
            "rotating buffers require explicit graph_repetitions so every "
            "graph traverses the complete buffer cycle")
    if args.graph_repetitions is not None:
        lower_repetitions, higher_repetitions = args.graph_repetitions
        if lower_repetitions <= 0 or higher_repetitions <= lower_repetitions:
            raise ValueError(
                "graph_repetitions must be positive and strictly increasing")
        if args.buffer_count != 1:
            if (args.cache_eviction_mib == 0
                    and lower_repetitions < args.buffer_count):
                raise ValueError(
                    "every rotating graph must traverse the complete buffer "
                    "cycle: lower graph repetition count must be at least "
                    "buffer_count")
            if (args.cache_eviction_mib != 0
                    and higher_repetitions > args.buffer_count):
                raise ValueError(
                    "sweep-assisted rotating graphs must not reuse an "
                    "operator buffer within one replay: buffer_count must be "
                    "at least the higher graph repetition count")

    config = load_pipeline_config(str(args.config))
    dataset_path = Path(get_kernel_dataset_path(config, args.family))
    dataset = load_kernel_dataset_document(dataset_path)
    if args.case_id is None:
        case_ids = list(dataset)
    else:
        case_ids = list(args.case_id)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case-id values must be unique")
    plans = build_metax_c500_family_event_plans(
        config_path=args.config,
        family=args.family,
        selected_case_ids=case_ids,
    )
    unsharded_case_count = len(plans)
    plans = [
        plan for index, plan in enumerate(plans)
        if index % args.shard_count == args.shard_index
    ]
    metadata = {
        "schema_version": 1,
        "hardware_id": "metax_c500",
        "family": args.family,
        "method": (
            "paired_differential_events_around_cuda_graph_replay"
            if args.unique_buffers_per_replay
            else "external_events_around_cuda_graph_replay"),
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
        "canonical_dataset_sha256": file_sha256(dataset_path),
        "expected_case_count": len(plans),
        "unsharded_case_count": unsharded_case_count,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "sample_count_per_repetition": args.samples,
        "warmup_iterations": args.warmup,
        "convergence_rule": (
            None if args.unique_buffers_per_replay
            else "abs(delta) <= max(500 ns, 5% of larger median)"),
        "differential_estimator": (
            "(T_high-T_low)/(N_high-N_low)"
            if args.unique_buffers_per_replay else None),
        "buffer_policy": (
            "unique_buffer_per_selected_replay_call"
            if args.unique_buffers_per_replay else "fixed"),
        "buffer_count": (
            None if args.unique_buffers_per_replay else args.buffer_count),
        "cache_eviction_mib": args.cache_eviction_mib,
        "minimum_distinct_input_mib": args.minimum_distinct_input_mib,
    }
    trace_batch_manifest, trace_batch_payload = metax_c500.prepare_graph_trace_batch_manifest(
        args.trace_batch_manifest, args.resume, args.dry_run,
        baseline_commit, source_snapshot_sha256, args.family, plans)
    rows = []
    for index, plan in enumerate(plans):
        if args.unique_buffers_per_replay:
            distinct_input_contract = build_vector_distinct_input_contract(
                plan["canonical_attributes"],
                args.minimum_distinct_input_mib)
            selected_repetitions = distinct_input_contract[
                "repetition_counts"]
            if (args.graph_repetitions is not None
                    and tuple(args.graph_repetitions) != selected_repetitions):
                raise ValueError(
                    "explicit graph repetitions disagree with the "
                    "distinct-input threshold selection rule")
            repetitions = selected_repetitions
        else:
            distinct_input_contract = None
            repetitions = (
                tuple(args.graph_repetitions)
                if args.graph_repetitions is not None
                else _repetition_counts(plan["canonical_duration_ns"])
            )
        buffer_count = _resolve_buffer_count(
            family=plan["family"],
            fixed_buffer_count=args.buffer_count,
            unique_buffers_per_replay=args.unique_buffers_per_replay,
            repetitions=repetitions,
        )
        case_metadata = dict(metadata)
        case_metadata["buffer_count"] = buffer_count
        case_metadata["distinct_input_contract"] = distinct_input_contract
        output = output_dir / "cases" / f"{plan['case_id']}.json"
        if args.dry_run:
            print(json.dumps({
                "case_id": plan["case_id"],
                "graph_repetitions": list(repetitions),
                "buffer_count": buffer_count,
            }, sort_keys=True), flush=True)
            continue
        if args.resume and output.is_file():
            artifact = _load_artifact(
                output, plan, case_metadata, repetitions)
        else:
            capture_plan = (
                metax_c500.wrap_graph_trace_batch_plan(
                    plan, Path(__file__).resolve())
                if trace_batch_manifest is not None else plan)
            capture_result = capture_metax_c500_case(
                plan=capture_plan,
                output=output,
                baseline_commit=baseline_commit,
                source_snapshot_sha256=source_snapshot_sha256,
                timer_sanity_sha256=timer_sanity_sha256,
                warmup_iterations=args.warmup,
                sample_count=args.samples,
                graph_repetitions=repetitions,
                buffer_count=buffer_count,
                cache_eviction_mib=args.cache_eviction_mib,
                minimum_distinct_input_mib=(
                    args.minimum_distinct_input_mib
                    if args.unique_buffers_per_replay else None),
            )
            artifact = _load_artifact(
                output, plan, case_metadata, repetitions)
            if trace_batch_payload is not None:
                metax_c500.record_graph_trace_batch_case(
                    trace_batch_manifest, trace_batch_payload, index, plan,
                    output, artifact, capture_result["stdout"])
        row = _measured_row(plan, repetitions, artifact)
        rows.append(row)
        _write_tables(output_dir, args.family, rows, metadata)
        print(json.dumps({
            "completed": index + 1,
            "expected": len(plans),
            "case_id": plan["case_id"],
            "status": row["status"],
            "convergence_status": row["convergence_status"],
        }, sort_keys=True), flush=True)

    if args.dry_run:
        print(json.dumps({"dry_run": True, "expected": len(plans)},
                         sort_keys=True))
        return 0
    if len(rows) != len(plans):
        raise AssertionError(f"{args.family} table is incomplete")
    metax_c500.complete_graph_trace_batch_manifest(trace_batch_manifest, trace_batch_payload, len(plans))
    failed = sum(row["status"] != "measured" for row in rows)
    print(json.dumps({
        "expected": len(plans),
        "measured": len(plans) - failed,
        "failed": failed,
        "output": str(
            (output_dir / f"{args.family}_table.csv").resolve()),
    }, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--trace-batch-child":
        metax_c500.execute_graph_trace_batch_child(sys.argv[2:])
    else:
        raise SystemExit(main())
