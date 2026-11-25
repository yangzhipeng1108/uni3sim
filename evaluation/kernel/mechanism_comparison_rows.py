"""Comparison row families for kernel mechanism reports: kernel,
counterfactual, baseline, bucket, before/after, impact, accuracy
gate, and summary rows."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from evaluation.kernel.mechanism_capture import (
    BUCKET_MAPE_LIMIT_PCT,
    BUCKET_SCHEMA_VERSION,
    CASE_APE_LIMIT_PCT,
    COMPARISON_STATUSES,
    DEMAND_ROLES,
    DIRECTIONAL_TRAFFIC_COMPONENTS,
    FAMILY_LABELS,
    HARDWARE,
    SUPPORT_LEVELS,
    TIME_ROLES,
    canonical_capture_comparison_evidence,
)

from evaluation.kernel.mechanism_report_primitives import (
    metric_summary,
    classify_comparison_status,
    observed_duration_ns,
    kernel_ordinal_for_scope,
    kernel_ordinal_sort_value,
    kernel_launch_class,
    task_execution_signature,
    physical_bucket,
    task_axis_summary,
    task_rows,
    error_direction,
    index_result_rows,
    require_same_result_identity,
    comparison_model_identity,
    positive_csv_float,
    component_ratio_summary,
)
from evaluation.kernel.mechanism_traffic_rows import (
    directional_kernel_traffic_fields,
    comparison_rows,
)


def counterfactual_rows(
        documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Aggregate one-counter-at-a-time kernel predictions at task scope."""

    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            forward_ns, observed_ns = task["forward_pair"]
            candidate_by_kernel = []
            components = set()
            for kernel in task["kernels"]:
                entries = kernel["mechanisms"]["single_counterfactuals"]
                index = {}
                for entry in entries:
                    component = entry["component"]
                    if component in index:
                        raise ValueError(
                            "kernel repeats a single-counterfactual component: "
                            f"{task['case_id']!r} {component!r}")
                    index[component] = entry
                    components.add(component)
                candidate_by_kernel.append(index)
            family = task["family"]
            execution_signature = task_execution_signature(task)
            bucket = physical_bucket(
                hardware_id=hardware_id,
                family=family,
                metadata=task["metadata"],
                result_scope="aggregate_task",
                execution_signature=execution_signature,
            )
            for component in sorted(components):
                counterfactual_ns = 0.0
                replacement_count = 0
                input_paths = set()
                providers = set()
                metrics = set()
                provenance_kinds = set()
                units = set()
                for kernel, index in zip(
                        task["kernels"], candidate_by_kernel, strict=True):
                    if component in index:
                        candidate = index[component]
                        if candidate["uses_target_duration"]:
                            raise ValueError(
                                "counterfactual candidate consumes target "
                                f"duration: {component!r}")
                        counterfactual_ns += candidate["predicted_ns"]
                        replacement_count += 1
                        input_paths.add(candidate["input_path"])
                        providers.add(candidate["provider"])
                        metrics.add(candidate["metric"])
                        provenance_kinds.add(candidate["provenance_kind"])
                        units.add(candidate["unit"])
                    else:
                        counterfactual_ns += kernel[
                            "mechanisms"]["forward"]["total_ns"]
                if replacement_count <= 0:
                    raise RuntimeError(
                        "counterfactual component has no replacements")
                if len(input_paths) != 1 or len(units) != 1:
                    raise ValueError(
                        "counterfactual component disagrees on input path or "
                        f"unit: {component!r}")
                forward_absolute_error_ns = abs(forward_ns - observed_ns)
                counterfactual_absolute_error_ns = abs(
                    counterfactual_ns - observed_ns)
                rows.append({
                    "hardware_id": hardware_id,
                    "family": family,
                    "case_id": task["case_id"],
                    "component": component,
                    "input_path": next(iter(input_paths)),
                    "unit": next(iter(units)),
                    "provider": json.dumps(sorted(providers)),
                    "metric": json.dumps(sorted(metrics)),
                    "provenance_kind": json.dumps(
                        sorted(provenance_kinds)),
                    "kernel_count": len(task["kernels"]),
                    "replaced_kernel_count": replacement_count,
                    "full_kernel_coverage": (
                        replacement_count == len(task["kernels"])),
                    "forward_predicted_ns": forward_ns,
                    "counterfactual_predicted_ns": counterfactual_ns,
                    "observed_ns": observed_ns,
                    "forward_absolute_error_ns": forward_absolute_error_ns,
                    "counterfactual_absolute_error_ns": (
                        counterfactual_absolute_error_ns),
                    "oracle_gain_ns": (
                        forward_absolute_error_ns
                        - counterfactual_absolute_error_ns),
                    "forward_error_direction": error_direction(
                        forward_ns, observed_ns),
                    "counterfactual_error_direction": error_direction(
                        counterfactual_ns, observed_ns),
                    **comparison_evidence,
                    **bucket,
                })
    return rows


def counterfactual_summary_rows(
        counterfactual_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rank buckets by absolute error and counters by cumulative oracle gain."""

    bucket_rows = {}
    for row in task_rows:
        key = (row["hardware_id"], row["physical_bucket_id"])
        if key not in bucket_rows:
            bucket_rows[key] = []
        bucket_rows[key].append(row)
    target_bucket_keys = [
        key for key in bucket_rows
        if key[0] in ("nvidia_a800", "ascend_910b")
    ]
    target_bucket_keys.sort(
        key=lambda key: (
            -sum(float(row["absolute_duration_error_ns"])
                 for row in bucket_rows[key]),
            key,
        ))
    target_bucket_rank = {
        key: rank
        for rank, key in enumerate(target_bucket_keys, start=1)
    }

    grouped = {}
    for row in counterfactual_rows:
        key = (
            row["hardware_id"], row["family"],
            row["physical_bucket_id"], row["component"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    summaries = []
    for key, rows in grouped.items():
        hardware_id, family, bucket_id, component = key
        bucket_key = (hardware_id, bucket_id)
        all_bucket_rows = bucket_rows[bucket_key]
        bucket_observed_ns = math.fsum(
            float(row["observed_ns"]) for row in all_bucket_rows)
        bucket_absolute_error_ns = math.fsum(
            float(row["absolute_duration_error_ns"])
            for row in all_bucket_rows)
        candidate_observed_ns = math.fsum(
            float(row["observed_ns"]) for row in rows)
        forward_absolute_error_ns = math.fsum(
            float(row["forward_absolute_error_ns"]) for row in rows)
        counterfactual_absolute_error_ns = math.fsum(
            float(row["counterfactual_absolute_error_ns"]) for row in rows)
        if bucket_key in target_bucket_rank:
            bucket_rank = target_bucket_rank[bucket_key]
        else:
            bucket_rank = None
        summaries.append({
            "target_bucket_rank": bucket_rank,
            "counter_rank_within_bucket": None,
            "hardware_id": hardware_id,
            "family": family,
            "physical_bucket_id": bucket_id,
            "physical_bucket_short_id": rows[0][
                "physical_bucket_short_id"],
            "bucket_case_count": len(all_bucket_rows),
            "bucket_total_absolute_error_ns": bucket_absolute_error_ns,
            "bucket_wmae_pct": (
                bucket_absolute_error_ns / bucket_observed_ns * 100.0),
            "component": component,
            "input_path": rows[0]["input_path"],
            "unit": rows[0]["unit"],
            "provider": json.dumps(sorted({row["provider"] for row in rows})),
            "metric": json.dumps(sorted({row["metric"] for row in rows})),
            "covered_case_count": len(rows),
            "full_kernel_coverage_case_count": sum(
                bool(row["full_kernel_coverage"]) for row in rows),
            "replaced_kernel_count": sum(
                int(row["replaced_kernel_count"]) for row in rows),
            "forward_absolute_error_ns": forward_absolute_error_ns,
            "counterfactual_absolute_error_ns": (
                counterfactual_absolute_error_ns),
            "cumulative_oracle_gain_ns": (
                forward_absolute_error_ns
                - counterfactual_absolute_error_ns),
            "forward_wmae_pct": (
                forward_absolute_error_ns / candidate_observed_ns * 100.0),
            "counterfactual_wmae_pct": (
                counterfactual_absolute_error_ns
                / candidate_observed_ns * 100.0),
            "underprediction_case_count": sum(
                row["forward_error_direction"] == "under" for row in rows),
            "overprediction_case_count": sum(
                row["forward_error_direction"] == "over" for row in rows),
            "comparison_statuses": json.dumps(sorted({
                row["comparison_status"] for row in rows
            })),
        })
    by_bucket = {}
    for row in summaries:
        key = (row["hardware_id"], row["physical_bucket_id"])
        if key not in by_bucket:
            by_bucket[key] = []
        by_bucket[key].append(row)
    for rows in by_bucket.values():
        rows.sort(key=lambda row: (
            -float(row["cumulative_oracle_gain_ns"]), row["component"]))
        for rank, row in enumerate(rows, start=1):
            row["counter_rank_within_bucket"] = rank
    summaries.sort(key=lambda row: (
        row["target_bucket_rank"] is None,
        float("inf") if row["target_bucket_rank"] is None
        else int(row["target_bucket_rank"]),
        int(row["counter_rank_within_bucket"]),
        row["hardware_id"],
        row["physical_bucket_id"],
        row["component"],
    ))
    return summaries


def kernel_rows(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        comparison_evidence = canonical_capture_comparison_evidence(
            hardware_id)
        for task in document["tasks"]:
            for kernel in task["kernels"]:
                mechanism = kernel["mechanisms"]
                timing_target = mechanism["timing_target"]
                if timing_target["availability"] == "unobserved":
                    if (timing_target["duration_ns"] is not None
                            or timing_target["source"] is not None
                            or not timing_target["unavailable_reason"]):
                        raise ValueError(
                            "unobserved kernel timing target is malformed")
                    continue
                if timing_target["availability"] != "observed":
                    raise ValueError(
                        "unknown kernel timing availability: "
                        f"{timing_target['availability']!r}")
                analytical = mechanism["analytical_input"]
                observation = mechanism["observation"]
                forward = mechanism["forward"]
                launch = mechanism["observed_launch"]
                execution_signature = json.dumps(
                    kernel_launch_class(mechanism),
                    separators=(",", ":"), sort_keys=True)
                bucket = physical_bucket(
                    hardware_id=hardware_id,
                    family=task["family"],
                    metadata=task["metadata"],
                    result_scope="real_kernel",
                    execution_signature=execution_signature,
                )
                rows.append({
                    "result_scope": "real_kernel",
                    "hardware_id": hardware_id,
                    "family": task["family"],
                    "op_type": kernel["op_type"],
                    "case_id": task["case_id"],
                    "kernel_ordinal": mechanism["kernel_ordinal"],
                    "kernel_kind": mechanism["kernel_kind"],
                    "variant": mechanism["variant"],
                    "kernel_name": mechanism["kernel_name"],
                    "actual_ns": kernel["actual_ns"],
                    "timing_source": timing_target["source"],
                    "predicted_ns": forward["total_ns"],
                    "signed_error_pct": (
                        (forward["total_ns"] - kernel["actual_ns"])
                        / kernel["actual_ns"] * 100.0),
                    "absolute_error_pct": (
                        abs(forward["total_ns"] - kernel["actual_ns"])
                        / kernel["actual_ns"] * 100.0),
                    "absolute_duration_error_ns": abs(
                        forward["total_ns"] - kernel["actual_ns"]),
                    **comparison_evidence,
                    "support_level": forward["support_level"],
                    "limitations": json.dumps(
                        forward["limitations"], ensure_ascii=False),
                    "bottleneck": forward["bottleneck_group"],
                    "shape": json.dumps(
                        mechanism["shape"], separators=(",", ":"),
                        sort_keys=True),
                    "task_execution_signature": task_execution_signature(task),
                    "grid_size": analytical["grid_size"],
                    "block_size": analytical["block_size"],
                    "wave_efficiency": analytical["wave_efficiency"],
                    "active_units": analytical["active_units"],
                    "predicted_l2_bytes": analytical[
                        "demand_per_resource"]["l2_cache"],
                    "predicted_dram_bytes": analytical[
                        "demand_per_resource"]["dram"],
                    "observed_l2_source_interface_bytes": observation[
                        "l2_source_interface_bytes"],
                    "observed_downstream_global_bytes": observation[
                        "downstream_global_bytes"],
                    "observed_l2_service_bytes": observation[
                        "canonical_l2_service_bytes"],
                    "observed_l2_service_status": observation[
                        "canonical_l2_service_status"],
                    **directional_kernel_traffic_fields(mechanism),
                    "l1_hit_rate": observation["l1_hit_rate"],
                    "l2_hit_rate": observation["l2_hit_rate"],
                    "achieved_occupancy_pct": observation[
                        "achieved_occupancy_pct"],
                    "theoretical_occupancy_pct": observation[
                        "theoretical_occupancy_pct"],
                    "pipe_tensor_pct": observation["pipe_tensor_pct"],
                    "pipe_fma_pct": observation["pipe_fma_pct"],
                    "pipe_alu_pct": observation["pipe_alu_pct"],
                    "pipe_sfu_pct": observation["pipe_sfu_pct"],
                    "pipe_mte_pct": observation["pipe_mte_pct"],
                    "pipe_ste_pct": observation["pipe_ste_pct"],
                    "launch_algorithm": launch["algorithm"],
                    "launch_grid_size": launch["grid_size"],
                    "launch_block_size": launch["block_size"],
                    "launch_execution_variant": launch["execution_variant"],
                    "launch_tile_shape": json.dumps(
                        None if launch["tiling"] is None
                        else launch["tiling"]["tile_shape"],
                        sort_keys=True),
                    "launch_num_kv_splits": (
                        None if launch["tiling"] is None
                        else launch["tiling"]["num_kv_splits"]),
                    **task_axis_summary(task["metadata"], task["family"]),
                    **bucket,
                })
    return rows


def baseline_rows(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = []
    for result_scope, source_rows in (
            ("real_kernel", kernel_rows),
            ("aggregate_task", task_rows)):
        for hardware_id, _ in HARDWARE:
            selected_hardware = [
                row for row in source_rows
                if row["hardware_id"] == hardware_id]
            for family in ("ALL", *FAMILY_LABELS):
                if family == "ALL":
                    selected = selected_hardware
                elif family in FAMILY_LABELS:
                    selected = [
                        row for row in selected_hardware
                        if row["family"] == family]
                else:
                    raise ValueError(
                        f"unknown baseline family: {family!r}")
                statuses = [classify_comparison_status(row) for row in selected]
                comparable = [
                    row for row in selected
                    if classify_comparison_status(row) == "comparable"
                ]
                diagnostic_metrics = (
                    metric_summary([
                        (
                            row["predicted_ns"],
                            observed_duration_ns(row, result_scope),
                        )
                        for row in selected
                    ])
                    if selected else None
                )
                metrics = (
                    metric_summary([
                        (
                            row["predicted_ns"],
                            observed_duration_ns(row, result_scope),
                        )
                        for row in comparable
                    ])
                    if comparable else None
                )
                if result_scope == "real_kernel":
                    task_count = len({row["case_id"] for row in selected})
                    kernel_invocation_count = len(selected)
                elif result_scope == "aggregate_task":
                    task_count = len(selected)
                    kernel_invocation_count = sum(
                        int(row["kernel_count"]) for row in selected)
                else:
                    raise AssertionError(
                        f"unhandled baseline scope: {result_scope!r}")
                support_counts = {
                    level: sum(
                        row["support_level"] == level for row in selected)
                    for level in SUPPORT_LEVELS
                }
                rows.append({
                    "result_scope": result_scope,
                    "hardware_id": hardware_id,
                    "family": family,
                    "row_count": len(selected),
                    "comparable_count": sum(
                        status == "comparable" for status in statuses),
                    "not_comparable_count": sum(
                        status == "not_comparable" for status in statuses),
                    "unobserved_count": sum(
                        status == "unobserved" for status in statuses),
                    "task_count": task_count,
                    "kernel_invocation_count": kernel_invocation_count,
                    "mape_pct": (
                        None if metrics is None else metrics["mape_pct"]),
                    "mdape_pct": (
                        None if metrics is None else metrics["mdape_pct"]),
                    "p50_ape_pct": (
                        None if metrics is None else metrics["p50_ape_pct"]),
                    "p90_ape_pct": (
                        None if metrics is None else metrics["p90_ape_pct"]),
                    "p99_ape_pct": (
                        None if metrics is None else metrics["p99_ape_pct"]),
                    "max_ape_pct": (
                        None if metrics is None else metrics["max_ape_pct"]),
                    "weighted_ape_pct": (
                        None if metrics is None
                        else metrics["weighted_ape_pct"]),
                    "mean_signed_error_pct": (
                        None if metrics is None
                        else metrics["mean_signed_error_pct"]),
                    "median_signed_error_pct": (
                        None if metrics is None
                        else metrics["median_signed_error_pct"]),
                    "diagnostic_mape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["mape_pct"]),
                    "diagnostic_mdape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["mdape_pct"]),
                    "diagnostic_p50_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["p50_ape_pct"]),
                    "diagnostic_p90_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["p90_ape_pct"]),
                    "diagnostic_p99_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["p99_ape_pct"]),
                    "diagnostic_max_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["max_ape_pct"]),
                    "diagnostic_weighted_ape_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["weighted_ape_pct"]),
                    "diagnostic_mean_signed_error_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["mean_signed_error_pct"]),
                    "diagnostic_median_signed_error_pct": (
                        None if diagnostic_metrics is None
                        else diagnostic_metrics["median_signed_error_pct"]),
                    **{
                        f"{level}_row_count": count
                        for level, count in support_counts.items()
                    },
                })
    return rows


def before_after_case_rows(
        before_kernel_rows: Sequence[Mapping[str, Any]],
        before_task_rows: Sequence[Mapping[str, Any]],
        after_kernel_rows: Sequence[Mapping[str, Any]],
        after_task_rows: Sequence[Mapping[str, Any]],
        ) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Join historical predictions onto current frozen physical buckets."""

    result = []
    scope_audits = []
    for result_scope, before_rows, after_rows in (
            ("real_kernel", before_kernel_rows, after_kernel_rows),
            ("aggregate_task", before_task_rows, after_task_rows)):
        before_index = index_result_rows(before_rows, result_scope)
        after_index = index_result_rows(after_rows, result_scope)
        before_keys = set(before_index)
        after_keys = set(after_index)
        missing_before = sorted(after_keys - before_keys)
        missing_after = sorted(before_keys - after_keys)
        if missing_before or missing_after:
            raise ValueError(
                f"{result_scope} comparison row sets differ: "
                f"missing_before={missing_before[:5]!r}, "
                f"missing_after={missing_after[:5]!r}")
        if result_scope == "real_kernel":
            identity_fields = (
                "hardware_id", "family", "op_type", "case_id",
                "kernel_ordinal", "kernel_name",
            )
            measured_field = "actual_ns"
        elif result_scope == "aggregate_task":
            identity_fields = (
                "hardware_id", "family", "case_id",
                "kernel_count", "gemm_m", "gemm_n", "gemm_k",
                "query_length", "kv_length", "executed_attention_splits",
                "attention_layout", "moe_experts", "moe_topk",
                "moe_max_tokens_per_expert", "vector_tokens",
            )
            measured_field = "observed_ns"
        else:
            raise AssertionError(
                f"unhandled before/after scope: {result_scope!r}")
        for key in sorted(after_index):
            before = before_index[key]
            after = after_index[key]
            require_same_result_identity(
                before, after, identity_fields, key)
            before_measured = positive_csv_float(
                before, measured_field, key)
            after_measured = positive_csv_float(after, measured_field, key)
            if before_measured != after_measured:
                raise ValueError(
                    f"comparison identity {key!r} changed {measured_field}: "
                    f"{before_measured!r} != {after_measured!r}")
            before_predicted = positive_csv_float(
                before, "predicted_ns", key)
            after_predicted = positive_csv_float(
                after, "predicted_ns", key)
            before_signed = (
                (before_predicted - after_measured) / after_measured * 100.0)
            after_signed = (
                (after_predicted - after_measured) / after_measured * 100.0)
            before_absolute = abs(before_signed)
            after_absolute = abs(after_signed)
            comparison_status = classify_comparison_status(after)
            comparison_gate_passed = comparison_status == "comparable"
            if result_scope == "real_kernel":
                kernel_ordinal = int(after["kernel_ordinal"])
                kernel_kind = after["kernel_kind"]
                before_variant_or_execution_graph = before["variant"]
                variant_or_execution_graph = after["variant"]
            elif result_scope == "aggregate_task":
                kernel_ordinal = None
                kernel_kind = None
                before_variant_or_execution_graph = before[
                    "task_execution_signature"]
                variant_or_execution_graph = after[
                    "task_execution_signature"]
            else:
                raise AssertionError(
                    f"unhandled before/after scope: {result_scope!r}")
            before_model_identity = comparison_model_identity(
                before, result_scope)
            after_model_identity = comparison_model_identity(
                after, result_scope)
            result.append({
                "result_scope": result_scope,
                "hardware_id": after["hardware_id"],
                "family": after["family"],
                "case_id": after["case_id"],
                "kernel_ordinal": kernel_ordinal,
                "kernel_kind": kernel_kind,
                "before_variant_or_execution_graph": (
                    before_variant_or_execution_graph),
                "variant_or_execution_graph": variant_or_execution_graph,
                "before_model_identity": before_model_identity,
                "after_model_identity": after_model_identity,
                "model_identity_changed": (
                    before_model_identity != after_model_identity),
                "timing_source": after["timing_source"],
                "measured_ns": after_measured,
                "before_predicted_ns": before_predicted,
                "after_predicted_ns": after_predicted,
                "prediction_change_ns": after_predicted - before_predicted,
                "before_signed_error_pct": before_signed,
                "after_signed_error_pct": after_signed,
                "before_absolute_error_pct": before_absolute,
                "after_absolute_error_pct": after_absolute,
                "ape_change_pct_points": after_absolute - before_absolute,
                "before_absolute_duration_error_ns": abs(
                    before_predicted - after_measured),
                "after_absolute_duration_error_ns": abs(
                    after_predicted - after_measured),
                "absolute_duration_error_change_ns": (
                    abs(after_predicted - after_measured)
                    - abs(before_predicted - after_measured)),
                "before_case_gate_passed": (
                    comparison_gate_passed
                    and before_absolute <= CASE_APE_LIMIT_PCT),
                "after_case_gate_passed": (
                    comparison_gate_passed
                    and after_absolute <= CASE_APE_LIMIT_PCT),
                "before_support_level": before["support_level"],
                "after_support_level": after["support_level"],
                "support_level_changed": (
                    before["support_level"] != after["support_level"]),
                "bucket_schema": after["bucket_schema"],
                "physical_bucket_id": after["physical_bucket_id"],
                "physical_bucket_short_id": after[
                    "physical_bucket_short_id"],
                "bucket_dimensions": after["bucket_dimensions"],
                "bucket_basis": after["bucket_basis"],
                "comparison_status": comparison_status,
                "comparison_reason": after["comparison_reason"],
            })
        scope_audits.append({
            "result_scope": result_scope,
            "before_row_count": len(before_rows),
            "after_row_count": len(after_rows),
            "joined_row_count": len(after_keys),
            "before_duplicate_key_count": 0,
            "after_duplicate_key_count": 0,
            "missing_before_count": 0,
            "missing_after_count": 0,
            "identity_mismatch_count": 0,
            "measurement_mismatch_count": 0,
            "model_identity_change_count": sum(
                row["model_identity_changed"]
                for row in result
                if row["result_scope"] == result_scope),
        })
    return result, {"scopes": scope_audits}


def before_after_bucket_rows(
        case_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in case_rows:
        key = (row["result_scope"], row["physical_bucket_id"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        result_scope, bucket_id = key
        values = groups[key]
        first = values[0]
        for row in values[1:]:
            for field in (
                    "hardware_id", "bucket_schema", "bucket_dimensions",
                    "bucket_basis", "physical_bucket_short_id"):
                if row[field] != first[field]:
                    raise ValueError(
                        f"before/after bucket {key!r} disagrees on {field}")
        families = sorted({row["family"] for row in values})
        statuses = [classify_comparison_status(row) for row in values]
        status_counts = {
            status: sum(value == status for value in statuses)
            for status in COMPARISON_STATUSES
        }
        comparison_coverage_passed = (
            status_counts["comparable"] == len(values))
        before_metrics = metric_summary([
            (float(row["before_predicted_ns"]), float(row["measured_ns"]))
            for row in values
        ])
        after_metrics = metric_summary([
            (float(row["after_predicted_ns"]), float(row["measured_ns"]))
            for row in values
        ])
        before_mape_passed = (
            comparison_coverage_passed
            and before_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        after_mape_passed = (
            comparison_coverage_passed
            and after_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        before_max_passed = (
            comparison_coverage_passed
            and before_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        after_max_passed = (
            comparison_coverage_passed
            and after_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        before_diagnostic_mape_passed = (
            before_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        after_diagnostic_mape_passed = (
            after_metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        before_diagnostic_max_passed = (
            before_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        after_diagnostic_max_passed = (
            after_metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        result.append({
            "result_scope": result_scope,
            "hardware_id": first["hardware_id"],
            "family": first["family"] if len(families) == 1 else "mixed",
            "physical_bucket_id": bucket_id,
            "physical_bucket_short_id": first["physical_bucket_short_id"],
            "bucket_schema": first["bucket_schema"],
            "bucket_dimensions": first["bucket_dimensions"],
            "bucket_basis": first["bucket_basis"],
            "row_count": len(values),
            "comparable_count": status_counts["comparable"],
            "not_comparable_count": status_counts["not_comparable"],
            "unobserved_count": status_counts["unobserved"],
            "comparison_coverage_pct": (
                status_counts["comparable"] / len(values) * 100.0),
            "comparison_coverage_gate_passed": (
                comparison_coverage_passed),
            "before_mape_pct": before_metrics["mape_pct"],
            "after_mape_pct": after_metrics["mape_pct"],
            "mape_change_pct_points": (
                after_metrics["mape_pct"] - before_metrics["mape_pct"]),
            "before_mdape_pct": before_metrics["mdape_pct"],
            "after_mdape_pct": after_metrics["mdape_pct"],
            "before_p90_ape_pct": before_metrics["p90_ape_pct"],
            "after_p90_ape_pct": after_metrics["p90_ape_pct"],
            "before_p99_ape_pct": before_metrics["p99_ape_pct"],
            "after_p99_ape_pct": after_metrics["p99_ape_pct"],
            "before_max_ape_pct": before_metrics["max_ape_pct"],
            "after_max_ape_pct": after_metrics["max_ape_pct"],
            "max_ape_change_pct_points": (
                after_metrics["max_ape_pct"]
                - before_metrics["max_ape_pct"]),
            "before_mean_signed_error_pct": before_metrics[
                "mean_signed_error_pct"],
            "after_mean_signed_error_pct": after_metrics[
                "mean_signed_error_pct"],
            "before_median_signed_error_pct": before_metrics[
                "median_signed_error_pct"],
            "after_median_signed_error_pct": after_metrics[
                "median_signed_error_pct"],
            "before_case_violation_count": sum(
                float(row["before_absolute_error_pct"])
                > CASE_APE_LIMIT_PCT for row in values),
            "after_case_violation_count": sum(
                float(row["after_absolute_error_pct"])
                > CASE_APE_LIMIT_PCT for row in values),
            "before_diagnostic_mape_gate_passed": (
                before_diagnostic_mape_passed),
            "after_diagnostic_mape_gate_passed": (
                after_diagnostic_mape_passed),
            "before_diagnostic_max_ape_gate_passed": (
                before_diagnostic_max_passed),
            "after_diagnostic_max_ape_gate_passed": (
                after_diagnostic_max_passed),
            "before_diagnostic_accuracy_gate_passed": (
                before_diagnostic_mape_passed
                and before_diagnostic_max_passed),
            "after_diagnostic_accuracy_gate_passed": (
                after_diagnostic_mape_passed
                and after_diagnostic_max_passed),
            "before_mape_gate_passed": before_mape_passed,
            "after_mape_gate_passed": after_mape_passed,
            "before_max_ape_gate_passed": before_max_passed,
            "after_max_ape_gate_passed": after_max_passed,
            "before_accuracy_gate_passed": (
                before_mape_passed and before_max_passed),
            "after_accuracy_gate_passed": (
                after_mape_passed and after_max_passed),
        })
    if not result:
        raise ValueError("before/after bucket report has no rows")
    return result


def before_after_summary_rows(
        case_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    for result_scope in ("real_kernel", "aggregate_task"):
        scope_rows = [
            row for row in case_rows
            if row["result_scope"] == result_scope]
        for hardware_id, _ in HARDWARE:
            hardware_rows = [
                row for row in scope_rows
                if row["hardware_id"] == hardware_id]
            if not hardware_rows:
                if result_scope == "aggregate_task":
                    raise ValueError(
                        "before/after summary has no rows for "
                        f"{result_scope}/{hardware_id}/ALL")
                # Canonical per-kernel timings are intentionally sparse: at
                # present only A800 non-Vector datasets expose real-kernel
                # rows.  Report only measured scope/family combinations rather
                # than fabricating empty summaries for OP-timed datasets.
                continue
            for family in ("ALL", *FAMILY_LABELS):
                if family == "ALL":
                    values = hardware_rows
                elif family in FAMILY_LABELS:
                    values = [
                        row for row in hardware_rows
                        if row["family"] == family
                    ]
                else:
                    raise ValueError(
                        f"unknown before/after summary family: {family!r}")
                if not values:
                    if result_scope == "real_kernel":
                        continue
                    raise ValueError(
                        "before/after summary has no rows for "
                        f"{result_scope}/{hardware_id}/{family}")
                before_metrics = metric_summary([
                    (
                        float(row["before_predicted_ns"]),
                        float(row["measured_ns"]),
                    )
                    for row in values
                ])
                after_metrics = metric_summary([
                    (
                        float(row["after_predicted_ns"]),
                        float(row["measured_ns"]),
                    )
                    for row in values
                ])
                # This table is explicitly a provenance-unverified numerical
                # diagnostic.  Count the configured APE condition itself here;
                # comparison eligibility is reported independently and must
                # not be mislabeled as a numerical error violation.
                before_pass_count = sum(
                    float(row["before_absolute_error_pct"])
                    <= CASE_APE_LIMIT_PCT for row in values)
                after_pass_count = sum(
                    float(row["after_absolute_error_pct"])
                    <= CASE_APE_LIMIT_PCT for row in values)
                result.append({
                    "result_scope": result_scope,
                    "hardware_id": hardware_id,
                    "family": family,
                    "row_count": len(values),
                    "before_mape_pct": before_metrics["mape_pct"],
                    "after_mape_pct": after_metrics["mape_pct"],
                    "mape_change_pct_points": (
                        after_metrics["mape_pct"]
                        - before_metrics["mape_pct"]),
                    "before_mdape_pct": before_metrics["mdape_pct"],
                    "after_mdape_pct": after_metrics["mdape_pct"],
                    "before_p90_ape_pct": before_metrics["p90_ape_pct"],
                    "after_p90_ape_pct": after_metrics["p90_ape_pct"],
                    "before_p99_ape_pct": before_metrics["p99_ape_pct"],
                    "after_p99_ape_pct": after_metrics["p99_ape_pct"],
                    "before_max_ape_pct": before_metrics["max_ape_pct"],
                    "after_max_ape_pct": after_metrics["max_ape_pct"],
                    "before_mean_signed_error_pct": before_metrics[
                        "mean_signed_error_pct"],
                    "after_mean_signed_error_pct": after_metrics[
                        "mean_signed_error_pct"],
                    "before_median_signed_error_pct": before_metrics[
                        "median_signed_error_pct"],
                    "after_median_signed_error_pct": after_metrics[
                        "median_signed_error_pct"],
                    "before_case_pass_count": before_pass_count,
                    "after_case_pass_count": after_pass_count,
                    "before_case_violation_count": (
                        len(values) - before_pass_count),
                    "after_case_violation_count": (
                        len(values) - after_pass_count),
                })
    return result


def bucket_rows(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in (*kernel_rows, *task_rows):
        key = (row["result_scope"], row["physical_bucket_id"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        result_scope, bucket_id = key
        values = groups[key]
        first = values[0]
        for row in values[1:]:
            for field in (
                    "bucket_schema", "hardware_id",
                    "bucket_basis", "bucket_dimensions"):
                if row[field] != first[field]:
                    raise RuntimeError(
                        f"physical bucket {bucket_id!r} disagrees on {field}")
        task_family_contexts = sorted({row["family"] for row in values})
        statuses = [classify_comparison_status(row) for row in values]
        status_counts = {
            status: sum(value == status for value in statuses)
            for status in COMPARISON_STATUSES
        }
        comparable_values = [
            row for row in values
            if classify_comparison_status(row) == "comparable"
        ]
        diagnostic_pairs = [
            (float(row["predicted_ns"]),
             observed_duration_ns(row, result_scope))
            for row in values
        ]
        diagnostic_metrics = metric_summary(diagnostic_pairs)
        metrics = (
            metric_summary([
                (float(row["predicted_ns"]),
                 observed_duration_ns(row, result_scope))
                for row in comparable_values
            ])
            if comparable_values else None
        )
        support_counts = {
            level: sum(row["support_level"] == level for row in values)
            for level in SUPPORT_LEVELS
        }
        unknown = {
            row["support_level"] for row in values
            if row["support_level"] not in SUPPORT_LEVELS
        }
        if unknown:
            raise ValueError(
                f"unknown Forward support levels: {sorted(unknown)}")
        complete_support_pct = (
            support_counts["complete"] / len(values) * 100.0)
        comparison_coverage_pct = (
            len(comparable_values) / len(values) * 100.0)
        comparison_coverage_passed = len(comparable_values) == len(values)
        mape_passed = (
            metrics is not None
            and metrics["mape_pct"] <= BUCKET_MAPE_LIMIT_PCT)
        max_ape_passed = (
            metrics is not None
            and metrics["max_ape_pct"] <= CASE_APE_LIMIT_PCT)
        support_passed = complete_support_pct == 100.0
        diagnostic_worst = max(
            values,
            key=lambda row: (
                float(row["absolute_error_pct"]),
                row["case_id"],
                kernel_ordinal_sort_value(row, result_scope),
            ),
        )
        worst = (
            max(
                comparable_values,
                key=lambda row: (
                    float(row["absolute_error_pct"]),
                    row["case_id"],
                    kernel_ordinal_sort_value(row, result_scope),
                ),
            )
            if comparable_values else None
        )
        worst_kernel_ordinal = (
            None if worst is None
            else kernel_ordinal_for_scope(worst, result_scope))
        result.append({
            "bucket_schema": first["bucket_schema"],
            "result_scope": result_scope,
            "hardware_id": first["hardware_id"],
            "family": (
                first["family"] if len(task_family_contexts) == 1
                else "mixed"),
            "task_family_contexts": json.dumps(
                task_family_contexts, separators=(",", ":")),
            "physical_bucket_id": bucket_id,
            "physical_bucket_short_id": first["physical_bucket_short_id"],
            "bucket_dimensions": first["bucket_dimensions"],
            "bucket_basis": first["bucket_basis"],
            "row_count": len(values),
            "comparable_count": status_counts["comparable"],
            "not_comparable_count": status_counts["not_comparable"],
            "unobserved_count": status_counts["unobserved"],
            "comparison_coverage_pct": comparison_coverage_pct,
            "comparison_coverage_gate_passed": comparison_coverage_passed,
            "sample_evidence": (
                "evidence_low_n" if len(values) < 3 else "multi_case"),
            "captured_prediction_count": len(values),
            "complete_support_count": support_counts["complete"],
            "complete_support_pct": complete_support_pct,
            **{
                f"{level}_count": support_counts[level]
                for level in SUPPORT_LEVELS
            },
            "mape_pct": None if metrics is None else metrics["mape_pct"],
            "mdape_pct": None if metrics is None else metrics["mdape_pct"],
            "p90_ape_pct": (
                None if metrics is None else metrics["p90_ape_pct"]),
            "p99_ape_pct": (
                None if metrics is None else metrics["p99_ape_pct"]),
            "max_ape_pct": (
                None if metrics is None else metrics["max_ape_pct"]),
            "mean_signed_error_pct": (
                None if metrics is None
                else metrics["mean_signed_error_pct"]),
            "median_signed_error_pct": (
                None if metrics is None
                else metrics["median_signed_error_pct"]),
            "diagnostic_mape_pct": diagnostic_metrics["mape_pct"],
            "diagnostic_mdape_pct": diagnostic_metrics["mdape_pct"],
            "diagnostic_p90_ape_pct": diagnostic_metrics["p90_ape_pct"],
            "diagnostic_p99_ape_pct": diagnostic_metrics["p99_ape_pct"],
            "diagnostic_max_ape_pct": diagnostic_metrics["max_ape_pct"],
            "diagnostic_mean_signed_error_pct": diagnostic_metrics[
                "mean_signed_error_pct"],
            "diagnostic_median_signed_error_pct": diagnostic_metrics[
                "median_signed_error_pct"],
            "worst_case_id": None if worst is None else worst["case_id"],
            "worst_kernel_ordinal": worst_kernel_ordinal,
            "diagnostic_worst_case_id": diagnostic_worst["case_id"],
            "diagnostic_worst_kernel_ordinal": kernel_ordinal_for_scope(
                diagnostic_worst, result_scope),
            "mape_gate_passed": mape_passed,
            "max_ape_gate_passed": max_ape_passed,
            "accuracy_bucket_gate_passed": (
                comparison_coverage_passed
                and mape_passed and max_ape_passed),
            "support_gate_passed": support_passed,
            "bucket_gate_passed": (
                comparison_coverage_passed
                and mape_passed and max_ape_passed and support_passed),
        })
    if not result:
        raise RuntimeError("physical bucket report produced no rows")
    return result


def severe_duration_rows(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]],
        threshold_pct: float = 50.0) -> list[Mapping[str, Any]]:
    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("severe duration threshold must be positive")
    result = []
    for row in kernel_rows:
        if row["absolute_error_pct"] < threshold_pct:
            continue
        result.append({
            "result_scope": row["result_scope"],
            "hardware_id": row["hardware_id"],
            "family": row["family"],
            "case_id": row["case_id"],
            "kernel_ordinal": row["kernel_ordinal"],
            "kernel_kind": row["kernel_kind"],
            "variant": row["variant"],
            "kernel_count": 1,
            "timing_source": row["timing_source"],
            "predicted_ns": row["predicted_ns"],
            "observed_ns": row["actual_ns"],
            "absolute_duration_error_ns": abs(
                row["predicted_ns"] - row["actual_ns"]),
            "signed_error_pct": row["signed_error_pct"],
            "absolute_error_pct": row["absolute_error_pct"],
            "physical_bucket_id": row["physical_bucket_id"],
            "comparison_status": classify_comparison_status(row),
            "comparison_reason": row["comparison_reason"],
        })
    for row in task_rows:
        if row["absolute_error_pct"] < threshold_pct:
            continue
        result.append({
            "result_scope": row["result_scope"],
            "hardware_id": row["hardware_id"],
            "family": row["family"],
            "case_id": row["case_id"],
            "kernel_ordinal": None,
            "kernel_kind": None,
            "variant": row["variant_signature"],
            "kernel_count": row["kernel_count"],
            "timing_source": row["timing_source"],
            "predicted_ns": row["predicted_ns"],
            "observed_ns": row["observed_ns"],
            "absolute_duration_error_ns": abs(
                row["predicted_ns"] - row["observed_ns"]),
            "signed_error_pct": row["signed_error_pct"],
            "absolute_error_pct": row["absolute_error_pct"],
            "physical_bucket_id": row["physical_bucket_id"],
            "comparison_status": classify_comparison_status(row),
            "comparison_reason": row["comparison_reason"],
        })
    result.sort(key=lambda row: (
        -float(row["observed_ns"]),
        row["result_scope"],
        row["hardware_id"],
        row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    return result


def case_impact_rows(
        bucket_rows: Sequence[Mapping[str, Any]],
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]],
        threshold_pct: float = CASE_APE_LIMIT_PCT,
        ) -> list[Mapping[str, Any]]:
    """Rank diagnostic row discrepancies by absolute time impact."""

    if not math.isfinite(threshold_pct) or threshold_pct <= 0:
        raise ValueError("case-impact threshold must be positive")
    bucket_index = {}
    for bucket in bucket_rows:
        key = (bucket["result_scope"], bucket["physical_bucket_id"])
        if key in bucket_index:
            raise RuntimeError(
                f"duplicate physical bucket in case-impact join: {key!r}")
        bucket_index[key] = bucket

    candidates = []
    for result_scope, source_rows in (
            ("real_kernel", kernel_rows),
            ("aggregate_task", task_rows)):
        for row in source_rows:
            absolute_error_pct = float(row["absolute_error_pct"])
            if absolute_error_pct <= threshold_pct:
                continue
            bucket_key = (result_scope, row["physical_bucket_id"])
            if bucket_key not in bucket_index:
                raise RuntimeError(
                    "case-impact row has no matching physical bucket: "
                    f"{bucket_key!r}")
            bucket = bucket_index[bucket_key]
            measured_ns = observed_duration_ns(row, result_scope)
            predicted_ns = float(row["predicted_ns"])
            if result_scope == "real_kernel":
                measured_duration_field = "actual_ns"
                kernel_kind = row["kernel_kind"]
                variant_or_execution_graph = row["variant"]
                kernel_count = 1
            elif result_scope == "aggregate_task":
                measured_duration_field = "observed_ns"
                kernel_kind = None
                variant_or_execution_graph = row["task_execution_signature"]
                kernel_count = int(row["kernel_count"])
            else:
                raise AssertionError(
                    f"unhandled case-impact scope: {result_scope!r}")
            candidates.append({
                "result_scope": result_scope,
                "hardware_id": row["hardware_id"],
                "family": row["family"],
                "case_id": row["case_id"],
                "kernel_ordinal": kernel_ordinal_for_scope(
                    row, result_scope),
                "kernel_kind": kernel_kind,
                "variant_or_execution_graph": variant_or_execution_graph,
                "kernel_count": kernel_count,
                "timing_source": row["timing_source"],
                "measured_duration_field": measured_duration_field,
                "measured_ns": measured_ns,
                "predicted_ns": predicted_ns,
                "absolute_duration_error_ns": abs(
                    predicted_ns - measured_ns),
                "signed_error_pct": float(row["signed_error_pct"]),
                "absolute_error_pct": absolute_error_pct,
                "case_ape_gate_passed": False,
                "physical_bucket_id": row["physical_bucket_id"],
                "bucket_mape_pct": float(bucket["diagnostic_mape_pct"]),
                "bucket_mape_gate_passed": bool(
                    bucket["diagnostic_mape_pct"]
                    <= BUCKET_MAPE_LIMIT_PCT),
                "bucket_max_ape_pct": float(
                    bucket["diagnostic_max_ape_pct"]),
                "bucket_max_ape_gate_passed": bool(
                    bucket["diagnostic_max_ape_pct"]
                    <= CASE_APE_LIMIT_PCT),
                "support_level": row["support_level"],
                "comparison_status": classify_comparison_status(row),
                "comparison_reason": row["comparison_reason"],
            })
    candidates.sort(key=lambda row: (
        -float(row["absolute_duration_error_ns"]),
        -float(row["absolute_error_pct"]),
        -float(row["measured_ns"]),
        row["result_scope"],
        row["hardware_id"],
        row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    return [
        {"impact_rank": rank, **row}
        for rank, row in enumerate(candidates, start=1)
    ]


def accuracy_gate(
        bucket_rows: Sequence[Mapping[str, Any]],
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    scopes = []
    all_case_violations = []
    all_accuracy_bucket_violations = []
    all_support_bucket_violations = []
    all_comparison_bucket_violations = []
    all_comparison_row_violations = []
    all_goal_bucket_violations = []
    for result_scope, rows in (
            ("real_kernel", kernel_rows),
            ("aggregate_task", task_rows)):
        selected_buckets = [
            row for row in bucket_rows
            if row["result_scope"] == result_scope
        ]
        if not rows or not selected_buckets:
            raise RuntimeError(
                f"accuracy gate has no rows for {result_scope}")
        comparable_rows = [
            row for row in rows
            if classify_comparison_status(row) == "comparable"
        ]
        diagnostic_metrics = metric_summary([
            (
                float(row["predicted_ns"]),
                observed_duration_ns(row, result_scope),
            )
            for row in rows
        ])
        metrics = (
            metric_summary([
                (
                    float(row["predicted_ns"]),
                    observed_duration_ns(row, result_scope),
                )
                for row in comparable_rows
            ])
            if comparable_rows else None
        )
        comparison_row_violations = [
            {
                "result_scope": result_scope,
                "hardware_id": row["hardware_id"],
                "family": row["family"],
                "case_id": row["case_id"],
                "kernel_ordinal": kernel_ordinal_for_scope(
                    row, result_scope),
                "comparison_status": classify_comparison_status(row),
                "comparison_reason": row["comparison_reason"],
                "physical_bucket_id": row["physical_bucket_id"],
            }
            for row in rows
            if classify_comparison_status(row) != "comparable"
        ]
        case_violations = [
            {
                "result_scope": result_scope,
                "hardware_id": row["hardware_id"],
                "family": row["family"],
                "case_id": row["case_id"],
                "kernel_ordinal": kernel_ordinal_for_scope(
                    row, result_scope),
                "absolute_error_pct": row["absolute_error_pct"],
                "physical_bucket_id": row["physical_bucket_id"],
            }
            for row in comparable_rows
            if row["absolute_error_pct"] > CASE_APE_LIMIT_PCT
        ]
        goal_bucket_violations = [
            {
                "result_scope": result_scope,
                "physical_bucket_id": row["physical_bucket_id"],
                "mape_pct": row["mape_pct"],
                "max_ape_pct": row["max_ape_pct"],
                "diagnostic_mape_pct": row["diagnostic_mape_pct"],
                "diagnostic_max_ape_pct": row[
                    "diagnostic_max_ape_pct"],
                "comparison_coverage_pct": row[
                    "comparison_coverage_pct"],
                "comparison_coverage_gate_passed": row[
                    "comparison_coverage_gate_passed"],
                "complete_support_pct": row["complete_support_pct"],
                "mape_gate_passed": row["mape_gate_passed"],
                "max_ape_gate_passed": row["max_ape_gate_passed"],
                "support_gate_passed": row["support_gate_passed"],
            }
            for row in selected_buckets
            if not row["bucket_gate_passed"]
        ]
        accuracy_bucket_violations = [
            row for row in goal_bucket_violations
            if row["comparison_coverage_gate_passed"]
            and (not row["mape_gate_passed"]
                 or not row["max_ape_gate_passed"])
        ]
        comparison_bucket_violations = [
            row for row in goal_bucket_violations
            if not row["comparison_coverage_gate_passed"]
        ]
        support_bucket_violations = [
            row for row in goal_bucket_violations
            if not row["support_gate_passed"]
        ]
        all_case_violations.extend(case_violations)
        all_accuracy_bucket_violations.extend(accuracy_bucket_violations)
        all_support_bucket_violations.extend(support_bucket_violations)
        all_comparison_bucket_violations.extend(
            comparison_bucket_violations)
        all_comparison_row_violations.extend(comparison_row_violations)
        all_goal_bucket_violations.extend(goal_bucket_violations)
        comparison_coverage_passed = not comparison_row_violations
        diagnostic_case_violation_count = sum(
            float(row["absolute_error_pct"]) > CASE_APE_LIMIT_PCT
            for row in rows
        )
        diagnostic_bucket_mape_violation_count = sum(
            float(row["diagnostic_mape_pct"]) > BUCKET_MAPE_LIMIT_PCT
            for row in selected_buckets
        )
        diagnostic_bucket_max_ape_violation_count = sum(
            float(row["diagnostic_max_ape_pct"]) > CASE_APE_LIMIT_PCT
            for row in selected_buckets
        )
        numeric_accuracy_passed = (
            bool(comparable_rows)
            and not case_violations
            and not accuracy_bucket_violations)
        scopes.append({
            "result_scope": result_scope,
            "row_count": len(rows),
            "comparable_row_count": len(comparable_rows),
            "not_comparable_row_count": sum(
                classify_comparison_status(row) == "not_comparable"
                for row in rows),
            "unobserved_row_count": sum(
                classify_comparison_status(row) == "unobserved"
                for row in rows),
            "bucket_count": len(selected_buckets),
            "mape_pct": None if metrics is None else metrics["mape_pct"],
            "p99_ape_pct": (
                None if metrics is None else metrics["p99_ape_pct"]),
            "max_ape_pct": (
                None if metrics is None else metrics["max_ape_pct"]),
            "mean_signed_error_pct": (
                None if metrics is None
                else metrics["mean_signed_error_pct"]),
            "diagnostic_mape_pct": diagnostic_metrics["mape_pct"],
            "diagnostic_p99_ape_pct": diagnostic_metrics["p99_ape_pct"],
            "diagnostic_max_ape_pct": diagnostic_metrics["max_ape_pct"],
            "diagnostic_mean_signed_error_pct": diagnostic_metrics[
                "mean_signed_error_pct"],
            "case_violation_count": len(case_violations),
            "diagnostic_case_violation_count": (
                diagnostic_case_violation_count),
            "diagnostic_bucket_mape_violation_count": (
                diagnostic_bucket_mape_violation_count),
            "diagnostic_bucket_max_ape_violation_count": (
                diagnostic_bucket_max_ape_violation_count),
            "comparison_row_violation_count": len(
                comparison_row_violations),
            "comparison_bucket_violation_count": len(
                comparison_bucket_violations),
            "accuracy_bucket_violation_count": len(
                accuracy_bucket_violations),
            "support_bucket_violation_count": len(
                support_bucket_violations),
            "goal_bucket_violation_count": len(goal_bucket_violations),
            "comparison_coverage_gate_passed": (
                comparison_coverage_passed),
            "numeric_accuracy_gate_passed": numeric_accuracy_passed,
            "accuracy_gate_passed": (
                comparison_coverage_passed and numeric_accuracy_passed),
            "model_closure_gate_passed": (
                comparison_coverage_passed
                and numeric_accuracy_passed
                and not goal_bucket_violations),
        })
    all_case_violations.sort(key=lambda row: (
        -float(row["absolute_error_pct"]), row["result_scope"],
        row["hardware_id"], row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    all_accuracy_bucket_violations.sort(key=lambda row: (
        -float(row["mape_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    all_comparison_bucket_violations.sort(key=lambda row: (
        float(row["comparison_coverage_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    all_comparison_row_violations.sort(key=lambda row: (
        row["result_scope"], row["hardware_id"], row["family"],
        row["case_id"],
        -1 if row["kernel_ordinal"] is None else int(row["kernel_ordinal"]),
    ))
    all_support_bucket_violations.sort(key=lambda row: (
        float(row["complete_support_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    all_goal_bucket_violations.sort(key=lambda row: (
        -float(row["diagnostic_mape_pct"]), row["result_scope"],
        row["physical_bucket_id"],
    ))
    return {
        "schema_version": 1,
        "bucket_schema": BUCKET_SCHEMA_VERSION,
        "thresholds": {
            "bucket_mape_pct_max_inclusive": BUCKET_MAPE_LIMIT_PCT,
            "case_ape_pct_max_inclusive": CASE_APE_LIMIT_PCT,
            "complete_support_pct_required": 100.0,
        },
        "mape_definition": (
            "arithmetic mean of per-row absolute percentage error; signed "
            "errors are never allowed to cancel"),
        "diagnostic_metric_policy": (
            "diagnostic_* fields retain arithmetic discrepancies for triage; "
            "only rows with comparison_status=comparable enter the accuracy "
            "denominator"),
        "scopes": scopes,
        "case_violations": all_case_violations,
        "comparison_row_violations": all_comparison_row_violations,
        "comparison_bucket_violations": all_comparison_bucket_violations,
        "accuracy_bucket_violations": all_accuracy_bucket_violations,
        "support_bucket_violations": all_support_bucket_violations,
        "goal_bucket_violations": all_goal_bucket_violations,
        "accuracy_gate_passed": all(
            row["accuracy_gate_passed"] for row in scopes),
        "model_closure_gate_passed": all(
            row["model_closure_gate_passed"] for row in scopes),
    }


def comparison_summary(
        comparison_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    groups = {}
    for row in comparison_rows:
        key = (row["hardware_id"], row["family"], row["component"], row["unit"])
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        hardware_id, family, component, unit = key
        values = groups[key]
        statuses = [classify_comparison_status(row) for row in values]
        comparable = [
            row for row in values
            if classify_comparison_status(row) == "comparable"
        ]
        metrics = component_ratio_summary(comparable)
        diagnostic_metrics = component_ratio_summary(values)
        result.append({
            "hardware_id": hardware_id,
            "family": family,
            "component": component,
            "unit": unit,
            "row_count": len(values),
            "comparable_count": sum(
                status == "comparable" for status in statuses),
            "not_comparable_count": sum(
                status == "not_comparable" for status in statuses),
            "unobserved_count": sum(
                status == "unobserved" for status in statuses),
            **metrics,
            "diagnostic_positive_pair_count": diagnostic_metrics[
                "positive_pair_count"],
            "diagnostic_zero_mismatch_count": diagnostic_metrics[
                "zero_mismatch_count"],
            "diagnostic_median_predicted_over_observed": diagnostic_metrics[
                "median_predicted_over_observed"],
            "diagnostic_p10_predicted_over_observed": diagnostic_metrics[
                "p10_predicted_over_observed"],
            "diagnostic_p90_predicted_over_observed": diagnostic_metrics[
                "p90_predicted_over_observed"],
            "diagnostic_mape_pct": diagnostic_metrics["mape_pct"],
        })
    return result


def traffic_outlier_rows(
        comparison_rows: Sequence[Mapping[str, Any]],
        relative_error_threshold: float = 0.5) -> list[Mapping[str, Any]]:
    """Return one self-contained row per severe directional traffic pair.

    A row is severe when its observed-relative error is at least the requested
    threshold.  Zero/non-zero mismatches are retained explicitly: predicted
    zero versus observed positive has 100% error, while observed zero versus
    predicted positive has no finite observed-relative percentage and is
    labelled as an infinite overprediction.  Dropping either case would hide
    exactly the missing-write and writeback-boundary defects this report is
    intended to expose.
    """

    if not 0.0 < relative_error_threshold < 1.0:
        raise ValueError("traffic outlier threshold must be within (0, 1)")
    traffic_components = frozenset({
        "demand:l2_cache",
        "demand:dram",
        *DIRECTIONAL_TRAFFIC_COMPONENTS,
    })
    result = []
    for row in comparison_rows:
        component = row["component"]
        if component not in traffic_components:
            continue
        predicted = float(row["predicted"])
        observed = float(row["observed"])
        if (not math.isfinite(predicted) or predicted < 0
                or not math.isfinite(observed) or observed < 0):
            raise ValueError(
                "traffic comparison values must be finite and non-negative: "
                f"component={component!r}, predicted={predicted!r}, "
                f"observed={observed!r}")
        if observed > 0:
            ratio = predicted / observed
            absolute_error_pct = abs(predicted - observed) / observed * 100.0
            zero_mismatch = (
                "predicted_zero_observed_positive"
                if predicted == 0 else None)
            severe = absolute_error_pct >= relative_error_threshold * 100.0
        elif predicted > 0:
            ratio = None
            absolute_error_pct = None
            zero_mismatch = "observed_zero_predicted_positive"
            severe = True
        else:
            continue
        if not severe:
            continue
        result.append({
            "hardware_id": row["hardware_id"],
            "family": row["family"],
            "case_id": row["case_id"],
            "kernel_ordinal": row["kernel_ordinal"],
            "variant": row["variant"],
            "kernel_name": row["kernel_name"],
            "actual_ns": row["actual_ns"],
            "timing_source": row["timing_source"],
            "component": component,
            "predicted_bytes": predicted,
            "observed_bytes": observed,
            "predicted_over_observed": ratio,
            "absolute_error_pct": absolute_error_pct,
            "zero_mismatch": zero_mismatch,
            "severity_reason": (
                "zero_mismatch" if zero_mismatch is not None
                else "observed_relative_error_threshold_met"),
            "relative_error_threshold_pct": (
                relative_error_threshold * 100.0),
            "observation_metric": row["metric"],
            "provenance_kind": row["provenance_kind"],
            "provider": row["provider"],
            "comparison_status": classify_comparison_status(row),
            "comparison_reason": row["comparison_reason"],
        })
    result.sort(key=lambda row: (
        row["actual_ns"] is None,
        -float(row["actual_ns"] or 0.0),
        row["hardware_id"],
        row["family"],
        row["case_id"],
        int(row["kernel_ordinal"]),
        row["component"],
    ))
    return result


def coverage_rows(
        documents: Sequence[Mapping[str, Any]],
        comparison_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    components = (
        *(f"demand:{role}" for role in DEMAND_ROLES),
        *DIRECTIONAL_TRAFFIC_COMPONENTS,
        *(f"service_time:{role}" for role in TIME_ROLES),
    )
    evidenced = {
        (hardware_id, family, component): 0
        for hardware_id, _ in HARDWARE
        for family in FAMILY_LABELS
        for component in components
    }
    positive_pairs = dict.fromkeys(evidenced, 0)
    for row in comparison_rows:
        key = (row["hardware_id"], row["family"], row["component"])
        if key not in evidenced:
            raise ValueError(f"unexpected evidence component: {key!r}")
        evidenced[key] += 1
        if row["predicted"] > 0 and row["observed"] > 0:
            positive_pairs[key] += 1
    rows = []
    for document in documents:
        hardware_id = document["hardware_id"]
        for family in FAMILY_LABELS:
            kernels = [
                kernel
                for task in document["tasks"] if task["family"] == family
                for kernel in task["kernels"]
            ]
            if not kernels:
                raise RuntimeError(
                    f"{hardware_id} has no {family} kernels")
            for component in components:
                key = (hardware_id, family, component)
                rows.append({
                    "hardware_id": hardware_id,
                    "family": family,
                    "component": component,
                    "kernel_count": len(kernels),
                    "profiler_evidenced_count": evidenced[key],
                    "positive_pair_count": positive_pairs[key],
                })
    return rows
