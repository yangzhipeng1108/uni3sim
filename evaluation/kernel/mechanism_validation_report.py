"""Evidence-preserving reports for single-kernel mechanism validation.

This module reads production predictor inputs and diagnostics.  It does not
reimplement or adjust a predictor formula, and measured duration is retained
only as an evaluation target.
"""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import argparse

import csv

from concurrent.futures import ProcessPoolExecutor

from dataclasses import fields, is_dataclass

import hashlib

import json

import math

from pathlib import Path

import platform

import subprocess

import sys

import time

from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from evaluation.kernel.dataset import (
    ATTENTION_OP_TYPES,
    MOE_OP_TYPES,
    VECTOR_OP_TYPES,
    get_hardware_config_path,
    get_kernel_dataset_path,
    load_kernel_dataset,
    to_float,
)

from evaluation.kernel.oracle_observation import (
    single_counterfactual_inputs,
)

from src.kernel.kernel_base import create_predictor

from evaluation.kernel.evaluation_runner import process_op_star

from evaluation.kernel.mechanism_capture import (
    BEFORE_AFTER_BUCKET_FIELDS,
    BEFORE_AFTER_CASE_FIELDS,
    BEFORE_AFTER_SUMMARY_FIELDS,
    BUCKET_MAPE_LIMIT_PCT,
    BUCKET_SCHEMA_VERSION,
    C500_SILU_MECHANISM_FIELDS,
    CASE_APE_LIMIT_PCT,
    CASE_IMPACT_FIELDS,
    COMPARISON_STATUSES,
    CURRENT_DATASET_COMPARISON_REASON,
    DATASET_FAMILIES,
    DEMAND_ROLES,
    DIRECTIONAL_TRAFFIC_COMPONENTS,
    FAMILY_LABELS,
    HARDWARE,
    NVIDIA_L2_SECTOR_BYTES,
    OUTPUT_ROOT,
    PHYSICAL_BUCKET_DISCRETE_CONTROLS,
    PROFILER_EVIDENCE_KINDS,
    REPOSITORY_ROOT,
    SEMANTICALLY_EXCLUDED_COMPONENTS,
    SEVERE_DURATION_FIELDS,
    SUPPORT_LEVELS,
    TASK_AGGREGATION_CONTRACT,
    TIME_ROLES,
    VECTOR_TYPES,
    canonical_capture_comparison_evidence as _canonical_capture_comparison_evidence,
    compute_demand_summary as _compute_demand_summary,
    family as _family,
    git_output as _git_output,
    is_semantically_comparable_component as _is_semantically_comparable_component,
    json_value as _json_value,
    launch_summary as _launch_summary,
    lfs_oids as _lfs_oids,
    memory_path_summary as _memory_path_summary,
    nvidia_l2_directional_summary as _nvidia_l2_directional_summary,
    observation_summary as _observation_summary,
    optional_ascend_kb_bytes as _optional_ascend_kb_bytes,
    optional_ascend_percent as _optional_ascend_percent,
    oracle_summary as _oracle_summary,
    pipeline_input_summary as _pipeline_input_summary,
    prediction_input_summary as _prediction_input_summary,
    prediction_summary as _prediction_summary,
    sha256 as _sha256,
    single_counterfactual_summary as _single_counterfactual_summary,
    tiling_summary as _tiling_summary,
    capture_kernel_mechanisms,
)

from evaluation.kernel.mechanism_report_plots import (
    plot_ape_cdf as _plot_ape_cdf,
    plot_ascend_moe_traffic_ratios as _plot_ascend_moe_traffic_ratios,
    plot_attention_split_residual as _plot_attention_split_residual,
    plot_before_after_bucket_accuracy as _plot_before_after_bucket_accuracy,
    plot_before_after_family_accuracy as _plot_before_after_family_accuracy,
    plot_bottlenecks as _plot_bottlenecks,
    plot_bucket_accuracy as _plot_bucket_accuracy,
    plot_c500_silu_mechanism as _plot_c500_silu_mechanism,
    plot_c500_vector_traffic_ratios as _plot_c500_vector_traffic_ratios,
    plot_case_impact_ranking as _plot_case_impact_ranking,
    plot_duration_ratio as _plot_duration_ratio,
    plot_final_latency as _plot_final_latency,
    plot_residual_by_family as _plot_residual_by_family,
    plot_residual_vs_derived_launch_rounds as _plot_residual_vs_derived_launch_rounds,
    plot_residual_vs_scale as _plot_residual_vs_scale,
    plot_support_coverage as _plot_support_coverage,
    plot_target_traffic_ratios as _plot_target_traffic_ratios,
    plot_vector_variant_residual as _plot_vector_variant_residual,
    render_plot_call as _render_plot_call,
)

from evaluation.kernel.mechanism_report_rows import (
    MECHANISM_HARDWARE as _MECHANISM_HARDWARE,
    MechanismHardwareContract as _MechanismHardwareContract,
    accuracy_gate as _accuracy_gate,
    ascend_directional_traffic_rows as _ascend_directional_traffic_rows,
    ascend_moe_tile_replay_rows as _ascend_moe_tile_replay_rows,
    baseline_rows as _baseline_rows,
    before_after_bucket_rows as _before_after_bucket_rows,
    before_after_case_rows as _before_after_case_rows,
    before_after_summary_rows as _before_after_summary_rows,
    bucket_rows as _bucket_rows,
    c500_silu_mechanism_rows as _c500_silu_mechanism_rows,
    c500_silu_mechanism_summary as _c500_silu_mechanism_summary,
    case_impact_rows as _case_impact_rows,
    comparison_model_identity as _comparison_model_identity,
    comparison_rows as _comparison_rows,
    classify_comparison_status as _comparison_status,
    comparison_summary as _comparison_summary,
    component_ratio_summary as _component_ratio_summary,
    counterfactual_rows as _counterfactual_rows,
    counterfactual_summary_rows as _counterfactual_summary_rows,
    coverage_rows as _coverage_rows,
    directional_kernel_traffic_fields as _directional_kernel_traffic_fields,
    directional_traffic_row as _directional_traffic_row,
    error_direction as _error_direction,
    format_percentage as _format_percentage,
    index_result_rows as _index_result_rows,
    kernel_launch_class as _kernel_launch_class,
    kernel_ordinal_for_scope as _kernel_ordinal_for_scope,
    kernel_ordinal_sort_value as _kernel_ordinal_sort_value,
    kernel_ordinal_suffix as _kernel_ordinal_suffix,
    kernel_rows as _kernel_rows,
    memory_path_bytes as _memory_path_bytes,
    metax_directional_traffic_rows as _metax_directional_traffic_rows,
    metric_summary as _metric_summary,
    nvidia_directional_traffic_rows as _nvidia_directional_traffic_rows,
    observed_duration_ns as _observed_duration_ns,
    physical_bucket as _physical_bucket,
    physical_bucket_definitions as _physical_bucket_definitions,
    plot_case_id as _plot_case_id,
    positive_csv_float as _positive_csv_float,
    read_result_csv as _read_result_csv,
    require_same_result_identity as _require_same_result_identity,
    result_join_key as _result_join_key,
    result_scope_title as _result_scope_title,
    severe_duration_rows as _severe_duration_rows,
    support_level as _support_level,
    task_axis_summary as _task_axis_summary,
    task_execution_signature as _task_execution_signature,
    task_rows as _task_rows,
    traffic_outlier_rows as _traffic_outlier_rows,
    verify_manifest_artifact as _verify_manifest_artifact,
    write_csv as _write_csv,
)


def _validate_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(
            "mechanism-validation output must be a run directory below "
            f"{root}")
    return resolved


def _hardware_work(hardware_id: str):
    raw_data = load_kernel_dataset(hardware_id, DATASET_FAMILIES)
    hardware_config = get_hardware_config_path(hardware_id)
    packed = [
        (
            task_key,
            entry,
            hardware_config,
            True,
            True,
            False,
            0.0,
            False,
            None,
            True,
        )
        for task_key, entry in sorted(raw_data.items())
    ]
    return hardware_config, packed


def _hardware_document(hardware_id, hardware_config, results):
    tasks = []
    for result in results:
        task_key, metadata, oracle_pair, forward_pair, _, _, kernels = result
        if not isinstance(forward_pair, tuple) or len(forward_pair) != 2:
            raise RuntimeError(
                f"{hardware_id} task {task_key!r} has no Forward result")
        tasks.append({
            "case_id": task_key,
            "family": _family(metadata["op_type"]),
            "metadata": metadata,
            "oracle_pair": oracle_pair,
            "forward_pair": list(forward_pair),
            "kernels": kernels,
        })
    return {
        "hardware_id": hardware_id,
        "hardware_config": str(Path(hardware_config).relative_to(REPOSITORY_ROOT)),
        "tasks": tasks,
    }


def _collect_hardware_matrix(jobs: int) -> list[Mapping[str, Any]]:
    """Collect all hardware with one interleaved process pool."""

    work = {}
    maximum_cases = 0
    for hardware_id, _ in HARDWARE:
        hardware_config, packed = _hardware_work(hardware_id)
        work[hardware_id] = {
            "hardware_config": hardware_config,
            "packed": packed,
            "results": [],
        }
        maximum_cases = max(maximum_cases, len(packed))
    owners, results = interleave_and_process(
        work, [hardware_id for hardware_id, _ in HARDWARE],
        maximum_cases, jobs, process_op_star)
    for hardware_id, result in zip(owners, results, strict=True):
        work[hardware_id]["results"].append(result)
    return [
        _hardware_document(
            hardware_id,
            work[hardware_id]["hardware_config"],
            work[hardware_id]["results"],
        )
        for hardware_id, _ in HARDWARE
    ]


def interleave_and_process(work, hardware_order, maximum_cases, jobs, process):
    """Run one interleaved process pool over every hardware's packed cases.

    Cases are interleaved round-robin across hardware so each worker sees the
    same mix the serial loop would, and the result list carries the hardware
    that produced each entry.  Serial for one worker or a single case.

    ``process`` is the caller's worker callable rather than a name imported
    here: each module keeps its own module-level binding so a test or caller
    can substitute the worker where it is defined.
    """

    interleaved = []
    owners = []
    for case_index in range(maximum_cases):
        for hardware_id in hardware_order:
            packed = work[hardware_id]["packed"]
            if case_index < len(packed):
                interleaved.append(packed[case_index])
                owners.append(hardware_id)
    if jobs == 1 or len(interleaved) <= 1:
        results = [process(item) for item in interleaved]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(process, interleaved))
    return owners, results


def _write_markdown_summary(
        path: Path, baseline_rows, bucket_rows, accuracy_gate,
        comparison_summary, case_impact_rows, c500_silu_rows, artifacts,
        before_after_summary, before_after_bucket_rows,
        before_results_dir) -> None:
    comparison_reasons = sorted({
        row["comparison_reason"]
        for row in accuracy_gate["comparison_row_violations"]
    })
    if not comparison_reasons:
        comparison_reason_text = "none"
    else:
        comparison_reason_text = ", ".join(
            f"`{reason}`" for reason in comparison_reasons)
    lines = [
        "# Kernel mechanism validation: generated data summary",
        "",
        "This file reports results computed from current Forward and canonical observations. Final latency is never used to infer or adjust model parameters. Causal conclusions about mechanisms require review reports and independent evidence.",
        "",
        f"**Important: canonical duration values can identify outliers, but capture-time runtime, source snapshot, and runner/operator binary identities were not recorded. These values are provenance-unverified diagnostics and are excluded from the formal {CASE_APE_LIMIT_PCT:.0f}%/{BUCKET_MAPE_LIMIT_PCT:.0f}% acceptance denominator.**",
        "",
        f"Current formal-comparison gaps: {comparison_reason_text}.",
        "",
        "## Definitions",
        "",
        "| Level | Row meaning | Observed field | Predicted field | Row threshold | Physical-bucket threshold |",
        "|---|---|---|---|---:|---:|",
        f"| `real_kernel` | One actual profiler kernel invocation with its own duration | `actual_ns` | Single-kernel Forward | APE ≤ {CASE_APE_LIMIT_PCT:.0f}% | Arithmetic mean MAPE within the bucket ≤ {BUCKET_MAPE_LIMIT_PCT:.0f}% |",
        f"| `aggregate_task` | One dataset task; observation preserves the canonical OP/kernel-sum boundary | `observed_ns` | `{TASK_AGGREGATION_CONTRACT}` | APE ≤ {CASE_APE_LIMIT_PCT:.0f}% | Arithmetic mean MAPE within the bucket ≤ {BUCKET_MAPE_LIMIT_PCT:.0f}% |",
        "",
        "## Accuracy gates",
        "",
        f"Accuracy gate: **{'Passed' if accuracy_gate['accuracy_gate_passed'] else 'Failed'}**. Full mechanism-closure gate: **{'Passed' if accuracy_gate['model_closure_gate_passed'] else 'Failed'}**. Each physical bucket must satisfy both MAPE ≤ {BUCKET_MAPE_LIMIT_PCT:.0f}% and APE for every row ≤ {CASE_APE_LIMIT_PCT:.0f}%; full closure also requires 100% Forward support within the bucket.",
        "",
        f"| Level | Total rows | Formally comparable rows | Unobserved rows | Buckets | Formal coverage passes | Formal accuracy passes | Diagnostic APE > {CASE_APE_LIMIT_PCT:.0f}% rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("real_kernel", "aggregate_task"):
        selected = [
            row for row in bucket_rows if row["result_scope"] == scope]
        scope_gate = next(
            row for row in accuracy_gate["scopes"]
            if row["result_scope"] == scope)
        accuracy_passed = sum(
            row["accuracy_bucket_gate_passed"] for row in selected)
        comparison_passed = sum(
            row["comparison_coverage_gate_passed"] for row in selected)
        lines.append(
            f"| {scope} | {scope_gate['row_count']} | "
            f"{scope_gate['comparable_row_count']} | "
            f"{scope_gate['unobserved_row_count']} | {len(selected)} | "
            f"{comparison_passed} | {accuracy_passed} | "
            f"{scope_gate['diagnostic_case_violation_count']} |")
    if before_after_summary is not None:
        if before_after_bucket_rows is None or before_results_dir is None:
            raise ValueError(
                "before/after summary requires bucket rows and run identity")
        lines.extend([
            "",
            "## Before/after diagnostics for identical cases and frozen physical buckets",
            "",
            f"Before predictions are read from `{before_results_dir}`. Rows are first checked one-to-one by hardware, family, case, and kernel ordinal (tasks have no ordinal), then assigned this run's frozen physical buckets. Any mismatch in observed latency or static launch identity fails immediately. Captures still lack software and binary identities, so these are diagnostic results rather than formal acceptance conclusions.",
            "",
            f"| Level | Hardware | Family | Rows | Before MAPE | After MAPE | Change | Before maximum APE | After maximum APE | Before >{CASE_APE_LIMIT_PCT:.0f}% | After >{CASE_APE_LIMIT_PCT:.0f}% |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in before_after_summary:
            if row["family"] == "ALL":
                continue
            lines.append(
                f"| {row['result_scope']} | {row['hardware_id']} | "
                f"{row['family']} | {row['row_count']} | "
                f"{row['before_mape_pct']:.2f}% | "
                f"{row['after_mape_pct']:.2f}% | "
                f"{row['mape_change_pct_points']:+.2f} pp | "
                f"{row['before_max_ape_pct']:.2f}% | "
                f"{row['after_max_ape_pct']:.2f}% | "
                f"{row['before_case_violation_count']} | "
                f"{row['after_case_violation_count']} |")
        lines.extend([
            "",
            f"| Level | Total buckets | Diagnostic before bucket mean ≤{BUCKET_MAPE_LIMIT_PCT:.0f}% | Diagnostic after bucket mean ≤{BUCKET_MAPE_LIMIT_PCT:.0f}% | Diagnostic before meeting both {BUCKET_MAPE_LIMIT_PCT:.0f}%/{CASE_APE_LIMIT_PCT:.0f}% | Diagnostic after meeting both {BUCKET_MAPE_LIMIT_PCT:.0f}%/{CASE_APE_LIMIT_PCT:.0f}% |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for result_scope in ("real_kernel", "aggregate_task"):
            selected = [
                row for row in before_after_bucket_rows
                if row["result_scope"] == result_scope]
            lines.append(
                f"| {result_scope} | {len(selected)} | "
                f"{sum(row['before_diagnostic_mape_gate_passed'] for row in selected)} | "
                f"{sum(row['after_diagnostic_mape_gate_passed'] for row in selected)} | "
                f"{sum(row['before_diagnostic_accuracy_gate_passed'] for row in selected)} | "
                f"{sum(row['after_diagnostic_accuracy_gate_passed'] for row in selected)} |")
    lines.extend([
        "",
        "## Full Forward baseline",
    ])
    for scope, title in (
            ("real_kernel", "Actual kernel-invocation baseline (`actual_ns`)"),
            (
                "aggregate_task",
                "Aggregate-task baseline (serial sum of Forward kernel predictions; "
                "`observed_ns` preserves canonical task timing)",
            )):
        lines.extend([
            "",
            f"### {title}",
            "",
            "| Hardware | Family | Rows | Comparable | Unobserved | Formal MAPE | Formal P90 | Formal Max | Diagnostic MAPE | Diagnostic P90 | Diagnostic Max | Diagnostic signed bias |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in baseline_rows:
            if row["result_scope"] != scope:
                continue
            lines.append(
                f"| {row['hardware_id']} | {row['family']} | "
                f"{row['row_count']} | {row['comparable_count']} | "
                f"{row['unobserved_count']} | "
                f"{_format_percentage(row['mape_pct'])} | "
                f"{_format_percentage(row['p90_ape_pct'])} | "
                f"{_format_percentage(row['max_ape_pct'])} | "
                f"{_format_percentage(row['diagnostic_mape_pct'])} | "
                f"{_format_percentage(row['diagnostic_p90_ape_pct'])} | "
                f"{_format_percentage(row['diagnostic_max_ape_pct'])} | "
                f"{_format_percentage(row['diagnostic_mean_signed_error_pct'], signed=True)} |")
    lines.extend([
        "",
        f"## Diagnostic APE above {CASE_APE_LIMIT_PCT:.0f}% with the largest absolute time impact",
        "",
        "Provenance-unverified durations are retained for diagnosis and ranked by `|predicted latency − observed latency|`, rather than percentage error or observed latency. This is not formal acceptance. See `case_impact_ranking_over_30pct.csv` for the full list.",
        "",
        "| Rank | Scope | Hardware | Family | Case | Kernel | Measured | Predicted | Abs. delta | APE | Bucket MAPE |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in case_impact_rows[:20]:
        ordinal = (
            "—" if row["kernel_ordinal"] is None
            else str(row["kernel_ordinal"]))
        lines.append(
            f"| {row['impact_rank']} | {row['result_scope']} | "
            f"{row['hardware_id']} | {row['family']} | "
            f"`{row['case_id']}` | {ordinal} | "
            f"{row['measured_ns'] / 1.0e6:.3f} ms | "
            f"{row['predicted_ns'] / 1.0e6:.3f} ms | "
            f"{row['absolute_duration_error_ns'] / 1.0e6:.3f} ms | "
            f"{row['absolute_error_pct']:.1f}% | "
            f"{row['bucket_mape_pct']:.1f}% |")
    c500_silu = _c500_silu_mechanism_summary(c500_silu_rows)
    if c500_silu["instruction_comparison_status"] == "comparable":
        instruction_summary = (
            f"Instruction-count MAPE is {c500_silu['instruction_mape_pct']:.6f}%"
            f"({c500_silu['instruction_exact_count']}/"
            f"{c500_silu['instruction_comparable_count']} match exactly; maximum error "
            f"{c500_silu['instruction_max_ape_pct']:.6f}%)")
    elif c500_silu["instruction_comparison_status"] == "unavailable":
        instruction_summary = (
            "Current Forward does not publish a static "
            "instruction lower bound directly comparable to mcProfiler total instructions; the comparison is explicitly unavailable")
    else:
        raise ValueError(
            "unsupported C500 SiLU instruction comparison status: "
            f"{c500_silu['instruction_comparison_status']!r}")
    lines.extend([
        "",
        "## C500 SiLU: established findings",
        "",
        f"Of {c500_silu['case_count']} cases, Forward underpredicts {c500_silu['underpredicted_count']}; MAPE is {c500_silu['duration_mape_pct']:.2f}%; maximum APE is {c500_silu['duration_max_ape_pct']:.2f}%; only {c500_silu['case_gate_passed_count']}/{c500_silu['case_count']} meet the per-case limit of {CASE_APE_LIMIT_PCT:.0f}%. The case with the largest absolute time difference is `{c500_silu['largest_absolute_delta_case_id']}`: observed {c500_silu['largest_absolute_delta_production_ns'] / 1000.0:.3f} µs, predicted {c500_silu['largest_absolute_delta_predicted_ns'] / 1000.0:.3f} µs, APE {c500_silu['largest_absolute_delta_ape_pct']:.2f}%.",
        "",
        f"Workload validation: {instruction_summary}; global-memory traffic MAPE is {c500_silu['traffic_mape_pct']:.3f}%; maximum error {c500_silu['traffic_max_ape_pct']:.3f}%. An unavailable instruction comparison cannot replace physical evidence; only published traffic demand is directly validated.",
        "",
        f"mcProfiler provides XCORE frequencies for these replays ({c500_silu['replay_core_clock_ghz_min']:.3f}–{c500_silu['replay_core_clock_ghz_max']:.3f} GHz). Converting with that frequency strictly within the same mcProfiler replay gives mean simultaneously active APs of {c500_silu['replay_mean_active_aps_min']:.2f}–{c500_silu['replay_mean_active_aps_max']:.2f}/104, Average Wave Life of {c500_silu['replay_wave_life_cycles_min']:.0f}–{c500_silu['replay_wave_life_cycles_max']:.0f} cycles, and DNoC read latency of {c500_silu['replay_dnoc_latency_cycles_min']:.1f}–{c500_silu['replay_dnoc_latency_cycles_max']:.1f} cycles, with VLS pipeline/wdata stalls. These show substantial waits and dependency chains inside the kernel, so a workload/throughput-only model omits significant time. However, these counters come from a separate instrumented replay; they cannot be divided by mcTracer production duration or directly treated as production utilization or service time.",
        "",
        f"Plot guide: panels A/B use a ratio axis, not a log axis. 1.0 is exact agreement; the green region is ±{BUCKET_MAPE_LIMIT_PCT:.0f}%; orange dashed lines mark per-case ±{CASE_APE_LIMIT_PCT:.0f}% boundaries. Panels C/D describe concurrency and waits within one mcProfiler replay without combining them with production timing. Full per-case values are in `c500_silu_mechanism.csv`.",
        "",
        "![C500 SiLU mechanism boundaries](figures/c500-silu-mechanism-boundary.png)",
    ])
    lines.extend([
        "",
        "## Oracle-normalized profiler intermediate diagnostics",
        "",
        "Only quantities with aligned field semantics and provenance kind `profiler` or `profiler_derived` are listed. Counter captures also lack runtime/source/binary identity, so formal metrics are `n/a`; values appear only in Diagnostic columns for investigation. Zero-valued prediction/observation placeholders and raw 910B compute counts without semantic mapping are excluded from ratios.",
        "",
        "| Hardware | Family | Component | Unit | Rows | Comparable | Formal MAPE | Diagnostic median pred/obs | Diagnostic P10 | Diagnostic P90 | Diagnostic MAPE | Diagnostic zero mismatch |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in comparison_summary:
        if row["diagnostic_median_predicted_over_observed"] is None:
            diagnostic_ratio = diagnostic_p10 = diagnostic_p90 = diagnostic_mape = "n/a"
        else:
            diagnostic_ratio = (
                f"{row['diagnostic_median_predicted_over_observed']:.3f}")
            diagnostic_p10 = (
                f"{row['diagnostic_p10_predicted_over_observed']:.3f}")
            diagnostic_p90 = (
                f"{row['diagnostic_p90_predicted_over_observed']:.3f}")
            diagnostic_mape = _format_percentage(
                row["diagnostic_mape_pct"])
        lines.append(
            f"| {row['hardware_id']} | {row['family']} | "
            f"{row['component']} | {row['unit']} | "
            f"{row['row_count']} | {row['comparable_count']} | "
            f"{_format_percentage(row['mape_pct'])} | "
            f"{diagnostic_ratio} | {diagnostic_p10} | {diagnostic_p90} | "
            f"{diagnostic_mape} | "
            f"{row['diagnostic_zero_mismatch_count']} |")
    lines.extend(["", "## Generated artifacts", ""])
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    path.write_text("\n".join(lines) + "\n")


def generate_report(
        output_dir: Path,
        jobs: int,
        before_results_dir: Path | None = None) -> Mapping[str, Any]:
    started_at = time.perf_counter()
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("jobs must be an integer")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    output_dir = _validate_output_directory(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    figures_dir = output_dir / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_identities = {}
    lfs = _lfs_oids()
    for hardware_id, _ in HARDWARE:
        for family in DATASET_FAMILIES:
            dataset_path = Path(get_kernel_dataset_path(hardware_id, family))
            relative = str(dataset_path.relative_to(REPOSITORY_ROOT))
            if relative not in lfs:
                raise ValueError(
                    f"canonical dataset is not tracked by Git LFS: {relative}")
            dataset_identities[relative] = {
                "lfs_oid_sha256": lfs[relative],
                "worktree_sha256": _sha256(dataset_path),
            }
    launch_data_identities = {}
    for relative, oid in sorted(lfs.items()):
        if not relative.startswith("launch_data/"):
            continue
        launch_path = REPOSITORY_ROOT / relative
        launch_data_identities[relative] = {
            "lfs_oid_sha256": oid,
            "worktree_sha256": _sha256(launch_path),
        }
    config_identities = {
        str(Path(get_hardware_config_path(hardware_id)).relative_to(REPOSITORY_ROOT)): (
            _sha256(Path(get_hardware_config_path(hardware_id))))
        for hardware_id, _ in HARDWARE
    }
    before_manifest = None
    before_manifest_path = None
    before_manifest_sha256 = None
    if before_results_dir is not None:
        before_results_dir = _validate_output_directory(before_results_dir)
        if before_results_dir == output_dir:
            raise ValueError(
                "before-results directory must differ from output directory")
        before_manifest_path = before_results_dir / "manifest.json"
        if not before_manifest_path.is_file():
            raise FileNotFoundError(
                f"before-results manifest is missing: {before_manifest_path}")
        before_manifest = json.loads(before_manifest_path.read_text())
        for field, current_identity in (
                ("dataset_identities", dataset_identities),
                ("launch_data_identities", launch_data_identities),
                ("hardware_config_sha256", config_identities)):
            if before_manifest[field] != current_identity:
                raise ValueError(
                    f"before-results {field} differs from the current run")
        _verify_manifest_artifact(
            before_results_dir, before_manifest, "kernel_results.csv")
        _verify_manifest_artifact(
            before_results_dir, before_manifest, "task_results.csv")
        before_manifest_sha256 = _sha256(before_manifest_path)

    documents = _collect_hardware_matrix(jobs)
    raw_paths = []
    for document in documents:
        raw_path = raw_dir / f"{document['hardware_id']}.json"
        raw_path.write_text(
            json.dumps(_json_value(document), indent=2, sort_keys=True) + "\n")
        raw_paths.append(raw_path)

    task_rows = _task_rows(documents)
    kernel_rows = _kernel_rows(documents)
    counterfactual_rows = _counterfactual_rows(documents)
    counterfactual_summary = _counterfactual_summary_rows(
        counterfactual_rows, task_rows)
    bucket_rows = _bucket_rows(kernel_rows, task_rows)
    accuracy_gate = _accuracy_gate(bucket_rows, kernel_rows, task_rows)
    severe_duration = _severe_duration_rows(kernel_rows, task_rows)
    accuracy_violations = [
        row for row in _severe_duration_rows(
            kernel_rows, task_rows, threshold_pct=CASE_APE_LIMIT_PCT)
        if row["comparison_status"] == "comparable"
        and row["absolute_error_pct"] > CASE_APE_LIMIT_PCT
    ]
    case_impact = _case_impact_rows(
        bucket_rows, kernel_rows, task_rows)
    comparisons = _comparison_rows(documents)
    baseline = _baseline_rows(kernel_rows, task_rows)
    comparison_summary = _comparison_summary(comparisons)
    coverage = _coverage_rows(documents, comparisons)
    traffic_outliers = _traffic_outlier_rows(comparisons)
    ascend_moe_tile_replay = _ascend_moe_tile_replay_rows(documents)
    if len(ascend_moe_tile_replay) != 352:
        raise RuntimeError(
            "Ascend GroupedMatmul replay check expected 352 kernels, got "
            f"{len(ascend_moe_tile_replay)}")
    c500_silu_mechanism = _c500_silu_mechanism_rows(documents)
    if len(c500_silu_mechanism) != 56:
        raise RuntimeError(
            "C500 SiLU mechanism check expected 56 kernels, got "
            f"{len(c500_silu_mechanism)}")

    before_after_cases = None
    before_after_buckets = None
    before_after_summary = None
    before_after_join_audit = None

    _write_csv(output_dir / "task_results.csv", task_rows, tuple(task_rows[0]))
    _write_csv(output_dir / "kernel_results.csv", kernel_rows, tuple(kernel_rows[0]))
    if before_results_dir is not None:
        before_kernel_rows = _read_result_csv(
            before_results_dir / "kernel_results.csv",
            (
                "hardware_id", "family", "op_type", "case_id",
                "kernel_ordinal", "kernel_kind", "variant", "kernel_name",
                "actual_ns", "predicted_ns", "support_level",
                "launch_algorithm", "launch_grid_size", "launch_block_size",
                "launch_execution_variant", "launch_tile_shape",
                "launch_num_kv_splits",
            ),
        )
        before_task_rows = _read_result_csv(
            before_results_dir / "task_results.csv",
            (
                "hardware_id", "family", "case_id", "predicted_ns",
                "observed_ns", "support_level", "kernel_count",
                "task_execution_signature", "gemm_m", "gemm_n", "gemm_k",
                "query_length", "kv_length",
                "executed_attention_splits", "attention_layout",
                "moe_experts", "moe_topk", "moe_max_tokens_per_expert",
                "vector_tokens",
            ),
        )
        after_kernel_rows = _read_result_csv(
            output_dir / "kernel_results.csv",
            (
                "result_scope", "hardware_id", "family", "op_type",
                "case_id", "kernel_ordinal", "kernel_kind", "variant",
                "kernel_name", "actual_ns", "timing_source", "predicted_ns",
                "support_level", "launch_algorithm", "launch_grid_size",
                "launch_block_size", "launch_execution_variant",
                "launch_tile_shape", "launch_num_kv_splits", "bucket_schema",
                "physical_bucket_id", "physical_bucket_short_id",
                "bucket_dimensions", "bucket_basis", "comparison_status",
                "comparison_reason",
            ),
        )
        after_task_rows = _read_result_csv(
            output_dir / "task_results.csv",
            (
                "result_scope", "hardware_id", "family", "case_id",
                "predicted_ns", "observed_ns", "timing_source",
                "support_level", "kernel_count", "task_execution_signature",
                "gemm_m", "gemm_n", "gemm_k", "query_length", "kv_length",
                "executed_attention_splits", "attention_layout",
                "moe_experts", "moe_topk", "moe_max_tokens_per_expert",
                "vector_tokens", "bucket_schema", "physical_bucket_id",
                "physical_bucket_short_id", "bucket_dimensions",
                "bucket_basis", "comparison_status", "comparison_reason",
            ),
        )
        before_after_cases, before_after_join_audit = _before_after_case_rows(
            before_kernel_rows,
            before_task_rows,
            after_kernel_rows,
            after_task_rows,
        )
        before_after_buckets = _before_after_bucket_rows(before_after_cases)
        before_after_summary = _before_after_summary_rows(before_after_cases)
        if before_manifest is None or before_manifest_path is None:
            raise AssertionError("before-results manifest was not loaded")
        if "forward_source_worktree_diff" in before_manifest:
            before_forward_diff_field = "forward_source_worktree_diff"
        elif "forward_source_diff" in before_manifest:
            before_forward_diff_field = "forward_source_diff"
        else:
            raise ValueError(
                "before-results manifest has no Forward source diff identity")
        before_forward_diff = before_manifest[before_forward_diff_field]
        if not isinstance(before_forward_diff, str):
            raise TypeError(
                "before-results Forward source diff identity must be text")
        before_after_join_audit = {
            "before_results_directory": str(
                before_results_dir.relative_to(REPOSITORY_ROOT)),
            "before_manifest_sha256": before_manifest_sha256,
            "before_git_head": before_manifest["git_head"],
            "before_forward_diff_field": before_forward_diff_field,
            "before_forward_diff_sha256": hashlib.sha256(
                before_forward_diff.encode()).hexdigest(),
            "current_git_head": _git_output("rev-parse", "HEAD"),
            "current_forward_worktree_diff_sha256": hashlib.sha256(
                _git_output("diff", "--", "src").encode()).hexdigest(),
            "dataset_identities_equal": True,
            "launch_data_identities_equal": True,
            "hardware_config_identities_equal": True,
            **before_after_join_audit,
        }
        _write_csv(
            output_dir / "before_after_case_results.csv",
            before_after_cases,
            BEFORE_AFTER_CASE_FIELDS,
        )
        _write_csv(
            output_dir / "before_after_bucket_results.csv",
            before_after_buckets,
            BEFORE_AFTER_BUCKET_FIELDS,
        )
        _write_csv(
            output_dir / "before_after_summary.csv",
            before_after_summary,
            BEFORE_AFTER_SUMMARY_FIELDS,
        )
        (output_dir / "before_after_join_audit.json").write_text(
            json.dumps(
                before_after_join_audit, indent=2, sort_keys=True) + "\n")
    _write_csv(
        output_dir / "bucket_results.csv",
        bucket_rows,
        tuple(bucket_rows[0]),
    )
    _write_csv(
        output_dir / "severe_duration_cases_over_50pct.csv",
        severe_duration,
        SEVERE_DURATION_FIELDS,
    )
    _write_csv(
        output_dir / "case_accuracy_violations_over_30pct.csv",
        accuracy_violations,
        SEVERE_DURATION_FIELDS,
    )
    _write_csv(
        output_dir / "case_impact_ranking_over_30pct.csv",
        case_impact,
        CASE_IMPACT_FIELDS,
    )
    (output_dir / "physical_bucket_definitions.json").write_text(
        json.dumps(
            _physical_bucket_definitions(), indent=2, sort_keys=True)
        + "\n")
    (output_dir / "accuracy_gate.json").write_text(
        json.dumps(accuracy_gate, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / "baseline.csv", baseline, tuple(baseline[0]))
    _write_csv(
        output_dir / "component_comparisons.csv",
        comparisons,
        tuple(comparisons[0]),
    )
    _write_csv(
        output_dir / "component_comparison_summary.csv",
        comparison_summary,
        tuple(comparison_summary[0]),
    )
    _write_csv(
        output_dir / "counterfactual_results.csv",
        counterfactual_rows,
        tuple(counterfactual_rows[0]),
    )
    _write_csv(
        output_dir / "counterfactual_summary.csv",
        counterfactual_summary,
        tuple(counterfactual_summary[0]),
    )
    _write_csv(output_dir / "evidence_coverage.csv", coverage, tuple(coverage[0]))
    _write_csv(
        output_dir / "traffic_outliers_over_50pct.csv",
        traffic_outliers,
        tuple(traffic_outliers[0]),
    )
    _write_csv(
        output_dir / "ascend_910b_moe_tile_replay_check.csv",
        ascend_moe_tile_replay,
        tuple(ascend_moe_tile_replay[0]),
    )
    _write_csv(
        output_dir / "c500_silu_mechanism.csv",
        c500_silu_mechanism,
        C500_SILU_MECHANISM_FIELDS,
    )

    figures = [
        figures_dir / "task-duration-overview-loglog.png",
        figures_dir / "final-residual-by-family.png",
        figures_dir / "residual-vs-workload-scale.png",
        figures_dir / "bottleneck-composition.png",
        figures_dir / "forward-support-coverage.png",
        figures_dir / "vector-variant-residual.png",
        figures_dir / "residual-vs-derived-launch-rounds.png",
        figures_dir / "attention-split-residual.png",
        figures_dir / "c500-vector-traffic-ratio-vs-observed-work.png",
        figures_dir / "ascend-910b-moe-l2-ratio-vs-interface-bytes.png",
        figures_dir / "kernel-duration-ratio-vs-observed.png",
        figures_dir / "task-duration-ratio-vs-observed.png",
        figures_dir / "physical-bucket-mape.png",
        figures_dir / "duration-ape-cdf.png",
        figures_dir / "case-impact-ranking.png",
        figures_dir / "c500-silu-mechanism-boundary.png",
    ]
    plot_calls = [
        (_plot_final_latency, (task_rows, figures[0]), figures[0]),
        (_plot_residual_by_family, (task_rows, figures[1]), figures[1]),
        (_plot_residual_vs_scale, (task_rows, figures[2]), figures[2]),
        (_plot_bottlenecks, (kernel_rows, figures[3]), figures[3]),
        (_plot_support_coverage, (task_rows, figures[4]), figures[4]),
        (_plot_vector_variant_residual, (task_rows, figures[5]), figures[5]),
        (_plot_residual_vs_derived_launch_rounds,
         (task_rows, figures[6]), figures[6]),
        (_plot_attention_split_residual, (task_rows, figures[7]), figures[7]),
        (_plot_c500_vector_traffic_ratios,
         (comparisons, figures[8]), figures[8]),
        (_plot_ascend_moe_traffic_ratios, (
            ascend_moe_tile_replay, figures[9]), figures[9]),
        (_plot_duration_ratio, (
            kernel_rows, figures[10], "real_kernel"), figures[10]),
        (_plot_duration_ratio, (
            task_rows, figures[11], "aggregate_task"), figures[11]),
        (_plot_bucket_accuracy, (bucket_rows, figures[12]), figures[12]),
        (_plot_ape_cdf, (kernel_rows, task_rows, figures[13]), figures[13]),
        (_plot_case_impact_ranking,
         (case_impact, figures[14]), figures[14]),
        (_plot_c500_silu_mechanism, (
            c500_silu_mechanism, figures[15]), figures[15]),
    ]
    if before_after_summary is not None:
        if before_after_buckets is None:
            raise AssertionError("before/after buckets were not generated")
        before_after_figures = (
            figures_dir / "before-after-family-mape.png",
            figures_dir / "before-after-bucket-mape.png",
        )
        plot_calls.extend((
            (_plot_before_after_family_accuracy, (
                before_after_summary, before_after_figures[0]),
             before_after_figures[0]),
            (_plot_before_after_bucket_accuracy, (
                before_after_buckets, before_after_figures[1]),
             before_after_figures[1]),
        ))
        figures.extend(before_after_figures)
    with ProcessPoolExecutor(max_workers=min(jobs, len(plot_calls))) as pool:
        rendered = tuple(pool.map(_render_plot_call, plot_calls))
    if rendered != tuple(str(path) for path in figures):
        raise RuntimeError("report figure workers returned out of order")
    import matplotlib

    generated_summary_path = output_dir / "generated_summary.md"
    manifest_path = output_dir / "manifest.json"
    artifact_paths = [
        *raw_paths,
        output_dir / "baseline.csv",
        output_dir / "task_results.csv",
        output_dir / "kernel_results.csv",
        output_dir / "bucket_results.csv",
        output_dir / "severe_duration_cases_over_50pct.csv",
        output_dir / "case_accuracy_violations_over_30pct.csv",
        output_dir / "case_impact_ranking_over_30pct.csv",
        output_dir / "physical_bucket_definitions.json",
        output_dir / "accuracy_gate.json",
        output_dir / "component_comparisons.csv",
        output_dir / "component_comparison_summary.csv",
        output_dir / "counterfactual_results.csv",
        output_dir / "counterfactual_summary.csv",
        output_dir / "evidence_coverage.csv",
        output_dir / "traffic_outliers_over_50pct.csv",
        output_dir / "ascend_910b_moe_tile_replay_check.csv",
        output_dir / "c500_silu_mechanism.csv",
        *figures,
        generated_summary_path,
        manifest_path,
    ]
    if before_after_cases is not None:
        artifact_paths.extend((
            output_dir / "before_after_case_results.csv",
            output_dir / "before_after_bucket_results.csv",
            output_dir / "before_after_summary.csv",
            output_dir / "before_after_join_audit.json",
        ))
    artifact_names = [
        str(path.relative_to(output_dir)) for path in artifact_paths]
    _write_markdown_summary(
        generated_summary_path,
        baseline,
        bucket_rows,
        accuracy_gate,
        comparison_summary,
        case_impact,
        c500_silu_mechanism,
        artifact_names,
        before_after_summary,
        before_after_buckets,
        (
            None if before_results_dir is None
            else str(before_results_dir.relative_to(REPOSITORY_ROOT))
        ),
    )
    artifact_sha256 = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in artifact_paths
        if path != manifest_path
    }
    command = (
        "python scripts/evaluation/kernel/kernel_mechanism_validation_report.py "
        f"--output-dir {output_dir.relative_to(REPOSITORY_ROOT)} --jobs {jobs}"
    )
    if before_results_dir is not None:
        command += (
            " --before-results-dir "
            f"{before_results_dir.relative_to(REPOSITORY_ROOT)}")
    manifest = {
        "schema_version": 8,
        "git_head": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_status_porcelain": _git_output("status", "--short"),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "command": command,
        "hardware_ids": [hardware_id for hardware_id, _ in HARDWARE],
        "dataset_families": list(DATASET_FAMILIES),
        "dataset_identities": dataset_identities,
        "launch_data_identities": launch_data_identities,
        "hardware_config_sha256": config_identities,
        "physical_bucket_schema": {
            "version": BUCKET_SCHEMA_VERSION,
            "definition_sha256": artifact_sha256[
                "physical_bucket_definitions.json"],
        },
        "comparison_baseline": (
            None if before_after_join_audit is None else {
                "results_directory": before_after_join_audit[
                    "before_results_directory"],
                "manifest_sha256": before_after_join_audit[
                    "before_manifest_sha256"],
                "git_head": before_after_join_audit["before_git_head"],
                "forward_diff_field": before_after_join_audit[
                    "before_forward_diff_field"],
                "forward_diff_sha256": before_after_join_audit[
                    "before_forward_diff_sha256"],
            }
        ),
        "report_source_sha256": {
            "dataset_pipeline/kernel_dataset/ascend_910b.py": _sha256(
                REPOSITORY_ROOT
                / "dataset_pipeline/kernel_dataset/ascend_910b.py"),
            "evaluation/kernel/evaluation_runner.py": _sha256(
                REPOSITORY_ROOT / "evaluation/kernel/evaluation_runner.py"),
            "evaluation/kernel/mechanism_validation_report.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/mechanism_validation_report.py"),
            "evaluation/kernel/observation_normalizer.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/observation_normalizer.py"),
            "evaluation/kernel/oracle_observation.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/oracle_observation.py"),
            "evaluation/kernel/profiler_observation.py": _sha256(
                REPOSITORY_ROOT
                / "evaluation/kernel/profiler_observation.py"),
            "scripts/evaluation/kernel/kernel_mechanism_validation_report.py": (
                _sha256(
                    REPOSITORY_ROOT
                    / "scripts/evaluation/kernel/"
                    "kernel_mechanism_validation_report.py")),
            "src/kernel/predictors/fused_moe.py": _sha256(
                REPOSITORY_ROOT / "src/kernel/predictors/fused_moe.py"),
            "src/kernel/predictors/vector.py": _sha256(
                REPOSITORY_ROOT / "src/kernel/predictors/vector.py"),
            "src/kernel/formula.py": _sha256(
                REPOSITORY_ROOT / "src/kernel/formula.py"),
            "src/kernel/launch_config/metax_c500/flash_attn_launch.py": (
                _sha256(
                    REPOSITORY_ROOT
                    / "src/kernel/launch_config/metax_c500/"
                    "flash_attn_launch.py")),
        },
        "forward_source_worktree_diff": _git_output(
            "diff", "--", "src"),
        "forward_source_index_diff": _git_output(
            "diff", "--cached", "--", "src"),
        "forward_source_untracked": _git_output(
            "ls-files", "--others", "--exclude-standard", "--", "src"),
        "hardware_execution_performed": False,
        "profiler_capture_performed": False,
        "artifact_sha256": artifact_sha256,
        "manifest_self_hash": (
            "omitted because a file cannot contain its own stable SHA256"),
        "elapsed_seconds_before_manifest_write": (
            time.perf_counter() - started_at),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "output_dir": str(output_dir),
        "task_count": len(task_rows),
        "kernel_count": len(kernel_rows),
        "bucket_count": len(bucket_rows),
        "accuracy_gate_passed": accuracy_gate["accuracy_gate_passed"],
        "model_closure_gate_passed": accuracy_gate[
            "model_closure_gate_passed"],
        "comparison_count": len(comparisons),
        "counterfactual_count": len(counterfactual_rows),
        "artifacts": artifact_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture and render cross-hardware kernel mechanism evidence "
            "without changing the production Forward model."))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--before-results-dir", type=Path)
    arguments = parser.parse_args()
    result = generate_report(
        arguments.output_dir,
        arguments.jobs,
        arguments.before_results_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
