"""Evaluate recorded Scheduler depth against an explicit depth-one counterfactual."""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
import time
from typing import Callable, Mapping, Sequence
from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.metrics.diagnostics.compact_study_driver import (
    require_positive_jobs,
    require_run_id,
    resolve_experiment_run_directory,
    run_case_jobs,
)
from evaluation.serving.simulation import simulation_runner

from evaluation.json_shape import require_finite_number as _finite_number


_TIMING_MODE = "device_plus_launch_plus_scheduler"
_DEPTH_MODES = (
    ("dataset_native", None),
    ("explicit_counterfactual", 1),
)
_REPORT_METRICS = (
    ("ttft_ms", "mean"),
    ("tpot_ms", "mean"),
    ("e2e_latency_ms", "mean"),
    ("request_throughput_per_s", "scalar"),
    ("output_token_throughput_per_s", "scalar"),
)


def run_scheduler_pipeline_depth_study(
    repository_root: str | Path,
    candidate_root: str | Path,
    run_id: str,
    *,
    jobs: int,
    load_validated_case: Callable[[Path], object],
    source_commit: str | None = None,
) -> dict[str, object]:
    """Run the 45 native and 45 depth-one CPU Forward cells in parallel."""

    root = Path(repository_root).resolve()
    candidate_base = Path(candidate_root).resolve(strict=True)
    require_run_id(run_id)
    require_positive_jobs(jobs)
    suite_runs = compact.official_metrics_suite_runs(root)
    configured_cells = tuple(
        (config, cell)
        for config, cells in suite_runs
        for cell in cells
    )
    if len(configured_cells) != 45:
        raise ValueError("Scheduler depth study requires the 45-case compact matrix")
    resolved_source_commit = (
        compact.repository_source_commit(root)
        if source_commit is None else _source_commit(source_commit))
    profiles = compact.resolve_host_launch_profile_paths(
        root,
        (),
        {cell.hardware.hardware_id for _config, cell in configured_cells},
    )
    if profiles is None:
        raise ValueError("Scheduler depth study requires Host launch profiles")
    expected_hardware_ids = {
        cell.hardware.hardware_id for _config, cell in configured_cells}
    if set(profiles) != expected_hardware_ids:
        raise ValueError("Scheduler depth study Host launch profiles are incomplete")
    case_jobs = []
    for config, cell in configured_cells:
        case_path = _candidate_case_path(candidate_base, cell)
        case = load_validated_case(case_path)
        _validate_candidate_case(config, cell, case)
        configured_depth = compact.scheduler_pipeline_depth_from_validated_case(case)
        if configured_depth != 2:
            raise ValueError(
                f"{cell.cell_id} native Scheduler depth must be 2, got "
                f"{configured_depth}")
        for mode, resolved_depth in _DEPTH_MODES:
            case_jobs.append((
                str(root),
                str(config.source_path),
                str(case_path),
                cell.cell_id,
                mode,
                configured_depth if resolved_depth is None else resolved_depth,
                str(profiles[cell.hardware.hardware_id]),
                resolved_source_commit,
                load_validated_case,
            ))
    output_dir = resolve_experiment_run_directory(
        root,
        Path("experiment_logs/serving/scheduler_depth"),
        run_id,
        label="Scheduler depth output",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = run_case_jobs(case_jobs, jobs, _run_one_case)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 2,
        "kind": "sharegpt_scheduler_pipeline_depth_study",
        "source_commit": resolved_source_commit,
        "candidate_root": str(candidate_base),
        "configurations": [
            {
                "path": str(config.source_path),
                "sha256": config.source_sha256,
                "suite_id": config.suite_id,
            }
            for config, _cells in suite_runs
        ],
        "cell_count": len(results),
        "native_cell_count": sum(
            result["depth_identity"]["mode"] == "dataset_native"
            for result in results
        ),
        "counterfactual_cell_count": sum(
            result["depth_identity"]["mode"] == "explicit_counterfactual"
            for result in results
        ),
        "completed_count": sum(
            result["prediction_status"] == "completed" for result in results),
        "failed_count": sum(
            result["prediction_status"] == "failed" for result in results),
        "jobs": min(jobs, len(case_jobs)),
        "serving_timing_mode": _TIMING_MODE,
        "evaluation_wall_time_seconds": elapsed,
        "cells": results,
    }
    summary.update(_paired_outputs(results))
    paths = _write_results(output_dir, summary)
    return {
        "schema_version": 2,
        "kind": summary["kind"],
        "run_directory": str(output_dir.relative_to(root)),
        "cell_count": summary["cell_count"],
        "completed_count": summary["completed_count"],
        "failed_count": summary["failed_count"],
        "jobs": summary["jobs"],
        "artifact_paths": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
    }


def materialize_scheduler_pipeline_depth_report(
    repository_root: str | Path,
    source_results: str | Path,
    run_id: str,
) -> dict[str, object]:
    """Create the complete paired report from an already executed study."""

    root = Path(repository_root).resolve()
    source_path = Path(source_results).resolve(strict=True)
    require_run_id(run_id)
    payload = json.loads(source_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("source Scheduler depth result must be an object")
    if payload["kind"] != "sharegpt_scheduler_pipeline_depth_study":
        raise ValueError("source result has the wrong study kind")
    cells = payload["cells"]
    if not isinstance(cells, list):
        raise TypeError("source study cells must be a list")
    derived = dict(payload)
    derived["schema_version"] = 2
    derived["report_source_commit"] = compact.repository_source_commit(root)
    derived["source_results"] = {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "declared_simulation_source_commit": payload["source_commit"],
    }
    derived.update(_paired_outputs(cells))
    output_dir = resolve_experiment_run_directory(
        root,
        Path("experiment_logs/serving/scheduler_depth"),
        run_id,
        label="Scheduler depth output",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = _write_results(output_dir, derived)
    return {
        "schema_version": 2,
        "kind": derived["kind"],
        "run_directory": str(output_dir.relative_to(root)),
        "paired_case_count": len(derived["paired_cases"]),
        "failed_count": derived["failed_count"],
        "artifact_paths": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
    }


def _run_one_case(
    repository_root: str,
    config_path: str,
    case_path: str,
    cell_id: str,
    mode: str,
    resolved_depth: int,
    host_profile_path: str,
    source_commit: str,
    load_validated_case: Callable[[Path], object],
) -> dict[str, object]:
    """Execute one fully explicit Forward cell in an isolated process."""

    root = Path(repository_root)
    config = compact.load_phase1_config(config_path)
    cell = compact.select_phase1_cells(config, (cell_id,), all_cases=False)[0]
    case = load_validated_case(Path(case_path))
    runtime = case.manifest["runtime"]
    configured_depth = compact.scheduler_pipeline_depth_from_validated_case(case)
    depth_identity = {
        "configured_depth": configured_depth,
        "resolved_depth": resolved_depth,
        "mode": mode,
        "evidence": runtime["scheduler_pipeline_depth_evidence"],
    }
    argv = compact.build_simulation_argv(
        root,
        config,
        cell,
        case,
        execution_mode="accelerated",
        serving_timing_mode=_TIMING_MODE,
        host_launch_profile_path=host_profile_path,
    )
    _replace_scheduler_pipeline_depth(argv, resolved_depth)
    payload = simulation_runner.run_cli_args(argv)
    provenance = compact._validate_completed_payload(
        root,
        config,
        cell,
        case,
        payload,
        source_commit,
        "accelerated",
        expected_serving_timing_mode=_TIMING_MODE,
    )
    metrics = compact._compare_prediction(
        compact._observed_metrics(case), payload, len(case.requests))
    host_timing = compact._validated_host_timing_payload(
        payload, _TIMING_MODE, cell.hardware.hardware_id)
    return {
        **_cell_identity(cell),
        "prediction_status": "completed",
        "failure": None,
        "depth_identity": depth_identity,
        "serving_timing_mode": _TIMING_MODE,
        "forward_argv_sha256": _canonical_sha256(argv),
        "performance_timing_status": provenance["performance_timing_status"],
        "accuracy_claim_status": provenance["accuracy_claim_status"],
        "metrics": _five_metric_projection(metrics),
        "host_timing_aggregate": host_timing["aggregate"],
    }


def _candidate_case_path(candidate_base: Path, cell: compact.Phase1Cell) -> Path:
    path = (
        candidate_base
        / "sharegpt_scheduler_depth_metadata"
        / cell.hardware.hardware_id
        / cell.model.dataset_model_id
        / cell.case_id
    )
    return path.resolve(strict=True)


def _validate_candidate_case(
    config: compact.Phase1Config,
    cell: compact.Phase1Cell,
    case: object,
) -> None:
    compact._validate_case_binding(config, cell, case)
    if case.manifest["schema_version"] != 5:
        raise ValueError(f"{cell.cell_id} candidate must use compact schema v5")


def _replace_scheduler_pipeline_depth(argv: list[str], depth: int) -> None:
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("resolved Scheduler depth must be a positive integer")
    matches = [
        index for index, value in enumerate(argv)
        if value == "--scheduler-pipeline-depth"
    ]
    if len(matches) != 1:
        raise ValueError("Forward argv must contain exactly one Scheduler depth")
    index = matches[0]
    if index + 1 >= len(argv):
        raise ValueError("Forward argv Scheduler depth has no value")
    argv[index + 1] = str(depth)
    async_flags = [
        index for index, value in enumerate(argv)
        if value in (
            "--vllm-async-scheduling",
            "--no-vllm-async-scheduling",
        )
    ]
    if len(async_flags) != 1:
        raise ValueError(
            "Forward argv must contain exactly one async scheduling flag")
    argv[async_flags[0]] = (
        "--vllm-async-scheduling"
        if depth > 1 else "--no-vllm-async-scheduling"
    )


def _cell_identity(cell: compact.Phase1Cell) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id,
        "case_id": cell.case_id,
        "target_qps": cell.qps,
    }


def _five_metric_projection(metrics: Mapping[str, object]) -> dict[str, object]:
    latency = metrics["latency"]
    throughput = metrics["throughput"]
    if not isinstance(latency, Mapping) or not isinstance(throughput, Mapping):
        raise TypeError("comparison metric families must be objects")
    result = {}
    for metric, statistic in _REPORT_METRICS:
        if statistic == "scalar":
            source = throughput
        elif statistic == "mean":
            source = latency
        else:
            raise ValueError(f"unsupported report statistic: {statistic!r}")
        value = source[metric]
        if statistic != "scalar":
            if not isinstance(value, Mapping):
                raise TypeError(f"latency metric {metric} must be an object")
            value = value[statistic]
        if not isinstance(value, Mapping):
            raise TypeError(f"comparison metric {metric} must be an object")
        result[metric] = dict(value)
    return result


def _write_results(output_dir: Path, summary: Mapping[str, object]) -> dict[str, Path]:
    paths = {
        "json": output_dir / "results.json",
        "csv": output_dir / "results.csv",
        "paired_manifest": output_dir / "paired_case_manifest.json",
        "markdown": output_dir / "report.md",
        "hashes": output_dir / "hashes.json",
    }
    _write_json(paths["json"], summary)
    _write_csv(paths["csv"], summary["paired_cases"])
    _write_json(paths["paired_manifest"], summary["paired_case_manifest"])
    paths["markdown"].write_text(_render_report(summary))
    digests = {
        name: _sha256(path) for name, path in paths.items() if name != "hashes"
    }
    _write_json(paths["hashes"], {"schema_version": 1, "sha256": digests})
    return paths


def _write_csv(path: Path, paired_cases: object) -> None:
    if not isinstance(paired_cases, Sequence):
        raise TypeError("paired study cases must be a sequence")
    fields = (
        "cell_id", "hardware_id", "model_id", "case_id", "target_qps",
        "configured_depth", "metric", "measured", "depth2_prediction",
        "depth1_prediction", "depth2_relative_error_percent",
        "depth1_relative_error_percent", "depth2_absolute_error",
        "depth1_absolute_error", "depth2_minus_depth1_absolute_error",
        "depth2_absolute_percentage_error",
        "depth1_absolute_percentage_error",
        "depth2_minus_depth1_percentage_points", "accuracy_outcome",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in paired_cases:
            if not isinstance(case, Mapping):
                raise TypeError("paired study case must be an object")
            metrics = case["metrics"]
            if not isinstance(metrics, Mapping):
                raise TypeError("paired study metrics must be an object")
            if case["status"] != "completed":
                continue
            for metric, _statistic in _REPORT_METRICS:
                comparison = metrics[metric]
                if not isinstance(comparison, Mapping):
                    raise TypeError("paired metric must be an object")
                writer.writerow({
                    "cell_id": case["cell_id"],
                    "hardware_id": case["hardware_id"],
                    "model_id": case["model_id"],
                    "case_id": case["case_id"],
                    "target_qps": case["target_qps"],
                    "configured_depth": case["configured_depth"],
                    "metric": metric,
                    **comparison,
                })


def _render_report(summary: Mapping[str, object]) -> str:
    aggregate = summary["aggregate"]
    if not isinstance(aggregate, Mapping):
        raise TypeError("study aggregate must be an object")
    lines = [
        "# Scheduler pipeline-depth study",
        "",
        "All cells use `device_plus_launch_plus_scheduler`; native rows consume the recorded Executor capacity and depth-one rows are explicitly labelled counterfactual.",
        "",
        "A negative depth2-minus-depth1 change means the recorded depth=2 prediction is more accurate. The counterfactual is not hardware measurement.",
        "",
    ]
    for scope in (*sorted(aggregate["by_hardware"]), "overall"):
        if scope == "overall":
            metrics = aggregate["overall"]
        elif scope in aggregate["by_hardware"]:
            metrics = aggregate["by_hardware"][scope]
        else:
            raise ValueError(f"unknown aggregate scope: {scope!r}")
        lines.extend([
            f"## {scope}",
            "",
            "| Metric | Depth2 MAPE | Depth1 MAPE | D2-D1 | Relative change | Improved | Worsened | Tied |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for metric, _statistic in _REPORT_METRICS:
            value = metrics[metric]
            mape = value["mean_ape_percent"]
            lines.append(
                f"| {metric} | {mape['depth2']:.2f}% | {mape['depth1']:.2f}% | "
                f"{value['depth2_minus_depth1_percentage_points']:+.2f} pp | "
                f"{value['relative_change_from_depth1_percent']:+.2f}% | "
                f"{value['improved_case_count']} | {value['worsened_case_count']} | "
                f"{value['tied_case_count']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _paired_outputs(cells: Sequence[object]) -> dict[str, object]:
    indexed: dict[tuple[str, str], Mapping[str, object]] = {}
    for value in cells:
        if not isinstance(value, Mapping):
            raise TypeError("study cell must be an object")
        depth = value["depth_identity"]
        if not isinstance(depth, Mapping):
            raise TypeError("study depth identity must be an object")
        key = (str(value["cell_id"]), str(depth["mode"]))
        if key in indexed:
            raise ValueError(f"duplicate Scheduler depth cell {key}")
        indexed[key] = value
    cell_ids = sorted({key[0] for key in indexed})
    paired_cases = [
        _pair_case(
            indexed[(cell_id, "dataset_native")],
            indexed[(cell_id, "explicit_counterfactual")],
        )
        for cell_id in cell_ids
    ]
    if len(paired_cases) != 45 or len(cells) != 90:
        raise ValueError("paired Scheduler depth report requires 45 cases and 90 cells")
    manifest = {
        "schema_version": 1,
        "kind": "sharegpt_scheduler_pipeline_depth_paired_case_manifest",
        "case_count": len(paired_cases),
        "cell_count": len(cells),
        "failed_case_count": sum(case["status"] != "completed" for case in paired_cases),
        "cases": [
            {
                key: case[key]
                for key in (
                    "cell_id", "hardware_id", "model_id", "case_id",
                    "target_qps", "configured_depth", "status",
                )
            }
            for case in paired_cases
        ],
    }
    completed = [case for case in paired_cases if case["status"] == "completed"]
    return {
        "paired_case_manifest": manifest,
        "paired_cases": paired_cases,
        "failed_cases": [
            case["cell_id"] for case in paired_cases
            if case["status"] != "completed"
        ],
        "aggregate": _aggregate_pairs(completed),
    }


def _pair_case(
    native: Mapping[str, object],
    counterfactual: Mapping[str, object],
) -> dict[str, object]:
    for field in ("cell_id", "hardware_id", "model_id", "case_id", "target_qps"):
        if native[field] != counterfactual[field]:
            raise ValueError(f"paired cells disagree on {field}")
    native_depth = native["depth_identity"]
    counterfactual_depth = counterfactual["depth_identity"]
    if not isinstance(native_depth, Mapping) or not isinstance(counterfactual_depth, Mapping):
        raise TypeError("paired depth identities must be objects")
    if native_depth["configured_depth"] != 2 or native_depth["resolved_depth"] != 2:
        raise ValueError("native paired cell must resolve recorded depth 2")
    if counterfactual_depth["configured_depth"] != 2 or counterfactual_depth["resolved_depth"] != 1:
        raise ValueError("counterfactual paired cell must preserve depth 2 and resolve depth 1")
    if native_depth["evidence"] != counterfactual_depth["evidence"]:
        raise ValueError("paired cells disagree on Scheduler depth evidence")
    if native["serving_timing_mode"] != counterfactual["serving_timing_mode"]:
        raise ValueError("paired cells disagree on Serving timing mode")
    status = "completed" if (
        native["prediction_status"] == "completed"
        and counterfactual["prediction_status"] == "completed"
    ) else "failed"
    metrics = {}
    if status == "completed":
        native_metrics = native["metrics"]
        counterfactual_metrics = counterfactual["metrics"]
        if not isinstance(native_metrics, Mapping) or not isinstance(counterfactual_metrics, Mapping):
            raise TypeError("completed paired metrics must be objects")
        for metric, _statistic in _REPORT_METRICS:
            depth2 = native_metrics[metric]
            depth1 = counterfactual_metrics[metric]
            if not isinstance(depth2, Mapping) or not isinstance(depth1, Mapping):
                raise TypeError("paired metric comparisons must be objects")
            measured = _finite_number(depth2["observed"], f"{metric} measured")
            if measured != _finite_number(depth1["observed"], f"{metric} paired measured"):
                raise ValueError(f"paired {metric} measured values differ")
            depth2_error = _finite_number(depth2["error"], f"{metric} depth2 error")
            depth1_error = _finite_number(depth1["error"], f"{metric} depth1 error")
            depth2_ape = _finite_percentage(depth2["absolute_percentage_error"], metric)
            depth1_ape = _finite_percentage(depth1["absolute_percentage_error"], metric)
            delta = depth2_ape - depth1_ape
            metrics[metric] = {
                "measured": measured,
                "depth2_prediction": _finite_number(depth2["predicted"], metric),
                "depth1_prediction": _finite_number(depth1["predicted"], metric),
                "depth2_relative_error_percent": depth2_error / measured * 100.0,
                "depth1_relative_error_percent": depth1_error / measured * 100.0,
                "depth2_absolute_error": abs(depth2_error),
                "depth1_absolute_error": abs(depth1_error),
                "depth2_minus_depth1_absolute_error": abs(depth2_error) - abs(depth1_error),
                "depth2_absolute_percentage_error": depth2_ape,
                "depth1_absolute_percentage_error": depth1_ape,
                "depth2_minus_depth1_percentage_points": delta,
                "accuracy_outcome": "improved" if delta < 0 else "worsened" if delta > 0 else "tied",
            }
    return {
        **{field: native[field] for field in (
            "cell_id", "hardware_id", "model_id", "case_id", "target_qps")},
        "configured_depth": 2,
        "scheduler_pipeline_depth_evidence": native_depth["evidence"],
        "status": status,
        "metrics": metrics,
    }


def _aggregate_pairs(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    hardware_ids = sorted({str(case["hardware_id"]) for case in cases})
    by_hardware = {
        hardware_id: _aggregate_scope([
            case for case in cases if case["hardware_id"] == hardware_id
        ])
        for hardware_id in hardware_ids
    }
    return {"by_hardware": by_hardware, "overall": _aggregate_scope(cases)}


def _aggregate_scope(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result = {}
    for metric, _statistic in _REPORT_METRICS:
        values = [case["metrics"][metric] for case in cases]
        depth2 = fmean(value["depth2_absolute_percentage_error"] for value in values)
        depth1 = fmean(value["depth1_absolute_percentage_error"] for value in values)
        delta = depth2 - depth1
        result[metric] = {
            "case_count": len(values),
            "mean_ape_percent": {"depth2": depth2, "depth1": depth1},
            "depth2_minus_depth1_percentage_points": delta,
            "relative_change_from_depth1_percent": delta / depth1 * 100.0,
            "improved_case_count": sum(value["accuracy_outcome"] == "improved" for value in values),
            "worsened_case_count": sum(value["accuracy_outcome"] == "worsened" for value in values),
            "tied_case_count": sum(value["accuracy_outcome"] == "tied" for value in values),
        }
    return result




def _finite_percentage(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} absolute percentage error must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} absolute percentage error must be finite")
    return result


def _source_commit(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("source_commit must be a full lowercase Git SHA")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False) + "\n")
