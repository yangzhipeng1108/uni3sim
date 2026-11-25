"""Comparison reports for manifest-driven real vLLM experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

from evaluation.serving.experiments.experiment_plan import (
    load_workload_artifact,
    vllm_bench_arrival_process,
    vllm_bench_arrival_times,
)
from evaluation.serving.metrics.latency_distribution import linear_percentile
from evaluation.serving.metrics.sharegpt_evaluation_report import (
    STATISTICS,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
)
from src.serving.backend.vllm import normalize_mtp_scenario_manifest

from evaluation.json_shape import write_json_file as _write_json


class ComparisonError(ValueError):
    pass


LATENCY_METRICS = ("ttft_ms", "tpot_ms", "e2el_ms")
THROUGHPUT_METRICS = (
    "request_throughput_rps",
    "output_token_throughput_tps",
    "total_token_throughput_tps",
)
PD_DIAGNOSTICS = ("prefill_time_ms", "kv_transfer_time_ms", "decode_time_ms")
PROVENANCE_REQUIRED_KEYS = (
    "git_commit",
    "hostname",
    "cuda_driver_summary",
    "python_executable",
    "vllm_version",
    "midlayer_command",
    "environment_overrides",
    "gpu_ids",
    "timestamp_utc",
)


def compare_cell(
    command_manifest_path: str | Path,
    *,
    ground_truth_path: str | Path | None = None,
    prediction_path: str | Path | None = None,
    output_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    diagnostic_mode: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(command_manifest_path)
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ComparisonError("command manifest must be a JSON object")
    outputs = _require_mapping_field(manifest, "outputs", "command manifest")
    ground_truth_value = (
        outputs["ground_truth"] if ground_truth_path is None else ground_truth_path)
    prediction_value = (
        outputs["prediction"] if prediction_path is None else prediction_path)
    output_value = (
        outputs["comparison_report"] if output_path is None else output_path)
    if not ground_truth_value:
        raise ComparisonError("ground truth path is required")
    if not prediction_value:
        raise ComparisonError("prediction path is required")
    if not output_value:
        raise ComparisonError("comparison output path is required")
    ground_truth_path = Path(ground_truth_value)
    prediction_path = Path(prediction_value)
    output_path = Path(output_value)
    markdown_value = (
        outputs["comparison_markdown"] if markdown_path is None else markdown_path)
    markdown_path = Path(markdown_value)

    ground_truth = _load_json(ground_truth_path)
    prediction = _load_json(prediction_path)
    _validate_vllm_prediction_runtime(prediction, manifest)
    _validate_real_vllm_ground_truth(ground_truth, manifest)
    if diagnostic_mode is None:
        if manifest["deployment"] == "disaggregated":
            mode = "pd"
        elif manifest["deployment"] == "colocated":
            mode = "colocated"
        else:
            raise ComparisonError(
                f"unknown deployment: {manifest['deployment']!r}")
    elif diagnostic_mode in ("pd", "colocated"):
        mode = diagnostic_mode
    else:
        raise ComparisonError(f"unknown diagnostic mode: {diagnostic_mode!r}")

    workload = _workload_identity(manifest)
    observed = _ingest_ground_truth(ground_truth)
    predicted = _ingest_prediction(prediction, manifest)
    _validate_identity(workload, observed, predicted)
    _validate_request_counts(workload, observed, predicted)
    _validate_prediction_adequacy(manifest, observed, predicted, mode)

    report = _build_report(
        manifest=manifest,
        manifest_path=manifest_path,
        workload=workload,
        observed=observed,
        predicted=predicted,
        diagnostic_mode=mode,
        ground_truth_path=ground_truth_path,
        prediction_path=prediction_path,
    )
    _write_json(output_path, report)
    _write_text(markdown_path, _render_cell_markdown(report))
    return report


def _validate_vllm_prediction_runtime(
    prediction: Any,
    manifest: dict[str, Any],
) -> None:
    """Reject vLLM predictions without effective scheduler provenance."""

    if "commands" not in manifest:
        raise ComparisonError(
            "command manifest must explicitly identify its prediction producer")
    commands = _require_mapping_field(
        manifest, "commands", "command manifest")
    if "colocated_midlayer_simulation" not in commands:
        recognized = {
            "legacy_prediction",
            "disaggregated_midlayer_simulation",
        }
        if not set(commands) & recognized:
            raise ComparisonError(
                "command manifest has no recognized prediction producer")
        return
    producer = commands["colocated_midlayer_simulation"]
    if not isinstance(producer, dict):
        raise ComparisonError(
            "colocated_midlayer_simulation command must be an object")
    if producer["kind"] != "colocated_midlayer_simulation":
        raise ComparisonError(
            "unsupported colocated prediction producer: "
            f"{producer['kind']!r}")
    if not isinstance(prediction, dict):
        raise ComparisonError(
            "canonical vLLM prediction must be a JSON object")
    if "backend" not in prediction or prediction["backend"] != "vllm":
        raise ComparisonError("canonical prediction backend must be vllm")
    if "version" not in prediction or prediction["version"] != 2:
        raise ComparisonError("vLLM prediction version must be 2")
    if ("workload_hash" not in prediction
            or prediction["workload_hash"] != manifest["workload"]["hash"]):
        raise ComparisonError(
            "vLLM prediction workload_hash disagrees with command manifest")
    if prediction["deployment"] != manifest["deployment"]:
        raise ComparisonError(
            "vLLM prediction deployment disagrees with command manifest")
    if "mtp_scenario" not in prediction:
        raise ComparisonError("vLLM prediction is missing mtp_scenario")
    try:
        mtp_scenario = normalize_mtp_scenario_manifest(
            prediction["mtp_scenario"])
    except (TypeError, ValueError) as error:
        raise ComparisonError(f"invalid prediction MTP scenario: {error}") from error
    if mtp_scenario["validation_status"] != "disabled":
        raise ComparisonError(
            "non-disabled MTP predictions cannot be compared with a real "
            "vLLM benchmark yet; fixed-count projection and acceptance "
            "replay require matching benchmark-side MTP provenance")
    if "mtp_scenario" not in manifest:
        raise ComparisonError("command manifest is missing mtp_scenario")
    try:
        manifest_mtp_scenario = normalize_mtp_scenario_manifest(
            manifest["mtp_scenario"])
        producer_mtp_scenario = normalize_mtp_scenario_manifest(
            producer["mtp_scenario"])
    except (KeyError, TypeError, ValueError) as error:
        raise ComparisonError(
            f"invalid command-manifest MTP scenario: {error}") from error
    if (manifest_mtp_scenario != mtp_scenario
            or producer_mtp_scenario != mtp_scenario):
        raise ComparisonError(
            "prediction, manifest, and simulation MTP scenarios disagree")
    if "scheduler_runtime" not in prediction:
        raise ComparisonError("vLLM prediction is missing scheduler_runtime")
    scheduler_runtime = prediction["scheduler_runtime"]
    if not isinstance(scheduler_runtime, dict) or not scheduler_runtime:
        raise ComparisonError(
            "vLLM prediction scheduler_runtime must be a non-empty object")
    if manifest["deployment"] != "colocated":
        raise ComparisonError(
            "vLLM scheduler runtime validation currently supports only "
            "colocated deployment")
    if set(scheduler_runtime) != {"main"}:
        raise ComparisonError(
            "colocated vLLM prediction scheduler_runtime must contain only "
            "the main engine")
    runtime = scheduler_runtime["main"]
    if not isinstance(runtime, dict):
        raise ComparisonError("vLLM scheduler_runtime.main must be an object")
    required = {
        "version",
        "source_commit",
        "requested_chunked_prefill",
        "effective_chunked_prefill",
        "effective_prefix_caching",
        "prefix_oracle_mode",
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "kv_block_size_tokens",
        "num_kv_blocks",
        "scheduler_policy",
        "num_lookahead_slots",
        "attention_kv_layout",
        "speculative_method",
        "mtp_num_speculative_tokens",
        "mtp_acceptance_policy",
        "mtp_fixed_accepted_tokens",
        "mtp_acceptance_source_id",
        "mtp_padded_drafter_batch",
        "mtp_draft_model_type",
        "mtp_num_mtp_layers",
        "mtp_draft_max_model_len",
    }
    missing = sorted(required - set(runtime))
    if missing:
        raise ComparisonError(
            "vLLM scheduler_runtime.main is missing fields: "
            + ", ".join(missing))
    if runtime["version"] != VLLM_SCHEDULER_VERSION:
        raise ComparisonError(
            "vLLM scheduler runtime version is not the audited version")
    if runtime["source_commit"] != VLLM_SCHEDULER_SOURCE_COMMIT:
        raise ComparisonError(
            "vLLM scheduler runtime source_commit is not audited")
    expected = _require_mapping_field(
        manifest, "parallelism", "colocated command manifest")
    field_pairs = (
        ("max_num_seqs", "max_num_seqs"),
        ("max_num_batched_tokens", "max_num_batched_tokens"),
        ("kv_block_size_tokens", "block_size"),
        ("num_kv_blocks", "num_gpu_blocks_override"),
    )
    for runtime_field, manifest_field in field_pairs:
        if manifest_field not in expected:
            raise ComparisonError(
                "colocated command manifest parallelism is missing "
                f"{manifest_field}")
        if runtime[runtime_field] != expected[manifest_field]:
            raise ComparisonError(
                "vLLM effective scheduler runtime disagrees with the command "
                f"manifest for {manifest_field}: "
                f"runtime={runtime[runtime_field]!r}, "
                f"manifest={expected[manifest_field]!r}")
    invariants = {
        "requested_chunked_prefill": True,
        "effective_chunked_prefill": True,
        "effective_prefix_caching": False,
        "prefix_oracle_mode": "trace_length_private_nonshared_v1",
        "scheduler_policy": "fcfs",
        "num_lookahead_slots": 0,
        "attention_kv_layout": "paged_varlen",
        "speculative_method": "disabled",
        "mtp_num_speculative_tokens": 0,
        "mtp_acceptance_policy": "disabled",
        "mtp_fixed_accepted_tokens": 0,
        "mtp_acceptance_source_id": "disabled",
        "mtp_padded_drafter_batch": False,
        "mtp_draft_model_type": "disabled",
        "mtp_num_mtp_layers": 0,
        "mtp_draft_max_model_len": 0,
    }
    for field, expected_value in invariants.items():
        if runtime[field] != expected_value:
            raise ComparisonError(
                f"vLLM scheduler runtime {field} must be "
                f"{expected_value!r}, got {runtime[field]!r}")

    if "vllm_benchmark" not in commands:
        raise ComparisonError("command manifest is missing vllm_benchmark")
    benchmark = commands["vllm_benchmark"]
    if not isinstance(benchmark, dict):
        raise ComparisonError("vllm_benchmark command must be an object")
    try:
        benchmark_mtp_scenario = normalize_mtp_scenario_manifest(
            benchmark["mtp_scenario"])
    except (KeyError, TypeError, ValueError) as error:
        raise ComparisonError(
            f"invalid benchmark MTP scenario: {error}") from error
    if benchmark_mtp_scenario != mtp_scenario:
        raise ComparisonError(
            "real benchmark and prediction MTP scenarios disagree")
    if "vllm_serve" in commands:
        serve = commands["vllm_serve"]
        if not isinstance(serve, dict):
            raise ComparisonError("vllm_serve command must be an object")
        try:
            serve_mtp_scenario = normalize_mtp_scenario_manifest(
                serve["mtp_scenario"])
        except (KeyError, TypeError, ValueError) as error:
            raise ComparisonError(
                f"invalid vLLM server MTP scenario: {error}") from error
        if serve_mtp_scenario != mtp_scenario:
            raise ComparisonError(
                "real server and prediction MTP scenarios disagree")
    producer_arrival = _require_mapping_field(
        producer, "arrival_process", "simulation command")
    benchmark_arrival = _require_mapping_field(
        benchmark, "arrival_process", "benchmark command")
    if producer_arrival != benchmark_arrival:
        raise ComparisonError(
            "real and simulated arrival-process contracts disagree")
    prediction_arrival = _require_mapping_field(
        prediction, "arrival_process", "vLLM prediction")
    if prediction_arrival != producer_arrival:
        raise ComparisonError(
            "prediction arrival process disagrees with command manifest")
    if prediction_arrival["kind"] != "vllm_bench_gamma":
        raise ComparisonError(
            "canonical vLLM comparison requires vllm_bench_gamma arrivals")
    arrival_required = {
        "request_rate_per_s",
        "burstiness",
        "seed",
        "normalization",
        "vllm_version",
        "vllm_source_commit",
        "source_symbol",
        "source_sha256",
    }
    arrival_missing = sorted(arrival_required - set(prediction_arrival))
    if arrival_missing:
        raise ComparisonError(
            "vLLM arrival-process provenance is missing fields: "
            + ", ".join(arrival_missing))
    expected_arrival_contract = vllm_bench_arrival_process(
        request_rate=prediction_arrival["request_rate_per_s"],
        seed=prediction_arrival["seed"],
    )
    if prediction_arrival != expected_arrival_contract:
        raise ComparisonError(
            "vLLM arrival-process provenance is not the audited contract")
    expected_arrivals = vllm_bench_arrival_times(
        num_requests=manifest["workload"]["num_requests"],
        request_rate=prediction_arrival["request_rate_per_s"],
        burstiness=prediction_arrival["burstiness"],
        seed=prediction_arrival["seed"],
    )
    if "request_arrivals_s" not in prediction:
        raise ComparisonError(
            "vLLM prediction is missing request_arrivals_s")
    request_arrivals = prediction["request_arrivals_s"]
    if not isinstance(request_arrivals, list):
        raise ComparisonError(
            "vLLM prediction request_arrivals_s must be a list")
    if len(request_arrivals) != len(expected_arrivals):
        raise ComparisonError(
            "vLLM prediction request-arrival count disagrees with workload")
    expected_request_ids = benchmark["vllm_consumed_request_ids"]
    if (not isinstance(expected_request_ids, list)
            or len(expected_request_ids) != len(expected_arrivals)
            or not all(
                isinstance(request_id, str) and request_id
                for request_id in expected_request_ids)):
        raise ComparisonError(
            "vllm_benchmark command has invalid consumed request IDs")
    actual_request_ids = []
    actual_arrivals = []
    for index, entry in enumerate(request_arrivals):
        if not isinstance(entry, dict) or set(entry) != {
                "request_id", "arrival_time_s"}:
            raise ComparisonError(
                f"vLLM prediction request_arrivals_s[{index}] must contain "
                "request_id and arrival_time_s")
        actual_request_ids.append(entry["request_id"])
        arrival_time = _require_number(
            entry["arrival_time_s"],
            f"request_arrivals_s[{index}].arrival_time_s",
        )
        if not math.isfinite(arrival_time) or arrival_time < 0:
            raise ComparisonError(
                "vLLM prediction request arrival times must be finite and "
                "non-negative")
        actual_arrivals.append(arrival_time)
    if actual_request_ids != expected_request_ids:
        raise ComparisonError(
            "vLLM prediction request-arrival IDs disagree with workload order")
    if tuple(actual_arrivals) != expected_arrivals:
        raise ComparisonError(
            "vLLM prediction request arrivals disagree with the pinned "
            "benchmark generator")


def _validate_real_vllm_ground_truth(
    ground_truth: Any,
    manifest: dict[str, Any],
) -> None:
    if "commands" not in manifest:
        return
    commands = _require_mapping_field(
        manifest, "commands", "command manifest")
    if "colocated_midlayer_simulation" not in commands:
        return
    if not isinstance(ground_truth, dict):
        raise ComparisonError(
            "canonical vLLM ground truth must be a JSON object")
    benchmark = commands["vllm_benchmark"]
    if not isinstance(benchmark, dict):
        raise ComparisonError("vllm_benchmark command must be an object")
    identity = _require_mapping_field(
        benchmark, "validation_identity", "vllm benchmark command")
    required_identity = {
        "workload_hash",
        "request_order_sha256",
        "arrival_contract_sha256",
        "scheduler_contract_sha256",
    }
    if set(identity) != required_identity:
        raise ComparisonError(
            "vLLM benchmark validation identity must contain exactly "
            f"{sorted(required_identity)}")
    for key in sorted(required_identity):
        if key not in ground_truth or ground_truth[key] != identity[key]:
            raise ComparisonError(
                "real vLLM ground truth does not carry the declared "
                f"invocation identity {key}")

    arrival = _require_mapping_field(
        benchmark, "arrival_process", "vllm benchmark command")
    traffic = {
        "request_rate": arrival["request_rate_per_s"],
        "burstiness": arrival["burstiness"],
        "num_prompts": manifest["workload"]["num_requests"],
    }
    for key, expected in traffic.items():
        if key not in ground_truth or ground_truth[key] != expected:
            raise ComparisonError(
                f"real vLLM ground truth {key} disagrees with command "
                f"manifest: expected={expected!r}")

    workload_path = manifest["workload"]["path"]
    artifact = load_workload_artifact(workload_path)
    if artifact.hash != identity["workload_hash"]:
        raise ComparisonError(
            "command workload artifact hash disagrees with validation identity")
    consumed_request_ids = benchmark["vllm_consumed_request_ids"]
    if consumed_request_ids != artifact.request_ids:
        raise ComparisonError(
            "vllm benchmark request IDs disagree with workload artifact order")
    request_order_sha256 = hashlib.sha256(
        json.dumps(
            artifact.request_ids,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if request_order_sha256 != identity["request_order_sha256"]:
        raise ComparisonError(
            "vllm benchmark request-order hash disagrees with workload artifact")
    details = {
        "input_lens": artifact.prompt_lengths,
        "output_lens": [
            artifact.token_controls["max_tokens"]
            for _ in artifact.request_ids
        ],
    }
    for key, expected in details.items():
        if key not in ground_truth or ground_truth[key] != expected:
            raise ComparisonError(
                f"real vLLM ground truth {key} does not preserve workload "
                "request order and token lengths")
    if "errors" not in ground_truth:
        raise ComparisonError(
            "real vLLM ground truth is missing detailed request errors")
    errors = ground_truth["errors"]
    if (not isinstance(errors, list)
            or len(errors) != artifact.num_requests
            or any(error not in (None, "") for error in errors)):
        raise ComparisonError(
            "real vLLM ground truth contains failed requests")


def aggregate_run_directory(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    run_root = Path(run_dir)
    output_root = Path(output_dir) if output_dir is not None else run_root / "aggregate"
    reports = [
        _load_json(path)
        for path in sorted(run_root.glob("*/comparison_report.json"))
    ]
    for report in reports:
        _validate_report_provenance(report)
    rows = [_aggregate_row(report, metric) for report in reports for metric in report["metrics"]]
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "aggregate_report.json"
    jsonl_path = output_root / "aggregate_report.jsonl"
    csv_path = output_root / "aggregate_report.csv"
    md_path = output_root / "aggregate_report.md"
    top_gaps = _top_modeling_gaps(rows)
    grouped = _group_rows(rows)
    _write_json(json_path, {
        "version": 1,
        "kind": "real_vllm_aggregate_report",
        "run_dir": str(run_root),
        "groups": grouped,
        "top_modeling_gaps": top_gaps,
    })
    _write_jsonl(jsonl_path, rows)
    _write_csv(csv_path, rows)
    _write_text(md_path, _render_aggregate_markdown(run_root, grouped, top_gaps))
    return {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
    }


def generate_final_report(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    run_root = Path(run_dir)
    output_root = Path(output_dir) if output_dir is not None else run_root / "final"
    cells = _final_report_cells(run_root)
    top_gaps = _final_top_modeling_gaps(cells)
    report = {
        "version": 1,
        "kind": "real_vllm_final_report",
        "run_dir": str(run_root),
        "measurement_policy": "metric errors are observations, not hard pass/fail thresholds",
        "cells": cells,
        "conclusions": _final_conclusions(cells),
        "top_modeling_gaps": top_gaps,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "final_report.json"
    md_path = output_root / "final_report.md"
    _write_json(json_path, report)
    _write_text(md_path, _render_final_markdown(report))
    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }


def _final_report_cells(run_root: Path) -> list[dict[str, Any]]:
    run_summary = _load_json(run_root / "run_summary.json")
    if not isinstance(run_summary, dict) or "cells" not in run_summary:
        raise ComparisonError("run_summary.json must be an object with cells")
    raw_cells = run_summary["cells"]
    if not isinstance(raw_cells, list):
        raise ComparisonError("run_summary.json must contain cells")
    cells = []
    for summary_cell in raw_cells:
        if not isinstance(summary_cell, dict):
            raise ComparisonError("run summary cells must be objects")
        manifest_path = Path(summary_cell["command_manifest"])
        if not manifest_path.is_absolute():
            manifest_path = run_root / summary_cell["cell_id"] / "command_manifest.json"
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ComparisonError(f"command manifest must be an object: {manifest_path}")
        _validate_final_manifest_provenance(manifest)
        outputs = _require_mapping_field(manifest, "outputs", "command manifest")
        status_path = Path(outputs["status"])
        if not status_path.exists():
            raise ComparisonError(
                f"experiment is missing status artifact: {status_path}")
        status = _load_json(status_path)
        if not isinstance(status, dict):
            raise ComparisonError(f"status must be an object: {status_path}")
        comparison_path = Path(outputs["comparison_report"])
        comparison = (
            _load_json(comparison_path)
            if comparison_path.exists()
            else None
        )
        cell_status = _consistent_cell_status(status, summary_cell, manifest)
        if cell_status == "success":
            _validate_success_comparison_artifact(
                manifest["cell_id"],
                comparison_path,
                comparison,
            )
        cells.append(_final_cell(summary_cell, manifest_path, manifest, status, comparison))
    return cells


def _validate_success_comparison_artifact(
    cell_id: str | None,
    comparison_path: Path | None,
    comparison: dict[str, Any] | None,
) -> None:
    if cell_id is None:
        label = "<unknown>"
    elif isinstance(cell_id, str) and cell_id:
        label = cell_id
    else:
        raise ComparisonError("cell_id must be a non-empty string or null")
    if comparison_path is None or not comparison_path.exists():
        raise ComparisonError(
            f"success cell {label} missing comparison report artifact: {comparison_path}")
    if not isinstance(comparison, dict):
        raise ComparisonError(
            f"success cell {label} comparison report is not a JSON object: {comparison_path}")
    kind = comparison["kind"]
    if kind != "real_vllm_comparison_report":
        raise ComparisonError(
            f"success cell {label} has unsupported comparison report kind {kind}: {comparison_path}")
    metrics = comparison["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ComparisonError(
            f"success cell {label} comparison report has no metric evidence: {comparison_path}")
    if not any(_final_metric_has_evidence(metric) for metric in metrics):
        raise ComparisonError(
            f"success cell {label} comparison report has no metric evidence: {comparison_path}")


def _final_metric_has_evidence(metric: Any) -> bool:
    if not isinstance(metric, dict):
        raise ComparisonError("comparison metric must be an object")
    required = {"absolute_percentage_error", "observed", "predicted"}
    if not required.issubset(metric):
        raise ComparisonError(
            f"comparison metric missing evidence fields: {sorted(required - set(metric))}")
    if _number_or_none(metric["absolute_percentage_error"]) is not None:
        return True
    return (
        _number_or_none(metric["observed"]) is not None
        and _number_or_none(metric["predicted"]) is not None
    )


def _validate_final_manifest_provenance(manifest: dict[str, Any]) -> None:
    provenance = manifest["provenance"]
    if not isinstance(provenance, dict):
        raise ComparisonError(
            f"command manifest for {manifest['cell_id']} missing provenance")
    missing = [
        key for key in (*PROVENANCE_REQUIRED_KEYS, "git_dirty", "git_status_short", "git_diff_hash")
        if key not in provenance or provenance[key] is None
    ]
    if missing:
        raise ComparisonError(
            f"command manifest for {manifest['cell_id']} has incomplete provenance: "
            + ", ".join(missing))


def _final_cell(
    summary_cell: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    cell_status = _consistent_cell_status(status, summary_cell, manifest)
    deployment = manifest["deployment"]
    model_type = manifest["model_type"]
    confidence = _confidence_label(cell_status, comparison)
    if cell_status == "blocked":
        known_blocker = _known_blocker(status)
    elif cell_status in ("success", "failed", "dry_run", "running"):
        known_blocker = None
    else:
        raise ComparisonError(f"unknown experiment status: {cell_status!r}")
    metrics = comparison["metrics"] if comparison is not None else []
    outputs = _require_mapping_field(manifest, "outputs", "command manifest")
    return {
        "cell_id": manifest["cell_id"],
        "status": cell_status,
        "confidence": confidence,
        "model": manifest["model"],
        "model_type": model_type,
        "deployment": deployment,
        "component_notes": _component_notes(manifest, status),
        "known_blocker": known_blocker,
        "metrics": metrics,
        "artifacts": {
            "command_manifest": str(manifest_path),
            "status": str(Path(manifest["output_root"]) / "status.json"),
            "comparison_report": outputs["comparison_report"],
            "logs": status["logs"],
        },
        "provenance": manifest["provenance"],
    }


def _confidence_label(status: str | None, comparison: dict[str, Any] | None) -> str:
    if status == "success" and comparison is not None:
        return "measured"
    elif status == "blocked":
        return "blocked-runtime"
    elif status == "failed":
        return "failed-run"
    elif status in (None, "dry_run", "running"):
        return "unverified"
    else:
        raise ComparisonError(f"unknown experiment status: {status!r}")


def _component_notes(manifest: dict[str, Any], status: dict[str, Any]) -> list[str]:
    notes = []
    deployment = manifest["deployment"]
    model_type = manifest["model_type"]
    if deployment == "disaggregated":
        notes.append("PD disaggregated serving is evaluated independently from colocated serving.")
        kv_connector = manifest["kv_connector"]
        if not isinstance(kv_connector, str) or not kv_connector:
            raise ComparisonError(
                "disaggregated command manifest requires kv_connector")
        notes.append(f"KV connector: {kv_connector}")
    elif deployment == "colocated" and model_type == "moe":
        notes.append("MoE colocated serving is evaluated independently from dense colocated serving.")
        _require_mapping_field(manifest, "moe", "command manifest")
    elif deployment == "colocated" and model_type == "dense":
        notes.append("Dense colocated serving baseline.")
    else:
        raise ComparisonError(
            f"unknown deployment/model_type: {deployment!r}/{model_type!r}")
    if status["status"] == "blocked":
        notes.append("Runtime blocker excluded from midlayer modeling-error ranking.")
    return notes


def _known_blocker(status: dict[str, Any]) -> dict[str, Any]:
    reproducers = status["reproducer_commands"]
    if not isinstance(reproducers, dict) or not reproducers:
        raise ComparisonError(
            "blocked status requires non-empty reproducer_commands")
    reproducer_refs = sorted(reproducers)
    return {
        "dependency_category": status["dependency_category"],
        "failed_stage": status["failed_stage"],
        "stderr_summary": status["stderr_summary"],
        "logs": status["logs"],
        "unblock_checklist": status["unblock_checklist"],
        "reproducer_refs": reproducer_refs,
    }


def _final_top_modeling_gaps(cells: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for cell in cells:
        if cell["status"] != "success":
            continue
        for metric in cell["metrics"]:
            if metric["informational"] is True:
                continue
            if metric["absolute_percentage_error"] is None:
                continue
            rows.append({
                "cell_id": cell["cell_id"],
                "metric": metric["name"],
                "statistic": metric["statistic"],
                "absolute_percentage_error": metric["absolute_percentage_error"],
                "diagnostic_family": metric["diagnostic_family"],
            })
    rows = sorted(rows, key=lambda row: row["absolute_percentage_error"], reverse=True)[:3]
    if rows:
        return {
            "evidence_backed": True,
            "items": rows,
            "message": "Top modeling gaps are ranked from successful comparison artifacts.",
            "current_unblock_items": [],
        }
    blockers = [
        {
            "cell_id": cell["cell_id"],
            "dependency_category": cell["known_blocker"]["dependency_category"],
            "failed_stage": cell["known_blocker"]["failed_stage"],
        }
        for cell in cells
        if cell["known_blocker"] is not None
    ]
    return {
        "evidence_backed": False,
        "items": [],
        "message": "Modeling-gap ranking is not yet evidence-backed by successful comparison artifacts.",
        "current_unblock_items": blockers,
    }


def _final_conclusions(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dense_colocated": _deployment_conclusion(
            cells,
            "Dense Colocated",
            lambda cell: cell["deployment"] == "colocated" and cell["model_type"] == "dense",
            "Dense colocated evidence is not generalized to MoE or PD.",
        ),
        "moe_colocated": _deployment_conclusion(
            cells,
            "MoE Colocated",
            lambda cell: cell["deployment"] == "colocated" and cell["model_type"] == "moe",
            "MoE requires separate EP/routing evidence.",
        ),
        "pd_disaggregated": _deployment_conclusion(
            cells,
            "PD Disaggregated",
            lambda cell: cell["deployment"] == "disaggregated",
            "PD requires separate TTFT/TPOT/KV-transfer evidence.",
        ),
    }


def _deployment_conclusion(
    cells: list[dict[str, Any]],
    label: str,
    predicate,
    scope_note: str,
) -> dict[str, Any]:
    selected = [cell for cell in cells if predicate(cell)]
    measured = [cell for cell in selected if cell["confidence"] == "measured"]
    blocked = [cell for cell in selected if cell["confidence"] == "blocked-runtime"]
    if measured:
        summary = f"{label} has measured comparison artifacts. {scope_note}"
    elif blocked:
        summary = f"{label} is currently blocked by runtime dependencies. {scope_note}"
    else:
        summary = f"{label} has no accepted cells in this run. {scope_note}"
    return {
        "cell_ids": [cell["cell_id"] for cell in selected],
        "measured": [cell["cell_id"] for cell in measured],
        "blocked": [cell["cell_id"] for cell in blocked],
        "summary": summary,
    }


def _render_final_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Real vLLM Final Modeling Report",
        "",
        f"Run directory: `{report['run_dir']}`",
        "",
        f"Measurement policy: {report['measurement_policy']}.",
        "",
        "## Cell Status",
        "",
        "| Cell | Status | Confidence | Deployment | Model Type | Blocker |",
        "|---|---|---|---|---|---|",
    ]
    for cell in report["cells"]:
        blocker = cell["known_blocker"]
        blocker_text = "" if blocker is None else blocker["dependency_category"]
        lines.append(
            f"| {cell['cell_id']} | {cell['status']} | {cell['confidence']} | "
            f"{cell['deployment']} | {cell['model_type']} | {blocker_text} |"
        )
    lines.extend(["", "## Independent Conclusions"])
    for key, title in (
        ("dense_colocated", "Dense Colocated"),
        ("moe_colocated", "MoE Colocated"),
        ("pd_disaggregated", "PD Disaggregated"),
    ):
        conclusion = report["conclusions"][key]
        lines.extend(["", f"### {title}", conclusion["summary"]])
    lines.extend(["", "## Top Modeling Gaps"])
    gaps = report["top_modeling_gaps"]
    lines.append(gaps["message"])
    for item in gaps["items"]:
        lines.append(
            f"- `{item['cell_id']}` {item['metric']} {item['statistic']}: "
            f"APE={item['absolute_percentage_error']:.6g}%"
        )
    if gaps["current_unblock_items"]:
        lines.append("")
        lines.append("Current unblock items:")
        for item in gaps["current_unblock_items"]:
            lines.append(
                f"- `{item['cell_id']}` {item['failed_stage']} "
                f"({item['dependency_category']})"
            )
    lines.extend(["", "## Blockers"])
    for cell in report["cells"]:
        blocker = cell["known_blocker"]
        if not blocker:
            continue
        lines.append(
            f"- `{cell['cell_id']}` failed at `{blocker['failed_stage']}` "
            f"with `{blocker['dependency_category']}`; logs: `{blocker['logs']}`"
        )
    return "\n".join(lines) + "\n"


def _workload_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    workload = _require_mapping_field(manifest, "workload", "command manifest")
    if "hash" not in workload or not workload["hash"]:
        raise ComparisonError("command manifest workload hash is required")
    token_controls = _require_mapping_field(
        manifest, "token_controls", "command manifest")
    return {
        "hash": workload["hash"],
        "num_requests": _require_int(workload["num_requests"], "workload.num_requests"),
        "requested_num_requests": workload["requested_num_requests"],
        "max_tokens": _require_int(
            token_controls["max_tokens"], "token_controls.max_tokens"),
    }


def _ingest_ground_truth(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ComparisonError("ground truth must be a JSON object")
    data = dict(data)
    for field in ("metrics", "total_output_tokens", "failed_requests"):
        if field not in data:
            data[field] = None
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    conversions: dict[tuple[str, str], str] = {}
    _collect_explicit_metric_groups(
        data["metrics"], metrics, conversions)
    for canonical, aliases in (
        ("ttft_ms", ("ttft",)),
        ("tpot_ms", ("tpot", "itl")),
        ("e2el_ms", ("e2el", "e2e")),
    ):
        for alias in aliases:
            _collect_latency_summaries(data, alias, canonical, metrics, conversions)
    for source, canonical, unit in (
        ("request_throughput", "request_throughput_rps", "requests/s"),
        ("request_throughput_rps", "request_throughput_rps", "requests/s"),
        ("output_throughput", "output_token_throughput_tps", "tokens/s"),
        ("output_token_throughput_tps", "output_token_throughput_tps", "tokens/s"),
        ("total_token_throughput", "total_token_throughput_tps", "tokens/s"),
        ("total_token_throughput_tps", "total_token_throughput_tps", "tokens/s"),
    ):
        _add_optional_scalar(data, source, metrics, canonical, "value", unit)
    if "total_token_throughput_tps" not in metrics and "duration" in data:
        duration = _require_number(data["duration"], "duration")
        if duration <= 0:
            raise ComparisonError("duration must be positive")
        total_input = _require_number(
            data["total_input_tokens"], "total_input_tokens")
        total_output = _require_number(
            data["total_output_tokens"], "total_output_tokens")
        _add_scalar(
            metrics, "total_token_throughput_tps", "value",
            (total_input + total_output) / duration, "tokens/s")
    return {
        "workload_hash": _extract_workload_hash(data),
        "completed": _request_count(data),
        "total_output_tokens": data["total_output_tokens"],
        "failed_requests": data["failed_requests"],
        "metrics": metrics,
        "unit_conversions": conversions,
    }


def _ingest_prediction(data: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    rows = _prediction_rows(data)
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    conversions: dict[tuple[str, str], str] = {}
    wrapper_hash = _extract_workload_hash(data) if isinstance(data, dict) else None
    scheduler_runtime = None
    arrival_process = None
    mtp_scenario = None
    if isinstance(data, dict):
        if "scheduler_runtime" in data:
            scheduler_runtime = data["scheduler_runtime"]
        if "arrival_process" in data:
            arrival_process = data["arrival_process"]
        if "mtp_scenario" in data:
            mtp_scenario = data["mtp_scenario"]
        simulation_metrics = data["metrics"]
        if isinstance(simulation_metrics, dict) and "requests" in simulation_metrics:
            for source, canonical, unit in (
                ("request_throughput_per_s", "request_throughput_rps", "requests/s"),
                ("output_token_throughput_per_s", "output_token_throughput_tps", "tokens/s"),
                ("token_throughput_per_s", "total_token_throughput_tps", "tokens/s"),
            ):
                _add_scalar(
                    metrics, canonical, "value", simulation_metrics[source], unit)
        elif isinstance(simulation_metrics, dict):
            _collect_explicit_metric_groups(
                simulation_metrics, metrics, conversions)
        elif not isinstance(simulation_metrics, list):
            raise ComparisonError("prediction metrics must be an object or list")
        _collect_latency_summaries(data, "ttft", "ttft_ms", metrics, conversions)
        _collect_latency_summaries(data, "tpot", "tpot_ms", metrics, conversions)
        _collect_latency_summaries(data, "itl", "tpot_ms", metrics, conversions)
        _collect_latency_summaries(data, "e2e", "e2el_ms", metrics, conversions)
        _collect_latency_summaries(data, "e2el", "e2el_ms", metrics, conversions)
        for source, canonical, unit in (
            ("request_throughput", "request_throughput_rps", "requests/s"),
            ("request_throughput_rps", "request_throughput_rps", "requests/s"),
            ("output_throughput", "output_token_throughput_tps", "tokens/s"),
            ("output_token_throughput_tps", "output_token_throughput_tps", "tokens/s"),
            ("total_token_throughput", "total_token_throughput_tps", "tokens/s"),
            ("total_token_throughput_tps", "total_token_throughput_tps", "tokens/s"),
        ):
            _add_optional_scalar(data, source, metrics, canonical, "value", unit)

    per_request_metrics = _per_request_metrics(rows)
    for metric_name, stats in per_request_metrics.items():
        if metric_name not in metrics:
            metrics[metric_name] = {}
        metrics[metric_name].update(stats)
    _derive_prediction_throughput(metrics, rows)
    return {
        "workload_hash": wrapper_hash,
        "completed": len(rows) if rows is not None else None,
        "metrics": metrics,
        "unit_conversions": conversions,
        "scheduler_runtime": scheduler_runtime,
        "arrival_process": arrival_process,
        "mtp_scenario": mtp_scenario,
    }


def _prediction_rows(data: Any) -> list[dict[str, Any]] | None:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if "metrics" not in data:
            raise ComparisonError(
                "prediction object must contain an explicit metrics object or list")
        raw_metrics = data["metrics"]
        if isinstance(raw_metrics, list):
            rows = raw_metrics
        elif isinstance(raw_metrics, dict) and "requests" in raw_metrics:
            requests = raw_metrics["requests"]
            if not isinstance(requests, dict):
                raise ComparisonError("prediction metrics.requests must be an object")
            rows = _simulation_request_rows(requests)
        elif isinstance(raw_metrics, dict):
            return None
        else:
            raise ComparisonError("prediction metrics must be an object or list")
    else:
        raise ComparisonError("prediction must be a JSON object or list")
    if not all(isinstance(row, dict) for row in rows):
        raise ComparisonError("prediction rows must be JSON objects")
    return rows


def _simulation_request_rows(
    requests: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for request_id, request_metrics in requests.items():
        if not isinstance(request_id, str) or not request_id:
            raise ComparisonError("simulation request IDs must be non-empty strings")
        if not isinstance(request_metrics, dict):
            raise ComparisonError(
                f"metrics.requests.{request_id} must be an object")
        required = {"ttft_s", "tpot_s", "latency_s"}
        if set(request_metrics) != required:
            raise ComparisonError(
                f"metrics.requests.{request_id} must contain exactly "
                f"{sorted(required)}")
        rows.append({
            "request_id": request_id,
            "ttft_ms": _require_number(
                request_metrics["ttft_s"],
                f"metrics.requests.{request_id}.ttft_s") * 1000.0,
            "tpot_ms": _require_number(
                request_metrics["tpot_s"],
                f"metrics.requests.{request_id}.tpot_s") * 1000.0,
            "e2e_ms": _require_number(
                request_metrics["latency_s"],
                f"metrics.requests.{request_id}.latency_s") * 1000.0,
        })
    return rows


def _per_request_metrics(rows: list[dict[str, Any]] | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not rows:
        return {}
    values: dict[str, list[float]] = {}
    for row in rows:
        for source, canonical in (
            ("ttft_ms", "ttft_ms"),
            ("ttft_disagg_ms", "ttft_ms"),
            ("tpot_ms", "tpot_ms"),
            ("tpot_disagg_ms", "tpot_ms"),
            ("e2e_ms", "e2el_ms"),
            ("e2el_ms", "e2el_ms"),
            ("e2e_disagg_ms", "e2el_ms"),
            ("prefill_time_ms", "prefill_time_ms"),
            ("kv_transfer_time_ms", "kv_transfer_time_ms"),
            ("decode_time_ms", "decode_time_ms"),
        ):
            if source in row:
                value = _require_number(row[source], source)
                if canonical not in values:
                    values[canonical] = []
                values[canonical].append(value)
    return {
        name: {
            stat: {"value": value, "unit": "ms"}
            for stat, value in _stats(metric_values).items()
        }
        for name, metric_values in values.items()
    }


def _derive_prediction_throughput(
    metrics: dict[str, dict[str, dict[str, Any]]],
    rows: list[dict[str, Any]] | None,
) -> None:
    if not rows:
        return
    timing_present = [
        "start_time_ms" in row and "end_time_ms" in row for row in rows]
    if not any(timing_present):
        return
    if not all(timing_present):
        raise ComparisonError("prediction timing fields are incomplete across requests")
    start_values = [
        _require_number(row["start_time_ms"], "start_time_ms") for row in rows]
    end_values = [
        _require_number(row["end_time_ms"], "end_time_ms") for row in rows]
    duration_s = (max(end_values) - min(start_values)) / 1000.0
    if duration_s <= 0:
        raise ComparisonError("prediction request time span must be positive")
    token_fields_present = [
        "num_decode_tokens" in row and "num_prefill_tokens" in row for row in rows]
    if not all(token_fields_present):
        raise ComparisonError("prediction token counts are incomplete across requests")
    output_tokens = sum(
        _require_number(row["num_decode_tokens"], "num_decode_tokens") for row in rows)
    prefill_tokens = sum(
        _require_number(row["num_prefill_tokens"], "num_prefill_tokens") for row in rows)
    _add_scalar(metrics, "request_throughput_rps", "value", len(rows) / duration_s, "requests/s")
    if output_tokens:
        _add_scalar(metrics, "output_token_throughput_tps", "value", output_tokens / duration_s, "tokens/s")
    if output_tokens or prefill_tokens:
        _add_scalar(metrics, "total_token_throughput_tps", "value", (output_tokens + prefill_tokens) / duration_s, "tokens/s")


def _collect_explicit_metric_groups(
    raw: Any,
    metrics: dict[str, dict[str, dict[str, Any]]],
    conversions: dict[tuple[str, str], str],
) -> None:
    if raw is None:
        return
    if not isinstance(raw, dict):
        raise ComparisonError("metrics must be an object")
    for metric_name, group in raw.items():
        canonical = _canonical_metric_name(metric_name)
        if not isinstance(group, dict):
            raise ComparisonError(f"metric group {metric_name!r} must be an object")
        for statistic, payload in group.items():
            if statistic not in (*STATISTICS, "value"):
                raise ComparisonError(
                    f"unknown statistic {statistic!r} for metric {metric_name!r}")
            if isinstance(payload, dict):
                if set(payload) != {"value", "unit"}:
                    raise ComparisonError(
                        f"metric payload {metric_name}.{statistic} must contain value/unit")
                value = payload["value"]
                unit = payload["unit"]
            else:
                value = payload
                unit = _default_unit(canonical)
            _store_metric(metrics, conversions, canonical, statistic, value, unit)


def _collect_latency_summaries(
    data: dict[str, Any],
    alias: str,
    canonical: str,
    metrics: dict[str, dict[str, dict[str, Any]]],
    conversions: dict[tuple[str, str], str],
) -> None:
    for statistic, prefixes in (
        ("mean", ("mean", "avg", "average")),
        ("p50", ("median", "p50")),
        ("p90", ("p90",)),
        ("p99", ("p99",)),
    ):
        for prefix in prefixes:
            for suffix, unit in (("_ms", "ms"), ("_ns", "ns")):
                key = f"{prefix}_{alias}{suffix}"
                if key in data:
                    _store_metric(metrics, conversions, canonical, statistic, data[key], unit)
                    break


def _store_metric(
    metrics: dict[str, dict[str, dict[str, Any]]],
    conversions: dict[tuple[str, str], str],
    canonical: str,
    statistic: str,
    value: Any,
    unit: str | None,
) -> None:
    number = _require_number(value, f"{canonical}.{statistic}")
    normalized, normalized_unit, conversion = _normalize_unit(canonical, number, unit)
    if canonical not in metrics:
        metrics[canonical] = {}
    if statistic in metrics[canonical]:
        raise ComparisonError(f"duplicate metric {canonical}.{statistic}")
    metrics[canonical][statistic] = {
        "value": normalized,
        "unit": normalized_unit,
    }
    if conversion:
        conversions[(canonical, statistic)] = conversion


def _add_scalar(
    metrics: dict[str, dict[str, dict[str, Any]]],
    metric_name: str,
    statistic: str,
    value: Any,
    unit: str,
) -> None:
    number = _require_number(value, f"{metric_name}.{statistic}")
    if metric_name not in metrics:
        metrics[metric_name] = {}
    if statistic in metrics[metric_name]:
        existing = metrics[metric_name][statistic]
        if existing != {"value": number, "unit": unit}:
            raise ComparisonError(
                f"conflicting metric values for {metric_name}.{statistic}")
        return
    metrics[metric_name][statistic] = {"value": number, "unit": unit}


def _add_optional_scalar(
    data: dict[str, Any],
    source: str,
    metrics: dict[str, dict[str, dict[str, Any]]],
    metric_name: str,
    statistic: str,
    unit: str,
) -> None:
    if source in data:
        _add_scalar(metrics, metric_name, statistic, data[source], unit)


def _normalize_unit(metric_name: str, value: float, unit: str | None) -> tuple[float, str, str | None]:
    expected = _default_unit(metric_name)
    if unit is None:
        return value, expected, None
    if metric_name.endswith("_ms") or metric_name in PD_DIAGNOSTICS:
        if unit == "ms":
            return value, "ms", None
        if unit == "ns":
            return value / 1_000_000.0, "ms", "ns_to_ms"
        raise ComparisonError(f"unsupported unit for {metric_name}: {unit}")
    if unit == expected:
        return value, expected, None
    raise ComparisonError(f"unsupported unit for {metric_name}: {unit}")


def _default_unit(metric_name: str) -> str:
    if metric_name == "request_throughput_rps":
        return "requests/s"
    if metric_name.endswith("_tps"):
        return "tokens/s"
    if metric_name in (*LATENCY_METRICS, *PD_DIAGNOSTICS):
        return "ms"
    raise ComparisonError(f"unknown metric name: {metric_name!r}")


def _canonical_metric_name(name: str) -> str:
    aliases = {
        "request_throughput": "request_throughput_rps",
        "output_throughput": "output_token_throughput_tps",
        "output_throughput_tps": "output_token_throughput_tps",
        "token_throughput": "total_token_throughput_tps",
        "total_token_throughput": "total_token_throughput_tps",
        "ttft": "ttft_ms",
        "tpot": "tpot_ms",
        "itl": "tpot_ms",
        "e2e": "e2el_ms",
        "e2el": "e2el_ms",
    }
    if name in aliases:
        return aliases[name]
    if name in (*LATENCY_METRICS, *THROUGHPUT_METRICS, *PD_DIAGNOSTICS):
        return name
    raise ComparisonError(f"unknown metric name: {name!r}")


def _build_report(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    workload: dict[str, Any],
    observed: dict[str, Any],
    predicted: dict[str, Any],
    diagnostic_mode: str,
    ground_truth_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    if manifest["deployment"] == "colocated":
        parallelism = _require_mapping_field(
            manifest, "parallelism", "colocated command manifest")
        normalized_prefill = {}
        normalized_decode = {}
    elif manifest["deployment"] == "disaggregated":
        prefill = _require_mapping_field(
            manifest, "prefill", "disaggregated command manifest")
        decode = _require_mapping_field(
            manifest, "decode", "disaggregated command manifest")
        parallelism = {}
        normalized_prefill = prefill
        normalized_decode = decode
    else:
        raise ComparisonError(
            f"unknown deployment: {manifest['deployment']!r}")
    percentiles = _requested_percentiles(manifest)
    stats_by_metric = {
        **{metric: ("value",) for metric in THROUGHPUT_METRICS},
        **{metric: percentiles for metric in LATENCY_METRICS},
        **{metric: ("mean",) for metric in PD_DIAGNOSTICS},
    }
    metrics = []
    for name, statistics_ in stats_by_metric.items():
        for statistic in statistics_:
            metrics.append(_metric_row(
                name=name,
                statistic=statistic,
                observed=observed,
                predicted=predicted,
                diagnostic_mode=diagnostic_mode,
                manifest=manifest,
            ))
    return {
        "version": 2,
        "kind": "real_vllm_comparison_report",
        "cell_id": manifest["cell_id"],
        "model": manifest["model"],
        "model_type": manifest["model_type"],
        "deployment": manifest["deployment"],
        "parallelism": parallelism,
        "parallelism_summary": _parallelism_summary(manifest),
        "prefill": normalized_prefill,
        "decode": normalized_decode,
        "workload": workload,
        "benchmark": manifest["benchmark"],
        "diagnostic_mode": diagnostic_mode,
        "artifacts": {
            "command_manifest": str(manifest_path),
            "ground_truth": str(ground_truth_path),
            "prediction": str(prediction_path),
        },
        "provenance": manifest["provenance"],
        "scheduler_runtime": predicted["scheduler_runtime"],
        "arrival_process": predicted["arrival_process"],
        "mtp_scenario": predicted["mtp_scenario"],
        "validation_scope": _comparison_validation_scope(manifest),
        "request_counts": {
            "expected": workload["num_requests"],
            "observed_completed": observed["completed"],
            "prediction_requests": predicted["completed"],
        },
        "metrics": metrics,
    }


def _comparison_validation_scope(manifest: dict[str, Any]) -> dict[str, str]:
    commands = _require_mapping_field(
        manifest, "commands", "command manifest")
    if "colocated_midlayer_simulation" in commands:
        return {
            "feature_status": "Projected",
            "aggregate_metrics": "measured_real_vs_modeled",
            "metadata": "declared_invocation_identity_only",
            "request_identity": "unverified_runtime",
            "actual_send_timestamps": "unverified_runtime",
            "scheduler_steps": "unverified_runtime",
            "reason": (
                "vLLM 0.11 bench serve does not persist artifact request IDs, "
                "actual send timestamps, or server scheduler-step traces"
            ),
        }
    return {
        "feature_status": "unscoped",
        "aggregate_metrics": "producer_specific",
        "metadata": "producer_specific",
        "request_identity": "producer_specific",
        "actual_send_timestamps": "producer_specific",
        "scheduler_steps": "producer_specific",
        "reason": "the manifest does not use the canonical vLLM producer",
    }


def _metric_row(
    *,
    name: str,
    statistic: str,
    observed: dict[str, Any],
    predicted: dict[str, Any],
    diagnostic_mode: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    observed_payload = _metric_payload(observed["metrics"], name, statistic)
    predicted_payload = _metric_payload(predicted["metrics"], name, statistic)
    observed_value = observed_payload["value"] if observed_payload is not None else None
    predicted_value = predicted_payload["value"] if predicted_payload is not None else None
    observed_unit = observed_payload["unit"] if observed_payload is not None else None
    predicted_unit = predicted_payload["unit"] if predicted_payload is not None else None
    if observed_unit is not None and predicted_unit is not None and observed_unit != predicted_unit:
        raise ComparisonError(
            f"unit mismatch for {name}.{statistic}: {observed_unit} != {predicted_unit}")
    unit = observed_unit if observed_unit is not None else (
        predicted_unit if predicted_unit is not None else _default_unit(name))
    signed_error = None
    absolute_error = None
    absolute_percentage_error = None
    if observed_value is not None and predicted_value is not None:
        signed_error = predicted_value - observed_value
        absolute_error = abs(signed_error)
        absolute_percentage_error = (
            absolute_error / abs(observed_value) * 100.0
            if observed_value != 0 else None
        )
    observed_available = observed_value is not None
    predicted_available = predicted_value is not None
    conversion_key = (name, statistic)
    observed_conversion = None
    if conversion_key in observed["unit_conversions"]:
        observed_conversion = observed["unit_conversions"][conversion_key]
    predicted_conversion = None
    if conversion_key in predicted["unit_conversions"]:
        predicted_conversion = predicted["unit_conversions"][conversion_key]
    return {
        "name": name,
        "statistic": statistic,
        "unit": unit,
        "observed": _round_or_none(observed_value),
        "predicted": _round_or_none(predicted_value),
        "observed_available": observed_available,
        "predicted_available": predicted_available,
        "availability_reason": _availability_reason(observed_available, predicted_available, statistic),
        "signed_error": _round_or_none(signed_error),
        "absolute_error": _round_or_none(absolute_error),
        "absolute_percentage_error": _round_or_none(absolute_percentage_error),
        "informational": _is_informational(name, diagnostic_mode, manifest),
        "diagnostic_family": "pd" if name in PD_DIAGNOSTICS else "primary",
        "unit_conversions": {
            "observed": observed_conversion,
            "predicted": predicted_conversion,
        },
    }


def _requested_percentiles(manifest: dict[str, Any]) -> tuple[str, ...]:
    requested = ["mean"]
    for percentile in manifest["benchmark"]["metric_percentiles"]:
        if percentile == 50 or percentile == 50.0:
            requested.append("p50")
        elif percentile == 90 or percentile == 90.0:
            requested.append("p90")
        elif percentile == 99 or percentile == 99.0:
            requested.append("p99")
        else:
            raise ComparisonError(
                f"unsupported metric percentile: {percentile!r}; expected 50, 90, or 99")
    for stat in ("p50", "p90", "p99"):
        if stat not in requested:
            requested.append(stat)
    return tuple(requested)


def _availability_reason(observed_available: bool, predicted_available: bool, statistic: str) -> str | None:
    if observed_available and predicted_available:
        return None
    if not observed_available and statistic.startswith("p"):
        return "observed percentile unavailable"
    if not observed_available:
        return "observed metric unavailable"
    return "predicted metric unavailable"


def _is_informational(name: str, diagnostic_mode: str, manifest: dict[str, Any]) -> bool:
    if diagnostic_mode == "colocated":
        return False
    elif diagnostic_mode == "pd" and name in THROUGHPUT_METRICS:
        comparison = _require_mapping_field(manifest, "comparison", "command manifest")
        return comparison["throughput_pass_fail"] is False
    elif diagnostic_mode == "pd":
        return False
    else:
        raise ComparisonError(
            f"unknown diagnostic mode: {diagnostic_mode!r}")


def _validate_identity(
    workload: dict[str, Any],
    observed: dict[str, Any],
    predicted: dict[str, Any],
) -> None:
    expected_hash = workload["hash"]
    for label, source in (("ground truth", observed), ("prediction", predicted)):
        source_hash = source["workload_hash"]
        if source_hash is not None and source_hash != expected_hash:
            raise ComparisonError(
                f"{label} workload hash mismatch: expected {expected_hash}, got {source_hash}")


def _validate_request_counts(
    workload: dict[str, Any],
    observed: dict[str, Any],
    predicted: dict[str, Any],
) -> None:
    expected = workload["num_requests"]
    completed = observed["completed"]
    if completed is None:
        raise ComparisonError("ground truth completed requests are required")
    if completed != expected:
        failed = observed["failed_requests"]
        if not failed:
            raise ComparisonError(
                f"completed requests mismatch: expected {expected}, got {completed}")
    prediction_count = predicted["completed"]
    if prediction_count is not None and prediction_count != completed:
        raise ComparisonError(
            f"prediction request count mismatch: expected {completed}, got {prediction_count}")
    total_output_tokens = observed["total_output_tokens"]
    max_tokens = workload["max_tokens"]
    if total_output_tokens is not None and max_tokens is not None and completed is not None:
        expected_tokens = completed * max_tokens
        if total_output_tokens != expected_tokens:
            raise ComparisonError(
                f"total_output_tokens mismatch: expected {expected_tokens}, got {total_output_tokens}")


def _validate_prediction_adequacy(
    manifest: dict[str, Any],
    observed: dict[str, Any],
    predicted: dict[str, Any],
    diagnostic_mode: str,
) -> None:
    if diagnostic_mode == "pd":
        required = [("ttft_ms", "mean"), ("tpot_ms", "mean")]
    elif diagnostic_mode == "colocated":
        required = [
            ("ttft_ms", "mean"),
            ("tpot_ms", "mean"),
            ("request_throughput_rps", "value"),
            ("output_token_throughput_tps", "value"),
            ("total_token_throughput_tps", "value"),
        ]
    else:
        raise ComparisonError(
            f"unknown diagnostic mode: {diagnostic_mode!r}")
    missing_predicted = [
        f"{metric}.{statistic}"
        for metric, statistic in required
        if metric not in predicted["metrics"]
        or statistic not in predicted["metrics"][metric]
    ]
    missing_observed = [
        f"{metric}.{statistic}"
        for metric, statistic in required
        if (metric not in observed["metrics"]
            or statistic not in observed["metrics"][metric])
        and not _is_informational(metric, diagnostic_mode, manifest)
    ]
    if missing_predicted:
        raise ComparisonError(
            "missing predicted primary metrics: " + ", ".join(missing_predicted))
    if missing_observed:
        raise ComparisonError(
            "missing observed primary metrics: " + ", ".join(missing_observed))


def _extract_workload_hash(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    candidates: list[tuple[str, Any]] = []
    if "workload_hash" in data:
        candidates.append(("workload_hash", data["workload_hash"]))
    if "workload" in data:
        workload = data["workload"]
        if not isinstance(workload, dict):
            raise ComparisonError("workload must be an object")
        for key in ("workload_hash", "hash"):
            if key in workload:
                candidates.append((f"workload.{key}", workload[key]))
    if "metadata" in data:
        metadata = data["metadata"]
        if not isinstance(metadata, dict):
            raise ComparisonError("metadata must be an object")
        if "workload_hash" in metadata:
            candidates.append(
                ("metadata.workload_hash", metadata["workload_hash"]))
    for label, value in candidates:
        if not isinstance(value, str) or not value:
            raise ComparisonError(f"{label} must be a non-empty string")
    hashes = {value for _, value in candidates}
    if len(hashes) > 1:
        raise ComparisonError(
            "conflicting workload hashes: "
            + ", ".join(f"{label}={value!r}" for label, value in candidates))
    if candidates:
        return candidates[0][1]
    return None


def _request_count(data: dict[str, Any]) -> int | None:
    candidates: list[tuple[str, Any]] = []
    for key in ("completed", "request_count", "num_requests"):
        if key in data:
            candidates.append((key, data[key]))
    if "workload" in data:
        workload = data["workload"]
        if not isinstance(workload, dict):
            raise ComparisonError("workload must be an object")
        for key in ("num_requests", "request_count"):
            if key in workload:
                candidates.append((f"workload.{key}", workload[key]))
    counts = {
        _require_int(value, label)
        for label, value in candidates
    }
    if len(counts) > 1:
        raise ComparisonError(
            "conflicting request counts: "
            + ", ".join(f"{label}={value!r}" for label, value in candidates))
    if candidates:
        return _require_int(candidates[0][1], candidates[0][0])
    return None


def _stats(values: list[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    return {
        "mean": statistics.fmean(sorted_values),
        "p50": linear_percentile(sorted_values, 50),
        "p90": linear_percentile(sorted_values, 90),
        "p99": linear_percentile(sorted_values, 99),
    }


def _parallelism_summary(report_or_manifest: dict[str, Any]) -> dict[str, Any]:
    deployment = report_or_manifest["deployment"]
    if deployment == "disaggregated":
        return {
            "deployment": "disaggregated",
            "prefill": _pd_side_parallelism(report_or_manifest["prefill"]),
            "decode": _pd_side_parallelism(report_or_manifest["decode"]),
        }
    elif deployment == "colocated":
        return {
            "deployment": "colocated",
            "parallelism": dict(report_or_manifest["parallelism"]),
        }
    else:
        raise ComparisonError(f"unknown deployment: {deployment!r}")


def _pd_side_parallelism(side: dict[str, Any]) -> dict[str, Any]:
    gpus = list(side["gpus"])
    engine_args = dict(side["engine_args"])
    return {
        "gpus": gpus,
        "gpu_count": len(gpus),
        "tp_size": side["tp_size"],
        "engine_tensor_parallel_size": engine_args["tensor_parallel_size"],
    }


def _aggregate_row(report: dict[str, Any], metric: dict[str, Any]) -> dict[str, Any]:
    group_key = _group_key(report)
    parallelism_summary = report["parallelism_summary"]
    return {
        "cell_id": report["cell_id"],
        "group_key": group_key,
        "model": report["model"],
        "deployment": report["deployment"],
        "parallelism": json.dumps(report["parallelism"], sort_keys=True),
        "parallelism_summary": json.dumps(parallelism_summary, sort_keys=True),
        "workload_hash": report["workload"]["hash"],
        "request_rate": report["benchmark"]["request_rate"],
        "name": metric["name"],
        "statistic": metric["statistic"],
        "unit": metric["unit"],
        "observed": metric["observed"],
        "predicted": metric["predicted"],
        "signed_error": metric["signed_error"],
        "absolute_error": metric["absolute_error"],
        "absolute_percentage_error": metric["absolute_percentage_error"],
        "informational": metric["informational"],
        "diagnostic_family": metric["diagnostic_family"],
    }


def _validate_report_provenance(report: dict[str, Any]) -> None:
    provenance = report["provenance"]
    if not isinstance(provenance, dict):
        raise ComparisonError(
            f"comparison report for {report['cell_id']} missing provenance")
    missing = [
        key for key in PROVENANCE_REQUIRED_KEYS
        if key not in provenance or provenance[key] in (None, "")
    ]
    if missing:
        raise ComparisonError(
            f"comparison report for {report['cell_id']} has incomplete provenance: "
            + ", ".join(missing))


def _group_key(report: dict[str, Any]) -> str:
    parallelism_summary = report["parallelism_summary"]
    return "|".join([
        str(report["model"]),
        str(report["deployment"]),
        json.dumps(parallelism_summary, sort_keys=True),
        str(report["workload"]["hash"]),
        str(report["benchmark"]["request_rate"]),
    ])


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["group_key"] not in grouped:
            grouped[row["group_key"]] = {
                "group_key": row["group_key"],
                "model": row["model"],
                "deployment": row["deployment"],
                "parallelism": row["parallelism"],
                "parallelism_summary": row["parallelism_summary"],
                "workload_hash": row["workload_hash"],
                "request_rate": row["request_rate"],
                "metric_count": 0,
            }
        group = grouped[row["group_key"]]
        group["metric_count"] += 1
    return list(grouped.values())


def _top_modeling_gaps(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row["absolute_percentage_error"] is not None
        and row["informational"] is False
    ]
    return sorted(
        candidates,
        key=lambda row: row["absolute_percentage_error"],
        reverse=True,
    )[:limit]


def _render_cell_markdown(report: dict[str, Any]) -> str:
    scope = report["validation_scope"]
    lines = [
        "# Comparison Report",
        "",
        f"Cell: `{report['cell_id']}`",
        f"Deployment: `{report['deployment']}`",
        f"Workload hash: `{report['workload']['hash']}`",
        f"Feature validation: `{scope['feature_status']}`",
        f"Validation boundary: {scope['reason']}",
        "",
        "| Metric | Stat | Observed | Predicted | Unit | APE | Note |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in report["metrics"]:
        ape = "" if row["absolute_percentage_error"] is None else f"{row['absolute_percentage_error']:.6g}"
        if row["informational"]:
            note = "informational"
        elif row["availability_reason"] is None:
            note = ""
        elif (isinstance(row["availability_reason"], str)
              and row["availability_reason"]):
            note = row["availability_reason"]
        else:
            raise ComparisonError(
                "metric availability_reason must be null or a non-empty string")
        lines.append(
            f"| {row['name']} | {row['statistic']} | {_format_value(row['observed'])} | "
            f"{_format_value(row['predicted'])} | {row['unit']} | {ape} | {note} |"
        )
    return "\n".join(lines) + "\n"


def _render_aggregate_markdown(
    run_root: Path,
    groups: list[dict[str, Any]],
    top_gaps: list[dict[str, Any]],
) -> str:
    lines = [
        "# Aggregate Report",
        "",
        f"Run directory: `{run_root}`",
        "",
        "## Groups",
    ]
    for group in groups:
        lines.append(
            f"- `{group['group_key']}` metrics={group['metric_count']}"
        )
    lines.extend(["", "## Top Modeling Gaps"])
    for row in top_gaps:
        lines.append(
            f"- `{row['cell_id']}` {row['name']} {row['statistic']}: "
            f"APE={row['absolute_percentage_error']:.6g}%"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cell_id", "group_key", "model", "deployment", "parallelism",
        "parallelism_summary", "workload_hash", "request_rate", "name",
        "statistic", "unit", "observed", "predicted", "signed_error",
        "absolute_error", "absolute_percentage_error", "informational",
        "diagnostic_family",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)




def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _consistent_cell_status(
    status: dict[str, Any],
    summary_cell: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    sources = {
        "status artifact": status["status"],
        "run summary": summary_cell["status"],
        "command manifest": manifest["status"],
    }
    for label, value in sources.items():
        if not isinstance(value, str):
            raise ComparisonError(f"{label} status must be a string")
    values = set(sources.values())
    if len(values) != 1:
        details = ", ".join(
            f"{label}={value!r}" for label, value in sources.items())
        raise ComparisonError(f"conflicting experiment statuses: {details}")
    return status["status"]


def _require_mapping_field(
    mapping: dict[str, Any],
    key: str,
    owner: str,
) -> dict[str, Any]:
    if key not in mapping:
        raise ComparisonError(f"{owner} missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise ComparisonError(f"{owner}.{key} must be an object")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{label} must be a number")
    return float(value)


def _metric_payload(
    metrics: dict[str, Any],
    metric_name: str,
    statistic: str,
) -> dict[str, Any] | None:
    if metric_name not in metrics:
        return None
    group = metrics[metric_name]
    if not isinstance(group, dict):
        raise ComparisonError(f"metric group {metric_name!r} must be an object")
    if statistic not in group:
        return None
    payload = group[statistic]
    if not isinstance(payload, dict) or set(payload) != {"value", "unit"}:
        raise ComparisonError(
            f"metric payload {metric_name}.{statistic} must contain value/unit")
    _require_number(payload["value"], f"{metric_name}.{statistic}.value")
    if not isinstance(payload["unit"], str) or not payload["unit"]:
        raise ComparisonError(f"{metric_name}.{statistic}.unit must be a non-empty string")
    return payload


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(
            f"optional numeric evidence must be a number or null, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonError(
            f"optional numeric evidence must be finite, got {value!r}")
    return result


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComparisonError(f"{label} must be an integer")
    return value


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
