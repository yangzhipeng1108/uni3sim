"""Reconstruct client TTFT from one absolute-clock Scheduler capture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence


_BOUNDARIES = (
    "device_execution_end_ns",
    "scheduler_update_start_ns",
    "scheduler_update_end_ns",
)
_REQUEST_STAGES = (
    "openai_create_completion",
    "tokenizer_batch",
    "engine_input_processor",
    "output_processor_add_request",
    "engine_core_ipc_enqueue",
    "engine_core_preprocess",
    "engine_core_add_request",
    "scheduler_add_request",
)


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    rank = percentile / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": fmean(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p99": _percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def analyze_oracle_ttft(
    ground_truth_path: str | Path,
    scheduler_trace_path: str | Path,
    request_add_trace_path: str | Path | None = None,
) -> dict[str, object]:
    ground_truth_path = Path(ground_truth_path).resolve(strict=True)
    scheduler_trace_path = Path(scheduler_trace_path).resolve(strict=True)
    clock_path = Path(f"{scheduler_trace_path}.clock.json").resolve(strict=True)
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    clock = json.loads(clock_path.read_text(encoding="utf-8"))
    benchmark_clock = ground_truth["metrics"]["benchmark_clock"]
    if benchmark_clock["clock_domain"] != clock["clock_domain"]:
        raise ValueError("benchmark and Scheduler clock domains differ")
    if benchmark_clock["host_boot_id"] != clock["host_boot_id"]:
        raise ValueError("benchmark and Scheduler host boot IDs differ")
    requests = ground_truth["metrics"]["per_request"]
    scheduler_rows = [
        json.loads(line)
        for line in scheduler_trace_path.read_text(encoding="utf-8").splitlines()
    ]
    if not requests or not scheduler_rows:
        raise ValueError("oracle TTFT reconstruction requires requests and steps")
    request_count = len(requests)
    request_add_ns: list[int] | None = None
    request_stages: list[dict[str, Mapping[str, object]]] | None = None
    request_stage_paths: list[Path] = []
    resolved_request_add_path: Path | None = None
    if request_add_trace_path is not None:
        resolved_request_add_path = Path(request_add_trace_path).resolve(strict=True)
        request_add_trace = json.loads(
            resolved_request_add_path.read_text(encoding="utf-8")
        )
        if set(request_add_trace) != {
            "schema_version", "clock_domain", "boundary", "request_count",
            "records",
        }:
            raise ValueError("Scheduler request-add trace fields are invalid")
        if request_add_trace["schema_version"] != 1:
            raise ValueError("Scheduler request-add trace schema is invalid")
        if request_add_trace["clock_domain"] != clock["clock_domain"]:
            raise ValueError("Scheduler request-add trace clock domain differs")
        if request_add_trace["boundary"] != "Scheduler.add_request return":
            raise ValueError("Scheduler request-add trace boundary is invalid")
        if request_add_trace["request_count"] != request_count:
            raise ValueError("Scheduler request-add trace count differs")
        request_add_ns = [0] * request_count
        seen_request_indexes = set()
        for value in request_add_trace["records"]:
            if set(value) != {
                "request_idx", "scheduler_add_end_monotonic_ns",
            }:
                raise ValueError("Scheduler request-add record fields are invalid")
            request_index = value["request_idx"]
            timestamp_ns = value["scheduler_add_end_monotonic_ns"]
            if (
                isinstance(request_index, bool)
                or not isinstance(request_index, int)
                or not 0 <= request_index < request_count
                or request_index in seen_request_indexes
            ):
                raise ValueError("Scheduler request-add index is invalid")
            if (
                isinstance(timestamp_ns, bool)
                or not isinstance(timestamp_ns, int)
                or timestamp_ns < 0
            ):
                raise ValueError("Scheduler request-add timestamp is invalid")
            seen_request_indexes.add(request_index)
            request_add_ns[request_index] = timestamp_ns
        if seen_request_indexes != set(range(request_count)):
            raise ValueError("Scheduler request-add trace coverage is incomplete")
        request_stage_paths = sorted(
            resolved_request_add_path.parent.glob(
                f"{resolved_request_add_path.name}.stages.*.jsonl"
            )
        )
        if request_stage_paths:
            request_stages = [dict() for _ in range(request_count)]
            for stage_path in request_stage_paths:
                for line in stage_path.read_text(encoding="utf-8").splitlines():
                    value = json.loads(line)
                    required = {
                        "schema_version",
                        "clock_domain",
                        "process_pid",
                        "request_idx",
                        "request_id",
                        "stage",
                        "start_monotonic_ns",
                        "end_monotonic_ns",
                        "thread_cpu_duration_ns",
                    }
                    if not required <= set(value):
                        raise ValueError("request stage record fields are incomplete")
                    if value["schema_version"] != 1:
                        raise ValueError("request stage schema version is invalid")
                    if value["clock_domain"] != clock["clock_domain"]:
                        raise ValueError("request stage clock domain differs")
                    request_index = value["request_idx"]
                    if (
                        isinstance(request_index, bool)
                        or not isinstance(request_index, int)
                        or not 0 <= request_index < request_count
                    ):
                        raise ValueError("request stage index is invalid")
                    stage = value["stage"]
                    if stage not in _REQUEST_STAGES:
                        raise ValueError(f"unsupported request stage: {stage!r}")
                    if stage in request_stages[request_index]:
                        raise ValueError(
                            "request stage was recorded more than once: "
                            f"request={request_index}, stage={stage}"
                        )
                    for field in (
                        "start_monotonic_ns",
                        "end_monotonic_ns",
                        "thread_cpu_duration_ns",
                    ):
                        timing_value = value[field]
                        if (
                            isinstance(timing_value, bool)
                            or not isinstance(timing_value, int)
                            or timing_value < 0
                        ):
                            raise TypeError(f"request stage {field} is invalid")
                    if value["end_monotonic_ns"] < value["start_monotonic_ns"]:
                        raise ValueError("request stage has negative wall duration")
                    request_stages[request_index][stage] = value
            for request_index, stages in enumerate(request_stages):
                if set(stages) != set(_REQUEST_STAGES):
                    raise ValueError(
                        "request stage coverage is incomplete: "
                        f"request={request_index}, stages={sorted(stages)}"
                    )
    scheduled_prefill = [0] * request_count
    first_step: list[Mapping[str, object] | None] = [None] * request_count
    first_token_step: list[Mapping[str, object] | None] = [None] * request_count
    for row in scheduler_rows:
        for scheduled in row["scheduled_requests"]:
            request_index = scheduled["request_idx"]
            if (
                isinstance(request_index, bool)
                or not isinstance(request_index, int)
                or not 0 <= request_index < request_count
            ):
                raise ValueError("Scheduler request index is invalid")
            prefill_tokens = scheduled["prefill_tokens"]
            if prefill_tokens <= 0 or first_token_step[request_index] is not None:
                continue
            if first_step[request_index] is None:
                first_step[request_index] = row
            scheduled_prefill[request_index] += prefill_tokens
            if scheduled_prefill[request_index] >= requests[request_index]["input_tokens"]:
                first_token_step[request_index] = row
    if any(row is None for row in first_step):
        raise ValueError("at least one request has no first Prefill step")
    if any(row is None for row in first_token_step):
        raise ValueError("at least one request has no first-token-producing step")
    origin_ns = clock["trace_origin_monotonic_ns"]
    per_request = []
    for request_index, request in enumerate(requests):
        start_row = first_step[request_index]
        token_row = first_token_step[request_index]
        if start_row is None or token_row is None:
            raise AssertionError("validated Scheduler boundary is missing")
        client_start_ns = request["ttft_start_monotonic_ns"]
        client_first_token_ns = request["first_token_monotonic_ns"]
        golden_ttft_ms = request["ttft_ms"]
        boundaries = {}
        for field in _BOUNDARIES:
            oracle_ms = (
                origin_ns + token_row[field] - client_start_ns
            ) / 1_000_000.0
            boundaries[field] = {
                "oracle_ttft_ms": oracle_ms,
                "golden_minus_oracle_ms": golden_ttft_ms - oracle_ms,
                "absolute_error_ms": abs(golden_ttft_ms - oracle_ms),
                "absolute_percentage_error": (
                    100.0 * abs(golden_ttft_ms - oracle_ms) / golden_ttft_ms
                ),
            }
        first_schedule_ns = origin_ns + start_row["schedule_start_ns"]
        scheduler_end_ns = origin_ns + token_row["scheduler_update_end_ns"]
        ingress_and_queue_ms = (first_schedule_ns - client_start_ns) / 1_000_000.0
        scheduler_path_ms = (scheduler_end_ns - first_schedule_ns) / 1_000_000.0
        post_scheduler_ms = (
            client_first_token_ns - scheduler_end_ns
        ) / 1_000_000.0
        if min(ingress_and_queue_ms, scheduler_path_ms, post_scheduler_ms) < 0.0:
            raise ValueError("aligned TTFT segment is negative")
        closure_ms = (
            ingress_and_queue_ms + scheduler_path_ms + post_scheduler_ms
        )
        if not math.isclose(closure_ms, golden_ttft_ms, abs_tol=1e-6):
            raise ValueError("aligned TTFT decomposition does not close")
        per_request.append({
            "request_idx": request_index,
            "golden_ttft_ms": golden_ttft_ms,
            "ingress_and_queue_ms": ingress_and_queue_ms,
            "scheduler_path_ms": scheduler_path_ms,
            "post_scheduler_ms": post_scheduler_ms,
            "boundaries": boundaries,
        })
        if request_add_ns is not None:
            add_end_ns = request_add_ns[request_index]
            if not client_start_ns <= add_end_ns <= first_schedule_ns:
                raise ValueError("Scheduler admission boundary is not causal")
            per_request[-1]["client_to_scheduler_add_ms"] = (
                add_end_ns - client_start_ns
            ) / 1_000_000.0
            per_request[-1]["scheduler_queue_ms"] = (
                first_schedule_ns - add_end_ns
            ) / 1_000_000.0
        if request_stages is not None:
            stages = request_stages[request_index]
            scheduler_add = stages["scheduler_add_request"]
            if scheduler_add["end_monotonic_ns"] != request_add_ns[request_index]:
                raise ValueError("Scheduler-add span disagrees with admission trace")
            tokenizer = stages["tokenizer_batch"]
            if set(tokenizer) != {
                "schema_version",
                "clock_domain",
                "process_pid",
                "request_idx",
                "request_id",
                "stage",
                "start_monotonic_ns",
                "end_monotonic_ns",
                "thread_cpu_duration_ns",
                "tokenizer_queue_duration_ns",
                "tokenizer_execution_start_monotonic_ns",
                "tokenizer_execution_wall_duration_ns",
                "tokenizer_batch_size",
            }:
                raise ValueError("tokenizer stage fields are invalid")
            tokenizer_queue_ns = tokenizer["tokenizer_queue_duration_ns"]
            if (
                isinstance(tokenizer_queue_ns, bool)
                or not isinstance(tokenizer_queue_ns, int)
                or tokenizer_queue_ns < 0
            ):
                raise TypeError("tokenizer queue duration is invalid")

            def wall_ns(stage: str) -> int:
                value = stages[stage]
                return int(value["end_monotonic_ns"]) - int(
                    value["start_monotonic_ns"]
                )

            engine_core_add = stages["engine_core_add_request"]
            scheduler_add_start_ns = int(scheduler_add["start_monotonic_ns"])
            engine_core_add_start_ns = int(
                engine_core_add["start_monotonic_ns"]
            )
            if scheduler_add_start_ns < engine_core_add_start_ns:
                raise ValueError("Scheduler admission predates EngineCore admission")
            explicit_wait_free_ns = (
                wall_ns("openai_create_completion")
                - tokenizer_queue_ns
                + wall_ns("engine_input_processor")
                + wall_ns("output_processor_add_request")
                + wall_ns("engine_core_ipc_enqueue")
                + wall_ns("engine_core_preprocess")
                + scheduler_add_start_ns
                - engine_core_add_start_ns
            )
            if explicit_wait_free_ns < 0:
                raise ValueError("wait-free pre-Scheduler duration is negative")
            enqueue = stages["engine_core_ipc_enqueue"]
            preprocess = stages["engine_core_preprocess"]
            ipc_queue_ns = int(preprocess["start_monotonic_ns"]) - int(
                enqueue["end_monotonic_ns"]
            )
            if ipc_queue_ns < 0:
                raise ValueError("EngineCore IPC queue duration is negative")
            openai = stages["openai_create_completion"]
            input_processor = stages["engine_input_processor"]
            output_processor = stages["output_processor_add_request"]
            engine_core_preprocess = stages["engine_core_preprocess"]
            handoff_boundaries = {
                "client_to_api_entry_ms": (
                    int(openai["start_monotonic_ns"]) - client_start_ns
                ) / 1_000_000.0,
                "api_to_engine_input_handoff_ms": (
                    int(input_processor["start_monotonic_ns"])
                    - int(openai["end_monotonic_ns"])
                ) / 1_000_000.0,
                "engine_input_to_output_processor_handoff_ms": (
                    int(output_processor["start_monotonic_ns"])
                    - int(input_processor["end_monotonic_ns"])
                ) / 1_000_000.0,
                "output_processor_to_ipc_enqueue_handoff_ms": (
                    int(enqueue["start_monotonic_ns"])
                    - int(output_processor["end_monotonic_ns"])
                ) / 1_000_000.0,
                "engine_core_admission_queue_ms": (
                    engine_core_add_start_ns
                    - int(engine_core_preprocess["end_monotonic_ns"])
                ) / 1_000_000.0,
            }
            if min(handoff_boundaries.values()) < 0:
                raise ValueError("request handoff duration is negative")
            per_request[-1]["explicit_wait_free_pre_scheduler_ms"] = (
                explicit_wait_free_ns / 1_000_000.0
            )
            per_request[-1]["tokenizer_queue_ms"] = (
                tokenizer_queue_ns / 1_000_000.0
            )
            per_request[-1]["engine_core_ipc_queue_ms"] = (
                ipc_queue_ns / 1_000_000.0
            )
            per_request[-1].update(handoff_boundaries)
            per_request[-1]["request_stages"] = stages
    boundary_summary = {}
    for field in _BOUNDARIES:
        oracle = [row["boundaries"][field]["oracle_ttft_ms"] for row in per_request]
        errors = [
            row["boundaries"][field]["golden_minus_oracle_ms"]
            for row in per_request
        ]
        absolute_errors = [abs(value) for value in errors]
        percentages = [
            row["boundaries"][field]["absolute_percentage_error"]
            for row in per_request
        ]
        boundary_summary[field] = {
            "oracle_ttft_ms": _summary(oracle),
            "golden_minus_oracle_ms": _summary(errors),
            "absolute_error_ms": _summary(absolute_errors),
            "mape_percent": fmean(percentages),
        }
    result = {
        "schema_version": (
            3 if request_stages is not None else 2 if request_add_ns is not None else 1
        ),
        "kind": "observed_scheduler_ttft_oracle",
        "semantics": {
            "strength": (
                "uses the observed request trajectory and observed host-side "
                "device/Scheduler completion boundaries; this is stronger than "
                "a perfect kernel-duration predictor"
            ),
            "golden": "client first-token receive minus vLLM benchmark request start",
            "oracle": "selected observed Scheduler boundary minus the same client start",
        },
        "sources": {
            "ground_truth": str(ground_truth_path),
            "scheduler_trace": str(scheduler_trace_path),
            "scheduler_clock": str(clock_path),
        },
        "clock": clock,
        "request_count": request_count,
        "golden_ttft_ms": _summary(
            [row["golden_ttft_ms"] for row in per_request]
        ),
        "segments_ms": {
            name: _summary([row[name] for row in per_request])
            for name in (
                "ingress_and_queue_ms",
                "scheduler_path_ms",
                "post_scheduler_ms",
            )
        },
        "boundaries": boundary_summary,
        "per_request": per_request,
    }
    if resolved_request_add_path is not None:
        result["sources"]["scheduler_request_add_trace"] = str(
            resolved_request_add_path
        )
        result["endpoint_segments_ms"] = {
            field: _summary([float(row[field]) for row in per_request])
            for field in (
                "client_to_scheduler_add_ms",
                "scheduler_queue_ms",
                "post_scheduler_ms",
            )
        }
    if request_stages is not None:
        result["sources"]["request_stage_traces"] = [
            str(path) for path in request_stage_paths
        ]
        result["wait_free_endpoint_segments_ms"] = {
            field: _summary([float(row[field]) for row in per_request])
            for field in (
                "explicit_wait_free_pre_scheduler_ms",
                "tokenizer_queue_ms",
                "engine_core_ipc_queue_ms",
                "client_to_api_entry_ms",
                "api_to_engine_input_handoff_ms",
                "engine_input_to_output_processor_handoff_ms",
                "output_processor_to_ipc_enqueue_handoff_ms",
                "engine_core_admission_queue_ms",
                "post_scheduler_ms",
            )
        }
    return result


def _plot_wait_free_endpoints(
    result: dict[str, object], output_path: Path
) -> None:
    if "wait_free_endpoint_segments_ms" not in result:
        raise ValueError("request-stage traces are required for endpoint plot")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite plot: {output_path}")
    import matplotlib.pyplot as plt

    rows = result["per_request"]
    request_indexes = [row["request_idx"] for row in rows]
    pre_scheduler = [
        row["explicit_wait_free_pre_scheduler_ms"] for row in rows
    ]
    post_scheduler = [row["post_scheduler_ms"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    figure.suptitle(
        "A800 Qwen3-4B TP1 QPS16: per-request TTFT endpoint overhead",
        fontsize=16,
    )
    axes[0].plot(request_indexes, pre_scheduler, linewidth=1.2)
    axes[0].set_ylabel("Pre-Scheduler\nprocessing (ms)", fontsize=12)
    axes[0].set_title("Observed queue and handoff intervals removed", fontsize=13)
    axes[1].plot(request_indexes, post_scheduler, linewidth=1.2)
    axes[1].set_ylabel("Post-Scheduler\nto client (ms)", fontsize=12)
    axes[1].set_xlabel("Request index", fontsize=12)
    axes[1].set_title("Scheduler update complete to client-visible first token", fontsize=13)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _write_fixed_endpoint_counterfactual(
    result: dict[str, object],
    oracle_result_path: Path,
    comparison_path: Path,
    output_path: Path,
    plot_path: Path,
) -> None:
    for path in (output_path, plot_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    cells = [
        cell for cell in comparison["cells"]
        if cell["cell_id"] == "nvidia_a800/qwen3-4b/tp1_qps16"
    ]
    if len(cells) != 1:
        raise ValueError("comparison must contain exactly one target A800 cell")
    predicted_rows = cells[0]["metrics"]["paired_request_latency"]["metrics"][
        "ttft_ms"
    ]
    predicted_by_index = {
        row["request_idx"]: row["predicted_ms"] for row in predicted_rows
    }
    measured_by_index = {
        row["request_idx"]: row["golden_ttft_ms"]
        for row in result["per_request"]
    }
    if set(predicted_by_index) != set(measured_by_index):
        raise ValueError("measured and predicted request identities differ")
    fixed_overhead_ms = (
        result["wait_free_endpoint_segments_ms"][
            "explicit_wait_free_pre_scheduler_ms"
        ]["mean"]
        + result["wait_free_endpoint_segments_ms"]["post_scheduler_ms"]["mean"]
    )
    indexes = sorted(measured_by_index)
    measured = [measured_by_index[index] for index in indexes]
    predicted = [predicted_by_index[index] for index in indexes]
    fixed = [value + fixed_overhead_ms for value in predicted]

    def metrics(values: list[float]) -> dict[str, float]:
        signed = [value - truth for value, truth in zip(values, measured)]
        absolute = [abs(value) for value in signed]
        percentages = [
            100.0 * value / truth for value, truth in zip(absolute, measured)
        ]
        return {
            "mean_predicted_ttft_ms": fmean(values),
            "mean_signed_error_ms": fmean(signed),
            "mae_ms": fmean(absolute),
            "mape_percent": fmean(percentages),
        }

    payload = {
        "schema_version": 3,
        "kind": "wait_free_fixed_endpoint_ttft_counterfactual",
        "cell_id": "nvidia_a800/qwen3-4b/tp1_qps16",
        "request_count": len(indexes),
        "measured_source": result["sources"]["ground_truth"],
        "predicted_source": str(comparison_path.resolve()),
        "fixed_overhead_ms": fixed_overhead_ms,
        "fixed_overhead_source": {
            "explicit_wait_free_pre_scheduler_mean_ms": result[
                "wait_free_endpoint_segments_ms"
            ]["explicit_wait_free_pre_scheduler_ms"]["mean"],
            "post_scheduler_mean_ms": result["wait_free_endpoint_segments_ms"][
                "post_scheduler_ms"
            ]["mean"],
            "oracle_result": str(oracle_result_path.resolve()),
        },
        "original": {
            **metrics(predicted),
            "mean_measured_ttft_ms": fmean(measured),
        },
        "fixed_added": metrics(fixed),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    order = sorted(range(len(indexes)), key=lambda index: measured[index])
    measured_sorted = [measured[index] for index in order]
    predicted_sorted = [predicted[index] for index in order]
    fixed_sorted = [fixed[index] for index in order]
    x_values = list(range(len(order)))
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    figure.suptitle(
        "A800 Qwen3-4B TP1 QPS16: wait-free endpoint correction",
        fontsize=16,
    )
    axes[0].plot(x_values, measured_sorted, label="Measured", linewidth=1.4)
    axes[0].plot(x_values, predicted_sorted, label="Predicted", linewidth=1.2)
    axes[0].plot(
        x_values,
        fixed_sorted,
        label=f"Predicted + {fixed_overhead_ms:.3f} ms",
        linewidth=1.2,
    )
    axes[0].set_ylabel("TTFT (ms)", fontsize=12)
    axes[0].legend(loc="best", fontsize=11)
    original_error = [
        value - truth for value, truth in zip(predicted_sorted, measured_sorted)
    ]
    fixed_error = [
        value - truth for value, truth in zip(fixed_sorted, measured_sorted)
    ]
    axes[1].plot(x_values, original_error, label="Original error", linewidth=1.2)
    axes[1].plot(x_values, fixed_error, label="Corrected error", linewidth=1.2)
    axes[1].axhline(0.0, color="0.3", linewidth=1.0)
    axes[1].set_ylabel("Predicted - measured (ms)", fontsize=12)
    axes[1].set_xlabel("Requests sorted by measured TTFT", fontsize=12)
    axes[1].legend(loc="best", fontsize=11)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.tick_params(labelsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def run_oracle_ttft_reconstruction_cli(
    argv: Sequence[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct client TTFT from an absolute-clock Scheduler capture."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scheduler-trace", type=Path, required=True)
    parser.add_argument("--scheduler-request-add-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path)
    parser.add_argument("--comparison-results", type=Path)
    parser.add_argument("--fixed-counterfactual-output", type=Path)
    parser.add_argument("--fixed-counterfactual-plot-output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {arguments.output}")
    result = analyze_oracle_ttft(
        arguments.ground_truth,
        arguments.scheduler_trace,
        arguments.scheduler_request_add_trace,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if arguments.plot_output is not None:
        _plot_wait_free_endpoints(result, arguments.plot_output)
    counterfactual_arguments = (
        arguments.comparison_results,
        arguments.fixed_counterfactual_output,
        arguments.fixed_counterfactual_plot_output,
    )
    if any(value is not None for value in counterfactual_arguments):
        if not all(value is not None for value in counterfactual_arguments):
            raise ValueError(
                "comparison results and both counterfactual outputs are required"
            )
        _write_fixed_endpoint_counterfactual(
            result,
            arguments.output,
            arguments.comparison_results,
            arguments.fixed_counterfactual_output,
            arguments.fixed_counterfactual_plot_output,
        )
    endpoint_segments = None
    if "endpoint_segments_ms" in result:
        endpoint_segments = result["endpoint_segments_ms"]
    wait_free_endpoint_segments = None
    if "wait_free_endpoint_segments_ms" in result:
        wait_free_endpoint_segments = result["wait_free_endpoint_segments_ms"]
    plot_output = None
    if arguments.plot_output is not None:
        plot_output = str(arguments.plot_output.resolve())
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "request_count": result["request_count"],
        "golden_ttft_ms": result["golden_ttft_ms"],
        "scheduler_update_end": result["boundaries"]["scheduler_update_end_ns"],
        "segments_ms": result["segments_ms"],
        "endpoint_segments_ms": endpoint_segments,
        "wait_free_endpoint_segments_ms": wait_free_endpoint_segments,
        "plot_output": plot_output,
    }, indent=2, sort_keys=True))
