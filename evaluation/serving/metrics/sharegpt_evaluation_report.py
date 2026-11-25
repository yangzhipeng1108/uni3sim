"""Aggregate-only CSV and PNG reports for ShareGPT Serving evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Mapping, Sequence

from evaluation.serving.metrics.latency_distribution import linear_percentile

from evaluation.json_shape import require_object as _mapping


_LATENCY_METRICS = ("ttft_ms", "tpot_ms", "e2e_latency_ms")
_LATENCY_DISTRIBUTION_FIELDS = {
    "ttft_ms": "ttft_s",
    "tpot_ms": "tpot_s",
    "e2e_latency_ms": "latency_s",
}
_PLOTTED_LATENCY_METRICS = {
    "ttft_ms": "ttft",
    "tpot_ms": "tpot",
}
STATISTICS = ("mean", "p50", "p90", "p99")
_HARDWARE_LABELS = {
    "nvidia_a800": "NVIDIA A800",
    "metax_c500": "MetaX C500",
    "ascend_910b": "Ascend 910B",
    "nvidia_h200": "NVIDIA H200",
}
_HARDWARE_RANK = {
    hardware_id: rank for rank, hardware_id in enumerate(_HARDWARE_LABELS)
}
MODEL_ORDER = (
    "qwen3-4b",
    "qwen3-8b",
    "llama-3.1-8b",
    "qwen3-32b",
    "qwen3-30b-a3b",
    "llama-3-70b",
    "llama-3.1-70b",
)
MODEL_RANK = {model_id: rank for rank, model_id in enumerate(MODEL_ORDER)}
_TIMING_MODES = (
    "device_only",
    "device_plus_launch",
    "device_plus_launch_plus_scheduler",
)
_TIMING_MODE_LABELS = {
    "device_only": "Device only",
    "device_plus_launch": "+Launch",
    "device_plus_launch_plus_scheduler": "+Scheduler",
}
_COMPARISON_LABELS = ("Observed",) + tuple(
    _TIMING_MODE_LABELS[mode] for mode in _TIMING_MODES
)
_COMPARISON_LINE_WIDTHS = (2.0, 1.6, 1.6, 1.6)
_MEASURED_COLOR = "#4c78a8"
_PREDICTED_COLOR = "#f58518"
_RELATIVE_ERROR_COLOR = "#d62728"
_LEGEND_FONTSIZE = 11
_FIGURE_TITLE_FONTSIZE = 14
_STANDARD_PLOT_DPI = 150
_OVERVIEW_PLOT_DPI = 165
_DENSE_PLOT_DPI = 175
_SCHEDULER_PLOT_DPI = 155


def _style_subplot_titles(figure: object) -> None:
    visible_axes = [axis for axis in figure.axes if axis.get_visible()]
    title_increment = 2.0 if len(visible_axes) == 1 else 1.0
    for axis in visible_axes:
        axis.title.set_fontsize(axis.title.get_fontsize() + title_increment)
        axis.title.set_fontweight("bold")


def write_sharegpt_evaluation_report(
    output_dir: str | Path,
    cells: Sequence[Mapping[str, object]],
    *,
    run_elapsed_wall_time_seconds: float,
    plot_jobs: int = 1,
) -> Mapping[str, Path]:
    """Write maintained tables and plots after a successful metrics run."""

    if type(plot_jobs) is not int:
        raise TypeError("ShareGPT plot_jobs must be an integer")
    if plot_jobs < 1:
        raise ValueError("ShareGPT plot_jobs must be positive")

    destination = Path(output_dir)
    if not destination.is_dir():
        raise ValueError(
            f"ShareGPT report output directory does not exist: {destination}"
        )
    normalized_cells = tuple(sorted(
        (
            _validated_cell(cell, require_paired_latency=True)
            for cell in cells
        ),
        key=_report_cell_sort_key,
    ))
    if not normalized_cells:
        raise ValueError("ShareGPT evaluation report requires at least one cell")
    cell_ids = [str(cell["cell_id"]) for cell in normalized_cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("ShareGPT evaluation report cell IDs must be unique")
    run_elapsed = _finite(
        run_elapsed_wall_time_seconds, "report run elapsed wall time"
    )
    if run_elapsed < 0.0:
        raise ValueError("report run elapsed wall time must be nonnegative")

    per_case_rows = [_per_case_row(cell) for cell in normalized_cells]
    percentile_rows = [
        row
        for cell in normalized_cells
        for row in _latency_percentile_rows(cell)
    ]
    speed_rows = [_speed_row(cell) for cell in normalized_cells]
    host_timing_rows = _host_timing_contribution_rows(normalized_cells)
    speed_summary_rows = _speed_summary_rows(
        speed_rows,
        run_elapsed_wall_time_seconds=run_elapsed,
    )
    memory_rows = [_memory_usage_row(cell) for cell in normalized_cells]
    memory_summary_rows = _memory_summary_rows(memory_rows)
    midlayer_wall_time_rows = _midlayer_wall_time_rows(normalized_cells)

    paths: dict[str, Path] = {
        "per_case_metrics": destination / "per_case_metrics.csv",
        "latency_percentiles_p0_p100": (
            destination / "latency_percentiles_p0_p100.csv"
        ),
        "simulation_speed": destination / "simulation_speed.csv",
        "simulation_speed_summary": (
            destination / "simulation_speed_summary.csv"
        ),
        "memory_usage": destination / "memory_usage.csv",
        "memory_usage_summary": destination / "memory_usage_summary.csv",
    }
    if midlayer_wall_time_rows:
        paths["midlayer_wall_time"] = destination / "midlayer_wall_time.csv"
    if host_timing_rows:
        paths["host_timing_contributions"] = (
            destination / "host_timing_contributions.csv"
        )
    _write_csv(paths["per_case_metrics"], per_case_rows)
    _write_csv(paths["latency_percentiles_p0_p100"], percentile_rows)
    _write_csv(paths["simulation_speed"], speed_rows)
    _write_csv(paths["simulation_speed_summary"], speed_summary_rows)
    _write_csv(paths["memory_usage"], memory_rows)
    _write_csv(paths["memory_usage_summary"], memory_summary_rows)
    if midlayer_wall_time_rows:
        _write_csv(paths["midlayer_wall_time"], midlayer_wall_time_rows)
    if host_timing_rows:
        _write_csv(paths["host_timing_contributions"], host_timing_rows)

    plot_dir = destination / "plots"
    plot_dir.mkdir(exist_ok=False)
    plot_started = time.perf_counter()
    plot_paths = _write_plots(
        plot_dir,
        normalized_cells,
        per_case_rows,
        percentile_rows,
        speed_rows,
        memory_rows,
        host_timing_rows,
        plot_jobs=plot_jobs,
    )
    plot_wall_time_seconds = time.perf_counter() - plot_started
    paths.update(plot_paths)
    timing_path = destination / "report_generation_timing.json"
    timing_path.write_text(json.dumps({
        "schema_version": 1,
        "plot_wall_time_seconds": plot_wall_time_seconds,
    }, indent=2, sort_keys=True) + "\n")
    paths["report_generation_timing"] = timing_path
    hashes_path = destination / "report_hashes.json"
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(paths.items())
    }
    hashes_path.write_text(
        json.dumps({"schema_version": 1, "sha256": hashes}, indent=2,
                   sort_keys=True) + "\n"
    )
    paths["report_hashes"] = hashes_path
    return paths


def write_sharegpt_latency_quantile_plots(
    output_dir: str | Path,
    cells: Sequence[Mapping[str, object]],
    *,
    filename_prefix: str,
) -> Mapping[str, Path]:
    """Write per-hardware measured-vs-predicted TTFT/TPOT quantile plots."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ShareGPT latency quantile plot rendering requires matplotlib"
        ) from error

    if not isinstance(filename_prefix, str) or not filename_prefix:
        raise ValueError("latency quantile plot filename prefix must be non-empty")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalized_cells = tuple(_validated_cell(cell) for cell in cells)
    if not normalized_cells:
        raise ValueError("ShareGPT latency quantile plots require at least one cell")
    per_case_rows = [_per_case_row(cell) for cell in normalized_cells]
    percentile_rows = [
        row
        for cell in normalized_cells
        for row in _latency_percentile_rows(cell)
    ]
    paths = {}
    present_hardware = {str(row["hardware_id"]) for row in per_case_rows}
    unknown_hardware = present_hardware - set(_HARDWARE_LABELS)
    if unknown_hardware:
        raise ValueError(f"unsupported ShareGPT hardware IDs: {sorted(unknown_hardware)}")
    for hardware_id in _HARDWARE_LABELS:
        hardware_rows = [
            row for row in per_case_rows if row["hardware_id"] == hardware_id
        ]
        if not hardware_rows:
            continue
        ordered_rows = sorted(hardware_rows, key=_case_sort_key)
        for metric_name, filename_metric in _PLOTTED_LATENCY_METRICS.items():
            path = destination / (
                f"{hardware_id}_{filename_prefix}_{filename_metric}_"
                "quantile_curves.png"
            )
            _write_latency_small_multiples(
                plt,
                path,
                hardware_id,
                metric_name,
                ordered_rows,
                percentile_rows,
            )
            paths[f"{hardware_id}_{filename_prefix}_{metric_name}"] = path
    return paths




def has_scheduler_progress_evidence(cells: Sequence[Mapping[str, object]]) -> bool:
    """A merged scheduler plot needs observed progress and predicted provenance for every case."""
    return bool(cells) and all(
        cell["prediction_status"] == "completed"
        and "observed_scheduler_progress" in cell
        and "forward_provenance" in cell
        for cell in cells
    )

def write_sharegpt_request_latency_plots(
    output_dir: str | Path,
    cells: Sequence[Mapping[str, object]],
    *,
    baseline_by_cell: Mapping[str, object] | None = None,
) -> Mapping[str, Path]:
    """Refresh standard paired/sorted latency figures without running prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    normalized = tuple(_validated_cell(cell, require_paired_latency=True) for cell in cells)
    if baseline_by_cell is not None and set(baseline_by_cell) != {cell["cell_id"] for cell in normalized}:
        raise ValueError("baseline statuses must cover exactly the standard case matrix")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for hardware_id in _HARDWARE_LABELS:
        hardware_cells = sorted([c for c in normalized if c["hardware_id"] == hardware_id], key=_case_sort_key)
        if not hardware_cells:
            continue
        for metric_name, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT")):
            path = destination / f"{hardware_id}_{metric_name[:-3]}_request_comparison.png"
            _write_latency_request_comparison_plot(plt, path, hardware_id, hardware_cells,
                                                  metric_name, label, baseline_by_cell=baseline_by_cell)
            paths[f"{hardware_id}_{metric_name}"] = path
    return paths

def _validated_cell(
    cell_value: Mapping[str, object], *, require_paired_latency: bool = False
) -> Mapping[str, object]:
    cell = _mapping(cell_value, "ShareGPT evaluation cell")
    required = {
        "cell_id",
        "hardware_id",
        "model_id",
        "case_id",
        "target_qps",
        "request_count",
        "evaluation_wall_time_seconds",
        "prediction_status",
        "performance_timing_status",
        "accuracy_claim_status",
        "metrics",
        "process_peak_rss_bytes",
    }
    missing = required - set(cell)
    if missing:
        raise ValueError(
            "ShareGPT evaluation cell is missing fields: "
            + ", ".join(sorted(missing))
        )
    for field in ("cell_id", "hardware_id", "model_id", "case_id"):
        if not isinstance(cell[field], str) or not cell[field]:
            raise TypeError(f"ShareGPT evaluation {field} must be non-empty text")
    if cell["hardware_id"] not in _HARDWARE_LABELS:
        raise ValueError(
            f"unknown ShareGPT evaluation hardware: {cell['hardware_id']}"
        )
    request_count = _integer(cell["request_count"], "request_count")
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    wall_seconds = _finite(cell["evaluation_wall_time_seconds"], "wall time")
    if wall_seconds < 0.0:
        raise ValueError("evaluation wall time must be nonnegative")
    peak_rss_bytes = _integer(
        cell["process_peak_rss_bytes"], "process peak RSS bytes")
    if peak_rss_bytes <= 0:
        raise ValueError("process peak RSS bytes must be positive")
    if cell["prediction_status"] not in (
        "completed",
        "unsupported_launch_coverage",
    ):
        raise ValueError(
            f"unknown prediction status: {cell['prediction_status']}"
        )
    metrics = _mapping(cell["metrics"], "ShareGPT cell metrics")
    for field in (
        "latency",
        "latency_quantiles_p0_p100",
        "throughput",
        "paired_request_error",
    ):
        if field not in metrics:
            raise ValueError(f"ShareGPT cell metrics is missing {field}")
    _validate_latency_quantile_comparison(
        _mapping(
            metrics["latency_quantiles_p0_p100"],
            "latency quantile comparison",
        ),
        completed=cell["prediction_status"] == "completed",
    )
    if require_paired_latency:
        if "paired_request_latency" not in metrics:
            raise ValueError(
                "ShareGPT cell metrics is missing paired_request_latency"
            )
        _validate_paired_request_latency(
            metrics["paired_request_latency"],
            request_count=request_count,
            completed=cell["prediction_status"] == "completed",
        )
        if "ttft_breakdown" not in metrics:
            raise ValueError("ShareGPT cell metrics is missing ttft_breakdown")
        missing_breakdown = (cell["prediction_status"] == "completed"
            and metrics["ttft_breakdown"] is None
            and "ttft_breakdown_unavailable_reason" in cell)
        if missing_breakdown:
            reason = cell["ttft_breakdown_unavailable_reason"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("unavailable TTFT breakdown requires an explicit evidence reason")
        else:
            _validate_ttft_breakdown(
                metrics["ttft_breakdown"],
                paired_latency=metrics["paired_request_latency"],
                request_count=request_count,
                completed=cell["prediction_status"] == "completed",
            )
    return cell


def _validate_paired_request_latency(
    value: object, *, request_count: int, completed: bool
) -> None:
    if not completed:
        if value is not None:
            raise ValueError("unsupported paired request latency must be null")
        return
    manifest = _mapping(value, "paired request latency")
    if set(manifest) != {
        "schema_version", "comparison_kind", "request_count", "metrics"
    }:
        raise ValueError("paired request latency fields are invalid")
    if (manifest["schema_version"] != 1
            or manifest["comparison_kind"]
            != "measured_vs_predicted_by_request_idx"
            or _integer(manifest["request_count"], "paired request count")
            != request_count):
        raise ValueError("paired request latency identity is invalid")
    metrics = _mapping(manifest["metrics"], "paired request latency metrics")
    if set(metrics) != set(_LATENCY_METRICS):
        raise ValueError("paired request latency metric set is invalid")
    for metric_name in _LATENCY_METRICS:
        records = _sequence(metrics[metric_name], f"paired {metric_name}")
        if len(records) != request_count:
            raise ValueError(f"paired {metric_name} does not cover every request")
        for expected_index, record_value in enumerate(records):
            record = _mapping(record_value, f"paired {metric_name} record")
            if set(record) != {"request_idx", "measured_ms", "predicted_ms"}:
                raise ValueError(f"paired {metric_name} record fields are invalid")
            if _integer(record["request_idx"], "paired request_idx") != expected_index:
                raise ValueError(f"paired {metric_name} request indexes are not contiguous")
            _finite(record["measured_ms"], f"paired {metric_name} measured")
            _finite(record["predicted_ms"], f"paired {metric_name} predicted")


def _validate_ttft_breakdown(
    value: object,
    *,
    paired_latency: object,
    request_count: int,
    completed: bool,
) -> None:
    if not completed:
        if value is not None:
            raise ValueError("unsupported TTFT breakdown must be null")
        return
    manifest = _mapping(value, "TTFT breakdown")
    if set(manifest) != {
        "schema_version", "boundary", "processing_scope",
        "measured_arrival_basis", "measured_negative_wait_clamp_count",
        "measured_minimum_raw_wait_ms", "request_count", "records",
    }:
        raise ValueError("TTFT breakdown fields are invalid")
    if (manifest["schema_version"] != 1
            or manifest["boundary"]
            != "request_arrival_to_first_schedule_to_first_token"
            or manifest["processing_scope"]
            != "first_schedule_to_client_visible_first_token"
            or manifest["measured_arrival_basis"]
            != "canonical_scheduled_arrival_time"
            or _integer(manifest["request_count"], "TTFT breakdown count")
            != request_count):
        raise ValueError("TTFT breakdown identity is invalid")
    clamp_count = _integer(
        manifest["measured_negative_wait_clamp_count"],
        "measured negative wait clamp count",
    )
    minimum_raw_wait = _finite(
        manifest["measured_minimum_raw_wait_ms"],
        "measured minimum raw wait",
    )
    if clamp_count < 0 or clamp_count > request_count:
        raise ValueError("measured negative wait clamp count is invalid")
    if (clamp_count == 0 and minimum_raw_wait < 0.0) or (
        clamp_count > 0 and minimum_raw_wait >= 0.0
    ):
        raise ValueError("measured wait clamp metadata is inconsistent")
    paired = _mapping(paired_latency, "paired request latency")
    paired_ttft = _sequence(
        _mapping(paired["metrics"], "paired request metrics")["ttft_ms"],
        "paired TTFT",
    )
    records = _sequence(manifest["records"], "TTFT breakdown records")
    if len(records) != request_count:
        raise ValueError("TTFT breakdown does not cover every request")
    fields = {
        "request_idx", "measured_wait_ms", "measured_processing_ms",
        "predicted_wait_ms", "predicted_processing_ms",
    }
    for request_idx, record_value in enumerate(records):
        record = _mapping(record_value, "TTFT breakdown record")
        if set(record) != fields:
            raise ValueError("TTFT breakdown record fields are invalid")
        if _integer(record["request_idx"], "TTFT breakdown request_idx") != request_idx:
            raise ValueError("TTFT breakdown request indexes are not contiguous")
        measured_total = _finite(
            record["measured_wait_ms"], "measured TTFT wait"
        ) + _finite(record["measured_processing_ms"], "measured TTFT processing")
        predicted_total = _finite(
            record["predicted_wait_ms"], "predicted TTFT wait"
        ) + _finite(record["predicted_processing_ms"], "predicted TTFT processing")
        paired_record = _mapping(paired_ttft[request_idx], "paired TTFT record")
        if not math.isclose(
            measured_total,
            _finite(paired_record["measured_ms"], "paired measured TTFT"),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("measured TTFT breakdown does not close")
        if not math.isclose(
            predicted_total,
            _finite(paired_record["predicted_ms"], "paired predicted TTFT"),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("predicted TTFT breakdown does not close")


def _validate_latency_quantile_comparison(
    value: Mapping[str, object],
    *,
    completed: bool,
) -> None:
    if set(value) != {"comparison_kind", "measured", "predicted"}:
        raise ValueError("latency quantile comparison fields are invalid")
    if value["comparison_kind"] != "measured_vs_predicted":
        raise ValueError("latency quantile comparison kind is invalid")
    _validate_canonical_quantile_projection(value["measured"], "measured")
    predicted = value["predicted"]
    if completed:
        _validate_canonical_quantile_projection(predicted, "predicted")
    elif predicted is not None:
        raise ValueError("unsupported prediction latency distribution must be null")


def _validate_canonical_quantile_projection(value: object, label: str) -> None:
    manifest = _mapping(value, f"{label} canonical quantile projection")
    expected_fields = {
        "schema_version",
        "source_schema_version",
        "quantile_representation",
        "percentile_method",
        *_LATENCY_DISTRIBUTION_FIELDS.values(),
    }
    if set(manifest) != expected_fields:
        raise ValueError(f"{label} latency quantile projection fields are invalid")
    if manifest["schema_version"] != 1 or manifest["source_schema_version"] != 2:
        raise ValueError(f"{label} latency quantile projection schema is unsupported")
    if manifest["quantile_representation"] != (
        "linear_interpolated_quantile_points_p0_p100"
    ):
        raise ValueError(f"{label} latency quantile representation is invalid")
    if manifest["percentile_method"] != (
        "linear_interpolation_rank_n_minus_1_v1"
    ):
        raise ValueError(f"{label} latency percentile method is invalid")
    for field in _LATENCY_DISTRIBUTION_FIELDS.values():
        distribution = _mapping(manifest[field], f"{label} {field}")
        if set(distribution) != {
            "availability",
            "eligible_request_count",
            "excluded_request_count",
            "exclusion_reason",
            "quantiles",
        }:
            raise ValueError(f"{label} {field} projection fields are invalid")
        quantiles = _mapping(
            distribution["quantiles"], f"{label} {field} quantiles"
        )
        percentiles = _sequence(
            quantiles["percentiles"], f"{label} {field} percentiles"
        )
        if tuple(percentiles) != tuple(range(101)):
            raise ValueError(f"{label} {field} must contain canonical P0-P100")
        _finite_series(
            quantiles["values_s"], f"{label} {field} quantile values", 101
        )


def _per_case_row(cell: Mapping[str, object]) -> dict[str, object]:
    metrics = _mapping(cell["metrics"], "ShareGPT cell metrics")
    latency = _mapping(metrics["latency"], "latency comparisons")
    throughput = _mapping(metrics["throughput"], "throughput comparisons")
    row = _identity_row(cell)
    row["evaluation_wall_time_seconds"] = _finite(
        cell["evaluation_wall_time_seconds"], "evaluation wall time"
    )
    for metric_name in (
        "request_throughput_per_s",
        "output_token_throughput_per_s",
        "total_token_throughput_per_s",
    ):
        comparison = _comparison(
            throughput[metric_name], f"{metric_name} comparison"
        )
        row[f"{metric_name}_measured"] = comparison["observed"]
        row[f"{metric_name}_predicted"] = comparison["predicted"]
        row[f"{metric_name}_absolute_percentage_error"] = comparison[
            "absolute_percentage_error"
        ]
    for metric_name in _LATENCY_METRICS:
        distribution = _mapping(latency[metric_name], metric_name)
        for statistic in STATISTICS:
            comparison = _comparison(
                distribution[statistic], f"{metric_name} {statistic}"
            )
            prefix = f"{metric_name}_{statistic}"
            row[f"{prefix}_measured"] = comparison["observed"]
            row[f"{prefix}_predicted"] = comparison["predicted"]
            row[f"{prefix}_absolute_percentage_error"] = comparison[
                "absolute_percentage_error"
            ]
    return row


def _latency_percentile_rows(
    cell: Mapping[str, object],
) -> list[dict[str, object]]:
    metrics = _mapping(cell["metrics"], "ShareGPT cell metrics")
    comparison = _mapping(
        metrics["latency_quantiles_p0_p100"], "latency quantile comparison"
    )
    measured_manifest = _mapping(comparison["measured"], "measured distributions")
    predicted_value = comparison["predicted"]
    predicted_manifest = (
        None
        if predicted_value is None
        else _mapping(predicted_value, "predicted distributions")
    )
    rows = []
    for metric_name, distribution_field in _LATENCY_DISTRIBUTION_FIELDS.items():
        measured_distribution = _mapping(
            measured_manifest[distribution_field], f"measured {distribution_field}"
        )
        measured_quantiles = _mapping(
            measured_distribution["quantiles"],
            f"measured {distribution_field} quantiles",
        )
        percentile_values = _sequence(
            measured_quantiles["percentiles"],
            f"measured {distribution_field} percentiles",
        )
        observed = _finite_series(
            measured_quantiles["values_s"], f"{metric_name} measured", 101
        )
        if predicted_manifest is None:
            predicted = None
        else:
            predicted_distribution = _mapping(
                predicted_manifest[distribution_field],
                f"predicted {distribution_field}",
            )
            predicted_quantiles = _mapping(
                predicted_distribution["quantiles"],
                f"predicted {distribution_field} quantiles",
            )
            if tuple(predicted_quantiles["percentiles"]) != tuple(percentile_values):
                raise ValueError(
                    f"{metric_name} measured and predicted percentile grids disagree"
                )
            predicted = _finite_series(
                predicted_quantiles["values_s"], f"{metric_name} predicted", 101
            )
        for index, percentile in enumerate(percentile_values):
            row = _identity_row(cell)
            row.update({
                "latency_metric": metric_name,
                "percentile": percentile,
                "measured_ms": observed[index] * 1000.0,
                "predicted_ms": (
                    None if predicted is None else predicted[index] * 1000.0
                ),
            })
            rows.append(row)
    return rows


def _speed_row(cell: Mapping[str, object]) -> dict[str, object]:
    metrics = _mapping(cell["metrics"], "ShareGPT cell metrics")
    throughput = _mapping(metrics["throughput"], "throughput comparisons")
    request_throughput = _comparison(
        throughput["request_throughput_per_s"],
        "request throughput comparison",
    )["observed"]
    measured_throughput = _finite(request_throughput, "measured request throughput")
    if measured_throughput <= 0.0:
        raise ValueError("measured request throughput must be positive")
    request_count = _integer(cell["request_count"], "request_count")
    wall_seconds = _finite(
        cell["evaluation_wall_time_seconds"], "evaluation wall time"
    )
    workload_seconds = request_count / measured_throughput
    row = {
        **_identity_row(cell),
        "request_count": request_count,
        "measured_request_throughput_per_s": measured_throughput,
        "observed_workload_seconds": workload_seconds,
        "evaluation_wall_time_seconds": wall_seconds,
        "wall_time_scope": "simulator_plus_comparison_cpu_wall",
    }
    if cell["prediction_status"] == "completed":
        row["speed_status"] = "available_completed_prediction"
        row["wall_seconds_per_observed_workload_second"] = (
            wall_seconds / workload_seconds
        )
        row["time_to_failure_seconds"] = None
    elif cell["prediction_status"] == "unsupported_launch_coverage":
        row["speed_status"] = "unavailable_prediction_unsupported"
        row["wall_seconds_per_observed_workload_second"] = None
        row["time_to_failure_seconds"] = wall_seconds
    else:
        raise ValueError(f"unknown speed status: {cell['prediction_status']}")
    return row


def _speed_summary_rows(
    speed_rows: Sequence[Mapping[str, object]],
    *,
    run_elapsed_wall_time_seconds: float,
) -> list[dict[str, object]]:
    hardware_ids = sorted({str(row["hardware_id"]) for row in speed_rows})
    rows = []
    for hardware_id in ("all", *hardware_ids):
        if hardware_id == "all":
            selected = list(speed_rows)
        elif hardware_id in hardware_ids:
            selected = [
                row for row in speed_rows if row["hardware_id"] == hardware_id
            ]
        else:
            raise ValueError(f"unknown speed summary hardware: {hardware_id}")
        completed = [
            row
            for row in selected
            if row["speed_status"] == "available_completed_prediction"
        ]
        unsupported = [
            row
            for row in selected
            if row["speed_status"] == "unavailable_prediction_unsupported"
        ]
        if len(completed) + len(unsupported) != len(selected):
            raise ValueError("speed summary contains unknown case status")
        wall_values = sorted(
            _finite(row["evaluation_wall_time_seconds"], "evaluation wall time")
            for row in completed
        )
        ratio_values = sorted(
            _finite(
                row["wall_seconds_per_observed_workload_second"],
                "simulation speed ratio",
            )
            for row in completed
        )
        failure_values = [
            _finite(row["time_to_failure_seconds"], "time to failure")
            for row in unsupported
        ]
        if hardware_id == "all":
            metrics_run_elapsed = run_elapsed_wall_time_seconds
            run_elapsed_scope = (
                "metrics_run_parent_wall_including_parallel_coordination"
            )
        elif hardware_id in hardware_ids:
            metrics_run_elapsed = None
            run_elapsed_scope = (
                "unavailable_per_hardware_cases_share_parallel_workers"
            )
        else:
            raise ValueError(f"unknown speed summary hardware: {hardware_id}")
        rows.append({
            "hardware_id": hardware_id,
            "case_count": len(selected),
            "completed_speed_case_count": len(completed),
            "unsupported_case_count": len(unsupported),
            "cumulative_parallel_worker_case_seconds": sum(wall_values),
            "cumulative_time_to_failure_seconds": sum(failure_values),
            "mean_completed_case_wall_time_seconds": (
                None if not wall_values else sum(wall_values) / len(wall_values)
            ),
            "evaluation_wall_time_p50_seconds": _percentile_or_none(
                wall_values, 50
            ),
            "evaluation_wall_time_p90_seconds": _percentile_or_none(
                wall_values, 90
            ),
            "evaluation_wall_time_p100_seconds": _percentile_or_none(
                wall_values, 100
            ),
            "wall_seconds_per_workload_second_p50": _percentile_or_none(
                ratio_values, 50
            ),
            "wall_seconds_per_workload_second_p90": _percentile_or_none(
                ratio_values, 90
            ),
            "wall_seconds_per_workload_second_p100": _percentile_or_none(
                ratio_values, 100
            ),
            "metrics_run_elapsed_wall_time_seconds": metrics_run_elapsed,
            "run_elapsed_scope": run_elapsed_scope,
            "wall_time_scope": "simulator_plus_comparison_cpu_wall",
        })
    return rows


def _memory_usage_row(cell: Mapping[str, object]) -> dict[str, object]:
    peak_rss_bytes = _integer(
        cell["process_peak_rss_bytes"], "process peak RSS bytes")
    if peak_rss_bytes <= 0:
        raise ValueError("process peak RSS bytes must be positive")
    return {
        **_identity_row(cell),
        "process_peak_rss_bytes": peak_rss_bytes,
        "process_peak_rss_gib": peak_rss_bytes / (1024 ** 3),
        "memory_scope": "isolated_case_process_peak_rss",
    }


def _memory_summary_rows(
    memory_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    hardware_ids = sorted({str(row["hardware_id"]) for row in memory_rows})
    selected_by_hardware = {
        "all": list(memory_rows),
        **{
            hardware_id: [
                row for row in memory_rows
                if row["hardware_id"] == hardware_id
            ]
            for hardware_id in hardware_ids
        },
    }
    rows = []
    for hardware_id, selected in selected_by_hardware.items():
        values = sorted(
            _integer(row["process_peak_rss_bytes"], "process peak RSS bytes")
            for row in selected
        )
        rows.append({
            "hardware_id": hardware_id,
            "case_count": len(selected),
            "mean_process_peak_rss_bytes": sum(values) / len(values),
            "process_peak_rss_p50_bytes": _percentile_or_none(values, 50),
            "process_peak_rss_p90_bytes": _percentile_or_none(values, 90),
            "process_peak_rss_p100_bytes": _percentile_or_none(values, 100),
            "memory_scope": "isolated_case_process_peak_rss",
        })
    return rows


def _midlayer_wall_time_rows(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = []
    expected_phases = (
        "workload_construction",
        "kernel_lowering",
        "kernel_prediction",
        "timeline_construction",
        "artifact_generation",
        "scheduler_and_orchestration",
    )
    for cell in cells:
        if cell["prediction_status"] != "completed":
            continue
        if "forward_provenance" not in cell:
            continue
        provenance = _mapping(cell["forward_provenance"], "forward provenance")
        timing = _mapping(provenance["midlayer_wall_time"], "Midlayer wall time")
        if set(timing) != {
            "schema_version", "scope", "simulation_wall_time_s",
            "phase_wall_time_s",
        }:
            raise ValueError("Midlayer wall time fields are invalid")
        if (timing["schema_version"] != 1
                or timing["scope"] != "midlayer_simulation_cpu_wall"):
            raise ValueError("Midlayer wall time identity is invalid")
        phases = _mapping(timing["phase_wall_time_s"], "Midlayer wall phases")
        if set(phases) != set(expected_phases):
            raise ValueError("Midlayer wall time phase fields are invalid")
        simulation_wall = _finite(
            timing["simulation_wall_time_s"], "simulation wall time"
        )
        phase_values = {
            name: _finite(phases[name], f"{name} wall time")
            for name in expected_phases
        }
        evaluation_wall = _finite(
            cell["evaluation_wall_time_seconds"], "evaluation wall time"
        )
        rows.append({
            **_identity_row(cell),
            "scope": timing["scope"],
            "simulation_wall_time_s": simulation_wall,
            **{f"{name}_s": value for name, value in phase_values.items()},
            "validation_and_comparison_wall_time_s": max(
                0.0, evaluation_wall - simulation_wall
            ),
            "case_evaluation_wall_time_s": evaluation_wall,
        })
    return rows


def _host_timing_contribution_rows(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return same-schedule Device, Launch, and Scheduler contributions."""

    rows = []
    full_mode = "device_plus_launch_plus_scheduler"
    for cell in cells:
        if cell["prediction_status"] != "completed":
            continue
        if "serving_timing_mode" not in cell:
            continue
        mode = cell["serving_timing_mode"]
        if mode != full_mode:
            continue
        timing = _mapping(cell["host_timing"], "Host timing contribution")
        if timing["timing_mode"] != full_mode:
            raise ValueError("Host timing mode disagrees with evaluation cell")
        aggregate = _mapping(
            timing["aggregate"], "Host timing contribution aggregate"
        )
        device_s = _finite(
            aggregate["predicted_device_s"], "predicted Device time"
        )
        execution_s = _finite(
            aggregate["execution_wall_s"], "execution wall time"
        )
        scheduler_s = _finite(
            aggregate["scheduler_cpu_s"], "Scheduler CPU time"
        )
        step_wall_s = _finite(
            aggregate["step_wall_s"], "summed active step time"
        )
        makespan_s = _finite(
            aggregate["schedule_makespan_s"], "schedule makespan"
        )
        launch_s = execution_s - device_s
        tolerance = max(1e-12, abs(execution_s) * 1e-12)
        if launch_s < -tolerance:
            raise ValueError(
                "Host launch timing cannot reduce predicted Device time"
            )
        launch_s = max(0.0, launch_s)
        rows.append({
            **_identity_row(cell),
            "contribution_basis": (
                "realized_full_mode_non_additive_async_timing_views"
            ),
            "device_active_s": device_s,
            "launch_critical_path_increment_s": launch_s,
            "scheduler_cpu_increment_s": scheduler_s,
            "active_step_total_s": step_wall_s,
            "schedule_makespan_s": makespan_s,
        })
    return rows


def _percentile_or_none(
    sorted_values: Sequence[float],
    percentile: float,
) -> float | None:
    if not sorted_values:
        return None
    return linear_percentile(sorted_values, percentile)


def _write_plots(
    plot_dir: Path,
    cells: Sequence[Mapping[str, object]],
    per_case_rows: Sequence[Mapping[str, object]],
    percentile_rows: Sequence[Mapping[str, object]],
    speed_rows: Sequence[Mapping[str, object]],
    memory_rows: Sequence[Mapping[str, object]],
    host_timing_rows: Sequence[Mapping[str, object]],
    *,
    plot_jobs: int = 1,
) -> Mapping[str, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ShareGPT evaluation report rendering requires matplotlib"
        ) from error

    paths = {}
    present_hardware = {str(row["hardware_id"]) for row in per_case_rows}
    hardware_ids = [
        hardware_id for hardware_id in _HARDWARE_LABELS
        if hardware_id in present_hardware
    ]
    overview_path = plot_dir / "three_hardware_serving_overview.png"
    _write_compact_serving_overview_plot(
        plt, overview_path, hardware_ids, per_case_rows
    )
    paths["plot_three_hardware_serving_overview"] = overview_path
    case_wall_path = plot_dir / "serving_case_evaluation_wall_time.png"
    _write_serving_case_wall_time_plot(
        plt, case_wall_path, hardware_ids, speed_rows
    )
    paths["plot_serving_case_evaluation_wall_time"] = case_wall_path
    case_memory_path = plot_dir / "serving_case_peak_rss.png"
    _write_serving_case_peak_rss_plot(
        plt, case_memory_path, hardware_ids, memory_rows
    )
    paths["plot_serving_case_peak_rss"] = case_memory_path
    tasks = [(
        plot_dir, hardware_id,
        sorted([cell for cell in cells if cell["hardware_id"] == hardware_id],
            key=_case_sort_key),
        [row for row in speed_rows if row["hardware_id"] == hardware_id],
    ) for hardware_id in hardware_ids]
    workers = min(plot_jobs, len(tasks))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        # Matplotlib is not thread-safe. Each spawned process owns its figures;
        # map preserves the existing hardware order and propagates worker errors.
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            for group_paths in executor.map(_write_hardware_plot_group, tasks):
                paths.update(group_paths)
    else:
        for task in tasks:
            paths.update(_write_hardware_plot_group(task))

    if host_timing_rows:
        overhead_path = (
            plot_dir / "three_hardware_host_timing_overhead_percent.png"
        )
        _write_host_timing_overhead_percent_plot(
            plt, overhead_path, hardware_ids, host_timing_rows
        )
        paths["plot_three_hardware_host_timing_overhead_percent"] = overhead_path
    return paths


def _write_hardware_plot_group(task) -> Mapping[str, Path]:
    """Render the unchanged figures for one independent hardware group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir, hardware_id, hardware_cells, speed_rows = task
    paths = {}
    three_mode_flags = ["timing_mode_results" in cell for cell in hardware_cells]
    if any(three_mode_flags):
        if not all(three_mode_flags):
            raise ValueError(
                f"{hardware_id} report cannot mix three-mode and legacy cells"
            )
        hardware_speed = [
            row for row in speed_rows if row["hardware_id"] == hardware_id
        ]
        throughput_path = plot_dir / (
            f"{hardware_id}_throughput_and_simulation_time.png"
        )
        _write_three_mode_throughput_and_speed_plot(
            plt,
            throughput_path,
            hardware_id,
            hardware_cells,
            hardware_speed,
        )
        paths[
            f"plot_{hardware_id}_throughput_and_simulation_time"
        ] = throughput_path
        for metric_name, filename_metric in _PLOTTED_LATENCY_METRICS.items():
            latency_path = plot_dir / (
                f"{hardware_id}_{filename_metric}_quantile_curves.png"
            )
            _write_three_mode_latency_quantile_curves_plot(
                plt,
                latency_path,
                hardware_id,
                hardware_cells,
                metric_name,
            )
            paths[
                f"plot_{hardware_id}_{filename_metric}_quantile_curves"
            ] = latency_path
        for cell in hardware_cells:
            if "observed_scheduler_progress" not in cell:
                continue
            scheduler_path = plot_dir / (
                f"{_plot_slug(str(cell['cell_id']))}_scheduler_steps.png"
            )
            _write_scheduler_comparison_plot(plt, scheduler_path, cell)
            paths[
                f"plot_{_plot_slug(str(cell['cell_id']))}_scheduler_steps"
            ] = scheduler_path
        return paths
    for metric_name, metric_label in (
        ("ttft_ms", "TTFT"),
        ("tpot_ms", "TPOT"),
    ):
        latency_path = plot_dir / (
            f"{hardware_id}_{metric_name[:-3]}_request_comparison.png"
        )
        _write_latency_request_comparison_plot(
            plt,
            latency_path,
            hardware_id,
            hardware_cells,
            metric_name,
            metric_label,
        )
        paths[
            f"plot_{hardware_id}_{metric_name[:-3]}_request_comparison"
        ] = latency_path
    breakdown_path = plot_dir / f"{hardware_id}_ttft_breakdown.png"
    _write_ttft_breakdown_plot(
        plt, breakdown_path, hardware_id, hardware_cells
    )
    paths[f"plot_{hardware_id}_ttft_breakdown"] = breakdown_path
    component_path = (
        plot_dir / f"{hardware_id}_ttft_component_comparison.png"
    )
    _write_ttft_component_comparison_plot(
        plt, component_path, hardware_id, hardware_cells
    )
    paths[f"plot_{hardware_id}_ttft_component_comparison"] = component_path

    if has_scheduler_progress_evidence(hardware_cells):
        scheduler_path = plot_dir / f"{hardware_id}_scheduler_progress.png"
        _write_scheduler_progress_grid(
            plt, scheduler_path, hardware_id, hardware_cells
        )
        paths[f"plot_{hardware_id}_scheduler_progress"] = scheduler_path

    return paths


def _write_compact_serving_overview_plot(
    plt: object,
    path: Path,
    hardware_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    metrics = (
        ("request_throughput_per_s", "Req Throughput", "requests/s"),
        ("ttft_ms_mean", "Mean TTFT", "ms"),
        ("tpot_ms_mean", "Mean TPOT", "ms/token"),
    )
    figure, axes = plt.subplots(
        len(metrics), len(hardware_ids),
        figsize=(5.8 * len(hardware_ids), 11.8),
        squeeze=False,
    )
    for column, hardware_id in enumerate(hardware_ids):
        hardware_rows = sorted(
            [row for row in rows if row["hardware_id"] == hardware_id],
            key=_case_sort_key,
        )
        positions = list(range(len(hardware_rows)))
        labels = [_case_label(row) for row in hardware_rows]
        for row_index, (prefix, metric_label, unit) in enumerate(metrics):
            axis = axes[row_index, column]
            measured = [
                _finite(row[f"{prefix}_measured"], f"{prefix} measured")
                for row in hardware_rows
            ]
            predicted = [
                math.nan
                if row[f"{prefix}_predicted"] is None
                else _finite(row[f"{prefix}_predicted"], f"{prefix} predicted")
                for row in hardware_rows
            ]
            axis.bar(
                [position - 0.19 for position in positions], measured,
                width=0.38, label="Measured", color=_MEASURED_COLOR,
                edgecolor="white", linewidth=0.35,
            )
            axis.bar(
                [position + 0.19 for position in positions], predicted,
                width=0.38, label="Predicted", color=_PREDICTED_COLOR,
                edgecolor="white", linewidth=0.35,
            )
            axis.set_ylim(bottom=0.0)
            axis.set_ylabel(f"{metric_label} ({unit})")
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#d5d5d5", linewidth=0.65, alpha=0.55)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(
                f"{_HARDWARE_LABELS[hardware_id]} {metric_label}",
                fontsize=10.5,
                pad=6,
            )
            axis.set_xticks(positions, labels, rotation=60, ha="right")
            axis.tick_params(axis="x", labelsize=6.2, pad=1)
            axis.margins(x=0.015)
            relative_error = [
                math.nan
                if (predicted_value != predicted_value or measured_value == 0.0)
                else (predicted_value - measured_value) / measured_value * 100.0
                for predicted_value, measured_value in zip(predicted, measured)
            ]
            error_axis = axis.twinx()
            error_axis.axhline(
                0.0, color=_RELATIVE_ERROR_COLOR, linewidth=0.7,
                alpha=0.35, zorder=1,
            )
            error_axis.plot(
                positions, relative_error,
                color=_RELATIVE_ERROR_COLOR, linewidth=1.35,
                marker="o", markersize=3.1, zorder=3,
                label="Signed relative error",
            )
            finite_error = [
                value for value in relative_error if value == value
            ]
            if finite_error:
                span = max(abs(value) for value in finite_error)
                span = max(span * 1.25, 5.0)
                error_axis.set_ylim(-span, span)
            error_axis.set_ylabel(
                "Signed relative error (%)",
                color=_RELATIVE_ERROR_COLOR, fontsize=8.5,
            )
            error_axis.tick_params(
                axis="y", labelsize=7.0, colors=_RELATIVE_ERROR_COLOR,
            )
            error_axis.spines["top"].set_visible(False)
            error_axis.spines["right"].set_color(_RELATIVE_ERROR_COLOR)
            if row_index == 0 and column == 0:
                error_handles, error_labels = (
                    error_axis.get_legend_handles_labels())
                axis._midlayer_error_legend = (error_handles, error_labels)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    error_legend = getattr(axes[0, 0], "_midlayer_error_legend", None)
    if error_legend is not None:
        handles = list(handles) + list(error_legend[0])
        legend_labels = list(legend_labels) + list(error_legend[1])
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=_LEGEND_FONTSIZE,
    )
    figure.suptitle(
        "Full Serving absolute measured vs predicted "
        "(bars, left axis) and signed relative error (line, right axis)",
        y=0.995,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.subplots_adjust(
        left=0.06, right=0.995, top=0.895, bottom=0.17,
        hspace=0.54, wspace=0.28,
    )
    figure.savefig(path, dpi=_OVERVIEW_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_serving_case_wall_time_plot(
    plt: object,
    path: Path,
    hardware_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    case_keys = []
    for row in sorted(rows, key=_case_sort_key):
        key = (str(row["model_id"]), str(row["case_id"]))
        if key not in case_keys:
            case_keys.append(key)
    positions = list(range(len(case_keys)))
    colors = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")
    width = 0.24
    figure, axis = plt.subplots(figsize=(16.5, 5.8), constrained_layout=True)
    for hardware_index, hardware_id in enumerate(hardware_ids):
        by_case = {
            (str(row["model_id"]), str(row["case_id"])): row
            for row in rows if row["hardware_id"] == hardware_id
        }
        values = [
            math.nan
            if case_key not in by_case
            else _finite(
                by_case[case_key]["evaluation_wall_time_seconds"],
                "Serving case evaluation wall time",
            )
            for case_key in case_keys
        ]
        bars = axis.bar(
            [
                position + (hardware_index - (len(hardware_ids) - 1) / 2) * width
                for position in positions
            ],
            values,
            width=width,
            color=colors[hardware_index],
            label=_HARDWARE_LABELS[hardware_id],
            edgecolor="white",
            linewidth=0.35,
        )
        axis.bar_label(
            bars,
            labels=[
                "" if math.isnan(value) else f"{value:,.0f} s"
                for value in values
            ],
            padding=2,
            fontsize=7.2,
            rotation=90,
        )
    axis.set_yscale("log")
    axis.set_ylabel("case worker wall time (seconds, log scale)")
    axis.set_xticks(
        positions,
        [f"{model}\n{case}" for model, case in case_keys],
        rotation=45,
        ha="right",
    )
    axis.set_title(
        "Full Serving evaluation time by case (parallel worker wall time)"
    )
    axis.set_axisbelow(True)
    axis.grid(
        axis="y", which="both", color="#d0d0d0", linewidth=0.65, alpha=0.58
    )
    axis.margins(y=0.18)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        ncol=len(hardware_ids), loc="upper center",
        fontsize=_LEGEND_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_OVERVIEW_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_serving_case_peak_rss_plot(
    plt: object,
    path: Path,
    hardware_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    case_keys = []
    for row in sorted(rows, key=_case_sort_key):
        key = (str(row["model_id"]), str(row["case_id"]))
        if key not in case_keys:
            case_keys.append(key)
    positions = list(range(len(case_keys)))
    colors = ("#4c78a8", "#f58518", "#54a24b", "#b279a2")
    width = 0.24
    figure, axis = plt.subplots(figsize=(16.5, 5.8), constrained_layout=True)
    for hardware_index, hardware_id in enumerate(hardware_ids):
        by_case = {
            (str(row["model_id"]), str(row["case_id"])): row
            for row in rows if row["hardware_id"] == hardware_id
        }
        values = [
            math.nan
            if case_key not in by_case
            else _finite(
                by_case[case_key]["process_peak_rss_gib"],
                "Serving case peak RSS",
            )
            for case_key in case_keys
        ]
        bars = axis.bar(
            [
                position + (hardware_index - (len(hardware_ids) - 1) / 2) * width
                for position in positions
            ],
            values,
            width=width,
            color=colors[hardware_index],
            label=_HARDWARE_LABELS[hardware_id],
            edgecolor="white",
            linewidth=0.35,
        )
        axis.bar_label(
            bars,
            labels=[
                "" if math.isnan(value) else f"{value:,.2f} GiB"
                for value in values
            ],
            padding=2,
            fontsize=7.2,
            rotation=90,
        )
    axis.set_yscale("log")
    axis.set_ylabel("isolated case process peak RSS (GiB, log scale)")
    axis.set_xticks(
        positions,
        [f"{model}\n{case}" for model, case in case_keys],
        rotation=45,
        ha="right",
    )
    axis.set_title("Full Serving peak memory by case")
    axis.set_axisbelow(True)
    axis.grid(
        axis="y", which="both", color="#d0d0d0", linewidth=0.65, alpha=0.58
    )
    axis.margins(y=0.18)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        ncol=len(hardware_ids), loc="upper center",
        fontsize=_LEGEND_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_OVERVIEW_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_latency_request_comparison_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
    metric_name: str,
    metric_label: str,
    *,
    baseline_by_cell: Mapping[str, object] | None = None,
) -> None:
    case_columns = min(3, len(cells))
    row_count = math.ceil(len(cells) / case_columns)
    figure, axes = plt.subplots(
        row_count,
        case_columns * 2,
        figsize=(4.1 * case_columns * 2, 3.75 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for case_index, cell in enumerate(cells):
        row_index = case_index // case_columns
        pair_column = (case_index % case_columns) * 2
        scatter_axis = axes[row_index, pair_column]
        sorted_axis = axes[row_index, pair_column + 1]
        case_title = _case_label(cell).replace("\n", " / ")
        baseline = None if baseline_by_cell is None else baseline_by_cell[str(cell["cell_id"])]
        native = (baseline["metrics_ms"][metric_name]
                  if baseline is not None and baseline["status"] == "available" else None)
        if baseline is not None and native is None:
            sorted_axis.text(0.03, 0.97, f"LLMServingSim 2.0: {baseline['status']}",
                             transform=sorted_axis.transAxes, ha="left", va="top", fontsize=8)
        paired_value = _mapping(cell["metrics"], "cell metrics")[
            "paired_request_latency"
        ]
        if paired_value is None:
            for axis, plot_type in (
                (scatter_axis, "Paired request scatter"),
                (sorted_axis, "Sorted distribution"),
            ):
                axis.text(
                    0.5, 0.5, "Prediction unsupported",
                    transform=axis.transAxes, ha="center", va="center",
                )
                axis.set_title(f"{case_title}\n{plot_type}", fontsize=10.5)
                axis.set_box_aspect(1.0)
            continue
        paired = _mapping(paired_value, "paired request latency")
        values = _sequence(
            _mapping(paired["metrics"], "paired latency metrics")[metric_name],
            f"paired {metric_name}",
        )
        measured = [
            _finite(
                _mapping(value, "paired latency")["measured_ms"],
                "measured latency",
            )
            for value in values
        ]
        predicted = [
            _finite(
                _mapping(value, "paired latency")["predicted_ms"],
                "predicted latency",
            )
            for value in values
        ]
        extra = [] if native is None else native
        if native is not None and len(native) != len(measured):
            raise ValueError("baseline paired metric count differs")
        upper = max(1.0, max((*measured, *predicted, *extra)) * 1.03)
        scatter_axis.scatter(
            measured, predicted, s=11, alpha=0.55,
            color=_PREDICTED_COLOR, edgecolors="none", rasterized=True, label="Midlayer",
        )
        if native is not None:
            scatter_axis.scatter(measured, native, s=13, alpha=0.6, marker="x",
                                 color="#009E73", label="LLMServingSim 2.0", rasterized=True)
            scatter_axis.legend(loc="lower right", fontsize=8)
        scatter_axis.plot(
            [0.0, upper], [0.0, upper], color="#666666",
            linestyle="--", linewidth=1.0,
        )
        scatter_axis.set_xlim(0.0, upper)
        scatter_axis.set_ylim(0.0, upper)
        scatter_axis.set_aspect("equal", adjustable="box")
        percentage_errors = [
            abs(predicted_value - measured_value) / measured_value * 100.0
            for measured_value, predicted_value
            in zip(measured, predicted, strict=True)
            if measured_value != 0.0
        ]
        if not percentage_errors:
            raise ValueError("paired latency scatter has no positive measurements")
        scatter_axis.text(
            0.04, 0.96,
            f"n={len(values)}  Midlayer MAPE={sum(percentage_errors) / len(percentage_errors):.1f}%",
            transform=scatter_axis.transAxes,
            ha="left", va="top", fontsize=8.5,
        )
        percentiles = (
            [0.0]
            if len(values) == 1
            else [
                index / (len(values) - 1) * 100.0
                for index in range(len(values))
            ]
        )
        sorted_axis.plot(
            percentiles, sorted(measured), color=_MEASURED_COLOR,
            linewidth=1.5, label="Measured",
        )
        sorted_axis.plot(
            percentiles, sorted(predicted), color=_PREDICTED_COLOR,
            linewidth=1.5, label="Midlayer",
        )
        if native is not None:
            sorted_axis.plot(percentiles, sorted(native), color="#009E73",
                             linestyle="--", linewidth=1.7, label="LLMServingSim 2.0")
        sorted_axis.set_xlim(0.0, 100.0)
        sorted_axis.set_ylim(0.0, upper)
        sorted_axis.set_box_aspect(1.0)
        scatter_axis.set_title(
            f"{case_title}\nPaired request scatter", fontsize=10.5, pad=7
        )
        sorted_axis.set_title(
            f"{case_title}\nSorted distribution", fontsize=10.5, pad=7
        )
        scatter_axis.set_xlabel(f"Measured {metric_label} (ms)")
        scatter_axis.set_ylabel(f"Predicted {metric_label} (ms)")
        sorted_axis.set_xlabel("Cumulative requests (%)")
        sorted_axis.set_ylabel(f"{metric_label} (ms)")
        scatter_axis.grid(alpha=0.25)
        sorted_axis.grid(alpha=0.25)
        sorted_axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
    for unused_index in range(len(cells), row_count * case_columns):
        row_index = unused_index // case_columns
        pair_column = (unused_index % case_columns) * 2
        axes[row_index, pair_column].set_visible(False)
        axes[row_index, pair_column + 1].set_visible(False)
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} {metric_label}: paired requests and sorted distribution",
        y=1.025,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_DENSE_PLOT_DPI, bbox_inches="tight")
    if baseline_by_cell is not None:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _write_ttft_breakdown_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
) -> None:
    case_columns = min(3, len(cells))
    row_count = math.ceil(len(cells) / case_columns)
    figure, axes = plt.subplots(
        row_count, case_columns * 2,
        figsize=(4.1 * case_columns * 2, 3.75 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    component_colors = {
        "measured": (_MEASURED_COLOR, "#b8cee5"),
        "predicted": (_PREDICTED_COLOR, "#f8c58f"),
    }
    for case_index, cell in enumerate(cells):
        row_index = case_index // case_columns
        pair_column = (case_index % case_columns) * 2
        measured_axis = axes[row_index, pair_column]
        predicted_axis = axes[row_index, pair_column + 1]
        case_title = _case_label(cell).replace("\n", " / ")
        breakdown_value = _mapping(cell["metrics"], "cell metrics")[
            "ttft_breakdown"
        ]
        if breakdown_value is None:
            for axis, state_label in (
                (measured_axis, "Measured composition"),
                (predicted_axis, "Predicted composition"),
            ):
                axis.text(
                    0.5, 0.5, ("Observed TTFT split unavailable\nSee case evidence reason"
                        if cell["prediction_status"] == "completed" else "Prediction unsupported"),
                    transform=axis.transAxes, ha="center", va="center",
                )
                axis.set_title(
                    f"{case_title}\n{state_label}", fontsize=10.5, pad=7
                )
                axis.set_box_aspect(1.0)
            continue
        breakdown = _mapping(breakdown_value, "TTFT breakdown")
        records = _sequence(breakdown["records"], "TTFT breakdown records")
        if not records:
            raise ValueError("TTFT breakdown records cannot be empty")
        ordered_records = sorted(
            (_mapping(record, "TTFT breakdown record") for record in records),
            key=lambda record: _finite(
                record["measured_wait_ms"], "measured TTFT wait"
            ) + _finite(
                record["measured_processing_ms"],
                "measured TTFT processing",
            ),
        )
        count = len(ordered_records)
        percentiles = (
            [0.0]
            if count == 1
            else [index / (count - 1) * 100.0 for index in range(count)]
        )
        values: dict[str, dict[str, list[float]]] = {}
        for prefix in ("measured", "predicted"):
            values[prefix] = {
                part: [
                    _finite(
                        _mapping(record, "TTFT breakdown record")[
                            f"{prefix}_{part}_ms"
                        ],
                        f"{prefix} TTFT {part}",
                    )
                    for record in ordered_records
                ]
                for part in ("wait", "processing")
            }
        totals_by_prefix = {
            prefix: [
                wait + processing
                for wait, processing in zip(
                    values[prefix]["wait"],
                    values[prefix]["processing"],
                    strict=True,
                )
            ]
            for prefix in ("measured", "predicted")
        }
        upper = max(
            1.0,
            max((*totals_by_prefix["measured"], *totals_by_prefix["predicted"]))
            * 1.03,
        )
        for prefix, axis, state_label in (
            ("measured", measured_axis, "Measured composition"),
            ("predicted", predicted_axis, "Predicted composition"),
        ):
            waits = values[prefix]["wait"]
            totals = totals_by_prefix[prefix]
            wait_color, processing_color = component_colors[prefix]
            axis.fill_between(
                percentiles, 0.0, waits,
                step="post", color=wait_color, alpha=0.82,
                label=f"{prefix.title()} wait",
            )
            axis.fill_between(
                percentiles, waits, totals,
                step="post", color=processing_color, alpha=0.92,
                label=f"{prefix.title()} processing",
            )
            axis.step(
                percentiles, waits, where="post", color=wait_color,
                linewidth=1.2, linestyle="-",
            )
            axis.step(
                percentiles, totals, where="post", color=wait_color,
                linewidth=1.5,
            )
            axis.set_xlim(0.0, 100.0)
            axis.set_ylim(0.0, upper)
            axis.set_box_aspect(1.0)
            axis.set_title(
                f"{case_title}\n{state_label}", fontsize=10.5, pad=7
            )
            axis.set_xlabel("Requests sorted by measured TTFT (%)")
            axis.set_ylabel("TTFT (ms)")
            axis.grid(alpha=0.25)
            axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
        mean_wait_error = (
            sum(values["predicted"]["wait"])
            - sum(values["measured"]["wait"])
        ) / count
        mean_processing_error = (
            sum(values["predicted"]["processing"])
            - sum(values["measured"]["processing"])
        ) / count
        predicted_axis.text(
            0.04, 0.96,
            f"mean signed error (ms)\n"
            f"Wait {mean_wait_error:+.1f}\n"
            f"Processing {mean_processing_error:+.1f}\n"
            f"TTFT {mean_wait_error + mean_processing_error:+.1f}",
            transform=predicted_axis.transAxes,
            ha="left", va="top", fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    for unused_index in range(len(cells), row_count * case_columns):
        row_index = unused_index // case_columns
        pair_column = (unused_index % case_columns) * 2
        axes[row_index, pair_column].set_visible(False)
        axes[row_index, pair_column + 1].set_visible(False)
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} TTFT per-request composition",
        y=1.025,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_DENSE_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_ttft_component_comparison_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
) -> None:
    """Compare independently sorted Wait and Processing distributions."""

    case_columns = min(3, len(cells))
    row_count = math.ceil(len(cells) / case_columns)
    figure, axes = plt.subplots(
        row_count, case_columns * 2,
        figsize=(4.1 * case_columns * 2, 3.75 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for case_index, cell in enumerate(cells):
        row_index = case_index // case_columns
        pair_column = (case_index % case_columns) * 2
        case_title = _case_label(cell).replace("\n", " / ")
        breakdown_value = _mapping(cell["metrics"], "cell metrics")[
            "ttft_breakdown"
        ]
        if breakdown_value is None:
            for part_index, part_label in enumerate(("Processing", "Wait")):
                axis = axes[row_index, pair_column + part_index]
                axis.text(
                    0.5, 0.5, ("Observed TTFT split unavailable\nSee case evidence reason"
                        if cell["prediction_status"] == "completed" else "Prediction unsupported"),
                    transform=axis.transAxes, ha="center", va="center",
                )
                axis.set_title(
                    f"{case_title}\n{part_label}", fontsize=10.5, pad=7
                )
                axis.set_box_aspect(1.0)
            continue
        records = [
            _mapping(record, "TTFT breakdown record")
            for record in _sequence(
                _mapping(breakdown_value, "TTFT breakdown")["records"],
                "TTFT breakdown records",
            )
        ]
        if not records:
            raise ValueError("TTFT breakdown records cannot be empty")
        for part_index, (part, part_label) in enumerate((
            ("processing", "Processing"),
            ("wait", "Wait"),
        )):
            axis = axes[row_index, pair_column + part_index]
            line_style = {"processing": "-", "wait": "--"}[part]
            measured = sorted([
                _finite(
                    record[f"measured_{part}_ms"],
                    f"measured TTFT {part}",
                )
                for record in records
            ])
            predicted = sorted([
                _finite(
                    record[f"predicted_{part}_ms"],
                    f"predicted TTFT {part}",
                )
                for record in records
            ])
            percentiles = (
                [0.0]
                if len(records) == 1
                else [
                    index / (len(records) - 1) * 100.0
                    for index in range(len(records))
                ]
            )
            axis.step(
                percentiles, measured, where="post",
                color=_MEASURED_COLOR, linewidth=1.5,
                linestyle=line_style, label="Measured",
            )
            axis.step(
                percentiles, predicted, where="post",
                color=_PREDICTED_COLOR, linewidth=1.5,
                linestyle=line_style, label="Predicted",
            )
            mean_error = sum(
                predicted_value - measured_value
                for measured_value, predicted_value in zip(
                    measured, predicted, strict=True
                )
            ) / len(records)
            axis.text(
                0.04, 0.96, f"mean signed error {mean_error:+.1f} ms",
                transform=axis.transAxes,
                ha="left", va="top", fontsize=8.5,
                bbox={
                    "facecolor": "white", "alpha": 0.78,
                    "edgecolor": "none",
                },
            )
            axis.set_xlim(0.0, 100.0)
            axis.set_ylim(
                0.0, max(1.0, max((*measured, *predicted)) * 1.03)
            )
            axis.set_box_aspect(1.0)
            axis.set_title(
                f"{case_title}\n{part_label} · independently sorted",
                fontsize=10.5,
                pad=7,
            )
            axis.set_xlabel("Request percentile (%)")
            axis.set_ylabel(f"{part_label} time (ms)")
            axis.grid(alpha=0.25)
            axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
    for unused_index in range(len(cells), row_count * case_columns):
        row_index = unused_index // case_columns
        pair_column = (unused_index % case_columns) * 2
        axes[row_index, pair_column].set_visible(False)
        axes[row_index, pair_column + 1].set_visible(False)
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} TTFT component-first comparison",
        y=1.025,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_DENSE_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_scheduler_progress_grid(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
) -> None:
    case_columns = min(3, len(cells))
    row_count = math.ceil(len(cells) / case_columns)
    figure, axes = plt.subplots(
        row_count, case_columns * 2,
        figsize=(4.1 * case_columns * 2, 3.7 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for case_index, cell in enumerate(cells):
        row_index = case_index // case_columns
        pair_column = (case_index % case_columns) * 2
        token_axis = axes[row_index, pair_column]
        time_axis = axes[row_index, pair_column + 1]
        observed = _scheduler_series(
            _mapping(
                cell["observed_scheduler_progress"],
                "observed scheduler progress",
            ),
            "observed",
        )
        predicted = _scheduler_series(
            _mapping(
                _mapping(cell["forward_provenance"], "forward provenance")[
                    "scheduler_progress"
                ],
                "predicted scheduler progress",
            ),
            "predicted full",
        )
        for label, series, color in (
            ("Measured", observed, _MEASURED_COLOR),
            ("Predicted", predicted, _PREDICTED_COLOR),
        ):
            steps = [entry["step_index"] for entry in series]
            token_axis.step(
                steps,
                [entry["cumulative_scheduled_tokens"] for entry in series],
                where="post", label=label, color=color, linewidth=1.5,
            )
            time_axis.plot(
                steps,
                [entry["cumulative_elapsed_ms"] for entry in series],
                label=label, color=color, linewidth=1.5, linestyle="--",
            )
        case_title = _case_label(cell).replace("\n", " / ")
        token_axis.set_title(
            f"{case_title}\nCumulative tokens · step lines",
            fontsize=10.5,
            pad=7,
        )
        time_axis.set_title(
            f"{case_title}\nCumulative progress time · dashed lines",
            fontsize=10.5,
            pad=7,
        )
        token_axis.set_xlabel("Scheduler step")
        time_axis.set_xlabel("Scheduler step")
        token_axis.set_ylabel("scheduled tokens")
        time_axis.set_ylabel("milliseconds")
        token_axis.set_box_aspect(1.0)
        time_axis.set_box_aspect(1.0)
        token_axis.grid(alpha=0.25)
        time_axis.grid(alpha=0.25)
        token_axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
        time_axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
    for unused_index in range(len(cells), row_count * case_columns):
        row_index = unused_index // case_columns
        pair_column = (unused_index % case_columns) * 2
        axes[row_index, pair_column].set_visible(False)
        axes[row_index, pair_column + 1].set_visible(False)
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} measured vs predicted Scheduler progress",
        y=1.025,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_DENSE_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_host_timing_contribution_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot non-additive timing views on the realized full-mode schedule."""

    positions = list(range(len(rows)))
    device = [
        _finite(row["device_active_s"], "Device contribution") for row in rows
    ]
    launch = [
        _finite(
            row["launch_critical_path_increment_s"], "Host launch contribution"
        )
        for row in rows
    ]
    scheduler = [
        _finite(row["scheduler_cpu_increment_s"], "Scheduler contribution")
        for row in rows
    ]
    makespan = [
        _finite(row["schedule_makespan_s"], "schedule makespan") for row in rows
    ]
    execution = [
        device_value + launch_value
        for device_value, launch_value in zip(device, launch, strict=True)
    ]
    figure, axis = plt.subplots(
        figsize=(max(10.0, 0.75 * len(rows)), 6.0),
        constrained_layout=True,
    )
    width = 0.19
    axis.bar(
        [position - 1.5 * width for position in positions],
        device,
        width=width,
        label="Summed Device active time",
        color="#4c78a8",
    )
    axis.bar(
        [position - 0.5 * width for position in positions],
        execution,
        width=width,
        label="Device + Launch execution wall",
        color="#f58518",
    )
    axis.bar(
        [position + 0.5 * width for position in positions],
        scheduler,
        width=width,
        label="Summed Scheduler CPU time",
        color="#54a24b",
    )
    axis.bar(
        [position + 1.5 * width for position in positions],
        makespan,
        width=width,
        label="Schedule makespan",
        color="#b279a2",
    )
    axis.set_title(
        f"{_HARDWARE_LABELS[hardware_id]} full-mode timing views (non-additive)"
    )
    axis.set_ylabel("seconds")
    axis.set_xticks(
        positions,
        [_case_label(row) for row in rows],
        rotation=35,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper center", fontsize=_LEGEND_FONTSIZE)
    _style_subplot_titles(figure)
    figure.savefig(path, dpi=_STANDARD_PLOT_DPI)
    plt.close(figure)


def _write_host_timing_overhead_percent_plot(
    plt: object,
    path: Path,
    hardware_ids: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Plot three-hardware Launch and Scheduler shares in one figure."""

    figure, axes = plt.subplots(
        1, len(hardware_ids),
        figsize=(5.2 * len(hardware_ids), 4.8),
        sharey=True,
        squeeze=False,
    )
    width = 0.38
    for column, hardware_id in enumerate(hardware_ids):
        axis = axes[0, column]
        hardware_rows = sorted(
            [row for row in rows if row["hardware_id"] == hardware_id],
            key=_case_sort_key,
        )
        positions = list(range(len(hardware_rows)))
        launch_percent = []
        scheduler_percent = []
        for row in hardware_rows:
            makespan = _finite(
                row["schedule_makespan_s"], "schedule makespan"
            )
            if makespan <= 0.0:
                raise ValueError("schedule makespan must be positive")
            launch_percent.append(
                100.0
                * _finite(
                    row["launch_critical_path_increment_s"],
                    "Host launch contribution",
                )
                / makespan
            )
            scheduler_percent.append(
                100.0
                * _finite(
                    row["scheduler_cpu_increment_s"],
                    "Scheduler contribution",
                )
                / makespan
            )
        axis.bar(
            [position - width / 2.0 for position in positions],
            launch_percent,
            width=width,
            label="Launch critical-path increment",
            color="#f58518",
        )
        axis.bar(
            [position + width / 2.0 for position in positions],
            scheduler_percent,
            width=width,
            label="Scheduler CPU increment",
            color="#54a24b",
        )
        axis.set_title(_HARDWARE_LABELS[hardware_id], fontsize=10.5, pad=5)
        axis.set_xticks(
            positions,
            [_case_label(row) for row in hardware_rows],
            rotation=58,
            ha="right",
        )
        axis.tick_params(axis="x", labelsize=6.2, pad=1)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_ylabel("share of schedule makespan (%)\n(non-additive)")
        axis.margins(x=0.015)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#d5d5d5", linewidth=0.65, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=_LEGEND_FONTSIZE,
    )
    figure.suptitle(
        "Host timing overhead share across hardware",
        y=0.995,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(figure)
    figure.subplots_adjust(
        left=0.06, right=0.995, top=0.79, bottom=0.30, wspace=0.16,
    )
    figure.savefig(
        path, dpi=_OVERVIEW_PLOT_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def _write_three_mode_throughput_and_speed_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
    speed_rows: Sequence[Mapping[str, object]],
) -> None:
    if len(speed_rows) != len(cells):
        raise ValueError("three-mode speed rows must match report cells")
    speed_by_cell = {str(row["cell_id"]): row for row in speed_rows}
    if len(speed_by_cell) != len(speed_rows):
        raise ValueError("three-mode speed row cell IDs must be unique")
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14.0, 9.0),
        sharex=True,
    )
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    positions = list(range(len(cells)))
    width = 0.19
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    throughput_by_comparison = [[], [], [], []]
    for cell in cells:
        for index, value in enumerate(
            _three_mode_metric_values(cell, "request_throughput_per_s")
        ):
            throughput_by_comparison[index].append(value)
    for label, color, offset, values in zip(
        _COMPARISON_LABELS,
        colors,
        offsets,
        throughput_by_comparison,
        strict=True,
    ):
        axes[0].bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=label,
            color=color,
        )
    axes[0].set_title("Request throughput: observed and three timing modes")
    axes[0].set_ylabel("requests/s")
    axes[0].grid(axis="y", alpha=0.25)
    case_labels = [_case_label(cell) for cell in cells]
    axes[0].set_xticks(positions, case_labels, rotation=42, ha="right")
    axes[0].tick_params(axis="x", labelbottom=True, labelsize=8)

    ordered_speed_rows = []
    for cell in cells:
        cell_id = str(cell["cell_id"])
        if cell_id not in speed_by_cell:
            raise ValueError(f"missing three-mode speed row for {cell_id}")
        ordered_speed_rows.append(speed_by_cell[cell_id])
    wall_seconds = [
        _finite(row["evaluation_wall_time_seconds"], "evaluation wall time")
        for row in ordered_speed_rows
    ]
    speed_bars = axes[1].bar(positions, wall_seconds, color=colors[0], width=0.68)
    axes[1].bar_label(
        speed_bars,
        labels=[f"{value:.1f}s" for value in wall_seconds],
        padding=2,
        fontsize=7,
    )
    for position, bar, row in zip(
        positions, speed_bars, ordered_speed_rows, strict=True
    ):
        ratio = _finite(
            row["wall_seconds_per_observed_workload_second"],
            "simulation speed ratio",
        )
        axes[1].annotate(
            f"{ratio:.2f}× real",
            (position, bar.get_height()),
            xytext=(0, 15),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[1].set_title("Simulation wall time per case")
    axes[1].set_ylabel("wall time (s)")
    axes[1].set_xticks(
        positions,
        case_labels,
        rotation=42,
        ha="right",
    )
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        _three_mode_speed_title(hardware_id, cells),
        y=0.995,
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        fontsize=_LEGEND_FONTSIZE,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.82,
        bottom=0.20,
        hspace=0.62,
    )
    figure.savefig(path, dpi=_STANDARD_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_three_mode_latency_quantile_curves_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
    metric_name: str,
) -> None:
    if metric_name not in _PLOTTED_LATENCY_METRICS:
        raise ValueError(f"unsupported three-mode quantile metric: {metric_name}")
    metric_label = _PLOTTED_LATENCY_METRICS[metric_name].upper()
    case_columns = min(3, len(cells))
    row_count = math.ceil(len(cells) / case_columns)
    figure, axes = plt.subplots(
        row_count,
        case_columns,
        figsize=(5.2 * case_columns, max(4.8, 2.75 * row_count + 1.6)),
        squeeze=False,
    )
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    line_styles = ("-", "--", "-.", ":")
    percentile_axis = tuple(range(101))
    for case_index, cell in enumerate(cells):
        row_index = case_index // case_columns
        column_index = case_index % case_columns
        axis = axes[row_index, column_index]
        series = _three_mode_latency_quantile_values(cell, metric_name)
        for label, color, line_style, line_width, values in zip(
            _COMPARISON_LABELS,
            colors,
            line_styles,
            _COMPARISON_LINE_WIDTHS,
            series,
            strict=True,
        ):
            axis.plot(
                percentile_axis,
                values,
                label=label,
                color=color,
                linestyle=line_style,
                linewidth=line_width,
            )
        axis.set_title(
            f"{_case_label(cell).replace(chr(10), ' / ')} — {metric_label}",
            fontsize=8.5,
        )
        axis.set_xlabel("Percentile")
        axis.set_ylabel("Latency (ms)")
        axis.set_xlim(0.0, 100.0)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)

    for empty_case_index in range(
        len(cells), row_count * case_columns
    ):
        row_index = empty_case_index // case_columns
        column_index = empty_case_index % case_columns
        axes[row_index, column_index].set_visible(False)

    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} {metric_label} quantiles: "
        "observed vs three simulation modes",
        y=0.998,
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.90,
        bottom=0.05,
        hspace=0.72,
        wspace=0.34,
    )
    figure.savefig(path, dpi=_STANDARD_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _three_mode_latency_quantile_values(
    cell: Mapping[str, object], metric_name: str
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    if metric_name not in _PLOTTED_LATENCY_METRICS:
        raise ValueError(f"unsupported three-mode quantile metric: {metric_name}")
    distribution_field = _LATENCY_DISTRIBUTION_FIELDS[metric_name]
    modes = _mapping(cell["timing_mode_results"], "timing mode results")
    if set(modes) != set(_TIMING_MODES):
        raise ValueError("three-mode results must contain exactly the supported modes")
    observed_values = None
    predicted_by_mode = []
    for mode in _TIMING_MODES:
        result = _mapping(modes[mode], f"{mode} result")
        metrics = _mapping(result["metrics"], f"{mode} metrics")
        comparison = _mapping(
            metrics["latency_quantiles_p0_p100"],
            f"{mode} latency quantile comparison",
        )
        _validate_latency_quantile_comparison(comparison, completed=True)
        measured = _quantile_values_ms(
            comparison["measured"], distribution_field, f"{mode} measured"
        )
        predicted = _quantile_values_ms(
            comparison["predicted"], distribution_field, f"{mode} predicted"
        )
        if observed_values is None:
            observed_values = measured
        elif measured != observed_values:
            raise ValueError(
                f"three-mode observed {metric_name} quantiles must be identical"
            )
        predicted_by_mode.append(predicted)
    if observed_values is None:
        raise AssertionError("supported timing modes cannot be empty")
    return (
        observed_values,
        predicted_by_mode[0],
        predicted_by_mode[1],
        predicted_by_mode[2],
    )


def _quantile_values_ms(
    manifest_value: object, distribution_field: str, label: str
) -> tuple[float, ...]:
    manifest = _mapping(manifest_value, f"{label} distributions")
    distribution = _mapping(
        manifest[distribution_field], f"{label} {distribution_field}"
    )
    quantiles = _mapping(
        distribution["quantiles"], f"{label} {distribution_field} quantiles"
    )
    return tuple(
        value * 1000.0
        for value in _finite_series(
            quantiles["values_s"], f"{label} {distribution_field}", 101
        )
    )


def _three_mode_metric_values(
    cell: Mapping[str, object], metric_name: str
) -> tuple[float, float, float, float]:
    modes = _mapping(cell["timing_mode_results"], "timing mode results")
    if set(modes) != set(_TIMING_MODES):
        raise ValueError("three-mode results must contain exactly the supported modes")
    if metric_name == "request_throughput_per_s":
        section_name = "throughput"
        statistic = None
    elif metric_name in _PLOTTED_LATENCY_METRICS:
        section_name = "latency"
        statistic = "mean"
    else:
        raise ValueError(f"unsupported three-mode plot metric: {metric_name}")
    values = []
    observed_value = None
    for mode in _TIMING_MODES:
        result = _mapping(modes[mode], f"{mode} result")
        metrics = _mapping(result["metrics"], f"{mode} metrics")
        section = _mapping(metrics[section_name], f"{mode} {section_name}")
        value = section[metric_name]
        if statistic is not None:
            value = _mapping(value, f"{mode} {metric_name}")[statistic]
        comparison = _comparison(value, f"{mode} {metric_name}")
        observed = _finite(comparison["observed"], f"{mode} observed {metric_name}")
        predicted = comparison["predicted"]
        if predicted is None:
            raise ValueError(f"{mode} {metric_name} prediction is unavailable")
        if observed_value is None:
            observed_value = observed
        elif observed != observed_value:
            raise ValueError(
                f"three-mode observed {metric_name} values must be identical"
            )
        values.append(_finite(predicted, f"{mode} predicted {metric_name}"))
    if observed_value is None:
        raise AssertionError("supported timing modes cannot be empty")
    return (observed_value, values[0], values[1], values[2])


def _three_mode_speed_title(
    hardware_id: str, cells: Sequence[Mapping[str, object]]
) -> str:
    wall_seconds = sum(
        _finite(cell["evaluation_wall_time_seconds"], "evaluation wall time")
        for cell in cells
    )
    workload_seconds = 0.0
    for cell in cells:
        observed_throughput = _three_mode_metric_values(
            cell, "request_throughput_per_s"
        )[0]
        if observed_throughput <= 0.0:
            raise ValueError("observed request throughput must be positive")
        workload_seconds += (
            _integer(cell["request_count"], "request_count")
            / observed_throughput
        )
    if wall_seconds == 0.0:
        speed_text = "0.000x observed real-world time"
    else:
        real_time_factor = workload_seconds / wall_seconds
        wall_per_real = wall_seconds / workload_seconds
        speed_text = (
            f"{wall_per_real:.2f}x observed real-world time; "
            f"{real_time_factor:.3f}x real-time"
        )
    return (
        f"{_HARDWARE_LABELS[hardware_id]}: observed vs three simulation modes\n"
        f"simulation wall {wall_seconds:.3f} s; {speed_text}"
    )


def _write_scheduler_comparison_plot(
    plt: object, path: Path, cell: Mapping[str, object]
) -> None:
    if "timing_mode_results" not in cell:
        _write_full_scheduler_comparison_plot(plt, path, cell)
        return
    observed = _mapping(
        cell["observed_scheduler_progress"], "observed scheduler progress"
    )
    series_by_label = {"Observed": _scheduler_series(observed, "observed")}
    modes = _mapping(cell["timing_mode_results"], "timing mode results")
    if set(modes) != set(_TIMING_MODES):
        raise ValueError("three-mode results must contain exactly the supported modes")
    for mode in _TIMING_MODES:
        result = _mapping(modes[mode], f"{mode} result")
        provenance = _mapping(result["forward_provenance"], f"{mode} provenance")
        progress = _mapping(
            provenance["scheduler_progress"], f"{mode} scheduler progress"
        )
        series_by_label[_TIMING_MODE_LABELS[mode]] = _scheduler_series(
            progress, mode
        )
    figure, axes = plt.subplots(
        3, 2, figsize=(15.0, 10.5)
    )
    fields = (
        ("step_duration_ms", "Step duration", "ms"),
        ("request_count", "Scheduled requests", "requests"),
        ("prefill_tokens", "Prefill tokens", "tokens"),
        ("decode_tokens", "Decode tokens", "tokens"),
        ("cumulative_scheduled_tokens", "Cumulative scheduled tokens", "tokens"),
        ("cumulative_completed_requests", "Cumulative completed requests", "requests"),
    )
    for axis, (field, title, unit) in zip(axes.flat, fields, strict=True):
        for label, series in series_by_label.items():
            axis.plot(
                [entry["step_index"] for entry in series],
                [entry[field] for entry in series],
                label=label,
                linewidth=1.0,
                alpha=0.85,
            )
        axis.set_title(title)
        axis.set_xlabel("Scheduler step")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
    figure.suptitle(
        f"{_HARDWARE_LABELS[str(cell['hardware_id'])]} {_case_label(cell)} "
        "Scheduler progress",
        y=0.985,
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    figure.subplots_adjust(top=0.88, bottom=0.07, hspace=0.40, wspace=0.18)
    figure.savefig(path, dpi=_STANDARD_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _write_full_scheduler_comparison_plot(
    plt: object, path: Path, cell: Mapping[str, object]
) -> None:
    observed = _scheduler_series(
        _mapping(
            cell["observed_scheduler_progress"],
            "observed scheduler progress",
        ),
        "observed",
    )
    provenance = _mapping(cell["forward_provenance"], "forward provenance")
    predicted = _scheduler_series(
        _mapping(provenance["scheduler_progress"], "predicted scheduler progress"),
        "predicted full",
    )
    series_by_label = {"Observed": observed, "Predicted full": predicted}
    colors = {"Observed": "#4c78a8", "Predicted full": "#e45756"}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))

    for label, series in series_by_label.items():
        step_indexes = [entry["step_index"] for entry in series]
        axes[0, 0].plot(
            step_indexes,
            [entry["scheduled_tokens"] for entry in series],
            label=label, color=colors[label], linewidth=1.2, alpha=0.88,
        )
        axes[0, 1].plot(
            step_indexes,
            [entry["prefill_fraction"] * 100.0 for entry in series],
            label=f"{label} prefill", color=colors[label], linewidth=1.2,
        )
        axes[0, 1].plot(
            step_indexes,
            [entry["decode_fraction"] * 100.0 for entry in series],
            label=f"{label} decode", color=colors[label], linewidth=1.2,
            linestyle="--",
        )
        axes[1, 0].plot(
            step_indexes,
            [entry["cumulative_scheduled_tokens"] for entry in series],
            label=label, color=colors[label], linewidth=1.4,
        )
        axes[1, 1].plot(
            step_indexes,
            [entry["cumulative_elapsed_ms"] for entry in series],
            label=label, color=colors[label], linewidth=1.4,
        )
    plot_specs = (
        (axes[0, 0], "Tokens scheduled per step", "tokens"),
        (axes[0, 1], "Prefill/decode token mix", "share of step tokens (%)"),
        (axes[1, 0], "Cumulative scheduled tokens", "tokens"),
        (axes[1, 1], "Cumulative scheduler progress time", "milliseconds"),
    )
    for axis, title, ylabel in plot_specs:
        axis.set_title(title)
        axis.set_xlabel("Scheduler step")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=9)
    figure.suptitle(
        f"{_HARDWARE_LABELS[str(cell['hardware_id'])]} "
        f"{_case_label(cell).replace(chr(10), ' / ')} — observed vs full simulation",
        y=0.985,
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    figure.subplots_adjust(top=0.91, bottom=0.08, hspace=0.36, wspace=0.24)
    figure.savefig(path, dpi=_SCHEDULER_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _scheduler_series(
    progress: Mapping[str, object], label: str
) -> tuple[dict[str, float | int], ...]:
    values = _sequence(progress["series"], f"{label} scheduler series")
    if not values:
        raise ValueError(f"{label} scheduler series must not be empty")
    normalized = []
    origin = None
    for expected_index, raw_entry in enumerate(values):
        entry = _mapping(raw_entry, f"{label} scheduler step {expected_index}")
        step_index = _integer(entry["step_index"], f"{label} step index")
        if step_index != expected_index:
            raise ValueError(f"{label} scheduler step indexes must be contiguous")
        start = _finite(entry["start_time_s"], f"{label} step start")
        end = _finite(entry["end_time_s"], f"{label} step end")
        if end < start:
            raise ValueError(f"{label} scheduler step ends before it starts")
        if origin is None:
            origin = start
        scheduled_tokens = _integer(
            entry["scheduled_tokens"], f"{label} scheduled tokens"
        )
        prefill_tokens = _integer(
            entry["prefill_tokens"], f"{label} prefill tokens"
        )
        decode_tokens = _integer(
            entry["decode_tokens"], f"{label} decode tokens"
        )
        if prefill_tokens + decode_tokens != scheduled_tokens:
            raise ValueError(f"{label} prefill/decode tokens do not sum")
        normalized.append({
            "step_index": step_index,
            "step_duration_ms": (end - start) * 1000.0,
            "request_count": _integer(
                entry["request_count"], f"{label} request count"
            ),
            "scheduled_tokens": scheduled_tokens,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_fraction": (
                0.0 if scheduled_tokens == 0 else prefill_tokens / scheduled_tokens
            ),
            "decode_fraction": (
                0.0 if scheduled_tokens == 0 else decode_tokens / scheduled_tokens
            ),
            "cumulative_elapsed_ms": (end - origin) * 1000.0,
            "cumulative_scheduled_tokens": _integer(
                entry["cumulative_scheduled_tokens"],
                f"{label} cumulative scheduled tokens",
            ),
            "cumulative_completed_requests": _integer(
                entry["cumulative_completed_requests"],
                f"{label} cumulative completed requests",
            ),
        })
    return tuple(normalized)


def _plot_slug(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value)
    return slug.strip("_").lower()


def _write_latency_small_multiples(
    plt: object,
    path: Path,
    hardware_id: str,
    metric_name: str,
    hardware_rows: Sequence[Mapping[str, object]],
    percentile_rows: Sequence[Mapping[str, object]],
) -> None:
    column_count = min(3, len(hardware_rows))
    row_count = math.ceil(len(hardware_rows) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.0 * column_count, 3.8 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    if metric_name not in _PLOTTED_LATENCY_METRICS:
        raise ValueError(f"unsupported latency plot metric: {metric_name}")
    title = _PLOTTED_LATENCY_METRICS[metric_name].upper()
    for index, case in enumerate(hardware_rows):
        axis = axes.flat[index]
        selected = [
            row
            for row in percentile_rows
            if row["cell_id"] == case["cell_id"]
            and row["latency_metric"] == metric_name
        ]
        if len(selected) != 101:
            raise ValueError(
                f"{case['cell_id']} {metric_name} does not contain P0-P100"
            )
        axis.plot(
            [row["percentile"] for row in selected],
            [row["measured_ms"] for row in selected],
            label="Measured",
            linewidth=2,
        )
        if selected[0]["predicted_ms"] is None:
            if any(row["predicted_ms"] is not None for row in selected):
                raise ValueError(
                    f"{case['cell_id']} has partial predicted percentiles"
                )
            axis.text(
                0.5,
                0.5,
                "Prediction unsupported",
                transform=axis.transAxes,
                ha="center",
            )
        else:
            if any(row["predicted_ms"] is None for row in selected):
                raise ValueError(
                    f"{case['cell_id']} has partial predicted percentiles"
                )
            axis.plot(
                [row["percentile"] for row in selected],
                [row["predicted_ms"] for row in selected],
                label="Predicted",
                linewidth=2,
            )
        axis.set_title(_case_label(case))
        axis.set_xlabel("Percentile")
        axis.set_ylabel("ms")
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=_LEGEND_FONTSIZE)
    for index in range(len(hardware_rows), row_count * column_count):
        axes.flat[index].set_visible(False)
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} {title} P0-P100",
        y=1.025,
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    figure.savefig(path, dpi=_STANDARD_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


def _identity_row(cell: Mapping[str, object]) -> dict[str, object]:
    if "observation_quality" in cell:
        quality = _mapping(cell["observation_quality"], "observation quality")
        quality_status = quality["status"]
        failed_metrics = ",".join(quality["failed_metrics"])
    else:
        quality_status = "legacy_contract"
        failed_metrics = ""
    return {
        "cell_id": cell["cell_id"],
        "hardware_id": cell["hardware_id"],
        "model_id": cell["model_id"],
        "case_id": cell["case_id"],
        "target_qps": cell["target_qps"],
        "prediction_status": cell["prediction_status"],
        "performance_timing_status": cell["performance_timing_status"],
        "accuracy_claim_status": cell["accuracy_claim_status"],
        "observer_quality_status": quality_status,
        "observer_failed_metrics": failed_metrics,
    }


def _case_label(row: Mapping[str, object]) -> str:
    return f"{row['model_id']}\n{row['case_id']}"


def _case_sort_key(row: Mapping[str, object]) -> tuple[int, float, str]:
    model_id = str(row["model_id"])
    if model_id not in MODEL_RANK:
        raise ValueError(f"unknown ShareGPT report model: {model_id}")
    return (
        MODEL_RANK[model_id],
        _finite(row["target_qps"], "target QPS"),
        str(row["case_id"]),
    )


def _report_cell_sort_key(
    row: Mapping[str, object],
) -> tuple[int, int, float, str]:
    hardware_id = str(row["hardware_id"])
    if hardware_id not in _HARDWARE_RANK:
        raise ValueError(f"unknown ShareGPT report hardware: {hardware_id}")
    return (_HARDWARE_RANK[hardware_id], *_case_sort_key(row))


def _comparison(value: object, label: str) -> Mapping[str, object]:
    comparison = _mapping(value, label)
    if set(comparison) != {
        "observed",
        "predicted",
        "error",
        "absolute_percentage_error",
    }:
        raise ValueError(f"{label} fields are invalid")
    _finite(comparison["observed"], f"{label} observed")
    for field in ("predicted", "error", "absolute_percentage_error"):
        if comparison[field] is not None:
            _finite(comparison[field], f"{label} {field}")
    return comparison


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty report CSV: {path.name}")
    fieldnames = tuple(rows[0])
    for row in rows:
        if tuple(row) != fieldnames:
            raise ValueError(f"inconsistent report CSV fields: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def _sequence(value: object, label: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence")
    return value


def _finite_series(value: object, label: str, length: int) -> list[float]:
    sequence = _sequence(value, label)
    if len(sequence) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return [_finite(item, label) for item in sequence]


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value
