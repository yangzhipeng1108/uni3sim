"""TTFT and TPOT quantile plots for Scheduling Trajectory Replay results."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.serving.metrics.latency_distribution import linear_percentile

from evaluation.json_shape import require_object as _mapping


_HARDWARE_LABELS = {
    "nvidia_a800": "NVIDIA A800",
    "metax_c500": "MetaX C500",
    "ascend_910b": "Ascend 910B",
}
HARDWARE_ORDER = tuple(_HARDWARE_LABELS)
_METRICS = {
    "ttft_ms": "TTFT",
    "tpot_ms": "TPOT",
}
_MODES = (
    "device_only",
    "device_plus_launch",
    "device_plus_launch_plus_observed_scheduler_cpu",
)
_SERIES = (
    ("Trace reconstructed", "#4c78a8", "-", 2.0),
    ("Device only", "#f58518", "--", 1.6),
    ("Device + Launch", "#54a24b", "-.", 1.6),
    ("Device + Launch + observed Scheduler CPU", "#e45756", ":", 1.8),
)
_BREAKDOWN_COMPONENTS = (
    ("reconstruction", "Trace reconstruction", "#9d9da1"),
    ("device_direct", "Device direct", "#4c78a8"),
    ("device_feedback", "Device feedback", "#72b7b2"),
    ("launch_direct", "Launch direct", "#f58518"),
    ("launch_feedback", "Launch feedback", "#ffbf79"),
    ("scheduler_direct", "Scheduler CPU direct", "#54a24b"),
    ("scheduler_feedback", "Scheduler feedback", "#e45756"),
)


def load_scheduling_trajectory_replay_results(
    input_dir: str | Path,
) -> tuple[Mapping[str, object], ...]:
    """Load supported replay JSON files from one evaluation directory."""

    source = Path(input_dir)
    if not source.is_dir():
        raise ValueError(f"replay result directory does not exist: {source}")
    paths = sorted(source.glob("*.json"))
    if not paths:
        raise ValueError(f"replay result directory contains no JSON files: {source}")
    cells = []
    for path in paths:
        value = json.loads(path.read_text())
        cells.append(_validated_cell(_mapping(value, str(path))))
    cell_ids = [str(cell["cell_id"]) for cell in cells]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("replay result cell IDs must be unique")
    return tuple(cells)


def write_scheduling_trajectory_replay_quantile_plots(
    output_dir: str | Path,
    cells: Sequence[Mapping[str, object]],
) -> Mapping[str, Path]:
    """Write per-hardware TTFT and TPOT quantile small-multiple plots."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Scheduling Trajectory Replay report rendering requires matplotlib"
        ) from error

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    normalized = tuple(_validated_cell(cell) for cell in cells)
    if not normalized:
        raise ValueError("replay report requires at least one result")
    by_hardware: dict[str, list[Mapping[str, object]]] = {}
    for cell in normalized:
        hardware_id, _, _ = _cell_identity(cell)
        if hardware_id not in by_hardware:
            by_hardware[hardware_id] = []
        by_hardware[hardware_id].append(cell)
    paths = {}
    unknown_hardware = set(by_hardware) - set(HARDWARE_ORDER)
    if unknown_hardware:
        raise ValueError(f"unsupported hardware IDs: {sorted(unknown_hardware)}")
    for hardware_id in HARDWARE_ORDER:
        if hardware_id not in by_hardware:
            continue
        hardware_cells = by_hardware[hardware_id]
        ordered_cells = sorted(hardware_cells, key=_case_sort_key)
        for metric_name, metric_label in _METRICS.items():
            path = destination / (
                f"{hardware_id}_scheduling_trajectory_replay_"
                f"{metric_name.removesuffix('_ms')}_quantile_curves.png"
            )
            _write_quantile_plot(
                plt,
                path,
                hardware_id,
                ordered_cells,
                metric_name,
                metric_label,
            )
            paths[f"{hardware_id}_{metric_name}"] = path
    return paths


def write_scheduler_depth_diagnostic_plots(
    output_dir: str | Path,
    cases: Sequence[Mapping[str, object]],
    full_cells: Mapping[str, Mapping[str, object]],
) -> Mapping[str, Path]:
    """Render Serving means and signed error components in the existing style."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Scheduler-depth diagnostic rendering requires matplotlib"
        ) from error
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    normalized = tuple(_validated_breakdown_case(case) for case in cases)
    if not normalized:
        raise ValueError("Scheduler-depth diagnostic plots require cases")
    paths = {}
    present_hardware = {str(case["hardware_id"]) for case in normalized}
    unknown_hardware = present_hardware - set(HARDWARE_ORDER)
    if unknown_hardware:
        raise ValueError(f"unsupported hardware IDs: {sorted(unknown_hardware)}")
    hardware_ids = tuple(
        hardware_id
        for hardware_id in HARDWARE_ORDER
        if hardware_id in present_hardware
    )
    for hardware_id in hardware_ids:
        hardware_cases = sorted(
            (case for case in normalized if case["hardware_id"] == hardware_id),
            key=_breakdown_case_sort_key,
        )
        serving_path = destination / f"{hardware_id}_serving_metrics.png"
        _write_serving_metric_plot(
            plt, serving_path, hardware_id, hardware_cases, full_cells
        )
        paths[f"{hardware_id}_serving_metrics"] = serving_path
        breakdown_path = destination / f"{hardware_id}_error_breakdown.png"
        _write_error_breakdown_plot(
            plt, breakdown_path, hardware_id, hardware_cases
        )
        paths[f"{hardware_id}_error_breakdown"] = breakdown_path
    return paths


def _write_serving_metric_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cases: Sequence[Mapping[str, object]],
    full_cells: Mapping[str, Mapping[str, object]],
) -> None:
    figure, axes = plt.subplots(
        3, 1, figsize=(max(11.0, 0.78 * len(cases)), 12.0),
        constrained_layout=True,
    )
    definitions = (
        ("request_throughput_per_s", "Request throughput", "requests/s"),
        ("ttft_ms", "Mean TTFT", "milliseconds"),
        ("tpot_ms", "Mean TPOT", "milliseconds/token"),
    )
    positions = list(range(len(cases)))
    labels = [
        f"{case['model_id']}\nqps{case['target_qps']}" for case in cases
    ]
    for axis, (metric, title, unit) in zip(axes, definitions, strict=True):
        observed = []
        predicted = []
        for case in cases:
            cell_id = str(case["cell_id"])
            if cell_id not in full_cells:
                raise ValueError(f"Serving plot lacks full cell {cell_id}")
            full = _mapping(full_cells[cell_id], "full Serving cell")
            comparison = _full_metric_comparison(full, metric)
            observed.append(_finite(comparison["observed"], f"observed {metric}"))
            predicted.append(_finite(comparison["predicted"], f"predicted {metric}"))
        axis.bar(
            [position - 0.2 for position in positions],
            observed,
            width=0.4,
            label="Measured",
            color="#4c78a8",
        )
        axis.bar(
            [position + 0.2 for position in positions],
            predicted,
            width=0.4,
            label="Predicted depth=2",
            color="#f58518",
        )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} Serving evaluation", y=1.055
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _full_metric_comparison(
    full: Mapping[str, object], metric: str
) -> Mapping[str, object]:
    metrics = _mapping(full["metrics"], "full Serving metrics")
    if metric == "request_throughput_per_s":
        throughput = _mapping(metrics["throughput"], "full throughput metrics")
        return _mapping(throughput[metric], f"full {metric}")
    if metric in {"ttft_ms", "tpot_ms"}:
        latency = _mapping(metrics["latency"], "full latency metrics")
        distribution = _mapping(latency[metric], f"full {metric}")
        return _mapping(distribution["mean"], f"full {metric} mean")
    raise ValueError(f"unsupported full Serving plot metric: {metric}")


def _write_error_breakdown_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cases: Sequence[Mapping[str, object]],
) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(max(11.0, 0.78 * len(cases)), 9.0),
        constrained_layout=True,
    )
    positions = list(range(len(cases)))
    labels = [
        f"{case['model_id']}\nqps{case['target_qps']}" for case in cases
    ]
    for axis, metric in zip(axes, ("ttft_ms", "tpot_ms"), strict=True):
        positive_bottom = [0.0] * len(cases)
        negative_bottom = [0.0] * len(cases)
        for component, label, color in _BREAKDOWN_COMPONENTS:
            values = [
                _finite(
                    case["metrics"][metric]["component_over_observed"][component],
                    f"{metric} {component}",
                )
                for case in cases
            ]
            bottoms = [
                positive_bottom[index] if value >= 0.0 else negative_bottom[index]
                for index, value in enumerate(values)
            ]
            axis.bar(
                positions, values, bottom=bottoms, label=label, color=color
            )
            for index, value in enumerate(values):
                if value >= 0.0:
                    positive_bottom[index] += value
                else:
                    negative_bottom[index] += value
        final_error = [
            _finite(
                case["metrics"][metric]["final_signed_relative_error"],
                f"{metric} final error",
            )
            for case in cases
        ]
        axis.scatter(
            positions,
            final_error,
            marker="D",
            color="#222222",
            s=20,
            label="Final signed error",
            zorder=3,
        )
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(metric.removesuffix("_ms").upper())
        axis.set_ylabel("signed delta / measured")
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} signed error breakdown", y=1.065
    )
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _validated_breakdown_case(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    case = _mapping(value, "Scheduler-depth breakdown case")
    _cell_identity(case)
    if case["hardware_id"] != str(case["cell_id"]).split("/", 1)[0]:
        raise ValueError("breakdown hardware disagrees with cell ID")
    expected_components = {
        name for name, _label, _color in _BREAKDOWN_COMPONENTS
    }
    for metric in ("ttft_ms", "tpot_ms"):
        result = _mapping(case["metrics"][metric], f"breakdown {metric}")
        components = _mapping(
            result["component_over_observed"], f"breakdown {metric} components"
        )
        if set(components) != expected_components:
            raise ValueError("breakdown plot component fields differ")
    return case


def _breakdown_case_sort_key(
    case: Mapping[str, object],
) -> tuple[str, int, str]:
    return (
        str(case["model_id"]),
        _integer(case["target_qps"], "target QPS"),
        str(case["case_id"]),
    )


def _write_quantile_plot(
    plt: object,
    path: Path,
    hardware_id: str,
    cells: Sequence[Mapping[str, object]],
    metric_name: str,
    metric_label: str,
) -> None:
    column_count = min(3, len(cells))
    row_count = math.ceil(len(cells) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, max(4.8, 2.75 * row_count + 1.6)),
        squeeze=False,
    )
    percentile_axis = tuple(range(101))
    for case_index, cell in enumerate(cells):
        axis = axes[case_index // column_count, case_index % column_count]
        for (label, color, line_style, line_width), values in zip(
            _SERIES,
            _quantile_series(cell, metric_name),
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
        _, model_id, case_id = _cell_identity(cell)
        axis.set_title(f"{model_id} / {case_id} — {metric_label}", fontsize=8.5)
        axis.set_xlabel("Percentile")
        axis.set_ylabel("Latency (ms)")
        axis.set_xlim(0.0, 100.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xticks((0, 25, 50, 75, 100))
        axis.tick_params(labelsize=8)
        axis.grid(alpha=0.25)
    for empty_index in range(len(cells), row_count * column_count):
        axes[empty_index // column_count, empty_index % column_count].set_visible(
            False
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(
        f"{_HARDWARE_LABELS[hardware_id]} {metric_label} quantiles: "
        "Scheduling Trajectory Replay",
        y=0.998,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.977),
        ncol=4,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.93,
        bottom=0.05,
        hspace=0.72,
        wspace=0.34,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _quantile_series(
    cell: Mapping[str, object], metric_name: str
) -> tuple[tuple[float, ...], ...]:
    trace = _mapping(cell["trace_reconstructed"], "trace reconstructed")
    modes = _mapping(
        cell["scheduling_trajectory_replay_modes"], "replay modes"
    )
    results = [trace]
    for mode in _MODES:
        mode_result = _mapping(modes[mode], f"{mode} result")
        results.append(_mapping(mode_result["metrics"], f"{mode} metrics"))
    distributions = []
    expected_request_ids = None
    for label, result in zip(
        ("trace_reconstructed",) + _MODES, results, strict=True
    ):
        rows = _sequence(result["per_request"], f"{label} per-request metrics")
        request_values = []
        request_ids = []
        for row_value in rows:
            row = _mapping(row_value, f"{label} request")
            request_ids.append(_integer(row["request_idx"], "request index"))
            request_values.append(_finite(row[metric_name], f"{label} {metric_name}"))
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(f"{label} request indices must be unique")
        if expected_request_ids is None:
            expected_request_ids = tuple(request_ids)
        elif tuple(request_ids) != expected_request_ids:
            raise ValueError("all replay modes must contain identical request indices")
        sorted_values = sorted(request_values)
        distributions.append(
            tuple(linear_percentile(sorted_values, percentile) for percentile in range(101))
        )
    return tuple(distributions)


def _validated_cell(value: Mapping[str, object]) -> Mapping[str, object]:
    cell = _mapping(value, "Scheduling Trajectory Replay result")
    if _integer(cell["schema_version"], "schema version") not in (2, 4, 5):
        raise ValueError(
            "Scheduling Trajectory Replay report requires schema version 2, 4, or 5"
        )
    if cell["analysis_kind"] != "scheduling_trajectory_replay":
        raise ValueError("unexpected Scheduling Trajectory Replay analysis kind")
    _cell_identity(cell)
    _mapping(cell["trace_reconstructed"], "trace reconstructed")
    modes = _mapping(cell["scheduling_trajectory_replay_modes"], "replay modes")
    if set(modes) != set(_MODES):
        raise ValueError("replay result must contain exactly the three timing modes")
    for metric_name in _METRICS:
        _quantile_series(cell, metric_name)
    return cell


def _cell_identity(cell: Mapping[str, object]) -> tuple[str, str, str]:
    cell_id = cell["cell_id"]
    if not isinstance(cell_id, str):
        raise TypeError("cell ID must be a string")
    pieces = cell_id.split("/")
    if len(pieces) != 3 or any(not piece for piece in pieces):
        raise ValueError(f"invalid replay cell ID: {cell_id}")
    hardware_id, model_id, case_id = pieces
    if hardware_id not in _HARDWARE_LABELS:
        raise ValueError(f"unsupported replay hardware ID: {hardware_id}")
    qps_parts = case_id.rsplit("_qps", 1)
    if len(qps_parts) != 2 or not qps_parts[0] or not qps_parts[1].isdigit():
        raise ValueError(f"invalid replay case ID: {case_id}")
    return hardware_id, model_id, case_id


def _case_sort_key(cell: Mapping[str, object]) -> tuple[str, int, str]:
    _, model_id, case_id = _cell_identity(cell)
    qps_text = case_id.rsplit("_qps", 1)[1]
    return model_id, int(qps_text), case_id




def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _finite(value: object, label: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    return value
