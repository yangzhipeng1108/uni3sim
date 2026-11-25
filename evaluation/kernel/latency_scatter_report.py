"""Render cross-hardware actual-versus-Forward latency scatter plots."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple

from evaluation.kernel.dataset import (
    ATTENTION_OP_TYPES,
    MOE_OP_TYPES,
    VECTOR_OP_TYPES,
    get_hardware_config_path,
    load_kernel_evaluation_dataset,
)
from evaluation.kernel.evaluation_runner import process_op_star


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARDWARE = (
    ("nvidia_a800", "NVIDIA A800"),
    ("nvidia_h200", "NVIDIA H200"),
    ("metax_c500", "MetaX C500"),
    ("ascend_910b", "Ascend 910B"),
)
SERIES = (
    ("GEMM", "#0072B2", "o"),
    ("MoE", "#D55E00", "s"),
    ("FlashAttention", "#009E73", "^"),
    ("Vector", "#CC79A7", "D"),
)
VECTOR_TYPES = VECTOR_OP_TYPES
_FIGURE_TITLE_FONTSIZE = 14
_PLOT_DPI = 200


def _style_subplot_titles(figure: object) -> None:
    for axis in figure.axes:
        axis.title.set_fontsize(axis.title.get_fontsize() + 1.0)
        axis.title.set_fontweight("bold")


def _family_index(op_type: str) -> int:
    if op_type == "GEMM":
        return 0
    elif op_type in MOE_OP_TYPES:
        return 1
    elif op_type in ATTENTION_OP_TYPES:
        return 2
    elif op_type in VECTOR_TYPES:
        return 3
    else:
        raise ValueError(f"unsupported latency-scatter op type {op_type!r}")


def _hardware_points(
        hardware_id: str, jobs: int) -> Tuple[str, List[Tuple[float, float, int]]]:
    raw_data = load_kernel_evaluation_dataset(hardware_id)
    hardware_config = get_hardware_config_path(hardware_id)
    packed = [
        (
            task_key,
            entry,
            hardware_config,
            False,
            True,
            False,
            0.0,
            False,
            None,
        )
        for task_key, entry in sorted(raw_data.items())
    ]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(process_op_star, packed))

    points = []
    for task_key, meta, _, forward_pair, _, _, _ in results:
        if forward_pair == "ERR":
            continue
        if not isinstance(forward_pair, tuple) or len(forward_pair) != 2:
            raise RuntimeError(
                f"{hardware_id} task {task_key!r} has no Forward result")
        predicted_ns, actual_ns = forward_pair
        points.append((
            actual_ns / 1000.0,
            predicted_ns / 1000.0,
            _family_index(meta["op_type"]),
        ))
    return hardware_id, points


def collect_latency_points(
        jobs: int, *, semantic_input_dir: Path | None = None
        ) -> Dict[str, List[Tuple[float, float, int]]]:
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("jobs must be an integer")
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if semantic_input_dir is not None:
        points = {}
        for hardware_id, _label in HARDWARE:
            artifact = json.loads((semantic_input_dir / f"{hardware_id}.json").read_text())
            if (artifact["schema_version"] != 1 or artifact["hardware_id"] != hardware_id
                    or artifact["scope"] != "kernel_evaluation_semantic_result"):
                raise ValueError("Kernel semantic artifact schema or hardware identity mismatch")
            cases = artifact["cases"]
            if not cases or artifact["case_count"] != len(cases) or len({case["case_id"] for case in cases}) != len(cases):
                raise ValueError("Kernel semantic case count or identifiers are inconsistent")
            selected = []
            for case in cases:
                pair = case["forward"]
                if pair == "ERR":
                    continue
                if not isinstance(pair, list) or len(pair) != 2 or any(
                        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                        for value in pair):
                    raise ValueError("Kernel scatter requires saved positive finite Forward pairs")
                selected.append((pair[1] / 1000.0, pair[0] / 1000.0,
                    _family_index(case["report_meta"]["op_type"])))
            points[hardware_id] = selected
        return points
    with ThreadPoolExecutor(max_workers=len(HARDWARE)) as pool:
        return dict(pool.map(
            lambda item: _hardware_points(item[0], jobs), HARDWARE))


def summarize_latency_points(
        points: List[Tuple[float, float, int]]) -> Dict[str, float | int]:
    """Summarize relative error without changing the plotted point set."""

    if not points:
        raise ValueError("latency summary requires at least one point")
    ratios = []
    absolute_errors = []
    for actual_us, predicted_us, _family in points:
        if actual_us <= 0.0 or predicted_us <= 0.0:
            raise ValueError("latency summary requires positive latencies")
        ratio = predicted_us / actual_us
        ratios.append(ratio)
        absolute_errors.append(abs(ratio - 1.0))
    return {
        "count": len(points),
        "mape_percent": sum(absolute_errors) / len(points) * 100.0,
        "median_ape_percent": median(absolute_errors) * 100.0,
        "mean_prediction_over_actual": sum(ratios) / len(points),
    }




def render_compact_latency_report(
    points_by_hardware: Dict[str, List[Tuple[float, float, int]]],
    overall_output: Path,
    detailed_output: Path,
) -> None:
    """Render the standard 1x3 overview and 4x6 bucket detail figures."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import (
            FixedLocator,
            LogFormatterMathtext,
            LogLocator,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "kernel latency scatter rendering requires matplotlib") from error

    for hardware_id, _title in HARDWARE:
        if hardware_id not in points_by_hardware:
            raise ValueError(f"missing latency points for {hardware_id}")
        for family_index, (family, _color, _marker) in enumerate(SERIES):
            if not any(
                point[2] == family_index
                for point in points_by_hardware[hardware_id]
            ):
                raise ValueError(
                    f"{hardware_id} has no latency points for {family}")

    all_values = [
        value
        for hardware_id, _title in HARDWARE
        for point in points_by_hardware[hardware_id]
        for value in point[:2]
    ]
    if not all_values or min(all_values) <= 0.0:
        raise ValueError("latency report requires positive data")
    overall_lower = 10 ** math.floor(math.log10(min(all_values)))
    overall_upper = 10 ** math.ceil(math.log10(max(all_values)))
    legend_handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="None", markersize=7,
            markerfacecolor=color, markeredgecolor="none", label=family,
        )
        for family, color, marker in SERIES
    ]
    legend_handles.append(Line2D(
        [0], [0], color="#666666", linewidth=1.2,
        linestyle="--", label="Ideal: predicted = measured",
    ))
    overview, overview_axes = plt.subplots(
        1, len(HARDWARE), figsize=(4.8 * len(HARDWARE), 4.8), squeeze=False,
        sharex=True, sharey=True,
    )
    decade_ticks = [
        10.0 ** exponent
        for exponent in range(
            math.floor(math.log10(overall_lower)),
            math.ceil(math.log10(overall_upper)) + 1,
        )
    ]
    for column, (hardware_id, title) in enumerate(HARDWARE):
        axis = overview_axes[0, column]
        points = points_by_hardware[hardware_id]
        for family_index, (_family, color, marker) in enumerate(SERIES):
            selected = [point for point in points if point[2] == family_index]
            axis.scatter(
                [point[0] for point in selected],
                [point[1] for point in selected],
                s=10, alpha=0.48, c=color, marker=marker,
                edgecolors="none", rasterized=True,
            )
        axis.plot(
            [overall_lower, overall_upper], [overall_lower, overall_upper],
            color="#666666", linewidth=1.2, linestyle="--", zorder=0,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(overall_lower, overall_upper)
        axis.set_ylim(overall_lower, overall_upper)
        axis.set_xticks(decade_ticks)
        axis.set_yticks(decade_ticks)
        axis.xaxis.set_major_formatter(LogFormatterMathtext())
        axis.yaxis.set_major_formatter(LogFormatterMathtext())
        axis.xaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=range(2, 10)))
        axis.tick_params(
            axis="both", which="both", labelbottom=True, labelleft=True
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("Measured latency (µs)")
        axis.set_ylabel("Predicted latency (µs)")
        axis.grid(
            which="major", color="#b8b8b8", linewidth=0.75, alpha=0.52
        )
        axis.grid(
            which="minor", color="#c7c7c7", linewidth=0.45,
            alpha=0.68, linestyle=":"
        )
        summary = summarize_latency_points(points)
        axis.text(
            0.04, 0.96,
            f"n={summary['count']:,}  MAPE={summary['mape_percent']:.1f}%\n"
            f"MedAPE={summary['median_ape_percent']:.1f}%  "
            f"mean P/M={summary['mean_prediction_over_actual']:.3f}×",
            transform=axis.transAxes, ha="left", va="top", fontsize=8.2,
        )
    overview.suptitle(
        "Single-kernel Forward accuracy — all buckets",
        y=1.06,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    overview.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        fontsize=11,
    )
    overview.subplots_adjust(top=0.92)
    _style_subplot_titles(overview)
    overall_output.parent.mkdir(parents=True, exist_ok=True)
    overview.savefig(
        overall_output, dpi=_PLOT_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(overview)

    detail, detail_axes = plt.subplots(
        len(SERIES), len(HARDWARE) * 2,
        figsize=(20.0 * len(HARDWARE) / 3, 13.2), squeeze=False,
    )
    for family_index, (family, color, marker) in enumerate(SERIES):
        selected_by_hardware = {
            hardware_id: [
                point for point in points_by_hardware[hardware_id]
                if point[2] == family_index
            ]
            for hardware_id, _title in HARDWARE
        }
        family_values = [
            value
            for hardware_id, _title in HARDWARE
            for point in selected_by_hardware[hardware_id]
            for value in point[:2]
        ]
        family_lower = 10 ** math.floor(math.log10(min(family_values)))
        family_upper = 10 ** math.ceil(math.log10(max(family_values)))
        family_exponents = list(range(
            math.floor(math.log10(family_lower)),
            math.ceil(math.log10(family_upper)) + 1,
        ))
        tick_stride = 2 if len(family_exponents) > 4 else 1
        tick_indices = list(range(0, len(family_exponents), tick_stride))
        if tick_indices[-1] != len(family_exponents) - 1:
            tick_indices.append(len(family_exponents) - 1)
        family_ticks = [10.0 ** family_exponents[index] for index in tick_indices]
        for hardware_index, (hardware_id, title) in enumerate(HARDWARE):
            selected = selected_by_hardware[hardware_id]
            scatter_axis = detail_axes[family_index, hardware_index * 2]
            cdf_axis = detail_axes[family_index, hardware_index * 2 + 1]
            scatter_axis.scatter(
                [point[0] for point in selected],
                [point[1] for point in selected],
                s=12, alpha=0.58, c=color, marker=marker,
                edgecolors="none", rasterized=True,
            )
            scatter_axis.plot(
                [family_lower, family_upper], [family_lower, family_upper],
                color="#666666", linewidth=1.1, linestyle="--", zorder=0,
            )
            scatter_axis.set_xscale("log")
            scatter_axis.set_yscale("log")
            scatter_axis.set_xlim(family_lower, family_upper)
            scatter_axis.set_ylim(family_lower, family_upper)
            # Use one explicit locator for both dimensions.  Automatic log
            # locators can otherwise choose different visible ticks after
            # the square aspect ratio is laid out.
            scatter_axis.xaxis.set_major_locator(FixedLocator(family_ticks))
            scatter_axis.yaxis.set_major_locator(FixedLocator(family_ticks))
            scatter_axis.xaxis.set_major_formatter(LogFormatterMathtext())
            scatter_axis.yaxis.set_major_formatter(LogFormatterMathtext())
            scatter_axis.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=range(2, 10))
            )
            scatter_axis.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=range(2, 10))
            )
            scatter_axis.tick_params(axis="both", which="major", labelsize=8)
            scatter_axis.set_aspect("equal", adjustable="box")
            scatter_axis.grid(
                which="major", color="#999999", linewidth=0.65, alpha=0.58
            )
            scatter_axis.grid(
                which="minor", color="#b3b3b3", linewidth=0.42,
                alpha=0.48, linestyle=":"
            )
            scatter_axis.set_xlabel("Measured (µs)")
            scatter_axis.set_ylabel("Predicted (µs)")
            summary = summarize_latency_points(selected)
            scatter_axis.text(
                0.04, 0.96,
                f"n={summary['count']:,}  MAPE={summary['mape_percent']:.1f}%",
                transform=scatter_axis.transAxes,
                ha="left", va="top", fontsize=7.5,
            )
            errors = sorted(
                abs(predicted_us / actual_us - 1.0) * 100.0
                for actual_us, predicted_us, _category in selected
            )
            cumulative = [
                (index + 1) / len(errors) * 100.0
                for index in range(len(errors))
            ]
            cdf_axis.step(
                cumulative, errors, where="post", color=color, linewidth=1.7
            )
            cdf_axis.set_xlim(0.0, 100.0)
            cdf_axis.set_ylim(0.0, 100.0)
            cdf_axis.set_xticks((0, 20, 40, 60, 80, 100))
            cdf_axis.set_yticks((0, 20, 40, 60, 80, 100))
            cdf_axis.set_xlabel("Cumulative OP fraction (%)")
            cdf_axis.set_ylabel("Absolute percentage error (%)")
            cdf_axis.grid(color="#b0b0b0", linewidth=0.55, alpha=0.35)
            p90 = errors[math.ceil(0.9 * len(errors)) - 1]
            cdf_axis.text(
                0.04, 0.96,
                f"P50={median(errors):.1f}%  P90={p90:.1f}%",
                transform=cdf_axis.transAxes,
                ha="left", va="top", fontsize=7.5,
            )
            scatter_axis.set_title(
                f"{title}\n{family} Scatter", color=color, fontsize=9.5
            )
            cdf_axis.set_title(
                f"{title}\n{family} APE CDF", color=color, fontsize=9.5
            )
            if hardware_index == 0:
                scatter_axis.annotate(
                    family,
                    xy=(-0.42, 0.5), xycoords="axes fraction",
                    rotation=90, ha="center", va="center",
                    fontsize=11, color=color,
                )
    detail.suptitle(
        "Single-kernel Forward accuracy by operator bucket",
        y=0.997,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    _style_subplot_titles(detail)
    detail.subplots_adjust(
        left=0.06, right=0.995, top=0.94, bottom=0.055,
        hspace=0.48, wspace=0.40,
    )
    detailed_output.parent.mkdir(parents=True, exist_ok=True)
    detail.savefig(
        detailed_output, dpi=_PLOT_DPI, bbox_inches="tight", facecolor="white"
    )
    plt.close(detail)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot actual versus Forward latency for the production "
            "hardware evaluation datasets."))
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "experiment_logs/evaluation/kernel/cross_hardware/kernel-latency-scatter.png"
        ),
    )
    parser.add_argument(
        "--detailed-output",
        type=Path,
        help="4x6 bucket scatter/CDF output; defaults beside --output",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--semantic-input-dir", type=Path,
        help="Read this run's saved Kernel semantic artifacts instead of rerunning prediction")
    args = parser.parse_args()
    points = collect_latency_points(args.jobs, semantic_input_dir=args.semantic_input_dir)
    detailed_output = args.detailed_output
    if detailed_output is None:
        detailed_output = args.output.with_name(
            f"{args.output.stem}-by-bucket{args.output.suffix}"
        )
    render_compact_latency_report(points, args.output, detailed_output)
    print(args.output)
    print(detailed_output)
    return 0
