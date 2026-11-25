"""Mechanism report plots; extracted without changing the established contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from evaluation.kernel.mechanism_capture import (
    BUCKET_MAPE_LIMIT_PCT,
    CASE_APE_LIMIT_PCT,
    FAMILY_LABELS,
    HARDWARE,
)

from evaluation.kernel.mechanism_report_rows import (
    c500_silu_mechanism_summary,
    kernel_ordinal_suffix,
    observed_duration_ns,
    plot_case_id,
    result_scope_title,
)


def plot_final_latency(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(
        FAMILY_LABELS,
        ("#0072B2", "#009E73", "#D55E00", "#CC79A7"),
        strict=True,
    ))
    figure, axes = plt.subplots(
        3, 4, figsize=(16.8, 12.0), squeeze=False)
    for row_index, (hardware_id, title) in enumerate(HARDWARE):
        for column_index, family in enumerate(FAMILY_LABELS):
            axis = axes[row_index][column_index]
            selected = [
                row for row in task_rows
                if row["hardware_id"] == hardware_id
                and row["family"] == family
            ]
            if not selected:
                raise RuntimeError(
                    f"no final-latency rows for {hardware_id}/{family}")
            observed = np.asarray([
                row["observed_ns"] / 1000.0 for row in selected])
            predicted = np.asarray([
                row["predicted_ns"] / 1000.0 for row in selected])
            lower = 10 ** math.floor(math.log10(min(
                float(np.min(observed)), float(np.min(predicted)))))
            upper = 10 ** math.ceil(math.log10(max(
                float(np.max(observed)), float(np.max(predicted)))))
            guide = np.asarray([lower, upper])
            axis.fill_between(
                guide, guide * 0.8, guide * 1.2,
                color="#009E73", alpha=0.10)
            axis.plot(
                guide, guide * 0.6, color="#D55E00", linewidth=0.8,
                linestyle=":")
            axis.plot(
                guide, guide * 1.4, color="#D55E00", linewidth=0.8,
                linestyle=":")
            axis.plot(
                guide, guide, color="#444444", linestyle="--", linewidth=1)
            axis.scatter(
                observed, predicted, s=10, alpha=0.45,
                color=colors[family], edgecolors="none", rasterized=True)
            worst = max(
                selected,
                key=lambda row: (
                    float(row["absolute_duration_error_ns"]),
                    float(row["absolute_error_pct"]),
                    float(row["observed_ns"]),
                    row["case_id"],
                ),
            )
            worst_observed = worst["observed_ns"] / 1000.0
            worst_predicted = worst["predicted_ns"] / 1000.0
            midpoint = math.sqrt(lower * upper)
            axis.annotate(
                f"largest |Δt|: {plot_case_id(worst['case_id'])}\n"
                f"APE {worst['absolute_error_pct']:.1f}%",
                (worst_observed, worst_predicted),
                xytext=(-5 if worst_observed > midpoint else 5, 8),
                textcoords="offset points",
                horizontalalignment=(
                    "right" if worst_observed > midpoint else "left"),
                fontsize=6.5,
                arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                            "color": "#666666"},
            )
            axis.set(
                xscale="log", yscale="log",
                xlim=(lower, upper), ylim=(lower, upper))
            axis.set_title(f"{title} · {family} · n={len(selected)}")
            axis.set_xlabel("Observed task duration (µs)")
            axis.set_ylabel("Predicted task duration (µs)")
            axis.grid(True, which="major", alpha=0.25)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task-duration log-log overview only: "
        f"center exact, green ±{BUCKET_MAPE_LIMIT_PCT:.0f}%, "
        f"dotted ±{CASE_APE_LIMIT_PCT:.0f}%")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_duration_ratio(
        rows: Sequence[Mapping[str, Any]], path: Path,
        result_scope: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if result_scope == "real_kernel":
        observed_field = "actual_ns"
        scope_label = "kernel"
    elif result_scope == "aggregate_task":
        observed_field = "observed_ns"
        scope_label = "task"
    else:
        raise ValueError(f"unknown duration-ratio scope: {result_scope!r}")
    colors = dict(zip(
        FAMILY_LABELS,
        ("#0072B2", "#009E73", "#D55E00", "#CC79A7"),
        strict=True,
    ))
    ratio_upper = max(
        float(row["predicted_ns"]) / float(row[observed_field])
        for row in rows) * 1.06
    ratio_upper = max(ratio_upper, 1.5)
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.2), sharey=True)
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        hardware_rows = [
            row for row in rows if row["hardware_id"] == hardware_id]
        if not hardware_rows:
            axis.text(
                0.5, 0.5, "No per-kernel timing",
                ha="center", va="center", transform=axis.transAxes)
            axis.set_title(title)
            axis.set_axis_off()
            continue
        for family in FAMILY_LABELS:
            selected = [
                row for row in hardware_rows if row["family"] == family]
            ratios = np.asarray([
                row["predicted_ns"] / row[observed_field]
                for row in selected
            ])
            axis.scatter(
                [row[observed_field] / 1000.0 for row in selected],
                ratios,
                s=10, alpha=0.45, edgecolors="none",
                color=colors[family], label=family, rasterized=True,
            )
        axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.10)
        axis.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--")
        axis.axhline(0.6, color="#D55E00", linewidth=0.9, linestyle=":")
        axis.axhline(1.4, color="#D55E00", linewidth=0.9, linestyle=":")
        axis.set_xscale("log")
        axis.set_ylim(0.0, ratio_upper)
        worst = max(
            hardware_rows,
            key=lambda row: (
                float(row["absolute_duration_error_ns"]),
                float(row["absolute_error_pct"]),
                float(row[observed_field]),
                row["case_id"],
            ),
        )
        worst_ratio = worst["predicted_ns"] / worst[observed_field]
        worst_x = worst[observed_field] / 1000.0
        ordinal = kernel_ordinal_suffix(worst, result_scope)
        observed_values = [
            float(row[observed_field]) / 1000.0 for row in hardware_rows]
        horizontal_midpoint = math.sqrt(
            min(observed_values) * max(observed_values))
        place_left = worst_x > horizontal_midpoint
        place_below = worst_ratio >= 1.0
        axis.annotate(
            f"largest |Δt|: {plot_case_id(worst['case_id'])}{ordinal}\n"
            f"{worst_ratio:.3g}×, APE {worst['absolute_error_pct']:.1f}%",
            (worst_x, worst_ratio),
            xytext=(-6 if place_left else 6,
                    -10 if place_below else 10),
            textcoords="offset points",
            horizontalalignment="right" if place_left else "left",
            verticalalignment="top" if place_below else "bottom",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                        "color": "#666666"},
        )
        axis.set_title(title)
        axis.set_xlabel(f"Observed {scope_label} duration (µs)")
        axis.set_ylabel("Predicted ÷ observed duration")
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.grid(True, which="major", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · "
        f"{scope_label} duration ratio on a linear y-axis; "
        f"green ±{BUCKET_MAPE_LIMIT_PCT:.0f}%, "
        f"dotted ±{CASE_APE_LIMIT_PCT:.0f}%",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_case_impact_ranking(
        impact_rows: Sequence[Mapping[str, Any]], path: Path,
        rows_per_panel: int = 10) -> None:
    """Plot the largest row-level accuracy failures on a linear time axis."""

    if (isinstance(rows_per_panel, bool)
            or not isinstance(rows_per_panel, int)
            or rows_per_panel <= 0):
        raise ValueError("case-impact rows_per_panel must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure, axes = plt.subplots(
        2, 3, figsize=(19.0, 13.0), squeeze=False)
    scopes = (
        ("real_kernel", "Real kernel invocation"),
        ("aggregate_task", "Aggregate task"),
    )
    colors = {"under": "#0072B2", "over": "#D55E00"}
    for row_index, (result_scope, scope_title) in enumerate(scopes):
        for column_index, (hardware_id, hardware_title) in enumerate(HARDWARE):
            axis = axes[row_index][column_index]
            selected = [
                row for row in impact_rows
                if row["result_scope"] == result_scope
                and row["hardware_id"] == hardware_id
            ][:rows_per_panel]
            if not selected:
                axis.text(
                    0.5, 0.5,
                    f"No APE > {CASE_APE_LIMIT_PCT:.0f}% rows",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue
            selected = list(reversed(selected))
            positions = np.arange(len(selected))
            impact_ms = np.asarray([
                float(row["absolute_duration_error_ns"]) / 1.0e6
                for row in selected
            ])
            bar_colors = [
                colors["under"]
                if float(row["signed_error_pct"]) < 0
                else colors["over"]
                for row in selected
            ]
            axis.barh(positions, impact_ms, color=bar_colors, alpha=0.82)
            labels = []
            for row in selected:
                suffix = kernel_ordinal_suffix(row, result_scope)
                labels.append(
                    f"#{row['impact_rank']} "
                    f"{plot_case_id(row['case_id'], width=34)}{suffix}")
            axis.set_yticks(positions, labels, fontsize=6.2)
            upper = max(impact_ms) * 1.62
            if upper <= 0:
                raise ValueError("case-impact plot requires positive deltas")
            axis.set_xlim(0.0, upper)
            for position, value, row in zip(
                    positions, impact_ms, selected, strict=True):
                predicted_ms = float(row["predicted_ns"]) / 1.0e6
                measured_ms = float(row["measured_ns"]) / 1.0e6
                axis.text(
                    value + upper * 0.012,
                    position,
                    f"APE {float(row['absolute_error_pct']):.1f}%\n"
                    f"P/O {predicted_ms:.3g}/{measured_ms:.3g} ms",
                    verticalalignment="center",
                    fontsize=6.2,
                )
            axis.set_title(f"{hardware_title} · {scope_title}")
            axis.set_xlabel("|Predicted − measured| (ms)")
            axis.grid(True, axis="x", alpha=0.25)
    figure.legend(
        handles=(
            Patch(facecolor=colors["under"], label="Underprediction"),
            Patch(facecolor=colors["over"], label="Overprediction"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · largest APE > "
        f"{CASE_APE_LIMIT_PCT:.0f}% rows "
        "ranked by absolute time impact "
        "(linear axis)",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=1.6)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_bucket_accuracy(bucket_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bucket_families = (*FAMILY_LABELS, "mixed")
    colors = dict(zip(
        bucket_families,
        ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#777777"),
        strict=True,
    ))
    error_upper = max(
        CASE_APE_LIMIT_PCT * 1.25,
        max(
            max(
                float(row["diagnostic_mape_pct"]),
                float(row["diagnostic_max_ape_pct"]))
            for row in bucket_rows) * 1.08,
    )
    figure, axes = plt.subplots(
        2, 3, figsize=(15.8, 8.4), sharey=True, squeeze=False)
    for row_index, result_scope in enumerate(
            ("real_kernel", "aggregate_task")):
        for column_index, (hardware_id, title) in enumerate(HARDWARE):
            axis = axes[row_index][column_index]
            selected = sorted(
                (
                    row for row in bucket_rows
                    if row["result_scope"] == result_scope
                    and row["hardware_id"] == hardware_id
                ),
                key=lambda row: (
                    row["diagnostic_mape_pct"],
                    row["physical_bucket_id"]),
            )
            if not selected:
                axis.text(
                    0.5, 0.5, "No per-kernel timing",
                    ha="center", va="center", transform=axis.transAxes)
                axis.set_title(title)
                axis.set_axis_off()
                continue
            for family in bucket_families:
                positions = [
                    index for index, row in enumerate(selected)
                    if row["family"] == family
                ]
                if not positions:
                    continue
                axis.scatter(
                    positions,
                    [
                        selected[index]["diagnostic_mape_pct"]
                        for index in positions
                    ],
                    s=14, alpha=0.75, edgecolors="none",
                    color=colors[family], label=family, rasterized=True,
                )
            axis.scatter(
                range(len(selected)),
                [row["diagnostic_max_ape_pct"] for row in selected],
                s=10, marker="x", linewidths=0.65, color="#333333",
                label="Worst case in bucket", rasterized=True,
            )
            passed = sum(
                row["diagnostic_mape_pct"] <= BUCKET_MAPE_LIMIT_PCT
                and row["diagnostic_max_ape_pct"] <= CASE_APE_LIMIT_PCT
                for row in selected)
            full_passed = sum(row["bucket_gate_passed"] for row in selected)
            axis.axhline(
                BUCKET_MAPE_LIMIT_PCT, color="#D55E00", linewidth=1.0,
                linestyle="--")
            axis.axhline(
                CASE_APE_LIMIT_PCT, color="#666666", linewidth=1.0,
                linestyle=":")
            axis.set_ylim(0.0, error_upper)
            worst = max(
                selected,
                key=lambda row: (
                    float(row["diagnostic_mape_pct"]),
                    float(row["diagnostic_max_ape_pct"]),
                    row["physical_bucket_id"],
                ),
            )
            ordinal = (
                "" if worst["diagnostic_worst_kernel_ordinal"] is None
                else f" k{worst['diagnostic_worst_kernel_ordinal']}")
            axis.text(
                0.02,
                0.98,
                "diagnostic max\n"
                f"{plot_case_id(worst['diagnostic_worst_case_id'])}"
                f"{ordinal}\n"
                f"bucket {worst['diagnostic_mape_pct']:.1f}%, "
                f"case {worst['diagnostic_max_ape_pct']:.1f}%",
                transform=axis.transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=6.5,
            )
            axis.set_title(
                f"{title}\ndiagnostic thresholds {passed}/{len(selected)}; "
                f"full closure {full_passed}/{len(selected)}")
            axis.set_xlabel("Buckets sorted by MAPE")
            if result_scope == "real_kernel":
                scope_label = "Kernel bucket"
            elif result_scope == "aggregate_task":
                scope_label = "Task bucket"
            else:
                raise AssertionError(
                    f"unhandled result scope: {result_scope!r}")
            axis.set_ylabel(f"{scope_label} error (%)")
            axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=4, frameon=False)
    figure.suptitle(
        "Provenance-unverified physical-bucket diagnostic; circles are MAPE, "
        f"x marks are worst cases; dashed {BUCKET_MAPE_LIMIT_PCT:.0f}%, "
        f"dotted {CASE_APE_LIMIT_PCT:.0f}%",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_before_after_family_accuracy(summary_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    selected = [row for row in summary_rows if row["family"] != "ALL"]
    upper = max(
        CASE_APE_LIMIT_PCT * 1.2,
        max(
            max(
                float(row["before_mape_pct"]),
                float(row["after_mape_pct"]))
            for row in selected) * 1.08,
    )
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 7.0), sharex=True)
    for axis, result_scope in zip(
            axes, ("real_kernel", "aggregate_task"), strict=True):
        values = [
            row for hardware_id, _ in HARDWARE
            for family in FAMILY_LABELS
            for row in selected
            if row["result_scope"] == result_scope
            and row["hardware_id"] == hardware_id
            and row["family"] == family
        ]
        if (result_scope == "aggregate_task"
                and len(values) != len(HARDWARE) * len(FAMILY_LABELS)):
            raise ValueError(
                f"before/after family plot is incomplete for {result_scope}")
        if not values:
            if result_scope != "real_kernel":
                raise AssertionError(
                    f"unhandled empty result scope: {result_scope!r}")
            axis.text(
                0.5,
                0.5,
                "No canonical real-kernel timings",
                transform=axis.transAxes,
                horizontalalignment="center",
                verticalalignment="center",
            )
            axis.set_yticks([])
            axis.set_xlabel("Mean absolute percentage error (%)")
            axis.set_title(result_scope_title(result_scope))
            axis.grid(True, axis="x", alpha=0.25)
            continue
        labels = [
            f"{next(title for identifier, title in HARDWARE if identifier == row['hardware_id'])} · {row['family']}"
            for row in values
        ]
        positions = np.arange(len(values))
        before = np.asarray(
            [float(row["before_mape_pct"]) for row in values])
        after = np.asarray(
            [float(row["after_mape_pct"]) for row in values])
        for position, before_value, after_value in zip(
                positions, before, after, strict=True):
            axis.plot(
                (before_value, after_value),
                (position, position),
                color="#777777",
                linewidth=1.0,
                zorder=1,
            )
        axis.scatter(
            before, positions, facecolors="none", edgecolors="#0072B2",
            linewidths=1.0, s=34, zorder=2)
        axis.scatter(
            after, positions, color="#D55E00", marker="s", s=28, zorder=3)
        axis.axvline(
            BUCKET_MAPE_LIMIT_PCT, color="#009E73", linestyle="--",
            linewidth=1.1)
        axis.set_xlim(0.0, upper)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Mean absolute percentage error (%)")
        axis.set_title(result_scope_title(result_scope))
        axis.grid(True, axis="x", alpha=0.25)
    figure.legend(
        handles=(
            Line2D(
                (), (), marker="o", markerfacecolor="none",
                markeredgecolor="#0072B2", linestyle="none",
                label="Before"),
            Line2D(
                (), (), marker="s", color="#D55E00", linestyle="none",
                label="After"),
            Line2D(
                (), (), color="#009E73", linestyle="--",
                label=(
                    f"{BUCKET_MAPE_LIMIT_PCT:.0f}% bucket-mean target")),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · same rows before and after",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_before_after_bucket_accuracy(bucket_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = {
        "nvidia_a800": "#0072B2",
        "metax_c500": "#D55E00",
        "ascend_910b": "#009E73",
    }
    titles = dict(HARDWARE)
    upper = max(
        CASE_APE_LIMIT_PCT * 1.2,
        max(
            max(
                float(row["before_mape_pct"]),
                float(row["after_mape_pct"]))
            for row in bucket_rows) * 1.08,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.6, 6.5), sharex=True, sharey=True)
    for axis, result_scope in zip(
            axes, ("real_kernel", "aggregate_task"), strict=True):
        selected = [
            row for row in bucket_rows
            if row["result_scope"] == result_scope]
        if not selected:
            raise ValueError(
                f"before/after bucket plot has no {result_scope} rows")
        for hardware_id, _ in HARDWARE:
            values = [
                row for row in selected
                if row["hardware_id"] == hardware_id]
            axis.scatter(
                [float(row["before_mape_pct"]) for row in values],
                [float(row["after_mape_pct"]) for row in values],
                color=colors[hardware_id],
                s=20,
                alpha=0.72,
                edgecolors="none",
                label=titles[hardware_id],
                rasterized=True,
            )
        max_violations = [
            row for row in selected
            if not row["after_diagnostic_max_ape_gate_passed"]]
        axis.scatter(
            [float(row["before_mape_pct"]) for row in max_violations],
            [float(row["after_mape_pct"]) for row in max_violations],
            marker="x", color="#333333", s=19, linewidths=0.65,
            label=f"Contains APE > {CASE_APE_LIMIT_PCT:.0f}%",
            rasterized=True)
        axis.plot(
            (0.0, upper), (0.0, upper), color="#666666", linewidth=1.0)
        axis.axvline(
            BUCKET_MAPE_LIMIT_PCT, color="#CC79A7", linestyle="--",
            linewidth=1.0)
        axis.axhline(
            BUCKET_MAPE_LIMIT_PCT, color="#CC79A7", linestyle="--",
            linewidth=1.0)
        axis.set_xlim(0.0, upper)
        axis.set_ylim(0.0, upper)
        mape_before = sum(
            row["before_diagnostic_mape_gate_passed"] for row in selected)
        mape_after = sum(
            row["after_diagnostic_mape_gate_passed"] for row in selected)
        full_before = sum(
            row["before_diagnostic_accuracy_gate_passed"]
            for row in selected)
        full_after = sum(
            row["after_diagnostic_accuracy_gate_passed"]
            for row in selected)
        axis.set_title(
            result_scope_title(result_scope)
            + " buckets"
            + f"\nMAPE pass {mape_before}→{mape_after}/{len(selected)}; "
            + f"both gates {full_before}→{full_after}")
        axis.set_xlabel("Before: bucket MAPE (%)")
        axis.set_ylabel("After: bucket MAPE (%)")
        axis.grid(True, alpha=0.2)
    handles = [
        Line2D(
            (), (), marker="o", color=colors[hardware_id],
            linestyle="none", label=title)
        for hardware_id, title in HARDWARE
    ]
    handles.extend((
        Line2D(
            (), (), marker="x", color="#333333", linestyle="none",
            label=("Bucket contains a case above "
                   f"{CASE_APE_LIMIT_PCT:.0f}%")),
        Line2D(
            (), (), color="#CC79A7", linestyle="--",
            label=f"{BUCKET_MAPE_LIMIT_PCT:.0f}% bucket-mean target"),
    ))
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · each point is one frozen physical bucket; lower is better",
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.0)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_ape_cdf(
        kernel_rows: Sequence[Mapping[str, Any]],
        task_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(
        (hardware_id for hardware_id, _ in HARDWARE),
        ("#0072B2", "#D55E00", "#009E73"),
        strict=True,
    ))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    error_upper = max(
        CASE_APE_LIMIT_PCT * 1.25,
        max(float(row["absolute_error_pct"]) for row in kernel_rows) * 1.04,
        max(float(row["absolute_error_pct"]) for row in task_rows) * 1.04,
    )
    for axis, (result_scope, rows, label) in zip(
            axes,
            (
                ("real_kernel", kernel_rows, "Real kernels"),
                ("aggregate_task", task_rows, "Aggregate tasks"),
            ),
            strict=True):
        for hardware_id, title in HARDWARE:
            hardware_rows = [
                row for row in rows if row["hardware_id"] == hardware_id]
            values = np.sort(np.asarray([
                row["absolute_error_pct"] for row in hardware_rows]))
            if len(values) == 0:
                continue
            y = np.arange(1, len(values) + 1) / len(values) * 100.0
            axis.step(
                values, y, where="post",
                color=colors[hardware_id], label=title)
        axis.axvline(
            BUCKET_MAPE_LIMIT_PCT, color="#777777", linewidth=0.9,
            linestyle=":")
        axis.axvline(
            CASE_APE_LIMIT_PCT, color="#D55E00", linewidth=1.0,
            linestyle="--")
        worst = max(
            rows,
            key=lambda row: (
                float(row["absolute_error_pct"]),
                observed_duration_ns(row, result_scope),
                row["case_id"],
            ),
        )
        ordinal = kernel_ordinal_suffix(worst, result_scope)
        axis.scatter(
            [worst["absolute_error_pct"]], [100.0], s=24,
            color=colors[worst["hardware_id"]], marker="D", zorder=3)
        axis.annotate(
            f"max APE: {plot_case_id(worst['case_id'])}{ordinal}\n"
            f"{worst['absolute_error_pct']:.1f}%",
            (worst["absolute_error_pct"], 100.0),
            xytext=(-6, -8), textcoords="offset points",
            horizontalalignment="right", verticalalignment="top",
            fontsize=6.5,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                        "color": "#666666"},
        )
        axis.set_xlim(0.0, error_upper)
        axis.set_ylim(0.0, 100.0)
        axis.set_title(label)
        axis.set_xlabel("Absolute percentage error (%)")
        axis.set_ylabel("Rows at or below error (%)")
        axis.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=3, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · complete error distribution on "
        f"a linear axis; guides at {BUCKET_MAPE_LIMIT_PCT:.0f}% and "
        f"{CASE_APE_LIMIT_PCT:.0f}%",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_residual_by_family(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    positions = np.arange(len(FAMILY_LABELS))
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        values = [
            [
                row["signed_error_pct"] for row in task_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            for family in FAMILY_LABELS
        ]
        axis.boxplot(values, positions=positions, widths=0.65, showfliers=True)
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.axhline(-CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.axhline(CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.set_xticks(positions, FAMILY_LABELS, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Signed duration error (%)")
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task residuals by family")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_residual_vs_scale(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "nvidia_a800": "#0072B2",
        "metax_c500": "#D55E00",
        "ascend_910b": "#009E73",
    }
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
    for axis, family in zip(axes, FAMILY_LABELS, strict=True):
        for hardware_id, title in HARDWARE:
            selected = [
                row for row in task_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            axis.scatter(
                [row["scale"] for row in selected],
                [row["signed_error_pct"] for row in selected],
                s=10, alpha=0.4, edgecolors="none", color=colors[hardware_id],
                label=title, rasterized=True,
            )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_title(family)
        axis.set_xlabel("Canonical workload scale")
        axis.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("Signed duration error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task residual versus workload "
        "scale",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_target_traffic_ratios(
        comparison_rows, path: Path, *, hardware_id: str, family: str,
        title: str, component_labels: Sequence[tuple[str, str]],
        category_order: Sequence[str], category_for_variant,
        logarithmic_ratio_axis: bool) -> None:
    """Plot traffic error itself so log-log coordinate distance cannot hide it."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    colors = {
        category: color for category, color in zip(
            category_order,
            ("#D55E00", "#0072B2", "#999999"),
            strict=True,
        )
    }
    markers = {
        category: marker for category, marker in zip(
            category_order, ("D", "s", "o"), strict=True)
    }
    figure, raw_axes = plt.subplots(
        1, len(component_labels),
        figsize=(7.1 * len(component_labels), 5.2), sharex=True)
    axes = np.atleast_1d(raw_axes)
    legend_handles = []
    legend_labels = []
    for axis, (component, component_label) in zip(
            axes, component_labels, strict=True):
        selected = [
            row for row in comparison_rows
            if row["hardware_id"] == hardware_id
            and row["family"] == family
            and row["component"] == component
            and row["predicted"] > 0 and row["observed"] > 0
        ]
        if not selected:
            raise RuntimeError(
                f"no traffic comparison rows for {hardware_id}/{family}/"
                f"{component}")
        severe = [
            row for row in selected
            if abs(row["predicted"] / row["observed"] - 1.0)
            > CASE_APE_LIMIT_PCT / 100.0
        ]
        for category in category_order:
            subset = [
                row for row in selected
                if category_for_variant(row["variant"]) == category
            ]
            if not subset:
                continue
            artist = axis.scatter(
                [row["observed"] for row in subset],
                [row["predicted"] / row["observed"] for row in subset],
                s=18 if category != category_order[-1] else 10,
                alpha=0.7 if category != category_order[-1] else 0.3,
                edgecolors="none", color=colors[category],
                marker=markers[category], label=category, rasterized=True,
            )
            if category not in legend_labels:
                legend_handles.append(artist)
                legend_labels.append(category)
        axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.08)
        axis.axhline(0.6, color="#666666", linewidth=0.8, linestyle=":")
        axis.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--")
        axis.axhline(1.4, color="#666666", linewidth=0.8, linestyle=":")
        axis.set_xscale("log")
        if logarithmic_ratio_axis:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0.0, 2.05)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.set_title(
            f"{component_label}\n{len(severe)}/{len(selected)} kernels "
            f">{CASE_APE_LIMIT_PCT:.0f}% off")
        axis.set_xlabel("Observed interface work")
        axis.set_ylabel("Predicted ÷ observed traffic")
        axis.grid(True, which="major", alpha=0.25)
        observed_values = [float(row["observed"]) for row in selected]
        observed_midpoint = math.sqrt(
            min(observed_values) * max(observed_values))
        labelled = []
        for category in category_order[:-1]:
            category_rows = [
                row for row in severe
                if category_for_variant(row["variant"]) == category
            ]
            if category_rows:
                labelled.append(max(
                    category_rows, key=lambda row: float(row["observed"])))
        for index, row in enumerate(labelled):
            ratio = row["predicted"] / row["observed"]
            observed_work = float(row["observed"])
            case_label = (
                row["case_id"]
                .removeprefix("FUSEDMOE_")
                .removeprefix("VECTOR_ROPE_")
                .removesuffix("_B1")
                .replace("_TOPK", "_K")
                .replace("_INT", "_I")
            )
            label = (
                f"{case_label}"
                f" / {row['variant']} / {ratio:.3g}×")
            place_left = observed_work > observed_midpoint
            axis.annotate(
                label,
                (observed_work, ratio),
                xytext=(-5 if place_left else 5, 6 + 13 * index),
                textcoords="offset points",
                horizontalalignment="right" if place_left else "left",
                fontsize=6.5,
                arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                            "color": "#666666"},
            )
    figure.suptitle(title)
    figure.legend(
        legend_handles, legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.93), ncol=len(legend_labels), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_c500_vector_traffic_ratios(comparison_rows, path: Path) -> None:
    def category(variant: str) -> str:
        if variant in ("rope", "rotary"):
            return "RoPE"
        elif variant in ("rmsnorm", "add_rmsnorm"):
            return "RMSNorm"
        elif variant in (
                "activation", "silu", "kv_cache", "kvcache",
                "paged_kv_cache",
                "elementwise", "elementwise_pytorch_vec4"):
            return "Other Vector"
        else:
            raise ValueError(f"unknown C500 Vector variant: {variant!r}")

    plot_target_traffic_ratios(
        comparison_rows,
        path,
        hardware_id="metax_c500",
        family="Vector",
        title=(
            "PROVENANCE-UNVERIFIED DIAGNOSTIC · MetaX C500 Vector "
            "directional traffic versus historical MACA 3.2.1 observation"),
        component_labels=(
            ("traffic:l2c_source_read", "L2C source-side read bytes"),
            ("traffic:downstream_global_read", "Downstream global read bytes"),
            ("traffic:downstream_global_write", "Downstream global write bytes"),
            ("demand:dram", "Downstream global read + write bytes"),
        ),
        category_order=("RoPE", "RMSNorm", "Other Vector"),
        category_for_variant=category,
        logarithmic_ratio_axis=False,
    )


def plot_ascend_moe_traffic_ratios(replay_rows, path: Path) -> None:
    """Plot the two independently comparable GroupedMatmul interfaces."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if len(replay_rows) != 352:
        raise RuntimeError(
            "Ascend GroupedMatmul traffic plot requires 352 replay rows")

    colors = {"w1": "#D55E00", "w2": "#0072B2"}
    markers = {"w1": "D", "w2": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.2), sharex=True)
    for variant in ("w1", "w2"):
        replay = [row for row in replay_rows if row["variant"] == variant]
        axes[0].scatter(
            [row["predicted_mte2_l2_read_bytes"] for row in replay],
            [row["read_predicted_over_observed"]
             for row in replay],
            s=18, alpha=0.7, edgecolors="none", color=colors[variant],
            marker=markers[variant], label=f"GroupedMatmul {variant}",
            rasterized=True)
        axes[1].scatter(
            [row["predicted_aic_output_l2_write_bytes"] for row in replay],
            [row["output_predicted_over_observed"] for row in replay],
            s=18, alpha=0.7, edgecolors="none", color=colors[variant],
            marker=markers[variant], label=f"GroupedMatmul {variant}",
            rasterized=True)

    read_within_five_pct = sum(
        row["read_absolute_error_pct"] <= 5.0 for row in replay_rows)
    output_within_twenty_pct = sum(
        row["output_absolute_error_pct"] <= 20.0 for row in replay_rows)
    mean_read_error = float(np.mean([
        row["read_absolute_error_pct"] for row in replay_rows]))
    mean_output_error = float(np.mean([
        row["output_absolute_error_pct"] for row in replay_rows]))
    axes[0].set_title(
        "Production MTE2 read → observed GM-to-L1\n"
        f"{read_within_five_pct}/352 within 5%; mean {mean_read_error:.2f}%")
    axes[1].set_title(
        "Production AIC output → observed write interface\n"
        f"{output_within_twenty_pct}/352 within 20%; mean {mean_output_error:.2f}%")
    for axis in axes:
        axis.set_xscale("log")
        axis.axhline(1.0, color="#333333", linewidth=1.1, linestyle="--")
        axis.set_xlabel("Modeled interface bytes")
        axis.set_ylabel("Predicted ÷ observed interface bytes")
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.grid(True, which="major", alpha=0.25)
    for axis in axes:
        axis.set_ylim(0.78, 1.22)
        axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.08)
        axis.axhline(0.8, color="#666666", linewidth=0.8, linestyle=":")
        axis.axhline(1.2, color="#666666", linewidth=0.8, linestyle=":")

    for index, variant in enumerate(("w1", "w2")):
        row = max(
            (item for item in replay_rows if item["variant"] == variant),
            key=lambda item: float(item["predicted_mte2_l2_read_bytes"]))
        ratio = row["read_predicted_over_observed"]
        modeled_bytes = row["predicted_mte2_l2_read_bytes"]
        axes[0].annotate(
            f"largest {variant}: {modeled_bytes / 2**20:.1f} MiB, "
            f"{ratio:.4f}×",
            (modeled_bytes, ratio),
            xytext=(-8, 12 + index * 14),
            textcoords="offset points", horizontalalignment="right",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                        "color": "#666666"},
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · Ascend 910B MoE production "
        "read/output demand versus counters")
    figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
        ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_c500_silu_mechanism(
        rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Plot demand closure and replay-only latency evidence without mixing clocks."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("C500 SiLU mechanism plot requires rows")
    duration_us = np.asarray([
        float(row["production_duration_ns"]) / 1000.0 for row in rows])
    predicted_ratio = np.asarray([
        float(row["predicted_over_production"]) for row in rows])
    waves = np.asarray([float(row["waves"]) for row in rows])
    summary = c500_silu_mechanism_summary(rows)
    figure, axes = plt.subplots(2, 2, figsize=(14.8, 10.2))

    duration_axis = axes[0][0]
    duration_axis.scatter(
        duration_us, predicted_ratio, s=22, alpha=0.72,
        color="#0072B2", edgecolors="none")
    duration_axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.10)
    duration_axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    duration_axis.axhline(0.6, color="#D55E00", linestyle=":", linewidth=1)
    duration_axis.axhline(1.4, color="#D55E00", linestyle=":", linewidth=1)
    duration_axis.set_xscale("log")
    duration_axis.set_ylim(0.0, 1.5)
    duration_axis.set_xlabel("Canonical task duration (µs)")
    duration_axis.set_ylabel("Forward ÷ production duration")
    duration_axis.set_title(
        "A · Final time: "
        f"{summary['underpredicted_count']}/{summary['case_count']} "
        f"underpredicted; MAPE {summary['duration_mape_pct']:.1f}%")
    duration_axis.grid(True, which="major", alpha=0.25)
    largest_delta_row = max(
        rows,
        key=lambda item: abs(
            float(item["predicted_duration_ns"])
            - float(item["production_duration_ns"])),
    )
    x_value = float(largest_delta_row["production_duration_ns"]) / 1000.0
    y_value = float(largest_delta_row["predicted_over_production"])
    duration_axis.annotate(
        f"largest absolute miss\n"
        f"{plot_case_id(largest_delta_row['case_id'], 27)} · {y_value:.2f}×",
        (x_value, y_value), xytext=(-8, 14),
        textcoords="offset points", horizontalalignment="right",
        fontsize=7,
        arrowprops={"arrowstyle": "-", "linewidth": 0.5,
                    "color": "#666666"})

    demand_axis = axes[0][1]
    instruction_rows = [
        row for row in rows
        if row["instruction_comparison_status"] == "comparable"
    ]
    if instruction_rows:
        demand_axis.scatter(
            [float(row["production_duration_ns"]) / 1000.0
             for row in instruction_rows],
            [float(row["instruction_predicted_over_observed"])
             for row in instruction_rows],
            marker="o", s=24, alpha=0.7, color="#0072B2",
            edgecolors="none", label="Instructions")
    demand_axis.scatter(
        duration_us,
        [float(row["traffic_predicted_over_observed"]) for row in rows],
        marker="^", s=27, alpha=0.7, color="#D55E00",
        edgecolors="none", label="Global bytes")
    demand_axis.axhspan(0.8, 1.2, color="#009E73", alpha=0.10)
    demand_axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    demand_axis.axhline(0.6, color="#D55E00", linestyle=":", linewidth=1)
    demand_axis.axhline(1.4, color="#D55E00", linestyle=":", linewidth=1)
    demand_axis.set_xscale("log")
    demand_axis.set_ylim(0.55, 1.45)
    demand_axis.set_xlabel("Production mcTracer duration (µs)")
    demand_axis.set_ylabel("Predicted demand ÷ mcProfiler counter")
    if summary["instruction_comparison_status"] == "comparable":
        instruction_title = (
            f"instruction MAPE {summary['instruction_mape_pct']:.3f}%")
    elif summary["instruction_comparison_status"] == "unavailable":
        instruction_title = "instruction comparison unavailable"
    else:
        raise ValueError(
            "unsupported C500 SiLU instruction comparison status: "
            f"{summary['instruction_comparison_status']!r}")
    demand_axis.set_title(
        f"B · Work amount: {instruction_title}, bytes "
        f"MAPE {summary['traffic_mape_pct']:.2f}%")
    demand_axis.legend(frameon=False)
    demand_axis.grid(True, which="major", alpha=0.25)

    replay_axis = axes[1][0]
    replay_axis.scatter(
        waves,
        [float(row["instrumented_mean_active_aps"]) for row in rows],
        s=23, alpha=0.72, color="#009E73", edgecolors="none")
    replay_axis.axhline(
        104.0, color="#333333", linestyle="--", linewidth=1,
        label="104 physical APs")
    replay_axis.set_xscale("log")
    replay_axis.set_ylim(0.0, 110.0)
    replay_axis.set_xlabel("Dispatched waves in mcProfiler replay")
    replay_axis.set_ylabel("Time-averaged active AP count in that replay")
    replay_axis.set_title(
        "C · Same-replay activity only; not production utilization")
    replay_axis.legend(frameon=False)
    replay_axis.grid(True, which="major", alpha=0.25)

    wait_axis = axes[1][1]
    colors = np.asarray([
        float(row["instrumented_dnoc_read_latency_cycles"])
        for row in rows])
    points = wait_axis.scatter(
        waves,
        [float(row["instrumented_average_wave_life_cycles"])
         for row in rows],
        c=colors, cmap="viridis", s=25, alpha=0.78, edgecolors="none")
    wait_axis.set_xscale("log")
    wait_axis.set_xlabel("Dispatched waves in mcProfiler replay")
    wait_axis.set_ylabel("Average wave lifetime (XCORE cycles)")
    wait_axis.set_title(
        "D · Replay shows long-lived waves and read latency")
    wait_axis.grid(True, which="major", alpha=0.25)
    colorbar = figure.colorbar(points, ax=wait_axis)
    colorbar.set_label("DNoC read latency (cycles)")

    figure.suptitle(
        "MetaX C500 SiLU mechanism boundary · production time and "
        "instrumented replay are shown separately")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_vector_variant_residual(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical_variants = (
        "rmsnorm", "add_rmsnorm", "silu", "rope", "kv_cache",
        "elementwise",
    )
    aliases = {
        "rmsnorm": "rmsnorm",
        "add_rmsnorm": "add_rmsnorm",
        "silu": "silu",
        "rope": "rope",
        "rotary": "rope",
        "kv_cache": "kv_cache",
        "kvcache": "kv_cache",
        "paged_kv_cache": "kv_cache",
        "elementwise": "elementwise",
        "elementwise_pytorch_vec4": "elementwise",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharey=True)
    positions = np.arange(len(canonical_variants))
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        selected = [
            row for row in task_rows
            if row["hardware_id"] == hardware_id and row["family"] == "Vector"
        ]
        unknown = {
            row["variant_signature"] for row in selected
            if row["variant_signature"] not in aliases
        }
        if unknown:
            raise ValueError(
                f"unknown Vector variant signatures: {sorted(unknown)}")
        values = [
            [
                row["signed_error_pct"] for row in selected
                if aliases[row["variant_signature"]] == variant
            ]
            for variant in canonical_variants
        ]
        if any(not group for group in values):
            raise RuntimeError(
                f"{hardware_id} lacks a canonical Vector variant")
        axis.boxplot(
            values, positions=positions, widths=0.65, showfliers=True)
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.axhline(-CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.axhline(CASE_APE_LIMIT_PCT, color="#D55E00",
                     linewidth=0.9, linestyle=":")
        axis.set_xticks(
            positions, canonical_variants, rotation=30, ha="right")
        axis.set_title(title)
        axis.set_ylabel("Signed task-duration error (%)")
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · Vector task residuals by "
        "variant")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_residual_vs_derived_launch_rounds(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "GEMM": "#0072B2",
        "FlashAttention": "#009E73",
        "FusedMoE": "#D55E00",
        "Vector": "#CC79A7",
    }
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        selected_hardware = [
            row for row in task_rows if row["hardware_id"] == hardware_id]
        for family in FAMILY_LABELS:
            selected = [
                row for row in selected_hardware if row["family"] == family]
            axis.scatter(
                [row["max_derived_launch_rounds"] for row in selected],
                [row["signed_error_pct"] for row in selected],
                s=10, alpha=0.4, edgecolors="none", color=colors[family],
                label=family, rasterized=True,
            )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("Maximum derived grid rounds")
        axis.set_ylabel("Signed task-duration error (%)")
        axis.grid(True, which="major", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · task residual versus derived "
        "launch rounds",
        y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_attention_split_residual(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.7), sharey=True)
    for axis, (hardware_id, title) in zip(axes, HARDWARE, strict=True):
        selected = [
            row for row in task_rows
            if row["hardware_id"] == hardware_id
            and row["family"] == "FlashAttention"
            and row["executed_attention_splits"] is not None
        ]
        if not selected:
            axis.text(
                0.5, 0.5, "No externally comparable split contract",
                ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        axis.scatter(
            [row["executed_attention_splits"] for row in selected],
            [row["signed_error_pct"] for row in selected],
            s=12, alpha=0.45, edgecolors="none", rasterized=True,
        )
        axis.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
        axis.set_xscale("log", base=2)
        axis.set_title(title)
        axis.set_xlabel("Observed executed KV splits")
        axis.set_ylabel("Signed task-duration error (%)")
        axis.grid(True, which="major", alpha=0.3)
    figure.suptitle(
        "PROVENANCE-UNVERIFIED DIAGNOSTIC · FlashAttention task residual "
        "versus executed split count")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_bottlenecks(kernel_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ("launch", "pipeline", "compute", "memory")

    def category(name: str) -> str:
        if name == "launch":
            return "launch"
        elif name == "pipeline":
            return "pipeline"
        elif name.startswith("compute") or name in {
                "tensor_pipe", "vector_pipe", "fma_sfu_pipe", "alu_pipe",
                "warp_issue", "scalar_aiv"}:
            return "compute"
        return "memory"

    labels = [
        f"{hardware_id}\n{family}"
        for hardware_id, _ in HARDWARE for family in FAMILY_LABELS
    ]
    counts = {name: [] for name in categories}
    unavailable_indices = []
    for hardware_id, _ in HARDWARE:
        for family in FAMILY_LABELS:
            selected = [
                row for row in kernel_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            total = len(selected)
            if total == 0:
                unavailable_indices.append(len(counts[categories[0]]))
                for name in categories:
                    counts[name].append(0.0)
                continue
            grouped = {name: 0 for name in categories}
            for row in selected:
                name = category(row["bottleneck"])
                grouped[name] += 1
            for name in categories:
                counts[name].append(grouped[name] / total * 100.0)
    figure, axis = plt.subplots(figsize=(14.2, 5.1))
    bottom = np.zeros(len(labels))
    for name in categories:
        values = np.asarray(counts[name])
        axis.bar(labels, values, bottom=bottom, label=name)
        bottom += values
    axis.set_ylabel("Kernel share (%)")
    axis.set_ylim(0, 100)
    for index in unavailable_indices:
        axis.text(
            index, 3.0, "n/a\nOP timing", ha="center", va="bottom",
            fontsize=7, color="#666666")
    axis.legend(ncol=4, frameon=False, loc="upper center")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_support_coverage(task_rows, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = ("complete", "partial", "aggregate_only", "compatibility")
    labels = [
        f"{title}\n{family}"
        for _, title in HARDWARE for family in FAMILY_LABELS
    ]
    counts = {level: [] for level in levels}
    for hardware_id, _ in HARDWARE:
        for family in FAMILY_LABELS:
            selected = [
                row for row in task_rows
                if row["hardware_id"] == hardware_id and row["family"] == family
            ]
            if not selected:
                raise RuntimeError(
                    f"no support rows for {hardware_id}/{family}")
            for level in levels:
                counts[level].append(
                    sum(row["support_level"] == level for row in selected)
                    / len(selected) * 100.0)
    figure, axis = plt.subplots(figsize=(14.2, 5.1))
    bottom = np.zeros(len(labels))
    for level in levels:
        values = np.asarray(counts[level])
        axis.bar(labels, values, bottom=bottom, label=level)
        bottom += values
    axis.set_ylabel("Task share (%)")
    axis.set_ylim(0, 100)
    axis.set_title("Forward diagnostic support levels (not validation status)")
    axis.legend(ncol=4, frameon=False, loc="upper center")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def render_plot_call(call) -> str:
    """Render one independent report figure in a worker process."""

    plot, arguments, path = call
    plot(*arguments)
    return str(path)
