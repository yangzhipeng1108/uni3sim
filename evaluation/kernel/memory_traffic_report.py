"""Render cross-hardware measured-versus-Forward memory-traffic scatter plots.

One point per dataset case and metric: the profiler-measured and
Forward-predicted bytes of every matched kernel in the case are summed
separately.  Measured L2 is the normalized source-interface byte count and
measured HBM is the downstream global-memory byte count from the mechanism
observation; predicted bytes are the Forward resource demands for the
``l2_cache`` and ``dram`` roles.  FA serving capture rows carry op-level
evidence and cannot provide per-kernel mechanism captures; they are excluded
from the comparison with their case ids recorded in the summary.  The
comparison is evaluation evidence only; nothing here flows back into Forward
parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from evaluation.kernel.evaluation_runner import process_op_star
from evaluation.kernel.mechanism_validation_report import (
    _hardware_work, _memory_path_bytes, interleave_and_process)
from evaluation.kernel.dataset import (
    ATTENTION_OP_TYPES,
    MOE_OP_TYPES,
    VECTOR_OP_TYPES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARDWARE = (
    ("nvidia_a800", "NVIDIA A800"),
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
METRICS = (("l2", "L2"), ("hbm", "HBM"))
# The traffic gates are stated per kernel family, and FlashAttention's MLA
# path is a separate launch program from the non-MLA one (different operand
# geometry and a different predictor branch), so the two are reported apart.
# `case_id` is the only place the dataset records which one a case exercises.
GATE_FAMILIES = ("GEMM", "MoE", "FlashAttention (MLA)",
                 "FlashAttention (non-MLA)", "Vector")
# A case is "out of band" when its predicted/measured ratio leaves [0.67, 1.5]
# -- the same 1.5x window the goal tables count.
OUT_OF_BAND_LOW = 0.67
OUT_OF_BAND_HIGH = 1.5
MIB = 2 ** 20
_PLOT_DPI = 200
_FIGURE_TITLE_FONTSIZE = 14


def _family(op_type: str) -> str:
    if op_type == "GEMM":
        return "GEMM"
    if op_type in MOE_OP_TYPES:
        return "MoE"
    if op_type in ATTENTION_OP_TYPES:
        return "FlashAttention"
    if op_type in VECTOR_TYPES:
        return "Vector"
    raise ValueError(f"unsupported memory-traffic op type {op_type!r}")


def _gate_family(family: str, case_id: str) -> str:
    """Return the per-family gate bucket for one case.

    Splits the FlashAttention series into its MLA and non-MLA programs and
    passes every other family through unchanged.
    """

    if family not in ("GEMM", "MoE", "FlashAttention", "Vector"):
        raise ValueError(f"unsupported traffic family {family!r}")
    if family != "FlashAttention":
        return family
    return ("FlashAttention (MLA)" if "MLA" in case_id
            else "FlashAttention (non-MLA)")


def _finite_or_none(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _cold_comparable_hbm_bytes(hardware_id: str, family: str, mechanism):
    """Return the model's cold DRAM traffic for one kernel.

    The measured side of this figure is a flushed counter.  A Forward DRAM
    demand is only comparable to it when the modelled execution is cold too.
    On A800 it is not: the timing capture leaves `--timing-cache-eviction-mib`
    at zero, so a warmup loop leaves the operands in L2, and the Vector
    predictor credits that residency (`_resident_operand_dram_read_bytes`
    and `_completion_dependent_dram_write_bytes` are defined and called
    nowhere outside `vector.py`).  The credit is right for the latency target
    and wrong for this one.  Measured: a production elementwise launch
    reaches 2.29 TB/s at a 15 MiB footprint against a 0.90 TB/s cold ceiling
    (microbench/nvidia_a800/references/cold_single_pass_vector_drain_sm80.json).

    Rebuild the cold envelope for exactly that population from the kernel's
    own paths: cold, every byte the read path presents to L2 is a compulsory
    miss, while the store side is unchanged by cache history because whether
    a dirty line drains before the kernel ends is a capacity question the
    write path's DRAM node already answers.

    Everything else is returned unchanged.  Other families never take the
    credit -- verified on the A800 FA split pair, whose main and combine
    kernels predict 512.008 and 0.126 MiB against measured 512.117 and
    0.126 MiB -- and the other two targets have no equivalent cold/warm
    capture measurement, so correcting them would be a guess.

    ``compulsory_bytes`` is deliberately not used here: it is op-level for
    multi-kernel ops (that same FA combine kernel reports its op's full
    512 MiB), so it cannot be compared per kernel.
    """

    demand = mechanism["analytical_input"]["demand_per_resource"]
    if hardware_id != "nvidia_a800" or family != "Vector":
        return _finite_or_none(demand["dram"])
    # The Vector compulsory envelope is this kernel's unique traffic: its
    # unique read extent plus its stores at full extent.  Taking the read
    # side from it rather than from the read path's L2 ingress matters for
    # the two-pass norms, whose ingress also carries a second pass that hits
    # L2 and is not a DRAM miss even cold.  Swap only its full-extent store
    # term for the write path's DRAM node, which is the capacity-scoped
    # store the model already derives and cache history does not change.
    write_l2_bytes = 0.0
    write_dram_bytes = 0.0
    for path in mechanism["analytical_input"]["memory_paths"]:
        if not str(path["name"]).endswith("write"):
            continue
        for node in path["nodes"]:
            if node["resource_role"] == "dram":
                write_dram_bytes += float(node["bytes"])
            elif node["resource_role"] == "l2_cache":
                write_l2_bytes += float(node["bytes"])
    cold_bytes = (
        float(mechanism["compulsory_bytes"]) - write_l2_bytes
        + write_dram_bytes)
    if cold_bytes < 0.0:
        raise ValueError(
            "A800 Vector cold traffic envelope is negative: "
            f"{mechanism['kernel_name']!r}")
    return cold_bytes


def _predicted_l2_bytes(hardware_id: str, mechanism) -> float | None:
    """Return the Forward L2 demand comparable to this hardware's L2 counter.

    C500's canonical mcProfiler roofline exposes only ``l2c_memacs``, an
    upstream *read-side* request-byte count -- there is no mcProfiler L2
    write-byte counter, so
    the canonical converter (`dataset_pipeline/kernel_dataset/metax_c500.py`)
    stores the write side as 0 rather than fabricate one
    (`observation.l2_source_interface_bytes` is therefore read-only for this
    hardware).  Comparing that to the full ``demand_per_resource["l2_cache"]``
    (read+write) mixes two different physical quantities and is not a
    like-for-like traffic check; `mechanism_validation_report.py` already
    treats the aggregate ``metax_c500`` L2 role as
    non-semantically-comparable for the same reason
    (`SEMANTICALLY_EXCLUDED_COMPONENTS`) and instead validates the directional
    ``traffic:l2c_source_read`` component with this same
    ``_memory_path_bytes(access_kind="read", resource_role="l2_cache")``
    extraction.  Reuse it here so the family-level traffic report compares
    read against read for C500, while every other hardware keeps the
    unmodified aggregate demand (this must not change A800/910B/PIM
    predicted bytes at all).
    """

    demand = mechanism["analytical_input"]["demand_per_resource"]
    if hardware_id != "metax_c500":
        return _finite_or_none(demand["l2_cache"])
    return _finite_or_none(_memory_path_bytes(
        mechanism["analytical_input"],
        access_kind="read",
        resource_role="l2_cache",
    ))


def _kernel_traffic_points(hardware_id, result) -> list[dict[str, object]]:
    task_key, metadata, _oracle, forward_pair, _roofline, _pipeweave, kernels = (
        result)
    if not isinstance(forward_pair, tuple) or len(forward_pair) != 2:
        raise RuntimeError(f"task {task_key!r} has no Forward result")
    family = _family(str(metadata["op_type"]))
    points = []
    for kernel in kernels:
        mechanism = kernel["mechanisms"]
        try:
            observation = mechanism["observation"]
            points.append({
                "case_id": task_key,
                "family": family,
                "op_type": str(metadata["op_type"]),
                "kernel_ordinal": mechanism["kernel_ordinal"],
                "kernel_name": mechanism["kernel_name"],
                "predicted_l2_bytes": _predicted_l2_bytes(
                    hardware_id, mechanism),
                # The measured side is a cold launch: the profiler that
                # supplies `downstream_global_bytes` flushes caches between
                # replays.  `demand_per_resource["dram"]` is not that
                # quantity -- it is the traffic the *timed* execution serves,
                # and this dataset's timing capture is a warm launch whose
                # operands a preceding warmup left in L2 (measured: a
                # production elementwise launch reaches 2.29 TB/s at a 15 MiB
                # footprint against a 0.90 TB/s cold ceiling; see
                # microbench/nvidia_a800/references/
                # cold_single_pass_vector_drain_sm80.json).  Scoring it
                # against a flushed counter compares two cache states.
                #
                # The counter-comparable quantity is built from two values
                # every predictor already derives.  Cold, every unique input
                # byte misses, which is exactly the compulsory envelope minus
                # its own write term.  The store side is unchanged by cache
                # history: whether a dirty line drains before the kernel ends
                # is a capacity question, and the write path's DRAM node
                # already answers it.
                "predicted_hbm_bytes": _cold_comparable_hbm_bytes(
                    hardware_id, family, mechanism),
                "measured_l2_bytes": _finite_or_none(
                    observation["l2_source_interface_bytes"]),
                "measured_hbm_bytes": _finite_or_none(
                    observation["downstream_global_bytes"]),
            })
        except KeyError as error:
            raise KeyError(
                f"task {task_key!r} kernel mechanism is missing traffic "
                f"field {error}") from error
    return points


def collect_hardware_traffic(jobs: int) -> dict[str, dict[str, object]]:
    """Evaluate every canonical dataset case with mechanism capture enabled."""

    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("memory-traffic jobs must be an integer")
    if jobs <= 0:
        raise ValueError("memory-traffic jobs must be positive")
    work = {}
    maximum_cases = 0
    for hardware_id, _title in HARDWARE:
        hardware_config, packed_all = _hardware_work(hardware_id)
        packed = [
            item for item in packed_all if "capture_evidence" not in item[1]]
        work[hardware_id] = {
            "hardware_config": str(hardware_config),
            "dataset_case_count": len(packed),
            "fa_serving_capture_case_ids": sorted(
                str(item[0]) for item in packed_all
                if "capture_evidence" in item[1]),
            "packed": packed,
            "kernel_points": [],
        }
        maximum_cases = max(maximum_cases, len(packed))
    owners, results = interleave_and_process(
        work, [hardware_id for hardware_id, _ in HARDWARE],
        maximum_cases, jobs, process_op_star)
    for hardware_id, result in zip(owners, results, strict=True):
        work[hardware_id]["kernel_points"].extend(
            _kernel_traffic_points(hardware_id, result))
    return {
        hardware_id: {
            "hardware_config": entry["hardware_config"],
            "dataset_case_count": entry["dataset_case_count"],
            "fa_serving_capture_case_ids": entry[
                "fa_serving_capture_case_ids"],
            "kernel_points": entry["kernel_points"],
        }
        for hardware_id, entry in work.items()
    }


def aggregate_case_traffic(
        kernel_points: list[dict[str, object]],
        metric: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    """Sum matched kernel counters to one point per dataset case.

    Kernels missing either side of the comparison are unmatched and skipped;
    a case enters the plotted set only when at least one kernel matched and
    both summed traffics are positive.  Negative byte counts are invalid
    measurements and abort the report instead of being silently dropped.
    """

    if metric not in ("l2", "hbm"):
        raise ValueError(f"unknown traffic metric {metric!r}")
    measured_key = f"measured_{metric}_bytes"
    predicted_key = f"predicted_{metric}_bytes"
    grouped: dict[str, dict[str, object]] = {}
    unmatched_kernel_count = 0
    for point in kernel_points:
        case_id = str(point["case_id"])
        if case_id not in grouped:
            grouped[case_id] = {
                "case_id": case_id,
                "family": point["family"],
                "measured_bytes": 0.0,
                "predicted_bytes": 0.0,
                "matched_kernel_count": 0,
                "total_kernel_count": 0,
            }
        case = grouped[case_id]
        case["total_kernel_count"] += 1
        measured = point[measured_key]
        predicted = point[predicted_key]
        if measured is None or predicted is None:
            unmatched_kernel_count += 1
            continue
        if measured < 0.0 or predicted < 0.0:
            raise ValueError(
                f"{case_id!r} has negative {metric} traffic: "
                f"measured={measured}, predicted={predicted}")
        case["measured_bytes"] += measured
        case["predicted_bytes"] += predicted
        case["matched_kernel_count"] += 1
    cases = []
    excluded = []
    for case in grouped.values():
        if case["matched_kernel_count"] == 0:
            reason = "no_matched_kernel_traffic"
        elif case["measured_bytes"] <= 0.0 or case["predicted_bytes"] <= 0.0:
            reason = "non_positive_case_traffic"
        else:
            cases.append(case)
            continue
        excluded.append({
            "case_id": case["case_id"],
            "family": case["family"],
            "reason": reason,
            "matched_kernel_count": case["matched_kernel_count"],
            "total_kernel_count": case["total_kernel_count"],
        })
    return cases, excluded, unmatched_kernel_count


def aggregate_traffic(
        traffic_by_hardware: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    aggregated = {}
    for metric, _label in METRICS:
        per_hardware = {}
        for hardware_id, _title in HARDWARE:
            cases, excluded, unmatched = aggregate_case_traffic(
                traffic_by_hardware[hardware_id]["kernel_points"], metric)
            per_hardware[hardware_id] = {
                "cases": cases,
                "excluded_cases": excluded,
                "unmatched_kernel_count": unmatched,
            }
        aggregated[metric] = per_hardware
    return aggregated


def _summarize_cases(
        cases: list[dict[str, object]]) -> dict[str, float | int]:
    if not cases:
        raise ValueError("traffic summary requires at least one case")
    ratios = [
        float(case["predicted_bytes"]) / float(case["measured_bytes"])
        for case in cases
    ]
    absolute_errors = [abs(ratio - 1.0) * 100.0 for ratio in ratios]
    out_of_band_count = sum(
        1 for ratio in ratios
        if ratio < OUT_OF_BAND_LOW or ratio > OUT_OF_BAND_HIGH)
    return {
        "case_count": len(cases),
        "mape_percent": sum(absolute_errors) / len(absolute_errors),
        "median_ape_percent": median(absolute_errors),
        "mean_prediction_over_measured": sum(ratios) / len(ratios),
        # Added for the per-family gates; the four values above keep their
        # existing definitions untouched.
        "median_prediction_over_measured": median(ratios),
        "out_of_band_case_count": out_of_band_count,
    }


def build_traffic_summary(
        traffic_by_hardware: dict[str, dict[str, object]],
        aggregated: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    hardware_documents = []
    for hardware_id, _title in HARDWARE:
        entry = traffic_by_hardware[hardware_id]
        metrics = {}
        for metric, _label in METRICS:
            bucket = aggregated[metric][hardware_id]
            cases = bucket["cases"]
            # Per-family breakdown.  The traffic gates are stated per family,
            # and a whole-card mean hides the case a single rescale makes:
            # two families can both sit far from 1.0 while their pooled mean
            # looks correct.  Families with no case on this hardware are
            # omitted rather than reported as an empty summary.
            by_family: dict[str, list[dict[str, object]]] = {
                name: [] for name in GATE_FAMILIES}
            for case in cases:
                by_family[_gate_family(
                    str(case["family"]), str(case["case_id"]))].append(case)
            metrics[metric] = {
                **_summarize_cases(cases),
                "unmatched_kernel_count": bucket["unmatched_kernel_count"],
                "excluded_cases": bucket["excluded_cases"],
                "families": {
                    name: _summarize_cases(family_cases)
                    for name, family_cases in by_family.items()
                    if family_cases
                },
            }
        hardware_documents.append({
            "hardware_id": hardware_id,
            "hardware_config": entry["hardware_config"],
            "dataset_case_count": entry["dataset_case_count"],
            "fa_serving_capture_case_ids": entry[
                "fa_serving_capture_case_ids"],
            "kernel_count": len(entry["kernel_points"]),
            "metrics": metrics,
        })
    return {
        "schema_version": 1,
        "traffic_definition": {
            "l2_measured": "observation.l2_source_interface_bytes",
            "l2_predicted": (
                "analytical_input.demand_per_resource.l2_cache, except "
                "metax_c500 which uses the read-only "
                "analytical_input.memory_paths sum "
                "(access_kind=read, resource_role=l2_cache): mcProfiler's "
                "l2c_memacs is a read-side-only request-byte counter, so "
                "l2_source_interface_bytes "
                "is read-only for this hardware and the read+write aggregate "
                "demand is not a like-for-like comparison for it"),
            "hbm_measured": "observation.downstream_global_bytes",
            "hbm_predicted": "analytical_input.demand_per_resource.dram",
            "aggregation": (
                "sum measured and predicted bytes separately over each "
                "dataset case's matched kernels"),
        },
        "hardware": hardware_documents,
    }


def render_memory_traffic(
        aggregated: dict[str, dict[str, dict[str, object]]],
        output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import LogFormatterMathtext, LogLocator
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "kernel memory-traffic rendering requires matplotlib") from error

    for metric, _label in METRICS:
        for hardware_id, _title in HARDWARE:
            cases = aggregated[metric][hardware_id]["cases"]
            if not cases:
                raise ValueError(
                    f"{hardware_id} has no {metric} traffic cases")
            families = {str(case["family"]) for case in cases}
            for family, _color, _marker in SERIES:
                if family not in families:
                    raise ValueError(
                        f"{hardware_id} has no {metric} traffic cases for "
                        f"{family}")

    limits = {}
    for metric, _label in METRICS:
        values = [
            float(value) / MIB
            for hardware_id, _title in HARDWARE
            for case in aggregated[metric][hardware_id]["cases"]
            for value in (case["measured_bytes"], case["predicted_bytes"])
        ]
        limits[metric] = (
            10 ** math.floor(math.log10(min(values))),
            10 ** math.ceil(math.log10(max(values))),
        )

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })
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
    figure, axes = plt.subplots(
        len(METRICS), len(HARDWARE), figsize=(14.4, 9.8), squeeze=False,
    )
    for row, (metric, metric_label) in enumerate(METRICS):
        lower, upper = limits[metric]
        decade_ticks = [
            10.0 ** exponent
            for exponent in range(
                math.floor(math.log10(lower)),
                math.ceil(math.log10(upper)) + 1,
            )
        ]
        for column, (hardware_id, title) in enumerate(HARDWARE):
            axis = axes[row, column]
            cases = aggregated[metric][hardware_id]["cases"]
            for family, color, marker in SERIES:
                selected = [
                    case for case in cases if case["family"] == family]
                axis.scatter(
                    [float(case["measured_bytes"]) / MIB for case in selected],
                    [float(case["predicted_bytes"]) / MIB for case in selected],
                    s=12, alpha=0.55, c=color, marker=marker,
                    edgecolors="none", rasterized=True,
                )
            axis.plot(
                [lower, upper], [lower, upper],
                color="#666666", linewidth=1.2, linestyle="--", zorder=0,
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_xticks(decade_ticks)
            axis.set_yticks(decade_ticks)
            axis.xaxis.set_major_formatter(LogFormatterMathtext())
            axis.yaxis.set_major_formatter(LogFormatterMathtext())
            axis.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=range(2, 10)))
            axis.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=range(2, 10)))
            axis.set_aspect("equal", adjustable="box")
            axis.grid(
                which="major", color="#b8b8b8", linewidth=0.75, alpha=0.52)
            axis.grid(
                which="minor", color="#c7c7c7", linewidth=0.45,
                alpha=0.68, linestyle=":")
            summary = _summarize_cases(cases)
            axis.text(
                0.04, 0.96,
                f"n={summary['case_count']:,}  "
                f"MAPE={summary['mape_percent']:.1f}%\n"
                f"MedAPE={summary['median_ape_percent']:.1f}%  "
                f"mean P/M={summary['mean_prediction_over_measured']:.3f}×",
                transform=axis.transAxes, ha="left", va="top", fontsize=8.2,
            )
            if row == 0:
                axis.set_title(title)
            if row == len(METRICS) - 1:
                axis.set_xlabel(f"Measured {metric_label} traffic (MiB)")
            if column == 0:
                axis.set_ylabel(f"Predicted {metric_label} traffic (MiB)")
    figure.suptitle(
        "Kernel memory traffic — Forward versus measured, per dataset case",
        y=1.04,
        fontweight="bold",
        fontsize=_FIGURE_TITLE_FONTSIZE,
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
        fontsize=11,
    )
    figure.subplots_adjust(
        left=0.07, right=0.995, bottom=0.065, top=0.90,
        hspace=0.30, wspace=0.16,
    )
    for axis in figure.axes:
        axis.title.set_fontsize(axis.title.get_fontsize() + 1.0)
        axis.title.set_fontweight("bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=_PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot measured versus Forward L2/HBM memory traffic for the "
            "three canonical production hardware datasets."))
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "experiment_logs/evaluation/kernel/cross_hardware/kernel-memory-traffic.png"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="traffic summary JSON; defaults beside --output",
    )
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    traffic = collect_hardware_traffic(args.jobs)
    aggregated = aggregate_traffic(traffic)
    document = build_traffic_summary(traffic, aggregated)
    summary_output = args.summary_output
    if summary_output is None:
        summary_output = args.output.with_suffix(".json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_memory_traffic(aggregated, args.output)
    print(args.output)
    print(summary_output)
    return 0
