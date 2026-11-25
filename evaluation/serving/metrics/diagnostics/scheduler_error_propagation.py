"""Measure how dynamic Scheduling propagates synthetic step-duration errors."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence

from evaluation.serving.metrics.diagnostics.scheduler_depth_breakdown import (
    _prepare_final_serving_quantile_case,
)
from evaluation.serving.simulation import simulation_runner
from src.serving.execution import ServingExecutionComponents, ServingExecutionContext
from src.serving.kernel_lowering.execution_timeline import ExecutionTimeline
from src.serving.schedule.schedule_step import ScheduleStep


CELL_IDS = (
    "nvidia_a800/qwen3-4b/tp1_qps4",
    "nvidia_a800/qwen3-4b/tp1_qps16",
    "nvidia_a800/qwen3-4b/tp1_qps64",
)
CONFIG_PATH = "configs/evaluation/experiments/sharegpt_phase1.json"
HOST_PROFILE_PATH = (
    "configs/host_timing/host_launch_profiles/nvidia_a800.json"
)
SCENARIOS = {
    "oracle": (0.0, 0.0),
    "fixed_minus_10": (-0.10, 0.0),
    "fixed_plus_10": (0.10, 0.0),
    "random_unbiased_10": (0.0, 0.10),
    "random_minus_10_sigma_10": (-0.10, 0.10),
    "random_plus_10_sigma_10": (0.10, 0.10),
}
SCENARIO_LABELS = {
    "oracle": "Oracle\n$\\mu=0\\%,\\ \\sigma=0\\%$",
    "fixed_minus_10": "$\\mu=-10\\%$\n$\\sigma=0\\%$",
    "fixed_plus_10": "$\\mu=+10\\%$\n$\\sigma=0\\%$",
    "random_unbiased_10": "$\\mu=0\\%$\n$\\sigma=10\\%$",
    "random_minus_10_sigma_10": "$\\mu=-10\\%$\n$\\sigma=10\\%$",
    "random_plus_10_sigma_10": "$\\mu=+10\\%$\n$\\sigma=10\\%$",
}


def _step_normal(seed: int, schedule: ScheduleStep) -> float:
    request_signature = tuple(
        (
            request.request_id,
            request.phase,
            request.input_tokens,
            request.output_tokens,
            request.computed_tokens,
        )
        for request in schedule.requests
    )
    signature = repr(
        (seed, schedule.step_id, schedule.pool_name, request_signature)
    ).encode("utf-8")
    digest = hashlib.sha256(signature).digest()
    first = int.from_bytes(digest[:8], "big")
    second = int.from_bytes(digest[8:16], "big")
    denominator = float(1 << 64)
    u1 = (first + 0.5) / denominator
    u2 = (second + 0.5) / denominator
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _synthetic_oracle_duration_s(schedule: ScheduleStep) -> float:
    """Return a transparent shape-aware duration without lowering kernels."""

    prefill_tokens = sum(request.input_tokens for request in schedule.requests)
    decode_tokens = sum(request.output_tokens for request in schedule.requests)
    return 0.0015 + 3.5e-6 * prefill_tokens + 80.0e-6 * decode_tokens


def _request_metrics(
    payload: Mapping[str, object],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise TypeError("simulation metrics must be a mapping")
    requests = metrics["requests"]
    if not isinstance(requests, Mapping) or not requests:
        raise TypeError("simulation request metrics must be a non-empty mapping")
    ttft: list[float] = []
    tpot: list[float] = []
    per_request = []
    for request_id in sorted(requests):
        request = requests[request_id]
        if not isinstance(request, Mapping):
            raise TypeError("per-request metrics must be mappings")
        ttft_ms = None
        tpot_ms = None
        if request["ttft_s"] is not None:
            ttft_ms = 1000.0 * float(request["ttft_s"])
            ttft.append(ttft_ms)
        if request["tpot_s"] is not None:
            tpot_ms = 1000.0 * float(request["tpot_s"])
            tpot.append(tpot_ms)
        per_request.append(
            {
                "request_id": request_id,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
            }
        )
    if not ttft or not tpot:
        raise ValueError("target case must provide TTFT and TPOT samples")
    return (
        {
            "mean_ttft_ms": statistics.fmean(ttft),
            "mean_tpot_ms": statistics.fmean(tpot),
            "request_throughput_per_s": float(
                metrics["request_throughput_per_s"]
            ),
        },
        per_request,
    )


def run_scheduler_error_propagation_worker(
    repository_root: Path,
    cell_id: str,
    scenario: str,
    seed: int,
    output_path: Path,
    load_validated_case: Callable[..., object],
) -> None:
    bias, sigma = SCENARIOS[scenario]
    case_path = f"serving_dataset/sharegpt/{cell_id}"
    _, _, _, _, argv = _prepare_final_serving_quantile_case(
        str(repository_root),
        str(repository_root / CONFIG_PATH),
        str(repository_root / case_path),
        cell_id,
        str(repository_root / HOST_PROFILE_PATH),
        load_validated_case,
    )
    timing_index = argv.index("--serving-timing-mode") + 1
    argv[timing_index] = "device_only"
    profile_index = argv.index("--host-launch-profile")
    del argv[profile_index : profile_index + 2]
    argv.extend(("--step-retention", "full", "--execution-workers", "1"))
    original_execute_step = ServingExecutionContext.execute_step
    multipliers: list[float] = []
    oracle_durations_s: list[float] = []

    def execute_step_with_error(
        context: ServingExecutionContext,
        schedule: ScheduleStep,
        **kwargs: object,
    ) -> ServingExecutionComponents:
        del context, kwargs
        oracle_duration_s = _synthetic_oracle_duration_s(schedule)
        multiplier = 1.0 + bias
        if sigma:
            multiplier += sigma * _step_normal(seed, schedule)
        if multiplier <= 0.0:
            raise ValueError("synthetic step-duration multiplier must be positive")
        multipliers.append(multiplier)
        oracle_durations_s.append(oracle_duration_s)
        return ServingExecutionComponents(
            workload=SimpleNamespace(nodes=(), block_repeats=()),
            kernel_plan=SimpleNamespace(calls=(), logical_call_count=0),
            predictions={},
            execution_timeline=ExecutionTimeline(
                intervals=(),
                duration_s=oracle_duration_s * multiplier,
                predicted_device_time_s=oracle_duration_s * multiplier,
            ),
        )

    ServingExecutionContext.execute_step = execute_step_with_error
    original_system_resource_manifest = simulation_runner.system_resource_manifest
    original_critical_path_manifest = simulation_runner.critical_path_manifest
    original_execution_artifact_manifest = simulation_runner.execution_artifact_manifest
    simulation_runner.system_resource_manifest = lambda *_args, **_kwargs: {
        "availability": "omitted_for_step_error_counterfactual"
    }
    simulation_runner.critical_path_manifest = lambda *_args, **_kwargs: {
        "availability": "omitted_for_step_error_counterfactual"
    }
    simulation_runner.execution_artifact_manifest = lambda *_args, **_kwargs: {
        "artifact_sha256": "0" * 64,
        "availability": "omitted_for_step_error_counterfactual",
    }
    try:
        payload = simulation_runner.run_cli_args(argv)
    finally:
        ServingExecutionContext.execute_step = original_execute_step
        simulation_runner.system_resource_manifest = original_system_resource_manifest
        simulation_runner.critical_path_manifest = original_critical_path_manifest
        simulation_runner.execution_artifact_manifest = original_execution_artifact_manifest
    if not multipliers:
        raise RuntimeError("simulation did not execute any modeled steps")
    summary_metrics, per_request = _request_metrics(payload)
    result = {
        "schema_version": 3,
        "kind": "scheduler_step_error_propagation_run",
        "source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repository_root, text=True
        ).strip(),
        "cell_id": cell_id,
        "canonical_case_path": str((repository_root / case_path).resolve()),
        "scenario": scenario,
        "seed": seed,
        "configured_step_bias_fraction": bias,
        "configured_step_sigma_fraction": sigma,
        "oracle_step_duration_model": {
            "kind": "direct_shape_analytical",
            "fixed_seconds": 0.0015,
            "prefill_seconds_per_token": 3.5e-6,
            "decode_seconds_per_token": 80.0e-6,
            "kernel_lowering": "not_executed",
        },
        "oracle_step_duration_mean_ms": 1000.0
        * statistics.fmean(oracle_durations_s),
        "realized_step_error_mean_percent": 100.0
        * statistics.fmean(multiplier - 1.0 for multiplier in multipliers),
        "realized_step_error_std_percent": 100.0
        * statistics.pstdev(multiplier - 1.0 for multiplier in multipliers),
        "modeled_step_count": len(multipliers),
        "metrics": summary_metrics,
        "per_request": per_request,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _relative_error(value: float, oracle: float) -> float:
    return 100.0 * (value - oracle) / oracle


def _cdf_summary(
    scenario_runs: Sequence[Mapping[str, object]], metric: str
) -> dict[str, list[float]]:
    sorted_runs = []
    for run in scenario_runs:
        values = sorted(
            float(request[metric])
            for request in run["per_request"]
            if request[metric] is not None
        )
        if not values:
            raise ValueError(f"CDF metric has no samples: {metric}")
        sorted_runs.append(values)
    sample_count = len(sorted_runs[0])
    if any(len(values) != sample_count for values in sorted_runs):
        raise ValueError(f"CDF sample counts disagree for {metric}")
    return {
        "probability": [
            (index + 1) / sample_count for index in range(sample_count)
        ],
        "mean_ms": [
            statistics.fmean(values[index] for values in sorted_runs)
            for index in range(sample_count)
        ],
        "minimum_ms": [
            min(values[index] for values in sorted_runs)
            for index in range(sample_count)
        ],
        "maximum_ms": [
            max(values[index] for values in sorted_runs)
            for index in range(sample_count)
        ],
    }


def _aggregate(run_paths: Sequence[Path], output_dir: Path) -> Path:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in run_paths]
    oracle_runs = [run for run in runs if run["scenario"] == "oracle"]
    if len(oracle_runs) != 1:
        raise ValueError("exactly one Oracle run is required")
    oracle = oracle_runs[0]["metrics"]
    rows = []
    for scenario in SCENARIOS:
        scenario_runs = [run for run in runs if run["scenario"] == scenario]
        if not scenario_runs:
            raise ValueError(f"missing scenario runs: {scenario}")
        metric_errors = {
            "mean_ttft_ms": [
                _relative_error(run["metrics"]["mean_ttft_ms"], oracle["mean_ttft_ms"])
                for run in scenario_runs
            ],
            "mean_tpot_ms": [
                _relative_error(run["metrics"]["mean_tpot_ms"], oracle["mean_tpot_ms"])
                for run in scenario_runs
            ],
            "request_throughput_per_s": [
                _relative_error(
                    run["metrics"]["request_throughput_per_s"],
                    oracle["request_throughput_per_s"],
                )
                for run in scenario_runs
            ],
        }
        step_errors = [run["realized_step_error_mean_percent"] for run in scenario_runs]
        rows.append(
            {
                "scenario": scenario,
                "label": SCENARIO_LABELS[scenario].replace("\n", " "),
                "run_count": len(scenario_runs),
                "step_error_mean_percent": statistics.fmean(step_errors),
                "step_error_range_percent": [min(step_errors), max(step_errors)],
                "metric_error_percent": {
                    name: {
                        "mean": statistics.fmean(values),
                        "range": [min(values), max(values)],
                    }
                    for name, values in metric_errors.items()
                },
                "cdf": {
                    "ttft_ms": _cdf_summary(scenario_runs, "ttft_ms"),
                    "tpot_ms": _cdf_summary(scenario_runs, "tpot_ms"),
                },
            }
        )
    payload = {
        "schema_version": 3,
        "kind": "scheduler_step_error_propagation",
        "source_commit": oracle_runs[0]["source_commit"],
        "cell_id": oracle_runs[0]["cell_id"],
        "canonical_case_path": oracle_runs[0]["canonical_case_path"],
        "oracle_metrics": oracle,
        "cdf_metric_error_references": {
            "source": "per-request Oracle metric multiplied directly",
            "scale_factors": [0.9, 1.1],
        },
        "rows": rows,
    }
    json_path = output_dir / "scheduler_error_propagation.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(payload, output_dir / "scheduler_error_propagation.png")
    return json_path


def _plot(payload: Mapping[str, object], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = payload["rows"]
    labels = [SCENARIO_LABELS[row["scenario"]] for row in rows]
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 13.5))
    bar_panels = (
        ("step", "Mean step-duration error", "Step duration error (%)", axes[0, 0]),
        (
            "request_throughput_per_s",
            "Request throughput error",
            "Req/s error (%)",
            axes[0, 1],
        ),
        ("mean_ttft_ms", "Mean TTFT error", "TTFT error (%)", axes[1, 0]),
        ("mean_tpot_ms", "Mean TPOT error", "TPOT error (%)", axes[1, 1]),
    )
    negative_color = "#0072B2"
    neutral_color = "#404040"
    positive_color = "#D55E00"
    colors = [
        "#000000",
        negative_color,
        positive_color,
        neutral_color,
        negative_color,
        positive_color,
    ]
    line_styles = ["-", "-", "-", "--", "--", "--"]
    markers = ["o", "s", "^", "D", "v", "P"]
    hatches = [None, None, None, "///", "///", "///"]
    for metric, title, ylabel, axis in bar_panels:
        means = []
        lows = []
        highs = []
        for row in rows:
            if metric == "step":
                mean = row["step_error_mean_percent"]
                low, high = row["step_error_range_percent"]
            elif metric in {
                "request_throughput_per_s",
                "mean_ttft_ms",
                "mean_tpot_ms",
            }:
                summary = row["metric_error_percent"][metric]
                mean = summary["mean"]
                low, high = summary["range"]
            else:
                raise ValueError(f"unsupported error propagation metric: {metric}")
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)
        x = range(len(rows))
        axis.axhline(0.0, color="#555555", linewidth=0.9)
        bars = axis.bar(x, means, color=colors, alpha=0.82)
        for bar, hatch in zip(bars, hatches):
            if hatch is not None:
                bar.set_hatch(hatch)
        for index, row in enumerate(rows):
            if row["run_count"] > 1:
                axis.errorbar(
                    index,
                    means[index],
                    yerr=([lows[index]], [highs[index]]),
                    fmt="none",
                    ecolor="#202020",
                    elinewidth=1.2,
                    capsize=3,
                    clip_on=True,
                )
        axis.set_title(title, fontsize=15)
        axis.set_ylabel(ylabel, fontsize=12)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(list(x), labels, fontsize=9)
        lower_bound = min(
            0.0,
            *(mean - low for mean, low in zip(means, lows)),
        )
        upper_bound = max(
            0.0,
            *(mean + high for mean, high in zip(means, highs)),
        )
        span = max(upper_bound - lower_bound, 1.0)
        axis.set_ylim(lower_bound - 0.14 * span, upper_bound + 0.18 * span)
        for index, mean in enumerate(means):
            if mean >= 0:
                anchor = mean + highs[index]
                vertical = 4
                alignment = "bottom"
            else:
                anchor = mean - lows[index]
                vertical = -4
                alignment = "top"
            axis.annotate(
                f"{mean:+.1f}%",
                (index, anchor),
                xytext=(0, vertical),
                textcoords="offset points",
                ha="center",
                va=alignment,
                fontsize=9,
                clip_on=True,
            )
    for metric, title, axis in (
        ("ttft_ms", "TTFT CDF", axes[2, 0]),
        ("tpot_ms", "TPOT CDF", axes[2, 1]),
    ):
        for index, (color, row) in enumerate(zip(colors, rows)):
            cdf = row["cdf"][metric]
            scenario = row["scenario"]
            if scenario == "oracle":
                linewidth = 3.2
                zorder = 5
            elif scenario in SCENARIOS:
                linewidth = 1.8
                zorder = 3
            else:
                raise ValueError(
                    f"unsupported error propagation scenario: {scenario}"
                )
            axis.plot(
                cdf["probability"],
                cdf["mean_ms"],
                color=color,
                linewidth=linewidth,
                linestyle=line_styles[index],
                marker=markers[index],
                markevery=16,
                markersize=5,
                zorder=zorder,
            )
            if row["run_count"] > 1:
                axis.fill_between(
                    cdf["probability"],
                    cdf["minimum_ms"],
                    cdf["maximum_ms"],
                    color=color,
                    alpha=0.10,
                )
        oracle_cdf = rows[0]["cdf"][metric]
        for scale in (
            0.9,
            1.1,
        ):
            axis.plot(
                oracle_cdf["probability"],
                [scale * value for value in oracle_cdf["mean_ms"]],
                color=negative_color if scale < 1.0 else positive_color,
                linewidth=2.4,
                linestyle=":",
                zorder=4,
            )
        axis.set_title(title, fontsize=15)
        axis.set_xlabel("CDF", fontsize=12)
        axis.set_ylabel(
            f"{metric.removesuffix('_ms').upper()} (ms)", fontsize=12
        )
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.25)
        semantic_legend = (
            Line2D([], [], color=negative_color, linewidth=2.5, label="$\\mu=-10\\%$"),
            Line2D([], [], color=neutral_color, linewidth=2.5, label="$\\mu=0\\%$"),
            Line2D([], [], color=positive_color, linewidth=2.5, label="$\\mu=+10\\%$"),
            Line2D([], [], color="#303030", linewidth=2.2, linestyle="-", label="$\\sigma=0\\%$"),
            Line2D([], [], color="#303030", linewidth=2.2, linestyle="--", label="$\\sigma=10\\%$"),
            Line2D([], [], color="#303030", linewidth=2.2, linestyle=":", label="Metric exactly ±10%"),
        )
        axis.legend(
            handles=semantic_legend,
            loc="upper left",
            frameon=False,
            fontsize=9,
            ncol=2,
        )
    fig.suptitle(
        "Scheduler-only step-error propagation · per-step "
        "$\\epsilon \\sim N(\\mu, \\sigma^2)$ · random cases ×3\n"
        f"A800 · Qwen3-4B · {payload['cell_id'].split('/')[-1].upper()} "
        "· 128 requests",
        fontsize=16,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def aggregate_scheduler_error_propagation(
    run_paths: Sequence[Path], output_dir: Path
) -> Path:
    return _aggregate(run_paths, output_dir)


def run_scheduler_error_propagation_suite(
    repository_root: Path,
    cell_id: str,
    output_dir: Path,
    seeds: Sequence[int],
    jobs: int,
    worker_command: Sequence[str],
) -> None:
    if not worker_command:
        raise ValueError("scheduler error propagation worker command is empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [("oracle", 0), ("fixed_minus_10", 0), ("fixed_plus_10", 0)]
    tasks.extend(
        (scenario, seed)
        for scenario in (
            "random_unbiased_10",
            "random_minus_10_sigma_10",
            "random_plus_10_sigma_10",
        )
        for seed in seeds
    )
    pending = []
    completed_paths = []
    for scenario, seed in tasks:
        path = output_dir / "runs" / f"{scenario}_seed_{seed}.json"
        reusable = False
        if path.is_file():
            cached = json.loads(path.read_text(encoding="utf-8"))
            reusable = cached["schema_version"] == 3
        if reusable:
            completed_paths.append(path)
        else:
            pending.append((scenario, seed))
    running: list[tuple[subprocess.Popen[bytes], Path]] = []
    while pending or running:
        while pending and len(running) < jobs:
            scenario, seed = pending.pop(0)
            path = output_dir / "runs" / f"{scenario}_seed_{seed}.json"
            command = (
                *worker_command,
                "--worker",
                "--repository-root",
                str(repository_root),
                "--cell-id",
                cell_id,
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--output",
                str(path),
            )
            running.append((subprocess.Popen(command), path))
        process, path = running.pop(0)
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, process.args)
        completed_paths.append(path)
    aggregate_scheduler_error_propagation(completed_paths, output_dir)
