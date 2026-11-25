"""Cumulative serving-mechanism ablation ladder for one ShareGPT case.

Each rung is an independent closed-loop simulation of a cumulative runtime
configuration: kernel only, then graph replay, graph padding, the Scheduler
host CPU and finally the asynchronous scheduling depth.  Rungs are cumulative
configurations, not additive timing components.  Rungs whose graph capture
differs from the captured deployment are counterfactual runtimes and are
compared without provenance validation; the captured rungs go through the
validated case evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from evaluation.serving.metrics import sharegpt_compact_evaluation as compact
from evaluation.serving.simulation import simulation_runner



# (label, serving timing mode, graph capture, Scheduler pipeline depth,
#  async scheduling); depth and async None keep the captured runtime.
LADDER = (
    ("Kernel\nonly", "device_only", "none", 1, False),
    ("+Graph", "device_plus_launch", "exact", 1, False),
    ("+Graph\npadding", "device_plus_launch", "captured", 1, False),
    ("+Sched\nCPU", "device_plus_launch_plus_scheduler", "captured", 1, False),
    ("+Async\ndepth", "device_plus_launch_plus_scheduler", "captured", None,
     None),
)


def _flag_span(argv: list[str], flag: str) -> tuple[int, int]:
    start = argv.index(flag)
    end = start + 1
    while end < len(argv) and not argv[end].startswith("--"):
        end += 1
    return start, end


def _patch(argv, *, capture: str, depth: int | None, async_scheduling,
           max_capture: int) -> list[str]:
    argv = list(argv)
    if capture == "none":
        argv[argv.index("--graph-mode") + 1] = "NONE"
        for flag in ("--graph-capture-sizes",
                     "--graph-piecewise-eager-workload-kinds"):
            if flag in argv:
                start, end = _flag_span(argv, flag)
                del argv[start:end]
    elif capture == "exact":
        start, end = _flag_span(argv, "--graph-capture-sizes")
        argv[start:end] = ["--graph-capture-sizes",
                           *(str(n) for n in range(1, max_capture + 1))]
    elif capture != "captured":
        raise ValueError(f"unknown capture mode: {capture!r}")
    if depth is not None:
        argv[argv.index("--scheduler-pipeline-depth") + 1] = str(depth)
    if async_scheduling is False and "--vllm-async-scheduling" in argv:
        argv[argv.index("--vllm-async-scheduling")] = (
            "--no-vllm-async-scheduling")
    return argv


def run_mechanism_ablation(
    root: str | Path,
    case_id: str,
    load_validated_case: Callable[[Path], object],
) -> dict[str, object]:
    """Run every ladder rung for ``case_id`` and return the rung means."""

    root = Path(root)
    matches = [
        (config, cell)
        for config, cells in compact.official_metrics_suite_runs(root)
        for cell in cells
        if compact._canonical_cell_id(cell) == case_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one official serving case for {case_id}")
    config, cell = matches[0]
    case = load_validated_case(compact.compact_case_path(root, config, cell))
    commit = compact.repository_source_commit(root)
    hardware_id = cell.hardware.hardware_id
    launch = compact.resolve_host_launch_profile_paths(
        root, [], {hardware_id})[hardware_id]
    scheduler = compact.resolve_scheduler_host_profile_paths(
        root, [], {hardware_id})[hardware_id]
    observed = compact._observed_metrics(case)
    max_capture = max(cell.hardware.graph_execution.capture_sizes)

    rungs = []
    violation = None
    for label, mode, capture, depth, async_scheduling in LADDER:
        host_launch = None if mode == "device_only" else launch
        scheduler_host = (scheduler
                          if mode == "device_plus_launch_plus_scheduler"
                          else None)

        def run(argv, capture=capture, depth=depth,
                async_scheduling=async_scheduling):
            return simulation_runner.run_cli_args(
                _patch(argv, capture=capture, depth=depth,
                       async_scheduling=async_scheduling,
                       max_capture=max_capture))

        if capture == "captured":
            result = compact.evaluate_validated_case(
                root, config, cell, case, run,
                source_commit=commit, alignment_checked=False,
                execution_mode="accelerated",
                serving_timing_mode=mode,
                host_launch_profile_path=host_launch,
                scheduler_host_profile_path=scheduler_host,
            )
            latency = result["metrics"]["latency"]
            status = result["prediction_status"]
            validated = True
        else:
            argv = compact.build_simulation_argv(
                root, config, cell, case,
                execution_mode="accelerated", serving_timing_mode=mode,
                host_launch_profile_path=host_launch,
                scheduler_host_profile_path=scheduler_host)
            latency = compact._compare_prediction(
                observed, run(argv), len(case.requests))["latency"]
            status = "counterfactual_runtime"
            validated = False

        rungs.append({
            "label": label,
            "serving_timing_mode": mode,
            "capture_mode": capture,
            "scheduler_pipeline_depth": depth,
            "async_scheduling": async_scheduling,
            "prediction_status": status,
            "provenance_validated": validated,
            "ttft_ms": latency["ttft_ms"]["mean"]["predicted"],
            "tpot_ms": latency["tpot_ms"]["mean"]["predicted"],
        })
        if label == "+Graph\npadding" and len(rungs) > 1:
            # Recorded, not raised: the monotonicity outcome is itself the
            # comparable result across cases.
            graph_replay = rungs[-2]
            decreasing = {
                metric: (rungs[-1][metric], graph_replay[metric])
                for metric in ("ttft_ms", "tpot_ms")
                if rungs[-1][metric] < graph_replay[metric]
            }
            if decreasing:
                violation = decreasing

    return {
        "schema_version": 2,
        "cell_id": case_id,
        "source_commit": commit,
        "independent_simulation_per_rung": True,
        "note": "rungs are cumulative configurations, not additive terms",
        "graph_padding_monotonicity_violation": violation,
        "observed": {
            "ttft_ms": latency["ttft_ms"]["mean"]["observed"],
            "tpot_ms": latency["tpot_ms"]["mean"]["observed"],
        },
        "rungs": rungs,
    }
