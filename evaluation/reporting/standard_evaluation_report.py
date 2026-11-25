"""Assemble separate canonical Kernel and full-only Serving reports."""

from __future__ import annotations

from evaluation.serving.metrics.sharegpt_evaluation_report import (
    MODEL_RANK,
    has_scheduler_progress_evidence,
)

from datetime import datetime
import json
import os
from pathlib import Path
from statistics import fmean
import time
from typing import Mapping, Sequence
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from evaluation.json_shape import require_object as _mapping

from evaluation.json_shape import require_finite_number as _finite

from evaluation.json_shape import require_list_or_tuple as _sequence


HARDWARE_ORDER = EVALUATED_HARDWARE_IDS
# Kernel evaluation also reports nvidia_h200.  The three paper hardware are
# required; the others are rendered from whatever status their card produced
# (PASS, SKIPPED or FAIL) rather than dropped.
KERNEL_HARDWARE_ORDER = (
    "nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")
REQUIRED_KERNEL_HARDWARE = HARDWARE_ORDER
_KERNEL_SHORT_LABELS = {
    "nvidia_a800": "A800",
    "nvidia_h200": "H200",
    "metax_c500": "C500",
    "ascend_910b": "910B",
}
# Hardware whose kernel records are NSYS launch facts only (no Nsight Compute
# L2/HBM counters), so they have no measured traffic to compare against.
_LAUNCH_ONLY_KERNEL_HARDWARE = ("nvidia_h200",)
# Serving additionally reports H200 when a run selects an H200 candidate root.
SERVING_HARDWARE_ORDER = (*HARDWARE_ORDER, "nvidia_h200")
HARDWARE_LABELS = {
    "nvidia_a800": "NVIDIA A800",
    "nvidia_h200": "NVIDIA H200",
    "metax_c500": "MetaX C500",
    "ascend_910b": "Ascend 910B",
    "nvidia_h200": "NVIDIA H200",
}
MIDLAYER_WALL_TIME_PHASES = (
    "workload_construction",
    "kernel_lowering",
    "kernel_prediction",
    "timeline_construction",
    "artifact_generation",
    "scheduler_and_orchestration",
)


def write_kernel_evaluation_report(
    output_path: str | Path,
    state_value: Mapping[str, object],
    *,
    with_html: bool = False,
) -> float:
    """Validate and write the standalone Kernel evaluation report."""

    started = time.perf_counter()
    output = Path(output_path).resolve()
    state = _validate_state(state_value)
    kernel = _validate_component(state["kernel"], "kernel")
    kernel_wall_times = _mapping(
        kernel["hardware_evaluation_wall_time_s"],
        "Kernel hardware evaluation wall times",
    )
    if (not set(REQUIRED_KERNEL_HARDWARE) <= set(kernel_wall_times)
            or not set(kernel_wall_times) <= set(KERNEL_HARDWARE_ORDER)):
        raise ValueError("Kernel hardware wall-time set is invalid")
    for hardware_id in kernel_wall_times:
        if _finite(
            kernel_wall_times[hardware_id], f"{hardware_id} Kernel wall time"
        ) < 0.0:
            raise ValueError("Kernel hardware wall time must be nonnegative")
    kernel_payload = _mapping(
        json.loads(Path(str(kernel["result_path"])).read_text()),
        "kernel result",
    )
    traffic_payload = _mapping(
        json.loads(Path(str(kernel["traffic_summary_path"])).read_text()),
        "kernel memory traffic summary",
    )
    lines = _render_kernel_markdown(
        output, state, kernel, kernel_payload, traffic_payload,
        report_assembly_wall_time_s=0.0,
        with_html=with_html,
    )
    if with_html:
        from evaluation.reporting.kernel_overview import write_kernel_matrix_overview
        write_kernel_matrix_overview(kernel_payload, {"state": state, "kernel": kernel}, output.with_suffix(".html"))
    assembly_wall = time.perf_counter() - started
    lines = _render_kernel_markdown(
        output, state, kernel, kernel_payload, traffic_payload,
        report_assembly_wall_time_s=assembly_wall,
        with_html=with_html,
    )
    _write_markdown_atomic(output, lines)
    return time.perf_counter() - started


def write_serving_evaluation_report(
    output_path: str | Path,
    state_value: Mapping[str, object],
    *,
    llmservingsim_results: Path | None = None,
    refresh_latency_plots: bool = False,
) -> float:
    """Validate and write the standalone full-only Serving report."""
    started = time.perf_counter()
    state = _validate_state(state_value)
    serving = _validate_component(state["serving"], "serving")
    payload = json.loads(Path(str(serving["result_path"])).read_text())
    return _write_serving_evaluation_report_payload(output_path, state_value, payload,
        started=started, llmservingsim_results=llmservingsim_results,
        refresh_latency_plots=refresh_latency_plots)


def _write_serving_evaluation_report_payload(
    output_path: str | Path,
    state_value: Mapping[str, object],
    serving_payload: Mapping[str, object],
    *,
    started: float,
    llmservingsim_results: Path | None = None,
    refresh_latency_plots: bool = False,
) -> float:
    """Validate and write the standalone full-only Serving report."""

    output = Path(output_path).resolve()
    state = _validate_state(state_value)
    serving = _validate_component(state["serving"], "serving")
    serving_payload = _mapping(serving_payload, "Serving result")
    cells = _sequence(serving_payload["cells"], "Serving cells")
    if not cells:
        raise ValueError("standard report requires at least one Serving cell")
    baseline_statuses = None
    if llmservingsim_results is not None:
        from evaluation.reporting.llmservingsim_report import match_standard_baselines
        baseline_statuses = match_standard_baselines(llmservingsim_results.resolve(), serving_payload)
        coverage_path = output.parent / "llmservingsim_coverage.json"
        coverage_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_path.write_text(json.dumps({"source": str(llmservingsim_results.resolve()),
                                             "cases": baseline_statuses}, indent=2, ensure_ascii=False) + "\n")
    if llmservingsim_results is not None or refresh_latency_plots:
        from evaluation.serving.metrics.sharegpt_evaluation_report import write_sharegpt_request_latency_plots
        paired_cells = [cell for cell in cells if cell["prediction_status"] == "completed"]
        if paired_cells:
            write_sharegpt_request_latency_plots(Path(str(serving["result_path"])).parent / "plots",
                                                paired_cells, baseline_by_cell=baseline_statuses)
    lines = _render_serving_markdown(
        output, state, serving, cells, baseline_statuses=baseline_statuses,
        llmservingsim_results=llmservingsim_results,
        report_assembly_wall_time_s=0.0,
    )
    assembly_wall = time.perf_counter() - started
    lines = _render_serving_markdown(
        output, state, serving, cells, baseline_statuses=baseline_statuses,
        llmservingsim_results=llmservingsim_results,
        report_assembly_wall_time_s=assembly_wall,
    )
    _write_markdown_atomic(output, lines)
    return time.perf_counter() - started


def _validate_state(value: object) -> Mapping[str, object]:
    state = _mapping(value, "standard report state")
    if set(state) != {
        "schema_version", "source_commit", "generated_at", "update_mode",
        "kernel", "serving",
    }:
        raise ValueError("standard report state fields are invalid")
    if state["schema_version"] != 1:
        raise ValueError("standard report state schema_version must equal 1")
    return state


def _write_markdown_atomic(output: Path, lines: Sequence[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)


def _validate_component(value: object, label: str) -> Mapping[str, object]:
    component = _mapping(value, f"{label} component")
    required = {
        "generated_at", "source_commit", "result_path",
        "evaluation_wall_time_s", "plot_wall_time_s",
    }
    if label == "kernel":
        required.update((
            "hardware_evaluation_wall_time_s",
            "traffic_plot_path",
            "traffic_summary_path",
            "traffic_plot_wall_time_s",
        ))
    elif label == "serving":
        pass
    else:
        raise ValueError(f"unknown standard report component: {label}")
    if not required <= set(component):
        raise ValueError(f"{label} component fields are incomplete")
    for field in ("generated_at", "source_commit", "result_path"):
        if not isinstance(component[field], str) or not component[field]:
            raise TypeError(f"{label} {field} must be non-empty text")
    result_path = Path(str(component["result_path"]))
    if not result_path.is_file():
        raise ValueError(f"{label} result does not exist: {result_path}")
    for field in ("evaluation_wall_time_s", "plot_wall_time_s"):
        value = _finite(component[field], f"{label} {field}")
        if value < 0.0:
            raise ValueError(f"{label} {field} must be nonnegative")
    if label == "kernel":
        value = _finite(
            component["traffic_plot_wall_time_s"],
            "kernel traffic plot wall time",
        )
        if value < 0.0:
            raise ValueError("kernel traffic plot wall time must be nonnegative")
    return component


def _kernel_wall_time_table(kernel: Mapping[str, object]) -> list[str]:
    timings = _mapping(
        kernel["hardware_evaluation_wall_time_s"],
        "Kernel hardware evaluation wall times",
    )
    lines = [
        "| Hardware | Dataset evaluation wall time |",
        "|---|---:|",
    ]
    for hardware_id in KERNEL_HARDWARE_ORDER:
        if hardware_id not in timings:
            continue
        lines.append(
            f"| {HARDWARE_LABELS[hardware_id]} | "
            f"{_finite(timings[hardware_id], f'{hardware_id} Kernel wall time'):.3f} s |"
        )
    return lines


def _kernel_traffic_table(payload: Mapping[str, object]) -> list[str]:
    hardware_entries = {}
    for entry in _sequence(payload["hardware"], "traffic hardware"):
        hardware = _mapping(entry, "traffic hardware entry")
        hardware_id = str(hardware["hardware_id"])
        if hardware_id in hardware_entries:
            raise ValueError(f"duplicate traffic hardware entry: {hardware_id}")
        hardware_entries[hardware_id] = hardware
    if set(hardware_entries) != set(HARDWARE_ORDER):
        raise ValueError("traffic summary hardware set is invalid")
    lines = [
        "| Hardware | L2 cases | L2 MAPE | L2 MedAPE | L2 mean P/M | "
        "HBM cases | HBM MAPE | HBM MedAPE | HBM mean P/M |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware_id in HARDWARE_ORDER:
        metrics = _mapping(
            hardware_entries[hardware_id]["metrics"],
            f"{hardware_id} traffic metrics",
        )
        cells = []
        for metric in ("l2", "hbm"):
            summary = _mapping(metrics[metric], f"{hardware_id} {metric} traffic")
            case_count = summary["case_count"]
            if isinstance(case_count, bool) or not isinstance(case_count, int):
                raise TypeError(f"{hardware_id} {metric} case count must be an integer")
            cells.append(str(case_count))
            cells.append(
                f"{_finite(summary['mape_percent'], f'{hardware_id} {metric} MAPE'):.2f}%")
            cells.append(
                f"{_finite(summary['median_ape_percent'], f'{hardware_id} {metric} MedAPE'):.2f}%")
            cells.append(
                f"{_finite(summary['mean_prediction_over_measured'], f'{hardware_id} {metric} mean P/M'):.3f}×")
        lines.append(f"| {HARDWARE_LABELS[hardware_id]} | {' | '.join(cells)} |")
    return lines


def _kernel_traffic_exclusion_note(payload: Mapping[str, object]) -> list[str]:
    parts = []
    for entry in _sequence(payload["hardware"], "traffic hardware"):
        hardware = _mapping(entry, "traffic hardware entry")
        count = len(_sequence(
            hardware["fa_serving_capture_case_ids"],
            "FA serving capture case ids",
        ))
        if count:
            parts.append(f"{HARDWARE_LABELS[str(hardware['hardware_id'])]} {count} cases")
    if not parts:
        return []
    return [
        "FA serving-capture rows have only op-level evidence and no per-kernel traffic; excluded from this plot ("
        + ", ".join(parts)
        + "); details of cases filtered during metric aggregation are in the structured results.",
        "",
    ]


def _render_kernel_markdown(
    output: Path,
    state: Mapping[str, object],
    kernel: Mapping[str, object],
    kernel_payload: Mapping[str, object],
    traffic_payload: Mapping[str, object],
    *,
    report_assembly_wall_time_s: float,
    with_html: bool = False,
) -> list[str]:
    header = [
        "# Midlayer Single-kernel Forward standard evaluation report",
        "",
    ]
    if with_html:
        header += [
            "Offline Kernel overview: [HTML](" + str(output.with_suffix(".html")) + ")",
            "",
        ]
    return [
        *header,
        f"Generated at: `{state['generated_at']}`; Kernel results at: `{kernel['generated_at']}`; source commit: `{kernel['source_commit']}`.",
        "",
        "## Kernel dataset run time",
        "",
        "| Evaluation wall time | Plot wall time | Traffic plot wall time | Markdown assembly wall time |",
        "|---:|---:|---:|---:|",
        f"| {_finite(kernel['evaluation_wall_time_s'], 'kernel evaluation wall'):.3f} s | {_finite(kernel['plot_wall_time_s'], 'kernel plot wall'):.3f} s | {_finite(kernel['traffic_plot_wall_time_s'], 'kernel traffic plot wall'):.3f} s | {report_assembly_wall_time_s:.3f} s |",
        "",
        *_kernel_wall_time_table(kernel),
        "",
        "## Single-kernel Forward accuracy",
        "",
        *_kernel_table(kernel_payload),
        "",
        *_kernel_unsupported_note(kernel_payload),
        f"![All Kernel buckets by hardware]({_relative(output, kernel['overall_plot_path'])})",
        "",
        f"![Per-bucket scatter and OP-level APE CDF by hardware]({_relative(output, kernel['detailed_plot_path'])})",
        "",
        "The CDF plots cumulative OP fraction on the x-axis and absolute percentage error on the y-axis, avoiding the visual compression of relative error in log-log latency scatter plots.",
        "",
        "## Memory traffic Forward accuracy",
        "",
        "Each point is one dataset case's L2 or HBM traffic, summed over kernels that have both predicted and measured bytes. Measured L2 uses source-interface bytes and HBM uses downstream global-memory bytes, consistent with mechanism validation. Predictions use Forward `l2_cache` / `dram` resource demand. This comparison is an evaluation diagnostic only and never feeds back into Forward parameters.",
        "",
        *_kernel_traffic_table(traffic_payload),
        "",
        *_kernel_traffic_exclusion_note(traffic_payload),
        *_kernel_launch_only_traffic_note(kernel_payload),
        *(
            [f"![Predicted/measured L2 and HBM traffic on three hardware platforms]({_relative(output, kernel['traffic_plot_path'])})",
             ""]
            if Path(str(kernel["traffic_plot_path"])).is_file() else
            ["This report-only rebuild uses a semantic artifact without per-kernel L2/HBM counter "
             "mechanisms. A traffic plot requires an original capture run with `capture_mechanisms=True`; no plot is generated here.",
             ""]
        ),
        "## Reviewable artifacts",
        "",
        f"- Structured Kernel results: `{kernel['result_path']}`",
        f"- Structured memory traffic results: `{kernel['traffic_summary_path']}`",
        f"- Report state: `{output.parent / 'state.json'}`",
        "",
    ]


def _serving_scenario_note(cells):
    scenarios = set()
    scopes = set()
    for cell in cells:
        if "effective_prediction_contract" not in cell:
            continue
        contract = cell["effective_prediction_contract"]
        if "h200_serving_scenario" not in contract:
            continue
        scenario = contract["h200_serving_scenario"]
        if scenario != "analytical_bf16_v1":
            raise ValueError("unknown saved H200 Serving scenario: " + str(scenario))
        if cell["hardware_id"] != "nvidia_h200":
            raise ValueError("H200 Serving scenario requires H200 case hardware")
        scenarios.add(scenario)
        scopes.add(contract["scenario_scope"])
    if not scenarios:
        return []
    return [
        "> **H200 uses the physical analytical reference model `analytical_bf16_v1`.** Original capture and effective simulation contracts are retained separately; native provider/launch equivalence remains unaccepted.",
        "",
        "Requests, Graph, TP/EP, and Scheduler parallel settings follow the selected case. Launch, Norm declarations, BF16 GEMM/expert reference programs, TP2 communication, and Host profiles follow the explicit scenario. Unknown TP2 token IDs/graph tails use declared synthetic-input assumptions; embedding locality is approximate.",
        "",
        *("Effective scenario scope: " + scope for scope in sorted(scopes)),
        "",
    ]


def _render_serving_markdown(
    output: Path,
    state: Mapping[str, object],
    serving: Mapping[str, object],
    cells: Sequence[object],
    *,
    report_assembly_wall_time_s: float,
    baseline_statuses: Mapping[str, object] | None = None,
    llmservingsim_results: Path | None = None,
) -> list[str]:
    normalized_cells = tuple(_mapping(cell, "Serving cell") for cell in cells)
    hardware_present = {str(cell["hardware_id"]) for cell in normalized_cells}
    unknown_hardware = hardware_present - set(SERVING_HARDWARE_ORDER)
    if unknown_hardware:
        raise ValueError(f"unknown Serving hardware: {sorted(unknown_hardware)}")
    serving_root = Path(str(serving["result_path"])).parent
    timing_modes = sorted({str(cell["serving_timing_mode"]) for cell in normalized_cells
        if cell["prediction_status"] == "completed" and "serving_timing_mode" in cell})
    if not timing_modes:
        timing_modes = [str(serving["serving_timing_mode"]) if "serving_timing_mode" in serving
            else "device_plus_launch_plus_scheduler"]
    timing_description = ", ".join(f"`{mode}`" for mode in timing_modes)
    lines = [
        "# Midlayer Full Serving standard evaluation report",
        "",
        f"Generated at: `{state['generated_at']}`; Serving results at: `{serving['generated_at']}`; source commit: `{serving['source_commit']}`.",
        "",
        f"Serving timing mode: {timing_description}. Main latency plots pair measured and predicted values by request index; sorted plots compare overall distributions. Report wall time is actual Midlayer execution time, rather than simulated request duration.",
        "",
        *_serving_scenario_note(normalized_cells),
        "## Run time",
        "",
        "| Evaluation wall time | Plot wall time | Markdown assembly wall time |",
        "|---:|---:|---:|",
        f"| {_finite(serving['evaluation_wall_time_s'], 'Serving evaluation wall'):.3f} s | {_finite(serving['plot_wall_time_s'], 'Serving plot wall'):.3f} s | {report_assembly_wall_time_s:.3f} s |",
        "",
        "Independent cases run concurrently; accumulated case/phase CPU wall times below cannot be directly summed and compared with parent-process elapsed time.",
        "",
        "## Full Serving overview",
        "",
        *_serving_summary_table(normalized_cells),
        "",
        "Scored / completed is the actual accuracy denominator: all completed cases are included, with equal weighting of per-case, per-request paired MAPE. Capture differences above 5% are diagnostic labels only; they neither block runs nor exclude cases. Historical capture-validation conclusions, ground truth, and original passed flags are retained. Dev-only counts completed cases with `accuracy_claim_status=unsupported_development_only`: collective timing uses the production ProtoBridge analytical model, but communication profiles for all three hardware platforms explicitly contain non-production development parameters (semantic mapping gaps: `profiler_output/collective/protobridge_mapping_gaps_20260827.json`), limiting accuracy interpretation. Failed counts execution failures; Unsupported counts explicitly unsupported cases. Neither enters accuracy metrics.",
        "",
        f"![Full Serving overview on three hardware platforms]({_relative(output, serving_root / 'plots/three_hardware_serving_overview.png')})",
        "",
        "## Actual Midlayer wall-time breakdown",
        "",
        *_wall_time_table(normalized_cells),
        "",
    ]
    links = []
    for cell in normalized_cells:
        if "interactive_report" in cell:
            receipt = cell["interactive_report"]
            if "path" in receipt:
                links.append("| " + str(cell["cell_id"]) + " | [HTML](" + _relative(output, Path(str(receipt["path"]))) + ") |")
    if links:
        lines.extend(["## Offline interactive reports", "", "| Case | Report |", "|---|---|", *links, ""])
    if baseline_statuses is not None:
        available = sum(entry["status"] == "available" for entry in baseline_statuses.values())
        lines.extend((
            "## LLMServingSim 2.0 comparison definitions", "",
            f"Baseline status is listed for all {len(normalized_cells)} standard-matrix cases. Matching, validated LLMServingSim 2.0 results exist for {available} cases and are overlaid on TTFT/TPOT paired scatter and sorted distribution plots below. Observed and Midlayer data use this standard run. LLMServingSim runs CPU simulation with existing compatible LUTs or reuses accepted results, verifying canonical requests, runtime, and observed metrics individually.", "",
            "LLMServingSim compute uses measured LUTs, retaining single-layer dummy-weight/eager capture, interpolation/extrapolation, and software-execution differences. Analytical describes the network backend, rather than purely analytical Transformer compute. Request/KV budgets are aligned, while runtimes are not fully equivalent. Cases without scheduling trajectories or host/Wait/Processing decomposition have no corresponding curves.", "",
            f"[Per-case status, original result locations, and comparison limitations]({_relative(output, output.parent / 'llmservingsim_coverage.json')})", "",
            f"Baseline LUT sources: [Historical comparison report]({_relative(output, llmservingsim_results / 'report.md')}), [Independent acceptance commit and scope]({_relative(output, llmservingsim_results / 'main-acceptance.json')}). Midlayer source commit and simulation time are at the top of this report. Native run manifests are in coverage JSON. Newly added QPS cases neither download weights nor recapture LUTs.", "",
        ))
    for hardware_id in SERVING_HARDWARE_ORDER:
        hardware_cells = sorted(
            [cell for cell in normalized_cells if cell["hardware_id"] == hardware_id],
            key=_case_sort_key,
        )
        if not hardware_cells:
            continue
        label = HARDWARE_LABELS[hardware_id]
        lines.extend((f"## {label} Serving", ""))
        unavailable_splits = [cell for cell in hardware_cells
            if cell["prediction_status"] == "completed"
            and "ttft_breakdown_unavailable_reason" in cell
            and cell["metrics"]["ttft_breakdown"] is None]
        if unavailable_splits:
            lines.extend((
                "TTFT decomposition unavailable: cases below lack observed boundary evidence. Decomposition plots show unavailable status only; total TTFT, per-request paired errors, and accuracy denominators remain valid.",
                "", "| Case | Missing evidence |", "|---|---|",
                *(f"| `{cell['model_id']} / {cell['case_id']}` | "
                  f"{cell['ttft_breakdown_unavailable_reason']} |" for cell in unavailable_splits),
                "",
            ))
        if hardware_id == "nvidia_h200":
            source = (
                "canonical `serving_dataset/sharegpt/nvidia_h200`"
                if serving["nvidia_h200_dataset_root"] is None
                else "Non-canonical candidate dataset "
                     f"`{serving['nvidia_h200_dataset_root']}`")
            lines.extend((
                f"H200 ground truth is a vLLM `{serving['nvidia_h200_vllm_version']}` "
                f"ShareGPT capture ({source}); Host timing uses this vLLM version's "
                "`nvidia_h200_vllm*` profile. Items not measured on that version are marked "
                "`substitution` in the references cited by the profile.",
                "",
            ))
        lines.extend((
            "### TTFT: per-request pairing and sorted distributions",
            "",
            f"![{label} TTFT paired scatter and sorted distribution]({_relative(output, serving_root / 'plots' / f'{hardware_id}_ttft_request_comparison.png')})",
            "",
            "Plot guide: the left panel pairs each case strictly by request index; x is measured TTFT and y is predicted TTFT for the same request. The diagonal is exact agreement. The right panel sorts measured and predicted values independently into ascending quantile curves, without request pairing. All axes start at 0. Left panels expose per-request bias, outliers, and systematic over/underprediction; right panels compare overall TTFT distributions and tails.",
            "",
            "### TTFT view 1: measured / predicted total decomposition",
            "",
            f"![{label} TTFT wait and processing breakdown]({_relative(output, serving_root / 'plots' / f'{hardware_id}_ttft_breakdown.png')})",
            "",
            "Plot guide: requests are sorted by ascending measured TTFT for each case. Both measured (left) and predicted (right) use that order, so equal x coordinates identify the same request. The x-axis is neither chronological nor a Wait quantile. Dark regions are Wait, light regions are Processing, and the outer stack boundary is total TTFT. Only measured total TTFT is necessarily monotonic; changing Wait or predicted boundaries are not time-series spikes. The right panel also labels mean signed ΔWait, ΔProcessing, and ΔTTFT.",
            "Use: with independent clock-alignment evidence, retain per-request pairing to determine whether Wait or Processing dominates a request's TTFT error. Wait spans the explicitly bound request start to first scheduling and includes ingress and queuing, rather than only Scheduler-internal queuing. Processing spans first scheduling to the client's first token, including Host, launch, Device, and return path, rather than only Device time. Client TTFT start, scheduled arrival, and the first Scheduler-step origin cannot be subtracted without alignment. Missing evidence produces no decomposition values.",
            "",
            "### TTFT view 2: processing / queuing components",
            "",
            f"![{label} TTFT component-first comparison]({_relative(output, serving_root / 'plots' / f'{hardware_id}_ttft_component_comparison.png')})",
            "",
            "Plot guide: this component-first view shows Processing on the left and Wait on the right. Blue is measured, orange is predicted, solid lines are Processing, and dashed lines are Wait. Measured and predicted values for each component are independently sorted ascending; equal x coordinates represent equal quantiles, without guaranteeing the same request.",
            "Use: compare component distributions, quantile errors, and tails, rather than a specific request's error. Per-request comparisons use the paired scatter or view 1 above.",
            "",
            "### TPOT: per-request pairing and sorted distributions",
            "",
            f"![{label} TPOT paired scatter and sorted distribution]({_relative(output, serving_root / 'plots' / f'{hardware_id}_tpot_request_comparison.png')})",
            "",
            "### Measured vs predicted scheduling progress",
            "",
        ))
        progress_plot = serving_root / "plots" / f"{hardware_id}_scheduler_progress.png"
        if (all(cell["prediction_status"] == "completed" for cell in hardware_cells)
                and has_scheduler_progress_evidence(hardware_cells) and progress_plot.is_file()):
            lines.extend((
                f"![{label} merged scheduler progress]({_relative(output, progress_plot)})",
                "",
            ))
        elif any(cell["prediction_status"] != "completed" for cell in hardware_cells):
            lines.extend((
                "Some cases failed or are explicitly unsupported, so no combined scheduling-progress plot is generated for this hardware. See the per-case table for status and reasons.",
                "",
            ))
        else:
            lines.extend((
                "Complete observed Scheduler steps or predicted progress evidence is missing; no combined scheduling-progress plot is generated. Gaps are never filled with zero values or synthetic trajectories.",
                "",
            ))
        lines.extend((
            "### Per-case signed error",
            "",
            *_serving_case_table(hardware_cells),
            "",
            "Status=Dev-only has the same meaning as in the overview. Capture Δ lists differences between capture and prediction runtime contracts. The permanent `source_commit` difference (prediction pins the Scheduler source commit; capture records the full deployment commit chain) is excluded by design.",
            "",
        ))
        if baseline_statuses is not None:
            from evaluation.reporting.llmservingsim_report import baseline_status_table
            lines.extend((*baseline_status_table(hardware_cells, baseline_statuses), ""))
    lines.extend((
        "## Host timing overhead share on three hardware platforms",
        "",
        f"![Host timing overhead share on three hardware platforms]({_relative(output, serving_root / 'plots/three_hardware_host_timing_overhead_percent.png')})",
        "",
        "## Reviewable artifacts",
        "",
        f"- Structured Serving results: `{serving['result_path']}`",
        f"- Report state: `{output.parent / 'state.json'}`",
        "",
        "## Serving case run time",
        "",
        f"![Serving case evaluation wall time on three hardware platforms]({_relative(output, serving_root / 'plots/serving_case_evaluation_wall_time.png')})",
        "",
        "Each bar is the case worker's wall time in this concurrent batch. Comparisons identify tails, but this is not a single-case benchmark on a dedicated CPU.",
        "",
        "## Serving case peak memory",
        "",
        f"![Serving case peak RSS on three hardware platforms]({_relative(output, serving_root / 'plots/serving_case_peak_rss.png')})",
        "",
        "Each bar is the peak RSS of an isolated case subprocess. The y-axis uses a logarithmic GiB scale to compare models and request arrival rates.",
        "",
    ))
    if "workload_path" in serving and serving["workload_path"] == "virtual_tensor":
        lines[4:4] = ["Per-request metrics from the virtual source route are paired. Source/provider equivalence and normal capacity remain unaccepted (Source-unvalidated); a recorded commit alone does not prove the executed source snapshot was frozen. Observed Scheduler, Host decomposition, and CPU wall time use only evidence present in this run and canonical data.", ""]
        rendered = []
        for line in lines:
            if line.startswith("![") and "](" in line and line.endswith(")"):
                label, relative = line[2:-1].split("](", 1)
                if not (output.parent / relative).is_file():
                    reason = ("selected timing mode omits full Host overhead or no completed Host results exist"
                        if "host_timing" in relative else "no completed case has the pairing/scheduling evidence required by this plot")
                    rendered.append(f"{label}: not generated ({reason}).")
                    continue
            rendered.append(line)
        return rendered
    return lines


def _kernel_card_entries(
        payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    matrix = _mapping(payload["kernel_matrix"], "kernel matrix")
    entries = {}
    for hardware_id in KERNEL_HARDWARE_ORDER:
        if hardware_id not in matrix:
            if hardware_id in REQUIRED_KERNEL_HARDWARE:
                raise ValueError(f"{hardware_id} kernel result is missing")
            continue
        entry = _mapping(matrix[hardware_id], f"{hardware_id} kernel result")
        if (hardware_id in REQUIRED_KERNEL_HARDWARE
                and entry["status"] != "PASS"):
            raise ValueError(f"{hardware_id} kernel evaluation is not PASS")
        entries[hardware_id] = entry
    return entries


def _forward_scored_counts(
        entry: Mapping[str, object], label: str) -> dict[int, int]:
    if "forward_coverage" not in entry or entry["forward_coverage"] is None:
        return {}
    coverage = _mapping(entry["forward_coverage"], f"{label} Forward coverage")
    scored = {}
    for item in _sequence(coverage["scored"], f"{label} scored rows"):
        row_index, count, total = _sequence(item, f"{label} scored row")
        scored[_integer(row_index, f"{label} scored row index")] = _integer(
            count, f"{label} scored count")
    return scored


def _kernel_table(payload: Mapping[str, object]) -> list[str]:
    entries = _kernel_card_entries(payload)
    normalized = {}
    ordered_keys = []
    labels = {}
    for hardware_id, entry in entries.items():
        if entry["status"] != "PASS":
            continue
        scored_counts = _forward_scored_counts(entry, hardware_id)
        rows = _sequence(entry["rows"], f"{hardware_id} kernel rows")
        by_key = {}
        parent = "ALL"
        for row_index, row_value in enumerate(rows):
            row = _sequence(row_value, f"{hardware_id} kernel row")
            if len(row) != 6:
                raise ValueError("kernel matrix row must contain six fields")
            label = str(row[0])
            clean_label = _kernel_row_key(label)
            if label == "ALL" or not label.startswith("│"):
                parent = clean_label
                key = clean_label
            else:
                key = f"{parent}/{clean_label}"
            count = _integer(row[1], f"{hardware_id} {key} count")
            count_text = (
                f"{scored_counts[row_index]}/{count}"
                if row_index in scored_counts else str(count))
            if row[3] is None:
                by_key[key] = (count_text, None)
            else:
                forward = _sequence(
                    row[3], f"{hardware_id} {key} Forward metrics")
                if len(forward) != 3:
                    raise ValueError(
                        "Forward metrics must contain MAPE/MedAPE/WMAE")
                by_key[key] = (
                    count_text,
                    *(
                        _finite(value, f"{hardware_id} {key} metric")
                        for value in forward
                    ),
                )
            labels[key] = label
            if key not in ordered_keys:
                ordered_keys.append(key)
        normalized[hardware_id] = by_key
    if not ordered_keys:
        raise ValueError("kernel matrix is empty")
    header = "| Bucket |"
    for hardware_id in entries:
        short = _KERNEL_SHORT_LABELS[hardware_id]
        header += f" {short} N | {short} MAPE | {short} MedAPE | {short} WMAE |"
    lines = [header, "|---|" + "---:|" * (4 * len(entries))]
    for key in ordered_keys:
        values = []
        for hardware_id, entry in entries.items():
            if entry["status"] != "PASS":
                tag = str(entry["status"]).lower()
                values.extend((tag, tag, tag, tag))
                continue
            if key not in normalized[hardware_id]:
                values.extend(("—", "—", "—", "—"))
                continue
            metrics = normalized[hardware_id][key]
            count = metrics[0]
            if metrics[1] is None:
                values.extend((count, "Err", "Err", "Err"))
                continue
            _, mape, medape, wmae = metrics
            values.extend((
                count, f"{mape:.2f}%", f"{medape:.2f}%", f"{wmae:.2f}%"
            ))
        lines.append(f"| {labels[key]} | {' | '.join(values)} |")
    return lines


def _kernel_unsupported_note(payload: Mapping[str, object]) -> list[str]:
    lines = []
    for hardware_id, entry in _kernel_card_entries(payload).items():
        if entry["status"] != "PASS":
            lines.append(
                f"{HARDWARE_LABELS[hardware_id]} Kernel evaluation "
                f"{entry['status']}: {entry['summary']}.")
            continue
        if "forward_coverage" not in entry or entry["forward_coverage"] is None:
            continue
        coverage = _mapping(entry["forward_coverage"], hardware_id)
        unsupported = _mapping(
            coverage["unsupported_cases"], f"{hardware_id} unsupported cases")
        if not unsupported:
            continue
        parts = ", ".join(
            f"{op_type} {count}" for op_type, count in unsupported.items()
            if op_type != "ALL")
        lines.append(
            f"{HARDWARE_LABELS[hardware_id]}: explicitly unsupported Forward cases "
            f"total {unsupported['ALL']} ({parts}), excluded from MAPE; the N column's "
            "`scored/N` gives the number of scored cases in that bucket. See this hardware's "
            "evaluation log, “Forward unsupported cases” section, for reasons.")
    if not lines:
        return []
    return [*lines, ""]


def _kernel_launch_only_traffic_note(
        payload: Mapping[str, object]) -> list[str]:
    present = [
        HARDWARE_LABELS[hardware_id]
        for hardware_id, entry in _kernel_card_entries(payload).items()
        if hardware_id in _LAUNCH_ONLY_KERNEL_HARDWARE
        and entry["status"] == "PASS"
    ]
    if not present:
        return []
    return [
        ", ".join(present)
        + " kernel records use NSYS launch facts; some records include "
        "Nsight Compute counter sidecars. Current standard traffic evaluation "
        "does not consume these sidecars yet, so those records are excluded from this table and plot.",
        "",
    ]


def _serving_prediction_status(cell: Mapping[str, object]) -> str:
    status = cell["prediction_status"]
    if status not in ("completed", "failed", "unsupported", "unsupported_launch_coverage"):
        raise ValueError(f"unknown Serving prediction status: {status!r}")
    return status


def _serving_summary_table(cells: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Hardware | Completed / total | Scored / completed | Observer Δ>5% | Dev-only | Failed | Unsupported | Req/s MAPE | TTFT paired MAPE | TPOT paired MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware_id in SERVING_HARDWARE_ORDER:
        selected = [cell for cell in cells if cell["hardware_id"] == hardware_id]
        if not selected:
            continue
        statuses = [_serving_prediction_status(cell) for cell in selected]
        completed = [cell for cell in selected if cell["prediction_status"] == "completed"]
        scored = completed
        provisional = sum(bool(_observer_delta_metrics(cell)) for cell in completed)
        dev_only = sum(
            1
            for cell in completed
            if cell["accuracy_claim_status"] == "unsupported_development_only"
        )
        failed = statuses.count("failed")
        unsupported = sum(status in ("unsupported", "unsupported_launch_coverage") for status in statuses)
        if not completed:
            lines.append(
                f"| {HARDWARE_LABELS[hardware_id]} | 0 / {len(selected)} | "
                f"0 / 0 | 0 | 0 | {failed} | {unsupported} | Err | Err | Err |"
            )
            continue
        throughput_errors = []
        latency_errors = {"ttft_ms": [], "tpot_ms": []}
        for cell in scored:
            metrics = _mapping(cell["metrics"], "Serving metrics")
            throughput = _mapping(metrics["throughput"], "Serving throughput")
            request_rate = _mapping(
                throughput["request_throughput_per_s"], "request throughput"
            )
            throughput_errors.append(_finite(
                request_rate["absolute_percentage_error"], "request throughput APE"
            ))
            paired = _mapping(metrics["paired_request_error"], "paired errors")
            for metric_name in latency_errors:
                latency_errors[metric_name].append(_finite(
                    _mapping(paired[metric_name], metric_name)["mape_percent"],
                    f"{metric_name} MAPE",
                ))
        lines.append(
            f"| {HARDWARE_LABELS[hardware_id]} | {len(completed)} / "
            f"{len(selected)} | {len(scored)} / {len(completed)} | "
            f"{provisional} | {dev_only} | {failed} | {unsupported} | "
            f"{fmean(throughput_errors):.2f}% | "
            f"{fmean(latency_errors['ttft_ms']):.2f}% | "
            f"{fmean(latency_errors['tpot_ms']):.2f}% |"
        )
    return lines


def _wall_time_table(cells: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Hardware | Cases | Simulation | Workload | Lowering | Kernel prediction | Timeline | Artifact | Scheduler/orchestration | Validation/comparison |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hardware_id in SERVING_HARDWARE_ORDER:
        selected = [
            cell for cell in cells
            if cell["hardware_id"] == hardware_id
            and cell["prediction_status"] == "completed"
        ]
        if not selected:
            continue
        totals = {phase: 0.0 for phase in MIDLAYER_WALL_TIME_PHASES}
        simulation_total = 0.0
        validation_total = 0.0
        for cell in selected:
            provenance = _mapping(cell["forward_provenance"], "forward provenance")
            timing = _mapping(provenance["midlayer_wall_time"], "Midlayer wall time")
            simulation_wall = _finite(
                timing["simulation_wall_time_s"], "simulation wall time"
            )
            phases = _mapping(timing["phase_wall_time_s"], "Midlayer phases")
            simulation_total += simulation_wall
            for phase in MIDLAYER_WALL_TIME_PHASES:
                totals[phase] += _finite(phases[phase], phase)
            validation_total += max(
                0.0,
                _finite(cell["evaluation_wall_time_seconds"], "case wall time")
                - simulation_wall,
            )
        lines.append(
            f"| {HARDWARE_LABELS[hardware_id]} | {len(selected)} | {simulation_total:.3f}s | "
            f"{totals['workload_construction']:.3f}s | {totals['kernel_lowering']:.3f}s | "
            f"{totals['kernel_prediction']:.3f}s | {totals['timeline_construction']:.3f}s | "
            f"{totals['artifact_generation']:.3f}s | {totals['scheduler_and_orchestration']:.3f}s | "
            f"{validation_total:.3f}s |"
        )
    return lines


_ACCURACY_CLAIM_LABELS = {
    "source_projected_unvalidated": "Source-unvalidated",
    "supported_for_accuracy_interpretation": "OK",
    "unsupported_development_only": "Dev-only",
    "debug_capacity_not_for_accuracy": "Debug-capacity",
    "provisional_observer_effect": "Observer diagnostic",
}
_CAPTURE_DELTA_OMITTED_FIELDS = frozenset({"source_commit"})


def _observer_delta_metrics(cell: Mapping[str, object]) -> tuple[str, ...] | None:
    if "observation_quality" not in cell:
        return None
    quality = _mapping(cell["observation_quality"], "Serving observation quality")
    if "metric_delta_percent" not in quality or quality["metric_delta_percent"] is None:
        return None
    deltas = _mapping(quality["metric_delta_percent"], "observer effect deltas")
    return tuple(sorted(name for name, value in deltas.items()
                        if _finite(value, f"observer delta {name}") > 5.0))


def _capture_delta(cell: Mapping[str, object]) -> str:
    comparison = _mapping(
        cell["capture_runtime_comparison"], "capture runtime comparison"
    )
    fields = _mapping(comparison["fields"], "capture runtime fields")
    mismatches = [
        field
        for field, entry in fields.items()
        if field not in _CAPTURE_DELTA_OMITTED_FIELDS
        and _mapping(entry, f"capture field {field}")["status"] == "mismatch"
    ]
    return ", ".join(mismatches) if mismatches else "—"


def _serving_case_table(cells: Sequence[Mapping[str, object]]) -> list[str]:
    lines = [
        "| Model / case | Status | Observer Δ (5% label) | Req/s pred/measured | TTFT signed error | TPOT signed error | TTFT paired MAPE | TPOT paired MAPE | Capture Δ |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        prediction_status = _serving_prediction_status(cell)
        if prediction_status != "completed":
            reason = str(cell["prediction_reason"])
            label = "Failed" if prediction_status == "failed" else "Unsupported"
            lines.append(
                f"| `{cell['model_id']} / {cell['case_id']}` | "
                f"{label}: {reason} | — | Err | Err | Err | Err | Err | — |"
            )
            continue
        claim = str(cell["accuracy_claim_status"])
        if claim not in _ACCURACY_CLAIM_LABELS:
            raise ValueError(f"unknown accuracy claim status: {claim!r}")
        status = _ACCURACY_CLAIM_LABELS[claim]
        observer_metrics = _observer_delta_metrics(cell)
        quality_text = (
            "legacy" if observer_metrics is None
            else "Δ≤5%" if not observer_metrics
            else "Δ>5%: " + ", ".join(observer_metrics)
        )
        capture_delta = _capture_delta(cell)
        metrics = _mapping(cell["metrics"], "Serving metrics")
        throughput = _mapping(
            _mapping(metrics["throughput"], "throughput")["request_throughput_per_s"],
            "request throughput",
        )
        latency = _mapping(metrics["latency"], "latency")
        paired = _mapping(metrics["paired_request_error"], "paired error")
        req_ratio = _finite(throughput["predicted"], "predicted req/s") / _finite(
            throughput["observed"], "measured req/s"
        )
        values = []
        for metric_name in ("ttft_ms", "tpot_ms"):
            mean = _mapping(
                _mapping(latency[metric_name], metric_name)["mean"],
                f"{metric_name} mean",
            )
            observed = _finite(mean["observed"], f"{metric_name} observed")
            predicted = _finite(mean["predicted"], f"{metric_name} predicted")
            values.append((predicted - observed) / observed * 100.0)
        ttft_mape = _finite(
            _mapping(paired["ttft_ms"], "TTFT paired error")["mape_percent"],
            "TTFT MAPE",
        )
        tpot_mape = _finite(
            _mapping(paired["tpot_ms"], "TPOT paired error")["mape_percent"],
            "TPOT MAPE",
        )
        lines.append(
            f"| `{cell['model_id']} / {cell['case_id']}` | {status} | {quality_text} | "
            f"{req_ratio:.3f}× | "
            f"{values[0]:+.2f}% | {values[1]:+.2f}% | "
            f"{ttft_mape:.2f}% | {tpot_mape:.2f}% | {capture_delta} |"
        )
    return lines


def _relative(output: Path, target: object) -> str:
    return Path(os.path.relpath(Path(str(target)).resolve(), output.parent)).as_posix()


def _kernel_row_key(label: str) -> str:
    return (
        label.replace("├", "").replace("└", "").replace("─", "")
        .replace("│", "").strip()
    )


def _case_sort_key(cell: Mapping[str, object]) -> tuple[int, float, str]:
    model_id = str(cell["model_id"])
    if model_id not in MODEL_RANK:
        raise ValueError(f"unknown standard-report model: {model_id}")
    return (
        MODEL_RANK[model_id],
        _finite(cell["target_qps"], "target QPS"),
        str(cell["case_id"]),
    )


def _plot_slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_").lower()








def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
