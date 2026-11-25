"""Render an optional, explicitly scoped external-baseline TTFT comparison."""

from __future__ import annotations

import json
import math
import os
import shlex
from pathlib import Path

from evaluation.serving.metrics.llmservingsim_comparison import distribution, finite_nonnegative


MODELS = ("qwen3-4b", "llama-3.1-8b")
SERIES = ("Measured", "Midlayer", "LLMServingSim 2.0")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())



def _native_result_path(directory: Path, comparison: dict) -> Path:
    historical = Path(comparison["native_run_manifest"])
    if (historical.name != "run.json"
            or historical.parts[-5:-2] != (directory.parent.name, directory.name, "llmservingsim")
            or not historical.parent.name.startswith("run-")):
        raise ValueError("accepted native run manifest does not match the case")
    return directory / "llmservingsim" / historical.parent.name / "normalized.json"

def load_ttft_case(directory: Path, model: str, *, hardware_id: str = "ascend_910b",
                   case_id: str = "tp1_qps16") -> dict:
    """Validate the accepted case identities and summaries without running simulation."""
    mapping = _read(directory / "request_mapping.json")["requests"]
    truth = _read(directory / "canonical/metrics.json")["per_request"]
    prediction = _read(directory / "midlayer-request-metrics/prediction.json")["requests"]
    comparison = _read(directory / "comparison.json")
    native = _read(_native_result_path(directory, comparison))["requests"]
    expected = list(range(128))
    if (comparison["case"] != f"{hardware_id}/{model}/{case_id}"
            or comparison["llmservingsim_status"] != "completed"):
        raise ValueError("baseline case identity/status mismatch")
    if ([r["native_request_id"] for r in mapping] != expected
            or [r["request_id"] for r in mapping] != [str(i) for i in expected]
            or len(truth) != 128 or len(native) != 128
            or set(prediction) != {f"request-{i:012d}" for i in expected}):
        raise ValueError("expected exactly 128 matching request identities")
    truth_by_id = {r["request_idx"]: r for r in truth}
    native_by_id = {r["request_id"]: r for r in native}
    if set(truth_by_id) != set(expected) or set(native_by_id) != {str(i) for i in expected}:
        raise ValueError("duplicate, missing or unknown request identity")
    values = {name: [] for name in SERIES}
    for index, request in enumerate(mapping):
        measured = truth_by_id[index]
        simulated = native_by_id[str(index)]
        arrival_ms = finite_nonnegative(request["arrival_time_ms"], "canonical arrival")
        native_arrival_ms = finite_nonnegative(simulated["arrival_time_s"], "native arrival") * 1000
        if (simulated["native_request_id"] != index
                or measured["actual_output_tokens"] != request["output_tokens"]
                or simulated["output_tokens"] != request["output_tokens"]
                or simulated["input_tokens"] != request["input_tokens"]
                or abs(native_arrival_ms - arrival_ms) > 0.000001):
            raise ValueError("request workload identity mismatch")
        values["Measured"].append(finite_nonnegative(measured["ttft_ms"], "measured TTFT") / 1000)
        values["Midlayer"].append(finite_nonnegative(prediction[f"request-{index:012d}"]["ttft_s"], "Midlayer TTFT"))
        values["LLMServingSim 2.0"].append(finite_nonnegative(simulated["ttft_s"], "LLMServingSim TTFT"))
    summaries = {name: distribution(rows) for name, rows in values.items()}
    saved = (comparison["midlayer_vs_truth"]["ttft_s"]["reference"],
             comparison["midlayer_vs_truth"]["ttft_s"]["prediction"],
             comparison["llmservingsim_vs_truth"]["ttft_s"]["prediction"])
    for name, reference in zip(SERIES, saved):
        for key in ("count", "mean", "p50", "p90", "p99"):
            if not math.isclose(summaries[name][key], reference[key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{name} TTFT differs from accepted comparison: {key}")
    return {"model": model, "source": str(directory.resolve()), "ttft_s": values,
            "summary_s": summaries, "limitations": comparison["profile_evidence"]["limitations"]}


def render_llmservingsim_section(results_root: Path, output: Path) -> list[str]:
    """Write both model panels once and return Markdown for a scoped report section."""
    cases = [load_ttft_case(results_root / "targets/ascend_910b" / model, model) for model in MODELS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    png = plot_dir / "ascend_910b_llmservingsim_ttft_cdf.png"
    pdf = png.with_suffix(".pdf")
    colors = ("#222222", "#0072B2", "#D55E00")
    styles = ("-", "--", "-.")
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
        for ax, case in zip(axes, cases):
            for name, color, style in zip(SERIES, colors, styles):
                ordered = sorted(value * 1000 for value in case["ttft_s"][name])
                ax.step([0.0, *ordered], [0.0, *[(i + 1) / len(ordered) for i in range(len(ordered))]],
                        where="post", label=name, color=color, linestyle=style, linewidth=2)
            ax.set_title({"qwen3-4b": "Qwen3-4B", "llama-3.1-8b": "Llama-3.1-8B"}[case["model"]])
            ax.set(xlabel="TTFT (ms)", xlim=(0, None), ylim=(0, 1))
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right", fontsize=9)
        axes[0].set_ylabel("Cumulative fraction of requests")
        fig.suptitle("Ascend 910B | TP1 | QPS16 | BF16 | 128 requests per model")
        fig.savefig(png, dpi=180)
        fig.savefig(pdf)
        plt.close(fig)
    summary_path = output.parent / "llmservingsim_ttft_summary.json"
    summary_path.write_text(json.dumps({"unit": "seconds", "cases": cases}, indent=2) + "\n")
    relative = lambda path: os.path.relpath(path, output.parent)
    lines = ["## LLMServingSim 2.0: 910B TTFT distributions", "",
        "This section reuses accepted historical comparison results: Ascend 910B, BF16, TP1, QPS16, 128 requests for each of two models. It does not imply a new run or a refreshed full matrix. Measured, Midlayer, and LLMServingSim 2.0 share request IDs, input/output lengths, and arrival times (native nanosecond rounding within 1 ns).", "",
        f"![910B three-way TTFT distribution]({relative(png)})", "",
        f"[PDF]({relative(pdf)}) · [Per-request data and statistics]({relative(summary_path)})", "",
        "The x-axis is time to first token (ms); the y-axis is the empirical cumulative distribution. Axes start at zero; curves are independently sorted, so a shared x coordinate does not identify the same request. No smoothing, imputation, or resampling is applied.", "",
        "| Model | Method | Requests | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        for name in SERIES:
            stat = case["summary_s"][name]
            lines.append(f"| {case['model']} | {name} | {stat['count']} | " + " | ".join(f"{stat[k] * 1000:.3f}" for k in ("mean", "p50", "p90", "p99")) + " |")
    lines += ["", "LLMServingSim 2.0 uses measured compute LUTs captured from single-layer dummy-weight/eager execution, retaining interpolation, extrapolation, and software-execution differences. Analytical describes its network backend, rather than purely analytical Transformer compute. Request and KV budgets are aligned, while runtimes are not fully equivalent. This plot does not attribute TTFT differences to a specific Graph/CPU overhead.", "", "Input directories and original comparison limitations:", ""]
    for case in cases:
        lines += [f"- `{case['source']}`", f"- {case['model']} limitations: `{json.dumps(case['limitations'], ensure_ascii=False)}`"]
    lines += ["", "Regenerate (reads saved results only; runs no performance experiment):", "", "```bash",
              "python scripts/evaluation/run_standard_evaluation_report.py "
              "--llmservingsim-report-only --llmservingsim-results "
              + shlex.quote(str(results_root.resolve())) + " --output-dir "
              + shlex.quote(str(output.parent.resolve())), "```", ""]
    return lines


def write_llmservingsim_report(results_root: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Serving TTFT comparison report", "", *render_llmservingsim_section(results_root, output)]
    output.write_text("\n".join(lines) + "\n")
    return output


def _same_canonical_inputs(current: Path, archived: Path, cell: dict) -> bool:
    """Match workload semantics, not only model and display labels."""
    import csv
    current_manifest = _read(current / "manifest.json")
    archived_manifest = _read(archived / "manifest.json")
    for name in ("hardware_id", "model_id", "case_id"):
        if (current_manifest["identity"][name] != cell[name]
                or current_manifest["identity"][name] != archived_manifest["identity"][name]):
            return False
    for name in ("dtype", "tensor_parallel_size", "pipeline_parallel_size",
                 "expert_parallel_size", "target_qps", "capture_enable_prefix_caching",
                 "scheduler_pipeline_depth"):
        if current_manifest["runtime"][name] != archived_manifest["runtime"][name]:
            return False
    if current_manifest["runtime"]["target_qps"] != cell["target_qps"]:
        return False
    with (current / "requests.csv").open(newline="") as stream:
        current_rows = list(csv.DictReader(stream))
    with (archived / "requests.csv").open(newline="") as stream:
        archived_rows = list(csv.DictReader(stream))
    if len(current_rows) != cell["request_count"] or len(current_rows) != len(archived_rows):
        return False
    for now, old in zip(current_rows, archived_rows, strict=True):
        if set(now) != set(old):
            return False
        for field in now:
            a = finite_nonnegative(float(now[field]), "canonical request field")
            b = finite_nonnegative(float(old[field]), "archived request field")
            if a != b:
                return False
    return True


def match_standard_baselines(results_root: Path, serving_payload: dict) -> dict:
    """Return one explicit baseline status for every standard case; never synthesize data."""
    statuses = {}
    for cell in serving_payload["cells"]:
        cell_id = cell["cell_id"]
        if cell_id in statuses:
            raise ValueError("duplicate standard report case")
        hardware, model, case_id = cell["hardware_id"], cell["model_id"], cell["case_id"]
        if cell_id != f"{hardware}/{model}/{case_id}":
            raise ValueError("standard report cell identity mismatch")
        status = {"status": "missing_profile", "reason": "The accepted performance tables lack a matching LUT for this hardware/model/precision/TP combination. Existing tables cover Qwen3-4B/Llama-3.1-8B BF16 TP1 on three hardware platforms only; this does not imply the upstream architecture is unsupported.",
                  "metrics_ms": None, "summary_ms": None, "source": None}
        statuses[cell_id] = status
        if hardware == "nvidia_h200":
            status.update(status="deferred_by_user", reason="The user deferred H200 LLMServingSim experiments.")
            continue
        if hardware not in ("nvidia_a800", "metax_c500", "ascend_910b") or model not in MODELS or case_id not in ("tp1_qps4", "tp1_qps16", "tp1_qps64"):
            continue
        if "workload_path" in serving_payload and serving_payload["workload_path"] == "virtual_tensor":
            if cell["prediction_status"] != "completed":
                status.update(status="not_comparable", reason="Prediction for this virtual case is incomplete; per-request comparison is unavailable.")
                continue
            metrics = cell["metrics"]
            if metrics["comparison_kind"] == "measured_vs_predicted_anonymous_distributions":
                if metrics["paired_request_comparison"] != "not_available_saved_result_omits_request_mapping":
                    raise ValueError("unknown anonymous virtual comparison layout")
                status.update(status="not_comparable", reason="The virtual result retains only anonymous latency distributions, without per-request pairing evidence; it cannot be overlaid as a per-request comparison.")
                continue
            if (metrics["comparison_kind"] != "measured_vs_predicted_by_request_idx"
                    or metrics["paired_request_comparison"] != "paired_before_anonymous_projection"):
                raise ValueError("unknown virtual LLMServingSim comparison layout")
        base_directory = results_root / "targets" / hardware / model
        if (base_directory / case_id).is_dir():
            directory = (base_directory / case_id).resolve()
        elif case_id == "tp1_qps16" and (base_directory / "comparison.json").is_file():
            directory = base_directory
        else:
            status.update(status="not_run", reason="A compatible TP1 LUT exists, but complete native simulation results for this QPS case have not been generated.")
            continue
        validated = load_ttft_case(directory, model, hardware_id=hardware, case_id=case_id)
        status["source"] = str(directory.resolve())
        base = (Path(serving_payload["hardware_dataset_roots"][hardware])
                if hardware in serving_payload["hardware_dataset_roots"]
                else Path(serving_payload["dataset_root"]))
        canonical = base / hardware / model / case_id
        if not _same_canonical_inputs(canonical, directory / "canonical", cell):
            status.update(status="input_mismatch", reason="Canonical identity/runtime/request inputs differ between the standard case and accepted baseline; overlay is prohibited.")
            continue
        paired = cell["metrics"]["paired_request_latency"]
        if paired is None:
            status.update(status="not_comparable", reason="LLMServingSim results exist, but this standard case lacks Midlayer per-request paired metrics; no overlay is produced.")
            continue
        comparison = _read(directory / "comparison.json")
        native = _read(_native_result_path(directory, comparison))["requests"]
        native_by_id = {row["request_id"]: row for row in native}
        truth = {row["request_idx"]: row for row in _read(directory / "canonical/metrics.json")["per_request"]}
        metrics, summaries = {}, {}
        matched = True
        for field_ms, field_s in (("ttft_ms", "ttft_s"), ("tpot_ms", "tpot_s"), ("e2e_latency_ms", "latency_s")):
            records = paired["metrics"][field_ms]
            if len(records) != 128 or [r["request_idx"] for r in records] != list(range(128)):
                raise ValueError("standard paired metrics have invalid request identities")
            values = [finite_nonnegative(native_by_id[str(i)][field_s], field_s) for i in range(128)]
            stat = distribution(values)
            reference = comparison["llmservingsim_vs_truth"][field_s]["prediction"]
            for name in ("count", "mean", "p50", "p90", "p99"):
                if not math.isclose(stat[name], reference[name], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("native metric differs from accepted comparison")
            for i, row in enumerate(records):
                observed = finite_nonnegative(row["measured_ms"], "standard measured metric")
                if not math.isclose(observed, truth[i][field_ms], rel_tol=1e-10, abs_tol=1e-8):
                    matched = False
            metrics[field_ms] = [v * 1000 for v in values]
            summaries[field_ms] = {name: stat[name] * 1000 for name in ("mean", "p50", "p90", "p99")}
        if not matched:
            status.update(status="input_mismatch", reason="Per-request observed TTFT/TPOT/E2E differ between the standard report and baseline; overlay is prohibited.")
            continue
        status.update(status="available", reason="Inputs for all 128 requests and per-request observations match. Compute uses measured LUTs with capture and runtime limitations retained.",
                      metrics_ms=metrics, summary_ms=summaries, limitations=validated["limitations"],
                      native_run_manifest=str(_native_result_path(directory, comparison).with_name("run.json")))
    return statuses


def baseline_status_table(cells, statuses: dict) -> list[str]:
    lines = ["### LLMServingSim 2.0: per-case coverage and error", "",
             "Errors are signed relative errors of means. Missing/deferred entries are excluded from accuracy means and do not indicate upstream unsupported status. Available TTFT/TPOT results are included in the per-request and distribution plots above.", "",
             "| Model / Case | LLMServingSim 2.0 | TTFT mean error | TPOT mean error | E2E mean error | Reason |",
             "|---|---|---:|---:|---:|---|"]
    for cell in cells:
        entry = statuses[cell["cell_id"]]
        errors = []
        for metric in ("ttft_ms", "tpot_ms", "e2e_latency_ms"):
            if entry["status"] == "available":
                observed = sum(row["measured_ms"] for row in cell["metrics"]["paired_request_latency"]["metrics"][metric]) / cell["request_count"]
                error = f"{(entry['summary_ms'][metric]['mean'] / observed - 1) * 100:+.2f}%" if observed > 0 else "undefined (zero mean)"
            else:
                error = "—"
            errors.append(error)
        lines.append(f"| {cell['model_id']} / {cell['case_id']} | {entry['status']} | " + " | ".join(errors) + f" | {entry['reason']} |")
    return lines
