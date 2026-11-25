"""Offline Kernel overviews over the existing semantic artifact, without prediction."""
from pathlib import Path
import json
import time

from evaluation.reporting.serving_interactive import write_status_report
from evaluation.reporting.serving_viewer_ui import (
    attach_report_costs, report_experiment_configuration, _configuration_metadata,
)


def default_kernel_report_path():
    import tempfile
    root = Path(__file__).resolve().parents[2] / "experiment_logs/evaluation/kernel/reports"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="run-", dir=root)) / "kernel.html"


def write_kernel_overview(artifact, metrics, arguments, output, *, elapsed_s,
                          hardware_snapshot=None, failure=None):
    """Publish recorded comparisons; None stays unavailable and unsupported stays explicit."""
    started = time.perf_counter()
    metrics = json.loads(json.dumps(metrics, allow_nan=False))
    output = Path(output).resolve()
    facts = [{"label": "Hardware", "value": arguments["hardware_id"]},
             {"label": "Evaluation mode", "value": arguments["mode"]}]
    links = [{"label": "Kernel output cost", "path": str(output.with_suffix(".report.json"))}]
    if failure is not None:
        status = "unsupported" if failure["type"] == "NotImplementedError" else "failed"
        reason = failure["type"] + ": " + failure["message"]
    else:
        if artifact is None:
            raise ValueError("successful Kernel overview requires semantic results")
        facts.append({"label": "Total cases", "value": artifact["case_count"]})
        status = "complete"
        for method, values in metrics.items():
            if values is None:
                continue
            text = {"ok": "Completed", "unsupported": "Unsupported", "out_of_domain": "Out of pricing domain"}[values["status"]]
            if values["status"] == "ok":
                text = f"Scored {values['scored']} / {values['N']} · MAPE {values['MAPE']:.2f}% · Median {values['MedAPE']:.2f}% · P90 {values['P90APE']:.2f}%"
            text += f" · Unsupported {values['unsupported']}" if "unsupported" in values else ""
            text += f" · Out of pricing domain {values['out_of_domain']}" if "out_of_domain" in values else ""
            facts.append({"label": method + " error / coverage", "value": text})
            if values["status"] != "ok" or ("unsupported" in values and values["unsupported"]) or ("out_of_domain" in values and values["out_of_domain"]):
                status = "partial"
        if not any(value is not None and value["status"] == "ok" for value in metrics.values()):
            status = "unsupported"
        reason = "Kernel evaluation only. Errors cover paired cases; unsupported and out-of-domain cases are listed separately. Prediction availability and mechanism acceptance are recorded separately in coverage and diagnostics."
        # HTML and semantic sidecar both retain all saved cases.
        semantic_path = output.with_suffix(".semantic.json")
        semantic_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_path.write_text(json.dumps(artifact, allow_nan=False, indent=2) + "\n")
        links.append({"label": "Full Kernel semantic results", "path": str(semantic_path)})
        # First-screen result chart: actual-vs-Forward latency scatter from the
        # same saved pairs the accuracy table scores, colored by kernel family
        # with the existing scatter palette.
        chart_path = output.with_suffix(".latency.png")
        import math

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        label_font = FontProperties(family="DejaVu Sans")

        from evaluation.kernel.latency_scatter_report import (
            SERIES as _SCATTER_SERIES,
            _family_index as _family_index,
        )
        points = []
        for case in artifact["cases"]:
            pair = case["forward"]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            op_type = case["report_meta"]["op_type"]
            points.append((pair[1] / 1000.0, pair[0] / 1000.0,
                           _family_index(op_type)))
        if points:
            values = [v for point in points for v in point[:2]]
            lower = 10 ** math.floor(math.log10(min(values)))
            upper = 10 ** math.ceil(math.log10(max(values)))
            figure, axis = plt.subplots(figsize=(6.4, 6.0))
            for family_index_i, (_family, color, marker) in enumerate(_SCATTER_SERIES):
                subset = [p for p in points if p[2] == family_index_i]
                if subset:
                    axis.scatter([p[0] for p in subset], [p[1] for p in subset],
                        s=14, c=color, marker=marker, alpha=0.75, linewidths=0, label=_family)
            axis.plot([lower, upper], [lower, upper], color="#666666",
                linewidth=1.0, linestyle="--", label="Ideal: predicted = measured")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlim(lower, upper)
            axis.set_ylim(lower, upper)
            axis.set_xlabel("Measured latency (µs)", fontproperties=label_font)
            axis.set_ylabel("Predicted latency (µs)", fontproperties=label_font)
            axis.set_title(arguments["hardware_id"] + " · Forward predicted vs measured (" + str(len(points)) + " cases)", fontproperties=label_font)
            axis.legend(prop=label_font, fontsize=8)
            axis.grid(True, which="both", linewidth=0.3, alpha=0.4)
            figure.tight_layout()
            chart_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(chart_path, dpi=150, facecolor="white")
            plt.close(figure)
            links.append({"label": "Forward latency scatter", "path": str(chart_path)})
    payload = {"main_chain_status": {"status": status, "reason": reason}, "report_facts": facts,
        "report_links": links,
        "visualization_run_costs": {"scope": "Kernel evaluation before report finalization", "evaluation_before_report_s": elapsed_s}}
    config = report_experiment_configuration(payload)
    config["provenance"]["kernel_evaluation"] = {"arguments": _configuration_metadata(arguments),
        "dataset": artifact["dataset"] if artifact is not None else None,
        "hardware_config": artifact["hardware_config"] if artifact is not None else None, "metrics": metrics}
    config["files"] = [] if hardware_snapshot is None else [hardware_snapshot]
    payload["visualization_experiment_configuration"] = config
    if artifact is not None and failure is None:
        from evaluation.reporting.kernel_viewer import kernel_display_projection
        payload["kernel_overview"] = kernel_display_projection(artifact, metrics)
        if any(level != "complete" for level in payload["kernel_overview"]["forward_kernel_support"]):
            if status == "complete":
                status = "partial"
                payload["main_chain_status"]["status"] = status
    receipt = write_status_report(payload, output)
    costs = {"scope": "Kernel main body through HTML and projection write; receipt attachment/write measured separately",
        "html_generation_s": receipt["page_generation_s"],
        "evaluation_before_report_s": elapsed_s, "report_generation_s": time.perf_counter() - started,
        "total_wall_through_result_write_s": elapsed_s + time.perf_counter() - started}
    attach_started = time.perf_counter()
    attach_report_costs(receipt, costs)
    costs["cost_metadata_attachment_s"] = time.perf_counter() - attach_started
    receipt["report_output_costs"] = costs
    receipt_path = output.with_suffix(".report.json")
    receipt["cost_receipt_path"] = str(receipt_path)
    receipt_path.write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    receipt["total_wall_through_report_receipt_s"] = elapsed_s + time.perf_counter() - started
    return receipt


def regenerate_kernel_overview(source, output):
    """Rebuild a saved Kernel page, retaining original metrics, cases and costs."""
    from evaluation.reporting.kernel_viewer import kernel_display_projection

    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.suffix != ".html":
        raise ValueError("regenerated Kernel report needs a separate .html output")
    if output.exists() or any(output.with_suffix(suffix).exists() for suffix in
            (".visualization.json", ".semantic.json", ".report.json")):
        raise FileExistsError("regenerated report output already exists")
    artifact = json.loads(source.with_suffix(".semantic.json").read_text())
    previous = json.loads(source.with_suffix(".visualization.json").read_text())
    original_receipt = json.loads(source.with_suffix(".report.json").read_text())
    configuration = report_experiment_configuration(previous)
    metrics = configuration["provenance"]["kernel_evaluation"]["metrics"]
    display = kernel_display_projection(artifact, metrics)
    costs = dict(original_receipt["report_output_costs"])
    if "html_generation_s" not in costs and "page_generation_s" in original_receipt:
        costs["html_generation_s"] = original_receipt["page_generation_s"]
    payload = {"main_chain_status": dict(previous["main_chain_status"]),
        "visualization_experiment_configuration": configuration,
        "visualization_run_costs": costs, "kernel_overview": display,
        "report_links": [{"label": "Original Kernel report", "path": str(source)},
            {"label": "Full Kernel semantic results", "path": str(output.with_suffix(".semantic.json"))},
            {"label": "Kernel output cost", "path": str(output.with_suffix(".report.json"))}]}
    payload["main_chain_status"]["reason"] = (
        "Page rebuilt from saved experiment results. Errors cover paired cases; "
        "original summary status and paired subsets are shown separately, with Forward support limits in coverage and diagnostics.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".semantic.json").write_bytes(source.with_suffix(".semantic.json").read_bytes())
    receipt = write_status_report(payload, output)
    costs["regeneration_wall_s"] = time.perf_counter() - started
    attach_report_costs(receipt, costs)
    receipt["report_output_costs"] = costs
    receipt["regenerated_from"] = str(source)
    receipt["cost_receipt_path"] = str(output.with_suffix(".report.json"))
    output.with_suffix(".report.json").write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    return receipt


def write_kernel_matrix_overview(payload, configuration, output):
    """Link existing matrix outcomes and child reports without reparsing predictions."""
    started = time.perf_counter()
    output = Path(output).resolve()
    facts, links = [], []
    if "kernel" in configuration:
        for field, label in (("overall_plot_path", "All Kernel latency scatter"),
                             ("detailed_plot_path", "Per-operator latency and error distributions")):
            links.append({"label": label, "path": configuration["kernel"][field]})
    statuses = []
    for hardware_id, entry in payload["kernel_matrix"].items():
        status = entry["status"]
        if status not in ("PASS", "FAIL", "SKIPPED"):
            raise ValueError("unknown Kernel matrix status: " + status)
        statuses.append(status)
        facts.append({"label": hardware_id + " · " + status, "value": entry["summary"]})
    reports = payload["kernel_reports"] if "kernel_reports" in payload else {}
    for hardware_id, report in reports.items():
        if report["status"] == "available":
            links.append({"label": hardware_id + " Kernel overview", "path": report["path"]})
        else:
            facts.append({"label": hardware_id + " report coverage", "value": report["status"]})
    if not reports:
        facts.append({"label": "Per-hardware overview coverage", "value": "Not recorded in this historical matrix"})
    status = "failed" if "FAIL" in statuses else "partial" if "SKIPPED" in statuses else "complete"
    run = {"main_chain_status": {"status": status, "reason": "Kernel matrix overview. Existing accuracy rows and plots remain in the standard Markdown report; Serving results are separate."},
        "report_facts": facts, "report_links": links}
    config = report_experiment_configuration(run)
    config["provenance"]["kernel_report"] = _configuration_metadata(configuration)
    run["visualization_experiment_configuration"] = config
    receipt = write_status_report(run, output)
    receipt["report_output_costs"] = {"report_generation_s": time.perf_counter() - started,
        "scope": "matrix HTML/projection generation and writes; evaluation/plots/Markdown assembly reported separately"}
    path = output.with_suffix(".report.json")
    receipt["cost_receipt_path"] = str(path)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
