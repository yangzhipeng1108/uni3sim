"""Update the canonical Kernel/full-Serving report from existing evaluators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]

from evaluation.reporting.standard_evaluation_report import (
    KERNEL_HARDWARE_ORDER,
    REQUIRED_KERNEL_HARDWARE,
    timestamp_now,
    write_kernel_evaluation_report,
    write_serving_evaluation_report,
    _write_serving_evaluation_report_payload,
)

from evaluation.json_shape import write_json_atomic as _write_json_atomic


DEFAULT_OUTPUT = ROOT / "experiment_logs/evaluation/standard_report"
KERNEL_RUNS_ROOT = ROOT / "experiment_logs/evaluation/kernel_runs"
SERVING_RUNS_ROOT = ROOT / "experiment_logs/evaluation/serving_runs"
DEFAULT_SERVING_PYTHON = Path(sys.executable)
_KERNEL_TIMING_LINE = re.compile(
    r"^\[kernel\]\s+(?P<hardware>[a-z0-9_]+):\s+"
    r"(?P<status>PASS|FAIL|SKIPPED)\s+\((?P<seconds>[0-9]+(?:\.[0-9]+)?)s\)$"
)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--kernel-only", action="store_true")
    mode.add_argument("--serving-only", action="store_true")
    mode.add_argument("--all", action="store_true")
    mode.add_argument("--refresh-serving-report", action="store_true")
    mode.add_argument("--refresh-kernel-report", action="store_true")
    mode.add_argument("--llmservingsim-report-only", action="store_true")
    parser.add_argument("--llmservingsim-results", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jobs", type=int, default=0,
        metavar="N", help="parallel Serving case workers; 0 uses one worker per selected case (default)",
    )
    parser.add_argument(
        "--kernel-jobs", type=int, default=16,
        metavar="N",
    )
    parser.add_argument(
        "--case", action="append", default=[],
        help="representative Serving hardware/model/case_id; repeat as needed",
    )
    parser.add_argument(
        "--reuse-serving-results",
        action="append",
        type=Path,
        default=[],
        help="unsupported here; use the independent scheduler-depth diagnostic --reuse-results",
    )
    parser.add_argument(
        "--nvidia-h200-vllm-version", choices=("0.30.0", "0.11.0", "none"),
        default="0.30.0",
        help=("H200 Serving cells of this captured vLLM version (canonical: "
              "0.30.0); 'none' leaves H200 out; 0.11.0 requires the independent diagnostic"),
    )
    parser.add_argument(
        "--nvidia-h200-dataset-root", type=Path,
        help=(
            "unsupported here; candidate roots belong to the independent scheduler-depth diagnostic"),
    )
    parser.add_argument(
        "--serving-python",
        type=Path,
        default=DEFAULT_SERVING_PYTHON,
        help="Python executable containing the pinned vLLM runtime",
    )
    parser.add_argument("--serving-timing-mode", choices=("device_only", "device_plus_launch",
        "device_plus_launch_plus_scheduler"), default="device_plus_launch_plus_scheduler")
    parser.add_argument("--serving-execution-mode", choices=("accelerated", "fast"),
        default="fast", help="fast source-structure execution (default), or accelerated for detailed memory-binding execution")
    parser.add_argument("--serving-execution-workers", type=int, default=0,
        help="per-case streaming tail workers; 0 keeps the automatic one-worker-per-case default")
    parser.add_argument("--serving-state-audit", action="store_true",
        help="write per-rank work and source-memory lifecycle sidecars")
    parser.add_argument("--h200-serving-scenario", choices=("analytical_bf16_v1", "captured"),
        default="analytical_bf16_v1",
        help=("H200 physical analytical reference (default), or captured for the original "
              "strict launch contract; reports retain both captured/effective settings; "
              "other hardware retains its normal settings"))
    parser.add_argument("--vllm-sdk-python", type=Path)
    parser.add_argument("--vllm-source-root", type=Path)
    parser.add_argument("--reuse-virtual-run", type=Path,
        help="existing virtual run directory; validate exact commands and regenerate metrics/report without simulation")
    parser.add_argument(
        "--with-html", action="store_true",
        help="Also write the offline interactive HTML report and its projections (default: Markdown-only)")
    args = parser.parse_args(argv)
    try:
        _validate_serving_source_options(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _run_checked(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    with log_path.open("w") as log_file, subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("evaluation subprocess stdout is unavailable")
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}; see {log_path}"
        )
    return elapsed


def _run_kernel(stamp: str, jobs: int, *, with_html: bool = False) -> dict[str, object]:
    component_dir = KERNEL_RUNS_ROOT / stamp
    component_dir.mkdir(parents=True, exist_ok=False)
    result_path = component_dir / "kernel_matrix.json"
    evaluation_log = component_dir / "evaluation.log"
    command = [
        sys.executable,
        "scripts/evaluation/run_full_evaluation.py",
        "--part", "kernel",
        # Keep cards serial: each card receives the entire fixed CPU set for
        # its own 45-case-worker measurement instead of three cards competing
        # with 135 processes.  The standard report's --kernel-jobs therefore
        # controls case workers, matching the public report contract.
        "--kernel-jobs", "1",
        "--kernel-case-jobs", str(jobs),
        "--kernel-case-timing-dir", str(component_dir / "case_timing"),
        "--kernel-semantic-output-dir", str(component_dir / "semantic"),
        "--dump-json", str(result_path),
    ]
    if with_html:
        command.append("--with-html")
    evaluation_wall = _run_checked(command, evaluation_log)
    hardware_wall_times = {}
    for line in evaluation_log.read_text().splitlines():
        match = _KERNEL_TIMING_LINE.fullmatch(line)
        if match is None:
            continue
        hardware_id = match.group("hardware")
        if hardware_id in hardware_wall_times:
            raise ValueError(f"duplicate Kernel timing row for {hardware_id}")
        hardware_wall_times[hardware_id] = float(match.group("seconds"))
    if not set(REQUIRED_KERNEL_HARDWARE) <= set(hardware_wall_times):
        raise ValueError("Kernel evaluation log is missing hardware wall times")
    reported_hardware = sorted(
        hardware_id for hardware_id in KERNEL_HARDWARE_ORDER
        if hardware_id in hardware_wall_times)
    overall_plot = component_dir / "kernel_latency_overall.png"
    detailed_plot = component_dir / "kernel_latency_by_bucket.png"
    plot_wall = _run_checked([
        sys.executable,
        "scripts/evaluation/kernel/kernel_latency_scatter_report.py",
        "--jobs", str(jobs),
        "--semantic-input-dir", str(component_dir / "semantic"),
        "--output", str(overall_plot),
        "--detailed-output", str(detailed_plot),
    ], component_dir / "plot.log")
    traffic_plot = component_dir / "kernel_memory_traffic.png"
    traffic_summary = component_dir / "kernel_memory_traffic.json"
    traffic_wall = _run_checked([
        sys.executable,
        "scripts/evaluation/kernel/kernel_memory_traffic_report.py",
        "--jobs", str(jobs),
        "--output", str(traffic_plot),
        "--summary-output", str(traffic_summary),
    ], component_dir / "traffic_plot.log")
    return {
        "generated_at": timestamp_now(),
        "source_commit": _source_commit(),
        "result_path": str(result_path.resolve()),
        "overall_plot_path": str(overall_plot.resolve()),
        "detailed_plot_path": str(detailed_plot.resolve()),
        "traffic_plot_path": str(traffic_plot.resolve()),
        "traffic_summary_path": str(traffic_summary.resolve()),
        "evaluation_wall_time_s": evaluation_wall,
        "hardware_evaluation_wall_time_s": {
            hardware_id: hardware_wall_times[hardware_id]
            for hardware_id in reported_hardware
        },
        "semantic_artifact_paths": {
            hardware_id: str((component_dir / "semantic" / f"{hardware_id}.json").resolve())
            for hardware_id in reported_hardware
            if (component_dir / "semantic" / f"{hardware_id}.json").is_file()
        },
        "plot_wall_time_s": plot_wall,
        "traffic_plot_wall_time_s": traffic_wall,
    }


def _load_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(
            f"{path} does not exist; use --all for the first standard report run"
        )
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value["schema_version"] != 1:
        raise ValueError("standard report state is invalid")
    return value


def _validate_serving_source_options(args):
    if (args.refresh_serving_report or args.refresh_kernel_report
            or args.llmservingsim_report_only):
        return
    if (args.reuse_serving_results or args.nvidia_h200_dataset_root is not None
            or args.nvidia_h200_vllm_version == "0.11.0"):
        raise ValueError(
            "standard Serving reports execute only the virtual Tensor path; "
            "--reuse-serving-results, --nvidia-h200-dataset-root and H2000.11.0 "
            "belong to scripts/evaluation/serving/diagnostics/run_scheduler_depth_breakdown.py "
            "--full-serving-report-only (use --reuse-results there); "
            "historical state remains readable with --kernel-only")


def _run_serving(args, stamp, *, payloads=None):
    _validate_serving_source_options(args)
    if args.vllm_sdk_python is None or args.vllm_source_root is None:
        raise ValueError("virtual Tensor evaluation needs --vllm-sdk-python and --vllm-source-root")
    if not args.vllm_sdk_python.is_file():
        raise ValueError("virtual Tensor SDK Python does not exist")
    if args.nvidia_h200_vllm_version not in ("0.30.0", "none"):
        raise ValueError("virtual Tensor evaluation requires H200 0.30.0 or none")
    run_dir = (SERVING_RUNS_ROOT / f"virtual-report-{stamp}" if args.reuse_virtual_run is None
        else args.reuse_virtual_run.resolve(strict=True))
    if SERVING_RUNS_ROOT.resolve() not in run_dir.parents:
        raise ValueError("virtual reuse must stay inside the Serving run directory")
    SERVING_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = SERVING_RUNS_ROOT / f"virtual-report-{stamp}.md"
    command = [str(args.serving_python.resolve(strict=True)),
        "scripts/evaluation/serving/report_symbolic_model.py",
        "--serving-matrix-output", str(run_dir), "--output", str(report_path),
        "--jobs", str(args.jobs), "--serving-python", str(args.serving_python.resolve(strict=True)),
        # A venv executable is often a symlink. Resolving it discards the
        # environment's pyvenv.cfg and selects a different dependency set.
        "--vllm-sdk-python", str(args.vllm_sdk_python.absolute()),
        "--vllm-source-root", str(args.vllm_source_root.resolve(strict=True)),
        "--h200-capture-version", args.nvidia_h200_vllm_version,
        "--serving-timing-mode", args.serving_timing_mode,
        "--serving-execution-mode", args.serving_execution_mode,
        "--serving-execution-workers", str(args.serving_execution_workers)]
    if args.serving_state_audit:
        command.append("--serving-state-audit")
    if args.with_html:
        # The standard report's single HTML switch also opts the Serving
        # matrix into its offline interactive pages; Markdown is unaffected.
        command.append("--interactive-report")
    if args.h200_serving_scenario not in (None, "captured"):
        command += ["--h200-serving-scenario", args.h200_serving_scenario]
    for case in args.case:
        command.extend(("--case", case))
    if args.reuse_virtual_run is not None:
        command.append("--reuse-completed-virtual")
    wall = _run_checked(command, SERVING_RUNS_ROOT / f"virtual-report-{stamp}.log")
    result_path = run_dir / "results.json"
    payload = json.loads(result_path.read_text())
    if payloads is not None:
        payloads[str(result_path.resolve())] = payload
    if payload["workload_path"] != "virtual_tensor":
        raise ValueError("standard Serving evaluator returned a non-virtual workload identity")
    recorded_workers = payload["serving_execution_workers"]
    if (isinstance(recorded_workers, bool) or not isinstance(recorded_workers, int)
            or recorded_workers < 0):
        raise ValueError("standard Serving evaluator returned an invalid serving_execution_workers field")
    recorded_workers_auto = payload["serving_execution_workers_auto"]
    if type(recorded_workers_auto) is not bool:
        raise ValueError("standard Serving evaluator returned an invalid serving_execution_workers_auto field")
    return {"generated_at": timestamp_now(), "source_commit": payload["source_commit"], "workload_path": payload["workload_path"],
        "serving_timing_mode": args.serving_timing_mode,
        "serving_execution_mode": args.serving_execution_mode,
        "serving_execution_workers": recorded_workers,
        "serving_execution_workers_auto": recorded_workers_auto,
        "serving_state_audit": args.serving_state_audit,
        "nvidia_h200_vllm_version": args.nvidia_h200_vllm_version, "nvidia_h200_dataset_root": None,
        "result_path": str(result_path), "case_count": payload["cell_count"],
        **({"simulation_batch_timing": payload["simulation_batch_timing"]}
            if "simulation_batch_timing" in payload else {}),
        **({"serving_promoted_cases": payload["serving_promoted_cases"],
            "serving_worker_grant_counts": payload["serving_worker_grant_counts"]}
            if "serving_promoted_cases" in payload else {}),
        "evaluation_wall_time_s": payload["evaluation_wall_time_seconds"],
        "command_wall_time_s": wall, "plot_wall_time_s": 0.0}


def _source_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _assemble_serving_report(report, state, args, *, refresh=False, payloads=None):
    started = time.perf_counter()
    wall = _assemble_serving_markdown(report, state, args, refresh=refresh, payloads=payloads)
    if not args.with_html:
        return wall
    from evaluation.reporting.serving_interactive import write_status_report
    # Reuse the completed Markdown's plots and table values. No prediction,
    # step recording, or report projection is needed in the default mode.
    markdown = report.read_text()
    links = [{"label": label, "path": str((report.parent / path).resolve())}
        for label, path in re.findall(r"!\[([^\]]*)\]\(([^\n]+)\)", markdown)[:2]]
    links.append({"label": "Full Markdown report", "path": str(report)})
    key = str(Path(state["serving"]["result_path"]).resolve())
    payload = (payloads[key] if payloads is not None and key in payloads
        else json.loads(Path(key).read_text()))
    rows = []
    completed = 0
    for cell in payload["cells"]:
        status = cell["prediction_status"]
        if status == "completed":
            completed += 1
            values = []
            for metric, label in (("ttft_ms", "TTFT"), ("tpot_ms", "TPOT")):
                mean = cell["metrics"]["latency"][metric]["mean"]
                values.append(f"{label}: measured {mean['observed']:.3f} / predicted {mean['predicted']:.3f} ms")
            text = "; ".join(values)
        elif status in ("failed", "unsupported"):
            text = status + ": " + str(cell["prediction_reason"])
        else:
            raise ValueError("unknown Serving prediction status: " + status)
        rows.append({"label": cell["cell_id"], "value": text})
    write_status_report({"main_chain_status": {"status": "complete" if completed == len(rows) else "partial",
            "reason": "Saved Serving evaluation summary; a completed run does not imply accepted accuracy."},
        "report_facts": [{"label": "Completed / total cases", "value": f"{completed} / {len(rows)}"},
            {"label": "Serving results table", "value": rows}],
        "report_links": links}, report.with_suffix(".html"))
    with report.open("a") as stream:
        stream.write("\nOffline interactive overview: [HTML](" + report.with_suffix(".html").name + ")\n")
    return time.perf_counter() - started


def _assemble_serving_markdown(report, state, args, *, refresh=False, payloads=None):
    """Render known paired or anonymous evaluations without rerunning prediction."""
    serving = state["serving"]
    key = str(Path(serving["result_path"]).resolve())
    payload = (payloads[key] if payloads is not None and key in payloads
        else json.loads(Path(serving["result_path"]).read_text()))
    if "report_layout" in payload:
        if payload["report_layout"] != "standard_paired":
            raise ValueError("unknown virtual Serving report layout")
        from evaluation.serving.metrics.sharegpt_evaluation_report import write_sharegpt_evaluation_report
        completed = [cell for cell in payload["cells"] if cell["prediction_status"] == "completed"]
        result_dir = Path(serving["result_path"]).parent
        timing_path = result_dir / "report_generation_timing.json"
        if completed and not timing_path.is_file():
            write_sharegpt_evaluation_report(result_dir, completed,
                run_elapsed_wall_time_seconds=payload["evaluation_wall_time_seconds"],
                plot_jobs=len(completed) if args.jobs == 0 else args.jobs)
        serving["plot_wall_time_s"] = (json.loads(timing_path.read_text())["plot_wall_time_seconds"]
            if timing_path.is_file() else 0.0)
        return _write_serving_evaluation_report_payload(report, state, payload, started=time.perf_counter(),
            llmservingsim_results=args.llmservingsim_results, refresh_latency_plots=refresh)
    if "workload_path" in payload:
        if payload["workload_path"] != "virtual_tensor":
            raise ValueError("unknown standard Serving report layout")
        from evaluation.serving.metrics.symbolic_model import write_virtual_serving_evaluation_report
        started = time.perf_counter()
        write_virtual_serving_evaluation_report(serving["result_path"], report)
        if args.llmservingsim_results is not None:
            from evaluation.reporting.llmservingsim_report import match_standard_baselines, baseline_status_table
            from evaluation.serving.metrics.sharegpt_evaluation_report import write_sharegpt_request_latency_plots
            root = args.llmservingsim_results.resolve(strict=True)
            statuses = match_standard_baselines(root, payload)
            _write_json_atomic(report.parent / "llmservingsim_coverage.json", {"source": str(root), "cases": statuses})
            paired_cells = [cell for cell in payload["cells"] if cell["prediction_status"] == "completed"
                and cell["metrics"]["comparison_kind"] == "measured_vs_predicted_by_request_idx"]
            section = ["## LLMServingSim 2.0", "",
                "New Midlayer requests are paired by canonical request_idx before anonymous persistence. LLMServingSim reuses matching LUT results and verifies request and ground-truth identities individually. Older anonymous results remain not_comparable.", "",
                *baseline_status_table(payload["cells"], statuses)]
            if paired_cells:
                plots = write_sharegpt_request_latency_plots(report.parent / "plots", paired_cells,
                    baseline_by_cell={cell["cell_id"]: statuses[cell["cell_id"]] for cell in paired_cells})
                for label, path in plots.items():
                    section.extend(["", f"![{label}]({os.path.relpath(path, report.parent)})"])
            with report.open("a") as stream:
                stream.write("\n\n" + "\n".join(section) + "\n")
        return time.perf_counter() - started
    return _write_serving_evaluation_report_payload(report, state, payload, started=time.perf_counter(),
        llmservingsim_results=args.llmservingsim_results, refresh_latency_plots=refresh)


def run(argv=None) -> tuple[Path, ...]:
    args = _arguments(argv)
    output_dir = args.output_dir.resolve()
    if output_dir != ROOT and ROOT not in output_dir.parents:
        raise ValueError("standard report output must remain inside the repository")
    if args.refresh_serving_report:
        state = _load_state(output_dir / "state.json")
        report = output_dir / "serving_report.md"
        wall = _assemble_serving_report(report, state, args, refresh=True)
        hashes_path = output_dir / "hashes.json"
        if hashes_path.is_file():
            hashes = json.loads(hashes_path.read_text())
            hashes["sha256"]["serving_report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()
            hashes["report_assembly_wall_time_s"]["serving_report.md"] = wall
            _write_json_atomic(hashes_path, hashes)
        return (report,)
    if args.refresh_kernel_report:
        state = _load_state(output_dir / "state.json")
        report = output_dir / "kernel_report.md"
        wall = write_kernel_evaluation_report(report, state,
            with_html=args.with_html)
        hashes_path = output_dir / "hashes.json"
        if hashes_path.is_file():
            hashes = json.loads(hashes_path.read_text())
            hashes["sha256"]["kernel_report.md"] = hashlib.sha256(report.read_bytes()).hexdigest()
            hashes["report_assembly_wall_time_s"]["kernel_report.md"] = wall
            _write_json_atomic(hashes_path, hashes)
        return (report,)
    if args.llmservingsim_report_only:
        if args.llmservingsim_results is None:
            raise ValueError("--llmservingsim-report-only requires --llmservingsim-results")
        if (output_dir / "state.json").exists():
            raise ValueError("report-only output must not replace an existing standard report")
        from evaluation.reporting.llmservingsim_report import write_llmservingsim_report
        return (write_llmservingsim_report(args.llmservingsim_results.resolve(), output_dir / "serving_report.md"),)
    if args.kernel_only and args.llmservingsim_results is not None:
        raise ValueError("LLMServingSim plots require a Serving report mode")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    # Run-owned parsed objects; no global or cross-experiment result cache.
    serving_payloads = {}
    if args.all:
        # Both component runners inherit the caller's fixed CPU affinity.
        # Running their 45-worker pools together makes the Kernel card timing
        # depend on the concurrent Serving mix, so it is not a valid
        # cross-hardware timing comparison.  Each component still evaluates
        # its independent cases in parallel internally.
        kernel = _run_kernel(stamp, args.kernel_jobs, with_html=args.with_html)
        serving = _run_serving(args, stamp, payloads=serving_payloads)
        update_mode = "all"
    else:
        if state_path.is_file():
            previous = _load_state(state_path)
        elif args.serving_only:
            previous = {"kernel": None}
        else:
            previous = _load_state(state_path)
        if args.kernel_only:
            kernel = _run_kernel(stamp, args.kernel_jobs, with_html=args.with_html)
            serving = previous["serving"]
            update_mode = "kernel-only"
        elif args.serving_only:
            kernel = previous["kernel"]
            serving = _run_serving(args, stamp, payloads=serving_payloads)
            update_mode = "serving-only"
        else:
            raise AssertionError("one standard report mode must be selected")
    if update_mode in {"all", "serving-only"} and serving["workload_path"] != "virtual_tensor":
        raise ValueError("new standard Serving results must identify the virtual Tensor path")
    state = {
        "schema_version": 1,
        "source_commit": _source_commit(),
        "generated_at": timestamp_now(),
        "update_mode": update_mode,
        "kernel": kernel,
        "serving": serving,
    }
    _write_json_atomic(state_path, state)
    report_paths = []
    assembly_wall_times = {}
    if update_mode in {"all", "kernel-only"}:
        kernel_report = output_dir / "kernel_report.md"
        assembly_wall_times["kernel_report.md"] = write_kernel_evaluation_report(
            kernel_report, state, with_html=args.with_html,
        )
        report_paths.append(kernel_report)
    if update_mode in {"all", "serving-only"}:
        serving_report = output_dir / "serving_report.md"
        assembly_wall_times["serving_report.md"] = _assemble_serving_report(serving_report, state, args, payloads=serving_payloads)
        report_paths.append(serving_report)
        _write_json_atomic(state_path, state)
    legacy_report = output_dir / "report.md"
    if legacy_report.exists():
        legacy_report.unlink()
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            output_dir / "kernel_report.md",
            output_dir / "serving_report.md",
        )
        if path.is_file()
    }
    hashes["state.json"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    _write_json_atomic(output_dir / "hashes.json", {
        "schema_version": 2,
        "report_assembly_wall_time_s": assembly_wall_times,
        "sha256": hashes,
    })
    return tuple(report_paths)


def main(argv=None) -> int:
    report_paths = run(argv)
    for report_path in report_paths:
        print(report_path)
        if report_path.name == "serving_report.md" and (report_path.parent / "state.json").is_file():
            state = json.loads((report_path.parent / "state.json").read_text())
            serving = state["serving"]
            result = json.loads(Path(serving["result_path"]).read_text())
            if "workload_path" in result and result["workload_path"] == "virtual_tensor" and any(cell["prediction_status"] != "completed" for cell in result["cells"]):
                return 1
    return 0
