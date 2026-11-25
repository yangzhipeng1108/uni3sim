"""Report-only assembly for single-kernel Forward evaluation results.

Rebuilds the standard per-bucket accuracy table and the cross-hardware
latency scatter figures without re-running any prediction: inputs are saved
semantic evaluation artifacts produced by ``run_kernel_evaluation.py
--semantic-output``.  The saved forward / roofline pairs and report metadata
drive ``define_subsets`` / ``compute_metrics`` exactly like the live
evaluator, so numbers match the original run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from evaluation.kernel.evaluation_runner import compute_metrics, define_subsets


REPORT_HARDWARE = ("nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b")


def load_semantic_hardware(
        semantic_json: dict[str, Path | None]) -> dict[str, dict[str, object]]:
    """Load saved semantic artifacts and rebuild per-hardware pair/meta maps."""
    hardware = {}
    for hardware_id in REPORT_HARDWARE:
        if hardware_id not in semantic_json:
            continue
        path = semantic_json[hardware_id]
        if path is None:
            continue
        if not path.is_file():
            raise ValueError(
                f"{hardware_id} semantic artifact does not exist: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "scope" not in payload:
            raise ValueError(
                f"{path} is not a kernel semantic result artifact")
        if payload["scope"] != "kernel_evaluation_semantic_result":
            raise ValueError(
                f"{path} is not a kernel semantic result artifact")
        op_meta = {}
        op_pairs_forward = {}
        op_pairs_roofline = {}
        for case in payload["cases"]:
            task_key = case["case_id"]
            op_meta[task_key] = case["report_meta"]
            forward = case["forward"]
            roofline = case["roofline"]
            op_pairs_forward[task_key] = (
                tuple(forward) if isinstance(forward, list) else forward)
            op_pairs_roofline[task_key] = (
                tuple(roofline) if isinstance(roofline, list) else roofline)
        hardware[hardware_id] = {
            "mode": payload["mode"],
            "jobs": payload["jobs"],
            "op_meta": op_meta,
            "op_pairs_forward": op_pairs_forward,
            "op_pairs_roofline": op_pairs_roofline,
        }
    if not hardware:
        raise ValueError("no usable semantic artifacts were provided")
    return hardware


def build_kernel_matrix_rows(
        hardware: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    """Rebuild the per-card bucket table rows the standard report consumes."""
    matrix = {}
    for hardware_id, entry in hardware.items():
        op_meta = entry["op_meta"]
        op_pairs_forward = entry["op_pairs_forward"]
        op_pairs_roofline = entry["op_pairs_roofline"]
        subsets = define_subsets(op_meta)
        rows = []
        scored_rows = []
        for row_index, (label, keys, _indent, _is_last) in enumerate(subsets):
            if not keys:
                continue
            forward_metrics = compute_metrics(
                [op_pairs_forward[key] for key in keys],
                exclude_unsupported=True)
            roofline_metrics = compute_metrics(
                [op_pairs_roofline[key] for key in keys],
                exclude_unsupported=True)
            available = [m for m in (forward_metrics, roofline_metrics)
                         if m is not None]
            if not available:
                continue
            counts = {metrics["N"] for metrics in available}
            if len(counts) != 1:
                raise ValueError(
                    f"{hardware_id} {label!r} methods disagree on sample count")
            sample_count = available[0]["N"]

            def _format(metrics):
                if metrics is None:
                    return None
                if metrics["status"] == "unsupported":
                    return None
                if metrics["status"] != "ok":
                    return None
                return [metrics["MAPE"], metrics["MedAPE"], metrics["WMAE"]]

            rows.append([
                label, sample_count, None,
                _format(forward_metrics), _format(roofline_metrics), None,
            ])
            if (forward_metrics is not None
                    and forward_metrics["status"] == "ok"
                    and forward_metrics["unsupported"]):
                scored_rows.append([
                    row_index, forward_metrics["scored"], forward_metrics["N"]])
        forward_coverage = None
        if scored_rows:
            forward_coverage = {"scored": scored_rows, "unsupported_cases": {}}
        matrix[hardware_id] = {
            "status": "PASS",
            "summary": (
                f"report-only rebuild from saved semantic results "
                f"({len(op_meta)} cases)"),
            "rows": rows,
            "forward_coverage": forward_coverage,
        }
    return matrix


def collect_saved_latency_points(
        hardware: dict[str, dict[str, object]],
) -> dict[str, list[tuple[float, float, int]]]:
    """Build scatter points straight from saved forward pairs (no prediction)."""
    from evaluation.kernel.latency_scatter_report import _family_index

    points_by_hardware = {}
    for hardware_id, entry in hardware.items():
        points = []
        for task_key, pair in entry["op_pairs_forward"].items():
            if pair == "ERR":
                continue
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    f"{hardware_id} task {task_key!r} has no Forward result")
            predicted_ns, actual_ns = pair
            op_type = entry["op_meta"][task_key]["op_type"]
            points.append((
                actual_ns / 1000.0,
                predicted_ns / 1000.0,
                _family_index(op_type),
            ))
        points_by_hardware[hardware_id] = points
    return points_by_hardware


def write_kernel_matrix_report(
        semantic_json: dict[str, Path | None],
        matrix_output: Path,
        overall_output: Path | None,
        detailed_output: Path | None,
        wall_time_output: Path | None = None,
) -> dict[str, object]:
    """Rebuild the kernel matrix JSON and optional latency scatter figures."""
    started = time.perf_counter()
    hardware = load_semantic_hardware(semantic_json)
    matrix = build_kernel_matrix_rows(hardware)
    matrix_path = Path(matrix_output)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps({
        "scope": "kernel_matrix_report_only",
        "source": "saved semantic evaluation artifacts; no prediction rerun",
        "kernel_matrix": matrix,
    }, indent=2) + "\n")

    resolved_detailed = detailed_output
    plot_wall_s = None
    if overall_output is not None:
        from evaluation.kernel.latency_scatter_report import (
            render_compact_latency_report,
        )
        points = collect_saved_latency_points(hardware)
        if resolved_detailed is None:
            resolved_detailed = Path(overall_output).with_name(
                f"{Path(overall_output).stem}-by-bucket{Path(overall_output).suffix}")
        plot_started = time.perf_counter()
        render_compact_latency_report(
            points, Path(overall_output), Path(resolved_detailed))
        plot_wall_s = time.perf_counter() - plot_started

    wall_record = {
        "matrix_wall_time_s": time.perf_counter() - started,
        "plot_wall_time_s": plot_wall_s,
        "cases": {hw: len(entry["op_meta"]) for hw, entry in hardware.items()},
    }
    if wall_time_output is not None:
        wall_path = Path(wall_time_output)
        wall_path.parent.mkdir(parents=True, exist_ok=True)
        wall_path.write_text(json.dumps(wall_record, indent=2) + "\n")
    return {
        "matrix_path": str(matrix_path),
        "overall_path": (str(overall_output) if overall_output is not None
                         else None),
        "detailed_path": (str(resolved_detailed)
                          if resolved_detailed is not None else None),
        "wall_record": wall_record,
    }
