"""Compare frozen, detailed, and fast Serving execution results."""

from __future__ import annotations

import json
import math
from pathlib import Path


_STATUSES = ("completed", "failed", "unsupported")


def _load(path):
    result_path = Path(path).resolve(strict=True)
    return result_path, json.loads(result_path.read_text())


def _cells_by_id(payload, label):
    if not isinstance(payload, dict) or not isinstance(payload["cells"], list):
        raise ValueError(f"{label} Serving results must contain a cell list")
    result = {}
    for cell in payload["cells"]:
        cell_id = cell["cell_id"]
        if cell_id in result:
            raise ValueError(f"duplicate {label} Serving cell: {cell_id}")
        if cell["prediction_status"] not in _STATUSES:
            raise ValueError(f"unknown {label} prediction status for {cell_id}")
        result[cell_id] = cell
    return result


def _predicted(cell, group, metric, statistic="mean"):
    if group == "throughput":
        value = cell["metrics"][group][metric]["predicted"]
    else:
        value = cell["metrics"][group][metric][statistic]["predicted"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid predicted metric {group}.{metric}.{statistic} for {cell['cell_id']}")
    return float(value)


def _deviation(detailed, fast):
    if detailed == 0.0:
        return 0.0 if fast == 0.0 else None
    return abs(fast / detailed - 1.0) * 100.0


def _passes(deviation):
    return deviation is not None and deviation <= 3.0 + 1e-12


def _rank_state_summary(result):
    memory = result["source_activation_memory"]
    ranks = []
    for pool_name, pool in memory["pools"].items():
        for rank, facts in pool["ranks"].items():
            ranks.append({"pool": pool_name, "rank": rank,
                "known_activation_peak_bytes": facts["known_activation_peak_bytes"]})
    return {"reported_steps": memory["reported_steps"],
        "total_simulation_steps": memory["total_simulation_steps"],
        "retention": memory["retention"], "ranks": ranks,
        "logical_kernel_calls": result["num_kernel_calls"],
        "scope": "per-rank source-memory peaks, not a complete event/byte audit"}


def _optional_state_audit(cell):
    if "result_path" not in cell:
        return None
    path = Path(cell["result_path"]).parent / "state_audit.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _audit_call_count(audit):
    return sum(count for rank in audit["ranks"]
        for count in rank["logical_calls_by_kind"].values())


def _case_comparison(cell_id, baseline, detailed, fast):
    row = {"cell_id": cell_id, "metric_gates_pass": False,
        "status": {"baseline": baseline["prediction_status"],
            "detailed": detailed["prediction_status"], "fast": fast["prediction_status"]}}
    if "result_path" in detailed:
        detailed_result = json.loads(Path(detailed["result_path"]).read_text())
    else:
        detailed_result = None
    if "result_path" in fast:
        fast_result = json.loads(Path(fast["result_path"]).read_text())
    else:
        fast_result = None
    if "result_path" in baseline:
        baseline_result = json.loads(Path(baseline["result_path"]).read_text())
    else:
        baseline_result = None
    for label, cell, result in (("baseline", baseline, baseline_result),
            ("detailed", detailed, detailed_result), ("fast", fast, fast_result)):
        entry = {"prediction_status": cell["prediction_status"]}
        if result is not None:
            progress = result["scheduler_progress"]["summary"]
            entry.update({
                "simulation_wall_time_s": result["simulation_wall_time_s"],
                "phase_wall_time_s": result["evaluation_wall_time_breakdown_s"],
                "request_count": progress["request_count"],
                "completed_requests": progress["completed_requests"],
                "prefill_tokens": progress["prefill_tokens"],
                "decode_tokens": progress["decode_tokens"],
                "scheduled_tokens": progress["scheduled_tokens"],
                "step_count": progress["step_count"],
                "num_kernel_calls": result["num_kernel_calls"],
                "source_activation_memory": result["source_activation_memory"],
                "rank_state_summary": _rank_state_summary(result),
                "execution_artifact": result["execution_artifact"],
            })
            audit = _optional_state_audit(cell)
            if audit is not None:
                entry["state_audit"] = audit
                entry["state_audit_kernel_calls_match"] = (
                    _audit_call_count(audit) == result["num_kernel_calls"])
        if cell["prediction_status"] == "completed":
            entry.update({
                "ttft_ms_mean": _predicted(cell, "latency", "ttft_ms"),
                "tpot_ms_mean": _predicted(cell, "latency", "tpot_ms"),
                "request_throughput_per_s": _predicted(
                    cell, "throughput", "request_throughput_per_s"),
                "output_token_throughput_per_s": _predicted(
                    cell, "throughput", "output_token_throughput_per_s"),
                "ttft_ms_p90": _predicted(cell, "latency", "ttft_ms", "p90"),
                "tpot_ms_p90": _predicted(cell, "latency", "tpot_ms", "p90"),
                "ttft_observed_ape_percent": cell["metrics"]["latency"]["ttft_ms"]["mean"]["absolute_percentage_error"],
                "tpot_observed_ape_percent": cell["metrics"]["latency"]["tpot_ms"]["mean"]["absolute_percentage_error"],
            })
        row[label] = entry
    if detailed["prediction_status"] == fast["prediction_status"] == "completed":
        deviations = {
            "ttft_ms_mean": _deviation(row["detailed"]["ttft_ms_mean"], row["fast"]["ttft_ms_mean"]),
            "tpot_ms_mean": _deviation(row["detailed"]["tpot_ms_mean"], row["fast"]["tpot_ms_mean"]),
            "request_throughput_per_s": _deviation(
                row["detailed"]["request_throughput_per_s"], row["fast"]["request_throughput_per_s"]),
        }
        row["fast_vs_detailed_percent"] = deviations
        row["metric_gates_pass"] = all(_passes(value) for value in deviations.values())
        row["conservation"] = {
            "requests_equal": row["detailed"]["completed_requests"] == row["fast"]["completed_requests"],
            "prefill_tokens_equal": row["detailed"]["prefill_tokens"] == row["fast"]["prefill_tokens"],
            "decode_tokens_equal": row["detailed"]["decode_tokens"] == row["fast"]["decode_tokens"],
            "scheduled_tokens_equal": row["detailed"]["scheduled_tokens"] == row["fast"]["scheduled_tokens"],
            "kernel_calls_equal": row["detailed"]["num_kernel_calls"] == row["fast"]["num_kernel_calls"],
            "state_audit_available": "state_audit" in row["detailed"] and "state_audit" in row["fast"],
            "state_audit_equal": (
                "state_audit" in row["detailed"] and "state_audit" in row["fast"]
                and row["detailed"]["state_audit"] == row["fast"]["state_audit"]),
            "state_audit_kernel_calls_match": (
                "state_audit_kernel_calls_match" in row["detailed"]
                and "state_audit_kernel_calls_match" in row["fast"]
                and row["detailed"]["state_audit_kernel_calls_match"]
                and row["fast"]["state_audit_kernel_calls_match"]),
            "source_activation_memory_equal": (
                row["detailed"]["source_activation_memory"] == row["fast"]["source_activation_memory"]),
            "execution_artifact_sha256_equal": (
                row["detailed"]["execution_artifact"]["artifact_sha256"]
                == row["fast"]["execution_artifact"]["artifact_sha256"]),
        }
        if baseline["prediction_status"] == "completed":
            row["baseline_over_fast_simulation_speedup"] = (
                row["baseline"]["simulation_wall_time_s"] / row["fast"]["simulation_wall_time_s"])
    if baseline["prediction_status"] == detailed["prediction_status"] == "completed":
        row["baseline_vs_detailed_percent"] = {
            "ttft_ms_mean": _deviation(row["baseline"]["ttft_ms_mean"], row["detailed"]["ttft_ms_mean"]),
            "tpot_ms_mean": _deviation(row["baseline"]["tpot_ms_mean"], row["detailed"]["tpot_ms_mean"]),
            "request_throughput_per_s": _deviation(
                row["baseline"]["request_throughput_per_s"], row["detailed"]["request_throughput_per_s"]),
        }
        row["baseline_detailed_conservation"] = {
            "requests_equal": row["baseline"]["completed_requests"] == row["detailed"]["completed_requests"],
            "scheduled_tokens_equal": row["baseline"]["scheduled_tokens"] == row["detailed"]["scheduled_tokens"],
            "kernel_calls_equal": row["baseline"]["num_kernel_calls"] == row["detailed"]["num_kernel_calls"],
            "source_activation_memory_equal": (
                row["baseline"]["source_activation_memory"] == row["detailed"]["source_activation_memory"]),
            "execution_artifact_sha256_equal": (
                row["baseline"]["execution_artifact"]["artifact_sha256"]
                == row["detailed"]["execution_artifact"]["artifact_sha256"]),
        }
    return row


def compare_result_payloads(baseline_payload, detailed_payload, fast_payload):
    baseline = _cells_by_id(baseline_payload, "baseline")
    detailed = _cells_by_id(detailed_payload, "detailed")
    fast = _cells_by_id(fast_payload, "fast")
    if set(baseline) != set(detailed) or set(detailed) != set(fast):
        raise ValueError("baseline, detailed, and fast Serving case sets differ")
    case_ids = tuple(baseline)
    for cell_id in case_ids:
        for key in ("captured_prediction_contract", "captured_runtime", "request_count", "requests_sha256"):
            if baseline[cell_id][key] != detailed[cell_id][key] or detailed[cell_id][key] != fast[cell_id][key]:
                raise ValueError(f"baseline/detailed/fast input mismatch for {cell_id}: {key}")
    cases = [_case_comparison(cell_id, baseline[cell_id], detailed[cell_id], fast[cell_id])
        for cell_id in case_ids]
    statuses = {label: {status: sum(cell[label]["prediction_status"] == status for cell in cases)
        for status in _STATUSES} for label in ("baseline", "detailed", "fast")}
    batch = {}
    for label, payload in (("baseline", baseline_payload), ("detailed", detailed_payload), ("fast", fast_payload)):
        phase_sums = {}
        wall_sum = 0.0
        for case in cases:
            entry = case[label]
            if "phase_wall_time_s" not in entry:
                continue
            wall_sum += entry["simulation_wall_time_s"]
            for name, value in entry["phase_wall_time_s"].items():
                phase_sums[name] = phase_sums[name] + value if name in phase_sums else value
        batch[label] = {
            "evaluation_wall_time_s": payload["evaluation_wall_time_seconds"],
            "per_case_simulation_wall_sum_s": wall_sum,
            "per_case_phase_sum_s": phase_sums,
            "per_case_sums_note": "diagnostic sums only; not the concurrent batch wall-time boundary",
            "simulation_batch_timing": (payload["simulation_batch_timing"]
                if "simulation_batch_timing" in payload else {
                    "availability": "legacy_absolute_case_boundaries_unavailable"}),
        }
    detailed_wall = detailed_payload["simulation_batch_timing"]
    fast_wall = fast_payload["simulation_batch_timing"]
    if detailed_wall["availability"] == fast_wall["availability"] == "available":
        speedup = detailed_wall["wall_time_s"] / fast_wall["wall_time_s"]
        case_count_matches = (detailed_wall["case_count"] == fast_wall["case_count"] == len(cases)
            and detailed_payload["cell_count"] == fast_payload["cell_count"] == len(cases))
        batch["detailed_over_fast_simulation_speedup"] = speedup
        batch["simulation_case_count_matches"] = case_count_matches
        batch["simulation_speed_gate_pass"] = case_count_matches and speedup >= 10.0
    else:
        batch["detailed_over_fast_simulation_speedup"] = None
        batch["simulation_speed_gate_pass"] = False
    metric_gate = all(case["metric_gates_pass"] for case in cases)
    completed_gate = all(status == "completed" for cell in cases for status in cell["status"].values())
    return {
        "schema_version": "serving_execution_mode_comparison_v1",
        "case_count": len(cases),
        "case_ids": list(case_ids),
        "status_counts": statuses,
        "batch": batch,
        "gates": {
            "S1_comparable_inputs": True,
            "S2_simulation_speed_at_least_10x": batch["simulation_speed_gate_pass"],
            "S3_every_case_metric_within_3_percent": metric_gate,
            "S5_all_cases_completed": completed_gate,
        },
        "goal_complete": False,
        "cases": cases,
    }


def _number(value):
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _field(mapping, name):
    return mapping[name] if name in mapping else None


def _metric(row, name, unit=""):
    baseline = _field(row["baseline"], name)
    detailed = _field(row["detailed"], name)
    fast = _field(row["fast"], name)
    deviations = _field(row, "fast_vs_detailed_percent")
    deviation = None if deviations is None else _field(deviations, name)
    suffix = "" if deviation is None else f" ({deviation:.3f}%)"
    return f"{_number(baseline)} / {_number(detailed)} / {_number(fast)}{unit}{suffix}"


def _batch_wall(batch, label):
    return _field(batch[label]["simulation_batch_timing"], "wall_time_s")


def write_markdown_report(comparison, output, *, commands=None, environment=None):
    speedup = _field(comparison["batch"], "detailed_over_fast_simulation_speedup")
    lines = [
        "# Serving detailed/fast execution comparison",
        "",
        "Columns within each metric are **baseline / detailed / fast**. Percentages in parentheses are `abs(fast / detailed - 1) × 100%`; lower than 3% passes S3. Wall times are per-case CPU simulation times, not predicted device latency and not report-generation time.",
        "",
        f"- Cases: {comparison['case_count']}; status counts: `{json.dumps(comparison['status_counts'], sort_keys=True)}`",
        f"- S2 pure simulation: baseline n/a / detailed {_number(_batch_wall(comparison['batch'], 'detailed'))} s / fast {_number(_batch_wall(comparison['batch'], 'fast'))} s; detailed/fast speedup **{_number(speedup)}×** (10× required)",
        f"- S3 every-case mean TTFT/TPOT/request-throughput within 3%: **{'Pass' if comparison['gates']['S3_every_case_metric_within_3_percent'] else 'Fail'}**",
        f"- S5 all cases completed: **{'Pass' if comparison['gates']['S5_all_cases_completed'] else 'Fail'}**",
        "- Artifact/source-memory equality is reported as evidence, not as a substitute for the separate per-rank state audit.",
        "",
        "## Batch timing and phase sums",
        "",
        "Per-case sums below are diagnostic totals, not the concurrent batch wall-time boundary.",
        "",
        "| Mode | Outer evaluation s | Pure simulation batch s | Per-case wall sum s |",
        "|---|---:|---:|---:|",
    ]
    for label in ("baseline", "detailed", "fast"):
        lines.append(f"| {label} | {_number(comparison['batch'][label]['evaluation_wall_time_s'])} | "
            f"{_number(_batch_wall(comparison['batch'], label))} | "
            f"{_number(comparison['batch'][label]['per_case_simulation_wall_sum_s'])} |")
    lines.extend(["", "| Phase | Baseline sum s | Detailed sum s | Fast sum s |", "|---|---:|---:|---:|"])
    for phase in ("workload_construction", "kernel_lowering", "kernel_prediction",
            "timeline_construction", "artifact_generation", "scheduler_and_orchestration"):
        lines.append(f"| {phase} | {_number(_field(comparison['batch']['baseline']['per_case_phase_sum_s'], phase))} | "
            f"{_number(_field(comparison['batch']['detailed']['per_case_phase_sum_s'], phase))} | "
            f"{_number(_field(comparison['batch']['fast']['per_case_phase_sum_s'], phase))} |")
    lines.append("")
    if environment is not None:
        lines.extend(["## Environment", "", "```json", json.dumps(environment, indent=2, sort_keys=True), "```", ""])
    if commands is not None:
        lines.append("## Commands")
        lines.append("")
        for label, command in commands.items():
            lines.extend([f"- {label}: `{command}`"])
        lines.append("")
    if "supplement" in comparison:
        supplement = comparison["supplement"]
        lines.extend(["## Acceptance supplement", "",
            f"- S4/S6/S7 gates: `{json.dumps(supplement['gates'], sort_keys=True)}`",
            f"- Overall goal complete: **{comparison['goal_complete']}**", ""])
        if "gate_evidence" in supplement:
            lines.extend(["| Gate | Actual | Evidence |", "|---|---|---|"])
            for gate, evidence in supplement["gate_evidence"].items():
                lines.append(f"| {gate} | {evidence['actual']} | {evidence['evidence']} |")
            lines.append("")
        if "optimizations" in supplement:
            lines.extend(["| Optimization | Work avoided | Upper bound | Actual/scope |", "|---|---|---|---|"])
            for optimization in supplement["optimizations"]:
                lines.append(f"| {optimization['name']} | {optimization['work_avoided']} | "
                    f"{optimization['upper_bound']} | {optimization['actual_and_scope']} |")
            lines.append("")
    lines.extend([
        "## Per-case results",
        "",
        "| Case | Baseline sim s | Detailed sim s | Fast sim s | Baseline/fast | TTFT ms B/D/F | TPOT ms B/D/F | Req/s B/D/F | Requests/tokens D/F | Memory/artifact/audit equal | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for case in comparison["cases"]:
        baseline = case["baseline"]
        detailed = case["detailed"]
        fast = case["fast"]
        speedup = _field(case, "baseline_over_fast_simulation_speedup")
        requests = f"{_field(detailed, 'completed_requests')}/{_field(fast, 'completed_requests')}"
        tokens = f"{_field(detailed, 'scheduled_tokens')}/{_field(fast, 'scheduled_tokens')}"
        conservation = _field(case, "conservation")
        if conservation is None:
            conservation = {}
        state = (f"memory={_field(conservation, 'source_activation_memory_equal')}; "
            f"artifact={_field(conservation, 'execution_artifact_sha256_equal')}; "
            f"audit={_field(conservation, 'state_audit_equal')}; "
            f"calls={_field(conservation, 'state_audit_kernel_calls_match')}")
        status = "/".join(case["status"].values())
        lines.append(
            f"| {case['cell_id']} | {_number(_field(baseline, 'simulation_wall_time_s'))} | "
            f"{_number(_field(detailed, 'simulation_wall_time_s'))} | {_number(_field(fast, 'simulation_wall_time_s'))} | "
            f"{_number(speedup)} | {_metric(case, 'ttft_ms_mean')} | {_metric(case, 'tpot_ms_mean')} | "
            f"{_metric(case, 'request_throughput_per_s')} | {requests}; {tokens} | {state} | {status} |")
    lines.extend([
        "",
        "## Detailed mode versus frozen baseline",
        "",
        "These differences are diagnostic and are not the fast/detail S3 gate. Values are baseline / detailed with the absolute percentage difference in parentheses.",
        "",
        "| Case | TTFT ms | TPOT ms | Req/s | Requests/tokens equal | Memory/artifact equal |",
        "|---|---:|---:|---:|---|---|",
    ])
    for case in comparison["cases"]:
        deviations = _field(case, "baseline_vs_detailed_percent")
        if deviations is None:
            continue
        baseline = case["baseline"]
        detailed = case["detailed"]
        conservation = case["baseline_detailed_conservation"]
        lines.append(
            f"| {case['cell_id']} | "
            f"{_number(baseline['ttft_ms_mean'])} / {_number(detailed['ttft_ms_mean'])} ({deviations['ttft_ms_mean']:.3f}%) | "
            f"{_number(baseline['tpot_ms_mean'])} / {_number(detailed['tpot_ms_mean'])} ({deviations['tpot_ms_mean']:.3f}%) | "
            f"{_number(baseline['request_throughput_per_s'])} / {_number(detailed['request_throughput_per_s'])} ({deviations['request_throughput_per_s']:.3f}%) | "
            f"{conservation['requests_equal']}; {conservation['scheduled_tokens_equal']} | "
            f"{conservation['source_activation_memory_equal']}; {conservation['execution_artifact_sha256_equal']} |")
    lines.extend(["", "Goal completion is not inferred from this report alone: S4, S6, and S7 require their own evidence.", ""])
    Path(output).write_text("\n".join(lines))


def compare_result_files(baseline_path, detailed_path, fast_path, output_dir, *, commands=None, environment=None, supplement=None):
    baseline_path, baseline = _load(baseline_path)
    detailed_path, detailed = _load(detailed_path)
    fast_path, fast = _load(fast_path)
    comparison = compare_result_payloads(baseline, detailed, fast)
    comparison["inputs"] = {"baseline": str(baseline_path), "detailed": str(detailed_path), "fast": str(fast_path)}
    if commands is not None:
        comparison["commands"] = dict(commands)
    if environment is not None:
        comparison["environment"] = environment
    if supplement is not None:
        if (not isinstance(supplement, dict) or not isinstance(supplement["gates"], dict)
                or set(supplement["gates"]) != {"S4_key_state", "S6_general_approximation", "S7_reusable_delivery"}
                or any(type(value) is not bool for value in supplement["gates"].values())):
            raise ValueError("comparison supplement must provide boolean S4, S6, and S7 gates")
        comparison["supplement"] = supplement
        comparison["gates"].update(supplement["gates"])
        comparison["goal_complete"] = all(comparison["gates"].values())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "serving_speed_report.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    write_markdown_report(comparison, markdown_path, commands=commands, environment=environment)
    return json_path, markdown_path, comparison
