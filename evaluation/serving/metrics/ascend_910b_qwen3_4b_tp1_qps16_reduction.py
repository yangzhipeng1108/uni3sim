#!/usr/bin/env python3
"""Reproduce the audited 910B Qwen3-4B QPS16 timing reductions.

This script reads, but does not modify, the historical canonical msprof capture
and the later no-profiler model-Event capture.  Every input is explicit and is
rejected unless its SHA256 matches the audited source identity.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.json_shape import require_integer as _integer
from evaluation.json_shape import file_sha256 as _sha256
from evaluation.json_shape import require_interval_union_size


EXPECTED_SHA256 = {
    "alignment": "b1a85e0bae0c0afa19554130d374ac3a8db55c5041cb546fcb4d2cc754959d7d",
    "device_events": "3ced41a6b4d0d5793e5a5733ccd403b624b61b4d7093cd96b974dbe31c21c1f7",
    "engine_steps": "07734b8ff40dd276f5862bba79ef040483634620ee609654f20f3c5d427ff2e8",
    "msprof_db": "ddc1cc458e7811d864a706d9364bedc175506d0f19906670803a64998737bba4",
    "scheduler_sidecar": "0484d326dd11d5e95a64bef32cb72ea9d72c6c29f3befe7a46c2b6153986a053",
}
FULL_GRAPH_EXPECTED_SHA256 = {
    "full_graph_device_events": "81bafd35eba4334befa5934d8ccc4b0de37d6d4cf95265b6ec7a6210d1139247",
    "full_graph_engine_steps": "2be7de4d4c75321f04b173c0fd1f15139c0a3ef82e367038b62241b1dc269de3",
    "full_graph_ground_truth": "dbaf733c1574917f26263b3522fc70fb64af5d6fc32f973b874355191420ac0d",
}
PAIRED_GRAPH_EXPECTED_SHA256 = {
    "paired_full_decode_only_device_events": "d8e8f464e44320f9947e90c3bdcbf53f878834d642efbd3c154d05077bf8ad26",
    "paired_full_decode_only_ground_truth": "46473fb197b8514e09e54a219792342d1491eeeb7f5cf45caa2cc3505c6a2b64",
    "paired_full_graph_device_events": "578f2466b41440cdf4392fcd4a14a332cc312920796115be4dff5280ecfb80e7",
    "paired_full_graph_ground_truth": "797737d720e4916704f8bcf6d7ab806fd3a6ff03ec440f998f17325569b6ff21",
    "paired_full_graph_512_device_events": "d79f5d9770951dfc52e139e58b6098e057dcda87d072ac1a9758bed8e1a9fede",
    "paired_full_graph_512_ground_truth": "580262540c044ede01308e5b384a85d223f4dd060c96971c4d7590464fe537ba",
}
DYNAMIC_PROFILER_EXPECTED_SHA256 = {
    "dynamic_profiler_device_events": "ab8ffc5f914b6ed2ed4b9c88dab8643450bf80de3c86066cb4d3de31eac77fb9",
    "dynamic_profiler_ground_truth": "b0137fe4aeb33cd5b132c5f0af74d06f480f99fcde8b613649a0ff6d4170ea63",
    "dynamic_profiler_msprof_db": "d938a9efb3f2a4f12232306fbdd4ab89d71e23f7c8df69b52238cfa7a59fd2cb",
}
WARMUP_MODEL_CALL_COUNT = 8
EVENT_RECORD_TASK_TYPE = 17
GRAPH_NOTIFY_WAIT_TASK_TYPE = 114
TERMINAL_KERNEL = "aclnnEmbedding_GatherV2AiCore_GatherV2"
PHASES = ("prefill", "mixed", "decode")
EVENT_PHASES = ("model_and_logits_eager", "model_and_logits_graph")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msprof-db", required=True, type=Path)
    parser.add_argument("--scheduler-sidecar", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--device-events", required=True, type=Path)
    parser.add_argument("--engine-steps", required=True, type=Path)
    parser.add_argument("--full-graph-device-events", required=True, type=Path)
    parser.add_argument("--full-graph-engine-steps", required=True, type=Path)
    parser.add_argument("--full-graph-ground-truth", required=True, type=Path)
    parser.add_argument(
        "--paired-full-decode-only-device-events", required=True, type=Path
    )
    parser.add_argument(
        "--paired-full-decode-only-ground-truth", required=True, type=Path
    )
    parser.add_argument("--paired-full-graph-device-events", required=True, type=Path)
    parser.add_argument("--paired-full-graph-ground-truth", required=True, type=Path)
    parser.add_argument(
        "--paired-full-graph-512-device-events", required=True, type=Path
    )
    parser.add_argument(
        "--paired-full-graph-512-ground-truth", required=True, type=Path
    )
    parser.add_argument("--dynamic-profiler-device-events", required=True, type=Path)
    parser.add_argument("--dynamic-profiler-ground-truth", required=True, type=Path)
    parser.add_argument("--dynamic-profiler-msprof-db", required=True, type=Path)
    return parser.parse_args()




def _verify_sources(
    paths: dict[str, Path], expected_sha256: dict[str, str]
) -> dict[str, str]:
    if set(paths) != set(expected_sha256):
        raise ValueError("source paths do not cover the exact audited input set")
    observed = {}
    for name, expected in expected_sha256.items():
        path = paths[name]
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{name} SHA256 mismatch for {path}: expected {expected}, got {actual}"
            )
        observed[name] = actual
    return observed


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records




def _interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return require_interval_union_size(intervals, "TASK interval")


def _scheduled_tokens(record: dict[str, object], label: str) -> int:
    if "scheduled" not in record or not isinstance(record["scheduled"], dict):
        raise TypeError(f"{label}.scheduled must be an object")
    values = record["scheduled"].values()
    tokens = [_integer(value, f"{label}.scheduled token count") for value in values]
    return sum(tokens)


def _sample_orders(alignment_path: Path) -> dict[str, int]:
    payload = _load_json(alignment_path)
    if not isinstance(payload, dict) or payload["kind"] != "sharegpt_alignment_steps":
        raise ValueError("alignment is not sharegpt_alignment_steps")
    phases = payload["phases"]
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ValueError("alignment must contain exactly prefill, mixed, and decode")
    orders = {}
    for phase in PHASES:
        phase_payload = phases[phase]
        if not isinstance(phase_payload, dict) or phase_payload["status"] != "available":
            raise ValueError(f"alignment phase {phase} is unavailable")
        sample = phase_payload["sample"]
        if not isinstance(sample, dict):
            raise TypeError(f"alignment phase {phase}.sample must be an object")
        order = _integer(sample["source_execution_order"], f"{phase} source order")
        expected_step_id = f"step-{order:012d}"
        if sample["source_step_id"] != expected_step_id:
            raise ValueError(
                f"{phase} source step mismatch: {sample['source_step_id']!r} "
                f"!= {expected_step_id!r}"
            )
        launches = sample["launches"]
        if not isinstance(launches, list):
            raise TypeError(f"alignment phase {phase}.launches must be a list")
        rank_zero = [entry for entry in launches if entry["rank"] == 0]
        if len(rank_zero) != 1:
            raise ValueError(f"alignment phase {phase} must contain exactly rank 0")
        orders[phase] = order
    return orders


def _historical_task_reduction(
    database: Path,
    scheduler_path: Path,
    alignment_path: Path,
) -> dict[str, object]:
    scheduler = _load_jsonl(scheduler_path)
    orders = _sample_orders(alignment_path)
    by_iter = {}
    production_work = []
    for row_number, record in enumerate(scheduler, start=1):
        iter_id = _integer(record["iter_id"], f"scheduler row {row_number} iter_id")
        if iter_id in by_iter:
            raise ValueError(f"duplicate Scheduler iter_id {iter_id}")
        by_iter[iter_id] = record
        tokens = _scheduled_tokens(record, f"scheduler row {row_number}")
        request_ids = tuple(record["scheduled"])
        if tokens and not all(request_id.startswith("warmup_") for request_id in request_ids):
            if any(request_id.startswith("warmup_") for request_id in request_ids):
                raise ValueError("Scheduler row mixes warmup and production requests")
            production_work.append(record)
    if not production_work:
        raise ValueError("Scheduler contains no production work")
    production_index = {
        _integer(record["iter_id"], "production iter_id"): index
        for index, record in enumerate(production_work)
    }
    first_production_ts_ns = (
        _integer(production_work[0]["ts_us"], "first production ts_us") * 1000
    )

    uri = f"file:{database}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"TASK", "COMPUTE_TASK_INFO", "STRING_IDS"}
        if not required.issubset(tables):
            raise ValueError(f"msprof database lacks tables {sorted(required - tables)}")
        terminals = connection.execute(
            "SELECT t.startNs, t.endNs FROM TASK t "
            "JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
            "JOIN STRING_IDS s ON c.name=s.id WHERE s.value=? "
            "ORDER BY t.startNs, t.globalTaskId",
            (TERMINAL_KERNEL,),
        ).fetchall()
        anchors = [terminal for terminal in terminals if terminal[1] < first_production_ts_ns]
        if not anchors:
            raise ValueError("no completed forward terminal precedes production work")
        anchor = anchors[-1]
        work_terminals = [terminal for terminal in terminals if terminal[0] > anchor[1]]

        samples = {}
        for phase in PHASES:
            order = orders[phase]
            if order not in by_iter or order not in production_index:
                raise ValueError(f"{phase} source order {order} is not production work")
            terminal_index = production_index[order]
            if terminal_index >= len(work_terminals):
                raise ValueError(f"msprof capture does not reach {phase} source order {order}")
            lower_ns = anchor[1] if terminal_index == 0 else work_terminals[terminal_index - 1][1]
            upper_ns = work_terminals[terminal_index][1]
            all_tasks = connection.execute(
                "SELECT startNs, endNs FROM TASK "
                "WHERE startNs>? AND endNs<=? ORDER BY startNs, globalTaskId",
                (lower_ns, upper_ns),
            ).fetchall()
            compute_task_count = connection.execute(
                "SELECT COUNT(*) FROM TASK t JOIN COMPUTE_TASK_INFO c "
                "ON t.globalTaskId=c.globalTaskId "
                "WHERE t.startNs>? AND t.endNs<=?",
                (lower_ns, upper_ns),
            ).fetchone()[0]
            samples[phase] = {
                "source_step_id": f"step-{order:012d}",
                "scheduled_tokens": _scheduled_tokens(by_iter[order], f"iter {order}"),
                "window_ns": [lower_ns, upper_ns],
                "all_task_count": len(all_tasks),
                "compute_task_count": compute_task_count,
                "all_task_union_ms": _interval_union_ns(all_tasks) / 1_000_000,
            }
    return {
        "terminal_kernel": TERMINAL_KERNEL,
        "interval_rule": "union_all_TASK_intervals_without_double_counting_overlap",
        "samples": samples,
    }


def _pearson(values: Sequence[tuple[int, float]]) -> float:
    if len(values) < 2:
        raise ValueError("Pearson correlation needs at least two observations")
    x_mean = sum(value[0] for value in values) / len(values)
    y_mean = sum(value[1] for value in values) / len(values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in values)
    x_square = sum((x - x_mean) ** 2 for x, _ in values)
    y_square = sum((y - y_mean) ** 2 for _, y in values)
    denominator = math.sqrt(x_square * y_square)
    if denominator == 0.0:
        raise ValueError("Pearson correlation has zero variance")
    return numerator / denominator


def _production_event_pairs(
    event_path: Path,
    engine_path: Path,
) -> tuple[int, list[tuple[dict[str, object], dict[str, object]]]]:
    event_jsonl = _load_jsonl(event_path)
    headers = [
        row
        for row in event_jsonl
        if "record_kind" in row and row["record_kind"] == "header"
    ]
    footers = [
        row
        for row in event_jsonl
        if "record_kind" in row and row["record_kind"] == "footer"
    ]
    events = [row for row in event_jsonl if "sequence_id" in row]
    if len(headers) != 1 or len(footers) != 1:
        raise ValueError("device Event trace must contain one header and one footer")
    if headers[0]["profiler_started"] is not False:
        raise ValueError("device Event trace must be a no-profiler run")
    if footers[0]["complete"] is not True or footers[0]["record_count"] != len(events):
        raise ValueError("device Event footer is incomplete or has the wrong count")
    sequence_ids = [
        _integer(row["sequence_id"], "Event sequence_id") for row in events
    ]
    if sequence_ids != list(range(len(events))):
        raise ValueError("device Event sequence IDs are not contiguous")

    engine_jsonl = _load_jsonl(engine_path)
    metadata = [row for row in engine_jsonl if row["record_type"] == "metadata"]
    steps = [row for row in engine_jsonl if row["record_type"] == "step"]
    if len(metadata) != 1 or metadata[0]["completed_record_count"] != len(steps):
        raise ValueError("EngineCore trace metadata count mismatch")
    work_steps = [
        row
        for row in steps
        if _integer(row["total_scheduled_tokens"], "EngineCore total_scheduled_tokens") > 0
    ]
    if len(work_steps) != len(events):
        raise ValueError(
            f"work-bearing EngineCore steps ({len(work_steps)}) != Events ({len(events)})"
        )
    pairs = list(zip(work_steps, events, strict=True))
    warmup_count = 0
    for step, _ in pairs:
        request_ids = tuple(step["scheduled_tokens_by_request"])
        if request_ids and all("warmup" in request_id for request_id in request_ids):
            warmup_count += 1
            continue
        break
    production = pairs[warmup_count:]
    if not production or any(
        "warmup" in request_id
        for step, _ in production
        for request_id in step["scheduled_tokens_by_request"]
    ):
        raise ValueError("warmup records are not one leading contiguous prefix")
    return warmup_count, production


def _event_reduction(event_path: Path, engine_path: Path) -> dict[str, object]:
    warmup_count, production = _production_event_pairs(event_path, engine_path)

    phases = {}
    for event_phase in EVENT_PHASES:
        observations = [
            (
                _integer(step["total_scheduled_tokens"], "production scheduled tokens"),
                float(event["event_duration_ns"]) / 1_000_000,
                _integer(event["wall_duration_ns"], "Event wall_duration_ns"),
            )
            for step, event in production
            if event["phase"] == event_phase
        ]
        if not observations:
            raise ValueError(f"no production observations for {event_phase}")
        token_event = [(tokens, event_ms) for tokens, event_ms, _ in observations]
        phases[event_phase] = {
            "step_count": len(observations),
            "scheduled_tokens": sum(tokens for tokens, _, _ in observations),
            "scheduled_token_range": [
                min(tokens for tokens, _, _ in observations),
                max(tokens for tokens, _, _ in observations),
            ],
            "event_s": sum(event_ms for _, event_ms, _ in observations) / 1000,
            "host_call_wall_s": sum(wall_ns for _, _, wall_ns in observations) / 1_000_000_000,
            "scheduled_tokens_to_event_ms_pearson_r": _pearson(token_event),
        }
        if event_phase == "model_and_logits_graph":
            batch16 = sorted(event_ms for tokens, event_ms in token_event if tokens == 16)
            if not batch16:
                raise ValueError("Graph trace contains no batch-16 observations")
            phases[event_phase]["batch16_event_ms"] = batch16
            phases[event_phase]["batch16_event_range_ms"] = [batch16[0], batch16[-1]]
    full_graph_candidates = [
        (step, event)
        for step, event in production
        if event["phase"] == "model_and_logits_eager"
        and _integer(
            step["total_scheduled_tokens"],
            "production scheduled tokens",
        ) <= 256
    ]
    return {
        "warmup_model_call_count": warmup_count,
        "production_model_call_count": len(production),
        "phases": phases,
        "full_graph_candidate_at_256_token_cap": {
            "step_count": len(full_graph_candidates),
            "scheduled_tokens": sum(
                _integer(step["total_scheduled_tokens"], "production scheduled tokens")
                for step, _ in full_graph_candidates
            ),
            "measured_eager_event_s": sum(
                float(event["event_duration_ns"])
                for _, event in full_graph_candidates
            ) / 1_000_000_000,
            "interpretation": (
                "Static eligibility in this FULL_DECODE_ONLY trace; compare "
                "with a separately validated FULL-mode canary before claiming "
                "actual Graph coverage or aggregate savings."
            ),
        },
    }


def _step_phase(step: dict[str, object]) -> str:
    prefill_tokens = _integer(step["prefill_token_count"], "prefill token count")
    decode_tokens = _integer(step["decode_token_count"], "decode token count")
    if prefill_tokens and decode_tokens:
        return "mixed"
    if prefill_tokens:
        return "prefill"
    if decode_tokens:
        return "decode"
    raise ValueError("work-bearing step has neither prefill nor decode tokens")


def _event_mode(event: dict[str, object]) -> str:
    phase = event["phase"]
    if phase == "model_and_logits_graph":
        return "graph"
    if phase == "model_and_logits_eager":
        return "eager"
    raise ValueError(f"unsupported Event phase: {phase!r}")


def _step_structure_key(step: dict[str, object]) -> tuple[int, ...]:
    return (
        _integer(step["total_scheduled_tokens"], "total scheduled tokens"),
        _integer(step["request_count"], "request count"),
        _integer(step["prefill_token_count"], "prefill token count"),
        _integer(step["decode_token_count"], "decode token count"),
        _integer(step["prefill_request_count"], "prefill request count"),
        _integer(step["decode_request_count"], "decode request count"),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _selected_event_ms(
    production: Sequence[tuple[dict[str, object], dict[str, object]]],
    *,
    phase: str,
    mode: str,
    maximum_tokens: int | None = None,
    minimum_tokens: int | None = None,
) -> list[tuple[dict[str, object], float]]:
    selected = []
    for step, event in production:
        tokens = _integer(step["total_scheduled_tokens"], "scheduled tokens")
        if _step_phase(step) != phase or _event_mode(event) != mode:
            continue
        if maximum_tokens is not None and tokens > maximum_tokens:
            continue
        if minimum_tokens is not None and tokens < minimum_tokens:
            continue
        selected.append((step, float(event["event_duration_ns"]) / 1_000_000))
    if not selected:
        raise ValueError(f"no Event observations for {phase}/{mode}")
    return selected


def _event_distribution(
    selected: Sequence[tuple[dict[str, object], float]],
) -> dict[str, object]:
    durations = [duration_ms for _, duration_ms in selected]
    return {
        "step_count": len(selected),
        "scheduled_tokens": sum(
            _integer(step["total_scheduled_tokens"], "scheduled tokens")
            for step, _ in selected
        ),
        "event_mean_ms": statistics.fmean(durations),
        "event_p50_ms": statistics.median(durations),
    }


def _exact_structure_comparison(
    baseline: Sequence[tuple[dict[str, object], float]],
    full_graph: Sequence[tuple[dict[str, object], float]],
    *,
    include_matched_calls: bool,
) -> dict[str, object]:
    baseline_by_key: dict[tuple[int, ...], list[float]] = {}
    full_graph_by_key: dict[tuple[int, ...], list[float]] = {}
    for step, duration_ms in baseline:
        key = _step_structure_key(step)
        if key not in baseline_by_key:
            baseline_by_key[key] = []
        baseline_by_key[key].append(duration_ms)
    for step, duration_ms in full_graph:
        key = _step_structure_key(step)
        if key not in full_graph_by_key:
            full_graph_by_key[key] = []
        full_graph_by_key[key].append(duration_ms)
    ratios = []
    matched_calls = []
    for key in sorted(set(baseline_by_key) & set(full_graph_by_key)):
        baseline_values = sorted(baseline_by_key[key])
        full_graph_values = sorted(full_graph_by_key[key])
        for baseline_value, full_value in zip(
            baseline_values, full_graph_values, strict=False
        ):
            ratio = full_value / baseline_value
            ratios.append(ratio)
            if include_matched_calls:
                matched_calls.append(
                    {
                        "total_scheduled_tokens": key[0],
                        "request_count": key[1],
                        "prefill_token_count": key[2],
                        "decode_token_count": key[3],
                        "prefill_request_count": key[4],
                        "decode_request_count": key[5],
                        "full_decode_only_event_ms": baseline_value,
                        "full_event_ms": full_value,
                        "full_over_decode_only_event_ratio": ratio,
                        "reduction_percent": (1.0 - ratio) * 100.0,
                    }
                )
    if not ratios:
        raise ValueError("no exact step-structure matches across Graph modes")
    result = {
        "matching_fields": [
            "total_scheduled_tokens",
            "request_count",
            "prefill_token_count",
            "decode_token_count",
            "prefill_request_count",
            "decode_request_count",
        ],
        "matched_call_count": len(ratios),
        "full_over_decode_only_event_ratio_p10": _percentile(ratios, 10.0),
        "full_over_decode_only_event_ratio_p50": statistics.median(ratios),
        "full_over_decode_only_event_ratio_p90": _percentile(ratios, 90.0),
        "minimum_improvement_percent": (1.0 - max(ratios)) * 100.0,
        "maximum_improvement_percent": (1.0 - min(ratios)) * 100.0,
    }
    if include_matched_calls:
        result["matched_calls"] = matched_calls
    return result


def _full_graph_comparison(
    baseline_event_path: Path,
    baseline_engine_path: Path,
    full_graph_event_path: Path,
    full_graph_engine_path: Path,
    full_graph_ground_truth_path: Path,
) -> dict[str, object]:
    _, baseline = _production_event_pairs(
        baseline_event_path, baseline_engine_path
    )
    _, full_graph = _production_event_pairs(
        full_graph_event_path, full_graph_engine_path
    )
    comparisons = {}
    selections = (
        (
            "mixed_at_or_below_256_eager_to_graph",
            _selected_event_ms(
                baseline, phase="mixed", mode="eager", maximum_tokens=256
            ),
            _selected_event_ms(
                full_graph, phase="mixed", mode="graph", maximum_tokens=256
            ),
        ),
        (
            "mixed_above_256_eager_control",
            _selected_event_ms(
                baseline, phase="mixed", mode="eager", minimum_tokens=257
            ),
            _selected_event_ms(
                full_graph, phase="mixed", mode="eager", minimum_tokens=257
            ),
        ),
        (
            "decode_graph_control",
            _selected_event_ms(baseline, phase="decode", mode="graph"),
            _selected_event_ms(full_graph, phase="decode", mode="graph"),
        ),
    )
    for name, baseline_selected, full_graph_selected in selections:
        comparisons[name] = {
            "full_decode_only": _event_distribution(baseline_selected),
            "full": _event_distribution(full_graph_selected),
            "exact_step_structure": _exact_structure_comparison(
                baseline_selected,
                full_graph_selected,
                include_matched_calls=name != "decode_graph_control",
            ),
        }
    baseline_event_s = sum(
        float(event["event_duration_ns"]) for _, event in baseline
    ) / 1_000_000_000
    full_graph_event_s = sum(
        float(event["event_duration_ns"]) for _, event in full_graph
    ) / 1_000_000_000
    ground_truth = _load_json(full_graph_ground_truth_path)
    if not isinstance(ground_truth, dict):
        raise TypeError("FULL Graph ground truth must be an object")
    metrics = ground_truth["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("FULL Graph ground-truth metrics must be an object")
    max_dispatch_lag_ms = float(metrics["max_dispatch_lag_ms"])
    return {
        "interpretation_status": "phase_mechanism_only",
        "aggregate_comparison_valid": False,
        "aggregate_comparison_limitation": (
            "Scheduler trajectories differ and FULL exceeded the canonical "
            "arrival-dispatch gate; request metrics and total Event time are "
            "not a controlled Graph A/B."
        ),
        "full_decode_only_production_step_count": len(baseline),
        "full_graph_production_step_count": len(full_graph),
        "full_decode_only_model_event_s": baseline_event_s,
        "full_graph_model_event_s": full_graph_event_s,
        "full_graph_max_dispatch_lag_ms": max_dispatch_lag_ms,
        "dispatch_lag_limit_ms": 100.0,
        "dispatch_lag_gate_passed": max_dispatch_lag_ms <= 100.0,
        "comparisons": comparisons,
    }


def _complete_device_events(event_path: Path) -> list[dict[str, object]]:
    rows = _load_jsonl(event_path)
    headers = [
        row
        for row in rows
        if "record_kind" in row and row["record_kind"] == "header"
    ]
    footers = [
        row
        for row in rows
        if "record_kind" in row and row["record_kind"] == "footer"
    ]
    events = [row for row in rows if "sequence_id" in row]
    if len(headers) != 1 or len(footers) != 1:
        raise ValueError("paired device Event trace needs one header and one footer")
    if headers[0]["profiler_started"] is not False:
        raise ValueError("paired device Event trace must be a no-profiler run")
    if footers[0]["complete"] is not True:
        raise ValueError("paired device Event trace is incomplete")
    if footers[0]["record_count"] != len(events):
        raise ValueError("paired device Event footer count disagrees with records")
    if [row["sequence_id"] for row in events] != list(range(len(events))):
        raise ValueError("paired device Event sequence IDs are not contiguous")
    return events


def _validated_ground_truth(
    path: Path,
    expected_graph_mode: str,
    expected_compilation_config: dict[str, object],
) -> dict[str, object]:
    ground_truth = _load_json(path)
    if not isinstance(ground_truth, dict):
        raise TypeError("paired ground truth must be an object")
    if ground_truth["status"] != "success":
        raise ValueError("paired ground truth did not complete successfully")
    server = ground_truth["server"]
    if not isinstance(server, dict) or server["engine"]["graph_mode"] != expected_graph_mode:
        raise ValueError("paired ground truth used the wrong Graph mode")
    server_argv = server["argv"]
    compilation_index = server_argv.index("--compilation-config")
    compilation_config = json.loads(server_argv[compilation_index + 1])
    if compilation_config != expected_compilation_config:
        raise ValueError("paired ground truth used the wrong compilation config")
    metrics = ground_truth["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("paired ground-truth metrics must be an object")
    if metrics["completed_requests"] != 128:
        raise ValueError("paired ground truth did not complete all requests")
    if metrics["total_input_tokens"] != 28159:
        raise ValueError("paired ground-truth input work changed")
    if metrics["total_output_tokens"] != 30470:
        raise ValueError("paired ground-truth output work changed")
    if float(metrics["max_dispatch_lag_ms"]) > 100.0:
        raise ValueError("paired ground truth failed the dispatch-lag gate")
    return ground_truth


def _server_argv_without_compilation_config(
    ground_truth: dict[str, object],
) -> tuple[object, ...]:
    server_argv = ground_truth["server"]["argv"]
    compilation_index = server_argv.index("--compilation-config")
    return tuple(
        server_argv[:compilation_index] + server_argv[compilation_index + 2 :]
    )


def _paired_metric_summary(metrics: dict[str, object]) -> dict[str, float]:
    return {
        "duration_s": float(metrics["duration_s"]),
        "request_throughput_per_s": float(metrics["request_throughput_per_s"]),
        "ttft_mean_ms": float(metrics["ttft"]["mean_ms"]),
        "tpot_mean_ms": float(metrics["tpot"]["mean_ms"]),
        "e2e_mean_ms": float(metrics["e2e"]["mean_ms"]),
        "max_dispatch_lag_ms": float(metrics["max_dispatch_lag_ms"]),
    }


def _phase_event_summary(events: Sequence[dict[str, object]]) -> dict[str, object]:
    result = {}
    for phase in EVENT_PHASES:
        selected = [row for row in events if row["phase"] == phase]
        durations_ms = [float(row["event_duration_ns"]) / 1_000_000 for row in selected]
        host_call_wall_ms = [
            float(row["wall_duration_ns"]) / 1_000_000 for row in selected
        ]
        result[phase] = {
            "call_count": len(selected),
            "event_s": sum(durations_ms) / 1000.0,
            "event_p50_ms": statistics.median(durations_ms),
            "host_call_wall_s": sum(host_call_wall_ms) / 1000.0,
            "host_call_wall_p50_ms": statistics.median(host_call_wall_ms),
        }
    return result


def _graph_run_summary(
    metrics: dict[str, object], events: Sequence[dict[str, object]]
) -> dict[str, object]:
    event_s = sum(float(row["event_duration_ns"]) for row in events) / 1_000_000_000
    host_call_wall_s = (
        sum(float(row["wall_duration_ns"]) for row in events) / 1_000_000_000
    )
    return {
        "metrics": _paired_metric_summary(metrics),
        "production_model_call_count": len(events),
        "production_model_event_s": event_s,
        "production_model_host_call_wall_s": host_call_wall_s,
        "phase_events": _phase_event_summary(events),
    }


def _relative_to_baseline_percent(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, float]:
    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    return {
        "duration_change": (
            candidate_metrics["duration_s"] / baseline_metrics["duration_s"] - 1.0
        )
        * 100.0,
        "throughput_change": (
            candidate_metrics["request_throughput_per_s"]
            / baseline_metrics["request_throughput_per_s"]
            - 1.0
        )
        * 100.0,
        "ttft_change": (
            candidate_metrics["ttft_mean_ms"] / baseline_metrics["ttft_mean_ms"]
            - 1.0
        )
        * 100.0,
        "tpot_change": (
            candidate_metrics["tpot_mean_ms"] / baseline_metrics["tpot_mean_ms"]
            - 1.0
        )
        * 100.0,
        "e2e_change": (
            candidate_metrics["e2e_mean_ms"] / baseline_metrics["e2e_mean_ms"]
            - 1.0
        )
        * 100.0,
        "model_event_change": (
            candidate["production_model_event_s"]
            / baseline["production_model_event_s"]
            - 1.0
        )
        * 100.0,
    }


def _paired_full_graph_end_to_end(
    full_decode_only_event_path: Path,
    full_decode_only_ground_truth_path: Path,
    full_graph_event_path: Path,
    full_graph_ground_truth_path: Path,
    full_graph_512_event_path: Path,
    full_graph_512_ground_truth_path: Path,
) -> dict[str, object]:
    full_decode_only_ground_truth = _validated_ground_truth(
        full_decode_only_ground_truth_path,
        "acl_graph_full_decode_only",
        {"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    full_graph_ground_truth = _validated_ground_truth(
        full_graph_ground_truth_path,
        "acl_graph_full",
        {"cudagraph_mode": "FULL"},
    )
    full_graph_512_ground_truth = _validated_ground_truth(
        full_graph_512_ground_truth_path,
        "acl_graph_full",
        {"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 512},
    )
    baseline_server_argv = _server_argv_without_compilation_config(
        full_decode_only_ground_truth
    )
    if baseline_server_argv != _server_argv_without_compilation_config(
        full_graph_ground_truth
    ):
        raise ValueError("paired FULL run changed a non-Graph server argument")
    if baseline_server_argv != _server_argv_without_compilation_config(
        full_graph_512_ground_truth
    ):
        raise ValueError("paired FULL-512 run changed a non-Graph server argument")
    baseline_metrics = full_decode_only_ground_truth["metrics"]
    full_metrics = full_graph_ground_truth["metrics"]
    full_512_metrics = full_graph_512_ground_truth["metrics"]
    baseline_requests = baseline_metrics["per_request"]
    full_requests = full_metrics["per_request"]
    full_512_requests = full_512_metrics["per_request"]
    invariant_fields = (
        "request_id",
        "scheduled_arrival_s",
        "input_tokens",
        "output_tokens",
    )
    if [
        tuple(row[field] for field in invariant_fields)
        for row in baseline_requests
    ] != [
        tuple(row[field] for field in invariant_fields)
        for row in full_requests
    ]:
        raise ValueError("paired Graph runs changed request identity or work")
    if [
        tuple(row[field] for field in invariant_fields)
        for row in baseline_requests
    ] != [
        tuple(row[field] for field in invariant_fields)
        for row in full_512_requests
    ]:
        raise ValueError("paired 512-capture run changed request identity or work")

    baseline_events = _complete_device_events(full_decode_only_event_path)
    full_events = _complete_device_events(full_graph_event_path)
    full_512_events = _complete_device_events(full_graph_512_event_path)
    expected_baseline_warmup = ["model_and_logits_eager"] + [
        "model_and_logits_graph"
    ] * 7
    expected_full_warmup = ["model_and_logits_graph"] * 8
    if [row["phase"] for row in baseline_events[:WARMUP_MODEL_CALL_COUNT]] != (
        expected_baseline_warmup
    ):
        raise ValueError("FULL_DECODE_ONLY warmup Event sequence changed")
    if [row["phase"] for row in full_events[:WARMUP_MODEL_CALL_COUNT]] != (
        expected_full_warmup
    ):
        raise ValueError("FULL warmup Event sequence changed")
    if [row["phase"] for row in full_512_events[:WARMUP_MODEL_CALL_COUNT]] != (
        expected_full_warmup
    ):
        raise ValueError("FULL 512 warmup Event sequence changed")
    baseline_production_events = baseline_events[WARMUP_MODEL_CALL_COUNT:]
    full_production_events = full_events[WARMUP_MODEL_CALL_COUNT:]
    full_512_production_events = full_512_events[WARMUP_MODEL_CALL_COUNT:]
    baseline_summary = _graph_run_summary(
        baseline_metrics, baseline_production_events
    )
    full_summary = _graph_run_summary(full_metrics, full_production_events)
    full_512_summary = _graph_run_summary(
        full_512_metrics, full_512_production_events
    )
    return {
        "status": "accepted_same_client_graph_ab",
        "client_mode": "one_prearmed_event_loop_per_request",
        "request_invariant_fields": list(invariant_fields),
        "full_decode_only": baseline_summary,
        "full_default_256": full_summary,
        "full_max_capture_512": full_512_summary,
        "full_default_256_relative_to_full_decode_only_percent": (
            _relative_to_baseline_percent(baseline_summary, full_summary)
        ),
        "full_max_capture_512_relative_to_full_decode_only_percent": (
            _relative_to_baseline_percent(baseline_summary, full_512_summary)
        ),
        "interpretation": (
            "The request stream and client implementation are identical, both "
            "runs pass the arrival gate, and only Graph mode changes. Scheduler "
            "trajectories are allowed to differ because that feedback is part of "
            "the end-to-end effect."
        ),
    }


def _complete_profiled_device_events(event_path: Path) -> list[dict[str, object]]:
    rows = _load_jsonl(event_path)
    headers = [
        row
        for row in rows
        if "record_kind" in row and row["record_kind"] == "header"
    ]
    footers = [
        row
        for row in rows
        if "record_kind" in row and row["record_kind"] == "footer"
    ]
    events = [row for row in rows if "sequence_id" in row]
    if len(headers) != 1 or len(footers) != 1:
        raise ValueError("profiled device Event trace needs one header and one footer")
    if headers[0]["profiler_started"] is not True:
        raise ValueError("dynamic-profiler trace must record profiler_started=true")
    if footers[0]["complete"] is not True:
        raise ValueError("dynamic-profiler Event trace is incomplete")
    if footers[0]["record_count"] != len(events):
        raise ValueError("dynamic-profiler Event footer count disagrees with records")
    if [row["sequence_id"] for row in events] != list(range(len(events))):
        raise ValueError("dynamic-profiler Event sequence IDs are not contiguous")
    if len(events) <= WARMUP_MODEL_CALL_COUNT:
        raise ValueError("dynamic-profiler trace has no production calls")
    return events[WARMUP_MODEL_CALL_COUNT:]


def _pair_event_markers(
    events: Sequence[dict[str, object]], marker_ns: Sequence[int]
) -> tuple[list[tuple[int, int]], float]:
    pairs = []
    previous_end_index = -1
    maximum_error_ns = 0.0
    for event in events:
        duration_ns = float(event["event_duration_ns"])
        candidates = []
        candidate_stop = min(previous_end_index + 501, len(marker_ns))
        for start_index in range(previous_end_index + 1, candidate_stop):
            target_ns = marker_ns[start_index] + duration_ns
            end_index = bisect.bisect_left(
                marker_ns, target_ns, lo=start_index + 1
            )
            for candidate_end_index in (end_index - 1, end_index):
                if not start_index < candidate_end_index < len(marker_ns):
                    continue
                error_ns = abs(
                    marker_ns[candidate_end_index]
                    - marker_ns[start_index]
                    - duration_ns
                )
                if error_ns < 30.0:
                    candidates.append(
                        (start_index, candidate_end_index, error_ns)
                    )
        if not candidates:
            raise ValueError(
                "no monotonic profiler marker pair matches Event sequence "
                f"{event['sequence_id']} within 30 ns"
            )
        start_index, end_index, error_ns = min(
            candidates, key=lambda candidate: candidate[0]
        )
        pairs.append((marker_ns[start_index], marker_ns[end_index]))
        previous_end_index = end_index
        maximum_error_ns = max(maximum_error_ns, error_ns)
    return pairs, maximum_error_ns


def _clipped_task_union_ns(
    tasks: Sequence[tuple[int, int, int, bool]],
    lower_ns: int,
    upper_ns: int,
    *,
    include_compute_only: bool = False,
    exclude_graph_wait: bool = False,
    include_graph_wait_only: bool = False,
) -> int:
    intervals = []
    for start_ns, end_ns, task_type, has_compute_info in tasks:
        if end_ns <= lower_ns or start_ns >= upper_ns:
            continue
        if include_compute_only and not has_compute_info:
            continue
        if exclude_graph_wait and task_type == GRAPH_NOTIFY_WAIT_TASK_TYPE:
            continue
        if include_graph_wait_only and task_type != GRAPH_NOTIFY_WAIT_TASK_TYPE:
            continue
        intervals.append((max(start_ns, lower_ns), min(end_ns, upper_ns)))
    return _interval_union_ns(intervals)


def _dynamic_profiler_reduction(
    database: Path,
    event_path: Path,
    ground_truth_path: Path,
) -> dict[str, object]:
    ground_truth = _validated_ground_truth(
        ground_truth_path,
        "acl_graph_full",
        {"cudagraph_mode": "FULL", "max_cudagraph_capture_size": 512},
    )
    events = _complete_profiled_device_events(event_path)
    phase_counts = {
        phase: sum(event["phase"] == phase for event in events)
        for phase in EVENT_PHASES
    }
    if phase_counts != {
        "model_and_logits_eager": 22,
        "model_and_logits_graph": 1370,
    }:
        raise ValueError(f"dynamic-profiler phase counts changed: {phase_counts}")

    uri = f"file:{database}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"TASK", "COMPUTE_TASK_INFO", "CANN_API", "STRING_IDS"}
        if not required.issubset(tables):
            raise ValueError(
                f"dynamic msprof database lacks tables {sorted(required - tables)}"
            )
        marker_ns = [
            row[0]
            for row in connection.execute(
                "SELECT t.startNs FROM TASK t "
                "JOIN CANN_API a ON t.connectionId=a.connectionId "
                "JOIN STRING_IDS s ON a.name=s.id "
                "WHERE t.taskType=? AND s.value='aclrtRecordEvent' "
                "ORDER BY t.startNs",
                (EVENT_RECORD_TASK_TYPE,),
            )
        ]
        pairs, maximum_marker_error_ns = _pair_event_markers(events, marker_ns)
        tasks = [
            (row[0], row[1], row[2], bool(row[3]))
            for row in connection.execute(
                "SELECT t.startNs, t.endNs, t.taskType, "
                "c.globalTaskId IS NOT NULL FROM TASK t "
                "LEFT JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
                "ORDER BY t.startNs"
            )
        ]
        graph_wait = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT t.streamId), "
            "COUNT(c.globalTaskId), COUNT(DISTINCT a.connectionId), "
            "COUNT(DISTINCT s.value), MIN(s.value), SUM(t.endNs-t.startNs) "
            "FROM TASK t "
            "LEFT JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
            "LEFT JOIN CANN_API a ON t.connectionId=a.connectionId "
            "LEFT JOIN STRING_IDS s ON a.name=s.id WHERE t.taskType=?",
            (GRAPH_NOTIFY_WAIT_TASK_TYPE,),
        ).fetchone()
    if graph_wait[:6] != (1370, 1, 0, 1370, 1, "aclmdlRIExecuteAsync"):
        raise ValueError(f"Graph NOTIFY_WAIT contract changed: {graph_wait[:6]}")

    task_start_ns = [task[0] for task in tasks]
    maximum_task_duration_ns = max(task[1] - task[0] for task in tasks)
    phase_totals: dict[str, dict[str, int]] = {
        phase: {
            "call_count": 0,
            "event_ns": 0,
            "all_task_union_ns": 0,
            "non_wait_task_union_ns": 0,
            "compute_task_union_ns": 0,
        }
        for phase in EVENT_PHASES
    }
    for event, (lower_ns, upper_ns) in zip(events, pairs, strict=True):
        left = bisect.bisect_left(
            task_start_ns, lower_ns - maximum_task_duration_ns
        )
        right = bisect.bisect_right(task_start_ns, upper_ns)
        window_tasks = tasks[left:right]
        phase = str(event["phase"])
        totals = phase_totals[phase]
        totals["call_count"] += 1
        totals["event_ns"] += upper_ns - lower_ns
        totals["all_task_union_ns"] += _clipped_task_union_ns(
            window_tasks, lower_ns, upper_ns
        )
        totals["non_wait_task_union_ns"] += _clipped_task_union_ns(
            window_tasks,
            lower_ns,
            upper_ns,
            exclude_graph_wait=True,
        )
        totals["compute_task_union_ns"] += _clipped_task_union_ns(
            window_tasks,
            lower_ns,
            upper_ns,
            include_compute_only=True,
        )

    phase_summary = {}
    for phase, totals in phase_totals.items():
        event_ns = totals["event_ns"]
        non_wait_ns = totals["non_wait_task_union_ns"]
        compute_ns = totals["compute_task_union_ns"]
        phase_summary[phase] = {
            "call_count": totals["call_count"],
            "marker_event_s": event_ns / 1_000_000_000,
            "reported_event_s": sum(
                float(event["event_duration_ns"])
                for event in events
                if event["phase"] == phase
            )
            / 1_000_000_000,
            "all_task_union_s": totals["all_task_union_ns"] / 1_000_000_000,
            "non_wait_task_union_s": non_wait_ns / 1_000_000_000,
            "compute_info_task_union_s": compute_ns / 1_000_000_000,
            "no_non_wait_task_upper_bound_s": (event_ns - non_wait_ns)
            / 1_000_000_000,
            "no_non_wait_task_upper_bound_percent": (event_ns - non_wait_ns)
            / event_ns
            * 100.0,
            "compute_info_activity_lower_bound_percent": compute_ns
            / event_ns
            * 100.0,
        }

    envelope_start_ns = pairs[0][0]
    envelope_end_ns = pairs[-1][1]
    left = bisect.bisect_left(
        task_start_ns, envelope_start_ns - maximum_task_duration_ns
    )
    right = bisect.bisect_right(task_start_ns, envelope_end_ns)
    envelope_tasks = tasks[left:right]
    all_task_union_ns = _clipped_task_union_ns(
        envelope_tasks, envelope_start_ns, envelope_end_ns
    )
    non_wait_task_union_ns = _clipped_task_union_ns(
        envelope_tasks,
        envelope_start_ns,
        envelope_end_ns,
        exclude_graph_wait=True,
    )
    compute_task_union_ns = _clipped_task_union_ns(
        envelope_tasks,
        envelope_start_ns,
        envelope_end_ns,
        include_compute_only=True,
    )
    graph_wait_union_ns = _clipped_task_union_ns(
        envelope_tasks,
        envelope_start_ns,
        envelope_end_ns,
        include_graph_wait_only=True,
    )
    envelope_ns = envelope_end_ns - envelope_start_ns
    return {
        "status": "accepted_mechanism_evidence_not_fixed_trajectory_replacement",
        "metrics": _paired_metric_summary(ground_truth["metrics"]),
        "production_model_call_count": len(events),
        "phase_call_counts": phase_counts,
        "marker_pairing": {
            "pair_count": len(pairs),
            "maximum_elapsed_time_error_ns": maximum_marker_error_ns,
            "rule": (
                "earliest monotonic aclrtRecordEvent TASK pair after the prior "
                "end marker whose elapsed time matches torch.npu.Event within 30 ns"
            ),
        },
        "phase_event_task_split": phase_summary,
        "production_envelope": {
            "span_s": envelope_ns / 1_000_000_000,
            "all_task_union_s": all_task_union_ns / 1_000_000_000,
            "non_wait_task_union_s": non_wait_task_union_ns / 1_000_000_000,
            "compute_info_task_union_lower_bound_s": compute_task_union_ns
            / 1_000_000_000,
            "compute_info_activity_lower_bound_percent": compute_task_union_ns
            / envelope_ns
            * 100.0,
            "no_identified_compute_task_upper_bound_s": (
                envelope_ns - compute_task_union_ns
            )
            / 1_000_000_000,
            "graph_notify_wait_union_s": graph_wait_union_ns / 1_000_000_000,
            "graph_notify_wait_increment_to_union_s": (
                all_task_union_ns - non_wait_task_union_ns
            )
            / 1_000_000_000,
        },
        "graph_notify_wait_contract": {
            "task_type": GRAPH_NOTIFY_WAIT_TASK_TYPE,
            "exported_task_label": "NOTIFY_WAIT",
            "row_count": graph_wait[0],
            "stream_count": graph_wait[1],
            "compute_task_info_join_count": graph_wait[2],
            "distinct_connection_count": graph_wait[3],
            "cann_api_name": graph_wait[5],
            "raw_duration_sum_s": graph_wait[6] / 1_000_000_000,
            "interpretation": (
                "One nested synchronization wait per Graph call; exclude it "
                "from additive physical compute accounting."
            ),
        },
        "scope_limitation": (
            "This profiled run has 1370 Graph calls versus 1388 in the accepted "
            "no-profiler FULL-512 run. Its task unions prove mechanism and bound "
            "host/runtime-only time, but cannot be subtracted from the other "
            "trajectory's prediction to assign an exact kernel-error total."
        ),
    }


def main() -> None:
    arguments = _arguments()
    baseline_paths = {
        "msprof_db": arguments.msprof_db,
        "scheduler_sidecar": arguments.scheduler_sidecar,
        "alignment": arguments.alignment,
        "device_events": arguments.device_events,
        "engine_steps": arguments.engine_steps,
    }
    full_graph_paths = {
        "full_graph_device_events": arguments.full_graph_device_events,
        "full_graph_engine_steps": arguments.full_graph_engine_steps,
        "full_graph_ground_truth": arguments.full_graph_ground_truth,
    }
    paired_graph_paths = {
        "paired_full_decode_only_device_events": (
            arguments.paired_full_decode_only_device_events
        ),
        "paired_full_decode_only_ground_truth": (
            arguments.paired_full_decode_only_ground_truth
        ),
        "paired_full_graph_device_events": arguments.paired_full_graph_device_events,
        "paired_full_graph_ground_truth": arguments.paired_full_graph_ground_truth,
        "paired_full_graph_512_device_events": (
            arguments.paired_full_graph_512_device_events
        ),
        "paired_full_graph_512_ground_truth": (
            arguments.paired_full_graph_512_ground_truth
        ),
    }
    dynamic_profiler_paths = {
        "dynamic_profiler_device_events": arguments.dynamic_profiler_device_events,
        "dynamic_profiler_ground_truth": arguments.dynamic_profiler_ground_truth,
        "dynamic_profiler_msprof_db": arguments.dynamic_profiler_msprof_db,
    }
    paired_graph = _paired_full_graph_end_to_end(
        paired_graph_paths["paired_full_decode_only_device_events"],
        paired_graph_paths["paired_full_decode_only_ground_truth"],
        paired_graph_paths["paired_full_graph_device_events"],
        paired_graph_paths["paired_full_graph_ground_truth"],
        paired_graph_paths["paired_full_graph_512_device_events"],
        paired_graph_paths["paired_full_graph_512_ground_truth"],
    )
    dynamic_profiler = _dynamic_profiler_reduction(
        dynamic_profiler_paths["dynamic_profiler_msprof_db"],
        dynamic_profiler_paths["dynamic_profiler_device_events"],
        dynamic_profiler_paths["dynamic_profiler_ground_truth"],
    )
    no_profiler_full_512 = paired_graph["full_max_capture_512"]
    dynamic_profiler["relative_to_accepted_no_profiler_percent"] = {
        "duration": (
            dynamic_profiler["metrics"]["duration_s"]
            / no_profiler_full_512["metrics"]["duration_s"]
            - 1.0
        )
        * 100.0,
        "model_event": (
            sum(
                phase["reported_event_s"]
                for phase in dynamic_profiler["phase_event_task_split"].values()
            )
            / no_profiler_full_512["production_model_event_s"]
            - 1.0
        )
        * 100.0,
        "production_model_call_delta": (
            dynamic_profiler["production_model_call_count"]
            - no_profiler_full_512["production_model_call_count"]
        ),
    }
    result = {
        "schema_version": 1,
        "cell_id": "ascend_910b/qwen3-4b/tp1_qps16",
        "source_sha256": {
            **_verify_sources(baseline_paths, EXPECTED_SHA256),
            **_verify_sources(full_graph_paths, FULL_GRAPH_EXPECTED_SHA256),
            **_verify_sources(paired_graph_paths, PAIRED_GRAPH_EXPECTED_SHA256),
            **_verify_sources(
                dynamic_profiler_paths, DYNAMIC_PROFILER_EXPECTED_SHA256
            ),
        },
        "historical_msprof": _historical_task_reduction(
            baseline_paths["msprof_db"],
            baseline_paths["scheduler_sidecar"],
            baseline_paths["alignment"],
        ),
        "same_run_device_events": _event_reduction(
            baseline_paths["device_events"], baseline_paths["engine_steps"]
        ),
        "full_graph_canary_comparison": _full_graph_comparison(
            baseline_paths["device_events"],
            baseline_paths["engine_steps"],
            full_graph_paths["full_graph_device_events"],
            full_graph_paths["full_graph_engine_steps"],
            full_graph_paths["full_graph_ground_truth"],
        ),
        "paired_full_graph_end_to_end": paired_graph,
        "dynamic_profiler_task_attribution": dynamic_profiler,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
