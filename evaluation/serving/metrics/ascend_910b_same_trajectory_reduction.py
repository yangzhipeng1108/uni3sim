#!/usr/bin/env python3
"""Reduce the matched 910B Event, Scheduler, and msprof trajectory."""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence

from evaluation.serving.traces import scheduling_trajectory_replay
from src.serving.host_timing import HostLaunchProfile

from evaluation.json_shape import require_integer as _integer
from evaluation.json_shape import file_sha256 as _sha256

from evaluation.json_shape import require_object_dict as _mapping

from evaluation.json_shape import require_list_or_tuple as _sequence
from evaluation.json_shape import require_positive_number as _positive_number
from evaluation.json_shape import require_interval_union_size


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CELL_ID = "ascend_910b/qwen3-4b/tp1_qps16"
EXPECTED_SHA256 = {
    "device_events": "95f2cd28bd56ae25b960407bdc80a0d322729f9ea84ea5e316abf82fba541519",
    "ground_truth": "3dace9bdc3ee934ec79246fdfd0e55a4f3570cb34e6800326ea7490e3799694e",
    "msprof_db": "769191c6e3a08a8c812d592a3fb0700fff9c7c2a685c5ce8f8e5496f962eaacb",
    "scheduler_trace": "269c1e37e2efd6455a22cf530195e2f9d3807e26ffb5587f2c505cfc8744d581",
}
WARMUP_MODEL_CALL_COUNT = 8
EXPECTED_REQUEST_COUNT = 128
EXPECTED_INPUT_TOKENS = 28_159
EXPECTED_OUTPUT_TOKENS = 30_470
EXPECTED_SCHEDULED_TOKENS = 58_501
PHASES = ("prefill", "mixed", "decode")
EVENT_PHASES = ("model_and_logits_eager", "model_and_logits_graph")
EVENT_HEADER_FIELDS = {
    "schema_version",
    "record_kind",
    "hardware_id",
    "timer",
    "profiler_started",
    "synchronize_policy",
    "source_sha256",
    "patch_sha256",
    "record_count",
}
EVENT_FIELDS = {
    "sequence_id",
    "phase",
    "wall_start_ns",
    "start_record_before_ns",
    "start_record_after_ns",
    "wall_end_ns",
    "end_record_before_ns",
    "end_record_after_ns",
    "stream_pointer",
    "wall_duration_ns",
    "event_duration_ns",
}
EVENT_FOOTER_FIELDS = {
    "schema_version",
    "record_kind",
    "record_count",
    "complete",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-events", required=True, type=Path)
    parser.add_argument("--scheduler-trace", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--msprof-db", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    return parser.parse_args()




def _verify_sources(paths: Mapping[str, Path]) -> dict[str, str]:
    if set(paths) != set(EXPECTED_SHA256):
        raise ValueError("source paths do not cover the exact r18 input set")
    observed = {}
    for name, expected in EXPECTED_SHA256.items():
        actual = _sha256(paths[name])
        if actual != expected:
            raise ValueError(
                f"{name} SHA256 mismatch for {paths[name]}: "
                f"expected {expected}, got {actual}"
            )
        observed[name] = actual
    return observed


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    return rows








def _positive_number(value: object, label: str) -> float:
    return require_positive_number(value, label)


def _exact_fields(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _phase(prefill_tokens: int, decode_tokens: int) -> str:
    if prefill_tokens and decode_tokens:
        return "mixed"
    if prefill_tokens:
        return "prefill"
    if decode_tokens:
        return "decode"
    raise ValueError("work-bearing step has no prefill or decode tokens")


def _step_workload_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        list(rows), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_ground_truth(path: Path) -> dict[str, object]:
    value = _mapping(_load_json(path), "r18 ground truth")
    if value["status"] != "failed":
        raise ValueError("r18 must retain its dispatch-lag failure status")
    failure = _mapping(value["failure"], "r18 failure")
    if failure["type"] != "RuntimeError":
        raise ValueError("r18 failure type changed")
    server = _mapping(value["server"], "r18 server")
    if server["hardware_id"] != "ascend_910b":
        raise ValueError("r18 hardware identity changed")
    engine = _mapping(server["engine"], "r18 engine")
    if engine["graph_mode"] != "acl_graph_full":
        raise ValueError("r18 Graph mode changed")
    argv = _sequence(server["argv"], "r18 server argv")
    compilation_index = argv.index("--compilation-config")
    if json.loads(argv[compilation_index + 1]) != {
        "cudagraph_mode": "FULL",
        "max_cudagraph_capture_size": 512,
    }:
        raise ValueError("r18 compilation config changed")
    metrics = _mapping(value["metrics"], "r18 metrics")
    if metrics["completed_requests"] != EXPECTED_REQUEST_COUNT:
        raise ValueError("r18 did not complete the expected request set")
    if metrics["total_input_tokens"] != EXPECTED_INPUT_TOKENS:
        raise ValueError("r18 input-token total changed")
    if metrics["total_output_tokens"] != EXPECTED_OUTPUT_TOKENS:
        raise ValueError("r18 output-token total changed")
    dispatch_limit_ms = _positive_number(
        server["max_dispatch_lag_ms"], "r18 dispatch-lag limit"
    )
    dispatch_lag_ms = _positive_number(
        metrics["max_dispatch_lag_ms"], "r18 maximum dispatch lag"
    )
    if dispatch_lag_ms <= dispatch_limit_ms:
        raise ValueError("r18 unexpectedly passes the dispatch-lag gate")
    scheduler = _mapping(value["scheduler_trace"], "r18 Scheduler summary")
    if scheduler["row_count"] != 1392:
        raise ValueError("r18 Scheduler row count changed")
    if scheduler["scheduled_tokens"] != EXPECTED_SCHEDULED_TOKENS:
        raise ValueError("r18 Scheduler token total changed")
    if scheduler["sha256"] != EXPECTED_SHA256["scheduler_trace"]:
        raise ValueError("r18 Scheduler summary SHA256 changed")
    requests = _sequence(metrics["per_request"], "r18 per-request metrics")
    if len(requests) != EXPECTED_REQUEST_COUNT:
        raise ValueError("r18 per-request metric count changed")
    request_contract = []
    for index, raw_request in enumerate(requests):
        request = _mapping(raw_request, f"r18 request {index}")
        request_id = request["request_id"]
        if request_id != f"request-{index:012d}":
            raise ValueError(f"r18 request {index} identity changed")
        request_contract.append(
            {
                "request_id": request_id,
                "input_tokens": _integer(
                    request["input_tokens"], f"r18 request {index} input_tokens"
                ),
                "output_tokens": _integer(
                    request["output_tokens"],
                    f"r18 request {index} output_tokens",
                    minimum=1,
                ),
            }
        )
    return {
        "status": value["status"],
        "failure": dict(failure),
        "metrics": {
            "completed_requests": metrics["completed_requests"],
            "duration_s": metrics["duration_s"],
            "request_throughput_per_s": metrics["request_throughput_per_s"],
            "ttft": metrics["ttft"],
            "tpot": metrics["tpot"],
            "e2e": metrics["e2e"],
            "max_dispatch_lag_ms": dispatch_lag_ms,
        },
        "dispatch_lag_limit_ms": dispatch_limit_ms,
        "requests": request_contract,
        "model_config_sha256": value["workload"]["model_config_sha256"],
    }


def _validate_scheduler_trace(
    path: Path,
    request_contract: Sequence[Mapping[str, object]],
    *,
    schema_version: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return scheduling_trajectory_replay.load_and_validate_scheduler_trace(
        path, request_contract, schema_version=schema_version
    )

def _validate_device_events(
    path: Path, scheduler_rows: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = _load_jsonl(path)
    if len(rows) <= WARMUP_MODEL_CALL_COUNT + 2:
        raise ValueError("r18 device Event trace is incomplete")
    header = _mapping(rows[0], "Event header")
    footer = _mapping(rows[-1], "Event footer")
    events = rows[1:-1]
    _exact_fields(header, EVENT_HEADER_FIELDS, "Event header")
    _exact_fields(footer, EVENT_FOOTER_FIELDS, "Event footer")
    if (
        header["schema_version"] != 1
        or header["record_kind"] != "header"
        or header["hardware_id"] != "ascend_910b"
        or header["timer"] != "torch.npu.Event_current_stream"
        or header["profiler_started"] is not True
        or header["synchronize_policy"] != "once_at_worker_shutdown"
    ):
        raise ValueError("r18 Event header contract changed")
    if (
        footer["schema_version"] != 1
        or footer["record_kind"] != "footer"
        or footer["complete"] is not True
        or footer["record_count"] != len(events)
        or header["record_count"] != len(events)
    ):
        raise ValueError("r18 Event footer or record count changed")
    previous_wall_end = -1
    for sequence_id, event in enumerate(events):
        _exact_fields(event, EVENT_FIELDS, f"Event {sequence_id}")
        if event["sequence_id"] != sequence_id:
            raise ValueError("Event sequence IDs are not contiguous")
        if event["phase"] not in EVENT_PHASES:
            raise ValueError(f"Event {sequence_id} phase is unsupported")
        wall_start = _integer(event["wall_start_ns"], f"Event {sequence_id} start")
        wall_end = _integer(event["wall_end_ns"], f"Event {sequence_id} end")
        wall_duration = _integer(
            event["wall_duration_ns"], f"Event {sequence_id} wall duration", minimum=1
        )
        if wall_end - wall_start != wall_duration:
            raise ValueError(f"Event {sequence_id} wall duration differs")
        if wall_start < previous_wall_end:
            raise ValueError("device Event calls overlap or are out of order")
        previous_wall_end = wall_end
        _positive_number(
            event["event_duration_ns"], f"Event {sequence_id} device duration"
        )
    production = events[WARMUP_MODEL_CALL_COUNT:]
    if len(production) != len(scheduler_rows):
        raise ValueError("production Event and Scheduler step counts disagree")
    origin_lower_ns = max(
        int(event["wall_end_ns"]) - int(step["update_start_ns"])
        for event, step in zip(production, scheduler_rows, strict=True)
    )
    origin_upper_ns = min(
        int(event["wall_start_ns"]) - int(step["schedule_end_ns"])
        for event, step in zip(production, scheduler_rows, strict=True)
    )
    if origin_lower_ns > origin_upper_ns:
        raise ValueError(
            "no common clock origin places every Event inside its Scheduler step"
        )
    phase_counts = {
        phase: sum(event["phase"] == phase for event in production)
        for phase in EVENT_PHASES
    }
    return production, {
        "warmup_model_call_count": WARMUP_MODEL_CALL_COUNT,
        "production_model_call_count": len(production),
        "production_event_phase_counts": phase_counts,
        "scheduler_clock_origin_interval_ns": [
            origin_lower_ns,
            origin_upper_ns,
        ],
        "scheduler_clock_origin_interval_width_ns": (
            origin_upper_ns - origin_lower_ns
        ),
        "pairing_rule": (
            "drop the exact eight-call warmup prefix, pair production Events and "
            "non-idle Scheduler rows by order, and require one common monotonic-clock "
            "origin that contains every Event inside schedule_end..update_start"
        ),
    }


def _interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return require_interval_union_size(intervals, "TASK interval")


def _pair_event_markers(
    events: Sequence[Mapping[str, object]], marker_ns: Sequence[int]
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
                    candidates.append((start_index, candidate_end_index, error_ns))
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
    exclude_task_type: int | None = None,
    include_task_type_only: int | None = None,
) -> int:
    intervals = []
    for start_ns, end_ns, task_type, has_compute_info in tasks:
        if end_ns <= lower_ns or start_ns >= upper_ns:
            continue
        if include_compute_only and not has_compute_info:
            continue
        if exclude_task_type is not None and task_type == exclude_task_type:
            continue
        if include_task_type_only is not None and task_type != include_task_type_only:
            continue
        intervals.append((max(start_ns, lower_ns), min(end_ns, upper_ns)))
    return _interval_union_ns(intervals)


def _task_type_id(connection: sqlite3.Connection, label: str) -> int:
    rows = connection.execute(
        "SELECT id FROM STRING_IDS WHERE value=? ORDER BY id", (label,)
    ).fetchall()
    if len(rows) != 1:
        raise ValueError(f"msprof task type label {label!r} is not unique: {rows}")
    return _integer(rows[0][0], f"msprof {label} task type")


def _profiled_task_windows(
    database: Path,
    events: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
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
            raise ValueError(f"msprof database lacks tables {sorted(required - tables)}")
        event_record_type = _task_type_id(connection, "EVENT_RECORD")
        graph_wait_type = _task_type_id(connection, "NOTIFY_WAIT")
        marker_ns = [
            row[0]
            for row in connection.execute(
                "SELECT t.startNs FROM TASK t "
                "JOIN CANN_API a ON t.connectionId=a.connectionId "
                "JOIN STRING_IDS s ON a.name=s.id "
                "WHERE t.taskType=? AND s.value='aclrtRecordEvent' "
                "ORDER BY t.startNs",
                (event_record_type,),
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
            (graph_wait_type,),
        ).fetchone()
    graph_event_count = sum(
        event["phase"] == "model_and_logits_graph" for event in events
    )
    if graph_wait[:6] != (
        graph_event_count,
        1,
        0,
        graph_event_count,
        1,
        "aclmdlRIExecuteAsync",
    ):
        raise ValueError(f"Graph NOTIFY_WAIT contract changed: {graph_wait[:6]}")
    task_start_ns = [task[0] for task in tasks]
    maximum_task_duration_ns = max(task[1] - task[0] for task in tasks)
    windows = []
    for lower_ns, upper_ns in pairs:
        left = bisect.bisect_left(
            task_start_ns, lower_ns - maximum_task_duration_ns
        )
        right = bisect.bisect_right(task_start_ns, upper_ns)
        window_tasks = tasks[left:right]
        windows.append(
            {
                "marker_start_ns": lower_ns,
                "marker_end_ns": upper_ns,
                "event_ns": upper_ns - lower_ns,
                "all_task_union_ns": _clipped_task_union_ns(
                    window_tasks, lower_ns, upper_ns
                ),
                "non_wait_task_union_ns": _clipped_task_union_ns(
                    window_tasks,
                    lower_ns,
                    upper_ns,
                    exclude_task_type=graph_wait_type,
                ),
                "compute_info_task_union_ns": _clipped_task_union_ns(
                    window_tasks,
                    lower_ns,
                    upper_ns,
                    include_compute_only=True,
                ),
                "graph_notify_wait_union_ns": _clipped_task_union_ns(
                    window_tasks,
                    lower_ns,
                    upper_ns,
                    include_task_type_only=graph_wait_type,
                ),
            }
        )
    return windows, {
        "event_record_task_type": event_record_type,
        "graph_notify_wait_task_type": graph_wait_type,
        "marker_pair_count": len(pairs),
        "maximum_elapsed_time_error_ns": maximum_marker_error_ns,
        "graph_notify_wait_row_count": graph_wait[0],
        "graph_notify_wait_stream_count": graph_wait[1],
        "graph_notify_wait_compute_info_join_count": graph_wait[2],
        "graph_notify_wait_distinct_connection_count": graph_wait[3],
        "graph_notify_wait_cann_api_name": graph_wait[5],
        "graph_notify_wait_raw_duration_s": graph_wait[6] / 1_000_000_000,
        "graph_notify_wait_interpretation": (
            "one nested synchronization wait per Graph call; excluded from "
            "additive physical task accounting"
        ),
    }


def _predict_trajectory(
    repository_root: Path,
    scheduler_rows: Sequence[Mapping[str, object]],
    ground_truth: Mapping[str, object],
    *,
    host_launch_profile: HostLaunchProfile | None = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    return scheduling_trajectory_replay.predict_scheduling_trajectory_steps(
        repository_root,
        repository_root.resolve()
        / "configs/evaluation/experiments/sharegpt_phase1.json",
        CELL_ID,
        scheduler_rows,
        ground_truth["requests"],
        str(ground_truth["model_config_sha256"]),
        host_launch_profile=host_launch_profile,
    )

def _summarize_groups(
    steps: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    groups: dict[str, list[Mapping[str, object]]] = {
        "all": list(steps),
        **{
            phase: [step for step in steps if step["phase"] == phase]
            for phase in PHASES
        },
        **{
            event_phase: [
                step for step in steps if step["event_phase"] == event_phase
            ]
            for event_phase in EVENT_PHASES
        },
        **{
            f"{phase}/{event_phase}": [
                step
                for step in steps
                if step["phase"] == phase
                and step["event_phase"] == event_phase
            ]
            for phase in PHASES
            for event_phase in EVENT_PHASES
        },
        **{
            f"request_count/{label}": [
                step
                for step in steps
                if lower <= int(step["scheduled_request_count"]) <= upper
            ]
            for label, lower, upper in (
                ("1", 1, 1),
                ("2-4", 2, 4),
                ("5-8", 5, 8),
                ("9-16", 9, 16),
                ("17-32", 17, 32),
                ("33+", 33, EXPECTED_REQUEST_COUNT),
            )
        },
    }
    summaries = {}
    for name, selected in groups.items():
        if not selected:
            continue
        totals = {
            field: sum(float(step[field]) for step in selected)
            for field in (
                "predicted_timeline_ns",
                "event_ns",
                "all_task_union_ns",
                "non_wait_task_union_ns",
                "compute_info_task_union_ns",
                "graph_notify_wait_union_ns",
            )
        }
        event_ns = totals["event_ns"]
        predicted_ns = totals["predicted_timeline_ns"]
        non_wait_ns = totals["non_wait_task_union_ns"]
        compute_ns = totals["compute_info_task_union_ns"]
        if min(event_ns, predicted_ns, non_wait_ns, compute_ns) <= 0:
            raise ValueError(f"matched trajectory group {name} has zero duration")
        prediction_gap_ns = event_ns - predicted_ns
        if prediction_gap_ns <= 0:
            raise ValueError(
                f"matched trajectory group {name} does not underpredict Event time"
            )
        bottlenecks: defaultdict[str, float] = defaultdict(float)
        for step in selected:
            for category, duration_ns in step[
                "predicted_critical_path_by_bottleneck_ns"
            ].items():
                bottlenecks[category] += float(duration_ns)
        summaries[name] = {
            "step_count": len(selected),
            "scheduled_prefill_tokens": sum(
                int(step["scheduled_prefill_tokens"]) for step in selected
            ),
            "scheduled_decode_tokens": sum(
                int(step["scheduled_decode_tokens"]) for step in selected
            ),
            "predicted_timeline_s": predicted_ns / 1_000_000_000,
            "observed_event_s": event_ns / 1_000_000_000,
            "observed_all_task_union_s": (
                totals["all_task_union_ns"] / 1_000_000_000
            ),
            "observed_non_wait_task_union_s": non_wait_ns / 1_000_000_000,
            "observed_compute_info_task_union_s": compute_ns / 1_000_000_000,
            "nested_graph_notify_wait_union_s": (
                totals["graph_notify_wait_union_ns"] / 1_000_000_000
            ),
            "predicted_over_event": predicted_ns / event_ns,
            "compute_info_over_event": compute_ns / event_ns,
            "non_wait_tasks_over_event": non_wait_ns / event_ns,
            "event_over_predicted": event_ns / predicted_ns,
            "compute_info_over_predicted": compute_ns / predicted_ns,
            "kernel_prediction_to_compute_lower_bound_gap_s": (
                compute_ns - predicted_ns
            )
            / 1_000_000_000,
            "non_compute_task_increment_s": (non_wait_ns - compute_ns)
            / 1_000_000_000,
            "event_outside_non_wait_task_union_s": (event_ns - non_wait_ns)
            / 1_000_000_000,
            "prediction_gap_attribution_share": {
                "compute_info_above_prediction": (
                    compute_ns - predicted_ns
                )
                / prediction_gap_ns,
                "non_compute_task_increment": (non_wait_ns - compute_ns)
                / prediction_gap_ns,
                "event_outside_non_wait_task_union": (event_ns - non_wait_ns)
                / prediction_gap_ns,
            },
            "predicted_critical_path_by_bottleneck_s": {
                category: duration_ns / 1_000_000_000
                for category, duration_ns in sorted(bottlenecks.items())
            },
        }
    return summaries


def reduce_same_trajectory(
    repository_root: Path,
    device_events_path: Path,
    scheduler_trace_path: Path,
    ground_truth_path: Path,
    msprof_db_path: Path,
) -> dict[str, object]:
    paths = {
        "device_events": device_events_path,
        "scheduler_trace": scheduler_trace_path,
        "ground_truth": ground_truth_path,
        "msprof_db": msprof_db_path,
    }
    source_sha256 = _verify_sources(paths)
    ground_truth = _validate_ground_truth(ground_truth_path)
    scheduler_rows, scheduler_summary = _validate_scheduler_trace(
        scheduler_trace_path, ground_truth["requests"]
    )
    events, event_summary = _validate_device_events(
        device_events_path, scheduler_rows
    )
    task_windows, task_contract = _profiled_task_windows(msprof_db_path, events)
    predictions, kernel_contributions, predictor_identity = _predict_trajectory(
        repository_root, scheduler_rows, ground_truth
    )
    if not (
        len(scheduler_rows)
        == len(events)
        == len(task_windows)
        == len(predictions)
    ):
        raise ValueError("matched trajectory component row counts disagree")
    steps = []
    for row, event, task_window, prediction in zip(
        scheduler_rows, events, task_windows, predictions, strict=True
    ):
        steps.append(
            {
                "step_index": row["step_index"],
                "step_id": row["step_id"],
                "phase": row["phase"],
                "event_phase": event["phase"],
                "scheduled_request_count": row["scheduled_request_count"],
                "scheduled_prefill_tokens": row["scheduled_prefill_tokens"],
                "scheduled_decode_tokens": row["scheduled_decode_tokens"],
                "scheduled_total_tokens": row["scheduled_total_tokens"],
                **task_window,
                **prediction,
            }
        )
    return {
        "schema_version": 1,
        "cell_id": CELL_ID,
        "status": "mechanism_only_dispatch_lag_gate_failed",
        "formal_request_metrics_allowed": False,
        "forward_duration_observation_consumed": False,
        "source_sha256": source_sha256,
        "predictor_source": predictor_identity,
        "ground_truth": {
            "status": ground_truth["status"],
            "failure": ground_truth["failure"],
            "metrics": ground_truth["metrics"],
            "dispatch_lag_limit_ms": ground_truth["dispatch_lag_limit_ms"],
        },
        "scheduler_trace": scheduler_summary,
        "event_scheduler_pairing": event_summary,
        "msprof_task_contract": task_contract,
        "groups": _summarize_groups(steps),
        "predicted_kernel_contributions_by_phase": kernel_contributions,
        "steps": steps,
        "interpretation": (
            "The failed arrival gate forbids treating request metrics as formal. "
            "Event, Scheduler, predictor, and TASK values are nevertheless paired "
            "on the exact 1392-step production trajectory. NOTIFY_WAIT is nested "
            "Graph synchronization and is excluded from additive non-wait unions; "
            "COMPUTE_TASK_INFO union is a measured lower bound, not a fitted target."
        ),
    }


def main() -> None:
    arguments = _arguments()
    result = reduce_same_trajectory(
        arguments.repository_root,
        arguments.device_events,
        arguments.scheduler_trace,
        arguments.ground_truth,
        arguments.msprof_db,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
