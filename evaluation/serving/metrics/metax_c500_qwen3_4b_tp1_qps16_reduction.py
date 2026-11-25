#!/usr/bin/env python3
"""Reproduce the C500 Qwen3-4B QPS16 device/predictor reduction."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Callable, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
from evaluation.serving.metrics import compact_step_alignment
from evaluation.serving.traces.scheduling_trajectory_replay import (
    SCHEDULED_REQUEST_FIELDS,
    SCHEDULER_FIELDS,
)
from evaluation.serving.metrics import (
    sharegpt_compact_evaluation as compact_evaluation,
)
from src.kernel.hardware_spec import HWSpec
from src.serving.execution import ServingExecutionContext
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)

from evaluation.json_shape import require_integer as _integer
from evaluation.json_shape import file_sha256 as _sha256

from evaluation.json_shape import require_object_dict as _mapping

from evaluation.json_shape import require_list_or_tuple as _sequence
from evaluation.json_shape import require_positive_number as _positive_number
from evaluation.json_shape import require_interval_union_size


CELL_ID = "metax_c500/qwen3-4b/tp1_qps16"
PHASES = ("prefill", "mixed", "decode")
EXPECTED_SHA256 = {
    "alignment": "c03ca9d32cb9f64def3345f751903fb9c2919a322e05a0b2094e5ba3e937bc3f",
    "nsys_json": "480dc75127fa73bfbaffee06705c1f5d28e0cac3acc738a7696862fadbc92adb",
    "scheduler_sidecar": "46f5ec62f6887546008907dcf600af0d1d7398a3ca991c3af27ea6f13b776562",
    "sync_boundaries": "394d2117523ddb1149ce31afe473fb18ff77926f9c45664dca6fac80f2b0ffc7",
}
EXPECTED_NSYS_FIELDS = {"block", "dur_ns", "grid", "name", "smem", "ts_ns"}
EXPECTED_SCHEDULED_PREFILL_TOKENS = 28_159
EXPECTED_SCHEDULED_DECODE_TOKENS = 30_342
EXPECTED_NO_PROFILER_HEADER_FIELDS = {
    "schema_version",
    "record_kind",
    "hardware_id",
    "container_image",
    "vllm_version",
    "plugin_entrypoint",
    "vllm_use_v2_model_runner",
    "timer",
    "host_clock",
    "profiler_started",
    "synchronize_policy",
    "model_runner_source_path",
    "model_runner_pre_patch_sha256",
    "model_runner_post_patch_sha256",
    "gpu_worker_sha256",
    "plugin_gpu_worker_sha256",
    "patch_sha256",
    "device_index",
    "device_name",
    "record_count",
}
EXPECTED_NO_PROFILER_HEADER = {
    "schema_version": 1,
    "record_kind": "header",
    "hardware_id": "metax_c500",
    "container_image": (
        "harbor.sh.mxcr.io/ai-release/maca/"
        "vllm-metax:0.24.0-latest-torch2.10-py312-ubuntu22.04-amd64"
    ),
    "vllm_version": "0.24.0",
    "plugin_entrypoint": "metax -> vllm_metax:register",
    "vllm_use_v2_model_runner": False,
    "timer": "torch.cuda.Event_current_stream",
    "host_clock": "time.monotonic_ns",
    "profiler_started": False,
    "synchronize_policy": "once_at_declared_scheduled_token_total_or_shutdown",
    "model_runner_pre_patch_sha256": (
        "d2b8ec101122749e1a2610383d2ab74c5a21605e9d7995eb68192bc769986694"
    ),
    "model_runner_post_patch_sha256": (
        "8d5deb838e61b8681ca7044478093b22cb41ec771eb828462b619d32d0438389"
    ),
    "gpu_worker_sha256": (
        "271c3e2eaab7f1ca36fcd7c8e1620db9db4705f46ba5b3257dd6a3937f56b7df"
    ),
    "plugin_gpu_worker_sha256": (
        "9453e9efb1648b25222952f252f79686e4a403d363759288f5ebc6f6f69cd39c"
    ),
    "patch_sha256": (
        "a40b8be1361583b8970ba3ec4cc3fa6a7c2a7ea296057d9bd722400f2387f276"
    ),
    "device_index": 0,
    "device_name": "MetaX C500",
}
EXPECTED_NO_PROFILER_INTERVAL_FIELDS = {
    "schema_version",
    "record_kind",
    "sequence_id",
    "scope",
    "phase",
    "runtime_cudagraph_mode",
    "scheduled_request_count",
    "scheduled_total_tokens",
    "scheduled_prefill_tokens",
    "scheduled_decode_tokens",
    "padded_total_tokens",
    "wall_start_ns",
    "wall_end_ns",
    "stream_pointer",
    "wall_duration_ns",
    "event_duration_ns",
}
EXPECTED_REQUEST_COUNT = 128


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys-json", required=True, type=Path)
    parser.add_argument("--sync-boundaries", required=True, type=Path)
    parser.add_argument("--scheduler-sidecar", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="checkout whose current predictor is evaluated",
    )
    parser.add_argument(
        "--no-profiler-device-events",
        type=Path,
        help="optional same-run C500 device-Event JSONL",
    )
    parser.add_argument(
        "--no-profiler-device-events-sha256",
        help="required audited SHA256 when --no-profiler-device-events is supplied",
    )
    parser.add_argument(
        "--no-profiler-scheduler-trace",
        type=Path,
        help="same-run Scheduler JSONL paired with the optional device Events",
    )
    parser.add_argument(
        "--no-profiler-scheduler-trace-sha256",
        help="required audited SHA256 for the same-run Scheduler JSONL",
    )
    return parser.parse_args()




def _verify_sources(paths: Mapping[str, Path]) -> dict[str, str]:
    observed = {}
    if set(paths) != set(EXPECTED_SHA256):
        raise ValueError("source paths do not cover the exact audited input set")
    for name, expected in EXPECTED_SHA256.items():
        actual = _sha256(paths[name])
        if actual != expected:
            raise ValueError(
                f"{name} SHA256 mismatch for {paths[name]}: "
                f"expected {expected}, got {actual}"
            )
        observed[name] = actual
    return observed








def _exact_fields(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            record = _mapping(json.loads(line), f"{path}:{line_number}")
            records.append(record)
    return records


def _step_workload_sha256(rows: Sequence[Mapping[str, int]]) -> str:
    payload = json.dumps(
        list(rows), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summarize_no_profiler_device_events(
    path: Path, expected_sha256: str
) -> dict[str, object]:
    """Validate and summarize the frozen v0.24 C500 Event instrumentation."""

    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"no-profiler device Event SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    rows = _load_jsonl(path)
    if len(rows) < 3:
        raise ValueError("no-profiler device Event capture is incomplete")
    header = rows[0]
    footer = rows[-1]
    intervals = rows[1:-1]
    _exact_fields(header, EXPECTED_NO_PROFILER_HEADER_FIELDS, "Event header")
    _exact_fields(
        footer,
        {"schema_version", "record_kind", "record_count", "complete"},
        "Event footer",
    )
    model_runner_source_path = header["model_runner_source_path"]
    if (not isinstance(model_runner_source_path, str)
            or not model_runner_source_path.strip()
            or ".." in model_runner_source_path.split("/")
            or Path(model_runner_source_path).name != "gpu_model_runner.py"):
        raise ValueError("no-profiler device Event model runner path is invalid")
    for field, expected_value in EXPECTED_NO_PROFILER_HEADER.items():
        if header[field] != expected_value:
            raise ValueError(
                "no-profiler device Event header contract differs: "
                f"{field} expected {expected_value!r}, got {header[field]!r}"
            )
    if (
        footer["schema_version"] != 1
        or footer["record_kind"] != "footer"
        or footer["complete"] is not True
    ):
        raise ValueError("no-profiler device Event footer contract differs")
    header_count = _integer(header["record_count"], "Event header record_count")
    footer_count = _integer(footer["record_count"], "Event footer record_count")
    if header_count != len(intervals) or footer_count != len(intervals):
        raise ValueError("no-profiler device Event record counts disagree")

    phase_rows: dict[str, list[dict[str, object]]] = {
        phase: [] for phase in PHASES
    }
    step_workloads = []
    previous_wall_end_ns = None
    for sequence_id, raw_interval in enumerate(intervals):
        interval = dict(raw_interval)
        _exact_fields(
            interval,
            EXPECTED_NO_PROFILER_INTERVAL_FIELDS,
            f"Event interval {sequence_id}",
        )
        if interval["schema_version"] != 1 or interval["record_kind"] != "interval":
            raise ValueError(f"Event interval {sequence_id} contract differs")
        if interval["sequence_id"] != sequence_id:
            raise ValueError("no-profiler device Event sequence is not contiguous")
        if interval["scope"] != "model_and_logits":
            raise ValueError(f"Event interval {sequence_id} scope differs")
        phase = interval["phase"]
        if phase not in PHASES:
            raise ValueError(f"Event interval {sequence_id} phase is unsupported")
        if interval["runtime_cudagraph_mode"] not in {"NONE", "PIECEWISE", "FULL"}:
            raise ValueError(
                f"Event interval {sequence_id} cudagraph mode is unsupported"
            )
        request_count = _integer(
            interval["scheduled_request_count"],
            f"Event interval {sequence_id} scheduled_request_count",
            minimum=1,
        )
        scheduled_total = _integer(
            interval["scheduled_total_tokens"],
            f"Event interval {sequence_id} scheduled_total_tokens",
            minimum=1,
        )
        scheduled_prefill = _integer(
            interval["scheduled_prefill_tokens"],
            f"Event interval {sequence_id} scheduled_prefill_tokens",
        )
        scheduled_decode = _integer(
            interval["scheduled_decode_tokens"],
            f"Event interval {sequence_id} scheduled_decode_tokens",
        )
        padded_total = _integer(
            interval["padded_total_tokens"],
            f"Event interval {sequence_id} padded_total_tokens",
            minimum=1,
        )
        if scheduled_total != scheduled_prefill + scheduled_decode:
            raise ValueError(f"Event interval {sequence_id} token partition differs")
        if padded_total < scheduled_total:
            raise ValueError(f"Event interval {sequence_id} padding is too small")
        expected_phase = (
            "mixed"
            if scheduled_prefill and scheduled_decode
            else "prefill"
            if scheduled_prefill
            else "decode"
        )
        if phase != expected_phase:
            raise ValueError(f"Event interval {sequence_id} phase disagrees with tokens")
        if request_count > scheduled_total:
            raise ValueError(f"Event interval {sequence_id} request count exceeds tokens")
        wall_start_ns = _integer(
            interval["wall_start_ns"], f"Event interval {sequence_id} wall_start_ns"
        )
        wall_end_ns = _integer(
            interval["wall_end_ns"], f"Event interval {sequence_id} wall_end_ns"
        )
        wall_duration_ns = _integer(
            interval["wall_duration_ns"],
            f"Event interval {sequence_id} wall_duration_ns",
            minimum=1,
        )
        _integer(
            interval["stream_pointer"],
            f"Event interval {sequence_id} stream_pointer",
        )
        if wall_end_ns - wall_start_ns != wall_duration_ns:
            raise ValueError(f"Event interval {sequence_id} wall duration differs")
        if previous_wall_end_ns is not None and wall_start_ns < previous_wall_end_ns:
            raise ValueError("no-profiler Event host intervals overlap")
        previous_wall_end_ns = wall_end_ns
        interval["event_duration_ns"] = _positive_number(
            interval["event_duration_ns"],
            f"Event interval {sequence_id} event_duration_ns",
        )
        phase_rows[phase].append(interval)
        step_workloads.append(
            {
                "scheduled_request_count": request_count,
                "scheduled_prefill_tokens": scheduled_prefill,
                "scheduled_decode_tokens": scheduled_decode,
                "scheduled_total_tokens": scheduled_total,
            }
        )

    prefill_tokens = sum(
        int(row["scheduled_prefill_tokens"])
        for rows in phase_rows.values()
        for row in rows
    )
    decode_tokens = sum(
        int(row["scheduled_decode_tokens"])
        for rows in phase_rows.values()
        for row in rows
    )
    if prefill_tokens != EXPECTED_SCHEDULED_PREFILL_TOKENS:
        raise ValueError(
            "no-profiler prefill token conservation failed: "
            f"expected {EXPECTED_SCHEDULED_PREFILL_TOKENS}, got {prefill_tokens}"
        )
    if decode_tokens != EXPECTED_SCHEDULED_DECODE_TOKENS:
        raise ValueError(
            "no-profiler decode token conservation failed: "
            f"expected {EXPECTED_SCHEDULED_DECODE_TOKENS}, got {decode_tokens}"
        )

    phases = {}
    for phase in PHASES:
        selected = phase_rows[phase]
        if not selected:
            raise ValueError(f"no-profiler Event capture lacks {phase} steps")
        event_durations_ns = [float(row["event_duration_ns"]) for row in selected]
        mode_counts: defaultdict[str, int] = defaultdict(int)
        for row in selected:
            mode_counts[str(row["runtime_cudagraph_mode"])] += 1
        phases[phase] = {
            "step_count": len(selected),
            "scheduled_tokens": sum(
                int(row["scheduled_total_tokens"]) for row in selected
            ),
            "scheduled_prefill_tokens": sum(
                int(row["scheduled_prefill_tokens"]) for row in selected
            ),
            "scheduled_decode_tokens": sum(
                int(row["scheduled_decode_tokens"]) for row in selected
            ),
            "event_s": sum(event_durations_ns) / 1_000_000_000,
            "event_p50_ms": statistics.median(event_durations_ns) / 1_000_000,
            "host_call_wall_s": sum(
                int(row["wall_duration_ns"]) for row in selected
            )
            / 1_000_000_000,
            "runtime_cudagraph_mode_counts": dict(sorted(mode_counts.items())),
        }
    return {
        "status": "validated_same_run_no_profiler_device_events",
        "source_sha256": actual_sha256,
        "record_count": len(intervals),
        "scheduled_prefill_tokens": prefill_tokens,
        "scheduled_decode_tokens": decode_tokens,
        "scheduled_tokens": prefill_tokens + decode_tokens,
        "step_workload_sha256": _step_workload_sha256(step_workloads),
        "phases": phases,
    }


def _summarize_no_profiler_scheduler_trace(
    path: Path, expected_sha256: str
) -> dict[str, object]:
    """Validate the request-to-step trace paired with C500 device Events."""

    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"no-profiler Scheduler trace SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    rows = _load_jsonl(path)
    if not rows:
        raise ValueError("no-profiler Scheduler trace is empty")
    previous_source_iteration = -1
    previous_step_end_ns = -1
    cumulative_tokens = 0
    finished_requests: set[int] = set()
    step_workloads = []
    phase_counts = {phase: 0 for phase in PHASES}
    for step_index, row in enumerate(rows):
        _exact_fields(
            row,
            SCHEDULER_FIELDS,
            f"Scheduler step {step_index}",
        )
        if row["step_index"] != step_index:
            raise ValueError("no-profiler Scheduler step indexes are not contiguous")
        source_iteration = _integer(
            row["source_iteration_id"],
            f"Scheduler step {step_index} source_iteration_id",
        )
        if source_iteration <= previous_source_iteration:
            raise ValueError("no-profiler Scheduler source iterations are not ordered")
        previous_source_iteration = source_iteration
        if row["step_id"] != f"vllm:{source_iteration}":
            raise ValueError(f"Scheduler step {step_index} identity differs")
        step_start_ns = _integer(
            row["step_start_ns"], f"Scheduler step {step_index} step_start_ns"
        )
        schedule_end_ns = _integer(
            row["schedule_end_ns"], f"Scheduler step {step_index} schedule_end_ns"
        )
        update_start_ns = _integer(
            row["update_start_ns"], f"Scheduler step {step_index} update_start_ns"
        )
        step_end_ns = _integer(
            row["step_end_ns"], f"Scheduler step {step_index} step_end_ns"
        )
        if not step_start_ns <= schedule_end_ns <= update_start_ns <= step_end_ns:
            raise ValueError(f"Scheduler step {step_index} timestamps are unordered")
        if step_start_ns < previous_step_end_ns:
            raise ValueError("no-profiler Scheduler steps overlap")
        previous_step_end_ns = step_end_ns

        scheduled_requests = _sequence(
            row["scheduled_requests"],
            f"Scheduler step {step_index} scheduled_requests",
        )
        normalized_requests = []
        for request_order, raw_request in enumerate(scheduled_requests):
            request = _mapping(
                raw_request,
                f"Scheduler step {step_index} request {request_order}",
            )
            _exact_fields(
                request,
                SCHEDULED_REQUEST_FIELDS,
                f"Scheduler step {step_index} request {request_order}",
            )
            request_idx = _integer(
                request["request_idx"],
                f"Scheduler step {step_index} request_idx",
            )
            if request_idx >= EXPECTED_REQUEST_COUNT:
                raise ValueError(f"Scheduler step {step_index} request index is invalid")
            prefill_tokens = _integer(
                request["prefill_tokens"],
                f"Scheduler step {step_index} request prefill_tokens",
            )
            decode_tokens = _integer(
                request["decode_tokens"],
                f"Scheduler step {step_index} request decode_tokens",
            )
            scheduled_tokens = _integer(
                request["scheduled_tokens"],
                f"Scheduler step {step_index} request scheduled_tokens",
                minimum=1,
            )
            if scheduled_tokens != prefill_tokens + decode_tokens:
                raise ValueError(
                    f"Scheduler step {step_index} request token partition differs"
                )
            normalized_requests.append(
                (request_idx, prefill_tokens, decode_tokens, scheduled_tokens)
            )
        request_indexes = [request[0] for request in normalized_requests]
        if request_indexes != sorted(set(request_indexes)):
            raise ValueError(
                f"Scheduler step {step_index} request indexes are not unique and sorted"
            )
        request_count = _integer(
            row["scheduled_request_count"],
            f"Scheduler step {step_index} scheduled_request_count",
            minimum=1,
        )
        prefill_tokens = _integer(
            row["prefill_tokens"], f"Scheduler step {step_index} prefill_tokens"
        )
        decode_tokens = _integer(
            row["decode_tokens"], f"Scheduler step {step_index} decode_tokens"
        )
        scheduled_tokens = _integer(
            row["scheduled_tokens"],
            f"Scheduler step {step_index} scheduled_tokens",
            minimum=1,
        )
        if request_count != len(normalized_requests):
            raise ValueError(f"Scheduler step {step_index} request count differs")
        if prefill_tokens != sum(request[1] for request in normalized_requests):
            raise ValueError(f"Scheduler step {step_index} prefill tokens differ")
        if decode_tokens != sum(request[2] for request in normalized_requests):
            raise ValueError(f"Scheduler step {step_index} decode tokens differ")
        if scheduled_tokens != prefill_tokens + decode_tokens:
            raise ValueError(f"Scheduler step {step_index} token partition differs")
        prefill_request_count = _integer(
            row["prefill_request_count"],
            f"Scheduler step {step_index} prefill_request_count",
        )
        decode_request_count = _integer(
            row["decode_request_count"],
            f"Scheduler step {step_index} decode_request_count",
        )
        if prefill_request_count != sum(
            request[1] > 0 for request in normalized_requests
        ):
            raise ValueError(f"Scheduler step {step_index} prefill request count differs")
        if decode_request_count != sum(
            request[2] > 0 for request in normalized_requests
        ):
            raise ValueError(f"Scheduler step {step_index} decode request count differs")

        phase = (
            "mixed"
            if prefill_tokens and decode_tokens
            else "prefill"
            if prefill_tokens
            else "decode"
        )
        phase_counts[phase] += 1
        cumulative_tokens += scheduled_tokens
        if row["cumulative_scheduled_tokens"] != cumulative_tokens:
            raise ValueError(f"Scheduler step {step_index} cumulative tokens differ")
        newly_finished = _sequence(
            row["finished_request_indexes"],
            f"Scheduler step {step_index} finished_request_indexes",
        )
        normalized_finished = [
            _integer(value, f"Scheduler step {step_index} finished request")
            for value in newly_finished
        ]
        if normalized_finished != sorted(set(normalized_finished)) or any(
            value >= EXPECTED_REQUEST_COUNT for value in normalized_finished
        ):
            raise ValueError(f"Scheduler step {step_index} finished requests differ")
        duplicate_finished = finished_requests & set(normalized_finished)
        if duplicate_finished:
            raise ValueError("no-profiler Scheduler completed a request twice")
        finished_requests.update(normalized_finished)
        if row["cumulative_completed_requests"] != len(finished_requests):
            raise ValueError(f"Scheduler step {step_index} completion count differs")
        step_workloads.append(
            {
                "scheduled_request_count": request_count,
                "scheduled_prefill_tokens": prefill_tokens,
                "scheduled_decode_tokens": decode_tokens,
                "scheduled_total_tokens": scheduled_tokens,
            }
        )

    prefill_tokens = sum(row["scheduled_prefill_tokens"] for row in step_workloads)
    decode_tokens = sum(row["scheduled_decode_tokens"] for row in step_workloads)
    if prefill_tokens != EXPECTED_SCHEDULED_PREFILL_TOKENS:
        raise ValueError("no-profiler Scheduler prefill token conservation failed")
    if decode_tokens != EXPECTED_SCHEDULED_DECODE_TOKENS:
        raise ValueError("no-profiler Scheduler decode token conservation failed")
    if finished_requests != set(range(EXPECTED_REQUEST_COUNT)):
        raise ValueError("no-profiler Scheduler did not complete the exact request set")
    return {
        "status": "validated_same_run_no_profiler_scheduler_trace",
        "source_sha256": actual_sha256,
        "record_count": len(rows),
        "completed_requests": len(finished_requests),
        "scheduled_prefill_tokens": prefill_tokens,
        "scheduled_decode_tokens": decode_tokens,
        "scheduled_tokens": cumulative_tokens,
        "step_workload_sha256": _step_workload_sha256(step_workloads),
        "phase_step_counts": phase_counts,
    }


def _summarize_no_profiler_same_run(
    device_event_path: Path,
    device_event_sha256: str,
    scheduler_trace_path: Path,
    scheduler_trace_sha256: str,
) -> dict[str, object]:
    device_events = _summarize_no_profiler_device_events(
        device_event_path, device_event_sha256
    )
    scheduler_trace = _summarize_no_profiler_scheduler_trace(
        scheduler_trace_path, scheduler_trace_sha256
    )
    if device_events["step_workload_sha256"] != scheduler_trace[
        "step_workload_sha256"
    ]:
        raise ValueError(
            "no-profiler device Event and Scheduler step workloads disagree"
        )
    return {
        "status": "validated_paired_no_profiler_device_and_scheduler",
        "step_workload_sha256": device_events["step_workload_sha256"],
        "device_events": device_events,
        "scheduler_trace": scheduler_trace,
    }


def _interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return require_interval_union_size(intervals, "device event")


def _load_rank_zero_events(path: Path) -> list[dict[str, object]]:
    root = _mapping(_load_json(path), "nsys projection")
    _exact_fields(root, {"kernels", "per_rank", "bottleneck_rank"}, "nsys projection")
    if root["bottleneck_rank"] != "0":
        raise ValueError("nsys projection bottleneck rank must be string '0'")
    per_rank = _mapping(root["per_rank"], "nsys per_rank")
    _exact_fields(per_rank, {"0"}, "nsys per_rank")
    rank_zero = _mapping(per_rank["0"], "nsys rank 0")
    _exact_fields(rank_zero, {"kernels"}, "nsys rank 0")
    events = _sequence(rank_zero["kernels"], "nsys rank 0 kernels")
    top_level_events = _sequence(root["kernels"], "nsys top-level kernels")
    if len(top_level_events) != len(events):
        raise ValueError("nsys dual-view kernel counts disagree")
    normalized = []
    for index, raw_event in enumerate(events):
        event = _mapping(raw_event, f"nsys rank 0 kernel {index}")
        _exact_fields(event, EXPECTED_NSYS_FIELDS, f"nsys rank 0 kernel {index}")
        name = event["name"]
        if not isinstance(name, str) or not name:
            raise TypeError(f"nsys rank 0 kernel {index} name must be non-empty")
        timestamp_ns = _integer(event["ts_ns"], f"kernel {index} ts_ns")
        duration_ns = _integer(event["dur_ns"], f"kernel {index} dur_ns")
        for field in ("grid", "block"):
            geometry = _sequence(event[field], f"kernel {index} {field}")
            if len(geometry) != 3:
                raise ValueError(f"kernel {index} {field} must have three dimensions")
            for dimension in geometry:
                _integer(dimension, f"kernel {index} {field} dimension", minimum=1)
        _integer(event["smem"], f"kernel {index} smem")
        normalized.append(
            {
                "name": name,
                "start_ns": timestamp_ns,
                "end_ns": timestamp_ns + duration_ns,
                "raw_index": index,
            }
        )
    normalized.sort(key=lambda event: (event["start_ns"], event["raw_index"]))
    return normalized


def _load_syncs(path: Path) -> list[tuple[int, int]]:
    values = _sequence(_load_json(path), "sync boundaries")
    syncs = []
    previous_end = None
    for index, value in enumerate(values):
        pair = _sequence(value, f"sync boundary {index}")
        if len(pair) != 2:
            raise ValueError(f"sync boundary {index} must contain start and end")
        start_ns = _integer(pair[0], f"sync boundary {index} start")
        end_ns = _integer(pair[1], f"sync boundary {index} end")
        if end_ns <= start_ns:
            raise ValueError(f"sync boundary {index} has non-positive duration")
        if previous_end is not None and start_ns < previous_end:
            raise ValueError("sync boundaries overlap or are out of order")
        syncs.append((start_ns, end_ns))
        previous_end = end_ns
    if not syncs:
        raise ValueError("sync boundary list is empty")
    return syncs




def _first_model_entry_ns(events: Sequence[Mapping[str, object]]) -> int:
    for event in events:
        name = event["name"]
        if "rms_norm_kernel" in name or "fused_add_rms_norm" in name:
            return event["start_ns"]
    raise ValueError("nsys projection has no first-model-entry signature")


def _partition_raw_steps(
    events: Sequence[Mapping[str, object]],
    syncs: Sequence[tuple[int, int]],
    scheduler: Sequence[Mapping[str, object]],
    source_orders: Mapping[str, int],
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    first_model_entry_ns = _first_model_entry_ns(events)
    last_raw_start_ns = max(event["start_ns"] for event in events)
    filtered_syncs = [
        sync
        for sync in syncs
        if sync[1] >= first_model_entry_ns and sync[0] <= last_raw_start_ns
    ]
    if not filtered_syncs or len(filtered_syncs) > len(scheduler):
        raise ValueError("filtered sync prefix cannot align to non-idle Scheduler prefix")
    aligned_scheduler = scheduler[: len(filtered_syncs)]
    aligned_iterations = [record["iter_id"] for record in aligned_scheduler]
    if len(aligned_iterations) != len(set(aligned_iterations)):
        raise ValueError("aligned Scheduler prefix contains duplicate iterations")
    bounds = [first_model_entry_ns] + [sync[1] for sync in filtered_syncs]
    windows_by_phase = {}
    for phase in PHASES:
        order = source_orders[phase]
        if order not in aligned_iterations:
            raise ValueError(f"alignment {phase} source order {order} is outside prefix")
        window_index = aligned_iterations.index(order)
        lower_ns = bounds[window_index]
        upper_ns = bounds[window_index + 1]
        start_values = [event["start_ns"] for event in events]
        first_event = bisect_right(start_values, lower_ns - 1)
        final_event = bisect_right(start_values, upper_ns - 1)
        selected = events[first_event:final_event]
        crossing = [event for event in selected if event["end_ns"] > upper_ns]
        compute = [
            event
            for event in selected
            if not event["name"].lower().startswith(("memcpy", "memset"))
        ]
        copies = [
            event
            for event in selected
            if event["name"].lower().startswith(("memcpy", "memset"))
        ]
        if not compute:
            raise ValueError(f"raw {phase} partition contains no compute kernels")
        compute_intervals = [
            (event["start_ns"], event["end_ns"]) for event in compute
        ]
        copy_intervals = [
            (event["start_ns"], event["end_ns"]) for event in copies
        ]
        windows_by_phase[phase] = {
            "source_step_id": f"step-{order:012d}",
            "window_ns": [lower_ns, upper_ns],
            "structural_window_ms": (upper_ns - lower_ns) / 1_000_000,
            "cross_boundary_event_count": len(crossing),
            "compute_kernel_count": len(compute),
            "compute_busy_ms": _interval_union_ns(compute_intervals) / 1_000_000,
            "compute_active_span_ms": (
                max(end_ns for _, end_ns in compute_intervals)
                - min(start_ns for start_ns, _ in compute_intervals)
            )
            / 1_000_000,
            "copy_or_memset_count": len(copies),
            "copy_or_memset_busy_ms": _interval_union_ns(copy_intervals)
            / 1_000_000,
        }
    metadata = {
        "raw_event_count": len(events),
        "raw_sync_count": len(syncs),
        "filtered_sync_count": len(filtered_syncs),
        "non_idle_scheduler_count": len(scheduler),
        "aligned_iteration_count": len(aligned_scheduler),
        "first_model_entry_ns": first_model_entry_ns,
        "last_raw_start_ns": last_raw_start_ns,
        "partition_rule": (
            "filter sync intervals to profiler bounds; drop zero-token Scheduler "
            "records; associate retained sync-end prefix with non-idle Scheduler "
            "prefix; partition events by start timestamp in [bound[i], bound[i+1])"
        ),
    }
    return metadata, windows_by_phase


def _alignment_source_orders(alignment: Path) -> dict[str, int]:
    payload = _mapping(_load_json(alignment), "alignment")
    if payload["kind"] != "sharegpt_alignment_steps":
        raise ValueError("alignment kind is not sharegpt_alignment_steps")
    phases = _mapping(payload["phases"], "alignment phases")
    if set(phases) != set(PHASES):
        raise ValueError("alignment must contain prefill, mixed, and decode")
    orders = {}
    for phase in PHASES:
        phase_payload = _mapping(phases[phase], f"alignment {phase}")
        if phase_payload["status"] != "available":
            raise ValueError(f"alignment {phase} is unavailable")
        sample = _mapping(phase_payload["sample"], f"alignment {phase} sample")
        order = _integer(sample["source_execution_order"], f"alignment {phase} order")
        if sample["source_step_id"] != f"step-{order:012d}":
            raise ValueError(f"alignment {phase} step identity disagrees")
        orders[phase] = order
    return orders


def _predicted_kernel_contributions(
    phase: str,
    calls: Sequence[object],
    predictions: Mapping[str, object],
    occurrence_counts: Mapping[str, int],
) -> list[dict[str, object]]:
    """Aggregate one aligned step by actionable kernel identity."""

    grouped: dict[
        tuple[str, str | None, str | None, tuple[tuple[str, int], ...]],
        dict[str, object],
    ] = {}
    for call in calls:
        spec = call.kernel_spec
        shape_items = tuple(sorted(spec.shape.items()))
        key = (spec.kind.value, spec.variant, spec.dtype, shape_items)
        if key not in grouped:
            grouped[key] = {
                "phase": phase,
                "kernel_kind": spec.kind.value,
                "variant": spec.variant,
                "dtype": spec.dtype,
                "shape": dict(shape_items),
                "unique_call_count": 0,
                "logical_call_count": 0,
                "predicted_total_ms": 0.0,
            }
        row = grouped[key]
        occurrences = occurrence_counts[call.call_id]
        row["unique_call_count"] += 1
        row["logical_call_count"] += occurrences
        row["predicted_total_ms"] += (
            predictions[call.call_id].t_total_s * occurrences * 1000
        )

    total_ms = sum(row["predicted_total_ms"] for row in grouped.values())
    if total_ms <= 0:
        raise ValueError(f"{phase} predicted kernel contribution must be positive")
    rows = []
    for row in grouped.values():
        row["predicted_total_share"] = row["predicted_total_ms"] / total_ms
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -row["predicted_total_ms"],
            row["kernel_kind"],
            "" if row["variant"] is None else row["variant"],
            tuple(row["shape"].items()),
        ),
    )


def _normalize_historical_production_steps(
    scheduler_records: Sequence[Mapping[str, object]],
    case: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reconstruct duration-free ScheduleSteps from the historical sidecar."""

    non_idle = _load_non_idle_scheduler_records(scheduler_records)
    if list(non_idle[0]["scheduled"].items()) != [("warmup_0", 64)]:
        raise ValueError("historical Scheduler prefix does not start with warmup_0")
    production = non_idle[1:]
    request_count = len(case.requests)
    computed_tokens = [0] * request_count
    normalized = []
    phase_counts = {phase: 0 for phase in PHASES}
    phase_tokens = {
        phase: {"prefill": 0, "decode": 0} for phase in PHASES
    }
    for step_index, raw in enumerate(production):
        source_iteration = _integer(
            raw["iter_id"], f"production Scheduler step {step_index} iter_id"
        )
        scheduled = _mapping(
            raw["scheduled"], f"production Scheduler step {step_index} scheduled"
        )
        requests = []
        prefill_total = 0
        decode_total = 0
        seen_indexes = set()
        for raw_request_id, raw_tokens in scheduled.items():
            if not raw_request_id.startswith("req_"):
                raise ValueError("production Scheduler request identity differs")
            request_index_text = raw_request_id.removeprefix("req_")
            if not request_index_text.isdigit():
                raise ValueError("production Scheduler request index is malformed")
            request_index = int(request_index_text)
            if request_index >= request_count or request_index in seen_indexes:
                raise ValueError("production Scheduler request index is invalid")
            seen_indexes.add(request_index)
            scheduled_tokens = _integer(
                raw_tokens,
                f"production Scheduler step {step_index} request tokens",
                minimum=1,
            )
            request = case.requests[request_index]
            computed_before = computed_tokens[request_index]
            prompt_remaining = max(0, request.input_tokens - computed_before)
            prefill_tokens = min(scheduled_tokens, prompt_remaining)
            decode_tokens = scheduled_tokens - prefill_tokens
            if prefill_tokens and decode_tokens:
                raise ValueError(
                    "historical Scheduler scheduled prefill and decode for one "
                    "request in the same step"
                )
            target_work = request.input_tokens + request.output_tokens - 1
            if computed_before + scheduled_tokens > target_work:
                raise ValueError("historical Scheduler over-scheduled a request")
            requests.append(
                {
                    "request_idx": request_index,
                    "request_id": f"request-{request_index:012d}",
                    "computed_tokens": computed_before,
                    "prefill_tokens": prefill_tokens,
                    "decode_tokens": decode_tokens,
                    "scheduled_tokens": scheduled_tokens,
                }
            )
            computed_tokens[request_index] += scheduled_tokens
            prefill_total += prefill_tokens
            decode_total += decode_tokens
        if prefill_total and decode_total:
            phase = "mixed"
        elif prefill_total:
            phase = "prefill"
        elif decode_total:
            phase = "decode"
        else:
            raise ValueError("production Scheduler step has no model work")
        if int(raw["total_tokens"]) != prefill_total + decode_total:
            raise ValueError("production Scheduler step token partition differs")
        phase_counts[phase] += 1
        phase_tokens[phase]["prefill"] += prefill_total
        phase_tokens[phase]["decode"] += decode_total
        normalized.append(
            {
                "step_index": step_index,
                "source_iteration_id": source_iteration,
                "step_id": f"step-{source_iteration:012d}",
                "phase": phase,
                "scheduled_request_count": len(requests),
                "scheduled_prefill_tokens": prefill_total,
                "scheduled_decode_tokens": decode_total,
                "scheduled_total_tokens": prefill_total + decode_total,
                "requests": requests,
                "finished_request_indexes": [
                    int(request["request_idx"])
                    for request in requests
                    if computed_tokens[int(request["request_idx"])]
                    == (
                        case.requests[int(request["request_idx"])].input_tokens
                        + case.requests[int(request["request_idx"])].output_tokens
                        - 1
                    )
                ],
            }
        )
    expected_work = [
        request.input_tokens + request.output_tokens - 1
        for request in case.requests
    ]
    if computed_tokens != expected_work:
        raise ValueError("historical Scheduler does not conserve request work")
    return normalized, {
        "production_step_count": len(normalized),
        "request_count": request_count,
        "scheduled_prefill_tokens": sum(request.input_tokens for request in case.requests),
        "scheduled_decode_tokens": sum(request.output_tokens - 1 for request in case.requests),
        "phase_step_counts": phase_counts,
        "phase_tokens": phase_tokens,
    }


def _load_non_idle_scheduler_records(
    records: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    non_idle = []
    previous_iter = None
    previous_timestamp = None
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"scheduler record {index}")
        _exact_fields(
            record,
            {"iter_id", "ts_us", "scheduled", "total_tokens", "finished"},
            f"scheduler record {index}",
        )
        iteration = _integer(record["iter_id"], f"scheduler record {index} iter_id")
        timestamp = _integer(record["ts_us"], f"scheduler record {index} ts_us")
        total_tokens = _integer(
            record["total_tokens"], f"scheduler record {index} total_tokens"
        )
        scheduled = _mapping(record["scheduled"], f"scheduler record {index} scheduled")
        if sum(
            _integer(tokens, f"scheduler record {index} scheduled tokens")
            for tokens in scheduled.values()
        ) != total_tokens:
            raise ValueError(f"scheduler record {index} token total disagrees")
        if previous_iter is not None and iteration <= previous_iter:
            raise ValueError("Scheduler iter_id must strictly increase")
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("Scheduler ts_us must strictly increase")
        previous_iter = iteration
        previous_timestamp = timestamp
        if total_tokens:
            non_idle.append(record)
    if not non_idle:
        raise ValueError("Scheduler has no non-idle records")
    return non_idle


def _profiled_prefix_windows(
    events: Sequence[Mapping[str, object]],
    syncs: Sequence[tuple[int, int]],
    production_steps: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    first_model_entry_ns = _first_model_entry_ns(events)
    last_raw_start_ns = max(int(event["start_ns"]) for event in events)
    filtered_syncs = [
        sync
        for sync in syncs
        if sync[1] >= first_model_entry_ns and sync[0] <= last_raw_start_ns
    ]
    if len(filtered_syncs) < 2:
        raise ValueError("profiled prefix lacks production step windows")
    profiled_count = len(filtered_syncs) - 1
    if profiled_count > len(production_steps):
        raise ValueError("profiled prefix exceeds production Scheduler trajectory")
    bounds = [first_model_entry_ns] + [sync[1] for sync in filtered_syncs]
    start_values = [int(event["start_ns"]) for event in events]
    windows = []
    for production_index, step in enumerate(production_steps[:profiled_count]):
        window_index = production_index + 1
        lower_ns = bounds[window_index]
        upper_ns = bounds[window_index + 1]
        first_event = bisect_right(start_values, lower_ns - 1)
        final_event = bisect_right(start_values, upper_ns - 1)
        selected = events[first_event:final_event]
        if any(int(event["end_ns"]) > upper_ns for event in selected):
            raise ValueError("profiled prefix contains a cross-boundary event")
        compute = [
            event
            for event in selected
            if not str(event["name"]).lower().startswith(("memcpy", "memset"))
        ]
        copies = [
            event
            for event in selected
            if str(event["name"]).lower().startswith(("memcpy", "memset"))
        ]
        if not compute:
            raise ValueError("profiled production window contains no compute kernel")
        compute_intervals = [
            (int(event["start_ns"]), int(event["end_ns"])) for event in compute
        ]
        copy_intervals = [
            (int(event["start_ns"]), int(event["end_ns"])) for event in copies
        ]
        windows.append(
            {
                **step,
                "window_start_ns": lower_ns,
                "window_end_ns": upper_ns,
                "structural_window_ns": upper_ns - lower_ns,
                "compute_kernel_count": len(compute),
                "compute_busy_ns": _interval_union_ns(compute_intervals),
                "compute_active_span_ns": (
                    max(end for _, end in compute_intervals)
                    - min(start for start, _ in compute_intervals)
                ),
                "copy_or_memset_count": len(copies),
                "copy_or_memset_busy_ns": _interval_union_ns(copy_intervals),
            }
        )
    included_prefill = sum(
        int(step["scheduled_prefill_tokens"]) for step in windows
    )
    included_decode = sum(
        int(step["scheduled_decode_tokens"]) for step in windows
    )
    return windows, {
        "profiled_production_step_count": len(windows),
        "unprofiled_tail_step_count": len(production_steps) - len(windows),
        "included_prefill_tokens": included_prefill,
        "included_decode_tokens": included_decode,
        "included_tokens": included_prefill + included_decode,
        "unprofiled_tail_tokens": sum(
            int(step["scheduled_total_tokens"])
            for step in production_steps[len(windows):]
        ),
        "coverage_fraction": (
            (included_prefill + included_decode)
            / sum(int(step["scheduled_total_tokens"]) for step in production_steps)
        ),
    }


def _schedule_step_from_historical(
    row: Mapping[str, object], attention_kv_layout: str | None
) -> ScheduleStep:
    requests = tuple(
        ScheduledRequest(
            request_id=str(request["request_id"]),
            phase=("prefill" if int(request["prefill_tokens"]) else "decode"),
            input_tokens=int(request["prefill_tokens"]),
            output_tokens=int(request["decode_tokens"]),
            computed_tokens=int(request["computed_tokens"]),
        )
        for request in row["requests"]
    )
    return ScheduleStep(
        step_id=str(row["step_id"]),
        pool_name="main",
        requests=requests,
        attention_kv_layout=attention_kv_layout,
    )


def _predict_profiled_prefix(
    repository_root: Path,
    rows: Sequence[Mapping[str, object]],
    case: object,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    root = repository_root.resolve()
    config_path = root / "configs/evaluation/experiments/sharegpt_phase1.json"
    config = compact_evaluation.load_phase1_config(config_path)
    cell = compact_evaluation.select_phase1_cells(config, [CELL_ID], all_cases=False)[0]
    model_path = root / cell.model.simulation_model_path
    hardware_path = root / cell.hardware.hardware_config_path
    model = load_model_config(str(model_path))
    hardware, prediction, deployment = (
        compact_step_alignment.load_case_setup(
            config, model, hardware_path))

    execution = ServingExecutionContext(
        model,
        deployment,
        {hardware.hardware_id: hardware},
        prediction["launch_policy"],
        prediction["prediction_method"],
        RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            prediction["attention_kv_layout"],
            prediction["attention_kv_distribution"],
        ),
        execution_mode="accelerated",
    )
    alignment_by_order = {
        sample.source_execution_order: sample
        for sample in case.alignment_steps.values()
        if sample is not None
    }
    results = []
    for row in rows:
        step = _schedule_step_from_historical(row, None)
        source_iteration = int(row["source_iteration_id"])
        if source_iteration in alignment_by_order:
            observed = compact_step_alignment.deserialize_schedule_step(
                alignment_by_order[source_iteration].scheduler
            )
            if asdict(step) != asdict(observed):
                raise ValueError(
                    "reconstructed historical step disagrees with canonical "
                    f"alignment at iteration {source_iteration}"
                )
        components = execution.execute_step(step)
        results.append(
            {
                "predicted_timeline_ns": (
                    components.execution_timeline.duration_s * 1_000_000_000.0
                ),
                "predicted_critical_path_by_bottleneck_ns": {
                    category.removeprefix("kernel:"): seconds * 1_000_000_000.0
                    for category, seconds in (
                        components.execution_timeline.critical_path_contributions
                    )
                },
            }
        )
    return results, {
        "phase1_config": _sha256(config_path),
        "hardware_config": _sha256(hardware_path),
        "model_config": _sha256(model_path),
        "execution_cache": execution.cache_stats(),
        "execution_performance": execution.performance_stats(),
    }


def _summarize_profiled_prefix(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    groups = {
        "all": list(rows),
        **{
            phase: [row for row in rows if row["phase"] == phase]
            for phase in PHASES
        },
    }
    result = {}
    for name, selected in groups.items():
        if not selected:
            continue
        predicted_ns = sum(float(row["predicted_timeline_ns"]) for row in selected)
        busy_ns = sum(int(row["compute_busy_ns"]) for row in selected)
        span_ns = sum(int(row["compute_active_span_ns"]) for row in selected)
        structural_ns = sum(int(row["structural_window_ns"]) for row in selected)
        if min(predicted_ns, busy_ns, span_ns, structural_ns) <= 0:
            raise ValueError("profiled-prefix aggregate contains zero duration")
        gap_ns = structural_ns - predicted_ns
        components = {
            "compute_busy_above_prediction": busy_ns - predicted_ns,
            "within_compute_span_inactive": span_ns - busy_ns,
            "outside_compute_active_span": structural_ns - span_ns,
        }
        bottlenecks: defaultdict[str, float] = defaultdict(float)
        for row in selected:
            for category, duration_ns in row[
                "predicted_critical_path_by_bottleneck_ns"
            ].items():
                bottlenecks[category] += float(duration_ns)
        result[name] = {
            "step_count": len(selected),
            "scheduled_prefill_tokens": sum(
                int(row["scheduled_prefill_tokens"]) for row in selected
            ),
            "scheduled_decode_tokens": sum(
                int(row["scheduled_decode_tokens"]) for row in selected
            ),
            "predicted_timeline_s": predicted_ns / 1_000_000_000.0,
            "observed_compute_busy_s": busy_ns / 1_000_000_000.0,
            "observed_compute_active_span_s": span_ns / 1_000_000_000.0,
            "observed_structural_window_s": structural_ns / 1_000_000_000.0,
            "compute_busy_over_predicted": busy_ns / predicted_ns,
            "compute_active_span_over_predicted": span_ns / predicted_ns,
            "structural_window_over_predicted": structural_ns / predicted_ns,
            "gap_components_s": {
                key: value / 1_000_000_000.0 for key, value in components.items()
            },
            "gap_component_share": {
                key: value / gap_ns for key, value in components.items()
            },
            "predicted_critical_path_by_bottleneck_s": {
                key: value / 1_000_000_000.0
                for key, value in sorted(bottlenecks.items())
            },
        }
    return result


def _percentile(values: Sequence[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    rank = percentile / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _replay_historical_trajectory_request_metrics(
    rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    case: object,
) -> dict[str, object]:
    """Replay historical batch membership with current Device predictions."""

    if len(rows) != len(predictions):
        raise ValueError("historical trajectory prediction count differs")
    first_token_s: list[float | None] = [None] * len(case.requests)
    completion_s: list[float | None] = [None] * len(case.requests)
    current_s = 0.0
    for row, prediction in zip(rows, predictions, strict=True):
        required_arrival_s = max(
            case.requests[int(request["request_idx"])].arrival_time_ms / 1000.0
            for request in row["requests"]
        )
        current_s = max(current_s, required_arrival_s)
        current_s += float(prediction["predicted_timeline_ns"]) / 1_000_000_000.0
        for request in row["requests"]:
            request_index = int(request["request_idx"])
            if (
                first_token_s[request_index] is None
                and int(request["prefill_tokens"]) > 0
                and int(request["computed_tokens"])
                + int(request["prefill_tokens"])
                >= case.requests[request_index].input_tokens
            ):
                first_token_s[request_index] = current_s
        for request_index in row["finished_request_indexes"]:
            completion_s[request_index] = current_s
    if any(value is None for value in first_token_s + completion_s):
        raise ValueError("historical trajectory request timing is incomplete")
    metric_values = {"ttft_ms": [], "tpot_ms": [], "e2e_latency_ms": []}
    for request_index, request in enumerate(case.requests):
        arrival_s = request.arrival_time_ms / 1000.0
        first_s = float(first_token_s[request_index])
        final_s = float(completion_s[request_index])
        if first_s < arrival_s or final_s < first_s:
            raise ValueError("historical trajectory request timing is invalid")
        metric_values["ttft_ms"].append((first_s - arrival_s) * 1000.0)
        metric_values["tpot_ms"].append(
            (final_s - first_s) * 1000.0 / (request.output_tokens - 1)
        )
        metric_values["e2e_latency_ms"].append((final_s - arrival_s) * 1000.0)
    return {
        "request_count": len(case.requests),
        "duration_s": current_s,
        "request_throughput_per_s": len(case.requests) / current_s,
        "latency": {
            metric: {
                "mean": statistics.fmean(values),
                **{
                    f"p{percentile}": _percentile(values, percentile)
                    for percentile in range(101)
                },
            }
            for metric, values in metric_values.items()
        },
    }


def _predict_steps(
    repository_root: Path,
    alignment_path: Path,
    load_compact_case: Callable[[Path], object],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    root = repository_root.resolve()
    config_path = root / "configs/evaluation/experiments/sharegpt_phase1.json"
    config = compact_evaluation.load_phase1_config(config_path)
    cell = compact_evaluation.select_phase1_cells(
        config, [CELL_ID], all_cases=False
    )[0]
    canonical_case_path = compact_evaluation.compact_case_path(root, config, cell)
    if alignment_path.resolve() != (
        canonical_case_path / "alignment_steps.json.zst"
    ).resolve():
        raise ValueError("alignment path is not the selected canonical compact case")
    manifest_path = canonical_case_path / "manifest.json"
    if _sha256(manifest_path) != cell.manifest_sha256:
        raise ValueError("compact manifest SHA256 disagrees with phase-1 contract")
    case = load_compact_case(canonical_case_path)
    model_path = root / cell.model.simulation_model_path
    hardware_path = root / cell.hardware.hardware_config_path
    model = load_model_config(str(model_path))
    hardware = HWSpec(hardware_path)
    if model.num_experts != 0:
        raise ValueError("this deterministic CLI is restricted to dense Qwen3-4B")
    prediction = config.prediction.values
    deployment = compact_step_alignment.build_deployment(
        prediction, model, hardware.hardware_id
    )
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        prediction["attention_kv_layout"],
        prediction["attention_kv_distribution"],
    )
    execution = ServingExecutionContext(
        model,
        deployment,
        {hardware.hardware_id: hardware},
        prediction["launch_policy"],
        prediction["prediction_method"],
        runtime_inputs,
        execution_mode="reference",
    )
    results = {}
    for phase in PHASES:
        sample = case.alignment_steps[phase]
        if sample is None:
            raise ValueError(f"compact case lacks alignment phase {phase}")
        step = compact_step_alignment.deserialize_schedule_step(sample.scheduler)
        if step.step_id != sample.source_step_id:
            raise ValueError(f"compact {phase} step identity disagrees")
        components = execution.execute_step(step)
        occurrence_counts = dict(
            components.execution_timeline.call_occurrence_counts
        )
        by_kind: defaultdict[str, float] = defaultdict(float)
        for call in components.kernel_plan.calls:
            prediction_result = components.predictions[call.call_id]
            by_kind[call.kernel_spec.kind.value] += (
                prediction_result.t_total_s * occurrence_counts[call.call_id]
            )
        prefill_tokens = sum(
            request.scheduled_tokens
            for request in step.requests
            if request.phase == "prefill"
        )
        decode_tokens = sum(
            request.scheduled_tokens
            for request in step.requests
            if request.phase == "decode"
        )
        observed_launch_count = sum(
            len(group["events"]) for group in sample.launches
        )
        results[phase] = {
            "step_id": step.step_id,
            "source_execution_order": sample.source_execution_order,
            "request_count": len(step.requests),
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "total_tokens": step.token_count,
            "observed_compute_launch_count": observed_launch_count,
            "predicted_unique_call_count": len(components.kernel_plan.calls),
            "predicted_logical_call_count": sum(occurrence_counts.values()),
            "predicted_timeline_ms": (
                components.execution_timeline.duration_s * 1000
            ),
            "predicted_by_kernel_kind_ms": {
                kind: seconds * 1000 for kind, seconds in sorted(by_kind.items())
            },
            "predicted_kernel_contributions": _predicted_kernel_contributions(
                phase,
                components.kernel_plan.calls,
                components.predictions,
                occurrence_counts,
            ),
            "predicted_critical_path_by_bottleneck_ms": {
                category.removeprefix("kernel:"): seconds * 1000
                for category, seconds in (
                    components.execution_timeline.critical_path_contributions
                )
            },
        }
    source_identity = {
        "phase1_config": _sha256(config_path),
        "hardware_config": _sha256(hardware_path),
        "model_config": _sha256(model_path),
        "serving_execution": _sha256(root / "src/serving/execution.py"),
        "execution_timeline": _sha256(
            root / "src/serving/kernel_lowering/execution_timeline.py"
        ),
    }
    return results, source_identity


def _combine_results(
    predictor: Mapping[str, Mapping[str, object]],
    raw: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    combined = {}
    for phase in PHASES:
        predicted = predictor[phase]
        observed = raw[phase]
        if predicted["step_id"] != observed["source_step_id"]:
            raise ValueError(f"{phase} raw and predictor step identity disagree")
        if predicted["observed_compute_launch_count"] != observed["compute_kernel_count"]:
            raise ValueError(f"{phase} compact and raw compute launch counts disagree")
        predicted_ms = predicted["predicted_timeline_ms"]
        busy_ms = observed["compute_busy_ms"]
        span_ms = observed["compute_active_span_ms"]
        inactive_ms = span_ms - busy_ms
        span_gap_ms = span_ms - predicted_ms
        combined[phase] = {
            **predicted,
            "raw_partition": observed,
            "comparison": {
                "raw_compute_busy_over_predicted": busy_ms / predicted_ms,
                "raw_compute_span_over_predicted": span_ms / predicted_ms,
                "raw_busy_minus_predicted_ms": busy_ms - predicted_ms,
                "within_span_device_inactive_ms": inactive_ms,
                "raw_span_minus_predicted_ms": span_gap_ms,
                "inactive_share_of_span_prediction_gap": inactive_ms / span_gap_ms,
            },
        }
    return combined


def main(load_compact_case: Callable[[Path], object]) -> None:
    arguments = _arguments()
    no_profiler_arguments = (
        arguments.no_profiler_device_events,
        arguments.no_profiler_device_events_sha256,
        arguments.no_profiler_scheduler_trace,
        arguments.no_profiler_scheduler_trace_sha256,
    )
    if any(value is None for value in no_profiler_arguments) and not all(
        value is None for value in no_profiler_arguments
    ):
        raise ValueError(
            "device Event and Scheduler trace paths and SHA256 values must be "
            "supplied together"
        )
    sources = {
        "nsys_json": arguments.nsys_json,
        "sync_boundaries": arguments.sync_boundaries,
        "scheduler_sidecar": arguments.scheduler_sidecar,
        "alignment": arguments.alignment,
    }
    source_sha256 = _verify_sources(sources)
    source_orders = _alignment_source_orders(arguments.alignment)

    # The predictor receives only the canonical duration-free compact case.
    predictor, predictor_source_sha256 = _predict_steps(
        arguments.repository_root,
        arguments.alignment,
        load_compact_case,
    )

    # Historical durations remain in this evaluation-only reduction branch.
    events = _load_rank_zero_events(arguments.nsys_json)
    syncs = _load_syncs(arguments.sync_boundaries)
    scheduler_records = _load_jsonl(arguments.scheduler_sidecar)
    scheduler = _load_non_idle_scheduler_records(scheduler_records)
    raw_metadata, raw_steps = _partition_raw_steps(
        events, syncs, scheduler, source_orders
    )
    case = load_compact_case(arguments.alignment.resolve().parent)
    production_steps, production_summary = _normalize_historical_production_steps(
        scheduler_records, case
    )
    profiled_rows, prefix_coverage = _profiled_prefix_windows(
        events, syncs, production_steps
    )
    trajectory_predictions, prefix_predictor_source = _predict_profiled_prefix(
        arguments.repository_root, production_steps, case
    )
    prefix_predictions = trajectory_predictions[: len(profiled_rows)]
    combined_prefix_rows = [
        {**row, **prediction}
        for row, prediction in zip(
            profiled_rows, prefix_predictions, strict=True
        )
    ]
    result = {
        "schema_version": 2,
        "cell_id": CELL_ID,
        "status": "historical_cross_version_profiler_diagnostic",
        "source_sha256": source_sha256,
        "predictor_source_sha256": predictor_source_sha256,
        "forward_duration_observation_consumed": False,
        "raw_structural_prefix": raw_metadata,
        "steps": _combine_results(predictor, raw_steps),
        "profiled_trajectory_prefix": {
            "status": "historical_profiler_mechanism_only",
            "formal_request_metrics_allowed": False,
            "scheduler": production_summary,
            "coverage": prefix_coverage,
            "predictor_source": prefix_predictor_source,
            "groups": _summarize_profiled_prefix(combined_prefix_rows),
            "predicted_request_metrics_on_historical_batch_trajectory": (
                _replay_historical_trajectory_request_metrics(
                    production_steps, trajectory_predictions, case
                )
            ),
            "interpretation": (
                "The profiler perturbs device/runtime timing and the final three "
                "Decode steps lie outside the captured event prefix. Aggregate "
                "ratios are mechanism evidence only and cannot calibrate Forward."
            ),
        },
        "no_profiler_same_run": (
            {
                "status": "not_supplied",
                "reason": "an idle C500 capture has not been supplied",
            }
            if arguments.no_profiler_device_events is None
            else _summarize_no_profiler_same_run(
                arguments.no_profiler_device_events,
                arguments.no_profiler_device_events_sha256,
                arguments.no_profiler_scheduler_trace,
                arguments.no_profiler_scheduler_trace_sha256,
            )
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
