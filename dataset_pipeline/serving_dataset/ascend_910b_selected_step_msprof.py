"""Correlate 910B MSTX end marks with selected-step device tasks."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Iterable

from dataset_pipeline.serving_dataset.interval_ops import (
    clipped_union_ns,
    intersection_ns,
    merged_clipped_intervals,
    merged_intervals,
    subtract_intervals,
)
from dataset_pipeline.serving_dataset.json_shape import (
    interval_union_ns,
    file_sha256,
)


_MARKER_KEYS = {
    "schema_version",
    "source",
    "process_pid",
    "worker_rank",
    "marker_ordinal",
    "trace_step_id",
    "stream_handle",
    "mark_issue_start_ns",
    "mark_issue_end_ns",
}
_LEGACY_MARKER_KEYS = {
    "marker_ordinal",
    "process_pid",
    "schema_version",
    "source",
    "trace_step_id",
    "worker_rank",
}

_EVENT_BOUNDARY_KEYS_V1 = {
    "schema_version",
    "source",
    "process_pid",
    "worker_rank",
    "marker_ordinal",
    "trace_step_id",
    "stream_handle",
    "event_handle",
    "event_issue_start_ns",
    "event_issue_end_ns",
    "event_issue_start_wall_ns",
    "event_issue_end_wall_ns",
}

_EVENT_BOUNDARY_KEYS_V2 = _EVENT_BOUNDARY_KEYS_V1 | {
    "worker_execution_start_ns",
    "worker_execution_end_ns",
    "worker_execution_clock",
    "worker_thread_cpu_start_ns",
    "worker_thread_cpu_end_ns",
    "worker_thread_cpu_clock",
}

_FRAMEWORK_TIMING_KEYS = {
    "schema_version",
    "source",
    "process_pid",
    "worker_rank",
    "trace_step_id",
    "thread_cpu_clock",
    "wall_clock",
    "nesting_semantics",
    "torch_dispatch_count_by_source",
    "records",
}
_FRAMEWORK_TIMING_RECORD_KEYS = {
    "call_count",
    "exclusive_thread_cpu_ns",
    "exclusive_wall_ns",
    "inclusive_thread_cpu_ns",
    "inclusive_wall_ns",
    "maximum_wall_ns",
    "minimum_wall_ns",
}
_MODEL_RUNNER_METADATA_RECORDS = (
    "model_runner_attention_metadata",
    "model_runner_update_cos_sin",
    "model_runner_state_update",
    "model_runner_forward_context_enter",
    "model_runner_forward_context_exit",
    "model_runner_batch_descriptor",
    "model_runner_preprocess",
    "model_runner_ubatch_slices",
)




def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _framework_timing_record(
    records: object, name: str,
) -> dict[str, int]:
    if not isinstance(records, dict) or name not in records:
        raise ValueError(f"framework timing lacks required record: {name}")
    record = records[name]
    if not isinstance(record, dict) or set(record) != _FRAMEWORK_TIMING_RECORD_KEYS:
        raise ValueError(f"framework timing record schema disagrees: {name}")
    normalized = {}
    for field, value in record.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"framework timing record is invalid: {name}.{field}")
        normalized[field] = value
    if normalized["call_count"] < 1:
        raise ValueError(f"framework timing record has no calls: {name}")
    if normalized["inclusive_wall_ns"] < normalized["exclusive_wall_ns"]:
        raise ValueError(f"framework timing wall nesting is invalid: {name}")
    if normalized["inclusive_thread_cpu_ns"] < normalized["exclusive_thread_cpu_ns"]:
        raise ValueError(f"framework timing CPU nesting is invalid: {name}")
    return normalized


def _model_runner_outer_timing(
    profile_path: Path,
    expected_pid: int,
    expected_rank: int,
    expected_step_id: str,
    worker_execution: dict[str, object],
) -> dict[str, object]:
    """Partition a same-run worker span around ``NPUModelRunner`` source scopes.

    This is an evaluation-only source timing observation.  It deliberately
    does not create a Forward parameter: independent mechanism microbenchmarks
    remain required before any component can enter a Host timing profile.
    """

    profile = _read_json(profile_path)
    if set(profile) != _FRAMEWORK_TIMING_KEYS:
        raise ValueError("framework timing sidecar schema disagrees")
    if (
        profile["schema_version"] != 2
        or profile["source"] != "selected_function_aggregate_timing"
        or profile["process_pid"] != expected_pid
        or profile["worker_rank"] != expected_rank
        or profile["trace_step_id"] != expected_step_id
        or profile["thread_cpu_clock"] != "time.thread_time_ns"
        or profile["wall_clock"] != "time.perf_counter_ns"
        or profile["nesting_semantics"]
        != "exclusive subtracts only nested instrumented functions"
        or not isinstance(profile["torch_dispatch_count_by_source"], dict)
    ):
        raise ValueError("framework timing sidecar identity disagrees")
    if worker_execution["status"] != "observed":
        raise ValueError(
            "framework timing requires a same-run worker execution boundary")
    worker_span_ns = worker_execution["wall_span_ns"]
    worker_thread_cpu_span_ns = worker_execution["thread_cpu_span_ns"]
    if (isinstance(worker_span_ns, bool) or not isinstance(worker_span_ns, int)
            or isinstance(worker_thread_cpu_span_ns, bool)
            or not isinstance(worker_thread_cpu_span_ns, int)
            or worker_span_ns < 0 or worker_thread_cpu_span_ns < 0):
        raise ValueError("worker execution span is invalid")

    records = profile["records"]
    execute = _framework_timing_record(records, "model_runner_execute")
    forward = _framework_timing_record(records, "model_forward")
    input_prepare = _framework_timing_record(
        records, "model_runner_input_prepare")
    logits_prepare = _framework_timing_record(
        records, "model_runner_compute_logits")
    metadata_records = {
        name: _framework_timing_record(records, name)
        for name in _MODEL_RUNNER_METADATA_RECORDS
    }
    if execute["inclusive_wall_ns"] < forward["inclusive_wall_ns"]:
        raise ValueError("ModelRunner execute timing is shorter than model forward")
    if worker_span_ns < execute["inclusive_wall_ns"]:
        raise ValueError("worker execution is shorter than ModelRunner execution")
    if worker_thread_cpu_span_ns < execute["inclusive_thread_cpu_ns"]:
        raise ValueError(
            "worker CPU execution is shorter than ModelRunner CPU execution")

    metadata_wall_ns = sum(
        record["inclusive_wall_ns"] for record in metadata_records.values())
    components = {
        "input_prepare_ns": input_prepare["inclusive_wall_ns"],
        "logits_prepare_ns": logits_prepare["inclusive_wall_ns"],
        "forward_metadata_ns": metadata_wall_ns,
        "model_runner_uninstrumented_control_ns": execute["exclusive_wall_ns"],
    }
    model_runner_outer_ns = (
        execute["inclusive_wall_ns"] - forward["inclusive_wall_ns"]
    )
    component_sum_ns = sum(components.values())
    if component_sum_ns != model_runner_outer_ns:
        raise ValueError(
            "ModelRunner outer timing does not equal its exclusive source "
            "component partition")
    executor_outer_ns = worker_span_ns - execute["inclusive_wall_ns"]
    executor_outer_thread_cpu_ns = (
        worker_thread_cpu_span_ns - execute["inclusive_thread_cpu_ns"])
    executor_outer_non_cpu_wait_ns = (
        executor_outer_ns - executor_outer_thread_cpu_ns)
    if executor_outer_non_cpu_wait_ns < 0:
        raise ValueError(
            "worker outer wall time is shorter than its measured CPU service")
    return {
        "status": "observed_same_run_source_function_timing",
        "production_forward_eligible": False,
        "reason": (
            "selected Serving source durations are diagnostic evidence only; "
            "independent mechanism microbenchmarks are required for Forward"
        ),
        "inputs": {
            "path": str(profile_path),
            "sha256": file_sha256(profile_path),
        },
        "model_forward_wall_ns": forward["inclusive_wall_ns"],
        "model_runner_outer_wall_ns": model_runner_outer_ns,
        "model_runner_outer_components_ns": components,
        "executor_outer_wall_ns": executor_outer_ns,
        "executor_outer_thread_cpu_ns": executor_outer_thread_cpu_ns,
        "executor_outer_non_cpu_wait_ns": executor_outer_non_cpu_wait_ns,
        "worker_execution_wall_ns": worker_span_ns,
        "worker_execution_thread_cpu_ns": worker_thread_cpu_span_ns,
    }


def summarize_ascend_910b_model_runner_outer_timing(
    framework_timing_path: Path,
    event_boundary_path: Path,
) -> dict[str, object]:
    """Validate and summarize one exact worker's source timing sidecar.

    The boundary sidecar supplies the outer worker envelope.  The framework
    sidecar supplies nested ``NPUModelRunner`` source scopes.  Keeping both
    inputs explicit prevents a source aggregate from being joined to another
    rank, process, or injected step by filename convention alone.
    """

    framework = _read_json(framework_timing_path)
    for field in ("process_pid", "worker_rank", "trace_step_id"):
        if field not in framework:
            raise ValueError(f"framework timing sidecar lacks {field}")
    process_pid = framework["process_pid"]
    worker_rank = framework["worker_rank"]
    trace_step_id = framework["trace_step_id"]
    if (
        isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid < 1
        or isinstance(worker_rank, bool)
        or not isinstance(worker_rank, int)
        or worker_rank < 0
        or not isinstance(trace_step_id, str)
        or not trace_step_id
    ):
        raise ValueError("framework timing sidecar identity is invalid")
    boundaries = _event_boundary_rows(event_boundary_path, worker_rank)
    matches = [
        row for row in boundaries
        if row["process_pid"] == process_pid
        and row["trace_step_id"] == trace_step_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "framework timing requires exactly one matching event boundary")
    boundary = matches[0]
    if boundary["schema_version"] != 2:
        raise ValueError(
            "framework timing requires an event boundary with worker clocks")
    worker_start_ns = boundary["worker_execution_start_ns"]
    worker_end_ns = boundary["worker_execution_end_ns"]
    worker_thread_cpu_start_ns = boundary["worker_thread_cpu_start_ns"]
    worker_thread_cpu_end_ns = boundary["worker_thread_cpu_end_ns"]
    if (worker_end_ns < worker_start_ns
            or worker_thread_cpu_end_ns < worker_thread_cpu_start_ns):
        raise ValueError("worker execution boundary is negative")
    worker_execution = {
        "status": "observed",
        "wall_span_ns": worker_end_ns - worker_start_ns,
        "thread_cpu_span_ns": (
            worker_thread_cpu_end_ns - worker_thread_cpu_start_ns),
    }
    result = _model_runner_outer_timing(
        framework_timing_path,
        process_pid,
        worker_rank,
        trace_step_id,
        worker_execution,
    )
    return {
        **result,
        "trace_step_id": trace_step_id,
        "worker_rank": worker_rank,
        "process_pid": process_pid,
        "event_boundary": {
            "path": str(event_boundary_path),
            "sha256": file_sha256(event_boundary_path),
        },
    }


def _require_columns(
    connection: sqlite3.Connection, table: str, required: set[str]
) -> None:
    columns = {
        row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"910B table {table} is missing columns: {missing}")




def _interval_complement_rows(
    intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int,
) -> list[tuple[int, int]]:
    if end_ns < start_ns:
        raise ValueError("interval complement ends before it starts")
    clipped = merged_clipped_intervals(intervals, start_ns, end_ns)
    gaps = []
    cursor = start_ns
    for start, end in clipped:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < end_ns:
        gaps.append((cursor, end_ns))
    return gaps


def _exclusive_overlap_breakdown(
    intervals: Iterable[tuple[int, int]],
    previous_step_intervals: Iterable[tuple[int, int]],
    selected_device_intervals: Iterable[tuple[int, int]],
) -> dict[str, int]:
    category_intervals = merged_intervals(intervals)
    previous_overlap_ns = intersection_ns(
        category_intervals, previous_step_intervals)
    after_previous = subtract_intervals(
        category_intervals, previous_step_intervals)
    selected_device_overlap_ns = intersection_ns(
        after_previous, selected_device_intervals)
    exposed_intervals = subtract_intervals(
        after_previous, selected_device_intervals)
    result = {
        "total_ns": interval_union_ns(category_intervals),
        "overlapped_by_previous_step_ns": previous_overlap_ns,
        "overlapped_by_selected_device_ns": selected_device_overlap_ns,
        "exposed_device_idle_ns": interval_union_ns(exposed_intervals),
    }
    if result["total_ns"] != (
        result["overlapped_by_previous_step_ns"]
        + result["overlapped_by_selected_device_ns"]
        + result["exposed_device_idle_ns"]
    ):
        raise AssertionError("exclusive Host overlap breakdown does not conserve time")
    return result


def _frame_summary(frame: object) -> tuple[str, str, int | None]:
    if not isinstance(frame, dict):
        raise ValueError("sampled stack frame must be an object")
    name = frame["name"]
    filename = frame["file"]
    line = frame["line"]
    if (
        not isinstance(name, str)
        or not isinstance(filename, str)
        or isinstance(line, bool)
        or (line is not None and (not isinstance(line, int) or line < 0))
    ):
        raise ValueError("sampled stack frame identity is invalid")
    return name, filename, line


def _sample_group_rows(
    weighted_stacks: Iterable[tuple[int, list[dict[str, object]]]],
    selector: str,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int | None], dict[str, object]] = {}
    for duration_ns, stack in weighted_stacks:
        if selector == "leaf":
            frame = stack[-1]
        elif selector == "vllm_callsite":
            candidates = [
                frame for frame in stack
                if (
                    "/vllm-workspace/vllm/" in str(frame["file"])
                    or "/vllm-workspace/vllm-ascend/" in str(frame["file"])
                )
                and "vllm_scheduler_trace.py" not in str(frame["file"])
            ]
            frame = candidates[-1] if candidates else stack[-1]
        else:
            raise ValueError(f"unsupported stack grouping selector: {selector}")
        name, filename, line = _frame_summary(frame)
        key = (name, filename, line)
        if key not in groups:
            groups[key] = {
                "name": name,
                "file": filename,
                "line": line,
                "sampled_ns": 0,
                "sample_count": 0,
            }
        groups[key]["sampled_ns"] += duration_ns
        groups[key]["sample_count"] += 1
    return sorted(
        groups.values(),
        key=lambda row: (
            -row["sampled_ns"], row["file"],
            -1 if row["line"] is None else row["line"],
        ),
    )[:30]


def _stack_sample_interval_attribution(
    profile_path: Path,
    expected_pid: int,
    expected_rank: int,
    expected_step_id: str,
    intervals: Iterable[tuple[int, int]],
) -> dict[str, object]:
    profile = _read_json(profile_path)
    expected_keys = {
        "schema_version",
        "source",
        "process_pid",
        "worker_rank",
        "worker_thread_native_id",
        "trace_step_id",
        "sample_interval_s",
        "worker_call_start_wall_ns",
        "worker_call_end_wall_ns",
        "samples",
    }
    if set(profile) != expected_keys:
        raise ValueError("selected-step stack-sample schema disagrees")
    if (
        profile["schema_version"] != 1
        or profile["source"] != "python_current_frames_statistical_sampling"
        or profile["process_pid"] != expected_pid
        or profile["worker_rank"] != expected_rank
        or profile["trace_step_id"] != expected_step_id
        or profile["sample_interval_s"] != 0.001
    ):
        raise ValueError("selected-step stack-sample identity disagrees")
    start_ns = profile["worker_call_start_wall_ns"]
    end_ns = profile["worker_call_end_wall_ns"]
    if (
        isinstance(start_ns, bool)
        or not isinstance(start_ns, int)
        or isinstance(end_ns, bool)
        or not isinstance(end_ns, int)
        or end_ns <= start_ns
    ):
        raise ValueError("selected-step sampled worker interval is invalid")
    target_intervals = merged_intervals(intervals)
    target_ns = interval_union_ns(target_intervals)
    matching_stacks = []
    previous_wall_ns = 0
    for sample in profile["samples"]:
        if not isinstance(sample, dict) or set(sample) != {"wall_ns", "stack"}:
            raise ValueError("selected-step stack sample is invalid")
        wall_ns = sample["wall_ns"]
        stack = sample["stack"]
        if (
            isinstance(wall_ns, bool)
            or not isinstance(wall_ns, int)
            or wall_ns <= previous_wall_ns
            or not isinstance(stack, list)
            or not stack
        ):
            raise ValueError("selected-step stack sample fields are invalid")
        previous_wall_ns = wall_ns
        for frame in stack:
            _frame_summary(frame)
        if any(begin_ns <= wall_ns < finish_ns for begin_ns, finish_ns in target_intervals):
            matching_stacks.append(stack)
    weighted_stacks = []
    if matching_stacks:
        base_weight_ns, remainder_ns = divmod(target_ns, len(matching_stacks))
        for index, stack in enumerate(matching_stacks):
            weighted_stacks.append((
                base_weight_ns + (1 if index < remainder_ns else 0),
                stack,
            ))
    return {
        "clock_domain": "msprof_host_wall_ns",
        "attribution": (
            "exact target interval from msprof; duration is distributed by "
            "the fraction of 1 ms statistical Python-stack samples"
        ),
        "target_interval_ns": target_ns,
        "matching_sample_count": len(matching_stacks),
        "total_sample_count": len(profile["samples"]),
        "profiled_worker_call_ns": end_ns - start_ns,
        "top_leaf_frames": _sample_group_rows(weighted_stacks, "leaf"),
        "top_vllm_callsites": _sample_group_rows(
            weighted_stacks, "vllm_callsite"),
        "inputs": {
            "profile": str(profile_path),
            "profile_sha256": file_sha256(profile_path),
        },
    }


def _event_boundary_rows(
    path: Path, expected_rank: int,
) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    for ordinal, row in enumerate(rows):
        schema_version = row["schema_version"]
        expected_keys = (
            _EVENT_BOUNDARY_KEYS_V1
            if schema_version == 1
            else _EVENT_BOUNDARY_KEYS_V2
            if schema_version == 2
            else None
        )
        if expected_keys is None or set(row) != expected_keys:
            raise ValueError(f"910B Event boundary schema disagrees in {path}")
        if (
            row["source"] != "ascend_aclrt_event_record_boundary"
        ):
            raise ValueError(f"910B Event boundary provenance disagrees in {path}")
        if row["worker_rank"] != expected_rank:
            raise ValueError(f"910B Event boundary rank disagrees in {path}")
        if row["marker_ordinal"] != ordinal:
            raise ValueError(f"910B Event boundary ordinals are not contiguous in {path}")
        for key in (
            "process_pid",
            "stream_handle",
            "event_handle",
            "event_issue_start_ns",
            "event_issue_end_ns",
            "event_issue_start_wall_ns",
            "event_issue_end_wall_ns",
        ):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"910B Event boundary {key} is invalid in {path}")
        if (
            row["event_issue_end_ns"] < row["event_issue_start_ns"]
            or row["event_issue_end_wall_ns"] < row["event_issue_start_wall_ns"]
        ):
            raise ValueError(f"910B Event boundary interval is invalid in {path}")
        if schema_version == 2:
            if (
                row["worker_execution_clock"] != "time.perf_counter_ns"
                or row["worker_thread_cpu_clock"] != "time.thread_time_ns"
            ):
                raise ValueError(
                    f"910B worker execution clock disagrees in {path}")
            for key in (
                "worker_execution_start_ns",
                "worker_execution_end_ns",
                "worker_thread_cpu_start_ns",
                "worker_thread_cpu_end_ns",
            ):
                value = row[key]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(
                        f"910B Event boundary {key} is invalid in {path}")
            if (
                row["worker_execution_end_ns"] < row["worker_execution_start_ns"]
                or row["worker_thread_cpu_end_ns"]
                < row["worker_thread_cpu_start_ns"]
            ):
                raise ValueError(
                    f"910B worker execution interval is invalid in {path}")
            if (
                row["worker_thread_cpu_end_ns"]
                - row["worker_thread_cpu_start_ns"]
                > row["worker_execution_end_ns"]
                - row["worker_execution_start_ns"]
            ):
                raise ValueError(
                    f"910B worker thread CPU exceeds wall time in {path}")
        if not isinstance(row["trace_step_id"], str):
            raise ValueError(f"910B Event boundary step ID is invalid in {path}")
    return rows


def _interval_distance_ns(
    first_start_ns: int,
    first_end_ns: int,
    second_start_ns: int,
    second_end_ns: int,
) -> int:
    if first_end_ns < second_start_ns:
        return second_start_ns - first_end_ns
    if second_end_ns < first_start_ns:
        return first_start_ns - second_end_ns
    return 0


def _planned_dispatch_counts(
    replay: dict[str, object], selected_step_id: str, expected_rank_count: int,
) -> tuple[dict[int, int], dict[int, int], str]:
    comparisons = replay["step_comparison"]
    if not isinstance(comparisons, list):
        raise TypeError("selected-step replay comparison must be a list")
    selected = [row for row in comparisons if row["step_id"] == selected_step_id]
    if len(selected) != 1:
        raise ValueError("selected-step replay must contain the selected step exactly once")
    selected_comparison = selected[0]
    graph_execution = selected_comparison["step_fingerprint"]["graph_execution"]
    runtime_graph_mode = graph_execution["runtime_mode"]
    if runtime_graph_mode != "NONE":
        raise ValueError(
            "physical per-launch correlation requires an eager runtime step"
        )
    contributions = selected_comparison["predicted_kernel_contributions"]
    if not isinstance(contributions, list):
        raise TypeError("predicted kernel contributions must be a list")
    timeline_counts = {rank: 0 for rank in range(expected_rank_count)}
    participant_counts = {rank: 0 for rank in range(expected_rank_count)}
    for contribution in contributions:
        rank = contribution["rank"]
        count = contribution["occurrence_count"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in timeline_counts
        ):
            raise ValueError("predicted contribution rank is invalid")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("predicted contribution occurrence count is invalid")
        timeline_counts[rank] += count
        kernel_spec = contribution["kernel_spec"]
        if kernel_spec["kind"] == "COMMUNICATION":
            transfer = kernel_spec["transfer"]
            participants = transfer["participants"]
            if (
                not isinstance(participants, list)
                or not participants
                or len(set(participants)) != len(participants)
            ):
                raise ValueError("collective participants are invalid")
            for participant in participants:
                if participant not in participant_counts:
                    raise ValueError("collective participant rank is invalid")
                participant_counts[participant] += count
        else:
            participant_counts[rank] += count
    return timeline_counts, participant_counts, runtime_graph_mode


_FIA_KERNEL = "FusedInferAttentionScore"
_TENSOR_MOVE_KERNEL = "aclnnInplaceCopy_TensorMoveAiCore_TensorMove"
_MOE_ROUTING_SEQUENCE = (
    "MoeGatingTopK",
    "aclnnIndex_IndexAiCore_Index",
    "aclnnNeScalar_NotEqualAiCore_NotEqual",
    "aclnnMul_CastAiCore_Cast",
    "aclnnMul_MulAiCore_Mul",
    "aclnnMoeInitRoutingCustom_MoeInitRoutingCustom_MoeInitRoutingCustom",
)


def _source_sequence_classification(
    physical_launches: list[tuple[object, ...]],
    unmapped_excess_launches: int,
) -> dict[str, object]:
    """Classify source-backed repeated helpers without name-total guesses."""

    names = tuple(str(row[5]) for row in physical_launches)
    attention_positions = tuple(
        index for index, name in enumerate(names) if name == _FIA_KERNEL
    )
    attention_first_copies = sum(
        index + 1 < len(names) and names[index + 1] == _TENSOR_MOVE_KERNEL
        for index in attention_positions
    )
    attention_second_copies = sum(
        index + 2 < len(names)
        and names[index + 1] == _TENSOR_MOVE_KERNEL
        and names[index + 2] == _TENSOR_MOVE_KERNEL
        for index in attention_positions
    )
    tensor_move_launches = names.count(_TENSOR_MOVE_KERNEL)
    non_attention_tensor_moves = (
        tensor_move_launches
        - attention_first_copies
        - attention_second_copies
    )
    if non_attention_tensor_moves < 0:
        raise ValueError("Attention TensorMove sequence classification overlaps")

    sequence_length = len(_MOE_ROUTING_SEQUENCE)
    moe_routing_sequences = sum(
        names[index:index + sequence_length] == _MOE_ROUTING_SEQUENCE
        for index in range(len(names) - sequence_length + 1)
    )
    # GatingTopK, Cast and MoeInitRouting already exist in the current plan;
    # Index, NotEqual and Mul are the three source-visible missing helpers.
    attention_missing_launches = (
        attention_first_copies + attention_second_copies
    )
    moe_missing_launches = 3 * moe_routing_sequences
    layer_regular_missing_launches = (
        attention_missing_launches + moe_missing_launches
    )
    residual_unmapped_launches = (
        unmapped_excess_launches - layer_regular_missing_launches
    )
    if residual_unmapped_launches < 0:
        raise ValueError(
            "source-sequence candidates exceed the plan/physical count delta"
        )
    return {
        "method": "ordered_kernel_adjacency",
        "attention_occurrences": len(attention_positions),
        "attention_first_output_copies": attention_first_copies,
        "attention_second_output_copies": attention_second_copies,
        "attention_missing_launches": attention_missing_launches,
        "tensor_move_launches": tensor_move_launches,
        "non_attention_tensor_move_launches": non_attention_tensor_moves,
        "moe_routing_sequences": moe_routing_sequences,
        "moe_missing_launches": moe_missing_launches,
        "layer_regular_missing_launches": layer_regular_missing_launches,
        "residual_unmapped_launches": residual_unmapped_launches,
    }


def analyze_ascend_910b_selected_step_dispatches(
    run_dir: Path,
    validation_path: Path,
    replay_path: Path,
    selected_step_id: str,
    expected_rank_count: int,
) -> dict[str, object]:
    """Correlate one selected eager step's plan calls with physical launches.

    This consumes the flattened ``msprof_rank*.db`` capture layout emitted by
    the selected-step injection workflow.  It deliberately reports observed
    multiplicity without feeding profiler counts back into Forward.
    """

    root = run_dir.resolve(strict=True)
    validation_file = validation_path.resolve(strict=True)
    replay_file = replay_path.resolve(strict=True)
    if isinstance(expected_rank_count, bool) or expected_rank_count < 1:
        raise ValueError("expected rank count must be a positive integer")
    validation = _read_json(validation_file)
    if validation["injected_step_id"] != selected_step_id:
        raise ValueError("selected step disagrees with injection validation")
    if validation["structural_fingerprint_match"] is not True:
        raise ValueError("INVALID EXPERIMENT — structural_fingerprint_match")
    replay = _read_json(replay_file)
    timeline_counts, participant_counts, runtime_graph_mode = _planned_dispatch_counts(
        replay, selected_step_id, expected_rank_count
    )
    scheduler_trace_path = root / "scheduler_steps.raw.jsonl"
    scheduler_clock_path = Path(f"{scheduler_trace_path}.clock.json")
    scheduler_rows = _read_jsonl(scheduler_trace_path)
    scheduler_step_ids = [str(row["step_id"]) for row in scheduler_rows]
    if len(set(scheduler_step_ids)) != len(scheduler_step_ids):
        raise ValueError("Scheduler trace contains duplicate step IDs")
    if selected_step_id not in scheduler_step_ids:
        raise ValueError("Scheduler trace does not contain selected step")
    selected_index = scheduler_step_ids.index(selected_step_id)
    if selected_index == 0:
        raise ValueError("selected step has no preceding Scheduler step")
    selected_scheduler_row = scheduler_rows[selected_index]
    previous_scheduler_row = scheduler_rows[selected_index - 1]
    scheduler_clock = _read_json(scheduler_clock_path)
    if scheduler_clock["clock_domain"] != "shared_host_time.perf_counter_ns":
        raise ValueError("Scheduler trace clock domain is unsupported")
    scheduler_origin_ns = scheduler_clock["trace_origin_monotonic_ns"]
    if isinstance(scheduler_origin_ns, bool) or not isinstance(
        scheduler_origin_ns, int
    ):
        raise ValueError("Scheduler trace origin is invalid")
    selected_step_start_ns = int(selected_scheduler_row["step_start_ns"])
    selected_schedule_end_ns = int(selected_scheduler_row["schedule_end_ns"])
    selected_update_start_ns = int(selected_scheduler_row["update_start_ns"])
    selected_step_end_ns = int(selected_scheduler_row["step_end_ns"])
    previous_update_start_ns = int(previous_scheduler_row["update_start_ns"])
    serialized_service_start_ns = max(
        selected_schedule_end_ns, previous_update_start_ns)
    serialized_service_ns = (
        selected_update_start_ns - serialized_service_start_ns
    )
    if serialized_service_ns <= 0:
        raise ValueError("Scheduler trace has non-positive serialized service")
    previous_step_overlap_ns = max(
        0,
        min(previous_update_start_ns, selected_update_start_ns)
        - selected_step_start_ns,
    )

    profiles = []
    seen_pids = set()
    for rank in range(expected_rank_count):
        database = root / f"msprof_rank{rank}.db"
        if not database.is_file() or database.stat().st_size == 0:
            raise ValueError(f"910B combined profiler database is missing: {database}")
        boundary_paths = tuple(root.glob(
            f"scheduler_steps.raw.jsonl.ascend-event-boundary.rank-{rank}.pid-*.jsonl"
        ))
        if len(boundary_paths) != 1:
            raise ValueError(f"910B rank {rank} must have exactly one Event sidecar")
        boundary_path = boundary_paths[0]
        boundary_rows = _event_boundary_rows(boundary_path, rank)
        selected_boundaries = [
            row for row in boundary_rows if row["trace_step_id"] == selected_step_id
        ]
        if len(selected_boundaries) != 1:
            raise ValueError("Event sidecar must contain the selected step exactly once")
        boundary = selected_boundaries[0]
        pid = boundary["process_pid"]
        if pid in seen_pids:
            raise ValueError("910B Event sidecars must identify distinct worker PIDs")
        seen_pids.add(pid)

        uri = f"file:{database}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            for table, columns in (
                ("TASK", {"startNs", "endNs", "connectionId", "globalTaskId", "taskType"}),
                ("CANN_API", {"startNs", "endNs", "type", "globalTid", "connectionId", "name"}),
                ("COMPUTE_TASK_INFO", {"globalTaskId", "name"}),
                ("STRING_IDS", {"id", "value"}),
            ):
                _require_columns(connection, table, columns)
            marker_candidates = connection.execute(
                "SELECT a.startNs, a.endNs, a.connectionId, a.globalTid "
                "FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id "
                "WHERE s.value='aclrtRecordEvent' ORDER BY a.startNs"
            ).fetchall()
            ranked_candidates = sorted(
                (
                    _interval_distance_ns(
                        start_ns,
                        end_ns,
                        boundary["event_issue_start_wall_ns"],
                        boundary["event_issue_end_wall_ns"],
                    ),
                    start_ns,
                    end_ns,
                    connection_id,
                    global_tid,
                )
                for start_ns, end_ns, connection_id, global_tid in marker_candidates
                if global_tid >> 32 == pid
            )
            if not ranked_candidates or ranked_candidates[0][0] > 5_000_000:
                raise ValueError("selected Event boundary has no nearby aclrtRecordEvent")
            if (
                len(ranked_candidates) > 1
                and ranked_candidates[0][0] == ranked_candidates[1][0]
            ):
                raise ValueError("selected Event boundary matches multiple aclrtRecordEvent calls")
            (
                marker_error_ns,
                marker_api_start_ns,
                marker_api_end_ns,
                marker_connection_id,
                marker_global_tid,
            ) = ranked_candidates[0]
            marker_tasks = connection.execute(
                "SELECT startNs, endNs FROM TASK WHERE connectionId=? ORDER BY startNs",
                (marker_connection_id,),
            ).fetchall()
            if len(marker_tasks) != 1:
                raise ValueError("selected aclrtRecordEvent must map to exactly one TASK")

            node_launches = connection.execute(
                "SELECT a.startNs, a.endNs, a.connectionId, a.globalTid, "
                "c.globalTaskId, cs.value, ts.value, t.startNs, t.endNs "
                "FROM CANN_API a "
                "JOIN TASK t ON t.connectionId=a.connectionId "
                "LEFT JOIN COMPUTE_TASK_INFO c ON t.globalTaskId=c.globalTaskId "
                "LEFT JOIN STRING_IDS cs ON c.name=cs.id "
                "JOIN STRING_IDS ts ON t.taskType=ts.id "
                "WHERE a.type=10000 AND a.startNs<? ORDER BY a.startNs",
                (marker_api_start_ns,),
            ).fetchall()
            by_connection: dict[int, list[tuple[object, ...]]] = {}
            for row in node_launches:
                connection_id = row[2]
                if connection_id not in by_connection:
                    by_connection[connection_id] = []
                by_connection[connection_id].append(row)
            physical_launches = []
            for connection_id, rows in by_connection.items():
                primary = [
                    row for row in rows
                    if row[4] is not None or row[6] == "AI_CORE"
                ]
                if len(primary) != 1:
                    raise ValueError(
                        "each node launch must map to exactly one compute or collective TASK"
                    )
                row = primary[0]
                kind = "compute" if row[4] is not None else "collective"
                name = row[5] if row[4] is not None else "collective"
                physical_launches.append((
                    row[0], row[1], connection_id, row[3], kind, name,
                    row[7], row[8],
                ))
            host_tids = {row[3] for row in physical_launches}
            if len(host_tids) != 1 or next(iter(host_tids)) != marker_global_tid:
                raise ValueError("selected launches do not use one worker Host lane")
            host_intervals = merged_intervals(
                (row[0], row[1]) for row in physical_launches
            )
            host_program_start_ns = physical_launches[0][0]
            host_program_end_ns = physical_launches[-1][1]
            all_cann_intervals = merged_intervals(
                (max(start_ns, host_program_start_ns),
                 min(end_ns, host_program_end_ns))
                for start_ns, end_ns in connection.execute(
                    "SELECT startNs, endNs FROM CANN_API "
                    "WHERE endNs>? AND startNs<? ORDER BY startNs",
                    (host_program_start_ns, host_program_end_ns),
                )
            )
            host_non_launch_intervals = subtract_intervals(
                all_cann_intervals, host_intervals)
            host_outside_cann_intervals = _interval_complement_rows(
                all_cann_intervals,
                host_program_start_ns,
                host_program_end_ns,
            )
            host_launch_union_ns = sum(
                end_ns - start_ns for start_ns, end_ns in host_intervals
            )
            host_api_visible_union_ns = sum(
                end_ns - start_ns for start_ns, end_ns in all_cann_intervals
            )
            device_task_intervals = merged_intervals(
                (row[6], row[7]) for row in physical_launches
            )
            clipped_device_task_intervals = merged_clipped_intervals(
                device_task_intervals,
                host_program_start_ns,
                host_program_end_ns,
            )
            wall_offset_start_ns = (
                boundary["event_issue_start_wall_ns"]
                - boundary["event_issue_start_ns"]
            )
            wall_offset_end_ns = (
                boundary["event_issue_end_wall_ns"]
                - boundary["event_issue_end_ns"]
            )
            if abs(wall_offset_end_ns - wall_offset_start_ns) > 1_000_000:
                raise ValueError("910B Host wall/monotonic clock mapping is unstable")
            wall_minus_monotonic_ns = (
                wall_offset_start_ns + wall_offset_end_ns
            ) // 2
            previous_overlap_end_ns = (
                scheduler_origin_ns
                + min(previous_update_start_ns, selected_update_start_ns)
                + wall_minus_monotonic_ns
            )
            selected_start_wall_ns = (
                scheduler_origin_ns
                + selected_step_start_ns
                + wall_minus_monotonic_ns
            )
            previous_step_intervals = (
                [(selected_start_wall_ns, previous_overlap_end_ns)]
                if previous_overlap_end_ns > selected_start_wall_ns
                else []
            )
            host_breakdown = {
                "launch_api": _exclusive_overlap_breakdown(
                    host_intervals,
                    previous_step_intervals,
                    clipped_device_task_intervals,
                ),
                "non_launch_cann_api": _exclusive_overlap_breakdown(
                    host_non_launch_intervals,
                    previous_step_intervals,
                    clipped_device_task_intervals,
                ),
                "outside_cann_api_unknown": _exclusive_overlap_breakdown(
                    host_outside_cann_intervals,
                    previous_step_intervals,
                    clipped_device_task_intervals,
                ),
            }
            exposed_outside_cann_intervals = subtract_intervals(
                subtract_intervals(
                    host_outside_cann_intervals, previous_step_intervals),
                clipped_device_task_intervals,
            )
            host_breakdown_total_ns = {
                field: sum(category[field] for category in host_breakdown.values())
                for field in (
                    "total_ns",
                    "overlapped_by_previous_step_ns",
                    "overlapped_by_selected_device_ns",
                    "exposed_device_idle_ns",
                )
            }
            if host_breakdown_total_ns["total_ns"] != (
                host_program_end_ns - host_program_start_ns
            ):
                raise AssertionError("Host category breakdown does not cover its window")
            device_no_task_intervals = _interval_complement_rows(
                clipped_device_task_intervals,
                host_program_start_ns,
                host_program_end_ns,
            )
            device_task_union_ns = interval_union_ns(
                clipped_device_task_intervals)
            device_no_task_gap_ns = interval_union_ns(
                device_no_task_intervals)
            cann_api_in_device_gap_ns = intersection_ns(
                all_cann_intervals, device_no_task_intervals)
            launch_gap_groups: dict[tuple[str, str], dict[str, object]] = {}
            for previous, current in zip(
                physical_launches, physical_launches[1:], strict=False
            ):
                key = (str(previous[5]), str(current[5]))
                if key not in launch_gap_groups:
                    launch_gap_groups[key] = {
                        "previous_name": key[0],
                        "next_name": key[1],
                        "count": 0,
                        "additive_gap_ns": 0,
                    }
                launch_gap_groups[key]["count"] += 1
                launch_gap_groups[key]["additive_gap_ns"] += (
                    current[0] - previous[1]
                )
            kind_counts = {
                kind: sum(row[4] == kind for row in physical_launches)
                for kind in ("compute", "collective")
            }
            kernel_groups: dict[str, dict[str, object]] = {}
            for (
                start_ns,
                end_ns,
                _connection_id,
                _global_tid,
                kind,
                name,
                _device_start_ns,
                _device_end_ns,
            ) in physical_launches:
                key = f"{kind}:{name}"
                if key not in kernel_groups:
                    kernel_groups[key] = {
                        "kind": kind,
                        "name": name,
                        "count": 0,
                        "host_api_additive_ns": 0,
                    }
                kernel_groups[key]["count"] += 1
                kernel_groups[key]["host_api_additive_ns"] += end_ns - start_ns

        observed_count = len(physical_launches)
        planned_count = participant_counts[rank]
        unmapped_excess_launches = observed_count - planned_count
        if observed_count == planned_count:
            dispatch_mapping_status = "complete"
        elif observed_count > planned_count:
            dispatch_mapping_status = "incomplete_observed_excess"
        else:
            dispatch_mapping_status = "incomplete_planned_excess"
        if boundary["schema_version"] == 1:
            worker_execution = {
                "status": "unobserved_legacy_sidecar",
            }
        else:
            worker_execution_span_ns = (
                boundary["worker_execution_end_ns"]
                - boundary["worker_execution_start_ns"]
            )
            worker_thread_cpu_ns = (
                boundary["worker_thread_cpu_end_ns"]
                - boundary["worker_thread_cpu_start_ns"]
            )
            worker_execution = {
                "status": "observed",
                "clock_domain": boundary["worker_execution_clock"],
                "thread_cpu_clock": boundary["worker_thread_cpu_clock"],
                "wall_span_ns": worker_execution_span_ns,
                "thread_cpu_ns": worker_thread_cpu_ns,
                "non_cpu_wall_ns": worker_execution_span_ns - worker_thread_cpu_ns,
            }
        framework_timing_path = root / (
            f"{scheduler_trace_path.name}.framework-timing."
            f"rank-{rank}.pid-{pid}.json"
        )
        if framework_timing_path.exists():
            framework_timing = _model_runner_outer_timing(
                framework_timing_path,
                pid,
                rank,
                selected_step_id,
                worker_execution,
            )
        else:
            framework_timing = {"status": "not_collected"}
        stack_sample_path = root / (
            f"{scheduler_trace_path.name}.stack-sample-cpu-profile."
            f"rank-{rank}.pid-{pid}.json"
        )
        if stack_sample_path.exists():
            stack_sample_attribution = {
                "status": "observed_same_run_statistical_attribution",
                "exposed_outside_cann_api": _stack_sample_interval_attribution(
                    stack_sample_path,
                    pid,
                    rank,
                    selected_step_id,
                    exposed_outside_cann_intervals,
                ),
            }
        else:
            stack_sample_attribution = {"status": "not_collected"}
        profiles.append({
            "rank": rank,
            "process_pid": pid,
            "host_thread_id": marker_global_tid & 0xFFFFFFFF,
            "planned_timeline_occurrences": timeline_counts[rank],
            "planned_participant_expanded_occurrences": planned_count,
            "observed_physical_launches": observed_count,
            "observed_compute_launches": kind_counts["compute"],
            "observed_collective_launches": kind_counts["collective"],
            "planned_to_observed_delta_launches": unmapped_excess_launches,
            "unmapped_excess_launches": unmapped_excess_launches,
            "dispatch_mapping_status": dispatch_mapping_status,
            "source_sequence_classification": _source_sequence_classification(
                physical_launches,
                unmapped_excess_launches,
            ),
            "worker_execution": worker_execution,
            "framework_timing": framework_timing,
            "stack_sample_attribution": stack_sample_attribution,
            "host_launch_additive_ns": sum(row[1] - row[0] for row in physical_launches),
            "host_launch_union_ns": host_launch_union_ns,
            "host_program_start_ns": host_program_start_ns,
            "host_program_end_ns": host_program_end_ns,
            "host_program_span_ns": host_program_end_ns - host_program_start_ns,
            "host_api_visible_union_ns": host_api_visible_union_ns,
            "host_non_launch_api_union_ns": (
                host_api_visible_union_ns - host_launch_union_ns
            ),
            "host_outside_profiler_api_ns": (
                host_program_end_ns - host_program_start_ns
                - host_api_visible_union_ns
            ),
            "host_overlap_breakdown": {
                "semantics": (
                    "exclusive partition: previous-step pipeline overlap first, "
                    "then selected-step Device-TASK overlap, then exposed Device idle"
                ),
                "categories": host_breakdown,
                "total": host_breakdown_total_ns,
            },
            "device_task_union_ns": device_task_union_ns,
            "device_no_task_gap_ns": device_no_task_gap_ns,
            "cann_api_in_device_gap_ns": cann_api_in_device_gap_ns,
            "outside_cann_api_in_device_gap_ns": (
                device_no_task_gap_ns - cann_api_in_device_gap_ns
            ),
            "observed_timeline": {
                "schema_version": 1,
                "clock_domain": "msprof_host_wall_ns",
                "window_semantics": (
                    "first_selected_launch_api_start_to_"
                    "last_selected_launch_api_end"
                ),
                "window_start_ns": host_program_start_ns,
                "window_end_ns": host_program_end_ns,
                "physical_dispatch_events": tuple({
                    "ordinal": ordinal,
                    "kind": row[4],
                    "name": row[5],
                    "connection_id": row[2],
                    "host_api_start_ns": row[0],
                    "host_api_end_ns": row[1],
                    "device_task_start_ns": row[6],
                    "device_task_end_ns": row[7],
                } for ordinal, row in enumerate(physical_launches)),
                "device_task_union_intervals": tuple({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                } for start_ns, end_ns in clipped_device_task_intervals),
                "device_no_task_intervals": tuple({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                } for start_ns, end_ns in device_no_task_intervals),
                "cann_api_union_intervals": tuple({
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                } for start_ns, end_ns in all_cann_intervals),
            },
            "marker_api_connection_id": marker_connection_id,
            "marker_api_start_ns": marker_api_start_ns,
            "marker_api_end_ns": marker_api_end_ns,
            "marker_device_start_ns": marker_tasks[0][0],
            "marker_device_end_ns": marker_tasks[0][1],
            "marker_sidecar_distance_ns": marker_error_ns,
            "kernel_groups": sorted(
                kernel_groups.values(),
                key=lambda group: (-group["count"], group["name"]),
            ),
            "launch_gap_groups": sorted(
                launch_gap_groups.values(),
                key=lambda group: (
                    -group["additive_gap_ns"],
                    group["previous_name"],
                    group["next_name"],
                ),
            ),
            "inputs": {
                "database": str(database),
                "database_sha256": file_sha256(database),
                "event_sidecar": str(boundary_path),
                "event_sidecar_sha256": file_sha256(boundary_path),
            },
        })
    mapping_complete = all(
        profile["dispatch_mapping_status"] == "complete"
        for profile in profiles
    )
    return {
        "schema_version": 7,
        "kind": "ascend_910b_selected_step_dispatch_correlation",
        "comparison_status": "valid_same_run_dispatch_correlation",
        "production_dispatch_mapping_complete": mapping_complete,
        "production_host_timing_eligibility": (
            "eligible" if mapping_complete
            else "ineligible_incomplete_physical_dispatch_mapping"
        ),
        "canonical_step_id": validation["canonical_step_id"],
        "selected_step_id": selected_step_id,
        "structural_fingerprint_match": True,
        "runtime_graph_mode": runtime_graph_mode,
        "clock_domain": "msprof_host_wall_ns",
        "scheduler_pipeline_breakdown": {
            "clock_domain": "shared_host_time.perf_counter_ns",
            "selected_step_full_ns": selected_step_end_ns - selected_step_start_ns,
            "selected_scheduler_plan_ns": (
                selected_schedule_end_ns - selected_step_start_ns
            ),
            "overlapped_by_previous_step_ns": previous_step_overlap_ns,
            "serialized_service_ns": serialized_service_ns,
            "selected_scheduler_update_ns": (
                selected_step_end_ns - selected_update_start_ns
            ),
        },
        "profiles": profiles,
        "inputs": {
            "validation": str(validation_file),
            "validation_sha256": file_sha256(validation_file),
            "replay": str(replay_file),
            "replay_sha256": file_sha256(replay_file),
        },
    }


def _profile_identity(info_path: Path) -> tuple[int, int]:
    info = _read_json(info_path)
    pid_text = info["pid"]
    if not isinstance(pid_text, str) or not pid_text.isdecimal():
        raise ValueError("910B PROF pid must be a decimal string")
    device_info = info["DeviceInfo"]
    if not isinstance(device_info, list) or len(device_info) != 1:
        raise ValueError("910B PROF must identify exactly one device")
    device = device_info[0]
    if not isinstance(device, dict):
        raise ValueError("910B PROF device identity must be an object")
    device_id = device["id"]
    if isinstance(device_id, bool) or not isinstance(device_id, int):
        raise ValueError("910B PROF device ID must be an integer")
    return int(pid_text), device_id


def _marker_rows(path: Path, expected_rank: int, expected_pid: int) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    for ordinal, row in enumerate(rows):
        if row["schema_version"] == 3:
            if set(row) != _MARKER_KEYS or row["source"] != "ascend_mstx_device_end_mark":
                raise ValueError(f"910B MSTX marker schema/provenance disagrees in {path}")
        elif row["schema_version"] == 2:
            if set(row) != _LEGACY_MARKER_KEYS or row["source"] != "ascend_aclprof_step_timestamp":
                raise ValueError(f"910B legacy marker schema/provenance disagrees in {path}")
        else:
            raise ValueError(f"910B marker schema disagrees in {path}")
        if row["process_pid"] != expected_pid or row["worker_rank"] != expected_rank:
            raise ValueError(f"910B MSTX marker process/rank disagrees in {path}")
        if row["marker_ordinal"] != ordinal:
            raise ValueError(f"910B MSTX marker ordinals are not contiguous in {path}")
        if not isinstance(row["trace_step_id"], str):
            raise ValueError(f"910B MSTX marker step ID is invalid in {path}")
        if row["schema_version"] == 3 and (
            not isinstance(row["stream_handle"], int) or row["stream_handle"] < 1
        ):
            raise ValueError(f"910B MSTX marker stream is invalid in {path}")
        if row["schema_version"] == 3 and (
            not isinstance(row["mark_issue_start_ns"], int)
            or not isinstance(row["mark_issue_end_ns"], int)
            or row["mark_issue_end_ns"] < row["mark_issue_start_ns"]
        ):
            raise ValueError(f"910B MSTX marker host interval is invalid in {path}")
    return rows


def analyze_ascend_910b_selected_step_msprof(
    run_dir: Path,
    scheduler_trace_path: Path,
    validation_path: Path,
    selected_step_id: str,
    expected_rank_count: int,
) -> dict[str, object]:
    """Return a fail-closed, same-run selected-step timeline."""

    root = run_dir.resolve(strict=True)
    trace_path = scheduler_trace_path.resolve(strict=True)
    validation_file = validation_path.resolve(strict=True)
    if isinstance(expected_rank_count, bool) or expected_rank_count < 1:
        raise ValueError("expected rank count must be a positive integer")
    validation = _read_json(validation_file)
    if validation["injected_step_id"] != selected_step_id:
        raise ValueError("selected step disagrees with injection validation")
    if validation["structural_fingerprint_match"] is not True:
        raise ValueError("INVALID EXPERIMENT — structural_fingerprint_match")

    trace_rows = _read_jsonl(trace_path)
    trace_ids = [row["step_id"] for row in trace_rows]
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("Scheduler trace contains duplicate step IDs")
    if selected_step_id not in trace_ids:
        raise ValueError("Scheduler trace does not contain selected step")
    selected_index = trace_ids.index(selected_step_id)
    if selected_index == 0:
        raise ValueError("selected step has no preceding device end boundary")
    previous_step_id = trace_ids[selected_index - 1]
    selected_trace = trace_rows[selected_index]
    previous_trace = trace_rows[selected_index - 1]
    clock = _read_json(Path(f"{trace_path}.clock.json"))
    if clock["clock_domain"] != "shared_host_time.perf_counter_ns":
        raise ValueError("Scheduler trace clock domain is unsupported")
    origin_ns = clock["trace_origin_monotonic_ns"]
    if isinstance(origin_ns, bool) or not isinstance(origin_ns, int):
        raise ValueError("Scheduler trace origin is invalid")

    info_paths = tuple(
        path
        for path in sorted(root.glob("msprof/PROF_*/device_*/info.json.*"))
        if not path.name.endswith(".done")
    )
    if len(info_paths) != expected_rank_count:
        raise ValueError("910B run must contain exactly one PROF per expected rank")
    profiles = []
    all_task_intervals = []
    seen_ranks = set()
    seen_devices = set()
    for info_path in info_paths:
        pid, device_id = _profile_identity(info_path)
        marker_paths = tuple(root.glob(
            f"{trace_path.name}.ascend-step-marker.rank-*.pid-{pid}.jsonl"
        ))
        if len(marker_paths) != 1:
            raise ValueError(f"910B PROF pid {pid} must have exactly one marker sidecar")
        marker_path = marker_paths[0]
        rank_text = marker_path.name.split(".rank-", 1)[1].split(".pid-", 1)[0]
        if not rank_text.isdecimal():
            raise ValueError("910B marker rank filename is invalid")
        rank = int(rank_text)
        if rank in seen_ranks or device_id in seen_devices:
            raise ValueError("910B rank/device identity is duplicated")
        seen_ranks.add(rank)
        seen_devices.add(device_id)
        markers = _marker_rows(marker_path, rank, pid)
        marker_step_ids = [row["trace_step_id"] for row in markers]
        if marker_step_ids != trace_ids:
            raise ValueError("910B MSTX marker step sequence disagrees with Scheduler trace")

        device_dir = info_path.parent
        step_db = device_dir / "sqlite" / "step_trace.db"
        task_db = device_dir / "sqlite" / "ascend_task.db"
        time_db = device_dir / "sqlite" / "time.db"
        for database in (step_db, task_db, time_db):
            if not database.is_file() or database.stat().st_size == 0:
                raise ValueError(f"910B profiler database is missing or empty: {database}")
        legacy_boundaries = markers[0]["schema_version"] == 2
        with sqlite3.connect(step_db) as connection:
            if legacy_boundaries:
                _require_columns(
                    connection, "step_trace_data",
                    {"step_start", "step_end", "iter_id"},
                )
                step_rows = connection.execute(
                    "SELECT step_start, step_end, iter_id "
                    "FROM step_trace_data ORDER BY iter_id"
                ).fetchall()
                if len(step_rows) != len(markers) or [row[2] for row in step_rows] != list(range(1, len(markers) + 1)):
                    raise ValueError("910B step-trace iteration count disagrees with marker sidecar")
                flips = []
            else:
                _require_columns(
                    connection, "DeviceTaskFlip",
                    {"stream_id", "timestamp", "task_id", "flip_num"},
                )
                flips = connection.execute(
                    "SELECT stream_id, timestamp, task_id, flip_num "
                    "FROM DeviceTaskFlip ORDER BY timestamp"
                ).fetchall()
                if len(flips) != len(markers):
                    raise ValueError("910B device flip count disagrees with marker sidecar")
                if any(flip[2:] != (65535, 65535) for flip in flips):
                    raise ValueError("910B device flip sentinel identity is invalid")
        with sqlite3.connect(time_db) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(\"Time\")")}
            _require_columns(connection, "Time", {"device_id", "dev_mon", "host_mon"})
            if legacy_boundaries and "dev_cntvct" not in columns:
                raise ValueError("910B legacy step trace lacks a device clock anchor")
            anchor_columns = "device_id, dev_mon, host_mon"
            if legacy_boundaries:
                anchor_columns = "device_id, dev_mon, host_mon, dev_cntvct"
            anchors = connection.execute(f"SELECT {anchor_columns} FROM Time").fetchall()
        if len(anchors) != 1 or anchors[0][0] != device_id:
            raise ValueError("910B device/host clock anchor is ambiguous")
        if legacy_boundaries:
            _, _device_anchor_ns, host_anchor_ns, _device_cntvct_ns = anchors[0]
        else:
            _, _device_anchor_ns, host_anchor_ns = anchors[0]
        with sqlite3.connect(task_db) as connection:
            _require_columns(
                connection,
                "AscendTask",
                {
                    "start_time", "duration", "host_task_type",
                    "device_task_type", "stream_id", "task_id",
                },
            )
            task_rows = connection.execute(
                "SELECT start_time, duration, host_task_type, device_task_type, "
                "stream_id, task_id FROM AscendTask ORDER BY start_time"
            ).fetchall()
        if legacy_boundaries:
            # The legacy step table identifies the first and last device task
            # of each iteration.  Resolve those task IDs in the exported task
            # table; this preserves the profiler's own boundaries even though
            # the legacy marker sidecar has no host timestamps.
            task_by_id: dict[int, list[tuple[object, ...]]] = {}
            for row in task_rows:
                if row[5] not in task_by_id:
                    task_by_id[row[5]] = []
                task_by_id[row[5]].append(row)
            legacy_bounds: list[tuple[int, int]] = []
            with sqlite3.connect(step_db) as connection:
                step_marks = connection.execute(
                    "SELECT task_id, tag_id FROM StepTrace ORDER BY index_id, tag_id"
                ).fetchall()
            if len(step_marks) != 2 * len(markers):
                raise ValueError("910B StepTrace boundary count disagrees with marker sidecar")
            for ordinal in range(len(markers)):
                start_id, start_tag = step_marks[2 * ordinal]
                end_id, end_tag = step_marks[2 * ordinal + 1]
                if start_tag != 0 or end_tag != 1:
                    raise ValueError("910B StepTrace boundary tags are invalid")
                start_rows = task_by_id[start_id] if start_id in task_by_id else ()
                end_rows = task_by_id[end_id] if end_id in task_by_id else ()
                starts = [
                    row[0] for row in start_rows
                    if row[1] > 0 and row[3] in {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "COMM"}
                ]
                ends = [
                    row[0] + row[1] for row in end_rows
                    if row[1] > 0 and row[3] in {"AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "COMM"}
                ]
                if not starts or not ends:
                    raise ValueError("910B StepTrace task IDs have no physical task rows")
                legacy_bounds.append((min(starts), max(ends)))
            if any(end <= start for start, end in legacy_bounds):
                raise ValueError("910B legacy StepTrace boundaries are not increasing")
            if any(legacy_bounds[i][0] < legacy_bounds[i - 1][1] for i in range(1, len(legacy_bounds))):
                raise ValueError("910B legacy StepTrace boundaries overlap")
            previous_boundary_ns = legacy_bounds[selected_index - 1][1]
            selected_boundary_ns = legacy_bounds[selected_index][1]
            raw_previous_boundary_ns = previous_boundary_ns
            raw_selected_boundary_ns = selected_boundary_ns
            task_host_offset = 0
        else:
            mapped_flips = [
                host_anchor_ns + timestamp - _device_anchor_ns
                for _stream_id, timestamp, _task_id, _flip_num in flips
            ]
            for marker, mapped_ns in zip(markers, mapped_flips, strict=True):
                if abs(mapped_ns - marker["mark_issue_start_ns"]) > 5_000_000_000:
                    raise ValueError("910B device/host clock mapping exceeds five seconds")
            previous_boundary_ns = mapped_flips[selected_index - 1]
            selected_boundary_ns = mapped_flips[selected_index]
            raw_previous_boundary_ns = flips[selected_index - 1][1]
            raw_selected_boundary_ns = flips[selected_index][1]
            task_host_offset = host_anchor_ns - _device_anchor_ns
        if selected_boundary_ns <= previous_boundary_ns:
            raise ValueError("910B selected-step device boundaries are not increasing")
        selected_tasks = []
        groups: dict[tuple[str, str], dict[str, object]] = {}
        for start_time, duration, host_type, device_type, stream_id, task_id in task_rows:
            end_time = start_time + duration
            if start_time < raw_previous_boundary_ns < end_time or start_time < raw_selected_boundary_ns < end_time:
                raise ValueError("910B task crosses a selected-step device boundary")
            if not (
                start_time >= raw_previous_boundary_ns
                and end_time <= raw_selected_boundary_ns
                and duration > 0
            ):
                continue
            host_start_ns = start_time + task_host_offset
            host_end_ns = host_start_ns + duration
            selected_tasks.append((host_start_ns, host_end_ns))
            key = (host_type, device_type)
            if key not in groups:
                groups[key] = {
                    "host_task_type": host_type,
                    "device_task_type": device_type,
                    "count": 0,
                    "additive_duration_ns": 0,
                }
            groups[key]["count"] += 1
            groups[key]["additive_duration_ns"] += duration
        if not selected_tasks:
            raise ValueError("910B selected-step boundary contains no device tasks")
        all_task_intervals.extend(selected_tasks)
        profiles.append({
            "rank": rank,
            "process_pid": pid,
            "device_id": device_id,
            "previous_step_id": previous_step_id,
            "selected_step_id": selected_step_id,
            "previous_boundary_host_ns": previous_boundary_ns,
            "selected_boundary_host_ns": selected_boundary_ns,
            "selected_device_task_count": len(selected_tasks),
            "selected_device_task_union_ns": interval_union_ns(selected_tasks),
            "task_groups": sorted(
                groups.values(), key=lambda group: -group["additive_duration_ns"]
            ),
            "profiler_databases": {
                "step_trace_sha256": file_sha256(step_db),
                "ascend_task_sha256": file_sha256(task_db),
                "time_sha256": file_sha256(time_db),
                "marker_sidecar_sha256": file_sha256(marker_path),
            },
        })
    if seen_ranks != set(range(expected_rank_count)):
        raise ValueError("910B marker sidecars do not cover every expected rank")

    service_start_relative_ns = max(
        selected_trace["schedule_end_ns"], previous_trace["update_start_ns"]
    )
    service_end_relative_ns = selected_trace["update_start_ns"]
    service_start_ns = origin_ns + service_start_relative_ns
    service_end_ns = origin_ns + service_end_relative_ns
    serialized_service_ns = service_end_ns - service_start_ns
    exposed_device_ns = clipped_union_ns(
        all_task_intervals, service_start_ns, service_end_ns
    )
    device_union_ns = interval_union_ns(all_task_intervals)
    return {
        "schema_version": 1,
        "kind": "ascend_910b_selected_step_msprof_timeline",
        "comparison_status": "valid_same_run_device_timeline",
        "canonical_step_id": validation["canonical_step_id"],
        "selected_step_id": selected_step_id,
        "previous_step_id": previous_step_id,
        "structural_fingerprint_match": True,
        "clock_domain": "shared_host_time.perf_counter_ns",
        "observed_full_step_ms": (
            selected_trace["step_end_ns"] - selected_trace["step_start_ns"]
        ) / 1_000_000,
        "serialized_service_ms": serialized_service_ns / 1_000_000,
        "rpc_envelope_ms": (
            selected_trace["device_execution_end_ns"]
            - selected_trace["device_execution_start_ns"]
        ) / 1_000_000,
        "worker_submit_ms": None,
        "device_task_union_ms": device_union_ns / 1_000_000,
        "exposed_device_task_union_ms": exposed_device_ns / 1_000_000,
        "uncovered_service_ms": (serialized_service_ns - exposed_device_ns)
        / 1_000_000,
        "runtime_inter_task_gap_ms": None,
        "profiles": sorted(profiles, key=lambda profile: profile["rank"]),
        "inputs": {
            "scheduler_trace": str(trace_path),
            "scheduler_trace_sha256": file_sha256(trace_path),
            "validation": str(validation_file),
            "validation_sha256": file_sha256(validation_file),
        },
    }
