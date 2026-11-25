"""Compact, request-safe Scheduler progress diagnostics."""

from __future__ import annotations

import hashlib
import json
import math

from src.serving.request_gen.request_stream import RequestStream
from src.serving.schedule.schedule_trace import ScheduleTrace


_TIME_AXIS_BY_MODE = {
    "device_only": "predicted_device_only_virtual_time",
    "device_plus_launch": "predicted_device_plus_launch_virtual_time",
    "device_plus_launch_plus_scheduler": (
        "predicted_device_plus_launch_plus_scheduler_virtual_time"),
}
_SCHEMA_VERSION_BY_MODE = {
    "device_only": 1,
    "device_plus_launch": 2,
    "device_plus_launch_plus_scheduler": 2,
}


def scheduler_progress_manifest(
    requests: RequestStream,
    schedule_trace: ScheduleTrace,
) -> dict[str, object]:
    """Summarize invariant scheduled work without retaining request payloads."""

    if not isinstance(requests, RequestStream):
        raise TypeError("scheduler progress requires a RequestStream")
    if not isinstance(schedule_trace, ScheduleTrace):
        raise TypeError("scheduler progress requires a ScheduleTrace")
    timing_modes = {entry.timing_mode for entry in schedule_trace.entries}
    if len(timing_modes) != 1:
        raise ValueError("scheduler progress cannot mix Serving timing modes")
    timing_mode = next(iter(timing_modes))

    expected_input_tokens = sum(request.input_tokens for request in requests)
    expected_output_tokens = sum(request.output_tokens for request in requests)
    cumulative_tokens = 0
    cumulative_prefill_tokens = 0
    cumulative_decode_tokens = 0
    scheduler_prefix_hit_tokens = 0
    recomputed_tokens = 0
    # Recompute splits into two physically distinct kinds, and the split is
    # causal, not positional.  Preemption releases a resident request's KV;
    # the resume replay re-forwards everything the request had computed, and
    # that replay is real extra work.  Its liability at the preemption event
    # is min(emitted frontier, high-water mark): the request holds
    # input + emitted tokens, but only positions previously forwarded (below
    # the high-water mark) count as recompute on the resume rows.  The
    # frontier (not the computed position) is the liability because a
    # request preempted mid-resume has a computed position below it.
    # One refinement, specific to MTP verify mechanics: when the high-water
    # mark has reached the emitted frontier under MTP, position frontier - 1
    # was already forwarded speculatively and its candidate rejected, so the
    # resume replay's tail re-forward of that position doubles as the
    # canonical forward emitting the frontier token -- the same role the
    # free MTP verify refeed plays without preemption.  That tail forward is
    # not replay waste, and the liability excludes it; charging it would
    # double-count one token per such preemption.  The exclusion is gated on
    # the run using MTP because its premise (a rejected speculative
    # candidate at the tail) only exists under speculative decoding.  In an
    # STP run the same watermark >= frontier relation arises without any
    # rejection: a PD decode pool seeds the request with the transferred
    # prompt KV (the high-water mark starts at the prompt length) while the
    # first output token was emitted by the prefill pool, so every decode
    # forward advances the watermark in lockstep with the emitted frontier.
    # There the replay below the watermark is entirely waste -- the
    # canonical tail emission comes from forwarding the position AT the
    # watermark, which lies above the recomputed window -- and shaving one
    # token off the liability would leak one unaccounted scheduled token
    # per decode-pool preemption.
    # MTP rejection rollback also trips the high-watermark recompute
    # counter, including positions at or below the emitted frontier (the
    # verify window re-feeds the last emitted position); that recompute is
    # part of normal verify mechanics and its waste is already accounted as
    # rejected speculative work, so it is only extra when a preemption
    # replay liability is still open.  The rollback bucket is MTP-only by
    # construction: in an STP run every recomputed position must be covered
    # by an open preemption liability, and any remainder fails the
    # conservation check below.
    preemption_recomputed_tokens = 0
    mtp_rollback_recomputed_tokens = 0
    input_length_by_id = {
        request.request_id: request.input_tokens for request in requests
    }
    emitted_output_tokens: dict[tuple[str, str], int] = {}
    high_watermark: dict[tuple[str, str], int] = {}
    pending_preemption_replay: dict[tuple[str, str], int] = {}
    speculative_overhead_tokens = 0
    mtp_proposed_tokens = 0
    mtp_verified_tokens = 0
    mtp_accepted_tokens = 0
    mtp_rejected_tokens = 0
    mtp_raw_output_tokens = 0
    mtp_emitted_tokens = 0
    discarded_async_decode_tokens = 0
    # MTP mode is a run-global scheduler property, and the preemption
    # liability rule below depends on it, so derive it from the whole trace
    # up front rather than from the rows seen so far.
    uses_mtp = any(
        request.mtp_acceptance_source_kind != "disabled"
        for entry in schedule_trace.entries
        for request in entry.backend_step.schedule.requests
    )
    finished_request_ids: set[str] = set()
    series = []
    for step_index, entry in enumerate(schedule_trace.entries):
        schedule = entry.backend_step.schedule
        prefill_tokens = sum(
            request.input_tokens
            for request in schedule.requests
        )
        decode_tokens = sum(
            request.output_tokens
            for request in schedule.requests
        )
        scheduled_tokens = prefill_tokens + decode_tokens
        if scheduled_tokens != schedule.token_count:
            raise ValueError(
                "scheduler progress token split disagrees with ScheduleStep")
        finished_request_ids.update(entry.backend_step.finished_request_ids)
        cumulative_tokens += scheduled_tokens
        cumulative_prefill_tokens += prefill_tokens
        cumulative_decode_tokens += decode_tokens
        scheduler_prefix_hit_tokens += sum(
            request.scheduler_prefix_hit_tokens + request.content_prefix_hit_tokens
            for request in schedule.requests
        )
        recomputed = sum(
            request.recomputed_tokens for request in schedule.requests
        )
        recomputed_tokens += recomputed
        for request in schedule.requests:
            key = (schedule.pool_name, request.request_id)
            pending = pending_preemption_replay.get(key, 0)
            extra = min(request.recomputed_tokens, pending)
            preemption_recomputed_tokens += extra
            mtp_rollback_recomputed_tokens += (
                request.recomputed_tokens - extra)
            pending -= extra
            if pending:
                pending_preemption_replay[key] = pending
            else:
                pending_preemption_replay.pop(key, None)
            high_watermark[key] = max(
                high_watermark.get(key, 0),
                request.computed_tokens + request.scheduled_tokens)
        for request_id in schedule.preempted_request_ids:
            key = (schedule.pool_name, request_id)
            frontier = (
                input_length_by_id[request_id]
                + emitted_output_tokens.get(key, 0))
            watermark = high_watermark.get(key, frontier)
            pending = min(frontier, watermark)
            if uses_mtp and watermark >= frontier and pending:
                # MTP-only: position frontier - 1 was forwarded
                # speculatively and its candidate rejected, so the resume
                # replay's tail re-forward of it doubles as the canonical
                # forward emitting the frontier token, not replay waste.
                pending -= 1
            pending_preemption_replay[key] = pending
        for request_id, count in entry.backend_step.generated_tokens.items():
            key = (schedule.pool_name, request_id)
            emitted_output_tokens[key] = (
                emitted_output_tokens.get(key, 0) + count)
        for request in schedule.requests:
            emitted_tokens = entry.backend_step.generated_tokens[
                request.request_id]
            if request.mtp_acceptance_source_kind == "disabled":
                if (request.output_tokens > 0
                        and emitted_tokens < request.raw_model_output_tokens):
                    discarded_async_decode_tokens += (
                        request.raw_model_output_tokens - emitted_tokens)
                continue
            mtp_proposed_tokens += request.mtp_proposed_tokens
            mtp_verified_tokens += request.mtp_verified_tokens
            mtp_accepted_tokens += request.mtp_accepted_tokens
            mtp_rejected_tokens += request.mtp_rejected_tokens
            mtp_raw_output_tokens += request.raw_model_output_tokens
            mtp_emitted_tokens += emitted_tokens
            if emitted_tokens > request.raw_model_output_tokens:
                raise ValueError(
                    "scheduler progress MTP emitted tokens exceed raw model "
                    "output")
            speculative_overhead_tokens += (
                request.mtp_rejected_tokens
                + request.raw_model_output_tokens
                - emitted_tokens
            )
        series.append({
            "step_index": step_index,
            "step_id": schedule.step_id,
            "start_time_s": float(entry.start_time_s),
            "end_time_s": float(entry.end_time_s),
            "request_count": len(schedule.requests),
            "prefill_request_count": sum(
                request.phase == "prefill" for request in schedule.requests
            ),
            "decode_request_count": sum(
                request.phase == "decode" for request in schedule.requests
            ),
            "scheduled_tokens": scheduled_tokens,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "cumulative_scheduled_tokens": cumulative_tokens,
            "cumulative_prefill_tokens": cumulative_prefill_tokens,
            "cumulative_decode_tokens": cumulative_decode_tokens,
            "cumulative_completed_requests": len(finished_request_ids),
        })

    expected_scheduled_tokens = expected_input_tokens + expected_output_tokens
    # vLLM emits the final token as a completion event rather than scheduling
    # another model-forward token for each non-empty request.
    expected_scheduled_tokens -= sum(
        request.output_tokens > 0 for request in requests
    )
    expected_scheduled_tokens -= scheduler_prefix_hit_tokens
    # Preemption replay is real extra forward work: the replayed prefix is
    # counted inside the resume step's scheduled tokens, while the canonical
    # request specs never contain it.  Add it on the expected side.  MTP
    # rejection-rollback recompute is NOT added: those replayed positions are
    # canonical output tokens emitting for the first time, and the wasted
    # speculative pass over them is already counted as speculative overhead.
    expected_scheduled_tokens += preemption_recomputed_tokens
    expected_scheduled_tokens += speculative_overhead_tokens
    expected_scheduled_tokens += discarded_async_decode_tokens
    if cumulative_tokens != expected_scheduled_tokens:
        base_expected_tokens = (
            expected_input_tokens
            + expected_output_tokens
            - sum(request.output_tokens > 0 for request in requests)
            - scheduler_prefix_hit_tokens
            + preemption_recomputed_tokens
        )
        raise ValueError(
            "scheduler progress does not conserve canonical request tokens: "
            f"scheduled={cumulative_tokens}, expected={expected_scheduled_tokens}, "
            f"base={base_expected_tokens}, "
            f"mtp_overhead={speculative_overhead_tokens}, "
            f"discarded_async_decode={discarded_async_decode_tokens}, "
            f"mtp_proposed={mtp_proposed_tokens}, "
            f"mtp_verified={mtp_verified_tokens}, "
            f"mtp_accepted={mtp_accepted_tokens}, "
            f"mtp_rejected={mtp_rejected_tokens}, "
            f"mtp_raw_output={mtp_raw_output_tokens}, "
            f"mtp_emitted={mtp_emitted_tokens}, "
            f"prefix_hits={scheduler_prefix_hit_tokens}, "
            f"recomputed={recomputed_tokens}, "
            f"preemption_recomputed={preemption_recomputed_tokens}, "
            f"mtp_rollback_recomputed={mtp_rollback_recomputed_tokens}, "
            f"uses_mtp={uses_mtp}")
    if len(finished_request_ids) != len(requests.requests):
        raise ValueError(
            "scheduler progress does not finish every canonical request")

    body = {
        "schema_version": _SCHEMA_VERSION_BY_MODE[timing_mode],
        "time_axis": _TIME_AXIS_BY_MODE[timing_mode],
        "token_accounting": (
            "input_plus_output_minus_terminal_plus_mtp_overhead_v1"
            if uses_mtp
            else "input_plus_output_minus_terminal_plus_async_discard_v1"
            if discarded_async_decode_tokens
            else "input_plus_output_minus_terminal_token_v1"
        ),
        "summary": {
            "request_count": len(requests.requests),
            "step_count": len(series),
            "scheduled_request_rows": sum(
                row["request_count"] for row in series
            ),
            "scheduled_tokens": cumulative_tokens,
            "prefill_tokens": cumulative_prefill_tokens,
            "decode_tokens": cumulative_decode_tokens,
            "completed_requests": len(finished_request_ids),
            "end_time_s": float(schedule_trace.end_time_s),
        },
        "series": series,
    }
    serialized = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        **body,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def validate_scheduler_progress_manifest(value: object) -> dict[str, object]:
    """Validate and copy a Scheduler progress manifest at the owner boundary."""

    if not isinstance(value, dict):
        raise TypeError("scheduler progress manifest must be an object")
    expected_fields = {
        "schema_version", "time_axis", "token_accounting", "summary",
        "series", "sha256",
    }
    if set(value) != expected_fields:
        raise ValueError("scheduler progress manifest fields are invalid")
    if value["schema_version"] not in (1, 2):
        raise ValueError("scheduler progress schema_version must equal 1 or 2")
    if value["time_axis"] not in set(_TIME_AXIS_BY_MODE.values()):
        raise ValueError("scheduler progress time axis is invalid")
    if ((value["schema_version"] == 1)
            != (value["time_axis"] == _TIME_AXIS_BY_MODE["device_only"])):
        raise ValueError("scheduler progress schema and time axis disagree")
    if value["token_accounting"] not in (
            "input_plus_output_minus_terminal_token_v1",
            "input_plus_output_minus_terminal_plus_async_discard_v1",
            "input_plus_output_minus_terminal_plus_mtp_overhead_v1",
    ):
        raise ValueError("scheduler progress token accounting is invalid")
    if not isinstance(value["summary"], dict):
        raise TypeError("scheduler progress summary must be an object")
    if not isinstance(value["series"], list):
        raise TypeError("scheduler progress series must be a list")
    expected_summary_fields = {
        "request_count", "step_count", "scheduled_request_rows",
        "scheduled_tokens", "prefill_tokens", "decode_tokens",
        "completed_requests", "end_time_s",
    }
    if set(value["summary"]) != expected_summary_fields:
        raise ValueError("scheduler progress summary fields are invalid")
    integer_fields = expected_summary_fields - {"end_time_s"}
    if any(type(value["summary"][field]) is not int for field in integer_fields):
        raise TypeError("scheduler progress summary counts must be integers")
    end_time_s = value["summary"]["end_time_s"]
    if (isinstance(end_time_s, bool)
            or not isinstance(end_time_s, (int, float))
            or not math.isfinite(end_time_s)
            or end_time_s < 0):
        raise ValueError("scheduler progress end time must be non-negative")
    if value["summary"]["step_count"] != len(value["series"]):
        raise ValueError("scheduler progress step count disagrees with series")
    row_fields = {
        "step_index", "step_id", "start_time_s", "end_time_s",
        "request_count", "prefill_request_count", "decode_request_count",
        "scheduled_tokens", "prefill_tokens", "decode_tokens",
        "cumulative_scheduled_tokens", "cumulative_prefill_tokens",
        "cumulative_decode_tokens", "cumulative_completed_requests",
    }
    cumulative_tokens = 0
    cumulative_prefill_tokens = 0
    cumulative_decode_tokens = 0
    cumulative_completed_requests = 0
    # PD executions legitimately interleave events from independent prefill
    # and decode pools.  The series preserves global append order for
    # cumulative accounting, while the time axis is required to be monotonic
    # within each pool identified by the step-id prefix.
    previous_time_by_pool: dict[str, tuple[float, float]] = {}
    step_ids = set()
    scheduled_request_rows = 0
    for expected_index, row_value in enumerate(value["series"]):
        if not isinstance(row_value, dict) or set(row_value) != row_fields:
            raise ValueError("scheduler progress row fields are invalid")
        row = row_value
        if row["step_index"] != expected_index:
            raise ValueError("scheduler progress step indices are not canonical")
        if not isinstance(row["step_id"], str) or not row["step_id"]:
            raise TypeError("scheduler progress step_id must be non-empty text")
        if ":" not in row["step_id"]:
            raise ValueError("scheduler progress step_id must include a pool prefix")
        if row["step_id"] in step_ids:
            raise ValueError("scheduler progress step IDs must be unique")
        step_ids.add(row["step_id"])
        start_time_s = row["start_time_s"]
        end_time_s = row["end_time_s"]
        for name, timestamp in (
                ("start_time_s", start_time_s),
                ("end_time_s", end_time_s)):
            if (isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp)
                    or timestamp < 0):
                raise ValueError(
                    f"scheduler progress {name} must be non-negative")
        pool_id = row["step_id"].split(":", 1)[0]
        if pool_id in previous_time_by_pool:
            previous_start_time_s, previous_end_time_s = previous_time_by_pool[pool_id]
        else:
            previous_start_time_s, previous_end_time_s = 0.0, 0.0
        if (start_time_s < previous_start_time_s
                or end_time_s < previous_end_time_s
                or end_time_s < start_time_s):
            raise ValueError("scheduler progress time axis is non-monotonic")
        previous_time_by_pool[pool_id] = (start_time_s, end_time_s)
        row_integer_fields = row_fields - {
            "step_id", "start_time_s", "end_time_s",
        }
        if any(type(row[field]) is not int for field in row_integer_fields):
            raise TypeError("scheduler progress row counts must be integers")
        if any(row[field] < 0 for field in row_integer_fields):
            raise ValueError("scheduler progress row counts must be non-negative")
        if row["request_count"] != (
                row["prefill_request_count"] + row["decode_request_count"]):
            raise ValueError("scheduler progress request split is inconsistent")
        if row["scheduled_tokens"] != (
                row["prefill_tokens"] + row["decode_tokens"]):
            raise ValueError("scheduler progress token split is inconsistent")
        cumulative_tokens += row["scheduled_tokens"]
        cumulative_prefill_tokens += row["prefill_tokens"]
        cumulative_decode_tokens += row["decode_tokens"]
        if row["cumulative_scheduled_tokens"] != cumulative_tokens:
            raise ValueError("scheduler progress cumulative tokens are invalid")
        if row["cumulative_prefill_tokens"] != cumulative_prefill_tokens:
            raise ValueError("scheduler progress cumulative prefill is invalid")
        if row["cumulative_decode_tokens"] != cumulative_decode_tokens:
            raise ValueError("scheduler progress cumulative decode is invalid")
        if row["cumulative_completed_requests"] < cumulative_completed_requests:
            raise ValueError("scheduler progress completion count decreased")
        cumulative_completed_requests = row["cumulative_completed_requests"]
        scheduled_request_rows += row["request_count"]
    expected_summary = {
        "request_count": value["summary"]["request_count"],
        "step_count": len(value["series"]),
        "scheduled_request_rows": scheduled_request_rows,
        "scheduled_tokens": cumulative_tokens,
        "prefill_tokens": cumulative_prefill_tokens,
        "decode_tokens": cumulative_decode_tokens,
        "completed_requests": cumulative_completed_requests,
        "end_time_s": max(
            end_time for _, end_time in previous_time_by_pool.values()),
    }
    if value["summary"] != expected_summary:
        raise ValueError("scheduler progress summary disagrees with series")
    if cumulative_completed_requests != value["summary"]["request_count"]:
        raise ValueError("scheduler progress did not complete every request")
    if (not isinstance(value["sha256"], str)
            or len(value["sha256"]) != 64):
        raise ValueError("scheduler progress SHA256 is invalid")
    body = {field: value[field] for field in expected_fields - {"sha256"}}
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    if value["sha256"] != digest:
        raise ValueError("scheduler progress SHA256 mismatch")
    return dict(value)
