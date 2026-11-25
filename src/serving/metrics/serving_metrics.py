"""TTFT, TPOT, latency, and throughput aggregation."""

from dataclasses import dataclass
import math
from typing import Mapping

from src.serving.request_gen.request_stream import RequestStream
from src.serving.schedule.schedule_trace import ScheduleTrace


@dataclass(frozen=True)
class RequestMetrics:
    request_id: str
    ttft_s: float | None
    tpot_s: float | None
    latency_s: float


@dataclass(frozen=True)
class ServingMetrics:
    requests: Mapping[str, RequestMetrics]
    request_throughput_per_s: float
    output_token_throughput_per_s: float
    token_throughput_per_s: float


def compute_serving_metrics(requests: RequestStream,
                            schedule_trace: ScheduleTrace,
                            first_token_delivery_s=0.0) -> ServingMetrics:
    """Aggregate client-visible latency from one virtual-time schedule.

    ``first_token_delivery_s`` is the Host cost of carrying one produced token
    out of the engine and onto the client connection.  The schedule trace ends
    at the Scheduler's own output update, so a client-visible TTFT is only
    complete once that delivery is charged.  Callers running without a Host
    request path profile leave it at zero, which reproduces the engine-visible
    boundary exactly.

    The delivery is walked once per output of every producing step, so a
    caller that models that fan-out passes a callable of the step's output
    count instead of a single number. First and last token endpoints include
    their own delivery envelopes. Per-request output order is preserved when
    two different fan-out envelopes overlap; this adds no CPU queue service
    or feedback to the Scheduler. A zero delivery keeps the engine boundary.

    A disaggregated request produces tokens on more than one pool (the PD
    prefill pool emits the first token at handoff, the decode pool emits the
    rest).  Client-visible timing is attributed to the terminal pool — the
    pool that produces the request's last token: TTFT runs until that pool's
    first produced token, so a request queued for a decode slot after its
    prefill handoff shows that wait in TTFT rather than as an inflated first
    inter-token gap, and TPOT divides the terminal pool's own production
    window by the gaps it actually produced.
    """

    if schedule_trace.observation_end_time_s is not None:
        raise ValueError("complete-cohort metrics reject a right-censored observation")
    if callable(first_token_delivery_s):
        delivery_of = first_token_delivery_s
    else:
        if (isinstance(first_token_delivery_s, bool)
                or not isinstance(first_token_delivery_s, (int, float))
                or not math.isfinite(first_token_delivery_s)
                or first_token_delivery_s < 0):
            raise ValueError(
                "first token delivery time must be finite and non-negative")
        constant_delivery_s = float(first_token_delivery_s)

        def delivery_of(step_output_count: int) -> float:
            return constant_delivery_s
    arrival = {request.request_id: request.arrival_time_s for request in requests}
    expected_outputs = {request.request_id: request.output_tokens for request in requests}
    first_end = {}
    first_delivery_s = {}
    last_end = {}
    completion_end = {}
    produced = {request.request_id: 0 for request in requests}
    pool_produced = {}
    last_pool = {}
    for entry in schedule_trace.entries:
        for request_id in entry.backend_step.finished_request_ids:
            if request_id not in arrival:
                raise ValueError(
                    f"schedule finished unknown request {request_id!r}")
            # A PD handoff marks the request finished in the P scheduler and
            # later in the D scheduler.  The global completion is the last
            # such event on the unified trace.
            completion_end[request_id] = entry.end_time_s
        step_output_count = sum(
            1 for scheduled in entry.backend_step.schedule.requests
            if entry.backend_step.generated_tokens[scheduled.request_id])
        for scheduled in entry.backend_step.schedule.requests:
            request_id = scheduled.request_id
            if request_id not in arrival:
                raise ValueError(f"schedule contains unknown request {request_id!r}")
            output_count = entry.backend_step.generated_tokens[request_id]
            if output_count:
                pool_name = entry.backend_step.schedule.pool_name
                if request_id not in first_end:
                    first_end[request_id] = {}
                    last_end[request_id] = {}
                    pool_produced[request_id] = {}
                delivery_s = delivery_of(step_output_count)
                if (isinstance(delivery_s, bool)
                        or not isinstance(delivery_s, (int, float))
                        or not math.isfinite(delivery_s) or delivery_s < 0):
                    raise ValueError(
                        "token delivery time must be finite and non-negative")
                delivered_end_s = entry.end_time_s + delivery_s
                if pool_name in last_end[request_id]:
                    delivered_end_s = max(
                        delivered_end_s, last_end[request_id][pool_name])
                if pool_name not in first_end[request_id]:
                    first_end[request_id][pool_name] = entry.end_time_s
                    first_delivery_s[(request_id, pool_name)] = delivery_s
                last_end[request_id][pool_name] = delivered_end_s
                last_pool[request_id] = pool_name
                if pool_name not in pool_produced[request_id]:
                    pool_produced[request_id][pool_name] = 0
                pool_produced[request_id][pool_name] += output_count
                produced[request_id] += output_count
    per_request = {}
    client_completion_end = {}
    for request_id in arrival:
        if produced[request_id] != expected_outputs[request_id]:
            raise ValueError(
                f"request {request_id!r} produced {produced[request_id]} tokens; "
                f"expected {expected_outputs[request_id]}")
        if request_id not in completion_end:
            raise ValueError(f"request {request_id!r} never finished")
        client_completion_end[request_id] = completion_end[request_id]
        if produced[request_id] == 0:
            ttft = None
            tpot = None
        elif request_id not in first_end or request_id not in last_end:
            raise ValueError(
                f"request {request_id!r} has no output-token timing")
        else:
            terminal_pool = last_pool[request_id]
            terminal_produced = pool_produced[request_id][terminal_pool]
            first_token_end_s = (first_end[request_id][terminal_pool]
                + first_delivery_s[(request_id, terminal_pool)])
            last_token_end_s = last_end[request_id][terminal_pool]
            client_completion_end[request_id] = max(
                completion_end[request_id], last_token_end_s)
            # Preserve the original TTFT arithmetic exactly while correcting
            # the other endpoints to the same client-visible boundary.
            ttft = (first_end[request_id][terminal_pool] - arrival[request_id]
                + first_delivery_s[(request_id, terminal_pool)])
            if terminal_produced == 1:
                tpot = 0.0
            else:
                # Delivery cancels only when the two endpoint envelopes are
                # equal. A changing batch fan-out must retain the difference.
                tpot = (
                    last_token_end_s - first_token_end_s
                ) / (terminal_produced - 1)
        latency = client_completion_end[request_id] - arrival[request_id]
        if latency < 0:
            raise ValueError(
                f"request {request_id!r} has non-monotonic timing")
        if ttft is not None and ttft < 0:
            raise ValueError(
                f"request {request_id!r} has non-monotonic timing")
        per_request[request_id] = RequestMetrics(
            request_id=request_id,
            ttft_s=ttft,
            tpot_s=tpot,
            latency_s=latency,
        )
    duration = max(client_completion_end.values())
    if duration <= 0:
        raise ValueError("serving request completion window must be positive")
    total_tokens = sum(request.input_tokens + expected_outputs[request.request_id]
                       for request in requests)
    return ServingMetrics(
        requests=per_request,
        request_throughput_per_s=len(requests) / duration,
        output_token_throughput_per_s=sum(expected_outputs.values()) / duration,
        token_throughput_per_s=total_tokens / duration,
    )


def engine_clock_first_schedule_start_s(
    schedule_trace: ScheduleTrace,
    requests: RequestStream,
) -> dict[str, float]:
    """First-schedule instant of every request on the engine's own clock.

    An observed Scheduler trace carries no client clock.  Its timestamps are
    the engine's monotonic counter, normalized so that the run's first
    ``Scheduler.schedule`` call sits at the first request's arrival, and a
    measured wait is read off that normalized timeline.  The Host admission
    path that runs before that first ``schedule`` call is therefore outside
    the observed wait by construction, which is why observed waits go negative
    for requests admitted faster than the first one.

    The simulated trace runs on the client clock, where the same admission
    path is inside the wait.  Reporting the simulated first-schedule instant
    against the simulated run's own first step puts both on the engine clock,
    so a predicted wait covers the Scheduler queueing an observed wait covers
    and nothing else.  Total latency is untouched: the admission cost is not
    removed from the model, only attributed to the segment the observed
    boundary puts it in.

    The origin is one run's first admission, so a request admitted faster than
    that one -- a shorter prompt spends less time in the tokenizer -- lands
    marginally before its own arrival on this clock.  A request is never
    scheduled before it arrives, so the instant is held at the arrival, the
    same bound the observed breakdown applies to its own normalized trace.
    """

    entries = schedule_trace.entries
    if not entries:
        raise ValueError("an engine clock needs at least one scheduled step")
    arrival_s = {request.request_id: request.arrival_time_s
                 for request in requests}
    # A unified trace can interleave two pools, so the run's first schedule is
    # the earliest step start rather than the first entry.
    origin_s = min(entry.start_time_s for entry in entries)
    first_schedule_start_s: dict[str, float] = {}
    for entry in entries:
        for scheduled in entry.backend_step.schedule.requests:
            request_id = scheduled.request_id
            if request_id in first_schedule_start_s:
                continue
            if request_id not in arrival_s:
                raise ValueError(
                    f"schedule contains unknown request {request_id!r}")
            first_schedule_start_s[request_id] = max(
                arrival_s[request_id], entry.start_time_s - origin_s)
    return first_schedule_start_s
