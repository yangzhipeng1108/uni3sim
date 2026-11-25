"""Drive one scheduler backend on a virtual clock."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Iterable

from src.serving.request_gen.closed_loop import ClosedLoopClients
from src.serving.request_gen.request_stream import RequestStream

from .backend_protocol import BulkRequestSchedulerBackend, SchedulerBackend
from .pipeline_depth import validate_scheduler_pipeline_depth
from .schedule_step import BackendStep
from .schedule_trace import ScheduleTrace, ScheduleTraceEntry
from src.serving.host_timing import validate_serving_timing_mode


StepDuration = Callable[[BackendStep], float]
StepSubmit = Callable[[BackendStep], None]
StepExecutionAt = Callable[[BackendStep, float, float], tuple[float, float]]
StepComplete = Callable[[BackendStep, float], None]
TailDrain = Callable[
    [SchedulerBackend],
    Iterable[tuple[BackendStep, float] | tuple[BackendStep, float, float]],
]


@dataclass
class _InFlightStep:
    step: BackendStep
    plan_start_s: float
    scheduler_plan_cpu_s: float
    engine_submit_host_s: float
    engine_wake_host_s: float
    execution_start_s: float | None
    execution_end_s: float | None
    execution_wall_s: float | None
    preplanned_update_cpu_s: float | None = None
    duration_pending: bool = False


def run_single_pool_schedule(
    backend: SchedulerBackend,
    requests: RequestStream,
    step_duration: StepDuration,
    scheduler_pipeline_depth: int,
    *,
    async_output_release: str,
    on_step_complete: StepComplete | None = None,
    tail_drain: TailDrain | None = None,
    timing_mode: str = "device_only",
    closed_loop: ClosedLoopClients | None = None,
    submit_step_duration: StepSubmit | None = None,
    step_execution_at: StepExecutionAt | None = None,
    engine_core_worker_call_is_synchronous: bool = False,
) -> ScheduleTrace:
    """Drive the original FIFO clock, optionally submitting pure CPU work early.

    With submit_step_duration, pure CPU preparation starts early. At FIFO
    resolution step_duration supplies a duration, or step_execution_at consumes
    the actual Host/Device origins and supplies both services. Unknown device
    ends remain unknown; only a known Host boundary may advance the clock.
    """
    if on_step_complete is not None and not callable(on_step_complete):
        raise TypeError("single-pool step-complete callback must be callable")
    if tail_drain is not None and not callable(tail_drain):
        raise TypeError("single-pool tail drain must be callable")
    if closed_loop is not None and not isinstance(
            closed_loop, ClosedLoopClients):
        raise TypeError(
            "single-pool closed_loop must be a ClosedLoopClients")
    if step_execution_at is not None and not callable(step_execution_at):
        raise TypeError("single-pool step execution callback must be callable")
    if not isinstance(engine_core_worker_call_is_synchronous, bool):
        raise TypeError("EngineCore worker call mode must be a bool")
    if engine_core_worker_call_is_synchronous and step_execution_at is None:
        raise ValueError("synchronous EngineCore worker call requires step execution callback")
    depth = validate_scheduler_pipeline_depth(
        scheduler_pipeline_depth, "single-pool scheduler pipeline depth")
    # Which in-flight step's device completion releases the oldest output.
    # "self" follows vLLM 0.18 core.py:492-506, which blocks on the oldest
    # future, and gpu_model_runner.py:241-253/261, which records that step's
    # copy event behind its own work. "successor" remains a declared legacy
    # approximation; same-run 910B QPS16/QPS64 device boundaries rule out a
    # universal successor release; preserve FIFO and graph dependency semantics.
    if async_output_release == "self":
        release_index = 0
    elif async_output_release == "successor":
        release_index = -1
    else:
        raise ValueError(
            "single-pool async output release must be 'self' or 'successor', "
            f"got {async_output_release!r}")
    validate_serving_timing_mode(timing_mode)
    if submit_step_duration is not None and engine_core_worker_call_is_synchronous:
        raise ValueError("a synchronous EngineCore worker call cannot defer its Host return")
    if submit_step_duration is not None:
        if not callable(submit_step_duration):
            raise TypeError("single-pool duration submit callback must be callable")
        if (depth != 2 or release_index != 0
                or timing_mode != "device_plus_launch_plus_scheduler"
                or tail_drain is not None or closed_loop is not None):
            raise NotImplementedError(
                "deferred CPU execution requires depth-two self release, full Host timing, "
                "and no tail drain or closed-loop client")
    if tail_drain is not None and not (
            (depth == 1 and timing_mode == "device_only")
            or (depth == 2 and timing_mode == "device_plus_launch_plus_scheduler")):
        raise NotImplementedError(
            "single-pool tail drain requires depth-one device-only or depth-two full Host timing")
    preplanned_steps = None
    preplanned_active = False
    pending = iter(requests.requests)
    next_request = next(pending, None)
    pending_injected: list = []
    virtual_time = 0.0
    cpu_ready_s = 0.0
    device_ready_s = 0.0
    worker_ready_s = 0.0
    in_flight: list[_InFlightStep] = []
    pending_update: tuple[_InFlightStep, float] | None = None
    waiting_for_output = False
    entries: list[ScheduleTraceEntry] = []
    # Closed-loop arrivals pending admission, kept in injection (callback)
    # order.  Admission scans the whole list each pass and takes every
    # request whose arrival instant has passed, so per-lane external gaps
    # stay causal regardless of callback order; next-time events below use
    # the earliest pending arrival.
    # Request IDs the closed-loop client has already been notified about;
    # backends may keep a finished request scheduled in later steps (the
    # closed-loop contract is one notification per request lifetime).
    notified_finished: set = set()

    def checked_duration(value, name="duration"):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"predicted step {name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"predicted step {name} must be finite and non-negative")
        return float(value)

    def checked_worker_execution(step, start_s, floor_s):
        duration_s, worker_host_s = step_execution_at(step, start_s, floor_s)
        duration_s = checked_duration(duration_s)
        worker_host_s = checked_duration(worker_host_s, "worker Host")
        if start_s + duration_s < floor_s:
            raise ValueError("cross-step execution cannot finish before the prior device step")
        return duration_s, worker_host_s

    def resolve_oldest_duration():
        nonlocal device_ready_s, worker_ready_s
        current = in_flight[0]
        if not current.duration_pending:
            return
        host_start_s = (current.plan_start_s + current.scheduler_plan_cpu_s
            + current.engine_submit_host_s)
        if step_execution_at is None:
            duration = checked_duration(step_duration(current.step))
            current.execution_start_s = max(host_start_s, device_ready_s)
        else:
            current.execution_start_s = max(host_start_s, worker_ready_s)
            duration, worker_host_s = checked_worker_execution(
                current.step, current.execution_start_s, device_ready_s)
            worker_ready_s = current.execution_start_s + worker_host_s
        current.execution_end_s = current.execution_start_s + duration
        current.execution_wall_s = duration
        current.duration_pending = False
        device_ready_s = current.execution_end_s

    def inject_successors(step: BackendStep, completion_time_s: float) -> None:
        if closed_loop is None:
            return
        for request_id in step.finished_request_ids:
            if request_id in notified_finished:
                # The backend kept a finished request scheduled in later
                # steps (e.g. the fixture backends that mark every step
                # finished); the closed-loop contract is one notification
                # per request lifetime.
                continue
            notified_finished.add(request_id)
            successors = closed_loop.successors_of(
                request_id, completion_time_s)
            for successor in successors:
                pending_injected.append(successor)

    while (next_request is not None or pending_injected or backend.has_work()
           or in_flight or pending_update is not None or preplanned_active):
        if pending_update is not None and cpu_ready_s <= virtual_time:
            completed, completion_time_s = pending_update
            if not in_flight or in_flight[0] is not completed:
                raise RuntimeError("single-pool Scheduler slot order is inconsistent")
            in_flight.pop(0)
            waiting_for_output = False
            if on_step_complete is not None:
                on_step_complete(completed.step, completion_time_s)
            inject_successors(completed.step, completion_time_s)
            pending_update = None

        arrived_requests = []
        seen_arrivals: set = set()
        # Admit *every* injected successor whose arrival instant has passed,
        # independent of callback order: two requests finishing in one step
        # may yield successor arrivals 10 s and 1 s; the 1 s one must not
        # wait behind the 10 s one.
        still_pending_injected: list = []
        for injected in pending_injected:
            if injected.arrival_time_s <= virtual_time:
                arrived_requests.append(injected)
                seen_arrivals.add(id(injected))
            else:
                still_pending_injected.append(injected)
        pending_injected = still_pending_injected
        while next_request is not None and next_request.arrival_time_s <= virtual_time:
            if id(next_request) not in seen_arrivals:
                arrived_requests.append(next_request)
                seen_arrivals.add(id(next_request))
            next_request = next(pending, None)
        if arrived_requests:
            if isinstance(backend, BulkRequestSchedulerBackend):
                backend.add_requests(tuple(arrived_requests))
            else:
                for arrived_request in arrived_requests:
                    backend.add_request(arrived_request)

        if (next_request is None and not pending_injected and not preplanned_active
                and tail_drain is not None
                and ((depth == 1 and timing_mode == "device_only"
                    and not in_flight and pending_update is None and backend.has_work())
                    or (depth == 2 and timing_mode == "device_plus_launch_plus_scheduler"
                        and backend.has_work()))):
            if depth == 1 and timing_mode == "device_only":
                for step, duration_s in tail_drain(backend):
                    if isinstance(duration_s, bool) or not isinstance(duration_s, Real):
                        raise TypeError("predicted step duration must be numeric")
                    if not math.isfinite(duration_s) or duration_s < 0:
                        raise ValueError(
                            "predicted step duration must be finite and non-negative")
                    completion_time_s = virtual_time + float(duration_s)
                    entries.append(ScheduleTraceEntry(
                        step,
                        virtual_time,
                        completion_time_s,
                        timing_mode,
                        0.0,
                        0.0,
                        float(duration_s),
                    ))
                    if on_step_complete is not None:
                        on_step_complete(step, completion_time_s)
                    inject_successors(step, completion_time_s)
                    virtual_time = completion_time_s
                if backend.has_work():
                    raise RuntimeError(
                        "single-pool tail drain returned with pending work")
                if pending_injected:
                    # A finished client's successor arrived during the drain;
                    # return to the main loop so it is scheduled causally.
                    continue
                break
            for current in in_flight:
                current.preplanned_update_cpu_s = backend.scheduler_update_cpu_s(
                    current.step, timing_mode)
            preplanned_steps = iter(tail_drain(backend))
            preplanned_active = True

        if (pending_update is None and not waiting_for_output and len(in_flight) < depth
                and cpu_ready_s <= virtual_time
                and (preplanned_active or backend.has_work())):
            step = None
            duration_s = None
            update_from_plan = None
            exhausted = False
            if preplanned_active:
                planned = next(preplanned_steps, None)
                if planned is None:
                    preplanned_active = False
                    exhausted = True
                else:
                    if not isinstance(planned, tuple) or len(planned) != 3:
                        raise TypeError("preplanned tail requires step, duration and update CPU")
                    step, duration_s, update_from_plan = planned
            else:
                step = backend.plan_step(virtual_time)
            if step is None:
                # A source-derived backend can have only submitted steps left;
                # EngineCore then waits for the oldest future rather than
                # inventing a new batch.
                if not in_flight and not exhausted:
                    raise RuntimeError(
                        "backend has work but plan_step returned no step")
                if in_flight:
                    # An empty source decision enters future.result(). New
                    # arrivals may queue, but cannot interrupt that blocking
                    # call and start another plan before the oldest output.
                    waiting_for_output = True
            else:
                deferred = duration_s is None and submit_step_duration is not None
                if deferred:
                    if submit_step_duration(step) is not None:
                        raise TypeError("duration submit callback must return None")
                elif step_execution_at is None:
                    if duration_s is None:
                        duration_s = step_duration(step)
                    duration_s = checked_duration(duration_s)
                scheduler_plan_cpu_s = {
                    "device_only": 0.0,
                    "device_plus_launch": 0.0,
                    "device_plus_launch_plus_scheduler": step.scheduler_plan_cpu_s,
                }[timing_mode]
                engine_submit_host_s = {
                    "device_only": 0.0,
                    "device_plus_launch": 0.0,
                    "device_plus_launch_plus_scheduler": step.engine_submit_host_s,
                }[timing_mode]
                engine_wake_host_s = {
                    "device_only": 0.0,
                    "device_plus_launch": 0.0,
                    "device_plus_launch_plus_scheduler": step.engine_wake_host_s,
                }[timing_mode]
                # The EngineCore busy loop pays its submit-lane host chain
                # (input-prep sync, pinned H2D, pre-replay sync wall cost,
                # sampler/thread submits, queue poll) after planning; the
                # device start still waits for both the host and the device.
                if step_execution_at is None or deferred:
                    execution_start_s = (None if deferred else max(
                        virtual_time + scheduler_plan_cpu_s + engine_submit_host_s,
                        device_ready_s))
                    execution_end_s = None if deferred else execution_start_s + duration_s
                else:
                    # WorkerProc.worker_busy_loop runs one execute_model Host
                    # call at a time. Its async output thread may wait for
                    # the preceding Device event while the worker launches
                    # the next step. The callback retains the per-call DAG.
                    execution_start_s = max(virtual_time + scheduler_plan_cpu_s + engine_submit_host_s, worker_ready_s)
                    duration_s, worker_host_s = checked_worker_execution(
                        step, execution_start_s, device_ready_s)
                    execution_end_s = execution_start_s + float(duration_s)
                    worker_ready_s = execution_start_s + float(worker_host_s)
                in_flight.append(_InFlightStep(
                    step,
                    virtual_time,
                    scheduler_plan_cpu_s,
                    engine_submit_host_s,
                    engine_wake_host_s,
                    execution_start_s,
                    execution_end_s,
                    None if deferred else duration_s,
                    update_from_plan,
                    deferred,
                ))
                cpu_ready_s = (
                    virtual_time + scheduler_plan_cpu_s + engine_submit_host_s)
                if engine_core_worker_call_is_synchronous:
                    # UniProcExecutor.collective_rpc(non_block=True) calls
                    # run_method(driver_worker, ...) on the EngineCore thread.
                    # Only get_output() is handed to its async output thread;
                    # MultiprocExecutor instead enqueues the RPC immediately.
                    # The next Scheduler plan therefore waits for this call's
                    # Host return, even while its own Device event is pending.
                    cpu_ready_s = max(cpu_ready_s, worker_ready_s)
                if not deferred:
                    device_ready_s = execution_end_s
                if len(in_flight) < depth:
                    releasing = in_flight[release_index]
                    if releasing.duration_pending:
                        host_boundary = (releasing.plan_start_s
                            + releasing.scheduler_plan_cpu_s + releasing.engine_submit_host_s)
                        if host_boundary > virtual_time:
                            continue
                        # A zero Host boundary permits a zero device end now;
                        # determine it before admitting another plan.
                        resolve_oldest_duration()
                    if releasing.execution_end_s > virtual_time:
                        # vLLM 0.18 core.py:476-482: filling the batch queue
                        # has priority over getting model outputs.
                        continue

        if in_flight and in_flight[0].duration_pending:
            # A busy EngineCore cannot consume a Device completion yet.
            # Advance its known Host boundary (or an earlier arrival) first,
            # then preserve the normal queue-fill priority before resolving
            # the oldest pure CPU result. Zero-cost Host boundaries still
            # resolve here, including a potentially zero Device duration.
            if cpu_ready_s <= virtual_time:
                resolve_oldest_duration()

        if (pending_update is None and in_flight
                and not in_flight[release_index].duration_pending
                and in_flight[release_index].execution_end_s <= virtual_time
                and cpu_ready_s <= virtual_time):
            completed = in_flight[0]
            update_value = (completed.preplanned_update_cpu_s
                if completed.preplanned_update_cpu_s is not None
                else backend.complete_step(completed.step, virtual_time))
            if timing_mode == "device_plus_launch_plus_scheduler":
                if (isinstance(update_value, bool)
                        or not isinstance(update_value, Real)
                        or not math.isfinite(update_value)
                        or update_value < 0):
                    raise ValueError(
                        "Scheduler update CPU time must be finite and non-negative")
                scheduler_update_cpu_s = float(update_value)
            elif timing_mode in ("device_only", "device_plus_launch"):
                scheduler_update_cpu_s = 0.0
            else:
                raise ValueError(f"unsupported Serving timing mode: {timing_mode!r}")
            # The worker thread wakes (D2H event sync, tolist, output put)
            # after the releasing step's device work finishes; the Scheduler
            # update consumes the woken output, so the completion queue wait
            # includes the wake host chain.
            update_start_s = max(
                virtual_time,
                in_flight[release_index].execution_end_s
                + completed.engine_wake_host_s)
            completion_time_s = update_start_s + scheduler_update_cpu_s
            entries.append(ScheduleTraceEntry(
                completed.step,
                completed.plan_start_s,
                completion_time_s,
                timing_mode,
                completed.scheduler_plan_cpu_s,
                scheduler_update_cpu_s,
                completed.execution_wall_s,
                completed.execution_start_s - (
                    completed.plan_start_s + completed.scheduler_plan_cpu_s),
                update_start_s - completed.execution_end_s,
            ))
            pending_update = (completed, completion_time_s)
            cpu_ready_s = completion_time_s
            if on_step_complete is not None:
                on_step_complete(completed.step, completion_time_s)
            inject_successors(completed.step, completion_time_s)
            pending_update = None
            in_flight.pop(0)
            waiting_for_output = False
            continue

        next_times = []
        if next_request is not None:
            next_times.append(next_request.arrival_time_s)
        if pending_injected:
            # Scan *all* pending injected arrivals for the next causal event,
            # not just the first in callback order: two successors injected
            # by one step may arrive 10 s and 1 s out; the 1 s one must
            # advance the engine clock first.
            next_times.append(
                min(item.arrival_time_s for item in pending_injected))
        if pending_update is not None and cpu_ready_s > virtual_time:
            next_times.append(cpu_ready_s)
        elif cpu_ready_s > virtual_time:
            next_times.append(cpu_ready_s)
        if (in_flight
                and not in_flight[release_index].duration_pending
                and in_flight[release_index].execution_end_s > virtual_time):
            next_times.append(in_flight[release_index].execution_end_s)
        if not next_times:
            if backend.has_work():
                raise RuntimeError(
                    "single-pool Scheduler has work but no causal event")
            break
        next_time = min(next_times)
        if next_time <= virtual_time:
            raise RuntimeError("single-pool Scheduler made no causal progress")
        virtual_time = next_time

    return ScheduleTrace(sorted(entries, key=lambda entry: entry.start_time_s))
