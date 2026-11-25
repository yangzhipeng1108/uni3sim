"""Drive multiple worker-pool scheduler backends on a shared virtual clock."""

from collections import deque
import heapq
import math
from dataclasses import dataclass, replace
from numbers import Real
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Optional, Tuple

from src.serving.deployment.deployment_config import RouterConfig
from src.serving.request_gen.closed_loop import ClosedLoopClients
from src.serving.request_gen.request_stream import RequestSpec, RequestStream

from .backend_protocol import SchedulerBackend, backend_phase
from .pipeline_depth import validate_scheduler_pipeline_depths
from .schedule_step import BackendStep
from .schedule_trace import (
    RouteDecision,
    ScheduleTrace,
    ScheduleTraceEntry,
    TransferTraceEntry,
    normalize_transfer_reserved_resource_ids,
    normalize_transfer_bandwidth_segments,
)
from src.serving.host_timing import validate_serving_timing_mode


_COMPLETION_PRIORITY = 0
_TRANSFER_COMPLETION_PRIORITY = 1
_ARRIVAL_PRIORITY = 2
_ROUTE_DISPATCH_PRIORITY = 3
_CPU_READY_PRIORITY = 4


@dataclass(frozen=True)
class HandoffTransfer:
    transfer_id: str
    transfer_duration_s: float
    transfer_resource_id: str
    transfer_link: str
    transfer_reserved_resource_ids: Tuple[str, ...]
    transfer_byte_count: int
    transfer_submission_s: float = 0.0
    reserved_interval: Tuple[float, float] | None = None
    bandwidth_segments: Tuple[Tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.transfer_id, str) or not self.transfer_id:
            raise ValueError("handoff transfer ID must be non-empty")
        if (isinstance(self.transfer_duration_s, bool)
                or not isinstance(self.transfer_duration_s, Real)
                or not math.isfinite(self.transfer_duration_s)
                or self.transfer_duration_s < 0):
            raise ValueError(
                "handoff transfer duration must be finite and non-negative")
        if (isinstance(self.transfer_submission_s, bool)
                or not isinstance(self.transfer_submission_s, Real)
                or not math.isfinite(self.transfer_submission_s)
                or self.transfer_submission_s < 0):
            raise ValueError(
                "handoff transfer submission must be finite and non-negative")
        if self.reserved_interval is not None:
            if (not isinstance(self.reserved_interval, tuple)
                    or len(self.reserved_interval) != 2
                    or any(isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or value < 0
                        for value in self.reserved_interval)):
                raise ValueError("handoff reserved interval must be a finite nonnegative pair")
            start, end = self.reserved_interval
            if end < start or not math.isclose(
                    end - start, self.transfer_duration_s, rel_tol=1e-12,
                    abs_tol=max(1e-15, 4 * max(math.ulp(start), math.ulp(end)))):
                raise ValueError("handoff reserved interval disagrees with physical duration")
        if (not isinstance(self.transfer_resource_id, str)
                or not self.transfer_resource_id):
            raise ValueError("handoff transfer resource ID must be non-empty")
        if not isinstance(self.transfer_link, str) or not self.transfer_link:
            raise ValueError("handoff transfer link must be non-empty")
        reserved_resources = normalize_transfer_reserved_resource_ids(
            self.transfer_resource_id,
            self.transfer_link,
            self.transfer_reserved_resource_ids,
        )
        object.__setattr__(
            self, "transfer_reserved_resource_ids", reserved_resources)
        if (isinstance(self.transfer_byte_count, bool)
                or not isinstance(self.transfer_byte_count, int)):
            raise TypeError("handoff transfer byte count must be an integer")
        if self.transfer_byte_count <= 0:
            raise ValueError("handoff transfer byte count must be positive")
        if not isinstance(self.bandwidth_segments, (list, tuple)):
            raise TypeError("handoff bandwidth segments require an explicit array")
        if self.bandwidth_segments and self.reserved_interval is None:
            raise ValueError("handoff bandwidth segments require their reserved interval")
        if self.reserved_interval is not None:
            object.__setattr__(self, "bandwidth_segments", normalize_transfer_bandwidth_segments(
                self.bandwidth_segments, *self.reserved_interval, self.transfer_byte_count))
        else:
            object.__setattr__(self, "bandwidth_segments", ())


@dataclass(frozen=True)
class HandoffDispatch:
    decision: RouteDecision
    request: RequestSpec
    transfers: Tuple[HandoffTransfer, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.decision, RouteDecision):
            raise TypeError("handoff decision must be a RouteDecision")
        if self.decision.stage != "prefill_to_decode":
            raise ValueError("handoff requires a prefill_to_decode decision")
        if not isinstance(self.request, RequestSpec):
            raise TypeError("handoff request must be a RequestSpec")
        if self.request.request_id != self.decision.request_id:
            raise ValueError("handoff request and route decision IDs disagree")
        if (not isinstance(self.transfers, (list, tuple))
                or not self.transfers
                or any(not isinstance(transfer, HandoffTransfer)
                       for transfer in self.transfers)):
            raise TypeError(
                "handoff transfers must contain HandoffTransfer objects")
        transfers = tuple(self.transfers)
        transfer_ids = tuple(transfer.transfer_id for transfer in transfers)
        if len(transfer_ids) != len(set(transfer_ids)):
            raise ValueError("handoff physical transfer IDs must be unique")
        object.__setattr__(self, "transfers", transfers)


def reserve_pd_handoff_destination(router, backends, candidates, request, source, now):
    """Commit only the least-loaded destination that really has native KV.

    Failed try-reservations are atomic/no-op in the native adapter. A fully
    backpressured head remains pending at P; no route or transfer is issued.
    The caller records the complete pre-admission snapshot and rejected IDs.
    """
    loads = router.queued_tokens
    if (not candidates or len(candidates) != len(set(candidates))
            or not set(candidates) <= set(loads) or not set(candidates) <= set(backends)):
        raise ValueError("PD capacity admission requires known distinct destinations")
    if (isinstance(now, bool) or not isinstance(now, Real) or not math.isfinite(now) or now < 0
            or source not in backends):
        raise ValueError("PD capacity admission requires a valid source and event time")
    snapshot = tuple((name, loads[name]) for name in sorted(candidates))
    rejected = []
    for name in sorted(candidates, key=lambda name: (loads[name], name)):
        blocks = backends[name].try_reserve_pd_destination(request)
        if blocks is None:
            rejected.append(name)
            continue
        decision = router.route(request.request_id, "prefill_to_decode", (name,),
            1, now, source_engine_id=source)
        return decision, blocks, snapshot, tuple(rejected)
    return None


class LeastQueuedTokensRouter:
    """Deterministic token-backlog router with committed-load reservations."""

    def __init__(
        self,
        deployment_id: str,
        engine_ids: Iterable[str],
        config: RouterConfig = RouterConfig(),
    ) -> None:
        if not isinstance(deployment_id, str) or not deployment_id:
            raise ValueError("router deployment_id must be non-empty")
        supplied = tuple(engine_ids)
        if not supplied:
            raise ValueError("router requires at least one engine")
        if any(not isinstance(engine_id, str) or not engine_id
               for engine_id in supplied):
            raise ValueError("router engine IDs must be non-empty strings")
        if len(supplied) != len(set(supplied)):
            raise ValueError("router engine IDs must be unique")
        if not isinstance(config, RouterConfig):
            raise TypeError("router config must be RouterConfig")
        self.deployment_id = deployment_id
        self.config = config
        self._queued_tokens = {
            engine_id: 0 for engine_id in sorted(supplied)
        }
        self._assignments: dict[tuple[str, str], tuple[str, int]] = {}
        self._decisions: list[RouteDecision] = []
        self._dispatch_resource_ready_s = 0.0

    @property
    def queued_tokens(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._queued_tokens))

    @property
    def decisions(self) -> Tuple[RouteDecision, ...]:
        return tuple(self._decisions)

    def route(
        self,
        request_id: str,
        stage: str,
        candidate_engine_ids: Iterable[str],
        token_count: int,
        now_s: float,
        *,
        source_engine_id: Optional[str] = None,
        queue_enter_time_s: Optional[float] = None,
    ) -> RouteDecision:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("routed request_id must be non-empty")
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise TypeError("routed token count must be an integer")
        if token_count <= 0:
            raise ValueError("routed token count must be positive")
        candidates = tuple(candidate_engine_ids)
        if not candidates:
            raise ValueError("routing requires at least one candidate engine")
        if len(candidates) != len(set(candidates)):
            raise ValueError("routing candidate engine IDs must be unique")
        unknown = sorted(set(candidates) - set(self._queued_tokens))
        if unknown:
            raise ValueError(f"routing references unknown engines: {unknown}")
        assignment_key = (stage, request_id)
        if assignment_key in self._assignments:
            raise ValueError(
                f"request {request_id!r} already has a {stage!r} assignment")
        if (isinstance(now_s, bool) or not isinstance(now_s, (int, float))
                or not math.isfinite(now_s) or now_s < 0):
            raise ValueError("routing time must be finite and non-negative")
        if queue_enter_time_s is None:
            queue_enter_time_s = now_s
        snapshot = tuple(
            (engine_id, self._queued_tokens[engine_id])
            for engine_id in sorted(candidates)
        )
        selected = min(snapshot, key=lambda item: (item[1], item[0]))[0]
        dispatch_start_time_s = max(
            float(now_s), self._dispatch_resource_ready_s)
        dispatch_time_s = (
            dispatch_start_time_s
            + self.config.dispatch_service_time_s)
        self._dispatch_resource_ready_s = dispatch_time_s
        decision = RouteDecision(
            deployment_id=self.deployment_id,
            request_id=request_id,
            stage=stage,
            policy="least_queued_tokens",
            source_engine_id=source_engine_id,
            candidate_queued_tokens=snapshot,
            selected_engine_id=selected,
            reason="minimum_visible_queued_tokens_then_engine_id",
            queue_enter_time_s=queue_enter_time_s,
            decision_time_s=now_s,
            queue_exit_time_s=dispatch_start_time_s,
            dispatch_time_s=dispatch_time_s,
        )
        self._queued_tokens[selected] += token_count
        self._assignments[assignment_key] = (selected, token_count)
        self._decisions.append(decision)
        return decision

    def set_request_tokens(
        self,
        request_id: str,
        stage: str,
        remaining_tokens: int,
        engine_id: str,
    ) -> None:
        if (isinstance(remaining_tokens, bool)
                or not isinstance(remaining_tokens, int)):
            raise TypeError("remaining routed tokens must be an integer")
        if remaining_tokens < 0:
            raise ValueError("remaining routed tokens cannot be negative")
        assignment_key = (stage, request_id)
        if assignment_key not in self._assignments:
            raise ValueError(
                f"request {request_id!r} has no {stage!r} assignment")
        if not isinstance(engine_id, str) or not engine_id:
            raise ValueError("updated route engine_id must be non-empty")
        assigned_engine_id, previous = self._assignments[assignment_key]
        if engine_id != assigned_engine_id:
            raise ValueError(
                f"request {request_id!r} is assigned to "
                f"{assigned_engine_id!r}, not {engine_id!r}")
        updated = (
            self._queued_tokens[assigned_engine_id]
            - previous + remaining_tokens
        )
        if updated < 0:
            raise RuntimeError("router queued-token ledger became negative")
        self._queued_tokens[assigned_engine_id] = updated
        if remaining_tokens == 0:
            del self._assignments[assignment_key]
        else:
            self._assignments[assignment_key] = (
                assigned_engine_id, remaining_tokens)

    def assert_drained(self) -> None:
        """Require every committed request reservation to be released."""

        nonzero = {
            engine_id: tokens
            for engine_id, tokens in self._queued_tokens.items()
            if tokens != 0
        }
        if nonzero or self._assignments:
            raise RuntimeError(
                "router queued-token ledger was not fully released: "
                f"queued_tokens={nonzero}, assignments={self._assignments}")


def run_multi_pool_schedule(
    backends: Mapping[str, SchedulerBackend],
    requests: RequestStream,
    route_request: Callable[[RequestSpec, float], RouteDecision],
    step_duration: Callable[[str, BackendStep, float], float],
    scheduler_pipeline_depths: Mapping[str, int],
    on_completion: Callable[
        [str, BackendStep, float], Iterable[HandoffDispatch]
    ] | None = None,
    reserve_transfer: Callable[
        [Tuple[str, ...], float, float], Tuple[float, float]
    ] | None = None,
    tail_drain: Callable[
        [str, SchedulerBackend, float],
        Iterable[tuple[BackendStep, float]] | None,
    ] | None = None,
    on_step_complete: Callable[[BackendStep, float], None] | None = None,
    timing_mode: str = "device_only",
    closed_loop: ClosedLoopClients | None = None,
    on_transfer_complete: Callable[[str, str, float], None] | None = None,
    has_pending_handoffs: Callable[[], bool] | None = None,
    observation_end_time_s: float | None = None,
    on_queue_observation: Callable[[float], None] | None = None,
) -> ScheduleTrace:
    if not backends:
        raise ValueError("multi-pool schedule requires at least one backend")
    if on_queue_observation is not None and not callable(on_queue_observation):
        raise TypeError("queue observation callback must be callable")
    if observation_end_time_s is not None and (isinstance(observation_end_time_s, bool)
            or not isinstance(observation_end_time_s, Real)
            or not math.isfinite(observation_end_time_s) or observation_end_time_s <= 0):
        raise ValueError("observation end must be finite and positive")
    if observation_end_time_s is not None and timing_mode != "device_only":
        raise NotImplementedError("finite observations currently support device-only timing")
    if on_step_complete is not None and not callable(on_step_complete):
        raise TypeError("multi-pool step-complete callback must be callable")
    if tail_drain is not None and not callable(tail_drain):
        raise TypeError("multi-pool tail drain must be callable")
    if has_pending_handoffs is not None and not callable(has_pending_handoffs):
        raise TypeError("multi-pool pending-handoff query must be callable")
    if closed_loop is not None and not isinstance(
            closed_loop, ClosedLoopClients):
        raise TypeError("multi-pool closed_loop must be a ClosedLoopClients")
    validate_serving_timing_mode(timing_mode)
    depths = validate_scheduler_pipeline_depths(
        scheduler_pipeline_depths,
        set(backends),
        "multi-pool scheduler pipeline depths",
    )
    events = [(request.arrival_time_s, _ARRIVAL_PRIORITY, index,
               "arrival", request)
              for index, request in enumerate(requests.requests)]
    heapq.heapify(events)
    cpu_ready_at = {name: 0.0 for name in backends}
    device_ready_at = {name: 0.0 for name in backends}
    last_admission_at = {name: 0.0 for name in backends}
    in_flight_by_pool = {name: deque() for name in backends}
    entries = []
    pending_entry_by_step_id = {}
    route_decisions = []
    transfers = []
    transfer_ready_at: dict[str, float] = {}
    completed_prefills: dict[tuple[str, str], float] = {}
    sequence = len(events)
    last_observed_time_s = None

    def observe_queues(now):
        nonlocal last_observed_time_s
        if on_queue_observation is not None:
            on_queue_observation(now)
            last_observed_time_s = now

    def handoff_waiting() -> bool:
        # Capacity waits live in the caller, outside the event heap. D must
        # progress through normal completions so it can wake a waiting P copy.
        if has_pending_handoffs is None:
            return False
        waiting = has_pending_handoffs()
        if not isinstance(waiting, bool):
            raise TypeError("multi-pool pending-handoff query must return bool")
        return waiting

    def inject_closed_loop_successors(
            pool_name: str, step: BackendStep, completion_time_s: float) -> int:
        """Inject finished clients' successors as new arrival events.

        Only a terminal pool (decode or colocated phase) completes a client
        turn; a prefill handoff finish is not a client-visible completion.
        """

        nonlocal sequence
        if closed_loop is None:
            return 0
        if backend_phase(backends[pool_name].config.role) not in (
                "decode", "colocated"):
            return 0
        injected = 0
        for request_id in step.finished_request_ids:
            successor = closed_loop.notify_finished(
                request_id, completion_time_s)
            if successor is not None:
                heapq.heappush(events, (
                    completion_time_s,
                    _ARRIVAL_PRIORITY,
                    sequence,
                    "arrival",
                    successor,
                ))
                sequence += 1
                injected += 1
        return injected

    while (events or any(backend.has_work() for backend in backends.values())
           or any(in_flight_by_pool.values())):
        active = tuple(
            pool_name for pool_name, backend in backends.items()
            if backend.has_work()
        )
        has_external_event = any(
            event[3] in ("arrival", "transfer_completion", "route_dispatch")
            for event in events
        )
        if (tail_drain is not None and observation_end_time_s is None and on_queue_observation is None
                and timing_mode == "device_only"
                and len(active) == 1
                and not any(in_flight_by_pool.values())
                and not has_external_event
                and not handoff_waiting()):
            drained = tail_drain(
                active[0],
                backends[active[0]],
                max(
                    max(
                        (entry.end_time_s for entry in entries),
                        default=0.0,
                    ),
                    last_admission_at[active[0]],
                ),
            )
            if drained is not None:
                tail_time = max(
                    max(
                        (entry.end_time_s for entry in entries),
                        default=0.0,
                    ),
                    last_admission_at[active[0]],
                )
                injected = 0
                for step, duration_s in drained:
                    if (isinstance(duration_s, bool)
                            or not isinstance(duration_s, Real)
                            or not math.isfinite(duration_s)
                            or duration_s < 0):
                        raise TypeError(
                            "multi-pool tail drain returned an invalid "
                            "step duration")
                    completion_time = tail_time + float(duration_s)
                    handoffs = ()
                    if on_completion is not None:
                        handoffs = tuple(on_completion(
                            active[0], step, completion_time))
                    if handoffs:
                        raise RuntimeError(
                            "multi-pool tail drain produced handoffs")
                    entries.append(ScheduleTraceEntry(
                        step,
                        tail_time,
                        completion_time,
                        timing_mode,
                        0.0,
                        0.0,
                        float(duration_s),
                    ))
                    if on_step_complete is not None:
                        on_step_complete(step, completion_time)
                    injected += inject_closed_loop_successors(
                        active[0], step, completion_time)
                    tail_time = completion_time
                if backends[active[0]].has_work():
                    raise RuntimeError(
                        "multi-pool tail drain returned with pending work")
                if injected:
                    # Finished clients' successors arrived during the drain;
                    # return to the main loop so they are scheduled causally.
                    continue
                break
        if not events:
            raise RuntimeError("multi-pool scheduler has work but no causal event")
        now = events[0][0]
        if observation_end_time_s is not None and now > observation_end_time_s:
            observe_queues(observation_end_time_s)
            break
        while events and events[0][0] <= now:
            _, _, _, event_kind, payload = heapq.heappop(events)
            if event_kind == "arrival":
                decision = route_request(payload, now)
                if not isinstance(decision, RouteDecision):
                    raise TypeError("route_request must return RouteDecision")
                if decision.request_id != payload.request_id:
                    raise ValueError(
                        "arrival route decision request ID disagrees with input")
                if decision.stage not in (
                        "ingress_to_colocated", "ingress_to_prefill"):
                    raise ValueError(
                        "arrival route decision has an invalid route stage")
                if decision.decision_time_s != now:
                    raise ValueError(
                        "arrival route decision time must equal event time")
                pool_name = decision.selected_engine_id
                if pool_name not in backends:
                    raise ValueError(f"request routed to unknown pool {pool_name!r}")
                route_decisions.append(decision)
                heapq.heappush(events, (
                    decision.dispatch_time_s,
                    _ROUTE_DISPATCH_PRIORITY,
                    sequence,
                    "route_dispatch",
                    (pool_name, payload),
                ))
                sequence += 1
            elif event_kind == "completion":
                pool_name, step = payload
                pending_pool = in_flight_by_pool[pool_name]
                if (not pending_pool
                        or pending_pool[0]["step"] is not step):
                    raise RuntimeError(
                        "multi-pool Scheduler completion is not FIFO")
                pending = pending_pool[0]
                if cpu_ready_at[pool_name] > now:
                    heapq.heappush(events, (
                        cpu_ready_at[pool_name],
                        _COMPLETION_PRIORITY,
                        sequence,
                        "completion",
                        (pool_name, step),
                    ))
                    sequence += 1
                    continue
                update_value = backends[pool_name].complete_step(step, now)
                scheduler_timing_enabled = (
                    timing_mode == "device_plus_launch_plus_scheduler")
                if scheduler_timing_enabled:
                    if (isinstance(update_value, bool)
                            or not isinstance(update_value, Real)
                            or not math.isfinite(update_value)
                            or update_value < 0):
                        raise ValueError(
                            "Scheduler update CPU time must be finite and non-negative")
                    scheduler_update_cpu_s = float(update_value)
                else:
                    scheduler_update_cpu_s = 0.0
                completion_time = now + scheduler_update_cpu_s
                pending_pool.popleft()
                pending_entry_by_step_id.pop(step.schedule.step_id)
                entries.append(ScheduleTraceEntry(
                    step,
                    pending["start_time_s"],
                    completion_time,
                    timing_mode,
                    pending["scheduler_plan_cpu_s"],
                    scheduler_update_cpu_s,
                    pending["execution_wall_s"],
                    pending["execution_start_s"] - (
                        pending["start_time_s"]
                        + pending["scheduler_plan_cpu_s"]),
                    now - pending["execution_end_s"],
                ))
                cpu_ready_at[pool_name] = completion_time
                heapq.heappush(events, (
                    completion_time,
                    _CPU_READY_PRIORITY,
                    sequence,
                    "cpu_ready",
                    pool_name,
                ))
                sequence += 1
                if on_step_complete is not None:
                    on_step_complete(step, completion_time)
                if backend_phase(backends[pool_name].config.role) == "prefill":
                    for request_id in step.finished_request_ids:
                        completed_prefills[(pool_name, request_id)] = completion_time
                if on_completion is not None:
                    for handoff in on_completion(
                            pool_name, step, completion_time):
                        if not isinstance(handoff, HandoffDispatch):
                            raise TypeError(
                                "on_completion must yield HandoffDispatch objects")
                        decision = handoff.decision
                        source = decision.source_engine_id
                        source_key = (source, handoff.request.request_id)
                        # Destination KV may become available at a later D
                        # completion. Such a handoff still needs a real earlier
                        # P completion, and each source can dispatch only once.
                        if (source_key not in completed_prefills
                                or completed_prefills[source_key] > completion_time):
                            raise ValueError(
                                "handoff source engine disagrees with completion or has no completed prefill")
                        if decision.decision_time_s != completion_time:
                            raise ValueError(
                                "handoff route decision time must equal completion")
                        del completed_prefills[source_key]
                        destination = decision.selected_engine_id
                        if destination not in backends:
                            raise ValueError(
                                f"handoff routed to unknown pool {destination!r}")
                        transfer_end_times = []
                        for physical in handoff.transfers:
                            # Host submission makes the copy ready; it does
                            # not occupy endpoint/fabric device resources.
                            transfer_release_time = (
                                decision.dispatch_time_s
                                + physical.transfer_submission_s)
                            reserved_resources = (
                                physical.transfer_reserved_resource_ids)
                            if (physical.reserved_interval is not None
                                    or reserve_transfer is not None):
                                # Production's execution timeline can reserve
                                # once and pass that canonical interval here.
                                reserved = (
                                    physical.reserved_interval
                                    if physical.reserved_interval is not None
                                    else reserve_transfer(
                                        reserved_resources,
                                        transfer_release_time,
                                        physical.transfer_duration_s))
                                if (not isinstance(reserved, (list, tuple))
                                        or len(reserved) != 2):
                                    raise TypeError(
                                        "reserve_transfer must return a "
                                        "start/end pair")
                                transfer_start, transfer_end = reserved
                                if any(
                                        isinstance(value, bool)
                                        or not isinstance(value, Real)
                                        or not math.isfinite(value)
                                        for value in reserved):
                                    raise ValueError(
                                        "reserved transfer interval must be "
                                        "finite")
                                if (transfer_start < transfer_release_time
                                        or transfer_end < transfer_start
                                        or not math.isclose(
                                            transfer_end - transfer_start,
                                            physical.transfer_duration_s,
                                            rel_tol=1e-12,
                                            abs_tol=max(
                                                1e-15,
                                                4.0 * max(
                                                    math.ulp(transfer_start),
                                                    math.ulp(transfer_end),
                                                    math.ulp(
                                                        physical.transfer_duration_s),
                                                ),
                                            ),
                                        )):
                                    raise ValueError(
                                        "reserved transfer interval disagrees "
                                        "with its release time or predicted "
                                        "duration")
                            else:
                                resource_ready = 0.0
                                for resource_id in reserved_resources:
                                    if resource_id in transfer_ready_at:
                                        resource_ready = max(
                                            resource_ready,
                                            transfer_ready_at[resource_id],
                                        )
                                transfer_start = max(
                                    transfer_release_time, resource_ready)
                                transfer_end = (
                                    transfer_start
                                    + physical.transfer_duration_s)
                                for resource_id in reserved_resources:
                                    transfer_ready_at[resource_id] = transfer_end
                            transfers.append(TransferTraceEntry(
                                transfer_id=physical.transfer_id,
                                request_ids=(handoff.request.request_id,),
                                payload="kv",
                                source_pool_id=source,
                                destination_pool_id=destination,
                                resource_id=physical.transfer_resource_id,
                                link=physical.transfer_link,
                                reserved_resource_ids=reserved_resources,
                                byte_count=physical.transfer_byte_count,
                                start_time_s=transfer_start,
                                end_time_s=transfer_end,
                                bandwidth_segments=physical.bandwidth_segments,
                            ))
                            transfer_end_times.append(transfer_end)
                        transfer_end = max(transfer_end_times)
                        route_decisions.append(decision)
                        admitted_request = replace(
                            handoff.request, arrival_time_s=transfer_end)
                        heapq.heappush(events, (
                            transfer_end,
                            _TRANSFER_COMPLETION_PRIORITY,
                            sequence,
                            "transfer_completion",
                            (destination, admitted_request, source),
                        ))
                        sequence += 1
                inject_closed_loop_successors(pool_name, step, completion_time)
            elif event_kind == "transfer_completion":
                destination, request, source = payload
                if destination not in backends:
                    raise ValueError(
                        f"transfer completed for unknown pool {destination!r}")
                if on_transfer_complete is not None:
                    on_transfer_complete(source, request.request_id, now)
                backends[destination].add_request(request)
                last_admission_at[destination] = now
            elif event_kind == "route_dispatch":
                destination, request = payload
                if destination not in backends:
                    raise ValueError(
                        f"route dispatched to unknown pool {destination!r}")
                backends[destination].add_request(request)
                last_admission_at[destination] = now
            elif event_kind == "cpu_ready":
                if payload not in backends:
                    raise ValueError(
                        f"Scheduler CPU-ready event references unknown pool "
                        f"{payload!r}")
            else:
                raise ValueError(f"unknown scheduler event: {event_kind!r}")
        observe_queues(now)
        # A completed depth-one step leaves a bookkeeping CPU-ready event in
        # the heap.  When all real external events and in-flight work are
        # gone, discard that inert marker and let the top-of-loop tail path
        # batch the remaining decode steps.
        tail_active = tuple(
            pool_name for pool_name, backend in backends.items()
            if backend.has_work()
        )
        tail_has_external_event = any(
            event[3] in ("arrival", "transfer_completion", "route_dispatch")
            for event in events
        )
        if observation_end_time_s is not None and now >= observation_end_time_s:
            break
        if (tail_drain is not None and observation_end_time_s is None and on_queue_observation is None
                and timing_mode == "device_only"
                and len(tail_active) == 1
                and not any(in_flight_by_pool.values())
                and not tail_has_external_event
                and not handoff_waiting()
                and backends[tail_active[0]].config.role == "decode"):
            events = [event for event in events if event[3] != "cpu_ready"]
            heapq.heapify(events)
            continue
        progress = False
        for pool_name in sorted(backends):
            backend = backends[pool_name]
            if (not backend.has_work() or cpu_ready_at[pool_name] > now
                    or len(in_flight_by_pool[pool_name]) >= depths[pool_name]):
                continue
            step = backend.plan_step(now)
            if step is None:
                continue
            scheduler_plan_cpu_s = {
                "device_only": 0.0,
                "device_plus_launch": 0.0,
                "device_plus_launch_plus_scheduler": step.scheduler_plan_cpu_s,
            }[timing_mode]
            execution_start_time = max(
                now + scheduler_plan_cpu_s, device_ready_at[pool_name])
            duration = step_duration(pool_name, step, execution_start_time)
            if isinstance(duration, bool) or not isinstance(duration, Real):
                raise TypeError("predicted step duration must be numeric")
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(
                    "predicted step duration must be finite and non-negative")
            end_time = execution_start_time + duration
            if step.schedule.step_id in pending_entry_by_step_id:
                raise ValueError("duplicate pending Scheduler step ID")
            pending_entry_by_step_id[step.schedule.step_id] = {
                "start_time_s": now,
                "scheduler_plan_cpu_s": scheduler_plan_cpu_s,
                "execution_wall_s": float(duration),
                "execution_start_s": execution_start_time,
                "execution_end_s": end_time,
            }
            in_flight_by_pool[pool_name].append({
                "step": step,
                **pending_entry_by_step_id[step.schedule.step_id],
            })
            cpu_ready_at[pool_name] = now + scheduler_plan_cpu_s
            device_ready_at[pool_name] = end_time
            heapq.heappush(events, (
                end_time,
                _COMPLETION_PRIORITY,
                sequence,
                "completion",
                (pool_name, step),
            ))
            sequence += 1
            heapq.heappush(events, (
                cpu_ready_at[pool_name],
                _CPU_READY_PRIORITY,
                sequence,
                "cpu_ready",
                pool_name,
            ))
            sequence += 1
            progress = True
        if (not progress and not events
                and (any(backend.has_work() for backend in backends.values())
                     or any(in_flight_by_pool.values()))):
            raise RuntimeError("multi-pool scheduler deadlocked")
    if (observation_end_time_s is not None
            and (last_observed_time_s is None or last_observed_time_s < observation_end_time_s)):
        observe_queues(observation_end_time_s)
    return ScheduleTrace(
        sorted(
            entries,
            key=lambda entry: (
                entry.start_time_s,
                entry.backend_step.schedule.pool_name,
            ),
        ),
        route_decisions,
        sorted(
            transfers,
            key=lambda transfer: (
                transfer.start_time_s,
                transfer.transfer_id,
            ),
        ),
        observation_end_time_s=observation_end_time_s,
        pending_entries=tuple(ScheduleTraceEntry(
            pending["step"], pending["start_time_s"], pending["execution_end_s"])
            for pool in sorted(in_flight_by_pool)
            for pending in in_flight_by_pool[pool]),
    )
