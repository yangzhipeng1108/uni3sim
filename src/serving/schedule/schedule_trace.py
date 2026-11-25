"""Virtual-time scheduler, router, and transfer trace contracts."""

from dataclasses import dataclass
import math
from typing import Iterable, Iterator, Optional, Tuple

from .schedule_step import BackendStep
from src.serving.host_timing import validate_serving_timing_mode


_ROUTE_STAGES = {
    "ingress_to_colocated",
    "ingress_to_prefill",
    "prefill_to_decode",
}


def normalize_transfer_bandwidth_segments(segments, start, end, byte_count):
    """Validate optional constant-rate byte spans of one physical transfer."""
    if not isinstance(segments, (tuple, list)):
        raise TypeError("transfer bandwidth segments require an explicit array")
    normalized = tuple(tuple(row) for row in segments)
    previous = start
    for row in normalized:
        if (len(row) != 3 or any(isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) for value in row)):
            raise ValueError("transfer bandwidth segments require finite start/end/bytes")
        a, b, count = row
        if a < previous or b <= a or b > end or count <= 0:
            raise ValueError("transfer bandwidth segments must be ordered within their envelope")
        previous = b
    if normalized and (not math.isclose(normalized[0][0], start, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(normalized[-1][1], end, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(math.fsum(row[2] for row in normalized), byte_count, rel_tol=1e-12, abs_tol=1e-9)):
        raise ValueError("transfer bandwidth segments disagree with envelope or payload bytes")
    return normalized


def normalize_transfer_reserved_resource_ids(
    resource_id: str,
    link: str,
    values: Tuple[str, ...],
) -> Tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            "transfer reserved_resource_ids must be a list or tuple")
    resources = tuple(values)
    if (len(resources) not in (3, 4)
            or any(not isinstance(item, str) or not item
                   for item in resources)):
        raise ValueError(
            "P2P transfer reserved resources require two endpoints and a fabric or node TX/RX")
    if len(resources) != len(set(resources)):
        raise ValueError("transfer reserved_resource_ids must be unique")
    fabric_resource = f"communication:fabric:{resource_id}"
    if len(resources) == 4:
        if (link != "infiniband" or not resources[2].startswith("communication:infiniband:node:")
                or not resources[2].endswith(":tx")
                or not resources[3].startswith("communication:infiniband:node:")
                or not resources[3].endswith(":rx")):
            raise ValueError("node-port transfer requires directional InfiniBand TX/RX identities")
    elif resources[-1] != fabric_resource:
        raise ValueError(
            "transfer reserved resources must end with its fabric resource")
    endpoint_resources = resources[:2]
    if any(
            not resource.startswith("communication:endpoint:")
            or not resource.endswith(f":{link}")
            for resource in endpoint_resources):
        raise ValueError(
            "transfer reserved resources require canonical link endpoints")
    return resources


@dataclass(frozen=True)
class RouteDecision:
    deployment_id: str
    request_id: str
    stage: str
    policy: str
    source_engine_id: Optional[str]
    candidate_queued_tokens: Tuple[Tuple[str, int], ...]
    selected_engine_id: str
    reason: str
    queue_enter_time_s: float
    decision_time_s: float
    queue_exit_time_s: float
    dispatch_time_s: float

    def __post_init__(self) -> None:
        for name in (
                "deployment_id", "request_id", "stage", "policy",
                "selected_engine_id", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"route decision {name} must be non-empty")
        if self.stage not in _ROUTE_STAGES:
            raise ValueError(f"unknown route stage: {self.stage!r}")
        if self.policy != "least_queued_tokens":
            raise ValueError(f"unknown route policy: {self.policy!r}")
        if self.stage == "prefill_to_decode":
            if (not isinstance(self.source_engine_id, str)
                    or not self.source_engine_id):
                raise ValueError(
                    "prefill_to_decode requires a source engine ID")
        elif self.source_engine_id is not None:
            raise ValueError(
                f"{self.stage} cannot carry a source engine ID")
        candidates = tuple(self.candidate_queued_tokens)
        if not candidates:
            raise ValueError("route decision requires at least one candidate")
        engine_ids = []
        for engine_id, queued_tokens in candidates:
            if not isinstance(engine_id, str) or not engine_id:
                raise ValueError("candidate engine IDs must be non-empty")
            if (isinstance(queued_tokens, bool)
                    or not isinstance(queued_tokens, int)):
                raise TypeError("candidate queued-token loads must be integers")
            if queued_tokens < 0:
                raise ValueError(
                    "candidate queued-token loads cannot be negative")
            engine_ids.append(engine_id)
        if engine_ids != sorted(engine_ids):
            raise ValueError("route candidates must use stable engine-ID order")
        if len(engine_ids) != len(set(engine_ids)):
            raise ValueError("route candidate engine IDs must be unique")
        if self.selected_engine_id not in engine_ids:
            raise ValueError("selected engine must be one of the candidates")
        expected = min(candidates, key=lambda item: (item[1], item[0]))[0]
        if self.selected_engine_id != expected:
            raise ValueError(
                "selected engine does not satisfy least_queued_tokens: "
                f"expected={expected!r}, actual={self.selected_engine_id!r}")
        object.__setattr__(self, "candidate_queued_tokens", candidates)
        times = (
            self.queue_enter_time_s,
            self.decision_time_s,
            self.queue_exit_time_s,
            self.dispatch_time_s,
        )
        if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in times):
            raise ValueError("route decision times must be finite and non-negative")
        if not (
                self.queue_enter_time_s <= self.decision_time_s
                <= self.queue_exit_time_s <= self.dispatch_time_s):
            raise ValueError("route decision times must be monotonic")


@dataclass(frozen=True)
class TransferTraceEntry:
    transfer_id: str
    request_ids: Tuple[str, ...]
    payload: str
    source_pool_id: str
    destination_pool_id: str
    resource_id: str
    link: str
    reserved_resource_ids: Tuple[str, ...]
    byte_count: int
    start_time_s: float
    end_time_s: float
    bandwidth_segments: Tuple[Tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bandwidth_segments", normalize_transfer_bandwidth_segments(
            self.bandwidth_segments, self.start_time_s, self.end_time_s, self.byte_count))
        for name in (
                "transfer_id", "payload", "source_pool_id",
                "destination_pool_id", "resource_id", "link"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"transfer trace {name} must be non-empty")
        if self.payload not in ("kv", "activation"):
            raise ValueError(f"unknown transfer payload: {self.payload!r}")
        if not isinstance(self.request_ids, (list, tuple)):
            raise TypeError("transfer request_ids must be a list or tuple")
        request_ids = tuple(self.request_ids)
        if (not request_ids
                or any(not isinstance(request_id, str) or not request_id
                       for request_id in request_ids)):
            raise ValueError(
                "transfer request_ids must contain non-empty strings")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("transfer request_ids must be unique")
        object.__setattr__(self, "request_ids", request_ids)
        reserved_resource_ids = normalize_transfer_reserved_resource_ids(
            self.resource_id,
            self.link,
            self.reserved_resource_ids,
        )
        object.__setattr__(
            self, "reserved_resource_ids", reserved_resource_ids)
        if self.source_pool_id == self.destination_pool_id:
            raise ValueError("cross-pool transfer endpoints must be distinct")
        if isinstance(self.byte_count, bool) or not isinstance(
                self.byte_count, int):
            raise TypeError("transfer byte_count must be an integer")
        if self.byte_count <= 0:
            raise ValueError("transfer byte_count must be positive")
        if (isinstance(self.start_time_s, bool)
                or isinstance(self.end_time_s, bool)
                or not isinstance(self.start_time_s, (int, float))
                or not isinstance(self.end_time_s, (int, float))
                or not math.isfinite(self.start_time_s)
                or not math.isfinite(self.end_time_s)
                or self.start_time_s < 0
                or self.end_time_s < self.start_time_s):
            raise ValueError("invalid transfer trace interval")


@dataclass(frozen=True)
class ScheduleTraceEntry:
    backend_step: BackendStep
    start_time_s: float
    end_time_s: float
    timing_mode: str = "device_only"
    scheduler_plan_cpu_s: float = 0.0
    scheduler_update_cpu_s: float = 0.0
    execution_wall_s: float | None = None
    device_queue_wait_s: float = 0.0
    completion_queue_wait_s: float = 0.0

    def __post_init__(self) -> None:
        if (isinstance(self.start_time_s, bool)
                or isinstance(self.end_time_s, bool)
                or not isinstance(self.start_time_s, (int, float))
                or not isinstance(self.end_time_s, (int, float))
                or not math.isfinite(self.start_time_s)
                or not math.isfinite(self.end_time_s)
                or self.start_time_s < 0
                or self.end_time_s < self.start_time_s):
            raise ValueError("invalid schedule trace interval")
        validate_serving_timing_mode(self.timing_mode)
        for name in (
                "scheduler_plan_cpu_s", "scheduler_update_cpu_s",
                "device_queue_wait_s", "completion_queue_wait_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"schedule trace {name} must be finite and non-negative")
        if self.execution_wall_s is None:
            if (self.scheduler_plan_cpu_s or self.scheduler_update_cpu_s
                    or self.device_queue_wait_s
                    or self.completion_queue_wait_s):
                raise ValueError(
                    "explicit Scheduler timing requires execution wall time")
            object.__setattr__(
                self, "execution_wall_s", self.end_time_s - self.start_time_s)
        else:
            if (isinstance(self.execution_wall_s, bool)
                    or not isinstance(self.execution_wall_s, (int, float))
                    or not math.isfinite(self.execution_wall_s)
                    or self.execution_wall_s < 0):
                raise ValueError(
                    "schedule trace execution wall time must be finite and non-negative")
            expected = (
                self.scheduler_plan_cpu_s
                + self.device_queue_wait_s
                + self.execution_wall_s
                + self.completion_queue_wait_s
                + self.scheduler_update_cpu_s)
            if not math.isclose(
                    self.end_time_s - self.start_time_s,
                    expected,
                    rel_tol=1e-9,
                    abs_tol=1e-12):
                raise ValueError("schedule trace timing decomposition is inconsistent")


@dataclass(frozen=True)
class ScheduleTrace:
    entries: Tuple[ScheduleTraceEntry, ...]
    route_decisions: Tuple[RouteDecision, ...]
    transfers: Tuple[TransferTraceEntry, ...]
    observation_end_time_s: float | None
    pending_entries: Tuple[ScheduleTraceEntry, ...]

    def __init__(
        self,
        entries: Iterable[ScheduleTraceEntry],
        route_decisions: Iterable[RouteDecision] = (),
        transfers: Iterable[TransferTraceEntry] = (),
        *,
        observation_end_time_s: float | None = None,
        pending_entries: Iterable[ScheduleTraceEntry] = (),
    ):
        normalized = tuple(entries)
        if observation_end_time_s is not None and (isinstance(observation_end_time_s, bool)
                or not isinstance(observation_end_time_s, (int, float))
                or not math.isfinite(observation_end_time_s) or observation_end_time_s <= 0):
            raise ValueError("observation end must be finite and positive")
        pending = tuple(pending_entries)
        if pending and observation_end_time_s is None:
            raise ValueError("pending entries require an explicit observation end")
        if any(not isinstance(entry, ScheduleTraceEntry) for entry in pending):
            raise TypeError("pending entries must be ScheduleTraceEntry objects")
        if observation_end_time_s is not None:
            if any(entry.end_time_s > observation_end_time_s for entry in normalized):
                raise ValueError("completed observation entries exceed its cutoff")
            if any(not entry.start_time_s < observation_end_time_s < entry.end_time_s for entry in pending):
                raise ValueError("pending entries must straddle the observation cutoff")
        if not normalized and observation_end_time_s is None:
            raise ValueError("ScheduleTrace requires at least one entry")
        if any(not isinstance(entry, ScheduleTraceEntry)
               for entry in normalized):
            raise TypeError("ScheduleTrace requires ScheduleTraceEntry objects")
        normalized_decisions = tuple(route_decisions)
        if any(not isinstance(decision, RouteDecision)
               for decision in normalized_decisions):
            raise TypeError("ScheduleTrace route_decisions must be RouteDecision objects")
        normalized_transfers = tuple(transfers)
        if any(not isinstance(transfer, TransferTraceEntry)
               for transfer in normalized_transfers):
            raise TypeError(
                "ScheduleTrace transfers must be TransferTraceEntry objects")
        object.__setattr__(self, "entries", normalized)
        object.__setattr__(self, "route_decisions", normalized_decisions)
        object.__setattr__(self, "transfers", normalized_transfers)
        object.__setattr__(self, "observation_end_time_s", observation_end_time_s)
        object.__setattr__(self, "pending_entries", pending)

    def __iter__(self) -> Iterator[ScheduleTraceEntry]:
        return iter(self.entries)

    @property
    def end_time_s(self) -> float:
        if self.observation_end_time_s is not None:
            return self.observation_end_time_s
        return max(entry.end_time_s for entry in self.entries)
