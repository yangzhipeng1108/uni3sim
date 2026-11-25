"""External memory capability and event protocol, with fail-closed validation.

This module does not price kernels. A session can exchange memory events with an
external adapter; an integrated compute/memory coordinator remains unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Protocol

from src.kernel.memory_access import (
    AddressLevel, EventKind, EventRef, Granularity, MemoryAccess, MemoryAccessProgram,
    Operation, Region, Semantics, Span, Strided, Typed, check_type, requested_bytes,
    unique_by,
)


class Resource(str, Enum):
    LOGICAL_STORAGE = "logical_storage"
    L1 = "l1"
    L2 = "l2"
    DRAM = "dram"
    COALESCING = "coalescing"
    SOURCE_QUEUE = "source_queue"


class InitialState(str, Enum):
    COLD = "cold"
    UNKNOWN = "unknown"
    CHECKPOINT = "checkpoint"


@dataclass(frozen=True)
class ResourceOwner(Typed):
    resource: Resource
    owner: str


@dataclass(frozen=True)
class BackendCapabilities(Typed):
    backend_id: str
    hardware_ids: tuple[str, ...]
    granularities: tuple[Granularity, ...]
    semantics: tuple[Semantics, ...]
    address_levels: tuple[AddressLevel, ...]
    region_kinds: tuple[str, ...]
    entries: tuple[str | None, ...]
    policies: tuple[str | None, ...]
    request_shape_refs: tuple[str | None, ...]
    events: tuple[EventKind, ...]
    resources: tuple[Resource, ...]
    stateful: bool
    synthetic: bool

    def __post_init__(self):
        super().__post_init__()
        from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS
        if any(hw not in CANONICAL_HARDWARE_IDS for hw in self.hardware_ids):
            raise ValueError("unknown hardware capability")
        if any(kind not in ("span", "strided") for kind in self.region_kinds):
            raise ValueError("unsupported region capability")
        for values in (self.hardware_ids, self.granularities, self.semantics, self.address_levels,
                       self.region_kinds, self.entries, self.policies, self.request_shape_refs,
                       self.events, self.resources):
            if len(set(values)) != len(values):
                raise ValueError("duplicate capability")
        if self.backend_id == "midlayer":
            raise ValueError("external backend identity cannot shadow Midlayer")


@dataclass(frozen=True)
class ClockContract(Typed):
    tick_seconds_numerator: int
    tick_seconds_denominator: int
    rounding: str
    arbitration: str

    def __post_init__(self):
        super().__post_init__()
        if min(self.tick_seconds_numerator, self.tick_seconds_denominator) == 0:
            raise ValueError("tick scale must be positive")
        if self.rounding != "ceil" or self.arbitration != "stable_submission_order":
            raise NotImplementedError("unsupported clock or same-tick arbitration")


@dataclass(frozen=True)
class SessionConfig(Typed):
    session_id: str
    backend_id: str
    backend_config_id: str
    hardware_id: str
    device_id: str
    memory_domain: str
    clock: ClockContract
    initial_state: InitialState
    checkpoint_ref: str | None
    ownership: tuple[ResourceOwner, ...]

    def __post_init__(self):
        super().__post_init__()
        if (self.initial_state is InitialState.CHECKPOINT) != (self.checkpoint_ref is not None):
            raise ValueError("checkpoint state requires an explicit reference")
        resources = unique_by(self.ownership, "resource")
        if not resources:
            raise ValueError("resource ownership must be explicit")
        if any(owner.owner not in ("midlayer", self.backend_id) for owner in self.ownership):
            raise ValueError("unknown resource owner")


@dataclass(frozen=True)
class RequestIdentity(Typed):
    session_id: str
    backend_config_id: str
    program_id: str
    binding_id: str
    call_id: str
    access_id: str
    occurrence: int


@dataclass(frozen=True)
class MemoryRequest(Typed):
    identity: RequestIdentity
    access: MemoryAccess


@dataclass(frozen=True)
class Submission(Typed):
    identity: RequestIdentity
    kind: EventKind
    tick: int
    next_wakeup: int | None

    def __post_init__(self):
        super().__post_init__()
        if self.kind is EventKind.ACCEPTED:
            if self.next_wakeup is not None:
                raise ValueError("accepted request cannot report blocked wakeup")
        elif self.kind is EventKind.BLOCKED:
            if self.next_wakeup is None or self.next_wakeup <= self.tick:
                raise ValueError("blocked submission requires a future wakeup")
        else:
            raise ValueError("invalid submission status")


@dataclass(frozen=True)
class MemoryEvent(Typed):
    identity: RequestIdentity
    kind: EventKind
    tick: int
    allocation_id: str
    region: Region
    visibility_domain: str


class ClosePolicy(str, Enum):
    REQUIRE_DRAINED = "require_drained"
    PRESERVE = "preserve"


@dataclass(frozen=True)
class CloseResult(Typed):
    session_id: str
    pending: tuple[RequestIdentity, ...]
    state_ref: str | None


class BackendSession(Protocol):
    def bind_program(self, program: MemoryAccessProgram) -> None: ...
    def try_submit(self, request: MemoryRequest, submit_tick: int) -> Submission: ...
    def advance(self, until_tick: int) -> tuple[MemoryEvent, ...]: ...
    def next_event_tick(self) -> int | None: ...
    def close(self, policy: ClosePolicy) -> CloseResult: ...


class MemoryBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def open_session(self, config: SessionConfig) -> BackendSession: ...


def negotiate(program, capabilities, config):
    if not isinstance(program, MemoryAccessProgram) or not isinstance(capabilities, BackendCapabilities):
        raise TypeError("typed program and backend capabilities required")
    if not isinstance(config, SessionConfig):
        raise TypeError("typed session config required")
    program.__post_init__()
    capabilities.__post_init__()
    config.__post_init__()
    if capabilities.backend_id != config.backend_id or program.hardware_id != config.hardware_id:
        raise ValueError("backend/hardware identity mismatch")
    if not program.coverage.executable:
        raise ValueError("summary representations cannot be submitted")
    if program.tile_programs:
        raise NotImplementedError('backend requires explicitly bounded tile expansion; compressed logical loops are not an empty executable program')
    if (program.hardware_id not in capabilities.hardware_ids
            or program.coverage.granularity not in capabilities.granularities
            or program.coverage.semantics not in capabilities.semantics):
        raise NotImplementedError("backend rejects program coverage")
    external = {owner.resource for owner in config.ownership if owner.owner == config.backend_id}
    if not external or not external <= set(capabilities.resources):
        raise NotImplementedError("backend cannot own requested resources")
    # Logical programs cannot transfer ownership of physical cache/DRAM timing.
    if program.coverage.semantics is Semantics.LOGICAL and external != {Resource.LOGICAL_STORAGE}:
        raise NotImplementedError("logical accesses do not define physical resource takeover")
    if program.coverage.semantics is Semantics.ISSUED and Resource.DRAM in external and Resource.L2 not in external:
        raise NotImplementedError("DRAM-only takeover requires transaction requests")
    for allocation in program.allocations:
        if (allocation.device_id != config.device_id or allocation.memory_domain != config.memory_domain
                or allocation.address_level not in capabilities.address_levels):
            raise NotImplementedError("backend rejects allocation domain/address level")
        if program.coverage.semantics is Semantics.TRANSACTION and allocation.address_level is not AddressLevel.PHYSICAL:
            raise NotImplementedError("transaction execution requires physical mapping")
    required = {EventKind.ACCEPTED, EventKind.BLOCKED, EventKind.RETIRED, EventKind.WAKEUP}
    required.update(a.completion for a in program.accesses)
    required.update(d.kind for a in program.accesses for d in a.dependencies
                    if d.kind in (EventKind.READ_READY, EventKind.WRITE_VISIBLE))
    if not required <= set(capabilities.events):
        raise NotImplementedError("backend lacks required event semantics")
    for access in program.accesses:
        kind = "span" if isinstance(access.region, Span) else "strided" if isinstance(access.region, Strided) else None
        if (kind not in capabilities.region_kinds or access.entry not in capabilities.entries
                or access.policy not in capabilities.policies
                or access.request_shape_ref not in capabilities.request_shape_refs):
            raise NotImplementedError("backend rejects region/entry/policy/request geometry")


def adapter_boundary(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        self._adapter_touched = False
        try:
            return method(self, *args, **kwargs)
        except Exception:
            if self._adapter_touched:
                self._failed = True
            raise
    return guarded


class ValidatedMemorySession:
    """Adapter boundary guard. Ticks are absolute in one negotiated time domain.

    All same-tick submissions use the explicit stable caller order. Compressed
    reads expand to distinct request occurrences; dependency events release only
    after all occurrences have reached that event. No duration result is exposed.
    """
    def __init__(self, backend: MemoryBackend, config: SessionConfig):
        if backend is None:
            raise ValueError("external mode requires a backend")
        self.capabilities = backend.capabilities()
        if not isinstance(self.capabilities, BackendCapabilities):
            raise TypeError("typed capabilities required")
        if not isinstance(config, SessionConfig):
            raise TypeError("typed session config required")
        config.__post_init__()
        if self.capabilities.backend_id != config.backend_id:
            raise ValueError("backend identity mismatch")
        self.capabilities.__post_init__()
        external = {o.resource for o in config.ownership if o.owner == config.backend_id}
        if config.hardware_id not in self.capabilities.hardware_ids or not external or not external <= set(self.capabilities.resources):
            raise NotImplementedError("session capability/ownership mismatch")
        self.config = config
        self._raw = backend.open_session(config)
        self._programs = {}
        self._allocations = {}
        self._lifecycle = {}
        self._shared_signals = {}
        self._requests = {}
        self._blocked = {}
        self._signals = {}
        self._event_ticks = {}
        self._wakeups = set()
        self._closed = False
        self._failed = False
        self._adapter_touched = False
        self.now = 0
        self.accepted_requested_bytes = 0

    def _open(self):
        if self._closed or self._failed:
            raise RuntimeError("session is closed or failed")

    @adapter_boundary
    def bind_program(self, program: MemoryAccessProgram):
        self._open()
        negotiate(program, self.capabilities, self.config)
        if program.program_id in self._programs:
            raise ValueError("program identity already bound")
        if self._programs and not self.capabilities.stateful:
            raise NotImplementedError("backend does not support multiple programs per session")
        # Reuse requires exact caller-declared storage/lifetime identity, never
        # shape equivalence. Lifecycle declarations persist at session scope.
        for allocation in program.allocations:
            if allocation.allocation_id in self._allocations:
                if allocation != self._allocations[allocation.allocation_id]:
                    raise ValueError("shared allocation binding/lifetime changed")
        for event in program.events:
            if event.allocation_id is not None and event.event_id in self._lifecycle:
                if event != self._lifecycle[event.event_id]:
                    raise ValueError("shared lifecycle event definition changed")
        self._adapter_touched = True
        self._raw.bind_program(program)
        self._programs[program.program_id] = program
        for allocation in program.allocations:
            self._allocations[allocation.allocation_id] = allocation
        for event in program.events:
            if event.allocation_id is not None:
                self._lifecycle[event.event_id] = event

    def _program(self, program_id):
        if program_id not in self._programs:
            raise ValueError("unbound program")
        return self._programs[program_id]

    def _identity(self, program, access, occurrence):
        check_type(occurrence, int)
        if occurrence >= access.repeat:
            raise ValueError("repeat occurrence out of range")
        scope = unique_by(program.scopes, "scope_id")[access.scope_id]
        return RequestIdentity(self.config.session_id, self.config.backend_config_id,
            program.program_id, program.binding_id, scope.call_id, access.access_id, occurrence)

    def _dependency_tick(self, program, dep):
        shared_key = (dep.node_id, dep.kind)
        if dep.node_id in self._lifecycle and shared_key in self._shared_signals:
            return self._shared_signals[shared_key]
        key = (program.program_id, dep.node_id, dep.kind)
        if key in self._signals:
            return self._signals[key]
        accesses = unique_by(program.accesses, "access_id")
        if dep.node_id not in accesses:
            raise ValueError("dependency has not been signaled")
        access = accesses[dep.node_id]
        ticks = []
        for occurrence in range(access.repeat):
            identity = self._identity(program, access, occurrence)
            event_key = (identity, dep.kind)
            if event_key not in self._event_ticks:
                raise ValueError("memory dependency has not completed")
            ticks.append(self._event_ticks[event_key])
        return max(ticks)

    def signal(self, program_id: str, event_id: str, tick: int):
        """Publish a declared caller lifecycle/compute event, never a memory result."""
        self._open()
        check_type(tick, int)
        if tick != self.now:
            raise ValueError("signals must occur at the current session tick")
        program = self._program(program_id)
        events = unique_by(program.events, "event_id")
        if event_id not in events:
            raise ValueError("only declared caller events can be signaled")
        event = events[event_id]
        key = (program_id, event_id, event.kind)
        if key in self._signals or (event.allocation_id is not None and (event_id, event.kind) in self._shared_signals):
            raise ValueError("duplicate caller event")
        if event.kind is EventKind.FREED:
            if any(request.access.allocation_id == event.allocation_id
                   and (identity, EventKind.RETIRED) not in self._event_ticks
                   for identity, request in self._requests.items()):
                raise ValueError("free must await all session requests")
            if any(unique_by(self._program(identity.program_id).accesses, "access_id")[identity.access_id].allocation_id == event.allocation_id for identity in self._blocked):
                raise ValueError("free must await blocked session work")
        if any(self._dependency_tick(program, dep) > tick for dep in event.dependencies):
            raise ValueError("signal precedes dependency")
        self._signals[key] = tick
        if event.allocation_id is not None:
            self._shared_signals[(event_id, event.kind)] = tick

    @adapter_boundary
    def try_submit(self, program_id: str, access_id: str, occurrence: int, submit_tick: int):
        self._open()
        check_type(submit_tick, int)
        if submit_tick != self.now:
            raise ValueError("submit must use current tick; advance explicitly")
        program = self._program(program_id)
        accesses = unique_by(program.accesses, "access_id")
        if access_id not in accesses:
            raise ValueError("unknown access")
        access = accesses[access_id]
        allocation = self._allocations[access.allocation_id]
        if allocation.freed_event is not None and (allocation.freed_event, EventKind.FREED) in self._shared_signals:
            raise ValueError("access references freed session storage")
        identity = self._identity(program, access, occurrence)
        if identity in self._requests:
            raise ValueError("request already accepted; cannot resubmit")
        if identity in self._blocked:
            wakeup_tick = self._blocked[identity]
            if submit_tick < wakeup_tick or (identity, wakeup_tick) not in self._wakeups:
                raise ValueError("retry before a validated wakeup")
        if any(self._dependency_tick(program, dep) > submit_tick for dep in access.dependencies):
            raise ValueError("submission precedes dependency")
        self._adapter_touched = True
        response = self._raw.try_submit(MemoryRequest(identity, access), submit_tick)
        if not isinstance(response, Submission):
            raise TypeError("typed submission response required")
        if response.identity != identity or response.tick != submit_tick:
            raise ValueError("submission identity/time mismatch")
        if response.kind is EventKind.ACCEPTED:
            self._requests[identity] = MemoryRequest(identity, access)
            self._event_ticks[(identity, EventKind.ACCEPTED)] = submit_tick
            self.accepted_requested_bytes += requested_bytes(access.region)
            if identity in self._blocked:
                del self._blocked[identity]
        elif response.kind is EventKind.BLOCKED:
            self._blocked[identity] = response.next_wakeup
        else:
            raise ValueError("invalid submission event")
        return response

    @adapter_boundary
    def next_event_tick(self):
        self._open()
        self._adapter_touched = True
        tick = self._raw.next_event_tick()
        if tick is not None:
            check_type(tick, int)
            if tick < self.now:
                raise ValueError("backend next event is in the past")
        return tick

    @adapter_boundary
    def advance(self, until_tick: int):
        self._open()
        check_type(until_tick, int)
        if until_tick < self.now:
            raise ValueError("clock cannot move backwards")
        self._adapter_touched = True
        events = self._raw.advance(until_tick)
        check_type(events, tuple[MemoryEvent, ...])
        previous = self.now
        for event in events:
            if not previous <= event.tick <= until_tick:
                raise ValueError("backend event order/time mismatch")
            previous = event.tick
            identity = event.identity
            if event.kind is EventKind.WAKEUP:
                if identity not in self._blocked or event.tick != self._blocked[identity]:
                    raise ValueError("wakeup identity/time mismatch")
                program = self._program(identity.program_id)
                access = unique_by(program.accesses, "access_id")[identity.access_id]
                if identity != self._identity(program, access, identity.occurrence):
                    raise ValueError("wakeup binding identity mismatch")
            else:
                if identity not in self._requests:
                    raise ValueError("event for an unaccepted or foreign request")
                access = self._requests[identity].access
                accepted_tick = self._event_ticks[(identity, EventKind.ACCEPTED)]
                if event.tick < accepted_tick:
                    raise ValueError("event precedes acceptance")
                if event.kind not in (access.completion, EventKind.RETIRED):
                    raise ValueError("event completion kind mismatch")
                if event.kind is EventKind.RETIRED:
                    ready_key = (identity, access.completion)
                    if ready_key not in self._event_ticks or event.tick < self._event_ticks[ready_key]:
                        raise ValueError("retirement precedes completion")
            if (event.allocation_id != access.allocation_id or event.region != access.region
                    or event.visibility_domain != access.visibility_domain):
                raise ValueError("event allocation/region/visibility mismatch")
            if event.kind is EventKind.WAKEUP:
                wakeup_key = (identity, event.tick)
                if wakeup_key in self._wakeups:
                    raise ValueError("duplicate backend wakeup")
                self._wakeups.add(wakeup_key)
            else:
                key = (identity, event.kind)
                if key in self._event_ticks:
                    raise ValueError("duplicate backend event")
                self._event_ticks[key] = event.tick
        self.now = until_tick
        return events

    @adapter_boundary
    def close(self, policy: ClosePolicy):
        self._open()
        check_type(policy, ClosePolicy)
        pending = tuple(identity for identity in self._requests
                        if (identity, EventKind.RETIRED) not in self._event_ticks)
        if policy is ClosePolicy.REQUIRE_DRAINED and (pending or self._blocked):
            raise ValueError("session is not drained")
        if self._blocked:
            raise ValueError("blocked work must be resolved before closing")
        if policy is ClosePolicy.PRESERVE and not self.capabilities.stateful:
            raise NotImplementedError("backend cannot preserve state")
        self._adapter_touched = True
        result = self._raw.close(policy)
        if not isinstance(result, CloseResult) or result.session_id != self.config.session_id or set(result.pending) != set(pending) or len(result.pending) != len(set(result.pending)):
            raise ValueError("close result identity/pending mismatch")
        if policy is ClosePolicy.PRESERVE and result.state_ref is None:
            raise ValueError("preserve requires state reference")
        self._closed = True
        return result


@dataclass(frozen=True)
class MemoryExecutionContext(Typed):
    session: ValidatedMemorySession
    program_id: str
    binding_id: str
    call_id: str

    def validate(self, kernel, hardware, method):
        if not isinstance(self.session, ValidatedMemorySession):
            raise TypeError("validated memory session required")
        self.session._open()
        program = self.session._program(self.program_id)
        if (program.binding_id != self.binding_id or any(s.call_id != self.call_id for s in program.scopes)
                or program.hardware_id != hardware.hardware_id or kernel.hardware_id != program.hardware_id):
            raise ValueError("memory context call/program/binding/hardware mismatch")
        if method != "midlayer":
            raise NotImplementedError("external memory context is only defined for Forward")
        raise NotImplementedError(
            "external prediction requires a compute/memory coordinator; "
            "this release supports untimed export and validated event sessions only")
