"""Strict Host launch timing contract for Serving simulation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Tuple


SERVING_TIMING_MODES = (
    "device_only",
    "device_plus_launch",
    "device_plus_launch_plus_scheduler",
)
LAUNCH_KINDS = ("eager", "graph_replay", "runtime_prepare")
STEP_PHASES = ("prefill", "mixed", "decode")
SCHEDULER_HOST_OPERATIONS = ("plan", "update")
WORKER_REQUEST_STATE_UNITS = MappingProxyType({"vllm030_v2_state_" + name: unit for name, unit in (
    ("fixed", "per_worker_step"), ("cached_request", "per_cached_request"),
    ("finish_nonempty", "per_step_with_finished_requests"), ("finished_request", "per_finished_request"),
    ("admit_nonempty", "per_step_with_admitted_requests"), ("admitted_request", "per_admitted_request"),
    ("admitted_token", "per_newly_uploaded_token_id"))})


@dataclass(frozen=True)
class WorkerRequestState:
    """Non-duration source counts for one ordinary worker state update."""

    source_version: str
    max_requests: int
    max_model_len: int
    cached_requests: int
    finished_requests: int
    admitted_requests: int
    admitted_token_ids: int
    kv_block_size_tokens: int

    def __post_init__(self) -> None:
        if self.source_version != "vllm_0_30_0":
            raise ValueError("worker request state requires explicit vLLM0.30 source")
        for name in ("max_requests", "max_model_len", "cached_requests", "finished_requests",
                "admitted_requests", "admitted_token_ids", "kv_block_size_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"worker request state {name} must be a nonnegative integer")
        if self.max_requests == 0 or self.max_model_len == 0 or self.kv_block_size_tokens == 0:
            raise ValueError("worker request state requires positive configured capacities")
        if self.cached_requests + self.admitted_requests > self.max_requests:
            raise ValueError("worker request state exceeds configured request slots")
        if self.finished_requests > self.max_requests:
            raise ValueError("worker finish count exceeds configured request slots")
        if ((self.admitted_requests == 0) != (self.admitted_token_ids == 0)
                or self.admitted_token_ids < self.admitted_requests
                or self.admitted_token_ids > self.admitted_requests * self.max_model_len):
            raise ValueError("worker admission token count disagrees with fresh requests")


@dataclass(frozen=True)
class SchedulerHostProfile:
    """Deterministic Scheduler Host service selected from static step state."""

    hardware_id: str
    vllm_version: str
    source_identity: str
    max_scheduled_request_count: int
    supported_phases: Tuple[str, ...]
    plan_intercept_s: float
    plan_per_request_s: float
    update_intercept_s: float
    update_per_request_s: float
    profile_path: str
    profile_sha256: str
    prefill_plan_intercept_s: float | None = None
    prefill_plan_per_request_s: float | None = None
    prefill_update_intercept_s: float | None = None
    prefill_update_per_request_s: float | None = None
    fallback_dispatch_min_scheduled_request_count: int | None = None
    fallback_dispatch_max_scheduled_request_count: int | None = None
    fallback_dispatch_supported_phases: Tuple[str, ...] | None = None
    fallback_dispatch_plan_s: float | None = None
    fallback_dispatch_update_s: float | None = None
    fallback_dispatch_occurrence_contract: str | None = None
    shared_cpu_timing_target: str | None = None

    def __post_init__(self) -> None:
        if self.shared_cpu_timing_target not in (None, "0.30.0"):
            raise ValueError("unsupported shared CPU timing target")
        for name in ("hardware_id", "vllm_version", "source_identity",
                     "profile_path", "profile_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Scheduler Host profile {name} must be non-empty")
        if (len(self.profile_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.profile_sha256)):
            raise ValueError("Scheduler Host profile SHA256 is invalid")
        if (isinstance(self.max_scheduled_request_count, bool)
                or not isinstance(self.max_scheduled_request_count, int)
                or self.max_scheduled_request_count < 1):
            raise ValueError("Scheduler Host maximum request count must be positive")
        if not isinstance(self.supported_phases, tuple) or not self.supported_phases:
            raise ValueError("Scheduler Host supported phases must be non-empty")
        if (any(not isinstance(phase, str) or phase not in STEP_PHASES
                for phase in self.supported_phases)
                or len(set(self.supported_phases)) != len(self.supported_phases)):
            raise ValueError("Scheduler Host supported phases are invalid")
        for name in ("plan_intercept_s", "plan_per_request_s",
                     "update_intercept_s", "update_per_request_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"Scheduler Host {name} must be finite positive")
        # Pure-prefill steps admit new requests, which costs several times one
        # decode step's planning; they therefore own measured parameters rather
        # than borrowing the mixed/decode line.  Coverage and parameters must
        # agree in both directions so an uncovered phase can never be charged.
        prefill_names = (
            "prefill_plan_intercept_s", "prefill_plan_per_request_s",
            "prefill_update_intercept_s", "prefill_update_per_request_s")
        supports_prefill = "prefill" in self.supported_phases
        for name in prefill_names:
            value = getattr(self, name)
            if not supports_prefill:
                if value is not None:
                    raise ValueError(
                        f"Scheduler Host {name} is set without prefill coverage")
                continue
            if (value is None or isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"Scheduler Host {name} must be finite positive")
        fallback_values = (
            self.fallback_dispatch_min_scheduled_request_count,
            self.fallback_dispatch_max_scheduled_request_count,
            self.fallback_dispatch_supported_phases,
            self.fallback_dispatch_plan_s,
            self.fallback_dispatch_update_s,
            self.fallback_dispatch_occurrence_contract,
        )
        if any(value is not None for value in fallback_values):
            minimum, maximum, phases, plan_s, update_s, contract = fallback_values
            if (isinstance(minimum, bool) or not isinstance(minimum, int)
                    or minimum <= self.max_scheduled_request_count):
                raise ValueError("fallback dispatch minimum request count is invalid")
            if (isinstance(maximum, bool) or not isinstance(maximum, int)
                    or maximum < minimum):
                raise ValueError("fallback dispatch maximum request count is invalid")
            if (not isinstance(phases, tuple) or not phases
                    or any(phase not in STEP_PHASES for phase in phases)
                    or len(phases) != len(set(phases))):
                raise ValueError("fallback dispatch phases are invalid")
            for name, value in (("fallback_dispatch_plan_s", plan_s),
                                ("fallback_dispatch_update_s", update_s)):
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Scheduler Host {name} must be finite positive")
            if not isinstance(contract, str) or not contract:
                raise ValueError("fallback dispatch occurrence contract is invalid")

    def supports(self, schedule: object, operation: str) -> bool:
        """Allow count extrapolation for characterized scheduling phases."""

        if operation not in SCHEDULER_HOST_OPERATIONS:
            raise ValueError(f"unknown Scheduler Host operation: {operation!r}")
        requests = getattr(schedule, "requests")
        if not isinstance(requests, tuple) or not requests:
            raise ValueError("Scheduler Host profile requires scheduled requests")
        phase = schedule_phase(schedule)
        count = len(requests)
        if phase in self.supported_phases:
            return True
        return (
            self.fallback_dispatch_supported_phases is not None
            and phase in self.fallback_dispatch_supported_phases
            and self.fallback_dispatch_min_scheduled_request_count is not None
            and self.fallback_dispatch_max_scheduled_request_count is not None
            and self.fallback_dispatch_min_scheduled_request_count <= count
            <= self.fallback_dispatch_max_scheduled_request_count
        )

    def service_time_s(self, schedule: object, operation: str) -> float:
        """Use measured dispatch ranges or the phase's affine cost model.

        The maximum observed count records characterization coverage, not a
        validity limit for the affine model. Unsupported phases still fail.
        """

        if not self.supports(schedule, operation):
            count = len(getattr(schedule, "requests"))
            phase = schedule_phase(schedule)
            raise ValueError(
                "Scheduler Host profile has no coverage for "
                f"phase={phase!r} scheduled_request_count={count}")
        count = len(schedule.requests)
        phase = schedule_phase(schedule)
        if (self.fallback_dispatch_supported_phases is not None
                and phase in self.fallback_dispatch_supported_phases
                and self.fallback_dispatch_min_scheduled_request_count <= count
                <= self.fallback_dispatch_max_scheduled_request_count):
            return (self.fallback_dispatch_plan_s
                    if operation == "plan" else self.fallback_dispatch_update_s)
        if phase == "prefill":
            if operation == "plan":
                return (self.prefill_plan_intercept_s
                        + self.prefill_plan_per_request_s * count)
            return (self.prefill_update_intercept_s
                    + self.prefill_update_per_request_s * count)
        if operation == "plan":
            return self.plan_intercept_s + self.plan_per_request_s * count
        return self.update_intercept_s + self.update_per_request_s * count

    def manifest(self) -> dict[str, object]:
        service_time_s = {
            "plan": {
                "intercept": self.plan_intercept_s,
                "per_request": self.plan_per_request_s,
            },
            "update": {
                "intercept": self.update_intercept_s,
                "per_request": self.update_per_request_s,
            },
        }
        if "prefill" in self.supported_phases:
            service_time_s["prefill"] = {
                "plan": {
                    "intercept": self.prefill_plan_intercept_s,
                    "per_request": self.prefill_plan_per_request_s,
                },
                "update": {
                    "intercept": self.prefill_update_intercept_s,
                    "per_request": self.prefill_update_per_request_s,
                },
            }
        if self.fallback_dispatch_supported_phases is not None:
            service_time_s["fallback_dispatch"] = {
                "min_scheduled_request_count": (
                    self.fallback_dispatch_min_scheduled_request_count),
                "max_scheduled_request_count": (
                    self.fallback_dispatch_max_scheduled_request_count),
                "supported_phases": list(self.fallback_dispatch_supported_phases),
                "plan": self.fallback_dispatch_plan_s,
                "update": self.fallback_dispatch_update_s,
                "occurrence_contract": self.fallback_dispatch_occurrence_contract,
            }
        return {
            "hardware_id": self.hardware_id,
            "vllm_version": self.vllm_version,
            "source_identity": self.source_identity,
            **({"shared_cpu_timing_target": self.shared_cpu_timing_target}
               if self.shared_cpu_timing_target is not None else {}),
            "model": {
                "kind": "scheduled_request_count_affine",
                "state_dimensions": ["scheduled_request_count"],
            },
            "max_scheduled_request_count": self.max_scheduled_request_count,
            "supported_phases": list(self.supported_phases),
            "service_time_s": service_time_s,
            "profile_path": self.profile_path,
            "profile_sha256": self.profile_sha256,
        }


ENGINE_CORE_HOST_ROUND_LANES = ("graph", "eager")
ENGINE_CORE_HOST_ROUND_LANE_DISPATCH = (
    "decode_steps_to_graph_replay_lane"
    "_prefill_and_mixed_steps_to_eager_dispatch_lane"
)


@dataclass(frozen=True)
class EngineCoreHostRoundProfile:
    """Deterministic EngineCore host-round service selected by step lane.

    The profile prices only the EngineCore busy-loop segments no other Host
    profile owns: the submit-lane prologue (input-prepare event sync, pinned
    H2D commit, CANN pre-replay synchronize wall cost, sampler submit,
    worker-thread submit, queue poll and decision) and the output wake chain
    (worker wake, D2H event sync, tolist, output put).  Pure decode steps
    dispatch to the CANN graph-replay lane; prefill and mixed steps dispatch
    to the eager lane.  Every value comes from the independent EngineCore
    host-round mechanism microbenchmark; no Serving trace duration and no
    device stub duration enters this profile.
    """

    hardware_id: str
    vllm_version: str
    source_identity: str
    max_scheduled_request_count: int
    graph_submit_intercept_s: float
    graph_wake_intercept_s: float
    graph_wake_per_request_s: float
    eager_submit_intercept_s: float
    eager_wake_intercept_s: float
    eager_wake_per_request_s: float
    profile_path: str
    profile_sha256: str

    def __post_init__(self) -> None:
        for name in ("hardware_id", "vllm_version", "source_identity",
                     "profile_path", "profile_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"EngineCore host-round profile {name} must be non-empty")
        if (len(self.profile_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.profile_sha256)):
            raise ValueError(
                "EngineCore host-round profile SHA256 is invalid")
        if (isinstance(self.max_scheduled_request_count, bool)
                or not isinstance(self.max_scheduled_request_count, int)
                or self.max_scheduled_request_count < 1):
            raise ValueError(
                "EngineCore host-round maximum request count must be positive")
        for name in ("graph_submit_intercept_s", "graph_wake_intercept_s",
                     "eager_submit_intercept_s", "eager_wake_intercept_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"EngineCore host-round {name} must be finite positive")
        for name in ("graph_wake_per_request_s",
                     "eager_wake_per_request_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"EngineCore host-round {name} must be finite non-negative")

    def service_times_s(self, schedule: object) -> Tuple[float, float]:
        """Return (submit, wake) service; uncovered states fail closed."""

        requests = getattr(schedule, "requests")
        if not isinstance(requests, tuple) or not requests:
            raise ValueError(
                "EngineCore host-round profile requires scheduled requests")
        count = len(requests)
        phase = schedule_phase(schedule)
        if not 1 <= count <= self.max_scheduled_request_count:
            raise ValueError(
                "EngineCore host-round profile has no coverage for "
                f"phase={phase!r} scheduled_request_count={count}")
        if phase == "decode":
            return (
                self.graph_submit_intercept_s,
                self.graph_wake_intercept_s
                + self.graph_wake_per_request_s * count,
            )
        return (
            self.eager_submit_intercept_s,
            self.eager_wake_intercept_s
            + self.eager_wake_per_request_s * count,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "hardware_id": self.hardware_id,
            "vllm_version": self.vllm_version,
            "source_identity": self.source_identity,
            "model": {
                "kind": "lane_dispatch_scheduled_request_count_affine",
                "state_dimensions": ["scheduled_request_count"],
                "lane_dispatch": ENGINE_CORE_HOST_ROUND_LANE_DISPATCH,
                "lanes": list(ENGINE_CORE_HOST_ROUND_LANES),
            },
            "max_scheduled_request_count": self.max_scheduled_request_count,
            "service_time_s": {
                "graph_lane": {
                    "submit": {
                        "intercept": self.graph_submit_intercept_s,
                    },
                    "wake": {
                        "intercept": self.graph_wake_intercept_s,
                        "per_request": self.graph_wake_per_request_s,
                    },
                },
                "eager_lane": {
                    "submit": {
                        "intercept": self.eager_submit_intercept_s,
                    },
                    "wake": {
                        "intercept": self.eager_wake_intercept_s,
                        "per_request": self.eager_wake_per_request_s,
                    },
                },
            },
            "profile_path": self.profile_path,
            "profile_sha256": self.profile_sha256,
        }


@dataclass(frozen=True)
class RuntimePrepareEvent:
    """One source-authored Host mechanism with explicit multiplicity.

    ``graph_capturable`` distinguishes helpers executed inside an ACL graph
    from source-authored boundary work that remains outside the graph even
    when the associated model call is captured.  The latter must survive the
    runtime-Graph filtering step and is still charged before/after replay.
    ``replaces_eager_enqueue`` marks the unique completion of a group whose
    measured Host services already include the actual native submissions.
    That occurrence retains its eager-submit fact, but adds no generic
    enqueue service; captured occurrences still use ordinary graph replay.
    ``additional_eager_enqueue_count`` retains source Host submissions whose
    zero mathematical work has no device call in the lowered plan. They use
    the existing launch profile, disappear under capture, and create no
    device work or launch-gap events.
    """

    mechanism: str
    count: int = 1
    graph_capturable: bool = True
    replaces_eager_enqueue: bool = False
    additional_eager_enqueue_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.mechanism, str) or not self.mechanism:
            raise ValueError("runtime prepare mechanism must be non-empty")
        if (isinstance(self.count, bool) or not isinstance(self.count, int)
                or self.count < 1):
            raise ValueError("runtime prepare event count must be positive")
        if not isinstance(self.graph_capturable, bool):
            raise TypeError("runtime prepare graph_capturable must be boolean")
        if type(self.replaces_eager_enqueue) is not bool:
            raise TypeError("runtime prepare replaces_eager_enqueue must be boolean")
        if self.replaces_eager_enqueue and not self.graph_capturable:
            raise ValueError("a complete eager submit must be graph capturable")
        if (type(self.additional_eager_enqueue_count) is not int
                or self.additional_eager_enqueue_count < 0):
            raise ValueError("additional eager enqueue count must be a non-negative integer")
        if self.additional_eager_enqueue_count and (
                not self.graph_capturable or self.replaces_eager_enqueue):
            raise ValueError("additional enqueues require a capturable incremental prepare event")


@dataclass(frozen=True)
class HostRuntimePrepareProfile:
    """Diagnostic candidate costs for source-authored prepare mechanisms.

    This is deliberately separate from ``HostLaunchProfile``: a candidate
    prepare cost is not production-eligible until its independent mechanism
    measurement and source occurrence metadata are both present.
    """

    hardware_id: str
    service_time_s_by_mechanism: Tuple[Tuple[str, float], ...]
    lane_service_time_s_by_mechanism: Tuple[Tuple[str, int, str, float], ...] = ()
    profile_path: str | None = None
    profile_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hardware_id, str) or not self.hardware_id:
            raise ValueError("runtime prepare hardware_id must be non-empty")
        if not isinstance(self.service_time_s_by_mechanism, (list, tuple)):
            raise TypeError(
                "runtime prepare service costs must be a sequence")
        entries = tuple(self.service_time_s_by_mechanism)
        normalized = []
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    "runtime prepare service entry must be a pair")
            mechanism, value = entry
            if not isinstance(mechanism, str) or not mechanism:
                raise ValueError(
                    "runtime prepare mechanism must be non-empty")
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    "runtime prepare service time must be finite and positive")
            normalized.append((mechanism, float(value)))
        if len({mechanism for mechanism, _ in normalized}) != len(normalized):
            raise ValueError("runtime prepare mechanisms must be unique")
        object.__setattr__(self, "service_time_s_by_mechanism", tuple(normalized))

        if not isinstance(self.lane_service_time_s_by_mechanism, (list, tuple)):
            raise TypeError(
                "lane runtime prepare service costs must be a sequence")
        lane_entries = tuple(self.lane_service_time_s_by_mechanism)
        normalized_lanes = []
        for entry in lane_entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise ValueError(
                    "lane runtime prepare service entry must be a quadruple")
            pool_name, rank, mechanism, value = entry
            if not isinstance(pool_name, str) or not pool_name:
                raise ValueError("runtime prepare lane pool name must be non-empty")
            if (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
                raise ValueError(
                    "runtime prepare lane rank must be a non-negative integer")
            if not isinstance(mechanism, str) or not mechanism:
                raise ValueError(
                    "runtime prepare mechanism must be non-empty")
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    "runtime prepare lane service time must be finite and positive")
            normalized_lanes.append((pool_name, rank, mechanism, float(value)))
        lane_keys = tuple(
            (pool_name, rank, mechanism)
            for pool_name, rank, mechanism, _ in normalized_lanes
        )
        if len(set(lane_keys)) != len(lane_keys):
            raise ValueError("runtime prepare lane mechanisms must be unique")
        object.__setattr__(
            self,
            "lane_service_time_s_by_mechanism",
            tuple(normalized_lanes),
        )
        if (self.profile_path is None) != (self.profile_sha256 is None):
            raise ValueError(
                "runtime prepare profile path and SHA256 must be supplied together")
        if self.profile_path is not None:
            if not isinstance(self.profile_path, str) or not self.profile_path:
                raise ValueError("runtime prepare profile path must be non-empty")
            if (not isinstance(self.profile_sha256, str)
                    or len(self.profile_sha256) != 64
                    or any(character not in "0123456789abcdef"
                           for character in self.profile_sha256)):
                raise ValueError(
                    "runtime prepare profile SHA256 must be a lowercase digest")

    def service_time_s(
        self,
        mechanism: str,
        *,
        pool_name: str | None = None,
        rank: int | None = None,
    ) -> float:
        if not isinstance(mechanism, str) or not mechanism:
            raise ValueError("runtime prepare mechanism must be non-empty")
        if (pool_name is None) != (rank is None):
            raise ValueError(
                "runtime prepare service lookup requires both pool_name and rank")
        lane_entries = tuple(
            entry for entry in self.lane_service_time_s_by_mechanism
            if entry[2] == mechanism
        )
        if lane_entries:
            if pool_name is None:
                raise ValueError(
                    "lane-specific runtime prepare service requires a Host lane")
            if not isinstance(pool_name, str) or not pool_name:
                raise ValueError("runtime prepare lane pool name must be non-empty")
            if (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0):
                raise ValueError(
                    "runtime prepare lane rank must be a non-negative integer")
            for candidate_pool, candidate_rank, _, value in lane_entries:
                if (candidate_pool, candidate_rank) == (pool_name, rank):
                    return value
            raise ValueError(
                "runtime prepare mechanism has lane-specific services but "
                f"does not cover Host lane {(pool_name, rank)!r}: {mechanism!r}")
        for candidate, value in self.service_time_s_by_mechanism:
            if candidate == mechanism:
                return value
        raise ValueError(
            f"runtime prepare mechanism is absent from profile: {mechanism!r}")

    def manifest(self) -> dict[str, object]:
        return {
            "hardware_id": self.hardware_id,
            "service_time_s_by_mechanism": {
                mechanism: value
                for mechanism, value in self.service_time_s_by_mechanism
            },
            "lane_service_time_s_by_mechanism": [
                {
                    "pool_name": pool_name,
                    "rank": rank,
                    "mechanism": mechanism,
                    "service_time_s": value,
                }
                for pool_name, rank, mechanism, value
                in self.lane_service_time_s_by_mechanism
            ],
            "profile_path": self.profile_path,
            "profile_sha256": self.profile_sha256,
        }


def validate_serving_timing_mode(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Serving timing mode must be a string")
    if value not in SERVING_TIMING_MODES:
        raise ValueError(f"unknown Serving timing mode: {value!r}")
    return value


def schedule_phase(schedule) -> str:
    phases = {request.phase for request in schedule.requests}
    if phases == {"prefill"}:
        return "prefill"
    if phases == {"decode"}:
        return "decode"
    if phases == {"prefill", "decode"}:
        return "mixed"
    raise ValueError(f"unsupported Scheduler step phase set: {sorted(phases)!r}")


@dataclass(frozen=True)
class RuntimeGraphSegmentDescriptor:
    """One source-authored Graph replay segment on a concrete Host lane."""

    segment_id: str
    pool_name: str
    rank: int
    call_occurrences: Tuple[Tuple[str, int], ...]
    sync_before_replay: bool = False

    def __post_init__(self) -> None:
        for name in ("segment_id", "pool_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"runtime Graph segment {name} must be non-empty")
        if (isinstance(self.rank, bool) or not isinstance(self.rank, int)
                or self.rank < 0):
            raise ValueError(
                "runtime Graph segment rank must be a non-negative integer")
        if not isinstance(self.call_occurrences, (list, tuple)):
            raise TypeError(
                "runtime Graph segment call occurrences must be a sequence")
        occurrences = tuple(
            tuple(occurrence) for occurrence in self.call_occurrences)
        if not occurrences:
            raise ValueError(
                "runtime Graph segment must cover at least one call occurrence")
        for occurrence in occurrences:
            if len(occurrence) != 2:
                raise ValueError(
                    "runtime Graph segment occurrence must be a call/index pair")
            call_id, occurrence_index = occurrence
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(
                    "runtime Graph segment call ID must be non-empty")
            if (isinstance(occurrence_index, bool)
                    or not isinstance(occurrence_index, int)
                    or occurrence_index < -1):
                raise ValueError(
                    "runtime Graph segment occurrence index must be >= -1")
        if len(occurrences) != len(set(occurrences)):
            raise ValueError(
                "runtime Graph segment call occurrences must be unique")
        if not isinstance(self.sync_before_replay, bool):
            raise TypeError(
                "runtime Graph segment sync_before_replay must be a bool")
        object.__setattr__(self, "call_occurrences", occurrences)


DEVICE_LAUNCH_GAP_CLASSES = (
    "graph_node", "graph_entry", "graph_exit", "eager")
DEVICE_LAUNCH_GAP_BOUNDARY = "device_side_host_ahead"


@dataclass(frozen=True)
class DeviceLaunchGapProfile:
    """Device-side idle between consecutive launches on one stream.

    Measured with the Host already ahead of the device, so every gap is the
    runtime/command-processor service between one launch retiring and the
    next one starting, classified by the launch boundary it crosses:
    ``graph_node`` (next node inside one replayed graph), ``graph_entry``
    (first node of a replay after other work), ``graph_exit`` (eager launch
    after a replay) and ``eager`` (eager launch after an eager launch).
    """

    gap_s: Mapping[str, float]
    runtime_version: str
    framework_version: str
    measurement_artifact: str
    measurement_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.gap_s, Mapping) or set(self.gap_s) != set(
                DEVICE_LAUNCH_GAP_CLASSES):
            raise ValueError(
                "device launch gap classes must be exactly "
                f"{list(DEVICE_LAUNCH_GAP_CLASSES)}")
        for name, value in self.gap_s.items():
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"device launch gap {name} must be finite and positive")
        object.__setattr__(
            self, "gap_s", MappingProxyType(
                {name: float(value) for name, value in self.gap_s.items()}))
        for name in ("runtime_version", "framework_version",
                     "measurement_artifact"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"device launch gap {name} must be non-empty")
        if (len(self.measurement_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.measurement_sha256)):
            raise ValueError(
                "device launch gap measurement_sha256 must be a lowercase "
                "SHA256")

    def manifest(self) -> dict[str, object]:
        return {
            "gap_s": dict(self.gap_s),
            "runtime_version": self.runtime_version,
            "framework_version": self.framework_version,
            "measurement": {
                "artifact": self.measurement_artifact,
                "sha256": self.measurement_sha256,
                "boundary": DEVICE_LAUNCH_GAP_BOUNDARY,
            },
        }


@dataclass(frozen=True)
class HostLaunchProfile:
    """One independently measured Host/runtime launch service profile."""

    hardware_id: str
    host_cpu: str
    operating_system: str
    driver_version: str
    runtime_name: str
    runtime_version: str
    framework_name: str
    framework_version: str
    eager_enqueue_s: float
    graph_replay_enqueue_s: float
    measurement_artifact: str
    measurement_sha256: str
    source_commit: str
    source_artifact: str
    source_sha256: str
    # ``None`` when the profile carries no device-side launch gap evidence;
    # the timeline then charges no inter-launch device idle.
    device_launch_gaps: DeviceLaunchGapProfile | None

    def __post_init__(self) -> None:
        if (self.device_launch_gaps is not None
                and not isinstance(
                    self.device_launch_gaps, DeviceLaunchGapProfile)):
            raise TypeError(
                "Host launch profile device_launch_gaps must be a "
                "DeviceLaunchGapProfile or None")
        for name in (
                "hardware_id", "host_cpu", "operating_system",
                "driver_version", "runtime_name", "runtime_version",
                "framework_name", "framework_version",
                "measurement_artifact", "source_commit", "source_artifact",
                "source_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Host launch profile {name} must be non-empty")
        for name in ("measurement_sha256", "source_sha256"):
            value = getattr(self, name)
            if (len(value) != 64
                    or any(character not in "0123456789abcdef"
                           for character in value)):
                raise ValueError(
                    f"Host launch profile {name} must be a lowercase SHA256")
        if (len(self.source_commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in self.source_commit)):
            raise ValueError(
                "Host launch profile source_commit must be a full lowercase Git SHA")
        for name in ("eager_enqueue_s", "graph_replay_enqueue_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"Host launch profile {name} must be finite and positive")
    def service_time_s(self, launch_kind: str) -> float:
        if launch_kind == "eager":
            return float(self.eager_enqueue_s)
        if launch_kind == "graph_replay":
            return float(self.graph_replay_enqueue_s)
        raise ValueError(f"unknown Host launch kind: {launch_kind!r}")

    def device_launch_gap_s(self, gap_class: str) -> float:
        if self.device_launch_gaps is None:
            return 0.0
        if gap_class not in DEVICE_LAUNCH_GAP_CLASSES:
            raise ValueError(f"unknown device launch gap class: {gap_class!r}")
        return self.device_launch_gaps.gap_s[gap_class]

    def manifest(self) -> dict[str, object]:
        manifest = self._base_manifest()
        if self.device_launch_gaps is not None:
            manifest["device_launch_gap"] = self.device_launch_gaps.manifest()
        return manifest

    def _base_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "hardware_id": self.hardware_id,
            "host_cpu": self.host_cpu,
            "operating_system": self.operating_system,
            "driver_version": self.driver_version,
            "runtime": {
                "name": self.runtime_name,
                "version": self.runtime_version,
            },
            "framework": {
                "name": self.framework_name,
                "version": self.framework_version,
            },
            "service_time_s": {
                "eager_enqueue": self.eager_enqueue_s,
                "graph_replay_enqueue": self.graph_replay_enqueue_s,
            },
            "measurement": {
                "artifact": self.measurement_artifact,
                "sha256": self.measurement_sha256,
                "boundary": "host_submit_only_no_device_sync",
            },
            "source": {
                "commit": self.source_commit,
                "path": self.source_artifact,
                "sha256": self.source_sha256,
            },
        }


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _positive_seconds(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise TypeError(f"{label} must be finite positive seconds")
    return float(value)


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if (len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return digest


def load_scheduler_host_profile(path: str | Path) -> SchedulerHostProfile:
    """Load an exact-schema deterministic Scheduler Host service profile."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = _mapping(json.loads(raw), "Scheduler Host profile")
    required_profile_fields = {
        "schema_version", "hardware_id", "scheduler", "model",
        "max_scheduled_request_count", "supported_phases", "service_time_s",
        "measurement",
    }
    allowed_profile_fields = required_profile_fields | {"fallback_dispatch", "shared_cpu_timing_target"}
    if (not required_profile_fields.issubset(payload)
            or set(payload) - allowed_profile_fields):
        _exact_fields(payload, required_profile_fields, "Scheduler Host profile")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Scheduler Host profile schema_version must equal 1")
    scheduler = _mapping(payload["scheduler"], "Scheduler Host scheduler")
    model = _mapping(payload["model"], "Scheduler Host model")
    measurement = _mapping(payload["measurement"], "Scheduler Host measurement")
    _exact_fields(scheduler, {"vllm_version", "source_identity"},
                  "Scheduler Host scheduler")
    _exact_fields(model, {"kind", "state_dimensions", "statistic"},
                  "Scheduler Host model")
    _exact_fields(measurement, {
        "boundary", "time_unit", "training_selection", "artifacts",
    },
                  "Scheduler Host measurement")
    if model["kind"] != "scheduled_request_count_affine":
        raise ValueError("Scheduler Host model kind is invalid")
    if model["state_dimensions"] != ["scheduled_request_count"]:
        raise ValueError("Scheduler Host model state dimensions are invalid")
    if model["statistic"] != "median_of_count_medians_theil_sen":
        raise ValueError("Scheduler Host model statistic is invalid")
    if measurement["boundary"] != (
            "vllm_scheduler_trace_host_monotonic_plan_and_update"):
        raise ValueError("Scheduler Host measurement boundary is invalid")
    if measurement["time_unit"] != "seconds":
        raise ValueError("Scheduler Host measurement time unit is invalid")
    training_selection = _mapping(
        measurement["training_selection"], "Scheduler Host training selection")
    _exact_fields(training_selection, {
        "excluded_target_qps", "minimum_samples_per_phase",
        "eligible_step_rule",
    }, "Scheduler Host training selection")
    excluded_target_qps = training_selection["excluded_target_qps"]
    if (isinstance(excluded_target_qps, bool)
            or not isinstance(excluded_target_qps, int)
            or excluded_target_qps < 1):
        raise ValueError("Scheduler Host excluded target QPS is invalid")
    minimum_samples_per_phase = training_selection["minimum_samples_per_phase"]
    if (isinstance(minimum_samples_per_phase, bool)
            or not isinstance(minimum_samples_per_phase, int)
            or minimum_samples_per_phase < 1):
        raise ValueError(
            "Scheduler Host minimum samples per phase is invalid")
    if training_selection["eligible_step_rule"] != (
            "nonzero_scheduled_request_count_with_positive_plan_and_update"):
        raise ValueError("Scheduler Host eligible step rule is invalid")
    artifacts = measurement["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Scheduler Host measurement artifacts must be non-empty")
    repository_root = Path(__file__).resolve().parents[2]
    for artifact in artifacts:
        item = _mapping(artifact, "Scheduler Host measurement artifact")
        _exact_fields(item, {"path", "sha256"},
                      "Scheduler Host measurement artifact")
        artifact_path = Path(_string(item["path"], "Scheduler Host artifact path"))
        if not artifact_path.is_absolute():
            artifact_path = (repository_root / artifact_path).resolve()
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != _sha256(
                item["sha256"], "Scheduler Host artifact SHA256"):
            raise ValueError(f"Scheduler Host artifact SHA256 mismatch: {artifact_path}")
    maximum = payload["max_scheduled_request_count"]
    if (isinstance(maximum, bool) or not isinstance(maximum, int)
            or maximum < 1):
        raise ValueError("Scheduler Host maximum request count must be positive")
    supported_phases_raw = payload["supported_phases"]
    if not isinstance(supported_phases_raw, list) or not supported_phases_raw:
        raise ValueError("Scheduler Host supported phases must be a non-empty list")
    supported_phases = tuple(supported_phases_raw)
    if (any(not isinstance(phase, str) or phase not in STEP_PHASES
            for phase in supported_phases)
            or len(set(supported_phases)) != len(supported_phases)):
        raise ValueError("Scheduler Host supported phases are invalid")
    services = _mapping(payload["service_time_s"], "Scheduler Host services")
    supports_prefill = "prefill" in supported_phases
    _exact_fields(
        services,
        {"plan", "update", "prefill"} if supports_prefill else {"plan", "update"},
        "Scheduler Host services")
    coefficients = []
    for operation in SCHEDULER_HOST_OPERATIONS:
        item = _mapping(services[operation], f"Scheduler Host {operation}")
        _exact_fields(item, {"intercept", "per_request"},
                      f"Scheduler Host {operation}")
        coefficients.extend((
            _positive_seconds(item["intercept"],
                              f"Scheduler Host {operation} intercept"),
            _positive_seconds(item["per_request"],
                              f"Scheduler Host {operation} per-request"),
        ))
    prefill_coefficients = [None, None, None, None]
    if supports_prefill:
        prefill = _mapping(services["prefill"], "Scheduler Host prefill services")
        _exact_fields(prefill, {"plan", "update"},
                      "Scheduler Host prefill services")
        prefill_coefficients = []
        for operation in SCHEDULER_HOST_OPERATIONS:
            item = _mapping(prefill[operation],
                            f"Scheduler Host prefill {operation}")
            _exact_fields(item, {"intercept", "per_request"},
                          f"Scheduler Host prefill {operation}")
            prefill_coefficients.extend((
                _positive_seconds(
                    item["intercept"],
                    f"Scheduler Host prefill {operation} intercept"),
                    _positive_seconds(
                        item["per_request"],
                        f"Scheduler Host prefill {operation} per-request"),
            ))
    fallback_values = (None, None, None, None, None, None)
    if "fallback_dispatch" in payload:
        fallback = _mapping(
            payload["fallback_dispatch"], "Scheduler Host fallback dispatch")
        _exact_fields(fallback, {
            "min_scheduled_request_count", "max_scheduled_request_count",
            "supported_phases", "service_time_s", "occurrence_contract",
        }, "Scheduler Host fallback dispatch")
        fallback_minimum = fallback["min_scheduled_request_count"]
        fallback_maximum = fallback["max_scheduled_request_count"]
        for name, value in (("minimum", fallback_minimum),
                            ("maximum", fallback_maximum)):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 1):
                raise ValueError(f"Scheduler Host fallback dispatch {name} is invalid")
        if fallback_minimum <= maximum or fallback_maximum < fallback_minimum:
            raise ValueError("Scheduler Host fallback dispatch range is invalid")
        fallback_phases_raw = fallback["supported_phases"]
        if (not isinstance(fallback_phases_raw, list) or not fallback_phases_raw
                or any(phase not in STEP_PHASES for phase in fallback_phases_raw)
                or len(fallback_phases_raw) != len(set(fallback_phases_raw))):
            raise ValueError("Scheduler Host fallback dispatch phases are invalid")
        fallback_services = _mapping(
            fallback["service_time_s"], "Scheduler Host fallback dispatch services")
        _exact_fields(fallback_services, {"plan", "update"},
                      "Scheduler Host fallback dispatch services")
        fallback_plan = _positive_seconds(
            fallback_services["plan"], "Scheduler Host fallback dispatch plan")
        fallback_update = _positive_seconds(
            fallback_services["update"], "Scheduler Host fallback dispatch update")
        occurrence_contract = _string(
            fallback["occurrence_contract"],
            "Scheduler Host fallback dispatch occurrence contract")
        fallback_values = (
            fallback_minimum, fallback_maximum, tuple(fallback_phases_raw),
            fallback_plan, fallback_update, occurrence_contract)
    return SchedulerHostProfile(
        hardware_id=_string(payload["hardware_id"], "Scheduler Host hardware ID"),
        vllm_version=_string(scheduler["vllm_version"], "Scheduler Host vLLM version"),
        source_identity=_string(
            scheduler["source_identity"], "Scheduler Host source identity"),
        shared_cpu_timing_target=(payload["shared_cpu_timing_target"]
            if "shared_cpu_timing_target" in payload else None),
        max_scheduled_request_count=maximum,
        supported_phases=supported_phases,
        plan_intercept_s=coefficients[0],
        plan_per_request_s=coefficients[1],
        update_intercept_s=coefficients[2],
        update_per_request_s=coefficients[3],
        prefill_plan_intercept_s=prefill_coefficients[0],
        prefill_plan_per_request_s=prefill_coefficients[1],
        prefill_update_intercept_s=prefill_coefficients[2],
        prefill_update_per_request_s=prefill_coefficients[3],
        fallback_dispatch_min_scheduled_request_count=fallback_values[0],
        fallback_dispatch_max_scheduled_request_count=fallback_values[1],
        fallback_dispatch_supported_phases=fallback_values[2],
        fallback_dispatch_plan_s=fallback_values[3],
        fallback_dispatch_update_s=fallback_values[4],
        fallback_dispatch_occurrence_contract=fallback_values[5],
        profile_path=str(source_path),
        profile_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _non_negative_seconds(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise TypeError(f"{label} must be finite non-negative seconds")
    return float(value)


def load_engine_core_host_round_profile(
    path: str | Path,
) -> EngineCoreHostRoundProfile:
    """Load an exact-schema deterministic EngineCore host-round profile."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = _mapping(json.loads(raw), "EngineCore host-round profile")
    _exact_fields(payload, {
        "schema_version", "hardware_id", "scheduler", "model",
        "max_scheduled_request_count", "service_time_s", "measurement",
    }, "EngineCore host-round profile")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError(
            "EngineCore host-round profile schema_version must equal 1")
    scheduler = _mapping(payload["scheduler"], "EngineCore host-round scheduler")
    model = _mapping(payload["model"], "EngineCore host-round model")
    measurement = _mapping(
        payload["measurement"], "EngineCore host-round measurement")
    _exact_fields(scheduler, {"vllm_version", "source_identity"},
                  "EngineCore host-round scheduler")
    _exact_fields(model, {
        "kind", "state_dimensions", "lane_dispatch", "statistic",
    }, "EngineCore host-round model")
    if model["kind"] != "lane_dispatch_scheduled_request_count_affine":
        raise ValueError("EngineCore host-round model kind is invalid")
    if model["state_dimensions"] != ["scheduled_request_count"]:
        raise ValueError(
            "EngineCore host-round model state dimensions are invalid")
    if model["lane_dispatch"] != ENGINE_CORE_HOST_ROUND_LANE_DISPATCH:
        raise ValueError("EngineCore host-round model lane dispatch is invalid")
    if model["statistic"] != "median_of_count_medians_then_least_squares_slope":
        raise ValueError("EngineCore host-round model statistic is invalid")
    _exact_fields(measurement, {
        "boundary", "time_unit", "chain_definition", "training_selection",
        "artifacts",
    }, "EngineCore host-round measurement")
    if measurement["boundary"] != "vllm_engine_core_host_round_mechanism_probe":
        raise ValueError(
            "EngineCore host-round measurement boundary is invalid")
    if measurement["time_unit"] != "seconds":
        raise ValueError(
            "EngineCore host-round measurement time unit is invalid")
    _string(measurement["chain_definition"],
            "EngineCore host-round chain definition")
    training_selection = _mapping(
        measurement["training_selection"],
        "EngineCore host-round training selection")
    _exact_fields(training_selection, {
        "eligible_round_rule", "function_form_rule", "parameter_run",
        "retest_runs",
    }, "EngineCore host-round training selection")
    _string(training_selection["eligible_round_rule"],
            "EngineCore host-round eligible round rule")
    _string(training_selection["function_form_rule"],
            "EngineCore host-round function form rule")
    _string(training_selection["parameter_run"],
            "EngineCore host-round parameter run")
    retest_runs = training_selection["retest_runs"]
    if (not isinstance(retest_runs, list) or not retest_runs
            or any(not isinstance(run, str) or not run
                   for run in retest_runs)
            or len(set(retest_runs)) != len(retest_runs)):
        raise ValueError(
            "EngineCore host-round retest runs must be unique non-empty strings")
    artifacts = measurement["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(
            "EngineCore host-round measurement artifacts must be non-empty")
    repository_root = Path(__file__).resolve().parents[2]
    for artifact in artifacts:
        item = _mapping(artifact, "EngineCore host-round measurement artifact")
        _exact_fields(item, {"path", "sha256"},
                      "EngineCore host-round measurement artifact")
        artifact_path = Path(
            _string(item["path"], "EngineCore host-round artifact path"))
        if not artifact_path.is_absolute():
            artifact_path = (repository_root / artifact_path).resolve()
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != _sha256(
                item["sha256"], "EngineCore host-round artifact SHA256"):
            raise ValueError(
                f"EngineCore host-round artifact SHA256 mismatch: {artifact_path}")
    maximum = payload["max_scheduled_request_count"]
    if (isinstance(maximum, bool) or not isinstance(maximum, int)
            or maximum < 1):
        raise ValueError(
            "EngineCore host-round maximum request count must be positive")
    services = _mapping(
        payload["service_time_s"], "EngineCore host-round services")
    _exact_fields(services, {"graph_lane", "eager_lane"},
                  "EngineCore host-round services")
    submit_intercepts = {}
    wake_intercepts = {}
    wake_per_request = {}
    for lane in ENGINE_CORE_HOST_ROUND_LANES:
        lane_services = _mapping(
            services[f"{lane}_lane"], f"EngineCore host-round {lane} lane")
        _exact_fields(lane_services, {"submit", "wake"},
                      f"EngineCore host-round {lane} lane")
        submit = _mapping(
            lane_services["submit"],
            f"EngineCore host-round {lane} submit")
        _exact_fields(submit, {"intercept"},
                      f"EngineCore host-round {lane} submit")
        submit_intercepts[lane] = _positive_seconds(
            submit["intercept"],
            f"EngineCore host-round {lane} submit intercept")
        wake = _mapping(
            lane_services["wake"], f"EngineCore host-round {lane} wake")
        _exact_fields(wake, {"intercept", "per_request"},
                      f"EngineCore host-round {lane} wake")
        wake_intercepts[lane] = _positive_seconds(
            wake["intercept"], f"EngineCore host-round {lane} wake intercept")
        wake_per_request[lane] = _non_negative_seconds(
            wake["per_request"],
            f"EngineCore host-round {lane} wake per-request")
    return EngineCoreHostRoundProfile(
        hardware_id=_string(
            payload["hardware_id"], "EngineCore host-round hardware ID"),
        vllm_version=_string(
            scheduler["vllm_version"], "EngineCore host-round vLLM version"),
        source_identity=_string(
            scheduler["source_identity"],
            "EngineCore host-round source identity"),
        max_scheduled_request_count=maximum,
        graph_submit_intercept_s=submit_intercepts["graph"],
        graph_wake_intercept_s=wake_intercepts["graph"],
        graph_wake_per_request_s=wake_per_request["graph"],
        eager_submit_intercept_s=submit_intercepts["eager"],
        eager_wake_intercept_s=wake_intercepts["eager"],
        eager_wake_per_request_s=wake_per_request["eager"],
        profile_path=str(source_path),
        profile_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_host_runtime_prepare_profile(path: str | Path) -> HostRuntimePrepareProfile:
    """Load one independently measured runtime-prepare profile.

    The profile contains only generic runtime mechanisms and their independent
    microbenchmark service times.  Source occurrence rules stay in Serving;
    it deliberately has no per-model, per-shape, or Serving-duration field.
    """

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = _mapping(json.loads(raw), "Host runtime prepare profile")
    _exact_fields(payload, {
        "schema_version", "hardware_id", "runtime", "framework",
        "service_time_s", "measurement",
    }, "Host runtime prepare profile")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Host runtime prepare profile schema_version must equal 1")
    runtime = _mapping(payload["runtime"], "Host runtime prepare runtime")
    framework = _mapping(payload["framework"], "Host runtime prepare framework")
    service = _mapping(payload["service_time_s"], "Host runtime prepare service")
    measurement = _mapping(payload["measurement"], "Host runtime prepare measurement")
    _exact_fields(runtime, {"name", "version"}, "Host runtime prepare runtime")
    _exact_fields(framework, {"name", "version"}, "Host runtime prepare framework")
    _string(runtime["name"], "runtime prepare runtime name")
    _string(runtime["version"], "runtime prepare runtime version")
    _string(framework["name"], "runtime prepare framework name")
    _string(framework["version"], "runtime prepare framework version")
    _exact_fields(measurement, {"boundary", "artifacts"},
                  "Host runtime prepare measurement")
    if measurement["boundary"] != "independent_runtime_mechanism_microbenchmark":
        raise ValueError("runtime prepare measurement boundary is invalid")
    artifacts = measurement["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("runtime prepare measurement artifacts must be non-empty")
    state_enabled = set(service) & set(WORKER_REQUEST_STATE_UNITS)
    if state_enabled and state_enabled != set(WORKER_REQUEST_STATE_UNITS):
        raise ValueError("worker request state profile requires its complete service group")
    state_references = []
    for artifact in artifacts:
        item = _mapping(artifact, "runtime prepare measurement artifact")
        _exact_fields(item, {"path", "sha256"},
                      "runtime prepare measurement artifact")
        _string(item["path"], "runtime prepare measurement artifact path")
        digest = _string(item["sha256"], "runtime prepare measurement artifact SHA256")
        if (len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("runtime prepare measurement artifact SHA256 is invalid")
        if state_enabled:
            artifact_path = Path(item["path"])
            if not artifact_path.is_absolute():
                artifact_path = Path(__file__).resolve().parents[2] / artifact_path
            content = artifact_path.read_bytes()
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError("worker state profile measurement SHA256 mismatch")
            reference = _mapping(json.loads(content), "runtime prepare reference")
            if "kind" in reference and reference["kind"] == "independent_vllm030_v2_worker_request_state":
                state_references.append(reference)
    if state_enabled:
        if len(state_references) != 1:
            raise ValueError("worker state profile requires exactly one independent state reference")
        reference = state_references[0]
        if reference["production_duration_consumed"] is not False or reference["time_unit"] != "seconds":
            raise ValueError("worker state reference has invalid independent time provenance")
        if dict(reference["units"]) != dict(WORKER_REQUEST_STATE_UNITS):
            raise ValueError("worker state reference units differ from source-count services")
        values = _mapping(reference["services_s"], "worker state reference services")
        _exact_fields(values, set(WORKER_REQUEST_STATE_UNITS), "worker state reference services")
        if any(service[name] != values[name] for name in WORKER_REQUEST_STATE_UNITS):
            raise ValueError("worker state profile services differ from their independent reference")
        geometry = reference["geometry"]
        if (geometry["max_requests"], geometry["max_model_len"], geometry["block_size_tokens"]) != (256, 8192, 16):
            raise ValueError("worker state reference geometry differs from measured scope")
    if not service:
        raise ValueError("runtime prepare service profile must be non-empty")
    if any(not isinstance(mechanism, str) or not mechanism for mechanism in service):
        raise ValueError("runtime prepare mechanism names must be non-empty strings")
    return HostRuntimePrepareProfile(
        hardware_id=_string(payload["hardware_id"], "runtime prepare hardware_id"),
        service_time_s_by_mechanism=tuple(
            (mechanism, _positive_seconds(value, f"runtime prepare {mechanism}"))
            for mechanism, value in service.items()
        ),
        profile_path=str(source_path),
        profile_sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_host_launch_profile(path: str | Path) -> HostLaunchProfile:
    """Load one exact-schema profile and verify its retained measurement."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = _mapping(json.loads(raw), "Host launch profile")
    profile_fields = {
        "schema_version", "hardware_id", "host_cpu", "operating_system",
        "driver_version", "runtime", "framework", "service_time_s",
        "measurement", "source",
    }
    # ``device_launch_gap`` is the one optional section of the schema.
    if "device_launch_gap" in payload:
        profile_fields.add("device_launch_gap")
    _exact_fields(payload, profile_fields, "Host launch profile")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 2:
        raise ValueError("Host launch profile schema_version must equal 2")
    runtime = _mapping(payload["runtime"], "Host launch runtime")
    framework = _mapping(payload["framework"], "Host launch framework")
    service = _mapping(payload["service_time_s"], "Host launch service time")
    measurement = _mapping(payload["measurement"], "Host launch measurement")
    source = _mapping(payload["source"], "Host launch source")
    _exact_fields(runtime, {"name", "version"}, "Host launch runtime")
    _exact_fields(framework, {"name", "version"}, "Host launch framework")
    _exact_fields(service, {"eager_enqueue", "graph_replay_enqueue"},
                  "Host launch service time")
    _exact_fields(measurement, {"artifact", "sha256", "boundary"},
                  "Host launch measurement")
    _exact_fields(source, {"commit", "path", "sha256"}, "Host launch source")
    if measurement["boundary"] != "host_submit_only_no_device_sync":
        raise ValueError("Host launch measurement boundary is not submit-only")
    artifact_value = _string(measurement["artifact"], "measurement artifact")
    repository_root = next(
        candidate for candidate in source_path.parents
        if (candidate / "pyproject.toml").is_file())
    artifact_path = Path(artifact_value)
    if not artifact_path.is_absolute():
        artifact_path = (repository_root / artifact_path).resolve()
    actual_artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    expected_artifact_sha256 = _string(
        measurement["sha256"], "measurement SHA256")
    if actual_artifact_sha256 != expected_artifact_sha256:
        raise ValueError(
            "Host launch measurement SHA256 mismatch: "
            f"expected={expected_artifact_sha256}, actual={actual_artifact_sha256}")
    source_artifact = _string(source["path"], "source path")
    source_artifact_path = Path(source_artifact)
    if not source_artifact_path.is_absolute():
        source_artifact_path = (repository_root / source_artifact_path).resolve()
    actual_source_sha256 = hashlib.sha256(
        source_artifact_path.read_bytes()).hexdigest()
    expected_source_sha256 = _string(source["sha256"], "source SHA256")
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "Host launch source SHA256 mismatch: "
            f"expected={expected_source_sha256}, actual={actual_source_sha256}")
    device_launch_gaps = None
    if "device_launch_gap" in payload:
        gap = _mapping(payload["device_launch_gap"], "device launch gap")
        _exact_fields(gap, {
            "gap_s", "runtime_version", "framework_version", "measurement",
        }, "device launch gap")
        gap_values = _mapping(gap["gap_s"], "device launch gap values")
        _exact_fields(gap_values, set(DEVICE_LAUNCH_GAP_CLASSES),
                      "device launch gap values")
        gap_measurement = _mapping(
            gap["measurement"], "device launch gap measurement")
        _exact_fields(gap_measurement, {"artifact", "sha256", "boundary"},
                      "device launch gap measurement")
        if gap_measurement["boundary"] != DEVICE_LAUNCH_GAP_BOUNDARY:
            raise ValueError(
                "device launch gap measurement boundary must be "
                f"{DEVICE_LAUNCH_GAP_BOUNDARY!r}")
        gap_artifact = _string(
            gap_measurement["artifact"], "device launch gap artifact")
        gap_artifact_path = Path(gap_artifact)
        if not gap_artifact_path.is_absolute():
            gap_artifact_path = (repository_root / gap_artifact_path).resolve()
        expected_gap_sha256 = _sha256(
            gap_measurement["sha256"], "device launch gap SHA256")
        actual_gap_sha256 = hashlib.sha256(
            gap_artifact_path.read_bytes()).hexdigest()
        if actual_gap_sha256 != expected_gap_sha256:
            raise ValueError(
                "device launch gap measurement SHA256 mismatch: "
                f"expected={expected_gap_sha256}, actual={actual_gap_sha256}")
        device_launch_gaps = DeviceLaunchGapProfile(
            gap_s={
                name: _positive_seconds(
                    gap_values[name], f"device launch gap {name}")
                for name in DEVICE_LAUNCH_GAP_CLASSES
            },
            runtime_version=_string(
                gap["runtime_version"], "device launch gap runtime version"),
            framework_version=_string(
                gap["framework_version"],
                "device launch gap framework version"),
            measurement_artifact=gap_artifact,
            measurement_sha256=expected_gap_sha256,
        )
    return HostLaunchProfile(
        hardware_id=_string(payload["hardware_id"], "hardware_id"),
        host_cpu=_string(payload["host_cpu"], "host_cpu"),
        operating_system=_string(payload["operating_system"], "operating_system"),
        driver_version=_string(payload["driver_version"], "driver_version"),
        runtime_name=_string(runtime["name"], "runtime name"),
        runtime_version=_string(runtime["version"], "runtime version"),
        framework_name=_string(framework["name"], "framework name"),
        framework_version=_string(framework["version"], "framework version"),
        eager_enqueue_s=_positive_seconds(
            service["eager_enqueue"], "eager enqueue time"),
        graph_replay_enqueue_s=_positive_seconds(
            service["graph_replay_enqueue"], "Graph replay enqueue time"),
        measurement_artifact=artifact_value,
        measurement_sha256=expected_artifact_sha256,
        source_commit=_string(source["commit"], "source commit"),
        source_artifact=source_artifact,
        source_sha256=expected_source_sha256,
        device_launch_gaps=device_launch_gaps,
    )


@dataclass(frozen=True)
class HostRequestPathProfile:
    """Per-request Host cost on the client-facing path around one request.

    Two disjoint source regions are owned here, both measured by the
    independent ``host_request_path`` microbench of the serving Host:

    ``admission`` covers the runtime path from the client submitting a request
    to that request becoming visible to the Scheduler, that is the OpenAI
    entrypoint handler entry, the asynchronous tokenizer wrapper, the marginal
    tokenizer encode of the prompt, ``Processor.process_inputs`` with the
    msgpack request build, and the EngineCore IPC transit.  It ends at
    ``Scheduler.add_request``.

    ``first_token_delivery`` covers the path after the Scheduler has produced a
    step's outputs: the EngineCore output IPC, the incremental detokenize and
    ``RequestOutput`` assembly in the output processor, and the streamed HTTP
    chunk that carries the first token to the client.

    That path is walked once per output of the step, not once per step.
    ``OutputProcessor.process_outputs`` iterates the step's ``EngineCoreOutput``
    list -- one entry per request the Scheduler put in that step -- and the same
    API-server event loop then writes one streamed chunk per connected request.
    A request's own chunk therefore leaves behind the other outputs of its step,
    so ``output_delivery_per_output`` charges the marginal cost of one further
    output in the same step and ``first_token_delivery`` is the single-output
    value.  Both come from the same ``host_request_path`` microbench.

    Neither region overlaps an existing Host mechanism.  ``Scheduler.schedule``
    and ``Scheduler.update_from_output`` belong to :class:`SchedulerHostProfile`,
    kernel enqueue belongs to :class:`HostLaunchProfile`, and per-call prepare
    helpers belong to :class:`HostRuntimePrepareProfile`.  The EngineCore
    busy-loop wait between a queued request and the next ``schedule`` call is
    deliberately charged by neither: the simulated step loop reproduces it.

    Reaching the tokenizer and running it are two separate costs.  How the
    handler reaches it is a per-runtime source fact and only one of the two
    forms is ever in domain: a runtime whose source has the batching wrapper
    pays that wrapper's uncontended call, and a runtime without it dispatches
    the encode straight onto the executor and pays that round trip instead.
    Running it then costs every runtime the same shape whichever way it was
    reached -- a fixed cost to enter the tokenizer for a call plus a marginal
    cost per prompt token -- so ``tokenizer_encode_intercept_s`` is charged on
    both paths rather than standing in for the entry form.
    """

    hardware_id: str
    runtime_version: str
    source_identity: str
    host_cpu: str
    api_entry_s: float
    input_processor_intercept_s: float
    input_processor_per_token_s: float
    admission_enqueue_s: float
    engine_core_ipc_s: float
    first_token_delivery_s: float
    output_delivery_per_output_s: float
    profile_path: str
    profile_sha256: str
    tokenizer_async_call_s: float | None = None
    tokenizer_encode_intercept_s: float | None = None
    tokenizer_encode_per_token_s: float | None = None
    tokenizer_render_region_intercept_s: float | None = None
    tokenizer_render_region_per_token_s: float | None = None
    shared_cpu_timing_target: str | None = None

    def supports_runtime_version(self, version: str) -> bool:
        return self.runtime_version == version or self.shared_cpu_timing_target == version

    def __post_init__(self) -> None:
        if self.shared_cpu_timing_target not in (None, "0.30.0"):
            raise ValueError("unsupported shared CPU timing target")
        for name in ("hardware_id", "runtime_version", "source_identity",
                     "host_cpu", "profile_path", "profile_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Host request path {name} must be non-empty")
        if (len(self.profile_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.profile_sha256)):
            raise ValueError("Host request path profile SHA256 is invalid")
        for name in ("api_entry_s",
                     "input_processor_intercept_s",
                     "input_processor_per_token_s",
                     "admission_enqueue_s", "engine_core_ipc_s",
                     "first_token_delivery_s",
                     "output_delivery_per_output_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"Host request path {name} must be finite positive")
        # Exactly one tokenizer entry form is in domain, and which one is a
        # source fact of the runtime.  A runtime whose handler reaches the
        # tokenizer through a batching wrapper is priced as that wrapper's
        # uncontended call plus the encode it then runs.  A runtime whose
        # handler instead calls a renderer is priced as that render region,
        # measured end to end -- it already contains the encode, so pricing an
        # encode on top of it would charge the same work twice.
        wrapper_form = (self.tokenizer_async_call_s is not None,
                        self.tokenizer_encode_intercept_s is not None,
                        self.tokenizer_encode_per_token_s is not None)
        region_form = (self.tokenizer_render_region_intercept_s is not None,
                       self.tokenizer_render_region_per_token_s is not None)
        if any(wrapper_form) and not all(wrapper_form):
            raise ValueError(
                "the batching wrapper entry form needs its call, encode "
                "intercept and encode per-token terms together")
        if any(region_form) and not all(region_form):
            raise ValueError(
                "the render region entry form needs both of its terms")
        if all(wrapper_form) == all(region_form):
            raise ValueError(
                "Host request path requires exactly one tokenizer entry form: "
                "a batching wrapper call with its encode, or a render region")
        for name in ("tokenizer_async_call_s", "tokenizer_encode_intercept_s",
                     "tokenizer_encode_per_token_s",
                     "tokenizer_render_region_intercept_s",
                     "tokenizer_render_region_per_token_s"):
            value = getattr(self, name)
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(
                    f"Host request path {name} must be finite positive")

    def admission_service_time_s(self, prompt_tokens: int) -> float:
        """Host cost charged once per request before Scheduler visibility.

        The occurrence is fixed by source: every request traverses the
        entrypoint handler, the tokenizer, the input processor and one IPC hop
        exactly once, whatever its prompt length or the load on the engine.
        Two terms depend on the request: the tokenizer encode, through the
        prompt length the same source path tokenizes, and the input processor,
        which copies and validates that prompt's token ids.
        """

        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            raise TypeError("prompt token count must be an integer")
        if prompt_tokens <= 0:
            raise ValueError("prompt token count must be positive")
        if self.tokenizer_async_call_s is not None:
            tokenizer_entry = (
                self.tokenizer_async_call_s
                + self.tokenizer_encode_intercept_s
                + self.tokenizer_encode_per_token_s * prompt_tokens)
        else:
            tokenizer_entry = (
                self.tokenizer_render_region_intercept_s
                + self.tokenizer_render_region_per_token_s * prompt_tokens)
        return (
            self.api_entry_s
            + tokenizer_entry
            + self.input_processor_intercept_s
            + self.input_processor_per_token_s * prompt_tokens
            + self.admission_enqueue_s
            + self.engine_core_ipc_s
        )

    def first_token_delivery_service_time_s(
        self, step_output_count: int
    ) -> float:
        """Host cost of delivering one token out of a step carrying ``n`` outputs.

        The occurrence is fixed by source: the output processor walks one
        entry per request scheduled in the step and the API server writes one
        streamed chunk per connected request, all on one event loop.  The
        delivered request pays the single-output path once plus the marginal
        cost of each further output the same step has to carry.
        """

        if (isinstance(step_output_count, bool)
                or not isinstance(step_output_count, int)):
            raise TypeError("step output count must be an integer")
        if step_output_count < 1:
            raise ValueError("a delivered token needs at least one step output")
        return (
            self.first_token_delivery_s
            + self.output_delivery_per_output_s * (step_output_count - 1)
        )

    def manifest(self) -> dict[str, object]:
        return {
            "hardware_id": self.hardware_id,
            "runtime_version": self.runtime_version,
            "source_identity": self.source_identity,
            **({"shared_cpu_timing_target": self.shared_cpu_timing_target}
               if self.shared_cpu_timing_target is not None else {}),
            "host_cpu": self.host_cpu,
            "service_time_s": {
                "api_entry": self.api_entry_s,
                "tokenizer_async_call": self.tokenizer_async_call_s,
                "tokenizer_render_region_intercept": (
                    self.tokenizer_render_region_intercept_s),
                "tokenizer_render_region_per_token": (
                    self.tokenizer_render_region_per_token_s),
                "tokenizer_encode_intercept": self.tokenizer_encode_intercept_s,
                "tokenizer_encode_per_token": self.tokenizer_encode_per_token_s,
                "input_processor_intercept": self.input_processor_intercept_s,
                "input_processor_per_token": self.input_processor_per_token_s,
                "admission_enqueue": self.admission_enqueue_s,
                "engine_core_ipc": self.engine_core_ipc_s,
                "first_token_delivery": self.first_token_delivery_s,
                "output_delivery_per_output": (
                    self.output_delivery_per_output_s),
            },
            "profile_path": self.profile_path,
            "profile_sha256": self.profile_sha256,
        }


def load_host_request_path_profile(path: str | Path) -> HostRequestPathProfile:
    """Load one exact-schema per-request Host request path profile."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    payload = _mapping(json.loads(raw), "Host request path profile")
    required = {"schema_version", "hardware_id", "runtime", "host",
                "service_time_s", "measurement"}
    if not required.issubset(payload) or set(payload) - (required | {"shared_cpu_timing_target"}):
        _exact_fields(payload, required, "Host request path profile")
    if (type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1):
        raise ValueError("Host request path profile schema_version must equal 1")
    runtime = _mapping(payload["runtime"], "Host request path runtime")
    _exact_fields(runtime, {"version", "source_identity"},
                  "Host request path runtime")
    host = _mapping(payload["host"], "Host request path host")
    _exact_fields(host, {"cpu"}, "Host request path host")
    service = _mapping(payload["service_time_s"], "Host request path services")
    _exact_fields(service, {
        "api_entry", "tokenizer_async_call",
        "tokenizer_render_region_intercept",
        "tokenizer_render_region_per_token",
        "tokenizer_encode_intercept",
        "tokenizer_encode_per_token",
        "input_processor_intercept", "input_processor_per_token",
        "admission_enqueue", "engine_core_ipc",
        "first_token_delivery", "output_delivery_per_output",
    }, "Host request path services")
    measurement = _mapping(payload["measurement"], "Host request path measurement")
    _exact_fields(measurement, {
        "boundary", "time_unit", "statistic", "artifacts",
    }, "Host request path measurement")
    if measurement["boundary"] != "host_only_no_device_no_model":
        raise ValueError("Host request path measurement boundary is invalid")
    if measurement["time_unit"] != "seconds":
        raise ValueError("Host request path measurement time unit is invalid")
    if measurement["statistic"] != "median_of_independent_primitive_samples":
        raise ValueError("Host request path measurement statistic is invalid")
    artifacts = measurement["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Host request path artifacts must be non-empty")
    repository_root = Path(__file__).resolve().parents[2]
    for artifact in artifacts:
        item = _mapping(artifact, "Host request path artifact")
        _exact_fields(item, {"path", "sha256"}, "Host request path artifact")
        artifact_path = Path(_string(item["path"], "Host request path artifact"))
        if not artifact_path.is_absolute():
            artifact_path = (repository_root / artifact_path).resolve()
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != _sha256(
                item["sha256"], "Host request path artifact SHA256"):
            raise ValueError(
                f"Host request path artifact SHA256 mismatch: {artifact_path}")

    def _optional_seconds(value: object, label: str) -> float | None:
        if value is None:
            return None
        return _positive_seconds(value, label)

    return HostRequestPathProfile(
        hardware_id=_string(payload["hardware_id"], "hardware_id"),
        runtime_version=_string(runtime["version"], "runtime version"),
        shared_cpu_timing_target=(payload["shared_cpu_timing_target"]
            if "shared_cpu_timing_target" in payload else None),
        source_identity=_string(
            runtime["source_identity"], "runtime source identity"),
        host_cpu=_string(host["cpu"], "Host CPU"),
        api_entry_s=_positive_seconds(service["api_entry"], "API entry time"),
        tokenizer_async_call_s=_optional_seconds(
            service["tokenizer_async_call"], "tokenizer async call time"),
        tokenizer_render_region_intercept_s=_optional_seconds(
            service["tokenizer_render_region_intercept"],
            "tokenizer render region fixed cost"),
        tokenizer_render_region_per_token_s=_optional_seconds(
            service["tokenizer_render_region_per_token"],
            "tokenizer render region per-token cost"),
        tokenizer_encode_intercept_s=_optional_seconds(
            service["tokenizer_encode_intercept"], "tokenizer encode intercept"),
        tokenizer_encode_per_token_s=_optional_seconds(
            service["tokenizer_encode_per_token"], "tokenizer per-token time"),
        input_processor_intercept_s=_positive_seconds(
            service["input_processor_intercept"],
            "input processor fixed call time"),
        input_processor_per_token_s=_positive_seconds(
            service["input_processor_per_token"],
            "input processor per-token time"),
        admission_enqueue_s=_positive_seconds(
            service["admission_enqueue"],
            "request registration and enqueue time"),
        engine_core_ipc_s=_positive_seconds(
            service["engine_core_ipc"],
            "EngineCore IPC region time"),
        first_token_delivery_s=_positive_seconds(
            service["first_token_delivery"], "first token delivery time"),
        output_delivery_per_output_s=_positive_seconds(
            service["output_delivery_per_output"],
            "marginal output delivery time"),
        profile_path=str(source_path),
        profile_sha256=hashlib.sha256(raw).hexdigest(),
    )
