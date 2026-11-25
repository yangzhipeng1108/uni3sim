"""Strict Midlayer contract adapter for the ProtoBridge upstream surrogate."""

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.kernel_spec import KernelKind, KernelPrediction, KernelSpec
from src.kernel.launch_config.communication_profile import (
    development_protobridge_inputs,
    development_protobridge_inputs_for,
)
from src.kernel.launch_config.select_launch_config import (
    select_collective_launch,
)
from src.kernel.predictors.common import prediction_from_ns

from .upstream import load_upstream_module_and_verify_license as _load_upstream

_upstream = _load_upstream()
TheoryDerivedSurrogate = _upstream.TheoryDerivedSurrogate
Topology = _upstream.Topology


_AUTHOR_PROFILE_SOURCE_KIND = "author_profile"
_DEVELOPMENT_PROFILE_SOURCE_KIND = "development_only"
_SYNTHETIC_PROFILE_SOURCE_KIND = "synthetic_test_only"
_TIMELINE_CONTENTION_CONTRACT = "execution_timeline_owns_contention"
_UPSTREAM_OPERATION_CONTRACTS = MappingProxyType({
    (
        "all_reduce",
        "ring",
        "ring",
        "per_rank_input",
        "sum",
    ): ("allreduce", "ring"),
    (
        "all_gather",
        "ring",
        "ring",
        "per_rank_input",
        None,
    ): ("allgather", "ring"),
    (
        "all_reduce",
        "fullmesh",
        "fully_connected",
        "per_rank_input",
        "sum",
    ): ("allreduce", "fullmesh"),
    (
        "all_gather",
        "fullmesh",
        "fully_connected",
        "per_rank_input",
        None,
    ): ("allgather", "fullmesh"),
    (
        "all_reduce",
        "fc2",
        "fully_connected",
        "per_rank_input",
        "sum",
    ): ("allreduce", "fullmesh"),
    (
        "all_gather",
        "fc2",
        "fully_connected",
        "per_rank_input",
        None,
    ): ("allgather", "fullmesh"),
    (
        "all_reduce",
        "mesh",
        "fully_connected",
        "per_rank_input",
        "sum",
    ): ("allreduce", "fullmesh"),
    (
        "all_gather",
        "mesh",
        "fully_connected",
        "per_rank_input",
        None,
    ): ("allgather", "fullmesh"),
    (
        "all_reduce",
        "tree",
        "tree",
        "per_rank_input",
        "sum",
    ): ("allreduce", "tree"),
    # Midlayer's pairwise AllToAll is ProtoBridge's N-1 phase ring schedule.
    (
        "all_to_all",
        "pairwise",
        "fully_connected",
        "per_rank_input",
        None,
    ): ("alltoall", "ring"),
    # ProtoBridge uses ring bandwidth parameters for its one-hop Send/Recv.
    (
        "send_recv",
        "send_recv",
        "point_to_point",
        "payload",
        None,
    ): ("sendrecv", "ring"),
})
_SURROGATE_PARAMETER_NAMES = (
    "packet_bw_bytes_per_us",
    "ring_bw_bytes_per_us",
    "tree_bw_per_level_bytes_per_us",
    "nvls_bw_bytes_per_us",
    "fullmesh_bw_bytes_per_us",
    "link_latency_us",
    "credit_bdp_ring_packets",
    "credit_bdp_tree_packets",
    "spray_chunk_bytes",
    "bulk_chunk_bytes",
    "num_lanes",
    "t_min_switch_us",
    "fec_n",
    "fec_k",
    "fec_t",
    "fec_symbol_bits",
    "fec_encode_ns",
    "fec_decode_ns",
    "startup_us",
    "nvls_startup_us",
    "retry_source_bw_bytes_per_us",
    "retry_source_latency_us",
    "wire_efficiency",
)
_SURROGATE_INTEGRAL_PARAMETERS = frozenset({
    "spray_chunk_bytes",
    "bulk_chunk_bytes",
    "num_lanes",
    "fec_n",
    "fec_k",
    "fec_t",
    "fec_symbol_bits",
})
_SURROGATE_STRICTLY_POSITIVE_PARAMETERS = frozenset({
    "packet_bw_bytes_per_us",
    "ring_bw_bytes_per_us",
    "tree_bw_per_level_bytes_per_us",
    "nvls_bw_bytes_per_us",
    "fullmesh_bw_bytes_per_us",
    "spray_chunk_bytes",
    "bulk_chunk_bytes",
    "num_lanes",
    "fec_n",
    "fec_k",
    "fec_symbol_bits",
    "retry_source_bw_bytes_per_us",
    "wire_efficiency",
})


@dataclass(frozen=True)
class ProtoBridgeTopology:
    """Immutable Midlayer-owned topology descriptor for one profile."""

    bisection_bw_bytes_per_us: float
    hop_count: float

    def __post_init__(self) -> None:
        for field_name in ("bisection_bw_bytes_per_us", "hop_count"):
            value = getattr(self, field_name)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0):
                raise ValueError(
                    f"ProtoBridge topology {field_name} must be finite and "
                    "positive")
            object.__setattr__(self, field_name, float(value))


@dataclass(frozen=True)
class ProtoBridgeProtocolPolicy:
    """Internal NCCL protocol timing selected from a collective's byte size.

    Callers provide only the physical collective and its byte count.  The
    adapter selects the vendor protocol and both of its fixed costs here:
    one launch/startup cost and one cost for every ring phase.
    """

    ll_upper_exclusive_bytes: int
    ll128_upper_inclusive_bytes: int
    small_startup_us: float
    ll128_startup_us: float
    simple_startup_us: float
    small_phase_fixed_us: float
    ll128_phase_fixed_us: float
    simple_phase_fixed_us: float
    simple_allreduce_ring_bw_by_participant: Tuple[Tuple[int, float], ...] = ()
    large_message_min_bytes: Optional[int] = None
    large_message_wire_efficiency: Optional[float] = None

    def __post_init__(self) -> None:
        for field_name in (
                "ll_upper_exclusive_bytes", "ll128_upper_inclusive_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"ProtoBridge protocol {field_name} must be an integer")
            if value <= 0:
                raise ValueError(
                    f"ProtoBridge protocol {field_name} must be positive")
        if self.ll_upper_exclusive_bytes >= self.ll128_upper_inclusive_bytes:
            raise ValueError(
                "ProtoBridge protocol LL boundary must be below the LL128 "
                "boundary")
        for field_name in (
                "small_startup_us", "ll128_startup_us", "simple_startup_us",
                "small_phase_fixed_us", "ll128_phase_fixed_us",
                "simple_phase_fixed_us"):
            value = getattr(self, field_name)
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0):
                raise ValueError(
                    f"ProtoBridge protocol {field_name} must be finite and "
                    "non-negative")
            object.__setattr__(self, field_name, float(value))
        seen_participants = set()
        normalized_bandwidths = []
        for entry in self.simple_allreduce_ring_bw_by_participant:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "ProtoBridge simple AllReduce bandwidth entries must be "
                    "(participant_count, bytes_per_us) tuples")
            participant_count, bandwidth = entry
            if (isinstance(participant_count, bool)
                    or not isinstance(participant_count, int)
                    or participant_count < 2):
                raise ValueError(
                    "ProtoBridge simple AllReduce bandwidth participant "
                    "count must be an integer of at least two")
            if participant_count in seen_participants:
                raise ValueError(
                    "ProtoBridge simple AllReduce bandwidth has duplicate "
                    f"participant count {participant_count}")
            if (isinstance(bandwidth, bool)
                    or not isinstance(bandwidth, (int, float))
                    or not math.isfinite(bandwidth)
                    or bandwidth <= 0):
                raise ValueError(
                    "ProtoBridge simple AllReduce bandwidth must be finite "
                    "and positive")
            seen_participants.add(participant_count)
            normalized_bandwidths.append((participant_count, float(bandwidth)))
        object.__setattr__(
            self,
            "simple_allreduce_ring_bw_by_participant",
            tuple(sorted(normalized_bandwidths)),
        )
        minimum = self.large_message_min_bytes
        efficiency = self.large_message_wire_efficiency
        if (minimum is None) != (efficiency is None):
            raise ValueError("large-message threshold and efficiency must be paired")
        if minimum is not None:
            if (isinstance(minimum, bool) or not isinstance(minimum, int)
                    or minimum <= 0):
                raise ValueError("large-message threshold must be positive integer")
            if (isinstance(efficiency, bool)
                    or not isinstance(efficiency, (int, float))
                    or not math.isfinite(efficiency)
                    or not 0 < efficiency <= 1):
                raise ValueError("large-message wire efficiency must be in (0, 1]")
            object.__setattr__(self, "large_message_wire_efficiency", float(efficiency))

    def select(self, message_bytes: int) -> Tuple[str, float]:
        if isinstance(message_bytes, bool) or not isinstance(message_bytes, int):
            raise TypeError("ProtoBridge protocol message bytes must be integer")
        if message_bytes <= 0:
            raise ValueError("ProtoBridge protocol message bytes must be positive")
        if message_bytes < self.ll_upper_exclusive_bytes:
            return "ll", self.small_startup_us
        if message_bytes <= self.ll128_upper_inclusive_bytes:
            return "ll128", self.ll128_startup_us
        return "simple", self.simple_startup_us

    def startup_for_protocol(self, protocol: str) -> float:
        if protocol == "ll":
            return self.small_startup_us
        if protocol == "ll128":
            return self.ll128_startup_us
        if protocol == "simple":
            return self.simple_startup_us
        raise NotImplementedError(
            f"unsupported ProtoBridge vendor protocol: {protocol!r}")

    def phase_fixed_for_protocol(self, protocol: str) -> float:
        if protocol == "ll":
            return self.small_phase_fixed_us
        if protocol == "ll128":
            return self.ll128_phase_fixed_us
        if protocol == "simple":
            return self.simple_phase_fixed_us
        raise NotImplementedError(
            f"unsupported ProtoBridge vendor protocol: {protocol!r}")

    def ring_bandwidth_for(
        self,
        protocol: str,
        operation: str,
        participant_count: int,
        default_bytes_per_us: float,
    ) -> float:
        """Return an internal protocol/rank bandwidth override, if defined."""

        if protocol != "simple" or operation != "all_reduce":
            return default_bytes_per_us
        for count, bandwidth in self.simple_allreduce_ring_bw_by_participant:
            if count == participant_count:
                return bandwidth
        return default_bytes_per_us


def _profile_content_sha256(profile: "ProtoBridgeProfile") -> str:
    if profile.surrogate_topology is None:
        topology_payload = None
    else:
        topology_payload = {
            "bisection_bw_bytes_per_us": (
                profile.surrogate_topology.bisection_bw_bytes_per_us),
            "hop_count": profile.surrogate_topology.hop_count,
        }
    payload = {
        "algorithm": profile.algorithm,
        "bytes_semantics": profile.bytes_semantics,
        "credits": profile.credits,
        "hardware_id": profile.hardware_id,
        "model_parameters": {
            name: profile.model_parameters[name]
            for name in _SURROGATE_PARAMETER_NAMES
        },
        "operation": profile.operation,
        "profile_version": profile.profile_version,
        "provenance": profile.provenance,
        "protocol_policy": {
            "ll128_upper_inclusive_bytes": (
                profile.protocol_policy.ll128_upper_inclusive_bytes),
            "ll_upper_exclusive_bytes": (
                profile.protocol_policy.ll_upper_exclusive_bytes),
            "ll128_phase_fixed_us": (
                profile.protocol_policy.ll128_phase_fixed_us),
            "ll128_startup_us": profile.protocol_policy.ll128_startup_us,
            "simple_phase_fixed_us": (
                profile.protocol_policy.simple_phase_fixed_us),
            "simple_allreduce_ring_bw_by_participant": (
                profile.protocol_policy.simple_allreduce_ring_bw_by_participant),
            "simple_startup_us": profile.protocol_policy.simple_startup_us,
            "small_phase_fixed_us": (
                profile.protocol_policy.small_phase_fixed_us),
            "small_startup_us": profile.protocol_policy.small_startup_us,
        },
        "reduction": profile.reduction,
        "source_kind": profile.source_kind,
        "link": profile.link,
        "surrogate_topology": topology_payload,
        "timeline_contention_contract": (
            profile.timeline_contention_contract),
        "topology": profile.topology,
    }
    if profile.protocol_policy.large_message_min_bytes is not None:
        payload["protocol_policy"]["large_message_min_bytes"] = (
            profile.protocol_policy.large_message_min_bytes)
        payload["protocol_policy"]["large_message_wire_efficiency"] = (
            profile.protocol_policy.large_message_wire_efficiency)
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True)
class ProtoBridgeProfile:
    """Versioned hardware/profile binding outside the surrogate formula."""

    config_sha256: Optional[str]
    profile_version: str
    provenance: str
    hardware_id: str
    source_kind: str
    link: str
    operation: str
    algorithm: str
    topology: str
    bytes_semantics: str
    reduction: Optional[str]
    credits: int
    timeline_contention_contract: str
    model_parameters: Mapping[str, float | int]
    protocol_policy: ProtoBridgeProtocolPolicy
    surrogate_topology: Optional[ProtoBridgeTopology] = None

    def __post_init__(self) -> None:
        supplied_digest = self.config_sha256
        if supplied_digest is not None and (
                not isinstance(supplied_digest, str)
                or len(supplied_digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in supplied_digest)):
            raise ValueError(
                "ProtoBridge profile config_sha256 must be null or a lowercase "
                "SHA256")
        for field_name in ("profile_version", "provenance"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ProtoBridge profile {field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip())
        for field_name in (
                "source_kind", "link", "operation", "algorithm", "topology",
                "bytes_semantics", "timeline_contention_contract"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ProtoBridge profile {field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip().lower())
        object.__setattr__(
            self, "hardware_id", canonical_hardware_id(self.hardware_id))
        if self.reduction is not None:
            if not isinstance(self.reduction, str) or not self.reduction.strip():
                raise ValueError(
                    "ProtoBridge profile reduction must be non-empty or null")
            object.__setattr__(
                self, "reduction", self.reduction.strip().lower())
        if self.source_kind not in (
                _AUTHOR_PROFILE_SOURCE_KIND,
                _DEVELOPMENT_PROFILE_SOURCE_KIND,
                _SYNTHETIC_PROFILE_SOURCE_KIND):
            raise ValueError(
                "ProtoBridge profile source_kind must be author_profile, "
                "development_only, or synthetic_test_only")
        if self.link not in ("nvlink", "metaxlink", "hccs", "infiniband"):
            raise ValueError(
                "ProtoBridge profile link must be nvlink, metaxlink, hccs, "
                "or infiniband")
        contract = (
            self.operation,
            self.algorithm,
            self.topology,
            self.bytes_semantics,
            self.reduction,
        )
        if contract not in _UPSTREAM_OPERATION_CONTRACTS:
            raise NotImplementedError(
                "unsupported ProtoBridge operation contract: "
                f"{contract!r}")
        if (isinstance(self.credits, bool)
                or not isinstance(self.credits, int)
                or self.credits <= 0):
            raise ValueError(
                "ProtoBridge profile credits must be a positive integer")
        if self.timeline_contention_contract != _TIMELINE_CONTENTION_CONTRACT:
            raise NotImplementedError(
                "ProtoBridge profile must confirm that ExecutionTimeline owns "
                "endpoint and fabric contention")
        if not isinstance(self.model_parameters, Mapping):
            raise TypeError("ProtoBridge model_parameters must be a mapping")
        supplied_parameters = set(self.model_parameters)
        required_parameters = set(_SURROGATE_PARAMETER_NAMES)
        if supplied_parameters != required_parameters:
            raise ValueError(
                "ProtoBridge model_parameters must provide the complete "
                "surrogate constructor contract: "
                f"missing={sorted(required_parameters - supplied_parameters)}, "
                f"extra={sorted(supplied_parameters - required_parameters)}")
        normalized_parameters = {}
        for name in _SURROGATE_PARAMETER_NAMES:
            value = self.model_parameters[name]
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise TypeError(
                    f"ProtoBridge model parameter {name} must be finite numeric")
            if name in _SURROGATE_INTEGRAL_PARAMETERS and not isinstance(
                    value, int):
                raise TypeError(
                    f"ProtoBridge model parameter {name} must be an integer")
            if name in _SURROGATE_STRICTLY_POSITIVE_PARAMETERS and value <= 0:
                raise ValueError(
                    f"ProtoBridge model parameter {name} must be positive")
            if name not in _SURROGATE_STRICTLY_POSITIVE_PARAMETERS and value < 0:
                raise ValueError(
                    f"ProtoBridge model parameter {name} cannot be negative")
            if name == "wire_efficiency" and value > 1:
                raise ValueError(
                    "ProtoBridge model parameter wire_efficiency cannot "
                    "exceed one")
            if name in _SURROGATE_INTEGRAL_PARAMETERS:
                normalized_parameters[name] = value
            else:
                normalized_parameters[name] = float(value)
        object.__setattr__(
            self,
            "model_parameters",
            MappingProxyType(normalized_parameters),
        )
        if (self.surrogate_topology is not None
                and not isinstance(
                    self.surrogate_topology, ProtoBridgeTopology)):
            raise TypeError(
                "ProtoBridge surrogate_topology must be an immutable "
                "ProtoBridgeTopology or null")
        if not isinstance(self.protocol_policy, ProtoBridgeProtocolPolicy):
            raise TypeError(
                "ProtoBridge protocol_policy must be an immutable "
                "ProtoBridgeProtocolPolicy")
        expected_digest = _profile_content_sha256(self)
        if supplied_digest is None:
            object.__setattr__(self, "config_sha256", expected_digest)
        elif supplied_digest != expected_digest:
            raise ValueError(
                "ProtoBridge profile config_sha256 does not match normalized "
                f"profile content: expected {expected_digest}, got "
                f"{supplied_digest}")


def _development_model_parameters(
    *,
    per_link_bytes_per_us: float,
    num_lanes: int,
    link_latency_us: float,
    small_startup_us: float,
    spray_chunk_bytes: int,
    t_min_switch_us: float,
    wire_efficiency: float,
) -> Mapping[str, float | int]:
    aggregate = per_link_bytes_per_us * num_lanes
    return {
        "packet_bw_bytes_per_us": 0.40 * aggregate,
        "ring_bw_bytes_per_us": 0.40 * aggregate,
        "tree_bw_per_level_bytes_per_us": 1.05 * per_link_bytes_per_us,
        "nvls_bw_bytes_per_us": 1.20 * aggregate,
        # FullMesh already sends to all peers concurrently. This parameter is
        # therefore one dedicated peer path, not the sum of all peer links.
        "fullmesh_bw_bytes_per_us": per_link_bytes_per_us,
        "link_latency_us": link_latency_us,
        "credit_bdp_ring_packets": 91.5,
        "credit_bdp_tree_packets": 35.5,
        "spray_chunk_bytes": spray_chunk_bytes,
        "bulk_chunk_bytes": 8 * 1024 * 1024,
        "num_lanes": num_lanes,
        "t_min_switch_us": t_min_switch_us,
        "fec_n": 544,
        "fec_k": 514,
        "fec_t": 15,
        "fec_symbol_bits": 10,
        "fec_encode_ns": 50.0,
        "fec_decode_ns": 80.0,
        "startup_us": small_startup_us,
        "nvls_startup_us": small_startup_us,
        "retry_source_bw_bytes_per_us": 4_800_000.0,
        "retry_source_latency_us": 0.3,
        "wire_efficiency": wire_efficiency,
    }


def _development_profile(
    hardware_id: str,
    operation: str,
    algorithm: str,
    link: str,
    per_link_bytes_per_us: float,
    num_lanes: int,
    link_latency_us: float,
    small_startup_us: float,
    ll128_startup_us: float,
    simple_startup_us: float,
    small_phase_fixed_us: float,
    ll128_phase_fixed_us: float,
    simple_phase_fixed_us: float,
    simple_allreduce_ring_bw_by_participant: Tuple[Tuple[int, float], ...],
    spray_chunk_bytes: int,
    t_min_switch_us: float,
    wire_efficiency: float,
    large_message_min_bytes: Optional[int],
    large_message_wire_efficiency: Optional[float],
    provenance_note: str,
    selector_provenance: str,
    surrogate_topology: Optional[ProtoBridgeTopology],
) -> ProtoBridgeProfile:
    if operation == "all_reduce":
        reduction = "sum"
        bytes_semantics = "per_rank_input"
    elif operation == "all_gather":
        reduction = None
        bytes_semantics = "per_rank_input"
    elif operation == "send_recv":
        reduction = None
        bytes_semantics = "payload"
    else:
        raise ValueError(
            f"unsupported development communication operation: {operation!r}")
    if algorithm == "ring":
        topology = "ring"
    elif algorithm == "tree":
        topology = "tree"
    elif algorithm in ("fc2", "mesh"):
        topology = "fully_connected"
    elif algorithm == "send_recv":
        topology = "point_to_point"
    else:
        raise ValueError(
            f"unsupported development collective algorithm: {algorithm!r}")
    return ProtoBridgeProfile(
        config_sha256=None,
        profile_version="midlayer-protobridge-development-v4",
        provenance=(
            provenance_note
            + "development-only profile from user-confirmed wire/topology/"
            "startup inputs; FullMesh uses one per-peer wire path, while "
            "eta_ring=0.40 and remaining H200-derived mechanism parameters "
            "are placeholders requiring later source-backed update; "
            f"vendor selection is bound to {selector_provenance}"),
        hardware_id=hardware_id,
        source_kind=_DEVELOPMENT_PROFILE_SOURCE_KIND,
        link=link,
        operation=operation,
        algorithm=algorithm,
        topology=topology,
        bytes_semantics=bytes_semantics,
        reduction=reduction,
        credits=64,
        timeline_contention_contract=_TIMELINE_CONTENTION_CONTRACT,
        model_parameters=_development_model_parameters(
            per_link_bytes_per_us=per_link_bytes_per_us,
            num_lanes=num_lanes,
            link_latency_us=link_latency_us,
            small_startup_us=small_startup_us,
            spray_chunk_bytes=spray_chunk_bytes,
            t_min_switch_us=t_min_switch_us,
            wire_efficiency=wire_efficiency,
        ),
        protocol_policy=ProtoBridgeProtocolPolicy(
            64 * 1024,
            4 * 1024 * 1024,
            small_startup_us,
            ll128_startup_us,
            simple_startup_us,
            small_phase_fixed_us,
            ll128_phase_fixed_us,
            simple_phase_fixed_us,
            simple_allreduce_ring_bw_by_participant,
            large_message_min_bytes,
            large_message_wire_efficiency,
        ),
        surrogate_topology=surrogate_topology,
    )


def _profile_from_hardware_inputs(
    hardware: Mapping[str, object],
    operation: str,
    algorithm: str,
) -> ProtoBridgeProfile:
    if ("large_message_min_bytes" in hardware) != (
            "large_message_wire_efficiency" in hardware):
        raise ValueError("large-message hardware inputs must be paired")
    if operation == "all_reduce" and "large_message_min_bytes" in hardware:
        large_message_min_bytes = hardware["large_message_min_bytes"]
        large_message_wire_efficiency = hardware["large_message_wire_efficiency"]
    else:
        large_message_min_bytes = None
        large_message_wire_efficiency = None
    bisection = hardware["bisection_bw_bytes_per_us"]
    if bisection is None:
        topology = None
    else:
        topology = ProtoBridgeTopology(
            float(bisection), float(hardware["hop_count"]))
    return _development_profile(
        hardware_id=str(hardware["hardware_id"]),
        operation=operation,
        algorithm=algorithm,
        link=str(hardware["link"]),
        per_link_bytes_per_us=float(hardware["per_link_bytes_per_us"]),
        num_lanes=int(hardware["num_lanes"]),
        link_latency_us=float(hardware["link_latency_us"]),
        small_startup_us=float(hardware["small_startup_us"]),
        ll128_startup_us=float(hardware["ll128_startup_us"]),
        simple_startup_us=float(hardware["simple_startup_us"]),
        small_phase_fixed_us=float(hardware["small_phase_fixed_us"]),
        ll128_phase_fixed_us=float(hardware["ll128_phase_fixed_us"]),
        simple_phase_fixed_us=float(hardware["simple_phase_fixed_us"]),
        simple_allreduce_ring_bw_by_participant=tuple(
            hardware["simple_allreduce_ring_bw_by_participant"]),
        spray_chunk_bytes=int(hardware["spray_chunk_bytes"]),
        t_min_switch_us=float(hardware["t_min_switch_us"]),
        wire_efficiency=float(hardware["wire_efficiency"]),
        large_message_min_bytes=large_message_min_bytes,
        large_message_wire_efficiency=large_message_wire_efficiency,
        provenance_note=str(hardware["provenance_note"]),
        selector_provenance=(
            f"{hardware['communication_library']} "
            f"{hardware['library_version']} / "
            f"{hardware['selector_topology']} / "
            f"{hardware['selector_rule_version']}"
        ),
        surrogate_topology=topology,
    )


_REGISTERED_PROFILES: Tuple[ProtoBridgeProfile, ...] = tuple(
    _profile_from_hardware_inputs(hardware, operation, algorithm)
    for hardware in development_protobridge_inputs()
    for operation, algorithms in (
        ("all_reduce", hardware["all_reduce_algorithms"]),
        ("all_gather", hardware["all_gather_algorithms"]),
        ("send_recv", ("send_recv",)),
    )
    for algorithm in algorithms
)


def resolve_registered_profile(config_sha256: str) -> ProtoBridgeProfile:
    matches = tuple(
        profile
        for profile in _REGISTERED_PROFILES
        if profile.config_sha256 == config_sha256
    )
    if not matches:
        raise NotImplementedError(
            "unsupported ProtoBridge backend config: no registered hardware "
            f"profile exists for SHA256 {config_sha256}")
    if len(matches) != 1:
        raise RuntimeError(
            "ProtoBridge profile registry contains duplicate SHA256 "
            f"{config_sha256}")
    profile = matches[0]
    if profile.source_kind not in (
            _AUTHOR_PROFILE_SOURCE_KIND, _DEVELOPMENT_PROFILE_SOURCE_KIND):
        raise RuntimeError(
            "ProtoBridge registry contains a non-runnable profile")
    return profile


def development_profile_config_sha256(
    hardware_id: str,
    operation: str,
    algorithm: Optional[str] = None,
    link: Optional[str] = None,
) -> str:
    """Return the unique development profile digest for one contract."""

    normalized_hardware_id = canonical_hardware_id(hardware_id)
    if operation not in ("all_reduce", "all_gather", "send_recv"):
        raise ValueError(
            f"unsupported development communication operation: {operation!r}")
    if link is None:
        normalized_link = None
    elif isinstance(link, str) and link.strip():
        normalized_link = link.strip().lower()
    else:
        raise ValueError(
            "development ProtoBridge link must be non-empty or null")
    candidates = tuple(
        profile for profile in _REGISTERED_PROFILES
        if profile.hardware_id == normalized_hardware_id
        and profile.operation == operation
        and profile.source_kind == _DEVELOPMENT_PROFILE_SOURCE_KIND
        and (normalized_link is None or profile.link == normalized_link)
    )
    if not candidates:
        raise NotImplementedError(
            "no development ProtoBridge profile for "
            f"{normalized_hardware_id}/{operation}/link={normalized_link}")
    if algorithm is None:
        candidate_algorithms = {profile.algorithm for profile in candidates}
        if len(candidate_algorithms) == 1:
            normalized_algorithm = next(iter(candidate_algorithms))
        else:
            hardware = development_protobridge_inputs_for(
                normalized_hardware_id, link=normalized_link)
            if operation == "all_reduce":
                default_algorithm = hardware["default_all_reduce_algorithm"]
            elif operation == "all_gather":
                default_algorithm = hardware["default_all_gather_algorithm"]
            else:
                default_algorithm = "send_recv"
            normalized_algorithm = str(default_algorithm)
    elif isinstance(algorithm, str) and algorithm.strip():
        normalized_algorithm = algorithm.strip().lower()
    else:
        raise ValueError(
            "development ProtoBridge algorithm must be non-empty or null")
    matches = tuple(
        profile for profile in candidates
        if profile.algorithm == normalized_algorithm
    )
    if len(matches) != 1:
        raise NotImplementedError(
            "no unique development ProtoBridge profile for "
            f"{normalized_hardware_id}/{operation}/{normalized_algorithm}/"
            f"link={normalized_link}")
    digest = matches[0].config_sha256
    if digest is None:
        raise AssertionError("registered ProtoBridge profile lacks digest")
    return digest


def is_development_profile_sha256(config_sha256: str) -> bool:
    return any(
        profile.config_sha256 == config_sha256
        and profile.source_kind == _DEVELOPMENT_PROFILE_SOURCE_KIND
        for profile in _REGISTERED_PROFILES
    )


def _selected_vendor_protocol(
    transfer,
    profile: ProtoBridgeProfile,
) -> str:
    if (profile.source_kind == _SYNTHETIC_PROFILE_SOURCE_KIND
            or transfer.operation not in ("all_reduce", "all_gather")):
        protocol, _startup_us = profile.protocol_policy.select(transfer.bytes)
        return protocol
    hardware = development_protobridge_inputs_for(
        profile.hardware_id, link=profile.link)
    if transfer.operation == "all_reduce":
        in_place = True
    elif transfer.operation == "all_gather":
        in_place = False
    else:
        raise AssertionError("validated collective operation disappeared")
    selection = select_collective_launch(
        profile.hardware_id,
        communication_library=str(hardware["communication_library"]),
        library_version=str(hardware["library_version"]),
        operation=transfer.operation,
        message_bytes=transfer.bytes,
        participant_count=len(transfer.participants),
        dtype="bfloat16",
        reduction=transfer.reduction,
        in_place=in_place,
        topology=str(hardware["selector_topology"]),
    )
    if selection.rule_version != hardware["selector_rule_version"]:
        raise RuntimeError(
            "ProtoBridge profile selector provenance changed: "
            f"expected={hardware['selector_rule_version']!r}, "
            f"selected={selection.rule_version!r}")
    if selection.vendor_algorithm != profile.algorithm:
        raise NotImplementedError(
            "ProtoBridge profile does not match the selected vendor "
            f"algorithm: selected={selection.vendor_algorithm!r}, "
            f"profile={profile.algorithm!r}")
    if selection.protocol_status == "selected":
        if selection.vendor_protocol is None:
            raise AssertionError("selected vendor protocol disappeared")
        return selection.vendor_protocol
    if selection.protocol_status == "unobservable":
        protocol, _startup_us = profile.protocol_policy.select(transfer.bytes)
        return protocol
    if selection.protocol_status == "unresolved":
        raise NotImplementedError(
            "ProtoBridge cannot choose startup for an unresolved vendor "
            f"protocol at message_bytes={transfer.bytes}")
    raise AssertionError("validated vendor protocol status disappeared")


def _predict_with_profile(
    kernel: KernelSpec,
    hardware,
    profile: ProtoBridgeProfile,
) -> KernelPrediction:
    if not isinstance(kernel, KernelSpec):
        raise TypeError(
            "ProtoBridge adapter requires a canonical KernelSpec")
    if kernel.kind is not KernelKind.COMMUNICATION:
        raise ValueError(
            "ProtoBridge adapter accepts only COMMUNICATION kernels")
    transfer = kernel.transfer
    if transfer is None:
        raise ValueError(
            "ProtoBridge COMMUNICATION kernel lacks transfer configuration")
    if transfer.backend != "protobridge":
        raise ValueError(
            "ProtoBridge adapter requires backend='protobridge'")
    if transfer.backend_config_sha256 != profile.config_sha256:
        raise ValueError(
            "ProtoBridge transfer backend_config_sha256 does not match the "
            "resolved profile")
    hardware_id = canonical_hardware_id(hardware.hardware_id)
    if kernel.hardware_id != hardware_id:
        raise ValueError(
            "ProtoBridge KernelSpec hardware_id does not match HWSpec")
    if profile.hardware_id != hardware_id:
        raise NotImplementedError(
            "unsupported ProtoBridge hardware/profile pairing: "
            f"profile={profile.hardware_id}, requested={hardware_id}")
    if transfer.link != profile.link:
        raise NotImplementedError(
            "unsupported ProtoBridge hardware/link pairing: "
            f"profile={profile.link}, requested={transfer.link}")
    transfer_contract = (
        transfer.operation,
        transfer.algorithm,
        transfer.topology,
        transfer.bytes_semantics,
        transfer.reduction,
    )
    profile_contract = (
        profile.operation,
        profile.algorithm,
        profile.topology,
        profile.bytes_semantics,
        profile.reduction,
    )
    if transfer_contract != profile_contract:
        raise NotImplementedError(
            "unsupported ProtoBridge operation contract: "
            f"transfer={transfer_contract!r}, profile={profile_contract!r}")
    from src.kernel.memory_communication import has_communication_source
    if has_communication_source(kernel):
        from dataclasses import replace
        from src.kernel.memory_access_builder import communication_memory_access_program
        from src.kernel.memory_path_solver import collective_source_work
        payload, participants, operation = collective_source_work(communication_memory_access_program(kernel, hardware))
        if operation != transfer.operation or participants != len(transfer.participants):
            raise ValueError("ProtoBridge source collective contract does not match its call")
        transfer = replace(transfer, bytes=payload)
    upstream_collective, upstream_algorithm = (
        _UPSTREAM_OPERATION_CONTRACTS[transfer_contract])
    if transfer.operation == "all_gather":
        # Midlayer records one rank's input shard. ProtoBridge's D is the
        # complete gathered tensor, whose per-rank traffic is D*(N-1)/N.
        upstream_data_bytes = (
            transfer.bytes * len(transfer.participants))
    elif transfer.operation in (
            "all_reduce", "all_to_all", "send_recv"):
        # For these operations ProtoBridge's D has the same semantics as the
        # canonical per-rank input or point-to-point payload.
        upstream_data_bytes = transfer.bytes
    else:
        raise ValueError(
            f"unhandled ProtoBridge operation: {transfer.operation!r}")
    protocol = _selected_vendor_protocol(transfer, profile)
    model_parameters = dict(profile.model_parameters)
    if (profile.protocol_policy.large_message_min_bytes is not None
            and transfer.operation == "all_reduce"
            and len(transfer.participants) == 2
            and transfer.bytes >= profile.protocol_policy.large_message_min_bytes):
        model_parameters["wire_efficiency"] = (
            profile.protocol_policy.large_message_wire_efficiency)
    # The upstream surrogate exposes one fixed per-phase latency.  Selecting
    # it here keeps that vendor detail inside the adapter: Serving and kernel
    # callers still provide only a physical collective.
    model_parameters["link_latency_us"] = (
        profile.protocol_policy.phase_fixed_for_protocol(protocol))
    model_parameters["ring_bw_bytes_per_us"] = (
        profile.protocol_policy.ring_bandwidth_for(
            protocol,
            transfer.operation,
            len(transfer.participants),
            float(model_parameters["ring_bw_bytes_per_us"]),
        ))
    surrogate = TheoryDerivedSurrogate(**model_parameters)
    # ExecutionTimeline reserves every endpoint and the shared fabric for the
    # complete call interval. The profile contract therefore requires zero
    # competing load here so ProtoBridge does not price that contention twice.
    if profile.surrogate_topology is None:
        upstream_topology = None
    else:
        upstream_topology = Topology(
            bisection_bw_bytes_per_us=(
                profile.surrogate_topology.bisection_bw_bytes_per_us),
            hop_count=profile.surrogate_topology.hop_count,
        )
    startup_us = profile.protocol_policy.startup_for_protocol(protocol)
    latency_us = surrogate.predict(
        data_bytes=upstream_data_bytes,
        num_gpus=len(transfer.participants),
        algo=upstream_algorithm,
        credits=profile.credits,
        ber=0.0,
        mem_bw=0.0,
        startup_us=startup_us,
        fec_enabled=False,
        llr_enabled=False,
        competing_bw=0.0,
        fec_n=None,
        fec_k=None,
        fec_t=None,
        retry_mode="gobackn",
        retry_buffer_entries=None,
        retry_buffer_bytes=None,
        optical_fraction=1.0,
        optical_hops=1.0,
        strict_reliability=False,
        retry_limit=None,
        include_credit=True,
        include_fec=True,
        include_retransmission=True,
        include_contention=True,
        topology=upstream_topology,
        collective=upstream_collective,
        phase="full",
    )
    if (isinstance(latency_us, bool)
            or not isinstance(latency_us, (int, float))
            or not math.isfinite(latency_us)
            or latency_us < 0):
        raise ValueError(
            "ProtoBridge surrogate returned a non-finite or negative latency")
    return prediction_from_ns(float(latency_us) * 1000.0, "communication")


def predict_protobridge(
    kernel: KernelSpec,
    hardware,
) -> KernelPrediction:
    """Predict with an explicitly registered profile or fail loud."""

    transfer = kernel.transfer
    if transfer is None or transfer.backend_config_sha256 is None:
        raise ValueError(
            "ProtoBridge production prediction requires backend config SHA256")
    profile = resolve_registered_profile(transfer.backend_config_sha256)
    return _predict_with_profile(kernel, hardware, profile)


def predict_synthetic_for_testing(
    kernel: KernelSpec,
    hardware,
    profile: ProtoBridgeProfile,
) -> KernelPrediction:
    """Exercise interface mapping with an explicitly non-production profile."""

    if profile.source_kind != _SYNTHETIC_PROFILE_SOURCE_KIND:
        raise ValueError(
            "synthetic ProtoBridge test entry requires synthetic_test_only")
    return _predict_with_profile(kernel, hardware, profile)
