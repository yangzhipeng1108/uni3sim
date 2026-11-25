"""Host DRAM access contract and independently parameterized backend."""

from dataclasses import dataclass
from enum import Enum
import math
from math import gcd
from numbers import Real
from typing import Protocol, runtime_checkable


class HostDramDirection(str, Enum):
    READ = "read"
    WRITE = "write"
    MIXED = "mixed"


class HostDramAccessPattern(str, Enum):
    STREAMING = "streaming"


class HostDramPath(str, Enum):
    LOCAL = "local"
    CROSS_SOCKET = "cross_socket"


class HostDramModelKind(str, Enum):
    ANALYTICAL = "analytical"


def _require_integer(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Host DRAM {name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"Host DRAM {name} must be positive")
    if not positive and value < 0:
        raise ValueError(f"Host DRAM {name} cannot be negative")
    return value


def _require_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Host DRAM {name} must be a non-empty string")
    return value


def _require_time(name: str, value: object) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(
            f"Host DRAM prediction {name} must be finite and non-negative")
    return float(value)


@dataclass(frozen=True)
class HostDramParameterProvenance:
    """Evidence required for one mechanism parameter."""

    source: str
    measurement_method: str
    physical_meaning: str
    applicability: str

    def __post_init__(self) -> None:
        for name in (
                "source", "measurement_method", "physical_meaning",
                "applicability"):
            _require_non_empty_string(name, getattr(self, name))


@dataclass(frozen=True)
class HostDramMechanismParameter:
    """One dimensional mechanism parameter with complete provenance."""

    name: str
    value: float
    unit: str
    provenance: HostDramParameterProvenance

    def __post_init__(self) -> None:
        _require_non_empty_string("parameter name", self.name)
        _require_non_empty_string("parameter unit", self.unit)
        if (isinstance(self.value, bool) or not isinstance(self.value, Real)
                or not math.isfinite(self.value) or self.value <= 0):
            raise ValueError(
                "Host DRAM parameter value must be finite and positive")
        object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.provenance, HostDramParameterProvenance):
            raise TypeError(
                "Host DRAM parameter provenance must be canonical")


@dataclass(frozen=True)
class HostDramAccess:
    """One isolated Host DRAM access whose service demand is predicted.

    ``bytes`` is total controller traffic for this access.
    ``concurrency`` selects an independently measured operating point; the
    analytical backend never interpolates between concurrency levels.

    This contract deliberately contains no historical queue or address-timing
    state. Queueing and resource contention belong to ``ExecutionTimeline``.
    A future Ramulator adapter that needs ordered address events must introduce
    an explicit extension rather than infer them from this aggregate access.
    """

    hardware_id: str
    direction: HostDramDirection
    bytes: int
    read_bytes: int
    write_bytes: int
    access_pattern: HostDramAccessPattern
    initiator_numa_node: int
    memory_numa_node: int
    initiator_socket_id: int
    memory_socket_id: int
    path: HostDramPath
    concurrency: int
    controller_resource_id: str

    def __post_init__(self) -> None:
        _require_non_empty_string("hardware_id", self.hardware_id)
        object.__setattr__(
            self, "direction", HostDramDirection(self.direction))
        object.__setattr__(
            self, "access_pattern",
            HostDramAccessPattern(self.access_pattern))
        object.__setattr__(self, "path", HostDramPath(self.path))
        _require_integer("bytes", self.bytes, positive=True)
        _require_integer("read_bytes", self.read_bytes, positive=False)
        _require_integer("write_bytes", self.write_bytes, positive=False)
        for name in (
                "initiator_numa_node", "memory_numa_node",
                "initiator_socket_id", "memory_socket_id"):
            _require_integer(name, getattr(self, name), positive=False)
        _require_integer("concurrency", self.concurrency, positive=True)
        _require_non_empty_string(
            "controller_resource_id", self.controller_resource_id)
        if self.read_bytes + self.write_bytes != self.bytes:
            raise ValueError(
                "Host DRAM read/write bytes must sum to total bytes")
        if self.direction is HostDramDirection.READ:
            valid_direction = (
                self.read_bytes == self.bytes and self.write_bytes == 0)
        elif self.direction is HostDramDirection.WRITE:
            valid_direction = (
                self.read_bytes == 0 and self.write_bytes == self.bytes)
        elif self.direction is HostDramDirection.MIXED:
            valid_direction = self.read_bytes > 0 and self.write_bytes > 0
        else:
            raise RuntimeError(
                f"unknown Host DRAM direction: {self.direction.value!r}")
        if not valid_direction:
            raise ValueError(
                "Host DRAM direction disagrees with read/write bytes")
        same_socket = self.initiator_socket_id == self.memory_socket_id
        if self.path is HostDramPath.LOCAL and not same_socket:
            raise ValueError(
                "local Host DRAM path requires the same socket")
        if self.path is HostDramPath.CROSS_SOCKET and same_socket:
            raise ValueError(
                "cross-socket Host DRAM path requires distinct sockets")
        if (self.initiator_numa_node == self.memory_numa_node
                and not same_socket):
            raise ValueError(
                "one Host DRAM NUMA node cannot map to multiple sockets")


@dataclass(frozen=True)
class HostDramCapability:
    """One exact topology, direction, mix, and concurrency operating point."""

    hardware_id: str
    direction: HostDramDirection
    access_pattern: HostDramAccessPattern
    initiator_numa_node: int
    memory_numa_node: int
    initiator_socket_id: int
    memory_socket_id: int
    path: HostDramPath
    concurrency: int
    controller_resource_id: str
    read_ratio_numerator: int
    read_ratio_denominator: int
    min_bytes: int
    max_bytes: int

    def __post_init__(self) -> None:
        _require_non_empty_string("hardware_id", self.hardware_id)
        object.__setattr__(
            self, "direction", HostDramDirection(self.direction))
        object.__setattr__(
            self, "access_pattern",
            HostDramAccessPattern(self.access_pattern))
        object.__setattr__(self, "path", HostDramPath(self.path))
        for name in (
                "initiator_numa_node", "memory_numa_node",
                "initiator_socket_id", "memory_socket_id",
                "read_ratio_numerator"):
            _require_integer(name, getattr(self, name), positive=False)
        for name in (
                "concurrency", "read_ratio_denominator", "min_bytes",
                "max_bytes"):
            _require_integer(name, getattr(self, name), positive=True)
        _require_non_empty_string(
            "controller_resource_id", self.controller_resource_id)
        same_socket = self.initiator_socket_id == self.memory_socket_id
        if self.path is HostDramPath.LOCAL and not same_socket:
            raise ValueError(
                "local Host DRAM capability requires the same socket")
        if self.path is HostDramPath.CROSS_SOCKET and same_socket:
            raise ValueError(
                "cross-socket Host DRAM capability requires distinct "
                "sockets")
        if (self.initiator_numa_node == self.memory_numa_node
                and not same_socket):
            raise ValueError(
                "one Host DRAM NUMA node cannot map to multiple sockets")
        if self.read_ratio_numerator > self.read_ratio_denominator:
            raise ValueError("Host DRAM read ratio cannot exceed one")
        if gcd(
                self.read_ratio_numerator,
                self.read_ratio_denominator) != 1:
            raise ValueError("Host DRAM read ratio must be reduced")
        if self.direction is HostDramDirection.READ:
            valid_ratio = (
                self.read_ratio_numerator == 1
                and self.read_ratio_denominator == 1)
        elif self.direction is HostDramDirection.WRITE:
            valid_ratio = self.read_ratio_numerator == 0
        elif self.direction is HostDramDirection.MIXED:
            valid_ratio = (
                0 < self.read_ratio_numerator
                < self.read_ratio_denominator)
        else:
            raise RuntimeError(
                f"unknown Host DRAM direction: {self.direction.value!r}")
        if not valid_ratio:
            raise ValueError(
                "Host DRAM direction disagrees with read ratio")
        if self.min_bytes > self.max_bytes:
            raise ValueError(
                "Host DRAM min_bytes cannot exceed max_bytes")

    def supports(self, access: HostDramAccess) -> bool:
        if not isinstance(access, HostDramAccess):
            raise TypeError(
                "Host DRAM capability requires a HostDramAccess")
        return (
            access.hardware_id == self.hardware_id
            and access.direction is self.direction
            and access.access_pattern is self.access_pattern
            and access.initiator_numa_node == self.initiator_numa_node
            and access.memory_numa_node == self.memory_numa_node
            and access.initiator_socket_id == self.initiator_socket_id
            and access.memory_socket_id == self.memory_socket_id
            and access.path is self.path
            and access.concurrency == self.concurrency
            and access.controller_resource_id == self.controller_resource_id
            and self.min_bytes <= access.bytes <= self.max_bytes
            and access.read_bytes * self.read_ratio_denominator
            == access.bytes * self.read_ratio_numerator
        )


@dataclass(frozen=True)
class HostDramOperatingPoint:
    capability: HostDramCapability
    fixed_access_latency: HostDramMechanismParameter
    streaming_bandwidth: HostDramMechanismParameter

    def __post_init__(self) -> None:
        if not isinstance(self.capability, HostDramCapability):
            raise TypeError(
                "Host DRAM operating point capability must be canonical")
        if not isinstance(
                self.fixed_access_latency, HostDramMechanismParameter):
            raise TypeError(
                "Host DRAM fixed access latency must be a canonical "
                "parameter")
        if not isinstance(
                self.streaming_bandwidth, HostDramMechanismParameter):
            raise TypeError(
                "Host DRAM streaming bandwidth must be a canonical "
                "parameter")
        if (self.fixed_access_latency.name != "fixed_access_latency"
                or self.fixed_access_latency.unit != "ns"):
            raise ValueError(
                "Host DRAM fixed access latency requires name "
                "'fixed_access_latency' and unit 'ns'")
        if (self.streaming_bandwidth.name != "streaming_bandwidth"
                or self.streaming_bandwidth.unit != "bytes/s"):
            raise ValueError(
                "Host DRAM streaming bandwidth requires name "
                "'streaming_bandwidth' and unit 'bytes/s'")


@dataclass(frozen=True)
class HostDramPredictionProvenance:
    backend_name: str
    backend_version: str
    model_kind: HostDramModelKind
    source_reference: str
    parameters: tuple[HostDramMechanismParameter, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("backend_name", "backend_version", "source_reference"):
            _require_non_empty_string(name, getattr(self, name))
        object.__setattr__(
            self, "model_kind", HostDramModelKind(self.model_kind))
        if not isinstance(self.parameters, (list, tuple)):
            raise TypeError(
                "Host DRAM prediction parameters must be a list or tuple")
        parameters = tuple(self.parameters)
        if any(
                not isinstance(item, HostDramMechanismParameter)
                for item in parameters):
            raise TypeError(
                "Host DRAM prediction parameters must be canonical")
        names = tuple(item.name for item in parameters)
        if len(names) != len(set(names)):
            raise ValueError(
                "Host DRAM prediction parameter names must be unique")
        if (self.model_kind is HostDramModelKind.ANALYTICAL
                and set(names) != {
                    "fixed_access_latency", "streaming_bandwidth"}):
            raise ValueError(
                "analytical Host DRAM provenance requires fixed access "
                "latency and streaming bandwidth parameters")
        object.__setattr__(self, "parameters", parameters)
        if not isinstance(self.limitations, (list, tuple)):
            raise TypeError(
                "Host DRAM prediction limitations must be a list or tuple")
        limitations = tuple(self.limitations)
        if any(
                not isinstance(item, str) or not item
                for item in limitations):
            raise ValueError(
                "Host DRAM prediction limitations must be non-empty strings")
        object.__setattr__(self, "limitations", limitations)


@dataclass(frozen=True)
class HostDramPrediction:
    """Isolated-access service demand with controller provenance."""

    access: HostDramAccess
    t_fixed_access_s: float
    t_streaming_s: float
    t_queue_s: float
    t_total_s: float
    controller_resource_id: str
    provenance: HostDramPredictionProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.access, HostDramAccess):
            raise TypeError("Host DRAM prediction access must be canonical")
        for name in (
                "t_fixed_access_s", "t_streaming_s", "t_queue_s",
                "t_total_s"):
            object.__setattr__(
                self, name, _require_time(name, getattr(self, name)))
        _require_non_empty_string(
            "prediction controller_resource_id",
            self.controller_resource_id)
        if self.controller_resource_id != self.access.controller_resource_id:
            raise ValueError(
                "Host DRAM prediction controller disagrees with its access")
        expected_total = (
            self.t_fixed_access_s
            + self.t_streaming_s
            + self.t_queue_s)
        if self.t_queue_s != 0.0:
            raise ValueError(
                "isolated Host DRAM prediction cannot include queue time")
        if not math.isclose(
                self.t_total_s, expected_total,
                rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                "Host DRAM prediction total disagrees with its breakdown")
        if not isinstance(self.provenance, HostDramPredictionProvenance):
            raise TypeError(
                "Host DRAM prediction provenance must be canonical")


@runtime_checkable
class HostDramIsolatedAccessBackend(Protocol):
    """Backend boundary for isolated Host DRAM service demand.

    This is not a stable Ramulator timing interface. A future simulator
    adapter must propose explicit ordered-address and timing inputs while
    preserving this isolated analytical path.
    """

    @property
    def backend_name(self) -> str: ...

    @property
    def capabilities(self) -> tuple[HostDramCapability, ...]: ...

    def predict(self, access: HostDramAccess) -> HostDramPrediction: ...


class HostDramAnalyticalBackend:
    """Isolated latency-plus-bandwidth model over exact mechanism points.

    For selected fixed latency ``L`` and bandwidth ``B``, the prediction is
    ``t_service = L + bytes / B``. Concurrency, direction, read/write mix,
    topology, and access pattern select ``L`` and ``B`` exactly; they are
    never interpolated or fit from serving latency. Queueing is intentionally
    excluded and must be scheduled once by ``ExecutionTimeline``.
    """

    def __init__(
        self,
        operating_points: tuple[HostDramOperatingPoint, ...],
    ) -> None:
        if not isinstance(operating_points, (list, tuple)):
            raise TypeError(
                "Host DRAM operating points must be a list or tuple")
        points = tuple(operating_points)
        if (not points
                or any(
                    not isinstance(item, HostDramOperatingPoint)
                    for item in points)):
            raise ValueError(
                "Host DRAM analytical backend requires canonical operating "
                "points")
        capabilities = tuple(item.capability for item in points)
        numa_socket_map: dict[tuple[str, int], int] = {}
        for index, capability in enumerate(capabilities):
            for numa_node, socket_id in (
                    (capability.initiator_numa_node,
                     capability.initiator_socket_id),
                    (capability.memory_numa_node,
                     capability.memory_socket_id)):
                key = (capability.hardware_id, numa_node)
                if (key in numa_socket_map
                        and numa_socket_map[key] != socket_id):
                    raise ValueError(
                        "one Host DRAM NUMA node cannot map to multiple "
                        "sockets across capabilities")
                numa_socket_map[key] = socket_id
            for previous in capabilities[:index]:
                if self._capabilities_overlap(previous, capability):
                    raise ValueError(
                        "Host DRAM analytical operating points cannot "
                        "overlap")
        self._operating_points = points
        self._numa_socket_map = numa_socket_map

    @property
    def backend_name(self) -> str:
        return "host_dram_analytical_v1"

    @property
    def capabilities(self) -> tuple[HostDramCapability, ...]:
        return tuple(item.capability for item in self._operating_points)

    def predict(self, access: HostDramAccess) -> HostDramPrediction:
        if not isinstance(access, HostDramAccess):
            raise TypeError(
                "Host DRAM analytical backend requires a HostDramAccess")
        for numa_node, socket_id in (
                (access.initiator_numa_node, access.initiator_socket_id),
                (access.memory_numa_node, access.memory_socket_id)):
            key = (access.hardware_id, numa_node)
            if (key in self._numa_socket_map
                    and self._numa_socket_map[key] != socket_id):
                raise ValueError(
                    "Host DRAM access maps a known NUMA node to a "
                    "different socket")
        matches = tuple(
            point for point in self._operating_points
            if point.capability.supports(access)
        )
        if not matches:
            raise NotImplementedError(
                "no exact Host DRAM operating point supports the access")
        if len(matches) != 1:
            raise RuntimeError(
                "multiple Host DRAM operating points support one access")
        point = matches[0]
        t_fixed_access_s = point.fixed_access_latency.value * 1e-9
        bandwidth_bytes_s = point.streaming_bandwidth.value
        t_streaming_s = access.bytes / bandwidth_bytes_s
        t_queue_s = 0.0
        t_total_s = t_fixed_access_s + t_streaming_s + t_queue_s
        provenance = HostDramPredictionProvenance(
            backend_name=self.backend_name,
            backend_version="1",
            model_kind=HostDramModelKind.ANALYTICAL,
            source_reference=(
                "src.serving.kv_cache.host_dram."
                "HostDramAnalyticalBackend"),
            parameters=(
                point.fixed_access_latency,
                point.streaming_bandwidth,
            ),
            limitations=(
                "isolated_access_service_demand_only",
                "streaming_access_only",
                "exact_operating_point_no_interpolation",
                "execution_timeline_owns_queueing_and_contention",
                "no_address_or_channel_timing_contract",
            ),
        )
        return HostDramPrediction(
            access=access,
            t_fixed_access_s=t_fixed_access_s,
            t_streaming_s=t_streaming_s,
            t_queue_s=t_queue_s,
            t_total_s=t_total_s,
            controller_resource_id=access.controller_resource_id,
            provenance=provenance,
        )

    @staticmethod
    def _capabilities_overlap(
        left: HostDramCapability,
        right: HostDramCapability,
    ) -> bool:
        exact_left = (
            left.hardware_id,
            left.direction,
            left.access_pattern,
            left.initiator_numa_node,
            left.memory_numa_node,
            left.initiator_socket_id,
            left.memory_socket_id,
            left.path,
            left.concurrency,
            left.controller_resource_id,
            left.read_ratio_numerator,
            left.read_ratio_denominator,
        )
        exact_right = (
            right.hardware_id,
            right.direction,
            right.access_pattern,
            right.initiator_numa_node,
            right.memory_numa_node,
            right.initiator_socket_id,
            right.memory_socket_id,
            right.path,
            right.concurrency,
            right.controller_resource_id,
            right.read_ratio_numerator,
            right.read_ratio_denominator,
        )
        byte_ranges_overlap = (
            max(left.min_bytes, right.min_bytes)
            <= min(left.max_bytes, right.max_bytes)
        )
        return (
            exact_left == exact_right
            and byte_ranges_overlap
        )
