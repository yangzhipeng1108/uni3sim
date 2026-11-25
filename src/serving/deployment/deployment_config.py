"""User-facing worker-pool and route configuration."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from src.kernel.hardware_ids import canonical_hardware_id
from src.serving.kv_cache.kv_cache_config import KVCacheConfig
from src.serving.parallelism.parallel_config import ParallelConfig


_WORKER_ROLE_PHASES = {
    "colocated": "colocated",
    "attention": "colocated",
    "ffn": "colocated",
    "prefill": "prefill",
    "decode": "decode",
    "prefill_attention": "prefill",
    "prefill_ffn": "prefill",
    "decode_attention": "decode",
    "decode_ffn": "decode",
}

_WORKER_ROLE_OPERATOR_GROUPS = {
    "colocated": None,
    "attention": "attention",
    "ffn": "ffn",
    "prefill": None,
    "decode": None,
    "prefill_attention": "attention",
    "prefill_ffn": "ffn",
    "decode_attention": "attention",
    "decode_ffn": "ffn",
}


@dataclass(frozen=True)
class CollectiveConfig:
    """Physical backend selected for one intra-pool collective operation."""

    operation: str
    algorithm: str
    link: str
    topology: str
    resource_id: str
    backend: str
    backend_config_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in (
                "operation", "algorithm", "link", "topology",
                "resource_id", "backend"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"collective {field_name} must be a non-empty string")
        for field_name in (
                "operation", "algorithm", "link", "topology", "backend"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, value.strip().lower())
        object.__setattr__(self, "resource_id", self.resource_id.strip())
        if self.operation in ("all_reduce", "all_gather"):
            if self.operation == "all_reduce":
                collective_layouts = {
                    "fc2": "fully_connected",
                    "mesh": "fully_connected",
                    "ring": "ring",
                    "tree": "tree",
                }
            elif self.operation == "all_gather":
                collective_layouts = {
                    "fc2": "fully_connected",
                    "mesh": "fully_connected",
                    "ring": "ring",
                }
            else:
                raise AssertionError(
                    "validated collective operation disappeared")
            if self.algorithm not in collective_layouts:
                raise ValueError(
                    f"{self.operation} collective has unsupported algorithm "
                    f"{self.algorithm!r}")
            expected_topology = collective_layouts[self.algorithm]
            if self.topology != expected_topology:
                raise ValueError(
                    f"{self.operation} {self.algorithm} collective requires "
                    f"{expected_topology} topology")
        elif self.operation == 'broadcast':
            if (self.algorithm != 'direct' or self.topology != 'fully_connected'
                    or self.backend != 'analytical_roofline_v1' or self.link != 'nvlink'):
                raise ValueError('source PP broadcast requires an explicit analytical direct NVLink fabric')
        elif self.operation in ("all_to_all", "all_gatherv", "reduce_scatterv"):
            if self.algorithm != "pairwise":
                raise ValueError(
                    "all_to_all collective requires the pairwise algorithm")
            if self.topology != "fully_connected":
                raise ValueError(
                    "all_to_all collective requires fully_connected topology")
        else:
            raise ValueError(
                f"unknown collective operation: {self.operation!r}")
        if self.operation in ("all_gatherv", "reduce_scatterv") and (
                self.backend != "analytical_roofline_v1" or self.link == "roce"):
            raise ValueError("variable collectives require an analytical direct pairwise fabric")
        if self.backend not in ("analytical_roofline_v1", "protobridge"):
            raise ValueError(
                "collective backend must be analytical_roofline_v1 or "
                "protobridge")
        if self.backend_config_sha256 is not None:
            digest = self.backend_config_sha256
            if (not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "collective backend_config_sha256 must be a lowercase "
                    "SHA256")
        if self.backend == "analytical_roofline_v1":
            if self.link not in ("nvlink", "metaxlink", "hccs", "infiniband", "roce"):
                raise ValueError(
                    "analytical Roofline collective supports declared NVLink, MetaXLink, HCCS, InfiniBand or RoCE")
            if (self.operation in ("all_reduce", "all_gather")
                    and self.algorithm != "ring"
                    and not (self.link == "roce" and self.algorithm == "mesh")):
                raise ValueError(
                    "analytical Roofline all_reduce/all_gather requires the "
                    "ring algorithm")
            if self.backend_config_sha256 is not None:
                raise ValueError(
                    "analytical Roofline collective does not accept a backend "
                    "config SHA256")
        elif self.backend == "protobridge":
            if self.link not in ("nvlink", "metaxlink", "hccs"):
                raise ValueError(
                    "ProtoBridge collective link must be nvlink, metaxlink, "
                    "or hccs")
            if self.backend_config_sha256 is None:
                raise ValueError(
                    "ProtoBridge collective requires a backend config SHA256")
        else:
            raise ValueError(f"unknown collective backend: {self.backend!r}")


@dataclass(frozen=True)
class PipelineP2PConfig:
    """Physical backend for one ordered intra-pool pipeline boundary."""

    source_stage: int
    destination_stage: int
    algorithm: str
    link: str
    topology: str
    resource_id: str
    backend: str
    backend_config_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        for field_name in ("source_stage", "destination_stage"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"pipeline P2P {field_name} must be an integer")
            if value < 0:
                raise ValueError(
                    f"pipeline P2P {field_name} cannot be negative")
        if self.destination_stage != self.source_stage + 1:
            raise ValueError(
                "pipeline P2P must connect adjacent forward stages")
        for field_name in (
                "algorithm", "link", "topology", "resource_id", "backend"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"pipeline P2P {field_name} must be a non-empty string")
        for field_name in ("algorithm", "link", "topology", "backend"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, value.strip().lower())
        object.__setattr__(self, "resource_id", self.resource_id.strip())
        if self.algorithm != "send_recv":
            raise ValueError("pipeline P2P algorithm must be send_recv")
        if self.topology != "point_to_point":
            raise ValueError(
                "pipeline P2P topology must be point_to_point")
        if self.link not in ("nvlink", "metaxlink", "hccs", "infiniband"):
            raise ValueError(
                "pipeline P2P link must be nvlink, metaxlink, hccs, or "
                "infiniband")
        if self.backend not in ("analytical_roofline_v1", "protobridge"):
            raise ValueError(
                "pipeline P2P backend must be analytical_roofline_v1 or "
                "protobridge")
        if self.backend_config_sha256 is not None:
            digest = self.backend_config_sha256
            if (not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "pipeline P2P backend_config_sha256 must be a lowercase "
                    "SHA256")
        if self.backend == "analytical_roofline_v1":
            if self.link not in ("nvlink", "infiniband"):
                raise ValueError(
                    "analytical Roofline pipeline P2P supports only the "
                    "NVLink or InfiniBand paths")
            if self.backend_config_sha256 is not None:
                raise ValueError(
                    "analytical Roofline pipeline P2P does not accept a "
                    "backend config SHA256")
        elif self.backend == "protobridge":
            if self.link not in ("nvlink", "metaxlink", "hccs"):
                raise ValueError(
                    "ProtoBridge pipeline P2P link must be nvlink, "
                    "metaxlink, or hccs")
            if self.backend_config_sha256 is None:
                raise ValueError(
                    "ProtoBridge pipeline P2P requires a backend config "
                    "SHA256")
        else:
            raise ValueError(
                f"unknown pipeline P2P backend: {self.backend!r}")


def _normalize_worker_role(role: str) -> str:
    if not isinstance(role, str) or not role.strip():
        raise TypeError("worker role must be a non-empty string")
    normalized = role.strip().lower().replace("-", "_")
    if normalized not in _WORKER_ROLE_PHASES:
        raise ValueError(f"unknown worker role: {role!r}")
    return normalized


def worker_role_phase(role: str) -> str:
    """Return the phase of one canonical worker role or reject it."""

    normalized = _normalize_worker_role(role)
    return _WORKER_ROLE_PHASES[normalized]


def worker_role_operator_group(role: str) -> Optional[str]:
    """Return ``attention``/``ffn`` or explicit ``None`` for unsplit roles."""

    normalized = _normalize_worker_role(role)
    return _WORKER_ROLE_OPERATOR_GROUPS[normalized]


def normalize_worker_node_ids(values, worker_count):
    if not isinstance(values, (tuple, list)):
        raise TypeError("worker node_ids must be an explicit list or tuple")
    nodes = tuple(values)
    if nodes and (len(nodes) != worker_count
            or any(type(node) is not str or not node or node != node.strip() for node in nodes)):
        raise ValueError("worker node_ids requires one nonempty node identity per placed GPU")
    return nodes


@dataclass(frozen=True)
class WorkerPoolConfig:
    name: str
    role: str
    hardware_id: str
    parallelism: ParallelConfig = ParallelConfig()
    ranks: Tuple[int, ...] = ()
    kv_cache: Optional[KVCacheConfig] = None
    collectives: Tuple[CollectiveConfig, ...] = ()
    pipeline_p2p: Tuple[PipelineP2PConfig, ...] = ()
    # Explicit logical cache-layout scenario, not an observed GPU provider.
    # None retains the legacy same-hardware logical payload contract only.
    kv_transfer_layout: Optional[str] = None
    pipeline_layer_partition: Optional[Tuple[int, ...]] = None
    # In logical placement order. Empty means unknown, never rank//8 or
    # pool-name inference. Equal IDs share one physical node's TX/RX ports.
    node_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kv_transfer_layout not in (None, "BLHNC", "BLNHC", "LBHNC"):
            raise ValueError("unknown declared KV transfer layout")
        for field in ("name", "role"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"worker pool {field} must be a non-empty string")
        worker_role_phase(self.role)
        object.__setattr__(self, "hardware_id", canonical_hardware_id(self.hardware_id))
        if not isinstance(self.parallelism, ParallelConfig):
            raise TypeError("worker pool parallelism must be a ParallelConfig")
        object.__setattr__(self, "node_ids", normalize_worker_node_ids(
            self.node_ids, self.parallelism.worker_count))
        if self.pipeline_layer_partition is not None:
            partition = self.pipeline_layer_partition
            if (not isinstance(partition, tuple) or len(partition) != self.parallelism.pipeline_parallel_size
                    or any(type(count) is not int or count <= 0 for count in partition)):
                raise ValueError("worker pipeline partition requires one positive integer layer count per stage")
        if self.kv_cache is not None and not isinstance(
                self.kv_cache, KVCacheConfig):
            raise TypeError("worker pool kv_cache must be a KVCacheConfig or null")
        if not isinstance(self.collectives, (list, tuple)):
            raise TypeError("worker pool collectives must be a list or tuple")
        collectives = tuple(self.collectives)
        if any(not isinstance(item, CollectiveConfig) for item in collectives):
            raise TypeError(
                "worker pool collectives must contain CollectiveConfig objects")
        operations = [item.operation for item in collectives]
        if len(operations) != len(set(operations)):
            raise ValueError(
                "worker pool collective operations must be unique")
        object.__setattr__(self, "collectives", collectives)
        if not isinstance(self.pipeline_p2p, (list, tuple)):
            raise TypeError("worker pool pipeline_p2p must be a list or tuple")
        pipeline_p2p = tuple(self.pipeline_p2p)
        if any(not isinstance(item, PipelineP2PConfig)
               for item in pipeline_p2p):
            raise TypeError(
                "worker pool pipeline_p2p must contain PipelineP2PConfig "
                "objects")
        stage_pairs = [
            (item.source_stage, item.destination_stage)
            for item in pipeline_p2p
        ]
        if len(stage_pairs) != len(set(stage_pairs)):
            raise ValueError(
                "worker pool pipeline P2P stage pairs must be unique")
        object.__setattr__(self, "pipeline_p2p", pipeline_p2p)
        if (not isinstance(self.ranks, (list, tuple))
                or any(isinstance(rank, bool) or not isinstance(rank, int)
                       for rank in self.ranks)):
            raise TypeError("worker pool ranks must be integers")
        ranks = tuple(self.ranks)
        if any(rank < 0 for rank in ranks):
            raise ValueError("worker pool ranks cannot be negative")
        if len(ranks) != len(set(ranks)):
            raise ValueError("worker pool ranks must be unique")
        object.__setattr__(self, "ranks", ranks)
        if self.ranks and len(self.ranks) != self.parallelism.worker_count:
            raise ValueError(
                f"pool {self.name} has {len(self.ranks)} ranks but parallel plan "
                f"requires {self.parallelism.worker_count}")


@dataclass(frozen=True)
class RouteConfig:
    source_pool: str
    destination_pool: str
    payload: str
    algorithm: str
    link: str
    topology: str = "point_to_point"
    resource_id: Optional[str] = None
    backend: Optional[str] = None
    backend_config_sha256: Optional[str] = None
    kv_transport_scope: str = "shared_fabric"

    def __post_init__(self) -> None:
        if self.kv_transport_scope not in ("shared_fabric", "tp_rank_paired_links"):
            raise ValueError("unknown KV transport scope")
        for field_name in (
                "source_pool", "destination_pool", "payload", "algorithm",
                "link", "topology"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"route {field_name} must be a non-empty string")
        for field_name in ("payload", "algorithm", "link", "topology"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, value.strip().lower())
        if self.source_pool == self.destination_pool:
            raise ValueError("deployment routes must connect distinct pools")
        for field_name in ("resource_id", "backend"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"route {field_name} must be a non-empty string")
        object.__setattr__(
            self, "resource_id", self.resource_id.strip())
        object.__setattr__(
            self, "backend", self.backend.strip().lower())
        if self.algorithm != "send_recv":
            raise ValueError("route algorithm must be send_recv")
        if self.topology != "point_to_point":
            raise ValueError("route topology must be point_to_point")
        if self.backend not in ("analytical_roofline_v1", "protobridge"):
            raise ValueError(
                "route backend must be analytical_roofline_v1 or protobridge")
        if self.backend_config_sha256 is not None:
            digest = self.backend_config_sha256
            if (not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "route backend_config_sha256 must be a lowercase SHA256")
        if self.backend == "analytical_roofline_v1":
            if self.backend_config_sha256 is not None:
                raise ValueError(
                    "analytical Roofline route does not accept a backend "
                    "config SHA256")
        elif self.backend == "protobridge":
            if self.backend_config_sha256 is None:
                raise ValueError(
                    "ProtoBridge route requires a backend config SHA256")
        else:
            raise ValueError(f"unknown route backend: {self.backend!r}")
        if self.kv_transport_scope == "tp_rank_paired_links" and (
                self.payload != "kv" or self.link != "infiniband"
                or self.backend != "analytical_roofline_v1"):
            raise ValueError("rank-paired KV links require analytical InfiniBand KV transport")


@dataclass(frozen=True)
class RouterConfig:
    """Modeled serial CPU dispatch resource shared by all route stages."""

    resource_id: str = "router_cpu"
    dispatch_service_time_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError("router resource_id must be a non-empty string")
        if (isinstance(self.dispatch_service_time_s, bool)
                or not isinstance(
                    self.dispatch_service_time_s, (int, float))
                or not math.isfinite(self.dispatch_service_time_s)
                or self.dispatch_service_time_s < 0):
            raise ValueError(
                "router dispatch service time must be finite and non-negative")
        object.__setattr__(
            self, "dispatch_service_time_s",
            float(self.dispatch_service_time_s))


@dataclass(frozen=True)
class DeploymentConfig:
    mode: str
    worker_pools: Tuple[WorkerPoolConfig, ...]
    routes: Tuple[RouteConfig, ...] = ()
    router: RouterConfig = RouterConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("deployment mode must be a string")
        if self.mode not in ("colocated", "pd_disagg", "af_disagg", "pd_af_disagg"):
            raise ValueError(f"unknown deployment mode: {self.mode!r}")
        if not isinstance(self.worker_pools, (list, tuple)):
            raise TypeError("deployment worker_pools must be a list or tuple")
        if not isinstance(self.routes, (list, tuple)):
            raise TypeError("deployment routes must be a list or tuple")
        if not isinstance(self.router, RouterConfig):
            raise TypeError("deployment router must be a RouterConfig")
        object.__setattr__(self, "worker_pools", tuple(self.worker_pools))
        object.__setattr__(self, "routes", tuple(self.routes))
        if not self.worker_pools:
            raise ValueError("deployment requires at least one worker pool")
        if any(not isinstance(pool, WorkerPoolConfig)
               for pool in self.worker_pools):
            raise TypeError(
                "deployment worker_pools must be WorkerPoolConfig objects")
        if any(not isinstance(route, RouteConfig) for route in self.routes):
            raise TypeError("deployment routes must be RouteConfig objects")
        if (self.mode == "colocated"
                and len(self.worker_pools) == 1
                and self.router.dispatch_service_time_s != 0.0):
            raise ValueError(
                "router dispatch service requires a routed multi-pool "
                "deployment")
        names = [pool.name for pool in self.worker_pools]
        if len(names) != len(set(names)):
            raise ValueError("worker pool names must be unique")


def collective_ring_order(
    algorithm: str,
    participants: tuple[int, ...],
) -> tuple[int, ...]:
    if algorithm == "ring":
        return participants
    if algorithm in ("fc2", "mesh", "tree"):
        return ()
    raise ValueError(f"unsupported Serving collective algorithm: {algorithm!r}")
