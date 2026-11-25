"""Normalized deployment plan with one ParallelPlan per pool."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Tuple

from src.serving.parallelism.parallel_plan import ParallelPlan, build_parallel_plan

from .deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    RouterConfig,
    normalize_worker_node_ids,
)


@dataclass(frozen=True)
class WorkerPool:
    name: str
    role: str
    hardware_id: str
    ranks: Tuple[int, ...]
    parallel_plan: ParallelPlan
    kv_cache: object = None
    collectives: Tuple[CollectiveConfig, ...] = ()
    pipeline_p2p: Tuple[PipelineP2PConfig, ...] = ()
    kv_transfer_layout: str | None = None
    node_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", normalize_worker_node_ids(self.node_ids, len(self.ranks)))
        if self.kv_transfer_layout not in (None, "BLHNC", "BLNHC", "LBHNC"):
            raise ValueError("unknown declared KV transfer layout")
        pipeline_size = self.parallel_plan.config.pipeline_parallel_size
        for config in self.pipeline_p2p:
            if config.destination_stage >= pipeline_size:
                raise ValueError(
                    "pipeline P2P stage pair is outside the configured "
                    f"pipeline: {(config.source_stage, config.destination_stage)!r} "
                    f"vs PP={pipeline_size}")

    def physical_rank_for_logical_rank(self, logical_rank: int) -> int:
        """Resolve one plan-local rank through the pool's ordered placement."""

        if isinstance(logical_rank, bool) or not isinstance(logical_rank, int):
            raise TypeError("logical rank must be an integer")
        world_size = self.parallel_plan.rank_groups.world_size
        if not 0 <= logical_rank < world_size:
            raise ValueError(
                f"logical rank {logical_rank} is outside world size "
                f"{world_size}")
        if len(self.ranks) != world_size:
            raise RuntimeError(
                f"pool {self.name!r} placement has {len(self.ranks)} physical "
                f"ranks for logical world size {world_size}")
        return self.ranks[logical_rank]

    def logical_rank_for_physical_rank(self, physical_rank: int) -> int:
        if isinstance(physical_rank, bool) or not isinstance(physical_rank, int):
            raise TypeError("physical rank must be an integer")
        if physical_rank not in self.ranks:
            raise ValueError(
                f"physical rank {physical_rank} is not assigned to pool "
                f"{self.name!r}")
        return self.ranks.index(physical_rank)

    def node_id_for_physical_rank(self, physical_rank: int) -> str:
        if not self.node_ids:
            raise ValueError("physical node placement is unknown; explicit node_ids required")
        return self.node_ids[self.logical_rank_for_physical_rank(physical_rank)]

    def physical_ranks_for_pipeline_stage(
        self,
        stage_index: int,
    ) -> Tuple[int, ...]:
        """Resolve all logical ranks at one PP stage to physical ranks."""

        return tuple(
            self.physical_rank_for_logical_rank(logical_rank)
            for logical_rank in (
                self.parallel_plan.logical_ranks_for_pipeline_stage(
                    stage_index)
            )
        )

    def physical_rank_group(
        self,
        dimension: str,
        logical_rank: int,
    ) -> Tuple[int, ...]:
        """Resolve one ordered logical parallel group to physical ranks."""

        logical_group = self.parallel_plan.rank_groups.for_rank(
            dimension, logical_rank)
        return tuple(
            self.physical_rank_for_logical_rank(group_rank)
            for group_rank in logical_group
        )

    def collective_for(self, operation: str) -> CollectiveConfig:
        if not isinstance(operation, str) or not operation.strip():
            raise TypeError("collective operation must be a non-empty string")
        normalized = operation.strip().lower()
        matches = tuple(
            collective for collective in self.collectives
            if collective.operation == normalized
        )
        if len(matches) != 1:
            raise ValueError(
                f"pool {self.name!r} requires exactly one {normalized!r} "
                f"collective config, found {len(matches)}")
        return matches[0]

    def pipeline_p2p_for(
        self,
        source_stage: int,
        destination_stage: int,
    ) -> PipelineP2PConfig:
        for field_name, value in (
                ("source_stage", source_stage),
                ("destination_stage", destination_stage)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"pipeline P2P {field_name} must be an integer")
        matches = tuple(
            config for config in self.pipeline_p2p
            if config.source_stage == source_stage
            and config.destination_stage == destination_stage
        )
        if len(matches) != 1:
            raise ValueError(
                f"pool {self.name!r} requires exactly one pipeline P2P "
                f"config for {(source_stage, destination_stage)!r}, found "
                f"{len(matches)}")
        return matches[0]


@dataclass(frozen=True)
class PDKVTransferRankPair:
    """One physical P-to-D edge and its head/layer interval ownership."""

    source_logical_rank: int
    destination_logical_rank: int
    source_rank: int
    destination_rank: int
    kv_head_start: int
    kv_head_end: int
    layer_start: int
    layer_end: int


def pd_kv_transfer_rank_pairs(
    source: WorkerPool,
    destination: WorkerPool,
    num_key_value_heads: int,
    *, source_data_parallel_rank: int | None = None, destination_data_parallel_rank: int | None = None,
    attention_layout: str = "standard_gqa",
    kv_transport_scope: str = "shared_fabric",
) -> Tuple[PDKVTransferRankPair, ...]:
    """Intersect cache ownership, selecting each physical source copy once.

    Ordinary GQA shards cache heads over TP. MLA/DSA instead replicate one
    compressed cache head on every TP/PCP rank, independently of their
    query-head or expert partitions. PCP's post-gather cache is complete.
    """

    if not isinstance(source, WorkerPool) or not isinstance(
            destination, WorkerPool):
        raise TypeError("PD KV rank mapping requires WorkerPool endpoints")
    source_parallelism = source.parallel_plan.config
    destination_parallelism = destination.parallel_plan.config
    if attention_layout not in ("standard_gqa", "mla", "dsa_mla"):
        raise ValueError("PD KV rank mapping requires an explicit supported attention layout")
    if source.parallel_plan.num_layers != destination.parallel_plan.num_layers:
        raise ValueError("PD endpoints must describe the same decoder layers")
    for endpoint, parallelism in (
            (source, source_parallelism),
            (destination, destination_parallelism)):
        if parallelism.context_parallel_size != 1 and attention_layout == "standard_gqa":
            raise NotImplementedError("vLLM PCP cache ownership is defined only for MLA")
        if len(endpoint.ranks) != parallelism.worker_count:
            raise RuntimeError(
                f"PD KV transfer endpoint {endpoint.name!r} placement "
                "disagrees with its parallel plan")
    if isinstance(num_key_value_heads, bool) or not isinstance(num_key_value_heads, int):
        raise TypeError("PD KV transfer num_key_value_heads must be an integer")
    if num_key_value_heads < 1:
        raise ValueError("PD KV transfer num_key_value_heads must be positive")

    selected = []
    for config, rank in ((source_parallelism, source_data_parallel_rank),
            (destination_parallelism, destination_data_parallel_rank)):
        if rank is None:
            if config.data_parallel_size != 1:
                raise ValueError("PD transfer must select the actual request's DP replica explicitly")
            rank = 0
        if type(rank) is not int or not 0 <= rank < config.data_parallel_size:
            raise ValueError("PD request DP replica is outside the endpoint placement")
        selected.append(rank)

    def head_range(lane, tp):
        if attention_layout in ("mla", "dsa_mla"):
            return 0, 1
        if tp <= num_key_value_heads:
            if num_key_value_heads % tp:
                raise ValueError("num_key_value_heads must be evenly divisible by TP")
            width = num_key_value_heads // tp
            return lane * width, (lane + 1) * width
        if tp % num_key_value_heads:
            raise ValueError("replicated KV heads require TP divisible by num_key_value_heads")
        head = lane // (tp // num_key_value_heads)
        return head, head + 1

    def owners(endpoint, replica):
        config = endpoint.parallel_plan.config
        tp = config.tensor_parallel_size
        width = config.worker_count // config.data_parallel_size
        base = replica * width
        for logical in range(base, base + width):
            physical = endpoint.ranks[logical]
            context_stage, lane = divmod(logical - base, tp)
            stage = context_stage // config.context_parallel_size
            layers = tuple(i for i, owner in enumerate(
                endpoint.parallel_plan.layer_to_pipeline_stage) if owner == stage)
            if not layers or layers != tuple(range(layers[0], layers[-1] + 1)):
                raise ValueError("PD requires contiguous nonempty PP layer ownership")
            yield logical, physical, head_range(lane, tp), (layers[0], layers[-1] + 1)

    mappings = []
    # First enumerate disjoint PP/head intervals. Below, equal-TP compressed
    # caches select a matching source replica for each destination copy.
    source_owners = []
    seen = set()
    all_source_owners = tuple(owners(source, selected[0]))
    for owner in all_source_owners:
        identity = owner[2:]
        if identity not in seen:
            seen.add(identity)
            source_owners.append(owner)
    for source_logical, source_rank, source_heads, source_layers in source_owners:
        for destination_logical, destination_rank, destination_heads, destination_layers in owners(destination, selected[1]):
            head_start = max(source_heads[0], destination_heads[0])
            head_end = min(source_heads[1], destination_heads[1])
            layer_start = max(source_layers[0], destination_layers[0])
            layer_end = min(source_layers[1], destination_layers[1])
            if head_start < head_end and layer_start < layer_end:
                mappings.append(PDKVTransferRankPair(
                    source_logical, destination_logical, source_rank, destination_rank,
                    head_start, head_end, layer_start, layer_end))
    if not mappings:
        raise RuntimeError("PD KV transfer produced no ownership edges")
    if (attention_layout in ("mla", "dsa_mla")
            and source_parallelism.tensor_parallel_size == destination_parallelism.tensor_parallel_size):
        # Complete compressed caches exist on every TP rank. Select the
        # matching local replica, independently of whether the NIC fabric
        # is shared or per-rank. Replication does not imply a P0 broadcast.
        tp = source_parallelism.tensor_parallel_size
        paired = []
        for pair in mappings:
            candidates = tuple(owner for owner in all_source_owners
                if owner[0] % tp == pair.destination_logical_rank % tp
                and owner[2][0] <= pair.kv_head_start < pair.kv_head_end <= owner[2][1]
                and owner[3][0] <= pair.layer_start < pair.layer_end <= owner[3][1])
            if not candidates:
                raise ValueError("matching TP replica does not own the complete KV interval")
            logical, physical, _, _ = candidates[0]
            paired.append(replace(pair, source_logical_rank=logical, source_rank=physical))
        mappings = paired
    if kv_transport_scope == "tp_rank_paired_links":
        if (attention_layout not in ("mla", "dsa_mla")
                or source_parallelism.tensor_parallel_size != destination_parallelism.tensor_parallel_size
                or any(config.pipeline_parallel_size != 1 or config.data_parallel_size != 1
                    or config.context_parallel_size != 1
                    for config in (source_parallelism, destination_parallelism))
                or len(mappings) != destination_parallelism.tensor_parallel_size):
            raise NotImplementedError("rank-paired KV links require equal-TP PP1/DP1/CP1 complete MLA replicas")
        # Select each matching complete source replica without changing the
        # destination's payload or its head/layer ownership interval.
        mappings = [replace(pair, source_logical_rank=pair.destination_logical_rank,
            source_rank=source.physical_rank_for_logical_rank(pair.destination_logical_rank))
            for pair in mappings]
    elif kv_transport_scope != "shared_fabric":
        raise ValueError("unknown KV transport scope")
    return tuple(mappings)


@dataclass(frozen=True)
class TransferRoute:
    source_pool: str
    destination_pool: str
    payload: str
    algorithm: str
    link: str
    topology: str
    resource_id: str
    backend: str
    backend_config_sha256: str | None
    kv_transport_scope: str = "shared_fabric"

    def physical_resource_id(self, tp_local_rank: int) -> str:
        if type(tp_local_rank) is not int or tp_local_rank < 0:
            raise ValueError("link mapping requires a nonnegative TP local rank")
        if self.kv_transport_scope == "shared_fabric":
            return self.resource_id
        if self.kv_transport_scope == "tp_rank_paired_links":
            return f"{self.resource_id}:tp-rank-{tp_local_rank}"
        raise ValueError("unknown KV transport scope")


@dataclass(frozen=True)
class DeploymentPlan:
    deployment_id: str
    mode: str
    worker_pools: Mapping[str, WorkerPool]
    routes: Tuple[TransferRoute, ...]
    router: RouterConfig = RouterConfig()
    # A DP request executes on one member pool, not on all replica ranks.
    # The original pool identity and ordered replicas preserve the full
    # requested DP placement while each Scheduler uses a local TP/PP plan.
    data_parallel_groups: Mapping[str, Tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.deployment_id, str) or not self.deployment_id:
            raise ValueError("deployment_id must be a non-empty string")
        if not isinstance(self.router, RouterConfig):
            raise TypeError("deployment plan router must be a RouterConfig")
        object.__setattr__(self, "worker_pools", MappingProxyType(dict(self.worker_pools)))
        groups = ({name: (name,) for name in self.worker_pools}
            if self.data_parallel_groups is None else self.data_parallel_groups)
        if not isinstance(groups, Mapping):
            raise TypeError("data_parallel_groups must be a mapping")
        normalized_groups = {}
        members = []
        for group, replicas in groups.items():
            if not isinstance(group, str) or not group:
                raise ValueError("DP group identity must be non-empty")
            if not isinstance(replicas, (list, tuple)) or not replicas:
                raise ValueError("DP group requires an ordered nonempty replica list")
            if any(not isinstance(name, str) or name not in self.worker_pools for name in replicas):
                raise ValueError("DP group references an unknown replica pool")
            normalized_groups[group] = tuple(replicas)
            members.extend(replicas)
            if len(replicas) > 1:
                pools = tuple(self.worker_pools[name] for name in replicas)
                first = pools[0]
                if any(pool.hardware_id != first.hardware_id or pool.role != first.role
                        or pool.kv_transfer_layout != first.kv_transfer_layout
                        or pool.parallel_plan.config != first.parallel_plan.config for pool in pools):
                    raise ValueError("DP replicas must preserve their parent's execution configuration")
                if first.parallel_plan.config.enable_expert_parallel:
                    raise ValueError("independent DP pools cannot replace cross-DP expert ownership")
        if len(members) != len(set(members)) or set(members) != set(self.worker_pools):
            raise ValueError("DP groups must cover every worker pool exactly once")
        object.__setattr__(self, "data_parallel_groups", MappingProxyType(normalized_groups))
        if (self.mode == "colocated"
                and len(self.worker_pools) == 1
                and self.router.dispatch_service_time_s != 0.0):
            raise ValueError(
                "router dispatch service requires a routed multi-pool "
                "deployment")
        for route in self.routes:
            if route.source_pool not in self.worker_pools:
                raise ValueError(f"unknown source pool {route.source_pool!r}")
            if route.destination_pool not in self.worker_pools:
                raise ValueError(f"unknown destination pool {route.destination_pool!r}")
            if route.kv_transport_scope not in ("shared_fabric", "tp_rank_paired_links"):
                raise ValueError("unknown KV transport scope")
            if route.kv_transport_scope == "tp_rank_paired_links":
                if self.mode != "pd_disagg":
                    raise NotImplementedError("rank-paired KV links require plain PD deployment")
                if route.payload != "kv" or route.link != "infiniband" or route.backend != "analytical_roofline_v1":
                    raise ValueError("rank-paired KV links require analytical InfiniBand KV transport")
                source = self.worker_pools[route.source_pool].parallel_plan.config
                destination = self.worker_pools[route.destination_pool].parallel_plan.config
                if (source.tensor_parallel_size != destination.tensor_parallel_size
                        or any(config.pipeline_parallel_size != 1 or config.data_parallel_size != 1
                            or config.context_parallel_size != 1 for config in (source, destination))):
                    raise NotImplementedError("rank-paired links require equal-TP PP1/DP1/CP1 pools")
        collective_resource_contracts = {}
        physical_resource_contracts = {}
        operation_profile_contracts = {}
        for pool in self.worker_pools.values():
            for collective in pool.collectives:
                contract = (
                    pool.hardware_id,
                    collective.link,
                    collective.topology,
                    collective.backend,
                )
                if collective.resource_id in collective_resource_contracts:
                    if (collective_resource_contracts[collective.resource_id]
                            != contract):
                        raise ValueError(
                            "collective resource "
                            f"{collective.resource_id!r} has inconsistent "
                            "hardware/link/backend contracts")
                else:
                    collective_resource_contracts[
                        collective.resource_id] = contract
                physical_contract = (
                    (pool.hardware_id,),
                    collective.link,
                    collective.backend,
                )
                if collective.resource_id in physical_resource_contracts:
                    if (physical_resource_contracts[collective.resource_id]
                            != physical_contract):
                        raise ValueError(
                            "communication resource "
                            f"{collective.resource_id!r} has inconsistent "
                            "cross-category physical contracts")
                else:
                    physical_resource_contracts[
                        collective.resource_id] = physical_contract
                profile_key = (
                    collective.resource_id, collective.operation)
                if profile_key in operation_profile_contracts:
                    if (operation_profile_contracts[profile_key]
                            != collective.backend_config_sha256):
                        raise ValueError(
                            "communication resource operation has "
                            "inconsistent backend profiles: "
                            f"{profile_key!r}")
                else:
                    operation_profile_contracts[profile_key] = (
                        collective.backend_config_sha256)
            for pipeline_p2p in pool.pipeline_p2p:
                physical_contract = (
                    (pool.hardware_id,),
                    pipeline_p2p.link,
                    pipeline_p2p.backend,
                )
                if pipeline_p2p.resource_id in physical_resource_contracts:
                    if (physical_resource_contracts[
                            pipeline_p2p.resource_id] != physical_contract):
                        raise ValueError(
                            "communication resource "
                            f"{pipeline_p2p.resource_id!r} has inconsistent "
                            "cross-category physical contracts")
                else:
                    physical_resource_contracts[
                        pipeline_p2p.resource_id] = physical_contract
                profile_key = (pipeline_p2p.resource_id, "send_recv")
                if profile_key in operation_profile_contracts:
                    if (operation_profile_contracts[profile_key]
                            != pipeline_p2p.backend_config_sha256):
                        raise ValueError(
                            "communication resource operation has "
                            "inconsistent backend profiles: "
                            f"{profile_key!r}")
                else:
                    operation_profile_contracts[profile_key] = (
                        pipeline_p2p.backend_config_sha256)
        resource_contracts = {}
        for route in self.routes:
            source_hardware = self.worker_pools[
                route.source_pool].hardware_id
            destination_hardware = self.worker_pools[
                route.destination_pool].hardware_id
            contract = (
                tuple(sorted((source_hardware, destination_hardware))),
                route.link,
                route.topology,
                route.backend,
                route.backend_config_sha256,
            )
            resources = ((route.resource_id,)
                if route.kv_transport_scope == "shared_fabric" else tuple(
                    route.physical_resource_id(i) for i in range(
                        self.worker_pools[route.source_pool].parallel_plan.config.tensor_parallel_size)))
            for link_resource_id in resources:
                if link_resource_id in resource_contracts:
                    if resource_contracts[link_resource_id] != contract:
                        raise ValueError(
                            f"communication resource {link_resource_id!r} has "
                            "inconsistent hardware/link/backend contracts")
                else:
                    resource_contracts[link_resource_id] = contract
                physical_contract = (
                    tuple(sorted(set((
                        source_hardware,
                        destination_hardware,
                    )))),
                    route.link,
                    route.backend,
                )
                if link_resource_id in physical_resource_contracts:
                    if (physical_resource_contracts[link_resource_id]
                            != physical_contract):
                        raise ValueError(
                            "communication resource "
                            f"{link_resource_id!r} has inconsistent "
                            "cross-category physical contracts")
                else:
                    physical_resource_contracts[
                        link_resource_id] = physical_contract
                profile_key = (link_resource_id, "send_recv")
                if profile_key in operation_profile_contracts:
                    if (operation_profile_contracts[profile_key]
                            != route.backend_config_sha256):
                        raise ValueError(
                            "communication resource operation has inconsistent "
                            f"backend profiles: {profile_key!r}")
                else:
                    operation_profile_contracts[profile_key] = (
                        route.backend_config_sha256)

    def pools_for_role(self, role: str) -> Tuple[WorkerPool, ...]:
        return tuple(pool for pool in self.worker_pools.values() if pool.role == role)


def deployment_plan_manifest(plan: DeploymentPlan) -> dict[str, object]:
    """Return the resolved, JSON-safe deployment needed to replay a run."""

    if not isinstance(plan, DeploymentPlan):
        raise TypeError("deployment manifest requires a DeploymentPlan")
    worker_pools = []
    for pool_name in sorted(plan.worker_pools):
        pool = plan.worker_pools[pool_name]
        worker_pools.append({
            "name": pool.name,
            "role": pool.role,
            "hardware_id": pool.hardware_id,
            "ranks": list(pool.ranks),
            **({"node_ids": list(pool.node_ids)} if pool.node_ids else {}),
            "kv_transfer_layout": pool.kv_transfer_layout,
            "parallelism": asdict(pool.parallel_plan.config),
            "kv_cache": (
                None if pool.kv_cache is None else asdict(pool.kv_cache)
            ),
            "collectives": [
                asdict(collective)
                for collective in sorted(
                    pool.collectives,
                    key=lambda item: item.operation,
                )
            ],
            "pipeline_p2p": [
                asdict(config)
                for config in sorted(
                    pool.pipeline_p2p,
                    key=lambda item: (
                        item.source_stage,
                        item.destination_stage,
                    ),
                )
            ],
        })
    routes = [
        {key: value for key, value in asdict(route).items()
            if key != "kv_transport_scope" or value != "shared_fabric"}
        for route in sorted(
            plan.routes,
            key=lambda route: (
                route.source_pool,
                route.destination_pool,
                route.payload,
                route.algorithm,
                route.link,
                route.topology,
                route.resource_id,
            ),
        )
    ]
    return {
        "schema_version": 7,
        "deployment_id": plan.deployment_id,
        "mode": plan.mode,
        "router": asdict(plan.router),
        "worker_pools": worker_pools,
        "data_parallel_groups": {name: list(replicas)
            for name, replicas in sorted(plan.data_parallel_groups.items())},
        "routes": routes,
    }


def build_deployment_plan(config: DeploymentConfig, *, num_layers: int,
                          num_experts: int = 0,
                          pipeline_layer_partition=None) -> DeploymentPlan:
    config_fields = asdict(config)
    # The optional layout's None value denotes the unchanged legacy logical
    # contract, so preserve pre-extension deployment identities in that case.
    for pool_fields in config_fields["worker_pools"]:
        if not pool_fields["node_ids"]:
            del pool_fields["node_ids"]
        if pool_fields["kv_transfer_layout"] is None:
            del pool_fields["kv_transfer_layout"]
        if pool_fields["pipeline_layer_partition"] is None:
            del pool_fields["pipeline_layer_partition"]
    for route_fields in config_fields["routes"]:
        if route_fields["kv_transport_scope"] == "shared_fabric":
            del route_fields["kv_transport_scope"]
    if pipeline_layer_partition is None:
        config_identity = config_fields
    else:
        config_identity = {
            "config": config_fields,
            "pipeline_layer_partition": pipeline_layer_partition,
        }
    config_payload = json.dumps(
        config_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    deployment_id = hashlib.sha256(config_payload).hexdigest()
    next_rank = 0
    pools = {}
    replica_groups = {}
    used_ranks = set()
    for pool_config in config.worker_pools:
        partition = pool_config.pipeline_layer_partition
        if pipeline_layer_partition is not None:
            if partition is not None and partition != tuple(pipeline_layer_partition):
                raise ValueError("pool and deployment pipeline layer partitions disagree")
            partition = pipeline_layer_partition
        if pool_config.ranks:
            ranks = pool_config.ranks
        else:
            ranks = tuple(range(next_rank, next_rank + pool_config.parallelism.worker_count))
            next_rank += pool_config.parallelism.worker_count
        overlap = used_ranks.intersection(ranks)
        if overlap:
            raise ValueError(f"physical ranks assigned to multiple pools: {sorted(overlap)}")
        used_ranks.update(ranks)
        split_dp = (config.mode in ("colocated", "pd_disagg") and num_experts == 0
            and not pool_config.parallelism.enable_expert_parallel
            and pool_config.parallelism.data_parallel_size > 1)
        replica_count = pool_config.parallelism.data_parallel_size if split_dp else 1
        local_parallel = (replace(pool_config.parallelism, data_parallel_size=1)
            if split_dp else pool_config.parallelism)
        names = tuple(f"{pool_config.name}:dp{index}" for index in range(replica_count)) if split_dp else (pool_config.name,)
        replica_groups[pool_config.name] = names
        for replica_index, name in enumerate(names):
            if name in pools:
                raise ValueError("expanded DP pool identities collide")
            local_ranks = (ranks[replica_index * local_parallel.worker_count:
                (replica_index + 1) * local_parallel.worker_count] if split_dp else ranks)
            pools[name] = WorkerPool(
                name=name,
                role=pool_config.role,
                hardware_id=pool_config.hardware_id,
                ranks=local_ranks,
                parallel_plan=build_parallel_plan(
                    local_parallel,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    pipeline_layer_partition=partition),
                kv_cache=pool_config.kv_cache,
                collectives=pool_config.collectives,
                pipeline_p2p=pool_config.pipeline_p2p,
                kv_transfer_layout=pool_config.kv_transfer_layout,
                node_ids=(pool_config.node_ids[replica_index * local_parallel.worker_count:
                    (replica_index + 1) * local_parallel.worker_count]
                    if split_dp else pool_config.node_ids),
            )
    routes = []
    for route in config.routes:
        if route.source_pool not in replica_groups or route.destination_pool not in replica_groups:
            raise ValueError("deployment route references an unknown original pool")
        for source in replica_groups[route.source_pool]:
            for destination in replica_groups[route.destination_pool]:
                routes.append(TransferRoute(**replace(route, source_pool=source,
                    destination_pool=destination).__dict__))
    return DeploymentPlan(
        deployment_id, config.mode, pools, tuple(routes), config.router, replica_groups)
