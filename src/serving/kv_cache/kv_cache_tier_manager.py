"""Modeled Host-tier residency for full KV blocks and shared prefix loads."""

from dataclasses import dataclass
import math
from numbers import Real

from .kv_cache_config import KVCacheConfig
from .kv_cache_events import KVCacheAction, KVCacheEvent


@dataclass(frozen=True, order=True)
class _KVCacheBlock:
    request_id: str
    token_start: int
    token_end: int


@dataclass(frozen=True)
class KVCachePhysicalTransferCompletion:
    node_id: str
    call_id: str
    pool_name: str
    rank: int
    bytes: int
    link: str
    start_time_s: float
    end_time_s: float

    def __post_init__(self) -> None:
        for name in ("node_id", "call_id", "pool_name", "link"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"physical KV transfer completion {name} must be "
                    "non-empty")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int):
            raise TypeError(
                "physical KV transfer completion rank must be an integer")
        if self.rank < 0:
            raise ValueError(
                "physical KV transfer completion rank cannot be negative")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise TypeError(
                "physical KV transfer completion bytes must be an integer")
        if self.bytes <= 0:
            raise ValueError(
                "physical KV transfer completion bytes must be positive")
        for name in ("start_time_s", "end_time_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"physical KV transfer completion {name} must be finite "
                    "and non-negative")
        if self.end_time_s <= self.start_time_s:
            raise ValueError(
                "physical KV transfer completion interval must be positive")


@dataclass(frozen=True)
class KVCacheTransferCompletion:
    """Atomic completion of one logical event across all owner ranks."""

    ready_event: str
    action: KVCacheAction
    bytes: int
    source_tier: str
    destination_tier: str
    link: str
    physical_transfers: tuple[KVCachePhysicalTransferCompletion, ...]

    def __post_init__(self) -> None:
        for name in (
                "ready_event", "source_tier", "destination_tier", "link"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"KV transfer completion {name} must be non-empty")
        object.__setattr__(self, "action", KVCacheAction(self.action))
        if self.action not in (KVCacheAction.LOAD, KVCacheAction.EVICT,
                               KVCacheAction.STORE):
            raise ValueError("KV transfer completion requires load, evict or store")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise TypeError("KV transfer completion bytes must be an integer")
        if self.bytes <= 0:
            raise ValueError("KV transfer completion bytes must be positive")
        if not isinstance(self.physical_transfers, (list, tuple)):
            raise TypeError(
                "KV transfer physical completions must be a list or tuple")
        physical_transfers = tuple(self.physical_transfers)
        if (not physical_transfers
                or any(
                    not isinstance(item, KVCachePhysicalTransferCompletion)
                    for item in physical_transfers)):
            raise ValueError(
                "KV transfer completion requires canonical physical "
                "completions")
        object.__setattr__(
            self, "physical_transfers", physical_transfers)
        ranks = tuple(item.rank for item in physical_transfers)
        if ranks != tuple(sorted(ranks)) or len(ranks) != len(set(ranks)):
            raise ValueError(
                "KV transfer physical completions require unique ranks in "
                "ascending order")
        if len({item.node_id for item in physical_transfers}) != len(
                physical_transfers):
            raise ValueError(
                "KV transfer physical completion node IDs must be unique")
        if len({item.call_id for item in physical_transfers}) != len(
                physical_transfers):
            raise ValueError(
                "KV transfer physical completion call IDs must be unique")
        if len({item.pool_name for item in physical_transfers}) != 1:
            raise ValueError(
                "KV transfer physical completions must belong to one pool")
        if any(
                item.bytes != self.bytes or item.link != self.link
                for item in physical_transfers):
            raise ValueError(
                "KV transfer physical completion bytes/link disagree with "
                "the logical event")

    @property
    def pool_name(self) -> str:
        return self.physical_transfers[0].pool_name

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(item.rank for item in self.physical_transfers)

    @property
    def start_time_s(self) -> float:
        return min(item.start_time_s for item in self.physical_transfers)

    @property
    def end_time_s(self) -> float:
        return max(item.end_time_s for item in self.physical_transfers)

    @property
    def call_id(self) -> str:
        if len(self.physical_transfers) != 1:
            raise ValueError(
                "multi-rank KV completion has no singular call_id")
        return self.physical_transfers[0].call_id


@dataclass(frozen=True)
class _KVCacheInflightTransfer:
    event: KVCacheEvent
    pool_name: str
    ranks: tuple[int, ...]


class KVCacheTierManager:
    """Track HBM and Host residency, including completed dual-copy stores.

    The caller remains responsible for request admission, batching,
    preemption, and production victim selection.  This manager owns block
    residency, destination-capacity reservation, and transfer completion;
    its LRU preparation API is a modeled fixture helper.
    """

    def __init__(
        self, config: KVCacheConfig, *, fixed_prefix_hbm_percent: int | None = None,
    ) -> None:
        if not isinstance(config, KVCacheConfig):
            raise TypeError("KV tier manager requires KVCacheConfig")
        if config.offload_policy != "lru":
            raise ValueError(
                "modeled Host KV offload currently requires policy 'lru'")
        if fixed_prefix_hbm_percent is not None and (
                type(fixed_prefix_hbm_percent) is not int
                or not 0 <= fixed_prefix_hbm_percent <= 100):
            raise ValueError("fixed prefix HBM percent must be an integer from 0 to 100")
        self.fixed_prefix_hbm_percent = fixed_prefix_hbm_percent
        tiers_by_name = {
            tier.name.lower(): tier for tier in config.tiers
        }
        if set(tiers_by_name) != {"hbm", "host"}:
            raise ValueError(
                "modeled Host KV offload requires exactly hbm and host tiers")
        if any(tier.name != tier.name.lower() for tier in config.tiers):
            raise ValueError(
                "modeled Host KV offload requires lowercase tier names")
        hbm = tiers_by_name["hbm"]
        host = tiers_by_name["host"]
        if hbm.link is not None:
            raise ValueError("HBM KV tier cannot declare an offload link")
        if host.link != "pcie":
            raise ValueError(
                "modeled Host KV offload currently requires the PCIe link")
        if hbm.block_size_tokens != host.block_size_tokens:
            raise NotImplementedError(
                "modeled Host KV offload currently requires equal HBM and "
                "Host block sizes")
        self.config = config
        self._tiers_by_name = tiers_by_name
        self._block_size_tokens = hbm.block_size_tokens
        self._block_bytes = config.bytes_per_token * self._block_size_tokens
        for tier in (hbm, host):
            if tier.capacity_bytes < self._block_bytes:
                raise ValueError(
                    f"KV tier {tier.name!r} must hold at least one block")
            if tier.capacity_bytes % self._block_bytes:
                raise ValueError(
                    f"KV tier {tier.name!r} capacity must be an exact block "
                    "multiple")
        self._resident_tier: dict[_KVCacheBlock, str] = {}
        self._host_replicas: set[_KVCacheBlock] = set()
        self._host_alias_source: dict[_KVCacheBlock, _KVCacheBlock] = {}
        self._saw_store = False
        self._saw_shared_load = False
        self._used_bytes = {"hbm": 0, "host": 0}
        self._reserved_bytes = {"hbm": 0, "host": 0}
        self._last_touch: dict[_KVCacheBlock, int] = {}
        self._inflight_by_event: dict[str, _KVCacheInflightTransfer] = {}
        self._inflight_by_block: dict[_KVCacheBlock, str] = {}
        self._seen_ready_events: set[str] = set()
        self._sequence = 0
        self._touch_sequence = 0

    @property
    def block_bytes(self) -> int:
        return self._block_bytes

    @property
    def block_size_tokens(self) -> int:
        return self._block_size_tokens

    def fixed_prefix_block_counts(self, effective_hit_tokens: int) -> tuple[int, int]:
        """Round HBM blocks to nearest integer, ties upward; Host gets the rest."""

        percent = self.fixed_prefix_hbm_percent
        if percent is None:
            raise ValueError("fixed prefix placement is not configured")
        if (type(effective_hit_tokens) is not int
                or effective_hit_tokens < 0
                or effective_hit_tokens % self._block_size_tokens):
            raise ValueError("effective prefix hit must be non-negative full blocks")
        blocks = effective_hit_tokens // self._block_size_tokens
        hbm_blocks = (blocks * percent + 50) // 100
        return hbm_blocks, blocks - hbm_blocks

    def allocate(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
    ) -> KVCacheEvent:
        block = self._block(request_id, token_start, token_end)
        return self.apply_capacity_event(
            self._capacity_event(block, KVCacheAction.ALLOCATE, "hbm"))

    def apply_capacity_event(self, event: KVCacheEvent) -> KVCacheEvent:
        """Atomically apply one aligned full-block range capacity intent."""

        if not isinstance(event, KVCacheEvent):
            raise TypeError("KV capacity intent must be a KVCacheEvent")
        if event.action not in (
                KVCacheAction.ALLOCATE, KVCacheAction.RELEASE):
            raise ValueError(
                "KV capacity intent requires allocate or release")
        blocks = self._blocks(
            event.request_id, event.token_start, event.token_end)
        expected_bytes = len(blocks) * self._block_bytes
        if event.bytes != expected_bytes:
            raise ValueError(
                "KV capacity intent bytes disagree with its full-block range")
        tier = event.tier
        if tier not in self._tiers_by_name:
            raise ValueError(
                f"KV capacity intent references unknown tier {tier!r}")
        if event.action is KVCacheAction.ALLOCATE:
            if tier != "hbm" and self.fixed_prefix_hbm_percent is None:
                raise NotImplementedError(
                    "modeled KV allocation currently requires the HBM tier")
            conflicts = tuple(
                block for block in blocks
                if (block in self._resident_tier
                    or block in self._inflight_by_block)
            )
            if conflicts:
                raise ValueError(
                    "cannot allocate resident or in-flight KV blocks: "
                    f"{conflicts!r}")
            self._require_destination_capacity(tier, len(blocks))
            for block in blocks:
                self._resident_tier[block] = tier
                self._used_bytes[tier] += self._block_bytes
                self._touch(block)
        elif event.action is KVCacheAction.RELEASE:
            non_resident = tuple(
                block for block in blocks
                if block not in self._resident_tier)
            if non_resident:
                raise ValueError(
                    "cannot release non-resident KV blocks: "
                    f"{non_resident!r}")
            inflight = tuple(
                block for block in blocks
                if block in self._inflight_by_block)
            if inflight:
                raise ValueError(
                    "cannot release in-flight KV blocks: "
                    f"{inflight!r}")
            wrong_tier = tuple(
                block for block in blocks
                if (self._resident_tier[block] != tier
                    and not (tier == "host"
                        and self._resident_tier[block] == "hbm"
                        and block in self._host_replicas)))
            if wrong_tier:
                raise ValueError(
                    "KV release tier disagrees with current residency: "
                    f"{wrong_tier!r}")
            pinned_sources = tuple(
                source for source in blocks
                if any(alias_source == source
                       and (alias in self._inflight_by_block
                            or self._resident_tier[alias] == "host")
                       for alias, alias_source in self._host_alias_source.items()))
            if pinned_sources:
                raise ValueError(
                    "cannot release Host source with an unfinished shared load: "
                    f"{pinned_sources!r}")
            self._used_bytes[tier] -= expected_bytes
            if self._used_bytes[tier] < 0:
                raise RuntimeError("KV tier used-byte ledger became negative")
            for block in blocks:
                if block in self._host_alias_source:
                    del self._host_alias_source[block]
                retains_host = (block in self._host_replicas
                    and self._saw_store and tier == "hbm")
                if not retains_host:
                    for alias, source in tuple(self._host_alias_source.items()):
                        if source == block:
                            del self._host_alias_source[alias]
                if block in self._host_replicas:
                    self._host_replicas.remove(block)
                    if tier == "host":
                        # The CPU pool evicted a stored copy while its GPU
                        # source block is still owned by a live request.
                        continue
                    if self._saw_store:
                        # Native eager stores survive request completion as
                        # CPU prefix-cache entries. A later CPU eviction must
                        # explicitly release this Host-resident block.
                        self._resident_tier[block] = "host"
                        continue
                    self._used_bytes["host"] -= self._block_bytes
                del self._resident_tier[block]
                del self._last_touch[block]
        else:
            raise RuntimeError(
                f"unknown KV capacity action: {event.action.value!r}")
        return event

    def register_transfer_intent(
        self,
        event: KVCacheEvent,
        *,
        pool_name: str,
        ranks: tuple[int, ...],
    ) -> KVCacheEvent:
        """Validate and reserve one framework-specified load/evict intent."""

        if not isinstance(event, KVCacheEvent):
            raise TypeError("KV transfer intent must be a KVCacheEvent")
        if event.action not in (KVCacheAction.LOAD, KVCacheAction.EVICT,
                                KVCacheAction.STORE):
            raise ValueError("KV transfer intent requires load, evict or store")
        if (event.action is KVCacheAction.STORE
                and self.fixed_prefix_hbm_percent is not None):
            raise ValueError("native store cannot combine with fixed prefix tiers")
        pool_name, ranks = self._transfer_owner(pool_name, ranks)
        blocks = self._blocks(
            event.request_id, event.token_start, event.token_end)
        transfer_bytes = len(blocks) * self._block_bytes
        if event.bytes != transfer_bytes:
            raise ValueError(
                "KV transfer intent bytes disagree with its full blocks")
        if event.tier != "host":
            raise NotImplementedError(
                "modeled KV transfer intent currently requires the Host tier")
        if event.action is KVCacheAction.LOAD:
            expected_source = "host"
            expected_destination = "hbm"
        elif event.action in (KVCacheAction.EVICT, KVCacheAction.STORE):
            expected_source = "hbm"
            expected_destination = "host"
        else:
            raise RuntimeError(
                f"unknown KV transfer action: {event.action.value!r}")
        source = event.source_tier
        destination = event.destination_tier
        if source is None or destination is None:
            raise RuntimeError("validated KV transfer intent lacks endpoints")
        if source != expected_source or destination != expected_destination:
            raise ValueError(
                "KV transfer intent direction disagrees with its action")
        shared_sources = ()
        if event.action is KVCacheAction.LOAD and event.source_request_id is not None:
            shared_sources = self._blocks(
                event.source_request_id, event.token_start, event.token_end)
            if any(block in self._resident_tier for block in blocks):
                raise ValueError("shared KV load destination is already resident")
            if any(source_block not in self._resident_tier
                   or self._resident_tier[source_block] != "host"
                   for source_block in shared_sources):
                raise ValueError("shared KV load source is not Host-resident")
            if any(source_block in self._inflight_by_block
                   for source_block in shared_sources):
                raise ValueError("shared KV load source is still in flight")
        else:
            if any(block not in self._resident_tier for block in blocks):
                raise ValueError("KV transfer source block is not resident")
            if any(self._resident_tier[block] != source for block in blocks):
                raise ValueError("KV transfer source disagrees with residency")
        if any(block in self._inflight_by_block for block in blocks):
            raise ValueError("cannot transfer an in-flight KV block")
        if (event.action is KVCacheAction.STORE
                and any(block in self._host_replicas for block in blocks)):
            raise ValueError("cannot store a KV block that already has a Host copy")
        ready_event = event.ready_event
        if ready_event is None:
            raise RuntimeError("validated KV transfer intent lacks ready_event")
        if ready_event in self._seen_ready_events:
            raise ValueError("KV transfer ready_event values must be unique")
        self._require_destination_capacity(destination, len(blocks))
        if shared_sources:
            self._saw_shared_load = True
            for block, source_block in zip(blocks, shared_sources, strict=True):
                self._resident_tier[block] = "host"
                self._host_alias_source[block] = source_block
                self._touch(block)
        self._reserved_bytes[destination] += transfer_bytes
        self._inflight_by_event[ready_event] = _KVCacheInflightTransfer(
            event, pool_name, ranks)
        for block in blocks:
            self._inflight_by_block[block] = ready_event
        self._seen_ready_events.add(ready_event)
        if event.action is KVCacheAction.STORE:
            self._saw_store = True
        return event

    def assert_quiescent(self) -> None:
        """Reject a completed run with unfinished transfers or reservations."""

        if self._inflight_by_event or self._inflight_by_block:
            raise RuntimeError(
                "KV tier manager has in-flight transfers at run completion")
        if any(self._reserved_bytes[tier] for tier in self._reserved_bytes):
            raise RuntimeError(
                "KV tier manager has reserved capacity at run completion")
        if self._host_alias_source:
            raise RuntimeError("KV tier manager retains shared load request aliases")

    def touch(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
    ) -> None:
        block = self._block(request_id, token_start, token_end)
        if block not in self._resident_tier:
            raise ValueError("cannot touch a non-resident KV block")
        if block in self._inflight_by_block:
            raise ValueError("cannot touch an in-flight KV block")
        self._touch(block)

    def prepare_evict(
        self,
        *,
        pool_name: str,
        ranks: tuple[int, ...],
    ) -> KVCacheEvent:
        candidates = sorted(
            (
                block for block, tier in self._resident_tier.items()
                if tier == "hbm" and block not in self._inflight_by_block
            ),
            key=lambda block: (self._last_touch[block], block),
        )
        if not candidates:
            raise ValueError("no resident HBM KV block is available for eviction")
        self._require_destination_capacity("host")
        return self._prepare_transfer(
            candidates[0], KVCacheAction.EVICT, "hbm", "host",
            pool_name=pool_name, ranks=ranks)

    def prepare_load(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
        *,
        pool_name: str,
        ranks: tuple[int, ...],
    ) -> KVCacheEvent:
        block = self._block(request_id, token_start, token_end)
        if block not in self._resident_tier:
            raise ValueError("cannot load a non-resident KV block")
        if self._resident_tier[block] != "host":
            raise ValueError("KV load requires a Host-resident source block")
        if block in self._inflight_by_block:
            raise ValueError("cannot load an in-flight KV block")
        self._require_destination_capacity("hbm")
        return self._prepare_transfer(
            block, KVCacheAction.LOAD, "host", "hbm",
            pool_name=pool_name, ranks=ranks)

    def prepare_store(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
        *,
        pool_name: str,
        ranks: tuple[int, ...],
    ) -> KVCacheEvent:
        block = self._block(request_id, token_start, token_end)
        if block not in self._resident_tier:
            raise ValueError("cannot store a non-resident KV block")
        if self._resident_tier[block] != "hbm":
            raise ValueError("KV store requires an HBM-resident source block")
        if block in self._host_replicas:
            raise ValueError("KV block is already stored in Host")
        if block in self._inflight_by_block:
            raise ValueError("cannot store an in-flight KV block")
        self._require_destination_capacity("host")
        return self._prepare_transfer(
            block, KVCacheAction.STORE, "hbm", "host",
            pool_name=pool_name, ranks=ranks)

    def complete_transfer(
        self,
        completion: KVCacheTransferCompletion,
    ) -> KVCacheEvent:
        if not isinstance(completion, KVCacheTransferCompletion):
            raise TypeError(
                "KV transfer completion must come from the canonical "
                "execution timeline")
        ready_event = completion.ready_event
        if ready_event not in self._inflight_by_event:
            raise ValueError(f"unknown KV transfer completion: {ready_event!r}")
        call_fragment = f":kv-offload:{ready_event}"
        if any(
                physical.call_id != physical.node_id
                or not (
                    physical.call_id.endswith(call_fragment)
                    or f"{call_fragment}:tp-rank-" in physical.call_id
                )
                for physical in completion.physical_transfers):
            raise ValueError(
                "KV transfer completion call IDs do not preserve ready_event")
        inflight = self._inflight_by_event[ready_event]
        event = inflight.event
        if (completion.pool_name != inflight.pool_name
                or completion.ranks != inflight.ranks):
            raise ValueError(
                "KV transfer completion owner ranks disagree with its intent")
        if (completion.action is not event.action
                or completion.bytes != event.bytes
                or completion.source_tier != event.source_tier
                or completion.destination_tier != event.destination_tier
                or completion.link != self._tiers_by_name["host"].link):
            raise ValueError(
                "KV transfer completion provenance disagrees with its intent")
        blocks = self._blocks(
            event.request_id, event.token_start, event.token_end)
        source = event.source_tier
        destination = event.destination_tier
        if source is None or destination is None:
            raise RuntimeError("in-flight KV transfer lacks tier endpoints")
        if any(self._resident_tier[block] != source for block in blocks):
            raise RuntimeError("in-flight KV source residency changed")
        retain_source_copy = (event.action is KVCacheAction.STORE
            or (self.fixed_prefix_hbm_percent is not None
                and event.action is KVCacheAction.LOAD)
            or (event.action is KVCacheAction.LOAD
                and event.source_request_id is not None))
        if not retain_source_copy:
            self._used_bytes[source] -= event.bytes
        self._reserved_bytes[destination] -= event.bytes
        self._used_bytes[destination] += event.bytes
        if min(
                self._used_bytes[source],
                self._reserved_bytes[destination]) < 0:
            raise RuntimeError("KV tier capacity ledger became negative")
        for block in blocks:
            if event.action is not KVCacheAction.STORE:
                self._resident_tier[block] = destination
            if retain_source_copy and event.source_request_id is None:
                self._host_replicas.add(block)
        del self._inflight_by_event[ready_event]
        for block in blocks:
            del self._inflight_by_block[block]
        if event.action is KVCacheAction.LOAD:
            for block in blocks:
                self._touch(block)
        elif event.action is KVCacheAction.EVICT:
            pass
        elif event.action is KVCacheAction.STORE:
            pass
        else:
            raise RuntimeError(
                f"unknown in-flight KV action: {event.action.value!r}")
        return event

    def release(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
    ) -> KVCacheEvent:
        block = self._block(request_id, token_start, token_end)
        if block not in self._resident_tier:
            raise ValueError("cannot release a non-resident KV block")
        if block in self._inflight_by_block:
            raise ValueError("cannot release an in-flight KV block")
        tier = self._resident_tier[block]
        return self.apply_capacity_event(
            self._capacity_event(block, KVCacheAction.RELEASE, tier))

    def manifest(self) -> dict[str, object]:
        residency = []
        for block in sorted(self._resident_tier):
            if block in self._inflight_by_block:
                inflight_ready_event = self._inflight_by_block[block]
            else:
                inflight_ready_event = None
            row = {
                "request_id": block.request_id,
                "token_start": block.token_start,
                "token_end": block.token_end,
                "tier": self._resident_tier[block],
                "inflight_ready_event": inflight_ready_event,
            }
            if (self.fixed_prefix_hbm_percent is not None or self._saw_store
                    or block in self._host_alias_source):
                row["valid_copies"] = (
                    ("hbm", "host") if (block in self._host_replicas
                        or (block in self._host_alias_source
                            and self._resident_tier[block] == "hbm"))
                    else (self._resident_tier[block],))
            residency.append(row)
        manifest = {
            "schema_version": (4 if self._saw_store or self._saw_shared_load else
                               3 if self.fixed_prefix_hbm_percent is not None else 2),
            "configured_policy": "lru",
            "victim_policy_validation": (
                "external_transfer_intent_not_validated"),
            "validation_status": ("modeled_shared_prefix_full_blocks"
                if self._saw_shared_load else "modeled_request_private_full_blocks"),
            "block_size_tokens": self._block_size_tokens,
            "block_bytes": self._block_bytes,
            "tiers": {
                tier: {
                    "capacity_bytes": self._tiers_by_name[tier].capacity_bytes,
                    "used_bytes": self._used_bytes[tier],
                    "reserved_bytes": self._reserved_bytes[tier],
                    "resident_blocks": sum(
                        resident == tier and (tier != "host"
                            or block not in self._host_alias_source)
                        for block, resident in self._resident_tier.items())
                        + (len(self._host_replicas) if tier == "host" else 0),
                }
                for tier in ("hbm", "host")
            },
            "residency": residency,
            "inflight_ready_events": tuple(sorted(self._inflight_by_event)),
        }
        if self.fixed_prefix_hbm_percent is not None:
            manifest["fixed_prefix_hbm_percent"] = self.fixed_prefix_hbm_percent
        return manifest

    def _block(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
    ) -> _KVCacheBlock:
        blocks = self._blocks(request_id, token_start, token_end)
        if len(blocks) != 1:
            raise ValueError("KV transfer requires exactly one full block")
        return blocks[0]

    def _blocks(
        self,
        request_id: str,
        token_start: int,
        token_end: int,
    ) -> tuple[_KVCacheBlock, ...]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("KV block request_id must be non-empty")
        for name, value in (
                ("token_start", token_start), ("token_end", token_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"KV block {name} must be an integer")
        if token_start < 0 or token_end <= token_start:
            raise ValueError("KV block token range must be positive")
        if token_start % self._block_size_tokens:
            raise ValueError("KV block token_start must be block-aligned")
        if token_end % self._block_size_tokens:
            raise ValueError("KV block token_end must be block-aligned")
        return tuple(
            _KVCacheBlock(
                request_id,
                block_start,
                block_start + self._block_size_tokens,
            )
            for block_start in range(
                token_start, token_end, self._block_size_tokens)
        )

    def _touch(self, block: _KVCacheBlock) -> None:
        self._last_touch[block] = self._touch_sequence
        self._touch_sequence += 1

    def _require_destination_capacity(
        self,
        tier: str,
        block_count: int = 1,
    ) -> None:
        if (isinstance(block_count, bool)
                or not isinstance(block_count, int)):
            raise TypeError("KV destination block_count must be an integer")
        if block_count < 1:
            raise ValueError("KV destination block_count must be positive")
        config = self._tiers_by_name[tier]
        available = (
            config.capacity_bytes
            - self._used_bytes[tier]
            - self._reserved_bytes[tier]
        )
        if available < block_count * self._block_bytes:
            raise ValueError(f"KV tier {tier!r} has no free block capacity")

    def _prepare_transfer(
        self,
        block: _KVCacheBlock,
        action: KVCacheAction,
        source: str,
        destination: str,
        *,
        pool_name: str,
        ranks: tuple[int, ...],
    ) -> KVCacheEvent:
        if self._resident_tier[block] != source:
            raise ValueError("KV transfer source disagrees with residency")
        ready_event = (
            f"kv-transfer-{self._sequence:08d}:{action.value}:"
            f"{block.request_id}:{block.token_start}-{block.token_end}"
        )
        self._sequence += 1
        event = KVCacheEvent(
            request_id=block.request_id,
            action=action,
            tier="host",
            token_start=block.token_start,
            token_end=block.token_end,
            bytes=self._block_bytes,
            source_tier=source,
            destination_tier=destination,
            ready_event=ready_event,
        )
        return self.register_transfer_intent(
            event, pool_name=pool_name, ranks=ranks)

    @staticmethod
    def _transfer_owner(
        pool_name: str,
        ranks: tuple[int, ...],
    ) -> tuple[str, tuple[int, ...]]:
        if not isinstance(pool_name, str) or not pool_name:
            raise ValueError("KV transfer owner pool_name must be non-empty")
        if not isinstance(ranks, (list, tuple)):
            raise TypeError("KV transfer owner ranks must be a list or tuple")
        normalized_ranks = tuple(ranks)
        if (not normalized_ranks
                or any(
                    isinstance(rank, bool) or not isinstance(rank, int)
                    or rank < 0
                    for rank in normalized_ranks)):
            raise ValueError(
                "KV transfer owner ranks must be non-negative integers")
        if (normalized_ranks != tuple(sorted(normalized_ranks))
                or len(normalized_ranks) != len(set(normalized_ranks))):
            raise ValueError(
                "KV transfer owner ranks must be unique and ascending")
        return pool_name, normalized_ranks

    def _capacity_event(
        self,
        block: _KVCacheBlock,
        action: KVCacheAction,
        tier: str,
    ) -> KVCacheEvent:
        return KVCacheEvent(
            request_id=block.request_id,
            action=action,
            tier=tier,
            token_start=block.token_start,
            token_end=block.token_end,
            bytes=self._block_bytes,
        )
