"""Cross-framework facts emitted by cache managers."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class KVCacheAction(str, Enum):
    HIT = "hit"
    MISS = "miss"
    ALLOCATE = "allocate"
    RELEASE = "release"
    LOAD = "load"
    EVICT = "evict"
    STORE = "store"


@dataclass(frozen=True)
class KVCacheEvent:
    request_id: str
    action: KVCacheAction
    tier: str
    token_start: int
    token_end: int
    bytes: int
    source_tier: Optional[str] = None
    destination_tier: Optional[str] = None
    ready_event: Optional[str] = None
    # Native content-cache loads may read a Host block saved by another
    # request at the same prefix position. None keeps request-private loads.
    source_request_id: Optional[str] = None

    def __post_init__(self) -> None:
        for name in ("request_id", "tier"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"KV cache event {name} must be a non-empty string")
        object.__setattr__(self, "action", KVCacheAction(self.action))
        for name in ("token_start", "token_end", "bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"KV cache event {name} must be an integer")
        if self.token_start < 0 or self.token_end < self.token_start:
            raise ValueError("invalid KV cache token range")
        if self.bytes < 0:
            raise ValueError("KV cache event bytes cannot be negative")
        for name in ("source_tier", "destination_tier", "ready_event"):
            value = getattr(self, name)
            if value is not None and (
                    not isinstance(value, str) or not value):
                raise ValueError(
                    f"KV cache event {name} must be non-empty or null")
        if self.source_request_id is not None:
            if (not isinstance(self.source_request_id, str)
                    or not self.source_request_id
                    or self.source_request_id == self.request_id):
                raise ValueError("shared KV source request must be distinct and non-empty")
            if self.action is not KVCacheAction.LOAD:
                raise ValueError("shared KV source request requires a load event")
        if self.action in (KVCacheAction.LOAD, KVCacheAction.EVICT,
                           KVCacheAction.STORE):
            if not self.source_tier or not self.destination_tier:
                raise ValueError(f"{self.action.value} requires source and destination tiers")
            if self.source_tier.lower() == self.destination_tier.lower():
                raise ValueError(
                    f"{self.action.value} requires distinct tier endpoints")
            if self.token_end == self.token_start or self.bytes == 0:
                raise ValueError(
                    f"{self.action.value} requires a positive transfer")
            if self.ready_event is None:
                raise ValueError(
                    f"{self.action.value} requires a ready_event")
        elif self.source_tier is not None or self.destination_tier is not None:
            raise ValueError(
                f"{self.action.value} cannot carry source/destination tiers")
        elif self.ready_event is not None:
            raise ValueError(
                f"{self.action.value} cannot carry a ready_event")
        if self.action in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE):
            if self.token_end == self.token_start or self.bytes == 0:
                raise ValueError(
                    f"{self.action.value} requires positive cache capacity")


def host_kv_transfer_attributes(event, *, cache_config, model_bytes_per_token, scheduled_request_ids):
    """Validate one externally owned Host/HBM DMA intent before graph emission.

    Page storage and transfer retirement belong to KVCacheTierManager. This
    projects its exact per-rank payload, not tensor addresses or DMA scratch.
    """
    from .kv_cache_config import KVCacheConfig
    if not isinstance(event, KVCacheEvent) or event.action not in (
            KVCacheAction.LOAD, KVCacheAction.EVICT, KVCacheAction.STORE):
        raise TypeError("Host KV transfer requires a normalized load/evict/store event")
    if not isinstance(cache_config, KVCacheConfig):
        raise ValueError("KV offload event requires a configured cache tier plan")
    if type(model_bytes_per_token) is not int or model_bytes_per_token <= 0:
        raise ValueError("Host KV transfer needs positive per-rank model bytes per token")
    if cache_config.bytes_per_token != model_bytes_per_token:
        raise ValueError("KV offload bytes_per_token disagrees with the model KV layout")
    source = event.source_tier
    destination = event.destination_tier
    if source is None or destination is None:
        raise RuntimeError("validated KV offload event lacks endpoints")
    source_is_hbm = source.lower() == "hbm"
    destination_is_hbm = destination.lower() == "hbm"
    if (event.action in (KVCacheAction.EVICT, KVCacheAction.STORE)
            and source_is_hbm and not destination_is_hbm):
        offload_tier_name = destination.lower()
    elif (event.action is KVCacheAction.LOAD
            and destination_is_hbm and not source_is_hbm):
        offload_tier_name = source.lower()
    else:
        raise ValueError(
            "KV load/evict direction disagrees with its tier endpoints")
    if offload_tier_name != "host":
        raise NotImplementedError(
            "modeled KV offload currently supports only the Host tier")
    if event.tier.lower() != offload_tier_name:
        raise ValueError(
            "KV offload event tier disagrees with its non-HBM endpoint")
    matching_tiers = [
        tier for tier in cache_config.tiers
        if tier.name.lower() == offload_tier_name
    ]
    if len(matching_tiers) != 1:
        raise ValueError(
            "KV offload event must match exactly one configured tier")
    offload_tier = matching_tiers[0]
    if offload_tier.link is None:
        raise ValueError("KV offload tier requires a physical link")
    hbm_tiers = [
        tier for tier in cache_config.tiers
        if tier.name.lower() == "hbm"
    ]
    if len(hbm_tiers) != 1:
        raise ValueError(
            "KV offload event requires exactly one configured HBM tier")
    hbm_tier = hbm_tiers[0]
    if offload_tier.block_size_tokens != hbm_tier.block_size_tokens:
        raise NotImplementedError(
            "modeled Host KV offload currently requires equal HBM "
            "and Host block sizes")
    transfer_token_count = event.token_end - event.token_start
    if (event.token_start % hbm_tier.block_size_tokens
            or transfer_token_count % hbm_tier.block_size_tokens):
        raise ValueError(
            "KV offload event must cover aligned full HBM blocks")
    expected_bytes = (
        transfer_token_count
        * cache_config.bytes_per_token
    )
    if event.bytes != expected_bytes:
        raise ValueError(
            "KV offload event bytes disagree with its token range")
    if event.action is KVCacheAction.LOAD and event.request_id not in scheduled_request_ids:
        raise ValueError("modeled KV demand load must target an active request in the same scheduler step")
    attributes = {
        "bytes": event.bytes,
        "direction": event.action.value,
        "source_tier": event.source_tier,
        "destination_tier": event.destination_tier,
        "tier": event.tier,
        "algorithm": "dma",
        "link": offload_tier.link,
        "request_id": event.request_id,
        "token_start": event.token_start,
        "token_end": event.token_end,
        "ready_event": event.ready_event,
    }
    if event.source_request_id is not None:
        attributes["source_request_id"] = event.source_request_id
    return attributes
