"""KV-cache capacity and offload configuration passed to a backend."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class KVCacheTierConfig:
    name: str
    capacity_bytes: int
    block_size_tokens: int
    link: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("KV cache tier name must be a non-empty string")
        for name in ("capacity_bytes", "block_size_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"KV cache tier {name} must be an integer")
        if self.link is not None and (
                not isinstance(self.link, str) or not self.link.strip()):
            raise ValueError("KV cache tier link must be non-empty or null")
        if self.capacity_bytes < 0:
            raise ValueError("KV cache tier capacity cannot be negative")
        if self.block_size_tokens < 1:
            raise ValueError("block_size_tokens must be positive")
        if self.name.lower() != "hbm" and not self.link:
            raise ValueError(f"offload tier {self.name!r} requires a link")


@dataclass(frozen=True)
class KVCacheConfig:
    tiers: Tuple[KVCacheTierConfig, ...]
    bytes_per_token: int
    offload_policy: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tiers, (list, tuple)):
            raise TypeError("KVCacheConfig tiers must be a list or tuple")
        object.__setattr__(self, "tiers", tuple(self.tiers))
        if not self.tiers:
            raise ValueError("KVCacheConfig requires at least one tier")
        if any(not isinstance(tier, KVCacheTierConfig) for tier in self.tiers):
            raise TypeError("KVCacheConfig tiers must be KVCacheTierConfig objects")
        names = [tier.name.lower() for tier in self.tiers]
        if len(names) != len(set(names)):
            raise ValueError("KV cache tier names must be unique")
        if "hbm" not in names:
            raise ValueError("KVCacheConfig requires an HBM tier")
        if isinstance(self.bytes_per_token, bool) or not isinstance(
                self.bytes_per_token, int):
            raise TypeError("bytes_per_token must be an integer")
        if self.bytes_per_token < 1:
            raise ValueError("bytes_per_token must be positive")
        if self.offload_policy is not None and (
                not isinstance(self.offload_policy, str)
                or not self.offload_policy.strip()):
            raise ValueError("offload_policy must be non-empty or null")

    @property
    def hbm_blocks(self) -> int:
        hbm = next(tier for tier in self.tiers if tier.name.lower() == "hbm")
        block_bytes = self.bytes_per_token * hbm.block_size_tokens
        return hbm.capacity_bytes // block_bytes
