"""Control-plane backend protocol shared by vLLM and SGLang."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Optional, Protocol, runtime_checkable

from src.serving.request_gen.request_stream import RequestSpec
from src.serving.deployment.deployment_config import worker_role_phase

from .schedule_step import BackendStep
from .pipeline_depth import validate_scheduler_pipeline_depth


def backend_phase(role: str) -> str:
    """Classify a validated worker role without an implicit default phase."""

    return worker_role_phase(role)


def scheduler_effective_prefix_hit_tokens(
    request: RequestSpec,
    block_size_tokens: int,
) -> int:
    """Normalize a Trace prefix hit to reusable Scheduler block state.

    The current contract matches pinned vLLM 0.11: only complete blocks are
    reusable, and the final prompt token is recomputed to produce logits.
    """

    if not isinstance(request, RequestSpec):
        raise TypeError(
            "Scheduler prefix-hit normalization requires a RequestSpec")
    if isinstance(block_size_tokens, bool) or not isinstance(
            block_size_tokens, int):
        raise TypeError("Scheduler KV block size must be an integer")
    if block_size_tokens <= 0:
        raise ValueError("Scheduler KV block size must be positive")
    reusable_tokens = min(
        request.prefix_hit_tokens,
        request.input_tokens - 1,
    )
    return reusable_tokens - reusable_tokens % block_size_tokens


@dataclass(frozen=True)
class BackendConfig:
    pool_name: str
    role: str
    max_batched_tokens: int
    max_requests: int
    num_kv_blocks: int
    kv_block_size_tokens: int = 16
    kv_bytes_per_token: int = 0
    scheduler_pipeline_depth: int = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in ("pool_name", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"backend {name} must be a non-empty string")
        backend_phase(self.role)
        for name in (
                "max_batched_tokens", "max_requests", "num_kv_blocks",
                "kv_block_size_tokens", "kv_bytes_per_token"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"backend {name} must be an integer")
        if min(
            self.max_batched_tokens,
            self.max_requests,
            self.num_kv_blocks,
            self.kv_block_size_tokens,
        ) < 1:
            raise ValueError("backend limits and num_kv_blocks must be positive")
        if self.kv_bytes_per_token < 0:
            raise ValueError("kv_bytes_per_token cannot be negative")
        validate_scheduler_pipeline_depth(
            self.scheduler_pipeline_depth,
            "backend scheduler pipeline depth",
        )

    @property
    def is_prefill_only(self) -> bool:
        return backend_phase(self.role) == "prefill"

    @property
    def is_decode_only(self) -> bool:
        return backend_phase(self.role) == "decode"


def backend_config_sha256(config: BackendConfig) -> str:
    """Hash normalized Scheduler capacity/role inputs for replay binding."""

    if not isinstance(config, BackendConfig):
        raise TypeError("backend config hashing requires BackendConfig")
    payload = {
        "schema_version": 1,
        "backend_config": asdict(config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class SchedulerBackend(Protocol):
    config: BackendConfig

    def add_request(self, request: RequestSpec) -> None: ...

    def plan_step(self, virtual_time_s: float) -> Optional[BackendStep]:
        """Return work, or None to await an outstanding executor completion."""
        ...

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float: ...

    def has_work(self) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class BulkRequestSchedulerBackend(Protocol):
    def add_requests(self, requests: tuple[RequestSpec, ...]) -> None: ...


@runtime_checkable
class TailBatchSchedulerBackend(Protocol):
    def drain_steps(
        self,
        max_steps: int,
    ) -> tuple[tuple[BackendStep, ...], bool]: ...
