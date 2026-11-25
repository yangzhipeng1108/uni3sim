"""Versioned external LOAD/EVICT intents for modeled Host KV offload."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Tuple

from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
    worker_role_phase,
)
from src.serving.deployment.deployment_plan import DeploymentPlan
from src.serving.request_gen.request_stream import (
    RequestStream,
    request_stream_sha256,
)
from src.serving.schedule.backend_protocol import (
    BackendConfig,
    backend_config_sha256,
)
from src.serving.schedule.schedule_step import (
    BackendStep,
    schedule_step_sha256,
)
from src.serving.workload.model_config import (
    ModelConfig,
    is_validated_qwen3_tp2_host_kv_model_scope,
)

from .kv_cache_config import KVCacheConfig, KVCacheTierConfig
from .kv_cache_events import KVCacheAction, KVCacheEvent


_ROOT_FIELDS = {
    "schema_version",
    "artifact_id",
    "provenance",
    "context",
    "steps",
}
_PROVENANCE_FIELDS = {
    "source_kind",
    "producer",
    "producer_version",
    "source_reference",
}
_CONTEXT_FIELDS = {
    "request_stream_sha256",
    "model_config_sha256",
    "deployment_id",
    "scheduler_config_sha256",
    "scheduler_runtime_sha256",
    "scheduler_step_count",
    "pool_name",
    "backend_role",
    "kv_cache",
}
_KV_CACHE_FIELDS = {"tiers", "bytes_per_token", "offload_policy"}
_TIER_FIELDS = {"name", "capacity_bytes", "block_size_tokens", "link"}
_STEP_FIELDS = {
    "step_index", "step_id", "schedule_step_sha256", "events",
}
_EVENT_FIELDS = {
    "request_id",
    "action",
    "tier",
    "token_start",
    "token_end",
    "bytes",
    "source_tier",
    "destination_tier",
    "ready_event",
}


def kv_lifecycle_replay_event_payload(event: KVCacheEvent) -> dict[str, object]:
    """Serialize a request-private event in the unchanged v1 replay shape."""

    if not isinstance(event, KVCacheEvent):
        raise TypeError("KV lifecycle replay payload requires a KVCacheEvent")
    if event.action not in (KVCacheAction.LOAD, KVCacheAction.EVICT):
        raise ValueError("KV lifecycle replay accepts only LOAD/EVICT intents")
    if event.source_request_id is not None:
        raise ValueError("KV lifecycle replay cannot encode shared Host sources")
    payload = asdict(event)
    del payload["source_request_id"]
    if set(payload) != _EVENT_FIELDS:
        raise RuntimeError("KV lifecycle replay event payload changed fields")
    return payload


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _require_exact_fields(
    mapping: Mapping[str, object],
    fields: set[str],
    name: str,
) -> None:
    supplied = set(mapping)
    if supplied != fields:
        missing = sorted(fields - supplied)
        unexpected = sorted(supplied - fields)
        raise ValueError(
            f"{name} fields disagree: missing={missing}, "
            f"unexpected={unexpected}")


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _require_sha256(value: object, name: str) -> str:
    digest = _require_non_empty_string(value, name)
    if (len(digest) != 64
            or any(character not in "0123456789abcdef"
                   for character in digest)):
        raise ValueError(f"{name} must be lowercase SHA256 hex")
    return digest


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"KV lifecycle replay contains duplicate key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class KVLifecycleReplayProvenance:
    source_kind: str
    producer: str
    producer_version: str
    source_reference: str

    def __post_init__(self) -> None:
        if self.source_kind not in (
                "framework_observation", "synthetic_fixture"):
            raise ValueError(
                "KV lifecycle replay source_kind must be "
                "framework_observation or synthetic_fixture")
        for name in ("producer", "producer_version", "source_reference"):
            _require_non_empty_string(getattr(self, name), name)


@dataclass(frozen=True)
class KVLifecycleReplayContext:
    request_stream_sha256: str
    model_config_sha256: str
    deployment_id: str
    scheduler_config_sha256: str
    scheduler_runtime_sha256: str
    scheduler_step_count: int
    pool_name: str
    backend_role: str
    kv_cache: KVCacheConfig

    def __post_init__(self) -> None:
        for name in (
                "request_stream_sha256", "model_config_sha256",
                "deployment_id", "scheduler_config_sha256"):
            _require_sha256(getattr(self, name), name)
        _require_sha256(
            self.scheduler_runtime_sha256, "scheduler_runtime_sha256")
        _require_non_negative_int(
            self.scheduler_step_count, "scheduler_step_count")
        if self.scheduler_step_count < 1:
            raise ValueError("scheduler_step_count must be positive")
        _require_non_empty_string(self.pool_name, "pool_name")
        _require_non_empty_string(self.backend_role, "backend_role")
        if not isinstance(self.kv_cache, KVCacheConfig):
            raise TypeError("KV lifecycle replay requires KVCacheConfig")


@dataclass(frozen=True)
class KVLifecycleReplayStep:
    step_index: int
    step_id: str
    schedule_step_sha256: str
    events: Tuple[KVCacheEvent, ...]

    def __post_init__(self) -> None:
        _require_non_negative_int(self.step_index, "step_index")
        _require_non_empty_string(self.step_id, "step_id")
        _require_sha256(
            self.schedule_step_sha256, "schedule_step_sha256")
        if not isinstance(self.events, (list, tuple)):
            raise TypeError("KV lifecycle replay step events must be a list")
        object.__setattr__(self, "events", tuple(self.events))
        if any(not isinstance(event, KVCacheEvent) for event in self.events):
            raise TypeError(
                "KV lifecycle replay events must be KVCacheEvent objects")
        phases = []
        for event in self.events:
            if event.action is KVCacheAction.LOAD:
                phases.append(0)
            elif event.action is KVCacheAction.EVICT:
                phases.append(1)
            else:
                raise ValueError(
                    "KV lifecycle replay accepts only LOAD/EVICT intents")
        if phases != sorted(phases):
            raise ValueError(
                "KV lifecycle replay step must order LOAD before EVICT")


@dataclass(frozen=True)
class KVLifecycleReplay:
    source_path: str
    source_sha256: str
    artifact_id: str
    provenance: KVLifecycleReplayProvenance
    context: KVLifecycleReplayContext
    steps: Tuple[KVLifecycleReplayStep, ...]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.source_path, "source_path")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_non_empty_string(self.artifact_id, "artifact_id")
        if not isinstance(self.provenance, KVLifecycleReplayProvenance):
            raise TypeError(
                "KV lifecycle replay provenance must be canonical")
        if not isinstance(self.context, KVLifecycleReplayContext):
            raise TypeError("KV lifecycle replay context must be canonical")
        if not isinstance(self.steps, (list, tuple)):
            raise TypeError("KV lifecycle replay steps must be a list")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ValueError("KV lifecycle replay requires at least one step")
        if any(not isinstance(step, KVLifecycleReplayStep)
               for step in self.steps):
            raise TypeError("KV lifecycle replay steps must be canonical")
        indices = tuple(step.step_index for step in self.steps)
        expected_indices = tuple(range(self.context.scheduler_step_count))
        if indices != expected_indices:
            raise ValueError(
                "KV lifecycle replay must cover every Scheduler step: "
                f"expected={expected_indices}, actual={indices}")
        expected_step_ids = tuple(
            f"{self.context.pool_name}:vllm:{index}" for index in indices)
        actual_step_ids = tuple(step.step_id for step in self.steps)
        if actual_step_ids != expected_step_ids:
            raise ValueError(
                "KV lifecycle replay step IDs disagree with pool/index: "
                f"expected={expected_step_ids}, actual={actual_step_ids}")
        ready_events = tuple(
            event.ready_event
            for step in self.steps
            for event in step.events
        )
        if len(ready_events) != len(set(ready_events)):
            raise ValueError(
                "KV lifecycle replay ready_event values must be unique")
        self._validate_event_geometry()

    @property
    def cache_config(self) -> KVCacheConfig:
        return self.context.kv_cache

    def _validate_event_geometry(self) -> None:
        tiers = {tier.name.lower(): tier for tier in self.cache_config.tiers}
        if set(tiers) != {"hbm", "host"}:
            raise ValueError(
                "KV lifecycle replay requires exactly hbm and host tiers")
        hbm = tiers["hbm"]
        host = tiers["host"]
        if hbm.name != "hbm" or host.name != "host":
            raise ValueError(
                "KV lifecycle replay tier names must be lowercase")
        if hbm.link is not None or host.link != "pcie":
            raise ValueError(
                "KV lifecycle replay requires HBM and Host over PCIe")
        if hbm.block_size_tokens != host.block_size_tokens:
            raise NotImplementedError(
                "KV lifecycle replay requires equal HBM/Host block size")
        if self.cache_config.offload_policy != "lru":
            raise ValueError(
                "KV lifecycle replay currently requires LRU offload policy")
        block_tokens = hbm.block_size_tokens
        block_bytes = block_tokens * self.cache_config.bytes_per_token
        for step in self.steps:
            for event in step.events:
                if (event.tier != "host"
                        or event.token_start % block_tokens
                        or event.token_end - event.token_start != block_tokens
                        or event.bytes != block_bytes):
                    raise ValueError(
                        "KV lifecycle replay events must cover one aligned "
                        "full Host KV block")
                if event.action is KVCacheAction.LOAD:
                    expected_endpoints = ("host", "hbm")
                elif event.action is KVCacheAction.EVICT:
                    expected_endpoints = ("hbm", "host")
                else:
                    raise RuntimeError(
                        f"unknown replay action: {event.action.value!r}")
                if (event.source_tier, event.destination_tier) != (
                        expected_endpoints):
                    raise ValueError(
                        "KV lifecycle replay event endpoints disagree with "
                        f"{event.action.value}")

    def bind(
        self,
        model: ModelConfig,
        deployment: DeploymentPlan,
        requests: RequestStream,
        backend_config: BackendConfig,
        scheduler_runtime_sha256: str,
    ) -> "KVLifecycleReplayCursor":
        if not isinstance(model, ModelConfig):
            raise TypeError("KV lifecycle replay binding requires ModelConfig")
        if not isinstance(deployment, DeploymentPlan):
            raise TypeError(
                "KV lifecycle replay binding requires DeploymentPlan")
        if not isinstance(requests, RequestStream):
            raise TypeError(
                "KV lifecycle replay binding requires RequestStream")
        if not isinstance(backend_config, BackendConfig):
            raise TypeError(
                "KV lifecycle replay binding requires BackendConfig")
        scheduler_runtime_sha256 = _require_sha256(
            scheduler_runtime_sha256, "scheduler_runtime_sha256")
        actual_context = {
            "request_stream_sha256": request_stream_sha256(requests),
            "model_config_sha256": model.config_sha256,
            "deployment_id": deployment.deployment_id,
            "scheduler_config_sha256": backend_config_sha256(backend_config),
            "scheduler_runtime_sha256": scheduler_runtime_sha256,
            "pool_name": backend_config.pool_name,
            "backend_role": backend_config.role,
        }
        for name, actual in actual_context.items():
            expected = getattr(self.context, name)
            if actual != expected:
                raise ValueError(
                    f"KV lifecycle replay {name} mismatch: "
                    f"expected={expected!r}, actual={actual!r}")
        if self.context.pool_name not in deployment.worker_pools:
            raise ValueError(
                "KV lifecycle replay owner pool is absent from deployment: "
                f"{self.context.pool_name!r}")
        if len(deployment.worker_pools) == 1:
            pass
        elif (deployment.mode == "af_disagg"
              and len(deployment.worker_pools) == 2):
            attention_pools = tuple(
                pool for pool in deployment.worker_pools.values()
                if worker_role_operator_group(pool.role) == "attention"
                and worker_role_phase(pool.role) == "colocated"
            )
            ffn_pools = tuple(
                pool for pool in deployment.worker_pools.values()
                if worker_role_operator_group(pool.role) == "ffn"
                and worker_role_phase(pool.role) == "colocated"
            )
            if len(attention_pools) != 1 or len(ffn_pools) != 1:
                raise NotImplementedError(
                    "AF KV lifecycle replay requires exactly one colocated "
                    "Attention owner and one colocated FFN worker pool")
            if attention_pools[0].name != self.context.pool_name:
                raise ValueError(
                    "AF KV lifecycle replay owner must be the Attention pool")
            if (worker_role_operator_group(backend_config.role) != "attention"
                    or worker_role_phase(backend_config.role) != "colocated"):
                raise ValueError(
                    "AF KV lifecycle replay backend role must be colocated "
                    "Attention")
            if ffn_pools[0].kv_cache is not None:
                raise ValueError(
                    "AF KV lifecycle replay requires the FFN pool to have no "
                    "KV cache")
        else:
            raise NotImplementedError(
                "KV lifecycle replay supports one worker pool or one "
                "standalone AF Attention-owner/FFN pair")
        pool = deployment.worker_pools[self.context.pool_name]
        if pool.kv_cache != self.cache_config:
            raise ValueError(
                "KV lifecycle replay cache config disagrees with deployment")
        if len(pool.ranks) == 1:
            pass
        elif len(pool.ranks) == 2:
            parallelism = pool.parallel_plan.config
            if not (
                deployment.mode == "colocated"
                and len(deployment.worker_pools) == 1
                and pool.hardware_id == "nvidia_a800"
                and pool.ranks == (0, 1)
                and parallelism.worker_count == 2
                and parallelism.tensor_parallel_size == 2
                and parallelism.pipeline_parallel_size == 1
                and parallelism.expert_parallel_size == 1
                and parallelism.sequence_parallel_size == 1
                and parallelism.context_parallel_size == 1
                and parallelism.data_parallel_size == 1
                and len(pool.collectives) == 2
                and {item.operation for item in pool.collectives}
                == {"all_reduce", "all_gather"}
                and all(
                    item.backend == "analytical_roofline_v1"
                    for item in pool.collectives)
                and is_validated_qwen3_tp2_host_kv_model_scope(model)
            ):
                raise NotImplementedError(
                    "multi-rank KV lifecycle replay supports only the "
                    "validated colocated nvidia_a800 dense BF16 Qwen3 TP=2 "
                    "scope")
        else:
            raise NotImplementedError(
                "KV lifecycle replay currently requires one physical rank or "
                "the validated colocated Qwen3 TP=2 owner")
        hbm = next(
            tier for tier in self.cache_config.tiers
            if tier.name == "hbm")
        if (backend_config.kv_block_size_tokens
                != hbm.block_size_tokens):
            raise ValueError(
                "KV lifecycle replay block size disagrees with Scheduler")
        if (backend_config.kv_bytes_per_token
                != self.cache_config.bytes_per_token):
            raise ValueError(
                "KV lifecycle replay bytes/token disagrees with Scheduler")
        block_bytes = (
            hbm.block_size_tokens * self.cache_config.bytes_per_token)
        if (backend_config.num_kv_blocks * block_bytes
                != hbm.capacity_bytes):
            raise ValueError(
                "KV lifecycle replay HBM capacity disagrees with Scheduler "
                "num_kv_blocks")
        requests_by_id = {
            request.request_id: request for request in requests.requests
        }
        for step in self.steps:
            for event in step.events:
                if event.request_id not in requests_by_id:
                    raise ValueError(
                        "KV lifecycle replay references unknown request: "
                        f"{event.request_id!r}")
                request = requests_by_id[event.request_id]
                total_tokens = request.input_tokens + request.output_tokens
                block_tokens = hbm.block_size_tokens
                allocated_extent = (
                    math.ceil(total_tokens / block_tokens) * block_tokens)
                if event.token_end > allocated_extent:
                    raise ValueError(
                        "KV lifecycle replay block exceeds request token "
                        f"extent: request={event.request_id!r}, "
                        f"token_end={event.token_end}, "
                        f"allocated_extent={allocated_extent}")
        return KVLifecycleReplayCursor(self)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "provenance": asdict(self.provenance),
            "context": {
                "request_stream_sha256": (
                    self.context.request_stream_sha256),
                "model_config_sha256": self.context.model_config_sha256,
                "deployment_id": self.context.deployment_id,
                "scheduler_config_sha256": (
                    self.context.scheduler_config_sha256),
                "scheduler_runtime_sha256": (
                    self.context.scheduler_runtime_sha256),
                "scheduler_step_count": self.context.scheduler_step_count,
                "pool_name": self.context.pool_name,
                "backend_role": self.context.backend_role,
                "kv_cache": asdict(self.context.kv_cache),
            },
            "steps": tuple({
                "step_index": step.step_index,
                "step_id": step.step_id,
                "schedule_step_sha256": step.schedule_step_sha256,
                "events": tuple(kv_lifecycle_replay_event_payload(event)
                                for event in step.events),
            } for step in self.steps),
        }


class KVLifecycleReplayCursor:
    """Consume a bound replay exactly once in native Scheduler step order."""

    def __init__(self, replay: KVLifecycleReplay) -> None:
        if not isinstance(replay, KVLifecycleReplay):
            raise TypeError("KV lifecycle cursor requires a canonical replay")
        self.replay = replay
        self._next_scheduler_step = 0

    def events_for_step(self, backend_step: BackendStep) -> Tuple[KVCacheEvent, ...]:
        if not isinstance(backend_step, BackendStep):
            raise TypeError("KV lifecycle replay requires BackendStep")
        if self._next_scheduler_step >= len(self.replay.steps):
            raise ValueError(
                "KV lifecycle replay observed more Scheduler steps than "
                "the versioned artifact")
        replay_step = self.replay.steps[self._next_scheduler_step]
        actual_step_id = backend_step.schedule.step_id
        if actual_step_id != replay_step.step_id:
            raise ValueError(
                "KV lifecycle replay Scheduler step order mismatch: "
                f"expected={replay_step.step_id!r}, "
                f"actual={actual_step_id!r}")
        actual_step_sha256 = schedule_step_sha256(backend_step.schedule)
        if actual_step_sha256 != replay_step.schedule_step_sha256:
            raise ValueError(
                "KV lifecycle replay ScheduleStep SHA256 mismatch: "
                f"step_id={actual_step_id!r}, "
                f"expected={replay_step.schedule_step_sha256!r}, "
                f"actual={actual_step_sha256!r}")
        self._next_scheduler_step += 1
        return replay_step.events

    def assert_exhausted(self) -> None:
        if self._next_scheduler_step != len(self.replay.steps):
            remaining = tuple(
                step.step_id
                for step in self.replay.steps[self._next_scheduler_step:]
            )
            raise RuntimeError(
                "KV lifecycle replay contains unconsumed steps: "
                f"{remaining!r}")


def load_kv_lifecycle_replay(path: str | Path) -> KVLifecycleReplay:
    replay_path = Path(path)
    payload = replay_path.read_bytes()
    try:
        root = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid KV lifecycle replay JSON: {replay_path}") from error
    root = _require_mapping(root, "KV lifecycle replay")
    _require_exact_fields(root, _ROOT_FIELDS, "KV lifecycle replay")
    if root["schema_version"] != 1:
        raise ValueError(
            "unsupported KV lifecycle replay schema_version: "
            f"{root['schema_version']!r}")

    provenance_raw = _require_mapping(
        root["provenance"], "KV lifecycle replay provenance")
    _require_exact_fields(
        provenance_raw, _PROVENANCE_FIELDS,
        "KV lifecycle replay provenance")
    provenance = KVLifecycleReplayProvenance(
        source_kind=provenance_raw["source_kind"],
        producer=provenance_raw["producer"],
        producer_version=provenance_raw["producer_version"],
        source_reference=provenance_raw["source_reference"],
    )

    context_raw = _require_mapping(
        root["context"], "KV lifecycle replay context")
    _require_exact_fields(
        context_raw, _CONTEXT_FIELDS, "KV lifecycle replay context")
    cache_raw = _require_mapping(
        context_raw["kv_cache"], "KV lifecycle replay cache config")
    _require_exact_fields(
        cache_raw, _KV_CACHE_FIELDS, "KV lifecycle replay cache config")
    tiers_raw = cache_raw["tiers"]
    if not isinstance(tiers_raw, list):
        raise TypeError("KV lifecycle replay cache tiers must be a list")
    tiers = []
    for index, tier_value in enumerate(tiers_raw):
        tier_raw = _require_mapping(
            tier_value, f"KV lifecycle replay tier {index}")
        _require_exact_fields(
            tier_raw, _TIER_FIELDS, f"KV lifecycle replay tier {index}")
        tiers.append(KVCacheTierConfig(
            name=tier_raw["name"],
            capacity_bytes=tier_raw["capacity_bytes"],
            block_size_tokens=tier_raw["block_size_tokens"],
            link=tier_raw["link"],
        ))
    cache_config = KVCacheConfig(
        tiers=tuple(tiers),
        bytes_per_token=cache_raw["bytes_per_token"],
        offload_policy=cache_raw["offload_policy"],
    )
    context = KVLifecycleReplayContext(
        request_stream_sha256=_require_sha256(
            context_raw["request_stream_sha256"],
            "request_stream_sha256"),
        model_config_sha256=_require_sha256(
            context_raw["model_config_sha256"], "model_config_sha256"),
        deployment_id=_require_sha256(
            context_raw["deployment_id"], "deployment_id"),
        scheduler_config_sha256=_require_sha256(
            context_raw["scheduler_config_sha256"],
            "scheduler_config_sha256"),
        scheduler_runtime_sha256=_require_sha256(
            context_raw["scheduler_runtime_sha256"],
            "scheduler_runtime_sha256"),
        scheduler_step_count=_require_non_negative_int(
            context_raw["scheduler_step_count"],
            "scheduler_step_count"),
        pool_name=_require_non_empty_string(
            context_raw["pool_name"], "pool_name"),
        backend_role=_require_non_empty_string(
            context_raw["backend_role"], "backend_role"),
        kv_cache=cache_config,
    )

    steps_raw = root["steps"]
    if not isinstance(steps_raw, list):
        raise TypeError("KV lifecycle replay steps must be a list")
    steps = []
    for step_position, step_value in enumerate(steps_raw):
        step_raw = _require_mapping(
            step_value, f"KV lifecycle replay step {step_position}")
        _require_exact_fields(
            step_raw, _STEP_FIELDS,
            f"KV lifecycle replay step {step_position}")
        events_raw = step_raw["events"]
        if not isinstance(events_raw, list):
            raise TypeError(
                "KV lifecycle replay step events must be a list")
        events = []
        for event_position, event_value in enumerate(events_raw):
            event_raw = _require_mapping(
                event_value,
                f"KV lifecycle replay event {step_position}:"
                f"{event_position}")
            _require_exact_fields(
                event_raw, _EVENT_FIELDS,
                f"KV lifecycle replay event {step_position}:"
                f"{event_position}")
            if event_raw["action"] not in ("load", "evict"):
                raise ValueError(
                    "KV lifecycle replay accepts only LOAD/EVICT intents")
            events.append(KVCacheEvent(
                request_id=event_raw["request_id"],
                action=event_raw["action"],
                tier=event_raw["tier"],
                token_start=event_raw["token_start"],
                token_end=event_raw["token_end"],
                bytes=event_raw["bytes"],
                source_tier=event_raw["source_tier"],
                destination_tier=event_raw["destination_tier"],
                ready_event=event_raw["ready_event"],
            ))
        steps.append(KVLifecycleReplayStep(
            step_index=_require_non_negative_int(
                step_raw["step_index"], "step_index"),
            step_id=_require_non_empty_string(
                step_raw["step_id"], "step_id"),
            schedule_step_sha256=_require_sha256(
                step_raw["schedule_step_sha256"],
                "schedule_step_sha256"),
            events=tuple(events),
        ))
    return KVLifecycleReplay(
        source_path=str(replay_path.resolve()),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_id=_require_non_empty_string(
            root["artifact_id"], "artifact_id"),
        provenance=provenance,
        context=context,
        steps=tuple(steps),
    )
