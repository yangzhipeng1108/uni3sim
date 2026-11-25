"""Run-time workload quantities that are not static model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import functools
import hashlib
import json
import math
import re
import threading
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Optional, Tuple

from .expert_frequency import ExpertFrequencyProfile
from .graph_padding import GraphExecutionConfig


# Declared TARGET process state, not the environment of this simulator.
# Installed MACA native allocator; PYTORCH_CUDA_ALLOC_CONF,
# PYTORCH_NO_CUDA_MEMORY_CACHING and PYTORCH_DISABLE_UNROLL_FLOAT_OPT are
# unset before their process-lifetime cached initialization.
MACA_COPY_RUNTIME_SOURCE = "torch_2_4_maca_3_7_2_native_default_unset_env"


_ROUTING_REPLAY_ROOT_FIELDS = {
    "schema_version",
    "artifact_id",
    "canonical_payload_sha256",
    "provenance",
    "context",
    "records",
}
_ROUTING_REPLAY_PROVENANCE_FIELDS = {
    "producer",
    "vllm_version",
    "vllm_source_commit",
    "vllm_package_manifest_sha256",
    "model_config_sha256",
}
_ROUTING_REPLAY_CONTEXT_FIELDS = {
    "model_config_sha256",
    "request_stream_sha256",
    "scheduler_config_sha256",
    "pool_name",
    "tensor_parallel_size",
    "expert_parallel_size",
    "expert_rank_group",
    "expert_placement_strategy",
    "num_experts",
    "topk",
}
_ROUTING_REPLAY_RECORD_FIELDS = {
    "step_id",
    "layer_index",
    "moe_invocation_id",
    "schedule_step_sha256",
    "tokens",
    "tokens_per_expert",
}
_SEEDED_ROUTING_ALGORITHM = (
    "sha256_weighted_rejection_topk_without_replacement_model_step_v2")
_PROFILE_ROUTING_ALGORITHM = (
    "sha256_profile_weighted_topk_without_replacement_model_step_v1")
_UNIFORM_ROUTING_DISTRIBUTION = "maximum_entropy_uniform_topk_v1"
_SKEW_ROUTING_DISTRIBUTION = "stable_hotset_skew_sensitivity_v1"
_PROFILE_ROUTING_DISTRIBUTION = "model_profile_layer_mean_v1"
# Per-MoE-layer realization of the same imported profile: each MoE layer's
# histogram is drawn from that layer's own row (the router of every layer is
# a separate gate); the MTP layer, which the profile does not cover, draws
# from the model-level mean row.  One histogram is shared by every step with
# the same MoE layer and token count (the price depends only on the
# histogram), so it is drawn once per (layer, tokens) signature.
_PROFILE_LAYER_ROUTING_ALGORITHM = (
    "sha256_seeded_pcg64_gumbel_topk_per_moe_layer_v1")
_PROFILE_LAYER_ROUTING_DISTRIBUTION = "model_profile_per_moe_layer_v1"
_PROFILE_ROUTING_REALIZATIONS = {
    _PROFILE_ROUTING_DISTRIBUTION: (
        _PROFILE_ROUTING_ALGORITHM, "model_step_block_signature"),
    _PROFILE_LAYER_ROUTING_DISTRIBUTION: (
        _PROFILE_LAYER_ROUTING_ALGORITHM, "model_moe_layer_token_count"),
}
_REPLAY_ROUTING_DISTRIBUTION = "captured_expert_histogram_v1"
_REPLAY_ROUTING_ALGORITHM = "expert_histogram_replay_v1"
_ROUTING_PROVENANCE_FIELDS = {
    "schema_version",
    "source_kind",
    "source_id",
    "algorithm",
    "distribution",
    "hot_expert_count",
    "hot_expert_weight",
    "root_seed",
    "policy_descriptor_sha256",
    "replay_source_sha256",
    "replay_canonical_payload_sha256",
    "record_sha256",
    "step_id",
    "layer_index",
    "moe_invocation_id",
    "schedule_step_sha256",
    "mechanism_source",
    "validation_status",
}
_PROFILE_ROUTING_PROVENANCE_FIELDS = _ROUTING_PROVENANCE_FIELDS | {
    "profile_sha256",
    "profile_manifest_sha256",
    "sharing_scope",
    "execution_signature_sha256",
}


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    normalized = {}
    for key, value in pairs:
        if key in normalized:
            raise ValueError(
                f"duplicate expert routing replay JSON field: {key!r}")
        normalized[key] = value
    return normalized


def _require_exact_fields(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expert routing replay {label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"expert routing replay {label} fields must be exactly "
            f"{sorted(expected)}, got {sorted(value)}")
    return value


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"expert routing replay {name} must be a non-empty string")
    return value


def _require_integer(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expert routing replay {name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"expert routing replay {name} must be positive")
    if not positive and value < 0:
        raise ValueError(
            f"expert routing replay {name} cannot be negative")
    return value


_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require_sha256(name: str, value: object) -> str:
    normalized = _require_string(name, value)
    if (len(normalized) != 64
            or (_LOWERCASE_SHA256.fullmatch(normalized) is None if type(normalized) is str
                else any(character not in "0123456789abcdef" for character in normalized))):
        raise ValueError(
            f"expert routing replay {name} must be a lowercase SHA256")
    return normalized


def _validate_expert_histogram(
    tokens_per_expert: Tuple[int, ...],
    *,
    tokens: int,
    num_experts: int,
    topk: int,
    label: str,
) -> Tuple[int, ...]:
    counts = tuple(tokens_per_expert)
    if len(counts) != num_experts:
        raise ValueError(f"{label} count length must match num_experts")
    if any(isinstance(count, bool) or not isinstance(count, int)
           or count < 0 for count in counts):
        raise ValueError(f"{label} counts must be non-negative integers")
    expected_routed_tokens = tokens * topk
    if sum(counts) != expected_routed_tokens:
        raise ValueError(
            f"{label} counts must sum to tokens * top-k: "
            f"{sum(counts)} != {expected_routed_tokens}")
    if any(count > tokens for count in counts):
        raise ValueError(
            f"{label} count cannot exceed tokens for any expert")
    return counts


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExpertRoutingReplayProvenance:
    producer: str
    vllm_version: str
    vllm_source_commit: str
    vllm_package_manifest_sha256: str
    model_config_sha256: str

    def __post_init__(self) -> None:
        if self.producer != "vllm_model_runner_observation":
            raise ValueError(
                "expert routing replay producer must be "
                "'vllm_model_runner_observation'")
        _require_string("vllm_version", self.vllm_version)
        commit = _require_string(
            "vllm_source_commit", self.vllm_source_commit)
        if (len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(
                "expert routing replay vllm_source_commit must be a "
                "lowercase 40-character Git commit")
        _require_sha256(
            "vllm_package_manifest_sha256",
            self.vllm_package_manifest_sha256,
        )
        _require_sha256(
            "model_config_sha256", self.model_config_sha256)

    def manifest(self) -> dict[str, object]:
        return {
            "producer": self.producer,
            "vllm_version": self.vllm_version,
            "vllm_source_commit": self.vllm_source_commit,
            "vllm_package_manifest_sha256": (
                self.vllm_package_manifest_sha256),
            "model_config_sha256": self.model_config_sha256,
        }


@dataclass(frozen=True)
class ExpertRoutingReplayContext:
    model_config_sha256: str
    request_stream_sha256: str
    scheduler_config_sha256: str
    pool_name: str
    tensor_parallel_size: int
    expert_parallel_size: int
    expert_rank_group: Tuple[int, ...]
    expert_placement_strategy: str
    num_experts: int
    topk: int

    def __post_init__(self) -> None:
        _require_sha256("model_config_sha256", self.model_config_sha256)
        _require_sha256(
            "request_stream_sha256", self.request_stream_sha256)
        _require_sha256(
            "scheduler_config_sha256", self.scheduler_config_sha256)
        _require_string("pool_name", self.pool_name)
        for name in (
                "tensor_parallel_size", "expert_parallel_size",
                "num_experts", "topk"):
            _require_integer(name, getattr(self, name), positive=True)
        if not isinstance(self.expert_rank_group, (list, tuple)):
            raise TypeError(
                "expert routing replay expert_rank_group must be a sequence")
        rank_group = tuple(self.expert_rank_group)
        if len(rank_group) != self.expert_parallel_size:
            raise ValueError(
                "expert routing replay expert rank group size disagrees with "
                "expert_parallel_size")
        if any(isinstance(rank, bool) or not isinstance(rank, int)
               for rank in rank_group):
            raise TypeError(
                "expert routing replay expert ranks must be integers")
        if any(rank < 0 for rank in rank_group):
            raise ValueError(
                "expert routing replay expert ranks cannot be negative")
        if len(rank_group) != len(set(rank_group)):
            raise ValueError(
                "expert routing replay expert ranks must be unique")
        object.__setattr__(self, "expert_rank_group", rank_group)
        if self.expert_placement_strategy != "linear":
            raise ValueError(
                "expert routing replay currently requires linear expert "
                "placement")
        if self.topk > self.num_experts:
            raise ValueError(
                "expert routing replay topk cannot exceed num_experts")

    def manifest(self) -> dict[str, object]:
        return {
            "model_config_sha256": self.model_config_sha256,
            "request_stream_sha256": self.request_stream_sha256,
            "scheduler_config_sha256": self.scheduler_config_sha256,
            "pool_name": self.pool_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "expert_rank_group": self.expert_rank_group,
            "expert_placement_strategy": self.expert_placement_strategy,
            "num_experts": self.num_experts,
            "topk": self.topk,
        }


@dataclass(frozen=True)
class ExpertRoutingReplayRecord:
    step_id: str
    layer_index: int
    moe_invocation_id: str
    schedule_step_sha256: str
    tokens: int
    tokens_per_expert: Tuple[int, ...]

    def __post_init__(self) -> None:
        _require_string("record step_id", self.step_id)
        _require_integer("record layer_index", self.layer_index, positive=False)
        _require_string("record moe_invocation_id", self.moe_invocation_id)
        expected_invocation_id = (
            f"{self.step_id}:layer-{self.layer_index}:moe")
        if self.moe_invocation_id != expected_invocation_id:
            raise ValueError(
                "expert routing replay record invocation ID disagrees with "
                "its step/layer identity: "
                f"{self.moe_invocation_id!r} != {expected_invocation_id!r}")
        _require_sha256(
            "record schedule_step_sha256", self.schedule_step_sha256)
        _require_integer("record tokens", self.tokens, positive=True)
        if not isinstance(self.tokens_per_expert, (list, tuple)):
            raise TypeError(
                "expert routing replay tokens_per_expert must be a sequence")
        counts = tuple(self.tokens_per_expert)
        if not counts:
            raise ValueError(
                "expert routing replay tokens_per_expert cannot be empty")
        if any(isinstance(count, bool) or not isinstance(count, int)
               for count in counts):
            raise TypeError(
                "expert routing replay token counts must be integers")
        if any(count < 0 for count in counts):
            raise ValueError(
                "expert routing replay token counts cannot be negative")
        object.__setattr__(self, "tokens_per_expert", counts)

    def manifest(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "layer_index": self.layer_index,
            "moe_invocation_id": self.moe_invocation_id,
            "schedule_step_sha256": self.schedule_step_sha256,
            "tokens": self.tokens,
            "tokens_per_expert": self.tokens_per_expert,
        }

    @property
    def record_sha256(self) -> str:
        return _canonical_json_sha256(self.manifest())


@dataclass(frozen=True, init=False)
class ExpertRoutingReplay:
    source_path: str
    source_sha256: str
    canonical_payload_sha256: str
    artifact_id: str
    provenance: ExpertRoutingReplayProvenance
    context: ExpertRoutingReplayContext
    records: Tuple[ExpertRoutingReplayRecord, ...]

    @classmethod
    def _from_loaded_payload(
        cls,
        *,
        source_path: str,
        source_sha256: str,
        canonical_payload_sha256: str,
        artifact_id: str,
        provenance: ExpertRoutingReplayProvenance,
        context: ExpertRoutingReplayContext,
        records: Tuple[ExpertRoutingReplayRecord, ...],
    ) -> "ExpertRoutingReplay":
        instance = object.__new__(cls)
        object.__setattr__(instance, "source_path", source_path)
        object.__setattr__(instance, "source_sha256", source_sha256)
        object.__setattr__(
            instance,
            "canonical_payload_sha256",
            canonical_payload_sha256,
        )
        object.__setattr__(instance, "artifact_id", artifact_id)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(instance, "context", context)
        object.__setattr__(instance, "records", records)
        instance._validate()
        return instance

    def _validate(self) -> None:
        _require_string("source_path", self.source_path)
        _require_sha256("source_sha256", self.source_sha256)
        _require_sha256(
            "canonical_payload_sha256", self.canonical_payload_sha256)
        _require_string("artifact_id", self.artifact_id)
        if not isinstance(self.provenance, ExpertRoutingReplayProvenance):
            raise TypeError(
                "expert routing replay provenance has the wrong type")
        if not isinstance(self.context, ExpertRoutingReplayContext):
            raise TypeError("expert routing replay context has the wrong type")
        if self.provenance.model_config_sha256 != (
                self.context.model_config_sha256):
            raise ValueError(
                "expert routing replay provenance and context model hashes "
                "disagree")
        if not isinstance(self.records, (list, tuple)):
            raise TypeError("expert routing replay records must be a sequence")
        records = tuple(self.records)
        if not records or any(
                not isinstance(record, ExpertRoutingReplayRecord)
                for record in records):
            raise TypeError("expert routing replay requires routing records")
        keys = tuple(
            (record.step_id, record.layer_index) for record in records)
        if len(keys) != len(set(keys)):
            raise ValueError(
                "expert routing replay contains duplicate step/layer records")
        invocation_ids = tuple(
            record.moe_invocation_id for record in records)
        if len(invocation_ids) != len(set(invocation_ids)):
            raise ValueError(
                "expert routing replay contains duplicate invocation IDs")
        expected_step_prefix = f"{self.context.pool_name}:vllm:"
        for record in records:
            if not record.step_id.startswith(expected_step_prefix):
                raise ValueError(
                    "expert routing replay step ID disagrees with its "
                    f"Scheduler pool: {record.step_id!r}")
            step_suffix = record.step_id[len(expected_step_prefix):]
            if not step_suffix.isdigit():
                raise ValueError(
                    "expert routing replay step ID must end in a non-negative "
                    "Scheduler step index")
            _validate_expert_histogram(
                record.tokens_per_expert,
                tokens=record.tokens,
                num_experts=self.context.num_experts,
                topk=self.context.topk,
                label="expert routing replay",
            )
        object.__setattr__(self, "records", records)
        expected_payload_sha256 = expert_routing_replay_payload_sha256(
            self.artifact_id,
            self.provenance,
            self.context,
            records,
        )
        if self.canonical_payload_sha256 != expected_payload_sha256:
            raise ValueError(
                "expert routing replay canonical payload SHA256 mismatch: "
                f"declared={self.canonical_payload_sha256}, "
                f"actual={expected_payload_sha256}")

    def record_for(
        self,
        step_id: str,
        layer_index: int,
    ) -> ExpertRoutingReplayRecord:
        matches = tuple(
            record for record in self.records
            if (record.step_id == step_id
                and record.layer_index == layer_index)
        )
        if len(matches) != 1:
            raise ValueError(
                "expert routing replay is missing the exact step/layer "
                f"record: step_id={step_id!r}, layer_index={layer_index}")
        return matches[0]

    def assert_consumed(self, invocation_ids: Tuple[str, ...]) -> None:
        if not isinstance(invocation_ids, (list, tuple)):
            raise TypeError(
                "consumed expert routing invocation IDs must be a sequence")
        consumed = tuple(invocation_ids)
        if any(not isinstance(value, str) or not value for value in consumed):
            raise ValueError(
                "consumed expert routing invocation IDs must be non-empty")
        if len(consumed) != len(set(consumed)):
            raise ValueError(
                "consumed expert routing invocation IDs must be unique")
        expected = tuple(record.moe_invocation_id for record in self.records)
        if set(consumed) != set(expected):
            missing = sorted(set(expected) - set(consumed))
            extra = sorted(set(consumed) - set(expected))
            raise RuntimeError(
                "expert routing replay coverage disagrees with the executed "
                f"MoE invocations: missing={missing}, extra={extra}")

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "canonical_payload_sha256": self.canonical_payload_sha256,
            "provenance": self.provenance.manifest(),
            "context": self.context.manifest(),
            "record_count": len(self.records),
            "record_ids": tuple(
                record.moe_invocation_id for record in self.records),
        }


def expert_routing_replay_payload_sha256(
    artifact_id: str,
    provenance: ExpertRoutingReplayProvenance,
    context: ExpertRoutingReplayContext,
    records: Tuple[ExpertRoutingReplayRecord, ...],
) -> str:
    """Hash normalized replay semantics, excluding the declared digest."""

    _require_string("artifact_id", artifact_id)
    if not isinstance(provenance, ExpertRoutingReplayProvenance):
        raise TypeError(
            "expert routing replay payload requires canonical provenance")
    if not isinstance(context, ExpertRoutingReplayContext):
        raise TypeError("expert routing replay payload requires context")
    if not isinstance(records, (list, tuple)):
        raise TypeError("expert routing replay payload records must be a sequence")
    normalized_records = tuple(records)
    if any(not isinstance(record, ExpertRoutingReplayRecord)
           for record in normalized_records):
        raise TypeError(
            "expert routing replay payload records must be canonical")
    payload = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "provenance": provenance.manifest(),
        "context": context.manifest(),
        "records": tuple(
            record.manifest()
            for record in sorted(
                normalized_records,
                key=lambda record: (record.step_id, record.layer_index),
            )
        ),
    }
    return _canonical_json_sha256(payload)


def load_expert_routing_replay(path: str | Path) -> ExpertRoutingReplay:
    """Read and fully validate one immutable expert-routing artifact."""

    source_path = Path(path)
    payload = source_path.read_bytes()

    def reject_constant(value: str) -> object:
        raise ValueError(
            f"expert routing replay JSON constant is invalid: {value}")

    raw = json.loads(
        payload,
        object_pairs_hook=_strict_json_object,
        parse_constant=reject_constant,
    )
    root = _require_exact_fields(
        raw, _ROUTING_REPLAY_ROOT_FIELDS, "root")
    schema_version = _require_integer(
        "schema_version", root["schema_version"], positive=True)
    if schema_version != 1:
        raise ValueError("expert routing replay schema_version must be 1")
    provenance_raw = _require_exact_fields(
        root["provenance"],
        _ROUTING_REPLAY_PROVENANCE_FIELDS,
        "provenance",
    )
    context_raw = _require_exact_fields(
        root["context"], _ROUTING_REPLAY_CONTEXT_FIELDS, "context")
    records_raw = root["records"]
    if not isinstance(records_raw, list) or not records_raw:
        raise TypeError(
            "expert routing replay records must be a non-empty list")
    records = []
    for record_index, record_raw in enumerate(records_raw):
        record = _require_exact_fields(
            record_raw,
            _ROUTING_REPLAY_RECORD_FIELDS,
            f"records[{record_index}]",
        )
        records.append(ExpertRoutingReplayRecord(**record))
    return ExpertRoutingReplay._from_loaded_payload(
        source_path=str(source_path.resolve()),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        canonical_payload_sha256=root["canonical_payload_sha256"],
        artifact_id=root["artifact_id"],
        provenance=ExpertRoutingReplayProvenance(**provenance_raw),
        context=ExpertRoutingReplayContext(**context_raw),
        records=tuple(records),
    )


class _ValidatedRoutingProvenance(Mapping):
    """Owner-created immutable public ledger plus its exact checked context."""

    __slots__ = ("_data", "_proof")

    def __init__(self):
        raise TypeError("routing certificates require complete owner validation")

    def __setattr__(self, name, value):
        raise TypeError("validated routing provenance is immutable")

    def __delattr__(self, name):
        raise TypeError("validated routing provenance is immutable")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def copy(self):
        return dict(self._data)

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        # Transport only public values. The receiver must validate again.
        return dict, (dict(self._data),)


def _routing_certificate_contexts_native(records):
    """Validate one synchronous native batch; never export a reusable guard.

    False declines optimization. Callers retain ordinary full validation for
    unknown mappings, subclasses, changed behavior or mismatching contexts.
    There are no caller callbacks between the behavior check and proof reads.
    """
    if type(records) is not tuple:
        return False
    if _ValidatedRoutingProvenance is not _VALIDATED_ROUTING_PROVENANCE_TYPE:
        return False
    if not all(globals()[name] is function and function.__code__ is code
            and function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults
            for name, function, code, defaults, kwdefaults in _ROUTING_CERTIFICATE_VALIDATION_SOURCE):
        return False
    if not all(getattr(_ValidatedRoutingProvenance, name) is function and function.__code__ is code
            for name, function, code in _ROUTING_CERTIFICATE_MAPPING_SOURCE):
        return False
    if dict(vars(_ValidatedRoutingProvenance)) != _ROUTING_CERTIFICATE_CLASS_SOURCE:
        return False
    for record in records:
        if type(record) is not tuple or len(record) != 6:
            return False
        provenance, counts, tokens, num_experts, topk, invocation = record
        if (type(provenance) is not _VALIDATED_ROUTING_PROVENANCE_TYPE
                or type(counts) is not tuple
                or type(tokens) is not int or type(num_experts) is not int
                or type(topk) is not int or type(invocation) is not str):
            return False
        proof = provenance._proof
        if (type(proof) is not tuple or len(proof) != 3
                or type(proof[2]) is not tuple or len(proof[2]) != 4
                or provenance._data is not proof[0] or counts is not proof[1]
                or type(proof[2][0]) is not int or type(proof[2][1]) is not int
                or type(proof[2][2]) is not int or type(proof[2][3]) is not str
                or (tokens, num_experts, topk, invocation) != proof[2]):
            return False
    return True


def _routing_certificate_context(provenance, counts, tokens, num_experts, topk, invocation):
    if (type(provenance) is not _VALIDATED_ROUTING_PROVENANCE_TYPE
            or _ValidatedRoutingProvenance is not _VALIDATED_ROUTING_PROVENANCE_TYPE
            or type(counts) is not tuple
            or any(type(value) is not int for value in (tokens, num_experts, topk))
            or type(invocation) is not str):
        return False
    if not all(globals()[name] is function and function.__code__ is code
            and function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults
            for name, function, code, defaults, kwdefaults in _ROUTING_CERTIFICATE_VALIDATION_SOURCE):
        return False
    if not all(getattr(_ValidatedRoutingProvenance, name) is function and function.__code__ is code
            for name, function, code in _ROUTING_CERTIFICATE_MAPPING_SOURCE):
        return False
    if dict(vars(_ValidatedRoutingProvenance)) != _ROUTING_CERTIFICATE_CLASS_SOURCE:
        return False
    proof = provenance._proof
    return (type(proof) is tuple and len(proof) == 3
        and type(proof[2]) is tuple and len(proof[2]) == 4
        and provenance._data is proof[0] and counts is proof[1]
        and (tokens, num_experts, topk, invocation) == proof[2])


def _certify_routing_provenance(provenance, counts, tokens, num_experts, topk, invocation):
    values = dict(provenance)
    if (type(counts) is not tuple or any(type(count) is not int for count in counts)
            or any(type(value) is not int for value in (tokens, num_experts, topk))
            or type(invocation) is not str
            or any(type(key) is not str or type(value) not in (str, int, type(None))
                for key, value in values.items())):
        return MappingProxyType(values)
    return _construct_native_routing_provenance(values, counts, tokens, num_experts, topk, invocation)


def _construct_native_routing_provenance(values, counts, tokens, num_experts, topk, invocation):
    """Owner-only immutable constructor after the caller's complete native proof."""
    data = MappingProxyType(values)
    result = object.__new__(_ValidatedRoutingProvenance)
    object.__setattr__(result, "_data", data)
    object.__setattr__(result, "_proof", (data, counts, (tokens, num_experts, topk, invocation)))
    return result


@dataclass(frozen=True)
class ExpertRoutingResolution:
    tokens_per_expert: Tuple[int, ...]
    routing_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens_per_expert, (list, tuple)):
            raise TypeError("resolved expert routing counts must be a sequence")
        counts = tuple(self.tokens_per_expert)
        if not counts or any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in counts):
            raise TypeError(
                "resolved expert routing counts must contain integers")
        if any(count < 0 for count in counts):
            raise ValueError(
                "resolved expert routing counts cannot be negative")
        if not isinstance(self.routing_provenance, Mapping):
            raise TypeError("resolved expert routing provenance must be a mapping")
        object.__setattr__(self, "tokens_per_expert", counts)
        provenance = self.routing_provenance
        if type(provenance) is _ValidatedRoutingProvenance:
            proof = provenance._proof
            if (type(proof) is tuple and len(proof) == 3
                    and type(proof[2]) is tuple and len(proof[2]) == 4
                    and _routing_certificate_context(provenance, counts, *proof[2])):
                object.__setattr__(self, "routing_provenance", provenance)
                return
        object.__setattr__(self, "routing_provenance", MappingProxyType(dict(provenance)))


def _validate_routing_invocation(step_id, layer_index, moe_invocation_id):
    if layer_index is not None and (
            isinstance(layer_index, bool) or not isinstance(layer_index, int)
            or layer_index < 0):
        raise ValueError(
            "FUSED_MOE routing layer_index must be non-negative or null")
    if layer_index is None:
        if (not moe_invocation_id.startswith(f"{step_id}:mtp-pass-")
                or not moe_invocation_id.endswith(":moe")):
            raise ValueError(
                "FUSED_MOE MTP routing invocation disagrees with its step")
    else:
        expected_invocation_id = f"{step_id}:layer-{layer_index}:moe"
        if moe_invocation_id != expected_invocation_id:
            raise ValueError(
                "FUSED_MOE routing invocation disagrees with step/layer: "
                f"{moe_invocation_id!r} != {expected_invocation_id!r}")


def validate_expert_routing_provenance(
    provenance: Mapping[str, object],
    *,
    tokens_per_expert: Tuple[int, ...],
    tokens: int,
    num_experts: int,
    topk: int,
    moe_invocation_id: str,
) -> Mapping[str, object]:
    """Validate graph-level routing provenance against its MoE semantics."""

    if _routing_certificate_context(provenance, tokens_per_expert, tokens, num_experts, topk, moe_invocation_id):
        return provenance
    if not isinstance(provenance, Mapping):
        raise TypeError("FUSED_MOE routing provenance must be a mapping")
    # Validate the same independently owned snapshot that certification sees.
    # An unknown Mapping may expose different lookup and copy semantics.
    provenance = dict(provenance)
    if "source_kind" not in provenance:
        raise ValueError("FUSED_MOE routing provenance lacks source_kind")
    source_kind = provenance["source_kind"]
    is_profile = source_kind == "profile_observation"
    if is_profile:
        expected_fields = _PROFILE_ROUTING_PROVENANCE_FIELDS
        profile_sha_fields = (
            "profile_sha256",
            "profile_manifest_sha256",
            "execution_signature_sha256",
        )
    else:
        expected_fields = _ROUTING_PROVENANCE_FIELDS
        profile_sha_fields = ()
    if set(provenance) != expected_fields:
        raise ValueError(
            "FUSED_MOE routing provenance fields disagree with the canonical "
            "contract")
    if provenance["schema_version"] != 2:
        raise ValueError("FUSED_MOE routing provenance schema_version must be 2")
    for name in (
            "source_kind", "source_id", "algorithm", "record_sha256",
            "step_id", "moe_invocation_id", "schedule_step_sha256",
            "mechanism_source", "validation_status"):
        if not isinstance(provenance[name], str) or not provenance[name]:
            raise ValueError(
                f"FUSED_MOE routing provenance {name} must be non-empty")
    sha_fields = (
            "record_sha256", "schedule_step_sha256",
            "policy_descriptor_sha256", "replay_source_sha256",
            "replay_canonical_payload_sha256") + profile_sha_fields
    for name in sha_fields:
        value = provenance[name]
        if value is not None:
            _require_sha256(f"routing provenance {name}", value)
    if provenance["moe_invocation_id"] != moe_invocation_id:
        raise ValueError(
            "FUSED_MOE routing provenance invocation ID disagrees with node")
    step_id, layer_index = provenance["step_id"], provenance["layer_index"]
    _validate_routing_invocation(step_id, layer_index, moe_invocation_id)
    if (isinstance(tokens, bool) or not isinstance(tokens, int)
            or tokens <= 0):
        raise ValueError("FUSED_MOE routing tokens must be positive")
    for name, value in (("num_experts", num_experts), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"FUSED_MOE routing {name} must be positive")
    counts = _validate_expert_histogram(
        tokens_per_expert,
        tokens=tokens,
        num_experts=num_experts,
        topk=topk,
        label="FUSED_MOE routing",
    )
    if source_kind == "explicit_assumption":
        if provenance["algorithm"] != _SEEDED_ROUTING_ALGORITHM:
            raise ValueError(
                "modeled FUSED_MOE routing uses a non-canonical algorithm")
        root_seed = provenance["root_seed"]
        if (isinstance(root_seed, bool) or not isinstance(root_seed, int)
                or root_seed < 0):
            raise ValueError(
                "modeled FUSED_MOE routing requires a non-negative seed")
        distribution = provenance["distribution"]
        hot_expert_count = provenance["hot_expert_count"]
        hot_expert_weight = provenance["hot_expert_weight"]
        _validate_seeded_distribution(
            distribution,
            hot_expert_count,
            hot_expert_weight,
            num_experts=num_experts,
        )
        expected_policy_sha256 = _canonical_json_sha256({
            "schema_version": 2,
            "source_kind": source_kind,
            "source_id": provenance["source_id"],
            "algorithm": provenance["algorithm"],
            "distribution": distribution,
            "hot_expert_count": hot_expert_count,
            "hot_expert_weight": hot_expert_weight,
            "root_seed": root_seed,
        })
        if (provenance["policy_descriptor_sha256"]
                != expected_policy_sha256
                or provenance["replay_source_sha256"] is not None
                or provenance["replay_canonical_payload_sha256"] is not None
                or provenance["mechanism_source"] != "Modeled"
                or provenance["validation_status"] != "Projected"):
            raise ValueError(
                "modeled FUSED_MOE routing provenance is inconsistent")
        expected_record_sha256 = _routing_record_sha256(policy_descriptor_sha256=expected_policy_sha256, step_id=step_id, layer_index=layer_index, moe_invocation_id=moe_invocation_id, schedule_step_sha256=provenance['schedule_step_sha256'], tokens=tokens, num_experts=num_experts, topk=topk, tokens_per_expert=counts)
    elif source_kind == "profile_observation":
        if provenance["distribution"] not in _PROFILE_ROUTING_REALIZATIONS:
            raise ValueError(
                "profile FUSED_MOE routing uses an unknown distribution")
        algorithm, sharing_scope = _PROFILE_ROUTING_REALIZATIONS[
            provenance["distribution"]]
        if provenance["algorithm"] != algorithm:
            raise ValueError(
                "profile FUSED_MOE routing uses a non-canonical algorithm")
        if (provenance["hot_expert_count"] is not None
                or provenance["hot_expert_weight"] is not None
                or provenance["root_seed"] is not None
                or provenance["policy_descriptor_sha256"] is None
                or provenance["replay_source_sha256"] is not None
                or provenance["replay_canonical_payload_sha256"] is not None
                or provenance["sharing_scope"] != sharing_scope
                or provenance["mechanism_source"] != "ImportedProfile"
                or provenance["validation_status"]
                != "SourceProvenanceUnavailable"):
            raise ValueError(
                "profile FUSED_MOE routing provenance is inconsistent")
        expected_record_sha256 = _routing_record_sha256(policy_descriptor_sha256=provenance['policy_descriptor_sha256'], step_id=step_id, layer_index=layer_index, moe_invocation_id=moe_invocation_id, schedule_step_sha256=provenance['schedule_step_sha256'], tokens=tokens, num_experts=num_experts, topk=topk, tokens_per_expert=counts)
    elif source_kind == "replay_observation":
        if provenance["algorithm"] != _REPLAY_ROUTING_ALGORITHM:
            raise ValueError(
                "replayed FUSED_MOE routing uses a non-canonical algorithm")
        if layer_index is None:
            raise ValueError(
                "expert routing replay v1 cannot identify an MTP layer")
        if (provenance["distribution"] != _REPLAY_ROUTING_DISTRIBUTION
                or provenance["hot_expert_count"] is not None
                or provenance["hot_expert_weight"] is not None
                or provenance["root_seed"] is not None
                or provenance["policy_descriptor_sha256"] is not None
                or provenance["replay_source_sha256"] is None
                or provenance["replay_canonical_payload_sha256"] is None
                or provenance["mechanism_source"] != "Captured"
                or provenance["validation_status"] != "Validated"):
            raise ValueError(
                "replayed FUSED_MOE routing provenance is inconsistent")
        expected_record_sha256 = ExpertRoutingReplayRecord(
            step_id=step_id,
            layer_index=layer_index,
            moe_invocation_id=moe_invocation_id,
            schedule_step_sha256=provenance["schedule_step_sha256"],
            tokens=tokens,
            tokens_per_expert=counts,
        ).record_sha256
    else:
        raise ValueError(
            f"unknown FUSED_MOE routing source: {source_kind!r}")
    if provenance["record_sha256"] != expected_record_sha256:
        raise ValueError(
            "FUSED_MOE routing record SHA256 disagrees with node semantics")
    return _certify_routing_provenance(provenance, counts, tokens, num_experts, topk, moe_invocation_id)


def _routing_record_sha256(*, policy_descriptor_sha256, step_id, layer_index,
        moe_invocation_id, schedule_step_sha256, tokens, num_experts, topk, tokens_per_expert):
    """One canonical per-invocation record, shared by resolve and validation."""
    return _canonical_json_sha256({
        'policy_descriptor_sha256': policy_descriptor_sha256,
        'step_id': step_id,
        'layer_index': layer_index,
        'moe_invocation_id': moe_invocation_id,
        'schedule_step_sha256': schedule_step_sha256,
        'tokens': tokens,
        'num_experts': num_experts,
        'topk': topk,
        'tokens_per_expert': tokens_per_expert,
    })


# Transient native batch only; always restored or cleared when resolve_layers exits.
_ROUTING_RECORD_JSON_LOCAL = threading.local()
_ROUTING_RECORD_JSON_ENCODER_TYPE = json.JSONEncoder
_ROUTING_RECORD_JSON_ENCODER_BASES = json.JSONEncoder.__bases__
_ROUTING_RECORD_JSON_ENCODER_MRO = json.JSONEncoder.__mro__
_ROUTING_RECORD_JSON_ENCODER_NAMESPACE = tuple(vars(json.JSONEncoder).items())
_ROUTING_RECORD_JSON_SOURCE = tuple((owner, name, getattr(owner, name),
        getattr(owner, name).__code__, getattr(owner, name).__defaults__,
        None if getattr(owner, name).__kwdefaults__ is None
        else tuple(getattr(owner, name).__kwdefaults__.items()))
    for owner, names in ((json, ("dumps",)),
        (json.JSONEncoder, ("__init__", "encode", "iterencode")),
        (json.encoder, ("_make_iterencode",))) for name in names)
_ROUTING_RECORD_JSON_NATIVE_SOURCE = tuple((name, getattr(json.encoder, name))
    for name in ("encode_basestring", "c_make_encoder"))


def _routing_record_json_supported():
    if json.JSONEncoder is not _ROUTING_RECORD_JSON_ENCODER_TYPE:
        return False
    cls = json.JSONEncoder
    namespace = vars(cls)
    if (cls.__bases__ is not _ROUTING_RECORD_JSON_ENCODER_BASES
            or cls.__mro__ is not _ROUTING_RECORD_JSON_ENCODER_MRO
            or len(namespace) != len(_ROUTING_RECORD_JSON_ENCODER_NAMESPACE)
            or any(name is not original_name or value is not original_value
                for (name, value), (original_name, original_value)
                in zip(namespace.items(), _ROUTING_RECORD_JSON_ENCODER_NAMESPACE, strict=True))):
        return False
    for owner, name, function, code, defaults, kwdefaults in _ROUTING_RECORD_JSON_SOURCE:
        current = getattr(owner, name)
        if (current is not function or current.__code__ is not code
                or current.__defaults__ is not defaults):
            return False
        actual_kwdefaults = current.__kwdefaults__
        if kwdefaults is None:
            if actual_kwdefaults is not None:
                return False
        elif (type(actual_kwdefaults) is not dict
                or len(actual_kwdefaults) != len(kwdefaults)
                or any(name is not expected_name or value is not expected_value
                    for (name, value), (expected_name, expected_value)
                    in zip(actual_kwdefaults.items(), kwdefaults, strict=True))):
            return False
    return all(getattr(json.encoder, name) is function
        for name, function in _ROUTING_RECORD_JSON_NATIVE_SOURCE)


def _native_rebound_routing_record_sha256(original, provenance, *,
        layer_index, invocation, tokens, num_experts, topk):
    """Use only the original of a currently active callback-free native batch."""
    local = vars(_ROUTING_RECORD_JSON_LOCAL)
    current = local["record"] if "record" in local else None
    if (current is not None and current[0] is original
            and current[1] is original.routing_provenance
            and current[2] is original.routing_provenance._data
            and current[3] is original.tokens_per_expert
            and current[4:7] == (tokens, num_experts, topk)):
        # These are the first two keys of the ordinary sorted canonical record.
        dynamic = json.dumps({"layer_index": layer_index,
            "moe_invocation_id": invocation}, sort_keys=True,
            separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(dynamic[:-1] + b"," + current[7]).hexdigest()
    return _routing_record_sha256(
        policy_descriptor_sha256=provenance["policy_descriptor_sha256"],
        step_id=provenance["step_id"], layer_index=layer_index,
        moe_invocation_id=invocation,
        schedule_step_sha256=provenance["schedule_step_sha256"],
        tokens=tokens, num_experts=num_experts, topk=topk,
        tokens_per_expert=original.tokens_per_expert)


def _shared_resolution_immutable(resolution):
    """A step proof may retain only the canonical immutable scalar ledger."""
    return (type(resolution) is ExpertRoutingResolution
        and type(resolution.tokens_per_expert) is tuple
        and all(type(count) is int for count in resolution.tokens_per_expert)
        and type(resolution.routing_provenance) is _ValidatedRoutingProvenance
        and type(resolution.routing_provenance._proof) is tuple
        and len(resolution.routing_provenance._proof) == 3
        and type(resolution.routing_provenance._proof[2]) is tuple
        and len(resolution.routing_provenance._proof[2]) == 4
        and _routing_certificate_context(resolution.routing_provenance, resolution.tokens_per_expert,
            *resolution.routing_provenance._proof[2]))


def _rebind_validated_routing_resolution(original, *, layer_index, invocation,
        tokens, num_experts, topk):
    """Owner-only rebind of one fully validated immutable step realization.

    Only layer/invocation/record differ. The caller owns the current-step proof;
    public construction and graph-level validation remain fully strict.
    """
    if not _routing_certificate_context(original.routing_provenance, original.tokens_per_expert,
            tokens, num_experts, topk, original.routing_provenance["moe_invocation_id"]):
        raise ValueError("routing rebind requires its checked immutable context")
    return _rebind_routing_resolution_native(original, layer_index=layer_index,
        invocation=invocation, tokens=tokens, num_experts=num_experts, topk=topk)


def _rebind_routing_resolution_native(original, *, layer_index, invocation,
        tokens, num_experts, topk):
    """Internal materializer; caller owns a checked synchronous native batch."""
    native = (type(original) is ExpertRoutingResolution
        and type(original.routing_provenance) is _VALIDATED_ROUTING_PROVENANCE_TYPE
        and type(original.routing_provenance._data) is MappingProxyType
        and type(invocation) is str and type(layer_index) in (int, type(None))
        and type(tokens) is int and type(num_experts) is int and type(topk) is int)
    provenance = (original.routing_provenance._data.copy() if native
        else dict(original.routing_provenance))
    _validate_routing_invocation(provenance["step_id"], layer_index, invocation)
    provenance["layer_index"], provenance["moe_invocation_id"] = layer_index, invocation
    if native:
        provenance["record_sha256"] = _native_rebound_routing_record_sha256(
            original, provenance, layer_index=layer_index, invocation=invocation,
            tokens=tokens, num_experts=num_experts, topk=topk)
    else:
        provenance["record_sha256"] = _routing_record_sha256(
            policy_descriptor_sha256=provenance["policy_descriptor_sha256"],
            step_id=provenance["step_id"], layer_index=layer_index,
            moe_invocation_id=invocation, schedule_step_sha256=provenance["schedule_step_sha256"],
            tokens=tokens, num_experts=num_experts, topk=topk,
            tokens_per_expert=original.tokens_per_expert)
    result = object.__new__(ExpertRoutingResolution)
    object.__setattr__(result, "tokens_per_expert", original.tokens_per_expert)
    # The caller verified the original native certificate in this synchronous
    # scope. Its readonly static fields and immutable count tuple are unchanged;
    # this function validated the actual layer/invocation and recomputed its SHA.
    constructor = (_construct_native_routing_provenance
        if native and type(provenance["record_sha256"]) is str else _certify_routing_provenance)
    object.__setattr__(result, "routing_provenance", constructor(
        provenance, original.tokens_per_expert, tokens, num_experts, topk, invocation))
    return result


@dataclass(frozen=True)
class ExpertRoutingInput:
    source_kind: str
    source_id: str
    algorithm: str
    root_seed: Optional[int]
    request_stream_sha256: Optional[str]
    replay: Optional[ExpertRoutingReplay]
    vllm_runtime_binding_sha256: Optional[str]
    distribution: str
    hot_expert_count: Optional[int]
    hot_expert_weight: Optional[int]
    profile: Optional[ExpertFrequencyProfile] = None

    def __post_init__(self) -> None:
        if self.source_kind not in (
                "not_applicable", "explicit_assumption",
                "profile_observation", "replay_observation"):
            raise ValueError(
                f"unknown expert routing source kind: {self.source_kind!r}")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("expert routing source_id must be non-empty")
        if not isinstance(self.algorithm, str) or not self.algorithm:
            raise ValueError("expert routing algorithm must be non-empty")
        if not isinstance(self.distribution, str) or not self.distribution:
            raise ValueError("expert routing distribution must be non-empty")
        if self.source_kind == "not_applicable":
            if (self.source_id != "not_applicable"
                    or self.algorithm != "not_applicable"
                    or self.root_seed is not None
                    or self.request_stream_sha256 is not None
                    or self.replay is not None
                    or self.vllm_runtime_binding_sha256 is not None
                    or self.distribution != "not_applicable"
                    or self.hot_expert_count is not None
                    or self.hot_expert_weight is not None
                    or self.profile is not None):
                raise ValueError(
                    "not-applicable expert routing cannot carry a source")
        elif self.source_kind == "explicit_assumption":
            if self.algorithm != _SEEDED_ROUTING_ALGORITHM:
                raise ValueError(
                    "explicit expert routing assumption requires the "
                    f"{_SEEDED_ROUTING_ALGORITHM!r} algorithm")
            if (isinstance(self.root_seed, bool)
                    or not isinstance(self.root_seed, int)):
                raise TypeError(
                    "explicit expert routing root_seed must be an integer")
            if self.root_seed < 0:
                raise ValueError(
                    "explicit expert routing root_seed cannot be negative")
            _validate_seeded_distribution(
                self.distribution,
                self.hot_expert_count,
                self.hot_expert_weight,
                num_experts=None,
            )
            if (self.request_stream_sha256 is not None
                    or self.replay is not None
                    or self.vllm_runtime_binding_sha256 is not None
                    or self.profile is not None):
                raise ValueError(
                    "explicit expert routing assumption cannot carry replay "
                    "inputs")
        elif self.source_kind == "profile_observation":
            if self.distribution not in _PROFILE_ROUTING_REALIZATIONS:
                raise ValueError(
                    "profile expert routing uses an unknown distribution")
            if self.algorithm != _PROFILE_ROUTING_REALIZATIONS[
                    self.distribution][0]:
                raise ValueError(
                    "profile expert routing requires the canonical profile "
                    "sampling algorithm")
            if (self.root_seed is not None
                    or self.request_stream_sha256 is not None
                    or self.replay is not None
                    or self.vllm_runtime_binding_sha256 is not None
                    or self.hot_expert_count is not None
                    or self.hot_expert_weight is not None):
                raise ValueError(
                    "profile expert routing cannot carry assumption or replay inputs")
            if not isinstance(self.profile, ExpertFrequencyProfile):
                raise TypeError(
                    "profile_observation requires an ExpertFrequencyProfile")
            if self.source_id != self.profile.canonical_model_id:
                raise ValueError(
                    "profile expert routing source_id must match canonical model ID")
        elif self.source_kind == "replay_observation":
            if self.algorithm != _REPLAY_ROUTING_ALGORITHM:
                raise ValueError(
                    "expert routing replay requires the "
                    f"{_REPLAY_ROUTING_ALGORITHM!r} algorithm")
            if self.root_seed is not None:
                raise ValueError("expert routing replay cannot carry a seed")
            if (self.distribution != _REPLAY_ROUTING_DISTRIBUTION
                    or self.hot_expert_count is not None
                    or self.hot_expert_weight is not None):
                raise ValueError(
                    "expert routing replay requires captured histogram "
                    "distribution without assumption parameters")
            if self.profile is not None:
                raise ValueError("expert routing replay cannot carry a profile")
            if self.request_stream_sha256 is None:
                raise ValueError(
                    "expert routing replay requires request stream identity")
            _require_sha256(
                "request_stream_sha256", self.request_stream_sha256)
            if not isinstance(self.replay, ExpertRoutingReplay):
                raise TypeError(
                    "replay_observation requires an ExpertRoutingReplay")
            if self.source_id != self.replay.artifact_id:
                raise ValueError(
                    "expert routing source_id must match replay artifact_id")
            if (self.request_stream_sha256
                    != self.replay.context.request_stream_sha256):
                raise ValueError(
                    "expert routing replay request stream SHA256 mismatch")
            if self.vllm_runtime_binding_sha256 is not None:
                _require_sha256(
                    "vllm_runtime_binding_sha256",
                    self.vllm_runtime_binding_sha256,
                )
        else:
            raise ValueError(
                f"unknown expert routing source kind: {self.source_kind!r}")

    @classmethod
    def not_applicable(cls) -> "ExpertRoutingInput":
        return cls(
            "not_applicable", "not_applicable", "not_applicable",
            None, None, None, None, "not_applicable", None, None,
        )

    @classmethod
    def seeded_assumption(
        cls,
        source_id: str,
        root_seed: int,
    ) -> "ExpertRoutingInput":
        return cls(
            "explicit_assumption",
            source_id,
            _SEEDED_ROUTING_ALGORITHM,
            root_seed,
            None,
            None,
            None,
            _UNIFORM_ROUTING_DISTRIBUTION,
            None,
            None,
        )

    @classmethod
    def skewed_assumption(
        cls,
        source_id: str,
        root_seed: int,
        *,
        hot_expert_count: int,
        hot_expert_weight: int,
    ) -> "ExpertRoutingInput":
        return cls(
            "explicit_assumption",
            source_id,
            _SEEDED_ROUTING_ALGORITHM,
            root_seed,
            None,
            None,
            None,
            _SKEW_ROUTING_DISTRIBUTION,
            hot_expert_count,
            hot_expert_weight,
        )

    @classmethod
    def profile_observation(
        cls,
        profile: ExpertFrequencyProfile,
        *,
        per_moe_layer: bool = False,
    ) -> "ExpertRoutingInput":
        if not isinstance(profile, ExpertFrequencyProfile):
            raise TypeError(
                "profile expert routing requires an ExpertFrequencyProfile")
        distribution = (
            _PROFILE_LAYER_ROUTING_DISTRIBUTION if per_moe_layer
            else _PROFILE_ROUTING_DISTRIBUTION)
        return cls(
            "profile_observation",
            profile.canonical_model_id,
            _PROFILE_ROUTING_REALIZATIONS[distribution][0],
            None,
            None,
            None,
            None,
            distribution,
            None,
            None,
            profile,
        )

    @classmethod
    def replay_observation(
        cls,
        replay: ExpertRoutingReplay,
        request_stream_sha256: str,
    ) -> "ExpertRoutingInput":
        if not isinstance(replay, ExpertRoutingReplay):
            raise TypeError(
                "expert routing replay input must be ExpertRoutingReplay")
        return cls(
            "replay_observation",
            replay.artifact_id,
            _REPLAY_ROUTING_ALGORITHM,
            None,
            request_stream_sha256,
            replay,
            None,
            _REPLAY_ROUTING_DISTRIBUTION,
            None,
            None,
        )

    def bind_vllm_runtime(
        self,
        *,
        version: str,
        source_commit: str,
        package_python_manifest_sha256: str,
        model_config_sha256: str,
    ) -> "ExpertRoutingInput":
        """Bind a replay to the effective pinned vLLM runtime."""

        if self.source_kind != "replay_observation" or self.replay is None:
            raise ValueError(
                "only an expert routing replay can bind a vLLM runtime")
        actual = {
            "vllm_version": _require_string("vLLM version", version),
            "vllm_source_commit": _require_string(
                "vLLM source commit", source_commit),
            "vllm_package_manifest_sha256": _require_sha256(
                "vLLM package manifest SHA256",
                package_python_manifest_sha256,
            ),
            "model_config_sha256": _require_sha256(
                "vLLM model config SHA256", model_config_sha256),
        }
        expected = self.replay.provenance.manifest()
        expected.pop("producer")
        if actual != expected:
            raise RuntimeError(
                "expert routing replay provenance disagrees with the "
                f"effective vLLM runtime: expected={expected}, "
                f"actual={actual}")
        binding_sha256 = _canonical_json_sha256({
            "schema_version": 1,
            "producer": self.replay.provenance.producer,
            **actual,
        })
        return ExpertRoutingInput(
            self.source_kind,
            self.source_id,
            self.algorithm,
            self.root_seed,
            self.request_stream_sha256,
            self.replay,
            binding_sha256,
            self.distribution,
            self.hot_expert_count,
            self.hot_expert_weight,
            self.profile,
        )

    @functools.cached_property
    def policy_descriptor_sha256(self) -> Optional[str]:
        """Content address of this routing policy, excluding any step input.

        Every field it hashes is frozen on construction, so the address is a
        constant of the object; the routing resolution reads it several times
        per MoE invocation and would otherwise re-serialize it each time.
        """

        if self.source_kind == "profile_observation":
            if self.profile is None:
                raise RuntimeError("profile expert routing lost its profile")
            return _canonical_json_sha256({
                "schema_version": 1,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": self.distribution,
                "profile_sha256": self.profile.sha256,
                "profile_manifest_sha256": self.profile.manifest_sha256,
                "sharing_scope": _PROFILE_ROUTING_REALIZATIONS[
                    self.distribution][1],
            })
        if self.source_kind != "explicit_assumption":
            return None
        return _canonical_json_sha256({
            "schema_version": 2,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "algorithm": self.algorithm,
            "distribution": self.distribution,
            "hot_expert_count": self.hot_expert_count,
            "hot_expert_weight": self.hot_expert_weight,
            "root_seed": self.root_seed,
        })

    def assert_request_stream_sha256(self, actual_sha256: str) -> None:
        _require_sha256("actual request_stream_sha256", actual_sha256)
        if (self.source_kind == "replay_observation"
                and actual_sha256 != self.request_stream_sha256):
            raise ValueError(
                "expert routing replay request stream SHA256 disagrees with "
                "SimulationConfig.requests")

    def assert_scheduler_config(
        self,
        *,
        pool_name: str,
        scheduler_config_sha256: str,
    ) -> None:
        if not isinstance(pool_name, str) or not pool_name:
            raise ValueError("expert routing Scheduler pool must be non-empty")
        _require_sha256(
            "actual scheduler_config_sha256", scheduler_config_sha256)
        if self.source_kind != "replay_observation":
            return
        if self.replay is None:
            raise RuntimeError(
                "expert routing replay source lost its replay payload")
        if pool_name != self.replay.context.pool_name:
            raise ValueError(
                "expert routing replay Scheduler pool disagrees with the "
                "executed backend")
        if scheduler_config_sha256 != (
                self.replay.context.scheduler_config_sha256):
            raise ValueError(
                "expert routing replay Scheduler config SHA256 mismatch")

    def resolve(
        self,
        *,
        step_id: str,
        layer_index: Optional[int],
        moe_invocation_id: str,
        schedule_step_sha256: str,
        tokens: int,
        num_experts: int,
        topk: int,
        model_config_sha256: str,
        pool_name: str,
        tensor_parallel_size: int,
        expert_parallel_size: int,
        expert_rank_group: Tuple[int, ...],
        expert_placement_strategy: str,
        moe_layer_ordinal: Optional[int] = None,
        assumption_model_semantics_sha256: Optional[str] = None,
    ) -> ExpertRoutingResolution:
        _require_string("resolution step_id", step_id)
        _require_string("resolution moe_invocation_id", moe_invocation_id)
        _require_sha256(
            "resolution schedule_step_sha256", schedule_step_sha256)
        for name, value in (
                ("tokens", tokens), ("num_experts", num_experts),
                ("topk", topk),
                ("tensor_parallel_size", tensor_parallel_size),
                ("expert_parallel_size", expert_parallel_size)):
            _require_integer(name, value, positive=True)
        if topk > num_experts:
            raise ValueError("expert routing topk cannot exceed num_experts")
        _require_sha256("model_config_sha256", model_config_sha256)
        if assumption_model_semantics_sha256 is not None:
            _require_sha256("assumption_model_semantics_sha256", assumption_model_semantics_sha256)
            if self.source_kind != "explicit_assumption":
                raise ValueError("only explicit routing assumptions can share normalized model semantics")
        _require_string("pool_name", pool_name)
        if expert_placement_strategy != "linear":
            raise ValueError(
                "expert routing currently requires linear expert placement")
        rank_group = tuple(expert_rank_group)
        if len(rank_group) != expert_parallel_size:
            raise ValueError(
                "expert routing rank group size disagrees with "
                "expert_parallel_size")
        if self.source_kind == "not_applicable":
            raise ValueError(
                "MoE workload requires explicit expert routing input")
        if self.source_kind == "explicit_assumption":
            _validate_seeded_distribution(
                self.distribution,
                self.hot_expert_count,
                self.hot_expert_weight,
                num_experts=num_experts,
            )
            routing_group_id = _shared_routing_group_id(
                step_id=step_id,
                layer_index=layer_index,
                tokens=tokens,
                num_experts=num_experts,
                topk=topk,
                model_config_sha256=(model_config_sha256 if assumption_model_semantics_sha256 is None
                                     else assumption_model_semantics_sha256),
                pool_name=pool_name,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_rank_group=rank_group,
                expert_placement_strategy=expert_placement_strategy,
            )
            def build_counts():
                return seeded_expert_token_counts(
                    tokens,
                    num_experts,
                    topk,
                    source_id=self.source_id,
                    root_seed=self.root_seed,
                    moe_invocation_id=routing_group_id,
                    distribution=self.distribution,
                    hot_expert_count=self.hot_expert_count,
                    hot_expert_weight=self.hot_expert_weight,
                )

            counts = build_counts()
            record_sha256 = _routing_record_sha256(policy_descriptor_sha256=self.policy_descriptor_sha256, step_id=step_id, layer_index=layer_index, moe_invocation_id=moe_invocation_id, schedule_step_sha256=schedule_step_sha256, tokens=tokens, num_experts=num_experts, topk=topk, tokens_per_expert=counts)
            provenance = {
                "schema_version": 2,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": self.distribution,
                "hot_expert_count": self.hot_expert_count,
                "hot_expert_weight": self.hot_expert_weight,
                "root_seed": self.root_seed,
                "policy_descriptor_sha256": (
                    self.policy_descriptor_sha256),
                "replay_source_sha256": None,
                "replay_canonical_payload_sha256": None,
                "record_sha256": record_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "mechanism_source": "Modeled",
                "validation_status": "Projected",
            }
        elif self.source_kind == "profile_observation":
            if self.profile is None:
                raise RuntimeError("profile expert routing lost its profile")
            if len(self.profile.mean_probabilities) != num_experts:
                raise ValueError(
                    "expert-frequency profile width disagrees with num_experts")
            routing_group_id = _shared_routing_group_id(
                step_id=step_id,
                layer_index=layer_index,
                tokens=tokens,
                num_experts=num_experts,
                topk=topk,
                model_config_sha256=model_config_sha256,
                pool_name=pool_name,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_rank_group=rank_group,
                expert_placement_strategy=expert_placement_strategy,
            )
            per_layer = (
                self.distribution == _PROFILE_LAYER_ROUTING_DISTRIBUTION)
            if per_layer:
                if layer_index is None:
                    if moe_layer_ordinal is not None:
                        raise ValueError(
                            "an MTP MoE invocation has no profile layer row")
                    probabilities = self.profile.mean_probabilities
                    layer_key = "mtp_model_mean"
                else:
                    _require_integer(
                        "moe_layer_ordinal", moe_layer_ordinal,
                        positive=False)
                    if moe_layer_ordinal >= len(
                            self.profile.layer_probabilities):
                        raise ValueError(
                            "moe_layer_ordinal exceeds the profile layers")
                    probabilities = self.profile.layer_probabilities[
                        moe_layer_ordinal]
                    layer_key = moe_layer_ordinal
                # Drop the step from the signature: the histogram is a
                # function of the MoE layer and its token count.
                routing_group_id = _canonical_json_sha256({
                    "schema_version": 1,
                    "moe_layer": layer_key,
                    "tokens": tokens,
                    "num_experts": num_experts,
                    "topk": topk,
                    "model_config_sha256": model_config_sha256,
                    "pool_name": pool_name,
                    "tensor_parallel_size": tensor_parallel_size,
                    "expert_parallel_size": expert_parallel_size,
                    "expert_rank_group": rank_group,
                    "expert_placement_strategy": expert_placement_strategy,
                })
            elif moe_layer_ordinal is not None and not isinstance(
                    moe_layer_ordinal, int):
                raise TypeError("moe_layer_ordinal must be an integer")

            def build_counts():
                if per_layer:
                    return profiled_layer_expert_token_counts(
                        tokens,
                        topk,
                        probabilities=probabilities,
                        profile_sha256=self.profile.sha256,
                        routing_group_id=routing_group_id,
                    )
                return profiled_expert_token_counts(
                    tokens,
                    topk,
                    probabilities=self.profile.mean_probabilities,
                    profile_sha256=self.profile.sha256,
                    routing_group_id=routing_group_id,
                )

            counts = build_counts()
            record_sha256 = _routing_record_sha256(policy_descriptor_sha256=self.policy_descriptor_sha256, step_id=step_id, layer_index=layer_index, moe_invocation_id=moe_invocation_id, schedule_step_sha256=schedule_step_sha256, tokens=tokens, num_experts=num_experts, topk=topk, tokens_per_expert=counts)
            provenance = {
                "schema_version": 2,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": self.distribution,
                "hot_expert_count": None,
                "hot_expert_weight": None,
                "root_seed": None,
                "policy_descriptor_sha256": self.policy_descriptor_sha256,
                "replay_source_sha256": None,
                "replay_canonical_payload_sha256": None,
                "profile_sha256": self.profile.sha256,
                "profile_manifest_sha256": self.profile.manifest_sha256,
                "sharing_scope": _PROFILE_ROUTING_REALIZATIONS[
                    self.distribution][1],
                "execution_signature_sha256": routing_group_id,
                "record_sha256": record_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "mechanism_source": "ImportedProfile",
                "validation_status": "SourceProvenanceUnavailable",
            }
        elif self.source_kind == "replay_observation":
            if self.vllm_runtime_binding_sha256 is None:
                raise RuntimeError(
                    "expert routing replay must bind the effective vLLM "
                    "runtime before workload resolution")
            if layer_index is None:
                raise NotImplementedError(
                    "expert routing replay v1 supports only main decoder "
                    "layer invocations")
            if self.replay is None:
                raise RuntimeError(
                    "expert routing replay source lost its replay payload")
            actual_context = ExpertRoutingReplayContext(
                model_config_sha256=model_config_sha256,
                request_stream_sha256=self.request_stream_sha256,
                scheduler_config_sha256=(
                    self.replay.context.scheduler_config_sha256),
                pool_name=pool_name,
                tensor_parallel_size=tensor_parallel_size,
                expert_parallel_size=expert_parallel_size,
                expert_rank_group=rank_group,
                expert_placement_strategy=expert_placement_strategy,
                num_experts=num_experts,
                topk=topk,
            )
            if actual_context != self.replay.context:
                raise ValueError(
                    "expert routing replay context disagrees with the "
                    "executed model/deployment")
            record = self.replay.record_for(step_id, layer_index)
            if record.moe_invocation_id != moe_invocation_id:
                raise ValueError(
                    "expert routing replay invocation ID disagrees with the "
                    "executed workload")
            if record.schedule_step_sha256 != schedule_step_sha256:
                raise ValueError(
                    "expert routing replay ScheduleStep SHA256 disagrees "
                    "with the executed Scheduler decision")
            if record.tokens != tokens:
                raise ValueError(
                    "expert routing replay token count disagrees with the "
                    f"executed batch: {record.tokens} != {tokens}")
            counts = record.tokens_per_expert
            provenance = {
                "schema_version": 2,
                "source_kind": self.source_kind,
                "source_id": self.source_id,
                "algorithm": self.algorithm,
                "distribution": _REPLAY_ROUTING_DISTRIBUTION,
                "hot_expert_count": None,
                "hot_expert_weight": None,
                "root_seed": None,
                "policy_descriptor_sha256": None,
                "replay_source_sha256": self.replay.source_sha256,
                "replay_canonical_payload_sha256": (
                    self.replay.canonical_payload_sha256),
                "record_sha256": record.record_sha256,
                "step_id": step_id,
                "layer_index": layer_index,
                "moe_invocation_id": moe_invocation_id,
                "schedule_step_sha256": schedule_step_sha256,
                "mechanism_source": "Captured",
                "validation_status": "Validated",
            }
        else:
            raise ValueError(
                f"unknown expert routing source kind: {self.source_kind!r}")
        counts = _validate_expert_histogram(
            counts,
            tokens=tokens,
            num_experts=num_experts,
            topk=topk,
            label="expert routing",
        )
        validated_provenance = validate_expert_routing_provenance(
            provenance,
            tokens_per_expert=counts,
            tokens=tokens,
            num_experts=num_experts,
            topk=topk,
            moe_invocation_id=moe_invocation_id,
        )
        return ExpertRoutingResolution(counts, validated_provenance)

    def resolve_layers(self, layers, *, step_id, schedule_step_sha256, tokens,
            num_experts, topk, model_config_sha256, pool_name,
            tensor_parallel_size, expert_parallel_size, expert_placement_strategy,
            assumption_model_semantics_sha256=None):
        """Resolve one step, sharing only an already-declared common realization.

        No state survives this call. Per-layer profiles and replay keep their
        ordinary resolve path; every invocation still owns its record/hash.
        """
        if type(layers) is not tuple:
            raise TypeError("routing layers must be an immutable tuple")
        if not layers:
            raise ValueError("routing layers cannot be empty")
        common = dict(step_id=step_id, schedule_step_sha256=schedule_step_sha256,
            tokens=tokens, num_experts=num_experts, topk=topk,
            model_config_sha256=model_config_sha256, pool_name=pool_name,
            tensor_parallel_size=tensor_parallel_size, expert_parallel_size=expert_parallel_size,
            expert_placement_strategy=expert_placement_strategy,
            assumption_model_semantics_sha256=assumption_model_semantics_sha256)
        shared_common = (all(type(value) is str for value in (
                step_id, schedule_step_sha256, model_config_sha256, pool_name, expert_placement_strategy))
            and all(type(value) is int for value in (
                tokens, num_experts, topk, tensor_parallel_size, expert_parallel_size))
            and type(assumption_model_semantics_sha256) in (str, type(None)))
        shared = (shared_common and type(self) is ExpertRoutingInput
            and self.resolve.__func__ is _EXPERT_ROUTING_RESOLVE_IMPLEMENTATION
            and (self.source_kind == "explicit_assumption" or
                self.source_kind == "profile_observation"
                and self.distribution != _PROFILE_LAYER_ROUTING_DISTRIBUTION))
        if shared and (self.resolve.__code__ is not _EXPERT_ROUTING_RESOLVE_CODE
                or self.resolve.__defaults__ != _EXPERT_ROUTING_RESOLVE_DEFAULTS
                or self.resolve.__kwdefaults__ != _EXPERT_ROUTING_RESOLVE_KWDEFAULTS
                or ExpertRoutingResolution is not _EXPERT_ROUTING_RESOLUTION_TYPE
                or not all(globals()[name] is function and function.__code__ is code
                    and function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults
                    for name, function, code, defaults, kwdefaults in _EXPERT_ROUTING_STEP_SHARED_SOURCE)
                or not all(getattr(ExpertRoutingResolution, name) is function and function.__code__ is code
                    and function.__defaults__ == defaults and function.__kwdefaults__ == kwdefaults
                    for name, function, code, defaults, kwdefaults in _EXPERT_ROUTING_RESOLUTION_SOURCE)):
            raise RuntimeError("step-shared routing source implementation changed")
        if shared and self.source_kind == "profile_observation":
            if (self.profile is None or type(self.profile.mean_probabilities) is not tuple
                    or any(type(value) not in (int, float) for value in self.profile.mean_probabilities)):
                raise TypeError("step-shared profile probabilities must be immutable numeric tuples")
        native_layers = shared and all(type(record) is tuple and len(record) == 4
            and type(record[0]) in (int, type(None)) and type(record[1]) is str
            and type(record[2]) in (int, type(None)) and type(record[3]) is tuple
            and all(type(rank) is int for rank in record[3]) for record in layers)
        if native_layers and all(record[3] == layers[0][3]
                and (record[0] is None) == (layers[0][0] is None) for record in layers):
            layer_index, invocation, ordinal, ranks = layers[0]
            first = self.resolve(**dict(common, layer_index=layer_index, moe_invocation_id=invocation,
                moe_layer_ordinal=ordinal, expert_rank_group=ranks))
            if (type(first) is ExpertRoutingResolution
                    and _routing_certificate_contexts_native(((first.routing_provenance,
                        first.tokens_per_expert, tokens, num_experts, topk, invocation),))):
                # Everything after the guard is native input and owner-pinned
                # code. No user callback, guard token or state escapes this loop.
                results = [first]
                local = vars(_ROUTING_RECORD_JSON_LOCAL)
                had_previous = "record" in local
                previous = local["record"] if had_previous else None
                local["record"] = None
                try:
                    if len(layers) > 1 and _routing_record_json_supported():
                        data = first.routing_provenance._data
                        common_record = {
                            "policy_descriptor_sha256": data["policy_descriptor_sha256"],
                            "step_id": data["step_id"],
                            "schedule_step_sha256": data["schedule_step_sha256"],
                            "tokens": tokens, "num_experts": num_experts, "topk": topk,
                            "tokens_per_expert": first.tokens_per_expert,
                        }
                        tail = json.dumps(common_record, sort_keys=True,
                            separators=(",", ":"), allow_nan=False,
                            ensure_ascii=False).encode("utf-8")[1:]
                        local["record"] = (first, first.routing_provenance, data,
                            first.tokens_per_expert, tokens, num_experts, topk, tail)
                    for layer_index, invocation, ordinal, ranks in layers[1:]:
                        _require_string("resolution moe_invocation_id", invocation)
                        results.append(_rebind_routing_resolution_native(first,
                            layer_index=layer_index, invocation=invocation,
                            tokens=tokens, num_experts=num_experts, topk=topk))
                    return tuple(results)
                finally:
                    if had_previous:
                        local["record"] = previous
                    else:
                        del local["record"]
        groups, results = {}, []
        for layer_index, invocation, ordinal, ranks in layers:
            ranks = tuple(ranks)
            arguments = dict(common, layer_index=layer_index, moe_invocation_id=invocation,
                moe_layer_ordinal=ordinal, expert_rank_group=ranks)
            # resolve's legacy boundary accepts JSON-serializable non-int
            # ranks. Preserve their exact signature through ordinary resolve;
            # Python bool/float equality must never merge them with int ranks.
            shared_group = shared and all(type(rank) is int for rank in ranks)
            key = ranks, layer_index is None
            if not shared_group or key not in groups:
                resolution = self.resolve(**arguments)
                if shared_group and _shared_resolution_immutable(resolution):
                    groups[key] = resolution
            else:
                _require_string("resolution moe_invocation_id", invocation)
                if self.source_kind == "profile_observation" and ordinal is not None and not isinstance(ordinal, int):
                    raise TypeError("moe_layer_ordinal must be an integer")
                resolution = _rebind_validated_routing_resolution(groups[key],
                    layer_index=layer_index, invocation=invocation,
                    tokens=tokens, num_experts=num_experts, topk=topk)
            results.append(resolution)
        return tuple(results)

    def assert_replay_consumed(
        self,
        invocation_ids: Tuple[str, ...],
    ) -> None:
        if self.source_kind == "replay_observation":
            if self.replay is None:
                raise RuntimeError(
                    "expert routing replay source lost its replay payload")
            self.replay.assert_consumed(invocation_ids)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "algorithm": self.algorithm,
            "distribution": self.distribution,
            "hot_expert_count": self.hot_expert_count,
            "hot_expert_weight": self.hot_expert_weight,
            "root_seed": self.root_seed,
            "policy_descriptor_sha256": self.policy_descriptor_sha256,
            "vllm_runtime_binding_sha256": (
                self.vllm_runtime_binding_sha256),
            "replay": (
                None if self.replay is None else self.replay.manifest()),
            "profile": (
                None if self.profile is None else self.profile.manifest()),
        }


_EXPERT_ROUTING_RESOLVE_IMPLEMENTATION = ExpertRoutingInput.resolve
_EXPERT_ROUTING_RESOLVE_CODE = ExpertRoutingInput.resolve.__code__
_EXPERT_ROUTING_RESOLVE_DEFAULTS = ExpertRoutingInput.resolve.__defaults__
_EXPERT_ROUTING_RESOLVE_KWDEFAULTS = dict(ExpertRoutingInput.resolve.__kwdefaults__)
_EXPERT_ROUTING_RESOLUTION_TYPE = ExpertRoutingResolution

@dataclass(frozen=True)
class K3SourceInputs:
    """Actual Scheduler cache coordinates for one ordinary K3 source step."""

    request_ids: tuple[str, ...]
    slot_mapping: tuple[int, ...]
    state_indices_by_layer: tuple[tuple[int, ...], ...]
    cache_capacity_tokens: int
    state_pool_slots: int
    prefill_output_source: str | None

    def __post_init__(self) -> None:
        if (not isinstance(self.request_ids, tuple) or not self.request_ids
                or any(not isinstance(request_id, str) or not request_id for request_id in self.request_ids)
                or len(set(self.request_ids)) != len(self.request_ids)):
            raise ValueError("K3 source requires distinct nonempty request IDs")
        if type(self.cache_capacity_tokens) is not int or self.cache_capacity_tokens <= 0:
            raise ValueError("K3 MLA cache capacity must be positive")
        if type(self.state_pool_slots) is not int or self.state_pool_slots <= 1:
            raise ValueError("K3 recurrent-state pool must contain a null block and a request block")
        if (not isinstance(self.slot_mapping, tuple)
                or any(type(slot) is not int or not 0 <= slot < self.cache_capacity_tokens for slot in self.slot_mapping)
                or len(set(self.slot_mapping)) != len(self.slot_mapping)):
            raise ValueError("K3 source requires disjoint valid MLA write slots")
        if (not isinstance(self.state_indices_by_layer, tuple) or len(self.state_indices_by_layer) != 93
                or sum(bool(indices) for indices in self.state_indices_by_layer) != 69
                or any(not isinstance(indices, tuple) for indices in self.state_indices_by_layer)
                or any(indices and len(indices) != len(self.request_ids) for indices in self.state_indices_by_layer)
                or any(type(index) is not int or not 0 < index < self.state_pool_slots
                    for indices in self.state_indices_by_layer for index in indices)
                or any(len(set(indices)) != len(indices) for indices in self.state_indices_by_layer)):
            raise ValueError("K3 source requires 69 layer-specific recurrent-state blocks per request")
        if self.prefill_output_source not in (None, "supports_out"):
            raise ValueError("K3 prefill output source must be explicit supports_out or None")


@dataclass(frozen=True)
class K3DataParallelSourceInputs:
    """Actual independent hybrid-cache coordinates in physical DP order.

    None is an idle replica, never a synthetic cache/state allocation.
    Latent-tail scenario belongs to the run, not the native cache receipt.
    """

    source_inputs_by_replica: tuple[K3SourceInputs | None, ...]
    # Idle workers retain their hybrid-pool address extents. These are
    # capacities, not request coordinates, GPU arena binding or state IDs.
    cache_capacity_tokens_by_replica: tuple[int, ...] | None = None
    state_pool_slots_by_replica: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        states = self.source_inputs_by_replica
        if (not isinstance(states, tuple) or len(states) < 2 or not any(state is not None for state in states)
                or any(state is not None and not isinstance(state, K3SourceInputs) for state in states)):
            raise ValueError("K3 DP requires actual typed hybrid states and explicit idle replicas")
        requests = tuple(request_id for state in states if state is not None for request_id in state.request_ids)
        if len(set(requests)) != len(requests):
            raise ValueError("K3 DP hybrid states must retain disjoint original request IDs")
        caches,slots=self.cache_capacity_tokens_by_replica,self.state_pool_slots_by_replica
        if (caches is None)!=(slots is None):
            raise ValueError("K3 DP retained capacities require both MLA and recurrent pools")
        if caches is not None and (type(caches) is not tuple or type(slots) is not tuple
                or len(caches)!=len(states) or len(slots)!=len(states)
                or any(type(value) is not int or value<1 for value in caches)
                or any(type(value) is not int or value<=1 for value in slots)
                or any(state is not None and (state.cache_capacity_tokens!=caches[index]
                    or state.state_pool_slots!=slots[index]) for index,state in enumerate(states))):
            raise ValueError("K3 DP retained capacities must match each actual hybrid pool")


@dataclass(frozen=True)
class DSparkReplicaInputs:
    """One native Scheduler's actual cache tables and declared token feedback.

    Tables retain original native group IDs. They cannot be reconstructed
    from aggregate allocation/release events or shared with another DP owner.
    Empty request rows retain genuine idle-worker geometry. The optional
    capacity normalizes legacy frozen-cohort frames to their current batch.
    Legacy indices declare identity rows; the native bridge supplies the
    source-derived resident indices independently of scheduled batch order.
    Admissions retain scheduled_new_reqs order and require complete fresh
    prompts at the source boundary before any new row is staged.
    """
    request_ids: tuple[str, ...]
    prompt_lengths: tuple[int, ...]
    sampled_token_ids: tuple[tuple[int, ...], ...]
    block_tables: tuple[tuple[tuple[int, ...], ...], ...]
    block_sizes: tuple[int, ...]
    target_swa_group: int
    draft_layer_groups: tuple[int, ...]
    num_kv_blocks: int
    request_state_capacity: int | None = None
    idle_request_state_source: str | None = None
    request_state_indices: tuple[int, ...] | None = None
    admitted_request_ids: tuple[str, ...] = ()

    def __post_init__(self):
        batch = len(self.request_ids)
        if (type(self.request_ids) is not tuple
                or any(type(key) is not str or not key for key in self.request_ids)
                or len(set(self.request_ids)) != batch
                or type(self.prompt_lengths) is not tuple or len(self.prompt_lengths) != batch
                or any(type(n) is not int or n < 1 for n in self.prompt_lengths)
                or type(self.sampled_token_ids) is not tuple or len(self.sampled_token_ids) != batch
                or any(type(ids) is not tuple or not 1 <= len(ids) <= 6
                    or any(type(token) is not int or token < 0 for token in ids) for ids in self.sampled_token_ids)):
            raise ValueError("DSpark native frame requires original requests, prompts and declared sampled prefixes")
        if self.request_state_capacity is None:
            object.__setattr__(self,'request_state_capacity',max(1,batch))
        if type(self.request_state_capacity) is not int or self.request_state_capacity<max(1,batch):
            raise ValueError('DSpark native frame requires a fixed positive request-state capacity')
        if self.request_state_indices is None:
            object.__setattr__(self,'request_state_indices',tuple(range(batch)))
        if (type(self.request_state_indices) is not tuple or len(self.request_state_indices)!=batch
                or len(set(self.request_state_indices))!=batch
                or any(type(index) is not int or not 0<=index<self.request_state_capacity
                    for index in self.request_state_indices)):
            raise ValueError('DSpark request-state indices must name distinct resident rows in scheduled request order')
        if (type(self.admitted_request_ids) is not tuple or len(set(self.admitted_request_ids))!=len(self.admitted_request_ids)
                or any(type(key) is not str or key not in self.request_ids for key in self.admitted_request_ids)):
            raise ValueError('DSpark admitted request IDs must be distinct current requests in native admission order')
        if (batch==0 and self.idle_request_state_source not in ('never_admitted_constructor','retained_request_state')
                or batch>0 and self.idle_request_state_source is not None):
            raise ValueError('DSpark idle state requires an explicit never-admitted constructor or retained request-state source')
        if (type(self.block_tables) is not tuple or not self.block_tables
                or type(self.block_sizes) is not tuple or len(self.block_sizes) != len(self.block_tables)
                or any(type(size) is not int or size < 1 for size in self.block_sizes)
                or type(self.num_kv_blocks) is not int or self.num_kv_blocks < 1
                or type(self.draft_layer_groups) is not tuple or len(self.draft_layer_groups) != 3
                or any(type(group) is not int or not 0 <= group < len(self.block_tables)
                    for group in (self.target_swa_group, *self.draft_layer_groups))):
            raise ValueError("DSpark native frame requires actual group IDs, block sizes and capacity")
        for table in self.block_tables:
            if (type(table) is not tuple or len(table) != batch
                    or any(type(row) is not tuple or any(type(block) is not int or not 0 <= block < self.num_kv_blocks
                        for block in row) for row in table)):
                raise ValueError("DSpark native tables must cover every original request with valid physical block IDs")


@dataclass(frozen=True)
class DSparkSourceInputs:
    draft_model: object
    replicas: tuple[DSparkReplicaInputs | None, ...]

    def __post_init__(self):
        from .model_config import ModelConfig
        if not isinstance(self.draft_model, ModelConfig) or not self.draft_model.is_dspark_draft:
            raise ValueError("DSpark source requires its registered draft ModelConfig")
        if (type(self.replicas) is not tuple or not self.replicas
                or any(state is not None and not isinstance(state, DSparkReplicaInputs) for state in self.replicas)
                or not any(state is not None and bool(state.request_ids) for state in self.replicas)):
            raise ValueError("DSpark source requires physical DP order and explicit idle replicas")
        ids = tuple(key for state in self.replicas if state is not None for key in state.request_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("DSpark source replicas must retain disjoint original request IDs")


@dataclass(frozen=True)
class V32PaddedMTPInputs:
    """Shared0.27/0.30 padded scenarios, never numerically generated drafts.

    Request IDs follow the incoming ScheduleStep. Draft IDs follow its
    explicitly packed vLLM decode-prefix row order, first all rows then one
    retained row per request. The forward validates those extents/identities.
    """
    request_ids: tuple[str, ...]
    source_kind: str
    source_id: str
    draft_input_token_ids: tuple[tuple[int, ...], ...]
    discarded_rows: tuple[bool, ...]
    block_size_tokens: int
    parallel_drafting: bool
    use_local_argmax_reduction: bool
    probabilistic: bool

    def __post_init__(self):
        if (not isinstance(self.request_ids, tuple) or not self.request_ids
                or any(not isinstance(value, str) or not value for value in self.request_ids)
                or len(set(self.request_ids)) != len(self.request_ids)
                or self.source_kind not in ("declared_synthetic", "explicit_assumption")
                or not isinstance(self.source_id, str) or not self.source_id.strip()):
            raise ValueError("V32 MTP needs unique request IDs and a declared source scenario")
        if (not isinstance(self.draft_input_token_ids, tuple) or not self.draft_input_token_ids
                or any(not isinstance(ids, tuple) or not ids for ids in self.draft_input_token_ids)
                or any(type(token) is not int or not 0 <= token < 2**31
                    for ids in self.draft_input_token_ids for token in ids)):
            raise ValueError("V32 MTP needs independent explicit INT32 IDs for every draft pass")
        if (not isinstance(self.discarded_rows, tuple) or len(self.discarded_rows) != len(self.request_ids)
                or any(type(value) is not bool for value in self.discarded_rows)):
            raise ValueError("V32 MTP discard scenario must cover every request with explicit booleans")
        # A discarded sample still has a physical padded drafter row. Its
        # supported replay/backup semantics are checked by the source owner
        # against the current ScheduledRequest, not inferred from this flag.
        if type(self.block_size_tokens) is not int or self.block_size_tokens <= 0:
            raise ValueError("V32 MTP requires its explicit positive cache block size")
        if any(type(value) is not bool or value for value in (
                self.parallel_drafting, self.use_local_argmax_reduction, self.probabilistic)):
            raise NotImplementedError("V32 source requires sequential full-logits deterministic greedy drafting")


@dataclass(frozen=True)
class DSparkFirstProposalInputs:
    """Declared first-proposal transcript, distinct from ordinary V4/MTP."""
    request_ids: tuple[str,...]
    draft_model: "ModelConfig"
    target_positions: tuple[int,...]
    target_slot_mapping: tuple[int,...]
    anchor_token_ids: tuple[int,...]
    draft_token_ids: tuple[tuple[int,...],...]
    source_kind: str
    source_id: str
    confidence_enabled: bool = False
    confidence_head_available: bool | None = None

    # One schema for first and later proposals. None means first proposal;
    # verification supplies every predecessor/greedy fact explicitly.
    predecessor_step_id: str | None = None
    previous_retained_kv_tokens: int | None = None
    previous_anchor_token_ids: tuple[int, ...] | None = None
    previous_draft_token_ids: tuple[tuple[int, ...], ...] | None = None
    target_greedy_token_ids: tuple[int, ...] | None = None
    target_bonus_token_ids: tuple[int, ...] | None = None
    accepted_drafts: int | None = None
    finish_request: bool = False
    delivered_token_count: int | None = None


    def __post_init__(self):
        from .model_config import ModelConfig
        if (type(self.request_ids)is not tuple or len(self.request_ids)!=1
                or any(type(value)is not str or not value for value in self.request_ids)
                or not isinstance(self.draft_model,ModelConfig) or self.draft_model.dspark_block_size!=5
                or self.source_kind not in ("declared_synthetic","explicit_assumption")
                or type(self.source_id)is not str or not self.source_id):
            raise ValueError("DSpark first proposal needs an actual registered draft config and explicit one-request transcript")
        self.validate_progress()
        if len(self.target_positions)!=len(self.target_slot_mapping):
            raise ValueError("DSpark target positions and logical slots must cover the same source rows")
        if (type(self.anchor_token_ids)is not tuple or len(self.anchor_token_ids)!=1
                or type(self.draft_token_ids)is not tuple or len(self.draft_token_ids)!=5
                or any(type(ids)is not tuple or len(ids)!=1 for ids in self.draft_token_ids)
                or any(type(v)is not int or not 0<=v<self.draft_model.vocab_size
                    for ids in (self.anchor_token_ids,*self.draft_token_ids) for v in ids)):
            raise ValueError("DSpark first proposal requires actual bounded token scenarios for every sequential sample")
        if type(self.confidence_enabled)is not bool or self.confidence_head_available not in (None,True,False):
            raise TypeError("DSpark confidence mode and optional checkpoint presence must be typed explicitly")
        if self.confidence_head_available is not None and type(self.confidence_head_available)is not bool:
            raise TypeError("DSpark checkpoint head presence cannot be inferred from an integer")
        if self.confidence_enabled:
            raise NotImplementedError("DSpark first proposal currently requires confidence-disabled sampling")


    @property
    def is_verification(self):
        return self.predecessor_step_id is not None

    @property
    def retained_kv_tokens(self):
        return (self.previous_retained_kv_tokens + self.accepted_drafts + 1
            if self.is_verification else len(self.target_positions))

    def validate_progress(self):
        if type(self.finish_request) is not bool:
            raise ValueError("DSpark request completion flag must be explicit and typed")
        valid_count=1 if self.accepted_drafts is None else self.accepted_drafts+1
        if self.delivered_token_count is not None and (type(self.delivered_token_count) is not int
                or not 0<=self.delivered_token_count<=valid_count
                or (self.delivered_token_count<valid_count and not self.finish_request)):
            raise ValueError("DSpark delivered prefix must match raw output or its explicit terminal truncation")
        facts = (self.previous_retained_kv_tokens, self.previous_anchor_token_ids,
            self.previous_draft_token_ids, self.target_greedy_token_ids, self.accepted_drafts)
        if not self.is_verification:
            if any(value is not None for value in (*facts, self.target_bonus_token_ids)):
                raise ValueError("DSpark first proposal cannot carry verification facts without a predecessor")
            expected = tuple(range(len(self.target_positions)))
        else:
            if (type(self.predecessor_step_id) is not str or not self.predecessor_step_id
                    or type(self.previous_retained_kv_tokens) is not int or self.previous_retained_kv_tokens <= 0
                    or type(self.accepted_drafts) is not int or not 0 <= self.accepted_drafts <= 5
                    or type(self.previous_anchor_token_ids) is not tuple or len(self.previous_anchor_token_ids) != 1
                    or type(self.previous_draft_token_ids) is not tuple or len(self.previous_draft_token_ids) != 5
                    or any(type(ids) is not tuple or len(ids) != 1 for ids in self.previous_draft_token_ids)
                    or type(self.target_greedy_token_ids) is not tuple or len(self.target_greedy_token_ids) != 5):
                raise ValueError("DSpark verification requires explicit predecessor, live KV and five target greedy IDs")
            tokens = (*self.previous_anchor_token_ids, *(ids[0] for ids in self.previous_draft_token_ids),
                *self.target_greedy_token_ids)
            if any(type(token) is not int or not 0 <= token < self.draft_model.vocab_size for token in tokens):
                raise ValueError("DSpark verification transcript has an invalid token ID")
            accepted = self.accepted_drafts
            previous = tuple(ids[0] for ids in self.previous_draft_token_ids)
            if (previous[:accepted] != self.target_greedy_token_ids[:accepted]
                    or (accepted < 5 and previous[accepted] == self.target_greedy_token_ids[accepted])):
                raise ValueError("DSpark accepted count must equal the deterministic greedy matching prefix")
            if accepted == 5:
                if (type(self.target_bonus_token_ids) is not tuple or len(self.target_bonus_token_ids) != 1
                        or type(self.target_bonus_token_ids[0]) is not int
                        or not 0 <= self.target_bonus_token_ids[0] < self.draft_model.vocab_size):
                    raise ValueError("DSpark all-accepted verification requires its actually consumed bonus scenario")
                terminal = self.target_bonus_token_ids
            else:
                if self.target_bonus_token_ids is not None:
                    raise ValueError("DSpark rejection returns before reading the sixth bonus logits row")
                terminal = (self.target_greedy_token_ids[accepted],)
            if self.anchor_token_ids != terminal:
                raise ValueError("DSpark next anchor must equal the terminal verification token")
            expected = tuple(range(self.previous_retained_kv_tokens, self.previous_retained_kv_tokens + 6))
        if (self.target_positions != expected or self.target_slot_mapping != expected
                or any(type(value) is not int for values in (self.target_positions, self.target_slot_mapping) for value in values)):
            raise ValueError("DSpark positions and slots must match the current declared logical arena")


@dataclass(frozen=True)
class RuntimeInputs:
    expert_routing: ExpertRoutingInput
    attention_kv_layout: str
    attention_kv_distribution: str
    graph_execution: GraphExecutionConfig = field(
        default_factory=GraphExecutionConfig.disabled)
    # Serving framework whose model-runner dispatch the workload follows
    # (e.g. SGLang's MHA-vs-absorbed MLA prefill choice).
    serving_framework: str = "vllm"
    # Explicit model-source path selection; separate from Scheduler runtime
    # identity. Legacy callers retain their pinned model dispatch.
    model_execution_version: str = "legacy"
    # A source protocol choice, never proof that this runner executed.
    # None retains the model-only projection with proposer gaps explicit.
    mtp_proposer_execution_source: str | None = None
    # Explicit source scenario, not evidence that real metadata was observed.
    # The eager dict has exactly the scheduled, valid rows and max KV length.
    v4_eager_metadata_source: str | None = None
    # Conditional modular CUDA/top6 projection, not an observed provider.
    v4_routing_source: str | None = None
    # Explicit legacy greedy sampler source scenario, never an observed runner.
    mtp_draft_sampling_source: str | None = None
    # Explicit ordinary materializations, plain greedy or metadata-present
    # rejection source stages; actual library/runtime selection stays unbound.
    target_sampling_source: str | None = None
    # Exact target token IDs in scheduled row order, when supplied by a caller.
    # None means unknown ownership, never a uniform TP ownership assumption.
    embedding_token_ids: tuple[int, ...] | None = None
    # Caller request-order prompt lengths for ordinary GLM/Kimi MLA source
    # packing. These distinguish short prompt extends from finished decodes.
    symbolic_source_prompt_lengths: tuple[int, ...] | None = None
    # Optional real/declared prompt lengths in caller request order, used
    # only to distinguish short extends from completed decodes in V4 source.
    v4_source_prompt_lengths: tuple[int, ...] | None = None
    # Source metadata constructor input, independent of checkpoint context
    # limit and never evidence of an observed runtime configuration.
    v4_source_max_model_len: int | None = None
    v4_source_max_batched_tokens: int | None = None
    # Explicit value-dependent output domain, not inferred from finite logits
    # or evidence of an actual indexer/provider. Every row has exactly
    # min(index_topk, completed_states) distinct, valid causal indices.
    v4_indexer_selection_source: str | None = None
    # Declared process-lifetime env value before cached source dispatch runs.
    # False determines the optional mHC fallback, not the forced broadcast
    # call or whether a DeepGEMM implementation is installed/executable.
    v4_source_use_deep_gemm: bool | None = None
    # Shared metadata-only source profile, distinct from actual runtime execution.
    symbolic_forward_source: str | None = None
    # Declared entry history for the single unlocked source workspace lane,
    # in TP-rank order. Execution carries its resulting capacity forward;
    # None remains unknown, never an implicit cold start or provider receipt.
    symbolic_workspace_initial_capacity_bytes: tuple[int, ...] | None = None
    # Non-V4 source constructor limits and the explicit GLM attention domain.
    # These do not infer provider choice or dynamic selected expert/token IDs.
    symbolic_source_max_model_len: int | None = None
    symbolic_source_max_batched_tokens: int | None = None
    symbolic_source_max_requests: int | None = None
    # Explicit per-pass source scenario, including padded/stale draft rows.
    # Never derived from target IDs or presented as observed model output.
    v3_mtp_draft_input_token_ids: tuple[tuple[int, ...], ...] | None = None
    v3_mtp_attention_source: str | None = None
    v3_mtp_batch_invariant: bool | None = None
    glm_indexer_selection_source: str | None = None
    glm_prefill_attention_source: str | None = None
    model_source_inputs: object | None = None
    maca_copy_runtime_source: str | None = None
    # Explicit ordinary model-run scenario; never inferred from Scheduler rows or provider receipts.
    k3_latent_tail_tier: str | None = None
    # This source profile historically traces Triton KDA. Native decode is
    # an explicit alternate source, never inferred from an installed GPU op.
    k3_kda_decode_backend: str = "triton"
    # Request-ID keyed prompt lengths for variable-size Production windows.
    # The workload builder projects these to each Scheduler step's row order.
    symbolic_source_prompt_lengths_by_request: Mapping[str, int] | None = None
    dspark_all_token_ids_storage: str | None = None
    # Pinned0.30 sparse_attn_indexer env default; a declared override affects
    # no-metadata temporary allocation, never fitted operator timing.
    glm_indexer_max_logits_mb: int = 512
    # Standalone/warm source may explicitly declare this per-DP proposer
    # history. A run-owned SourceProgramCache binds constructor and updates.
    glm_mtp_legacy_skip_topk_by_data_rank: tuple[bool, ...] | None = None
    # Explicit vLLM0.30 CUDA Norm source branch; None stays unknown. False
    # declares the upstream environment default, not an observed provider.
    norm_source_batch_invariant: bool | None = None
    # Explicit finite-tile analytical program, not a captured cuBLAS launch.
    bf16_gemm_model: str | None = None
    # Explicit eager prepared-expert reference; captured aggregate is retained.
    bf16_expert_model: str | None = None
    # Explicit input approximation for the ordinary TP2 source. Unknown
    # scheduled IDs and retained tails are declared zero, never observed.
    dense_source_input_policy: str | None = None
    dense_source_capture_limits: tuple[int, int] | None = None
    # Keep the target's declared K when GMR skips the proposer near its cap.
    glm_target_num_speculative_tokens: int = 0

    def __post_init__(self) -> None:
        if type(self.glm_target_num_speculative_tokens) is not int or self.glm_target_num_speculative_tokens < 0:
            raise ValueError("GLM target speculative depth must be an explicit nonnegative integer")
        if self.glm_target_num_speculative_tokens and (
                self.symbolic_forward_source not in ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                or self.model_execution_version != "vllm_0_30_0" or self.serving_framework != "vllm"
                or self.graph_execution.enabled
                or type(self.symbolic_source_max_model_len) is not int or self.symbolic_source_max_model_len <= 0
                or type(self.symbolic_source_max_batched_tokens) is not int or self.symbolic_source_max_batched_tokens <= 0
                or self.target_sampling_source not in ("vllm_0_30_rejection_raw_logits_bf16", "vllm_0_30_plain_greedy_bf16")
                or (isinstance(self.model_source_inputs, V32PaddedMTPInputs)
                    and len(self.model_source_inputs.draft_input_token_ids) != self.glm_target_num_speculative_tokens)):
            raise ValueError("GLM target speculative depth requires its explicit eager source and matching frame depth")
        if self.bf16_expert_model is not None and (
                self.bf16_expert_model != "triton_reference"
                or self.serving_framework != "vllm"
                or self.model_execution_version != "vllm_0_30_0"
                or self.symbolic_forward_source != "cuda_eager_inplace"):
            raise ValueError("BF16 expert reference requires explicit vLLM0.30 CUDA analytical source")
        if self.dense_source_input_policy is None:
            if self.dense_source_capture_limits is not None:
                raise ValueError("dense source capture limits require an explicit input policy")
        elif (self.dense_source_input_policy != "declared_zero_tp2"
                or type(self.dense_source_capture_limits) is not tuple
                or len(self.dense_source_capture_limits) != 2
                or any(type(value) is not int or value <= 0 for value in self.dense_source_capture_limits)
                or self.serving_framework != "vllm" or self.model_execution_version != "vllm_0_30_0"
                or self.symbolic_forward_source != "cuda_eager_inplace"
                or self.target_sampling_source != "vllm_0_30_plain_greedy_bf16"
                or self.mtp_draft_sampling_source is not None
                or self.graph_execution.configured_mode != "FULL_AND_PIECEWISE"):
            raise ValueError("declared dense input policy requires ordinary0.30 FULL/PIECEWISE and explicit capture limits")
        if self.bf16_gemm_model is not None and (
                self.bf16_gemm_model != "tiled_reference"
                or self.serving_framework != "vllm"
                or self.model_execution_version != "vllm_0_30_0"
                or self.symbolic_forward_source != "cuda_eager_inplace"):
            raise ValueError("BF16 GEMM reference requires explicit vLLM0.30 CUDA analytical source")
        if self.norm_source_batch_invariant is not None:
            if type(self.norm_source_batch_invariant) is not bool:
                raise TypeError("Norm source batch-invariant declaration must be boolean")
            if (self.norm_source_batch_invariant is not False
                    or self.serving_framework != "vllm"
                    or self.model_execution_version != "vllm_0_30_0"
                    or self.symbolic_forward_source != "cuda_eager_inplace"):
                raise ValueError("Norm source declaration requires non-batch-invariant vLLM0.30 CUDA source")
        if type(self.glm_indexer_max_logits_mb) is not int or self.glm_indexer_max_logits_mb <= 0:
            raise ValueError("GLM indexer temporary logits MiB must be an explicit positive integer")
        if self.k3_kda_decode_backend not in ("triton","native") or (
                self.k3_kda_decode_backend=="native" and self.symbolic_forward_source!="vllm_0_30_k3_mxfp4_sync"):
            raise ValueError("K3 KDA decode requires an explicit Triton or native source backend")
        if self.glm_mtp_legacy_skip_topk_by_data_rank is not None and (
                type(self.glm_mtp_legacy_skip_topk_by_data_rank) is not tuple or not self.glm_mtp_legacy_skip_topk_by_data_rank
                or any(type(value) is not bool for value in self.glm_mtp_legacy_skip_topk_by_data_rank)
                or self.model_execution_version != "vllm_0_30_0"
                or self.symbolic_forward_source not in ("vllm_0_30_v32_bf16_sync_explicit_allreduce",
                    "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                or self.mtp_draft_sampling_source != "vllm_0_30_legacy_full_logits_greedy"):
            raise ValueError("GLM MTP legacy skip-topk history must declare a boolean per DP replica")
        if self.maca_copy_runtime_source is not None and (
                self.maca_copy_runtime_source != MACA_COPY_RUNTIME_SOURCE
                or self.serving_framework != "vllm"
                or self.model_execution_version != "vllm_0_24_0"):
            raise ValueError("MACA Copy runtime source requires its declared vLLM0.24 native unset-env profile")
        if self.model_source_inputs is not None:
            from .graph_padding import VLLMDenseGraphInputs
            from .traced_sglang import SGLangEagleChainState, SGLangPrefillInputs, SGLangGraphInputs
            if not isinstance(self.model_source_inputs, (SGLangEagleChainState, SGLangPrefillInputs, SGLangGraphInputs,
                    K3SourceInputs, K3DataParallelSourceInputs, DSparkSourceInputs, V32PaddedMTPInputs, VLLMDenseGraphInputs, DSparkFirstProposalInputs)):
                raise TypeError("model-source inputs require a registered typed source state")
            if isinstance(self.model_source_inputs,DSparkFirstProposalInputs) and (
                    self.serving_framework!="vllm" or self.model_execution_version!="vllm_0_30_0"
                    or self.symbolic_forward_source!="vllm_0_30_v4_sync_fp8_ds_mla"
                    or self.target_sampling_source!="vllm_0_30_plain_greedy_bf16"
                    or self.graph_execution.enabled or self.mtp_draft_sampling_source is not None
                    or self.mtp_proposer_execution_source is not None or self.v4_source_use_deep_gemm is not False):
                raise ValueError("DSpark first proposal requires its declared eager0.30 paired V4 source")
            if isinstance(self.model_source_inputs, (K3SourceInputs, K3DataParallelSourceInputs)) and (
                    self.serving_framework != "vllm" or self.model_execution_version != "vllm_0_30_0"
                    or self.symbolic_forward_source != "vllm_0_30_k3_mxfp4_sync"
                    or self.graph_execution.enabled):
                raise ValueError("K3 Scheduler coordinates require their matching eager vLLM0.30 source")
            if isinstance(self.model_source_inputs, VLLMDenseGraphInputs) and (
                    self.serving_framework != "vllm" or self.model_execution_version != "vllm_0_30_0"
                    or self.symbolic_forward_source != "cuda_eager_inplace" or not self.graph_execution.enabled):
                raise ValueError("dense graph source inputs require their matching captured vLLM0.30 CUDA source")
            if isinstance(self.model_source_inputs, SGLangGraphInputs) and (
                    self.serving_framework != "sglang" or self.model_execution_version != "sglang_0_5_19"
                    or self.graph_execution.configured_mode != "FULL_DECODE_ONLY"
                    or self.symbolic_forward_source not in ("sglang_0_5_19_deepseek_v3_decode_fp8",
                        "sglang_0_5_19_deepseek_v3_decode_fp8_flashinfer_ar_norm",
                        "sglang_0_5_19_deepseek_v3_target_fp8", "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm")):
                raise ValueError("SGLang graph source inputs require their matching FULL_DECODE_ONLY source")
            if isinstance(self.model_source_inputs, SGLangPrefillInputs) and (
                    self.serving_framework != "sglang" or self.model_execution_version != "sglang_0_5_19"
                    or self.symbolic_forward_source not in ("sglang_0_5_19_deepseek_v3_target_fp8",
                        "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm")):
                raise ValueError("prefill source inputs require their matching SGLang target")
            if isinstance(self.model_source_inputs, SGLangEagleChainState) and (
                    self.serving_framework != "sglang" or self.model_execution_version != "sglang_0_5_19"
                    or self.symbolic_forward_source not in ("sglang_0_5_19_deepseek_v3_target_fp8",
                        "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm")
                    or self.target_sampling_source != "sglang_0_5_19_plain_greedy_fp32"):
                raise ValueError("EAGLE source state requires its matching SGLang target and sampler")
        if not isinstance(self.expert_routing, ExpertRoutingInput):
            raise TypeError("expert_routing must be ExpertRoutingInput")
        if self.symbolic_forward_source == "vllm_0_30_v4_dspark":
            if (self.serving_framework != "vllm" or self.model_execution_version != "vllm_0_30_0"
                    or self.graph_execution.enabled or self.attention_kv_layout != "mla"
                    or self.attention_kv_distribution != "uniform" or self.mtp_proposer_execution_source is not None
                    or self.target_sampling_source != "vllm_0_30_V2_greedy"
                    or self.mtp_draft_sampling_source != "vllm_0_30_dspark_greedy_full_vocab"
                    or self.v4_source_use_deep_gemm is not False
                    or self.v4_eager_metadata_source != "vllm_0_30_eager_dict_all_valid"
                    or self.v4_routing_source != "vllm_0_30_modular_cuda_top6_pure_text"
                    or self.dspark_all_token_ids_storage not in ('declared_uva_host','declared_gpu_fallback')
                    or self.expert_routing.source_kind not in ("explicit_assumption", "profile_observation", "replay_observation")
                    or any(type(limit) is not int or limit < 1 for limit in (self.v4_source_max_model_len,self.v4_source_max_batched_tokens))
                    or type(self.symbolic_workspace_initial_capacity_bytes) is not tuple
                    or not self.symbolic_workspace_initial_capacity_bytes
                    or any(type(size) is not int or size < 0 for size in self.symbolic_workspace_initial_capacity_bytes)
                    or self.model_source_inputs is not None and not isinstance(self.model_source_inputs, DSparkSourceInputs)):
                raise ValueError("DSpark runtime requires its explicit eager V2/Markov source, limits and typed native frame")
            if any(getattr(self, name) is not None for name in (
                    "maca_copy_runtime_source","k3_latent_tail_tier","symbolic_source_prompt_lengths",
                    "symbolic_source_prompt_lengths_by_request","symbolic_source_max_model_len",
                    "symbolic_source_max_batched_tokens","symbolic_source_max_requests","v3_mtp_draft_input_token_ids",
                    "v3_mtp_attention_source","v3_mtp_batch_invariant","glm_indexer_selection_source",
                    "glm_prefill_attention_source","v4_source_prompt_lengths")):
                raise ValueError("DSpark runtime cannot carry foreign model source declarations")
            if self.v4_indexer_selection_source not in (None,"declared_causal_topk"):
                raise ValueError("DSpark runtime requires its explicit causal target indexer selection")
            return
        if isinstance(self.model_source_inputs, DSparkSourceInputs):
            raise ValueError("DSpark native frame requires the explicit DSpark runtime source")
        if self.dspark_all_token_ids_storage is not None:
            raise ValueError("DSpark token history storage requires the explicit DSpark runtime source")
        glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
        k3_source = self.symbolic_forward_source == "vllm_0_30_k3_mxfp4_sync"
        if self.k3_latent_tail_tier not in (None, "ALLREDUCE_OVERLAP", "COLUMN_PARALLEL"):
            raise ValueError("K3 latent tail requires an explicit supported ordinary source scenario")
        if self.k3_latent_tail_tier is not None and (not k3_source or self.serving_framework != "vllm"
                or self.model_execution_version != "vllm_0_30_0" or self.graph_execution.enabled
                or self.mtp_draft_sampling_source is not None or self.mtp_proposer_execution_source is not None):
            raise ValueError("K3 latent tail belongs only to the declared eager0.30 target source")
        dsv3_source = self.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
        dsv3_source011 = self.symbolic_forward_source == "vllm_0_11_deepseek_v3_sync_fp8"
        sglang_sources = ("sglang_0_5_19_deepseek_v3_decode_fp8",
            "sglang_0_5_19_deepseek_v3_decode_fp8_flashinfer_ar_norm",
            "sglang_0_5_19_deepseek_v3_target_fp8",
            "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm")
        sglang_source = self.symbolic_forward_source in sglang_sources
        # Complete ordinary API inputs can be bound only once model and
        # deployment are available. This is a pending source declaration;
        # the common binder must resolve it before executing any workload.
        pending_dense011 = (self.symbolic_forward_source is None
            and self.model_execution_version == "vllm_0_11_0"
            and self.serving_framework == "vllm"
            and self.target_sampling_source in ("vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_greedy_fp16")
            and self.attention_kv_layout == "paged_varlen" and self.attention_kv_distribution == "uniform"
            and (not self.graph_execution.enabled or self.expert_routing.source_kind in ("not_applicable", "explicit_assumption", "profile_observation"))
            and self.model_source_inputs is None
            and self.mtp_draft_sampling_source is None and self.mtp_proposer_execution_source is None)
        pending_dense024 = (self.symbolic_forward_source is None
            and self.model_execution_version == "vllm_0_24_0" and self.serving_framework == "vllm"
            and self.target_sampling_source == "vllm_0_24_plain_greedy_bf16"
            and self.expert_routing.source_kind in ("not_applicable", "explicit_assumption", "profile_observation")
            and self.attention_kv_layout == "paged_varlen" and self.attention_kv_distribution == "uniform"
            and self.model_source_inputs is None and self.mtp_draft_sampling_source is None
            and self.mtp_proposer_execution_source is None)
        pending_dense018 = (self.symbolic_forward_source is None
            and self.model_execution_version == "vllm_0_18_0" and self.serving_framework == "vllm"
            and self.target_sampling_source == "vllm_0_18_plain_greedy_bf16"
            and self.expert_routing.source_kind in ("not_applicable", "explicit_assumption", "profile_observation")
            and self.attention_kv_layout == "paged_varlen" and self.attention_kv_distribution == "uniform"
            and self.model_source_inputs is None and self.mtp_draft_sampling_source is None
            and self.mtp_proposer_execution_source is None)
        pending_mla030 = (self.symbolic_forward_source is None
            and self.model_execution_version == "vllm_0_30_0" and self.serving_framework == "vllm"
            and self.attention_kv_layout == "mla" and self.attention_kv_distribution == "uniform"
            and self.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            and self.model_source_inputs is None
            and self.mtp_draft_sampling_source is None and self.mtp_proposer_execution_source is None
            and (self.symbolic_source_max_model_len is not None
                or self.symbolic_source_max_batched_tokens is not None))
        if self.symbolic_source_prompt_lengths is not None:
            if (not isinstance(self.symbolic_source_prompt_lengths, tuple) or not self.symbolic_source_prompt_lengths
                    or any(type(value) is not int or value < 0 for value in self.symbolic_source_prompt_lengths)):
                raise ValueError("symbolic source prompt lengths require explicit nonnegative integers in caller request order")
            if (not pending_mla030 and self.symbolic_forward_source not in (
                    *glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce")
                    or self.model_execution_version != "vllm_0_30_0" or self.serving_framework != "vllm"
                    or self.mtp_draft_sampling_source is not None or self.model_source_inputs is not None):
                raise ValueError("symbolic source prompt lengths require an ordinary GLM/Kimi0.30 MLA source")
        if self.symbolic_source_prompt_lengths_by_request is not None:
            values = self.symbolic_source_prompt_lengths_by_request
            # A D-only MTP program still carries P's canonical prompt map
            # until per-step routing selects target-only prefill. D consumes
            # its typed MTP frame instead and clears this ordinary-source map.
            pending_glm_mtp_transport = (self.symbolic_forward_source in glm_sources
                and self.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"
                and self.model_source_inputs is None and self.embedding_token_ids is None)
            if (self.symbolic_source_prompt_lengths is not None
                    or not isinstance(values, Mapping) or not values
                    or any(not isinstance(key, str) or not key
                        or type(length) is not int or length < 0
                        for key, length in values.items())):
                raise ValueError("source prompt lengths by request require unique request IDs and nonnegative lengths")
            if (not pending_mla030 and self.symbolic_forward_source not in (
                    *glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce")
                    or self.model_execution_version != "vllm_0_30_0"
                    or self.serving_framework != "vllm"
                    or ((self.mtp_draft_sampling_source is not None or self.model_source_inputs is not None)
                        and not pending_glm_mtp_transport)):
                raise ValueError("source prompt lengths by request require an ordinary GLM/Kimi0.30 MLA source")
        source_version = ("sglang_0_5_19" if sglang_source else "vllm_0_27_1" if dsv3_source
            else "vllm_0_18_0" if pending_dense018 or (self.symbolic_forward_source == "vllm_ascend_eager"
                and self.model_execution_version == "vllm_0_18_0")
            else "vllm_0_24_0" if pending_dense024 or (self.symbolic_forward_source == "cuda_eager_inplace"
                and self.model_execution_version == "vllm_0_24_0")
            else "vllm_0_11_0" if self.symbolic_forward_source == "vllm_0_11_glm4_sync"
                or (self.symbolic_forward_source == "cuda_eager_inplace" and self.model_execution_version == "vllm_0_11_0")
                or pending_dense011 or dsv3_source011
            else "vllm_0_30_0")
        source_framework = "sglang" if sglang_source else "vllm"
        mla_sources = (*glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce", "vllm_0_27_1_deepseek_v3_sync_fp8",
            "vllm_0_11_deepseek_v3_sync_fp8")
        glm4_source = self.symbolic_forward_source == "vllm_0_11_glm4_sync"
        static_glm4_logits = (glm4_source and self.target_sampling_source in (None, "vllm_0_11_plain_sampler_bf16")
            and not self.graph_execution.enabled and self.mtp_draft_sampling_source is None
            and self.mtp_proposer_execution_source is None and self.model_source_inputs is None)
        dense011_source = (self.symbolic_forward_source == "cuda_eager_inplace"
            and self.model_execution_version == "vllm_0_11_0")
        dense024_source = (self.symbolic_forward_source == "cuda_eager_inplace"
            and self.model_execution_version == "vllm_0_24_0")
        dense018_source = (self.symbolic_forward_source == "vllm_ascend_eager"
            and self.model_execution_version == "vllm_0_18_0")
        dense_fp16_sampler = (self.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager")
            and self.target_sampling_source in ("vllm_0_11_plain_greedy_fp16", "vllm_0_30_plain_greedy_fp16")
            and self.mtp_draft_sampling_source is None)
        glm4_mtp = self.mtp_draft_sampling_source == "vllm_0_11_eager_full_logits_greedy"
        dsv3_mtp = self.mtp_draft_sampling_source == "vllm_0_27_1_eager_full_logits_greedy"
        glm_mtp = (self.symbolic_forward_source in glm_sources
            and self.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy")
        if glm_mtp:
            # Both absent declare per-step BackendStep transport. The actual
            # forward still requires a typed frame and current target IDs.
            transported = self.model_source_inputs is None and self.embedding_token_ids is None
            if ((not transported and (not isinstance(self.model_source_inputs, V32PaddedMTPInputs)
                    or self.embedding_token_ids is None))
                    or self.target_sampling_source != "vllm_0_30_rejection_raw_logits_bf16"):
                raise ValueError("V32 MTP requires its paired rejection source, target IDs and typed padded scenario")
        elif isinstance(self.model_source_inputs, V32PaddedMTPInputs) and not dsv3_mtp:
            raise ValueError("padded scenario requires its matching GLM or V3 legacy source")
        if dsv3_mtp:
            if (not dsv3_source or self.v3_mtp_attention_source not in (
                    "vllm_0_27_1_causal_triton_mla_source", "vllm_0_27_1_causal_flashmla_bf16_source")
                    or self.embedding_token_ids is None):
                raise ValueError("V3 MTP requires its explicit model, causal source and target IDs")
            if self.v3_mtp_attention_source == "vllm_0_27_1_causal_flashmla_bf16_source":
                if self.v3_mtp_batch_invariant is not False:
                    raise ValueError("V3 FlashMLA source requires explicit BATCH_INVARIANT=False")
            elif self.v3_mtp_batch_invariant is not None:
                raise ValueError("V3 BATCH_INVARIANT declaration requires the explicit FlashMLA source")
            if isinstance(self.model_source_inputs, V32PaddedMTPInputs):
                if any(self.model_source_inputs.discarded_rows):
                    raise NotImplementedError("V3 padded source requires the full non-discarded proposer subset")
                if self.v3_mtp_draft_input_token_ids is not None:
                    raise ValueError("V3 MTP draft IDs require a single owner: static IDs or current typed frame")
                draft_ids = self.model_source_inputs.draft_input_token_ids
            else:
                draft_ids = self.v3_mtp_draft_input_token_ids
            if (not isinstance(draft_ids, tuple) or len(draft_ids) != 2
                    or any(not isinstance(ids, tuple) or not ids for ids in draft_ids)
                    or any(type(token) is not int or not 0 <= token < 2**31 for ids in draft_ids for token in ids)):
                raise ValueError("V3 MTP requires independent explicit INT32 draft IDs for both passes")
        elif (self.v3_mtp_draft_input_token_ids is not None or self.v3_mtp_attention_source is not None
                or self.v3_mtp_batch_invariant is not None):
            raise ValueError("V3 draft input declarations require the explicit0.27.1 MTP caller")
        if self.glm_indexer_selection_source not in (None, "declared_causal_topk"):
            raise ValueError("unsupported GLM indexer selection source")
        if self.glm_prefill_attention_source not in (None, "force_sparse_mqa"):
            raise ValueError("unsupported GLM prefill attention source")
        if k3_source or self.symbolic_forward_source in mla_sources or glm4_mtp or pending_mla030:
            for limit in (self.symbolic_source_max_model_len, self.symbolic_source_max_batched_tokens):
                if type(limit) is not int or limit <= 0:
                    raise ValueError("symbolic MLA/K3 requires explicit positive constructor limits")
            if self.expert_routing.source_kind == "not_applicable":
                raise ValueError("symbolic MLA/K3 requires an explicit expert routing input")
        elif self.symbolic_source_max_model_len is not None or self.symbolic_source_max_batched_tokens is not None:
            raise ValueError("symbolic MLA constructor limits require their matching model source")
        if self.symbolic_forward_source in glm_sources:
            if self.glm_indexer_selection_source is None or self.symbolic_workspace_initial_capacity_bytes is None:
                raise ValueError("symbolic GLM requires declared indexer cardinality and workspace entry history")
        elif not pending_mla030 and (self.glm_indexer_selection_source is not None or self.glm_prefill_attention_source is not None):
            raise ValueError("GLM source options require an explicit GLM forward")
        if self.symbolic_workspace_initial_capacity_bytes is not None:
            if (not isinstance(self.symbolic_workspace_initial_capacity_bytes, tuple)
                    or not self.symbolic_workspace_initial_capacity_bytes
                    or any(type(size) is not int or size < 0 for size in self.symbolic_workspace_initial_capacity_bytes)):
                raise ValueError("symbolic workspace history requires a nonempty tuple of nonnegative integer capacities")
            if not pending_mla030 and self.symbolic_forward_source not in ("vllm_0_30_v4_sync_fp8_ds_mla", *glm_sources):
                raise ValueError("symbolic workspace history requires a declared V4 or GLM source")
        if self.v4_indexer_selection_source not in (None, "declared_causal_topk"):
            raise ValueError("unsupported V4 indexer selection source")
        if self.v4_indexer_selection_source is not None and (
                self.model_execution_version != "vllm_0_30_0"
                or self.v4_eager_metadata_source != "vllm_0_30_eager_dict_all_valid"
                or self.symbolic_forward_source != "vllm_0_30_v4_sync_fp8_ds_mla"):
            raise ValueError("V4 indexer selection declaration requires explicit0.30 eager source metadata")
        if self.symbolic_source_max_requests is not None:
            if not (glm4_mtp or dsv3_mtp or glm_mtp) or type(self.symbolic_source_max_requests) is not int or self.symbolic_source_max_requests <= 0:
                raise ValueError("symbolic max requests requires its declared MTP source")
        elif glm4_mtp or dsv3_mtp or glm_mtp:
            raise ValueError("MTP requires its explicit constructor request capacity")
        if self.symbolic_forward_source not in (None, "cuda_eager_inplace", "vllm_ascend_eager", "vllm_0_11_glm4_sync", "vllm_0_30_v4_sync_fp8_ds_mla", "vllm_0_30_k3_mxfp4_sync", *sglang_sources, *mla_sources):
            raise ValueError("unsupported symbolic_forward_source")
        if k3_source and self.expert_routing.source_kind not in ("explicit_assumption", "profile_observation", "replay_observation"):
            raise ValueError("K3 source requires an explicit expert-routing input")
        symbolic_decode_graph = ((self.symbolic_forward_source == "cuda_eager_inplace" or pending_dense011 or pending_dense024)
            and self.graph_execution.uniform_decode_query_len == 1
            and ((self.graph_execution.configured_mode == "FULL_DECODE_ONLY"
                  and not self.graph_execution.piecewise_eager_workload_kinds)
                 or (self.graph_execution.configured_mode in ("PIECEWISE", "FULL_AND_PIECEWISE")
                     and set(self.graph_execution.piecewise_eager_workload_kinds) == {"flash_attn", "kv_update"})))
        symbolic_decode_graph = symbolic_decode_graph or ((dense018_source or pending_dense018)
            and self.graph_execution.uniform_decode_query_len == 1
            and ((self.graph_execution.configured_mode == "FULL"
                    and not self.graph_execution.piecewise_eager_workload_kinds)
                or (self.graph_execution.configured_mode == "PIECEWISE"
                    and set(self.graph_execution.piecewise_eager_workload_kinds) == {"flash_attn", "kv_update"})))
        sglang_decode_graph = (sglang_source and self.graph_execution.configured_mode == "FULL_DECODE_ONLY"
            and self.graph_execution.uniform_decode_query_len == 1
            and not self.graph_execution.piecewise_eager_workload_kinds)
        if sglang_decode_graph and self.model_source_inputs is not None and isinstance(self.model_source_inputs, SGLangEagleChainState):
            state = self.model_source_inputs
            if state.proposal_depth != 2:
                raise NotImplementedError("SGLang EAGLE graph source currently requires its existing topk1/steps2 branch")
            batch = len(state.request_ids)
            if state.phase == "verify":
                from .traced_sglang import resolve_eagle_capture_batch
                resolve_eagle_capture_batch(state, self.graph_execution.capture_sizes)
        kimi_graph = ((self.symbolic_forward_source == "vllm_0_30_k2_6_int4_sync_explicit_allreduce" or pending_mla030)
            and self.graph_execution.uniform_decode_query_len == 1
            and ((self.graph_execution.configured_mode in ("FULL", "FULL_DECODE_ONLY")
                    and not self.graph_execution.piecewise_eager_workload_kinds)
                or (self.graph_execution.configured_mode in ("PIECEWISE", "FULL_AND_PIECEWISE")
                    and set(self.graph_execution.piecewise_eager_workload_kinds) == {"flash_attn", "kv_update"})))
        symbolic_decode_graph = symbolic_decode_graph or sglang_decode_graph or kimi_graph
        if self.symbolic_forward_source is not None:
            if self.model_execution_version != source_version or self.serving_framework != source_framework:
                raise ValueError("symbolic forward requires its explicit framework/source version")
            if ((self.graph_execution.enabled and not symbolic_decode_graph)
                    or self.mtp_proposer_execution_source is not None):
                raise NotImplementedError("initial symbolic forward requires eager target-off source")
            if dsv3_source and self.target_sampling_source not in (
                    ("vllm_0_27_1_greedy_rejection_bf16",) if dsv3_mtp
                    else (None, "vllm_0_27_1_plain_greedy_bf16")):
                raise NotImplementedError("DeepSeek-V3 target requires its paired explicit0.27.1 greedy source")
            if sglang_source and self.target_sampling_source not in (None, "sglang_0_5_19_plain_greedy_fp32"):
                raise NotImplementedError("SGLang source requires its explicit FP32 plain greedy sampler")
            if self.target_sampling_source == "sglang_0_5_19_plain_greedy_fp32" and self.symbolic_forward_source not in (
                    "sglang_0_5_19_deepseek_v3_target_fp8", "sglang_0_5_19_deepseek_v3_target_fp8_flashinfer_ar_norm"):
                raise ValueError("SGLang plain greedy requires a complete target source profile")
            static_model_only = (self.symbolic_forward_source == "cuda_eager_inplace"
                and self.model_execution_version == "vllm_0_30_0"
                and self.target_sampling_source in (None, "vllm_0_30_plain_sampler_bf16")
                and not self.graph_execution.enabled and self.mtp_draft_sampling_source is None
                and self.model_source_inputs is None
                and self.expert_routing.source_kind in ("not_applicable", "explicit_assumption", "profile_observation"))
            static_mla_logits = (self.symbolic_forward_source in (
                    *glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce", "vllm_0_30_v4_sync_fp8_ds_mla")
                and self.target_sampling_source in (None, "vllm_0_30_plain_sampler_bf16") and not self.graph_execution.enabled
                and self.mtp_draft_sampling_source is None and self.model_source_inputs is None)
            if not (dsv3_source or sglang_source or static_model_only or static_mla_logits or static_glm4_logits or dense_fp16_sampler) and self.target_sampling_source != ("vllm_0_30_rejection_raw_logits_bf16" if glm_mtp else
                    "vllm_0_11_greedy_rejection_bf16" if glm4_mtp else
                    "vllm_0_11_plain_greedy_bf16" if glm4_source or dense011_source or dsv3_source011 else
                    "vllm_0_18_plain_greedy_bf16" if dense018_source else
                    "vllm_0_24_plain_greedy_bf16" if dense024_source else "vllm_0_30_plain_greedy_bf16"):
                raise NotImplementedError("initial symbolic forward requires declared plain greedy sampling")
            expected_layout = "paged_varlen" if self.symbolic_forward_source in ("cuda_eager_inplace", "vllm_ascend_eager", "vllm_0_11_glm4_sync") else "mla"
            if self.attention_kv_layout != expected_layout or self.attention_kv_distribution != "uniform":
                raise NotImplementedError("symbolic forward requires its declared uniform logical KV layout")
            if self.symbolic_forward_source == "vllm_0_30_v4_sync_fp8_ds_mla" and (
                    self.v4_eager_metadata_source != "vllm_0_30_eager_dict_all_valid"
                    or self.v4_routing_source not in ("vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text")
                    or self.v4_source_max_model_len is None or self.v4_source_max_batched_tokens is None
                    or self.v4_source_use_deep_gemm is None):
                raise ValueError("symbolic V4 requires explicit eager metadata, modular routing, batch/context limits and DeepGEMM source branch")
        if self.v4_source_use_deep_gemm is not None:
            if type(self.v4_source_use_deep_gemm) is not bool:
                raise TypeError("v4_source_use_deep_gemm must be boolean or None")
            if self.model_execution_version != "vllm_0_30_0" or self.serving_framework != "vllm":
                raise ValueError("V4 DeepGEMM env declaration requires explicit vLLM 0.30")
            if self.graph_execution.enabled:
                raise NotImplementedError("V4 DeepGEMM env declaration currently requires eager source execution")
        if self.v4_source_prompt_lengths is not None:
            if (not isinstance(self.v4_source_prompt_lengths, tuple) or not self.v4_source_prompt_lengths
                    or any(type(value) is not int or value < 0 for value in self.v4_source_prompt_lengths)):
                raise ValueError("v4_source_prompt_lengths requires a nonempty tuple of nonnegative integer prompt lengths")
            if self.v4_eager_metadata_source is None:
                raise ValueError("v4_source_prompt_lengths requires explicit V4 source metadata")
        if self.embedding_token_ids is not None:
            if not isinstance(self.embedding_token_ids, tuple) or not self.embedding_token_ids:
                raise TypeError("embedding_token_ids requires a nonempty tuple")
            if any(type(value) is not int or not 0 <= value < 2**31 for value in self.embedding_token_ids):
                raise ValueError("embedding_token_ids requires nonnegative INT32 token IDs")
            if self.model_execution_version != source_version or self.serving_framework != source_framework:
                raise ValueError("embedding_token_ids requires its explicit framework/source version")
            if ((self.graph_execution.enabled and not symbolic_decode_graph)
                    or self.mtp_proposer_execution_source is not None):
                raise NotImplementedError("embedding_token_ids requires eager target-off source")
        if self.target_sampling_source not in (None, "sglang_0_5_19_plain_greedy_fp32", "vllm_0_30_plain_sampler_bf16",
                                               "vllm_0_27_1_plain_greedy_bf16",
                                               "vllm_0_27_1_greedy_rejection_bf16",
                                               "vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_sampler_bf16",
                                               "vllm_0_24_plain_greedy_bf16",
                                               "vllm_0_18_plain_greedy_bf16",
                                               "vllm_0_11_plain_greedy_fp16",
                                               "vllm_0_11_greedy_rejection_bf16",
                                               "vllm_0_30_plain_greedy_bf16",
                                               "vllm_0_30_plain_greedy_fp16",
                                               "vllm_0_30_rejection_raw_logits_bf16"):
            raise ValueError("unsupported target_sampling_source")
        if self.target_sampling_source is not None:
            if self.target_sampling_source == "sglang_0_5_19_plain_greedy_fp32":
                if (not sglang_source or self.model_execution_version != "sglang_0_5_19"
                        or self.serving_framework != "sglang"):
                    raise ValueError("SGLang sampler requires its matching target source/framework")
            elif self.model_execution_version != ("vllm_0_18_0" if self.target_sampling_source == "vllm_0_18_plain_greedy_bf16" else "vllm_0_24_0" if self.target_sampling_source == "vllm_0_24_plain_greedy_bf16" else "vllm_0_27_1" if self.target_sampling_source in (
                    "vllm_0_27_1_plain_greedy_bf16", "vllm_0_27_1_greedy_rejection_bf16") else "vllm_0_11_0" if self.target_sampling_source in (
                    "vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_greedy_fp16", "vllm_0_11_plain_sampler_bf16",
                    "vllm_0_11_greedy_rejection_bf16") else "vllm_0_30_0") or self.serving_framework != "vllm":
                raise ValueError("target sampling source requires its explicit vLLM source version")
            if self.target_sampling_source == "vllm_0_11_plain_sampler_bf16" and not static_glm4_logits:
                raise ValueError("vLLM0.11 cast-only sampler requires its eager target-off GLM4 source")
            if self.target_sampling_source == "vllm_0_27_1_plain_greedy_bf16" and not dsv3_source:
                raise ValueError("vLLM0.27.1 target sampler requires its DeepSeek-V3 common source")
            if self.target_sampling_source == "vllm_0_27_1_greedy_rejection_bf16" and not dsv3_mtp:
                raise ValueError("vLLM0.27.1 rejection sampler requires its paired V3 MTP caller")
            if self.target_sampling_source in ("vllm_0_11_plain_greedy_bf16", "vllm_0_11_plain_greedy_fp16") and not (glm4_source or dense011_source or pending_dense011 or dsv3_source011):
                raise ValueError("vLLM0.11 target sampler requires its declared dense, GLM4 or DeepSeek-V3 common source")
            if self.target_sampling_source == "vllm_0_24_plain_greedy_bf16" and not (dense024_source or pending_dense024):
                raise ValueError("vLLM0.24 target sampler requires its declared dense common source")
            if self.target_sampling_source == "vllm_0_18_plain_greedy_bf16" and not (dense018_source or pending_dense018):
                raise ValueError("vLLM0.18 target sampler requires its declared Ascend dense source")
            if self.target_sampling_source == "vllm_0_11_greedy_rejection_bf16" and not glm4_mtp:
                raise ValueError("vLLM0.11 rejection sampler requires its paired GLM4 MTP caller")
            if ((self.graph_execution.enabled and not symbolic_decode_graph)
                    or self.mtp_proposer_execution_source is not None):
                raise NotImplementedError("target sampler source requires eager and excludes GPU-v2")
            if (self.target_sampling_source in ("vllm_0_11_plain_sampler_bf16", "vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16")
                    and self.mtp_draft_sampling_source is not None):
                raise NotImplementedError("plain target sampler source excludes MTP")
        if self.mtp_draft_sampling_source not in (None, "vllm_0_30_legacy_full_logits_greedy", "vllm_0_11_eager_full_logits_greedy",
                "vllm_0_27_1_eager_full_logits_greedy"):
            raise ValueError("unsupported mtp_draft_sampling_source")
        if self.mtp_draft_sampling_source is not None:
            if self.model_execution_version != ("vllm_0_27_1" if dsv3_mtp else "vllm_0_11_0" if glm4_mtp else "vllm_0_30_0") or self.serving_framework != "vllm":
                raise ValueError("MTP sampling source requires explicit vLLM 0.30")
            if glm4_mtp and (not glm4_source or self.target_sampling_source != "vllm_0_11_greedy_rejection_bf16"):
                raise ValueError("GLM4 MTP requires its explicit0.11 target/rejection/proposer source pair")
            if self.mtp_proposer_execution_source is not None or self.graph_execution.enabled:
                raise NotImplementedError("legacy MTP sampling source requires eager and excludes GPU-v2")
        if self.v4_routing_source not in (None, "vllm_0_30_triton_top6_pure_text",
                                          "vllm_0_30_modular_cuda_top6_pure_text"):
            raise ValueError("unsupported v4_routing_source")
        if self.v4_routing_source is not None:
            if self.model_execution_version != "vllm_0_30_0" or self.serving_framework != "vllm":
                raise ValueError("V4 routing source requires explicit vLLM 0.30")
            if self.graph_execution.enabled:
                raise NotImplementedError("V4 routing source currently requires eager")
        if self.v4_eager_metadata_source not in (None, "vllm_0_30_eager_dict_all_valid"):
            raise ValueError("unsupported v4_eager_metadata_source")
        if self.v4_eager_metadata_source is not None:
            if self.model_execution_version != "vllm_0_30_0" or self.serving_framework != "vllm":
                raise ValueError("V4 metadata source requires explicit vLLM 0.30")
        if self.v4_source_max_model_len is not None:
            if type(self.v4_source_max_model_len) is not int or self.v4_source_max_model_len <= 0:
                raise ValueError("v4_source_max_model_len must be a positive integer")
            if self.v4_eager_metadata_source is None:
                raise ValueError("v4_source_max_model_len requires explicit V4 source metadata")
        if self.v4_source_max_batched_tokens is not None:
            if type(self.v4_source_max_batched_tokens) is not int or self.v4_source_max_batched_tokens <= 0:
                raise ValueError("v4_source_max_batched_tokens must be a positive integer")
            if self.v4_eager_metadata_source is None:
                raise ValueError("v4_source_max_batched_tokens requires explicit V4 source metadata")
        if self.mtp_proposer_execution_source not in (None, "vllm_0_30_gpu_v2"):
            raise ValueError("unsupported mtp_proposer_execution_source")
        if self.mtp_proposer_execution_source is not None:
            if self.model_execution_version != "vllm_0_30_0" or self.serving_framework != "vllm":
                raise ValueError("GPU-v2 MTP source requires the explicit vLLM 0.30 model path")
        if self.model_execution_version not in ("legacy", "vllm_0_30_0", "vllm_0_24_0", "vllm_0_18_0", "vllm_0_11_0", "vllm_0_27_1", "sglang_0_5_19"):
            raise ValueError("unsupported model_execution_version")
        if self.model_execution_version == "sglang_0_5_19" and not sglang_source:
            raise ValueError("SGLang0.5.19 model execution requires its explicit shared source profile")
        if sglang_source and self.expert_routing.source_kind == "not_applicable":
            raise ValueError("SGLang MoE source requires explicit expert routing")
        if self.model_execution_version == "vllm_0_27_1" and not dsv3_source:
            raise ValueError("vLLM0.27.1 model execution requires its explicit DeepSeek-V3 source")
        if self.model_execution_version == "vllm_0_11_0" and not (glm4_source or dense011_source or pending_dense011 or dsv3_source011):
            raise ValueError("vLLM0.11 model execution requires its declared dense, GLM4 or DeepSeek-V3 common source")
        if (self.model_execution_version == "vllm_0_30_0"
                and self.serving_framework != "vllm"):
            raise ValueError("vLLM model-source version requires vLLM framework")
        if self.serving_framework not in ("vllm", "sglang"):
            raise ValueError(
                f"unsupported serving_framework: {self.serving_framework!r}")
        if not isinstance(self.graph_execution, GraphExecutionConfig):
            raise TypeError(
                "graph_execution must be a GraphExecutionConfig")
        if self.v4_eager_metadata_source is not None and self.graph_execution.enabled:
            raise NotImplementedError("V4 eager metadata source requires capture-disabled eager")
        if self.mtp_proposer_execution_source is not None and self.graph_execution.enabled:
            raise NotImplementedError("GPU-v2 MTP source projection currently requires eager")
        if not isinstance(self.attention_kv_layout, str):
            raise TypeError("attention_kv_layout must be a string")
        if self.attention_kv_layout not in (
                "contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"):
            raise ValueError(
                "unsupported attention_kv_layout: "
                f"{self.attention_kv_layout!r}")
        if not isinstance(self.attention_kv_distribution, str):
            raise TypeError("attention_kv_distribution must be a string")
        if self.attention_kv_distribution not in ("uniform", "skew"):
            raise ValueError(
                "unsupported attention_kv_distribution: "
                f"{self.attention_kv_distribution!r}")


def _shared_routing_group_id(
    *,
    step_id: str,
    layer_index: Optional[int],
    tokens: int,
    num_experts: int,
    topk: int,
    model_config_sha256: str,
    pool_name: str,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    expert_rank_group: Tuple[int, ...],
    expert_placement_strategy: str,
) -> str:
    if layer_index is None:
        block_class = "mtp"
    else:
        block_class = "main_decoder"
    return _canonical_json_sha256({
        "schema_version": 1,
        "step_id": step_id,
        "block_class": block_class,
        "tokens": tokens,
        "num_experts": num_experts,
        "topk": topk,
        "model_config_sha256": model_config_sha256,
        "pool_name": pool_name,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "expert_rank_group": tuple(expert_rank_group),
        "expert_placement_strategy": expert_placement_strategy,
    })


def _iter_seeded_expert_routes(
    num_tokens: int,
    num_experts: int,
    topk: int,
    *,
    source_id: str,
    root_seed: int,
    moe_invocation_id: str,
    distribution: str = _UNIFORM_ROUTING_DISTRIBUTION,
    hot_expert_count: Optional[int] = None,
    hot_expert_weight: Optional[int] = None,
) -> Iterator[Tuple[int, ...]]:
    """Sample per-token distinct expert IDs for routing and transport alike."""

    for name, value in (
            ("num_tokens", num_tokens), ("num_experts", num_experts),
            ("topk", topk), ("root_seed", root_seed)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_tokens < 0 or num_experts < 1 or not 1 <= topk <= num_experts:
        raise ValueError("invalid seeded MoE routing dimensions")
    if root_seed < 0:
        raise ValueError("root_seed cannot be negative")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    if not isinstance(moe_invocation_id, str) or not moe_invocation_id:
        raise ValueError("moe_invocation_id must be a non-empty string")
    _validate_seeded_distribution(
        distribution,
        hot_expert_count,
        hot_expert_weight,
        num_experts=num_experts,
    )
    sample_space = 1 << 64
    prefix = (
        f"midlayer:{_SEEDED_ROUTING_ALGORITHM}:{source_id}:"
        f"{root_seed}:{moe_invocation_id}:{distribution}:"
        f"{hot_expert_count}:{hot_expert_weight}"
    ).encode("utf-8")
    if distribution == _UNIFORM_ROUTING_DISTRIBUTION:
        hot_experts = ()
        cold_experts = tuple(range(num_experts))
        hot_weight = 1
    elif distribution == _SKEW_ROUTING_DISTRIBUTION:
        ranked_experts = tuple(sorted(
            range(num_experts),
            key=lambda expert_id: hashlib.sha256(
                prefix + b":hot-rank:" + str(expert_id).encode("ascii")
            ).digest(),
        ))
        hot_experts = ranked_experts[:hot_expert_count]
        hot_set = set(hot_experts)
        cold_experts = tuple(
            expert_id for expert_id in range(num_experts)
            if expert_id not in hot_set)
        hot_weight = hot_expert_weight
    else:
        raise AssertionError("validated expert routing distribution vanished")
    for token_index in range(num_tokens):
        selected = []
        available_hot = list(hot_experts)
        available_cold = list(cold_experts)
        counter = 0
        for _ in range(topk):
            total_weight = (
                len(available_hot) * hot_weight + len(available_cold))
            unbiased_limit = sample_space - sample_space % total_weight
            while True:
                digest = hashlib.sha256(
                    prefix
                    + b":"
                    + str(token_index).encode("ascii")
                    + b":"
                    + str(counter).encode("ascii")
                ).digest()
                counter += 1
                value = int.from_bytes(digest[:8], "big")
                if value < unbiased_limit:
                    break
            slot = value % total_weight
            hot_slots = len(available_hot) * hot_weight
            if slot < hot_slots:
                expert_id = available_hot.pop(slot // hot_weight)
            else:
                expert_id = available_cold.pop(slot - hot_slots)
            selected.append(expert_id)
        yield tuple(selected)


def seeded_expert_routes(
    num_tokens: int, num_experts: int, topk: int, *, source_id: str,
    root_seed: int, moe_invocation_id: str,
    distribution: str = _UNIFORM_ROUTING_DISTRIBUTION,
    hot_expert_count: Optional[int] = None, hot_expert_weight: Optional[int] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Materialize the stable distinct per-token draws used by dispatch."""
    return tuple(_iter_seeded_expert_routes(num_tokens, num_experts, topk,
        source_id=source_id, root_seed=root_seed, moe_invocation_id=moe_invocation_id,
        distribution=distribution, hot_expert_count=hot_expert_count,
        hot_expert_weight=hot_expert_weight))


def seeded_expert_token_counts(
    num_tokens: int, num_experts: int, topk: int, *, source_id: str,
    root_seed: int, moe_invocation_id: str,
    distribution: str = _UNIFORM_ROUTING_DISTRIBUTION,
    hot_expert_count: Optional[int] = None, hot_expert_weight: Optional[int] = None,
) -> Tuple[int, ...]:
    """Histogram of the same stable draws used by token dispatch."""
    routes = _iter_seeded_expert_routes(num_tokens, num_experts, topk, source_id=source_id,
        root_seed=root_seed, moe_invocation_id=moe_invocation_id, distribution=distribution,
        hot_expert_count=hot_expert_count, hot_expert_weight=hot_expert_weight)
    first = next(routes, None)  # Execute dimension/distribution validation before allocating the histogram.
    counts = [0] * num_experts
    for selected in chain(() if first is None else (first,), routes):
        for expert_id in selected:
            counts[expert_id] += 1
    return tuple(counts)


def profiled_expert_token_counts(
    num_tokens: int,
    topk: int,
    *,
    probabilities: Tuple[float, ...],
    profile_sha256: str,
    routing_group_id: str,
) -> Tuple[int, ...]:
    """Deterministically sample one shared histogram from profile weights."""

    for name, value in (("num_tokens", num_tokens), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_tokens < 0 or topk < 1:
        raise ValueError("invalid profiled MoE routing dimensions")
    _require_sha256("profile_sha256", profile_sha256)
    _require_sha256("routing_group_id", routing_group_id)
    weights = tuple(probabilities)
    if not weights:
        raise ValueError("profiled MoE routing requires expert probabilities")
    if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in weights):
        raise ValueError(
            "profiled MoE routing probabilities must be finite and non-negative")
    if topk > len(weights):
        raise ValueError("profiled MoE routing topk cannot exceed expert count")
    if sum(weight > 0 for weight in weights) < topk:
        raise ValueError(
            "profiled MoE routing requires at least topk positive probabilities")
    algorithm = _PROFILE_ROUTING_ALGORITHM
    weight_types = tuple(type(weight) for weight in weights)
    if (type(num_tokens) is int and type(topk) is int
            and type(profile_sha256) is str and type(routing_group_id) is str and type(algorithm) is str
            and all(kind in (int, float) for kind in weight_types)):
        return _cached_profiled_expert_counts(num_tokens, topk, weights,
            weight_types, profile_sha256, routing_group_id, algorithm)
    return _draw_profiled_expert_counts(num_tokens, topk, weights,
        profile_sha256, routing_group_id, algorithm)


_PROFILED_EXPERT_COUNTS = None


def _cached_profiled_expert_counts(num_tokens, topk, weights, weight_types,
                                  profile_sha256, routing_group_id, algorithm):
    global _PROFILED_EXPERT_COUNTS
    if _PROFILED_EXPERT_COUNTS is None:
        from src.serving.kernel_lowering.execution_timeline import _ExperimentCache
        _PROFILED_EXPERT_COUNTS = _ExperimentCache()
    key = (num_tokens, topk, weights, weight_types, profile_sha256, routing_group_id, algorithm)
    hash(key)
    if key in _PROFILED_EXPERT_COUNTS:
        _PROFILED_EXPERT_COUNTS.move_to_end(key)
        return _PROFILED_EXPERT_COUNTS[key]
    result = _profiled_expert_counts_uncached(num_tokens, topk, weights,
        weight_types, profile_sha256, routing_group_id, algorithm)
    _PROFILED_EXPERT_COUNTS[key] = result
    while len(_PROFILED_EXPERT_COUNTS) > 128:
        _PROFILED_EXPERT_COUNTS.popitem(last=False)
    return result


def _profiled_expert_counts_uncached(num_tokens, topk, weights, weight_types,
                                  profile_sha256, routing_group_id, algorithm):
    # Exact immutable inputs, including scalar types, name the existing draws.
    # Boundary checks and per-layer provenance stay outside this finite memo.
    if all(kind is float for kind in weight_types):
        from .profiled_routing_native import sample
        prefix = (f"midlayer:{algorithm}:{profile_sha256}:{routing_group_id}").encode("utf-8")
        ratios = tuple(
            int.from_bytes(hashlib.sha256(
                prefix + b":" + str(token_index).encode("ascii") + b":"
                + str(route_index).encode("ascii")).digest()[:8], "big")
            / (1 << 64)
            for token_index in range(num_tokens) for route_index in range(topk))
        return sample(weights, ratios, topk)
    return _draw_profiled_expert_counts(num_tokens, topk, weights,
        profile_sha256, routing_group_id, algorithm)


def _draw_profiled_expert_counts(num_tokens, topk, weights,
                                profile_sha256, routing_group_id, algorithm):
    counts = [0] * len(weights)
    prefix = (
        f"midlayer:{algorithm}:{profile_sha256}:"
        f"{routing_group_id}"
    ).encode("utf-8")
    sample_space = 1 << 64
    for token_index in range(num_tokens):
        available = list(range(len(weights)))
        for route_index in range(topk):
            total_weight = sum(weights[expert_id] for expert_id in available)
            if not math.isfinite(total_weight) or total_weight <= 0:
                raise ValueError(
                    "profiled MoE routing exhausted positive expert weights")
            digest = hashlib.sha256(
                prefix
                + b":"
                + str(token_index).encode("ascii")
                + b":"
                + str(route_index).encode("ascii")
            ).digest()
            target = int.from_bytes(digest[:8], "big") / sample_space * total_weight
            cumulative = 0.0
            selected_index = None
            for available_index, expert_id in enumerate(available):
                cumulative += weights[expert_id]
                if target < cumulative:
                    selected_index = available_index
                    break
            if selected_index is None:
                selected_index = len(available) - 1
            expert_id = available.pop(selected_index)
            counts[expert_id] += 1
    return tuple(counts)


def profiled_layer_expert_token_counts(
    num_tokens: int,
    topk: int,
    *,
    probabilities: Tuple[float, ...],
    profile_sha256: str,
    routing_group_id: str,
) -> Tuple[int, ...]:
    """Deterministically sample one MoE layer's histogram from its profile row.

    Each token draws ``topk`` distinct experts, weighted by the row, without
    replacement; the Gumbel-top-k construction gives the same distribution as
    sequential weighted draws.  The PCG64 stream is seeded from the profile
    and the routing group identity.
    """

    import numpy as np

    for name, value in (("num_tokens", num_tokens), ("topk", topk)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if num_tokens < 0 or topk < 1:
        raise ValueError("invalid profiled MoE routing dimensions")
    _require_sha256("profile_sha256", profile_sha256)
    _require_sha256("routing_group_id", routing_group_id)
    weights = np.asarray(probabilities, dtype=np.float64)
    if weights.ndim != 1 or not weights.size:
        raise ValueError("profiled MoE routing requires expert probabilities")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            "profiled MoE routing probabilities must be finite and non-negative")
    if int((weights > 0).sum()) < topk:
        raise ValueError(
            "profiled MoE routing requires at least topk positive probabilities")
    seed = int.from_bytes(hashlib.sha256(
        f"midlayer:{_PROFILE_LAYER_ROUTING_ALGORITHM}:{profile_sha256}:"
        f"{routing_group_id}".encode("utf-8")).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    with np.errstate(divide="ignore"):
        log_weights = np.log(weights)
    keys = log_weights[None, :] + generator.gumbel(
        size=(num_tokens, weights.size))
    selected = np.argpartition(-keys, topk - 1, axis=1)[:, :topk]
    counts = np.bincount(selected.ravel(), minlength=weights.size)
    return tuple(int(count) for count in counts)


def _validate_seeded_distribution(
    distribution: object,
    hot_expert_count: object,
    hot_expert_weight: object,
    *,
    num_experts: Optional[int],
) -> None:
    if distribution == _UNIFORM_ROUTING_DISTRIBUTION:
        if hot_expert_count is not None or hot_expert_weight is not None:
            raise ValueError(
                "maximum-entropy uniform routing cannot carry hot-expert "
                "parameters")
        return
    if distribution != _SKEW_ROUTING_DISTRIBUTION:
        raise ValueError(
            f"unknown explicit expert routing distribution: {distribution!r}")
    for name, value in (
            ("hot_expert_count", hot_expert_count),
            ("hot_expert_weight", hot_expert_weight)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expert routing {name} must be an integer")
    if hot_expert_count < 1:
        raise ValueError("expert routing hot_expert_count must be positive")
    if hot_expert_weight <= 1:
        raise ValueError(
            "expert routing hot_expert_weight must be greater than one")
    if num_experts is not None and hot_expert_count >= num_experts:
        raise ValueError(
            "skew routing hot_expert_count must be smaller than num_experts")


def vllm_0_30_source_model_inputs(step, *, embedding_token_ids, prompt_lengths, num_speculative_tokens=0,
                                decode_threshold=None, require_uniform=False):
    """Source decode-prefix packing of declared inputs, not a runtime receipt."""
    if type(num_speculative_tokens) is not int or num_speculative_tokens < 0:
        raise NotImplementedError("source packing requires an explicit nonnegative integer speculation depth")
    # DeepseekV32IndexerMetadataBuilder keeps speculative_config in both the
    # target and legacy drafter: next_n=1+num_speculative_tokens, PCP1.
    if decode_threshold is None:
        decode_threshold = 1 + num_speculative_tokens
    elif type(decode_threshold) is not int or decode_threshold <= 0:
        raise ValueError("source packing decode threshold requires a positive integer")
    if type(require_uniform) is not bool:
        raise TypeError("source packing require_uniform must be boolean")
    regions = []
    offsets, total = [], 0
    prompts = prompt_lengths
    if prompts is not None and len(prompts) != len(step.requests):
        raise ValueError("V4 source prompt lengths must cover every caller request")
    for index, request in enumerate(step.requests):
        offsets.append(total)
        total += request.scheduled_tokens
        if request.computed_tokens == 0:
            regions.append(3)
        elif request.scheduled_tokens > decode_threshold:
            regions.append(2)
        elif prompts is not None:
            regions.append(0 if request.computed_tokens >= prompts[index] else 1)
        elif request.phase == "decode":
            regions.append(0)
        else:
            raise NotImplementedError("V4 short-extend classification requires explicit num_prompt_tokens via v4_source_prompt_lengths")
    targets = sorted(regions)
    misplaced = [index for index, region in enumerate(regions) if region != targets[index]]
    sources = sorted(misplaced, key=lambda index: regions[index])
    order = list(range(len(regions)))
    for destination, source in zip(misplaced, sources, strict=True):
        order[destination] = source
    token_rows = tuple(row for index in order for row in range(offsets[index], offsets[index] + step.requests[index].scheduled_tokens))
    if embedding_token_ids is not None:
        if len(embedding_token_ids) != total:
            raise ValueError("embedding_token_ids must cover every scheduled token row before source packing")
        embedding_token_ids = tuple(embedding_token_ids[row] for row in token_rows)
    packed_step = replace(step, requests=tuple(step.requests[index] for index in order))
    # The backend's split uses a prefix of query lengths, independently of
    # the Scheduler's phase or done-prefilling status. Uniform backends end
    # that prefix at the first different query length.
    if require_uniform:
        if not packed_step.requests:
            raise ValueError("uniform source packing requires a nonempty request batch")
        first_query = packed_step.requests[0].scheduled_tokens
        first_prefill = (0 if first_query > decode_threshold else next(
            (index for index, request in enumerate(packed_step.requests)
                if request.scheduled_tokens != first_query), len(order)))
    else:
        first_prefill = next((index for index, request in enumerate(packed_step.requests)
            if request.scheduled_tokens > decode_threshold), len(order))
    attention_phases = ["decode" if index < first_prefill else "prefill" for index in range(len(order))]
    metadata = {"program": ("vllm_0_30_spec_decode_prefix_model_input" if num_speculative_tokens else
        "vllm_0_30_decode_prefix_model_input"), "decode_threshold": decode_threshold,
        "require_uniform": require_uniform,
        "initial_request_ids": [request.request_id for request in step.requests],
        "request_permutation": order, "token_row_permutation": list(token_rows),
        "model_request_ids": [request.request_id for request in packed_step.requests],
        "model_query_lengths": [request.scheduled_tokens for request in packed_step.requests],
        "model_attention_phases": attention_phases,
        "model_reorder_regions": [regions[index] for index in order],
        "initial_prompt_lengths": None if prompts is None else list(prompts),
        "model_positions": [position for request in packed_step.requests for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens)],
        "initial_order_source": "explicit caller ScheduleStep order; not observed persistent InputBatch",
        "output_request_ids": [request.request_id for request in packed_step.requests],
        "extra_GPU_permutation_copy": False, "actual_runtime_bound": False}
    return packed_step, embedding_token_ids, metadata


def dsv4_c128_metadata_width(max_seq_len, source_max_model_len):
    """Batch-wide next-power-of-two width, capped by source constructor extent.

    Unknown max_model_len is harmless only where every valid constructor
    capacity gives the same width. A 128-aligned lower bound is not a point
    width when the next-power-of-two interval is nontrivial.
    """
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("C128 metadata requires a positive maximum sequence length")
    count = max_seq_len // 128
    upper = max(128, 1 << (max(count, 1) - 1).bit_length())
    if source_max_model_len is None:
        minimum_capacity = ((max_seq_len + 128 * 128 - 1) // (128 * 128)) * 128
        lower = min(upper, minimum_capacity)
        if lower != upper:
            raise NotImplementedError("C128 metadata width requires v4_source_max_model_len; constructor cap changes the point width")
        return upper
    if type(source_max_model_len) is not int or source_max_model_len < max_seq_len:
        raise ValueError("v4_source_max_model_len must cover all scheduled sequence lengths")
    capacity = ((source_max_model_len + 128 * 128 - 1) // (128 * 128)) * 128
    return min(upper, capacity)


# This step-local optimization never accepts a changed nested routing implementation.
_EXPERT_ROUTING_STEP_SHARED_SOURCE = tuple((name, globals()[name], globals()[name].__code__,
        globals()[name].__defaults__,
        None if globals()[name].__kwdefaults__ is None else dict(globals()[name].__kwdefaults__))
    for name in ("_shared_routing_group_id", "_validate_seeded_distribution",
        "seeded_expert_token_counts", "profiled_expert_token_counts",
        "_routing_record_sha256", "validate_expert_routing_provenance",
        "_validate_routing_invocation", "_shared_resolution_immutable",
        "_routing_record_json_supported", "_native_rebound_routing_record_sha256",
        "_rebind_validated_routing_resolution", "_rebind_routing_resolution_native", "_routing_certificate_contexts_native", "_routing_certificate_context",
        "_certify_routing_provenance", "_construct_native_routing_provenance"))
_EXPERT_ROUTING_RESOLUTION_SOURCE = tuple((name, getattr(ExpertRoutingResolution, name),
        getattr(ExpertRoutingResolution, name).__code__,
        getattr(ExpertRoutingResolution, name).__defaults__,
        None if getattr(ExpertRoutingResolution, name).__kwdefaults__ is None
            else dict(getattr(ExpertRoutingResolution, name).__kwdefaults__))
    for name in ("__init__", "__post_init__"))


# Certificates are valid only under the same strict owner implementations.
_VALIDATED_ROUTING_PROVENANCE_TYPE = _ValidatedRoutingProvenance
_ROUTING_CERTIFICATE_VALIDATION_SOURCE = tuple((name, globals()[name], globals()[name].__code__,
    globals()[name].__defaults__, None if globals()[name].__kwdefaults__ is None
    else dict(globals()[name].__kwdefaults__))
    for name in ("_routing_record_json_supported", "_native_rebound_routing_record_sha256",
        "_routing_certificate_contexts_native", "_rebind_routing_resolution_native", "_require_string", "validate_expert_routing_provenance", "_require_sha256", "_validate_expert_histogram",
        "_validate_routing_invocation", "_routing_record_sha256", "_canonical_json_sha256",
        "_certify_routing_provenance", "_construct_native_routing_provenance"))
_ROUTING_CERTIFICATE_MAPPING_SOURCE = tuple((name, getattr(_ValidatedRoutingProvenance, name),
    getattr(_ValidatedRoutingProvenance, name).__code__)
    for name in ("__init__", "__getitem__", "__iter__", "__len__", "__contains__", "keys", "items", "values",
        "__setattr__", "__delattr__", "copy", "__deepcopy__", "__reduce__"))
_ROUTING_CERTIFICATE_CLASS_SOURCE = dict(vars(_ValidatedRoutingProvenance))
