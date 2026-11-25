"""Versioned external MTP acceptance observations for exact replay."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.serving.request_gen.request_stream import RequestStream
from src.serving.schedule.schedule_step import BackendStep


COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT = (
    "midlayer_vllm_0_11_length_only_mtp_acceptance_v1")
PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT = (
    "midlayer_vllm_0_11_seeded_decode_fresh_mtp_pd_v1")
CLOSED_LOOP_MTP_ADMISSION_ADAPTER_CONTRACT = (
    "midlayer_vllm_0_11_seeded_closed_loop_mtp_admission_v1")

_ROOT_FIELDS = {
    "schema_version",
    "artifact_id",
    "provenance",
    "scheduler_context",
    "steps",
}
_PROVENANCE_FIELDS = {
    "producer",
    "vllm_version",
    "vllm_source_commit",
    "vllm_package_manifest_sha256",
    "model_config_sha256",
}
_CONTEXT_COMMON_FIELDS = {
    "adapter_contract",
    "pool_name",
    "backend_role",
    "scheduler_policy",
    "max_batched_tokens",
    "max_requests",
    "num_kv_blocks",
    "kv_block_size_tokens",
    "max_model_len",
    "dtype",
    "num_speculative_tokens",
    "draft_model_type",
    "num_mtp_layers",
    "draft_max_model_len",
    "padded_drafter_batch",
    "prefix_oracle_mode",
    "attention_kv_layout",
    "chunked_prefill",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
    "long_prefill_token_threshold",
}
_CONTEXT_V1_FIELDS = _CONTEXT_COMMON_FIELDS | {
    "request_stream_sha256",
}
_CONTEXT_V2_FIELDS = _CONTEXT_COMMON_FIELDS | {
    "ingress_request_stream_sha256",
    "decode_admission_contract_sha256",
}
_STEP_FIELDS = {
    "step_index",
    "step_id",
    "selected_request_ids",
    "finished_request_ids",
    "preempted_request_ids",
    "mtp_drafter_executed",
    "requests",
}
_REQUEST_FIELDS = {
    "request_id",
    "phase",
    "input_tokens",
    "output_tokens",
    "computed_tokens",
    "recomputed_tokens",
    "prefix_replay_tokens",
    "resumed_from_preemption",
    "trace_prefix_hit_tokens",
    "scheduler_prefix_hit_tokens",
    "raw_model_output_tokens",
    "emitted_tokens",
    "mtp_proposed_tokens",
    "mtp_verified_tokens",
    "mtp_accepted_tokens",
    "mtp_rejected_tokens",
    "mtp_proposer_tokens",
    "mtp_next_proposed_tokens",
}


def _strict_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    normalized = {}
    for key, value in pairs:
        if key in normalized:
            raise ValueError(
                f"duplicate MTP acceptance replay JSON field: {key!r}")
        normalized[key] = value
    return normalized


def _require_integer(name: str, value: object, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"MTP acceptance replay {name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"MTP acceptance replay {name} must be positive")
    if not positive and value < 0:
        raise ValueError(
            f"MTP acceptance replay {name} cannot be negative")
    return value


def _require_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"MTP acceptance replay {name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: object) -> str:
    normalized = _require_string(name, value)
    if (len(normalized) != 64
            or any(character not in "0123456789abcdef"
                   for character in normalized)):
        raise ValueError(
            f"MTP acceptance replay {name} must be a lowercase SHA256")
    return normalized


def _require_string_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"MTP acceptance replay {name} must be a list or tuple")
    normalized = tuple(value)
    if any(not isinstance(item, str) or not item for item in normalized):
        raise ValueError(
            f"MTP acceptance replay {name} must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"MTP acceptance replay {name} must contain unique IDs")
    return normalized


def pd_decode_admission_contract_sha256(requests: RequestStream) -> str:
    """Hash seeded D admissions without projected transfer completion times.

    The contract binds the admission set and per-request fields, not the
    admission order: a decode instance physically admits requests in KV
    transfer completion order, which legitimately departs from ingress
    order when same-step handoffs carry different byte counts.  Acceptance
    sampling is keyed by (step_id, request_id), so admission order carries
    no acceptance semantics.  Sorting by request_id keeps both the planned
    and the actually-submitted side canonical.
    """

    if not isinstance(requests, RequestStream):
        raise TypeError(
            "PD decode admission contract requires a RequestStream")
    admissions = [
        {
            "request_id": request.request_id,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "prefix_hit_tokens": 0,
            "computed_tokens": request.input_tokens,
            "terminal_seed_tokens": 1,
            "initial_draft_tokens": 0,
        }
        for request in requests
        if request.output_tokens > 1
    ]
    admissions.sort(key=lambda admission: admission["request_id"])
    if not admissions:
        raise ValueError(
            "PD decode MTP replay requires at least one request with "
            "output_tokens greater than one")
    encoded = json.dumps(
        {
            "adapter_contract": PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
            "admissions": admissions,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def closed_loop_admission_contract_sha256(requests: RequestStream) -> str:
    """Hash seeded ingress admissions for a closed-loop client run.

    A zero-think-time closed-loop population submits each client's
    successor at the client's own completion instant, so Scheduler
    admission order and stamped arrival times are client-driven and
    legitimately depart from the canonical stream.  The contract binds
    the admission set and per-request seeded fields, not the admission
    order or arrival stamps: acceptance sampling is keyed by (step_id,
    request_id), so neither carries acceptance semantics.  Sorting by
    request_id keeps both the planned and the actually-submitted side
    canonical.  Fields are ingress-semantics: the request arrives with
    nothing computed.
    """

    if not isinstance(requests, RequestStream):
        raise TypeError(
            "closed-loop admission contract requires a RequestStream")
    admissions = [
        {
            "request_id": request.request_id,
            "input_tokens": request.input_tokens,
            "output_tokens": request.output_tokens,
            "prefix_hit_tokens": request.prefix_hit_tokens,
            "computed_tokens": 0,
            "terminal_seed_tokens": 1,
            "initial_draft_tokens": 0,
        }
        for request in requests
        if request.output_tokens > 1
    ]
    admissions.sort(key=lambda admission: admission["request_id"])
    if not admissions:
        raise ValueError(
            "closed-loop MTP replay requires at least one request with "
            "output_tokens greater than one")
    encoded = json.dumps(
        {
            "adapter_contract": CLOSED_LOOP_MTP_ADMISSION_ADAPTER_CONTRACT,
            "admissions": admissions,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MTPAcceptanceProvenance:
    producer: str
    vllm_version: str
    vllm_source_commit: str
    vllm_package_manifest_sha256: str
    model_config_sha256: str

    def __post_init__(self) -> None:
        if self.producer != "vllm_model_runner_observation":
            raise ValueError(
                "MTP acceptance replay producer must be "
                "'vllm_model_runner_observation'")
        _require_string("vllm_version", self.vllm_version)
        commit = _require_string(
            "vllm_source_commit", self.vllm_source_commit)
        if (len(commit) != 40
                or any(character not in "0123456789abcdef"
                       for character in commit)):
            raise ValueError(
                "MTP acceptance replay vllm_source_commit must be a "
                "lowercase 40-character Git commit")
        _require_sha256(
            "vllm_package_manifest_sha256",
            self.vllm_package_manifest_sha256,
        )
        _require_sha256(
            "model_config_sha256", self.model_config_sha256)


@dataclass(frozen=True)
class MTPAcceptanceSchedulerContext:
    adapter_contract: str
    request_stream_sha256: str | None
    ingress_request_stream_sha256: str | None
    decode_admission_contract_sha256: str | None
    pool_name: str
    backend_role: str
    scheduler_policy: str
    max_batched_tokens: int
    max_requests: int
    num_kv_blocks: int
    kv_block_size_tokens: int
    max_model_len: int
    dtype: str
    num_speculative_tokens: int
    draft_model_type: str
    num_mtp_layers: int
    draft_max_model_len: int
    padded_drafter_batch: bool
    prefix_oracle_mode: str
    attention_kv_layout: str
    chunked_prefill: bool
    max_num_partial_prefills: int
    max_long_partial_prefills: int
    long_prefill_token_threshold: int

    def __post_init__(self) -> None:
        for name in (
                "adapter_contract", "pool_name", "backend_role",
                "scheduler_policy", "dtype",
                "draft_model_type", "prefix_oracle_mode",
                "attention_kv_layout"):
            _require_string(name, getattr(self, name))
        for name in (
                "max_batched_tokens", "max_requests", "num_kv_blocks",
                "kv_block_size_tokens", "max_model_len",
                "num_speculative_tokens", "num_mtp_layers",
                "draft_max_model_len", "max_num_partial_prefills",
                "max_long_partial_prefills"):
            _require_integer(name, getattr(self, name), positive=True)
        _require_integer(
            "long_prefill_token_threshold",
            self.long_prefill_token_threshold,
            positive=False,
        )
        if not isinstance(self.padded_drafter_batch, bool):
            raise TypeError(
                "MTP acceptance replay padded_drafter_batch must be boolean")
        if not self.padded_drafter_batch:
            raise ValueError(
                "MTP acceptance replay requires the pinned padded drafter")
        if not isinstance(self.chunked_prefill, bool):
            raise TypeError(
                "MTP acceptance replay chunked_prefill must be boolean")
        if not self.chunked_prefill:
            raise ValueError(
                "MTP acceptance replay requires chunked prefill")
        if self.adapter_contract == COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT:
            _require_sha256(
                "request_stream_sha256", self.request_stream_sha256)
            if (self.ingress_request_stream_sha256 is not None
                    or self.decode_admission_contract_sha256 is not None):
                raise ValueError(
                    "colocated MTP replay cannot carry PD request identities")
            if self.backend_role != "colocated":
                raise ValueError(
                    "colocated MTP replay requires backend_role='colocated'")
        elif self.adapter_contract == PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT:
            if self.request_stream_sha256 is not None:
                raise ValueError(
                    "PD decode MTP replay cannot carry the colocated request "
                    "stream field")
            _require_sha256(
                "ingress_request_stream_sha256",
                self.ingress_request_stream_sha256,
            )
            _require_sha256(
                "decode_admission_contract_sha256",
                self.decode_admission_contract_sha256,
            )
            if self.backend_role != "decode" or self.pool_name != "decode-0":
                raise ValueError(
                    "PD decode MTP replay requires backend_role='decode' and "
                    "pool_name='decode-0'")
            if self.max_requests != 1:
                raise ValueError(
                    "PD decode MTP replay requires max_requests=1")
        else:
            raise ValueError(
                "MTP acceptance replay adapter contract is unsupported")
        if self.scheduler_policy != "fcfs":
            raise ValueError(
                "MTP acceptance replay currently requires FCFS scheduling")
        if self.prefix_oracle_mode != "trace_length_private_nonshared_v1":
            raise ValueError(
                "MTP acceptance replay requires the private trace-prefix "
                "oracle contract")
        if self.attention_kv_layout != "paged_varlen":
            raise ValueError(
                "MTP acceptance replay requires paged_varlen attention KV")

    @property
    def ingress_identity_sha256(self) -> str:
        if self.adapter_contract == COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT:
            if self.request_stream_sha256 is None:
                raise RuntimeError(
                    "colocated MTP replay lost its request stream identity")
            return self.request_stream_sha256
        elif self.adapter_contract == PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT:
            if self.ingress_request_stream_sha256 is None:
                raise RuntimeError(
                    "PD decode MTP replay lost its ingress identity")
            return self.ingress_request_stream_sha256
        else:
            raise RuntimeError("unknown MTP replay adapter contract")


@dataclass(frozen=True)
class MTPAcceptanceRequestObservation:
    request_id: str
    phase: str
    input_tokens: int
    output_tokens: int
    computed_tokens: int
    recomputed_tokens: int
    prefix_replay_tokens: int
    resumed_from_preemption: bool
    trace_prefix_hit_tokens: int
    scheduler_prefix_hit_tokens: int
    raw_model_output_tokens: int
    emitted_tokens: int
    mtp_proposed_tokens: int
    mtp_verified_tokens: int
    mtp_accepted_tokens: int
    mtp_rejected_tokens: int
    mtp_proposer_tokens: int
    mtp_next_proposed_tokens: int

    def __post_init__(self) -> None:
        _require_string("request_id", self.request_id)
        if self.phase not in ("prefill", "decode"):
            raise ValueError(
                f"unknown MTP acceptance replay phase: {self.phase!r}")
        for name in _REQUEST_FIELDS - {
                "request_id", "phase", "resumed_from_preemption"}:
            _require_integer(name, getattr(self, name), positive=False)
        if not isinstance(self.resumed_from_preemption, bool):
            raise TypeError(
                "MTP acceptance replay resumed_from_preemption must be "
                "boolean")
        if self.input_tokens + self.output_tokens <= 0:
            raise ValueError(
                "MTP acceptance replay request must schedule tokens")
        if self.mtp_verified_tokens > self.mtp_proposed_tokens:
            raise ValueError(
                "MTP acceptance replay verified tokens cannot exceed "
                "proposed tokens")
        if self.mtp_accepted_tokens > self.mtp_verified_tokens:
            raise ValueError(
                "MTP acceptance replay accepted tokens cannot exceed "
                "verified tokens")
        if self.mtp_rejected_tokens != (
                self.mtp_verified_tokens - self.mtp_accepted_tokens):
            raise ValueError(
                "MTP acceptance replay rejected tokens disagree with "
                "verified minus accepted")
        if self.mtp_verified_tokens:
            if self.raw_model_output_tokens != (
                    self.mtp_accepted_tokens + 1):
                raise ValueError(
                    "MTP acceptance replay raw output must contain the "
                    "accepted prefix plus one terminal token")
        elif self.mtp_accepted_tokens:
            raise ValueError(
                "MTP acceptance replay cannot accept unverified drafts")
        if self.raw_model_output_tokens not in (
                0, self.mtp_accepted_tokens + 1):
            raise ValueError(
                "MTP acceptance replay raw output must be zero or the "
                "accepted prefix plus one terminal token")
        if self.emitted_tokens > self.raw_model_output_tokens:
            raise ValueError(
                "MTP acceptance replay emitted tokens cannot exceed raw "
                "model output")
        if self.scheduler_prefix_hit_tokens > self.trace_prefix_hit_tokens:
            raise ValueError(
                "MTP acceptance replay scheduler prefix hit cannot exceed "
                "the trace hit")
        scheduled_tokens = self.input_tokens + self.output_tokens
        if self.recomputed_tokens > scheduled_tokens:
            raise ValueError(
                "MTP acceptance replay recomputed tokens cannot exceed "
                "scheduled tokens")
        if self.prefix_replay_tokens > scheduled_tokens:
            raise ValueError(
                "MTP acceptance replay prefix replay tokens cannot exceed "
                "scheduled tokens")
        if (self.recomputed_tokens + self.prefix_replay_tokens
                > scheduled_tokens):
            raise ValueError(
                "MTP acceptance replay recompute and prefix replay cannot "
                "overlap beyond scheduled work")
        if self.scheduler_prefix_hit_tokens > self.computed_tokens:
            raise ValueError(
                "MTP acceptance replay scheduler prefix hit cannot exceed "
                "computed tokens")

    @property
    def pre_acceptance_identity(self) -> tuple[object, ...]:
        return (
            self.request_id,
            self.phase,
            self.input_tokens,
            self.output_tokens,
            self.computed_tokens,
            self.recomputed_tokens,
            self.prefix_replay_tokens,
            self.resumed_from_preemption,
            self.trace_prefix_hit_tokens,
            self.scheduler_prefix_hit_tokens,
            self.mtp_proposed_tokens,
            self.mtp_verified_tokens,
            self.mtp_proposer_tokens,
        )


@dataclass(frozen=True)
class MTPAcceptanceStepObservation:
    step_index: int
    step_id: str
    selected_request_ids: tuple[str, ...]
    finished_request_ids: tuple[str, ...]
    preempted_request_ids: tuple[str, ...]
    mtp_drafter_executed: bool
    requests: tuple[MTPAcceptanceRequestObservation, ...]

    def __post_init__(self) -> None:
        _require_integer("step_index", self.step_index, positive=False)
        _require_string("step_id", self.step_id)
        step_id_parts = self.step_id.rsplit(":", 2)
        if (len(step_id_parts) != 3
                or not step_id_parts[0]
                or step_id_parts[1] != "vllm"
                or step_id_parts[2] != str(self.step_index)):
            raise ValueError(
                "MTP acceptance replay step ID must be canonical "
                "<pool>:vllm:<step_index>")
        for name in (
                "selected_request_ids", "finished_request_ids",
                "preempted_request_ids"):
            object.__setattr__(
                self, name, _require_string_tuple(name, getattr(self, name)))
        if not isinstance(self.mtp_drafter_executed, bool):
            raise TypeError(
                "MTP acceptance replay mtp_drafter_executed must be boolean")
        if not isinstance(self.requests, (list, tuple)):
            raise TypeError(
                "MTP acceptance replay requests must be a list or tuple")
        normalized_requests = tuple(self.requests)
        if not normalized_requests or any(
                not isinstance(item, MTPAcceptanceRequestObservation)
                for item in normalized_requests):
            raise TypeError(
                "MTP acceptance replay steps require request observations")
        object.__setattr__(self, "requests", normalized_requests)
        request_ids = tuple(item.request_id for item in normalized_requests)
        if request_ids != self.selected_request_ids:
            raise ValueError(
                "MTP acceptance replay selected request order disagrees with "
                "request observations")
        selected = set(self.selected_request_ids)
        if not set(self.finished_request_ids).issubset(selected):
            raise ValueError(
                "MTP acceptance replay finished requests must be selected")
        folded_resumes = {request.request_id for request in normalized_requests
            if request.resumed_from_preemption and request.computed_tokens == 0
            and request.recomputed_tokens > 0}
        if selected.intersection(self.preempted_request_ids) - folded_resumes:
            raise ValueError(
                "MTP acceptance replay cannot select and preempt the same "
                "request in one step")
        finished = set(self.finished_request_ids)
        for request in normalized_requests:
            if request.request_id in finished:
                if request.mtp_next_proposed_tokens != 0:
                    raise ValueError(
                        "MTP acceptance replay finished request cannot retain "
                        "next proposals")
            elif request.emitted_tokens != request.raw_model_output_tokens:
                raise ValueError(
                    "MTP acceptance replay may truncate emitted tokens only "
                    "for a finished request")

    def request_for(self, request_id: str) -> MTPAcceptanceRequestObservation:
        matches = tuple(
            request for request in self.requests
            if request.request_id == request_id)
        if len(matches) != 1:
            raise ValueError(
                "MTP acceptance replay step requires exactly one request "
                f"observation for {request_id!r}; found {len(matches)}")
        return matches[0]


@dataclass(frozen=True)
class MTPAcceptanceReplay:
    schema_version: int
    source_path: str
    source_sha256: str
    artifact_id: str
    provenance: MTPAcceptanceProvenance
    scheduler_context: MTPAcceptanceSchedulerContext
    steps: tuple[MTPAcceptanceStepObservation, ...]

    def __post_init__(self) -> None:
        _require_integer("schema_version", self.schema_version, positive=True)
        if self.schema_version == 1:
            if (self.scheduler_context.adapter_contract
                    != COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT):
                raise ValueError(
                    "MTP acceptance replay schema v1 requires the colocated "
                    "adapter contract")
        elif self.schema_version == 2:
            if (self.scheduler_context.adapter_contract
                    != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT):
                raise ValueError(
                    "MTP acceptance replay schema v2 requires the PD decode "
                    "adapter contract")
        else:
            raise ValueError(
                "MTP acceptance replay schema_version must be 1 or 2")
        _require_string("source_path", self.source_path)
        _require_sha256("source_sha256", self.source_sha256)
        _require_string("artifact_id", self.artifact_id)
        if not isinstance(self.provenance, MTPAcceptanceProvenance):
            raise TypeError(
                "MTP acceptance replay provenance has the wrong type")
        if not isinstance(
                self.scheduler_context, MTPAcceptanceSchedulerContext):
            raise TypeError(
                "MTP acceptance replay scheduler context has the wrong type")
        if not isinstance(self.steps, (list, tuple)):
            raise TypeError("MTP acceptance replay steps must be a sequence")
        normalized_steps = tuple(self.steps)
        if not normalized_steps or any(
                not isinstance(step, MTPAcceptanceStepObservation)
                for step in normalized_steps):
            raise TypeError(
                "MTP acceptance replay requires step observations")
        object.__setattr__(self, "steps", normalized_steps)
        indices = tuple(step.step_index for step in normalized_steps)
        if indices != tuple(range(len(normalized_steps))):
            raise ValueError(
                "MTP acceptance replay step indices must be contiguous from "
                f"zero; got {indices}")
        depth = self.scheduler_context.num_speculative_tokens
        first_seen_request_ids = set()
        for step in normalized_steps:
            expected_step_id = (
                f"{self.scheduler_context.pool_name}:vllm:{step.step_index}")
            if step.step_id != expected_step_id:
                raise ValueError(
                    "MTP acceptance replay step ID disagrees with its "
                    f"Scheduler pool: expected={expected_step_id!r}, "
                    f"actual={step.step_id!r}")
            proposer_depths = {
                request.mtp_proposer_tokens for request in step.requests}
            if step.mtp_drafter_executed:
                if proposer_depths != {depth}:
                    raise ValueError(
                        "MTP acceptance replay padded drafter step must "
                        f"propose depth {depth} for every request")
            elif proposer_depths != {0}:
                raise ValueError(
                    "MTP acceptance replay non-drafter step cannot propose "
                    "tokens")
            for request in step.requests:
                if request.mtp_proposed_tokens > depth:
                    raise ValueError(
                        "MTP acceptance replay installed proposals exceed "
                        "the configured MTP depth")
                if request.mtp_next_proposed_tokens > depth:
                    raise ValueError(
                        "MTP acceptance replay next proposals exceed the "
                        "configured MTP depth")
            if self.schema_version == 2:
                if (len(step.requests) != 1
                        or step.preempted_request_ids):
                    raise ValueError(
                        "PD decode MTP replay requires one selected request "
                        "and no preemption per step")
                request = step.requests[0]
                if (request.phase != "decode"
                        or request.input_tokens != 0
                        or request.trace_prefix_hit_tokens != 0
                        or request.scheduler_prefix_hit_tokens != 0
                        or request.resumed_from_preemption):
                    raise ValueError(
                        "PD decode MTP replay steps require decode-only work "
                        "without prefix reinjection or preemption")
                if request.request_id not in first_seen_request_ids:
                    if (request.computed_tokens <= 0
                            or request.output_tokens != 1
                            or request.raw_model_output_tokens != 1
                            or request.emitted_tokens != 1
                            or any((
                                request.mtp_proposed_tokens,
                                request.mtp_verified_tokens,
                                request.mtp_accepted_tokens,
                                request.mtp_rejected_tokens,
                            ))):
                        raise ValueError(
                            "PD decode MTP replay first step requires full "
                            "prompt state, one terminal seed, and no installed "
                            "draft tokens")
                    first_seen_request_ids.add(request.request_id)

    def step_for(self, step_index: int) -> MTPAcceptanceStepObservation:
        _require_integer("step_index", step_index, positive=False)
        if step_index >= len(self.steps):
            raise RuntimeError(
                "MTP acceptance replay is missing Scheduler step "
                f"{step_index}; artifact has {len(self.steps)} steps")
        return self.steps[step_index]

    def assert_exhausted(self, consumed_steps: int) -> None:
        _require_integer("consumed_steps", consumed_steps, positive=False)
        if consumed_steps != len(self.steps):
            raise RuntimeError(
                "MTP acceptance replay has unconsumed Scheduler steps: "
                f"consumed={consumed_steps}, artifact={len(self.steps)}")


def mtp_acceptance_step_from_backend_step(
    step_index: int,
    step: BackendStep,
) -> MTPAcceptanceStepObservation:
    """Normalize one planned backend step without its source classification."""

    if not isinstance(step, BackendStep):
        raise TypeError(
            "MTP acceptance replay normalization requires BackendStep")
    if any(request.mtp_padding_tokens for request in step.schedule.requests):
        raise NotImplementedError("MTP replay schema does not represent native padding rows")
    proposer_depths = {
        request.mtp_proposer_tokens for request in step.schedule.requests}
    if proposer_depths == {0}:
        drafter_executed = False
    elif len(proposer_depths) == 1:
        drafter_executed = True
    else:
        raise ValueError(
            "MTP acceptance replay requires one step-global proposer depth")
    requests = tuple(
        MTPAcceptanceRequestObservation(
            request_id=request.request_id,
            phase=request.phase,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            computed_tokens=request.computed_tokens,
            recomputed_tokens=request.recomputed_tokens,
            prefix_replay_tokens=request.prefix_replay_tokens,
            resumed_from_preemption=request.resumed_from_preemption,
            trace_prefix_hit_tokens=request.trace_prefix_hit_tokens,
            scheduler_prefix_hit_tokens=(
                request.scheduler_prefix_hit_tokens),
            raw_model_output_tokens=request.raw_model_output_tokens,
            emitted_tokens=step.generated_tokens[request.request_id],
            mtp_proposed_tokens=request.mtp_proposed_tokens,
            mtp_verified_tokens=request.mtp_verified_tokens,
            mtp_accepted_tokens=request.mtp_accepted_tokens,
            mtp_rejected_tokens=request.mtp_rejected_tokens,
            mtp_proposer_tokens=request.mtp_proposer_tokens,
            mtp_next_proposed_tokens=request.mtp_next_proposed_tokens,
        )
        for request in step.schedule.requests
    )
    return MTPAcceptanceStepObservation(
        step_index=step_index,
        step_id=step.schedule.step_id,
        selected_request_ids=tuple(
            request.request_id for request in step.schedule.requests),
        finished_request_ids=step.schedule.finished_request_ids,
        preempted_request_ids=step.schedule.preempted_request_ids,
        mtp_drafter_executed=drafter_executed,
        requests=requests,
    )


def _require_exact_fields(
    value: object,
    expected: set[str],
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"MTP acceptance replay {label} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"MTP acceptance replay {label} fields must be exactly "
            f"{sorted(expected)}, got {sorted(value)}")
    return value


def load_mtp_acceptance_replay(path: str | Path) -> MTPAcceptanceReplay:
    """Read and fully validate one immutable MTP acceptance artifact."""

    source_path = Path(path)
    payload = source_path.read_bytes()
    def reject_constant(value: str) -> object:
        raise ValueError(
            f"MTP acceptance replay JSON constant is invalid: {value}")

    raw = json.loads(
        payload,
        object_pairs_hook=_strict_json_object,
        parse_constant=reject_constant,
    )
    root = _require_exact_fields(raw, _ROOT_FIELDS, "root")
    schema_version = _require_integer(
        "schema_version", root["schema_version"], positive=True)
    if schema_version not in (1, 2):
        raise ValueError(
            "MTP acceptance replay schema_version must be 1 or 2")
    provenance_raw = _require_exact_fields(
        root["provenance"], _PROVENANCE_FIELDS, "provenance")
    if schema_version == 1:
        supplied_context = _require_exact_fields(
            root["scheduler_context"],
            _CONTEXT_V1_FIELDS,
            "scheduler_context",
        )
        context_raw = {
            **supplied_context,
            "ingress_request_stream_sha256": None,
            "decode_admission_contract_sha256": None,
        }
    elif schema_version == 2:
        supplied_context = _require_exact_fields(
            root["scheduler_context"],
            _CONTEXT_V2_FIELDS,
            "scheduler_context",
        )
        context_raw = {
            **supplied_context,
            "request_stream_sha256": None,
        }
    else:
        raise AssertionError("validated an unknown MTP replay schema")
    steps_raw = root["steps"]
    if not isinstance(steps_raw, list) or not steps_raw:
        raise TypeError(
            "MTP acceptance replay steps must be a non-empty list")
    steps = []
    for step_index, raw_step in enumerate(steps_raw):
        step = _require_exact_fields(
            raw_step, _STEP_FIELDS, f"steps[{step_index}]")
        requests_raw = step["requests"]
        if not isinstance(requests_raw, list) or not requests_raw:
            raise TypeError(
                f"MTP acceptance replay steps[{step_index}].requests must "
                "be a non-empty list")
        requests = []
        for request_index, raw_request in enumerate(requests_raw):
            request = _require_exact_fields(
                raw_request,
                _REQUEST_FIELDS,
                f"steps[{step_index}].requests[{request_index}]",
            )
            requests.append(MTPAcceptanceRequestObservation(**request))
        steps.append(MTPAcceptanceStepObservation(
            step_index=step["step_index"],
            step_id=step["step_id"],
            selected_request_ids=step["selected_request_ids"],
            finished_request_ids=step["finished_request_ids"],
            preempted_request_ids=step["preempted_request_ids"],
            mtp_drafter_executed=step["mtp_drafter_executed"],
            requests=tuple(requests),
        ))
    return MTPAcceptanceReplay(
        schema_version=schema_version,
        source_path=str(source_path.resolve()),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        artifact_id=root["artifact_id"],
        provenance=MTPAcceptanceProvenance(**provenance_raw),
        scheduler_context=MTPAcceptanceSchedulerContext(**context_raw),
        steps=tuple(steps),
    )


def mtp_acceptance_replay_manifest(
    replay: MTPAcceptanceReplay,
) -> dict[str, object]:
    """Serialize replay identity without duplicating its observation payload."""

    if not isinstance(replay, MTPAcceptanceReplay):
        raise TypeError("MTP acceptance replay manifest requires a replay")
    common = {
        "artifact_id": replay.artifact_id,
        "config_path": replay.source_path,
        "config_sha256": replay.source_sha256,
        "producer": replay.provenance.producer,
        "vllm_version": replay.provenance.vllm_version,
        "vllm_source_commit": replay.provenance.vllm_source_commit,
        "vllm_package_manifest_sha256": (
            replay.provenance.vllm_package_manifest_sha256),
        "model_config_sha256": replay.provenance.model_config_sha256,
        "pool_name": replay.scheduler_context.pool_name,
        "backend_role": replay.scheduler_context.backend_role,
        "num_speculative_tokens": (
            replay.scheduler_context.num_speculative_tokens),
        "step_count": len(replay.steps),
    }
    if replay.schema_version == 1:
        return {
            "schema_version": 1,
            **common,
            "request_stream_sha256": (
                replay.scheduler_context.request_stream_sha256),
        }
    elif replay.schema_version == 2:
        return {
            "schema_version": 2,
            **common,
            "artifact_schema_version": replay.schema_version,
            "adapter_contract": (
                replay.scheduler_context.adapter_contract),
            "ingress_request_stream_sha256": (
                replay.scheduler_context.ingress_request_stream_sha256),
            "decode_admission_contract_sha256": (
                replay.scheduler_context.decode_admission_contract_sha256),
        }
    else:
        raise AssertionError("unknown MTP replay schema version")


def normalize_mtp_acceptance_replay_manifest(
    value: object,
) -> dict[str, object]:
    """Validate a serialized replay identity without its observation payload."""

    common_fields = {
        "schema_version",
        "artifact_id",
        "config_path",
        "config_sha256",
        "producer",
        "vllm_version",
        "vllm_source_commit",
        "vllm_package_manifest_sha256",
        "model_config_sha256",
        "pool_name",
        "backend_role",
        "num_speculative_tokens",
        "step_count",
    }
    if not isinstance(value, dict):
        raise TypeError(
            "MTP acceptance replay manifest must be a mapping")
    if "schema_version" not in value:
        raise ValueError(
            "MTP acceptance replay manifest requires schema_version")
    schema_version = _require_integer(
        "manifest schema_version", value["schema_version"], positive=True)
    if schema_version == 1:
        fields = common_fields | {"request_stream_sha256"}
    elif schema_version == 2:
        fields = common_fields | {
            "artifact_schema_version",
            "adapter_contract",
            "ingress_request_stream_sha256",
            "decode_admission_contract_sha256",
        }
    else:
        raise ValueError(
            "MTP acceptance replay manifest schema_version must be 1 or 2")
    raw = _require_exact_fields(value, fields, "manifest")
    for name in (
            "artifact_id", "config_path", "producer", "vllm_version",
            "pool_name", "backend_role"):
        _require_string(f"manifest {name}", raw[name])
    if raw["producer"] != "vllm_model_runner_observation":
        raise ValueError(
            "MTP acceptance replay manifest producer must be "
            "'vllm_model_runner_observation'")
    commit = _require_string(
        "manifest vllm_source_commit", raw["vllm_source_commit"])
    if (len(commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in commit)):
        raise ValueError(
            "MTP acceptance replay manifest vllm_source_commit must be a "
            "lowercase 40-character Git commit")
    for name in (
            "config_sha256", "vllm_package_manifest_sha256",
            "model_config_sha256"):
        _require_sha256(f"manifest {name}", raw[name])
    if schema_version == 1:
        _require_sha256(
            "manifest request_stream_sha256",
            raw["request_stream_sha256"],
        )
    elif schema_version == 2:
        artifact_schema_version = _require_integer(
            "manifest artifact_schema_version",
            raw["artifact_schema_version"],
            positive=True,
        )
        if artifact_schema_version != 2:
            raise ValueError(
                "PD decode MTP replay manifest requires artifact schema v2")
        if raw["adapter_contract"] != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT:
            raise ValueError(
                "PD decode MTP replay manifest has the wrong adapter contract")
        if raw["backend_role"] != "decode" or raw["pool_name"] != "decode-0":
            raise ValueError(
                "PD decode MTP replay manifest requires decode-0/decode")
        _require_sha256(
            "manifest ingress_request_stream_sha256",
            raw["ingress_request_stream_sha256"],
        )
        _require_sha256(
            "manifest decode_admission_contract_sha256",
            raw["decode_admission_contract_sha256"],
        )
    else:
        raise AssertionError("validated an unknown MTP replay manifest schema")
    _require_integer(
        "manifest num_speculative_tokens",
        raw["num_speculative_tokens"],
        positive=True,
    )
    _require_integer("manifest step_count", raw["step_count"], positive=True)
    return dict(raw)


def mtp_acceptance_steps_manifest(
    steps: Iterable[MTPAcceptanceStepObservation],
) -> list[dict[str, Any]]:
    """Serialize matched normalized outcomes for a report boundary."""

    normalized = tuple(steps)
    if any(not isinstance(step, MTPAcceptanceStepObservation)
           for step in normalized):
        raise TypeError(
            "MTP acceptance step manifest requires step observations")
    return [
        {
            "step_index": step.step_index,
            "step_id": step.step_id,
            "selected_request_ids": list(step.selected_request_ids),
            "finished_request_ids": list(step.finished_request_ids),
            "preempted_request_ids": list(step.preempted_request_ids),
            "mtp_drafter_executed": step.mtp_drafter_executed,
            "requests": [
                {
                    name: getattr(request, name)
                    for name in sorted(_REQUEST_FIELDS)
                }
                for request in step.requests
            ],
        }
        for step in normalized
    ]
