"""Normalized scheduler output for one virtual execution step."""

from dataclasses import InitVar, asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Mapping, Tuple

from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.host_timing import WorkerRequestState


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    phase: str
    input_tokens: int
    output_tokens: int
    computed_tokens: int = 0
    recomputed_tokens: int = 0
    prefix_replay_tokens: int = 0
    resumed_from_preemption: bool = False
    trace_prefix_hit_tokens: int = 0
    scheduler_prefix_hit_tokens: int = 0
    prefix_scheduler_delta_tokens: int = field(init=False)
    raw_model_output_tokens: int = 0
    mtp_proposed_tokens: int = 0
    mtp_verified_tokens: int = 0
    mtp_accepted_tokens: int = 0
    mtp_rejected_tokens: int = field(init=False)
    mtp_proposer_tokens: int = 0
    mtp_next_proposed_tokens: int = 0
    mtp_proposer_mode: str = "disabled"
    mtp_acceptance_source_kind: str = "disabled"
    mtp_acceptance_source_id: str = "disabled"
    content_prefix_hit_tokens: int = 0
    data_parallel_rank: int = 0
    mtp_padding_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("scheduled request_id must be a non-empty string")
        if not isinstance(self.phase, str):
            raise TypeError("scheduled request phase must be a string")
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"unknown request phase: {self.phase!r}")
        for name in (
                "data_parallel_rank", "input_tokens", "output_tokens", "computed_tokens",
                "recomputed_tokens", "prefix_replay_tokens",
                "trace_prefix_hit_tokens", "scheduler_prefix_hit_tokens",
                "content_prefix_hit_tokens",
                "raw_model_output_tokens",
                "mtp_proposed_tokens", "mtp_verified_tokens", "mtp_padding_tokens",
                "mtp_accepted_tokens", "mtp_proposer_tokens",
                "mtp_next_proposed_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"scheduled {name} must be an integer")
        if min(
                self.data_parallel_rank,
                self.input_tokens,
                self.output_tokens,
                self.computed_tokens,
                self.recomputed_tokens,
                self.prefix_replay_tokens,
                self.content_prefix_hit_tokens,
                self.trace_prefix_hit_tokens,
                self.scheduler_prefix_hit_tokens,
                self.raw_model_output_tokens,
                self.mtp_proposed_tokens,
                self.mtp_verified_tokens,
                self.mtp_padding_tokens,
                self.mtp_accepted_tokens,
                self.mtp_proposer_tokens,
                self.mtp_next_proposed_tokens,
        ) < 0:
            raise ValueError("scheduled token counts cannot be negative")
        if self.input_tokens + self.output_tokens == 0:
            raise ValueError("a scheduled request must execute at least one token")
        if self.recomputed_tokens > self.scheduled_tokens:
            raise ValueError(
                "recomputed_tokens cannot exceed scheduled tokens")
        if self.prefix_replay_tokens > self.scheduled_tokens:
            raise ValueError(
                "prefix_replay_tokens cannot exceed scheduled tokens")
        if (self.recomputed_tokens + self.prefix_replay_tokens
                > self.scheduled_tokens):
            raise ValueError(
                "recomputed and prefix replay tokens cannot overlap beyond "
                "the scheduled work")
        if not isinstance(self.resumed_from_preemption, bool):
            raise TypeError(
                "resumed_from_preemption must be boolean")
        if self.content_prefix_hit_tokens > self.computed_tokens:
            raise ValueError("content prefix hit cannot exceed computed tokens")
        if self.content_prefix_hit_tokens and (self.trace_prefix_hit_tokens or self.scheduler_prefix_hit_tokens):
            raise ValueError("content and trace prefix hits cannot be mixed")
        if self.scheduler_prefix_hit_tokens > self.trace_prefix_hit_tokens:
            raise ValueError(
                "scheduler prefix hit cannot exceed the trace oracle hit")
        if self.scheduler_prefix_hit_tokens > self.computed_tokens:
            raise ValueError(
                "scheduler prefix hit cannot exceed computed tokens")
        object.__setattr__(
            self,
            "prefix_scheduler_delta_tokens",
            self.trace_prefix_hit_tokens - self.scheduler_prefix_hit_tokens,
        )
        # vLLM0.30 may schedule a full speculative query before any real
        # proposal exists. Those -1 rows are physically verified and rejected.
        if self.mtp_padding_tokens and (
                self.mtp_padding_tokens != self.mtp_verified_tokens
                or self.mtp_proposed_tokens or self.mtp_accepted_tokens
                or self.phase != "decode"
                or self.scheduled_tokens != self.mtp_padding_tokens + 1):
            raise ValueError("MTP padding requires a decode anchor and only rejected placeholder rows")
        if self.mtp_verified_tokens > self.mtp_proposed_tokens + self.mtp_padding_tokens:
            raise ValueError(
                "MTP verified tokens cannot exceed proposed tokens")
        if self.mtp_accepted_tokens > self.mtp_verified_tokens:
            raise ValueError(
                "MTP accepted tokens cannot exceed verified tokens")
        if self.raw_model_output_tokens:
            if self.raw_model_output_tokens != self.mtp_accepted_tokens + 1:
                raise ValueError(
                    "raw model output must contain the accepted MTP prefix "
                    "followed by exactly one terminal token")
        elif self.mtp_verified_tokens:
            raise ValueError(
                "an MTP verification step must produce a raw model output")
        object.__setattr__(
            self,
            "mtp_rejected_tokens",
            self.mtp_verified_tokens - self.mtp_accepted_tokens,
        )
        for name in (
                "mtp_proposer_mode", "mtp_acceptance_source_kind",
                "mtp_acceptance_source_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"scheduled {name} must be a non-empty string")
        if self.mtp_acceptance_source_kind == "disabled":
            if self.mtp_proposer_mode != "disabled":
                raise ValueError(
                    "disabled MTP source requires proposer mode 'disabled'")
            if self.mtp_acceptance_source_id != "disabled":
                raise ValueError(
                    "disabled MTP source kind requires source ID 'disabled'")
            if any((
                    self.mtp_proposed_tokens,
                    self.mtp_verified_tokens,
                    self.mtp_padding_tokens,
                    self.mtp_accepted_tokens,
                    self.mtp_proposer_tokens,
                    self.mtp_next_proposed_tokens,
            )):
                raise ValueError(
                    "disabled MTP source cannot carry speculative tokens")
        elif self.mtp_acceptance_source_kind in (
                "explicit_assumption", "replay_observation"):
            if self.mtp_proposer_mode != "padded":
                raise ValueError(
                    "enabled MTP requires pinned padded proposer mode")
            if self.mtp_acceptance_source_id == "disabled":
                raise ValueError(
                    "enabled MTP requires a non-disabled source ID")
            if (self.mtp_proposed_tokens == 0
                    and self.mtp_proposer_tokens == 0
                    and self.mtp_next_proposed_tokens == 0
                    and self.mtp_verified_tokens == 0
                    and self.mtp_padding_tokens == 0):
                raise ValueError(
                    "enabled MTP requires current or next proposed tokens")
        else:
            raise ValueError(
                "unknown MTP acceptance source kind: "
                f"{self.mtp_acceptance_source_kind!r}")

    @property
    def scheduled_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ScheduleStep:
    step_id: str
    pool_name: str
    requests: Tuple[ScheduledRequest, ...]
    finished_request_ids: Tuple[str, ...] = ()
    preempted_request_ids: Tuple[str, ...] = ()
    attention_kv_layout: str | None = None
    transfer_only: InitVar[bool] = False

    def __post_init__(self, transfer_only: bool) -> None:
        for name in ("step_id", "pool_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"schedule {name} must be a non-empty string")
        if not isinstance(self.requests, (list, tuple)):
            raise TypeError("ScheduleStep requests must be a list or tuple")
        object.__setattr__(self, "requests", tuple(self.requests))
        if type(transfer_only) is not bool:
            raise TypeError("ScheduleStep transfer_only must be boolean")
        object.__setattr__(self, "_transfer_only", transfer_only)
        if not self.requests and not transfer_only:
            raise ValueError("ScheduleStep requires at least one request")
        if self.requests and transfer_only:
            raise ValueError("transfer-only ScheduleStep cannot execute requests")
        if any(not isinstance(request, ScheduledRequest)
               for request in self.requests):
            raise TypeError("ScheduleStep requests must be ScheduledRequest objects")
        request_ids = tuple(request.request_id for request in self.requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("ScheduleStep request IDs must be unique")
        if not isinstance(self.finished_request_ids, (list, tuple)):
            raise TypeError(
                "finished_request_ids must be a list or tuple")
        object.__setattr__(
            self, "finished_request_ids", tuple(self.finished_request_ids))
        if any(not isinstance(request_id, str) or not request_id
               for request_id in self.finished_request_ids):
            raise ValueError(
                "finished_request_ids must contain non-empty strings")
        if len(self.finished_request_ids) != len(set(self.finished_request_ids)):
            raise ValueError("finished_request_ids must be unique")
        scheduled_ids = set(request_ids)
        if not set(self.finished_request_ids).issubset(scheduled_ids):
            raise ValueError(
                "finished requests must belong to the current schedule step")
        if not isinstance(self.preempted_request_ids, (list, tuple)):
            raise TypeError(
                "preempted_request_ids must be a list or tuple")
        object.__setattr__(
            self,
            "preempted_request_ids",
            tuple(self.preempted_request_ids),
        )
        if any(not isinstance(request_id, str) or not request_id
               for request_id in self.preempted_request_ids):
            raise ValueError(
                "preempted_request_ids must contain non-empty strings")
        if len(self.preempted_request_ids) != len(
                set(self.preempted_request_ids)):
            raise ValueError("preempted_request_ids must be unique")
        preempted_ids = set(self.preempted_request_ids)
        # Native empty preemption decisions can be folded into the next
        # actual executor step. Only an explicit zero-base recompute/resume
        # may then be both preempted and selected (and possibly finished).
        folded_resumes = {request.request_id for request in self.requests
            if request.resumed_from_preemption and request.computed_tokens == 0
            and request.recomputed_tokens > 0}
        if (preempted_ids & set(self.finished_request_ids)) - folded_resumes:
            raise ValueError(
                "preempted and finished request IDs must be disjoint")
        if (preempted_ids & scheduled_ids) - folded_resumes:
            raise ValueError(
                "preempted requests cannot execute in the same schedule step")
        if self.attention_kv_layout is not None:
            if not isinstance(self.attention_kv_layout, str):
                raise TypeError(
                    "ScheduleStep attention_kv_layout must be a string or null")
            if self.attention_kv_layout not in (
                    "contiguous", "paged", "paged_varlen",
                    "chunked_prefill", "mla"):
                raise ValueError(
                    "unsupported ScheduleStep attention_kv_layout: "
                    f"{self.attention_kv_layout!r}")

    @property
    def is_transfer_only(self) -> bool:
        return self._transfer_only

    @property
    def token_count(self) -> int:
        return sum(request.scheduled_tokens for request in self.requests)


def schedule_step_sha256(step: ScheduleStep) -> str:
    """Hash the complete normalized Scheduler decision for replay binding."""

    if not isinstance(step, ScheduleStep):
        raise TypeError("schedule step hashing requires a ScheduleStep")
    normalized = asdict(step)
    # Keep existing length-only/trace replay identities stable. A nonzero
    # native content hit is additional decision evidence and enters the hash.
    for request in normalized["requests"]:
        if request["content_prefix_hit_tokens"] == 0:
            del request["content_prefix_hit_tokens"]
        if request["data_parallel_rank"] == 0:
            del request["data_parallel_rank"]
        if request["mtp_padding_tokens"] == 0:
            del request["mtp_padding_tokens"]
    payload = {
        "schema_version": 1,
        "schedule_step": normalized,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class BackendStep:
    schedule: ScheduleStep
    kv_cache_events: Tuple[KVCacheEvent, ...] = ()
    generated_tokens: Mapping[str, int] = field(
        default_factory=dict)
    scheduler_plan_cpu_s: float = field(default=0.0, compare=False)
    engine_submit_host_s: float = field(default=0.0, compare=False)
    engine_wake_host_s: float = field(default=0.0, compare=False)
    embedding_token_ids: tuple[int, ...] | None = None
    embedding_token_source: str | None = None
    # Typed model-source metadata, passed unchanged to the virtual forward.
    model_source_inputs: object | None = None
    worker_request_state: WorkerRequestState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ScheduleStep):
            raise TypeError("BackendStep requires a ScheduleStep")
        if self.worker_request_state is not None:
            if not isinstance(self.worker_request_state, WorkerRequestState):
                raise TypeError("worker request state must be typed source counts")
            if (self.worker_request_state.admitted_requests + self.worker_request_state.cached_requests
                    != len(self.schedule.requests)):
                raise ValueError("worker request state must cover the exact scheduled requests")
        if self.embedding_token_ids is None:
            if self.embedding_token_source is not None:
                raise ValueError("embedding token source requires token IDs")
        else:
            if (not isinstance(self.embedding_token_source, str) or self.embedding_token_source not in (
                    "vllm_length_only_simulation", "vllm_explicit_request_token_ids", "sglang_explicit_request_token_ids", "sglang_declared_synthetic_fixture_token_ids")):
                raise ValueError("unsupported BackendStep embedding token source")
            if (not isinstance(self.embedding_token_ids, tuple)
                    or len(self.embedding_token_ids) != self.schedule.token_count
                    or any(type(token) is not int or token < 0 for token in self.embedding_token_ids)):
                raise ValueError("embedding token IDs must be nonnegative integers covering the scheduled row order")
        if self.model_source_inputs is not None:
            from src.serving.workload.graph_padding import VLLMDenseGraphInputs
            from src.serving.workload.traced_sglang import SGLangEagleChainState, SGLangPrefillInputs, SGLangGraphInputs
            from src.serving.workload.runtime_inputs import K3SourceInputs, K3DataParallelSourceInputs, DSparkSourceInputs, DSparkFirstProposalInputs, V32PaddedMTPInputs
            state = self.model_source_inputs
            if not isinstance(state, (SGLangEagleChainState, SGLangPrefillInputs, SGLangGraphInputs,
                    K3SourceInputs, K3DataParallelSourceInputs, DSparkSourceInputs, DSparkFirstProposalInputs, V32PaddedMTPInputs, VLLMDenseGraphInputs)):
                raise TypeError("BackendStep model-source inputs require a registered typed source state")
            if (self.embedding_token_ids is None
                    or (not isinstance(state, (K3DataParallelSourceInputs, DSparkSourceInputs))
                        and state.request_ids != tuple(request.request_id for request in self.schedule.requests))):
                raise ValueError("model-source input identity must match the explicit scheduled rows")
            if isinstance(state,DSparkFirstProposalInputs):
                if (len(self.schedule.requests)!=1 or len(state.target_positions)!=self.schedule.token_count
                        or self.embedding_token_source!="vllm_explicit_request_token_ids"
                        or any(request.mtp_proposer_tokens!=5 or request.mtp_next_proposed_tokens!=(0 if state.finish_request else 5)
                            or request.mtp_proposer_mode!="padded" or request.mtp_acceptance_source_id!=state.source_id
                            or request.mtp_acceptance_source_kind!="explicit_assumption" for request in self.schedule.requests)):
                    raise ValueError("DSpark source requires matching explicit target rows and proposal identity")
                request = self.schedule.requests[0]
                if state.finish_request != (request.request_id in self.schedule.finished_request_ids):
                    raise ValueError("DSpark terminal state must match actual scheduled completion")
                expected_delivery=(request.raw_model_output_tokens if state.delivered_token_count is None else state.delivered_token_count)
                if request.request_id in self.generated_tokens and self.generated_tokens[request.request_id]!=expected_delivery:
                    raise ValueError("DSpark delivery count disagrees with actual Scheduler truncation")
                if state.is_verification:
                    if (request.phase!="decode" or request.input_tokens!=0 or request.output_tokens!=6
                            or request.computed_tokens!=state.previous_retained_kv_tokens
                            or request.mtp_proposed_tokens!=5 or request.mtp_verified_tokens!=5
                            or request.mtp_accepted_tokens!=state.accepted_drafts
                            or request.raw_model_output_tokens!=state.accepted_drafts+1
                            or self.embedding_token_ids!=(state.previous_anchor_token_ids[0],
                                *(ids[0] for ids in state.previous_draft_token_ids))):
                        raise ValueError("DSpark verification schedule must consume its previous anchor and five proposals")
                elif (request.phase!="prefill" or request.computed_tokens!=0 or request.raw_model_output_tokens!=1
                        or request.mtp_verified_tokens!=0):
                    raise ValueError("DSpark first proposal state requires matching explicit fresh target rows and proposal identity")
            if isinstance(state, DSparkSourceInputs):
                if any(request.data_parallel_rank >= len(state.replicas) for request in self.schedule.requests):
                    raise ValueError("DSpark native frame lacks a request's physical DP owner")
                for owner, local in enumerate(state.replicas):
                    requests = tuple(request for request in self.schedule.requests if request.data_parallel_rank == owner)
                    if local is None and requests or local is not None and (
                            local.request_ids != tuple(request.request_id for request in requests)
                            or any(len(ids) != request.mtp_accepted_tokens + 1
                                or request.mtp_proposer_mode != "padded" for request, ids in zip(requests,local.sampled_token_ids,strict=True))):
                        raise ValueError("DSpark native frame must preserve each replica's request order and raw feedback")
            elif isinstance(state, VLLMDenseGraphInputs):
                if (self.embedding_token_source not in ("vllm_length_only_simulation", "vllm_explicit_request_token_ids")
                        or any(request.mtp_proposer_mode != "disabled" or request.mtp_proposer_tokens
                            or request.mtp_verified_tokens for request in self.schedule.requests)):
                    raise ValueError("dense graph metadata requires explicit ordinary vLLM rows without MTP")
            elif isinstance(state, K3DataParallelSourceInputs):
                states = state.source_inputs_by_replica
                if any(request.data_parallel_rank >= len(states) for request in self.schedule.requests):
                    raise ValueError("K3 DP hybrid frame lacks an original request's replica owner")
                for owner, local in enumerate(states):
                    requests = tuple(request for request in self.schedule.requests if request.data_parallel_rank == owner)
                    if (local is None) != (not requests):
                        raise ValueError("K3 DP hybrid frame must preserve active and idle original request owners")
                    if local is not None and (local.request_ids != tuple(request.request_id for request in requests)
                            or len(local.slot_mapping) != sum(request.scheduled_tokens for request in requests)
                            or any(request.mtp_proposer_mode != "disabled" for request in requests)
                            or (any(request.phase == "prefill" for request in requests)
                                != (local.prefill_output_source == "supports_out"))):
                        raise ValueError("K3 DP hybrid frame must match each replica's original request rows and prefill source")
            elif isinstance(state, K3SourceInputs):
                if (len(state.slot_mapping) != self.schedule.token_count
                        or any(request.mtp_proposer_mode != "disabled" for request in self.schedule.requests)
                        or (any(request.phase == "prefill" for request in self.schedule.requests)
                            != (state.prefill_output_source == "supports_out"))):
                    raise ValueError("K3 source metadata must cover ordinary scheduled rows and the declared prefill output branch")
            elif isinstance(state, SGLangGraphInputs):
                if (self.embedding_token_source != "sglang_explicit_request_token_ids"
                        or any(request.phase != "decode" or request.scheduled_tokens != 1
                            or request.mtp_proposer_tokens or request.mtp_verified_tokens
                            for request in self.schedule.requests)):
                    raise ValueError("SGLang graph metadata requires ordinary explicit single-token decode rows")
            elif isinstance(state, SGLangPrefillInputs):
                if (self.embedding_token_source != "sglang_explicit_request_token_ids"
                        or any(request.phase != "prefill" or request.mtp_proposer_tokens
                            or request.mtp_verified_tokens for request in self.schedule.requests)):
                    raise ValueError("SGLang prefill metadata requires ordinary explicit prefill rows")
                contexts = tuple(request.computed_tokens for request in self.schedule.requests)
                total_kv = sum(contexts) + self.schedule.token_count
                if state.prefix_chunk_kv_indices:
                    width = state.max_kv_chunk_capacity // len(contexts)
                    if total_kv <= state.max_kv_chunk_capacity or sum(contexts) < 8192 or width <= 0:
                        raise ValueError("SGLang chunked indices require an over-capacity long-prefix schedule")
                    lengths = tuple(sum(max(0, min(width, context - start)) for context in contexts)
                        for start in range(0, max(contexts), width))
                    if (len(state.prefix_chunk_kv_indices) != len(lengths)
                            or any(len(indices) != length for indices, length in
                                zip(state.prefix_chunk_kv_indices, lengths, strict=True))):
                        raise ValueError("SGLang prefix chunk metadata must cover every source-ordered scheduled prefix chunk")
                elif len(state.one_shot_kv_indices) != total_kv:
                    raise ValueError("SGLang prefill metadata must cover the complete scheduled KV rows")
            elif isinstance(state, SGLangEagleChainState):
                if self.embedding_token_source != "sglang_explicit_request_token_ids":
                    raise ValueError("SGLang source state requires its explicit token transcript")
                phase = "prefill" if all(request.phase == "prefill" for request in self.schedule.requests) else "verify"
                if state.phase != phase:
                    raise ValueError("model-source input phase disagrees with the schedule")
                if phase == "verify" and state.accept_lens != tuple(request.mtp_accepted_tokens + 1 for request in self.schedule.requests):
                    raise ValueError("declared source acceptance must match the Scheduler acceptance policy")
            else:
                depths = {request.mtp_proposer_tokens for request in self.schedule.requests}
                verify_only = (depths == {0} and any(request.mtp_verified_tokens
                    for request in self.schedule.requests)
                    and all(request.mtp_next_proposed_tokens == 0 for request in self.schedule.requests))
                if ((depths != {len(state.draft_input_token_ids)} and not verify_only)
                        or len(state.draft_input_token_ids[0]) != self.schedule.token_count
                        or any(len(ids) != len(state.request_ids) for ids in state.draft_input_token_ids[1:])
                        or any(request.mtp_proposer_mode != "padded" for request in self.schedule.requests)):
                    raise ValueError("V32 MTP source must cover the scheduled padded draft rows and depth")
        if not isinstance(self.kv_cache_events, (list, tuple)):
            raise TypeError("BackendStep kv_cache_events must be a list or tuple")
        object.__setattr__(self, "kv_cache_events", tuple(self.kv_cache_events))
        if any(not isinstance(event, KVCacheEvent)
               for event in self.kv_cache_events):
            raise TypeError("BackendStep kv_cache_events must be KVCacheEvent objects")
        if self.schedule.is_transfer_only:
            if (not any(event.action is KVCacheAction.STORE
                        for event in self.kv_cache_events)
                    or any(event.action not in (
                        KVCacheAction.STORE, KVCacheAction.RELEASE)
                        for event in self.kv_cache_events)
                    or self.embedding_token_ids is not None
                    or self.model_source_inputs is not None
                    or self.worker_request_state is not None
                    or self.schedule.finished_request_ids
                    or self.schedule.preempted_request_ids):
                raise ValueError("transfer-only step requires stores and optional releases")
            if self.generated_tokens:
                raise ValueError(
                    "transfer-only step cannot emit generated tokens")
        folded_resumes = set(self.schedule.preempted_request_ids).intersection(
            request.request_id for request in self.schedule.requests)
        for request_id in folded_resumes:
            allocations = [index for index, event in enumerate(self.kv_cache_events)
                if event.request_id == request_id and event.action is KVCacheAction.ALLOCATE
                and event.tier == 'hbm' and event.bytes > 0]
            if not allocations or not any(index < allocations[0]
                    and event.request_id == request_id and event.action is KVCacheAction.RELEASE
                    and event.tier == 'hbm' and event.bytes > 0
                    for index, event in enumerate(self.kv_cache_events)):
                raise ValueError('folded native preemption requires HBM RELEASE before ALLOCATE')
        if not isinstance(self.generated_tokens, Mapping):
            raise TypeError("generated_tokens must be a mapping")
        requests_by_id = {
            request.request_id: request for request in self.schedule.requests
        }
        scheduled_ids = set(requests_by_id)
        supplied = dict(self.generated_tokens)
        unknown = set(supplied) - scheduled_ids
        missing = scheduled_ids - set(supplied)
        if unknown or missing:
            raise ValueError(
                "generated_tokens must contain exactly the scheduled requests: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}")
        for request_id in scheduled_ids:
            count = supplied[request_id]
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(
                    "generated token counts must be integers")
            if count < 0:
                raise ValueError(
                    "generated token counts cannot be negative")
            scheduled_request = requests_by_id[request_id]
            raw_count = scheduled_request.raw_model_output_tokens
            if count > raw_count:
                raise ValueError(
                    "emitted tokens cannot exceed raw model output tokens: "
                    f"request={request_id!r}, emitted={count}, raw={raw_count}")
            if (request_id not in self.schedule.finished_request_ids
                    and count not in (0, raw_count)):
                raise ValueError(
                    "a nonterminal request must emit its complete raw model "
                    "output or discard it entirely: "
                    f"request={request_id!r}, emitted={count}, raw={raw_count}")
        object.__setattr__(self, "generated_tokens", supplied)
        if (isinstance(self.scheduler_plan_cpu_s, bool)
                or not isinstance(self.scheduler_plan_cpu_s, Real)
                or not math.isfinite(self.scheduler_plan_cpu_s)
                or self.scheduler_plan_cpu_s < 0):
            raise ValueError(
                "Scheduler plan CPU time must be finite and non-negative")
        for name in ("engine_submit_host_s", "engine_wake_host_s"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(
                    f"EngineCore host-round {name} must be finite and "
                    "non-negative")

    @property
    def finished_request_ids(self) -> Tuple[str, ...]:
        return self.schedule.finished_request_ids


def is_non_mtp_async_terminal_placeholder(request: ScheduledRequest) -> bool:
    """Return whether a scheduled row is vLLM's discarded async tail shape."""

    return (
        request.phase == "decode"
        and request.input_tokens == 0
        and request.output_tokens == 1
        and request.recomputed_tokens == 0
        and request.prefix_replay_tokens == 0
        and not request.resumed_from_preemption
        and request.raw_model_output_tokens == 1
        and request.mtp_proposer_mode == "disabled"
        and request.mtp_acceptance_source_kind == "disabled"
        and request.mtp_acceptance_source_id == "disabled"
        and request.mtp_proposed_tokens == 0
        and request.mtp_verified_tokens == 0
        and request.mtp_accepted_tokens == 0
        and request.mtp_proposer_tokens == 0
        and request.mtp_next_proposed_tokens == 0
    )
