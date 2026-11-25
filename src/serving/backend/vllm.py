"""No-GPU adapter for the real vLLM v1 Scheduler control plane.

The native construction path instantiates ``vllm.v1.core.sched.Scheduler``
directly.  It supplies the KV capacity instead of starting a worker profiler,
and feeds fabricated ``ModelRunnerOutput`` objects back after the virtual
execution time has elapsed.  No model runner, weights, CUDA context, KV tensor,
or sampler is created.

All vLLM imports are lazy so the rest of Midlayer remains usable in an
environment that does not install vLLM.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import islice, repeat
import math
import os
from pathlib import Path
import time
from typing import Any

from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.backend.mtp_acceptance_replay import (
    COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT,
    MTPAcceptanceReplay,
    MTPAcceptanceSchedulerContext,
    PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
    closed_loop_admission_contract_sha256 as _closed_loop_admission_contract_sha256,
    mtp_acceptance_replay_manifest,
    mtp_acceptance_step_from_backend_step,
    normalize_mtp_acceptance_replay_manifest,
    pd_decode_admission_contract_sha256,
)
from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
    normalize_mtp_acceptance_assumption_manifest,
)
from src.serving.request_gen.request_stream import (
    RequestSpec,
    RequestStream,
    request_stream_sha256,
)
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model, pipeline_layer_ranges
from src.serving.schedule.backend_protocol import (
    BackendConfig,
    scheduler_effective_prefix_hit_tokens,
)
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest
from src.serving.host_timing import (
    EngineCoreHostRoundProfile,
    SchedulerHostProfile,
)
from fractions import Fraction

from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    DEEPSEEK_V4_FLASH_CONFIG_SHA256,
    DEEPSEEK_V4_PRO_CONFIG_SHA256,
    GLM_BLOCK_FP8_CONFIG_SHA256S,
    KIMI_K2_7_CODE_CONFIG_SHA256,
    KIMI_K3_CONFIG_SHA256,
    is_validated_llama_tp2_model_scope,
    is_validated_qwen3_moe_ep2_model_scope,
)
from src.serving.deployment.validated_scope import (
    TARGET_MODEL_PRODUCTION_PARALLELISM,
    is_validated_target_scheduler_scope,
)


# The native Scheduler supplies decisions only. It has no independently
# measured Forward timing service, so local process elapsed time is excluded
# from virtual time rather than sampled at simulation runtime.
_VIRTUAL_SCHEDULER_CPU_S = 0.0


@dataclass(frozen=True)
class _VLLM030RequestRows:
    """Source projection of GPU RequestState rows, not a GPU state receipt.

    GPUModelRunner.finish_requests sorts removals; RequestState.add_request
    pops the free stack. Admission order comes from scheduled_new_reqs, not
    the model batch's independently ordered num_scheduled_tokens mapping.
    """
    capacity: int
    residents: tuple[tuple[str, int], ...]
    free_indices: tuple[int, ...]

    @classmethod
    def empty(cls, capacity):
        if type(capacity) is not int or capacity<1:
            raise ValueError('native request rows require a positive capacity')
        return cls(capacity,(),tuple(range(capacity)))

    def advance(self, *, finished_ids, new_ids):
        if (type(finished_ids) is not tuple or type(new_ids) is not tuple
                or len(set(finished_ids))!=len(finished_ids) or len(set(new_ids))!=len(new_ids)
                or any(type(key) is not str or not key for key in (*finished_ids,*new_ids))):
            raise ValueError('native request rows require distinct explicit finished/new request IDs')
        residents=dict(self.residents);free=list(self.free_indices)
        for key in sorted(finished_ids):
            if key in residents:
                free.append(residents[key]);del residents[key]
        for key in new_ids:
            # add_requests also removes a streaming request before re-adding.
            if key in residents:
                free.append(residents[key]);del residents[key]
            if not free:
                raise ValueError('native request row capacity exhausted before admission')
            residents[key]=free.pop()
        return type(self)(self.capacity,tuple(residents.items()),tuple(free))


class _LengthOnlyTokenIds(Sequence[int]):
    """Mutable constant-token sequence for the no-model Scheduler adapter."""

    __slots__ = ("_length", "_token_id")

    def __init__(self, token_id: int, length: int) -> None:
        self._token_id = _vllm_integer(token_id, "length-only token ID")
        self._length = _vllm_integer(length, "length-only token count")
        if self._length < 0:
            raise ValueError("vLLM length-only token count cannot be negative")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item):
        if isinstance(item, slice):
            selected = range(self._length)[item]
            return [self._token_id] * len(selected)
        index = item
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError("length-only token index out of range")
        return self._token_id

    def __iter__(self):
        return repeat(self._token_id, self._length)

    def __contains__(self, item) -> bool:
        return self._length > 0 and item == self._token_id

    def __eq__(self, other) -> bool:
        if isinstance(other, _LengthOnlyTokenIds):
            return (self._token_id, self._length) == (
                other._token_id, other._length)
        if isinstance(other, Sequence):
            return len(other) == self._length and all(
                token_id == self._token_id for token_id in other)
        return False

    def copy(self) -> "_LengthOnlyTokenIds":
        return _LengthOnlyTokenIds(self._token_id, self._length)

    def append(self, token_id: int) -> None:
        if token_id != self._token_id:
            raise ValueError(
                "length-only vLLM request received a non-dummy token")
        self._length += 1

    def extend(self, token_ids: Iterable[int]) -> None:
        appended = tuple(token_ids)
        if any(token_id != self._token_id for token_id in appended):
            raise ValueError(
                "length-only vLLM request received a non-dummy token")
        self._length += len(appended)

    def index(self, token_id: int, start: int = 0, stop: int | None = None) -> int:
        if token_id != self._token_id:
            raise ValueError(f"{token_id!r} is not in length-only token IDs")
        normalized_start, normalized_stop, _ = slice(
            start, stop, 1).indices(self._length)
        if normalized_start >= normalized_stop:
            raise ValueError(f"{token_id!r} is not in length-only token IDs")
        return normalized_start


def _vllm_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"vLLM {field} must be an integer")
    return value


def _scheduled_ordinary_token_ids(
    token_ids: Sequence[int], *, computed_tokens: int, rows: int,
    num_tokens: int, pending_outputs: int, allow_pending_outputs: bool,
    declared_token_ids: tuple[int, ...] | None = None,
) -> tuple[int, ...] | None:
    """Keep native async output placeholders unknown, never replace IDs."""
    if len(token_ids) != num_tokens:
        raise RuntimeError("native token prefix length disagrees with num_tokens")
    ordinary = tuple(_vllm_integer(token_id, "scheduled length-only input token ID")
        for token_id in token_ids[computed_tokens:computed_tokens + rows])
    if len(ordinary) == rows:
        return ordinary
    # vLLM0.11/0.30 AsyncScheduler schedules num_tokens + output placeholders,
    # while Request.all_token_ids contains only the resolved contiguous prefix.
    # The pre-schedule placeholder count, not the increment for this step,
    # bounds the missing input suffix. Existing IDs remain validated above.
    if (allow_pending_outputs and pending_outputs > 0
            and computed_tokens >= 0 and rows > 0
            and computed_tokens + rows <= num_tokens + pending_outputs):
        if declared_token_ids is not None:
            if (not isinstance(declared_token_ids, tuple)
                    or any(type(token) is not int or token < 0 for token in declared_token_ids)
                    or tuple(declared_token_ids[:num_tokens]) != tuple(token_ids)
                    or computed_tokens + rows > len(declared_token_ids)):
                raise ValueError("explicit async token transcript disagrees with the resolved native prefix or scheduled rows")
            return declared_token_ids[computed_tokens:computed_tokens + rows]
        return None
    raise RuntimeError("native scheduled token prefix does not cover the declared query")


def _vllm_request_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("vLLM request ID must be a non-empty string")
    return value


def _validate_mtp_replay_submitted_requests(
    context: MTPAcceptanceSchedulerContext,
    accepted_requests: RequestStream,
) -> None:
    """Bind a replay contract to requests actually admitted by Scheduler."""

    if not isinstance(context, MTPAcceptanceSchedulerContext):
        raise TypeError(
            "MTP replay submitted-request validation requires a Scheduler "
            "context")
    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "MTP replay submitted-request validation requires a RequestStream")
    if context.adapter_contract == COLOCATED_MTP_REPLAY_ADAPTER_CONTRACT:
        actual_identity = request_stream_sha256(accepted_requests)
        expected_identity = context.request_stream_sha256
        identity_name = "request stream"
    elif context.adapter_contract == PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT:
        actual_identity = pd_decode_admission_contract_sha256(
            accepted_requests)
        expected_identity = context.decode_admission_contract_sha256
        identity_name = "decode admission contract"
    else:
        raise RuntimeError("unknown MTP replay adapter contract")
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"MTP acceptance replay {identity_name} hash disagrees with the "
            "requests actually submitted to the Scheduler: "
            f"expected={expected_identity}, actual={actual_identity}")


def _validate_fixed_count_mtp_submitted_requests(
    expected_request_stream_sha256: str,
    accepted_requests: RequestStream,
) -> None:
    """Bind a hash-scoped fixed-count MTP run to submitted requests."""

    if (not isinstance(expected_request_stream_sha256, str)
            or len(expected_request_stream_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in expected_request_stream_sha256)):
        raise ValueError(
            "fixed-count MTP request stream identity must be a SHA256")
    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "fixed-count MTP request validation requires a RequestStream")
    actual_identity = request_stream_sha256(accepted_requests)
    if actual_identity != expected_request_stream_sha256:
        raise RuntimeError(
            "fixed-count MTP request stream hash disagrees with the requests "
            "actually submitted to the Scheduler: "
            f"expected={expected_request_stream_sha256}, "
            f"actual={actual_identity}")


def _admission_order_diagnostic(accepted_requests: RequestStream) -> str:
    """Summarize the actual Scheduler admission order for hash mismatches."""

    request_ids = tuple(
        request.request_id for request in accepted_requests)
    return (
        f"count={len(request_ids)}, "
        f"ids={request_ids!r}")


def _validate_assumption_mtp_submitted_requests(
    assumption: MTPAcceptanceAssumption,
    decode_admission_contract_sha256: str | None,
    accepted_requests: RequestStream,
    closed_loop_admission_contract_sha256: str | None = None,
) -> None:
    """Bind a maximum-entropy policy to actual Scheduler admissions."""

    if not isinstance(assumption, MTPAcceptanceAssumption):
        raise TypeError(
            "maximum-entropy MTP submitted-request validation requires an "
            "assumption")
    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "maximum-entropy MTP request validation requires RequestStream")
    if (decode_admission_contract_sha256 is not None
            and closed_loop_admission_contract_sha256 is not None):
        raise ValueError(
            "maximum-entropy MTP cannot bind both a decode admission "
            "contract and a closed-loop admission contract")
    if closed_loop_admission_contract_sha256 is not None:
        expected_identity = closed_loop_admission_contract_sha256
        actual_identity = _closed_loop_admission_contract_sha256(
            accepted_requests)
        identity_name = "closed-loop admission contract"
    elif decode_admission_contract_sha256 is None:
        expected_identity = assumption.request_stream_sha256
        actual_identity = request_stream_sha256(accepted_requests)
        identity_name = "request stream"
    else:
        expected_identity = decode_admission_contract_sha256
        actual_identity = pd_decode_admission_contract_sha256(
            accepted_requests)
        identity_name = "decode admission contract"
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"maximum-entropy MTP {identity_name} hash disagrees with the "
            "requests actually submitted to the Scheduler: "
            f"expected={expected_identity}, actual={actual_identity}; "
            f"divergence={_admission_order_diagnostic(accepted_requests)}")


def _validate_standalone_af_mtp_preemption_requests(
    accepted_requests: RequestStream,
) -> None:
    """Reject direct-API request streams outside the validated AF slice."""

    if not isinstance(accepted_requests, RequestStream):
        raise TypeError(
            "standalone AF MTP preemption validation requires RequestStream")
    request_shapes = tuple(
        (
            request.arrival_time_s,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in accepted_requests
    )
    if request_shapes != (
            (0.0, 32, 2, 0),
            (0.0, 32, 4, 32)):
        raise NotImplementedError(
            "standalone AF MTP preemption direct API requires exactly two "
            "same-time requests shaped (32->2, prefix=0) and "
            "(32->4, prefix=32) in that order")


@dataclass(frozen=True)
class VLLMSchedulerConfig:
    """Inputs needed to construct vLLM's CPU scheduler without a worker."""

    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    prefix_cache_mode: str = "trace"
    async_scheduling: bool = False
    dummy_token_id: int = 42
    mtp_num_speculative_tokens: int = 0
    mtp_acceptance_policy: str = "disabled"
    mtp_fixed_accepted_tokens: int = 0
    mtp_acceptance_source_id: str = "disabled"
    mtp_acceptance_replay: MTPAcceptanceReplay | None = None
    mtp_draft_token_ids_by_step: Mapping[str, Mapping[str, tuple[int, ...]]] | None = None
    dspark_draft_model: str | None = None
    dspark_source_inputs_by_step: Mapping[str, object] | None = None
    mtp_acceptance_assumption: MTPAcceptanceAssumption | None = None
    request_stream_sha256: str | None = None
    decode_admission_contract_sha256: str | None = None
    closed_loop_admission_contract_sha256: str | None = None
    authorized_h200_pd_mtp_sweep: bool = False
    scheduler_host_profile: SchedulerHostProfile | None = None
    engine_core_host_round_profile: EngineCoreHostRoundProfile | None = None
    scheduler_version: str = "0.11.0"
    context_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_layer_partition: tuple[int, ...] | None = None
    simple_cpu_offload_bytes_per_rank: int | None = None
    async_terminal_guard_source: str | None = None
    speculative_method: str = "mtp"
    dspark_draft_model: str | None = None
    emit_glm_mtp_source_inputs: bool = False

    def __post_init__(self) -> None:
        if type(self.emit_glm_mtp_source_inputs) is not bool:
            raise TypeError("GLM MTP source export requires a boolean")
        if self.emit_glm_mtp_source_inputs and (
                self.scheduler_version != "0.30.0" or self.speculative_method != "mtp"
                or self.mtp_num_speculative_tokens < 1 or self.prefix_cache_mode != "trace"):
            raise ValueError("GLM MTP native source export requires explicit0.30 MTP and length-only trace inputs")
        if self.speculative_method not in ("mtp", "dspark"):
            raise ValueError("Scheduler speculative method must be mtp or dspark")
        if self.speculative_method == "dspark":
            if (not isinstance(self.dspark_draft_model, str) or not self.dspark_draft_model
                    or self.scheduler_version != "0.30.0" or self.mtp_num_speculative_tokens != 5
                    or self.mtp_acceptance_policy != "fixed_count" or self.async_scheduling
                    or self.context_parallel_size != 1 or self.enable_prefix_caching):
                raise NotImplementedError("DSpark Scheduler requires an explicit draft, native0.30 CP1 sync depth5 fixed-count source")
        elif self.dspark_draft_model is not None:
            raise ValueError("DSpark draft path requires the explicit dspark speculative method")
        if self.prefix_cache_mode not in ("trace", "token_ids", "trace_no_hit"):
            raise ValueError("prefix_cache_mode must be trace, token_ids or trace_no_hit")
        if self.prefix_cache_mode == "trace_no_hit" and (
                not self.enable_prefix_caching or self.scheduler_version != "0.30.0"
                or not self.async_scheduling or self.tensor_parallel_size not in (1, 2)
                or self.pipeline_parallel_size != 1 or (self.enable_expert_parallel and self.tensor_parallel_size != 2)
                or self.mtp_num_speculative_tokens):
            raise NotImplementedError(
                "trace_no_hit requires 0.30 ordinary TP1/TP2 async prefix caching")
        if self.prefix_cache_mode == "token_ids":
            if not self.enable_prefix_caching:
                raise ValueError("token_ids mode requires enable_prefix_caching")
            if (self.scheduler_version != "0.11.0"
                    and not (self.scheduler_version == "0.30.0"
                        and self.simple_cpu_offload_bytes_per_rank is not None)):
                raise NotImplementedError("token prefix cache requires native0.11 or explicit 0.30 SimpleCPU offload")
            if self.async_scheduling:
                raise NotImplementedError("token prefix cache does not support async scheduling")
        if self.simple_cpu_offload_bytes_per_rank is not None:
            if (type(self.simple_cpu_offload_bytes_per_rank) is not int
                    or self.simple_cpu_offload_bytes_per_rank <= 0):
                raise ValueError("SimpleCPU Host capacity per rank must be a positive integer")
            if (self.scheduler_version != "0.30.0"
                    or self.prefix_cache_mode != "token_ids"
                    or not self.enable_prefix_caching
                    or self.pipeline_parallel_size != 1
                    or self.enable_expert_parallel
                    or self.mtp_num_speculative_tokens != 0):
                raise NotImplementedError("0.30 SimpleCPU offload requires token IDs, prefix caching, PP1, EP-off and MTP-off")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("vLLM model path must be a non-empty string")
        for name in (
                "tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size", "data_parallel_size",
                "dummy_token_id", "mtp_num_speculative_tokens",
                "mtp_fixed_accepted_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"vLLM {name} must be an integer")
        if min(self.tensor_parallel_size, self.pipeline_parallel_size, self.context_parallel_size, self.data_parallel_size) < 1:
            raise ValueError("vLLM parallel sizes must be positive")
        if self.context_parallel_size > 1 and (self.scheduler_version != "0.30.0"
                or self.pipeline_parallel_size != 1 or self.mtp_num_speculative_tokens):
            raise NotImplementedError("PCP CPU cache projection requires vLLM0.30 PP1 and MTP-off")
        if self.data_parallel_size > 1 and (self.scheduler_version != "0.30.0"
                or self.mtp_num_speculative_tokens and (self.context_parallel_size != 1
                    or self.mtp_num_speculative_tokens != (5 if self.speculative_method == "dspark" else 2)
                    or self.mtp_acceptance_policy != "fixed_count")):
            raise NotImplementedError("coordinated GLM DP requires vLLM0.30 MTP whole-shard TP/CP1 depth2 fixed-count")
        if self.pipeline_layer_partition is not None and self.scheduler_version != "0.30.0":
            raise NotImplementedError("explicit pipeline partition requires the vLLM0.30 cache/source contract")
        if self.pipeline_layer_partition is not None and (not isinstance(self.pipeline_layer_partition, tuple)
                or len(self.pipeline_layer_partition) != self.pipeline_parallel_size
                or any(type(layers) is not int or layers < 1 for layers in self.pipeline_layer_partition)):
            raise ValueError("vLLM pipeline_layer_partition requires one positive integer per PP stage")
        if self.dummy_token_id < 0:
            raise ValueError("vLLM dummy_token_id cannot be negative")
        if min(
                self.mtp_num_speculative_tokens,
                self.mtp_fixed_accepted_tokens) < 0:
            raise ValueError("vLLM MTP token counts cannot be negative")
        # Numerical first-proposal frames opt into a narrower contract than
        # the existing CPU Scheduler/cache projection with a paired draft.
        if self.dspark_source_inputs_by_step is not None:
            from src.serving.workload.runtime_inputs import DSparkFirstProposalInputs
            if (type(self.dspark_draft_model) is not str or not self.dspark_draft_model
                    or not isinstance(self.dspark_source_inputs_by_step,Mapping) or not self.dspark_source_inputs_by_step
                    or self.scheduler_version!="0.30.0" or self.tensor_parallel_size!=8 or self.pipeline_parallel_size!=1
                    or self.enable_expert_parallel or self.async_scheduling or self.enable_prefix_caching
                    or self.enable_chunked_prefill or self.mtp_num_speculative_tokens!=5
                    or self.mtp_acceptance_policy!="fixed_count" or self.dtype!="bfloat16"):
                raise NotImplementedError("DSpark Scheduler requires paired0.30 eagerTP8 K5 fixed-count source and explicit frames")
            frames=dict(self.dspark_source_inputs_by_step)
            if any(type(key)is not str or not key or not isinstance(value,DSparkFirstProposalInputs)
                    or value.source_id!=self.mtp_acceptance_source_id for key,value in frames.items()):
                raise ValueError("DSpark Scheduler frames must be explicit typed states with matching source identity")
            from src.serving.workload.model_config import load_model_config
            target=load_model_config(self.model)
            draft=load_model_config(self.dspark_draft_model)
            if (draft.model_id!=target.model_id+"-DSpark" or draft.dspark_block_size!=5
                    or any(value.draft_model.config_sha256!=draft.config_sha256
                        or value.draft_model.model_id!=draft.model_id for value in frames.values())):
                raise ValueError("DSpark source frames must bind the actual paired draft config identity")
            object.__setattr__(self,"dspark_source_inputs_by_step",frames)
        if self.mtp_draft_token_ids_by_step is not None:
            if self.prefix_cache_mode != "token_ids" or not self.mtp_num_speculative_tokens:
                raise ValueError("explicit MTP draft IDs require token_ids mode and MTP")
            if not isinstance(self.mtp_draft_token_ids_by_step, Mapping):
                raise TypeError("MTP draft frames must be a step/request mapping")
            frames = {}
            for step_id, rows in self.mtp_draft_token_ids_by_step.items():
                if not isinstance(step_id, str) or not step_id or not isinstance(rows, Mapping):
                    raise ValueError("MTP draft frames require nonempty step IDs and request mappings")
                frame = {}
                for request_id, tokens in rows.items():
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError("MTP draft frames require nonempty request IDs")
                    if (not isinstance(tokens, tuple) or len(tokens) != self.mtp_num_speculative_tokens
                            or any(type(token) is not int or token < 0 for token in tokens)):
                        raise ValueError("MTP draft IDs must be nonnegative integer tuples covering proposer depth")
                    frame[request_id] = tokens
                frames[step_id] = frame
            # Plain copied dictionaries preserve multiprocessing spawn/pickle.
            object.__setattr__(self, "mtp_draft_token_ids_by_step", frames)
        if self.max_model_len is not None:
            if isinstance(self.max_model_len, bool) or not isinstance(
                    self.max_model_len, int):
                raise TypeError("vLLM max_model_len must be an integer or null")
            if self.max_model_len < 1:
                raise ValueError("vLLM max_model_len must be positive")
        if (self.scheduler_host_profile is not None
                and not isinstance(self.scheduler_host_profile,
                                   SchedulerHostProfile)):
            raise TypeError("vLLM Scheduler Host profile is invalid")
        if (self.engine_core_host_round_profile is not None
                and not isinstance(self.engine_core_host_round_profile,
                                   EngineCoreHostRoundProfile)):
            raise TypeError(
                "vLLM EngineCore host-round profile is invalid")
        if (isinstance(self.gpu_memory_utilization, bool)
                or not isinstance(self.gpu_memory_utilization, (int, float))
                or not math.isfinite(self.gpu_memory_utilization)
                or not 0 < self.gpu_memory_utilization <= 1):
            raise ValueError(
                "vLLM gpu_memory_utilization must be finite and in (0, 1]")
        if not isinstance(self.dtype, str):
            raise TypeError("vLLM dtype must be a string")
        if self.dtype not in ("bfloat16", "float16"):
            raise ValueError(
                "vLLM dtype must be 'bfloat16' or 'float16', got "
                f"{self.dtype!r}")
        for name in (
                "trust_remote_code", "enable_chunked_prefill",
                "enable_prefix_caching", "enable_expert_parallel",
                "async_scheduling"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"vLLM {name} must be boolean")
        if not isinstance(self.authorized_h200_pd_mtp_sweep, bool):
            raise TypeError("H200 PD MTP sweep authorization must be boolean")
        if self.scheduler_version not in ("0.11.0", "0.30.0"):
            raise ValueError("vLLM scheduler_version must be 0.11.0 or 0.30.0")
        if self.async_terminal_guard_source is not None:
            if self.async_terminal_guard_source != "vllm_0_18_0_running_max_tokens_guard":
                raise ValueError("unknown vLLM async terminal guard source")
            if (self.scheduler_version != "0.11.0"
                    or not self.async_scheduling
                    or self.mtp_num_speculative_tokens != 0):
                raise NotImplementedError(
                    "vLLM 0.18 terminal guard bridge requires native 0.11 "
                    "async scheduling without speculative tokens")
        if self.scheduler_version == "0.11.0" and not self.enable_chunked_prefill:
            raise ValueError(
                "the pinned vLLM 0.11 Scheduler contract requires chunked "
                "prefill; enable_chunked_prefill must be true")
        if not isinstance(self.mtp_acceptance_policy, str):
            raise TypeError("vLLM MTP acceptance policy must be a string")
        if not isinstance(self.mtp_acceptance_source_id, str):
            raise TypeError("vLLM MTP acceptance source ID must be a string")
        if (self.mtp_acceptance_replay is not None
                and not isinstance(
                    self.mtp_acceptance_replay, MTPAcceptanceReplay)):
            raise TypeError(
                "vLLM MTP acceptance replay must be MTPAcceptanceReplay or null")
        if (self.mtp_acceptance_assumption is not None
                and not isinstance(
                    self.mtp_acceptance_assumption,
                    MTPAcceptanceAssumption)):
            raise TypeError(
                "vLLM MTP acceptance assumption must be "
                "MTPAcceptanceAssumption or null")
        if (self.request_stream_sha256 is not None
                and (not isinstance(self.request_stream_sha256, str)
                     or len(self.request_stream_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character in self.request_stream_sha256))):
            raise ValueError(
                "vLLM request stream SHA256 must be a lowercase SHA256 or null")
        if (self.decode_admission_contract_sha256 is not None
                and (not isinstance(
                    self.decode_admission_contract_sha256, str)
                     or len(self.decode_admission_contract_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character
                            in self.decode_admission_contract_sha256))):
            raise ValueError(
                "vLLM decode admission contract SHA256 must be a lowercase "
                "SHA256 or null")
        if (self.closed_loop_admission_contract_sha256 is not None
                and (not isinstance(
                    self.closed_loop_admission_contract_sha256, str)
                     or len(self.closed_loop_admission_contract_sha256) != 64
                     or any(character not in "0123456789abcdef"
                            for character
                            in self.closed_loop_admission_contract_sha256))):
            raise ValueError(
                "vLLM closed-loop admission contract SHA256 must be a "
                "lowercase SHA256 or null")
        if (self.closed_loop_admission_contract_sha256 is not None
                and self.decode_admission_contract_sha256 is not None):
            raise ValueError(
                "vLLM closed-loop admission contract cannot coexist with a "
                "decode admission contract")
        if (self.closed_loop_admission_contract_sha256 is not None
                and self.mtp_acceptance_policy != "maximum_entropy"):
            raise ValueError(
                "vLLM closed-loop admission contract requires the "
                "maximum-entropy MTP acceptance policy")
        if self.mtp_num_speculative_tokens == 0:
            if self.mtp_acceptance_policy != "disabled":
                raise ValueError(
                    "disabled vLLM MTP requires acceptance policy 'disabled'")
            if self.mtp_fixed_accepted_tokens != 0:
                raise ValueError(
                    "disabled vLLM MTP cannot accept speculative tokens")
            if self.mtp_acceptance_source_id != "disabled":
                raise ValueError(
                    "disabled vLLM MTP requires source ID 'disabled'")
            if self.mtp_acceptance_replay is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry an acceptance replay")
            if self.mtp_acceptance_assumption is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry an acceptance assumption")
            if self.request_stream_sha256 is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry a request stream hash")
            if self.decode_admission_contract_sha256 is not None:
                raise ValueError(
                    "disabled vLLM MTP cannot carry a decode admission hash")
        elif self.mtp_acceptance_policy == "fixed_count":
            if (self.mtp_fixed_accepted_tokens
                    > self.mtp_num_speculative_tokens):
                raise ValueError(
                    "fixed MTP accepted tokens cannot exceed proposals")
            if (not self.mtp_acceptance_source_id
                    or self.mtp_acceptance_source_id == "disabled"):
                raise ValueError(
                    "fixed MTP acceptance requires a provenance source ID")
            if self.mtp_acceptance_replay is not None:
                raise ValueError(
                    "fixed-count MTP cannot carry an acceptance replay")
            if self.mtp_acceptance_assumption is not None:
                raise ValueError(
                    "fixed-count MTP cannot carry an acceptance assumption")
            if self.decode_admission_contract_sha256 is not None:
                raise ValueError(
                    "fixed-count MTP cannot carry a decode admission hash")
        elif self.mtp_acceptance_policy == "maximum_entropy":
            if self.mtp_fixed_accepted_tokens != 0:
                raise ValueError(
                    "maximum-entropy MTP cannot carry fixed accepted tokens")
            if self.mtp_acceptance_replay is not None:
                raise ValueError(
                    "maximum-entropy MTP cannot carry an acceptance replay")
            if self.mtp_acceptance_assumption is None:
                raise ValueError(
                    "maximum-entropy MTP requires an acceptance assumption")
            assumption = self.mtp_acceptance_assumption
            if self.mtp_acceptance_source_id != assumption.source_id:
                raise ValueError(
                    "maximum-entropy MTP source ID disagrees with the "
                    "assumption")
            if self.mtp_num_speculative_tokens != assumption.proposal_depth:
                raise ValueError(
                    "maximum-entropy MTP proposal depth disagrees with the "
                    "assumption")
            if self.request_stream_sha256 != assumption.request_stream_sha256:
                raise ValueError(
                    "maximum-entropy MTP request stream hash disagrees with "
                    "the assumption")
        elif self.mtp_acceptance_policy == "replay":
            if self.mtp_fixed_accepted_tokens != 0:
                raise ValueError(
                    "replayed MTP cannot carry fixed accepted tokens")
            if self.mtp_acceptance_replay is None:
                raise ValueError(
                    "replayed MTP requires an acceptance replay artifact")
            if self.mtp_acceptance_assumption is not None:
                raise ValueError(
                    "replayed MTP cannot carry an acceptance assumption")
            if self.request_stream_sha256 is None:
                raise ValueError(
                    "replayed MTP requires the actual request stream SHA256")
            replay = self.mtp_acceptance_replay
            if self.mtp_acceptance_source_id != replay.artifact_id:
                raise ValueError(
                    "replayed MTP source ID must equal the artifact ID")
            if (self.mtp_num_speculative_tokens
                    != replay.scheduler_context.num_speculative_tokens):
                raise ValueError(
                    "replayed MTP proposal depth disagrees with the artifact")
            if (self.request_stream_sha256
                    != replay.scheduler_context.ingress_identity_sha256):
                raise ValueError(
                    "replayed MTP request stream hash disagrees with the "
                    "artifact")
            if replay.schema_version == 1:
                if self.decode_admission_contract_sha256 is not None:
                    raise ValueError(
                        "colocated MTP replay cannot carry a decode admission "
                        "hash")
            elif replay.schema_version == 2:
                if (self.decode_admission_contract_sha256
                        != replay.scheduler_context.
                        decode_admission_contract_sha256):
                    raise ValueError(
                        "replayed MTP decode admission hash disagrees with "
                        "the artifact")
            else:
                raise AssertionError("unknown MTP replay schema version")
        else:
            raise ValueError(
                "enabled vLLM MTP requires fixed_count, maximum_entropy, or "
                "replay acceptance")


def _validate_vllm030_host_profiles(scheduler_profile, engine_profile=None):
    """Only native-version service may pace the real 0.30 control plane."""
    from src.serving.backend.vllm_scheduler_process import VLLM030_SOURCE_COMMIT
    for role, profile in (("Scheduler", scheduler_profile), ("EngineCore", engine_profile)):
        if profile is None:
            continue
        # Timing proxies do not change the separately pinned native Scheduler.
        if (role == "Scheduler" and isinstance(profile, SchedulerHostProfile)
                and profile.shared_cpu_timing_target == "0.30.0"):
            prefix = f"vllm={profile.vllm_version};independent_cpu_mechanism=true;source_files_sha256="
            identity = profile.source_identity
            digest = identity[len(prefix):] if identity.startswith(prefix) else ""
            if (profile.vllm_version not in ("0.11.0", "0.30.0")
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise NotImplementedError("shared CPU timing requires explicit independent source-file evidence")
            continue
        if profile.vllm_version != "0.30.0":
            raise NotImplementedError(f"0.30 {role} Host service requires an exact0.30 profile")
        fields = profile.source_identity.split(";")
        # The existing Scheduler fitter emits one recorded commit this way.
        canonical_scheduler_source = (role == "Scheduler" and profile.source_identity
            == f"vllm=0.30.0;source_identities={VLLM030_SOURCE_COMMIT}")
        explicit_pinned_source = (
            {field for field in fields if field.startswith("source_commit=")}
                == {f"source_commit={VLLM030_SOURCE_COMMIT}"}
            and all(field == "source_identities=vllm=0.30.0" for field in fields if field.startswith("source_identities="))
            and all(field == "vllm=0.30.0" for field in fields if field.startswith("vllm=")))
        if ((not explicit_pinned_source and not canonical_scheduler_source)
                or any("substitute=" in field for field in fields)):
            raise NotImplementedError(f"0.30 {role} Host service requires its pinned source commit without version substitution")


def _validate_vllm030_dtype_scope(model, scheduler_config, role):
    """Bind native cache dtype to the declared unquantized model source."""
    expected = {"bf16": "bfloat16", "fp16": "float16"}
    if model.compute_dtype not in expected or scheduler_config.dtype != expected[model.compute_dtype]:
        raise ValueError("0.30 Scheduler dtype disagrees with source model compute dtype")
    if model.compute_dtype == "fp16":
        from src.serving.workload.model_config import is_dense_parallel_model_scope
        if (role != "colocated" or scheduler_config.async_scheduling
                or scheduler_config.enable_expert_parallel or scheduler_config.mtp_num_speculative_tokens
                or scheduler_config.simple_cpu_offload_bytes_per_rank is not None
                or not is_dense_parallel_model_scope(model, scheduler_config.tensor_parallel_size,
                    scheduler_config.pipeline_parallel_size)):
            raise NotImplementedError("0.30 FP16 control plane requires ordinary eager dense Qwen3 TP/PP source")


@dataclass(frozen=True)
class _SchedulerModelView:
    """The text-model fields consumed by the pinned vLLM Scheduler.

    This is deliberately not a second model schema.  Values come from the
    canonical Serving ``ModelConfig`` and expose only the control-plane view
    used by vLLM 0.11.  In particular, constructing this view does not invoke
    vLLM's model registry, tokenizer, model loader, or accelerator platform.
    """

    max_model_len: int
    vocab_size: int
    dtype: Any
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    skip_tokenizer_init: bool = True

    def get_vocab_size(self) -> int:
        return self.vocab_size


@dataclass(frozen=True)
class _SchedulerDraftHFConfig:
    model_type: str
    num_nextn_predict_layers: int


@dataclass(frozen=True)
class _SchedulerDraftModelConfig:
    hf_config: _SchedulerDraftHFConfig
    max_model_len: int


@dataclass(frozen=True)
class _SchedulerSpeculativeConfig:
    method: str
    num_speculative_tokens: int
    draft_model_config: _SchedulerDraftModelConfig
    disable_padded_drafter_batch: bool = False

    def use_eagle(self) -> bool:
        return self.method == "mtp"


@dataclass(frozen=True)
class _SchedulerOnlyVllmConfig:
    """Pinned vLLM Scheduler dependency bundle without ``EngineArgs``."""

    model_config: _SchedulerModelView
    cache_config: Any
    parallel_config: Any
    scheduler_config: Any
    structured_outputs_config: Any
    kv_geometry_representative_pipeline_stage_rank: int
    kv_geometry_representative_layer_start: int
    kv_geometry_representative_layer_end: int
    lora_config: None = None
    speculative_config: _SchedulerSpeculativeConfig | None = None
    kv_transfer_config: None = None
    kv_events_config: None = None


def mtp_scenario_manifest(config: VLLMSchedulerConfig) -> dict[str, object]:
    """Return the canonical, replayable MTP scenario declaration."""

    if not isinstance(config, VLLMSchedulerConfig):
        raise TypeError("MTP scenario requires VLLMSchedulerConfig")
    if config.mtp_acceptance_policy == "disabled":
        validation_status = "disabled"
    elif config.mtp_acceptance_policy in ("fixed_count", "maximum_entropy"):
        validation_status = "projected_explicit_assumption"
    elif config.mtp_acceptance_policy == "replay":
        validation_status = "replayed_external_observation"
    else:
        raise AssertionError(
            "validated MTP config has an unknown acceptance policy: "
            f"{config.mtp_acceptance_policy!r}")
    manifest = {
        "num_speculative_tokens": config.mtp_num_speculative_tokens,
        "acceptance_policy": config.mtp_acceptance_policy,
        "fixed_accepted_tokens": config.mtp_fixed_accepted_tokens,
        "acceptance_source_id": config.mtp_acceptance_source_id,
        "acceptance_replay": (
            None
            if config.mtp_acceptance_replay is None
            else mtp_acceptance_replay_manifest(
                config.mtp_acceptance_replay)
        ),
        "validation_status": validation_status,
    }
    if config.mtp_acceptance_policy == "maximum_entropy":
        if config.mtp_acceptance_assumption is None:
            raise AssertionError(
                "validated maximum-entropy MTP assumption disappeared")
        manifest["acceptance_assumption"] = (
            config.mtp_acceptance_assumption.manifest())
    return manifest


def normalize_mtp_scenario_manifest(value: Any) -> dict[str, object]:
    """Validate an external MTP scenario at its owner boundary."""

    if not isinstance(value, Mapping):
        raise TypeError("MTP scenario must be a mapping")
    raw = dict(value)
    common_fields = {
        "num_speculative_tokens",
        "acceptance_policy",
        "fixed_accepted_tokens",
        "acceptance_source_id",
        "acceptance_replay",
        "validation_status",
    }
    if "acceptance_policy" not in raw:
        raise ValueError("MTP scenario is missing acceptance_policy")
    policy = raw["acceptance_policy"]
    fields = set(common_fields)
    if policy == "maximum_entropy":
        fields.add("acceptance_assumption")
    if set(raw) != fields:
        raise ValueError(
            "MTP scenario fields must be exactly "
            f"{sorted(fields)}, got {sorted(raw)}")
    if policy == "replay":
        replay_manifest = normalize_mtp_acceptance_replay_manifest(
            raw["acceptance_replay"])
        normalized = {
            "num_speculative_tokens": raw["num_speculative_tokens"],
            "acceptance_policy": policy,
            "fixed_accepted_tokens": raw["fixed_accepted_tokens"],
            "acceptance_source_id": raw["acceptance_source_id"],
            "acceptance_replay": replay_manifest,
            "validation_status": "replayed_external_observation",
        }
        if (isinstance(raw["num_speculative_tokens"], bool)
                or not isinstance(raw["num_speculative_tokens"], int)
                or raw["num_speculative_tokens"] <= 0):
            raise ValueError(
                "replayed MTP num_speculative_tokens must be positive")
        if (isinstance(raw["fixed_accepted_tokens"], bool)
                or not isinstance(raw["fixed_accepted_tokens"], int)
                or raw["fixed_accepted_tokens"] != 0):
            raise ValueError(
                "replayed MTP fixed_accepted_tokens must be integer zero")
        if raw["acceptance_source_id"] != replay_manifest["artifact_id"]:
            raise ValueError(
                "replayed MTP source ID must equal the artifact ID")
        if (raw["num_speculative_tokens"]
                != replay_manifest["num_speculative_tokens"]):
            raise ValueError(
                "replayed MTP depth disagrees with the replay manifest")
    elif policy in ("disabled", "fixed_count"):
        if raw["acceptance_replay"] is not None:
            raise ValueError(
                "disabled or fixed-count MTP cannot carry acceptance replay")
        config = VLLMSchedulerConfig(
            "mtp-scenario-contract",
            mtp_num_speculative_tokens=raw["num_speculative_tokens"],
            mtp_acceptance_policy=policy,
            mtp_fixed_accepted_tokens=raw["fixed_accepted_tokens"],
            mtp_acceptance_source_id=raw["acceptance_source_id"],
        )
        normalized = mtp_scenario_manifest(config)
    elif policy == "maximum_entropy":
        if raw["acceptance_replay"] is not None:
            raise ValueError(
                "maximum-entropy MTP cannot carry acceptance replay")
        assumption_manifest = (
            normalize_mtp_acceptance_assumption_manifest(
                raw["acceptance_assumption"]))
        assumption = MTPAcceptanceAssumption(
            assumption_manifest["source_id"],
            assumption_manifest["proposal_depth"],
            assumption_manifest["mean_acceptance_rate"],
            assumption_manifest["root_seed"],
            assumption_manifest["request_stream_sha256"],
            (assumption_manifest["sampling_key"] if "sampling_key" in assumption_manifest
             else "step_stream_v1"),
        )
        config = VLLMSchedulerConfig(
            "mtp-scenario-contract",
            mtp_num_speculative_tokens=raw["num_speculative_tokens"],
            mtp_acceptance_policy=policy,
            mtp_fixed_accepted_tokens=raw["fixed_accepted_tokens"],
            mtp_acceptance_source_id=raw["acceptance_source_id"],
            mtp_acceptance_assumption=assumption,
            request_stream_sha256=assumption.request_stream_sha256,
        )
        normalized = mtp_scenario_manifest(config)
    else:
        raise ValueError(f"unknown MTP acceptance policy: {policy!r}")
    comparable = dict(raw)
    if policy == "maximum_entropy":
        comparable["acceptance_assumption"] = assumption_manifest
    if comparable != normalized:
        raise ValueError(
            "MTP scenario validation status disagrees with its policy: "
            f"expected={normalized['validation_status']!r}, "
            f"actual={raw['validation_status']!r}")
    return normalized



def _bind_vllm030_source_cache_projection(backend, scheduler_config, budgets, layouts):
    import hashlib
    import json
    cp, tp = scheduler_config.context_parallel_size, scheduler_config.tensor_parallel_size
    pp = scheduler_config.pipeline_parallel_size
    original = asdict(scheduler_config)
    digest = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    info = backend._control_plane.runtime_info
    native_arenas = tuple(info["shared_cache_arena_bytes_by_rank"])
    if len(native_arenas) != tp * pp:
        raise ValueError("source cache projection requires the complete genuine CP1 worker allocation")
    info.update(source_scheduler_config=original, source_scheduler_config_sha256=digest,
        source_context_parallel_size=cp, effective_context_parallel_size=1,
        source_expert_parallel_size=tp * cp if scheduler_config.enable_expert_parallel else 1,
        source_sequence_parallel_size=1,
        source_model_sequence_parallel=False, source_moe_sequence_parallel=False,
        context_parallel_projection=("single_native_cp1_scheduler_with_full_cache_on_each_source_pcp_rank"
            if cp > 1 else "native_cp1_cache_control_plane"),
        native_allocation_scope=("genuine_cp1_receipt; physical_pcp_replicas_are_source_projection"
            if cp > 1 else "genuine_cp1_dp1_cache_control_plane_receipt"),
        physical_available_memory_by_rank=budgets, native_shared_cache_arena_bytes_by_rank=native_arenas,
        supported_layouts_by_rank=layouts,
        shared_cache_arena_bytes_by_rank=tuple(native_arenas[stage * tp + lane]
            for stage in range(pp) for c in range(cp) for lane in range(tp)),
        physical_cache_replicas=tuple({"physical_local_rank": (stage * cp + c) * tp + lane,
            "pipeline_parallel_rank": stage, "context_parallel_rank": c, "tensor_parallel_rank": lane,
            "native_worker_index": stage * tp + lane, "allocated_tensor_bytes": native_arenas[stage * tp + lane],
            "cache_content_scope": "source_stage_cache_not_runtime_model_observed"}
            for stage in range(pp) for c in range(cp) for lane in range(tp)))

def _validated_request_queue_snapshot(value) -> dict:
    fields = {"waiting_request_ids", "running_request_ids", "deferred_request_ids",
        "pending_step_ids", "P_retained_request_ids", "D_reserved_request_ids"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("native queue snapshot has invalid fields")
    for name, identifiers in value.items():
        if (type(identifiers) is not tuple
                or any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
                or len(identifiers) != len(set(identifiers))):
            raise ValueError(f"native queue snapshot has invalid {name}")
    request_queues = (value["waiting_request_ids"], value["running_request_ids"], value["deferred_request_ids"])
    flattened = tuple(identifier for queue in request_queues for identifier in queue)
    if len(flattened) != len(set(flattened)):
        raise ValueError("native waiting, running and deferred queues overlap")
    return value


class VLLMBackend:
    """Framework-independent facade over a vLLM scheduler control plane.

    ``from_model`` is the production constructor.  Direct construction remains
    useful for a version-specific bridge supplied by an embedding application;
    that bridge must already implement the normalized control-plane methods.
    """

    def __init__(self, config: BackendConfig, control_plane: Any) -> None:
        self.config = config
        self._control_plane = control_plane
        self._accepted_request_ids: set[str] = set()
        self._in_process_baseline_libraries = None
        self._in_process_model_directory = None

    @classmethod
    def from_data_parallel_replicas(cls, replicas: tuple["VLLMBackend", ...], *, pool_name: str,
            source_scheduler_config: VLLMSchedulerConfig | None = None) -> "VLLMBackend":
        """Join actual independent native Schedulers into one synchronous DP step.

        This composes the control plane only. Expert sharing, physical rank
        ownership and DP/SP model computation still come from the deployment
        and source forward; they are not replaced by independent expert copies.
        """
        control = _DataParallelControlPlane(replicas, pool_name=pool_name, source_scheduler_config=source_scheduler_config)
        return cls(control.backend_config, control)

    @property
    def data_parallel_size(self) -> int:
        return len(self._control_plane.replicas) if isinstance(self._control_plane, _DataParallelControlPlane) else 1

    def request_data_parallel_rank(self, request_id: str) -> int:
        if isinstance(self._control_plane, _DataParallelControlPlane):
            return self._control_plane.request_data_parallel_rank(request_id)
        # A single replica has one physical placement even before decode
        # admission; reserve_pd_destination binds its pages before handoff.
        return 0

    @classmethod
    def from_model(
        cls,
        config: BackendConfig,
        scheduler_config: VLLMSchedulerConfig,
        *,
        in_process: bool = False,
    ) -> "VLLMBackend":
        if type(in_process) is not bool:
            raise TypeError("vLLM in_process must be boolean")
        _validate_native_scheduler_scope(config, scheduler_config)
        from src.serving.backend.vllm_scheduler_process import (
            VLLMSchedulerProcess,
        )

        if not in_process:
            control_plane = VLLMSchedulerProcess(config, scheduler_config)
            return cls(config, control_plane)
        if scheduler_config.scheduler_version != "0.11.0":
            raise NotImplementedError("in-process Scheduler requires the pinned 0.11 control plane")
        from src.serving.backend.vllm_scheduler_process import (
            _WORKER_ENVIRONMENT_FLAG,
            _install_cpu_device_guard,
            _runtime_info,
            _verify_no_accelerator_access,
            _verify_pinned_vllm_source,
        )
        _, version, package_manifest = _verify_pinned_vllm_source()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = "1"
        os.environ[_WORKER_ENVIRONMENT_FLAG] = "1"
        import tempfile
        configured_path = Path(scheduler_config.model).resolve()
        temporary_model_directory = None
        if configured_path.is_file():
            if configured_path.suffix != ".json":
                raise ValueError("vLLM scheduler model config file must use .json")
            temporary_model_directory = tempfile.TemporaryDirectory(
                prefix="midlayer-vllm-model-config-")
            model_path = Path(temporary_model_directory.name)
            (model_path / "config.json").write_bytes(configured_path.read_bytes())
        else:
            model_path = configured_path
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise ValueError("vLLM scheduler model must be a JSON config file or local config directory")
        scheduler_config = replace(scheduler_config, model=str(model_path))
        try:
            baseline_libraries = _install_cpu_device_guard()
            control_plane = _NativeVLLMControlPlane(config, scheduler_config)
            _verify_no_accelerator_access(baseline_libraries)
            control_plane.runtime_info = _runtime_info(
                version, package_manifest, scheduler_config,
                config, control_plane.vllm_config, control_plane.scheduler)
        except BaseException:
            if temporary_model_directory is not None:
                temporary_model_directory.cleanup()
            raise
        backend = cls(config, control_plane)
        backend._in_process_baseline_libraries = baseline_libraries
        backend._in_process_model_directory = temporary_model_directory
        return backend

    @classmethod
    def from_vllm030_scheduler(
        cls,
        config: BackendConfig,
        scheduler_config: VLLMSchedulerConfig,
        scheduler: Any,
        *,
        source_root: Path,
        worker_kv_configs: tuple[Any, ...],
    ) -> "VLLMBackend":
        """Bind a prepared real 0.30 CPU Scheduler to the simulation protocol.

        The caller owns process isolation, device guards and cache allocation.
        This entry verifies source/config/cache geometry, but cannot establish
        available-KV provenance, H200 workers or model/sampler execution. Model
        feedback remains the explicit length-only simulation used by this owner.
        """
        if not isinstance(config, BackendConfig) or not isinstance(scheduler_config, VLLMSchedulerConfig):
            raise TypeError("0.30 bridge requires BackendConfig and VLLMSchedulerConfig")
        if scheduler_config.scheduler_version != "0.30.0":
            raise ValueError("0.30 bridge requires explicit scheduler_version=0.30.0")
        if scheduler_config.context_parallel_size != 1 or scheduler_config.data_parallel_size != 1:
            raise NotImplementedError("prepared native CPU bridge requires CP1/DP1; PCP uses its explicit cache projection")
        ordinary_async = (scheduler_config.async_scheduling
            and config.role == "colocated" and config.scheduler_pipeline_depth == 2
            and scheduler_config.tensor_parallel_size in (1, 2)
            and scheduler_config.pipeline_parallel_size == 1
            and (not scheduler_config.enable_expert_parallel or scheduler_config.tensor_parallel_size == 2)
            and scheduler_config.mtp_num_speculative_tokens == 0)
        if ((config.role not in ("colocated", "prefill", "decode")
                and worker_role_operator_group(config.role) != "attention")
                or (not ordinary_async and config.scheduler_pipeline_depth != 1)
                or (scheduler_config.async_scheduling and not ordinary_async)
                or (scheduler_config.enable_prefix_caching
                    and scheduler_config.prefix_cache_mode != "trace_no_hit"
                    and scheduler_config.simple_cpu_offload_bytes_per_rank is None)
                or scheduler_config.dtype not in ("bfloat16", "float16")):
            raise NotImplementedError("0.30 prepared bridge requires synchronous depth1 or ordinary TP1/TP2 async depth2, with only bounded trace_no_hit prefix caching")
        if scheduler_config.dtype == "float16":
            from src.serving.workload.model_config import load_model_config
            fp16_model = load_model_config(scheduler_config.model)
            if (fp16_model.model_type != "qwen3" or fp16_model.attention_layout != "standard_gqa"
                    or fp16_model.num_experts != 0 or fp16_model.compute_dtype != "fp16"
                    or fp16_model.weight_dtype != "fp16" or config.role != "colocated"
                    or scheduler_config.tensor_parallel_size not in (1, 2)
                    or scheduler_config.pipeline_parallel_size != 1 or scheduler_config.enable_expert_parallel
                    or scheduler_config.mtp_num_speculative_tokens != 0 or scheduler_config.async_scheduling
                    or scheduler_config.enable_prefix_caching
                    or scheduler_config.simple_cpu_offload_bytes_per_rank is not None):
                raise NotImplementedError("FP16 native source requires synchronous dense Qwen TP1/2 PP/CP/DP1 EP/MTP-off")
        _validate_vllm030_host_profiles(scheduler_config.scheduler_host_profile,
            scheduler_config.engine_core_host_round_profile)
        from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
        from src.serving.kv_cache.vllm_allocation import (
            normalize_vllm030_model_allocation,
            vllm030_uses_plain_mla_cache,
            vllm030_supports_expert_parallel_cache,
            vllm030_worker_cache_specs,
            validate_vllm030_cache_placement,
        )
        from src.serving.workload.model_config import load_model_config
        identity = vllm030_source_identity(source_root)
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec, KVCacheConfig, UniformTypeKVCacheSpecs
        from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

        expected_scheduler_type = AsyncScheduler if ordinary_async else Scheduler
        if type(scheduler) is not expected_scheduler_type:
            raise TypeError("0.30 bridge requires the matching real upstream Scheduler class")
        if scheduler.get_num_unfinished_requests() != 0:
            raise ValueError("0.30 prepared bridge requires an unused Scheduler")
        native = scheduler.vllm_config
        if native.device_config.device_type != "cpu":
            raise NotImplementedError("0.30 prepared bridge currently requires a guarded CPU process")
        model = load_model_config(scheduler_config.model)
        effective_dtype = str(native.model_config.dtype)
        expected_dtype = {"bfloat16": "torch.bfloat16", "float16": "torch.float16"}[scheduler_config.dtype]
        if effective_dtype != expected_dtype:
            raise ValueError("native model dtype disagrees with requested Scheduler dtype")
        if scheduler_config.dtype == "float16" and not native.model_config.enforce_eager:
            raise NotImplementedError("FP16 native source requires explicit eager ModelConfig")
        _validate_vllm030_dtype_scope(model, scheduler_config, config.role)
        if ordinary_async and not ((model.model_type in ("qwen3", "llama") and not model.num_experts
                and not scheduler_config.enable_expert_parallel)
                or (scheduler_config.tensor_parallel_size == 2 and scheduler_config.enable_expert_parallel
                    and is_validated_qwen3_moe_ep2_model_scope(model))):
            raise NotImplementedError("0.30 async bridge requires ordinary dense Qwen3/Llama or validated Qwen3MoE TP=EP2")
        glm_pd_source = (model.model_type == "glm_moe_dsa"
            and model.model_id in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3")
            and model.num_attention_heads % scheduler_config.tensor_parallel_size == 0
            and (scheduler_config.mtp_num_speculative_tokens == 0
                or (config.role == "decode" and scheduler_config.pipeline_parallel_size == 1
                    and not scheduler_config.enable_expert_parallel)))
        kimi_pd_source = (model.model_id == "moonshotai/Kimi-K2.6"
            and scheduler_config.enable_expert_parallel
            and vllm030_supports_expert_parallel_cache(model, scheduler_config.tensor_parallel_size)
            and scheduler_config.mtp_num_speculative_tokens == 0)
        llama_pd_source = (model.model_type == "llama"
            and model.attention_layout == "standard_gqa"
            and scheduler_config.tensor_parallel_size == 1
            and scheduler_config.pipeline_parallel_size == 1
            and scheduler_config.context_parallel_size == 1
            and scheduler_config.data_parallel_size == 1
            and not scheduler_config.enable_expert_parallel
            and scheduler_config.mtp_num_speculative_tokens == 0)
        if (config.role in ("prefill", "decode")
                and model.model_type not in ("qwen3", "qwen3_moe")
                and not (llama_pd_source or glm_pd_source or kimi_pd_source)):
            raise NotImplementedError("0.30 metadata PD bridge requires Qwen3, ordinary Llama TP1, GLM or K2.6 supported pools")
        if worker_role_operator_group(config.role) == "attention" and (
                model.model_type not in ("qwen3", "llama", "qwen3_moe")
                or scheduler_config.tensor_parallel_size not in ((1, 2) if config.role == "attention" else (1,))
                or scheduler_config.enable_expert_parallel):
            raise NotImplementedError("0.30 AF metadata bridge requires ordinary Qwen3/Qwen3MoE/Llama TP1/2 attention")
        if ((model.model_id not in ("zai-org/GLM-5.2", "zai-org/GLM-5.2-FP8", "zai-org/GLM-5.3",
                                   "deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro",
                                   "moonshotai/Kimi-K2.6", "moonshotai/Kimi-K3")
                and model.model_type not in ("qwen3", "qwen3_moe", "llama")
                and not vllm030_uses_plain_mla_cache(model))
                or load_model_config(native.model_config.model).config_sha256 != model.config_sha256):
            raise ValueError("0.30 bridge requires matching accurate source model configs")
        if scheduler_config.enable_expert_parallel and not vllm030_supports_expert_parallel_cache(
                model, scheduler_config.tensor_parallel_size):
            raise NotImplementedError("0.30 prepared bridge requires Qwen3MoE EP2 or GLM/K2.6 EP8 cache scope")
        if scheduler_config.mtp_num_speculative_tokens:
            from src.serving.kv_cache.vllm_allocation import validate_vllm030_glm_mtp_config, validate_vllm030_dspark_config
            if scheduler_config.speculative_method == "dspark":
                validate_vllm030_dspark_config(native, model)
                draft = load_model_config(scheduler_config.dspark_draft_model)
                if load_model_config(native.speculative_config.draft_model_config.model).config_sha256 != draft.config_sha256:
                    raise ValueError("DSpark native draft differs from its declared checkpoint")
            else:
                validate_vllm030_glm_mtp_config(native, model)
            legal_role = (config.role == "colocated" or (
                scheduler_config.speculative_method == "mtp" and config.role == "decode"
                and scheduler_config.pipeline_parallel_size == 1))
            if (not legal_role or scheduler_config.pipeline_parallel_size > model.num_hidden_layers
                    or (scheduler_config.speculative_method == "dspark" and scheduler_config.mtp_num_speculative_tokens != 5)
                    or scheduler_config.mtp_acceptance_policy not in ("fixed_count", "maximum_entropy")
                    or native.speculative_config.num_speculative_tokens != scheduler_config.mtp_num_speculative_tokens):
                raise NotImplementedError("0.30 MTP bridge requires GLM sequential whole-shard TP colocated/nonempty PP or PD decode/PP1 and explicit acceptance")
        elif native.speculative_config is not None:
            raise ValueError("0.30 native Scheduler unexpectedly enables speculation")
        if (native.parallel_config.pipeline_parallel_size != scheduler_config.pipeline_parallel_size
                or native.parallel_config.tensor_parallel_size != scheduler_config.tensor_parallel_size
                or native.parallel_config.enable_expert_parallel != scheduler_config.enable_expert_parallel):
            raise ValueError("0.30 native Scheduler mode disagrees with bridge scope")
        expected_cpu_bytes = scheduler_config.simple_cpu_offload_bytes_per_rank
        native_transfer = native.kv_transfer_config
        if expected_cpu_bytes is None:
            if native_transfer is not None:
                raise ValueError("0.30 bridge requires native KV transfer to be disabled")
        else:
            if (native_transfer is None
                    or native_transfer.kv_connector != "SimpleCPUOffloadConnector"
                    or native_transfer.kv_role != "kv_both"
                    or native_transfer.kv_buffer_device != "cpu"
                    or native_transfer.kv_load_failure_policy != "fail"
                    or native_transfer.kv_connector_extra_config != {
                        "cpu_bytes_to_use": expected_cpu_bytes * native.parallel_config.world_size,
                        "cpu_bytes_to_use_per_rank": expected_cpu_bytes,
                        "lazy_offload": False,
                        "kv_offload_backend": "cpu",
                    }):
                raise ValueError("0.30 native SimpleCPU offload disagrees with bridge declaration")
        for actual, expected, name in (
                (scheduler.max_num_scheduled_tokens, config.max_batched_tokens, "scheduled batch tokens"),
                (native.scheduler_config.max_num_seqs, config.max_requests, "request limit"),
                (native.model_config.max_model_len, scheduler_config.max_model_len, "model length"),
                (native.scheduler_config.enable_chunked_prefill, scheduler_config.enable_chunked_prefill, "chunked prefill"),
                (native.scheduler_config.async_scheduling, scheduler_config.async_scheduling, "async scheduling"),
                (native.cache_config.enable_prefix_caching, scheduler_config.enable_prefix_caching, "prefix caching"),
                (native.cache_config.gpu_memory_utilization, scheduler_config.gpu_memory_utilization, "memory utilization"),
                (scheduler.block_size, config.kv_block_size_tokens, "block size"),
                (scheduler.kv_cache_config.num_blocks, config.num_kv_blocks, "block count")):
            if actual != expected:
                raise ValueError(f"0.30 effective {name} disagrees with bridge declaration")
        if native.cache_config.num_gpu_blocks_override is not None:
            raise ValueError("0.30 bridge rejects debug block overrides")
        if scheduler_config.pipeline_parallel_size > 1:
            from vllm.distributed.utils import get_pp_indices
            expected_ranges = (pipeline_layer_ranges(model.num_hidden_layers, scheduler_config.pipeline_parallel_size,
                scheduler_config.pipeline_layer_partition) if scheduler_config.pipeline_layer_partition is not None else
                pipeline_layer_ranges_for_model(model, scheduler_config.pipeline_parallel_size))
            actual_ranges = tuple(get_pp_indices(model.num_hidden_layers, stage, scheduler_config.pipeline_parallel_size)
                for stage in range(scheduler_config.pipeline_parallel_size))
            if actual_ranges != expected_ranges:
                raise ValueError("native PP stage cache ownership disagrees with its declared source partition")
        rank_specs = vllm030_worker_cache_specs(native, model)
        expected_specs = rank_specs[0]
        if (not isinstance(worker_kv_configs, tuple)
                or len(worker_kv_configs) != len(rank_specs)
                or any(not isinstance(worker, KVCacheConfig) for worker in worker_kv_configs)):
            raise TypeError("0.30 bridge requires the complete native worker cache configurations")
        arena_bytes_by_rank = []
        for rank, (worker, expected) in enumerate(zip(worker_kv_configs, rank_specs, strict=True)):
            if worker.num_blocks != config.num_kv_blocks:
                raise ValueError("0.30 native worker block count disagrees with bridge")
            arena_bytes_by_rank.append(validate_vllm030_cache_placement(native, worker,
                model=model if scheduler_config.dspark_draft_model is not None else None))
            tp = scheduler_config.tensor_parallel_size
            if worker.kv_cache_groups != worker_kv_configs[rank // tp * tp].kv_cache_groups:
                raise ValueError("0.30 PP1 worker cache groups disagree across TP ranks")
            actual_specs = {}
            for group in worker.kv_cache_groups:
                if not group.layer_names:
                    members = {}
                elif isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
                    members = group.kv_cache_spec.kv_cache_specs
                else:
                    members = {name: group.kv_cache_spec for name in group.layer_names}
                if set(members) != set(group.layer_names) or set(actual_specs).intersection(members):
                    raise ValueError("0.30 cache group has inconsistent or duplicate owners: "
                        f"rank={rank}, layers={group.layer_names}, members={tuple(members)}")
                actual_specs.update(members)
            if actual_specs != expected:
                raise ValueError("0.30 native worker cache specs disagree with the complete source-projected cache map")
        if scheduler.kv_cache_config != generate_scheduler_kv_cache_config(list(worker_kv_configs)):
            raise ValueError("0.30 Scheduler cache representative disagrees with native worker configs")
        allocation = normalize_vllm030_model_allocation(
            native, model, worker_kv_configs, scheduler.kv_cache_config,
            source_root=source_root)
        page_bytes = max(sum(spec.page_size_bytes for spec in specs.values()) for specs in rank_specs)
        pool_row_bytes_by_rank = None
        if model.model_type == "deepseek_v4":
            reporting_rate = model.kv_bytes_per_token_per_parallel_rank(scheduler_config.tensor_parallel_size, 1)
            if config.kv_bytes_per_token != -(-reporting_rate.numerator // reporting_rate.denominator):
                raise ValueError("0.30 V4 bridge nominal KV reporting bytes disagree with model geometry")
            # All groups alias the same physical pool. A live block in any
            # group reserves the complete pool row, including padding.
            if any(arena % allocation.num_kv_blocks for arena in arena_bytes_by_rank):
                raise ValueError("0.30 V4 arena does not have integral physical pool rows")
            pool_row_bytes_by_rank = tuple(arena // allocation.num_kv_blocks for arena in arena_bytes_by_rank)
            tp = scheduler_config.tensor_parallel_size
            if any(size != pool_row_bytes_by_rank[rank // tp * tp]
                    for rank, size in enumerate(pool_row_bytes_by_rank)):
                raise ValueError("0.30 V4 physical pool rows disagree across TP ranks")
            # Scheduler block IDs are shared by all stages. The aggregate
            # capacity ledger has no rank axis: use a conservative maximum
            # stage row, while retaining every actual worker arena/row below.
            row_bytes = max(pool_row_bytes_by_rank)
        elif model.model_id != "moonshotai/Kimi-K3" and page_bytes != config.kv_bytes_per_token * config.kv_block_size_tokens:
            raise ValueError("0.30 bridge KV bytes disagree with source cache geometry")
        # Upstream deliberately reduces each uniform worker group to its first
        # spec for Scheduler block management. Traffic must retain the full
        # worker group page, including all main and indexer states.
        geometry = tuple((group.kv_cache_spec.block_size,
                          row_bytes if model.model_type == "deepseek_v4" else
                          sum(expected_specs[name].page_size_bytes for name in group.layer_names)
                          if type(group.kv_cache_spec) in (FullAttentionSpec, MLAAttentionSpec) else group.kv_cache_spec.page_size_bytes)
                         for group in worker_kv_configs[0].kv_cache_groups)
        single_group = all(len(worker.kv_cache_groups) == 1 for worker in worker_kv_configs)
        if single_group:
            if (len(geometry) != 1 or any(len(worker.kv_cache_groups) != 1
                    or worker.kv_cache_groups[0].kv_cache_spec.block_size != geometry[0][0] for worker in worker_kv_configs)
                    or any(arena % allocation.num_kv_blocks for arena in arena_bytes_by_rank)):
                raise ValueError("0.30 single-group cache requires a common lifecycle and integral pool rows")
            pool_row_bytes_by_rank = tuple(arena // allocation.num_kv_blocks for arena in arena_bytes_by_rank)
            # The native pool has common block IDs but PP workers can have
            # different row widths. Its rank-free capacity ledger reports a
            # maximum-stage physical row; the receipt keeps actual arenas.
            geometry = ((geometry[0][0], max(pool_row_bytes_by_rank)),)
        k3_cache_groups = None
        if model.model_id == "moonshotai/Kimi-K3":
            from vllm.v1.kv_cache_interface import MambaSpec
            from src.serving.kv_cache.vllm_allocation import vllm030_model_cache_specs
            source_specs = vllm030_model_cache_specs(native, model)
            kda_group_by_layer, mla_groups = {}, set()
            for worker in worker_kv_configs:
                for group_index, group in enumerate(worker.kv_cache_groups):
                    for name in group.layer_names:
                        if isinstance(source_specs[name], MambaSpec):
                            layer = int(name.split(".")[3])
                            if layer in kda_group_by_layer and kda_group_by_layer[layer] != group_index:
                                raise ValueError("K3 workers disagree on a KDA layer's cache group")
                            kda_group_by_layer[layer] = group_index
                        else:
                            mla_groups.add(group_index)
            if len(kda_group_by_layer) != 69 or len(mla_groups) != 1:
                raise ValueError("K3 workers require 69 KDA groups and one shared MLA group")
            k3_cache_groups = (kda_group_by_layer, mla_groups.pop())
        control_plane = _NativeVLLMControlPlane(
            config, scheduler_config, prepared_scheduler=scheduler,
            prepared_kv_geometry=geometry, prepared_model=model, prepared_k3_cache_groups=k3_cache_groups)
        if scheduler_config.speculative_method == "dspark":
            draft = load_model_config(scheduler_config.dspark_draft_model)
            final_worker = worker_kv_configs[-scheduler_config.tensor_parallel_size]
            memberships = {name: index for index, group in enumerate(final_worker.kv_cache_groups) for name in group.layer_names}
            first_memberships = {name: index for index, group in enumerate(worker_kv_configs[0].kv_cache_groups) for name in group.layer_names}
            control_plane._dspark_cache_metadata = (
                draft, tuple(group.kv_cache_spec.block_size for group in final_worker.kv_cache_groups),
                first_memberships['model.layers.0.attn.swa_cache'],
                tuple(memberships[f'model.layers.{layer}.attn.swa_cache'] for layer in draft.dspark_layer_indices))
        control_plane.runtime_info = {
            **identity, "platform": "guarded CPU, not H200 runtime", "model_config_sha256": model.config_sha256,
            **({'k3_source_capacities':control_plane._k3_source_capacities()} if model.model_id=='moonshotai/Kimi-K3' else {}),
            "source_model_type": model.model_type,
            "requested_dtype": scheduler_config.dtype,
            "effective_dtype": str(native.model_config.dtype).removeprefix("torch."),
            "capacity_source": "caller-prepared native allocation; capacity provenance unverified",
            "cache_specs_source": "Midlayer source projection, not constructed model layers",
            "cache_geometry_source": "complete worker groups; Scheduler representative does not carry aggregate traffic",
            "kv_reporting_bytes_semantics": "maximum_stage_rate; every stage's actual arena remains in native receipt",
            "scheduler_cache_representative_pipeline_rank": 0,
            "cache_placement_source_validated": True,
            "native_allocation": asdict(allocation),
            "kv_group_geometry": geometry,
            "kv_capacity_event_bytes_semantics": ("maximum_stage_physical_pool_row; aggregate ledger, not per-rank usage"
                if model.model_type == "deepseek_v4" or single_group else "worker_group_page_bytes"),
            "physical_pool_row_bytes_by_rank": pool_row_bytes_by_rank,
            "shared_cache_arena_bytes_by_rank": tuple(arena_bytes_by_rank),
            "model_feedback_source": "synthetic length-only simulator token feedback; not model sampling",
            "requested_dtype": scheduler_config.dtype,
            "cache_dtype": native.cache_config.cache_dtype,
            "effective_chunked_prefill": native.scheduler_config.enable_chunked_prefill,
            "requested_prefix_caching": scheduler_config.enable_prefix_caching,
            "effective_prefix_caching": native.cache_config.enable_prefix_caching,
            "prefix_cache_mode": scheduler_config.prefix_cache_mode,
            "effective_tensor_parallel_size": native.parallel_config.tensor_parallel_size,
            "effective_pipeline_parallel_size": native.parallel_config.pipeline_parallel_size,
            "effective_expert_parallel": native.parallel_config.enable_expert_parallel,
            "effective_sequence_parallel_moe": native.parallel_config.use_sequence_parallel_moe,
            "effective_max_num_batched_tokens": native.scheduler_config.max_num_batched_tokens,
            "effective_max_num_scheduled_tokens": scheduler.max_num_scheduled_tokens,
            "effective_cache_block_size": native.cache_config.block_size,
            "normal_capacity_accepted": False, "runtime_acceptance": False,
            "mtp_scenario": mtp_scenario_manifest(scheduler_config),
            "speculative_method": "disabled" if native.speculative_config is None else native.speculative_config.method,
            "mtp_num_speculative_tokens": 0 if native.speculative_config is None else native.speculative_config.num_speculative_tokens,
            "mtp_draft_architecture": (None if native.speculative_config is None else
                native.speculative_config.draft_model_config.hf_config.architectures[0]),
        }
        if model.model_id == "moonshotai/Kimi-K3":
            from src.serving.kv_cache.vllm_allocation import vllm030_kimi_k3_cache_storage_manifest
            control_plane.runtime_info["k3_native_cache_storage"] = vllm030_kimi_k3_cache_storage_manifest(
                native, model, worker_kv_configs)
        if scheduler_config.speculative_method == 'dspark':
            _,sizes,target_group,draft_groups=control_plane._dspark_cache_metadata
            control_plane.runtime_info.update(dspark_draft_config_sha256=draft.config_sha256,
                dspark_cache_geometry={'block_sizes':sizes,'target_swa_group':target_group,'draft_layer_groups':draft_groups},
                dspark_draft_layer_indices=draft.dspark_layer_indices,
                dspark_new_drafting_slots=native.speculative_config.max_num_new_slots_for_drafting,
                dspark_target_verification_rows=1+scheduler_config.mtp_num_speculative_tokens,
                dspark_query_rows=draft.dspark_block_size)
        if scheduler_config.pipeline_parallel_size > 1:
            control_plane.runtime_info.update(
                effective_pipeline_layer_partition=tuple(end - first for first, end in actual_ranges),
                effective_pipeline_layer_ranges=actual_ranges,
                native_worker_cache_owners_by_rank=tuple(tuple(specs) for specs in rank_specs))
        return cls(config, control_plane)

    @classmethod
    def from_vllm030_capacity_plan(
        cls, capacity_plan, scheduler_config: VLLMSchedulerConfig, *,
        python_executable: Path, source_root: Path, max_batched_tokens: int, max_requests: int,
        supported_layouts_by_rank: tuple[tuple[str, ...], ...],
        prepared_process=None,
    ) -> "VLLMBackend":
        """Bind an analytical normal-capacity plan to its real 0.30 process.

        Reconcile actual returned allocation facts with the planner before
        admitting any request. This does not upgrade policy/reserve provenance
        or CPU control-plane execution to H200 runtime acceptance.
        """
        from src.serving.kv_cache.kv_capacity import KVCapacityPlan
        from src.serving.workload.model_config import load_model_config, is_qwen3_moe_source_model_scope, is_kimi_k3_data_parallel_model_scope, is_deepseek_v4_data_parallel_model_scope
        from src.serving.parallelism.parallel_config import ParallelConfig
        from src.serving.kv_cache.vllm_allocation import vllm030_supports_expert_parallel_cache

        if not isinstance(capacity_plan, KVCapacityPlan) or not isinstance(scheduler_config, VLLMSchedulerConfig):
            raise TypeError("0.30 analytical backend requires a capacity plan and Scheduler config")
        if (capacity_plan.source_mode != "analytical" or capacity_plan.native_allocation is None
                or capacity_plan.native_allocation.version != "0.30.0"
                or scheduler_config.scheduler_version != "0.30.0"):
            raise ValueError("0.30 analytical backend requires native0.30 analytical facts, not debug capacity")
        model = load_model_config(scheduler_config.model)
        if prepared_process is not None and (model.attention_layout != "standard_gqa"
                or capacity_plan.worker_role != "colocated" or scheduler_config.mtp_num_speculative_tokens
                or scheduler_config.tensor_parallel_size not in (1, 2)
                or any(value != 1 for value in (scheduler_config.pipeline_parallel_size,
                    scheduler_config.context_parallel_size, scheduler_config.data_parallel_size))):
            raise ValueError("allocator Scheduler handoff requires ordinary colocated GQA TP1/2 CP1/DP1/PP1")
        global_qwen_ep_cache = (is_qwen3_moe_source_model_scope(model)
            and scheduler_config.data_parallel_size > 1 and scheduler_config.context_parallel_size == 1)
        global_k3_ep_cache = (scheduler_config.data_parallel_size > 1
            and is_kimi_k3_data_parallel_model_scope(model, ParallelConfig(
                tensor_parallel_size=scheduler_config.tensor_parallel_size,
                data_parallel_size=scheduler_config.data_parallel_size,
                context_parallel_size=scheduler_config.context_parallel_size,
                pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
                enable_expert_parallel=scheduler_config.enable_expert_parallel)))
        global_v4_ep_cache = (scheduler_config.data_parallel_size > 1
            and is_deepseek_v4_data_parallel_model_scope(model, ParallelConfig(
                tensor_parallel_size=scheduler_config.tensor_parallel_size, data_parallel_size=scheduler_config.data_parallel_size,
                context_parallel_size=scheduler_config.context_parallel_size, pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
                enable_expert_parallel=scheduler_config.enable_expert_parallel)))
        expected_ep = (scheduler_config.tensor_parallel_size * scheduler_config.context_parallel_size * scheduler_config.data_parallel_size
            if scheduler_config.enable_expert_parallel else 1)
        if (capacity_plan.model_config_sha256 != model.config_sha256
                or capacity_plan.tensor_parallel_size != scheduler_config.tensor_parallel_size
                or capacity_plan.pipeline_parallel_size != scheduler_config.pipeline_parallel_size
                or capacity_plan.pipeline_layer_partition != scheduler_config.pipeline_layer_partition
                or capacity_plan.context_parallel_size != scheduler_config.context_parallel_size
                or capacity_plan.data_parallel_size != scheduler_config.data_parallel_size
                or capacity_plan.expert_parallel_size != expected_ep
                or (capacity_plan.worker_role not in ("colocated", "prefill", "decode")
                    and worker_role_operator_group(capacity_plan.worker_role) != "attention")
                or (scheduler_config.enable_expert_parallel and not (vllm030_supports_expert_parallel_cache(
                    model, scheduler_config.tensor_parallel_size)
                    or global_qwen_ep_cache or global_k3_ep_cache or global_v4_ep_cache
                    or ((scheduler_config.context_parallel_size > 1 or scheduler_config.data_parallel_size > 1)
                        and model.model_type == "glm_moe_dsa")))):
            raise ValueError("0.30 analytical backend model/parallelism disagrees with capacity plan")
        if capacity_plan.data_parallel_size > 1:
            local_rank_count = (capacity_plan.tensor_parallel_size * capacity_plan.pipeline_parallel_size
                * capacity_plan.context_parallel_size)
            if len(supported_layouts_by_rank) != local_rank_count * capacity_plan.data_parallel_size:
                raise ValueError("coordinated normal capacity requires complete ordered DP layouts")
            replicas = []
            try:
                for replica, receipt in enumerate(capacity_plan.native_allocations):
                    start, end = replica * local_rank_count, (replica + 1) * local_rank_count
                    local_settings = replace(scheduler_config, data_parallel_size=1)
                    budgets = tuple(budget.available_kv_bytes for budget in capacity_plan.rank_budgets[start:end])
                    layouts = supported_layouts_by_rank[start:end]
                    backend = cls.from_vllm030_byte_budget(local_settings,
                        python_executable=python_executable, source_root=source_root,
                        pool_name=f"{capacity_plan.pool_name}:dp{replica}", role=capacity_plan.worker_role,
                        max_batched_tokens=max_batched_tokens, max_requests=max_requests,
                        kv_block_size_tokens=capacity_plan.kv_block_size_tokens,
                        available_memory_by_rank=budgets, supported_layouts_by_rank=layouts)
                    replicas.append(backend)
                    if (backend.config.max_batched_tokens != max_batched_tokens or backend.config.max_requests != max_requests
                            or backend.runtime_info["native_allocation"] != asdict(receipt)):
                        raise ValueError("DP native allocator receipt disagrees with its bound analytical rank budgets")
                    # The coordinator requires the original local source
                    # declaration as well as the genuine native receipt.
                    # Already bound PCP children retain their raw/replicated
                    # arena distinction; unbound CP1 children use the same
                    # normalization as the ordinary byte-budget DP entry.
                    if "source_scheduler_config" not in backend.runtime_info:
                        _bind_vllm030_source_cache_projection(backend, local_settings, budgets, layouts)
                return cls.from_data_parallel_replicas(tuple(replicas), pool_name=capacity_plan.pool_name,
                    source_scheduler_config=scheduler_config if scheduler_config.data_parallel_size > 1 else None)
            except BaseException as primary_error:
                cleanup_errors = []
                for backend in replicas:
                    try:
                        backend.close()
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                if cleanup_errors:
                    raise primary_error from cleanup_errors[0]
                raise
        arguments = dict(python_executable=python_executable, source_root=source_root,
            pool_name=capacity_plan.pool_name, max_batched_tokens=max_batched_tokens, max_requests=max_requests,
            role=capacity_plan.worker_role, kv_block_size_tokens=capacity_plan.kv_block_size_tokens,
            available_memory_by_rank=tuple(budget.available_kv_bytes for budget in capacity_plan.rank_budgets),
            supported_layouts_by_rank=supported_layouts_by_rank)
        if prepared_process is None:
            backend = cls.from_vllm030_byte_budget(scheduler_config, **arguments)
        else:
            from src.serving.backend.vllm_scheduler_process import VLLM030SchedulerProcess
            if type(prepared_process) is not VLLM030SchedulerProcess:
                raise TypeError("allocator handoff requires its actual native Scheduler process")
            prepared_process.claim_initial_state(scheduler_config, **arguments)
            backend = cls(prepared_process.backend_config, prepared_process)
        if (backend.config.max_batched_tokens != max_batched_tokens or backend.config.max_requests != max_requests
                or backend.runtime_info["native_allocation"] != asdict(capacity_plan.native_allocation)):
            backend.close()
            raise ValueError("0.30 effective allocation or limits disagree with analytical capacity plan")
        return backend

    @classmethod
    def from_vllm030_byte_budget(
        cls, scheduler_config: VLLMSchedulerConfig, *, python_executable: Path,
        source_root: Path, pool_name: str, max_batched_tokens: int, max_requests: int,
        kv_block_size_tokens: int, available_memory_by_rank: tuple[int, ...],
        supported_layouts_by_rank: tuple[tuple[str, ...], ...], role: str = "colocated",
    ) -> "VLLMBackend":
        """Start isolated 0.30 control-plane simulation on declared KV budgets.

        This is distinct from the normal-capacity ``from_model`` path. It
        records upstream CPU rewrites and explicit synthetic model feedback;
        the inputs do not establish H200 memory, provider or model execution.
        """
        from src.serving.backend.vllm_scheduler_process import VLLM030SchedulerProcess
        if scheduler_config.data_parallel_size > 1:
            from src.serving.workload.model_config import (load_model_config, is_qwen3_moe_source_model_scope,
                is_glm_mtp_data_parallel_model_scope, is_glm_fp8_data_parallel_model_scope, is_deepseek_v3_data_parallel_model_scope,
                is_deepseek_v4_data_parallel_model_scope, is_kimi_k2_data_parallel_model_scope, is_kimi_k3_data_parallel_model_scope)
            from src.serving.parallelism.parallel_config import ParallelConfig
            model = load_model_config(scheduler_config.model)
            dp, cp, tp = scheduler_config.data_parallel_size, scheduler_config.context_parallel_size, scheduler_config.tensor_parallel_size
            qwen_moe = is_qwen3_moe_source_model_scope(model) and cp == 1
            target_parallel = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
                data_parallel_size=dp, context_parallel_size=cp, enable_expert_parallel=scheduler_config.enable_expert_parallel)
            deepseek_target = (not scheduler_config.mtp_num_speculative_tokens
                and any(predicate(model, target_parallel) for predicate in (
                    is_deepseek_v3_data_parallel_model_scope, is_deepseek_v4_data_parallel_model_scope, is_kimi_k2_data_parallel_model_scope)))
            k3_target = (not scheduler_config.mtp_num_speculative_tokens
                and is_kimi_k3_data_parallel_model_scope(model, target_parallel))
            dspark = (scheduler_config.speculative_method == "dspark" and role == "colocated"
                and is_deepseek_v4_data_parallel_model_scope(model, target_parallel))
            fp8_mtp = (scheduler_config.mtp_num_speculative_tokens > 0 and model.weight_dtype == "fp8"
                and is_glm_mtp_data_parallel_model_scope(model, ParallelConfig(tensor_parallel_size=tp,
                    pipeline_parallel_size=scheduler_config.pipeline_parallel_size, data_parallel_size=dp,
                    context_parallel_size=cp, enable_expert_parallel=scheduler_config.enable_expert_parallel,
                    sequence_parallel_size=tp if scheduler_config.enable_expert_parallel else 1)))
            fp8_target = (not scheduler_config.mtp_num_speculative_tokens and role == "colocated"
                and is_glm_fp8_data_parallel_model_scope(model, target_parallel))
            if not ((model.model_type == "glm_moe_dsa" and model.weight_dtype == "bf16") or qwen_moe or fp8_mtp or fp8_target or deepseek_target or k3_target or dspark):
                raise NotImplementedError("coordinated CPU DP requires supported GLM, ordinary Qwen3MoE or whole-shard V3/V4 target source")
            if scheduler_config.mtp_num_speculative_tokens and model.model_type != "glm_moe_dsa" and not dspark:
                raise NotImplementedError("coordinated CPU MTP DP requires its explicit BF16 GLM source")
            width = cp * tp * scheduler_config.pipeline_parallel_size
            if (not isinstance(available_memory_by_rank, tuple) or len(available_memory_by_rank) != dp * width
                    or not isinstance(supported_layouts_by_rank, tuple) or len(supported_layouts_by_rank) != dp * width):
                raise ValueError("PCP DP requires complete ordered DP x CP x TP budgets and layouts")
            children = []
            try:
                for replica in range(dp):
                    start, end = replica * width, (replica + 1) * width
                    local_settings = replace(scheduler_config, data_parallel_size=1)
                    child = cls.from_vllm030_byte_budget(local_settings,
                        python_executable=python_executable, source_root=source_root, pool_name=f"{pool_name}:dp{replica}",
                        max_batched_tokens=max_batched_tokens, max_requests=max_requests, kv_block_size_tokens=kv_block_size_tokens,
                        available_memory_by_rank=available_memory_by_rank[start:end],
                        supported_layouts_by_rank=supported_layouts_by_rank[start:end], role=role)
                    children.append(child)
                    if qwen_moe or scheduler_config.mtp_num_speculative_tokens or deepseek_target or k3_target or fp8_target:
                        _bind_vllm030_source_cache_projection(child, local_settings,
                            available_memory_by_rank[start:end], supported_layouts_by_rank[start:end])
                return cls.from_data_parallel_replicas(tuple(children), pool_name=pool_name,
                    source_scheduler_config=scheduler_config)
            except BaseException as primary_error:
                cleanup_errors = []
                for child in children:
                    try:
                        child.close()
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                if cleanup_errors:
                    raise primary_error from cleanup_errors[0]
                raise
        if scheduler_config.context_parallel_size > 1 or (
                scheduler_config.enable_expert_parallel and scheduler_config.tensor_parallel_size == 1):
            from src.serving.workload.model_config import (load_model_config, is_qwen3_moe_source_model_scope,
                is_deepseek_v3_parallel_model_scope, is_deepseek_v4_parallel_model_scope, is_kimi_k2_parallel_model_scope, is_glm_mtp_tensor_model_scope, is_kimi_k3_parallel_model_scope,
                is_glm_fp8_tensor_model_scope, GLM_BLOCK_FP8_CONFIG_SHA256S)
            model = load_model_config(scheduler_config.model)
            cp, tp = scheduler_config.context_parallel_size, scheduler_config.tensor_parallel_size
            pp = scheduler_config.pipeline_parallel_size
            qwen_cache_projection = cp == tp == 1 and is_qwen3_moe_source_model_scope(model)
            deepseek_cache_projection = (cp == tp == 1 and not scheduler_config.mtp_num_speculative_tokens
                and any(predicate(model, tp, pp) for predicate in (
                    is_deepseek_v3_parallel_model_scope, is_deepseek_v4_parallel_model_scope, is_kimi_k2_parallel_model_scope, is_kimi_k3_parallel_model_scope)))
            fp8_mtp_cache_projection = (cp == tp == 1 and scheduler_config.mtp_num_speculative_tokens == 2
                and model.weight_dtype == "fp8" and is_glm_mtp_tensor_model_scope(model, 1))
            fp8_target_cache_projection = (cp == tp == 1 and not scheduler_config.mtp_num_speculative_tokens
                and role == "colocated" and model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S
                and 1 <= pp <= model.num_hidden_layers and is_glm_fp8_tensor_model_scope(model, 1))
            dspark_cache_projection = (cp == tp == 1 and role == "colocated"
                and scheduler_config.speculative_method == "dspark"
                and is_deepseek_v4_parallel_model_scope(model, tp, pp))
            if not (deepseek_cache_projection or fp8_mtp_cache_projection or fp8_target_cache_projection or dspark_cache_projection) and ((not qwen_cache_projection and model.model_type != "glm_moe_dsa") or model.weight_dtype != "bf16"
                    or model.compute_dtype != "bf16" or model.num_attention_heads % tp
                    or (scheduler_config.context_parallel_size > 1 and scheduler_config.pipeline_parallel_size != 1)
                    or scheduler_config.mtp_num_speculative_tokens and not (cp == tp == 1
                        and model.model_type == "glm_moe_dsa" and scheduler_config.mtp_num_speculative_tokens == 2
                        and scheduler_config.mtp_acceptance_policy == "fixed_count")):
                raise NotImplementedError("PCP cache projection requires BF16 GLM; DP-only EP also permits Qwen3MoE or ordinary V3/V4")
            if (not isinstance(available_memory_by_rank, tuple) or len(available_memory_by_rank) != tp * cp * pp
                    or any(type(value) is not int or value <= 0 for value in available_memory_by_rank)
                    or not isinstance(supported_layouts_by_rank, tuple) or len(supported_layouts_by_rank) != tp * cp * pp
                    or any(layouts != supported_layouts_by_rank[0] for layouts in supported_layouts_by_rank)):
                raise ValueError("PCP CPU projection requires every physical rank's positive budget and equal cache layouts")
            projected = replace(scheduler_config, context_parallel_size=1,
                enable_expert_parallel=scheduler_config.enable_expert_parallel and tp > 1)
            backend = cls.from_vllm030_byte_budget(projected, python_executable=python_executable,
                source_root=source_root, pool_name=pool_name, max_batched_tokens=max_batched_tokens,
                max_requests=max_requests, kv_block_size_tokens=kv_block_size_tokens, role=role,
                available_memory_by_rank=tuple(min(available_memory_by_rank[(stage * cp + c) * tp + lane] for c in range(cp))
                    for stage in range(pp) for lane in range(tp)), supported_layouts_by_rank=supported_layouts_by_rank[:tp * pp])
            try:
                _bind_vllm030_source_cache_projection(backend, scheduler_config,
                    available_memory_by_rank, supported_layouts_by_rank)
            except BaseException as primary_error:
                try:
                    backend.close()
                except BaseException as cleanup_error:
                    raise primary_error from cleanup_error
                raise
            return backend
        control = VLLM030SchedulerProcess(
            scheduler_config, python_executable=python_executable, source_root=source_root,
            pool_name=pool_name, max_batched_tokens=max_batched_tokens,
            role=role,
            max_requests=max_requests, kv_block_size_tokens=kv_block_size_tokens,
            available_memory_by_rank=available_memory_by_rank,
            supported_layouts_by_rank=supported_layouts_by_rank)
        backend = cls(control.backend_config, control)
        try:
            from src.serving.workload.model_config import load_model_config
            model = load_model_config(scheduler_config.model)
            if (model.model_type == "glm_moe_dsa" and model.weight_dtype == "bf16"
                    and not scheduler_config.mtp_num_speculative_tokens):
                _bind_vllm030_source_cache_projection(backend, scheduler_config,
                    available_memory_by_rank, supported_layouts_by_rank)
        except BaseException as primary_error:
            try:
                backend.close()
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        return backend

    @property
    def runtime_info(self) -> Any:
        if not hasattr(self._control_plane, "runtime_info"):
            raise RuntimeError(
                "vLLM runtime provenance is available only for from_model")
        return self._control_plane.runtime_info

    def add_request(self, request: RequestSpec) -> None:
        if request.request_id in self._accepted_request_ids:
            raise ValueError(
                f"duplicate vLLM request ID {request.request_id!r}")
        self._control_plane.add_request(request)
        self._accepted_request_ids.add(request.request_id)

    def _idle_dspark_source_inputs(self,draft):
        """Native allocator geometry without a fabricated scheduled request."""
        from src.serving.workload.runtime_inputs import DSparkReplicaInputs
        info=self.runtime_info
        if (type(info) is not dict or 'dspark_cache_geometry' not in info
                or any(key not in info for key in ('speculative_method','dspark_draft_config_sha256'))
                or info['speculative_method']!='dspark' or info['dspark_draft_config_sha256']!=draft.config_sha256):
            raise ValueError('DSpark idle source requires its genuine native Scheduler/cache owner')
        geometry=info['dspark_cache_geometry']
        if type(geometry) is not dict or set(geometry)!={'block_sizes','target_swa_group','draft_layer_groups'}:
            raise ValueError('DSpark idle native geometry must retain its complete original group contract')
        sizes=tuple(geometry['block_sizes'])
        return DSparkReplicaInputs((),(),(),tuple(() for _ in sizes),sizes,geometry['target_swa_group'],tuple(geometry['draft_layer_groups']),
            self.config.num_kv_blocks,self.config.max_requests,
            'retained_request_state' if self._accepted_request_ids else 'never_admitted_constructor')

    def add_requests(self, requests: tuple[RequestSpec, ...]) -> None:
        if not isinstance(requests, tuple) or any(
                not isinstance(request, RequestSpec) for request in requests):
            raise TypeError(
                "vLLM bulk admission requires a tuple of RequestSpec objects")
        request_ids = tuple(request.request_id for request in requests)
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("duplicate vLLM request ID in admission batch")
        duplicate_ids = self._accepted_request_ids.intersection(request_ids)
        if duplicate_ids:
            raise ValueError(
                f"duplicate vLLM request IDs: {sorted(duplicate_ids)!r}")
        self._control_plane.add_requests(requests)
        self._accepted_request_ids.update(request_ids)

    def retain_pd_kv(self, request_id: str):
        return self._control_plane.retain_pd_kv(request_id)

    def release_pd_kv(self, request_id: str):
        return self._control_plane.release_pd_kv(request_id)

    def reserve_pd_destination(self, request: RequestSpec):
        return self._control_plane.reserve_pd_destination(request)

    def try_reserve_pd_destination(self, request: RequestSpec):
        """Return native block groups, or None while destination KV is full."""
        return self._control_plane.try_reserve_pd_destination(request)

    def plan_step(self, virtual_time_s: float):
        return self._control_plane.plan_step(virtual_time_s)

    def enable_modeled_simple_cpu_offload(self) -> None:
        if not hasattr(self._control_plane, "enable_modeled_simple_cpu_offload"):
            raise NotImplementedError("this vLLM backend has no modeled SimpleCPU connector")
        self._control_plane.enable_modeled_simple_cpu_offload()

    def take_modeled_simple_cpu_releases(self) -> tuple[KVCacheEvent, ...]:
        return self._control_plane.take_modeled_simple_cpu_releases()

    def drain_steps(
        self,
        max_steps: int,
    ) -> tuple[tuple[BackendStep, ...], bool]:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("vLLM tail batch size must be an integer")
        if max_steps < 1:
            raise ValueError("vLLM tail batch size must be positive")
        if hasattr(self._control_plane, "drain_steps"):
            result = self._control_plane.drain_steps(max_steps)
            if (not isinstance(result, tuple) or len(result) != 2
                    or not isinstance(result[0], tuple)
                    or any(not isinstance(step, BackendStep)
                           for step in result[0])
                    or not isinstance(result[1], bool)):
                raise TypeError(
                    "vLLM control plane returned an invalid tail batch")
            if not result[0] and result[1]:
                raise RuntimeError(
                    "vLLM tail batch made no progress while work remains")
            return result

        steps = []
        while len(steps) < max_steps and self.has_work():
            step = self.plan_step(0.0)
            if step is None:
                raise RuntimeError(
                    "vLLM tail has work but plan_step returned no step")
            self.complete_step(step, 0.0)
            steps.append(step)
        return tuple(steps), self.has_work()

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        return self._control_plane.complete_step(
            step,
            virtual_time_s,
            execute_model=False,
            execute_transfers=False,
            sample_tokens=False,
        )

    def scheduler_update_cpu_s(self, step: BackendStep, timing_mode: str) -> float:
        if timing_mode not in ("device_only", "device_plus_launch", "device_plus_launch_plus_scheduler"):
            raise ValueError(f"unsupported Serving timing mode: {timing_mode!r}")
        if timing_mode != "device_plus_launch_plus_scheduler":
            return 0.0
        profile = self._control_plane.scheduler_config.scheduler_host_profile
        if profile is None:
            return _VIRTUAL_SCHEDULER_CPU_S
        return profile.service_time_s(step.schedule, "update")

    def measure_scheduler_service(
        self,
        *,
        sample_count: int,
        requests_per_batch: int,
        prompt_tokens: int,
        output_tokens: int,
        include_step_records: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        """Measure native Scheduler calls in its isolated CPU process.

        This is an independent control-plane microbenchmark hook.  It is not
        used by Forward: returned wall-clock samples are for an external
        profile runner to reduce and version before any service value can
        enter the simulation contract.
        """

        if not hasattr(self._control_plane, "measure_scheduler_service"):
            raise RuntimeError(
                "Scheduler service measurement requires the native vLLM "
                "control-plane process")
        if type(include_step_records) is not bool:
            raise TypeError("include_step_records must be boolean")
        arguments = dict(sample_count=sample_count, requests_per_batch=requests_per_batch,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        expected = {"schedule_ns", "update_ns"}
        counts = {"scheduled_request_count", "prefill_request_count", "decode_request_count"}
        if include_step_records:
            arguments["include_step_records"] = True
            expected.update(counts | {"schedule_start_ns", "schedule_end_ns",
                "scheduler_update_start_ns", "scheduler_update_end_ns"})
        result = self._control_plane.measure_scheduler_service(**arguments)
        if (not isinstance(result, dict) or set(result) != expected):
            raise TypeError("vLLM Scheduler service measurement is malformed")
        for name, samples in result.items():
            if (not isinstance(samples, tuple)
                    or len(samples) != sample_count
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           or value < (0 if name in counts else 1) for value in samples)):
                raise TypeError(
                    f"vLLM Scheduler service {name} samples are invalid")
        if include_step_records:
            for index in range(sample_count):
                if (result["scheduled_request_count"][index] < 1
                        or result["scheduled_request_count"][index] != result["prefill_request_count"][index] + result["decode_request_count"][index]
                        or result["schedule_end_ns"][index] - result["schedule_start_ns"][index] != result["schedule_ns"][index]
                        or result["scheduler_update_end_ns"][index] - result["scheduler_update_start_ns"][index] != result["update_ns"][index]
                        or result["schedule_end_ns"][index] > result["scheduler_update_start_ns"][index]):
                    raise ValueError("Scheduler measurement records disagree with native timings or request counts")
        return result

    def has_work(self) -> bool:
        result = self._control_plane.has_work()
        if not isinstance(result, bool):
            raise TypeError("vLLM control_plane.has_work() must return boolean")
        return result

    def request_queue_snapshot(self) -> dict:
        """Read native queue membership, independently of client concurrency."""
        from src.serving.backend.vllm_scheduler_process import VLLMSchedulerProcess
        if (not isinstance(self._control_plane, VLLMSchedulerProcess)
                or self.runtime_info["gpu_worker_processes_started"] != 0):
            raise NotImplementedError("native queue observation requires an isolated CPU Scheduler process")
        return self._control_plane.request_queue_snapshot()

    def close(self) -> None:
        try:
            if hasattr(self._control_plane, "close"):
                self._control_plane.close()
            if self._in_process_baseline_libraries is not None:
                from src.serving.backend.vllm_scheduler_process import _verify_no_accelerator_access
                _verify_no_accelerator_access(self._in_process_baseline_libraries)
        finally:
            if self._in_process_model_directory is not None:
                self._in_process_model_directory.cleanup()

    def abort_observation(self) -> None:
        """Cancel an explicitly finite CPU observation with unfinished work."""
        from src.serving.backend.vllm_scheduler_process import VLLMSchedulerProcess
        if (not isinstance(self._control_plane, VLLMSchedulerProcess)
                or self.runtime_info["gpu_worker_processes_started"] != 0):
            raise NotImplementedError("censored observations require an isolated CPU Scheduler process")
        self._control_plane.abort_observation()



class _DataParallelControlPlane:
    """One process-local coordinator over disjoint native DP cache managers.

    A group step is published only after every ready replica has submitted its
    actual schedule. Empty replicas contribute no request or fabricated token.
    Model execution may then synchronize their shared expert ranks explicitly.
    """
    def __init__(self, replicas, *, pool_name, source_scheduler_config=None):
        if (not isinstance(replicas, tuple) or len(replicas) < 2
                or any(not isinstance(backend, VLLMBackend) or backend.data_parallel_size != 1 for backend in replicas)
                or len({id(backend) for backend in replicas}) != len(replicas)
                or not isinstance(pool_name, str) or not pool_name):
            raise ValueError("DP coordination requires distinct single-replica vLLM backends and one pool identity")
        first = replicas[0].config
        if first.scheduler_pipeline_depth != 1 or any(replace(backend.config, pool_name=first.pool_name, num_kv_blocks=first.num_kv_blocks) != first
                for backend in replicas):
            raise ValueError("synchronous DP Schedulers require matching per-replica limits and pipeline depth one")
        infos = tuple(backend.runtime_info for backend in replicas)
        if any(info["version"] != "0.30.0" or info["source_identity"] != infos[0]["source_identity"] for info in infos):
            raise ValueError("DP native Schedulers require the same explicit vLLM0.30 source")
        geometry = ("model_config_sha256", "effective_tensor_parallel_size", "effective_pipeline_parallel_size", "effective_expert_parallel")
        if any(tuple(info[key] for key in geometry) != tuple(infos[0][key] for key in geometry) for info in infos):
            raise ValueError("DP native Schedulers require the same model and local TP/PP/cache geometry")
        if source_scheduler_config is None and all("source_scheduler_config" in info for info in infos):
            source_scheduler_config = replace(VLLMSchedulerConfig(**infos[0]["source_scheduler_config"]),
                data_parallel_size=len(replicas))
        if source_scheduler_config is not None:
            if (not isinstance(source_scheduler_config, VLLMSchedulerConfig)
                    or source_scheduler_config.data_parallel_size != len(replicas)
                    or any("source_scheduler_config" not in info or info["source_scheduler_config"] !=
                        asdict(replace(source_scheduler_config, data_parallel_size=1)) for info in infos)
                    or any(info["effective_sequence_parallel_moe"] is not False for info in infos)):
                raise ValueError("PCP DP source config disagrees with its actual Scheduler projections")
            if any("source_model_type" not in info or not isinstance(info["source_model_type"], str)
                    or not info["source_model_type"] or info["source_model_type"] != infos[0]["source_model_type"]
                    for info in infos):
                raise ValueError("DP source model family must be bound and agree across native replicas")
        self._source_scheduler_config = source_scheduler_config
        self._source_model_type = (None if source_scheduler_config is None else
            infos[0]["source_model_type"])
        self.replicas, self.pool_name = replicas, pool_name
        self.backend_config = replace(first, pool_name=pool_name, max_batched_tokens=first.max_batched_tokens * len(replicas),
            max_requests=first.max_requests * len(replicas), num_kv_blocks=min(backend.config.num_kv_blocks for backend in replicas))
        self._owners, self._admitted, self._next_replica = {}, set(), 0
        self._pending, self._step_index, self._closed = None, 0, False

    @property
    def runtime_info(self):
        infos = tuple(backend.runtime_info for backend in self.replicas)
        result = {"version": "0.30.0", "source_identity": infos[0]["source_identity"],
            "model_config_sha256": infos[0]["model_config_sha256"],
            "platform": "coordinated real CPU Schedulers; no GPU model workers",
            "data_parallel_size": len(self.replicas), "replicas": infos,
            "num_kv_blocks_semantics": "minimum_per_replica; each native Scheduler retains its actual independent capacity",
            "request_data_parallel_ranks": dict(self._owners),
            "supported_layouts_by_rank": tuple(layout for info in infos for layout in info["supported_layouts_by_rank"]),
            "model_feedback_source": "explicit request transcripts or synthetic length-only feedback from each native replica",
            "control_plane_scope": "per-DP native Scheduler and KV managers; model TP/EP/SP collectives belong to source forward",
            "normal_capacity_accepted": False, "runtime_acceptance": False,
            "gpu_worker_processes_started": 0, "weights_loaded": False, "model_executed": False}
        if self._source_scheduler_config is not None:
            import hashlib
            import json
            original = asdict(self._source_scheduler_config)
            digest = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            width = (self._source_scheduler_config.tensor_parallel_size * self._source_scheduler_config.context_parallel_size
                * self._source_scheduler_config.pipeline_parallel_size)
            moe_sp = (self._source_scheduler_config.enable_expert_parallel
                and self._source_scheduler_config.tensor_parallel_size > 1
                and (self._source_model_type not in ("deepseek_v4", "kimi_linear") or self._source_scheduler_config.pipeline_parallel_size == 1))
            result.update(source_scheduler_config=original, source_scheduler_config_sha256=digest,
                source_context_parallel_size=self._source_scheduler_config.context_parallel_size,
                effective_context_parallel_size=1,
                source_expert_parallel_size=(self._source_scheduler_config.tensor_parallel_size
                    * self._source_scheduler_config.context_parallel_size * self._source_scheduler_config.data_parallel_size
                    if self._source_scheduler_config.enable_expert_parallel else 1),
                source_sequence_parallel_size=(self._source_scheduler_config.tensor_parallel_size if moe_sp else 1),
                source_model_sequence_parallel=(self._source_scheduler_config.pipeline_parallel_size == 1
                    and moe_sp and self._source_model_type in ("glm_moe_dsa", "deepseek_v3", "deepseek_v4", "kimi_k2", "kimi_linear")),
                source_moe_sequence_parallel=moe_sp,
                effective_sequence_parallel_moe=infos[0]["effective_sequence_parallel_moe"],
                effective_expert_parallel=infos[0]["effective_expert_parallel"],
                effective_expert_parallel_size_per_data_parallel_rank=(self._source_scheduler_config.tensor_parallel_size
                    if infos[0]["effective_expert_parallel"] else 1),
                native_allocation_scope=("per_dp_genuine_cp1_receipts; physical_pcp_replicas_are_source_projection"
                    if self._source_scheduler_config.context_parallel_size > 1 else "per_dp_genuine_cp1_receipts; no_context_replication"),
                native_allocations_by_data_parallel_rank=tuple(info["native_allocation"] for info in infos),
                physical_cache_replicas=tuple({**record, "data_parallel_rank": dp,
                    "physical_local_rank": dp * width + record["physical_local_rank"]}
                    for dp, info in enumerate(infos) for record in info["physical_cache_replicas"]))
            if (self._source_scheduler_config.enable_expert_parallel and self._source_scheduler_config.tensor_parallel_size == 1
                    and self._source_scheduler_config.context_parallel_size > 1):
                result["source_expert_exchange_scope"] = (
                    "analytical_global_expert_dependency_projection; native_modular_provider_not_accepted")
        return result

    def _select_replica(self, request_id):
        if request_id not in self._owners:
            self._owners[request_id] = self._next_replica
            self._next_replica = (self._next_replica + 1) % len(self.replicas)
        return self._owners[request_id]

    def request_data_parallel_rank(self, request_id):
        if request_id not in self._owners:
            raise ValueError("request has no selected DP replica")
        return self._owners[request_id]

    def add_request(self, request):
        self.add_requests((request,))

    def add_requests(self, requests):
        if (not isinstance(requests, tuple) or any(not isinstance(request, RequestSpec) for request in requests)
                or len({request.request_id for request in requests}) != len(requests)
                or self._admitted.intersection(request.request_id for request in requests)):
            raise ValueError("DP admission requires distinct fresh request identities")
        selected = tuple(self._select_replica(request.request_id) for request in requests)
        for index, backend in enumerate(self.replicas):
            batch = tuple(request for request, owner in zip(requests, selected, strict=True) if owner == index)
            if batch:
                backend.add_requests(batch)
        self._admitted.update(request.request_id for request in requests)

    def retain_pd_kv(self, request_id):
        return self.replicas[self.request_data_parallel_rank(request_id)].retain_pd_kv(request_id)

    def release_pd_kv(self, request_id):
        return self.replicas[self.request_data_parallel_rank(request_id)].release_pd_kv(request_id)

    def reserve_pd_destination(self, request):
        return self.replicas[self._select_replica(request.request_id)].reserve_pd_destination(request)

    def try_reserve_pd_destination(self, request):
        return self.replicas[self._select_replica(request.request_id)].try_reserve_pd_destination(request)

    def plan_step(self, virtual_time_s):
        if self._pending is not None:
            raise RuntimeError("synchronous DP step must complete before planning another")
        parts = tuple((index, step) for index, backend in enumerate(self.replicas)
            if (step := backend.plan_step(virtual_time_s)) is not None)
        if not parts:
            return None
        steps = tuple(step for _, step in parts)
        layouts = tuple(step.schedule.attention_kv_layout for step in steps)
        if len(set(layouts)) != 1:
            raise ValueError("DP native Scheduler attention layout must agree across active replicas")
        mtp = (self._source_scheduler_config is not None
            and (self._source_model_type == "glm_moe_dsa" and self._source_scheduler_config.mtp_num_speculative_tokens == 2
                or self._source_model_type == "deepseek_v4" and self._source_scheduler_config.speculative_method == "dspark"
                    and self._source_scheduler_config.mtp_num_speculative_tokens == 5)
            and self._source_scheduler_config.context_parallel_size == 1
            and self._source_scheduler_config.mtp_acceptance_policy == "fixed_count")
        k3 = (self._source_scheduler_config is not None and self._source_model_type == "kimi_linear"
            and self._source_scheduler_config.mtp_num_speculative_tokens == 0)
        dspark = self._source_scheduler_config is not None and self._source_scheduler_config.speculative_method == "dspark"
        if k3:
            from src.serving.workload.runtime_inputs import K3SourceInputs, K3DataParallelSourceInputs
            if any(not isinstance(step.model_source_inputs, K3SourceInputs) for step in steps):
                raise ValueError("coordinated K3 requires each active Scheduler's actual typed hybrid state")
        if dspark:
            from src.serving.workload.runtime_inputs import DSparkSourceInputs
            if any(not isinstance(step.model_source_inputs, DSparkSourceInputs) or len(step.model_source_inputs.replicas) != 1
                    for step in steps):
                raise ValueError("DSpark DP requires each actual Scheduler's source frame")
        if (not (k3 or dspark) and any(step.model_source_inputs is not None for step in steps)
                or len({step.embedding_token_source for step in steps}) != 1
                or not mtp and any(request.mtp_proposer_mode != "disabled" for step in steps for request in step.schedule.requests)):
            raise NotImplementedError("DP coordination requires ordinary eager or explicit BF16 whole-shard TP GLM MTP Scheduler steps")
        sources = tuple(step.embedding_token_ids is not None for step in steps)
        if len(set(sources)) != 1:
            raise ValueError("DP step token transcript availability must agree across active replicas")
        if k3 and not sources[0]:
            raise ValueError("coordinated K3 requires actual or explicitly synthetic Scheduler token IDs")
        if k3:
            states = {owner: step.model_source_inputs for owner, step in parts}
            capacities=tuple(backend.runtime_info['k3_source_capacities'] for backend in self.replicas)
            source_state = K3DataParallelSourceInputs(tuple(states[owner] if owner in states else None
                for owner in range(len(self.replicas))),tuple(value[0] for value in capacities),tuple(value[1] for value in capacities))
        elif dspark:
            draft = steps[0].model_source_inputs.draft_model
            if any(step.model_source_inputs.draft_model.config_sha256 != draft.config_sha256 for step in steps):
                raise ValueError("DSpark DP draft identities disagree across physical replicas")
            states = {owner: step.model_source_inputs.replicas[0] for owner, step in parts}
            idle={}
            for owner,backend in enumerate(self.replicas):
                if owner not in states:
                    idle[owner]=backend._idle_dspark_source_inputs(draft)
            source_state = DSparkSourceInputs(draft, tuple(states[owner] if owner in states else idle[owner]
                for owner in range(len(self.replicas))))
        else:
            source_state = None
        schedule = ScheduleStep(f"{self.pool_name}:dp-step-{self._step_index}", self.pool_name,
            tuple(replace(request, data_parallel_rank=index) for index, step in parts for request in step.schedule.requests),
            finished_request_ids=tuple(value for step in steps for value in step.schedule.finished_request_ids),
            preempted_request_ids=tuple(value for step in steps for value in step.schedule.preempted_request_ids),
            attention_kv_layout=layouts[0])
        aggregate = BackendStep(schedule,
            kv_cache_events=tuple(event for step in steps for event in step.kv_cache_events),
            generated_tokens={request_id: tokens for step in steps for request_id, tokens in step.generated_tokens.items()},
            scheduler_plan_cpu_s=sum(step.scheduler_plan_cpu_s for step in steps),
            engine_submit_host_s=sum(step.engine_submit_host_s for step in steps),
            engine_wake_host_s=sum(step.engine_wake_host_s for step in steps),
            embedding_token_ids=tuple(token for step in steps for token in step.embedding_token_ids) if sources[0] else None,
            embedding_token_source=steps[0].embedding_token_source, model_source_inputs=source_state)
        self._pending = aggregate, parts
        self._step_index += 1
        return aggregate

    def complete_step(self, step, virtual_time_s, *, execute_model, execute_transfers, sample_tokens):
        if any(type(flag) is not bool for flag in (execute_model, execute_transfers, sample_tokens)):
            raise TypeError("DP completion execution flags must be explicit booleans")
        if execute_model or execute_transfers or sample_tokens:
            raise NotImplementedError("DP coordinator only drives native CPU Scheduler feedback")
        if self._pending is None or self._pending[0] is not step:
            raise ValueError("DP completion requires the exact current group step")
        services = tuple(self.replicas[index].complete_step(local, virtual_time_s) for index, local in self._pending[1])
        self._pending = None
        return sum(services)

    def has_work(self):
        return self._pending is not None or any(backend.has_work() for backend in self.replicas)

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for backend in self.replicas:
            try:
                backend.close()
            except Exception as error:
                errors.append(error)
        pending = self._pending is not None
        self._pending = None
        if errors:
            raise errors[0]
        if pending:
            raise RuntimeError("cannot close DP Schedulers with a pending group step")

_EMPTY_NATIVE_PREEMPTION_PROGRESS = object()


def _glm_mtp_first_pass_input_ids(schedule, embedding_ids, *, depth,
                                dummy_token_id, drafter_executes):
    """Emit the first draft frame in the target's source-declared packed order."""
    from src.serving.workload.runtime_inputs import vllm_0_30_source_model_inputs
    from src.serving.workload.traced_sampling import _mtp_retained_last_rows
    packed, packed_ids, _ = vllm_0_30_source_model_inputs(
        schedule, embedding_token_ids=embedding_ids, prompt_lengths=None,
        num_speculative_tokens=depth)
    if not drafter_executes:
        return packed_ids
    first = list(packed_ids[1:]) + [dummy_token_id]
    for index in _mtp_retained_last_rows(packed.requests):
        first[index] = dummy_token_id
    return tuple(first)


class _NativeVLLMControlPlane:
    def __init__(self, config: BackendConfig, scheduler_config: VLLMSchedulerConfig,
                 *, prepared_scheduler: Any = None,
                 prepared_kv_geometry: tuple[tuple[int, int], ...] | None = None,
                 prepared_model: Any = None, prepared_k3_cache_groups: Any = None) -> None:
        self.config = config
        self.scheduler_config = scheduler_config
        self.scheduler_host_profile = scheduler_config.scheduler_host_profile
        self.engine_core_host_round_profile = (
            scheduler_config.engine_core_host_round_profile)
        self._scheduler_host_uncovered_reported: set[str] = set()
        if prepared_scheduler is None:
            self.scheduler, self.vllm_config, dsv4_capacity_context = (
                _build_native_scheduler(config, scheduler_config))
        else:
            if scheduler_config.scheduler_version != "0.30.0":
                raise ValueError("prepared Scheduler requires explicit 0.30 bridge")
            self.scheduler = prepared_scheduler
            self.vllm_config = prepared_scheduler.vllm_config
            dsv4_capacity_context = None
        if self.vllm_config.cache_config.enable_prefix_caching:
            from vllm.v1.core.kv_cache_utils import init_none_hash
            if scheduler_config.scheduler_version == "0.30.0":
                from vllm.utils.hashing import get_hash_fn_by_name
            elif scheduler_config.scheduler_version == "0.11.0":
                from vllm.utils import get_hash_fn_by_name
            else:
                raise ValueError("unsupported vLLM prefix hash API version")
            init_none_hash(get_hash_fn_by_name(self.vllm_config.cache_config.prefix_caching_hash_algo))
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MLAAttentionSpec,
        )
        kv_groups = self.scheduler.kv_cache_config.kv_cache_groups
        self._prepared_v4 = prepared_model is not None and prepared_model.model_type == "deepseek_v4"
        self._prepared_k3 = prepared_model is not None and prepared_model.model_id == "moonshotai/Kimi-K3"
        self._prepared_mla_token_ids = prepared_model is not None and prepared_model.model_type in ("glm_moe_dsa", "kimi_k2", "deepseek_v3")
        self._k3_kda_group_by_layer = {}
        self._k3_mla_group = None
        if self._prepared_k3:
            from src.serving.workload.model_config import is_kimi_k3_parallel_model_scope
            if not is_kimi_k3_parallel_model_scope(prepared_model, scheduler_config.tensor_parallel_size,
                    scheduler_config.pipeline_parallel_size, expert_parallel=scheduler_config.enable_expert_parallel):
                raise NotImplementedError("ordinary K3 Scheduler source requires whole TP/stage-local EP and nonempty PP")
            if prepared_k3_cache_groups is not None:
                self._k3_kda_group_by_layer, self._k3_mla_group = prepared_k3_cache_groups
            else:
                from vllm.v1.kv_cache_interface import MambaSpec
                from src.serving.kv_cache.vllm_allocation import vllm030_model_cache_specs
                specs = vllm030_model_cache_specs(self.vllm_config, prepared_model)
                kda_group_by_layer, mla_groups = {}, []
                for group_index, group in enumerate(kv_groups):
                    if not group.layer_names:
                        continue
                    kinds = {"kda" if isinstance(specs[name], MambaSpec) else "mla" for name in group.layer_names}
                    if len(kinds) != 1:
                        raise ValueError("K3 cache group mixes KDA and MLA owners")
                    if kinds == {"kda"}:
                        if len(group.layer_names) != 1:
                            raise ValueError("ordinary K3 requires one KDA owner per state group")
                        name, = group.layer_names
                        layer = int(name.split(".")[3])
                        if layer in kda_group_by_layer:
                            raise ValueError("K3 cache groups duplicate a KDA layer")
                        kda_group_by_layer[layer] = group_index
                    else:
                        mla_groups.append(group_index)
                if len(kda_group_by_layer) != 69 or len(mla_groups) != 1:
                    raise ValueError("K3 Scheduler requires 69 KDA groups and one shared MLA group")
                self._k3_kda_group_by_layer = kda_group_by_layer
                self._k3_mla_group = mla_groups[0]
            self.attention_kv_layout = "mla"
        if self._prepared_k3:
            self._dsv4_groups = None
            self._dsv4_model = None
        elif dsv4_capacity_context is not None:
            # DeepSeek-V4 runs one Scheduler group per cache lifecycle class
            # (review-06 R1); the attention layout remains the MLA-style
            # compressed contract.
            self._dsv4_groups, self._dsv4_model = dsv4_capacity_context
            self.attention_kv_layout = "mla"
            if len(kv_groups) != len(self._dsv4_groups):
                raise RuntimeError(
                    "DeepSeek-V4 Scheduler group count disagrees with the "
                    f"ModelConfig cache groups: scheduler={len(kv_groups)}, "
                    f"model={len(self._dsv4_groups)}")
        elif self._prepared_v4:
            self._dsv4_groups = None
            self._dsv4_model = prepared_model
            self.attention_kv_layout = "mla"
        else:
            self._dsv4_groups = None
            self._dsv4_model = None
            if len(kv_groups) != 1:
                raise NotImplementedError(
                    "the normalized vLLM bridge requires one KV cache group")
            kv_spec = kv_groups[0].kv_cache_spec
            if isinstance(kv_spec, MLAAttentionSpec):
                self.attention_kv_layout = "mla"
            elif type(kv_spec) is FullAttentionSpec:
                self.attention_kv_layout = "paged_varlen"
            else:
                raise NotImplementedError(
                    "the normalized vLLM bridge does not support KV spec "
                    f"{type(kv_spec).__name__}")
        if scheduler_config.scheduler_version == "0.30.0":
            if prepared_kv_geometry is None or len(prepared_kv_geometry) != len(kv_groups):
                raise ValueError("0.30 bridge requires complete worker cache traffic geometry")
            self._kv_group_geometry = prepared_kv_geometry
            from vllm.v1.kv_cache_interface import SlidingWindowMLASpec, UniformTypeKVCacheSpecs
            sliding = []
            for group in kv_groups:
                spec = group.kv_cache_spec
                members = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
                flags = {isinstance(member, SlidingWindowMLASpec) for member in members}
                if len(flags) != 1:
                    raise ValueError("0.30 cache group mixes full and sliding lifecycles")
                sliding.append(next(iter(flags)))
            self._sliding_cache_groups = tuple(sliding)
        else:
            self._kv_group_geometry = tuple(
                (group.kv_cache_spec.block_size,
                 group.kv_cache_spec.page_size_bytes)
                for group in kv_groups)
            if self._dsv4_groups is None:
                # A uniform Scheduler group shares block IDs across layers;
                # its spec page is per layer (or already aggregate for DSA).
                # Capacity events charge the actual worker tensor rows once.
                native_cache = self.scheduler.kv_cache_config
                row_bytes, remainder = divmod(sum(tensor.size for tensor in native_cache.kv_cache_tensors),
                    native_cache.num_blocks)
                if remainder or row_bytes <= 0:
                    raise ValueError("0.11 worker KV tensors must contain whole positive cache rows")
                group, = kv_groups
                self._kv_group_geometry = ((group.kv_cache_spec.block_size, row_bytes),)
            self._sliding_cache_groups = (tuple(group.lifecycle == "sliding_window" for group in self._dsv4_groups)
                if self._dsv4_groups is not None else (False,) * len(kv_groups))
        self._requests: dict[str, RequestSpec] = {}
        self._deferred_requests: deque[RequestSpec] = deque()
        self._first_schedule: set[str] = set()
        # EngineCore queues scheduled futures in FIFO completion order.  The
        # simulator mirrors that control-plane queue so a pipeline-depth
        # scheduler can submit a later step before output processing releases
        # the earliest slot.
        self._pending: deque[tuple[BackendStep, Any]] = deque()
        self._pd_retained_blocks = {}
        self._pd_reserved_requests = {}
        # Empty native preemption decisions have no executor work. Preserve
        # their actual releases until the next model step, before allocations.
        self._empty_preemption_releases: list[KVCacheEvent] = []
        self._empty_preemption_ids: tuple[str, ...] = ()
        self._unpublished_empty_preemptions: list[str] = []
        self._step_index = 0
        self._dspark_request_rows = _VLLM030RequestRows.empty(self.config.max_requests)
        self._accepted_request_ids: set[str] = set()
        self._computed_high_watermark: dict[str, int] = {}
        self._trace_prefix_hit_tokens: dict[str, int] = {}
        self._scheduler_prefix_hit_tokens: dict[str, int] = {}
        self._prefix_replay_active: dict[str, bool] = {}
        self._accepted_requests: list[RequestSpec] = []
        self._nohit_marker_by_id: dict[str, int] = {}
        self._mtp_request_scope_validated = False
        self._service_measurement_samples: dict[str, list[int]] | None = None
        self._modeled_simple_cpu_offload_enabled = False
        self._simple_cpu_cpu_owner: dict[int, tuple[str, int]] = {}
        self._simple_cpu_finished_sources: dict[int, tuple[str, int, int, int]] = {}
        self._simple_cpu_completed_releases: list[KVCacheEvent] = []

    def take_modeled_simple_cpu_releases(self) -> tuple[KVCacheEvent, ...]:
        if not self._modeled_simple_cpu_offload_enabled or self._pending:
            raise ValueError("SimpleCPU releases require a completed modeled step")
        events = tuple(self._simple_cpu_completed_releases)
        self._simple_cpu_completed_releases.clear()
        return events

    def enable_modeled_simple_cpu_offload(self) -> None:
        if self.scheduler_config.simple_cpu_offload_bytes_per_rank is None:
            raise ValueError("modeled SimpleCPU offload requires the native connector")
        if self._modeled_simple_cpu_offload_enabled:
            raise ValueError("modeled SimpleCPU offload was already enabled")
        if self._requests or self._deferred_requests or self._pending:
            raise ValueError("modeled SimpleCPU offload must bind before requests")
        self._modeled_simple_cpu_offload_enabled = True

    def _simple_cpu_store_events(
        self,
        native_output: Any,
        blocks_after_native: dict[str, tuple[tuple[int, ...], ...]],
    ) -> tuple[KVCacheEvent, ...]:
        from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata

        meta = native_output.kv_connector_metadata
        if not isinstance(meta, SimpleCPUOffloadMetadata):
            raise TypeError("native SimpleCPU Scheduler returned unexpected metadata")
        if meta.load_event >= 0 or meta.load_gpu_blocks or meta.load_cpu_blocks:
            raise NotImplementedError(
                "modeled SimpleCPU load-only scheduling and completion are not connected")
        if meta.store_event < 0:
            if meta.store_gpu_blocks or meta.store_cpu_blocks:
                raise ValueError("SimpleCPU store blocks lack an event ID")
            return ()
        if (not meta.store_gpu_blocks
                or len(meta.store_gpu_blocks) != len(meta.store_cpu_blocks)
                or len(set(meta.store_gpu_blocks)) != len(meta.store_gpu_blocks)
                or len(set(meta.store_cpu_blocks)) != len(meta.store_cpu_blocks)):
            raise ValueError("native SimpleCPU store metadata has invalid block pairs")
        manager = self.scheduler.connector.scheduler_manager
        if manager is None:
            raise RuntimeError("native SimpleCPU store has no Scheduler manager")
        request_ids = (manager._store_event_to_reqs[meta.store_event]
            if meta.store_event in manager._store_event_to_reqs else ())
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("native eager store request provenance is ambiguous")
        host_releases = []
        events = []
        finished_releases = []
        for gpu_id, cpu_id in zip(
                meta.store_gpu_blocks, meta.store_cpu_blocks, strict=True):
            if cpu_id in self._simple_cpu_cpu_owner:
                old_request_id, old_start = self._simple_cpu_cpu_owner.pop(cpu_id)
                old_block_tokens, old_page_bytes = self._kv_group_geometry[0]
                host_releases.append(KVCacheEvent(
                    request_id=old_request_id,
                    action=KVCacheAction.RELEASE,
                    tier="host",
                    token_start=old_start,
                    token_end=old_start + old_block_tokens,
                    bytes=old_page_bytes,
                ))
            matches = []
            for request_id in request_ids:
                if request_id not in blocks_after_native:
                    continue
                for group_index, group in enumerate(blocks_after_native[request_id]):
                    matches.extend((request_id, group_index, index)
                        for index, block_id in enumerate(group)
                        if block_id == gpu_id)
            if len(matches) == 1 and gpu_id not in self._simple_cpu_finished_sources:
                request_id, group_index, index = matches[0]
                block_tokens, page_bytes = self._kv_group_geometry[group_index]
                token_start = index * block_tokens
            elif not matches and gpu_id in self._simple_cpu_finished_sources:
                request_id, token_start, block_tokens, page_bytes = (
                    self._simple_cpu_finished_sources.pop(gpu_id))
                finished_releases.append(KVCacheEvent(
                    request_id=request_id, action=KVCacheAction.RELEASE,
                    tier="hbm", token_start=token_start,
                    token_end=token_start + block_tokens, bytes=page_bytes))
            else:
                raise NotImplementedError(
                    "native SimpleCPU store block has ambiguous source "
                    f"provenance: gpu_id={gpu_id}, matches={matches!r}, "
                    f"request_ids={request_ids!r}")
            if (block_tokens != self.config.kv_block_size_tokens
                    or page_bytes != self.config.kv_bytes_per_token * block_tokens):
                raise NotImplementedError(
                    "modeled SimpleCPU store requires one complete rank-local KV page")
            events.append(KVCacheEvent(
                request_id=request_id,
                action=KVCacheAction.STORE,
                tier="host",
                token_start=token_start,
                token_end=token_start + block_tokens,
                bytes=page_bytes,
                source_tier="hbm",
                destination_tier="host",
                ready_event=(f"simple-cpu-store:{meta.store_event}:"
                             f"gpu-{gpu_id}:cpu-{cpu_id}"),
            ))
        return tuple(host_releases + events + finished_releases)

    def _simple_cpu_load_events(self, meta: Any) -> tuple[KVCacheEvent, ...]:
        if (meta.load_event < 0
                or not meta.load_gpu_blocks
                or len(meta.load_gpu_blocks) != len(meta.load_cpu_blocks)
                or meta.load_event not in meta.load_event_to_reqs):
            raise ValueError("native SimpleCPU load-only metadata is incomplete")
        request_ids = meta.load_event_to_reqs[meta.load_event]
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise ValueError("native SimpleCPU load request identities are invalid")
        events = []
        for gpu_id, cpu_id in zip(
                meta.load_gpu_blocks, meta.load_cpu_blocks, strict=True):
            if cpu_id not in self._simple_cpu_cpu_owner:
                raise RuntimeError(
                    "native CPU load references a slot without a completed store")
            source_request_id, source_start = self._simple_cpu_cpu_owner[cpu_id]
            matches = []
            for request_id in request_ids:
                for group_index, group in enumerate(self._block_ids(request_id)):
                    matches.extend((request_id, group_index, index)
                        for index, block_id in enumerate(group)
                        if block_id == gpu_id)
            if len(matches) != 1:
                raise NotImplementedError(
                    "native CPU load GPU slot lacks one destination position: "
                    f"gpu_id={gpu_id}, matches={matches!r}")
            request_id, group_index, index = matches[0]
            block_tokens, page_bytes = self._kv_group_geometry[group_index]
            token_start = index * block_tokens
            if source_start != token_start or source_request_id == request_id:
                raise NotImplementedError(
                    "modeled SimpleCPU load requires a matching cross-request prefix block")
            events.append(KVCacheEvent(
                request_id=request_id,
                action=KVCacheAction.LOAD,
                tier="host",
                token_start=token_start,
                token_end=token_start + block_tokens,
                bytes=page_bytes,
                source_tier="host",
                destination_tier="hbm",
                ready_event=(f"simple-cpu-load:{meta.load_event}:"
                             f"cpu-{cpu_id}:gpu-{gpu_id}"),
                source_request_id=source_request_id,
            ))
        return tuple(events)

    def _without_shared_load_allocations(
        self, events: list[KVCacheEvent], loads: tuple[KVCacheEvent, ...],
    ) -> list[KVCacheEvent]:
        load_blocks = {(event.request_id, event.token_start) for event in loads}
        if len(load_blocks) != len(loads):
            raise ValueError("native SimpleCPU load positions are duplicated")
        retained = []
        removed = set()
        block_tokens = self.config.kv_block_size_tokens
        for event in events:
            if event.action is not KVCacheAction.ALLOCATE:
                retained.append(event)
                continue
            for start in range(event.token_start, event.token_end, block_tokens):
                key = (event.request_id, start)
                if key in load_blocks:
                    removed.add(key)
                else:
                    retained.append(replace(event,
                        token_start=start, token_end=start + block_tokens,
                        bytes=self.config.kv_bytes_per_token * block_tokens))
        if removed != load_blocks:
            raise RuntimeError(
                "native SimpleCPU load blocks lack matching GPU allocations")
        return retained

    def probe_simple_cpu_connector(
        self, requests: tuple[RequestSpec, ...],
    ) -> tuple[dict[str, object], ...]:
        """Observe native eager connector decisions with instant test completions.

        This is a CPU control-plane probe, not a simulated transfer timeline.
        Requests are admitted serially so a later request can see completed
        stores from an earlier one. The returned block IDs are native pool
        slots, not stable content identities.
        """
        if self.scheduler_config.simple_cpu_offload_bytes_per_rank is None:
            raise ValueError("SimpleCPU probe requires the native connector")
        if self._requests or self._pending:
            raise ValueError("SimpleCPU probe requires an unused control plane")
        if not isinstance(requests, tuple) or not requests or any(
                not isinstance(request, RequestSpec) for request in requests):
            raise TypeError("SimpleCPU probe requires nonempty RequestSpec tuple")
        from vllm.v1.outputs import KVConnectorOutput
        from vllm.v1.simple_kv_offload.metadata import (
            SimpleCPUOffloadMetadata, SimpleCPUOffloadWorkerMetadata,
        )

        records: list[dict[str, object]] = []
        world_size = self.vllm_config.parallel_config.world_size
        for request in requests:
            self.add_request(request)
            steps = 0
            while (self.scheduler.get_num_unfinished_requests()
                   or self.scheduler.connector.has_pending_push_work()):
                steps += 1
                if steps > request.input_tokens + request.output_tokens + 16:
                    raise RuntimeError("SimpleCPU probe exceeded its bounded step count")
                output = self.scheduler.schedule()
                meta = output.kv_connector_metadata
                if not isinstance(meta, SimpleCPUOffloadMetadata):
                    raise TypeError("native SimpleCPU Scheduler returned unexpected metadata")
                if (len(meta.store_gpu_blocks) != len(meta.store_cpu_blocks)
                        or len(meta.load_gpu_blocks) != len(meta.load_cpu_blocks)):
                    raise ValueError("native SimpleCPU metadata has unequal block pairs")
                records.append({
                    "request_id": request.request_id,
                    "scheduled_tokens": dict(output.num_scheduled_tokens),
                    "load_event": meta.load_event,
                    "load_gpu_blocks": tuple(meta.load_gpu_blocks),
                    "load_cpu_blocks": tuple(meta.load_cpu_blocks),
                    "load_event_to_reqs": tuple(sorted((event, tuple(reqs)) for event, reqs in meta.load_event_to_reqs.items())),
                    "store_event": meta.store_event,
                    "store_gpu_blocks": tuple(meta.store_gpu_blocks),
                    "store_cpu_blocks": tuple(meta.store_cpu_blocks),
                    "request_statuses": tuple(sorted(
                        (req_id, native.status.name)
                        for req_id, native in self.scheduler.requests.items())),
                    "computed_tokens_after_schedule": tuple(sorted(
                        (req_id, native.num_computed_tokens)
                        for req_id, native in self.scheduler.requests.items())),
                })
                generated = {
                    req_id: int(
                        self.scheduler.requests[req_id].num_computed_tokens
                        >= self.scheduler.requests[req_id].num_tokens)
                    for req_id in output.num_scheduled_tokens
                }
                runner = _fake_model_runner_output(
                    output, generated, self.scheduler_config.dummy_token_id)
                finished_recving = (
                    set(meta.load_event_to_reqs[meta.load_event])
                    if meta.load_event >= 0 else set())
                completed_stores = (
                    {meta.store_event: world_size}
                    if meta.store_event >= 0 else {})
                runner.kv_connector_output = KVConnectorOutput(
                    finished_recving=finished_recving,
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(completed_stores),
                )
                self.scheduler.update_from_output(output, runner)
        return tuple(records)

    def _scheduler_host_service_s(self, schedule, operation: str) -> float:
        profile = self.scheduler_host_profile
        if profile is None:
            return _VIRTUAL_SCHEDULER_CPU_S
        return profile.service_time_s(schedule, operation)

    def _engine_core_host_round_s(self, schedule) -> tuple[float, float]:
        """Return one step's EngineCore submit/wake Host service.

        Hardware without an independently measured EngineCore host-round
        profile is explicitly schema-gated off: the wiring registry only
        registers measured hardware, so those steps carry zero EngineCore
        Host pacing and stay bit-identical.
        """

        profile = self.engine_core_host_round_profile
        if profile is None:
            return 0.0, 0.0
        return profile.service_times_s(schedule)

    def add_request(self, request: RequestSpec) -> None:
        if (self.scheduler_config.scheduler_version == "0.30.0"
                and self.scheduler_config.async_scheduling and request.prefix_hit_tokens):
            raise NotImplementedError("0.30 prepared bridge has no bound prefix-oracle replay contract")
        if self.scheduler_config.speculative_method == "dspark" and request.output_token_ids is not None:
            raise NotImplementedError("DSpark CPU source currently requires explicit synthetic output feedback")
        if self.scheduler_config.prefix_cache_mode == "trace_no_hit" and (
                request.prompt_token_ids is not None
                or request.output_token_ids is not None):
            raise ValueError("trace_no_hit accepts only length-only requests with observed zero hits")
        if self._mtp_request_scope_validated:
            raise RuntimeError(
                "hash-bound MTP request scope was already "
                "validated; later request admission is unsupported")
        if request.request_id in self._accepted_request_ids:
            raise ValueError(f"duplicate vLLM request ID {request.request_id!r}")
        if self.scheduler_config.prefix_cache_mode == "trace_no_hit":
            # A distinct first token makes every complete prompt block hash
            # distinct across requests. vLLM chains later block hashes over
            # that first block. The representative only drives native cache
            # misses; its values are never exposed as model input identities.
            self._nohit_marker_by_id[request.request_id] = len(self._accepted_requests) + 1
        self._accepted_requests.append(request)
        self._accepted_request_ids.add(request.request_id)
        self._deferred_requests.append(request)
        self._materialize_waiting_prefix()

    def _materialize_waiting_prefix(self) -> None:
        while (self._deferred_requests
               and len(self.scheduler.waiting) < self.config.max_requests):
            self._materialize_request(self._deferred_requests.popleft())

    def _materialize_request(self, request: RequestSpec) -> None:
        if self.config.is_prefill_only:
            logical_output_limit = min(1, request.output_tokens)
        else:
            logical_output_limit = request.output_tokens
        # Pinned vLLM requires max_tokens > 0 and its prompt-completion step
        # always produces one raw terminal sample.  A length-only Trace may
        # legitimately request zero emitted tokens, so drive that native step
        # with one physical slot and truncate the raw sample at the adapter
        # boundary.  The prompt/KV work remains scheduled and observable.
        native_output_limit = max(1, logical_output_limit)
        native_input = request
        if self.scheduler_config.prefix_cache_mode == "trace_no_hit":
            marker = self._nohit_marker_by_id[request.request_id]
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if marker >= vocab_size:
                raise ValueError("trace_no_hit request count exceeds the model vocabulary")
            native_input = replace(request,
                prompt_token_ids=(marker,) + (self.scheduler_config.dummy_token_id,) * (request.input_tokens - 1),
                output_token_ids=(self.scheduler_config.dummy_token_id,) * request.output_tokens)
        reserved = request.request_id in self._pd_reserved_requests
        if reserved:
            native, original = self._pd_reserved_requests.pop(request.request_id)
            if replace(request, arrival_time_s=original.arrival_time_s) != original:
                raise ValueError("PD destination admission changed the reserved request identity")
        else:
            native = _make_native_request(
                self.vllm_config, native_input,
                dummy_token_id=self.scheduler_config.dummy_token_id,
                max_output_tokens=native_output_limit,
                scheduler_version=self.scheduler_config.scheduler_version)
        if self.scheduler_config.prefix_cache_mode == "trace":
            _verify_content_hash_state_disabled(native, self.scheduler_config.scheduler_version)
        if self.config.is_decode_only:
            if request.prefix_hit_tokens != 0:
                raise NotImplementedError(
                    "decode-only vLLM pools cannot re-inject a trace prefix hit")
            # Metadata-only PD connector: the prefill pool has already produced
            # the first token and the transfer edge has delivered all prompt KV.
            # Seeding these two pieces of native Request state makes vLLM
            # allocate/account the transferred blocks and schedule decode only.
            if not reserved:
                native.num_computed_tokens = request.input_tokens
                if request.output_token_ids is not None:
                    if request.output_tokens < 2:
                        raise ValueError("decode-only admission requires remaining output after the transferred first token")
                    native.append_output_token_ids(request.output_token_ids[0])
                else:
                    native.append_output_token_ids(self.scheduler_config.dummy_token_id)
            trace_prefix_hit_tokens = 0
            scheduler_prefix_hit_tokens = 0
            computed_high_watermark = native.num_computed_tokens
        else:
            trace_prefix_hit_tokens = request.prefix_hit_tokens
            if self.scheduler_config.scheduler_version == "0.30.0":
                # The 0.30 WAITING path accepts a positive computed count as
                # externally supplied KV and allocates its private blocks with
                # the remaining prompt. Keep one prompt token for logits and
                # only claim complete blocks; the unfinished tail is replayed.
                reusable_tokens = min(
                    trace_prefix_hit_tokens, request.input_tokens - 1)
                scheduler_prefix_hit_tokens = (
                    reusable_tokens - reusable_tokens
                    % self.config.kv_block_size_tokens)
            else:
                scheduler_prefix_hit_tokens = scheduler_effective_prefix_hit_tokens(
                    request,
                    self.config.kv_block_size_tokens,
                )
            # A positive computed count bypasses vLLM's content-hash lookup.
            # The native Scheduler still allocates private blocks atomically
            # with the remaining prompt work, so capacity and admission remain
            # under Scheduler control without fabricating token identities.
            native.num_computed_tokens = scheduler_prefix_hit_tokens
            computed_high_watermark = scheduler_prefix_hit_tokens
        self.scheduler.add_request(native)
        self._requests[request.request_id] = request
        self._computed_high_watermark[request.request_id] = _vllm_integer(
            computed_high_watermark, "initial computed token high-water mark")
        self._trace_prefix_hit_tokens[
            request.request_id] = trace_prefix_hit_tokens
        self._scheduler_prefix_hit_tokens[
            request.request_id] = scheduler_prefix_hit_tokens
        self._prefix_replay_active[request.request_id] = (
            trace_prefix_hit_tokens > scheduler_prefix_hit_tokens)

    def retain_pd_kv(self, request_id):
        """Hold native physical blocks across the P Scheduler's finish/free."""
        if not self.config.is_prefill_only or request_id in self._pd_retained_blocks:
            raise ValueError("PD source retention requires a fresh prefill request")
        manager = self.scheduler.kv_cache_manager
        groups = manager.get_blocks(request_id).blocks
        blocks = tuple({id(block): block for group in groups for block in group
            if not block.is_null}.values())
        if not blocks:
            raise ValueError("PD source has no allocated KV to retain")
        if self.scheduler_config.scheduler_version == "0.11.0":
            manager.block_pool.touch((list(blocks),))
        elif self.scheduler_config.scheduler_version == "0.30.0":
            manager.block_pool.touch(blocks)
        else:
            raise ValueError("PD retention requires a version-bound block pool API")
        self._pd_retained_blocks[request_id] = blocks
        return tuple(tuple(block.block_id for block in group) for group in groups)

    def release_pd_kv(self, request_id):
        if request_id not in self._pd_retained_blocks:
            raise ValueError("PD source release has no matching retained blocks")
        blocks = self._pd_retained_blocks.pop(request_id)
        self.scheduler.kv_cache_manager.block_pool.free_blocks(blocks)
        return tuple(block.block_id for block in blocks)

    def reserve_pd_destination(self, request):
        """Allocate the destination's real native cache before its copy starts."""
        result = self.try_reserve_pd_destination(request)
        if result is None:
            raise ValueError("insufficient destination KV capacity for PD handoff")
        return result

    def try_reserve_pd_destination(self, request):
        """Attempt an atomic native allocation without fabricating free KV.

        A fresh request rejected by allocate_slots owns no blocks. Keep it out
        of the reserved/admitted maps until a later causal completion frees KV.
        Existing full-ISL native recompute admission also precedes new copies.
        Invalid identities and mode declarations still raise immediately.
        """
        if (not self.config.is_decode_only or request.output_tokens < 2
                or request.prefix_hit_tokens or request.request_id in self._accepted_request_ids
                or request.request_id in self._pd_reserved_requests):
            raise ValueError("invalid or duplicate PD destination reservation")
        if (self.scheduler_config.scheduler_version == "0.30.0"
                and self.scheduler.scheduler_reserve_full_isl):
            for head in self.scheduler.waiting:
                if head.status.name == "PREEMPTED":
                    # A full-ISL native head must reclaim its sequence before
                    # later external handoffs consume its just-freed pages.
                    # Otherwise WAITING transferred prefixes can pin the
                    # pool while native FCFS cannot readmit its local head.
                    return None
                break
        native = _make_native_request(self.vllm_config, request,
            dummy_token_id=self.scheduler_config.dummy_token_id,
            max_output_tokens=request.output_tokens,
            scheduler_version=self.scheduler_config.scheduler_version)
        manager = self.scheduler.kv_cache_manager
        if self.scheduler_config.scheduler_version == "0.30.0":
            # Use the same native protection as async connector KV loads:
            # a partially prefilling request still owns its remaining full
            # sequence admission claim, even before those pages are allocated.
            allocated = manager.allocate_slots(native, request.input_tokens,
                reserved_blocks=self.scheduler._inflight_prefill_reserved_blocks())
        elif self.scheduler_config.scheduler_version == "0.11.0":
            allocated = manager.allocate_slots(native, request.input_tokens)
        else:
            raise ValueError("unsupported native PD reservation Scheduler version")
        if allocated is None:
            return None
        native.num_computed_tokens = request.input_tokens
        native.append_output_token_ids(request.output_token_ids[0]
            if request.output_token_ids is not None else self.scheduler_config.dummy_token_id)
        self._pd_reserved_requests[request.request_id] = (native, request)
        return tuple(tuple(block.block_id for block in group)
            for group in manager.get_blocks(request.request_id).blocks)

    def add_requests(self, requests: tuple[RequestSpec, ...]) -> None:
        for request in requests:
            self.add_request(request)

    def drain_steps(
        self,
        max_steps: int,
    ) -> tuple[tuple[BackendStep, ...], bool]:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int):
            raise TypeError("vLLM tail batch size must be an integer")
        if max_steps < 1:
            raise ValueError("vLLM tail batch size must be positive")
        steps = []
        pending = [step for step, _ in self._pending]
        while len(steps) < max_steps and self.has_work():
            if len(pending) < self.config.scheduler_pipeline_depth:
                step = self.plan_step(0.0)
                if step is None:
                    if not pending:
                        raise RuntimeError(
                            "vLLM tail has work but plan_step returned no step")
                    break
                steps.append(step)
                pending.append(step)
            else:
                self.complete_step(
                    pending.pop(0),
                    0.0,
                    execute_model=False,
                    execute_transfers=False,
                    sample_tokens=False,
                )
        has_more = bool(self._deferred_requests) or self.scheduler.has_unfinished_requests()
        while pending and not has_more:
            self.complete_step(
                pending.pop(0),
                0.0,
                execute_model=False,
                execute_transfers=False,
                sample_tokens=False,
            )
        return tuple(steps), self.has_work()

    def pending_step_ids(self) -> tuple[str, ...]:
        return tuple(step.schedule.step_id for step, _ in self._pending)

    def plan_step(self, virtual_time_s: float) -> BackendStep | None:
        seen = set()
        while True:
            self._empty_preemption_ids = ()
            step = self._plan_step_once(virtual_time_s)
            if step is not _EMPTY_NATIVE_PREEMPTION_PROGRESS:
                return step
            preempted = set(self._empty_preemption_ids)
            if (not preempted or not preempted.issubset(self._requests)
                    or seen.intersection(preempted)):
                raise RuntimeError("native empty preemption control did not advance")
            seen.update(preempted)

    def _plan_step_once(self, virtual_time_s: float) -> Any:
        if (self.scheduler_config.simple_cpu_offload_bytes_per_rank is not None
                and not self._modeled_simple_cpu_offload_enabled):
            raise NotImplementedError(
                "0.30 SimpleCPU offload needs modeled connector store/load metadata and transfer completion before request execution")
        if (not self._mtp_request_scope_validated
                and self.scheduler_config.mtp_acceptance_policy == "fixed_count"
                and self.scheduler_config.request_stream_sha256 is not None):
            accepted_requests = RequestStream(self._accepted_requests)
            if (self.config.role == "attention"
                    and self.config.max_requests == 2):
                _validate_standalone_af_mtp_preemption_requests(
                    accepted_requests)
            _validate_fixed_count_mtp_submitted_requests(
                self.scheduler_config.request_stream_sha256,
                accepted_requests,
            )
            self._mtp_request_scope_validated = True
        self._materialize_waiting_prefix()
        if not self.has_work():
            return None

        decision_scope = tuple(self.scheduler.running) + tuple(islice(
            self.scheduler.waiting,
            self.config.max_requests,
        ))
        decision_scope_ids = tuple(
            _vllm_request_id(native_request.request_id)
            for native_request in decision_scope
        )
        if len(decision_scope_ids) != len(set(decision_scope_ids)):
            raise RuntimeError(
                "vLLM decision scope contains duplicate request IDs")
        status_before_native = {
            request_id: native_request.status.name
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        computed_before_native = {
            request_id: _vllm_integer(
                native_request.num_computed_tokens,
                "computed token count before schedule",
            )
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        blocks_before_native = {
            request_id: self._block_ids(request_id)
            for request_id in decision_scope_ids
        }
        draft_tokens_before_native = {
            request_id: tuple(
                _vllm_integer(token_id, "MTP proposed token ID")
                for token_id in native_request.spec_token_ids
            )
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        output_placeholders_before_native = {
            request_id: _vllm_integer(
                native_request.num_output_placeholders,
                "output placeholder count before schedule",
            )
            for request_id, native_request in zip(
                decision_scope_ids, decision_scope, strict=True)
        }
        plan_started_ns = (
            time.perf_counter_ns()
            if self._service_measurement_samples is not None else None)
        guarded_terminal_requests = []
        if self.scheduler_config.async_terminal_guard_source is not None:
            # vLLM 0.18 scheduler.py's RUNNING loop skips a request whose
            # pending async output already guarantees max_tokens.  The
            # pinned 0.11 Scheduler lacks that branch.  Suppress only its
            # one-token, no-MTP terminal placeholder during schedule();
            # restore the native request before output processing.
            for native_request in self.scheduler.running:
                pending = native_request.num_output_placeholders
                if (pending > 0
                        and native_request.num_computed_tokens + 2 - pending
                        >= native_request.num_prompt_tokens
                        + native_request.max_tokens):
                    if (pending != 1
                            or native_request.num_tokens_with_spec
                            != native_request.num_computed_tokens):
                        raise NotImplementedError(
                            "vLLM 0.18 terminal guard bridge encountered "
                            "an unsupported pending-output state")
                    native_request.num_output_placeholders = 0
                    guarded_terminal_requests.append(native_request)
        try:
            native_output = self.scheduler.schedule()
        finally:
            for native_request in guarded_terminal_requests:
                native_request.num_output_placeholders = 1
        if any(native_request.request_id in native_output.num_scheduled_tokens
               for native_request in guarded_terminal_requests):
            raise RuntimeError(
                "vLLM 0.18 terminal guard did not suppress a terminal row")
        if self._modeled_simple_cpu_offload_enabled:
            from vllm.v1.outputs import KVConnectorOutput
            from vllm.v1.simple_kv_offload.metadata import (
                SimpleCPUOffloadMetadata, SimpleCPUOffloadWorkerMetadata,
            )
            load_events = []
            control_steps = 0
            while True:
                meta = native_output.kv_connector_metadata
                if not isinstance(meta, SimpleCPUOffloadMetadata):
                    raise TypeError("native SimpleCPU Scheduler returned unexpected metadata")
                if meta.load_event < 0:
                    break
                if (native_output.num_scheduled_tokens
                        or meta.store_event >= 0):
                    raise NotImplementedError(
                        "modeled SimpleCPU load with concurrent model/store work is unsupported")
                control_steps += 1
                if control_steps > self.config.max_requests:
                    raise RuntimeError("native SimpleCPU load-only scheduling did not advance")
                load_events.extend(self._simple_cpu_load_events(meta))
                runner = _fake_model_runner_output(
                    native_output, {}, self.scheduler_config.dummy_token_id)
                runner.kv_connector_output = KVConnectorOutput(
                    finished_recving=set(meta.load_event_to_reqs[meta.load_event]),
                    kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata({}),
                )
                self.scheduler.update_from_output(native_output, runner)
                native_output = self.scheduler.schedule()
            store_events = self._simple_cpu_store_events(
                native_output, {
                    request_id: self._block_ids(request_id)
                    for request_id in decision_scope_ids
                })
        else:
            load_events = []
            store_events = ()
        if plan_started_ns is not None:
            plan_finished_ns = time.perf_counter_ns()
            self._service_measurement_samples["schedule_ns"].append(plan_finished_ns - plan_started_ns)
            if "schedule_start_ns" in self._service_measurement_samples:
                self._service_measurement_samples["schedule_start_ns"].append(plan_started_ns)
                self._service_measurement_samples["schedule_end_ns"].append(plan_finished_ns)
        status_after_native = {}
        computed_after_native = {}
        blocks_after_native = {}
        preempted_request_ids = []
        for request_id, computed_before in computed_before_native.items():
            if request_id not in self.scheduler.requests:
                raise RuntimeError(
                    "vLLM removed a request during schedule() before model "
                    f"output: {request_id!r}")
            native_request = self.scheduler.requests[request_id]
            computed_after = _vllm_integer(
                native_request.num_computed_tokens,
                "computed token count after schedule",
            )
            status_name = native_request.status.name
            status_after_native[request_id] = status_name
            computed_after_native[request_id] = computed_after
            blocks_after_native[request_id] = self._block_ids(request_id)
            newly_preempted = (
                status_before_native[request_id] != "PREEMPTED"
                and status_name == "PREEMPTED"
            )
            if computed_after < computed_before:
                if status_name != "PREEMPTED" or computed_after != 0:
                    raise RuntimeError(
                        "vLLM regressed computed tokens without canonical "
                        f"recompute preemption for {request_id!r}: "
                        f"before={computed_before}, after={computed_after}, "
                        f"status_before={status_before_native[request_id]!r}, "
                        f"status_after={status_name!r}, "
                        f"blocks_before={blocks_before_native[request_id]}, "
                        f"blocks_after={blocks_after_native[request_id]}")
                newly_preempted = True
            if newly_preempted:
                if any(blocks_after_native[request_id]):
                    raise RuntimeError(
                        "vLLM preempted request retained KV blocks: "
                        f"{request_id!r}")
                preempted_request_ids.append(request_id)
                self._prefix_replay_active[request_id] = False
        cached_request_ids = tuple(
            _vllm_request_id(request_id)
            for request_id in native_output.scheduled_cached_reqs.req_ids)
        resumed_request_ids = _resumed_native_request_ids(
            native_output.scheduled_cached_reqs, cached_request_ids,
            self.scheduler_config.scheduler_version)
        scheduled_counts = {}
        for request_id, count in native_output.num_scheduled_tokens.items():
            normalized_id = _vllm_request_id(request_id)
            if normalized_id in scheduled_counts:
                raise ValueError(
                    f"vLLM returned duplicate request ID {normalized_id!r}")
            scheduled_counts[normalized_id] = _vllm_integer(
                count, "scheduled token count")
        if not scheduled_counts:
            if (preempted_request_ids
                    and self.scheduler_config.scheduler_version == "0.30.0"
                    and not self.scheduler_config.async_scheduling
                    and not self._modeled_simple_cpu_offload_enabled):
                releases = tuple(event for request_id in preempted_request_ids
                    for event in self._capacity_events(request_id,
                        KVCacheAction.RELEASE, blocks_before_native[request_id],
                        blocks_after_native[request_id]))
                if not releases:
                    raise RuntimeError("empty native preemption freed no KV capacity")
                if set(preempted_request_ids).intersection(self._unpublished_empty_preemptions):
                    raise RuntimeError("native empty preemption repeated before model progress")
                self._empty_preemption_releases.extend(releases)
                self._unpublished_empty_preemptions.extend(preempted_request_ids)
                self._empty_preemption_ids = tuple(preempted_request_ids)
                # v0.30 skips WAITING after any RUNNING preemption. Repeat
                # only this proven state transition, at the same virtual time.
                return _EMPTY_NATIVE_PREEMPTION_PROGRESS
            if self._pending or self._pd_retained_blocks:
                # EngineCore can ask the Scheduler for another batch while an
                # earlier executor future is still outstanding.  An empty
                # decision is the source state machine's signal to wait for
                # the oldest future; it does not consume another slot.
                return None
            if self._pd_reserved_requests:
                if not self.config.is_decode_only:
                    raise RuntimeError("PD destination capacity wait requires a decode pool")
                # These blocks belong to already reserved, in-transit
                # handoffs. Only their real dispatch/completion can unblock
                # the native waiting queue; the multi-pool event owner must
                # provide that cause or report its ordinary deadlock.
                return None
            if self._modeled_simple_cpu_offload_enabled:
                if load_events or not any(
                        event.action is KVCacheAction.STORE
                        for event in store_events):
                    raise RuntimeError(
                        "native SimpleCPU returned empty work without a store")
                schedule = ScheduleStep(
                    step_id=f"{self.config.pool_name}:vllm:{self._step_index}",
                    pool_name=self.config.pool_name, requests=(),
                    attention_kv_layout=self.attention_kv_layout,
                    transfer_only=True)
                step = BackendStep(
                    schedule, kv_cache_events=store_events,
                    generated_tokens={},
                    scheduler_plan_cpu_s=self._scheduler_host_service_s(
                        schedule, "plan"))
                self._pending.append((step, native_output))
                self._step_index += 1
                return step
            raise RuntimeError(
                "vLLM Scheduler returned an empty step while work remains: "
                f"pool={self.config.pool_name!r}, role={self.config.role!r}")
        preempted_request_ids, preemption_releases = self._preemption_step_payload(
            preempted_request_ids, blocks_before_native, blocks_after_native)
        untracked_scheduled_ids = set(scheduled_counts) - set(
            decision_scope_ids)
        if untracked_scheduled_ids:
            raise RuntimeError(
                "vLLM scheduled requests outside the bounded decision scope: "
                f"{sorted(untracked_scheduled_ids)!r}")
        scheduled_spec_tokens = {}
        scheduled_padding_tokens = {}
        for request_id, token_ids in (
                native_output.scheduled_spec_decode_tokens.items()):
            normalized_id = _vllm_request_id(request_id)
            if normalized_id in scheduled_spec_tokens:
                raise ValueError(
                    "vLLM returned duplicate speculative request ID "
                    f"{normalized_id!r}")
            if normalized_id not in scheduled_counts:
                raise RuntimeError(
                    "vLLM returned speculative tokens for an unscheduled "
                    f"request: {normalized_id!r}")
            normalized_tokens = tuple(
                _vllm_integer(token_id, "MTP verified token ID")
                for token_id in token_ids
            )
            proposed_tokens = draft_tokens_before_native[normalized_id]
            # Scheduler.schedule (0.30) pads a newly admitted decode with -1
            # when prompt KV and the P seed are present but no draft exists.
            # These rows are not proposals and never enter the A60 draw.
            padding = (
                self.scheduler_config.scheduler_version == "0.30.0"
                and self.config.is_decode_only and not proposed_tokens
                and bool(normalized_tokens)
                and normalized_tokens == (-1,) * len(normalized_tokens)
                and len(normalized_tokens) == self.scheduler_config.mtp_num_speculative_tokens
                and scheduled_counts[normalized_id] == len(normalized_tokens) + 1
                and computed_before_native[normalized_id] >= self._requests[normalized_id].input_tokens)
            if padding:
                if self.scheduler_config.mtp_acceptance_replay is not None:
                    raise NotImplementedError("MTP replay schema does not represent native padding rows")
                scheduled_padding_tokens[normalized_id] = len(normalized_tokens)
            elif normalized_tokens != proposed_tokens[:len(normalized_tokens)]:
                raise RuntimeError(
                    "vLLM scheduled speculative tokens are not a prefix of "
                    f"the installed proposal for {normalized_id!r}")
            scheduled_spec_tokens[normalized_id] = normalized_tokens
        if (self.scheduler_config.mtp_num_speculative_tokens == 0
                and scheduled_spec_tokens):
            raise RuntimeError(
                "vLLM scheduled speculative tokens while MTP is disabled")
        if self.scheduler_config.mtp_num_speculative_tokens:
            speculative_config = self.vllm_config.speculative_config
            if speculative_config is None:
                raise RuntimeError("effective vLLM MTP config disappeared")
            draft_model_config = speculative_config.draft_model_config
            if draft_model_config is None:
                raise RuntimeError("effective vLLM MTP draft model disappeared")
            effective_drafter_max_model_len = draft_model_config.max_model_len
            if effective_drafter_max_model_len is None:
                effective_drafter_max_model_len = (
                    self.vllm_config.model_config.max_model_len)
            effective_drafter_max_model_len = _vllm_integer(
                effective_drafter_max_model_len,
                "MTP drafter max model length",
            )
            max_scheduled_sequence_length = max(
                computed_after_native[request_id]
                for request_id in scheduled_counts)
            mtp_drafter_executes = (
                max_scheduled_sequence_length
                + self.scheduler_config.mtp_num_speculative_tokens
                <= effective_drafter_max_model_len
            )
        else:
            mtp_drafter_executes = False
        replay_step = None
        replay = self.scheduler_config.mtp_acceptance_replay
        if replay is not None:
            replay_step = replay.step_for(self._step_index)
            actual_step_id = (
                f"{self.config.pool_name}:vllm:{self._step_index}")
            if replay_step.step_id != actual_step_id:
                raise RuntimeError(
                    "MTP acceptance replay step ID disagrees with the native "
                    f"Scheduler: expected={replay_step.step_id!r}, "
                    f"actual={actual_step_id!r}")
            selected_request_ids = tuple(scheduled_counts)
            if replay_step.selected_request_ids != selected_request_ids:
                raise RuntimeError(
                    "MTP acceptance replay selected request order disagrees "
                    f"with the native Scheduler: "
                    f"expected={replay_step.selected_request_ids}, "
                    f"actual={selected_request_ids}")
            if (replay_step.preempted_request_ids
                    != tuple(preempted_request_ids)):
                raise RuntimeError(
                    "MTP acceptance replay preempted requests disagree with "
                    f"the native Scheduler: "
                    f"expected={replay_step.preempted_request_ids}, "
                    f"actual={tuple(preempted_request_ids)}")
            if replay_step.mtp_drafter_executed != mtp_drafter_executes:
                raise RuntimeError(
                    "MTP acceptance replay drafter execution disagrees with "
                    f"the native Scheduler at step {self._step_index}")
        if (self.scheduler_config.scheduler_version == "0.30.0"
                and self.scheduler.use_v2_model_runner):
            # Pinned Scheduler.schedule moves resumed requests into
            # scheduled_new_reqs for MRV2 and clears its cached resume set.
            # Preserve the native PREEMPTED transition rather than treating
            # a freshly transferred worker record as a newly arrived request.
            resumed_request_ids = _vllm030_v2_resumed_request_ids(
                native_output.scheduled_new_reqs, resumed_request_ids,
                status_before_native)
        resumed_set = set(resumed_request_ids)
        scheduled_ids = set(scheduled_counts)
        if not resumed_set.issubset(scheduled_ids):
            raise RuntimeError(
                "vLLM marked an unscheduled request as resumed from preemption")
        for request_id in resumed_request_ids:
            if status_before_native[request_id] != "PREEMPTED":
                raise RuntimeError(
                    "vLLM resumed flag lacks a PREEMPTED source state for "
                    f"{request_id!r}")
            if status_after_native[request_id] != "RUNNING":
                raise RuntimeError(
                    "vLLM resumed request did not return to RUNNING: "
                    f"{request_id!r}")
        transitioned_to_running = {
            request_id
            for request_id in scheduled_counts
            if status_before_native[request_id] == "PREEMPTED"
            and status_after_native[request_id] == "RUNNING"
        }
        if transitioned_to_running != resumed_set:
            raise RuntimeError(
                "vLLM resumed flags disagree with native request states: "
                f"flags={sorted(resumed_set)}, "
                f"states={sorted(transitioned_to_running)}")

        scheduled_requests = []
        generated_tokens = {}
        finished_request_ids = []
        cache_events = list(preemption_releases)
        for request_id, count in scheduled_counts.items():
            if count <= 0:
                raise ValueError(
                    f"vLLM scheduled non-positive token count for {request_id!r}")
            request = self._requests[request_id]
            if request_id not in self.scheduler.requests:
                raise RuntimeError(f"vLLM lost scheduled request {request_id!r}")
            native_request = self.scheduler.requests[request_id]

            computed_after_schedule = computed_after_native[request_id]
            computed_before_schedule = computed_after_schedule - count
            if computed_before_schedule < 0:
                raise ValueError(
                    f"vLLM computed-token count regressed for {request_id!r}: "
                    f"after={computed_after_schedule}, scheduled={count}")
            content_hit = 0
            if (self.scheduler_config.prefix_cache_mode == "token_ids"
                    and not self.config.is_decode_only
                    and status_before_native[request_id] in ("WAITING", "PREEMPTED")):
                content_hit = (computed_before_schedule
                               if status_before_native[request_id] == "WAITING" else 0)
                self._computed_high_watermark[request_id] = max(
                    self._computed_high_watermark[request_id], computed_before_schedule)
            if (computed_before_schedule != computed_before_native[request_id]
                    and not (self.scheduler_config.prefix_cache_mode == "token_ids"
                             and computed_before_native[request_id] == 0)):
                raise RuntimeError(
                    "vLLM scheduled-token base disagrees with native request "
                    f"state for {request_id!r}: "
                    f"before={computed_before_native[request_id]}, "
                    f"derived={computed_before_schedule}")
            high_watermark = self._computed_high_watermark[request_id]
            if high_watermark < computed_before_schedule:
                raise RuntimeError(
                    "vLLM computed high-water mark regressed for "
                    f"{request_id!r}: high={high_watermark}, "
                    f"before={computed_before_schedule}")
            recomputed_tokens = max(
                0,
                min(computed_after_schedule, high_watermark)
                - computed_before_schedule,
            )
            prefix_replay_tokens = 0
            if self._prefix_replay_active[request_id]:
                scheduler_prefix_end = self._scheduler_prefix_hit_tokens[
                    request_id]
                trace_prefix_end = self._trace_prefix_hit_tokens[request_id]
                replay_start = max(
                    computed_before_schedule,
                    scheduler_prefix_end,
                )
                replay_end = min(
                    computed_after_schedule,
                    trace_prefix_end,
                )
                prefix_replay_tokens = max(0, replay_end - replay_start)
                if computed_after_schedule >= trace_prefix_end:
                    self._prefix_replay_active[request_id] = False
            prompt_tokens = request.input_tokens
            phase = (
                "prefill" if computed_before_schedule < prompt_tokens else "decode"
            )
            if phase == "prefill":
                input_tokens = min(
                    count, prompt_tokens - computed_before_schedule)
            elif phase == "decode":
                input_tokens = 0
            else:
                raise RuntimeError(f"vLLM produced unknown phase: {phase!r}")
            output_tokens = count - input_tokens
            first_schedule = request_id not in self._first_schedule
            generated_tokens[request_id] = 0

            before_blocks = blocks_before_native[request_id]
            after_blocks = blocks_after_native[request_id]
            for group_index, (before_group, after_group) in enumerate(
                    zip(before_blocks, after_blocks, strict=True)):
                if self._sliding_cache_groups[group_index]:
                    # Windowed caches legitimately free blocks as the window
                    # slides (SlidingWindowManager.remove_skipped_blocks);
                    # that is the review-06 R1 lifecycle, not preemption.
                    continue
                if after_group[:len(before_group)] != before_group:
                    raise RuntimeError(
                        "vLLM replaced retained KV blocks without preemption "
                        f"for {request_id!r}: before={before_blocks}, "
                        f"after={after_blocks}")
            if self._prepared_v4:
                cache_events.extend(self._capacity_events(
                    request_id, KVCacheAction.RELEASE, before_blocks, after_blocks))
            cache_events.extend(self._capacity_events(
                request_id,
                KVCacheAction.ALLOCATE,
                before_blocks,
                after_blocks,
            ))

            if first_schedule:
                self._first_schedule.add(request_id)
                cached_tokens = min(computed_before_schedule, prompt_tokens)
                if cached_tokens and not (self.config.is_decode_only
                        and self.scheduler_config.prefix_cache_mode == "token_ids"):
                    cache_events.append(
                        self._cache_event(
                            request_id,
                            KVCacheAction.HIT,
                            0,
                            cached_tokens,
                        )
                    )
            if input_tokens:
                miss_start = min(computed_before_schedule, prompt_tokens)
                miss_end = miss_start + input_tokens
                if miss_end > prompt_tokens:
                    raise RuntimeError(
                        "vLLM prefill chunk exceeds the prompt boundary for "
                        f"{request_id!r}: start={miss_start}, end={miss_end}, "
                        f"prompt={prompt_tokens}")
                cache_events.append(
                    self._cache_event(
                        request_id,
                        KVCacheAction.MISS,
                        miss_start,
                        miss_end,
                    )
                )

            generation_boundary = _vllm_integer(
                native_request.num_tokens,
                "request token count before model output",
            )
            proposed_tokens = len(draft_tokens_before_native[request_id])
            verified_tokens = (
                len(scheduled_spec_tokens[request_id])
                if request_id in scheduled_spec_tokens
                else 0
            )
            padding_tokens = (scheduled_padding_tokens[request_id]
                if request_id in scheduled_padding_tokens else 0)
            if verified_tokens > proposed_tokens + padding_tokens:
                raise RuntimeError(
                    "vLLM verified more MTP tokens than were proposed for "
                    f"{request_id!r}")
            accepted_tokens = 0
            proposer_tokens = (
                self.scheduler_config.mtp_num_speculative_tokens
                if mtp_drafter_executes else 0)
            retained_proposed_tokens = len(native_request.spec_token_ids)
            next_proposed_tokens = retained_proposed_tokens
            raw_model_output_tokens = 0
            expected_replay_request = None
            if replay_step is not None:
                expected_replay_request = replay_step.request_for(request_id)
                actual_pre_acceptance_identity = (
                    request_id,
                    phase,
                    input_tokens,
                    output_tokens,
                    computed_before_schedule,
                    recomputed_tokens,
                    prefix_replay_tokens,
                    request_id in resumed_set,
                    self._trace_prefix_hit_tokens[request_id]
                    if first_schedule else 0,
                    self._scheduler_prefix_hit_tokens[request_id]
                    if first_schedule else 0,
                    proposed_tokens,
                    verified_tokens,
                    proposer_tokens,
                )
                if (expected_replay_request.pre_acceptance_identity
                        != actual_pre_acceptance_identity):
                    raise RuntimeError(
                        "MTP acceptance replay request identity disagrees "
                        f"with the native Scheduler at step {self._step_index} "
                        f"for {request_id!r}: "
                        f"expected={expected_replay_request.pre_acceptance_identity}, "
                        f"actual={actual_pre_acceptance_identity}")
            # The pinned model runner discards samples while the scheduled
            # sequence length is still behind Request.num_tokens.  This is
            # stricter than the prompt boundary during chunked recomputation.
            if computed_after_schedule >= generation_boundary:
                generated_before = (
                    len(native_request.output_token_ids)
                    + output_placeholders_before_native[request_id]
                )
                if self.config.is_prefill_only:
                    logical_output_limit = min(1, request.output_tokens)
                else:
                    logical_output_limit = request.output_tokens
                native_output_limit = max(1, logical_output_limit)
                native_output_remaining = (
                    native_output_limit - generated_before)
                model_remaining = (
                    (self.vllm_config.model_config.max_model_len
                     if self.scheduler_config.scheduler_version == "0.30.0"
                     else self.vllm_config.scheduler_config.max_model_len)
                    - native_request.num_tokens
                )
                raw_output_capacity = min(
                    native_output_remaining, model_remaining)
                if raw_output_capacity <= 0:
                    if (not self.scheduler_config.async_scheduling
                            or output_placeholders_before_native[request_id] <= 0
                            or native_output_remaining > 0):
                        raise RuntimeError(
                            "vLLM scheduled a generation boundary with no "
                            f"output capacity for {request_id!r}")
                    # AsyncScheduler may submit one more decode while the
                    # terminal token is still an unresolved output
                    # placeholder.  The model executes that step, but
                    # update_from_output drops its sample after the earlier
                    # future finishes and removes the request.
                    raw_generated_tokens = 1
                    raw_model_output_tokens = raw_generated_tokens
                    generated_tokens[request_id] = 0
                    next_proposed_tokens = 0
                else:
                    if padding_tokens:
                        accepted_tokens = 0
                    elif expected_replay_request is not None:
                        accepted_tokens = (
                            expected_replay_request.mtp_accepted_tokens)
                    elif (verified_tokens
                          and self.scheduler_config.mtp_acceptance_policy
                          == "maximum_entropy"):
                        assumption = (
                            self.scheduler_config.mtp_acceptance_assumption)
                        if assumption is None:
                            raise AssertionError(
                                "validated maximum-entropy assumption disappeared")
                        accepted_tokens = assumption.accepted_count(
                            step_id=(
                                f"{self.config.pool_name}:vllm:"
                                f"{self._step_index}"),
                            request_id=request_id,
                            proposed_tokens=verified_tokens,
                            output_tokens_before=(generated_before if assumption.sampling_key
                                == "request_output_prefix_v1" else None),
                        )
                    elif verified_tokens:
                        accepted_tokens = min(
                            self.scheduler_config.mtp_fixed_accepted_tokens,
                            verified_tokens,
                        )
                    raw_generated_tokens = accepted_tokens + 1
                    raw_model_output_tokens = raw_generated_tokens
                    logical_output_remaining = max(
                        0, logical_output_limit - generated_before)
                    generated_tokens[request_id] = min(
                        raw_generated_tokens,
                        logical_output_remaining,
                        model_remaining,
                    )
                    if raw_generated_tokens >= raw_output_capacity:
                        finished_request_ids.append(request_id)
                        next_proposed_tokens = 0
                    elif proposer_tokens:
                        next_proposed_tokens = proposer_tokens
            elif proposer_tokens:
                next_proposed_tokens = proposer_tokens

            if self.scheduler_config.mtp_num_speculative_tokens == 0:
                mtp_proposer_mode = "disabled"
                mtp_source_kind = "disabled"
                mtp_source_id = "disabled"
            elif proposed_tokens or verified_tokens or padding_tokens or proposer_tokens:
                mtp_proposer_mode = "padded"
                mtp_source_kind = (
                    "replay_observation"
                    if replay_step is not None
                    else "explicit_assumption"
                )
                mtp_source_id = (
                    self.scheduler_config.mtp_acceptance_source_id)
            else:
                mtp_proposer_mode = "disabled"
                mtp_source_kind = "disabled"
                mtp_source_id = "disabled"
            scheduled_requests.append(
                ScheduledRequest(
                    request_id=request_id,
                    phase=phase,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    computed_tokens=computed_before_schedule,
                    recomputed_tokens=recomputed_tokens,
                    prefix_replay_tokens=prefix_replay_tokens,
                    content_prefix_hit_tokens=content_hit,
                    resumed_from_preemption=request_id in resumed_set,
                    trace_prefix_hit_tokens=(
                        self._trace_prefix_hit_tokens[request_id]
                        if first_schedule else 0
                    ),
                    scheduler_prefix_hit_tokens=(
                        self._scheduler_prefix_hit_tokens[request_id]
                        if first_schedule else 0
                    ),
                    raw_model_output_tokens=raw_model_output_tokens,
                    mtp_proposed_tokens=proposed_tokens,
                    mtp_verified_tokens=verified_tokens,
                    mtp_padding_tokens=padding_tokens,
                    mtp_accepted_tokens=accepted_tokens,
                    mtp_proposer_tokens=proposer_tokens,
                    mtp_next_proposed_tokens=next_proposed_tokens,
                    mtp_proposer_mode=mtp_proposer_mode,
                    mtp_acceptance_source_kind=mtp_source_kind,
                    mtp_acceptance_source_id=mtp_source_id,
                )
            )
            self._computed_high_watermark[request_id] = max(
                high_watermark, computed_after_schedule)

        if load_events:
            cache_events = self._without_shared_load_allocations(
                cache_events, tuple(load_events))
            cache_events.extend(load_events)
        cache_events.extend(store_events)
        for request_id in finished_request_ids:
            if self._modeled_simple_cpu_offload_enabled:
                for group_index, group in enumerate(blocks_after_native[request_id]):
                    block_tokens, page_bytes = self._kv_group_geometry[group_index]
                    for index, gpu_id in enumerate(group):
                        if gpu_id in self._simple_cpu_finished_sources:
                            raise RuntimeError("finished SimpleCPU GPU slot is still pinned")
                        self._simple_cpu_finished_sources[gpu_id] = (
                            request_id, index * block_tokens,
                            block_tokens, page_bytes)
            else:
                cache_events.extend(self._capacity_events(
                    request_id,
                    KVCacheAction.RELEASE,
                    blocks_after_native[request_id],
                    tuple(() for _ in blocks_after_native[request_id]),
                ))

        model_source_inputs = None
        if self.scheduler_config.dspark_source_inputs_by_step is not None:
            name=f"{self.config.pool_name}:vllm:{self._step_index}"
            frames=self.scheduler_config.dspark_source_inputs_by_step
            if name not in frames:
                raise ValueError("native DSpark Scheduler produced a step without an explicit source frame")
            model_source_inputs=frames[name]
            if len(scheduled_requests)!=1 or model_source_inputs.request_ids!=(scheduled_requests[0].request_id,):
                raise ValueError("native DSpark source currently requires one matching complete request")
        if self._prepared_k3:
            scheduled_requests.sort(key=lambda request: 0 if request.phase == "decode" else 1)
            model_source_inputs = self._k3_source_inputs(scheduled_requests, blocks_after_native)
        schedule = ScheduleStep(
            step_id=f"{self.config.pool_name}:vllm:{self._step_index}",
            pool_name=self.config.pool_name,
            requests=tuple(scheduled_requests),
            finished_request_ids=tuple(finished_request_ids),
            preempted_request_ids=tuple(preempted_request_ids),
            attention_kv_layout=self.attention_kv_layout,
        )
        if self.scheduler_config.speculative_method == "dspark":
            from src.serving.workload.runtime_inputs import DSparkReplicaInputs, DSparkSourceInputs
            draft, sizes, target_group, draft_groups = self._dspark_cache_metadata
            if any(self._requests[request.request_id].output_token_ids is not None for request in schedule.requests):
                raise NotImplementedError("DSpark native source frame currently requires declared synthetic output feedback")
            declared = _fake_model_runner_output(native_output,
                {request.request_id: request.raw_model_output_tokens for request in schedule.requests}, self.scheduler_config.dummy_token_id)
            sampled = dict(zip(declared.req_ids, declared.sampled_token_ids, strict=True))
            request_rows=self._dspark_request_rows.advance(
                finished_ids=tuple(set(native_output.finished_req_ids).union(native_output.preempted_req_ids)),
                new_ids=tuple(data.req_id for data in native_output.scheduled_new_reqs))
            resident_indices=dict(request_rows.residents)
            if any(request.request_id not in resident_indices for request in schedule.requests):
                raise ValueError('DSpark native schedule lost its actual RequestState row admission')
            frame = DSparkReplicaInputs(tuple(request.request_id for request in schedule.requests),
                tuple(self._requests[request.request_id].input_tokens for request in schedule.requests),
                tuple(tuple(sampled[request.request_id]) for request in schedule.requests),
                tuple(tuple(tuple(blocks_after_native[request.request_id][group]) for request in schedule.requests)
                    for group in range(len(sizes))), sizes, target_group, draft_groups, self.config.num_kv_blocks,self.config.max_requests,
                request_state_indices=tuple(resident_indices[request.request_id] for request in schedule.requests))
            frame=replace(frame,admitted_request_ids=tuple(data.req_id for data in native_output.scheduled_new_reqs))
            model_source_inputs = DSparkSourceInputs(draft, (frame,))
            self._dspark_request_rows=request_rows
        engine_submit_host_s, engine_wake_host_s = (
            self._engine_core_host_round_s(schedule))
        embedding_ids = None
        if (self.scheduler_config.prefix_cache_mode != "trace_no_hit"
                and (self._prepared_v4 or self._prepared_k3 or self._prepared_mla_token_ids
                or self.attention_kv_layout == "paged_varlen"
                or self.scheduler_config.prefix_cache_mode == "token_ids")):
            scheduled_ids = []
            unresolved_input_ids = False
            for request in schedule.requests:
                # Scheduler Request.all_token_ids excludes the speculative
                # suffix. GPUModelRunner appends the actual scheduled draft
                # IDs after the non-speculative prefix for each query.
                ordinary_rows = request.scheduled_tokens - request.mtp_verified_tokens
                native_request = self.scheduler.requests[request.request_id]
                ordinary = _scheduled_ordinary_token_ids(
                    native_request.all_token_ids,
                    computed_tokens=request.computed_tokens, rows=ordinary_rows,
                    num_tokens=_vllm_integer(native_request.num_tokens, "native token count"),
                    pending_outputs=output_placeholders_before_native[request.request_id],
                    allow_pending_outputs=(self.scheduler_config.async_scheduling
                        and self.config.scheduler_pipeline_depth > 1
                        and self.scheduler_config.mtp_num_speculative_tokens == 0
                        and request.mtp_verified_tokens == 0),
                    declared_token_ids=(None if self._requests[request.request_id].output_token_ids is None else
                        self._requests[request.request_id].prompt_token_ids
                        + self._requests[request.request_id].output_token_ids))
                if ordinary is None:
                    unresolved_input_ids = True
                else:
                    scheduled_ids.extend(ordinary)
                if request.mtp_verified_tokens:
                    drafts = scheduled_spec_tokens[request.request_id]
                    if len(drafts) != request.mtp_verified_tokens:
                        raise RuntimeError("native scheduled draft IDs do not cover the declared verification rows")
                    # GPUModelRunner._calc_spec_decode_metadata gathers the
                    # original -1 IDs, then _preprocess clamps embedding IDs.
                    scheduled_ids.extend(max(0, _vllm_integer(token_id,
                        "scheduled speculative input token ID")) for token_id in drafts)
            if not unresolved_input_ids:
                embedding_ids = tuple(scheduled_ids)
        worker_request_state = None
        if self.scheduler_config.emit_glm_mtp_source_inputs:
            if model_source_inputs is not None:
                raise ValueError("GLM MTP source frame has two owners")
            if (embedding_ids is None or len(embedding_ids) != schedule.token_count
                    or any(self._requests[r.request_id].prompt_token_ids is not None
                        or self._requests[r.request_id].output_token_ids is not None for r in schedule.requests)):
                raise ValueError("GLM MTP automatic source frames require declared synthetic length-only token feedback")
            from src.serving.workload.runtime_inputs import V32PaddedMTPInputs
            depth = self.scheduler_config.mtp_num_speculative_tokens
            ids = (self.scheduler_config.dummy_token_id,) * len(schedule.requests)
            expected_ids = tuple(token for request in schedule.requests
                for token in ((self.scheduler_config.dummy_token_id,)
                    * (request.scheduled_tokens - request.mtp_padding_tokens)
                    + (0,) * request.mtp_padding_tokens))
            if embedding_ids != expected_ids:
                raise ValueError("GLM synthetic frame IDs disagree with native placeholder clamp")
            if any(request.mtp_proposer_mode == "padded" for request in schedule.requests):
                # GPUModelRunner runs the step-global first-pass shift on the
                # same packed batch the target forward consumes, and replaces
                # one retained row per request in packed request order.
                first_draft_ids = _glm_mtp_first_pass_input_ids(
                    schedule, embedding_ids, depth=depth,
                    dummy_token_id=self.scheduler_config.dummy_token_id,
                    drafter_executes=mtp_drafter_executes)
                model_source_inputs = V32PaddedMTPInputs(
                    request_ids=tuple(r.request_id for r in schedule.requests),
                    source_kind="declared_synthetic",
                    source_id="vllm030-length-only/" + self.scheduler_config.mtp_acceptance_source_id,
                    draft_input_token_ids=(tuple(first_draft_ids),) + (ids,) * (depth - 1),
                    discarded_rows=tuple(_vllm030_discard_sampled_tokens(
                        computed_tokens=request.computed_tokens,
                        scheduled_tokens=request.scheduled_tokens,
                        num_tokens=self.scheduler.requests[request.request_id].num_tokens)
                        for request in schedule.requests),
                    block_size_tokens=self.config.kv_block_size_tokens,
                    parallel_drafting=False, use_local_argmax_reduction=False, probabilistic=False)
        if (self.scheduler_config.scheduler_version == "0.30.0"
                and self.scheduler_config.mtp_num_speculative_tokens == 0
                and self.scheduler_config.pipeline_parallel_size == 1
                and self.scheduler_config.max_model_len is not None):
            # The ordinary fresh V2 worker receives the whole prompt, even
            # when only a chunk is scheduled. Resumed/prefix-hit admissions
            # require their actual extended payload and remain unavailable.
            new_data = native_output.scheduled_new_reqs
            new_ids = tuple(data.req_id for data in new_data)
            cached_ids = tuple(native_output.scheduled_cached_reqs.req_ids)
            if (all(data.num_computed_tokens == 0 and data.prompt_token_ids is not None for data in new_data)
                    and len(set(new_ids + cached_ids)) == len(new_ids) + len(cached_ids)
                    and set(new_ids + cached_ids) == set(native_output.num_scheduled_tokens)):
                from src.serving.host_timing import WorkerRequestState
                worker_request_state = WorkerRequestState("vllm_0_30_0", self.config.max_requests,
                    self.scheduler_config.max_model_len, len(cached_ids),
                    len(set(native_output.finished_req_ids).union(native_output.preempted_req_ids)),
                    len(new_ids), sum(len(data.prompt_token_ids) for data in new_data), self.config.kv_block_size_tokens)
        step = BackendStep(
            schedule=schedule,
            kv_cache_events=tuple(cache_events),
            generated_tokens=generated_tokens,
            # Native Scheduler elapsed time is never a Forward input. A
            # versioned profile may instead supply deterministic service for
            # the exact Scheduler state emitted by this step.
            scheduler_plan_cpu_s=(
                self._scheduler_host_service_s(schedule, "plan")
            ),
            # Same contract for the EngineCore busy-loop segments no other
            # Host profile owns: an independently measured versioned profile
            # supplies deterministic submit/wake service, never a trace
            # duration.
            engine_submit_host_s=engine_submit_host_s,
            engine_wake_host_s=engine_wake_host_s,
            embedding_token_ids=embedding_ids,
            embedding_token_source=(("vllm_explicit_request_token_ids"
                if all(self._requests[r.request_id].prompt_token_ids is not None for r in schedule.requests)
                else "vllm_length_only_simulation")
                if embedding_ids is not None else None),
            model_source_inputs=model_source_inputs,
            worker_request_state=worker_request_state,
        )
        if replay_step is not None:
            actual_replay_step = mtp_acceptance_step_from_backend_step(
                self._step_index, step)
            if actual_replay_step != replay_step:
                raise RuntimeError(
                    "MTP acceptance replay outcome disagrees with the native "
                    f"Scheduler at step {self._step_index}: "
                    f"expected={replay_step!r}, actual={actual_replay_step!r}")
        self._empty_preemption_releases.clear()
        self._unpublished_empty_preemptions.clear()
        self._pending.append((step, native_output))
        self._step_index += 1
        return step

    def complete_step(
        self,
        step: BackendStep,
        virtual_time_s: float,
        *,
        execute_model: bool,
        execute_transfers: bool,
        sample_tokens: bool,
    ) -> float:
        if execute_model or execute_transfers or sample_tokens:
            raise ValueError(
                "the vLLM simulator backend never executes a data plane")
        if not self._pending or self._pending[0][0] is not step:
            raise ValueError("completion does not match the pending vLLM step")
        _, native_output = self._pending[0]
        scheduled_by_id = {
            request.request_id: request
            for request in step.schedule.requests
        }
        present_before_update = {
            request_id
            for request_id in native_output.num_scheduled_tokens
            if request_id in self.scheduler.requests
        }
        for request_id in native_output.num_scheduled_tokens:
            if request_id not in self.scheduler.requests:
                if step.generated_tokens[request_id] != 0:
                    raise RuntimeError(
                        "vLLM lost a pending request with visible output "
                        f"before update: {request_id!r}")
                continue
            native_request = self.scheduler.requests[request_id]
            normalized_request = scheduled_by_id[request_id]
            expected_raw_generated = (
                normalized_request.raw_model_output_tokens)
            actual_emitted = step.generated_tokens[request_id]
            if not 0 <= actual_emitted <= expected_raw_generated:
                raise RuntimeError(
                    "vLLM generated-token contract disagrees with prompt "
                    f"completion for {request_id!r}: "
                    f"raw={expected_raw_generated}, emitted={actual_emitted}")
        raw_generated_tokens = {
            request_id: scheduled_by_id[request_id].raw_model_output_tokens
            for request_id in native_output.num_scheduled_tokens
        }
        runner_output = _fake_model_runner_output(
            native_output,
            raw_generated_tokens,
            self.scheduler_config.dummy_token_id,
        )
        if self._modeled_simple_cpu_offload_enabled:
            from vllm.v1.outputs import KVConnectorOutput
            from vllm.v1.simple_kv_offload.metadata import (
                SimpleCPUOffloadMetadata, SimpleCPUOffloadWorkerMetadata,
            )
            meta = native_output.kv_connector_metadata
            if not isinstance(meta, SimpleCPUOffloadMetadata):
                raise TypeError("pending SimpleCPU step lost connector metadata")
            completed = ({meta.store_event:
                self.vllm_config.parallel_config.world_size}
                if meta.store_event >= 0 else {})
            runner_output.kv_connector_output = KVConnectorOutput(
                kv_connector_worker_meta=(
                    SimpleCPUOffloadWorkerMetadata(completed)),
            )
        if any(request_id in self._requests and self._requests[request_id].output_token_ids is not None
                for request_id in runner_output.req_ids):
            for index, request_id in enumerate(runner_output.req_ids):
                if request_id not in self._requests:
                    if raw_generated_tokens[request_id]:
                        raise RuntimeError("completed request cannot receive further output IDs")
                    continue
                request = self._requests[request_id]
                if request.output_token_ids is None:
                    continue
                if raw_generated_tokens[request_id] and request.output_tokens:
                    native_request = self.scheduler.requests[request_id]
                    offset = len(native_request.output_token_ids)
                    visible = step.generated_tokens[request_id]
                    real_tokens = list(request.output_token_ids[offset:offset + visible])
                    if len(real_tokens) != visible:
                        raise ValueError("real output transcript does not cover the native generated tokens")
                    accepted = min(scheduled_by_id[request_id].mtp_accepted_tokens, visible)
                    scheduled_drafts = native_output.scheduled_spec_decode_tokens
                    if accepted and tuple(real_tokens[:accepted]) != tuple(scheduled_drafts[request_id][:accepted]):
                        raise ValueError("accepted MTP draft IDs disagree with the real output transcript")
                    # Native update computes rejection from the raw count, then
                    # appends one token at a time and stops at max_tokens/model
                    # length. Slots beyond that stop are never appended or hashed.
                    runner_output.sampled_token_ids[index] = real_tokens + [
                        self.scheduler_config.dummy_token_id
                    ] * (raw_generated_tokens[request_id] - visible)
                elif raw_generated_tokens[request_id]:
                    # Zero-output prompt completion still needs a terminal sample.
                    # Disable hashing before its dummy identity reaches native state.
                    self.scheduler.requests[request_id].get_hash_new_full_blocks = None
        proposer_depths = {
            request.mtp_proposer_tokens
            for request in step.schedule.requests
        }
        if step.schedule.is_transfer_only or proposer_depths == {0}:
            mtp_drafter_executed = False
        elif proposer_depths == {
                self.scheduler_config.mtp_num_speculative_tokens}:
            mtp_drafter_executed = True
        else:
            raise RuntimeError(
                "padded vLLM MTP requires one step-global proposer depth: "
                f"{sorted(proposer_depths)}")
        draft_rows = None
        if mtp_drafter_executed:
            if self.scheduler_config.dspark_source_inputs_by_step is not None:
                state=step.model_source_inputs
                if state is None or len(native_output.num_scheduled_tokens)!=1:
                    raise ValueError("DSpark completion lost its actual paired source state")
                draft_rows=[[ids[0] for ids in state.draft_token_ids]]
            elif self.scheduler_config.prefix_cache_mode == "token_ids":
                draft_rows = _validated_mtp_draft_rows(
                    self.scheduler_config.mtp_draft_token_ids_by_step,
                    step.schedule.step_id, tuple(native_output.num_scheduled_tokens),
                    self.scheduler_config.mtp_num_speculative_tokens,
                    self.vllm_config.model_config.get_vocab_size())
            else:
                draft_rows = [[self.scheduler_config.dummy_token_id]
                    * scheduled_by_id[request_id].mtp_proposer_tokens
                    for request_id in native_output.num_scheduled_tokens]
        update_started_ns = (
            time.perf_counter_ns()
            if self._service_measurement_samples is not None else None)
        native_results = self.scheduler.update_from_output(
            native_output, runner_output)
        if self._modeled_simple_cpu_offload_enabled:
            store_events = tuple(event for event in step.kv_cache_events
                                 if event.action is KVCacheAction.STORE)
            if len(store_events) != len(meta.store_cpu_blocks):
                raise RuntimeError(
                    "modeled SimpleCPU stores disagree with completed native event")
            for event, cpu_id in zip(store_events, meta.store_cpu_blocks,
                                     strict=True):
                if cpu_id in self._simple_cpu_cpu_owner:
                    raise RuntimeError("completed CPU slot was already owned")
                self._simple_cpu_cpu_owner[cpu_id] = (
                    event.request_id, event.token_start)
            manager = self.scheduler.connector.scheduler_manager
            if manager is None:
                raise RuntimeError("modeled SimpleCPU lost its Scheduler manager")
            pinned_gpu_ids = manager._in_flight_store_gpu_blocks
            for gpu_id, (request_id, start, block_tokens, page_bytes) in tuple(
                    self._simple_cpu_finished_sources.items()):
                if gpu_id in pinned_gpu_ids:
                    continue
                self._simple_cpu_completed_releases.append(KVCacheEvent(
                    request_id=request_id, action=KVCacheAction.RELEASE,
                    tier="hbm", token_start=start,
                    token_end=start + block_tokens, bytes=page_bytes))
                del self._simple_cpu_finished_sources[gpu_id]
        if update_started_ns is not None:
            update_finished_ns = time.perf_counter_ns()
            self._service_measurement_samples["update_ns"].append(update_finished_ns - update_started_ns)
            if "scheduler_update_start_ns" in self._service_measurement_samples:
                self._service_measurement_samples["scheduler_update_start_ns"].append(update_started_ns)
                self._service_measurement_samples["scheduler_update_end_ns"].append(update_finished_ns)
        if draft_rows is not None:
            from vllm.v1.outputs import DraftTokenIds
            self.scheduler.update_draft_token_ids(DraftTokenIds(
                req_ids=list(native_output.num_scheduled_tokens),
                draft_token_ids=draft_rows,
            ))
        self._pending.popleft()
        # See plan_step(): do not feed local wall-clock measurement noise
        # back into the virtual scheduler timeline.
        update_cpu_s = self._scheduler_host_service_s(step.schedule, "update")
        expected_finished = set(step.finished_request_ids)
        released = {
            request_id for request_id in present_before_update
            if request_id not in self.scheduler.requests
        }
        output_finished = {
            output.request_id
            for client_results in native_results.values()
            for output in client_results.outputs
            if output.finish_reason is not None
        }
        emitted_tokens = {
            request_id: 0 for request_id in step.generated_tokens
        }
        for client_results in native_results.values():
            for output in client_results.outputs:
                if output.request_id not in emitted_tokens:
                    raise RuntimeError(
                        "vLLM emitted output for an unscheduled request: "
                        f"{output.request_id!r}")
                request = self._requests[output.request_id]
                if request.output_tokens == 0:
                    emitted_count = 0
                else:
                    emitted_count = len(output.new_token_ids)
                emitted_tokens[output.request_id] += emitted_count
        if emitted_tokens != step.generated_tokens:
            raise RuntimeError(
                "vLLM emitted-token counts disagree with the planned step: "
                f"planned={step.generated_tokens}, emitted={emitted_tokens}")
        if released != expected_finished or output_finished != expected_finished:
            raise RuntimeError(
                "vLLM finish/release state disagrees with the planned step: "
                f"planned={sorted(expected_finished)}, "
                f"released={sorted(released)}, "
                f"output_finished={sorted(output_finished)}")
        for request_id in native_output.num_scheduled_tokens:
            if request_id in self.scheduler.requests:
                computed_tokens = _vllm_integer(
                    self.scheduler.requests[request_id].num_computed_tokens,
                    "computed token count after model output",
                )
                self._computed_high_watermark[request_id] = max(
                    self._computed_high_watermark[request_id],
                    computed_tokens,
                )
        for request_id in step.finished_request_ids:
            del self._requests[request_id]
            del self._computed_high_watermark[request_id]
            del self._trace_prefix_hit_tokens[request_id]
            del self._scheduler_prefix_hit_tokens[request_id]
            del self._prefix_replay_active[request_id]
        return update_cpu_s

    def measure_scheduler_service(
        self,
        *,
        sample_count: int,
        requests_per_batch: int,
        prompt_tokens: int,
        output_tokens: int,
        include_step_records: bool = False,
    ) -> dict[str, tuple[int, ...]]:
        """Measure Scheduler-only service over a synthetic control workload."""

        for name, value in (
                ("sample_count", sample_count),
                ("requests_per_batch", requests_per_batch),
                ("prompt_tokens", prompt_tokens),
                ("output_tokens", output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"Scheduler microbenchmark {name} must be an integer")
            if value < 1:
                raise ValueError(f"Scheduler microbenchmark {name} must be positive")
        if self._pending:
            raise RuntimeError(
                "Scheduler microbenchmark requires no pending model output")
        if self.has_work():
            raise RuntimeError(
                "Scheduler microbenchmark requires a newly constructed backend")
        if requests_per_batch > self.config.max_requests:
            raise ValueError(
                "Scheduler microbenchmark requests_per_batch exceeds capacity")
        if prompt_tokens > self.config.max_batched_tokens:
            raise ValueError(
                "Scheduler microbenchmark prompt_tokens exceeds batch capacity")

        if type(include_step_records) is not bool:
            raise TypeError("include_step_records must be boolean")
        samples: dict[str, list[int]] = {"schedule_ns": [], "update_ns": []}
        if include_step_records:
            samples.update({name: [] for name in ("schedule_start_ns", "schedule_end_ns",
                "scheduler_update_start_ns", "scheduler_update_end_ns", "scheduled_request_count",
                "prefill_request_count", "decode_request_count")})
        self._service_measurement_samples = samples
        request_index = 0
        try:
            while len(samples["schedule_ns"]) < sample_count:
                if not self.has_work():
                    self.add_requests(tuple(
                        RequestSpec(
                            0.0,
                            f"scheduler-microbench-{request_index + offset}",
                            prompt_tokens,
                            output_tokens,
                        )
                        for offset in range(requests_per_batch)
                    ))
                    request_index += requests_per_batch
                step = self.plan_step(0.0)
                if step is None:
                    raise RuntimeError(
                        "Scheduler microbenchmark failed to produce a step")
                self.complete_step(
                    step,
                    0.0,
                    execute_model=False,
                    execute_transfers=False,
                    sample_tokens=False,
                )
                if include_step_records:
                    samples["scheduled_request_count"].append(len(step.schedule.requests))
                    samples["prefill_request_count"].append(sum(request.phase == "prefill" for request in step.schedule.requests))
                    samples["decode_request_count"].append(sum(request.phase == "decode" for request in step.schedule.requests))
            # The sample boundary may fall within a request's decode run.
            # Finish it without measuring the tail so the isolated native
            # Scheduler can close with no outstanding requests.
            self._service_measurement_samples = None
            while self.has_work():
                step = self.plan_step(0.0)
                if step is None:
                    raise RuntimeError(
                        "Scheduler microbenchmark failed to drain a request")
                self.complete_step(
                    step, 0.0, execute_model=False,
                    execute_transfers=False, sample_tokens=False,
                )
        finally:
            self._service_measurement_samples = None
        if (len(samples["schedule_ns"]) != sample_count
                or len(samples["update_ns"]) != sample_count):
            raise RuntimeError("Scheduler microbenchmark sample counts disagree")
        return {
            name: tuple(values[:sample_count])
            for name, values in samples.items()
        }

    def has_work(self) -> bool:
        if self._pending:
            return True
        if self._deferred_requests:
            return True
        if not callable(self.scheduler.has_unfinished_requests):
            raise TypeError(
                "vLLM scheduler has_unfinished_requests must be callable")
        result = self.scheduler.has_unfinished_requests()
        if not isinstance(result, bool):
            raise TypeError(
                "vLLM scheduler has_unfinished_requests() must return boolean")
        if self._modeled_simple_cpu_offload_enabled:
            pending_push = self.scheduler.connector.has_pending_push_work()
            if not isinstance(pending_push, bool):
                raise TypeError("SimpleCPU pending push status must be boolean")
            return result or pending_push
        return result

    def request_queue_snapshot(self) -> dict:
        """Snapshot actual Scheduler containers without scheduling or feedback.

        Native running membership may include a submitted executor step. It
        is a control-plane state, not a measurement of physical GPU activity.
        PD ownership not yet admitted to the Scheduler stays separate.
        """
        if self.scheduler_config.scheduler_version != "0.30.0":
            raise NotImplementedError("native queue snapshots currently bind vLLM0.30")
        result = {
            "waiting_request_ids": tuple(request.request_id for request in self.scheduler.waiting),
            "running_request_ids": tuple(request.request_id for request in self.scheduler.running),
            "deferred_request_ids": tuple(request.request_id for request in self._deferred_requests),
            "pending_step_ids": tuple(step.schedule.step_id for step, _ in self._pending),
            "P_retained_request_ids": tuple(sorted(self._pd_retained_blocks)),
            "D_reserved_request_ids": tuple(sorted(self._pd_reserved_requests)),
        }
        return _validated_request_queue_snapshot(result)

    def close(self) -> None:
        if self.has_work():
            raise RuntimeError("cannot close a prepared vLLM bridge with unfinished work")
        if self._pd_retained_blocks or self._pd_reserved_requests:
            raise RuntimeError("cannot close a vLLM bridge with live PD cache ownership")
        if (self._modeled_simple_cpu_offload_enabled
                and (self._simple_cpu_finished_sources
                     or self._simple_cpu_completed_releases)):
            raise RuntimeError(
                "cannot close SimpleCPU bridge with unretired finished KV pages")
        self.assert_mtp_contract_exhausted()

    def assert_mtp_contract_exhausted(self) -> None:
        replay = self.scheduler_config.mtp_acceptance_replay
        if replay is not None:
            if self._pending:
                raise RuntimeError(
                    "cannot close vLLM Scheduler with a pending replay step")
            replay.assert_exhausted(self._step_index)
            if self.has_work():
                raise RuntimeError(
                    "MTP acceptance replay is exhausted while the native "
                    "Scheduler still has unfinished requests")
            _validate_mtp_replay_submitted_requests(
                replay.scheduler_context,
                RequestStream(self._accepted_requests),
            )
        elif self.scheduler_config.mtp_acceptance_policy == "maximum_entropy":
            assumption = self.scheduler_config.mtp_acceptance_assumption
            if assumption is None:
                raise AssertionError(
                    "validated maximum-entropy assumption disappeared")
            _validate_assumption_mtp_submitted_requests(
                assumption,
                self.scheduler_config.decode_admission_contract_sha256,
                RequestStream(self._accepted_requests),
                self.scheduler_config.closed_loop_admission_contract_sha256,
            )
        elif self.scheduler_config.request_stream_sha256 is not None:
            if not self._mtp_request_scope_validated:
                raise RuntimeError(
                    "hash-bound MTP request scope was never "
                    "validated")
            accepted_requests = RequestStream(self._accepted_requests)
            if self.scheduler_config.mtp_acceptance_policy == "fixed_count":
                if (self.config.role == "attention"
                        and self.config.max_requests == 2):
                    _validate_standalone_af_mtp_preemption_requests(
                        accepted_requests)
                _validate_fixed_count_mtp_submitted_requests(
                    self.scheduler_config.request_stream_sha256,
                    accepted_requests)
            else:
                raise AssertionError(
                    "request-scoped MTP policy is not fixed-count")

    def _cache_event(
        self,
        request_id: str,
        action: KVCacheAction,
        start: int,
        end: int,
    ) -> KVCacheEvent:
        if self._dsv4_model is not None:
            # Lifecycle-exact reporting charge: streams across the whole
            # range, windowed caches only their in-window intersection
            # (review-06 R1).
            exact_bytes = self._dsv4_model.dsv4_cache_event_bytes(start, end)
            event_bytes = (
                -(-exact_bytes.numerator // exact_bytes.denominator))
        else:
            event_bytes = (end - start) * self.config.kv_bytes_per_token
        return KVCacheEvent(
            request_id=request_id,
            action=action,
            tier="hbm",
            token_start=start,
            token_end=end,
            bytes=event_bytes,
        )

    def _preemption_step_payload(self, current_ids, before_blocks, after_blocks):
        """Publish prior empty controls with the next actual executor step."""
        ids = tuple(self._unpublished_empty_preemptions) + tuple(current_ids)
        if len(ids) != len(set(ids)):
            raise RuntimeError("native preemption repeated before publication")
        releases = tuple(self._empty_preemption_releases) + tuple(
            event for request_id in current_ids
            for event in self._capacity_events(request_id, KVCacheAction.RELEASE,
                before_blocks[request_id], after_blocks[request_id]))
        return ids, releases

    def _capacity_events(
        self,
        request_id: str,
        action: KVCacheAction,
        before_blocks: tuple[tuple[int, ...], ...],
        after_blocks: tuple[tuple[int, ...], ...],
    ) -> list[KVCacheEvent]:
        """Emit capacity events for changed pages in each cache group.

        Each group's block range is expressed in that group's own block
        tokens; bytes charge the group's real page bytes, so windowed
        DeepSeek-V4 caches report their bounded per-request footprint. Native
        mixed groups charge physical pool rows and exclude null placeholders;
        separate contiguous slot runs preserve sliding-window release ranges.
        """

        if action not in (KVCacheAction.ALLOCATE, KVCacheAction.RELEASE):
            raise ValueError(
                "capacity events support only allocate/release actions")
        events = []
        if self._prepared_v4:
            null_id = self.scheduler.kv_cache_manager.block_pool.null_block.block_id
            for group_index, (before, after) in enumerate(zip(before_blocks, after_blocks, strict=True)):
                target, other = (after, before) if action is KVCacheAction.ALLOCATE else (before, after)
                retained = set(other)
                changed = [index for index, block_id in enumerate(target) if block_id != null_id and block_id not in retained]
                block_tokens, pool_row_bytes = self._kv_group_geometry[group_index]
                runs = []
                for index in changed:
                    if runs and runs[-1][1] == index:
                        runs[-1][1] = index + 1
                    else:
                        runs.append([index, index + 1])
                for start, end in runs:
                    events.append(KVCacheEvent(request_id=request_id, action=action, tier="hbm",
                        token_start=start * block_tokens, token_end=end * block_tokens,
                        bytes=(end - start) * pool_row_bytes))
            return events
        for group_index, (before, after) in enumerate(
                zip(before_blocks, after_blocks, strict=True)):
            if action is KVCacheAction.ALLOCATE:
                start_block, end_block = len(before), len(after)
            else:
                start_block, end_block = 0, len(before)
            if end_block <= start_block:
                continue
            block_tokens, page_bytes = self._kv_group_geometry[group_index]
            events.append(KVCacheEvent(
                request_id=request_id,
                action=action,
                tier="hbm",
                token_start=start_block * block_tokens,
                token_end=end_block * block_tokens,
                bytes=(end_block - start_block) * page_bytes,
            ))
        return events

    def _k3_source_capacities(self):
        """Native Scheduler address extents; GPU arena binding is separate."""
        if not self._prepared_k3 or self._k3_mla_group is None:
            raise RuntimeError("K3 source coordinates require the prepared hybrid Scheduler")
        mla_block_tokens,_=self._kv_group_geometry[self._k3_mla_group]
        return self.config.num_kv_blocks*mla_block_tokens,self.config.num_kv_blocks

    def _k3_source_inputs(self, requests, blocks_by_request):
        from src.serving.workload.runtime_inputs import K3SourceInputs

        cache_capacity,state_capacity=self._k3_source_capacities()
        slot_mapping = []
        state_indices_by_layer = [[] for _ in range(93)]
        mla_group = self._k3_mla_group
        mla_block_tokens, _ = self._kv_group_geometry[mla_group]
        for request in requests:
            blocks = blocks_by_request[request.request_id]
            for layer, group_index in self._k3_kda_group_by_layer.items():
                state_block = blocks[group_index]
                if len(state_block) != 1:
                    raise RuntimeError("ordinary K3 requires exactly one recurrent-state block per layer and request")
                state_indices_by_layer[layer].append(state_block[0])
            mla_blocks = blocks[mla_group]
            required_blocks = -(-(request.computed_tokens + request.scheduled_tokens) // mla_block_tokens)
            if len(mla_blocks) < required_blocks:
                raise RuntimeError("K3 Scheduler did not allocate the complete MLA context")
            for position in range(request.computed_tokens, request.computed_tokens + request.scheduled_tokens):
                slot_mapping.append(mla_blocks[position // mla_block_tokens] * mla_block_tokens
                                    + position % mla_block_tokens)
        return K3SourceInputs(
            request_ids=tuple(request.request_id for request in requests),
            slot_mapping=tuple(slot_mapping),
            state_indices_by_layer=tuple(tuple(indices) for indices in state_indices_by_layer),
            cache_capacity_tokens=cache_capacity,
            state_pool_slots=state_capacity,
            prefill_output_source=("supports_out" if any(request.phase == "prefill" for request in requests) else None),
        )

    def _block_ids(self, request_id: str) -> tuple[tuple[int, ...], ...]:
        block_groups = self.scheduler.kv_cache_manager.get_blocks(
            request_id).get_block_ids()
        if len(block_groups) != len(self._kv_group_geometry):
            raise RuntimeError(
                "vLLM KV cache group count changed after construction: "
                f"{len(block_groups)} vs {len(self._kv_group_geometry)}")
        normalized = tuple(
            tuple(_vllm_integer(block_id, "KV block ID")
                  for block_id in group)
            for group in block_groups
        )
        null_block_id = _vllm_integer(
            self.scheduler.kv_cache_manager.block_pool.null_block.block_id,
            "KV null block ID")
        for group_index, group in enumerate(normalized):
            live = [block_id for block_id in group
                    if block_id != null_block_id]
            if len(live) != len(set(live)):
                raise RuntimeError(
                    "vLLM allocated duplicate KV block IDs within a group: "
                    f"request={request_id} group={group_index} ids={group}")
        return normalized


def _build_native_scheduler(
    config: BackendConfig,
    scheduler_config: VLLMSchedulerConfig,
):
    _validate_native_scheduler_scope(config, scheduler_config)
    return _construct_native_scheduler(config, scheduler_config)


def _validate_native_scheduler_scope(
    config: BackendConfig,
    scheduler_config: VLLMSchedulerConfig,
) -> None:
    if scheduler_config.scheduler_version != "0.11.0":
        raise NotImplementedError("from_model remains the pinned 0.11 path; use the explicit prepared 0.30 bridge")
    from src.serving.workload.model_config import load_model_config
    scope_model = load_model_config(scheduler_config.model)
    if scheduler_config.prefix_cache_mode == "token_ids" and scope_model.model_type == "deepseek_v4":
        raise NotImplementedError("token prefix cache requires a supported single KV cache group; V4 mixed cache is unsupported")
    if scope_model.config_sha256 == KIMI_K3_CONFIG_SHA256:
        raise NotImplementedError("official Kimi-K3 requires hybrid KDA/NoPE MLA vLLM 0.30 capacity and Scheduler contracts")
    if scope_model.config_sha256 == KIMI_K2_7_CODE_CONFIG_SHA256:
        raise NotImplementedError(
            "official Kimi-K2.7-Code has no bound vLLM 0.30 normal Scheduler adapter; "
            "the existing from_model Scheduler is pinned to 0.11")
    if scope_model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S:
        raise NotImplementedError(
            "official GLM block-FP8 requires the explicit vLLM 0.30 adapter; "
            "the existing production Scheduler is pinned to 0.11")
    # The Scheduler target scope is one declarative table
    # (src.serving.deployment.validated_scope).  It pins the named target
    # config identities on every row except the 32-layer DeepSeek-V3 TP4
    # row, and deliberately over-approximates: the PP2/PP4 rows accept every
    # named target model in any role, and the PD EP8 row accepts every named
    # target model; the narrower per-model checks in
    # ``_construct_native_scheduler`` and ``_validate_ep_scheduler_model_scope``
    # fail closed afterwards.
    target_scope = is_validated_target_scheduler_scope(
        scope_model,
        role=config.role,
        tensor_parallel_size=scheduler_config.tensor_parallel_size,
        pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
        expert_parallel=scheduler_config.enable_expert_parallel,
    )
    operator_group = worker_role_operator_group(config.role)
    if operator_group == "ffn":
        raise NotImplementedError(
            "the pinned vLLM Scheduler cannot attach to an FFN-only pool")
    if config.role not in (
            "colocated", "prefill", "decode", "attention",
            "prefill-attention", "decode-attention"):
        raise NotImplementedError(
            "the pinned vLLM Scheduler role has not been validated for the "
            f"current deployment composition: {config.role!r}")
    if (scheduler_config.pipeline_parallel_size != 1
            and "VLLM_PP_LAYER_PARTITION" in os.environ):
        raise NotImplementedError(
            "the current vLLM PP Scheduler control-plane baseline requires "
            "the audited uniform layer partition and rejects "
            "VLLM_PP_LAYER_PARTITION")
    if (scheduler_config.tensor_parallel_size not in (1, 2)
            and not target_scope):
        raise NotImplementedError(
            "the current vLLM Scheduler control-plane baseline supports "
            "only TP=1 or independently validated TP=2")
    if (not target_scope
            and scheduler_config.tensor_parallel_size != 1
            and scheduler_config.pipeline_parallel_size != 1
            and (scheduler_config.tensor_parallel_size,
                 scheduler_config.pipeline_parallel_size) != (2, 2)):
        raise NotImplementedError(
            "the current vLLM Scheduler control-plane baseline combines "
            "only the validated TP=2, PP=2 slice")
    if (not target_scope
            and scheduler_config.enable_expert_parallel and (
            scheduler_config.tensor_parallel_size != 2
            or scheduler_config.pipeline_parallel_size != 1)):
        raise NotImplementedError(
            "the validated vLLM EP Scheduler slice requires TP=2 and PP=1")
    standalone_af_tp2 = (
        config.role == "attention"
        and scheduler_config.tensor_parallel_size == 2
        and scheduler_config.pipeline_parallel_size == 1
        and not scheduler_config.enable_expert_parallel
        and scheduler_config.mtp_num_speculative_tokens == 0
    )
    standalone_af_ep2 = (
        config.role == "attention"
        and scheduler_config.tensor_parallel_size == 2
        and scheduler_config.pipeline_parallel_size == 1
        and scheduler_config.enable_expert_parallel
        and scheduler_config.mtp_num_speculative_tokens == 0
        and config.max_batched_tokens == 8
        and config.max_requests == 1
        and config.num_kv_blocks == 8
        and config.kv_block_size_tokens == 16
        and scheduler_config.max_model_len == 128
    )
    if ((scheduler_config.tensor_parallel_size != 1
         or scheduler_config.pipeline_parallel_size != 1)
            and config.role != "colocated"
            and not standalone_af_tp2
            and not standalone_af_ep2
            and not target_scope):
        raise NotImplementedError(
            "the current vLLM parallel Scheduler control-plane baseline supports "
            "only colocated deployment or the validated standalone AF "
            "Attention TP=2/EP=2 slices")
    if ((scheduler_config.tensor_parallel_size != 1
         or scheduler_config.pipeline_parallel_size != 1)
            and scheduler_config.mtp_num_speculative_tokens != 0
            and not target_scope):
        raise NotImplementedError(
            "the current vLLM MTP baseline requires TP=PP=1")
    dense_colocated_block128 = (
        config.kv_block_size_tokens == 128
        and config.role == "colocated"
        and scheduler_config.tensor_parallel_size == 1
        and scheduler_config.pipeline_parallel_size == 1
        and not scheduler_config.enable_expert_parallel
        and scheduler_config.mtp_num_speculative_tokens == 0
        and scope_model.model_type == "qwen3"
        and scope_model.attention_layout == "standard_gqa"
        and scope_model.num_experts == 0
        and scope_model.compute_dtype == scope_model.weight_dtype == "bf16"
    )
    # The CPU Scheduler and FullAttentionSpec receive the requested geometry
    # directly; no GPU worker or version-specific attention backend is selected.
    # Keep all other nonbaseline block geometries outside this validated slice.
    if config.kv_block_size_tokens != 16 and not dense_colocated_block128:
        raise NotImplementedError(
            "the pinned CPU-only vLLM Scheduler requires a 16-token KV block "
            "size outside the Qwen3 BF16 colocated TP1 PP1 MTP0 block128 slice")
    if ((config.is_decode_only or config.role == "attention")
            and scheduler_config.mtp_num_speculative_tokens != 0):
        if config.max_requests != 1:
            if config.is_decode_only:
                if not scheduler_config.authorized_h200_pd_mtp_sweep:
                    raise NotImplementedError(
                        "the seeded decode-only vLLM MTP baseline requires "
                        "max_requests=1")
            else:
                af_preemption_scope = (
                    config.max_requests == 2
                    and config.max_batched_tokens == 64
                    and config.num_kv_blocks == 6
                    and scheduler_config.max_model_len == 128
                    and scheduler_config.mtp_num_speculative_tokens == 2
                    and scheduler_config.mtp_acceptance_policy == "fixed_count"
                    and scheduler_config.mtp_fixed_accepted_tokens == 1
                    and scheduler_config.mtp_acceptance_replay is None
                    and scheduler_config.request_stream_sha256 is not None
                )
                if not af_preemption_scope:
                    raise NotImplementedError(
                        "the standalone AF attention vLLM MTP baseline requires "
                        "either "
                        "max_requests=1 or the exact hash-bound AF preemption "
                        "scope")
        if config.is_decode_only:
            if scheduler_config.mtp_acceptance_policy not in (
                    "fixed_count", "maximum_entropy", "replay"):
                raise NotImplementedError(
                    "the seeded decode-only vLLM MTP baseline supports only "
                    "fixed_count, maximum_entropy, or replay acceptance")
            replay = scheduler_config.mtp_acceptance_replay
            if (scheduler_config.mtp_acceptance_policy == "replay"
                    and (replay is None
                         or replay.schema_version != 2
                         or replay.scheduler_context.adapter_contract
                         != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT)):
                raise NotImplementedError(
                    "decode-only vLLM MTP replay requires the PD decode "
                    "replay schema v2 adapter contract")
        elif (scheduler_config.mtp_acceptance_policy != "fixed_count"
              or scheduler_config.mtp_acceptance_replay is not None):
            raise NotImplementedError(
                "standalone AF attention vLLM MTP currently requires "
                "fixed_count acceptance without a replay artifact")
    elif (config.role != "colocated"
          and scheduler_config.mtp_num_speculative_tokens != 0):
        raise NotImplementedError(
            "non-colocated vLLM MTP is validated only for the decode role "
            "or standalone AF attention role")


def _validate_ep_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    if not scheduler_config.enable_expert_parallel:
        return
    if model.model_type in TARGET_MODEL_PRODUCTION_PARALLELISM:
        target_parallel = TARGET_MODEL_PRODUCTION_PARALLELISM[model.model_type]
    elif model.model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        target_parallel = None
    else:
        raise AssertionError(
            f"unhandled Scheduler model type: {model.model_type!r}")
    if target_parallel is not None and (
            scheduler_config.tensor_parallel_size,
            scheduler_config.pipeline_parallel_size) in (target_parallel, (8, 1)):
        return
    if not is_validated_qwen3_moe_ep2_model_scope(model):
        raise NotImplementedError(
            "the validated vLLM EP Scheduler slice requires BF16 Qwen3MoE "
            "with an even expert count, valid top-k, all-sparse decoder "
            "layers, and no shared experts")


def _validate_tp2_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    if (scheduler_config.tensor_parallel_size != 2
            or scheduler_config.pipeline_parallel_size != 1
            or scheduler_config.enable_expert_parallel):
        return
    if model.model_type == "qwen3":
        if model.compute_dtype == "bf16":
            scheduler_dtype = "bfloat16"
        elif model.compute_dtype == "fp16":
            scheduler_dtype = "float16"
        else:
            raise NotImplementedError(
                f"unsupported Qwen3 compute dtype: {model.compute_dtype!r}"
            )
        supported_model = (
            model.compute_dtype in ("bf16", "fp16")
            and model.weight_dtype == model.compute_dtype
            and scheduler_config.dtype == scheduler_dtype
        )
    elif model.model_type == "llama":
        supported_model = (
            is_validated_llama_tp2_model_scope(model)
            and model.compute_dtype == "bf16"
            and model.weight_dtype == "bf16"
            and scheduler_config.dtype == "bfloat16"
        )
    else:
        raise NotImplementedError(
            f"unsupported TP2 Scheduler model type: {model.model_type!r}"
        )
    if (not supported_model
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"):
        raise NotImplementedError(
            "the validated vLLM TP=2 Scheduler slice requires dense BF16 or "
            "FP16 Qwen3 with matching model and Scheduler dtypes, or the "
            "exact source-backed Llama fixture in BF16, with standard GQA and "
            "expert parallelism disabled")


def _validate_tp2_pp2_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    if (scheduler_config.tensor_parallel_size,
            scheduler_config.pipeline_parallel_size) != (2, 2):
        return
    if (scheduler_config.enable_expert_parallel
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or scheduler_config.dtype != "bfloat16"):
        raise NotImplementedError(
            "the validated vLLM TP=2, PP=2 Scheduler slice requires dense "
            "BF16 Qwen3 with standard GQA and expert parallelism disabled")


def _validate_tp1_pp_scheduler_model_scope(
    scheduler_config: VLLMSchedulerConfig,
    model,
) -> None:
    pipeline_size = scheduler_config.pipeline_parallel_size
    if (pipeline_size == 1
            or scheduler_config.tensor_parallel_size != 1):
        return
    if model.model_type == "kimi_k2":
        target_pipeline_size = 16
    elif model.model_type == "glm_moe_dsa":
        target_pipeline_size = 32
    elif model.model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        target_pipeline_size = None
    else:
        raise AssertionError(
            f"unhandled Scheduler model type: {model.model_type!r}")
    if target_pipeline_size is not None:
        if (pipeline_size != target_pipeline_size
                or scheduler_config.enable_expert_parallel
                or model.compute_dtype != "bf16"
                or scheduler_config.dtype != "bfloat16"
                or scheduler_config.mtp_num_speculative_tokens != 0):
            raise NotImplementedError(
                "target Scheduler requires Kimi-K2.6 TP1/PP16 or "
                "GLM-5.2 TP1/PP32 with EP and MTP disabled")
        return
    if (scheduler_config.tensor_parallel_size != 1
            or scheduler_config.enable_expert_parallel
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.tie_word_embeddings
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or scheduler_config.dtype != "bfloat16"):
        raise NotImplementedError(
            "the validated vLLM TP=1 uniform PP Scheduler path requires "
            "untied dense BF16 Qwen3 with standard GQA and expert "
            "parallelism disabled")
    if model.num_hidden_layers % pipeline_size:
        raise NotImplementedError(
            "the validated vLLM TP=1 uniform PP Scheduler path requires "
            "model layers must be divisible by pipeline parallel size: "
            f"layers={model.num_hidden_layers}, PP={pipeline_size}")


def _construct_native_scheduler(
    config: BackendConfig,
    scheduler_config: VLLMSchedulerConfig,
):
    from src.serving.backend.vllm_scheduler_process import (
        _WORKER_ENVIRONMENT_FLAG,
    )
    import os

    if (_WORKER_ENVIRONMENT_FLAG not in os.environ
            or os.environ[_WORKER_ENVIRONMENT_FLAG] != "1"):
        raise RuntimeError(
            "native vLLM Scheduler construction is restricted to the "
            "isolated CPU worker")
    try:
        import torch
        from vllm.config import (
            CacheConfig,
            ParallelConfig,
            SchedulerConfig,
            StructuredOutputsConfig,
        )
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MLAAttentionSpec,
            SlidingWindowSpec,
        )
        from vllm.v1.structured_output import StructuredOutputManager
    except ImportError as exc:
        raise RuntimeError(
            "vLLMBackend.from_model requires a vLLM v1 installation"
        ) from exc

    from src.serving.workload.model_config import load_model_config

    model = load_model_config(scheduler_config.model)
    _validate_ep_scheduler_model_scope(scheduler_config, model)
    _validate_tp2_scheduler_model_scope(scheduler_config, model)
    _validate_tp2_pp2_scheduler_model_scope(scheduler_config, model)
    _validate_tp1_pp_scheduler_model_scope(scheduler_config, model)
    if scheduler_config.dtype == "bfloat16":
        requested_compute_dtype = "bf16"
        torch_dtype = torch.bfloat16
    elif scheduler_config.dtype == "float16":
        requested_compute_dtype = "fp16"
        torch_dtype = torch.float16
    else:
        raise AssertionError(
            f"unhandled validated vLLM dtype: {scheduler_config.dtype!r}")
    if model.compute_dtype != requested_compute_dtype:
        raise ValueError(
            "vLLM requested dtype disagrees with the canonical model "
            f"config: requested={scheduler_config.dtype!r}, "
            f"model={model.compute_dtype!r}")

    effective_max_model_len = scheduler_config.max_model_len
    if effective_max_model_len is None:
        effective_max_model_len = model.max_position_embeddings
    if effective_max_model_len > model.max_position_embeddings:
        raise ValueError(
            "vLLM max_model_len exceeds the canonical model limit: "
            f"requested={effective_max_model_len}, "
            f"model={model.max_position_embeddings}")

    from vllm.platforms import current_platform
    if not current_platform.is_cpu():
        raise RuntimeError(
            "isolated vLLM Scheduler worker did not select CpuPlatform")

    if (scheduler_config.mtp_num_speculative_tokens
            and model.mtp_architecture not in (
                "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp")):
        raise NotImplementedError(
            "the pinned vLLM MTP bridge requires a source-backed GLM or "
            "DeepSeek-V3 MTP architecture")
    native_mtp_engine_config = (
        scheduler_config.mtp_num_speculative_tokens > 0
        and model.mtp_architecture == "glm4_moe_mtp")
    dsv4_capacity_context = None
    if native_mtp_engine_config:
        from vllm.engine.arg_utils import EngineArgs

        engine_args = EngineArgs(
            model=scheduler_config.model,
            tensor_parallel_size=scheduler_config.tensor_parallel_size,
            pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
            dtype=scheduler_config.dtype,
            max_model_len=effective_max_model_len,
            gpu_memory_utilization=scheduler_config.gpu_memory_utilization,
            enforce_eager=True,
            trust_remote_code=scheduler_config.trust_remote_code,
            enable_chunked_prefill=scheduler_config.enable_chunked_prefill,
            enable_prefix_caching=(scheduler_config.prefix_cache_mode == "token_ids"),
            max_num_batched_tokens=config.max_batched_tokens,
            max_num_seqs=config.max_requests,
            block_size=config.kv_block_size_tokens,
            load_format="auto",
            skip_tokenizer_init=True,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": (
                    scheduler_config.mtp_num_speculative_tokens),
            },
        )
        vllm_config = engine_args.create_engine_config()
        native_model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        num_layers = native_model_config.get_num_layers(parallel_config)
        layer_start = 0
        layer_end = num_layers
        num_kv_heads = native_model_config.get_num_kv_heads(parallel_config)
        head_size = native_model_config.get_head_size()
        cache_dtype = native_model_config.dtype
        kv_spec_layout = "standard_gqa"
    else:
        if scheduler_config.trust_remote_code:
            raise NotImplementedError(
                "scheduler-only construction never executes remote model "
                "code; trust_remote_code must be false")
        if model.attention_layout == "standard_gqa":
            if model.weight_dtype != model.compute_dtype:
                raise NotImplementedError(
                    "scheduler-only standard-GQA construction does not yet "
                    "support mixed or quantized weight storage")
        elif model.attention_layout == "mla":
            if model.model_type == "kimi_k2":
                if (model.architecture != "DeepseekV3ForCausalLM"
                        or model.compute_dtype != "bf16"
                        or scheduler_config.dtype != "bfloat16"
                        or (scheduler_config.tensor_parallel_size,
                            scheduler_config.pipeline_parallel_size) not in (
                            (1, 1), (8, 2), (8, 1))
                        or scheduler_config.enable_expert_parallel
                        != (scheduler_config.tensor_parallel_size == 8)
                        or model.num_mtp_layers != 0):
                    raise NotImplementedError(
                        "the validated scheduler-only MLA slice supports exactly "
                        "Kimi-K2.6 BF16 compute with TP1/PP1, TP=EP=8/PP2, or "
                        "TP=EP=8/PP1, and MTP disabled")
            elif model.model_type == "deepseek_v3":
                dsv3_parallel_scope = (
                    (scheduler_config.tensor_parallel_size,
                     scheduler_config.pipeline_parallel_size,
                     scheduler_config.enable_expert_parallel,
                     config.role)
                    in (
                        (1, 1, False, "colocated"),
                        (8, 1, False, "colocated"),
                        (8, 1, False, "prefill"),
                        (8, 1, False, "decode"),
                    )
                    or (
                        # 32-layer local variant measured on 4x H200.
                        model.config_sha256 == DEEPSEEK_V3_L32_CONFIG_SHA256
                        and (scheduler_config.tensor_parallel_size,
                             scheduler_config.pipeline_parallel_size,
                             scheduler_config.enable_expert_parallel,
                             config.role) == (4, 1, False, "colocated")
                    )
                )
                if (model.architecture != "DeepseekV3ForCausalLM"
                        or model.compute_dtype != "bf16"
                        or model.weight_dtype != "fp8"
                        or scheduler_config.dtype != "bfloat16"
                        or not dsv3_parallel_scope
                        or model.mtp_architecture != "deepseek_mtp"
                        or model.num_mtp_layers != 1):
                    raise NotImplementedError(
                        "the validated scheduler-only MLA slice supports exactly "
                        "DeepSeek-V3 FP8 with TP1/PP1 colocated or H200 "
                        "TP8/PP1/EP1 colocated/prefill/decode, and its "
                        "single source-backed MTP layer")
            else:
                raise AssertionError(
                    f"unhandled MLA Scheduler model type: "
                    f"{model.model_type!r}")
        elif model.attention_layout == "dsa_mla":
            dsa_parallel_scope = (
                (scheduler_config.tensor_parallel_size,
                 scheduler_config.pipeline_parallel_size,
                 scheduler_config.enable_expert_parallel,
                 config.role)
                in (
                    (1, 1, False, "colocated"),
                    (8, 4, True, "colocated"),
                    (8, 1, False, "colocated"),
                    (8, 1, True, "prefill"),
                    (8, 1, True, "decode"),
                    (8, 1, False, "prefill"),
                    (8, 1, False, "decode"),
                )
            )
            if (model.model_type != "glm_moe_dsa"
                    or model.architecture != "GlmMoeDsaForCausalLM"
                    or model.compute_dtype != "bf16"
                    or model.weight_dtype not in ("bf16", "fp8")
                    or scheduler_config.dtype != "bfloat16"
                    or not dsa_parallel_scope
                    or (scheduler_config.mtp_num_speculative_tokens > 0
                        and (model.mtp_architecture != "glm_moe_dsa_mtp"
                             or model.num_mtp_layers != 1))):
                raise NotImplementedError(
                    "the validated scheduler-only DSA slice supports exactly "
                    "GLM-5.2 BF16 with TP1/PP1 colocated, TP=EP=8/PP4 "
                    "colocated, H200 aggregate TP8/PP1/EP1 colocated, or "
                    "H200 PD roles TP8/PP1 with EP8 or EP1, and its single "
                    "source-backed MTP layer")
        elif model.attention_layout == "dsv4_sparse_mla":
            dsv4_parallel_scope = (
                (model.config_sha256 == DEEPSEEK_V4_FLASH_CONFIG_SHA256
                 and (scheduler_config.tensor_parallel_size,
                      scheduler_config.pipeline_parallel_size,
                      scheduler_config.enable_expert_parallel,
                      config.role) == (4, 1, False, "colocated"))
                or (model.config_sha256 == DEEPSEEK_V4_PRO_CONFIG_SHA256
                    and (scheduler_config.tensor_parallel_size,
                         scheduler_config.pipeline_parallel_size,
                         scheduler_config.enable_expert_parallel,
                         config.role) == (8, 1, False, "colocated"))
            )
            if (model.model_type != "deepseek_v4"
                    or model.architecture != "DeepseekV4ForCausalLM"
                    or model.compute_dtype != "bf16"
                    or model.weight_dtype != "fp8"
                    or scheduler_config.dtype != "bfloat16"
                    or not dsv4_parallel_scope
                    or scheduler_config.mtp_num_speculative_tokens > 0):
                raise NotImplementedError(
                    "the validated scheduler-only DeepSeek-V4 sparse-MLA "
                    "slice supports exactly DeepSeek-V4-Flash FP8 with "
                    "TP4/PP1 colocated or DeepSeek-V4-Pro FP8 with TP8/PP1 "
                    "colocated on H200, with MTP disabled")
        else:
            raise AssertionError(
                f"unhandled attention layout: {model.attention_layout!r}")
        native_scheduler_config = SchedulerConfig(
            runner_type="generate",
            max_num_batched_tokens=config.max_batched_tokens,
            max_num_seqs=config.max_requests,
            max_model_len=effective_max_model_len,
            max_num_partial_prefills=1,
            max_long_partial_prefills=1,
            long_prefill_token_threshold=0,
            enable_chunked_prefill=(
                scheduler_config.enable_chunked_prefill),
            is_multimodal_model=False,
            policy="fcfs",
            num_lookahead_slots=(
                scheduler_config.mtp_num_speculative_tokens),
            disable_chunked_mm_input=False,
            disable_hybrid_kv_cache_manager=False,
            async_scheduling=scheduler_config.async_scheduling,
        )
        cache_config = CacheConfig(
            block_size=config.kv_block_size_tokens,
            gpu_memory_utilization=scheduler_config.gpu_memory_utilization,
            cache_dtype="auto",
            is_attention_free=False,
            sliding_window=None,
            enable_prefix_caching=(scheduler_config.prefix_cache_mode == "token_ids"),
        )
        parallel_config = ParallelConfig(
            tensor_parallel_size=scheduler_config.tensor_parallel_size,
            pipeline_parallel_size=scheduler_config.pipeline_parallel_size,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
            enable_expert_parallel=(
                scheduler_config.enable_expert_parallel),
            enable_eplb=False,
            distributed_executor_backend="uni",
            decode_context_parallel_size=1,
        )
        if (scheduler_config.pipeline_parallel_size > 1
                and model.model_type not in ("kimi_k2", "glm_moe_dsa")):
            from vllm.engine.arg_utils import EngineArgs

            native_model_config = EngineArgs(
                model=scheduler_config.model,
                tensor_parallel_size=scheduler_config.tensor_parallel_size,
                pipeline_parallel_size=(
                    scheduler_config.pipeline_parallel_size),
                dtype=scheduler_config.dtype,
                max_model_len=effective_max_model_len,
                trust_remote_code=False,
                load_format="auto",
                skip_tokenizer_init=True,
            ).create_model_config()
            native_model_config.verify_with_parallel_config(parallel_config)
        if (model.num_attention_heads
                % scheduler_config.tensor_parallel_size):
            raise ValueError(
                "model attention heads must be divisible by tensor parallel "
                "size: "
                f"{model.num_attention_heads} vs "
                f"{scheduler_config.tensor_parallel_size}")
        if (model.attention_layout == "standard_gqa"
                and model.num_hidden_layers
                % scheduler_config.pipeline_parallel_size):
            raise ValueError(
                "model layers must be divisible by pipeline parallel size: "
                f"{model.num_hidden_layers} vs "
                f"{scheduler_config.pipeline_parallel_size}")
        layer_ranges = pipeline_layer_ranges_for_model(
            model, scheduler_config.pipeline_parallel_size)
        if model.attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 is PP=1-only (rank-specific compress-ratio layer
            # placement), so the single stage is trivially representative.
            representative_stage = 0
        else:
            representative_stage = max(
                range(len(layer_ranges)),
                key=lambda stage: model.kv_bytes_per_token_for_layer_range(
                    scheduler_config.tensor_parallel_size,
                    *layer_ranges[stage],
                ),
            )
        layer_start, layer_end = layer_ranges[representative_stage]
        num_layers = layer_end - layer_start
        model_view = _SchedulerModelView(
            max_model_len=effective_max_model_len,
            vocab_size=model.vocab_size,
            dtype=torch_dtype,
        )
        if scheduler_config.mtp_num_speculative_tokens:
            speculative_config = _SchedulerSpeculativeConfig(
                method="mtp",
                num_speculative_tokens=(
                    scheduler_config.mtp_num_speculative_tokens),
                draft_model_config=_SchedulerDraftModelConfig(
                    hf_config=_SchedulerDraftHFConfig(
                        model_type=model.mtp_architecture,
                        num_nextn_predict_layers=model.num_mtp_layers,
                    ),
                    max_model_len=effective_max_model_len,
                ),
            )
        else:
            speculative_config = None
        vllm_config = _SchedulerOnlyVllmConfig(
            model_config=model_view,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=native_scheduler_config,
            structured_outputs_config=StructuredOutputsConfig(),
            kv_geometry_representative_pipeline_stage_rank=(
                representative_stage),
            kv_geometry_representative_layer_start=layer_start,
            kv_geometry_representative_layer_end=layer_end,
            speculative_config=speculative_config,
        )
        if model.attention_layout == "standard_gqa":
            num_kv_heads = model.kv_heads_per_tensor_parallel_rank(
                scheduler_config.tensor_parallel_size)
            head_size = model.head_dim
            kv_spec_layout = "standard_gqa"
        elif model.attention_layout == "mla":
            num_kv_heads = 1
            head_size = model.kv_lora_rank + model.qk_rope_head_dim
            kv_spec_layout = "mla"
        elif model.attention_layout == "dsa_mla":
            # The pinned control-plane Scheduler only consumes block counts.
            # Represent DSA as one manager group whose page is the exact sum
            # of every compressed-MLA and full-indexer cache page; the tensor
            # list below still preserves the real per-layer ownership/bytes.
            dsa_bytes_per_token = (
                model.kv_bytes_per_token_for_layer_range(
                    scheduler_config.tensor_parallel_size,
                    layer_start,
                    layer_end,
                ))
            if dsa_bytes_per_token % model.dtype_bytes:
                raise RuntimeError(
                    "GLM-5.2 DSA KV bytes must be representable by the "
                    "Scheduler aggregate cache dtype")
            num_kv_heads = 1
            head_size = dsa_bytes_per_token // model.dtype_bytes
            kv_spec_layout = "dsa_aggregate"
        elif model.attention_layout == "dsv4_sparse_mla":
            # DeepSeek-V4 decomposes into per-lifecycle Scheduler cache
            # groups (ModelConfig.dsv4_scheduler_cache_groups): windowed
            # caches (SWA window, FP32 compressor partial states) become
            # SlidingWindow groups whose per-request footprint caps at the
            # upstream window, and unbounded streams (compressed KV,
            # indexer K) become FullAttention groups with the upstream
            # 576-byte-aligned page geometry.  With prefix caching disabled
            # the pinned 0.11.0 coordinator imposes no group-type
            # constraints, so each group gets its real manager lifecycle.
            dsv4_groups = model.dsv4_scheduler_cache_groups(
                config.kv_block_size_tokens)
            dsv4_capacity_context = (dsv4_groups, model)
            num_kv_heads = 1
            head_size = 0
            kv_spec_layout = "dsv4_lifecycle"
        else:
            raise AssertionError(
                "validated scheduler layout changed before KV construction")
        cache_dtype = torch_dtype

    if not isinstance(vllm_config.model_config.dtype, torch.dtype):
        raise TypeError(
            "vLLM model_config.dtype must be a torch.dtype, got "
            f"{type(vllm_config.model_config.dtype).__name__}")
    if kv_spec_layout == "standard_gqa":
        cache_spec = FullAttentionSpec(
            block_size=config.kv_block_size_tokens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=cache_dtype,
        )
    elif kv_spec_layout in ("mla", "dsa_aggregate"):
        cache_spec = MLAAttentionSpec(
            block_size=config.kv_block_size_tokens,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=cache_dtype,
            cache_dtype_str=None,
        )
    elif kv_spec_layout == "dsv4_lifecycle":
        cache_spec = None
    else:
        raise AssertionError(
            f"unhandled Scheduler KV spec layout: {kv_spec_layout!r}")
    if kv_spec_layout == "dsv4_lifecycle":
        # One Scheduler group per cache lifecycle class; virtual layer names
        # carry the component kind so each physical layer's caches join the
        # groups of their real lifecycle (review-06 R1).
        dsv4_kv_group_specs = []
        dsv4_kv_tensors = []
        for group in dsv4_groups:
            if group.lifecycle == "sliding_window":
                # SlidingWindowSpec pages charge the K+V pair
                # (2 * block * head * dtype); the FP32 state pages divide
                # exactly, the SWA row is ceiled to an even 586 bytes in the
                # ModelConfig group table.
                if group.page_bytes % (2 * group.block_tokens):
                    raise RuntimeError(
                        "DeepSeek-V4 sliding cache group page must cover "
                        f"whole K+V token bytes: kind={group.kind!r}, "
                        f"page={group.page_bytes}, "
                        f"block_tokens={group.block_tokens}")
                group_spec = SlidingWindowSpec(
                    block_size=group.block_tokens,
                    num_kv_heads=1,
                    head_size=group.page_bytes // (2 * group.block_tokens),
                    dtype=torch.uint8,
                    sliding_window=group.window_tokens,
                )
            elif group.lifecycle == "stream":
                # MLAAttentionSpec pages charge the single compressed
                # latent row (block * head * dtype), matching the upstream
                # compressed-KV/indexer-K storage.
                if group.page_bytes % group.block_tokens:
                    raise RuntimeError(
                        "DeepSeek-V4 stream cache group page must cover "
                        f"whole token bytes: kind={group.kind!r}, "
                        f"page={group.page_bytes}, "
                        f"block_tokens={group.block_tokens}")
                group_spec = MLAAttentionSpec(
                    block_size=group.block_tokens,
                    num_kv_heads=1,
                    head_size=group.page_bytes // group.block_tokens,
                    dtype=torch.uint8,
                    cache_dtype_str=None,
                )
            else:
                raise AssertionError(
                    f"unhandled DeepSeek-V4 cache lifecycle: "
                    f"{group.lifecycle!r}")
            if group_spec.page_size_bytes != group.page_bytes:
                raise RuntimeError(
                    "DeepSeek-V4 Scheduler spec page disagrees with the "
                    f"ModelConfig cache group: kind={group.kind!r}, "
                    f"spec={group_spec.page_size_bytes}, "
                    f"model={group.page_bytes}")
            group_layer_names = [
                f"model.layers.{index}.self_attn.attn:{group.kind}"
                for index in group.layer_indices
            ]
            dsv4_kv_group_specs.append(KVCacheGroupSpec(
                layer_names=group_layer_names, kv_cache_spec=group_spec))
            dsv4_kv_tensors.extend(
                KVCacheTensor(
                    size=config.num_kv_blocks * group.page_bytes,
                    shared_by=[layer_name],
                )
                for layer_name in group_layer_names
            )
        kv_cache_config = KVCacheConfig(
            num_blocks=config.num_kv_blocks,
            kv_cache_tensors=dsv4_kv_tensors,
            kv_cache_groups=dsv4_kv_group_specs,
        )
        layer_names = [
            layer_name
            for group in dsv4_kv_group_specs
            for layer_name in group.layer_names
        ]
        page_bytes = None
    elif model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3",
                            "deepseek_v4"):
        layer_names = [
            f"model.layers.{index}.self_attn.attn"
            for index in range(layer_start, layer_end)
        ]
    elif model.model_type in ("qwen3", "qwen3_moe", "llama", "glm4_moe"):
        layer_names = [
            f"model.layers.{index}.self_attn.attn"
            for index in range(num_layers)
        ]
    else:
        raise AssertionError(
            f"unhandled Scheduler model type: {model.model_type!r}")
    if kv_spec_layout != "dsv4_lifecycle":
        page_bytes = cache_spec.page_size_bytes
    if kv_spec_layout == "dsa_aggregate":
        compressed_page_bytes = (
            config.kv_block_size_tokens
            * (model.kv_lora_rank + model.qk_rope_head_dim)
            * model.dtype_bytes
        )
        index_page_bytes = config.kv_block_size_tokens * (
            model.index_head_dim + model.index_head_dim // 128 * 4)
        dsa_tensor_page_bytes = []
        for indexer_type in model.indexer_types[layer_start:layer_end]:
            if indexer_type == "full":
                layer_page_bytes = (
                    compressed_page_bytes + index_page_bytes)
            elif indexer_type == "shared":
                layer_page_bytes = compressed_page_bytes
            else:
                raise AssertionError(
                    f"unhandled GLM-5.2 indexer type: {indexer_type!r}")
            dsa_tensor_page_bytes.append(layer_page_bytes)
        tensor_page_bytes = tuple(dsa_tensor_page_bytes)
        if sum(tensor_page_bytes) != page_bytes:
            raise RuntimeError(
                "GLM-5.2 per-layer DSA cache tensors disagree with the "
                "aggregate Scheduler cache spec")
    elif kv_spec_layout in ("standard_gqa", "mla"):
        tensor_page_bytes = (page_bytes,) * len(layer_names)
    elif kv_spec_layout == "dsv4_lifecycle":
        tensor_page_bytes = None
    else:
        raise AssertionError(
            f"unhandled Scheduler tensor layout: {kv_spec_layout!r}")
    if kv_spec_layout != "dsv4_lifecycle":
        kv_cache_config = KVCacheConfig(
            num_blocks=config.num_kv_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=config.num_kv_blocks * layer_page_bytes,
                    shared_by=[layer_name],
                )
                for layer_name, layer_page_bytes in zip(
                    layer_names, tensor_page_bytes, strict=True)
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=layer_names, kv_cache_spec=cache_spec)
            ],
        )
    actual_tensor_owners = tuple(
        tuple(tensor.shared_by)
        for tensor in kv_cache_config.kv_cache_tensors
    )
    expected_tensor_owners = tuple((layer_name,) for layer_name in layer_names)
    if actual_tensor_owners != expected_tensor_owners:
        raise RuntimeError(
            "vLLM Scheduler KV tensors must be owned one per layer: "
            f"actual={actual_tensor_owners!r}, "
            f"expected={expected_tensor_owners!r}")
    if kv_spec_layout != "dsv4_lifecycle":
        if page_bytes % config.kv_block_size_tokens:
            raise RuntimeError(
                "vLLM Scheduler KV page bytes must divide by block tokens")
    total_tensor_bytes = sum(
        tensor.size for tensor in kv_cache_config.kv_cache_tensors)
    capacity_denominator = (
        config.num_kv_blocks * config.kv_block_size_tokens)
    if model.attention_layout == "dsv4_sparse_mla":
        # The lifecycle groups no longer have a uniform per-token byte rate:
        # windowed caches cap per request at their window, streams charge
        # 576-aligned pages.  The constructed tensors are validated
        # component-by-component against the ModelConfig group table above;
        # BackendConfig keeps the conservative ceiled full-rate per-token
        # total as the cross-lane reporting charge (evaluation computes it
        # from the same ModelConfig Fraction).
        model_kv_bytes_per_token = (
            model.kv_bytes_per_token_per_parallel_rank(
                scheduler_config.tensor_parallel_size,
                scheduler_config.pipeline_parallel_size))
        model_kv_bytes_per_token_ceiled = (
            -(-model_kv_bytes_per_token.numerator
              // model_kv_bytes_per_token.denominator))
        if config.kv_bytes_per_token != model_kv_bytes_per_token_ceiled:
            raise RuntimeError(
                "BackendConfig KV bytes disagree with the ceiled DeepSeek-V4 "
                "per-rank attention-cache capacity: "
                f"backend={config.kv_bytes_per_token}, "
                f"model={model_kv_bytes_per_token_ceiled}")
    else:
        if total_tensor_bytes % capacity_denominator:
            raise RuntimeError(
                "vLLM Scheduler KV tensor bytes must divide by its block "
                "capacity")
        constructed_kv_bytes_per_token = (
            total_tensor_bytes // capacity_denominator)
        model_kv_bytes_per_token = model.kv_bytes_per_token_for_layer_range(
            scheduler_config.tensor_parallel_size,
            layer_start,
            layer_end,
        )
        if constructed_kv_bytes_per_token != model_kv_bytes_per_token:
            raise RuntimeError(
                "vLLM Scheduler KV spec disagrees with ModelConfig capacity: "
                f"constructed={constructed_kv_bytes_per_token}, "
                f"model={model_kv_bytes_per_token}")
        if config.kv_bytes_per_token != model_kv_bytes_per_token:
            raise RuntimeError(
                "BackendConfig KV bytes disagree with the representative "
                "maximum-KV-geometry pipeline stage: "
                f"backend={config.kv_bytes_per_token}, "
                f"model={model_kv_bytes_per_token}")
    vllm_config.cache_config.num_gpu_blocks = config.num_kv_blocks
    vllm_config.cache_config.num_gpu_blocks_override = config.num_kv_blocks
    _validate_effective_scheduler_config(
        vllm_config,
        config,
        scheduler_config,
    )
    scheduler_class = (
        AsyncScheduler if scheduler_config.async_scheduling else Scheduler
    )
    scheduler = scheduler_class(
        vllm_config,
        kv_cache_config,
        StructuredOutputManager(vllm_config),
    )
    return scheduler, vllm_config, dsv4_capacity_context


def _validate_effective_scheduler_config(
    vllm_config: Any,
    config: BackendConfig,
    requested: VLLMSchedulerConfig,
) -> None:
    """Fail if pinned vLLM rewrites a scheduler limit outside our contract."""

    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    if requested.enable_chunked_prefill is not True:
        raise RuntimeError("requested vLLM chunked prefill must be true")
    if scheduler.enable_chunked_prefill is not True:
        raise RuntimeError("vLLM effective enable_chunked_prefill is not true")
    if scheduler.chunked_prefill_enabled is not True:
        raise RuntimeError("vLLM effective chunked_prefill_enabled is not true")
    if scheduler.max_num_batched_tokens != config.max_batched_tokens:
        raise RuntimeError(
            "vLLM rewrote max_num_batched_tokens: "
            f"requested={config.max_batched_tokens}, "
            f"effective={scheduler.max_num_batched_tokens}")
    if scheduler.max_num_seqs != config.max_requests:
        raise RuntimeError(
            "vLLM rewrote max_num_seqs: "
            f"requested={config.max_requests}, "
            f"effective={scheduler.max_num_seqs}")
    expected_max_model_len = requested.max_model_len
    if expected_max_model_len is None:
        expected_max_model_len = vllm_config.model_config.max_model_len
    if scheduler.max_model_len != expected_max_model_len:
        raise RuntimeError(
            "vLLM rewrote max_model_len: "
            f"expected={expected_max_model_len}, "
            f"effective={scheduler.max_model_len}")
    if scheduler.policy != "fcfs":
        raise RuntimeError(
            f"vLLM scheduler policy must be 'fcfs', got {scheduler.policy!r}")
    if (scheduler.max_num_partial_prefills != 1
            or scheduler.max_long_partial_prefills != 1
            or scheduler.long_prefill_token_threshold != 0):
        raise RuntimeError(
            "vLLM partial-prefill policy differs from the pinned baseline")
    if scheduler.disable_chunked_mm_input is not False:
        raise RuntimeError(
            "vLLM chunked multimodal input must remain disabled")
    if scheduler.async_scheduling is not requested.async_scheduling:
        raise RuntimeError(
            "vLLM effective async scheduling disagrees with the requested "
            "Scheduler control-plane mode")
    expected_lookahead = requested.mtp_num_speculative_tokens
    if scheduler.num_lookahead_slots != expected_lookahead:
        raise RuntimeError(
            "vLLM speculative lookahead disagrees with MTP config: "
            f"requested={expected_lookahead}, "
            f"effective={scheduler.num_lookahead_slots}")
    speculative = vllm_config.speculative_config
    if requested.mtp_num_speculative_tokens == 0:
        if speculative is not None:
            raise RuntimeError("vLLM speculative decoding must be disabled")
    else:
        if speculative is None:
            raise RuntimeError("vLLM dropped the requested MTP configuration")
        if speculative.method != "mtp":
            raise RuntimeError(
                "vLLM speculative method must remain 'mtp', got "
                f"{speculative.method!r}")
        if (speculative.num_speculative_tokens
                != requested.mtp_num_speculative_tokens):
            raise RuntimeError(
                "vLLM rewrote the requested MTP proposal count")
        if speculative.disable_padded_drafter_batch is not False:
            raise RuntimeError(
                "pinned vLLM MTP workload requires padded drafter batches")
        draft_model_config = speculative.draft_model_config
        if draft_model_config is None:
            raise RuntimeError("vLLM MTP requires an effective draft model")
        draft_hf_config = draft_model_config.hf_config
        if draft_hf_config.model_type not in (
                "glm4_moe_mtp", "glm_moe_dsa_mtp", "deepseek_mtp"):
            raise NotImplementedError(
                "Serving MTP workload requires a source-backed GLM or "
                "DeepSeek-V3 MTP architecture")
        if draft_hf_config.num_nextn_predict_layers != 1:
            raise NotImplementedError(
                "Serving GLM/DeepSeek MTP workload currently supports one "
                "MTP layer")
        if draft_hf_config.model_type == "glm4_moe_mtp":
            if draft_hf_config.num_hidden_layers != 0:
                raise RuntimeError(
                    "vLLM GLM4 MTP draft config must start at decoder layer "
                    "zero")
        elif draft_hf_config.model_type in ("glm_moe_dsa_mtp", "deepseek_mtp"):
            if not isinstance(
                    draft_model_config, _SchedulerDraftModelConfig):
                raise RuntimeError(
                    "GLM-5.2/DeepSeek-V3 MTP must use the scheduler-only "
                    "draft contract")
        else:
            raise AssertionError("validated MTP draft architecture changed")
        effective_draft_max_model_len = draft_model_config.max_model_len
        if effective_draft_max_model_len is None:
            effective_draft_max_model_len = vllm_config.model_config.max_model_len
        if (isinstance(effective_draft_max_model_len, bool)
                or not isinstance(effective_draft_max_model_len, int)
                or effective_draft_max_model_len <= 0):
            raise RuntimeError(
                "vLLM MTP draft max model length must be a positive integer")
    if vllm_config.kv_transfer_config is not None:
        raise RuntimeError("vLLM native KV transfer must be disabled")
    if cache.enable_prefix_caching != (requested.prefix_cache_mode == "token_ids"):
        raise RuntimeError("vLLM effective prefix caching disagrees with requested mode")
    if cache.block_size != config.kv_block_size_tokens:
        raise RuntimeError(
            "vLLM rewrote KV block size: "
            f"requested={config.kv_block_size_tokens}, "
            f"effective={cache.block_size}")
    if cache.gpu_memory_utilization != requested.gpu_memory_utilization:
        raise RuntimeError(
            "vLLM rewrote gpu_memory_utilization: "
            f"requested={requested.gpu_memory_utilization}, "
            f"effective={cache.gpu_memory_utilization}")
    if cache.num_gpu_blocks != config.num_kv_blocks:
        raise RuntimeError(
            "vLLM effective KV block count disagrees with BackendConfig")
    if cache.num_gpu_blocks_override != config.num_kv_blocks:
        raise RuntimeError(
            "vLLM KV block override disagrees with BackendConfig")
    if cache.cache_dtype != "auto":
        raise RuntimeError("vLLM KV cache dtype must remain model-derived")
    if cache.is_attention_free is not False:
        raise RuntimeError("vLLM text baseline requires attention KV cache")
    if cache.sliding_window is not None:
        raise NotImplementedError(
            "the vLLM scheduler baseline does not support sliding-window KV")
    expected_world_size = (
        requested.tensor_parallel_size * requested.pipeline_parallel_size)
    if (parallel.tensor_parallel_size != requested.tensor_parallel_size
            or parallel.pipeline_parallel_size
            != requested.pipeline_parallel_size
            or parallel.data_parallel_size != 1
            or parallel.data_parallel_rank != 0
            or parallel.decode_context_parallel_size != 1
            or parallel.world_size != expected_world_size):
        raise RuntimeError(
            "vLLM effective parallel config disagrees with the requested "
            "Scheduler control-plane scope")
    if (parallel.enable_expert_parallel
            is not requested.enable_expert_parallel):
        raise RuntimeError(
            "vLLM effective expert-parallel enablement disagrees with the "
            "requested Scheduler control-plane scope")


def _make_native_request(
    vllm_config: Any,
    request: RequestSpec,
    *,
    dummy_token_id: int,
    max_output_tokens: int | None = None,
    scheduler_version: str = "0.11.0",
):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    dummy_token_id = _vllm_integer(dummy_token_id, "dummy_token_id")
    if not hasattr(vllm_config.model_config, "get_vocab_size"):
        raise TypeError("vLLM model_config.get_vocab_size must be callable")
    get_vocab_size = vllm_config.model_config.get_vocab_size
    if not callable(get_vocab_size):
        raise TypeError("vLLM model_config.get_vocab_size must be callable")
    vocab_size = _vllm_integer(
        get_vocab_size(),
        "model vocab_size",
    )
    if vocab_size <= 0:
        raise ValueError("vLLM model vocab_size must be positive")
    if not 0 <= dummy_token_id < vocab_size:
        raise ValueError(
            "vLLM dummy_token_id must be within the model vocabulary")
    if max_output_tokens is not None:
        max_output_tokens = _vllm_integer(
            max_output_tokens, "max_output_tokens")
        if max_output_tokens <= 0:
            raise ValueError("vLLM max_output_tokens must be positive")
    output_limit = (
        request.output_tokens
        if max_output_tokens is None
        else max_output_tokens
    )
    block_hasher = None
    request.validate_token_identity(vocab_size, require_complete=vllm_config.cache_config.enable_prefix_caching)
    if vllm_config.cache_config.enable_prefix_caching:
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher
        if scheduler_version == "0.30.0":
            from vllm.utils.hashing import get_hash_fn_by_name
        elif scheduler_version == "0.11.0":
            from vllm.utils import get_hash_fn_by_name
        else:
            raise ValueError("unsupported vLLM prefix hash API version")
        hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
        block_hasher = get_request_block_hasher(vllm_config.cache_config.block_size, hash_fn)
    version_fields = {}
    if scheduler_version == "0.11.0":
        version_fields["eos_token_id"] = None
    elif scheduler_version != "0.30.0":
        raise ValueError("unsupported vLLM Request API version")
    return Request(
        request_id=request.request_id,
        prompt_token_ids=(list(request.prompt_token_ids) if request.prompt_token_ids is not None
                          else _LengthOnlyTokenIds(dummy_token_id, request.input_tokens)),
        sampling_params=SamplingParams(
            max_tokens=output_limit),
        pooling_params=None,
        arrival_time=request.arrival_time_s,
        block_hasher=block_hasher,
        **version_fields,
    )


def _vllm030_v2_resumed_request_ids(new_requests, cached_resumed_ids,
                                   status_before):
    """Read MRV2 resume ownership from native new records and prior states."""
    if cached_resumed_ids:
        raise RuntimeError("vLLM 0.30 MRV2 cannot retain cached resume flags")
    new_ids = tuple(_vllm_request_id(request.req_id) for request in new_requests)
    if len(new_ids) != len(set(new_ids)):
        raise ValueError("vLLM 0.30 MRV2 new request IDs contain duplicates")
    if not set(new_ids).issubset(status_before):
        raise ValueError("vLLM 0.30 MRV2 new record has an unknown request")
    return tuple(request_id for request_id in new_ids
                 if status_before[request_id] == "PREEMPTED")


def _vllm030_discard_sampled_tokens(*, computed_tokens, scheduled_tokens, num_tokens):
    """GMR _prepare_inputs discard mask, before sampling/output clipping.

    gpu_model_runner.py uses optimistic_seq_lens < CachedRequestState.num_tokens.
    The bridge's schedule base plus its physical row count is that same
    optimistic sequence length. Native Request.num_tokens excludes drafts.
    """
    computed = _vllm_integer(computed_tokens, "sample discard computed tokens")
    scheduled = _vllm_integer(scheduled_tokens, "sample discard scheduled tokens")
    boundary = _vllm_integer(num_tokens, "sample discard native token boundary")
    if computed < 0 or scheduled <= 0 or boundary < 0:
        raise ValueError("sample discard requires nonnegative native positions and positive scheduled rows")
    return computed + scheduled < boundary


def _resumed_native_request_ids(cached: Any, request_ids: tuple[str, ...],
                                scheduler_version: str) -> tuple[str, ...]:
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("vLLM cached request IDs contain duplicates")
    if scheduler_version == "0.30.0":
        resumed = cached.resumed_req_ids
        if not isinstance(resumed, set) or any(not isinstance(value, str) for value in resumed):
            raise TypeError("vLLM 0.30 resumed_req_ids must be a set of request IDs")
        if not resumed.issubset(request_ids):
            raise ValueError("vLLM 0.30 resumed an uncached request")
        return tuple(request_id for request_id in request_ids if request_id in resumed)
    if scheduler_version == "0.11.0":
        flags = tuple(cached.resumed_from_preemption)
        if len(request_ids) != len(flags):
            raise RuntimeError("vLLM returned inconsistent resumed-from-preemption metadata")
        if any(not isinstance(flag, bool) for flag in flags):
            raise TypeError("vLLM resumed-from-preemption flags must be boolean")
        return tuple(request_id for request_id, resumed in zip(request_ids, flags, strict=True) if resumed)
    raise ValueError("unsupported vLLM cached request API version")


def _verify_content_hash_state_disabled(native_request: Any,
                                        scheduler_version: str = "0.11.0") -> None:
    """Fail if a synthetic length-only request acquires content identity."""

    if native_request.block_hashes != []:
        raise RuntimeError(
            "length-only vLLM request unexpectedly contains block hashes")
    if scheduler_version == "0.11.0":
        hasher = native_request.get_hash_new_full_blocks
    elif scheduler_version == "0.30.0":
        hasher = native_request._block_hasher
    else:
        raise ValueError("unsupported vLLM request hashing API version")
    if hasher is not None:
        raise RuntimeError(
            "length-only vLLM request unexpectedly enables content hashing")


def _validated_mtp_draft_rows(frames, step_id, request_ids, depth, vocab_size):
    """Validate the actual executed proposer batch immediately before feedback."""
    if frames is None or step_id not in frames:
        raise ValueError(f"executed MTP proposer requires real draft IDs for step {step_id!r}")
    frame = frames[step_id]
    if not isinstance(frame, Mapping) or set(frame) != set(request_ids):
        raise ValueError("MTP draft request IDs must exactly match the executed proposer batch")
    if any(not isinstance(tokens, tuple) or len(tokens) != depth
            or any(type(token) is not int or token < 0 for token in tokens)
            for tokens in frame.values()):
        raise ValueError("MTP draft IDs must be nonnegative integer tuples covering proposer depth")
    if any(token >= vocab_size for tokens in frame.values() for token in tokens):
        raise ValueError("MTP draft IDs must be within the model vocabulary")
    return [list(frame[request_id]) for request_id in request_ids]


def _fake_model_runner_output(
    native_output: Any,
    generated_tokens: dict[str, int],
    dummy_token_id: int,
):
    from vllm.v1.outputs import ModelRunnerOutput

    request_ids = list(native_output.num_scheduled_tokens)
    if tuple(generated_tokens) != tuple(request_ids):
        raise ValueError(
            "generated_tokens must preserve native vLLM request order: "
            f"native={request_ids}, normalized={list(generated_tokens)}")
    scheduled_spec_tokens = native_output.scheduled_spec_decode_tokens
    sampled_token_ids = []
    for request_id in request_ids:
        generated = _vllm_integer(
            generated_tokens[request_id], "generated token count")
        draft_token_ids = (
            list(scheduled_spec_tokens[request_id])
            if request_id in scheduled_spec_tokens
            else []
        )
        if draft_token_ids:
            if not 1 <= generated <= len(draft_token_ids) + 1:
                raise ValueError(
                    "speculative vLLM output must contain an accepted draft "
                    "prefix followed by one terminal token")
            sampled_token_ids.append(
                draft_token_ids[:generated - 1] + [dummy_token_id])
        elif generated in (0, 1):
            sampled_token_ids.append([dummy_token_id] * generated)
        else:
            raise ValueError(
                "non-speculative vLLM steps must generate zero or one token")
    return ModelRunnerOutput(
        req_ids=request_ids,
        req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
