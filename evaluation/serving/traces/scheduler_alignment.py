"""Observation-conditioned validation of captured Scheduler decisions.

Compact Scheduler timestamps and canonical request arrivals do not currently
share a proven clock origin. The primary verdict is therefore explicitly
inconclusive for admission timing. A separate diagnostic admits each request
when it first appears in the observation and checks the pinned production
Scheduler policy without claiming an exact end-to-end replay.
"""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from evaluation.serving.metrics.sharegpt_compact_evaluation import (
    Phase1Cell,
    Phase1Config,
    compact_case_path,
    load_phase1_config,
    resolve_repository_input,
    select_phase1_cells,
)
from evaluation.serving.experiments.experiment_manifest import (
    RUNTIME_MEMORY_POLICY_PATH_TEMPLATE,
    resolve_runtime_memory_policy_path,
)
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
    vllm_scheduler_runtime_sha256,
)
from src.kernel.hardware_spec import HWSpec
from src.serving.kv_cache.kv_capacity import (
    load_runtime_memory_policy,
    resolve_analytical_kv_capacity_plan,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig, backend_config_sha256
from src.serving.schedule.schedule_step import BackendStep
from src.serving.workload.model_config import load_model_config


_FIELDS = (
    "request_order",
    "phase",
    "scheduled_tokens",
    "total_scheduled_tokens",
    "finished_set",
    "pool",
    "preempted_set",
    "computed_tokens",
    "recomputed_tokens",
    "prefix_replay_tokens",
)


class ValidatedCompactCase(Protocol):
    case_dir: Path
    manifest: Mapping[str, object]
    scheduler_pipeline_depth: int
    requests: Sequence[object]
    scheduler: Sequence[Mapping[str, object]] | None


class SchedulerBackend(Protocol):
    config: BackendConfig

    def add_request(self, request: RequestSpec) -> None: ...

    def plan_step(self, virtual_time_s: float) -> BackendStep | None: ...

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> None: ...

    def has_work(self) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SchedulerReplayContract:
    model_path: Path
    expected_model_id: str
    expected_model_config_sha256: str
    target_qps: float
    dtype: str
    deployment: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    max_batched_tokens: int
    max_requests: int
    num_kv_blocks: int
    kv_block_size_tokens: int
    max_model_len: int
    enable_chunked_prefill: bool
    enable_prefix_caching: bool
    gpu_memory_utilization: float = 0.9


@dataclass(frozen=True)
class ObservedRequest:
    request_idx: int
    scheduled_tokens: int
    phase: str
    computed_tokens: int | None
    recomputed_tokens: int | None
    prefix_replay_tokens: int | None


@dataclass(frozen=True)
class ObservedStep:
    index: int
    scheduler_time_us: float | None
    pool: str | None
    total_scheduled_tokens: int | None
    requests: tuple[ObservedRequest, ...]
    finished_request_idxs: tuple[int, ...] | None
    preempted_request_idxs: tuple[int, ...] | None
    availability: Mapping[str, str]


@dataclass(frozen=True)
class SchedulerAlignmentCase:
    hardware_id: str
    model_id: str
    model_revision: str | None
    case_id: str
    requests_sha256: str
    scheduler_sha256: str
    runtime: Mapping[str, object]
    scheduler_pipeline_depth: int
    requests: RequestStream
    observed_steps: tuple[ObservedStep, ...]


@dataclass(frozen=True)
class RuntimeFieldResult:
    field: str
    status: str
    capture_value: object
    replay_value: object


@dataclass(frozen=True)
class SchedulerMismatch:
    step_index: int
    canonical_step_id: str
    field: str
    observed_item_count: int
    replay_item_count: int
    observed_total_tokens: int | None
    replay_total_tokens: int
    first_differing_position: int | None


@dataclass(frozen=True)
class FieldResult:
    field: str
    status: str
    compared_steps: int
    matched_steps: int
    mismatched_steps: int
    unsupported_steps: int
    first_mismatch_step_index: int | None


@dataclass(frozen=True)
class SchedulerAlignmentReport:
    schema_version: int
    status: str
    policy_diagnostic_status: str
    admission_timing_status: str
    admission_mode: str
    hardware_id: str
    model_id: str
    case_id: str
    requests_sha256: str
    scheduler_sha256: str
    model_config_sha256: str
    backend_config_sha256: str
    scheduler_runtime_sha256: str | None
    observed_step_count: int
    observed_work_step_count: int
    admitted_request_count: int
    observed_finished_request_count: int | None
    replay_finished_request_count: int
    runtime_contract_status: str
    runtime_fields: tuple[RuntimeFieldResult, ...]
    field_results: tuple[FieldResult, ...]
    first_mismatch: SchedulerMismatch | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _Counter:
    compared: int = 0
    matched: int = 0
    mismatched: int = 0
    unsupported: int = 0
    first_mismatch: int | None = None

    def record(self, equal: bool, step_index: int) -> None:
        self.compared += 1
        if equal:
            self.matched += 1
        else:
            self.mismatched += 1
            if self.first_mismatch is None:
                self.first_mismatch = step_index


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return float(value)


def _optional_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _number(value, context)


def _optional_integer(value: object, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context)


def _optional_indices(
    value: object,
    request_count: int,
    context: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context} must be a list or null")
    indices = tuple(_integer(item, context) for item in value)
    if len(indices) != len(set(indices)):
        raise ValueError(f"{context} contains duplicate request indices")
    if any(index >= request_count for index in indices):
        raise ValueError(f"{context} references a request outside requests.csv")
    return indices


def _sha256(value: object, context: str) -> str:
    text = _string(value, context)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{context} must be a lowercase SHA256")
    return text


def scheduler_alignment_case_from_compact(
    compact: ValidatedCompactCase,
) -> SchedulerAlignmentCase:
    """Adapt an already validated compact-case object without reparsing it."""

    manifest = _mapping(compact.manifest, "compact manifest")
    if manifest["schema_version"] != 5:
        raise ValueError(
            "Scheduler alignment requires compact schema v5 configured "
            "Scheduler depth evidence")
    identity = _mapping(manifest["identity"], "manifest identity")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    scheduler_pipeline_depth = compact.scheduler_pipeline_depth
    requests_manifest = _mapping(manifest["requests"], "manifest requests")
    scheduler_manifest = _mapping(
        _mapping(manifest["alignment"], "manifest alignment")["scheduler"],
        "scheduler alignment manifest",
    )
    if scheduler_manifest["status"] != "available":
        raise ValueError("Scheduler alignment evidence must be available")
    if scheduler_manifest["coverage"] != "complete":
        raise ValueError("Scheduler alignment evidence must have complete coverage")
    if compact.scheduler is None:
        raise ValueError("validated compact case has no Scheduler records")
    requests = RequestStream(
        RequestSpec(
            arrival_time_s=float(row.arrival_time_ms) / 1000.0,
            request_id=f"request-{index:012d}",
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
            prefix_hit_tokens=row.cached_hit_tokens,
        )
        for index, row in enumerate(compact.requests)
    )
    observed = tuple(
        _observed_step(index, record, len(requests))
        for index, record in enumerate(compact.scheduler)
    )
    model_revision = identity["model_revision"]
    if model_revision is not None:
        model_revision = _string(model_revision, "model_revision")
    return SchedulerAlignmentCase(
        hardware_id=_string(identity["hardware_id"], "hardware_id"),
        model_id=_string(identity["model_id"], "model_id"),
        model_revision=model_revision,
        case_id=_string(identity["case_id"], "case_id"),
        requests_sha256=_sha256(
            requests_manifest["requests_sha256"], "requests_sha256"),
        scheduler_sha256=_sha256(
            scheduler_manifest["sha256"], "scheduler_sha256"),
        runtime=runtime,
        scheduler_pipeline_depth=scheduler_pipeline_depth,
        requests=requests,
        observed_steps=observed,
    )


def _observed_step(
    index: int,
    record_value: Mapping[str, object],
    request_count: int,
) -> ObservedStep:
    record = _mapping(record_value, f"Scheduler record {index}")
    availability = _mapping(
        record["field_availability"],
        f"Scheduler availability {index}",
    )
    scheduled = []
    for value in record["scheduled_requests"]:
        request = _mapping(value, f"Scheduler request in step {index}")
        request_idx = _integer(request["request_idx"], "request_idx")
        if request_idx >= request_count:
            raise ValueError("Scheduler request_idx is outside requests.csv")
        scheduled.append(ObservedRequest(
            request_idx=request_idx,
            scheduled_tokens=_integer(request["scheduled_tokens"], "scheduled_tokens"),
            phase=_string(request["phase"], "phase"),
            computed_tokens=_optional_integer(request["computed_tokens"], "computed_tokens"),
            recomputed_tokens=_optional_integer(request["recomputed_tokens"], "recomputed_tokens"),
            prefix_replay_tokens=_optional_integer(
                request["prefix_replay_tokens"], "prefix_replay_tokens"),
        ))
    total = _optional_integer(record["total_scheduled_tokens"], "total_scheduled_tokens")
    if total is not None and total != sum(request.scheduled_tokens for request in scheduled):
        raise ValueError("Scheduler total_scheduled_tokens disagrees with requests")
    pool = record["pool"]
    if pool is not None:
        pool = _string(pool, "pool")
    return ObservedStep(
        index=index,
        scheduler_time_us=_optional_number(
            record["scheduler_time_us"], "scheduler_time_us"),
        pool=pool,
        total_scheduled_tokens=total,
        requests=tuple(scheduled),
        finished_request_idxs=_optional_indices(
            record["finished_request_idxs"], request_count, "finished_request_idxs"),
        preempted_request_idxs=_optional_indices(
            record["preempted_request_idxs"], request_count, "preempted_request_idxs"),
        availability=availability,
    )


def _runtime_comparison(
    case: SchedulerAlignmentCase,
    contract: SchedulerReplayContract,
    backend_hash: str,
) -> tuple[str, tuple[RuntimeFieldResult, ...]]:
    replay_values = {
        "model_revision": load_model_config(str(contract.model_path)).model_revision,
        "dtype": contract.dtype,
        "deployment": contract.deployment,
        "tensor_parallel_size": contract.tensor_parallel_size,
        "pipeline_parallel_size": contract.pipeline_parallel_size,
        "expert_parallel_size": contract.expert_parallel_size,
        "target_qps": contract.target_qps,
        "seed": None,
        "scheduler_config_sha256": backend_hash,
        "runtime_version": VLLM_SCHEDULER_VERSION,
        "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
        "capture_enable_prefix_caching": contract.enable_prefix_caching,
        "max_batched_tokens": contract.max_batched_tokens,
        "max_requests": contract.max_requests,
        "num_kv_blocks": contract.num_kv_blocks,
        "kv_block_size_tokens": contract.kv_block_size_tokens,
        "max_model_len": contract.max_model_len,
        "enable_chunked_prefill": contract.enable_chunked_prefill,
    }
    capture_values = {
        "model_revision": case.model_revision,
        **dict(case.runtime),
        "max_batched_tokens": None,
        "max_requests": None,
        "num_kv_blocks": None,
        "kv_block_size_tokens": None,
        "max_model_len": None,
        "enable_chunked_prefill": None,
    }
    results = []
    for field in replay_values:
        capture = capture_values[field]
        replay = replay_values[field]
        if capture is None:
            status = "unsupported"
        elif field == "seed" and replay is None:
            status = "not_applicable"
        elif capture == replay:
            status = "matched"
        else:
            status = "mismatched"
        results.append(RuntimeFieldResult(field, status, capture, replay))
    if any(result.status == "mismatched" for result in results):
        return "mismatched", tuple(results)
    if any(result.status == "unsupported" for result in results):
        return "incomplete", tuple(results)
    return "matched", tuple(results)


def _validate_model_binding(
    case: SchedulerAlignmentCase,
    contract: SchedulerReplayContract,
) -> str:
    if case.model_id != contract.expected_model_id:
        raise ValueError("compact case model_id disagrees with replay contract")
    expected = _sha256(
        contract.expected_model_config_sha256,
        "expected model config SHA256",
    )
    actual = load_model_config(str(contract.model_path)).config_sha256
    if actual != expected:
        raise ValueError("model config SHA256 disagrees with replay contract")
    return actual


def align_vllm_scheduler_case(
    compact: ValidatedCompactCase,
    contract: SchedulerReplayContract,
) -> SchedulerAlignmentReport:
    """Run the pinned CPU Scheduler after strict identity/runtime binding."""

    case = scheduler_alignment_case_from_compact(compact)
    model_sha = _validate_model_binding(case, contract)
    model = load_model_config(str(contract.model_path))
    backend_config = BackendConfig(
        "main",
        "colocated",
        contract.max_batched_tokens,
        contract.max_requests,
        contract.num_kv_blocks,
        kv_block_size_tokens=contract.kv_block_size_tokens,
        kv_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(
            contract.tensor_parallel_size,
            contract.pipeline_parallel_size,
        ),
        scheduler_pipeline_depth=case.scheduler_pipeline_depth,
    )
    backend_hash = backend_config_sha256(backend_config)
    runtime_status, runtime_fields = _runtime_comparison(
        case,
        contract,
        backend_hash,
    )
    if runtime_status == "mismatched":
        return _runtime_mismatch_report(
            case,
            model_sha,
            backend_hash,
            runtime_fields,
        )
    scheduler_config = VLLMSchedulerConfig(
        str(contract.model_path),
        tensor_parallel_size=contract.tensor_parallel_size,
        pipeline_parallel_size=contract.pipeline_parallel_size,
        enable_expert_parallel=contract.expert_parallel_size > 1,
        dtype=contract.dtype,
        max_model_len=contract.max_model_len,
        gpu_memory_utilization=contract.gpu_memory_utilization,
        enable_chunked_prefill=contract.enable_chunked_prefill,
        enable_prefix_caching=contract.enable_prefix_caching,
    )
    backend = VLLMBackend.from_model(backend_config, scheduler_config)
    try:
        return align_scheduler_observation_conditioned(
            case,
            backend,
            model_config_sha256=model_sha,
            scheduler_runtime_sha256=vllm_scheduler_runtime_sha256(
                backend.runtime_info),
            runtime_contract_status=runtime_status,
            runtime_fields=runtime_fields,
        )
    finally:
        backend.close()


def _runtime_mismatch_report(
    case: SchedulerAlignmentCase,
    model_sha: str,
    backend_hash: str,
    runtime_fields: tuple[RuntimeFieldResult, ...],
) -> SchedulerAlignmentReport:
    return SchedulerAlignmentReport(
        schema_version=2,
        status="runtime_mismatch",
        policy_diagnostic_status="not_run",
        admission_timing_status="unsupported_no_shared_clock_anchor",
        admission_mode="not_run",
        hardware_id=case.hardware_id,
        model_id=case.model_id,
        case_id=case.case_id,
        requests_sha256=case.requests_sha256,
        scheduler_sha256=case.scheduler_sha256,
        model_config_sha256=model_sha,
        backend_config_sha256=backend_hash,
        scheduler_runtime_sha256=None,
        observed_step_count=len(case.observed_steps),
        observed_work_step_count=sum(bool(step.requests) for step in case.observed_steps),
        admitted_request_count=0,
        observed_finished_request_count=None,
        replay_finished_request_count=0,
        runtime_contract_status="mismatched",
        runtime_fields=runtime_fields,
        field_results=(),
        first_mismatch=None,
    )


def _request_indices(
    request_ids: Sequence[str],
    request_idx_by_id: Mapping[str, int],
) -> tuple[int, ...]:
    values = []
    for request_id in request_ids:
        if request_id not in request_idx_by_id:
            raise RuntimeError("Scheduler returned a request outside the canonical stream")
        values.append(request_idx_by_id[request_id])
    return tuple(values)


def _difference(expected: Sequence[object], actual: Sequence[object]) -> int | None:
    for position, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return position
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def _field_result(field: str, counter: _Counter) -> FieldResult:
    if counter.mismatched:
        status = "mismatched"
    elif counter.compared and counter.unsupported:
        status = "partial"
    elif counter.compared:
        status = "matched"
    else:
        status = "unsupported"
    return FieldResult(
        field,
        status,
        counter.compared,
        counter.matched,
        counter.mismatched,
        counter.unsupported,
        counter.first_mismatch,
    )


def align_scheduler_observation_conditioned(
    case: SchedulerAlignmentCase,
    backend: SchedulerBackend,
    *,
    model_config_sha256: str,
    scheduler_runtime_sha256: str | None,
    runtime_contract_status: str,
    runtime_fields: tuple[RuntimeFieldResult, ...],
) -> SchedulerAlignmentReport:
    """Check policy after conditioning admission on first observed schedule."""

    request_idx_by_id = {
        f"request-{index:012d}": index for index in range(len(case.requests))
    }
    requests_by_id = {request.request_id: request for request in case.requests}
    requests_by_idx = tuple(
        requests_by_id[f"request-{index:012d}"]
        for index in range(len(case.requests))
    )
    counters = {field: _Counter() for field in _FIELDS}
    admitted = set()
    observed_finished = set()
    replay_finished = set()
    pending = None
    prior_finished: tuple[str, ...] = ()
    first_mismatch = None
    for observed in case.observed_steps:
        if observed.scheduler_time_us is None:
            time_s = float(observed.index)
        else:
            time_s = observed.scheduler_time_us / 1_000_000.0
        if pending is not None:
            backend.complete_step(pending, time_s)
            pending = None
        for request in observed.requests:
            if request.request_idx not in admitted:
                backend.add_request(requests_by_idx[request.request_idx])
                admitted.add(request.request_idx)
        actual = backend.plan_step(time_s) if backend.has_work() else None
        actual_requests = () if actual is None else actual.schedule.requests
        observed_indices = tuple(request.request_idx for request in observed.requests)
        actual_indices = _request_indices(
            tuple(request.request_id for request in actual_requests),
            request_idx_by_id,
        )
        observed_tokens = tuple(request.scheduled_tokens for request in observed.requests)
        actual_tokens = tuple(request.scheduled_tokens for request in actual_requests)
        comparisons = (
            ("request_order", observed_indices, actual_indices, "available"),
            (
                "phase",
                tuple(request.phase for request in observed.requests),
                tuple(request.phase for request in actual_requests),
                "available",
            ),
            ("scheduled_tokens", observed_tokens, actual_tokens, "available"),
            (
                "total_scheduled_tokens",
                () if observed.total_scheduled_tokens is None else (
                    observed.total_scheduled_tokens,),
                (sum(actual_tokens),),
                observed.availability["total_scheduled_tokens"],
            ),
            (
                "finished_set",
                () if observed.finished_request_idxs is None else tuple(
                    sorted(observed.finished_request_idxs)),
                tuple(sorted(_request_indices(prior_finished, request_idx_by_id))),
                observed.availability["finished_request_idxs"],
            ),
            (
                "pool",
                () if observed.pool is None else (observed.pool,),
                (backend.config.pool_name,),
                observed.availability["pool"],
            ),
            (
                "preempted_set",
                () if observed.preempted_request_idxs is None else tuple(
                    sorted(observed.preempted_request_idxs)),
                () if actual is None else tuple(sorted(_request_indices(
                    actual.schedule.preempted_request_ids,
                    request_idx_by_id,
                ))),
                observed.availability["preempted_request_idxs"],
            ),
        )
        for field, expected, replayed, availability in comparisons:
            if availability == "unavailable":
                counters[field].unsupported += 1
                continue
            equal = tuple(expected) == tuple(replayed)
            counters[field].record(equal, observed.index)
            if not equal and first_mismatch is None:
                first_mismatch = SchedulerMismatch(
                    observed.index,
                    f"step-{observed.index:012d}",
                    field,
                    len(expected),
                    len(replayed),
                    observed.total_scheduled_tokens,
                    sum(actual_tokens),
                    _difference(expected, replayed),
                )
        for field in ("computed_tokens", "recomputed_tokens", "prefix_replay_tokens"):
            availability = observed.availability[f"scheduled_requests.{field}"]
            expected = tuple(getattr(request, field) for request in observed.requests)
            replayed = tuple(getattr(request, field) for request in actual_requests)
            if availability == "unavailable":
                counters[field].unsupported += 1
                continue
            positions = tuple(
                index for index, value in enumerate(expected) if value is not None)
            equal = all(
                index < len(replayed) and expected[index] == replayed[index]
                for index in positions
            )
            if availability == "partial":
                counters[field].unsupported += 1
            counters[field].record(equal, observed.index)
            if not equal and first_mismatch is None:
                first_mismatch = SchedulerMismatch(
                    observed.index,
                    f"step-{observed.index:012d}",
                    field,
                    len(expected),
                    len(replayed),
                    observed.total_scheduled_tokens,
                    sum(actual_tokens),
                    _difference(expected, replayed),
                )
        if observed.finished_request_idxs is not None:
            duplicate = observed_finished.intersection(observed.finished_request_idxs)
            if duplicate:
                raise ValueError("observed Scheduler finished a request more than once")
            observed_finished.update(observed.finished_request_idxs)
        replay_indices = _request_indices(prior_finished, request_idx_by_id)
        if replay_finished.intersection(replay_indices):
            raise RuntimeError("replayed Scheduler finished a request more than once")
        replay_finished.update(replay_indices)
        if actual is None:
            prior_finished = ()
        else:
            pending = actual
            prior_finished = actual.schedule.finished_request_ids
    if pending is not None:
        raise ValueError("Scheduler observation is truncated before the pending step completes")
    if len(admitted) != len(case.requests):
        raise ValueError("Scheduler observation does not admit every canonical request")
    if backend.has_work():
        raise ValueError("Scheduler observation ended before the replay backend was exhausted")
    observed_finished_count = None
    if all(step.finished_request_idxs is not None for step in case.observed_steps):
        observed_finished_count = len(observed_finished)
        if observed_finished != set(range(len(case.requests))):
            raise ValueError("Scheduler observation has incomplete finished-request coverage")
    if replay_finished != set(range(len(case.requests))):
        raise RuntimeError("replayed Scheduler has incomplete finished-request coverage")
    field_results = tuple(_field_result(field, counters[field]) for field in _FIELDS)
    has_mismatch = any(result.status == "mismatched" for result in field_results)
    has_unsupported = any(
        result.status in ("partial", "unsupported") for result in field_results)
    if has_mismatch:
        policy_status = "mismatched"
    elif has_unsupported:
        policy_status = "matched_with_unsupported"
    else:
        policy_status = "matched"
    return SchedulerAlignmentReport(
        schema_version=2,
        status="inconclusive_clock_alignment",
        policy_diagnostic_status=policy_status,
        admission_timing_status="unsupported_no_shared_clock_anchor",
        admission_mode="observation_conditioned_first_schedule",
        hardware_id=case.hardware_id,
        model_id=case.model_id,
        case_id=case.case_id,
        requests_sha256=case.requests_sha256,
        scheduler_sha256=case.scheduler_sha256,
        model_config_sha256=model_config_sha256,
        backend_config_sha256=backend_config_sha256(backend.config),
        scheduler_runtime_sha256=scheduler_runtime_sha256,
        observed_step_count=len(case.observed_steps),
        observed_work_step_count=sum(bool(step.requests) for step in case.observed_steps),
        admitted_request_count=len(admitted),
        observed_finished_request_count=observed_finished_count,
        replay_finished_request_count=len(replay_finished),
        runtime_contract_status=runtime_contract_status,
        runtime_fields=runtime_fields,
        field_results=field_results,
        first_mismatch=first_mismatch,
    )


def reports_summary(
    reports: Sequence[SchedulerAlignmentReport],
) -> dict[str, object]:
    """Aggregate a batch without copying request-level information."""

    status_counts = {
        "runtime_mismatch": 0,
        "inconclusive_clock_alignment": 0,
    }
    policy_counts = {
        "not_run": 0,
        "mismatched": 0,
        "matched_with_unsupported": 0,
        "matched": 0,
    }
    for report in reports:
        if report.status not in status_counts:
            status_counts[report.status] = 0
        status_counts[report.status] += 1
        if report.policy_diagnostic_status not in policy_counts:
            policy_counts[report.policy_diagnostic_status] = 0
        policy_counts[report.policy_diagnostic_status] += 1
    return {
        "schema_version": 2,
        "case_count": len(reports),
        "status_counts": status_counts,
        "policy_diagnostic_status_counts": policy_counts,
        "cases": [report.to_dict() for report in reports],
    }


def _file_sha256(path: Path) -> str:
    return file_sha256(path)


def _derive_analytical_num_kv_blocks(
    *,
    prediction: Mapping[str, object],
    hardware_id: str,
    hardware_config_path: Path,
    hardware_config_sha256: str,
    model_id: str,
    model_path: Path,
    worker_role: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel_size: int,
    kv_block_size_tokens: int,
    max_model_len: int,
    max_batched_tokens: int,
    max_requests: int,
    deployment: str,
) -> tuple[int, float]:
    template = _string(
        prediction["runtime_memory_policy_path"],
        "runtime_memory_policy_path",
    )
    expected_template = RUNTIME_MEMORY_POLICY_PATH_TEMPLATE
    if template != expected_template:
        raise ValueError("runtime memory policy path template mismatch")
    repository_root = Path(__file__).resolve().parents[3]
    policy_path = resolve_runtime_memory_policy_path(
        template,
        repository_root=repository_root,
        hardware_id=hardware_id,
        model_id=model_id,
    )
    hardware = HWSpec(str(hardware_config_path))
    if hardware.hardware_id != hardware_id:
        raise ValueError("Scheduler alignment hardware identity mismatch")
    model = load_model_config(str(model_path))
    parallelism = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        enable_expert_parallel=expert_parallel_size > 1,
    )
    if parallelism.expert_parallel_size != expert_parallel_size:
        raise ValueError("Scheduler alignment expert parallel size mismatch")
    policies = load_runtime_memory_policy(policy_path)
    policies.validate_scheduler_pools(("main",))
    plan = resolve_analytical_kv_capacity_plan(
        "main",
        hardware=hardware,
        hardware_config_sha256=hardware_config_sha256,
        model=model,
        parallelism=parallelism,
        deployment=deployment,
        worker_role=worker_role,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        max_batched_tokens=max_batched_tokens,
        max_requests=max_requests,
        policy=policies.policy_for_pool("main"),
    )
    policy = policies.policy_for_pool("main")
    utilization = (
        policy.gpu_memory_utilization_numerator
        / policy.gpu_memory_utilization_denominator)
    return plan.num_kv_blocks, utilization


def _phase1_contract(
    config: Phase1Config,
    cell: Phase1Cell,
    repository_root: Path,
) -> SchedulerReplayContract:
    values = config.prediction.values
    configured_model = cell.model.simulation_model_path
    compute_dtype = values["compute_dtype"]
    if compute_dtype == "bf16":
        dtype = "bfloat16"
    elif compute_dtype == "fp16":
        dtype = "float16"
    else:
        raise ValueError(f"unsupported Scheduler dtype: {compute_dtype!r}")
    model_path = resolve_repository_input(
        repository_root, configured_model, "simulation model config")
    num_kv_blocks, gpu_memory_utilization = (
        _derive_analytical_num_kv_blocks(
        prediction=values,
        hardware_id=cell.hardware.hardware_id,
        hardware_config_path=(
            repository_root / cell.hardware.hardware_config_path),
        hardware_config_sha256=cell.hardware.hardware_config_sha256,
        model_id=cell.model.dataset_model_id,
        model_path=model_path,
        worker_role="colocated",
        tensor_parallel_size=_integer(
            values["tensor_parallel_size"], "tensor_parallel_size", 1),
        pipeline_parallel_size=_integer(
            values["pipeline_parallel_size"], "pipeline_parallel_size", 1),
        expert_parallel_size=_integer(
            values["expert_parallel_size"], "expert_parallel_size", 1),
        kv_block_size_tokens=_integer(
            values["kv_block_size_tokens"], "kv_block_size_tokens", 1),
        max_model_len=_integer(values["max_model_len"], "max_model_len", 1),
        max_batched_tokens=_integer(
            values["max_batched_tokens"], "max_batched_tokens", 1),
        max_requests=_integer(values["max_requests"], "max_requests", 1),
        deployment=_string(values["deployment"], "deployment"),
    ))
    return SchedulerReplayContract(
        model_path=model_path,
        expected_model_id=cell.model.dataset_model_id,
        expected_model_config_sha256=cell.model.model_config_sha256,
        target_qps=float(cell.qps),
        dtype=dtype,
        deployment=_string(values["deployment"], "deployment"),
        tensor_parallel_size=_integer(
            values["tensor_parallel_size"], "tensor_parallel_size", 1),
        pipeline_parallel_size=_integer(
            values["pipeline_parallel_size"], "pipeline_parallel_size", 1),
        expert_parallel_size=_integer(
            values["expert_parallel_size"], "expert_parallel_size", 1),
        max_batched_tokens=_integer(
            values["max_batched_tokens"], "max_batched_tokens", 1),
        max_requests=_integer(values["max_requests"], "max_requests", 1),
        num_kv_blocks=num_kv_blocks,
        kv_block_size_tokens=_integer(
            values["kv_block_size_tokens"], "kv_block_size_tokens", 1),
        max_model_len=_integer(values["max_model_len"], "max_model_len", 1),
        enable_chunked_prefill=values["enable_chunked_prefill"],
        enable_prefix_caching=values["enable_prefix_caching"],
        gpu_memory_utilization=gpu_memory_utilization,
    )


def _validate_cell_files(case_path: Path, cell: Phase1Cell) -> None:
    if _file_sha256(case_path / "manifest.json") != cell.manifest_sha256:
        raise ValueError(f"phase-1 manifest SHA256 mismatch for {cell.cell_id}")
    if _file_sha256(case_path / "requests.csv") != cell.requests_sha256:
        raise ValueError(f"phase-1 requests SHA256 mismatch for {cell.cell_id}")




def _phase1_reports(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    load_validated_case,
) -> tuple[SchedulerAlignmentReport, ...]:
    root = Path(repository_root).resolve()
    reports = []
    for cell in cells:
        case_path = compact_case_path(root, config, cell)
        _validate_cell_files(case_path, cell)
        compact = load_validated_case(case_path)
        reports.append(align_vllm_scheduler_case(
            compact,
            _phase1_contract(config, cell, root),
        ))
    return tuple(reports)


def _model_bindings(values: Sequence[str]) -> Mapping[str, tuple[Path, str]]:
    bindings = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError(
                "--extra-model must be MODEL_ID=MODEL_PATH=CONFIG_SHA256")
        model_id, model_path, model_sha = parts
        if model_id in bindings:
            raise ValueError(f"duplicate --extra-model binding for {model_id!r}")
        bindings[_string(model_id, "extra model ID")] = (
            Path(_string(model_path, "extra model path")).resolve(),
            _sha256(model_sha, "extra model config SHA256"),
        )
    return bindings


def _extra_contract(
    compact: ValidatedCompactCase,
    config: Phase1Config,
    bindings: Mapping[str, tuple[Path, str]],
    repository_root: Path,
) -> SchedulerReplayContract:
    prediction = config.prediction.values
    case = scheduler_alignment_case_from_compact(compact)
    if case.model_id not in bindings:
        raise ValueError(f"no --extra-model binding for {case.model_id!r}")
    model_path, model_sha = bindings[case.model_id]
    compute_dtype = case.runtime["dtype"]
    if compute_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"unsupported Scheduler dtype: {compute_dtype!r}")
    hardware_entries = {
        entry.hardware_id: entry for entry in config.hardware
    }
    if case.hardware_id not in hardware_entries:
        raise ValueError(
            f"no hardware config binding for {case.hardware_id!r}")
    hardware = hardware_entries[case.hardware_id]
    tensor_parallel_size = _integer(
        case.runtime["tensor_parallel_size"], "tensor_parallel_size", 1)
    pipeline_parallel_size = _integer(
        case.runtime["pipeline_parallel_size"], "pipeline_parallel_size", 1)
    expert_parallel_size = _integer(
        case.runtime["expert_parallel_size"], "expert_parallel_size", 1)
    kv_block_size_tokens = _integer(
        prediction["kv_block_size_tokens"], "kv_block_size_tokens", 1)
    max_model_len = _integer(
        prediction["max_model_len"], "max_model_len", 1)
    num_kv_blocks, gpu_memory_utilization = (
        _derive_analytical_num_kv_blocks(
        prediction=prediction,
        hardware_id=case.hardware_id,
        hardware_config_path=repository_root / hardware.hardware_config_path,
        hardware_config_sha256=hardware.hardware_config_sha256,
        model_id=case.model_id,
        model_path=model_path,
        worker_role="colocated",
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        max_batched_tokens=_integer(
            prediction["max_batched_tokens"], "max_batched_tokens", 1),
        max_requests=_integer(
            prediction["max_requests"], "max_requests", 1),
        deployment=_string(prediction["deployment"], "deployment"),
    ))
    return SchedulerReplayContract(
        model_path=model_path,
        expected_model_id=case.model_id,
        expected_model_config_sha256=model_sha,
        target_qps=float(case.runtime["target_qps"]),
        dtype=compute_dtype,
        deployment=_string(prediction["deployment"], "deployment"),
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        expert_parallel_size=expert_parallel_size,
        max_batched_tokens=_integer(
            prediction["max_batched_tokens"], "max_batched_tokens", 1),
        max_requests=_integer(prediction["max_requests"], "max_requests", 1),
        num_kv_blocks=num_kv_blocks,
        kv_block_size_tokens=kv_block_size_tokens,
        max_model_len=max_model_len,
        enable_chunked_prefill=prediction["enable_chunked_prefill"],
        enable_prefix_caching=prediction["enable_prefix_caching"],
        gpu_memory_utilization=gpu_memory_utilization,
    )


def run_scheduler_alignment_cli(
    argv: Sequence[str] | None,
    *,
    repository_root: str | Path,
    load_validated_case,
) -> dict[str, object]:
    """Parse and execute the tracked Phase-1 suite batch entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/evaluation/experiments/sharegpt_phase1.json",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", action="append", default=[])
    selection.add_argument("--all-cases", action="store_true")
    parser.add_argument("--extra-case", action="append", default=[])
    parser.add_argument("--extra-model", action="append", default=[])
    args = parser.parse_args(argv)
    if not args.case and not args.all_cases and not args.extra_case:
        parser.error("select --case/--all-cases and/or at least one --extra-case")
    root = Path(repository_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_phase1_config(config_path)
    reports = []
    if args.case or args.all_cases:
        cells = select_phase1_cells(config, args.case, all_cases=args.all_cases)
        reports.extend(_phase1_reports(
            root,
            config,
            cells,
            load_validated_case,
        ))
    bindings = _model_bindings(args.extra_model)
    for case_value in args.extra_case:
        case_path = Path(case_value)
        if not case_path.is_absolute():
            case_path = root / case_path
        compact = load_validated_case(case_path)
        reports.append(align_vllm_scheduler_case(
            compact,
            _extra_contract(compact, config, bindings, root),
        ))
    identities = tuple(
        (report.hardware_id, report.model_id, report.case_id)
        for report in reports
    )
    if len(identities) != len(set(identities)):
        raise ValueError("Scheduler alignment batch contains duplicate cases")
    return reports_summary(reports)
