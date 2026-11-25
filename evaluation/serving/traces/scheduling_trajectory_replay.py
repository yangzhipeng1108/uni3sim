"""Evaluate current Forward by replaying an observed scheduling trajectory.

The observed trace supplies only step membership, order, request progress, and
optional Scheduler timestamps. Observed step durations never enter Forward.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Callable, Mapping, Sequence

import zstandard

from evaluation.serving.metrics import compact_step_alignment
from evaluation.serving.metrics import sharegpt_compact_evaluation
from evaluation.serving.simulation import execution_artifact
from src.kernel.hardware_spec import HWSpec
from src.serving.execution import ServingExecutionContext
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
)
from src.serving.kernel_lowering.execution_timeline import (
    build_execution_timeline,
    build_runtime_graph_segments,
)
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
from src.serving.workload.model_config import load_model_config
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.runtime_inputs import RuntimeInputs

from evaluation.json_shape import require_integer as _integer
from evaluation.json_shape import file_sha256 as _sha256

from evaluation.json_shape import require_object_dict as _mapping

from evaluation.json_shape import require_list_or_tuple as _sequence

from evaluation.json_shape import percentile_value as _percentile


PHASES = ("prefill", "mixed", "decode")
LATENCY_FIELDS = ("ttft_ms", "tpot_ms", "e2e_latency_ms")
SCHEDULER_FIELDS = {
    "step_index",
    "step_id",
    "source_iteration_id",
    "step_start_ns",
    "schedule_end_ns",
    "update_start_ns",
    "step_end_ns",
    "scheduled_requests",
    "scheduled_request_count",
    "prefill_request_count",
    "decode_request_count",
    "prefill_tokens",
    "decode_tokens",
    "scheduled_tokens",
    "finished_request_indexes",
    "cumulative_scheduled_tokens",
    "cumulative_completed_requests",
}
SCHEDULED_REQUEST_FIELDS = {
    "request_idx",
    "prefill_tokens",
    "decode_tokens",
    "scheduled_tokens",
}
SCHEDULER_RUNTIME_FIELDS = {
    "schedule_start_ns",
    "scheduler_update_start_ns",
    "scheduler_update_end_ns",
    "device_execution_start_ns",
    "device_execution_end_ns",
    "device_execution_clock",
    "device_execution_source",
    "in_flight_step_count",
}
SCHEDULER_FLAT_GRAPH_FIELDS = {
    "capture_bucket_tokens",
    "capture_limit_tokens",
    "configured_cudagraph_mode",
    "graph_fallback_reason",
    "num_unpadded_tokens",
    "padded_total_tokens",
    "padding_tokens",
    "runtime_cudagraph_mode",
}
SCHEDULER_RANK_OBSERVATION_FIELDS = {"rank_observations"}
RANK_GRAPH_OBSERVATION_FIELDS = {
    "schema_version",
    "rank",
    "trace_step_id",
    "configured_cudagraph_mode",
    "runtime_cudagraph_mode",
    "dispatch_input_key",
    "selected_graph_key",
    "scheduled_tokens",
    "num_unpadded_tokens",
    "padded_total_tokens",
    "padding_tokens",
    "graph_fallback_reason",
    "worker_execution_clock",
    "worker_execution_start_ns",
    "worker_execution_end_ns",
    "worker_execution_source",
    "source",
}
GRAPH_DISPATCH_KEY_FIELDS = {"num_tokens", "uniform_decode"}
STEP_FINGERPRINT_FIELDS = {
    "schema_version",
    "cell_id",
    "hardware_id",
    "model_id",
    "step_id",
    "request_workload",
    "token_totals",
    "parallelism",
    "graph_execution",
    "normalized_launch_program_sha256",
    "normalized_launch_call_count",
}


def run_scheduling_trajectory_replay_case(
    repository_root: str | Path,
    config_path: str | Path,
    cell_id: str,
    *,
    load_validated_case: Callable[[Path], object],
    scheduler_trace: str | Path | None,
    scheduler_schema_version: int,
    host_launch_profile: HostLaunchProfile | None,
    execution_mode: str,
    counterfactual_pipeline_depth: int | None,
) -> dict[str, object]:
    """Resolve one canonical case and run its trajectory replay."""

    root = Path(repository_root).resolve()
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = root / resolved_config_path
    config = sharegpt_compact_evaluation.load_phase1_config(resolved_config_path)
    cell = sharegpt_compact_evaluation.select_phase1_cells(
        config, [cell_id], all_cases=False
    )[0]
    case_path = sharegpt_compact_evaluation.compact_case_path(root, config, cell)
    case = load_validated_case(case_path)
    sharegpt_compact_evaluation._validate_case_binding(config, cell, case)
    configured_depth = (
        sharegpt_compact_evaluation.scheduler_pipeline_depth_from_validated_case(case)
    )
    scheduler_path = (
        case_path / "scheduler_steps.jsonl.zst"
        if scheduler_trace is None
        else Path(scheduler_trace).resolve()
    )
    if counterfactual_pipeline_depth is None:
        resolved_depth = configured_depth
        depth_mode = "dataset_native"
    else:
        resolved_depth = counterfactual_pipeline_depth
        if resolved_depth < 1:
            raise ValueError("counterfactual pipeline depth must be positive")
        depth_mode = "explicit_counterfactual"
    runtime = case.manifest["runtime"]
    depth_identity = {
        "configured_depth": configured_depth,
        "resolved_depth": resolved_depth,
        "mode": depth_mode,
        "evidence": runtime["scheduler_pipeline_depth_evidence"],
    }
    return analyze_scheduling_trajectory_replay(
        root,
        resolved_config_path,
        cell.cell_id,
        scheduler_path,
        request_contract_from_compact_case(case),
        cell.model.model_config_sha256,
        scheduler_schema_version=scheduler_schema_version,
        host_launch_profile=host_launch_profile,
        execution_mode=execution_mode,
        pipeline_depth=resolved_depth,
        pipeline_depth_identity=depth_identity,
    )












def _exact_fields(
    value: Mapping[str, object], expected: set[str], label: str
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _invalid_experiment(field: str, observed: object, replayed: object) -> None:
    raise ValueError(
        f"INVALID EXPERIMENT — {field}: observed={observed!r}, "
        f"replayed={replayed!r}"
    )


def _first_fingerprint_mismatch(
    observed: object,
    replayed: object,
    path: str,
) -> tuple[str, object, object] | None:
    if isinstance(observed, Mapping) and isinstance(replayed, Mapping):
        observed_fields = set(observed)
        replayed_fields = set(replayed)
        if observed_fields != replayed_fields:
            return (
                f"{path}.fields",
                sorted(observed_fields),
                sorted(replayed_fields),
            )
        for field in sorted(observed_fields):
            mismatch = _first_fingerprint_mismatch(
                observed[field], replayed[field], f"{path}.{field}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(observed, (list, tuple)) and isinstance(replayed, (list, tuple)):
        if len(observed) != len(replayed):
            return (f"{path}.length", len(observed), len(replayed))
        for index, (left, right) in enumerate(zip(observed, replayed, strict=True)):
            mismatch = _first_fingerprint_mismatch(
                left, right, f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if observed != replayed:
        return (path, observed, replayed)
    return None


def require_exact_step_fingerprints(
    observed: Mapping[str, object],
    replayed: Mapping[str, object],
) -> None:
    """Fail closed before reporting a cross-run step comparison."""

    mismatch = _first_fingerprint_mismatch(observed, replayed, "fingerprint")
    if mismatch is not None:
        _invalid_experiment(*mismatch)
    if set(observed) != STEP_FINGERPRINT_FIELDS:
        _invalid_experiment(
            "fingerprint.fields",
            sorted(STEP_FINGERPRINT_FIELDS),
            sorted(observed),
        )


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string or null")
    return value


def _graph_dispatch_key(value: object, label: str) -> dict[str, object]:
    key = _mapping(value, label)
    _exact_fields(key, GRAPH_DISPATCH_KEY_FIELDS, label)
    return {
        "num_tokens": _integer(key["num_tokens"], f"{label} num_tokens", minimum=1),
        "uniform_decode": _boolean(
            key["uniform_decode"], f"{label} uniform_decode"
        ),
    }


def _validate_graph_observation(
    graph: Mapping[str, object],
    *,
    scheduled_tokens: int,
    label: str,
) -> dict[str, object]:
    configured_mode = graph["configured_mode"]
    if configured_mode not in (
        "NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
        "FULL_AND_PIECEWISE",
    ):
        raise ValueError(f"{label} configured Graph mode is invalid")
    runtime_mode = graph["runtime_mode"]
    if runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
        raise ValueError(f"{label} runtime Graph mode is invalid")
    logical_tokens = _integer(
        graph["logical_tokens"], f"{label} logical_tokens", minimum=1
    )
    model_input_tokens = _integer(
        graph["model_input_tokens"], f"{label} model_input_tokens", minimum=1
    )
    padding_tokens = _integer(
        graph["padding_tokens"], f"{label} padding_tokens"
    )
    if logical_tokens != scheduled_tokens:
        raise ValueError(f"{label} logical tokens disagree with Scheduler tokens")
    if model_input_tokens != logical_tokens + padding_tokens:
        raise ValueError(f"{label} padding conservation failed")
    capture_bucket = graph["capture_bucket_tokens"]
    fallback_reason = _optional_string(
        graph["fallback_reason"], f"{label} fallback_reason"
    )
    if runtime_mode == "NONE":
        if capture_bucket is not None:
            raise ValueError(f"{label} eager fallback cannot carry a Graph bucket")
        if configured_mode == "NONE":
            if fallback_reason is not None:
                raise ValueError(f"{label} disabled Graph cannot carry a fallback")
        elif fallback_reason is None:
            raise ValueError(f"{label} Graph fallback requires a reason")
        if padding_tokens != 0:
            raise ValueError(f"{label} eager fallback cannot carry padding")
    else:
        capture_bucket = _integer(
            capture_bucket, f"{label} capture_bucket_tokens", minimum=1
        )
        if capture_bucket != model_input_tokens:
            raise ValueError(f"{label} Graph bucket disagrees with padded tokens")
        if fallback_reason is not None:
            raise ValueError(f"{label} selected Graph cannot carry a fallback")
    capture_limit = graph["capture_limit_tokens"]
    if capture_limit is not None:
        capture_limit = _integer(
            capture_limit, f"{label} capture_limit_tokens", minimum=1
        )
        if capture_bucket is not None and capture_bucket > capture_limit:
            raise ValueError(f"{label} Graph bucket exceeds capture limit")
    uniform_decode = _boolean(
        graph["uniform_decode"], f"{label} uniform_decode"
    )
    rank_ids = tuple(graph["rank_ids"])
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
           for rank in rank_ids):
        raise TypeError(f"{label} rank_ids must be non-negative integers")
    if tuple(sorted(set(rank_ids))) != rank_ids:
        raise ValueError(f"{label} rank_ids must be unique and sorted")
    return {
        "configured_mode": configured_mode,
        "runtime_mode": runtime_mode,
        "uniform_decode": uniform_decode,
        "logical_tokens": logical_tokens,
        "model_input_tokens": model_input_tokens,
        "capture_bucket_tokens": capture_bucket,
        "capture_limit_tokens": capture_limit,
        "padding_tokens": padding_tokens,
        "fallback_reason": fallback_reason,
        "rank_ids": list(rank_ids),
    }


def _flat_graph_observation(
    row: Mapping[str, object],
    *,
    step_index: int,
) -> dict[str, object]:
    requests = _sequence(
        row["scheduled_requests"], f"Scheduler step {step_index} requests"
    )
    uniform_decode = all(
        _integer(
            _mapping(request, "scheduled request")["scheduled_tokens"],
            "scheduled request tokens",
            minimum=1,
        ) == 1
        for request in requests
    )
    return _validate_graph_observation({
        "configured_mode": row["configured_cudagraph_mode"],
        "runtime_mode": row["runtime_cudagraph_mode"],
        "uniform_decode": uniform_decode,
        "logical_tokens": row["num_unpadded_tokens"],
        "model_input_tokens": row["padded_total_tokens"],
        "capture_bucket_tokens": row["capture_bucket_tokens"],
        "capture_limit_tokens": row["capture_limit_tokens"],
        "padding_tokens": row["padding_tokens"],
        "fallback_reason": row["graph_fallback_reason"],
        "rank_ids": (),
    }, scheduled_tokens=_integer(
        row["scheduled_tokens"], f"Scheduler step {step_index} scheduled tokens",
        minimum=1,
    ), label=f"Scheduler step {step_index} Graph observation")


def _rank_graph_observation(
    row: Mapping[str, object],
    *,
    step_index: int,
) -> dict[str, object]:
    observations = _sequence(
        row["rank_observations"],
        f"Scheduler step {step_index} rank observations",
    )
    if not observations:
        raise ValueError(
            f"Scheduler step {step_index} rank observations are empty"
        )
    identities = []
    ranks = []
    for position, value in enumerate(observations):
        observation = _mapping(
            value, f"Scheduler step {step_index} rank observation {position}"
        )
        _exact_fields(
            observation,
            RANK_GRAPH_OBSERVATION_FIELDS,
            f"Scheduler step {step_index} rank observation {position}",
        )
        if observation["schema_version"] != 1:
            raise ValueError("Scheduler rank observation schema must equal 1")
        rank = _integer(
            observation["rank"], "Scheduler rank observation rank"
        )
        if observation["trace_step_id"] != row["step_id"]:
            raise ValueError("Scheduler rank observation step identity differs")
        dispatch = _graph_dispatch_key(
            observation["dispatch_input_key"],
            "Scheduler rank observation dispatch_input_key",
        )
        selected_value = observation["selected_graph_key"]
        selected = (
            None
            if selected_value is None
            else _graph_dispatch_key(
                selected_value,
                "Scheduler rank observation selected_graph_key",
            )
        )
        identity = {
            "configured_mode": observation["configured_cudagraph_mode"],
            "runtime_mode": observation["runtime_cudagraph_mode"],
            "uniform_decode": dispatch["uniform_decode"],
            "logical_tokens": observation["num_unpadded_tokens"],
            "model_input_tokens": observation["padded_total_tokens"],
            "capture_bucket_tokens": (
                None if selected is None else selected["num_tokens"]
            ),
            "capture_limit_tokens": None,
            "padding_tokens": observation["padding_tokens"],
            "fallback_reason": observation["graph_fallback_reason"],
        }
        if dispatch["num_tokens"] != observation["padded_total_tokens"]:
            raise ValueError(
                "Scheduler Graph dispatch tokens disagree with padded rank row"
            )
        if selected is not None and selected["uniform_decode"] != dispatch["uniform_decode"]:
            raise ValueError("Scheduler selected Graph decode mode differs")
        identities.append(identity)
        ranks.append(rank)
    if ranks != sorted(set(ranks)):
        raise ValueError("Scheduler Graph rank observations are not unique and sorted")
    reference = identities[0]
    for identity in identities[1:]:
        mismatch = _first_fingerprint_mismatch(
            reference, identity, "graph_rank_identity"
        )
        if mismatch is not None:
            _invalid_experiment(*mismatch)
    return _validate_graph_observation({
        **reference,
        "rank_ids": tuple(ranks),
    }, scheduled_tokens=_integer(
        row["scheduled_tokens"], f"Scheduler step {step_index} scheduled tokens",
        minimum=1,
    ), label=f"Scheduler step {step_index} Graph observation")


def _phase(prefill_tokens: int, decode_tokens: int) -> str:
    if prefill_tokens and decode_tokens:
        return "mixed"
    if prefill_tokens:
        return "prefill"
    if decode_tokens:
        return "decode"
    raise ValueError("work-bearing step has no prefill or decode tokens")


def _step_workload_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return _canonical_sha256(list(rows))


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_replay_graph_identity(
    row: Mapping[str, object],
    graph_execution,
    *,
    expected_rank_ids: tuple[int, ...],
) -> dict[str, object]:
    observation = row["graph_execution_observation"]
    if observation is None:
        _invalid_experiment(
            "graph_execution_observation",
            "required",
            None,
        )
    graph = _mapping(observation, "Scheduler Graph observation")
    uniform_decode = all(
        int(request["scheduled_tokens"])
        == graph_execution.uniform_decode_query_len
        for request in row["requests"]
    )
    resolved = graph_execution.resolve(
        int(row["scheduled_total_tokens"]),
        uniform_decode=uniform_decode,
    )
    expected = {
        "configured_mode": graph_execution.configured_mode,
        "runtime_mode": resolved.runtime_mode,
        "uniform_decode": uniform_decode,
        "logical_tokens": resolved.logical_tokens,
        "model_input_tokens": resolved.model_input_tokens,
        "capture_bucket_tokens": (
            None
            if resolved.runtime_mode == "NONE"
            else resolved.model_input_tokens
        ),
        "padding_tokens": resolved.padding_tokens,
    }
    for field, value in expected.items():
        if graph[field] != value:
            _invalid_experiment(
                f"graph_execution.{field}", graph[field], value
            )
    capture_limit = graph["capture_limit_tokens"]
    expected_capture_limit = (
        None
        if not graph_execution.capture_sizes
        else max(graph_execution.capture_sizes)
    )
    if capture_limit is not None and capture_limit != expected_capture_limit:
        _invalid_experiment(
            "graph_execution.capture_limit_tokens",
            capture_limit,
            expected_capture_limit,
        )
    observed_rank_ids = tuple(graph["rank_ids"])
    if observed_rank_ids and observed_rank_ids != expected_rank_ids:
        _invalid_experiment(
            "graph_execution.rank_ids",
            list(observed_rank_ids),
            list(expected_rank_ids),
        )
    return {
        **expected,
        "capture_limit_tokens": expected_capture_limit,
        "fallback_reason": graph["fallback_reason"],
        "rank_ids": list(expected_rank_ids),
    }


def _normalized_launch_program(
    components,
) -> tuple[list[dict[str, object]], int]:
    occurrence_counts = dict(
        components.execution_timeline.call_occurrence_counts
    )
    calls = []
    logical_call_count = 0
    for call in components.kernel_plan.topological_order():
        occurrences = occurrence_counts[call.call_id]
        logical_call_count += occurrences
        calls.append({
            "pool_name": call.pool_name,
            "rank": call.rank,
            "resource": call.resource,
            "occurrence_count": occurrences,
            "kernel_spec": execution_artifact._kernel_spec_manifest(
                call.kernel_spec
            ),
        })
    return calls, logical_call_count


def _step_fingerprint(
    *,
    cell_id: str,
    hardware_id: str,
    model_id: str,
    row: Mapping[str, object],
    parallelism: Mapping[str, object],
    graph_execution: Mapping[str, object],
    normalized_launch_program: Sequence[Mapping[str, object]],
    normalized_launch_call_count: int,
) -> dict[str, object]:
    request_workload = []
    for request in row["requests"]:
        computed_tokens = int(request["computed_tokens"])
        scheduled_tokens = int(request["scheduled_tokens"])
        request_workload.append({
            "request_idx": int(request["request_idx"]),
            "request_id": str(request["request_id"]),
            "computed_tokens_before_step": computed_tokens,
            "kv_context_tokens_after_step": computed_tokens + scheduled_tokens,
            "prefill_tokens": int(request["prefill_tokens"]),
            "decode_tokens": int(request["decode_tokens"]),
            "scheduled_tokens": scheduled_tokens,
        })
    fingerprint = {
        "schema_version": 1,
        "cell_id": cell_id,
        "hardware_id": hardware_id,
        "model_id": model_id,
        "step_id": row["step_id"],
        "request_workload": request_workload,
        "token_totals": {
            "prefill_tokens": row["scheduled_prefill_tokens"],
            "decode_tokens": row["scheduled_decode_tokens"],
            "scheduled_tokens": row["scheduled_total_tokens"],
        },
        "parallelism": dict(parallelism),
        "graph_execution": dict(graph_execution),
        "normalized_launch_program_sha256": _canonical_sha256(
            normalized_launch_program
        ),
        "normalized_launch_call_count": normalized_launch_call_count,
    }
    _exact_fields(fingerprint, STEP_FINGERPRINT_FIELDS, "step fingerprint")
    return fingerprint


def request_contract_from_compact_case(case: object) -> list[dict[str, object]]:
    """Convert a validated compact case to the scheduling-replay request contract."""

    requests = []
    for index, request in enumerate(case.requests):
        if request.cached_hit_tokens != 0:
            raise ValueError("scheduling trajectory replay requires zero effective prefix hit")
        if request.output_tokens <= 1:
            raise ValueError("scheduling trajectory replay requires at least two output tokens")
        requests.append(
            {
                "request_id": f"request-{index:012d}",
                "arrival_time_s": request.arrival_time_ms / 1000.0,
                "input_tokens": request.input_tokens,
                "output_tokens": request.output_tokens,
            }
        )
    if not requests:
        raise ValueError("scheduling trajectory replay requires at least one request")
    return requests


def load_and_validate_scheduler_trace(
    path: Path,
    request_contract: Sequence[Mapping[str, object]],
    *,
    schema_version: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load an instrumented Scheduler trace and reconstruct Forward step inputs."""

    rows = []
    if path.suffix == ".zst":
        try:
            trace_bytes = zstandard.ZstdDecompressor().decompress(
                path.read_bytes())
        except zstandard.ZstdError as error:
            raise ValueError(f"Scheduler trace is invalid zstd: {path}") from error
        try:
            trace_text = trace_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Scheduler trace must be UTF-8: {path}") from error
    else:
        trace_text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError("Scheduler trace is empty")
    if schema_version is None and "schema_version" in rows[0]:
        versions = {
            _integer(row["schema_version"], "Scheduler schema version", minimum=1)
            for row in rows
        }
        if len(versions) != 1:
            raise ValueError("Scheduler schemas vary across rows")
        schema_version = next(iter(versions))
    request_count = len(request_contract)
    if request_count == 0:
        raise ValueError("request contract is empty")
    input_tokens = [
        _integer(request["input_tokens"], f"request {index} input_tokens", minimum=1)
        for index, request in enumerate(request_contract)
    ]
    output_tokens = [
        _integer(request["output_tokens"], f"request {index} output_tokens", minimum=1)
        for index, request in enumerate(request_contract)
    ]
    request_ids = [str(request["request_id"]) for request in request_contract]
    if len(set(request_ids)) != request_count:
        raise ValueError("request IDs are not unique")
    scheduled_prefill = [0] * request_count
    scheduled_decode = [0] * request_count
    previous_source_iteration = -1
    previous_timestamps = [-1, -1, -1, -1]
    scheduler_pipeline_overlaps_ns = []
    cumulative_tokens = 0
    finished_requests: set[int] = set()
    async_discard_placeholder_step_count = 0
    async_discard_placeholder_token_count = 0
    normalized_rows = []
    workloads = []
    phase_counts = {phase: 0 for phase in PHASES}
    for step_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"Scheduler step {step_index}")
        base_fields = (
            SCHEDULER_FIELDS
            if schema_version is None
            else SCHEDULER_FIELDS | {"schema_version"}
        )
        graph_observation = None
        if schema_version is None:
            _exact_fields(row, base_fields, f"Scheduler step {step_index}")
        elif schema_version == 3 and set(row) == (
            base_fields | SCHEDULER_RUNTIME_FIELDS | SCHEDULER_FLAT_GRAPH_FIELDS
        ):
            graph_observation = _flat_graph_observation(
                row, step_index=step_index
            )
        elif schema_version in (4, 5) and set(row) == (
            base_fields | SCHEDULER_RUNTIME_FIELDS
            | SCHEDULER_RANK_OBSERVATION_FIELDS
        ):
            graph_observation = _rank_graph_observation(
                row, step_index=step_index
            )
        else:
            _exact_fields(row, base_fields, f"Scheduler step {step_index}")
        if schema_version is not None and row["schema_version"] != schema_version:
            raise ValueError(
                f"Scheduler step {step_index} schema_version must equal {schema_version}"
            )
        if row["step_index"] != step_index:
            raise ValueError("Scheduler step indexes are not contiguous")
        source_iteration = _integer(
            row["source_iteration_id"],
            f"Scheduler step {step_index} source_iteration_id",
        )
        if source_iteration <= previous_source_iteration:
            raise ValueError("Scheduler source iterations are not strictly ordered")
        previous_source_iteration = source_iteration
        if row["step_id"] != f"vllm:{source_iteration}":
            raise ValueError(f"Scheduler step {step_index} identity differs")
        timestamps = [
            _integer(row[field], f"Scheduler step {step_index} {field}")
            for field in (
                "step_start_ns",
                "schedule_end_ns",
                "update_start_ns",
                "step_end_ns",
            )
        ]
        if timestamps != sorted(timestamps):
            raise ValueError(f"Scheduler step {step_index} timestamps are unordered")
        if any(
            current <= previous
            for current, previous in zip(timestamps, previous_timestamps, strict=True)
        ):
            raise ValueError("Scheduler timestamp series are not strictly ordered")
        if previous_timestamps[-1] >= 0 and timestamps[0] < previous_timestamps[-1]:
            scheduler_pipeline_overlaps_ns.append(previous_timestamps[-1] - timestamps[0])
        previous_timestamps = timestamps
        scheduled_requests = _sequence(
            row["scheduled_requests"],
            f"Scheduler step {step_index} scheduled_requests",
        )
        normalized_requests = []
        async_discard_placeholder_indexes: set[int] = set()
        for request_order, raw_request in enumerate(scheduled_requests):
            request = _mapping(
                raw_request, f"Scheduler step {step_index} request {request_order}"
            )
            _exact_fields(
                request,
                SCHEDULED_REQUEST_FIELDS,
                f"Scheduler step {step_index} request {request_order}",
            )
            request_idx = _integer(
                request["request_idx"], f"Scheduler step {step_index} request_idx"
            )
            if request_idx >= request_count:
                raise ValueError(f"Scheduler step {step_index} request index is invalid")
            prefill = _integer(
                request["prefill_tokens"],
                f"Scheduler step {step_index} request prefill_tokens",
            )
            decode = _integer(
                request["decode_tokens"],
                f"Scheduler step {step_index} request decode_tokens",
            )
            scheduled = _integer(
                request["scheduled_tokens"],
                f"Scheduler step {step_index} request scheduled_tokens",
                minimum=1,
            )
            if prefill and decode:
                raise ValueError("one request cannot mix prefill and decode")
            if scheduled != prefill + decode:
                raise ValueError("Scheduler request token partition differs")
            if scheduled_prefill[request_idx] + prefill > input_tokens[request_idx]:
                raise ValueError("Scheduler over-scheduled request input tokens")
            # vLLM's async-scheduling RUNNING-request loop
            # (vllm/v1/core/sched/scheduler.py) computes num_new_tokens from
            # num_output_placeholders with no check against request.max_tokens
            # in the 0.11.0 runtime A800 was captured on. vLLM 0.24.0 (C500)
            # and 0.18.0 (910B) both add an explicit guard -- "Avoid
            # scheduling an extra step when we are sure that the previous
            # step has reached request.max_tokens" -- that 0.11.0 lacks, so it
            # schedules one placeholder-only decode step for a request whose
            # prior (still-unconfirmed, async depth-2) step already reached
            # max_tokens. That step's token is never delivered to the client.
            # Identified positively by the request already appearing in an
            # earlier step's finished_request_indexes -- not inferred from
            # this budget check itself -- so a step that does not match this
            # evidence still raises exactly as before.
            is_async_discard_placeholder = (
                decode > 0 and request_idx in finished_requests
            )
            if is_async_discard_placeholder:
                async_discard_placeholder_indexes.add(request_idx)
            elif scheduled_decode[request_idx] + decode > output_tokens[request_idx] - 1:
                raise ValueError("Scheduler over-scheduled request output tokens")
            normalized_requests.append(
                {
                    "request_idx": request_idx,
                    "request_id": request_ids[request_idx],
                    "computed_tokens": scheduled_prefill[request_idx]
                    + scheduled_decode[request_idx],
                    "prefill_tokens": prefill,
                    "decode_tokens": decode,
                    "scheduled_tokens": scheduled,
                }
            )
        indexes = [int(request["request_idx"]) for request in normalized_requests]
        if indexes != sorted(set(indexes)):
            raise ValueError("Scheduler request indexes are not unique and sorted")
        request_count_value = _integer(
            row["scheduled_request_count"],
            f"Scheduler step {step_index} scheduled_request_count",
            minimum=1,
        )
        prefill_tokens_value = _integer(
            row["prefill_tokens"], f"Scheduler step {step_index} prefill_tokens"
        )
        decode_tokens_value = _integer(
            row["decode_tokens"], f"Scheduler step {step_index} decode_tokens"
        )
        scheduled_tokens_value = _integer(
            row["scheduled_tokens"],
            f"Scheduler step {step_index} scheduled_tokens",
            minimum=1,
        )
        if request_count_value != len(normalized_requests):
            raise ValueError("Scheduler request count differs")
        if prefill_tokens_value != sum(
            int(request["prefill_tokens"]) for request in normalized_requests
        ):
            raise ValueError("Scheduler prefill-token total differs")
        if decode_tokens_value != sum(
            int(request["decode_tokens"]) for request in normalized_requests
        ):
            raise ValueError("Scheduler decode-token total differs")
        if scheduled_tokens_value != prefill_tokens_value + decode_tokens_value:
            raise ValueError("Scheduler step token partition differs")
        if row["prefill_request_count"] != sum(
            int(request["prefill_tokens"]) > 0 for request in normalized_requests
        ):
            raise ValueError("Scheduler prefill request count differs")
        if row["decode_request_count"] != sum(
            int(request["decode_tokens"]) > 0 for request in normalized_requests
        ):
            raise ValueError("Scheduler decode request count differs")
        phase = _phase(prefill_tokens_value, decode_tokens_value)
        phase_counts[phase] += 1
        cumulative_tokens += scheduled_tokens_value
        if row["cumulative_scheduled_tokens"] != cumulative_tokens:
            raise ValueError("Scheduler cumulative token count differs")
        finished = [
            _integer(value, f"Scheduler step {step_index} finished request")
            for value in _sequence(
                row["finished_request_indexes"],
                f"Scheduler step {step_index} finished requests",
            )
        ]
        if finished != sorted(set(finished)) or any(
            value >= request_count for value in finished
        ):
            raise ValueError("Scheduler finished request indexes differ")
        if not set(finished).issubset(set(indexes)):
            raise ValueError("Scheduler finished a request outside its current step")
        if finished_requests & set(finished):
            raise ValueError("Scheduler completed a request twice")
        finished_requests.update(finished)
        if row["cumulative_completed_requests"] != len(finished_requests):
            raise ValueError("Scheduler cumulative completion count differs")
        for request in normalized_requests:
            request_idx = int(request["request_idx"])
            scheduled_prefill[request_idx] += int(request["prefill_tokens"])
            if request_idx in async_discard_placeholder_indexes:
                async_discard_placeholder_step_count += 1
                async_discard_placeholder_token_count += int(request["decode_tokens"])
            else:
                scheduled_decode[request_idx] += int(request["decode_tokens"])
        workload = {
            "step_id": row["step_id"],
            "requests": normalized_requests,
            "scheduled_request_count": request_count_value,
            "scheduled_prefill_tokens": prefill_tokens_value,
            "scheduled_decode_tokens": decode_tokens_value,
            "scheduled_total_tokens": scheduled_tokens_value,
            "graph_execution_observation": graph_observation,
        }
        workloads.append(workload)
        normalized_rows.append(
            {
                "step_index": step_index,
                "step_id": row["step_id"],
                "phase": phase,
                "step_start_ns": timestamps[0],
                "schedule_end_ns": timestamps[1],
                "update_start_ns": timestamps[2],
                "step_end_ns": timestamps[3],
                "requests": normalized_requests,
                "finished_request_indexes": finished,
                "scheduled_request_count": request_count_value,
                "scheduled_prefill_tokens": prefill_tokens_value,
                "scheduled_decode_tokens": decode_tokens_value,
                "scheduled_total_tokens": scheduled_tokens_value,
                "graph_execution_observation": graph_observation,
            }
        )
    if scheduled_prefill != input_tokens:
        raise ValueError("Scheduler does not conserve each request's input tokens")
    expected_decode = [value - 1 for value in output_tokens]
    if scheduled_decode != expected_decode:
        raise ValueError("Scheduler does not conserve each request's model output work")
    if finished_requests != set(range(request_count)):
        raise ValueError("Scheduler did not complete the exact request set")
    expected_tokens = (
        sum(input_tokens) + sum(expected_decode) + async_discard_placeholder_token_count
    )
    if cumulative_tokens != expected_tokens:
        raise ValueError("Scheduler aggregate token count differs from requests")
    return normalized_rows, {
        "record_count": len(rows),
        "completed_requests": len(finished_requests),
        "scheduled_prefill_tokens": sum(scheduled_prefill),
        "scheduled_decode_tokens": sum(scheduled_decode),
        "scheduled_tokens": cumulative_tokens,
        "phase_step_counts": phase_counts,
        "step_workload_sha256": _step_workload_sha256(workloads),
        "pipelined_adjacent_step_overlap_count": len(scheduler_pipeline_overlaps_ns),
        "maximum_pipelined_adjacent_step_overlap_ns": (
            max(scheduler_pipeline_overlaps_ns)
            if scheduler_pipeline_overlaps_ns
            else 0
        ),
        "async_discard_placeholder_step_count": async_discard_placeholder_step_count,
        "async_discard_placeholder_token_count": async_discard_placeholder_token_count,
    }


def predict_scheduling_trajectory_steps(
    repository_root: Path,
    config_path: Path,
    cell_id: str,
    scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    model_config_sha256: str,
    *,
    host_launch_profile: HostLaunchProfile | None = None,
    host_runtime_prepare_profile: HostRuntimePrepareProfile | None = None,
    runtime_prepare_event_builder: Callable[
        [KernelPlan], Mapping[
            tuple[str, int, str, int], tuple[RuntimePrepareEvent, ...]
        ]
    ] | None = None,
    execution_mode: str = "accelerated",
    include_step_kernel_contributions: bool = False,
    graph_execution_override: GraphExecutionConfig | None = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    """Run current zero-fit Forward for every observed step without its duration."""

    root = repository_root.resolve()
    resolved_config_path = config_path.resolve()
    config = sharegpt_compact_evaluation.load_phase1_config(resolved_config_path)
    cell = sharegpt_compact_evaluation.select_phase1_cells(
        config, [cell_id], all_cases=False
    )[0]
    model_path = root / cell.model.simulation_model_path
    hardware_path = root / cell.hardware.hardware_config_path
    model = load_model_config(str(model_path))
    if (
        model_config_sha256 != cell.model.model_config_sha256
        or model.config_sha256 != model_config_sha256
    ):
        _invalid_experiment(
            "model_config_sha256",
            model_config_sha256,
            model.config_sha256,
        )
    hardware = HWSpec(hardware_path)
    if (host_runtime_prepare_profile is None) != (
            runtime_prepare_event_builder is None):
        raise ValueError(
            "runtime prepare profile and event builder must be supplied together")
    if (host_runtime_prepare_profile is not None
            and host_launch_profile is None):
        raise ValueError(
            "runtime prepare replay requires a Host launch profile")
    if (host_runtime_prepare_profile is not None
            and host_runtime_prepare_profile.hardware_id
            != hardware.hardware_id):
        raise ValueError(
            "runtime prepare profile hardware differs from replay hardware")
    prediction = config.prediction.values
    deployment = compact_step_alignment.build_deployment(
        prediction, model, hardware.hardware_id
    )
    expert_routing = sharegpt_compact_evaluation._expected_expert_routing(
        root, cell
    )
    case_path = sharegpt_compact_evaluation.compact_case_path(root, config, cell)
    case_manifest = json.loads((case_path / "manifest.json").read_text())
    runtime_version = case_manifest["runtime"]["runtime_version"]
    # H200 captures vLLM 0.30 but the current standard prediction contract
    # intentionally reuses the supported 0.11 dense CUDA model source.
    source_versions = {
        ("nvidia_a800", "0.11.0"): ("vllm_0_11_0", "vllm_0_11"),
        ("metax_c500", "0.24.0"): ("vllm_0_24_0", "vllm_0_24"),
        ("ascend_910b", "0.18.0"): ("vllm_0_18_0", "vllm_0_18"),
        ("nvidia_h200", "0.30.0"): ("vllm_0_11_0", "vllm_0_11"),
    }
    source_key = (cell.hardware.hardware_id, runtime_version)
    if source_key not in source_versions:
        raise ValueError(f"unsupported trajectory source identity: {source_key!r}")
    model_execution_version, sampler_prefix = source_versions[source_key]
    runtime_inputs = RuntimeInputs(
        expert_routing=expert_routing,
        attention_kv_layout=prediction["attention_kv_layout"],
        attention_kv_distribution=prediction["attention_kv_distribution"],
        model_execution_version=model_execution_version,
        target_sampling_source=(
            f"{sampler_prefix}_plain_greedy_{model.compute_dtype}"
        ),
        graph_execution=(
            cell.hardware.graph_execution
            if graph_execution_override is None
            else graph_execution_override
        ),
    )
    execution = ServingExecutionContext(
        model,
        deployment,
        {hardware.hardware_id: hardware},
        prediction["launch_policy"],
        prediction["prediction_method"],
        runtime_inputs,
        execution_mode=execution_mode,
    )
    results = []
    contribution_groups: dict[
        tuple[str, str, str | None, str | None, tuple[tuple[str, int], ...]],
        dict[str, object],
    ] = {}
    for row in scheduler_rows:
        scheduled_requests = []
        for request in row["requests"]:
            prefill_tokens = int(request["prefill_tokens"])
            decode_tokens = int(request["decode_tokens"])
            request_idx = int(request["request_idx"])
            reaches_generation_boundary = (
                decode_tokens > 0
                or int(request["computed_tokens"]) + prefill_tokens
                >= int(requests[request_idx]["input_tokens"])
            )
            scheduled_requests.append(
                ScheduledRequest(
                    request_id=str(request["request_id"]),
                    phase="prefill" if prefill_tokens else "decode",
                    input_tokens=prefill_tokens,
                    output_tokens=decode_tokens,
                    computed_tokens=int(request["computed_tokens"]),
                    raw_model_output_tokens=int(reaches_generation_boundary),
                )
            )
        step = ScheduleStep(
            step_id=str(row["step_id"]),
            pool_name="main",
            requests=tuple(scheduled_requests),
            finished_request_ids=tuple(
                requests[index]["request_id"]
                for index in row["finished_request_indexes"]
            ),
            attention_kv_layout=prediction["attention_kv_layout"],
        )
        expected_ranks = tuple(deployment.worker_pools[step.pool_name].ranks)
        graph_identity = _validated_replay_graph_identity(
            row,
            runtime_inputs.graph_execution,
            expected_rank_ids=expected_ranks,
        )
        components = execution.execute_step(step)
        piecewise_eager_kinds = (
            runtime_inputs.graph_execution.piecewise_eager_workload_kinds)
        host_timeline = (
            None
            if host_launch_profile is None
            else build_execution_timeline(
                components.kernel_plan,
                components.predictions,
                host_launch_profiles={hardware.hardware_id: host_launch_profile},
                host_runtime_prepare_profiles=(
                    None
                    if host_runtime_prepare_profile is None
                    else {
                        hardware.hardware_id: host_runtime_prepare_profile,
                    }
                ),
                runtime_prepare_events=(
                    None
                    if host_runtime_prepare_profile is None
                    else runtime_prepare_event_builder(
                        components.kernel_plan)
                ),
                runtime_graph_mode=str(graph_identity["runtime_mode"]),
                runtime_graph_segments=build_runtime_graph_segments(
                    components.workload,
                    components.kernel_plan,
                    str(graph_identity["runtime_mode"]),
                    piecewise_eager_kinds,
                    sync_before_replay=(
                        host_launch_profile is not None
                        and host_launch_profile.runtime_name == "CANN"
                        and host_launch_profile.framework_name == "torch_npu"
                    ),
                ),
            )
        )
        occurrence_counts = dict(components.execution_timeline.call_occurrence_counts)
        normalized_launch_program, normalized_launch_call_count = (
            _normalized_launch_program(components)
        )
        step_kernel_contributions = []
        for call in components.kernel_plan.calls:
            spec = call.kernel_spec
            shape_items = tuple(sorted(spec.shape.items()))
            key = (
                str(row["phase"]),
                spec.kind.value,
                spec.variant,
                spec.dtype,
                shape_items,
            )
            if key not in contribution_groups:
                contribution_groups[key] = {
                    "phase": row["phase"],
                    "kernel_kind": spec.kind.value,
                    "variant": spec.variant,
                    "dtype": spec.dtype,
                    "shape": dict(shape_items),
                    "step_indexes": set(),
                    "lowered_call_site_count": 0,
                    "logical_call_count": 0,
                    "predicted_additive_s": 0.0,
                }
            contribution = contribution_groups[key]
            occurrences = occurrence_counts[call.call_id]
            contribution["step_indexes"].add(row["step_index"])
            contribution["lowered_call_site_count"] += 1
            contribution["logical_call_count"] += occurrences
            contribution["predicted_additive_s"] += (
                components.predictions[call.call_id].t_total_s * occurrences
            )
            if include_step_kernel_contributions:
                step_kernel_contributions.append({
                    "call_id": call.call_id,
                    "pool_name": call.pool_name,
                    "rank": call.rank,
                    "resource": call.resource,
                    "occurrence_count": occurrences,
                    "predicted_per_occurrence_s": components.predictions[
                        call.call_id
                    ].t_total_s,
                    "predicted_additive_s": components.predictions[
                        call.call_id
                    ].t_total_s * occurrences,
                    "kernel_spec": execution_artifact._kernel_spec_manifest(spec),
                })
        by_bottleneck = {
            category.removeprefix("kernel:"): seconds
            for category, seconds in components.execution_timeline.critical_path_contributions
        }
        results.append(
            {
                "step_fingerprint": _step_fingerprint(
                    cell_id=cell.cell_id,
                    hardware_id=hardware.hardware_id,
                    model_id=cell.model.dataset_model_id,
                    row=row,
                    parallelism={
                        "tensor_parallel_size": prediction[
                            "tensor_parallel_size"
                        ],
                        "pipeline_parallel_size": prediction[
                            "pipeline_parallel_size"
                        ],
                        "expert_parallel_size": prediction[
                            "expert_parallel_size"
                        ],
                    },
                    graph_execution=graph_identity,
                    normalized_launch_program=normalized_launch_program,
                    normalized_launch_call_count=normalized_launch_call_count,
                ),
                "predicted_timeline_ns": components.execution_timeline.duration_s
                * 1_000_000_000,
                **(
                    {}
                    if host_timeline is None
                    else {
                        "predicted_device_plus_launch_ns": host_timeline.duration_s
                        * 1_000_000_000,
                        "predicted_launch_cpu_ns": host_timeline.launch_cpu_time_s
                        * 1_000_000_000,
                        "predicted_launch_critical_path_ns": (
                            host_timeline.launch_critical_path_s
                            * 1_000_000_000
                        ),
                        "predicted_host_device_overlap_ns": (
                            host_timeline.host_device_overlap_s
                            * 1_000_000_000
                        ),
                        "predicted_execution_timeline": (
                            execution_artifact._execution_timeline_manifest(
                                components.kernel_plan,
                                host_timeline,
                                components.predictions,
                                origin_time_s=0.0,
                            )
                        ),
                    }
                ),
                "predicted_critical_path_by_bottleneck_ns": {
                    category: seconds * 1_000_000_000
                    for category, seconds in sorted(by_bottleneck.items())
                },
                **(
                    {"predicted_kernel_contributions": step_kernel_contributions}
                    if include_step_kernel_contributions
                    else {}
                ),
            }
        )
    contributions = {phase: [] for phase in PHASES}
    phase_additive_s = {
        phase: sum(
            float(value["predicted_additive_s"])
            for value in contribution_groups.values()
            if value["phase"] == phase
        )
        for phase in PHASES
    }
    for value in contribution_groups.values():
        phase = str(value["phase"])
        result = {name: field for name, field in value.items() if name != "step_indexes"}
        result["step_count"] = len(value["step_indexes"])
        result["predicted_additive_share_of_phase"] = (
            float(value["predicted_additive_s"]) / phase_additive_s[phase]
        )
        contributions[phase].append(result)
    for phase in PHASES:
        contributions[phase].sort(
            key=lambda row: (
                -row["predicted_additive_s"],
                row["kernel_kind"],
                "" if row["variant"] is None else row["variant"],
                tuple(row["shape"].items()),
            )
        )
    return results, contributions, {
        "phase1_config": _sha256(resolved_config_path),
        "hardware_config": _sha256(hardware_path),
        "model_config": _sha256(model_path),
        "serving_execution": _sha256(root / "src/serving/execution.py"),
        "execution_timeline": _sha256(
            root / "src/serving/kernel_lowering/execution_timeline.py"
        ),
        "execution_cache": execution.cache_stats(),
        "execution_performance": execution.performance_stats(),
        "expert_routing": expert_routing.manifest(),
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }




def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        **{
            f"p{percentile}": _percentile(values, percentile)
            for percentile in (0, 50, 90, 99, 100)
        },
    }


def reconstruct_request_metrics(
    scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    completion_times_s: Sequence[float],
) -> dict[str, object]:
    """Reconstruct TTFT, TPOT, E2E latency, and throughput from step completions."""

    if len(scheduler_rows) != len(completion_times_s):
        raise ValueError("step completion count differs from Scheduler row count")
    first_token_s: list[float | None] = [None] * len(requests)
    completion_s: list[float | None] = [None] * len(requests)
    for row, step_completion_s in zip(scheduler_rows, completion_times_s, strict=True):
        for scheduled in row["requests"]:
            request_index = int(scheduled["request_idx"])
            if (
                first_token_s[request_index] is None
                and int(scheduled["prefill_tokens"]) > 0
                and int(scheduled["computed_tokens"])
                + int(scheduled["prefill_tokens"])
                >= int(requests[request_index]["input_tokens"])
            ):
                first_token_s[request_index] = step_completion_s
        for request_index in row["finished_request_indexes"]:
            if completion_s[request_index] is not None:
                raise ValueError("request completed more than once")
            completion_s[request_index] = step_completion_s
    if any(value is None for value in first_token_s):
        raise ValueError("at least one request has no first-token timestamp")
    if any(value is None for value in completion_s):
        raise ValueError("at least one request has no completion timestamp")
    per_request = []
    for index, request in enumerate(requests):
        first = float(first_token_s[index])
        completion = float(completion_s[index])
        arrival = float(request["arrival_time_s"])
        ttft_s = first - arrival
        e2e_s = completion - arrival
        if ttft_s < 0.0 or e2e_s < ttft_s:
            raise ValueError("scheduling trajectory replay produced an invalid request timeline")
        tpot_s = (completion - first) / (int(request["output_tokens"]) - 1)
        per_request.append(
            {
                "request_idx": index,
                "ttft_ms": ttft_s * 1000.0,
                "tpot_ms": tpot_s * 1000.0,
                "e2e_latency_ms": e2e_s * 1000.0,
            }
        )
    minimum_arrival_s = min(float(request["arrival_time_s"]) for request in requests)
    final_completion_s = max(float(value) for value in completion_s)
    duration_s = final_completion_s - minimum_arrival_s
    return {
        "request_count": len(requests),
        "duration_s": duration_s,
        "request_throughput_per_s": len(requests) / duration_s,
        "latency": {
            field: _summarize_values([float(row[field]) for row in per_request])
            for field in LATENCY_FIELDS
        },
        "per_request": per_request,
    }


def reconstruct_ttft_decomposition(
    scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    completion_times_s: Sequence[float],
    service_times_s: Sequence[float],
) -> dict[str, object]:
    """Split each request's TTFT into Prefill service and waiting time."""

    if len(scheduler_rows) != len(completion_times_s):
        raise ValueError("step completion count differs from Scheduler row count")
    if len(scheduler_rows) != len(service_times_s):
        raise ValueError("step service count differs from Scheduler row count")
    request_count = len(requests)
    first_token_s: list[float | None] = [None] * request_count
    prefill_service_s = [0.0] * request_count
    for row, step_completion_s, step_service_s in zip(
        scheduler_rows, completion_times_s, service_times_s, strict=True
    ):
        if not math.isfinite(step_service_s) or step_service_s <= 0.0:
            raise ValueError("step service duration must be finite and positive")
        for scheduled in row["requests"]:
            request_index = int(scheduled["request_idx"])
            prefill_tokens = int(scheduled["prefill_tokens"])
            if prefill_tokens == 0 or first_token_s[request_index] is not None:
                continue
            prefill_service_s[request_index] += step_service_s
            if (
                int(scheduled["computed_tokens"]) + prefill_tokens
                >= int(requests[request_index]["input_tokens"])
            ):
                first_token_s[request_index] = step_completion_s
    if any(value is None for value in first_token_s):
        raise ValueError("at least one request has no first-token timestamp")
    rows = []
    for request_index, request in enumerate(requests):
        arrival_s = float(request["arrival_time_s"])
        ttft_s = float(first_token_s[request_index]) - arrival_s
        raw_compute_s = prefill_service_s[request_index]
        compute_s = min(raw_compute_s, ttft_s)
        queue_s = ttft_s - compute_s
        rows.append({
            "request_idx": request_index,
            "queue_ms": queue_s * 1000.0,
            "compute_ms": compute_s * 1000.0,
            "prefill_service_before_arrival_boundary_ms": (
                raw_compute_s - compute_s
            ) * 1000.0,
            "ttft_ms": ttft_s * 1000.0,
        })
    return {
        "semantics": {
            "compute_ms": (
                "sum of replay step service for every Prefill chunk of the "
                "request through first-token production, clipped to the "
                "request arrival-to-first-token interval"
            ),
            "queue_ms": (
                "TTFT minus Prefill step service; includes admission waiting, "
                "inter-chunk waiting, and Scheduler CPU outside model service"
            ),
            "closure": "ttft_ms = queue_ms + compute_ms per request",
            "prefill_service_before_arrival_boundary_ms": (
                "Prefill service excluded because the normalized Server step "
                "interval begins before the Client arrival boundary"
            ),
        },
        "per_request": rows,
    }


def substitute_service_preserving_observed_waiting(
    scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    observed_service_times_s: Sequence[float],
    predicted_service_times_s: Sequence[float],
) -> dict[str, object]:
    """Replace step service while holding each request's observed waiting fixed.

    This is a per-request evaluation counterfactual, not a globally feasible
    replay timeline.  It isolates the direct latency effect of changing step
    service before any clock drift or Scheduler rebatching is allowed.
    """

    if len(scheduler_rows) != len(observed_service_times_s):
        raise ValueError("observed service duration count differs from Scheduler rows")
    if len(scheduler_rows) != len(predicted_service_times_s):
        raise ValueError("predicted service duration count differs from Scheduler rows")
    observed_metrics = reconstruct_request_metrics(
        scheduler_rows,
        requests,
        [int(row["step_end_ns"]) / 1_000_000_000.0 for row in scheduler_rows],
    )
    request_count = len(requests)
    observed_prefill_s = [0.0] * request_count
    predicted_prefill_s = [0.0] * request_count
    observed_decode_s = [0.0] * request_count
    predicted_decode_s = [0.0] * request_count
    first_token_seen = [False] * request_count
    for row, observed_service_s, predicted_service_s in zip(
        scheduler_rows,
        observed_service_times_s,
        predicted_service_times_s,
        strict=True,
    ):
        if min(observed_service_s, predicted_service_s) <= 0.0:
            raise ValueError("step service durations must be positive")
        for scheduled in row["requests"]:
            request_index = int(scheduled["request_idx"])
            if not first_token_seen[request_index]:
                if int(scheduled["prefill_tokens"]) > 0:
                    observed_prefill_s[request_index] += observed_service_s
                    predicted_prefill_s[request_index] += predicted_service_s
                    if (
                        int(scheduled["computed_tokens"])
                        + int(scheduled["prefill_tokens"])
                        >= int(requests[request_index]["input_tokens"])
                    ):
                        first_token_seen[request_index] = True
            elif int(scheduled["decode_tokens"]) > 0:
                observed_decode_s[request_index] += observed_service_s
                predicted_decode_s[request_index] += predicted_service_s
    if not all(first_token_seen):
        raise ValueError("at least one request has no first-token step")
    observed_by_request = {
        int(row["request_idx"]): row for row in observed_metrics["per_request"]
    }
    rows = []
    ttft_decomposition = []
    tpot_decomposition = []
    for request_index, request in enumerate(requests):
        observed = observed_by_request[request_index]
        observed_ttft_ms = float(observed["ttft_ms"])
        observed_output_interval_ms = (
            float(observed["tpot_ms"])
            * (int(request["output_tokens"]) - 1)
        )
        observed_prefill_ms = min(
            observed_prefill_s[request_index] * 1000.0,
            observed_ttft_ms,
        )
        observed_decode_ms = min(
            observed_decode_s[request_index] * 1000.0,
            observed_output_interval_ms,
        )
        ttft_waiting_ms = observed_ttft_ms - observed_prefill_ms
        output_waiting_ms = observed_output_interval_ms - observed_decode_ms
        predicted_prefill_ms = predicted_prefill_s[request_index] * 1000.0
        predicted_decode_ms = predicted_decode_s[request_index] * 1000.0
        ttft_ms = ttft_waiting_ms + predicted_prefill_ms
        output_interval_ms = output_waiting_ms + predicted_decode_ms
        output_intervals = int(request["output_tokens"]) - 1
        tpot_ms = output_interval_ms / output_intervals
        e2e_latency_ms = ttft_ms + output_interval_ms
        rows.append({
            "request_idx": request_index,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_latency_ms": e2e_latency_ms,
        })
        ttft_decomposition.append({
            "request_idx": request_index,
            "queue_ms": ttft_waiting_ms,
            "compute_ms": predicted_prefill_ms,
            "ttft_ms": ttft_ms,
        })
        tpot_decomposition.append({
            "request_idx": request_index,
            "waiting_per_output_token_ms": output_waiting_ms / output_intervals,
            "service_per_output_token_ms": predicted_decode_ms / output_intervals,
            "tpot_ms": tpot_ms,
        })
    return {
        "request_count": request_count,
        "duration_s": observed_metrics["duration_s"],
        "request_throughput_per_s": observed_metrics["request_throughput_per_s"],
        "latency": {
            field: _summarize_values([float(row[field]) for row in rows])
            for field in LATENCY_FIELDS
        },
        "per_request": rows,
        "ttft_decomposition": {
            "semantics": {
                "compute_ms": "predicted Prefill service on the observed request path",
                "queue_ms": "observed TTFT residual held fixed",
                "closure": "ttft_ms = queue_ms + compute_ms per request",
            },
            "per_request": ttft_decomposition,
        },
        "tpot_decomposition": {
            "semantics": {
                "service_per_output_token_ms": "predicted Decode service divided by observed output intervals",
                "waiting_per_output_token_ms": "observed inter-token residual held fixed",
                "closure": "tpot_ms = waiting_per_output_token_ms + service_per_output_token_ms per request",
            },
            "per_request": tpot_decomposition,
        },
    }


def replay_scheduling_trajectory(
    scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
    service_times_s: Sequence[float],
    *,
    include_observed_scheduler_cpu: bool,
    pipeline_depth: int,
    include_graph_replay_barrier: bool = False,
) -> tuple[list[float], dict[str, object]]:
    """Replay fixed batches with a serial device and configurable in-flight slots.

    ``include_graph_replay_barrier`` is a diagnostic-only counterfactual for
    the Ascend ACLGraphWrapper source contract.  The wrapper calls
    ``current_stream().synchronize()`` immediately before an ordinary graph
    replay.  The regular replay intentionally models each step in isolation;
    this option carries the device completion of the preceding step into the
    next step's Host dispatch clock so that the cross-step wait is visible.
    It is never enabled by the production Serving Forward.
    """

    if isinstance(pipeline_depth, bool) or not isinstance(pipeline_depth, int) or pipeline_depth < 1:
        raise ValueError("pipeline_depth must be a positive integer")
    if len(scheduler_rows) != len(service_times_s):
        raise ValueError("service duration count differs from Scheduler row count")
    if not isinstance(include_graph_replay_barrier, bool):
        raise TypeError("graph replay barrier flag must be boolean")
    first_step_by_request: dict[int, int] = {}
    schedule_ends: list[float] = []
    execution_ends: list[float] = []
    completion_times: list[float] = []
    graph_barrier_waits: list[float] = []
    plan_total_s = 0.0
    update_total_s = 0.0
    for step_index, (row, service_s) in enumerate(
        zip(scheduler_rows, service_times_s, strict=True)
    ):
        if not math.isfinite(service_s) or service_s <= 0.0:
            raise ValueError("step service duration must be finite and positive")
        indexes = [int(value["request_idx"]) for value in row["requests"]]
        for request_index in indexes:
            if request_index not in first_step_by_request:
                first_step_by_request[request_index] = step_index
        required_arrival_s = max(
            float(requests[request_index]["arrival_time_s"])
            for request_index in indexes
        )
        slot_ready_s = (
            completion_times[step_index - pipeline_depth]
            if step_index >= pipeline_depth
            else 0.0
        )
        scheduler_ready_s = schedule_ends[-1] if schedule_ends else 0.0
        schedule_start_s = max(slot_ready_s, scheduler_ready_s, required_arrival_s)
        plan_s = (
            (int(row["schedule_end_ns"]) - int(row["step_start_ns"]))
            / 1_000_000_000.0
            if include_observed_scheduler_cpu
            else 0.0
        )
        update_s = (
            (int(row["step_end_ns"]) - int(row["update_start_ns"]))
            / 1_000_000_000.0
            if include_observed_scheduler_cpu
            else 0.0
        )
        plan_total_s += plan_s
        update_total_s += update_s
        graph_barrier_wait_s = 0.0
        if include_graph_replay_barrier:
            observation = row["graph_execution_observation"]
            if not isinstance(observation, Mapping):
                raise ValueError(
                    "graph replay barrier diagnostic requires a Graph observation")
            runtime_mode = observation["runtime_mode"]
            if runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
                raise ValueError(
                    "graph replay barrier diagnostic encountered an invalid "
                    f"runtime mode: {runtime_mode!r}")
            if runtime_mode in ("FULL", "PIECEWISE") and execution_ends:
                # Replay is invoked after Scheduler planning.  A source-level
                # synchronize blocks that invocation until all prior work on
                # the stream has completed.  The device itself remains serial;
                # only the Host dispatch clock is delayed here.
                nominal_replay_s = schedule_start_s + plan_s
                graph_barrier_wait_s = max(
                    0.0, execution_ends[-1] - nominal_replay_s)
        graph_barrier_waits.append(graph_barrier_wait_s)
        schedule_end_s = schedule_start_s + plan_s + graph_barrier_wait_s
        execution_start_s = max(schedule_end_s, execution_ends[-1] if execution_ends else 0.0)
        execution_end_s = execution_start_s + service_s
        completion_s = execution_end_s + update_s
        schedule_ends.append(schedule_end_s)
        execution_ends.append(execution_end_s)
        completion_times.append(completion_s)
    if set(first_step_by_request) != set(range(len(requests))):
        raise ValueError("captured trajectory did not admit every request")
    return completion_times, {
        "service_total_s": sum(service_times_s),
        "observed_scheduler_plan_cpu_total_s": plan_total_s,
        "observed_scheduler_update_cpu_total_s": update_total_s,
        "replayed_end_time_s": completion_times[-1],
        "graph_replay_barrier_enabled": include_graph_replay_barrier,
        "graph_replay_barrier_wait_total_s": sum(graph_barrier_waits),
        "graph_replay_barrier_wait_max_s": (
            max(graph_barrier_waits) if graph_barrier_waits else 0.0
        ),
        "graph_replay_barrier_wait_count": sum(
            wait_s > 0.0 for wait_s in graph_barrier_waits
        ),
        "graph_replay_barrier_waits_s": graph_barrier_waits,
    }


def observed_serialized_service_times(
    scheduler_rows: Sequence[Mapping[str, object]],
) -> tuple[list[float], dict[str, object]]:
    """Derive serialized model/runtime service from Scheduler timestamps."""

    durations = []
    by_phase: defaultdict[str, float] = defaultdict(float)
    queued_step_count = 0
    previous_execution_end_ns = 0
    for row in scheduler_rows:
        schedule_end_ns = int(row["schedule_end_ns"])
        if schedule_end_ns < previous_execution_end_ns:
            queued_step_count += 1
        execution_start_ns = max(schedule_end_ns, previous_execution_end_ns)
        execution_end_ns = int(row["update_start_ns"])
        if execution_end_ns <= execution_start_ns:
            raise ValueError("Scheduler timestamps imply non-positive service time")
        duration_s = (execution_end_ns - execution_start_ns) / 1_000_000_000.0
        durations.append(duration_s)
        by_phase[str(row["phase"])] += duration_s
        previous_execution_end_ns = execution_end_ns
    return durations, {
        "definition": (
            "update_start - max(schedule_end, previous_update_start); this is "
            "serialized model/runtime service, not pure Kernel duration"
        ),
        "queued_step_count": queued_step_count,
        "total_s": sum(durations),
        "by_phase_s": dict(sorted(by_phase.items())),
    }


def summarize_prediction_totals(
    scheduler_rows: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate predicted device and optional Host-launch time by phase."""

    fields = [("device", "predicted_timeline_ns")]
    if predictions and "predicted_device_plus_launch_ns" in predictions[0]:
        fields.append(("device_plus_launch", "predicted_device_plus_launch_ns"))
    result = {}
    for name, field in fields:
        by_phase: defaultdict[str, float] = defaultdict(float)
        for row, prediction in zip(scheduler_rows, predictions, strict=True):
            by_phase[str(row["phase"])] += float(prediction[field]) / 1_000_000_000.0
        result[name] = {
            "total_s": sum(by_phase.values()),
            "by_phase_s": dict(sorted(by_phase.items())),
        }
    return result


def compare_step_service_totals(
    observed: Mapping[str, object], predictions: Mapping[str, object]
) -> dict[str, object]:
    """Compare aggregate predicted service with serialized observed service."""

    observed_total_s = float(observed["total_s"])
    observed_by_phase = _mapping(observed["by_phase_s"], "observed service by phase")
    result = {}
    for mode, raw_prediction in predictions.items():
        prediction = _mapping(raw_prediction, f"{mode} prediction")
        predicted_total_s = float(prediction["total_s"])
        predicted_by_phase = _mapping(
            prediction["by_phase_s"], f"{mode} prediction by phase"
        )
        if set(predicted_by_phase) != set(observed_by_phase):
            raise ValueError(f"{mode} prediction phases differ from observed phases")
        result[mode] = {
            "overall": {
                "observed_serialized_service_s": observed_total_s,
                "predicted_s": predicted_total_s,
                "relative_error_percent": 100.0
                * (predicted_total_s - observed_total_s)
                / observed_total_s,
            },
            "by_phase": {
                phase: {
                    "observed_serialized_service_s": float(observed_by_phase[phase]),
                    "predicted_s": float(predicted_by_phase[phase]),
                    "relative_error_percent": 100.0
                    * (
                        float(predicted_by_phase[phase])
                        - float(observed_by_phase[phase])
                    )
                    / float(observed_by_phase[phase]),
                }
                for phase in observed_by_phase
            },
        }
    return result


def attribute_latency_gap(
    measured: Mapping[str, object],
    trace_reconstructed: Mapping[str, object],
    scheduling_trajectory_prediction: Mapping[str, object],
    standard_prediction: Mapping[str, object],
) -> dict[str, object]:
    """Split standard-to-measured latency gaps into exact additive components."""

    result = {}
    for field in LATENCY_FIELDS:
        actual = float(measured["latency"][field]["mean"])
        trace = float(trace_reconstructed["latency"][field]["mean"])
        replayed = float(
            scheduling_trajectory_prediction["latency"][field]["mean"]
        )
        standard = float(standard_prediction["latency"][field]["mean"])
        total = actual - standard
        components = {
            "scheduler_trajectory_and_batching_ms": replayed - standard,
            "step_service_and_queueing_on_scheduling_trajectory_ms": trace - replayed,
            "request_metric_boundary_residual_ms": actual - trace,
        }
        result[field] = {
            "measured_ms": actual,
            "standard_prediction_ms": standard,
            "scheduling_trajectory_replay_prediction_ms": replayed,
            "trace_reconstructed_ms": trace,
            "total_gap_ms": total,
            "measured_over_standard_prediction": actual / standard,
            "additive_components_ms": components,
            "additive_component_share_of_total_gap": {
                name: value / total for name, value in components.items()
            },
            "closure_error_ms": total - sum(components.values()),
        }
    return result


def compare_request_metrics(
    observed: Mapping[str, object], predicted: Mapping[str, object]
) -> dict[str, object]:
    """Compare scheduling-trajectory replay metrics with trace reconstruction."""

    observed_throughput = float(observed["request_throughput_per_s"])
    predicted_throughput = float(predicted["request_throughput_per_s"])
    result = {
        "request_throughput_per_s": {
            "observed": observed_throughput,
            "predicted": predicted_throughput,
            "relative_error_percent": 100.0
            * (predicted_throughput - observed_throughput)
            / observed_throughput,
        },
        "latency": {},
    }
    for field in LATENCY_FIELDS:
        observed_mean = float(observed["latency"][field]["mean"])
        predicted_mean = float(predicted["latency"][field]["mean"])
        result["latency"][field] = {
            "observed_mean_ms": observed_mean,
            "predicted_mean_ms": predicted_mean,
            "relative_error_percent": 100.0
            * (predicted_mean - observed_mean)
            / observed_mean,
        }
    return result


def analyze_scheduling_trajectory_replay(
    repository_root: Path,
    config_path: Path,
    cell_id: str,
    scheduler_trace_path: Path,
    requests: Sequence[Mapping[str, object]],
    model_config_sha256: str,
    *,
    scheduler_schema_version: int | None = 3,
    host_launch_profile: HostLaunchProfile | None = None,
    execution_mode: str = "accelerated",
    pipeline_depth: int,
    pipeline_depth_identity: Mapping[str, object],
) -> dict[str, object]:
    """Replay an observed scheduling trajectory and return JSON-ready data."""

    if not isinstance(pipeline_depth_identity, Mapping):
        raise TypeError("pipeline depth identity must be a mapping")
    if set(pipeline_depth_identity) != {
            "configured_depth", "resolved_depth", "mode", "evidence"}:
        raise ValueError("pipeline depth identity fields are invalid")
    if pipeline_depth_identity["resolved_depth"] != pipeline_depth:
        raise ValueError("pipeline depth identity disagrees with resolved depth")
    if (isinstance(pipeline_depth_identity["configured_depth"], bool)
            or not isinstance(pipeline_depth_identity["configured_depth"], int)
            or pipeline_depth_identity["configured_depth"] < 1):
        raise ValueError("configured pipeline depth must be a positive integer")
    if pipeline_depth_identity["mode"] not in (
            "dataset_native", "explicit_counterfactual"):
        raise ValueError("pipeline depth identity mode is invalid")
    if not isinstance(pipeline_depth_identity["evidence"], Mapping):
        raise TypeError("pipeline depth evidence must be a mapping")
    scheduler_rows, scheduler_summary = load_and_validate_scheduler_trace(
        scheduler_trace_path, requests, schema_version=scheduler_schema_version
    )
    predictions, contributions, predictor_identity = predict_scheduling_trajectory_steps(
        repository_root,
        config_path,
        cell_id,
        scheduler_rows,
        requests,
        model_config_sha256,
        host_launch_profile=host_launch_profile,
        execution_mode=execution_mode,
    )
    service_modes = {
        "device_only": (
            [float(value["predicted_timeline_ns"]) / 1_000_000_000.0 for value in predictions],
            False,
        )
    }
    if host_launch_profile is not None:
        device_plus_launch = [
            float(value["predicted_device_plus_launch_ns"]) / 1_000_000_000.0
            for value in predictions
        ]
        service_modes["device_plus_launch"] = (device_plus_launch, False)
        service_modes["device_plus_launch_plus_observed_scheduler_cpu"] = (
            device_plus_launch,
            True,
        )
    modes = {}
    for name, (service_times_s, include_scheduler_cpu) in service_modes.items():
        completion_times, timing = replay_scheduling_trajectory(
            scheduler_rows,
            requests,
            service_times_s,
            include_observed_scheduler_cpu=include_scheduler_cpu,
            pipeline_depth=pipeline_depth,
        )
        metrics = reconstruct_request_metrics(
            scheduler_rows, requests, completion_times
        )
        metrics["ttft_decomposition"] = reconstruct_ttft_decomposition(
            scheduler_rows,
            requests,
            completion_times,
            service_times_s,
        )
        modes[name] = {
            "timing": timing,
            "metrics": metrics,
        }
    graph_barrier_diagnostic = None
    if host_launch_profile is not None:
        diagnostic_name = "device_plus_launch_plus_observed_scheduler_cpu"
        diagnostic_services, diagnostic_scheduler_cpu = service_modes[
            diagnostic_name]
        diagnostic_completion_times, diagnostic_timing = (
            replay_scheduling_trajectory(
                scheduler_rows,
                requests,
                diagnostic_services,
                include_observed_scheduler_cpu=diagnostic_scheduler_cpu,
                pipeline_depth=pipeline_depth,
                include_graph_replay_barrier=True,
            )
        )
        diagnostic_metrics = reconstruct_request_metrics(
            scheduler_rows, requests, diagnostic_completion_times
        )
        graph_barrier_diagnostic = {
            "status": "diagnostic_counterfactual_not_forward",
            "service_mode": diagnostic_name,
            "source_contract": {
                "mechanism": "ACLGraphWrapper.current_stream_synchronize_before_replay",
                "runtime_modes": ("FULL", "PIECEWISE"),
                "elision": (
                    "runtime NONE and draft EAGLE paths do not "
                    "consume this barrier"
                ),
                "pipeline_depth": pipeline_depth,
            },
            "timing": diagnostic_timing,
            "metrics": diagnostic_metrics,
            "interpretation": (
                "The barrier wait is carried across adjacent predicted steps "
                "using only request arrivals, pipeline depth, source graph "
                "mode, and Forward service. It is an evaluation diagnostic; "
                "no observed step duration is consumed and no additive wait "
                "parameter is inserted into production Forward."
            ),
        }
    observed_times, observed_summary = observed_serialized_service_times(scheduler_rows)
    trace_metrics = reconstruct_request_metrics(
        scheduler_rows,
        requests,
        [int(row["step_end_ns"]) / 1_000_000_000.0 for row in scheduler_rows],
    )
    trace_metrics["ttft_decomposition"] = reconstruct_ttft_decomposition(
        scheduler_rows,
        requests,
        [int(row["step_end_ns"]) / 1_000_000_000.0 for row in scheduler_rows],
        observed_times,
    )
    waiting_preserved_modes = {}
    for name, (service_times_s, _) in service_modes.items():
        waiting_preserved_modes[name] = {
            "metrics": substitute_service_preserving_observed_waiting(
                scheduler_rows,
                requests,
                observed_times,
                service_times_s,
            )
        }
    for mode in modes.values():
        mode["comparison_to_trace_reconstructed"] = compare_request_metrics(
            trace_metrics, mode["metrics"]
        )
    prediction_totals = summarize_prediction_totals(scheduler_rows, predictions)
    step_comparison = []
    for row, observed_s, prediction in zip(
        scheduler_rows, observed_times, predictions, strict=True
    ):
        predicted_s = float(prediction["predicted_timeline_ns"]) / 1_000_000_000.0
        step_comparison.append(
            {
                "step_index": row["step_index"],
                "step_id": row["step_id"],
                "phase": row["phase"],
                "step_fingerprint": prediction["step_fingerprint"],
                "observed_serialized_service_s": observed_s,
                "predicted_device_s": predicted_s,
                "prediction_over_observed": predicted_s / observed_s,
                "relative_error_percent": 100.0 * (predicted_s - observed_s) / observed_s,
            }
        )
    return {
        "schema_version": 5,
        "analysis_kind": "scheduling_trajectory_replay",
        "cell_id": cell_id,
        "forward_duration_observation_consumed": False,
        "contract": {
            "schedule_membership_and_order": "observed_and_replayed",
            "step_duration": "current_zero_fit_forward",
            "observed_duration_role": "evaluation_comparison_only",
            "pipeline_depth": dict(pipeline_depth_identity),
        },
        "source_sha256": {
            "scheduler_trace": _sha256(scheduler_trace_path),
            "evaluation_config": _sha256(config_path),
        },
        "scheduler_trace": scheduler_summary,
        "predictor_source": predictor_identity,
        "trace_reconstructed": trace_metrics,
        "waiting_preserved_service_substitution_modes": waiting_preserved_modes,
        "scheduling_trajectory_replay_modes": modes,
        "graph_replay_barrier_diagnostic": graph_barrier_diagnostic,
        "step_service": {
            "observed": observed_summary,
            "prediction": prediction_totals,
            "comparison": compare_step_service_totals(
                observed_summary, prediction_totals
            ),
        },
        "step_comparison": step_comparison,
        "predicted_kernel_contributions_by_phase": contributions,
    }
