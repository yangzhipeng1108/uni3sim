"""Build one schema-v6 PD compact candidate from a measured 1P1D run."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Mapping, Sequence

from .compact_candidate_writer import AvailabilityPayload
from .compact_case import (
    CANONICAL_REQUEST_COLUMNS,
    CanonicalRequestRow,
    raw_source_set_sha256,
    read_canonical_request_rows,
)
from .pd_compact import (
    PDCompactCasePayload,
    write_pd_disaggregated_candidate,
)
from dataset_pipeline.serving_dataset.json_shape import compact_mapping
from dataset_pipeline.serving_dataset.json_shape import compact_sequence


_ROLES = ("prefill", "decode")
_REQUEST_ID = re.compile(r"request-([0-9]+)$")
_RUN_CONTRACT_SCHEMA_VERSION = 1
_TPOT_NUMERICAL_TOLERANCE_MS = 1e-3
_TRANSFER_ROLE_BY_CONNECTOR = {
    "P2pNcclConnector": "decode",
    "P2pMcclConnector": "prefill",
    "P2pHcclConnector": "prefill",
    "MooncakeConnector": "prefill",
    "MooncakeConnectorV1": "decode",
}


@dataclass(frozen=True)
class PDRunArtifacts:
    """Raw files needed to publish one measured PD compact candidate."""

    source_case_dir: Path
    frozen_manifest_path: Path
    frozen_requests_path: Path
    run_contract_path: Path
    ground_truth_path: Path
    observer_effect_report_path: Path
    prefill_scheduler_path: Path
    decode_scheduler_path: Path
    router_handoffs_path: Path
    connector_event_paths: tuple[Path, ...]
    raw_root: Path


def build_pd_disaggregated_candidate(
    output_dir: str | Path,
    *,
    artifacts: PDRunArtifacts,
    candidate_root: str | Path | None = None,
) -> Path:
    """Convert one successful measured run without altering canonical data.

    ``source_case_dir`` is read only to prove that the copied frozen request
    inputs are byte-identical to the corresponding colocated case.  Every raw
    file consumed by the candidate must instead live under ``raw_root`` so the
    resulting inventory is stable and independently reviewable.
    """

    _validate_artifacts(artifacts)
    source_manifest = _read_object(
        artifacts.source_case_dir / "manifest.json", "source case manifest"
    )
    source_requests_path = artifacts.source_case_dir / "requests.csv"
    if not source_requests_path.is_file():
        raise ValueError("source case has no requests.csv")
    if source_requests_path.read_bytes() != artifacts.frozen_requests_path.read_bytes():
        raise ValueError("frozen PD requests differ from the source colocated case")
    requests = read_canonical_request_rows(artifacts.frozen_requests_path)
    contract = _load_run_contract(artifacts.run_contract_path)
    observer_effect = _load_observer_effect_report(
        artifacts.observer_effect_report_path
    )
    _validate_contract_matches_source(contract, source_manifest, requests)
    scheduler_paths = {
        "prefill": artifacts.prefill_scheduler_path,
        "decode": artifacts.decode_scheduler_path,
    }
    scheduler_rows = {
        role: _read_jsonl(scheduler_paths[role], f"{role} Scheduler trace")
        for role in _ROLES
    }
    capture_identity = _capture_identity(contract)
    for role in _ROLES:
        _validate_raw_scheduler_identity(
            scheduler_rows[role], role=role, capture_identity=capture_identity
        )
    runtime = _runtime_from_contract(contract, scheduler_rows)
    router_rows = _read_jsonl(artifacts.router_handoffs_path, "router handoff log")
    connector_rows = tuple(
        row
        for path in artifacts.connector_event_paths
        for row in _read_jsonl(path, "connector handoff log")
    )
    _validate_connector_identity(connector_rows, capture_identity)
    raw_sources = _raw_sources(artifacts)
    raw_by_path = {
        source["relative_path"]: source for source in raw_sources
    }
    source_ids = _source_ids(artifacts, raw_by_path)
    ground_truth = _read_object(artifacts.ground_truth_path, "PD ground truth")
    metric_payload = _metric_payload(
        ground_truth,
        requests,
        source_archive_id=source_ids["ground_truth"],
        source_sha256=raw_by_path[
            _raw_relative(artifacts.ground_truth_path, artifacts.raw_root)
        ]["sha256"],
    )
    handoff_rows = _build_handoffs(
        router_rows=router_rows,
        connector_rows=connector_rows,
        decode_scheduler_rows=scheduler_rows["decode"],
        request_count=len(requests),
        capture_identity=capture_identity,
    )
    _validate_handoff_kv_tokens(
        handoff_rows, scheduler_rows["prefill"], len(requests)
    )
    request_field_provenance = _rebound_request_field_provenance(
        source_manifest, source_ids["frozen_requests"]
    )
    cached_hit_evidence = _pd_cached_hit_evidence(
        source_manifest,
        raw_sources=raw_sources,
        source_archive_ids=(
            source_ids["frozen_requests"],
            source_ids["scheduler_prefill"],
            source_ids["scheduler_decode"],
        ),
    )
    capture = {
        **capture_identity,
        "requests_source_archive_id": source_ids["frozen_requests"],
        "metrics_source_archive_id": source_ids["ground_truth"],
        "scheduler_source_archive_ids": [
            source_ids["scheduler_prefill"],
            source_ids["scheduler_decode"],
        ],
        "handoff_source_archive_ids": [
            source_ids["router_handoffs"],
            *source_ids["connector_events"],
        ],
        "scheduler_trace_enabled": True,
        "profiler_attached": False,
        "observer_effect_source_archive_id": source_ids["observer_effect"],
        "observer_effect_validation": observer_effect,
    }
    scheduler_provenance = {
        "prefill": _scheduler_provenance(
            raw_by_path[_raw_relative(artifacts.prefill_scheduler_path, artifacts.raw_root)],
            capture_identity["run_id"],
        ),
        "decode": _scheduler_provenance(
            raw_by_path[_raw_relative(artifacts.decode_scheduler_path, artifacts.raw_root)],
            capture_identity["run_id"],
        ),
    }
    router_source = raw_by_path[
        _raw_relative(artifacts.router_handoffs_path, artifacts.raw_root)
    ]
    connector_sources = [
        raw_by_path[_raw_relative(path, artifacts.raw_root)]
        for path in artifacts.connector_event_paths
    ]
    handoff_provenance = {
        "router_source_archive_id": router_source["archive_id"],
        "router_source_sha256": router_source["sha256"],
        "connector_source_archive_ids": [
            source["archive_id"] for source in connector_sources
        ],
        "connector_source_sha256s": [
            source["sha256"] for source in connector_sources
        ],
        "profiler_attached": False,
    }
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    return write_pd_disaggregated_candidate(
        output,
        PDCompactCasePayload(
            identity=_mapping(contract["identity"], "PD run identity"),
            runtime=runtime,
            requests=requests,
            request_field_provenance=request_field_provenance,
            cached_hit_value_evidence=cached_hit_evidence,
            metrics=AvailabilityPayload(
                status="available", reason=None, payload=metric_payload
            ),
            scheduler_steps={
                role: _without_schema_version(scheduler_rows[role], role)
                for role in _ROLES
            },
            scheduler_provenance=scheduler_provenance,
            pd_handoffs=handoff_rows,
            pd_handoffs_provenance=handoff_provenance,
            raw_sources=raw_sources,
            provenance_gaps=_string_list(
                contract["provenance_gaps"], "PD run provenance_gaps"
            ),
            converter={
                "implementation_id": (
                    "dataset_pipeline.serving_dataset.pd_compact_candidate."
                    "build_pd_disaggregated_candidate"
                ),
                "source_commit": _converter_source_commit(),
                "selection_rule": "physical_pd_role_trajectory_v1",
            },
            capture=capture,
        ),
        candidate_root=candidate_root,
    )


def _validate_artifacts(artifacts: PDRunArtifacts) -> None:
    if not isinstance(artifacts, PDRunArtifacts):
        raise TypeError("artifacts must be PDRunArtifacts")
    raw_root = artifacts.raw_root.resolve()
    if not raw_root.is_dir():
        raise ValueError(f"PD raw root is not a directory: {raw_root}")
    if not artifacts.source_case_dir.is_dir():
        raise ValueError("PD source case directory is missing")
    required = (
        artifacts.frozen_manifest_path,
        artifacts.frozen_requests_path,
        artifacts.run_contract_path,
        artifacts.ground_truth_path,
        artifacts.observer_effect_report_path,
        artifacts.prefill_scheduler_path,
        artifacts.decode_scheduler_path,
        artifacts.router_handoffs_path,
        *artifacts.connector_event_paths,
    )
    if not artifacts.connector_event_paths:
        raise ValueError("PD candidate requires connector event files")
    resolved = set()
    for path in required:
        absolute = path.resolve()
        if not absolute.is_file():
            raise ValueError(f"PD raw input is missing: {absolute}")
        if raw_root not in absolute.parents:
            raise ValueError(f"PD raw input escapes raw root: {absolute}")
        if absolute in resolved:
            raise ValueError(f"PD raw input is duplicated: {absolute}")
        resolved.add(absolute)


def _load_run_contract(path: Path) -> Mapping[str, object]:
    contract = _read_object(path, "PD run contract")
    _exact_keys(
        contract,
        (
            "schema_version", "kind", "identity", "runtime",
            "capture_identity", "observer_effect_validation", "provenance_gaps",
        ),
        "PD run contract",
    )
    if contract["schema_version"] != _RUN_CONTRACT_SCHEMA_VERSION:
        raise ValueError("PD run contract schema_version is invalid")
    if contract["kind"] != "sharegpt_pd_run_contract":
        raise ValueError("PD run contract kind is invalid")
    _capture_identity(contract)
    _runtime_contract(contract)
    return contract


def _load_observer_effect_report(path: Path) -> Mapping[str, object]:
    report = _read_object(path, "PD observer-effect report")
    _exact_keys(
        report,
        (
            "status", "validation_case_id", "baseline_run_id",
            "instrumented_run_id", "threshold_percent", "metric_delta_percent",
            "passed",
        ),
        "PD observer-effect report",
    )
    if report["status"] not in (
        "measured_for_this_case", "inherited_from_representative_case",
    ):
        raise ValueError("PD observer-effect status is invalid")
    for field in ("validation_case_id", "baseline_run_id", "instrumented_run_id"):
        _text(report[field], f"PD observer-effect {field}")
    if report["threshold_percent"] != 3.0:
        raise ValueError("PD observer-effect threshold must equal three percent")
    deltas = _mapping(report["metric_delta_percent"], "PD observer-effect deltas")
    _exact_keys(
        deltas,
        ("request_throughput", "output_token_throughput", "ttft", "tpot", "e2e"),
        "PD observer-effect deltas",
    )
    for field, value in deltas.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"PD observer-effect {field} must be non-negative")
        if value > 3.0:
            raise ValueError("PD observer-effect exceeds the three-percent gate")
    if report["passed"] is not True:
        raise ValueError("PD observer-effect must pass")
    return report


def _capture_identity(contract: Mapping[str, object]) -> Mapping[str, object]:
    capture = _mapping(contract["capture_identity"], "PD run capture identity")
    _exact_keys(
        capture, ("run_id", "run_origin_monotonic_ns", "host_boot_id"),
        "PD run capture identity",
    )
    _text(capture["run_id"], "PD run ID")
    _integer(capture["run_origin_monotonic_ns"], "PD run origin", minimum=0)
    _text(capture["host_boot_id"], "PD run host boot ID")
    return capture


def _runtime_contract(contract: Mapping[str, object]) -> None:
    runtime = _mapping(contract["runtime"], "PD run runtime")
    _exact_keys(
        runtime,
        (
            "dtype", "deployment", "target_qps", "seed", "runtime_version",
            "source_commit", "capture_enable_prefix_caching", "instances",
        ),
        "PD run runtime",
    )
    instances = _mapping(runtime["instances"], "PD run instances")
    _exact_keys(instances, _ROLES, "PD run instances")
    for role in _ROLES:
        instance = _mapping(instances[role], f"PD run {role} instance")
        _exact_keys(
            instance,
            (
                "tensor_parallel_size", "expert_parallel_size",
                "pipeline_parallel_size", "graph",
            ),
            f"PD run {role} instance",
        )
        graph = _mapping(instance["graph"], f"PD run {role} graph")
        _exact_keys(
            graph, ("configured_mode", "capture_limit_tokens"),
            f"PD run {role} graph",
        )


def _validate_contract_matches_source(
    contract: Mapping[str, object],
    source_manifest: Mapping[str, object],
    requests: Sequence[CanonicalRequestRow],
) -> None:
    source_identity = _mapping(source_manifest["identity"], "source identity")
    identity = _mapping(contract["identity"], "PD run identity")
    for field in ("case_id", "hardware_id", "model_id"):
        if identity[field] != source_identity[field]:
            raise ValueError(f"PD run identity {field} differs from source case")
    if identity["source_kind"] != "sharegpt_pd_disaggregated":
        raise ValueError("PD run source_kind is invalid")
    if identity["topology"] != "1p1d":
        raise ValueError("PD run topology is invalid")
    source_requests = _mapping(source_manifest["requests"], "source requests")
    if source_requests["row_count"] != len(requests):
        raise ValueError("PD frozen requests row count disagrees with source case")


def _validate_raw_scheduler_identity(
    rows: Sequence[Mapping[str, object]],
    *,
    role: str,
    capture_identity: Mapping[str, object],
) -> None:
    if not rows:
        raise ValueError(f"PD {role} Scheduler trace is empty")
    for index, row in enumerate(rows):
        for field in ("run_id", "run_origin_monotonic_ns", "host_boot_id"):
            if row[field] != capture_identity[field]:
                raise ValueError(
                    f"PD {role} Scheduler trace row {index} has a different {field}"
                )
        if row["instance_role"] != role:
            raise ValueError(f"PD Scheduler trace has the wrong role: {role}")
        if row["clock_domain"] != "shared_host_time.perf_counter_ns":
            raise ValueError("PD Scheduler trace clock domain is invalid")


def _runtime_from_contract(
    contract: Mapping[str, object],
    scheduler_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> Mapping[str, object]:
    source = _mapping(contract["runtime"], "PD run runtime")
    instances_source = _mapping(source["instances"], "PD run instances")
    instances: dict[str, object] = {}
    for role in _ROLES:
        rows = scheduler_rows[role]
        evidence = rows[0]["scheduler_pipeline_depth_evidence"]
        for row in rows:
            if row["scheduler_pipeline_depth_evidence"] != evidence:
                raise ValueError(f"PD {role} Scheduler depth evidence changed in one run")
        source_instance = _mapping(instances_source[role], f"PD run {role} instance")
        instances[role] = {
            **dict(source_instance),
            "scheduler_pipeline_depth": _mapping(
                evidence, f"PD {role} Scheduler depth evidence"
            )["configured_depth"],
            "scheduler_pipeline_depth_evidence": evidence,
        }
    return {**dict(source), "instances": instances}


def _validate_connector_identity(
    rows: Sequence[Mapping[str, object]], capture_identity: Mapping[str, object]
) -> None:
    if not rows:
        raise ValueError("PD connector evidence is empty")
    for index, row in enumerate(rows):
        for field in ("run_id", "run_origin_monotonic_ns", "host_boot_id"):
            if row[field] != capture_identity[field]:
                raise ValueError(
                    f"PD connector row {index} has a different {field}"
                )
        if row["clock_domain"] != "shared_host_time.perf_counter_ns":
            raise ValueError("PD connector event clock domain is invalid")


def _metric_payload(
    ground_truth: Mapping[str, object],
    requests: Sequence[CanonicalRequestRow],
    *,
    source_archive_id: object,
    source_sha256: object,
) -> Mapping[str, object]:
    if ground_truth["status"] != "success":
        raise ValueError("PD ground truth did not succeed")
    metrics = _mapping(ground_truth["metrics"], "PD ground-truth metrics")
    rows = _sequence(metrics["per_request"], "PD ground-truth per_request")
    if len(rows) != len(requests):
        raise ValueError("PD ground-truth request count is inconsistent")
    converted = []
    for expected_idx, (value, request) in enumerate(zip(rows, requests, strict=True)):
        row = _mapping(value, f"PD ground-truth request {expected_idx}")
        if _request_index(row["request_id"], "PD ground-truth request ID") != expected_idx:
            raise ValueError("PD ground-truth request order is not canonical")
        if row["output_tokens"] != request.output_tokens:
            raise ValueError("PD ground-truth output token count differs from frozen input")
        converted.append({
            "request_idx": expected_idx,
            "actual_output_tokens": row["output_tokens"],
            "ttft_ms": row["ttft_ms"],
            "tpot_ms": _normalized_tpot_ms(row["tpot_ms"]),
            "e2e_latency_ms": row["e2e_ms"],
        })
    duration_s = metrics["duration_s"]
    if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool) or duration_s <= 0:
        raise ValueError("PD ground-truth duration must be positive")
    input_tokens = sum(request.input_tokens for request in requests)
    output_tokens = sum(request.output_tokens for request in requests)
    return {
        "metric_provenance": {
            "source_kind": "serving_runtime",
            "source_archive_id": source_archive_id,
            "source_sha256": source_sha256,
            "profiler_attached": False,
        },
        "per_request": converted,
        "summary": {
            "request_count": len(requests),
            "request_throughput_per_s": metrics["request_throughput_per_s"],
            "output_token_throughput_per_s": metrics["output_token_throughput_per_s"],
            "total_token_throughput_per_s": (input_tokens + output_tokens) / duration_s,
        },
    }


def _normalized_tpot_ms(value: object) -> object:
    """Canonicalize only a sub-microsecond negative timestamp residual."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("PD ground-truth TPOT must be numeric")
    if -_TPOT_NUMERICAL_TOLERANCE_MS <= value < 0:
        return 0.0
    return value


def _raw_sources(artifacts: PDRunArtifacts) -> list[Mapping[str, object]]:
    entries = (
        ("frozen-manifest", "frozen_case_manifest", artifacts.frozen_manifest_path, "sharegpt_pd_runner"),
        ("frozen-requests", "frozen_requests", artifacts.frozen_requests_path, "sharegpt_pd_runner"),
        ("run-contract", "pd_run_contract", artifacts.run_contract_path, "sharegpt_pd_runner"),
        ("pd-ground-truth", "ground_truth", artifacts.ground_truth_path, "sharegpt_pd_runner"),
        ("observer-effect", "observer_effect", artifacts.observer_effect_report_path, "pd_observer_effect"),
        ("scheduler-prefill", "scheduler_prefill", artifacts.prefill_scheduler_path, "vllm_scheduler_trace"),
        ("scheduler-decode", "scheduler_decode", artifacts.decode_scheduler_path, "vllm_scheduler_trace"),
        ("router-handoffs", "router_handoffs", artifacts.router_handoffs_path, "request_router"),
        *tuple(
            (
                f"connector-events-{index}",
                "connector_handoffs",
                path,
                "vllm_scheduler_trace",
            )
            for index, path in enumerate(artifacts.connector_event_paths)
        ),
    )
    sources = []
    for archive_id, role, path, tool in entries:
        bytes_value = path.stat().st_size
        sources.append({
            "archive_id": archive_id,
            "role": role,
            "relative_path": _raw_relative(path, artifacts.raw_root),
            "bytes": bytes_value,
            "sha256": _sha256_file(path),
            "tool": tool,
            "tool_version": "1",
        })
    return sources


def _source_ids(
    artifacts: PDRunArtifacts, raw_by_path: Mapping[str, Mapping[str, object]]
) -> Mapping[str, object]:
    return {
        "frozen_requests": raw_by_path[
            _raw_relative(artifacts.frozen_requests_path, artifacts.raw_root)
        ]["archive_id"],
        "ground_truth": raw_by_path[
            _raw_relative(artifacts.ground_truth_path, artifacts.raw_root)
        ]["archive_id"],
        "observer_effect": raw_by_path[
            _raw_relative(artifacts.observer_effect_report_path, artifacts.raw_root)
        ]["archive_id"],
        "scheduler_prefill": raw_by_path[
            _raw_relative(artifacts.prefill_scheduler_path, artifacts.raw_root)
        ]["archive_id"],
        "scheduler_decode": raw_by_path[
            _raw_relative(artifacts.decode_scheduler_path, artifacts.raw_root)
        ]["archive_id"],
        "router_handoffs": raw_by_path[
            _raw_relative(artifacts.router_handoffs_path, artifacts.raw_root)
        ]["archive_id"],
        "connector_events": [
            raw_by_path[_raw_relative(path, artifacts.raw_root)]["archive_id"]
            for path in artifacts.connector_event_paths
        ],
    }


def _scheduler_provenance(
    source: Mapping[str, object], run_id: object
) -> Mapping[str, object]:
    return {
        "source_archive_id": source["archive_id"],
        "source_sha256": source["sha256"],
        "capture_run_id": run_id,
        "instrumentation_id": "dataset_pipeline.serving_dataset.vllm_scheduler_trace",
        "instrumentation_version": 1,
        "buffering": "in_memory_append_then_single_write",
        "profiler_attached": False,
    }


def _build_handoffs(
    *,
    router_rows: Sequence[Mapping[str, object]],
    connector_rows: Sequence[Mapping[str, object]],
    decode_scheduler_rows: Sequence[Mapping[str, object]],
    request_count: int,
    capture_identity: Mapping[str, object],
) -> list[Mapping[str, object]]:
    routers: dict[int, Mapping[str, object]] = {}
    for row in router_rows:
        if row["event_kind"] != "pd_router_handoff":
            raise ValueError("router handoff log contains an unknown event kind")
        request_idx = row["request_idx"]
        if request_idx is None:
            continue
        _integer(request_idx, "router request_idx", minimum=0)
        if request_idx in routers:
            raise ValueError("router handoff log repeats a formal request")
        if row["router_status"] != "completed" or row["failure"] is not None:
            raise ValueError("router handoff log contains a failed formal request")
        for field in ("run_id", "run_origin_monotonic_ns", "host_boot_id"):
            if row[field] != capture_identity[field]:
                raise ValueError("router handoff identity disagrees with capture")
        routers[request_idx] = row
    if set(routers) != set(range(request_count)):
        raise ValueError("router handoff log does not cover every formal request")
    events_by_request: dict[str, list[Mapping[str, object]]] = {}
    for row in connector_rows:
        request_id = _canonical_connector_request_id(row["base_request_id"])
        if _REQUEST_ID.fullmatch(request_id) is None:
            continue
        if request_id not in events_by_request:
            events_by_request[request_id] = []
        events_by_request[request_id].append(row)
    first_schedules = _first_schedules(decode_scheduler_rows, request_count)
    handoffs = []
    for request_idx in range(request_count):
        router = routers[request_idx]
        external_id = router["external_request_id"]
        if _request_index(external_id, "router external request ID") != request_idx:
            raise ValueError("router external request ID does not match request index")
        events = events_by_request[external_id]
        grouped = _events_by_kind(events, external_id)
        for failure in ("transfer_failed", "consumer_load_failed"):
            if failure in grouped:
                raise ValueError(f"PD connector reports {failure} for {external_id}")
        producer = _single_event(grouped, "producer_ready", external_id)
        load_scheduled = _single_event(
            grouped, "consumer_load_scheduled", external_id
        )
        transfer_starts = _events(grouped, "transfer_start", external_id)
        transfer_ends = _events(grouped, "transfer_end", external_id)
        load_starts = _events(grouped, "consumer_load_start", external_id)
        load_ends = _events(grouped, "consumer_load_end", external_id)
        _validate_handoff_event_group(
            request_id=external_id,
            producer=producer,
            load_scheduled=load_scheduled,
            transfer_starts=transfer_starts,
            transfer_ends=transfer_ends,
            load_starts=load_starts,
            load_ends=load_ends,
        )
        producer_tokens = producer["kv_tokens"]
        producer_blocks = producer["kv_blocks"]
        _integer(producer_tokens, "PD producer KV tokens", minimum=1)
        _integer(producer_blocks, "PD producer KV blocks", minimum=1)
        handoffs.append({
            "request_idx": request_idx,
            "external_request_id": external_id,
            "producer_request_id": router["producer_request_id"],
            "consumer_request_id": router["consumer_request_id"],
            "handoff_transport": router["handoff_transport"],
            "router_received_ns": router["router_received"],
            "producer_control_response_ns": router["producer_control_response"],
            "consumer_request_sent_ns": router["consumer_request_sent"],
            "consumer_first_visible_token_ns": router["consumer_first_visible_token"],
            "consumer_stream_completed_ns": router["consumer_stream_completed"],
            "producer_ready_ns": producer["timestamp_ns"],
            "transfer_start_ns": min(row["timestamp_ns"] for row in transfer_starts),
            "transfer_end_ns": max(row["timestamp_ns"] for row in transfer_ends),
            "consumer_load_start_ns": min(row["timestamp_ns"] for row in load_starts),
            "consumer_load_end_ns": max(row["timestamp_ns"] for row in load_ends),
            "consumer_first_schedule_ns": first_schedules[request_idx],
            "producer_kv_tokens": producer_tokens,
            "producer_kv_blocks": producer_blocks,
            "transfer_bytes": _transfer_bytes(transfer_starts),
            "run_id": capture_identity["run_id"],
            "run_origin_monotonic_ns": capture_identity["run_origin_monotonic_ns"],
            "host_boot_id": capture_identity["host_boot_id"],
            "status": "completed",
        })
    return handoffs


def _canonical_connector_request_id(value: object) -> str:
    _text(value, "PD connector base request ID")
    if not isinstance(value, str):
        raise AssertionError("PD connector base request ID was validated")
    if value.startswith("cmpl-"):
        return value[5:]
    return value


def _events_by_kind(
    rows: Sequence[Mapping[str, object]], request_id: str
) -> Mapping[str, list[Mapping[str, object]]]:
    if not rows:
        raise ValueError(f"PD connector has no events for {request_id}")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        event = row["event"]
        _text(event, "PD connector event")
        if event not in grouped:
            grouped[event] = []
        grouped[event].append(row)
    return grouped


def _events(
    grouped: Mapping[str, Sequence[Mapping[str, object]]], event: str, request_id: str
) -> Sequence[Mapping[str, object]]:
    if event not in grouped or not grouped[event]:
        raise ValueError(f"PD connector is missing {event} for {request_id}")
    return grouped[event]


def _single_event(
    grouped: Mapping[str, Sequence[Mapping[str, object]]], event: str, request_id: str
) -> Mapping[str, object]:
    values = _events(grouped, event, request_id)
    if len(values) != 1:
        raise ValueError(f"PD connector has multiple {event} events for {request_id}")
    return values[0]


def _validate_handoff_event_group(
    *,
    request_id: str,
    producer: Mapping[str, object],
    load_scheduled: Mapping[str, object],
    transfer_starts: Sequence[Mapping[str, object]],
    transfer_ends: Sequence[Mapping[str, object]],
    load_starts: Sequence[Mapping[str, object]],
    load_ends: Sequence[Mapping[str, object]],
) -> None:
    event_groups = (
        transfer_starts,
        transfer_ends,
        load_starts,
        load_ends,
    )
    all_events = (producer, load_scheduled, *tuple(
        row for rows in event_groups for row in rows
    ))
    connectors = {row["connector"] for row in all_events}
    if len(connectors) != 1:
        raise ValueError(f"PD connector changes within request {request_id}")
    connector = next(iter(connectors))
    if connector not in _TRANSFER_ROLE_BY_CONNECTOR:
        raise ValueError(f"PD connector is unsupported: {connector!r}")
    if producer["instance_role"] != "prefill":
        raise ValueError("PD producer_ready must come from the Prefill instance")
    if load_scheduled["instance_role"] != "decode":
        raise ValueError("PD consumer_load_scheduled must come from the Decode instance")
    expected_transfer_role = _TRANSFER_ROLE_BY_CONNECTOR[connector]
    if any(row["instance_role"] != expected_transfer_role for rows in (
        transfer_starts, transfer_ends
    ) for row in rows):
        raise ValueError("PD transfer events use the wrong instance role")
    if any(row["instance_role"] != "decode" for rows in (
        load_starts, load_ends
    ) for row in rows):
        raise ValueError("PD consumer load events must come from the Decode instance")
    tensor_flags = tuple("tensor_id" in row for rows in event_groups for row in rows)
    if any(tensor_flags):
        if not all(tensor_flags):
            raise ValueError("PD tensor handoff events must all carry tensor_id")
        paired = tuple(
            _events_by_tensor_id(rows, request_id) for rows in event_groups
        )
        tensor_ids = set(paired[0])
        if any(set(values) != tensor_ids for values in paired[1:]):
            raise ValueError("PD tensor handoff start/end sets differ")
        for tensor_id in tensor_ids:
            if paired[0][tensor_id]["timestamp_ns"] > paired[1][tensor_id]["timestamp_ns"]:
                raise ValueError("PD tensor transfer end predates its start")
            if paired[2][tensor_id]["timestamp_ns"] > paired[3][tensor_id]["timestamp_ns"]:
                raise ValueError("PD tensor load end predates its start")
        return
    if all(len(rows) == 1 for rows in event_groups):
        return
    by_process = tuple(
        _events_by_process_pid(rows, request_id) for rows in event_groups
    )
    process_ids = set(by_process[0])
    if not process_ids or any(
        set(values) != process_ids for values in by_process[1:]
    ):
        raise ValueError("PD aggregate handoff process sets differ across boundaries")
    for process_id in process_ids:
        if (
            by_process[0][process_id]["timestamp_ns"]
            > by_process[1][process_id]["timestamp_ns"]
        ):
            raise ValueError("PD aggregate transfer end predates its start")
        if (
            by_process[2][process_id]["timestamp_ns"]
            > by_process[3][process_id]["timestamp_ns"]
        ):
            raise ValueError("PD aggregate load end predates its start")


def _events_by_process_pid(
    rows: Sequence[Mapping[str, object]], request_id: str
) -> Mapping[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for row in rows:
        process_id = row["process_pid"]
        _integer(process_id, "PD aggregate handoff process PID", minimum=1)
        if not isinstance(process_id, int):
            raise AssertionError("PD aggregate handoff process PID was validated")
        if process_id in result:
            raise ValueError("PD aggregate handoff repeats a process boundary")
        result[process_id] = row
    return result


def _events_by_tensor_id(
    rows: Sequence[Mapping[str, object]], request_id: str
) -> Mapping[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        tensor_id = row["tensor_id"]
        _text(tensor_id, "PD connector tensor_id")
        if not isinstance(tensor_id, str):
            raise AssertionError("PD connector tensor_id was validated")
        if tensor_id.split("#", 1)[0] not in (request_id, f"cmpl-{request_id}"):
            raise ValueError("PD connector tensor_id disagrees with request")
        if tensor_id in result:
            raise ValueError("PD connector repeats a tensor event")
        result[tensor_id] = row
    return result


def _transfer_bytes(rows: Sequence[Mapping[str, object]]) -> object:
    byte_values = [row["byte_count"] for row in rows]
    if all(value is None for value in byte_values):
        return {
            "status": "unobserved_by_connector_runtime",
            "reason": "connector runtime did not expose transfer byte counts",
        }
    # C500 records the producer's physical tensor transfers (with bytes) and
    # the Decode runtime's aggregate transfer transition (without bytes).  The
    # latter is a duplicate timing boundary, not an unobserved payload.  Keep
    # only physically measured byte records; never synthesize a value for the
    # aggregate event.
    observed_values = [value for value in byte_values if value is not None]
    for value in observed_values:
        _integer(value, "PD transfer byte count", minimum=1)
    return sum(observed_values)


def _first_schedules(
    rows: Sequence[Mapping[str, object]], request_count: int
) -> Mapping[int, object]:
    starts: dict[int, object] = {}
    for row in rows:
        for work in _sequence(row["scheduled_requests"], "decode scheduled requests"):
            request = _mapping(work, "decode scheduled request")
            request_idx = request["request_idx"]
            _integer(request_idx, "decode Scheduler request_idx", minimum=0)
            if request_idx >= request_count:
                raise ValueError("decode Scheduler request index is out of range")
            if request_idx not in starts:
                starts[request_idx] = row["schedule_start_ns"]
    if set(starts) != set(range(request_count)):
        raise ValueError("decode Scheduler does not schedule every PD request")
    return starts


def _validate_handoff_kv_tokens(
    handoffs: Sequence[Mapping[str, object]],
    prefill_scheduler_rows: Sequence[Mapping[str, object]],
    request_count: int,
) -> None:
    prefill_tokens = [0 for _ in range(request_count)]
    for row in prefill_scheduler_rows:
        for work in _sequence(row["scheduled_requests"], "prefill scheduled requests"):
            request = _mapping(work, "prefill scheduled request")
            request_idx = request["request_idx"]
            _integer(request_idx, "prefill Scheduler request_idx", minimum=0)
            prefill_tokens[request_idx] += request["prefill_tokens"]
    for handoff in handoffs:
        request_idx = handoff["request_idx"]
        if handoff["producer_kv_tokens"] != prefill_tokens[request_idx]:
            raise ValueError("PD connector KV tokens disagree with Prefill Scheduler")


def _rebound_request_field_provenance(
    manifest: Mapping[str, object], frozen_request_archive_id: object
) -> Mapping[str, object]:
    requests = _mapping(manifest["requests"], "source requests")
    provenance = _mapping(requests["field_provenance"], "source request provenance")
    _exact_keys(provenance, CANONICAL_REQUEST_COLUMNS, "source request provenance")
    rebound: dict[str, object] = {}
    for field in CANONICAL_REQUEST_COLUMNS:
        value = _mapping(provenance[field], f"source request provenance {field}")
        rebound[field] = {
            **dict(value),
            "source_archive_id": (
                None if value["source_archive_id"] is None
                else frozen_request_archive_id
            ),
        }
    return rebound


def _pd_cached_hit_evidence(
    manifest: Mapping[str, object],
    *,
    raw_sources: Sequence[Mapping[str, object]],
    source_archive_ids: Sequence[object],
) -> Mapping[str, object] | None:
    requests = _mapping(manifest["requests"], "source requests")
    value = requests["cached_hit_value_evidence"]
    if value is None:
        return None
    source = _mapping(value, "source cached-hit evidence")
    if source["representative_value"] != 0 or source["net_prefix_work_reduction_tokens"] != 0:
        raise ValueError("PD cannot reuse a source case with non-zero prefix work")
    return {
        "kind": "scheduler_equivalent_representative",
        "logical_prefix_hit_observability": "unavailable",
        "representative_value": 0,
        "net_prefix_work_reduction_tokens": 0,
        "derivation": "pd_role_local_scheduler_token_conservation",
        "source_archive_ids": list(source_archive_ids),
        "raw_source_set_sha256": raw_source_set_sha256(raw_sources),
    }


def _without_schema_version(
    rows: Sequence[Mapping[str, object]], role: str
) -> list[Mapping[str, object]]:
    normalized = []
    for index, row in enumerate(rows):
        if "schema_version" in row:
            raise ValueError(
                f"raw PD {role} Scheduler row {index} already owns a schema version"
            )
        normalized.append(dict(row))
    return normalized


def _raw_relative(path: Path, raw_root: Path) -> str:
    return path.resolve().relative_to(raw_root.resolve()).as_posix()


def _read_jsonl(path: Path, label: str) -> tuple[Mapping[str, object], ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"{label} is empty or contains an empty row")
    return tuple(_mapping(_read_json(line, label), label) for line in lines)


def _read_object(path: Path, label: str) -> Mapping[str, object]:
    return _mapping(_read_json(path.read_text(encoding="utf-8"), label), label)


def _read_json(text: str, label: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _request_index(value: object, label: str) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be request-<index>")
    match = _REQUEST_ID.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} must be request-<index>")
    return int(match.group(1))


def _converter_source_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("git did not return a full source commit")
    return commit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    return compact_mapping(value, label)


def _sequence(value: object, label: str) -> Sequence[object]:
    return compact_sequence(value, label, expected='a sequence')


def _string_list(value: object, label: str) -> list[str]:
    values = _sequence(value, label)
    result = []
    for index, item in enumerate(values):
        _text(item, f"{label} {index}")
        result.append(item)
    return result


def _exact_keys(value: Mapping[str, object], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(f"{label} fields differ: missing={sorted(required - actual)}, extra={sorted(actual - required)}")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")


def _integer(value: object, label: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
