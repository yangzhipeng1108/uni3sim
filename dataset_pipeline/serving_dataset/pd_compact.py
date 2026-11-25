"""Schema-v6 compact cases for measured 1P1D ShareGPT Serving runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping, Sequence

from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS

from .compact_candidate_writer import AvailabilityPayload
from .compact_case import (
    CANONICAL_REQUEST_COLUMNS,
    PD_DISAGGREGATED_SCHEMA_VERSION,
    CanonicalRequestRow,
    canonical_requests_csv_bytes,
    canonical_scheduler_steps_jsonl_bytes,
    raw_source_set_sha256,
    read_canonical_request_rows,
    validate_converter_source_commit,
)
from .compressed_artifact import (
    COMPRESSION_ALGORITHM,
    read_compressed_canonical_bytes,
    write_compressed_canonical_bytes,
)
from dataset_pipeline.serving_dataset.json_shape import compact_mapping
from dataset_pipeline.serving_dataset.json_shape import compact_sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_ROOT_NAME = "serving_dataset_compaction"
_PD_ROOT_NAME = "sharegpt_pd_disaggregated"
_SHA256_LENGTH = 64
_ROLES = ("prefill", "decode")
_GRAPH_MODES = {
    "nvidia_a800": {"FULL", "PIECEWISE", "NONE"},
    "metax_c500": {"FULL", "PIECEWISE", "NONE"},
    "ascend_910b": {"FULL", "NONE"},
}
_DECODE_SCHEDULES_TERMINAL_OUTPUT_TOKEN = {
    "nvidia_a800": True,
    "metax_c500": False,
    "ascend_910b": False,
}
_SCHEDULER_EXTRA_FIELDS = {
    "nvidia_a800": {"rank_observations"},
    "metax_c500": {
        "configured_cudagraph_mode", "runtime_cudagraph_mode",
        "num_unpadded_tokens", "capture_bucket_tokens", "capture_limit_tokens",
        "padded_total_tokens", "padding_tokens", "graph_fallback_reason",
    },
    "ascend_910b": {
        "configured_cudagraph_mode", "runtime_cudagraph_mode",
        "num_unpadded_tokens", "capture_bucket_tokens", "capture_limit_tokens",
        "padded_total_tokens", "padding_tokens", "graph_fallback_reason",
    },
}


@dataclass(frozen=True)
class PDCompactCase:
    """A fully validated schema-v6 PD compact case."""

    case_dir: Path
    manifest: Mapping[str, object]
    requests: tuple[CanonicalRequestRow, ...]
    metrics: Mapping[str, object]
    scheduler_steps: Mapping[str, tuple[Mapping[str, object], ...]]
    pd_handoffs: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class PDCompactCasePayload:
    """Normalized writer input for one immutable measured PD case."""

    identity: Mapping[str, object]
    runtime: Mapping[str, object]
    requests: Sequence[CanonicalRequestRow]
    request_field_provenance: Mapping[str, object]
    cached_hit_value_evidence: Mapping[str, object] | None
    metrics: AvailabilityPayload
    scheduler_steps: Mapping[str, Sequence[Mapping[str, object]]]
    scheduler_provenance: Mapping[str, Mapping[str, object]]
    pd_handoffs: Sequence[Mapping[str, object]]
    pd_handoffs_provenance: Mapping[str, object]
    raw_sources: Sequence[Mapping[str, object]]
    provenance_gaps: Sequence[str]
    converter: Mapping[str, object]
    capture: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Mapping) or not isinstance(
            self.runtime, Mapping
        ):
            raise TypeError("PD compact identity and runtime must be objects")
        if (
            isinstance(self.requests, (str, bytes, bytearray))
            or not isinstance(self.requests, Sequence)
            or not self.requests
            or any(not isinstance(value, CanonicalRequestRow) for value in self.requests)
        ):
            raise ValueError("PD compact requests must be non-empty canonical rows")
        if not isinstance(self.metrics, AvailabilityPayload):
            raise TypeError("PD compact metrics must be AvailabilityPayload")
        if self.metrics.status != "available":
            raise ValueError("PD compact candidates require measured metrics")
        if set(self.scheduler_steps) != set(_ROLES):
            raise ValueError("PD compact scheduler steps must cover prefill/decode")
        if set(self.scheduler_provenance) != set(_ROLES):
            raise ValueError("PD compact scheduler provenance must cover prefill/decode")
        for role in _ROLES:
            values = self.scheduler_steps[role]
            if (
                isinstance(values, (str, bytes, bytearray))
                or not isinstance(values, Sequence)
                or not values
            ):
                raise ValueError(f"PD compact {role} Scheduler rows are required")
            if not isinstance(self.scheduler_provenance[role], Mapping):
                raise TypeError(f"PD compact {role} Scheduler provenance is invalid")
        if (
            isinstance(self.pd_handoffs, (str, bytes, bytearray))
            or not isinstance(self.pd_handoffs, Sequence)
            or not self.pd_handoffs
        ):
            raise ValueError("PD compact handoff rows are required")
        if not isinstance(self.pd_handoffs_provenance, Mapping):
            raise TypeError("PD compact handoff provenance must be an object")
        if (
            isinstance(self.raw_sources, (str, bytes, bytearray))
            or not isinstance(self.raw_sources, Sequence)
            or not self.raw_sources
        ):
            raise ValueError("PD compact raw sources are required")
        if (
            isinstance(self.provenance_gaps, (str, bytes, bytearray))
            or not isinstance(self.provenance_gaps, Sequence)
        ):
            raise TypeError("PD compact provenance gaps must be a sequence")
        if not isinstance(self.converter, Mapping) or not isinstance(
            self.capture, Mapping
        ):
            raise TypeError("PD compact converter and capture must be objects")


def write_pd_disaggregated_candidate(
    output_dir: str | Path,
    payload: PDCompactCasePayload,
    *,
    candidate_root: str | Path | None = None,
) -> Path:
    """Write and reload-validate one non-overwriting schema-v6 PD candidate."""

    if not isinstance(payload, PDCompactCasePayload):
        raise TypeError("payload must be PDCompactCasePayload")
    allowed_root = Path(
        _REPOSITORY_ROOT / "profiler_output" / _CANDIDATE_ROOT_NAME
        if candidate_root is None
        else candidate_root
    ).resolve()
    if (
        allowed_root.name != _CANDIDATE_ROOT_NAME
        or allowed_root.parent.name != "profiler_output"
    ):
        raise ValueError(
            "PD compact candidate root must be "
            "<execution-root>/profiler_output/serving_dataset_compaction"
        )
    if not allowed_root.is_dir():
        raise ValueError(f"PD compact candidate root is not a directory: {allowed_root}")
    destination = Path(output_dir).resolve()
    if destination == allowed_root or allowed_root not in destination.parents:
        raise ValueError("PD compact candidate destination escapes candidate root")
    identity = dict(payload.identity)
    _validate_identity(identity)
    expected_relative = (
        _PD_ROOT_NAME,
        identity["hardware_id"],
        identity["model_id"],
        identity["case_id"],
    )
    if destination.relative_to(allowed_root).parts != expected_relative:
        raise ValueError("PD compact candidate destination disagrees with identity")
    if destination.exists():
        raise FileExistsError(f"PD compact candidate already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("PD compact candidate parent does not exist")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.candidate-", dir=destination.parent)
    )
    try:
        request_bytes = canonical_requests_csv_bytes(payload.requests)
        (temporary / "requests.csv").write_bytes(request_bytes)
        request_sha = hashlib.sha256(request_bytes).hexdigest()
        metrics_entry = _write_metrics(
            temporary, payload.metrics, request_sha
        )
        scheduler_entry = {}
        for role in _ROLES:
            rows = [
                {"schema_version": PD_DISAGGREGATED_SCHEMA_VERSION, **dict(row)}
                for row in payload.scheduler_steps[role]
            ]
            filename = f"scheduler_steps_{role}.jsonl.zst"
            content = canonical_scheduler_steps_jsonl_bytes(rows)
            compressed = write_compressed_canonical_bytes(
                temporary / filename, content)
            scheduler_entry[role] = {
                "path": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
                "compression": COMPRESSION_ALGORITHM,
                "row_count": len(rows),
                "timing_boundary": _scheduler_timing_boundary(),
                "provenance": dict(payload.scheduler_provenance[role]),
            }
        handoff_rows = [
            {"schema_version": PD_DISAGGREGATED_SCHEMA_VERSION, **dict(row)}
            for row in payload.pd_handoffs
        ]
        handoff_bytes = canonical_scheduler_steps_jsonl_bytes(handoff_rows)
        (temporary / "pd_handoffs.jsonl").write_bytes(handoff_bytes)
        raw_sources = [dict(value) for value in payload.raw_sources]
        manifest = {
            "schema_version": PD_DISAGGREGATED_SCHEMA_VERSION,
            "identity": identity,
            "runtime": dict(payload.runtime),
            "requests": {
                "path": "requests.csv",
                "row_count": len(payload.requests),
                "sha256": request_sha,
                "field_provenance": dict(payload.request_field_provenance),
                "cached_hit_value_evidence": (
                    None
                    if payload.cached_hit_value_evidence is None
                    else dict(payload.cached_hit_value_evidence)
                ),
            },
            "metrics": metrics_entry,
            "alignment_steps": {
                "status": "not_collected",
                "reason": (
                    "PD has role-local Scheduling and cross-instance KV handoff; "
                    "this non-profiler capture does not observe kernel alignment"
                ),
                "path": None,
                "sha256": None,
            },
            "scheduler_steps": scheduler_entry,
            "pd_handoffs": {
                "path": "pd_handoffs.jsonl",
                "sha256": hashlib.sha256(handoff_bytes).hexdigest(),
                "row_count": len(handoff_rows),
                "timing_boundary": _handoff_timing_boundary(),
                "provenance": dict(payload.pd_handoffs_provenance),
            },
            "raw_sources": raw_sources,
            "provenance": {
                "raw_source_set_sha256": raw_source_set_sha256(raw_sources),
                "gaps": list(payload.provenance_gaps),
            },
            "converter": dict(payload.converter),
            "capture": dict(payload.capture),
        }
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        load_pd_disaggregated_case(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def load_pd_disaggregated_case(case_dir: str | Path) -> PDCompactCase:
    """Load and strictly validate one schema-v6 1P1D compact case."""

    root = Path(case_dir)
    if not root.is_dir():
        raise ValueError(f"PD compact case is not a directory: {root}")
    manifest = _read_object(root / "manifest.json")
    _validate_manifest(manifest)
    requests = read_canonical_request_rows(root / "requests.csv")
    request_entry = _mapping(manifest["requests"], "PD manifest requests")
    request_bytes = (root / "requests.csv").read_bytes()
    if request_bytes != canonical_requests_csv_bytes(requests):
        raise ValueError("PD requests.csv is not canonical serialization")
    _verify_sha(root / "requests.csv", request_entry["sha256"], "PD requests.csv")
    if request_entry["row_count"] != len(requests):
        raise ValueError("PD requests row_count disagrees with requests.csv")
    _validate_request_provenance(request_entry, manifest["raw_sources"])
    raw_source_sha_by_id = _raw_source_sha_by_id(manifest["raw_sources"])
    metrics = _load_metrics(
        root,
        manifest["metrics"],
        request_entry["sha256"],
        requests,
        raw_source_sha_by_id,
    )
    runtime = _mapping(manifest["runtime"], "PD manifest runtime")
    capture = _mapping(manifest["capture"], "PD manifest capture")
    scheduler_entries = _mapping(
        manifest["scheduler_steps"], "PD manifest scheduler_steps"
    )
    scheduler_steps = {
        role: _load_scheduler_rows(
            root,
            scheduler_entries[role],
            role=role,
            request_count=len(requests),
            request_input_tokens=tuple(row.input_tokens for row in requests),
            request_output_tokens=tuple(row.output_tokens for row in requests),
            runtime=runtime,
            capture=capture,
            hardware_id=_mapping(manifest["identity"], "PD identity")["hardware_id"],
            raw_source_sha_by_id=raw_source_sha_by_id,
        )
        for role in _ROLES
    }
    handoffs = _load_handoffs(
        root,
        manifest["pd_handoffs"],
        requests=requests,
        prefill_scheduler_rows=scheduler_steps["prefill"],
        capture=capture,
        raw_source_sha_by_id=raw_source_sha_by_id,
    )
    _validate_file_set(root)
    return PDCompactCase(
        case_dir=root,
        manifest=manifest,
        requests=requests,
        metrics=metrics,
        scheduler_steps=scheduler_steps,
        pd_handoffs=handoffs,
    )


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    _exact_keys(
        manifest,
        (
            "schema_version", "identity", "runtime", "requests", "metrics",
            "alignment_steps", "scheduler_steps", "pd_handoffs", "raw_sources",
            "provenance", "converter", "capture",
        ),
        "PD manifest",
    )
    if manifest["schema_version"] != PD_DISAGGREGATED_SCHEMA_VERSION:
        raise ValueError("PD manifest schema_version is invalid")
    identity = _mapping(manifest["identity"], "PD manifest identity")
    _validate_identity(identity)
    _validate_runtime(_mapping(manifest["runtime"], "PD manifest runtime"))
    _validate_requests_entry(_mapping(manifest["requests"], "PD manifest requests"))
    _validate_metrics_entry(_mapping(manifest["metrics"], "PD manifest metrics"))
    _validate_alignment_entry(
        _mapping(manifest["alignment_steps"], "PD manifest alignment_steps")
    )
    _validate_scheduler_entries(
        _mapping(manifest["scheduler_steps"], "PD manifest scheduler_steps")
    )
    _validate_handoff_entry(
        _mapping(manifest["pd_handoffs"], "PD manifest pd_handoffs")
    )
    raw_sources = _sequence(manifest["raw_sources"], "PD manifest raw_sources")
    raw_hash = raw_source_set_sha256(tuple(
        _mapping(value, "PD manifest raw source") for value in raw_sources
    ))
    provenance = _mapping(manifest["provenance"], "PD manifest provenance")
    _exact_keys(provenance, ("raw_source_set_sha256", "gaps"), "PD provenance")
    if provenance["raw_source_set_sha256"] != raw_hash:
        raise ValueError("PD raw source set SHA is inconsistent")
    for index, gap in enumerate(_sequence(provenance["gaps"], "PD provenance gaps")):
        _text(gap, f"PD provenance gap {index}")
    converter = _mapping(manifest["converter"], "PD manifest converter")
    _exact_keys(converter, ("implementation_id", "source_commit", "selection_rule"), "PD converter")
    _text(converter["implementation_id"], "PD converter implementation_id")
    validate_converter_source_commit(converter["source_commit"])
    if converter["selection_rule"] != "physical_pd_role_trajectory_v1":
        raise ValueError("PD converter selection rule is invalid")
    _validate_capture(_mapping(manifest["capture"], "PD manifest capture"), raw_sources)


def _validate_identity(identity: Mapping[str, object]) -> None:
    _exact_keys(
        identity,
        (
            "source_kind", "topology", "case_id", "hardware_id", "model_id",
            "raw_model_id", "model_revision", "tokenizer_revision",
        ),
        "PD identity",
    )
    if identity["source_kind"] != "sharegpt_pd_disaggregated":
        raise ValueError("PD identity source_kind is invalid")
    if identity["topology"] != "1p1d":
        raise ValueError("PD identity topology must be 1p1d")
    _text(identity["case_id"], "PD identity case_id")
    if identity["hardware_id"] not in _GRAPH_MODES:
        raise ValueError("PD identity hardware_id is unsupported")
    if identity["hardware_id"] not in CANONICAL_HARDWARE_IDS:
        raise ValueError("PD identity hardware_id is not canonical")
    _text(identity["model_id"], "PD identity model_id")
    for field in ("raw_model_id", "model_revision", "tokenizer_revision"):
        _optional_text(identity[field], f"PD identity {field}")


def _validate_runtime(runtime: Mapping[str, object]) -> None:
    _exact_keys(
        runtime,
        (
            "dtype", "deployment", "target_qps", "seed", "runtime_version",
            "source_commit", "capture_enable_prefix_caching", "instances",
        ),
        "PD runtime",
    )
    if runtime["dtype"] != "bfloat16":
        raise ValueError("PD runtime dtype must be bfloat16")
    if runtime["deployment"] != "pd_disaggregated":
        raise ValueError("PD runtime deployment is invalid")
    _finite_positive(runtime["target_qps"], "PD runtime target_qps")
    _integer(runtime["seed"], "PD runtime seed", minimum=0)
    _text(runtime["runtime_version"], "PD runtime version")
    _sha256_or_commit(runtime["source_commit"], "PD runtime source_commit")
    if runtime["capture_enable_prefix_caching"] is not True:
        raise ValueError("PD runtime must enable prefix caching")
    instances = _mapping(runtime["instances"], "PD runtime instances")
    _exact_keys(instances, _ROLES, "PD runtime instances")
    for role in _ROLES:
        instance = _mapping(instances[role], f"PD runtime {role}")
        _exact_keys(
            instance,
            (
                "tensor_parallel_size", "expert_parallel_size",
                "pipeline_parallel_size", "scheduler_pipeline_depth",
                "scheduler_pipeline_depth_evidence", "graph",
            ),
            f"PD runtime {role}",
        )
        for field in (
            "tensor_parallel_size", "expert_parallel_size", "pipeline_parallel_size"
        ):
            _integer(instance[field], f"PD runtime {role} {field}", minimum=1)
        if instance["scheduler_pipeline_depth"] != 2:
            raise ValueError("PD runtime configured Scheduler depth must equal two")
        _validate_depth_evidence(
            _mapping(
                instance["scheduler_pipeline_depth_evidence"],
                f"PD runtime {role} depth evidence",
            ),
            expected_depth=2,
            expected_pp=instance["pipeline_parallel_size"],
        )
        graph = _mapping(instance["graph"], f"PD runtime {role} graph")
        _exact_keys(graph, ("configured_mode", "capture_limit_tokens"), f"PD graph {role}")
        _text(graph["configured_mode"], f"PD graph {role} configured_mode")
        _integer(graph["capture_limit_tokens"], f"PD graph {role} capture limit", minimum=1)


def _validate_requests_entry(entry: Mapping[str, object]) -> None:
    _exact_keys(
        entry,
        ("path", "row_count", "sha256", "field_provenance", "cached_hit_value_evidence"),
        "PD requests",
    )
    if entry["path"] != "requests.csv":
        raise ValueError("PD requests path is invalid")
    _integer(entry["row_count"], "PD request row_count", minimum=1)
    _sha256(entry["sha256"], "PD requests SHA")
    provenance = _mapping(entry["field_provenance"], "PD request provenance")
    _exact_keys(provenance, CANONICAL_REQUEST_COLUMNS, "PD request provenance")
    for field in CANONICAL_REQUEST_COLUMNS:
        value = _mapping(provenance[field], f"PD request provenance {field}")
        _exact_keys(value, ("kind", "source_archive_id", "reason"), f"PD request provenance {field}")
        _text(value["kind"], f"PD request provenance {field} kind")
        _optional_text(value["source_archive_id"], f"PD request provenance {field} source")
        _optional_text(value["reason"], f"PD request provenance {field} reason")


def _validate_metrics_entry(entry: Mapping[str, object]) -> None:
    _exact_keys(entry, ("status", "reason", "path", "sha256"), "PD metrics entry")
    if entry["status"] != "available" or entry["reason"] is not None:
        raise ValueError("PD metrics must be available")
    if entry["path"] != "metrics.json":
        raise ValueError("PD metrics path is invalid")
    _sha256(entry["sha256"], "PD metrics SHA")


def _validate_alignment_entry(entry: Mapping[str, object]) -> None:
    _exact_keys(entry, ("status", "reason", "path", "sha256"), "PD alignment entry")
    if entry["status"] != "not_collected":
        raise ValueError("PD alignment status must be not_collected")
    _text(entry["reason"], "PD alignment reason")
    if entry["path"] is not None or entry["sha256"] is not None:
        raise ValueError("PD not_collected alignment must not declare a file")


def _scheduler_timing_boundary() -> Mapping[str, object]:
    return {
        "clock": "time.perf_counter_ns",
        "origin": "shared_run_origin_monotonic_ns",
        "step_start": "immediately_before_scheduler_schedule",
        "schedule_end": "immediately_after_scheduler_schedule",
        "update_start": "immediately_before_scheduler_output_update",
        "step_end": "immediately_after_scheduler_output_update",
    }


def _handoff_timing_boundary() -> Mapping[str, object]:
    return {
        "clock": "time.perf_counter_ns",
        "origin": "shared_run_origin_monotonic_ns",
        "router_received": "router_received_client_request",
        "producer_ready": "connector_observed_producer_ready",
        "transfer_start": "connector_observed_transfer_start",
        "transfer_end": "connector_observed_transfer_end",
        "consumer_load": "connector_observed_consumer_load",
        "client_first_token": "router_flushed_first_decode_sse_token",
    }


def _validate_scheduler_entries(entries: Mapping[str, object]) -> None:
    _exact_keys(entries, _ROLES, "PD Scheduler entries")
    for role in _ROLES:
        entry = _mapping(entries[role], f"PD Scheduler {role} entry")
        _exact_keys(
            entry,
            (
                "path", "sha256", "compressed_sha256", "compression",
                "row_count", "timing_boundary", "provenance",
            ),
            f"PD Scheduler {role} entry",
        )
        if entry["path"] != f"scheduler_steps_{role}.jsonl.zst":
            raise ValueError(f"PD Scheduler {role} path is invalid")
        _sha256(entry["sha256"], f"PD Scheduler {role} SHA")
        _sha256(
            entry["compressed_sha256"],
            f"PD Scheduler {role} compressed SHA",
        )
        if entry["compression"] != COMPRESSION_ALGORITHM:
            raise ValueError(f"PD Scheduler {role} compression is invalid")
        _integer(entry["row_count"], f"PD Scheduler {role} row_count", minimum=1)
        if _mapping(entry["timing_boundary"], f"PD Scheduler {role} timing") != _scheduler_timing_boundary():
            raise ValueError("PD Scheduler timing boundary is invalid")
        _validate_scheduler_provenance(
            _mapping(entry["provenance"], f"PD Scheduler {role} provenance")
        )


def _validate_scheduler_provenance(provenance: Mapping[str, object]) -> None:
    _exact_keys(
        provenance,
        (
            "source_archive_id", "source_sha256", "capture_run_id",
            "instrumentation_id", "instrumentation_version", "buffering",
            "profiler_attached",
        ),
        "PD Scheduler provenance",
    )
    _text(provenance["source_archive_id"], "PD Scheduler source archive")
    _sha256(provenance["source_sha256"], "PD Scheduler source SHA")
    _text(provenance["capture_run_id"], "PD Scheduler capture run ID")
    _text(provenance["instrumentation_id"], "PD Scheduler instrumentation ID")
    _integer(provenance["instrumentation_version"], "PD Scheduler instrumentation version", minimum=1)
    if provenance["buffering"] != "in_memory_append_then_single_write":
        raise ValueError("PD Scheduler buffering contract is invalid")
    if provenance["profiler_attached"] is not False:
        raise ValueError("PD Scheduler must not attach a profiler")


def _validate_handoff_entry(entry: Mapping[str, object]) -> None:
    _exact_keys(
        entry, ("path", "sha256", "row_count", "timing_boundary", "provenance"), "PD handoff entry"
    )
    if entry["path"] != "pd_handoffs.jsonl":
        raise ValueError("PD handoff path is invalid")
    _sha256(entry["sha256"], "PD handoff SHA")
    _integer(entry["row_count"], "PD handoff row_count", minimum=1)
    if _mapping(entry["timing_boundary"], "PD handoff timing") != _handoff_timing_boundary():
        raise ValueError("PD handoff timing boundary is invalid")
    provenance = _mapping(entry["provenance"], "PD handoff provenance")
    _exact_keys(
        provenance,
        (
            "router_source_archive_id", "router_source_sha256",
            "connector_source_archive_ids", "connector_source_sha256s",
            "profiler_attached",
        ),
        "PD handoff provenance",
    )
    _text(provenance["router_source_archive_id"], "PD router source archive")
    _sha256(provenance["router_source_sha256"], "PD router source SHA")
    archive_ids = _string_sequence(
        provenance["connector_source_archive_ids"], "PD connector source archives"
    )
    source_shas = _sequence(
        provenance["connector_source_sha256s"], "PD connector source SHAs"
    )
    if len(archive_ids) != len(source_shas):
        raise ValueError("PD connector source IDs and SHAs differ in length")
    for value in source_shas:
        _sha256(value, "PD connector source SHA")
    if provenance["profiler_attached"] is not False:
        raise ValueError("PD handoff capture must not attach a profiler")


def _validate_capture(capture: Mapping[str, object], raw_sources: Sequence[object]) -> None:
    _exact_keys(
        capture,
        (
            "run_id", "run_origin_monotonic_ns", "host_boot_id",
            "requests_source_archive_id", "metrics_source_archive_id",
            "scheduler_source_archive_ids", "handoff_source_archive_ids",
            "scheduler_trace_enabled", "profiler_attached",
            "observer_effect_source_archive_id", "observer_effect_validation",
        ),
        "PD capture",
    )
    _text(capture["run_id"], "PD capture run_id")
    _integer(capture["run_origin_monotonic_ns"], "PD capture origin", minimum=0)
    _text(capture["host_boot_id"], "PD capture host_boot_id")
    source_ids = {
        _mapping(value, "PD raw source")["archive_id"] for value in raw_sources
    }
    for field in ("requests_source_archive_id", "metrics_source_archive_id"):
        _text(capture[field], f"PD capture {field}")
        if capture[field] not in source_ids:
            raise ValueError("PD capture references an unknown raw source")
    _text(
        capture["observer_effect_source_archive_id"],
        "PD capture observer-effect source archive",
    )
    if capture["observer_effect_source_archive_id"] not in source_ids:
        raise ValueError("PD capture references an unknown observer-effect source")
    for field in ("scheduler_source_archive_ids", "handoff_source_archive_ids"):
        values = _string_sequence(capture[field], f"PD capture {field}")
        if not values or not set(values).issubset(source_ids):
            raise ValueError("PD capture source archive references are invalid")
    if capture["scheduler_trace_enabled"] is not True:
        raise ValueError("PD capture requires Scheduler tracing")
    if capture["profiler_attached"] is not False:
        raise ValueError("PD capture must not attach a profiler")
    _validate_observer_effect(
        _mapping(capture["observer_effect_validation"], "PD observer effect")
    )


def _validate_observer_effect(validation: Mapping[str, object]) -> None:
    _exact_keys(
        validation,
        (
            "status", "validation_case_id", "baseline_run_id",
            "instrumented_run_id", "threshold_percent", "metric_delta_percent",
            "passed",
        ),
        "PD observer effect",
    )
    if validation["status"] not in (
        "measured_for_this_case", "inherited_from_representative_case"
    ):
        raise ValueError("PD observer effect status is invalid")
    for field in ("validation_case_id", "baseline_run_id", "instrumented_run_id"):
        _text(validation[field], f"PD observer effect {field}")
    if validation["threshold_percent"] != 3.0:
        raise ValueError("PD observer effect threshold must equal 3.0")
    deltas = _mapping(validation["metric_delta_percent"], "PD observer effect deltas")
    _exact_keys(
        deltas,
        ("request_throughput", "output_token_throughput", "ttft", "tpot", "e2e"),
        "PD observer effect deltas",
    )
    for key, value in deltas.items():
        _finite_nonnegative(value, f"PD observer effect {key}")
        if value > 3.0:
            raise ValueError("PD observer effect exceeds the three percent gate")
    if validation["passed"] is not True:
        raise ValueError("PD observer effect must pass")


def _load_metrics(
    root: Path,
    entry_value: object,
    requests_sha: object,
    requests: Sequence[CanonicalRequestRow],
    raw_source_sha_by_id: Mapping[object, object],
) -> Mapping[str, object]:
    entry = _mapping(entry_value, "PD metrics entry")
    path = _case_file(root, entry["path"])
    _verify_sha(path, entry["sha256"], "PD metrics")
    metrics = _read_object(path)
    _exact_keys(
        metrics,
        ("schema_version", "requests_sha256", "metric_provenance", "per_request", "summary"),
        "PD metrics",
    )
    if metrics["schema_version"] != PD_DISAGGREGATED_SCHEMA_VERSION:
        raise ValueError("PD metrics schema_version is invalid")
    if metrics["requests_sha256"] != requests_sha:
        raise ValueError("PD metrics request SHA is inconsistent")
    provenance = _mapping(metrics["metric_provenance"], "PD metrics provenance")
    _exact_keys(provenance, ("source_kind", "source_archive_id", "source_sha256", "profiler_attached"), "PD metrics provenance")
    if provenance["source_kind"] != "serving_runtime":
        raise ValueError("PD metrics provenance source_kind is invalid")
    _text(provenance["source_archive_id"], "PD metrics source archive")
    _sha256(provenance["source_sha256"], "PD metrics source SHA")
    _validate_raw_source_binding(
        provenance["source_archive_id"],
        provenance["source_sha256"],
        raw_source_sha_by_id,
        "PD metrics provenance",
    )
    if provenance["profiler_attached"] is not False:
        raise ValueError("PD metrics must not use a profiler")
    rows = _sequence(metrics["per_request"], "PD metrics per_request")
    if len(rows) != len(requests):
        raise ValueError("PD metrics request count is inconsistent")
    for expected_index, value in enumerate(rows):
        row = _mapping(value, f"PD metric request {expected_index}")
        _exact_keys(row, ("request_idx", "actual_output_tokens", "ttft_ms", "tpot_ms", "e2e_latency_ms"), "PD metric request")
        if row["request_idx"] != expected_index:
            raise ValueError("PD metric request indexes must be contiguous")
        _integer(row["actual_output_tokens"], "PD actual_output_tokens", minimum=1)
        if row["actual_output_tokens"] != requests[expected_index].output_tokens:
            raise ValueError("PD metrics output tokens disagree with requests.csv")
        for field in ("ttft_ms", "tpot_ms", "e2e_latency_ms"):
            _finite_nonnegative(row[field], f"PD metric {field}")
    summary = _mapping(metrics["summary"], "PD metrics summary")
    _exact_keys(
        summary,
        (
            "request_count", "request_throughput_per_s",
            "output_token_throughput_per_s", "total_token_throughput_per_s",
        ),
        "PD metrics summary",
    )
    if summary["request_count"] != len(requests):
        raise ValueError("PD metrics summary request_count is inconsistent")
    for field in (
        "request_throughput_per_s", "output_token_throughput_per_s", "total_token_throughput_per_s"
    ):
        _finite_positive(summary[field], f"PD metrics summary {field}")
    return metrics


def _load_scheduler_rows(
    root: Path,
    entry_value: object,
    *,
    role: str,
    request_count: int,
    request_input_tokens: Sequence[int],
    request_output_tokens: Sequence[int],
    runtime: Mapping[str, object],
    capture: Mapping[str, object],
    hardware_id: object,
    raw_source_sha_by_id: Mapping[object, object],
) -> tuple[Mapping[str, object], ...]:
    entry = _mapping(entry_value, f"PD Scheduler {role} entry")
    path = _case_file(root, entry["path"])
    content = read_compressed_canonical_bytes(
        path,
        compressed_sha256=entry["compressed_sha256"],
        content_sha256=entry["sha256"],
        label=f"PD Scheduler {role}",
    )
    rows = _read_jsonl_bytes(content, f"PD Scheduler {role}")
    if entry["row_count"] != len(rows):
        raise ValueError("PD Scheduler row_count is inconsistent")
    scheduler_provenance = _mapping(
        entry["provenance"], f"PD Scheduler {role} provenance"
    )
    _validate_raw_source_binding(
        scheduler_provenance["source_archive_id"],
        scheduler_provenance["source_sha256"],
        raw_source_sha_by_id,
        f"PD Scheduler {role} provenance",
    )
    instances = _mapping(runtime["instances"], "PD runtime instances")
    instance = _mapping(instances[role], f"PD runtime {role}")
    _validate_scheduler_rows(
        rows,
        role=role,
        request_count=request_count,
        request_input_tokens=request_input_tokens,
        request_output_tokens=request_output_tokens,
        expected_depth=instance["scheduler_pipeline_depth"],
        expected_depth_evidence=instance["scheduler_pipeline_depth_evidence"],
        expected_graph=instance["graph"],
        capture=capture,
        hardware_id=hardware_id,
    )
    return rows


def _validate_scheduler_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    role: str,
    request_count: int,
    request_input_tokens: Sequence[int],
    request_output_tokens: Sequence[int],
    expected_depth: object,
    expected_depth_evidence: object,
    expected_graph: object,
    capture: Mapping[str, object],
    hardware_id: object,
) -> None:
    if hardware_id not in _GRAPH_MODES:
        raise ValueError("PD Scheduler hardware is unsupported")
    _integer(expected_depth, "PD Scheduler expected depth", minimum=1)
    base = {
        "schema_version", "step_index", "step_id", "source_iteration_id",
        "step_start_ns", "schedule_end_ns", "update_start_ns", "step_end_ns",
        "scheduled_requests", "scheduled_request_count", "prefill_request_count",
        "decode_request_count", "prefill_tokens", "decode_tokens", "scheduled_tokens",
        "finished_request_indexes", "cumulative_scheduled_tokens",
        "cumulative_completed_requests", "run_id", "run_origin_monotonic_ns",
        "host_boot_id", "instance_role", "instance_id", "clock_domain",
        "schedule_start_ns", "scheduler_update_start_ns", "scheduler_update_end_ns",
        "in_flight_step_count", "scheduler_pipeline_depth_evidence",
        "device_execution_clock", "device_execution_source",
        "device_execution_start_ns", "device_execution_end_ns",
    }
    expected_fields = base | _SCHEDULER_EXTRA_FIELDS[hardware_id]
    if (
        len(request_input_tokens) != request_count
        or len(request_output_tokens) != request_count
    ):
        raise ValueError("PD Scheduler input-token contract is inconsistent")
    graph_contract = _mapping(expected_graph, "PD Scheduler expected Graph")
    configured_graph_mode = graph_contract["configured_mode"]
    capture_limit_tokens = graph_contract["capture_limit_tokens"]
    total_prefill = [0 for _ in range(request_count)]
    total_decode = [0 for _ in range(request_count)]
    finished = set()
    cumulative_tokens = 0
    cumulative_finished = 0
    previous_source_iteration = -1
    for index, value in enumerate(rows):
        row = _mapping(value, f"PD Scheduler {role} row {index}")
        _exact_keys(row, tuple(sorted(expected_fields)), f"PD Scheduler {role} row")
        if row["schema_version"] != PD_DISAGGREGATED_SCHEMA_VERSION:
            raise ValueError("PD Scheduler row schema version is invalid")
        if row["step_index"] != index:
            raise ValueError("PD Scheduler step indexes must be contiguous")
        _text(row["step_id"], "PD Scheduler step_id")
        _integer(row["source_iteration_id"], "PD Scheduler source iteration", minimum=0)
        if row["source_iteration_id"] <= previous_source_iteration:
            raise ValueError("PD Scheduler source iteration must increase")
        previous_source_iteration = row["source_iteration_id"]
        if row["run_id"] != capture["run_id"]:
            raise ValueError("PD Scheduler run_id disagrees with capture")
        if row["run_origin_monotonic_ns"] != capture["run_origin_monotonic_ns"]:
            raise ValueError("PD Scheduler origin disagrees with capture")
        if row["host_boot_id"] != capture["host_boot_id"]:
            raise ValueError("PD Scheduler host boot ID disagrees with capture")
        if row["instance_role"] != role:
            raise ValueError("PD Scheduler role disagrees with stream")
        _text(row["instance_id"], "PD Scheduler instance_id")
        if row["clock_domain"] != "shared_host_time.perf_counter_ns":
            raise ValueError("PD Scheduler clock domain is invalid")
        for field in (
            "step_start_ns", "schedule_end_ns", "update_start_ns", "step_end_ns",
            "schedule_start_ns", "scheduler_update_start_ns", "scheduler_update_end_ns",
            "device_execution_start_ns", "device_execution_end_ns",
        ):
            _integer(row[field], f"PD Scheduler {field}", minimum=0)
        if not (
            row["step_start_ns"] == row["schedule_start_ns"]
            <= row["schedule_end_ns"]
            <= row["device_execution_start_ns"]
            <= row["device_execution_end_ns"]
            <= row["update_start_ns"] == row["scheduler_update_start_ns"]
            <= row["step_end_ns"] == row["scheduler_update_end_ns"]
        ):
            raise ValueError("PD Scheduler timing boundaries are not causal")
        if row["device_execution_clock"] != "time.perf_counter_ns":
            raise ValueError("PD Scheduler device clock is invalid")
        if row["device_execution_source"] != "host_monotonic_executor_rpc_envelope":
            raise ValueError("PD Scheduler device source is invalid")
        _integer(row["in_flight_step_count"], "PD Scheduler in-flight count", minimum=1)
        if row["in_flight_step_count"] > expected_depth:
            raise ValueError("PD Scheduler in-flight count exceeds configured depth")
        if row["scheduler_pipeline_depth_evidence"] != expected_depth_evidence:
            raise ValueError("PD Scheduler depth evidence disagrees with runtime")
        scheduled = _sequence(row["scheduled_requests"], "PD Scheduler scheduled requests")
        seen = set()
        prefill_count = 0
        decode_count = 0
        prefill_tokens = 0
        decode_tokens = 0
        for request in scheduled:
            work = _mapping(request, "PD Scheduler scheduled request")
            _exact_keys(work, ("request_idx", "prefill_tokens", "decode_tokens", "scheduled_tokens"), "PD Scheduler scheduled request")
            request_idx = work["request_idx"]
            _integer(request_idx, "PD Scheduler request_idx", minimum=0)
            if request_idx >= request_count or request_idx in seen:
                raise ValueError("PD Scheduler scheduled request index is invalid")
            seen.add(request_idx)
            for field in ("prefill_tokens", "decode_tokens"):
                _integer(work[field], f"PD Scheduler request {field}", minimum=0)
            _integer(work["scheduled_tokens"], "PD Scheduler scheduled_tokens", minimum=1)
            if work["scheduled_tokens"] != work["prefill_tokens"] + work["decode_tokens"]:
                raise ValueError("PD Scheduler token split is inconsistent")
            total_prefill[request_idx] += work["prefill_tokens"]
            total_decode[request_idx] += work["decode_tokens"]
            if work["prefill_tokens"]:
                prefill_count += 1
                prefill_tokens += work["prefill_tokens"]
            if work["decode_tokens"]:
                decode_count += 1
                decode_tokens += work["decode_tokens"]
        derived = {
            "scheduled_request_count": len(scheduled),
            "prefill_request_count": prefill_count,
            "decode_request_count": decode_count,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "scheduled_tokens": prefill_tokens + decode_tokens,
        }
        for field, expected in derived.items():
            if row[field] != expected:
                raise ValueError(f"PD Scheduler {field} is inconsistent")
        cumulative_tokens += derived["scheduled_tokens"]
        if row["cumulative_scheduled_tokens"] != cumulative_tokens:
            raise ValueError("PD Scheduler cumulative token count is inconsistent")
        completed = _sequence(row["finished_request_indexes"], "PD Scheduler finished requests")
        for request_idx in completed:
            _integer(request_idx, "PD Scheduler finished request_idx", minimum=0)
            if request_idx >= request_count or request_idx in finished:
                raise ValueError("PD Scheduler finished request index is invalid")
            finished.add(request_idx)
        cumulative_finished += len(completed)
        if row["cumulative_completed_requests"] != cumulative_finished:
            raise ValueError("PD Scheduler cumulative completion count is inconsistent")
        _validate_graph_row(
            row,
            hardware_id,
            configured_graph_mode=configured_graph_mode,
            capture_limit_tokens=capture_limit_tokens,
        )
    if finished != set(range(request_count)):
        raise ValueError("PD Scheduler must finish each request exactly once")
    if role == "prefill" and tuple(total_prefill) != tuple(request_input_tokens):
        raise ValueError("PD Prefill Scheduler prompt-token conservation failed")
    if role == "decode":
        if any(value != 1 for value in total_prefill):
            raise ValueError("PD Decode Scheduler must consume one handoff token per request")
        include_terminal = _DECODE_SCHEDULES_TERMINAL_OUTPUT_TOKEN[hardware_id]
        expected_decode = tuple(
            value if include_terminal else max(value - 1, 0)
            for value in request_output_tokens
        )
        if tuple(total_decode) != expected_decode:
            raise ValueError("PD Decode Scheduler output-token conservation failed")


def _validate_graph_row(
    row: Mapping[str, object],
    hardware_id: object,
    *,
    configured_graph_mode: object,
    capture_limit_tokens: object,
) -> None:
    if hardware_id == "nvidia_a800":
        observations = _sequence(row["rank_observations"], "PD A800 rank observations")
        if not observations:
            raise ValueError("PD A800 Graph observations cannot be empty")
        for value in observations:
            observation = _mapping(value, "PD A800 rank observation")
            if observation["configured_cudagraph_mode"] != configured_graph_mode:
                raise ValueError("PD A800 configured Graph mode disagrees with runtime")
            if observation["runtime_cudagraph_mode"] not in _GRAPH_MODES[hardware_id]:
                raise ValueError("PD A800 runtime Graph mode is invalid")
        return
    if row["configured_cudagraph_mode"] != configured_graph_mode:
        raise ValueError("PD configured Graph mode disagrees with runtime")
    if row["capture_limit_tokens"] != capture_limit_tokens:
        raise ValueError("PD Graph capture limit disagrees with runtime")
    if row["runtime_cudagraph_mode"] not in _GRAPH_MODES[hardware_id]:
        raise ValueError("PD runtime Graph mode is invalid")
    for field in ("num_unpadded_tokens", "capture_limit_tokens", "padded_total_tokens", "padding_tokens"):
        _integer(row[field], f"PD Graph {field}", minimum=0)
    if row["num_unpadded_tokens"] != row["scheduled_tokens"]:
        raise ValueError("PD Graph unpadded tokens are inconsistent")
    if row["padded_total_tokens"] != row["num_unpadded_tokens"] + row["padding_tokens"]:
        raise ValueError("PD Graph padding is inconsistent")
    mode = row["runtime_cudagraph_mode"]
    if mode == "NONE":
        if row["capture_bucket_tokens"] is not None or row["graph_fallback_reason"] is None:
            raise ValueError("PD Graph NONE mode is inconsistent")
    elif mode in _GRAPH_MODES[hardware_id]:
        _integer(row["capture_bucket_tokens"], "PD Graph capture bucket", minimum=1)
        if row["capture_bucket_tokens"] != row["padded_total_tokens"]:
            raise ValueError("PD Graph capture bucket is inconsistent")
        if row["graph_fallback_reason"] is not None:
            raise ValueError("PD Graph execution cannot have a fallback reason")
    else:
        raise ValueError("PD runtime Graph mode is invalid")


def _load_handoffs(
    root: Path,
    entry_value: object,
    *,
    requests: Sequence[CanonicalRequestRow],
    prefill_scheduler_rows: Sequence[Mapping[str, object]],
    capture: Mapping[str, object],
    raw_source_sha_by_id: Mapping[object, object],
) -> tuple[Mapping[str, object], ...]:
    entry = _mapping(entry_value, "PD handoff entry")
    provenance = _mapping(entry["provenance"], "PD handoff provenance")
    _validate_raw_source_binding(
        provenance["router_source_archive_id"],
        provenance["router_source_sha256"],
        raw_source_sha_by_id,
        "PD router provenance",
    )
    connector_ids = _string_sequence(
        provenance["connector_source_archive_ids"],
        "PD connector source archives",
    )
    connector_shas = _sequence(
        provenance["connector_source_sha256s"],
        "PD connector source SHAs",
    )
    for source_id, source_sha in zip(connector_ids, connector_shas, strict=True):
        _validate_raw_source_binding(
            source_id,
            source_sha,
            raw_source_sha_by_id,
            "PD connector provenance",
        )
    path = _case_file(root, entry["path"])
    _verify_sha(path, entry["sha256"], "PD handoffs")
    rows = _read_jsonl(path, "PD handoffs")
    if entry["row_count"] != len(rows) or len(rows) != len(requests):
        raise ValueError("PD handoff row count is inconsistent")
    seen = set()
    for value in rows:
        row = _mapping(value, "PD handoff row")
        _exact_keys(
            row,
            (
                "schema_version", "request_idx", "external_request_id",
                "producer_request_id", "consumer_request_id", "handoff_transport",
                "router_received_ns", "producer_control_response_ns",
                "consumer_request_sent_ns", "consumer_first_visible_token_ns",
                "consumer_stream_completed_ns", "producer_ready_ns", "transfer_start_ns",
                "transfer_end_ns", "consumer_load_start_ns", "consumer_load_end_ns",
                "consumer_first_schedule_ns", "producer_kv_tokens", "producer_kv_blocks",
                "transfer_bytes", "run_id", "run_origin_monotonic_ns", "host_boot_id",
                "status",
            ),
            "PD handoff row",
        )
        if row["schema_version"] != PD_DISAGGREGATED_SCHEMA_VERSION:
            raise ValueError("PD handoff schema version is invalid")
        request_idx = row["request_idx"]
        _integer(request_idx, "PD handoff request_idx", minimum=0)
        if request_idx >= len(requests) or request_idx in seen:
            raise ValueError("PD handoff request index is invalid")
        seen.add(request_idx)
        for field in ("external_request_id", "producer_request_id", "consumer_request_id"):
            _text(row[field], f"PD handoff {field}")
        if row["handoff_transport"] not in (
            "p2p_request_id", "openai_kv_transfer_params"
        ):
            raise ValueError("PD handoff transport is invalid")
        for field in (
            "router_received_ns", "producer_control_response_ns", "consumer_request_sent_ns",
            "consumer_first_visible_token_ns", "consumer_stream_completed_ns", "producer_ready_ns",
            "transfer_start_ns", "transfer_end_ns", "consumer_load_start_ns",
            "consumer_load_end_ns", "consumer_first_schedule_ns",
        ):
            _integer(row[field], f"PD handoff {field}", minimum=0)
        if not (
            row["router_received_ns"] <= row["producer_control_response_ns"]
            <= row["consumer_request_sent_ns"] <= row["consumer_first_visible_token_ns"]
            <= row["consumer_stream_completed_ns"]
        ):
            raise ValueError("PD client handoff boundaries are not causal")
        if not (
            row["producer_ready_ns"] <= row["transfer_start_ns"]
            <= row["transfer_end_ns"]
        ):
            raise ValueError("PD producer/transfer boundaries are not causal")
        if row["consumer_load_start_ns"] > row["consumer_load_end_ns"]:
            raise ValueError("PD consumer load boundaries are not causal")
        if row["consumer_request_sent_ns"] > row["consumer_first_schedule_ns"]:
            raise ValueError("PD consumer scheduling predates its request")
        if row["consumer_load_end_ns"] > row["consumer_first_visible_token_ns"]:
            raise ValueError("PD client observed a token before external KV load")
        for field in ("producer_kv_tokens", "producer_kv_blocks"):
            _integer(row[field], f"PD handoff {field}", minimum=1)
        _validate_transfer_bytes(row["transfer_bytes"])
        if row["run_id"] != capture["run_id"]:
            raise ValueError("PD handoff run_id disagrees with capture")
        if row["run_origin_monotonic_ns"] != capture["run_origin_monotonic_ns"]:
            raise ValueError("PD handoff origin disagrees with capture")
        if row["host_boot_id"] != capture["host_boot_id"]:
            raise ValueError("PD handoff host boot ID disagrees with capture")
        if row["status"] != "completed":
            raise ValueError("PD handoff status must be completed")
    if seen != set(range(len(requests))):
        raise ValueError("PD handoffs must cover every request exactly once")
    _validate_handoff_kv_tokens(rows, prefill_scheduler_rows, requests)
    return rows


def _validate_transfer_bytes(value: object) -> None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return
    evidence = _mapping(value, "PD transfer byte evidence")
    _exact_keys(evidence, ("status", "reason"), "PD transfer byte evidence")
    if evidence["status"] != "unobserved_by_connector_runtime":
        raise ValueError("PD transfer byte evidence status is invalid")
    _text(evidence["reason"], "PD transfer byte evidence reason")


def _validate_handoff_kv_tokens(
    handoffs: Sequence[Mapping[str, object]],
    prefill_scheduler_rows: Sequence[Mapping[str, object]],
    requests: Sequence[CanonicalRequestRow],
) -> None:
    prefill_tokens = [0 for _ in requests]
    for row in prefill_scheduler_rows:
        for value in _sequence(
            row["scheduled_requests"], "PD Prefill scheduled requests"
        ):
            work = _mapping(value, "PD Prefill scheduled request")
            request_idx = work["request_idx"]
            prefill_tokens[request_idx] += work["prefill_tokens"]
    for handoff in handoffs:
        request_idx = handoff["request_idx"]
        expected_tokens = requests[request_idx].input_tokens
        if prefill_tokens[request_idx] != expected_tokens:
            raise ValueError("PD Prefill Scheduler KV tokens disagree with requests.csv")
        if handoff["producer_kv_tokens"] != expected_tokens:
            raise ValueError("PD connector KV tokens disagree with requests.csv")


def _raw_source_sha_by_id(raw_sources_value: object) -> Mapping[object, object]:
    sources = _sequence(raw_sources_value, "PD raw sources")
    result: dict[object, object] = {}
    for value in sources:
        source = _mapping(value, "PD raw source")
        archive_id = source["archive_id"]
        if archive_id in result:
            raise ValueError("PD raw sources repeat an archive ID")
        result[archive_id] = source["sha256"]
    return result


def _validate_raw_source_binding(
    archive_id: object,
    source_sha: object,
    raw_source_sha_by_id: Mapping[object, object],
    label: str,
) -> None:
    if archive_id not in raw_source_sha_by_id:
        raise ValueError(f"{label} references an unknown raw source")
    if raw_source_sha_by_id[archive_id] != source_sha:
        raise ValueError(f"{label} SHA disagrees with raw source")


def _validate_depth_evidence(
    evidence: Mapping[str, object], *, expected_depth: object, expected_pp: object
) -> None:
    _exact_keys(
        evidence,
        (
            "source_kind", "source_field", "configured_depth", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity",
        ),
        "PD depth evidence",
    )
    source_pair = (evidence["source_kind"], evidence["source_field"])
    if source_pair not in (
        ("runtime_executor_property", "max_concurrent_batches"),
        ("runtime_vllm_config_field", "vllm_config.max_concurrent_batches"),
    ):
        raise ValueError("PD depth evidence source kind/field is invalid")
    if evidence["configured_depth"] != expected_depth:
        raise ValueError("PD depth evidence configured depth is inconsistent")
    if evidence["pipeline_parallel_size"] != expected_pp:
        raise ValueError("PD depth evidence pipeline parallel size is inconsistent")
    _text(evidence["executor_class"], "PD depth evidence executor_class")
    _text(evidence["runtime_version"], "PD depth evidence runtime_version")
    source_identity = _mapping(evidence["source_identity"], "PD depth source identity")
    _exact_keys(source_identity, ("module_file", "module_name"), "PD depth source identity")
    _text(source_identity["module_file"], "PD depth module_file")
    _text(source_identity["module_name"], "PD depth module_name")


def _validate_request_provenance(
    entry: Mapping[str, object], raw_sources_value: object
) -> None:
    raw_ids = {
        _mapping(value, "PD raw source")["archive_id"]
        for value in _sequence(raw_sources_value, "PD raw sources")
    }
    provenance = _mapping(entry["field_provenance"], "PD request provenance")
    for field in CANONICAL_REQUEST_COLUMNS:
        value = _mapping(provenance[field], f"PD request provenance {field}")
        if value["source_archive_id"] is not None and value["source_archive_id"] not in raw_ids:
            raise ValueError("PD request provenance references an unknown raw source")
    cached_evidence = entry["cached_hit_value_evidence"]
    cached_provenance = _mapping(
        provenance["Cached hit [tokens]"], "PD cached-hit provenance"
    )
    if cached_provenance["kind"] == "observed_actual":
        if cached_evidence is not None:
            raise ValueError("PD observed cached hit must not add representative evidence")
        return
    evidence = _mapping(cached_evidence, "PD cached-hit value evidence")
    _exact_keys(
        evidence,
        (
            "kind", "logical_prefix_hit_observability", "representative_value",
            "net_prefix_work_reduction_tokens", "derivation", "source_archive_ids",
            "raw_source_set_sha256",
        ),
        "PD cached-hit value evidence",
    )
    if evidence["representative_value"] != 0 or evidence["net_prefix_work_reduction_tokens"] != 0:
        raise ValueError("PD cached-hit representative must encode zero")
    source_ids = _string_sequence(
        evidence["source_archive_ids"], "PD cached-hit source archives"
    )
    if not source_ids or not set(source_ids).issubset(raw_ids):
        raise ValueError("PD cached-hit evidence references an unknown raw source")
    expected_raw_sha = raw_source_set_sha256(tuple(
        _mapping(value, "PD raw source")
        for value in _sequence(raw_sources_value, "PD raw sources")
    ))
    if evidence["raw_source_set_sha256"] != expected_raw_sha:
        raise ValueError("PD cached-hit evidence raw source set is inconsistent")


def _validate_file_set(root: Path) -> None:
    expected = {
        Path("manifest.json"), Path("requests.csv"), Path("metrics.json"),
        Path("scheduler_steps_prefill.jsonl.zst"),
        Path("scheduler_steps_decode.jsonl.zst"),
        Path("pd_handoffs.jsonl"),
    }
    actual = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(
            f"PD compact case file set mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _write_metrics(root: Path, availability: AvailabilityPayload, request_sha: str) -> Mapping[str, object]:
    metrics = _mapping(availability.payload, "PD available metrics")
    if "schema_version" in metrics or "requests_sha256" in metrics:
        raise ValueError("PD metrics payload must not own schema fields")
    content = _json_bytes({
        "schema_version": PD_DISAGGREGATED_SCHEMA_VERSION,
        "requests_sha256": request_sha,
        **dict(metrics),
    })
    (root / "metrics.json").write_bytes(content)
    return {"status": "available", "reason": None, "path": "metrics.json", "sha256": hashlib.sha256(content).hexdigest()}


def _read_jsonl(path: Path, label: str) -> tuple[Mapping[str, object], ...]:
    return _read_jsonl_bytes(path.read_bytes(), label)


def _read_jsonl_bytes(
    content: bytes, label: str,
) -> tuple[Mapping[str, object], ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    if not text.endswith("\n"):
        raise ValueError(f"{label} must end with one newline")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError(f"{label} cannot contain empty rows")
    rows = tuple(_mapping(_read_json(line, label), label) for line in lines)
    if content != canonical_scheduler_steps_jsonl_bytes(rows):
        raise ValueError(f"{label} is not in canonical JSONL serialization")
    return rows


def _read_object(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"required PD JSON file is absent: {path}")
    return _mapping(_read_json(path.read_text(encoding="utf-8"), str(path)), str(path))


def _read_json(text: str, label: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label} JSON: {error}") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _case_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TypeError("PD compact case path must be non-empty text")
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("PD compact case path must be relative")
    path = root / candidate
    if not path.is_file():
        raise ValueError(f"PD compact case file is absent: {relative}")
    return path


def _verify_sha(path: Path, expected: object, label: str) -> None:
    _sha256(expected, f"{label} SHA")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA is inconsistent")


def _json_bytes(value: object) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError(f"PD compact JSON is not finite: {error}") from error


def _mapping(value: object, label: str) -> Mapping[str, object]:
    return compact_mapping(value, label)


def _sequence(value: object, label: str) -> Sequence[object]:
    return compact_sequence(value, label, expected='a sequence')


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    values = tuple(_sequence(value, label))
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    for item in values:
        _text(item, label)
    return values


def _exact_keys(value: Mapping[str, object], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(f"{label} keys mismatch; missing={sorted(required - actual)}, extra={sorted(actual - required)}")


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be non-empty text")


def _optional_text(value: object, label: str) -> None:
    if value is not None:
        _text(value, label)


def _integer(value: object, label: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer >= {minimum}")


def _finite_positive(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise TypeError(f"{label} must be finite and positive")


def _finite_nonnegative(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise TypeError(f"{label} must be finite and non-negative")


def _sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        raise TypeError(f"{label} must be a lowercase SHA256")


def _sha256_or_commit(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) not in (40, _SHA256_LENGTH)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TypeError(f"{label} must be a lowercase Git commit or SHA256")
