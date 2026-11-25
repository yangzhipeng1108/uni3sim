"""CPU-only writer for schema-v2 ShareGPT compact candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from .compact_case import (
    ALIGNMENT_FILENAME,
    LATEST_SCHEMA_VERSION,
    OBSERVER_QUALITY_SCHEMA_VERSION,
    RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
    SCHEDULER_CAPTURE_SCHEMA_VERSION,
    SCHEDULER_STEPS_FILENAME,
    PHASES,
    SUPPORTED_SCHEMA_VERSIONS,
    CanonicalRequestRow,
    canonical_requests_csv_bytes,
    canonical_scheduler_steps_jsonl_bytes,
    raw_source_set_sha256,
    validate_compact_case,
)
from .compressed_artifact import (
    COMPRESSION_ALGORITHM,
    write_compressed_canonical_bytes,
)

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SERVING_CANDIDATE_ROOT = _REPOSITORY_ROOT / "profiler_output" / "serving_dataset_compaction"


@dataclass(frozen=True)
class AvailabilityPayload:
    """A present or unavailable file payload with an explicit reason."""

    status: str
    reason: str | None
    payload: object | None

    def __post_init__(self) -> None:
        if self.status not in ("available", "unavailable"):
            raise ValueError("availability status must be available or unavailable")
        if self.status == "available":
            if self.reason is not None or self.payload is None:
                raise ValueError("available payload requires null reason and non-null payload")
        elif (
            self.reason is None
            or not isinstance(self.reason, str)
            or not self.reason.strip()
            or self.payload is not None
        ):
            raise ValueError("unavailable payload requires reason and null payload")


@dataclass(frozen=True)
class CompactCasePayload:
    """Normalized source-independent input accepted by the versioned writer."""

    identity: Mapping[str, object]
    runtime: Mapping[str, object]
    requests: Sequence[CanonicalRequestRow]
    request_field_provenance: Mapping[str, object]
    cached_hit_value_evidence: Mapping[str, object] | None
    metrics: AvailabilityPayload
    alignment_steps: Mapping[str, object]
    raw_sources: Sequence[Mapping[str, object]]
    provenance_gaps: Sequence[str]
    converter: Mapping[str, object]
    schema_version: int = 2
    scheduler_steps: Sequence[Mapping[str, object]] | None = None
    scheduler_step_schema_version: int | None = None
    scheduler_timing_boundary: Mapping[str, object] | None = None
    scheduler_provenance: Mapping[str, object] | None = None
    capture: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Mapping) or not isinstance(self.runtime, Mapping):
            raise TypeError("compact case identity and runtime must be objects")
        if (
            isinstance(self.requests, (str, bytes, bytearray))
            or not isinstance(self.requests, Sequence)
            or not self.requests
        ):
            raise ValueError("compact case requests must be a non-empty sequence")
        if any(not isinstance(row, CanonicalRequestRow) for row in self.requests):
            raise TypeError("compact case requests must contain CanonicalRequestRow")
        if not isinstance(self.request_field_provenance, Mapping):
            raise TypeError("request_field_provenance must be an object")
        if self.cached_hit_value_evidence is not None and not isinstance(
            self.cached_hit_value_evidence, Mapping
        ):
            raise TypeError("cached_hit_value_evidence must be an object or null")
        if not isinstance(self.metrics, AvailabilityPayload):
            raise TypeError("metrics must be AvailabilityPayload")
        if not isinstance(self.alignment_steps, Mapping):
            raise TypeError("alignment_steps must be an object")
        if (
            isinstance(self.raw_sources, (str, bytes, bytearray))
            or not isinstance(self.raw_sources, Sequence)
        ):
            raise TypeError("raw_sources must be a sequence")
        if any(not isinstance(value, Mapping) for value in self.raw_sources):
            raise TypeError("raw_sources must contain objects")
        if (
            isinstance(self.provenance_gaps, (str, bytes, bytearray))
            or not isinstance(self.provenance_gaps, Sequence)
        ):
            raise TypeError("provenance_gaps must be a sequence")
        if not isinstance(self.converter, Mapping):
            raise TypeError("converter must be an object")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("compact case schema_version is unsupported")
        scheduler_values = (
            self.scheduler_steps,
            self.scheduler_timing_boundary,
            self.scheduler_provenance,
            self.capture,
        )
        if self.schema_version == 2:
            if (self.scheduler_step_schema_version is not None
                    or any(value is not None for value in scheduler_values)):
                raise ValueError("schema-v2 payload must not contain scheduler capture")
        elif self.schema_version in (
            SCHEDULER_CAPTURE_SCHEMA_VERSION,
            RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
            LATEST_SCHEMA_VERSION,
            OBSERVER_QUALITY_SCHEMA_VERSION,
        ):
            if any(value is None for value in scheduler_values):
                raise ValueError("scheduler-capture payload requires complete capture")
            if (
                isinstance(self.scheduler_steps, (str, bytes, bytearray))
                or not isinstance(self.scheduler_steps, Sequence)
                or not self.scheduler_steps
            ):
                raise ValueError("scheduler-capture scheduler_steps must be non-empty")
            for field, value in (
                ("scheduler_timing_boundary", self.scheduler_timing_boundary),
                ("scheduler_provenance", self.scheduler_provenance),
                ("capture", self.capture),
            ):
                if not isinstance(value, Mapping):
                    raise TypeError(f"{field} must be an object")
            if self.scheduler_step_schema_version is not None:
                if (isinstance(self.scheduler_step_schema_version, bool)
                        or self.scheduler_step_schema_version not in (
                            SCHEDULER_CAPTURE_SCHEMA_VERSION,
                            RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
                            LATEST_SCHEMA_VERSION,
                        )):
                    raise ValueError(
                        "scheduler_step_schema_version is unsupported")
        else:
            raise AssertionError("compact case schema_version was validated")


def write_compact_case_candidate(
    output_dir: str | Path,
    payload: CompactCasePayload,
    *,
    candidate_root: str | Path | None = None,
) -> Path:
    """Atomically write and validate one candidate below the approved root."""

    if not isinstance(payload, CompactCasePayload):
        raise TypeError("payload must be CompactCasePayload")
    allowed_root = Path(
        _SERVING_CANDIDATE_ROOT if candidate_root is None else candidate_root
    ).resolve()
    if (
        allowed_root.name != "serving_dataset_compaction"
        or allowed_root.parent.name != "profiler_output"
    ):
        raise ValueError(
            "compact candidate root must be exactly "
            "<execution-root>/profiler_output/serving_dataset_compaction"
        )
    if not allowed_root.is_dir():
        raise ValueError(f"compact candidate root is not a directory: {allowed_root}")
    destination = Path(output_dir).resolve()
    if destination == allowed_root or allowed_root not in destination.parents:
        raise ValueError(f"compact candidate destination must be below {allowed_root}")
    identity = dict(payload.identity)
    relative = destination.relative_to(allowed_root)
    campaign_prefix = ()
    if relative.parts and relative.parts[0] == "sharegpt_scheduler_depth_metadata":
        campaign_prefix = ("sharegpt_scheduler_depth_metadata",)
        relative = Path(*relative.parts[1:])
    source_kind = identity["source_kind"]
    if source_kind == "sharegpt":
        expected_parts = (
            identity["hardware_id"], identity["model_id"], identity["case_id"]
        )
    elif source_kind == "production":
        expected_parts = ("production", identity["model_id"], identity["case_id"])
    else:
        raise ValueError(f"unsupported compact candidate source_kind: {source_kind!r}")
    if destination.relative_to(allowed_root).parts != campaign_prefix + expected_parts:
        raise ValueError(
            "candidate destination does not match its source-kind identity"
        )
    if destination.exists():
        raise FileExistsError(f"compact candidate destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"compact candidate parent is not a directory: {destination.parent}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.candidate-", dir=destination.parent)
    )
    try:
        request_bytes = canonical_requests_csv_bytes(payload.requests)
        (temporary / "requests.csv").write_bytes(request_bytes)
        request_sha = hashlib.sha256(request_bytes).hexdigest()
        metrics_entry = _write_metrics(
            temporary, payload.metrics, request_sha, payload.schema_version
        )
        alignment_bytes = _alignment_bytes(
            payload.alignment_steps, payload.schema_version
        )
        compressed_alignment = write_compressed_canonical_bytes(
            temporary / ALIGNMENT_FILENAME, alignment_bytes)
        alignment_sha = hashlib.sha256(alignment_bytes).hexdigest()
        raw_sources = [dict(value) for value in payload.raw_sources]
        manifest = {
            "schema_version": payload.schema_version,
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
                "status": "available",
                "path": ALIGNMENT_FILENAME,
                "sha256": alignment_sha,
                "compressed_sha256": hashlib.sha256(
                    compressed_alignment).hexdigest(),
                "compression": COMPRESSION_ALGORITHM,
                "phases": _phase_manifest(payload.alignment_steps),
            },
            "raw_sources": raw_sources,
            "provenance": {
                "raw_source_set_sha256": raw_source_set_sha256(raw_sources),
                "gaps": list(payload.provenance_gaps),
            },
            "converter": dict(payload.converter),
        }
        if payload.schema_version in (
            SCHEDULER_CAPTURE_SCHEMA_VERSION,
            RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
            LATEST_SCHEMA_VERSION,
            OBSERVER_QUALITY_SCHEMA_VERSION,
        ):
            scheduler_step_schema_version = (
                (LATEST_SCHEMA_VERSION if payload.schema_version ==
                 OBSERVER_QUALITY_SCHEMA_VERSION else payload.schema_version)
                if payload.scheduler_step_schema_version is None
                else payload.scheduler_step_schema_version
            )
            scheduler_rows = [
                {"schema_version": scheduler_step_schema_version, **dict(value)}
                for value in payload.scheduler_steps
            ]
            scheduler_bytes = canonical_scheduler_steps_jsonl_bytes(
                scheduler_rows
            )
            compressed_scheduler = write_compressed_canonical_bytes(
                temporary / SCHEDULER_STEPS_FILENAME, scheduler_bytes)
            manifest["scheduler_steps"] = {
                "path": SCHEDULER_STEPS_FILENAME,
                "sha256": hashlib.sha256(scheduler_bytes).hexdigest(),
                "compressed_sha256": hashlib.sha256(
                    compressed_scheduler).hexdigest(),
                "compression": COMPRESSION_ALGORITHM,
                "row_count": len(scheduler_rows),
                "timing_boundary": dict(payload.scheduler_timing_boundary),
                "provenance": dict(payload.scheduler_provenance),
            }
            manifest["capture"] = dict(payload.capture)
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        validate_compact_case(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _write_metrics(
    root: Path,
    availability: AvailabilityPayload,
    request_sha: str,
    schema_version: int,
) -> Mapping[str, object]:
    if availability.status == "unavailable":
        return {
            "status": "unavailable",
            "reason": availability.reason,
            "path": None,
            "sha256": None,
        }
    metrics = _mapping(availability.payload, "available metrics payload")
    if "schema_version" in metrics or "requests_sha256" in metrics:
        raise ValueError(
            "metrics payload must not declare writer-owned schema_version or requests_sha256"
        )
    bound = {
        "schema_version": schema_version,
        "requests_sha256": request_sha,
        **dict(metrics),
    }
    content = _json_bytes(bound)
    (root / "metrics.json").write_bytes(content)
    return {
        "status": "available",
        "reason": None,
        "path": "metrics.json",
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _alignment_bytes(value: Mapping[str, object], schema_version: int) -> bytes:
    if "schema_version" in value or "kind" in value:
        raise ValueError("alignment_steps payload must not declare writer-owned schema_version or kind")
    return _json_bytes(
        {
            "schema_version": schema_version,
            "kind": "sharegpt_alignment_steps",
            **dict(value),
        }
    )


def _phase_manifest(alignment_steps: Mapping[str, object]) -> Mapping[str, object]:
    phases = _mapping(alignment_steps["phases"], "alignment_steps phases")
    if set(phases) != set(PHASES):
        raise ValueError("alignment_steps phases must exactly equal prefill/decode/mixed")
    output = {}
    for phase in PHASES:
        value = _mapping(phases[phase], f"alignment phase {phase}")
        if "status" not in value or "sample" not in value:
            raise ValueError(f"alignment phase {phase} requires status and sample")
        if value["status"] == "available":
            sample = _mapping(value["sample"], f"alignment phase {phase} sample")
            output[phase] = {"status": "available", "rank_ids": list(sample["rank_ids"])}
        elif value["status"] == "not_observed":
            if value["sample"] is not None:
                raise ValueError(f"not_observed phase {phase} requires null sample")
            output[phase] = {"status": "not_observed", "rank_ids": []}
        else:
            raise ValueError(f"unsupported alignment phase status: {value['status']!r}")
    return output




def _json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(f"compact JSON payload is not finite JSON data: {error}") from error
    return (text + "\n").encode("utf-8")
