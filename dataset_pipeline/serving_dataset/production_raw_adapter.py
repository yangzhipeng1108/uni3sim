"""Build compact Production candidates from already-normalized immutable values."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import math
from pathlib import Path, PurePath
import re
from typing import Sequence

from .compact_candidate_writer import (
    AvailabilityPayload,
    CompactCasePayload,
    write_compact_case_candidate,
)
from .compact_case import (
    CANONICAL_REQUEST_COLUMNS,
    CanonicalRequestRow,
    validate_converter_source_commit,
)
from dataset_pipeline.serving_dataset.json_shape import require_nonempty_string


_MODEL_ID_BY_SCHEMA_PROFILE = {
    "glm_5_2_debug_csv_v1": "glm-5.2",
    "kimi_k2_6_debug_csv_v1": "kimi-k2.6",
}
_RAW_SOURCE_ID = "production-trace"
_CONVERTER_IMPLEMENTATION = "midlayer.production_raw_adapter.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DEBUG_PRODUCTION_CSV_FIELDS = (
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Normal Input Tokens [tokens]",
    "Cached hit [tokens]",
    "Total Token [tokens]",
    "Inference Time [ms]",
    "Ttft [ms]",
    "gen_speed [token/s]",
)
_TTFT_UNAVAILABLE_SENTINEL = "-1.0"
_TTFT_UNAVAILABLE_REASON = "source_unavailable_sentinel_minus_one"
_GENERATION_SPEED_UNAVAILABLE_SENTINEL = ""
_GENERATION_SPEED_UNAVAILABLE_REASON = "source_unavailable_sentinel_empty"


@dataclass(frozen=True)
class _ProductionCompactRecord:
    """One normalized physical source row owned by the raw adapter."""

    request_idx: int
    arrival_time_ms: float
    input_tokens: int
    output_tokens: int
    normal_input_tokens: int
    prefix_hit_tokens: int
    total_tokens: int
    inference_time_ms: float
    ttft_ms: float | None
    ttft_unavailable_reason: str | None
    generation_speed_tokens_per_s: float | None
    generation_speed_unavailable_reason: str | None

    def __post_init__(self) -> None:
        _integer(self.request_idx, "record request_idx", minimum=0)
        CanonicalRequestRow(
            self.arrival_time_ms,
            self.input_tokens,
            self.output_tokens,
            self.prefix_hit_tokens,
        )
        _integer(
            self.normal_input_tokens,
            "record normal_input_tokens",
            minimum=0,
        )
        _integer(self.total_tokens, "record total_tokens", minimum=0)
        if self.normal_input_tokens != self.input_tokens - self.prefix_hit_tokens:
            raise ValueError(
                "record violates normal_input_tokens == input_tokens - "
                "prefix_hit_tokens")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError(
                "record violates total_tokens == input_tokens + output_tokens")
        _finite_non_negative(self.inference_time_ms, "record inference_time_ms")
        if self.ttft_ms is None:
            if self.ttft_unavailable_reason != _TTFT_UNAVAILABLE_REASON:
                raise ValueError(
                    "record with unavailable ttft_ms requires a canonical reason"
                )
        elif self.ttft_unavailable_reason is not None:
            raise ValueError(
                "record with available ttft_ms cannot declare an unavailable reason"
            )
        else:
            _finite_non_negative(self.ttft_ms, "record ttft_ms")
        if self.generation_speed_tokens_per_s is None:
            if (
                self.generation_speed_unavailable_reason
                != _GENERATION_SPEED_UNAVAILABLE_REASON
            ):
                raise ValueError(
                    "record with unavailable generation speed requires a canonical reason"
                )
        elif self.generation_speed_unavailable_reason is not None:
            raise ValueError(
                "record with available generation speed cannot declare an unavailable reason"
            )
        else:
            _finite_non_negative(
                self.generation_speed_tokens_per_s,
                "record generation_speed_tokens_per_s",
            )


@dataclass(frozen=True)
class _ProductionRawSourceIdentity:
    """Identity derived internally from the immutable raw bytes."""

    source_name: str
    byte_count: int
    sha256: str
    upstream_success_provenance: str

    def __post_init__(self) -> None:
        _path_segment(self.source_name, "source_name")
        _integer(self.byte_count, "source byte_count", minimum=1)
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("source sha256 must be lowercase 64-digit hexadecimal")
        _nonempty_string(
            self.upstream_success_provenance,
            "source upstream_success_provenance",
        )


def _parse_production_raw_csv_bytes(
    source_bytes: bytes,
    *,
    schema_profile: str,
    source_name: str,
    upstream_success_provenance: str,
) -> tuple[tuple[_ProductionCompactRecord, ...], _ProductionRawSourceIdentity]:
    """Strictly extract normalized records from one immutable nine-column CSV."""

    if not isinstance(source_bytes, bytes):
        raise TypeError("source_bytes must be bytes")
    if schema_profile not in _MODEL_ID_BY_SCHEMA_PROFILE:
        raise ValueError(f"unsupported Production schema_profile: {schema_profile!r}")
    source = _ProductionRawSourceIdentity(
        source_name,
        len(source_bytes),
        hashlib.sha256(source_bytes).hexdigest(),
        upstream_success_provenance,
    )
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Production CSV must be UTF-8") from error
    reader = csv.reader(io.StringIO(source_text), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Production CSV is empty") from None
    except csv.Error as error:
        raise ValueError("invalid Production CSV header syntax") from error
    if tuple(header) != _DEBUG_PRODUCTION_CSV_FIELDS:
        raise ValueError(
            "Production CSV header does not match the required nine-column "
            f"schema; received {len(header)} columns")
    records = []
    try:
        for request_idx, values in enumerate(reader):
            if len(values) != len(_DEBUG_PRODUCTION_CSV_FIELDS):
                raise ValueError(
                    f"Production CSV row {request_idx} has {len(values)} "
                    f"columns; expected {len(_DEBUG_PRODUCTION_CSV_FIELDS)}")
            input_tokens = _decimal_integer(values[1], "input_tokens", request_idx)
            output_tokens = _decimal_integer(values[2], "output_tokens", request_idx)
            normal_input_tokens = _decimal_integer(
                values[3], "normal_input_tokens", request_idx)
            prefix_hit_tokens = _decimal_integer(
                values[4], "prefix_hit_tokens", request_idx)
            total_tokens = _decimal_integer(values[5], "total_tokens", request_idx)
            ttft_ms, ttft_unavailable_reason = _optional_ttft(
                values[7], request_idx
            )
            generation_speed, generation_speed_unavailable_reason = (
                _optional_generation_speed(values[8], request_idx)
            )
            records.append(
                _ProductionCompactRecord(
                    request_idx=request_idx,
                    arrival_time_ms=_decimal_number(
                        values[0], "arrival_time_ms", request_idx),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    normal_input_tokens=normal_input_tokens,
                    prefix_hit_tokens=prefix_hit_tokens,
                    total_tokens=total_tokens,
                    inference_time_ms=_decimal_number(
                        values[6], "inference_time_ms", request_idx),
                    ttft_ms=ttft_ms,
                    ttft_unavailable_reason=ttft_unavailable_reason,
                    generation_speed_tokens_per_s=generation_speed,
                    generation_speed_unavailable_reason=(
                        generation_speed_unavailable_reason
                    ),
                )
            )
    except csv.Error as error:
        raise ValueError(
            f"invalid Production CSV syntax near row {len(records)}") from error
    if not records:
        raise ValueError("Production CSV contains no request rows")
    return tuple(records), source


def audit_production_raw_csv_bytes(
    source_bytes: bytes,
    *,
    schema_profile: str,
    source_name: str,
    upstream_success_provenance: str,
) -> dict[str, object]:
    """Return aggregate-only evidence from the same strict parser used by build."""

    records, source = _parse_production_raw_csv_bytes(
        source_bytes,
        schema_profile=schema_profile,
        source_name=source_name,
        upstream_success_provenance=upstream_success_provenance,
    )
    arrivals = tuple(record.arrival_time_ms for record in records)
    try:
        arrival_time_total_ms = math.fsum(arrivals)
    except OverflowError as error:
        raise ValueError(
            "Production CSV arrival_time_total_ms must be finite"
        ) from error
    _finite_non_negative(arrival_time_total_ms, "arrival_time_total_ms")
    row_count = len(records)
    unavailable_ttft_count = sum(
        record.ttft_ms is None for record in records
    )
    ttft_reasons = sorted(
        {
            record.ttft_unavailable_reason
            for record in records
            if record.ttft_unavailable_reason is not None
        }
    )
    unavailable_generation_speed_count = sum(
        record.generation_speed_tokens_per_s is None for record in records
    )
    generation_speed_reasons = sorted(
        {
            record.generation_speed_unavailable_reason
            for record in records
            if record.generation_speed_unavailable_reason is not None
        }
    )
    return {
        "schema_version": 1,
        "schema_profile": schema_profile,
        "source_sha256": source.sha256,
        "source_bytes": source.byte_count,
        "schema": _DEBUG_PRODUCTION_CSV_FIELDS,
        "statistics": {
            "request_count": row_count,
            "arrival_time_total_ms": arrival_time_total_ms,
            "arrival_time_min_ms": min(arrivals),
            "arrival_time_max_ms": max(arrivals),
            "input_tokens_total": sum(record.input_tokens for record in records),
            "output_tokens_total": sum(record.output_tokens for record in records),
            "prefix_hit_tokens_total": sum(
                record.prefix_hit_tokens for record in records),
            "logical_miss_tokens_total": sum(
                record.normal_input_tokens for record in records),
        },
        "aggregate_consistency": {
            "logical_miss_identity": {
                "rule": "normal_input_tokens == input_tokens - prefix_hit_tokens",
                "matching_rows": row_count,
                "row_count": row_count,
            },
            "total_token_identity": {
                "rule": "total_tokens == input_tokens + output_tokens",
                "matching_rows": row_count,
                "row_count": row_count,
            },
        },
        "observation_columns": (
            "Inference Time [ms]",
            "Ttft [ms]",
            "gen_speed [token/s]",
        ),
        "observation_coverage": {
            "ttft_ms": {
                "coverage": (
                    "complete" if unavailable_ttft_count == 0 else "partial"
                ),
                "available_request_count": row_count - unavailable_ttft_count,
                "unavailable_request_count": unavailable_ttft_count,
                "unavailable_reasons": ttft_reasons,
            },
            "generation_speed_tokens_per_s": {
                "coverage": (
                    "complete"
                    if unavailable_generation_speed_count == 0
                    else "partial"
                ),
                "available_request_count": (
                    row_count - unavailable_generation_speed_count
                ),
                "unavailable_request_count": unavailable_generation_speed_count,
                "unavailable_reasons": generation_speed_reasons,
            },
        },
        "observation_consumption": "evaluation_only_not_forward",
    }


def build_production_compact_payload(
    source_bytes: bytes,
    *,
    source_name: str,
    model_id: str,
    case_id: str,
    schema_profile: str,
    upstream_success_provenance: str,
    converter_source_commit: str,
) -> CompactCasePayload:
    """Parse immutable raw bytes and build one cryptographically bound payload."""

    validate_converter_source_commit(converter_source_commit)
    _validate_identity(model_id, case_id, schema_profile)
    normalized_records, source = _parse_production_raw_csv_bytes(
        source_bytes,
        schema_profile=schema_profile,
        source_name=source_name,
        upstream_success_provenance=upstream_success_provenance,
    )
    normalized_records = _validate_records(normalized_records)
    requests = tuple(
        CanonicalRequestRow(
            row.arrival_time_ms,
            row.input_tokens,
            row.output_tokens,
            row.prefix_hit_tokens,
        )
        for row in normalized_records
    )
    unavailable_ttft_count = sum(
        row.ttft_ms is None for row in normalized_records
    )
    unavailable_generation_speed_count = sum(
        row.generation_speed_tokens_per_s is None for row in normalized_records
    )
    metrics = {
        "metric_provenance": {
            "source_kind": "production_csv",
            "source_archive_id": _RAW_SOURCE_ID,
            "source_sha256": source.sha256,
            "profiler_attached": False,
        },
        "field_coverage": {
            "inference_time_ms": {
                "coverage": "complete",
                "unavailable_request_count": 0,
                "reason": None,
            },
            "ttft_ms": {
                "coverage": (
                    "complete" if unavailable_ttft_count == 0 else "partial"
                ),
                "unavailable_request_count": unavailable_ttft_count,
                "reason": (
                    None
                    if unavailable_ttft_count == 0
                    else _TTFT_UNAVAILABLE_REASON
                ),
            },
            "generation_speed_tokens_per_s": {
                "coverage": (
                    "complete"
                    if unavailable_generation_speed_count == 0
                    else "partial"
                ),
                "unavailable_request_count": unavailable_generation_speed_count,
                "reason": (
                    None
                    if unavailable_generation_speed_count == 0
                    else _GENERATION_SPEED_UNAVAILABLE_REASON
                ),
            },
        },
        "per_request": [
            {
                "request_idx": row.request_idx,
                "actual_output_tokens": None,
                "ttft_ms": row.ttft_ms,
                "tpot_ms": None,
                "e2e_latency_ms": None,
                "inference_time_ms": row.inference_time_ms,
                "generation_speed_tokens_per_s": (
                    row.generation_speed_tokens_per_s
                ),
            }
            for row in normalized_records
        ],
        "summary": {
            "request_count": len(normalized_records),
            "request_throughput_per_s": None,
            "output_token_throughput_per_s": None,
            "total_token_throughput_per_s": None,
        },
    }
    runtime = {
        "dtype": None,
        "deployment": None,
        "tensor_parallel_size": None,
        "expert_parallel_size": None,
        "pipeline_parallel_size": None,
        "target_qps": None,
        "seed": None,
        "scheduler_config_sha256": None,
        "runtime_version": None,
        "source_commit": None,
        "capture_enable_prefix_caching": None,
    }
    raw_sources = (
        {
            "archive_id": _RAW_SOURCE_ID,
            "role": "production-trace",
            "relative_path": source.source_name,
            "bytes": source.byte_count,
            "sha256": source.sha256,
            "tool": None,
            "tool_version": None,
        },
    )
    request_provenance_reason = (
        "source rows are successful requests under upstream attestation: "
        f"{source.upstream_success_provenance}"
    )
    return CompactCasePayload(
        identity={
            "source_kind": "production",
            "case_id": case_id,
            "hardware_id": None,
            "model_id": model_id,
            "raw_model_id": None,
            "model_revision": None,
            "tokenizer_revision": None,
        },
        runtime=runtime,
        requests=requests,
        request_field_provenance={
            column: {
                "kind": "observed_actual",
                "source_archive_id": _RAW_SOURCE_ID,
                "reason": request_provenance_reason,
            }
            for column in CANONICAL_REQUEST_COLUMNS
        },
        cached_hit_value_evidence=None,
        metrics=AvailabilityPayload("available", None, metrics),
        alignment_steps={
            "operator_signatures": {},
            "launch_signatures": {},
            "phases": {
                phase: {"status": "not_observed", "sample": None}
                for phase in ("prefill", "decode", "mixed")
            },
        },
        raw_sources=raw_sources,
        provenance_gaps=tuple(
            gap["field"] + ": " + gap["reason"]
            for gap in _production_provenance_gaps(runtime)
        ),
        converter={
            "implementation_id": _CONVERTER_IMPLEMENTATION,
            "source_commit": converter_source_commit,
            "selection_rule": "physical_row_order_v1",
        },
    )


def build_production_compact_candidate(
    source_bytes: bytes,
    output_dir: str | Path,
    *,
    source_name: str,
    model_id: str,
    case_id: str,
    schema_profile: str,
    upstream_success_provenance: str,
    converter_source_commit: str,
) -> Path:
    """Build and atomically validate one non-overwriting Production candidate."""

    validate_converter_source_commit(converter_source_commit)
    payload = build_production_compact_payload(
        source_bytes,
        source_name=source_name,
        model_id=model_id,
        case_id=case_id,
        schema_profile=schema_profile,
        upstream_success_provenance=upstream_success_provenance,
        converter_source_commit=converter_source_commit,
    )
    return write_compact_case_candidate(output_dir, payload)


def _validate_records(
    records: Sequence[_ProductionCompactRecord],
) -> tuple[_ProductionCompactRecord, ...]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("records must be a sequence")
    normalized = tuple(records)
    if not normalized:
        raise ValueError("records cannot be empty")
    if any(not isinstance(record, _ProductionCompactRecord) for record in normalized):
        raise TypeError("records must contain normalized Production records")
    for position, record in enumerate(normalized):
        if record.request_idx != position:
            raise ValueError(
                "records must retain contiguous zero-based physical request_idx order"
            )
    return normalized


def _validate_identity(model_id: str, case_id: str, schema_profile: str) -> None:
    if schema_profile not in _MODEL_ID_BY_SCHEMA_PROFILE:
        raise ValueError(f"unsupported Production schema_profile: {schema_profile!r}")
    expected_model_id = _MODEL_ID_BY_SCHEMA_PROFILE[schema_profile]
    if model_id != expected_model_id:
        raise ValueError(
            f"schema_profile {schema_profile!r} requires model_id "
            f"{expected_model_id!r}")
    _path_segment(case_id, "case_id")


def _path_segment(value: object, description: str) -> None:
    _nonempty_string(value, description)
    if PurePath(value).name != value or value in (".", ".."):
        raise ValueError(f"{description} must be one path segment")


def _nonempty_string(value: object, description: str) -> None:
    require_nonempty_string(value, description)


def _integer(value: object, description: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an integer")
    if value < minimum:
        raise ValueError(f"{description} must be at least {minimum}")


def _decimal_integer(value: object, description: str, request_idx: int) -> int:
    parsed = _decimal(value, description, request_idx)
    if parsed != parsed.to_integral_value():
        raise TypeError(
            f"Production CSV row {request_idx} {description} must be an "
            "integer-valued decimal")
    return int(parsed)


def _decimal_number(value: object, description: str, request_idx: int) -> float:
    parsed = _decimal(value, description, request_idx)
    numeric = float(parsed)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(
            f"Production CSV row {request_idx} {description} must be finite "
            "and non-negative")
    return numeric


def _optional_ttft(value: object, request_idx: int) -> tuple[float | None, str | None]:
    """Preserve TTFT, recognizing only the audited source unavailable sentinel."""

    if value == _TTFT_UNAVAILABLE_SENTINEL:
        return None, _TTFT_UNAVAILABLE_REASON
    return _decimal_number(value, "ttft_ms", request_idx), None


def _optional_generation_speed(
    value: object,
    request_idx: int,
) -> tuple[float | None, str | None]:
    """Preserve generation speed, recognizing only the audited empty sentinel."""

    if value == _GENERATION_SPEED_UNAVAILABLE_SENTINEL:
        return None, _GENERATION_SPEED_UNAVAILABLE_REASON
    return _decimal_number(value, "generation_speed_tokens_per_s", request_idx), None


def _decimal(value: object, description: str, request_idx: int) -> Decimal:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"Production CSV row {request_idx} {description} must be numeric")
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as error:
        raise TypeError(
            f"Production CSV row {request_idx} {description} must be numeric") from error
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(
            f"Production CSV row {request_idx} {description} must be finite "
            "and non-negative")
    return parsed


def _finite_non_negative(value: object, description: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{description} must be finite and non-negative")


def _production_provenance_gaps(
    runtime: dict[str, object],
) -> list[dict[str, str]]:
    gaps = [
        {
            "field": "identity.model_revision",
            "reason": "the Production CSV does not record an immutable model revision",
        },
        {
            "field": "identity.tokenizer_revision",
            "reason": "the Production CSV does not record an immutable tokenizer revision",
        },
    ]
    for key in runtime:
        gaps.append(
            {
                "field": f"runtime.{key}",
                "reason": (
                    "the Production CSV does not record this runtime provenance field"
                ),
            }
        )
    gaps.extend(
        (
            {
                "field": f"raw_sources.{_RAW_SOURCE_ID}.tool",
                "reason": "the Production CSV does not identify its producing tool",
            },
            {
                "field": f"raw_sources.{_RAW_SOURCE_ID}.tool_version",
                "reason": "the Production CSV does not identify a producing tool version",
            },
        )
    )
    return gaps
