"""Strict successful-request production Trace ingestion and replay.

The reader normalizes external files into the existing :class:`RequestSpec`
contract.  It deliberately ignores unmapped fields: instance placement,
routing outcomes and scheduler results are not production Trace
inputs to the simulator. Token IDs enter only through explicit field mappings.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Tuple

from .request_stream import RequestSpec, RequestStream, request_stream_sha256


_SOURCE_FORMATS = ("csv", "jsonl", "parquet")
_TIME_UNIT_TO_SECONDS = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
}
_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+")
_CANONICAL_FIELDS = (
    "arrival_time",
    "input_tokens",
    "output_tokens",
    "prefix_hit_tokens",
    "request_id",
)
_DEBUG_CSV_SCHEMA_PROFILES = (
    "glm_5_2_debug_csv_v1",
    "kimi_k2_6_debug_csv_v1",
)
_DEBUG_CSV_ARRIVAL_TIME_FIELD = "Query Time Stamp [ms]"
_DEBUG_CSV_INPUT_TOKENS_FIELD = "In Token [tokens]"
_DEBUG_CSV_OUTPUT_TOKENS_FIELD = "Out Token [tokens]"
_DEBUG_CSV_PREFIX_HIT_TOKENS_FIELD = "Cached hit [tokens]"


@dataclass(frozen=True)
class SuccessfulRequestFilter:
    """Proof or exact field rule establishing the successful-only boundary."""

    mode: str
    field_name: str | None
    accepted_value: str | int | bool | None
    provenance: str

    def __post_init__(self) -> None:
        if self.mode not in ("upstream_attested", "field_equals"):
            raise ValueError(
                "success filter mode must be upstream_attested or field_equals")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("success filter provenance must be a non-empty string")
        if self.mode == "upstream_attested":
            if self.field_name is not None or self.accepted_value is not None:
                raise ValueError(
                    "upstream_attested success filter cannot declare a field or value")
        elif self.mode == "field_equals":
            _validate_field_name(self.field_name, "success filter field_name")
            if (self.accepted_value is None
                    or isinstance(self.accepted_value, float)
                    or not isinstance(self.accepted_value, (str, int, bool))):
                raise TypeError(
                    "field_equals accepted_value must be a string, integer, or bool")
        else:
            raise AssertionError(f"unknown success filter mode: {self.mode!r}")


@dataclass(frozen=True)
class ProductionTraceReaderConfig:
    source_format: str
    arrival_time_field: str
    input_tokens_field: str
    output_tokens_field: str
    prefix_hit_tokens_field: str
    request_id_field: str | None
    input_time_unit: str
    successful_request_filter: SuccessfulRequestFilter
    prompt_token_ids_field: str | None = None
    output_token_ids_field: str | None = None

    def __post_init__(self) -> None:
        if self.source_format not in _SOURCE_FORMATS:
            raise ValueError(
                f"source_format must be one of {_SOURCE_FORMATS}, got "
                f"{self.source_format!r}")
        if self.input_time_unit not in _TIME_UNIT_TO_SECONDS:
            raise ValueError(
                "input_time_unit must be one of s, ms, us, or ns")
        if not isinstance(
                self.successful_request_filter, SuccessfulRequestFilter):
            raise TypeError(
                "successful_request_filter must be SuccessfulRequestFilter")
        mapped_fields = (
            self.arrival_time_field,
            self.input_tokens_field,
            self.output_tokens_field,
            self.prefix_hit_tokens_field,
        )
        for canonical_name, field_name in zip(
                _CANONICAL_FIELDS[:4], mapped_fields):
            _validate_field_name(field_name, f"{canonical_name} field")
        if self.request_id_field is not None:
            _validate_field_name(self.request_id_field, "request_id field")
            mapped_fields += (self.request_id_field,)
        for name in ("prompt_token_ids_field", "output_token_ids_field"):
            field_name = getattr(self, name)
            if field_name is not None:
                _validate_field_name(field_name, name)
                mapped_fields += (field_name,)
        if self.output_token_ids_field is not None and self.prompt_token_ids_field is None:
            raise ValueError("output token IDs require a prompt token ID mapping")
        if len(mapped_fields) != len(set(mapped_fields)):
            raise ValueError("production Trace field mappings must be distinct")
        success_filter = self.successful_request_filter
        if (success_filter.mode == "field_equals"
                and success_filter.field_name in mapped_fields):
            raise ValueError(
                "success filter field must be distinct from request fields")


@dataclass(frozen=True)
class ProductionTraceManifest:
    source_path: str
    source_sha256: str
    source_format: str
    input_time_unit: str
    arrival_time_field: str
    input_tokens_field: str
    output_tokens_field: str
    prefix_hit_tokens_field: str
    request_id_field: str | None
    successful_request_filter: SuccessfulRequestFilter
    source_row_count: int
    filtered_out_row_count: int
    request_count: int
    arrival_time_total_s: float
    arrival_time_min_s: float
    arrival_time_max_s: float
    input_tokens_total: int
    output_tokens_total: int
    prefix_hit_tokens_total: int
    logical_miss_tokens_total: int
    normalized_request_stream_sha256: str
    prompt_token_ids_field: str | None = None
    output_token_ids_field: str | None = None


@dataclass(frozen=True)
class ProductionTrace:
    requests: RequestStream
    manifest: ProductionTraceManifest

    def __post_init__(self) -> None:
        if not isinstance(self.requests, RequestStream):
            raise TypeError("ProductionTrace requests must be a RequestStream")
        if not isinstance(self.manifest, ProductionTraceManifest):
            raise TypeError(
                "ProductionTrace manifest must be ProductionTraceManifest")
        if len(self.requests) != self.manifest.request_count:
            raise ValueError(
                "ProductionTrace request count disagrees with its manifest")


@dataclass(frozen=True)
class ProductionTraceReplay:
    requests: RequestStream
    replay_kind: str
    arrival_rate_scale: float | None
    arrival_anchor_s: float
    arrival_time_total_s: float
    arrival_time_min_s: float
    arrival_time_max_s: float
    request_stream_sha256: str

    def __post_init__(self) -> None:
        if self.replay_kind not in ("original_replay", "arrival_rate_scale"):
            raise ValueError(f"unknown replay kind: {self.replay_kind!r}")
        if not isinstance(self.requests, RequestStream):
            raise TypeError("ProductionTraceReplay requests must be RequestStream")


@dataclass(frozen=True)
class _SourceRow:
    source_row_index: int
    values: Mapping[str, object]


def load_production_trace(
    path: str | Path,
    config: ProductionTraceReaderConfig,
) -> ProductionTrace:
    """Load one successful-request Trace, rejecting the whole file on errors."""

    if not isinstance(config, ProductionTraceReaderConfig):
        raise TypeError("config must be ProductionTraceReaderConfig")
    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(
            f"production Trace source is not a file: {source_path}")
    return parse_production_trace_bytes(
        source_path.read_bytes(),
        config,
        source_path=source_path,
    )


def parse_production_trace_bytes(
    source_bytes: bytes,
    config: ProductionTraceReaderConfig,
    *,
    source_path: str | Path,
) -> ProductionTrace:
    """Parse one immutable external Trace byte payload with strict semantics."""

    if not isinstance(source_bytes, bytes):
        raise TypeError("production Trace source payload must be bytes")
    if not isinstance(config, ProductionTraceReaderConfig):
        raise TypeError("config must be ProductionTraceReaderConfig")
    if not isinstance(source_path, (str, Path)) or not str(source_path):
        raise ValueError("production Trace source_path must be non-empty")
    resolved_source_path = Path(source_path).resolve()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    rows = _read_source_rows(source_bytes, config)
    requests = []
    filtered_out = 0
    seen_request_ids = set()
    for row in rows:
        if not _is_successful_row(row, config.successful_request_filter):
            filtered_out += 1
            continue
        request_id = _request_id(row, config.request_id_field)
        if request_id in seen_request_ids:
            raise ValueError(
                f"production Trace source row {row.source_row_index} has a "
                "duplicate request_id")
        seen_request_ids.add(request_id)
        arrival_time_s = _parse_arrival_time(
            _required_value(row, config.arrival_time_field),
            config.input_time_unit,
            row.source_row_index,
        )
        input_tokens = _parse_token_count(
            _required_value(row, config.input_tokens_field),
            "input_tokens",
            row.source_row_index,
        )
        output_tokens = _parse_token_count(
            _required_value(row, config.output_tokens_field),
            "output_tokens",
            row.source_row_index,
        )
        prefix_hit_tokens = _parse_token_count(
            _required_value(row, config.prefix_hit_tokens_field),
            "prefix_hit_tokens",
            row.source_row_index,
        )
        try:
            request = RequestSpec(
                arrival_time_s=arrival_time_s,
                request_id=request_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prefix_hit_tokens=prefix_hit_tokens,
                prompt_token_ids=_read_token_ids(row, config.prompt_token_ids_field),
                output_token_ids=_read_token_ids(row, config.output_token_ids_field),
            )
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"invalid production Trace source row "
                f"{row.source_row_index}: {error}") from error
        requests.append(request)
    if not requests:
        raise ValueError(
            "production Trace contains no successful requests after filtering")
    stream = RequestStream(requests)
    arrival_total, arrival_min, arrival_max = _arrival_statistics(stream)
    manifest = ProductionTraceManifest(
        source_path=str(resolved_source_path),
        source_sha256=source_sha256,
        source_format=config.source_format,
        input_time_unit=config.input_time_unit,
        arrival_time_field=config.arrival_time_field,
        input_tokens_field=config.input_tokens_field,
        output_tokens_field=config.output_tokens_field,
        prefix_hit_tokens_field=config.prefix_hit_tokens_field,
        prompt_token_ids_field=config.prompt_token_ids_field,
        output_token_ids_field=config.output_token_ids_field,
        request_id_field=config.request_id_field,
        successful_request_filter=config.successful_request_filter,
        source_row_count=len(rows),
        filtered_out_row_count=filtered_out,
        request_count=len(stream),
        arrival_time_total_s=arrival_total,
        arrival_time_min_s=arrival_min,
        arrival_time_max_s=arrival_max,
        input_tokens_total=sum(request.input_tokens for request in stream),
        output_tokens_total=sum(request.output_tokens for request in stream),
        prefix_hit_tokens_total=sum(
            request.prefix_hit_tokens for request in stream),
        logical_miss_tokens_total=sum(
            request.input_tokens - request.prefix_hit_tokens
            for request in stream),
        normalized_request_stream_sha256=request_stream_sha256(stream),
    )
    return ProductionTrace(stream, manifest)


def production_trace_manifest(trace: ProductionTrace) -> dict[str, object]:
    """Return the versioned JSON-serializable source provenance manifest."""

    if not isinstance(trace, ProductionTrace):
        raise TypeError("trace must be ProductionTrace")
    manifest = trace.manifest
    success_filter = manifest.successful_request_filter
    if manifest.request_id_field is None:
        request_id_policy = "stable_zero_based_source_row_v1"
    else:
        request_id_policy = "source_field"
    return {
        "schema_version": 1,
        "successful_only": True,
        "source_path": manifest.source_path,
        "source_sha256": manifest.source_sha256,
        "source_format": manifest.source_format,
        "field_mapping": {
            "arrival_time": manifest.arrival_time_field,
            "input_tokens": manifest.input_tokens_field,
            "output_tokens": manifest.output_tokens_field,
            "prefix_hit_tokens": manifest.prefix_hit_tokens_field,
            "request_id": manifest.request_id_field,
            **({"prompt_token_ids": manifest.prompt_token_ids_field,
                "output_token_ids": manifest.output_token_ids_field}
               if manifest.prompt_token_ids_field is not None else {}),
        },
        "input_time_unit": manifest.input_time_unit,
        "normalized_time_unit": "s",
        "request_id_policy": request_id_policy,
        "equal_arrival_order": "source_row_ascending",
        "invalid_row_policy": "reject_trace",
        "unmapped_field_policy": "ignore",
        "prefix_oracle": (
            "native_token_identity_inputs_v1" if manifest.prompt_token_ids_field is not None
            else "trace_observed_request_level_length_unmodified_v1"),
        "scheduler_prefix_alignment_owner": "scheduler_adapter",
        "success_filter": {
            "mode": success_filter.mode,
            "field_name": success_filter.field_name,
            "accepted_value": success_filter.accepted_value,
            "provenance": success_filter.provenance,
        },
        "statistics": {
            "source_row_count": manifest.source_row_count,
            "filtered_out_row_count": manifest.filtered_out_row_count,
            "request_count": manifest.request_count,
            "arrival_time_total_s": manifest.arrival_time_total_s,
            "arrival_time_min_s": manifest.arrival_time_min_s,
            "arrival_time_max_s": manifest.arrival_time_max_s,
            "input_tokens_total": manifest.input_tokens_total,
            "output_tokens_total": manifest.output_tokens_total,
            "prefix_hit_tokens_total": manifest.prefix_hit_tokens_total,
            "logical_miss_tokens_total": manifest.logical_miss_tokens_total,
        },
        "normalized_request_stream_sha256": (
            manifest.normalized_request_stream_sha256),
    }


def production_trace_debug_schema_config(
    schema_profile: str,
    *,
    successful_request_filter: SuccessfulRequestFilter,
) -> ProductionTraceReaderConfig:
    """Return one explicitly selected GLM/Kimi debug-CSV mapping."""

    if schema_profile == "glm_5_2_debug_csv_v1":
        pass
    elif schema_profile == "kimi_k2_6_debug_csv_v1":
        pass
    else:
        raise ValueError(
            "production Trace debug schema_profile must be one of "
            f"{_DEBUG_CSV_SCHEMA_PROFILES}, got {schema_profile!r}")
    return ProductionTraceReaderConfig(
        source_format="csv",
        arrival_time_field=_DEBUG_CSV_ARRIVAL_TIME_FIELD,
        input_tokens_field=_DEBUG_CSV_INPUT_TOKENS_FIELD,
        output_tokens_field=_DEBUG_CSV_OUTPUT_TOKENS_FIELD,
        prefix_hit_tokens_field=_DEBUG_CSV_PREFIX_HIT_TOKENS_FIELD,
        request_id_field=None,
        input_time_unit="ms",
        successful_request_filter=successful_request_filter,
    )


def build_production_trace_replay(
    trace: ProductionTrace,
    *,
    arrival_rate_scale: float | None,
) -> ProductionTraceReplay:
    """Build an original replay or one explicit arrival-rate scale point."""

    if not isinstance(trace, ProductionTrace):
        raise TypeError("trace must be ProductionTrace")
    original = trace.requests
    anchor = original.requests[0].arrival_time_s
    if arrival_rate_scale is None:
        requests = original
        replay_kind = "original_replay"
    else:
        if (isinstance(arrival_rate_scale, bool)
                or not isinstance(arrival_rate_scale, (int, float))
                or not math.isfinite(arrival_rate_scale)):
            raise TypeError("arrival_rate_scale must be a finite number")
        if arrival_rate_scale <= 0:
            raise ValueError("arrival_rate_scale must be positive")
        scale = float(arrival_rate_scale)
        transformed = []
        for request in original:
            arrival_time_s = (
                anchor + (request.arrival_time_s - anchor) / scale)
            transformed.append(RequestSpec(
                arrival_time_s=arrival_time_s,
                request_id=request.request_id,
                input_tokens=request.input_tokens,
                output_tokens=request.output_tokens,
                prefix_hit_tokens=request.prefix_hit_tokens,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=request.output_token_ids,
            ))
        requests = RequestStream(transformed)
        replay_kind = "arrival_rate_scale"
        arrival_rate_scale = scale
    _validate_replay_conservation(original, requests)
    arrival_total, arrival_min, arrival_max = _arrival_statistics(requests)
    return ProductionTraceReplay(
        requests=requests,
        replay_kind=replay_kind,
        arrival_rate_scale=arrival_rate_scale,
        arrival_anchor_s=anchor,
        arrival_time_total_s=arrival_total,
        arrival_time_min_s=arrival_min,
        arrival_time_max_s=arrival_max,
        request_stream_sha256=request_stream_sha256(requests),
    )


def select_production_trace_window(
    trace: ProductionTrace,
    *,
    request_start: int | None = None,
    request_end: int | None = None,
    arrival_start_s: float | None = None,
    arrival_end_s: float | None = None,
) -> RequestStream:
    """Select an explicit half-open request and/or arrival window without rewriting rows."""

    if not isinstance(trace, ProductionTrace):
        raise TypeError("production window requires ProductionTrace")
    if ((request_start is None) != (request_end is None)):
        raise ValueError("production request window requires both start and end")
    if ((arrival_start_s is None) != (arrival_end_s is None)):
        raise ValueError("production arrival window requires both start and end")
    if request_start is None and arrival_start_s is None:
        raise ValueError("production window requires an explicit range")
    if request_start is not None:
        if (type(request_start) is not int or type(request_end) is not int
                or not 0 <= request_start < request_end <= len(trace.requests)):
            raise ValueError("production request window must be a valid half-open range")
    else:
        request_start, request_end = 0, len(trace.requests)
    if arrival_start_s is not None:
        if (type(arrival_start_s) not in (int, float)
                or type(arrival_end_s) not in (int, float)
                or not math.isfinite(arrival_start_s)
                or not math.isfinite(arrival_end_s)
                or not 0 <= arrival_start_s < arrival_end_s):
            raise ValueError("production arrival window must be finite, nonnegative and half-open")
    selected = trace.requests.requests[request_start:request_end]
    if arrival_start_s is not None:
        selected = tuple(request for request in selected
            if arrival_start_s <= request.arrival_time_s < arrival_end_s)
    if not selected:
        raise ValueError("production window selected no requests")
    return RequestStream(selected)


def build_production_trace_arrival_rate_sweep(
    trace: ProductionTrace,
    arrival_rate_scales: Iterable[float],
) -> Tuple[ProductionTraceReplay, ...]:
    """Build deterministic replay points without resampling request fields."""

    scales = tuple(arrival_rate_scales)
    if not scales:
        raise ValueError("arrival-rate sweep requires at least one scale")
    replays = tuple(
        build_production_trace_replay(trace, arrival_rate_scale=scale)
        for scale in scales
    )
    normalized_scales = tuple(
        replay.arrival_rate_scale for replay in replays)
    if len(normalized_scales) != len(set(normalized_scales)):
        raise ValueError("arrival-rate sweep scales must be unique")
    return replays


def production_trace_replay_manifest(
    replay: ProductionTraceReplay,
) -> dict[str, object]:
    if not isinstance(replay, ProductionTraceReplay):
        raise TypeError("replay must be ProductionTraceReplay")
    if replay.replay_kind == "original_replay":
        arrival_transform = "identity"
    elif replay.replay_kind == "arrival_rate_scale":
        arrival_transform = (
            "t_anchor + (t - t_anchor) / arrival_rate_scale")
    else:
        raise ValueError(f"unknown replay kind: {replay.replay_kind!r}")
    return {
        "schema_version": 1,
        "kind": replay.replay_kind,
        "arrival_rate_scale": replay.arrival_rate_scale,
        "arrival_transform": arrival_transform,
        "arrival_anchor_s": replay.arrival_anchor_s,
        "request_count": len(replay.requests),
        "arrival_time_total_s": replay.arrival_time_total_s,
        "arrival_time_min_s": replay.arrival_time_min_s,
        "arrival_time_max_s": replay.arrival_time_max_s,
        "request_distribution_policy": "preserve_ids_and_token_lengths",
        "interpretation": "successful_path_performance_only",
        "request_stream_sha256": replay.request_stream_sha256,
    }


def _validate_field_name(value: object, description: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")


def _required_field_names(
    config: ProductionTraceReaderConfig,
) -> Tuple[str, ...]:
    fields = [
        config.arrival_time_field,
        config.input_tokens_field,
        config.output_tokens_field,
        config.prefix_hit_tokens_field,
    ]
    if config.request_id_field is not None:
        fields.append(config.request_id_field)
    for field_name in (config.prompt_token_ids_field, config.output_token_ids_field):
        if field_name is not None:
            fields.append(field_name)
    success_filter = config.successful_request_filter
    if success_filter.mode == "field_equals":
        if success_filter.field_name is None:
            raise AssertionError("validated field_equals filter lacks a field")
        fields.append(success_filter.field_name)
    elif success_filter.mode == "upstream_attested":
        pass
    else:
        raise AssertionError(f"unknown success filter mode: {success_filter.mode!r}")
    return tuple(fields)


def _read_source_rows(
    source_bytes: bytes,
    config: ProductionTraceReaderConfig,
) -> Tuple[_SourceRow, ...]:
    if config.source_format == "csv":
        return _read_csv_rows(source_bytes, config)
    elif config.source_format == "jsonl":
        return _read_jsonl_rows(source_bytes)
    elif config.source_format == "parquet":
        return _read_parquet_rows(source_bytes, config)
    else:
        raise AssertionError(
            f"unknown source format after validation: {config.source_format!r}")


def _decode_text(source_bytes: bytes, source_format: str) -> str:
    try:
        return source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{source_format} production Trace must be UTF-8") from error


def _read_csv_rows(
    source_bytes: bytes,
    config: ProductionTraceReaderConfig,
) -> Tuple[_SourceRow, ...]:
    reader = csv.reader(
        io.StringIO(_decode_text(source_bytes, "CSV")),
        strict=True,
    )
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("CSV production Trace is empty") from None
    except csv.Error as error:
        raise ValueError(
            f"invalid CSV production Trace header: {error}") from error
    if not header or any(not name.strip() for name in header):
        raise ValueError("CSV production Trace header contains an empty field")
    if len(header) != len(set(header)):
        raise ValueError("CSV production Trace header contains duplicate fields")
    missing = [name for name in _required_field_names(config) if name not in header]
    if missing:
        raise ValueError(
            f"CSV production Trace is missing mapped fields: {missing}")
    rows = []
    try:
        for source_row_index, values in enumerate(reader):
            if len(values) != len(header):
                raise ValueError(
                    f"CSV production Trace source row {source_row_index} has "
                    f"{len(values)} columns; expected {len(header)}")
            rows.append(_SourceRow(
                source_row_index,
                {name: value for name, value in zip(header, values)},
            ))
    except csv.Error as error:
        raise ValueError(
            f"invalid CSV production Trace near source row "
            f"{len(rows)}: {error}") from error
    return tuple(rows)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON object field {name!r}")
        result[name] = value
    return result


def _read_jsonl_rows(source_bytes: bytes) -> Tuple[_SourceRow, ...]:
    text = _decode_text(source_bytes, "JSONL")
    if not text:
        return ()
    rows = []
    for source_row_index, line in enumerate(text.splitlines()):
        if not line.strip():
            raise ValueError(
                f"JSONL production Trace source row {source_row_index} is blank")
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid JSONL production Trace source row "
                f"{source_row_index}: {error}") from error
        if not isinstance(value, dict):
            raise TypeError(
                f"JSONL production Trace source row {source_row_index} "
                "must be an object")
        rows.append(_SourceRow(source_row_index, value))
    return tuple(rows)


def _read_parquet_rows(
    source_bytes: bytes,
    config: ProductionTraceReaderConfig,
) -> Tuple[_SourceRow, ...]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Parquet production Trace support requires pyarrow") from error
    try:
        parquet_file = pq.ParquetFile(pa.BufferReader(source_bytes))
        column_names = parquet_file.schema_arrow.names
        if len(column_names) != len(set(column_names)):
            raise ValueError(
                "Parquet production Trace contains duplicate column names")
        required = _required_field_names(config)
        missing = [name for name in required if name not in column_names]
        if missing:
            raise ValueError(
                f"Parquet production Trace is missing mapped fields: {missing}")
        table = pq.read_table(
            pa.BufferReader(source_bytes),
            columns=list(required),
        )
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid Parquet production Trace: {error}") from error
    return tuple(
        _SourceRow(source_row_index, values)
        for source_row_index, values in enumerate(table.to_pylist())
    )


def _required_value(row: _SourceRow, field_name: str) -> object:
    if field_name not in row.values:
        raise ValueError(
            f"production Trace source row {row.source_row_index} is missing "
            f"mapped field {field_name!r}")
    value = row.values[field_name]
    if value is None:
        raise ValueError(
            f"production Trace source row {row.source_row_index} has null "
            f"mapped field {field_name!r}")
    return value


def _is_successful_row(
    row: _SourceRow,
    success_filter: SuccessfulRequestFilter,
) -> bool:
    if success_filter.mode == "upstream_attested":
        return True
    elif success_filter.mode == "field_equals":
        if success_filter.field_name is None:
            raise AssertionError("validated field_equals filter lacks a field")
        value = _required_value(row, success_filter.field_name)
        return (type(value) is type(success_filter.accepted_value)
                and value == success_filter.accepted_value)
    else:
        raise AssertionError(f"unknown success filter mode: {success_filter.mode!r}")


def _request_id(row: _SourceRow, field_name: str | None) -> str:
    if field_name is None:
        return f"trace-row-{row.source_row_index:012d}"
    value = _required_value(row, field_name)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(
            f"production Trace source row {row.source_row_index} "
            "request_id must be a non-empty string")
    return value


def _parse_arrival_time(
    value: object,
    input_time_unit: str,
    source_row_index: int,
) -> float:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(
                f"production Trace source row {source_row_index} has an empty "
                "arrival_time")
        try:
            numeric = float(text)
        except ValueError as error:
            raise ValueError(
                f"production Trace source row {source_row_index} arrival_time "
                "is not numeric") from error
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"production Trace source row {source_row_index} arrival_time "
            "must be numeric")
    else:
        numeric = float(value)
    normalized = numeric * _TIME_UNIT_TO_SECONDS[input_time_unit]
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(
            f"production Trace source row {source_row_index} arrival_time "
            "must normalize to a finite non-negative number of seconds")
    return normalized


def _parse_token_count(
    value: object,
    field_name: str,
    source_row_index: int,
) -> int:
    if isinstance(value, str):
        text = value.strip()
        if _INTEGER_TEXT.fullmatch(text) is not None:
            return int(text)
        try:
            decimal_value = Decimal(text)
        except InvalidOperation as error:
            raise TypeError(
                f"production Trace source row {source_row_index} {field_name} "
                "must be an integer-valued decimal") from error
        if (not decimal_value.is_finite()
                or decimal_value != decimal_value.to_integral_value()):
            raise TypeError(
                f"production Trace source row {source_row_index} {field_name} "
                "must be an integer-valued decimal")
        return int(decimal_value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"production Trace source row {source_row_index} {field_name} "
            "must be an integer")
    return value


def _arrival_statistics(
    requests: RequestStream,
) -> tuple[float, float, float]:
    arrivals = tuple(request.arrival_time_s for request in requests)
    total = sum(arrivals)
    if not math.isfinite(total):
        raise ValueError("production Trace arrival-time total is not finite")
    return total, min(arrivals), max(arrivals)


def _validate_replay_conservation(
    original: RequestStream,
    replay: RequestStream,
) -> None:
    original_non_time = tuple(
        (
            request.request_id,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
            request.prompt_token_ids,
            request.output_token_ids,
        )
        for request in original
    )
    replay_non_time = tuple(
        (
            request.request_id,
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
            request.prompt_token_ids,
            request.output_token_ids,
        )
        for request in replay
    )
    if replay_non_time != original_non_time:
        raise RuntimeError(
            "arrival-rate scaling changed request identity or token lengths")


def _read_token_ids(row: _SourceRow, field: str | None) -> tuple[int, ...] | None:
    if field is None:
        return None
    value = _required_value(row, field)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise TypeError(f"token IDs field {field!r} must contain an integer array")
    return tuple(value)
