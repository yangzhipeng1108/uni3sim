"""Canonical ``requests.csv`` ingestion for Serving workloads."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .production_trace import (
    parse_production_trace_bytes,
    ProductionTraceReaderConfig,
    SuccessfulRequestFilter,
)
from .request_stream import RequestSpec, RequestStream


CANONICAL_REQUEST_COLUMNS = (
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Cached hit [tokens]",
)
CANONICAL_REQUEST_ID_POLICY = "physical_row_request_id_v1"
CANONICAL_EQUAL_ARRIVAL_ORDER = "physical_row_ascending"


def load_canonical_requests(path: str | Path) -> RequestStream:
    """Load one schema-v1 ``requests.csv`` into the common RequestStream.

    The physical zero-based data-row index is the only request identity stored
    by this file contract.  Arrival-time sorting remains stable, so equal-time
    requests retain their physical-file order.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"canonical requests source is not a file: {source_path}")
    return parse_canonical_requests_bytes(
        source_path.read_bytes(),
        source_path=source_path,
    )


def parse_canonical_requests_bytes(
    source_bytes: bytes,
    *,
    source_path: str | Path,
) -> RequestStream:
    """Parse one immutable schema-v1 ``requests.csv`` byte payload."""

    if not isinstance(source_bytes, bytes):
        raise TypeError("canonical requests source payload must be bytes")
    try:
        source_text = source_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("canonical requests CSV must be UTF-8") from error
    reader = csv.reader(io.StringIO(source_text), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("canonical requests CSV is empty") from None
    except csv.Error as error:
        raise ValueError(f"invalid canonical requests CSV header: {error}") from error
    if tuple(header) != CANONICAL_REQUEST_COLUMNS:
        raise ValueError(
            "canonical requests CSV header does not match the required "
            f"four-column schema; received {len(header)} columns"
        )

    trace = parse_production_trace_bytes(
        source_bytes,
        ProductionTraceReaderConfig(
            source_format="csv",
            arrival_time_field=CANONICAL_REQUEST_COLUMNS[0],
            input_tokens_field=CANONICAL_REQUEST_COLUMNS[1],
            output_tokens_field=CANONICAL_REQUEST_COLUMNS[2],
            prefix_hit_tokens_field=CANONICAL_REQUEST_COLUMNS[3],
            request_id_field=None,
            input_time_unit="ms",
            successful_request_filter=SuccessfulRequestFilter(
                mode="upstream_attested",
                field_name=None,
                accepted_value=None,
                provenance=(
                    "canonical requests.csv contains workload inputs only"),
            ),
        ),
        source_path=source_path,
    )
    requests = []
    trace_prefix = "trace-row-"
    for request in trace.requests:
        if not request.request_id.startswith(trace_prefix):
            raise RuntimeError(
                "canonical requests parser received an unexpected generated "
                "request ID")
        row_index_text = request.request_id[len(trace_prefix):]
        if len(row_index_text) != 12 or not row_index_text.isdigit():
            raise RuntimeError(
                "canonical requests parser received an invalid generated "
                "request ID")
        requests.append(RequestSpec(
            arrival_time_s=request.arrival_time_s,
            request_id=f"request-{int(row_index_text):012d}",
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            prefix_hit_tokens=request.prefix_hit_tokens,
        ))
    return RequestStream(requests)
