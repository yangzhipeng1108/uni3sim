"""Shape validators shared by the evaluation report and metrics modules.

The report writers all read their inputs from JSON payloads and all need the
same "this must be an object" check.  It lives in a leaf module -- rather than
in one of the reporting modules -- so any of them can import it without
depending on a sibling's module surface.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def require_object(value: object, label: str) -> Mapping[str, object]:
    """Require a JSON object, the shape every report input is read from."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def file_sha256(path: Path) -> str:
    """Hash one captured artifact; the Scheduler-trace readers all need it."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_integer(value: object, label: str, *, minimum: int = 0) -> int:
    """Require an integer at or above ``minimum``."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def require_finite(value: object, label: str) -> float:
    """Require a finite number (bool is not an integer here)."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def require_object_value(value: object, label: str) -> Mapping[str, object]:
    """Require a JSON object whose payload the callers index as a mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def require_object_dict(value: object, label: str) -> Mapping[str, object]:
    """Require a JSON object whose payload is a concrete ``dict``."""

    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def write_json_file(path: Path, data: dict) -> None:
    """Write one report JSON file, creating its directory if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def require_text(value: object, label: str) -> str:
    """Require non-empty text."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be non-empty text")
    return value


def require_finite_number(value: object, label: str) -> float:
    """Require a finite number, with the "numeric" wording the report uses."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def require_list_or_tuple(value: object, label: str) -> Sequence[object]:
    """Require a JSON array represented as a list or tuple."""

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be a sequence")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    """Write one JSON file via a temporary sibling and an atomic replace."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def require_positive_number(value: object, label: str) -> float:
    """Require a finite number strictly above zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def require_interval_union_size(
    intervals: Sequence[tuple[int, int]], description: str
) -> int:
    """Total length of the union of half-open ``(start, end)`` intervals.

    Overlapping and touching intervals are merged, so an event stream's busy
    time is counted once.  ``description`` names the interval source in the
    error raised for an inverted interval.
    """

    ordered = sorted(intervals)
    if not ordered:
        return 0
    current_start, current_end = ordered[0]
    if current_end < current_start:
        raise ValueError(f"{description} ends before it starts")
    total = 0
    for start_ns, end_ns in ordered[1:]:
        if end_ns < start_ns:
            raise ValueError(f"{description} ends before it starts")
        if start_ns > current_end:
            total += current_end - current_start
            current_start, current_end = start_ns, end_ns
        elif end_ns > current_end:
            current_end = end_ns
    return total + current_end - current_start


def require_positive_metric(mapping: Mapping, name: str, context: str) -> int:
    """Require a positive integer metric named ``context.name``."""

    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{context}.{name} must be an integer, got "
            f"{type(value).__name__}"
        )
    if value <= 0:
        raise ValueError(f"{context}.{name} must be positive, got {value}")
    return value


def percentile_value(values: Sequence[float], percentile: int) -> float:
    """Linear-interpolated percentile over a non-empty sample."""

    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = percentile / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
