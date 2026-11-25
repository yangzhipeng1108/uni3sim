"""Shape validators shared by the compact Serving case readers.

The compact schema is validated in several modules -- the case schema itself,
the candidate writer, the raw adapters and the scheduler migrations -- and
each needs the same "this must be a JSON object / array" check.  They are
declared here, in a leaf module with no intra-package imports, so any of them
can import it without creating a cycle through the package ``__init__``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def compact_mapping(value: object, description: str) -> Mapping[str, object]:
    """Require a JSON object."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be an object")
    return value


def compact_sequence(
    value: object, description: str, *, expected: str = "an array"
) -> Sequence[object]:
    """Require a JSON array; a string or bytes is not a sequence here.

    ``expected`` is the whole predicate phrase, article included.  It defaults
    to "an array" (the compact schema's own wording); readers that have always
    said "a sequence" pass that, so their published messages are unchanged.
    """

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{description} must be {expected}")
    return value


def require_nonempty_string(value: object, description: str) -> str:
    """Require a string holding more than whitespace."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a non-empty string")
    return value


def require_exact_keys(
    value: Mapping[str, object], expected: Sequence[str], description: str
) -> None:
    """Require a mapping whose key set is exactly ``expected``."""

    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        raise ValueError(
            f"{description} keys mismatch; "
            f"missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``object_pairs_hook`` that fails closed on a duplicated JSON key."""

    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def file_sha256(path: Path) -> str:
    """Hash one captured artifact; the three selected-step readers share it."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Total covered length of half-open integer intervals, in nanoseconds."""

    ordered = sorted(intervals)
    if not ordered:
        return 0
    start, end = ordered[0]
    total = 0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def layer_index(module_name: str) -> int:
    """Return the layer ordinal encoded in a module path, or -1."""

    parts = module_name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return -1
