"""JSON shape validators and content hashes shared by the launch-data modules.

The A800, C500, 910B, and H200 launch-data validators all read untrusted JSON
capture payloads and apply the same closed-schema checks.  They are declared
in this leaf module -- it imports nothing outside the standard library -- so
each hardware module shares one implementation instead of a per-module copy.

Content hashes, in the order a capture flows through them:

- ``sha256_bytes`` digests in-memory bytes: candidate payloads, regenerated
  manifests, and operation keys.
- ``canonical_sequence_sha256`` digests an ordered string sequence by its
  compact JSON encoding; it binds the sorted GEMM workload-key domain.
- ``sorted_json_sha256`` digests an arbitrary JSON payload by its compact,
  key-sorted encoding; the 910B and H200 builders bind provenance snapshots
  and postflight rehashes with it.

Two sibling serializations deliberately stay separate, because each hashes a
different canonical text: ``canonical_projection.expanded_evidence_sha256``
covers the indent=2 canonical JSON exactly as archived under
``profiler_output/``, and
``dataset_pipeline.kernel_dataset.config.canonical_json_sha256`` covers the
event-capture encoding (``ensure_ascii=False``, ``allow_nan=False``).  File
bytes are digested by ``dataset_pipeline.kernel_dataset.config.file_sha256``,
the one streaming (1 MiB block) implementation, and the mapping-plus-key
validator family (``require_positive_int``, ``require_non_empty_string``)
stays in that same module, its original owner.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping, Sequence


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def require_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def require_array(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return value


def require_exact_fields(
        value: Mapping[str, object], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{context} fields differ: expected={sorted(expected)}, "
            f"actual={sorted(value)}")


def require_integer(
        value: object, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be {qualifier}")
    return value


def require_nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def require_sha256_hex(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA256 hex")
    return value


def require_field_finite_number(
        mapping: Mapping[str, object], key: str, context: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{context}.{key} must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ValueError(f"{context}.{key} must be finite, got {value!r}")
    return value


def parse_dim3_csv(value: object, context: str) -> list[int]:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a comma-separated string")
    parts = value.split(",")
    if len(parts) != 3 or any(re.fullmatch(r"[0-9]+", part) is None
                              for part in parts):
        raise ValueError(f"{context} must contain three unsigned integers")
    dimensions = [int(part) for part in parts]
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError(f"{context} dimensions must be positive")
    return dimensions


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sequence_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sorted_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
