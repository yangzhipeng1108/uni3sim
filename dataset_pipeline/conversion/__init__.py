"""Profiler and trace format conversion implementations."""


from __future__ import annotations

from typing import Mapping


def require_object(value: object, context: str) -> Mapping[str, object]:
    """Require a JSON object, the shape each conversion reader is given."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be an object")
    return value
