"""Shared Scheduler pipeline-depth contract.

The depth is a configured control-plane capacity, not a measured maximum and
not a device-parallelism setting.  It limits plan/submit steps that have not
yet completed Scheduler output processing.
"""

from __future__ import annotations

from typing import Mapping


def validate_scheduler_pipeline_depth(value: object, description: str) -> int:
    """Return one explicit configured Scheduler capacity or fail loudly."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be a non-bool integer")
    if value < 1:
        raise ValueError(f"{description} must be positive")
    return value


def validate_scheduler_pipeline_depths(
    values: Mapping[str, object],
    pool_names: set[str],
    description: str,
) -> dict[str, int]:
    """Validate a complete per-pool depth mapping without an implicit value."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{description} must be a mapping")
    supplied = set(values)
    if supplied != pool_names:
        raise ValueError(
            f"{description} keys disagree with scheduler pools: "
            f"missing={sorted(pool_names - supplied)!r}, "
            f"unexpected={sorted(supplied - pool_names)!r}"
        )
    normalized = {}
    for pool_name in sorted(pool_names):
        if not isinstance(pool_name, str) or not pool_name:
            raise ValueError("scheduler pool names must be non-empty strings")
        normalized[pool_name] = validate_scheduler_pipeline_depth(
            values[pool_name], f"{description}[{pool_name!r}]")
    return normalized
