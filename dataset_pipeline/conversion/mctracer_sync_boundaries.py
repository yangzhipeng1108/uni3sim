"""Derive structural C500 step boundaries from native mcTracer sync events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from dataset_pipeline.conversion.mctracer_chrome import iter_mctracer_trace_events


_PRIMARY_SYNC_NAMES = frozenset(("mcEventSynchronize", "mcDeviceSynchronize"))
_STREAM_SYNC_NAME = "mcStreamSynchronize"


def _timestamp(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _sync_window(event: Mapping[str, object], index: int) -> tuple[str, int, int] | None:
    if "name" not in event:
        return None
    name = event["name"]
    if not isinstance(name, str):
        raise TypeError(f"mcTracer traceEvents[{index}].name must be a string")
    if name not in _PRIMARY_SYNC_NAMES and name != _STREAM_SYNC_NAME:
        return None
    if "ph" not in event:
        raise ValueError(f"mcTracer sync traceEvents[{index}] is missing ph")
    if event["ph"] != "X":
        raise ValueError(f"mcTracer sync traceEvents[{index}] must use ph=X")
    if "ts" not in event or "dur" not in event:
        raise ValueError(
            f"mcTracer sync traceEvents[{index}] is missing ts or dur"
        )
    start = _timestamp(event["ts"], f"mcTracer sync traceEvents[{index}].ts", minimum=0)
    duration = _timestamp(
        event["dur"], f"mcTracer sync traceEvents[{index}].dur", minimum=0
    )
    end = start + duration
    if end <= start:
        raise ValueError(f"mcTracer sync traceEvents[{index}] has a non-positive duration")
    return name, start, end


def collect_sync_windows(trace_path: str | Path) -> tuple[tuple[int, int], ...]:
    """Return ordered primary sync windows, or stream sync windows if primary is absent."""

    primary = []
    stream = []
    for index, event in iter_mctracer_trace_events(trace_path):
        parsed = _sync_window(event, index)
        if parsed is None:
            continue
        name, start, end = parsed
        if name in _PRIMARY_SYNC_NAMES:
            primary.append((start, end))
        else:
            stream.append((start, end))
    windows = primary if primary else stream
    if not windows:
        raise ValueError("mcTracer trace has no supported synchronization events")
    ordered = tuple(sorted(windows))
    previous_end = 0
    for index, (start, end) in enumerate(ordered):
        if start < previous_end:
            raise ValueError(
                f"mcTracer synchronization windows overlap at index {index}"
            )
        previous_end = end
    return ordered


def write_sync_windows(
    trace_path: str | Path,
    output_path: str | Path,
) -> tuple[tuple[int, int], ...]:
    """Write one new, fail-closed structural-boundary sidecar."""

    windows = collect_sync_windows(trace_path)
    path = Path(output_path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(windows, stream, separators=(",", ":"))
        stream.write("\n")
    return windows
