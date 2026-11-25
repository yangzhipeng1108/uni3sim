"""Half-open integer-interval algebra for the selected-step timeline readers.

Both selected-step device-timeline readers -- the C500 mcTracer correlation
and the 910B msprof correlation -- partition one serialized host-service
window against device-task intervals with the same operators.  They are
stated here once, next to ``json_shape.interval_union_ns`` (the union-length
primitive they compose), so the two readers cannot drift apart on what an
exposed or idle interval means.  Like the readers, these operators work in
whatever nanosecond clock domain the caller's intervals already share.
"""

from __future__ import annotations

from typing import Iterable

from dataset_pipeline.serving_dataset.json_shape import interval_union_ns


def merged_intervals(
    intervals: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sorted union rows of half-open intervals; touching intervals merge."""

    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if end < start:
            raise ValueError("interval ends before it starts")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def clipped_union_ns(
    intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> int:
    """Union length of ``intervals`` restricted to ``[start_ns, end_ns)``."""

    return interval_union_ns(
        (max(start_ns, start), min(end_ns, end))
        for start, end in intervals
        if min(end_ns, end) > max(start_ns, start)
    )


def merged_clipped_intervals(
    intervals: Iterable[tuple[int, int]], start_ns: int, end_ns: int
) -> list[tuple[int, int]]:
    """Union rows of ``intervals`` restricted to ``[start_ns, end_ns)``."""

    return merged_intervals(
        (max(start_ns, start), min(end_ns, end))
        for start, end in intervals
        if min(end_ns, end) > max(start_ns, start)
    )


def subtract_intervals(
    intervals: Iterable[tuple[int, int]],
    removed: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Union rows of ``intervals`` minus the union of ``removed``."""

    remaining = merged_intervals(intervals)
    removed_union = merged_intervals(removed)
    result = []
    for start_ns, end_ns in remaining:
        cursor_ns = start_ns
        for removed_start_ns, removed_end_ns in removed_union:
            if removed_end_ns <= cursor_ns:
                continue
            if removed_start_ns >= end_ns:
                break
            if removed_start_ns > cursor_ns:
                result.append((cursor_ns, min(removed_start_ns, end_ns)))
            cursor_ns = max(cursor_ns, removed_end_ns)
            if cursor_ns >= end_ns:
                break
        if cursor_ns < end_ns:
            result.append((cursor_ns, end_ns))
    return result


def intersection_ns(
    first: Iterable[tuple[int, int]], second: Iterable[tuple[int, int]]
) -> int:
    """Total overlap length of two sets of half-open intervals."""

    left = merged_intervals(first)
    right = merged_intervals(second)
    left_index = 0
    right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        total += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return total
