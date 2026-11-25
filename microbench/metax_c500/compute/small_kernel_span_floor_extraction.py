#!/usr/bin/env python3
"""Reduce the C500 small-launch span sweep to a span-floor reference.

Reads the mcTracer captures written by ``small_kernel_span_floor.cpp`` over the
(grid, threads, body, pattern) sweep and emits the
``small_kernel_span_floor_reference_v1`` document the HWSpec consumes.

The published value per grid is the smallest span seen for the memory-touching
body across every measured block size, both launch patterns and every launch in
the capture.  Taking the minimum is what composing by maximum requires: the
number has to be a bound that no launch of that grid beats, not a typical
duration.  The empty body is reported alongside but never published, because it
describes a launch that touches nothing and every production kernel touches
memory.

No production operator, kernel or Serving duration participates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

CASE_PATTERN = re.compile(r"^sp_b(?P<body>[01])_p(?P<pattern>[01])"
                          r"_g(?P<grid>\d+)_t(?P<threads>\d+)$")
PATTERN_NAME = {"0": "isolated", "1": "back_to_back"}
BODY_NAME = {"0": "empty_cta", "1": "one_store_per_cta"}
PROBE_KERNELS = ("truly_empty_cta", "one_store_cta")


def _spans(case_dir: Path) -> list[float]:
    values: list[float] = []
    for trace in sorted(case_dir.glob("tracer_out*.json")):
        events = json.loads(trace.read_text())["traceEvents"]
        for event in events:
            name = event.get("name", "")
            if "dur" not in event:
                continue
            if any(name.startswith(probe) for probe in PROBE_KERNELS):
                values.append(event["dur"] / 1000.0)
    return values


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--source", required=True,
                        help="the probe source this capture was built from")
    parser.add_argument("--capture-path", required=True,
                        help="repository-relative path of the raw capture")
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--launches-per-point", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    root = Path(args.sweep_root)
    cells: dict[tuple, dict] = {}
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        match = CASE_PATTERN.match(case_dir.name)
        if match is None:
            continue
        values = _spans(case_dir)
        if not values:
            raise SystemExit(f"no probe kernels in {case_dir}")
        ordered = sorted(values)
        cells[(int(match["grid"]), int(match["threads"]),
               match["body"], match["pattern"])] = {
            "grid_ctas": int(match["grid"]),
            "threads_per_cta": int(match["threads"]),
            "body": BODY_NAME[match["body"]],
            "pattern": PATTERN_NAME[match["pattern"]],
            "launch_count": len(ordered),
            "span_us_min": round(ordered[0], 4),
            "span_us_median": round(statistics.median(ordered), 4),
            "span_us_max": round(ordered[-1], 4),
        }
    if not cells:
        raise SystemExit(f"no sweep cells under {root}")

    grids = sorted({key[0] for key in cells})
    threads = sorted({key[1] for key in cells})
    floor_rows = []
    empty_reference = {}
    for grid in grids:
        store = [cell for key, cell in cells.items()
                 if key[0] == grid and key[2] == "1"]
        empty = [cell for key, cell in cells.items()
                 if key[0] == grid and key[2] == "0"]
        if not store:
            raise SystemExit(f"grid {grid} has no memory-touching cell")
        isolated = [c["span_us_median"] for c in store
                    if c["pattern"] == "isolated"]
        back = [c["span_us_median"] for c in store
                if c["pattern"] == "back_to_back"]
        floor_rows.append({
            "grid_ctas": grid,
            "minimum_span_ns": round(
                min(c["span_us_min"] for c in store) * 1000.0, 1),
            "isolated_median_ns": (
                round(statistics.median(isolated) * 1000.0, 1)
                if isolated else None),
            "back_to_back_median_ns": (
                round(statistics.median(back) * 1000.0, 1)
                if back else None),
            "block_size_spread_ns": round(
                (max(c["span_us_min"] for c in store)
                 - min(c["span_us_min"] for c in store)) * 1000.0, 1),
        })
        if empty:
            empty_reference[str(grid)] = round(
                min(c["span_us_min"] for c in empty) * 1000.0, 1)

    source_path = Path(args.source)
    document = {
        "schema": "small_kernel_span_floor_reference_v1",
        "hardware_id": "metax_c500",
        "mechanism": "small_kernel_span_floor",
        "production_or_evaluation_duration_consumed": False,
        "allowed_use": (
            "A lower bound on the mcTracer-reported duration of a launch that "
            "performs at least one global memory access, given its CTA grid. "
            "It composes with an execution time by maximum and never adds to "
            "one. It does not describe a launch that touches no memory: the "
            "empty-CTA reference below is well under these values, and every "
            "production kernel touches memory."),
        "forbidden_use": (
            "An additive launch cost, a typical or median duration, a host "
            "launch latency, a MACA-event interval, a per-shape correction, or "
            "a floor for a launch with no global memory access."),
        "scope": {
            "timer_domain": "mctracer_uninstrumented_kernel_duration",
            "minimum_grid_ctas": grids[0],
            "maximum_grid_ctas": grids[-1],
            "threads_per_cta_measured": threads,
        },
        "derived_model": {
            "empty_cta_reference_ns": empty_reference,
            "floor_by_grid_ctas": floor_rows,
            "selection_rule": (
                "At each measured grid retain the smallest duration seen for "
                "the memory-touching body across every measured block size, "
                "both launch patterns and every launch in the capture. The "
                "published value is therefore a floor rather than a typical "
                "duration, which is what composing by maximum requires. A grid "
                "below the first measured point takes the first point and one "
                "above the last takes the last; the surface is not "
                "extrapolated."),
        },
        "measured_cells": [cells[key] for key in sorted(cells)],
        "provenance": {
            "source": args.source,
            "source_sha256": hashlib.sha256(
                source_path.read_bytes()).hexdigest(),
            "captures": [{
                "path": args.capture_path,
                "profiler": "mcTracer",
                "device_index": args.device_index,
                "launches_per_point": args.launches_per_point,
            }],
        },
    }
    Path(args.output).write_text(json.dumps(document, indent=4,
                                           sort_keys=True) + "\n")
    print("%-8s %-12s %-14s %-16s %s" % (
        "grid", "floor_ns", "isolated_ns", "back_to_back_ns", "block_spread_ns"))
    for row in floor_rows:
        print("%-8d %-12.1f %-14s %-16s %.1f" % (
            row["grid_ctas"], row["minimum_span_ns"],
            row["isolated_median_ns"], row["back_to_back_median_ns"],
            row["block_size_spread_ns"]))
    print("empty-CTA reference ns:", empty_reference)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
