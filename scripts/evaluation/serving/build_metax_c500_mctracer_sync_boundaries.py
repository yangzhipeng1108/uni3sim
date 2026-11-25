#!/usr/bin/env python3
"""Create one C500 native mcTracer structural-boundary sidecar for a case."""

from __future__ import annotations

import argparse
from pathlib import Path

from dataset_pipeline.conversion.mctracer_chrome import iter_mctracer_device_events
from dataset_pipeline.conversion.mctracer_sync_boundaries import write_sync_windows


def _device_trace_paths(trace_dir: Path) -> tuple[Path, ...]:
    candidates = tuple(sorted(
        (*trace_dir.glob("mctracer-*.json"), *trace_dir.glob("tracer_out-*.json"))
    ))
    if not candidates:
        raise FileNotFoundError(f"no native mcTracer trace in {trace_dir}")
    paths = []
    for path in candidates:
        events = iter_mctracer_device_events(path)
        try:
            first = next(events, None)
        finally:
            events.close()
        if first is not None:
            paths.append(path)
    return tuple(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--rank", type=int, required=True)
    arguments = parser.parse_args(argv)
    if arguments.rank < 0:
        raise ValueError("rank must be non-negative")
    trace_dir = arguments.case_dir / "mctracer_out"
    if not trace_dir.is_dir():
        raise FileNotFoundError(f"missing native mcTracer directory: {trace_dir}")
    paths = _device_trace_paths(trace_dir)
    if len(paths) != 1:
        raise ValueError("C500 single-rank boundary builder requires exactly one device trace")
    output = arguments.case_dir / f".mctracer_sync_ends_{arguments.rank}.json"
    windows = write_sync_windows(paths[0], output)
    print(f"wrote {output}: {len(windows)} structural synchronization windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
