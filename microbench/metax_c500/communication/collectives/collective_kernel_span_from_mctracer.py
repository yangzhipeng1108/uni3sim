#!/usr/bin/env python3
"""Reduce an mcTracer capture of the collective sweep to per-size kernel spans.

``collective_protocol_latency_runner.py`` times with MACA events, which on this
deployment also contain the eager launch gap and, for the rank that arrives
first, the wait for its peer.  The C500 profiling contract makes mcTracer the
source of an uninstrumented kernel duration, so the span this reduction reports
is read from the trace rather than from the runner's own event timings.

mcTracer interposes launches, which widens the gap *between* collectives and so
inflates the early rank's spans; it does not change the late rank's span, which
is what the extraction consumes.  Both ranks are reported here so that asymmetry
stays visible rather than being averaged away.

Pairing is by device: each ``tracer_out-<pid>.json`` carries ``args.device_id``
and each ``rank-<n>.json`` carries its physical device, and the sweep's cells are
in the order the runner swept them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _mccl_spans(trace_path: Path, operation: str) -> tuple[list[dict], int]:
    events = json.loads(trace_path.read_text())["traceEvents"]
    # MCCL spells the AllReduce entry point ``mcclKernel_`` and the AllGather
    # one ``mcclkernel_``; match without regard to that case difference.
    marker = ("allreduce_fc2" if operation == "all_reduce"
              else "allgather_fc2")
    selected = [
        event for event in events
        if event.get("name", "").lower().startswith(f"mcclkernel_{marker}")
        and "dur" in event
    ]
    if not selected:
        raise ValueError(f"no FC2 {marker} kernels in {trace_path}")
    device_ids = {event["args"]["device_id"] for event in selected}
    if len(device_ids) != 1:
        raise ValueError(f"trace spans several devices: {trace_path}")
    selected.sort(key=lambda event: event["ts"])
    return selected, device_ids.pop()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--operation", required=True,
                        choices=("all_reduce", "all_gather"))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    sweep_dir = Path(args.sweep_dir)
    rank_documents = {}
    for path in sorted(sweep_dir.glob("rank-*.json")):
        document = json.loads(path.read_text())
        rank_documents[document["device"]["physical_device"]] = document
    if not rank_documents:
        raise SystemExit(f"no rank documents in {sweep_dir}")

    traces = {}
    for path in sorted(sweep_dir.glob("tracer_out-*.json")):
        spans, device_id = _mccl_spans(path, args.operation)
        if device_id in traces:
            raise SystemExit(f"two traces claim device {device_id}")
        traces[device_id] = (path, spans)
    if set(traces) != set(rank_documents):
        raise SystemExit(
            f"trace devices {sorted(traces)} do not match rank devices "
            f"{sorted(rank_documents)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for device_id, document in sorted(rank_documents.items()):
        trace_path, spans = traces[device_id]
        sizes = [cell["message_bytes"] for cell in document["cells"]]
        if len(spans) % len(sizes):
            raise SystemExit(
                f"trace holds {len(spans)} kernels for {len(sizes)} swept "
                "sizes, which is not a whole number of collectives per size")
        per_size = len(spans) // len(sizes)
        cells = []
        for index, message_bytes in enumerate(sizes):
            chunk = spans[index * per_size:(index + 1) * per_size]
            values = sorted(event["dur"] / 1000.0 for event in chunk)
            names = sorted({
                event["name"].split("(")[0].replace("mcclKernel_", "")
                for event in chunk
            })
            cells.append({
                "message_bytes": message_bytes,
                "kernel_count": len(values),
                "vendor_kernels": names,
                "kernel_span_us_min": round(values[0], 4),
                "kernel_span_us_median": round(statistics.median(values), 4),
                "kernel_span_us_max": round(values[-1], 4),
            })
        reduced = {
            "schema_version": 1,
            "kind": "metax_c500_collective_kernel_span_sweep",
            "hardware_id": "metax_c500",
            "communication_library": document["communication_library"],
            "timer_domain": "mctracer_uninstrumented_kernel_duration",
            "measures": "one collective's device kernel span over synthetic "
                        "message sizes; no production workload participates",
            "operation": document["operation"],
            "dtype": document["dtype"],
            "world_size": document["world_size"],
            "rank": document["rank"],
            "run_id": document["run_id"],
            "source_commit": document["source_commit"],
            "device": document["device"],
            "mctracer_trace": trace_path.name,
            "cells": cells,
        }
        target = output_dir / f"span-rank-{document['rank']}.json"
        target.write_text(json.dumps(reduced, indent=2) + "\n")
        written.append(target)
        print(f"rank {document['rank']} device {device_id} "
              f"({per_size} kernels per size)")
        for cell in cells:
            print("   %-9d min %7.2f  median %7.2f  max %8.2f us  %s" % (
                cell["message_bytes"], cell["kernel_span_us_min"],
                cell["kernel_span_us_median"], cell["kernel_span_us_max"],
                ",".join(cell["vendor_kernels"])))
    for target in written:
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
