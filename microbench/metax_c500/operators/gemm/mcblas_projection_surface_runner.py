#!/usr/bin/env python3
"""Map the mcBLAS compiled-program surface of the deployed dense decode projections.

mcBLAS is a closed vendor library, so its per-program operand pipeline cannot be
read from source.  The published FusedMoE route measures each deployed compiled
program on non-production geometry and archives the coefficients under that
program's launch signature; the two existing mcBLAS references did that for the
single ``tn_64x32x256_4m1n8k_256t`` island that the Qwen3-30B-A3B TP2 projections
select.  The dense TP1 deployments select on a much wider (K, N) range, and the
vendor's selector is known to move with both K and N, so the island boundaries
have to be mapped before any coefficient may be extracted.

Two modes, deliberately separate invocations:

``identity``
    One eager ``F.linear`` per cell under mcTracer, so the trace carries the
    selected kernel symbol and its launch geometry.  Eager is used here because
    attribution by trace ordinal is unambiguous when each cell issues a fixed,
    known number of launches.  Durations from this mode are *not* consumed --
    mcTracer interposes launches.

``time``
    MACA graph replay over rotating weight buffers.  Graph replay is required
    because an eager launch costs about 6.5 us of host dispatch on this
    deployment while a decode projection runs for tens of microseconds.  The
    buffers rotate because in serving a weight is never L2-resident when it is
    re-read: every other layer's weights stream between two reads of the same
    one.

Nothing here selects a Forward parameter, and no production or evaluation
duration is read.  The deployed shapes are run so that a mechanism extracted on
non-production geometry can afterwards be *scored* against them.
"""

from __future__ import annotations

import argparse
import json
import statistics


def _cells(shapes: str, m_values: str) -> list[tuple[int, int, int]]:
    """Expand the shape list into the exact ordered cell sequence.

    An entry is either ``K,N`` -- expanded over every swept M -- or an explicit
    ``K,N,M`` triple, which lets a sweep target one island without dragging the
    whole M grid through shapes that leave it.
    """
    ms = [int(value) for value in m_values.split(",") if value.strip()]
    out = []
    for entry in shapes.split(";"):
        if not entry.strip():
            continue
        fields = [int(value) for value in entry.split(",")]
        if len(fields) == 3:
            out.append((fields[2], fields[1], fields[0]))
        elif len(fields) == 2:
            for m in ms:
                out.append((m, fields[1], fields[0]))
        else:
            raise ValueError(f"shape entry must be K,N or K,N,M: {entry!r}")
    return out


def _buffers_for(torch, n: int, k: int, requested: int, budget_bytes: int):
    per = n * k * 2
    count = max(2, min(requested, budget_bytes // per))
    return [torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
            for _ in range(count)]


def _time_cell(torch, m, n, k, weights, replays, inner) -> float:
    functional = torch.nn.functional
    activation = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    for weight in weights:
        functional.linear(activation, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in range(inner):
            functional.linear(activation, weights[index % len(weights)])
    torch.cuda.synchronize()
    spans = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        spans.append(start.elapsed_time(end) * 1000.0 / inner)
    del graph, activation
    # The minimum is the cleanest estimate of the uninterrupted replay; the
    # median is reported next to it so that dispersion stays visible.
    return min(spans), statistics.median(spans)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("identity", "time"))
    parser.add_argument("--shapes", required=True,
                        help="semicolon-separated K,N pairs")
    parser.add_argument("--m-values", required=True,
                        help="comma-separated M values swept for every pair")
    parser.add_argument("--buffers", type=int, default=8)
    parser.add_argument("--buffer-budget-gb", type=float, default=12.0)
    parser.add_argument("--replays", type=int, default=9)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--identity-launches", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    import torch
    if "metax" not in torch.__version__.lower():
        raise RuntimeError("probe requires the MetaX torch build")
    torch.cuda.set_device(arguments.device)
    functional = torch.nn.functional
    budget = int(arguments.buffer_budget_gb * (1024 ** 3))

    cells = _cells(arguments.shapes, arguments.m_values)
    rows = []
    if arguments.mode == "identity":
        # One eager launch group per cell, separated by full synchronisation so
        # the trace ordinal of every launch is determined by this loop alone.
        for m, n, k in cells:
            activation = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
            functional.linear(activation, weight)
            torch.cuda.synchronize()
            for _ in range(arguments.identity_launches):
                functional.linear(activation, weight)
                torch.cuda.synchronize()
            rows.append({"m": m, "n": n, "k": k,
                         "launches": arguments.identity_launches})
            print("identity m=%d n=%d k=%d" % (m, n, k), flush=True)
            del activation, weight
            torch.cuda.empty_cache()
    else:
        print("%-6s %-8s %-8s %-4s %-11s %-11s %-10s"
              % ("M", "N", "K", "buf", "min_us", "median_us", "GB/s"), flush=True)
        grouped: dict[tuple[int, int], list] = {}
        for m, n, k in cells:
            grouped.setdefault((n, k), []).append(m)
        for (n, k), ms in grouped.items():
            weights = _buffers_for(torch, n, k, arguments.buffers, budget)
            for m in sorted(ms):
                low, mid = _time_cell(torch, m, n, k, weights,
                                      arguments.replays, arguments.inner)
                weight_bytes = n * k * 2
                rows.append({
                    "m": m, "n": n, "k": k,
                    "buffers": len(weights),
                    "min_us": round(low, 4),
                    "median_us": round(mid, 4),
                    "weight_bytes": weight_bytes,
                    "weight_gb_s": round(weight_bytes / (low * 1e-6) / 1e9, 2),
                })
                print("%-6d %-8d %-8d %-4d %-11.3f %-11.3f %-10.1f"
                      % (m, n, k, len(weights), low, mid,
                         weight_bytes / (low * 1e-6) / 1e9), flush=True)
            del weights
            torch.cuda.empty_cache()

    document = {
        "schema_version": 1,
        "kind": "metax_c500_mcblas_projection_surface",
        "mode": arguments.mode,
        "hardware_id": "metax_c500",
        "torch_version": torch.__version__,
        "device_index": arguments.device,
        "dtype": "bfloat16",
        "timer_domain": (None if arguments.mode == "identity"
                         else "maca_graph_replay_event"),
        "identity_tool": ("mcTracer" if arguments.mode == "identity" else None),
        "production_or_evaluation_duration_consumed": False,
        "inner_per_graph": (None if arguments.mode == "identity"
                            else arguments.inner),
        "replays": (None if arguments.mode == "identity"
                    else arguments.replays),
        "cells": rows,
        "cell_order": [{"k": k, "n": n, "m": m} for m, n, k in cells],
    }
    with open(arguments.output, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    print("wrote %s" % arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
