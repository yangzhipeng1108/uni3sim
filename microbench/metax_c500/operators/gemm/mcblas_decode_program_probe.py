#!/usr/bin/env python3
"""Identify and time the deployed mcBLAS decode GEMM compiled programs.

mcBLAS is a closed vendor library, so its kernels cannot be read from source
the way the Triton FusedMoE programs can.  The FusedMoE work established the
alternative: measure each deployed compiled program's own operand pipeline over
non-production geometry and publish the coefficients keyed by the program's
launch signature.  This probe is the first half of that for the decode GEMM
programs -- it discovers which (M, N, K) select which compiled program, and
times each one.

Identification is by mcTracer kernel symbol; timing is by MACA graph replay,
because an eager launch on this deployment costs about 6.5 us of host dispatch
on top of the kernel and a decode projection only runs for about 18.

Nothing here selects a Forward parameter.  The shapes are swept to establish
program selection and per-round cadence; the production shapes are run only so
the resulting mechanism can be scored against them afterwards, never fitted.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys


def _timed_graph(torch, a, b, replays: int, inner: int) -> float:
    """Return microseconds per projection, measured inside one graph replay.

    The deployed path is a Linear layer, ``x @ W.T``, which mcBLAS compiles as
    a ``tn`` program.  A plain ``torch.matmul`` of two row-major operands picks
    an ``nn`` program instead, with a split-K ``DevDstElementwiseAdd``
    epilogue, and so does not exercise the deployed kernels at all.
    """

    functional = torch.nn.functional
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            functional.linear(a, b)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            functional.linear(a, b)
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
    del graph
    return statistics.median(spans)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", required=True,
                        help="semicolon-separated M,N,K triples")
    parser.add_argument("--replays", type=int, default=9)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)

    import torch
    if "metax" not in torch.__version__.lower():
        raise RuntimeError("probe requires the MetaX torch build")
    torch.cuda.set_device(0)

    rows = []
    print("%-8s %-8s %-8s %-12s" % ("M", "N", "K", "us_per_matmul"))
    for triple in arguments.shapes.split(";"):
        if not triple.strip():
            continue
        m, n, k = (int(v) for v in triple.split(","))
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        # Linear weight layout is (out_features, in_features).
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        us = _timed_graph(torch, a, b, arguments.replays, arguments.inner)
        weight_bytes = k * n * 2
        rows.append({
            "m": m, "n": n, "k": k,
            "us": round(us, 4),
            "weight_bytes": weight_bytes,
            "achieved_gb_s": round(weight_bytes / (us * 1e-6) / 1e9, 2),
        })
        print("%-8d %-8d %-8d %-12.3f" % (m, n, k, us))
        del a, b
        torch.cuda.empty_cache()

    document = {
        "schema_version": 1,
        "kind": "metax_c500_mcblas_decode_program_probe",
        "hardware_id": "metax_c500",
        "torch_version": torch.__version__,
        "timer_domain": "maca_graph_replay_event",
        "dtype": "bfloat16",
        "measures": "per-matmul device time of the vendor-selected mcBLAS "
                    "program; program identity is read from a companion "
                    "mcTracer capture of the same shape list",
        "production_or_evaluation_duration_consumed": False,
        "inner_per_graph": arguments.inner,
        "replays": arguments.replays,
        "cells": rows,
    }
    with open(arguments.output, "w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    print("wrote %s" % arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
