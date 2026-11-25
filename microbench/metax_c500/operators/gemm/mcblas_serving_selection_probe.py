#!/usr/bin/env python3
"""Identify the mcBLAS program the deployment stack selects per GEMM shape.

Runs each (M, N, K) as torch.nn.functional.linear(x, W) -- the exact call
vLLM's Linear layers make -- twice over: eager and inside MACA graph
capture/replay.  Program identity (kernel symbol, grid, block, smem) is read
from the companion mcTracer capture of this process; per-mode runs keep the
shape order identical so the kernel sequence chunks deterministically.

Nothing here feeds a Forward parameter: identity is a discrete launch fact,
and the graph-replay timing is recorded only to validate the operand-pipeline
mechanism afterwards, never to fit it.

Usage: mcblas_serving_selection_probe.py --mode eager|graph \
          --shapes "M,N,K;..." --output OUT.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

INNER = 20
EAGER_CALLS = 10
REPLAYS = 5


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--shapes", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import torch
    if "metax" not in torch.__version__.lower():
        raise RuntimeError("probe requires the MetaX torch build")
    torch.cuda.set_device(0)

    shapes = []
    for triple in args.shapes.split(";"):
        if triple.strip():
            shapes.append(tuple(int(v) for v in triple.split(",")))

    rows = []
    for m, n, k in shapes:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        if args.mode == "eager":
            for _ in range(3):
                torch.nn.functional.linear(a, b)
            torch.cuda.synchronize()
            for _ in range(EAGER_CALLS):
                torch.nn.functional.linear(a, b)
            torch.cuda.synchronize()
            us = None
        else:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    torch.nn.functional.linear(a, b)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(INNER):
                    torch.nn.functional.linear(a, b)
            torch.cuda.synchronize()
            spans = []
            for _ in range(REPLAYS):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                spans.append(start.elapsed_time(end) * 1000.0 / INNER)
            us = statistics.median(spans)
            del graph
        rows.append({"m": m, "n": n, "k": k,
                     "us_per_linear_graph_replay": us})
        print(f"{args.mode} M={m} N={n} K={k} us={us}", flush=True)
        del a, b
        torch.cuda.empty_cache()

    doc = {
        "schema_version": 1,
        "kind": "metax_c500_mcblas_serving_selection_probe",
        "hardware_id": "metax_c500",
        "mode": args.mode,
        "torch_version": torch.__version__,
        "eager_calls_per_shape": EAGER_CALLS if args.mode == "eager" else 0,
        "graph_inner_per_replay": INNER if args.mode == "graph" else 0,
        "graph_replays": REPLAYS if args.mode == "graph" else 0,
        "production_or_evaluation_duration_consumed": False,
        "cells": rows,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
