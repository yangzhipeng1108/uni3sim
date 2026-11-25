#!/usr/bin/env python3
"""Probe whether a MACA graph replay overlaps a dependent kernel chain.

Captures two forms of the same 36-link dependent GEMM chain
(x <- W_i x, alternating two projection geometries) and compares:

- ``chained``: all 36 links inside one cuda graph, one replay timed.
- ``isolated``: each link timed alone (own graph, rotating cold weights).

If chained < isolated sum by much more than dispersion, graph replay is not
fully serializing dependent kernels on this stack (PDL-style tail overlap or
equivalent), and standalone kernel durations overstate in-graph cost.

Mechanism probe only: non-production M, no production or evaluation duration
is consumed by any Forward artifact.
"""

from __future__ import annotations

import argparse
import json
import statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=160)
    parser.add_argument("--links", type=int, default=36)
    parser.add_argument("--k-a", type=int, default=2560)
    parser.add_argument("--n-a", type=int, default=19456)
    parser.add_argument("--replays", type=int, default=9)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    if "metax" not in torch.__version__.lower():
        raise RuntimeError("probe requires the MetaX torch build")
    torch.cuda.set_device(args.device)
    functional = torch.nn.functional

    m, k_a, n_a = args.m, args.k_a, args.n_a
    k_b, n_b = n_a, k_a
    links = args.links
    weights = []
    for i in range(links):
        if i % 2 == 0:
            weights.append(torch.randn((n_a, k_a), device="cuda",
                                       dtype=torch.bfloat16))
        else:
            weights.append(torch.randn((n_b, k_b), device="cuda",
                                       dtype=torch.bfloat16))
    torch.cuda.synchronize()

    def time_graph(build, inner=1):
        x = torch.randn((m, k_a), device="cuda", dtype=torch.bfloat16)
        build(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                build(x)
        torch.cuda.synchronize()
        spans = []
        for _ in range(args.replays):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            spans.append(start.elapsed_time(end) * 1000.0 / inner)
        del graph, x
        return min(spans), statistics.median(spans)

    def chained(x):
        h = x
        for w in weights:
            h = functional.linear(h, w)
        return h

    chain_min, chain_med = time_graph(chained)

    # Isolated per-link timing: single link per graph, rotating the same
    # weights so cold-DRAM conditions match the chain.
    iso = []
    for i in range(links):
        w = weights[i]

        def one(x, w=w, first=(i % 2 == 0)):
            return functional.linear(x if first else torch.zeros(
                (m, k_b), device="cuda", dtype=torch.bfloat16), w)

        lo, mid = time_graph(one)
        iso.append(lo)
        print(f"link {i:2d} isolated min {lo:9.2f} us", flush=True)
    iso_sum = sum(iso)
    print(f"chained graph min {chain_min:.2f} us (median {chain_med:.2f})")
    print(f"isolated sum min {iso_sum:.2f} us")
    print(f"ratio chained/isolated = {chain_min / iso_sum:.4f}")
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump({
            "schema_version": 1,
            "kind": "metax_c500_graph_chain_overlap_probe",
            "m": m,
            "links": links,
            "geometry_a": [n_a, k_a],
            "geometry_b": [n_b, k_b],
            "replays": args.replays,
            "chained_min_us": chain_min,
            "chained_median_us": chain_med,
            "isolated_min_us": iso,
            "isolated_sum_us": iso_sum,
            "ratio": chain_min / iso_sum,
            "production_or_evaluation_duration_consumed": False,
        }, stream, indent=2)
        stream.write("\n")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
