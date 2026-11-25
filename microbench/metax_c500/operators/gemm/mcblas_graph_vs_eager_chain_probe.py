#!/usr/bin/env python3
"""Compare dependent-chain execution: eager loop vs single graph replay.

Same 36-link dependent GEMM chain as mcblas_graph_chain_overlap_probe,
timed three ways: (a) one graph containing the whole chain, (b) one eager
launch per link (host dispatch between links), (c) sum of isolated
single-link graph replays.  (a) vs (c) isolates in-graph dependent overlap;
(b) vs (a) isolates host dispatch contribution.

Mechanism probe only: no production or evaluation duration is consumed by any
Forward artifact.
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
    k_b = n_a
    weights = [
        torch.randn((n_a, k_a), device="cuda", dtype=torch.bfloat16)
        if i % 2 == 0 else
        torch.randn((k_a, n_a), device="cuda", dtype=torch.bfloat16)
        for i in range(args.links)
    ]
    x0 = torch.randn((m, k_a), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()

    def run_chain():
        h = x0
        for w in weights:
            h = functional.linear(h, w)
        return h

    def timed(fn, inner=1):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                fn()
        torch.cuda.synchronize()
        spans = []
        for _ in range(args.replays):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize()
            spans.append(s.elapsed_time(e) * 1000.0 / inner)
        del graph
        return min(spans), statistics.median(spans)

    def timed_eager():
        run_chain()
        torch.cuda.synchronize()
        spans = []
        for _ in range(args.replays):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            run_chain()
            e.record()
            torch.cuda.synchronize()
            spans.append(s.elapsed_time(e) * 1000.0)
        return min(spans), statistics.median(spans)

    chain_graph = timed(run_chain)
    eager = timed_eager()
    iso = []
    for i, w in enumerate(weights):
        inp = x0 if i % 2 == 0 else torch.randn(
            (m, k_b), device="cuda", dtype=torch.bfloat16)

        def one(w=w, inp=inp):
            return functional.linear(inp, w)

        lo, _ = timed(one)
        iso.append(lo)
    iso_sum = sum(iso)
    print(f"chain graph : min {chain_graph[0]:9.2f} med {chain_graph[1]:9.2f} us")
    print(f"eager chain : min {eager[0]:9.2f} med {eager[1]:9.2f} us")
    print(f"isolated sum: {iso_sum:9.2f} us")
    print(f"graph/isolated = {chain_graph[0] / iso_sum:.4f}")
    print(f"eager/graph  = {eager[0] / chain_graph[0]:.4f}")
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump({
            "schema_version": 1,
            "kind": "metax_c500_graph_vs_eager_chain_probe",
            "m": m, "links": args.links,
            "geometry_a": [n_a, k_a], "geometry_b": [k_a, n_a],
            "replays": args.replays,
            "chain_graph_min_us": chain_graph[0],
            "chain_graph_median_us": chain_graph[1],
            "eager_min_us": eager[0],
            "eager_median_us": eager[1],
            "isolated_min_us": iso,
            "isolated_sum_us": iso_sum,
            "graph_over_isolated": chain_graph[0] / iso_sum,
            "eager_over_graph": eager[0] / chain_graph[0],
            "production_or_evaluation_duration_consumed": False,
        }, stream, indent=2)
        stream.write("\n")
    print("wrote %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
