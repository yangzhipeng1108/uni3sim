"""Probe whether GPU memory pressure during CUDA graph capture changes mcBLAS
kernel selection on MetaX C500.

Mechanism question (independent of any production workload duration):
at decode shapes, does the mcBLAS heuristic pick a different tile program
when the capture happens with most of device memory already allocated,
the way vLLM captures with gpu_memory_utilization?

Each pressure level runs in a fresh subprocess so the mcBLAS per-shape
selection cache cannot carry state across levels. The parent drives one
subprocess per level; the tracer wraps the parent.

Usage:
  python3 mcblas_mem_pressure_capture_probe.py            # parent: all levels
  python3 mcblas_mem_pressure_capture_probe.py --child --alloc-gb 50
"""

import argparse
import json
import os
import subprocess
import sys


# Qwen3-4B decode gate_up projection: (tokens, N=19456, K=2560) bf16.
M_LIST = [2, 4, 8, 16, 32]
N = 19456
K = 2560


def run_child(alloc_gb: float) -> None:
    import torch
    import torch.nn.functional as F

    torch.cuda.init()
    holders = []
    if alloc_gb > 0:
        chunk = torch.empty(int(1 << 30) // 2, dtype=torch.bfloat16, device="cuda")
        n_chunks = int(alloc_gb)
        holders = [chunk.clone() for _ in range(n_chunks)]
    free_b, total_b = torch.cuda.mem_get_info()
    print(f"ALLOC_GB={alloc_gb} free_before={free_b/1e9:.1f}GB total={total_b/1e9:.1f}GB", flush=True)

    w = torch.empty(N, K, dtype=torch.bfloat16, device="cuda").normal_(0, 0.02)
    outs = {}
    graphs = {}
    # Eager warmup per shape.
    for m in M_LIST:
        x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
        outs[m] = F.linear(x, w)
    torch.cuda.synchronize()

    # Capture one graph per m under inference_mode on a side stream,
    # mirroring vLLM's capture wrapper (shared pool not used; pressure is
    # the variable under test).
    for m in M_LIST:
        x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
        with torch.inference_mode():
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    y = F.linear(x, w)
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                y = F.linear(x, w)
            graphs[m] = (g, x, y)
    torch.cuda.synchronize()

    # Replay each graph so the captured program records are attributed.
    for m in M_LIST:
        g, x, y = graphs[m]
        for _ in range(3):
            g.replay()
    torch.cuda.synchronize()

    # Eager calls again after capture, under pressure, to separate
    # capture-time selection from steady-state eager selection.
    for m in M_LIST:
        x = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
        outs[m] = F.linear(x, w)
    torch.cuda.synchronize()

    print(f"CHILD_DONE alloc_gb={alloc_gb}", flush=True)
    del holders, graphs, outs
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--alloc-gb", type=float, default=0.0)
    args = parser.parse_args()

    if args.child:
        run_child(args.alloc_gb)
        return

    for alloc_gb in [0.0, 20.0, 40.0, 50.0, 56.0]:
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", "--alloc-gb", str(alloc_gb)],
            env=env,
            capture_output=True,
            text=True,
        )
        print(proc.stdout, flush=True)
        if proc.returncode != 0:
            print(f"LEVEL_FAILED alloc_gb={alloc_gb} rc={proc.returncode}", flush=True)
            print(proc.stderr[-2000:], flush=True)
    print("ALL_LEVELS_DONE", flush=True)


if __name__ == "__main__":
    main()
