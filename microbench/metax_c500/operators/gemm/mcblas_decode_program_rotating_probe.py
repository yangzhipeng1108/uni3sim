"""Same deployed program, but rotate weight buffers so no replay reuses L2.

In serving, 48 distinct layer weights stream between two reads of the same
weight, so a weight is never L2-resident when it is re-read.  The existing
probe replays one weight back to back, which can retain part of it.  This
variant rotates over `--buffers` distinct weights so the working set greatly
exceeds L2, reproducing the serving residency condition.
"""
import argparse, json, torch, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--shapes", required=True)
p.add_argument("--buffers", type=int, default=8)
p.add_argument("--replays", type=int, default=9)
p.add_argument("--inner", type=int, default=20)
a = p.parse_args()
torch.cuda.set_device(0)
print("%-6s %-7s %-7s %-4s %-12s" % ("M","N","K","buf","us_per_matmul"))
for triple in a.shapes.split(";"):
    if not triple.strip():
        continue
    m, n, k = (int(v) for v in triple.split(","))
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
          for _ in range(a.buffers)]
    for _ in range(3):
        for w in ws:
            F.linear(x, w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(a.inner):
            F.linear(x, ws[i % a.buffers])
    torch.cuda.synchronize()
    best = None
    for _ in range(a.replays):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000.0 / a.inner
        best = us if best is None else min(best, us)
    mb = n * k * 2 * a.buffers / 1e6
    print("%-6d %-7d %-7d %-4d %-12.3f  (working set %.1f MB)"
          % (m, n, k, a.buffers, best, mb))
    del x, ws, g
    torch.cuda.empty_cache()
