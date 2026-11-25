"""CUDA-graph dependent-node gap of PyTorch elementwise kernels versus a trivial kernel (H200).

Mechanism probe (no production operator, input or duration).  In the served
SGLang decode graph the CUPTI-domain gap before plain PyTorch kernels is
~0.42 us, while the trivial-node probe (cuda_graph_node_gap.cu) measures
64-96 ns.  This script builds CUDA graphs of 64 dependent launches of one
kernel class each -- torch fill_ (vectorized FillFunctor), in-place scalar
add_ (vectorized BUnary), a strided copy_ (elementwise direct_copy), their
alternation, and a trivial Triton kernel as the control -- on small tensors,
and replays each graph 20 times.  Run it under
``nsys profile --cuda-graph-trace=node``; the reducer reads the per-node
start/end from the sqlite export.
With a second argument ``torch_profiler <out.json.gz>`` the replays run
under torch.profiler (CUDA activities, as SGLang's /start_profile does) and
the trace is exported instead, so the two timers can be compared.
Modes distinct / distinct_big / same_big: 64 distinct Triton binaries per
graph (small or ~2048-op code) versus one big binary repeated, to test
whether per-node code fetch sets the gap.
Modes fork_<k>: a chain of fills with a side-stream node forked and joined
every k nodes, to test whether a branched graph (not a pure chain) sets the gap.
A suffix x<nodes> (e.g. fork_32x2048, fillx2048) sets the graph length; the
served decode graph is ~2000 nodes with a side-stream fork every ~33 nodes.
Usage: python torch_graph_node_gap.py <mode> [torch_profiler <trace.json.gz>]
"""
import sys

import torch
import triton
import triton.language as tl

NODES = 64
REPLAYS = 20


@triton.jit
def _touch(ptr):
    tl.store(ptr, tl.load(ptr) + 1)


# One compiled binary per VARIANT value; UNROLL grows the code size (a chain
# of dependent integer ops that the compiler cannot fold).
@triton.jit
def _distinct(ptr, VARIANT: tl.constexpr, UNROLL: tl.constexpr):
    x = tl.load(ptr)
    for i in tl.static_range(UNROLL):
        x = x * 3 + (VARIANT + i)
    tl.store(ptr, x)


def main():
    mode, _, length = sys.argv[1].partition('x')
    nodes = int(length) if length else NODES
    dev = 'cuda'
    ids = torch.zeros(4, 9, dtype=torch.int32, device=dev)
    w = torch.ones(4, 9, dtype=torch.float32, device=dev)
    src = torch.ones(4, 8, dtype=torch.float32, device=dev)
    sink = torch.zeros(1, dtype=torch.int32, device=dev)

    side = torch.cuda.Stream()
    fork_every = int(mode.split('_')[1]) if mode.startswith('fork_') else 0
    w2 = torch.ones(4, 9, dtype=torch.float32, device=dev)

    def op(i):
        if fork_every:
            # Main chain of fills with a parallel side-stream node forked
            # and joined every fork_every nodes (a DAG, like SGLang's
            # per-layer alt-stream RMSNorm), instead of a pure chain.
            ids.fill_(i)
            if i % fork_every == 0:
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    w2.mul_(1.0001)
                torch.cuda.current_stream().wait_stream(side)
            return
        kind = mode if mode != 'mixed' else ('fill', 'add', 'copy')[i % 3]
        if kind == 'fill':
            ids.fill_(i)
        elif kind == 'add':
            w.mul_(1.0001)
        elif kind == 'copy':
            w[:, :8].copy_(src)
        elif kind == 'triton':
            _touch[(1,)](sink)
        elif kind == 'distinct':
            _distinct[(1,)](sink, VARIANT=i, UNROLL=1)
        elif kind == 'distinct_big':
            _distinct[(1,)](sink, VARIANT=i, UNROLL=2048)
        elif kind == 'same_big':
            _distinct[(1,)](sink, VARIANT=0, UNROLL=2048)
        else:
            raise ValueError(mode)

    for i in range(nodes):
        op(i)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for i in range(nodes):
                op(i)
    if len(sys.argv) == 4 and sys.argv[2] == 'torch_profiler':
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(REPLAYS):
                graph.replay()
            torch.cuda.synchronize()
        prof.export_chrome_trace(sys.argv[3])
    else:
        for _ in range(3 + REPLAYS):
            graph.replay()
        torch.cuda.synchronize()
    print('done', sys.argv[1])


if __name__ == '__main__':
    main()
