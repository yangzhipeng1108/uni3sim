"""Independent default-PyTorch capture -> first replay -> warm replay probe.

No weights, model, attention, routing or production input. Measures native
Graph objects with1/37 segments and1/8/32 dependent add nodes per segment.
Capture completes default instantiation before timing. No deferred-instantiate
price or production-duration correction is inferred from these windows.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import inspect
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch

SEGMENTS = (1, 37)
NODES = (1, 8, 32)
ELEMENTS = (4096, 32768)


def preflight():
    x = torch.ones(4096)
    for _ in range(37):
        for _ in range(8):
            torch.add(x, 1, out=x)
    assert torch.all(x == 1 + 37 * 8)
    assert not torch.cuda.is_initialized()
    return {"cpu_preflight_only": True, "state_update_checksum": True}


def measure(segments, nodes, elements, trial):
    x = torch.ones(elements, device="cuda", dtype=torch.float32)
    # Initialize primitive dispatch/module and all event APIs before capture.
    for _ in range(4):
        torch.add(x, 1, out=x)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(); end.record(); end.synchronize()
    graphs = []
    started = time.perf_counter_ns()
    for index in range(segments):
        graph = torch.cuda.CUDAGraph()  # default keep_graph=False, eager instantiate
        # vLLM suppresses repeated gc/empty-cache work after the first piece.
        gc_scope = patch("gc.collect", lambda: None) if index else nullcontext()
        cache_scope = patch("torch.cuda.empty_cache", lambda: None) if index else nullcontext()
        with gc_scope, cache_scope, torch.cuda.graph(graph):
            for _ in range(nodes):
                torch.add(x, 1, out=x)
        graphs.append(graph)
    capture_ns = time.perf_counter_ns() - started
    readings = []
    # The exact same already-instantiated Graph objects are replayed repeatedly.
    for repeat in range(6):
        x.fill_(1)
        torch.cuda.synchronize()
        begin.record()
        start = time.perf_counter_ns()
        for graph in graphs:
            graph.replay()
        enqueued = time.perf_counter_ns()
        end.record(); end.synchronize()
        completed = time.perf_counter_ns()
        assert torch.all(x == 1 + segments * nodes).item()
        readings.append({"replay_index": repeat, "first_replay": repeat == 0,
            "host_enqueue_ns": enqueued - start, "completion_ns": completed - start,
            "cuda_event_ns": begin.elapsed_time(end) * 1e6})
    return {"segments": segments, "nodes_per_segment": nodes, "elements": elements,
        "trial": trial, "capture_and_instantiation_ns": capture_ns, "samples": readings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    if args.cpu_preflight:
        print(json.dumps(preflight())); return
    if args.output is None or args.output.exists():
        raise ValueError("requires new raw output")
    if "H200" not in torch.cuda.get_device_name(0):
        raise ValueError("authorized H200 required")
    rows = []
    for trial in range(5):
        cells = [(segments, nodes, elements) for segments in SEGMENTS for nodes in NODES for elements in ELEMENTS]
        if trial % 2:
            cells.reverse()
        for segments, nodes, elements in cells:
            rows.append(measure(segments, nodes, elements, trial))
    payload = {"kind": "independent_pytorch_segmented_graph_first_replay", "cpu_preflight_only": False,
        "production_duration_consumed": False, "model_or_weights_executed": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
        "graph_enter_source": inspect.getsource(torch.cuda.graph.__enter__),
        "cudagraph_source": inspect.getsource(torch.cuda.CUDAGraph),
        "boundary": "capture/default instantiation outside; Host replay enqueue and event/completion separately",
        "limitations": ["independent add nodes, not production compiler wrappers or native model",
            "CUDA event includes Host enqueue gaps; it is not pure kernel service",
            "first-versus-warm association does not prove driver/module root cause",
            "no direct graph-upload API or profiler trace is used"], "rows": rows}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "complete", "cells": len(rows)}))


if __name__ == "__main__":
    main()
