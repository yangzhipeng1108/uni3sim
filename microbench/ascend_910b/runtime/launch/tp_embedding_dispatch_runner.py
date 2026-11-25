#!/usr/bin/env python3
"""Inspect Ascend's TP helper and measure independent primitive Host submits.

The source arm extracts the installed no-added-vocabulary helper unchanged,
records its FX program for both ranks, and consumes no duration. The device
arm times individual primitive dispatches, never the production helper or a
Serving workload. Synchronization and operand allocation are outside timing.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time


def source_helper(path, torch):
    source = path.read_text()
    tree = ast.parse(source)
    owner, = (node for node in tree.body if isinstance(node, ast.ClassDef)
              and node.name == "AscendVocabParallelEmbedding")
    helper, = (node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name == "_get_masked_input_and_mask")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[helper.name], {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "helper_source": ast.get_source_segment(source, helper),
        "forward_source": [ast.get_source_segment(source, node) for node in owner.body
                           if isinstance(node, ast.FunctionDef) and node.name == "_forward_origin"],
    }


def inspect_source(path, torch):
    helper, identity = source_helper(path, torch)
    rows = []
    for rank in (0, 1):
        class Mask(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start, self.end = rank * 64, (rank + 1) * 64

            def forward(self, ids):
                return helper(self, ids, self.start, self.end, 0, 128, 128)

        graph = torch.fx.symbolic_trace(Mask())
        inputs = torch.tensor([0, 63, 64, 127], dtype=torch.int32)
        ids, mask = graph(inputs)
        expected_mask = (inputs < rank * 64) | (inputs >= (rank + 1) * 64)
        assert torch.equal(mask, expected_mask)
        assert torch.equal(ids, torch.where(expected_mask, 0, inputs - rank * 64))
        rows.append({"rank": rank, "fx_code": graph.code,
                     "nodes": dict(Counter(str(node.target) for node in graph.graph.nodes
                                           if node.op.startswith("call_"))),
                     "output_dtypes": [str(ids.dtype), str(mask.dtype)]})
    return {"source": identity, "rank_programs": rows,
            "measurement": "source_structure_only_no_duration"}


def inspect_optimized_npu_graph(path, torch, device):
    """Record the installed backend's graph after its actual optimization.

    This arm has no timed regions. It must precede using primitive counts;
    the local CPU FX graph is only a source-level diagnostic.
    """
    import torch_npu  # noqa: F401
    import torch_npu.dynamo.torchair as torchair
    from torchair._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
    from unittest.mock import patch
    helper, _ = source_helper(path, torch)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    original = AclConcreteGraph.optimize_graph_without_runtime
    records = []

    def observed(graph, *args, **kwargs):
        result = original(graph, *args, **kwargs)
        records.append({"fx_code": graph.fx_graph.code,
                        "nodes": [{"op": node.op, "target": str(node.target)}
                                  for node in graph.fx_graph.graph.nodes]})
        return result

    class OutputMask(torch.nn.Module):
        def forward(self, output, mask):
            return (output.masked_fill_(mask.unsqueeze(-1), 0),)

    rows = []
    for rank in (0, 1):
        class Mask(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.start, self.end = rank * 64, (rank + 1) * 64

            def forward(self, ids):
                return helper(self, ids, self.start, self.end, 0, 128, 128)

        ids = torch.arange(17, dtype=torch.int32, device=device)
        output = torch.ones((17, 128), dtype=torch.bfloat16, device=device)
        mask = ids >= 8
        for stage, module, args in (("mask_ids", Mask(), (ids,)),
                                    ("mask_output", OutputMask(), (output, mask))):
            torch._dynamo.reset()
            records.clear()
            config = torchair.CompilerConfig()
            config.mode = "reduce-overhead"
            config.debug.run_eagerly = True
            config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
            with patch.object(AclConcreteGraph, "optimize_graph_without_runtime", observed):
                compiled = torch.compile(module, fullgraph=True, dynamic=True,
                    backend=torchair.get_npu_backend(compiler_config=config))
                compiled(*args)
                torch.npu.synchronize()
            if not records:
                raise RuntimeError("installed backend did not expose an optimized FX graph")
            rows.append({"rank": rank, "stage": stage, "optimized_graphs": list(records)})
    return rows


def measure_primitives(torch, device, trials):
    import torch_npu  # noqa: F401
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    # Fixed synthetic mechanism domains, independent of evaluation cases.
    domains = ((17, 128), (257, 128), (17, 1024))
    results = []
    for rows, width in domains:
        ids = torch.arange(rows, dtype=torch.int32, device=device)
        boolean = ids >= 64
        offset = 64 * boolean
        output = torch.ones((rows, width), dtype=torch.bfloat16, device=device)
        expanded = boolean.unsqueeze(-1)
        expert_map = torch.arange(16, dtype=torch.int32, device=device)
        topk_ids = (ids % 16).view(-1, 1)
        weights = torch.ones((rows, 1), dtype=torch.bfloat16, device=device)
        owner_mask = boolean.view(-1, 1)
        calls = {
            "ge_scalar": lambda: torch.ops.aten.ge.Scalar(ids, 64),
            "lt_scalar": lambda: torch.ops.aten.lt.Scalar(ids, 128),
            "bool_and": lambda: torch.ops.aten.bitwise_and.Tensor(boolean, boolean),
            "scalar_mul_bool": lambda: torch.ops.aten.mul.Tensor(boolean, 64),
            "scalar_zero_mul_bool": lambda: torch.ops.aten.mul.Tensor(boolean, 0),
            "sub_int32_int64": lambda: torch.ops.aten.sub.Tensor(ids, offset),
            "bool_mul_int64": lambda: torch.ops.aten.mul.Tensor(boolean, offset),
            "bool_invert": lambda: torch.ops.aten.bitwise_not.default(boolean),
            "unsqueeze": lambda: torch.ops.aten.unsqueeze.default(boolean, -1),
            "masked_fill_inplace": lambda: torch.ops.aten.masked_fill_.Scalar(output, expanded, 0),
            "index_int32": lambda: expert_map[topk_ids],
            "ne_scalar": lambda: topk_ids != -1,
            "bf16_mul_bool": lambda: weights * owner_mask,
            "abs_int32": lambda: torch.abs(ids),
        }
        samples = {name: [] for name in calls}
        for _ in range(5):
            for call in calls.values():
                call()
        torch.npu.synchronize()
        for _ in range(trials):
            for name, call in calls.items():
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                result = call()
                samples[name].append(time.perf_counter_ns() - start)
                torch.npu.synchronize()
                del result
        results.append({"rows": rows, "width": width, "samples_ns": samples,
                        "dispatch_entry_source": {name: inspect.getsource(call).strip() for name, call in calls.items()},
                        "median_host_submit_s": {name: statistics.median(values) / 1e9
                                                 for name, values in samples.items()}})
    return results


def main():
    started = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--trials", type=int, default=60)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("trials must be positive")
    import torch
    torch.set_num_threads(1)
    result = inspect_source(args.source, torch)
    result.update({"hardware_id": "ascend_910b", "torch_version": torch.__version__,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "device": args.device, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if args.device is not None:
        result["optimized_npu_graphs"] = inspect_optimized_npu_graph(args.source, torch, args.device)
        result["primitive_host_submit"] = measure_primitives(torch, args.device, args.trials)
        result["pricing_status"] = "raw_full_host_submit_only_not_additive_prepare; enqueue_overlap_must_be_resolved_before_publication"
    result["started_utc"] = started
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
