"""Actual source allocations of the C4 local-to-global decode index helper."""

import re
from types import SimpleNamespace

from .qwen3_source import _Tensor


DECODE_STORAGE_SOURCE_FILES = {
    "decode_cache_storage.py": "vllm/models/deepseek_v4/common/ops/cache_utils.py"}


def decode_index_storage_reference(source, raw, inputs):
    count = sum(q for q, phase in zip(inputs["query_lengths"], inputs["request_phases"], strict=True) if phase == "decode")
    if not count:
        return []
    source_inputs = (_Tensor((count, raw["index_topk"]), "int32"), _Tensor((count,), "int32"),
        SimpleNamespace(parameter="external valid INT32 block table; shape and addresses unbound"),
        SimpleNamespace(parameter="positive external kernel block size"), _Tensor((count,), "bool"))
    allocations, seen = [], []

    def empty(*shape, dtype, device):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = _Tensor(shape, dtype)
        allocations.append(out)
        return out

    def empty_like(value):
        return empty(value.shape, dtype=value.dtype, device=value.device)

    def kernel(indices, lengths, *arguments):
        if ((indices, lengths) != tuple(allocations) or arguments != source_inputs
                or indices.shape != source_inputs[0].shape or lengths.shape != (count,)
                or indices.dtype != "int32" or lengths.dtype != "int32"):
            raise ValueError("C4 index conversion source tensor program changed")
        seen.append(True)

    function = source.function("decode_cache_storage.py", "compute_global_topk_indices_and_lens",
        metadata_globals={"torch": SimpleNamespace(empty_like=empty_like, empty=empty, int32="int32"),
            "_COMPUTE_GLOBAL_TOPK_INDICES_AND_LENS_KERNEL": kernel})
    outputs = function(*source_inputs)
    if len(allocations) != 2 or len(seen) != 1 or tuple(outputs) != tuple(allocations):
        raise ValueError("C4 index helper must return its two fresh allocations")
    definition = next(node for node in source.trees["decode_cache_storage.py"].body
        if hasattr(node, "name") and node.name == "compute_global_topk_indices_and_lens")
    objects = [{"role": role, "shape": value.shape, "dtype": value.dtype, "bytes": 4 * value.numel()}
        for role, value in zip(("decode-global-topk", "decode-topk-lens"), outputs, strict=True)]
    return [{"rank": rank, "layer": layer, "phase": "decode", "objects": objects,
        "source": {"path": str(source.paths["decode_cache_storage.py"]), "symbol": definition.name, "line": definition.lineno},
        "condition": "C4 decode with declared valid metadata; block size/physical addresses are external, buffer extents do not depend on them"}
        for rank in range(inputs["TP"]) for layer, ratio in enumerate(raw["compress_ratios"][:raw["num_hidden_layers"]]) if ratio == 4]


def compare_v4_decode_index_storage(reference, traces, graph):
    expected = {(r["rank"], r["layer"]): r for r in reference}
    if len(expected) != len(reference):
        raise ValueError("C4 source index storage repeats a layer/rank")
    actual = {}
    for trace in traces:
        views = {v.view_id: v for v in trace.views}
        allocations = {a.allocation_id: a for a in trace.allocations}
        events = {event: i for i, event in enumerate(trace.event_order)}
        for node in graph.nodes:
            if node.rank != trace.rank or node.variant != "dsv4_sparse_decode" or node.shape["compress_ratio"] != 4:
                continue
            match = re.search(r"(?:^|:)layer-(\d+):attn:sparse(?::\d+)?$", node.node_id)
            if match is None:
                continue
            key = trace.rank, int(match[1])
            if key in actual:
                raise ValueError("actual C4 decode index storage repeats a layer/rank")
            objects, valid = [], "symbolic_decode_index_buffers" in node.attributes
            if valid:
                ends = [i for event, i in events.items() if event.endswith("return:" + node.node_id)]
                if len(ends) != 1:
                    raise ValueError("decode storage needs its actual source call boundary")
                for identifier in node.attributes["symbolic_decode_index_buffers"]:
                    view = views[identifier]
                    allocation = allocations[view.allocation_id]
                    objects.append({"role": identifier.removeprefix(node.node_id + ":"),
                        "shape": view.shape, "dtype": view.dtype, "bytes": allocation.size_bytes})
                    valid = (valid and events[allocation.allocated_event] < ends[0]
                        and allocation.freed_event is not None and events[allocation.freed_event] > ends[0]
                        and identifier in node.attributes["symbolic_tensor_outputs"])
                valid = valid and len({views[v].allocation_id for v in node.attributes["symbolic_decode_index_buffers"]}) == 2
            actual[key] = {"objects": objects, "live_through_decode_then_released": valid}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        want = None if key not in expected else {"objects": expected[key]["objects"], "live_through_decode_then_released": True}
        got = actual[key] if key in actual else None
        differences = (["unexpected"] if want is None else ["missing"] if got is None else
            [field for field in want if want[field] != got[field]])
        checks.append({"rank": key[0], "layer": key[1], "expected": want, "actual": got,
            "source": expected[key]["source"] if key in expected else None,
            "differences": differences, "status": "mismatched" if differences else "matched"})
    return {"scope": "C4 decode index helper's two source allocations and lifetime; sparse callee/planner storage is separate",
        "total": len(checks), "matched": sum(not r["differences"] for r in checks),
        "mismatched": sum(bool(r["differences"]) for r in checks), "not_checked": 0, "checks": checks,
        "complete_sparse_storage_accepted": False, "runtime_accepted": False}
