"""Original pure-text routing wrappers' source allocation requests.

Kernel math and routing values remain separate from this object contract.
"""

import ast
from math import prod
import re
from types import SimpleNamespace

from .qwen3_source import _Tensor
from ..simulation.v4_flash_reference import _cpp_anchor
from .v4_routing_math_source import RoutingMathSource


ROUTING_SOURCE_FILES = {
    "fused_topk_bias_router.py": "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py",
    "dsv4_topk.py": "vllm/model_executor/layers/fused_moe/router/dsv4_topk.py"}


class _RoutingLogits(_Tensor):
    def __init__(self, value, empty):
        super().__init__(value.shape, value.dtype, value.storage, strides=value.strides)
        self.empty = empty

    def new_empty(self, shape, *, dtype):
        return self.empty(*shape, dtype=dtype)


class _Kernel:
    def __init__(self, callback):
        self.callback = callback

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.callback(grid, *args, **kwargs)
        return launch


class _UnknownPDL:
    def __bool__(self):
        raise ValueError("routing source cannot choose unknown compiler/provider PDL support")


class RoutingObjectsSource:
    def __init__(self, source, raw, rank, rows):
        self.source, self.raw, self.rank, self.rows, self.active = source, raw, rank, rows, None
        self.math_source = RoutingMathSource(source, raw)
        torch = SimpleNamespace(float32="fp32", int32="int32", uint32="uint32", int64="int64", empty=self.empty)
        platform = SimpleNamespace(is_cuda=lambda: True, is_arch_support_pdl=lambda: _UnknownPDL())
        common = {"torch": torch, "current_platform": platform, "_TOPK": 6}
        can_use = source.function("dsv4_topk.py", "can_use_dsv4_topk", metadata_globals=common)
        normal = source.function("dsv4_topk.py", "dsv4_topk", metadata_globals={**common,
            "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            "_dsv4_topk_kernel": _Kernel(self.normal_kernel)})
        raw_hash = source.function("fused_topk_bias_router.py", "vllm_topk_softplus_sqrt", metadata_globals={
            **common, "ops": SimpleNamespace(topk_hash_softplus_sqrt=self.hash_kernel),
            "_get_padding_mask": lambda rows: None})
        self.wrapper = source.function("fused_topk_bias_router.py", "fused_topk_bias", metadata_globals={
            **common, "can_use_dsv4_topk": can_use, "dsv4_topk": normal,
            "vllm_topk_softplus_sqrt": raw_hash,
            "rocm_aiter_ops": SimpleNamespace(is_fused_moe_enabled=lambda: False)})
        self.hash_launcher = _cpp_anchor(source.root, "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
            "void topkGatingSoftplusSqrtKernelLauncher(")
        compact = re.sub(r"\s+", "", self.hash_launcher["source"])
        start = compact.index("if(use_hash&&topk==6&&renormalize&&")
        fast = compact[start:compact.index("return;", start)]
        if "launchDsv4HashTopk<IndType,HashIndType>" not in fast or "token_expert_indices" in fast:
            raise ValueError("routing hash fast path now consumes its wrapper temporary")

    def empty(self, *shape, dtype, device=None):
        if self.active is None:
            raise ValueError("routing allocation has no original source wrapper")
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        result = _Tensor(shape, dtype)
        self.active["allocations"].append(result)
        return result

    def normal_kernel(self, grid, logits, bias, weights, ids, scale, input_ids, bias_vl, image_sentinel_lo, **options):
        if (self.active["hashed"] or grid != (logits.shape[0],) or options["HAS_VL"] or bias_vl is not None
                or image_sentinel_lo != 0 or weights.dtype != "fp32" or ids.dtype != "int32"):
            raise ValueError("routing source requires the ordinary pure-text Triton object program")
        self.active["kernel_calls"] += 1
        self.active["math_work"] = self.math_source.run(logits.shape[0], logits.shape[1], False, scale)

    def hash_kernel(self, weights, ids, temporary, logits, renormalize, scale, bias, input_ids, table,
                    *, is_padding, bias_vl, image_sentinel_lo):
        if (not self.active["hashed"] or not renormalize or bias is not None or is_padding is not None
                or bias_vl is not None or image_sentinel_lo != 0 or input_ids.dtype != "int32"
                or table.dtype != "int32" or temporary.dtype != "int32"):
            raise ValueError("routing source requires the declared INT32 pure-text CUDA hash fast path")
        self.active["kernel_calls"] += 1
        self.active["math_work"] = self.math_source.run(logits.shape[0], logits.shape[1], True, scale)

    def run(self, layer, hidden, logits, ids, table, bias):
        if self.active is not None:
            raise ValueError("routing source wrappers cannot overlap")
        hashed = table is not None
        self.active = {"hashed": hashed, "allocations": [], "kernel_calls": 0}
        try:
            output = self.wrapper(hidden, _RoutingLogits(logits, self.empty), self.raw["scoring_func"], bias,
                self.raw["num_experts_per_tok"], True, "int32", ids, table,
                self.raw["routed_scaling_factor"], bias_vl=None, image_sentinel_lo=0)
            roles = ("topk_weights", "topk_ids", "token_expert_indices") if hashed else ("topk_weights", "topk_ids")
            allocated = self.active["allocations"]
            if len(allocated) != len(roles) or self.active["kernel_calls"] != 1 or any(
                    returned is not allocated[index] for index, returned in enumerate(output)):
                raise ValueError("routing source allocation/return program changed")
            self.rows.append({"rank": self.rank, "layer": layer, "hashed": hashed,
                "math_work": self.active["math_work"],
                "objects": {role: {"shape": value.shape, "dtype": value.dtype,
                    "bytes": prod(value.shape) * 4, "returned": role != "token_expert_indices",
                    **({"read_by_kernel": False, "written_by_kernel": False} if role == "token_expert_indices" else {})}
                    for role, value in zip(roles, allocated, strict=True)},
                "source": {"path": str(self.source.paths["fused_topk_bias_router.py"]), "symbol": "fused_topk_bias",
                    "line": next(node for node in self.source.trees["fused_topk_bias_router.py"].body
                        if isinstance(node, ast.FunctionDef) and node.name == "fused_topk_bias").lineno},
                "hash_fast_launcher": self.hash_launcher if hashed else None,
                "condition": "NVIDIA modular pure-text top6, INT32 IDs/table, no padding/VL/EPLB/capture; provider unbound"})
            return output
        finally:
            self.active = None


def compare_v4_routing_objects(reference, traces, graph):
    expected = {(row["rank"], row["layer"]): row for row in reference["routing_object_work"]}
    if len(expected) != len(reference["routing_object_work"]):
        raise ValueError("routing source repeats a rank/layer")
    actual = {}
    for trace in traces:
        allocations = {value.allocation_id: value for value in trace.allocations}
        views = {value.view_id: value for value in trace.views}
        for node in graph.nodes:
            if node.variant != "dsv4_moe_route" or node.rank != trace.rank:
                continue
            match = re.search(r"(?:^|:)layer-(\d+):ffn:route$", node.node_id)
            if match is None:
                raise ValueError("routing object call has no model layer identity")
            key = trace.rank, int(match[1])
            if key in actual:
                raise ValueError("routing actual repeats a rank/layer")
            objects = {}
            prefix = node.node_id + ":"
            for name, allocation in allocations.items():
                if not name.startswith(prefix):
                    continue
                role = name[len(prefix):]
                view = views[name]
                objects[role] = {"shape": view.shape, "dtype": view.dtype, "bytes": allocation.size_bytes,
                    "returned": name in node.attributes["symbolic_tensor_outputs"]}
                if role == "token_expert_indices":
                    objects[role].update(read_by_kernel=name in node.attributes["symbolic_tensor_inputs"],
                        written_by_kernel=name in node.attributes["symbolic_tensor_outputs"])
                    returned = trace.event_order.index(next(event for event in trace.event_order
                        if event.endswith("return:" + node.node_id)))
                    if (allocation.freed_event is None or trace.event_order.index(allocation.allocated_event) >= returned
                            or trace.event_order.index(allocation.freed_event) <= returned):
                        objects[role]["incorrect_wrapper_lifetime"] = True
            actual[key] = objects
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual else
            [role for role in sorted(set(expected[key]["objects"]) | set(actual[key]))
                if role not in expected[key]["objects"] or role not in actual[key]
                or expected[key]["objects"][role] != actual[key][role]])
        checks.append({"rank": key[0], "layer": key[1], "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "status": "mismatched" if differences else "matched"})
    return {"scope": "original pure-text routing wrapper allocation/return requests and hash-unused temporary lifetime",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("logical wrapper requests, not compiled allocator/spills/capacity", "route math, values and persistent table/bias storage remain separate"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
