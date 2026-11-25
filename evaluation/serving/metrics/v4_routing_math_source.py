"""Execute the original ordinary top6 kernel with metadata registers.

Only six fixed selection iterations execute. No logit, expert selection,
array payload, tensor, compiler or device instruction is evaluated.
"""

import ast
from collections import Counter
from copy import deepcopy
from math import prod
import re

from ..simulation.v4_flash_reference import _cpp_anchor
from ..simulation.v4_flash_routing_reference import routing_source_reference


def _broadcast(left, right):
    n = max(len(left), len(right))
    a, b = (1,) * (n - len(left)) + left, (1,) * (n - len(right)) + right
    if any(x != y and x != 1 and y != 1 for x, y in zip(a, b, strict=True)):
        raise ValueError("routing source shapes cannot broadcast")
    return tuple(max(x, y) for x, y in zip(a, b, strict=True))


class _Value:
    def __init__(self, owner, shape=(), dtype="fp32", *, interval=None, active=None):
        self.owner, self.shape, self.dtype = owner, tuple(shape), dtype
        self.interval, self.active = interval, active

    def __bool__(self):
        raise ValueError("routing source requires an undeclared logit/value branch")

    def operand(self, other):
        if isinstance(other, _Value):
            if other.owner is not self.owner:
                raise ValueError("routing source mixes kernel contexts")
            return other
        if type(other) in (int, float):
            return _Value(self.owner, dtype="fp32" if type(other) is float else "int32")
        return NotImplemented

    def binary(self, other, operation):
        other = self.operand(other)
        if other is NotImplemented:
            return NotImplemented
        shape = _broadcast(self.shape, other.shape)
        dtype = "fp32" if "fp32" in (self.dtype, other.dtype) else "int32"
        if dtype == "fp32":
            self.owner.arithmetic["compute_fma"] += prod(shape)
        else:
            self.owner.index_actions[operation] += prod(shape)
        return _Value(self.owner, shape, dtype)

    def __add__(self, other):
        return self.binary(other, "add")

    __radd__ = __add__

    def __mul__(self, other):
        return self.binary(other, "multiply")

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.binary(other, "divide")

    def __rtruediv__(self, other):
        return self.operand(other).binary(self, "divide")

    def compare(self, other, operation):
        scalar = other
        other = self.operand(other)
        if other is NotImplemented:
            return NotImplemented
        shape = _broadcast(self.shape, other.shape)
        dtype = "fp32" if "fp32" in (self.dtype, other.dtype) else "int32"
        self.owner.typed[(dtype, operation)] += prod(shape)
        active = None
        if operation == "less_than" and self.interval is not None and type(scalar) is int:
            begin, end = self.interval
            active = max(0, min(end, scalar) - begin)
        return _Value(self.owner, shape, "bool", active=active)

    def __lt__(self, other):
        return self.compare(other, "less_than")

    def __gt__(self, other):
        return self.compare(other, "greater_than")

    def __eq__(self, other):
        return self.compare(other, "equal")

    def to(self, dtype):
        if dtype not in ("fp32", "int32"):
            raise ValueError("routing math source has an unbound conversion")
        if dtype != self.dtype:
            self.owner.conversions[(self.dtype, dtype)] += prod(self.shape)
        return _Value(self.owner, self.shape, dtype)


class _Pointer:
    def __init__(self, owner, role, dtype, shape=()):
        self.owner, self.role, self.dtype, self.shape = owner, role, dtype, shape

    def __add__(self, index):
        if not isinstance(index, _Value) or index.dtype != "int32":
            raise ValueError("routing source pointer requires a logical integer offset")
        return _Pointer(self.owner, self.role, self.dtype, _broadcast(self.shape, index.shape))


class _TL:
    float32, int32 = "fp32", "int32"

    def __init__(self):
        self.arithmetic, self.typed, self.index_actions, self.conversions = Counter(), Counter(), Counter(), Counter()
        self.traffic = Counter()

    def program_id(self, axis):
        if axis != 0:
            raise ValueError("routing source requires the row program axis")
        return _Value(self, dtype="int32")

    def arange(self, start, end):
        if type(start) is not int or type(end) is not int or start != 0 or end <= start:
            raise ValueError("routing source range must be an explicit zero-based extent")
        return _Value(self, (end - start,), "int32", interval=(start, end))

    def zeros(self, shape, *, dtype):
        return _Value(self, tuple(shape), dtype)

    def static_range(self, count):
        if count != 6:
            raise ValueError("routing source requires exactly six source selection rounds")
        return range(count)

    def load(self, pointer, *, mask, other):
        if mask.active is None or mask.shape != pointer.shape:
            raise ValueError("routing source load has an undeclared value-dependent mask")
        self.traffic[(pointer.role, "read")] += 4 * mask.active
        return _Value(self, pointer.shape, pointer.dtype)

    def store(self, pointer, value, *, mask):
        if mask.active is None or mask.shape != value.shape or pointer.dtype != value.dtype:
            raise ValueError("routing source store has an unbound mask/type")
        self.traffic[(pointer.role, "write")] += 4 * mask.active

    def where(self, condition, first, second):
        if not isinstance(condition, _Value) or condition.dtype != "bool":
            raise ValueError("routing source select requires a typed predicate")
        prototype = _Value(self)
        a, b = prototype.operand(first), prototype.operand(second)
        shape = _broadcast(condition.shape, _broadcast(a.shape, b.shape))
        dtype = "fp32" if "fp32" in (a.dtype, b.dtype) else "int32"
        self.typed[(dtype, "select")] += prod(shape)
        return _Value(self, shape, dtype)

    def unary(self, value):
        self.arithmetic["compute_sfu"] += prod(value.shape)
        return _Value(self, value.shape, value.dtype)

    exp, log, sqrt = unary, unary, unary

    def reduce(self, value, axis, operation):
        if axis != 0 or len(value.shape) != 1:
            raise ValueError("routing source requires a vector reduction")
        count = value.shape[0] - 1
        if operation == "sum":
            if value.dtype != "fp32":
                raise ValueError("routing source sum requires FP32 values")
            self.arithmetic["compute_fma"] += count
        else:
            self.typed[(value.dtype, operation)] += count
        return _Value(self, dtype=value.dtype)

    def sum(self, value, *, axis):
        return self.reduce(value, axis, "sum")

    def max(self, value, *, axis):
        return self.reduce(value, axis, "maximum")

    def min(self, value, *, axis):
        return self.reduce(value, axis, "minimum")


class _ExcludePDL(ast.NodeTransformer):
    """Keep unknown PDL control out of the logical arithmetic/byte program."""
    def visit_If(self, node):
        if isinstance(node.test, ast.Name) and node.test.id == "HAS_VL":
            return [self.visit(item) for item in node.orelse] if node.orelse else ast.copy_location(ast.Pass(), node)
        if isinstance(node.test, ast.Name) and node.test.id == "launch_pdl":
            if (node.orelse or len(node.body) != 1 or ast.unparse(node.body[0]) not in (
                    "tl.extra.cuda.gdc_wait()", "tl.extra.cuda.gdc_launch_dependents()")):
                raise ValueError("routing PDL branch changes math/memory and cannot be excluded")
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


class TritonRoutingMathSource:
    def __init__(self, source):
        kernels = [node for node in ast.walk(source.trees["dsv4_topk.py"])
            if isinstance(node, ast.FunctionDef) and node.name == "_dsv4_topk_kernel"]
        self.kernel, = kernels
        self.source = source
        self.active = _ExcludePDL().visit(deepcopy(self.kernel))
        bindings = {
            "row": ["tl.program_id(0)"], "expert_offsets": ["tl.arange(0, BLOCK_N)"],
            "expert_mask": ["expert_offsets < NUM_EXPERTS"], "topk_offsets": ["tl.arange(0, 8)"],
            "output_mask": ["topk_offsets < 6"], "output_offsets": ["row * 6 + topk_offsets"],
            "bias": ["tl.load(correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)"],
            "logits": ["tl.load(gating_output_ptr + row * NUM_EXPERTS + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)"],
            "weights": ["tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))"],
            "current": ["tl.where(expert_mask, weights + bias, -float('inf'))",
                "tl.where(current == current, current, -1e30)", "tl.where(expert_offsets == expert_id, -float('inf'), current)"],
            "max_value": ["tl.max(current, axis=0)"],
            "candidate": ["tl.where(current == max_value, expert_offsets, NUM_EXPERTS)"],
            "expert_id": ["tl.min(candidate, axis=0).to(tl.int32)"],
            "selected_weight": ["tl.sum(tl.where(expert_offsets == expert_id, weights, 0.0), axis=0)"],
            "is_slot": ["topk_offsets == slot"],
            "selected_weights": ["tl.zeros([8], dtype=tl.float32)", "tl.where(is_slot, selected_weight, selected_weights)"],
            "selected_ids": ["tl.zeros([8], dtype=tl.int32)", "tl.where(is_slot, expert_id, selected_ids)"],
            "weight_sum": ["tl.sum(selected_weights, axis=0)"]}
        for name, expressions in bindings.items():
            assignments = [node for node in ast.walk(self.active) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
            if [ast.unparse(node.value) for node in assignments] != [ast.unparse(ast.parse(value, mode="eval").body) for value in expressions]:
                raise ValueError("routing source value/index binding changed: " + name)
        updates = [node for node in ast.walk(self.active) if isinstance(node, ast.AugAssign)]
        if [ast.unparse(node) for node in updates] != [ast.unparse(ast.parse(
                "selected_weights *= routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)").body[0])]:
            raise ValueError("routing source output normalization changed")

    def run(self, rows, experts):
        tl = _TL()
        function = self.source._compile(deepcopy(self.active), "_dsv4_topk_kernel", {"tl": tl})
        pointers = [_Pointer(tl, role, dtype) for role, dtype in
            (("logits", "fp32"), ("bias_per_row", "fp32"), ("weights", "fp32"), ("ids", "int32"))]
        function(*pointers, _Value(tl), None, None, 0, experts, 1 << (experts - 1).bit_length(), False, _Value(tl, dtype="bool"))
        return {"demand": {"compute_tensor": 0, "compute_reduction": 0,
                "compute_fma": rows * tl.arithmetic["compute_fma"], "compute_sfu": rows * tl.arithmetic["compute_sfu"], "compute_alu": 0},
            "typed_actions": {dtype + ":" + operation: rows * count for (dtype, operation), count in sorted(tl.typed.items())},
            "operand_bytes": {role + ":" + access: rows * count for (role, access), count in sorted(tl.traffic.items())},
            "unpriced_index_address_actions": {name: rows * count for name, count in tl.index_actions.items()},
            "conversions": {left + ":" + right: rows * count for (left, right), count in tl.conversions.items()},
            "source": {"path": str(self.source.paths["dsv4_topk.py"]), "line": self.kernel.lineno,
                "symbol": self.kernel.name},
            "condition": "ordinary pure-text top6; padding expressions retained; no logit/value branch chosen; PDL service excluded"}


def _actions(actions):
    totals = Counter()
    for action in actions.values():
        operation = "equal" if action["operation"] == "equal_self" else action["operation"]
        totals[action["dtype"] + ":" + operation] += action["elements"]
    return dict(totals)


class RoutingMathSource:
    def __init__(self, source, raw):
        self.source, self.raw = source, raw
        self.ordinary = TritonRoutingMathSource(source)
        self.hash = _cpp_anchor(source.root, "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
            "void dsv4HashTopkSoftplusSqrt(")
        widening = _cpp_anchor(source.root, "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu",
            "int64_t load_index_as_int64(")["source"]
        if re.sub(r"\s+", "", widening[widening.index("{") + 1:-1]) != "returnstatic_cast<int64_t>(ptr[offset]);":
            raise ValueError("hash routing ID widening needs a newly derived source program")
        compact = re.sub(r"\s+", "", re.sub(r"#[^\n]*|//[^\n]*", "", self.hash["source"]))
        for call in ("cudaGridDependencySynchronize();", "cudaTriggerProgrammaticLaunchCompletion();"):
            compact = compact.replace(call, "")
        prefix = compact[compact.index("{") + 1:compact.index("if(is_image&&!is_pad_row){")]
        if prefix != ("constintwarp=(blockIdx.x*blockDim.x+threadIdx.x)/32;constintlane=threadIdx.x%32;"
                "if(warp>=num_rows)return;constint64_ttoken_id=load_index_as_int64(input_ids,warp);"
                "constboolis_pad_row=is_padding!=nullptr&&is_padding[warp];"
                "constboolis_image=bias_vl!=nullptr&&image_sentinel_lo>0&&token_id>=image_sentinel_lo&&token_id<image_sentinel_lo+5;"
                "intexpert=0;floatweight=0.f;"):
            raise ValueError("hash routing prefix needs a newly derived source program")
        marker = "}elseif(lane<6&&!is_pad_row){"
        begin, end = compact.index(marker) + len(marker), compact.index("floatweight_sum=weight;")
        if compact[begin:end] != ("expert=static_cast<int>(tid2eid[token_id*6+lane]);"
                "constfloatx=input[warp*num_experts+expert];"
                "weight=sqrtf(fmaxf(x,0.f)+__logf(1.f+__expf(-fabsf(x))));"
                "if(isnan(weight)){weight=0.f;}}"):
            raise ValueError("hash routing active expressions need a newly derived source program")
        if compact[end:-1] != ("floatweight_sum=weight;for(intmask=16;mask>0;mask>>=1){"
                "weight_sum+=VLLM_SHFL_XOR_SYNC(weight_sum,mask);}if(lane<6){constintoffset=warp*6+lane;"
                "output[offset]=weight*routed_scaling_factor/(weight_sum>0.f?weight_sum:1.f);"
                "indices[offset]=!is_pad_row?static_cast<OutIndType>(expert):static_cast<OutIndType>(-1);}"):
            raise ValueError("hash routing reduction/output needs a newly derived source program")

    def run(self, rows, experts, hashed, scale):
        expected = routing_source_reference(self.raw, rows, hashed)
        if experts != self.raw["n_routed_experts"]:
            raise ValueError("routing source runtime width disagrees with its raw checkpoint config")
        if hashed:
            result = {"demand": expected["demand"], "typed_actions": _actions(expected["unpriced_typed_actions"]),
                "operand_bytes": {row["role"] + ":" + row["access"]: row["bytes"] for row in expected["operands"]},
                "source": {name: self.hash[name] for name in ("path", "line", "symbol")},
                "condition": "declared CUDA pure-text top6, no padding/VL; NaN conditional assignment count remains unknown"}
        else:
            result = self.ordinary.run(rows, experts)
        result.update(routing_source=expected["source"], conditional_action_counts=expected["conditional_action_counts"],
            shape={"rows": rows, "experts": experts, "topk": self.raw["num_experts_per_tok"]},
            input_dtype="fp32", routed_scaling_factor=scale,
            source_program=expected["source_program"], arithmetic_groups=expected["arithmetic_groups"],
            unpriced_typed_actions=expected["unpriced_typed_actions"],
            baseline_read_write_bytes=expected["baseline_read_write_bytes"], logical_bytes=expected["logical_bytes"])
        result["source_role_read_write_bytes"] = {access: sum(count for role, count in result["operand_bytes"].items()
            if role.endswith(":" + access)) for access in ("read", "write")}
        return result


def compare_v4_routing_math(reference, plan, hardware):
    from src.kernel.kernel_base import create_predictor
    from src.kernel.predictors.vector import dsv4_route_source_work
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    fields = ("routing_source", "shape", "input_dtype", "routed_scaling_factor", "demand", "typed_actions", "operand_bytes", "conditional_action_counts",
        "source_program", "arithmetic_groups", "unpriced_typed_actions", "source_role_read_write_bytes",
        "baseline_read_write_bytes", "logical_bytes")
    expected = {(row["rank"], row["layer"]): row["math_work"] for row in reference["routing_object_work"]}
    if len(expected) != len(reference["routing_object_work"]):
        raise ValueError("routing math source repeats a rank/layer")
    actual = {}
    for call in plan.calls:
        if call.kernel_spec.variant != "dsv4_moe_route":
            continue
        matches = [re.search(r"(?:^|:)layer-(\d+):ffn:route$", name) for name in call.source_node_ids]
        match, = (match for match in matches if match is not None)
        key = call.rank, int(match[1])
        if key in actual:
            raise ValueError("routing math actual repeats a rank/layer")
        work = dsv4_route_source_work(call.kernel_spec)
        owner, compulsory = create_predictor(call.kernel_spec.kind).resource_input(call.kernel_spec, hardware_for_call(call, hardware))
        traffic = {role: {access: sum(node.bytes for path in owner.memory_paths if path.access_kind == access
            for node in path.nodes if node.resource_role == role) for access in ("read", "write")} for role in ("dram", "l2_cache")}
        actual[key] = {"routing_source": work["source"], "demand": work["demand"],
            "shape": {name: call.kernel_spec.shape[name] for name in ("rows", "experts", "topk")},
            "input_dtype": call.kernel_spec.dtype, "routed_scaling_factor": call.kernel_spec.attributes["routed_scaling_factor"],
            "typed_actions": _actions(work["unpriced_typed_actions"]),
            "operand_bytes": {row["role"] + ":" + row["access"]: row["bytes"] for row in work["operands"]},
            **{name: work[name] for name in ("conditional_action_counts", "source_program", "arithmetic_groups", "unpriced_typed_actions")},
            "source_role_read_write_bytes": traffic["l2_cache"], "baseline_read_write_bytes": traffic["dram"], "logical_bytes": compulsory}
        actual[key]["demand"] = {name: owner.demand_per_resource[name] for name in work["demand"]}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual else
            [name for name in fields if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "layer": key[1], "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None,
            "status": "mismatched" if differences else "matched"})
    return {"scope": "genuine ordinary Triton metadata execution and fully checked active CUDA hash math/IO program",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("value-dependent conditional counts remain bounds; no real logit/expert assignment is inferred",
            "source expression extents and minimal reduction trees, not compiler instructions/physical traffic",
            "index/address issue, PDL service and actual provider remain unpriced/unbound"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
