"""Original Q/KV norm wrapper and independent FP32 expression analysis."""

import ast
from copy import deepcopy
import re
from types import SimpleNamespace

from .qwen3_source import _Tensor
from .v4_finish_source import _Math


NORM_SOURCE_FILES = {"fused_qkv_norm.py": "vllm/models/common/ops/fused_qk_rmsnorm.py",
    "ir_layernorm.py": "vllm/ir/ops/layernorm.py", "rms_cuda_storage.py": "vllm/kernels/vllm_c.py"}


class _RMSValue(_Tensor):
    """Execute the IR reference's scalar math on shape/type metadata only."""
    def __init__(self, shape, dtype, actions, allocations, role, *, storage=None):
        super().__init__(shape, dtype, storage)
        self.actions, self.allocations, self.role = actions, allocations, role

    def result(self, shape, dtype, role, action):
        for name, count in action.items():
            if name not in self.actions:
                self.actions[name] = 0
            self.actions[name] += count
        self.allocations.append({"role": role, "shape": tuple(shape), "dtype": dtype})
        return _RMSValue(shape, dtype, self.actions, self.allocations, role)

    def to(self, dtype):
        if dtype == self.dtype:
            return self
        if (self.dtype, dtype) not in (("bf16", "fp32"), ("fp32", "bf16")):
            raise ValueError("RMS source conversion changed")
        return self.result(self.shape, dtype, "input_fp32" if dtype == "fp32" else "normalized_bf16",
            {self.dtype + "_to_" + dtype: self.numel()})

    def pow(self, exponent):
        if exponent != 2 or self.role != "input_fp32":
            raise ValueError("RMS source requires the squared FP32 input")
        return self.result(self.shape, self.dtype, "squared_input", {"square_multiply": self.numel()})

    def mean(self, dim, keepdim):
        if dim != -1 or not keepdim or self.role != "squared_input":
            raise ValueError("RMS source requires the row's mean square")
        rows, width = self.shape
        return self.result((rows, 1), self.dtype, "variance", {"sum_add": rows * (width - 1), "mean_divide": rows})

    def __add__(self, epsilon):
        if self.role != "variance" or type(epsilon) not in (int, float) or epsilon <= 0:
            raise ValueError("RMS source requires a scalar epsilon on the variance")
        return self.result(self.shape, self.dtype, "variance_plus_epsilon", {"epsilon_add": self.numel()})

    def rsqrt(self):
        if self.role != "variance_plus_epsilon":
            raise ValueError("RMS source requires rsqrt of the adjusted variance")
        return self.result(self.shape, self.dtype, "inverse_rms", {"rsqrt": self.numel()})

    def __mul__(self, other):
        if self.role == "input_fp32" and other.role == "inverse_rms" and other.shape == (self.shape[0], 1):
            role, action = "normalized_fp32", "normalize_multiply"
        elif self.role == "normalized_bf16" and other.role == "weight" and other.shape == (self.shape[-1],):
            role, action = "output", "gamma_multiply"
        else:
            raise ValueError("RMS source multiply/rounding program changed")
        if self.dtype != other.dtype:
            raise ValueError("RMS source multiply types disagree")
        return self.result(self.shape, self.dtype, role, {action: self.numel()})


class FinalRMSNormSource:
    def __init__(self, source, rows):
        self.source, self.rows = source, rows
        self.function = source.function("ir_layernorm.py", "rms_norm", metadata_globals={
            "torch": SimpleNamespace(float32="fp32", rsqrt=lambda value: value.rsqrt())})

    def run(self, rank, value, width, epsilon):
        if value.dtype != "bf16" or len(value.shape) != 2 or value.shape[-1] != width:
            raise ValueError("final RMSNorm requires its dense BF16 source width")
        actions, allocations = {}, []
        x = _RMSValue(value.shape, value.dtype, actions, allocations, "input", storage=value.storage)
        weight = _RMSValue((width,), "bf16", actions, allocations, "weight")
        output = self.function(x, weight, epsilon)
        if output.role != "output" or output.storage in (x.storage, weight.storage):
            raise ValueError("final RMS source must return the independent gamma-multiply output")
        conversions = {name: actions.pop(name) for name in ("bf16_to_fp32", "fp32_to_bf16")}
        self.rows.append({"rank": rank, "epsilon": epsilon, "mathematical_primitives": actions,
            "typed_conversions": conversions, "reference_allocations": allocations,
            "source_input_layout": {"shape": value.shape, "dtype": value.dtype,
                "strides_elements": value.strides, "offset_elements": value.storage_offset},
            "operands": [("bf16", value.numel(), "read"), ("bf16", width, "read"),
                ("bf16", output.numel(), "write")],
            "source": {"path": str(self.source.paths["ir_layernorm.py"]), "symbol": "rms_norm",
                "line": next(node for node in self.source.trees["ir_layernorm.py"].body
                    if isinstance(node, ast.FunctionDef) and node.name == "rms_norm").lineno}})
        return output


def compare_v4_final_norm_work(reference, plan):
    from src.kernel.predictors.vector import rmsnorm_logical_work

    expected = {row["rank"]: row for row in reference["final_norm_work"]}
    if len(expected) != len(reference["final_norm_work"]):
        raise ValueError("source repeats final RMSNorm for a rank")
    actual = {}
    for call in plan.calls:
        if not any(name == "final-norm" or name.endswith(":final-norm") for name in call.source_node_ids):
            continue
        if call.rank in actual:
            raise ValueError("actual repeats final RMSNorm for a rank")
        work = rmsnorm_logical_work(call.kernel_spec)
        actual[call.rank] = {"epsilon": work["epsilon"], "mathematical_primitives": work["mathematical_primitives"],
            "operands": [(item["dtype"], item["elements"], item["access"]) for item in work["operands"]]}
    fields = ("epsilon", "mathematical_primitives", "operands")
    checks = []
    for rank in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if rank not in expected else ["missing"] if rank not in actual else
            [name for name in fields if expected[rank][name] != actual[rank][name]])
        checks.append({"rank": rank, "differences": differences, "status": "mismatched" if differences else "matched",
            "expected": expected[rank] if rank in expected else None, "actual": actual[rank] if rank in actual else None})
    return {"scope": "original IR RMS reference's ideal math/typed logical IO, with its own rounding and allocation requests",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("IR reference is not selected as the actual H200 implementation",
            "CUDA retains FP32 before gamma; batch-invariant uses sqrt/divide: numerical equivalence is not asserted",
            "reference allocations are not assumed materialized by compiled providers or a capacity bound"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


class QKVNormSource:
    def __init__(self, source, rows):
        self.source, self.rows, self.active = source, rows, None
        self.kernel = source.method("FusedQKVRMSNormKernel", "kernel")
        allowed = {"pid_task", "launch_pdl"}
        for node in ast.walk(self.kernel):
            if isinstance(node, ast.If) and not {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)} <= allowed:
                raise ValueError("Q/KV norm math requires an undeclared value branch")
        self.expressions = {}
        for name in ("variance", "rrms", "y"):
            assignment, = (node for node in self.kernel.body if isinstance(node, ast.Assign)
                and any(ast.unparse(target) == name for target in node.targets))
            self.expressions[name] = assignment.value
        if {name: ast.unparse(value) for name, value in self.expressions.items()} != {
                "variance": "tl.sum(x * x, axis=0) / SIZE", "rrms": "tl.rsqrt(variance + eps)", "y": "x * rrms * w"}:
            raise ValueError("Q/KV norm source must retain FP32 weighted RMS arithmetic")
        for name in ("x", "w"):
            assignment, = (node for node in self.kernel.body if isinstance(node, ast.Assign)
                and any(ast.unparse(target) == name for target in node.targets))
            if not ast.unparse(assignment.value).endswith(".to(tl.float32)"):
                raise ValueError("Q/KV norm must widen both BF16 value and weight to FP32")
        store, = (node for node in ast.walk(self.kernel) if isinstance(node, ast.Call) and ast.unparse(node.func) == "tl.store")
        if ast.unparse(store.args[1]) != "y.to(row_out.dtype.element_ty)":
            raise ValueError("Q/KV norm must retain the single final output conversion")
        self.host = source.function("fused_qkv_norm.py", "fused_q_kv_rmsnorm", metadata_globals={
            "torch": SimpleNamespace(empty=lambda shape, *, dtype, device: _Tensor(shape, dtype)),
            "_FUSED_Q_KV_RMSNORM_KERNEL": self.device})

    def device(self, qr, kv, q_out, kv_out, q_weight, kv_weight, epsilon):
        if self.active is None:
            raise ValueError("Q/KV norm device port has no original wrapper")
        rank, layer = self.active
        tokens = qr.shape[0]
        counts = []
        for value, weight, output in ((qr, q_weight, q_out), (kv, kv_weight, kv_out)):
            if (value.dtype != "bf16" or weight.dtype != "bf16" or output.dtype != "bf16"
                    or weight.shape != (value.shape[1],) or output.shape != value.shape
                    or output.storage is value.storage or not output.is_contiguous()):
                raise ValueError("Q/KV norm wrapper requires independent dense BF16 outputs and weights")
            math = _Math({"x": (value.shape[1],), "w": (value.shape[1],), "SIZE": (), "eps": ()})
            for name in ("variance", "rrms", "y"):
                math.values[name] = math.evaluate(deepcopy(self.expressions[name]))
            counts.append(math.counts)
        elements = sum(value.numel() for value in (qr, kv))
        reductions = tokens * (qr.shape[1] + kv.shape[1] - 2)
        self.rows.append({"rank": rank, "layer": layer, "epsilon": epsilon,
            "mathematical_primitives": {"square_multiply": elements, "sum_add": reductions,
                "mean_divide": 2 * tokens, "epsilon_add": 2 * tokens, "rsqrt": 2 * tokens,
                "normalize_multiply": elements, "gamma_multiply": elements},
            "typed_actions": {"bf16_to_fp32": 2 * elements, "fp32_to_bf16": elements},
            "demand": {"compute_tensor": 0, "compute_vector": 0, "compute_reduction": reductions,
                "compute_fma": tokens * sum(count["compute_fma"] for count in counts) - reductions,
                "compute_sfu": tokens * sum(count["compute_sfu"] for count in counts), "compute_alu": 0},
            "read_bytes": 2 * elements + 2 * (qr.shape[1] + kv.shape[1]), "write_bytes": 2 * elements,
            "source": {"path": str(self.source.paths["fused_qkv_norm.py"]), "symbol": "FusedQKVRMSNormKernel.kernel",
                "line": self.kernel.lineno, "expressions": {name: ast.unparse(value) for name, value in self.expressions.items()}}})

    def run(self, rank, layer, qr, kv, q_weight, kv_weight, epsilon):
        if self.active is not None:
            raise ValueError("Q/KV norm reference does not allow overlapping calls")
        self.active = rank, layer
        try:
            return self.host(qr, kv, q_weight, kv_weight, epsilon)
        finally:
            self.active = None


def compare_v4_qkv_norm_work(reference, plan, hardware):
    from src.kernel.kernel_base import create_predictor
    from src.kernel.predictors.vector import dsv4_qkv_rmsnorm_logical_work
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    names = ("epsilon", "mathematical_primitives", "typed_actions", "demand", "read_bytes", "write_bytes")
    expected, actual, sources = {}, {}, {}
    for row in reference["qkv_norm_work"]:
        key = row["rank"], row["layer"]
        if key in expected:
            raise ValueError("source repeats a fused norm invocation")
        expected[key], sources[key] = {name: row[name] for name in names}, row["source"]
    for call in plan.calls:
        if call.kernel_spec.variant != "dsv4_q_kv_rmsnorm":
            continue
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:q-kv-rmsnorm(?::\d+)?$", name) for name in call.source_node_ids]
        match, = (match for match in matches if match is not None)
        key = call.rank, int(match[1])
        if key in actual:
            raise ValueError("actual repeats a fused norm invocation")
        work = dsv4_qkv_rmsnorm_logical_work(call.kernel_spec)
        actual[key] = {name: work[name] for name in names}
        owner, _ = create_predictor(call.kernel_spec.kind).resource_input(call.kernel_spec, hardware_for_call(call, hardware))
        actual[key]["demand"] = {name: owner.demand_per_resource[name] for name in work["demand"]}
        for access in ("read", "write"):
            actual[key][access + "_bytes"] = sum(node.bytes for path in owner.memory_paths if path.access_kind == access
                for node in path.nodes if node.resource_role == "l2_cache")
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual else
            [name for name in names if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "layer": key[1], "differences": differences,
            "status": "mismatched" if differences else "matched", "expected": expected[key] if key in expected else None,
            "actual": actual[key] if key in actual else None, "source": sources[key] if key in sources else None})
    return {"scope": "original Q/KV norm wrapper, live FP32 math and typed logical IO",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("minimal live reductions, not padded/shuffle/compiler issue", "logical unique IO, not per-row gamma reload or physical traffic",
            "conversion issue, numerical values, provider and complete storage lifetimes remain unaccepted"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
