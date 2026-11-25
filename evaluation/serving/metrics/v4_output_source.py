"""Independent output-projection source work on payload-free metadata."""

import ast
from copy import deepcopy
import re
from types import SimpleNamespace

from .v4_finish_source import _Math
from .qwen3_source import _Tensor
from .v4_fp8_source import FP8LinearSource


OUTPUT_SOURCE_FILES = {
    "o_proj.py": "vllm/models/deepseek_v4/nvidia/ops/o_proj.py",
    "inv_rope.py": "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
    "linear.py": "vllm/model_executor/layers/linear.py",
}


class _QuantConstants(ast.NodeTransformer):
    """Only the known constexpr reciprocal/negative FP8 range is folded."""
    def visit_BinOp(self, node):
        if (isinstance(node.op, ast.Div) and isinstance(node.left, ast.Constant) and node.left.value == 1.0
                and isinstance(node.right, ast.Name) and node.right.id == "fp8_max"):
            return ast.copy_location(ast.Constant(1.0 / 448.0), node)
        return self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Name) and node.operand.id == "fp8_max":
            return ast.copy_location(ast.Constant(-448.0), node)
        return self.generic_visit(node)


class InverseRopeMathSource:
    def __init__(self, source):
        self.function = source.method("FusedInvRopeFP8QuantKernel", "kernel")
        self.path = str(source.paths["inv_rope.py"])
        allowed = {"launch_pdl", "pid_token", "num_tokens", "QUANTIZE", "TMA_ALIGNED_SCALES"}
        for node in ast.walk(self.function):
            if isinstance(node, ast.If):
                if not {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)} <= allowed:
                    raise ValueError("inverse-RoPE math requires an undeclared value-dependent branch")
        x_assignments = [node for node in ast.walk(self.function) if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == "x" for target in node.targets)]
        if len(x_assignments) != 2 or ast.unparse(x_assignments[1].value) != "tl.where(is_rope, rotated, x)":
            raise ValueError("inverse-RoPE source value flow has extra or changed arithmetic")

    def expression(self, target):
        assignments = [node for node in ast.walk(self.function) if isinstance(node, ast.Assign)
            and any(ast.unparse(value) == target for value in node.targets)]
        if len(assignments) != 1:
            raise ValueError("inverse-RoPE source mathematical assignment is ambiguous/missing: " + target)
        node, = assignments
        return deepcopy(node.value), {"path": self.path, "line": node.lineno,
            "target": target, "expression": ast.unparse(node.value)}

    def analyze(self, tokens, heads, head_dim, rope_dim):
        if (any(type(value) is not int or value <= 0 for value in (tokens, heads, head_dim, rope_dim))
                or head_dim % 128 or rope_dim % 2 or rope_dim > head_dim):
            raise ValueError("inverse-RoPE source requires positive dimensions and complete group128/even RoPE")
        chunks = head_dim // 128
        bindings = []
        rotation = 0
        eager = 0
        for target, operation in (("x_add", ast.Add), ("x_sub", ast.Sub)):
            expression, binding = self.expression(target)
            if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, operation):
                raise ValueError("inverse rotation must preserve its source add/subtract alternatives")
            if ast.unparse(expression) != "x * cos_v " + ("+" if target == "x_add" else "-") + " x_partner * sin_v":
                raise ValueError("inverse rotation trigonometric input flow changed")
            useful = _Math({name: (rope_dim // 2,) for name in ("x", "x_partner", "cos_v", "sin_v")})
            useful.evaluate(expression)
            rotation += useful.counts["compute_fma"]
            complete = _Math({name: (head_dim,) for name in ("x", "x_partner", "cos_v", "sin_v")})
            complete.evaluate(expression)
            eager += complete.counts["compute_fma"]
            bindings.append(binding)
        math = _Math({"x": (head_dim,), "eps": (), "fp8_max": ()}, parameters={
            "CHUNKS_PER_HEAD": chunks, "QUANT_GROUP_SIZE": 128, "HEAD_DIM": head_dim})
        for target in ("x_2d", "block_absmax", "scale_raw", "scales", "scales_exp", "x_quant"):
            expression, binding = self.expression(target)
            if target == "scales_exp" and {node.id for node in ast.walk(expression) if isinstance(node, ast.Name)} != {
                    "tl", "scales", "CHUNKS_PER_HEAD", "QUANT_GROUP_SIZE", "HEAD_DIM"}:
                raise ValueError("quantizer broadcast must consume the rounded source scale")
            if target == "x_quant":
                if (not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Attribute)
                        or expression.func.attr != "to" or len(expression.args) != 1
                        or ast.unparse(expression.args[0]) != "tl.float8e4nv"):
                    raise ValueError("inverse-RoPE quantized result must retain its FP8 conversion")
                expression = expression.func.value
                if ast.unparse(expression) != "tl.clamp(x / scales_exp, -fp8_max, fp8_max)":
                    raise ValueError("inverse-RoPE quantization must divide by its rounded scale before the FP8 clamp")
            expression = _QuantConstants().visit(expression)
            math.values[target] = math.evaluate(expression)
            bindings.append(binding)
        per_head = tokens * heads
        elements, groups = per_head * head_dim, per_head * chunks
        scale_backing = -(-tokens // 4) * 4 * heads * chunks
        return {"elements": elements, "scale_groups": groups,
            "demand": {"compute_tensor": 0, "compute_vector": 0, "compute_reduction": 0,
                "compute_fma": per_head * (rotation + math.counts["compute_fma"]),
                "compute_sfu": per_head * math.counts["compute_sfu"], "compute_alu": 0},
            "typed_actions": {"fp32_abs": per_head * math.counts["typed_absolute"],
                "fp32_amax_and_epsilon_max": per_head * math.counts["typed_maximum"],
                "scale_ceil": per_head * math.counts["typed_ceil"],
                "fp32_clamp_comparisons": 2 * per_head * math.counts["typed_clamp"], "fp32_to_fp8": elements},
            "read_bytes": 2 * elements + 8 * tokens + 4 * tokens * rope_dim,
            "write_bytes": elements + 4 * scale_backing,
            "scale_zero_padding_write_bytes": 4 * (scale_backing - groups),
            "source_eager_rotation_expression_flops": per_head * eager,
            "source_eager_rotation_issue_priced": False, "source_expressions": bindings,
            "scope": "useful selected rotation, complete live FP32 quantization and padded scale stores; compiler CSE/issue and reloads unbound"}


class _UnboundPDL:
    def __bool__(self):
        raise ValueError("output reference requires an undeclared PDL runtime decision")


class OutputProjectionSource:
    """Original SM90 o_proj and inverse-RoPE wrapper/kernel-host geometry.

    Only the quantizer and FP8 einsum device bodies are metadata ports.
    They do not select an actual provider or create numerical tensors.
    """
    def __init__(self, source, calls, rank, tp, matrix_work, inverse_work, fp8_work):
        self.source, self.calls, self.rank, self.tp = source, calls, rank, tp
        self.matrix_work, self.inverse_work, self.fp8_work = matrix_work, inverse_work, fp8_work
        self.math = InverseRopeMathSource(source)
        self.linear_source = FP8LinearSource(source)
        platform = SimpleNamespace(get_device_capability=lambda: SimpleNamespace(major=9),
            is_arch_support_pdl=_UnboundPDL)
        self.recipe = source.function("o_proj.py", "compute_fp8_einsum_recipe", metadata_globals={"current_platform": platform})()
        if self.recipe != ((1, 128, 128), False):
            raise ValueError("output reference requires the declared SM90 FP32-scale recipe")
        torch = SimpleNamespace(float32="fp32", int32="int32", float8_e4m3fn="fp8", bfloat16="bf16",
            empty=self.empty, finfo=lambda dtype: SimpleNamespace(max=448.0))
        self.host = source.function("inv_rope.py", "_fused_inv_rope_fp8_quant_kernel_impl", metadata_globals={
            "torch": torch, "current_platform": platform, "_FUSED_INV_ROPE_FP8_QUANT_KERNEL": self.quantizer})
        torch.ops = SimpleNamespace(vllm=SimpleNamespace(fused_inv_rope_fp8_quant_kernel=self.host))
        self.inverse = source.function("inv_rope.py", "fused_inv_rope_fp8_quant", metadata_globals={"torch": torch},
            metadata_imports={(0, "vllm.utils.deep_gemm", "get_tma_aligned_size"): lambda size, alignment: -(-size // alignment) * alignment})
        self.raw = source.function("o_proj.py", "deep_gemm_fp8_o_proj", metadata_globals={"torch": torch,
            "fused_inv_rope_fp8_quant": lambda *args, **kwargs: self.inverse_call(*args, **kwargs),
            "fp8_einsum": self.einsum})

    @staticmethod
    def empty(*shape, dtype, device=None):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _Tensor(shape, dtype)

    def quantizer(self, o, positions, cache, values, scales, tokens, *, heads_per_group, quant_group_size,
                  chunks_per_head, nope_dim, half_rope, tma_aligned_scales, quantize, fp8_max, launch_pdl, grid):
        if (quant_group_size != 128 or chunks_per_head * quant_group_size != o.shape[-1]
                or nope_dim + 2 * half_rope != o.shape[-1] or tma_aligned_scales or not quantize or fp8_max != 448.0):
            raise ValueError("inverse-RoPE metadata kernel requires declared FP8/FP32 SM90 geometry")
        groups = o.shape[1] // heads_per_group
        width = heads_per_group * o.shape[2]
        aligned = -(-tokens // 4) * 4
        if (grid != (aligned, o.shape[1]) or values.shape != (groups, tokens, width) or values.dtype != "fp8"
                or scales.shape != (groups, tokens, width // 128) or scales.dtype != "fp32"
                or scales.strides != (width // 128 * aligned, 1, aligned)):
            raise ValueError("inverse-RoPE source buffer layout or padding launch changed")
        work = self.math.analyze(tokens, o.shape[1], o.shape[2], 2 * half_rope)
        self.inverse_work.append({"rank": self.rank, "layer": self.layer_index, **work,
            "source_input_layout": {"shape": o.shape, "dtype": o.dtype, "strides_elements": o.strides},
            "source_buffer_layouts": {name: {"shape": tensor.shape, "dtype": tensor.dtype, "strides_elements": tensor.strides}
                for name, tensor in (("values", values), ("scales", scales))}})

    def inverse_call(self, o, positions, cache, **kwargs):
        return self.calls.bind(self.rank, "inverse_rope", lambda o, pos, table: self.inverse(o, pos, table, **kwargs))(o, positions, cache)

    def einsum(self, equation, activation, weights, output, *, recipe):
        if equation != "bhr,hdr->bhd" or recipe != self.recipe[0]:
            raise ValueError("source grouped output einsum/recipe changed")
        values, scales = activation
        weight, weight_scales = weights
        def run(values, scales):
            tokens, groups, width = values.shape
            if (weight.shape != (groups, output.shape[-1], width) or weight.dtype != "fp8"
                    or values.dtype != "fp8" or scales.dtype != "fp32" or weight_scales.dtype != "fp32"):
                raise ValueError("source grouped output requires FP8 values/weight and FP32 scales")
            self.matrix_work.append({"rank": self.rank, "layer": self.layer_index, "role": "grouped_output_projection",
                "shape": {"b": groups, "m": tokens, "n": output.shape[-1], "k": width},
                "tensor_flops": 2 * tokens * groups * output.shape[-1] * width,
                "batch_layout": "einsum_bhr_hdr_bhd", "a_batch_stride_elements": values.strides[1],
                "b_batch_stride_elements": weight.strides[0], "c_batch_stride_elements": output.strides[1],
                "output_dtype": output.dtype, "output_bytes": output.numel() * 2})
            return output
        self.calls.bind(self.rank, "grouped_output_projection", run)(values, scales)

    def layer(self, layer, fields, rotary):
        rank, calls = self.rank, self.calls
        self.layer_index = layer
        groups = fields.value(fields.expression("self.n_local_groups"))
        heads = fields.value(fields.expression("self.n_local_heads"))
        dim = fields.value(fields.expression("self.head_dim"))
        latent = fields.value(fields.expression("self.o_lora_rank"))
        args, kwargs, _ = fields.call("self.wo_a", "ColumnParallelLinear")
        width, columns = args
        if fields.keyword(kwargs, "bias") or fields.keyword(kwargs, "return_bias") or columns // self.tp != groups * latent:
            raise ValueError("source grouped wo_a constructor differs from the TP projection")
        wo_a = SimpleNamespace(weight=_Tensor((groups, latent, width), "fp8"),
            weight_scale_inv=_Tensor((groups, -(-latent // 128), width // 128), "fp32"))
        b_args, b_kwargs, _ = fields.call("self.wo_b", "RowParallelLinear")
        b_in, b_out = b_args
        if fields.keyword(b_kwargs, "bias") or fields.keyword(b_kwargs, "return_bias") or b_in // self.tp != groups * latent:
            raise ValueError("source wo_b must consume the flattened local grouped result")
        reduce = calls.bind(rank, "output_all_reduce", lambda x: _Tensor(x.shape, x.dtype))
        def project(x):
            if x.shape[-1] != b_in // self.tp or x.dtype != "bf16":
                raise ValueError("source wo_b requires BF16 grouped output")
            tokens = x.shape[0]
            self.fp8_work.append({"rank": rank, "layer": layer, "role": "output_linear_projection",
                **self.linear_source.analyze(x, b_out)})
            self.matrix_work.append({"rank": rank, "layer": layer, "role": "output_linear_projection",
                "shape": {"b": 1, "m": tokens, "n": b_out, "k": b_in // self.tp},
                "tensor_flops": 2 * tokens * b_out * (b_in // self.tp), "output_dtype": "bf16", "output_bytes": 2 * tokens * b_out})
            return _Tensor((tokens, b_out)), None
        b_call = calls.bind(rank, "output_linear_projection", project)
        constructor = self.source.method("RowParallelLinear", "__init__")
        pairs = (*zip(constructor.args.args[-len(constructor.args.defaults):], constructor.args.defaults, strict=True),
            *zip(constructor.args.kwonlyargs, constructor.args.kw_defaults, strict=True))
        defaults = {name.arg: ast.literal_eval(value) for name, value in pairs if value is not None
            if name.arg in ("input_is_parallel", "skip_bias_add", "return_bias", "reduce_results")}
        for name in tuple(defaults):
            if name in b_kwargs:
                defaults[name] = fields.keyword(b_kwargs, name)
        row_module = SimpleNamespace(**defaults, tp_size=self.tp, tp_rank=rank, bias=None,
            quant_method=SimpleNamespace(apply=lambda owner, x, bias: b_call(x)[0]))
        row_forward = self.source.forward("RowParallelLinear", metadata_globals={
            "tensor_model_parallel_all_reduce": reduce}).__get__(row_module)
        def forward(o, positions):
            self.layer_index = layer
            return self.raw(o, positions, rotary, wo_a, row_forward, n_groups=groups, heads_per_group=heads // groups,
                nope_dim=fields.value(fields.expression("self.nope_head_dim")),
                rope_dim=fields.value(fields.expression("self.rope_head_dim")), o_lora_rank=latent,
                einsum_recipe=self.recipe[0], tma_aligned_scales=self.recipe[1])
        return calls.bind(rank, "output_projection", forward)


def compare_v4_output_work(reference, plan, hardware):
    """Source expression/constructor work against actual owner accounts."""
    from src.kernel.kernel_base import create_predictor
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.vector import dsv4_inverse_rope_quant_logical_work
    from src.kernel.predictors.gemm_block_fp8 import block_fp8_gemm_roofline_work
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    expected = {}
    for row in reference["inverse_rope_work"]:
        key = row["rank"], row["layer"], "inverse_rope"
        if key in expected:
            raise ValueError("inverse-RoPE reference repeats a source invocation")
        expected[key] = {name: row[name] for name in
            ("elements", "scale_groups", "demand", "typed_actions", "read_bytes", "write_bytes", "scale_zero_padding_write_bytes")}
    for row in reference["output_matrix_work"]:
        names = ("shape", "tensor_flops", "output_dtype", "output_bytes")
        if row["role"] == "grouped_output_projection":
            names += ("batch_layout", "a_batch_stride_elements", "b_batch_stride_elements", "c_batch_stride_elements")
        key = row["rank"], row["layer"], row["role"]
        if key in expected:
            raise ValueError("output projection reference repeats a source invocation")
        expected[key] = {name: row[name] for name in names}
    actual = {}
    roles = {"inverse-rope": "inverse_rope", "wo-a": "grouped_output_projection", "wo-b": "output_linear_projection"}
    for call in plan.calls:
        matches = [re.search(r"(?:^|:)layer-(\d+):attn:out:(inverse-rope|wo-a|wo-b)$", name) for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("output comparison cannot collapse distinct source operators")
        match, = matches
        key, kernel = (call.rank, int(match[1]), roles[match[2]]), call.kernel_spec
        if key[2] == "inverse_rope":
            if key in actual:
                raise ValueError("actual inverse-RoPE has duplicate calls")
            work = dsv4_inverse_rope_quant_logical_work(kernel)
            owner, _ = create_predictor(kernel.kind).resource_input(kernel, hardware_for_call(call, hardware))
            actual[key] = {name: work[name] for name in ("elements", "scale_groups", "typed_actions", "scale_zero_padding_write_bytes")}
            actual[key]["demand"] = {name: owner.demand_per_resource[name] for name in work["demand"]}
            actual[key].update({access + "_bytes": sum(node.bytes for path in owner.memory_paths if path.access_kind == access
                for node in path.nodes if node.resource_role == "l2_cache") for access in ("read", "write")})
        elif kernel.kind is KernelKind.GEMM:
            if key in actual:
                raise ValueError("output comparison found duplicate matmul calls")
            work = block_fp8_gemm_roofline_work(kernel)
            actual[key] = {"shape": {name: kernel.shape[name] for name in ("b", "m", "n", "k")},
                "tensor_flops": work["mathematical_flops"], "output_dtype": "bf16", "output_bytes": work["operand_payload_bytes"]["output"]}
            if key[2] == "grouped_output_projection":
                actual[key].update({name: kernel.attributes[name] for name in
                    ("batch_layout", "a_batch_stride_elements", "b_batch_stride_elements", "c_batch_stride_elements")})
                if work["activation_quantizer"] is not None:
                    actual[key]["unexpected_quantizer"] = True
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual
            else [name for name in expected[key] if expected[key][name] != actual[key][name]])
        if key in actual and "unexpected_quantizer" in actual[key]:
            differences.append("unexpected_quantizer")
        checks.append({"rank": key[0], "layer": key[1], "role": key[2], "differences": differences,
            "status": "matched" if not differences else "mismatched", "expected": expected[key] if key in expected else None,
            "actual": actual[key] if key in actual else None, "source": reference["source_bindings"][key[2]] if key in expected else None})
    return {"scope": "inverse-RoPE source useful math/typed actions/payload and grouped/row FP8 matrix geometry, layout and 2BMNK",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "limitations": ("source eager/masked rotation issue, CSE and native services unbound", "declared FP32-scale SM90 source, not actual loaded weight/provider ABI",
            "rotary cache numerical initialization, numerical matmul and physical cache/allocator workspace unverified"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
