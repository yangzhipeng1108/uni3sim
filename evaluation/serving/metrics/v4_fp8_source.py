"""Execute the declared FP32-scale linear wrapper on metadata only.

This describes a conditional logical external-quantization program. It
does not choose a runtime class, SDK, numerical values or packed scales.
In-call quantization may share its operand frontier without sharing its
private allocation/layout or numerical implementation.
"""

import ast
from collections import namedtuple
from types import SimpleNamespace
import re
import json
from pathlib import Path

from .qwen3_source import _Tensor
from ..simulation.v4_flash_reference import (external_fp8_quantizer_reference,
    flashinfer_quantizer_source_program, fp8_quantizer_source_bindings, _flashinfer_quantizer_anchor)


FP8_SOURCE_FILES = {
    "fp8_linear.py": "vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py",
    "input_quant_fp8.py": "vllm/model_executor/layers/quantization/input_quant_fp8.py",
    "fp8_utils.py": "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
    "quant_utils.py": "vllm/model_executor/layers/quantization/utils/quant_utils.py",
}


class _BlockScale(_Tensor):
    def __init__(self, *, data, input_dim, output_dim, weight_loader):
        if (input_dim, output_dim, weight_loader) != (1, 0, None):
            raise ValueError("source weight-scale axes/loading contract changed")
        super().__init__(data.shape, data.dtype, data.storage)

    def __setitem__(self, index, value):
        if index != slice(None):
            raise ValueError("source scale initialization is no longer a full fill")
        # Only a load-time initializer; no payload is created or charged to
        # the forward's traffic. Runtime scale values remain unknown.


class _QuantLaunch:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.function(grid, *args, **kwargs)


class _UnknownSupport:
    def __bool__(self):
        raise ValueError("source quantization reached undeclared DeepGEMM runtime support")


class _MetadataCustomOpBase:
    def __init__(self, *, compile_native):
        if type(compile_native) is not bool:
            raise ValueError("metadata CustomOp construction requires a boolean compile declaration")
        # No framework dispatch/registration. The genuine subclass body runs.


class FP8LinearSource:
    def __init__(self, source):
        self.source = source
        base = source.fields("Fp8BlockScaledMMLinearKernel", {
            "act_scale_descriptor": {"static": False, "group_shape": (1, 128)}})
        _, quant_args, _ = base.call("self.quant_fp8", "QuantFP8")
        self.static = base.keyword(quant_args, "static")
        self.group_shape = base.keyword(quant_args, "group_shape")
        self.use_ue8m0 = base.keyword(quant_args, "use_ue8m0")
        self.use_triton = base.value(base.expression("self.use_triton"))
        quant_flags = [node.value for node in source.classes["Fp8BlockScaledMMLinearKernel"].body
            if isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "apply_input_quant"]
        if (self.static is not False or self.group_shape != (1, 128) or self.use_ue8m0 is not False
                or len(quant_flags) != 1 or ast.literal_eval(quant_flags[0]) is not True):
            raise ValueError("source FP8 wrapper no longer matches dynamic unpacked group128 quantization")
        group_type = type("ReferenceQuantGroup", (namedtuple("QuantGroupFields", ("row", "col")),), {
            "is_per_group": source.forward("GroupShape", "is_per_group")})
        for node in source.trees["quant_utils.py"].body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                    and ast.unparse(node.targets[0].value) == "GroupShape"):
                if not isinstance(node.value, ast.Call) or ast.unparse(node.value.func) != "GroupShape":
                    raise ValueError("source quantization group constants changed")
                setattr(group_type, node.targets[0].attr, group_type(*(ast.literal_eval(arg) for arg in node.value.args)))
        quant_type = source.metadata_constructor("QuantFP8", metadata_base=_MetadataCustomOpBase, metadata_globals={
            "GroupShape": group_type, "is_deep_gemm_supported": _UnknownSupport,
            "rocm_aiter_ops": SimpleNamespace(is_linear_fp8_enabled=_UnknownSupport)})
        self.quant = quant_type(static=self.static, group_shape=group_type(*self.group_shape),
            num_token_padding=_UnknownSupport(), use_ue8m0=self.use_ue8m0)
        if self.quant.column_major_scales or self.quant.tma_aligned_scales:
            raise ValueError("source FP8 reference requires the declared row-major FP32 scales")
        self.wrapper = source.forward("Fp8BlockScaledMMLinearKernel", "apply_weights")
        self.records = []
        torch = SimpleNamespace(float32="fp32", empty=self.empty,
            finfo=lambda dtype: SimpleNamespace(min=object()))
        self.scales = source.function("fp8_utils.py", "create_fp8_scale_parameter", metadata_globals={
            "torch": torch, "BlockQuantScaleParameter": _BlockScale,
            "ChannelQuantScaleParameter": type("ExcludedChannelScale", (), {}),
            "PerTensorScaleParameter": type("ExcludedTensorScale", (), {}),
            "set_weight_attrs": lambda scale, attrs: None})
        self.quantize = source.function("fp8_utils.py", "per_token_group_quant_fp8", metadata_globals={
            "torch": torch, "current_platform": SimpleNamespace(is_cuda_alike=lambda: False, is_xpu=lambda: False),
            "get_fp8_min_max": lambda: (-448.0, 448.0),
            "triton": SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            "_per_token_group_quant_fp8": _QuantLaunch(self.quantizer)})
        self.quant_forward = source.forward("QuantFP8", "forward_cuda", metadata_globals={"_FP8_DTYPE": "fp8"},
            metadata_imports={(0, "vllm.model_executor.layers.quantization.utils", "fp8_utils"):
                SimpleNamespace(per_token_group_quant_fp8=self.quantize)})

    @staticmethod
    def empty(*shape, dtype, device=None):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return _Tensor(shape, dtype)

    def quantizer(self, grid, x, q, scales, group_size, columns, row_stride, eps, *,
                  fp8_min, fp8_max, use_ue8m0, BLOCK, num_warps, num_stages):
        if (group_size != 128 or columns != x.shape[-1] or row_stride != x.stride(0)
                or grid != (x.numel() // group_size,) or BLOCK != group_size
                or eps != 1e-10 or (fp8_min, fp8_max) != (-448.0, 448.0)
                or (num_warps, num_stages) != (1, 1) or type(use_ue8m0) is not bool):
            raise ValueError("source full-group quantization launch/operand domain changed")
        if x.dtype != "bf16" or q.dtype != "fp8" or scales.dtype != "fp32":
            raise ValueError("source quantizer requires BF16, FP8 and FP32 typed ports")
        self.records.append((x, q, scales))

    def analyze(self, x, output_columns):
        if x.dtype != "bf16" or x.ndim != 2 or x.shape[1] % 128:
            raise ValueError("source FP8 linear requires full-group BF16 rows")
        rows, columns = x.shape
        before = len(self.records)
        weight = _Tensor((output_columns, columns), "fp8")
        scales = self.scales(_BlockScale, [output_columns], columns, [128, 128], None)
        params = SimpleNamespace(weight=weight, block_scale=scales, input_scale=None, input_scale_ub=None)
        calls = []
        def matmul(*, A, B, As, Bs):
            if len(self.records) != before + 1:
                raise ValueError("source linear must have one quantization producer")
            original, q, activation_scales = self.records[-1]
            if original.storage is not x.storage or (A, B, As, Bs) != (q, weight, activation_scales, scales):
                raise ValueError("source matmul does not consume the quantizer outputs and borrowed weights")
            out = _Tensor((A.shape[0], B.shape[0]), "bf16")
            calls.append((A, B, As, Bs, out))
            return out
        layer = SimpleNamespace(config=SimpleNamespace(out_dtype="bf16"), apply_input_quant=True, use_triton=self.use_triton,
            _get_layer_params=lambda layer: params,
            quant_fp8=lambda *args, **kwargs: self.quant_forward(self.quant, *args, **kwargs),
            apply_block_scaled_mm=matmul)
        result = self.wrapper(layer, object(), x)
        if len(calls) != 1 or len(self.records) != before + 1:
            raise ValueError("source linear must execute exactly one quantizer and one matmul")
        a, b, a_scale, b_scale, c = calls[0]
        if result.storage is not c.storage or result.dtype != "bf16":
            raise ValueError("source linear output view/cast unexpectedly materializes")
        width = {"bf16": 2, "fp8": 1, "fp32": 4}
        def payload(tensor):
            return {"shape": tensor.shape, "dtype": tensor.dtype, "strides_elements": tensor.strides,
                "bytes": tensor.numel() * width[tensor.dtype]}
        work = {"quantizer": {"input": payload(x), "quantized_input": payload(a), "activation_scales": payload(a_scale)},
            "matmul": {name: payload(tensor) for name, tensor in
                (("matmul_input", a), ("weight_payload", b), ("activation_scales", a_scale), ("weight_scales", b_scale), ("output", c))},
            "quantizer_outputs_consumed_once": True, "final_view_aliases_matmul_output": True,
            "source_condition": "external dynamic group128 quantization, row-major FP32 scale payload, no bias; no runtime provider bound",
            "quantizer_arithmetic_checked": False}
        self.records[before:] = []
        return work


def _fp8_source_calls(plan):
    grouped = {}
    roles = {"gate-up": "ffn_shared_gate_up", "down": "ffn_shared_down", "wo-b": "output_linear_projection",
        "fused-wqa-wkv": "fused_qkv_projection", "wq-b": "query_projection", "indexer-wq-b": "indexer_query_projection"}
    for call in plan.calls:
        matches = [re.search(r"(?:^|:)layer-(\d+):(?:ffn:shared:(gate-up|down)|attn:out:(wo-b)|attn:(fused-wqa-wkv|wq-b|indexer-wq-b))$", name) for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("FP8 source comparison cannot merge distinct linears")
        match, = matches
        name, = (value for value in match.groups()[1:] if value is not None)
        key = call.rank, int(match[1]), roles[name]
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(call)
    return grouped


def compare_v4_fp8_dataflow(reference, plan):
    """Compare logical operand frontiers after either supported packaging.

    Only the actual side consumes production owner exports. No duration,
    provider equivalence, private workspace layout or numerical equality is
    inferred from equal logical payloads.
    """
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.gemm_block_fp8 import block_fp8_gemm_roofline_work, block_fp8_linear_logical_work
    from src.kernel.predictors.vector import fp8_group_quant_source_work

    expected = {}
    for row in (*reference["ffn_fp8_dataflow_work"], *reference["output_fp8_work"], *reference["input_fp8_work"]):
        key = row["rank"], row["layer"], row["role"]
        if key in expected:
            raise ValueError("source FP8 dataflow repeats an invocation identity")
        quant = row["quantizer"]
        expected[key] = {"quantizer_input_dtype": quant["input"]["dtype"],
            "mathematical_flops": 2 * row["matmul"]["matmul_input"]["shape"][0]
                * row["matmul"]["matmul_input"]["shape"][1] * row["matmul"]["weight_payload"]["shape"][0],
            "quantized_dtype": quant["quantized_input"]["dtype"], "scale_dtype": quant["activation_scales"]["dtype"],
            "quantizer_read_bytes": quant["input"]["bytes"],
            "quantizer_write_bytes": quant["quantized_input"]["bytes"] + quant["activation_scales"]["bytes"],
            "matmul_role_bytes": {name: operand["bytes"] for name, operand in row["matmul"].items()},
            "logical_bytes": sum(operand["bytes"] for operand in quant.values())
                + sum(operand["bytes"] for operand in row["matmul"].values()),
            "one_quantizer_feeds_matmul": row["quantizer_outputs_consumed_once"]}
    grouped = _fp8_source_calls(plan)
    actual = {}
    for key, calls in grouped.items():
        matrices = [call for call in calls if call.kernel_spec.kind is KernelKind.GEMM]
        quantizers = [call for call in calls if call.kernel_spec.kind is KernelKind.VECTOR
            and call.kernel_spec.variant == "fp8_group_quant"]
        if len(matrices) != 1:
            actual[key] = {"invalid": "requires exactly one matmul"}
            continue
        matrix, = matrices
        kernel = matrix.kernel_spec
        work = (block_fp8_linear_logical_work(kernel) if kernel.launch is None
            else block_fp8_gemm_roofline_work(kernel))
        internal = work["activation_quantizer"] if "activation_quantizer" in work else None
        if len(quantizers) + int(internal is not None) != 1 or len(calls) != 1 + len(quantizers):
            actual[key] = {"invalid": "requires one quantizer, without extra or duplicate stages"}
            continue
        matrix_bytes = {**work["operand_payload_bytes"], **work["scale_bytes"]}
        if internal is not None:
            quant = {"quantizer_input_dtype": internal["input_dtype"], "quantized_dtype": internal["quantized_dtype"],
                "scale_dtype": internal["scale_dtype"], "quantizer_read_bytes": internal["bf16_input_read_bytes"],
                "quantizer_write_bytes": internal["fp8_output_write_bytes"] + internal["scale_output_write_bytes"]}
            feeds = internal["matmul_reads_quantized_output"]
            total = work["logical_bytes"]
        else:
            producer, = quantizers
            quant_work = fp8_group_quant_source_work(producer.kernel_spec)
            objects = {operand["role"]: operand for operand in quant_work["operands"]}
            quant = {"quantizer_input_dtype": objects["input"]["dtype"],
                "quantized_dtype": objects["quantized_input"]["dtype"], "scale_dtype": objects["activation_scales"]["dtype"],
                "quantizer_read_bytes": quant_work["source_read_write_bytes"]["read"],
                "quantizer_write_bytes": quant_work["source_read_write_bytes"]["write"]}
            feeds = producer.call_id in matrix.dependencies
            total = work["logical_bytes"] + quant_work["logical_bytes"]
        actual[key] = {**quant, "mathematical_flops": work["mathematical_flops"],
            "matmul_role_bytes": matrix_bytes, "logical_bytes": total,
            "one_quantizer_feeds_matmul": feeds}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual
            else ["stage_contract"] if "invalid" in actual[key]
            else [name for name in expected[key] if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None})
    return {"scope": "source-executed FP8 wrapper/QuantFP8/full-group allocation logical operand frontier and quantizer-to-matmul edge",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "source_paths": {name: reference["source_paths"][name] for name in FP8_SOURCE_FILES},
        "limitations": ("conditional external FP32-scale source program, without selected backend/provider",
            "in-call and external forms share only the checked logical frontier; private tensor layout/alias/packing not accepted",
            "quantizer arithmetic, values, rounding, compiler traffic and library workspace unverified"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}


def fp8_quantizer_reference(reference, source_root, flashinfer_root):
    """Both packaging programs from genuine local source, no selected backend."""
    rows = []
    for row in (*reference["ffn_fp8_dataflow_work"], *reference["output_fp8_work"], *reference["input_fp8_work"]):
        tokens, columns = row["quantizer"]["input"]["shape"]
        groups = tokens * (columns // 128)
        rows.append({"rank": row["rank"], "layer": row["layer"], "role": row["role"],
            "external": external_fp8_quantizer_reference(tokens, columns),
            "in_call": {"program": flashinfer_quantizer_source_program(tokens, columns),
                "ideal_fma_bounds": [tokens * columns + groups, tokens * columns + 33 * groups],
                "stage_order": ["quantize_bf16_and_write_scales", "packed_fp8_matmul"]}})
    return {"rows": rows, "vllm_source": fp8_quantizer_source_bindings(Path(source_root)),
        "sdk_source": _flashinfer_quantizer_anchor(flashinfer_root),
        "scope": "conditional SDK in-call and CUDA/Triton external programs; scalar source counts, not actual provider or instructions"}


def compare_v4_quantizer_work(reference, plan):
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.gemm_block_fp8 import block_fp8_gemm_roofline_work
    from src.kernel.predictors.vector import fp8_group_quant_source_work

    fields = ("operands", "source_alternatives", "source_read_write_bytes", "scale_layouts", "value_program",
        "backend_conditions", "selected_backend", "selected_scale_ue8m0", "selected_tma_alignment")
    expected = {(row["rank"], row["layer"], row["role"]): row for row in reference["rows"]}
    if len(expected) != len(reference["rows"]):
        raise ValueError("quantizer reference repeats a source invocation")
    actual = {}
    for key, calls in _fp8_source_calls(plan).items():
        matrices = [call for call in calls if call.kernel_spec.kind is KernelKind.GEMM]
        quantizers = [call for call in calls if call.kernel_spec.variant == "fp8_group_quant"]
        if len(matrices) != 1:
            actual[key] = {"invalid": "requires one matrix"}
            continue
        matrix, = matrices
        if quantizers:
            if len(quantizers) != 1 or len(calls) != 2:
                actual[key] = {"invalid": "requires one external quantizer"}
                continue
            producer, = quantizers
            work = fp8_group_quant_source_work(producer.kernel_spec)
            actual[key] = {"form": "external", "program": {name: work[name] for name in fields},
                "producer_feeds_matrix": producer.call_id in matrix.dependencies}
        elif matrix.kernel_spec.launch is not None:
            work = block_fp8_gemm_roofline_work(matrix.kernel_spec)
            quant = work["activation_quantizer"]
            if quant is None or len(calls) != 1:
                actual[key] = {"invalid": "missing in-call quantizer"}
                continue
            groups = quant["scale_output_write_bytes"] // 4
            common = quant["common_fp_multiply_values"] + groups
            actual[key] = {"form": "in_call", "program": {"program": quant["source_program"],
                "ideal_fma_bounds": [common, common + 32 * groups], "stage_order": work["source_substage_order"]}}
        else:
            actual[key] = {"invalid": "quantizer source form missing"}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        want, got = None, None
        if key not in expected or key not in actual:
            differences = ["unexpected" if key not in expected else "missing"]
        elif "invalid" in actual[key]:
            differences = ["source_form"]
        else:
            got = actual[key]
            form = got["form"]
            want = {"form": form, "program": ({name: expected[key][form][name] for name in fields}
                if form == "external" else expected[key][form])}
            if form == "external":
                want["producer_feeds_matrix"] = True
            want, got = (json.loads(json.dumps(value)) for value in (want, got))
            differences = [name for name in want if want[name] != got[name]]
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": want, "actual": got})
    return {"scope": reference["scope"], "total": len(checks),
        "matched": sum(not row["differences"] for row in checks), "mismatched": sum(bool(row["differences"]) for row in checks),
        "not_checked": 0, "checks": checks,
        "limitations": ("external CUDA/Triton and rounded/unrounded programs remain conditional; no numerical equivalence asserted",
            "SDK amax-zero branch count stays unknown and its ideal arithmetic stays bounded",
            "typed comparisons/conversions/ceil, SIMD replication and actual provider/native issue services are not priced"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
