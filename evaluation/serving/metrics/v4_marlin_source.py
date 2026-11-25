"""Original Marlin host program on payload-free cache views.

The custom-op leaves describe source math/dataflow, not a loaded provider.
Routing values, conditional FP32 scratch traffic and GPU issue remain open.
"""

import json
from math import ceil, prod
import re
from types import SimpleNamespace

from .qwen3_source import _Tensor
from ..simulation.v4_flash_reference import _cpp_anchor, _marlin_moe_reference


MARLIN_SOURCE_FILES = {
    "marlin_moe.py": "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
    "marlin_utils.py": "vllm/model_executor/layers/quantization/utils/marlin_utils.py",
    "moe_utils.py": "vllm/model_executor/layers/fused_moe/utils.py",
    "modular_kernel.py": "vllm/model_executor/layers/fused_moe/modular_kernel.py",
}


def marlin_projection(work):
    """Shared comparison representation; no prediction or expected formulas."""
    return json.loads(json.dumps({name: work[name] for name in (
        "mathematical_primitives", "operands", "semantic_fields", "source_views",
        "router_weight_source_program", "weight_payload_per_active_expert",
        "source_stage_order", "roofline_stage_templates", "modular_buffer_program")}))


def _typed_source_bindings(root):
    """Retain the C++ expressions supporting typed leaves, not ISA claims."""
    bindings = []
    for relative, symbol, expressions in (
            ("csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h", "auto write_result =", (
                "c_scalar_t2topk_weight_score", "if(mul_topk_weights)topk_weight_score=sh_block_topk_weights[row]",
                "sh_red_half2[a]=__hmul2(sh_red_half2[a],topk_weight_score)",
                "C[true_idx]=*reinterpret_cast<int4*>(sh_red_half2)")),
            ("csrc/libtorch_stable/moe/moe_align_sum_kernels.cu", "__global__ void moe_sum_vec_kernel(", (
                "floatacc[VEC]", "acc[j]=0.f", "k<TOPK", "acc[j]+=static_cast<float>(packed.val[j])")),
            ("csrc/libtorch_stable/moe/moe_align_sum_kernels.cu", "__global__ void moe_sum_vec_dynamic_kernel(", (
                "floatacc[VEC]", "acc[j]=0.f", "k<topk", "acc[j]+=static_cast<float>(packed.val[j])")),
            ("csrc/libtorch_stable/moe/moe_align_sum_kernels.cu", "__global__ void moe_sum_scalar_kernel(", (
                "floatx=0.f", "k<topk", "x+=static_cast<float>(VLLM_LDG("))):
        anchor = _cpp_anchor(root, relative, symbol)
        compact = re.sub(r"\s+", "", anchor["source"])
        if any(expression not in compact for expression in expressions):
            raise ValueError("Marlin typed weight/reduction source changed: " + symbol)
        bindings.append({name: anchor[name] for name in ("path", "line", "symbol")})
    relative = "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
    text = (root / relative).read_text()
    for expression in ("float topk_weight_tmp = topk_weights_ptr[idx]",
            "Cdtype::num2num2(Cdtype::float2num(topk_weight_tmp))"):
        if expression not in text:
            raise ValueError("Marlin FP32 to output-dtype routing weight conversion changed")
        bindings.append({"path": str(root / relative), "line": text[:text.index(expression)].count("\n") + 1,
            "expression": expression})
    return bindings


class _Cache(_Tensor):
    """A contiguous view that supports the original shrink-cache idiom."""
    def view(self, *shape):
        value = super().view(*shape)
        return _Cache(value.shape, value.dtype, value.storage,
            strides=value.strides, storage_offset=value.storage_offset)

    def flatten(self, start_dim=0):
        if start_dim != 0 or not self.is_contiguous():
            raise ValueError("Marlin cache flatten requires contiguous full storage view")
        return self.view(self.numel())

    def __getitem__(self, item):
        if self.ndim != 1 or not isinstance(item, slice):
            raise ValueError("Marlin cache shrink requires a flat prefix slice")
        start, stop, step = item.indices(self.numel())
        if start != 0 or step != 1 or stop <= 0:
            raise ValueError("Marlin cache shrink requires a nonempty unit-step prefix")
        return _Cache((stop,), self.dtype, self.storage, storage_offset=self.storage_offset)


class MarlinSource:
    def __init__(self, source, raw, tp):
        self.source, self.raw, self.tp, self.active = source, raw, tp, None
        self.bindings = _typed_source_bindings(source.root)
        self.quant = object()
        scalar_types = SimpleNamespace(**{name: object() for name in
            ("uint4", "uint8b128", "uint4b8", "float8_e4m3fn")}, float4_e2m1f=self.quant)
        self.activation_enum = SimpleNamespace(is_gated=True)
        torch = SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32", int8="int8",
            float8_e4m3fn="fp8", empty_like=self.empty_like)
        globals = {"torch": torch, "math": SimpleNamespace(ceil=ceil),
            "MoEActivation": SimpleNamespace(SILU=self.activation_enum),
            "ops": SimpleNamespace(moe_wna16_marlin_gemm=self.gemm, moe_sum=self.reduce),
            "ScalarType": SimpleNamespace(from_id=self.quant_from_id), "scalar_types": scalar_types,
            "marlin_make_workspace_new": self.locks,
            "marlin_moe_intermediate_size": source.function("marlin_utils.py", "marlin_moe_intermediate_size"),
            "_resize_cache": source.function("moe_utils.py", "_resize_cache", metadata_globals={"prod": prod}),
            "moe_align_block_size": self.align}
        inner = source.function("marlin_moe.py", "_fused_marlin_moe", metadata_globals=globals)
        outer = source.function("marlin_moe.py", "fused_marlin_moe", metadata_globals={**globals, "_fused_marlin_moe": inner})
        self.apply = source.forward("MarlinExperts", "apply", metadata_globals={"fused_marlin_moe": outer})
        self.workspace_shapes = source.forward("MarlinExperts", "workspace_shapes")
        self.moe_sum = source.forward("MarlinExperts", "moe_sum", metadata_globals=globals)
        self.apply_activation = source.forward("FusedMoEExpertsModular", "activation",
            metadata_globals={"apply_moe_activation": self.activation})

    def quant_from_id(self, identity):
        if identity != "declared_e2m1":
            raise ValueError("Marlin source requires declared E2M1 prepared weights")
        return self.quant

    def empty_like(self, tensor):
        raise ValueError("Marlin source unexpectedly allocates a replacement caller output")

    def locks(self, device, blocks):
        if device != "reference_metadata_only" or blocks != 4 or self.active["locks"] is not None:
            raise ValueError("Marlin source lock request changed")
        # The original helper's SM-dependent allocation is separately bound;
        # do not invent an SM count or represent its request as memory traffic.
        self.active["locks"] = object()
        self.active["order"].append("workspace_zero")
        return self.active["locks"]

    def align(self, ids, block, experts, expert_map, *, ignore_invalid_experts):
        active = self.active
        if (ids is not active["ids"] or experts != self.raw["n_routed_experts"]
                or expert_map is not None or not ignore_invalid_experts
                or block != active["reference"]["routing_alignment"]["block_size_m"]):
            raise ValueError("Marlin source alignment arguments changed")
        active["order"].append("alignment")
        alignment = active["reference"]["routing_alignment"]
        active["aligned"] = tuple(_Tensor((size,), "int32") for size in
            (alignment["sorted_ids_capacity"], alignment["expert_ids_capacity"], 1))
        return active["aligned"]

    def gemm(self, x, out, weight, bias, scales, a_scales, global_scale, zeros, locks,
             sorted_ids, expert_ids, post_pad, weights, **options):
        a = self.active
        stage = "w13" if weight is a["w1"] else "w2_with_router_weight" if weight is a["w2"] else None
        if stage is None:
            raise ValueError("Marlin source uses an unknown prepared matrix")
        first = stage == "w13"
        tokens, hidden, width, topk = a["tokens"], self.raw["hidden_size"], self.raw["moe_intermediate_size"] // self.tp, self.raw["num_experts_per_tok"]
        expected_options = {"moe_block_size": a["reference"]["routing_alignment"]["block_size_m"],
            "top_k": topk if first else 1, "mul_topk_weights": not first, "b_q_type": self.quant,
            "size_m": tokens if first else tokens * topk, "size_n": 2 * width if first else hidden,
            "size_k": hidden if first else width, "use_atomic_add": False, "use_fp32_reduce": True, "is_zp_float": False}
        expected_input = a["input"] if first else a["activated"]
        shape = (tokens * topk, 2 * width if first else hidden)
        if (options != expected_options or x is not expected_input or out.shape != shape or out.dtype != "bf16"
                or out.storage is not a["workspace2"].storage or out.storage_offset != a["workspace2"].storage_offset
                or scales is not a["scales1" if first else "scales2"] or weights is not a["weights"]
                or (sorted_ids, expert_ids, post_pad) != a["aligned"] or locks is not a["locks"]
                or any(value is not None for value in (bias, a_scales, global_scale, zeros))):
            raise ValueError("Marlin source matrix ports, cache view or typed options changed")
        a["order"].append(stage)
        a["gate_up" if first else "weighted_output"] = out
        return out

    def activation(self, kind, out, x, *, topk_ids, expert_map, activation_config, valid_token_counts):
        a = self.active
        rows = a["tokens"] * self.raw["num_experts_per_tok"]
        width = self.raw["moe_intermediate_size"] // self.tp
        gate = a["gate_up"]
        if (kind is not self.activation_enum or x.shape != (rows, 2 * width) or out.shape != (rows, width)
                or x.storage is not gate.storage or x.storage_offset != gate.storage_offset
                or out.storage is not a["workspace13"].storage or out.storage_offset != a["workspace13"].storage_offset
                or topk_ids is not a["ids"] or expert_map is not None or (x.dtype, out.dtype) != ("bf16", "bf16")
                or activation_config is not a["activation_config"] or valid_token_counts is not None):
            raise ValueError("Marlin source activation cache ports changed")
        a["activated"] = out
        a["order"].append("activation")

    def reduce(self, x, out):
        a = self.active
        hidden, topk = self.raw["hidden_size"], self.raw["num_experts_per_tok"]
        weighted = a["weighted_output"]
        if (x.shape != (a["tokens"], topk, hidden) or x.dtype != "bf16" or out is not a["output"]
                or x.storage is not weighted.storage or x.storage_offset != weighted.storage_offset):
            raise ValueError("Marlin source top-k reduction views/output changed")
        a["order"].append("topk_reduce")

    def run(self, x, weights, ids, activation_config):
        if self.active is not None:
            raise ValueError("Marlin metadata source invocations cannot overlap")
        tokens, hidden, width, experts, topk = (x.shape[0], self.raw["hidden_size"],
            self.raw["moe_intermediate_size"] // self.tp, self.raw["n_routed_experts"], self.raw["num_experts_per_tok"])
        if (x.shape != (tokens, hidden) or x.dtype != "bf16" or not x.is_contiguous()
                or weights.shape != (tokens, topk) or weights.dtype != "fp32"
                or ids.shape != weights.shape or ids.dtype != "int32"):
            raise ValueError("Marlin source requires contiguous BF16 input, FP32 weights and INT32 IDs")
        reference = _marlin_moe_reference(self.raw, tokens, self.tp)
        workspace_shapes = self.workspace_shapes(None, tokens, width, hidden, topk, experts, experts, None, self.activation_enum)
        ws13, ws2, output_shape = workspace_shapes
        requested = reference["modular_buffer_program"]["requests"]
        if (2 * prod(ws13) != requested[0]["bytes"] or 2 * prod(ws2) != requested[1]["bytes"]
                or output_shape != (tokens, hidden)):
            raise ValueError("Marlin source workspace/output request shapes changed")
        # Distinct logical ports: the caller arena's common backing/byte offsets
        # are validated separately; this execution tests the original swap.
        output = _Tensor(output_shape, "bf16")
        a = {"tokens": tokens, "reference": reference, "input": x, "weights": weights, "ids": ids,
            "w1": _Tensor((experts, hidden // 16, 4 * width), "int32"),
            "w2": _Tensor((experts, width // 16, 2 * hidden), "int32"),
            "scales1": _Tensor((experts, hidden // 32, 2 * width), "fp8_e8m0"),
            "scales2": _Tensor((experts, width // 32, hidden), "fp8_e8m0"),
            "workspace13": _Cache(ws13), "workspace2": _Cache(ws2), "output": output,
            "order": [], "locks": None, "activation_config": activation_config}
        expert = SimpleNamespace(w1_scale=a["scales1"], w2_scale=a["scales2"], w1_bias=None, w2_bias=None,
            g1_alphas=None, g2_alphas=None, a1_gscale=None, a2_gscale=None, w1_zp=None, w2_zp=None,
            quant_type_id="declared_e2m1", input_dtype=None, _lora_context=None, activation_config=activation_config)
        expert.activation = self.apply_activation.__get__(expert)
        expert.moe_sum = self.moe_sum.__get__(expert)
        self.active = a
        try:
            result = self.apply(expert, output, x, a["w1"], a["w2"], weights, ids, self.activation_enum,
                experts, None, None, None, a["workspace13"], a["workspace2"], None, False)
            if result is not None or a["order"] != ["alignment", "workspace_zero", "w13", "activation", "w2_with_router_weight", "topk_reduce"]:
                raise ValueError("Marlin source producer/consumer stage order changed")
            return output, marlin_projection(reference)
        finally:
            self.active = None
