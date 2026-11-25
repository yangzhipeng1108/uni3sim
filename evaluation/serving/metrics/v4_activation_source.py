"""FFN activation/add mathematical accounts from independent source.

This reuses the existing scalar/packed C++ source reference. No numerical
tensor, issue rate, backend equivalence or production duration is needed.
"""

import re

from evaluation.serving.simulation.v4_flash_reference import (
    clamped_activation_arithmetic_reference, clamped_activation_source_reference,
)


class FFNVectorSource:
    def __init__(self, source, rank, work):
        self.rank, self.work = rank, work
        self.source_binding = clamped_activation_source_reference(source.root)

    def activation(self, layer, role, rows, width, dtype, limit):
        if dtype != "bf16" or type(limit) not in (float, int) or limit <= 0:
            raise ValueError("FFN activation reference requires positive clamp and BF16 operands")
        elements = rows * width
        self.work.append({"rank": self.rank, "layer": layer, "role": role,
            "elements": elements, "dtype": dtype, "clamp_limit": limit, "alpha": 1.0, "beta": 0.0,
            "read_bytes": 4 * elements, "write_bytes": 2 * elements,
            "activation_result_rounding_dtype": dtype, "rounding_before_up_multiply": True,
            **clamped_activation_arithmetic_reference(elements, dtype)})

    def addition(self, layer, left, right):
        if left.shape != right.shape or left.dtype != "bf16" or right.dtype != "bf16":
            raise ValueError("source shared/routed sum requires matching BF16 tensors")
        elements = left.numel()
        self.work.append({"rank": self.rank, "layer": layer, "role": "ffn_add",
            "elements": elements, "dtype": "bf16", "read_bytes": 4 * elements, "write_bytes": 2 * elements,
            "operation": "add", "reads": 2, "writes": 1,
            "baseline_demand": {"compute_fma": elements, "compute_alu": 0, "compute_sfu": 0}})


def compare_v4_ffn_vector_work(reference, plan, hardware):
    """Compare actual owner inputs with source math and logical payload.

    Routed activation uses its stage account inside the fused expert call;
    this cannot accept the remaining fused kernel or private workspace.
    """
    from src.kernel.kernel_base import create_predictor
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.predictors.fused_moe import dsv4_marlin_moe_source_work
    from src.kernel.predictors.vector import clamped_silu_logical_work, dense_bf16_source_logical_work
    from src.serving.kernel_lowering.kernel_plan import hardware_for_call

    expected = {}
    for row in reference["ffn_vector_work"]:
        key = row["rank"], row["layer"], row["role"]
        if key in expected:
            raise ValueError("FFN vector reference repeats an invocation")
        fields = ("elements", "dtype", "read_bytes", "write_bytes", "baseline_demand")
        if row["role"] != "ffn_add":
            fields += ("clamp_limit", "alpha", "beta", "mathematical_primitives", "source_negation_programs",
                "activation_result_rounding_dtype", "rounding_before_up_multiply")
        else:
            fields += ("operation", "reads", "writes")
        expected[key] = {name: row[name] for name in fields}
    actual = {}
    for call in plan.calls:
        matches = [re.search(r"(?:^|:)layer-(\d+):ffn:(shared:silu-mul|shared-add|routed)$", name)
            for name in call.source_node_ids]
        matches = [match for match in matches if match is not None]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError("FFN vector comparison cannot collapse source invocations")
        match, = matches
        role = {"shared:silu-mul": "ffn_shared_activation", "shared-add": "ffn_add", "routed": "ffn_routed_activation"}[match[2]]
        key = call.rank, int(match[1]), role
        if key in actual:
            raise ValueError("FFN vector comparison found duplicate lowered calls")
        kernel = call.kernel_spec
        if role == "ffn_routed_activation":
            if kernel.kind is not KernelKind.FUSED_MOE:
                actual[key] = {"invalid": "routed activation requires fused expert owner"}
                continue
            work = dsv4_marlin_moe_source_work(kernel)
            elements = kernel.shape["tokens"] * kernel.shape["topk"] * kernel.shape["intermediate"]
            activation, = (stage for stage in work["roofline_stage_templates"] if stage["stage"] == "activation")
            semantics = work["semantic_fields"]
            actual[key] = {"elements": elements, "dtype": kernel.dtype,
                "read_bytes": activation["nonweight_read_write_bytes"]["read"],
                "write_bytes": activation["nonweight_read_write_bytes"]["write"],
                "baseline_demand": work["activation_resource_groups"],
                "clamp_limit": semantics["clamp_limit"], "alpha": semantics["alpha"], "beta": semantics["beta"],
                **{name: semantics[name] for name in ("activation_result_rounding_dtype", "rounding_before_up_multiply")},
                "mathematical_primitives": {name: work["mathematical_primitives"]["activation_" + name] for name in
                    ("fp32_multiply", "fp32_add", "fp32_divide", "fp32_exp", "sign_negate", "fp32_minmax")},
                "source_negation_programs": work["activation_negation_programs"]}
        else:
            if (kernel.kind is not KernelKind.VECTOR or
                    role == "ffn_shared_activation" and (kernel.variant != "silu" or kernel.attributes["swiglu_limit"] is None) or
                    role == "ffn_add" and kernel.variant not in ("elementwise", "elementwise_pytorch_vec4")):
                actual[key] = {"invalid": "unexpected activation/add operation contract"}
                continue
            if role == "ffn_shared_activation":
                work = clamped_silu_logical_work(kernel)
                traffic = {access: sum(operand["bytes"] for operand in work["operands"]
                    if operand["access"] == access) for access in ("read", "write")}
                actual[key] = {"elements": work["elements"], "dtype": kernel.dtype,
                    "clamp_limit": kernel.attributes["swiglu_limit"], "alpha": work["alpha"], "beta": work["beta"],
                    **{name: work[name] for name in ("activation_result_rounding_dtype", "rounding_before_up_multiply")},
                    "read_bytes": traffic["read"], "write_bytes": traffic["write"], "baseline_demand": work["baseline_demand"],
                    "mathematical_primitives": work["mathematical_primitives"], "source_negation_programs": work["source_negation_programs"]}
            else:
                if ("mathematical_program" in kernel.attributes
                        and kernel.attributes["mathematical_program"] == "dense_bf16_source_flops_bytes_v1"):
                    work = dense_bf16_source_logical_work(kernel)
                    traffic = {access: sum(operand["bytes"] for operand in work["operands"]
                        if operand["access"] == access) for access in ("read", "write")}
                    demand = {name: work["demand"][name] for name in ("compute_fma", "compute_alu", "compute_sfu")}
                else:
                    owner, _ = create_predictor(kernel.kind).resource_input(kernel, hardware_for_call(call, hardware))
                    traffic = {access: sum(node.bytes for path in owner.memory_paths if path.access_kind == access
                        for node in path.nodes if node.resource_role == "l2_cache") for access in ("read", "write")}
                    demand = {name: owner.demand_per_resource[name] for name in ("compute_fma", "compute_alu", "compute_sfu")}
                actual[key] = {"elements": kernel.shape["element_count"], "dtype": kernel.dtype,
                    "read_bytes": traffic["read"], "write_bytes": traffic["write"], "baseline_demand": demand,
                    **{name: kernel.attributes[name] for name in ("operation", "reads", "writes")}}
    checks = []
    for key in sorted(set(expected) | set(actual)):
        differences = (["unexpected"] if key not in expected else ["missing"] if key not in actual
            else ["stage_contract"] if "invalid" in actual[key]
            else [name for name in expected[key] if expected[key][name] != actual[key][name]])
        checks.append({"rank": key[0], "layer": key[1], "role": key[2],
            "status": "matched" if not differences else "mismatched", "differences": differences,
            "expected": expected[key] if key in expected else None, "actual": actual[key] if key in actual else None})
    return {"scope": "shared/routed clamped SiLU mathematical primitive counts, typed negation alternatives and logical payload; shared sum math/payload",
        "total": len(checks), "matched": sum(not row["differences"] for row in checks),
        "mismatched": sum(bool(row["differences"]) for row in checks), "not_checked": 0, "checks": checks,
        "source": reference["ffn_vector_source_binding"],
        "limitations": ("clamp comparisons/negation/casts counted but native issue services not priced",
            "vector logical payload follows production typed operand accounts; cache-adjusted DRAM is a different quantity",
            "scalar/packed form is not selected and numerical equivalence is not asserted",
            "routed activation is a component account, not whole expert padding/weights/workspace acceptance",
            "actual provider, GPU numeric data and physical traffic remain unverified"),
        "whole_model_source_alignment_accepted": False, "runtime_accepted": False}
