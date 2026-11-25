"""Closed projected-input Attention source programs, without native launch claims.

Operator geometries follow the reference's discrete kernel signatures and
typed storage, rather than its schematic labels or measured durations.
"""

from src.kernel.kernel_spec import KernelKind, KernelSpec


VARIANT = "projected_attention_path"
PROGRAM = "sglang_0_5_19_projected_attention_source_v1"
FAMILIES = ("MLA", "DSA", "CSA", "HCA", "KDA", "GLM-5.3-Flash DSA")


def stages_for(family):
    if family == "KDA":
        return ("convolution", "recurrent", "output_norm")
    if family == "MLA":
        return ("q_fp8_quant", "attention_core", "combine")
    if family == "DSA":
        return ("indexer_q_prepare", "indexer_logits", "topk_page_transform",
                "attention_core", "combine")
    if family == "CSA":
        return ("swa_kv_quant_store", "compressor_update_norm_rope",
                "compressed_kv_quant_store", "compressed_index_cache_store",
                "indexer_q_prepare", "indexer_logits", "topk_page_transform",
                "attention_core", "combine")
    if family == "HCA":
        return ("swa_kv_quant_store", "compressor_update_norm_rope",
                "compressed_kv_quant_store", "attention_core", "combine")
    if family == "GLM-5.3-Flash DSA":
        return ("indexer_q_prepare", "kpool_cache_update", "indexer_logits",
                "topk_page_transform", "index_pad_for_tilelang", "attention_core", "combine")
    raise ValueError("unknown projected-input Attention family")


def stage_kind(family, stage):
    if stage not in stages_for(family):
        raise ValueError("stage is absent from the declared source path")
    if stage == "attention_core":
        return KernelKind.FLASH_ATTN if family == "MLA" else KernelKind.SPARSE_ATTN
    if stage in ("indexer_logits", "topk_page_transform"):
        return KernelKind.SPARSE_ATTN_INDEXER
    return KernelKind.VECTOR


def validate_shape(shape):
    fields = {"batch", "q_len", "kv_len", "head_dim", "num_query_heads",
              "num_kv_heads", "dtype_bytes"}
    if set(shape) != fields:
        raise ValueError("projected Attention requires its exact physical shape fields")
    if (any(type(value) is not int or value <= 0 for value in shape.values())
            or shape["batch"] != 1 or shape["q_len"] not in (1, 4)
            or shape["kv_len"] < shape["q_len"] or shape["num_query_heads"] != 64
            or shape["num_kv_heads"] != 1 or shape["dtype_bytes"] not in (2, 4)):
        raise ValueError("projected Attention requires BS1/H64, T1/T4 and BF16 input geometry")


def normalize_attributes(kind, attributes, shape, dtype):
    fields = {"family", "stage", "source_program", "split_count", "selector_passes"}
    if set(attributes) != fields:
        raise ValueError("projected Attention requires explicit complete source attributes")
    from src.kernel.predictors.attention_verify import VERIFY_PROGRAM, ONLINE_HCA_PROGRAM, normalize_verify_attributes
    if attributes["source_program"] in (VERIFY_PROGRAM, ONLINE_HCA_PROGRAM):
        return normalize_verify_attributes(kind, attributes, shape, dtype)
    family, stage = attributes["family"], attributes["stage"]
    expected_dtype = "int32" if kind is KernelKind.SPARSE_ATTN_INDEXER else "bf16"
    if (family not in FAMILIES or attributes["source_program"] != PROGRAM
            or dtype != expected_dtype or stage_kind(family, stage) is not kind):
        raise ValueError("projected Attention family/stage/kind/source mismatch")
    expected_dim = 128 if family == "KDA" else 576 if family in ("MLA", "DSA") else 512
    if shape["head_dim"] != expected_dim:
        raise ValueError("projected Attention must preserve the traced native head dimension")
    if family in ("CSA", "HCA", "KDA") and shape["q_len"] != 1:
        raise NotImplementedError("reference compressed complete path supports T1 only")
    if (type(attributes["split_count"]) is not int or not 1 <= attributes["split_count"] <= 160
            or type(attributes["selector_passes"]) is not int
            or not 1 <= attributes["selector_passes"] <= 4):
        raise ValueError("projected Attention requires explicit bounded source split/selector scenarios")
    return dict(attributes)


def stage_spec(family, stage, context, query_tokens, *, hardware_id, split_count, selector_passes):
    kind = stage_kind(family, stage)
    return KernelSpec(
        kind=kind, hardware_id=hardware_id,
        dtype="int32" if kind is KernelKind.SPARSE_ATTN_INDEXER else "bf16",
        variant=VARIANT, execution_support="unsupported",
        unsupported_reason="launch_selector_unavailable",
        shape={"batch": 1, "q_len": query_tokens, "kv_len": context,
               "num_query_heads": 64, "num_kv_heads": 1,
               "head_dim": 128 if family == "KDA" else 576 if family in ("MLA", "DSA") else 512},
        attributes={"family": family, "stage": stage, "source_program": PROGRAM,
                    "split_count": split_count, "selector_passes": selector_passes})


def source_work(kernel):
    if kernel.variant != VARIANT or kernel.execution_support != "unsupported" or kernel.launch is not None:
        raise ValueError("projected Attention work requires an unbound source spec")
    from src.kernel.predictors.attention_verify import VERIFY_PROGRAM, ONLINE_HCA_PROGRAM, verify_source_work
    if kernel.attributes["source_program"] in (VERIFY_PROGRAM, ONLINE_HCA_PROGRAM):
        return verify_source_work(kernel)
    family, stage = kernel.attributes["family"], kernel.attributes["stage"]
    context, query_tokens = kernel.shape["kv_len"], kernel.shape["q_len"]
    if family == "KDA":
        from src.kernel.predictors.vector import kimi_k3_kda_decode_logical_work
        spec = KernelSpec(kind=KernelKind.VECTOR, hardware_id=kernel.hardware_id, dtype="bf16",
            variant="kimi_k3_kda_decode", execution_support="unsupported",
            unsupported_reason="launch_selector_unavailable",
            shape={"element_count": 8192, "rows": 1, "heads": 64, "head_dim": 128},
            attributes={"stage": stage, "state_indices": (1,), "state_pool_slots": 2,
                        "gate_lower_bound": -5.0})
        work = kimi_k3_kda_decode_logical_work(spec)
        # IS_KDA/USE_LOWER_BOUND/USE_QK_L2NORM_IN_KERNEL retain the same
        # bounded-gate equations and BV32 source decomposition. SGLang's
        # wrapper writes output directly; there is no packed-Kimi copy.
        return {**work, "source": [
            "sglang/v0.5.19/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py",
            work["source"]], "tensor_dtype": "bf16", "limitations": [*work["limitations"],
                "Explicit bounded gate -5 and in-kernel QK L2 normalization source scenario; reference flags not observed",
                "Recurrence grid4x1x64, one warp/three stages is source geometry only; no occupancy/issue price",
                "Active SGLang state slot0 is valid; Kimi positive/null slot convention is not an allocation requirement here",
                "No extra output copy: SGLang recurrence output is the gated-norm input"]}
    if stage in ("attention_core", "combine", "q_fp8_quant"):
        from src.kernel.predictors.attention_reference_mla import attention_work
        return attention_work(family, stage, context, query_tokens, kernel.attributes["split_count"])
    if stage in ("swa_kv_quant_store", "compressor_update_norm_rope",
                 "compressed_kv_quant_store", "compressed_index_cache_store"):
        from src.kernel.predictors.attention_reference_compressed import compressed_work
        return compressed_work(family, stage, context, query_tokens)
    from src.kernel.predictors.attention_reference_indexer import indexer_work
    return indexer_work(family, stage, context, query_tokens, kernel.attributes["selector_passes"])


def source_prediction(kernel, hardware):
    from src.kernel.predictors.common import prediction_from_ns, roofline_ns
    work = source_work(kernel)
    steps = work["substeps"] if "substeps" in work else (work,)
    total_ns = 0
    for step in steps:
        demand = dict(step["demand"])
        if demand["compute_tensor"]:
            dtype = step["tensor_dtype"] if "tensor_dtype" in step else (
                "fp8" if kernel.attributes["stage"] == "indexer_logits" else "bf16")
            # Convert work into primary tensor-rate equivalents using only
            # HWSpec's independently declared dtype throughput ratio.
            demand["compute_tensor"] *= (hardware.resource_by_role("compute_tensor").capacity
                / hardware.resource_by_role("compute_tensor_" + dtype).capacity)
        logical_bytes = sum(operand["bytes"] for operand in step["operands"])
        total_ns += roofline_ns(demand, hardware, logical_bytes)
    return prediction_from_ns(total_ns,
                              "analytical_projected_attention")
