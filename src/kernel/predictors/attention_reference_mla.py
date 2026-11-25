"""Typed mathematical work for the reference projected-input attention paths.

This owner exports work, never empirical durations.  It does not relabel a
BF16 FA3 program as FP8.  Dense MLA follows SGLang's homogeneous-FP8 SM90
FlashMLA extension.  Other families retain their distinct cache formats and
explicitly caller-declared split scenarios.  The caller must price tensor
work with ``tensor_dtype`` rather than treating every matrix operation as
BF16.  Operand bytes are cold logical role traffic, not observed HBM bytes.
"""


_FAMILIES = ("MLA", "DSA", "CSA", "HCA", "GLM-5.3-Flash DSA")
_STAGES = ("q_fp8_quant", "attention_core", "combine")
_ROLES = ("compute_tensor", "compute_vector", "compute_reduction",
          "compute_fma", "compute_sfu", "compute_alu")
_DENSE_SOURCE = (
    "SGLang v0.5.19 kernels/aot/cmake/flashmla.cmake pinned FlashMLA; "
    "csrc/extension/sm90/dense_fp8/{dense_fp8_python_api.cpp,"
    "flash_fwd_mla_metadata.cu,flash_fwd_mla_kernel.h}"
)
_SPARSE_SOURCE = (
    "SGLang v0.5.19 kernels/aot/cmake/flashmla.cmake pinned FlashMLA; "
    "csrc/sm90/decode/sparse_fp8/{config.h,splitkv_mla.cuh}; "
    "reference non-duration KernelTemplate ModelType0/1 identities"
)


def _positive_integer(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def dense_mla_source_split_count(context, query_tokens, sm_count):
    """Exact batch-one/Hq64/Hkv1 SM90 homogeneous-FP8 metadata rule.

    The pinned API uses M=N=64 and fixed_overhead_num_blocks=5.  Its
    metadata kernel gives each worker ceil((KV_blocks+5)/SM_parts) useful
    KV blocks for a single request; the final worker closes the request.
    This derives *active splits*, not allocation slots or a GPU-count guess.
    No invocation timing or profiler duration participates.
    """
    for name, value in (("context", context), ("query_tokens", query_tokens),
                        ("sm_count", sm_count)):
        _positive_integer(name, value)
    if query_tokens not in (1, 4) or context < query_tokens:
        raise ValueError("reference MLA supports T1/T4 with context >= queries")
    parts = sm_count // query_tokens
    if parts < 1:
        raise ValueError("SM count cannot host one packed query/head tile")
    blocks = _ceil_div(context, 64)
    useful_blocks_per_worker = _ceil_div(blocks + 5, parts)
    return _ceil_div(blocks, useful_blocks_per_worker)


def _geometry(family, context, query_tokens):
    positions = tuple(range(context - query_tokens + 1, context + 1))
    if family == "MLA":
        return 576, 512, 576, 1, positions, positions, "fp8"
    if family == "DSA":
        selected = tuple(min(position, 2048) for position in positions)
        return 576, 512, 656, 2, selected, selected, "bf16"
    if family == "CSA":
        selected = tuple(min(position, 128) + min(position // 4, 512)
                         for position in positions)
        return 512, 512, 584, 2, selected, selected, "bf16"
    if family == "HCA":
        selected = tuple(min(position, 128) + position // 128
                         for position in positions)
        return 512, 512, 584, 2, selected, selected, "bf16"
    if family == "GLM-5.3-Flash DSA":
        # Full kpool4 groups plus the unpooled 0..3-token causal tail.
        selected = tuple(min(position // 4, 512) * 4 + position % 4
                         for position in positions)
        issued = (2112,) * query_tokens
        return 512, 512, 1024, 2, selected, issued, "bf16"
    raise ValueError(f"unknown reference family: {family!r}")


def attention_work(family, stage, context, query_tokens, split_count):
    """Return work for one reference stage at batch1, Hq64, T1/T4.

    ``context`` is the final visible KV extent; T4 uses bottom-right causal
    positions N-3..N. ``split_count`` is an active-split fact or an explicit
    scenario, never inferred from the reference's eight-GPU title. For
    families other than dense MLA, no native split-selector claim is made.
    ``combine`` is the internal real split reducer, not the reference's
    empty outer combine wrapper. S=1 writes final O directly and has no
    split scratch/reducer mathematical work.
    """
    if family not in _FAMILIES or stage not in _STAGES:
        raise ValueError("unknown reference attention family or stage")
    for name, value in (("context", context), ("query_tokens", query_tokens),
                        ("split_count", split_count)):
        _positive_integer(name, value)
    if query_tokens not in (1, 4) or context < query_tokens:
        raise ValueError("reference work requires T1/T4 and nonempty causal rows")
    if stage == "q_fp8_quant" and family != "MLA":
        raise ValueError("this static query quantization belongs only to MLA")
    qk_dim, value_dim, record_bytes, query_bytes, selected, issued, tensor_dtype = (
        _geometry(family, context, query_tokens))
    if split_count > min(selected):
        raise ValueError("active splits cannot exceed the least-populated query row")
    heads, rows = 64, 64 * query_tokens
    demand = {role: 0.0 for role in _ROLES}
    operands = []

    def operand(role, byte_count, access):
        operands.append({"role": role, "bytes": int(byte_count), "access": access})

    limitations = [
        "mathematical/source-role work, not native instruction scheduling or measured HBM traffic",
        "query tiles read their own selected cache rows; no unproven cross-query cache reuse credited",
        "cold logical reads/writes; transaction expansion, cache hits, occupancy, launch and physical scratch residency unbound",
        "softmax comparisons and dtype conversions reported but no invented instruction throughput assigned",
    ]
    if family != "MLA":
        limitations.append(
            "split_count is an explicit active-split scenario; native family selector/provider is not bound")
    if family in ("CSA", "HCA"):
        limitations.append(
            "reference Kimi compressed-attention geometry; no claim that the vLLM DeepSeek-V4 provider is identical")
    if family == "GLM-5.3-Flash DSA":
        limitations.append(
            "TileLang tensor work retains2112 padded selected slots; useful softmax omits masked padding; compiler scalar issue remains unbound")
        if split_count != 1:
            limitations.append(
                "split TileLang is only a mathematical scenario, not an asserted native split implementation")
    result = {
        "family": family, "stage": stage, "demand": demand,
        "operands": operands, "tensor_dtype": tensor_dtype,
        "source": (_DENSE_SOURCE if family == "MLA" else
                   _SPARSE_SOURCE if family in ("DSA", "CSA", "HCA") else
                   "reference kpool4v1 TileLang BF16 latent512, selected2048+tail, padded2112; "
                   "mathematical softmax and logical operand traffic"),
        "limitations": limitations,
        "shape": {"batch": 1, "query_tokens": query_tokens,
                  "context": context, "query_heads": heads,
                  "qk_head_dim": qk_dim, "v_head_dim": value_dim,
                  "cache_record_bytes": record_bytes, "split_count": split_count},
        "selected_records_per_query": selected,
        "tensor_issued_records_per_query": issued,
        "split_selection": ("caller supplied; may use dense_mla_source_split_count"
                            if family == "MLA" else "explicit scenario; native selector unbound"),
        "max_comparisons": 0, "dtype_conversions": 0,
    }
    if stage == "q_fp8_quant":
        elements = rows * qk_dim
        operand("prepared_query_bf16", elements * 2, "read")
        operand("static_query_scale_fp32", 4, "read")
        operand("quantized_query_fp8", elements, "write")
        demand["compute_fma"] = float(elements)
        result["max_comparisons"] = 2 * elements
        result["dtype_conversions"] = elements
        result["source"] = (
            "reference trace static sglang::per_tensor_quant_fp8_kernel<bf16,e4m3>; "
            "SGLang per_tensor_quant_fp8.py is_static=True; input scale precomputed")
        limitations.append("static scale supplied; no dynamic absmax reduction is implied")
        return result

    partial_rows = rows * split_count
    if stage == "attention_core":
        valid_pairs = sum(selected) * heads
        issued_pairs = sum(issued) * heads
        # Each active split owns a separate CTA query panel.  Cache chunks
        # partition the selected records, but Q is reloaded by every split.
        operand("prepared_query_split_role_reads",
                split_count * rows * qk_dim * query_bytes, "read")
        operand("selected_shared_cache_role_reads", sum(selected) * record_bytes, "read")
        if family != "MLA":
            operand("selected_indices_int32", sum(selected) * 4, "read")
        else:
            operand("query_and_cache_descales_fp32",
                    8 * query_tokens * split_count, "read")
        # QK and PV are separate matrix products. Repeated K/V roles share
        # the same latent cache record, rather than charging two caches.
        demand["compute_tensor"] = float(2 * issued_pairs * (qk_dim + value_dim))
        demand["compute_fma"] = float(4 * valid_pairs)
        demand["compute_sfu"] = float(valid_pairs + 2 * partial_rows)
        result["max_comparisons"] = valid_pairs - partial_rows
        result["valid_score_pairs"] = valid_pairs
        if family in ("DSA", "CSA", "HCA"):
            fp8_values = 512 if family == "DSA" else 448
            converted = sum(selected) * fp8_values
            demand["compute_fma"] += float(converted)
            result["dtype_conversions"] = converted
            limitations.append(
                "mixed cache scaled FP8 fields dequantized once per selected-record role; exact compiler conversion issue unbound")
        if split_count == 1:
            operand("attention_output_bf16", rows * value_dim * 2, "write")
            if family != "GLM-5.3-Flash DSA":
                operand("attention_lse_fp32", rows * 4, "write")
        else:
            operand("partial_output_fp32", partial_rows * value_dim * 4, "write")
            operand("partial_lse_fp32", partial_rows * 4, "write")
        return result

    if split_count == 1:
        limitations.append("single active split: mathematical reducer is identity; dense source kernel returns before data loads")
        return result
    operand("partial_output_fp32", partial_rows * value_dim * 4, "read")
    operand("partial_lse_fp32", partial_rows * 4, "read")
    operand("attention_output_bf16", rows * value_dim * 2, "write")
    operand("attention_lse_fp32", rows * 4, "write")
    # Pinned dense reducer computes global logsumexp, then exp(lse-global)
    # weights and an FP32 weighted partial-output sum. No reciprocal guess.
    demand["compute_fma"] = float(rows * (3 * split_count + 1 + 2 * split_count * value_dim))
    demand["compute_sfu"] = float(rows * (2 * split_count + 1))
    result["max_comparisons"] = rows * (split_count - 1)
    result["dtype_conversions"] = rows * value_dim
    result["source"] = (_DENSE_SOURCE + "; flash_fwd_splitkv_mla_combine_kernel"
                        if family == "MLA" else
                        "mathematical FP32 logsumexp-weighted split merge; native family reducer unbound")
    return result
