"""vLLM0.11 router and EP indexing submissions on Torch2.8/SM90.

Pinned sources: fused_moe.py::vllm_topk_softmax/fused_topk and
moe_align_block_size.py. Torch v2.8.0 TensorAdvancedIndexingUtils.h converts
INT32 indices to Long; IndexKernel.cu uses launch_kernel<128,4> for this
one-dimensional map. CUDALoops.cuh and Reduce.cuh define cast, broadcast
division and row-sum geometry. No timing or another device selector is used.
"""

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import ceil_div, require_dtype, require_shape


LEGACY_ROUTER_STAGES = frozenset({
    "logits_cast", "normalize_sum", "normalize_div", "expert_ids_long", "expert_map_lookup",
})


def select_legacy_router_stage(shape, dtype, variant, attributes, hardware):
    """Return one physical helper; source metadata remains selection-only.

    normalize_div uses the semantic operation div, which needs the root-owned
    public elementwise contract extension before KernelSpec can accept it.
    The returned launch does not assert that a generic helper price describes
    this particular source program.
    """
    if hardware is not None and hardware.hardware_id != "nvidia_h200":
        raise ValueError("legacy router source requires H200 HWSpec")
    if attributes["model_execution_version"] != "vllm_0_11_0":
        raise NotImplementedError("legacy router stage binds only vLLM0.11/Torch2.8")
    stage = attributes["source_moe_router_stage"]
    if stage not in LEGACY_ROUTER_STAGES:
        raise NotImplementedError("legacy router physical stage is not bound")
    if attributes["source_input_layout"] != "contiguous_aligned_16":
        raise NotImplementedError("legacy router requires aligned contiguous source allocations")
    elements, = require_shape(shape, ("element_count",))
    integer_fields = (("index_count", "index_bytes", "gather_factor") if stage == "expert_map_lookup"
                      else ("reduction_factor",) if stage == "normalize_sum"
                      else ("reads", "writes"))
    if any(type(attributes[name]) is not int for name in integer_fields):
        raise TypeError("router helper arity, reduction and index extents must be integers")
    if stage in ("expert_ids_long", "expert_map_lookup"):
        if attributes["source_topk_id_dtype"] != "int32":
            raise NotImplementedError("legacy expert-map indexing binds INT32 expert IDs")
        if attributes["source_has_expert_map"] is not True:
            raise ValueError("expert-map stages require the explicit source map")
        if 1 + 8 * (elements - 1) > 2**31 - 1:
            raise NotImplementedError("expert-map indexing requires complete32-bit sub-iterators")
        require_dtype(dtype, ("int32",))
        if stage == "expert_ids_long":
            if (variant != "elementwise" or attributes["operation"] != "cast_int32_int64"
                    or attributes["tensor_iterator_source"] != "pytorch_2_8_sm90"):
                raise ValueError("expert-map widening requires its typed INT32-to-INT64 Copy")
            from src.kernel.launch_config.nvidia_h200.vector_launch import pytorch_int32_int64_cast_launch
            return pytorch_int32_int64_cast_launch(elements, variant)
        if variant != "gather":
            raise ValueError("expert-map lookup requires a materialized gather")
        if (attributes["index_count"] != elements or attributes["index_bytes"] != 8
                or attributes["gather_factor"] != 1 or attributes["scattered"] is not True
                or attributes["materialized_output"] is not True):
            raise ValueError("expert-map lookup requires all Long indices and materialized INT32 outputs")
        return LaunchConfig("pytorch_2_8_sm90_moe_expert_map_index", (ceil_div(elements, 512), 1, 1),
                            (128, 1, 1), execution_variant=variant)
    require_dtype(dtype, ("fp32", "float32"))
    tokens, experts, topk = require_shape(shape, ("tokens", "experts", "topk"))
    if topk > experts:
        raise ValueError("topk exceeds global experts")
    if 1 + 4 * (tokens * experts - 1) > 2**31 - 1:
        raise NotImplementedError("legacy router requires complete32-bit iterators")
    if stage == "logits_cast":
        if (variant != "elementwise" or elements != tokens * experts
                or attributes["operation"] != "cast_bf16_float32"
                or attributes["reads"] != 1 or attributes["writes"] != 1
                or attributes["routing_input_dtype"] != "bf16"):
            raise ValueError("legacy router logits cast requires all BF16 logits and FP32 outputs")
        # Copy.cu's FP32 lambda uses dynamic input conversion. CUDALoops.cuh
        # selects launch_unrolled_kernel with four values/thread, not vector8.
        return LaunchConfig("pytorch_2_8_sm90_moe_logits_float", (ceil_div(elements, 512), 1, 1),
                            (128, 1, 1), execution_variant=variant)
    if attributes["source_router_renormalize"] is not True:
        raise ValueError("selected-weight normalization requires its explicit source flag")
    if stage == "normalize_div":
        if (variant != "elementwise" or elements != tokens * topk
                or attributes["operation"] != "div"
                or attributes["reads"] != 2 or attributes["writes"] != 1):
            raise ValueError("router normalization divides all FP32 selected weights by broadcast row sums")
        # The row sum has a broadcast stride0 when topk>1. Its iterator is
        # noncontiguous despite both original allocations being contiguous.
        values_per_thread = 8 if topk == 1 else 2
        return LaunchConfig("pytorch_2_8_sm90_moe_selected_weight_div",
            (ceil_div(elements, 128 * values_per_thread), 1, 1), (128, 1, 1), execution_variant=variant)
    if variant != "reduce" or elements != tokens or attributes["reduction_factor"] != topk:
        raise ValueError("router selected-weight sum requires one complete topk row per output")
    # Contiguous FP32 [tokens,topk] -> [tokens,1]. Reduce.cuh maps block.x
    # to the fastest reduced dimension. All supported expert counts are at
    # most576, so this row reduction cannot need cross-CTA partial buffers.
    if topk <= 1 or experts > 576:
        raise NotImplementedError("router sum binds nontrivial rows with at most576 experts")
    dim0 = topk // 4 if topk > 128 else topk
    dim0_pow2 = min(1 << (dim0.bit_length() - 1), 512)
    dim1_pow2 = min(1 << (tokens.bit_length() - 1), 512)
    x = min(dim0_pow2, 32)
    y = min(dim1_pow2, 512 // x)
    x = min(dim0_pow2, 512 // y)
    reduce_y = ceil_div(topk, x) >= min(y * 16, 256)
    return LaunchConfig("pytorch_2_8_sm90_moe_selected_weight_sum",
        (ceil_div(tokens, 1 if reduce_y else y), 1, 1), (x, y, 1), execution_variant=variant)
