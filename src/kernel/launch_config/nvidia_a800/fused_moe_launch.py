"""NVIDIA A800 fused-MoE grouped GEMM launch selection."""

from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.selection_primitives import (
    ceil_div, fused_moe_local_expert_count, require_dtype, require_shape,
    require_variant,
    select_gpu_fused_moe_launch)
from src.kernel.launch_config import hardware_skeleton as _skeleton


def moe_alignment_capacity(tokens, topk, experts, block_size):
    """Installed vLLM 0.11 public sorted-ID allocation."""
    return tokens * topk + experts * (block_size - 1)


_SPECIALIZED_TOPK_EXPERT_COUNTS = {
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512,
}

_PYTORCH_SIGNED_INDEX_MAX = 2 ** 31 - 1
_PYTORCH_REDUCTION_ELEMENT_BYTES = 2
_PYTORCH_MOE_REDUCE_EVIDENCE_REFERENCE = (
    "microbench/nvidia_a800/references/"
    "pytorch_moe_reduce_vllm_0_11_0_torch_2_8_0.json")
_PYTORCH_MOE_REDUCE_EVIDENCE_SHA256 = (
    "33062446fa27ee81684005ffd8d82cd8dd8cf64599bbb9710ffd540bf7860747")
_A800_TUNED_CONFIG_EVIDENCE_REFERENCE = (
    "microbench/nvidia_a800/references/"
    "fused_moe_tuned_config_vllm_0_11_0_a800.json")

# vLLM 0.11.0 on the deployed A800 names its bundled config by the local
# expert count and W2 output width.  Each matching file has a nearest-M map
# with tile/warp/stage facts that the generic source default cannot represent.
# Do not silently use that default until a versioned config record and the
# compiled launch resources have been brought into this selector.
_A800_BUNDLED_TUNED_BF16_CONFIG_KEYS = frozenset({
    (64, 640),
    (64, 1280),
    (160, 192),
    (256, 64),
})


def pytorch_2_8_moe_reduce_output_element_slices(
        shape: Mapping[str, int]) -> tuple[tuple[int, int], ...]:
    """Return ``(offset, count)`` output slices emitted by ``sum_out``.

    vLLM 0.11.0 delegates top-k values outside 2/3/4 to ``at::sum_out``.
    PyTorch's reduction TensorIterator orders the contiguous BF16/FP16 input
    as ``[topk, hidden, tokens]`` and recursively bisects the dimension with
    the largest byte extent until both element count and every operand byte
    offset fit signed 32-bit indexing.  For ``topk > 1`` this partitions whole
    tokens.  For ``topk == 1`` TensorIterator first coalesces all three logical
    dimensions and therefore partitions the flattened output elements.
    """

    tokens, topk, hidden = require_shape(
        shape, ("tokens", "topk", "hidden"))
    if topk in (2, 3, 4):
        raise ValueError(
            "PyTorch moe_reduce token sharding is not selected for vLLM's "
            "specialized top-k 2/3/4 CUDA kernels")

    if topk == 1:
        # The size-one reduction dimension coalesces with hidden and then with
        # tokens because input and output are both contiguous.  The resulting
        # one-dimensional iterator has a two-byte input and output stride.
        def flat_can_use_32bit_indexing(element_count: int) -> bool:
            return (
                element_count <= _PYTORCH_SIGNED_INDEX_MAX
                and 1 + (element_count - 1) * _PYTORCH_REDUCTION_ELEMENT_BYTES
                <= _PYTORCH_SIGNED_INDEX_MAX
            )

        def split_flat(
                output_offset: int,
                output_count: int) -> tuple[tuple[int, int], ...]:
            if flat_can_use_32bit_indexing(output_count):
                return ((output_offset, output_count),)
            if output_count < 2:
                raise AssertionError(
                    "PyTorch flat reduction split requires two elements")
            first = output_count // 2
            second = output_count - first
            return (
                split_flat(output_offset, first)
                + split_flat(output_offset + first, second)
            )

        slices = split_flat(0, tokens * hidden)
        if sum(count for _, count in slices) != tokens * hidden:
            raise AssertionError(
                "PyTorch flat moe_reduce slices do not conserve output work")
        return slices

    # TensorIterator's post-reorder dimensions and byte strides for a real
    # top-k reduction.  The output is viewed as [tokens, 1, hidden] before the
    # reduction dimension is moved to the front.  These are static
    # tensor-layout facts, not profiler data.
    input_strides = (
        hidden * _PYTORCH_REDUCTION_ELEMENT_BYTES,
        _PYTORCH_REDUCTION_ELEMENT_BYTES,
        topk * hidden * _PYTORCH_REDUCTION_ELEMENT_BYTES,
    )
    output_strides = (
        0,
        _PYTORCH_REDUCTION_ELEMENT_BYTES,
        hidden * _PYTORCH_REDUCTION_ELEMENT_BYTES,
    )

    def can_use_32bit_indexing(token_count: int) -> bool:
        dimensions = (topk, hidden, token_count)
        element_count = topk * hidden * token_count
        if element_count > _PYTORCH_SIGNED_INDEX_MAX:
            return False
        for strides in (output_strides, input_strides):
            max_offset = 1 + sum(
                (extent - 1) * stride
                for extent, stride in zip(dimensions, strides)
            )
            if max_offset > _PYTORCH_SIGNED_INDEX_MAX:
                return False
        return True

    def dimension_to_split(token_count: int) -> int:
        dimensions = (topk, hidden, token_count)
        max_extent = -1
        selected = -1
        # PyTorch scans dimensions from slowest to fastest and retains the
        # first dimension on ties.
        for dimension in range(2, -1, -1):
            extent = max(
                (dimensions[dimension] - 1) * abs(strides[dimension])
                for strides in (output_strides, input_strides)
            )
            if extent > max_extent:
                max_extent = extent
                selected = dimension
        if selected < 0:
            raise AssertionError("PyTorch reduction found no split dimension")
        return selected

    def split_tokens(
            token_offset: int,
            token_count: int) -> tuple[tuple[int, int], ...]:
        if can_use_32bit_indexing(token_count):
            return ((token_offset * hidden, token_count * hidden),)
        selected = dimension_to_split(token_count)
        if selected != 2:
            raise NotImplementedError(
                "evidence_pending: PyTorch 2.8 moe_reduce requires a "
                "non-token 32-bit index "
                f"split for shape tokens={tokens}, topk={topk}, "
                f"hidden={hidden}; its partial-reduction lifecycle is not "
                "represented")
        if token_count < 2:
            raise AssertionError("token split requires at least two tokens")
        first = token_count // 2
        second = token_count - first
        return (
            split_tokens(token_offset, first)
            + split_tokens(token_offset + first, second)
        )

    slices = split_tokens(0, tokens)
    if (sum(count for _, count in slices) != tokens * hidden
            or any(offset < 0 or count <= 0 for offset, count in slices)):
        raise AssertionError(
            "PyTorch moe_reduce slices do not conserve output work")
    return slices


def select_fused_moe_reduce_work_fractions(
        shape: Mapping[str, int], *, hardware) -> tuple[float, ...]:
    """Return source-derived work fractions for physical reduce launches."""

    del hardware
    tokens, hidden = require_shape(shape, ("tokens", "hidden"))
    return tuple(
        output_count / (tokens * hidden)
        for _, output_count
        in pytorch_2_8_moe_reduce_output_element_slices(shape)
    )


def uses_specialized_topk_softmax(experts: int) -> bool:
    """Whether vLLM 0.11.0 emits its one-kernel CUDA top-k path."""

    return experts in _SPECIALIZED_TOPK_EXPERT_COUNTS


def select_fused_moe_recipe(
        shape: Mapping[str, int], *, hardware) -> tuple[str, ...]:
    tokens, experts, topk, _, _ = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    del hardware
    if not uses_specialized_topk_softmax(experts):
        raise NotImplementedError(
            "vLLM 0.11.0 topk_softmax expert family requires separate "
            "softmax and top-k launches that are not yet represented")
    if tokens * topk < 1024 and experts <= 64:
        prefix = ("moe_align_small_batch",)
    else:
        prefix = ("moe_align", "moe_count_and_sort")
    if topk in (2, 3, 4):
        reduction = ("moe_sum",)
    else:
        reduction = (
            "moe_reduce",
        ) * len(pytorch_2_8_moe_reduce_output_element_slices(shape))
    return (
        ("moe_topk",)
        + prefix
        + ("w1", "moe_act_and_mul", "w2")
        + reduction
    )


# CUDA's retained FusedMoE recipe has no typed Cast helper.
select_fused_moe_cast_roles = _skeleton.no_typed_cast_roles_selector(
    select_fused_moe_recipe)


def select_fused_moe_launch(shape: Mapping[str, int], dtype: Optional[str],
                            variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    require_dtype(dtype, ("bf16", "fp16"))
    execution_variant = require_variant(
        variant, ("w1", "w2", "routed_experts"), "FUSED_MOE")
    if execution_variant == "routed_experts":
        tokens, experts, topk, hidden, intermediate = require_shape(
            shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
        if attributes["routed_weight_dtype"] != "int4":
            raise ValueError(
                "A800 routed_experts requires symmetric INT4 weights")
        group_size = attributes["weight_group_size"]
        if group_size != 32:
            raise NotImplementedError(
                "A800 Kimi routed_experts requires INT4 group size 32")
        expert_parallel_size = attributes["expert_parallel_size"]
        if expert_parallel_size < 1 or experts % expert_parallel_size:
            raise ValueError(
                "A800 routed_experts requires experts divisible by EP")
        if expert_parallel_size == 1:
            raw_counts = attributes["tokens_per_expert"]
        else:
            raw_counts = attributes["local_tokens_per_expert"]
        if (not isinstance(raw_counts, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError(
                "A800 routed_experts routing counts must be integers")
        local_counts = tuple(raw_counts)
        if any(value < 0 for value in local_counts):
            raise ValueError(
                "A800 routed_experts routing counts cannot be negative")
        if len(local_counts) != experts:
            raise ValueError(
                "A800 routed_experts local routing count has wrong length")
        if (expert_parallel_size == 1
                and sum(local_counts) != tokens * topk):
            raise ValueError(
                "A800 routed_experts routing does not conserve rows")
        if expert_parallel_size > 1:
            owned_experts = tuple(attributes["owned_experts"])
            if any(
                    count and expert_id not in owned_experts
                    for expert_id, count in enumerate(local_counts)):
                raise ValueError(
                    "A800 routed_experts local rows include an unowned expert")

        # vLLM's pinned Marlin path scans these M blocks in priority order and
        # selects the first whose average expert capacity is below 0.9 blocks.
        block_m = 64
        for candidate in (8, 16, 32, 48, 64):
            if sum(local_counts) / (experts // expert_parallel_size) / candidate < 0.9:
                block_m = candidate
                break
        if block_m <= 16:
            tile_k, tile_n, block_threads = 128, 128, 256
        else:
            tile_k, tile_n, block_threads = 64, 256, 256
        padded_rows = sum(
            ceil_div(count, block_m) * block_m
            for count in local_counts if count > 0)
        if padded_rows <= 0:
            raise ValueError("A800 routed_experts has no active expert rows")
        populated_blocks = padded_rows // block_m
        w13_tiles = populated_blocks * ceil_div(2 * intermediate, tile_n)
        w2_tiles = populated_blocks * ceil_div(hidden, tile_n)
        return LaunchConfig(
            algorithm="vllm_marlin_moe_w4a16_aggregate",
            grid=(w13_tiles + w2_tiles, 1, 1),
            block=(block_threads, 1, 1),
            tiling=TilingSpec(tile_shape={
                "m": block_m, "n": tile_n, "k": tile_k}),
            parameters={
                # This aggregate launch models one rank's local expert shard.
                # Preserve EP because it selected the owned experts and padded
                # local rows above, and the predictor validates that contract.
                **({
                    "expert_parallel_size": float(expert_parallel_size),
                } if expert_parallel_size > 1 else {}),
                "group_size_m": 1.0,
                "padded_rows": float(padded_rows),
                "mte2_tile_bytes": 0.0,
                "weight_tile_rows": 0.0,
                "operand_replay_a": 1.0,
                "operand_replay_b": 1.0,
            },
            execution_variant=execution_variant,
        )
    del hardware
    _, experts, _, hidden, _ = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    local_experts = fused_moe_local_expert_count(experts, attributes)
    if (local_experts, hidden) in _A800_BUNDLED_TUNED_BF16_CONFIG_KEYS:
        raise NotImplementedError(
            "A800 FUSED_MOE source selects a bundled tuned config for local "
            f"experts={local_experts}, hidden={hidden}; its nearest-batch "
            "runtime resolution is not yet represented")
    # vLLM 0.11.0 forwards the BF16 configuration without ``num_warps`` to
    # Triton.  Triton's NVIDIA backend defaults that option to four warps, so
    # both grouped-GEMM projections launch 4 * 32 = 128 threads per CTA.
    return select_gpu_fused_moe_launch(
        shape, execution_variant, attributes, block_threads=128)
