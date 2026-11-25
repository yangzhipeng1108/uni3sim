"""Shared data and shape primitives for hardware launch selectors."""

import math
import re
from typing import Dict, Iterable, List, Mapping, Tuple

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.fused_moe_tiling import derive_fused_moe_tiling


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def pytorch_2_8_integer_copy_launch(elements: int, variant: str, architecture: str) -> LaunchConfig:
    """Pinned dynamic-cast Int-to-Long Copy on the verified sm80/sm90 entries.

    Both separately retained cubins use128 threads, four scalar slots and32
    registers. The architecture labels identify those compiled programs;
    this is not an inference that an arbitrary CUDA architecture shares them.
    """
    if architecture not in ("sm80", "sm90"):
        raise ValueError("integer Copy requires a verified Torch2.8 architecture")
    if elements <= 0 or 8 * (elements - 1) > 2**31 - 1:
        raise NotImplementedError("integer cast requires one contiguous 32-bit-indexable iterator")
    return LaunchConfig(algorithm="pytorch_2_8_" + architecture + "_int32_to_int64",
        execution_variant=variant, grid=(ceil_div(elements, 512), 1, 1), block=(128, 1, 1),
        parameters={"registers_per_thread": 32.0, "values_per_thread": 4.0})


def pytorch_2_8_embedding_gather_domain(rows, hidden, vocab, elements, sm_count):
    """Contiguous BF16/Long source domain with at most one small-index value/thread."""
    return (rows > 0 and vocab > 0 and elements == rows * hidden
        and hidden > 0 and hidden % 128 == 0 and hidden <= sm_count * 8 * 128
        and max(2 * vocab * hidden, 2 * elements, 8 * rows) < 2**31)


def pytorch_2_8_row_gather_launch(rows, hidden, variant, architecture, *, small_index):
    """Same host geometry, with separately verified sm80/sm90 compiled entries."""
    if architecture not in ("sm80", "sm90") or type(small_index) is not bool:
        raise ValueError("row gather requires a verified architecture and explicit source branch")
    if small_index:
        if variant != "gather" or not 0 < rows <= 16:
            raise ValueError("small-index program requires at most16 embedding rows")
        return LaunchConfig(algorithm="pytorch_2_8_" + architecture + "_embedding_small_index",
            execution_variant=variant, block=(128, 1, 1), grid=(hidden // 128, 1, 1),
            parameters={"registers_per_thread": 24.0, "values_per_thread": 1.0})
    return LaunchConfig(algorithm="pytorch_2_8_" + architecture + "_vectorized_row_gather",
        execution_variant=variant, grid=(rows, ceil_div(2 * hidden, 4096), 1),
        block=(min(256, 32 * ceil_div(2 * hidden, 16 * 32)), 1, 1),
        parameters={"registers_per_thread": 24.0, "values_per_thread": 8.0})


_REGULAR_VECTOR_VARIANTS = (
    "rmsnorm", "add_rmsnorm", "elementwise", "rotary", "rope",
    "activation", "silu",
    "kv_cache", "kvcache", "reduce", "gather", "softmax",
    "gemm_reduce", "gemm_elementwise",
)

_CANN_VECTOR_VARIANTS = _REGULAR_VECTOR_VARIANTS + (
    "qkv_rmsnorm_rope",
    "moe_swiglu",
    "moe_init_routing", "moe_init_routing_v2", "moe_init_routing_v3",
    "moe_finalize_routing",
    "moe_compute_expert_tokens", "moe_transpose", "moe_cast",
    "moe_range", "moe_topk", "moe_softmax", "moe_gating",
    "moe_gating_topk", "moe_abs", "moe_token_unpermute",
)

_GPU_VECTOR_VARIANTS = _REGULAR_VECTOR_VARIANTS + (
    "elementwise_pytorch_vec4",
    "moe_topk", "moe_softmax", "moe_gating",
    "moe_act_and_mul", "moe_silu", "moe_mul", "moe_sum", "moe_align",
    "moe_align_opt_128", "moe_align_small_batch",
    "moe_count_and_sort", "moe_reduce", "moe_fill",
    "moe_routing_weight_norm", "moe_routing_weight_scale",
    "moe_routed_id_extent", "moe_expert_map_lookup", "moe_router_logit_cast",
    "moe_routed_id_widen",
    "moe_enable_if",
)


def ceil_div(value: int, divisor: int) -> int:
    for name, item in (("value", value), ("divisor", divisor)):
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"ceil_div {name} must be an integer, got {item!r}")
        if item <= 0:
            raise ValueError(f"ceil_div {name} must be positive, got {item}")
    return (value + divisor - 1) // divisor




def require_shape(shape: Mapping[str, int], names: Iterable[str]) -> Tuple[int, ...]:
    names = tuple(names)
    missing = [name for name in names if name not in shape]
    if missing:
        raise ValueError(f"missing launch shape fields: {', '.join(missing)}")
    raw_values = tuple(shape[name] for name in names)
    invalid_types = {
        name: value for name, value in zip(names, raw_values)
        if isinstance(value, bool) or not isinstance(value, int)
    }
    if invalid_types:
        raise TypeError(
            f"launch shape fields must be integers: {invalid_types}")
    values = raw_values
    if any(value <= 0 for value in values):
        raise ValueError(f"launch shape fields must be positive: {dict(zip(names, values))}")
    return values


def require_dtype(dtype: str, supported: Iterable[str]) -> str:
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("launch selection requires a non-empty dtype")
    normalized = dtype.strip().lower()
    allowed = frozenset(supported)
    if normalized not in allowed:
        raise ValueError(
            f"unsupported launch dtype {normalized!r}; expected "
            f"{sorted(allowed)}")
    return normalized


def require_variant(variant: str, supported: Iterable[str], owner: str) -> str:
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError(f"{owner} launch requires a non-empty variant")
    normalized = variant.strip().lower()
    allowed = frozenset(supported)
    if normalized not in allowed:
        raise ValueError(
            f"unsupported {owner} launch variant {normalized!r}; expected "
            f"{sorted(allowed)}")
    return normalized


def require_cann_vector_variant(variant: str) -> str:
    """Validate vector variants implemented by the CANN launch package."""

    return require_variant(variant, _CANN_VECTOR_VARIANTS, "CANN VECTOR")


def require_gpu_vector_variant(variant: str) -> str:
    """Validate vector variants implemented by GPU launch packages."""

    return require_variant(variant, _GPU_VECTOR_VARIANTS, "GPU VECTOR")


def require_mapping(value: object, context: str) -> Dict[str, object]:
    """Require a JSON object, the shape every launch record is read from."""

    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return value


def require_sequence(value: object, context: str) -> List[object]:
    """Require a JSON array, the shape every launch record list is read from."""

    if not isinstance(value, list):
        raise TypeError(f"{context} must be an array")
    return value


def require_exact_fields(
        value: Mapping[str, object], expected: set, context: str) -> None:
    """Require a record's fields to be exactly ``expected``.

    Dataset records are versioned artifacts, so a missing or extra field is a
    contract violation rather than something to default or ignore.
    """

    if set(value) != expected:
        raise ValueError(
            f"{context} fields differ: expected={sorted(expected)}, "
            f"actual={sorted(value)}")


def require_sha256(value: object, context: str) -> str:
    """Require a lowercase SHA256 hex digest."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be lowercase SHA256 hex")
    return value


def require_boolean(attributes: Mapping[str, object], name: str) -> bool:
    value = attributes[name]
    if not isinstance(value, bool):
        raise ValueError(f"launch attribute {name!r} must be boolean")
    return value


def require_non_empty_string(
        attributes: Mapping[str, object], name: str) -> str:
    value = attributes[name]
    if not isinstance(value, str):
        raise TypeError(f"launch attribute {name!r} must be a string")
    if not value:
        raise ValueError(f"launch attribute {name!r} must not be empty")
    return value


def require_positive_integer(attributes: Mapping[str, object], name: str) -> int:
    value = attributes[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"launch attribute {name!r} must be an integer")
    if value <= 0:
        raise ValueError(f"launch attribute {name!r} must be positive")
    return value


def require_non_negative_integer(
        attributes: Mapping[str, object], name: str) -> int:
    value = attributes[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"launch attribute {name!r} must be an integer")
    if value < 0:
        raise ValueError(f"launch attribute {name!r} cannot be negative")
    return value


def require_paged_kv_cache_contract(
        attributes: Mapping[str, object]) -> None:
    """Require the source-closed, unquantized paged-KV update semantics."""

    required = {
        "cache_layout",
        "cache_head_layout",
        "kv_cache_dtype",
        "cache_storage_matches_input",
        "slot_mapping_all_valid",
        "vector_copy_base_alignment_bytes",
    }
    missing = required - set(attributes)
    if missing:
        raise ValueError(
            f"KV-cache launch attributes are missing {sorted(missing)}")
    cache_layout = require_non_empty_string(attributes, "cache_layout")
    if cache_layout != "paged":
        raise NotImplementedError(
            "KV-cache launch selection supports only the paged cache layout")
    cache_head_layout = require_non_empty_string(
        attributes, "cache_head_layout")
    if cache_head_layout not in ("hnd", "nhd"):
        raise NotImplementedError(
            "KV-cache launch selection requires cache_head_layout to be "
            "'hnd' or 'nhd'")
    kv_cache_dtype = require_non_empty_string(attributes, "kv_cache_dtype")
    if kv_cache_dtype != "auto":
        raise NotImplementedError(
            "KV-cache launch selection supports only kv_cache_dtype='auto'")
    if not require_boolean(attributes, "cache_storage_matches_input"):
        raise NotImplementedError(
            "KV-cache launch selection requires cache storage to match the "
            "input dtype")
    if not require_boolean(attributes, "slot_mapping_all_valid"):
        raise NotImplementedError(
            "KV-cache traffic requires every launched slot mapping to be "
            "non-negative")
    alignment = require_positive_integer(
        attributes, "vector_copy_base_alignment_bytes")
    if alignment & (alignment - 1):
        raise ValueError(
            "KV-cache vector_copy_base_alignment_bytes must be a positive "
            "power of two")




def select_analytical_gpu_flash_attention_splits(
        work_items: int, launch_slots: int, num_n_blocks: int,
        max_splits: int = 128) -> int:
    """Apply FlashAttention 2.6.3's source-defined split-KV policy.

    The 80% early-exit and 85%-of-best-wave-efficiency rule are copied from
    Dao-AILab/flash-attention v2.6.3
    ``csrc/flash_attn/flash_api.cpp::num_splits_heuristic``.  They describe the
    production operator's discrete launch selection; neither value is an eval
    threshold or a latency-model parameter.
    """

    if min(work_items, launch_slots, num_n_blocks, max_splits) <= 0:
        raise ValueError("FLASH_ATTN split-selection inputs must be positive")
    if work_items >= 0.8 * launch_slots:
        return 1
    capped_splits = min(max_splits, launch_slots, num_n_blocks)
    efficiency = []
    max_efficiency = 0.0
    for splits in range(1, capped_splits + 1):
        eligible = (
            splits == 1
            or ceil_div(num_n_blocks, splits)
            != ceil_div(num_n_blocks, splits - 1)
        )
        waves = work_items * splits / launch_slots
        value = waves / math.ceil(waves) if eligible else 0.0
        efficiency.append(value)
        max_efficiency = max(max_efficiency, value)
    for splits, value in enumerate(efficiency, start=1):
        if value >= 0.85 * max_efficiency:
            return splits
    raise RuntimeError("FLASH_ATTN split selection found no eligible candidate")


def launch_policy(attributes: Mapping[str, object]) -> str:
    value = attributes["launch_policy"]
    if not isinstance(value, str):
        raise ValueError("launch_policy must be a string")
    normalized = value.strip().lower()
    if normalized not in ("injected_launch", "analytical_launch"):
        raise ValueError(
            f"unknown launch_policy {value!r}; expected injected_launch or "
            "analytical_launch")
    return normalized


def fused_moe_local_expert_count(experts: int, attributes: Mapping[str, object]) -> int:
    """Resolve a launch's local count without inventing a physical rank.

    The legacy equal-shard interface needs only E and EP: every rank has
    exactly E/EP experts. Uneven placement needs an explicit owned interval
    or rank; its base/remainder partition follows the canonical source rule.
    Optional ownership facts, when present, must agree with that rule.
    """
    from src.kernel.kernel_spec import linear_expert_partition

    ep_size = require_positive_integer(attributes, "expert_parallel_size")
    if "expert_rank" in attributes:
        expected = linear_expert_partition(experts, ep_size, attributes["expert_rank"])
    else:
        expected = None
    if "owned_experts" in attributes:
        raw_owned = attributes["owned_experts"]
        if (not isinstance(raw_owned, (list, tuple))
                or any(type(expert) is not int for expert in raw_owned)):
            raise TypeError("FUSED_MOE owned_experts must contain integer expert IDs")
        owned = tuple(raw_owned)
        if expected is not None:
            valid = owned == expected
        else:
            valid = any(owned == linear_expert_partition(experts, ep_size, rank)
                for rank in range(ep_size))
        if not valid:
            raise ValueError("FUSED_MOE owned_experts must match canonical linear expert placement")
        return len(owned)
    if expected is not None:
        return len(expected)
    local_experts, remainder = divmod(experts, ep_size)
    if remainder:
        raise ValueError("uneven FUSED_MOE launch requires explicit expert ownership or rank")
    return local_experts


def select_gpu_fused_moe_launch(
    shape: Mapping[str, int],
    variant: str,
    attributes: Mapping[str, object],
    *,
    block_threads: int,
) -> LaunchConfig:
    """Shared Triton grouped-GEMM launch used by GPU-family backends."""
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    if isinstance(block_threads, bool) or not isinstance(block_threads, int):
        raise TypeError("FUSED_MOE block_threads must be an integer")
    if block_threads <= 0:
        raise ValueError("FUSED_MOE block_threads must be positive")
    ep_size = require_positive_integer(attributes, "expert_parallel_size")
    local_experts = fused_moe_local_expert_count(experts, attributes)
    if local_experts == 0:
        raise NotImplementedError("empty expert ranks have a zero contribution rather than a grouped GEMM launch")
    if tokens <= local_experts:
        tile_m, tile_n, tile_k = 16, 32, 64
        group_size_m = 1
    else:
        tile_m, tile_n, tile_k = 64, 64, 32
        group_size_m = 8
    projections = derive_fused_moe_tiling(
        tokens, experts, topk, hidden, intermediate,
        tile_m, tile_n, tile_k, group_size_m=group_size_m,
    )
    parameters = {
        "group_size_m": float(projections[projection]["GSM"]),
        "padded_rows": float(projections[projection]["EM"]),
        "mte2_tile_bytes": 0.0,
        "weight_tile_rows": 0.0,
        "operand_replay_a": 1.0,
        "operand_replay_b": 1.0,
    }
    if ep_size > 1:
        # This repeated launch fact lets the generic predictor distinguish a
        # source-backed rank-local launch from a manually forged EP
        # KernelSpec.  Backends without EP launch evidence never return such
        # a LaunchConfig.
        parameters["expert_parallel_size"] = float(ep_size)
    return LaunchConfig(
        algorithm=f"triton_fused_moe_{projection}",
        grid=(projections[projection]["grid"], 1, 1),
        block=(block_threads, 1, 1),
        tiling=TilingSpec(tile_shape={"m": tile_m, "n": tile_n, "k": tile_k}),
        parameters=parameters,
        execution_variant=projection,
    )


def select_gpu_vector_launch(
    shape: Mapping[str, int], variant: str, attributes: Mapping[str, object],
) -> LaunchConfig:
    del attributes
    execution_variant = require_gpu_vector_variant(variant)
    if execution_variant in ("rmsnorm", "add_rmsnorm"):
        work_items, = require_shape(shape, ("rows",))
        grid = (work_items, 1, 1)
    elif execution_variant.startswith("moe_"):
        tokens, experts, topk, hidden, intermediate = require_shape(
            shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
        if execution_variant.startswith("moe_align"):
            grid = (1, 1, 1)
        elif execution_variant in (
                "moe_count_and_sort", "moe_fill", "moe_enable_if"):
            grid = (ceil_div(tokens * topk + experts, 256), 1, 1)
        elif execution_variant in (
                "moe_act_and_mul", "moe_silu", "moe_mul"):
            grid = (ceil_div(tokens * topk * intermediate, 256), 1, 1)
        elif execution_variant in ("moe_sum", "moe_reduce"):
            grid = (ceil_div(tokens * hidden, 256), 1, 1)
        elif execution_variant in ("moe_topk", "moe_softmax", "moe_gating"):
            grid = (ceil_div(tokens * experts, 256), 1, 1)
        elif execution_variant in (
                "moe_routing_weight_norm", "moe_routing_weight_scale",
                "moe_routed_id_extent", "moe_expert_map_lookup",
                "moe_router_logit_cast", "moe_routed_id_widen"):
            grid = (ceil_div(tokens * topk, 256), 1, 1)
        else:
            raise ValueError(
                f"unhandled GPU MoE auxiliary launch: {execution_variant!r}")
    else:
        work_items, = require_shape(shape, ("element_count",))
        grid = (ceil_div(work_items, 256), 1, 1)
    block = 256
    return LaunchConfig(
        algorithm=execution_variant,
        grid=grid,
        block=(block, 1, 1),
        execution_variant=execution_variant,
    )
