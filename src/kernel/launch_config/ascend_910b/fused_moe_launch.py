"""Huawei 910B grouped-matmul launch selection."""

from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.fused_moe_tiling import (
    CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES,
    cann_grouped_matmul_k_program,
)
from src.kernel.launch_config.selection_primitives import (
    launch_policy, require_dtype, require_positive_integer, require_shape,
    require_variant)
from src.kernel.launch_config import hardware_skeleton as _skeleton


_FUSED_MOE_EP_EVIDENCE_REFERENCE = (
    "microbench/ascend_910b/references/"
    "fused_moe_ep_vllm_ascend_0_18_0.json")
_FUSED_MOE_EP_EVIDENCE_SHA256 = (
    "e93af6cd9ef2471dc75e936abbd22324def1445f28e08562682e1be5a8dba6ea")

def _rank_local_routed_rows(
        attributes: Mapping[str, object], experts: int,
        global_routed_rows: int) -> int:
    raw_counts = attributes["local_tokens_per_expert"]
    if not isinstance(raw_counts, (list, tuple)):
        raise TypeError("local_tokens_per_expert must be a sequence")
    if len(raw_counts) != experts:
        raise ValueError(
            "local_tokens_per_expert has "
            f"{len(raw_counts)} entries, expected {experts}")
    counts = []
    for value in raw_counts:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "local_tokens_per_expert values must be integers")
        if value < 0:
            raise ValueError(
                "local_tokens_per_expert values must be non-negative")
        counts.append(value)
    routed_rows = sum(counts)
    if routed_rows <= 0:
        raise NotImplementedError(
            "Ascend 910B upstream invokes GroupedMatmul for a zero-row EP "
            "rank, but its legal base-M is not source-backed")
    if routed_rows > global_routed_rows:
        raise ValueError(
            "local_tokens_per_expert sum cannot exceed tokens * topk")
    return routed_rows


def select_fused_moe_recipe(
        shape: Mapping[str, int], *, hardware) -> tuple[str, ...]:
    _, experts, _, hidden, _ = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    del hardware
    # vLLM-Ascend's unquantized public FusedMoE call dispatches the CANN
    # public chain, not the older standalone helper sequence retained in the
    # historical per-kernel dataset.  MoeInitRoutingV2 is selected by its
    # host CheckV2Case predicate: E=128, H=2048, full active-expert range,
    # int32 expert-token output and no quantization.  The latter three inputs
    # are fixed by this public vLLM entry point; E/H are the shape controls
    # available at this selector boundary.  This is a compiler/source ABI
    # dispatch, never a duration or case-name branch.
    init_variant = (
        "moe_init_routing_v2"
        if experts == 128 and hidden == 2048
        else "moe_init_routing_v3")
    recipe = (
        "moe_gating_topk", init_variant,
    )
    if init_variant == "moe_init_routing_v2":
        recipe += ("moe_cast",)
    return recipe + (
        "w1", "moe_swiglu", "w2", "moe_abs", "moe_token_unpermute",
    )


def select_fused_moe_cast_roles(
        shape: Mapping[str, int], *, hardware) -> tuple[str, ...]:
    """Return the source-ABI Cast role selected by public FusedMoE."""

    recipe = select_fused_moe_recipe(shape, hardware=hardware)
    if recipe.count("moe_cast") == 0:
        return ()
    if recipe.count("moe_cast") == 1:
        # V2 produces int32 expert-token metadata while GroupedMatmulV5
        # consumes the int64 form.  Its one public dynamic Cast is therefore
        # the E-element expert-token conversion, not a TOPK-id conversion.
        return ("expert_tokens",)
    raise AssertionError("Ascend public FusedMoE recipe has unexpected casts")


# 910B's retained CANN recipe has no ``moe_reduce`` invocation.
select_fused_moe_reduce_work_fractions = (
    _skeleton.select_no_fused_moe_reduce_work_fractions)


def select_fused_moe_launch(shape: Mapping[str, int], dtype: Optional[str],
                            variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    require_dtype(dtype, ("bf16",))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    expert_parallel_size = require_positive_integer(
        attributes, "expert_parallel_size")
    if expert_parallel_size not in (1, 2):
        raise NotImplementedError(
            "Ascend 910B FUSED_MOE source evidence supports only EP1 or EP2")
    if expert_parallel_size == 2 and experts != 128:
        raise NotImplementedError(
            "Ascend 910B FUSED_MOE EP2 source evidence requires 128 experts")
    if experts % expert_parallel_size:
        raise ValueError(
            "expert_parallel_size must divide FUSED_MOE experts")
    global_routed_rows = tokens * topk
    if expert_parallel_size > 1:
        _rank_local_routed_rows(
            attributes, experts, global_routed_rows)
    del hidden, intermediate
    launch_policy(attributes)
    core_count = int(hardware.num_compute_units)
    # CANN 8.5.1 BF16 GroupedMatmul launch payloads select a 16-row aligned
    # base-M capped at 128, with base-N=256 and base-K=64.  These are discrete
    # compiler launch facts; they are not inferred from kernel duration.
    # 0.18 init-routing retains a global T*K-row expanded_x allocation.
    # Native GMM's TCube descriptor follows that physical x extent, not the
    # device group-list sum. Local expert counts still determine math/traffic.
    # See grouped_matmul_ep_input_rows_cann_8_5_1.json's same-x/different-count
    # controls and W1/W2 output-shape observations.
    tile_m = min(128, ((global_routed_rows + 15) // 16) * 16)
    parameters = {
        "group_size_m": 0.0,
        "padded_rows": 0.0,
        # Traffic-derived residency curves have no descriptor/IR mapping
        # and must not be interpolated into production Forward.
        "weight_tile_rows": 0.0,
        # Retained schema field; zero explicitly means unavailable and is
        # not consumed by the aggregate-byte NPU Forward.
        "mte2_tile_bytes": 0.0,
        "operand_replay_a": 1.0,
        "operand_replay_b": 1.0,
    }
    program = cann_grouped_matmul_k_program(tile_m, global_routed_rows)
    if program is not None:
        for name, value in zip(CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES,
                               program):
            parameters[name] = float(value)
    if expert_parallel_size > 1:
        parameters["expert_parallel_size"] = float(expert_parallel_size)
    return LaunchConfig(
        algorithm=f"cann_grouped_matmul_{projection}",
        grid=(core_count, 1, 1),
        block=(1, 1, 1),
        tiling=TilingSpec(tile_shape={"m": tile_m, "n": 256, "k": 64}),
        parameters=parameters,
        execution_variant=projection,
    )
