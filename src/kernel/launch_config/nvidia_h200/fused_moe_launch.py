"""NVIDIA H200 fused-MoE grouped-GEMM launch selection.

The Hopper vLLM 0.24.0 stack runs DeepSeek-V3's routed experts through the
Triton ``fused_moe_kernel`` with ``use_fp8_w8a8`` and a [128, 128] weight
block shape.  ``try_get_optimal_moe_config`` resolves the bundled
``E=256,N=<2048/TP>,device_name=NVIDIA_H200,dtype=fp8_w8a8,
block_shape=[128,128].json`` file by the nearest tuned batch key; the
resolved tile/warp/stage configuration and the Triton 3.6 compile facts of
every range are recorded in
``microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json``
(``deepseek_v3_selection`` / ``compile_metadata``).  The tables below restate
that record; nothing here is a duration.

Launch derivation follows the installed source
(``vllm/model_executor/layers/fused_moe/fused_moe.py`` sha256 57a3f7ef...):

* ``_prepare_expert_assignment`` skips ``moe_align_block_size`` when
  ``num_tokens * top_k * 4 <= global_num_experts`` (no expert map, no
  INT8/INT4 W8A16 block path).  The kernel is then compiled with
  ``naive_block_assignment=True``: every routed token owns one M tile whose
  row 0 is its only valid row, and ``EM = num_tokens * top_k * BLOCK_SIZE_M``.
* otherwise ``moe_align_block_size`` sizes ``sorted_token_ids`` as
  ``numel + E * (BLOCK_SIZE_M - 1)``, capped at ``numel * BLOCK_SIZE_M`` when
  ``numel < E``, and ``invoke_fused_moe_triton_kernel`` further caps ``EM`` at
  ``A.size(0) * top_k * BLOCK_SIZE_M`` when ``A.size(0) < BLOCK_SIZE_M``
  (W1: A = tokens, top_k = topk; W2: A = tokens * topk, top_k = 1).
* ``grid = cdiv(EM, BLOCK_SIZE_M) * cdiv(B.size(1), BLOCK_SIZE_N)`` with
  ``B.size(1) = 2 * intermediate`` (W1) or ``hidden`` (W2); the block is
  ``num_warps * 32`` threads; ``BLOCK_SIZE_K`` is clamped to the 128 block.
* W2 multiplies the router weight (``MUL_ROUTED_WEIGHT=True``), W1 does not.

The explicitly versioned0.30 source path additionally selects unquantized
and block-FP8 configurations from this hardware's own runtime sources.
Requests outside the bound domains remain explicit gaps. The dispatcher
never projects another hardware package.
"""

from typing import Mapping, Optional


def moe_alignment_capacity(tokens, topk, experts, block_size):
    """Installed vLLM 0.24 CUDA wrapper's unpadded sorted-ID allocation."""
    routed = tokens * topk
    capacity = routed + experts * (block_size - 1)
    return min(capacity, routed * block_size) if routed < experts else capacity

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.selection_primitives import (
    ceil_div, require_shape, require_variant)
from src.kernel.model_contract import (
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
)
from src.kernel.launch_config.nvidia_h200.moe_source_launch import (
    select_source_moe_launch, select_source_moe_recipe)


def select_fused_moe_recipe(shape, hardware, attributes=None):
    """Select the declared source sequence, never a hardware-only runtime."""
    if attributes is None or "model_execution_version" not in attributes:
        raise NotImplementedError("H200 MoE recipe requires explicit runtime/config source facts")
    return select_source_moe_recipe(shape, attributes, hardware)


def select_fused_moe_cast_roles(shape, hardware, attributes=None):
    # The CUDA recipes use named router casts, not the NPU moe_cast ABI.
    select_fused_moe_recipe(shape, hardware, attributes)
    return ()


def select_fused_moe_reduce_work_fractions(shape, hardware, attributes=None):
    recipe = select_fused_moe_recipe(shape, hardware, attributes)
    # The0.11 Torch sum consumes the complete routed output once.
    return (1.0,) if recipe.count("moe_reduce") == 1 else ()


H200_TUNED_CONFIG_EVIDENCE_REFERENCE = (
    "microbench/nvidia_h200/references/"
    "fused_moe_tuned_config_vllm_0_24_0_h200.json")

FP8_BLOCK_ALGORITHM_PREFIX = "vllm_triton_fused_moe_fp8_block128"
FP8_BLOCK_SIZE = 128

# DeepSeek-V3 routed-expert domain of the bundled H200 FP8 block configs.
_FP8_GLOBAL_EXPERTS = 256
_FP8_TOPK = 8
_FP8_HIDDEN = 7168

# (m_min, m_max, config_id) per bundled file, keyed by N = intermediate per
# rank.  The m ranges are the nearest-key partition of the tuned batch keys.
_FP8_SELECTION_RANGES = {
    512: (
        (1, 1, "cb47dc9924056440"),
        (2, 3, "32e1b6206e8cd459"),
        (4, 6, "0e0b351e3086258f"),
        (7, 12, "96f97a9b394c141b"),
        (13, 20, "cb47dc9924056440"),
        (21, 28, "c707ac95391bd157"),
        (29, 80, "cb47dc9924056440"),
        (81, 112, "96f97a9b394c141b"),
        (113, 192, "0e0b351e3086258f"),
        (193, 384, "cb47dc9924056440"),
        (385, 768, "a2b5ff407c7333d6"),
        (769, 1792, "57bc196faf4fc400"),
        (1793, 32768, "d26bcbdc1237b406"),
    ),
    256: (
        (1, 3, "12d9d4bdf00fb2ff"),
        (4, 6, "35591cb2c794b91b"),
        (7, 12, "12d9d4bdf00fb2ff"),
        (13, 20, "63d65789df08a862"),
        (21, 28, "12d9d4bdf00fb2ff"),
        (29, 40, "b336fde8d25c71c0"),
        (41, 80, "50e8c1da81dc66d8"),
        (81, 112, "63d65789df08a862"),
        (113, 384, "65c0c8e5705f60f2"),
        (385, 768, "63d65789df08a862"),
        (769, 3584, "bc3c0c641b7ec1b3"),
        (3585, 32768, "12974297821ec13c"),
    ),
}

# config_id -> (BM, BN, BK, GROUP_SIZE_M, num_warps, num_stages,
#               W1 registers, W2 registers, dynamic shared bytes,
#               resident CTAs/SM) from the reference's compile_metadata
# (Triton 3.6, sm_90, aligned-assignment specialization).
_FP8_KERNEL_CONFIGS = {
    "cb47dc9924056440": (16, 128, 128, 1, 4, 3, 100, 100, 37376, 4),
    "32e1b6206e8cd459": (16, 128, 128, 16, 8, 4, 80, 76, 55296, 3),
    "0e0b351e3086258f": (16, 128, 128, 16, 4, 3, 100, 100, 37376, 4),
    "96f97a9b394c141b": (16, 128, 128, 32, 4, 3, 100, 100, 37376, 4),
    "c707ac95391bd157": (16, 128, 128, 16, 4, 4, 100, 100, 55808, 4),
    "a2b5ff407c7333d6": (32, 128, 128, 32, 4, 3, 160, 159, 41472, 3),
    "57bc196faf4fc400": (64, 128, 128, 32, 4, 4, 236, 237, 98816, 2),
    "d26bcbdc1237b406": (64, 128, 128, 16, 4, 4, 236, 237, 98816, 2),
    "12d9d4bdf00fb2ff": (16, 128, 128, 16, 4, 4, 100, 100, 55808, 4),
    "35591cb2c794b91b": (16, 128, 128, 64, 4, 4, 100, 100, 55808, 4),
    "63d65789df08a862": (16, 128, 128, 16, 4, 3, 100, 100, 37376, 4),
    "b336fde8d25c71c0": (16, 128, 128, 16, 4, 5, 100, 100, 74240, 3),
    "50e8c1da81dc66d8": (16, 128, 128, 64, 4, 3, 100, 100, 37376, 4),
    "65c0c8e5705f60f2": (16, 128, 128, 32, 4, 3, 100, 100, 37376, 4),
    "bc3c0c641b7ec1b3": (64, 128, 128, 32, 4, 4, 236, 237, 98816, 2),
    "12974297821ec13c": (64, 128, 128, 16, 4, 3, 236, 237, 74240, 2),
}

# SGLang 0.5.12 runs its own ``fused_moe_kernel``
# (moe_runner/triton_utils/fused_moe_triton_kernels.py) with the shared
# expert fused in (E = 257, top-k 9) and the H200 configs
# ``configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,
# block_shape=[128, 128].json`` (up) and ``..._down.json`` (down, USE_TMA);
# SGLang logs the triton_3_6_0 -> triton_3_5_1 fallback.  Key -> (BM, BN, BK,
# GROUP_SIZE_M, num_warps, num_stages); ``try_get_optimal_moe_config`` takes
# ``configs[min(keys, key=lambda x: abs(x - M))]`` (ties to the smaller key).
SGLANG_ALGORITHM_PREFIX = "sglang_triton_fused_moe_fp8_block128"
_SGLANG_INTERMEDIATE = 256
_SGLANG_UP_CONFIGS = {
    1: (16, 64, 128, 1, 4, 5), 2: (16, 64, 128, 1, 4, 5),
    4: (16, 64, 128, 32, 4, 5), 8: (16, 64, 128, 1, 4, 5),
    16: (16, 64, 128, 1, 4, 3), 24: (16, 128, 128, 16, 4, 3),
    32: (16, 64, 128, 16, 4, 5), 48: (16, 64, 128, 1, 4, 5),
    64: (16, 64, 128, 16, 4, 4), 96: (16, 64, 128, 16, 4, 5),
    128: (16, 64, 128, 16, 4, 3), 256: (16, 128, 128, 32, 4, 4),
    512: (32, 128, 128, 1, 4, 5), 1024: (64, 128, 128, 1, 4, 4),
}
_SGLANG_DOWN_CONFIGS = {
    1: (16, 128, 128, 16, 4, 2), 2: (16, 128, 64, 64, 4, 3),
    4: (16, 128, 128, 1, 4, 2), 8: (16, 128, 128, 1, 4, 2),
    16: (16, 128, 128, 1, 4, 2), 24: (16, 128, 64, 64, 4, 3),
    32: (16, 256, 64, 1, 4, 3), 48: (16, 128, 64, 1, 4, 3),
    64: (16, 128, 128, 64, 4, 2), 96: (16, 128, 128, 64, 4, 2),
    128: (16, 128, 128, 64, 4, 2), 256: (16, 128, 128, 64, 4, 2),
    512: (32, 128, 128, 16, 4, 2), 1024: (64, 128, 128, 64, 4, 3),
}


def _sglang_nearest(configs, tokens):
    return configs[min(configs, key=lambda key: abs(key - tokens))]


def sglang_fp8_block_program(tile_m, tile_n, tile_k, warps, stages,
                             use_tma):
    """Name of the measured SGLang tile-program row (HWSpec family)."""

    return (f"sgl_{'tma_' if use_tma else ''}bm{tile_m}_bn{tile_n}"
            f"_bk{tile_k}_w{warps}_s{stages}")


def sglang_expanded_rows(tokens: int, topk: int, experts: int,
                         tile_m: int) -> int:
    """SGLang ``moe_align_block_size`` ``sorted_token_ids`` length."""

    numel = tokens * topk
    if numel < experts + 1:
        return numel * tile_m
    return numel + (experts + 1) * (tile_m - 1)


def _sglang_domain(shape, dtype, variant, attributes) -> bool:
    if "fused_shared_experts" not in attributes:
        return False
    if not _fp8_block_domain(shape, dtype, variant, attributes):
        raise ValueError(
            "SGLang fused shared-expert FUSED_MOE is source-backed only for "
            "the DeepSeek-V3 FP8 block domain")
    if shape["intermediate"] != _SGLANG_INTERMEDIATE:
        raise NotImplementedError(
            "SGLang H200 fused-MoE configs are enumerated for N=256 (TP8)")
    return True


def _select_sglang_fp8_block_launch(shape, variant, attributes, hardware):
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    fused = attributes["fused_shared_experts"]
    kernel_experts, kernel_topk = experts + fused, topk + fused
    up = _sglang_nearest(_SGLANG_UP_CONFIGS, tokens)
    config = up if projection == "w1" else _sglang_nearest(
        _SGLANG_DOWN_CONFIGS, tokens)
    tile_m, tile_n, tile_k, group_size_m, warps, stages = config
    use_tma = projection == "w2"
    # One sorted_token_ids (sized with the up config's BLOCK_SIZE_M) feeds
    # both projections; the grid divides it by each config's BLOCK_SIZE_M.
    expanded_rows = sglang_expanded_rows(
        tokens, kernel_topk, kernel_experts, up[0])
    projection_n = 2 * intermediate if projection == "w1" else hidden
    grid = ceil_div(expanded_rows, tile_m) * ceil_div(projection_n, tile_n)
    program = sglang_fp8_block_program(
        tile_m, tile_n, tile_k, warps, stages, use_tma)
    family = hardware.grouped_gemm_fp8_block_tile_pipelines
    if family is None or program not in family:
        raise NotImplementedError(
            f"SGLang fused-MoE tile program {program} is not measured")
    return LaunchConfig(
        algorithm=(f"{SGLANG_ALGORITHM_PREFIX}_aligned_"
                   f"{'tma_' if use_tma else ''}{projection}"),
        grid=(grid, 1, 1),
        block=(32 * warps, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=stages),
        parameters={
            "group_size_m": float(group_size_m),
            "padded_rows": float(expanded_rows),
            "mte2_tile_bytes": 0.0,
            "weight_tile_rows": 0.0,
            "operand_replay_a": 1.0,
            "operand_replay_b": 1.0,
            "dynamic_shared_memory_bytes": float(
                family[program]["dynamic_shared_memory_bytes_per_cta"]),
            "num_warps": float(warps),
            "num_stages": float(stages),
            SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER: (
                1.0 if projection == "w2" else 0.0),
        },
        execution_variant=projection,
    )




def fp8_block_naive_assignment(tokens: int, topk: int, experts: int) -> bool:
    """vLLM 0.24 ``_prepare_expert_assignment`` naive-block predicate."""

    return tokens * topk * 4 <= experts


def _fp8_block_domain(shape: Mapping[str, int], dtype: Optional[str],
                      variant: Optional[str],
                      attributes: Mapping[str, object]) -> bool:
    if variant is None or variant.strip().lower() not in ("w1", "w2"):
        return False
    if dtype not in ("bf16",):
        return False
    # Both quantization attributes are schema-optional: a legacy spec that
    # declares no routed quantization contract is a BF16 launch.
    if ("routed_weight_dtype" not in attributes
            or attributes["routed_weight_dtype"] != "fp8"):
        return False
    # The Serving FP8 contract records scale granularity as group size 0 for
    # both per-tensor and 128x128 block scales; within the enumerated
    # (E, topk, hidden) domain the model contract admits only DeepSeek-V3,
    # whose checkpoint schema requires [128, 128] weight blocks.
    if attributes["weight_group_size"] != 0:
        return False
    if attributes["expert_parallel_size"] != 1:
        return False
    _, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    return (
        (experts, topk, hidden) == (_FP8_GLOBAL_EXPERTS, _FP8_TOPK, _FP8_HIDDEN)
        and intermediate in _FP8_SELECTION_RANGES)


def fp8_block_kernel_config(tokens: int, intermediate: int) -> dict:
    """Return the bundled H200 FP8 block config vLLM resolves for M tokens."""

    if intermediate not in _FP8_SELECTION_RANGES:
        raise NotImplementedError(
            "H200 FP8 block fused-MoE tuned config is enumerated only for "
            f"N in {sorted(_FP8_SELECTION_RANGES)}, got {intermediate}")
    matches = tuple(
        config_id
        for low, high, config_id in _FP8_SELECTION_RANGES[intermediate]
        if low <= tokens <= high)
    if len(matches) != 1:
        raise NotImplementedError(
            "H200 FP8 block fused-MoE tuned config is enumerated for "
            f"M in [1, 32768], got {tokens}")
    config_id, = matches
    (tile_m, tile_n, tile_k, group_size_m, warps, stages, registers_w1,
     registers_w2, shared_bytes, resident) = _FP8_KERNEL_CONFIGS[config_id]
    return {
        "config_id": config_id,
        "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
        "group_size_m": group_size_m,
        "num_warps": warps, "num_stages": stages,
        "registers_per_thread": {"w1": registers_w1, "w2": registers_w2},
        "dynamic_shared_memory_bytes": shared_bytes,
        "resident_ctas_per_sm": resident,
    }


def fp8_block_expanded_rows(
        tokens: int, topk: int, experts: int, tile_m: int,
        projection: str) -> int:
    """Return the source ``EM`` passed to ``fused_moe_kernel``."""

    if fp8_block_naive_assignment(tokens, topk, experts):
        return tokens * topk * tile_m
    numel = tokens * topk
    padded = numel + experts * (tile_m - 1)
    if numel < experts:
        padded = min(numel * tile_m, padded)
    if projection == "w1":
        a_rows, kernel_topk = tokens, topk
    elif projection == "w2":
        a_rows, kernel_topk = tokens * topk, 1
    else:
        raise ValueError(f"FUSED_MOE projection must be w1 or w2: {projection!r}")
    if a_rows < tile_m:
        padded = min(padded, a_rows * kernel_topk * tile_m)
    return padded


def _select_fp8_block_launch(
        shape: Mapping[str, int], variant: str) -> LaunchConfig:
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    config = fp8_block_kernel_config(tokens, intermediate)
    tile_m, tile_n, tile_k = (
        config["tile_m"], config["tile_n"], config["tile_k"])
    if tile_k != FP8_BLOCK_SIZE:
        raise AssertionError("FP8 block BLOCK_SIZE_K must equal the 128 block")
    projection_n = 2 * intermediate if projection == "w1" else hidden
    expanded_rows = fp8_block_expanded_rows(
        tokens, topk, experts, tile_m, projection)
    grid = ceil_div(expanded_rows, tile_m) * ceil_div(projection_n, tile_n)
    assignment = (
        "naive" if fp8_block_naive_assignment(tokens, topk, experts)
        else "aligned")
    return LaunchConfig(
        algorithm=f"{FP8_BLOCK_ALGORITHM_PREFIX}_{assignment}_{projection}",
        grid=(grid, 1, 1),
        block=(32 * config["num_warps"], 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=config["num_stages"]),
        parameters={
            "group_size_m": float(config["group_size_m"]),
            "padded_rows": float(expanded_rows),
            "mte2_tile_bytes": 0.0,
            "weight_tile_rows": 0.0,
            "operand_replay_a": 1.0,
            "operand_replay_b": 1.0,
            "dynamic_shared_memory_bytes": float(
                config["dynamic_shared_memory_bytes"]),
            "num_warps": float(config["num_warps"]),
            "num_stages": float(config["num_stages"]),
            SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER: (
                1.0 if projection == "w2" else 0.0),
        },
        execution_variant=projection,
    )


# ---------------------------------------------------------------------------
# DeepSeek-V4 MXFP4 MARLIN aggregate (vLLM 0.30, SM90).
#
# On SM90 vLLM 0.30 selects ``fused_marlin_moe`` for MXFP4 weights (TRTLLM
# and DeepGEMM MoE require SM100; ``marlin_moe.py:296-321``).  The launch
# rule below restates only discrete source facts plus non-duration launch
# evidence from ``kc_wq_moe_mxfp4_deepseek_v4__v030`` (grid/block/registers
# only):
#
# * ``block_size_m``: first of (8, 16, 32, 48, 64) with
#   ``M * topk / E_local / block_size_m < 0.9`` where ``M`` is the token
#   count, or ``ceil(tokens * E_local / E_global)`` under EP
#   (``marlin_moe.py:304-314``).
# * ``blocks = sms * exec_cfg.blocks_per_sm`` (``ops.cu:341``); the register
#   cap ``min(255*1024 // (regs * 128 * 4), 4)`` for single-M-block configs
#   and 2 for multi-M-block configs (``ops.cu:226-229``) with the evidenced
#   per-``block_size_m`` registers 119 (bsm 8) / 142 (bsm 16) / 205+ (bsm >=
#   32, large-batch cap 2) reproduces every measured grid.
# * the work clamp ``ops.cu:231-234`` with the evidenced thread_n=64 small-
#   batch config: work = (prob_n // 64) * prob_m * top_k * 4 with
#   (prob_m, top_k) = (tokens, topk) for w13 and (tokens * topk, 1) for w2;
#   when ``work < sms * allow`` the source sets
#   ``allow = max(work // sms, 1)`` (integer division).
# * the block is (128, 1, 1) for every measured MXFP4 launch.
# ---------------------------------------------------------------------------

_MXFP4_BLOCK_M_CANDIDATES = (8, 16, 32, 48, 64)
# Evidenced registers per thread of the selected small-batch MARLIN configs.
_MXFP4_SMALL_BATCH_REGISTERS = {8: 119, 16: 142}
_MXFP4_DEVICE_REG_BYTES = 255 * 1024
_MXFP4_CLAMP_THREAD_N = 64


def _dsv4_mxfp4_domain(shape: Mapping[str, int], dtype: Optional[str],
                       variant: Optional[str],
                       attributes: Mapping[str, object]) -> bool:
    if variant is None or variant.strip().lower() != "routed_experts":
        return False
    if dtype != "bf16":
        return False
    if attributes["routed_weight_dtype"] != "mxfp4":
        return False
    if attributes["weight_group_size"] != 32:
        return False
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    del tokens, topk
    expert_parallel_size = attributes["expert_parallel_size"]
    if expert_parallel_size < 1 or experts % expert_parallel_size:
        return False
    # MARLIN tiles N by 128-wide slices (max_thread_n) and K by 64.
    return (2 * intermediate) % 128 == 0 and hidden % 128 == 0


def _dsv4_mxfp4_block_size_m(tokens: int, topk: int, local_experts: int,
                             global_experts: int,
                             expert_parallel_size: int) -> int:
    if expert_parallel_size > 1:
        m_estimate = ceil_div(tokens * local_experts, global_experts)
    else:
        m_estimate = tokens
    for candidate in _MXFP4_BLOCK_M_CANDIDATES:
        if m_estimate * topk / local_experts / candidate < 0.9:
            return candidate
    return _MXFP4_BLOCK_M_CANDIDATES[-1]


def _dsv4_mxfp4_blocks_per_sm(block_m: int, prob_n: int, prob_m: int,
                              top_k: int, sms: int) -> int:
    if ceil_div(block_m, 16) == 1:
        registers = _MXFP4_SMALL_BATCH_REGISTERS[block_m]
        register_cap = _MXFP4_DEVICE_REG_BYTES // (registers * 128 * 4)
        allow = max(min(register_cap, 4), 1)
    else:
        allow = 2
    work = (prob_n // _MXFP4_CLAMP_THREAD_N) * prob_m * top_k * 4
    if work < sms * allow:
        allow = max(work // sms, 1)
    return allow


def _select_dsv4_mxfp4_launch(shape: Mapping[str, int],
                              attributes: Mapping[str, object],
                              hardware) -> LaunchConfig:
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    expert_parallel_size = attributes["expert_parallel_size"]
    local_experts = experts // expert_parallel_size
    if expert_parallel_size == 1:
        raw_counts = attributes["tokens_per_expert"]
    else:
        raw_counts = attributes["local_tokens_per_expert"]
    if (not isinstance(raw_counts, (list, tuple))
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in raw_counts)):
        raise TypeError(
            "H200 MXFP4 routed_experts routing counts must be integers")
    local_counts = tuple(raw_counts)
    if any(value < 0 for value in local_counts):
        raise ValueError(
            "H200 MXFP4 routed_experts routing counts cannot be negative")
    if len(local_counts) != experts:
        raise ValueError(
            "H200 MXFP4 routed_experts local routing count has wrong length")
    if (expert_parallel_size == 1
            and sum(local_counts) != tokens * topk):
        raise ValueError(
            "H200 MXFP4 routed_experts routing does not conserve rows")
    if expert_parallel_size > 1:
        owned_experts = tuple(attributes["owned_experts"])
        if any(
                count and expert_id not in owned_experts
                for expert_id, count in enumerate(local_counts)):
            raise ValueError(
                "H200 MXFP4 routed_experts local rows include an unowned "
                "expert")

    block_m = _dsv4_mxfp4_block_size_m(
        tokens, topk, local_experts, experts, expert_parallel_size)
    sms = int(hardware.num_compute_units)
    allow_w13 = _dsv4_mxfp4_blocks_per_sm(
        block_m, 2 * intermediate, tokens, topk, sms)
    allow_w2 = _dsv4_mxfp4_blocks_per_sm(
        block_m, hidden, tokens * topk, 1, sms)
    padded_rows = sum(
        ceil_div(count, block_m) * block_m
        for count in local_counts if count > 0)
    if padded_rows <= 0:
        raise ValueError("H200 MXFP4 routed_experts has no active expert rows")
    return LaunchConfig(
        algorithm="vllm_marlin_moe_mxfp4_aggregate",
        grid=(sms * (allow_w13 + allow_w2), 1, 1),
        block=(128, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": block_m, "n": 128, "k": 64}),
        parameters={
            # This aggregate launch models one rank's local expert shard.
            # Preserve EP because it selected the owned experts, the local
            # rows and the M estimate above.
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
        execution_variant="routed_experts",
    )

def owns_select_fused_moe_launch(shape: Mapping[str, int],
                                 dtype: Optional[str],
                                 variant: Optional[str],
                                 attributes: Mapping[str, object],
                                 hardware) -> bool:
    """Whether the request is in an H200-owned fused-MoE domain."""

    del hardware
    return (variant in ("w1", "w2") and "model_execution_version" in attributes
            or _fp8_block_domain(shape, dtype, variant, attributes)
            or _dsv4_mxfp4_domain(shape, dtype, variant, attributes))


def select_fused_moe_launch(shape: Mapping[str, int], dtype: Optional[str],
                            variant: Optional[str],
                            attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    if variant in ("w1", "w2") and "model_execution_version" in attributes:
        return select_source_moe_launch(shape, dtype, variant, attributes, hardware)
    if _sglang_domain(shape, dtype, variant, attributes):
        return _select_sglang_fp8_block_launch(
            shape, variant, attributes, hardware)
    if _dsv4_mxfp4_domain(shape, dtype, variant, attributes):
        return _select_dsv4_mxfp4_launch(shape, attributes, hardware)
    del hardware
    if not _fp8_block_domain(shape, dtype, variant, attributes):
        raise NotImplementedError(
            "H200 FUSED_MOE requires a bound runtime/config for its unquantized, "
            "FP8 block or MXFP4 MARLIN source program")
    return _select_fp8_block_launch(shape, variant)
