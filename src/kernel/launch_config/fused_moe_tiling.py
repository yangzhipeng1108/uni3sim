"""Shared fused-MoE launch geometry and source-program contracts."""

import math
from typing import Optional


def fused_moe_output_relayout_sources():
    """Source modules whose discrete output-layout facts affect prediction."""
    from src.kernel.launch_config.nvidia_h200 import moe_source_launch
    return (moe_source_launch,)


def observed_fused_moe_output_relayout_bytes(kernel, hardware):
    """Resolve optional compiled output-layout facts in the launch owner."""
    if hardware.hardware_id != "nvidia_h200":
        return None
    source, = fused_moe_output_relayout_sources()
    parameters = kernel.launch.parameters
    return source.observed_output_relayout_bytes(kernel.dtype, kernel.launch.tiling,
        kernel.launch.block_size, parameters["dynamic_shared_memory_bytes"],
        parameters["registers_per_thread"], parameters["static_shared_bytes"])


def observed_fused_moe_output_conversion_warp_instructions(kernel, hardware):
    """Resolve packed output conversion work for the same compiled class."""
    if hardware.hardware_id != "nvidia_h200":
        return None
    source, = fused_moe_output_relayout_sources()
    parameters = kernel.launch.parameters
    return source.observed_output_conversion_warp_instructions(
        kernel.dtype, kernel.launch.tiling, kernel.launch.block_size,
        parameters["dynamic_shared_memory_bytes"], parameters["registers_per_thread"],
        parameters["static_shared_bytes"])


def source_fused_moe_output_conversion_warp_instructions(kernel, hardware):
    """Resolve the source-level packed BF16 conversion lower bound."""
    if hardware.hardware_id != "nvidia_h200" or kernel.launch.algorithm not in (
            f"triton_grouped_gemm:{kernel.variant}",
            f"triton_fused_moe_{kernel.variant}",
            f"vllm_0_11_sm90_unquantized_{kernel.variant}"):
        return None
    source, = fused_moe_output_relayout_sources()
    return source.source_output_conversion_warp_instructions(
        kernel.dtype, kernel.launch.tiling)


def observed_fused_moe_operand_shared_work(kernel, hardware, *, assignment_mode):
    """Resolve static shared-memory work only for a closed physical class."""
    if hardware.hardware_id != "nvidia_h200":
        return None
    if kernel.launch.algorithm not in (
            f"triton_grouped_gemm:{kernel.variant}",
            f"triton_fused_moe_{kernel.variant}"):
        return None
    parameters = kernel.launch.parameters
    required = ("dynamic_shared_memory_bytes", "registers_per_thread",
        "static_shared_bytes", "group_size_m")
    if any(name not in parameters for name in required):
        return None
    source, = fused_moe_output_relayout_sources()
    return source.observed_operand_shared_work(kernel.dtype, kernel.variant, assignment_mode,
        kernel.launch.tiling, parameters["group_size_m"], kernel.launch.block_size,
        parameters["dynamic_shared_memory_bytes"], parameters["registers_per_thread"],
        parameters["static_shared_bytes"])


def observed_fused_moe_matrix_preparation_issue(kernel, hardware, *, assignment_mode):
    """Resolve closed post-wait, pre-matrix instruction work per physical K."""
    if hardware.hardware_id != "nvidia_h200":
        return None
    if kernel.launch.algorithm not in (
            f"triton_grouped_gemm:{kernel.variant}",
            f"triton_fused_moe_{kernel.variant}"):
        return None
    parameters = kernel.launch.parameters
    required = ("dynamic_shared_memory_bytes", "registers_per_thread",
        "static_shared_bytes", "group_size_m")
    if any(name not in parameters for name in required):
        return None
    source, = fused_moe_output_relayout_sources()
    return source.observed_matrix_preparation_issue(kernel.dtype, kernel.variant,
        assignment_mode, kernel.launch.tiling, parameters["group_size_m"],
        kernel.launch.block_size, parameters["dynamic_shared_memory_bytes"],
        parameters["registers_per_thread"], parameters["static_shared_bytes"])


def observed_fused_moe_startup_entry_issue(kernel, hardware, *, assignment_mode):
    """Resolve a source instruction floor only for a closed physical class."""
    if hardware.hardware_id != "nvidia_h200":
        return None
    if kernel.launch.algorithm not in (
            f"triton_grouped_gemm:{kernel.variant}",
            f"triton_fused_moe_{kernel.variant}",
            f"vllm_0_11_sm90_unquantized_{kernel.variant}"):
        return None
    parameters = kernel.launch.parameters
    required = ("dynamic_shared_memory_bytes", "registers_per_thread",
        "static_shared_bytes", "group_size_m")
    if any(name not in parameters for name in required):
        return None
    source, = fused_moe_output_relayout_sources()
    return source.observed_startup_entry_issue(kernel.dtype, kernel.variant, assignment_mode,
        kernel.launch.tiling, parameters["group_size_m"], kernel.launch.block_size,
        parameters["dynamic_shared_memory_bytes"], parameters["registers_per_thread"],
        parameters["static_shared_bytes"])


# CANN 8.5.1 TCubeTiling payloads.  These are discrete source-program facts,
# not duration-derived parameters.  They live in the shared launch owner so
# launch selection and the generic predictor validate the same contract
# without importing a hardware launch package into the predictor.
_CANN_GROUPED_MATMUL_K_PROGRAM_BY_BASE_M = {
    16: (120, 8, 60, 4, 1, 1, 2, 2, 1),
    32: (56, 8, 28, 4, 1, 1, 2, 2, 1),
    64: (24, 8, 12, 4, 1, 1, 2, 2, 1),
    80: (24, 8, 12, 4, 1, 1, 2, 2, 1),
    128: (8, 8, 4, 4, 1, 1, 2, 2, 1),
}

CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES = (
    "cann_depth_a1", "cann_depth_b1",
    "cann_step_ka", "cann_step_kb",
    "cann_step_m", "cann_step_n",
    "cann_db_l0a", "cann_db_l0b", "cann_db_l0c",
)

# Exact physical-input-row payloads supplement the original base-M table.
# Both W1/W2 and E64/H2048/I768 versus E512/H4096/I1024 agree. Uncaptured
# dynamic rows remain unbound; these tuples contain no timing information.
# Evidence: grouped_matmul_ep_input_rows_cann_8_5_1.json.
_CANN_GROUPED_MATMUL_K_PROGRAM_BY_EXACT_ROWS = {
    (48, 40): (40, 8, 20, 4, 1, 1, 2, 2, 1),
    (48, 48): (40, 8, 20, 4, 1, 1, 2, 2, 1),
    (96, 88): (16, 8, 8, 4, 1, 1, 2, 2, 1),
    (96, 96): (16, 8, 8, 4, 1, 1, 2, 2, 1),
    (112, 104): (16, 8, 8, 4, 1, 1, 2, 2, 1),
    (112, 112): (16, 8, 8, 4, 1, 1, 2, 2, 1),
}


def cann_grouped_matmul_k_program(
        base_m: int, routed_rows: int) -> Optional[tuple[int, ...]]:
    """Return the captured K program for physical input descriptor rows."""

    if isinstance(base_m, bool) or not isinstance(base_m, int):
        raise TypeError("base_m must be an integer")
    if isinstance(routed_rows, bool) or not isinstance(routed_rows, int):
        raise TypeError("routed_rows must be an integer")
    if base_m <= 0 or routed_rows <= 0:
        raise ValueError("base_m and routed_rows must be positive")
    key = (base_m, routed_rows)
    if key in _CANN_GROUPED_MATMUL_K_PROGRAM_BY_EXACT_ROWS:
        return _CANN_GROUPED_MATMUL_K_PROGRAM_BY_EXACT_ROWS[key]
    if base_m in _CANN_GROUPED_MATMUL_K_PROGRAM_BY_BASE_M:
        return _CANN_GROUPED_MATMUL_K_PROGRAM_BY_BASE_M[base_m]
    return None


def _padded_rows(projection_rows: int, tokens: int, topk: int,
                 experts: int, tile_m: int) -> int:
    maximum_sorted = tokens * topk + experts * (tile_m - 1)
    if projection_rows < tile_m:
        return min(maximum_sorted, tokens * topk * tile_m)
    return maximum_sorted


def derive_fused_moe_tiling(tokens: int, experts: int, topk: int,
                            hidden: int, intermediate: int,
                            tile_m: int, tile_n: int, tile_k: int,
                            group_size_m: int,
                            reuse_stage1_routing: bool = False) -> dict:
    """Derive both projection grids from one shared Triton configuration."""

    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(
            f"tile_m={tile_m} and tile_n={tile_n} must both be positive")
    projection_specs = {
        "w1": (tokens, hidden, 2 * intermediate),
        "w2": (tokens * topk, intermediate, hidden),
    }
    result = {}
    stage1_padded = None
    for name, (rows, projection_k, projection_n) in projection_specs.items():
        if name == "w2" and reuse_stage1_routing:
            if stage1_padded is None:
                raise RuntimeError("FusedMoE W2 routing reuse lacks W1 rows")
            padded = stage1_padded
        else:
            padded = _padded_rows(rows, tokens, topk, experts, tile_m)
        if name == "w1":
            stage1_padded = padded
        result[name] = {
            "N": projection_n,
            "K": projection_k,
            "BM": tile_m,
            "BN": tile_n,
            "BK": tile_k,
            "GSM": group_size_m,
            "EM": padded,
            "grid": math.ceil(padded / tile_m) * math.ceil(projection_n / tile_n),
        }
    return result
