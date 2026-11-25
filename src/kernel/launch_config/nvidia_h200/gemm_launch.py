"""NVIDIA H200 launch selection for observed dense and FP8 block GEMMs.

vLLM 0.24.0 prices a DeepSeek-V3 FP8 linear (128x128 e4m3 weight blocks,
dynamic 1x128 activation groups, BF16 output) through
``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``
(``vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py``), which
dispatches on the token count M with ``torch.cond(M < 32, ...)``:

* ``M < 32``: FlashInfer ``fp8_blockscale_gemm`` (TensorRT-LLM
  ``CutlassFp8BlockScaleGemmRunner::gemm`` with BF16 input): one
  ``scale_1x128_kernel<<<num_sms * 8, 256>>>`` activation quantisation
  followed by ``deep_gemm::fp8_gemm_kernel_swapAB`` whose configuration comes
  from ``deep_gemm::jit::get_best_gemm_config(N, M, K, 1, num_sms, false,
  true)`` (``flashinfer/data/csrc/nv_internal/tensorrt_llm/deep_gemm/
  jit_utils.cuh``);
* ``M >= 32``: ``per_token_group_quant_fp8`` (a separate kernel, outside this
  selector) and DeepGEMM ``fp8_gemm_nt`` -> ``sm90_fp8_gemm_1d2d_impl`` whose
  configuration comes from ``get_best_config<SM90ArchSpec>`` (DeepGEMM
  ``891d57b4`` ``csrc/jit_kernels/heuristics/sm90.hpp``, the commit pinned by
  vLLM 0.24.0 ``cmake/external_projects/deepgemm.cmake``).

Both heuristics are reproduced here as static source facts; nothing is
fitted.  ``owns_select_gemm_launch`` declares this module's request domain
(a GEMM whose lowering declares block-FP8 ``weight_quantization``); the
dispatcher also routes ordinary injected BF16/FP16 GEMMs to the observed
H200 cuBLASLt table. Analytical ordinary GEMMs remain projected.
"""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Mapping, Optional

import numpy as np

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_WEIGHT_QUANTIZATION,
    DEEPGEMM_1D2D_ALGORITHM,
    DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
    TRTLLM_SWAP_AB_ALGORITHM,
)
from src.kernel.launch_config.gemm_launch_selection import (
    ObservedGridLaunchSelection,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.gemm_workload import (
    analytical_cublas_tiling,
    gemm_launch_selection_key,
    cublaslt_main_block_threads,
)
from src.kernel.launch_config.launch_program import GpuCtaRasterProgram
from src.kernel.launch_config.selection_primitives import (
    ceil_div,
    launch_policy,
    require_dtype,
    require_shape,
    require_variant,
)
from src.kernel.model_contract import (
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
)
from src.kernel.launch_config.tiling_spec import parse_nvjet_gemm_symbol


def cublaslt_narrow_tensor_instruction_parameters(runtime_name: str) -> dict[str, float]:
    """Bind only instruction shapes observed in retained NVJet source entries.

    The vertical 64/128/256-by-8 entries use HGMMA.64x8x16.F32. The
    horizontal 64-by-32 and 64-by-64 entries are independently exported by
    nvjet_instruction_counter_retry_20261107 and use N32/N64 respectively;
    their dynamic counts also match the complete physical tile work.
    Cluster size and split-K change ownership, not these mathematical cores.
    Other token widths/layouts remain unbound; a tile width alone does not
    generally certify an instruction width.
    """
    symbol = parse_nvjet_gemm_symbol(runtime_name)
    if (symbol is None or symbol.types != "hsh" or symbol.tile_k != 64
            or symbol.layout != "TNT" or symbol.cooperative is not None):
        return {}
    if (symbol.tile_m == 8 and symbol.tile_n in (64, 128, 256)
            and symbol.raster == "v"):
        instruction_n = 8
    elif (symbol.tile_m in (32, 64) and symbol.tile_n == 64
            and symbol.raster == "h"):
        instruction_n = symbol.tile_m
    else:
        return {}
    return {"tensor_instruction_m": 64.0, "tensor_instruction_n": float(instruction_n),
        "tensor_instruction_k": 16.0}


def cublaslt_uniform_split_k_parameters(
        runtime_name: str, *, split_k: int) -> dict[str, float]:
    """Bind retained 64/128/256-by-8 cores' equal, rounded-up K slices.

    All three exported producer loops use the same uniform C0 slice bound;
    duration-free driver arguments and the 128-by-8 HGMMA count agree on
    S * ceil(K_tiles / S), including zero-filled Tensor work. Instruction
    geometry alone does not certify the partition policy of other cores.
    """
    if type(split_k) is not int or split_k <= 0:
        raise ValueError("uniform split-K binding requires a positive split count")
    symbol = parse_nvjet_gemm_symbol(runtime_name)
    if (split_k == 1 or symbol is None or not symbol.split_k or symbol.types != "hsh"
            or symbol.tile_m != 8 or symbol.tile_n not in (64, 128, 256)
            or symbol.tile_k != 64
            or symbol.layout != "TNT" or symbol.raster != "v"
            or symbol.cooperative is not None):
        return {}
    return {"uniform_split_k_tiles": 1.0}


def cublaslt_fp32_partial_reducer_source_parameters(
        *, element_count: int, split_k: int,
        grid: tuple[int, int, int], block: tuple[int, int, int]) -> dict[str, float]:
    """Read/issue lower bounds for the retained SM90 FP32-partial reducer.

    A warp owns 32 contiguous output columns; threadIdx.y selects one of
    sixteen rows. Row loops advance by grid.y*16. Four 32-byte sectors
    carry each warp's FP32 partial. The issue bound counts only loads,
    zero-initialized additions, alpha multiplication, pack and store;
    parameter/address/control work remains explicitly unpriced.
    """
    for name, value in (("element_count", element_count), ("split_k", split_k)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"SM90 reducer {name} must be a positive integer")
    if split_k <= 1 or block != (32, 16, 1):
        raise ValueError("SM90 reducer requires split-K and block=(32,16,1)")
    if len(grid) != 3 or any(type(v) is not int or v <= 0 for v in grid) or grid[2] != 1:
        raise ValueError("SM90 reducer requires a positive 2D grid")
    columns = 32 * grid[0]
    if element_count % columns:
        raise ValueError("SM90 reducer requires complete source rows")
    rows = element_count // columns
    if grid[1] > ceil_div(rows, 16):
        raise ValueError("SM90 reducer grid contains wholly idle row CTAs")
    row_iterations = ceil_div(rows, grid[1] * 16)
    return {
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER: float(4 * split_k * row_iterations),
        SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER: float(min(rows, 16)),
        "warp_instructions_lower_bound": float((element_count // 32) * (2 * split_k + 3)),
    }

# vLLM ``_dynamic_flashinfer_deepgemm_blockscale_gemm_impl``: M < 32 -> swapAB.
_SWAP_AB_M_THRESHOLD = 32
# FlashInfer eligibility (``should_use_flashinfer_for_blockscale_fp8_gemm``).
_N_MULTIPLE = 64
_K_MULTIPLE = 128
_BLOCK_K = 128
# Shared-memory capacity constant hard-coded by both heuristics.
_SM90_SMEM_CAPACITY = 232448
_TMA_THREADS = 128


@dataclass(frozen=True)
class Fp8BlockGemmConfig:
    """One source-derived FP8 block-scaled GEMM launch (problem coordinates).

    ``tile_m``/``tile_n`` are the token/output-channel extents of one tile in
    GEMM coordinates (for swap-AB the kernel's BLOCK_N covers tokens and its
    BLOCK_M covers output channels).  ``multicast_on_a`` is the DeepGEMM
    ``kIsTMAMulticastOnA`` flag (activation multicast across the cluster).
    """

    algorithm: str
    tile_m: int
    tile_n: int
    stages: int
    cluster: int
    multicast_on_a: bool
    math_threads: int
    smem_bytes: int
    swizzle_cd: int


def _trtllm_swap_ab_smem(stages: int, k: int, block_m: int, block_n: int) -> int:
    smem_d = block_n * block_m * 2
    smem_a_per_stage = block_m * _BLOCK_K
    smem_scales_a = ceil_div(k, _BLOCK_K) * 4
    smem_b_per_stage = block_n * _BLOCK_K
    smem_scales_b = ceil_div(block_n * 4, 128) * 128
    smem_barrier = stages * 8 * 2
    return (smem_d + stages * smem_a_per_stage + stages * smem_scales_b
            + stages * smem_b_per_stage + ceil_div(smem_scales_a, 8) * 8
            + smem_barrier)


def trtllm_swap_ab_config(m: int, n: int, k: int,
                          num_sms: int) -> Fp8BlockGemmConfig:
    """``get_best_gemm_config(N, M, K, 1, num_sms, false, swap_ab=true)``."""

    shape_m, shape_n = n, m  # the heuristic is called with swapped M/N
    block_m = 64 if shape_m <= 64 else 128

    def waves(bm, bn):
        return ceil_div(ceil_div(shape_m, bm) * ceil_div(shape_n, bn), num_sms)

    def last_wave_util(bm, bn):
        util = (ceil_div(shape_m, bm) * ceil_div(shape_n, bn)) % num_sms
        return num_sms if util == 0 else util

    best_m = best_n = 0
    for block_n in range(16, 129, 8):
        if best_m == 0:
            success = True
        elif waves(block_m, block_n) < waves(best_m, best_n):
            success = True
        elif waves(block_m, block_n) == waves(best_m, best_n):
            util = last_wave_util(block_m, block_n)
            best_util = last_wave_util(best_m, best_n)
            success = util > best_util or (
                util == best_util and (
                    block_m > best_m
                    or (block_m == best_m and block_n < best_n)))
        else:
            success = False
        if success:
            best_m, best_n = block_m, block_n
    candidates = (6, 5, 4) if 128 % best_n else (8, 7, 6, 5, 4)
    stages = smem = None
    for candidate in candidates:
        size = _trtllm_swap_ab_smem(candidate, k, best_m, best_n)
        if size <= _SM90_SMEM_CAPACITY:
            stages, smem = candidate, size
            break
    if stages is None:
        raise ValueError(
            f"TRT-LLM swap-AB FP8 GEMM has no legal stage count for K={k}")
    multicast = 1
    if (shape_n >= 1024 and shape_m % (best_m * 2) == 0
            and num_sms % 2 == 0):
        multicast = 2
    return Fp8BlockGemmConfig(
        algorithm=TRTLLM_SWAP_AB_ALGORITHM,
        tile_m=best_n, tile_n=best_m, stages=stages, cluster=multicast,
        multicast_on_a=False,
        math_threads=(best_m // 64) * 128, smem_bytes=smem,
        swizzle_cd=0)


def _swizzle_mode(block_bytes: int) -> int:
    for mode in (128, 64, 32, 16):
        if block_bytes % mode == 0:
            return mode
    raise AssertionError("DeepGEMM swizzle mode is unreachable")


def _align(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def _deepgemm_1d2d_pipeline(k: int, block_m: int, block_n: int):
    smem_cd = _align(block_m * block_n * 2, 1024)
    smem_barriers = 16 * 8 * 2
    smem_a_per_stage = block_m * _BLOCK_K
    smem_b_per_stage = block_n * _BLOCK_K
    smem_sfa_per_stage = _align(block_m * 4, 128)
    use_uniform_sfb = 1 if _BLOCK_K % block_n == 0 else 2
    smem_extra_sfb = _align(ceil_div(k, _BLOCK_K) * 4 * use_uniform_sfb, 8)
    smem_extra = smem_cd + smem_barriers + smem_extra_sfb
    smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage
    stages = min((_SM90_SMEM_CAPACITY - smem_extra) // smem_per_stage, 16)
    return smem_extra + stages * smem_per_stage, stages


def deepgemm_1d2d_config(m: int, n: int, k: int,
                         num_sms: int) -> Fp8BlockGemmConfig:
    """``get_best_config<SM90ArchSpec>`` for a Normal K-major 1D2D GEMM."""

    block_m_candidates = [64, 128]
    if m <= 16:
        block_m_candidates.append(16)
    if m <= 32:
        block_m_candidates.append(32)
    block_m_candidates.append(256)  # BF16 output supports 256
    block_n_candidates = list(range(16, 193, 16))
    candidates = []
    for cluster_m in (1, 2):
        for cluster_n in (1, 2):
            if cluster_m * cluster_n > 2:
                continue
            if num_sms % (cluster_m * cluster_n):
                continue
            for block_m in block_m_candidates:
                for block_n in block_n_candidates:
                    if (block_n > _BLOCK_K
                            and block_n % (block_n - _BLOCK_K) != 0
                            and _BLOCK_K % (block_n - _BLOCK_K) != 0):
                        continue
                    if block_m > 128 and block_n > 128:
                        continue
                    # K-major FP8 A/B swizzle is 128 B for every candidate.
                    _, stages = _deepgemm_1d2d_pipeline(k, block_m, block_n)
                    if stages < 3 or (
                            block_m * block_n < 128 * 192 and stages < 4):
                        continue
                    candidates.append((block_m, block_n, cluster_m, cluster_n))
    if not candidates:
        raise ValueError("DeepGEMM SM90 1D2D has no layout candidate")

    l2_bandwidth_per_cycle = int(min(64.0 * num_sms, 8e6 / 1.3e3))
    l1_bandwidth_per_cycle = 128 * num_sms

    def num_cycles(layout):
        block_m, block_n, cluster_m, cluster_n = layout
        num_blocks = ceil_div(m, block_m) * ceil_div(n, block_n)
        num_waves = ceil_div(num_blocks, num_sms)
        bytes_l2_ab = k * (block_m // cluster_n + block_n // cluster_m)
        bytes_l1_ab = k * (block_m + block_n)
        bytes_l1_tc = k * (max(64, block_m) + block_n) + block_m * block_n * 2
        bytes_cd = block_m * block_n * 2
        l2_cycles = ((bytes_l2_ab + bytes_cd) * num_blocks
                     // l2_bandwidth_per_cycle)
        l1_cycles = ((bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks
                     // l1_bandwidth_per_cycle)
        # C++: int64 / float, float wave efficiency, truncated to int64.
        efficiency = np.float32(num_blocks) / np.float32(num_waves * num_sms)
        cycles = int(np.float32(max(l1_cycles, l2_cycles)) / efficiency)
        if cluster_m * cluster_n > 1 and num_waves <= 1:
            return np.iinfo(np.int64).max
        return cycles

    best = candidates[0]
    best_cycles = num_cycles(best)
    for layout in candidates[1:]:
        cycles = num_cycles(layout)
        if cycles < best_cycles:
            best, best_cycles = layout, cycles
    block_m, block_n, cluster_m, cluster_n = best
    smem, stages = _deepgemm_1d2d_pipeline(k, block_m, block_n)
    return Fp8BlockGemmConfig(
        algorithm=DEEPGEMM_1D2D_ALGORITHM,
        tile_m=block_m, tile_n=block_n, stages=stages,
        cluster=cluster_m * cluster_n, multicast_on_a=cluster_n > 1,
        math_threads=128 if block_m <= 64 else 256,
        smem_bytes=smem, swizzle_cd=_swizzle_mode(block_n * 2))


def select_block_fp8_gemm_config(
        m: int, n: int, k: int, num_sms: int) -> Fp8BlockGemmConfig:
    """Return the vLLM 0.24 kernel configuration for one block-FP8 linear."""

    if n % _N_MULTIPLE or k % _K_MULTIPLE:
        raise NotImplementedError(
            "vLLM 0.24 FlashInfer/DeepGEMM block-FP8 linear requires "
            f"N % {_N_MULTIPLE} == 0 and K % {_K_MULTIPLE} == 0, got "
            f"N={n}, K={k}")
    if m < _SWAP_AB_M_THRESHOLD:
        return trtllm_swap_ab_config(m, n, k, num_sms)
    return deepgemm_1d2d_config(m, n, k, num_sms)


def owns_select_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Own block-FP8 linears and ordinary injected dense observations."""

    if "bf16_gemm_model" in attributes:
        return True
    if "weight_quantization" not in attributes:
        return (variant in ("gemm", "gemv")
                and launch_policy(attributes) == "injected_launch")
    if attributes["weight_quantization"] != BLOCK_FP8_WEIGHT_QUANTIZATION:
        raise ValueError(
            "unsupported weight_quantization "
            f"{attributes['weight_quantization']!r}")
    return True


def _require_block_fp8_output_storage(attributes):
    if ("output_dtype" in attributes
            and attributes["output_dtype"] not in (None, "bf16")):
        raise ValueError("known packed block-FP8 GEMM launch requires BF16 output storage")


def _require_real_fp16_runtime(shape, dtype, variant, attributes, hardware):
    # KernelSpec already checks hardware_id before its no-HWSpec geometry
    # replay. The exact source profile, dtype and shape checks below still
    # apply; None does not choose another device or relax program scope.
    if hardware is not None:
        if hardware.hardware_id != "nvidia_h200":
            raise ValueError("real FP16 GemmEx geometry requires H200 HWSpec")
        if hardware.num_compute_units != 132:
            raise NotImplementedError("real FP16 GemmEx binds the 132-SM H200")
    require_dtype(dtype, ("fp16",))
    require_variant(variant, ("gemm", "gemv"), "real FP16 single-kernel GemmEx")
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    if variant == "gemv" and m != 1:
        raise ValueError("real FP16 gemv requires one logical input row")
    required = {"model_execution_version", "source_gemm_runtime", "launch_policy"}
    if not required.issubset(attributes):
        raise ValueError("real FP16 GemmEx requires explicit source_launch_inputs runtime facts")
    if (attributes["model_execution_version"] != "vllm_0_30_0"
            or attributes["source_gemm_runtime"] != "vllm_0_30_torch_2_13_cu130_cublas13_1_1_fp16_unquantized"
            or attributes["launch_policy"] != "injected_launch"
            or "weight_quantization" in attributes):
        raise NotImplementedError("real FP16 GemmEx source/runtime identity is not bound")
    return b, m, n, k


def select_real_fp16_single_gemm_launch(shape, dtype, variant, attributes, hardware):
    """Observed GemmEx13.1.1 -> LtHSH geometry, nominal-math partial only.

    The explicit source runtime marker binds the ordinary unquantized
    FP16 F.linear caller and its contiguous MK/NK -> MN public interface.
    It is prebound by the consumer, never guessed from the environment.
    Internal CTA/K mapping and workspace meaning remain unobserved.
    """
    b, m, n, k = _require_real_fp16_runtime(shape, dtype, variant, attributes, hardware)
    if b == 1 and k == 5120 and n == 51200 and m in (1, 4):
        grid, cluster, dynamic = (4, 25, 1), (4, 1, 1), 216220
    elif (b, m, n, k) == (1, 1, 151936, 5120):
        grid, cluster, dynamic = (2, 66, 1), (2, 1, 1), 217268
    else:
        raise NotImplementedError("real FP16 GemmEx single-call geometry is unobserved or external-reducer semantics are unbound")
    parameters = dict.fromkeys(("compulsory_traffic", "weight_tile_rows", "mte2_tile_bytes", "aiv_mixed",
        "n_major_traffic", "split_k_partial_bytes_per_element", "split_k_partial_dram_bytes_per_element"), 0.0)
    parameters.update(registers_per_thread=168.0, dynamic_shared_bytes=float(dynamic))
    return LaunchConfig("h200_vllm030_torch213_fp16_observed_gemm", grid, (384, 1, 1),
        cluster=cluster, parameters=parameters, execution_variant="gemm")


def select_real_fp16_external_gemm_launch(shape, dtype, variant, attributes, hardware, *, reducer=False):
    """Six exact observed launches with an explicit aggregate reduction model.

    Configured 2/5/10 FP32 contributors are a lightweight model assumption
    (the selected API configuration requests these split counts, compute32F), not an observation of
    effective K partitions, buffer layout or binary instructions. Consumers
    must label the estimate partial and retain both physical launches.
    """
    b, m, n, k = _require_real_fp16_runtime(shape, dtype, variant, attributes, hardware)
    if b != 1 or m not in (1, 4):
        raise NotImplementedError("real FP16 external aggregate shape is unobserved")
    if (n, k) == (10240, 5120):
        main_grid, dynamic, reducer_grid, modeled_splits = (4, 28, 1), 221444, (320, 1, 1), 2
    elif (n, k) == (5120, 8192):
        main_grid, dynamic, reducer_grid, modeled_splits = (4, 25, 1), 219348, (160, 1, 1), 5
    elif (n, k) == (5120, 25600):
        main_grid, dynamic, reducer_grid, modeled_splits = (4, 30, 1), 216204, (160, 1, 1), 10
    else:
        raise NotImplementedError("real FP16 external aggregate shape is unobserved")
    model = {"modeled_split_k": float(modeled_splits), "modeled_partial_bytes_per_element": 4.0,
             "split_k_count_observed": 0.0}
    if reducer:
        return LaunchConfig("h200_vllm030_torch213_fp16_external_aggregate_reduce",
            reducer_grid, (32, 16, 1), parameters={**model, "registers_per_thread": 32.0},
            execution_variant="gemm_reduce")
    parameters = dict.fromkeys(("compulsory_traffic", "weight_tile_rows", "mte2_tile_bytes", "aiv_mixed",
        "n_major_traffic", "split_k_partial_bytes_per_element", "split_k_partial_dram_bytes_per_element"), 0.0)
    parameters.update(model, registers_per_thread=168.0, dynamic_shared_bytes=float(dynamic))
    return LaunchConfig("h200_vllm030_torch213_fp16_external_aggregate_gemm", main_grid, (384, 1, 1),
        cluster=(4, 1, 1), parameters=parameters, execution_variant="gemm")


def select_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select an observed dense launch or the vLLM 0.24 block-FP8 launch.

    Outside ``owns_select_gemm_launch`` this selector has no rule and fails.
    """

    if "bf16_gemm_model" in attributes:
        if (attributes["bf16_gemm_model"] != "tiled_reference"
                or launch_policy(attributes) != "analytical_launch"
                or dtype != "bf16" or variant not in ("gemm", "gemv")
                or attributes["weight_dtype"] != "bf16"
                or attributes["weight_group_size"] != 0
                or attributes["has_bias"] is not False
                or "weight_quantization" in attributes
                or ("output_dtype" in attributes and attributes["output_dtype"] not in (None, "bf16"))):
            raise ValueError("tiled BF16 reference requires an analytical homogeneous unbiased GEMM")
        b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
        if b != 1:
            raise ValueError("ordinary tiled BF16 reference requires b=1")
        return _analytical_bf16_launch(b, m, n, k, "gemm")
    if not owns_select_gemm_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 GEMM selector owns block-FP8 or injected ordinary linears")
    if "source_gemm_runtime" in attributes:
        if (shape["n"], shape["k"]) in ((10240, 5120), (5120, 8192), (5120, 25600)):
            return select_real_fp16_external_gemm_launch(shape, dtype, variant, attributes, hardware)
        return select_real_fp16_single_gemm_launch(shape, dtype, variant, attributes, hardware)
    if "weight_quantization" not in attributes:
        return _select_dense_observed_launch(shape, dtype, variant, attributes)
    _require_block_fp8_output_storage(attributes)
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_variant(variant, ("gemm",), "H200 block-FP8 GEMM")
    require_dtype(dtype, ("bf16",))
    if b != 1:
        raise ValueError("H200 block-FP8 linear is a single (b=1) GEMM")
    if hardware.max_smem_per_block_bytes < _SM90_SMEM_CAPACITY:
        raise ValueError(
            "H200 block-FP8 heuristics assume 232448 B opt-in shared memory")
    num_sms = int(hardware.num_compute_units)
    if "fp8_linear_runner" in attributes:
        if attributes["fp8_linear_runner"] != "sglang_deepgemm":
            raise ValueError(
                f"unknown fp8_linear_runner {attributes['fp8_linear_runner']!r}")
        # SGLang deepgemm_w8a8_block_fp8_linear_with_fallback: DeepGEMM
        # 1d2d at every M (no FlashInfer swap-AB branch).
        if n % _N_MULTIPLE or k % _K_MULTIPLE:
            raise NotImplementedError(
                "SGLang DeepGEMM block-FP8 linear requires N % 64 == 0 and "
                "K % 128 == 0 (else Triton fallback, not derived)")
        config = deepgemm_1d2d_config(m, n, k, num_sms)
    else:
        config = select_block_fp8_gemm_config(m, n, k, num_sms)
    return LaunchConfig(
        algorithm=config.algorithm,
        grid=(num_sms, 1, 1),
        block=(_TMA_THREADS + config.math_threads, 1, 1),
        cluster=(config.cluster, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": config.tile_m, "n": config.tile_n,
                        "k": _BLOCK_K},
            pipeline_depth=config.stages,
            split_k=1,
        ),
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="gemm",
    )


def owns_select_target_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether this package selects the target-model GEMM.

    The MLA absorbed projections are strided-batched GEMMs (``torch.bmm``
    over the local heads, ``vllm/model_executor/layers/attention/
    mla_attention.py`` ``forward_mqa`` / ``_v_up_proj``); the MoE router is
    owned where vLLM dispatches its dedicated DSV3 router kernel.
    """

    if variant == "batched_gemm":
        weight_dtype = attributes["weight_dtype"]
        if weight_dtype == "bf16":
            if attributes["weight_group_size"] != 0:
                raise ValueError(
                    "H200 BF16 batched GEMM requires unquantized weights")
            return True
        # DeepSeek-V4 wo_a FP8 einsum (deepseek_v4/nvidia/ops/o_proj.py).
        if weight_dtype == "fp8":
            _require_block_fp8_output_storage(attributes)
            if attributes["weight_group_size"] != 128:
                raise ValueError(
                    "H200 FP8 batched GEMM requires 128x128 block scales")
            return True
        raise ValueError(
            "H200 batched GEMM owns only BF16 and block-scaled FP8 "
            f"weights, got {weight_dtype!r}")
    if variant == "moe_router":
        return (_uses_dsv3_router_gemm(shape, attributes)
                or _observed_sglang_router_launch(shape, attributes)
                is not None)
    return False


# vLLM 0.24 ``GateLinear`` (vllm/model_executor/layers/fused_moe/router/
# gate_linear.py): tier 1 is ``ops.dsv3_router_gemm`` on Hopper when
# (hidden, experts) is an instantiated DSV3 shape, the router weight is BF16
# without bias, and the token count is at most ``_dsv3_max_batch`` (16 on
# Hopper).  Larger token counts take tier 3 (cuBLAS ``torch.mm`` with an FP32
# output), which this package does not own.
DSV3_ROUTER_SHAPES = frozenset({(7168, 256), (6144, 256), (7168, 384)})
DSV3_ROUTER_MAX_TOKENS_HOPPER = 16
# csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu
# ``invokeRouterGemmFloatOutput``: gridDim = kNumExperts, blockDim = 128;
# each CTA reduces one expert column over the hidden dimension with SIMT FP32
# FMAs (VPT = 16 B / sizeof(bf16) = 8 elements per thread per K iteration)
# for every token of the batch, then a warp shuffle + shared-memory reduction.
DSV3_ROUTER_GEMM_ALGORITHM = "vllm_dsv3_router_gemm_float_out"
_DSV3_ROUTER_BLOCK_THREADS = 128


def _uses_dsv3_router_gemm(
        shape: Mapping[str, int], attributes: Mapping[str, object]) -> bool:
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    return ((k, n) in DSV3_ROUTER_SHAPES
            and b * m <= DSV3_ROUTER_MAX_TOKENS_HOPPER
            and attributes["weight_dtype"] == "bf16"
            and attributes["weight_group_size"] == 0)


# SGLang v0.5.19 ``MoEGate.forward`` (python/sglang/srt/models/
# deepseek_v2.py:540-563) runs the DSV3 router kernel only for <= 16 tokens
# on SM >= 90; every larger CUDA batch calls ``linear_bf16_fp32``
# (python/sglang/kernels/ops/attention/dsv4/gemm.py:144), whose default
# ``SGLANG_OPT_BF16_FP32_GEMM_ALGO="cublas"`` (srt/environ.py:1392) is
# ``torch.mm(x, W.t(), out_dtype=torch.float32)`` -- the call the H200
# grid-block-only launch selection observed.  The workload builder declares
# this runner on SGLang router nodes only, so no vLLM request can reach it.
SGLANG_ROUTER_LINEAR_RUNNER = "sglang_linear_bf16_fp32_cublas"
OBSERVED_GRID_ROUTER_ALGORITHM = "cublaslt_observed_grid:moe_router"
OBSERVED_GRID_REDUCER_ALGORITHM = "observed_gemm_aux:gemm_reduce"
# The observation's cuBLAS/CUDA revision (manifest ``software_revision``).
_OBSERVED_ROUTER_SOFTWARE_REVISION = "cuda_12.9"


@lru_cache(maxsize=1)
def _observed_launch_lut():
    return load_gemm_launch_selection(
        "nvidia_h200", software_revision=_OBSERVED_ROUTER_SOFTWARE_REVISION)


H200_DENSE_OBSERVED_ALGORITHM = "h200_cublaslt129_injected_gemm"


def _h200_v6_beta_text(beta) -> str:
    """Project one schema-normalized beta onto its v6 typed-key text.

    The producer stores an omitted beta as "" and the capture runner
    normalizes numeric 0 to ""; only those spellings are the ordinary
    identity.  Integer-valued betas project through ``str(int)`` so the
    typed key stores a canonical decimal (``1``, never ``1.0``).  A
    non-zero, non-integer, or non-numeric beta is a distinct typed fact
    and must not alias the beta=0 row.
    """

    if beta is None or beta == "" or beta == "0" or beta == 0:
        return ""
    if isinstance(beta, bool):
        # The KernelSpec boundary rejects bool; a direct selector caller
        # that bypassed the boundary must not silently treat False as the
        # ordinary identity (False == 0 in Python).
        raise ValueError("H200 v6 typed beta must not be bool")
    if isinstance(beta, int):
        return str(beta)
    if isinstance(beta, float):
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(
                "H200 v6 typed beta must be a non-negative finite number")
        if beta.is_integer():
            return str(int(beta))
        return format(beta, "g")
    # A string that survived the boundary (e.g. a direct caller passing a
    # decimal string) parses through the same canonical-decimal rule.
    try:
        value = float(beta)
    except (TypeError, ValueError):
        raise ValueError(
            f"H200 v6 typed beta is not a non-negative finite number: "
            f"{beta!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "H200 v6 typed beta must be a non-negative finite number")
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def _h200_v6_output_dtype_text(out_dtype, dtype_name: str) -> str:
    """Project one schema-normalized output_dtype onto its v6 typed-key text.

    None or "" is the homogeneous-output ordinary identity.  An explicit
    output_dtype equal to the input dtype is the same homogeneous fact and
    projects to "" so a typed row cannot be requested under two spellings.
    """

    if out_dtype is None or out_dtype == "":
        return ""
    out_text = str(out_dtype)
    if out_text == dtype_name:
        return ""
    return out_text


def _h200_v6_key_suffix_from_public(shape, attributes, dtype_name) -> tuple:
    """Project the canonical public KernelSpec GEMM fields onto the v6 suffix.

    The public contract reuses the existing KernelSpec fields: ``shape.b``
    for batch count, the three ``*_batch_stride_elements`` attributes for
    strided calls, ``output_dtype`` for mixed-output, and a schema-defined
    ``beta`` attribute whose ordinary value is 0.  No parallel public
    ``batch`` or ``strides`` payload is introduced.  The private v6 suffix
    is formed from these canonical fields; unknown keys fail closed at the
    attribute layer, not here.

    ``beta`` and ``output_dtype`` are schema-normalized at the KernelSpec
    boundary: an absent ``beta`` means the ordinary value 0 and an absent
    ``output_dtype`` means the homogeneous output.  The projection encodes
    that schema contract directly (the boundary is the single source of
    truth for the default) so a direct selector caller that bypassed the
    boundary still resolves to the ordinary identity rather than being
    rejected mid-projection.  ``dtype_name`` is the canonical input dtype
    so an explicit output_dtype equal to the input dtype projects to the
    homogeneous ("") suffix.
    """
    if "beta" in attributes:
        beta = attributes["beta"]
    else:
        beta = 0
    beta_text = _h200_v6_beta_text(beta)
    if "output_dtype" in attributes:
        out_dtype = attributes["output_dtype"]
    else:
        out_dtype = None
    out_dtype_text = _h200_v6_output_dtype_text(out_dtype, dtype_name)
    batch_value = int(shape["b"])
    if batch_value > 1:
        stride_a = attributes["a_batch_stride_elements"]
        stride_b = attributes["b_batch_stride_elements"]
        stride_c = attributes["c_batch_stride_elements"]
        stride_values = (int(stride_a), int(stride_b), int(stride_c))
    else:
        stride_values = (None, None, None)
    return (beta_text, out_dtype_text, batch_value, *stride_values)


def _dense_observed_row(shape, dtype, variant, attributes):
    dtype_name = require_dtype(dtype, ("bf16", "fp16"))
    key = gemm_launch_selection_key(shape, dtype_name, variant, attributes)
    # Typed-call rows must not silently alias an ordinary v5 row: a typed
    # identity (beta/output dtype/batch/strides) requires an exact v6 row;
    # a missing typed row is an explicit observability gap.  Ordinary calls
    # keep the published 13-field key.
    lut = _observed_launch_lut()
    # A typed fact is any non-ordinary value of the canonical public fields:
    # beta (ordinary 0), output_dtype (ordinary None), or a batched call
    # (shape.b > 1).  Ordinary calls carry beta=0, output_dtype=None, b=1.
    # The membership test encodes the schema contract directly: the
    # KernelSpec boundary always populates these fields, and a direct
    # selector caller that omitted them resolves to the schema default
    # (beta 0, output_dtype None) rather than inventing a parallel source.
    if "beta" in attributes:
        beta = attributes["beta"]
    else:
        beta = 0
    if "output_dtype" in attributes:
        out_dtype = attributes["output_dtype"]
    else:
        out_dtype = None
    batch = int(shape["b"])
    beta_is_typed = (_h200_v6_beta_text(beta) != "")
    out_dtype_is_typed = (
        _h200_v6_output_dtype_text(out_dtype, dtype_name) != "")
    typed_requested = beta_is_typed or out_dtype_is_typed or batch > 1
    if typed_requested:
        # A typed request must fully characterize its typed identity: a
        # non-ordinary beta must state its output_dtype (None means the
        # caller did not state it, not the homogeneous identity), and a
        # batched call must state its strides.  An output_dtype-only
        # request is complete because None already means homogeneous
        # output per the schema boundary; an output_dtype equal to the
        # input dtype is likewise the homogeneous fact and completes a
        # beta-only request.
        missing_typed = []
        if batch > 1:
            for name in ("a_batch_stride_elements",
                         "b_batch_stride_elements",
                         "c_batch_stride_elements"):
                if name not in attributes or attributes[name] is None:
                    missing_typed.append(name)
        if missing_typed:
            raise NotImplementedError(
                f"partial typed identity without full v6 typed key fields: "
                f"{missing_typed}")
        # The typed request is fully characterised; form the v6 suffix.
        suffix = _h200_v6_key_suffix_from_public(shape, attributes, dtype_name)
        typed_key = (*key, *suffix)
        if typed_key in lut:
            return _require_tiled_grid(lut[typed_key], typed_key)
        # Every typed fact is ordinary (beta 0, output dtype None,
        # batch 1, strides None) but the caller explicitly supplied them;
        # fall through to the ordinary alias below rather than treating this
        # as a typed miss.
        if suffix == ("", "", 1, None, None, None):
            ordinary_key = key
            if ordinary_key not in lut:
                ordinary_key = typed_key
            if ordinary_key in lut:
                return _require_tiled_grid(lut[ordinary_key], ordinary_key)
        raise NotImplementedError(
            f"uncharacterized NVIDIA H200 typed GEMM launch key "
            f"(beta={suffix[0]!r}, output_dtype={suffix[1]!r}, "
            f"batch={suffix[2]}, strides={suffix[3:]}) "
            f"for typed request {typed_key}")
    # A v6 LUT keys every ordinary call by its 19-field typed identity
    # (beta "", output dtype "", batch 1, strides null); the plain
    # 13-field request matches that alias when its typed facts are all
    # ordinary.  The M=1 output-head call is labelled "gemv" by the
    # serving stack but captured under the "gemm" variant because
    # cuBLASLt receives the same physical GEMM descriptor; the selector
    # resolves the alias only when the physical shape (b=1, m=1) and
    # all other key fields are identical.
    gemm_key = key
    if variant == "gemv" and int(shape["m"]) == 1 and batch == 1:
        gemm_key = (dtype_name, "gemm", *key[2:])
    for lookup_key in (key, gemm_key, (*key, "", "", 1, None, None, None),
                       (*gemm_key, "", "", 1, None, None, None)):
        if lookup_key in lut:
            return _require_tiled_grid(lut[lookup_key], lookup_key)
    raise NotImplementedError(f"uncharacterized NVIDIA H200 GEMM launch key {key}")


def _require_tiled_grid(row, key):
    if row.observed_grid is None:
        raise NotImplementedError("H200 dense GEMM requires an observed launch row")
    return row


def select_legacy_bf16_external_aggregate_launch(shape, dtype, variant, attributes, hardware, *, reducer=False):
    """Replay the released exact CUDA12.9 rows, with modeled3/2 FP32 partials."""
    if hardware is not None:
        if hardware.hardware_id != "nvidia_h200" or hardware.num_compute_units != 132:
            raise ValueError("legacy BF16 aggregate requires the132-SM H200 source profile")
    require_dtype(dtype, ("bf16",))
    require_variant(variant, ("gemm", "gemv"), "legacy BF16 aggregate")
    if "weight_quantization" in attributes or "source_gemm_runtime" in attributes:
        raise ValueError("legacy BF16 aggregate cannot borrow a quantized or Torch13 runtime")
    if attributes["launch_policy"] != "injected_launch":
        raise NotImplementedError("legacy BF16 aggregate requires the published injected source")
    row = _dense_observed_row(shape, dtype, variant, attributes)
    model = _legacy_bf16_external_aggregate_parameters(shape, dtype, attributes, row)
    if model is None:
        raise NotImplementedError("legacy BF16 aggregate is confined to released published source shapes")
    if reducer:
        return select_gemm_reducer_launch(shape, dtype, variant, attributes, hardware)
    return _select_dense_observed_launch(shape, dtype, variant, attributes)




def _legacy_bf16_external_aggregate_parameters(shape, dtype, attributes, row):
    """Configured split count as an explicit lightweight model, not K mapping."""
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    if dtype != "bf16" or b != 1:
        return None
    if "beta" in attributes and attributes["beta"] not in (None, "", "0", 0):
        return None
    if "output_dtype" in attributes and attributes["output_dtype"] not in (None, "bf16"):
        return None
    if (m, n, k) == (12, 4608, 7168):
        modeled_splits = 3
    elif k == 7168 and n == 20480 and m in (1, 2, 4, 6, 8):
        modeled_splits = 2
    elif (m, n, k) in ((1, 2560, 4096), (1, 4096, 4096)):
        modeled_splits = 2
    elif (m, n, k) in ((1, 5120, 8192), (1, 2560, 9728), (1, 5120, 25600)):
        modeled_splits = 5
    elif (m, n, k) in ((1, 4096, 12288), (1, 4096, 14336)):
        modeled_splits = 3
    else:
        return None
    if (row.reduction != "fp32_external" or row.observed_grid is None
            or row.observed_grid.observability != "grid_block_only"):
        raise ValueError("legacy BF16 aggregate requires its published grid-only FP32 external row")
    return {"modeled_split_k": float(modeled_splits),
            "modeled_partial_bytes_per_element": 4.0,
            "split_k_count_observed": 0.0}


def _select_dense_observed_launch(shape, dtype, variant, attributes):
    row = _dense_observed_row(shape, dtype, variant, attributes)
    observation = row.observed_grid
    partial_bytes = 4.0 if row.reduction == "fp32_external" else 0.0
    if row.reduction not in ("none", "fp32_external"):
        raise NotImplementedError("H200 dense reducer kind is unsupported")
    if observation.observability == "tiled_grid":
        # Only a tiled-grid observation may carry tile/pipeline facts;
        # grid-block-only rows keep them unknown rather than invented.
        tiling = TilingSpec(
            tile_shape={"m": row.tile_m, "n": row.tile_n, "k": row.tile_k},
            pipeline_depth=row.pipeline_depth, split_k=row.split_k)
    else:
        # A grid-block-only observation (the v6 candidate rows) records the
        # true main/reducer geometry but not the tile/pipeline/split facts.
        # The split count, tile, and pipeline depth stay unknown (None); the
        # partial traffic is declared at its observed FP32-external width.
        # The public prediction bridge for these rows is Goal 05's delivered
        # no-tile path (commit 99bff0446); this selector keeps the real
        # unknown values rather than fabricating a placeholder tile.
        tiling = None
    model = _legacy_bf16_external_aggregate_parameters(shape, dtype, attributes, row)
    return LaunchConfig(
        algorithm=(H200_DENSE_OBSERVED_ALGORITHM if model is None else
                   "h200_cublaslt129_bf16_external_aggregate_gemm"),
        grid=observation.grid, block=(observation.block_threads, 1, 1),
        tiling=tiling,
        parameters={
            "compulsory_traffic": 0.0, "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0, "aiv_mixed": 0.0, "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": partial_bytes,
            "split_k_partial_dram_bytes_per_element": 0.0,
            **({} if model is None else model),
        },
        execution_variant="gemm",
    )


def _observed_sglang_router_launch(
        shape: Mapping[str, int], attributes: Mapping[str, object],
        ) -> ObservedGridLaunchSelection | None:
    """Return the observed launch of an SGLang cuBLAS router GEMM, if any.

    A request outside the observed rows keeps the projected analytical
    launch (``cublas_analytical_gemm:moe_router``), whose name states that no
    launch was observed.
    """

    if "router_linear_runner" not in attributes:
        return None
    if attributes["router_linear_runner"] != SGLANG_ROUTER_LINEAR_RUNNER:
        raise ValueError(
            "unknown router_linear_runner "
            f"{attributes['router_linear_runner']!r}")
    if _uses_dsv3_router_gemm(shape, attributes):
        return None
    if (attributes["weight_dtype"] != "bf16"
            or attributes["weight_group_size"] != 0):
        raise ValueError("SGLang cuBLAS router requires unquantized BF16")
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    key = ("bf16", "gemm", 1, b * m, n, k, "row_major", "row_major",
           "row_major", False, True, "none", False)
    lut = _observed_launch_lut()
    # The canonical v5 artifact stores router rows under the plain 13-field
    # key with output_dtype="fp32" recorded in the launch fields.  A v6
    # artifact stores them under the 19-field typed key with
    # output_dtype_key="fp32".  Try the v6 key first, then fall back to the
    # v5 13-field key so both eras remain loadable.
    v6_router_key = (*key, "", "fp32", 1, None, None, None)
    if v6_router_key in lut:
        selection = lut[v6_router_key].observed_grid
    elif key in lut:
        selection = lut[key].observed_grid
    else:
        return None
    if selection is None or selection.output_dtype != "fp32":
        raise ValueError(
            "H200 router launch row must be an FP32-output observed grid")
    return selection


def _select_observed_router_launch(
        selection: ObservedGridLaunchSelection) -> LaunchConfig:
    return LaunchConfig(
        algorithm=OBSERVED_GRID_ROUTER_ALGORITHM,
        grid=selection.grid,
        block=(selection.block_threads, 1, 1),
        # Neither the tile nor the split count is observed.
        tiling=None,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 4.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
            "split_k_count_observed": 0.0,
        },
        execution_variant="moe_router",
    )


def owns_select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Own reducers for observed dense linears and SGLang routers."""

    return ((variant in ("gemm", "gemv")
             and "weight_quantization" not in attributes
             and launch_policy(attributes) == "injected_launch")
            or (variant == "moe_router"
                and _observed_sglang_router_launch(shape, attributes) is not None))


def select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig | None:
    """Return the observed ``cublasLt::splitKreduce_kernel`` launch."""

    if "source_gemm_runtime" in attributes:
        if (shape["n"], shape["k"]) in ((10240, 5120), (5120, 8192), (5120, 25600)):
            return select_real_fp16_external_gemm_launch(shape, dtype, variant, attributes, hardware, reducer=True)
        select_real_fp16_single_gemm_launch(shape, dtype, variant, attributes, hardware)
        return None
    if variant in ("gemm", "gemv"):
        row = _dense_observed_row(shape, dtype, variant, attributes)
        if row.reduction == "none":
            return None
        observation = row.observed_grid
        if row.reduction != "fp32_external" or observation.reducer_grid is None:
            raise ValueError("H200 dense launch lost its FP32 reducer")
        model = _legacy_bf16_external_aggregate_parameters(shape, dtype, attributes, row)
        return LaunchConfig(
            algorithm=("logical_gemm_aux:gemm_reduce" if model is None else
                       "h200_cublaslt129_bf16_external_aggregate_reduce"),
            grid=(*observation.reducer_grid, 1),
            block=(*observation.reducer_block, 1),
            parameters={"split_k_partial_bytes_per_element": 4.0,
                        "split_k_partial_dram_bytes_per_element": 0.0,
                        **({} if model is None else model)},
            execution_variant="gemm_reduce")
    selection = (
        _observed_sglang_router_launch(shape, attributes)
        if variant == "moe_router" else None)
    if selection is None:
        raise ValueError(
            "H200 reducer selector only owns observed SGLang router launches")
    return LaunchConfig(
        algorithm=OBSERVED_GRID_REDUCER_ALGORITHM,
        grid=(*selection.reducer_grid, 1),
        block=(*selection.reducer_block, 1),
        tiling=None,
        parameters={
            "split_k_partial_bytes_per_element": 4.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
            "split_k_count_observed": 0.0,
        },
        execution_variant="gemm_reduce",
    )


def _select_dsv3_router_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str]) -> LaunchConfig:
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_dtype(dtype, ("bf16", "fp32"))
    if k % (8 * _DSV3_ROUTER_BLOCK_THREADS):
        raise ValueError("dsv3_router_gemm needs whole K iterations")
    return LaunchConfig(
        algorithm=DSV3_ROUTER_GEMM_ALGORITHM,
        grid=(n, 1, 1),
        block=(_DSV3_ROUTER_BLOCK_THREADS, 1, 1),
        tiling=None,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="moe_router",
    )


def select_target_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select one strided-batched BF16 GEMM launch.

    A strided-batched GEMM is ``b`` independent problems, each with its own A
    and B operand: an output tile can never span two batch problems, so the
    launch holds the tiles of one (m, n) problem once per batch (cuBLAS puts
    the batch index on a grid axis).  The per-problem tile is the analytical
    cuBLAS tiler the projected package uses for one GEMM, applied to the
    per-batch rows ``m`` instead of the folded ``b * m``.
    """

    if not owns_select_target_gemm_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 target GEMM selector only owns strided-batched GEMMs and "
            "the vLLM DSV3 router GEMM")
    if variant == "moe_router":
        observed = _observed_sglang_router_launch(shape, attributes)
        if observed is not None:
            require_dtype(dtype, ("bf16", "fp32"))
            return _select_observed_router_launch(observed)
        return _select_dsv3_router_gemm_launch(shape, dtype)
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_variant(variant, ("batched_gemm",), "H200 batched GEMM")
    if launch_policy(attributes) != "analytical_launch":
        raise NotImplementedError(
            "H200 batched GEMM has no captured launch selection")
    if attributes["weight_dtype"] == "fp8":
        # DeepSeek-V4 wo_a einsum (deepseek_v4/nvidia/ops/o_proj.py
        # deep_gemm_fp8_o_proj, SM90 recipe (1, 128, 128)): one DeepGEMM
        # SM90 block-scaled problem per o_group, configured by the same
        # source heuristic as the dense block-FP8 linear; the activation
        # quantisation is the separate fused_inv_rope_fp8_quant kernel.
        require_dtype(dtype, ("bf16",))
        if (attributes["batch_layout"] != "einsum_bhr_hdr_bhd"
                or attributes["c_batch_stride_elements"] != n):
            raise ValueError("H200 FP8 einsum requires token-major bhr,hdr->bhd output with group stride n")
        if hardware.max_smem_per_block_bytes < _SM90_SMEM_CAPACITY:
            raise ValueError(
                "H200 FP8 einsum heuristics assume 232448 B opt-in shared "
                "memory")
        config = select_block_fp8_gemm_config(
            m, n, k, int(hardware.num_compute_units))
        return LaunchConfig(
            algorithm=DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
            grid=(int(hardware.num_compute_units), 1, 1),
            block=(_TMA_THREADS + config.math_threads, 1, 1),
            cluster=(config.cluster, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": config.tile_m, "n": config.tile_n,
                            "k": _BLOCK_K},
                pipeline_depth=config.stages,
                split_k=1,
            ),
            parameters={
                "compulsory_traffic": 0.0,
                "weight_tile_rows": 0.0,
                "mte2_tile_bytes": 0.0,
                "aiv_mixed": 0.0,
                "n_major_traffic": 0.0,
                "split_k_partial_bytes_per_element": 0.0,
                "split_k_partial_dram_bytes_per_element": 0.0,
            },
            execution_variant="batched_gemm",
            source_program=GpuCtaRasterProgram("none", 0, False, 1),
        )
    require_dtype(dtype, ("bf16",))
    return _analytical_bf16_launch(b, m, n, k, "batched_gemm")


def _analytical_bf16_launch(b, m, n, k, variant):
    """Frozen analytical tile program; no vendor geometry or split claim.

    Ordinary and strided-batched references share the existing static tiler.
    Every problem keeps its real rows and a separate batch grid axis.
    """
    tile_m, tile_n, tile_k, depth = analytical_cublas_tiling(m, n, k)
    return LaunchConfig(
        algorithm=f"cublas_analytical_gemm:{variant}",
        grid=(ceil_div(m, tile_m) * ceil_div(n, tile_n), b, 1),
        block=(cublaslt_main_block_threads(tile_m, tile_n), 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=depth,
            split_k=1,
        ),
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant=variant,
        # No vendor raster map was captured: the traffic model rasters each
        # batch problem's tile grid directly.
        source_program=GpuCtaRasterProgram("none", 0, False, 1),
    )
