"""Complete-spec GEMM/GEMV physical prediction."""

from src.kernel.prediction_cache import runtime_lru_cache

import dataclasses
import math
from typing import Tuple

from src.kernel.bank_stream_model import bank_stream_ns, matmul_stream
from src.kernel.formula import PerformanceFormula
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import DTYPE_BYTES, KernelKind, KernelSpec
from src.kernel.launch_config.launch_program import (
    CannGemmLaunchProgram,
    CannGemmTilePipelineProgram,
    GpuCtaRasterProgram,
)
from src.kernel.launch_config.cann_gemm_schedule import (
    cann_gemm_persistent_descriptor_trace,
    cann_gemm_source_tile_assignments,
    cann_gemm_source_tile_order,
    derive_cann_matmul_v3_l2_region_schedule,
)
from src.kernel.launch_config.tiling_spec import (
    gemm_logical_workgroups,
)
from src.kernel.cache_traffic_sim import (
    GemmSourceOrderTraffic,
    simulate_gemm_n_major,
    simulate_gemm_source_order_traffic,
    simulate_gemm_tile_program,
    simulate_cann_gemm_persistent_stateful_dual_accounted,
)
from src.kernel.predictors.gemm_block_fp8 import (
    block_fp8_linear_logical_work,
    is_block_fp8_gemm,
    predict_block_fp8_gemm,
    roofline_block_fp8_gemm,
)
from src.kernel.predictors.common import (
    _with_graph_regime,
    FULL_RESOURCE_OCCUPANCY,
    PredictionInput,
    attach_pipeline_work_plan,
    attach_structured_contract,
    attach_uniform_pipeline,
    bind_pipeline_stage_template,
    append_npu_write_path,
    build_forward_demand,
    build_aggregate_gpu_input,
    gpu_memory_source,
    gpu_partial_limitations,
    npu_launch_floor_contract,
    prediction_from_ns,
    roofline_ns,
    split_global_memory_paths,
    validated_l2_miss_rate,
)
from src.kernel.model_contract import (
    AnalyticalDependencyPath,
    CannGroupedMatmulEnvelopeDemand,
    CannSingleCoreSplitKEnvelopeDemand,
    ComputeServiceDemand,
    MemoryAccessClassDemand,
    MemoryAccessClassSignature,
    MemoryExactRequestDemand,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    PhaseMemoryDemand,
    PipelineBuffer,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
)


def _unquantized_matmul_math(shape, dtype, element_bytes, operand_prefix):
    """Ideal 2BMNK arithmetic and unique typed A/B/C operand payloads."""
    b, m, n, k = (shape[name] for name in ("b", "m", "n", "k"))
    return {"mathematical_tensor_flops": 2 * b * m * n * k,
        "operands": tuple({"role": operand_prefix + "_" + role, "dtype": dtype,
            "elements": elements, "bytes": element_bytes * elements, "access": access}
            for role, elements, access in (
                ("a", b * m * k, "read"), ("b", b * k * n, "read"),
                ("c", b * m * n, "write")))}


def _bf16_bmm_math(shape):
    return _unquantized_matmul_math(shape, "bf16", 2, "bmm")


def packed_fp4_source_logical_work(kernel: KernelSpec):
    """Declared packed weights and quantized matrix input, without a launch.

    Matrix rows may describe ordinary batches or ragged experts sharing K/N.
    The source contract conserves rows; each matrix's weights are read once.
    Conversion to FP8 is a register-level mathematical operand conversion,
    never a resident or global-memory FP8 weight copy.
    """
    if (kernel.kind is not KernelKind.GEMM or kernel.variant != "gemm"
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] != "packed_fp4_source_v1"
            or kernel.launch is not None or kernel.execution_support != "unsupported"):
        raise ValueError("packed-FP4 work requires its explicit unbound source program")
    b, m, n, k = (kernel.shape[name] for name in ("b", "m", "n", "k"))
    rows = b * m
    count = kernel.attributes["weight_matrix_count"]
    compute_format = kernel.attributes["compute_format"]
    if compute_format == "fp8":
        input_values = rows * k
        input_scales = 4 * rows * ((k + 127) // 128)
        compute_role = "compute_tensor_fp8"
    elif compute_format == "fp4":
        input_values = rows * k // 2
        input_scales = rows * (k // 32)
        compute_role = "compute_tensor_fp4"
    else:
        raise AssertionError("validated packed-FP4 compute format disappeared")
    activation_bytes = input_values + input_scales
    input_bytes = DTYPE_BYTES[kernel.dtype] * rows * k
    if kernel.dtype == "fp8":
        input_bytes += 4 * rows * ((k + 127) // 128)
    quant_bytes = (0 if kernel.attributes["input_quantized"] else input_bytes + activation_bytes)
    weight_values = count * k * n // 2
    weight_scales = count * n * (k // 32)
    output_bytes = DTYPE_BYTES[kernel.attributes["output_dtype"]] * rows * n
    return {"mathematical_tensor_flops": 2 * rows * n * k,
        "compute_role": compute_role, "matrix_row_counts": kernel.attributes["matrix_row_counts"],
        "weight_matrix_count": count, "packed_weight_value_bytes": weight_values,
        "weight_scale_bytes": weight_scales, "activation_value_bytes": input_values,
        "activation_scale_bytes": input_scales, "output_bytes": output_bytes,
        "quantization_logical_bytes": quant_bytes,
        "matrix_logical_bytes": activation_bytes + weight_values + weight_scales + output_bytes,
        "limitations": ("user_declared_FP4_storage_and_compute_format_scenario_not_native_provider",
            "group32_E8M0_weight_scales_explicit_not_observed_checkpoint_layout",
            "no_resident_FP8_weight_copy_or_global_weight_dequantization_tensor",
            "logical_operand_bytes_not_observed_DRAM_transactions_or_cache_hits",
            "quantized_input_flag_declares_compute_format_storage_not_numerical_equivalence",
            "GPU_vector_quantization_requires_explicit_memory_only_approximation")}


def unquantized_fp16_gemm_logical_work(kernel: KernelSpec):
    return _unquantized_16bit_gemm_logical_work(kernel, "fp16", "unquantized_fp16_flops_bytes_v1")


def unquantized_bf16_gemm_logical_work(kernel: KernelSpec):
    return _unquantized_16bit_gemm_logical_work(kernel, "bf16", "unquantized_bf16_flops_bytes_v1")


def _unquantized_16bit_gemm_logical_work(kernel: KernelSpec, source_dtype, program):
    """Homogeneous FP16 dense matmul math without a provider or launch.

    Reuse the same useful matmul operation and payload counts as ideal BF16
    BMM. This does not bind library layouts, scratch or physical traffic.
    """
    if (kernel.kind is not KernelKind.GEMM or kernel.variant != "gemm"
            or kernel.dtype != source_dtype or kernel.shape["b"] != 1
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] != program
            or kernel.attributes["weight_dtype"] != source_dtype
            or kernel.attributes["weight_group_size"] != 0
            or kernel.attributes["has_bias"] is not False
            or kernel.attributes["output_dtype"] not in (None, source_dtype)
            or kernel.execution_support != "unsupported" or kernel.launch is not None):
        raise ValueError("FP16 GEMM logical work requires homogeneous unquantized unbound bias-free FP16 GEMM")
    return {**_unquantized_matmul_math(kernel.shape, source_dtype, 2, "gemm"),
        "limitations": (
            "ideal_mathematical_FLOPs_and_logical_operand_payload_only",
            "logical_tensor_dtypes_do_not_bind_accumulation_rounding_or_numerical_equivalence",
            "library_provider_compiler_launch_internal_workspace_and_physical_traffic_unbound")}


def bf16_batched_gemm_logical_work(kernel: KernelSpec):
    """Ideal BF16 A/B/C payload and2BMNK math, without a library launch.

    The strided canonical footprint is not a claim about the physical view's
    leading strides. Source views can be transposed or interleaved, with no
    packing copy added to this mathematical baseline.
    """
    if (kernel.kind is not KernelKind.GEMM or kernel.variant != "batched_gemm"
            or kernel.dtype != "bf16" or kernel.attributes["weight_dtype"] != "bf16"
            or kernel.attributes["weight_group_size"] != 0
            or kernel.execution_support != "unsupported" or kernel.launch is not None):
        raise ValueError("BF16 BMM logical work requires unquantized unbound BF16 batched GEMM")
    return {**_bf16_bmm_math(kernel.shape), "limitations": (
        "ideal_mathematical_FLOPs_and_logical_operand_payload_only",
        "source_views_not_packed_physical_strides_or_materialized_copies",
        "library_provider_compiler_launch_and_internal_workspace_unbound")}


def kimi_k3_nope_bmm_logical_work(kernel: KernelSpec):
    """BF16 NoPE absorption/value BMM math without a packed-stride claim.

    vLLM0.30 kimi_k3/nvidia/mla.py transposes query/head and output views.
    Constructor/postload weight views and backend latent output layouts do
    not establish a generic packed batch ABI or an actual library launch.
    Only complete logical A/B/C tensor extents and2BMNK FLOPs are assigned.
    """
    if (kernel.kind is not KernelKind.GEMM or kernel.variant != "kimi_k3_nope_bmm"
            or kernel.execution_support != "unsupported" or kernel.launch is not None):
        raise ValueError("K3 NoPE BMM requires an unbound source KernelSpec")
    return {
        "source": "vllm/models/kimi_k3/nvidia/mla.py:KimiK3MLAAttention._attention/_v_up_proj",
        "source_position": kernel.attributes["source_position"],
        **_bf16_bmm_math(kernel.shape),
        "limitations": (
            "source_TP1_BF16_BMM_not_actual_backend_or_provider_execution",
            "transpose_views_not_materialized_copies_or_packed_batch_stride_evidence",
            "logical_operand_bytes_not_observed_dram_cache_hits_or_internal_library_temporaries",
            "library_kernel_compiler_launch_and_physical_tensor_layout_binding_unavailable"),
    }


_MCBLAS_GEMM_ALGORITHMS = frozenset({"metax_mcblas_gemm"})

_CANN_MATMUL_V3_ASW_65536_ALGORITHM = "cann_matmul_v3_asw_65536"
_CANN_OPERAND_A_L2_HIT_SERVICE = "cann_grouped_moe_operand_a_l2_hit"
_CANN_OPERAND_B_L2_HIT_SERVICE = "cann_grouped_moe_operand_b_l2_hit"
_CANN_OPERAND_A_DRAM_MISS_SERVICE = "cann_grouped_moe_operand_a_dram_miss"
_CANN_OPERAND_B_DRAM_MISS_SERVICE = "cann_grouped_moe_operand_b_dram_miss"
_CANN_SPLIT_K_MTE2_GROUP_SERVICE = (
    "aic_mte2_nd2nz_33copy_large_stride_group_completion_24_workers")
_CANN_SPLIT_K_SOURCE_QUEUE_FLOOR = (
    "mechanism:cann_deterministic_split_k_source_queue_completion")
_CANN_NBUFFER33_MTE1_PAIR_SERVICE = (
    "cann_nbuffer33_l1_to_l0_pair_completion")
_CANN_FIXPIPE_FINITE_SERVICE_BY_GEOMETRY = {
    (16, 176, 22): "aic_fixpipe_bf16_finite_stream_16x176_22_workers",
    (32, 256, 16): "aic_fixpipe_bf16_finite_stream_32x256_16_workers",
    (32, 256, 20): "aic_fixpipe_bf16_finite_stream_32x256_20_workers",
    (32, 256, 22): "aic_fixpipe_bf16_finite_stream_32x256_22_workers",
    (48, 256, 22): "aic_fixpipe_bf16_finite_stream_48x256_22_workers",
    (64, 256, 20): "aic_fixpipe_bf16_finite_stream_64x256_20_workers",
    (64, 256, 22): "aic_fixpipe_bf16_finite_stream_64x256_22_workers",
    (128, 256, 24): "aic_fixpipe_bf16_finite_stream_128x256_24_workers",
}
_SM80_STAGED_LDGSTS_L2_READ_ROLE = "staged_ldgsts_l2_read"
_SM80_K32_DEPTH6_LDGSTS_L2_READ_ROLE = (
    "staged_ldgsts_k32_depth6_l2_read")
_SM80_DEPTH3_OPERAND_L2_HIT_SERVICE = (
    "sm80_depth3_operand_l2_hit")
_SM80_DEPTH3_OPERAND_DRAM_MISS_SERVICE = (
    "sm80_depth3_operand_dram_miss")
_FINITE_ASYNC_OPERAND_L2_HIT_SERVICE = (
    "sm80_64x64_depth6_operand_l2_hit")
_FINITE_ASYNC_OPERAND_DRAM_MISS_SERVICE = (
    "sm80_64x64_depth6_operand_dram_miss")
_SM80_DEPTH3_ADDITIONAL_L2_ARRIVALS_PER_LAUNCH = 35

# CUDA 12.2 libcublasLt.so.12, extracted SM80 cubin SHA256:
# b3ca54942b099f2b822873135d8dd32c37079f804ab1ff6e2feff9dd2f48036b
_SM80_128X128_K32_DEPTH5_SPLIT1_FIXED_NON_HMMA_WARP_INSTRUCTIONS = 4624
_SM80_128X128_K32_DEPTH5_SPLIT2_FIXED_NON_HMMA_WARP_INSTRUCTIONS = 5446
_SM80_128X128_K32_DEPTH5_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE = 228

# CUDA 12.2 libcublasLt.so.12 depth-three split symbols, extracted SM80
# cubin SHA256: b3ca54942b099f2b822873135d8dd32c37079f804ab1ff6e2feff9dd2f48036b
# and 06008e27431fd0cef94019bd3a00b552eedd883a689a619310e8ddae706cb226.
# See the matching cuBLASLt depth-three SASS reference.
_SM80_128X256_K64_DEPTH3_ORDERED_EPILOGUE_WARP_INSTRUCTIONS = 8256
_SM80_128X256_K64_DEPTH3_FIRST_EPILOGUE_SAVING_WARP_INSTRUCTIONS = 112
_SM80_128X256_K64_DEPTH3_SPLIT2_FIXED_NON_HMMA_WARP_INSTRUCTIONS = 4744
_SM80_128X256_K64_DEPTH3_SPLIT2_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE = 688
_SM80_64X64_K64_DEPTH5_FIXED_NON_HMMA_WARP_INSTRUCTIONS = 3076
_SM80_64X128_K64_DEPTH6_FIXED_NON_HMMA_WARP_INSTRUCTIONS = 4880


@dataclasses.dataclass(frozen=True)
class _CannNpuGemmPipelineGeometry:
    """One rectangular AIC GEMM worker program derived from physical limits."""

    worker_count: int
    worker_m_tiles: int
    worker_n_tiles: int
    output_tiles_per_worker: int
    k_tiles: int
    a_l1_k_tiles: int
    b_l1_k_tiles: int
    k_slabs: int
    a_descriptors_per_worker: int
    b_descriptors_per_worker: int
    a_descriptor_bytes: int
    b_descriptor_bytes: int


@dataclasses.dataclass(frozen=True)
class _CannDeterministicSplitKTraffic:
    """Source-derived traffic phases for one CANN split-K launch."""

    operand_l2_read_bytes: float
    operand_dram_read_bytes: float
    workspace_l2_write_bytes: float
    workspace_dram_write_bytes: float
    workspace_l2_read_bytes: float
    output_l2_write_bytes: float
    output_dram_write_bytes: float

    @property
    def l2_read_bytes(self) -> float:
        return self.operand_l2_read_bytes + self.workspace_l2_read_bytes

    @property
    def l2_bytes(self) -> float:
        return (
            self.l2_read_bytes
            + self.workspace_l2_write_bytes
            + self.output_l2_write_bytes)

    @property
    def dram_bytes(self) -> float:
        return (
            self.operand_dram_read_bytes
            + self.workspace_dram_write_bytes
            + self.output_dram_write_bytes)


def _operand_element_bytes(kernel: KernelSpec) -> Tuple[int, int, int]:
    output_bytes = int(kernel.shape["dtype_bytes"])
    if kernel.variant == "moe_router":
        return 2, 2, output_bytes
    return output_bytes, output_bytes, output_bytes


def _tile(kernel: KernelSpec) -> Tuple[int, int, int, int | None]:
    tiling = kernel.launch.tiling
    if tiling is None or not {"m", "n", "k"}.issubset(tiling.tile_shape):
        raise ValueError("tensor GEMM requires m/n/k tiling in LaunchConfig")
    return (
        int(tiling.tile_shape["m"]),
        int(tiling.tile_shape["n"]),
        int(tiling.tile_shape["k"]),
        tiling.pipeline_depth,
)
_CANN_LEGACY_L1_K_SLAB_ELEMENTS = 16 * 16


def _mcblas_exact_physical_factor(
        kernel: KernelSpec, logical_tiles: int) -> int | None:
    """Return the exact physical-CTA factor for a retained mcBLAS launch."""

    tiling = kernel.launch.tiling
    if (kernel.launch.algorithm not in _MCBLAS_GEMM_ALGORITHMS
            or tiling is None or tiling.split_k is None
            or kernel.launch.grid_size < logical_tiles):
        return None
    factor, padding = divmod(kernel.launch.grid_size, logical_tiles)
    return factor if padding == 0 else None


def _mcblas_exact_subtile_factor(
        kernel: KernelSpec, logical_tiles: int) -> int:
    """Return a source-visible integer physical decomposition of each tile."""

    factor = _mcblas_exact_physical_factor(kernel, logical_tiles)
    return 1 if factor is None else factor


def _mcblas_non_split_padded_tensor_flops(
        kernel: KernelSpec) -> float | None:
    """Return fixed-tile Tensor work for an exact non-split mcBLAS grid.

    The retained mcBLAS tile is the MMA program shape, not only a scheduling
    hint.  When the physical grid is a one-to-one mapping of semantic output
    tiles, boundary CTAs still execute the complete fixed-shape MMA tile and
    predicate only their output publication.  Count that padded Tensor work;
    opaque physical decompositions and split-K programs remain unchanged.
    """

    tiling = kernel.launch.tiling
    if (kernel.launch.algorithm not in _MCBLAS_GEMM_ALGORITHMS
            or tiling is None or tiling.split_k != 1):
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if k % tile_k:
        raise ValueError(
            "mcBLAS padded Tensor work requires complete K tiles")
    logical_tiles = gemm_logical_workgroups(
        b, m, n, tile_m, tile_n, tiling.split_k)
    if _mcblas_exact_physical_factor(kernel, logical_tiles) != 1:
        return None
    return float(2 * logical_tiles * tile_m * tile_n * k)


def _kernel_logical_workgroups(
        kernel: KernelSpec, tile_m: int, tile_n: int, split_k: int) -> int:
    """Return the logical tile jobs of one GEMM kernel.

    A strided-batched GEMM is ``b`` independent problems with their own
    operands, so its tiles are counted per problem; every other variant folds
    ``b`` into the rows of one problem.
    """

    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    if kernel.variant == "batched_gemm":
        return b * gemm_logical_workgroups(1, m, n, tile_m, tile_n, split_k)
    return gemm_logical_workgroups(b, m, n, tile_m, tile_n, split_k)


# The vLLM DSV3 router kernel (H200 launch package,
# DSV3_ROUTER_GEMM_ALGORITHM): one SIMT CTA per expert column.
_SIMT_ROUTER_ALGORITHMS = frozenset({"vllm_dsv3_router_gemm_float_out"})


# The H200 SGLang cuBLAS router launch observed with grid, block and a
# separate FP32-partial reducer but no tile or split count
# (``cublaslt_observed_grid:moe_router``).
_OBSERVED_GRID_ROUTER_ALGORITHMS = frozenset({
    "cublaslt_observed_grid:moe_router"})
# Unobserved split count: the external partial traffic is priced at the
# smallest split a split-K launch can have.
_UNOBSERVED_SPLIT_K_LOWER_BOUND = 2
_OBSERVED_GRID_LIMITATIONS = (
    "launch_tile_unobserved_operand_replay_at_compulsory_lower_bound",
    "split_k_count_unobserved_priced_at_lower_bound",
    "gemm_bf16_mma_occupancy_tile_unobserved",
    "launch_observed_on_cuda_12_9_not_serving_cuda_13_0",
)


def _is_observed_grid_router(kernel: KernelSpec) -> bool:
    """Return whether a router GEMM is a tile-less observed cuBLAS launch."""

    if (kernel.variant != "moe_router"
            or kernel.launch.algorithm not in _OBSERVED_GRID_ROUTER_ALGORITHMS):
        return False
    if (kernel.launch.tiling is not None
            or kernel.launch.parameters["split_k_count_observed"] != 0.0
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] <= 0):
        raise ValueError(
            "observed-grid router launch must be untiled with an unobserved "
            "split count and external partials")
    return True


def _is_simt_router(kernel: KernelSpec) -> bool:
    """Return whether a router GEMM runs as a SIMT per-expert reduction."""

    return (kernel.variant == "moe_router"
            and kernel.launch.algorithm in _SIMT_ROUTER_ALGORITHMS
            and kernel.launch.tiling is None)


def _uses_exact_gemv_launch(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether a versioned GEMV launch exposes its physical workers."""

    block_x, block_y, block_z = kernel.launch.block
    b, m, n = (
        int(kernel.shape[name]) for name in ("b", "m", "n"))
    legacy_one_dimensional = (
        block_y == 1 and block_x % hardware.parallel_width == 0)
    output_packed = (
        kernel.launch.grid_size * block_y == b * m * n
        and block_x == hardware.parallel_width)
    return (
        kernel.variant == "gemv"
        and kernel.launch.algorithm in {
            "nvidia_cublas_gemv", "metax_mcblas_gemv",
            "ascend_cann_gemv", "gemm:gemv"}
        and kernel.launch.grid == (kernel.launch.grid_size, 1, 1)
        and block_z == 1
        and (legacy_one_dimensional or output_packed)
    )


def _uniform_split_k_padding(kernel: KernelSpec) -> tuple[int, int]:
    """Logical and physically issued K tiles for an explicit uniform split.

    Other programs retain their established K-work approximation. This is a
    discrete source policy, never inferred from a duration or a split count.
    """
    if "uniform_split_k_tiles" not in kernel.launch.parameters:
        return (0, 0)
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k is None:
        raise ValueError("uniform split-K work requires tiling and split count")
    tile_k = tiling.tile_shape["k"]
    k_tiles = kernel.shape["k"] // tile_k
    return k_tiles, tiling.split_k * ((k_tiles + tiling.split_k - 1) // tiling.split_k)


def _logical_execution_geometry(
        kernel: KernelSpec, hardware) -> Tuple[int, float, int]:
    """Return semantic GEMM work plus source-explicit subtile geometry.

    GEMM workgroups normally follow directly from semantic shape, retained
    tile, and split-K.  A non-split mcBLAS grid that is an exact integer
    multiple of those tiles is a discrete decomposition fact rather than
    opaque padding, so each physical subtile executes the same tile pipeline.
    Other physical grid encodings remain outside the performance contract.
    """

    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    if _is_simt_router(kernel) or _is_observed_grid_router(kernel):
        # The source launch is one reduction CTA per expert column; each CTA
        # covers every token of the batch.  An observed-grid launch states
        # its CTAs directly.
        task_count = kernel.launch.grid_size
    elif kernel.variant == "gemv":
        if _uses_exact_gemv_launch(kernel, hardware):
            # These versioned source families expose one physical reduction
            # worker per grid entry, either directly or with output lanes in
            # block Y.  Other GEMV launches retain semantic output geometry.
            task_count = kernel.launch.grid_size
        else:
            # A generic SIMT GEMV assigns a reduction to every semantic output
            # element.  Treat those reductions as logical scheduling jobs when
            # the physical worker packing is not an exact retained fact.
            task_count = b * m * n
    elif kernel.variant in ("gemm", "batched_gemm", "moe_router"):
        tile_m, tile_n, _, _ = _tile(kernel)
        tiling = kernel.launch.tiling
        if tiling is None or tiling.split_k is None:
            raise ValueError("tensor GEMM requires split_k in retained tiling")
        task_count = _kernel_logical_workgroups(
            kernel, tile_m, tile_n, tiling.split_k)
        task_count *= _mcblas_exact_subtile_factor(kernel, task_count)
    else:
        raise ValueError(
            f"unsupported GEMM execution variant: {kernel.variant!r}")
    active_units = min(task_count, hardware.num_compute_units)
    if active_units <= 0:
        raise ValueError("GEMM logical execution has no active units")
    waves = task_count / active_units
    wave_efficiency = waves / math.ceil(waves)
    return (
        task_count,
        wave_efficiency,
        active_units,
    )


@runtime_lru_cache(maxsize=256)
def _cached_raster_tile_program_traffic(
        raster: GpuCtaRasterProgram, rows: int, n: int, k: int,
        tile_m: int, tile_n: int, tile_k: int,
        a_element_bytes: int, b_element_bytes: int, c_element_bytes: int,
        split_k: int, accumulate_in_place: bool, units: int,
        capacity_bytes: int, residency_profile: tuple,
        cluster_m: int, cluster_n: int):
    """Replay one immutable raster program through the shared miss core."""

    # A source-bound cluster job owns one rectangle of output CTA tiles.
    # Its invariant A/B panels reach L2 once per cluster, so replay that
    # rectangle as one cache object group, with complete valid global tails.
    # Tensor work and per-CTA local SMEM remain in their existing owners.
    tile_m *= cluster_m
    tile_n *= cluster_n
    m_tiles = math.ceil(rows / tile_m)
    n_tiles = math.ceil(n / tile_n)
    schedule = raster.schedule(m_tiles, n_tiles)
    resident_workers = units // (cluster_m * cluster_n)
    if resident_workers <= 0:
        raise ValueError("raster cluster cannot reside on the declared workers")
    return simulate_gemm_tile_program(
        schedule, m_extent=rows, n_extent=n, k_extent=k,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        a_element_bytes=a_element_bytes, b_element_bytes=b_element_bytes,
        c_element_bytes=c_element_bytes, split_k=split_k,
        accumulate_in_place=accumulate_in_place,
        wave_size=min(len(schedule), resident_workers),
        capacity_bytes=capacity_bytes,
        residency_profile=residency_profile)


def _gpu_raster_tensor_traffic(
        kernel: KernelSpec, hardware, logical_geometry=None, *, split_source=None) -> Tuple[float, float] | None:
    """Price a GPU GEMM whose launch retains its vendor CTA raster map.

    The raster program states the CTA-to-tile order the library actually
    dispatches, so the tile stream it implies can go straight through the
    shared L2-to-DRAM miss core instead of a schedule-free envelope.  A launch
    that retains no raster map, or an untiled one, stays outside this domain.
    """

    raster = kernel.launch.source_program
    if not isinstance(raster, GpuCtaRasterProgram):
        return None
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k is None:
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if logical_geometry is None:raise ValueError('Raster replay requires unified operand geometry')
    # The unified source owns extents/widths; retained raster owns replay.
    b, m, n, k, a_element_bytes, b_element_bytes, c_element_bytes = logical_geometry
    if kernel.variant == "batched_gemm":
        rows = m
        batches = b
    elif kernel.variant in ("gemm", "moe_router"):
        rows = b * m
        batches = 1
    else:
        raise AssertionError(
            f"unhandled tensor GEMM variant: {kernel.variant!r}")
    partial_element_bytes = int(
        kernel.launch.parameters["split_k_partial_bytes_per_element"])
    if split_source is not None:
        partial_element_bytes = split_source['partial_element_bytes']
    # A K split either accumulates into the output tile in place or gives
    # every slice its own partial buffer of the retained element width.
    accumulate_in_place = raster.k_slices > 1 and partial_element_bytes == 0
    output_element_bytes = (
        partial_element_bytes if partial_element_bytes > 0
        else c_element_bytes)
    if raster.is_cluster_grouped:
        cluster_m = int(kernel.launch.parameters["cluster_output_m_tiles"])
        cluster_n = int(kernel.launch.parameters["cluster_output_n_tiles"])
        units = min(hardware.num_compute_units, kernel.launch.grid_size)
    else:
        cluster_m = cluster_n = 1
        units = hardware.num_compute_units
    traffic = _cached_raster_tile_program_traffic(
        raster, rows, n, k, tile_m, tile_n, tile_k,
        a_element_bytes, b_element_bytes, output_element_bytes,
        raster.k_slices, accumulate_in_place,
        int(units),
        int(hardware.l2_replacement_capacity_bytes),
        hardware.l2_replacement_residency_profile, cluster_m, cluster_n)
    return (
        float(traffic.l2_bytes) * batches,
        float(traffic.dram_bytes) * batches,
    )


def _gemm_operand_bytes(program, role, operation):
    """Resolve source operand roles without imposing IDs on actual callers."""
    from src.kernel.memory_access import Operation
    from src.kernel.memory_path_solver import logical_view_unique_bytes
    from src.kernel.memory_access_builder import MemoryAccessWork
    accesses = (program.accesses if isinstance(program, MemoryAccessWork) or program.accesses
                else program.tile_programs[0].accesses)
    if (len(accesses) != 3 or tuple(access.operation for access in accesses)
            != (Operation.READ, Operation.READ, Operation.WRITE)):
        raise ValueError("GEMM operand projection requires two reads and one output")
    index = {"A": 0, "B": 1, "C": 2}[role]
    return logical_view_unique_bytes(program, accesses[index].view_id, operation)


def _cluster_operand_read_bytes(kernel: KernelSpec, geometry):
    """Unshared and cluster-invariant operand bytes of an aggregate GEMM.

    A is invariant across the cluster's output-N axis; B across output-M.
    Keep partial clusters and independent batch problems separate. This is
    an explicit reuse abstraction, not a native descriptor/raster claim.
    Local shared-memory delivery still occurs once per consuming CTA.
    """
    if "cluster_output_m_tiles" not in kernel.launch.parameters:
        return None
    b, m, n, k, a_width, b_width, _ = geometry
    tile_m, tile_n, _, _ = _tile(kernel)
    cluster_m = int(kernel.launch.parameters["cluster_output_m_tiles"])
    cluster_n = int(kernel.launch.parameters["cluster_output_n_tiles"])
    batches = b if kernel.variant == "batched_gemm" else 1
    rows = m if kernel.variant == "batched_gemm" else b * m
    m_tiles, n_tiles = math.ceil(rows / tile_m), math.ceil(n / tile_n)
    # Keep the original local tile service, including edge padding, separate
    # from global ingress. TMA zero fill supplies out-of-bounds cells without
    # reading those cells from global memory. Summing valid row/column extents
    # across tiles equals the complete logical operand extent.
    a_panel = min(rows, tile_m) * k * a_width
    b_panel = tile_n * k * b_width
    unshared = batches * m_tiles * n_tiles * (a_panel + b_panel)
    shared = batches * (
        math.ceil(n_tiles / cluster_n) * rows * k * a_width
        + math.ceil(m_tiles / cluster_m) * n * k * b_width)
    return float(unshared), float(shared)


def _cluster_output_padding_bytes(kernel, geometry):
    """Preserve the existing local output service while cropping global C."""
    if kernel.launch.tiling.split_k != 1:
        return 0.0
    b, m, n, _, _, _, c_width = geometry
    tile_m, tile_n, _, _ = _tile(kernel)
    batches, rows = (b, m) if kernel.variant == "batched_gemm" else (1, b * m)
    padded = math.ceil(rows / tile_m) * math.ceil(n / tile_n) * min(rows, tile_m) * tile_n
    return float(batches * (padded - rows * n) * c_width)


@runtime_lru_cache(maxsize=256)
def _aggregate_split_k_operand_lru(
        rows, n, k, tile_m, tile_n, tile_k, split_k, cluster_m, cluster_n,
        a_width, b_width, partial_width, units, capacity, residency,
        uniform_split_k_tiles):
    """Replay an aggregate split program through the shared miss core.

    Keep the established N-major coarse traversal, interleaving resident
    cluster jobs by K step. This does not claim an observed native raster.
    Partial objects participate in replacement; their declared DRAM write
    policy remains separately owned by the external-reduction contract.
    """
    macro_m, macro_n = tile_m * cluster_m, tile_n * cluster_n
    mt, nt = math.ceil(rows / macro_m), math.ceil(n / macro_n)
    schedule = tuple((mi, ni, zi) for zi in range(split_k)
        for ni in range(nt) for mi in range(mt))
    workers = min(len(schedule), units // (cluster_m * cluster_n))
    if workers <= 0:
        raise ValueError("split-K cluster cannot reside on the hardware")
    traffic = simulate_gemm_tile_program(schedule, m_extent=rows, n_extent=n,
        k_extent=k, tile_m=macro_m, tile_n=macro_n, tile_k=tile_k,
        a_element_bytes=a_width, b_element_bytes=b_width,
        c_element_bytes=partial_width, split_k=split_k,
        accumulate_in_place=False, wave_size=workers, capacity_bytes=capacity,
        residency_profile=residency, uniform_split_k_tiles=uniform_split_k_tiles)
    return traffic.a_l2_bytes + traffic.b_l2_bytes, traffic.a_dram_bytes + traffic.b_dram_bytes


def _tensor_traffic(kernel: KernelSpec, hardware, logical_program=None, *, split_source=None) -> Tuple[float, float]:
    """Tile-schedule traffic independent of physical launch metadata."""

    from src.kernel.memory_access_builder import has_gpu_nonsplit_gemm_source, gemm_prediction_source as gemm_memory_access_program,has_public_tensor_gemm_source
    from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_source,gpu_split_gemm_source_demands
    if split_source is None and has_gpu_split_gemm_source(kernel,hardware):
        split_source = gpu_split_gemm_source_demands(gemm_memory_access_program(kernel,hardware),expected_kernel=kernel)
    if logical_program is None and has_gpu_nonsplit_gemm_source(kernel, hardware):
        logical_program = gemm_memory_access_program(kernel, hardware)
    from src.kernel.memory_router import has_public_router_abi,public_router_source_work
    if has_public_router_abi(kernel):
        if logical_program is None:logical_program=gemm_memory_access_program(kernel,hardware)
        source=public_router_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
        geometry=(kernel.shape['b'],kernel.shape['m'],source['experts'],source['hidden'],*source['operand_widths'])
    elif has_public_tensor_gemm_source(kernel,hardware):
        from src.kernel.memory_path_solver import public_gemm_source_work
        if logical_program is None:logical_program=gemm_memory_access_program(kernel,hardware)
        source=public_gemm_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
        geometry=source['geometry']
    elif split_source is not None:
        geometry=split_source['operand_geometry']
    else:
        from src.kernel.memory_path_solver import logical_gemm_operand_geometry
        if kernel.variant == "batched_gemm":
            geometry = logical_gemm_operand_geometry(logical_program, batched=True)
        else:
            geometry = (1, *logical_gemm_operand_geometry(logical_program))
    if split_source is not None:
        geometry = split_source['operand_geometry']
        input_bytes=float(split_source['operand_read_bytes'])
        output_bytes=float(split_source['output_bytes'])
    elif has_public_tensor_gemm_source(kernel,hardware) or has_public_router_abi(kernel):
        activation,weights,output_bytes=map(float,source['nominal_operand_bytes'])
        input_bytes=activation+weights
    else:
        from src.kernel.memory_access import Operation
        from src.kernel.memory_path_solver import logical_view_unique_bytes
        input_bytes=float(sum(_gemm_operand_bytes(logical_program,role,Operation.READ) for role in ('A','B')))
        output_bytes=float(_gemm_operand_bytes(logical_program,'C',Operation.WRITE))
    raster_traffic = _gpu_raster_tensor_traffic(kernel, hardware, geometry,split_source=split_source)
    if raster_traffic is not None:
        if ("cluster_output_m_tiles" in kernel.launch.parameters
                and not kernel.launch.source_program.is_cluster_grouped):
            raise NotImplementedError("cluster operand reuse with an explicit raster is unmodeled")
        return raster_traffic

    b, m, n, k, a_element_bytes, b_element_bytes, c_element_bytes = geometry
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k is None:
        raise ValueError("tensor GEMM requires split_k in retained tiling")
    split_k = tiling.split_k
    if split_source is not None:
        split_k = split_source['split_k']
    grid_z = split_k
    m_total = b * m
    if kernel.variant == "batched_gemm":
        weight_batches = b
    elif kernel.variant in ("gemm", "moe_router"):
        weight_batches = 1
    else:
        raise AssertionError(
            f"unhandled tensor GEMM variant: {kernel.variant!r}")
    partial_element_bytes = float(
        kernel.launch.parameters["split_k_partial_bytes_per_element"])
    if split_source is not None:
        partial_element_bytes = float(split_source['partial_element_bytes'])
    partial_dram_element_bytes = float(
        kernel.launch.parameters[
            "split_k_partial_dram_bytes_per_element"])
    if partial_element_bytes != 0 or partial_dram_element_bytes != 0:
        if grid_z == 1:
            raise ValueError(
                "non-split GEMM cannot declare a partial element width")
    if tile_k <= 0:
        raise ValueError(f"GEMM tile_k must be positive, got {tile_k}")
    if k % tile_k != 0:
        raise ValueError(
            "GEMM K dimension is not divisible by retained tile_k and the "
            "performance plan provides no source-backed tail-tile semantics: "
            f"K={k}, tile_k={tile_k}")
    if kernel.variant == "batched_gemm":
        per_batch_m = m
    elif kernel.variant in ("gemm", "moe_router"):
        per_batch_m = m_total
    else:
        raise AssertionError(
            f"unhandled tensor GEMM variant: {kernel.variant!r}")
    effective_m = min(tile_m, per_batch_m)
    m_tiles = math.ceil(per_batch_m / tile_m)
    n_tiles = math.ceil(n / tile_n)
    k_steps = k // tile_k
    a_tile = effective_m * tile_k * a_element_bytes
    b_tile = tile_k * tile_n * b_element_bytes
    c_tile = effective_m * tile_n * c_element_bytes
    cluster_reads = _cluster_operand_read_bytes(kernel, geometry)
    if grid_z > 1:
        # A CUDA grid-z split partitions K, but every logical (M, N) output
        # tile still executes all of its K-tile loads across the slices.  A
        # source-visible external reducer materializes one partial matrix per
        # slice.  Without that reducer, the retained single-kernel launch
        # accumulates in the destination: the first slice initializes C and
        # each remaining slice reads it before every slice publishes C.  This
        # request accounting is fixed by GEMM semantics plus grid_z and is
        # independently validated by A800 L2 read/write counters.  A retained
        # mcBLAS split-K launch whose CTA raster does not decode (grid.x/y do
        # not state a power-of-two swizzle group) shares this same accounting
        # rather than the raster-schedule replay: mcBLAS's opaque mainloop
        # still executes a full a_tile/b_tile pair per (M tile, N tile,
        # K step) regardless of which axis the physical grid enumerates first,
        # so the logical-tile multiplicity is unaffected by that axis choice.
        # Validated against C500 GEMM_B512_M1_K5120_N1536 (grid (12,4,2),
        # tile 128x128x128, split_k=2): this formula gives 125,829,120 L2
        # read bytes against a measured 125,878,272 (99.96%), where the prior
        # compulsory-only shortcut gave 20,971,520 (0.206x, the launch-family
        # non-monotonicity this fix corrects).
        logical_tiles = weight_batches * m_tiles * n_tiles
        operand_l2_bytes = float(
            logical_tiles * k_steps * (a_tile + b_tile))
        if cluster_reads is not None:
            _, operand_l2_bytes = cluster_reads
        if partial_element_bytes > 0:
            if split_source is not None:
                partial_l2_bytes=float(split_source['write_bytes'])
            else:
                partial_l2_bytes=float(source['declared_partial_bytes'])
            partial_dram_bytes=partial_l2_bytes*partial_dram_element_bytes/partial_element_bytes
            if (cluster_reads is not None
                    and "operand_transport_tma" in kernel.launch.parameters):
                lru_read_l2, lru_read_dram = _aggregate_split_k_operand_lru(
                    per_batch_m, n, k, tile_m, tile_n, tile_k, split_k,
                    int(kernel.launch.parameters["cluster_output_m_tiles"]),
                    int(kernel.launch.parameters["cluster_output_n_tiles"]),
                    a_element_bytes, b_element_bytes, int(partial_element_bytes),
                    hardware.num_compute_units, int(hardware.l2_replacement_capacity_bytes),
                    hardware.l2_replacement_residency_profile,
                    "uniform_split_k_tiles" in kernel.launch.parameters)
                if weight_batches * lru_read_l2 != operand_l2_bytes:
                    raise ValueError("split-K LRU does not conserve operand ingress")
                return (operand_l2_bytes + partial_l2_bytes,
                    weight_batches * lru_read_dram + partial_dram_bytes)
            return (
                operand_l2_bytes + partial_l2_bytes,
                input_bytes + partial_dram_bytes,
            )
        accumulator_l2_bytes = float(
            (grid_z - 1) * output_bytes + grid_z * output_bytes)
        return (
            operand_l2_bytes + accumulator_l2_bytes,
            input_bytes + output_bytes,
        )
    if kernel.launch.algorithm in _MCBLAS_GEMM_ALGORITHMS:
        # Tile semantics and independent non-duration hierarchy counters close
        # the logical L2-ingress request stream.  The opaque layout/pipeline
        # identities still do not specify physical CTA traversal, residency,
        # or eviction, so they cannot drive an L2-to-DRAM cache simulation.
        logical_tiles = weight_batches * m_tiles * n_tiles
        l2_total = logical_tiles * (
            k_steps * (a_tile + b_tile) + c_tile)
        if cluster_reads is not None:
            unshared, shared = cluster_reads
            l2_total -= unshared - shared
        compulsory = input_bytes+output_bytes
        validated_l2_miss_rate(compulsory, float(l2_total))
        return float(l2_total), compulsory
    wave_size = min(m_tiles * n_tiles, hardware.num_compute_units)
    l2_total, dram_bytes = simulate_gemm_n_major(
        m_tiles, n_tiles, k_steps, a_tile, b_tile, c_tile,
        wave_size, hardware.l2_replacement_capacity_bytes)
    if kernel.variant == "batched_gemm":
        l2_total *= b
        dram_bytes *= b
    if cluster_reads is not None:
        _, shared = cluster_reads
        # Only valid cells request global bytes. Keep the established padded
        # cache-object replay as an explicit conservative miss approximation;
        # remove its padded output before capping read misses by valid ingress.
        padded_output = weight_batches * m_tiles * n_tiles * c_tile
        l2_total = shared + output_bytes
        dram_bytes = min(dram_bytes - padded_output, shared) + output_bytes
    return float(l2_total), float(dram_bytes)


def _cann_gemm_tile_pipeline_program(
        kernel: KernelSpec) -> CannGemmTilePipelineProgram | None:
    program = kernel.launch.source_program
    if program is None:
        return None
    if isinstance(program, (CannGemmLaunchProgram, GpuCtaRasterProgram)):
        return None
    if not isinstance(program, CannGemmTilePipelineProgram):
        raise TypeError("CANN GEMM source program must be typed")
    return program


def _cann_gemm_launch_program(
        kernel: KernelSpec) -> CannGemmLaunchProgram | None:
    program = kernel.launch.source_program
    if program is None or isinstance(
            program, (CannGemmTilePipelineProgram, GpuCtaRasterProgram)):
        return None
    if not isinstance(program, CannGemmLaunchProgram):
        raise TypeError("CANN GEMM launch program must be typed")
    return program


def _uses_cann_region_program(kernel: KernelSpec) -> bool:
    program = _cann_gemm_launch_program(kernel)
    return (
        program is not None
        and (program.base_m, program.base_n, program.base_k) in {
            (128, 256, 64), (256, 128, 64)}
        and program.traversal_order in {
            "m_then_n_snake", "n_then_m_snake"}
        and program.worker_grid_k == 1
        and program.split_k == 1)


def _uses_cann_persistent_program(kernel: KernelSpec) -> bool:
    program = _cann_gemm_launch_program(kernel)
    return (
        program is not None
        and program.traversal_order == "persistent_n_then_m"
        and program.worker_grid_k == 1
        and program.split_k == 1)


def _uses_cann_persistent_execution(kernel: KernelSpec) -> bool:
    """Recognize typed persistent programs and the legacy untyped path."""

    program = _cann_gemm_launch_program(kernel)
    return (
        _uses_cann_persistent_program(kernel)
        or (program is None and kernel.launch.algorithm == "cann_matmul_v3"))


def _cann_gemm_warm_operand_context_covers_trace(
        kernel: KernelSpec, trace, replacement_capacity: int,
        predecessor_dirty_bytes: int) -> bool:
    """Return whether explicit predecessor replays retain every operand.

    A replay context proves that immutable A/B allocations have been accessed
    by prior identical source programs.  Result allocations are deliberately
    treated as distinct on every predecessor invocation: this is the safe
    upper bound for a caller that does not serialize allocator addresses.  If
    that complete upper bound fits L2, ordinary LRU has no legal operand miss
    on the current invocation.  Larger traces remain on the analytical cold
    path until their full predecessor descriptor state is supplied.
    """

    replays = kernel.l2_execution_context.same_program_operand_replays
    if replays == 0 or predecessor_dirty_bytes != 0:
        return False
    object_sizes = {}
    for key, size in zip(trace.access_keys, trace.access_sizes, strict=True):
        if key in object_sizes:
            prior_size = object_sizes[key]
        else:
            object_sizes[key] = size
            prior_size = size
        if prior_size != size:
            raise ValueError(
                "CANN GEMM LRU object changed byte size within one source "
                "program")
    operand_bytes = sum(
        size for key, size in object_sizes.items()
        if key >> 61 in (0, 1))
    output_bytes = sum(
        size for key, size in object_sizes.items() if key >> 61 == 2)
    if operand_bytes + output_bytes <= 0:
        raise ValueError("CANN GEMM source trace has no cache objects")
    return operand_bytes + (replays + 1) * output_bytes <= replacement_capacity


@runtime_lru_cache(maxsize=128)
def _cached_cann_gemm_region_program_traffic(
        rows: int, n: int, k: int, element_bytes: int,
        l2_replacement_capacity_bytes: int,
        program: CannGemmLaunchProgram) -> GemmSourceOrderTraffic:
    """Reuse one source-order replay for an identical immutable program."""

    m_tiles = math.ceil(rows / program.base_m)
    n_tiles = math.ceil(n / program.base_n)
    schedule = cann_gemm_source_tile_order(
        m_tiles, n_tiles,
        program.m_l2_regions, program.n_l2_regions,
        program.m_l2_block, program.n_l2_block,
        program.traversal_order, program.aic_workers)
    return simulate_gemm_source_order_traffic(
        schedule, rows, n, k,
        program.base_m, program.base_n, program.base_k,
        element_bytes, min(program.aic_workers, len(schedule)),
        l2_replacement_capacity_bytes)


def _cann_gemm_region_program_traffic(
        kernel: KernelSpec, hardware) -> GemmSourceOrderTraffic | None:
    """Replay a normalized CANN region program through the shared LRU."""

    program = _cann_gemm_launch_program(kernel)
    if program is None or not _uses_cann_region_program(kernel):
        return None
    if (not program.operand_uses_l2("a")
            or not program.operand_uses_l2("b")):
        raise NotImplementedError(
            "CANN region-program L2 bypass traffic is unsupported")
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    return _cached_cann_gemm_region_program_traffic(
        rows, n, k, int(kernel.shape["dtype_bytes"]),
        int(hardware.l2_replacement_capacity_bytes), program)


def _cann_gemm_persistent_operand_traffic(
        kernel: KernelSpec, hardware, *,
        predecessor_dirty_bytes: int | None = None,
        ) -> GemmSourceOrderTraffic | None:
    """Replay one normalized CANN operand stream through the shared LRU.

    ``predecessor_dirty_bytes`` is execution-context state, not a property of
    the GEMM launch.  A standalone KernelSpec therefore owns no anonymous
    predecessor lines.  A caller with an explicit inter-op cache contract can
    supply the dirty occupancy without changing the launch program or the LRU
    formula.
    """

    program = _cann_gemm_launch_program(kernel)
    if program is None or not _uses_cann_persistent_program(kernel):
        return None
    if (not program.operand_uses_l2("a")
            or not program.operand_uses_l2("b")):
        # The current aggregate NPU memory path requires every DRAM byte to
        # traverse L2.  Keep explicit bypass programs on the existing path
        # until that contract represents bypass traffic separately.
        return None
    if predecessor_dirty_bytes is None:
        predecessor_dirty_bytes = kernel.anonymous_l2_dirty_bytes
    if (isinstance(predecessor_dirty_bytes, bool)
            or not isinstance(predecessor_dirty_bytes, int)):
        raise TypeError("CANN GEMM predecessor dirty bytes must be an integer")
    replacement_capacity = int(hardware.l2_replacement_capacity_bytes)
    if not 0 <= predecessor_dirty_bytes <= replacement_capacity:
        raise ValueError(
            "CANN GEMM predecessor dirty bytes must fit replacement capacity")
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    output_bytes = b * m * n * int(kernel.shape["dtype_bytes"])
    (a_requested_bytes, a_miss_bytes,
     b_requested_bytes, b_miss_bytes,
     dirty_eviction_bytes, dirty_end,
     anonymous_dirty_end) = simulate_cann_gemm_persistent_stateful_dual_accounted(
        aic_workers=program.aic_workers,
        worker_grid_m=program.worker_grid_m,
        worker_grid_n=program.worker_grid_n,
        worker_grid_k=program.worker_grid_k,
        single_core_m=program.single_core_m,
        single_core_n=program.single_core_n,
        single_core_k=program.single_core_k,
        base_m=program.base_m,
        base_n=program.base_n,
        base_k=program.base_k,
        a_l1_k_tiles=program.a_l1_k_tiles,
        b_l1_k_tiles=program.b_l1_k_tiles,
        step_m=program.step_m,
        step_n=program.step_n,
        rows=b * m,
        n=n,
        k=k,
        element_bytes=int(kernel.shape["dtype_bytes"]),
        capacity_bytes=replacement_capacity,
        initial_dirty_bytes=predecessor_dirty_bytes)
    if kernel.l2_execution_context.same_program_operand_replays:
        trace = cann_gemm_persistent_descriptor_trace(
            program, b * m, n, k,
            element_bytes=int(kernel.shape["dtype_bytes"]))
        if _cann_gemm_warm_operand_context_covers_trace(
                kernel, trace, replacement_capacity,
                predecessor_dirty_bytes):
            # The canonical Timeline runner binds five preceding identical
            # x/w launches to this context. Its validated L2 contract keeps
            # immutable A/B operands resident when the complete predecessor
            # output footprint also fits replacement capacity. The current
            # output remains a normal write stream and is still charged below.
            a_miss_bytes = 0
            b_miss_bytes = 0
    # Only displaced predecessor bytes write back; untouched predecessor
    # lines remain resident. Trace-owned output lines reach DRAM either
    # through an in-program eviction or through the terminal output flush.
    trace_dirty_end = dirty_end - anonymous_dirty_end
    if trace_dirty_end < 0:
        raise AssertionError("CANN GEMM LRU lost anonymous dirty state")
    predecessor_dirty_writeback_bytes = (
        predecessor_dirty_bytes - anonymous_dirty_end)
    dirty_writeback_bytes = dirty_eviction_bytes + trace_dirty_end
    if predecessor_dirty_writeback_bytes > dirty_writeback_bytes:
        raise AssertionError(
            "CANN GEMM predecessor writeback exceeds total")
    return GemmSourceOrderTraffic(
        l2_bytes=(
            a_requested_bytes + b_requested_bytes + output_bytes),
        dram_bytes=(
            a_miss_bytes + b_miss_bytes
            + dirty_writeback_bytes),
        a_l2_bytes=a_requested_bytes,
        a_dram_bytes=a_miss_bytes,
        b_l2_bytes=b_requested_bytes,
        b_dram_bytes=b_miss_bytes,
        output_l2_bytes=output_bytes,
        dirty_writeback_bytes=dirty_writeback_bytes,
        predecessor_dirty_writeback_bytes=(
            predecessor_dirty_writeback_bytes),
    )


def _attach_cann_gemm_persistent_cache_paths(
        inp: PredictionInput,
        traffic: GemmSourceOrderTraffic) -> PredictionInput:
    """Expose dirty L2 victims as a directional DRAM write stream."""

    if inp.mem_model != "npu_tile":
        raise ValueError("CANN GEMM cache paths require npu_tile memory")
    paths = tuple(path for path in inp.memory_paths if path.name == "mte2_read")
    if len(paths) != 1 or len(inp.memory_paths) != 1:
        raise ValueError("CANN GEMM cache paths require one MTE2 read path")
    original = paths[0]
    if original.access_class_demands:
        raise ValueError("CANN GEMM cache paths require an untyped read path")

    read_dram_bytes = traffic.a_dram_bytes + traffic.b_dram_bytes
    writeback_bytes = traffic.dirty_writeback_bytes
    if not math.isclose(
            traffic.l2_bytes,
            inp.demand_per_resource["l2_cache"],
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("CANN GEMM cache paths do not conserve L2 traffic")
    if not math.isclose(
            read_dram_bytes + writeback_bytes,
            inp.demand_per_resource["dram"],
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("CANN GEMM cache paths do not conserve DRAM traffic")

    def node(resource_role: str, byte_count: int) -> MemoryNodeDemand:
        return MemoryNodeDemand(
            resource_role=resource_role,
            transactions=None,
            bytes=float(byte_count),
            transaction_bytes=None,
        )

    # The existing source-backed L2 override owns the aggregate A/B ingress
    # and FixPipe allocation stream. Dirty victims are an L2 egress generated
    # by that same stream; expose only their directional DRAM continuation so
    # the cache service is not charged twice.
    read_nodes = [node("l2_cache", traffic.l2_bytes)]
    if read_dram_bytes > 0:
        read_nodes.append(node("dram", read_dram_bytes))
    separated = [dataclasses.replace(original, nodes=tuple(read_nodes))]
    if writeback_bytes > 0:
        separated.append(MemoryPathDemand(
            name="l2_dirty_writeback",
            access_kind="write",
            initial_transactions=None,
            nodes=(node("dram", writeback_bytes),),
            source=original.source,
        ))
    return dataclasses.replace(inp, memory_paths=tuple(separated))


def _cann_deterministic_split_k_program(
        kernel: KernelSpec) -> CannGemmTilePipelineProgram | None:
    program = _cann_gemm_tile_pipeline_program(kernel)
    if (program is None or program.program_kind
            != "cann_gemm_deterministic_split_k_v1"):
        return None
    return program


def _cann_single_core_split_k_program(
        kernel: KernelSpec) -> CannGemmTilePipelineProgram | None:
    program = _cann_gemm_tile_pipeline_program(kernel)
    if (program is None or program.program_kind
            != "cann_gemm_single_core_split_k_v1"):
        return None
    return program


def _cann_deterministic_split_k_traffic(
        kernel: KernelSpec) -> _CannDeterministicSplitKTraffic | None:
    """Project source-visible AIC partials and AIV reduction traffic."""

    program = _cann_deterministic_split_k_program(kernel)
    if program is None:
        return None
    from src.kernel.memory_cann_gemm import (
        has_cann_deterministic_gemm_source,bound_cann_gemm_source,standalone_cann_deterministic_bindings,
        build_cann_deterministic_source,cann_deterministic_source_demands,cann_deterministic_traffic_projection,
    )
    if has_cann_deterministic_gemm_source(kernel):
        bound = bound_cann_gemm_source(kernel)
        if bound is None:
            from src.kernel.memory_access import Granularity
            bindings = standalone_cann_deterministic_bindings(kernel,call_id='gemm.operands',binding_id='gemm.operands')
            work = cann_deterministic_source_demands(build_cann_deterministic_source(kernel,bindings,Granularity.KERNEL,
                program_id='gemm.operands'))
        else:
            _,work = bound
        return _CannDeterministicSplitKTraffic(**cann_deterministic_traffic_projection(work))
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])
    m_groups = math.ceil(rows / program.single_core_m)
    n_groups = math.ceil(n / program.single_core_n)
    output_groups = m_groups * n_groups

    # MK order replays A once per N slab and B once per M slab.  The workspace
    # reserves padded ``singleSize`` strides, but FixPipe's SetSingleShape
    # transfers only the semantic M/N tail for each AIC partial.  The AIV
    # reduction reads those same elements and publishes the final BF16 C.
    operand_l2_read_bytes = float(
        n_groups * rows * k * element_bytes
        + m_groups * k * n * element_bytes)
    operand_dram_read_bytes = float(
        rows * k * element_bytes + k * n * element_bytes)
    k_groups = math.ceil(k / program.single_core_k)
    # The deterministic source invokes IterateAll once for every K chunk.
    # The first chunk initializes the output slab and every later chunk
    # atomically accumulates another complete partial into the same ping/pong
    # workspace.  Counting only one publication per AIC loses the additional
    # chunks owned by workers whose source loop has multiple iterations.
    workspace_l2_write_bytes = float(
        k_groups * rows * n * program.partial_element_bytes)
    workspace_l2_read_bytes = float(
        program.aic_workers * rows * n * program.partial_element_bytes)
    output_bytes = float(rows * n * element_bytes)
    return _CannDeterministicSplitKTraffic(
        operand_l2_read_bytes=operand_l2_read_bytes,
        operand_dram_read_bytes=operand_dram_read_bytes,
        workspace_l2_write_bytes=workspace_l2_write_bytes,
        workspace_dram_write_bytes=workspace_l2_write_bytes,
        workspace_l2_read_bytes=workspace_l2_read_bytes,
        output_l2_write_bytes=output_bytes,
        output_dram_write_bytes=output_bytes,
    )


def _cann_deterministic_split_k_mte1_operand_bytes(
        kernel: KernelSpec,
        program: CannGemmTilePipelineProgram) -> tuple[float, float] | None:
    """Return source-ordered deterministic Split-K A1/B1-to-L0 traffic.

    Both source-identified schedulers emit the same semantic A/B tiles from
    A1/B1 into L0 that MTE2 has already materialized.  The two stages have
    distinct engines and source lifetimes, so retain the same operand
    conservation while exposing separate L0A/L0B paths.  Programs whose
    scheduler ownership is not decoded remain unsupported here.
    """

    if program.scheduler_policy not in ("n_buffer_33", "l2_split_matmul"):
        return None
    from src.kernel.memory_cann_gemm import has_cann_deterministic_gemm_source,bound_cann_gemm_source
    if has_cann_deterministic_gemm_source(kernel):
        bound = bound_cann_gemm_source(kernel)
        if bound is None:
            # The public traffic entry compiles the same source outside a prediction context.
            # Separate A/B roles are recovered from the source, not split from aggregate bytes.
            from src.kernel.memory_cann_gemm import standalone_cann_deterministic_bindings,build_cann_deterministic_source,cann_deterministic_source_demands
            from src.kernel.memory_access import Granularity
            binding = standalone_cann_deterministic_bindings(kernel,call_id='gemm.mte1',binding_id='gemm.mte1')
            work = cann_deterministic_source_demands(build_cann_deterministic_source(kernel,binding,Granularity.KERNEL,program_id='gemm.mte1'))
        else:
            _,work = bound
        return float(work['totals']['input_a']),float(work['totals']['input_b'])
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])
    m_groups = math.ceil(rows / program.single_core_m)
    n_groups = math.ceil(n / program.single_core_n)
    a_bytes = float(n_groups * rows * k * element_bytes)
    b_bytes = float(m_groups * k * n * element_bytes)
    if a_bytes <= 0.0 or b_bytes <= 0.0:
        raise ValueError("NBuffer33 MTE1 path requires positive A/B traffic")
    return a_bytes, b_bytes


def _cann_nbuffer33_mte1_operand_bytes(
        kernel: KernelSpec,
        program: CannGemmTilePipelineProgram) -> tuple[float, float] | None:
    """Return the NBuffer33 subset that has an exact pair service."""

    if program.scheduler_policy != "n_buffer_33":
        return None
    return _cann_deterministic_split_k_mte1_operand_bytes(kernel, program)


def _cann_single_core_split_k_traffic(
        kernel: KernelSpec,
        *,
        l2_replacement_capacity_bytes: int | None,
        ) -> _CannDeterministicSplitKTraffic | None:
    """Project key-65632's per-K-chunk workspace lifecycle."""

    program = _cann_single_core_split_k_program(kernel)
    if program is None:
        return None
    from src.kernel.memory_cann_gemm import (
        has_cann_single_core_gemm_source,bound_cann_gemm_source,standalone_cann_single_core_bindings,
        build_cann_single_core_source,cann_single_core_source_demands,cann_single_core_traffic_projection,
    )
    if has_cann_single_core_gemm_source(kernel):
        bound = bound_cann_gemm_source(kernel)
        if bound is None:
            from src.kernel.memory_access import Granularity
            binding = standalone_cann_single_core_bindings(kernel,call_id='gemm.operands',binding_id='gemm.operands')
            work = cann_single_core_source_demands(build_cann_single_core_source(kernel,binding,Granularity.KERNEL,
                program_id='gemm.operands'))
        else:
            _,work = bound
        return _CannDeterministicSplitKTraffic(**cann_single_core_traffic_projection(work,l2_replacement_capacity_bytes))
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])
    m_groups = math.ceil(rows / program.single_core_m)
    n_groups = math.ceil(n / program.single_core_n)
    if m_groups * n_groups != program.aic_workers:
        raise ValueError(
            "CANN single-core split-K requires one output block per AIC")
    k_groups = math.ceil(k / program.single_core_k)
    inner_block_m = program.step_m * program.base_m
    m_stripes = 0
    for m_start in range(0, rows, program.single_core_m):
        m_extent = min(program.single_core_m, rows - m_start)
        m_stripes += math.ceil(m_extent / inner_block_m)

    # Every AIC owns one complete output block.  A is replayed across N
    # blocks and B across M blocks.  Each serial K chunk publishes the whole
    # FP32 partial block to GM (the first write is non-atomic and subsequent
    # writes are atomic); paired AIVs read the completed workspace once and
    # publish BF16 C.  This exactly mirrors the legacy source loop and its
    # full-workspace rule without expanding output tiles or K chunks.
    operand_l2_read_bytes = float(
        n_groups * rows * k * element_bytes
        + m_stripes * k * n * element_bytes)
    operand_dram_read_bytes = float(
        rows * k * element_bytes + k * n * element_bytes)
    output_elements = rows * n
    workspace_write_bytes = float(
        k_groups * output_elements * program.partial_element_bytes)
    workspace_read_bytes = float(
        output_elements * program.partial_element_bytes)
    output_bytes = float(output_elements * element_bytes)
    if (l2_replacement_capacity_bytes is not None
            and (isinstance(l2_replacement_capacity_bytes, bool)
                 or not isinstance(l2_replacement_capacity_bytes, int)
                 or l2_replacement_capacity_bytes <= 0)):
        raise ValueError(
            "CANN single-core split-K L2 capacity must be positive")
    workspace_dram_write_bytes = 0.0
    if (l2_replacement_capacity_bytes is not None
            and workspace_write_bytes > l2_replacement_capacity_bytes):
        # The source publishes one complete FP32 partial block for every
        # serial K chunk.  Once the aggregate workspace exceeds replacement
        # capacity, those publications cannot remain an L2-only stream; keep
        # the source write volume on the external path instead of claiming
        # residency from the one-buffer marker.
        workspace_dram_write_bytes = workspace_write_bytes
    return _CannDeterministicSplitKTraffic(
        operand_l2_read_bytes=operand_l2_read_bytes,
        operand_dram_read_bytes=operand_dram_read_bytes,
        workspace_l2_write_bytes=workspace_write_bytes,
        # The source consumes one inner-M stripe before advancing.  For the
        # capacity-qualified small workspace domain, intermediate atomic
        # updates remain L2-resident; the large-workspace domain above keeps
        # the complete publication stream on DRAM.
        workspace_dram_write_bytes=workspace_dram_write_bytes,
        workspace_l2_read_bytes=workspace_read_bytes,
        output_l2_write_bytes=output_bytes,
        output_dram_write_bytes=output_bytes,
    )


def _cann_gemm_tile_pipeline_traffic(
        kernel: KernelSpec,
        *,
        l2_replacement_capacity_bytes: int | None,
        ) -> _CannDeterministicSplitKTraffic | None:
    deterministic = _cann_deterministic_split_k_traffic(kernel)
    if deterministic is not None:
        return deterministic
    return _cann_single_core_split_k_traffic(
        kernel,
        l2_replacement_capacity_bytes=l2_replacement_capacity_bytes)


def _cann_deterministic_split_k_aiv_data_workers(
        kernel: KernelSpec) -> int | None:
    """Return AIVs that execute the split-K data body, not only SyncAll.

    Every launched AIV participates in the source barriers.  The reduction
    body has a separate admission predicate: ``vIndex * coreSize <
    coreOutSize``.  On the L2-split path the source recomputes ``coreSize``
    from the *current* M tail before that predicate, rather than from the
    padded ``singleCoreM`` allocation.  N cancels from the inequality, so
    the active data-worker count follows directly from the semantic rows in
    the current M slab.  The analytical selector currently admits exactly
    one M slab; fail loudly if that contract changes.
    """

    program = _cann_gemm_tile_pipeline_program(kernel)
    if program is None:
        return None
    b, m = (int(kernel.shape[name]) for name in ("b", "m"))
    rows = b * m
    if program.program_kind == "cann_gemm_single_core_split_k_v1":
        return program.aiv_workers
    if math.ceil(rows / program.single_core_m) != 1:
        raise ValueError(
            "CANN deterministic split-K AIV admission requires one M slab")
    rows_per_aiv = math.ceil(rows / program.aiv_workers)
    active = min(program.aiv_workers, math.ceil(rows / rows_per_aiv))
    if active <= 0:
        raise ValueError(
            "CANN deterministic split-K requires an active AIV data worker")
    return active


def _non_gpu_tensor_traffic(kernel: KernelSpec, hardware, logical_program) -> Tuple[float, float]:
    b, m, n, k = (int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])
    from src.kernel.memory_cann_gemm import has_cann_gemm_source, cann_gemm_source_demands
    from src.kernel.memory_router import has_public_router_abi,public_router_source_work
    if has_public_router_abi(kernel):
        source=public_router_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
        rows,n,k=source['logical_rows'],source['experts'],source['hidden']
        # Preserve the original non-GPU numerical template's nominal-width
        # projection. Actual public A/B views remain BF16 independently.
        activation,weights,output=map(float,source['nominal_operand_bytes'])
    elif has_cann_gemm_source(kernel):
        work = cann_gemm_source_demands(logical_program)
        b,m,n,k = work['geometry']
        rows = b*m
        activation,weights,output = map(float,work['operand_bytes'])
    elif (hardware.kernel_memory_model in ("pim_bank", "systolic_scratchpad")
            and kernel.variant == "gemm"):
        from src.kernel.memory_access import Operation
        from src.kernel.memory_path_solver import logical_view_unique_bytes
        activation = float(_gemm_operand_bytes(logical_program, "A", Operation.READ))
        weights = float(_gemm_operand_bytes(logical_program, "B", Operation.READ))
        output = float(_gemm_operand_bytes(logical_program, "C", Operation.WRITE))
    else:
        from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program,has_public_tensor_gemm_source
        if logical_program is None:logical_program=gemm_memory_access_program(kernel,hardware)
        if has_public_tensor_gemm_source(kernel,hardware):
            from src.kernel.memory_path_solver import public_gemm_source_work
            source=public_gemm_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
            activation,weights,output=map(float,source['nominal_operand_bytes'])
        else:
            from src.kernel.memory_access import Operation
            from src.kernel.memory_path_solver import logical_view_unique_bytes
            activation,weights=(float(_gemm_operand_bytes(logical_program,role,Operation.READ)) for role in ('A','B'))
            output=float(_gemm_operand_bytes(logical_program,'C',Operation.WRITE))
    split_k_traffic = _cann_gemm_tile_pipeline_traffic(
        kernel, l2_replacement_capacity_bytes=None)
    if split_k_traffic is not None:
        return split_k_traffic.l2_bytes, split_k_traffic.dram_bytes
    region_traffic = _cann_gemm_region_program_traffic(kernel, hardware)
    if region_traffic is not None:
        return float(region_traffic.l2_bytes), float(region_traffic.dram_bytes)
    persistent_traffic = _cann_gemm_persistent_operand_traffic(
        kernel, hardware)
    if persistent_traffic is not None:
        return (
            float(persistent_traffic.l2_bytes),
            float(persistent_traffic.dram_bytes),
        )
    if kernel.launch.algorithm == _CANN_MATMUL_V3_ASW_65536_ALGORITHM:
        a_l2_bytes, b_l2_bytes, output_l2_bytes, _, _ = (
            _cann_matmul_v3_asw_65536_operand_traffic(kernel))
        return (
            a_l2_bytes + b_l2_bytes + output_l2_bytes,
            activation + weights + output,
        )
    gemm_pipeline_replay = _cann_npu_gemm_operand_replay_bytes(
        kernel, hardware)
    if gemm_pipeline_replay is not None:
        # MTE2 reads the padded A/B tile schedule.  The terminal AIC FixPipe
        # write has no source-defined tile lifetime yet, so leave its small
        # output contribution in the aggregate DRAM lower bound only.
        return gemm_pipeline_replay, activation + weights + output
    tile_rows = float(kernel.launch.parameters["weight_tile_rows"])
    mte2_tile = float(kernel.launch.parameters["mte2_tile_bytes"])
    if hardware.aic_aiv_separated and (
            tile_rows != 0.0 or mte2_tile != 0.0):
        raise ValueError(
            "NPU GEMM traffic-derived tile parameters are unsupported "
            "without source-backed descriptor semantics")
    if tile_rows > 0:
        if hardware.aic_aiv_separated:
            rereads = rows / tile_rows
        else:
            rereads = float(math.ceil(rows / tile_rows))
            refetch_rows = hardware.memory_subsystem["gemm_l1_refetch_tile_rows"]
            if refetch_rows is not None and weights < hardware.l2_size_bytes:
                rereads = max(rereads, 1.0 + rows / refetch_rows)
        l2_total = rereads * weights + activation + output
    else:
        l2_total = activation + weights + output

    if hardware.l2_size_bytes and tile_rows > 0:
        if mte2_tile > 0:
            effective_rows = min(tile_rows, rows)
            estimated_k = max(16.0, mte2_tile / (effective_rows * element_bytes))
            k_steps = math.ceil(k / estimated_k)
            weight_step = (k / k_steps) * n * element_bytes
        else:
            weight_step = weights
        dram_bytes = l2_total if weight_step > hardware.l2_size_bytes else weights + activation + output
    else:
        dram_bytes = weights + activation + output
    return float(l2_total), float(dram_bytes)


def _cann_npu_gemm_pipeline_geometry(
        kernel: KernelSpec,
        hardware) -> _CannNpuGemmPipelineGeometry | None:
    """Derive one compact AIC GEMM program from tile and buffer geometry.

    A worker owns a rectangular part of the output tile grid.  The smallest
    padded decomposition is selected first, then the physical block with the
    most balanced A and B matrix extents minimizes one-sided operand pressure.
    Each of the two retained operand slots divides L1 equally between A1 and
    B1; their K depths follow directly from L1 capacity and the base Cube tile.
    No kernel symbol, workload duration, or accuracy-selected threshold enters
    this construction.
    """

    if (hardware.kernel_memory_model != "npu_tile"
            or kernel.variant != "gemm" or kernel.dtype != "bf16"):
        return None
    # A mixed AIC/AIV launch has an additional cross-domain handoff and cannot
    # be represented as a pure A/B-load-to-Cube worker program.  This is a
    # physical execution distinction retained in the launch facts, not a
    # selector-name branch; leave it aggregate until its tile handoff is
    # source-closed.
    if kernel.launch.parameters["aiv_mixed"] == 1.0:
        return None
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k != 1 or tiling.pipeline_depth != 2:
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
        raise ValueError("910B GEMM requires positive tile dimensions")
    k = int(kernel.shape["k"])
    if k % tile_k:
        return None
    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    m_tiles = math.ceil(b * m / tile_m)
    n_tiles = math.ceil(n / tile_n)
    output_tiles = m_tiles * n_tiles
    source_program = _cann_gemm_launch_program(kernel)
    if source_program is not None:
        if (
                source_program.aic_workers != kernel.launch.grid_size
                or source_program.split_k != 1
                or source_program.worker_grid_k != 1
                or source_program.single_core_k != k
                or (source_program.base_m, source_program.base_n,
                    source_program.base_k) != (tile_m, tile_n, tile_k)):
            raise ValueError(
                "normalized CANN GEMM program disagrees with its launch")
        if (source_program.single_core_m % tile_m
                or source_program.single_core_n % tile_n):
            raise ValueError(
                "normalized CANN GEMM worker extent is not tile-aligned")
        if source_program.traversal_order == "persistent_n_then_m":
            if (source_program.single_core_m * source_program.worker_grid_m
                    < b * m
                    or source_program.single_core_n
                        * source_program.worker_grid_n < n):
                raise ValueError(
                    "normalized CANN GEMM worker grid does not cover its "
                    "output")
            worker_m_tiles = source_program.single_core_m // tile_m
            worker_n_tiles = source_program.single_core_n // tile_n
            output_tiles_per_worker = worker_m_tiles * worker_n_tiles
            a_descriptor_m_groups = math.ceil(
                worker_m_tiles / source_program.step_m)
            a_descriptor_k_groups = math.ceil(
                k // tile_k / source_program.a_l1_k_tiles)
            if source_program.a_l1_k_tiles == k // tile_k:
                # MatMulV2's full-A program copies each worker-local M slab
                # into its single A1 buffer before the N-major output loop.
                # Every later N tile reuses that resident slab; only B1 and
                # the L0/Cube recurrence advance with worker_n_tiles.
                a_descriptors_per_worker = (
                    a_descriptor_m_groups * a_descriptor_k_groups)
            else:
                a_descriptors_per_worker = (
                    worker_n_tiles * a_descriptor_m_groups
                    * a_descriptor_k_groups)
            b_descriptors_per_worker = (
                worker_m_tiles
                * math.ceil(worker_n_tiles / source_program.step_n)
                * math.ceil(k // tile_k / source_program.b_l1_k_tiles))
        elif source_program.traversal_order in {
                "m_then_n_snake", "n_then_m_snake"}:
            if (source_program.m_l2_regions * source_program.m_l2_block
                    < m_tiles
                    or source_program.n_l2_regions
                        * source_program.n_l2_block < n_tiles):
                raise ValueError(
                    "normalized CANN GEMM regions do not cover its output")
            worker_m_tiles = source_program.single_core_m // tile_m
            worker_n_tiles = source_program.single_core_n // tile_n
            output_tiles_per_worker = math.ceil(
                output_tiles / source_program.aic_workers)
            a_descriptors_per_worker = (
                output_tiles_per_worker
                * math.ceil(k // tile_k / source_program.a_l1_k_tiles))
            b_descriptors_per_worker = (
                output_tiles_per_worker
                * math.ceil(k // tile_k / source_program.b_l1_k_tiles))
        else:
            raise ValueError(
                "normalized CANN GEMM traversal order is unsupported")
        k_tiles = source_program.single_core_k // tile_k
        a_l1_k_tiles = source_program.a_l1_k_tiles
        b_l1_k_tiles = source_program.b_l1_k_tiles
        a_descriptor_bytes = (
            tile_m * source_program.step_m
            * a_l1_k_tiles * tile_k
            * int(kernel.shape["dtype_bytes"]))
        b_descriptor_bytes = (
            tile_n * source_program.step_n
            * b_l1_k_tiles * tile_k
            * int(kernel.shape["dtype_bytes"]))
        l1_size_bytes = hardware.resource_by_role("l1_cache").size_bytes
        if (source_program.depth_a1 * a_descriptor_bytes
                + source_program.depth_b1 * b_descriptor_bytes
                > l1_size_bytes):
            raise ValueError(
                "normalized CANN GEMM operand slots exceed physical L1")
        return _CannNpuGemmPipelineGeometry(
            worker_count=source_program.aic_workers,
            worker_m_tiles=worker_m_tiles,
            worker_n_tiles=worker_n_tiles,
            output_tiles_per_worker=output_tiles_per_worker,
            k_tiles=k_tiles,
            a_l1_k_tiles=a_l1_k_tiles,
            b_l1_k_tiles=b_l1_k_tiles,
            k_slabs=math.ceil(
                k_tiles / math.lcm(a_l1_k_tiles, b_l1_k_tiles)),
            a_descriptors_per_worker=a_descriptors_per_worker,
            b_descriptors_per_worker=b_descriptors_per_worker,
            a_descriptor_bytes=a_descriptor_bytes,
            b_descriptor_bytes=b_descriptor_bytes,
        )
    launched_workers = kernel.launch.grid_size
    if launched_workers > int(hardware.num_compute_units):
        raise ValueError("910B GEMM launch exceeds the available AIC count")
    # CANN may launch a full AIC grid for a short output-tile tail.  Empty
    # blocks contribute launch overhead but do not own an A/B-to-Cube program;
    # the shared tile pipeline therefore contains only non-empty workers.
    worker_count = min(output_tiles, launched_workers)
    if worker_count <= 0:
        raise ValueError("910B GEMM has no active AIC workers")
    output_tiles_per_worker = math.ceil(output_tiles / worker_count)
    block_candidates = []
    for worker_m_tiles in range(1, m_tiles + 1):
        minimum_worker_n_tiles = math.ceil(
            output_tiles_per_worker / worker_m_tiles)
        for worker_n_tiles in range(minimum_worker_n_tiles, n_tiles + 1):
            block_count = (
                math.ceil(m_tiles / worker_m_tiles)
                * math.ceil(n_tiles / worker_n_tiles))
            if block_count > worker_count:
                continue
            block_capacity = worker_m_tiles * worker_n_tiles
            a_extent = worker_m_tiles * tile_m
            b_extent = worker_n_tiles * tile_n
            block_candidates.append((
                worker_count * block_capacity - output_tiles,
                worker_count - block_count,
                abs(math.log(a_extent / b_extent)),
                -min(a_extent, b_extent),
                worker_m_tiles,
                worker_n_tiles,
            ))
    if not block_candidates:
        return None
    _, _, _, _, worker_m_tiles, worker_n_tiles = min(block_candidates)

    element_bytes = int(kernel.shape["dtype_bytes"])
    a_base_bytes = tile_m * tile_k * element_bytes
    b_base_bytes = tile_k * tile_n * element_bytes
    l1_size_bytes = hardware.resource_by_role("l1_cache").size_bytes
    # MatMulV2's static tiler caps both non-full-load A1/B1 K slabs at sixteen
    # Cube K blocks.  Other schedules retain the equal per-slot L1 partition.
    legacy_k_slab_tiles = (
        min(k // tile_k, _CANN_LEGACY_L1_K_SLAB_ELEMENTS // tile_k)
        if (kernel.launch.algorithm == "ascend_cann_matmul"
            and _CANN_LEGACY_L1_K_SLAB_ELEMENTS % tile_k == 0)
        else 0)
    legacy_pair_count = (
        output_tiles_per_worker
        * math.ceil(k // tile_k / legacy_k_slab_tiles)
        if legacy_k_slab_tiles > 0 else 0)
    legacy_signature = (
        tile_m, _CANN_LEGACY_L1_K_SLAB_ELEMENTS, k,
        tile_n, _CANN_LEGACY_L1_K_SLAB_ELEMENTS, k,
        worker_count, legacy_pair_count,
    )
    legacy_references = (
        hardware.aic_mte2_legacy_double_l0c_pair_service,
        hardware.aic_mte2_legacy_single_l0c_pair_control,
    )
    legacy_source_slab = any(
        reference is not None
        and legacy_signature == tuple(
            reference[name] for name in (
                "a_rows", "a_columns", "a_source_row_stride_elements",
                "b_rows", "b_columns", "b_source_row_stride_elements",
                "active_aic_groups", "pairs_per_worker"))
        for reference in legacy_references)
    if not legacy_source_slab:
        legacy_k_slab_tiles = 0
    source_a_l1_k_tiles = legacy_k_slab_tiles
    source_b_l1_k_tiles = legacy_k_slab_tiles
    source_a_buffer_depth = pipeline_depth
    source_b_buffer_depth = pipeline_depth
    small_m_service = hardware.aic_mte2_cold_small_m_stride_service
    if small_m_service is not None:
        if kernel.launch.algorithm == "ascend_cann_matmul":
            descriptors = small_m_service["descriptors"]
            a_columns = {
                columns
                for rows, columns, stride, active_groups in descriptors
                if (rows == tile_m and stride == k
                    and active_groups == worker_count)
            }
            b_columns = {
                columns
                for rows, columns, stride, active_groups in descriptors
                if (rows == tile_n and stride == n
                    and active_groups == worker_count)
            }
            paired_columns = a_columns & b_columns
            if len(paired_columns) > 1:
                raise ValueError("910B GEMM has ambiguous source L1 K slabs")
            if paired_columns:
                source_k_slab_elements, = paired_columns
                if source_k_slab_elements % tile_k:
                    raise ValueError(
                        "910B GEMM source L1 K slab is not tile-aligned")
                source_a_l1_k_tiles = source_k_slab_elements // tile_k
                source_b_l1_k_tiles = source_k_slab_elements // tile_k
    source_l1_bytes = (
        source_a_buffer_depth * source_a_l1_k_tiles * a_base_bytes
        + source_b_buffer_depth * source_b_l1_k_tiles * b_base_bytes)
    if (source_a_l1_k_tiles > 0 and source_b_l1_k_tiles > 0
            and source_l1_bytes <= l1_size_bytes):
        a_l1_k_tiles = source_a_l1_k_tiles
        b_l1_k_tiles = source_b_l1_k_tiles
    else:
        operand_budget_bytes = l1_size_bytes // (2 * pipeline_depth)
        a_l1_k_tiles = min(k // tile_k, operand_budget_bytes // a_base_bytes)
        b_l1_k_tiles = min(k // tile_k, operand_budget_bytes // b_base_bytes)
    if a_l1_k_tiles <= 0 or b_l1_k_tiles <= 0:
        return None
    k_tiles = k // tile_k
    k_slab_tiles = math.lcm(a_l1_k_tiles, b_l1_k_tiles)
    a_descriptor_bytes = a_base_bytes * a_l1_k_tiles
    b_descriptor_bytes = b_base_bytes * b_l1_k_tiles
    if (source_a_buffer_depth * a_descriptor_bytes
            + source_b_buffer_depth * b_descriptor_bytes > l1_size_bytes):
        raise ValueError("910B GEMM operand slots exceed physical L1 capacity")
    return _CannNpuGemmPipelineGeometry(
        worker_count=worker_count,
        worker_m_tiles=worker_m_tiles,
        worker_n_tiles=worker_n_tiles,
        output_tiles_per_worker=output_tiles_per_worker,
        k_tiles=k_tiles,
        a_l1_k_tiles=a_l1_k_tiles,
        b_l1_k_tiles=b_l1_k_tiles,
        k_slabs=math.ceil(k_tiles / k_slab_tiles),
        a_descriptors_per_worker=(
            output_tiles_per_worker * math.ceil(k_tiles / a_l1_k_tiles)),
        b_descriptors_per_worker=(
            output_tiles_per_worker * math.ceil(k_tiles / b_l1_k_tiles)),
        a_descriptor_bytes=a_descriptor_bytes,
        b_descriptor_bytes=b_descriptor_bytes,
    )


def _cann_npu_gemm_fixpipe_finite_service(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> str | None:
    """Return the exact source-shaped finite FixPipe service, if covered."""

    mechanism = hardware.aic_fixpipe_bf16_finite_stream_service
    if mechanism is None or kernel.dtype != "bf16":
        return None
    tile_m, tile_n, _, _ = _tile(kernel)
    signature = (tile_m, tile_n, geometry.worker_count)
    if _uses_cann_persistent_execution(kernel):
        covered = (
            signature in mechanism["cells"]
            and mechanism["minimum_tiles_per_worker"]
                <= geometry.output_tiles_per_worker
                <= mechanism["maximum_tiles_per_worker"])
    elif (_uses_cann_region_program(kernel)
          or kernel.launch.algorithm == _CANN_MATMUL_V3_ASW_65536_ALGORITHM):
        long_stream = hardware.aic_fixpipe_bf16_long_stream_service
        covered = (
            long_stream is not None
            and signature == (
                long_stream["rows"], long_stream["columns"],
                long_stream["active_aic_groups"])
            and long_stream["minimum_tiles_per_worker"]
                < geometry.output_tiles_per_worker
                <= long_stream["maximum_tiles_per_worker"])
    else:
        covered = False
    if not covered:
        return None
    return _CANN_FIXPIPE_FINITE_SERVICE_BY_GEOMETRY[signature]


def _cann_full_region_envelope_work(
        m_tiles: int, n_tiles: int,
        program: CannGemmLaunchProgram,
        ) -> tuple[
            tuple[tuple[int, int], ...],
            tuple[tuple[int, ...], ...],
            tuple[tuple[tuple[int, int], ...], ...],
        ] | None:
    """Return exact envelope work without expanding a full region schedule.

    A full region program has one output-tile shape, so the grouped-MatMul
    recurrence only needs every worker's tile count.  The source traversal is
    still reflected in those counts: each region assigns ``rounds - 1`` tiles
    to every worker and one additional tile to its rotating tail workers.
    Tails retain the explicit source expansion below because they introduce
    shape classes and source-order transitions.
    """

    if (m_tiles != program.m_l2_regions * program.m_l2_block
            or n_tiles != program.n_l2_regions * program.n_l2_block):
        return None
    workers = program.aic_workers
    tiles_per_region = program.m_l2_block * program.n_l2_block
    rounds = math.ceil(tiles_per_region / workers)
    tail_workers = tiles_per_region % workers
    region_count = program.m_l2_regions * program.n_l2_regions
    if tail_workers == 0:
        counts = [region_count * rounds] * workers
    else:
        counts = [region_count * (rounds - 1)] * workers
        for region in range(region_count):
            block_start = region * tail_workers % workers
            for relative_worker in range(tail_workers):
                counts[(relative_worker + block_start) % workers] += 1
    if sum(counts) != m_tiles * n_tiles:
        raise AssertionError("CANN full region worker counts do not conserve tiles")
    return (
        ((program.base_m, program.base_n),),
        tuple((count,) for count in counts),
        tuple(
            ((0, count),) if count else ()
            for count in counts),
    )


def _cann_persistent_envelope_work(
        rows: int, n: int,
        program: CannGemmLaunchProgram,
        ) -> tuple[
            tuple[tuple[int, int], ...],
            tuple[tuple[int, ...], ...],
            tuple[tuple[tuple[int, int], ...], ...],
        ]:
    """Return every MatMulV2 worker's compact output-tile program.

    The persistent source assigns one rectangular M/N domain to each AIC and
    shifts the final domain inward when a grid edge is not factorial.  That
    overlap is real work: all launched workers execute their retained local
    tile loop.  Preserve each worker's source order and logical edge shapes so
    the finite A1/B1, MTE1, Cube and FixPipe recurrence is governed by the
    longest worker instead of aggregate work divided by the active AIC count.
    """

    if program.traversal_order != "persistent_n_then_m":
        raise ValueError(
            "CANN persistent envelope requires worker-local N/M order")
    aligned_m = math.ceil(rows / 16) * 16
    aligned_n = math.ceil(n / 16) * 16
    for extent, grid, dimension, name in (
            (program.single_core_m, program.worker_grid_m, aligned_m, "M"),
            (program.single_core_n, program.worker_grid_n, aligned_n, "N")):
        if extent * grid < dimension:
            raise ValueError(
                f"CANN persistent envelope does not cover its {name} axis")
    if program.worker_grid_k != 1:
        raise ValueError(
            "CANN persistent envelope does not support split-K workers")

    class_shapes: list[tuple[int, int]] = []
    class_index: dict[tuple[int, int], int] = {}
    class_sequences: list[list[int]] = []
    mn_workers = program.worker_grid_m * program.worker_grid_n
    for worker in range(program.aic_workers):
        k_worker, mn_worker = divmod(worker, mn_workers)
        if k_worker:
            raise ValueError(
                "CANN persistent envelope found an unexpected K worker")
        n_worker, m_worker = divmod(mn_worker, program.worker_grid_m)
        m_start = max(0, min(
            m_worker * program.single_core_m,
            aligned_m - program.single_core_m))
        n_start = max(0, min(
            n_worker * program.single_core_n,
            aligned_n - program.single_core_n))
        sequence = []
        for local_n in range(0, program.single_core_n, program.base_n):
            for local_m in range(0, program.single_core_m, program.base_m):
                logical_m = min(
                    program.base_m, rows - m_start - local_m)
                logical_n = min(
                    program.base_n, n - n_start - local_n)
                if logical_m <= 0 or logical_n <= 0:
                    raise ValueError(
                        "CANN persistent envelope contains an empty tile")
                shape = (logical_m, logical_n)
                if shape not in class_index:
                    class_index[shape] = len(class_shapes)
                    class_shapes.append(shape)
                sequence.append(class_index[shape])
        class_sequences.append(sequence)

    counts_by_worker = []
    source_class_runs_by_worker = []
    for sequence in class_sequences:
        counts = [0] * len(class_shapes)
        runs: list[tuple[int, int]] = []
        for current_class in sequence:
            counts[current_class] += 1
            if runs and runs[-1][0] == current_class:
                runs[-1] = (current_class, runs[-1][1] + 1)
            else:
                runs.append((current_class, 1))
        counts_by_worker.append(tuple(counts))
        source_class_runs_by_worker.append(tuple(runs))
    return (
        tuple(class_shapes), tuple(counts_by_worker),
        tuple(source_class_runs_by_worker),
    )


def _cann_npu_gemm_analytical_envelope(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry,
        traffic: GemmSourceOrderTraffic | None,
        ) -> CannGroupedMatmulEnvelopeDemand | None:
    """Build the common CANN A1/B1-to-Cube recurrence for MatMul.

    Both persistent MatMulV2 and region MatMulV3 invoke one complete Matmul
    program per output tile.  Reuse the same analytical K recurrence as
    GroupedMatmul: independent A1/B1 credits, L0 ping-pong, MTE1 and Cube
    service, and source-ordered output completion.  The retained launch program
    supplies every worker's full/tail tile sequence, so the longest finite
    worker determines completion without aggregate worker averaging.
    """

    program = _cann_gemm_launch_program(kernel)
    if program is None or traffic is None:
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    persistent_program = _uses_cann_persistent_program(kernel)
    region_program = _uses_cann_region_program(kernel)
    if (not persistent_program and not region_program) or pipeline_depth != 2:
        return None
    if (region_program
            and ((tile_m, tile_n, tile_k) != (128, 256, 64)
                 or hardware.aic_grouped_matmul_mte1_pair_completion_service
                    is None)):
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    if k % tile_k:
        return None
    m_tiles = math.ceil(rows / tile_m)
    n_tiles = math.ceil(n / tile_n)
    output_tiles = m_tiles * n_tiles
    if output_tiles < geometry.worker_count:
        return None
    if persistent_program:
        (class_shapes, counts_by_worker,
         source_class_runs_by_worker) = _cann_persistent_envelope_work(
             rows, n, program)
    else:
        full_region_work = _cann_full_region_envelope_work(
            m_tiles, n_tiles, program)
        if full_region_work is not None:
            (class_shapes, counts_by_worker,
             source_class_runs_by_worker) = full_region_work
        else:
            assignments = cann_gemm_source_tile_assignments(
                m_tiles, n_tiles,
                program.m_l2_regions, program.n_l2_regions,
                program.m_l2_block, program.n_l2_block,
                program.traversal_order, program.aic_workers)
            class_shapes = []
            class_index = {}
            counts_by_worker = [[] for _ in range(geometry.worker_count)]
            source_class_runs_by_worker = [
                [] for _ in range(geometry.worker_count)]
            for worker, m_index, n_index in assignments:
                shape = (
                    min(tile_m, rows - m_index * tile_m),
                    min(tile_n, n - n_index * tile_n),
                )
                if shape not in class_index:
                    class_index[shape] = len(class_shapes)
                    class_shapes.append(shape)
                    for counts in counts_by_worker:
                        counts.append(0)
                class_id = class_index[shape]
                counts_by_worker[worker][class_id] += 1
                runs = source_class_runs_by_worker[worker]
                if runs and runs[-1][0] == class_id:
                    runs[-1] = (class_id, runs[-1][1] + 1)
                else:
                    runs.append((class_id, 1))
            class_shapes = tuple(class_shapes)
            counts_by_worker = tuple(
                tuple(counts) for counts in counts_by_worker)
            source_class_runs_by_worker = tuple(
                tuple(runs) for runs in source_class_runs_by_worker)
    operand_l2_bytes = traffic.a_l2_bytes + traffic.b_l2_bytes
    if operand_l2_bytes <= 0:
        raise ValueError("CANN GEMM source program has no operand L2 traffic")
    for requested, missed, operand in (
            (traffic.a_l2_bytes, traffic.a_dram_bytes, "A"),
            (traffic.b_l2_bytes, traffic.b_dram_bytes, "B")):
        if not 0 <= missed <= requested:
            raise ValueError(
                f"CANN GEMM {operand} LRU traffic is not conservative")
    return CannGroupedMatmulEnvelopeDemand(
        worker_count=geometry.worker_count,
        zero_expert_count=0,
        class_shapes=class_shapes,
        counts_by_worker=counts_by_worker,
        source_class_runs_by_worker=source_class_runs_by_worker,
        projection_k=k,
        projection_n=n,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_step_k_tiles=program.a_l1_k_tiles,
        b_step_k_tiles=program.b_l1_k_tiles,
        a1_buffer_slots=program.depth_a1,
        b1_buffer_slots=program.depth_b1,
        l0a_buffer_slots=2,
        l0b_buffer_slots=2,
        l0c_buffer_slots=(
            _cann_legacy_l0c_pipeline_depth(kernel, hardware)
            if persistent_program else 1),
        dtype=kernel.dtype,
        element_bytes=int(kernel.shape["dtype_bytes"]),
        tensor_operations=float(math.fsum(
            sum(counts[index] for counts in counts_by_worker)
            * 2 * math.ceil(logical_m / 16) * 16
            * math.ceil(logical_n / 16) * 16 * k
            for index, (logical_m, logical_n) in enumerate(class_shapes))),
        a_l2_hit_fraction=(
            1.0 - traffic.a_dram_bytes / traffic.a_l2_bytes),
        b_l2_hit_fraction=(
            1.0 - traffic.b_dram_bytes / traffic.b_l2_bytes),
        source_kind="matmul_v2" if persistent_program else "matmul_v3",
        output_path_name="fixpipe_write",
    )


def _split_cann_npu_gemm_fixpipe_write_path(
        inp: PredictionInput, kernel: KernelSpec,
        geometry: _CannNpuGemmPipelineGeometry) -> PredictionInput:
    """Separate source-known AIC output bytes from the operand read path."""

    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "mte2_read")
    if len(read_paths) != 1:
        raise ValueError(
            "CANN GEMM finite FixPipe requires one MTE2 operand path")
    if any(path.name == "fixpipe_write" for path in inp.memory_paths):
        raise ValueError("CANN GEMM FixPipe path must attach exactly once")
    read_path, = read_paths
    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    output_bytes = float(b * m * n * int(kernel.shape["dtype_bytes"]))
    input_nodes = []
    fixpipe_nodes = []
    found_dram = False
    for node in read_path.nodes:
        if node.resource_role != "dram":
            input_nodes.append(node)
            continue
        found_dram = True
        input_bytes = node.bytes - output_bytes
        if input_bytes <= 0.0:
            raise ValueError(
                "CANN GEMM output bytes exhaust the operand DRAM path")
        input_nodes.append(dataclasses.replace(node, bytes=input_bytes))
        fixpipe_nodes.append(dataclasses.replace(node, bytes=output_bytes))
    if not found_dram:
        # A fully resident operand stream legitimately has no read DRAM
        # continuation.  Its terminal output still appears as the directional
        # dirty-L2 write path attached from the same source-order trace; make
        # that proven write the FixPipe completion path rather than inventing
        # a read dependency or charging the bytes twice.
        writebacks = tuple(
            path for path in inp.memory_paths
            if path.name == "l2_dirty_writeback")
        if len(writebacks) != 1:
            raise ValueError(
                "CANN GEMM resident operand path requires one output "
                "writeback continuation")
        writeback, = writebacks
        if (len(writeback.nodes) != 1
                or writeback.nodes[0].resource_role != "dram"
                or not math.isclose(
                    writeback.nodes[0].bytes, output_bytes,
                    rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError(
                "CANN GEMM resident output writeback must equal FixPipe "
                "output bytes")
        typed_read_path = dataclasses.replace(
            read_path, nodes=tuple(input_nodes))
        fixpipe_path = dataclasses.replace(writeback, name="fixpipe_write")
        return dataclasses.replace(
            inp,
            memory_paths=tuple(
                typed_read_path if path is read_path else path
                for path in inp.memory_paths
                if path is not writeback) + (fixpipe_path,),
        )
    typed_read_path = dataclasses.replace(
        read_path, nodes=tuple(input_nodes))
    fixpipe_path = MemoryPathDemand(
        name="fixpipe_write",
        access_kind="write",
        initial_transactions=None,
        nodes=tuple(fixpipe_nodes),
        source=MemorySourceDemand(
            active_units=geometry.worker_count,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    return dataclasses.replace(
        inp,
        memory_paths=tuple(
            typed_read_path if path is read_path else path
            for path in inp.memory_paths) + (fixpipe_path,),
    )


def _cann_npu_gemm_operand_replay_bytes(
        kernel: KernelSpec, hardware) -> float | None:
    """Return common padded MTE2 A/B replay bytes for one 910B GEMM launch."""

    geometry = _cann_npu_gemm_pipeline_geometry(kernel, hardware)
    if geometry is None:
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    base_pair_bytes = (
        tile_m * tile_k * element_bytes
        + tile_k * tile_n * element_bytes)
    return float(
        geometry.worker_count * geometry.output_tiles_per_worker
        * geometry.k_tiles * base_pair_bytes)


def _cann_npu_mte2_affine_service(
        hardware, active_aic_groups: int,
        *, stream: bool) -> Tuple[float, float] | None:
    """Return startup ns and byte/s capacity of one MTE2 operand channel.

    The final two independent mechanism points identify the physical
    startup-plus-bytes/bandwidth regime.  Forward consumes the resulting
    affine channel service; it does not use a production-kernel duration.
    """

    mechanism = hardware.aic_mte2_nd2nz_service
    if mechanism is None:
        return None
    points = (
        mechanism["stream_descriptor_bytes"]
        if stream else mechanism["descriptor_bytes"])
    lower_bytes, upper_bytes = points[-2:]
    service = (
        hardware.aic_mte2_nd2nz_stream_time_ns
        if stream else hardware.aic_mte2_nd2nz_serial_time_ns)
    lower_ns = service(lower_bytes, active_aic_groups)
    upper_ns = service(upper_bytes, active_aic_groups)
    ns_per_byte = (upper_ns - lower_ns) / (upper_bytes - lower_bytes)
    if ns_per_byte <= 0:
        raise ValueError("910B MTE2 large-descriptor bandwidth must be positive")
    startup_ns = lower_ns - ns_per_byte * lower_bytes
    # Linear interpolation across the independently measured active-AIC axis
    # can put a sub-nanosecond stream intercept just below zero.  A startup is
    # physically non-negative; reject material extrapolation instead of
    # turning measurement quantization into a negative service.
    if startup_ns < -1.0:
        raise ValueError("910B MTE2 large-descriptor startup cannot be negative")
    return max(startup_ns, 0.0), 1e9 / ns_per_byte


def _cann_npu_mte2_descriptor_channel_time_ns(
        hardware, rows: int, columns: int, active_aic_groups: int,
        descriptor_count: int, *, delayed_by_one_issue: bool,
        source_row_stride_elements: int | None = None,
        use_gemm_large_resident_service: bool = False,
        dram_miss_fraction: float | None = None) -> float | None:
    """Return one independent A1 or B1 descriptor-channel completion.

    Large descriptors use the directly measured resident completion and
    two-buffer issue interval at their physical ND2NZ geometry.  Smaller or
    unmeasured geometries retain the common affine byte service.  The B
    channel begins after the source's first A issue, represented by one extra
    issue interval rather than serializing both operand streams.
    """

    for name, value in (
            ("rows", rows), ("columns", columns),
            ("active_aic_groups", active_aic_groups),
            ("descriptor_count", descriptor_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"910B MTE2 descriptor {name} must be an integer")
        if value <= 0:
            raise ValueError(f"910B MTE2 descriptor {name} must be positive")
    if source_row_stride_elements is not None:
        if (isinstance(source_row_stride_elements, bool)
                or not isinstance(source_row_stride_elements, int)):
            raise TypeError(
                "910B MTE2 descriptor source row stride must be an integer")
        if source_row_stride_elements < columns:
            raise ValueError(
                "910B MTE2 descriptor source row stride must cover columns")
    if not isinstance(use_gemm_large_resident_service, bool):
        raise TypeError(
            "910B MTE2 GEMM large-resident selector must be boolean")
    use_gemm_large_resident_service = (
        use_gemm_large_resident_service
        and source_row_stride_elements is not None
        and hardware.aic_mte2_gemm_large_resident_supports(
            rows, columns, source_row_stride_elements, active_aic_groups))
    if use_gemm_large_resident_service:
        # This service is the local descriptor completion after source bytes
        # reach shared L2.  The common memory path independently prices the
        # LRU-derived DRAM misses, so importing the miss fraction here would
        # serialize two simultaneous constraints on the same bytes.
        serial_ns = hardware.aic_mte2_gemm_large_resident_serial_time_ns(
            rows, columns, source_row_stride_elements, active_aic_groups)
        stream_ns = hardware.aic_mte2_gemm_large_resident_stream_time_ns(
            rows, columns, source_row_stride_elements, active_aic_groups)
        return (
            serial_ns
            + (descriptor_count - 1 + int(delayed_by_one_issue)) * stream_ns)
    use_large_service = (
        hardware.aic_mte2_nd2nz_large_contiguous_supports(
            rows, columns, active_aic_groups)
        and (dram_miss_fraction is None
             or dram_miss_fraction == 0.0
             or hardware.aic_mte2_nd2nz_large_contiguous_cold_supports(
                 rows, columns, active_aic_groups)))
    if use_large_service:
        if dram_miss_fraction is None or dram_miss_fraction == 0.0:
            serial_ns = (
                hardware.aic_mte2_nd2nz_large_contiguous_serial_time_ns(
                    rows, columns, active_aic_groups))
            stream_ns = (
                hardware.aic_mte2_nd2nz_large_contiguous_stream_time_ns(
                    rows, columns, active_aic_groups))
        else:
            serial_ns = (
                hardware
                .aic_mte2_nd2nz_large_contiguous_completion_time_ns(
                    rows, columns, active_aic_groups,
                    dram_miss_fraction, stream=False))
            stream_ns = (
                hardware
                .aic_mte2_nd2nz_large_contiguous_completion_time_ns(
                    rows, columns, active_aic_groups,
                    dram_miss_fraction, stream=True))
        return (
            serial_ns
            + (descriptor_count - 1 + int(delayed_by_one_issue)) * stream_ns)
    serial_bridge_ns = (
        hardware.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            rows * columns * 2, active_aic_groups, stream=False,
            dram_miss_fraction=dram_miss_fraction))
    stream_bridge_ns = (
        hardware.aic_mte2_nd2nz_large_contiguous_extension_time_ns(
            rows * columns * 2, active_aic_groups, stream=True,
            dram_miss_fraction=dram_miss_fraction))
    if serial_bridge_ns is not None or stream_bridge_ns is not None:
        if serial_bridge_ns is None or stream_bridge_ns is None:
            raise AssertionError(
                "910B MTE2 contiguous bridge service is incomplete")
        return (
            serial_bridge_ns
            + (descriptor_count - 1 + int(delayed_by_one_issue))
            * stream_bridge_ns)
    serial_service = _cann_npu_mte2_affine_service(
        hardware, active_aic_groups, stream=False)
    stream_service = _cann_npu_mte2_affine_service(
        hardware, active_aic_groups, stream=True)
    if serial_service is None or stream_service is None:
        return None
    serial_startup_ns, _ = serial_service
    stream_startup_ns, single_channel_capacity_per_s = stream_service
    descriptor_bytes = rows * columns * 2
    return (
        serial_startup_ns
        + descriptor_count * descriptor_bytes
        / single_channel_capacity_per_s * 1e9
        + (descriptor_count - 1 + int(delayed_by_one_issue))
        * stream_startup_ns)


def _cann_npu_gemm_shared_source_queue_time_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry,
        program_traffic: GemmSourceOrderTraffic | None,
        dram_capacity_per_s: float | None) -> float | None:
    """Return one source-ordered A1/B1 queue's analytical completion.

    CANN issues A1 and B1 descriptors through the same AIC MTE2 pipe.  Their
    local resident services therefore add in source order; they are not two
    independent channels.  LRU-derived DRAM refills occupy the same finite
    source program and add their bandwidth service, while the ordinary memory
    path continues to enforce the full-chip L2/DRAM envelope independently.
    """

    program = _cann_gemm_launch_program(kernel)
    if program is None or program_traffic is None:
        return None
    tile_m, tile_n, _, pipeline_depth = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    a_rows = tile_m * program.step_m
    b_rows = tile_n * program.step_n
    if (geometry.a_descriptor_bytes % (a_rows * element_bytes)
            or geometry.b_descriptor_bytes % (b_rows * element_bytes)):
        raise ValueError(
            "910B GEMM source-queue descriptors are not rectangular")
    a_columns = geometry.a_descriptor_bytes // (a_rows * element_bytes)
    b_columns = geometry.b_descriptor_bytes // (b_rows * element_bytes)
    source_stride = int(kernel.shape["k"])
    if not hardware.aic_mte2_gemm_shared_source_queue_supports(
            dtype=kernel.dtype,
            active_aic_groups=geometry.worker_count,
            operand_buffer_depth=pipeline_depth,
            traversal_order=program.traversal_order,
            a_rows=a_rows,
            a_columns=a_columns,
            a_source_row_stride_elements=source_stride,
            b_rows=b_rows,
            b_columns=b_columns,
            b_source_row_stride_elements=source_stride):
        return None
    a_serial_ns = hardware.aic_mte2_gemm_shared_source_descriptor_time_ns(
        a_rows, a_columns, source_stride, 0.0, stream=False)
    a_stream_ns = hardware.aic_mte2_gemm_shared_source_descriptor_time_ns(
        a_rows, a_columns, source_stride, 0.0, stream=True)
    b_stream_ns = hardware.aic_mte2_gemm_shared_source_descriptor_time_ns(
        b_rows, b_columns, source_stride, 0.0, stream=True)
    if dram_capacity_per_s is None:
        dram_capacity_per_s = hardware.resource_by_role(
            "dram").effective_capacity(
                geometry.worker_count,
                sm_freq_hz=hardware.nominal_sm_freq_hz,
                dram_freq_hz=hardware.nominal_dram_freq_hz)
    if (not math.isfinite(dram_capacity_per_s)
            or dram_capacity_per_s <= 0.0):
        raise ValueError("910B GEMM source-queue DRAM capacity must be positive")
    return math.fsum((
        a_serial_ns,
        (geometry.a_descriptors_per_worker - 1) * a_stream_ns,
        geometry.b_descriptors_per_worker * b_stream_ns,
        (program_traffic.a_dram_bytes + program_traffic.b_dram_bytes)
        / dram_capacity_per_s * 1.0e9,
    ))


def _cann_legacy_l0c_pipeline_depth(kernel: KernelSpec, hardware) -> int:
    """Derive the legacy MatMulV2 accumulator depth from L0C capacity."""

    tile_m, tile_n, _, _ = _tile(kernel)
    accumulator_bytes = tile_m * tile_n * 4
    l0c_size_bytes = hardware.resource_by_role("l0c_buffer").size_bytes
    if accumulator_bytes > l0c_size_bytes:
        raise ValueError("910B GEMM accumulator tile exceeds physical L0C")
    return 2 if 2 * accumulator_bytes <= l0c_size_bytes else 1


def _cann_legacy_l1_pair_signature(
        kernel: KernelSpec,
        geometry: _CannNpuGemmPipelineGeometry) -> Tuple[int, ...] | None:
    """Return one legacy pair program's physical source signature."""

    if (not _uses_cann_persistent_program(kernel)
            and not (_cann_gemm_launch_program(kernel) is None
                     and kernel.launch.algorithm == "ascend_cann_matmul")
            or geometry.a_descriptors_per_worker
            != geometry.b_descriptors_per_worker):
        return None
    tile_m, tile_n, _, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    return (
        tile_m,
        geometry.a_descriptor_bytes // (tile_m * element_bytes),
        int(kernel.shape["k"]),
        tile_n,
        geometry.b_descriptor_bytes // (tile_n * element_bytes),
        int(kernel.shape["k"]),
        geometry.worker_count,
        geometry.a_descriptors_per_worker,
    )


def _cann_legacy_double_l0c_pair_signature(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> Tuple[int, ...] | None:
    """Return the exact measured legacy pair signature when supported."""

    service = hardware.aic_mte2_legacy_double_l0c_pair_service
    if (service is None
            or _cann_legacy_l0c_pipeline_depth(kernel, hardware) != 2):
        return None
    signature = _cann_legacy_l1_pair_signature(kernel, geometry)
    if signature is None:
        return None
    measured = tuple(
        service[name] for name in (
            "a_rows", "a_columns", "a_source_row_stride_elements",
            "b_rows", "b_columns", "b_source_row_stride_elements",
            "active_aic_groups", "pairs_per_worker"))
    return signature if signature == measured else None


def _cann_npu_gemm_cold_pair_stream_supported(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> bool:
    """Return whether the exact cold MatMulV3 pair service applies."""

    service = hardware.aic_mte2_cold_operand_pair_stream_service
    if service is None or not _uses_cann_persistent_execution(kernel):
        return False
    source_program = _cann_gemm_launch_program(kernel)
    # This service measures one base-K descriptor for each operand in every
    # pair. A typed launch that coalesces multiple base-K tiles into one L1
    # descriptor has different producer credits and descriptor issue work.
    if (source_program is not None
            and (source_program.a_l1_k_tiles != 1
                 or source_program.b_l1_k_tiles != 1)):
        return False
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if (tile_m, tile_k, tile_k, tile_n) != tuple(
            service[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns")):
        return False
    pair_count = geometry.output_tiles_per_worker * geometry.k_tiles
    if (geometry.worker_count != service["active_aic_groups"]
            or geometry.worker_n_tiles < geometry.worker_m_tiles
            or not (service["minimum_pairs_per_worker"] <= pair_count
                    <= service["maximum_pairs_per_worker"])):
        return False
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    operand_working_set_bytes = (
        b * m * k + k * n) * int(kernel.shape["dtype_bytes"])
    return operand_working_set_bytes > hardware.l2_size_bytes


def _cann_npu_gemm_mixed_ring_pair_time_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float | None:
    """Return the source-ordered resident-A/cold-B pair completion time."""

    mechanism = hardware.aic_mte2_mixed_ring_long_overlap_domain
    if (mechanism is None or not _uses_cann_persistent_execution(kernel)
            or kernel.dtype != mechanism["dtype"]):
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    source_program = _cann_gemm_launch_program(kernel)
    # The mixed-ring probe issues one base-K A descriptor followed by one
    # base-K B descriptor for every pair. A typed persistent program may
    # retain several base-K tiles in A1/B1 and issue one wider descriptor;
    # decomposing that descriptor back into synthetic base-tile pairs changes
    # the producer-credit lifetime and cannot reuse the pair service.
    if (source_program is not None
            and (source_program.a_l1_k_tiles != 1
                 or source_program.b_l1_k_tiles != 1)):
        return None
    pair_count = geometry.output_tiles_per_worker * geometry.k_tiles
    if ((tile_m, tile_k, tile_k, tile_n) != tuple(
            mechanism[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns"))
            or geometry.worker_count != mechanism["active_aic_groups"]
            or not (mechanism["minimum_service_pairs_per_worker"] <= pair_count
                    <= mechanism["maximum_service_pairs_per_worker"])
            or geometry.worker_n_tiles <= geometry.worker_m_tiles):
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    element_bytes = int(kernel.shape["dtype_bytes"])
    a_working_set_bytes = b * m * k * element_bytes
    b_working_set_bytes = k * n * element_bytes
    if not (a_working_set_bytes <= hardware.l2_size_bytes
            < b_working_set_bytes):
        return None
    return hardware.aic_mte2_mixed_ring_pair_time_ns(
        tile_m, tile_k, tile_k, tile_n,
        geometry.worker_count, pair_count, "resident", "cold")


def _cann_npu_gemm_cold_n_major_b_pair_time_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float | None:
    """Return the measured cold pair service with physical N-major B stride.

    The cold pair probes bound a source program that issues one base-K A
    descriptor and one base-K B descriptor for each pair.  The reference is
    not a bound for a program that retains either operand across several base
    K tiles: that changes both the MTE2 descriptor lifecycle and the point at
    which the MTE1 pair becomes available.  Intermediate source-selected A
    rows use the affine byte service between the independent endpoints only
    after this descriptor topology has been established from the launch
    program.
    """

    service = hardware.aic_mte2_cold_n_major_b_stride_service
    upper = hardware.aic_mte2_cold_operand_pair_stream_service
    if (service is None or upper is None
            or not _uses_cann_persistent_execution(kernel)):
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    pair_count = geometry.output_tiles_per_worker * geometry.k_tiles
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    signature = (
        tile_k, tile_k, tile_n, n, geometry.worker_count, pair_count)
    expected = tuple(
        service[name] for name in (
            "a_columns", "b_rows", "b_columns",
            "b_source_row_stride_elements", "active_aic_groups",
            "pairs_per_worker"))
    operand_working_set_bytes = (
        b * m * k + k * n) * int(kernel.shape["dtype_bytes"])
    lower_a_rows = service["a_rows"]
    upper_a_rows = upper["a_rows"]
    upper_geometry = tuple(
        upper[name] for name in ("a_columns", "b_rows", "b_columns"))
    lower_geometry = (
        service["a_columns"], service["b_rows"], service["b_columns"])
    if (signature != expected
            or upper_geometry != lower_geometry
            or not lower_a_rows <= tile_m <= upper_a_rows
            # The reference's 64x256 B descriptor is one base-K tile wide,
            # and its paired A request has the same one-base-K lifetime.
            # A wider A1/B1 descriptor is a different source program even if
            # its later MTE1 work decomposes into identical base-K pairs.
            or geometry.a_l1_k_tiles != 1
            or geometry.b_l1_k_tiles != 1
            or geometry.worker_n_tiles < geometry.worker_m_tiles
            or operand_working_set_bytes <= hardware.l2_size_bytes):
        return None
    lower_total_ns = hardware.aic_mte2_cold_n_major_b_pair_time_ns(
        lower_a_rows, tile_k, tile_k, tile_n, n,
        geometry.worker_count, pair_count)
    row_fraction = (
        (tile_m - lower_a_rows) / (upper_a_rows - lower_a_rows))
    extra_pair_ns = row_fraction * (
        upper["continuous_ns_per_pair"] - service["contiguous_pair_ns"])
    return lower_total_ns + pair_count * extra_pair_ns


def _cann_npu_mte2_serial_service(hardware, geometry):
    """Return the serial AIC MTE2 affine service for one worker count.

    The legacy and small-M stride penalties all compare a serial descriptor
    stream against the resident one, so they read the same serial service;
    None means the service is absent and the caller contributes no penalty.
    """

    return _cann_npu_mte2_affine_service(
        hardware, geometry.worker_count, stream=False)


def _cann_npu_gemm_cold_legacy_stride_penalty_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float:
    """Return exact serial ND2NZ stride penalties for legacy descriptors."""

    service = hardware.aic_mte2_cold_legacy_k14336_stride_service
    if (service is None or kernel.launch.algorithm != "ascend_cann_matmul"
            or kernel.dtype != "bf16"
            or geometry.worker_count != service["active_aic_groups"]):
        return 0.0
    serial_service = _cann_npu_mte2_serial_service(hardware, geometry)
    if serial_service is None:
        return 0.0
    serial_startup_ns, serial_capacity_per_s = serial_service
    tile_m, tile_n, _, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    source_row_stride_elements = int(kernel.shape["k"])
    candidates = (
        ("a", tile_m, geometry.a_descriptor_bytes // (
            tile_m * element_bytes), geometry.a_descriptor_bytes,
         geometry.a_descriptors_per_worker),
        ("b", tile_n, geometry.b_descriptor_bytes // (
            tile_n * element_bytes), geometry.b_descriptor_bytes,
         geometry.b_descriptors_per_worker),
    )
    penalty_ns = 0.0
    for operand, rows, columns, descriptor_bytes, descriptor_count in candidates:
        measured = service["descriptors"][operand]
        signature = (rows, columns, source_row_stride_elements)
        expected = tuple(measured[name] for name in (
            "rows", "columns", "source_row_stride_elements"))
        if signature != expected:
            continue
        if descriptor_bytes != measured["copied_bytes_per_descriptor"]:
            raise ValueError(
                "legacy K14336 descriptor byte geometry is inconsistent")
        contiguous_serial_ns = (
            serial_startup_ns
            + descriptor_bytes / serial_capacity_per_s * 1e9)
        serial_stride_delta_ns = (
            measured["serial_completion_ns"] - contiguous_serial_ns)
        if serial_stride_delta_ns <= 0:
            raise ValueError(
                "legacy K14336 stride must have a positive serial penalty")
        penalty_ns += descriptor_count * serial_stride_delta_ns
    return penalty_ns


def _cann_npu_gemm_cold_v3_stride_penalty_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float:
    """Return the exact serial stride penalty for a measured V3 B1 tile.

    A normalized persistent launch describes one worker-wide B1 descriptor,
    not one Cube base tile.  Derive that descriptor from the physical worker
    extent and L1 K depth, then count its repetitions across the worker's M
    stripes and K slabs.  Legacy launches retain their base-tile geometry.
    """

    service = hardware.aic_mte2_cold_v3_k1536_n5120_stride_service
    if (service is None or not _uses_cann_persistent_execution(kernel)
            or kernel.dtype != "bf16"):
        return 0.0
    element_bytes = int(kernel.shape["dtype_bytes"])
    program = _cann_gemm_launch_program(kernel)
    if program is not None and _uses_cann_persistent_program(kernel):
        rows = program.single_core_n
        columns = program.b_l1_k_tiles * program.base_k
        descriptor_bytes = rows * columns * element_bytes
        descriptor_count = (
            geometry.worker_m_tiles
            * math.ceil(program.single_core_k / columns))
    else:
        rows = _tile(kernel)[1]
        columns = geometry.b_descriptor_bytes // (rows * element_bytes)
        descriptor_bytes = geometry.b_descriptor_bytes
        descriptor_count = geometry.b_descriptors_per_worker
    signature = (
        rows,
        columns,
        int(kernel.shape["n"]),
        geometry.worker_count,
    )
    descriptors = service["descriptors"]
    if signature not in descriptors:
        return 0.0
    measured = descriptors[signature]
    if descriptor_bytes != measured["copied_bytes_per_descriptor"]:
        raise ValueError("V3 K1536 B1 descriptor byte geometry is inconsistent")
    serial_service = _cann_npu_mte2_serial_service(hardware, geometry)
    if serial_service is None:
        return 0.0
    serial_startup_ns, serial_capacity_per_s = serial_service
    contiguous_serial_ns = (
        serial_startup_ns
        + descriptor_bytes / serial_capacity_per_s * 1e9)
    serial_stride_delta_ns = (
        measured["serial_completion_ns"] - contiguous_serial_ns)
    if serial_stride_delta_ns <= 0:
        raise ValueError("V3 K1536 B1 stride must have a positive penalty")
    return descriptor_count * serial_stride_delta_ns


def _cann_npu_gemm_cold_small_m_stride_penalty_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float:
    """Return exact positive stride deltas for remaining small-M tiles."""

    service = hardware.aic_mte2_cold_small_m_stride_service
    if service is None or kernel.dtype != "bf16":
        return 0.0
    serial_service = _cann_npu_mte2_serial_service(hardware, geometry)
    if serial_service is None:
        return 0.0
    serial_startup_ns, serial_capacity_per_s = serial_service
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    candidates = (
        (tile_m, geometry.a_descriptor_bytes // (tile_m * element_bytes),
         int(kernel.shape["k"]), geometry.a_descriptor_bytes,
         geometry.a_descriptors_per_worker),
        (tile_n, geometry.b_descriptor_bytes // (tile_n * element_bytes),
         int(kernel.shape["n"]), geometry.b_descriptor_bytes,
         geometry.b_descriptors_per_worker),
    )
    penalty_ns = 0.0
    for rows, columns, stride, descriptor_bytes, count in candidates:
        signature = (rows, columns, stride, geometry.worker_count)
        if signature not in service["descriptors"]:
            continue
        measured = service["descriptors"][signature]
        if descriptor_bytes != measured["copied_bytes_per_descriptor"]:
            raise ValueError("small-M descriptor byte geometry is inconsistent")
        contiguous_ns = (
            serial_startup_ns
            + descriptor_bytes / serial_capacity_per_s * 1e9)
        delta_ns = measured["serial_completion_ns"] - contiguous_ns
        if delta_ns <= 0:
            raise ValueError("published small-M stride penalty is not positive")
        penalty_ns += count * delta_ns
    return penalty_ns


def _cann_npu_gemm_finite_b1_completion_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float | None:
    """Return the measured completion bound for one finite cold B1 stream."""

    service = hardware.aic_mte2_finite_stream_completion_service
    if (service is None or not _uses_cann_persistent_execution(kernel)
            or kernel.dtype != "bf16"):
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    if (b * m < tile_m or tile_n != service["rows"]
            or tile_k != 64
            or geometry.b_descriptor_bytes != service["descriptor_bytes"]):
        return None
    descriptor_count = geometry.b_descriptors_per_worker
    if not (service["minimum_descriptors_per_worker"]
            <= descriptor_count
            <= service["maximum_descriptors_per_worker"]):
        return None
    if n not in service["worker_classes"]:
        return None
    worker_class = service["worker_classes"][n]
    worker_points = worker_class["active_aic_groups"]
    if not worker_points[0] <= geometry.worker_count <= worker_points[-1]:
        return None
    return hardware.aic_mte2_finite_stream_completion_time_ns(
        tile_n, geometry.b_descriptor_bytes // (
            tile_n * int(kernel.shape["dtype_bytes"])),
        n, geometry.worker_count, descriptor_count)


def _cann_npu_mte2_descriptor_domain(kernel, hardware):
    """Resolve the descriptor-stream domain the MTE2 services are defined on.

    Both base-tile and A1/B1 MTE2 completion services apply only when the
    hardware declares the ND2NZ service and the launch's worker count falls in
    its measured active-AIC-group range; outside that domain the caller
    contributes nothing.
    """

    geometry = _cann_npu_gemm_pipeline_geometry(kernel, hardware)
    mechanism = hardware.aic_mte2_nd2nz_service
    if geometry is None or mechanism is None:
        return None
    worker_count = geometry.worker_count
    if not (mechanism["active_aic_groups"][0] <= worker_count
            <= mechanism["active_aic_groups"][-1]):
        return None
    return geometry, mechanism, worker_count


def _cann_npu_gemm_base_tile_mte2_duration_ns(
        kernel: KernelSpec, hardware) -> float | None:
    """Return resident and shared-cold constraints for base-tile MTE2."""

    resolved = _cann_npu_mte2_descriptor_domain(kernel, hardware)
    if resolved is None:
        return None
    geometry, mechanism, worker_count = resolved
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    a_descriptor_bytes = tile_m * tile_k * element_bytes
    b_descriptor_bytes = tile_k * tile_n * element_bytes
    descriptor_range = mechanism["descriptor_bytes"]
    stream_range = mechanism["stream_descriptor_bytes"]
    if not all(descriptor_range[0] <= value <= descriptor_range[-1]
               for value in (a_descriptor_bytes, b_descriptor_bytes)):
        return None
    if not all(stream_range[0] <= value <= stream_range[-1]
               for value in (a_descriptor_bytes, b_descriptor_bytes)):
        return None
    pair_count = geometry.output_tiles_per_worker * geometry.k_tiles
    if pair_count <= 0:
        raise ValueError("910B GEMM has no base-tile operand pairs")
    resident_ns = (
        hardware.aic_mte2_nd2nz_serial_time_ns(
            a_descriptor_bytes, worker_count)
        + (pair_count - 1) * hardware.aic_mte2_nd2nz_stream_time_ns(
            a_descriptor_bytes, worker_count)
        + pair_count * hardware.aic_mte2_nd2nz_stream_time_ns(
            b_descriptor_bytes, worker_count))
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    aggregate_dram_bytes = float(
        (b * m * k + k * n) * element_bytes)
    aggregate_replay_bytes = float(
        worker_count * pair_count
        * (a_descriptor_bytes + b_descriptor_bytes))
    if aggregate_dram_bytes > aggregate_replay_bytes:
        raise ValueError(
            "910B GEMM compulsory operands exceed padded MTE2 replay")
    cold_stream_ns = hardware.aic_mte2_nd2nz_cold_stream_time_ns(
        aggregate_dram_bytes, worker_count)
    # Descriptor execution and shared DRAM delivery are simultaneous limits
    # on one stream. Serially adding them would double-count bytes already
    # carried by the resident descriptor schedule.
    # The shared cold stream already carries the descriptor bytes.  Add only
    # the independently measured serial ND2NZ penalty over the contiguous
    # descriptor service, so the same bytes are not charged twice.
    legacy_stride_penalty_ns = (
        _cann_npu_gemm_cold_legacy_stride_penalty_ns(
            kernel, hardware, geometry))
    v3_stride_penalty_ns = _cann_npu_gemm_cold_v3_stride_penalty_ns(
        kernel, hardware, geometry)
    small_m_stride_penalty_ns = (
        _cann_npu_gemm_cold_small_m_stride_penalty_ns(
            kernel, hardware, geometry))
    return (max(resident_ns, cold_stream_ns)
            + legacy_stride_penalty_ns + v3_stride_penalty_ns
            + small_m_stride_penalty_ns)


def _cann_npu_gemm_mte2_duration_ns(
        kernel: KernelSpec, hardware,
        program_traffic: GemmSourceOrderTraffic | None = None,
        dram_capacity_per_s: float | None = None,
        ) -> float | None:
    """Return one AIC's common ND2NZ A/B stream completion time.

    The retained tile fixes descriptor geometry, while the common two-slot
    A1/B1 GEMM program fixes the operand stream. Independent resident and
    DRAM-cold pair probes preserve the corresponding source-level completion
    service inside their measured domains. The source disables its unit-flag
    handoff when two accumulator tiles fit L0C; that legacy program keeps its
    exact cold service. None of these branches consumes production duration.
    """

    resolved = _cann_npu_mte2_descriptor_domain(kernel, hardware)
    if resolved is None:
        return None
    geometry, mechanism, worker_count = resolved
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    pair_count = geometry.output_tiles_per_worker * geometry.k_tiles
    source_program = _cann_gemm_launch_program(kernel)
    persistent_issue = hardware.aic_mte2_gemm_persistent_issue_service
    if (source_program is not None and persistent_issue is not None
            and source_program.depth_a1 == source_program.depth_b1
            and geometry.a_descriptors_per_worker
                == geometry.b_descriptors_per_worker):
        signature = {
            "dtype": kernel.dtype,
            "active_aic_groups": geometry.worker_count,
            "worker_grid_m": source_program.worker_grid_m,
            "worker_grid_n": source_program.worker_grid_n,
            "operand_buffer_depth": source_program.depth_a1,
            "base_tile_m": tile_m,
            "base_tile_n": tile_n,
            "base_tile_k": tile_k,
            "a_rows": tile_m * source_program.step_m,
            "a_columns": source_program.a_l1_k_tiles * tile_k,
            "a_source_row_stride_elements": int(kernel.shape["k"]),
            "b_rows": tile_n * source_program.step_n,
            "b_columns": source_program.b_l1_k_tiles * tile_k,
            "b_source_row_stride_elements": int(kernel.shape["k"]),
            "output_tiles_per_worker": geometry.output_tiles_per_worker,
            "k_slabs": geometry.k_slabs,
            "mte2_programs_per_worker": (
                geometry.a_descriptors_per_worker),
        }
        if signature == {
                name: persistent_issue[name] for name in signature}:
            return hardware.aic_mte2_gemm_persistent_issue_time_ns(
                **signature)
    pair_stream = hardware.aic_mte2_operand_pair_stream_service
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    operand_working_set_bytes = (
        b * m * k + k * n) * int(kernel.shape["dtype_bytes"])
    if (pair_stream is not None
            and (tile_m, tile_k, tile_k, tile_n) == tuple(
                pair_stream[name] for name in (
                    "a_rows", "a_columns", "b_rows", "b_columns"))
            and worker_count == pair_stream["active_aic_groups"]
            and pair_stream["minimum_pairs_per_worker"]
                <= pair_count
                <= pair_stream["maximum_pairs_per_worker"]
            and geometry.worker_n_tiles >= geometry.worker_m_tiles
            and operand_working_set_bytes <= hardware.l2_size_bytes):
        return hardware.aic_mte2_operand_pair_stream_time_ns(
            tile_m, tile_k, tile_k, tile_n, worker_count, pair_count)
    mixed_ring_ns = _cann_npu_gemm_mixed_ring_pair_time_ns(
        kernel, hardware, geometry)
    if mixed_ring_ns is not None:
        return mixed_ring_ns
    if _cann_npu_gemm_cold_pair_stream_supported(
            kernel, hardware, geometry):
        return hardware.aic_mte2_cold_operand_pair_stream_time_ns(
            tile_m, tile_k, tile_k, tile_n, worker_count, pair_count)
    if (geometry.a_descriptors_per_worker <= 0
            or geometry.b_descriptors_per_worker <= 0):
        raise ValueError("910B GEMM has no MTE2 descriptors")
    shared_source_queue_ns = _cann_npu_gemm_shared_source_queue_time_ns(
        kernel, hardware, geometry, program_traffic, dram_capacity_per_s)
    element_bytes = int(kernel.shape["dtype_bytes"])
    a_rows = tile_m * (source_program.step_m
                       if source_program is not None else 1)
    b_rows = tile_n * (source_program.step_n
                       if source_program is not None else 1)
    if (geometry.a_descriptor_bytes % (a_rows * element_bytes)
            or geometry.b_descriptor_bytes % (b_rows * element_bytes)):
        raise ValueError("910B GEMM descriptor geometry is not rectangular")
    a_columns = geometry.a_descriptor_bytes // (a_rows * element_bytes)
    b_columns = geometry.b_descriptor_bytes // (b_rows * element_bytes)
    if program_traffic is None:
        a_miss_fraction = None
        b_miss_fraction = None
    else:
        a_miss_fraction = validated_l2_miss_rate(
            program_traffic.a_dram_bytes, program_traffic.a_l2_bytes)
        b_miss_fraction = validated_l2_miss_rate(
            program_traffic.b_dram_bytes, program_traffic.b_l2_bytes)
    use_gemm_large_resident_service = (
        source_program is not None
        and hardware.aic_mte2_gemm_large_resident_program_supports(
            active_aic_groups=worker_count,
            a_rows=a_rows,
            a_columns=a_columns,
            a_source_row_stride_elements=k,
            a_buffer_depth=source_program.depth_a1,
            a_descriptors_per_worker=geometry.a_descriptors_per_worker,
            b_rows=b_rows,
            b_columns=b_columns,
            b_source_row_stride_elements=k,
            b_buffer_depth=source_program.depth_b1,
            b_descriptors_per_worker=geometry.b_descriptors_per_worker,
        ))
    a_stream_ns = _cann_npu_mte2_descriptor_channel_time_ns(
        hardware, a_rows, a_columns, worker_count,
        geometry.a_descriptors_per_worker, delayed_by_one_issue=False,
        source_row_stride_elements=k,
        use_gemm_large_resident_service=use_gemm_large_resident_service,
        dram_miss_fraction=a_miss_fraction)
    b_stream_ns = _cann_npu_mte2_descriptor_channel_time_ns(
        hardware, b_rows, b_columns, worker_count,
        geometry.b_descriptors_per_worker, delayed_by_one_issue=True,
        source_row_stride_elements=k,
        use_gemm_large_resident_service=use_gemm_large_resident_service,
        dram_miss_fraction=b_miss_fraction)
    if a_stream_ns is None or b_stream_ns is None:
        return None
    shared_source_order_ns = None
    if (source_program is not None
            and source_program.traversal_order == "persistent_n_then_m"
            and operand_working_set_bytes > hardware.l2_size_bytes):
        shared_source_order_ns = (
            hardware.aic_mte2_gemm_capacity_constrained_source_order_time_ns(
                a_rows, a_columns, geometry.a_descriptors_per_worker,
                b_rows, b_columns, geometry.b_descriptors_per_worker,
                worker_count))
    # A persistent N-major program whose complete operand set fits shared L2
    # retains the two source-visible A1/B1 producer streams above. Once that
    # same ordered stream exceeds L2 capacity, replacements keep A and B on
    # one aggregate MTE2 issue path. Independent long-program probes establish
    # that shared byte capacity across two descriptor geometries, while a
    # longer exact source program is a holdout. Treat it as a competing issue
    # bound rather than serially appending it to either channel completion.
    return max(
        a_stream_ns,
        b_stream_ns,
        shared_source_queue_ns or 0.0,
        shared_source_order_ns or 0.0,
    )


def _cann_npu_gemm_source_order_mte1_pairs(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> int | None:
    """Return the measured MatMulV2 source-order MTE1 pair count.

    The independent ingress sweep holds the 128x256x64 Cube tile, A1 K=512,
    B1 K=256, two A1/B1 slots and two L0A/L0B slots fixed while varying both
    source reuse topology and finite program length.  The original sweep
    covers the complete 24-AIC envelope; the partial-AIC sweep adds exact
    22/23-AIC cells.  Keep this as a physical program-domain check: no kernel
    name, production duration, or accuracy result participates.
    """

    service = hardware.aic_grouped_matmul_mte1_pair_completion_service
    if (service is None
            or not _uses_cann_persistent_execution(kernel)
            or kernel.dtype != service["dtype"]):
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    if ((tile_m, tile_n, tile_k) != (
            service["m_rows"][-1], service["n_columns"],
            service["k_columns"])
            or pipeline_depth != 2
            or geometry.worker_m_tiles != 1
            or geometry.a_l1_k_tiles != 8
            or geometry.b_l1_k_tiles != 4
            or geometry.a_descriptor_bytes != 131_072
            or geometry.b_descriptor_bytes != 131_072
            or geometry.k_tiles != geometry.k_slabs * 8
            or _cann_legacy_l0c_pipeline_depth(kernel, hardware) != 1):
        return None
    complete_program = (
        geometry.worker_count == 24
        and 1 <= geometry.output_tiles_per_worker <= 9
        and 6 <= geometry.k_slabs <= 16)
    partial_program = (
        (geometry.worker_count, geometry.output_tiles_per_worker,
         geometry.k_slabs) in {
             (22, 2, 8), (22, 2, 16),
             (23, 3, 8), (23, 3, 16),
             (23, 5, 8), (23, 5, 16),
         }
        and _cann_npu_gemm_finite_b1_completion_ns(
            kernel, hardware, geometry) is None
        and _cann_npu_gemm_cold_n_major_b_pair_time_ns(
            kernel, hardware, geometry) is None)
    if not complete_program and not partial_program:
        return None
    return geometry.output_tiles_per_worker * geometry.k_tiles


def _cann_npu_gemm_measured_mte1_pair_completion_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float | None:
    """Return a source-local MTE1 completion for an exact measured pair."""

    service = hardware.aic_grouped_matmul_mte1_pair_completion_service
    if (service is None
            or not _uses_cann_persistent_program(kernel)
            or kernel.dtype != service["dtype"]):
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    minimum_n_columns = (
        service["minimum_n_columns"] - service["n_column_granularity"])
    measured_domain = (
        tile_n == service["n_columns"]
        and geometry.b_l1_k_tiles * tile_k
            == service["b_l1_k_extent"])
    affine_extension_domain = (
        _cann_npu_gemm_serial_l1_source_program(kernel, geometry)
        and tile_n == minimum_n_columns
        and geometry.b_l1_k_tiles * tile_k
            == 2 * service["b_l1_k_extent"])
    if (tile_m not in service["m_rows"]
            or tile_k != service["k_columns"]
            or pipeline_depth != 2
            or not (measured_domain or affine_extension_domain)):
        return None
    pair_count = geometry.output_tiles_per_worker * geometry.k_tiles
    if tile_n < service["minimum_n_columns"]:
        return (
            hardware
            .aic_grouped_matmul_mte1_pair_affine_extension_time_ns(
                tile_m, tile_n, tile_k, pair_count))
    return hardware.aic_grouped_matmul_mte1_pair_time_ns(
        tile_m, tile_n, tile_k, pair_count)


def _cann_npu_gemm_serial_l1_source_program(
        kernel: KernelSpec,
        geometry: _CannNpuGemmPipelineGeometry) -> bool:
    """Whether MatMulV2 drains one non-full L1 slab before the next copy."""

    program = _cann_gemm_launch_program(kernel)
    if program is None or not _uses_cann_persistent_program(kernel):
        return False
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    return (
        kernel.dtype == "bf16"
        and (tile_m, tile_n, tile_k) == (48, 96, 64)
        and pipeline_depth == 2
        and program.depth_a1 == 2
        and program.depth_b1 == 2
        and program.a_l1_k_tiles == 8
        and program.b_l1_k_tiles == 8
        and program.single_core_k == 4096
        and geometry.worker_count == 22
        and geometry.output_tiles_per_worker == 2)


def _cann_npu_gemm_region_service(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> dict | None:
    """Return the exact source-scheduled MatMulV3 region service."""

    if (not _uses_cann_region_program(kernel)
            and kernel.launch.algorithm != _CANN_MATMUL_V3_ASW_65536_ALGORITHM
            or kernel.dtype != "bf16" or geometry.worker_count != 24):
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    if pipeline_depth != 2:
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    m_tiles = math.ceil(b * m / tile_m)
    n_tiles = math.ceil(n / tile_n)
    signature = {
        "dtype": kernel.dtype,
        "active_aic_groups": geometry.worker_count,
        "operand_buffer_depth": pipeline_depth,
        "base_tile_m": tile_m,
        "base_tile_n": tile_n,
        "base_tile_k": tile_k,
        "a_rows": tile_m,
        "a_columns": geometry.a_l1_k_tiles * tile_k,
        "a_source_row_stride_elements": k,
        "b_rows": geometry.b_l1_k_tiles * tile_k,
        "b_columns": tile_n,
        "b_source_row_stride_elements": k,
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "maximum_output_tiles_per_worker": (
            geometry.output_tiles_per_worker),
        "k_slabs": geometry.k_slabs,
        "mte2_programs_per_worker": (
            geometry.output_tiles_per_worker * geometry.k_slabs),
        "mte1_pairs_per_worker": (
            geometry.output_tiles_per_worker * geometry.k_tiles),
    }
    matches = []
    for service in hardware.aic_mte2_gemm_region_completion_services:
        # An exact service carries the source program's decoded L2 schedule.
        # Match it by the remaining physical program signature rather than
        # re-deriving the schedule: key-65536 uses the legacy
        # Mc2L2cacheTilePara payload, whose large-M partition is distinct from
        # the generic MatMulV3 host-tiler heuristic.
        if signature == {name: service[name] for name in signature}:
            matches.append(service)
    if len(matches) > 1:
        raise ValueError(
            "CANN MatMulV3 region service signature is ambiguous")
    return matches[0] if matches else None


def _cann_npu_gemm_region_pipeline_completion_ns(
        region_service: dict | None, hardware) -> float | None:
    """Return the dependent-Cube completion of one physical region program."""

    pipeline_service = (
        hardware.aic_mte2_gemm_region_pipeline_completion_service)
    if region_service is None or pipeline_service is None:
        return None
    signature_fields = (
        "dtype", "active_aic_groups", "operand_buffer_depth",
        "base_tile_m", "base_tile_n", "base_tile_k", "a_rows",
        "a_columns", "a_source_row_stride_elements", "b_rows", "b_columns",
        "b_source_row_stride_elements", "m_tiles", "n_tiles", "m_regions",
        "n_regions", "m_tiles_per_region", "n_tiles_per_region",
        "maximum_output_tiles_per_worker", "k_slabs",
        "mte2_programs_per_worker", "mte1_pairs_per_worker",
    )
    if any(region_service[name] != pipeline_service[name]
           for name in signature_fields):
        return None
    return pipeline_service["pipeline_completion_ns"]


def _cann_npu_gemm_exact_source_order_service(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> dict | None:
    """Match one complete physical ingress program to measured service."""

    if (not _uses_cann_persistent_execution(kernel)
            or kernel.dtype != "bf16"
            or geometry.worker_count != 24):
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    if pipeline_depth != 2:
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    m_tiles = math.ceil(b * m / tile_m)
    m_groups = m_tiles // geometry.worker_m_tiles
    if m_groups <= 0 or geometry.worker_count % m_groups:
        return None
    n_groups = geometry.worker_count // m_groups
    mte1_base_k = min(tile_k, 64)
    source_a_columns = geometry.a_l1_k_tiles * tile_k
    source_b_rows = tile_n * (tile_k // mte1_base_k)
    source_b_columns = geometry.b_l1_k_tiles * mte1_base_k
    source_output_tiles = math.ceil(
        math.ceil(n / source_b_rows) / n_groups)
    exact_signature = {
        "dtype": kernel.dtype,
        "active_aic_groups": geometry.worker_count,
        "operand_buffer_depth": pipeline_depth,
        "base_tile_m": tile_m,
        "base_tile_n": tile_n,
        "base_tile_k": tile_k,
        "mte1_base_k": mte1_base_k,
        "a_rows": tile_m,
        "a_columns": source_a_columns,
        "a_source_row_stride_elements": k,
        "b_rows": source_b_rows,
        "b_columns": source_b_columns,
        "b_source_row_stride_elements": k,
        "m_groups": m_groups,
        "n_groups": n_groups,
        "output_tiles_per_worker": source_output_tiles,
        "k_slabs": geometry.k_slabs,
        "mte2_programs_per_worker": (
            source_output_tiles * geometry.k_slabs),
        "mte1_pairs_per_worker": (
            source_output_tiles * math.ceil(k / mte1_base_k)),
    }
    return hardware.aic_mte2_gemm_source_order_service(**exact_signature)


def _cann_npu_gemm_exact_finite_transition_service(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> dict | None:
    """Match an independently measured normalized finite source program."""

    program = _cann_gemm_launch_program(kernel)
    if (program is None or not _uses_cann_persistent_program(kernel)
            or program.traversal_order != "persistent_n_then_m"):
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    mte1_base_k = min(tile_k, 64)
    signature = {
        "dtype": kernel.dtype,
        "operand_buffer_depth": pipeline_depth,
        "active_aic_groups": geometry.worker_count,
        "m_groups": program.worker_grid_m,
        "n_groups": program.worker_grid_n,
        "base_tile_m": tile_m,
        "base_tile_n": tile_n,
        "base_tile_k": tile_k,
        "a_rows": tile_m * program.step_m,
        "a_columns": program.a_l1_k_tiles * tile_k,
        "b_rows": tile_n * program.step_n,
        "b_columns": program.b_l1_k_tiles * tile_k,
        "source_row_stride_elements": program.single_core_k,
        "output_tiles_per_worker": geometry.output_tiles_per_worker,
        "k_slabs": geometry.k_slabs,
        "source_descriptors_per_worker": (
            geometry.a_descriptors_per_worker
            + geometry.b_descriptors_per_worker),
        "completion_pairs_per_worker": (
            geometry.output_tiles_per_worker
            * math.ceil(program.single_core_k / mte1_base_k)),
    }
    return hardware.aic_gemm_finite_transition_service(**signature)


def _cann_npu_gemm_finite_ingress_completion_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float | None:
    """Return the measured finite-program drain inside its evidence domain.

    Completion-pair count alone does not identify a source program.  The
    retained probes have the same Cube/MTE1 loop shape only for their measured
    A1/B1 descriptors, macro tile, and resident AIC population.  Applying
    their affine drain to a smaller descriptor program turns a physical
    per-program lower bound into a spurious serial floor.
    """

    program = _cann_gemm_launch_program(kernel)
    if program is None or not _uses_cann_persistent_program(kernel):
        return None
    _, _, tile_k, _ = _tile(kernel)
    mte1_base_k = min(tile_k, 64)
    pair_count = (
        geometry.output_tiles_per_worker
        * math.ceil(program.single_core_k / mte1_base_k))
    service = hardware.aic_gemm_finite_ingress_completion_service
    if service is None:
        return None
    evidence_programs = (
        service["calibration"]["source_program"],
        service["cross_geometry_validation"]["source_program"],
    )
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    matching_source_program = next((
        source for source in evidence_programs
        if (
            source["dtype"] == kernel.dtype
            and geometry.worker_count in source["active_aic_groups"]
            and source["operand_buffer_depth"] == pipeline_depth
            and (source["base_tile_m"], source["base_tile_n"],
                 source["base_tile_k"]) == (tile_m, tile_n, tile_k)
            and source["a_rows"] == tile_m * program.step_m
            and source["a_columns"] == program.a_l1_k_tiles * tile_k
            and source["b_rows"] == tile_n * program.step_n
            and source["b_columns"] == program.b_l1_k_tiles * tile_k
        )), None)
    if matching_source_program is None:
        return None
    minimum_pairs = matching_source_program[
        "minimum_completion_pairs_per_worker"]
    maximum_pairs = matching_source_program[
        "maximum_completion_pairs_per_worker"]
    if not minimum_pairs <= pair_count <= maximum_pairs:
        return None
    descriptor_count = (
        geometry.a_descriptors_per_worker
        + geometry.b_descriptors_per_worker)
    if descriptor_count > service[
            "maximum_validated_source_descriptors_per_worker"]:
        return None
    completion_ns = hardware.aic_gemm_finite_ingress_completion_time_ns(
        completion_pairs_per_worker=pair_count)
    if completion_ns is None:
        raise AssertionError("finite ingress service disappeared")
    return completion_ns + _cann_npu_gemm_cold_v3_stride_penalty_ns(
        kernel, hardware, geometry)


def _cann_npu_gemm_short_pipeline_completion_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry) -> float | None:
    """Return a source-shaped finite MTE2/MTE1/Cube completion bound."""

    service = hardware.aic_gemm_short_pipeline_completion_service
    program = _cann_gemm_launch_program(kernel)
    if (service is None or program is None
            or not _uses_cann_persistent_program(kernel)
            or program.traversal_order != "persistent_n_then_m"):
        return None
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    signature = {
        "dtype": kernel.dtype,
        "active_aic_groups": geometry.worker_count,
        "operand_buffer_depth": pipeline_depth,
        "base_tile_m": tile_m,
        "base_tile_n": tile_n,
        "base_tile_k": tile_k,
        "a_rows": tile_m * program.step_m,
        "a_columns": program.a_l1_k_tiles * tile_k,
        "b_rows": tile_n * program.step_n,
        "b_columns": program.b_l1_k_tiles * tile_k,
        "source_row_stride_elements": program.single_core_k,
        "m_groups": program.worker_grid_m,
        "n_groups": program.worker_grid_n,
        "k_slabs": geometry.k_slabs,
    }
    if signature != {name: service[name] for name in signature}:
        return None
    output_tiles = geometry.output_tiles_per_worker
    if (geometry.a_descriptors_per_worker
            != output_tiles * service["a_descriptors_per_output_tile"]
            or geometry.b_descriptors_per_worker
            != output_tiles * service["b_descriptors_per_output_tile"]
            or math.ceil(program.single_core_k / min(tile_k, 64))
                != service["completion_pairs_per_output_tile"]):
        return None
    return hardware.aic_gemm_short_pipeline_completion_time_ns(
        output_tiles_per_worker=output_tiles)


def _cann_npu_gemm_dirty_source_completion_ns(
        kernel: KernelSpec, hardware,
        geometry: _CannNpuGemmPipelineGeometry,
        traffic: GemmSourceOrderTraffic | None) -> float | None:
    """Return a measured-program dirty-source completion lower bound.

    The shared LRU determines how many anonymous dirty predecessor bytes the
    normalized source program displaces.  The independent dirty-preamble
    probe supplies the corresponding replacement service and a clean
    MTE2-to-MTE1 completion for one physical program.  Descriptor grouping is
    allowed to differ when the normalized program has the same terminal
    base-K completion-pair count: the probe reports that count directly while
    ``k_slabs`` counts source descriptors and therefore changes with the L1
    descriptor width.  Workload identity and observed GEMM duration do not
    participate.
    """

    service = hardware.l2_dirty_replacement_source_program
    capacity = hardware.l2_dirty_replacement_capacity_bytes_per_s
    clean_completion_ns = (
        hardware.l2_dirty_replacement_clean_completion_ns)
    program = _cann_gemm_launch_program(kernel)
    if (service is None or capacity is None or clean_completion_ns is None
            or traffic is None or program is None
            or not _uses_cann_persistent_program(kernel)):
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    mte1_base_k = min(tile_k, 64)
    completion_pairs_per_worker = (
        geometry.output_tiles_per_worker
        * math.ceil(program.single_core_k / mte1_base_k))
    measured_completion_pairs_per_worker = (
        service["output_tiles_per_worker"]
        * math.ceil(service["source_row_stride"] / mte1_base_k))
    source_signature = {
        "dtype": kernel.dtype,
        "active_aic_groups": geometry.worker_count,
        "base_tile_n": tile_n,
        "base_tile_k": tile_k,
        "output_tiles_per_worker": geometry.output_tiles_per_worker,
        "source_row_stride": program.single_core_k,
    }
    if source_signature != {
            name: service[name] for name in source_signature}:
        return None
    # The isolated source program issues A and B into independent A1/B1
    # banks.  Its B stream is the longer stream for every base-M extent no
    # larger than the measured base-N extent, so changing only base-M cannot
    # shorten or extend the measured clean terminal completion.  Preserve the
    # same completion service over that resource-defined domain instead of
    # dropping it at an arbitrary M boundary.
    if tile_m > tile_n:
        return None
    if completion_pairs_per_worker != measured_completion_pairs_per_worker:
        return None
    return (
        clean_completion_ns
        + traffic.predecessor_dirty_writeback_bytes / capacity * 1.0e9)


def _attach_cann_npu_mixed_gemm_cold_stream_constraint(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Keep mixed Stream-K operand delivery beside its compute envelope."""

    if (hardware.aic_mte2_nd2nz_service is None
            or kernel.variant != "gemm" or kernel.dtype != "bf16"
            or kernel.launch.parameters["aiv_mixed"] != 1.0
            or _cann_gemm_tile_pipeline_program(kernel) is not None):
        return inp
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
    compulsory_operand_bytes = float(
        b * m * k * a_element_bytes + k * n * b_element_bytes)
    cold_stream_ns = hardware.aic_mte2_nd2nz_cold_stream_time_ns(
        compulsory_operand_bytes, kernel.launch.grid_size)
    if "l2_cache" in inp.resource_time_overrides_ns:
        existing_ns = inp.resource_time_overrides_ns["l2_cache"]
    else:
        existing_ns = 0.0
    return dataclasses.replace(
        inp,
        resource_time_overrides_ns={
            **inp.resource_time_overrides_ns,
            "l2_cache": max(existing_ns, cold_stream_ns),
        },
    )


def _attach_cann_deterministic_split_k_memory_paths(
        inp: PredictionInput, kernel: KernelSpec,
        hardware,
        traffic: _CannDeterministicSplitKTraffic | None = None,
        ) -> PredictionInput:
    """Attach source-distinct AIC/AIV workspace and output paths."""

    program = _cann_gemm_tile_pipeline_program(kernel)
    if traffic is None:
        traffic = _cann_gemm_tile_pipeline_traffic(
            kernel, l2_replacement_capacity_bytes=None)
    if program is None or traffic is None:
        return inp
    aiv_data_workers = _cann_deterministic_split_k_aiv_data_workers(kernel)
    if aiv_data_workers is None:
        raise AssertionError("CANN split-K AIV admission was not derived")
    if inp.mem_model != "npu_tile" or inp.launch_task_kind != "mixed":
        raise ValueError(
            "CANN deterministic split-K paths require a mixed NPU launch")
    if any(path.name == "fixpipe_l2_write" for path in inp.memory_paths):
        raise ValueError("CANN split-K FixPipe path must attach exactly once")
    if any(path.name == "aiv_workspace_mte2_read"
           for path in inp.memory_paths):
        raise ValueError(
            "CANN split-K AIV workspace path must attach exactly once")
    if program.program_kind == "cann_gemm_single_core_split_k_v1":
        operand_paths = tuple(
            path for path in inp.memory_paths if path.name == "mte2_read")
        if len(operand_paths) != 1:
            raise ValueError(
                "CANN single-core split-K requires one operand read path")
        operand_path = operand_paths[0]
        # The source issues every GM-to-L1 request through MTE2 before the
        # request reaches L2 or DRAM.  The INI exposes a conserved full-chip
        # 768-B/clk MTE2 ingress envelope, independent of cache residency.
        bounded_operand_path = dataclasses.replace(
            operand_path,
            nodes=(MemoryNodeDemand(
                resource_role="mte2",
                transactions=None,
                bytes=traffic.operand_l2_read_bytes,
                transaction_bytes=None,
                independent_service_streams=2,
                capacity_ceiling_resource_role="l2_cache",
            ),) + operand_path.nodes,
        )
        inp = dataclasses.replace(
            inp,
            memory_paths=tuple(
                bounded_operand_path if path is operand_path else path
                for path in inp.memory_paths),
        )
    workspace_read_path = MemoryPathDemand(
        name="aiv_workspace_mte2_read",
        access_kind="read",
        initial_transactions=None,
        nodes=(
            MemoryNodeDemand(
                resource_role="mte2",
                transactions=None,
                bytes=traffic.workspace_l2_read_bytes,
                transaction_bytes=None,
            ),
            MemoryNodeDemand(
                resource_role="l2_cache",
                transactions=None,
                bytes=traffic.workspace_l2_read_bytes,
                transaction_bytes=None,
            ),
        ),
        source=MemorySourceDemand(
            active_units=aiv_data_workers,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    fixpipe_nodes = [MemoryNodeDemand(
        resource_role="l2_cache",
        transactions=None,
        bytes=traffic.workspace_l2_write_bytes,
        transaction_bytes=None,
    )]
    if traffic.workspace_dram_write_bytes > 0.0:
        fixpipe_nodes.append(MemoryNodeDemand(
            resource_role="dram",
            transactions=None,
            bytes=traffic.workspace_dram_write_bytes,
            transaction_bytes=None,
        ))
    fixpipe_path = MemoryPathDemand(
        name="fixpipe_l2_write",
        access_kind="write",
        initial_transactions=None,
        nodes=tuple(fixpipe_nodes),
        source=MemorySourceDemand(
            active_units=program.aic_workers,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    demand = dict(inp.demand_per_resource)
    extra_paths = ()
    mte1_bytes = _cann_deterministic_split_k_mte1_operand_bytes(
        kernel, program)
    if mte1_bytes is not None:
        a_bytes, b_bytes = mte1_bytes
        operand_bytes = a_bytes + b_bytes
        if not math.isclose(
                operand_bytes, traffic.operand_l2_read_bytes,
                rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(
                "deterministic Split-K MTE1 traffic must conserve its MTE2 "
                "operands")
        mte1_path_name = (
            "nbuffer33_mte1_pair"
            if program.scheduler_policy == "n_buffer_33"
            else "split_k_mte1_pair")
        extra_paths = (MemoryPathDemand(
            name=mte1_path_name,
            access_kind="read",
            initial_transactions=None,
            nodes=(
                MemoryNodeDemand(
                    resource_role="l0a_buffer",
                    transactions=None,
                    bytes=a_bytes,
                    transaction_bytes=None,
                ),
                MemoryNodeDemand(
                    resource_role="l0b_buffer",
                    transactions=None,
                    bytes=b_bytes,
                    transaction_bytes=None,
                ),
            ),
            source=MemorySourceDemand(
                active_units=program.aic_workers,
                producers_per_unit=1.0,
                independent_requests_per_producer=None,
                frontend_rate_per_unit_s=None,
            ),
        ),)
    demand["l2_cache"] += (
        traffic.workspace_l2_read_bytes
        + traffic.workspace_l2_write_bytes)
    demand["dram"] += traffic.workspace_dram_write_bytes
    with_fixpipe = dataclasses.replace(
        inp,
        demand_per_resource=demand,
        memory_paths=inp.memory_paths + extra_paths + (
            workspace_read_path, fixpipe_path),
    )
    return append_npu_write_path(
        with_fixpipe,
        hardware,
        l2_write_bytes=traffic.output_l2_write_bytes,
        dram_write_bytes=traffic.output_dram_write_bytes,
        write_active_units=aiv_data_workers,
    )


def _attach_cann_deterministic_split_k_vector_service(
        inp: PredictionInput, kernel: KernelSpec) -> PredictionInput:
    """Bind source reduction additions and cast to the physical AIV domain."""

    program = _cann_gemm_tile_pipeline_program(kernel)
    if program is None:
        return inp
    aiv_data_workers = _cann_deterministic_split_k_aiv_data_workers(kernel)
    if aiv_data_workers is None:
        raise AssertionError("CANN split-K AIV admission was not derived")
    matches = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    if len(matches) != 1:
        raise ValueError(
            "CANN deterministic split-K requires one Vector compute demand")
    vector = matches[0]
    typed = dataclasses.replace(
        vector,
        schedule_efficiency=(
            vector.schedule_efficiency
            * aiv_data_workers / program.aiv_workers),
        service_demands=(ComputeServiceDemand(
            service="generic",
            operations=vector.operations,
        ),),
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            typed if demand is vector else demand
            for demand in inp.compute_demands),
    )


def _attach_cann_deterministic_split_k_pipeline(
        inp: PredictionInput, kernel: KernelSpec,
        hardware) -> PredictionInput:
    """Attach CANN's compressed AIC-to-workspace-to-AIV max-plus program."""

    program = _cann_deterministic_split_k_program(kernel)
    traffic = _cann_deterministic_split_k_traffic(kernel)
    if program is None or traffic is None:
        return inp
    if inp.mem_model != "npu_tile" or inp.launch_task_kind != "mixed":
        raise ValueError(
            "CANN deterministic split-K pipeline requires a mixed NPU launch")
    if inp.pipeline_plan is not None:
        raise ValueError("CANN GEMM pipeline must attach exactly once")

    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    output_groups = (
        math.ceil(b * m / program.single_core_m)
        * math.ceil(n / program.single_core_n))
    total_k_groups = math.ceil(k / program.single_core_k)
    full_k_groups, tail_k = divmod(k, program.single_core_k)
    base_full_groups, extra_full_workers = divmod(
        full_k_groups, program.aic_workers)
    if base_full_groups <= 0:
        raise ValueError("CANN split-K requires positive K work per AIC")
    aic_task_count = output_groups * total_k_groups
    input_read_fraction = 1.0 / aic_task_count
    tensor_fraction = 1.0 / aic_task_count
    workspace_write_fraction = 1.0 / (
        total_k_groups * output_groups)
    output_group_worker_count = program.aic_workers * output_groups
    workspace_read_fraction = 1.0 / output_group_worker_count
    vector_fraction = 1.0 / output_group_worker_count
    output_write_fraction = 1.0 / output_group_worker_count
    mte1_operand_bytes = _cann_deterministic_split_k_mte1_operand_bytes(
        kernel, program)
    nbuffer33_pair_bytes = _cann_nbuffer33_mte1_operand_bytes(kernel, program)
    mte2_group_service_available = (
        program.scheduler_policy == "n_buffer_33"
        and
        program.aic_workers == 24
        and program.single_core_m == 384
        and program.single_core_n == 4096
        and program.single_core_k == 384
        and program.base_m == 128
        and program.base_n == 128
        and program.base_k == 128
        and program.depth_a1 == 9
        and program.depth_b1 == 6
        and program.step_m == 3
        and program.step_n == 1
        and b * m <= 128
        and n % program.single_core_n == 0
        and k == 28672)
    source_queue_mechanism = (
        hardware.aic_deterministic_split_k_source_queue_service)
    aligned_rows = math.ceil(b * m / 16) * 16
    n_slabs = math.ceil(n / program.single_core_n)
    source_queue_supported = (
        source_queue_mechanism is not None
        and program.scheduler_policy == "l2_split_matmul"
        and kernel.dtype == source_queue_mechanism["dtype"]
        and program.aic_workers
        == source_queue_mechanism["active_aic_groups"]
        and n_slabs == source_queue_mechanism["n_slabs"]
        and program.single_core_n // program.base_n
        == source_queue_mechanism["n_tiles_per_slab"]
        and total_k_groups == source_queue_mechanism["k_chunks"]
        and extra_full_workers + int(bool(tail_k))
        == source_queue_mechanism["full_chunk_workers"]
        and base_full_groups + 1
        == source_queue_mechanism["full_chunks_per_worker"]
        and base_full_groups
        == source_queue_mechanism["regular_chunks_per_worker"]
        and program.single_core_k == source_queue_mechanism["k_chunk"]
        and tail_k == source_queue_mechanism["k_tail"]
        and k == source_queue_mechanism["physical_k"]
        and n == source_queue_mechanism["physical_n"]
        and program.single_core_k
        == source_queue_mechanism["a_descriptor_columns"]
        and program.base_n == source_queue_mechanism["b_descriptor_rows"]
        and program.single_core_k
        == source_queue_mechanism["b_descriptor_columns"]
        and k == source_queue_mechanism["b_source_row_stride"]
        and program.a1_buffer_slots
        == source_queue_mechanism["a1_buffer_slots"]
        and program.b1_buffer_slots
        == source_queue_mechanism["b1_buffer_slots"]
        and program.l0a_buffer_slots
        == source_queue_mechanism["l0a_buffer_slots"]
        and program.l0b_buffer_slots
        == source_queue_mechanism["l0b_buffer_slots"]
        and program.traversal_order
        == source_queue_mechanism["traversal_order"]
        and aligned_rows in source_queue_mechanism[
            "completion_ns_by_aligned_a_rows"])
    collective_mechanism = (
        hardware.aic_deterministic_split_k_collective_barrier_service)
    collective_supported = (
        source_queue_supported
        and collective_mechanism is not None
        and program.aic_workers
        == collective_mechanism["active_aic_groups"]
        and program.aiv_workers
        == collective_mechanism["active_aiv_workers"]
        and n_slabs == collective_mechanism["n_slabs"]
        and kernel.dtype == collective_mechanism["dtype"]
        and program.scheduler_policy
        == collective_mechanism["scheduler_policy"])
    source_queue_completion_s = (
        (hardware.aic_deterministic_split_k_source_queue_time_ns(
            aligned_rows, program.aic_workers)
         + (hardware.aic_deterministic_split_k_collective_barrier_time_ns(
             program.aic_workers, program.aiv_workers, n_slabs)
            if collective_supported else 0.0)) * 1e-9
        if source_queue_supported else None)

    def aic_task(
            name: str, *, first: bool, terminal: bool,
            k_columns: int,
            repeat_count: int = 1) -> PipelineTaskDemand:
        operand_stage_name = "operand_read"
        stages = [
            PipelineStageDemand(
                operand_stage_name, "memory:mte2_read", input_read_fraction,
                False,
                # Every output slab after the first enters only after the
                # deterministic source has observed both paired-AIV flags,
                # executed SyncAll and drained all AIC/AIV pipes.  The same
                # repeated task represents slab zero as well; all prior
                # resources are initially ready, so this recurrence adds no
                # synthetic launch prologue.
                prior_resources=(
                    (
                        "memory:mte2_read",
                        "memory:split_k_mte1_pair",
                        "compute:compute_tensor",
                        "memory:fixpipe_l2_write",
                        "memory:aiv_workspace_mte2_read",
                        "compute:compute_vector",
                        "memory:mte3_write",
                    )
                    if first and program.scheduler_policy == "l2_split_matmul"
                    else ()),
                memory_exact_request_demands=(
                    (MemoryExactRequestDemand(
                        resource_role="dram",
                        service=_CANN_SPLIT_K_MTE2_GROUP_SERVICE,
                        request_sizes_by_worker=((k_columns,),),
                    ),)
                    if mte2_group_service_available else ())),
        ]
        if mte1_operand_bytes is not None:
            a_total_bytes, b_total_bytes = mte1_operand_bytes
            if k_columns <= 0 or k_columns > k:
                raise ValueError("deterministic Split-K task has invalid K")
            pair_count = (
                math.ceil(b * m / program.base_m)
                * (program.single_core_n // program.base_n)
                * math.ceil(k_columns / program.base_k))
            if pair_count <= 0:
                raise ValueError("deterministic Split-K task has no MTE1 pair")
            exact_pair_domain = (
                nbuffer33_pair_bytes is not None
                and
                b * m % program.base_m == 0
                and n % program.single_core_n == 0
                and k_columns % program.base_k == 0)
            mte1_path_name = (
                "nbuffer33_mte1_pair"
                if nbuffer33_pair_bytes is not None
                else "split_k_mte1_pair")
            stages.append(PipelineStageDemand(
                "mte1_pair", f"memory:{mte1_path_name}", None,
                memory_node_demands=(
                    PhaseMemoryDemand(
                        mte1_path_name, "l0a_buffer",
                        a_total_bytes * k_columns / (k * output_groups)),
                    PhaseMemoryDemand(
                        mte1_path_name, "l0b_buffer",
                        b_total_bytes * k_columns / (k * output_groups)),
                ),
                memory_exact_request_demands=(
                    (MemoryExactRequestDemand(
                        resource_role="l0a_buffer",
                        service=_CANN_NBUFFER33_MTE1_PAIR_SERVICE,
                        request_sizes_by_worker=((1,) * pair_count,),
                    ),)
                    if exact_pair_domain else ()),
                memory_exact_request_replaces_resource_capacity=(
                    exact_pair_domain),
                predecessors=(operand_stage_name,),
            ))
            cube_predecessors = ("mte1_pair",)
        else:
            cube_predecessors = (operand_stage_name,)
        stages.append(PipelineStageDemand(
            "cube", "compute:compute_tensor", tensor_fraction,
            False, predecessors=cube_predecessors))
        stages.append(PipelineStageDemand(
            "workspace_first_write" if first else "workspace_write",
            "memory:fixpipe_l2_write",
            workspace_write_fraction, False,
            predecessors=("cube",)))
        return PipelineTaskDemand(
            name=name,
            worker=0,
            stages=tuple(stages),
            repeat_count=repeat_count,
        )

    reduction = PipelineTaskSequenceDemand(
        name="aiv_workspace_reduce",
        worker=0,
        tasks=(PipelineTaskDemand(
            name="workspace_reduce_and_cast",
            worker=0,
            stages=(
                PipelineStageDemand(
                    "workspace_read", "memory:aiv_workspace_mte2_read",
                    workspace_read_fraction, False,
                    prior_resources=("memory:fixpipe_l2_write",)),
                PipelineStageDemand(
                    "reduce_cast", "compute:compute_vector",
                    vector_fraction, False,
                    predecessors=("workspace_read",)),
                PipelineStageDemand(
                    "output_write", "memory:mte3_write",
                    output_write_fraction, False,
                    predecessors=("reduce_cast",)),
            ),
        ),),
    )

    worker_classes = []
    class_specs = []
    if extra_full_workers:
        class_specs.append((
            base_full_groups + 1, None, extra_full_workers))
    if tail_k:
        class_specs.append((base_full_groups, tail_k, 1))
    regular_workers = (
        program.aic_workers - extra_full_workers - int(bool(tail_k)))
    if regular_workers:
        class_specs.append((base_full_groups, None, regular_workers))
    if sum(spec[2] for spec in class_specs) != program.aic_workers:
        raise AssertionError("CANN split-K worker classes lost an AIC")
    for worker, (full_groups, tail_columns, multiplicity) in enumerate(
            class_specs):
        tasks = []
        total_groups = full_groups + int(tail_columns is not None)
        if full_groups:
            tasks.append(aic_task(
                "first_k_group", first=True,
                terminal=(total_groups == 1),
                k_columns=program.single_core_k))
            middle_groups = (
                full_groups - 1
                if tail_columns is not None else full_groups - 2)
            if middle_groups > 0:
                tasks.append(aic_task(
                    "middle_k_groups", first=False, terminal=False,
                    k_columns=program.single_core_k,
                    repeat_count=middle_groups))
            if full_groups > 1 and tail_columns is None:
                tasks.append(aic_task(
                    "final_k_group", first=False, terminal=True,
                    k_columns=program.single_core_k))
        if tail_columns is not None:
            tasks.append(aic_task(
                "tail_k_group", first=(full_groups == 0), terminal=True,
                k_columns=tail_columns))
        aic_tasks = tuple(tasks)
        if sum(task.repeat_count for task in aic_tasks) != total_groups:
            raise AssertionError("CANN split-K worker class lost a K group")
        output_group_sequence = PipelineTaskSequenceDemand(
            name=f"output_group_class{worker}",
            worker=worker,
            tasks=tuple(
                dataclasses.replace(task, worker=worker)
                for task in aic_tasks)
            + tuple(
                dataclasses.replace(task, worker=worker)
                for task in reduction.tasks),
        )
        is_longest_source_class = (
            source_queue_supported
            and tail_columns is None
            and full_groups
            == source_queue_mechanism["full_chunks_per_worker"])
        if is_longest_source_class:
            final_tasks = []
            terminal_aic_task = output_group_sequence.tasks[-2]
            for task in output_group_sequence.tasks:
                stages = tuple(
                    dataclasses.replace(
                        stage,
                        prior_resources=(
                            stage.prior_resources
                            + (_CANN_SPLIT_K_SOURCE_QUEUE_FLOOR,)))
                    if task is terminal_aic_task and stage.name == "cube"
                    else stage
                    for stage in task.stages)
                final_tasks.append(dataclasses.replace(task, stages=stages))
            final_sequence = dataclasses.replace(
                output_group_sequence,
                name=f"terminal_output_group_class{worker}",
                tasks=tuple(final_tasks))
            if output_groups == 1:
                patterns = (final_sequence,)
                order = (0,)
                repeat_count = 1
                tail_order = ()
            else:
                patterns = (output_group_sequence, final_sequence)
                order = (0,)
                repeat_count = output_groups - 1
                tail_order = (1,)
        else:
            patterns = (output_group_sequence,)
            order = (0,)
            repeat_count = output_groups
            tail_order = ()
        worker_classes.append(PipelineTaskProgramDemand(
            name=f"split_k_worker_class{worker}",
            worker=worker,
            patterns=patterns,
            order=order,
            repeat_count=repeat_count,
            tail_order=tail_order,
            worker_multiplicity=multiplicity,
        ))
    plan = PipelineWorkPlan(
        worker_count=program.aic_workers,
        tasks=tuple(worker_classes),
        buffers=(
            (PipelineBuffer(
                name="split_k_a1_credit",
                producer_stage="operand_read",
                consumer_stage="mte1_pair",
                slots=program.a1_buffer_slots,
                scope="worker",
            ),
             PipelineBuffer(
                name="split_k_b1_credit",
                producer_stage="operand_read",
                consumer_stage="mte1_pair",
                slots=program.b1_buffer_slots,
                scope="worker",
            ),
             PipelineBuffer(
                name="split_k_l0a_ping_pong",
                producer_stage="mte1_pair",
                consumer_stage="cube",
                slots=program.l0a_buffer_slots,
                scope="worker",
            ),
             PipelineBuffer(
                name="split_k_l0b_ping_pong",
                producer_stage="mte1_pair",
                consumer_stage="cube",
                slots=program.l0b_buffer_slots,
                scope="worker",
            ))
            if mte1_operand_bytes is not None else ())
        + (PipelineBuffer(
            name="split_k_workspace_ping_pong",
            producer_stage="workspace_first_write",
            consumer_stage="output_write",
            slots=program.workspace_buffer_count,
            scope="worker",
        ),),
        initial_resource_availability_s=(
            ((_CANN_SPLIT_K_SOURCE_QUEUE_FLOOR,
              source_queue_completion_s),)
            if source_queue_supported else ()),
    )
    attached = attach_pipeline_work_plan(
        inp, hardware, plan, promote_npu_partial=True)
    limitations = tuple(
        limitation for limitation in attached.limitations
        if limitation not in {
            "aiv_mixed_operation_decomposition_unavailable",
            "aic_operand_replay_evidence_unavailable",
        }
        and not (
            (mte2_group_service_available or source_queue_supported)
            and limitation ==
            "mte2_descriptor_and_credit_evidence_unavailable"))
    if not mte2_group_service_available and not source_queue_supported:
        limitations += (
            "cann_split_k_mte2_group_service_domain_unavailable",)
    if not collective_supported:
        limitations += ("aic_aiv_collective_barrier_lifecycle_unavailable",)
    return dataclasses.replace(attached, limitations=limitations)


def _attach_cann_single_core_split_k_pipeline(
        inp: PredictionInput, kernel: KernelSpec,
        hardware,
        traffic: _CannDeterministicSplitKTraffic | None = None,
        ) -> PredictionInput:
    """Attach key-65632's compact AIC/workspace/AIV source program."""

    program = _cann_single_core_split_k_program(kernel)
    if traffic is None:
        traffic = _cann_single_core_split_k_traffic(
            kernel, l2_replacement_capacity_bytes=None)
    if program is None or traffic is None:
        return inp
    if inp.mem_model != "npu_tile" or inp.launch_task_kind != "mixed":
        raise ValueError(
            "CANN single-core split-K pipeline requires a mixed NPU launch")
    if inp.pipeline_plan is not None:
        raise ValueError("CANN GEMM pipeline must attach exactly once")

    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])
    full_k_groups, tail_k = divmod(k, program.single_core_k)
    if full_k_groups <= 0:
        raise ValueError(
            "CANN single-core split-K requires a complete K group")
    total_tensor_operations = float(2 * rows * n * k)
    total_vector_operations = float(rows * n)

    block_classes = {}
    for m_start in range(0, rows, program.single_core_m):
        m_extent = min(program.single_core_m, rows - m_start)
        for n_start in range(0, n, program.single_core_n):
            n_extent = min(program.single_core_n, n - n_start)
            key = (m_extent, n_extent)
            if key in block_classes:
                block_classes[key] += 1
            else:
                block_classes[key] = 1
    if sum(block_classes.values()) != program.aic_workers:
        raise ValueError(
            "CANN single-core split-K classes must cover every AIC")

    resident_service = (
        hardware.aic_matmul_v3_split_k_resident_tile_service)
    k_group_widths = {program.single_core_k}
    if tail_k:
        k_group_widths.add(tail_k)
    resident_envelope_supported = (
        resident_service is not None
        and kernel.dtype == resident_service["dtype"]
        # The isolated resident service contains only AIC-local MTE1, Cube,
        # FixPipe and L0 lifetime.  Shared GM/L2 service is excluded, so the
        # per-worker completion remains valid for any non-empty subset of the
        # captured 24-AIC envelope.
        and program.aic_workers <= resident_service["active_aic_workers"]
        and program.base_m == resident_service["tile_m"]
        and program.base_n == resident_service["tile_n"]
        and all(
            width % math.gcd(*resident_service["duration_ns_by_tile_k"]) == 0
            and min(resident_service["duration_ns_by_tile_k"]) <= width <= 512
            for width in k_group_widths)
        and all(
            m_extent % program.base_m == 0
            and n_extent % program.base_n == 0
            for m_extent, n_extent in block_classes))
    if resident_envelope_supported:
        limitations = tuple(
            limitation for limitation in inp.limitations
            if limitation not in {
                "aiv_mixed_operation_decomposition_unavailable",
                "aic_operand_replay_evidence_unavailable",
                "aic_store_fixpipe_tile_lifecycle_unavailable",
                "mte2_descriptor_and_credit_evidence_unavailable",
            })
        return dataclasses.replace(
            inp,
            pipeline_plan=CannSingleCoreSplitKEnvelopeDemand(
                worker_count=program.aic_workers,
                class_shapes=tuple(block_classes),
                class_multiplicities=tuple(block_classes.values()),
                projection_k=k,
                single_core_k=program.single_core_k,
                base_m=program.base_m,
                base_n=program.base_n,
                base_k=program.base_k,
                step_m=program.step_m,
                step_n=program.step_n,
                # The source may allocate a spare A1 slot on its 2x4 path,
                # but only ``step_m`` A tiles participate in one stripe.
                a1_buffer_slots=min(
                    program.step_m, program.a1_buffer_slots),
                b1_buffer_slots=program.b1_buffer_slots,
                partial_element_bytes=program.partial_element_bytes,
                dtype=kernel.dtype,
                element_bytes=element_bytes,
            ),
            support_level="partial",
            limitations=limitations + (
                "aic_aiv_collective_barrier_lifecycle_unavailable",),
        )

    worker_classes = []
    for worker, ((m_extent, n_extent), multiplicity) in enumerate(
            block_classes.items()):
        inner_block_m = program.step_m * program.base_m

        def stripe_pattern(
                name: str, stripe_m: int) -> PipelineTaskSequenceDemand:
            def k_task(
                    task_name: str, k_columns: int, *,
                    stripe_start: bool = False,
                    repeat_count: int = 1) -> PipelineTaskDemand:
                operand_fraction = (
                    (stripe_m * k_columns + k_columns * n_extent)
                    * element_bytes / traffic.operand_l2_read_bytes)
                tensor_fraction = (
                    2.0 * stripe_m * n_extent * k_columns
                    / total_tensor_operations)
                workspace_write_fraction = (
                    stripe_m * n_extent * program.partial_element_bytes
                    / traffic.workspace_l2_write_bytes)
                return PipelineTaskDemand(
                    name=task_name,
                    worker=worker,
                    stages=(
                        PipelineStageDemand(
                            "operand_read", "memory:mte2_read",
                            operand_fraction, False,
                            prior_resources=(
                                (
                                    "compute:compute_tensor",
                                    "memory:fixpipe_l2_write",
                                )
                                if stripe_start else ())),
                        PipelineStageDemand(
                            "cube", "compute:compute_tensor",
                            tensor_fraction, False,
                            predecessors=("operand_read",)),
                        PipelineStageDemand(
                            "workspace_write", "memory:fixpipe_l2_write",
                            workspace_write_fraction, False,
                            predecessors=("cube",)),
                    ),
                    repeat_count=repeat_count,
                )

            # The key-65632 source ends every inner-M stripe with
            # ``PipeBarrier<PIPE_ALL>`` on both actor types.  Keep the first
            # K group separate so only the stripe boundary, rather than every
            # K-group recurrence, waits for the prior AIC pipeline to drain.
            tasks = [k_task(
                "first_full_k_group", program.single_core_k,
                stripe_start=True)]
            if full_k_groups > 1:
                tasks.append(k_task(
                    "remaining_full_k_groups", program.single_core_k,
                    repeat_count=full_k_groups - 1))
            if tail_k:
                tasks.append(k_task("tail_k_group", tail_k))
            area = stripe_m * n_extent
            tasks.append(PipelineTaskDemand(
                name="workspace_cast_store",
                worker=worker,
                stages=(
                    PipelineStageDemand(
                        "workspace_read", "memory:aiv_workspace_mte2_read",
                        (area * program.partial_element_bytes
                         / traffic.workspace_l2_read_bytes),
                        False,
                        prior_resources=(
                            "memory:fixpipe_l2_write",
                            "compute:compute_vector",
                            "memory:mte3_write",
                        )),
                    PipelineStageDemand(
                        "cast", "compute:compute_vector",
                        area / total_vector_operations,
                        False, predecessors=("workspace_read",)),
                    PipelineStageDemand(
                        "output_write", "memory:mte3_write",
                        area * element_bytes / traffic.output_l2_write_bytes,
                        False, predecessors=("cast",)),
                ),
            ))
            return PipelineTaskSequenceDemand(
                name=name,
                worker=worker,
                tasks=tuple(tasks),
            )

        full_stripes, tail_stripe_m = divmod(m_extent, inner_block_m)
        patterns = []
        order = ()
        repeat_count = 1
        tail_order = ()
        if full_stripes:
            patterns.append(stripe_pattern("full_m_stripe", inner_block_m))
            order = (0,)
            repeat_count = full_stripes
        if tail_stripe_m:
            patterns.append(stripe_pattern("tail_m_stripe", tail_stripe_m))
            if order:
                tail_order = (len(patterns) - 1,)
            else:
                order = (0,)
        worker_classes.append(PipelineTaskProgramDemand(
            name=f"single_core_split_k_class{worker}",
            worker=worker,
            patterns=tuple(patterns),
            order=order,
            repeat_count=repeat_count,
            tail_order=tail_order,
            worker_multiplicity=multiplicity,
        ))

    attached = attach_pipeline_work_plan(
        inp,
        hardware,
        PipelineWorkPlan(
            worker_count=program.aic_workers,
            tasks=tuple(worker_classes),
            buffers=(),
        ),
        promote_npu_partial=True,
    )
    limitations = tuple(
        limitation for limitation in attached.limitations
        if limitation not in {
            "aiv_mixed_operation_decomposition_unavailable",
            "aic_operand_replay_evidence_unavailable",
            "aic_store_fixpipe_tile_lifecycle_unavailable",
        })
    return dataclasses.replace(
        attached,
        limitations=limitations + (
            "aic_aiv_collective_barrier_lifecycle_unavailable",),
    )


def _attach_cann_npu_gemm_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        program_traffic: GemmSourceOrderTraffic | None = None,
        split_k_traffic: _CannDeterministicSplitKTraffic | None = None,
) -> PredictionInput:
    """Attach the common compressed AIC MTE2-to-Cube GEMM K loop."""

    if _cann_deterministic_split_k_program(kernel) is not None:
        return _attach_cann_deterministic_split_k_pipeline(
            inp, kernel, hardware)
    if _cann_single_core_split_k_program(kernel) is not None:
        return _attach_cann_single_core_split_k_pipeline(
            inp, kernel, hardware, split_k_traffic)

    geometry = _cann_npu_gemm_pipeline_geometry(kernel, hardware)
    if geometry is None:
        return inp
    if inp.mem_model != "npu_tile" or inp.launch_task_kind != "aic":
        raise ValueError("910B GEMM pipeline requires an AIC NPU launch")
    if inp.pipeline_plan is not None:
        raise ValueError("910B GEMM pipeline must attach exactly once")
    worker_count = geometry.worker_count
    _, _, _, pipeline_depth = _tile(kernel)
    if pipeline_depth != 2:
        raise AssertionError("910B GEMM operand pipeline depth changed")
    if geometry.k_slabs <= 0:
        raise ValueError("910B GEMM pipeline requires positive K-slab work")
    fixpipe_service = _cann_npu_gemm_fixpipe_finite_service(
        kernel, hardware, geometry)
    region_service = _cann_npu_gemm_region_service(
        kernel, hardware, geometry)
    region_pipeline_completion_ns = (
        _cann_npu_gemm_region_pipeline_completion_ns(
            region_service, hardware))
    tail_region_service = None
    if region_service is not None and "mte2_only_ns" in region_service:
        tail_region_service = region_service
    analytical_mte2_ns = None
    if "l2_cache" in inp.resource_time_overrides_ns:
        analytical_mte2_ns = inp.resource_time_overrides_ns["l2_cache"]
    # L0A/L0B and Cube use the same compact max-plus program for every CANN
    # GEMM.  When a base MTE1 transfer is slower than its Cube consumer, keep
    # the base K tile as the credit unit; otherwise coalesce the proven L1
    # descriptor depth into a slab.  This is a resource inequality, not a
    # kernel-name or workload-duration dispatch.
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    program = _cann_gemm_launch_program(kernel)
    finite_ingress_completion_ns = (
        _cann_npu_gemm_finite_ingress_completion_ns(
            kernel, hardware, geometry))
    short_pipeline_completion_ns = (
        _cann_npu_gemm_short_pipeline_completion_ns(
            kernel, hardware, geometry))
    cold_source_completion_ns = None
    dirty_replacement_completion_ns = None
    if (program_traffic is not None
            and program_traffic.predecessor_dirty_writeback_bytes > 0):
        dirty_capacity = hardware.l2_dirty_replacement_capacity_bytes_per_s
        if dirty_capacity is not None:
            dram_capacity = (
                hardware.resource_by_role("dram").effective_capacity(
                    worker_count,
                    sm_freq_hz=inp.sm_freq_hz,
                    dram_freq_hz=inp.dram_freq_hz)
                * inp.dram_capacity_multiplier)
            dirty_replacement_completion_ns = (
                (program_traffic.a_dram_bytes
                 + program_traffic.b_dram_bytes) / dram_capacity
                + (program_traffic.predecessor_dirty_writeback_bytes
                   / dirty_capacity)) * 1.0e9
    if (kernel.anonymous_l2_dirty_bytes > 0
            and finite_ingress_completion_ns is not None
            and program_traffic is not None):
        operand_dram_bytes = (
            program_traffic.a_dram_bytes + program_traffic.b_dram_bytes)
        descriptor_stride_penalty_ns = (
            _cann_npu_gemm_cold_legacy_stride_penalty_ns(
                kernel, hardware, geometry)
            + _cann_npu_gemm_cold_v3_stride_penalty_ns(
                kernel, hardware, geometry)
            + _cann_npu_gemm_cold_small_m_stride_penalty_ns(
                kernel, hardware, geometry))
        source_specific_ingress_available = any(
            service is not None
            for service in (
                _cann_npu_gemm_exact_source_order_service(
                    kernel, hardware, geometry),
                _cann_npu_gemm_exact_finite_transition_service(
                    kernel, hardware, geometry),
                _cann_npu_gemm_cold_n_major_b_pair_time_ns(
                    kernel, hardware, geometry),
            ))
        if operand_dram_bytes > 0 and not source_specific_ingress_available:
            mte1_completion_ns = (
                _cann_npu_gemm_measured_mte1_pair_completion_ns(
                    kernel, hardware, geometry))
            cold_source_completion_ns = (
                hardware.aic_mte2_nd2nz_cold_stream_time_ns(
                operand_dram_bytes, worker_count)
                + descriptor_stride_penalty_ns
                + (mte1_completion_ns or 0.0))
    model_legacy_mte1 = _uses_cann_persistent_execution(kernel)
    if model_legacy_mte1:
        operand_source = MemorySourceDemand(
            active_units=worker_count,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        )
        operand_pair_count = (
            worker_count * geometry.output_tiles_per_worker
            * geometry.k_tiles)
        local_operand_paths = (
            MemoryPathDemand(
                name="mte1_a_l1_to_l0",
                access_kind="read",
                initial_transactions=None,
                nodes=(MemoryNodeDemand(
                    resource_role="l0a_buffer",
                    transactions=None,
                    bytes=float(
                        operand_pair_count * tile_m * tile_k * element_bytes),
                    transaction_bytes=None,
                ),),
                source=operand_source,
            ),
            MemoryPathDemand(
                name="mte1_b_l1_to_l0",
                access_kind="read",
                initial_transactions=None,
                nodes=(MemoryNodeDemand(
                    resource_role="l0b_buffer",
                    transactions=None,
                    bytes=float(
                        operand_pair_count * tile_k * tile_n * element_bytes),
                    transaction_bytes=None,
                ),),
                source=operand_source,
            ),
        )
        pipeline_input = dataclasses.replace(
            inp, memory_paths=inp.memory_paths + local_operand_paths)
        measured_mte1_completion_ns = (
            _cann_npu_gemm_measured_mte1_pair_completion_ns(
                kernel, hardware, geometry))
        if (measured_mte1_completion_ns is not None
                and _cann_npu_gemm_serial_l1_source_program(
                    kernel, geometry)):
            pipeline_input = dataclasses.replace(
                pipeline_input,
                resource_time_overrides_ns={
                    **pipeline_input.resource_time_overrides_ns,
                    "l0b_buffer": measured_mte1_completion_ns,
                },
            )
    else:
        pipeline_input = inp
    exact_source_order_service = _cann_npu_gemm_exact_source_order_service(
        kernel, hardware, geometry)
    finite_transition_service = (
        _cann_npu_gemm_exact_finite_transition_service(
            kernel, hardware, geometry))
    if (exact_source_order_service is not None
            and finite_transition_service is not None):
        raise ValueError("CANN GEMM exact MTE2 services overlap")
    exact_mte2_service = (
        exact_source_order_service
        if exact_source_order_service is not None
        else finite_transition_service)
    if exact_mte2_service is not None:
        pipeline_input = dataclasses.replace(
            pipeline_input,
            resource_time_overrides_ns={
                **pipeline_input.resource_time_overrides_ns,
                # The source-matched capture contributes its MTE2-only
                # descriptor span.  Its MTE2-to-MTE1 completion measurement
                # cannot also become an L1 resource floor: the analytical
                # program below already models the same MTE1 consumer and its
                # two A1/B1 producer credits.  Installing both would turn
                # one producer/consumer overlap into a second serial path.
                # Replace aggregate ingress, rather than charging the same
                # L2 bytes through the base-K expansion a second time.
                "l2_cache": exact_mte2_service["mte2_only_ns"],
            },
        )
    # The finite-ingress probe deliberately omits Cube and drains every
    # synthetic MTE2/MTE1 queue before returning.  Its wall-clock completion
    # therefore validates descriptor topology and the dirty-source evidence
    # domain above, but is not a lower bound on the production MatmulImpl,
    # whose double-buffered MTE2, MTE1 and Cube stages advance concurrently.
    # The common analytical pipeline below already carries all three resource
    # demands and their source-order dependencies; installing the drained
    # probe as an execution floor would serialize work a second time.
    if cold_source_completion_ns is not None:
        tensor_capacity_per_worker = hardware.resource_by_role(
            "compute_tensor").effective_capacity(
                worker_count,
                sm_freq_hz=inp.sm_freq_hz,
                dram_freq_hz=inp.dram_freq_hz) / worker_count
        cube_completion_ns = (
            geometry.output_tiles_per_worker * geometry.k_tiles
            * 2.0 * tile_m * tile_n * tile_k
            / tensor_capacity_per_worker * 1e9)
        completion_pairs = (
            geometry.output_tiles_per_worker
            * math.ceil(program.single_core_k / min(tile_k, 64)))
        cold_source_pipeline_ns = (
            max(cold_source_completion_ns, cube_completion_ns)
            + min(cold_source_completion_ns, cube_completion_ns)
            / completion_pairs)
        pipeline_input = dataclasses.replace(
            pipeline_input,
            execution_time_floor_ns=max(
                pipeline_input.execution_time_floor_ns or 0.0,
                cold_source_pipeline_ns),
        )
    if dirty_replacement_completion_ns is not None:
        pipeline_input = dataclasses.replace(
            pipeline_input,
            execution_time_floor_ns=max(
                pipeline_input.execution_time_floor_ns or 0.0,
                dirty_replacement_completion_ns),
        )
    if short_pipeline_completion_ns is not None:
        # The independent probe preserves the normalized two-slot A1/B1,
        # MTE1 and Cube dependency chain while omitting FixPipe and the full
        # operator.  It is therefore one source-program completion bound,
        # composed by max with the common analytical pipeline rather than
        # appended as a residual or a second serial resource path.
        pipeline_input = dataclasses.replace(
            pipeline_input,
            execution_time_floor_ns=max(
                pipeline_input.execution_time_floor_ns or 0.0,
                short_pipeline_completion_ns),
        )
    if region_pipeline_completion_ns is not None:
        # The probe preserves the decoded compact-region source order and
        # dependent MTE1/Cube chain while omitting FixPipe/output storage.
        # It is one finite-program completion bound and composes by max;
        # adding it to L2 or tensor service would count the same work twice.
        pipeline_input = dataclasses.replace(
            pipeline_input,
            execution_time_floor_ns=max(
                pipeline_input.execution_time_floor_ns or 0.0,
                region_pipeline_completion_ns),
        )
    if tail_region_service is not None:
        if analytical_mte2_ns is None:
            raise ValueError(
                "CANN tail-region service requires analytical MTE2 ingress")
        pipeline_input = dataclasses.replace(
            pipeline_input,
            resource_time_overrides_ns={
                **pipeline_input.resource_time_overrides_ns,
                "l2_cache": max(
                    analytical_mte2_ns,
                    tail_region_service["mte2_only_ns"]),
            },
        )
    analytical_envelope = _cann_npu_gemm_analytical_envelope(
        kernel, hardware, geometry, program_traffic)
    if program is not None:
        pipeline_input = _split_cann_npu_gemm_fixpipe_write_path(
            pipeline_input, kernel, geometry)
    if analytical_envelope is not None:
        limitations = tuple(
            limitation for limitation in pipeline_input.limitations
            if limitation not in {
                "aic_operand_replay_evidence_unavailable",
                "aic_store_fixpipe_tile_lifecycle_unavailable",
                "mte_transaction_geometry_unavailable",
                "memory_first_request_latency_unavailable",
                "mte2_descriptor_and_credit_evidence_unavailable",
            })
        return dataclasses.replace(
            pipeline_input,
            pipeline_plan=analytical_envelope,
            support_level="partial",
            limitations=limitations + (
                "aic_store_fixpipe_tile_lifecycle_unavailable",),
        )
    l0a_capacity = hardware.resource_by_role("l0a_buffer").effective_capacity(
        worker_count,
        sm_freq_hz=inp.sm_freq_hz,
        dram_freq_hz=inp.dram_freq_hz) / worker_count
    l0b_capacity = hardware.resource_by_role("l0b_buffer").effective_capacity(
        worker_count,
        sm_freq_hz=inp.sm_freq_hz,
        dram_freq_hz=inp.dram_freq_hz) / worker_count
    tensor_capacity = hardware.resource_by_role(
        "compute_tensor").effective_capacity(
            worker_count,
            sm_freq_hz=inp.sm_freq_hz,
            dram_freq_hz=inp.dram_freq_hz) / worker_count
    mte1_base_s = (
        tile_m * tile_k * element_bytes / l0a_capacity
        + tile_k * tile_n * element_bytes / l0b_capacity)
    cube_base_s = 2.0 * tile_m * tile_n * tile_k / tensor_capacity
    base_pair_count = (
        geometry.output_tiles_per_worker * geometry.k_tiles)
    pair_stream = hardware.aic_mte2_operand_pair_stream_service
    pair_overlap_domain = hardware.aic_mte2_operand_pair_overlap_domain
    finite_pair_overlap_domain = (
        hardware.aic_mte2_operand_pair_finite_overlap_domain)
    mixed_ring_overlap_domain = (
        hardware.aic_mte2_mixed_ring_long_overlap_domain)
    m_major_pair_program = (tile_m, tile_n, tile_k) == (256, 128, 64)
    established_long_pair_program = (
        pair_stream is not None
        and pair_stream["minimum_pairs_per_worker"]
        <= base_pair_count
        <= pair_stream["maximum_pairs_per_worker"])
    established_cold_pair_program = (
        _cann_npu_gemm_cold_pair_stream_supported(
            kernel, hardware, geometry))
    cold_n_major_b_pair_duration_ns = (
        _cann_npu_gemm_cold_n_major_b_pair_time_ns(
            kernel, hardware, geometry))
    expanded_n_major_pair_program = (
        pair_overlap_domain is not None
        and (tile_m, tile_k, tile_k, tile_n) == tuple(
            pair_overlap_domain[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns")))
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    operand_working_set_bytes = (b * m * k + k * n) * element_bytes
    expanded_n_major_overlap_covered = (
        expanded_n_major_pair_program
        and operand_working_set_bytes <= hardware.l2_size_bytes
        and pair_overlap_domain["minimum_active_aic_groups"]
        <= worker_count <= pair_overlap_domain["maximum_active_aic_groups"]
        and pair_overlap_domain["minimum_pairs_per_worker"]
        <= base_pair_count
        <= pair_overlap_domain["maximum_pairs_per_worker"])
    finite_n_major_overlap_covered = (
        (_uses_cann_region_program(kernel)
         or kernel.launch.algorithm == _CANN_MATMUL_V3_ASW_65536_ALGORITHM)
        and finite_pair_overlap_domain is not None
        and kernel.dtype == finite_pair_overlap_domain["dtype"]
        and finite_pair_overlap_domain["source_residency"] == "l2_resident"
        and (tile_m, tile_k, tile_k, tile_n) == tuple(
            finite_pair_overlap_domain[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns"))
        and operand_working_set_bytes <= hardware.l2_size_bytes
        and worker_count == finite_pair_overlap_domain["active_aic_groups"]
        and finite_pair_overlap_domain["minimum_pairs_per_worker"]
        <= base_pair_count
        <= finite_pair_overlap_domain["maximum_pairs_per_worker"])
    mixed_ring_n_major_overlap_covered = (
        (_uses_cann_region_program(kernel)
         or kernel.launch.algorithm == _CANN_MATMUL_V3_ASW_65536_ALGORITHM)
        and mixed_ring_overlap_domain is not None
        and kernel.dtype == mixed_ring_overlap_domain["dtype"]
        and mixed_ring_overlap_domain["source_residency_pairs"] == [
            ["resident", "resident"],
            ["resident", "cold"],
            ["cold", "resident"],
            ["cold", "cold"],
        ]
        and (tile_m, tile_k, tile_k, tile_n) == tuple(
            mixed_ring_overlap_domain[name]
            for name in ("a_rows", "a_columns", "b_rows", "b_columns"))
        and operand_working_set_bytes > hardware.l2_size_bytes
        and mixed_ring_overlap_domain["cold_source_ring_bytes_per_operand"]
        > hardware.l2_size_bytes
        and worker_count == mixed_ring_overlap_domain["active_aic_groups"]
        and mixed_ring_overlap_domain["minimum_pairs_per_worker"]
        <= base_pair_count
        <= mixed_ring_overlap_domain["maximum_pairs_per_worker"])
    single_l0c_control = hardware.aic_mte2_legacy_single_l0c_pair_control
    legacy_pair_signature = _cann_legacy_l1_pair_signature(kernel, geometry)
    single_l0c_control_signature = (
        None if single_l0c_control is None else tuple(
            single_l0c_control[name] for name in (
                "a_rows", "a_columns", "a_source_row_stride_elements",
                "b_rows", "b_columns", "b_source_row_stride_elements",
                "active_aic_groups", "pairs_per_worker")))
    legacy_source_slab_program = (
        model_legacy_mte1
        and (_cann_legacy_double_l0c_pair_signature(
                 kernel, hardware, geometry) is not None
             or (_cann_legacy_l0c_pipeline_depth(kernel, hardware) == 1
                 and legacy_pair_signature
                     == single_l0c_control_signature)))
    use_l1_slabs = (
        program is not None
        or exact_source_order_service is not None
        or legacy_source_slab_program
        or (mte1_base_s <= cube_base_s
            and (m_major_pair_program or established_long_pair_program
                 or established_cold_pair_program
                 or expanded_n_major_overlap_covered
                 or finite_n_major_overlap_covered
                  or mixed_ring_n_major_overlap_covered)))
    if use_l1_slabs:
        work_units = geometry.k_slabs
        # Both CANN worker programs invoke one complete Matmul program per
        # assigned output tile.  Preserve that finite program boundary in the
        # compact max-plus sequence: aggregating every output tile into one
        # synthetic K slab makes pipeline fill/drain grow with the whole
        # matrix instead of remaining one base-tile startup/drain.
        sequence_repeats = (
            geometry.output_tiles_per_worker
            if program is not None
            else 1)
        if (program is None
                and kernel.launch.algorithm
                    == _CANN_MATMUL_V3_ASW_65536_ALGORITHM):
            b, m, n, k = (
                int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
            l2_schedule = derive_cann_matmul_v3_l2_region_schedule(
                b * m, n, k, tile_m, tile_n, worker_count,
                int(hardware.l2_size_bytes), element_bytes)
            if l2_schedule[1] == 1:
                sequence_repeats = l2_schedule[0]
        task_suffix = "k_slab"
        dram_capacity_per_s = (
            hardware.resource_by_role("dram").effective_capacity(
                worker_count,
                sm_freq_hz=inp.sm_freq_hz,
                dram_freq_hz=inp.dram_freq_hz)
            * inp.dram_capacity_multiplier)
        mte2_duration_ns = _cann_npu_gemm_mte2_duration_ns(
            kernel, hardware, program_traffic, dram_capacity_per_s)
    else:
        work_units = geometry.k_tiles
        sequence_repeats = geometry.output_tiles_per_worker
        task_suffix = "k_tile"
        mte2_duration_ns = _cann_npu_gemm_base_tile_mte2_duration_ns(
            kernel, hardware)
    work_fraction = 1.0 / (
        worker_count * work_units * sequence_repeats)

    def k_group_task(
            name: str, repeat_count: int = 1) -> PipelineTaskDemand:
        output_prior_resources = (
            ("memory:fixpipe_write",)
            if any(path.name == "fixpipe_write"
                   for path in pipeline_input.memory_paths)
            else ())
        if model_legacy_mte1:
            stages = (
                PipelineStageDemand(
                    "load_l1_group", "memory:mte2_read", work_fraction,
                    False),
                PipelineStageDemand(
                    "load_l0a_group", "memory:mte1_a_l1_to_l0",
                    work_fraction, False,
                    predecessors=("load_l1_group",)),
                PipelineStageDemand(
                    "load_l0b_group", "memory:mte1_b_l1_to_l0",
                    work_fraction, False,
                    predecessors=("load_l0a_group",)),
                PipelineStageDemand(
                    "cube_group", "compute:compute_tensor", work_fraction,
                    False, predecessors=("load_l0b_group",),
                    prior_resources=output_prior_resources),
            )
        else:
            stages = (
                PipelineStageDemand(
                    "load_l1_group", "memory:mte2_read", work_fraction,
                    False),
                PipelineStageDemand(
                    "mte1_cube_group", "compute:compute_tensor", work_fraction,
                    False, predecessors=("load_l1_group",),
                    prior_resources=output_prior_resources),
            )
        return PipelineTaskDemand(
            name=name,
            worker=0,
            stages=stages,
            repeat_count=repeat_count,
        )

    if work_units == 1:
        tasks = (k_group_task(f"only_{task_suffix}"),)
    else:
        tasks = [k_group_task(f"first_{task_suffix}")]
        if work_units > 2:
            tasks.append(k_group_task(
                f"middle_{task_suffix}s", work_units - 2))
        tasks.append(k_group_task(f"final_{task_suffix}"))
        tasks = tuple(tasks)
    if any(path.name == "fixpipe_write"
           for path in pipeline_input.memory_paths):
        tasks += (PipelineTaskDemand(
            name="finite_output_tile_stream",
            worker=0,
            stages=(PipelineStageDemand(
                "fixpipe_write", "memory:fixpipe_write",
                1.0 / (worker_count * sequence_repeats), False,
                memory_exact_request_demands=(
                    (MemoryExactRequestDemand(
                        resource_role="dram",
                        service=fixpipe_service,
                        request_sizes_by_worker=((sequence_repeats,),),
                    ),)
                    if fixpipe_service is not None else ()),
                prior_resources=(
                    "memory:mte2_read", "compute:compute_tensor"),
            ),),
        ),)
    sequence = PipelineTaskSequenceDemand(
        name=f"aic_worker_{task_suffix}_loop",
        worker=0,
        tasks=tasks,
        repeat_count=sequence_repeats,
    )
    if model_legacy_mte1:
        serial_l1_source_program = (
            _cann_npu_gemm_serial_l1_source_program(kernel, geometry))
        buffers = (
            PipelineBuffer(
                name="l1_a_worker_group_lifetime",
                producer_stage="load_l1_group",
                consumer_stage="load_l0a_group",
                slots=(1 if serial_l1_source_program else pipeline_depth),
                scope="worker",
            ),
            PipelineBuffer(
                name="l1_b_worker_group_lifetime",
                producer_stage="load_l1_group",
                consumer_stage="load_l0b_group",
                slots=(1 if serial_l1_source_program else pipeline_depth),
                scope="worker",
            ),
            PipelineBuffer(
                name="l0a_worker_group_lifetime",
                producer_stage="load_l0a_group",
                consumer_stage="cube_group",
                slots=pipeline_depth,
                scope="worker",
            ),
            PipelineBuffer(
                name="l0b_worker_group_lifetime",
                producer_stage="load_l0b_group",
                consumer_stage="cube_group",
                slots=pipeline_depth,
                scope="worker",
            ),
        )
    else:
        # MatMulV3's base branch selects MM_CFG_NO_PRELOAD.  Its source
        # completes CopyIn(A1, B1) before the worker enters ComputeKDB, so the
        # next logical L1 group cannot reuse a worker slot early when there is
        # only one output tile assigned to that worker.  Restrict this to the
        # independently measured multi-AIC MTE2 service domain; smaller grids
        # and multi-output programs retain their overlap envelope.
        mte2_service = hardware.aic_mte2_nd2nz_service
        minimum_multiworker_groups = (
            None if mte2_service is None else min(
                count for count in mte2_service["active_aic_groups"]
                if count > 1))
        l1_lifetime_slots = (
            1 if (_uses_cann_persistent_execution(kernel)
                  and minimum_multiworker_groups is not None
                  and geometry.worker_count >= minimum_multiworker_groups
                  and geometry.output_tiles_per_worker == 1)
            else pipeline_depth)
        buffers = (PipelineBuffer(
            name="l1_worker_group_lifetime",
            producer_stage="load_l1_group",
            consumer_stage="mte1_cube_group",
            slots=l1_lifetime_slots,
            scope="worker",
        ),)
    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=(PipelineTaskProgramDemand(
            name="uniform_aic_program",
            worker=0,
            patterns=(sequence,),
            order=(0,),
            worker_multiplicity=worker_count,
        ),),
        buffers=buffers,
    )
    attached = attach_pipeline_work_plan(
        pipeline_input, hardware, plan, promote_npu_partial=True)
    limitations = tuple(
        limitation for limitation in attached.limitations
        if limitation not in (
            "aic_operand_replay_evidence_unavailable",
            "mte_transaction_geometry_unavailable",
            "memory_first_request_latency_unavailable",
            "mte2_descriptor_and_credit_evidence_unavailable",
        ) and not (
            fixpipe_service is not None
            and limitation in {
                "aic_store_fixpipe_tile_lifecycle_unavailable",
                "aic_read_write_transaction_split_unavailable",
            }))
    if exact_mte2_service is None:
        if cold_n_major_b_pair_duration_ns is not None:
            attached = dataclasses.replace(
                attached,
                resource_time_overrides_ns={
                    **attached.resource_time_overrides_ns,
                    "l2_cache": cold_n_major_b_pair_duration_ns,
                },
            )
        elif "l2_cache" not in attached.resource_time_overrides_ns:
            if mte2_duration_ns is None:
                return dataclasses.replace(
                    attached,
                    limitations=limitations + (
                        "cann_npu_gemm_mte2_service_domain_unavailable",))
            attached = dataclasses.replace(
                attached,
                resource_time_overrides_ns={
                    **attached.resource_time_overrides_ns,
                    "l2_cache": mte2_duration_ns,
                },
            )

    return dataclasses.replace(attached, limitations=limitations)


def _cann_matmul_v3_asw_65536_operand_traffic(
        kernel: KernelSpec) -> Tuple[float, float, float, float, float]:
    """Return source-exact A/B replay, output, and full-descriptor bytes."""

    if kernel.variant != "gemm" or kernel.dtype != "bf16":
        raise ValueError(
            "CANN MatMulV3 ASW 65536 requires BF16 GEMM semantics")
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k != 1:
        raise ValueError(
            "CANN MatMulV3 ASW 65536 requires a non-split retained tile")
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if (tile_m, tile_n, tile_k) not in (
            (128, 256, 64), (256, 128, 64)):
        raise ValueError(
            "CANN MatMulV3 ASW 65536 requires a source-defined ASW tile")
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])

    # ASW invokes CopyIn for both operands at every output/K tile. SetSingleShape
    # clips the M/N tails, so replay uses semantic rather than padded bytes.
    a_l2_bytes = float(
        math.ceil(n / tile_n) * rows * k * element_bytes)
    b_l2_bytes = float(
        math.ceil(rows / tile_m) * n * k * element_bytes)
    output_l2_bytes = float(rows * n * element_bytes)

    # The independent service probes cover only complete 128x256x64 pairs.
    full_tile_count = (
        (rows // tile_m) * (n // tile_n) * (k // tile_k))
    typed_a_bytes = float(
        full_tile_count * tile_m * tile_k * element_bytes)
    typed_b_bytes = float(
        full_tile_count * tile_k * tile_n * element_bytes)
    return (
        a_l2_bytes, b_l2_bytes, output_l2_bytes,
        typed_a_bytes, typed_b_bytes,
    )


def _cann_gemm_operand_signature(
        kernel: KernelSpec, service: str,
        operand_role: str) -> MemoryAccessClassSignature:
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    if operand_role == "activation":
        logical_row_bytes = tile_m * tile_k * element_bytes
    elif operand_role == "weight":
        logical_row_bytes = tile_k * tile_n * element_bytes
    else:
        raise ValueError("CANN GEMM operand must be activation or weight")
    return MemoryAccessClassSignature(
        access_class=service,
        dtype=kernel.dtype,
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=logical_row_bytes,
        operand_role=operand_role,
        tile_n=tile_n,
        group_size_m=1,
    )


def _cann_matmul_v3_all_fit_operand_classes(
        kernel: KernelSpec, hardware,
        typed_a_bytes: float, typed_b_bytes: float
) -> Tuple[float, float, float, float] | None:
    """Split full descriptors into compulsory misses and later L2 hits."""

    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    rows = b * m
    element_bytes = int(kernel.shape["dtype_bytes"])
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    compulsory_bytes = (
        rows * k + k * n + rows * n) * element_bytes
    if compulsory_bytes > hardware.l2_size_bytes:
        return None
    k_tiles = k // tile_k
    a_miss_bytes = float(
        (rows // tile_m) * k_tiles * tile_m * tile_k * element_bytes)
    b_miss_bytes = float(
        (n // tile_n) * k_tiles * tile_k * tile_n * element_bytes)
    if a_miss_bytes > typed_a_bytes or b_miss_bytes > typed_b_bytes:
        raise ValueError("ASW compulsory full tiles exceed typed replay")
    return (
        typed_a_bytes - a_miss_bytes,
        a_miss_bytes,
        typed_b_bytes - b_miss_bytes,
        b_miss_bytes,
    )


def _attach_cann_gemm_operand_service(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        traffic: GemmSourceOrderTraffic | None = None) -> PredictionInput:
    """Bind normalized A/B hit and miss streams to measured services."""

    if inp.mem_model != "npu_tile":
        raise ValueError("CANN GEMM operand service requires npu_tile")
    paths = tuple(path for path in inp.memory_paths if path.name == "mte2_read")
    if len(paths) != 1:
        raise ValueError(
            "CANN MatMulV3 operand service requires one MTE2 read path")
    read_path = paths[0]
    if read_path.access_class_demands:
        raise ValueError("CANN MatMulV3 MTE2 path already has a typed service")
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if ((_uses_cann_region_program(kernel)
         or kernel.launch.algorithm == _CANN_MATMUL_V3_ASW_65536_ALGORITHM)
            and (tile_m, tile_n, tile_k) == (256, 128, 64)):
        geometry = _cann_npu_gemm_pipeline_geometry(kernel, hardware)
        service = hardware.aic_matmul_v3_asw_mte2_pair_stream_service
        if (geometry is None or service is None
                or geometry.worker_count != service["active_aic_groups"]):
            return inp
        pair_count = (
            geometry.output_tiles_per_worker * geometry.k_tiles)
        if pair_count < service["minimum_pairs_per_worker"]:
            return inp
        overrides = dict(inp.resource_time_overrides_ns)
        overrides["l2_cache"] = (
            hardware.aic_matmul_v3_asw_mte2_pair_time_ns(
                tile_m, tile_n, tile_k, geometry.worker_count, pair_count))
        limitations = tuple(
            limitation for limitation in inp.limitations
            if limitation != "aic_operand_replay_evidence_unavailable")
        return dataclasses.replace(
            inp,
            resource_time_overrides_ns=overrides,
            limitations=limitations,
        )
    if traffic is None:
        _, _, _, typed_a_bytes, typed_b_bytes = (
            _cann_matmul_v3_asw_65536_operand_traffic(kernel))
        all_fit_classes = _cann_matmul_v3_all_fit_operand_classes(
            kernel, hardware, typed_a_bytes, typed_b_bytes)
        if all_fit_classes is None:
            service_specs = (
                (_CANN_OPERAND_A_L2_HIT_SERVICE, "activation", typed_a_bytes),
                (_CANN_OPERAND_B_L2_HIT_SERVICE, "weight", typed_b_bytes),
            )
        else:
            a_hit_bytes, a_miss_bytes, b_hit_bytes, b_miss_bytes = (
                all_fit_classes)
            service_specs = (
                (_CANN_OPERAND_A_L2_HIT_SERVICE, "activation", a_hit_bytes),
                (_CANN_OPERAND_A_DRAM_MISS_SERVICE, "activation", a_miss_bytes),
                (_CANN_OPERAND_B_L2_HIT_SERVICE, "weight", b_hit_bytes),
                (_CANN_OPERAND_B_DRAM_MISS_SERVICE, "weight", b_miss_bytes),
            )
    else:
        service_specs = (
            (_CANN_OPERAND_A_L2_HIT_SERVICE, "activation",
             traffic.a_l2_bytes - traffic.a_dram_bytes),
            (_CANN_OPERAND_A_DRAM_MISS_SERVICE, "activation",
             traffic.a_dram_bytes),
            (_CANN_OPERAND_B_L2_HIT_SERVICE, "weight",
             traffic.b_l2_bytes - traffic.b_dram_bytes),
            (_CANN_OPERAND_B_DRAM_MISS_SERVICE, "weight",
             traffic.b_dram_bytes),
        )
    service_requests = tuple(
        (
            service_name,
            "read",
            "l2_cache",
            _cann_gemm_operand_signature(
                kernel, service_name, operand_role),
        )
        for service_name, operand_role, _ in service_specs
    )
    if any(not any(
            (candidate.service, candidate.access_kind,
             candidate.ingress_resource_role, candidate.signature) == request
            for candidate in hardware.memory_access_class_services)
            for request in service_requests):
        limitation = "cann_gemm_typed_operand_service_unavailable"
        return dataclasses.replace(
            inp,
            limitations=(
                inp.limitations if limitation in inp.limitations else
                inp.limitations + (limitation,)),
        )
    services = tuple(
        hardware.memory_access_class_service(*request)
        for request in service_requests)
    demands = tuple(
        MemoryAccessClassDemand(
            service=service.service,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=logical_bytes,
            signature=service.signature,
        )
        for service, (_, _, logical_bytes) in zip(services, service_specs)
        if logical_bytes > 0.0
    )
    typed_ingress_ns = math.fsum(
        logical_bytes / service.logical_byte_capacity_per_s
        for service, (_, _, logical_bytes) in zip(services, service_specs)
        if logical_bytes > 0.0
    ) * 1e9
    l2_capacity = hardware.resource_by_role("l2_cache").effective_capacity(
        inp.active_units,
        sm_freq_hz=inp.sm_freq_hz,
        dram_freq_hz=inp.dram_freq_hz,
    )
    generic_l2_ns = inp.demand_per_resource["l2_cache"] / l2_capacity * 1e9
    typed_read_path = dataclasses.replace(
        read_path,
        access_class_demands=demands,
    )
    updated_paths = tuple(
        typed_read_path if path is read_path else path
        for path in inp.memory_paths)
    overrides = dict(inp.resource_time_overrides_ns)
    geometry = _cann_npu_gemm_pipeline_geometry(kernel, hardware)
    finite_fixpipe_service = (
        None if geometry is None else _cann_npu_gemm_fixpipe_finite_service(
            kernel, hardware, geometry))
    terminal_fixpipe_ns = (
        hardware.aic_fixpipe_bf16_terminal_tile_time_ns(tile_m, tile_n)
        if finite_fixpipe_service is None else 0.0)
    overrides["l2_cache"] = (
        max(generic_l2_ns, typed_ingress_ns) + terminal_fixpipe_ns)
    limitations = tuple(
        limitation for limitation in inp.limitations
        if limitation != "aic_operand_replay_evidence_unavailable")
    return dataclasses.replace(
        inp,
        memory_paths=updated_paths,
        resource_time_overrides_ns=overrides,
        limitations=limitations,
    )


def _architecture_input(inp: PredictionInput, kernel: KernelSpec, hardware, logical_program) -> PredictionInput:
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        return dataclasses.replace(inp, mem_model="npu_tile")
    elif memory_model == "pim_bank":
        demand = dict(inp.demand_per_resource)
        # A bank-attached compute unit has no cache between itself and the
        # array it computes out of; the bank stream below owns that path.
        for role in ("xbar", "smem", "l1_cache", "l2_cache", "dram"):
            demand[role] = 0.0
        if (kernel.variant == "gemm" or (kernel.variant == "gemv"
                and kernel.shape["b"] * kernel.shape["m"] == 1)):
            from src.kernel.memory_access import unique_by
            views = unique_by(logical_program.views, "view_id")
            rows, k = views["A"].shape
            _, n = views["B"].shape
        else:
            rows = int(kernel.shape["b"]) * int(kernel.shape["m"])
            k, n = int(kernel.shape["k"]), int(kernel.shape["n"])
        stream = matmul_stream(rows, k, n)
        return dataclasses.replace(
            inp, demand_per_resource=demand, mem_model="pim_bank",
            active_units=hardware.num_compute_units,
            # A bank-attached launch has no CTA waves to quantize against:
            # the partitioner spreads the operand over every unit, and the
            # bank stream carries the resulting per-unit column count.
            eta=1.0, wave_efficiency=1.0,
            execution_time_floor_ns=max(
                inp.execution_time_floor_ns or 0.0,
                bank_stream_ns((stream,), hardware)))
    elif memory_model == "systolic_scratchpad":
        demand = dict(inp.demand_per_resource)
        for role in ("xbar", "smem", "l1_cache"):
            demand[role] = 0.0
        return dataclasses.replace(
            inp, demand_per_resource=demand, mem_model="systolic_scratchpad")
    elif memory_model == "gpu_mshr":
        return inp
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _uses_k_tiled_pipeline(kernel: KernelSpec) -> bool:
    """Return whether the physical GEMM exposes a staged K-tile loop."""

    _, _, tile_k, pipeline_depth = _tile(kernel)
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k is None:
        raise ValueError("staged GEMM requires retained split-K semantics")
    return (
        tiling.split_k >= 1
        and tile_k > 0
        and pipeline_depth is not None
    )


_FINITE_ASYNC_OPERAND_STREAM_ELEMENT_BYTES = {"fp16": 2, "bf16": 2}


def _finite_async_operand_stream_profile(kernel: KernelSpec, hardware):
    """Return a measured finite async-copy topology, when one exactly fits.

    This boundary intentionally uses only launch-visible physical facts:
    CTA geometry, operand-pair allocation, declared ring depth, and K-split
    semantics.  It does not select on a kernel symbol, source text, or an
    evaluation shape.  A hardware can publish another instance of the same
    profile with its own independent copy-only evidence.

    The probe (``staged_ldgsts_finite_stream.cu``) is copy-only: it stages
    LDGSTS bytes through the ring and never interprets them, so its measured
    completion is a function of element byte width alone, not of which
    floating-point format those bytes hold.  ``profile["dtype"]`` records
    only which format the calibration used to fix ``operand_pair_bytes``;
    the ``operand_pair_bytes`` check below already requires the production
    launch's own tile geometry to reproduce that exact byte count, so
    admitting another format of the same element width does not extend the
    mechanism past what it measured.
    """

    profile = hardware.finite_async_operand_stream_completion
    if profile is None:
        return None
    tiling = kernel.launch.tiling
    element_widths = _FINITE_ASYNC_OPERAND_STREAM_ELEMENT_BYTES
    if (hardware.kernel_memory_model != "gpu_mshr"
            or kernel.dtype not in element_widths
            or profile["dtype"] not in element_widths
            or element_widths[kernel.dtype]
            != element_widths[profile["dtype"]]
            or kernel.variant not in ("gemm", "batched_gemm", "moe_router")
            or kernel.launch.block_size != profile["threads_per_cta"]
            or kernel.launch.grid_size != profile["grid_ctas"]
            or hardware.num_compute_units < profile["grid_ctas"]
            or profile["resident_ctas_per_compute_unit"] != 1
            or tiling is None
            or tiling.split_k != 1
            or tiling.pipeline_depth != profile["declared_pipeline_depth"]
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return None
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if tile_k <= 0:
        raise ValueError("finite async operand stream has no K tile")
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
    operand_pair_bytes = (
        tile_m * tile_k * a_element_bytes
        + tile_k * tile_n * b_element_bytes)
    if operand_pair_bytes != profile["operand_pair_bytes"]:
        return None
    return profile


def _gpu_k_tiled_operand_buffer_slots(kernel: KernelSpec, hardware) -> int:
    """Return the measured number of complete operand-pair ring slots."""

    _, _, _, pipeline_depth = _tile(kernel)
    if pipeline_depth is None:
        raise ValueError("K-tiled GEMM has no pipeline depth")
    profile = _finite_async_operand_stream_profile(kernel, hardware)
    return (
        pipeline_depth if profile is None
        else profile["complete_operand_pair_slots"])


def _attach_gpu_k32_deep_cohort_operand_streams(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Price sustained A/B loads after a deep K32 CTA cohort fills."""

    tiling = kernel.launch.tiling
    if (tiling is None
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or tiling.split_k is None
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0
            or hardware.parallel_width != 32):
        return inp
    _, _, tile_k, pipeline_depth = _tile(kernel)
    if tile_k != 32:
        return inp
    if pipeline_depth is None:
        cohort_fill = hardware.max_blocks_per_unit
    elif pipeline_depth == 5:
        cohort_fill = pipeline_depth
    else:
        return inp
    ctas_per_worker = math.ceil(inp.grid_size / inp.active_units)
    if ctas_per_worker <= cohort_fill:
        return inp

    paths = []
    matched_read_nodes = 0
    for path in inp.memory_paths:
        nodes = []
        for node in path.nodes:
            if node.resource_role == "l2_cache":
                if path.name == "global_read":
                    # Staged SASS has distinct A/B operand load streams.  A
                    # complete depth of queued CTA cohorts can keep both live;
                    # the documented L2 crossbar remains the aggregate cap.
                    node = dataclasses.replace(
                        node,
                        independent_service_streams=2,
                        capacity_ceiling_resource_role="xbar",
                    )
                    matched_read_nodes += 1
            nodes.append(node)
        paths.append(dataclasses.replace(path, nodes=tuple(nodes)))
    if matched_read_nodes != 1:
        raise ValueError(
            "staged GPU GEMM requires exactly one global-read L2 node")
    return dataclasses.replace(inp, memory_paths=tuple(paths))


def _attach_sm80_staged_ldgsts_l2_service(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Bound exact block-128 K32 operand streams by their LDGSTS service."""

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or kernel.dtype != "fp16"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (128, 1, 1)
            or tiling is None
            or tiling.split_k is None):
        return inp
    tile_shape = tuple(
        tiling.tile_shape[name] for name in ("m", "n", "k"))
    if (tiling.pipeline_depth is None
            and tile_shape == (128, 128, 32)
            and _SM80_STAGED_LDGSTS_L2_READ_ROLE in hardware._role_map):
        service_role = _SM80_STAGED_LDGSTS_L2_READ_ROLE
    elif (tiling.split_k == 1
          and tiling.pipeline_depth == 6
          and tile_shape == (64, 128, 32)
          and math.ceil(inp.grid_size / inp.active_units) >= 2
          and _SM80_K32_DEPTH6_LDGSTS_L2_READ_ROLE in hardware._role_map):
        service_role = _SM80_K32_DEPTH6_LDGSTS_L2_READ_ROLE
    else:
        return inp
    paths = []
    matched_read_nodes = 0
    for path in inp.memory_paths:
        nodes = []
        for node in path.nodes:
            if path.name == "global_read" and node.resource_role == "l2_cache":
                # The CUTLASS launch has independent A/B staged operand
                # streams.  Two generic streams expose enough upstream work;
                # the independently measured LDGSTS resource is the exact
                # aggregate service ceiling for this access class.
                node = dataclasses.replace(
                    node,
                    independent_service_streams=2,
                    capacity_ceiling_resource_role=service_role,
                )
                matched_read_nodes += 1
            nodes.append(node)
        paths.append(dataclasses.replace(path, nodes=tuple(nodes)))
    if matched_read_nodes != 1:
        raise ValueError(
            "SM80 staged LDGSTS GEMM requires exactly one global-read L2 "
            "node")
    return dataclasses.replace(inp, memory_paths=tuple(paths))


def _sm80_depth3_operand_signature(
        service: str) -> MemoryAccessClassSignature:
    """Return the three-slot 48-KiB operand-stream service signature."""

    return MemoryAccessClassSignature(
        access_class=service,
        dtype="fp16",
        tile_m=128,
        tile_k=64,
        logical_row_bytes=(128 * 64 + 64 * 256) * 2,
    )


def _uses_sm80_depth3_operand_source_service(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether the launch matches the measured three-slot source.

    The two macro-tile orientations exchange the 16-KiB and 32-KiB A/B
    panels.  Both still issue 3,072 16-byte LDGSTS vectors through the same
    256-thread, three-slot, one-CTA-resident source program, so the measured
    aggregate service and one 35-arrival launch startup are orientation
    invariant.
    """

    tiling = kernel.launch.tiling
    return (
        hardware.kernel_memory_model == "gpu_mshr"
        and hardware.parallel_width == 32
        and kernel.dtype == "fp16"
        and kernel.launch.algorithm == "nvidia_cublaslt_gemm"
        and kernel.launch.block == (256, 1, 1)
        and tiling is not None
        and tiling.split_k is not None
        and tiling.split_k > 1
        and tiling.pipeline_depth == 3
        and tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
        in {(128, 256, 64), (256, 128, 64)}
        and kernel.launch.parameters[
            "split_k_partial_bytes_per_element"] == 0.0
        and hardware.has_memory_access_class_service(
            _SM80_DEPTH3_OPERAND_L2_HIT_SERVICE)
        and hardware.has_memory_access_class_service(
            _SM80_DEPTH3_OPERAND_DRAM_MISS_SERVICE)
    )


def _attach_sm80_depth3_operand_source_service(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Replace generic ingress with serial hit/miss LDGSTS source service."""

    if not _uses_sm80_depth3_operand_source_service(kernel, hardware):
        return inp
    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1:
        raise ValueError(
            "SM80 depth-three operand source requires one global read path")
    read_path, = read_paths
    nodes = {node.resource_role: node for node in read_path.nodes}
    if set(nodes) != {"l2_cache", "dram"}:
        raise ValueError(
            "SM80 depth-three operand source requires L2 and DRAM nodes")
    l2_ingress_bytes = nodes["l2_cache"].bytes
    dram_miss_bytes = nodes["dram"].bytes
    l2_hit_bytes = l2_ingress_bytes - dram_miss_bytes
    if l2_hit_bytes < 0.0:
        raise ValueError(
            "SM80 depth-three DRAM misses exceed L2 ingress bytes")
    service_bytes = (
        (_SM80_DEPTH3_OPERAND_L2_HIT_SERVICE, l2_hit_bytes),
        (_SM80_DEPTH3_OPERAND_DRAM_MISS_SERVICE, dram_miss_bytes),
    )
    demands = []
    for service_name, logical_bytes in service_bytes:
        if logical_bytes == 0.0:
            continue
        signature = _sm80_depth3_operand_signature(service_name)
        service = hardware.memory_access_class_service(
            service_name, "read", "l2_cache", signature)
        demands.append(MemoryAccessClassDemand(
            service=service.service,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=logical_bytes,
            signature=service.signature,
            replaces_ingress_path_service=True,
        ))
    if not demands:
        raise ValueError("SM80 depth-three operand source has no read bytes")
    typed_read_path = dataclasses.replace(
        read_path, access_class_demands=tuple(demands))
    warp_issue_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "warp_issue")
    tile_shape = tuple(
        kernel.launch.tiling.tile_shape[name]
        for name in ("m", "n", "k"))
    if not warp_issue_demands and tile_shape == (256, 128, 64):
        return dataclasses.replace(
            inp,
            memory_paths=tuple(
                typed_read_path if path is read_path else path
                for path in inp.memory_paths),
        )
    if len(warp_issue_demands) != 1:
        raise ValueError(
            "SM80 depth-three operand source requires one warp-issue demand")
    warp_issue_demand, = warp_issue_demands
    if (warp_issue_demand.service_demands
            or warp_issue_demand.exact_request_demands):
        raise ValueError(
            "SM80 depth-three operand source requires aggregate warp issue")
    warp_issue = hardware.resource_by_role("warp_issue")
    warp_issue_capacity = warp_issue.effective_capacity(
        inp.active_units,
        sm_freq_hz=inp.sm_freq_hz,
        dram_freq_hz=inp.dram_freq_hz,
    )
    if warp_issue.scope == "shared":
        execution_domain_units = hardware.execution_units_for_task(
            inp.launch_task_kind)
        warp_issue_capacity *= (
            min(inp.active_units, execution_domain_units)
            / execution_domain_units)
    elif warp_issue.scope == "per_unit":
        pass
    else:
        raise ValueError(
            "SM80 depth-three operand source has unknown warp-issue scope")
    effective_warp_issue_capacity = (
        warp_issue_capacity
        * warp_issue_demand.shape_efficiency
        * warp_issue_demand.schedule_efficiency)
    if effective_warp_issue_capacity <= 0.0:
        raise ValueError(
            "SM80 depth-three operand source has no warp-issue capacity")
    if hardware.l2_latency_ns is None:
        raise ValueError(
            "SM80 depth-three operand source requires L2 latency")
    # One 128x64 A plus 64x256 B panel contains 3,072 LDGSTS vectors, or
    # twelve issue rounds across 256 threads.  Three initially committed
    # groups expose 36 arrivals.  The dependency path's single global-read
    # latency owns the first, leaving one launch-local sequence of 35
    # source-ordered L2 arrivals before its first DEPBAR.LE SB0,1.  The
    # independent full-SM wave-scaling probe establishes that later CTA waves
    # overlap this finite fill with the preceding resident loop; multiplying
    # the source startup by the launch's CTA cohorts would therefore price the
    # same launch-local dependency as a serial per-wave latency.  DRAM misses
    # change steady source service, not this fixed instruction-side prologue.
    source_prologue_ns = (
        _SM80_DEPTH3_ADDITIONAL_L2_ARRIVALS_PER_LAUNCH
        * hardware.l2_latency_ns)
    analytical_warp_issue_ns = (
        warp_issue_demand.operations / effective_warp_issue_capacity * 1e9)
    if "warp_issue" in inp.resource_time_overrides_ns:
        raise ValueError(
            "SM80 depth-three operand source cannot replace a warp-issue "
            "override")
    return dataclasses.replace(
        inp,
        memory_paths=tuple(
            typed_read_path if path is read_path else path
            for path in inp.memory_paths),
        resource_time_overrides_ns={
            **inp.resource_time_overrides_ns,
            "warp_issue": analytical_warp_issue_ns + source_prologue_ns,
        },
    )


def _finite_async_operand_stream_signature(
        kernel: KernelSpec, service: str) -> MemoryAccessClassSignature:
    """Return the typed capability signature of one complete operand pair."""

    tile_m, tile_n, tile_k, _ = _tile(kernel)
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)

    return MemoryAccessClassSignature(
        access_class=service,
        dtype=kernel.dtype,
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=(
            tile_m * tile_k * a_element_bytes
            + tile_k * tile_n * b_element_bytes),
    )


def _uses_finite_async_operand_stream_access_service(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether typed ingress is available for a measured topology."""

    if _finite_async_operand_stream_profile(kernel, hardware) is None:
        return False
    for service in (
            _FINITE_ASYNC_OPERAND_L2_HIT_SERVICE,
            _FINITE_ASYNC_OPERAND_DRAM_MISS_SERVICE):
        signature = _finite_async_operand_stream_signature(kernel, service)
        if not any(
                candidate.service == service
                and candidate.access_kind == "read"
                and candidate.ingress_resource_role == "l2_cache"
                and candidate.signature == signature
                for candidate in hardware.memory_access_class_services):
            return False
    return True


def _attach_finite_async_operand_stream_access_service(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Replace generic ingress with measured async-copy tier services."""

    if not _uses_finite_async_operand_stream_access_service(kernel, hardware):
        return inp
    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1:
        raise ValueError(
            "SM80 64x64 depth-six source requires one global read path")
    read_path, = read_paths
    nodes = {node.resource_role: node for node in read_path.nodes}
    if set(nodes) != {"l2_cache", "dram"}:
        raise ValueError(
            "SM80 64x64 depth-six source requires L2 and DRAM nodes")
    l2_ingress_bytes = nodes["l2_cache"].bytes
    dram_miss_bytes = nodes["dram"].bytes
    l2_hit_bytes = l2_ingress_bytes - dram_miss_bytes
    if l2_hit_bytes < 0.0:
        raise ValueError(
            "SM80 64x64 depth-six DRAM misses exceed L2 ingress bytes")
    service_bytes = (
        (_FINITE_ASYNC_OPERAND_L2_HIT_SERVICE, l2_hit_bytes),
        (_FINITE_ASYNC_OPERAND_DRAM_MISS_SERVICE, dram_miss_bytes),
    )
    demands = []
    for service_name, logical_bytes in service_bytes:
        if logical_bytes == 0.0:
            continue
        service = hardware.memory_access_class_service(
            service_name, "read", "l2_cache",
            _finite_async_operand_stream_signature(kernel, service_name))
        demands.append(MemoryAccessClassDemand(
            service=service.service,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=logical_bytes,
            signature=service.signature,
            replaces_ingress_path_service=True,
        ))
    if not demands:
        raise ValueError(
            "SM80 64x64 depth-six source has no operand read bytes")
    typed_read_path = dataclasses.replace(
        read_path, access_class_demands=tuple(demands))
    return dataclasses.replace(
        inp,
        memory_paths=tuple(
            typed_read_path if path is read_path else path
            for path in inp.memory_paths),
    )


def _attach_tma_quiescent_read_ramp_floor(inp, kernel, hardware):
    """A same-transport finite-copy bound; quiescent start is an assumption."""
    profile = hardware.dram_finite_tma_read_stream_ramp
    if profile is None or hardware.kernel_memory_model != "gpu_mshr":
        return inp
    if kernel.l2_execution_context.same_program_operand_replays > 0:
        # Explicit immediately preceding memory activity contradicts this
        # isolated/quiescent start approximation. Keep the existing services.
        return inp
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    a_width, b_width, _ = _operand_element_bytes(kernel)
    request_bytes = tile_k * (tile_m * a_width + tile_n * b_width)
    if (kernel.launch.block_size != profile["threads_per_cta"]
            or tile_k != profile["tile_k"]
            or (a_width, b_width) != (profile["operand_element_bytes"],) * 2
            or not profile["request_total_bytes_min"] <= request_bytes <= profile["request_total_bytes_max"]):
        return inp
    grid = kernel.launch.grid_size
    if not profile["grids"][0] <= grid <= profile["grids"][-1]:
        return inp
    read_paths = tuple(path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1:
        return inp
    source = read_paths[0].source
    if source.producers_per_unit != 1.0:
        return inp
    # The common source publishes one aggregate producer per active unit.
    # Surplus launched CTAs need not carry useful operands, while a launch
    # cannot have more physical producers than its CTA count. Preserve every
    # byte and normalize its stream length to this bounded source population.
    # This is not a hardware credit count or a native CTA ownership claim.
    producers = min(grid, source.active_units)
    if not profile["grids"][0] <= producers <= profile["grids"][-1]:
        return inp
    dram_bytes = math.fsum(node.bytes for node in read_paths[0].nodes
        if node.resource_role == "dram")
    length = dram_bytes / producers
    lengths = profile["lengths_bytes_per_cta"]
    if not lengths[0] <= length <= lengths[-1]:
        return inp
    measured_grid = next(g for g in profile["grids"] if g >= producers)
    measured_length = next(n for n in lengths if n >= length)
    capacity = profile["dram_read_capacity_bytes_per_s"] * profile["ramp"][measured_grid, measured_length]
    completion_ns = 1e9 * dram_bytes / capacity
    # The measured copy window includes its own ring fill/drain. Formula
    # separately adds the existing launch-local transport prefix. Subtract
    # that whole overlapping prefix here so the final composition is
    # max(pipeline + prefix, copy_completion), rather than charging it twice.
    prefix_ns = (0.0 if inp.finite_mixed_operand_stream_startup_ns is None
        else inp.finite_mixed_operand_stream_startup_ns)
    body_floor_ns = completion_ns - prefix_ns
    if body_floor_ns <= 0.0:
        return inp
    return dataclasses.replace(inp,
        execution_time_floor_ns=max(
            0.0 if inp.execution_time_floor_ns is None else inp.execution_time_floor_ns,
            body_floor_ns),
        limitations=inp.limitations + (
            "tma_finite_read_completion_quiescent_start_aggregate_assumption",
            "tma_finite_read_length_dram_bytes_per_active_producer_approximation",))


def _attach_dram_finite_stream_ramp_floor(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Bound completion by the measured ramp of a finite DRAM read stream.

    The reference covers the retained ld.global.cg request class and measured
    CTA thread depths; an explicit TMA transport is outside that domain.
    A finite read stream does not reach the published DRAM read capacity: the
    achieved fraction rises with the bytes each CTA streams.  Pricing a
    weight-streaming launch at full capacity therefore predicts a shorter run
    than the memory system can deliver.  The measured surface is an upper
    envelope on that fraction, so compulsory read bytes divided by capacity
    times the fraction is a lower bound on completion and composes by maximum.

    The lookup takes the measured grid and per-CTA length at or above the
    requested ones, which keeps the fraction an upper envelope rather than
    interpolating below the published points.
    """

    profile = hardware.dram_finite_read_stream_ramp
    if "operand_transport_tma" in kernel.launch.parameters:
        if profile is not None:
            inp = dataclasses.replace(inp, limitations=inp.limitations + (
                "dram_finite_read_ramp_ldg_class_does_not_cover_tma_transport",))
        return _attach_tma_quiescent_read_ramp_floor(inp, kernel, hardware)
    if profile is None:
        return inp
    if hardware.kernel_memory_model != "gpu_mshr":
        return inp
    if kernel.variant not in ("gemm", "batched_gemm", "moe_router"):
        return inp
    # The retained ramp is a fixed-request-depth ld.global.cg stream.
    # TMA bypasses that LSU issue topology and has a separate finite operand
    # ring/startup contract; it cannot inherit this completion lower bound.
    # A v2 surface keys on the launch's request depth as well; v1 pins one
    # depth and keeps its own single surface.
    if profile["ramp_by_threads"] is None:
        if kernel.launch.block_size != profile["threads_per_cta"]:
            return dataclasses.replace(inp, limitations=inp.limitations + (
                "dram_finite_read_ramp_outside_measured_thread_domain",))
        ramp_table = profile["ramp"]
        grids_axis = profile["grids"]
        lengths_axis = profile["lengths_bytes_per_cta"]
    else:
        block_x, block_y, block_z = kernel.launch.block
        threads = block_x * block_y * block_z
        if threads not in profile["ramp_by_threads"]:
            return inp
        depth = profile["ramp_by_threads"][threads]
        ramp_table = depth["ramp"]
        grids_axis = depth["grids"]
        lengths_axis = depth["lengths_bytes_per_cta"]
    grid = kernel.launch.grid_size
    if not grids_axis[0] <= grid <= grids_axis[-1]:
        return inp
    if inp.active_units != min(grid, hardware.num_compute_units):
        return inp
    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1:
        return inp
    nodes = {node.resource_role: node for node in read_paths[0].nodes}
    if "dram" not in nodes:
        return inp
    dram_bytes = float(nodes["dram"].bytes)
    if dram_bytes <= 0.0:
        return inp
    # Each CTA streams its own slice of the operand that misses L2.
    per_cta_bytes = dram_bytes / grid
    if per_cta_bytes < lengths_axis[0]:
        return inp
    grids = [value for value in grids_axis if value >= grid]
    lengths = [value for value in lengths_axis if value >= per_cta_bytes]
    if not lengths:
        return inp
    key = (grids[0], lengths[0])
    if key not in ramp_table:
        return inp
    achieved = (
        profile["dram_read_capacity_bytes_per_s"] * ramp_table[key])
    completion_floor_ns = 1e9 * dram_bytes / achieved
    return dataclasses.replace(
        inp,
        execution_time_floor_ns=max(
            inp.execution_time_floor_ns or 0.0, completion_floor_ns),
    )


def _attach_finite_async_operand_stream_completion_floor(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Compose a copy-only finite-stream completion lower bound by max.

    The stream probe contains neither Tensor work nor output storage.  Its
    completion therefore competes with the common pipeline; adding it to a
    memory resource would charge the same operand movement a second time.
    """

    profile = _finite_async_operand_stream_profile(kernel, hardware)
    if profile is None:
        return inp
    _, _, tile_k, _ = _tile(kernel)
    k = int(kernel.shape["k"])
    tile_m, tile_n, _, _ = _tile(kernel)
    m_total = int(kernel.shape["b"]) * int(kernel.shape["m"])
    # The measured intercept is the finite stream's launch, ring fill and
    # drain for this CTA/ring topology, and the ring's round schedule is set
    # by the K tiles alone.  A launch that covers exactly one partial M tile
    # issues the same predicated LDGSTS ring; only its A-panel bytes shrink,
    # and those bytes are priced by the per-tier byte services.  Partial N or
    # multi-tile partial M would change the CTA population itself and stay
    # outside the measured topology.
    if k % tile_k or int(kernel.shape["n"]) % tile_n:
        return inp
    if m_total % tile_m and m_total > tile_m:
        return inp
    rounds = k // tile_k
    if not profile["minimum_rounds"] <= rounds <= profile["maximum_rounds"]:
        return inp
    if inp.active_units != profile["grid_ctas"]:
        return inp
    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1:
        raise ValueError(
            "finite async operand stream requires one global-read path")
    nodes = {node.resource_role: node for node in read_paths[0].nodes}
    if set(nodes) != {"l2_cache", "dram"}:
        raise ValueError(
            "finite async operand stream requires L2 and DRAM read nodes")
    if nodes["dram"].bytes == 0.0:
        completion_floor_ns = (
            profile["fixed_completion_ns"]
            + rounds * profile["l2_resident_ns_per_round"])
    elif nodes["dram"].bytes == nodes["l2_cache"].bytes:
        completion_floor_ns = (
            profile["fixed_completion_ns"]
            + rounds * profile["dram_unique_ns_per_round"])
    elif _uses_finite_async_operand_stream_access_service(kernel, hardware):
        # The probe's fixed completion is the intercept the two cache tiers
        # share: it is the finite stream's launch, ring fill and drain, which
        # depend on the CTA/ring topology and not on where a panel is served.
        # Only the steady per-round service differs by tier.  A mixed stream
        # therefore keeps that intercept and takes its steady time from the
        # two independently measured per-tier byte services already bound to
        # this ingress, instead of a synthesized weighted per-round latency
        # or a duration residual.
        steady_capacity_by_service = {
            service_name: hardware.memory_access_class_service(
                service_name, "read", "l2_cache",
                _finite_async_operand_stream_signature(
                    kernel, service_name)).logical_byte_capacity_per_s
            for service_name in (
                _FINITE_ASYNC_OPERAND_L2_HIT_SERVICE,
                _FINITE_ASYNC_OPERAND_DRAM_MISS_SERVICE)
        }
        l2_hit_bytes = nodes["l2_cache"].bytes - nodes["dram"].bytes
        # The two tiers are independently measured services of two distinct
        # physical levels, and the formula already prices them as separate
        # resource groups.  The steady part of a mixed stream is therefore the
        # binding tier, not the sum of both: composing them additively would
        # assert a shared serial ingress that no mechanism evidence
        # establishes.
        steady_s = max(
            l2_hit_bytes
            / steady_capacity_by_service[
                _FINITE_ASYNC_OPERAND_L2_HIT_SERVICE],
            nodes["dram"].bytes
            / steady_capacity_by_service[
                _FINITE_ASYNC_OPERAND_DRAM_MISS_SERVICE])
        completion_floor_ns = (
            profile["fixed_completion_ns"] + 1e9 * steady_s)
    else:
        # Without the typed per-tier ingress services a mixed stream has no
        # independent steady evidence; do not synthesize one.
        return inp
    return dataclasses.replace(
        inp,
        execution_time_floor_ns=max(
            inp.execution_time_floor_ns or 0.0, completion_floor_ns),
    )


def _attach_finite_mixed_operand_stream_startup(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Add one independently measured operand-transport fill per launch.

    Typed profiles preserve the legacy LDGSTS domain and the TMA ring's
    aggregate transport domain. Byte work, first-request latency and steady
    service remain in the common pipeline; a TMA profile subtracts the
    existing first-request latency before publishing its additional fill.
    Completion-floor profiles already include their finite fill and cannot
    receive this term a second time.
    """

    profiles = hardware.finite_mixed_operand_stream_startup
    if profiles is None:
        return inp
    transport = "ldgsts"
    if "operand_transport_tma" in kernel.launch.parameters:
        if kernel.launch.parameters["operand_transport_tma"] != 1.0:
            raise ValueError("operand_transport_tma must declare a TMA source")
        transport = "tma"
    candidates = tuple(profile for profile in profiles
        if profile["transport"] == transport
        and profile["threads_per_cta"] == kernel.launch.block_size)
    if not candidates:
        return inp
    if len(candidates) != 1:
        raise ValueError("finite operand stream transport profile is ambiguous")
    profile, = candidates
    if inp.active_units < profile["minimum_grid_ctas"]:
        return inp
    if transport == "tma" and inp.active_units > profile["one_wave_grid_ctas"]:
        return inp
    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or kernel.variant not in ("gemm", "batched_gemm", "moe_router")
            or kernel.launch.block_size != profile["threads_per_cta"]
            or tiling is None
            or tiling.split_k is None
            or tiling.pipeline_depth is None
            or not (profile["minimum_pipeline_depth"]
                    <= tiling.pipeline_depth
                    <= profile["maximum_pipeline_depth"])):
        return inp
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
    if (a_element_bytes != profile["dtype_element_bytes"]
            or b_element_bytes != profile["dtype_element_bytes"]):
        return inp
    _, _, tile_k, _ = _tile(kernel)
    if tile_k <= 0 or tile_k != profile["tile_k"]:
        return inp
    k = int(kernel.shape["k"])
    if k % tile_k:
        return inp
    # Each split-K slice streams its own K range, so the finite stream of
    # one CTA covers k / split_k tiles.
    rounds = k // (tile_k * tiling.split_k)
    if not profile["minimum_rounds"] <= rounds <= profile["maximum_rounds"]:
        return inp
    if (transport == "tma" and math.ceil(k / (tile_k * tiling.split_k))
            > profile["maximum_rounds"]):
        return inp
    if _finite_async_operand_stream_profile(kernel, hardware) is not None:
        return inp
    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1:
        raise ValueError(
            "finite mixed operand-stream startup requires one global-read "
            "path")
    nodes = {node.resource_role: node for node in read_paths[0].nodes}
    if set(nodes) != {"l2_cache", "dram"}:
        raise ValueError(
            "finite mixed operand-stream startup requires L2 and DRAM read "
            "nodes")
    if not 0.0 < nodes["dram"].bytes < nodes["l2_cache"].bytes:
        return inp
    if inp.finite_mixed_operand_stream_startup_ns is not None:
        raise ValueError(
            "finite mixed operand-stream startup attached twice")
    return dataclasses.replace(
        inp,
        finite_mixed_operand_stream_startup_ns=(
            profile["mixed_tier_startup_ns"]),
        limitations=inp.limitations + ((
            "tma_operand_ring_startup_aggregate_transport_approximation",
        ) if transport == "tma" else ()))


def _attach_k_tiled_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        *, compute_role: str) -> PredictionInput:
    """Attach the source-class K loop with one terminal output store per CTA."""

    if not _uses_k_tiled_pipeline(kernel):
        raise ValueError(
            "K-tiled GEMM pipeline requires a physical non-split staged "
            "launch")
    tile_m, tile_n, tile_k, pipeline_depth = _tile(kernel)
    if pipeline_depth is None:
        raise AssertionError("staged K-tiled GEMM pipeline has no depth")
    k = int(kernel.shape["k"])
    if k % tile_k:
        raise ValueError(
            "K-tiled GEMM pipeline requires K divisible by tile_k")
    total_k_tiles = k // tile_k
    if total_k_tiles <= 0:
        raise ValueError("K-tiled GEMM pipeline requires a positive K loop")
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k is None:
        raise AssertionError("K-tiled GEMM pipeline has no split-K contract")
    split_k = tiling.split_k
    if split_k > total_k_tiles:
        raise ValueError("K-tiled GEMM cannot assign an empty K slice")

    # PredictionInput.grid_size is the logical workgroup count derived from
    # shape, retained tile, and split-K.  Raw launch grid/block observations
    # are diagnostic metadata and must not alter GEMM Forward.
    cta_count = inp.grid_size
    if cta_count % split_k:
        raise ValueError("K-tiled GEMM grid does not divide by split-K")
    output_tile_count = cta_count // split_k
    if "uniform_split_k_tiles" in kernel.launch.parameters:
        # This source fact says every slice executes the same rounded loop.
        # Padded Tensor/SMEM work was already added to the resource demand;
        # conserve it over the issued stages rather than balanced K slices.
        k_tiles_by_split = (math.ceil(total_k_tiles / split_k),) * split_k
    else:
        base_k_tiles, extra_k_tiles = divmod(total_k_tiles, split_k)
        k_tiles_by_split = tuple(
            base_k_tiles + int(split_index < extra_k_tiles)
            for split_index in range(split_k))
    worker_count = min(cta_count, inp.active_units)
    k_tile_fraction = 1.0 / (output_tile_count * sum(k_tiles_by_split))
    cta_store_fraction = 1.0 / cta_count
    split1_issue_operations = (
        _sm80_depth5_split1_non_hmma_warp_instructions(kernel, hardware))
    if split1_issue_operations > 0.0:
        split1_loop_issue_fraction = (
            _SM80_128X128_K32_DEPTH5_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE
            / split1_issue_operations)
        split1_fixed_issue_fraction = (
            _SM80_128X128_K32_DEPTH5_SPLIT1_FIXED_NON_HMMA_WARP_INSTRUCTIONS
            / split1_issue_operations)
        if not math.isclose(
                cta_count * (
                    total_k_tiles
                    * _SM80_128X128_K32_DEPTH5_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE
                    + _SM80_128X128_K32_DEPTH5_SPLIT1_FIXED_NON_HMMA_WARP_INSTRUCTIONS),
                split1_issue_operations):
            raise ValueError(
                "SM80 depth-five split-one issue demand is not conserved")
    else:
        split1_loop_issue_fraction = 0.0
        split1_fixed_issue_fraction = 0.0
    source_fixed_issue = (
        _sm80_depth5_fixed_non_hmma_warp_instructions(kernel, hardware)
        + _sm80_depth6_fixed_non_hmma_warp_instructions(kernel, hardware)
        > 0.0)
    asynchronous_operand_completion = (
        "operand_transport_tma" in kernel.launch.parameters)

    def k_tile_task(
            name: str, *, first: bool,
            terminal: bool, repeat_count: int = 1) -> PipelineTaskDemand:
        # The retained staged program keeps one producer ring for the A/B
        # operand group.  A logical M tail predicates useful A bytes, but does
        # not turn that ring into a launch-wide serial A-then-B phase.
        stages = [PipelineStageDemand(
            "load", "memory:global_read", k_tile_fraction,
            first or asynchronous_operand_completion,
            pipelined_memory_completion=asynchronous_operand_completion)]
        stages.append(
            # SM80 staged GEMM consumes each LDGSTS operand stage through
            # source-visible LDSM instructions before the corresponding HMMA
            # group.  Keep that existing shared-memory demand in the finite
            # tile schedule instead of allowing it to overlap the whole
            # launch as an unrelated roofline resource.
            PipelineStageDemand(
                "operand_smem", "resource:smem", k_tile_fraction, False))
        if split1_loop_issue_fraction > 0.0:
            stages.append(PipelineStageDemand(
                "source_loop_issue", "compute:warp_issue",
                split1_loop_issue_fraction, False))
        stages.append(
            PipelineStageDemand(
                "compute", f"compute:{compute_role}",
                k_tile_fraction, False))
        if terminal:
            if split1_fixed_issue_fraction > 0.0:
                stages.append(PipelineStageDemand(
                    "source_fixed_issue", "compute:warp_issue",
                    split1_fixed_issue_fraction, False))
            elif source_fixed_issue:
                stages.append(PipelineStageDemand(
                    "source_fixed_issue", "compute:warp_issue",
                    cta_store_fraction, False))
            stages.append(PipelineStageDemand(
                "store", "memory:global_write",
                cta_store_fraction, True))
        return PipelineTaskDemand(
            name=name,
            worker=0,
            stages=tuple(stages),
            repeat_count=repeat_count,
        )

    worker_programs = []
    task_templates_by_k_tiles = {}

    def task_templates(k_tiles: int):
        """Build each immutable K-loop shape once before worker binding."""

        if k_tiles in task_templates_by_k_tiles:
            return task_templates_by_k_tiles[k_tiles]
        if k_tiles == 1:
            tasks = (k_tile_task(
                "only_k_tile", first=True, terminal=True),)
        else:
            tasks = [k_tile_task(
                "first_k_tile", first=True, terminal=False)]
            if k_tiles > 2:
                tasks.append(k_tile_task(
                    "middle_k_tiles", first=False, terminal=False,
                    repeat_count=k_tiles - 2))
            tasks.append(k_tile_task(
                "final_k_tile", first=False, terminal=True))
            tasks = tuple(tasks)
        task_templates_by_k_tiles[k_tiles] = tasks
        return tasks

    def cta_sequence(
            worker: int, k_tiles: int, repeat_count: int,
            name_suffix: str) -> PipelineTaskSequenceDemand:
        # Pipeline depth is the physical producer-ring capacity.  The buffer
        # recurrence below carries that finite state across every K tile and
        # repeated CTA; it therefore needs no tile-family refill rule or
        # manually accumulated completion latency.
        # Every source task template has already been validated and only its
        # worker binding differs here.  Reconstructing it with
        # ``dataclasses.replace`` repeats validation of the immutable stage
        # program for every CTA run.  The worker comes directly from the
        # validated ``range(worker_count)`` above, so retain the verified
        # fields and change just that binding.
        tasks = tuple(
            _rebind_pipeline_task_worker(task, worker)
            for task in task_templates(k_tiles))
        return PipelineTaskSequenceDemand(
            name=f"worker{worker}_{name_suffix}_cta_program",
            worker=worker,
            tasks=tasks,
            repeat_count=repeat_count,
        )

    worker_classes = {}
    for worker in range(worker_count):
        cta_indexes = tuple(range(worker, cta_count, worker_count))
        if not cta_indexes:
            raise ValueError("K-tiled GEMM has an empty active worker")
        runs = []
        for cta_index in cta_indexes:
            split_index = cta_index // output_tile_count
            k_tiles = k_tiles_by_split[split_index]
            if runs and runs[-1][0] == k_tiles:
                runs[-1] = (k_tiles, runs[-1][1] + 1)
            else:
                runs.append((k_tiles, 1))
        signature = tuple(runs)
        if signature in worker_classes:
            worker_classes[signature].append(worker)
        else:
            worker_classes[signature] = [worker]

    # CTA scheduling is round-robin over physical workers.  Workers with the
    # same run sequence therefore have an identical, independent recurrence.
    # Materializing a source sequence for each physical worker only to have
    # ``evaluate_pipeline`` discover this equivalence later repeats task
    # binding, contract resolution, and program construction.  Preserve the
    # complete physical mapping directly in the compact source program so the
    # resolved formula still conserves every worker's service and diagnostics.
    for class_index, (runs, workers) in enumerate(worker_classes.items()):
        physical_workers = tuple(workers)
        representative = physical_workers[0]
        patterns = tuple(
            cta_sequence(
                representative, k_tiles, repeat_count, f"run{run_index}")
            for run_index, (k_tiles, repeat_count) in enumerate(runs))
        worker_programs.append(PipelineTaskProgramDemand(
            name=f"k_tiled_cta_worker_class{class_index}",
            worker=representative,
            patterns=patterns,
            order=tuple(range(len(patterns))),
            worker_multiplicity=len(physical_workers),
            physical_workers=physical_workers,
        ))

    operand_buffer_slots = _gpu_k_tiled_operand_buffer_slots(kernel, hardware)
    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tuple(worker_programs),
        buffers=(PipelineBuffer(
            name="load_compute_buffer",
            producer_stage="load",
            consumer_stage="compute",
            slots=operand_buffer_slots,
            scope="worker",
        ),),
    )
    return attach_pipeline_work_plan(inp, hardware, plan)


def _rebind_pipeline_task_worker(
        task: PipelineTaskDemand, worker: int) -> PipelineTaskDemand:
    """Bind one internally validated immutable task template to a worker."""

    if (isinstance(worker, bool) or not isinstance(worker, int)
            or worker < 0):
        raise ValueError("pipeline task worker must be a non-negative integer")
    rebound = object.__new__(PipelineTaskDemand)
    object.__setattr__(rebound, "name", task.name)
    object.__setattr__(rebound, "worker", worker)
    object.__setattr__(rebound, "stages", task.stages)
    object.__setattr__(rebound, "repeat_count", task.repeat_count)
    return rebound


def _sm80_depth5_split2_non_hmma_warp_instructions(
        kernel: KernelSpec, hardware) -> float:
    """Return source-counted non-HMMA issue demand for one cuBLASLt program.

    The pinned ``128x128_ldg8_f2f_stages_32x5_tn`` SASS has four warps per
    CTA.  Its loop at 0x17b0--0x1f30 contains 121 warp instructions per K32
    tile, of which 64 are HMMA instructions.  Across four warps this leaves
    228 non-HMMA issues per tile.  The split-two control-flow path contributes
    another 5,446 non-HMMA warp instructions per CTA.  This exact affine
    count also closes against the canonical non-duration ``inst_executed``
    counters for every full-tile split-two launch; duration is not consumed.

    HMMA instructions stay owned by ``compute_tensor``.  Returning only the
    remaining issue demand prevents the source-order dependency below from
    charging Tensor instructions twice.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or "warp_issue" not in hardware._role_map
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (128, 1, 1)
            or tiling is None
            or tiling.split_k != 2
            or tiling.pipeline_depth != 5
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            != (128, 128, 32)
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 32:
        raise ValueError(
            "SM80 depth-five SASS issue schedule requires complete K32 tiles")
    output_tiles = gemm_logical_workgroups(b, m, n, 128, 128, 1)
    cta_count = output_tiles * tiling.split_k
    if (kernel.launch.grid_size != cta_count
            or kernel.launch.grid[2] != tiling.split_k):
        return 0.0
    return float(
        cta_count
        * _SM80_128X128_K32_DEPTH5_SPLIT2_FIXED_NON_HMMA_WARP_INSTRUCTIONS
        + output_tiles * (k // 32)
        * _SM80_128X128_K32_DEPTH5_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE)


def _sm80_depth5_split1_non_hmma_warp_instructions(
        kernel: KernelSpec, hardware) -> float:
    """Return source-counted issue demand for the non-split K32 program.

    The non-split path through the pinned
    ``128x128_ldg8_f2f_stages_32x5_tn`` SASS retains the same 228 non-HMMA
    warp instructions per K32 tile as the split-two path.  Its source control
    flow contributes 4,624 fixed non-HMMA warp instructions per CTA.  For the
    canonical 48-tile launch this gives 15,568 non-HMMA plus 12,288 HMMA warp
    instructions per CTA, closing exactly against the non-duration
    ``inst_executed`` count of 27,856.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or "warp_issue" not in hardware._role_map
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (128, 1, 1)
            or tiling is None
            or tiling.split_k != 1
            or tiling.pipeline_depth != 5
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            != (128, 128, 32)
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 32:
        raise ValueError(
            "SM80 depth-five SASS issue schedule requires complete K32 tiles")
    output_tiles = gemm_logical_workgroups(b, m, n, 128, 128, 1)
    if (kernel.launch.grid_size != output_tiles
            or kernel.launch.grid[2] != 1):
        return 0.0
    return float(
        output_tiles
        * _SM80_128X128_K32_DEPTH5_SPLIT1_FIXED_NON_HMMA_WARP_INSTRUCTIONS
        + output_tiles * (k // 32)
        * _SM80_128X128_K32_DEPTH5_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE)


def _sm80_depth3_split_epilogue_warp_instructions(
        kernel: KernelSpec, hardware) -> float:
    """Return source-counted ordered epilogue issue for SM80 depth-three programs.

    Both transposed CUDA 12.2 symbols have eight warps per CTA.  At
    0x3070--0x3090 every later K slice polls the per-output semaphore before
    entering the accumulator publication at 0x30a0--0x3d90.  That ordered
    suffix executes 1,032 instructions per warp: 208 inline instructions and
    eight calls to the 103-instruction 0x3da0--0x4400 store routine.  The
    first slice branches around fourteen prior-output load/add instructions,
    saving 112 warp instructions across its eight warps.  The separate
    split-two K64 loop source count is owned by
    ``_sm80_depth3_split2_non_hmma_warp_instructions``.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or "warp_issue" not in hardware._role_map
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (256, 1, 1)
            or tiling is None
            or tiling.split_k is None
            or tiling.split_k <= 1
            or tiling.pipeline_depth != 3
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            not in {(128, 256, 64), (256, 128, 64)}
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 64:
        raise ValueError(
            "SM80 depth-three SASS issue schedule requires complete K64 "
            "tiles")
    tile_m, tile_n, _ = (
        tiling.tile_shape[name] for name in ("m", "n", "k"))
    output_tiles = gemm_logical_workgroups(b, m, n, tile_m, tile_n, 1)
    cta_count = output_tiles * tiling.split_k
    if (kernel.launch.grid_size != cta_count
            or kernel.launch.grid[2] != tiling.split_k):
        return 0.0
    fixed_instructions_per_output = (
        _SM80_128X256_K64_DEPTH3_ORDERED_EPILOGUE_WARP_INSTRUCTIONS
        * tiling.split_k
        - _SM80_128X256_K64_DEPTH3_FIRST_EPILOGUE_SAVING_WARP_INSTRUCTIONS)
    return float(output_tiles * fixed_instructions_per_output)


def _sm80_depth3_split2_non_hmma_warp_instructions(
        kernel: KernelSpec, hardware) -> float:
    """Return CUDA 12.2 split-two K64 non-HMMA source issue work.

    The two bound CUDA 12.2 transposed symbols have identical 214-instruction
    K64 warp streams at ``0x1a20--0x2770``: 128 HMMA instructions remain
    Tensor work and the other 86 become 688 non-HMMA warp instructions per
    CTA.  The ``256x128`` named symbol's canonical non-duration instruction
    counters also close its distinct fixed path to 4,744 non-HMMA warp
    instructions per logical output's two split CTAs.  The transfer to the
    physical ``256x128x64`` tile deliberately includes only the static-equal
    K loop; it has no corresponding fixed-path counter.  No production
    duration participates in either source count.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or "warp_issue" not in hardware._role_map
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (256, 1, 1)
            or tiling is None or tiling.split_k != 2
            or tiling.pipeline_depth != 3
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            not in {(128, 256, 64), (256, 128, 64)}
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 64:
        raise ValueError(
            "SM80 depth-three split-two source issue requires complete K64 "
            "tiles")
    tile_m, tile_n, _ = (
        tiling.tile_shape[name] for name in ("m", "n", "k"))
    output_tiles = gemm_logical_workgroups(b, m, n, tile_m, tile_n, 1)
    if (kernel.launch.grid_size != output_tiles * tiling.split_k
            or kernel.launch.grid[2] != tiling.split_k):
        return 0.0
    fixed_instructions_per_output = (
        _SM80_128X256_K64_DEPTH3_SPLIT2_FIXED_NON_HMMA_WARP_INSTRUCTIONS
        if (tile_m, tile_n) == (128, 256) else 0)
    return float(output_tiles * (
        fixed_instructions_per_output
        + (k // 64)
        * _SM80_128X256_K64_DEPTH3_SPLIT2_NON_HMMA_WARP_INSTRUCTIONS_PER_K_TILE))


def _sm80_depth3_source_barrier_cycles(
        kernel: KernelSpec, hardware) -> float:
    """Return the source-visible K64 CTA barriers for the pinned SM80 tile.

    CUDA 12.2's ``256x128`` and ``128x256`` named symbols normalize to the
    two physical ``128x256x64``/``256x128x64`` macro tiles. Both of their
    0x1a20--0x2770 K64 loops reach one ``BAR.SYNC.DEFER_BLOCKING`` at 0x2380
    before the following operand/Tensor round. The independent 256-thread,
    one-CTA/SM probe provides that barrier service in SM cycles; this source
    count deliberately does not consume a production duration or a profiler
    duty value.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (256, 1, 1)
            or tiling is None or tiling.split_k is None
            or tiling.split_k < 1 or tiling.pipeline_depth != 3
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            not in {(128, 256, 64), (256, 128, 64)}
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    mechanism = hardware.sm80_depth3_cta_barrier_service_cycles
    if mechanism is None:
        raise ValueError(
            "SM80 depth-three source requires CTA-barrier service evidence")
    if (mechanism["threads_per_cta"] != 256
            or mechanism["total_shared_bytes"] != 147456
            or mechanism["resident_ctas_per_unit"] != 1):
        raise ValueError(
            "SM80 depth-three CTA barrier service does not match source "
            "residency")
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 64:
        raise ValueError(
            "SM80 depth-three barrier source requires complete K64 tiles")
    tile_m, tile_n, _ = (
        tiling.tile_shape[name] for name in ("m", "n", "k"))
    output_tiles = gemm_logical_workgroups(b, m, n, tile_m, tile_n, 1)
    cta_count = output_tiles * tiling.split_k
    if (kernel.launch.grid_size != cta_count
            or kernel.launch.grid[2] != tiling.split_k):
        return 0.0
    # A non-split CTA executes every K64 tile itself. Split-K distributes the
    # same complete K64 tiles across publications. Both therefore execute one
    # source-visible barrier for every logical output/K64 pair exactly once.
    return float(
        output_tiles * (k // 64) * mechanism["barrier_cycles"])


def _sm80_depth6_fixed_non_hmma_warp_instructions(
        kernel: KernelSpec, hardware) -> float:
    """Return fixed issue outside the 64x128x64 depth-six K loop.

    In the pinned ``128x64_sliced1x2_ldg8_f2f_stages_64x6_tn`` SASS, the
    0x1a40--0x2290 loop contributes 280 non-HMMA warp instructions per K64
    tile across four warps.  The source-counted prologue and epilogue contain
    another 4,880 non-HMMA warp instructions per CTA.  The latter cannot
    overlap the K-loop Tensor stream, so it is placed once in the terminal
    task; canonical non-duration counters close exactly at both 128 and 448
    K64 tiles.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or "warp_issue" not in hardware._role_map
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (128, 1, 1)
            or tiling is None
            or tiling.split_k != 1
            or tiling.pipeline_depth != 6
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            != (64, 128, 64)
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 64:
        raise ValueError(
            "SM80 depth-six SASS issue schedule requires complete K64 tiles")
    output_tiles = gemm_logical_workgroups(b, m, n, 64, 128, 1)
    if (kernel.launch.grid_size != output_tiles
            or kernel.launch.grid[2] != 1):
        return 0.0
    return float(
        output_tiles
        * _SM80_64X128_K64_DEPTH6_FIXED_NON_HMMA_WARP_INSTRUCTIONS)


def _sm80_depth5_fixed_non_hmma_warp_instructions(
        kernel: KernelSpec, hardware) -> float:
    """Return fixed issue outside the 64x64x64 depth-five K loop.

    In the pinned ``64x64_sliced1x2_ldg8_f2f_stages_64x5_tn`` SASS, the
    0x1500--0x1a40 loop contributes 212 non-HMMA warp instructions per K64
    tile across four warps.  The source-counted prologue and epilogue path
    contributes another 3,076 non-HMMA warp instructions per CTA.  The
    latter is ordered after the Tensor loop and is therefore attached to the
    terminal task.  The canonical non-duration ``inst_executed`` counter for
    the 64-tile path closes exactly at 24,836 instructions per CTA.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or "warp_issue" not in hardware._role_map
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (128, 1, 1)
            or tiling is None
            or tiling.split_k != 1
            or tiling.pipeline_depth != 5
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            != (64, 64, 64)
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return 0.0
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if k % 64:
        raise ValueError(
            "SM80 depth-five 64x64 SASS issue schedule requires complete "
            "K64 tiles")
    output_tiles = gemm_logical_workgroups(b, m, n, 64, 64, 1)
    if (kernel.launch.grid_size != output_tiles
            or kernel.launch.grid[2] != 1):
        return 0.0
    return float(
        output_tiles
        * _SM80_64X64_K64_DEPTH5_FIXED_NON_HMMA_WARP_INSTRUCTIONS)


def _apply_sm80_depth3_split_epilogue_issue_schedule(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Limit ordered epilogue issue to independent output-tile chains."""

    issue_operations = _sm80_depth3_split_epilogue_warp_instructions(
        kernel, hardware)
    if issue_operations == 0.0:
        return inp
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k is None:
        raise AssertionError("SM80 split epilogue lost retained split count")
    output_tiles, remainder = divmod(kernel.launch.grid_size, tiling.split_k)
    if remainder or output_tiles <= 0:
        raise ValueError("SM80 split epilogue has invalid output concurrency")
    active_output_chains = min(output_tiles, inp.active_units)
    issue = next(
        demand for demand in inp.compute_demands
        if demand.resource_role == "warp_issue")
    adjusted = dataclasses.replace(
        issue,
        schedule_efficiency=(
            issue.schedule_efficiency
            * active_output_chains / inp.active_units),
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            adjusted if demand is issue else demand
            for demand in inp.compute_demands),
    )


def _apply_sm80_depth6_fixed_issue_schedule(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Price physical fixed instructions without semantic-M utilization."""

    issue_operations = _sm80_depth6_fixed_non_hmma_warp_instructions(
        kernel, hardware)
    if issue_operations == 0.0:
        return inp
    issue = next(
        demand for demand in inp.compute_demands
        if demand.resource_role == "warp_issue")
    adjusted = dataclasses.replace(
        issue,
        schedule_efficiency=inp.wave_efficiency,
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            adjusted if demand is issue else demand
            for demand in inp.compute_demands),
    )


def _apply_sm80_depth5_fixed_issue_schedule(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Price physical fixed instructions without semantic-M utilization."""

    issue_operations = _sm80_depth5_fixed_non_hmma_warp_instructions(
        kernel, hardware)
    if issue_operations == 0.0:
        return inp
    issue = next(
        demand for demand in inp.compute_demands
        if demand.resource_role == "warp_issue")
    adjusted = dataclasses.replace(
        issue,
        schedule_efficiency=inp.wave_efficiency,
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            adjusted if demand is issue else demand
            for demand in inp.compute_demands),
    )


def _sm80_depth5_split2_sass_dependency(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Order non-HMMA issue around the program's Tensor instruction stream."""

    if _sm80_depth5_split2_non_hmma_warp_instructions(
            kernel, hardware) == 0.0:
        return None
    return AnalyticalDependencyPath(
        name="sm80_depth5_split2_sass_issue_tensor",
        services=("compute:warp_issue", "compute:compute_tensor"),
    )


def _sm80_depth3_non_split_sass_dependency(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Order the exact FP16 three-slot operand prologue before terminal store.

    The two retained macro-tile orientations only exchange the 16-KiB and
    32-KiB operand panels. Their static K64 loop and the independent copy-only
    probe have the same 49,152-byte LDGSTS issue stream, three committed
    groups, and one-CTA residency, so both retain the same finite launch-local
    completion sequence.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            # The retained binary and the independent wave-scaling probe are
            # both FP16-only.  BF16 needs its own source-program binding
            # before it can consume this finite-startup contract.
            or kernel.dtype != "fp16"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (256, 1, 1)
            or tiling is None
            or tiling.split_k != 1
            or tiling.pipeline_depth != 3
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            not in {(128, 256, 64), (256, 128, 64)}
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return None
    barrier_cycles = _sm80_depth3_source_barrier_cycles(kernel, hardware)
    # The pinned SM80 binaries issue three committed LDGSTS groups before the
    # first DEPBAR.LE SB0,1. The workload-independent depth-three probe shows
    # 35 additional L2 completion arrivals in their launch-local finite fill;
    # the dependency path's ordinary first-request latency owns the first one.
    return AnalyticalDependencyPath(
        name="sm80_depth3_non_split_load_mma_store",
        services=(
            "launch:submit",
            "memory:global_read",
            "memory_latency:global_read",
            "compute:compute_tensor",
            *(("compute:compute_reduction",) if barrier_cycles > 0.0 else ()),
            "memory:global_write",
            "memory_latency:global_write",
        ),
        concurrent_service_groups=((
            "memory:global_read", "compute:compute_tensor"),),
        serial_service_counts=((
            "memory_latency:global_read",
            1 + _SM80_DEPTH3_ADDITIONAL_L2_ARRIVALS_PER_LAUNCH,
        ),),
    )


def _sm80_depth3_split_sass_dependency(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Order the depth-three split program's published output chain.

    Operand ingress and HMMA advance concurrently through the K loop.  The
    pinned SASS then polls the per-output split semaphore and issues the
    accumulator publication before the terminal store.  The source-counted
    issue demand already contains that ordered suffix; keeping it outside the
    dependency path incorrectly lets a launch with more split publications
    become faster merely because it has fewer logical output tiles.
    """

    if _sm80_depth3_split_epilogue_warp_instructions(
            kernel, hardware) == 0.0:
        return None
    barrier_cycles = _sm80_depth3_source_barrier_cycles(kernel, hardware)
    services = (
        "launch:submit",
        "memory:global_read",
        "memory_latency:global_read",
        "compute:compute_tensor",
        *(("compute:compute_reduction",) if barrier_cycles > 0.0 else ()),
        "compute:warp_issue",
        "memory:global_write",
        "memory_latency:global_write",
    )
    return AnalyticalDependencyPath(
        name="sm80_depth3_split_load_mma_epilogue_store",
        services=services,
        concurrent_service_groups=((
            "memory:global_read", "compute:compute_tensor"),),
    )


def _sm80_depth3_transposed_split_source_dependency(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Order the transposed tile's split epilogue after the source loop.

    The shared operand-source adapter owns the launch-local LDGSTS finite fill
    for both macro-tile orientations. Do not encode that same 35-arrival
    startup here as memory latency: doing so would count a single source
    dependency once through the warp-issue schedule and again through this
    dependency path.
    """

    tiling = kernel.launch.tiling
    if (hardware.kernel_memory_model != "gpu_mshr"
            or hardware.parallel_width != 32
            or kernel.dtype != "fp16"
            or kernel.variant != "gemm"
            or kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or kernel.launch.block != (256, 1, 1)
            or tiling is None or tiling.split_k is None
            or tiling.split_k <= 1 or tiling.pipeline_depth != 3
            or tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            != (256, 128, 64)
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] != 0.0):
        return None
    barrier_cycles = _sm80_depth3_source_barrier_cycles(kernel, hardware)
    return AnalyticalDependencyPath(
        name="sm80_depth3_transposed_split_operand_source",
        services=(
            "launch:submit",
            "memory:global_read",
            "memory_latency:global_read",
            "compute:compute_tensor",
            *(("compute:compute_reduction",) if barrier_cycles > 0.0 else ()),
            "compute:warp_issue",
            "memory:global_write",
            "memory_latency:global_write",
        ),
        concurrent_service_groups=((
            "memory:global_read", "compute:compute_tensor"),),
    )


def _metax_gemv_wave8_completion(
        kernel: KernelSpec, hardware) -> Tuple[float, int] | None:
    """Return the resident eight-output GEMV reduction completion."""

    if (kernel.dtype != "bf16"
            or kernel.launch.algorithm != "metax_mcblas_gemv"):
        return None
    block_threads = kernel.launch.block[0]
    if block_threads == 64:
        service = hardware.metax_gemv_wave8_completion_service
    elif block_threads == 256:
        service = hardware.metax_gemv_block256_wave8_completion_service
    else:
        return None
    if service is None:
        return None
    if (kernel.launch.block != (service["block_threads"], 1, 1)
            or kernel.launch.grid != (kernel.launch.grid_size, 1, 1)):
        return None
    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    outputs_per_cta, remainder = divmod(b * m * n, kernel.launch.grid_size)
    if remainder or outputs_per_cta != service["outputs_per_cta"]:
        return None
    if not (service["minimum_grid_size"] <= kernel.launch.grid_size
            <= service["maximum_resident_grid_size"]):
        return None
    active_aps = min(kernel.launch.grid_size, service["ap_count"])
    base_ctas, extra_ctas = divmod(kernel.launch.grid_size, active_aps)
    ctas_by_ap = tuple(
        base_ctas + (1 if ap < extra_ctas else 0)
        for ap in range(active_aps))
    cohorts_by_ap = tuple(
        math.ceil(ctas / service["wave_completion_slots_per_ap"])
        for ctas in ctas_by_ap)
    completion_ns = (
        service["first_wave_cohort_completion_ns_per_ap"]
        + (max(cohorts_by_ap) - 1)
        * service["subsequent_wave_cohort_completion_ns_per_ap"])
    return completion_ns, sum(cohorts_by_ap)


def _uses_a800_analytical_gemv_stream(
        kernel: KernelSpec, hardware) -> bool:
    """Return whether an analytical A800 GEMV exposes a steady K stream."""

    b, m, n = (int(kernel.shape[name]) for name in ("b", "m", "n"))
    return (
        kernel.variant == "gemv"
        and kernel.launch.algorithm == "cublas_analytical_gemv"
        and kernel.launch.grid == (b * m * n, 1, 1)
        and kernel.launch.block == (hardware.parallel_width, 1, 1)
    )


def _gpu_gemv_dependency_pipeline(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Return the compact Load/Reduce/Store path of a 1-D GEMV.

    The analytical A800 launch describes a steady K stream rather than a
    source-exact CTA program.  Keep its complete read/FMA work in the
    recurrent services below; pipeline fill, first-request latency, launch,
    and the terminal store then remain fixed as K grows.
    """

    if (not _uses_exact_gemv_launch(kernel, hardware)
            and not _uses_a800_analytical_gemv_stream(kernel, hardware)
            and not _is_simt_router(kernel)):
        return None
    services = [
        "launch:submit",
        "memory:global_read",
        "memory_latency:global_read",
        "compute:compute_fma",
    ]
    if _metax_gemv_wave8_completion(kernel, hardware) is not None:
        services.append("compute:compute_reduction")
    services.extend((
        "memory:global_write",
        "memory_latency:global_write",
    ))
    return AnalyticalDependencyPath(
        name="gpu_gemv_load_reduce_store",
        services=tuple(services),
        concurrent_service_groups=((
            "memory:global_read", "compute:compute_fma"),),
        serial_service_counts=(),
    )


def _gpu_gemv_memory_source(
        kernel: KernelSpec, hardware, read_l2_bytes: float,
        active_units: int) -> Tuple[float, float | None, float | None]:
    """Return the source-bound scalar-half stream contract when available."""

    generic_producers, generic_requests = gpu_memory_source(hardware)
    fields = (
        hardware.gemv_scalar_half_issue_group_elements_per_lane,
        hardware.gemv_scalar_half_resident_ctas_per_sm,
        hardware.gemv_scalar_half_dram_transactions_per_sm_s,
        hardware.gemv_scalar_half_source_fill_ns,
    )
    if any(value is None for value in fields):
        if not all(value is None for value in fields):
            raise ValueError("incomplete GEMV scalar-half DRAM source contract")
        return generic_producers, generic_requests, None
    block_x, block_y, block_z = kernel.launch.block
    b, m, n = (
        int(kernel.shape[name]) for name in ("b", "m", "n"))
    if not (
            kernel.dtype == "fp16"
            and block_x == hardware.parallel_width
            and block_y == 4
            and block_z == 1
            and kernel.launch.grid_size * block_y == b * m * n):
        return generic_producers, generic_requests, None
    if active_units <= 0 or read_l2_bytes <= 0:
        raise ValueError("GEMV scalar-half source requires positive work")
    transaction_bytes = float(hardware.l2_transaction_bytes)
    transactions_per_unit = (
        read_l2_bytes / transaction_bytes / active_units)
    steady_rate = float(
        hardware.gemv_scalar_half_dram_transactions_per_sm_s)
    source_fill_s = float(hardware.gemv_scalar_half_source_fill_ns) * 1e-9
    effective_rate = transactions_per_unit / (
        source_fill_s + transactions_per_unit / steady_rate)
    producers = float(
        block_y * hardware.gemv_scalar_half_resident_ctas_per_sm)
    requests_per_producer = float(
        2 * hardware.gemv_scalar_half_issue_group_elements_per_lane)
    return producers, requests_per_producer, effective_rate


def _gpu_external_split_k_cohort_pipeline(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Return a compact cohort path for exact external split-K tiles."""

    tiling = kernel.launch.tiling
    if (tiling is None or tiling.split_k is None or tiling.split_k <= 1
            or kernel.launch.parameters[
                "split_k_partial_bytes_per_element"] <= 0.0):
        return None
    # Staged external split-K uses the same finite Load/SMEM/MMA/Store tile
    # program as every other staged cuBLASLt GEMM.  Only an unstaged launch
    # needs this compact aggregate dependency path.
    if tiling.pipeline_depth is not None:
        return None
    b, m, n = (
        int(kernel.shape[name]) for name in ("b", "m", "n"))
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    logical_tiles = _kernel_logical_workgroups(
        kernel, tile_m, tile_n, tiling.split_k)
    block_x, block_y, block_z = kernel.launch.block
    if kernel.launch.grid_size != logical_tiles:
        return None
    if (block_y != 1 or block_z != 1
            or block_x % hardware.parallel_width):
        return None
    if kernel.launch.grid_size <= hardware.num_compute_units:
        if hardware.parallel_width != 32:
            return None
        return AnalyticalDependencyPath(
            name="gpu_external_split_k_single_wave_load_mma_store",
            services=(
                "launch:submit",
                "memory:global_read",
                "memory_latency:global_read",
                "compute:compute_tensor",
                "memory:global_write",
                "memory_latency:global_write",
            ),
        )
    concurrent_service_groups = ((
        "memory:global_read", "compute:compute_tensor"),)
    path_name = "gpu_external_split_k_cohort_load_mma_store"
    serial_service_counts = ()
    if kernel.launch.algorithm in _MCBLAS_GEMM_ALGORITHMS:
        tile_m, tile_n, tile_k, _ = _tile(kernel)
        a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
        stage_bytes = (
            tile_m * tile_k * a_element_bytes
            + tile_k * tile_n * b_element_bytes)
        buffer_capacity = min(
            hardware.max_smem_per_unit_bytes,
            hardware.max_smem_per_block_bytes,
        )
        tensor_units = hardware.resource_by_role("compute_tensor").total_units
        if tensor_units % hardware.num_compute_units:
            raise ValueError(
                "mcBLAS Tensor units must divide evenly across compute units")
        tensor_subcores_per_unit = tensor_units // hardware.num_compute_units
        wave_phases = block_x // hardware.parallel_width
        serialized_worker_mapping = (
            (kernel.launch.grid[0] > 1 and kernel.launch.grid[1] == 1)
            or tile_m < tile_n
            or ((tile_m, tile_n, tile_k) == (128, 128, 128)
                and stage_bytes == buffer_capacity))
        if (serialized_worker_mapping
                and buffer_capacity // stage_bytes == 1
                and wave_phases == tensor_subcores_per_unit):
            k = int(kernel.shape["k"])
            if k % tile_k:
                raise ValueError(
                    "mcBLAS external split-K requires complete K tiles")
            ctas_per_worker = math.ceil(
                kernel.launch.grid_size / hardware.num_compute_units)
            k_tiles_per_split = math.ceil(
                k // tile_k / tiling.split_k)
            dependent_load_levels = ctas_per_worker * k_tiles_per_split
            if (tile_m, tile_n, tile_k) == (128, 128, 128):
                dependent_load_levels *= 2
                credit_fill_levels = 2 * 16
                dependent_load_levels = (
                    min(dependent_load_levels, credit_fill_levels)
                    + math.ceil(max(
                        0, dependent_load_levels
                        - credit_fill_levels) / 2))
                path_name = (
                    "mcblas_external_split_k_dual_stream_refill")
            refill_load_levels = dependent_load_levels - 1
            if refill_load_levels <= 0:
                raise ValueError(
                    "mcBLAS external split-K has no repeated load level")
            if path_name == "gpu_external_split_k_cohort_load_mma_store":
                path_name = "mcblas_external_split_k_single_buffer_refill"
            serial_service_counts = ((
                "memory_latency:global_read", refill_load_levels),)
        elif (stage_bytes > buffer_capacity
                and wave_phases == tensor_subcores_per_unit):
            k = int(kernel.shape["k"])
            if k % tile_k:
                raise ValueError(
                    "mcBLAS external split-K requires complete K tiles")
            ctas_per_worker = math.ceil(
                kernel.launch.grid_size / hardware.num_compute_units)
            k_tiles_per_split = math.ceil(
                k // tile_k / tiling.split_k)
            # The exact 64x128x256 ``2m1n8k`` program has distinct A and B
            # direct-to-shared streams.  Across queued CTA cohorts neither
            # stream has a second resident buffer, so both completion chains
            # must drain before the AP can reuse its single allocation.
            operand_streams = (
                2 if (tile_m, tile_n, tile_k) == (64, 128, 256)
                else 1)
            refill_load_levels = (
                ctas_per_worker * k_tiles_per_split * operand_streams - 1)
            if refill_load_levels <= 0:
                raise ValueError(
                    "mcBLAS external split-K has no repeated load level")
            path_name = "mcblas_external_split_k_oversized_refill"
            serial_service_counts = ((
                "memory_latency:global_read", refill_load_levels),)
    if (not serial_service_counts
            and hardware.parallel_width == 64
            and kernel.launch.grid_size > hardware.num_compute_units):
        tile_m, tile_n, tile_k, _ = _tile(kernel)
        a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
        stage_bytes = (
            tile_m * tile_k * a_element_bytes
            + tile_k * tile_n * b_element_bytes)
        buffer_capacity = min(
            hardware.max_smem_per_unit_bytes,
            hardware.max_smem_per_block_bytes,
        )
        ctas_per_worker = math.ceil(
            kernel.launch.grid_size / hardware.num_compute_units)
        tensor_units = hardware.resource_by_role("compute_tensor").total_units
        if tensor_units % hardware.num_compute_units:
            raise ValueError(
                "wave64 Tensor units must divide evenly across compute units")
        tensor_subcores_per_unit = tensor_units // hardware.num_compute_units
        wave_phases = block_x // hardware.parallel_width
        if (stage_bytes * 2 == buffer_capacity
                and wave_phases == tensor_subcores_per_unit):
            k = int(kernel.shape["k"])
            k_tiles_per_split = math.ceil(
                math.ceil(k / tile_k) / tiling.split_k)
            dependent_load_levels = (
                ctas_per_worker * k_tiles_per_split)
            if dependent_load_levels <= 1:
                raise ValueError(
                    "wave64 rolling split-K has no repeated load level")
            # The 64x64x128 4m1n8k source program uses one 32 KiB rolling
            # operand allocation on a 64 KiB AP.  Its four wave64 phases map
            # one-to-one to the four Tensor subcores and therefore advance
            # concurrently, while the next K tile cannot reuse the same
            # allocation before the current tile reaches its completion
            # barrier.  Count one refill level per K tile and assigned CTA;
            # aggregate read service already includes the first completion.
            path_name = "gpu_wave64_external_split_k_rolling_refill"
            serial_service_counts = ((
                "memory_latency:global_read",
                dependent_load_levels - 1,
            ),)
        elif ctas_per_worker > 1:
            buffer_depth = buffer_capacity // stage_bytes
            if buffer_depth > 0:
                serial_service_counts = ((
                    "memory_latency:global_read",
                    ctas_per_worker * buffer_depth - 1),)
    # Independent CTA cohorts hide per-split K-load latency behind other
    # workers.  Submission and terminal partial publication delimit the
    # aggregate read/Tensor steady state.  The external reducer remains a
    # separate physical kernel with its own prediction path.
    return AnalyticalDependencyPath(
        name=path_name,
        services=(
            "launch:submit",
            "memory:global_read",
            "memory_latency:global_read",
            "compute:compute_tensor",
            "memory:global_write",
            "memory_latency:global_write",
        ),
        concurrent_service_groups=concurrent_service_groups,
        serial_service_counts=serial_service_counts,
    )


def _gpu_unstaged_single_wave_dependency_pipeline(
        kernel: KernelSpec, hardware) -> AnalyticalDependencyPath | None:
    """Return Load/MMA/Store order for an unstaged single-wave GEMM."""

    tiling = kernel.launch.tiling
    if (kernel.launch.algorithm != "nvidia_cublaslt_gemm"
            or tiling is None or tiling.pipeline_depth is not None
            or tiling.split_k != 1
            or kernel.launch.parameters["compulsory_traffic"] != 1.0):
        return None
    b, m, n = (
        int(kernel.shape[name]) for name in ("b", "m", "n"))
    tile_m, tile_n, _, _ = _tile(kernel)
    logical_tiles = _kernel_logical_workgroups(
        kernel, tile_m, tile_n, tiling.split_k)
    if logical_tiles > hardware.num_compute_units:
        return None
    return AnalyticalDependencyPath(
        name="gpu_unstaged_single_wave_load_mma_store",
        services=(
            "launch:submit",
            "memory:global_read",
            "memory_latency:global_read",
            "compute:compute_tensor",
            "memory:global_write",
            "memory_latency:global_write",
        ),
    )


def _mcblas_subtile_dependency_pipeline(
        kernel: KernelSpec, hardware, active_units: int
        ) -> AnalyticalDependencyPath | None:
    """Return a compact source-visible mcBLAS Load/MMA/Store path."""

    tiling = kernel.launch.tiling
    if tiling is None:
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    logical_tiles = gemm_logical_workgroups(
        b, m, n, tile_m, tile_n, tiling.split_k)
    physical_factor = _mcblas_exact_physical_factor(kernel, logical_tiles)
    if physical_factor is None:
        return None
    block_x, block_y, block_z = kernel.launch.block
    if block_y != 1 or block_z != 1:
        raise ValueError("mcBLAS GEMM requires a one-dimensional block")
    if block_x % hardware.parallel_width:
        raise ValueError(
            "mcBLAS GEMM block width must contain complete wave64 phases")
    if k % tile_k:
        raise ValueError("mcBLAS subtile pipeline requires complete K tiles")
    wave_phases = block_x // hardware.parallel_width
    k_tiles = k // tile_k
    tensor_units = hardware.resource_by_role("compute_tensor").total_units
    if tensor_units % hardware.num_compute_units:
        raise ValueError(
            "mcBLAS Tensor units must divide evenly across compute units")
    tensor_subcores_per_unit = tensor_units // hardware.num_compute_units
    split_k_temporal_wave_cohorts = (
        b * m == tile_m
        and (tile_m, tile_n, tile_k) == (256, 256, 64)
        and wave_phases > tensor_subcores_per_unit
        and wave_phases % tensor_subcores_per_unit == 0)
    if (tiling.split_k > 1
            and kernel.launch.grid_size <= hardware.num_compute_units
            and (wave_phases <= tensor_subcores_per_unit
                 or b * m > tile_m
                 or split_k_temporal_wave_cohorts)):
        # The retained grid has one physical CTA for every semantic
        # (M, N, split-K) tile.  A split owns a contiguous subset of the K
        # tiles; the longest tail therefore determines the per-CTA dependent
        # Load/MMA wave64 levels.  Aggregate byte and Tensor services still
        # describe all CTAs, so only the intervening first-request latency is
        # repeated here.
        k_tiles_per_split = math.ceil(k_tiles / tiling.split_k)
        ctas_per_worker = math.ceil(
            kernel.launch.grid_size / active_units)
        if b * m > tile_m:
            temporal_wave_cohorts = math.ceil(
                wave_phases / tensor_subcores_per_unit)
            # The exact full-tile program retains separate A/B completion
            # chains even when its four wave phases occupy the Tensor
            # subcores concurrently across multiple M tiles.
            operand_streams = (
                2 if (tile_m, tile_n, tile_k) == (128, 128, 128)
                else 1)
            dependent_load_levels = (
                ctas_per_worker * k_tiles_per_split
                * temporal_wave_cohorts * operand_streams)
        elif split_k_temporal_wave_cohorts:
            # The 512-thread ``2m1n8k`` program spans two temporal cohorts
            # across the four physical Tensor subcores.  A split-local K loop
            # must advance both cohorts before its complete 64-KiB operand
            # allocation can be reused.  This is the split-K counterpart of
            # the exact non-split temporal-cohort path below.
            temporal_wave_cohorts = (
                wave_phases // tensor_subcores_per_unit)
            dependent_load_levels = (
                ctas_per_worker * k_tiles_per_split
                * temporal_wave_cohorts)
        else:
            dependent_load_levels = (
                ctas_per_worker * k_tiles_per_split * wave_phases)
            if ((tile_m, tile_n, tile_k) == (32, 256, 128)
                    and wave_phases == tensor_subcores_per_unit):
                # The four wave64 phases occupy the four physical Tensor
                # subcores as one CTA cohort.  Preserve one direct-to-shared
                # refill completion per K tile rather than serializing the
                # concurrent subcore phases.
                dependent_load_levels = (
                    ctas_per_worker * k_tiles_per_split)
            elif ((tile_m, tile_n, tile_k) == (128, 128, 128)
                    and wave_phases == tensor_subcores_per_unit):
                # The pinned ``4stage`` body issues sixteen direct-to-shared
                # loads for the next K tile and advances independent A/B
                # streams through nonzero gvmcnt thresholds.  The first two
                # 16-entry credit windows expose their complete fill; once
                # both are live, each pair of later wave-phase checkpoints
                # shares one first-request completion.
                credit_fill_levels = 2 * 16
                dependent_load_levels = (
                    min(dependent_load_levels, credit_fill_levels)
                    + math.ceil(max(
                        0, dependent_load_levels - credit_fill_levels) / 2))
            elif ((tile_m, tile_n, tile_k) == (64, 128, 256)
                    and wave_phases == tensor_subcores_per_unit):
                # The pinned ``2m1n8k`` program reaches gvmcnt(23) before
                # its first barrier, then interleaves MMA while the nonzero
                # thresholds drain two direct-to-shared requests at a time.
                # A slice no deeper than the four wave phases drains both A/B
                # streams before reaching that steady loop.  Deeper slices
                # preserve the complete 24-entry fill and overlap only later
                # steady A/B stream completions.
                if k_tiles_per_split <= wave_phases:
                    dependent_load_levels *= 2
                else:
                    credit_fill_levels = 24
                    dependent_load_levels = (
                        min(dependent_load_levels, credit_fill_levels)
                        + math.ceil(max(
                            0, dependent_load_levels
                            - credit_fill_levels) / 2))
        if dependent_load_levels <= 1:
            return None
        return AnalyticalDependencyPath(
            name="mcblas_split_k_load_mma_store",
            services=(
                "memory:global_read",
                "memory_latency:global_read",
                "compute:compute_tensor",
                "memory:global_write",
                "memory_latency:global_write",
            ),
            concurrent_service_groups=((
                "memory:global_read", "compute:compute_tensor"),),
            serial_service_counts=((
                "memory_latency:global_read",
                dependent_load_levels - 1,
            ),),
        )
    if physical_factor == 1:
        if tiling.split_k > 1:
            return None
        a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
        stage_bytes = (
            tile_m * tile_k * a_element_bytes
            + tile_k * tile_n * b_element_bytes)
        buffer_capacity = min(
            hardware.max_smem_per_unit_bytes,
            hardware.max_smem_per_block_bytes,
        )
        if ((tile_m, tile_n, tile_k) == (128, 256, 64)
                and stage_bytes <= buffer_capacity
                and buffer_capacity // stage_bytes == 1
                and wave_phases > tensor_subcores_per_unit
                and wave_phases % tensor_subcores_per_unit == 0):
            # The pinned ``2m1n8k`` binary executes eight wave64 phases on
            # four physical Tensor subcores.  Its rolling K loop retains one
            # complete 48-KiB operand allocation and therefore advances two
            # temporal wave cohorts before that allocation can be reused.
            # Aggregate read and Tensor services overlap; only the later
            # source-visible cohort completions add latency levels.
            source_temporal_wave_cohorts = (
                wave_phases // tensor_subcores_per_unit)
            if tile_m % source_temporal_wave_cohorts:
                raise ValueError(
                    "mcBLAS temporal cohorts must partition the M tile")
            rows_per_temporal_cohort = (
                tile_m // source_temporal_wave_cohorts)
            temporal_wave_cohorts = min(
                source_temporal_wave_cohorts,
                math.ceil(min(b * m, tile_m)
                          / rows_per_temporal_cohort),
            )
            ctas_per_worker = math.ceil(
                kernel.launch.grid_size / active_units)
            dependent_load_levels = (
                ctas_per_worker * k_tiles * temporal_wave_cohorts)
            if dependent_load_levels <= 1:
                raise ValueError(
                    "mcBLAS temporal wave cohort has no repeated load level")
            return AnalyticalDependencyPath(
                name="mcblas_temporal_wave_cohort_load_mma_store",
                services=(
                    "launch:submit",
                    "memory:global_read",
                    "memory_latency:global_read",
                    "compute:compute_tensor",
                    "memory:global_write",
                    "memory_latency:global_write",
                ),
                concurrent_service_groups=((
                    "memory:global_read", "compute:compute_tensor"),),
                serial_service_counts=((
                    "memory_latency:global_read",
                    dependent_load_levels - 1,
                ),),
            )
        if kernel.launch.grid_size > hardware.num_compute_units:
            if wave_phases != tensor_subcores_per_unit:
                return None
            if stage_bytes <= buffer_capacity:
                buffer_depth = buffer_capacity // stage_bytes
                if (buffer_depth == 1
                        and b * m > tile_m
                        and (tile_m, tile_n, tile_k)
                        == (128, 128, 128)):
                    # The pinned ``4stage`` body keeps independent A/B
                    # direct-to-shared streams.  Across successive CTA
                    # cohorts their first two 16-entry credit windows fill
                    # completely; later paired checkpoints share one
                    # completion exactly as in the split-K form above.
                    ctas_per_worker = math.ceil(
                        kernel.launch.grid_size / active_units)
                    dependent_load_levels = (
                        ctas_per_worker * k_tiles * 2)
                    credit_fill_levels = 2 * 16
                    dependent_load_levels = (
                        min(dependent_load_levels, credit_fill_levels)
                        + math.ceil(max(
                            0, dependent_load_levels
                            - credit_fill_levels) / 2))
                    return AnalyticalDependencyPath(
                        name=(
                            "mcblas_multiwave_dual_stream_"
                            "load_mma_store"),
                        services=(
                            "launch:submit",
                            "memory:global_read",
                            "memory_latency:global_read",
                            "compute:compute_tensor",
                            "memory:global_write",
                            "memory_latency:global_write",
                        ),
                        concurrent_service_groups=((
                            "memory:global_read",
                            "compute:compute_tensor"),),
                        serial_service_counts=((
                            "memory_latency:global_read",
                            dependent_load_levels - 1,
                        ),),
                    )
                if buffer_depth != 1 or b * m > tile_m:
                    return None
                # A full-width CTA occupies all Tensor subcores, and its
                # complete A/B pair consumes the sole AP-local buffer.  The
                # next CTA cohort on that AP therefore starts its load only
                # after the resident cohort completes MMA.
                return AnalyticalDependencyPath(
                    name="mcblas_multiwave_single_buffer_load_mma_store",
                    services=(
                        "launch:submit",
                        "memory_latency:global_read",
                        "memory:global_read",
                        "compute:compute_tensor",
                        "memory:global_write",
                        "memory_latency:global_write",
                    ),
                )
            ctas_per_worker = math.ceil(
                kernel.launch.grid_size / active_units)
            refill_load_levels = (
                ctas_per_worker * k_tiles * wave_phases - 1)
            if refill_load_levels <= 0:
                raise ValueError(
                    "mcBLAS multiwave refill requires a later CTA cohort")
            # A worker must advance through every retained K tile across its
            # assigned CTA cohorts because no complete macro tile fits in its
            # AP-local buffer.  Aggregate read service already carries the
            # first completion, hence N dependent levels contribute N-1
            # additional completions.  The four wave64 phases occupy the four
            # Tensor subcores concurrently, so depth is counted in K tiles
            # rather than expanded wave tasks.
            return AnalyticalDependencyPath(
                name="mcblas_oversized_multiwave_refill",
                services=(
                    "launch:submit",
                    "memory:global_read",
                    "memory_latency:global_read",
                    "compute:compute_tensor",
                    "memory:global_write",
                    "memory_latency:global_write",
                ),
                concurrent_service_groups=((
                    "memory:global_read", "compute:compute_tensor"),),
                serial_service_counts=((
                    "memory_latency:global_read", refill_load_levels),),
            )
        if stage_bytes > buffer_capacity:
            if wave_phases != tensor_subcores_per_unit:
                return None
            # The exact one-wave CTA cannot retain its complete macro-tile
            # operands in the AP-local buffer.  Its four wave64 phases map
            # one-to-one to the four physical Tensor subcores, so each
            # retained K tile must advance through those source-visible
            # load/MMA phases.  The compact dependency path below prices the
            # intervening first-request completions without expanding CTAs,
            # K iterations, or wave phases into explicit tasks.
        else:
            buffer_depth = buffer_capacity // stage_bytes
            if buffer_depth != 1:
                return None
            if ((tile_m, tile_n, tile_k) == (32, 64, 256)
                    and b * m < tile_m
                    and wave_phases == tensor_subcores_per_unit
                    and 2 * k_tiles <= 24):
                # The short partial-M programs issue distinct A and B
                # direct-to-shared streams.  Before their combined requests
                # fill the source-visible gvmcnt(23) window, every K tile must
                # complete both streams before the sole operand allocation is
                # reused.  Aggregate read service already contains the first
                # completion, so only the remaining levels are added here.
                return AnalyticalDependencyPath(
                    name="mcblas_short_partial_m_dual_stream_load_mma_store",
                    services=(
                        "launch:submit",
                        "memory_latency:global_read",
                        "memory:global_read",
                        "compute:compute_tensor",
                        "memory:global_write",
                        "memory_latency:global_write",
                    ),
                    serial_service_counts=((
                        "memory_latency:global_read", 2 * k_tiles - 1),),
                )
            if (tile_m, tile_n, tile_k) == (80, 64, 128):
                # The installed mcBLAS 64x80x128 ``sb_0_0`` program fills
                # its single shared allocation in nine direct-to-shared
                # pieces.  Its steady-state loop interleaves the next nine
                # loads with the current MMA work and gates reuse through
                # four gvmcnt/barrier phases, one for every wave64 phase in
                # the 256-thread block.  Aggregate read and Tensor services
                # therefore overlap, while every K tile still advances
                # through those source-visible completion levels.
                dependent_load_levels = k_tiles * wave_phases
                if dependent_load_levels <= 1:
                    raise ValueError(
                        "mcBLAS rolling subtile has no repeated load level")
                return AnalyticalDependencyPath(
                    name="mcblas_rolling_subtile_load_mma_store",
                    services=(
                        "launch:submit",
                        "memory:global_read",
                        "memory_latency:global_read",
                        "compute:compute_tensor",
                        "memory:global_write",
                        "memory_latency:global_write",
                    ),
                    concurrent_service_groups=((
                        "memory:global_read", "compute:compute_tensor"),),
                    serial_service_counts=((
                        "memory_latency:global_read",
                        dependent_load_levels - 1,
                    ),),
                )
            has_independent_cohorts = (
                b * m > tile_m
                or (b * m == tile_m
                    and (tile_m, tile_n, tile_k) == (128, 128, 128)
                    and stage_bytes == buffer_capacity
                    and wave_phases == tensor_subcores_per_unit)
            )
            if has_independent_cohorts:
                serial_service_counts = ()
                if ((tile_m, tile_n, tile_k) == (128, 128, 128)
                        and stage_bytes == buffer_capacity
                        and wave_phases == tensor_subcores_per_unit):
                    if k_tiles <= 1:
                        raise ValueError(
                            "mcBLAS resident cohort has no repeated K tile")
                    # The pinned ``4stage`` binary has one complete 64-KiB
                    # operand allocation.  Its steady loop advances the A
                    # and B direct-to-shared streams separately before that
                    # allocation is reused.  Count those two source-visible
                    # completions for every retained K tile; aggregate read
                    # service already contains the first completion.
                    serial_service_counts = ((
                        "memory_latency:global_read", 2 * k_tiles - 1),)
                # Distinct M tiles or a complete M tile distributed over all
                # Tensor-subcore waves expose independent operand cohorts.  A
                # one-wave exact grid can overlap aggregate read and Tensor
                # services, while the exact 128x128x128 program must still
                # complete both operand streams before reusing its sole
                # buffer.  Launch and terminal publication remain ordered.
                # M tails do not satisfy this contract and retain the serial
                # path below.
                return AnalyticalDependencyPath(
                    name="mcblas_resident_cohort_load_mma_store",
                    services=(
                        "launch:submit",
                        "memory:global_read",
                        "memory_latency:global_read",
                        "compute:compute_tensor",
                        "memory:global_write",
                        "memory_latency:global_write",
                    ),
                    concurrent_service_groups=((
                        "memory:global_read", "compute:compute_tensor"),),
                    serial_service_counts=serial_service_counts,
                )
            # With room for exactly one complete A/B tile pair, the next load
            # cannot overlap the resident tile's MMA.  Whole-launch service
            # times already compact all workers and K iterations, so the
            # dependency is one serial path rather than a CTA or K-tile
            # expansion.
            return AnalyticalDependencyPath(
                name="mcblas_single_buffer_load_mma_store",
                services=(
                    "launch:submit",
                    "memory_latency:global_read",
                    "memory:global_read",
                    "compute:compute_tensor",
                    "memory:global_write",
                    "memory_latency:global_write",
                ),
                serial_service_counts=((
                    "memory_latency:global_read", 2),),
            )
    ctas_per_worker = math.ceil(kernel.launch.grid_size / active_units)
    dependent_load_levels = ctas_per_worker * k_tiles * wave_phases
    if ((tile_m, tile_n, tile_k) == (64, 128, 256)
            and wave_phases == tensor_subcores_per_unit):
        # See the exact ``2m1n8k`` credit sequence above.  This branch is the
        # corresponding non-split one-wave program.  Before the first
        # 24-entry gvmcnt window is filled, the pinned source advances the A
        # and B operand streams independently; neither stream has a later
        # filled window behind which its completion can hide.  Deeper loops
        # retain the complete fill and overlap pairs of later completions.
        credit_fill_levels = 24
        if dependent_load_levels <= credit_fill_levels:
            dependent_load_levels *= 2
        else:
            dependent_load_levels = (
                credit_fill_levels
                + math.ceil(
                    (dependent_load_levels - credit_fill_levels) / 2))
    if dependent_load_levels <= 1:
        raise ValueError("mcBLAS subtile pipeline has no repeated load level")
    # The aggregate read path already carries its first completion lower
    # bound.  The remaining source-ordered wave64 barriers are additional
    # latency levels; aggregate bytes and Tensor work overlap in steady state.
    return AnalyticalDependencyPath(
        name="mcblas_subtile_load_mma_store",
        services=(
            "memory:global_read",
            "memory_latency:global_read",
            "compute:compute_tensor",
            "memory:global_write",
            "memory_latency:global_write",
        ),
        concurrent_service_groups=((
            "memory:global_read", "compute:compute_tensor"),),
        serial_service_counts=((
            "memory_latency:global_read", dependent_load_levels - 1),),
    )


def _mcblas_tile_pipeline(
        kernel: KernelSpec, hardware,
        active_units: int) -> PipelineWorkPlan | None:
    """Return one finite fragment pipeline for retained mcBLAS launches.

    The retained four-stage program advances one source operand stage through
    global load, shared storage, and its Tensor work.  Fragment capacity, CTA
    assignment, and tail-predicated aggregate work fully determine the
    recurrence; logical shape boundaries never select a different overlap
    formula.
    """

    PipelineStageDemand = bind_pipeline_stage_template

    tiling = kernel.launch.tiling
    if (tiling is None
            or kernel.launch.algorithm not in _MCBLAS_GEMM_ALGORITHMS
            or tiling.pipeline_depth is not None
            or tiling.split_k != 1):
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    logical_tiles = gemm_logical_workgroups(
        b, m, n, tile_m, tile_n, tiling.split_k)
    physical_factor = _mcblas_exact_physical_factor(kernel, logical_tiles)
    if physical_factor is None or k % tile_k:
        return None
    block_x, block_y, block_z = kernel.launch.block
    if (block_y, block_z) != (1, 1):
        raise ValueError("mcBLAS GEMM requires a one-dimensional block")
    if block_x % hardware.parallel_width:
        raise ValueError(
            "mcBLAS GEMM block width must contain complete wave64 phases")
    tensor_units = hardware.resource_by_role("compute_tensor").total_units
    if tensor_units % hardware.num_compute_units:
        raise ValueError(
            "mcBLAS Tensor units must divide evenly across compute units")
    tensor_subcores_per_unit = tensor_units // hardware.num_compute_units
    wave_phases = block_x // hardware.parallel_width
    # The retained BF16 ``256x{192,256}x64_2m1n8k_512t_4stage`` source
    # programs have eight wave64 phases on C500's four Tensor subcores, so
    # they advance two temporal cohorts.  The TN 256x192 symbol maps to the
    # retained (m, n, k) geometry (192, 256, 64).  Pinned MACA 3.7.2
    # xcore1000 disassembly (SHA256
    # dc20bc66fbdc871c592990929b844f77f557b05c77d01c1b988e9e2338df7a92)
    # shows four direct-to-shared operand stages before the first
    # gvmcnt/barrier handoff (the 256x192 program has seven ``ldg_b128``
    # instructions then ``gvmcnt(4)``).  Model those source
    # stages as the consumer-visible ring; treating all eight waves as
    # independently reusable fragments consumes an operand stage before the
    # program has completed its paired phase.  Other over-subscribed launches
    # have no matching source-program contract and remain aggregate-only.
    uses_bf16_4stage_source = (
        kernel.dtype == "bf16"
        and (tile_m, tile_n, tile_k) in {
            (192, 256, 64), (256, 256, 64)}
        and block_x == 512
        and physical_factor == 1)
    temporal_wave_cohorts = (
        wave_phases // tensor_subcores_per_unit
        if (uses_bf16_4stage_source
            and wave_phases % tensor_subcores_per_unit == 0)
        else 1)
    if (wave_phases != tensor_subcores_per_unit
            and temporal_wave_cohorts != 2):
        return None
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
    operand_tile_bytes = (
        tile_m * tile_k * a_element_bytes
        + tile_k * tile_n * b_element_bytes)
    buffer_capacity = min(
        hardware.max_smem_per_unit_bytes,
        hardware.max_smem_per_block_bytes,
    )
    source_stage_count = 4 if uses_bf16_4stage_source else wave_phases
    fragment_count = source_stage_count * physical_factor
    fragment_bytes = math.ceil(operand_tile_bytes / fragment_count)
    if fragment_bytes > buffer_capacity:
        return None
    resident_fragment_slots = buffer_capacity // fragment_bytes
    if resident_fragment_slots <= 0:
        raise ValueError("mcBLAS retained operand pipeline has no buffer")
    # When the complete operand tile fits, its source phases occupy disjoint
    # regions of the retained allocation and can advance independently.  An
    # oversized tile instead streams those phases through one cyclic ring;
    # byte capacity cannot be reinterpreted as multiple addressable rings.
    buffer_slots = (
        resident_fragment_slots
        if operand_tile_bytes <= buffer_capacity else 1)
    k_tiles = k // tile_k
    grid_size = kernel.launch.grid_size
    if active_units != min(grid_size, hardware.num_compute_units):
        raise ValueError("mcBLAS active AP count disagrees with launch grid")
    fragments_per_cta = k_tiles * source_stage_count
    work_fraction = 1.0 / (grid_size * fragments_per_cta)
    store_fraction = 1.0 / grid_size
    ctas_by_worker = tuple(
        (grid_size - worker + active_units - 1) // active_units
        for worker in range(active_units))
    programs = []
    for cta_count in tuple(dict.fromkeys(ctas_by_worker)):
        physical_workers = tuple(
            worker for worker, count in enumerate(ctas_by_worker)
            if count == cta_count)
        representative = physical_workers[0]
        def fragment_task(
                name: str, *, include_completion_latency: bool,
                repeat_count: int = 1) -> PipelineTaskDemand:
            return PipelineTaskDemand(
                name=name,
                worker=representative,
                repeat_count=repeat_count,
                stages=(
                PipelineStageDemand(
                    name="global_load",
                    service="memory:global_read",
                    work_fraction=work_fraction,
                    include_first_request_latency=include_completion_latency,
                ),
                PipelineStageDemand(
                    name="operand_buffer",
                    service="resource:smem",
                    work_fraction=work_fraction,
                ),
                PipelineStageDemand(
                    name="tensor_mma",
                    service="compute:compute_tensor",
                    work_fraction=work_fraction,
                ),
                ),
            )
        # The first fragment pays the finite request latency; subsequent
        # source phases retain the rolling load/MMA overlap guarded by the
        # source-visible gvmcnt/bsmcnt sequence.
        fragment_tasks = [fragment_task(
            "first_operand_fragment", include_completion_latency=True)]
        if fragments_per_cta > 1:
            fragment_tasks.append(fragment_task(
                "steady_operand_fragments", include_completion_latency=False,
                repeat_count=fragments_per_cta - 1))
        store = PipelineTaskDemand(
            name="mcblas_terminal_store",
            worker=representative,
            stages=(PipelineStageDemand(
                name="global_store",
                service="memory:global_write",
                work_fraction=store_fraction,
                include_first_request_latency=True,
                predecessors=(),
                prior_resources=(
                    "memory:global_read",
                    "resource:smem",
                    "compute:compute_tensor",
                ),
            ),),
        )
        cta = PipelineTaskSequenceDemand(
            name="mcblas_cta",
            worker=representative,
            tasks=tuple(fragment_tasks) + (store,),
        )
        programs.append(PipelineTaskProgramDemand(
            name=f"mcblas_{cta_count}_cta_worker_class",
            worker=representative,
            patterns=(cta,),
            order=(0,),
            repeat_count=cta_count,
            worker_multiplicity=len(physical_workers),
            physical_workers=physical_workers,
        ))
    return PipelineWorkPlan(
        worker_count=active_units,
        tasks=tuple(programs),
        buffers=(PipelineBuffer(
            name="mcblas_operand_fragments",
            producer_stage="operand_buffer",
            consumer_stage="tensor_mma",
            slots=buffer_slots,
            scope="worker",
        ),),
    )


def _mcblas_oversized_rolling_pipeline(
        kernel: KernelSpec, hardware, active_units: int
        ) -> PipelineWorkPlan | None:
    """Return the compact rolling program for mcBLAS 64x128x256 GEMM.

    The pinned ``2m1n8k`` binary first fills 24 direct-to-shared requests.
    Its BB_0 loop then issues the next 24 requests while executing the
    resident tile's 128 MMA instructions, decrementing the K extent by 256
    at the loop backedge.  BB_1 drains the final resident tile without a new
    global load.  Nonzero gvmcnt thresholds maintain that rolling credit
    window; they are not independent full-latency memory requests.

    Represent the exact CTA-to-AP assignment as at most two worker classes.
    Each class repeats one compact CTA program, so neither CTAs nor K tiles
    are materialized in the plan.
    """

    PipelineStageDemand = bind_pipeline_stage_template

    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k != 1:
        return None
    b, m, n, k = (
        int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if (tile_m, tile_n, tile_k) != (64, 128, 256):
        return None
    logical_tiles = gemm_logical_workgroups(
        b, m, n, tile_m, tile_n, tiling.split_k)
    if (_mcblas_exact_physical_factor(kernel, logical_tiles) != 1
            or kernel.launch.grid_size <= hardware.num_compute_units
            or k % tile_k):
        return None
    block_x, block_y, block_z = kernel.launch.block
    if (block_y, block_z) != (1, 1):
        raise ValueError("mcBLAS GEMM requires a one-dimensional block")
    if block_x % hardware.parallel_width:
        raise ValueError(
            "mcBLAS GEMM block width must contain complete wave64 phases")
    tensor_units = hardware.resource_by_role("compute_tensor").total_units
    if tensor_units % hardware.num_compute_units:
        raise ValueError(
            "mcBLAS Tensor units must divide evenly across compute units")
    tensor_subcores_per_unit = tensor_units // hardware.num_compute_units
    if block_x // hardware.parallel_width != tensor_subcores_per_unit:
        return None
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
    stage_bytes = (
        tile_m * tile_k * a_element_bytes
        + tile_k * tile_n * b_element_bytes)
    buffer_capacity = min(
        hardware.max_smem_per_unit_bytes,
        hardware.max_smem_per_block_bytes,
    )
    if stage_bytes <= buffer_capacity:
        return None

    k_tiles = k // tile_k
    if k_tiles <= 1:
        return None
    grid_size = kernel.launch.grid_size
    k_tile_fraction = 1.0 / (grid_size * k_tiles)
    cta_store_fraction = 1.0 / grid_size
    ctas_by_worker = tuple(
        (grid_size - worker + active_units - 1) // active_units
        for worker in range(active_units)
    )
    if min(ctas_by_worker) <= 0:
        raise ValueError("mcBLAS rolling pipeline has an inactive worker")

    def _k_tile(worker: int, name: str, *, first: bool,
                repeat_count: int = 1) -> PipelineTaskDemand:
        return PipelineTaskDemand(
            name=name,
            worker=worker,
            repeat_count=repeat_count,
            stages=(
                PipelineStageDemand(
                    "operand_load", "memory:global_read", k_tile_fraction,
                    include_first_request_latency=first),
                PipelineStageDemand(
                    "operand_shared", "resource:smem", k_tile_fraction),
                PipelineStageDemand(
                    "tensor_mma", "compute:compute_tensor",
                    k_tile_fraction),
            ),
        )

    programs = []
    for cta_count in tuple(dict.fromkeys(ctas_by_worker)):
        physical_workers = tuple(
            worker for worker, count in enumerate(ctas_by_worker)
            if count == cta_count)
        representative = physical_workers[0]
        cta = PipelineTaskSequenceDemand(
            name="cta",
            worker=representative,
            tasks=(
                _k_tile(representative, "first_k_tile", first=True),
                _k_tile(
                    representative, "rolling_k_tiles", first=False,
                    repeat_count=k_tiles - 1),
                PipelineTaskDemand(
                    name="terminal_store",
                    worker=representative,
                    stages=(PipelineStageDemand(
                        "output_store", "memory:global_write",
                        cta_store_fraction,
                        prior_resources=(
                            "memory:global_read",
                            "resource:smem",
                            "compute:compute_tensor",
                        )),),
                ),
            ),
        )
        programs.append(PipelineTaskProgramDemand(
            name=f"mcblas_{cta_count}_cta_worker_class",
            worker=representative,
            patterns=(cta,),
            order=(0,),
            repeat_count=cta_count,
            worker_multiplicity=len(physical_workers),
            physical_workers=physical_workers,
        ))
    return PipelineWorkPlan(
        worker_count=active_units,
        tasks=tuple(programs),
        buffers=(PipelineBuffer(
            name="mcblas_rolling_operand_fragments",
            producer_stage="operand_load",
            consumer_stage="tensor_mma",
            slots=2,
            scope="worker",
        ),),
    )


def _apply_gemm_bf16_mma_saturation(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Apply the workload-independent BF16 MMA occupancy surface to GEMM."""

    saturation = hardware.gemm_bf16_mma_saturation
    if saturation is None or kernel.dtype != saturation["dtype"]:
        return inp
    tensor_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_tensor")
    if len(tensor_demands) != 1:
        raise ValueError(
            "GEMM BF16 MMA saturation requires one Tensor compute demand")
    tensor = tensor_demands[0]
    tiling = kernel.launch.tiling
    if tiling is None:
        raise ValueError("GEMM BF16 MMA saturation requires tiled launch")
    block_threads = kernel.launch.block_size
    if block_threads % hardware.parallel_width:
        raise ValueError(
            "GEMM BF16 MMA saturation requires complete physical waves")
    waves_per_cta = block_threads // hardware.parallel_width
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    output_elements = tile_m * tile_n
    accumulator_width = saturation["accumulator_vector_width"]
    accumulator_denominator = block_threads * accumulator_width
    if output_elements % accumulator_denominator:
        accumulator_chains = None
    else:
        accumulator_chains = output_elements // accumulator_denominator
    a_element_bytes, b_element_bytes, _ = _operand_element_bytes(kernel)
    operand_allocation_bytes = (
        tile_m * tile_k * a_element_bytes
        + tile_k * tile_n * b_element_bytes)
    buffer_capacity = min(
        hardware.max_smem_per_unit_bytes,
        hardware.max_smem_per_block_bytes,
    )
    physical_allocation_bytes = min(
        operand_allocation_bytes, buffer_capacity)
    ctas_per_worker = math.ceil(inp.grid_size / inp.active_units)
    resident_ctas = min(
        hardware.max_blocks_per_unit,
        hardware.max_threads_per_unit // block_threads,
        hardware.max_warps_per_unit // waves_per_cta,
        buffer_capacity // physical_allocation_bytes,
        ctas_per_worker,
    )
    if resident_ctas <= 0:
        raise ValueError("GEMM BF16 MMA saturation has no resident CTA")
    tensor_units = hardware.resource_by_role("compute_tensor").total_units
    if tensor_units % hardware.num_compute_units:
        raise ValueError(
            "GEMM BF16 MMA saturation requires an integral Tensor-subcore "
            "count per compute unit")
    tensor_subcores_per_unit = tensor_units // hardware.num_compute_units
    active_waves_per_cta = waves_per_cta
    if (
            kernel.launch.algorithm in _MCBLAS_GEMM_ALGORITHMS
            and (tile_m, tile_n, tile_k) == (256, 256, 64)
            and waves_per_cta == 2 * tensor_subcores_per_unit
            and resident_ctas == 1):
        # The retained 2m1n8k program maps eight wave64 phases onto four
        # Tensor subcores as two temporal cohorts.  The independent MMA
        # surface describes simultaneous resident waves, so select its
        # four-wave point; the second cohort remains represented by the
        # source-program pipeline, not by an invented throughput penalty.
        active_waves_per_cta = tensor_subcores_per_unit
    occupancy_key = (
        resident_ctas * active_waves_per_cta, accumulator_chains)
    rates = saturation["flops_per_ap_cycle_by_occupancy"]
    limitations = inp.limitations
    if occupancy_key in rates:
        efficiency = (
            rates[occupancy_key]
            / saturation["architecture_peak_flops_per_ap_cycle"])
    else:
        efficiency = saturation["fallback_efficiency"]
        limitation = "gemm_bf16_mma_accumulator_surface_unavailable"
        if limitation not in limitations:
            limitations = (*limitations, limitation)
    adjusted = dataclasses.replace(
        tensor,
        schedule_efficiency=(
            tensor.schedule_efficiency * efficiency),
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            adjusted if demand is tensor else demand
            for demand in inp.compute_demands),
        limitations=limitations,
    )


def _apply_gemm_tensor_activity_clock(inp, kernel, hardware):
    """Compose activity and instruction service by the limiting capacity.

    The instruction-only experiment provides cycles, not a new operating
    clock. Its nominal-clock capacity is an optimistic upper bound. Taking
    the smaller capacity preserves an independently measured activity clock
    without multiplying two estimates of the same Tensor service.
    """
    activity = _apply_gemm_tensor_activity_clock_only(inp, kernel, hardware)
    profile = hardware.gemm_tensor_instruction_service
    fields = ("tensor_instruction_m", "tensor_instruction_n", "tensor_instruction_k")
    parameters = kernel.launch.parameters
    if profile is None or not all(name in parameters for name in fields):
        return activity
    m, n, k = (int(parameters[name]) for name in fields)
    if (kernel.dtype != "fp16" or kernel.launch.block_size != 384
            or m != profile["instruction_m"] or k != profile["instruction_k"]
            or n not in profile["instruction_rates"]):
        return dataclasses.replace(activity, limitations=activity.limitations + (
            "tensor_instruction_capacity_outside_measured_instruction_domain",))
    tile_m, tile_n, tile_k, _ = _tile(kernel)
    if tile_m % n or tile_n % m or tile_k % k:
        raise ValueError("Tensor instruction geometry does not divide the physical GEMM tile")
    capacity = hardware.resource_by_role("compute_tensor").effective_capacity(
        inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
    # Normalize the per-SM instruction service to the configured full-device
    # resource. PerformanceFormula._compute_capacity applies the active fraction
    # of a shared compute resource; using active_units here charges it twice.
    instruction_ratio = (profile["instruction_rates"][n]
        * hardware.nominal_sm_freq_hz * hardware.num_compute_units / capacity)
    demands = []
    for before, after in zip(inp.compute_demands, activity.compute_demands, strict=True):
        if before.resource_role == "compute_tensor":
            activity_ratio = after.schedule_efficiency / before.schedule_efficiency
            after = dataclasses.replace(after, schedule_efficiency=before.schedule_efficiency
                * min(activity_ratio, instruction_ratio))
        demands.append(after)
    return dataclasses.replace(activity, compute_demands=tuple(demands),
        limitations=activity.limitations + (
            "tensor_instruction_capacity_measured_window_upper_envelope",
            "tensor_instruction_capacity_nominal_clock_upper_bound_not_native_progress",))


def _apply_gemm_tensor_activity_clock_only(inp, kernel, hardware):
    profile = hardware.gemm_tensor_activity_clock
    if (profile is None or "operand_transport_tma" not in kernel.launch.parameters
            or _operand_element_bytes(kernel)[:2] != (2, 2)):
        return inp
    if "concurrency_profiles" in profile and inp.active_units in profile["concurrency_profiles"]:
        profile = profile["concurrency_profiles"][inp.active_units]
    if profile["flops_per_sm_cycle"] is not None and (
            kernel.dtype != "fp16"
            or kernel.launch.block_size != profile["threads_per_cta"]
            # Partial-profile surplus CTAs return before the same-kernel
            # 96-Mi-cycle prelude. They do not survive into the measured
            # steady activity; a launch with no such surplus has the same
            # exact active population. Dispatch remains separately priced.
            or not profile["minimum_active_units"] <= kernel.launch.grid_size
            <= profile["physical_grid_ctas"]):
        return dataclasses.replace(inp, limitations=inp.limitations + (
            "tensor_shared_service_outside_measured_instruction_or_launch_domain",))
    if not profile["minimum_active_units"] <= inp.active_units <= profile["maximum_active_units"]:
        return dataclasses.replace(inp, limitations=inp.limitations + (
            "tensor_activity_clock_outside_measured_concurrency_domain",))
    frequency = profile["sm_freq_hz"]
    if profile["activity_clock_points"] is not None:
        tile_m, tile_n, _, _ = _tile(kernel)
        outputs = tile_m * tile_n
        math_threads = profile["math_threads_per_cta"]
        if outputs % math_threads:
            return dataclasses.replace(inp, limitations=inp.limitations + (
                "tensor_accumulator_activity_fractional_ownership_unmodeled",))
        footprint = outputs // math_threads
        if footprint not in profile["activity_clock_points"]:
            return dataclasses.replace(inp, limitations=inp.limitations + (
                "tensor_accumulator_activity_outside_measured_footprint_domain",))
        frequency = profile["activity_clock_points"][footprint]
    ratio = frequency / hardware.nominal_sm_freq_hz
    if profile["flops_per_sm_cycle"] is not None:
        capacity = hardware.resource_by_role("compute_tensor").effective_capacity(
            inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
        ratio = (profile["flops_per_sm_cycle"] * frequency
            * hardware.num_compute_units / capacity)
    if not 0 < ratio <= 1:
        raise ValueError("Tensor activity service must be inside the configured capacity envelope")
    # Only Tensor cycle service uses this operating clock. Independently
    # measured memory byte/s and first-request ns remain in their own domain.
    return dataclasses.replace(inp, compute_demands=tuple(
        dataclasses.replace(demand, schedule_efficiency=demand.schedule_efficiency * ratio)
        if demand.resource_role == "compute_tensor" else demand
        for demand in inp.compute_demands), limitations=inp.limitations + (
            "tensor_activity_service_from_independent_probe_provisional_not_native_schedule",))


def _attach_cluster_tensor_completion_floor(inp, kernel, hardware):
    """Bound completion by whole cluster jobs without charging dummy math.

    A resident cluster processes an output-cluster job as a group. A partial
    output cluster still contains a real tile whose Tensor work must finish.
    Use the optimistic full-device cluster capacity, with one CTA per SM
    established by the staged operand allocation. This is a compute lower
    bound, not a raster or a replacement for the Load/Compute/Store pipeline.
    """
    parameters = kernel.launch.parameters
    if ("operand_transport_tma" not in parameters
            or "cluster_output_m_tiles" not in parameters
            or hardware.kernel_memory_model != "gpu_mshr"):
        return inp
    tiling = kernel.launch.tiling
    if tiling is None or tiling.split_k != 1 or tiling.pipeline_depth is None:
        return inp
    cluster_m = int(parameters["cluster_output_m_tiles"])
    cluster_n = int(parameters["cluster_output_n_tiles"])
    cluster_ctas = cluster_m * cluster_n
    if cluster_ctas == 1:
        return inp
    tile_m, tile_n, tile_k, depth = _tile(kernel)
    a_width, b_width, _ = _operand_element_bytes(kernel)
    ring_bytes = depth * tile_k * (tile_m * a_width + tile_n * b_width)
    if not hardware.max_smem_per_unit_bytes / 2 < ring_bytes <= hardware.max_smem_per_block_bytes:
        return dataclasses.replace(inp, limitations=inp.limitations + (
            "cluster_completion_single_cta_residency_unbound",))
    b, m, n, k = (kernel.shape[name] for name in ("b", "m", "n", "k"))
    if kernel.variant == "batched_gemm":
        batches, rows = b, m
    else:
        batches, rows = 1, b * m
    jobs = (batches * math.ceil(math.ceil(rows / tile_m) / cluster_m)
        * math.ceil(math.ceil(n / tile_n) / cluster_n))
    resident_clusters = min(
        hardware.num_compute_units // cluster_ctas,
        kernel.launch.grid_size // cluster_ctas)
    if resident_clusters <= 0:
        raise ValueError("Tensor cluster cannot reside in the declared physical launch")
    tensor_demands = tuple(d for d in inp.compute_demands if d.resource_role == "compute_tensor")
    if len(tensor_demands) != 1:
        raise ValueError("Tensor cluster completion requires one complete Tensor demand")
    tensor, = tensor_demands
    # Full tile work below already includes M/N tails; cluster rounds replace
    # ordinary CTA wave quantization. Retain only the intrinsic service factor.
    intrinsic_efficiency = (tensor.shape_efficiency * tensor.schedule_efficiency
        / (inp.eta * inp.wave_efficiency))
    per_sm_capacity = (hardware.resource_by_role("compute_tensor").effective_capacity(
        hardware.num_compute_units, inp.sm_freq_hz, inp.dram_freq_hz)
        / hardware.num_compute_units * intrinsic_efficiency)
    floor_ns = (math.ceil(jobs / resident_clusters) * 2 * tile_m * tile_n * k
        / per_sm_capacity * 1e9)
    previous = 0.0 if inp.execution_time_floor_ns is None else inp.execution_time_floor_ns
    limitation = "cluster_output_job_completion_lower_bound_aggregate"
    return dataclasses.replace(inp, execution_time_floor_ns=max(previous, floor_ns),
        limitations=inp.limitations if limitation in inp.limitations else (*inp.limitations, limitation))


def bf16_fp32_gemm_logical_work(kernel: KernelSpec):
    """Plain BF16 A/B and FP32 C, with no library/launch inference."""
    if kernel.variant != "gemm" or kernel.dtype != "bf16" or kernel.attributes["output_dtype"] != "fp32":
        raise ValueError("mixed-output mathematical work requires plain BF16 A/B and FP32 C")
    b, m, n, k = (kernel.shape[name] for name in ("b", "m", "n", "k"))
    read, write = float(2 * (b * m * k + k * n)), float(4 * b * m * n)
    demand = {role: 0.0 for role in (
        "compute_tensor", "compute_vector", "compute_reduction", "compute_fma", "compute_sfu", "compute_alu")}
    demand["compute_tensor"] = float(2 * b * m * n * k)
    return {"logical_bytes": read + write, "write_bytes": write, "demand": demand}


def _observed_dense_grid_input(kernel: KernelSpec, hardware):
    """Nominal resource envelope for a truthful, untiled dense observation.

    Grid/block describe the submitted kernel. They do not identify matrix
    tiles, active CTA assignments, split K, scratch extents or compiler issue.
    Only public dense mathematical work and compulsory operand bytes are
    projected; this is not a completed main/reducer or native program.
    """
    if (not HWSpec.source_contract_supported("h200_partial_source_programs", kernel.hardware_id)
            or kernel.hardware_id != hardware.hardware_id or hardware.kernel_memory_model != "gpu_mshr"
            or kernel.kind is not KernelKind.GEMM or kernel.variant != "gemm"
            or kernel.dtype not in ("bf16", "fp16") or kernel.shape["b"] != 1
            or kernel.launch.tiling is not None
            or kernel.attributes["output_dtype"] not in (None, kernel.dtype)):
        raise ValueError("observed dense grid requires its exact untiled homogeneous source contract")
    m, n, k = (kernel.shape[name] for name in ("m", "n", "k"))
    width = kernel.shape["dtype_bytes"]
    read, write = float(width * (m*k + k*n)), float(width * m*n)
    external = kernel.launch.algorithm in ("h200_vllm030_torch213_fp16_external_aggregate_gemm",
        "h200_cublaslt129_bf16_external_aggregate_gemm")
    if external:
        write = float(kernel.launch.parameters["modeled_split_k"] * m * n
            * kernel.launch.parameters["modeled_partial_bytes_per_element"])
    total = read + write
    demand = build_forward_demand(total, total, True,
        {"compute_tensor": float(2*m*n*k), "compute_fma": 0.0,
            "compute_reduction": 0.0, "compute_sfu": 0.0, "compute_alu": 0.0})
    # Unknown tile staging and cache policy do not establish these services.
    demand["l1_cache"] = demand["smem"] = demand["xbar"] = 0.0
    source_limit = ("actual_vllm030_torch213_cublasGemmEx13_1_1_FP16_call_observed_parameter_ABI_unbound"
        if kernel.launch.algorithm in ("h200_vllm030_torch213_fp16_observed_gemm",
            "h200_vllm030_torch213_fp16_external_aggregate_gemm") else
        "standalone_cublaslt129_call_not_target_runtime_binary_equivalence")
    inp = build_aggregate_gpu_input(kernel, hardware, demand,
        write_l2_bytes=write, write_dram_bytes=write, limitations=(
            *(('explicit_API_configured_split_k_model_assumption_not_observed_effective_partition',
               f'modeled_FP32_partial_writes_final_{kernel.dtype.upper()}_C_owned_by_separate_reducer',
               'aggregate_only_partial_buffer_ABI_and_CTA_K_mapping_unbound') if external else ()),
            "observed_grid_block_only_dense_mathematical_resource_projection",
            "tile_K_partition_split_count_partial_extent_and_reducer_layout_unobserved",
            "nominal_tensor_math_bound_not_observed_instruction_class_or_mma_efficiency",
            "compulsory_public_operand_bytes_only_native_replay_cache_and_scratch_traffic_unbound",
            "finite_observed_grid_not_a_CTA_to_matrix_or_K_partition_mapping",
            "compiler_residency_pipeline_overlap_and_typed_epilogue_services_unbound",
            source_limit,
            "no_tile_or_split_inference_new_rate_or_production_duration_fit",
            "partial_envelope_not_complete_main_reducer_or_accuracy_acceptance"))
    return dataclasses.replace(inp, support_level="partial", dram_capacity_multiplier=1.0), total


def _model(
        kernel: KernelSpec, hardware, *,
        cold_traffic: bool = False,
        ) -> Tuple[PredictionInput, float]:
    if kernel.launch is not None and kernel.launch.algorithm == "h200_cublaslt129_bf16_external_aggregate_gemm":
        kernel.validate_legacy_bf16_external_launch(hardware)
    if kernel.launch is not None and kernel.launch.algorithm in (
            "h200_vllm030_torch213_fp16_observed_gemm",
            "h200_vllm030_torch213_fp16_external_aggregate_gemm"):
        kernel.validate_observed_gemm_launch(hardware)
    from src.kernel.memory_access_builder import memory_execution_kernel
    if kernel.variant not in ("moe_router", "batched_gemm"):
        kernel = memory_execution_kernel(kernel, hardware)
    if (kernel.launch is not None and kernel.launch.algorithm in (
            "h200_cublaslt129_injected_gemm", "h200_vllm030_torch213_fp16_observed_gemm",
            "h200_vllm030_torch213_fp16_external_aggregate_gemm", "h200_cublaslt129_bf16_external_aggregate_gemm")
            and kernel.launch.tiling is None):
        return _observed_dense_grid_input(kernel, hardware)
    if (kernel.attributes["output_dtype"] is not None
            and kernel.attributes["output_dtype"] != kernel.dtype):
        # vLLM 0.30 V4 attention.py torch.mm(out_dtype=torch.float32):
        # BF16 A/B, FP32 accumulator/output. Unique tensor bytes only;
        # no transplant of homogeneous-output pipeline/epilogue timings.
        work = bf16_fp32_gemm_logical_work(kernel)
        total, write, flops = work["logical_bytes"], work["write_bytes"], work["demand"]
        demand = build_forward_demand(total, total, False, flops)
        # The generic tensor template's 2x local SMEM law is not bound to
        # this mixed-output launch. Publish only the known global objects.
        demand["smem"] = 0.0
        inp = build_aggregate_gpu_input(
            kernel, hardware, demand,
            write_l2_bytes=write, write_dram_bytes=write,
            limitations=(
                "bf16_inputs_fp32_output_roofline_only",
                "local_operand_and_epilogue_services_unpriced",
                "mixed_output_actual_cublas_launch_not_bound",
                "logical_unique_tensor_bytes_not_observed_dram",
                "operand_replay_and_fp32_store_pipeline_unmodeled",
                "production_duration_calibration_not_used",
            ))
        return inp, total
    shape = kernel.shape
    b, m, n, k = (int(shape[name]) for name in ("b", "m", "n", "k"))
    variant = kernel.variant.lower()
    memory_model = hardware.kernel_memory_model
    if variant not in ("gemm", "gemv", "batched_gemm", "moe_router"):
        raise ValueError(
            "GEMM variant must be gemm, gemv, batched_gemm, or moe_router")
    a_element_bytes, b_element_bytes, c_element_bytes = (
        _operand_element_bytes(kernel))
    element_bytes = int(kernel.shape["dtype_bytes"])
    m_total = b * m
    logical_program = None
    from src.kernel.memory_access_builder import has_public_tensor_gemm_source
    public_gemm_work=None
    if has_public_tensor_gemm_source(kernel,hardware):
        from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
        from src.kernel.memory_path_solver import public_gemm_source_work
        logical_program=gemm_memory_access_program(kernel,hardware)
        public_gemm_work=public_gemm_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
        b,m,n,k,a_element_bytes,b_element_bytes,c_element_bytes=public_gemm_work['geometry']
        m_total=b*m
    from src.kernel.memory_router import has_public_router_abi,public_router_source_work
    if has_public_router_abi(kernel):
        from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
        logical_program=gemm_memory_access_program(kernel,hardware)
        public_source=public_router_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
        m_total,n,k=public_source['logical_rows'],public_source['experts'],public_source['hidden']
        a_element_bytes,b_element_bytes,c_element_bytes=public_source['operand_widths']
    from src.kernel.memory_access_builder import has_gpu_nonsplit_gemm_source
    from src.kernel.memory_cann_gemm import has_cann_gemm_source, bound_cann_gemm_source, cann_gemm_source_demands
    unified_gpu_gemm = has_gpu_nonsplit_gemm_source(kernel, hardware)
    unified_cann_gemm = has_cann_gemm_source(kernel)
    from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_source,bound_gpu_split_gemm_source,gpu_split_gemm_source_demands
    split_source = None
    if has_gpu_split_gemm_source(kernel,hardware):
        bound = bound_gpu_split_gemm_source(kernel)
        if bound is None:
            from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
            split_source = gpu_split_gemm_source_demands(gemm_memory_access_program(kernel,hardware),expected_kernel=kernel)
        else:
            _,split_source = bound
        b,m,n,k,a_element_bytes,b_element_bytes,c_element_bytes = split_source['operand_geometry']
        m_total = b*m
    elif unified_cann_gemm:
        bound = bound_cann_gemm_source(kernel)
        if bound is None:
            from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
            logical_program = gemm_memory_access_program(kernel,hardware)
            work = cann_gemm_source_demands(logical_program)
        else:
            logical_program,work = bound
        b,m,n,k = work['geometry']
        m_total = b*m
    elif (unified_gpu_gemm or variant == "gemv"
            or (variant == "gemm" and memory_model in ("pim_bank", "systolic_scratchpad"))):
        from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
        logical_program = gemm_memory_access_program(kernel, hardware)
        if unified_gpu_gemm and variant=='moe_router':
            from src.kernel.memory_router import nonsplit_router_logical_price_program
            logical_program=nonsplit_router_logical_price_program(logical_program,expected_kernel=kernel,hardware=hardware)
        if unified_gpu_gemm and variant=='batched_gemm':
            from src.kernel.memory_access_builder import batched_gemm_logical_price_program
            logical_program=batched_gemm_logical_price_program(logical_program,expected_kernel=kernel)
    is_non_tensor = variant == "gemv" or _is_simt_router(kernel)
    is_observed_grid = _is_observed_grid_router(kernel)
    from src.kernel.memory_router import has_observed_router_abi,observed_router_source_work
    observed_router_work=None
    if has_observed_router_abi(kernel):
        from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
        observed_router_work=observed_router_source_work(gemm_memory_access_program(kernel,hardware),expected_kernel=kernel,hardware=hardware,
            legacy_split_lower_bound=_UNOBSERVED_SPLIT_K_LOWER_BOUND)
    split_k_program = _cann_gemm_tile_pipeline_program(kernel)
    split_k_traffic = _cann_gemm_tile_pipeline_traffic(
        kernel,
        l2_replacement_capacity_bytes=(
            int(hardware.l2_replacement_capacity_bytes)
            if cold_traffic and memory_model == "npu_tile" else None),
    )
    program_traffic = (
        _cann_gemm_region_program_traffic(kernel, hardware)
        if memory_model == "npu_tile" and not is_non_tensor else None)
    if (program_traffic is None and memory_model == "npu_tile"
            and not is_non_tensor):
        program_traffic = _cann_gemm_persistent_operand_traffic(
            kernel, hardware)
    # Keep the provenance-bound warmup context for source scheduling, but
    # retain the measured cold ingress contract for reported traffic. The
    # canonical Timeline observation executes warmups before timing while its
    # traffic capture still records the physical operand ingress; separating
    # these two owner contracts avoids turning a timing context into a traffic
    # lookup.
    traffic_program = program_traffic
    if (cold_traffic
            and program_traffic is not None
            and kernel.l2_execution_context.same_program_operand_replays
            and _uses_cann_persistent_program(kernel)):
        cold_context = dataclasses.replace(
            kernel.l2_execution_context, same_program_operand_replays=0)
        cold_kernel = dataclasses.replace(
            kernel, l2_execution_context=cold_context)
        traffic_program = _cann_gemm_persistent_operand_traffic(
            cold_kernel, hardware)
    tiling = kernel.launch.tiling
    if tiling is None and is_non_tensor:
        split_k = 1
    elif tiling is None and is_observed_grid:
        split_k = (_UNOBSERVED_SPLIT_K_LOWER_BOUND if observed_router_work is None else observed_router_work['legacy_pricing_split_lower_bound'])
    elif tiling is None:
        raise ValueError("tensor GEMM requires retained tiling")
    elif tiling.split_k is None and is_non_tensor:
        split_k = 1
    elif tiling.split_k is not None:
        split_k = tiling.split_k
    else:
        raise ValueError("tensor GEMM requires split_k in retained tiling")
    if split_source is not None:
        split_k = split_source['split_k']
    grid_z = split_k
    is_mixed_aic_aiv = (
        not is_non_tensor
        and memory_model == "npu_tile"
        and kernel.launch.parameters["aiv_mixed"] == 1.0)
    mcblas_padded_flops = None
    if is_non_tensor and variant == "moe_router":
        # dsv3_router_gemm: each expert CTA streams its weight row once and
        # re-reads every token row (L2-resident after the first CTAs) for the
        # batch it reduces; outputs are one element per (token, expert).
        from src.kernel.memory_router import has_simt_router_source,simt_router_source_work
        if has_simt_router_source(kernel,hardware):
            from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
            source=simt_router_source_work(gemm_memory_access_program(kernel,hardware),expected_kernel=kernel,hardware=hardware)
            logical_projection=source['legacy_logical_projection']
            weight_bytes=float(logical_projection['weight_read_bytes']);output_bytes=float(logical_projection['output_write_bytes'])
            l2_total=weight_bytes+float(logical_projection['activation_read_bytes'])+output_bytes
            dram_bytes=weight_bytes+float(logical_projection['activation_unique_bytes'])+output_bytes
        else:
            source=public_router_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
            l2_total=float(source['simt_l2_bytes'])
            dram_bytes=float(source['simt_dram_bytes'])
        flops = {
            "compute_tensor": 0.0,
            "compute_fma": 2.0 * m_total * k * n,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
        }
        compulsory = dram_bytes
    elif is_observed_grid:
        # Every operand element reaches at least one CTA and every K split
        # publishes one FP32 partial matrix for the separate reducer; the
        # tile-dependent operand replay is unobserved, so L2 ingress is its
        # compulsory lower bound.  Partials stay L2-resident (no DRAM width).
        if observed_router_work is None:
            source=public_router_source_work(logical_program,expected_kernel=kernel,hardware=hardware)
            operand_bytes=float(source['input_bytes'])
            if source['partial_bytes_per_class'] is None:
                raise ValueError('Observed-grid router requires its declared partial output source')
            partial_bytes=float(split_k*source['partial_bytes_per_class'])
        else:
            operand_bytes=float(observed_router_work['legacy_operand_read_bytes'])
            partial_bytes=float(observed_router_work['legacy_partial_write_lower_bound_bytes'])
        l2_total = operand_bytes + partial_bytes
        dram_bytes = operand_bytes
        flops = {
            "compute_tensor": 2.0 * m_total * k * n,
            "compute_fma": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
        }
        compulsory = operand_bytes
    elif is_non_tensor:
        from src.kernel.memory_access import Operation
        from src.kernel.memory_path_solver import logical_unique_bytes
        l2_total = float(logical_unique_bytes(logical_program, Operation.READ)
                         + logical_unique_bytes(logical_program, Operation.WRITE))
        dram_bytes = l2_total
        # CANN's canonical 910B GEMV launch is still ``cann_matmul_v3`` on
        # the AIC domain. Its scheduling macro tile is composed from
        # 16x16x16 Cube instructions; the instruction shape itself is not the
        # launch tile. Pricing that launch on the placeholder AIC scalar-FMA
        # role contradicts the selector's static execution contract. GPU GEMV
        # remains a SIMT FMA workload.
        if memory_model == "npu_tile":
            tensor_flops = 2.0 * k * n
            fma_flops = 0.0
        elif memory_model in (
                "gpu_mshr", "pim_bank", "systolic_scratchpad"):
            tensor_flops = 0.0
            fma_flops = 2.0 * k * n
        else:
            raise ValueError(
                f"unknown kernel_memory_model: {memory_model!r}")
        flops = {
            "compute_tensor": tensor_flops,
            "compute_fma": fma_flops,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
        }
        compulsory = l2_total
    else:
        if memory_model == "gpu_mshr":
            l2_total, dram_bytes = _tensor_traffic(kernel, hardware, logical_program,split_source=split_source)
        elif memory_model == "npu_tile" and split_k_traffic is not None:
            # Construct the AIC operand ingress first.  The source-distinct
            # AIV workspace ingress and both write paths are attached after
            # the structured contract exists.
            l2_total = split_k_traffic.operand_l2_read_bytes
            dram_bytes = split_k_traffic.operand_dram_read_bytes
        elif memory_model == "npu_tile" and traffic_program is not None:
            l2_total = float(traffic_program.l2_bytes)
            dram_bytes = float(traffic_program.dram_bytes)
        elif memory_model in (
                "npu_tile", "pim_bank", "systolic_scratchpad"):
            l2_total, dram_bytes = _non_gpu_tensor_traffic(kernel, hardware, logical_program)
        else:
            raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
        if variant == "moe_router":
            router_operations = float(m_total * n)
        elif variant in ("gemm", "batched_gemm"):
            router_operations = 0.0
        else:
            raise AssertionError(
                f"unhandled tensor GEMM variant: {variant!r}")
        tensor_flops = 2.0 * m_total * k * n
        logical_k_tiles, issued_k_tiles = _uniform_split_k_padding(kernel)
        if issued_k_tiles:
            tensor_flops *= issued_k_tiles / logical_k_tiles
        mcblas_padded_flops = _mcblas_non_split_padded_tensor_flops(kernel)
        if mcblas_padded_flops is not None:
            tensor_flops = mcblas_padded_flops
        if split_k_program is None:
            split_k_vector_flops = 0.0
        elif (split_k_program.program_kind
              == "cann_gemm_deterministic_split_k_v1"):
            split_k_vector_flops = float(
                m_total * n * split_k_program.aic_workers)
        elif (split_k_program.program_kind
              == "cann_gemm_single_core_split_k_v1"):
            split_k_vector_flops = float(m_total * n)
        else:
            raise ValueError("unknown CANN GEMM source program kind")
        flops = {
            "compute_tensor": tensor_flops,
            "compute_vector": split_k_vector_flops,
            "compute_fma": 0.0,
            "compute_alu": router_operations,
            "compute_sfu": router_operations,
            "compute_reduction": _sm80_depth3_source_barrier_cycles(
                kernel, hardware),
            "warp_issue": (
                _sm80_depth5_split1_non_hmma_warp_instructions(
                    kernel, hardware)
                + _sm80_depth5_split2_non_hmma_warp_instructions(
                    kernel, hardware)
                + _sm80_depth3_split_epilogue_warp_instructions(
                    kernel, hardware)
                + _sm80_depth3_split2_non_hmma_warp_instructions(
                    kernel, hardware)
                + _sm80_depth5_fixed_non_hmma_warp_instructions(
                    kernel, hardware)
                + _sm80_depth6_fixed_non_hmma_warp_instructions(
                    kernel, hardware)),
        }
        if split_k_traffic is not None:
            compulsory = split_k_traffic.dram_bytes
        elif (split_k > 1
                and kernel.launch.parameters[
                    "split_k_partial_bytes_per_element"] > 0):
            if split_source is not None:
                operand_bytes=split_source['operand_read_bytes'];partial_bytes=split_source['write_bytes']
            elif public_gemm_work is not None:
                operand_bytes=sum(public_gemm_work['nominal_operand_bytes'][:2]);partial_bytes=public_gemm_work['declared_partial_bytes']
            elif has_public_router_abi(kernel):
                operand_bytes=public_source['input_bytes'];partial_bytes=split_k*public_source['partial_bytes_per_class']
            else:raise ValueError('Split GEMM requires its operand and partial source')
            compulsory=float(operand_bytes+partial_bytes*kernel.launch.parameters['split_k_partial_dram_bytes_per_element']
                /kernel.launch.parameters['split_k_partial_bytes_per_element'])
        elif split_source is not None:
            compulsory = float(split_source['operand_read_bytes']+split_source['output_bytes'])
        elif (unified_gpu_gemm or unified_cann_gemm or (variant == "gemm" and memory_model in ("pim_bank", "systolic_scratchpad"))):
            # No split workspace enters this compulsory interface. Tile
            # replay/cache traffic remains a separate source projection.
            from src.kernel.memory_access import Operation
            from src.kernel.memory_path_solver import logical_requested_bytes,logical_unique_bytes
            from src.kernel.memory_access_builder import MemoryAccessWork
            projection=logical_unique_bytes if isinstance(logical_program,MemoryAccessWork) or logical_program.tile_programs else logical_requested_bytes
            if unified_cann_gemm:
                from src.kernel.memory_cann_gemm import cann_gemm_source_demands
                work = cann_gemm_source_demands(logical_program)
                compulsory = float(work['read_bytes'] + work['write_bytes'])
            else:
                compulsory = float(projection(logical_program, Operation.READ)
                                   + projection(logical_program, Operation.WRITE))
        elif public_gemm_work is not None:
            compulsory=float(sum(public_gemm_work['nominal_operand_bytes']))
        elif has_public_router_abi(kernel):
            compulsory=float(sum(public_source['nominal_operand_bytes']))
        else:
            raise ValueError('GEMM compulsory demand requires its unified operand source')
    demand = build_forward_demand(l2_total, dram_bytes, is_non_tensor, flops)
    logical_k_tiles, issued_k_tiles = _uniform_split_k_padding(kernel)
    if issued_k_tiles > logical_k_tiles:
        # Zero-filled K padding writes and reads complete local operand
        # panels, but does not read beyond the valid global tensor extents.
        tile_m, tile_n, tile_k, _ = _tile(kernel)
        output_tiles = _kernel_logical_workgroups(kernel, tile_m, tile_n, 1)
        demand["smem"] += (2 * output_tiles * (issued_k_tiles - logical_k_tiles)
            * tile_k * (tile_m * a_element_bytes + tile_n * b_element_bytes))
    if "operand_transport_tma" in kernel.launch.parameters:
        tile_m, tile_n, _, _ = _tile(kernel)
        batches, rows = (b, m) if variant == "batched_gemm" else (1, m_total)
        # The existing local projection uses min(rows, tile_m) for A. When
        # the whole problem is shorter than one tile, that omits the shared
        # cells written by TMA zero fill and read by the full Tensor tile.
        # This only corrects that whole-problem tail; retain the established
        # treatment of other boundary tiles and cluster restoration below.
        # Uniform K padding above separately owns its complete extra panels;
        # only the logical K extent needs this M-tail correction.
        demand["smem"] += (2 * batches * math.ceil(rows / tile_m)
            * math.ceil(n / tile_n) * (tile_m - min(rows, tile_m))
            * k * a_element_bytes)
    cluster_reads = _cluster_operand_read_bytes(kernel, (
        b, m, n, k, a_element_bytes, b_element_bytes, c_element_bytes))
    if cluster_reads is not None:
        unshared, shared = cluster_reads
        # Multicast/reuse saves global ingress, but every consumer still
        # receives and reads its local operand tile from shared memory.
        demand["smem"] += 2 * (unshared - shared + _cluster_output_padding_bytes(kernel, (
            b, m, n, k, a_element_bytes, b_element_bytes, c_element_bytes)))
    if split_k_traffic is not None:
        demand["mte3"] = split_k_traffic.output_l2_write_bytes
    if (traffic_program is not None
            and traffic_program.dirty_writeback_bytes > 0):
        validated_l2_miss_rate(
            traffic_program.a_dram_bytes + traffic_program.b_dram_bytes,
            traffic_program.a_l2_bytes + traffic_program.b_l2_bytes)
    else:
        validated_l2_miss_rate(dram_bytes, l2_total)
    if not is_non_tensor and dram_bytes > 0 and demand["xbar"] > dram_bytes:
        demand["xbar"] = dram_bytes

    logical_task_count, wave_efficiency, active_units = (
        _logical_execution_geometry(kernel, hardware))
    cann_program = _cann_gemm_launch_program(kernel)
    if memory_model == "npu_tile" and cann_program is not None:
        # A normalized CANN program records the launched physical AIC domain.
        # Semantic output-tile count is not an alternative occupancy estimate:
        # persistent tail shifting can deliberately keep fewer (or more)
        # workers active than min(logical tiles, device AICs).
        logical_task_count = cann_program.aic_workers
        active_units = cann_program.aic_workers
        wave_efficiency = 1.0
    gemv_wave8_completion = (
        _metax_gemv_wave8_completion(kernel, hardware)
        if is_non_tensor else None)
    if gemv_wave8_completion is not None:
        _, completion_cohorts = gemv_wave8_completion
        demand["compute_reduction"] = float(completion_cohorts)
    eta = FULL_RESOURCE_OCCUPANCY
    gpu_memory_model = hardware.kernel_memory_model == "gpu_mshr"
    if not is_non_tensor and not is_observed_grid and gpu_memory_model:
        tile_m, tile_n, _, _ = _tile(kernel)
        # Tensor instructions execute the complete retained M tile even when
        # the final semantic row range is predicated.  Account for that tail
        # in every output-tile row, not only when the whole launch fits in its
        # first tile.
        # Independent BMM problems cannot share the unused rows of their
        # boundary tiles. Both output dimensions consume the fixed Tensor
        # tile; predication suppresses publication, not the padded arithmetic.
        rows_per_problem = m if variant == "batched_gemm" else m_total
        utilization = (
            rows_per_problem / (math.ceil(rows_per_problem / tile_m) * tile_m)
            * n / (math.ceil(n / tile_n) * tile_n))
        if mcblas_padded_flops is None:
            eta *= utilization

    if memory_model == "npu_tile":
        launch_task_kind = "mixed" if is_mixed_aic_aiv else "aic"
        include_launch_floor, launch_limitations = (
            npu_launch_floor_contract(hardware, launch_task_kind))
        support_level = "aggregate_only"
        limitations = (() if (
            not is_mixed_aic_aiv or split_k_program is not None) else (
                "aiv_mixed_operation_decomposition_unavailable",
        )) + (
            "aic_operand_replay_evidence_unavailable",
            "aic_store_fixpipe_tile_lifecycle_unavailable",
            "aic_read_write_transaction_split_unavailable",
            "mte_transaction_geometry_unavailable",
            "memory_first_request_latency_unavailable",
            "mte2_descriptor_and_credit_evidence_unavailable",
            "workload_producer_independence_unavailable",
        ) + launch_limitations
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        launch_task_kind = "generic"
        include_launch_floor = True
        support_level = "compatibility"
        limitations = ("outside_a800_c500_910b_acceptance_scope",)
    elif memory_model == "gpu_mshr":
        launch_task_kind = "generic"
        include_launch_floor = True
        tiling = kernel.launch.tiling
        pipeline_depth = None if tiling is None else tiling.pipeline_depth
        analytical_gemv_stream = (
            is_non_tensor
            and _uses_a800_analytical_gemv_stream(kernel, hardware))
        gemv_pipeline = (
            _gpu_gemv_dependency_pipeline(kernel, hardware)
            if (is_non_tensor
                and (pipeline_depth is None or analytical_gemv_stream))
            else None)
        if analytical_gemv_stream:
            # A dependency path already owns the complete GEMV load/FMA/store
            # sequence.  The analytical launch's tile depth describes the
            # steady stream and must not create a second workload-sized
            # fill/drain pipeline.
            pipeline_depth = None
        external_split_k_pipeline = (
            None if is_non_tensor or is_observed_grid else
            _gpu_external_split_k_cohort_pipeline(kernel, hardware))
        unstaged_single_wave_pipeline = (
            None if is_non_tensor or is_observed_grid else
            _gpu_unstaged_single_wave_dependency_pipeline(kernel, hardware))
        mcblas_rolling_pipeline = (
            None if is_non_tensor or is_observed_grid else
            _mcblas_tile_pipeline(
                kernel, hardware, active_units))
        if (not is_non_tensor and not is_observed_grid
                and mcblas_rolling_pipeline is None):
            mcblas_rolling_pipeline = _mcblas_oversized_rolling_pipeline(
                kernel, hardware, active_units)
        mcblas_subtile_pipeline = (
            None if (is_non_tensor or is_observed_grid
                     or mcblas_rolling_pipeline is not None) else
            _mcblas_subtile_dependency_pipeline(
                kernel, hardware, active_units))
        sm80_sass_dependency = None
        if not is_non_tensor and not is_observed_grid:
            sm80_sass_dependency = (
                _sm80_depth5_split2_sass_dependency(kernel, hardware)
                or _sm80_depth3_transposed_split_source_dependency(
                    kernel, hardware)
                or _sm80_depth3_split_sass_dependency(kernel, hardware)
                or _sm80_depth3_non_split_sass_dependency(kernel, hardware))
        partial_element_bytes = kernel.launch.parameters[
            "split_k_partial_bytes_per_element"]
        if split_source is not None:
            partial_element_bytes = float(split_source['partial_element_bytes'])
        partial_dram_element_bytes = kernel.launch.parameters[
            "split_k_partial_dram_bytes_per_element"]
        if split_k == 1:
            split_limitations = ()
        elif partial_element_bytes > 0:
            if partial_dram_element_bytes > 0:
                split_limitations = ()
            else:
                split_limitations = (
                    "split_k_partial_dram_residency_unavailable",
                )
            if external_split_k_pipeline is not None:
                pipeline_depth = None
        elif pipeline_depth is not None:
            split_limitations = ()
        else:
            split_limitations = (
                "split_k_internal_reduction_lifecycle_unavailable",
            )
            # A tile pipeline cannot distribute final-output ownership over
            # all K slices without the internal synchronization contract.
            pipeline_depth = None
        if sm80_sass_dependency is not None:
            # The exact SASS path already owns the complete ordered
            # Load/Tensor/Store program and its finite operand prologue.
            pipeline_depth = None
        if (pipeline_depth is None
                and gemv_pipeline is None
                and external_split_k_pipeline is None
                and mcblas_rolling_pipeline is None
                and mcblas_subtile_pipeline is None):
            support_level = "aggregate_only"
            limitations = gpu_partial_limitations(
                hardware, "gemm_pipeline_overlap_evidence_unavailable",
                *split_limitations)
        else:
            support_level = "partial"
            limitations = gpu_partial_limitations(
                hardware, *split_limitations)
        if is_observed_grid:
            limitations += _OBSERVED_GRID_LIMITATIONS + (
                "production_duration_calibration_not_used",
            )
        elif variant in ("batched_gemm", "moe_router"):
            limitations += (
                "target_gemm_kernel_schedule_projected_from_bf16_gemm",
                "production_duration_calibration_not_used",
            )
    else:
        raise AssertionError(f"unhandled GEMM memory model: {memory_model!r}")
    if kernel.launch.algorithm == "h200_cublaslt129_injected_gemm":
        limitations += (
            "launch_observed_cublaslt_12_9_rank0_32mib_workspace",
            "launch_target_serving_runtime_equivalence_unverified",
            "hopper_cluster_tma_persistent_task_mapping_unmodeled",
        )
    if public_gemm_work is not None:
        limitations+=('gemm_input_storage_unbound_nominal_width_projection',)
    if issued_k_tiles > logical_k_tiles:
        limitations += (
            "uniform_split_k_zero_tail_global_ingress_averaged_across_stages",)
    if cluster_reads is not None:
        limitations += (
            "cluster_invariant_operand_reuse_aggregate_not_native_descriptor_mapping",
            "cluster_multicast_fabric_service_unmodeled",
        )
        if (memory_model == "gpu_mshr" and split_k > 1
                and partial_element_bytes > 0
                and "operand_transport_tma" in kernel.launch.parameters):
            limitations += ("split_k_operand_lru_n_major_aggregate_order_not_native_raster",)
        if (isinstance(kernel.launch.source_program, GpuCtaRasterProgram)
                and kernel.launch.source_program.is_cluster_grouped):
            limitations += (
                "cluster_grouped_source_raster_macro_tile_cache_approximation",)
        else:
            limitations += (
                "tma_global_tails_cropped_padded_cache_occupancy_approximation",)
    if launch_task_kind in ("aic", "aiv", "mixed"):
        task_floor_grid_size = active_units
        task_floor_block_size = 1
    elif launch_task_kind == "generic":
        task_floor_grid_size = logical_task_count
        task_floor_block_size = hardware.parallel_width
    else:
        raise ValueError(
            f"unsupported GEMM launch task kind: {launch_task_kind!r}")
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=task_floor_grid_size,
        block_size=task_floor_block_size,
        launch_grid=(task_floor_grid_size, 1, 1),
        launch_block=(task_floor_block_size, 1, 1),
        variant=variant,
        mem_model=memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind=launch_task_kind,
        include_dispatch_overhead=include_launch_floor,
        dram_capacity_multiplier=(
            hardware.dram_protocol_efficiency_l1cached),
        support_level=support_level,
        limitations=limitations,
    )
    normalized = _architecture_input(inp, kernel, hardware, logical_program)
    if memory_model == "gpu_mshr":
        if is_non_tensor:
            from src.kernel.memory_access import Operation
            from src.kernel.memory_path_solver import logical_requested_bytes
            if variant == "gemv":
                gemv_output_bytes = float(logical_requested_bytes(logical_program, Operation.WRITE))
            elif has_public_router_abi(kernel):
                gemv_output_bytes=float(public_source['nominal_operand_bytes'][2])
            else:
                gemv_output_bytes=float(source['legacy_logical_projection']['output_write_bytes'])
            producers, requests_per_producer, frontend_rate = (
                _gpu_gemv_memory_source(
                    kernel, hardware, l2_total - gemv_output_bytes,
                    active_units))
        else:
            producers, requests_per_producer = gpu_memory_source(hardware)
            frontend_rate = None
        transaction_bytes = None
        path_name = "global_memory"
        # The hardware LSU ceiling is supplied by MemoryHierarchy.  Compute
        # progress and Load/Compute overlap belong to the explicit pipeline;
        # folding them into a second workload-specific frontend rate would
        # price the same dependency twice.
    elif memory_model == "npu_tile":
        transaction_bytes = None
        producers, requests_per_producer = 1.0, None
        path_name = "mte2_read"
        frontend_rate = None
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        transaction_bytes = None
        producers, requests_per_producer = 1.0, 1.0
        path_name = None
        frontend_rate = None
    else:
        raise AssertionError(f"unhandled GEMM memory model: {memory_model!r}")
    structured = attach_structured_contract(
        normalized,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name=path_name,
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=frontend_rate,
    )
    if (_uses_cann_persistent_program(kernel)
            and traffic_program is not None):
        structured = _attach_cann_gemm_persistent_cache_paths(
            structured, traffic_program)
    if gemv_wave8_completion is not None:
        completion_ns, _ = gemv_wave8_completion
        structured = dataclasses.replace(
            structured,
            resource_time_overrides_ns={
                **structured.resource_time_overrides_ns,
                "compute_reduction": completion_ns,
            },
        )
    if not is_non_tensor and not is_observed_grid:
        structured = _apply_gemm_bf16_mma_saturation(
            structured, kernel, hardware)
        structured = _apply_gemm_tensor_activity_clock(structured, kernel, hardware)
        structured = _attach_cluster_tensor_completion_floor(structured, kernel, hardware)
        structured = _apply_sm80_depth3_split_epilogue_issue_schedule(
            structured, kernel, hardware)
        structured = _apply_sm80_depth5_fixed_issue_schedule(
            structured, kernel, hardware)
        structured = _apply_sm80_depth6_fixed_issue_schedule(
            structured, kernel, hardware)
    if split_k_traffic is not None:
        structured = _attach_cann_deterministic_split_k_memory_paths(
            structured, kernel, hardware, split_k_traffic)
        structured = _attach_cann_deterministic_split_k_vector_service(
            structured, kernel)
    if (_uses_cann_region_program(kernel)
            or kernel.launch.algorithm == _CANN_MATMUL_V3_ASW_65536_ALGORITHM):
        structured = _attach_cann_gemm_operand_service(
            structured, kernel, hardware, program_traffic)
    if memory_model == "gpu_mshr":
        if split_source is not None:
            output_bytes=float(split_source['output_bytes'])
        elif public_gemm_work is not None:
            output_bytes=float(public_gemm_work['nominal_operand_bytes'][2])
        elif has_public_router_abi(kernel):
            output_bytes=float(public_source['nominal_operand_bytes'][2])
        elif logical_program is not None:
            from src.kernel.memory_access import Operation
            from src.kernel.memory_path_solver import logical_requested_bytes
            output_bytes=float(logical_requested_bytes(logical_program,Operation.WRITE))
        elif is_non_tensor:
            output_bytes=float(source['legacy_logical_projection']['output_write_bytes'])
        elif observed_router_work is not None:
            output_bytes=float(observed_router_work['legacy_partial_write_lower_bound_bytes'])/split_k
        else:raise ValueError('GEMM writes require their unified source')
        partial_write_width = (float(split_source['partial_element_bytes']) if split_source is not None
            else kernel.launch.parameters['split_k_partial_bytes_per_element'])
        if split_k > 1 and partial_write_width > 0:
            if split_source is not None:write_l2=float(split_source['write_bytes'])
            elif public_gemm_work is not None:write_l2=float(public_gemm_work['declared_partial_bytes'])
            elif observed_router_work is not None:write_l2=float(observed_router_work['legacy_partial_write_lower_bound_bytes'])
            else:write_l2=float(split_k*public_source['partial_bytes_per_class'])
            write_dram=write_l2*kernel.launch.parameters['split_k_partial_dram_bytes_per_element']/partial_write_width
        elif (grid_z > 1
              and kernel.launch.algorithm not in _MCBLAS_GEMM_ALGORITHMS):
            write_l2 = grid_z * output_bytes
            write_dram = output_bytes
        else:
            write_l2 = min(output_bytes, l2_total)
            write_dram = min(output_bytes, dram_bytes)
        structured = split_global_memory_paths(
            structured,
            read_l2_bytes=l2_total - write_l2,
            read_dram_bytes=dram_bytes - write_dram,
            write_l2_bytes=write_l2,
            write_dram_bytes=write_dram,
            dependency_paths=tuple(
                path for path in (
                    gemv_pipeline, external_split_k_pipeline,
                    unstaged_single_wave_pipeline, mcblas_subtile_pipeline,
                    sm80_sass_dependency)
                if path is not None),
        )
        structured = _attach_sm80_depth3_operand_source_service(
            structured, kernel, hardware)
        structured = _attach_finite_async_operand_stream_access_service(
            structured, kernel, hardware)
        structured = _attach_gpu_k32_deep_cohort_operand_streams(
            structured, kernel, hardware)
        structured = _attach_sm80_staged_ldgsts_l2_service(
            structured, kernel, hardware)
        compute_role = "compute_fma" if is_non_tensor else "compute_tensor"
        if mcblas_rolling_pipeline is not None:
            structured = attach_pipeline_work_plan(
                structured, hardware, mcblas_rolling_pipeline)
            # The ramp is a completion lower bound that composes by maximum, so
            # it floors this source-derived pipeline rather than replacing it.
            # Without this the mechanism is unreachable on a device whose GEMM
            # launches all take the rolling-pipeline branch, which is every
            # mcBLAS launch on C500.
            structured = _attach_dram_finite_stream_ramp_floor(
                structured, kernel, hardware)
        elif pipeline_depth is not None:
            if _uses_k_tiled_pipeline(kernel):
                structured = _attach_k_tiled_pipeline(
                    structured, kernel, hardware,
                    compute_role=compute_role)
                structured = _attach_finite_async_operand_stream_completion_floor(
                    structured, kernel, hardware)
                structured = _attach_finite_mixed_operand_stream_startup(
                    structured, kernel, hardware)
                structured = _attach_dram_finite_stream_ramp_floor(
                    structured, kernel, hardware)
            else:
                structured = attach_uniform_pipeline(
                    structured,
                    hardware,
                    task_count=logical_task_count,
                    stages=(
                        ("load", "memory:global_read", True),
                        ("compute", f"compute:{compute_role}", False),
                        ("store", "memory:global_write", True),
                    ),
                    buffer_slots=pipeline_depth,
                )
    elif memory_model == "npu_tile":
        # All retained 910B tensor GEMM tiles use the same compressed
        # AL0/BL0 load-to-Cube program.  The terminal FixPipe store lifetime
        # remains intentionally aggregate until its static contract is known.
        structured = _attach_cann_npu_gemm_pipeline(
            structured, kernel, hardware, program_traffic, split_k_traffic)
        structured = _attach_cann_npu_mixed_gemm_cold_stream_constraint(
            structured, kernel, hardware)
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        pass
    else:
        raise AssertionError(
            f"unhandled GEMM memory model: {memory_model!r}")
    return structured, compulsory


class GEMMPredictor(KernelPredictor):
    kind = KernelKind.GEMM

    @staticmethod
    def _mathematical_prediction(hardware, latency_ns):
        if HWSpec.source_contract_supported("programmed_mxfp4_fp8_expert_service", hardware.hardware_id):
            latency_ns += hardware.fpga_execution["hbm_latency_ns"]
        return prediction_from_ns(latency_ns, "roofline")

    def analytical_without_launch(self, kernel: KernelSpec, hardware):
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in (
                    "dense_fp16_source_flops_bytes_v1", "unquantized_fp16_flops_bytes_v1")):
            raise NotImplementedError("dense FP16 mathematical work has no native launch")
        prediction = self.roofline_without_launch(kernel, hardware)
        return dataclasses.replace(prediction, bottleneck_group="analytical_gemm_source")

    def roofline_without_launch(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "gemm" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "packed_fp4_source_v1"):
            return self._packed_fp4_source_prediction(kernel, hardware)
        if (kernel.variant == "batched_gemm" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "batched_fp8_quantized_input_fp32_scales_v1"):
            b, m, n, k = (kernel.shape[key] for key in ("b", "m", "n", "k"))
            logical_bytes = b * (m * k + 4 * m * (k // 128) + k * n + 4 * (k // 128) * (n // 128) + 2 * m * n)
            seconds = max(2 * b * m * n * k / hardware.resource_by_role("compute_tensor_fp8").capacity,
                logical_bytes / hardware.resource_by_role("dram").capacity)
            return prediction_from_ns(seconds * 1e9, "roofline")
        if (kernel.variant == "gemm"
                and kernel.dtype == "bf16" and kernel.attributes["output_dtype"] == "fp32"):
            work = bf16_fp32_gemm_logical_work(kernel)
            return prediction_from_ns(roofline_ns(work["demand"], hardware, work["logical_bytes"]), "roofline")
        if kernel.variant == "gemm" and "mathematical_program" in kernel.attributes:
            if kernel.attributes["mathematical_program"] == "block_fp8_quantized_input_fp32_scales_v1":
                m, n, k = (kernel.shape[key] for key in ("m", "n", "k"))
                logical_bytes = m * k + 4 * m * (k // 128) + k * n + 4 * (k // 128) * (n // 128) + 2 * m * n
                seconds = max(2 * m * n * k / hardware.resource_by_role("compute_tensor_fp8").capacity,
                    logical_bytes / hardware.resource_by_role("dram").capacity)
                return prediction_from_ns(seconds * 1e9, "roofline")

            if kernel.attributes["mathematical_program"] in ("mxfp4_bf16_flops_bytes_v1", "mxfp4_fp8_flops_bytes_v1"):
                # Plain matrix service: BF16 or group128 FP8 A, BF16 C,
                # packed E2M1 B and one E8M0 scale per32 K values. Standalone HBM operands;
                # no GPU repacking, padding, quantizer or provider is inferred.
                m, n, k = (kernel.shape[key] for key in ("m", "n", "k"))
                activation_bytes = (m * k + 4 * m * k // 128
                    if kernel.dtype == "fp8" else 2 * m * k)
                logical_bytes = activation_bytes + 2 * m * n + k * n // 2 + k * n // 32
                execution = hardware.fpga_execution
                # Finite matrix wave count from the declared M/K/N lanes;
                # an incomplete M wave does not acquire extra MM cores.
                cycles = math.prod((extent + execution[lane] - 1) // execution[lane]
                    for extent, lane in ((m, "mm_m_parallel"), (k, "mm_k_parallel"), (n, "mm_n_parallel")))
                peak = (2 * execution["clock_hz"] * execution["mm_m_parallel"]
                    * execution["mm_k_parallel"] * execution["mm_n_parallel"])
                utilization = hardware.resource_by_role("compute_tensor").capacity / peak
                seconds = max(cycles / execution["clock_hz"] / utilization,
                    logical_bytes / hardware.resource_by_role("dram").capacity)
                return self._mathematical_prediction(hardware, seconds * 1e9)
            if kernel.attributes["mathematical_program"] in ("unquantized_fp16_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1"):
                work = {"fp16": unquantized_fp16_gemm_logical_work, "bf16": unquantized_bf16_gemm_logical_work}[kernel.dtype](kernel)
                demand = {role: 0.0 for role in (
                    "compute_tensor", "compute_vector", "compute_reduction", "compute_fma", "compute_sfu", "compute_alu")}
                demand["compute_tensor"] = float(work["mathematical_tensor_flops"])
                from src.kernel.memory_access_builder import mathematical_prediction_memory_bytes
                logical_bytes = mathematical_prediction_memory_bytes(kernel, hardware)
                return self._mathematical_prediction(hardware, roofline_ns(demand, hardware, logical_bytes))
            if kernel.attributes["mathematical_program"] == "unquantized_fp32_flops_bytes_v1":
                b, m, n, k = (kernel.shape[key] for key in ("b", "m", "n", "k"))
                demand = {role: 0.0 for role in (
                    "compute_tensor", "compute_vector", "compute_reduction", "compute_fma", "compute_sfu", "compute_alu")}
                demand["compute_fma"] = float(2 * b * m * n * k)
                from src.kernel.memory_access_builder import mathematical_prediction_memory_bytes
                logical_bytes = mathematical_prediction_memory_bytes(kernel, hardware)
                return self._mathematical_prediction(hardware, roofline_ns(demand, hardware, logical_bytes))
            work = block_fp8_linear_logical_work(kernel)
            seconds = max(work["mathematical_flops"] / float(hardware.resource_by_role("compute_tensor_fp8").capacity),
                work["logical_bytes"] / float(hardware.resource_by_role("dram").capacity))
            return self._mathematical_prediction(hardware, seconds * 1e9)
        if kernel.variant == "batched_gemm" and kernel.dtype == "bf16" and kernel.attributes["weight_dtype"] == "bf16":
            work = bf16_batched_gemm_logical_work(kernel)
        elif kernel.variant == "kimi_k3_nope_bmm":
            work = kimi_k3_nope_bmm_logical_work(kernel)
        else:
            return super().roofline_without_launch(kernel, hardware)
        demand = {role: 0.0 for role in (
            "compute_tensor", "compute_vector", "compute_reduction", "compute_fma", "compute_sfu", "compute_alu")}
        demand["compute_tensor"] = float(work["mathematical_tensor_flops"])
        logical_bytes = sum(operand["bytes"] for operand in work["operands"])
        return self._mathematical_prediction(hardware, roofline_ns(demand, hardware, logical_bytes))

    def _packed_fp4_source_prediction(self, kernel: KernelSpec, hardware):
        work = packed_fp4_source_logical_work(kernel)
        bandwidth = hardware.resource_by_role("dram").capacity
        quant_s = work["quantization_logical_bytes"] / bandwidth
        if HWSpec.source_contract_supported("programmed_mxfp4_fp8_expert_service", hardware.hardware_id):
            if kernel.attributes["compute_format"] != "fp8":
                raise NotImplementedError("FPGA packed-FP4 source binds its existing FP8 matrix service")
            execution = hardware.fpga_execution
            k, n = (kernel.shape[name] for name in ("k", "n"))
            m_waves = sum((rows + execution["mm_m_parallel"] - 1) // execution["mm_m_parallel"]
                for rows in work["matrix_row_counts"])
            cycles = m_waves * ((k + execution["mm_k_parallel"] - 1) // execution["mm_k_parallel"]) \
                * ((n + execution["mm_n_parallel"] - 1) // execution["mm_n_parallel"])
            peak = (2 * execution["clock_hz"] * execution["mm_m_parallel"]
                * execution["mm_k_parallel"] * execution["mm_n_parallel"])
            utilization = hardware.resource_by_role(work["compute_role"]).capacity / peak
            compute_s = cycles / execution["clock_hz"] / utilization
            if not kernel.attributes["input_quantized"]:
                if kernel.dtype != "bf16" or k % 128:
                    raise NotImplementedError("FPGA source input conversion requires the existing BF16/group128 quantizer")
                from src.kernel.kernel_base import predict
                quant = KernelSpec(KernelKind.VECTOR, hardware.hardware_id,
                    shape={"element_count": kernel.shape["b"] * kernel.shape["m"] * k,
                        "rows": kernel.shape["b"] * kernel.shape["m"], "columns": k},
                    dtype="bf16", variant="fp8_group_quant", attributes={"group_size": 128, "input_resource": "dram"},
                    execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
                quant_s = predict(quant, hardware, "roofline").t_total_s
        elif HWSpec.source_contract_supported("aggregate_gpu_source_roofline", hardware.hardware_id):
            if not kernel.attributes["input_quantized"] and hardware.gpu_vector_memory_only is not True:
                raise NotImplementedError("GPU packed-FP4 quantization requires the explicit vector-memory-only scenario")
            compute_s = work["mathematical_tensor_flops"] / hardware.resource_by_role(work["compute_role"]).capacity
        else:
            raise NotImplementedError("packed-FP4 source requires its declared aggregate GPU or programmed FPGA design")
        matrix_s = max(compute_s, work["matrix_logical_bytes"] / bandwidth)
        return self._mathematical_prediction(hardware, (quant_s + matrix_s) * 1e9)

    def __init__(self):
        self._cached_resource_kernel = None
        self._cached_resource_hardware = None
        self._cached_resource_value = None

    def resource_input(self, kernel: KernelSpec, hardware):
        if kernel.execution_support == "unsupported":
            return self._resource_input(kernel, hardware)
        if hardware.kernel_memory_model=='npu_tile':
            from src.kernel.memory_access_builder import memory_execution_kernel
            kernel=memory_execution_kernel(kernel,hardware)
        from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_source,gpu_split_gemm_source_demands,gpu_split_gemm_source_work
        if has_gpu_split_gemm_source(kernel,hardware):
            from src.kernel.memory_access_builder import gemm_memory_access_program, memory_execution_kernel
            execution = memory_execution_kernel(kernel,hardware)
            program = gemm_memory_access_program(execution,hardware)
            work = gpu_split_gemm_source_demands(program,expected_kernel=execution)
            with gpu_split_gemm_source_work(execution,program,work):
                return self._resource_input(kernel,hardware)
        from src.kernel.memory_cann_gemm import (has_cann_gemm_source,cann_gemm_source_demands,cann_gemm_source_work,
            has_cann_single_core_gemm_source,cann_single_core_source_demands,
            has_cann_deterministic_gemm_source,cann_deterministic_source_demands)
        if has_cann_deterministic_gemm_source(kernel):
            from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
            program = gemm_memory_access_program(kernel,hardware)
            work = cann_deterministic_source_demands(program)
            with cann_gemm_source_work(kernel,program,work):
                inp,compulsory = self._resource_input(kernel,hardware)
            return dataclasses.replace(inp,limitations=inp.limitations+(
                'cann_deterministic_logical_workspace_not_native_addresses_capacity_or_atomic_transactions',
                'cann_deterministic_legacy_pipeline_worker_classes_and_gm_cache_projections_retained',)),compulsory
        if has_cann_single_core_gemm_source(kernel):
            from src.kernel.memory_access_builder import gemm_prediction_source as gemm_memory_access_program
            program = gemm_memory_access_program(kernel,hardware)
            work = cann_single_core_source_demands(program)
            with cann_gemm_source_work(kernel,program,work):
                inp,compulsory = self._resource_input(kernel,hardware)
            return dataclasses.replace(inp,limitations=inp.limitations+(
                'cann_single_core_split_gemm_logical_workspace_state_not_native_addresses_or_atomic_cache_traffic',
                'cann_single_core_split_gemm_legacy_workspace_dram_projection_retained',)),compulsory
        if has_cann_gemm_source(kernel):
            from src.kernel.memory_access_builder import gemm_memory_access_program, memory_execution_kernel
            execution = memory_execution_kernel(kernel,hardware)
            program = gemm_memory_access_program(execution,hardware)
            work = cann_gemm_source_demands(program)
            with cann_gemm_source_work(execution,program,work):
                return self._resource_input(kernel,hardware)
        if _cann_deterministic_split_k_program(kernel) is not None:
            inp,compulsory = self._resource_input(kernel,hardware)
            return dataclasses.replace(inp,limitations=inp.limitations+(
                'cann_deterministic_native_source_graph_or_abi_unbound',)),compulsory
        if _cann_single_core_split_k_program(kernel) is not None:
            inp,compulsory = self._resource_input(kernel,hardware)
            return dataclasses.replace(inp,limitations=inp.limitations+(
                'cann_single_core_split_gemm_native_source_graph_or_abi_unbound',)),compulsory
        return self._resource_input(kernel,hardware)

    def _resource_input(self, kernel: KernelSpec, hardware):
        if (kernel.launch is None and kernel.variant == "gemm" and kernel.dtype == "bf16"
                and kernel.attributes["output_dtype"] == "fp32"):
            raise NotImplementedError("mixed-output source work has no native launch resource-input program")
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "packed_fp4_source_v1"):
            raise NotImplementedError("packed-FP4 source has no native launch or resource-input program")
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("mxfp4_bf16_flops_bytes_v1", "mxfp4_fp8_flops_bytes_v1")):
            raise NotImplementedError("MXFP4 plain MM program has no native launch or resource-input program")
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("unquantized_fp16_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1")):
            raise NotImplementedError("FP16 mathematical GEMM has no native launch or resource-input program")
        if is_block_fp8_gemm(kernel):
            raise NotImplementedError(
                "block-FP8 GEMM launches are priced by their source template "
                "(midlayer) and expose no resource-formula input")
        # One evaluator case materializes this complete source-derived input
        # for Oracle, Forward, and Roofline with the same immutable objects.
        # Reuse only that current construction; this is not a shape/result
        # lookup. Fresh outer maps keep the retained analytical input private
        # to the predictor while preserving its immutable nested contract.
        if (self._cached_resource_kernel is kernel
                and self._cached_resource_hardware is hardware):
            if self._cached_resource_value is None:
                raise RuntimeError("GEMM resource-input cache is incomplete")
            cached_input, cached_compulsory = self._cached_resource_value
            return dataclasses.replace(
                cached_input,
                demand_per_resource=dict(cached_input.demand_per_resource),
                resource_time_overrides_ns=dict(
                    cached_input.resource_time_overrides_ns),
            ), cached_compulsory
        # Keep the public resource-input contract on the warm analytical path;
        # traffic capture requests its explicitly named cold override below.
        resource_input, compulsory = _model(
            kernel, hardware, cold_traffic=False)
        self._cached_resource_kernel = kernel
        self._cached_resource_hardware = hardware
        self._cached_resource_value = (resource_input, compulsory)
        return resource_input, compulsory

    def forward_resource_input(self, kernel: KernelSpec, hardware):
        """Expose the measured cold ingress account to traffic capture only."""
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("unquantized_fp16_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1")):
            raise NotImplementedError("FP16 mathematical GEMM has no native launch or resource-input program")
        return _model(kernel, hardware, cold_traffic=True)

    def midlayer(self, kernel: KernelSpec, hardware):
        if ("mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] in ("unquantized_fp16_flops_bytes_v1", "unquantized_bf16_flops_bytes_v1")):
            raise NotImplementedError("FP16 mathematical GEMM has no native launch or resource-input program")
        if is_block_fp8_gemm(kernel):
            return predict_block_fp8_gemm(kernel, hardware)
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(
            _with_graph_regime(inp, kernel))

    def roofline(self, kernel: KernelSpec, hardware):
        if is_block_fp8_gemm(kernel):
            return roofline_block_fp8_gemm(kernel, hardware)
        inp, compulsory = self.resource_input(kernel, hardware)
        latency = roofline_ns(inp.demand_per_resource, hardware, compulsory)
        return prediction_from_ns(latency, "roofline")

    def pipeweave(self, kernel: KernelSpec, hardware, *, model_hardware=None):
        if (kernel.attributes["output_dtype"] is not None
            and kernel.attributes["output_dtype"] != kernel.dtype):
            raise NotImplementedError("mixed-output GEMM supports Roofline only")
        from src.kernel.predictors.pipeweave_adapter import predict_gemm
        return predict_gemm(
            kernel, hardware, model_hardware=model_hardware)
