"""Complete-spec fused-MoE grouped-projection prediction."""

from src.kernel.prediction_cache import RuntimeCache, runtime_lru_cache

import dataclasses
import math
from dataclasses import dataclass
from types import MethodType
from typing import Optional, Tuple

from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
)
from src.kernel.predictors.vector import clamped_silu_source_arithmetic, silu_source_arithmetic, VectorPredictor
from src.kernel.predictors.gemm_block_fp8 import _block_fp8_matmul_payloads
from src.kernel.predictors.gemm import _unquantized_matmul_math
from src.kernel.cache_traffic_sim import (
    simulate_cann_grouped_moe_operands,
    simulate_moe_tile_program,
)
from src.kernel.launch_config.fused_moe_tiling import (
    CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES,
    cann_grouped_matmul_k_program,
    fused_moe_output_relayout_sources,
    observed_fused_moe_output_relayout_bytes,
    observed_fused_moe_output_conversion_warp_instructions,
    source_fused_moe_output_conversion_warp_instructions,
    observed_fused_moe_startup_entry_issue,
    observed_fused_moe_operand_shared_work,
    observed_fused_moe_matrix_preparation_issue,
)
from src.kernel.model_contract import (
    AnalyticalPhase,
    PredictionDiagnostics,
    AnalyticalPhaseLedger,
    CannGroupedMatmulEnvelopeDemand,
    ComputeDemand,
    ComputeExactRequestDemand,
    GpuGroupedGemmPipelineEnvelopeDemand,
    MemoryAccessClassDemand,
    MemoryAccessClassService,
    MemoryAccessClassSignature,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS,
    MetaxGroupedOperandPipelineDemand,
    PhaseComputeDemand,
    PhaseMemoryDemand,
    PipelineTaskProgramDemand,
    PipelineWorkPlan,
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
)
from src.kernel.predictors.common import (
    _with_graph_regime,
    PredictionInput,
    attach_structured_contract,
    attach_recurrent_matmul_pipeline,
    merge_operand_read_paths,
    build_aggregate_gpu_input,
    bind_memory_access_class_template,
    build_forward_demand,
    gpu_memory_source,
    gpu_partial_limitations,
    launch_wave_geometry,
    npu_launch_floor_contract,
    prediction_from_ns,
    roofline_ns,
    sum_serial_kernel_predictions,
    validated_l2_miss_rate,
    merge_operand_read_paths,
    attach_recurrent_matmul_pipeline,
    attach_single_panel_matmul_pipeline,
    attach_async_copy_ring_capacity,
    attach_finite_async_copy_progress_floor,
    attach_coalesced_async_copy_progress_floor,
    attach_recurrent_matmul_scalar_epilogue,
    attach_single_panel_prefetched_scalar_epilogue,
)


_GROUPED_OPERAND_SERVICE = "grouped_fused_moe_operand_read"
_GROUPED_OPERAND_PIPELINE_SERVICE = (
    "grouped_fused_moe_operand_pipeline")
_GROUPED_WEIGHT_OVERLAPPED_DRAM_MISS_SERVICE = (
    "grouped_fused_moe_weight_overlapped_dram_miss_pipeline")
_NVIDIA_BM64_GSM8_FIVE_THREE_W1_SERVICE = (
    "nvidia_bm64_gsm8_5_3_w1_operand_pipeline")
_NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE = (
    "nvidia_eight_expert_68_active_72_max_fused_moe_w1_operand_pipeline")
_NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE = (
    "nvidia_eight_expert_132_active_136_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E512_DEEP_W2_SERVICE = (
    "nvidia_e512_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E128_DEEP_W2_SERVICE = (
    "nvidia_e128_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_SERVICE = (
    "nvidia_e128_h4096_4156_active_4222_max_"
    "fused_moe_w1_operand_pipeline")
_NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_M_TILE_CENSUS = (
    33, 32, 32, 32, 32, 32, 32, 31, 33, 31, 32, 32, 33, 31, 32, 33,
    31, 32, 32, 31, 33, 32, 33, 31, 33, 33, 33, 33, 31, 32, 31, 33,
    33, 33, 32, 33, 32, 33, 32, 32, 31, 33, 33, 33, 33, 32, 32, 32,
    32, 33, 33, 32, 32, 33, 33, 31, 32, 33, 32, 33, 33, 34, 32, 33,
    34, 34, 33, 33, 32, 32, 33, 32, 32, 34, 33, 33, 33, 32, 32, 33,
    32, 33, 34, 32, 33, 34, 32, 33, 34, 33, 33, 32, 32, 33, 32, 31,
    32, 33, 32, 33, 32, 32, 33, 33, 31, 32, 33, 33, 34, 32, 33, 32,
    31, 33, 33, 33, 32, 33, 33, 33, 33, 32, 33, 32, 34, 32, 32, 32)
_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT = {
    "w1": (
        "nvidia_e128_h2048_8250_active_8318_max_"
        "fused_moe_w1_operand_pipeline"),
    "w2": (
        "nvidia_e128_h2048_8250_active_8318_max_"
        "fused_moe_w2_operand_pipeline"),
}
_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_M_TILE_CENSUS = (
    64, 66, 65, 64, 63, 65, 66, 64, 64, 65, 64, 65, 63, 66, 64, 65,
    64, 64, 64, 63, 66, 66, 65, 65, 66, 62, 64, 64, 65, 65, 64, 63,
    64, 65, 65, 64, 66, 64, 64, 64, 65, 65, 63, 65, 65, 66, 63, 64,
    65, 64, 64, 65, 64, 64, 64, 64, 65, 64, 65, 66, 65, 65, 65, 65,
    63, 65, 66, 64, 66, 64, 64, 64, 64, 65, 63, 64, 66, 64, 63, 65,
    63, 64, 64, 64, 65, 66, 65, 64, 65, 65, 66, 64, 65, 65, 64, 66,
    63, 66, 66, 65, 63, 65, 63, 63, 64, 64, 63, 65, 63, 65, 64, 65,
    64, 64, 64, 64, 64, 63, 64, 64, 64, 65, 64, 65, 65, 64, 65, 66,
)
_NVIDIA_E8_H6144_DEEP_W2_SERVICE = (
    "nvidia_e8_h6144_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H6144_260_ACTIVE_264_MAX_W2_SERVICE = (
    "nvidia_e8_h6144_260_active_264_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H6144_DEEP_W1_SERVICE = (
    "nvidia_e8_h6144_deep_fused_moe_w1_operand_pipeline")
_NVIDIA_E8_H6144_DEEP_W1_M_TILE_CENSUS = (
    128, 129, 127, 128, 127, 130, 129, 130)
_NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W1_SERVICE = (
    "nvidia_e8_h6144_8_active_9_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W2_SERVICE = (
    "nvidia_e8_h6144_8_active_9_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H4096_515_ACTIVE_520_MAX_W2_SERVICE = (
    "nvidia_e8_h4096_515_active_520_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H4096_1027_ACTIVE_1032_MAX_W2_SERVICE = (
    "nvidia_e8_h4096_1027_active_1032_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY = {
    (515, 520): _NVIDIA_E8_H4096_515_ACTIVE_520_MAX_W2_SERVICE,
    (1027, 1032): _NVIDIA_E8_H4096_1027_ACTIVE_1032_MAX_W2_SERVICE,
}
_NVIDIA_E8_H4096_EXACT_W2_M_TILE_CENSUS_BY_TOPOLOGY = {
    (515, 520): (64, 65, 65, 64, 65, 65, 63, 64),
    (1027, 1032): (127, 131, 130, 127, 127, 128, 129, 128),
}
_NVIDIA_E512_H4096_DEEP_W1_SERVICE = (
    "nvidia_e512_h4096_deep_fused_moe_w1_operand_pipeline")
_NVIDIA_E512_H4096_DEEP_W1_M_TILE_CENSUS = (
    11, 10, 11, 10, 10, 10, 10, 10, 10, 11, 10, 10, 10, 11, 10, 11,
    10, 10, 10, 10, 11, 11, 11, 10, 11, 11, 10, 11, 10, 11, 10, 10,
    11, 11, 10, 10, 10, 10, 11, 11, 10, 11, 11, 11, 10, 11, 10, 10,
    11, 12, 10, 11, 10, 10, 11, 10, 10, 11, 11, 11, 11, 11, 11, 10,
    11, 11, 11, 11, 11, 10, 11, 10, 10, 11, 11, 11, 10, 10, 11, 11,
    11, 11, 11, 10, 10, 11, 10, 11, 11, 11, 10, 11, 10, 11, 10, 10,
    11, 10, 10, 11, 10, 11, 11, 11, 11, 11, 11, 10, 11, 10, 11, 10,
    10, 11, 11, 11, 11, 10, 11, 11, 10, 10, 11, 10, 11, 10, 10, 11,
    11, 10, 11, 10, 10, 10, 10, 11, 11, 10, 10, 11, 11, 10, 11, 10,
    11, 11, 11, 11, 10, 10, 11, 10, 11, 11, 11, 10, 10, 10, 10, 11,
    11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 11, 11, 11, 10, 11, 10, 11, 11, 10, 11, 11, 11, 10, 11,
    11, 11, 11, 11, 10, 10, 10, 11, 10, 11, 10, 10, 10, 11, 10, 11,
    10, 10, 11, 11, 10, 11, 11, 11, 11, 11, 10, 10, 10, 11, 11, 10,
    10, 10, 10, 11, 10, 10, 10, 10, 10, 11, 10, 10, 11, 10, 10, 11,
    10, 11, 11, 10, 10, 11, 10, 10, 11, 11, 11, 11, 10, 10, 11, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 10, 10, 11, 10, 10, 10,
    10, 11, 10, 10, 10, 10, 10, 10, 11, 10, 11, 11, 10, 9, 10, 11,
    11, 10, 11, 10, 11, 10, 10, 10, 10, 11, 11, 11, 11, 10, 10, 10,
    10, 11, 11, 10, 10, 10, 11, 10, 10, 10, 10, 11, 10, 11, 10, 11,
    11, 11, 11, 10, 10, 11, 11, 10, 11, 10, 11, 11, 11, 10, 10, 10,
    10, 11, 10, 11, 11, 10, 9, 10, 10, 11, 11, 10, 10, 10, 11, 10,
    11, 10, 11, 11, 11, 10, 11, 10, 10, 10, 11, 11, 11, 11, 10, 10,
    10, 11, 11, 11, 11, 11, 11, 10, 11, 10, 11, 11, 11, 11, 10, 11,
    11, 11, 10, 10, 10, 10, 10, 10, 11, 10, 11, 10, 11, 10, 11, 11,
    10, 11, 10, 10, 11, 11, 11, 10, 11, 11, 11, 11, 11, 10, 10, 11,
    11, 11, 10, 11, 10, 12, 10, 10, 11, 10, 10, 11, 10, 11, 10, 11,
    10, 10, 11, 10, 10, 10, 10, 11, 10, 10, 10, 10, 10, 11, 11, 10,
    11, 10, 10, 10, 11, 10, 11, 10, 10, 11, 10, 11, 10, 10, 11, 11,
    10, 10, 11, 10, 11, 11, 10, 11, 11, 10, 11, 11, 11, 10, 11, 10,
    11, 11, 10, 10, 10, 10, 11, 11, 10, 11, 11, 10, 11, 10, 11, 10,
    10, 11, 10, 10, 10, 10, 11, 10, 11, 11, 10, 10, 11, 9, 10, 11)
_NVIDIA_E256_H7168_DEEP_W1_SERVICE = (
    "nvidia_e256_h7168_deep_fused_moe_w1_operand_pipeline")
_NVIDIA_E256_H7168_DEEP_W2_SERVICE = (
    "nvidia_e256_h7168_deep_fused_moe_w2_operand_pipeline")
_NVIDIA_E256_H7168_MID_WORKSPACE_W1_SERVICE = (
    "nvidia_e256_h7168_mid_workspace_fused_moe_w1_operand_pipeline")
_NVIDIA_E256_H7168_MID_WORKSPACE_W2_SERVICE = (
    "nvidia_e256_h7168_mid_workspace_fused_moe_w2_operand_pipeline")
_NVIDIA_E256_H7168_2166_EXACT_W1_SERVICE = (
    "nvidia_e256_h7168_2166_exact_w1_operand_pipeline")
_NVIDIA_E256_H7168_2166_W1_M_TILE_CENSUS = (
    8, 9, 8, 8, 9, 8, 8, 9, 8, 9, 8, 8, 9, 9, 9, 8,
    8, 9, 8, 8, 8, 9, 8, 8, 8, 9, 9, 9, 9, 9, 8, 8,
    10, 8, 8, 8, 9, 8, 8, 8, 9, 8, 9, 8, 8, 9, 8, 8,
    8, 8, 8, 8, 9, 9, 9, 9, 9, 8, 8, 8, 9, 9, 9, 8,
    8, 9, 9, 8, 8, 9, 8, 8, 8, 8, 8, 9, 8, 9, 8, 8,
    9, 9, 8, 8, 8, 8, 8, 9, 8, 9, 8, 8, 9, 8, 8, 9,
    10, 8, 8, 8, 9, 9, 9, 8, 9, 9, 9, 8, 9, 8, 8, 8,
    8, 8, 9, 9, 8, 8, 8, 8, 9, 8, 9, 9, 8, 9, 8, 8,
    8, 8, 8, 9, 9, 8, 9, 9, 8, 9, 8, 8, 8, 8, 8, 8,
    8, 9, 9, 9, 8, 10, 8, 8, 8, 9, 8, 9, 8, 8, 8, 8,
    9, 9, 9, 8, 9, 8, 9, 9, 9, 8, 9, 8, 9, 9, 8, 8,
    8, 8, 9, 9, 8, 8, 9, 8, 9, 8, 9, 9, 8, 9, 8, 9,
    8, 9, 8, 9, 8, 9, 9, 9, 9, 8, 8, 8, 9, 8, 8, 9,
    8, 9, 9, 8, 8, 9, 8, 8, 9, 9, 9, 9, 9, 9, 8, 9,
    8, 8, 9, 8, 8, 8, 9, 9, 9, 9, 8, 8, 8, 8, 9, 9,
    8, 9, 9, 8, 9, 9, 8, 9, 8, 9, 9, 8, 8, 8, 9, 9)
_NVIDIA_E256_H7168_DEEP_M_TILE_CENSUS = (
    17, 17, 16, 16, 16, 16, 17, 17, 17, 16, 17, 16, 16, 17, 17, 17,
    16, 16, 16, 17, 17, 16, 16, 16, 16, 16, 17, 16, 16, 16, 17, 17,
    16, 16, 17, 17, 17, 16, 16, 17, 16, 17, 16, 17, 16, 16, 16, 17,
    16, 17, 16, 16, 17, 17, 16, 16, 16, 17, 17, 16, 17, 17, 16, 17,
    16, 16, 16, 17, 16, 17, 16, 16, 17, 17, 17, 16, 17, 16, 16, 16,
    17, 17, 16, 16, 16, 17, 16, 16, 17, 17, 16, 16, 17, 17, 16, 17,
    17, 17, 16, 17, 16, 16, 17, 17, 16, 16, 16, 17, 17, 17, 17, 17,
    16, 17, 16, 16, 16, 17, 17, 17, 17, 16, 17, 15, 17, 16, 16, 16,
    17, 17, 17, 17, 17, 16, 17, 16, 16, 16, 17, 16, 16, 17, 17, 16,
    17, 16, 16, 17, 16, 17, 16, 16, 17, 17, 17, 16, 16, 16, 16, 16,
    16, 17, 16, 17, 17, 16, 17, 16, 17, 17, 17, 17, 17, 16, 16, 16,
    17, 17, 16, 16, 17, 16, 17, 17, 16, 16, 17, 16, 17, 17, 17, 17,
    17, 16, 16, 16, 16, 17, 16, 16, 17, 16, 16, 17, 16, 17, 17, 17,
    17, 17, 17, 17, 17, 16, 16, 16, 17, 17, 17, 17, 17, 18, 16, 17,
    16, 16, 18, 16, 17, 17, 16, 17, 17, 17, 17, 16, 17, 16, 17, 17,
    16, 17, 17, 16, 18, 16, 16, 17, 17, 16, 16, 17, 17, 16, 17, 17)
_NVIDIA_E256_H7168_2166_ACTIVE_2300_MAX_W2_SERVICE = (
    "nvidia_e256_h7168_2166_active_2300_max_fused_moe_w2_operand_pipeline")
_NVIDIA_DEEP_H2880_W1_W2_SERVICE = (
    "nvidia_deep_h2880_fused_moe_operand_pipeline")
_NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W1_SERVICE = (
    "nvidia_e32_h2880_32_active_36_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W2_SERVICE = (
    "nvidia_e32_h2880_32_active_36_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W1_SERVICE = (
    "nvidia_e32_h2880_143_active_160_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W2_SERVICE = (
    "nvidia_e32_h2880_143_active_160_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E512_1001_ACTIVE_1144_MAX_W1_SERVICE = (
    "nvidia_e512_1001_active_1144_max_fused_moe_w1_operand_pipeline")
_NVIDIA_E512_1001_ACTIVE_1144_MAX_W2_SERVICE = (
    "nvidia_e512_1001_active_1144_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W1_SERVICE = (
    "nvidia_bm64_e128_574_active_638_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W2_SERVICE = (
    "nvidia_bm64_e128_574_active_638_max_fused_moe_w2_operand_pipeline")
_NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W1_SERVICE = (
    "nvidia_e128_h2880_1084_active_1150_exact_w1_operand_pipeline")
_NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W2_SERVICE = (
    "nvidia_e128_h2880_1084_active_1150_exact_w2_operand_pipeline")
_NVIDIA_E128_H2880_1084_ACTIVE_1150_M_TILE_CENSUS = (
    8, 8, 8, 8, 8, 9, 8, 8, 8, 9, 8, 9, 8, 8, 8, 9,
    9, 9, 8, 8, 9, 9, 8, 9, 8, 8, 9, 9, 8, 8, 9, 8,
    9, 9, 8, 8, 9, 8, 8, 8, 8, 8, 8, 9, 9, 8, 8, 9,
    9, 9, 8, 8, 9, 8, 8, 8, 8, 9, 8, 9, 9, 9, 8, 9,
    9, 8, 8, 8, 9, 9, 9, 8, 8, 9, 9, 9, 8, 9, 9, 9,
    9, 9, 8, 9, 8, 9, 8, 8, 9, 9, 8, 9, 8, 8, 9, 8,
    8, 9, 9, 8, 9, 8, 8, 8, 9, 9, 9, 8, 8, 8, 8, 8,
    9, 9, 9, 8, 9, 8, 9, 8, 8, 9, 9, 9, 9, 8, 9, 8)
_NVIDIA_BM16_H2880_31_ACTIVE_SERVICE = (
    "nvidia_bm16_h2880_31_active_fused_moe_operand_pipeline")
_NVIDIA_BM16_H2880_53_ACTIVE_SERVICE = (
    "nvidia_bm16_h2880_53_active_fused_moe_operand_pipeline")
_NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE = (
    "nvidia_bm16_e128_88_active_128_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE = (
    "nvidia_bm16_e128_88_active_128_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE = (
    "nvidia_bm16_e128_177_active_248_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE = (
    "nvidia_bm16_e128_177_active_248_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE = (
    "nvidia_bm16_e8_7_active_8_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE = (
    "nvidia_bm16_e8_7_active_8_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE = (
    "nvidia_bm16_e8_h6144_7_active_8_max_fused_moe_w1_operand_pipeline")
_NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE = (
    "nvidia_bm16_e8_h6144_7_active_8_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE = (
    "nvidia_bm16_e256_16_active_241_max_fused_moe_w2_operand_pipeline")
_NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE = (
    "nvidia_bm16_e512_20_active_482_max_fused_moe_w2_operand_pipeline")
_METAX_GROUPED_OPERAND_LONG_K_SERVICE = (
    "metax_grouped_fused_moe_long_k_operand_pipeline")
_METAX_GROUPED_OPERAND_WIDE_N_SERVICE = (
    "metax_grouped_fused_moe_wide_n_operand_pipeline")
_METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE = (
    "metax_global_deep_fused_moe_long_k_operand_pipeline")
_METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE = (
    "metax_global_large_fused_moe_wide_n_operand_pipeline")
_METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE = (
    "metax_long_lifetime_fused_moe_long_k_operand_pipeline")
_METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE = (
    "metax_long_lifetime_fused_moe_wide_n_operand_pipeline")
_METAX_REPEATED_OPERAND_DEEP_K_SERVICE = (
    "metax_repeated_fused_moe_deep_k_operand_pipeline")
_METAX_REPEATED_OPERAND_WIDE_N_SERVICE = (
    "metax_repeated_fused_moe_wide_n_operand_pipeline")
_METAX_REPEATED_OPERAND_BALANCED_N_SERVICE = (
    "metax_repeated_fused_moe_balanced_n_operand_pipeline")
_METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE = (
    "metax_repeated_fused_moe_mid_long_k_operand_pipeline")
_METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE = (
    "metax_repeated_fused_moe_mid_wide_n_operand_pipeline")
_METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE = (
    "metax_single_tile_fused_moe_long_k_operand_pipeline")
_METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE = (
    "metax_single_tile_fused_moe_wide_n_operand_pipeline")
_METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE = (
    "metax_eight_expert_eight_active_twelve_max_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE = (
    "metax_eight_expert_single_tile_large_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE = (
    "metax_eight_expert_single_tile_large_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE = (
    "metax_eight_expert_small_single_tile_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE = (
    "metax_eight_expert_small_single_tile_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE = (
    "metax_eight_expert_steady_fused_moe_wide_n_operand_pipeline")
_METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE = (
    "metax_eight_expert_steady_k192_n512_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE = (
    "metax_eight_expert_68_active_72_max_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W2_SERVICE = (
    "metax_eight_expert_68_active_72_max_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K128_N448_W1_SERVICE = (
    "metax_eight_expert_68_active_72_max_k128_n448_"
    "fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K448_N64_W2_SERVICE = (
    "metax_eight_expert_68_active_72_max_k448_n64_"
    "fused_moe_w2_operand_pipeline")
_METAX_E32_32_ACTIVE_48_MAX_W1_SERVICE = (
    "metax_e32_32_active_48_max_fused_moe_w1_operand_pipeline")
_METAX_E32_32_ACTIVE_48_MAX_W2_SERVICE = (
    "metax_e32_32_active_48_max_fused_moe_w2_operand_pipeline")
_METAX_E32_47_ACTIVE_64_MAX_W1_SERVICE = (
    "metax_e32_47_active_64_max_fused_moe_w1_operand_pipeline")
_METAX_E32_47_ACTIVE_64_MAX_W2_SERVICE = (
    "metax_e32_47_active_64_max_fused_moe_w2_operand_pipeline")
_METAX_E32_78_ACTIVE_96_MAX_W1_SERVICE = (
    "metax_e32_78_active_96_max_fused_moe_w1_operand_pipeline")
_METAX_E32_78_ACTIVE_96_MAX_W2_SERVICE = (
    "metax_e32_78_active_96_max_fused_moe_w2_operand_pipeline")
_METAX_E32_143_ACTIVE_160_MAX_W1_SERVICE = (
    "metax_e32_143_active_160_max_fused_moe_w1_operand_pipeline")
_METAX_E32_143_ACTIVE_160_MAX_W2_SERVICE = (
    "metax_e32_143_active_160_max_fused_moe_w2_operand_pipeline")
_METAX_E256_LONG_LIFETIME_LONG_K_SERVICE = (
    "metax_e256_long_lifetime_fused_moe_long_k_operand_pipeline")
_METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE = (
    "metax_e256_long_lifetime_fused_moe_wide_n_operand_pipeline")
_METAX_E256_LONG_LIFETIME_MID_W1_SERVICE = (
    "metax_e256_long_lifetime_mid_fused_moe_w1_operand_pipeline")
_METAX_E256_LONG_LIFETIME_MID_W2_SERVICE = (
    "metax_e256_long_lifetime_mid_fused_moe_w2_operand_pipeline")
_METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE = (
    "metax_e512_dense_ten_to_eleven_tile_fused_moe_w1_operand_pipeline")
_METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE = (
    "metax_e512_dense_ten_to_eleven_tile_fused_moe_w2_operand_pipeline")
_METAX_E512_DENSE_LARGE_W2_SERVICE = (
    "metax_e512_dense_activation_refreshed_large_w2_operand_pipeline")
_METAX_E512_1010_ACTIVE_1144_MAX_W2_SERVICE = (
    "metax_e512_1010_active_1144_max_fused_moe_w2_operand_pipeline")
_METAX_E128_LONG_LIFETIME_W1_SERVICE = (
    "metax_e128_long_lifetime_fused_moe_w1_operand_pipeline")
_METAX_E128_LONG_LIFETIME_W2_SERVICE = (
    "metax_e128_long_lifetime_fused_moe_w2_operand_pipeline")
_METAX_E128_DEEP_W2_SERVICE = (
    "metax_e128_deep_fused_moe_w2_operand_pipeline")
_METAX_E128_EIGHT_TILE_W1_SERVICE = (
    "metax_e128_eight_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_EIGHT_TILE_W2_SERVICE = (
    "metax_e128_eight_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE = (
    "metax_e128_topk16_eight_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE = (
    "metax_e128_eight_tile_wide_n_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE = (
    "metax_eight_expert_twelve_tile_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_DEEP_W2_SERVICE = (
    "metax_eight_expert_twelve_tile_deep_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE = (
    "metax_eight_expert_twelve_tile_k128_n448_"
    "fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE = (
    "metax_eight_expert_twelve_tile_k192_n512_"
    "fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE = (
    "metax_eight_expert_twenty_one_tile_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE = (
    "metax_eight_expert_twenty_one_tile_fused_moe_w2_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W1_SERVICE = (
    "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w1_operand_pipeline")
_METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W2_SERVICE = (
    "metax_eight_expert_twenty_active_twenty_four_max_fused_moe_w2_operand_pipeline")
_METAX_E128_SPARSE_TWO_TILE_W1_SERVICE = (
    "metax_e128_sparse_two_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_SPARSE_TWO_TILE_W2_SERVICE = (
    "metax_e128_sparse_two_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_188_ACTIVE_254_MAX_W1_SERVICE = (
    "metax_e128_188_active_254_max_fused_moe_w1_operand_pipeline")
_METAX_E128_188_ACTIVE_254_MAX_W2_SERVICE = (
    "metax_e128_188_active_254_max_fused_moe_w2_operand_pipeline")
_METAX_E128_SINGLE_TILE_W1_SERVICE = (
    "metax_e128_single_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_SINGLE_TILE_W2_SERVICE = (
    "metax_e128_single_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_SINGLE_TILE_MID_W1_SERVICE = (
    "metax_e128_single_tile_mid_fused_moe_w1_operand_pipeline")
_METAX_E128_SINGLE_TILE_MID_W2_SERVICE = (
    "metax_e128_single_tile_mid_fused_moe_w2_operand_pipeline")
_METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE = (
    "metax_e128_single_tile_large_fused_moe_w1_operand_pipeline")
_METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE = (
    "metax_e128_single_tile_large_fused_moe_w2_operand_pipeline")
_METAX_E128_UPPER_TWO_TILE_W1_SERVICE = (
    "metax_e128_upper_two_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_UPPER_TWO_TILE_W2_SERVICE = (
    "metax_e128_upper_two_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE = (
    "metax_e128_topk8_128_active_190_max_fused_moe_w1_operand_pipeline")
_METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE = (
    "metax_e128_topk16_128_active_190_max_fused_moe_w1_operand_pipeline")
_METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE = (
    "metax_e128_128_active_190_max_fused_moe_w2_operand_pipeline")
_METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE = (
    "metax_e128_repeated_source_topk8_fused_moe_w1_operand_pipeline")
_METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE = (
    "metax_e128_repeated_source_topk16_fused_moe_w1_operand_pipeline")
_METAX_E128_REPEATED_K24_N32_W2_SERVICE = (
    "metax_e128_repeated_k24_n32_fused_moe_w2_operand_pipeline")
_METAX_E256_EIGHT_TILE_LONG_K_SERVICE = (
    "metax_e256_eight_tile_fused_moe_long_k_operand_pipeline")
_METAX_E256_EIGHT_TILE_WIDE_N_SERVICE = (
    "metax_e256_eight_tile_fused_moe_wide_n_operand_pipeline")
_METAX_E256_EIGHT_TILE_MID_W1_SERVICE = (
    "metax_e256_eight_tile_mid_fused_moe_w1_operand_pipeline")
_METAX_E256_EIGHT_TILE_MID_W2_SERVICE = (
    "metax_e256_eight_tile_mid_fused_moe_w2_operand_pipeline")
_METAX_E256_FOUR_TILE_LONG_K_SERVICE = (
    "metax_e256_four_tile_fused_moe_long_k_operand_pipeline")
_METAX_E256_FOUR_TILE_WIDE_N_SERVICE = (
    "metax_e256_four_tile_fused_moe_wide_n_operand_pipeline")
_METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE = (
    "metax_e256_four_tile_mid_fused_moe_long_k_operand_pipeline")
_METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE = (
    "metax_e256_four_tile_mid_fused_moe_wide_n_operand_pipeline")
_METAX_E128_FOUR_TILE_W1_SERVICE = (
    "metax_e128_four_tile_fused_moe_w1_operand_pipeline")
_METAX_E128_FOUR_TILE_W2_SERVICE = (
    "metax_e128_four_tile_fused_moe_w2_operand_pipeline")
_METAX_E128_FOUR_TILE_LARGE_W1_SERVICE = (
    "metax_e128_four_tile_large_fused_moe_w1_operand_pipeline")
_METAX_E128_FOUR_TILE_LARGE_W2_SERVICE = (
    "metax_e128_four_tile_large_fused_moe_w2_operand_pipeline")
_METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE = (
    "metax_e256_upper_two_tile_fused_moe_long_k_operand_pipeline")
_METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE = (
    "metax_e256_upper_two_tile_fused_moe_wide_n_operand_pipeline")
_METAX_E256_SINGLE_TILE_W1_SERVICE = (
    "metax_e256_single_tile_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_W2_SERVICE = (
    "metax_e256_single_tile_fused_moe_w2_operand_pipeline")
_METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE = (
    "metax_e256_single_tile_mid_deep_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE = (
    "metax_e256_single_tile_mid_deep_fused_moe_w2_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE = (
    "metax_e256_single_tile_large_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE = (
    "metax_e256_single_tile_large_fused_moe_w2_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE = (
    "metax_e256_single_tile_large_deep_fused_moe_w1_operand_pipeline")
_METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE = (
    "metax_e256_single_tile_large_deep_fused_moe_w2_operand_pipeline")
_METAX_E512_SINGLE_TILE_W1_SERVICE = (
    "metax_e512_single_tile_fused_moe_w1_operand_pipeline")
_METAX_E512_SINGLE_TILE_W2_SERVICE = (
    "metax_e512_single_tile_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE = (
    "metax_bm16_e512_521_active_800_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE = (
    "metax_bm16_e512_521_active_800_max_fused_moe_w2_operand_pipeline")
_METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE = (
    "metax_bm64_e512_512_active_664_max_fused_moe_w1_operand_pipeline")
_METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE = (
    "metax_bm64_e512_512_active_664_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE = (
    "metax_bm16_active_grid_fused_moe_wide_n_operand_pipeline")
_METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE = (
    "metax_bm16_e256_sixteen_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E256_SIXTEEN_ACTIVE_DEEP_W2_SERVICE = (
    "metax_bm16_e256_sixteen_active_deep_"
    "fused_moe_w2_operand_pipeline")
_METAX_BM16_SPARSE_PADDED_W2_SERVICE = (
    "metax_bm16_sparse_padded_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE = (
    "metax_bm16_e512_sparse_padded_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_FORTY_ACTIVE_W1_SERVICE = (
    "metax_bm16_e512_forty_active_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE = (
    "metax_bm16_e512_forty_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_FORTY_ACTIVE_K8_N64_W2_SERVICE = (
    "metax_bm16_e512_forty_active_k8_n64_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_75_ACTIVE_80_MAX_W1_SERVICE = (
    "metax_bm16_e512_75_active_80_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_75_ACTIVE_485_MAX_W2_SERVICE = (
    "metax_bm16_e512_75_active_485_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_TWENTY_ACTIVE_W1_SERVICE = (
    "metax_bm16_e512_twenty_active_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_TWENTY_ACTIVE_W2_SERVICE = (
    "metax_bm16_e512_twenty_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_TWENTY_ACTIVE_K8_N64_W2_SERVICE = (
    "metax_bm16_e512_twenty_active_k8_n64_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_TEN_ACTIVE_W1_SERVICE = (
    "metax_bm16_e512_ten_active_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_TEN_ACTIVE_W2_SERVICE = (
    "metax_bm16_e512_ten_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_143_ACTIVE_490_MAX_W1_SERVICE = (
    "metax_bm16_e512_143_active_490_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_143_ACTIVE_490_MAX_W2_SERVICE = (
    "metax_bm16_e512_143_active_490_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E512_MID_SPARSE_W1_SERVICE = (
    "metax_bm16_e512_mid_sparse_fused_moe_w1_operand_pipeline")
_METAX_BM16_E512_MID_SPARSE_W2_SERVICE = (
    "metax_bm16_e512_mid_sparse_fused_moe_w2_operand_pipeline")
_METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE = (
    "metax_bm16_e256_sparse_colliding_fused_moe_w1_operand_pipeline")
_METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE = (
    "metax_bm16_e256_sparse_colliding_fused_moe_w2_operand_pipeline")
_METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE = (
    "metax_bm16_e256_106_active_248_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE = (
    "metax_bm16_e256_106_active_248_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_31_ACTIVE_32_MAX_W1_SERVICE = (
    "metax_bm16_31_active_32_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_31_ACTIVE_W2_SERVICE = (
    "metax_bm16_e128_31_active_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_53_ACTIVE_64_MAX_W1_SERVICE = (
    "metax_bm16_e128_53_active_64_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_53_ACTIVE_124_MAX_W2_SERVICE = (
    "metax_bm16_e128_53_active_124_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK8_16_ACTIVE_16_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk8_16_active_16_max_"
    "fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK8_16_ACTIVE_121_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk8_16_active_121_max_"
    "fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_16_ACTIVE_121_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_16_active_121_max_"
    "fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_16_ACTIVE_16_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_16_active_16_max_"
    "fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_152_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_152_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_64_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_124_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_128_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_128_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_136_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_136_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_184_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_184_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE = (
    "metax_bm16_e128_topk16_248_max_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE = (
    "metax_bm16_e128_topk16_248_max_fused_moe_w2_operand_pipeline")
_METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE = (
    "metax_bm16_four_active_mid_k_fused_moe_w1_operand_pipeline")
_METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_four_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W1_SERVICE = (
    "metax_bm16_e128_eight_active_mid_k_fused_moe_w1_operand_pipeline")
_METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_e128_eight_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_eight_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE = (
    "metax_bm16_e32_seven_active_mid_k_fused_moe_w1_operand_pipeline")
_METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE = (
    "metax_bm16_e32_seven_active_mid_k_fused_moe_w2_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_36_MAX_W1_SERVICE = (
    "metax_bm64_e32_32_active_36_max_fused_moe_w1_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_36_MAX_W2_SERVICE = (
    "metax_bm64_e32_32_active_36_max_fused_moe_w2_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_40_MAX_W1_SERVICE = (
    "metax_bm64_e32_32_active_40_max_fused_moe_w1_operand_pipeline")
_METAX_BM64_E32_32_ACTIVE_40_MAX_W2_SERVICE = (
    "metax_bm64_e32_32_active_40_max_fused_moe_w2_operand_pipeline")
_GROUPED_OPERAND_STREAM_COUNTS = (1, 2, 4, 8)
_CANN_OPERAND_A_L2_HIT_SERVICE = (
    "cann_grouped_moe_operand_a_l2_hit")
_CANN_OPERAND_A_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_a_dram_miss")
_CANN_OPERAND_B_L2_HIT_SERVICE = (
    "cann_grouped_moe_operand_b_l2_hit")
_CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE = (
    "cann_grouped_moe_operand_b_k2880_stride_l2_hit")
_CANN_OPERAND_B_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_b_dram_miss")
_CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_b_k2880_strided_dram_miss")
_CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE = (
    "cann_grouped_moe_operand_b_k6144_m1_dram_miss")
_CANN_GROUPED_K_OUTER_BASE_TILES = 4


def _projection(kernel: KernelSpec) -> Tuple[str, int, int, int]:
    from src.kernel.memory_fused_moe import fused_moe_projection
    return fused_moe_projection(kernel)


def _tile(kernel: KernelSpec) -> Tuple[int, int, int]:
    tiling = kernel.launch.tiling
    if tiling is None or not {"m", "n", "k"}.issubset(tiling.tile_shape):
        raise ValueError("FUSED_MOE requires m/n/k tiling in LaunchConfig")
    return tuple(int(tiling.tile_shape[name]) for name in ("m", "n", "k"))


def _routing(kernel: KernelSpec):
    from src.kernel.memory_fused_moe import fused_moe_routing
    return fused_moe_routing(kernel)


def _gpu_uses_naive_assignment(kernel):
    from src.kernel.memory_fused_moe_naive import has_h200_bf16_controls, h200_bf16_naive
    return has_h200_bf16_controls(kernel) and h200_bf16_naive(kernel)


def _gpu_valid_tile_rows(kernel, counts):
    """Keep token-wise routes separate when the source skips expert sorting."""
    if _gpu_uses_naive_assignment(kernel):
        return (1,) * sum(counts)
    tm, _, _ = _tile(kernel)
    return tuple(min(tm, count - start) for count in counts
                 for start in range(0, count, tm))


def _source_w2_routed_weight_program(kernel: KernelSpec) -> Optional[bool]:
    """Return whether C500 selected W2's routed-weight source program.

    The frozen chain writes the activation directly to ``intermediate_cache2``
    and passes that buffer to W2.  This marker therefore denotes only W2's
    later router-weight load and multiply, never a standalone activation copy.
    It remains optional because it is a C500-specific source fact.
    """

    parameter = SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER
    if parameter not in kernel.launch.parameters:
        return None
    raw_invocations = kernel.launch.parameters[parameter]
    if (isinstance(raw_invocations, bool)
            or not isinstance(raw_invocations, (int, float))
            or not float(raw_invocations).is_integer()):
        raise TypeError(
            "FUSED_MOE source W2 routed-weight program must be an integer")
    program_selected = int(raw_invocations)
    if program_selected not in (0, 1):
        raise ValueError(
            "FUSED_MOE source W2 routed-weight program must be zero or one")
    variant = kernel.variant.lower()
    if program_selected == 0 and variant != "w1":
        raise ValueError(
            "FUSED_MOE absent W2 routed-weight program is valid only for W1")
    if program_selected == 1 and variant != "w2":
        raise ValueError(
            "FUSED_MOE W2 routed-weight program is valid only for W2")
    return bool(program_selected)


def _source_routed_weight_scale_operations(
        kernel: KernelSpec, *, routed_rows: float,
        projection_n: int) -> float:
    """Return the source-defined C500 W2 router-scale ALU work.

    The C500 Triton W2 program selected by the routed-weight marker
    loads one router weight for each valid routed row and multiplies the
    accumulated logical output row before its masked store. W1 has no such
    source operation. Do not infer this edge for backends whose launch lacks
    the C500 source marker.
    """

    program_selected = _source_w2_routed_weight_program(kernel)
    if program_selected is None or not program_selected:
        return 0.0
    if routed_rows < 0.0:
        raise ValueError("FUSED_MOE routed rows cannot be negative")
    if projection_n <= 0:
        raise ValueError("FUSED_MOE routed-weight scale needs positive N")
    return routed_rows * float(projection_n)


def _source_bf16_w2_router_epilogue_work(kernel: KernelSpec, *, router_weight_bytes: int):
    """Source-only diagnostic work for the BF16 pointer-path W2 epilogue.

    vLLM fused_experts_impl defaults apply_router_weight_on_input=False;
    its W2 reads each route weight once per N program, multiplies the FP32
    accumulator, then casts/stores. The caller must bind that dispatch and
    the router tensor's actual element size; BF16 operands do not imply a
    BF16 router tensor.

    Return logical ingress bytes, unique footprint bytes and padded scalar
    multiplications. The unique footprint is an optimistic DRAM floor, not
    a cache observation. The FP32 scalar instruction capacity is distinct
    from FLOPs at the FMA resource's advertised two-FLOP instruction rate.
    """
    if kernel.dtype != "bf16" or kernel.variant != "w2":
        raise ValueError("source router epilogue requires BF16 W2")
    if (isinstance(router_weight_bytes, bool) or not isinstance(router_weight_bytes, int)
            or router_weight_bytes not in (2, 4)):
        raise ValueError("source router tensor requires bound two/four-byte elements")
    _, local_counts = _routing(kernel)
    tm, tn, _ = _tile(kernel)
    _, _, _, n = _projection(kernel)
    n_tiles = math.ceil(n / tn)
    ctas = len(_gpu_valid_tile_rows(kernel, local_counts)) * n_tiles
    unique_bytes = sum(local_counts) * router_weight_bytes
    return unique_bytes * n_tiles, unique_bytes, ctas * tm * tn


def _attach_gpu_bf16_w2_router_epilogue(inp, kernel, hardware):
    """Attach the source W2 BF16 router read and FP32 output multiply."""
    if (hardware.kernel_memory_model != "gpu_mshr" or kernel.dtype != "bf16"
            or kernel.variant != "w2"
            or kernel.launch.algorithm != "triton_grouped_gemm:w2"
            or _source_w2_routed_weight_program(kernel) is not None):
        return inp, 0
    ingress, unique, multiplies = _source_bf16_w2_router_epilogue_work(
        kernel, router_weight_bytes=2)
    plan = inp.pipeline_plan
    # The aggregate model retains separate activation/weight paths when no
    # source-bound resident pipeline is available. It still executes the
    # same router read and scalar multiply; a merged operand path is only
    # the representation used by the detailed pipeline.
    operand_paths = tuple(path for path in inp.memory_paths
        if path.access_kind == "read" and path.name != "source_router_weight_read")
    operand_names = tuple(path.name for path in operand_paths)
    if operand_names == ("operand_read",):
        operand, = operand_paths
    elif plan is None and operand_names == ("activation_read", "weight_read"):
        operand = operand_paths[0]
    else:
        raise ValueError("router epilogue requires a declared merged or split operand read")
    templates = {node.resource_role: node for node in operand.nodes}
    nodes = tuple(
        MemoryNodeDemand(
            resource_role=role, bytes=amount,
            transaction_bytes=templates[role].transaction_bytes,
            transactions=amount / templates[role].transaction_bytes)
        for role, amount in (("l2_cache", ingress), ("dram", unique)))
    read = MemoryPathDemand(
        name="source_router_weight_read", access_kind="read",
        initial_transactions=ingress / templates["l2_cache"].transaction_bytes,
        nodes=nodes,
        source=MemorySourceDemand(inp.active_units, 1.0, None, None),
        queues=operand.queues)
    recurrent = (
        isinstance(plan, PipelineWorkPlan)
        and all(isinstance(task, PipelineTaskProgramDemand) for task in plan.tasks))
    single_panel = (
        isinstance(plan, PipelineWorkPlan) and len(plan.tasks) == 1
        and plan.tasks[0].name == "single_panel_aggregate")
    if recurrent:
        tasks = len(_grouped_task_memory_bytes(kernel))
        after = attach_recurrent_matmul_scalar_epilogue(
            inp, task_count=tasks, read_path=read,
            scalar_multiplications=multiplies, prefetch_read=True)
    elif single_panel:
        after = attach_single_panel_prefetched_scalar_epilogue(
            inp, read_path=read, scalar_multiplications=multiplies)
    elif plan is None:
        after = _attach_aggregate_bf16_w2_router_epilogue(
            inp, read, multiplies)
    else:
        return dataclasses.replace(
            inp, limitations=inp.limitations + (
                "h200_w2_router_epilogue_plan_unbound",)), 0
    after.phase_ledger.validate_conservation(
        after.compute_demands, after.memory_paths)
    return after, unique


def _attach_aggregate_bf16_w2_router_epilogue(inp, read, multiplies):
    """Conserve router work in the existing schedule-free aggregate model."""
    if (any(d.resource_role == "compute_fma" for d in inp.compute_demands)
            or inp.demand_per_resource["compute_fma"] != 0
            or "compute_fma" in inp.resource_time_overrides_ns
            or any(path.name == read.name for path in inp.memory_paths)):
        raise ValueError("aggregate router epilogue requires unassigned scalar work")
    ledger = inp.phase_ledger
    if (ledger is None or ledger.phases[-1].name != "output_writeback"
            or any(phase.name == "source_scalar_epilogue_read" for phase in ledger.phases)):
        raise ValueError("aggregate router epilogue requires the original output ledger")
    demand = dict(inp.demand_per_resource)
    demand["compute_fma"] = float(multiplies)
    for node in read.nodes:
        demand[node.resource_role] += node.bytes
    output = ledger.phases[-1]
    conversion = (ledger.phases[-2] if len(ledger.phases) >= 2
                  and ledger.phases[-2].name == "output_conversion" else None)
    anchor = output if conversion is None else conversion
    read_phase = AnalyticalPhase("source_scalar_epilogue_read", (), (),
        tuple(PhaseMemoryDemand(read.name, node.resource_role, node.bytes)
              for node in read.nodes))
    multiply = AnalyticalPhase("source_scalar_epilogue_multiply",
        (read_phase.name,) + anchor.predecessors,
        (PhaseComputeDemand("compute_fma", "FP32_scalar_row_multiply", float(multiplies)),), ())
    if conversion is None:
        phases = ledger.phases[:-1] + (read_phase, multiply,
            dataclasses.replace(output, predecessors=(multiply.name,)))
    else:
        phases = ledger.phases[:-2] + (read_phase, multiply,
            dataclasses.replace(conversion, predecessors=(multiply.name,)), output)
    return dataclasses.replace(inp, demand_per_resource=demand,
        compute_demands=inp.compute_demands + (
            ComputeDemand("compute_fma", float(multiplies), 0.5, 1.0),),
        memory_paths=inp.memory_paths + (read,),
        phase_ledger=dataclasses.replace(ledger, phases=phases),
        limitations=inp.limitations + (
            "source_scalar_row_epilogue_after_matrix_before_output",
            "source_router_epilogue_uses_aggregate_resource_capacities_without_cta_schedule",))


def _source_routed_weight_read_traffic(
        kernel: KernelSpec, *, routed_rows: float,
        n_tiles: int) -> Tuple[float, float]:
    """Return C500 W2 FP32 router-weight ingress and unique bytes.

    The frozen MetaX Triton W2 program loads ``topk_weights_ptr`` after its
    K loop for every valid routed row in every output-N program.  The router
    tensor is created as FP32 by ``fused_topk``.  Its per-program requests
    must remain a distinct post-matmul stream: treating it as activation
    ingress would move a source-visible dependency ahead of the matrix
    multiply.  The source marker is C500-specific, so backends without it do
    not infer this read.
    """

    program_selected = _source_w2_routed_weight_program(kernel)
    if program_selected is None or not program_selected:
        return 0.0, 0.0
    if routed_rows < 0.0:
        raise ValueError("FUSED_MOE routed rows cannot be negative")
    if n_tiles <= 0:
        raise ValueError("FUSED_MOE routed-weight read needs output tiles")
    from src.kernel.memory_fused_moe import native_grouped_router_weight_traffic
    source=native_grouped_router_weight_traffic(kernel)
    if source is not None:
        ingress,compulsory=source
        if ingress!=compulsory*float(n_tiles) or compulsory!=float(routed_rows)*4:
            raise ValueError('Source router-weight requests disagree with actual routed/N tile geometry')
        return float(ingress),float(compulsory)
    router_weight_bytes = 4.0
    compulsory = routed_rows * router_weight_bytes
    return compulsory * float(n_tiles), compulsory


def _validate_expert_parallel_launch(kernel: KernelSpec) -> None:
    expert_parallel_size = int(kernel.attributes["expert_parallel_size"])
    if expert_parallel_size == 1:
        return
    parameters = kernel.launch.parameters
    if "expert_parallel_size" not in parameters:
        raise NotImplementedError(
            "FUSED_MOE expert-parallel launch is not source-backed")
    launch_value = parameters["expert_parallel_size"]
    if (isinstance(launch_value, bool)
            or not isinstance(launch_value, (int, float))
            or not float(launch_value).is_integer()):
        raise ValueError(
            "FUSED_MOE launch expert_parallel_size must be an integer")
    if int(launch_value) != expert_parallel_size:
        raise ValueError(
            "FUSED_MOE launch expert_parallel_size disagrees with the "
            "KernelSpec")


@dataclass(frozen=True)
class _CannGroupedMatmulWork:
    tensor_operations: int
    worker_count: int
    expert_group_count: int
    routed_rows: int
    populated_m_tiles: int
    n_tiles: int


@dataclass(frozen=True)
class _CannGroupedMatmulTraffic:
    """Source-derived AIC interfaces and unique GM footprint, in bytes."""

    activation_l2_read_bytes: float
    weight_l2_read_bytes: float
    output_l2_write_bytes: float
    dram_read_bytes: float
    dram_write_bytes: float

    @property
    def l2_read_bytes(self) -> float:
        return self.activation_l2_read_bytes + self.weight_l2_read_bytes

    @property
    def l2_total_bytes(self) -> float:
        return self.l2_read_bytes + self.output_l2_write_bytes

    @property
    def dram_total_bytes(self) -> float:
        return self.dram_read_bytes + self.dram_write_bytes


@dataclass(frozen=True)
class _CannGroupedOperandResidency:
    """Source-ordered full-tile service classes and all operand bytes."""

    a_hits_by_worker: Tuple[int, ...]
    a_misses_by_worker: Tuple[int, ...]
    b_hits_by_worker: Tuple[int, ...]
    b_misses_by_worker: Tuple[int, ...]
    pair_hit_hit_by_worker: Tuple[int, ...]
    pair_hit_miss_by_worker: Tuple[int, ...]
    pair_miss_hit_by_worker: Tuple[int, ...]
    pair_miss_miss_by_worker: Tuple[int, ...]
    tile_classes_by_worker: Tuple[Tuple[Tuple[int, int, int], ...], ...]
    l2_read_bytes: int
    dram_read_bytes: int


@dataclass(frozen=True)
class _GpuGroupedGemmTraffic:
    """Operand-specific grouped-GEMM hierarchy traffic, in bytes."""

    activation_l2_read_bytes: float
    weight_l2_read_bytes: float
    routed_weight_l2_read_bytes: float
    output_l2_write_bytes: float
    activation_compulsory_read_bytes: float
    weight_compulsory_read_bytes: float
    routed_weight_compulsory_read_bytes: float
    activation_dram_read_bytes: float
    weight_dram_read_bytes: float
    routed_weight_dram_read_bytes: float
    output_dram_write_bytes: float

    @property
    def l2_read_bytes(self) -> float:
        return (
            self.activation_l2_read_bytes
            + self.weight_l2_read_bytes
            + self.routed_weight_l2_read_bytes)

    @property
    def l2_total_bytes(self) -> float:
        return self.l2_read_bytes + self.output_l2_write_bytes

    @property
    def dram_read_bytes(self) -> float:
        return (
            self.activation_dram_read_bytes
            + self.weight_dram_read_bytes
            + self.routed_weight_dram_read_bytes)

    @property
    def dram_total_bytes(self) -> float:
        return self.dram_read_bytes + self.output_dram_write_bytes

    @property
    def compulsory_bytes(self) -> float:
        return (
            self.activation_compulsory_read_bytes
            + self.weight_compulsory_read_bytes
            + self.routed_weight_compulsory_read_bytes
            + self.output_l2_write_bytes)


def _cann_mn_block_index(
        relative_block: int, block_dim_m: int,
        block_dim_n: int) -> Tuple[int, int]:
    """Mirror CANN 8.5.1 ``MNBlockIdxCompute`` for Ascend 910B."""

    if relative_block < 0 or relative_block >= block_dim_m * block_dim_n:
        raise ValueError("GroupedMatmul relative block is out of range")
    if block_dim_m <= 0 or block_dim_n <= 0:
        raise ValueError("GroupedMatmul block dimensions must be positive")
    threshold_block_num = 8
    threshold_dim_m = 5
    if block_dim_m <= threshold_dim_m:
        return relative_block // block_dim_n, relative_block % block_dim_n

    threshold_m_dim_n = threshold_block_num * block_dim_n
    full_threshold_extent = (
        block_dim_m * block_dim_n // threshold_m_dim_n
        * threshold_m_dim_n)
    if relative_block >= full_threshold_extent:
        current_threshold_m = block_dim_m % threshold_block_num
    else:
        current_threshold_m = threshold_block_num
    if current_threshold_m <= 0:
        raise AssertionError("GroupedMatmul diagonal M threshold is empty")
    current_threshold_mn = current_threshold_m * threshold_block_num
    local_n_extent = (
        current_threshold_m * block_dim_n // current_threshold_mn
        * current_threshold_mn)
    if relative_block % threshold_m_dim_n >= local_n_extent:
        current_threshold_n = block_dim_n % threshold_block_num
    else:
        current_threshold_n = threshold_block_num
    if current_threshold_n <= 0:
        raise AssertionError("GroupedMatmul diagonal N threshold is empty")

    local_relative = (
        relative_block % threshold_m_dim_n % current_threshold_mn)
    m_index = (
        local_relative % current_threshold_m
        + relative_block // threshold_m_dim_n * threshold_block_num)
    n_index = (
        (local_relative
         + local_relative
         // math.lcm(current_threshold_m, current_threshold_n))
        % current_threshold_n
        + relative_block % threshold_m_dim_n
        // current_threshold_mn * threshold_block_num)
    if m_index >= block_dim_m or n_index >= block_dim_n:
        raise AssertionError("GroupedMatmul diagonal tile index is invalid")
    return m_index, n_index


@runtime_lru_cache(maxsize=256)
def _cann_mn_block_order(
        block_dim_m: int, block_dim_n: int) -> Tuple[Tuple[int, int], ...]:
    """Return one immutable CANN diagonal order for a launch tile grid."""

    if block_dim_m <= 0 or block_dim_n <= 0:
        raise ValueError("GroupedMatmul block dimensions must be positive")
    return tuple(
        _cann_mn_block_index(relative_block, block_dim_m, block_dim_n)
        for relative_block in range(block_dim_m * block_dim_n))


def _cann_grouped_matmul_work(kernel: KernelSpec) -> _CannGroupedMatmulWork:
    """Price version-bound BF16 launch tiles on their assigned AICs."""

    if kernel.launch.execution_variant != kernel.variant:
        raise ValueError(
            "CANN GroupedMatmul launch variant disagrees with the kernel")
    if kernel.launch.grid != (kernel.launch.grid_size, 1, 1):
        raise ValueError("CANN GroupedMatmul requires a one-dimensional grid")
    if kernel.launch.block != (1, 1, 1):
        raise ValueError("CANN GroupedMatmul requires one mixed task per block")
    worker_count = kernel.launch.grid_size
    _, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    if tile_m % 16 or tile_n % 16 or tile_k % 16:
        raise ValueError("CANN GroupedMatmul BF16 tiles must be 16-aligned")
    global_counts, local_counts = _routing(kernel)
    if any(rows < 0 for rows in global_counts):
        raise ValueError("CANN GroupedMatmul global expert rows cannot be negative")
    if any(rows < 0 for rows in local_counts):
        raise ValueError("CANN GroupedMatmul expert rows cannot be negative")
    expected_global_rows = (
        int(kernel.shape["tokens"]) * int(kernel.shape["topk"]))
    if sum(global_counts) != expected_global_rows:
        raise ValueError(
            "CANN GroupedMatmul global routing must conserve tokens*topk")
    if any(local > global_ for local, global_ in zip(
            local_counts, global_counts)):
        raise ValueError(
            "CANN GroupedMatmul local expert rows cannot exceed global rows")
    routed_rows = sum(local_counts)
    expected_tile = (
        min(128, ((expected_global_rows + 15) // 16) * 16), 256, 64)
    if (tile_m, tile_n, tile_k) != expected_tile:
        raise ValueError(
            "CANN 8.5.1 BF16 GroupedMatmul requires exact base tile "
            f"{expected_tile}, got {(tile_m, tile_n, tile_k)}")
    if projection_k % 16:
        raise ValueError("CANN GroupedMatmul K must be Cube-fragment aligned")

    populated_m_tiles = 0
    n_tiles = math.ceil(projection_n / tile_n)
    if int(kernel.attributes["expert_parallel_size"]) == 1:
        if (global_counts != local_counts
                or sum(local_counts) != expected_global_rows):
            raise ValueError(
                "single-rank CANN GroupedMatmul routing must conserve "
                "tokens*topk in both global and local counts")
    padded_n_total = (
        (n_tiles - 1) * tile_n
        + math.ceil((projection_n - (n_tiles - 1) * tile_n) / 16) * 16)
    tensor_operations = 0
    for rows in local_counts:
        if rows == 0:
            continue
        block_dim_m = math.ceil(rows / tile_m)
        block_dim_n = math.ceil(projection_n / tile_n)
        if block_dim_n != n_tiles:
            raise AssertionError(
                "CANN GroupedMatmul N-tile count changed across experts")
        populated_m_tiles += block_dim_m
        tail_rows = rows - (block_dim_m - 1) * tile_m
        padded_m_total = (
            (block_dim_m - 1) * tile_m
            + math.ceil(tail_rows / 16) * 16)
        tensor_operations += 2 * padded_m_total * padded_n_total * projection_k
    if tensor_operations <= 0:
        raise ValueError("CANN GroupedMatmul routing produced no Tensor work")
    return _CannGroupedMatmulWork(
        tensor_operations=tensor_operations,
        worker_count=worker_count,
        expert_group_count=len(local_counts),
        routed_rows=sum(local_counts),
        populated_m_tiles=populated_m_tiles,
        n_tiles=n_tiles,
    )


def _cann_grouped_b_l2_cache_disabled(kernel: KernelSpec) -> bool:
    """Return the static CANN B-cache mode for an all-single-M invocation.

    CANN 8.5.1's exact runtime ``GroupedMatmul`` source calls
    ``SetL2CacheHint(CACHE_MODE_DISABLE)`` on B when ``blockDimM == 1``.
    Restrict this to invocations whose every populated expert has exactly one
    source M tile: mixed launches need a separate per-expert residency trace.
    """

    expected_algorithm = f"cann_grouped_matmul_{kernel.variant}"
    if (kernel.launch.algorithm != expected_algorithm
            or kernel.dtype != "bf16"):
        return False
    tile_m, _, _ = _tile(kernel)
    _, local_counts = _routing(kernel)
    nonempty_rows = tuple(rows for rows in local_counts if rows > 0)
    if not nonempty_rows:
        raise ValueError("CANN GroupedMatmul has no populated experts")
    return all(rows <= tile_m for rows in nonempty_rows)


def _cann_grouped_operand_residency(
        kernel: KernelSpec, work: _CannGroupedMatmulWork,
        hardware) -> _CannGroupedOperandResidency:
    """Classify A/B tile requests using source order and physical L2 size."""

    if hardware.l2_size_bytes <= 0:
        raise ValueError(
            "CANN GroupedMatmul residency requires positive L2 capacity")
    _, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    _, local_counts = _routing(kernel)
    b_l2_cache_disabled = _cann_grouped_b_l2_cache_disabled(kernel)
    result = simulate_cann_grouped_moe_operands(
        projection_k,
        projection_n,
        tile_m,
        tile_n,
        tile_k,
        element_bytes,
        work.worker_count,
        hardware.l2_size_bytes,
        local_counts,
        b_l2_cache_disabled=b_l2_cache_disabled,
    )
    worker_fields = (
        "a_hits_by_worker",
        "a_misses_by_worker",
        "b_hits_by_worker",
        "b_misses_by_worker",
        "pair_hit_hit_by_worker",
        "pair_hit_miss_by_worker",
        "pair_miss_hit_by_worker",
        "pair_miss_miss_by_worker",
    )
    for field_name in worker_fields:
        values = result[field_name]
        if len(values) != work.worker_count:
            raise ValueError(
                "CANN GroupedMatmul residency worker count disagrees with "
                "the launch")
        if any(value < 0 for value in values):
            raise ValueError(
                "CANN GroupedMatmul residency counts cannot be negative")
    tile_classes_by_worker = result["tile_classes_by_worker"]
    if len(tile_classes_by_worker) != work.worker_count:
        raise ValueError(
            "CANN GroupedMatmul tile census worker count disagrees with "
            "the launch")
    if any(
            any(logical_m <= 0 or logical_n <= 0 or count <= 0
                for logical_m, logical_n, count in worker_classes)
            for worker_classes in tile_classes_by_worker):
        raise ValueError("CANN GroupedMatmul tile census is invalid")
    if result["l2_read_bytes"] <= 0 or result["dram_read_bytes"] <= 0:
        raise ValueError(
            "CANN GroupedMatmul residency must produce positive traffic")
    if result["dram_read_bytes"] > result["l2_read_bytes"]:
        raise ValueError(
            "CANN GroupedMatmul DRAM reads cannot exceed L2 ingress reads")
    return _CannGroupedOperandResidency(
        a_hits_by_worker=result["a_hits_by_worker"],
        a_misses_by_worker=result["a_misses_by_worker"],
        b_hits_by_worker=result["b_hits_by_worker"],
        b_misses_by_worker=result["b_misses_by_worker"],
        pair_hit_hit_by_worker=result["pair_hit_hit_by_worker"],
        pair_hit_miss_by_worker=result["pair_hit_miss_by_worker"],
        pair_miss_hit_by_worker=result["pair_miss_hit_by_worker"],
        pair_miss_miss_by_worker=result["pair_miss_miss_by_worker"],
        tile_classes_by_worker=tile_classes_by_worker,
        l2_read_bytes=result["l2_read_bytes"],
        dram_read_bytes=result["dram_read_bytes"],
    )


def _cann_grouped_operand_hit_fractions(
        kernel: KernelSpec,
        residency: _CannGroupedOperandResidency) -> Tuple[float, float]:
    """Recover exact full-stream A/B hit fractions when tiles are complete.

    The native source-order replay includes M/N tails in its aggregate L2 and
    DRAM byte totals, while its legacy per-worker counters intentionally cover
    only full descriptors.  When K and N are both exact tile multiples, every
    B request is represented by those counters.  Subtracting its exact bytes
    from the aggregate totals therefore recovers the complete A stream,
    including M tails, without adding an execution-order assumption.
    """

    _, _, projection_k, projection_n = _projection(kernel)
    _, tile_n, tile_k = _tile(kernel)
    a_total = sum(residency.a_hits_by_worker) + sum(
        residency.a_misses_by_worker)
    b_total = sum(residency.b_hits_by_worker) + sum(
        residency.b_misses_by_worker)
    a_hit_fraction = (
        sum(residency.a_hits_by_worker) / a_total if a_total else 0.0)
    b_hit_fraction = (
        sum(residency.b_hits_by_worker) / b_total if b_total else 0.0)
    if projection_k % tile_k or projection_n % tile_n:
        return a_hit_fraction, b_hit_fraction

    element_bytes = int(kernel.shape["dtype_bytes"])
    b_descriptor_bytes = tile_n * tile_k * element_bytes
    b_l2_read_bytes = b_total * b_descriptor_bytes
    b_dram_read_bytes = (
        sum(residency.b_misses_by_worker) * b_descriptor_bytes)
    a_l2_read_bytes = residency.l2_read_bytes - b_l2_read_bytes
    a_dram_read_bytes = residency.dram_read_bytes - b_dram_read_bytes
    if (a_l2_read_bytes <= 0 or a_dram_read_bytes < 0
            or a_dram_read_bytes > a_l2_read_bytes):
        raise AssertionError(
            "CANN GroupedMatmul source-order operand bytes are inconsistent")
    a_hit_fraction = 1.0 - a_dram_read_bytes / a_l2_read_bytes
    return a_hit_fraction, b_hit_fraction


def _gpu_operand_element_bytes(kernel: KernelSpec) -> int:
    """Return the A/B operand element width of the grouped GEMM.

    Pinned vLLM 0.11.0 ``fused_experts_impl`` quantizes the W1 input and the
    W2 input with ``moe_kernel_quantize_input`` before invoking the
    ``use_fp8_w8a8`` Triton kernel, so an FP8 routed-expert GEMM loads one-byte
    activations and one-byte weights; its output cache stays in the BF16
    compute dtype.  BF16 routed experts, and legacy specs that declare no
    routed quantization contract, load operands in the activation dtype.
    INT4 routed experts use the separate Marlin model.
    """

    activation_bytes = int(kernel.shape["dtype_bytes"])
    if "routed_weight_dtype" not in kernel.attributes:
        return activation_bytes
    routed_weight_dtype = kernel.attributes["routed_weight_dtype"]
    if routed_weight_dtype == "bf16":
        return activation_bytes
    if routed_weight_dtype == "fp8":
        return 1
    raise ValueError(
        "GPU grouped FUSED_MOE operand width requires bf16 or fp8 routed "
        f"weights, got {routed_weight_dtype!r}")


def _gpu_operand_replay_factors(kernel: KernelSpec) -> Tuple[int, int]:
    """Return launch-bound L2-ingress stream multiplicities.

    A value of one is the source-issued base stream.  MetaX BM64's value of
    two is a version-bound, same-invocation non-duration counter observation;
    it is not a cross-tile replay law.  In particular, BM16 contains counter
    outliers that prevent interpreting its value of one as a complete model
    of all physical L2 traffic.  The missing mechanism remains explicit in
    diagnostics instead of being inferred from production duration.
    """

    a_value = kernel.launch.parameters["operand_replay_a"]
    b_value = kernel.launch.parameters["operand_replay_b"]
    if (not float(a_value).is_integer()
            or not float(b_value).is_integer()):
        raise ValueError("FUSED_MOE operand replay counts must be integers")
    a_replay, b_replay = int(a_value), int(b_value)
    if a_replay <= 0 or b_replay <= 0:
        raise ValueError("FUSED_MOE operand replay counts must be positive")
    return a_replay, b_replay


def _gpu_grouped_operand_signature(
        kernel: KernelSpec, active_rows=None) -> MemoryAccessClassSignature:
    """Return the exact tile signature for the combined A+B read stream."""

    tile_m, tile_n, tile_k = _tile(kernel)
    if active_rows is None:
        active_rows = tile_m
    if (isinstance(active_rows, bool)
            or not isinstance(active_rows, int)
            or active_rows <= 0 or active_rows > tile_m):
        raise ValueError(
            "grouped FUSED_MOE active rows must be within the M tile")
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    # The signature keys a measured tile service class.  Every published
    # grouped-operand service was measured on BF16 tiles, so an FP8 operand
    # stream reuses its BF16 tile class as a declared proxy; the bytes it
    # moves are counted at the true operand width in the traffic model.
    element_bytes = int(kernel.shape["dtype_bytes"])
    return MemoryAccessClassSignature(
        access_class=_GROUPED_OPERAND_SERVICE,
        dtype=kernel.dtype,
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=(
            active_rows * tile_k + tile_k * tile_n) * element_bytes,
        operand_role="combined",
        tile_n=tile_n,
        group_size_m=int(raw_group_size),
    )


def _gpu_exact_grouped_operand_service(kernel: KernelSpec, hardware):
    """Return the exact combined service, without cross-tile fallback."""

    signature = _gpu_grouped_operand_signature(kernel)
    matches = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _GROUPED_OPERAND_SERVICE
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature == signature)
    if len(matches) > 1:
        raise ValueError(
            "grouped FUSED_MOE repeats an exact combined operand service")
    return matches[0] if matches else None


def _native_memory_service_catalog(hardware):
    """Index only plain native records for one synchronous selector call."""
    from types import GetSetDescriptorType
    hw_fields, service_fields = vars(HWSpec), vars(MemoryAccessClassService)
    if (type(hardware) is not HWSpec
            or HWSpec.__getattribute__ is not object.__getattribute__
            or MemoryAccessClassService.__getattribute__ is not object.__getattribute__
            or "memory_access_class_services" in hw_fields or "service" in service_fields
            or "__dict__" not in hw_fields or "__dict__" not in service_fields
            or type(hw_fields["__dict__"]) is not GetSetDescriptorType
            or type(service_fields["__dict__"]) is not GetSetDescriptorType
            or hw_fields["__dict__"].__objclass__ is not HWSpec
            or service_fields["__dict__"].__objclass__ is not MemoryAccessClassService
            or type(_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT) is not dict
            or type(_NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY) is not dict):
        return None
    namespace = vars(hardware)
    if ("memory_access_class_services" not in namespace
            or type(namespace["memory_access_class_services"]) is not tuple):
        return None
    table = namespace["memory_access_class_services"]
    if any(type(record) is not MemoryAccessClassService
            or "service" not in vars(record) or type(vars(record)["service"]) is not str
            for record in table):
        return None
    groups = {}
    for record in table:
        if record.service not in groups:
            groups[record.service] = []
        groups[record.service].append(record)
    catalog = {name: tuple(records) for name, records in groups.items()}
    return namespace, table, catalog, frozenset(catalog)


def _nvidia_source_depth_operand_service(kernel: KernelSpec, hardware):
    """Select full-source services for statically bounded launch depths."""

    # All indexed reads precede metadata callbacks. The final post-metadata
    # query below deliberately retains its live ordinary scan.
    catalog = _native_memory_service_catalog(hardware)

    def _services_for(name):
        nonlocal catalog
        if (catalog is not None and type(name) is str
                and "memory_access_class_services" in catalog[0]
                and catalog[0]["memory_access_class_services"] is catalog[1]):
            return catalog[2][name] if name in catalog[2] else ()
        catalog = None
        return tuple(service for service in hardware.memory_access_class_services
            if service.service == name)

    def _declared_names(names):
        nonlocal catalog
        if (catalog is not None and type(names) in (tuple, set, frozenset)
                and all(type(name) is str for name in names)
                and "memory_access_class_services" in catalog[0]
                and catalog[0]["memory_access_class_services"] is catalog[1]):
            return set(catalog[3].intersection(names))
        catalog = None
        return {service.service for service in hardware.memory_access_class_services
            if service.service in names}

    service_names = (
        _NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE,
        _NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE,
    )
    declared_w1_names = _declared_names(service_names)
    if declared_w1_names and declared_w1_names != set(service_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE source-depth service family must be "
            "complete")
    deep_w2_services = _services_for(_NVIDIA_E512_DEEP_W2_SERVICE)
    if len(deep_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its deep W2 source service")
    e128_deep_w2_services = _services_for(_NVIDIA_E128_DEEP_W2_SERVICE)
    if len(e128_deep_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E128 deep W2 source service")
    e128_h4096_deep_w1_services = _services_for(_NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_SERVICE)
    if len(e128_h4096_deep_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E128 H4096 4156/4222 "
            "W1 source service")
    e128_h2048_names = set(
        _NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT.values())
    declared_e128_h2048_names = _declared_names(e128_h2048_names)
    if (declared_e128_h2048_names
            and declared_e128_h2048_names != e128_h2048_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E128 H2048 exact source service "
            "family must be complete")
    e8_h6144_deep_w2_services = _services_for(_NVIDIA_E8_H6144_DEEP_W2_SERVICE)
    if len(e8_h6144_deep_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E8 H6144 deep W2 "
            "source service")
    e8_h6144_260_active_264_w2_services = _services_for(_NVIDIA_E8_H6144_260_ACTIVE_264_MAX_W2_SERVICE)
    if len(e8_h6144_260_active_264_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E8 H6144 260/264 W2 "
            "source service")
    e8_h6144_deep_w1_services = _services_for(_NVIDIA_E8_H6144_DEEP_W1_SERVICE)
    if len(e8_h6144_deep_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E8 H6144 deep W1 "
            "source service")
    e8_h4096_exact_w2_names = set(
        _NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY.values())
    declared_e8_h4096_exact_w2_names = _declared_names(e8_h4096_exact_w2_names)
    if (declared_e8_h4096_exact_w2_names
            and declared_e8_h4096_exact_w2_names
            != e8_h4096_exact_w2_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E8 H4096 exact W2 source service "
            "family must be complete")
    e512_h4096_deep_w1_services = _services_for(_NVIDIA_E512_H4096_DEEP_W1_SERVICE)
    if len(e512_h4096_deep_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E512 H4096 deep W1 "
            "source service")
    e256_h7168_names = {
        _NVIDIA_E256_H7168_DEEP_W1_SERVICE,
        _NVIDIA_E256_H7168_DEEP_W2_SERVICE,
    }
    declared_e256_h7168_names = _declared_names(e256_h7168_names)
    if (declared_e256_h7168_names
            and declared_e256_h7168_names != e256_h7168_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E256 H7168 source-depth service "
            "family must be complete")
    e256_h7168_mid_workspace_names = {
        _NVIDIA_E256_H7168_MID_WORKSPACE_W1_SERVICE,
        _NVIDIA_E256_H7168_MID_WORKSPACE_W2_SERVICE,
    }
    declared_e256_h7168_mid_workspace_names = _declared_names(e256_h7168_mid_workspace_names)
    if (declared_e256_h7168_mid_workspace_names
            and declared_e256_h7168_mid_workspace_names
            != e256_h7168_mid_workspace_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E256 H7168 mid-workspace source "
            "service family must be complete")
    e256_h7168_2166_w2_services = _services_for(_NVIDIA_E256_H7168_2166_ACTIVE_2300_MAX_W2_SERVICE)
    if len(e256_h7168_2166_w2_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E256 H7168 2166/2300 "
            "W2 source service")
    e256_h7168_2166_w1_services = _services_for(_NVIDIA_E256_H7168_2166_EXACT_W1_SERVICE)
    if len(e256_h7168_2166_w1_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its E256 H7168 2166/2300 "
            "W1 source service")
    e8_h6144_short_names = {
        _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W1_SERVICE,
        _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W2_SERVICE,
    }
    declared_e8_h6144_short_names = _declared_names(e8_h6144_short_names)
    if (declared_e8_h6144_short_names
            and declared_e8_h6144_short_names != e8_h6144_short_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E8 H6144 8/9 source service family "
            "must be complete")
    deep_h2880_services = _services_for(_NVIDIA_DEEP_H2880_W1_W2_SERVICE)
    if len(deep_h2880_services) > 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE repeats its deep H2880 source service")
    e32_h2880_short_names = {
        _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W1_SERVICE,
        _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W2_SERVICE,
    }
    declared_e32_h2880_short_names = _declared_names(e32_h2880_short_names)
    if (declared_e32_h2880_short_names
            and declared_e32_h2880_short_names != e32_h2880_short_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E32 H2880 short source-depth service "
            "family must be complete")
    e32_h2880_names = {
        _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W1_SERVICE,
        _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W2_SERVICE,
    }
    declared_e32_h2880_names = _declared_names(e32_h2880_names)
    if (declared_e32_h2880_names
            and declared_e32_h2880_names != e32_h2880_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E32 H2880 source-depth service "
            "family must be complete")
    e512_mid_depth_names = {
        _NVIDIA_E512_1001_ACTIVE_1144_MAX_W1_SERVICE,
        _NVIDIA_E512_1001_ACTIVE_1144_MAX_W2_SERVICE,
    }
    declared_e512_mid_depth_names = _declared_names(e512_mid_depth_names)
    if (declared_e512_mid_depth_names
            and declared_e512_mid_depth_names != e512_mid_depth_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E512 source-depth service family "
            "must be complete")
    e128_bm64_names = {
        _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W1_SERVICE,
        _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W2_SERVICE,
    }
    declared_e128_bm64_names = _declared_names(e128_bm64_names)
    if (declared_e128_bm64_names
            and declared_e128_bm64_names != e128_bm64_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE BM64 E128 source-depth service "
            "family must be complete")
    e128_h2880_1084_names = {
        _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W1_SERVICE,
        _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W2_SERVICE,
    }
    declared_e128_h2880_1084_names = _declared_names(e128_h2880_1084_names)
    if (declared_e128_h2880_1084_names
            and declared_e128_h2880_1084_names != e128_h2880_1084_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE E128 H2880 exact source service "
            "family must be complete")
    bm16_h2880_names = {
        _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE,
        _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE,
    }
    declared_bm16_h2880_names = _declared_names(bm16_h2880_names)
    if (declared_bm16_h2880_names
            and declared_bm16_h2880_names != bm16_h2880_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE BM16 H2880 source-depth service "
            "family must be complete")
    sparse_w2_names = {
        _NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE,
        _NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE,
    }
    declared_sparse_w2_names = _declared_names(sparse_w2_names)
    if (declared_sparse_w2_names
            and declared_sparse_w2_names != sparse_w2_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE sparse W2 source-depth service "
            "family must be complete")
    e128_shallow_names = {
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
    }
    declared_e128_shallow_names = _declared_names(e128_shallow_names)
    if (declared_e128_shallow_names
            and declared_e128_shallow_names != e128_shallow_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE shallow E128 source-depth service "
            "family must be complete")
    e128_mid_depth_names = {
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE,
    }
    declared_e128_mid_depth_names = _declared_names(e128_mid_depth_names)
    if (declared_e128_mid_depth_names
            and declared_e128_mid_depth_names != e128_mid_depth_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE mid-depth E128 source service "
            "family must be complete")
    e8_shallow_names = {
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE,
    }
    declared_e8_shallow_names = _declared_names(e8_shallow_names)
    if (declared_e8_shallow_names
            and declared_e8_shallow_names != e8_shallow_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE shallow E8 source service family "
            "must be complete")
    e8_h6144_names = {
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE,
    }
    declared_e8_h6144_names = _declared_names(e8_h6144_names)
    if (declared_e8_h6144_names
            and declared_e8_h6144_names != e8_h6144_names):
        raise ValueError(
            "NVIDIA grouped FUSED_MOE H6144 E8 source service family "
            "must be complete")
    if (not declared_w1_names and not deep_w2_services
            and not e128_deep_w2_services
            and not e128_h4096_deep_w1_services
            and not declared_e128_h2048_names
            and not e8_h6144_deep_w2_services
            and not e8_h6144_260_active_264_w2_services
            and not e8_h6144_deep_w1_services
            and not declared_e8_h4096_exact_w2_names
            and not e512_h4096_deep_w1_services
            and not declared_e256_h7168_names
            and not declared_e256_h7168_mid_workspace_names
            and not e256_h7168_2166_w2_services
            and not declared_e8_h6144_short_names
            and not deep_h2880_services
            and not declared_e32_h2880_short_names
            and not declared_e32_h2880_names
            and not declared_e512_mid_depth_names
            and not declared_e128_bm64_names
            and not declared_e128_h2880_1084_names
            and not declared_bm16_h2880_names
            and not declared_sparse_w2_names
            and not declared_e128_shallow_names
            and not declared_e128_mid_depth_names
            and not declared_e8_shallow_names
            and not declared_e8_h6144_names):
        return None
    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    e32_n_tiles_by_variant = {"w1": 90, "w2": 45}
    e512_service_by_variant = {
        "w1": _NVIDIA_E512_1001_ACTIVE_1144_MAX_W1_SERVICE,
        "w2": _NVIDIA_E512_1001_ACTIVE_1144_MAX_W2_SERVICE,
    }
    experts = int(kernel.shape["experts"])
    _, local_counts = _routing(kernel)
    n_tiles = math.ceil(projection_n / tile_n)
    if kernel.launch.grid_size % n_tiles:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE grid must contain whole M-tile rows")
    m_tile_census = tuple(
        math.ceil(rows / tile_m) if rows > 0 else 0
        for rows in local_counts)
    topology = (
        sum(m_tile_census),
        kernel.launch.grid_size // n_tiles,
    )
    routed_rows = sum(local_counts)
    if kernel.dtype != "bf16":
        return None
    if ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e8_h6144_names
            and experts == 8
            and topology == (7, 8)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (96, 1024))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (256, 192)))):
        service_name = {
            "w1": _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e8_shallow_names
            and experts == 8
            and topology == (7, 8)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (64, 896))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (224, 128)))):
        service_name = {
            "w1": _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e128_mid_depth_names
            and experts == 128
            and topology == (177, 248)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (32, 48))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (12, 64)))):
        service_name = {
            "w1": _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_e128_shallow_names
            and experts == 128
            and topology == (88, 128)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (64, 96))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (24, 128)))):
        service_name = {
            "w1": _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
        }[variant]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_sparse_w2_names
            and variant == "w2"
            and (math.ceil(projection_k / tile_k), n_tiles) == (8, 64)
            and (experts, topology) in {
                (256, (16, 241)),
                (512, (20, 482)),
            }):
        service_name = {
            (256, (16, 241)):
                _NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE,
            (512, (20, 482)):
                _NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE,
        }[(experts, topology)]
    elif ((tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0)
            and declared_bm16_h2880_names
            and experts == 128
            and math.ceil(projection_k / tile_k) == 45
            and (
                (variant == "w1" and n_tiles == 180)
                or (variant == "w2" and n_tiles == 90))):
        if ((variant == "w1" and topology == (31, 32))
                or (variant == "w2" and topology == (31, 122))):
            service_name = _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE
        elif topology == (53, 124):
            service_name = _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE
        else:
            return None
    elif ((tile_m, tile_n, tile_k, raw_group_size)
            != (64, 64, 32, 8.0)):
        return None
    elif (declared_e128_bm64_names
            and experts == 128
            and topology == (574, 638)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (64, 24))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (24, 32)))):
        service_name = {
            "w1": _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W1_SERVICE,
            "w2": _NVIDIA_BM64_E128_574_ACTIVE_638_MAX_W2_SERVICE,
        }[variant]
    elif (declared_e32_h2880_short_names
            and experts == 32
            and math.ceil(projection_k / tile_k) == 90
            and variant in ("w1", "w2")
            and n_tiles == e32_n_tiles_by_variant[variant]
            and topology == (32, 36)):
        service_by_topology = {
            (32, 36): {
                "w1": _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W1_SERVICE,
                "w2": _NVIDIA_E32_H2880_32_ACTIVE_36_MAX_W2_SERVICE,
            },
        }
        service_name = service_by_topology[topology][variant]
    elif (declared_e32_h2880_names
            and experts == 32
            and math.ceil(projection_k / tile_k) == 90
            and variant in ("w1", "w2")
            and n_tiles == e32_n_tiles_by_variant[variant]
            and topology == (143, 160)):
        service_by_topology = {
            (143, 160): {
                "w1": _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W1_SERVICE,
                "w2": _NVIDIA_E32_H2880_143_ACTIVE_160_MAX_W2_SERVICE,
            },
        }
        service_name = service_by_topology[topology][variant]
    elif (declared_e512_mid_depth_names
            and experts == 512
            and topology == (1001, 1144)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (128, 32))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (32, 64)))):
        service_name = e512_service_by_variant[variant]
    elif (declared_e128_h2880_1084_names
            and experts == 128
            and topology == (1084, 1150)
            and m_tile_census
            == _NVIDIA_E128_H2880_1084_ACTIVE_1150_M_TILE_CENSUS
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (90, 90))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (90, 45)))):
        service_name = {
            "w1": _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W1_SERVICE,
            "w2": _NVIDIA_E128_H2880_1084_ACTIVE_1150_EXACT_W2_SERVICE,
        }[variant]
    elif (deep_h2880_services
            and experts in (32, 128)
            and math.ceil(projection_k / tile_k) in (89, 90, 91)
            and (
                (variant == "w1" and n_tiles in (89, 90, 91))
                or (variant == "w2" and n_tiles in (44, 45, 46)))
            and topology[1] > (
                2 * hardware.num_compute_units * int(raw_group_size))):
        service_name = _NVIDIA_DEEP_H2880_W1_W2_SERVICE
    elif (deep_w2_services
            and variant == "w2"
            and experts == 512
            and (math.ceil(projection_k / tile_k), n_tiles) == (32, 64)
            and topology[1] > (
                2 * hardware.num_compute_units * int(raw_group_size))):
        service_name = _NVIDIA_E512_DEEP_W2_SERVICE
    elif (e128_deep_w2_services
            and variant == "w2"
            and experts == 128
            and (math.ceil(projection_k / tile_k), n_tiles) == (48, 64)
            and topology[1] > (
                2 * hardware.num_compute_units * int(raw_group_size))):
        service_name = _NVIDIA_E128_DEEP_W2_SERVICE
    elif (e128_h4096_deep_w1_services
            and variant == "w1"
            and experts == 128
            and source_rows == 32768
            and int(kernel.shape["topk"]) == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (128, 48)
            and topology == (4156, 4222)
            and m_tile_census
            == _NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_M_TILE_CENSUS):
        service_name = _NVIDIA_E128_H4096_4156_ACTIVE_4222_MAX_W1_SERVICE
    elif (declared_e128_h2048_names
            and experts == 128
            and int(kernel.shape["topk"]) == 16
            and routed_rows == 524288
            and topology == (8250, 8318)
            and m_tile_census
            == _NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_M_TILE_CENSUS
            and (
                (variant == "w1"
                 and source_rows == 32768
                 and (math.ceil(projection_k / tile_k), n_tiles) == (64, 24))
                or (variant == "w2"
                    and source_rows == routed_rows
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (24, 32)))):
        service_name = (
            _NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT[
                variant])
    elif (declared_e8_h6144_short_names
            and experts == 8
            and topology == (8, 9)
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (192, 512))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (512, 96)))):
        service_name = {
            "w1": _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W1_SERVICE,
            "w2": _NVIDIA_E8_H6144_8_ACTIVE_9_MAX_W2_SERVICE,
        }[variant]
    elif (e8_h6144_deep_w2_services
            and variant == "w2"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (512, 96)
            and topology[1] in (520, 1032)):
        service_name = _NVIDIA_E8_H6144_DEEP_W2_SERVICE
    elif (e8_h6144_260_active_264_w2_services
            and variant == "w2"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (512, 96)
            and topology == (260, 264)):
        service_name = _NVIDIA_E8_H6144_260_ACTIVE_264_MAX_W2_SERVICE
    elif (e8_h6144_deep_w1_services
            and variant == "w1"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (192, 512)
            and topology == (1028, 1032)
            and m_tile_census == _NVIDIA_E8_H6144_DEEP_W1_M_TILE_CENSUS):
        service_name = _NVIDIA_E8_H6144_DEEP_W1_SERVICE
    elif (declared_e8_h4096_exact_w2_names
            and variant == "w2"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (448, 64)
            and topology in _NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY
            and m_tile_census
            == _NVIDIA_E8_H4096_EXACT_W2_M_TILE_CENSUS_BY_TOPOLOGY[
                topology]):
        service_name = _NVIDIA_E8_H4096_EXACT_W2_SERVICE_BY_TOPOLOGY[
            topology]
    elif (e512_h4096_deep_w1_services
            and variant == "w1"
            and experts == 512
            and source_rows == 32768
            and routed_rows == 327680
            and topology == (5368, 5624)
            and m_tile_census == _NVIDIA_E512_H4096_DEEP_W1_M_TILE_CENSUS
            and (math.ceil(projection_k / tile_k), n_tiles) == (128, 32)):
        service_name = _NVIDIA_E512_H4096_DEEP_W1_SERVICE
    elif (declared_e256_h7168_mid_workspace_names
            and experts == 256
            and topology[1] == 1276
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (224, 64))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (64, 112)))):
        service_name = {
            "w1": _NVIDIA_E256_H7168_MID_WORKSPACE_W1_SERVICE,
            "w2": _NVIDIA_E256_H7168_MID_WORKSPACE_W2_SERVICE,
        }[variant]
    elif (e256_h7168_2166_w2_services
            and variant == "w2"
            and experts == 256
            and routed_rows == source_rows == 131072
            and topology == (2166, 2300)
            and (math.ceil(projection_k / tile_k), n_tiles) == (64, 112)):
        service_name = _NVIDIA_E256_H7168_2166_ACTIVE_2300_MAX_W2_SERVICE
    elif (e256_h7168_2166_w1_services
            and variant == "w1"
            and experts == 256
            and routed_rows == 131072
            and topology == (2166, 2300)
            and m_tile_census == _NVIDIA_E256_H7168_2166_W1_M_TILE_CENSUS
            and (math.ceil(projection_k / tile_k), n_tiles) == (224, 64)):
        service_name = _NVIDIA_E256_H7168_2166_EXACT_W1_SERVICE
    elif (declared_e256_h7168_names
            and experts == 256
            and topology == (4230, 4348)
            and m_tile_census == _NVIDIA_E256_H7168_DEEP_M_TILE_CENSUS
            and (
                (variant == "w1"
                 and (math.ceil(projection_k / tile_k), n_tiles)
                 == (224, 64))
                or (variant == "w2"
                    and (math.ceil(projection_k / tile_k), n_tiles)
                    == (64, 112)))):
        service_name = {
            "w1": _NVIDIA_E256_H7168_DEEP_W1_SERVICE,
            "w2": _NVIDIA_E256_H7168_DEEP_W2_SERVICE,
        }[variant]
    elif (declared_w1_names
            and variant == "w1"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (192, 512)
            and topology == (68, 72)):
        service_name = _NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE
    elif (declared_w1_names
            and variant == "w1"
            and experts == 8
            and (math.ceil(projection_k / tile_k), n_tiles) == (192, 512)
            and topology == (132, 136)):
        service_name = _NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE
    else:
        return None
    signature = dataclasses.replace(
        _gpu_grouped_operand_signature(kernel), access_class=service_name)
    matches = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == service_name
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature == signature)
    if len(matches) != 1:
        raise ValueError(
            "NVIDIA grouped FUSED_MOE source depth requires one exact "
            f"{service_name!r} signature")
    return matches[0]


def _metax_multi_tile_operand_pipeline_service(
        kernel: KernelSpec, hardware):
    """Select C500's source-shaped expert operand-pipeline envelope.

    The BM16 source clamps its launch storage for fewer than one M tile of
    activation rows. The BM64 source keeps one expert's weight pointer live
    across consecutive M tiles. Synthetic-routing mechanism probes separate
    these source-shaped lifetimes and their long-K versus wide-N pipelines;
    no production duration or case identity participates in the selection.
    """

    service_names = (
        _METAX_GROUPED_OPERAND_LONG_K_SERVICE,
        _METAX_GROUPED_OPERAND_WIDE_N_SERVICE,
        _METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE,
        _METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE,
        _METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE,
        _METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE,
        _METAX_REPEATED_OPERAND_DEEP_K_SERVICE,
        _METAX_REPEATED_OPERAND_WIDE_N_SERVICE,
        _METAX_REPEATED_OPERAND_BALANCED_N_SERVICE,
        _METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE,
        _METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE,
        _METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE,
        _METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE,
        _METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE,
        _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE,
        _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE,
        _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE,
        _METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W2_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K128_N448_W1_SERVICE,
        _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K448_N64_W2_SERVICE,
        _METAX_E32_32_ACTIVE_48_MAX_W1_SERVICE,
        _METAX_E32_32_ACTIVE_48_MAX_W2_SERVICE,
        _METAX_E32_47_ACTIVE_64_MAX_W1_SERVICE,
        _METAX_E32_47_ACTIVE_64_MAX_W2_SERVICE,
        _METAX_E32_78_ACTIVE_96_MAX_W1_SERVICE,
        _METAX_E32_78_ACTIVE_96_MAX_W2_SERVICE,
        _METAX_E32_143_ACTIVE_160_MAX_W1_SERVICE,
        _METAX_E32_143_ACTIVE_160_MAX_W2_SERVICE,
        _METAX_E256_LONG_LIFETIME_LONG_K_SERVICE,
        _METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE,
        _METAX_E256_LONG_LIFETIME_MID_W1_SERVICE,
        _METAX_E256_LONG_LIFETIME_MID_W2_SERVICE,
        _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE,
        _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE,
        _METAX_E512_DENSE_LARGE_W2_SERVICE,
        _METAX_E512_1010_ACTIVE_1144_MAX_W2_SERVICE,
        _METAX_E128_LONG_LIFETIME_W1_SERVICE,
        _METAX_E128_LONG_LIFETIME_W2_SERVICE,
        _METAX_E128_DEEP_W2_SERVICE,
        _METAX_E128_EIGHT_TILE_W1_SERVICE,
        _METAX_E128_EIGHT_TILE_W2_SERVICE,
        _METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE,
        _METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_DEEP_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W1_SERVICE,
        _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W2_SERVICE,
        _METAX_E128_SPARSE_TWO_TILE_W1_SERVICE,
        _METAX_E128_SPARSE_TWO_TILE_W2_SERVICE,
        _METAX_E128_188_ACTIVE_254_MAX_W1_SERVICE,
        _METAX_E128_188_ACTIVE_254_MAX_W2_SERVICE,
        _METAX_E128_SINGLE_TILE_W1_SERVICE,
        _METAX_E128_SINGLE_TILE_W2_SERVICE,
        _METAX_E128_SINGLE_TILE_MID_W1_SERVICE,
        _METAX_E128_SINGLE_TILE_MID_W2_SERVICE,
        _METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE,
        _METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE,
        _METAX_E128_UPPER_TWO_TILE_W1_SERVICE,
        _METAX_E128_UPPER_TWO_TILE_W2_SERVICE,
        _METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE,
        _METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE,
        _METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE,
        _METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE,
        _METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE,
        _METAX_E128_REPEATED_K24_N32_W2_SERVICE,
        _METAX_E256_EIGHT_TILE_LONG_K_SERVICE,
        _METAX_E256_EIGHT_TILE_WIDE_N_SERVICE,
        _METAX_E256_EIGHT_TILE_MID_W1_SERVICE,
        _METAX_E256_EIGHT_TILE_MID_W2_SERVICE,
        _METAX_E256_FOUR_TILE_LONG_K_SERVICE,
        _METAX_E256_FOUR_TILE_WIDE_N_SERVICE,
        _METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE,
        _METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE,
        _METAX_E128_FOUR_TILE_W1_SERVICE,
        _METAX_E128_FOUR_TILE_W2_SERVICE,
        _METAX_E128_FOUR_TILE_LARGE_W1_SERVICE,
        _METAX_E128_FOUR_TILE_LARGE_W2_SERVICE,
        _METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE,
        _METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE,
        _METAX_E256_SINGLE_TILE_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_W2_SERVICE,
        _METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE,
        _METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE,
        _METAX_E512_SINGLE_TILE_W1_SERVICE,
        _METAX_E512_SINGLE_TILE_W2_SERVICE,
        _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE,
        _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE,
        _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE,
        _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE,
        _METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE,
        _METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE,
        _METAX_BM16_E256_SIXTEEN_ACTIVE_DEEP_W2_SERVICE,
        _METAX_BM16_SPARSE_PADDED_W2_SERVICE,
        _METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE,
        _METAX_BM16_E512_FORTY_ACTIVE_W1_SERVICE,
        _METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE,
        _METAX_BM16_E512_FORTY_ACTIVE_K8_N64_W2_SERVICE,
        _METAX_BM16_E512_75_ACTIVE_80_MAX_W1_SERVICE,
        _METAX_BM16_E512_75_ACTIVE_485_MAX_W2_SERVICE,
        _METAX_BM16_E512_TWENTY_ACTIVE_W1_SERVICE,
        _METAX_BM16_E512_TWENTY_ACTIVE_W2_SERVICE,
        _METAX_BM16_E512_TWENTY_ACTIVE_K8_N64_W2_SERVICE,
        _METAX_BM16_E512_TEN_ACTIVE_W1_SERVICE,
        _METAX_BM16_E512_TEN_ACTIVE_W2_SERVICE,
        _METAX_BM16_E512_143_ACTIVE_490_MAX_W1_SERVICE,
        _METAX_BM16_E512_143_ACTIVE_490_MAX_W2_SERVICE,
        _METAX_BM16_E512_MID_SPARSE_W1_SERVICE,
        _METAX_BM16_E512_MID_SPARSE_W2_SERVICE,
        _METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE,
        _METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE,
        _METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE,
        _METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE,
        _METAX_BM16_31_ACTIVE_32_MAX_W1_SERVICE,
        _METAX_BM16_E128_31_ACTIVE_W2_SERVICE,
        _METAX_BM16_E128_53_ACTIVE_64_MAX_W1_SERVICE,
        _METAX_BM16_E128_53_ACTIVE_124_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK8_16_ACTIVE_16_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK8_16_ACTIVE_121_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_16_ACTIVE_121_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_16_ACTIVE_16_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE,
        _METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE,
        _METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE,
        _METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE,
        _METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W1_SERVICE,
        _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE,
        _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_36_MAX_W1_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_36_MAX_W2_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_40_MAX_W1_SERVICE,
        _METAX_BM64_E32_32_ACTIVE_40_MAX_W2_SERVICE,
    )
    declared = tuple(
        service for service in hardware.memory_access_class_services
        if service.service in service_names)
    if not declared:
        return None
    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    _, local_counts = _routing(kernel)
    populated_experts = sum(rows > 0 for rows in local_counts)
    routed_rows = sum(local_counts)
    populated_m_tiles = sum(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    projection_n_tiles = math.ceil(projection_n / tile_n)
    if kernel.launch.grid_size % projection_n_tiles:
        raise ValueError(
            "MetaX grouped FUSED_MOE grid must contain whole M-tile rows")
    maximum_m_tiles = kernel.launch.grid_size // projection_n_tiles
    if (tile_m, tile_n, tile_k, raw_group_size) == (16, 32, 64, 1.0):
        projection_tile_class = (
            math.ceil(projection_k / tile_k), projection_n_tiles)
        if (source_rows == 16
                and routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 241
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE
        elif (variant == "w2"
                and int(kernel.shape["experts"]) == 256
                and int(kernel.shape["topk"]) == 8
                and source_rows == routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 241
                and projection_tile_class == (32, 224)):
            service_name = (
                _METAX_BM16_E256_SIXTEEN_ACTIVE_DEEP_W2_SERVICE)
        elif (source_rows == 80
                and populated_experts == populated_m_tiles == 75
                and maximum_m_tiles == 485
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and int(kernel.shape["topk"]) == 10
                and variant == "w1"
                and source_rows == 4
                and routed_rows == 40
                and populated_experts == populated_m_tiles == 40
                and maximum_m_tiles == 40
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_FORTY_ACTIVE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and int(kernel.shape["topk"]) == 10
                and variant == "w2"
                and source_rows == routed_rows == 40
                and populated_experts == populated_m_tiles == 40
                and maximum_m_tiles == 483
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_FORTY_ACTIVE_K8_N64_W2_SERVICE
        elif (source_rows == 40
                and routed_rows == 40
                and populated_experts == populated_m_tiles == 40
                and maximum_m_tiles == 483
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and source_rows == 8
                and routed_rows == 80
                and populated_experts == populated_m_tiles == 75
                and maximum_m_tiles == 80
                and projection_tile_class == (64, 64)):
            service_name = _METAX_BM16_E512_75_ACTIVE_80_MAX_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and source_rows == 80
                and routed_rows == 80
                and populated_experts == populated_m_tiles == 75
                and maximum_m_tiles == 485
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_75_ACTIVE_485_MAX_W2_SERVICE
        elif (source_rows == 2
                and routed_rows == 20
                and populated_experts == populated_m_tiles == 20
                and maximum_m_tiles == 20
                and projection_tile_class == (64, 64)):
            service_name = _METAX_BM16_E512_TWENTY_ACTIVE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and int(kernel.shape["topk"]) == 10
                and variant == "w2"
                and source_rows == routed_rows == 20
                and populated_experts == populated_m_tiles == 20
                and maximum_m_tiles == 482
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_TWENTY_ACTIVE_K8_N64_W2_SERVICE
        elif (source_rows == 20
                and routed_rows == 20
                and populated_experts == populated_m_tiles == 20
                and maximum_m_tiles == 482
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_TWENTY_ACTIVE_W2_SERVICE
        elif (source_rows == 1
                and routed_rows == 10
                and populated_experts == populated_m_tiles == 10
                and maximum_m_tiles == 10
                and projection_tile_class == (64, 64)):
            service_name = _METAX_BM16_E512_TEN_ACTIVE_W1_SERVICE
        elif (source_rows == 10
                and routed_rows == 10
                and populated_experts == populated_m_tiles == 10
                and maximum_m_tiles == 10
                and projection_tile_class == (16, 128)):
            service_name = _METAX_BM16_E512_TEN_ACTIVE_W2_SERVICE
        elif (source_rows == 16
                and routed_rows == 160
                and populated_experts == populated_m_tiles == 143
                and maximum_m_tiles == 490
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_143_ACTIVE_490_MAX_W1_SERVICE
        elif (source_rows == 160
                and routed_rows == 160
                and populated_experts == populated_m_tiles == 143
                and maximum_m_tiles == 490
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_143_ACTIVE_490_MAX_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w1"
                and 16 < source_rows <= 64
                and routed_rows == 10 * source_rows
                and populated_experts == populated_m_tiles
                and 143 < populated_m_tiles <= 362
                and 490 < maximum_m_tiles <= 520
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_MID_SPARSE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w2"
                and 160 < source_rows <= 640
                and routed_rows == source_rows
                and populated_experts == populated_m_tiles
                and 143 < populated_m_tiles <= 362
                and 490 < maximum_m_tiles <= 520
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_MID_SPARSE_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w1"
                and source_rows == 512
                and routed_rows == 5120
                and populated_experts == 512
                and populated_m_tiles == 521
                and maximum_m_tiles == 800
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 512
                and variant == "w2"
                and source_rows == routed_rows == 5120
                and populated_experts == 512
                and populated_m_tiles == 521
                and maximum_m_tiles == 800
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE
        elif (source_rows == 8
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 57
                and maximum_m_tiles == 64
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE
        elif (source_rows == 64
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 57
                and maximum_m_tiles == 244
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE
        elif (source_rows == 16
                and routed_rows == 128
                and populated_experts == populated_m_tiles == 106
                and maximum_m_tiles == 248
                and projection_tile_class == (32, 32)):
            service_name = _METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE
        elif (source_rows == 128
                and routed_rows == 128
                and populated_experts == populated_m_tiles == 106
                and maximum_m_tiles == 248
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE
        elif (source_rows < tile_m
                and routed_rows == 32
                and populated_experts == populated_m_tiles == 31
                and maximum_m_tiles == 32
                and projection_tile_class in ((32, 32), (32, 48))):
            service_name = _METAX_BM16_31_ACTIVE_32_MAX_W1_SERVICE
        elif (source_rows == 32
                and routed_rows == 32
                and populated_experts == populated_m_tiles == 31
                and maximum_m_tiles == 122
                and projection_tile_class == (12, 64)):
            service_name = _METAX_BM16_E128_31_ACTIVE_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 8
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 53
                and maximum_m_tiles == 64
                and projection_tile_class == (32, 48)):
            service_name = _METAX_BM16_E128_53_ACTIVE_64_MAX_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 64
                and routed_rows == 64
                and populated_experts == populated_m_tiles == 53
                and maximum_m_tiles == 124
                and projection_tile_class == (12, 64)):
            service_name = _METAX_BM16_E128_53_ACTIVE_124_MAX_W2_SERVICE
        elif (variant == "w1"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 8
                and source_rows == 2
                and routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 16
                and projection_tile_class == (32, 48)):
            service_name = _METAX_BM16_E128_TOPK8_16_ACTIVE_16_MAX_W1_SERVICE
        elif (variant == "w2"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 8
                and source_rows == routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 121
                and projection_tile_class == (12, 64)):
            service_name = (
                _METAX_BM16_E128_TOPK8_16_ACTIVE_121_MAX_W2_SERVICE)
        elif (variant == "w2"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 16
                and source_rows == routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 121
                and projection_tile_class == (12, 64)):
            service_name = (
                _METAX_BM16_E128_TOPK16_16_ACTIVE_121_MAX_W2_SERVICE)
        elif (variant == "w1"
                and int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 16
                and source_rows == 1
                and routed_rows == 16
                and populated_experts == populated_m_tiles == 16
                and maximum_m_tiles == 16
                and projection_tile_class == (32, 48)):
            service_name = _METAX_BM16_E128_TOPK16_16_ACTIVE_16_MAX_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and int(kernel.shape["topk"]) == 16
                and int(kernel.shape["tokens"]) in (4, 8, 16, 32, 64, 128)
                and projection_tile_class in ((32, 48), (12, 64))):
            source_depth_services = {
                (4, 64, (32, 48)):
                    _METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE,
                (4, 124, (12, 64)):
                    _METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE,
                (8, 128, (32, 48)):
                    _METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE,
                (8, 128, (12, 64)):
                    _METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE,
                (16, 136, (32, 48)):
                    _METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE,
                (16, 136, (12, 64)):
                    _METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE,
                (32, 152, (32, 48)):
                    _METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE,
                (32, 152, (12, 64)):
                    _METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE,
                (64, 184, (32, 48)):
                    _METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE,
                (64, 184, (12, 64)):
                    _METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE,
                (128, 248, (32, 48)):
                    _METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE,
                (128, 248, (12, 64)):
                    _METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE,
            }
            source_depth_key = (
                int(kernel.shape["tokens"]),
                maximum_m_tiles,
                projection_tile_class,
            )
            if source_depth_key not in source_depth_services:
                return None
            active_tile_bounds = {
                4: (45, 58),
                8: (80, 92),
                16: (104, 120),
                32: (120, 132),
                64: (120, 136),
                128: (168, 190),
            }
            lower_active_tiles, upper_active_tiles = active_tile_bounds[
                int(kernel.shape["tokens"])]
            if not (lower_active_tiles <= populated_m_tiles
                    <= upper_active_tiles):
                return None
            service_name = source_depth_services[source_depth_key]
        elif (source_rows == 1
                and routed_rows == 4
                and populated_experts == populated_m_tiles == 4
                and maximum_m_tiles == 4
                and projection_tile_class == (45, 180)):
            service_name = _METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE
        elif (source_rows == 4
                and routed_rows == 4
                and populated_experts == populated_m_tiles == 4
                and maximum_m_tiles == 4
                and projection_tile_class == (45, 90)):
            service_name = _METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 2
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 8
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 180)):
            service_name = _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W1_SERVICE
        elif (int(kernel.shape["experts"]) == 128
                and source_rows == 8
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 8
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 90)):
            service_name = _METAX_BM16_E128_EIGHT_ACTIVE_MID_K_W2_SERVICE
        elif (source_rows == 8
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 8
                and maximum_m_tiles == 8
                and projection_tile_class == (12, 64)):
            service_name = _METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE
        elif (source_rows == 2
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 7
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 180)):
            service_name = _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE
        elif (source_rows == 8
                and routed_rows == 8
                and populated_experts == populated_m_tiles == 7
                and maximum_m_tiles == 8
                and projection_tile_class == (45, 90)):
            service_name = _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE
        elif (source_rows == 32
                and populated_experts == populated_m_tiles == 31
                and maximum_m_tiles == 242
                and projection_tile_class == (8, 64)):
            service_name = _METAX_BM16_SPARSE_PADDED_W2_SERVICE
        elif (variant == "w2"
                and source_rows < tile_m
                and populated_m_tiles == populated_experts == routed_rows
                and projection_k <= projection_n
                and projection_tile_class[0] == 8):
            service_name = _METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE
        else:
            service_name = None
        if service_name is None:
            return None
    elif (tile_m, tile_n, tile_k, raw_group_size) != (64, 64, 32, 8.0):
        return None
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles == 12
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = (
            _METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE)
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 10
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (192, 512)):
        service_name = _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 10
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (512, 96)):
        service_name = _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 9
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (128, 448)):
        service_name = _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and maximum_m_tiles <= 9
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and routed_rows >= 2 * tile_m
            and (projection_k, projection_n) == (14336, 4096)):
        service_name = _METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE
    elif (populated_m_tiles == populated_experts == 8
            and routed_rows >= 2 * tile_m
            and (projection_k, projection_n) == (4096, 28672)):
        service_name = _METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 21
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (128, 448)):
        service_name = _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 21
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 20
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (192, 512)):
        service_name = (
            _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W1_SERVICE)
    elif (populated_experts == 8
            and populated_m_tiles == 20
            and maximum_m_tiles == 24
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (512, 96)):
        service_name = (
            _METAX_EIGHT_EXPERT_TWENTY_ACTIVE_TWENTY_FOUR_MAX_W2_SERVICE)
    elif populated_experts == 8 and populated_m_tiles == 12:
        projection_tile_class = (
            math.ceil(projection_k / tile_k),
            math.ceil(projection_n / tile_n),
        )
        service_name = (
            _METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE
            if projection_tile_class == (128, 448)
            else _METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE
            if projection_tile_class == (192, 512)
            else _METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE
            if projection_tile_class == (448, 64)
            else _METAX_EIGHT_EXPERT_TWELVE_TILE_DEEP_W2_SERVICE
            if projection_tile_class == (512, 96)
            else None)
        if service_name is None:
            return None
    elif (populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 48
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_32_ACTIVE_48_MAX_W1_SERVICE
    elif (populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 48
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_32_ACTIVE_48_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and populated_experts == populated_m_tiles == 32
            and routed_rows == 256
            and maximum_m_tiles == 36
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_BM64_E32_32_ACTIVE_36_MAX_W1_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and populated_experts == populated_m_tiles == 32
            and routed_rows == 256
            and maximum_m_tiles == 36
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_BM64_E32_32_ACTIVE_36_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and variant == "w1"
            and source_rows == 128
            and routed_rows == 512
            and populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 40
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_BM64_E32_32_ACTIVE_40_MAX_W1_SERVICE
    elif (int(kernel.shape["experts"]) == 32
            and variant == "w2"
            and source_rows == routed_rows == 512
            and populated_experts == populated_m_tiles == 32
            and maximum_m_tiles == 40
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_BM64_E32_32_ACTIVE_40_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 512
            and source_rows == routed_rows == 40960
            and populated_experts == 512
            and populated_m_tiles == 1010
            and maximum_m_tiles == 1144
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (16, 32)):
        service_name = _METAX_E512_1010_ACTIVE_1144_MAX_W2_SERVICE
    elif (int(kernel.shape["experts"]) == 512
            and variant == "w1"
            and source_rows == 1024
            and routed_rows == 10240
            and populated_experts == populated_m_tiles == 512
            and maximum_m_tiles == 664
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (64, 16)):
        service_name = _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE
    elif (int(kernel.shape["experts"]) == 512
            and variant == "w2"
            and source_rows == routed_rows == 10240
            and populated_experts == populated_m_tiles == 512
            and maximum_m_tiles == 664
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (16, 32)):
        service_name = _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE
    elif populated_m_tiles <= populated_experts:
        projection_tile_class = (
            math.ceil(projection_k / tile_k),
            math.ceil(projection_n / tile_n),
        )
        if (populated_experts == populated_m_tiles == 128
                and routed_rows == 4096
                and maximum_m_tiles == 190
                and source_rows == 512
                and int(kernel.shape["topk"]) == 8
                and projection_tile_class == (64, 24)):
            service_name = _METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and routed_rows == 4096
                and maximum_m_tiles == 190
                and source_rows == 256
                and int(kernel.shape["topk"]) == 16
                and projection_tile_class == (64, 24)):
            service_name = _METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and routed_rows == source_rows == 4096
                and maximum_m_tiles == 190
                and projection_tile_class == (24, 32)):
            service_name = _METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles <= 158
                and projection_tile_class == (64, 24)):
            service_name = _METAX_E128_SINGLE_TILE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles <= 158
                and projection_tile_class == (24, 32)):
            service_name = _METAX_E128_SINGLE_TILE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 190
                and projection_tile_class == (90, 90)):
            service_name = _METAX_E128_SINGLE_TILE_MID_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 190
                and projection_tile_class == (90, 45)):
            service_name = _METAX_E128_SINGLE_TILE_MID_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 158
                and projection_tile_class == (128, 48)):
            service_name = _METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 128
                and maximum_m_tiles == 158
                and projection_tile_class == (48, 64)):
            service_name = _METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (64, 16)):
            service_name = _METAX_E256_SINGLE_TILE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (16, 32)):
            service_name = _METAX_E256_SINGLE_TILE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (64, 16)):
            service_name = _METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (16, 32)):
            service_name = _METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (224, 64)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 316
                and projection_tile_class == (64, 112)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (224, 64)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 256
                and maximum_m_tiles == 380
                and projection_tile_class == (64, 112)):
            service_name = _METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE
        elif (populated_experts == populated_m_tiles == 512
                and maximum_m_tiles == 824
                and projection_tile_class == (128, 32)):
            service_name = _METAX_E512_SINGLE_TILE_W1_SERVICE
        elif (populated_experts == populated_m_tiles == 512
                and maximum_m_tiles == 824
                and projection_tile_class == (32, 64)):
            service_name = _METAX_E512_SINGLE_TILE_W2_SERVICE
        else:
            return None
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 32)):
        service_name = _METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (32, 64)):
        service_name = _METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 2 * populated_experts):
        service_name = (
            _METAX_GROUPED_OPERAND_LONG_K_SERVICE
            if projection_k > projection_n
            else _METAX_GROUPED_OPERAND_WIDE_N_SERVICE)
    elif (populated_experts == 512
            and routed_rows == 327680
            and 10 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and maximum_m_tiles == 5624
            and source_rows == 32768
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (64, 16)):
        service_name = _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE
    elif (populated_experts == 512
            and routed_rows == 327680
            and 10 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and maximum_m_tiles == 5624
            and source_rows == 327680
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (16, 32)):
        service_name = _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE
    elif (populated_experts == 512
            and routed_rows == 327680
            and 10 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and maximum_m_tiles == 5624
            and source_rows == 327680
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (32, 64)):
        service_name = _METAX_E512_DENSE_LARGE_W2_SERVICE
    elif (populated_experts >= 4 * hardware.num_compute_units
            and populated_m_tiles <= 11 * populated_experts):
        service_name = (
            _METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE
            if projection_k > projection_n
            else _METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE)
    elif (populated_experts == 128
            and routed_rows == 16384
            and 315 <= populated_m_tiles <= 319
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 24)
            and source_rows == 2048
            and int(kernel.shape["topk"]) == 8):
        service_name = _METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE
    elif (populated_experts == 128
            and routed_rows == 16384
            and 315 <= populated_m_tiles <= 319
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 24)
            and source_rows == 1024
            and int(kernel.shape["topk"]) == 16):
        service_name = _METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE
    elif (populated_experts == 128
            and routed_rows == source_rows == 16384
            and 315 <= populated_m_tiles <= 319
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (24, 32)):
        service_name = _METAX_E128_REPEATED_K24_N32_W2_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 16)):
        service_name = _METAX_E256_LONG_LIFETIME_MID_W1_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (16, 32)):
        service_name = _METAX_E256_LONG_LIFETIME_MID_W2_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_LONG_LIFETIME_LONG_K_SERVICE
    elif (populated_experts == 256
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 16)):
        service_name = _METAX_E256_EIGHT_TILE_MID_W1_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (16, 32)):
        service_name = _METAX_E256_EIGHT_TILE_MID_W2_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_EIGHT_TILE_LONG_K_SERVICE
    elif (populated_experts == 256
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_EIGHT_TILE_WIDE_N_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 16)):
        service_name = _METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (16, 32)):
        service_name = _METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_FOUR_TILE_LONG_K_SERVICE
    elif (populated_experts == 256
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_FOUR_TILE_WIDE_N_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 90)):
        service_name = _METAX_E128_LONG_LIFETIME_W1_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_E128_LONG_LIFETIME_W2_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_DEEP_W2_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and int(kernel.shape["topk"]) == 16
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 24)):
        service_name = _METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 90)):
        service_name = _METAX_E128_EIGHT_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_E128_EIGHT_TILE_W2_SERVICE
    elif (populated_experts == 128
            and 5 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 573
            and maximum_m_tiles == 638
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_FOUR_TILE_LARGE_W1_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 573
            and maximum_m_tiles == 638
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_FOUR_TILE_LARGE_W2_SERVICE
    elif (populated_experts == 128
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 90)):
        service_name = _METAX_E128_FOUR_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 4 * populated_experts < populated_m_tiles
            and populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_E128_FOUR_TILE_W2_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 188
            and maximum_m_tiles == 254
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_188_ACTIVE_254_MAX_W1_SERVICE
    elif (populated_experts == 128
            and populated_m_tiles == 188
            and maximum_m_tiles == 254
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_188_ACTIVE_254_MAX_W2_SERVICE
    elif (populated_experts == 128
            and populated_experts < populated_m_tiles
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_SPARSE_TWO_TILE_W1_SERVICE
    elif (populated_experts == 128
            and populated_experts < populated_m_tiles
            and populated_m_tiles <= 2 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_SPARSE_TWO_TILE_W2_SERVICE
    elif (populated_experts == 128
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (128, 48)):
        service_name = _METAX_E128_UPPER_TWO_TILE_W1_SERVICE
    elif (populated_experts == 128
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and maximum_m_tiles == 382
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (48, 64)):
        service_name = _METAX_E128_UPPER_TWO_TILE_W2_SERVICE
    elif (populated_experts == 256
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (224, 64)):
        service_name = _METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE
    elif (populated_experts == 256
            and 2 * populated_experts < populated_m_tiles
            and 2 * populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (64, 112)):
        service_name = _METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k > projection_n
            and math.ceil(projection_k / tile_k) == 64
            and math.ceil(projection_n / tile_n) == 16):
        service_name = _METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k <= projection_n
            and math.ceil(projection_k / tile_k) == 16
            and math.ceil(projection_n / tile_n) == 32):
        service_name = _METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k > projection_n
            and math.ceil(projection_k / tile_k) >= 220):
        service_name = _METAX_REPEATED_OPERAND_DEEP_K_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (90, 45)):
        service_name = _METAX_REPEATED_OPERAND_BALANCED_N_SERVICE
    elif (populated_m_tiles >= hardware.num_compute_units
            and 2 * populated_m_tiles <= 5 * populated_experts
            and projection_k <= projection_n
            and math.ceil(projection_k / tile_k) >= 62):
        service_name = _METAX_REPEATED_OPERAND_WIDE_N_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (192, 512)):
        service_name = _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (512, 96)):
        service_name = _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_W2_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (128, 448)):
        service_name = (
            _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K128_N448_W1_SERVICE)
    elif (populated_experts == 8
            and populated_m_tiles == 68
            and maximum_m_tiles == 72
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (448, 64)):
        service_name = (
            _METAX_EIGHT_EXPERT_68_ACTIVE_72_MAX_K448_N64_W2_SERVICE)
    elif (populated_experts == 32
            and populated_m_tiles == 47
            and maximum_m_tiles == 64
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_47_ACTIVE_64_MAX_W1_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 47
            and maximum_m_tiles == 64
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_47_ACTIVE_64_MAX_W2_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 78
            and maximum_m_tiles == 96
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_78_ACTIVE_96_MAX_W1_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 78
            and maximum_m_tiles == 96
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_78_ACTIVE_96_MAX_W2_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 143
            and maximum_m_tiles == 160
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 90)):
        service_name = _METAX_E32_143_ACTIVE_160_MAX_W1_SERVICE
    elif (populated_experts == 32
            and populated_m_tiles == 143
            and maximum_m_tiles == 160
            and (
                math.ceil(projection_k / tile_k),
                projection_n_tiles,
            ) == (90, 45)):
        service_name = _METAX_E32_143_ACTIVE_160_MAX_W2_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles > 11 * populated_experts
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) == (192, 512)):
        service_name = _METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE
    elif (populated_experts == 8
            and populated_m_tiles > 11 * populated_experts
            and projection_k <= projection_n
            and (
                math.ceil(projection_k / tile_k),
                math.ceil(projection_n / tile_n),
            ) in ((128, 448), (192, 512))):
        service_name = _METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE
    else:
        return None
    signature = dataclasses.replace(
        _gpu_grouped_operand_signature(kernel), access_class=service_name)
    matches = tuple(
        service for service in declared
        if service.service == service_name
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature == signature)
    if len(matches) != 1:
        raise ValueError(
            "MetaX multi-tile grouped operand pipeline requires one exact "
            f"{service_name!r} signature")
    return matches[0]


def _gpu_split_operand_signature(
        kernel: KernelSpec, service: str,
        operand_role: str) -> MemoryAccessClassSignature:
    """Return one exact BM64 hit/miss service signature."""

    combined = _gpu_grouped_operand_signature(kernel)
    if operand_role == "combined":
        logical_row_bytes = combined.logical_row_bytes
    elif operand_role == "weight":
        _, tile_n, tile_k = _tile(kernel)
        logical_row_bytes = (
            tile_k * tile_n * int(kernel.shape["dtype_bytes"]))
    else:
        raise ValueError(
            "GPU split operand role must be combined or weight")
    return dataclasses.replace(
        combined,
        access_class=service,
        logical_row_bytes=logical_row_bytes,
        operand_role=operand_role,
    )


def _gpu_split_operand_services(kernel: KernelSpec, hardware):
    """Resolve the complete exact BM64 L2-hit/weight-miss pair."""

    tile_m, _, _ = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    _, source_rows, projection_k, _ = _projection(kernel)
    source_footprint_bytes = (
        source_rows * projection_k * _gpu_operand_element_bytes(kernel))
    _, local_counts = _routing(kernel)
    expert_m_tiles = tuple(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    overlaps_weight_miss = (
        source_footprint_bytes <= hardware.l2_size_bytes
        and bool(expert_m_tiles)
        and all(2 <= tiles <= group_size_m for tiles in expert_m_tiles))
    weight_service = (
        _GROUPED_WEIGHT_OVERLAPPED_DRAM_MISS_SERVICE
        if overlaps_weight_miss
        else _GROUPED_OPERAND_PIPELINE_SERVICE)
    specifications = (
        (_GROUPED_OPERAND_PIPELINE_SERVICE, "combined"),
        (weight_service, "weight"),
    )
    signatures = tuple(
        _gpu_split_operand_signature(kernel, service, operand_role)
        for service, operand_role in specifications)
    matches = tuple(
        tuple(
            candidate
            for candidate in hardware.memory_access_class_services
            if candidate.service == service
            and candidate.access_kind == "read"
            and candidate.ingress_resource_role == "l2_cache"
            and candidate.signature == signature)
        for (service, _), signature in zip(specifications, signatures))
    if not any(matches):
        return None
    if any(len(values) != 1 for values in matches):
        raise ValueError(
            "GPU grouped operand L2-hit/weight-miss services must be "
            "complete and exact")
    return {
        operand_role: values[0]
        for (_, operand_role), values in zip(specifications, matches)
    }


def _gpu_full_group_operand_services(kernel: KernelSpec, hardware):
    """Resolve exact complete-GSM8 services by weight-stream topology."""

    combined = _gpu_split_operand_signature(
        kernel, _GROUPED_OPERAND_PIPELINE_SERVICE, "combined")
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    services = {}
    for weight_streams in (
            value for value in _GROUPED_OPERAND_STREAM_COUNTS
                if value <= group_size_m):
        signature = dataclasses.replace(
            combined,
            logical_row_bytes=(
                    combined.logical_row_bytes * group_size_m),
            weight_streams_per_group=weight_streams,
        )
        matches = tuple(
            candidate
            for candidate in hardware.memory_access_class_services
            if candidate.service == _GROUPED_OPERAND_PIPELINE_SERVICE
            and candidate.access_kind == "read"
            and candidate.ingress_resource_role == "l2_cache"
            and candidate.signature == signature)
        if len(matches) > 1:
            raise ValueError(
                "GPU grouped operand repeats an exact full-group service")
        if matches:
            services[weight_streams] = matches[0]
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    exact_five_three_services = tuple(
        service for service in hardware.memory_access_class_services
        if service.service == _NVIDIA_BM64_GSM8_FIVE_THREE_W1_SERVICE)
    if len(exact_five_three_services) > 1:
        raise ValueError(
            "GPU grouped operand repeats its exact 5+3 service")
    if exact_five_three_services:
        if not (
                variant == "w1"
                and int(kernel.shape["experts"]) == 256
                and int(kernel.shape["topk"]) == 8
                and source_rows >= 8192
                and (projection_k, projection_n) == (7168, 4096)
                and (tile_m, tile_n, tile_k, group_size_m) == (64, 64, 32, 8)):
            return services
        signature = dataclasses.replace(
            combined,
            access_class=_NVIDIA_BM64_GSM8_FIVE_THREE_W1_SERVICE,
            logical_row_bytes=combined.logical_row_bytes * group_size_m,
            weight_streams_per_group=2,
        )
        matches = tuple(
            service for service in exact_five_three_services
            if service.access_kind == "read"
            and service.ingress_resource_role == "l2_cache"
            and service.signature == signature)
        if len(matches) != 1:
            raise ValueError(
                "GPU grouped operand exact 5+3 service requires one "
                "matching signature")
        services[(5, 3)] = matches[0]
    if services and 1 not in services:
        raise ValueError(
            "GPU grouped operand mixed-stream services require the "
            "single-stream service")
    return services


def _gpu_full_group_operand_bytes_by_weight_streams(
        kernel: KernelSpec, *, projection_k: int,
        projection_n: int) -> dict:
    """Return complete-GSM8 operand bytes by distinct expert streams.

    Triton's source order flattens populated expert M tiles and forms fixed
    groups of ``GROUP_SIZE_M``.  The compact sequence of expert M-tile runs
    inside each full group carries its ordered weight-stream topology.
    Counting runs and tail-row padding is O(experts), without CTA expansion.
    """

    tile_m, tile_n, _ = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    _, local_counts = _routing(kernel)
    populated_m_tiles = 0
    tile_runs_by_group = {}
    tail_padding_by_group = {}
    repeated_expert_boundary_groups = set()
    for count in local_counts:
        if count <= 0:
            continue
        expert_start = populated_m_tiles
        expert_m_tiles = math.ceil(count / tile_m)
        populated_m_tiles += expert_m_tiles
        first_group = expert_start // group_size_m
        last_group = (populated_m_tiles - 1) // group_size_m
        if first_group not in tile_runs_by_group:
            tile_runs_by_group[first_group] = []
        if first_group == last_group:
            tile_runs_by_group[first_group].append(expert_m_tiles)
        else:
            first_run = group_size_m - expert_start % group_size_m
            tile_runs_by_group[first_group].append(first_run)
            last_run = populated_m_tiles % group_size_m
            if last_run:
                if last_group not in tile_runs_by_group:
                    tile_runs_by_group[last_group] = []
                tile_runs_by_group[last_group].append(last_run)
        if expert_m_tiles > group_size_m + 1:
            repeated_expert_boundary_groups.update((
                expert_start // group_size_m,
                (populated_m_tiles - 1) // group_size_m,
            ))
        tail_rows = count % tile_m
        if tail_rows:
            group_index = (populated_m_tiles - 1) // group_size_m
            if group_index not in tail_padding_by_group:
                tail_padding_by_group[group_index] = 0
            tail_padding_by_group[group_index] += tile_m - tail_rows
    full_group_count = populated_m_tiles // group_size_m
    if full_group_count == 0:
        return {}
    group_count_by_topology = {1: 0}
    group_rows_by_topology = {1: 0}
    classified_topology_by_group = {}
    for group_index in range(full_group_count):
        tile_runs = tuple(
            tile_runs_by_group[group_index]
            if group_index in tile_runs_by_group else (group_size_m,))
        if sum(tile_runs) != group_size_m:
            raise AssertionError(
                "grouped FUSED_MOE full source group does not conserve M "
                "tiles")
        # A boundary can land exactly at the end of this GSM group.  The
        # compact tile-run sequence, unlike the old partial-boundary counter,
        # retains that source-order fact (e.g. the second run in 5+3).
        streams = len(tile_runs)
        if streams > group_size_m:
            raise ValueError(
                "grouped FUSED_MOE source group has too many weight streams")
        # The mixed-group source probes use a fixed fanout: every distinct B
        # stream owns the same number of M tiles in a complete GSM group.  A
        # stream count alone is not enough: e.g. 3+2+2+1 has four streams but
        # is not the measured 2+2+2+2 program.  Do not apply its service to
        # an uneven source topology.
        expected_run = group_size_m // streams
        is_uniform_exact_fanout = (
            streams in _GROUPED_OPERAND_STREAM_COUNTS
            and len(tile_runs) == streams
            and all(run == expected_run for run in tile_runs))
        topology = (
            streams if is_uniform_exact_fanout
            else (5, 3) if tile_runs == (5, 3) else None)
        if (topology is None
                or (streams > 1
                    and group_index in repeated_expert_boundary_groups)):
            continue
        classified_topology_by_group[group_index] = topology
        if topology not in group_count_by_topology:
            group_count_by_topology[topology] = 0
            group_rows_by_topology[topology] = 0
        group_count_by_topology[topology] += 1
        group_rows_by_topology[topology] += group_size_m * tile_m
    for group_index, padding_rows in tail_padding_by_group.items():
        if group_index not in classified_topology_by_group:
            continue
        topology = classified_topology_by_group[group_index]
        group_rows_by_topology[topology] -= padding_rows
    n_tiles = math.ceil(projection_n / tile_n)
    element_bytes = _gpu_operand_element_bytes(kernel)
    activation_replay, weight_replay = _gpu_operand_replay_factors(kernel)
    return {
        topology: float(
            n_tiles * projection_k * element_bytes * (
                group_rows * activation_replay
                + group_count_by_topology[topology]
                * group_size_m * tile_n * weight_replay))
        for topology, group_rows in group_rows_by_topology.items()
        if group_count_by_topology[topology] > 0
    }




def _gpu_grouped_operand_service_by_active_rows(kernel, hardware):
    """Return a complete exact masked-row service family when declared."""

    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    matching = tuple(
        service
        for service in hardware.memory_access_class_services
        if service.service == _GROUPED_OPERAND_SERVICE
        and service.access_kind == "read"
        and service.ingress_resource_role == "l2_cache"
        and service.signature.dtype == kernel.dtype
        and service.signature.tile_m == tile_m
        and service.signature.tile_n == tile_n
        and service.signature.tile_k == tile_k
        and service.signature.operand_role == "combined"
        and service.signature.group_size_m == int(raw_group_size))
    if not matching:
        raise ValueError(
            "grouped FUSED_MOE has no exact hardware signature family")
    by_rows = {}
    for active_rows in range(1, tile_m + 1):
        signature = _gpu_grouped_operand_signature(
            kernel, active_rows=active_rows)
        matches = tuple(
            service for service in matching
            if service.signature == signature)
        if len(matches) > 1:
            raise ValueError(
                "grouped FUSED_MOE repeats an exact masked-row service")
        if matches:
            by_rows[active_rows] = matches[0]
    if tile_m not in by_rows:
        raise ValueError(
            "grouped FUSED_MOE service family lacks its full-row signature")
    if len(by_rows) not in (1, tile_m):
        raise ValueError(
            "grouped FUSED_MOE masked-row service family must be complete")
    replay_fractions = {
        service.replay_to_dram_fraction for service in by_rows.values()}
    if len(replay_fractions) != 1:
        raise ValueError(
            "grouped FUSED_MOE masked-row services disagree on replay")
    return by_rows


def _cann_grouped_operand_signature(
        kernel: KernelSpec, service: str,
        operand_role: str) -> MemoryAccessClassSignature:
    """Return one exact full-tile AIC operand signature."""

    tile_m, tile_n, tile_k = _tile(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    if operand_role == "activation":
        logical_row_bytes = tile_m * tile_k * element_bytes
    elif operand_role == "weight":
        logical_row_bytes = tile_k * tile_n * element_bytes
    else:
        raise ValueError(
            "CANN GroupedMatmul operand role must be activation or weight")
    return MemoryAccessClassSignature(
        access_class=service,
        dtype=kernel.dtype,
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=logical_row_bytes,
        operand_role=operand_role,
        tile_n=tile_n,
        # This identifies one source descriptor service unit; it is not the
        # unused GPU launch ``group_size_m`` parameter.
        group_size_m=1,
    )


def _cann_grouped_operand_services(kernel: KernelSpec, hardware):
    """Return each complete source-invariant A/B capability pair.

    The A descriptor contains ``baseM`` and therefore remains bound to the
    captured tile-M.  The B descriptor is always ``baseK x baseN`` in the
    versioned CANN source and capture, so its measured service is reusable
    when the launch selector changes only ``baseM``.  Keep an unavailable A
    pair on the generic MTE2 byte-capacity path instead of disabling the B
    service and the complete analytical tile pipeline.
    """

    role, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    parameters = kernel.launch.parameters
    b_l2_hit_service = _CANN_OPERAND_B_L2_HIT_SERVICE
    b_dram_miss_service = _CANN_OPERAND_B_DRAM_MISS_SERVICE
    b_program_names = ("cann_depth_b1", "cann_step_n", "cann_step_kb")
    if all(name in parameters for name in b_program_names):
        b_program = tuple(parameters[name] for name in b_program_names)
        if (all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and float(value).is_integer()
                and value > 0
                for value in b_program)
                and hardware.has_memory_access_class_service(
                    _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE)
                and kernel.dtype == "bf16"
                and projection_k == 2880
                and (tile_n, tile_k) == (256, 64)
                and kernel.launch.grid_size == 24
                and int(parameters["cann_depth_b1"]) == (
                    2 * int(parameters["cann_step_n"])
                    * int(parameters["cann_step_kb"]))):
            b_l2_hit_service = (
                _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE)
    # CANN binds the B GlobalTensor before it selects the B L2 hint, and
    # binds that same 64x256 descriptor into Matmul after the hint.  Hence
    # the independent strided B ingress service applies to every DRAM miss
    # of these exact W2 programs; the source-order residency trace alone
    # decides whether a request is such a miss.  Unlike the K6144 service
    # below, this selection must not conflate the descriptor geometry with
    # the all-single-M condition that disables B caching.
    # The B descriptor is source-identical for every captured K2880 W2
    # program.  Select its service from that descriptor itself, rather than
    # maintaining a second base-M -> program table beside launch selection.
    # In particular M64 and M80 share this descriptor while their A streams
    # differ only in the amount of scheduled work.
    k2880_w2_b_descriptor_program = {
        "cann_depth_b1": 8,
        "cann_step_kb": 4,
        "cann_step_n": 1,
        "cann_db_l0b": 2,
    }
    _, local_counts = _routing(kernel)
    nonempty_local_rows = tuple(rows for rows in local_counts if rows > 0)
    if all(name in parameters for name in k2880_w2_b_descriptor_program):
        source_b_descriptor_matches = all(
            not isinstance(parameters[name], bool)
            and isinstance(parameters[name], (int, float))
            and float(parameters[name]).is_integer()
            and int(parameters[name]) == expected
            for name, expected in k2880_w2_b_descriptor_program.items())
        if (source_b_descriptor_matches
                and hardware.has_memory_access_class_service(
                    _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE)
                and kernel.dtype == "bf16"
                and role == "w2"
                and projection_k == projection_n == 2880
                and kernel.launch.grid_size == 24):
            b_dram_miss_service = (
                _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE)
    expected_source_program = cann_grouped_matmul_k_program(
        tile_m, int(kernel.shape["tokens"]) * int(kernel.shape["topk"]))
    source_k6144_m1_program_matches = (
        expected_source_program is not None
        and all(
            name in parameters
            for name in CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES)
        and all(
            not isinstance(parameters[name], bool)
            and isinstance(parameters[name], (int, float))
            and float(parameters[name]).is_integer()
            and int(parameters[name]) == expected
            for name, expected in zip(
                CANN_GROUPED_MATMUL_K_PROGRAM_PARAMETER_NAMES,
                expected_source_program)))
    if (source_k6144_m1_program_matches
            and hardware.has_memory_access_class_service(
                _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE)
            and kernel.dtype == "bf16"
            and role == "w1"
            and projection_k == 6144
            and kernel.launch.grid_size == 24
            and nonempty_local_rows
            and all(rows <= tile_m for rows in nonempty_local_rows)):
        b_dram_miss_service = (
            _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE)
    service_names = (
        _CANN_OPERAND_A_L2_HIT_SERVICE,
        _CANN_OPERAND_A_DRAM_MISS_SERVICE,
        b_l2_hit_service,
        b_dram_miss_service,
    )
    roles = ("activation", "activation", "weight", "weight")
    signatures = tuple(
        _cann_grouped_operand_signature(kernel, name, role)
        for name, role in zip(service_names, roles))
    matches = []
    for name, role, signature in zip(service_names, roles, signatures):
        values = []
        for candidate in hardware.memory_access_class_services:
            if (candidate.service != name
                    or candidate.access_kind != "read"
                    or candidate.ingress_resource_role != "l2_cache"):
                continue
            candidate_signature = candidate.signature
            if role == "weight":
                candidate_signature = dataclasses.replace(
                    candidate_signature, tile_m=signature.tile_m)
            if candidate_signature == signature:
                values.append(candidate)
        matches.append(tuple(values))
    matches = tuple(matches)
    if not any(matches):
        return None
    for operand_role, values in (
            ("activation", matches[:2]), ("weight", matches[2:])):
        if any(values) and any(len(item) != 1 for item in values):
            raise ValueError(
                "CANN GroupedMatmul operand residency service pair for "
                f"{operand_role} must be complete and unique")
    return {
        name: values[0]
        for name, values in zip(service_names, matches)
        if values
    }


def _apply_gpu_grouped_operand_downstream_replay(
        traffic: _GpuGroupedGemmTraffic,
        kernel: KernelSpec,
        hardware) -> _GpuGroupedGemmTraffic:
    """Map each operand's non-compulsory L2 replay to downstream reads."""

    service = _gpu_exact_grouped_operand_service(kernel, hardware)
    if service is None:
        return traffic
    if service.replay_to_dram_fraction is None:
        return traffic
    for role, l2_bytes, compulsory_bytes in (
            (
                "activation",
                traffic.activation_l2_read_bytes,
                traffic.activation_compulsory_read_bytes,
            ),
            (
                "weight",
                traffic.weight_l2_read_bytes,
                traffic.weight_compulsory_read_bytes,
            )):
        if compulsory_bytes < 0 or compulsory_bytes > l2_bytes:
            raise ValueError(
                f"grouped FUSED_MOE {role} compulsory bytes must be within "
                "its L2 request bytes")
    if (traffic.routed_weight_compulsory_read_bytes < 0
            or traffic.routed_weight_compulsory_read_bytes
            > traffic.routed_weight_l2_read_bytes):
        raise ValueError(
            "grouped FUSED_MOE routed-weight compulsory bytes must be within "
            "its L2 request bytes")
    if not math.isclose(
            traffic.routed_weight_dram_read_bytes,
            traffic.routed_weight_compulsory_read_bytes,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "grouped FUSED_MOE routed-weight DRAM bytes must retain only "
            "the unique source footprint")
    replay = service.replay_to_dram_fraction
    if replay is None:
        raise AssertionError("validated grouped operand replay is absent")
    return dataclasses.replace(
        traffic,
        activation_dram_read_bytes=(
            traffic.activation_compulsory_read_bytes
            + replay * (
                traffic.activation_l2_read_bytes
                - traffic.activation_compulsory_read_bytes)),
        weight_dram_read_bytes=(
            traffic.weight_compulsory_read_bytes
            + replay * (
                traffic.weight_l2_read_bytes
                - traffic.weight_compulsory_read_bytes)),
    )


def _gpu_grouped_weight_capacity_floor(
        kernel: KernelSpec, hardware, *, projection_k: int,
        projection_n: int, tile_m: int,
        local_counts: Tuple[int, ...], weight_l2_bytes: float,
        weight_compulsory_bytes: float) -> float:
    """Return the source-grouped lower bound on downstream weight reads.

    Triton's grouped-MoE program order keeps ``GROUP_SIZE_M`` consecutive M
    tiles in one source group.  An expert whose M tiles cross a group boundary
    can retain its weight matrix only while the distinct expert matrices in
    that group fit in L2.  When that footprint exceeds L2, the next group must
    stream the continuing expert again.  Count those group-boundary reloads
    from the routing histogram and matrix dimensions without expanding N/K
    CTAs or assuming a physical SM execution order.

    This floor is enabled only for the exact BM64 split hit/miss service.  Its
    independent probe establishes the two-slot source pipeline and strided
    weight-miss service; other GPU tile signatures retain the compulsory
    lower bound.
    """

    if (_gpu_split_operand_services(kernel, hardware) is None
            or int(kernel.attributes["expert_parallel_size"]) != 1):
        return weight_compulsory_bytes
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    group_size_m = int(raw_group_size)
    weight_matrix_bytes = (
        projection_k * projection_n * _gpu_operand_element_bytes(kernel))
    if weight_matrix_bytes <= 0 or hardware.l2_size_bytes <= 0:
        raise ValueError(
            "grouped FUSED_MOE weight residency requires positive capacity")

    spans = []
    populated_m_tiles = 0
    for count in local_counts:
        if count <= 0:
            continue
        expert_m_tiles = math.ceil(count / tile_m)
        first_group = populated_m_tiles // group_size_m
        last_group = (
            populated_m_tiles + expert_m_tiles - 1) // group_size_m
        spans.append((first_group, last_group))
        populated_m_tiles += expert_m_tiles
    if not spans:
        return weight_compulsory_bytes

    group_count = math.ceil(populated_m_tiles / group_size_m)
    group_expert_delta = [0] * (group_count + 1)
    for first_group, last_group in spans:
        group_expert_delta[first_group] += 1
        group_expert_delta[last_group + 1] -= 1
    oversize_group_prefix = [0]
    group_experts = 0
    for group_index in range(group_count):
        group_experts += group_expert_delta[group_index]
        if group_experts <= 0:
            raise ValueError(
                "grouped FUSED_MOE populated M group has no expert")
        oversize_group_prefix.append(
            oversize_group_prefix[-1]
            + int(group_experts * weight_matrix_bytes
                  > hardware.l2_size_bytes))

    weight_matrix_loads = len(spans)
    for first_group, last_group in spans:
        # The first group is already represented by the compulsory matrix.
        # Every later over-capacity group forces one source-ordered reload.
        weight_matrix_loads += (
            oversize_group_prefix[last_group + 1]
            - oversize_group_prefix[first_group + 1])
    capacity_floor = float(weight_matrix_loads * weight_matrix_bytes)
    if capacity_floor < weight_compulsory_bytes:
        raise AssertionError(
            "grouped FUSED_MOE capacity floor lost compulsory weights")
    return min(weight_l2_bytes, capacity_floor)


def _gpu_observed_resident_ctas(kernel: KernelSpec, hardware):
    """Physical CTA capacity from complete launch resources, not placement.

    The same register/shared/thread/warp allocation bounds feed cache replay
    and operand progress. Missing optional observations leave capacity unbound.
    """
    parameters = kernel.launch.parameters
    if "registers_per_thread" not in parameters:
        return None
    shared = parameters["dynamic_shared_memory_bytes"] + parameters["static_shared_bytes"]
    if shared > hardware.max_smem_per_block_bytes:
        raise ValueError("FUSED_MOE observed CTA exceeds block shared capacity")
    warps = math.ceil(kernel.launch.block_size / hardware.parallel_width)
    unit = hardware.register_allocation_unit_per_warp
    registers_per_warp = math.ceil(
        parameters["registers_per_thread"] * hardware.parallel_width / unit) * unit
    residents = min(hardware.max_blocks_per_unit,
        hardware.max_smem_per_unit_bytes // int(shared),
        hardware.max_regs_per_unit // (warps * registers_per_warp),
        hardware.max_threads_per_unit // kernel.launch.block_size,
        hardware.max_warps_per_unit // warps)
    if residents < 1:
        raise ValueError("FUSED_MOE observed CTA exceeds unit capacity")
    return residents


def _gpu_grouped_tile_program_traffic(
        kernel: KernelSpec, hardware, *, projection_k: int,
        projection_n: int, tile_m: int, tile_n: int, tile_k: int,
        n_tiles: int, m_tiles: int, local_counts: Tuple[int, ...],
        element_bytes: int, output_element_bytes: int, output_rows: int,
        routed_rows: int):
    """Replay the grouped-MoE weight stream through the shared miss core.

    The launch retains the group width its source swizzle uses, so the
    program-to-tile order is a launch fact and the routing histogram states
    which rows each M tile owns.  That is the whole access sequence; the
    replacement decision behind it is the same core the dense family uses.
    A launch that retains no group width keeps the previous envelope.

    The weight stream is always priced here.  The activation stream depends
    on the projection: W2 consumes the post-activation routed buffer, where
    every local route is its own row, so an M tile's rows are a distinct
    physical object and the replay is exact.  W1 consumes source tokens, and
    the routing histogram does not retain which routes came from one token,
    so two routes of a token would be counted as two objects; that projection
    keeps its source-unique bound instead.
    """

    group_size_m = int(kernel.launch.parameters["group_size_m"])
    if group_size_m <= 0 or not any(local_counts):
        return None
    if _gpu_uses_naive_assignment(kernel):
        # The histogram does not retain token-order expert IDs. It cannot
        # supply this unsorted source's per-route cache-reuse sequence.
        return None
    residents = _gpu_observed_resident_ctas(kernel, hardware)
    # The miss core's wave is explicitly a resident CTA window. Retain the
    # old one-CTA approximation only when launch resources do not close it.
    ctas_per_unit = 1 if residents is None else residents
    populated_m_tiles = sum(math.ceil(count / tile_m) for count in local_counts)
    traffic = simulate_moe_tile_program(
        m_tiles=m_tiles, n_tiles=n_tiles, group_size_m=group_size_m,
        k_extent=projection_k, block_k=tile_k,
        n_extent=projection_n, block_n=tile_n,
        element_bytes=element_bytes,
        c_row_bytes=tile_n * output_element_bytes, block_m=tile_m,
        wave_size=min(populated_m_tiles * n_tiles,
            int(hardware.num_compute_units) * ctas_per_unit),
        tokens_per_expert=local_counts,
        capacity_bytes=int(hardware.l2_replacement_capacity_bytes),
        residency_profile=hardware.l2_replacement_residency_profile)
    return traffic


def _gpu_traffic_components(
        kernel: KernelSpec, hardware) -> _GpuGroupedGemmTraffic:
    """Triton grouped-GEMM operand L2 and compulsory DRAM traffic.

    The launch grid covers the maximum routing workspace, while the kernel
    returns before loading operands for M tiles beyond
    ``num_tokens_post_padded``.  Within a populated tile, ``token_mask`` also
    suppresses activation loads and output stores for padding rows.  Traffic
    therefore follows semantic routed/source rows, not the max-EM launch
    extent or a fully populated BM tile.
    """

    if hardware.l2_size_bytes <= 0:
        raise ValueError("GPU FUSED_MOE requires positive L2 capacity")
    variant, source_rows, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    global_counts, local_counts = _routing(kernel)
    experts = int(kernel.shape["experts"])
    grid_size = kernel.launch.grid_size
    n_tiles = math.ceil(projection_n / tile_n)
    if grid_size % n_tiles:
        raise ValueError(
            "FUSED_MOE launch grid must be divisible by N-tile count")
    m_tiles = grid_size // n_tiles
    if m_tiles <= 0:
        raise ValueError("FUSED_MOE launch grid does not contain an M tile")
    configured_rows = int(kernel.launch.parameters["padded_rows"])
    configured_m_tiles = math.ceil(configured_rows / tile_m)
    if m_tiles != configured_m_tiles:
        raise ValueError(
            "FUSED_MOE launch grid disagrees with configured padded_rows")
    if tile_k <= 0:
        raise ValueError(f"FUSED_MOE tile_k must be positive, got {tile_k}")
    element_bytes = _gpu_operand_element_bytes(kernel)
    output_element_bytes = int(kernel.shape["dtype_bytes"])

    global_routed_rows = sum(global_counts)
    expected_routed_rows = (
        int(kernel.shape["tokens"]) * int(kernel.shape["topk"]))
    if global_routed_rows != expected_routed_rows:
        raise ValueError(
            "FUSED_MOE routing count sum disagrees with tokens * topk")
    global_populated_m_tiles = len(_gpu_valid_tile_rows(kernel, global_counts))
    if global_populated_m_tiles <= 0:
        raise ValueError(
            "FUSED_MOE routing must contain at least one populated M tile")
    if global_populated_m_tiles > m_tiles:
        raise ValueError(
            "FUSED_MOE populated M tiles exceed the configured launch grid")
    routed_rows = sum(local_counts)
    populated_m_tiles = len(_gpu_valid_tile_rows(kernel, local_counts))
    active_experts = sum(1 for count in local_counts if count > 0)

    # A is reloaded for each N tile, but token_mask admits only real routed
    # rows.  Every populated pid_m x pid_n program constructs and loads its
    # own B pointer stream.  The N coordinate is modulo N and has no N mask,
    # so a tail pid_n still issues the complete tile_n width (with wrapped
    # addresses).  C is masked in both M and N and therefore keeps only real
    # output columns.  With expert parallelism, pinned vLLM maps foreign
    # expert blocks to expert id -1 and explicitly writes zeros for their
    # valid routed rows before the cross-rank output reduction.  Each rank
    # therefore writes every global routed output row even though only its
    # locally owned rows load A/B and execute tensor operations.
    activation_l2 = (
        routed_rows * projection_k * n_tiles * element_bytes)
    expert_parallel_size = int(kernel.attributes["expert_parallel_size"])
    output_rows = (
        global_routed_rows if expert_parallel_size > 1 else routed_rows)
    output_l2 = output_rows * projection_n * output_element_bytes
    if not (tile_m <= 16 or tile_m >= 32):
        raise ValueError(f"unsupported fused-MoE M tile: {tile_m}")
    weight_l2 = (
        populated_m_tiles * projection_k * n_tiles * tile_n * element_bytes)

    # These are requests at the L2 ingress, so an L2 hit does not erase the A
    # request.  Independent NVIDIA NCU observations and MetaX layer-1 byte
    # counters both retain the complete valid-row A base stream and one C
    # stream; compiler-specific operand replay counts differ by launch profile.
    activation_replay, weight_replay = _gpu_operand_replay_factors(kernel)
    activation_l2 *= activation_replay
    weight_l2 *= weight_replay
    if variant == "w1":
        # The routing histogram does not retain source-token identities.  A
        # W1 input token contributes to at most ``topk`` local routes, so this
        # is the source-derived lower bound on unique rank-local input rows.
        local_source_rows = min(
            source_rows,
            math.ceil(routed_rows / int(kernel.shape["topk"])),
        )
    elif variant == "w2":
        # W2 consumes the post-activation routed buffer.  Every local route is
        # a distinct row even when several routes came from one source token.
        local_source_rows = routed_rows
    else:
        raise AssertionError(f"unexpected FUSED_MOE variant: {variant!r}")
    activation_compulsory = float(
        local_source_rows * projection_k * element_bytes)
    weight_compulsory = float(
        active_experts * projection_k * projection_n * element_bytes)
    routed_weight_l2, routed_weight_compulsory = (
        _source_routed_weight_read_traffic(
            kernel, routed_rows=float(routed_rows), n_tiles=n_tiles))
    output_compulsory = float(
        output_rows * projection_n * output_element_bytes)
    # Triton defines logical program groups but not physical AP/SM assignment.
    # Retain the uniquely addressed compulsory bytes generally.  For the
    # independently established BM64 two-slot service, additionally account
    # for source-group boundaries whose expert-weight footprint exceeds L2;
    # this remains a compact capacity floor, not a CTA-order cache replay.
    replay = _gpu_grouped_tile_program_traffic(
        kernel, hardware,
        projection_k=projection_k, projection_n=projection_n,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        n_tiles=n_tiles, m_tiles=m_tiles, local_counts=local_counts,
        element_bytes=element_bytes,
        output_element_bytes=output_element_bytes, output_rows=output_rows,
        routed_rows=routed_rows)
    activation_dram = activation_compulsory
    if replay is None:
        weight_dram = _gpu_grouped_weight_capacity_floor(
            kernel,
            hardware,
            projection_k=projection_k,
            projection_n=projection_n,
            tile_m=tile_m,
            local_counts=local_counts,
            weight_l2_bytes=float(weight_l2),
            weight_compulsory_bytes=weight_compulsory,
        )
    else:
        weight_dram = float(replay.b_dram_bytes)
        if variant == "w2":
            activation_dram = float(replay.a_dram_bytes)
    traffic = _GpuGroupedGemmTraffic(
        activation_l2_read_bytes=float(activation_l2),
        weight_l2_read_bytes=float(weight_l2),
        routed_weight_l2_read_bytes=routed_weight_l2,
        output_l2_write_bytes=float(output_l2),
        activation_compulsory_read_bytes=activation_compulsory,
        weight_compulsory_read_bytes=weight_compulsory,
        routed_weight_compulsory_read_bytes=routed_weight_compulsory,
        activation_dram_read_bytes=activation_dram,
        weight_dram_read_bytes=weight_dram,
        routed_weight_dram_read_bytes=routed_weight_compulsory,
        output_dram_write_bytes=output_compulsory,
    )
    traffic = _apply_gpu_grouped_operand_downstream_replay(
        traffic, kernel, hardware)
    validated_l2_miss_rate(
        traffic.dram_total_bytes, traffic.l2_total_bytes)
    return traffic


def _gpu_traffic(kernel: KernelSpec, hardware) -> Tuple[float, float, float]:
    """Return the legacy aggregate view of GPU grouped-GEMM traffic."""

    traffic = _gpu_traffic_components(kernel, hardware)
    return (
        traffic.l2_total_bytes,
        traffic.dram_total_bytes,
        traffic.compulsory_bytes,
    )


def _gpu_grouped_gemm_pipeline_envelope(
        kernel: KernelSpec, hardware,
        traffic: _GpuGroupedGemmTraffic
        ) -> Optional[GpuGroupedGemmPipelineEnvelopeDemand]:
    """Return the source-bound BM16 resident-wave K pipeline, when available."""

    mechanism = hardware.grouped_gemm_tile_pipeline
    if mechanism is None:
        return None
    tile_m, tile_n, tile_k = _tile(kernel)
    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    signature = (
        kernel.dtype,
        tile_m,
        tile_n,
        tile_k,
        int(raw_group_size),
        kernel.launch.block_size,
    )
    expected = (
        mechanism["dtype"],
        mechanism["tile_m"],
        mechanism["tile_n"],
        mechanism["tile_k"],
        mechanism["group_size_m"],
        mechanism["threads_per_cta"],
    )
    if signature != expected:
        return None
    variant, _, projection_k, projection_n = _projection(kernel)
    # A tile/thread match does not identify the compiled operand ring. An
    # observed launch may use a different stage count or register footprint
    # than the source program measured by this envelope.
    parameters = kernel.launch.parameters
    if ("dynamic_shared_memory_bytes" in parameters
            and parameters["dynamic_shared_memory_bytes"]
            != mechanism["dynamic_shared_memory_bytes_per_cta"]):
        return None
    if ("registers_per_thread" in parameters
            and parameters["registers_per_thread"]
            != mechanism["registers_per_thread_by_variant"][variant]):
        return None
    if (kernel.launch.tiling.pipeline_depth is not None
            and kernel.launch.tiling.pipeline_depth != mechanism["pipeline_depth"]):
        return None
    resident_ctas_by_variant = mechanism[
        "resident_ctas_per_sm_by_variant"]
    if set(resident_ctas_by_variant) != {"w1", "w2"}:
        raise ValueError(
            "GPU grouped-GEMM pipeline must declare W1/W2 residency")
    resident_ctas_per_sm = resident_ctas_by_variant[variant]
    if (isinstance(resident_ctas_per_sm, bool)
            or not isinstance(resident_ctas_per_sm, int)
            or resident_ctas_per_sm <= 0):
        raise ValueError(
            "GPU grouped-GEMM variant residency must be a positive integer")
    # Triton masks the final K iteration.  The aggregate traffic above uses
    # its logical, valid-K bytes, but this fixed-round pipeline envelope was
    # only independently established for complete K tiles.  Do not reject a
    # valid source launch or pretend that its partial round has the measured
    # complete-tile cadence.
    if projection_k % tile_k:
        return None
    _, local_counts = _routing(kernel)
    populated_m_tiles = len(_gpu_valid_tile_rows(kernel, local_counts))
    active_ctas = populated_m_tiles * math.ceil(projection_n / tile_n)
    if active_ctas == 0:
        if traffic.dram_read_bytes != 0.0:
            raise ValueError(
                "GPU grouped-GEMM has DRAM reads without an active CTA")
        return None
    return GpuGroupedGemmPipelineEnvelopeDemand(
        active_ctas=active_ctas,
        k_tiles=math.ceil(projection_k / tile_k),
        resident_ctas_per_sm=resident_ctas_per_sm,
        dram_read_bytes=traffic.dram_read_bytes,
    )


def _grouped_task_memory_bytes(kernel):
    """Partition source operand/output bytes in grouped program order.

    The routing census supplies each M tile's valid rows. The Triton group
    permutation selects M before N inside each group; empty reserved programs
    perform no matrix work. This is logical task order, not observed placement.
    """
    tm, tn, _ = _tile(kernel)
    _, _, k, n = _projection(kernel)
    _, counts = _routing(kernel)
    rows = _gpu_valid_tile_rows(kernel, counts)
    group_size = int(kernel.launch.parameters["group_size_m"])
    reserved_m = math.ceil(int(kernel.launch.parameters["padded_rows"]) / tm)
    element_bytes = _gpu_operand_element_bytes(kernel)
    output_bytes = int(kernel.shape["dtype_bytes"])
    a_replay, b_replay = _gpu_operand_replay_factors(kernel)
    result = []
    for first in range(0, len(rows), group_size):
        width = min(reserved_m - first, group_size)
        for column in range(0, n, tn):
            for row in rows[first:min(first + width, len(rows))]:
                result.append(((row * a_replay + tn * b_replay) * k * element_bytes,
                               row * min(tn, n - column) * output_bytes))
    return tuple(result)


def _compulsory_weight_request_topology(counts, *, tile_m, tile_n,
        projection_k, projection_n, element_bytes, group_size_m, wave_width):
    """Count first-touch requests sharing one cold weight stream per wave.

    Later waves are free in this optimistic progress work, including any
    cache eviction/reload. This changes neither cache replay nor DRAM bytes.
    The grouped source order and uniform populated wave remain analytical.
    """
    experts = tuple(e for e,count in enumerate(counts)
                    for _ in range(math.ceil(count/tile_m)))
    programs = tuple((expert, column)
        for first in range(0,len(experts),group_size_m)
        for column in range(0,projection_n,tile_n)
        for expert in experts[first:first+group_size_m])
    seen = set()
    requesting_bytes = unique_bytes = 0
    maximum_fanout = 0
    for first in range(0,len(programs),wave_width):
        pending = {}
        for key in programs[first:first+wave_width]:
            if key not in seen:
                if key not in pending:
                    pending[key] = 0
                pending[key] += 1
        for key,count in pending.items():
            _,column = key
            payload = min(tile_n,projection_n-column)*projection_k*element_bytes
            requesting_bytes += count*payload
            unique_bytes += payload
            maximum_fanout = max(maximum_fanout,count)
        seen.update(pending)
    return requesting_bytes, unique_bytes, maximum_fanout


def _attach_resident_operand_pipeline(inp, kernel, hardware):
    """Use source-known stages in a capacity-conserving resident CTA schedule."""
    depth = kernel.launch.tiling.pipeline_depth
    if (depth is None or inp.pipeline_plan is not None
            or kernel.attributes["expert_parallel_size"] != 1
            or hardware.gpu_store_completion_scope != "l2"
            or "dynamic_shared_memory_bytes" not in kernel.launch.parameters):
        return inp
    shared = kernel.launch.parameters["dynamic_shared_memory_bytes"]
    tm, tn, tk = _tile(kernel)
    panel = (tm + tn) * tk * _gpu_operand_element_bytes(kernel)
    if shared != depth * panel or shared > hardware.max_smem_per_block_bytes:
        return inp
    residents = _gpu_observed_resident_ctas(kernel, hardware)
    if residents is None:
        if shared > hardware.max_smem_per_unit_bytes // 2:
            # Older callers carry no register observation; only a one-CTA
            # shared-memory proof can admit their operand pipeline.
            residents = 1
        else:
            return inp
    if {d.resource_role for d in inp.compute_demands} != {"compute_tensor"}:
        return inp
    reads = tuple(path for path in inp.memory_paths if path.access_kind == "read")
    writes = tuple(path for path in inp.memory_paths if path.access_kind == "write")
    if len(writes) != 1 or writes[0].access_class_demands:
        return inp
    if len(reads) == 1 and reads[0].name == "operand_read":
        # The gather binder already conserves and joins A/B requests, with
        # their existing typed services. Reuse that exact ingress rather than
        # discarding its evidence or requiring a second untyped merge.
        merged = reads[0]
    elif len(reads) == 2 and not any(path.access_class_demands for path in reads):
        merged = merge_operand_read_paths(reads)
    else:
        return inp
    _, _, k, n = _projection(kernel)
    counts, _ = _routing(kernel)
    tasks = len(_gpu_valid_tile_rows(kernel, counts)) * math.ceil(n / tn)
    inp = dataclasses.replace(inp, memory_paths=(merged, *writes))
    if not merged.access_class_demands:
        inp = attach_async_copy_ring_capacity(inp, hardware,
            read_path_name=merged.name, dtype=kernel.dtype, tile_m=tm, tile_k=tk,
            panel_bytes=panel, buffer_slots=depth, resident_ctas_per_unit=residents,
            k_tiles=math.ceil(k / tk))
    output_relayout_bytes = 0.0
    output_conversion_warp_instructions = 0
    if "registers_per_thread" in kernel.launch.parameters:
        per_task_bytes = observed_fused_moe_output_relayout_bytes(kernel, hardware)
        if per_task_bytes is not None:
            output_relayout_bytes = tasks * per_task_bytes
            output_bytes = next(node.bytes for node in writes[0].nodes
                                if node.resource_role == "l2_cache")
            demand = dict(inp.demand_per_resource)
            # build_forward_demand already counts two SMEM transfers per
            # global byte. Keep its coarse operand proxy, replacing only the
            # valid-output contribution by the source's full padded tile.
            demand["smem"] += output_relayout_bytes - 2 * output_bytes
            inp = dataclasses.replace(inp, demand_per_resource=demand,
                limitations=inp.limitations + (
                    "moe_output_full_tile_shared_relayout_before_global_store",
                    "moe_output_conversion_execution_and_shared_bank_barrier_services_unpriced",))
    startup_issue = (None if merged.access_class_demands
        else observed_fused_moe_startup_entry_issue(kernel, hardware,
            assignment_mode="naive" if _gpu_uses_naive_assignment(kernel) else "aligned"))
    source_shared_work = None
    if depth == 1 and output_relayout_bytes == 0:
        shared_work = observed_fused_moe_operand_shared_work(kernel, hardware,
            assignment_mode="naive" if _gpu_uses_naive_assignment(kernel) else "aligned")
        if shared_work is not None:
            read_per_k, write_per_k, prologue_write, output_read, output_write = shared_work
            k_tiles = math.ceil(k / tk)
            source_shared_work = (tasks * (write_per_k * k_tiles + prologue_write),
                                  tasks * read_per_k * k_tiles,
                                  tasks * (output_read + output_write))
            demand = dict(inp.demand_per_resource)
            demand["smem"] = math.fsum(source_shared_work)
            inp = dataclasses.replace(inp, demand_per_resource=demand,
                limitations=inp.limitations + (
                    "moe_single_panel_shared_work_from_compiled_LDSM_LDGSTS_STSM_LDS",
                    "moe_single_panel_output_shared_read_is_full_tile_upper_bound",))
    per_task_conversion = source_fused_moe_output_conversion_warp_instructions(
        kernel, hardware)
    if per_task_conversion is not None:
        output_conversion_warp_instructions = tasks * per_task_conversion
    if depth == 1 and output_relayout_bytes == 0:
        coarse = attach_single_panel_matmul_pipeline(inp, hardware,
            task_count=tasks, dtype=kernel.dtype, tile_m=tm, tile_k=tk,
            panel_bytes=tn * tk * _gpu_operand_element_bytes(kernel),
            resident_ctas_per_unit=residents, row_stride_bytes=k * _gpu_operand_element_bytes(kernel),
            read_path_name=merged.name, write_path_name=writes[0].name,
            startup_entry_warp_instructions=startup_issue,
            source_shared_work=source_shared_work,
            output_conversion_warp_instructions=output_conversion_warp_instructions)
        if coarse is not inp:
            return coarse
        if source_shared_work is not None:
            inp = dataclasses.replace(inp, limitations=inp.limitations + (
                "single_panel_compiled_shared_work_is_aggregate_without_cohort_admission",))
    inp = attach_recurrent_matmul_pipeline(inp, hardware,
        task_count=tasks, k_tiles=math.ceil(k / tk), buffer_slots=depth,
        read_path_name=merged.name, write_path_name=writes[0].name,
        resident_ctas_per_unit=residents, output_relayout_bytes=output_relayout_bytes,
        output_conversion_warp_instructions=output_conversion_warp_instructions,
        task_memory_bytes=_grouped_task_memory_bytes(kernel),
        startup_entry_warp_instructions=startup_issue,
        matrix_preparation_source_work=(
            None if merged.access_class_demands else
            observed_fused_moe_matrix_preparation_issue(kernel, hardware,
                assignment_mode="naive" if _gpu_uses_naive_assignment(kernel) else "aligned")))
    return dataclasses.replace(inp, limitations=inp.limitations + (
        ("moe_startup_entry_issue_is_conservative_static_lower_count"
         if startup_issue is not None else "moe_startup_entry_issue_unbound"),
        "moe_entry_startup_bulk_cta_stage_is_analytical",
        "moe_uniform_cta_worker_assignment_is_analytical",
        "moe_resident_cta_lanes_share_unit_capacity_equally",
        "moe_source_ordered_valid_rows_partition_operand_and_output_bytes",
        "moe_task_downstream_cache_ratios_remain_aggregate_approximations",
        "moe_store_retirement_at_l2_with_aggregate_dram_drain",
    ))


def _metax_grouped_operand_pipeline_envelope(
        kernel: KernelSpec, hardware,
        traffic: _GpuGroupedGemmTraffic
        ) -> Optional[MetaxGroupedOperandPipelineDemand]:
    """Return the C500 per-CTA serial grouped-operand-pipeline demand.

    One resident CTA per AP drains its expert's operand stream round by
    round; a synthetic-routing mechanism probe over non-production K/N
    geometry separates the fixed per-CTA span from the per-round service for
    one exact compiled tile signature, and the DRAM/latency round service is
    priced from independent hardware facts.  Each deployed tile signature
    needs its own measurement: BM16/BN64/BK128 and BM32/BN64/BK128 have
    different compiled resources, so a launch whose signature has no
    measured mechanism keeps the generic memory model rather than borrowing
    another signature's coefficients.  No production duration or case
    identity participates in any mechanism.
    """

    raw_group_size = kernel.launch.parameters["group_size_m"]
    if (isinstance(raw_group_size, bool)
            or not isinstance(raw_group_size, (int, float))
            or not float(raw_group_size).is_integer()
            or raw_group_size <= 0):
        raise ValueError(
            "grouped FUSED_MOE group_size_m must be a positive integer")
    tile_m, tile_n, tile_k = _tile(kernel)
    launch_parameters = kernel.launch.parameters
    # Tile shape and block threads do not identify the compiled program on
    # their own: BM64/BN128/BK64 compiles to a four-warp and an eight-warp
    # variant with the same tile, threads and dynamic shared memory.  A launch
    # that does not record its compiled-program facts cannot be matched to a
    # measured mechanism at all.
    if not {"num_warps", "num_stages"}.issubset(launch_parameters):
        return None
    signature = (
        kernel.dtype,
        tile_m,
        tile_n,
        tile_k,
        int(raw_group_size),
        kernel.launch.block_size,
        int(launch_parameters["num_warps"]),
        int(launch_parameters["num_stages"]),
    )
    mechanism = None
    mechanism_key = None
    for candidate in METAX_GROUPED_OPERAND_PIPELINE_MECHANISM_KEYS:
        candidate_mechanism = getattr(hardware, candidate)
        if candidate_mechanism is None:
            continue
        expected = (
            candidate_mechanism["dtype"],
            candidate_mechanism["tile_m"],
            candidate_mechanism["tile_n"],
            candidate_mechanism["tile_k"],
            candidate_mechanism["group_size_m"],
            candidate_mechanism["threads_per_cta"],
            candidate_mechanism["num_warps"],
            candidate_mechanism["num_stages"],
        )
        if signature == expected:
            mechanism = candidate_mechanism
            mechanism_key = candidate
            break
    if mechanism_key is None:
        return None
    variant, _, projection_k, projection_n = _projection(kernel)
    # The per-round service was only independently established for complete
    # K tiles; do not pretend a masked partial round has the measured
    # complete-tile cadence.
    if projection_k % tile_k:
        return None
    _, local_counts = _routing(kernel)
    # Each mechanism declares how many M tiles one populated expert may span
    # inside its measured domain.  Beyond that bound several M tiles share one
    # expert weight stream, which the sweep did not establish; keep the
    # generic memory model instead of extrapolating the measured cadence.
    maximum_expert_m_tiles = mechanism["maximum_expert_m_tiles"]
    if maximum_expert_m_tiles is not None:
        for rows in local_counts:
            if rows > 0 and math.ceil(rows / tile_m) > maximum_expert_m_tiles:
                return None
    populated_m_tiles = sum(
        math.ceil(rows / tile_m) for rows in local_counts if rows > 0)
    active_ctas = populated_m_tiles * math.ceil(projection_n / tile_n)
    if active_ctas == 0:
        if traffic.dram_read_bytes != 0.0:
            raise ValueError(
                "MetaX grouped-GEMM has DRAM reads without an active CTA")
        return None
    return MetaxGroupedOperandPipelineDemand(
        active_ctas=active_ctas,
        k_tiles=projection_k // tile_k,
        dram_read_bytes=traffic.dram_read_bytes,
        mechanism_key=mechanism_key,
    )


def _cann_grouped_matmul_traffic(
        kernel: KernelSpec,
        work: _CannGroupedMatmulWork,
        residency: Optional[_CannGroupedOperandResidency] = None
        ) -> _CannGroupedMatmulTraffic:
    """Return the CANN 8.5.1 AIC ingress and output byte contract.

    The version-bound launch-tile census predicts one activation slab per
    output-N tile and one N-tail-clipped weight slab per populated expert M
    tile.  Canonical non-duration counters validate that relation against
    GM-to-L1; the logical C output is separately validated against the AIC
    write interface.  Neither interface proves per-invocation DRAM residency,
    descriptor timing, or a FixPipe lifecycle.  No target duration or
    per-shape traffic table is consumed here.
    """

    _, _, projection_k, projection_n = _projection(kernel)
    _, local_counts = _routing(kernel)
    active_experts = sum(1 for count in local_counts if count > 0)
    element_bytes = int(kernel.shape["dtype_bytes"])
    per_expert_windows = kernel.attributes["expert_weight_tile_rows"]
    selected_window = float(kernel.launch.parameters["weight_tile_rows"])
    if per_expert_windows is not None or selected_window != 0.0:
        raise ValueError(
            "NPU FUSED_MOE cannot consume traffic-derived residency windows "
            "without source-backed descriptor semantics")
    if work.routed_rows != sum(local_counts):
        raise ValueError(
            "CANN GroupedMatmul traffic and Tensor work disagree on routed rows")
    if work.populated_m_tiles < active_experts or work.n_tiles <= 0:
        raise ValueError("CANN GroupedMatmul produced an invalid tile census")

    activation_compulsory = (
        work.routed_rows * projection_k * element_bytes)
    output = work.routed_rows * projection_n * element_bytes
    l2_read = (
        activation_compulsory * work.n_tiles
        + work.populated_m_tiles * projection_k * projection_n
        * element_bytes)
    if residency is None:
        # CANN publishes the tensor ABI and the MTE2 ingress census but not a
        # cross-AIC cache replacement schedule.  A tile-by-tile LRU replay
        # would make that unobserved schedule part of Forward.  Use the
        # semantic unique operand footprint as the downstream analytical
        # lower bound, retaining the complete source-visible L2 ingress
        # stream above.  This is the same resource-envelope treatment used by
        # the GPU grouped projection when an execution-order contract is not
        # available.
        dram_read = float(
            activation_compulsory
            + active_experts * projection_k * projection_n * element_bytes)
    else:
        if residency.l2_read_bytes != l2_read:
            raise ValueError(
                "CANN GroupedMatmul LRU and analytical L2 traffic disagree: "
                f"lru={residency.l2_read_bytes}, analytical={l2_read}")
        dram_read = float(residency.dram_read_bytes)
    traffic = _CannGroupedMatmulTraffic(
        activation_l2_read_bytes=float(
            activation_compulsory * work.n_tiles),
        weight_l2_read_bytes=float(
            work.populated_m_tiles * projection_k * projection_n
            * element_bytes),
        output_l2_write_bytes=float(output),
        dram_read_bytes=dram_read,
        dram_write_bytes=float(output),
    )
    return traffic





def _cann_grouped_operand_service_paths_from_paths(
        memory_paths: Tuple[MemoryPathDemand, ...], kernel: KernelSpec,
        residency: _CannGroupedOperandResidency,
        hardware) -> Tuple[MemoryPathDemand, ...]:
    """Annotate one completed CANN path tuple with operand services."""

    if not isinstance(memory_paths, tuple):
        raise TypeError("CANN grouped operand memory paths must be a tuple")
    services = _cann_grouped_operand_services(kernel, hardware)
    if services is None:
        return memory_paths
    read_path, = tuple(
        path for path in memory_paths if path.name == "mte2_read")
    if read_path.access_class_demands:
        raise ValueError("CANN grouped operand path already has a typed service")
    b_l2_hit_service = (
        _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE
        if _CANN_OPERAND_B_K2880_STRIDE_L2_HIT_SERVICE in services
        else _CANN_OPERAND_B_L2_HIT_SERVICE)
    b_dram_miss_service = next(
        service_name for service_name in (
            _CANN_OPERAND_B_K2880_STRIDED_DRAM_MISS_SERVICE,
            _CANN_OPERAND_B_K6144_M1_DRAM_MISS_SERVICE,
            _CANN_OPERAND_B_DRAM_MISS_SERVICE)
        if service_name in services)
    service_counts = {
        _CANN_OPERAND_A_L2_HIT_SERVICE:
            sum(residency.a_hits_by_worker),
        _CANN_OPERAND_A_DRAM_MISS_SERVICE:
            sum(residency.a_misses_by_worker),
        b_l2_hit_service:
            sum(residency.b_hits_by_worker),
        b_dram_miss_service:
            sum(residency.b_misses_by_worker),
    }
    demands = tuple(
        MemoryAccessClassDemand(
            service=service_name,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=float(
                service_counts[service_name]
                * service.signature.logical_row_bytes),
            signature=service.signature,
        )
        for service_name, service in services.items()
        if service_counts[service_name] > 0
    )
    if not demands:
        return memory_paths
    l2_read_bytes = next(
        node.bytes for node in read_path.nodes
        if node.resource_role == "l2_cache")
    if math.fsum(demand.logical_bytes for demand in demands) > l2_read_bytes:
        raise AssertionError(
            "CANN grouped operand full descriptors exceed L2 ingress bytes")
    typed_read_path = dataclasses.replace(
        read_path,
        access_class_demands=demands,
    )
    paths = tuple(
        typed_read_path if path is read_path else path
        for path in memory_paths)
    return paths




def _cann_tile_class_census(residency, work):
    """Census the populated (M, N) tile classes across the AIC workers.

    Returns the sorted class keys, their index map, and each worker's count
    vector.  The two envelope callers name the returned keys differently
    (`class_keys` / `class_shapes`) but read the same triple; a residency plan
    with no populated class, or one whose census loses a worker, is a contract
    violation rather than an empty result.
    """

    class_keys = tuple(sorted({
        (logical_m, logical_n)
        for worker_classes in residency.tile_classes_by_worker
        for logical_m, logical_n, _ in worker_classes
    }))
    if not class_keys:
        raise ValueError("CANN GroupedMatmul has no populated tile classes")
    class_index = {key: index for index, key in enumerate(class_keys)}
    counts_by_worker = []
    for worker_classes in residency.tile_classes_by_worker:
        counts = [0] * len(class_keys)
        for logical_m, logical_n, count in worker_classes:
            counts[class_index[(logical_m, logical_n)]] = count
        counts_by_worker.append(tuple(counts))
    if len(counts_by_worker) != work.worker_count:
        raise AssertionError("CANN tile class census lost an AIC worker")
    return class_keys, class_index, counts_by_worker




def _cann_grouped_matmul_envelope_demand(
        kernel: KernelSpec, work: _CannGroupedMatmulWork,
        residency: _CannGroupedOperandResidency,
        hardware) -> Optional[CannGroupedMatmulEnvelopeDemand]:
    """Return the analytical source histogram used by Forward on 910B."""

    program_names = (
        "cann_depth_a1", "cann_depth_b1",
        "cann_step_ka", "cann_step_kb",
        "cann_step_m", "cann_step_n",
        "cann_db_l0a", "cann_db_l0b", "cann_db_l0c",
    )
    present = tuple(name in kernel.launch.parameters for name in program_names)
    if any(present) and not all(present):
        raise ValueError(
            "CANN GroupedMatmul launch has an incomplete K program")
    if not any(present):
        return None
    program = {}
    for name in program_names:
        value = kernel.launch.parameters[name]
        if not float(value).is_integer() or value <= 0:
            raise ValueError(
                f"CANN GroupedMatmul launch {name} must be a positive integer")
        program[name] = int(value)
    if program["cann_step_m"] != 1 or program["cann_step_n"] != 1:
        raise ValueError(
            "CANN GroupedMatmul analytical envelope requires unit M/N steps")
    if (program["cann_depth_a1"]
            % (program["cann_step_m"] * program["cann_step_ka"])
            or program["cann_depth_b1"]
            % (program["cann_step_n"] * program["cann_step_kb"])):
        raise ValueError(
            "CANN GroupedMatmul L1 depth must contain complete K groups")
    if program["cann_db_l0c"] != 1:
        raise ValueError(
            "CANN GroupedMatmul analytical envelope requires captured L0C")
    services = _cann_grouped_operand_services(kernel, hardware)
    if services is None:
        return None
    class_shapes, class_index, counts_by_worker = _cann_tile_class_census(
        residency, work)
    a_hit_fraction, b_hit_fraction = (
        _cann_grouped_operand_hit_fractions(kernel, residency))
    role, _, projection_k, projection_n = _projection(kernel)
    source_projection_n = (
        projection_n // 2 if role == "w1" else projection_n)
    _, local_counts = _routing(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)

    # ``GMMProcess`` issues each expert's MN tiles in its source diagonal
    # order and advances the AIC ring across expert boundaries.  Preserve the
    # order as compact runs, rather than materialising one Pipeline task per
    # CTA.  The controller recurrence consumes this exact static program;
    # counts_by_worker remains the independent conservation check above.
    source_runs_by_worker = [[] for _ in range(work.worker_count)]
    ring_offset = 0
    # The source program walks the full W1 output width.  The envelope's
    # source_projection_n retains the pre-existing per-projection service
    # field, while its histogram (and therefore this controller program)
    # covers both W1 halves exactly once.
    n_tiles = math.ceil(projection_n / tile_n)
    for rows in local_counts:
        if rows == 0:
            continue
        m_tiles = math.ceil(rows / tile_m)
        expert_tile_count = m_tiles * n_tiles
        for relative_block, (m_index, n_index) in enumerate(
                _cann_mn_block_order(m_tiles, n_tiles)):
            logical_m = min(tile_m, rows - m_index * tile_m)
            logical_n = min(tile_n, projection_n - n_index * tile_n)
            source_class_index = class_index[(logical_m, logical_n)]
            worker = (ring_offset + relative_block) % work.worker_count
            worker_runs = source_runs_by_worker[worker]
            if (worker_runs
                    and worker_runs[-1][0] == source_class_index):
                worker_runs[-1] = (
                    source_class_index, worker_runs[-1][1] + 1)
            else:
                worker_runs.append((source_class_index, 1))
        ring_offset = (ring_offset + expert_tile_count) % work.worker_count
    return CannGroupedMatmulEnvelopeDemand(
        worker_count=work.worker_count,
        zero_expert_count=sum(rows == 0 for rows in local_counts),
        class_shapes=class_shapes,
        counts_by_worker=tuple(counts_by_worker),
        source_class_runs_by_worker=tuple(
            tuple(runs) for runs in source_runs_by_worker),
        projection_k=projection_k,
        projection_n=source_projection_n,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_step_k_tiles=program["cann_step_ka"],
        b_step_k_tiles=program["cann_step_kb"],
        a1_buffer_slots=(
            program["cann_depth_a1"]
            // (program["cann_step_m"] * program["cann_step_ka"])),
        b1_buffer_slots=(
            program["cann_depth_b1"]
            // (program["cann_step_n"] * program["cann_step_kb"])),
        l0a_buffer_slots=program["cann_db_l0a"],
        l0b_buffer_slots=program["cann_db_l0b"],
        l0c_buffer_slots=program["cann_db_l0c"],
        dtype=kernel.dtype,
        element_bytes=int(kernel.shape["dtype_bytes"]),
        tensor_operations=float(work.tensor_operations),
        a_l2_hit_fraction=a_hit_fraction,
        b_l2_hit_fraction=b_hit_fraction,
    )




def _attach_prepared_expert_operand_pipeline(inp, kernel, hardware):
    """Coarse local-CTA progress with all original EP output traffic retained."""
    if (hardware.prepared_moe_operand_resources is None or kernel.dtype != "bf16"
            or kernel.launch.algorithm not in (
                "vllm_0_30_sm90_unquantized_aligned_w1", "vllm_0_30_sm90_unquantized_aligned_w2")
            or kernel.attributes["expert_parallel_size"] != 2):
        return inp
    parameters = kernel.launch.parameters
    required = ("registers_per_thread", "static_shared_bytes", "dynamic_shared_memory_bytes")
    if any(name not in parameters for name in required):
        return dataclasses.replace(inp, limitations=inp.limitations + (
            "moe_prepared_resident_operand_resource_class_unbound",))
    tm, tn, tk = _tile(kernel)
    depth = kernel.launch.tiling.pipeline_depth
    shared = parameters["dynamic_shared_memory_bytes"]
    if depth is None or shared != depth * (tm + tn) * tk * _gpu_operand_element_bytes(kernel):
        raise ValueError("prepared expert physical ring does not conserve shared memory")
    shared += parameters["static_shared_bytes"]
    if not 0 < shared <= hardware.max_smem_per_block_bytes:
        raise ValueError("prepared expert CTA exceeds shared-memory capacity")
    warps = math.ceil(kernel.launch.block_size / hardware.parallel_width)
    unit = hardware.register_allocation_unit_per_warp
    registers_per_warp = math.ceil(parameters["registers_per_thread"] * hardware.parallel_width / unit) * unit
    residents = min(hardware.max_blocks_per_unit, hardware.max_smem_per_unit_bytes // int(shared),
        hardware.max_regs_per_unit // (warps * registers_per_warp),
        hardware.max_threads_per_unit // kernel.launch.block_size, hardware.max_warps_per_unit // warps)
    if residents < 1:
        raise ValueError("prepared expert CTA has no architectural residency")
    _, local_counts = _routing(kernel)
    _, _, k, n = _projection(kernel)
    tasks = sum(math.ceil(count / tm) for count in local_counts) * math.ceil(n / tn)
    if not tasks:
        return dataclasses.replace(inp, limitations=inp.limitations + (
            "moe_prepared_no_local_matrix_tasks_retains_aggregate_zero_output",))
    if inp.pipeline_plan is not None or {d.resource_role for d in inp.compute_demands} != {"compute_tensor"}:
        raise NotImplementedError("prepared expert recurrent progress requires one unassigned Tensor stage")
    reads = tuple(path for path in inp.memory_paths if path.access_kind == "read")
    writes = tuple(path for path in inp.memory_paths if path.access_kind == "write")
    if len(reads) != 2 or len(writes) != 1 or writes[0].access_class_demands:
        raise NotImplementedError("prepared expert recurrent progress requires A/B ingress and one untyped output")
    merged = merge_operand_read_paths(reads)
    structured = dataclasses.replace(inp, memory_paths=(merged, *writes))
    structured = attach_recurrent_matmul_pipeline(structured, hardware,
        task_count=tasks, k_tiles=math.ceil(k / tk), buffer_slots=depth,
        read_path_name=merged.name, write_path_name=writes[0].name,
        resident_ctas_per_unit=residents)
    return dataclasses.replace(structured, limitations=structured.limitations + (
        "moe_prepared_operand_resource_class_modeled_not_native_attestation",
        "moe_prepared_uniform_local_CTA_cohort_and_equal_unit_capacity_share",
        "moe_prepared_global_nonlocal_zero_output_bytes_amortized_over_local_matrix_tasks",
        "moe_prepared_inactive_CTA_placement_output_conversion_and_control_unbound"))


def _prepared_ep_reference_prediction(kernel, hardware):
    """Price the declared mathematical aggregate through a scoped source recipe.

    Prepared probabilities/IDs already exist. The wrapper remains one
    analytical call; its source stages use the shared public predictors.
    This does not certify a native provider or synthesize routing state.
    """
    if (hardware.prepared_moe_operand_resources is None or kernel.dtype != "bf16"
            or kernel.attributes["mathematical_program"] != "unquantized_bf16_grouped_swiglu_v1"
            or kernel.attributes["expert_parallel_size"] != 2
            or kernel.attributes["routed_weight_dtype"] != "bf16"
            or kernel.attributes["routed_scaling_factor"] != 1.0
            or kernel.attributes["swiglu_limit"] is not None):
        return None
    reference = hardware.prepared_moe_operand_resources
    if reference is None:
        return None
    domain = reference["domain"]
    if (not domain["minimum_tokens"] <= kernel.shape["tokens"] <= domain["maximum_tokens"]
            or len(kernel.attributes["owned_experts"]) != domain["local_experts"]
            or any(kernel.shape[name] != domain[name] for name in (
                "experts", "topk", "hidden", "intermediate"))):
        return None
    from src.kernel.fused_moe_physical_specs import build_fused_moe_physical_specs
    from src.kernel.kernel_base import predict
    attributes = {name: kernel.attributes[name] for name in (
        "tokens_per_expert", "local_tokens_per_expert", "expert_parallel_size", "expert_rank",
        "owned_experts", "expert_weight_tile_rows")}
    selection = {**attributes, "launch_policy": "analytical_launch", "model_execution_version": "vllm_0_30_0",
        "source_moe_config": "bundled_or_default", "source_routing": "softmax", "source_batch_invariant": False,
        "source_has_expert_map": True, "source_input_layout": "contiguous_aligned_16", "source_topk_id_dtype": "int32",
        "source_router_state": "prepared_topk_weights_and_ids", "tensor_iterator_source": "pytorch_2_13_sm90",
        "routed_weight_dtype": "bf16", "routing_input_dtype": "bf16", "routing_weight_dtype": "fp32"}
    specs = build_fused_moe_physical_specs(hardware_id=hardware.hardware_id, shape=kernel.shape, dtype=kernel.dtype,
        prediction_attributes=attributes, selection_attributes=selection, hardware=hardware)
    parts = []
    for spec in specs:
        role = spec.variant
        if role == "elementwise":
            if spec.kind is not KernelKind.VECTOR or spec.launch.algorithm != "pytorch_2_13_sm90_fill_bf16":
                raise NotImplementedError("prepared BF16 reference has an unknown elementwise source stage")
            role = "moe_zero_output"
        parts.append((role, predict(spec, hardware, "midlayer")))
    names = tuple(name for name, _ in parts)
    if len(set(names)) != len(names):
        raise ValueError("prepared expert source stages must have unique roles")
    seconds = {name: value.t_total_s for name, value in parts}
    limitations = tuple(dict.fromkeys(item for _, value in parts
        for item in value.diagnostics.limitations)) + (
        "moe_prepared_BF16_source_recipe_is_an_analytical_reference_not_provider_attestation",
        "moe_prepared_stage_predictions_are_serial_inside_one_mathematical_aggregate",
        "moe_prepared_external_router_is_retained_outside_this_prepared_recipe")
    return dataclasses.replace(prediction_from_ns(math.fsum(seconds.values()) * 1e9, "prepared_expert_reference"),
        t_per_group=seconds, t_per_resource=seconds, bottleneck_group=max(seconds, key=seconds.__getitem__),
        diagnostics=PredictionDiagnostics(compute_time_s={}, memory=(), memory_access_classes=(), memory_arbitration=(),
            pipeline=None, critical_path=names, support_level="partial", limitations=limitations))


def _grouped_matmul_phase_ledger(
        inp: PredictionInput, *,
        memory_paths: Optional[Tuple[MemoryPathDemand, ...]] = None,
        compute_demands: Optional[Tuple[ComputeDemand, ...]] = None,
        matrix_preparation_issue_operations: float = 0.0,
        output_conversion_issue_operations: float = 0.0,
        ) -> AnalyticalPhaseLedger:
    """Build the hardware-neutral grouped-projection semantic ledger.

    All three accelerators expose the same operand ingress, matrix multiply,
    and output-writeback semantics.  Their resource services and the amount
    of source ordering available differ, but they no longer choose different
    workload representations merely because of the hardware backend.
    """

    if inp.phase_ledger is not None:
        raise ValueError("FUSED_MOE phase ledger must be attached exactly once")
    if memory_paths is None:
        memory_paths = inp.memory_paths
    if compute_demands is None:
        compute_demands = inp.compute_demands
    return _grouped_matmul_phase_ledger_from_demands(
        memory_paths, compute_demands,
        matrix_preparation_issue_operations=matrix_preparation_issue_operations,
        output_conversion_issue_operations=output_conversion_issue_operations)


def _grouped_matmul_phase_ledger_from_demands(
        memory_paths: Tuple[MemoryPathDemand, ...],
        compute_demands: Tuple[ComputeDemand, ...], *,
        matrix_preparation_issue_operations: float = 0.0,
        output_conversion_issue_operations: float = 0.0,
        ) -> AnalyticalPhaseLedger:
    """Build the shared projection ledger from completed resource demands."""

    if not isinstance(memory_paths, tuple):
        raise TypeError("FUSED_MOE phase-ledger memory paths must be a tuple")
    if not isinstance(compute_demands, tuple):
        raise TypeError("FUSED_MOE phase-ledger compute demands must be a tuple")
    preparation_work = matrix_preparation_issue_operations
    if (isinstance(preparation_work, bool) or not isinstance(preparation_work, (int, float))
            or not math.isfinite(preparation_work) or preparation_work < 0):
        raise ValueError("matrix preparation ledger requires finite nonnegative work")
    conversion_work = output_conversion_issue_operations
    if (isinstance(conversion_work, bool) or not isinstance(conversion_work, (int, float))
            or not math.isfinite(conversion_work) or conversion_work < 0):
        raise ValueError("output conversion ledger requires finite nonnegative work")
    issue_work = math.fsum(d.operations for d in compute_demands
        if d.resource_role == "warp_issue")
    if preparation_work + conversion_work > issue_work:
        raise ValueError("matrix preparation and output conversion exceed conserved issue work")
    read_paths = tuple(
        path for path in memory_paths if path.access_kind == "read")
    write_paths = tuple(
        path for path in memory_paths if path.access_kind == "write")
    if not write_paths:
        raise ValueError(
            "FUSED_MOE grouped projection requires a write path")
    def memory_demands(paths):
        return tuple(
            PhaseMemoryDemand(
                path_name=path.name,
                resource_role=node.resource_role,
                bytes=node.bytes,
            )
            for path in paths
            for node in path.nodes
        )

    if not compute_demands:
        if read_paths:
            raise ValueError(
                "FUSED_MOE grouped projection without matrix work cannot "
                "retain operand reads")
        # An expert-parallel rank with no local route still publishes the
        # source-defined zero-filled foreign rows for the later cross-rank
        # reduction.  It is a write-only semantic phase, not a degenerate
        # matrix multiply and not an invented empty read stage.
        return AnalyticalPhaseLedger(
            phases=(AnalyticalPhase(
                name="foreign_route_zero_writeback",
                predecessors=(),
                compute_demands=(),
                memory_demands=memory_demands(write_paths),
                completion_barrier=True,
            ),),
        )
    routed_weight_paths = tuple(
        path for path in read_paths if path.name == "routed_weight_read")
    if len(routed_weight_paths) > 1:
        raise ValueError(
            "FUSED_MOE grouped projection has duplicate routed-weight paths")
    ingress_read_paths = tuple(
        path for path in read_paths if path.name != "routed_weight_read")
    if not ingress_read_paths:
        raise ValueError("FUSED_MOE grouped projection requires operand reads")

    operation_by_resource = {
        "compute_tensor": "grouped_matrix_multiply",
        "compute_fma": "finite_group_scan_control",
        "compute_alu": "routed_weight_scale",
        "warp_issue": "conservative_entry_startup_issue",
    }
    if any(
            demand.resource_role not in operation_by_resource
            for demand in compute_demands):
        raise ValueError(
            "FUSED_MOE grouped projection has an unknown compute role")
    phase_compute_demands = tuple(
        PhaseComputeDemand(
            resource_role=demand.resource_role,
            operation=operation_by_resource[demand.resource_role],
            operations=demand.operations,
        )
        for demand in compute_demands
    )
    control_demands = tuple(
        demand for demand in phase_compute_demands
        if demand.resource_role == "compute_fma")
    routed_weight_scale_demands = tuple(
        demand for demand in phase_compute_demands
        if demand.resource_role == "compute_alu")
    matrix_demands = tuple(
        demand for demand in phase_compute_demands
        if demand.resource_role == "compute_tensor")
    startup_demands = tuple(
        dataclasses.replace(demand,
            operations=demand.operations-preparation_work-conversion_work)
        for demand in phase_compute_demands
        if demand.resource_role == "warp_issue"
        and demand.operations > preparation_work + conversion_work)
    preparation_demands = ((PhaseComputeDemand(
        resource_role="warp_issue", operation="conservative_matrix_preparation_issue",
        operations=preparation_work),) if preparation_work else ())
    conversion_demands = ((PhaseComputeDemand(
        resource_role="warp_issue", operation="packed_output_BF16_conversion",
        operations=conversion_work),) if conversion_work else ())
    if not matrix_demands:
        raise ValueError(
            "FUSED_MOE grouped projection requires Tensor matrix work")
    if routed_weight_scale_demands and len(routed_weight_paths) != 1:
        raise ValueError(
            "FUSED_MOE routed-weight scale requires one post-matmul "
            "routed-weight read")
    if not routed_weight_scale_demands and routed_weight_paths:
        raise ValueError(
            "FUSED_MOE routed-weight read requires routed-weight scale work")
    control_phase = (
        (AnalyticalPhase(
            name="group_list_scan",
            predecessors=(),
            compute_demands=control_demands,
            memory_demands=(),
        ),)
        if control_demands else ())
    ingress_predecessors = (("group_list_scan",) if control_demands else ())
    startup_phase = (
        (AnalyticalPhase(
            name="source_entry_issue",
            predecessors=ingress_predecessors,
            compute_demands=startup_demands,
            memory_demands=(),
        ),) if startup_demands else ())
    if startup_demands:
        ingress_predecessors = ("source_entry_issue",)
    preparation_phase = ((AnalyticalPhase(
        name="matrix_preparation_issue", predecessors=("operand_ingress",),
        compute_demands=preparation_demands, memory_demands=()),)
        if preparation_demands else ())
    routed_weight_read_phase = (
        (AnalyticalPhase(
            name="routed_weight_read",
            predecessors=("matrix_multiply",),
            compute_demands=(),
            memory_demands=memory_demands(routed_weight_paths),
        ),)
        if routed_weight_paths else ())
    scale_phase = (
        (AnalyticalPhase(
            name="routed_weight_scale",
            predecessors=("routed_weight_read",),
            compute_demands=routed_weight_scale_demands,
            memory_demands=(),
        ),)
        if routed_weight_scale_demands else ())
    output_predecessor = (
        "routed_weight_scale" if routed_weight_scale_demands else
        "output_conversion" if conversion_demands else "matrix_multiply")
    conversion_phase = ((AnalyticalPhase(
        name="output_conversion", predecessors=("matrix_multiply",),
        compute_demands=conversion_demands, memory_demands=()),)
        if conversion_demands else ())
    return AnalyticalPhaseLedger(
        phases=control_phase + startup_phase + (
            AnalyticalPhase(
                name="operand_ingress",
                predecessors=ingress_predecessors,
                compute_demands=(),
                memory_demands=memory_demands(ingress_read_paths),
            ),
        ) + preparation_phase + (AnalyticalPhase(
                name="matrix_multiply",
                predecessors=(("matrix_preparation_issue",)
                    if preparation_demands else ("operand_ingress",)),
                compute_demands=matrix_demands,
                memory_demands=(),
            ),
        ) + routed_weight_read_phase + scale_phase + conversion_phase + (
            AnalyticalPhase(
                name="output_writeback",
                predecessors=(output_predecessor,),
                compute_demands=(),
                memory_demands=memory_demands(write_paths),
                completion_barrier=True,
            ),
        ),
    )




def _cann_grouped_matmul_analytical_input(
        kernel: KernelSpec, hardware, *,
        demand_per_resource: dict, eta: float, wave_efficiency: float,
        active_units: int, include_dispatch_overhead: bool,
        limitations: Tuple[str, ...], traffic: _CannGroupedMatmulTraffic,
        work: _CannGroupedMatmulWork,
        residency: _CannGroupedOperandResidency,
        attach_pipeline: bool = True) -> PredictionInput:
    """Construct CANN's final grouped-projection envelope in one pass.

    The aggregate AIC input, path split, typed operand annotation and
    analytical envelope are all immutable descriptions of one source-defined
    launch.  Materializing their final form directly avoids validating the
    three intermediate descriptions that never reach the formula.
    """

    if hardware.kernel_memory_model != "npu_tile":
        raise ValueError("CANN grouped analytical input requires npu_tile")
    if not hardware.aic_aiv_separated:
        raise ValueError(
            "CANN grouped analytical input requires separated AIC/AIV")
    if not isinstance(limitations, tuple):
        raise TypeError("CANN grouped analytical limitations must be a tuple")
    final_demand = dict(demand_per_resource)
    for role in ("l1_cache", "smem", "xbar"):
        final_demand[role] = 0.0
    tensor_operations = float(final_demand["compute_tensor"])
    if tensor_operations <= 0:
        raise ValueError("CANN grouped analytical input requires Tensor work")
    source = MemorySourceDemand(
        active_units=active_units,
        producers_per_unit=1.0,
        independent_requests_per_producer=None,
        frontend_rate_per_unit_s=None,
    )
    paths = (
        MemoryPathDemand(
            name="mte2_read",
            access_kind="read",
            initial_transactions=None,
            nodes=(MemoryNodeDemand(
                resource_role="l2_cache",
                transactions=None,
                bytes=traffic.l2_read_bytes,
                transaction_bytes=None,
            ),),
            source=source,
        ),
        MemoryPathDemand(
            name="operand_dram_read",
            access_kind="read",
            initial_transactions=None,
            nodes=(MemoryNodeDemand(
                resource_role="dram",
                transactions=None,
                bytes=traffic.dram_read_bytes,
                transaction_bytes=None,
            ),),
            source=source,
        ),
        MemoryPathDemand(
            name="aic_output_l2_write",
            access_kind="write",
            initial_transactions=None,
            nodes=(
                MemoryNodeDemand(
                    resource_role="l2_cache",
                    transactions=None,
                    bytes=traffic.output_l2_write_bytes,
                    transaction_bytes=None,
                ),
                MemoryNodeDemand(
                    resource_role="dram",
                    transactions=None,
                    bytes=traffic.dram_write_bytes,
                    transaction_bytes=None,
                ),
            ),
            source=source,
        ),
    )
    paths = _cann_grouped_operand_service_paths_from_paths(
        paths, kernel, residency, hardware)
    compute_demands = [ComputeDemand(
        resource_role="compute_tensor",
        operations=tensor_operations,
        shape_efficiency=1.0,
        schedule_efficiency=1.0,
    )]
    scan_service_name = "cann_grouped_matmul_finite_group_scan_control"
    fma_resource = hardware.resource_by_role("compute_fma")
    if scan_service_name in fma_resource.exact_request_services:
        scan_service = fma_resource.exact_request_services[scan_service_name]
        scan_supported = work.expert_group_count in (
            scan_service["service_time_s_by_request_size"])
    else:
        scan_supported = False
    if scan_supported:
        scan_request = ComputeExactRequestDemand(
            service=scan_service_name,
            request_sizes_by_worker=tuple(
                (work.expert_group_count,)
                for _ in range(work.worker_count)),
        )
        scan_operations = scan_request.operations
        final_demand["compute_fma"] = scan_operations
        compute_demands.append(ComputeDemand(
            resource_role="compute_fma",
            operations=scan_operations,
            shape_efficiency=1.0,
            schedule_efficiency=1.0,
            exact_request_demands=(scan_request,),
        ))
    compute_demands = tuple(compute_demands)
    phase_ledger = (
        _grouped_matmul_phase_ledger_from_demands(paths, compute_demands)
        if attach_pipeline else None)
    pipeline_plan = (
        _cann_grouped_matmul_envelope_demand(kernel, work, residency, hardware)
        if attach_pipeline else None)
    analytical_limitations = (
        ("cann_grouped_matmul_uses_tile_class_analytical_envelope",)
        if pipeline_plan is not None else
        ("cann_grouped_matmul_tile_class_service_unavailable",))
    if not scan_supported:
        analytical_limitations += (
            "cann_grouped_matmul_finite_group_scan_service_unavailable",)
    resource_input = PredictionInput(
        demand_per_resource=final_demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model="npu_tile",
        compute_demands=compute_demands,
        memory_paths=paths,
        pipeline_plan=pipeline_plan,
        launch_task_kind="aic",
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        include_dispatch_overhead=include_dispatch_overhead,
        support_level=("partial" if pipeline_plan is not None
                       else "aggregate_only"),
        limitations=limitations + analytical_limitations,
        phase_ledger=phase_ledger,
    )
    return resource_input




def _split_gpu_grouped_gemm_memory_paths(
        inp: PredictionInput,
        traffic: _GpuGroupedGemmTraffic) -> PredictionInput:
    """Keep operand, post-matmul router and output paths distinct."""

    if inp.mem_model != "gpu_mshr":
        raise ValueError("GPU grouped-GEMM paths require gpu_mshr memory")
    if len(inp.memory_paths) != 1:
        raise ValueError(
            "GPU grouped-GEMM path split requires one aggregate path")
    original = inp.memory_paths[0]
    if original.name != "global_memory" or original.access_kind != "read":
        raise ValueError(
            "GPU grouped-GEMM path split requires the global memory path")
    if original.access_class_demands:
        raise ValueError(
            "GPU grouped-GEMM aggregate path already has a typed service")
    original_by_role = {
        node.resource_role: node.bytes for node in original.nodes}
    expected_by_role = {
        "l2_cache": traffic.l2_total_bytes,
        "dram": traffic.dram_total_bytes,
    }
    if set(original_by_role) != set(expected_by_role) or any(
            not math.isclose(
                original_by_role[role], expected,
                rel_tol=1e-12, abs_tol=1e-6)
            for role, expected in expected_by_role.items()):
        raise ValueError(
            "GPU grouped-GEMM aggregate path does not conserve traffic")
    transaction_bytes_by_role = {
        node.resource_role: node.transaction_bytes
        for node in original.nodes}

    def make_path(name, access_kind, l2_bytes, dram_bytes):
        if l2_bytes <= 0:
            raise ValueError(
                f"GPU grouped-GEMM {name} requires positive L2 traffic")
        if dram_bytes < 0 or dram_bytes > l2_bytes:
            raise ValueError(
                f"GPU grouped-GEMM {name} has invalid DRAM traffic")
        l2_transaction_bytes = transaction_bytes_by_role["l2_cache"]
        dram_transaction_bytes = transaction_bytes_by_role["dram"]
        if l2_transaction_bytes is None or dram_transaction_bytes is None:
            raise ValueError(
                "GPU grouped-GEMM paths require resolved transactions")
        nodes = [MemoryNodeDemand(
            resource_role="l2_cache",
            transactions=l2_bytes / l2_transaction_bytes,
            bytes=l2_bytes,
            transaction_bytes=l2_transaction_bytes,
        )]
        if dram_bytes > 0:
            nodes.append(MemoryNodeDemand(
                resource_role="dram",
                transactions=dram_bytes / dram_transaction_bytes,
                bytes=dram_bytes,
                transaction_bytes=dram_transaction_bytes,
            ))
        return MemoryPathDemand(
            name=name,
            access_kind=access_kind,
            initial_transactions=l2_bytes / l2_transaction_bytes,
            nodes=tuple(nodes),
            source=original.source,
            queues=original.queues,
        )

    paths = []
    for name, access_kind, l2_bytes, dram_bytes in (
        (
            "activation_read", "read",
            traffic.activation_l2_read_bytes,
            traffic.activation_dram_read_bytes,
        ),
        (
            "weight_read", "read",
            traffic.weight_l2_read_bytes,
            traffic.weight_dram_read_bytes,
        ),
        (
            "routed_weight_read", "read",
            traffic.routed_weight_l2_read_bytes,
            traffic.routed_weight_dram_read_bytes,
        ),
        (
            "output_write", "write",
            traffic.output_l2_write_bytes,
            traffic.output_dram_write_bytes,
        ),
    ):
        if l2_bytes > 0:
            paths.append(make_path(
                name, access_kind, l2_bytes, dram_bytes))
    if not paths:
        raise ValueError(
            "GPU grouped-GEMM requires at least one nonzero memory path")
    return dataclasses.replace(inp, memory_paths=tuple(paths))


def _architecture_input(inp: PredictionInput, kernel: KernelSpec, hardware):
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        # The CANN branch already constructs the canonical npu_tile contract.
        # Avoid re-validating an observationally identical intermediate input.
        return inp
    elif memory_model == "gpu_mshr":
        return inp
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _apply_h200_fused_moe_w2_active_sm_clock(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Use the independent active clock only for its source launch domain."""

    clock = hardware.h200_fused_moe_w2_active_sm_clock
    if clock is None:
        return inp
    source = clock["source_launch"]
    if (kernel.dtype != source["dtype"]
            or kernel.variant != source["variant"]
            or kernel.launch.algorithm != source["algorithm"]
            or kernel.launch.block_size != source["threads_per_cta"]
            or not source["grid_min"] <= kernel.launch.grid_size
            <= source["grid_max"]
            or _tile(kernel) != (
                source["tile_m"], source["tile_n"], source["tile_k"])):
        return inp
    if any(kernel.shape[name] != source[name] for name in (
            "experts", "hidden", "intermediate")):
        return inp
    params = kernel.launch.parameters
    required = ("group_size_m", "dynamic_shared_memory_bytes")
    if (not set(required).issubset(params)
            or any(params[name] != source[name] for name in required)):
        return inp
    if kernel.shape["intermediate"] // source["tile_k"] != 12:
        raise ValueError("H200 W2 clock scope disagrees with K rounds")
    return dataclasses.replace(
        inp,
        sm_freq_hz=clock["sm_clock_hz"],
        limitations=(*inp.limitations,
                     "h200_modal_w2_active_sm_clock_provisional",
                     "h200_w2_probe_register_routing_output_difference"),
    )


def _apply_metax_bm32_fused_moe_mma_occupancy(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Apply the independent C500 BM32 Tensor occupancy mechanism.

    The source-resolved Triton configuration and compiler resource record
    close this to BM32/BN64/BK128, four wave64 threads, and a 48-KiB dynamic
    shared allocation.  That allocation admits one CTA/AP on C500, so the
    register-only BF16 MMA point at four resident waves and two accumulator
    chains is directly applicable.  No Serving or kernel duration selects
    this path.
    """

    occupancy = hardware.metax_fused_moe_bm32_bf16_mma_occupancy
    if occupancy is None:
        return inp
    if kernel.dtype != occupancy["dtype"]:
        return inp
    if kernel.launch.algorithm not in {
            "triton_fused_moe_w1", "triton_fused_moe_w2"}:
        return inp
    if kernel.launch.block_size != occupancy["threads_per_cta"]:
        return inp
    if kernel.launch.grid_size < hardware.num_compute_units:
        return inp
    tiling = kernel.launch.tiling
    if tiling is None:
        raise ValueError("FUSED_MOE BM32 occupancy requires a tiled launch")
    if tuple(
            int(tiling.tile_shape[name]) for name in ("m", "n", "k")) != (
                occupancy["tile_m"], occupancy["tile_n"],
                occupancy["tile_k"]):
        return inp
    dynamic_shared_parameter = "dynamic_shared_memory_bytes"
    if dynamic_shared_parameter not in kernel.launch.parameters:
        return inp
    if kernel.launch.parameters[dynamic_shared_parameter] != occupancy[
            "dynamic_shared_memory_bytes"]:
        return inp
    tensor_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_tensor")
    if len(tensor_demands) != 1:
        raise ValueError(
            "FUSED_MOE BM32 occupancy requires one Tensor compute demand")
    tensor = tensor_demands[0]
    efficiency = (
        occupancy["flops_per_ap_cycle"]
        / occupancy["architecture_peak_flops_per_ap_cycle"])
    if not 0.0 < efficiency <= 1.0:
        raise ValueError("FUSED_MOE BM32 MMA occupancy efficiency is invalid")
    adjusted = dataclasses.replace(
        tensor,
        schedule_efficiency=tensor.schedule_efficiency * efficiency,
    )
    return dataclasses.replace(
        inp,
        compute_demands=tuple(
            adjusted if demand is tensor else demand
            for demand in inp.compute_demands),
    )


def _undeclared_grouped_operand_tile_class(
        kernel: KernelSpec, hardware) -> bool:
    """Whether no declared grouped-operand service covers the tile class.

    Grouped-operand services are keyed by the launched Triton tile class
    (dtype, BM, BN, BK, GROUP_SIZE_M).  A launch whose tile class no declared
    service carries (e.g. a newer vLLM default configuration than the one a
    hardware's services were measured for) is outside every typed service's
    domain: it keeps the aggregate grouped-GEMM memory path together with
    its explicit ``moe_gather_service_class_not_separated`` limitation, the
    same contract as a masked-BK gather tail.  A launch whose tile class is
    declared but whose routing topology has no exact service remains a
    contract error.
    """

    tile_m, tile_n, tile_k = _tile(kernel)
    group_size_m = int(kernel.launch.parameters["group_size_m"])
    launch_class = (kernel.dtype, tile_m, tile_n, tile_k, group_size_m)
    declared_classes = {
        (service.signature.dtype, service.signature.tile_m,
         service.signature.tile_n, service.signature.tile_k,
         service.signature.group_size_m)
        for service in hardware.memory_access_class_services
        if service.signature.operand_role is not None
    }
    return launch_class not in declared_classes


_native_has_memory_access_class_service = HWSpec.has_memory_access_class_service


def _attach_gpu_gather_service(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        *, traffic: _GpuGroupedGemmTraffic, routed_rows: float,
        projection_k: int, projection_n: int,
        element_bytes: int) -> PredictionInput:
    """Annotate exact grouped-operand services when hardware provides them."""

    # A800 and C500 select their own exact source service/signature below,
    # then share the immutable typed-memory contract binder.  Only the
    # source-conserved byte count remains dynamic for each prediction.
    MemoryAccessClassDemand = bind_memory_access_class_template

    metax_pipeline_service = _metax_multi_tile_operand_pipeline_service(
        kernel, hardware)
    nvidia_source_depth_service = _nvidia_source_depth_operand_service(
        kernel, hardware)
    if metax_pipeline_service is not None:
        combined_service = metax_pipeline_service
    elif nvidia_source_depth_service is not None:
        combined_service = nvidia_source_depth_service
    else:
        combined_service = _gpu_exact_grouped_operand_service(
            kernel, hardware)
    split_services = (
        None if nvidia_source_depth_service is not None
        else _gpu_split_operand_services(kernel, hardware))
    full_group_services = (
        {} if nvidia_source_depth_service is not None
        else _gpu_full_group_operand_services(kernel, hardware))
    has_service = hardware.has_memory_access_class_service
    if (type(hardware) is HWSpec
            and type(has_service) is MethodType
            and has_service.__func__ is _native_has_memory_access_class_service
            and has_service.__self__ is hardware
            and type(hardware.memory_access_class_services) is tuple
            and all(type(item) is MemoryAccessClassService
                and type(item.service) is str
                for item in hardware.memory_access_class_services)):
        # These queries are nonempty source constants. The native method
        # scans the same immutable records for every name; index them once
        # for this construction only. Rebuild after any HWSpec mutation.
        service_names = frozenset(
            item.service for item in hardware.memory_access_class_services)
        has_service = service_names.__contains__
    declared_grouped_service = any(
        has_service(service)
        for service in (
            _GROUPED_OPERAND_SERVICE,
            _GROUPED_OPERAND_PIPELINE_SERVICE,
            _GROUPED_WEIGHT_OVERLAPPED_DRAM_MISS_SERVICE,
            _NVIDIA_EIGHT_EXPERT_68_ACTIVE_72_MAX_W1_SERVICE,
            _NVIDIA_EIGHT_EXPERT_132_ACTIVE_136_MAX_W1_SERVICE,
            _NVIDIA_E512_DEEP_W2_SERVICE,
            _NVIDIA_E128_DEEP_W2_SERVICE,
            *_NVIDIA_E128_H2048_8250_ACTIVE_8318_MAX_SERVICE_BY_VARIANT.values(),
            _NVIDIA_E8_H6144_DEEP_W2_SERVICE,
            _NVIDIA_E8_H6144_DEEP_W1_SERVICE,
            _NVIDIA_E8_H4096_515_ACTIVE_520_MAX_W2_SERVICE,
            _NVIDIA_E8_H4096_1027_ACTIVE_1032_MAX_W2_SERVICE,
            _NVIDIA_E512_H4096_DEEP_W1_SERVICE,
            _NVIDIA_DEEP_H2880_W1_W2_SERVICE,
            _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE,
            _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE,
            _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
            _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
            _NVIDIA_BM16_E256_16_ACTIVE_241_MAX_W2_SERVICE,
            _NVIDIA_BM16_E512_20_ACTIVE_482_MAX_W2_SERVICE,
            _METAX_GROUPED_OPERAND_LONG_K_SERVICE,
            _METAX_GROUPED_OPERAND_WIDE_N_SERVICE,
            _METAX_GLOBAL_DEEP_OPERAND_LONG_K_SERVICE,
            _METAX_GLOBAL_LARGE_OPERAND_WIDE_N_SERVICE,
            _METAX_LONG_LIFETIME_OPERAND_LONG_K_SERVICE,
            _METAX_LONG_LIFETIME_OPERAND_WIDE_N_SERVICE,
            _METAX_REPEATED_OPERAND_DEEP_K_SERVICE,
            _METAX_REPEATED_OPERAND_WIDE_N_SERVICE,
            _METAX_REPEATED_OPERAND_BALANCED_N_SERVICE,
            _METAX_REPEATED_OPERAND_MID_LONG_K_SERVICE,
            _METAX_REPEATED_OPERAND_MID_WIDE_N_SERVICE,
            _METAX_SINGLE_TILE_OPERAND_LONG_K_SERVICE,
            _METAX_SINGLE_TILE_OPERAND_WIDE_N_SERVICE,
            _METAX_EIGHT_EXPERT_EIGHT_ACTIVE_TWELVE_MAX_W2_SERVICE,
            _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W1_SERVICE,
            _METAX_EIGHT_EXPERT_SINGLE_TILE_LARGE_W2_SERVICE,
            _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W1_SERVICE,
            _METAX_EIGHT_EXPERT_SMALL_SINGLE_TILE_W2_SERVICE,
            _METAX_EIGHT_EXPERT_STEADY_WIDE_N_SERVICE,
            _METAX_EIGHT_EXPERT_STEADY_K192_N512_W1_SERVICE,
            _METAX_E256_LONG_LIFETIME_LONG_K_SERVICE,
            _METAX_E256_LONG_LIFETIME_WIDE_N_SERVICE,
            _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W1_SERVICE,
            _METAX_E512_DENSE_TEN_TO_ELEVEN_TILE_W2_SERVICE,
            _METAX_E512_DENSE_LARGE_W2_SERVICE,
            _METAX_E128_LONG_LIFETIME_W1_SERVICE,
            _METAX_E128_LONG_LIFETIME_W2_SERVICE,
            _METAX_E128_DEEP_W2_SERVICE,
            _METAX_E128_EIGHT_TILE_W1_SERVICE,
            _METAX_E128_EIGHT_TILE_W2_SERVICE,
            _METAX_E128_TOPK16_EIGHT_TILE_W1_SERVICE,
            _METAX_E128_EIGHT_TILE_WIDE_N_W2_SERVICE,
            _METAX_EIGHT_EXPERT_TWELVE_TILE_W2_SERVICE,
            _METAX_EIGHT_EXPERT_TWELVE_TILE_K128_N448_W1_SERVICE,
            _METAX_EIGHT_EXPERT_TWELVE_TILE_K192_N512_W1_SERVICE,
            _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W1_SERVICE,
            _METAX_EIGHT_EXPERT_TWENTY_ONE_TILE_W2_SERVICE,
            _METAX_E128_SPARSE_TWO_TILE_W1_SERVICE,
            _METAX_E128_SPARSE_TWO_TILE_W2_SERVICE,
            _METAX_E128_SINGLE_TILE_W1_SERVICE,
            _METAX_E128_SINGLE_TILE_W2_SERVICE,
            _METAX_E128_SINGLE_TILE_MID_W1_SERVICE,
            _METAX_E128_SINGLE_TILE_MID_W2_SERVICE,
            _METAX_E128_SINGLE_TILE_LARGE_W1_SERVICE,
            _METAX_E128_SINGLE_TILE_LARGE_W2_SERVICE,
            _METAX_E128_UPPER_TWO_TILE_W1_SERVICE,
            _METAX_E128_UPPER_TWO_TILE_W2_SERVICE,
            _METAX_E128_TOPK8_128_ACTIVE_190_MAX_W1_SERVICE,
            _METAX_E128_TOPK16_128_ACTIVE_190_MAX_W1_SERVICE,
            _METAX_E128_128_ACTIVE_190_MAX_W2_SERVICE,
            _METAX_E256_EIGHT_TILE_LONG_K_SERVICE,
            _METAX_E256_EIGHT_TILE_WIDE_N_SERVICE,
            _METAX_E256_EIGHT_TILE_MID_W1_SERVICE,
            _METAX_E256_EIGHT_TILE_MID_W2_SERVICE,
            _METAX_E256_FOUR_TILE_LONG_K_SERVICE,
            _METAX_E256_FOUR_TILE_WIDE_N_SERVICE,
            _METAX_E256_FOUR_TILE_MID_LONG_K_SERVICE,
            _METAX_E256_FOUR_TILE_MID_WIDE_N_SERVICE,
            _METAX_E128_FOUR_TILE_W1_SERVICE,
            _METAX_E128_FOUR_TILE_W2_SERVICE,
            _METAX_E128_REPEATED_SOURCE_TOPK8_W1_SERVICE,
            _METAX_E128_REPEATED_SOURCE_TOPK16_W1_SERVICE,
            _METAX_E128_REPEATED_K24_N32_W2_SERVICE,
            _METAX_E256_UPPER_TWO_TILE_LONG_K_SERVICE,
            _METAX_E256_UPPER_TWO_TILE_WIDE_N_SERVICE,
            _METAX_E256_SINGLE_TILE_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_W2_SERVICE,
            _METAX_E256_SINGLE_TILE_MID_DEEP_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_MID_DEEP_W2_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_W2_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_DEEP_W1_SERVICE,
            _METAX_E256_SINGLE_TILE_LARGE_DEEP_W2_SERVICE,
            _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W1_SERVICE,
            _METAX_BM16_E512_ALL_ACTIVE_SHORT_QUEUE_W2_SERVICE,
            _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W1_SERVICE,
            _METAX_BM64_E512_ALL_ACTIVE_SINGLE_TILE_W2_SERVICE,
            _METAX_BM16_ACTIVE_GRID_WIDE_N_SERVICE,
            _METAX_BM16_E256_SIXTEEN_ACTIVE_W2_SERVICE,
            _METAX_BM16_SPARSE_PADDED_W2_SERVICE,
            _METAX_BM16_E512_SPARSE_PADDED_W2_SERVICE,
            _METAX_BM16_E512_FORTY_ACTIVE_W2_SERVICE,
            _METAX_BM16_E256_SPARSE_COLLIDING_W1_SERVICE,
            _METAX_BM16_E256_SPARSE_COLLIDING_W2_SERVICE,
            _METAX_BM16_E256_106_ACTIVE_248_MAX_W1_SERVICE,
            _METAX_BM16_E256_106_ACTIVE_248_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_152_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_152_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_64_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_124_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_128_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_128_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_136_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_136_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_184_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_184_MAX_W2_SERVICE,
            _METAX_BM16_E128_TOPK16_248_MAX_W1_SERVICE,
            _METAX_BM16_E128_TOPK16_248_MAX_W2_SERVICE,
            _METAX_BM16_FOUR_ACTIVE_MID_K_W1_SERVICE,
            _METAX_BM16_FOUR_ACTIVE_MID_K_W2_SERVICE,
            _METAX_BM16_EIGHT_ACTIVE_MID_K_W2_SERVICE,
            _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W1_SERVICE,
            _METAX_BM16_E32_SEVEN_ACTIVE_MID_K_W2_SERVICE,
        ))
    grouped_service = (
        combined_service is not None or split_services is not None)
    legacy_service = has_service(
        "indirect_row_gather")
    metax_fallback_family = has_service(
        _METAX_GROUPED_OPERAND_LONG_K_SERVICE)
    if (declared_grouped_service and legacy_service
            and not metax_fallback_family):
        raise ValueError(
            "FUSED_MOE hardware cannot declare grouped and legacy gather "
            "services together")
    if combined_service is not None and split_services is not None:
        raise ValueError(
            "FUSED_MOE launch cannot select combined and split grouped "
            "operand services together")
    if full_group_services and split_services is None:
        raise ValueError(
            "FUSED_MOE full-group service requires the split operand pair")
    if (declared_grouped_service and not grouped_service
            and not (metax_fallback_family and legacy_service)):
        if _undeclared_grouped_operand_tile_class(kernel, hardware):
            if "moe_gather_service_class_not_separated" not in (
                    inp.limitations):
                raise ValueError(
                    "aggregate grouped FUSED_MOE path lost its explicit "
                    "service-class limitation")
            return inp
        raise ValueError(
            "grouped FUSED_MOE has no exact hardware signature family")
    if not declared_grouped_service and not legacy_service:
        return inp
    if inp.mem_model != "gpu_mshr":
        raise ValueError(
            "FUSED_MOE indirect-row gather requires gpu_mshr memory")
    if grouped_service:
        tile_m, tile_n, _ = _tile(kernel)
        activation_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "activation_read")
        weight_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "weight_read")
        output_path, = tuple(
            path for path in inp.memory_paths
            if path.name == "output_write")
        routed_weight_paths = tuple(
            path for path in inp.memory_paths
            if path.name == "routed_weight_read")
        if len(routed_weight_paths) > 1:
            raise ValueError(
                "FUSED_MOE grouped operands have duplicate routed-weight "
                "paths")
        if routed_weight_paths and routed_weight_paths[0].access_class_demands:
            raise ValueError(
                "FUSED_MOE routed-weight path already has a typed service")
        if (activation_path.access_class_demands
                or weight_path.access_class_demands):
            raise ValueError(
                "FUSED_MOE grouped operand paths already have a typed service")
        if (activation_path.source != weight_path.source
                or activation_path.queues != weight_path.queues):
            raise ValueError(
                "FUSED_MOE grouped operand paths must share source and queues")
        activation_nodes = {
            node.resource_role: node for node in activation_path.nodes}
        weight_nodes = {
            node.resource_role: node for node in weight_path.nodes}
        if set(activation_nodes) != set(weight_nodes):
            raise ValueError(
                "FUSED_MOE grouped operand paths must traverse the same nodes")
        operand_nodes = []
        for role in ("l2_cache", "dram"):
            if role not in activation_nodes:
                continue
            activation_node = activation_nodes[role]
            weight_node = weight_nodes[role]
            if activation_node.transaction_bytes != weight_node.transaction_bytes:
                raise ValueError(
                    "FUSED_MOE grouped operand paths disagree on transaction "
                    f"size for {role!r}")
            transaction_bytes = activation_node.transaction_bytes
            bytes_total = activation_node.bytes + weight_node.bytes
            operand_nodes.append(MemoryNodeDemand(
                resource_role=role,
                transactions=(
                    bytes_total / transaction_bytes
                    if transaction_bytes is not None else None),
                bytes=bytes_total,
                transaction_bytes=transaction_bytes,
            ))
        l2_transaction_bytes = activation_nodes["l2_cache"].transaction_bytes
        if l2_transaction_bytes is None:
            raise ValueError(
                "FUSED_MOE grouped operand service requires L2 transactions")
        operand_bytes = (
            traffic.activation_l2_read_bytes
            + traffic.weight_l2_read_bytes)
        if split_services is not None:
            l2_hit_bytes = (
                operand_bytes
                - traffic.activation_dram_read_bytes
                - traffic.weight_dram_read_bytes)
            if l2_hit_bytes < 0.0:
                raise ValueError(
                    "grouped FUSED_MOE operand L2-hit bytes cannot be "
                    "negative")
            full_group_hit_bytes_by_weight_streams = {}
            variant, source_rows, _, _ = _projection(kernel)
            source_footprint_bytes = (
                source_rows * projection_k * element_bytes)
            if (full_group_services
                    and variant == "w1"
                    and source_footprint_bytes > hardware.l2_size_bytes):
                full_group_operand_bytes_by_weight_streams = (
                    _gpu_full_group_operand_bytes_by_weight_streams(
                        kernel, projection_k=projection_k,
                        projection_n=projection_n))
                dram_operand_bytes = (
                    traffic.activation_dram_read_bytes
                    + traffic.weight_dram_read_bytes)
                full_group_hit_bytes_by_weight_streams = {
                    topology: max(0.0, logical_bytes - dram_operand_bytes)
                    for topology, logical_bytes in (
                        full_group_operand_bytes_by_weight_streams.items())
                    if topology in full_group_services
                }
            classified_group_hit_bytes = math.fsum(
                full_group_hit_bytes_by_weight_streams.values())
            if classified_group_hit_bytes > l2_hit_bytes + 1e-6:
                raise ValueError(
                    "grouped FUSED_MOE classified group hits exceed total "
                    "L2-hit operands")
            split_demands = [(
                _GROUPED_OPERAND_PIPELINE_SERVICE,
                max(0.0, l2_hit_bytes - classified_group_hit_bytes),
                split_services["combined"].signature,
            )]
            split_demands.extend(
                (
                    full_group_services[topology].service,
                    logical_bytes,
                    full_group_services[topology].signature,
                )
                for topology, logical_bytes in sorted(
                    full_group_hit_bytes_by_weight_streams.items(),
                    key=lambda item: str(item[0]))
            )
            split_demands.append((
                split_services["weight"].service,
                traffic.weight_dram_read_bytes,
                split_services["weight"].signature,
            ))
            access_class_demands = tuple(
                MemoryAccessClassDemand(
                    service=service,
                    ingress_resource_role="l2_cache",
                    logical_bytes=logical_bytes,
                    signature=signature,
                )
                for service, logical_bytes, signature in split_demands
                if logical_bytes > 0.0)
        elif nvidia_source_depth_service is not None:
            access_class_demands = (MemoryAccessClassDemand(
                service=nvidia_source_depth_service.service,
                ingress_resource_role="l2_cache",
                logical_bytes=operand_bytes,
                signature=nvidia_source_depth_service.signature,
                replaces_ingress_path_service=True,
            ),)
        elif metax_pipeline_service is not None:
            access_class_demands = (MemoryAccessClassDemand(
                service=metax_pipeline_service.service,
                ingress_resource_role="l2_cache",
                logical_bytes=operand_bytes,
                signature=metax_pipeline_service.signature,
            ),)
        else:
            services_by_active_rows = (
                _gpu_grouped_operand_service_by_active_rows(
                    kernel, hardware))
            service = services_by_active_rows[tile_m]
            if len(services_by_active_rows) == 1:
                access_class_demands = (MemoryAccessClassDemand(
                    service=_GROUPED_OPERAND_SERVICE,
                    ingress_resource_role="l2_cache",
                    logical_bytes=operand_bytes,
                    signature=service.signature,
                ),)
            else:
                _, local_counts = _routing(kernel)
                n_tiles = math.ceil(projection_n / tile_n)
                programs_by_active_rows = {}
                for count in local_counts:
                    full_tiles, tail_rows = divmod(count, tile_m)
                    if full_tiles:
                        if tile_m in programs_by_active_rows:
                            programs_by_active_rows[tile_m] += (
                                full_tiles * n_tiles)
                        else:
                            programs_by_active_rows[tile_m] = (
                                full_tiles * n_tiles)
                    if tail_rows:
                        if tail_rows in programs_by_active_rows:
                            programs_by_active_rows[tail_rows] += n_tiles
                        else:
                            programs_by_active_rows[tail_rows] = n_tiles
                activation_replay, weight_replay = (
                    _gpu_operand_replay_factors(kernel))
                access_class_demands = tuple(
                    MemoryAccessClassDemand(
                        service=_GROUPED_OPERAND_SERVICE,
                        ingress_resource_role="l2_cache",
                        logical_bytes=float(
                            program_count
                            * projection_k
                            * element_bytes
                            * (
                                active_rows * activation_replay
                                + tile_n * weight_replay)),
                        signature=services_by_active_rows[
                            active_rows].signature,
                    )
                    for active_rows, program_count in sorted(
                        programs_by_active_rows.items()))
                segmented_operand_bytes = math.fsum(
                    demand.logical_bytes for demand in access_class_demands)
                if not math.isclose(
                        segmented_operand_bytes, operand_bytes,
                        rel_tol=1e-12, abs_tol=1e-6):
                    raise ValueError(
                        "grouped FUSED_MOE masked-row services do not "
                        "conserve operand traffic")
        operand_path = MemoryPathDemand(
            name="operand_read",
            access_kind="read",
            initial_transactions=operand_bytes / l2_transaction_bytes,
            nodes=tuple(operand_nodes),
            source=activation_path.source,
            queues=activation_path.queues,
            access_class_demands=access_class_demands,
        )
        paths = [operand_path, *routed_weight_paths, output_path]
        demand_by_path = None
    else:
        service_name = "indirect_row_gather"
        tile_m, tile_n, tile_k = _tile(kernel)
        activation_replay, _ = _gpu_operand_replay_factors(kernel)
        logical_bytes = float(
            activation_replay
            * routed_rows
            * projection_k
            * math.ceil(projection_n / tile_n)
            * element_bytes)
        signature = MemoryAccessClassSignature(
            access_class=service_name,
            dtype=kernel.dtype,
            tile_m=tile_m,
            tile_k=tile_k,
            # Measured services are keyed by their BF16 tile class.
            logical_row_bytes=tile_k * int(kernel.shape["dtype_bytes"]),
        )
        matching_services = tuple(
            service for service in hardware.memory_access_class_services
            if service.service == service_name
            and service.access_kind == "read"
            and service.ingress_resource_role == "l2_cache"
            and service.signature == signature)
        if len(matching_services) > 1:
            raise ValueError(
                "FUSED_MOE indirect gather repeats an exact hardware "
                "signature")
        if not matching_services:
            # Triton masks an incomplete K iteration.  The launch and its
            # logical bytes remain valid, but a full-BK gather measurement
            # cannot be reused for a BK tail.  Keep the aggregate memory path
            # and its explicit unsupported typed-service limitation instead
            # of fabricating a matching service signature.
            return inp
        demand_by_path = {
            "activation_read": MemoryAccessClassDemand(
                service=service_name,
                ingress_resource_role="l2_cache",
                logical_bytes=logical_bytes,
                signature=matching_services[0].signature,
            ),
        }
    if not grouped_service:
        matching_paths = tuple(
            path for path in inp.memory_paths if path.name == "activation_read")
        if len(matching_paths) != 1:
            raise ValueError(
                "FUSED_MOE gather service requires exactly one activation path")
        if matching_paths[0].access_class_demands:
            raise ValueError(
                "FUSED_MOE activation path already has a typed service")
        paths = []
        for path in inp.memory_paths:
            if path.name in demand_by_path:
                paths.append(dataclasses.replace(
                    path, access_class_demands=(demand_by_path[path.name],)))
            elif path.name in (
                    "activation_read", "weight_read", "routed_weight_read",
                    "output_write"):
                paths.append(path)
            else:
                raise ValueError(
                    "FUSED_MOE gather service encountered an unexpected "
                    f"memory path: {path.name!r}")
    absent_limitation = "moe_gather_service_class_not_separated"
    if absent_limitation not in inp.limitations:
        raise ValueError(
            "FUSED_MOE gather capability cannot clear an absent limitation")
    limitations = tuple(
        value for value in inp.limitations
        if value != absent_limitation
        and not (
            combined_service is not None
            and combined_service.replay_to_dram_fraction is not None
            and value == "moe_l2_residency_order_unavailable"))
    return dataclasses.replace(
        inp,
        memory_paths=tuple(paths),
        limitations=limitations,
    )


def dsv4_marlin_moe_source_work(kernel: KernelSpec):
    """Conditional EP-off W4A16 source math, typed uses and workspace views.

    This describes the retained modular Marlin program, not a loaded backend.
    Useful matrix FLOPs exclude padded rows; the source projection can also
    carry a separately named padded Tensor demand. Expert-dependent weight
    reads stay a coefficient rather than an invented expert assignment.
    """
    mathematical = ("mathematical_program" in kernel.attributes
        and kernel.attributes["mathematical_program"] == "dsv4_mxfp4_w4a16_swiglu_v1")
    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "routed_experts"
            or kernel.dtype != "bf16" or (not mathematical and (kernel.launch is None
                or kernel.launch.algorithm != "vllm_marlin_moe_mxfp4_aggregate"))
            or kernel.attributes["routed_weight_dtype"] != "mxfp4"
            or kernel.attributes["weight_group_size"] != 32
            or kernel.attributes["expert_parallel_size"] != 1
            or kernel.attributes["swiglu_limit"] is None):
        raise ValueError("V4 Marlin source work requires clamped BF16/MXFP4 W4A16 with EP off")
    tokens, topk, hidden, intermediate = (kernel.shape[name] for name in
                                         ("tokens", "topk", "hidden", "intermediate"))
    if hidden % 64 or intermediate % 32:
        raise ValueError("V4 Marlin checkpoint-repack source requires N%64=0 and K%32=0 for both matrices")
    rows = tokens * topk
    if sum(kernel.attributes["local_tokens_per_expert"]) != rows:
        raise ValueError("V4 Marlin EP-off source rows must conserve tokens*topk")
    activation = rows * intermediate
    activation_source = clamped_silu_source_arithmetic(activation, kernel.dtype)
    common_elements = max(rows * max(intermediate, hidden), tokens * hidden)
    second_elements = rows * max(2 * intermediate, hidden)
    second_offset = -(-(2 * common_elements) // 256) * 256
    requested_workspace_bytes = second_offset + -(-(2 * second_elements) // 256) * 256

    def use(role, dtype, elements, width, access, stage):
        return {"role": role, "dtype": dtype, "elements": elements, "bytes": elements * width,
                "access": access, "stage": stage}

    work = {
        "source_condition": "CUDA modular Marlin W4A16, EP-off, no LoRA, SILU with clamp; input_dtype=None/BF16, apply_router_weight_on_input=False; declared valid INT32 topk_ids, no token mask",
        "routed_rows": rows,
        "mathematical_primitives": {
            "useful_w13_tensor_flops": 4 * rows * hidden * intermediate,
            "useful_w2_tensor_flops": 2 * rows * hidden * intermediate,
            "activation_fp32_multiply": 2 * activation, "activation_fp32_add": 2 * activation,
            "activation_fp32_divide": activation, "activation_fp32_exp": activation,
            "activation_sign_negate": 2 * activation, "activation_fp32_minmax": 3 * activation,
            "activation_intermediate_bf16_rounding_values": activation,
            "router_weight_bf16_multiply": rows * hidden,
            "minimal_topk_fp32_add": tokens * hidden * (topk - 1),
            "source_topk_fp32_add": rows * hidden},
        "activation_resource_groups": activation_source["baseline_demand"],
        "activation_negation_programs": activation_source["source_negation_programs"],
        "activation_unpriced_source_actions": activation_source["unpriced_source_actions"],
        "router_weight_source_program": {
            "operation": "multiply", "input_dtype": "bf16", "weight_input_dtype": "fp32",
            "weight_converted_to": "bf16", "output_dtype": "bf16",
            "mathematical_flops": rows * hidden, "source_packed_lanes": 2,
            "mapped_resource_role": None, "native_issue_service_priced": False,
            "scope": "typed BF16 mathematical multiply; conversion, rounding and packed issue service not priced by INT32 or inferred FP32 throughput"},
        "dequant_source_program": {
            "condition": "BF16/MXFP4 E2M1 with E8M0 group32 scales; dequant_skip_flop=False; no zero point",
            "host_scale_preparation": "checkpoint E8M0 -> params_dtype BF16 -> Marlin permutation -> E8M0",
            "host_scale_group4_permutation": [0, 2, 1, 3], "host_preparation_is_per_forward_work": False,
            "packed_fragment_prepare": "b_quant_1=q; b_quant_0=q<<8",
            "dequant_call_nibble_offsets": [8, 24, 12, 28],
            "weight_bit_placement": "bf16_bits = ((nibble & 8) << 12) | ((nibble & 7) << 6)",
            "weight_bias_multiply": {"dtype": "bf16", "packed_lanes": 2, "bias_power_of_two": 126},
            "group_scale_decode": {"storage_dtype": "e8m0", "output_dtype": "bf16", "byte_to_bits": "scale_byte << 7",
                "source_word_byte_order": [0, 2, 1, 3],
                "exceptional_bytes": [
                    {"byte": 0, "source_value": "zero", "generic_E8M0_value": "2^-127"},
                    {"byte": 255, "source_value": "positive_infinity", "generic_E8M0_value": "NaN"}]},
            "group_scale_multiply": {"dtype": "bf16", "packed_lanes": 2},
            "ordered_value_program": ["integer bit placement", "BF16 bias multiply", "BF16 group-scale multiply", "BF16 MMA operand"],
            "per_weight_value_bf16_multiply_sites": 2, "scale_decode_values_per_unique_group_scale": 8,
            "global_dequant_tensor_materializations": [], "fragment_storage": "thread-local FragB/FragS values",
            "generic_MXFP4_numerical_equivalence_asserted": False,
            "checkpoint_values_or_BF16_intrinsic_lowering_bound": False,
            "mapped_resource_role": None, "native_issue_service_priced": False},
        "operands": (
            use("hidden_row_uses", "bf16", rows * hidden, 2, "read", "w13"),
            use("gate_up", "bf16", 2 * activation, 2, "write", "w13"),
            use("gate_up", "bf16", 2 * activation, 2, "read", "activation"),
            use("activated", "bf16", activation, 2, "write", "activation"),
            use("activated", "bf16", activation, 2, "read", "w2"),
            use("selected_weights", "fp32", rows, 4, "read", "w2"),
            use("weighted_expert_output", "bf16", rows * hidden, 2, "write", "w2"),
            use("weighted_expert_output", "bf16", rows * hidden, 2, "read", "reduce"),
            use("reduced_output", "bf16", tokens * hidden, 2, "write", "reduce")),
        "weight_payload_per_active_expert": {
            "packed_e2m1_dtype": "int32", "packed_e2m1_bytes": 3 * hidden * intermediate // 2,
            "group_size": 32, "scale_dtype": "fp8_e8m0", "scale_element_bytes": 1,
            "scale_elements": 2 * intermediate * (hidden // 32) + hidden * (intermediate // 32)},
        "checkpoint_repack_source_program": {
            "condition": "BF16 activations, E2M1 packed checkpoint rows, group32; N%64=0,K%32=0; contiguous final checkpoint dimension",
            "checkpoint_weight_dtype": "uint8", "checkpoint_scale_storage_dtype": "uint8",
            "prepared_weight_dtype": "int32", "prepared_scale_dtype": "fp8_e8m0",
            "values_per_prepared_word": 8, "weight_value_bits": 4,
            "host_weight_view": "weight[expert].view(int32).T.contiguous()",
            "repack_tile_shape_N_K": [64, 16], "prepared_words_per_tile": 128,
            "word_offset": "(tile_k*(N//64)+tile_n)*128+lane*4+warp",
            "source_n": "tile_n*64+warp*16+lane//4+((j//2)%2)*8",
            "source_k": "tile_k*16+2*(lane%4)+j//4+(j%2)*8",
            "coordinate_domain": "warp in[0,4),lane in[0,32),word nibble j in[0,8)",
            "scale_reorder_l": "(p//4)*4+[0,2,1,3][p%4]",
            "scale_transposed_index_grouped64": "(l//64)*64+(l%64)//8+8*(l%8)",
            "scale_transposed_index_single32": "(l//32)*32+2*((l%32)//8)+(l%2)+8*((l%8)//2)",
            "scale_checkpoint_coordinates": "n=t%N,group_k=t//N",
            "matrices": [{"stage": stage, "N": n, "K": k,
                "checkpoint_weight_shape": [kernel.shape["experts"], n, k // 2],
                "host_qweight_shape": [k // 8, n],
                "prepared_weight_shape": [kernel.shape["experts"], k // 16, 2 * n],
                "source_prepared_weight_tensor_bytes": kernel.shape["experts"] * n * k // 2,
                "checkpoint_scale_shape": [kernel.shape["experts"], n, k // 32],
                "prepared_scale_shape": [kernel.shape["experts"], k // 32, n],
                "scale_base_permutation": "grouped64" if k > 32 else "single32"}
                for stage, n, k in (("w13", 2 * intermediate, hidden), ("w2_with_router_weight", hidden, intermediate))],
            "expert_views_share_final_tensor": True, "expert_regions_disjoint": True,
            "final_weight_aliases_checkpoint_or_per_expert_result": False,
            "preparation_is_per_forward_work": False, "added_forward_payload_bytes": 0,
            "MMA_fragment_reader_or_actual_checkpoint_values_bound": False},
        "weight_reader_source_program": {
            "condition": "BF16/E2M1/E8M0 group32 source, contiguous prepared parameters and valid expert/slice indices",
            "copy_unit_bytes": 16, "packed_weight_tensor_dtype": "int32", "scale_tensor_dtype": "fp8_e8m0",
            "configurations_TK_TN_threads": [[128, 128, 256], [64, 256, 256], [64, 128, 128], [128, 64, 128]],
            "iterations": "I=TN*TK//(32*threads)",
            "global_B_int4_index": "expert*N*K//32+(slice_row*TK//16+stage_row)*(N//2)+slice_col*(TN//2)+stage_col",
            "shared_B_int4_read": "(TN//2)*q+tid+(tid//(TN//2))*((TN//2)*(I-1))",
            "warp_n": "(tid//32)%(TN//64)", "warp_k": "(tid//32)//(TN//64)",
            "subtile_n": "slice_col*TN+warp_n*64+j*16", "subtile_k": "slice_row*TK+16*(warp_k*I+q)",
            "regular_weight_B_n": "subtile_n+8*half+lane//4",
            "regular_weight_B_k": "subtile_k+2*(lane%4)+(v%2)+8*(v//2)",
            "M8_weight_A_n": "subtile_n+lane//4+8*((v//2)%2)",
            "M8_weight_A_k": "subtile_k+2*(lane%4)+(v%2)+8*(v//4)",
            "M8_weight_A_word_order": [["frag_b0", 0], ["frag_b1", 0], ["frag_b0", 1], ["frag_b1", 1]],
            "MMA_opcode": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
            "weight_operand_dtype": "bf16", "accumulator_dtype": "fp32",
            "operand_layout_source": "CUTLASS SM80_16x8x16_F32BF16BF16F32_TN A/B traits; same source PTX opcode",
            "packed_read_bytes_per_matrix_M_block": "N*K//2", "scale_read_bytes_per_matrix_M_block": "N*K//32",
            "prefetch_count": "min(R,S-1)+max(R-S+1,0)=R for assigned K tiles R>=0 and stages S>=2",
            "selected_configuration_or_GPU_order_bound": False,
            "activation_ldsm_output_or_numerical_MMA_bound": False,
            "cache_or_physical_DRAM_transactions_bound": False},
        "semantic_fields": {
            "activation_result_rounding_dtype": "bf16", "rounding_before_up_multiply": True,
            "alpha": 1.0, "beta": 0.0, "clamp_limit": kernel.attributes["swiglu_limit"],
            "router_weight_applied_on_input": False, "router_weight_multiply_dtype": "bf16",
            "router_weight_converted_from_fp32_before_multiply": True,
            "w2_result_rounding_before_router_weight": "bf16",
            "topk_accumulator_dtype": "fp32", "topk_accumulator_initial_value": 0.0,
            "topk_source_add_includes_zero_seed": True,
            "reduced_output_dtype": "bf16",
            "activation_intermediate_is_global_allocation": False},
        "source_views": (
            {"role": "gate_up", "base": "intermediate_cache13", "shape": (rows, 2 * intermediate), "new_allocation": False},
            {"role": "weighted_expert_output", "base": "intermediate_cache13", "shape": (rows, hidden), "new_allocation": False},
            {"role": "activated", "base": "intermediate_cache2", "shape": (rows, intermediate), "new_allocation": False}),
        "modular_buffer_program": {
            "condition": "FusedMoEKernelModularImpl.apply; Standard contiguous 2D BF16 input; EP-off/no LoRA; source output alias matches shape/dtype/device",
            "workspace_dtype": "bf16", "workspace_manager_backing_dtype": "uint8",
            "simultaneous_requests_share_backing": True, "request_alignment_bytes": 256,
            "requests": (
                {"role": "common_workspace", "offset_bytes": 0, "elements": common_elements, "bytes": 2 * common_elements},
                {"role": "workspace2", "offset_bytes": second_offset, "elements": second_elements, "bytes": 2 * second_elements}),
            "requested_workspace_bytes": requested_workspace_bytes,
            "requested_bytes_are_peak_capacity_or_traffic": False,
            "view_regions": (
                {"role": "activated", "offset_bytes": 0, "bytes": 2 * rows * intermediate,
                 "shape": (rows, intermediate), "strides_elements": (intermediate, 1)},
                {"role": "gate_up", "offset_bytes": second_offset, "bytes": 4 * rows * intermediate,
                 "shape": (rows, 2 * intermediate), "strides_elements": (2 * intermediate, 1)},
                {"role": "weighted_expert_output", "offset_bytes": second_offset, "bytes": 2 * rows * hidden,
                 "shape": (rows, hidden), "strides_elements": (hidden, 1)},
                {"role": "provisional_fused_out", "offset_bytes": 0, "bytes": 2 * tokens * hidden,
                 "shape": (tokens, hidden), "strides_elements": (hidden, 1)}),
            "resize_flatten_slice_view_allocates": False,
            "gate_up_and_weighted_expert_output_overlap": True,
            "gate_up_read_finishes_before_weighted_output_write": True,
            "workspace13_used_as_intermediate_cache2": True,
            "workspace2_used_as_intermediate_cache13": True,
            "provisional_fused_out_replaced_before_write": True,
            "output": {"dtype": "bf16", "shape": (tokens, hidden), "fresh_empty_like_hidden_states": True,
                       "aliases_workspace_or_input": False, "source_finalizer_receives_same_output_object": True},
            "empty_initialization_write_bytes": 0, "finalize_self_copy_bytes": 0,
            "actual_workspace_lane_capacity_addresses_or_GPU_order_bound": False,
            "scope": "source buffer requests/views and serial overwrite order; routing buffers, FP32 GEMM scratch, dequant temporaries, concurrent lane ownership and runtime remain separate"},
        "source_stage_order": ("w13", "activation", "w2_with_router_weight", "topk_reduce"),
        "declared_routing_is_checkpoint_assignment": False,
        "actual_provider_bound": False, "actual_workspace_capacity_bound": False,
        "scope": "useful math and per-stage source payload uses; expert assignment, GEMM padding, routing alignment, FP32 GEMM scratch and library instructions remain separate",
    }
    # These are logical stage programs, not four asserted GPU launches. The
    # wrapper materializes each producer before the next consumer runs.
    stages = []
    for stage, tensor, fma, alu, sfu, packed, scales in (
            ("w13", 4 * rows * hidden * intermediate, 0, 0, 0,
             hidden * intermediate, 2 * intermediate * (hidden // 32)),
            ("activation", 0, activation_source["baseline_demand"]["compute_fma"],
             0, activation_source["baseline_demand"]["compute_sfu"], 0, 0),
            ("w2_with_router_weight", 2 * rows * hidden * intermediate, 0, 0, 0,
             hidden * intermediate // 2, hidden * (intermediate // 32)),
            # All three no-mask source forms accumulate from FP32 zero.
            # Count each source +=, including the first term. This is not
            # an assertion about compiler seed elimination or instructions.
            ("topk_reduce", 0, rows * hidden, 0, 0, 0, 0)):
        operand_stage = {"w2_with_router_weight": "w2", "topk_reduce": "reduce"}
        role_stage = operand_stage[stage] if stage in operand_stage else stage
        stages.append({"stage": stage,
            "demand": {"compute_tensor": tensor, "compute_vector": 0,
                "compute_reduction": 0, "compute_fma": fma, "compute_alu": alu, "compute_sfu": sfu},
            "unpriced_typed_math": ([work["router_weight_source_program"]]
                if stage == "w2_with_router_weight" else []),
            "nonweight_read_write_bytes": {access: sum(operand["bytes"] for operand in work["operands"]
                if operand["stage"] == role_stage and operand["access"] == access)
                for access in ("read", "write")},
            "weight_payload_per_active_expert_bytes": {"packed_e2m1": packed, "fp8_e8m0_scales": scales}})
    if mathematical:
        # Keep the exact existing useful W4A16/activation/reduction ledger,
        # without manufacturing Marlin padding, CU count, locks or scratch.
        # This is a conditional source program, not a BW1100 backend claim.
        active = sum(count > 0 for count in kernel.attributes["local_tokens_per_expert"])
        for stage in stages:
            payload = dict(stage["nonweight_read_write_bytes"])
            payload["read"] += active * sum(stage["weight_payload_per_active_expert_bytes"].values())
            stage["read_write_bytes"] = payload
        work.update(stages=tuple(stages), active_experts=active, physical_launch_bound=False,
            normal_capacity_accepted=False, complete_source_service=False,
            source_condition="declared TP8 BF16/MXFP4 W4A16 mathematical projection of the retained source; no actual BW1100 Marlin/provider binding")
        return work
    experts = kernel.shape["experts"]
    # Host source chooses this tile before the alignment custom op. This
    # derivation uses only its source arguments, not the histogram or duration.
    block_m = 64
    for candidate in (8, 16, 32, 48, 64):
        if tokens * topk / experts / candidate < 0.9:
            block_m = candidate
            break
    capacity = rows + experts * (block_m - 1)
    if rows < experts:
        capacity = min(rows * block_m, capacity)
    blocks = (capacity + block_m - 1) // block_m
    small = rows < 1024 and experts <= 64
    if ((experts + 31) // 32) * 32 >= 1024:
        raise ValueError("Marlin alignment source requires padded_num_experts < 1024")
    if kernel.launch.tiling.tile_shape["m"] != block_m:
        raise ValueError("Marlin alignment M tile disagrees with the host source branch")
    alignment = {"block_size_m": block_m, "sorted_ids_capacity": capacity,
        "expert_ids_capacity": blocks, "post_pad_elements": 1,
        "cumsum_elements": 0 if small else experts + 1,
        "index_dtype": "int32", "sorted_sentinel": rows, "inactive_expert_sentinel": -1,
        "all_topk_ids_valid": True, "expert_map": None, "pad_sorted_ids": False,
        "program": "small_batch_expert" if small else "count_fill_then_sort",
        "source_sort_global_atomic_adds": 0 if small else rows,
        "sorted_initialization_and_overwrite_are_distinct_writes": True,
        "source_allocations_are_actual_live_capacity": False,
        "scope": "logical source payload, including unique prefix-boundary reads and per-route atomic read/modify/write; not compiled loads, atomic service time or DRAM traffic"}
    work["routing_alignment"] = alignment
    prepare = [{"stage": "routing_count_fill", "read": 4 * rows + 4 * (experts + 1),
                "write": 4 * (capacity + blocks + 1 + experts + 1)},
               {"stage": "routing_sort", "read": 8 * rows, "write": 8 * rows}]
    if small:
        prepare = [{"stage": "routing_align_small", "read": 8 * rows,
                    "write": 4 * (capacity + blocks + 1 + rows)}]
    prefix = [{"stage": part["stage"], "demand": {role: 0 for role in stages[0]["demand"]},
        "unpriced_typed_math": [],
        "nonweight_read_write_bytes": {access: part[access] for access in ("read", "write")},
        "weight_payload_per_active_expert_bytes": {"packed_e2m1": 0, "fp8_e8m0_scales": 0}}
        for part in prepare]
    work["workspace_zero"] = {"condition": "modular Marlin no-LoRA default workspace=None; existing=None",
        "dtype": "int32", "elements_per_sm": 4, "bytes_per_sm": 16,
        "initial_value": 0, "initialization_count_per_moe_call": 1,
        "shared_by": ("w13", "w2_with_router_weight"),
        "following_lock_accesses_bound": False, "actual_allocator_binding_verified": False}
    workspace = {"stage": "workspace_zero", "demand": {role: 0 for role in stages[0]["demand"]},
        "unpriced_typed_math": [],
        "nonweight_read_write_bytes": {"read": 0, "write": 0},
        "weight_payload_per_active_expert_bytes": {"packed_e2m1": 0, "fp8_e8m0_scales": 0}}
    stages = prefix + [workspace] + stages
    for stage in stages:
        stage["read_write_bytes_per_sm"] = {"read": 0, "write": 16 if stage["stage"] == "workspace_zero" else 0}
        matrix = stage["stage"] in ("w13", "w2_with_router_weight")
        stage["metadata_read_bytes_per_active_m_block"] = 4 * (block_m + 1) if matrix else 0
        stage["metadata_read_bytes_per_stage"] = 4 if matrix else 0
    work["source_stage_order"] = [stage["stage"] for stage in stages]
    work["roofline_stage_templates"] = stages
    work["gemm_metadata"] = {
        "index_dtype": "int32", "element_bytes": 4,
        "active_m_blocks_rule": "sum(ceil(count_e / block_size_m)) for declared valid expert histogram",
        "sorted_ids_elements_per_active_m_block": block_m,
        "expert_ids_elements_per_active_m_block": 1, "post_pad_elements_per_gemm": 1,
        "sorted_ids_backing_elements": capacity, "expert_ids_backing_elements": blocks,
        "sorted_ids_producer": "routing_align_small" if small else "routing_sort",
        "expert_ids_and_post_pad_producer": "routing_align_small" if small else "routing_count_fill",
        "consumers": ("w13", "w2_with_router_weight"),
        "padding_sentinel": rows, "allocation_tail_is_read": False,
        "input_row_mappings": (
            {"stage": "w13", "top_k": topk, "input_rows": tokens, "output_rows": rows,
             "input_row": "sorted_route_id // top_k", "output_row": "sorted_route_id"},
            {"stage": "w2_with_router_weight", "top_k": 1, "input_rows": rows, "output_rows": rows,
             "input_row": "sorted_route_id // top_k", "output_row": "sorted_route_id"}),
        "valid_route_id_domain": (0, rows),
        "valid_rows_count_expression": "count(sorted_route_id < prob_m * top_k) per active m block",
        "activation_read_guard": "row < block_num_valid_tokens", "output_scatter_guard": "row < block_num_valid_tokens",
        "within_expert_order_binding_asserted": False,
        "scope": "unique per-GEMM active-prefix input payload and source row mapping under declared routing; not CTA/split replay, DRAM traffic or checkpoint expert IDs"}
    # ops.cu requests a fresh FP32 scratch tensor for each GEMM. Its size is
    # independent of selected thread_n/thread_k, but traffic is not: the
    # kernel only reduces through C_tmp for multi-block slices.
    work["fp32_scratch_templates"] = [
        {"stage": stage, "dtype": "fp32", "element_bytes": 4,
         "allocation_condition": "use_fp32_reduce && !use_atomic_add",
         "use_fp32_reduce": True, "use_atomic_add": False,
         "sorted_capacity_term_elements": size_n * capacity,
         "per_sm_cap_elements": 4 * block_m * 256,
         "max_thread_n": 256, "m8_allocation_factor": 2 if block_m == 8 else 1,
         "allocation_rule": "min(sorted_capacity_term_elements, sm_count * per_sm_cap_elements) * m8_allocation_factor",
         "initialization": "empty, not zero-filled", "producer_dtype": "fp32", "consumer_dtype": "fp32",
         "access_condition": "slice_count > 1 && !use_atomic_add",
         "reads_previous_partial_if": "!first", "writes_partial_if": "!last",
         "read_write_bytes": None, "additional_fp32_additions": None,
         # BF16 uses thread_n_blocks/4 active warps and 8*M16 fragment
         # vectors per lane. A vector carries four FP32 values. The m8
         # specialization masks odd fragment indices; other M tiles use
         # the exact fragment row predicate, which is not live-row bytes.
         "fp32_values_per_n_column_by_valid_rows": [None] + [
             8 if block_m == 8 else 2 * sum(min(max(valid - 16 * group, 0), 8)
                 for group in range((block_m + 15) // 16)) for valid in range(1, block_m + 1)],
         "partial_access_rule": "V=thread_n*values_per_n_column[valid_rows]; read=4*V if !first else 0; write=4*V if !last else 0; FP32_add=V if !first else 0",
         "thread_n_source_candidates": (64, 128) if block_m <= 16 else (64, 128, 256),
         "per_slice_active_lane_tail_guard_bound": True,
         "selected_slice_census_bound": False,
         "lifetime": "local tensor of one moe_wna16_marlin_gemm call; W13 and W2 source uses are serial",
         "physical_allocation_or_cross_stage_reuse_bound": False,
         "scope": "requested source extent and conditional typed reduction; extent is not bytes touched or normal capacity"}
        for stage, size_n in (("w13", 2 * intermediate), ("w2_with_router_weight", hidden))]
    routing_objects = [("sorted_token_ids", capacity, "alignment through W2"),
                       ("expert_ids", blocks, "alignment through W2"),
                       ("num_tokens_post_padded", 1, "alignment through W2")]
    if not small:
        routing_objects.append(("cumsum", experts + 1, "local to alignment custom-op enqueue"))
    zero_objects = [("alignment", "expert_map", "int32")]
    for stage in ("w13", "w2_with_router_weight"):
        zero_objects.extend((stage, role, dtype) for role, dtype in (
            ("a_scales", "fp32"), ("global_scale", "fp32"), ("b_bias", "bf16"), ("b_zeros", "bf16")))
    work["tensor_object_program"] = {
        "condition": "successful CUDA Standard no-DP/EP modular Marlin W4A16; no LoRA, bias, zero-point or input quantization; valid routes",
        "prepare": {"hidden_states_alias": True,
                    "activation_scale": "None or borrowed prepare scale, unused by Marlin W4A16",
                    "topk_weights_alias": True, "topk_ids_alias": True,
                    "expert_tokens_meta": None, "expert_map": None,
                    "apply_router_weight_on_input": False, "global_input_copy_bytes": 0},
        "routing_buffers": [
            {"role": role, "dtype": "int32", "shape": [elements], "fresh": True,
             "initialization": "empty; source alignment kernels initialize consumed regions",
             "lifetime": lifetime}
            for role, elements, lifetime in routing_objects],
        "zero_extent_objects": [
            {"stage": stage, "role": role, "dtype": dtype, "elements": 0,
             "initialization_write_bytes": 0, "payload_access_bytes": 0,
             "lifetime": "local to the custom-op enqueue"}
            for stage, role, dtype in zero_objects],
        "storage_program_bindings": ["modular_buffer_program", "workspace_zero", "fp32_scratch_templates"],
        "parameter_program_bindings": ["checkpoint_repack_source_program", "weight_payload_per_active_expert"],
        "GEMM_returns_supplied_C": ["w13", "w2_with_router_weight"],
        "activation_input_view": {"base": "gate_up", "shape": [rows, 2 * intermediate],
            "strides_elements": [2 * intermediate, 1], "offset_elements": 0, "allocates": False},
        "reduction_input_view": {"base": "weighted_expert_output", "shape": [tokens, topk, hidden],
            "strides_elements": [topk * hidden, hidden, 1], "offset_elements": 0, "allocates": False},
        "activation_input_view_allocates": False, "reduction_input_view_allocates": False,
        "output": {"dtype": "bf16", "shape": [tokens, hidden], "fresh_empty_like_hidden_states": True,
                   "aliases_input_parameters_routes_or_workspace": False,
                   "written_by": "topk_reduce", "finalizer_returns_same_object": True},
        "buffer_lifetimes": [
            {"role": "gate_up", "producer": "w13", "last_reader": "activation"},
            {"role": "activated", "producer": "activation", "last_reader": "w2_with_router_weight"},
            {"role": "weighted_expert_output", "producer": "w2_with_router_weight", "last_reader": "topk_reduce"}],
        "FP32_scratch": "fresh per-GEMM local tensor, serial W13/W2 source lifetimes; conditional accesses stay parameterized",
        "register_dequant_global_tensors": [], "finalize_copy_bytes": 0,
        "arena_base_capacity_lane_and_GPU_fences_bound": False,
        "scope": "source tensor identity/views/temporaries/lifetimes; not selected slice traffic, actual allocator or provider"}
    return work


def _dsv4_marlin_sm_count(kernel: KernelSpec, hardware):
    if kernel.hardware_id != hardware.hardware_id:
        raise ValueError("Marlin scratch hardware identity disagrees with KernelSpec")
    sm_count = hardware.num_compute_units
    if isinstance(sm_count, bool) or sm_count <= 0 or int(sm_count) != sm_count:
        raise ValueError("Marlin scratch requires a positive integral hardware SM count")
    return int(sm_count)


@runtime_lru_cache(maxsize=256)
def _marlin_source_extra_partials(m_blocks, size_n, size_k, thread_k, thread_n, grid_blocks):
    """Two-tile Stream-K control only, not a selected native launch."""
    n_tiles, k_tiles = size_n // thread_n, size_k // thread_k
    total = m_blocks * n_tiles
    tail = total
    if total > grid_blocks:
        tail = total % grid_blocks
        if 3 * tail <= grid_blocks:
            tail += grid_blocks
    iterations = (k_tiles * tail + grid_blocks - 1) // grid_blocks
    extra = [0] * m_blocks
    for column in range(tail):
        begin, end = column * k_tiles, (column + 1) * k_tiles
        fragments = (end - 1) // iterations - begin // iterations + 1
        extra[(total - tail + column) // n_tiles] += fragments - 1
    return tuple(extra)


def dsv4_marlin_scratch_census(valid_rows, *, block_m, size_n, size_k, thread_k, thread_n, grid_blocks):
    """FP32 scratch accesses conditional on explicit source configuration.

    The V4 BF16/MXFP4 group32 path has group_blocks=2, smaller than each
    candidate thread_k_blocks, so its stripe length has no group rounding.
    A column with R partials has R-1 readers, R-1 writers and R-1 additions
    of the same guarded FP32 fragment. DP columns have R=1 and touch none.
    """
    integers = (block_m, size_n, size_k, thread_k, thread_n, grid_blocks)
    if any(type(value) is not int or value <= 0 for value in integers):
        raise ValueError("Marlin conditional census requires positive integer geometry")
    if (block_m not in (8, 16, 32, 48, 64) or thread_k not in (64, 128)
            or thread_n not in ((64, 128) if block_m <= 16 else (64, 128, 256))
            or size_n % thread_n or size_k % thread_k):
        raise ValueError("Marlin conditional census requires source-admissible V4 group32 tiles")
    if (not isinstance(valid_rows, tuple) or not valid_rows
            or any(type(rows) is not int or not 1 <= rows <= block_m for rows in valid_rows)):
        raise ValueError("Marlin conditional census requires nonempty valid M-block row counts")
    total = len(valid_rows) * (size_n // thread_n)
    if (any(value >= 2**31 for value in integers) or total >= 2**31
            or (size_k // thread_k) * min(total, 2 * grid_blocks) + grid_blocks >= 2**31):
        raise ValueError("Marlin conditional census exceeds source INT32 control domain")
    extra = _marlin_source_extra_partials(len(valid_rows), size_n, size_k, thread_k, thread_n, grid_blocks)
    values = 0
    for rows, merges in zip(valid_rows, extra, strict=True):
        per_column = (8 if block_m == 8 else
                      2 * sum(min(max(rows - 16 * group, 0), 8) for group in range((block_m + 15) // 16)))
        values += merges * thread_n * per_column
    return {"thread_k": thread_k, "thread_n": thread_n, "grid_blocks": grid_blocks,
        "read_bytes": 4 * values, "write_bytes": 4 * values, "fp32_additions": values,
        "extra_partial_count": sum(extra)}


def _dsv4_marlin_scratch_bounds(kernel, sm_count, block_m):
    histogram = kernel.attributes["local_tokens_per_expert"]
    valid_rows = tuple(rows for count in histogram for rows in
        ([block_m] * (count // block_m) + ([count % block_m] if count % block_m else [])))
    if kernel.launch.parameters["padded_rows"] != len(valid_rows) * block_m:
        raise ValueError("Marlin scratch census padding disagrees with the declared histogram")
    hidden, intermediate, tokens, topk = (kernel.shape[name] for name in ("hidden", "intermediate", "tokens", "topk"))
    candidates = ((128, 128), (64, 128), (128, 64)) if block_m <= 16 else ((64, 256), (64, 128), (128, 64))
    stages = []
    for name, n, k in (("w13", 2 * intermediate, hidden), ("w2_with_router_weight", hidden, intermediate)):
        configurations = []
        for tk, tn in candidates:
            if k % tk or n % tn:
                continue
            limit = min(4 if block_m <= 16 else 2, max((n // tn) * tokens * topk * 4 // sm_count, 1))
            for blocks_per_sm in range(1, limit + 1):
                configuration = dsv4_marlin_scratch_census(valid_rows,
                    block_m=block_m, size_n=n, size_k=k, thread_k=tk, thread_n=tn,
                    grid_blocks=sm_count * blocks_per_sm)
                configuration["tile_source_work"] = dsv4_marlin_tile_source_work(valid_rows,
                    block_m=block_m, size_n=n, size_k=k, thread_k=tk, thread_n=tn,
                    grid_blocks=sm_count * blocks_per_sm)
                configuration["metadata_source_work"] = dsv4_marlin_metadata_source_work(valid_rows,
                    block_m=block_m, size_n=n, size_k=k, thread_k=tk, thread_n=tn,
                    grid_blocks=sm_count * blocks_per_sm, mul_topk_weights=name == "w2_with_router_weight")
                configurations.append(configuration)
        if not configurations:
            raise ValueError("Marlin source configuration domain is empty")
        stages.append({"stage": name, "configuration_superset": configurations,
            "bounds": {field: {"lower": min(item[field] for item in configurations),
                               "upper": max(item[field] for item in configurations)}
                       for field in ("read_bytes", "write_bytes", "fp32_additions")}})
    return {"stages": stages, "declared_routing_is_checkpoint_assignment": False,
        "selected_source_configuration_bound": False, "bounds_used_for_prediction": False,
        "source_domain": "BF16/MXFP4 group32, valid histogram; auto candidate tiles and possible blocks_per_sm; kernel availability/numRegs/shared-memory selection remain unbound",
        "scope": "conditional source access census and enclosing bounds, not point traffic, duration, peak allocation or native provider acceptance"}


def dsv4_marlin_tile_source_work(valid_rows, *, block_m, size_n, size_k, thread_k, thread_n, grid_blocks):
    """Exact source counts conditional on one admissible auto configuration.

    A is fetched again for each N tile, with a live-row predicate; a K
    partition does not duplicate the K interval. The shared reduction has
    no live-row predicate, so even M tails reduce the padded accumulator.
    These counts never select a native configuration or alter point prices.
    """
    census = dsv4_marlin_scratch_census(valid_rows, block_m=block_m,
        size_n=size_n, size_k=size_k, thread_k=thread_k, thread_n=thread_n, grid_blocks=grid_blocks)
    candidates = {(128, 128): 256, (64, 128): 128, (128, 64): 128, (64, 256): 256}
    if (thread_k, thread_n) not in candidates or (block_m <= 16 and thread_n == 256):
        raise ValueError("Marlin tile work requires a source auto-config candidate")
    threads = candidates[thread_k, thread_n]
    reduction_groups = threads // (thread_n // 2)
    slices = len(valid_rows) * (size_n // thread_n) + census["extra_partial_count"]
    return {"threads": threads, "source_matrix_slices": slices,
        "activation_global_read_bytes": 2 * sum(valid_rows) * size_k * (size_n // thread_n),
        "threadblock_fp32_additions": slices * (reduction_groups - 1) * block_m * thread_n,
        "activation_read_is_physical_DRAM": False,
        "shared_reduction_counts_padded_rows": True,
        "counts_used_for_point_prediction": False}


def dsv4_marlin_metadata_source_work(valid_rows, *, block_m, size_n, size_k, thread_k, thread_n,
                                     grid_blocks, mul_topk_weights):
    """Header replay in the BF16/group32 scalar controller, per configuration.

    DP reloads every M/N tile. Stream-K keeps one header until its CTA
    crosses an M boundary; an exhausted slice does not reload. Weight
    reads include sentinel slots redirected to weight[0]. Uniform logical
    values and source-lane evaluations are separate, neither is DRAM traffic.
    """
    if type(mul_topk_weights) is not bool:
        raise ValueError("Marlin metadata requires a boolean router-weight branch")
    tile = dsv4_marlin_tile_source_work(valid_rows, block_m=block_m, size_n=size_n, size_k=size_k,
        thread_k=thread_k, thread_n=thread_n, grid_blocks=grid_blocks)
    n_tiles, k_tiles = size_n // thread_n, size_k // thread_k
    total = len(valid_rows) * n_tiles
    tail = total
    if total > grid_blocks:
        remainder = total % grid_blocks
        tail = remainder + grid_blocks if 3 * remainder <= grid_blocks else remainder
    prefix = total - tail
    stripe = (tail * k_tiles + grid_blocks - 1) // grid_blocks
    updates = [min(max(prefix - m * n_tiles, 0), n_tiles) for m in range(len(valid_rows))]
    for cta in range(grid_blocks):
        start, stop = cta * stripe, min((cta + 1) * stripe, tail * k_tiles)
        if start >= stop:
            continue
        first = (prefix + start // k_tiles) // n_tiles
        last = (prefix + (stop - 1) // k_tiles) // n_tiles
        for m in range(first, last + 1):
            updates[m] += 1
    headers = sum(updates)
    slots = block_m * headers
    weighted_slots = slots if mul_topk_weights else 0
    return {"header_updates_by_m_block": updates, "header_updates": headers,
        "sorted_ids_source_global_read_bytes": 4 * slots,
        "topk_weights_source_global_read_bytes": 4 * weighted_slots,
        "padding_weight_zero_index_reads": sum((block_m - rows) * count for rows, count in
            zip(valid_rows, updates, strict=True)) if mul_topk_weights else 0,
        "sorted_id_validity_comparisons": slots, "weight_slot_validity_comparisons": weighted_slots,
        "route_to_activation_integer_division_sites": slots,
        "expert_id_uniform_role_read_bytes": 4 * headers,
        "expert_id_source_lane_read_bytes": 4 * tile["threads"] * headers,
        "post_pad_uniform_entry_read_bytes": 4 * grid_blocks,
        "post_pad_source_lane_entry_read_bytes": 4 * tile["threads"] * grid_blocks,
        "source_reads_are_physical_DRAM": False, "counts_used_for_point_prediction": False}


def dsv4_marlin_moe_scratch_work(kernel: KernelSpec, hardware):
    """Bind requested scratch extents to declared hardware SMs, not a provider.

    No allocation bytes enter the traffic demand: slice ownership/first/last
    and active row/lane counts are still required to derive scratch accesses.
    """
    sm_count = _dsv4_marlin_sm_count(kernel, hardware)
    source = dsv4_marlin_moe_source_work(kernel)
    templates = source["fp32_scratch_templates"]
    stages = []
    for item in templates:
        elements = min(item["sorted_capacity_term_elements"], sm_count * item["per_sm_cap_elements"]) * item["m8_allocation_factor"]
        stages.append({"stage": item["stage"], "requested_elements": elements,
                       "requested_bytes": elements * item["element_bytes"], "dtype": item["dtype"]})
    return {"sm_count": sm_count, "sm_count_source": "declared HWSpec, not runtime device observation",
        "stages": stages, "requested_bytes_counted_as_traffic": False,
        "conditional_access_bounds": _dsv4_marlin_scratch_bounds(
            kernel, sm_count, source["routing_alignment"]["block_size_m"]),
        "aggregate_peak_live_bytes": None, "actual_allocator_binding_verified": False,
        "scope": "per-GEMM source requests only; no touched-byte, live allocator capacity or provider acceptance"}


def dsv4_marlin_moe_roofline_work(kernel: KernelSpec, hardware):
    """Bind source tensor padding and live payload to the declared histogram.

    Alignment's source payload is explicit; integer, atomic and compiler
    services remain unpriced. M-padding follows the source's BF16 MMA sites;
    its M8 transpose consumes eight logical rows, not sixteen. Scratch and
    cache reloads remain separate. The histogram is not checkpoint evidence.
    """
    source = dsv4_marlin_moe_source_work(kernel)
    sm_count = _dsv4_marlin_sm_count(kernel, hardware)
    active_experts = sum(count > 0 for count in kernel.attributes["local_tokens_per_expert"])
    block_m = source["routing_alignment"]["block_size_m"]
    m_blocks = sum((count + block_m - 1) // block_m for count in kernel.attributes["local_tokens_per_expert"])
    if kernel.launch.parameters["padded_rows"] != m_blocks * block_m:
        raise ValueError("Marlin metadata padded rows disagree with the declared routing histogram")
    tensor_stages = []
    for name, n, k in (("w13", 2 * kernel.shape["intermediate"], kernel.shape["hidden"]),
                       ("w2_with_router_weight", kernel.shape["hidden"], kernel.shape["intermediate"])):
        useful = 2 * source["routed_rows"] * n * k
        padded = 2 * m_blocks * block_m * n * k
        tensor_stages.append({"stage": name, "useful_tensor_flops": useful,
            "source_tensor_flops": padded, "padding_tensor_flops": padded - useful})
    tensor_work = {"live_rows": source["routed_rows"], "source_padded_rows": m_blocks * block_m,
        "source_block_m": block_m, "M8_uses_transposed_operands_without_extra_row_padding": True,
        "stages": tensor_stages, "actual_routing_or_compiled_provider_bound": False}
    matrix_demands = {stage["stage"]: stage["source_tensor_flops"] for stage in tensor_stages}
    dequant_stages = []
    for name, n, k in (("w13", 2 * kernel.shape["intermediate"], kernel.shape["hidden"]),
                       ("w2_with_router_weight", kernel.shape["hidden"], kernel.shape["intermediate"])):
        values = m_blocks * n * k
        dequant_stages.append({"stage": name, "weight_decode_values": values,
            "bf16_bias_multiply_values": values, "bf16_group_scale_multiply_values": values,
            "source_scale_decode_values": values // 4,
            "unique_logical_group_scale_uses": values // 32})
    dequant_work = {"declared_M_blocks": m_blocks, "stages": dequant_stages,
        "counts_are_source_sites_not_compiler_instructions": True, "added_global_tensor_bytes": 0,
        "mapped_resource_role": None, "native_issue_service_priced": False,
        "compiler_spill_or_actual_traffic_bound": False}
    weight_loads = []
    for name, n, k in (("w13", 2 * kernel.shape["intermediate"], kernel.shape["hidden"]),
                       ("w2_with_router_weight", kernel.shape["hidden"], kernel.shape["intermediate"])):
        weight_loads.append({"stage": name,
            "distinct_packed_parameter_bytes": active_experts * n * k // 2,
            "source_global_packed_weight_read_bytes": m_blocks * n * k // 2,
            "distinct_group_scale_parameter_bytes": active_experts * n * k // 32,
            "source_global_scale_read_bytes": m_blocks * n * k // 32})
    weight_load_work = {"declared_M_blocks": m_blocks, "declared_active_experts": active_experts,
        "stages": weight_loads, "source_global_visits_used_for_prediction": True,
        "projection": "source-global weight/scales reads; cache reuse and physical transaction amplification unmodeled",
        "physical_traffic_or_cache_outcomes_bound": False}
    stages = []
    for template in source["roofline_stage_templates"]:
        payload = template["nonweight_read_write_bytes"]
        weight_bytes = m_blocks * sum(template["weight_payload_per_active_expert_bytes"].values())
        stage_demand = dict(template["demand"])
        if template["stage"] in matrix_demands:
            stage_demand["compute_tensor"] = matrix_demands[template["stage"]]
        stages.append({**template, "demand": stage_demand, "read_write_bytes": {
            "read": (payload["read"] + weight_bytes + sm_count * template["read_write_bytes_per_sm"]["read"]
                     + m_blocks * template["metadata_read_bytes_per_active_m_block"] + template["metadata_read_bytes_per_stage"]),
            "write": payload["write"] + sm_count * template["read_write_bytes_per_sm"]["write"]}})
    demand = {role: sum(stage["demand"][role] for stage in stages) for role in stages[0]["demand"]}
    read_write = {access: sum(stage["read_write_bytes"][access] for stage in stages) for access in ("read", "write")}
    return {"stages": stages, "demand": demand, "read_write_bytes": read_write,
        "logical_bytes": sum(read_write.values()), "declared_active_experts": active_experts,
        "declared_active_m_blocks": m_blocks, "tensor_work": tensor_work, "dequant_work": dequant_work,
        "weight_load_work": weight_load_work,
        "routing_is_checkpoint_assignment": False, "price_composition": "sum_serial_stage_rooflines",
        "remaining_source_services": ["routing_integer_atomic_and_gemm_metadata_replay_services", "gemm_fp32_scratch",
                                      "weight_dequantization_and_compiler_services",
                                      "bf16_router_weight_multiply_conversion_and_rounding_service"],
        "scope": "serial source M-padded tensor FLOPs and live scalar/role-use bytes under declared routing; not full MoE source/native acceptance or a hardware latency bound"}


def _routed_experts_model(
        kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    """Aggregate the pinned Marlin W4A16 two-GEMM MoE program.

    Packed weights, BF16 group scales, BF16 activations, SwiGLU materialization,
    and the final top-k reduction are counted directly from the typed shape and
    source-selected M block.  This path intentionally has no duration-derived
    correction or per-shape latency table.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            "routed-experts Marlin aggregate is modeled only on gpu_mshr "
            "GPU hardware")
    if (kernel.attributes["swiglu_limit"] is not None
            and kernel.attributes["expert_parallel_size"] == 1):
        work = dsv4_marlin_moe_roofline_work(kernel, hardware)
        total = work["logical_bytes"]
        demand = build_forward_demand(total, total, False, work["demand"])
        return build_aggregate_gpu_input(kernel, hardware, demand,
            write_l2_bytes=work["read_write_bytes"]["write"],
            write_dram_bytes=work["read_write_bytes"]["write"],
            limitations=("v4_marlin_M_padded_tensor_live_scalar_serial_roofline_only",
                "declared_routing_not_checkpoint_assignment",
                "marlin_integer_atomic_metadata_consumer_scratch_unbound",
                "marlin_dequantization_compiler_and_provider_services_unbound",
                "bf16_router_weight_math_counted_packed_multiply_and_conversion_service_unpriced")), total
    # Two pinned Marlin MoE programs: the Kimi INT4 W4A16 aggregate (BF16
    # group scales) and the DeepSeek-V4 MXFP4 aggregate (E2M1 values with one
    # UE8M0 byte per 32-element group, vLLM 0.30
    # fused_moe/experts/marlin_moe.py with scalar type float4_e2m1f).
    scale_bytes_per_group = {
        "vllm_marlin_moe_w4a16_aggregate": 2,
        "vllm_marlin_moe_mxfp4_aggregate": 1,
    }
    if kernel.launch.algorithm not in scale_bytes_per_group:
        raise ValueError("routed_experts requires the pinned Marlin launch")
    expected_weight_dtype = {
        "vllm_marlin_moe_w4a16_aggregate": "int4",
        "vllm_marlin_moe_mxfp4_aggregate": "mxfp4",
    }[kernel.launch.algorithm]
    if kernel.dtype != "bf16":
        raise ValueError("routed_experts requires BF16 activations")
    if kernel.attributes["routed_weight_dtype"] != expected_weight_dtype:
        raise ValueError(
            f"{kernel.launch.algorithm} requires "
            f"{expected_weight_dtype} packed weights")
    group_size = int(kernel.attributes["weight_group_size"])
    if group_size != 32:
        raise ValueError("routed_experts requires group size 32")

    from src.kernel.memory_fused_moe import _bound_work
    source = _bound_work(kernel)
    tokens, topk, hidden, intermediate = (source['geometry'] if source is not None else
        tuple(int(kernel.shape[name]) for name in ("tokens", "topk", "hidden", "intermediate")))
    _, local_counts = _routing(kernel)
    routed_rows = sum(local_counts)
    expert_parallel_size = int(kernel.attributes["expert_parallel_size"])
    if expert_parallel_size == 1 and routed_rows != tokens * topk:
        raise ValueError("routed_experts local rows must conserve tokens*topk")
    if expert_parallel_size > 1 and routed_rows > tokens * topk:
        raise ValueError(
            "routed_experts local rows cannot exceed global tokens*topk")
    active_experts = sum(count > 0 for count in local_counts)
    if active_experts <= 0:
        raise ValueError("routed_experts requires an active expert")
    tile_m, _, _ = _tile(kernel)
    padded_rows = sum(
        math.ceil(count / tile_m) * tile_m
        for count in local_counts if count > 0)
    launch_padded_rows = kernel.launch.parameters["padded_rows"]
    if (not launch_padded_rows.is_integer()
            or int(launch_padded_rows) != padded_rows):
        raise ValueError(
            "routed_experts launch padded_rows disagrees with routing")

    populated_blocks = padded_rows // tile_m
    if source is not None:
        traffic_source = source['routed_traffic']
        activation_read = float(traffic_source['activation_read'])
        output_write = float(traffic_source['output_write'])
        weight_bytes_per_expert = float(traffic_source['weight_per_expert'])
        weight_l2 = float(populated_blocks * weight_bytes_per_expert)
        weight_compulsory = float(traffic_source['weight_compulsory'])
    else:
        # Rank-local EP wrapper remains outside the current source domain.
        # Preserve its explicit supported aggregate until its producer and
        # remote-row workspace semantics are bound.
        activation_bytes = int(kernel.shape["dtype_bytes"])
        packed_weight_bytes_per_expert = 3.0 * hidden * intermediate * 4.0 / 8.0
        scale_bytes_per_expert = float(
            (2 * intermediate * (hidden // group_size) + hidden * (intermediate // group_size))
            * scale_bytes_per_group[kernel.launch.algorithm])
        weight_bytes_per_expert = packed_weight_bytes_per_expert + scale_bytes_per_expert
        weight_l2 = float(populated_blocks * weight_bytes_per_expert)
        weight_compulsory = float(active_experts * weight_bytes_per_expert)
        routing_bytes = float(routed_rows * (4 + 4))
        activation_read = float(routed_rows * (2 * hidden + 4 * intermediate + 2 * intermediate + 2 * hidden) + routing_bytes)
        output_write = float(routed_rows * (4 * intermediate + 2 * intermediate + 2 * hidden)
                             + tokens * hidden * activation_bytes)
    traffic = _GpuGroupedGemmTraffic(
        activation_l2_read_bytes=activation_read,
        weight_l2_read_bytes=weight_l2,
        routed_weight_l2_read_bytes=0.0,
        output_l2_write_bytes=output_write,
        activation_compulsory_read_bytes=activation_read,
        weight_compulsory_read_bytes=weight_compulsory,
        routed_weight_compulsory_read_bytes=0.0,
        activation_dram_read_bytes=activation_read,
        weight_dram_read_bytes=weight_l2,
        routed_weight_dram_read_bytes=0.0,
        output_dram_write_bytes=output_write,
    )
    l2_total = traffic.l2_total_bytes
    dram_bytes = traffic.dram_total_bytes
    validated_l2_miss_rate(dram_bytes, l2_total)

    tensor_flops = float(6 * padded_rows * hidden * intermediate)
    swiglu_elements = float(routed_rows * intermediate)
    reduction_elements = float(tokens * hidden * max(0, topk - 1))
    # marlin_moe.py uses apply_router_weight_on_input=False for this retained
    # source branch. W2 sets mul_topk_weights=True; marlin_template.h multiplies
    # each live BF16 output by its selected weight (__hmul2) before top-k sum.
    # Count scalar logical elements, not packed instructions or padded GEMM rows.
    router_scale_elements = float(routed_rows * hidden)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 0.0,
        "compute_alu": 3.0 * swiglu_elements + router_scale_elements + reduction_elements,
        "compute_sfu": swiglu_elements,
    }
    if kernel.attributes["swiglu_limit"] is not None:
        # vLLM mxfp4.py -> make_mxfp4_moe_quant_config ->
        # ApplyMoEActivationConfig: gate upper clamp and up two-sided clamp,
        # EP-off has no expert_map, so apply_moe_activation dispatches to
        # CUDA silu_and_mul_with_clamp(..., alpha=1,beta=0). The source
        # retains alpha multiply and beta add, rounds SiLU to BF16, then
        # multiplies up. These runtime-parameter operations are counted
        # before compiler folding. Other EP projections are outside this
        # source-scoped correction; GEMM padding never adds live activation.
        if expert_parallel_size == 1:
            activation_groups = dsv4_marlin_moe_source_work(kernel)["activation_resource_groups"]
            flops["compute_fma"] = float(activation_groups["compute_fma"])
            flops["compute_alu"] = float(activation_groups["compute_alu"]) + router_scale_elements + reduction_elements
            flops["compute_sfu"] = float(activation_groups["compute_sfu"])
        else:
            flops["compute_fma"] = 3.0 * swiglu_elements
            flops["compute_alu"] = 3.0 * swiglu_elements + router_scale_elements + reduction_elements
            flops["compute_sfu"] = 2.0 * swiglu_elements
    demand = build_forward_demand(l2_total, dram_bytes, False, flops)
    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware)
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model="gpu_mshr",
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind="generic",
        include_dispatch_overhead=True,
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        support_level="aggregate_only",
        limitations=gpu_partial_limitations(
            hardware,
            "marlin_w4a16_physical_kernels_aggregated",
            "marlin_w4a16_worker_assignment_unavailable",
            "marlin_w4a16_l2_residency_order_unavailable",
            "marlin_w2_router_weight_scalar_math_not_packed_instruction_counts",
            "moe_gather_service_class_not_separated",
            "production_duration_calibration_not_used",
        ),
    )
    structured = attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name="global_memory",
        source_producers_per_unit=gpu_memory_source(hardware)[0],
        independent_requests_per_producer=gpu_memory_source(hardware)[1],
        frontend_rate_per_unit_s=None,
    )
    structured = _split_gpu_grouped_gemm_memory_paths(
        structured, traffic)
    if kernel.attributes["swiglu_limit"] is not None:
        structured = dataclasses.replace(structured,
            limitations=structured.limitations + (
                "clamped_mxfp4_routed_activation_roofline_only",
                "live_clamp_silu_math_not_compiled_instruction_counts",
                "actual_backend_and_clamped_activation_fusion_services_unbound"))
    return structured, traffic.compulsory_bytes


# wp11_moe_fp8: vLLM 0.24 FP8 block-quantized Triton grouped GEMM (H200).
_FP8_BLOCK_ALGORITHM_PREFIX = "vllm_triton_fused_moe_fp8_block128"
_FP8_BLOCK_SOURCE_ALGORITHM_PREFIXES = (
    "vllm_0_30_sm90_fp8_block128_swap_ab",
    "vllm_0_30_sm90_fp8_block128",
)
_FP8_BLOCK_SIZE = 128
_FP8_SCALE_BYTES = 4


# SGLang 0.5.12's own fused_moe_kernel (H200 launch selector): always
# aligned (moe_align_block_size), the _down configs read A and B via TMA.
_SGLANG_FP8_BLOCK_ALGORITHM_PREFIX = "sglang_triton_fused_moe_fp8_block128"


def _is_fp8_block_launch(kernel: KernelSpec) -> bool:
    return kernel.launch.algorithm.startswith(
        (_FP8_BLOCK_ALGORITHM_PREFIX, _SGLANG_FP8_BLOCK_ALGORITHM_PREFIX,
         *_FP8_BLOCK_SOURCE_ALGORITHM_PREFIXES))


def _is_sglang_fp8_block_launch(kernel: KernelSpec) -> bool:
    return kernel.launch.algorithm.startswith(
        _SGLANG_FP8_BLOCK_ALGORITHM_PREFIX)


def _sglang_fp8_block_uses_tma(kernel: KernelSpec) -> bool:
    variant = kernel.variant.lower()
    if kernel.launch.algorithm == (
            f"{_SGLANG_FP8_BLOCK_ALGORITHM_PREFIX}_aligned_tma_{variant}"):
        return True
    if kernel.launch.algorithm == (
            f"{_SGLANG_FP8_BLOCK_ALGORITHM_PREFIX}_aligned_{variant}"):
        return False
    raise ValueError(
        "SGLang FP8 block FUSED_MOE algorithm must name its aligned "
        f"{variant} program: {kernel.launch.algorithm!r}")


def _fp8_block_assignment(kernel: KernelSpec) -> str:
    if _is_sglang_fp8_block_launch(kernel):
        _sglang_fp8_block_uses_tma(kernel)
        return "aligned"
    variant = kernel.variant.lower()
    for prefix in (_FP8_BLOCK_ALGORITHM_PREFIX,
            *_FP8_BLOCK_SOURCE_ALGORITHM_PREFIXES):
        for assignment in ("naive", "aligned"):
            if kernel.launch.algorithm == f"{prefix}_{assignment}_{variant}":
                return assignment
    raise ValueError(
        "FP8 block FUSED_MOE algorithm must name the naive or aligned "
        f"{variant} source program: {kernel.launch.algorithm!r}")


def _fp8_block_counts(kernel: KernelSpec) -> Tuple[Tuple[int, ...], int]:
    """Per-expert rows the kernel sees and its top-k.

    With SGLang shared-experts fusion the shared expert is expert E of the
    routed kernel and every token is routed to it (topk.py appends the
    column), so it adds one expert with ``tokens`` rows and one to top-k.
    """

    _, local_counts = _routing(kernel)
    topk = int(kernel.shape["topk"])
    if "fused_shared_experts" in kernel.attributes:
        if not _is_sglang_fp8_block_launch(kernel):
            raise ValueError(
                "fused shared experts require the SGLang fused_moe program")
        fused = int(kernel.attributes["fused_shared_experts"])
        tokens = int(kernel.shape["tokens"])
        return tuple(local_counts) + (tokens,) * fused, topk + fused
    return tuple(local_counts), topk


def _fp8_block_launch_invariants(kernel):
    """Validate launch facts that do not depend on any measured mechanism row.

    Source program identity, CTA thread count, complete tile/shape geometry
    and the grid's fixed relationship to the expanded-row schedule are
    properties of the launch itself; a malformed value is a broken contract
    and must raise regardless of whether the hardware family carries the
    matching mechanism row. Only the missing row is a measured-pipeline
    gap. Routing conservation and EP scope are kernel-owned and validated
    by the traffic owner.
    """
    if not _is_fp8_block_launch(kernel):
        raise ValueError(
            "FP8 block FUSED_MOE requires its source algorithm identity")
    _fp8_block_assignment(kernel)
    parameters = kernel.launch.parameters
    warps, stages = (
        int(parameters[name]) for name in ("num_warps", "num_stages"))
    if kernel.launch.block_size != 32 * warps:
        raise ValueError(
            "FP8 block FUSED_MOE block must equal 32 x num_warps")
    variant, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    sglang = _is_sglang_fp8_block_launch(kernel)
    allowed_tile_k = (64, _FP8_BLOCK_SIZE) if sglang else (_FP8_BLOCK_SIZE,)
    if tile_k not in allowed_tile_k or projection_k % _FP8_BLOCK_SIZE:
        raise ValueError(
            "FP8 block FUSED_MOE requires complete 128-wide K blocks")
    if projection_n % _FP8_BLOCK_SIZE:
        raise ValueError(
            "FP8 block FUSED_MOE requires complete 128-wide N blocks")
    local_counts, kernel_topk = _fp8_block_counts(kernel)
    routed_rows = sum(local_counts)
    if routed_rows != int(kernel.shape["tokens"]) * kernel_topk:
        raise ValueError(
            "FP8 block FUSED_MOE routing does not conserve tokens * topk")
    if _fp8_block_assignment(kernel) == "naive":
        populated_m_tiles = routed_rows
    else:
        populated_m_tiles = sum(
            math.ceil(count / tile_m) for count in local_counts if count > 0)
    # The grid must be a whole multiple of the N-tile count, its M-tile
    # factor must reproduce the padded_rows schedule, and the grid must
    # cover every populated M tile the routing actually produces. These
    # are launch-owned facts: they hold for every source algorithm the
    # launch can name, so a violation is a broken contract regardless of
    # whether the hardware family carries the matching mechanism row.
    n_tiles = math.ceil(projection_n / tile_n)
    grid_m_tiles, remainder = divmod(kernel.launch.grid_size, n_tiles)
    expanded_m_tiles = math.ceil(int(parameters["padded_rows"]) / tile_m)
    if remainder or grid_m_tiles != expanded_m_tiles:
        raise ValueError(
            "FP8 block FUSED_MOE grid disagrees with its expanded rows")
    if populated_m_tiles > grid_m_tiles:
        raise ValueError(
            "FP8 block FUSED_MOE populated tiles exceed the launch grid")
    return tile_m, tile_n, tile_k, warps, stages


def _fp8_block_tile_program(kernel, family):
    """Resolve the measured FP8 tile-program name for this launch.

    vLLM 0.30 SM90 launches use the native ``bm*_bn*_bk*_w*_s*`` program
    whenever the family carries it; the measured SGLang BN64 row is the
    fallback only when the native program has no mechanism. SGLang launches
    always resolve through their own algorithm prefix. The same mapping
    feeds support-reason checks, stage pricing and launch selection, so a
    hardware family with only the native row is never rewritten onto the
    SGLang row it does not carry.
    """
    tile_m, tile_n, tile_k, warps, stages = _fp8_block_launch_invariants(kernel)
    program = f"bm{tile_m}_bn{tile_n}_bk{tile_k}_w{warps}_s{stages}"
    if _is_sglang_fp8_block_launch(kernel):
        prefix = "sgl_tma_" if _sglang_fp8_block_uses_tma(kernel) else "sgl_"
        return prefix + program
    if (tile_n == 64
            and kernel.launch.algorithm.startswith(
                _FP8_BLOCK_SOURCE_ALGORITHM_PREFIXES)
            and (family is None or program not in family)):
        return "sgl_" + program
    return program


def fp8_block_tile_support_reason(kernel: KernelSpec, hardware):
    """Return only the measured-pipeline gap; malformed launches still fail."""
    family = hardware.grouped_gemm_fp8_block_tile_pipelines
    program = _fp8_block_tile_program(kernel, family)
    if family is None or program not in family:
        return ("FP8 block FUSED_MOE tile program has no measured pipeline "
                f"mechanism on {hardware.hardware_id}: {program}")
    row = family[program]
    if kernel.launch.block_size != row["threads_per_cta"]:
        raise ValueError("FP8 block FUSED_MOE block disagrees with its program")
    if (int(kernel.launch.parameters["dynamic_shared_memory_bytes"])
            != row["dynamic_shared_memory_bytes_per_cta"]):
        raise ValueError(
            "FP8 block FUSED_MOE shared memory disagrees with its program")
    return None


def _fp8_block_tile_pipeline(kernel: KernelSpec, hardware):
    """Return the measured FP8 tile-program row selected by the launch."""
    reason = fp8_block_tile_support_reason(kernel, hardware)
    if reason is not None:
        raise NotImplementedError(reason)
    program = _fp8_block_tile_program(
        kernel, hardware.grouped_gemm_fp8_block_tile_pipelines)
    return program, hardware.grouped_gemm_fp8_block_tile_pipelines[program]


def _fp8_block_traffic(kernel: KernelSpec) -> Tuple[_GpuGroupedGemmTraffic, int]:
    """Source-derived FP8 block grouped-GEMM hierarchy traffic.

    ``fused_moe_kernel`` with ``use_fp8_w8a8`` and a [128, 128] block loads,
    per K block of every populated program, the valid FP8 A rows (one byte),
    the complete FP8 B panel (the N coordinate wraps, it is not masked), one
    FP32 per-token-group A scale per valid row and one FP32 B block scale
    per distinct ``offs_bn // 128`` group. SGLang BN256 spans two groups.
    The naive assignment gives every route its own M tile with one valid
    row; the aligned assignment packs each expert's routes into
    ``ceil(rows / BLOCK_SIZE_M)`` tiles.  W2 loads one FP32 router weight
    per valid row after its K loop (source routed-weight marker).  The BF16
    output is masked to real rows and columns.  DRAM carries the unique
    compulsory bytes; no cross-call residency is assumed.
    Returns the traffic and the populated M-tile count.
    """

    variant, source_rows, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, tile_k = _tile(kernel)
    sglang = _is_sglang_fp8_block_launch(kernel)
    # SGLang's _down configs also use BLOCK_SIZE_K 64: the kernel then reads
    # the same 128x128 scale block on each of its two K rounds.
    allowed_tile_k = (64, _FP8_BLOCK_SIZE) if sglang else (_FP8_BLOCK_SIZE,)
    if tile_k not in allowed_tile_k or projection_k % _FP8_BLOCK_SIZE:
        raise ValueError(
            "FP8 block FUSED_MOE requires complete 128-wide K blocks")
    if projection_n % _FP8_BLOCK_SIZE:
        raise ValueError(
            "FP8 block FUSED_MOE requires complete 128-wide N blocks")
    if int(kernel.attributes["expert_parallel_size"]) != 1:
        raise NotImplementedError(
            "FP8 block FUSED_MOE is source-backed only without expert "
            "parallelism")
    global_counts, local_counts = _routing(kernel)
    if global_counts != local_counts:
        raise ValueError(
            "FP8 block FUSED_MOE without EP must own every routed row")
    variant, _, projection_k, projection_n = _projection(kernel)
    sglang = _is_sglang_fp8_block_launch(kernel)
    # The launch-owned grid/padded_rows/populated-tile invariants were
    # validated up front by _fp8_block_launch_invariants, so a malformed
    # grid fails closed even when the measured-pipeline gap diverts the
    # Serving path onto mathematical partial.
    tile_m, tile_n, tile_k, _, _ = _fp8_block_launch_invariants(kernel)
    local_counts, kernel_topk = _fp8_block_counts(kernel)
    routed_rows = sum(local_counts)
    if variant == "w2":
        source_rows = routed_rows
    assignment = _fp8_block_assignment(kernel)
    if assignment == "naive":
        populated_m_tiles = routed_rows
    else:
        populated_m_tiles = sum(
            math.ceil(count / tile_m) for count in local_counts if count > 0)
    n_tiles = math.ceil(projection_n / tile_n)
    k_blocks = projection_k // _FP8_BLOCK_SIZE
    k_rounds = projection_k // tile_k
    active_experts = sum(1 for count in local_counts if count > 0)
    # A TMA descriptor load (SGLang _down) moves the whole BLOCK_SIZE_M x
    # BLOCK_SIZE_K box of the expert-sorted A, padding rows included.
    tma_a = sglang and _sglang_fp8_block_uses_tma(kernel)
    loaded_a_rows = populated_m_tiles * tile_m if tma_a else routed_rows
    # TMA loads every row of its payload box. Scalar scale loads retain
    # token_mask, so padding must not become an FP32 scale request.
    activation_l2 = float(
        (loaded_a_rows * projection_k
         + routed_rows * k_rounds * _FP8_SCALE_BYTES) * n_tiles)
    # SGLang BN256 spans two distinct 128-column scale blocks; BN64/128
    # references one. Repeated lanes are not distinct payload bytes.
    scale_blocks_per_n_tile = max(1, tile_n // _FP8_BLOCK_SIZE)
    weight_l2 = float(
        populated_m_tiles * n_tiles
        * (projection_k * tile_n
           + k_rounds * scale_blocks_per_n_tile * _FP8_SCALE_BYTES))
    if variant == "w1":
        unique_source_rows = min(
            source_rows, math.ceil(routed_rows / kernel_topk))
    elif variant == "w2":
        unique_source_rows = loaded_a_rows
    else:
        raise AssertionError(f"unexpected FUSED_MOE variant: {variant!r}")
    # BK64 reads a group scale twice at L2 request level, but the caller
    # materializes one FP32 value per 128-wide group. TMA padding likewise
    # does not add valid scale cells to the compulsory operand interface.
    unique_scale_rows = unique_source_rows if variant == "w1" else routed_rows
    activation_compulsory = float(
        unique_source_rows * projection_k
        + unique_scale_rows * k_blocks * _FP8_SCALE_BYTES)
    weight_compulsory = float(
        active_experts
        * (projection_k * projection_n
           + (projection_n // _FP8_BLOCK_SIZE) * k_blocks * _FP8_SCALE_BYTES))
    routed_weight_l2, routed_weight_compulsory = (
        _source_routed_weight_read_traffic(
            kernel, routed_rows=float(routed_rows), n_tiles=n_tiles))
    output_bytes = float(
        routed_rows * projection_n * int(kernel.shape["dtype_bytes"]))
    traffic = _GpuGroupedGemmTraffic(
        activation_l2_read_bytes=activation_l2,
        weight_l2_read_bytes=weight_l2,
        routed_weight_l2_read_bytes=routed_weight_l2,
        output_l2_write_bytes=output_bytes,
        activation_compulsory_read_bytes=activation_compulsory,
        weight_compulsory_read_bytes=weight_compulsory,
        routed_weight_compulsory_read_bytes=routed_weight_compulsory,
        activation_dram_read_bytes=activation_compulsory,
        weight_dram_read_bytes=weight_compulsory,
        routed_weight_dram_read_bytes=routed_weight_compulsory,
        output_dram_write_bytes=output_bytes,
    )
    validated_l2_miss_rate(traffic.dram_total_bytes, traffic.l2_total_bytes)
    return traffic, populated_m_tiles


def fp8_block_traffic_diagnostics(kernel: KernelSpec):
    """Expose the predictor's physical traffic and active/empty CTA geometry."""
    traffic, populated_m_tiles = _fp8_block_traffic(kernel)
    variant, _, _, projection_n = _projection(kernel)
    n_tiles = math.ceil(projection_n / _tile(kernel)[1])
    active_ctas = populated_m_tiles * n_tiles
    return {
        "traffic_bytes": dataclasses.asdict(traffic),
        "l2_total_bytes": traffic.l2_total_bytes,
        "dram_total_bytes": traffic.dram_total_bytes,
        "compulsory_bytes": traffic.compulsory_bytes,
        "populated_m_tiles": populated_m_tiles,
        "active_ctas": active_ctas,
        "empty_ctas": kernel.launch.grid_size - active_ctas,
    }


def _fp8_block_model(kernel: KernelSpec, hardware) -> Tuple[PredictionInput, float]:
    """Price the FP8 block Triton launch with the grouped-GEMM service model.

    The template is the resident-wave K pipeline of the BF16
    ``grouped_gemm_tile_pipeline``: one fixed span, then per resident wave one
    service round per BK128 block, each round the larger of the measured
    local K-round interval of this tile program and the DRAM round service,
    plus the early-return CTA tail.  Aggregate hierarchy traffic and Tensor
    work stay independent resource bounds.  Tensor operations are expressed
    in the primary Tensor variant's units using the HWSpec FP8/primary rate
    ratio; the per-block dequantization ``acc += dot * a_scale * b_scale``
    is FMA-pipe work (one multiply and one fused multiply-add per
    accumulator element per K block).
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            "FP8 block FUSED_MOE is modeled only for GPU memory hierarchies")
    if kernel.dtype != "bf16":
        raise ValueError("FP8 block FUSED_MOE requires BF16 compute dtype")
    if kernel.attributes["routed_weight_dtype"] != "fp8":
        raise ValueError("FP8 block FUSED_MOE requires FP8 routed weights")
    variant, _, projection_k, projection_n = _projection(kernel)
    tile_m, tile_n, _ = _tile(kernel)
    program, pipeline_row = _fp8_block_tile_pipeline(kernel, hardware)
    traffic, populated_m_tiles = _fp8_block_traffic(kernel)
    local_counts, _ = _fp8_block_counts(kernel)
    routed_rows = float(sum(local_counts))
    n_tiles = math.ceil(projection_n / tile_n)
    # K rounds of the tile program (BLOCK_SIZE_K 64 or 128); the scaled dot
    # accumulates once per round.
    k_blocks = projection_k // _tile(kernel)[2]
    fp8_rate = hardware.resource_by_role("compute_tensor_fp8").capacity
    primary_rate = hardware.resource_by_role("compute_tensor").capacity
    if fp8_rate <= 0.0 or primary_rate <= 0.0:
        raise ValueError("FP8 block FUSED_MOE requires positive Tensor rates")
    fp8_tensor_flops = float(
        2 * populated_m_tiles * tile_m * projection_k * n_tiles * tile_n)
    dequantization_flops = float(
        3 * populated_m_tiles * tile_m * n_tiles * tile_n * k_blocks)
    flops = {
        "compute_tensor": fp8_tensor_flops * primary_rate / fp8_rate,
        "compute_fma": dequantization_flops,
        "compute_alu": _source_routed_weight_scale_operations(
            kernel, routed_rows=routed_rows, projection_n=projection_n),
        "compute_sfu": 0.0,
    }
    demand = build_forward_demand(
        traffic.l2_total_bytes, traffic.dram_total_bytes, False, flops)
    eta, wave_efficiency, active_units = launch_wave_geometry(kernel, hardware)
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=variant,
        mem_model="gpu_mshr",
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind="generic",
        include_dispatch_overhead=True,
        dram_capacity_multiplier=hardware.dram_protocol_efficiency_l1cached,
        support_level="aggregate_only",
        limitations=gpu_partial_limitations(
            hardware,
            "moe_gather_service_class_not_separated",
            "moe_physical_worker_assignment_unavailable",
            "moe_l2_residency_order_unavailable",
            "moe_load_compute_store_buffer_lifecycle_unavailable",
        ),
    )
    producers, requests_per_producer = gpu_memory_source(hardware)
    structured = attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name="global_memory",
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
    )
    structured = _split_gpu_grouped_gemm_memory_paths(structured, traffic)
    pipeline_plan = GpuGroupedGemmPipelineEnvelopeDemand(
        active_ctas=populated_m_tiles * n_tiles,
        k_tiles=k_blocks,
        resident_ctas_per_sm=pipeline_row[
            "resident_ctas_per_sm_by_variant"][variant],
        dram_read_bytes=traffic.dram_read_bytes,
        fp8_block_program=program,
    )
    structured = dataclasses.replace(
        structured,
        pipeline_plan=pipeline_plan,
        phase_ledger=_grouped_matmul_phase_ledger(structured),
    )
    return structured, traffic.compulsory_bytes


def kimi_ep_marlin_logical_work(kernel: KernelSpec):
    """Useful EP-off I shards or EP8 W4A16 matrices and unmasked SiLU.

    EP-off retains all384 experts at the producer's actual local I. EP8
    DP/SP1 keeps token rows replicated while the linear expert map masks
    each rank's assignments. Ordinary (unclamped) apply_moe_activation still
    consumes every token/top-k row, including rows no local GEMM wrote.
    The map-aware final sum skips those intermediate rows and writes zero
    when none are local. No padding, issued loads, zero-grid launch or
    initialized values for the unwritten activation inputs are invented.
    """
    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "routed_experts"
            or not HWSpec.source_contract_supported("kimi_ep_marlin_swiglu_math", kernel.hardware_id) or kernel.dtype != "bf16"
            or kernel.launch is not None or kernel.execution_support != "unsupported"
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] != "kimi_ep_marlin_swiglu_v1"
            or kernel.attributes["routed_weight_dtype"] != "int4"
            or kernel.attributes["weight_group_size"] != 32
            or kernel.attributes["expert_parallel_size"] not in (1, 8)
            or kernel.attributes["routed_scaling_factor"] != 1.0
            or kernel.attributes["swiglu_limit"] is not None
            or (kernel.shape["experts"], kernel.shape["topk"], kernel.shape["hidden"]) != (384, 8, 7168)
            or (kernel.attributes["expert_parallel_size"], kernel.shape["intermediate"]) not in (
                (1, 2048), (1, 1024), (1, 512), (1, 256), (8, 2048))
            or kernel.shape["intermediate"] in (512, 1024) and not HWSpec.source_contract_supported("kimi_grouped_marlin_source_math", kernel.hardware_id)
            or kernel.shape["intermediate"] == 256 and not HWSpec.source_contract_supported("kimi_marlin_i256_source_math", kernel.hardware_id)):
        raise ValueError("Kimi EP Marlin work requires its exact unbound BF16/INT4 source program")
    tokens, topk, hidden, width = (kernel.shape[name] for name in ("tokens", "topk", "hidden", "intermediate"))
    counts, local = _routing(kernel)
    rank = kernel.attributes["expert_rank"]
    ep = kernel.attributes["expert_parallel_size"]
    local_experts = 384 // ep
    owned = tuple(range(rank * local_experts, (rank + 1) * local_experts))
    if (tuple(kernel.attributes["owned_experts"]) != owned or sum(counts) != tokens * topk
            or any(count > tokens for count in counts)
            or local != tuple(count if index in owned else 0 for index, count in enumerate(counts))):
        raise ValueError("Kimi EP Marlin requires distinct per-token experts, conserved global rows and its exact linear local expert map")
    routed, active = sum(local), sum(count > 0 for count in local)
    full_rows = tokens * topk
    elements = full_rows * width
    arithmetic = silu_source_arithmetic(elements, kernel.dtype, has_clamp=False)
    stages = []

    def stage(name, payloads, *, tensor=0, scalar=0, sfu=0):
        demand = {role: 0 for role in ("compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu")}
        demand.update(compute_tensor=tensor, compute_fma=scalar, compute_sfu=sfu)
        record = {"stage": name, "payload_bytes": payloads,
            "logical_bytes": sum(payloads.values()), "demand": demand}
        stages.append(record)
        return record

    w13 = stage("w13", {"hidden_row_uses": 2 * routed * hidden,
        "packed_int4_weights": active * 2 * hidden * width // 2,
        "bf16_group32_scales": 2 * active * 2 * width * (hidden // 32),
        "valid_gate_up_output": 4 * routed * width}, tensor=4 * routed * hidden * width)
    activation = stage("silu_mul", {"full_gate_up_input": 4 * elements,
        "full_activated_output": 2 * elements},
        scalar=arithmetic["baseline_demand"]["compute_fma"], sfu=arithmetic["baseline_demand"]["compute_sfu"])
    activation.update(elements=elements, source_rows=full_rows, source_arithmetic=arithmetic)
    w2 = stage("weighted_w2", {"local_activated_input": 2 * routed * width,
        "packed_int4_weights": active * hidden * width // 2,
        "bf16_group32_scales": 2 * active * hidden * (width // 32),
        "fp32_router_weight_uses": 4 * routed, "local_weighted_output": 2 * routed * hidden},
        tensor=2 * routed * width * hidden, scalar=routed * hidden)
    reduction = stage("map_aware_sum", {"local_intermediate_input": 2 * routed * hidden,
        "int32_topk_id_uses": 4 * full_rows if ep > 1 else 0,
        "int32_expert_map_uses": 4 * full_rows if ep > 1 else 0,
        "bf16_output": 2 * tokens * hidden}, scalar=routed * hidden)
    return {"global_routed_rows": full_rows, "local_routed_rows": routed, "active_local_experts": active,
        "matrix_stages": (w13, w2), "activation": activation, "weighted_sum": reduction,
        "source_stage_order": tuple(part["stage"] for part in stages), "stages": tuple(stages),
        "logical_bytes": sum(part["logical_bytes"] for part in stages),
        "actual_provider_bound": False, "complete_source_service": False, "normal_capacity_accepted": False,
        "limitations": ("useful_unpadded_local_matrix_math_not_issued_MMA_or_actual_launch",
            "full_unmasked_activation_includes_unwritten_nonlocal_rows_without_asserting_their_values",
            "INT4_dequantization_scale_conversion_padding_alignment_scan_and_integer_services_unpriced",
            "IDs_and_map_bytes_are_logical_role_uses_not_per_vector_pack_issued_loads_or_DRAM",
            "map_aware_sum_counts_source_acc_zero_plus_each_valid_value_not_unknown_per_token_minimal_adds",
            "callee_workspace_compiler_lifetimes_rounded_arithmetic_and_runtime_unbound")}


def unquantized_bf16_experts_logical_work(kernel: KernelSpec):
    """Useful BF16 expert stages; routing is an already supplied producer.

    Reuse the ordinary typed matmul and SiLU work. Weights are read once per
    active expert, never replaced by one shared dense matrix. This is not a
    claim about native padding, backend choice or deployed launch geometry.
    """
    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "routed_experts"
            or not HWSpec.source_contract_supported("unbound_bf16_expert_source_math", kernel.hardware_id) or kernel.dtype != "bf16"
            or kernel.attributes["mathematical_program"] not in (
                "unquantized_bf16_swiglu_v1", "unquantized_bf16_grouped_swiglu_v1",
                "unquantized_bf16_legacy_grouped_swiglu_v1")
            or kernel.launch is not None or kernel.execution_support != "unsupported"):
        raise ValueError("BF16 experts require their unbound H200 mathematical program")
    tokens, topk, hidden, width = (kernel.shape[name] for name in ("tokens", "topk", "hidden", "intermediate"))
    counts, local = _routing(kernel)
    ep_size, rank = (kernel.attributes[name] for name in ("expert_parallel_size", "expert_rank"))
    from src.kernel.kernel_spec import linear_expert_partition
    owned = linear_expert_partition(kernel.shape["experts"], ep_size, rank)
    if (tuple(kernel.attributes["owned_experts"]) != owned
            or local != tuple(count if index in owned else 0 for index, count in enumerate(counts))
            or sum(counts) != tokens * topk):
        raise ValueError("BF16 experts require conserved assignments and exact linear ownership")
    mapped = ep_size > 1
    full_rows, routed = tokens * topk, sum(local)
    legacy = kernel.attributes["mathematical_program"] == "unquantized_bf16_legacy_grouped_swiglu_v1"
    matrices = []
    for stage, k, n in (("w13", hidden, 2 * width), ("w2", width, hidden)):
        work = tuple(_unquantized_matmul_math({"b": 1, "m": count, "n": n, "k": k},
            "bf16", 2, stage) for count in local if count)
        operands = tuple(operand for part in work for operand in part["operands"])
        if legacy and full_rows > routed:
            #0.11's expert_id==-1 branch runs inside both GEMMs. It writes
            # zeros for valid nonlocal rows, then returns before A/B loads.
            # W13 has2*I columns;W2 hasH. This is not0.30's full W2 prezero.
            elements = (full_rows - routed) * n
            operands += ({"role": stage + "_nonlocal_zero", "dtype": "bf16", "elements": elements,
                "bytes": 2 * elements, "access": "write"},)
        matrices.append({"stage": stage,
            "mathematical_tensor_flops": sum(part["mathematical_tensor_flops"] for part in work),
            "operands": operands, "logical_bytes": sum(operand["bytes"] for operand in operands)})
    return {"routed_rows": routed, "active_experts": sum(count > 0 for count in local),
        "matrix_stages": matrices, "quantization_inputs": (),
        **_expert_swiglu_reduction_work(tokens, topk, hidden, width, routed, full_rows if mapped else routed, "bf16"),
        #0.30 filters nonlocal assignments before W1/W2. It zeroes the
        # entire routed output before W2; W1 does not initialize nonlocal
        # activation rows. Count the real separate zero, including rows W2
        # subsequently overwrites, rather than inventing per-GEMM zeros.
        "output_initialization_bytes": 2 * full_rows * hidden if mapped and not legacy else 0,
        "source_stage_order": ("w13", "silu_mul", *(("zero_output",) if mapped and not legacy else ()), "w2", "weighted_sum"),
        "actual_provider_bound": False, "complete_source_service": False}


def _expert_swiglu_reduction_work(tokens, topk, hidden, width, routed, vector_rows, dtype):
    """Shared mathematical activation and FP32-route weighted output work."""
    elements = vector_rows * width
    return {"activation": {"elements": elements, "logical_bytes": 6 * elements,
            "source_arithmetic": silu_source_arithmetic(elements, dtype, has_clamp=False),
            "exp_operations": elements, "reciprocal_operations": 0},
        "weighted_sum": {"multiply_flops": routed * hidden,
            "addition_flops": tokens * (topk - 1) * hidden,
            "logical_bytes": 2 * vector_rows * hidden + 4 * routed + 2 * tokens * hidden},
        "routing_id_payload_read_bytes": 4 * tokens * topk,
        "global_routed_rows": tokens * topk, "vector_rows": vector_rows}


def block_fp8_experts_logical_work(kernel: KernelSpec):
    """Declared useful W13 -> SwiGLU -> W2 -> weighted sum program.

    EP8 filters nonlocal assignment blocks. Activation and its quantizer
    process every slot; the whole W2 output is zeroed before local writes.
    Unwritten W1 activation values are not asserted to be initialized.
    Per-expert matmul operands are logical role uses, not measured DRAM or
    provider padding/metadata. Quantization reuses the independent source
    program; routing logits/selection remain a separate required producer.
    """
    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "routed_experts"
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] not in ("block_fp8_swiglu_v1", "block_fp8_replicated_ep_swiglu_v1")
            or kernel.launch is not None or kernel.execution_support != "unsupported"):
        raise ValueError("FP8 expert logical work requires its unbound mathematical program")
    tokens, topk, hidden, width = (kernel.shape[name] for name in ("tokens", "topk", "hidden", "intermediate"))
    counts, local = _routing(kernel)
    replicated_ep = kernel.attributes["mathematical_program"] == "block_fp8_replicated_ep_swiglu_v1"
    if not replicated_ep and (counts != local or tuple(kernel.attributes["owned_experts"]) != tuple(range(kernel.shape["experts"]))):
        raise ValueError("FP8 mathematical experts must preserve the complete expert axis")
    routed = sum(local)
    full_rows = tokens * topk
    if sum(counts) != full_rows or (not replicated_ep and routed != full_rows):
        raise ValueError("FP8 mathematical experts must conserve token/top-k assignments")
    if replicated_ep:
        rank = kernel.attributes["expert_rank"]
        owned = tuple(range(32 * rank, 32 * (rank + 1)))
        if (tuple(kernel.attributes["owned_experts"]) != owned or any(count > tokens for count in counts)
                or local != tuple(count if index in owned else 0 for index, count in enumerate(counts))):
            raise ValueError("replicated FP8 EP requires conserved distinct assignments and exact linear local experts")
    matrices = []
    for stage, k, n in (("w13", hidden, 2 * width), ("w2", width, hidden)):
        payload = {}
        for count in local:
            if count:
                operands, scales = _block_fp8_matmul_payloads(1, count, n, k)
                for name, value in {**operands, **scales}.items():
                    if name not in payload:
                        payload[name] = 0
                    payload[name] += value
        matrices.append({"stage": stage, "mathematical_fp8_flops": 2 * routed * k * n,
            "payload_bytes": payload, "logical_bytes": sum(payload.values())})
    vector_rows = full_rows if replicated_ep else routed
    return {"routed_rows": routed, "active_experts": sum(count > 0 for count in local),
        "matrix_stages": matrices,
        "quantization_inputs": ((tokens, hidden), (vector_rows, width)),
        **_expert_swiglu_reduction_work(tokens, topk, hidden, width, routed, vector_rows, kernel.dtype),
        "output_initialization_bytes": 2 * full_rows * hidden if replicated_ep else 0,
        "source_stage_order": ("quantize_input", "w13", "silu_mul", "quantize_activated",
            *(("zero_output",) if replicated_ep else ()), "w2", "weighted_sum"),
        "actual_provider_bound": False, "complete_source_service": False,
        "limitations": ("useful_unpadded_mathematical_tensor_program_only",
            "per_expert_operand_role_payload_not_observed_DRAM_or_cache_traffic",
            "scale_application_cast_comparison_address_and_alignment_services_unpriced",
            "internal_allocation_workspace_padding_compiler_and_backend_unbound",
            "routing_probability_scale_already_in_weights_no_duplicate_output_scale")}


def sglang0519_fp8_fused_shared_logical_work(kernel: KernelSpec):
    """Declared0.5.19 CUTLASS useful math and role payloads, without launch.

    Preserve the external256-expert histogram. The final routed column is
    one shared expert with every token, constructed only inside this callee
    work account. Quantization follows this caller's own unrounded CUDA JIT
    source, not a vLLM quantizer identity or a different SGLang recipe.
    """
    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "routed_experts"
            or not HWSpec.source_contract_supported("sglang_0_5_19_symbolic_fused_shared_moe", kernel.hardware_id)
            or kernel.dtype != "bf16" or kernel.launch is not None or kernel.execution_support != "unsupported"
            or kernel.attributes["mathematical_program"] != "sglang_0_5_19_fp8_fused_shared_swiglu_v1"
            or (kernel.shape["experts"], kernel.shape["topk"], kernel.shape["hidden"], kernel.shape["intermediate"]) != (256, 8, 7168, 256)
            or kernel.attributes["expert_parallel_size"] != 1 or kernel.attributes["expert_rank"] != 0
            or kernel.attributes["routed_weight_dtype"] != "fp8" or kernel.attributes["weight_group_size"] != 0
            or kernel.attributes["fused_shared_experts"] != 1 or kernel.attributes["routed_scaling_factor"] != 1.0
            or kernel.attributes["swiglu_limit"] is not None):
        raise ValueError("SGLang0.5.19 experts require their exact no-launch fused-shared source program")
    tokens, hidden, width = (kernel.shape[name] for name in ("tokens", "hidden", "intermediate"))
    counts, local = _routing(kernel)
    if (counts != local or tuple(kernel.attributes["owned_experts"]) != tuple(range(256))
            or sum(counts) != tokens * 8 or any(count > tokens for count in counts)):
        raise ValueError("SGLang routed histogram must retain256 conserved distinct-expert assignments")
    expert_rows = (*counts, tokens)
    routes = tokens * 9
    stages = []

    def stage(name, payload, *, fp8_flops=0, scalar=0, sfu=0):
        demand = {role: 0 for role in ("compute_tensor", "compute_vector", "compute_reduction",
            "compute_fma", "compute_sfu", "compute_alu")}
        demand.update(compute_fma=scalar, compute_sfu=sfu)
        value = {"stage": name, "payload_bytes": payload, "logical_bytes": sum(payload.values()),
            "mathematical_fp8_flops": fp8_flops, "demand": demand}
        stages.append(value)
        return value

    def quantize(name, rows, columns):
        elements = rows * columns
        groups = elements // 128
        # CUDA JIT source: raw_scale=amax*(1/448), quant_scale=448/amax,
        # followed by one value multiply. Abs/max/cast/clamp stay unpriced.
        value = stage(name, {"bf16_input": 2 * elements, "fp8_output": elements,
            "fp32_scale_output": 4 * groups}, scalar=elements + 2 * groups)
        value.update(elements=elements, groups=groups, source_group_size=128,
            source_scale_ue8m0=False, source_column_major=False, source_fuse_silu=False,
            source_masked_m=None)
        return value

    stage("prepare_route_maps", {"int32_topk_ids": 4 * routes,
        "int32_input_map": 4 * routes, "int32_output_map": 4 * routes})
    quantize("quantize_input", tokens, hidden)
    shuffle = stage("shuffle_input_and_scales", {"fp8_input_row_uses": routes * hidden,
        "fp8_shuffled_output": routes * hidden, "fp32_scale_row_uses": 4 * routes * (hidden // 128),
        "fp32_shuffled_scales": 4 * routes * (hidden // 128), "int32_input_map": 4 * routes})
    shuffle["source_requested_map_bytes"] = 4 * routes * -(-hidden // 4096)
    for name, reduction, columns in (("w13", hidden, 2 * width), ("w2", width, hidden)):
        payload = {}
        for count in expert_rows:
            if count:
                operands, scales = _block_fp8_matmul_payloads(1, count, columns, reduction)
                for role, amount in {**operands, **scales}.items():
                    if role not in payload:
                        payload[role] = 0
                    payload[role] += amount
        stage(name, payload, fp8_flops=2 * routes * reduction * columns)
        if name == "w13":
            elements = routes * width
            common = silu_source_arithmetic(elements, kernel.dtype, has_clamp=False)
            # This JIT source has no runtime alpha/beta multiply/add. Reuse
            # the common primitives, removing those two absent operations.
            arithmetic = {**common,
                "mathematical_primitives": {**common["mathematical_primitives"],
                    "fp32_multiply": elements, "fp32_add": elements},
                "baseline_demand": {**common["baseline_demand"],
                    "compute_fma": common["baseline_demand"]["compute_fma"] - 2 * elements},
                "source_expression": "(gate / (1 + exp(-gate))) * up"}
            activation = stage("silu_mul", {"gate_up_input": 4 * elements,
                "activated_output": 2 * elements}, scalar=arithmetic["baseline_demand"]["compute_fma"],
                sfu=arithmetic["baseline_demand"]["compute_sfu"])
            activation.update(elements=elements, source_arithmetic=arithmetic)
            quantize("quantize_activated", routes, width)
    stage("router_weight_cast", {"fp32_weights": 4 * routes, "bf16_weights": 2 * routes})
    stage("shuffle_weighted_sum", {"bf16_expert_output": 2 * routes * hidden,
        "bf16_router_weights": 2 * routes, "bf16_result": 2 * tokens * hidden,
        "int32_output_map": 4 * routes}, scalar=2 * routes * hidden)
    return {"routed_histogram": counts, "routed_rows": tokens * 8, "shared_rows": tokens,
        "expert_rows": expert_rows, "effective_topk": 9, "all_expert_rows": routes,
        "source_stage_order": tuple(item["stage"] for item in stages), "stages": tuple(stages),
        "logical_bytes": sum(item["logical_bytes"] for item in stages),
        "actual_provider_bound": False, "complete_source_service": False,
        "limitations": ("exact_declared_sglang0519_cutlass_caller_math_not_actual_provider",
            "useful_unpadded_FP8_matrix_work_not_issued_MMA_or_runtime_allocation",
            "per_expert_role_payload_not_observed_DRAM_or_cache_traffic",
            "quant_abs_max_shuffle_conversion_clamp_address_and_PDL_services_unpriced",
            "logical_map_role_footprints_not_replayed_lane_loads_or_observed_DRAM",
            "integer_offsets_problem_descriptors_scratch_compiler_and_native_launch_unbound",
            "weight_scale_application_and_finite_precision_services_unpriced",
            "FP32_router_weights_materialize_BF16_before_weighted_sum_not_a_second_output_scale")}


def kimi_k3_marlin_expert_logical_work(kernel: KernelSpec):
    """Standard-format W4A16 Marlin source roles with K3's exact SiTU.

    This is a conditional source program on supplied routing, not an auto
    backend selection or a GPU launch. Counts are live mathematical work;
    alignment, padded MMA, conversion, scratch and reduction implementation
    remain unbound. Repeated latent-role reads are not observed DRAM.
    """
    if (kernel.kind is not KernelKind.FUSED_MOE or kernel.variant != "kimi_k3_marlin_experts"
            or kernel.execution_support != "unsupported" or kernel.launch is not None):
        raise ValueError("K3 Marlin expert work requires an unbound source KernelSpec")
    tokens, experts, topk, hidden, intermediate = (kernel.shape[key] for key in (
        "tokens", "experts", "topk", "hidden", "intermediate"))
    routed = tokens * topk
    active = sum(count > 0 for count in kernel.attributes["tokens_per_expert"])
    operands = []

    def operand(role, dtype, count, access):
        size = (count // 2 if dtype == "e2m1_packed" else
                {"bf16": 2, "fp32": 4, "int32": 4, "ue8m0": 1}[dtype] * count)
        operands.append({"role": role, "dtype": dtype, "elements": count,
                         "bytes": size, "access": access})

    operand("latent_per_route", "bf16", routed * hidden, "read")
    operand("topk_ids", "int32", routed, "read")
    operand("topk_weights", "fp32", routed, "read")
    operand("w13_active_weights", "e2m1_packed", active * 2 * hidden * intermediate, "read")
    operand("w13_active_scales", "ue8m0", active * 2 * intermediate * (hidden // 32), "read")
    operand("w13_output", "bf16", routed * 2 * intermediate, "write")
    operand("situ_input", "bf16", routed * 2 * intermediate, "read")
    operand("situ_output", "bf16", routed * intermediate, "write")
    operand("w2_input", "bf16", routed * intermediate, "read")
    operand("w2_active_weights", "e2m1_packed", active * hidden * intermediate, "read")
    operand("w2_active_scales", "ue8m0", active * hidden * (intermediate // 32), "read")
    operand("weighted_w2_output", "bf16", routed * hidden, "write")
    operand("combine_input", "bf16", routed * hidden, "read")
    operand("combined_latent", "bf16", tokens * hidden, "write")
    return {
        "source": "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:_fused_marlin_moe/fused_marlin_moe",
        "activation_source": "vllm/model_executor/layers/fused_moe/activation.py:apply_moe_activation:MoEActivation.SITU",
        "source_stages": ("route_alignment", "w13", "situ_and_mul", "weighted_w2", "topk_sum"),
        "experts": experts, "routed_rows": routed, "active_experts": active,
        "weight_format": {"values": "e2m1_packed", "scale": "ue8m0", "group_size": 32},
        "activation": {"name": "situ", "beta": kernel.attributes["activation_beta"],
                       "linear_beta": kernel.attributes["activation_linear_beta"],
                       "arithmetic_dtype": "fp32", "input_output_dtype": "bf16",
                       "tanhf_calls": 3 * routed * intermediate,
                       "multiplications_with_precomputed_coefficients": 7 * routed * intermediate,
                       "additions": routed * intermediate},
        "tensor_flops_live": 6 * routed * hidden * intermediate,
        "router_output_weight_multiplications": routed * hidden,
        "topk_sum_additions": tokens * hidden * (topk - 1),
        "apply_router_weight_on_input": False,
        "operands": tuple(operands),
        "workspace_aliases": (("w13_output", "weighted_w2_output"),),
        "alias_lifetimes": "w13 is dead after SiTU; its backing cache13 storage is reused for W2 output",
        "temporary_alignment_workspace_bound": False,
        "source_launch": None, "demand": None, "seconds": None,
        "limitations": (
            "declared_TP1_Marlin_standard_BF16_W4A16_candidate_not_actual_backend_dispatch",
            "histogram_is_supplied_provenance_not_inferred_checkpoint_logits_or_observed_topk",
            "selected_IDs_FP32_weights_are_external_inputs_selection_not_in_this_aggregate",
            "live_tensor_math_excludes_padded_MMA_and_weight_dequantization_issue_services",
            "active_weight_storage_once_not_physical_reloads_cache_or_DRAM",
            "alignment_sorted_IDs_expert_IDs_post_padded_count_and_Marlin_scratch_unbound",
            "SiTU_three_tanhf_calls_not_one_instruction_per_tanh_or_SiLU",
            "FP32_accumulation_and_BF16_W2_weighted_store_rounding_retained_not_numerical_equivalence_proof",
            "several_physical_kernels_source_aggregate_not_one_fused_GPU_launch"),
    }


def kimi_k3_marlin_expert_roofline_work(kernel: KernelSpec):
    source = kimi_k3_marlin_expert_logical_work(kernel)
    routed, active = source["routed_rows"], source["active_experts"]
    tokens, topk = kernel.shape["tokens"], kernel.shape["topk"]
    hidden, intermediate = kernel.shape["hidden"], kernel.shape["intermediate"]
    operands = {operand["role"]: operand for operand in source["operands"]}

    def demand(tensor=0, fma=0, sfu=0):
        return {"compute_tensor": float(tensor), "compute_vector": 0.0,
                "compute_reduction": 0.0, "compute_fma": float(fma),
                "compute_sfu": float(sfu), "compute_alu": 0.0}

    def stage(name, roles, stage_demand):
        return {"stage": name, "demand": stage_demand,
                "logical_bytes": sum(operands[role]["bytes"] for role in roles)}

    activation_values = routed * intermediate
    stages = (
        stage("route_alignment", ("topk_ids",), demand()),
        stage("w13", ("latent_per_route", "w13_active_weights", "w13_active_scales", "w13_output"),
              demand(tensor=4 * routed * hidden * intermediate,
                     fma=4 * active * hidden * intermediate)),
        stage("situ_and_mul", ("situ_input", "situ_output"),
              demand(fma=8 * activation_values, sfu=3 * activation_values)),
        stage("weighted_w2", ("w2_input", "w2_active_weights", "w2_active_scales",
                              "topk_weights", "weighted_w2_output"),
              demand(tensor=2 * routed * hidden * intermediate,
                     fma=2 * active * hidden * intermediate + routed * hidden)),
        stage("topk_sum", ("combine_input", "combined_latent"),
              demand(fma=routed * hidden)),
    )
    return {"stages": stages, "logical_bytes": sum(item["logical_bytes"] for item in stages),
            "source": source["source"], "source_stages": source["source_stages"],
            "tensor_flops_live": source["tensor_flops_live"],
            "dequantization_math": "two_BF16_multiplies_per_active_weight_value_once",
            "topk_source_additions": routed * hidden,
            "topk_minimal_additions": tokens * hidden * (topk - 1),
            "price_composition": "sum_serial_stage_rooflines",
            "limitations": source["limitations"] + (
                "active_weight_dequantization_math_excludes_M_padding_reloads_and_native_issue_services",
                "public_role_bytes_and_live_math_are_an_analytical_lower_bound_not_native_latency_acceptance")}


def _model(
        kernel: KernelSpec, hardware, *, attach_pipeline: bool = True,
        ) -> Tuple[PredictionInput, float]:
    if (kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes
            and kernel.attributes["mathematical_program"] == "sglang_0_5_19_fp8_fused_shared_swiglu_v1"):
        raise NotImplementedError("SGLang0.5.19 mathematical experts have no native launch/resource schedule")
    if kernel.variant == "kimi_k3_marlin_experts":
        raise NotImplementedError("K3 Marlin native launch and resource schedule remain unbound")
    _validate_expert_parallel_launch(kernel)
    if kernel.variant.lower() == "routed_experts":
        return _routed_experts_model(kernel, hardware)
    if _is_fp8_block_launch(kernel):
        return _fp8_block_model(kernel, hardware)
    variant, _, projection_k, projection_n = _projection(kernel)
    _, local_counts = _routing(kernel)
    _, tile_n, _ = _tile(kernel)
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        cann_tensor_work = _cann_grouped_matmul_work(kernel)
        # GroupedMatmul's source order and the physical shared-L2 capacity
        # determine the requests that escape to DRAM.  Preserve that
        # source-derived residency accounting in the compressed analytical
        # envelope; replacing it with a unique-footprint lower bound erases
        # the repeated operand traffic of populated expert tiles.
        cann_residency = _cann_grouped_operand_residency(
            kernel, cann_tensor_work, hardware)
        cann_traffic = _cann_grouped_matmul_traffic(
            kernel, cann_tensor_work, cann_residency)
        gpu_traffic = None
        l2_total = cann_traffic.l2_total_bytes
        dram_bytes = cann_traffic.dram_total_bytes
        compulsory = dram_bytes
    elif memory_model == "gpu_mshr":
        gpu_traffic = _gpu_traffic_components(kernel, hardware)
        l2_total = gpu_traffic.l2_total_bytes
        dram_bytes = gpu_traffic.dram_total_bytes
        compulsory = gpu_traffic.compulsory_bytes
        cann_tensor_work = None
        cann_traffic = None
        cann_residency = None
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")

    routed_rows = float(sum(local_counts))
    if cann_tensor_work is None:
        tile_m, tile_n, tile_k = _tile(kernel)
        populated_m_tiles = len(_gpu_valid_tile_rows(kernel, local_counts))
        # Triton masks operand loads and output stores at the logical M/N
        # boundaries, but tl.dot still issues the complete tile.  Price the
        # physical MMA fragments rather than only the unpadded routed rows.
        # Masked K-tail loads supply zeros but still execute a complete dot.
        tensor_flops = float(
            2
            * populated_m_tiles
            * tile_m
            * (math.ceil(projection_k / tile_k) * tile_k)
            * math.ceil(projection_n / tile_n)
            * tile_n)
    else:
        tensor_flops = float(cann_tensor_work.tensor_operations)
    routed_weight_scale_operations = _source_routed_weight_scale_operations(
        kernel, routed_rows=routed_rows, projection_n=projection_n)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 0.0,
        "compute_alu": routed_weight_scale_operations,
        "compute_sfu": 0.0,
    }
    validated_l2_miss_rate(dram_bytes, l2_total)
    demand = build_forward_demand(l2_total, dram_bytes, False, flops)
    active_workgroups = None
    if memory_model == "gpu_mshr":
        # max-EM reserves expert padding that need not be populated. Triton
        # returns before operand loads/MMA/stores beyond num_tokens_post_padded.
        # Use the same actual routing census as traffic and tensor work for
        # capacity; retain the original physical launch in PredictionInput.
        global_counts, _ = _routing(kernel)
        active_workgroups = len(_gpu_valid_tile_rows(kernel, global_counts)) * math.ceil(projection_n / tile_n)
    eta, wave_efficiency, active_units = (
        launch_wave_geometry(
            kernel, hardware, active_workgroups=active_workgroups)
    )
    dram_efficiency = hardware.dram_protocol_efficiency_l1cached
    if memory_model == "npu_tile":
        pass
    elif memory_model == "gpu_mshr":
        n_tiles = math.ceil(projection_n / tile_n)
        if kernel.launch.grid_size % n_tiles:
            raise ValueError(
                "FUSED_MOE launch grid must be divisible by N-tile count")
    else:
        raise AssertionError(
            f"unhandled FUSED_MOE memory model: {memory_model!r}")

    if memory_model == "npu_tile":
        launch_task_kind = "aic"
        include_launch_floor, launch_limitations = (
            npu_launch_floor_contract(hardware, launch_task_kind))
        support_level = "aggregate_only"
        limitations = (
            "aic_grouped_matmul_dram_residency_uses_source_order_lru",
            "aic_output_writeback_dram_residency_unavailable",
            "aic_store_fixpipe_tile_lifecycle_unavailable",
            "aic_read_write_transaction_split_unavailable",
            "mte_transaction_geometry_unavailable",
            "memory_first_request_latency_unavailable",
            "mte2_descriptor_and_credit_evidence_unavailable",
            "workload_producer_independence_unavailable",
        ) + launch_limitations
    elif memory_model == "gpu_mshr":
        launch_task_kind = "generic"
        include_launch_floor = True
        support_level = "aggregate_only"
        limitations = gpu_partial_limitations(
            hardware,
            "moe_gather_service_class_not_separated",
            "moe_physical_worker_assignment_unavailable",
            "moe_l2_residency_order_unavailable",
            "moe_load_compute_store_buffer_lifecycle_unavailable",
            "moe_early_return_metadata_service_unmodeled",
        )
        if _gpu_uses_naive_assignment(kernel):
            limitations += (
                "moe_naive_assignment_preserves_one_matrix_tile_per_route",
                "moe_naive_token_order_expert_ids_unbound_compulsory_dram_lower_bound",)
        elif any(local_counts):
            limitations += ((
                "moe_cache_replay_uses_observed_resident_capacity_in_uniform_k_waves"
                if _gpu_observed_resident_ctas(kernel, hardware) is not None else
                "moe_cache_replay_one_cta_window_without_complete_resource_observation"),)
    else:
        raise AssertionError(
            f"unhandled FUSED_MOE memory model: {memory_model!r}")

    if memory_model == "npu_tile":
        if cann_tensor_work is None:
            raise AssertionError("CANN GroupedMatmul work is missing")
        if cann_traffic is None:
            raise AssertionError("CANN GroupedMatmul physical traffic is missing")
        if cann_residency is None:
            raise AssertionError("CANN GroupedMatmul residency is missing")
        return (
            _cann_grouped_matmul_analytical_input(
                kernel,
                hardware,
                demand_per_resource=demand,
                eta=eta,
                wave_efficiency=wave_efficiency,
                active_units=active_units,
                include_dispatch_overhead=include_launch_floor,
                limitations=limitations,
                traffic=cann_traffic,
                work=cann_tensor_work,
                residency=cann_residency,
                attach_pipeline=attach_pipeline,
            ),
            compulsory,
        )
    if gpu_traffic is None:
        raise AssertionError("GPU grouped-GEMM traffic is missing")
    nvidia_source_service = _nvidia_source_depth_operand_service(
        kernel, hardware)
    source_completion_services = {
        _NVIDIA_BM16_H2880_31_ACTIVE_SERVICE,
        _NVIDIA_BM16_H2880_53_ACTIVE_SERVICE,
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_88_ACTIVE_128_MAX_W2_SERVICE,
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W1_SERVICE,
        _NVIDIA_BM16_E128_177_ACTIVE_248_MAX_W2_SERVICE,
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_7_ACTIVE_8_MAX_W2_SERVICE,
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W1_SERVICE,
        _NVIDIA_BM16_E8_H6144_7_ACTIVE_8_MAX_W2_SERVICE,
    }
    pipeline_plan = (
        None
        if (nvidia_source_service is not None
            and nvidia_source_service.service in source_completion_services)
        else (
            _gpu_grouped_gemm_pipeline_envelope(
                kernel, hardware, gpu_traffic)
            or _metax_grouped_operand_pipeline_envelope(
                kernel, hardware, gpu_traffic)))
    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=variant,
        mem_model=memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        launch_task_kind=launch_task_kind,
        include_dispatch_overhead=include_launch_floor,
        dram_capacity_multiplier=dram_efficiency,
        support_level=support_level,
        limitations=limitations,
    )
    normalized = _architecture_input(inp, kernel, hardware)
    normalized = _apply_h200_fused_moe_w2_active_sm_clock(
        normalized, kernel, hardware)
    if memory_model != "gpu_mshr":
        raise AssertionError(
            f"unhandled FUSED_MOE memory model: {memory_model!r}")
    producers, requests_per_producer = gpu_memory_source(hardware)
    transaction_bytes = None
    path_name = "global_memory"
    frontend_rate = None
    structured = attach_structured_contract(
        normalized,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name=path_name,
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=frontend_rate,
    )
    structured = _apply_metax_bm32_fused_moe_mma_occupancy(
        structured, kernel, hardware)
    if l2_total > 0:
        if gpu_traffic is None:
            raise AssertionError("GPU grouped-GEMM traffic is missing")
        structured = _split_gpu_grouped_gemm_memory_paths(
            structured, gpu_traffic)
        tm, tn, tk = _tile(kernel)
        depth = kernel.launch.tiling.pipeline_depth
        parameters = kernel.launch.parameters
        # Only rank-local routes issue operand reads. Nonlocal zero-fill
        # output remains in the original write path and cannot qualify a
        # cold B-read mechanism for an empty local expert rank.
        valid_rows = _gpu_valid_tile_rows(kernel, local_counts)
        if (depth is not None and valid_rows
                and all(1 <= row <= tm for row in valid_rows)
                and "dynamic_shared_memory_bytes" in parameters
                and parameters["dynamic_shared_memory_bytes"]
                == depth * (tm + tn) * tk * _gpu_operand_element_bytes(kernel)):
            structured = attach_finite_async_copy_progress_floor(
                structured, hardware, read_path_name="weight_read",
                dtype=kernel.dtype, tile_m=tm, tile_k=tk,
                panel_bytes=tn * tk * _gpu_operand_element_bytes(kernel),
                buffer_slots=depth,
                row_stride_bytes=projection_k * _gpu_operand_element_bytes(kernel))
            progress_residents=_gpu_observed_resident_ctas(kernel,hardware)
            if (progress_residents is None and parameters['dynamic_shared_memory_bytes']
                    > hardware.max_smem_per_unit_bytes/2):
                # Preserve the old shared-only proof for legacy callers.
                progress_residents=1
            progress_available=False
            if progress_residents is not None:
                resident_tag='' if progress_residents==1 else f'_resident{progress_residents}'
                service_name = (f"coalesced_async_copy_threads{kernel.launch.block_size}"
                    f"_slots{depth}_b{tn*tk*_gpu_operand_element_bytes(kernel)}{resident_tag}_fanout1")
                progress_available=any(s.service==service_name for s in hardware.memory_access_class_services)
            if (kernel.attributes["expert_parallel_size"] == 1
                    and not _gpu_uses_naive_assignment(kernel)
                    and progress_available):
                requesting, unique, maximum = _compulsory_weight_request_topology(
                    local_counts, tile_m=tm, tile_n=tn,
                    projection_k=projection_k,projection_n=projection_n,
                    element_bytes=_gpu_operand_element_bytes(kernel),
                    group_size_m=int(parameters["group_size_m"]),
                    wave_width=int(hardware.num_compute_units)*progress_residents)
                structured = attach_coalesced_async_copy_progress_floor(
                    structured,hardware,read_path_name="weight_read",dtype=kernel.dtype,
                    tile_m=tm,tile_k=tk,panel_bytes=tn*tk*_gpu_operand_element_bytes(kernel),
                    buffer_slots=depth,row_stride_bytes=projection_k*_gpu_operand_element_bytes(kernel),
                    requesting_B_bytes=requesting,unique_B_bytes=unique,maximum_fanout=maximum,
                    resident_ctas_per_unit=progress_residents)
        if routed_rows > 0:
            structured = _attach_gpu_gather_service(
                structured,
                kernel,
                hardware,
                traffic=gpu_traffic,
                routed_rows=routed_rows,
                projection_k=projection_k,
                projection_n=projection_n,
                element_bytes=_gpu_operand_element_bytes(kernel),
            )
    structured = dataclasses.replace(structured, pipeline_plan=pipeline_plan)
    structured = _attach_prepared_expert_operand_pipeline(structured, kernel, hardware)
    if pipeline_plan is None:
        structured = _attach_resident_operand_pipeline(structured, kernel, hardware)
    preparation_operations = 0.0
    if "matmul_post_wait_matrix_preparation_issue_from_source" in structured.limitations:
        preparation = observed_fused_moe_matrix_preparation_issue(kernel, hardware,
            assignment_mode="naive" if _gpu_uses_naive_assignment(kernel) else "aligned")
        if preparation is None or active_workgroups is None:
            raise ValueError("source-bound matrix preparation lost its physical work contract")
        count, _ = preparation
        preparation_operations = float(count * active_workgroups
            * math.ceil(projection_k/tile_k)
            * math.ceil(kernel.launch.block_size/hardware.parallel_width))
    conversion_operations = 0.0
    if "matmul_output_packed_conversion_issue_from_compiled_source" in structured.limitations:
        per_task_conversion = source_fused_moe_output_conversion_warp_instructions(
            kernel, hardware)
        if per_task_conversion is None or active_workgroups is None:
            raise ValueError("compiled output conversion lost its physical work contract")
        conversion_operations = float(active_workgroups * per_task_conversion)
    structured = dataclasses.replace(structured,
        phase_ledger=_grouped_matmul_phase_ledger(structured,
            matrix_preparation_issue_operations=preparation_operations,
            output_conversion_issue_operations=conversion_operations))
    structured, router_compulsory = _attach_gpu_bf16_w2_router_epilogue(
        structured, kernel, hardware)
    compulsory += router_compulsory
    return structured, compulsory


class FusedMoEPredictor(KernelPredictor):
    kind = KernelKind.FUSED_MOE

    def __init__(self):
        self._cached_resource_kernel = None
        self._cached_resource_hardware = None
        self._cached_resource_value = None

    def analytical_without_launch(self, kernel: KernelSpec, hardware):
        if kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes:
            prepared = _prepared_ep_reference_prediction(kernel, hardware)
            if prepared is not None:
                return prepared
        if (kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "sglang_0_5_19_fp8_fused_shared_swiglu_v1"):
            return dataclasses.replace(self.roofline_without_launch(kernel, hardware),
                bottleneck_group="analytical_sglang_source")
        if (kernel.variant == "kimi_k3_marlin_experts"
                or (kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes)):
            prediction = self.roofline_without_launch(kernel, hardware)
            return dataclasses.replace(prediction, bottleneck_group="analytical_marlin_source")
        return super().analytical_without_launch(kernel, hardware)

    def resource_input(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "sglang_0_5_19_fp8_fused_shared_swiglu_v1"):
            raise NotImplementedError("SGLang0.5.19 mathematical experts have no native launch/resource schedule")
        if (kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes
                and kernel.attributes["mathematical_program"] == "kimi_ep_marlin_swiglu_v1"):
            raise NotImplementedError("Kimi mathematical expert shards have no native launch/resource schedule")
        from src.kernel.prediction_cache import _retain_public_resource_input
        value = _routed_public_resource_input(self, kernel, hardware)
        _retain_public_resource_input(kernel, hardware, value)
        return value

    def forward_resource_input(self, kernel: KernelSpec, hardware):
        # These prepared source stages have only the generic logical-dot
        # projection: its materialized TilePrograms supply exactly projection
        # dimensions and the unchanged global/local histograms to _model.
        # None of their selected views, event names or logical read/write
        # totals enter the numerical price. Keep that public source projection
        # available, while Forward evaluates its identical numeric contract.
        # Explicit operand bindings and an enclosing source-work producer must
        # retain their ordinary boundary checks.
        from contextvars import copy_context
        from src.kernel.memory_access_builder import _PREDICTION_MEMORY_BINDINGS
        from src.kernel.memory_fused_moe import _SOURCE_WORK
        context = copy_context()
        unbound = _PREDICTION_MEMORY_BINDINGS not in context
        if not unbound:
            invocation = context[_PREDICTION_MEMORY_BINDINGS]
            unbound = (invocation is not None and invocation.implicit
                and invocation.kernel is kernel and invocation.bindings is None)
        if (type(kernel) is KernelSpec and type(hardware) is HWSpec
                and hardware.hardware_id == kernel.hardware_id
                and hardware.prepared_moe_operand_resources is not None
                and kernel.dtype == "bf16" and kernel.variant in ("w1", "w2")
                and kernel.execution_support == "supported" and kernel.launch is not None
                and kernel.launch.algorithm == f"vllm_0_30_sm90_unquantized_aligned_{kernel.variant}"
                and kernel.attributes["expert_parallel_size"] == 2
                and all(name in kernel.launch.parameters for name in (
                    "registers_per_thread", "static_shared_bytes", "dynamic_shared_memory_bytes"))
                and unbound and _SOURCE_WORK not in context):
            from src.kernel.prediction_cache import _retain_public_resource_input
            value = self._resource_input(kernel, hardware)
            _retain_public_resource_input(kernel, hardware, value)
            return value
        return self.resource_input(kernel, hardware)

    def _source_resource_input(self, kernel: KernelSpec, hardware):
        from src.kernel.memory_fused_moe import has_fused_moe_source, fused_moe_source_work
        from src.kernel.memory_fused_moe import has_native_grouped_moe_analytical_work
        from src.kernel.memory_fused_moe_fp8 import has_fp8_fused_moe_source, fp8_fused_source_demands
        from src.kernel.memory_routed_moe import has_routed_moe_source, routed_moe_source_demands
        if has_routed_moe_source(kernel,hardware):
            from src.kernel.memory_access_builder import fused_moe_memory_access_program
            work = routed_moe_source_demands(fused_moe_memory_access_program(kernel,hardware),expected_kernel=kernel,expected_hardware=hardware)
            with fused_moe_source_work(kernel,work):
                return self._resource_input(kernel,hardware)
        if has_fp8_fused_moe_source(kernel):
            from src.kernel.memory_access_builder import fused_moe_memory_access_program
            work = fp8_fused_source_demands(fused_moe_memory_access_program(kernel,hardware),expected_kernel=kernel)
            with fused_moe_source_work(kernel,work):
                return self._resource_input(kernel,hardware)
        if has_native_grouped_moe_analytical_work(kernel,hardware):
            from src.kernel.memory_access_builder import fused_moe_prediction_work
            work=fused_moe_prediction_work(kernel,hardware)
            with fused_moe_source_work(kernel,work):
                return self._resource_input(kernel,hardware)
        if has_fused_moe_source(kernel,hardware):
            from src.kernel.memory_access_builder import fused_moe_memory_access_program
            from src.kernel.memory_path_solver import fused_moe_source_demands
            work = fused_moe_source_demands(fused_moe_memory_access_program(kernel,hardware))
            with fused_moe_source_work(kernel,work):
                return self._resource_input(kernel,hardware)
        return self._resource_input(kernel,hardware)

    def _resource_input(self, kernel: KernelSpec, hardware):
        # One evaluator case first materializes this analytical input for its
        # Formula-Ceiling observation, then immediately invokes Forward and
        # Roofline for the exact same immutable KernelSpec.  The source-derived
        # L2 replay and grouped-matmul envelope are independent of the method,
        # so retain just that one current-object construction.  Return fresh
        # outer mappings so callers cannot mutate the retained input; object
        # identity keeps the cache strictly within the currently evaluated
        # spec rather than becoming a shape/result lookup table.
        if (self._cached_resource_kernel is kernel
                and self._cached_resource_hardware is hardware):
            if self._cached_resource_value is None:
                raise RuntimeError("FusedMoE resource-input cache is incomplete")
            cached_input, cached_compulsory = self._cached_resource_value
            return dataclasses.replace(
                cached_input,
                demand_per_resource=dict(cached_input.demand_per_resource),
                resource_time_overrides_ns=dict(
                    cached_input.resource_time_overrides_ns),
            ), cached_compulsory
        resource_input, compulsory = _model(kernel, hardware)
        self._cached_resource_kernel = kernel
        self._cached_resource_hardware = hardware
        self._cached_resource_value = (resource_input, compulsory)
        return resource_input, compulsory

    def roofline_without_launch(self, kernel: KernelSpec, hardware):
        if kernel.variant == "routed_experts" and "mathematical_program" in kernel.attributes:
            if kernel.attributes["mathematical_program"] == "sglang_0_5_19_fp8_fused_shared_swiglu_v1":
                work = sglang0519_fp8_fused_shared_logical_work(kernel)
                bandwidth = float(hardware.resource_by_role("dram").capacity)
                fp8_capacity = float(hardware.resource_by_role("compute_tensor_fp8").capacity)
                times = {part["stage"]: (max(part["mathematical_fp8_flops"] / fp8_capacity,
                    part["logical_bytes"] / bandwidth) if part["mathematical_fp8_flops"] else
                    roofline_ns(part["demand"], hardware, part["logical_bytes"]) * 1e-9)
                    for part in work["stages"]}
                prediction = prediction_from_ns(sum(times.values()) * 1e9, "roofline")
                return dataclasses.replace(prediction, t_per_group=times,
                    diagnostics=PredictionDiagnostics(compute_time_s={}, memory=(), memory_access_classes=(),
                        memory_arbitration=(), pipeline=None, critical_path=work["source_stage_order"],
                        support_level="partial", limitations=work["limitations"]))
            if kernel.attributes["mathematical_program"] == "dsv4_mxfp4_w4a16_swiglu_v1":
                work = dsv4_marlin_moe_source_work(kernel)
                if (HWSpec.source_contract_supported("aggregate_gpu_source_roofline", hardware.hardware_id)
                        and hardware.gpu_vector_memory_only):
                    # W4A16 retains its existing BF16 tensor peak, not the
                    # FP8 peak. Activation/dequant/reduction issue is omitted
                    # only by this explicit approximation; bytes are retained.
                    ns = sum(roofline_ns({role: amount if role == "compute_tensor" else 0
                        for role, amount in stage["demand"].items()}, hardware,
                        sum(stage["read_write_bytes"].values())) for stage in work["stages"])
                    return prediction_from_ns(ns, "roofline")
                missing = sorted({role for stage in work["stages"] for role, amount in stage["demand"].items()
                    if amount > 0 and role not in hardware._role_map})
                if missing:
                    raise NotImplementedError(f"BW1100 W4A16 expert source Roofline requires {tuple(missing)}; typed dequantization/router-weight and native provider remain separate")
                return prediction_from_ns(sum(roofline_ns(stage["demand"], hardware,
                    sum(stage["read_write_bytes"].values())) for stage in work["stages"]), "roofline")
            if kernel.attributes["mathematical_program"] == "kimi_ep_marlin_swiglu_v1":
                work = kimi_ep_marlin_logical_work(kernel)
                return prediction_from_ns(sum(roofline_ns(part["demand"], hardware, part["logical_bytes"])
                    for part in work["stages"]), "roofline")
            bf16 = kernel.attributes["mathematical_program"] in ("unquantized_bf16_swiglu_v1",
                "unquantized_bf16_grouped_swiglu_v1", "unquantized_bf16_legacy_grouped_swiglu_v1")
            work = unquantized_bf16_experts_logical_work(kernel) if bf16 else block_fp8_experts_logical_work(kernel)
            vector = VectorPredictor()
            seconds = 0.0
            for rows, columns in work["quantization_inputs"]:
                quant = KernelSpec(kind=KernelKind.VECTOR, hardware_id=kernel.hardware_id,
                    shape={"element_count": rows * columns, "rows": rows, "columns": columns}, dtype="bf16",
                    variant="fp8_group_quant", attributes={"group_size": 128,
                        "source_program": "vllm_0_30_group_fp8_fp32_scales"},
                    launch=None, execution_support="unsupported", unsupported_reason="launch_selector_unavailable")
                seconds += vector.roofline_without_launch(quant, hardware).t_total_s
            bandwidth = float(hardware.resource_by_role("dram").capacity)
            seconds += work["output_initialization_bytes"] / bandwidth
            for stage in work["matrix_stages"]:
                seconds += max(stage["mathematical_tensor_flops" if bf16 else "mathematical_fp8_flops"]
                    / float(hardware.resource_by_role("compute_tensor" if bf16 else "compute_tensor_fp8").capacity),
                    stage["logical_bytes"] / bandwidth)
            demand = {role: 0 for role in ("compute_tensor", "compute_vector", "compute_reduction", "compute_fma", "compute_sfu", "compute_alu")}
            activation = work["activation"]
            demand.update(activation["source_arithmetic"]["baseline_demand"])
            seconds += roofline_ns(demand, hardware, activation["logical_bytes"]) * 1e-9
            reduction = work["weighted_sum"]
            demand["compute_fma"] = reduction["multiply_flops"] + reduction["addition_flops"]
            demand["compute_sfu"] = 0
            seconds += roofline_ns(demand, hardware,
                reduction["logical_bytes"] + work["routing_id_payload_read_bytes"]) * 1e-9
            return prediction_from_ns(seconds * 1e9, "roofline")
        if kernel.variant == "kimi_k3_marlin_experts":
            work = kimi_k3_marlin_expert_roofline_work(kernel)
            return prediction_from_ns(sum(roofline_ns(stage["demand"], hardware,
                stage["logical_bytes"]) for stage in work["stages"]), "roofline")
        return super().roofline_without_launch(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "routed_experts" and kernel.attributes["swiglu_limit"] is not None
                and kernel.attributes["expert_parallel_size"] == 1):
            work = dsv4_marlin_moe_roofline_work(kernel, hardware)
            limitations = (
                "v4_marlin_M_padded_tensor_live_scalar_serial_roofline_only",
                "declared_routing_not_checkpoint_assignment",
                "marlin_integer_atomic_metadata_consumer_scratch_unbound",
                "marlin_dequantization_compiler_and_provider_services_unbound",
                "bf16_router_weight_math_counted_packed_multiply_and_conversion_service_unpriced",
                "logical_stages_share_one_declared_aggregate_launch_boundary")
            formula = PerformanceFormula(hardware)
            stages = []
            for stage in work["stages"]:
                total = sum(stage["read_write_bytes"].values())
                demand = build_forward_demand(total, total, False, stage["demand"])
                inp = build_aggregate_gpu_input(kernel, hardware, demand,
                    write_l2_bytes=stage["read_write_bytes"]["write"],
                    write_dram_bytes=stage["read_write_bytes"]["write"],
                    limitations=limitations)
                stages.append((stage["stage"], formula.predict_kernel(dataclasses.replace(
                    _with_graph_regime(inp, kernel), include_dispatch_overhead=False))))
            aggregate_inp = _with_graph_regime(self.resource_input(kernel, hardware)[0], kernel)
            aggregate_execution = formula.predict_kernel(dataclasses.replace(
                aggregate_inp, include_dispatch_overhead=False))
            aggregate_full = formula.predict_kernel(aggregate_inp)
            launch_seconds = max(0.0, aggregate_full.t_total_s - aggregate_execution.t_total_s)
            aggregate_launch = dataclasses.replace(aggregate_full,
                t_total_s=launch_seconds,
                t_per_group={"aggregate_launch": launch_seconds},
                bottleneck_group="aggregate_launch",
                t_per_resource={"aggregate_launch": launch_seconds},
                diagnostics=dataclasses.replace(aggregate_full.diagnostics,
                    compute_time_s={},
                    memory=(),
                    memory_access_classes=(),
                    memory_arbitration=(),
                    pipeline=None,
                    critical_path=("aggregate_launch",),
                    limitations=aggregate_full.diagnostics.limitations + (
                        "aggregate_launch_boundary_is_exposed_increment_not_another_physical_kernel",)))
            return sum_serial_kernel_predictions((("aggregate_launch", aggregate_launch), *stages))
        inp, _ = self.forward_resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(
            _with_graph_regime(inp, kernel))

    def roofline(self, kernel: KernelSpec, hardware):
        if (kernel.variant == "routed_experts" and kernel.attributes["swiglu_limit"] is not None
                and kernel.attributes["expert_parallel_size"] == 1):
            work = dsv4_marlin_moe_roofline_work(kernel, hardware)
            return prediction_from_ns(sum(roofline_ns(stage["demand"], hardware,
                sum(stage["read_write_bytes"].values())) for stage in work["stages"]), "roofline")
        # Roofline consumes only aggregate demand and compulsory traffic, both
        # of which are identical in the complete Forward input.  Reusing it
        # avoids rebuilding the source residency census and then dropping its
        # pipeline representation without changing the roofline calculation.
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory), "roofline")

    def pipeweave(self, kernel: KernelSpec, hardware, *, model_hardware=None):
        if kernel.variant == "routed_experts" and kernel.attributes["swiglu_limit"] is not None:
            raise NotImplementedError("clamped routed experts supports Roofline only")
        if kernel.attributes["expert_parallel_size"] != 1:
            raise NotImplementedError(
                "PipeWeave FUSED_MOE does not model rank-local expert "
                "parallel execution")
        from src.kernel.predictors.pipeweave_adapter import predict_fused_moe
        return predict_fused_moe(
            kernel, hardware, model_hardware=model_hardware)




ROUTING_ORDER_SENSITIVE_RECIPE_VARIANTS = frozenset((
    # This kernel walks the routing histogram as a cumulative prefix stream and
    # resolves a sorted position back to the expert index that produced it, so
    # it reads the axis in a finer way than the grouped projections do and the
    # relabelling below is not established for it.
    "moe_compute_expert_tokens",
))


def _expert_axis_class(counts, owned_experts):
    """Decompose one routing histogram into its physical class.

    The grouped projection turns each populated expert of this rank's shard
    into M tiles, in launch order; every other term the predictor derives from
    the axis is a sum over the whole axis or over its populated entries.  The
    class is therefore the shard's populated counts in launch order, how many
    of its experts stayed empty, and the multiset of the counts outside it.
    """

    if not owned_experts:
        return (0, -1, (), 0, tuple(sorted((count for count in counts if count), reverse=True)),
            sum(count == 0 for count in counts))
    shard_first = owned_experts[0]
    shard_last = owned_experts[-1]
    if shard_last - shard_first != len(owned_experts) - 1:
        raise ValueError(
            "FUSED_MOE expert shard must be a contiguous expert range")
    shard_populated = []
    shard_empty = 0
    other_populated = []
    other_empty = 0
    for expert, count in enumerate(counts):
        if shard_first <= expert <= shard_last:
            if count:
                shard_populated.append(count)
            else:
                shard_empty += 1
        elif count:
            other_populated.append(count)
        else:
            other_empty += 1
    other_populated.sort(reverse=True)
    return (
        shard_first,
        shard_last,
        tuple(shard_populated),
        shard_empty,
        tuple(other_populated),
        other_empty,
    )


def canonical_expert_axis(tokens_per_expert, owned_experts):
    """Return the canonical routing of this histogram's physical class.

    The canonical member keeps each populated shard expert's rows in launch
    order and moves the shard's empty experts behind them, and orders the
    counts outside the shard by size.  The relabelling never crosses the shard
    boundary, so the contiguous expert placement, the local-equals-global mask
    inside the shard, and ``tokens * topk`` conservation all survive.
    """

    counts = tuple(tokens_per_expert)
    owned = tuple(owned_experts)
    (
        shard_first, shard_last, shard_populated, shard_empty,
        other_populated, other_empty,
    ) = _expert_axis_class(counts, owned)
    shard_values = iter(shard_populated + (0,) * shard_empty)
    other_values = iter(other_populated + (0,) * other_empty)
    canonical = tuple(
        next(shard_values) if shard_first <= expert <= shard_last
        else next(other_values)
        for expert in range(len(counts))
    )
    canonical_local = tuple(
        count if shard_first <= expert <= shard_last else 0
        for expert, count in enumerate(canonical)
    )
    return canonical, canonical_local


def prediction_equivalence_attributes(kernel: KernelSpec):
    """Name the physical routing class of one grouped-MoE launch.

    Every term this predictor derives from the routing histogram is one of
    three things: a sum over the whole expert axis, a sum over the populated
    experts, or the ordered sequence of populated local experts that the
    grouped launch turns into M tiles.  An expert index itself never enters a
    byte count, a tile count, or a service: it reaches the replay only as the
    label that separates one expert's weight objects from another's.

    The class therefore replaces the three expert-indexed contract fields.
    The shard is the contiguous placement of ``expert_rank`` inside
    ``expert_parallel_size`` and ``experts``, all of which the surrounding
    contract still carries, so the class plus that contract names exactly one
    physical routing up to the relabelling the hardware cannot observe.

    Returns ``None`` when the launch carries a per-expert payload that this
    relabelling would not carry with it.
    """

    if kernel.kind is not KernelKind.FUSED_MOE or kernel.launch is None:
        return None
    attributes = kernel.attributes
    if attributes["expert_weight_tile_rows"] is not None:
        # A per-expert residency window is indexed by expert, so the class
        # would have to carry it through the same relabelling.  No launch
        # reaching this path declares one; keep the exact contract instead.
        return None
    equivalence = {
        name: value for name, value in attributes.items()
        if name not in (
            "tokens_per_expert", "local_tokens_per_expert", "owned_experts")
    }
    equivalence["expert_axis_class"] = _expert_axis_class(
        tuple(attributes["tokens_per_expert"]),
        tuple(attributes["owned_experts"]),
    )
    return equivalence


class _RoutedHardwareData:
    """Retain every native HW data edge, including nested configuration data."""

    def __init__(self, hardware):
        from src.kernel.hardware_spec import Resource, ResourceGroup
        from src.kernel.model_contract import MemoryAccessClassSignature
        allowed = (HWSpec, Resource, ResourceGroup, MemoryAccessClassService,
            MemoryAccessClassSignature)
        self.rows, self.records, self.supported = [], [], True
        seen, pending = set(), [hardware]
        while pending:
            value = pending.pop()
            kind = type(value)
            if kind in (str, int, float, bool, bytes, type(None)):
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            if kind is dict:
                keys, values = tuple(value), tuple(value.values())
                self.rows.append((value, kind, keys, values))
                pending.extend(keys)
                pending.extend(values)
            elif kind is list:
                self.rows.append((value, kind, (), tuple(value)))
                pending.extend(value)
            elif kind is tuple:
                pending.extend(value)
            elif kind in allowed:
                state = vars(value)
                self.records.append((value, kind, state))
                pending.append(state)
            else:
                self.supported = False
                break

    def unchanged(self):
        if not self.supported:
            return False
        for value, kind, state in self.records:
            if type(value) is not kind or vars(value) is not state:
                return False
        for value, kind, keys, values in self.rows:
            if type(value) is not kind or len(value) != len(values):
                return False
            if kind is dict:
                if (any(actual is not old for actual, old in zip(value, keys))
                        or any(actual is not old for actual, old in zip(value.values(), values))):
                    return False
            elif any(actual is not old for actual, old in zip(value, values)):
                return False
        return True


def _routed_source_constant(value, active=()):
    """Snapshot native mutable source constants/defaults without object repr."""
    from enum import Enum
    from types import MappingProxyType
    kind = type(value)
    if kind in (str, int, bool, bytes, type(None)):
        return kind, value
    if kind is float:
        return kind, value.hex()
    if value is Ellipsis or value is dataclasses.MISSING or value is dataclasses._HAS_DEFAULT_FACTORY:
        # Generated dataclass constructors use these only as identity tokens.
        return kind, id(value)
    if isinstance(value, type):
        return type, value
    if any(value is parent for parent in active):
        raise TypeError("cyclic routed source dependency")
    active = (*active, value)
    if isinstance(value, Enum):
        return kind, _routed_source_constant(vars(value), active)
    if kind in (dict, MappingProxyType):
        return kind, tuple((_routed_source_constant(key, active), _routed_source_constant(item, active))
            for key, item in value.items())
    if kind in (tuple, list):
        return kind, tuple(_routed_source_constant(item, active) for item in value)
    if kind in (set, frozenset):
        return kind, frozenset(_routed_source_constant(item, active) for item in value)
    raise TypeError("unsupported mutable routed source dependency")


class _RoutedPublicSource:
    """Canonical Python bindings, functions and descriptor layouts for reuse.

    A changed source disables this optimization, rather than granting the new
    implementation the old expert-axis equivalence proof.
    """

    def __init__(self):
        from importlib import import_module
        from functools import cached_property
        from types import FunctionType
        self.modules = tuple(import_module(name) for name in (
            __name__, "src.kernel.predictors.common", "src.kernel.hardware_spec",
            "src.kernel.kernel_spec", "src.kernel.model_contract",
            "src.kernel.launch_config.launch_config", "src.kernel.launch_config.tiling_spec",
            "src.kernel.launch_config.fused_moe_tiling",
            "src.kernel.memory_access", "src.kernel.memory_access_builder",
            "src.kernel.memory_fused_moe", "src.kernel.memory_routed_moe",
            "src.kernel.memory_fused_moe_naive",
            "src.kernel.memory_moe_alignment", "src.kernel.memory_path_solver")) + fused_moe_output_relayout_sources()
        self.namespaces = ()
        self.classes, self.functions = [], []
        classes, functions = set(), set()
        for module in self.modules:
            for value in vars(module).values():
                if isinstance(value, type):
                    classes.update(value.__mro__)
                elif type(value) is FunctionType:
                    functions.add(value)
        for cls in classes:
            self.classes.append((cls, cls.__mro__, self._class_members(cls)))
            for member in vars(cls).values():
                if isinstance(member, (classmethod, staticmethod)):
                    member = member.__func__
                elif isinstance(member, property):
                    functions.update(f for f in (member.fget, member.fset, member.fdel)
                        if type(f) is FunctionType)
                elif isinstance(member, cached_property):
                    member = member.func
                if type(member) is FunctionType:
                    functions.add(member)
        pending = list(functions)
        while pending:
            function = pending.pop()
            if "__wrapped__" in vars(function):
                wrapped = vars(function)["__wrapped__"]
                if type(wrapped) is FunctionType and wrapped not in functions:
                    functions.add(wrapped)
                    pending.append(wrapped)
        self.functions = tuple((function, function.__code__, function.__defaults__,
            None if function.__kwdefaults__ is None else dict(function.__kwdefaults__))
            for function in functions)
        self.defaults, self.unsupported_defaults = [], []
        for function in functions:
            # Imported ABC methods (for example MutableMapping.pop) are not
            # called by the routed builder; their opaque stdlib sentinels do
            # not become dependencies of this source-specific reuse proof.
            if not function.__module__.startswith("src.kernel."):
                continue
            try:
                snapshot = _routed_source_constant((function.__defaults__, function.__kwdefaults__))
            except TypeError:
                self.unsupported_defaults.append(function.__qualname__)
            else:
                self.defaults.append((function, snapshot))
        # Mutable semantic reads in this exact routed source path. Other
        # fused/common pipeline tables are outside the admitted branch. These
        # dtype, identity and service-schema tables affect constructors and
        # validation even when no final traffic quantity reads them directly.
        from src.kernel import hardware_spec, kernel_spec, memory_access, model_contract
        self.constants = tuple((module, name, _routed_source_constant(vars(module)[name]))
            for module, names in ((hardware_spec, ("_SOURCE_CONTRACT_HARDWARE",)),
                (kernel_spec, ("DTYPE_BYTES",)),
                (memory_access, ("_SOURCE_IDENTITY_FIELDS",)),
                (model_contract, ("TYPED_COMPUTE_SERVICES_BY_ROLE", "_TYPED_COMPUTE_SERVICES",
                    "COMPUTE_EXACT_REQUEST_SERVICES_BY_ROLE", "_COMPUTE_EXACT_REQUEST_SERVICES",
                    "MEMORY_EXACT_REQUEST_SERVICES_BY_ROLE"))) for name in names)

    @staticmethod
    def _class_members(cls):
        members = dict(vars(cls))
        if "__slotnames__" in members:
            if type(members["__slotnames__"]) is not list or members["__slotnames__"]:
                return None
            del members["__slotnames__"]
        return members

    def seal(self):
        self.namespaces = tuple((module, self._namespace(module)) for module in self.modules)

    @staticmethod
    def _namespace(module):
        # These tokens memoize the source inspection itself. Their issuer's
        # function and every inspected namespace remain guarded separately.
        ignored = ("_default_source_epoch", "_default_source_rule", "_source_binding_epoch",
            "_source_recipe_signature", "_source_recipe_epoch")
        return {name: value for name, value in vars(module).items()
            if module.__name__ != "src.kernel.memory_access_builder" or name not in ignored}

    def unchanged(self):
        if self.unsupported_defaults:
            return False
        try:
            data_unchanged = (all(_routed_source_constant((function.__defaults__,
                    function.__kwdefaults__)) == snapshot for function, snapshot in self.defaults)
                and all(name in vars(module) and _routed_source_constant(vars(module)[name]) == snapshot
                    for module, name, snapshot in self.constants))
        except TypeError:
            return False
        return (data_unchanged and all(len(self._namespace(module)) == len(old)
                and all(name in vars(module) and vars(module)[name] is value
                    for name, value in old.items()) for module, old in self.namespaces)
            and all(cls.__mro__ == mro and self._class_members(cls) == members
                for cls, mro, members in self.classes)
            and all(function.__code__ is code and function.__defaults__ is defaults
                and function.__kwdefaults__ == keywords
                for function, code, defaults, keywords in self.functions))


def _routed_public_contract(value):
    """Only canonical recursively immutable specs receive value keys."""
    from types import MappingProxyType
    from src.kernel.launch_config import LaunchConfig, TilingSpec
    from src.kernel.kernel_spec import L2ExecutionContext
    kind = type(value)
    if kind in (str, int, bool, bytes, type(None)):
        return kind, value
    if kind is float and math.isfinite(value):
        return kind, value.hex()
    if kind is KernelKind:
        return kind, value.value
    if kind is tuple:
        return kind, tuple(_routed_public_contract(item) for item in value)
    if kind is MappingProxyType:
        return kind, tuple((_routed_public_contract(key), _routed_public_contract(item))
            for key, item in value.items())
    if kind in (KernelSpec, LaunchConfig, TilingSpec, L2ExecutionContext):
        if not kind.__dataclass_params__.frozen:
            raise TypeError("mutable routed resource contract")
        return kind, tuple((field.name, _routed_public_contract(getattr(value, field.name)))
            for field in dataclasses.fields(value))
    raise TypeError("noncanonical routed resource contract")


def _routed_public_resource_input(predictor, kernel, hardware):
    from contextvars import copy_context
    from types import MappingProxyType
    from src.kernel.prediction_cache import _RUNTIME_CACHES
    from src.kernel.memory_access_builder import _PREDICTION_MEMORY_BINDINGS
    context = copy_context()
    eligible = (_RUNTIME_CACHES in context and context[_RUNTIME_CACHES] is not None
        and _PREDICTION_MEMORY_BINDINGS not in context
        and type(predictor) is FusedMoEPredictor and type(kernel) is KernelSpec
        and set(vars(predictor)) == {"_cached_resource_kernel", "_cached_resource_hardware",
            "_cached_resource_value"}
        and type(hardware) is HWSpec and kernel.hardware_id == hardware.hardware_id
        and hardware.supports_source_contract("vllm_0_30_marlin_w4a16_ep1_public_resource_reuse")
        and hardware.kernel_memory_model == "gpu_mshr"
        and kernel.kind is KernelKind.FUSED_MOE and kernel.variant == "routed_experts"
        and kernel.execution_support == "supported" and kernel.dtype == "bf16"
        and kernel.launch is not None and kernel.launch.algorithm == "vllm_marlin_moe_w4a16_aggregate"
        and kernel.attributes["expert_parallel_size"] == 1
        and kernel.attributes["routed_weight_dtype"] == "int4"
        and kernel.attributes["weight_group_size"] == 32
        and kernel.attributes["swiglu_limit"] is None
        and kernel.attributes["expert_weight_tile_rows"] is None
        and "mathematical_program" not in kernel.attributes)
    if not eligible:
        # In particular a mutated HW object can leave the admitted memory
        # model/identity domain. Never let the older one-object cache mask
        # the current constructor's rejection or newly supported behavior.
        if (predictor._cached_resource_kernel is kernel
                and predictor._cached_resource_hardware is hardware
                and type(kernel) is KernelSpec and kernel.variant == "routed_experts"
                and _RUNTIME_CACHES in context and context[_RUNTIME_CACHES] is not None
                and _PREDICTION_MEMORY_BINDINGS not in context):
            predictor._cached_resource_kernel = None
            predictor._cached_resource_hardware = None
            predictor._cached_resource_value = None
        return predictor._source_resource_input(kernel, hardware)
    if not _ROUTED_PUBLIC_SOURCE.unchanged():
        _ROUTED_PUBLIC_REUSE.clear()
        predictor._cached_resource_kernel = None
        predictor._cached_resource_hardware = None
        predictor._cached_resource_value = None
        return predictor._source_resource_input(kernel, hardware)
    try:
        # Prove the original as well as the substituted attributes: a mapping
        # proxy alone does not freeze a nested custom or mutable payload.
        _routed_public_contract(kernel)
        counts, local_counts = _routing(kernel)
        tile_m, _, _ = _tile(kernel)
        routed_rows = sum(local_counts)
        active_experts = sum(count > 0 for count in local_counts)
        padded_rows = sum(math.ceil(count / tile_m) * tile_m for count in local_counts if count > 0)
        if (counts != kernel.attributes["tokens_per_expert"]
                or local_counts != kernel.attributes["local_tokens_per_expert"] or counts != local_counts
                or any(type(count) is not int or count < 0 for count in counts)
                or type(kernel.attributes["expert_parallel_size"]) is not int
                or type(kernel.attributes["expert_rank"]) is not int or kernel.attributes["expert_rank"] != 0
                or kernel.attributes["owned_experts"] != tuple(range(kernel.shape["experts"]))
                or routed_rows != kernel.shape["tokens"] * kernel.shape["topk"] or active_experts == 0
                or tile_m not in (8, 16, 32, 48, 64) or kernel.launch.parameters["padded_rows"] != padded_rows):
            raise TypeError("unproven EP1 routed resource class")
        # This public-input class is deliberately separate from prediction
        # equivalence: retained traffic consumes rows, active weight operands
        # and padded M rows only. Source metadata/scratch and expert order do
        # not enter the returned tuple; all other spec/launch fields stay exact.
        attributes = {name: value for name, value in kernel.attributes.items()
            if name not in ("tokens_per_expert", "local_tokens_per_expert", "owned_experts")}
        fields = tuple((field.name, _routed_public_contract(
            (MappingProxyType(attributes), (routed_rows, active_experts, padded_rows)) if field.name == "attributes"
            else getattr(kernel, field.name))) for field in dataclasses.fields(kernel))
    except TypeError:
        predictor._cached_resource_kernel = None
        predictor._cached_resource_hardware = None
        predictor._cached_resource_value = None
        return predictor._source_resource_input(kernel, hardware)
    identity = id(hardware)
    if identity in _ROUTED_PUBLIC_REUSE:
        retained, guard, entries = _ROUTED_PUBLIC_REUSE[identity]
        if retained is not hardware:
            raise RuntimeError("routed public hardware identity collision")
        if not guard.unchanged():
            del _ROUTED_PUBLIC_REUSE[identity]
            predictor._cached_resource_kernel = None
            predictor._cached_resource_hardware = None
            predictor._cached_resource_value = None
        elif fields in entries:
            inp, compulsory = entries[fields]
            return dataclasses.replace(inp, demand_per_resource=dict(inp.demand_per_resource),
                resource_time_overrides_ns=dict(inp.resource_time_overrides_ns)), compulsory
    # The older one-object evaluator cache is not scoped to this experiment
    # and has no live HW proof. Every new run/class must construct its own
    # current input, even when the caller retained the same predictor object.
    predictor._cached_resource_kernel = None
    predictor._cached_resource_hardware = None
    predictor._cached_resource_value = None
    value = predictor._source_resource_input(kernel, hardware)
    if identity not in _ROUTED_PUBLIC_REUSE:
        guard = _RoutedHardwareData(hardware)
        if not guard.supported:
            return value
        if len(_ROUTED_PUBLIC_REUSE) >= 4:
            _ROUTED_PUBLIC_REUSE.popitem(last=False)
        entries = {}
        _ROUTED_PUBLIC_REUSE[identity] = hardware, guard, entries
    inp, compulsory = value
    if len(entries) >= 128:
        entries.clear()
    entries[fields] = dataclasses.replace(inp,
        demand_per_resource=dict(inp.demand_per_resource),
        resource_time_overrides_ns=dict(inp.resource_time_overrides_ns)), compulsory
    return value


_ROUTED_PUBLIC_REUSE = RuntimeCache()
_ROUTED_PUBLIC_SOURCE = _RoutedPublicSource()
_ROUTED_PUBLIC_SOURCE.seal()
