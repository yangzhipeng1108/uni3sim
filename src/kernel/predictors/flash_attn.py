"""Complete-spec fused-attention physical prediction."""

from src.kernel.prediction_cache import RuntimeCache

from collections import OrderedDict
from functools import wraps
import dataclasses
from bisect import bisect_right
import heapq
import itertools
import math
import statistics
import struct
import threading
from types import MappingProxyType
from typing import NamedTuple, Optional, Tuple

from src.kernel.formula import PerformanceFormula, small_kernel_span_floor_s
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelPrediction, KernelSpec
from src.kernel.hardware_spec import HWSpec
from src.kernel.launch_config import FiaCrossDomainLaunchProgram, LaunchConfig, TilingSpec
from src.kernel.launch_config.tiling_spec import (
    METAX_MLA_KV_BLOCK,
    METAX_MLA_SCHEDULER_PARTITIONS,
    MetaxMlaSchedulerWork,
    metax_mla_combine_block_m,
    flash_attention_packed_varlen_scheduled_workgroups,
)
from src.kernel.predictors.common import (
    _with_graph_regime,
    PredictionInput,
    append_npu_write_path,
    attach_attention_pipeline_envelope,
    attach_pipeline_work_plan,
    attach_structured_contract,
    bind_pipeline_stage_template,
    build_forward_demand,
    causal_attention_pairs,
    gpu_memory_source,
    gpu_partial_limitations,
    graph_replayed_launch,
    launch_wave_geometry,
    npu_launch_floor_contract,
    pipeline_compute_service,
    prediction_from_ns,
    roofline_ns,
    split_global_memory_paths,
    validated_l2_miss_rate,
)
from src.kernel.memory_flash_attention import (
    has_fa3_prefill_source,fa3_prefill_source_work,current_fa3_prefill_source_work,
    has_fa3_combine_source,fa3_combine_source_work,current_fa3_combine_source_work,
    cann_split_fuse_stack_geometry,has_cann_tnd_source,cann_tnd_source_demands,cann_tnd_source_work,current_cann_tnd_source_work,
    has_metax_varlen_source,metax_varlen_source_work,current_metax_varlen_source_work,
    has_dense_flash_attention_source,has_metax_mla_source,has_fa2_split_source,fa2_split_source_work,current_fa2_split_source_work,
    has_cann_attention_source,cann_attention_source_demands,cann_attention_source_work,current_cann_attention_source_work,
    bound_cann_bsh_workspace_work,
    metax_mla_source_work,current_metax_mla_source_work,metax_mla_scheduler_work,
    has_grouped_mla_source,grouped_mla_source_work,current_grouped_mla_source_work,
)
from src.kernel.memory_access_builder import fa3_combine_prediction_work,fa3_prefill_prediction_work,cann_tnd_memory_access_program,dense_attention_memory_access_program,dense_attention_prediction_work,metax_mla_prediction_work,grouped_mla_prediction_work,cann_attention_memory_access_program
from src.kernel.cache_traffic_sim import simulate_attention_prefill_residency
from src.kernel.model_contract import (
    PredictionDiagnostics,
    AnalyticalPhase,
    AnalyticalPhaseLedger,
    AttentionPipelineClassDemand,
    AttentionPipelineEnvelopeDemand,
    AttentionPipelinePhaseDemand,
    ComputeServiceDemand,
    FiaAivMemoryStageDemand,
    FiaBlockReduceSumDescriptorDemand,
    FiaComputeStageDemand,
    FiaCrossDomainAicGroupDemand,
    FiaCrossDomainTaskDemand,
    FiaCrossDomainWorkPlan,
    FiaFixpipeDescriptorDemand,
    FiaFixpipeStageDemand,
    FiaGqaSourcePanelDemand,
    FiaGqaSourceOrderDemand,
    FiaGqaSourceRoundDemand,
    FiaMlaSourceOrderDemand,
    FiaMte1LoadPairDemand,
    FiaMte1StageDemand,
    FiaMte2Nd2NzDescriptorDemand,
    FiaMte2StageDemand,
    FiaSplitFuseRequestRunDemand,
    FiaSplitFuseWorkPlan,
    FiaTilingDataPreludeDemand,
    GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
    MemoryAccessClassDemand,
    MemoryAccessClassSignature,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemorySourceDemand,
    PhaseComputeDemand,
    PhaseMemoryDemand,
    PipelineBuffer,
    PipelineStageDemand,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
    PipelineTaskSequenceDemand,
    PipelineWorkPlan,
)


_CANN_SPLIT_FUSE_ALGORITHMS = {
    "cann_flash_attention_score:5000000000000200203",
    "cann_flash_attention_score:5000000000010200203",
}

# CANN 8.5.1 dispatches these standard GQA decode variants through the same
# FiaKernelNonQuant/FiaBlockCubeNonQuantGqa implementation.  That source closes
# the Cube M-rounding lower bound below, but not the dynamic AIC/AIV partition
# or the output-init/final-stage worker contract.
_CANN_STANDARD_GQA_DECODE_ALGORITHMS = {
    "cann_flash_attention_score:103000000010022221",
    "cann_flash_attention_score:103000000010222221",
    "cann_flash_attention_score:103000000010422221",
    "cann_flash_attention_score:103000000010522221",
    "cann_flash_attention_score:103000000010622221",
}

# CANN 8.5.1 dispatches these two MLA decode variants through
# FiaKernelNonQuantMla.  The exact device source fixes the QK operand width at
# 512 no-PE channels plus 64 RoPE channels, while V and the final output remain
# 512 channels.  Dynamic worker partition and workspace replay are deliberately
# not inferred from the variant identity.
_CANN_MLA_DECODE_ALGORITHMS = {
    "cann_flash_attention_score:105000000000222220",
    "cann_flash_attention_score:105000000000322220",
}

_CANN_FIA_SOFTMAX_WITH_BRC = {
    "103000000010022221": False,
    "103000000010222221": False,
    "103000000010422221": True,
    "103000000010522221": True,
    "103000000010622221": True,
    "105000000000222220": False,
    "105000000000322220": False,
}


def _strided_active_aic_group_is_measured(service, grid_size):
    """Use the normalized, discrete strided-service active-group domain."""

    domain = service["strided_active_aic_group_set"]
    if not isinstance(domain, frozenset):
        raise TypeError(
            "normalized strided active-AIC domain must be a frozenset")
    return grid_size in domain


_CANN_FIA_FP32_VECTOR_BUFFER_ELEMENTS = 8192
_CANN_FIA_FP32_BLOCK_ELEMENTS = 8
_CANN_FIA_FP32_REPEAT_ELEMENTS = 64
_CANN_FIA_AIV_INITIALIZATION_GENERIC_OPERATIONS = 2 * 512

# The pinned MetaX FlashMLA host ABI constructs Q/K with 512 content channels
# plus 64 RoPE channels and requests a 512-channel V/output.  The retained
# MACA 3.2.1 observation does not include matching device source, so this
# identity closes only aggregate semantic work and compulsory Q/K/V traffic;
# it does not publish a tile, replay schedule, or partial-scratch lifecycle.
_METAX_MLA_DECODE_ALGORITHMS = {
    "metax_flash_attention_persistent_queue",
}
_UNBOUND_MLA_SCRATCH_ALGORITHMS = (
    _METAX_MLA_DECODE_ALGORITHMS | {"metax_flash_mla_combine"})

_METAX_STATIC_TRAIT_DIRECT_PREFIX = (
    "metax_flash_attention_trait_direct:")


_METAX_372_Q64_PREFILL_ALGORITHM = (
    "metax_flash_attention_trait_direct:prefill_q64_full_tile")
_METAX_372_Q64_PACKED_PREFILL_ALGORITHM = (
    "metax_flash_attention_trait_packed:prefill_q64_full_tile")
_METAX_372_Q64_DECODE_ALGORITHM = (
    "metax_flash_attention_trait_direct:decode_q64_full_tile")
_METAX_372_Q128_DECODE_ALGORITHM = (
    "metax_flash_attention_trait_direct:decode")
_METAX_372_Q128_PREFILL_ALGORITHM = (
    "metax_flash_attention_trait_direct:prefill_q128_full_tile")
_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM = (
    "metax_flash_attention_trait_direct:"
    "prefill_q128_split_full_tile")

# The fixed MACA 3.7.2 D128/BlockM128/BlockN64/four-wave decode binary has
# one source-level KV-tile loop.  Two exact-symbol, full-tile layer-1
# observations at eight and sixteen KV tiles report 11,713 and 20,897 MTE
# instructions per CTA.  The loop therefore contributes 1,148 instructions
# per additional full tile, with 2,529 launch-local instructions outside the
# steady iterations.  These are instruction counts, not duration-derived
# coefficients.  Other instruction classes fail the nonnegative-integer
# affine gate and are deliberately not published.
_METAX_372_FA_MTE_FIXED_PER_CTA = 2_529
_METAX_372_FA_MTE_PER_FULL_KV_TILE = 1_148
_METAX_372_FA_MMA_PER_FULL_KV_TILE = 512
_METAX_372_FA_TENSOR_OPS_PER_MMA = 8_192
_METAX_372_FA_TENSOR_OPS_PER_AP_CYCLE = 2_048
_METAX_372_FA_WARP_ISSUE_PER_AP_CYCLE = 5
# Independent register-only BF16 MMA wave sweep, MACA 3.7.2, 1125 MHz:
# four resident waves/AP sustain 1612.573689 FLOP/AP/cycle.  The raw result is
# the C500 MMA wave-sweep result published by the matching hardware package
# (SHA256 e3bc531d5d7ccf090ad217799a1204be836c230f2ee4436026339d0e3fe9869b).
_METAX_372_BF16_MMA_FLOPS_PER_AP_CYCLE_4_WAVES = 1_612.573689
_METAX_372_BF16_MMA_FLOPS_PER_AP_CYCLE_BY_WAVES = (
    (1.0, 402.707944),
    (2.0, 806.112470),
    (4.0, 1_612.573689),
    (8.0, 1_861.110857),
    (16.0, 1_943.832179),
    (32.0, 1_994.127341),
)
_METAX_372_Q64_PREFILL_MMA_PER_KV_STEP_BY_HEAD_DIM = {
    64: 128,
    128: 256,
}
_METAX_372_Q64_PREFILL_MTE_PER_KV_STEP = 884
_METAX_372_Q64_PREFILL_MTE_PER_ACTIVE_Q_TILE = 1_956
_METAX_372_Q64_PREFILL_MTE_PER_SCHEDULED_CTA = 64
_METAX_372_Q64_PREFILL_MTE_PER_REQUEST_ADJUSTMENT = -220
_METAX_372_Q64_DECODE_MMA_PER_KV_STEP = 256
_METAX_372_TILE16_PAGED_MTE_FIXED = 389
_METAX_372_TILE16_PAGED_MTE_PER_KV_STEP = 155
_METAX_372_TILE16_CONTIGUOUS_MTE_FIXED = 519
_METAX_372_TILE16_CONTIGUOUS_MTE_PER_KV_STEP = 104
_METAX_372_TILE16_MMA_PER_KV_STEP = 16
_METAX_372_TILE16_D64_UNSPLIT_MTE_FIXED = 891
_METAX_372_TILE16_D64_SPLIT_MTE_FIXED = 443
_METAX_372_TILE16_D64_MTE_PER_KV_STEP = 89
_METAX_372_TILE16_D64_MMA_PER_KV_STEP = 8
_METAX_MLA_MMA_PER_KV_BLOCK = 416
_METAX_MLA_MTE_LAUNCH = 2_496
_METAX_MLA_MTE_PER_KV_BLOCK = 1_496
_METAX_MLA_MTE_PER_SEGMENT = 3_212
_METAX_MLA_MTE_PER_SAME_WORKER_TRANSITION = -848
_METAX_MLA_MTE_PER_UNSPLIT_REQUEST = 520
# The pinned MACA 3.7.2 Q64 trait compiles to 218 registers/thread and
# 32 KiB dynamic shared memory.  The compiler resource report for the exact
# 256-thread trait publishes a two-block register limit; no duration or
# achieved-occupancy observation participates in this discrete launch fact.
_METAX_372_Q64_PREFILL_RESIDENT_CTA_SLOTS_PER_AP = 2
_METAX_372_D64_Q64_PREFILL_RESIDENT_CTA_SLOTS_PER_AP = 4
_METAX_372_Q128_PREFILL_RESIDENT_CTA_SLOTS_PER_AP = 2
# The exact D128 BlockM64/BlockN64 decode artifact uses 154 registers/thread,
# 32 KiB dynamic shared memory and a 256-thread block.  Its compiler resource
# report publishes a three-block register limit per AP.
_METAX_372_Q64_DECODE_RESIDENT_CTA_SLOTS_PER_AP = 3
# The exact D128 BlockM128/BlockN64 decode artifact uses 238 registers/thread,
# 32 KiB dynamic shared memory and a 256-thread block.  Its compiler resource
# report therefore permits two resident CTAs per AP, matching the Q128 prefill
# instantiation of the same source trait.
_METAX_372_Q128_DECODE_RESIDENT_CTA_SLOTS_PER_AP = 2
# The pinned MACA 3.7.2 tile16 decode traits use one 64-thread wave.  D128
# uses 8 KiB dynamic shared memory, so its 14-block register limit is capped
# at eight CTAs/AP by C500's 64 KiB shared memory.  D64 uses 4 KiB dynamic
# shared memory, making its 16-CTA shared-memory ceiling tighter than the
# 23-block register limit.  These are static launch-resource bounds, not
# achieved-occupancy durations.
_METAX_372_TILE16_D128_RESIDENT_CTA_SLOTS_PER_AP = 8
_METAX_372_TILE16_D64_RESIDENT_CTA_SLOTS_PER_AP = 16
_SM80_FA2_MAIN_RESIDENT_CTA_SLOTS = 2
_A800_FA2_PAIRED_FIFO_MAX_CTAS = 512
_A800_FA2_PAGED_PERIOD32_LDGSTS_SERVICE = (
    "sm80_fa2_paged_q64_kv128_period32_ldgsts")
_A800_FA2_CHUNKED_Q8192_S4096_LDGSTS_SERVICE = (
    "sm80_fa2_chunked_q8192_s4096_two_resident_ldgsts")
_A800_GROUPED_MLA_STAGE1_ALGORITHM = "triton_grouped_mla_stage1"
_A800_GROUPED_MLA_STAGE2_ALGORITHM = "triton_grouped_mla_stage2"
# The pinned Triton artifact allocates 89,088 B dynamic shared memory per CTA;
# GA100 exposes 167,936 B per SM, so only one complete CTA can reside per SM.
_A800_GROUPED_MLA_STAGE1_RESIDENT_CTA_SLOTS = 1
# The pinned stage-2 Triton source launches 128-thread independent row CTAs.
# Its compiler resource report publishes 32 registers/thread and 1,024 B static
# shared memory.  On GA100 neither resource is tighter than the architectural
# 2,048-thread limit, which admits sixteen such CTAs per SM.  This is a static
# residency fact; no production duration or achieved-occupancy counter enters
# the slot count.
_A800_GROUPED_MLA_STAGE2_RESIDENT_CTA_SLOTS = 16
_CANN_SPLIT_FUSE_WORK_UNSET = object()

@dataclasses.dataclass(frozen=True)
class _AttentionTileWork:
    """One real request/Q-tile's bottom-right attention work."""

    query_rows: int
    attention_elements: int
    attended_kv: int
    kv_tiles: int


@dataclasses.dataclass(frozen=True)
class _AttentionCTAWork:
    """One source CTA's query rows and KV-step count.

    A zero-step split is still a real FA2 CTA.  The pinned source takes its
    empty-range branch and writes zero partial output plus ``-inf`` LSE without
    loading Q/K/V or issuing MMA/softmax work.
    """

    query_rows: int
    step_count: int


@dataclasses.dataclass(frozen=True)
class _GroupedMLASASSStageIssue:
    """Static warp instructions assigned to grouped-MLA source stages."""

    prologue: float
    qk: float
    softmax: float
    pv: float
    next_panel: float
    epilogue: float
    cta_end: float


@dataclasses.dataclass(frozen=True)
class _FA2SplitD128SASSStageIssue:
    """Static warp instructions assigned to split-FA2 source stages."""

    prologue: float
    first_qk_v: float
    first_softmax_pv: float
    later_qk_v: float
    later_softmax_pv: float
    next_panel: float
    epilogue: float
    cta_end: float


@dataclasses.dataclass(frozen=True)
class _AttentionTileAggregate:
    """Closed-form Q-tile totals without materializing logical CTAs."""

    tile_count: int
    query_rows: int
    attended_kv: int
    kv_tiles: int


# CANN 8.5.1 fused_infer_attention_score/kernel_common.hpp: the SplitFuse
# workspace geometry is fixed by source constants, not by tiling.
_CANN_SPLIT_FUSE_PRE_LAUNCH = 2          # PRE_LAUNCH
_CANN_SPLIT_FUSE_Q_TILE_CEIL = 128       # Q_TILE_CEIL
_CANN_SPLIT_FUSE_MAX_KV_STACK_LEN = 512  # MAX_KV_STACK_LEN
_CANN_SPLIT_FUSE_WORKSPACE_BLOCK_DB = (
    _CANN_SPLIT_FUSE_Q_TILE_CEIL * _CANN_SPLIT_FUSE_MAX_KV_STACK_LEN)


# AscendC clears each core's KFC message region at every MIX kernel entry
# (tikcpp/tikcfw/impl/dav_c220/kfc/kfc_comm.h ClearWorkspaceImpl):
#   size       = BIDIRECTION_NUM * MAX_MSG_COUNT * AlignTo32(sizeof(KfcMsg))
#                * MIX_NUM                        = 2 * 64 * 128 * 2 = 32768
#   sizeUbmsg  = MIX_NUM * AlignTo32(sizeof(MsgUBAvalied))  = 2 * 64  =   128
# addressed workspace + size * GetBlockIdxImpl(), so it is per AIC core, and
# written with copy_cbuf_to_gm, an L1->GM copy rather than a Fixpipe store --
# which is why it never appears in L0C_to_GM and why no operator source
# contains a writer for it.  MAX_MSG_COUNT is impl/kernel_macros.h:88.
_CANN_KFC_MAX_MSG_COUNT = 64
_CANN_KFC_BIDIRECTION_NUM = 2
_CANN_KFC_MIX_NUM = 2
_CANN_KFC_MSG_BYTES = 128
_CANN_KFC_UB_MSG_BYTES = 64
_CANN_KFC_CLEAR_BYTES_PER_CORE = (
    _CANN_KFC_BIDIRECTION_NUM * _CANN_KFC_MAX_MSG_COUNT
    * _CANN_KFC_MSG_BYTES * _CANN_KFC_MIX_NUM
    + _CANN_KFC_MIX_NUM * _CANN_KFC_UB_MSG_BYTES)


def _cann_kfc_workspace_clear_bytes(kernel: KernelSpec) -> float:
    """Return the GM bytes the AscendC MIX entry clears for this launch."""

    from src.kernel.memory_cann_mla import has_cann_mla_source,bound_cann_mla_source_work
    source=bound_cann_mla_source_work(kernel) if has_cann_mla_source(kernel) else bound_cann_bsh_workspace_work(kernel)
    if source is not None:return float(source['framework_clear_write_bytes'])
    cores = kernel.launch.grid_size
    if cores <= 0:
        raise ValueError("CANN FIA launch must occupy at least one AIC core")
    return float(cores * _CANN_KFC_CLEAR_BYTES_PER_CORE)


def _attach_cann_kfc_workspace_clear(
        inp: PredictionInput, kernel: KernelSpec) -> PredictionInput:
    """Charge the framework's per-core KFC clear at L2 and GM."""

    if inp.mem_model != "npu_tile":
        raise ValueError("CANN KFC workspace clear requires npu_tile")
    if any(path.name == "kfc_workspace_clear" for path in inp.memory_paths):
        raise ValueError("CANN KFC workspace clear must attach once")
    if inp.launch_task_kind != "mixed":
        # ClearWorkspaceImpl belongs to the MIX kernel entry, which pairs an
        # AIC with its AIV subcores.  An AIV-only launch opens no KFC channel
        # and clears nothing.
        return inp
    if not any(path.name == "fixpipe_l2_write" for path in inp.memory_paths):
        # Charge the framework clear only where the launch is modelled as a
        # CANN FIA MIX kernel end to end.  Without the source workspace there
        # is no Fixpipe path either, and claiming framework behaviour for a
        # launch whose structure is not modelled would be an extrapolation.
        return inp
    clear_bytes = _cann_kfc_workspace_clear_bytes(kernel)
    path = MemoryPathDemand(
        name="kfc_workspace_clear",
        access_kind="write",
        initial_transactions=None,
        nodes=(
            MemoryNodeDemand(
                resource_role="l2_cache",
                transactions=None,
                bytes=clear_bytes,
                transaction_bytes=None,
            ),
            MemoryNodeDemand(
                resource_role="dram",
                transactions=None,
                bytes=clear_bytes,
                transaction_bytes=None,
            ),
        ),
        source=MemorySourceDemand(
            active_units=inp.active_units,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    flat = dict(inp.demand_per_resource)
    flat["l2_cache"] += clear_bytes
    flat["dram"] += clear_bytes
    return dataclasses.replace(
        inp,
        demand_per_resource=flat,
        memory_paths=inp.memory_paths + (path,),
    )


def _cann_split_fuse_slot_write_bytes(work, head_dim: int):
    """Return the distinct GM extent each SplitFuse workspace ring touches.

    ``flash_attention_regular.h`` gives the launch four workspace
    destinations, two per engine domain, and every one of them is stored
    through a ``Layout(rowNum, cols, stride)`` whose distinct extent is
    ``rowNum * stride`` elements:

    ==================== ====== ======== =========================== =======
    store                engine buffer   extent                      element
    ==================== ====== ======== =========================== =======
    blockMmadQK          AIC    gS       rowNum * stackSeqTilePad    FP32
    epilogueOnlineSoftmax AIV   gP       rowNum * stackSeqTilePad    BF16
    blockMmadPV          AIC    gOTmp    rowNum * embedRoundV        FP32
    epilogueRescaleO     AIV    gOUpdate rowNum * embedRound         FP32
    ==================== ====== ======== =========================== =======

    ``stackSeqTilePad = blockStackNum * pagedBlockSize`` with
    ``blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize`` (l.293-295), so the
    stride is the stack ceiling however the pages divide it.  ``embedRound``
    and ``embedRoundV`` are ``RoundUp(embed, BLOCK_SIZE)`` (l.189-190); this
    path is the non-MLA FIA kernel, where the value head width equals the
    query head width, so the two agree.

    gS, gP and gOTmp are addressed
    ``coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1)
     + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB`` (l.318-321, 427-436): a
    ring of ``PRE_LAUNCH + 1`` slots per core, of which a core reaches
    ``min(PRE_LAUNCH + 1, its stack tiles)``.  gOUpdate is addressed
    ``coreIdx * WORKSPACE_BLOCK_SIZE_DB`` (l.478), one slot per core with no
    ring.

    ``WORKSPACE_BLOCK_SIZE_DB`` is the slot *capacity*, the widest tile
    (``Q_TILE_CEIL`` rows by ``MAX_KV_STACK_LEN`` columns) the host reserves
    for.  Charging it for every buffer overstates gOTmp and gOUpdate by the
    ratio of the stack ceiling to the head width -- fourfold at D=128 -- so
    the layout strides above are what the stores actually reach.

    Returns ``(aiv_bytes, aic_bytes)``, the two engine domains separately:
    they leave the core through different ports (MTE3 from UB, Fixpipe from
    L0C) and msprof counts them under separate write hit rates.
    """

    stack_pad = _CANN_SPLIT_FUSE_MAX_KV_STACK_LEN
    embed_round = ((head_dim + 15) // 16) * 16
    aiv_bytes = 0
    aic_bytes = 0
    for worker_slots in work.slot_rows_by_worker:
        if not any(worker_slots):
            continue
        ring_rows = sum(worker_slots)
        aiv_bytes += ring_rows * stack_pad * 2      # gP, BF16
        # gOUpdate has no ring: one slot per core, as wide as the widest
        # task that core ran, which is slot 0's row count.
        aiv_bytes += worker_slots[0] * embed_round * 4
        aic_bytes += ring_rows * stack_pad * 4      # gS, FP32
        aic_bytes += ring_rows * embed_round * 4    # gOTmp, FP32
    return float(aiv_bytes), float(aic_bytes)


def _cann_split_fuse_aiv_unique_write_bytes(
        work, head_dim: int, final_output_bytes: float,
        aiv_workspace_write_bytes: float):
    """Return the distinct GM extent the SplitFuse AIV stores touch.

    The AIV domain owns gP and gOUpdate; see
    :func:`_cann_split_fuse_slot_write_bytes` for the ring geometry.  The
    final output is written once.

    Verified against a live tiling capture: the same constants reproduce
    ``mm1OutSize``, ``smOnlineOutSize``, ``mm2OutSize`` and ``UpdateSize``
    exactly (910B FA prefill tiling probe; capture artifacts since cleaned).
    """

    workspace_bytes, _ = _cann_split_fuse_slot_write_bytes(work, head_dim)
    # A distinct extent can never exceed what the same stores actually moved,
    # which bounds the workspace share without appealing to any measured
    # traffic.
    workspace_unique = min(
        workspace_bytes, float(aiv_workspace_write_bytes))
    return float(workspace_unique + final_output_bytes)


def _cann_split_fuse_aic_unique_write_bytes(
        work, head_dim: int, aic_workspace_write_bytes: float):
    """Return the distinct GM extent the SplitFuse AIC Fixpipe stores touch.

    The AIC domain owns gS and gOTmp on the same ring as gP; see
    :func:`_cann_split_fuse_slot_write_bytes`.  Unlike the decode path --
    whose fixed per-core AIC write has no writer anywhere in
    ``fia_kernel_nonquant.h`` and stays unpriced -- both SplitFuse Fixpipe
    destinations are named, addressed and sized by the source.
    """

    _, workspace_bytes = _cann_split_fuse_slot_write_bytes(work, head_dim)
    return min(workspace_bytes, float(aic_workspace_write_bytes))


@dataclasses.dataclass(frozen=True)
class _CannSplitFuseWork:
    """Source-issued aggregate work for one fixed CANN SplitFuse launch."""

    task_count: int
    empty_task_count: int
    empty_output_elements: int
    empty_dup_operations: int
    attention_elements: int
    final_output_elements: int
    workspace_output_elements: int
    rescale_output_elements: int
    mask_mte2_bytes: int
    mmad_attention_elements: int
    kv_replay_positions: int
    # Widest `rowNum` to reach each ring slot of each AIC, as
    # `PRE_LAUNCH + 1` entries per worker.  The SplitFuse workspace is
    # addressed
    # `coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1)
    #  + curStackTileMod * WORKSPACE_BLOCK_SIZE_DB` with
    # `curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1)`, and
    # `stackSeqCount` is declared inside the per-query-block body
    # (flash_attention_regular.h:297), so every task restarts the ring at
    # slot 0 and a task reaching `n` stack tiles touches slots `0 ..
    # min(PRE_LAUNCH + 1, n) - 1`.  A core that runs twenty short tasks
    # therefore reuses the same one slot twenty times.
    #
    # Each store is laid out `Layout*(rowNum, cols, stride)`
    # (l.281, 323, 357, 432, 476), so a slot's distinct extent is the widest
    # `rowNum` that ever reached it times the layout stride -- resolved per
    # slot rather than as a worker-wide row maximum, which would pair a wide
    # shallow task's rows with a narrow deep task's slots.
    slot_rows_by_worker: tuple
    aic_mmad_attention_elements: tuple
    aiv_generic_operations: tuple
    aiv_exp_operations: tuple
    aiv_exp_invocations: tuple
    aiv_div_operations: tuple
    aiv_div_invocations: tuple


@dataclasses.dataclass(frozen=True)
class _CannWorkspaceLowerBound:
    """Unpadded mandatory GM traffic between CANN AIC/AIV phases."""

    score_elements: float
    output_elements: float
    read_bytes: float
    aic_write_bytes: float
    aiv_write_bytes: float


@dataclasses.dataclass(frozen=True)
class _CannStandardDecodeVectorLowerBound:
    """Unpadded mandatory AIV work in the standard CANN GQA pipeline."""

    vec1_operations: float
    vec2_operations: float


def _shape(kernel: KernelSpec):
    return tuple(int(kernel.shape[name]) for name in (
        "batch", "q_len", "kv_len", "head_dim",
        "num_query_heads", "num_kv_heads"))


def _request_lengths(kernel: KernelSpec):
    if ("source_program" in kernel.attributes
            and kernel.attributes["source_program"] == "ascend_0_18_full_fia_split_fuse"):
        return (tuple(kernel.attributes["model_input_q_lens"]),
            tuple(kernel.attributes["model_input_kv_lens"]))
    q_lens = tuple(int(value) for value in kernel.attributes["q_lens"])
    kv_lens = tuple(int(value) for value in kernel.attributes["kv_lens"])
    if len(q_lens) != len(kv_lens):
        raise ValueError("FLASH_ATTN q_lens and kv_lens must have equal length")
    return q_lens, kv_lens


def _has_graph_padding(kernel: KernelSpec) -> bool:
    """Whether graph padding lacks a complete physical source program.

    The model-input request lengths establish the captured launch ABI, while
    ``q_lens``/``kv_lens`` remain the only valid Q/KV work.  Pinned source
    worker schedules describe one real request per launched slot, so they
    cannot be reused for masked graph slots without turning padding into
    semantic attention work.  Keep their aggregate launch floor, but reject
    the per-CTA source schedule in that case.
    """

    # SplitFuse represents the physical tail through its distinct Q-loading
    # and local output epilogue. Its explicit source plan owns the padding.
    if ("source_program" in kernel.attributes
            and kernel.attributes["source_program"] == "ascend_0_18_full_fia_split_fuse"):
        return False
    names = (
        "model_input_q_lens", "model_input_kv_lens", "graph_runtime_mode",
    )
    present = tuple(name for name in names if name in kernel.attributes)
    if not present:
        return False
    if present != names:
        raise ValueError("FLASH_ATTN graph-domain fields must be complete")
    q_lens, kv_lens = _request_lengths(kernel)
    model_q_lens = tuple(
        int(value) for value in kernel.attributes["model_input_q_lens"])
    model_kv_lens = tuple(
        int(value) for value in kernel.attributes["model_input_kv_lens"])
    if len(model_q_lens) != int(kernel.shape["batch"]):
        raise ValueError(
            "FLASH_ATTN model-input Q lengths disagree with captured batch")
    if len(model_kv_lens) != int(kernel.shape["batch"]):
        raise ValueError(
            "FLASH_ATTN model-input KV lengths disagree with captured batch")
    return (q_lens, kv_lens) != (model_q_lens, model_kv_lens)


def _metax_full_graph_execution_kernel(kernel: KernelSpec) -> KernelSpec:
    """Use C500's captured slot ABI for a source-backed Graph program.

    The logical request lengths remain the Serving-workload contract. MACA's
    captured tile16 program, however, dispatches its source CTAs over every
    resolved Graph slot. For the narrow C500 Full-Graph families below, build
    the physical execution model from the already-recorded model-input slot
    arrays. This is a source/launch fact, not a duration-derived correction.
    """

    if not _has_graph_padding(kernel):
        return kernel
    if kernel.attributes["graph_runtime_mode"] != "FULL":
        return kernel
    if kernel.dtype.lower() != "bf16":
        return kernel
    _, q_len, _, head_dim, _, _ = _shape(kernel)
    if q_len != 1 or head_dim not in (64, 128):
        return kernel
    if kernel.attributes["kv_layout"] not in ("paged", "paged_varlen"):
        return kernel
    variant = kernel.variant.lower()
    algorithm = kernel.launch.algorithm
    if variant == "decode":
        if not algorithm.startswith(_METAX_STATIC_TRAIT_DIRECT_PREFIX):
            return kernel
    elif variant in ("combine", "split_combine"):
        if algorithm != "metax_flash_attention_generic":
            return kernel
    else:
        return kernel
    attributes = dict(kernel.attributes)
    attributes["q_lens"] = attributes["model_input_q_lens"]
    attributes["kv_lens"] = attributes["model_input_kv_lens"]
    return dataclasses.replace(kernel, attributes=attributes)


def _fa2_full_graph_combine_execution_kernel(kernel: KernelSpec) -> KernelSpec:
    """Use FA2's captured slot ABI for the pinned split-KV reducer.

    The reducer's output-row program consumes the captured scratch/output
    layout for every graph slot. The launch has already been selected from
    that layout in ``lower_workload``. Replacing the logical lengths here
    makes the source pipeline use the same physical ABI without a
    duration-derived correction.

    FA2's split main program instead branches on the per-request sequence
    state before its body. Its masked-slot path is not source-closed, so it
    retains the aggregate logical-work model. Limit this override to the
    pinned BF16 D128 paged reducer family.
    """

    if not _has_graph_padding(kernel):
        return kernel
    if kernel.attributes["graph_runtime_mode"] != "FULL":
        return kernel
    if kernel.dtype.lower() != "bf16":
        return kernel
    _, q_len, _, head_dim, _, _ = _shape(kernel)
    if q_len != 1 or head_dim != 128:
        return kernel
    if kernel.attributes["kv_layout"] not in ("paged", "paged_varlen"):
        return kernel
    variant = kernel.variant.lower()
    algorithm = kernel.launch.algorithm
    if variant not in ("combine", "split_combine"):
        return kernel
    if algorithm != "flash_attn_splitkv_combine":
        return kernel
    attributes = dict(kernel.attributes)
    attributes["q_lens"] = attributes["model_input_q_lens"]
    attributes["kv_lens"] = attributes["model_input_kv_lens"]
    return dataclasses.replace(kernel, attributes=attributes)


def _cann_fia_appended_slot_execution_kernel(kernel: KernelSpec) -> KernelSpec:
    """Use CANN's captured slot ABI when graph padding only appends slots.

    ``GraphAttentionShape.from_token_shape`` has exactly two padding shapes.
    A uniform decode is padded by *appending* dummy slots whose Q and KV
    lengths are both the uniform decode query length; every real request
    keeps its own lengths, so the model-input arrays are a prefix superset of
    the logical ones.  Non-uniform work is padded instead by *extending* the
    largest logical segment, which inflates a real request's KV length.

    ``_has_graph_padding`` rejects the per-CTA source schedule for both,
    because reusing a pinned one-request-per-slot schedule across an inflated
    segment would charge masked rows as semantic attention work.  That reason
    holds only for the extending shape.  For the appending shape the captured
    launch really does dispatch one slot per recorded request -- the real ones
    unchanged, plus dummy slots carrying a single KV row each -- so binding
    the source program to the model-input arrays prices the captured work
    exactly, and strictly above the unpadded price it supersets.  This is a
    source/launch fact taken from the recorded slot arrays, not a
    duration-derived correction.
    """

    if not _has_graph_padding(kernel):
        return kernel
    if kernel.attributes["graph_runtime_mode"] != "FULL":
        return kernel
    if kernel.launch.algorithm not in (
            _CANN_STANDARD_GQA_DECODE_ALGORITHMS
            | _CANN_MLA_DECODE_ALGORITHMS):
        return kernel
    if "cann_source_task_count" not in kernel.launch.parameters:
        return kernel
    q_lens, kv_lens = _request_lengths(kernel)
    model_q_lens = tuple(
        int(value) for value in kernel.attributes["model_input_q_lens"])
    model_kv_lens = tuple(
        int(value) for value in kernel.attributes["model_input_kv_lens"])
    logical_count = len(q_lens)
    if len(model_q_lens) <= logical_count:
        raise ValueError(
            "FLASH_ATTN graph padding cannot shrink the request slot count")
    # Appending shape: every logical request survives untouched in the same
    # slot order, and the padded tail is dummy slots with one KV row per
    # query row.  Anything else is the extending shape, which keeps the
    # existing fail-safe.
    if (model_q_lens[:logical_count] != q_lens
            or model_kv_lens[:logical_count] != kv_lens):
        return kernel
    if any(q_len != kv_len for q_len, kv_len
           in zip(model_q_lens[logical_count:],
                  model_kv_lens[logical_count:])):
        return kernel
    attributes = dict(kernel.attributes)
    attributes["q_lens"] = attributes["model_input_q_lens"]
    attributes["kv_lens"] = attributes["model_input_kv_lens"]
    return dataclasses.replace(kernel, attributes=attributes)


def _splits(kernel: KernelSpec) -> int:
    tiling = kernel.launch.tiling
    if tiling is None or tiling.num_kv_splits is None:
        raise ValueError("FLASH_ATTN requires num_kv_splits in LaunchConfig")
    return int(tiling.num_kv_splits)


def _split_scratch_dram_width(kernel: KernelSpec) -> int:
    return int(kernel.launch.parameters[
        "split_scratch_dram_bytes_per_element"])


def _metax_mla_scheduler_work(kernel: KernelSpec) -> MetaxMlaSchedulerWork:
    return metax_mla_scheduler_work(kernel)


def _is_source_bound_metax_mla_main(kernel: KernelSpec) -> bool:
    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm
            != "metax_flash_attention_persistent_queue"):
        return False
    work = _metax_mla_scheduler_work(kernel)
    del work
    if kernel.launch.grid != (1, 1, METAX_MLA_SCHEDULER_PARTITIONS):
        raise ValueError("MetaX FlashMLA main grid disagrees with source")
    if kernel.launch.block != (512, 1, 1):
        raise ValueError("MetaX FlashMLA main block disagrees with source")
    return True


def _metax_mla_combine_block_m(kernel: KernelSpec) -> int:
    batch, q_len, _, _, query_heads, _ = _shape(kernel)
    return metax_mla_combine_block_m(batch, q_len, query_heads)


def _is_source_bound_metax_mla_combine(kernel: KernelSpec) -> bool:
    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm != "metax_flash_mla_combine"):
        return False
    work = _metax_mla_scheduler_work(kernel)
    del work
    batch, q_len, _, _, query_heads, _ = _shape(kernel)
    block_m = _metax_mla_combine_block_m(kernel)
    expected_grid = (math.ceil(batch * query_heads * q_len / block_m), 1, 1)
    if kernel.launch.grid != expected_grid:
        raise ValueError(
            "MetaX FlashMLA combine grid disagrees with source: "
            f"launch={kernel.launch.grid}, source={expected_grid}")
    if kernel.launch.block != (256, 1, 1):
        raise ValueError("MetaX FlashMLA combine block disagrees with source")
    return True


def _metax_mla_main_output_bytes(kernel: KernelSpec) -> float:
    return float(current_metax_mla_source_work(kernel)['write_bytes'])


def _metax_mla_main_smem_bytes(kernel: KernelSpec) -> float:
    return float(current_metax_mla_source_work(kernel)['smem_projection_bytes'])


def _metax_mla_combine_traffic_bytes(kernel: KernelSpec):
    source = current_metax_mla_source_work(kernel)
    return float(source['read_bytes']),float(source['write_bytes'])


def _cann_standard_decode_grouped_rows(kernel: KernelSpec):
    """Return source-defined rows per KV-head task, if applicable."""

    if kernel.launch.algorithm not in _CANN_STANDARD_GQA_DECODE_ALGORITHMS:
        return None
    if kernel.variant.lower() != "decode":
        raise ValueError("CANN standard GQA variants require decode execution")
    if kernel.dtype.lower() != "bf16":
        raise ValueError("CANN standard GQA variants require BF16")
    _, q_len, _, _, query_heads, kv_heads = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    if q_len != 1 or any(length != 1 for length in q_lens):
        raise ValueError(
            "CANN standard GQA decode variants require one query row per "
            "request")
    if query_heads % kv_heads:
        raise ValueError(
            "CANN standard GQA variants require integral GQA groups")
    grouped_rows = query_heads // kv_heads
    if grouped_rows > 16:
        raise ValueError(
            "CANN standard GQA source contract covers at most 16 grouped rows")
    return grouped_rows


def _cann_source_schedule_parameters(kernel: KernelSpec):
    """Return an exact retained CANN FIA source schedule."""

    if _has_graph_padding(kernel):
        return None
    parameters = kernel.launch.parameters
    has_source_schedule = "cann_source_task_count" in parameters
    if not has_source_schedule:
        return None
    if kernel.launch.algorithm not in (
            _CANN_STANDARD_GQA_DECODE_ALGORITHMS
            | _CANN_MLA_DECODE_ALGORITHMS):
        raise ValueError(
            "CANN source-schedule parameters require a source-bound FIA "
            "variant")
    if kernel.launch.algorithm in _CANN_STANDARD_GQA_DECODE_ALGORITHMS:
        if _cann_standard_decode_grouped_rows(kernel) is None:
            raise AssertionError(
                "CANN source-schedule parameters lost their GQA contract")
    elif _cann_mla_dimensions(kernel) is None:
        raise AssertionError(
            "CANN source-schedule parameters lost their MLA contract")
    return parameters


def _cann_source_task_input_bytes(kernel: KernelSpec, source_schedule):
    """Return exact source-issued Q/K/V L2 reads for CANN FIA.

    The retained CANN payload is accepted by the launch selector only when a
    complete GQA query group fits in one S1 task.  Consequently each S2 task
    rereads that task's Q group, while the disjoint S2 task sequence traverses
    K and V exactly once.  DRAM traffic remains the compulsory unique Q/K/V
    footprint because the source schedule does not prove that replay misses
    L2.
    """

    from src.kernel.memory_cann_mla import has_cann_mla_source,bound_cann_mla_source_work
    source=bound_cann_mla_source_work(kernel) if has_cann_mla_source(kernel) else bound_cann_bsh_workspace_work(kernel)
    if source is not None:
        return tuple(float(source[name]) for name in ('priced_task_query_l2_bytes','key_requested_bytes','value_requested_bytes'))
    grouped_rows = _cann_standard_decode_grouped_rows(kernel)
    mla_dimensions = _cann_mla_dimensions(kernel)
    if grouped_rows is None and mla_dimensions is None:
        raise ValueError(
            "CANN source task input traffic requires source-bound FIA decode")
    batch, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    if any(length != 1 for length in q_lens):
        raise ValueError(
            "CANN source task input traffic requires one query row per "
            "request")
    task_count_value = source_schedule["cann_source_task_count"]
    if (not float(task_count_value).is_integer()
            or task_count_value <= 0):
        raise ValueError(
            "CANN source task input traffic requires an integral task count")
    task_count = int(task_count_value)
    # The logical positive-KV prefix needs score/value tasks. A validated
    # zero-KV physical tail is written only by output initialization.
    head_task_count = len(q_lens) * kv_heads
    if task_count < head_task_count:
        raise ValueError(
            "CANN source task count cannot cover every batch/KV-head pair")
    if grouped_rows is not None and query_heads != grouped_rows * kv_heads:
        raise ValueError(
            "CANN source grouped rows disagree with the query/KV head ratio")

    element_bytes = int(kernel.shape["dtype_bytes"])
    if mla_dimensions is None:
        qk_head_dim = head_dim
        value_head_dim = head_dim
    else:
        qk_head_dim, value_head_dim = mla_dimensions
    query_l2_bytes = float(
        task_count * (query_heads // kv_heads) * qk_head_dim * element_bytes)
    key_l2_bytes = float(
        kv_heads * sum(kv_lens) * qk_head_dim * element_bytes)
    value_l2_bytes = float(
        kv_heads * sum(kv_lens) * value_head_dim * element_bytes)
    unique_query_bytes = float(
        query_heads * sum(q_lens) * qk_head_dim * element_bytes)
    if query_l2_bytes < unique_query_bytes:
        raise ValueError(
            "CANN source task Q reads are below the unique Q footprint")
    return query_l2_bytes, key_l2_bytes, value_l2_bytes


def _cann_source_vector_service_demands(
        kernel: KernelSpec, source_schedule):
    """Decode an exact physical-AIV typed schedule from launch facts."""

    worker_count_value = source_schedule[
        "cann_source_vector_worker_count"]
    if (not float(worker_count_value).is_integer()
            or worker_count_value <= 0):
        raise ValueError(
            "CANN source Vector worker count must be a positive integer")
    worker_count = int(worker_count_value)
    if worker_count != 2 * kernel.launch.grid_size:
        raise ValueError(
            "CANN source Vector worker count disagrees with the 1:2 mixed "
            "launch")

    def operations(field):
        values = tuple(
            source_schedule[
                f"cann_source_vector_worker_{worker}_{field}_operations"]
            for worker in range(worker_count))
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError(
                f"CANN source Vector {field} worker operations are invalid")
        return values

    def invocations(field):
        values = tuple(
            source_schedule[
                f"cann_source_vector_worker_{worker}_{field}_invocations"]
            for worker in range(worker_count))
        if any(
                value < 0 or not math.isfinite(value)
                or not float(value).is_integer()
                for value in values):
            raise ValueError(
                f"CANN source Vector {field} worker invocations are invalid")
        return tuple(int(value) for value in values)

    demands = []
    for service in ("generic", "exp", "div"):
        worker_operations = operations(service)
        if service == "generic":
            worker_invocations = (0,) * worker_count
        elif service in ("exp", "div"):
            worker_invocations = invocations(service)
        else:
            raise AssertionError(
                f"unexpected CANN source Vector service: {service!r}")
        if service != "generic" and any(
                (worker_operation > 0) != (worker_invocation > 0)
                for worker_operation, worker_invocation in zip(
                    worker_operations, worker_invocations)):
            raise ValueError(
                f"CANN source Vector {service} worker operations and "
                "invocations must be positive together")
        aggregate_operations = math.fsum(worker_operations)
        if aggregate_operations == 0.0:
            if any(worker_invocations):
                raise ValueError(
                    f"CANN source Vector {service} has invocations without "
                    "operations")
            continue
        demands.append(ComputeServiceDemand(
            service=service,
            operations=aggregate_operations,
            serial_invocations=max(worker_invocations),
            operations_by_worker=worker_operations,
            serial_invocations_by_worker=worker_invocations,
        ))
    typed_operations = math.fsum(
        demand.operations for demand in demands)
    aggregate_operations = source_schedule[
        "cann_source_vector_operations"]
    if not math.isclose(
            typed_operations, aggregate_operations,
            rel_tol=0.0, abs_tol=max(
                math.ulp(typed_operations), math.ulp(aggregate_operations))):
        raise ValueError(
            "CANN source typed Vector worker schedule does not conserve the "
            "aggregate demand")
    return tuple(demands)


def _cann_fia_align(value, alignment):
    if value <= 0 or alignment <= 0:
        raise ValueError("CANN FIA alignment inputs must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _cann_fia_vector_row_chunks(rows, columns, *, softmax_with_brc):
    rows_per_chunk = _CANN_FIA_FP32_VECTOR_BUFFER_ELEMENTS // columns
    if not softmax_with_brc:
        rows_per_chunk = (
            rows_per_chunk // _CANN_FIA_FP32_BLOCK_ELEMENTS
            * _CANN_FIA_FP32_BLOCK_ELEMENTS)
    if rows_per_chunk <= 0:
        raise ValueError("CANN FIA Vector row chunk has no UB capacity")
    rows_per_chunk = min(rows_per_chunk, rows)
    full_chunks, tail_rows = divmod(rows, rows_per_chunk)
    chunks = [rows_per_chunk] * full_chunks
    if tail_rows:
        chunks.append(tail_rows)
    if not chunks or sum(chunks) != rows:
        raise AssertionError("CANN FIA Vector row chunks do not conserve rows")
    return tuple(chunks)


def _cann_fia_row_div_intrinsic_count(rows, actual_columns):
    full_repeats, tail_columns = divmod(
        actual_columns, _CANN_FIA_FP32_REPEAT_ELEMENTS)
    return min(full_repeats, rows) + int(tail_columns > 0)


def _cann_fia_brcb_physical_operations(rows):
    """Return elements written by the source Brcb repeat geometry."""

    if rows <= 0:
        raise ValueError("CANN FIA Brcb rows must be positive")
    return math.ceil(rows / 8) * 8 * 8


def _cann_fia_mla_vector_lane_rows(rows):
    """Return FiaBlockVecNonQuantMla's source-defined two-AIV M split."""

    if rows <= 0 or rows > 256:
        raise ValueError("CANN MLA Vector rows must be in [1, 256]")
    if rows <= 16:
        return rows, 0
    primary_rows = (((rows + 15) // 16 + 1) // 2) * 16
    primary_rows = min(primary_rows, rows)
    return primary_rows, rows - primary_rows


def _cann_fia_mla_vec1_lane_work(
        rows, actual_s2, score_columns, *, is_first, is_last):
    """Return one MLA AIV lane's typed Vec1 and nUpdate work."""

    if rows == 0:
        return (0, 0, 0, 0, 0, 0, ())
    generic = 0
    exp = 0
    div = 0
    exp_invocations = 0
    div_invocations = 0
    chunks = _cann_fia_vector_row_chunks(
        rows, score_columns, softmax_with_brc=False)
    for chunk_rows in chunks:
        chunk_actual = chunk_rows * actual_s2
        chunk_padded = chunk_rows * score_columns
        generic += 2 * chunk_actual + 3 * chunk_padded
        generic += (18 + int(is_first) + int(is_last)) * chunk_rows
        exp += chunk_padded + 2 * chunk_rows
        div += 2 * chunk_rows
        exp_invocations += 3
        div_invocations += 2
    nupdate_generic = (
        0 if is_first else _cann_fia_align(rows, 16) * 128)
    return (
        generic, exp, div, exp_invocations, div_invocations,
        nupdate_generic, chunks)


def _cann_fia_mla_vec2_lane_work(rows, output_columns, *, is_last):
    """Return one MLA AIV lane's final Vec2 work."""

    if rows == 0 or not is_last:
        return 0, 0, 0
    output_padded = rows * output_columns
    output_actual = rows * 512
    div_invocations = int(math.fsum(
        _cann_fia_row_div_intrinsic_count(chunk_rows, 512)
        for chunk_rows in _cann_fia_vector_row_chunks(
            rows, output_columns, softmax_with_brc=False)))
    return 4 * output_padded, output_actual, div_invocations


def _cann_fia_stage_services(values):
    services = []
    for service, operations, invocations in values:
        if operations == 0:
            if invocations != 0:
                raise ValueError(
                    "CANN FIA typed service has invocations without work")
            continue
        services.append(ComputeServiceDemand(
            service=service,
            operations=float(operations),
            serial_invocations=invocations,
        ))
    return tuple(services)


def _cann_fia_axis_slices(size, block_size):
    if size <= 0 or block_size <= 0:
        raise ValueError("CANN FIA source slice sizes must be positive")
    full, tail = divmod(size, block_size)
    slices = [block_size] * full
    if tail:
        slices.append(tail)
    return tuple(slices)


def _cann_fia_kv_source_slices(rows, *, paged):
    """Split one logical K/V panel at source-visible page boundaries."""

    if not paged:
        return (rows,)
    # CANN 8.5.1's paged GQA path resolves one 128-token block-table entry
    # before each ND-to-NZ copy.  The L1 tile may span several pages, but it
    # never turns those discontiguous pages into one physical descriptor.
    return _cann_fia_axis_slices(rows, 128)


def _cann_fia_mte2_stage(aic_group, geometries):
    if not geometries:
        raise ValueError("CANN FIA MTE2 stage must contain descriptors")
    return FiaMte2StageDemand(
        physical_worker=aic_group,
        descriptors=tuple(
            _cann_fia_mte2_descriptor(
                (rows, columns, source_row_stride_elements,
                 block_table_lookup_count), count=count)
            for (rows, columns, source_row_stride_elements,
                 block_table_lookup_count), run in (
                (geometry, tuple(group))
                for geometry, group in itertools.groupby(geometries)
            )
            for count in (len(run),)
        ),
    )


def _cann_fia_mte2_descriptor(geometry, *, count=1):
    (rows, columns, source_row_stride_elements,
     block_table_lookup_count) = geometry
    return FiaMte2Nd2NzDescriptorDemand(
        rows=rows,
        columns=columns,
        element_bytes=2,
        count=count,
        source_row_stride_elements=source_row_stride_elements,
        block_table_lookup_count=block_table_lookup_count,
    )


def _cann_fia_mte1_stage(
        aic_group, l0a_bytes, l0b_bytes, *, load_pair=None):
    return FiaMte1StageDemand(
        physical_worker=aic_group,
        l0a_bytes=l0a_bytes,
        l0b_bytes=l0b_bytes,
        load_pair=load_pair,
    )


def _cann_fia_fixpipe_stage(
        aic_group, rows, columns, destination_stride_elements):
    full_column_blocks, _tail_columns = divmod(columns, 128)
    descriptors = ()
    if full_column_blocks:
        descriptors = (FiaFixpipeDescriptorDemand(
            rows=rows,
            columns=128,
            destination_stride_elements=destination_stride_elements,
            count=full_column_blocks,
        ),)
    return FiaFixpipeStageDemand(
        physical_worker=aic_group,
        descriptors=descriptors,
    )


def _cann_fia_task_demand(
        kernel, hardware, aic_group, ordinal, task, *, repeat_count=1):
    """Derive one task's source compute and AIC MTE2 stages."""

    if not isinstance(kernel.launch.source_program, FiaCrossDomainLaunchProgram):
        raise TypeError("CANN FIA task demand requires a typed source program")
    runtime_variant = kernel.launch.algorithm.rsplit(":", 1)[-1]
    if runtime_variant not in _CANN_FIA_SOFTMAX_WITH_BRC:
        raise ValueError(
            "CANN FIA cross-domain plan has an unsupported source variant")
    softmax_with_brc = _CANN_FIA_SOFTMAX_WITH_BRC[runtime_variant]
    head_dim = int(kernel.shape["head_dim"])
    mla_dimensions = _cann_mla_dimensions(kernel)
    if mla_dimensions is None:
        qk_head_dim = head_dim
        value_head_dim = head_dim
    else:
        qk_head_dim, value_head_dim = mla_dimensions
    actual_m = task.actual_m
    actual_s2 = task.actual_s2
    mm1_tensor_logical = 2 * actual_m * actual_s2 * qk_head_dim
    mm1_tensor_physical = (
        2 * _cann_fia_align(actual_m, 16) * actual_s2 * qk_head_dim)
    mm2_tensor_logical = 2 * actual_m * actual_s2 * value_head_dim
    mm2_tensor_physical = (
        2 * _cann_fia_align(actual_m, 16) * actual_s2 * value_head_dim)
    score_columns = _cann_fia_align(actual_s2, 32)
    output_columns = _cann_fia_align(value_head_dim, 32)
    output_actual = actual_m * value_head_dim
    output_padded = actual_m * output_columns
    m_base_size = int(kernel.launch.parameters["cann_source_m_base_size"])
    s2_base_size = int(kernel.launch.parameters["cann_source_s2_base_size"])
    if actual_m > m_base_size or actual_s2 > s2_base_size:
        raise ValueError("CANN FIA task exceeds its source tile")
    is_mla = mla_dimensions is not None
    m_split_size = 128 if is_mla or s2_base_size == 1024 else 256
    n_split_size = 128 if is_mla else (
        256 if qk_head_dim <= 128 else 128)
    k_split_size = 256
    mm1_m_l1_slices = _cann_fia_axis_slices(actual_m, m_split_size)
    mm2_m_l1_slices = _cann_fia_axis_slices(actual_m, 128)
    n_l1_slices = _cann_fia_axis_slices(actual_s2, n_split_size)
    k_l1_slices = _cann_fia_axis_slices(actual_s2, k_split_size)
    if len(mm1_m_l1_slices) > 2:
        raise ValueError("CANN FIA Q snapshot exceeds its two L1 buffers")
    gqa_source_order = None
    mla_source_order = None
    if is_mla:
        # FiaBlockCubeNonQuantMla does not issue one 576-column ND2NZ
        # transfer.  Each 288-wide MM1 K half consists of a 256-column
        # no-PE copy and a separate 32-column RoPE copy.  Q is resident across
        # N tiles and its first half carries the complete 64-column RoPE row.
        qk_descriptors = tuple(
            (*descriptor, source_stride, 0)
            for rows in mm1_m_l1_slices
            for descriptor, source_stride in (
                ((rows, 256), 512), ((rows, 64), 64),
                ((rows, 256), 512))
        ) if task.is_head_first else ()
        qk_descriptors += tuple(
            (*descriptor, source_stride, 0)
            for rows in n_l1_slices
            for descriptor, source_stride in (
                ((rows, 256), 512), ((rows, 32), 64),
                ((rows, 32), 64), ((rows, 256), 512)))

        # MM2 copies one probability panel for each 256-column K slice, then
        # streams V as 128x128 descriptors for each output-D tile.  Keeping
        # those literal calls avoids inventing an out-of-domain 128-KiB
        # descriptor from the aggregate 512-column value width.
        pv_descriptors = tuple(
            (m_rows, k_rows, 512, 0)
            for m_rows in mm2_m_l1_slices
            for k_rows in k_l1_slices)
        pv_descriptors += tuple(
            (k_rows, d_columns, 512, 0)
            for k_rows in _cann_fia_axis_slices(actual_s2, 128)
            for d_columns in _cann_fia_axis_slices(value_head_dim, 128))
        # The aggregate MM2 stage groups equal P and V descriptors for its
        # service ledger.  Preserve the original source order separately:
        # ProcessMm2 copies the two 128-row V panels before its first-output
        # P panel, then reuses that P panel for later output-D tiles.
        mla_mm2_source_descriptors = []
        for output_ordinal, output_panel_columns in enumerate(
                _cann_fia_axis_slices(value_head_dim, 128)):
            for k_rows in k_l1_slices:
                mla_mm2_source_descriptors.extend(
                    (source_rows, output_panel_columns, 512, 0)
                    for source_rows in _cann_fia_axis_slices(k_rows, 128))
                if output_ordinal == 0:
                    mla_mm2_source_descriptors.extend(
                        (m_rows, k_rows, score_columns, 0)
                        for m_rows in mm2_m_l1_slices)
        mla_source_order = FiaMlaSourceOrderDemand(
            mm1_descriptors=tuple(
                _cann_fia_mte2_descriptor(descriptor)
                for descriptor in qk_descriptors),
            mm2_descriptors=tuple(
                _cann_fia_mte2_descriptor(descriptor)
                for descriptor in mla_mm2_source_descriptors),
        )
    else:
        qk_width_slices = _cann_fia_axis_slices(qk_head_dim, 256)
        contiguous_bsh = kernel.attributes["kv_layout"] == "contiguous"
        paged_kv = kernel.attributes["kv_layout"] in (
            "paged", "paged_varlen", "chunked_prefill")
        physical_kv_row_stride = (
            int(kernel.shape["num_kv_heads"]) * head_dim)
        # Paged KV rows retain the [block, token, kv-head, D] source stride.
        # Publish that stride for the independently measured D128/HKV8 page
        # geometry.  The descriptor service reference measures a repeated
        # 128x128 strided descriptor stream (20,000 iterations) on the
        # active-AIC axis; it does not impose a production descriptor-count
        # limit.  Descriptor count remains part of the outer source-order
        # schedule, while each eligible descriptor uses the measured local
        # strided service.
        measured_paged_kv_stride = (
            head_dim == 128
            and physical_kv_row_stride == 1024
            and (
                (
                    kernel.attributes["kv_layout"] == "paged"
                    and all(
                        len(group.tasks) == 1
                        for group in kernel.launch.source_program.aic_groups)
                )
                or (
                    paged_kv
                    and all(
                        task.actual_s2 % 128 == 0
                        for group in kernel.launch.source_program.aic_groups
                        for task in group.tasks)
                    and _strided_active_aic_group_is_measured(
                        hardware.aic_mte2_nd2nz_service,
                        kernel.launch.grid_size)
                )
            ))
        kv_source_row_stride = (
            physical_kv_row_stride
            if contiguous_bsh or measured_paged_kv_stride else None)
        qk_descriptors = tuple(
            (rows, columns, columns, 0)
            for rows in mm1_m_l1_slices
            for columns in qk_width_slices
        ) if task.is_head_first else ()
        qk_descriptors += tuple(
            (source_rows, columns, kv_source_row_stride, int(paged_kv))
            for rows in n_l1_slices
            for source_rows in _cann_fia_kv_source_slices(
                rows, paged=paged_kv)
            for columns in qk_width_slices)
        v_buffer_count = 4 if s2_base_size == 1024 else 2
        can_full_load_v = len(k_l1_slices) <= v_buffer_count
        pv_descriptors = tuple(
            (m_rows, k_rows, score_columns, 0)
            for m_rows in mm2_m_l1_slices
            for k_rows in k_l1_slices)
        if can_full_load_v:
            pv_descriptors += tuple(
                (source_rows, columns, kv_source_row_stride, int(paged_kv))
                for k_rows in k_l1_slices
                for source_rows in _cann_fia_kv_source_slices(
                    k_rows, paged=paged_kv)
                for columns in _cann_fia_axis_slices(value_head_dim, 256))
        else:
            pv_descriptors += tuple(
                (source_rows, columns, kv_source_row_stride, int(paged_kv))
                for _m_rows in mm2_m_l1_slices
                for k_rows in k_l1_slices
                for source_rows in _cann_fia_kv_source_slices(
                    k_rows, paged=paged_kv)
                for columns in _cann_fia_axis_slices(value_head_dim, 256))
        # ``FiaBlockCubeNonQuantGqa`` keeps its KP and V L1 lifetimes in
        # logical 256-KV rounds even when the last K/V copy is a short page.
        # Keep the aggregate stages compressed above, but publish the literal
        # round boundaries needed by the finite source-order solver.  This is
        # limited to the source topology represented here: one Q/D slice and
        # one MM2 M slice; other source programs retain their existing model.
        value_width_slices = _cann_fia_axis_slices(value_head_dim, 256)
        if (len(mm1_m_l1_slices) == 1
                and len(mm2_m_l1_slices) == 1
                and len(qk_width_slices) == 1
                and len(value_width_slices) == 1
                and len(n_l1_slices) == len(k_l1_slices)
                and (not task.is_head_first or len(qk_descriptors) >= 1)):
            query_descriptor = (
                _cann_fia_mte2_descriptor(qk_descriptors[0])
                if task.is_head_first else None)
            qk_columns, = qk_width_slices
            value_columns, = value_width_slices
            rounds = tuple(FiaGqaSourceRoundDemand(
                key_descriptors=tuple(
                    _cann_fia_mte2_descriptor(
                        (source_rows, qk_columns, kv_source_row_stride,
                         int(paged_kv)))
                    for source_rows in _cann_fia_kv_source_slices(
                        n_rows, paged=paged_kv)),
                probability_descriptor=_cann_fia_mte2_descriptor(
                    (mm2_m_l1_slices[0], k_rows, score_columns, 0)),
                value_descriptors=tuple(
                    _cann_fia_mte2_descriptor(
                        (source_rows, value_columns, kv_source_row_stride,
                         int(paged_kv)))
                    for source_rows in _cann_fia_kv_source_slices(
                        k_rows, paged=paged_kv)),
                panels=tuple(FiaGqaSourcePanelDemand(
                    source_columns=panel_columns,
                    mm1_mte1=_cann_fia_mte1_stage(
                        aic_group,
                        _cann_fia_align(actual_m, 16)
                        * _cann_fia_align(qk_head_dim, 16) * 2,
                        _cann_fia_align(panel_columns, 16)
                        * _cann_fia_align(qk_head_dim, 16) * 2),
                    mm1=FiaComputeStageDemand(
                        resource_role="compute_tensor",
                        physical_worker=aic_group,
                        logical_operations=float(
                            2 * actual_m * panel_columns * qk_head_dim),
                        physical_operations=float(
                            2 * _cann_fia_align(actual_m, 16)
                            * panel_columns * qk_head_dim),
                    ),
                    mm1_fixpipe=_cann_fia_fixpipe_stage(
                        aic_group, actual_m, panel_columns, score_columns),
                    mm2_mte1=_cann_fia_mte1_stage(
                        aic_group,
                        _cann_fia_align(actual_m, 16)
                        * _cann_fia_align(panel_columns, 16) * 2,
                        _cann_fia_align(panel_columns, 16)
                        * _cann_fia_align(value_head_dim, 16) * 2),
                    mm2=FiaComputeStageDemand(
                        resource_role="compute_tensor",
                        physical_worker=aic_group,
                        logical_operations=float(
                            2 * actual_m * panel_columns
                            * value_head_dim),
                        physical_operations=float(
                            2 * _cann_fia_align(actual_m, 16)
                            * panel_columns * value_head_dim),
                    ),
                ) for panel_columns in _cann_fia_axis_slices(k_rows, 128)),
            ) for n_rows, k_rows in zip(n_l1_slices, k_l1_slices))
            gqa_source_order = FiaGqaSourceOrderDemand(
                query_descriptor=query_descriptor, rounds=rounds)
    qk_k_l0_slices = _cann_fia_axis_slices(qk_head_dim, 128)
    mm1_l0a_bytes = 0
    mm1_l0b_bytes = 0
    mm1_load_pair_count = 0
    for m_rows in mm1_m_l1_slices:
        for n_rows in n_l1_slices:
            for m_l0_rows in _cann_fia_axis_slices(m_rows, 128):
                for n_l0_rows in _cann_fia_axis_slices(n_rows, 128):
                    for k_l0_rows in qk_k_l0_slices:
                        mm1_l0a_bytes += (
                            _cann_fia_align(m_l0_rows, 16)
                            * _cann_fia_align(k_l0_rows, 16) * 2)
                        mm1_l0b_bytes += (
                            _cann_fia_align(n_l0_rows, 16)
                            * _cann_fia_align(k_l0_rows, 16) * 2)
                        mm1_load_pair_count += 1
    mm2_l0a_bytes = 0
    mm2_l0b_bytes = 0
    mm2_load_pair_count = 0
    for m_rows in mm2_m_l1_slices:
        for k_l1_rows in k_l1_slices:
            for k_l0_rows in _cann_fia_axis_slices(k_l1_rows, 128):
                mm2_l0a_bytes += (
                    _cann_fia_align(m_rows, 16)
                    * _cann_fia_align(k_l0_rows, 16) * 2)
                mm2_l0b_bytes += (
                    _cann_fia_align(k_l0_rows, 16)
                    * _cann_fia_align(value_head_dim, 16) * 2)
                mm2_load_pair_count += 1

    mm1_load_pair = None
    mm2_load_pair = None
    if (not is_mla and qk_head_dim in (64, 128)
            and max(mm1_m_l1_slices) <= 16
            and all(
                value == 128
                for n_rows in n_l1_slices
                for value in _cann_fia_axis_slices(n_rows, 128))):
        mm1_load_pair = FiaMte1LoadPairDemand(
            stage_kind="mm1", head_dim=qk_head_dim,
            count=mm1_load_pair_count)
    if (not is_mla and value_head_dim in (64, 128)
            and max(mm2_m_l1_slices) <= 16
            and all(value == 128 for value in (
                _cann_fia_axis_slices(actual_s2, 128)))):
        mm2_load_pair = FiaMte1LoadPairDemand(
            stage_kind="mm2", head_dim=value_head_dim,
            count=mm2_load_pair_count)

    reduction_mechanisms = (
        hardware.fia_block_reduce_sum_service,
        hardware.fia_block_reduce_max_service,
    )
    brcb_mechanism = hardware.fia_brcb_service

    def block_reduce_descriptors(vector_row_chunks):
        if (any(mechanism is None for mechanism in reduction_mechanisms)
                or brcb_mechanism is None):
            return ()
        candidates = tuple(FiaBlockReduceSumDescriptorDemand(
            row_chunk_round=_cann_fia_align(chunk_rows, 8),
            column_count=actual_s2,
            column_alignment=32,
        ) for chunk_rows in vector_row_chunks)
        if all(
                all(
                    descriptor.row_chunk_round
                    <= mechanism["maximum_row_chunk"]
                    and descriptor.column_count
                    <= mechanism["maximum_column_count"]
                    and descriptor.row_chunk_round
                    * _cann_fia_align(
                        descriptor.column_count,
                        descriptor.column_alignment)
                    <= mechanism["maximum_source_elements"]
                    for mechanism in reduction_mechanisms)
                and brcb_mechanism["minimum_row_chunk"]
                <= descriptor.row_chunk_round
                <= brcb_mechanism["maximum_row_chunk"]
                and (descriptor.row_chunk_round
                     % brcb_mechanism["row_granularity"] == 0)
                and (descriptor.column_count != 512
                     or descriptor.row_chunk_round in (8, 16))
                for descriptor in candidates):
            return candidates
        return ()

    def vector_stage(worker, generic, exp, div, exp_invocations,
                     div_invocations):
        return FiaComputeStageDemand(
            resource_role="compute_vector",
            physical_worker=worker,
            logical_operations=float(generic + exp + div),
            physical_operations=float(generic + exp + div),
            typed_services=_cann_fia_stage_services((
                ("generic", generic, 0),
                ("exp", exp, exp_invocations),
                ("div", div, div_invocations),
            )),
        )

    vector_worker = 2 * aic_group
    vec1_update = None
    vec1_peer = None
    vec1_peer_update = None
    vec2_peer = None
    vec1_update_memory = None
    vec1_peer_memory = None
    vec1_peer_update_memory = None
    vec2_peer_memory = None
    vec1_peer_block_reduce_sum = ()
    if is_mla:
        lane_stages = []
        for lane, lane_rows in enumerate(
                _cann_fia_mla_vector_lane_rows(actual_m)):
            if lane_rows == 0:
                lane_stages.append(None)
                continue
            (vec1_generic, vec1_exp, vec1_div,
             vec1_exp_invocations, vec1_div_invocations,
             nupdate_generic, vector_row_chunks) = (
                 _cann_fia_mla_vec1_lane_work(
                     lane_rows, actual_s2, score_columns,
                     is_first=task.is_head_first,
                     is_last=task.is_head_last))
            vec2_generic, vec2_div, vec2_div_invocations = (
                _cann_fia_mla_vec2_lane_work(
                    lane_rows, output_columns, is_last=task.is_head_last))
            worker = vector_worker + lane
            update_stage = (
                None if nupdate_generic == 0 else vector_stage(
                    worker, nupdate_generic, 0, 0, 0, 0))
            vec1_memory = FiaAivMemoryStageDemand(
                physical_worker=worker,
                mte2_bytes=4 * lane_rows * score_columns,
                mte3_bytes=2 * lane_rows * score_columns,
            )
            update_memory = (
                None if update_stage is None else FiaAivMemoryStageDemand(
                    physical_worker=worker,
                    mte2_bytes=0,
                    mte3_bytes=(
                        _cann_fia_align(lane_rows, 16) * 512 * 4),
                ))
            if task.is_head_last:
                vec2_write_bytes = (
                    lane_rows * value_head_dim * 4
                    + lane_rows * 8 * 4 * 2
                    if task.is_s2_split_core
                    else lane_rows * value_head_dim * 2)
                vec2_memory = FiaAivMemoryStageDemand(
                    physical_worker=worker,
                    mte2_bytes=4 * lane_rows * output_columns,
                    mte3_bytes=vec2_write_bytes,
                )
            else:
                vec2_memory = FiaAivMemoryStageDemand(
                    physical_worker=worker, mte2_bytes=0, mte3_bytes=0)
            lane_stages.append((
                vector_stage(
                    worker, vec1_generic, vec1_exp, vec1_div,
                    vec1_exp_invocations, vec1_div_invocations),
                vec1_memory,
                update_stage,
                update_memory,
                vector_stage(
                    worker, vec2_generic, 0, vec2_div, 0,
                    vec2_div_invocations),
                vec2_memory,
                block_reduce_descriptors(vector_row_chunks),
            ))
        (vec1, vec1_memory, vec1_update, vec1_update_memory,
         vec2, vec2_memory,
         block_reduce_sum_descriptors) = lane_stages[0]
        if lane_stages[1] is not None:
            (vec1_peer, vec1_peer_memory,
             vec1_peer_update, vec1_peer_update_memory,
             vec2_peer, vec2_peer_memory,
             vec1_peer_block_reduce_sum) = lane_stages[1]
    else:
        vec1_generic = 0
        vec1_exp = 0
        vec1_exp_invocations = 0
        vector_row_chunks = _cann_fia_vector_row_chunks(
            actual_m, score_columns, softmax_with_brc=softmax_with_brc)
        for chunk_rows in vector_row_chunks:
            chunk_actual = chunk_rows * actual_s2
            chunk_padded = chunk_rows * score_columns
            vec1_generic += 2 * chunk_actual + 2 * chunk_padded
            vec1_exp += chunk_padded + chunk_rows
            vec1_exp_invocations += 2
        block_reduce_sum_descriptors = block_reduce_descriptors(
            vector_row_chunks)
        vec2_generic = 0
        vec2_div = 0
        vec2_div_invocations = 0
        if not task.is_head_first:
            vec2_generic += output_actual + output_padded
        if task.is_head_last:
            vec2_generic += output_padded
            if not task.is_s2_split_core:
                vec2_div = output_actual
                vec2_div_invocations = int(math.fsum(
                    _cann_fia_row_div_intrinsic_count(
                        chunk_rows, value_head_dim)
                    for chunk_rows in _cann_fia_vector_row_chunks(
                        actual_m, output_columns,
                        softmax_with_brc=softmax_with_brc)))
        vec2_generic += _cann_fia_brcb_physical_operations(actual_m)
        vec1 = vector_stage(
            vector_worker, vec1_generic, vec1_exp, 0,
            vec1_exp_invocations, 0)
        vec2 = vector_stage(
            vector_worker, vec2_generic, 0, vec2_div, 0,
            vec2_div_invocations)
        transfer_service = hardware.fia_aiv_workspace_transfer_service
        transfer_group_eligible = (
            transfer_service is not None
            and kernel.launch.grid_size
            in transfer_service["active_aic_groups"]
            and head_dim == 64
            and actual_m == 4)
        if transfer_group_eligible and actual_s2 == 512:
            vec1_completion_service = "vec1_m4_s512_ordered_chain"
        elif transfer_group_eligible and actual_s2 == 1024:
            vec1_completion_service = "vec1_m4_s1024_ordered_chain"
        else:
            vec1_completion_service = None
        vec1_memory = FiaAivMemoryStageDemand(
            physical_worker=vector_worker,
            mte2_bytes=4 * actual_m * score_columns,
            mte3_bytes=2 * actual_m * score_columns,
            completion_service=vec1_completion_service,
        )
        if task.is_head_last:
            vec2_write_bytes = (
                actual_m * value_head_dim * 4
                + actual_m * 8 * 4 * 2
                if task.is_s2_split_core
                else actual_m * value_head_dim * 2)
        else:
            vec2_write_bytes = 4 * output_padded
        vec2_completion_service = (
            "vec2_m4_final_ordered_chain"
            if transfer_group_eligible and task.is_head_last else None)
        vec2_memory = FiaAivMemoryStageDemand(
            physical_worker=vector_worker,
            mte2_bytes=(
                4 * output_padded
                + (0 if task.is_head_first else 4 * output_padded)),
            mte3_bytes=vec2_write_bytes,
            completion_service=vec2_completion_service,
        )

    return FiaCrossDomainTaskDemand(
        ordinal=ordinal,
        mm1_mte2=_cann_fia_mte2_stage(aic_group, qk_descriptors),
        mm1_mte1=_cann_fia_mte1_stage(
            aic_group, mm1_l0a_bytes, mm1_l0b_bytes,
            load_pair=mm1_load_pair),
        mm1=FiaComputeStageDemand(
            resource_role="compute_tensor",
            physical_worker=aic_group,
            logical_operations=float(mm1_tensor_logical),
            physical_operations=float(mm1_tensor_physical),
        ),
        mm1_fixpipe=_cann_fia_fixpipe_stage(
            aic_group, actual_m, actual_s2, score_columns),
        vec1=vec1,
        mm2_mte2=_cann_fia_mte2_stage(aic_group, pv_descriptors),
        mm2_mte1=_cann_fia_mte1_stage(
            aic_group, mm2_l0a_bytes, mm2_l0b_bytes,
            load_pair=mm2_load_pair),
        mm2=FiaComputeStageDemand(
            resource_role="compute_tensor",
            physical_worker=aic_group,
            logical_operations=float(mm2_tensor_logical),
            physical_operations=float(mm2_tensor_physical),
        ),
        mm2_fixpipe=_cann_fia_fixpipe_stage(
            aic_group, actual_m, value_head_dim, output_columns),
        vec2=vec2,
        vec1_memory=vec1_memory,
        vec2_memory=vec2_memory,
        vec1_update_memory=vec1_update_memory,
        vec1_peer_memory=vec1_peer_memory,
        vec1_peer_update_memory=vec1_peer_update_memory,
        vec2_peer_memory=vec2_peer_memory,
        vec1_block_reduce_sum=block_reduce_sum_descriptors,
        vec1_update=vec1_update,
        vec1_peer=vec1_peer,
        vec1_peer_update=vec1_peer_update,
        vec2_peer=vec2_peer,
        vec1_peer_block_reduce_sum=vec1_peer_block_reduce_sum,
        gqa_source_order=gqa_source_order,
        mla_source_order=mla_source_order,
        aic_pipeline_kind=(
            "fia_block_cube_nonquant_mla_multibuffer"
            if is_mla else "serial_stages"),
        repeat_count=repeat_count,
    )


def _cann_fia_initialization_stages(kernel, groups):
    """Return the two source Duplicate calls issued by every launched AIV."""

    worker_count = 2 * sum(group.class_count for group in groups)
    expected_operations = (
        worker_count * _CANN_FIA_AIV_INITIALIZATION_GENERIC_OPERATIONS)
    if kernel.launch.parameters[
            "cann_source_vector_initialization_operations"] != float(
                expected_operations):
        raise ValueError(
            "CANN FIA AIV initialization disagrees with the source launch")
    return tuple(
        FiaComputeStageDemand(
            resource_role="compute_vector",
            physical_worker=worker,
            logical_operations=float(
                _CANN_FIA_AIV_INITIALIZATION_GENERIC_OPERATIONS),
            physical_operations=float(
                _CANN_FIA_AIV_INITIALIZATION_GENERIC_OPERATIONS),
            typed_services=(ComputeServiceDemand(
                service="generic",
                operations=float(
                    _CANN_FIA_AIV_INITIALIZATION_GENERIC_OPERATIONS),
            ),),
        )
        for worker in range(worker_count)
    )


def _cann_fia_terminal_vector_stages(
        inp, kernel, groups, initialization_stages):
    """Split source-counted FlashDecode work from the main AIV schedule."""

    terminal_operations = kernel.launch.parameters[
        "cann_source_vector_terminal_operations"]
    if terminal_operations == 0:
        return ()
    vector_demand, = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector")
    aggregate_by_service = {
        demand.service: demand for demand in vector_demand.service_demands}
    if "generic" not in aggregate_by_service:
        raise ValueError(
            "CANN FlashDecode terminal requires aggregate generic Vector work")
    worker_count = 2 * sum(group.class_count for group in groups)
    main_by_service = {
        name: [0.0] * worker_count for name in aggregate_by_service}
    for stage in initialization_stages:
        for service in stage.typed_services:
            if service.service not in main_by_service:
                raise ValueError(
                    "CANN FIA initialization contains an unknown Vector "
                    "service")
            main_by_service[service.service][stage.physical_worker] += (
                service.operations)
    for group in groups:
        for task in group.tasks:
            stages = [task.vec1, task.vec2]
            for optional_stage in (
                    task.vec1_update, task.vec1_peer,
                    task.vec1_peer_update, task.vec2_peer):
                if optional_stage is not None:
                    stages.append(optional_stage)
            for stage in stages:
                for service in stage.typed_services:
                    if service.service not in main_by_service:
                        raise ValueError(
                            "CANN FIA main program contains an unknown Vector "
                            "service")
                    lane = (
                        stage.physical_worker - 2 * group.aic_worker)
                    if lane not in (0, 1):
                        raise ValueError(
                            "CANN FIA stage is outside its AIC group")
                    for aic_worker in group.aic_workers:
                        worker = 2 * aic_worker + lane
                        main_by_service[service.service][worker] += (
                            task.repeat_count * service.operations)

    terminal_by_worker = []
    for service_name, aggregate in aggregate_by_service.items():
        if len(aggregate.operations_by_worker) != worker_count:
            raise ValueError(
                "CANN FIA aggregate Vector worker count changed")
        residuals = tuple(
            aggregate.operations_by_worker[worker] -
            main_by_service[service_name][worker]
            for worker in range(worker_count))
        if any(value < 0 for value in residuals):
            raise ValueError(
                "CANN FIA main program exceeds aggregate Vector work")
        if service_name == "generic":
            terminal_by_worker = list(residuals)
        elif any(value != 0 for value in residuals):
            raise ValueError(
                "CANN FlashDecode terminal may only use generic Vector work")
    if not math.isclose(
            math.fsum(terminal_by_worker), terminal_operations,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN FlashDecode terminal workers do not conserve terminal work")
    return tuple(
        FiaComputeStageDemand(
            resource_role="compute_vector",
            physical_worker=worker,
            logical_operations=float(operations),
            physical_operations=float(operations),
            typed_services=(ComputeServiceDemand(
                service="generic",
                operations=float(operations),
            ),),
        )
        for worker, operations in enumerate(terminal_by_worker)
        if operations > 0)


def _attach_cann_fia_cross_domain_plan(
        inp, kernel, hardware, *, attach_pipeline):
    """Attach a bounded per-AIC source program to the unified formula."""

    if _has_graph_padding(kernel):
        return inp
    program = kernel.launch.source_program
    if program is None or not attach_pipeline:
        return inp
    if not isinstance(program, FiaCrossDomainLaunchProgram):
        raise TypeError("CANN FIA source program has an invalid type")
    if len(program.aic_groups) != kernel.launch.grid_size:
        raise ValueError("CANN FIA source program disagrees with launch grid")
    task_counts = tuple(
        len(group.tasks) for group in program.aic_groups)
    mechanism = hardware.fia_cross_domain_three_handoff
    sync_all_mechanism = hardware.fia_aiv_sync_all_service
    terminal_prelude_mechanism = (
        hardware.fia_flash_decode_terminal_prelude_service)
    eligible = (
        mechanism is not None
        and sync_all_mechanism is not None
        and all(
            mechanism["minimum_task_count"] <= task_count
            <= mechanism["maximum_task_count"]
            for task_count in task_counts)
        and mechanism["minimum_active_aic_groups"] <= len(program.aic_groups)
        <= mechanism["maximum_active_aic_groups"]
        and len(program.aic_groups)
        in mechanism["single_task_active_aic_groups"])
    if not eligible:
        limitation = "cann_fia_cross_domain_program_span_domain_unavailable"
        if limitation in inp.limitations:
            raise ValueError("CANN FIA span limitation was attached twice")
        return dataclasses.replace(
            inp, limitations=inp.limitations + (limitation,))

    def _source_task_runs(group):
        runs = []
        ordinal = 0
        run_start = 0
        while run_start < len(group.tasks):
            source_task = group.tasks[run_start]
            identity = (
                source_task.actual_m,
                source_task.actual_s2,
                source_task.is_head_first,
                source_task.is_head_last,
                source_task.is_s2_split_core,
            )
            run_end = run_start + 1
            while run_end < len(group.tasks):
                candidate = group.tasks[run_end]
                candidate_identity = (
                    candidate.actual_m,
                    candidate.actual_s2,
                    candidate.is_head_first,
                    candidate.is_head_last,
                    candidate.is_s2_split_core,
                )
                if candidate_identity != identity:
                    break
                run_end += 1
            repeat_count = run_end - run_start
            runs.append((ordinal, source_task, repeat_count))
            ordinal += repeat_count
            run_start = run_end
        return tuple(runs)

    source_classes = {}
    for group in program.aic_groups:
        task_runs = _source_task_runs(group)
        program_key = tuple(
            (
                task.actual_m,
                task.actual_s2,
                task.is_head_first,
                task.is_head_last,
                task.is_s2_split_core,
                repeat_count,
            )
            for _, task, repeat_count in task_runs)
        if program_key in source_classes:
            source_classes[program_key].append((group, task_runs))
        else:
            source_classes[program_key] = [(group, task_runs)]
    # A source run only differs in ordinal/repetition after its static tile
    # class and physical AIC have been fixed.  The launch program commonly
    # revisits that class many times (for example one Q-head round per KV
    # panel).  Materializing its ND2NZ, L1/L0 and Vector demand tree on every
    # revisit is a Python replay of the source loop, not additional physical
    # work.  Build one immutable demand template per source task class/AIC and
    # retain the source run's ordinal and repetition on its shallow wrapper.
    task_templates = {}

    def _task_demand_for_run(aic_group, ordinal, source_task, repeat_count):
        task_key = (
            aic_group,
            source_task.actual_m,
            source_task.actual_s2,
            source_task.is_head_first,
            source_task.is_head_last,
            source_task.is_s2_split_core,
        )
        if task_key in task_templates:
            template = task_templates[task_key]
        else:
            template = _cann_fia_task_demand(
                kernel,
                hardware,
                aic_group,
                0,
                source_task,
            )
            task_templates[task_key] = template
        return dataclasses.replace(
            template,
            ordinal=ordinal,
            repeat_count=repeat_count,
        )

    groups = []
    for matching_programs in source_classes.values():
        representative, task_runs = matching_programs[0]
        groups.append(FiaCrossDomainAicGroupDemand(
            aic_worker=representative.aic_group,
            aiv_workers=representative.aiv_workers,
            tasks=tuple(
                _task_demand_for_run(
                    representative.aic_group,
                    ordinal,
                    source_task,
                    repeat_count,
                )
                for ordinal, source_task, repeat_count in task_runs),
            equivalent_aic_workers=tuple(
                group.aic_group
                for group, _ in matching_programs[1:]),
        ))
    groups = tuple(groups)
    mte2_mechanism = hardware.aic_mte2_nd2nz_service
    def _mte2_descriptor_is_eligible(descriptor):
        source_stride = descriptor.source_row_stride_elements
        normalized_source_stride = (
            descriptor.columns if source_stride is None else source_stride)
        if hardware.aic_mte2_fia_mla_strided_supports(
                descriptor.rows,
                descriptor.columns,
                normalized_source_stride,
                kernel.launch.grid_size):
            return True
        if hardware.aic_mte2_nd2nz_cold_completion_supports(
                descriptor.rows,
                descriptor.columns,
                normalized_source_stride,
                kernel.launch.grid_size):
            return True
        # A probability-panel copy spans at most one 16-row Cube-aligned M
        # panel.  Its source stride is still published above, while the
        # existing byte service remains a conservative serial-completion
        # bound when no exact cold geometry owns it.  Long K/V panels retain
        # the strict strided-geometry gate.
        if (source_stride is None
                or source_stride == descriptor.columns
                or descriptor.rows <= 16):
            return (
                descriptor.bytes_per_descriptor
                <= mte2_mechanism["descriptor_bytes"][-1])
        return (
            _strided_active_aic_group_is_measured(
                mte2_mechanism, kernel.launch.grid_size)
            and (
                descriptor.rows,
                descriptor.columns,
                source_stride,
            ) in mte2_mechanism["strided_geometries"])

    mte2_eligible = (
        mte2_mechanism is not None
        and mte2_mechanism["active_aic_groups"][0]
        <= kernel.launch.grid_size
        <= mte2_mechanism["active_aic_groups"][-1]
        and all(
            _mte2_descriptor_is_eligible(descriptor)
            for group in groups
            for task in group.tasks
            for stage in (task.mm1_mte2, task.mm2_mte2)
            for descriptor in stage.descriptors
        )
    )
    if not mte2_eligible:
        limitation = "cann_fia_mte2_nd2nz_service_domain_unavailable"
        if limitation in inp.limitations:
            raise ValueError("CANN FIA MTE2 limitation was attached twice")
        return dataclasses.replace(
            inp, limitations=inp.limitations + (limitation,))
    fixpipe_mechanism = hardware.aic_fixpipe_full_tile_service
    fixpipe_eligible = (
        fixpipe_mechanism is not None
        and all(
            descriptor.rows in fixpipe_mechanism["supported_rows"]
            and descriptor.columns == fixpipe_mechanism["column_count"]
            and descriptor.destination_stride_elements
            % fixpipe_mechanism[
                "destination_stride_alignment_elements"] == 0
            and descriptor.destination_stride_elements
            <= fixpipe_mechanism[
                "maximum_destination_stride_elements"]
            for group in groups
            for task in group.tasks
            for stage in (task.mm1_fixpipe, task.mm2_fixpipe)
            for descriptor in stage.descriptors
        )
    )
    if not fixpipe_eligible:
        limitation = "cann_fia_fixpipe_service_domain_unavailable"
        if limitation in inp.limitations:
            raise ValueError("CANN FIA Fixpipe limitation was attached twice")
        return dataclasses.replace(
            inp, limitations=inp.limitations + (limitation,))
    initialization_stages = _cann_fia_initialization_stages(kernel, groups)
    terminal_vector_stages = _cann_fia_terminal_vector_stages(
        inp, kernel, groups, initialization_stages)
    tiling_mechanism = hardware.fia_tiling_data_materialization_service
    tiling_data_prelude = None
    if tiling_mechanism is None:
        limitation = "cann_fia_tiling_data_materialization_unavailable"
        if limitation in inp.limitations:
            raise ValueError("CANN FIA tiling limitation was attached twice")
        inp = dataclasses.replace(
            inp, limitations=inp.limitations + (limitation,))
    else:
        active_aic_groups = len(program.aic_groups)
        if not (
                tiling_mechanism["minimum_active_aic_groups"]
                <= active_aic_groups
                <= tiling_mechanism["maximum_active_aic_groups"]):
            raise ValueError(
                "CANN FIA tiling materialization is outside its measured "
                "active-group domain")
        payload_bytes = tiling_mechanism['payload_bytes']
        aiv_transfer_bytes = tiling_mechanism['aiv_transfer_bytes']
        if has_cann_attention_source(kernel):
            source = current_cann_attention_source_work(kernel,hardware)
            if (source['argument_aic_materializations'] != active_aic_groups
                    or source['argument_aiv_materializations'] != active_aic_groups*tiling_mechanism['aiv_per_group']):
                raise ValueError('CANN prelude actors disagree with actual argument source')
            payload_bytes = source['argument_payload_bytes']
            aiv_transfer_bytes = source['argument_aiv_transfer_bytes']
        tiling_data_prelude = FiaTilingDataPreludeDemand(
            mechanism_id=tiling_mechanism["model"],
            payload_bytes=payload_bytes,
            aiv_transfer_bytes=aiv_transfer_bytes,
            active_aic_groups=active_aic_groups,
        )
    plan = FiaCrossDomainWorkPlan(
        mechanism_id="cann_fia_1_2_three_handoff_v5",
        groups=groups,
        terminal_vector_stages=terminal_vector_stages,
        aiv_initialization_stages=initialization_stages,
        tiling_data_prelude=tiling_data_prelude,
        source_s2_base_size=int(
            kernel.launch.parameters["cann_source_s2_base_size"]),
    )
    unavailable = "cann_flash_attention_tile_pipeline_unavailable"
    if unavailable not in inp.limitations:
        raise ValueError(
            "CANN FIA source plan expected the aggregate pipeline limitation")
    limitations = tuple(
        value for value in inp.limitations if value != unavailable)
    fixpipe_unavailable = "cann_fia_fixpipe_service_evidence_unavailable"
    limitations = tuple(
        value for value in limitations if value != fixpipe_unavailable)
    nupdate_unavailable = "cann_mla_nupdate_pipeline_unavailable"
    limitations = tuple(
        value for value in limitations if value != nupdate_unavailable)
    has_fixpipe_tail = (
        int(kernel.shape["head_dim"]) % 128 != 0
        or any(
            task.actual_s2 % 128 != 0
            for group in program.aic_groups
            for task in group.tasks
        )
    )
    if has_fixpipe_tail:
        limitations += (
            "cann_fia_fixpipe_tail_service_evidence_unavailable",
        )
    limitations += (
        "cann_fia_cross_domain_edge_latency_distribution_unavailable",
    )
    if terminal_vector_stages and terminal_prelude_mechanism is None:
        limitations += (
            "cann_fia_flash_decode_syncall_latency_unavailable",
        )
    return dataclasses.replace(
        inp,
        pipeline_plan=plan,
        support_level="partial",
        limitations=limitations,
    )


def _cann_mla_dimensions(kernel: KernelSpec):
    """Return the source-fixed CANN MLA QK and value dimensions."""

    if kernel.launch.algorithm not in _CANN_MLA_DECODE_ALGORITHMS:
        return None
    if kernel.variant.lower() != "decode":
        raise ValueError("CANN MLA variants require decode execution")
    if kernel.dtype.lower() != "bf16":
        raise ValueError("CANN MLA variants require BF16")
    if kernel.attributes["kv_layout"] != "mla":
        raise ValueError("CANN MLA variants require kv_layout='mla'")
    if kernel.attributes["causal"] is not False:
        raise ValueError("CANN MLA decode variants require non-causal attention")
    _, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    if q_len != 1 or any(length != 1 for length in q_lens):
        raise ValueError("CANN MLA variants require one query row per request")
    if (head_dim, query_heads, kv_heads) != (512, 64, 1):
        raise ValueError(
            "CANN MLA source contract requires D=512, HQ=64, and HKV=1")
    return head_dim + 64, head_dim


def _metax_mla_dimensions(kernel: KernelSpec):
    """Return the host-ABI-proven MetaX MLA QK and value dimensions."""

    if kernel.launch.algorithm not in _METAX_MLA_DECODE_ALGORITHMS:
        return None
    if kernel.variant.lower() != "decode":
        raise ValueError("MetaX MLA launch requires decode execution")
    if kernel.dtype.lower() != "bf16":
        raise ValueError("MetaX MLA launch requires BF16")
    if kernel.attributes["kv_layout"] != "mla":
        raise ValueError("MetaX MLA launch requires kv_layout='mla'")
    if kernel.attributes["causal"] is not True:
        raise ValueError("MetaX MLA launch requires executed causal=True")
    _, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    if q_len != 1 or any(length != 1 for length in q_lens):
        raise ValueError("MetaX MLA launch requires one query row per request")
    if (head_dim, query_heads, kv_heads) != (512, 64, 1):
        raise ValueError(
            "MetaX MLA host ABI requires D=512, HQ=64, and HKV=1")
    return head_dim + 64, head_dim


def _mla_dimensions(kernel: KernelSpec):
    """Return one source-bound asymmetric MLA dimension contract, if any."""

    cann = _cann_mla_dimensions(kernel)
    metax = _metax_mla_dimensions(kernel)
    if cann is not None and metax is not None:
        raise AssertionError("FLASH_ATTN launch matched two MLA contracts")
    return cann if cann is not None else metax


def _is_unbound_metax_mla_scratch(kernel: KernelSpec) -> bool:
    """Whether MACA 3.2.1 MLA scratch exists but has no bound layout source."""

    return (
        kernel.launch.algorithm in _UNBOUND_MLA_SCRATCH_ALGORITHMS
        and not _is_source_bound_metax_mla_main(kernel)
        and not _is_source_bound_metax_mla_combine(kernel)
        and kernel.attributes["kv_layout"] == "mla"
        and kernel.variant.lower() in ("decode", "combine", "split_combine")
    )


def _is_source_bound_fa2_combine(kernel: KernelSpec) -> bool:
    """Whether the launch is the pinned SM80 FA2 split reducer."""

    return (not _has_graph_padding(kernel)
            and kernel.launch.algorithm == "flash_attn_splitkv_combine")


def _is_source_bound_metax_fa2_combine(kernel: KernelSpec) -> bool:
    """Whether the launch has one of the pinned MACA D128 reducer programs.

    The evidence inventory deliberately distinguishes the two-split full
    program from the kMaxSplits=8 partial-predicate program.  Runtime
    ``splits=8`` is the latter program's *full* predicate path, for which we
    have no isolated dependency fragment, so it must retain the generic
    analytical reducer rather than borrowing the partial-path service.
    """

    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm != "metax_flash_attention_generic"):
        return False
    _, _, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    splits = _splits(kernel)
    has_pinned_fragment = splits == 2 or 2 < splits < 8
    if not (
            kernel.dtype.lower() == "bf16"
            and kernel.variant.lower() in ("combine", "split_combine")
            and kernel.attributes["kv_layout"] != "mla"
            and head_dim == 128
            and has_pinned_fragment):
        return False
    positions = sum(q_lens) * query_heads
    if kernel.launch.block != (256, 1, 1):
        raise ValueError(
            "MetaX FA2 D128 combine requires the source-defined 256 threads")
    if kernel.launch.grid != (math.ceil(positions / 8), 1, 1):
        raise ValueError(
            "MetaX FA2 D128 combine grid disagrees with source kBlockM=8")
    return True


def _metax_fa2_combine_max_splits(kernel: KernelSpec) -> int:
    """Return MACA's compile-time SplitKV reducer width."""

    if not _is_source_bound_metax_fa2_combine(kernel):
        raise ValueError("MetaX FA2 combine requires its pinned source")
    return 2 if _splits(kernel) == 2 else 8


def _is_source_bound_grouped_mla_combine(kernel: KernelSpec) -> bool:
    """Whether the launch is vLLM 0.11's pinned Triton MLA reducer."""

    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm != _A800_GROUPED_MLA_STAGE2_ALGORITHM):
        return False
    batch, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    if not (
            kernel.dtype.lower() == "bf16"
            and kernel.variant.lower() in ("combine", "split_combine")
            and len(q_lens) == batch
            and max(q_lens) == q_len
            # The pinned Triton stage2 source is launch-generic in the value
            # head dimension and head count (grid = (batch, head_num), with
            # the batch axis carrying one query row per token, so a verify
            # window contributes q_len rows; BLOCK_DV = next_power_of_2(Lv));
            # audited deployments cover Lv 512 (GLM-5.2) and Lv 128
            # (Kimi-K2.6).
            and head_dim > 0
            and query_heads > 0
            and kv_heads == 1
            and kernel.attributes["kv_layout"] == "mla"
            and _splits(kernel) >= 1
            and kernel.launch.grid == (sum(q_lens), query_heads, 1)
            and kernel.launch.block == (128, 1, 1)):
        raise ValueError(
            "grouped-MLA stage2 marker disagrees with the pinned vLLM "
            "Triton source contract")
    return True


def _fa2_combine_max_splits(kernel: KernelSpec) -> int:
    """Return source template ``kMaxSplits`` for the executed split count."""

    if not _is_source_bound_fa2_combine(kernel):
        raise ValueError("FA2 combine template requires its pinned algorithm")
    splits = _splits(kernel)
    return 1 << (splits - 1).bit_length()


def _fa2_combine_block_m(kernel: KernelSpec) -> int:
    """Return the pinned FA2 combine output rows per CTA."""

    if not _is_source_bound_fa2_combine(kernel):
        raise ValueError("FA2 combine block-M requires its pinned algorithm")
    head_dim = int(kernel.shape["head_dim"])
    if kernel.launch.block_size != 128:
        raise ValueError("FA2 combine requires the source-defined 128 threads")
    if head_dim <= 0:
        raise ValueError("FA2 combine requires a positive head dimension")
    # flash_fwd_launch_template.h selects the smallest of its three explicit
    # instantiations based on divisibility.  Express that branch literally:
    # ``D % 128 == 0 ? 4 : (D % 64 == 0 ? 8 : 16)``.  In particular, 512/D
    # is not equivalent for head dimensions above 128.
    block_m = 4 if head_dim % 128 == 0 else (
        8 if head_dim % 64 == 0 else 16)
    return block_m


def _has_complete_gpu_combine_program(kernel: KernelSpec) -> bool:
    """Whether a pinned GPU reducer closes every launched CTA program."""

    if _is_source_bound_grouped_mla_combine(kernel):
        _, _, _, _, query_heads, _ = _shape(kernel)
        q_lens, _ = _request_lengths(kernel)
        return kernel.launch.grid_size == sum(q_lens) * query_heads
    if _is_source_bound_metax_fa2_combine(kernel):
        _, _, _, _, query_heads, _ = _shape(kernel)
        q_lens, _ = _request_lengths(kernel)
        return kernel.launch.grid_size == math.ceil(
            sum(q_lens) * query_heads / 8)
    if not _is_source_bound_fa2_combine(kernel):
        return False
    _, _, _, _, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    positions = sum(q_lens) * query_heads
    block_m = _fa2_combine_block_m(kernel)
    return kernel.launch.grid_size == math.ceil(positions / block_m)


def _has_uniform_fa2_combine_ctas(kernel: KernelSpec) -> bool:
    """Whether every launched reducer CTA executes the same active rows."""

    if not _has_complete_gpu_combine_program(kernel):
        return False
    if _is_source_bound_grouped_mla_combine(kernel):
        return True
    if _is_source_bound_metax_fa2_combine(kernel):
        _, _, _, _, query_heads, _ = _shape(kernel)
        q_lens, _ = _request_lengths(kernel)
        return sum(q_lens) * query_heads % 8 == 0
    _, _, _, _, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    positions = sum(q_lens) * query_heads
    return positions % _fa2_combine_block_m(kernel) == 0


def _apply_cann_standard_decode_tensor_padding(
        inp: PredictionInput, kernel: KernelSpec) -> PredictionInput:
    """Apply source-defined Cube padding and AIC/AIV worker balance."""

    grouped_rows = _cann_standard_decode_grouped_rows(kernel)
    source_schedule = _cann_source_schedule_parameters(kernel)
    if grouped_rows is None and source_schedule is None:
        return inp
    tensor_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_tensor"
    )
    if len(tensor_demands) != 1:
        raise ValueError(
            "CANN standard decode contract requires one Tensor compute demand")
    if source_schedule is not None:
        compute_demands = []
        for demand in inp.compute_demands:
            if demand.resource_role == "compute_tensor":
                compute_demands.append(dataclasses.replace(
                    demand,
                    shape_efficiency=source_schedule[
                        "cann_source_tensor_shape_efficiency"],
                    schedule_efficiency=source_schedule[
                        "cann_source_tensor_schedule_efficiency"],
                ))
            elif demand.resource_role == "compute_vector":
                compute_demands.append(dataclasses.replace(
                    demand,
                    shape_efficiency=1.0,
                    schedule_efficiency=1.0,
                    service_demands=(
                        _cann_source_vector_service_demands(
                            kernel, source_schedule)),
                ))
            else:
                raise ValueError(
                    "CANN source schedule has an unsupported compute demand: "
                    f"{demand.resource_role!r}")
        return dataclasses.replace(
            inp, compute_demands=tuple(compute_demands))

    # Both MM1 and MM2 pass Axis::AlignedSize() to Mmad, whose default Cube
    # alignment is 16 rows.  A dynamic mBase smaller than the complete GQA
    # group can only create more 16-row MMAD tasks, so G/16 is a conservative
    # source-backed lower bound even without the per-workload tiling payload.
    #
    # The same fixed CANN 8.5.1 FiaBlockVecNonQuant::SetMSplitInfo source maps
    # M <= 16 wholly to the first AIV in each 1:2 mixed AIC/AIV group.  The
    # second AIV still executes the cross-core wait/set protocol, but its
    # service_elements are zero.  The generic mixed-task capacity exposes two
    # AIVs per active AIC, so ordinary Vec1/Vec2 source work uses one half of
    # that capacity.  The FlashDecode-capable variant additionally dispatches
    # FiaBlockVecFlashDecode by dynamic usedVecNumOfFd.  That field is absent
    # from the production launch artifact, so its aggregate Vector lower bound
    # must retain full-AIV capacity rather than applying the ordinary-lane
    # partition to an unresolved terminal reducer.
    flash_decode_capable = kernel.launch.algorithm == (
        "cann_flash_attention_score:103000000010522221")
    compute_demands = []
    for demand in inp.compute_demands:
        if demand.resource_role == "compute_tensor":
            compute_demands.append(dataclasses.replace(
                demand,
                shape_efficiency=(
                    demand.shape_efficiency * grouped_rows / 16.0),
            ))
        elif demand.resource_role == "compute_vector":
            compute_demands.append(
                demand if flash_decode_capable else dataclasses.replace(
                    demand,
                    schedule_efficiency=demand.schedule_efficiency * 0.5,
                ))
        else:
            raise ValueError(
                "CANN standard decode has an unsupported compute demand: "
                f"{demand.resource_role!r}")
    return dataclasses.replace(inp, compute_demands=tuple(compute_demands))


def _attach_cann_source_schedule_phase_ledger(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        source_schedule) -> PredictionInput:
    """Publish the exact retained CANN worker schedule as a phase ledger."""

    if inp.mem_model != "npu_tile":
        raise ValueError("CANN source phase ledger requires npu_tile")
    if inp.phase_ledger is not None:
        raise ValueError("CANN source phase ledger attaches once")

    compute_by_role = {
        demand.resource_role: demand for demand in inp.compute_demands}
    if set(compute_by_role) != {"compute_tensor", "compute_vector"}:
        raise ValueError(
            "CANN source phase ledger requires Tensor and Vector demands")
    tensor_operations = compute_by_role["compute_tensor"].operations
    vector_operations = compute_by_role["compute_vector"].operations
    if not math.isclose(
            tensor_operations,
            source_schedule["cann_source_tensor_logical_operations"],
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN source Tensor operations disagree with structured demand")
    if not math.isclose(
            vector_operations,
            source_schedule["cann_source_vector_operations"],
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN source Vector operations disagree with structured demand")

    paths = {path.name: path for path in inp.memory_paths}
    if set(paths) != {
            "mte2_read", "mte3_write", "fixpipe_l2_write",
            "kfc_workspace_clear"}:
        raise ValueError(
            "CANN source phase ledger requires MTE2, MTE3, Fixpipe and KFC "
            f"paths, got {sorted(paths)}")
    read_nodes = {
        node.resource_role: node for node in paths["mte2_read"].nodes}
    write_nodes = {
        node.resource_role: node for node in paths["mte3_write"].nodes}
    if set(read_nodes) != {"l2_cache", "dram"}:
        raise ValueError("CANN source MTE2 path must contain L2 and DRAM")
    if set(write_nodes) != {"mte3", "l2_cache", "dram"}:
        raise ValueError(
            "CANN source MTE3 path must contain MTE3, L2, and DRAM")
    fixpipe_nodes = {
        node.resource_role: node
        for node in paths["fixpipe_l2_write"].nodes}
    if set(fixpipe_nodes) not in ({"l2_cache"}, {"l2_cache", "dram"}):
        raise ValueError(
            "CANN source Fixpipe path must reach L2 and may continue to DRAM")

    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    mla_dimensions = _cann_mla_dimensions(kernel)
    if mla_dimensions is None:
        qk_head_dim = head_dim
        value_head_dim = head_dim
    else:
        qk_head_dim, value_head_dim = mla_dimensions
    # K and V are read as per-kv_head column slices of a strided row; the
    # slice width is the tensor's own head dim.
    _kv_footprint = _cann_kv_dram_footprint_bytes(
        kv_heads, kv_lens, qk_head_dim, value_head_dim, element_bytes,
        mla_dimensions is not None)
    key_fill_share = _cann_kv_fill_share(
        hardware, kv_heads, qk_head_dim, element_bytes, _kv_footprint)
    value_fill_share = _cann_kv_fill_share(
        hardware, kv_heads, value_head_dim, element_bytes, _kv_footprint)
    from src.kernel.memory_cann_mla import has_cann_mla_source,current_cann_mla_source_work
    if has_cann_mla_source(kernel):
        source=current_cann_mla_source_work(kernel,hardware)
        unique_query_bytes=float(source['query_unique_bytes'])
        unique_key_bytes=float(source['key_interface_bytes'])
        unique_value_bytes=float(source['value_interface_bytes'])
    else:
        unique_query_bytes = float(
            query_heads * sum(q_lens) * qk_head_dim * element_bytes)
        unique_key_bytes = float(
            kv_heads * sum(kv_lens) * qk_head_dim * element_bytes)
        unique_value_bytes = float(
            kv_heads * sum(kv_lens) * value_head_dim * element_bytes)
    if mla_dimensions is not None:
        # MM1 already brought the whole latent in; MM2 reads its leading
        # channels, which is an L2 hit and no further GM footprint.
        unique_value_bytes = 0.0
    query_l2_bytes, key_l2_bytes, value_l2_bytes = (
        _cann_source_task_input_bytes(kernel, source_schedule))

    vec1_read_bytes = source_schedule["cann_source_vec1_read_bytes"]
    mm2_score_read_bytes = source_schedule[
        "cann_source_mm2_score_read_bytes"]
    vec2_mm2_read_bytes = source_schedule[
        "cann_source_vec2_mm2_read_bytes"]
    vec2_state_read_bytes = source_schedule[
        "cann_source_vec2_state_read_bytes"]
    terminal_read_bytes = source_schedule[
        "cann_source_flash_decode_terminal_read_bytes"]
    mm1_unique_write_bytes = source_schedule[
        "cann_source_mm1_unique_write_bytes"]
    mm2_unique_write_bytes = source_schedule[
        "cann_source_mm2_unique_write_bytes"]
    vec1_write_bytes = source_schedule["cann_source_vec1_write_bytes"]
    nupdate_write_bytes = source_schedule[
        "cann_source_nupdate_write_bytes"]
    vec2_write_bytes = source_schedule["cann_source_vec2_write_bytes"]
    terminal_write_bytes = source_schedule[
        "cann_source_flash_decode_terminal_write_bytes"]
    output_init_bytes = source_schedule["cann_source_output_init_bytes"]
    external_write_bytes = source_schedule[
        "cann_source_external_write_bytes"]
    final_output_bytes = external_write_bytes - output_init_bytes
    main_direct_output_bytes = final_output_bytes - terminal_write_bytes
    if main_direct_output_bytes < 0:
        raise ValueError(
            "CANN FlashDecode terminal writes exceed final output")
    terminal_operations = source_schedule[
        "cann_source_vector_terminal_operations"]
    has_flash_decode = terminal_operations > 0
    if has_flash_decode != (terminal_read_bytes > 0):
        raise ValueError(
            "CANN FlashDecode terminal operations and reads disagree")
    if has_flash_decode != (terminal_write_bytes > 0):
        raise ValueError(
            "CANN FlashDecode terminal operations and writes disagree")
    if not math.isclose(
            vec1_write_bytes, mm2_score_read_bytes,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN Vec1 writes do not conserve MM2 score reads")
    if not math.isclose(
            vec1_read_bytes, 2.0 * vec1_write_bytes,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN MM1 FP32 and Vec1 BF16 score widths disagree")

    expected_read_by_role = {
        "l2_cache": (
            query_l2_bytes + key_l2_bytes + value_l2_bytes + vec1_read_bytes
            + mm2_score_read_bytes + vec2_mm2_read_bytes
            + vec2_state_read_bytes + terminal_read_bytes),
        "dram": (
            unique_query_bytes + unique_key_bytes / key_fill_share
            + unique_value_bytes / value_fill_share),
    }
    for role, expected_bytes in expected_read_by_role.items():
        if not math.isclose(
                read_nodes[role].bytes, expected_bytes,
                rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise ValueError(
                "CANN source reads do not conserve "
                f"{role}: structured={read_nodes[role].bytes}, "
                f"source={expected_bytes}")
    aiv_write_bytes = (
        vec1_write_bytes + nupdate_write_bytes
        + vec2_write_bytes + terminal_write_bytes
        + output_init_bytes)
    aiv_unique_write_bytes = source_schedule[
        "cann_source_aiv_unique_write_bytes"]
    vec1_unique_write_bytes = source_schedule[
        "cann_source_vec1_unique_write_bytes"]
    vec2_unique_write_bytes = source_schedule[
        "cann_source_vec2_unique_write_bytes"]
    aic_write_bytes = vec1_read_bytes + vec2_mm2_read_bytes
    expected_write_by_role = {
        "mte3": aiv_write_bytes,
        "l2_cache": aiv_write_bytes,
        # DRAM sees the distinct extent the AIV stores touch, not the bytes
        # MTE3 moves: a rewritten GM line costs one drain however often it is
        # rewritten (gm_write_back_residency_cann_8_5_1.json).
        "dram": aiv_unique_write_bytes,
    }
    for role, expected_bytes in expected_write_by_role.items():
        if not math.isclose(
                write_nodes[role].bytes, expected_bytes,
                rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise ValueError(
                "CANN source writes do not conserve "
                f"{role}: structured={write_nodes[role].bytes}, "
                f"source={expected_bytes}")
    if not math.isclose(
            fixpipe_nodes["l2_cache"].bytes, aic_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(
            "CANN source Fixpipe writes do not conserve L2: "
            f"structured={fixpipe_nodes['l2_cache'].bytes}, "
            f"source={aic_write_bytes}")

    vec1_operations = source_schedule[
        "cann_source_vector_vec1_operations"]
    vec2_operations = source_schedule[
        "cann_source_vector_vec2_operations"]
    initialization_operations = source_schedule[
        "cann_source_vector_initialization_operations"]
    vector_main_operations = source_schedule[
        "cann_source_vector_main_operations"]
    if (initialization_operations <= 0
            or vec1_operations <= 0 or vec2_operations <= 0):
        raise ValueError("CANN source Vector phase operations must be positive")
    if not math.isclose(
            initialization_operations + vec1_operations + vec2_operations,
            vector_main_operations,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN source initialization/Vec1/Vec2 work does not conserve "
            "main Vector work")

    def path_demands(
            path_name: str, first_role: str, first_bytes: float,
            downstream_role: str, downstream_bytes: float):
        if first_bytes < 0 or downstream_bytes < 0:
            raise ValueError("CANN source phase bytes cannot be negative")
        if downstream_bytes > first_bytes:
            raise ValueError(
                "CANN source downstream bytes must be within ingress")
        demands = []
        if first_bytes > 0:
            demands.append(PhaseMemoryDemand(
                path_name=path_name,
                resource_role=first_role,
                bytes=first_bytes,
            ))
        if downstream_bytes > 0:
            demands.append(PhaseMemoryDemand(
                path_name=path_name,
                resource_role=downstream_role,
                bytes=downstream_bytes,
            ))
        return tuple(demands)

    def read_demands(l2_bytes: float, dram_bytes: float):
        return path_demands(
            "mte2_read", "l2_cache", l2_bytes, "dram", dram_bytes)

    def write_demands(mte3_bytes: float, dram_bytes: float):
        if dram_bytes < 0 or dram_bytes > mte3_bytes:
            raise ValueError(
                "CANN source DRAM writes must be within MTE3 writes")
        return (
            path_demands(
                "mte3_write", "mte3", mte3_bytes,
                "l2_cache", mte3_bytes)
            + (() if dram_bytes == 0 else (PhaseMemoryDemand(
                path_name="mte3_write",
                resource_role="dram",
                bytes=dram_bytes,
            ),))
        )

    def fixpipe_write_demands(l2_bytes: float, dram_bytes: float):
        if l2_bytes <= 0:
            raise ValueError("CANN source Fixpipe phase bytes must be positive")
        if dram_bytes < 0 or dram_bytes > l2_bytes:
            raise ValueError(
                "CANN source Fixpipe DRAM continuation must fit its L2 path")
        return (PhaseMemoryDemand(
            path_name="fixpipe_l2_write",
            resource_role="l2_cache",
            bytes=l2_bytes,
        ),) + (() if dram_bytes == 0 else (PhaseMemoryDemand(
            path_name="fixpipe_l2_write",
            resource_role="dram",
            bytes=dram_bytes,
        ),))

    phases = [AnalyticalPhase(
        name="kfc_workspace_clear",
        predecessors=(),
        compute_demands=(),
        # The MIX entry clears each core's KFC message region before any
        # operator phase runs.
        memory_demands=(
            PhaseMemoryDemand(
                path_name="kfc_workspace_clear",
                resource_role="l2_cache",
                bytes=_cann_kfc_workspace_clear_bytes(kernel),
            ),
            PhaseMemoryDemand(
                path_name="kfc_workspace_clear",
                resource_role="dram",
                bytes=_cann_kfc_workspace_clear_bytes(kernel),
            ),
        ),
        completion_barrier=True,
    ), AnalyticalPhase(
        name="aiv_initialization",
        predecessors=("kfc_workspace_clear",),
        compute_demands=(PhaseComputeDemand(
            resource_role="compute_vector",
            operation="aiv_source_buffer_initialization",
            operations=initialization_operations,
        ),),
        memory_demands=(),
        completion_barrier=True,
    )]
    vec1_predecessors = ("mm1", "aiv_initialization")
    if output_init_bytes > 0:
        phases.append(AnalyticalPhase(
            name="output_init",
            predecessors=("aiv_initialization",),
            compute_demands=(),
            # Real rows are overwritten by the final output. An empty-KV
            # padding row is initialized only, so its distinct bytes belong
            # to this phase instead of the score/value output phase.
            memory_demands=write_demands(output_init_bytes,
                max(0.0, output_init_bytes - final_output_bytes)),
            completion_barrier=True,
        ))
        vec1_predecessors = ("mm1", "output_init")
    phases.extend((
        AnalyticalPhase(
            name="mm1",
            predecessors=("kfc_workspace_clear",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_tensor",
                operation="qk_matmul",
                operations=0.5 * tensor_operations,
            ),),
            memory_demands=(
                read_demands(
                    query_l2_bytes + key_l2_bytes,
                    unique_query_bytes
                    + unique_key_bytes / key_fill_share)
                + fixpipe_write_demands(
                    vec1_read_bytes, mm1_unique_write_bytes)
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="vec1",
            predecessors=vec1_predecessors,
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_vector",
                operation="padded_online_softmax",
                operations=vec1_operations,
            ),),
            memory_demands=(
                read_demands(vec1_read_bytes, 0.0)
                + write_demands(
                    vec1_write_bytes + nupdate_write_bytes,
                    vec1_unique_write_bytes)
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="mm2",
            predecessors=("vec1",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_tensor",
                operation="pv_matmul",
                operations=0.5 * tensor_operations,
            ),),
            memory_demands=(
                read_demands(
                    value_l2_bytes + mm2_score_read_bytes,
                    unique_value_bytes / value_fill_share)
                + fixpipe_write_demands(
                    vec2_mm2_read_bytes, mm2_unique_write_bytes)
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="vec2",
            predecessors=("mm2",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_vector",
                operation="online_state_merge_and_output",
                operations=vec2_operations,
            ),),
            memory_demands=(
                read_demands(
                    vec2_mm2_read_bytes + vec2_state_read_bytes, 0.0)
                + write_demands(
                    vec2_write_bytes,
                    vec2_unique_write_bytes + main_direct_output_bytes)
            ),
            completion_barrier=True,
        ),
    ))
    if has_flash_decode:
        phases.append(AnalyticalPhase(
            name="flash_decode_terminal",
            predecessors=("vec2",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_vector",
                operation="flash_decode_partial_reduce",
                operations=terminal_operations,
            ),),
            memory_demands=(
                read_demands(terminal_read_bytes, 0.0)
                + write_demands(
                    terminal_write_bytes, terminal_write_bytes)
            ),
            completion_barrier=True,
        ))
    ledger = AnalyticalPhaseLedger(phases=tuple(phases))
    return dataclasses.replace(inp, phase_ledger=ledger)


def _attach_cann_standard_decode_phase_ledger(
        inp: PredictionInput, kernel: KernelSpec,
        hardware) -> PredictionInput:
    """Publish the source-defined CANN GQA dependency chain without timing it.

    CANN 8.5.1 ``FiaKernelNonQuant`` issues one logical task as
    MM1 -> Vec1 -> MM2 -> Vec2.  MM1 and MM2 share the AIC domain and must add
    within a task; Vec1 and Vec2 likewise share the AIV domain.  The source
    also exposes a three-task software pipeline.  This ledger publishes only
    the priced MTE2 consumers and AIV MTE3 producers forced by the four
    phases.  Source-issued AIC Fixpipe writes are conserved at their proven L2
    sink, but Fixpipe service, dynamic task partition, padding, and buffer
    lifetimes remain unavailable, so the ledger does not create a
    ``PipelineWorkPlan``.
    """

    source_schedule = _cann_source_schedule_parameters(kernel)
    if source_schedule is not None:
        return _attach_cann_source_schedule_phase_ledger(
            inp, kernel, hardware, source_schedule)
    if _cann_standard_decode_grouped_rows(kernel) is None:
        return inp
    if kernel.launch.algorithm == (
            "cann_flash_attention_score:103000000010522221"):
        # This source variant can activate FlashDecode.  Its split-core path
        # writes FP32 partials and appends a FiaBlockVecFlashDecode terminal
        # reducer instead of completing at the ordinary Vec2 BF16 writer.
        # The dynamic payload that distinguishes the active reducer is not in
        # the production launch contract, so publishing the four-phase ledger
        # for every invocation would be a false DAG.  Aggregate lower-bound
        # demand remains valid and the explicit FlashDecode limitations carry
        # the unresolved terminal stage.
        return inp
    if inp.mem_model != "npu_tile":
        raise ValueError("CANN standard decode phase ledger requires npu_tile")
    if inp.phase_ledger is not None:
        raise ValueError("CANN standard decode phase ledger attaches once")

    compute_by_role = {
        demand.resource_role: demand for demand in inp.compute_demands}
    if set(compute_by_role) != {"compute_tensor", "compute_vector"}:
        raise ValueError(
            "CANN standard decode phase ledger requires Tensor and Vector "
            f"demands, got {sorted(compute_by_role)}")
    tensor_operations = compute_by_role["compute_tensor"].operations
    vector_operations = compute_by_role["compute_vector"].operations
    vector_lower_bound = _cann_standard_decode_vector_lower_bound(kernel)
    if vector_lower_bound is None:
        raise AssertionError(
            "CANN standard decode lost its Vector lowering contract")
    if not math.isclose(
            vector_operations,
            vector_lower_bound.vec1_operations
            + vector_lower_bound.vec2_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(
            "CANN standard decode Vector phases do not conserve the "
            "source-lowered aggregate demand")

    paths = {path.name: path for path in inp.memory_paths}
    if set(paths) != {
            "mte2_read", "mte3_write", "fixpipe_l2_write",
            "kfc_workspace_clear"}:
        raise ValueError(
            "CANN standard decode phase ledger requires MTE2, MTE3, Fixpipe "
            f"and KFC paths, got {sorted(paths)}")
    read_nodes = {
        node.resource_role: node for node in paths["mte2_read"].nodes}
    write_nodes = {
        node.resource_role: node for node in paths["mte3_write"].nodes}
    if set(read_nodes) != {"l2_cache", "dram"}:
        raise ValueError(
            "CANN standard decode MTE2 path must contain L2 and DRAM")
    if set(write_nodes) != {"mte3", "l2_cache", "dram"}:
        raise ValueError(
            "CANN standard decode MTE3 path must contain MTE3, L2, and DRAM")
    fixpipe_nodes = {
        node.resource_role: node
        for node in paths["fixpipe_l2_write"].nodes}
    if set(fixpipe_nodes) != {"l2_cache"}:
        raise ValueError(
            "CANN standard decode Fixpipe path must terminate at L2")

    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    kv_fill_share = _cann_kv_fill_share(
        hardware, kv_heads, head_dim, element_bytes,
        float(2 * kv_heads * sum(kv_lens) * head_dim * element_bytes))
    query_bytes = query_heads * sum(q_lens) * head_dim * element_bytes
    key_bytes = kv_heads * sum(kv_lens) * head_dim * element_bytes
    value_bytes = key_bytes
    workspace = _cann_workspace_lower_bound(kernel)
    if workspace is None:
        raise AssertionError("CANN standard decode lost its workspace contract")
    expected_l2_read_bytes = float(
        query_bytes + key_bytes + value_bytes + workspace.read_bytes)
    expected_dram_read_bytes = float(
        query_bytes + (key_bytes + value_bytes) / kv_fill_share)
    expected_read_by_role = {
        "l2_cache": expected_l2_read_bytes,
        "dram": expected_dram_read_bytes,
    }
    for role, expected_bytes in expected_read_by_role.items():
        node = read_nodes[role]
        if not math.isclose(
                node.bytes, expected_bytes,
                rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise ValueError(
                "CANN standard decode input phase bytes do not conserve "
                f"{role}: structured={node.bytes}, source={expected_bytes}")
    final_output_bytes = float(
        query_heads * sum(q_lens) * head_dim * element_bytes)
    expected_write_by_role = {
        "mte3": final_output_bytes + workspace.aiv_write_bytes,
        "l2_cache": final_output_bytes + workspace.aiv_write_bytes,
        "dram": final_output_bytes,
    }
    for role, expected_bytes in expected_write_by_role.items():
        node = write_nodes[role]
        if not math.isclose(
                node.bytes, expected_bytes,
                rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise ValueError(
                "CANN standard decode AIV write bytes do not conserve "
                f"{role}: structured={node.bytes}, source={expected_bytes}")
    if not math.isclose(
            fixpipe_nodes["l2_cache"].bytes,
            workspace.aic_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise ValueError(
            "CANN standard decode AIC write bytes do not conserve L2: "
            f"structured={fixpipe_nodes['l2_cache'].bytes}, "
            f"source={workspace.aic_write_bytes}")

    def path_demands(
            path_name: str, first_role: str, first_bytes: float,
            downstream_role: str, downstream_bytes: float):
        if downstream_bytes < 0 or downstream_bytes > first_bytes:
            raise ValueError(
                "CANN FIA downstream phase bytes must be within ingress")
        demands = [PhaseMemoryDemand(
            path_name=path_name,
            resource_role=first_role,
            bytes=first_bytes,
        )]
        if downstream_bytes > 0:
            demands.append(PhaseMemoryDemand(
                path_name=path_name,
                resource_role=downstream_role,
                bytes=downstream_bytes,
            ))
        return tuple(demands)

    def read_demands(l2_bytes: float, dram_bytes: float):
        return path_demands(
            "mte2_read", "l2_cache", l2_bytes, "dram", dram_bytes)

    def mte3_write_demands(mte3_bytes: float, dram_bytes: float):
        if dram_bytes < 0 or dram_bytes > mte3_bytes:
            raise ValueError(
                "CANN FIA DRAM writes must be within MTE3 writes")
        return (
            path_demands(
                "mte3_write", "mte3", mte3_bytes,
                "l2_cache", mte3_bytes)
            + (() if dram_bytes == 0 else (PhaseMemoryDemand(
                path_name="mte3_write",
                resource_role="dram",
                bytes=dram_bytes,
            ),))
        )

    def fixpipe_write_demands(l2_bytes: float):
        if l2_bytes <= 0:
            raise ValueError(
                "CANN standard decode Fixpipe bytes must be positive")
        return (PhaseMemoryDemand(
            path_name="fixpipe_l2_write",
            resource_role="l2_cache",
            bytes=l2_bytes,
        ),)

    score_fp32_bytes = 4.0 * workspace.score_elements
    score_bf16_bytes = 2.0 * workspace.score_elements
    output_fp32_bytes = 4.0 * workspace.output_elements

    kfc_clear_bytes = _cann_kfc_workspace_clear_bytes(kernel)
    ledger = AnalyticalPhaseLedger(phases=(
        AnalyticalPhase(
            name="kfc_workspace_clear",
            predecessors=(),
            compute_demands=(),
            # The MIX entry clears each core's KFC message region before any
            # operator phase runs.
            memory_demands=(
                PhaseMemoryDemand(
                    path_name="kfc_workspace_clear",
                    resource_role="l2_cache",
                    bytes=kfc_clear_bytes,
                ),
                PhaseMemoryDemand(
                    path_name="kfc_workspace_clear",
                    resource_role="dram",
                    bytes=kfc_clear_bytes,
                ),
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="mm1",
            predecessors=("kfc_workspace_clear",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_tensor",
                operation="qk_matmul",
                operations=0.5 * tensor_operations,
            ),),
            memory_demands=(
                read_demands(
                    float(query_bytes + key_bytes),
                    float(query_bytes + key_bytes / kv_fill_share),
                )
                + fixpipe_write_demands(score_fp32_bytes)
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="vec1",
            predecessors=("mm1",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_vector",
                operation="online_softmax_and_source_lowering",
                operations=vector_lower_bound.vec1_operations,
            ),),
            memory_demands=(
                read_demands(score_fp32_bytes, 0.0)
                + mte3_write_demands(score_bf16_bytes, 0.0)
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="mm2",
            predecessors=("vec1",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_tensor",
                operation="pv_matmul",
                operations=0.5 * tensor_operations,
            ),),
            memory_demands=(
                read_demands(
                    float(value_bytes) + score_bf16_bytes,
                    float(value_bytes / kv_fill_share),
                )
                + fixpipe_write_demands(output_fp32_bytes)
            ),
            completion_barrier=True,
        ),
        AnalyticalPhase(
            name="vec2",
            predecessors=("mm2",),
            compute_demands=(PhaseComputeDemand(
                resource_role="compute_vector",
                operation="output_normalize_and_cast",
                operations=vector_lower_bound.vec2_operations,
            ),),
            memory_demands=(
                read_demands(output_fp32_bytes, 0.0)
                + mte3_write_demands(
                    final_output_bytes, final_output_bytes)
            ),
            completion_barrier=True,
        ),
    ))
    return dataclasses.replace(inp, phase_ledger=ledger)


def _attention_tile(kernel: KernelSpec):
    tiling = kernel.launch.tiling
    if tiling is None or not {"q", "kv"}.issubset(tiling.tile_shape):
        return None
    return int(tiling.tile_shape["q"]), int(tiling.tile_shape["kv"])


def _cann_split_fuse_tile(kernel: KernelSpec):
    """Return the fixed CANN SplitFuse marker without creating a GPU tile."""

    tiling = kernel.launch.tiling
    exact_algorithm = kernel.launch.algorithm in _CANN_SPLIT_FUSE_ALGORITHMS
    if tiling is None:
        if exact_algorithm:
            raise ValueError(
                "CANN SplitFuse runtime variant requires an exact tile marker")
        return None
    required = {"q", "kv_stack_ceiling"}
    axes = set(tiling.tile_shape)
    if not exact_algorithm and "kv_stack_ceiling" not in axes:
        return None
    present = required.intersection(axes)
    if axes != required:
        raise ValueError(
            "CANN SplitFuse tile axes are invalid: "
            f"missing={sorted(required - present)}, "
            f"unexpected={sorted(axes - required)}")
    if not exact_algorithm:
        raise ValueError(
            "CANN SplitFuse tile requires an exact SplitFuse runtime variant")
    full_decode = ("source_program" in kernel.attributes
        and kernel.attributes["source_program"] == "ascend_0_18_full_fia_split_fuse")
    if kernel.variant.lower() != "prefill" and not full_decode:
        raise ValueError("CANN SplitFuse tile requires prefill")
    if kernel.dtype.lower() != "bf16":
        raise ValueError("CANN SplitFuse tile requires BF16")
    if kernel.attributes["causal"] is not True:
        raise ValueError("CANN SplitFuse tile requires a causal mask")
    tile = tuple(
        int(tiling.tile_shape[name])
        for name in ("q", "kv_stack_ceiling")
    )
    if tile != (128, 512):
        raise ValueError(
            "unsupported CANN SplitFuse Q/KV-stack geometry: "
            f"{tile}")
    if _splits(kernel) != 1:
        raise ValueError("CANN SplitFuse launch requires one logical split")
    return tile


def _cann_split_fuse_stack_statistics(
        packed_rows: int, attended_kv: int, query_rows: int,
        kv_stack_ceiling: int) -> Tuple[int, Tuple[int, int]]:
    regions,exp_invocations = cann_split_fuse_stack_geometry(
        packed_rows,attended_kv,query_rows,kv_stack_ceiling)
    return sum(math.prod(shape)*repeat for shape,repeat in regions),exp_invocations



def _cann_split_fuse_work(kernel: KernelSpec):
    """Derive SplitFuse work from query-tile classes, without task expansion."""

    tile = _cann_split_fuse_tile(kernel)
    if tile is None:
        return None
    tile_q, kv_stack_ceiling = tile
    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    if query_heads % kv_heads:
        raise ValueError("CANN SplitFuse requires integral GQA head groups")
    worker_count = kernel.launch.grid_size
    if kernel.launch.grid != (worker_count, 1, 1):
        raise ValueError(
            "CANN SplitFuse AIC assignment requires a one-dimensional grid")
    request_lengths = _request_lengths(kernel)
    heads_per_kv = query_heads // kv_heads
    task_count = 0
    empty_task_count = 0
    empty_output_elements = 0
    empty_dup_operations = 0
    attention_elements = 0
    final_output_elements = 0
    workspace_output_elements = 0
    rescale_output_elements = 0
    mask_mte2_bytes = 0
    mmad_attention_elements = 0
    kv_replay_positions = 0
    aic_mmad_attention_elements = [0] * worker_count
    ring_depth = _CANN_SPLIT_FUSE_PRE_LAUNCH + 1
    slot_rows_by_worker = [[0] * ring_depth for _ in range(worker_count)]
    aiv_worker_count = 2 * worker_count
    aiv_generic_operations = [0] * aiv_worker_count
    aiv_exp_operations = [0] * aiv_worker_count
    aiv_exp_invocations = [0] * aiv_worker_count
    aiv_div_operations = [0] * aiv_worker_count
    aiv_div_invocations = [0] * aiv_worker_count
    embed_round = ((head_dim + 15) // 16) * 16
    vec2_row_tile = (8192 // embed_round // 8) * 8
    if vec2_row_tile <= 0:
        raise ValueError(
            "CANN SplitFuse Vec2 row tile is empty for the head dimension")
    div_calls_per_loop = math.ceil(head_dim / 64)
    task_geometry_cache = {}
    worker_run_counts_cache = {}

    def add_cyclic_run(
            target, repeat_count, worker_task_counts, per_task):
        """Accumulate one source-contiguous task run on its AIC ring."""

        if repeat_count <= 0:
            raise ValueError("CANN SplitFuse task-run count must be positive")
        if per_task == 0:
            return
        for worker, task_runs in enumerate(worker_task_counts):
            if task_runs == 0:
                continue
            target[worker] += task_runs * per_task

    def add_task_run(query_rows, attended_kv, bundled_heads, repeat_count):
        """Account for repeated identical head-bundle tasks without CTAs."""

        nonlocal task_count
        nonlocal rescale_output_elements
        nonlocal empty_task_count, empty_output_elements, empty_dup_operations
        if attended_kv == 0:
            # Only the independent FULL cached tail reaches this branch.
            # Source AIC still loads Q, while AIV duplicates a shared zero
            # column per participating lane and writes it to its heads.
            for _ in range(repeat_count):
                worker = (task_count + empty_task_count) % worker_count
                for lane in range(2):
                    lane_rows = (query_rows // 2 if lane == 0 else query_rows - query_rows // 2) if bundled_heads == 1 else query_rows
                    lane_heads = 1 if bundled_heads == 1 else (bundled_heads // 2 if lane == 0 else bundled_heads - bundled_heads // 2)
                    if lane_rows and lane_heads:
                        operations = lane_rows * embed_round
                        aiv_generic_operations[2 * worker + lane] += operations
                        empty_dup_operations += operations
                empty_output_elements += query_rows * bundled_heads * head_dim
                empty_task_count += 1
            return (0, 0, 0, 0, 0, 0)
        geometry_key = (query_rows, attended_kv, bundled_heads)
        if geometry_key not in task_geometry_cache:
            packed_rows = query_rows * bundled_heads
            task_stack_count = math.ceil(attended_kv / kv_stack_ceiling)
            task_attention_elements = packed_rows * attended_kv
            task_final_output_elements = packed_rows * head_dim
            task_workspace_output_elements = (
                packed_rows * head_dim * task_stack_count)
            task_mmad_rows = math.ceil(packed_rows / 16) * 16
            task_mmad_attention_elements = task_mmad_rows * attended_kv
            task_mask_bytes, exp_invocations_per_subblock = (
                _cann_split_fuse_stack_statistics(
                    packed_rows,
                    attended_kv,
                    query_rows,
                    kv_stack_ceiling,
                ))
            task_geometry_cache[geometry_key] = (
                packed_rows,
                task_attention_elements,
                task_final_output_elements,
                task_workspace_output_elements,
                task_mmad_attention_elements,
                task_mask_bytes,
                exp_invocations_per_subblock,
                task_stack_count,
            )
        (packed_rows,
         task_attention_elements,
         task_final_output_elements,
         task_workspace_output_elements,
         task_mmad_attention_elements,
         task_mask_bytes,
         exp_invocations_per_subblock,
         task_stack_count) = task_geometry_cache[geometry_key]
        # block_epilogue_rescale_o.hpp keeps the first PV result and, for
        # every subsequent KV stack, executes go *= dm followed by go += lo.
        # These output-wise operations are distinct from score softmax and
        # the terminal Div/Cast, even when the accumulator stays in UB.
        rescale_output_elements += (
            repeat_count * task_final_output_elements * (task_stack_count - 1))
        start_task = task_count % worker_count
        worker_run_key = (start_task, repeat_count)
        if worker_run_key not in worker_run_counts_cache:
            worker_run_counts_cache[worker_run_key] = tuple(
                0 if (first := (worker - start_task) % worker_count)
                >= repeat_count else (
                    1 + (repeat_count - 1 - first) // worker_count)
                for worker in range(worker_count))
        worker_task_counts = worker_run_counts_cache[worker_run_key]
        add_cyclic_run(
            aic_mmad_attention_elements,
            repeat_count,
            worker_task_counts,
            task_mmad_attention_elements,
        )
        task_slots = min(ring_depth, task_stack_count)
        for worker, worker_task_count in enumerate(worker_task_counts):
            if not worker_task_count:
                continue
            # The ring index restarts with every task, so this task widens
            # only the slots it actually reaches.
            worker_slots = slot_rows_by_worker[worker]
            for slot in range(task_slots):
                if packed_rows > worker_slots[slot]:
                    worker_slots[slot] = packed_rows
        for subblock in range(2):
            subblock_rows = (
                packed_rows // 2 if subblock == 0
                else packed_rows - packed_rows // 2)
            if subblock_rows == 0:
                continue
            aiv_worker_offset = subblock
            subblock_attention_elements = subblock_rows * attended_kv
            subblock_output_elements = subblock_rows * head_dim
            for worker, worker_task_count in enumerate(worker_task_counts):
                if worker_task_count == 0:
                    continue
                aiv_worker = 2 * worker + aiv_worker_offset
                aiv_generic_operations[aiv_worker] += worker_task_count * (
                    4 * subblock_attention_elements
                    + subblock_output_elements
                    + 2 * subblock_output_elements * (task_stack_count - 1))
                aiv_exp_operations[aiv_worker] += (
                    worker_task_count * subblock_attention_elements)
                aiv_exp_invocations[aiv_worker] += (
                    worker_task_count
                    * exp_invocations_per_subblock[subblock])
                aiv_div_operations[aiv_worker] += (
                    worker_task_count * subblock_output_elements)
                aiv_div_invocations[aiv_worker] += (
                    worker_task_count
                    * math.ceil(subblock_rows / vec2_row_tile)
                    * div_calls_per_loop)
        task_count += repeat_count
        return (
            repeat_count * task_attention_elements,
            repeat_count * task_final_output_elements,
            repeat_count * task_workspace_output_elements,
            repeat_count * task_mask_bytes,
            repeat_count * task_mmad_attention_elements,
            repeat_count * attended_kv,
        )

    def accumulate_task_run(contributions):
        """Merge one task-run contribution into the request-wide census."""

        nonlocal attention_elements
        nonlocal final_output_elements
        nonlocal workspace_output_elements
        nonlocal mask_mte2_bytes
        nonlocal mmad_attention_elements
        nonlocal kv_replay_positions
        (attention,
         final_output,
         workspace_output,
         mask_bytes,
         mmad_attention,
         replay_positions) = contributions
        attention_elements += attention
        final_output_elements += final_output
        workspace_output_elements += workspace_output
        mask_mte2_bytes += mask_bytes
        mmad_attention_elements += mmad_attention
        kv_replay_positions += replay_positions

    for query, key_value in zip(*request_lengths):
        if query == 0:
            continue
        if query > key_value and key_value != 0:
            raise ValueError(
                "CANN SplitFuse bottom-right workload requires q_len <= kv_len")
        # kernel_common.hpp::GetQNBlockTile packs an even number of query
        # heads while the complete bundle fits the fixed 128-row Q tile.
        even_heads = 2 * ((tile_q // query) // 2)
        query_heads_per_task = max(1, min(heads_per_kv, even_heads))
        full_head_tasks, tail_heads = divmod(
            heads_per_kv, query_heads_per_task)
        for start in range(0, query, tile_q):
            query_rows = min(tile_q, query - start)
            attended_kv = min(
                key_value,
                start + tile_q + max(key_value - query, 0),
            )
            if query_rows <= 0 or attended_kv < 0:
                raise ValueError(
                    "CANN SplitFuse task must have positive Q and KV work")
            # With no tail bundle, every KV head emits the same contiguous
            # full-bundle run.  Its source ordering is exactly one longer
            # cyclic run, so aggregate it before deriving the per-worker
            # histogram rather than replaying every KV-head iteration.
            if full_head_tasks and not tail_heads:
                contributions = add_task_run(
                    query_rows,
                    attended_kv,
                    query_heads_per_task,
                    full_head_tasks * kv_heads,
                )
                accumulate_task_run(contributions)
                continue
            for _ in range(kv_heads):
                if full_head_tasks:
                    contributions = add_task_run(
                        query_rows,
                        attended_kv,
                        query_heads_per_task,
                        full_head_tasks,
                    )
                    accumulate_task_run(contributions)
                if tail_heads:
                    contributions = add_task_run(
                        query_rows,
                        attended_kv,
                        tail_heads,
                        1,
                    )
                    accumulate_task_run(contributions)
    if has_cann_tnd_source(kernel):
        mask_mte2_bytes = current_cann_tnd_source_work(kernel,None)['priced_mask_read_bytes']
    if task_count <= 0:
        raise ValueError("CANN SplitFuse workload produced no tasks")
    if sum(aic_mmad_attention_elements) != mmad_attention_elements:
        raise AssertionError("CANN SplitFuse AIC work is not conserved")
    if sum(aiv_generic_operations) != (
            4 * attention_elements + final_output_elements + empty_dup_operations
            + 2 * rescale_output_elements):
        raise AssertionError(
            "CANN SplitFuse generic AIV work is not conserved")
    if sum(aiv_exp_operations) != attention_elements:
        raise AssertionError("CANN SplitFuse Exp work is not conserved")
    if sum(aiv_div_operations) != final_output_elements:
        raise AssertionError("CANN SplitFuse Div work is not conserved")
    result = _CannSplitFuseWork(
        task_count=task_count,
        empty_task_count=empty_task_count,
        empty_output_elements=empty_output_elements,
        empty_dup_operations=empty_dup_operations,
        attention_elements=attention_elements,
        final_output_elements=final_output_elements,
        workspace_output_elements=workspace_output_elements,
        rescale_output_elements=rescale_output_elements,
        mask_mte2_bytes=mask_mte2_bytes,
        mmad_attention_elements=mmad_attention_elements,
        kv_replay_positions=kv_replay_positions,
        slot_rows_by_worker=tuple(
            tuple(worker_slots) for worker_slots in slot_rows_by_worker),
        aic_mmad_attention_elements=tuple(aic_mmad_attention_elements),
        aiv_generic_operations=tuple(aiv_generic_operations),
        aiv_exp_operations=tuple(aiv_exp_operations),
        aiv_exp_invocations=tuple(aiv_exp_invocations),
        aiv_div_operations=tuple(aiv_div_operations),
        aiv_div_invocations=tuple(aiv_div_invocations),
    )
    return result


def _cann_split_fuse_vector_service_demands(work: _CannSplitFuseWork):
    """Return the source-issued typed AIV lower bound by physical worker."""

    worker_count = len(work.aiv_generic_operations)
    if worker_count <= 0:
        raise ValueError("CANN SplitFuse requires physical AIV workers")
    typed_work = (
        (
            "generic",
            work.aiv_generic_operations,
            (0,) * worker_count,
        ),
        (
            "exp",
            work.aiv_exp_operations,
            work.aiv_exp_invocations,
        ),
        (
            "div",
            work.aiv_div_operations,
            work.aiv_div_invocations,
        ),
    )
    demands = tuple(
        ComputeServiceDemand(
            service=service,
            operations=float(sum(operations_by_worker)),
            serial_invocations=max(invocations_by_worker),
            operations_by_worker=operations_by_worker,
            serial_invocations_by_worker=invocations_by_worker,
        )
        for service, operations_by_worker, invocations_by_worker in typed_work
    )
    expected_operations = float(
        5 * work.attention_elements + 2 * work.final_output_elements
        + work.empty_dup_operations + 2 * work.rescale_output_elements)
    if not math.isclose(
            math.fsum(demand.operations for demand in demands),
            expected_operations, rel_tol=0.0, abs_tol=0.0):
        raise AssertionError(
            "CANN SplitFuse typed AIV work is not conserved")
    return demands


def _attach_cann_split_fuse_vector_schedule(
        inp: PredictionInput, kernel: KernelSpec, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> PredictionInput:
    """Publish source-issued SplitFuse AIV work without inferring overlap."""

    work = (
        _cann_split_fuse_work(kernel)
        if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET
        else split_fuse_work)
    if work is None:
        return inp
    if inp.mem_model != "npu_tile":
        raise ValueError("CANN SplitFuse AIV schedule requires npu_tile")
    vector_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_vector"
    )
    if len(vector_demands) != 1:
        raise ValueError(
            "CANN SplitFuse requires exactly one Vector compute demand")
    vector = vector_demands[0]
    expected_operations = float(
        5 * work.attention_elements + 2 * work.final_output_elements
        + work.empty_dup_operations + 2 * work.rescale_output_elements)
    if not math.isclose(
            vector.operations, expected_operations,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "CANN SplitFuse typed Vector schedule does not conserve the "
            "aggregate demand")
    if vector.service_demands:
        raise ValueError(
            "CANN SplitFuse typed Vector schedule cannot replace existing "
            "service demands")
    service_demands = _cann_split_fuse_vector_service_demands(work)
    compute_demands = tuple(
        dataclasses.replace(demand, service_demands=service_demands)
        if demand is vector else demand
        for demand in inp.compute_demands
    )
    return dataclasses.replace(inp, compute_demands=compute_demands)


def _cann_fixpipe_l2_write_bytes(
        kernel: KernelSpec, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> Optional[float]:
    """Return source-issued FIA AIC writes that must reach the shared L2."""

    workspace = _cann_workspace_lower_bound(
        kernel, split_fuse_work=split_fuse_work)
    if workspace is None:
        return None
    source_schedule = _cann_source_schedule_parameters(kernel)
    if source_schedule is None:
        return workspace.aic_write_bytes
    write_bytes = (
        source_schedule["cann_source_vec1_read_bytes"]
        + source_schedule["cann_source_vec2_mm2_read_bytes"])
    if write_bytes < workspace.aic_write_bytes:
        raise ValueError(
            "CANN exact Fixpipe writes cannot be below the source lower bound")
    return float(write_bytes)


def _cann_fixpipe_dram_write_bytes(
        kernel: KernelSpec, l2_write_bytes: float, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> float:
    """Return the AIC Fixpipe writes whose distinct extent reaches GM.

    Both FIA paths route their cube results to GM through a per-core ring,
    and the probe that measured GM write-back residency found L0C->GM and
    UB->GM obey one rule, ``write_hit = 1 - slots/passes`` on both
    (gm_write_back_residency_cann_8_5_1.json), so the distinct extent is the
    widest write landing in each slot.

    Fixed decode rings mm1Res and mm2Res, walked as ``loop % PRELOAD_NUM``
    (fia_block_cube_nonquant_gqa.h ComputeMm1) with FixpipeCToGM landing at a
    sub-block offset inside the slot (:896).  SplitFuse prefill rings gS and
    gOTmp, walked as ``stackSeqCount % (PRE_LAUNCH + 1)``.

    Neither path prices the fixed per-AIC-core write that both of them show
    on top of these rings, about 33 KiB on decode and 36 KiB on prefill,
    invariant to every shape parameter.  Its size rules out all four
    ``workspaceParams`` buffers, which scale with shape, and its presence on
    two kernels with different workspace layouts puts it outside either
    layout; no store in the readable CANN 8.5.1 sources accounts for it, and
    naming a writer from the shortfall is the inference this model does not
    make.
    """

    source_schedule = _cann_source_schedule_parameters(kernel)
    if source_schedule is not None:
        dram_bytes = float(
            source_schedule["cann_source_aic_unique_write_bytes"])
    elif (split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET
            or split_fuse_work is None):
        return 0.0
    else:
        _, _, _, head_dim, _, _ = _shape(kernel)
        workspace = _cann_workspace_lower_bound(
            kernel, split_fuse_work=split_fuse_work)
        if workspace is None:
            raise ValueError(
                "CANN SplitFuse Fixpipe DRAM path requires a workspace bound")
        dram_bytes = _cann_split_fuse_aic_unique_write_bytes(
            split_fuse_work, head_dim, workspace.aic_write_bytes)
    if dram_bytes > l2_write_bytes:
        raise ValueError(
            "CANN Fixpipe DRAM continuation cannot exceed its L2 path")
    return float(dram_bytes)



def _attach_cann_fixpipe_l2_write(
        inp: PredictionInput, kernel: KernelSpec, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> PredictionInput:
    """Price source-issued FIA AIC writes at their proven L2 and GM sinks."""

    write_bytes = _cann_fixpipe_l2_write_bytes(
        kernel, split_fuse_work=split_fuse_work)
    if write_bytes is None:
        return inp
    if inp.mem_model != "npu_tile":
        raise ValueError("CANN FIA Fixpipe L2 path requires npu_tile")
    if any(path.name == "fixpipe_l2_write" for path in inp.memory_paths):
        raise ValueError("CANN FIA Fixpipe L2 path must attach once")
    if write_bytes <= 0:
        raise ValueError("CANN FIA Fixpipe L2 path requires positive writes")
    dram_bytes = _cann_fixpipe_dram_write_bytes(
        kernel, write_bytes, split_fuse_work=split_fuse_work)
    nodes = [MemoryNodeDemand(
        resource_role="l2_cache",
        transactions=None,
        bytes=write_bytes,
        transaction_bytes=None,
    )]
    if dram_bytes > 0:
        nodes.append(MemoryNodeDemand(
            resource_role="dram",
            transactions=None,
            bytes=dram_bytes,
            transaction_bytes=None,
        ))
    path = MemoryPathDemand(
        name="fixpipe_l2_write",
        access_kind="write",
        initial_transactions=None,
        nodes=tuple(nodes),
        source=MemorySourceDemand(
            active_units=inp.active_units,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )
    flat = dict(inp.demand_per_resource)
    flat["l2_cache"] += write_bytes
    flat["dram"] += dram_bytes
    return dataclasses.replace(
        inp,
        demand_per_resource=flat,
        memory_paths=inp.memory_paths + (path,),
    )


def _attach_cann_split_fuse_tensor_schedule(
        inp: PredictionInput, kernel: KernelSpec, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> PredictionInput:
    """Price the source-defined critical AIC without creating a pipeline."""

    work = (
        _cann_split_fuse_work(kernel)
        if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET
        else split_fuse_work)
    if work is None:
        return inp
    if inp.mem_model != "npu_tile":
        raise ValueError("CANN SplitFuse AIC schedule requires npu_tile")
    if len(work.aic_mmad_attention_elements) != inp.active_units:
        raise ValueError(
            "CANN SplitFuse AIC schedule disagrees with active units")
    tensor_demands = tuple(
        demand for demand in inp.compute_demands
        if demand.resource_role == "compute_tensor"
    )
    if len(tensor_demands) != 1:
        raise ValueError(
            "CANN SplitFuse requires exactly one Tensor compute demand")
    tensor = tensor_demands[0]
    if not math.isclose(
            tensor.schedule_efficiency, 1.0,
            rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "CANN SplitFuse source schedule cannot compose with another "
            "schedule efficiency")
    total_mmad_elements = sum(work.aic_mmad_attention_elements)
    expected_tensor_operations = (
        4 * int(kernel.shape["head_dim"]) * total_mmad_elements)
    if not math.isclose(
            tensor.operations, expected_tensor_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "CANN SplitFuse AIC schedule does not conserve Tensor work")
    critical_mmad_elements = max(work.aic_mmad_attention_elements)
    schedule_efficiency = (
        total_mmad_elements
        / (len(work.aic_mmad_attention_elements)
           * critical_mmad_elements)
    )
    compute_demands = []
    for demand in inp.compute_demands:
        if demand is tensor:
            compute_demands.append(dataclasses.replace(
                demand, schedule_efficiency=schedule_efficiency))
        else:
            compute_demands.append(demand)
    return dataclasses.replace(
        inp, compute_demands=tuple(compute_demands))


def _attach_cann_split_fuse_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        *, attach_pipeline: bool,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> PredictionInput:
    """Attach the compressed source AIC-to-dual-AIV SplitFuse program."""

    work = (
        _cann_split_fuse_work(kernel)
        if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET
        else split_fuse_work)
    if work is None or not attach_pipeline:
        return inp
    if inp.mem_model != "npu_tile" or inp.launch_task_kind != "mixed":
        raise ValueError("CANN SplitFuse pipeline requires a mixed NPU launch")
    if inp.pipeline_plan is not None:
        raise ValueError("CANN SplitFuse pipeline must attach exactly once")
    if hardware.fia_block_reduce_sum_service is None:
        raise ValueError(
            "CANN SplitFuse pipeline requires the FIA BlockReduceSum "
            "source service")
    if hardware.fia_block_reduce_max_service is None:
        raise ValueError(
            "CANN SplitFuse pipeline requires the FIA BlockReduceMax "
            "source service")
    if hardware.fia_brcb_service is None:
        raise ValueError(
            "CANN SplitFuse pipeline requires the FIA Brcb source service")
    tile_q, kv_stack_ceiling = _cann_split_fuse_tile(kernel)
    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    if query_heads % kv_heads:
        raise ValueError("CANN SplitFuse pipeline requires integral GQA")
    paged_kv_cache = kernel.attributes["kv_layout"] in (
        "paged", "paged_varlen", "chunked_prefill")
    request_runs = []
    for query_length, key_value_length in zip(*_request_lengths(kernel)):
        if key_value_length == 0:
            continue
        if (request_runs
                and request_runs[-1].query_length == query_length
                and request_runs[-1].key_value_length == key_value_length):
            previous = request_runs[-1]
            request_runs[-1] = dataclasses.replace(
                previous, repeat_count=previous.repeat_count + 1)
        else:
            request_runs.append(FiaSplitFuseRequestRunDemand(
                query_length=query_length,
                key_value_length=key_value_length,
                repeat_count=1,
            ))
    plan = FiaSplitFuseWorkPlan(
        mechanism_id="cann_split_fuse_aic_dual_aiv_v1",
        aic_worker_count=kernel.launch.grid_size,
        tile_q=tile_q,
        kv_stack_ceiling=kv_stack_ceiling,
        head_dim=head_dim,
        query_head_count=query_heads,
        kv_head_count=kv_heads,
        request_runs=tuple(request_runs),
        aic_l2_bytes=float(
            2 * (work.final_output_elements + work.empty_output_elements)
            + 4 * work.kv_replay_positions * head_dim
            + 6 * work.attention_elements
            + 4 * work.workspace_output_elements),
        aiv_l2_bytes=float(
            6 * work.attention_elements
            + work.mask_mte2_bytes
            + 4 * work.workspace_output_elements
            + 2 * (work.final_output_elements + work.empty_output_elements)),
        aiv_mte3_bytes=float(
            2 * work.attention_elements
            + 2 * (work.final_output_elements + work.empty_output_elements)),
        paged_kv_cache=paged_kv_cache,
        kv_page_size=128 if paged_kv_cache else None,
        # ``BlockMmadPV<PAGED_CACHE_FLAG>`` issues its V GM-to-L1 ingress
        # before its ``softmaxReady`` wait.  The template puts that ingress
        # ahead of the wait for both values of ``PAGED_CACHE_FLAG``; only the
        # address calculation differs.  The three-slot outer loop therefore
        # exposes the same PRE_LAUNCH=2 V prefix for contiguous, paged,
        # paged-varlen, and chunked-prefill SplitFuse programs.
        pv_v_ingress_before_softmax=True,
        empty_kv_tail_query_length=sum(q for q, k in zip(*_request_lengths(kernel)) if k == 0),
    )
    if plan.task_count != work.task_count:
        raise ValueError(
            "compressed SplitFuse source program changed the task count")
    if plan.empty_kv_tail_task_count != work.empty_task_count:
        raise ValueError("SplitFuse source program changed the empty-KV task count")
    unavailable = "cann_flash_attention_tile_pipeline_unavailable"
    if unavailable not in inp.limitations:
        raise ValueError(
            "CANN SplitFuse source plan expected the aggregate pipeline "
            "limitation")
    limitations = tuple(
        value for value in inp.limitations
        if value not in (
            unavailable,
            "cann_split_fuse_workspace_lifecycle_unavailable",
        ))
    mechanism = hardware.fia_cross_domain_three_handoff
    active_task_counts = tuple(
        task_count for task_count in plan.task_counts if task_count > 0)
    if (mechanism is None
            or len(active_task_counts) not in mechanism[
                "single_task_active_aic_groups"]
            or any(
                task_count < mechanism["minimum_task_count"]
                or task_count > mechanism["maximum_task_count"]
                for task_count in active_task_counts)):
        limitations += (
            "cann_split_fuse_handoff_span_domain_unavailable",
        )
    limitations += ("cann_split_fuse_mm2_dma_compute_overlap_unavailable",)
    return dataclasses.replace(
        inp,
        pipeline_plan=plan,
        support_level="partial",
        limitations=limitations,
    )


def _grouped_mla_tile(kernel: KernelSpec):
    """Return explicit A800 grouped-MLA launch semantics, if present."""

    if (kernel.launch is not None
            and kernel.launch.algorithm == "vllm_030_fa3_sm90_gqa_prefill_partial"):
        # This GQA tile also names qk_head/value_head but has no MLA KV axis.
        return None
    tiling = kernel.launch.tiling
    if tiling is None:
        return None
    required = {"query_heads", "kv", "qk_head", "value_head"}
    marker_axes = {"query_heads", "qk_head", "value_head"}
    if not marker_axes.intersection(tiling.tile_shape):
        return None
    present = required.intersection(tiling.tile_shape)
    if present != required:
        raise ValueError(
            "grouped MLA tile is incomplete: "
            f"missing={sorted(required - present)}")
    if kernel.attributes["kv_layout"] != "mla":
        raise ValueError("grouped MLA tile requires kv_layout='mla'")
    return tuple(int(tiling.tile_shape[name]) for name in (
        "query_heads", "kv", "qk_head", "value_head"))


def _floor_sum_nonnegative(count: int, divisor: int, stride: int,
                           offset: int) -> int:
    """Return ``sum(floor((stride*i + offset) / divisor), i=0..count-1)``."""

    if min(count, stride, offset) < 0 or divisor <= 0:
        raise ValueError("floor-sum inputs must be non-negative")
    result = 0
    while True:
        if stride >= divisor:
            result += count * (count - 1) * (stride // divisor) // 2
            stride %= divisor
        if offset >= divisor:
            result += count * (offset // divisor)
            offset %= divisor
        maximum = stride * count + offset
        if maximum < divisor:
            return result
        count = maximum // divisor
        offset = maximum % divisor
        stride, divisor = divisor, stride


def _sum_ceiled_arithmetic_progression(
        count: int, first: int, stride: int, divisor: int) -> int:
    """Sum positive integer ceiling divisions in logarithmic time."""

    if count < 0 or first <= 0 or stride <= 0 or divisor <= 0:
        raise ValueError("ceiling-progression inputs must be positive")
    return _floor_sum_nonnegative(
        count, divisor, stride, first + divisor - 1)


def _attention_tile_aggregate(
        kernel: KernelSpec, tile_q: int, tile_kv: int) -> _AttentionTileAggregate:
    """Aggregate tiled attention work using request-shape arithmetic."""

    if tile_q <= 0 or tile_kv <= 0:
        raise ValueError("FLASH_ATTN Q/KV tiles must be positive")
    causal = kernel.attributes["causal"]
    if not isinstance(causal, bool):
        raise TypeError("FLASH_ATTN causal must be boolean")
    request_shape_counts = {}
    for request_shape in zip(*_request_lengths(kernel)):
        request_shape_counts[request_shape] = (
            request_shape_counts[request_shape] + 1
            if request_shape in request_shape_counts else 1)

    tile_count = 0
    query_rows = 0
    attended_kv = 0
    kv_tiles = 0
    for (query, key_value), request_count in request_shape_counts.items():
        if query > key_value:
            raise ValueError(
                "FLASH_ATTN bottom-right workload requires q_len <= kv_len")
        full_tiles, tail_rows = divmod(query, tile_q)
        request_tiles = full_tiles + int(tail_rows > 0)
        if causal:
            context = key_value - query
            request_attended = (
                full_tiles * context
                + tile_q * full_tiles * (full_tiles + 1) // 2)
            request_kv_tiles = _sum_ceiled_arithmetic_progression(
                full_tiles, context + tile_q, tile_q, tile_kv)
            if tail_rows:
                request_attended += key_value
                request_kv_tiles += math.ceil(key_value / tile_kv)
        else:
            request_attended = request_tiles * key_value
            request_kv_tiles = (
                request_tiles * math.ceil(key_value / tile_kv))
        tile_count += request_count * request_tiles
        query_rows += request_count * query
        attended_kv += request_count * request_attended
        kv_tiles += request_count * request_kv_tiles
    if tile_count <= 0:
        raise ValueError("FLASH_ATTN workload produced no Q tiles")
    return _AttentionTileAggregate(
        tile_count=tile_count,
        query_rows=query_rows,
        attended_kv=attended_kv,
        kv_tiles=kv_tiles,
    )


def _attention_tile_class_counts(
        kernel: KernelSpec, tile_q: int, tile_kv: int):
    """Return real Q-tile classes without repeating identical requests."""

    if tile_q <= 0 or tile_kv <= 0:
        raise ValueError("FLASH_ATTN Q/KV tiles must be positive")
    causal = kernel.attributes["causal"]
    if not isinstance(causal, bool):
        raise TypeError("FLASH_ATTN causal must be boolean")
    request_shape_counts = {}
    for request_shape in zip(*_request_lengths(kernel)):
        request_shape_counts[request_shape] = (
            request_shape_counts[request_shape] + 1
            if request_shape in request_shape_counts else 1)
    result = {}
    for (query, key_value), request_count in request_shape_counts.items():
        if query > key_value:
            raise ValueError(
                "FLASH_ATTN bottom-right workload requires q_len <= kv_len")
        context = key_value - query
        for start in range(0, query, tile_q):
            end = min(query, start + tile_q)
            rows = end - start
            if causal:
                attended = context + end
                elements = rows * (
                    2 * context + start + 1 + end) // 2
            else:
                attended = key_value
                elements = rows * key_value
            if attended <= 0 or elements <= 0:
                raise ValueError(
                    "FLASH_ATTN real Q tile must attend positive KV work")
            item = _AttentionTileWork(
                query_rows=rows,
                attention_elements=elements,
                attended_kv=attended,
                kv_tiles=math.ceil(attended / tile_kv),
            )
            result[item] = (
                result[item] + request_count
                if item in result else request_count)
    if not result:
        raise ValueError("FLASH_ATTN workload produced no Q-tile classes")
    return tuple(result.items())


def _split_tile_counts(kv_tiles: int, splits: int,
                       *, include_empty: bool = False):
    """Return FA2's source-defined request-local KV range per split.

    ``compute_attn_1rowblock_splitkv`` constructs ``BlockInfo`` for the
    current batch item, then derives ``n_blocks_per_split`` from that
    request's ``actual_seqlen_k``.  A varlen launch therefore redistributes
    every request's own KV tiles across all configured splits; it does not use
    the launch-wide maximum sequence length as the split width.
    """

    if kv_tiles <= 0 or splits <= 0:
        raise ValueError("FLASH_ATTN KV tiles and splits must be positive")
    tiles_per_split = math.ceil(kv_tiles / splits)
    counts = tuple(
        min(tiles_per_split, max(0, kv_tiles - index * tiles_per_split))
        for index in range(splits)
    )
    return counts if include_empty else tuple(
        count for count in counts if count > 0)


def _cann_kv_fill_share(hardware, kv_heads: int, head_dim: int,
                        element_bytes: int,
                        kv_footprint_bytes: float) -> int:
    """Return adjacent kv_head slices served by one L2 read miss-allocate.

    CANN FIA loads K and V with ``DataCopy(Nd2NzParams)``
    (``fia_block_cube_nonquant_gqa.h`` CopyKToL1/CopyVToL1 ->
    ``CopyKvGmToL1`` -> ``CopySingleMatrixNDToNZ``).  One kv_head occupies
    ``head_dim * element_bytes`` contiguous bytes of a row whose stride is
    ``kv_heads * head_dim * element_bytes``, and the launch reads every
    kv_head of every row.  Adjacent slices that fall inside one fill span are
    served by a single miss-allocate, so the compulsory DRAM account divides
    by the number of slices per span.

    The span is measured by an independent ND2NZ probe that never executes
    attention (``fia_kv_line_fill_span``): the 256 B slice saturates at two
    workers and the 128 B slice at four, which fixes the span rather than
    reading it off any production traffic.

    The credit also requires the line to survive from one slice's access to
    its neighbour's.  Once the launch's own K/V stream exceeds the measured
    L2 replacement capacity it evicts those lines before the neighbour
    arrives, and CANN itself stops caching K/V at that point: the tiling
    raises ``l2CacheOffFlag`` and the kernel calls
    ``SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)`` on keyGm and valueGm
    (``fia_block_cube_nonquant_gqa.h``).  The exact tiling threshold is not
    exposed by any readable CANN artefact, so the boundary here is the
    independently measured replacement capacity the GEMM and MoE miss models
    already consume; shapes past it keep the full compulsory account.

    Outside the measured coverage -- no span configured for the target, a
    slice at least as wide as the span, or a row holding a single kv_head --
    the caller keeps the full compulsory account.
    """

    span = hardware.fia_kv_line_fill_span
    if span is None:
        return 1
    fill_span_bytes = int(span["fill_span_bytes"])
    slice_bytes = head_dim * element_bytes
    if slice_bytes <= 0 or slice_bytes >= fill_span_bytes:
        return 1
    if kv_footprint_bytes > float(hardware.l2_replacement_capacity_bytes):
        return 1
    return max(1, min(kv_heads, fill_span_bytes // slice_bytes))


def _tensor_traffic(
        kernel: KernelSpec, hardware, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET):
    batch, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    splits = _splits(kernel)
    scratch_dram_width = _split_scratch_dram_width(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    from src.kernel.memory_access_builder import has_public_attention_source,logical_interface_memory_access_program
    from src.kernel.memory_path_solver import public_attention_source_work
    public_work=(public_attention_source_work(logical_interface_memory_access_program(kernel,hardware),
        expected_kernel=kernel,hardware=hardware) if has_public_attention_source(kernel) else None)
    grouped_mla = _grouped_mla_tile(kernel)
    if grouped_mla is not None:
        if has_grouped_mla_source(kernel):
            source = current_grouped_mla_source_work(kernel)
            return float(source['l2_projection_bytes']),float(source['dram_projection_bytes'])
        head_tile, tile_kv, qk_head_dim, value_head_dim = grouped_mla
        if q_len != 1 or any(length != 1 for length in q_lens):
            raise ValueError("grouped MLA stage1 supports decode Q=1 only")
        head_groups = math.ceil(query_heads / head_tile)
        padded_kv = sum(math.ceil(length / tile_kv) * tile_kv
                        for length in kv_lens)
        kv_replay = (
            head_groups * padded_kv * public_work['kv_head_row_bytes'])
        q_replay = splits * public_work['query_bytes']
        scratch = public_work['write_bytes']
        l2_total = float(kv_replay + q_replay + scratch)
        scratch_dram = scratch/4*scratch_dram_width
        # Each head group replays both operand halves, which is why the L2
        # account above carries qk_head_dim + value_head_dim.  DRAM holds the
        # cached object once: MLA stores one latent per position and the value
        # channels are the leading channels of the same qk-wide object, so the
        # unique footprint is the qk width, not the sum of the two halves.
        dram = float(
            public_work['key_bytes']+public_work['query_bytes']
            + scratch_dram)
        return l2_total, min(l2_total, dram)

    cann_mla = _cann_mla_dimensions(kernel)
    if cann_mla is not None:
        from src.kernel.memory_cann_mla import has_cann_mla_source,current_cann_mla_source_work
        if has_cann_mla_source(kernel):
            work=current_cann_mla_source_work(kernel,hardware)
            compulsory=float(work['priced_interface_bytes'])
            return compulsory,compulsory
        compulsory = float(public_work['read_bytes']+public_work['write_bytes'])
        return compulsory, compulsory

    metax_mla = _metax_mla_dimensions(kernel)
    if metax_mla is not None:
        qk_head_dim, value_head_dim = metax_mla
        if _is_source_bound_metax_mla_main(kernel):
            source = current_metax_mla_source_work(kernel)
            return float(source['l2_projection_bytes']),float(source['dram_projection_bytes'])
        # Preserve the existing input-only price while the external scratch
        # lifecycle is unbound; mathematical outputs remain in the source.
        compulsory_input = float(public_work['read_bytes'])
        return compulsory_input, compulsory_input

    tile = _attention_tile(kernel)
    positions = query_heads * sum(q_lens)
    if has_cann_tnd_source(kernel):
        source = current_cann_tnd_source_work(kernel,hardware)
        q_bytes,unique_kv = source['query_unique_bytes'],source['kv_interface_bytes']
    elif has_metax_varlen_source(kernel):
        source = current_metax_varlen_source_work(kernel,hardware)
        q_bytes,unique_kv = source['query_unique_bytes'],source['kv_interface_bytes']
    elif has_cann_attention_source(kernel):
        source = current_cann_attention_source_work(kernel,hardware)
        q_bytes,unique_kv = source['query_unique_bytes'],source['kv_interface_bytes']
    elif has_fa2_split_source(kernel):
        source = current_fa2_split_source_work(kernel)
        q_bytes,unique_kv = source['query_unique_bytes'],source['kv_unique_bytes']
    elif has_dense_flash_attention_source(kernel):
        source=dense_attention_prediction_work(kernel,hardware)
        q_bytes,key_bytes,value_bytes=(source[role] for role in ('query','key','value'))
        unique_kv = key_bytes + value_bytes
    else:
        q_bytes=public_work['query_bytes']
        unique_kv=public_work['key_bytes']+public_work['value_bytes']
    # Compulsory KV misses, after the adjacent-slice sharing the CANN ND2NZ
    # K/V copy geometry produces.  L2 demand is unchanged: every requested
    # byte still crosses the source interface exactly once.
    #
    # The gate compares the unique K/V footprint, not the traffic a launch
    # replays: an independent probe grows the walk between two touches of one
    # line over a 256-fold range and the sharing is unchanged, because the two
    # adjacent slices are read concurrently and nothing flows between them
    # (fia_kv_line_share_replay_independence_cann_8_5_1.json).  Gating on
    # replayed traffic was tried, is falsified by that probe, and also moved
    # 910B aggregate traffic from 11.499 to 12.078 percent.
    if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET:
        split_fuse_work = _cann_split_fuse_work(kernel)
    kv_fill_share = _cann_kv_fill_share(
        hardware, kv_heads, head_dim, element_bytes, unique_kv)
    unique_kv_dram = unique_kv / kv_fill_share
    # Q is laid out the same way K and V are: one head occupies
    # head_dim * element_bytes contiguous bytes of a row whose stride is
    # query_heads * head_dim * element_bytes, and the launch reads every
    # query head, so the adjacent-slice credit the probe measured applies to
    # it unchanged.  Decode reads so little Q that leaving it at the full
    # account cost under half a percent, but a prefill launch reads as many
    # Q bytes as K and V together.
    q_fill_share = _cann_kv_fill_share(
        hardware, query_heads, head_dim, element_bytes, float(q_bytes))
    q_bytes_dram = q_bytes / q_fill_share
    if split_fuse_work is not None:
        # Q is issued once.  Each Q-block/GQA-head-group task issues its own K
        # and V source reads.  L2-path demand includes those source-issued
        # replays; DRAM retains only the independently known compulsory
        # Q/K/V/O lower bound because replay residency is unavailable.
        kv_replay = (
            2 * split_fuse_work.kv_replay_positions
            * head_dim * element_bytes)
        output = q_bytes
        # Vec1 also fetches the source-defined uint8 causal mask through MTE2.
        # Its L2 service is enumerable from the masked stack geometry, while
        # the mask's external-memory residency is not established; therefore
        # it does not increase the independently known DRAM lower bound.
        l2_total = float(
            q_bytes + kv_replay + output
            + split_fuse_work.mask_mte2_bytes)
        dram = float(q_bytes_dram + unique_kv_dram + output)
        return l2_total, dram
    if tile is None:
        # Untiled native operators (currently CANN) expose no replay geometry.
        # Preserve only the independently known logical input/output
        # footprint.  A split main launch materializes FP32 partial output and
        # LSE scratch; its combine consumer is modeled separately.  Global
        # scratch accesses traverse L2, but their DRAM residency is unknown and
        # is reported as an explicit support limitation below.
        if splits == 1:
            output = public_work['write_bytes'] if public_work is not None else _output_bytes(kernel,hardware)
            dram = float(unique_kv_dram + q_bytes + output)
        else:
            output = public_work['write_bytes'] if public_work is not None else _output_bytes(kernel,hardware)
            scratch_dram = output/4*scratch_dram_width
            dram = float(unique_kv_dram + q_bytes + scratch_dram)
        l2_total = float(unique_kv + q_bytes + output)
        return l2_total, dram
    tile_q, tile_kv = tile
    is_prefill = kernel.variant.lower() == "prefill"
    if is_prefill and splits != 1:
        raise ValueError("FLASH_ATTN prefill launch requires one KV split")
    replay_heads = query_heads if is_prefill else kv_heads
    unpadded_flops = kernel.launch.parameters["unpadded_flops"]
    analytical_fa2_prefill = (
        _is_source_bound_fa2_pipeline(kernel) and is_prefill)
    if analytical_fa2_prefill:
        aggregate = _attention_tile_aggregate(kernel, tile_q, tile_kv)
        if unpadded_flops == 0.0:
            replay_positions = aggregate.kv_tiles * tile_kv
        elif unpadded_flops == 1.0:
            replay_positions = aggregate.attended_kv
        else:
            raise ValueError(
                "KernelSpec allowed an invalid FLASH_ATTN unpadded_flops "
                "value")
        panel_count = aggregate.kv_tiles
        q_replay = q_bytes
    else:
        work = _attention_tile_class_counts(kernel, tile_q, tile_kv)
        if _has_metax_q64_prefill_pipeline(kernel):
            # The direct Q64 source emits its fixed-width K and V panel
            # copies before masking tail score lanes.  This is an ABI/source
            # property of the same trait used for full-tile prefill, so a
            # ragged request retains complete 64-token panel reads.
            replay_positions = sum(
                count * item.kv_tiles for item, count in work) * tile_kv
        elif (kernel.launch.algorithm.startswith(
                _METAX_STATIC_TRAIT_DIRECT_PREFIX)
                or _is_source_bound_metax_tile16_program(kernel)):
            # The versioned trait and the exact MMA counter close scheduled
            # Tensor work, not predicated global tail loads.  Keep memory
            # traffic at the independently known in-range extent.
            replay_positions = sum(
                count * item.attended_kv for item, count in work)
        elif unpadded_flops == 0.0:
            # A source/counter-bound complete tile program also issues the
            # full BlockN operand extent for every scheduled KV step.
            replay_positions = sum(
                count * item.kv_tiles for item, count in work) * tile_kv
        elif unpadded_flops == 1.0:
            # Static BlockM/BlockN metadata alone does not prove tail loads.
            # Keep the unique in-range positions while still using the tile
            # to derive Q/KV steps and split participation.
            replay_positions = sum(
                count * item.attended_kv for item, count in work)
        else:
            raise ValueError(
                "KernelSpec allowed an invalid FLASH_ATTN unpadded_flops "
                "value")
        panel_count = sum(count * item.kv_tiles for item, count in work)
        q_replay = sum(
            count * item.query_rows * query_heads * head_dim * element_bytes
            * len(_split_tile_counts(item.kv_tiles, splits))
            for item, count in work)
    kv_replay = (
        2 * replay_heads * replay_positions * head_dim * element_bytes)
    page_table_replay = 0
    page_table_bytes_per_panel = kernel.launch.parameters[
        "page_table_bytes_per_kv_panel"]
    if page_table_bytes_per_panel > 0:
        page_table_replay = (
            replay_heads * panel_count * page_table_bytes_per_panel)
    if splits == 1:
        # The pinned MACA 3.7.2 Q64 prefill epilogue writes the BF16 output
        # through shared memory and independently stores one FP32 log-sum-exp
        # value per query row.  The source-bound LSE store is part of the main
        # kernel ABI; it is not split scratch and was previously omitted.
        lse = (
            positions * 4
            if (_has_metax_q64_prefill_pipeline(kernel)
                or kernel.launch.algorithm in {
                    _METAX_372_Q128_PREFILL_ALGORITHM,
                    _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM,
                })
            else 0)
        output = (public_work['write_bytes'] if public_work is not None else
            _output_bytes(kernel,hardware) if has_dense_flash_attention_source(kernel) else q_bytes+lse)
        residency = _metax_prefill_residency_dram_bytes(
            kernel, hardware, tile_q, tile_kv, q_lens, kv_lens)
        if residency is None:
            dram = float(unique_kv + q_bytes + output)
        else:
            # The compulsory envelope above assumes every replayed K/V panel
            # is still resident when the next query tile reaches it.  The
            # grid states otherwise: it dispatches (query_heads, batch,
            # query_tiles) with x fastest, so two query tiles of one request
            # are query_heads * batch programs apart and every intervening
            # program's panels lie between them.  Whether the panel survives
            # that distance is a capacity question, and it is the shared miss
            # core -- the same solver the dense and grouped families use --
            # that answers it.  Replacing the assumption with the replay
            # leaves L2 untouched: `l2_total` below still counts every
            # requested byte once at the source interface.
            dram = float(residency + output)
    else:
        # Split main kernels write FP32 partial output plus one FP32 LSE value
        # per (position, split).  Every global write traverses L2; the explicit
        # launch fact determines whether that materialized width reaches DRAM.
        output = public_work['write_bytes'] if public_work is not None else _output_bytes(kernel,hardware)
        scratch_dram = output/4*scratch_dram_width
        dram = float(unique_kv + q_bytes + scratch_dram)
    l2_total = float(kv_replay + q_replay + page_table_replay + output)
    return l2_total, min(l2_total, dram)


def _metax_prefill_residency_dram_bytes(
        kernel: KernelSpec, hardware, tile_q, tile_kv, q_lens, kv_lens):
    """Return replayed K/V+Q DRAM read bytes for the MACA Q128 prefill program.

    Only the pinned non-split `prefill_q128_full_tile` variant is covered.
    That launch is the one whose grid fixes the dispatch order this replay
    depends on -- `(num_query_heads, batch, query_tiles)`, x fastest -- and
    whose whole-request causal reach lets the panel count per query tile be
    stated from the tile geometry alone.  Its split sibling partitions the
    K/V range across programs instead, so the same reach does not describe it
    and it keeps the compulsory envelope.

    Ragged requests are covered: the grid's tile dimension spans the longest
    request and a program past its own request's tile count returns before
    issuing a load, so the per-request reach follows from the retained
    per-request lengths alone.  The caller keeps the compulsory envelope for
    every uncovered launch.
    """

    # The three MACA prefill programs differ in the grid they state, and the
    # grid is what fixes how far apart two query tiles of one request sit.
    # `prefill_q128_full_tile` states (query_heads, batch, query_tiles); the
    # two split-path programs state (query_heads, query_tiles, batch).  x is
    # fastest in all three, so the query tile is the slowest dimension in the
    # first and the middle one in the others.
    tile_slowest = {
        _METAX_372_Q128_PREFILL_ALGORITHM: True,
        _METAX_372_Q64_PREFILL_ALGORITHM: False,
        _METAX_372_Q64_PACKED_PREFILL_ALGORITHM: False,
        _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM: False,
    }
    if kernel.launch.algorithm not in tile_slowest:
        return None
    if kernel.attributes["causal"] is not True:
        return None
    if len(q_lens) != len(kv_lens) or not q_lens:
        return None
    if any(length <= 1 for length in q_lens):
        return None
    if any(q > k for q, k in zip(q_lens, kv_lens, strict=True)):
        return None
    requested, missed = simulate_attention_prefill_residency(
        batch=int(kernel.shape["batch"]),
        query_heads=int(kernel.shape["num_query_heads"]),
        kv_heads=int(kernel.shape["num_kv_heads"]),
        head_dim=int(kernel.shape["head_dim"]),
        element_bytes=int(kernel.shape["dtype_bytes"]),
        query_tile=int(tile_q),
        kv_panel=int(tile_kv),
        query_lengths=tuple(int(length) for length in q_lens),
        kv_lengths=tuple(int(length) for length in kv_lens),
        wave_size=int(hardware.num_compute_units),
        capacity_bytes=int(hardware.l2_replacement_capacity_bytes),
        query_tile_is_slowest=tile_slowest[kernel.launch.algorithm],
    )
    if missed > requested:
        raise AssertionError(
            "attention prefill residency reported more misses than requests")
    return float(missed)


def _combine_traffic(kernel: KernelSpec, hardware):
    if has_fa2_split_source(kernel):
        source = current_fa2_split_source_work(kernel)
        return float(source['combine_l2_bytes']),float(source['combine_dram_bytes'])
    if has_grouped_mla_source(kernel):
        source = current_grouped_mla_source_work(kernel)
        return float(source['l2_projection_bytes']),float(source['dram_projection_bytes'])
    if _is_source_bound_metax_mla_combine(kernel):
        scratch_read, final_write = _metax_mla_combine_traffic_bytes(kernel)
        # The persistent main kernel's partial FP32 output/LSE is consumed by
        # a later launch.  C500's write path does not retain that scratch as a
        # guaranteed L2-resident handoff: source-sized scratch reads match the
        # downstream DNoC read counter across the canonical shape sweep.
        return scratch_read + final_write, scratch_read + final_write
    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    from src.kernel.memory_path_solver import logical_combine_source_traffic
    program=logical_interface_memory_access_program(kernel,hardware)
    return logical_combine_source_traffic(program,expected_kernel=kernel,hardware=hardware,
        known_output_only=_is_unbound_metax_mla_scratch(kernel))


def _tensor_flops(
        kernel: KernelSpec, hardware, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET):
    batch, q_len, kv_len, head_dim, query_heads, kv_heads = _shape(kernel)
    if query_heads % kv_heads:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    tile = _attention_tile(kernel)
    grouped_mla = _grouped_mla_tile(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    unpadded_flops = kernel.launch.parameters["unpadded_flops"]
    if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET:
        split_fuse_work = _cann_split_fuse_work(kernel)
    if split_fuse_work is not None:
        value = float(
            4 * head_dim * split_fuse_work.mmad_attention_elements)
        return {
            "compute_tensor": value,
            "compute_reduction": 0.0,
            "compute_fma": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
            "warp_issue": 0.0,
        }
    cann_mla = _cann_mla_dimensions(kernel)
    if cann_mla is not None:
        qk_head_dim, value_head_dim = cann_mla
        tokens = sum(query * key_value for query, key_value
                     in zip(q_lens, kv_lens))
        value = float(
            2 * query_heads * tokens * (qk_head_dim + value_head_dim))
        return {
            "compute_tensor": value,
            "compute_reduction": 0.0,
            "compute_fma": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
            "warp_issue": 0.0,
        }
    metax_mla = _metax_mla_dimensions(kernel)
    if metax_mla is not None:
        if _is_source_bound_metax_mla_main(kernel):
            work = _metax_mla_scheduler_work(kernel)
            value = float(
                work.kv_block_count * _METAX_MLA_MMA_PER_KV_BLOCK
                * _METAX_372_FA_TENSOR_OPS_PER_MMA)
        else:
            qk_head_dim, value_head_dim = metax_mla
            tokens = sum(query * key_value for query, key_value
                         in zip(q_lens, kv_lens))
            value = float(
                2 * query_heads * tokens * (qk_head_dim + value_head_dim))
        return {
            "compute_tensor": value,
            "compute_reduction": 0.0,
            "compute_fma": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
            "warp_issue": 0.0,
        }
    if grouped_mla is not None:
        head_tile, tile_kv, qk_head_dim, value_head_dim = grouped_mla
        if unpadded_flops == 1.0:
            tokens = sum(query * key_value for query, key_value
                         in zip(q_lens, kv_lens))
            value = float(
                2 * query_heads * tokens * (qk_head_dim + value_head_dim))
        elif unpadded_flops == 0.0:
            padded_heads = math.ceil(query_heads / head_tile) * head_tile
            padded_kv = sum(math.ceil(length / tile_kv) * tile_kv
                            for length in kv_lens)
            value = float(
                2 * padded_heads * padded_kv
                * (qk_head_dim + value_head_dim))
        else:
            raise AssertionError(
                "KernelSpec allowed an invalid FLASH_ATTN unpadded_flops value")
        result = {
            "compute_tensor": value,
            "compute_reduction": 0.0,
            "compute_fma": 0.0,
            "compute_alu": 0.0,
            "compute_sfu": 0.0,
            "warp_issue": 0.0,
        }
        if _splits(kernel) > 1:
            warp_instructions, fixed_dependency_cycles = (
                _fa2_main_sass_schedule(kernel))
            result["warp_issue"] = warp_instructions
            if kernel.launch.grid_size <= hardware.num_compute_units:
                result["compute_reduction"] = fixed_dependency_cycles
        return result
    if unpadded_flops == 1.0:
        attention_elements = math.fsum(
            (
                query * key_value
                if not kernel.attributes["causal"] else
                query * (key_value - query)
                + query * (query + 1) / 2
            )
            for query, key_value in zip(q_lens, kv_lens)
        )
        value = float(
            4 * query_heads * head_dim
            * attention_elements)
    elif unpadded_flops == 0.0:
        if tile is None:
            raise ValueError(
                "padded FLASH_ATTN FLOP accounting requires Q/KV tile geometry")
        tile_q, tile_kv = tile
        iterations = (
            2 * math.ceil(tile_q / 16) * math.ceil(tile_kv / 8)
            * math.ceil(head_dim / 16))
        class_counts, _ = _pipeline_cta_class_counts(
            kernel, tile_q, tile_kv, grouped_mla=False)
        value = float(
            sum(item.step_count * count for item, count in class_counts)
            * iterations * 4096)
    else:
        raise AssertionError(
            "KernelSpec allowed an invalid FLASH_ATTN unpadded_flops value")
    result = {
        "compute_tensor": value,
        "compute_reduction": 0.0,
        "compute_fma": 0.0,
        "compute_alu": 0.0,
        "compute_sfu": 0.0,
        "warp_issue": 0.0,
    }
    metax_barrier_cycles = _metax_q64_prefill_barrier_cycles(
        kernel, hardware)
    if metax_barrier_cycles > 0:
        result["compute_reduction"] = metax_barrier_cycles
    elif _is_source_bound_fa2_pipeline(kernel):
        result["compute_reduction"] = (
            _fa2_register_dependency_cycles(kernel, hardware)
            + _fa2_local_spill_dependency_cycles(kernel, hardware))
        has_main_sass_program = (
            (_splits(kernel) > 1
             and kernel.attributes["kv_layout"] == "contiguous")
            or _is_source_bound_fa2_dense_sass_program(kernel))
        if has_main_sass_program:
            warp_instructions, _ = (
                _fa2_main_sass_schedule(kernel))
            result["warp_issue"] = warp_instructions
            if kernel.launch.grid_size <= hardware.num_compute_units:
                result["compute_reduction"] = warp_instructions
    return result


def _combine_flops(kernel: KernelSpec, hardware):
    _, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    splits = _splits(kernel)
    q_lens, _ = _request_lengths(kernel)
    positions = sum(q_lens) * query_heads
    if _is_source_bound_metax_mla_combine(kernel):
        work = _metax_mla_scheduler_work(kernel)
        active_splits = tuple(
            count for count in work.splits_per_request if count > 1)
        active_positions = query_heads * len(active_splits)
        split_positions = query_heads * sum(active_splits)
        block_m = _metax_mla_combine_block_m(kernel)
        rows_per_request = q_len * query_heads
        if rows_per_request % block_m:
            raise ValueError(
                "MetaX FlashMLA combine rows are not CTA-aligned")
        ctas_per_request = rows_per_request // block_m
        barrier_mechanism = (
            hardware.flash_mla_combine_barrier_service_cycles)
        if barrier_mechanism is None:
            raise ValueError(
                "MetaX FlashMLA combine requires CTA-barrier evidence")
        if (kernel.launch.block_size
                != barrier_mechanism["threads_per_cta"]
                or barrier_mechanism["static_shared_bytes"] != 2560
                or barrier_mechanism["resident_ctas_per_unit"] != 2):
            raise ValueError(
                "MetaX FlashMLA combine does not match CTA-barrier evidence")
        # Canonical non-duration BSM counters close across all source-bound
        # FlashMLA shapes as four wave instructions per split iteration.  A
        # 256-thread CTA contains four waves, so every active split executes
        # one source CTA barrier before consuming its partial output.
        barrier_cycles = float(
            ctas_per_request * sum(active_splits)
            * barrier_mechanism["barrier_cycles"])
        max_splits = 128
        fma_operations = float(
            2 * head_dim * split_positions
            + (2 * max_splits - 1) * active_positions)
        sfu_operations = float(
            max_splits * active_positions
            + split_positions
            + active_positions)
    elif _is_source_bound_grouped_mla_combine(kernel):
        # vLLM 0.11's Triton reducer updates an FP32 output accumulator for
        # every split, then performs one reciprocal-scale pass and one final
        # log.  Count scalar multiply/add work on FMA and exp/log/reciprocal
        # work on SFU; the loop bound is the executed split count.
        fma_operations = (
            splits * (3.0 * head_dim + 2.0) + head_dim) * positions
        sfu_operations = float((2 * splits + 2) * positions)
    elif _is_source_bound_fa2_combine(kernel):
        # Every CTA instantiates the complete compile-time kBlockM row tile.
        # Tail rows predicate global traffic and stores, but the LSE and
        # partial-output arithmetic loops still execute for all tile rows.
        # The pinned launch selector and runtime-symbol guard establish both
        # grid size and kBlockM, so this padding is a static instruction fact.
        positions = kernel.launch.grid_size * _fa2_combine_block_m(kernel)
        # Each of the 128 threads executes the post-max exp and post-sum log
        # paths even when its compile-time LSE lane is padding.  Only the
        # scale-store path is predicated to the active split-by-row rectangle.
        # The accumulator loop performs four FP32 FMAs per thread and split.
        # Count source-issued scalar work rather than only logical output rows.
        max_splits = _fa2_combine_max_splits(kernel)
        if max_splits * _fa2_combine_block_m(kernel) > 128:
            raise ValueError(
                "FA2 combine scalar work requires one LSE value per thread")
        cta_count = kernel.launch.grid_size
        block_m = _fa2_combine_block_m(kernel)
        threads = kernel.launch.block_size
        fma_operations = float(cta_count * (
            2 * splits * head_dim * block_m
            + 2 * threads + splits * block_m))
        sfu_operations = float(
            cta_count * (2 * threads + splits * block_m))
    elif _is_source_bound_metax_fa2_combine(kernel):
        # The source-fragment service already covers the LSE transpose,
        # reductions, exp/log, scale write, and output conversion.  Counting
        # their scalar FMA/SFU instructions again here would schedule the
        # same issued instructions twice.  The fragment intentionally omits
        # only the Oaccum copy/FMA loop, so retain that work alone.
        fma_operations = float(2 * splits * head_dim * positions)
        sfu_operations = 0.0
    else:
        fma_operations = (
            2.0 * splits * head_dim + 2.0 * splits - 1.0) * positions
        sfu_operations = splits * positions
    result = {
        "compute_tensor": 0.0,
        "compute_reduction": 0.0,
        "compute_fma": fma_operations,
        "compute_alu": 0.0,
        "compute_sfu": float(sfu_operations),
        "warp_issue": 0.0,
    }
    if _is_source_bound_metax_mla_combine(kernel):
        result["compute_reduction"] = barrier_cycles
    if _is_source_bound_fa2_combine(kernel):
        dependency_stages = _fa2_combine_dependency_stages(kernel, hardware)
        oaccum_load_use = _fa2_oaccum_load_use_cycles(kernel, hardware)
        result["compute_reduction"] = (
            (math.fsum(dependency_stages.values())
             + math.fsum(oaccum_load_use)) * kernel.launch.grid_size)
    elif (_is_source_bound_metax_fa2_combine(kernel)
          and hardware.metax_fa2_combine_dependency_cycles is not None):
        dependency_stages = _metax_fa2_combine_dependency_stages(
            kernel, hardware)
        result["compute_reduction"] = (
            math.fsum(dependency_stages.values())
            * kernel.launch.grid_size)
    return result


def _pipeline_geometry(kernel: KernelSpec):
    tile = _attention_tile(kernel)
    if tile is not None:
        return tile[0], tile[1], float(kernel.shape["head_dim"]), False
    grouped = _grouped_mla_tile(kernel)
    if grouped is None:
        return None
    head_tile, tile_kv, qk_head_dim, value_head_dim = grouped
    # The common pipeline helper prices two equal QK/PV dimensions as 4*M*N*D.
    # Their mean preserves the exact grouped-MLA 2*M*N*(Dqk+Dv) work and load.
    effective_head_dim = (qk_head_dim + value_head_dim) / 2.0
    return head_tile, tile_kv, effective_head_dim, True


def _work_distribution(kernel: KernelSpec) -> str:
    """Return the explicit semantic-work to physical-grid mapping."""

    algorithm = kernel.launch.algorithm
    direct_prefixes = (
        "flash_attention:",
        _METAX_STATIC_TRAIT_DIRECT_PREFIX,
        "flash_attn_dense",
        "flash_attn_splitkv",
        "vllm_flash_attn_splitkv",
        _A800_GROUPED_MLA_STAGE1_ALGORITHM,
        "metax_flash_attention_generic",
        "metax_flash_attention_generic",
        "test_flash_attn_tiled_prefill",
    )
    if algorithm == "metax_flash_attention_persistent_queue":
        return "persistent_queue"
    elif algorithm == _METAX_372_Q64_PACKED_PREFILL_ALGORITHM:
        tile = _attention_tile(kernel)
        if tile is None:
            raise ValueError(
                "MetaX Q64 packed prefill requires a source-bound tile")
        scheduled = flash_attention_packed_varlen_scheduled_workgroups(
            batch=int(kernel.shape["batch"]),
            q_len=int(kernel.shape["q_len"]),
            tile_q=tile[0],
            query_heads=int(kernel.shape["num_query_heads"]),
            query_heads_per_workgroup=int(
                kernel.launch.tiling.tile_shape[
                    "query_heads_per_workgroup"]),
            splits=_splits(kernel),
        )
        # The packed ABI may launch either a smaller persistent worker pool or
        # the complete padded logical grid.  In the latter regime mcTracer's
        # physical grid and mcProfiler's WORKGROUPS both equal ``scheduled``;
        # every physical CTA therefore owns one logical job and no queue
        # assignment is needed to construct its source program.
        expected_direct_grid = (
            int(kernel.shape["num_query_heads"]),
            math.ceil(int(kernel.shape["q_len"]) / tile[0]),
            int(kernel.shape["batch"]),
        )
        if (kernel.launch.grid_size == scheduled
                and kernel.launch.grid == expected_direct_grid):
            return "direct_cta"
        return "packed_varlen_queue"
    elif algorithm in {
            "metax_flash_attention_packed_varlen_queue",
    }:
        return "packed_varlen_queue"
    elif algorithm == "metax_flash_attention_grouped_direct":
        return "direct_cta"
    elif algorithm.startswith(direct_prefixes):
        return "direct_cta"
    else:
        raise ValueError(
            "FLASH_ATTN launch has no registered work distribution: "
            f"{algorithm!r}")


def _has_semantic_tile_evidence(kernel: KernelSpec) -> bool:
    """Whether a launch label may publish its tile to semantic demand."""

    if (_is_source_bound_metax_tile16_program(kernel)
            or kernel.launch.algorithm
            == _METAX_372_Q64_PACKED_PREFILL_ALGORITHM):
        return True
    return kernel.launch.algorithm not in {
        "metax_flash_attention_grouped_direct",
        "metax_flash_attention_packed_varlen_queue",
        "metax_flash_attention_persistent_queue",
    }


def _has_worker_program_evidence(kernel: KernelSpec) -> bool:
    """Whether assignment-invariant CTA work closes the SM80 worker plan.

    The workload-independent CTA-assignment probe establishes a
    work-conserving queue, but not deterministic CTA-to-SM identity.  The
    conclusion therefore applies only to pinned source programs whose launched
    CTAs all have identical query-row and KV-step work.
    """

    if not (
            _is_source_bound_fa2_pipeline(kernel)
            or _is_source_bound_grouped_mla_ldgsts(kernel)):
        return False
    geometry = _pipeline_geometry(kernel)
    if geometry is None:
        return False
    tile_q, tile_kv, _, grouped_mla = geometry
    if _work_distribution(kernel) != "direct_cta":
        return False
    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=grouped_mla,
        include_empty_splits=True)
    active_ctas = sum(count for _, count in class_counts)
    return (
        active_ctas == kernel.launch.grid_size
        and class_counts[0][0].step_count > 0
        and len(class_counts) == 1)


def _has_complete_direct_cta_program(kernel: KernelSpec) -> bool:
    """Whether every launched direct FA2 CTA has source-enumerable work.

    A zero-step split is still a complete source program: it takes the pinned
    early-exit path and stores an empty partial result.  Rejecting those CTAs
    kept short split-KV decode launches on the aggregate model even though the
    launch stream and the early-exit work are both statically enumerable.
    """

    if not (
            _is_source_bound_fa2_pipeline(kernel)
            or _is_source_bound_grouped_mla_ldgsts(kernel)):
        return False
    geometry = _pipeline_geometry(kernel)
    if geometry is None:
        return False
    tile_q, tile_kv, _, grouped_mla = geometry
    if _work_distribution(kernel) != "direct_cta":
        return False
    if (_is_source_bound_fa2_pipeline(kernel)
            and kernel.attributes["kv_layout"] in (
                "chunked_prefill", "paged_varlen")
            and kernel.variant.lower() == "prefill"):
        if _splits(kernel) != 1:
            return False
        batch, _, _, _, query_heads, _ = _shape(kernel)
        q_lens, _ = _request_lengths(kernel)
        grid_x, grid_y, grid_z = kernel.launch.grid
        return (
            grid_y == batch
            and grid_z == query_heads
            and grid_x == math.ceil(max(q_lens) / tile_q)
        )
    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=grouped_mla,
        include_empty_splits=True)
    return (
        sum(count for _, count in class_counts) == kernel.launch.grid_size
        and (not grouped_mla or all(
            item.step_count > 0 for item, _ in class_counts))
    )


def _metax_pipeline_step_counts(
        kernel: KernelSpec, tile_q: int, tile_kv: int):
    """Expand logical step counts only for the bounded MetaX programs.

    The SM80 production path must consume ``_pipeline_cta_class_counts``
    directly: expanding this tuple for A800 would make Forward cost scale with
    the launch grid instead of the number of analytical work classes.
    """

    if (_is_source_bound_fa2_pipeline(kernel)
            or _is_source_bound_grouped_mla_ldgsts(kernel)):
        raise ValueError(
            "A800 attention must consume analytical CTA-class counts")
    class_counts, head_groups = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=(
            _is_source_bound_metax_tile16_program(kernel)
            and _splits(kernel) > 1),
    )
    return tuple(
        item.step_count
        for item, count in class_counts
        for _ in range(count // head_groups)
    ), head_groups


def _pipeline_cta_class_counts(
        kernel: KernelSpec, tile_q: int, tile_kv: int,
        grouped_mla: bool, *, include_empty_splits=False):
    """Return analytical CTA classes and counts without a CTA stream.

    Requests with identical ``(q_len, kv_len)`` have the same source program.
    Build that program once, then multiply its class counts by the number of
    matching requests and by the source-defined head-group multiplicity.  The
    result therefore scales with distinct request shapes and CTA classes, not
    with the physical launch grid.
    """

    if grouped_mla:
        query_tile = 1
        grouped = _grouped_mla_tile(kernel)
        if grouped is None:
            raise AssertionError("grouped MLA pipeline lost its grouped tile")
        head_groups = math.ceil(
            int(kernel.shape["num_query_heads"]) / grouped[0])
    else:
        query_tile = tile_q
        _, _, _, _, query_heads, kv_heads = _shape(kernel)
        tiling = kernel.launch.tiling
        # A one-token prefill still uses the prefill launch's head axis.
        # KernelSpec binds variant to launch.execution_variant; query length
        # alone cannot distinguish it from grouped decode.
        is_prefill = kernel.variant.lower() == "prefill"
        if (is_prefill and tiling is not None
                and "query_heads_per_workgroup" in tiling.tile_shape):
            query_heads_per_workgroup = int(
                tiling.tile_shape["query_heads_per_workgroup"])
            head_groups = math.ceil(
                query_heads / query_heads_per_workgroup)
        else:
            head_groups = query_heads if is_prefill else kv_heads

    request_shape_counts = {}
    for request_shape in zip(*_request_lengths(kernel)):
        request_shape_counts[request_shape] = (
            request_shape_counts[request_shape] + 1
            if request_shape in request_shape_counts else 1)

    configured_steps = kernel.attributes["fa_ksteps"]
    split_count = _splits(kernel)
    causal = kernel.attributes["causal"]
    if not isinstance(causal, bool):
        raise TypeError("FLASH_ATTN causal must be boolean")
    if (_is_source_bound_fa2_pipeline(kernel)
            or _is_source_bound_grouped_mla_ldgsts(kernel)):
        compact_counts = {}

        def _add(query_rows, step_count, count):
            if count <= 0:
                return
            item = _AttentionCTAWork(
                query_rows=query_rows, step_count=step_count)
            compact_counts[item] = (
                compact_counts[item] + count
                if item in compact_counts else count)

        def _add_moments(query_rows, cta_count, total_steps, multiplicity):
            if cta_count <= 0:
                return
            base_steps, higher_count = divmod(total_steps, cta_count)
            lower_count = cta_count - higher_count
            _add(query_rows, base_steps, lower_count * multiplicity)
            _add(query_rows, base_steps + 1, higher_count * multiplicity)

        for (query, key_value), request_count in request_shape_counts.items():
            if query > key_value:
                raise ValueError(
                    "FLASH_ATTN bottom-right workload requires q_len <= "
                    "kv_len")
            full_tiles, tail_rows = divmod(query, query_tile)
            multiplicity = request_count * head_groups
            if configured_steps is not None:
                _add(
                    query_tile, int(configured_steps),
                    full_tiles * split_count * multiplicity)
                _add(
                    tail_rows, int(configured_steps),
                    int(tail_rows > 0) * split_count * multiplicity)
                continue

            if split_count == 1:
                if causal:
                    context = key_value - query
                    full_steps = _sum_ceiled_arithmetic_progression(
                        full_tiles, context + query_tile,
                        query_tile, tile_kv)
                else:
                    full_steps = (
                        full_tiles * math.ceil(key_value / tile_kv))
                _add_moments(
                    query_tile, full_tiles, full_steps, multiplicity)
                if tail_rows:
                    _add(
                        tail_rows, math.ceil(key_value / tile_kv),
                        multiplicity)
                continue

            if causal and query > 1:
                raise ValueError(
                    "A800 causal prefill with multiple KV splits has no "
                    "constant-size analytical launch contract")
            kv_tiles = math.ceil(key_value / tile_kv)
            split_steps = _split_tile_counts(
                kv_tiles, split_count,
                include_empty=include_empty_splits)
            tile_multiplicity = full_tiles * multiplicity
            for step_count in split_steps:
                _add(query_tile, step_count, tile_multiplicity)
            if tail_rows:
                for step_count in split_steps:
                    _add(tail_rows, step_count, multiplicity)
        if not compact_counts:
            raise ValueError(
                "FLASH_ATTN pipeline produced no active CTA classes")
        return tuple(compact_counts.items()), head_groups

    class_counts = {}
    for (query, key_value), request_count in request_shape_counts.items():
        if query > key_value:
            raise ValueError(
                "FLASH_ATTN bottom-right workload requires q_len <= kv_len")
        context = key_value - query
        for start in range(0, query, query_tile):
            end = min(query, start + query_tile)
            rows = end - start
            attended = (
                context + end if kernel.attributes["causal"] else key_value)
            if attended <= 0:
                raise ValueError(
                    "FLASH_ATTN real Q tile must attend positive KV work")
            kv_tiles = math.ceil(attended / tile_kv)
            counts = (
                _split_tile_counts(
                    kv_tiles, split_count,
                    include_empty=include_empty_splits)
                if configured_steps is None
                else (int(configured_steps),) * split_count)
            for step_count in counts:
                item = _AttentionCTAWork(
                    query_rows=rows,
                    step_count=step_count,
                )
                multiplicity = request_count * head_groups
                class_counts[item] = (
                    class_counts[item] + multiplicity
                    if item in class_counts else multiplicity)
    if not class_counts:
        raise ValueError("FLASH_ATTN pipeline produced no active CTA classes")
    return tuple(class_counts.items()), head_groups


def _moment_preserving_cta_classes(class_counts):
    """Collapse CTA classes while preserving envelopes and panel work.

    For one query-row extent, the pinned source body has a fixed CTA envelope
    and a linear number of repeated KV-panel iterations.  Two adjacent integer
    panel counts preserve both totals without constructing launch-order CTAs or
    worker-local CTA streams.
    """

    by_rows = {}
    for item, count in class_counts:
        if item.query_rows not in by_rows:
            by_rows[item.query_rows] = [0, 0, 0]
        bucket = by_rows[item.query_rows]
        if item.step_count == 0:
            bucket[0] += count
        else:
            bucket[1] += count
            bucket[2] += count * item.step_count

    result = []
    for query_rows, (empty_count, active_count, total_steps) in (
            by_rows.items()):
        if empty_count:
            result.append((
                _AttentionCTAWork(query_rows=query_rows, step_count=0),
                empty_count,
            ))
        if not active_count:
            continue
        base_steps, higher_count = divmod(total_steps, active_count)
        lower_count = active_count - higher_count
        if lower_count:
            result.append((
                _AttentionCTAWork(
                    query_rows=query_rows, step_count=base_steps),
                lower_count,
            ))
        if higher_count:
            result.append((
                _AttentionCTAWork(
                    query_rows=query_rows, step_count=base_steps + 1),
                higher_count,
            ))
    if sum(count for _, count in result) != sum(
            count for _, count in class_counts):
        raise AssertionError("CTA class reduction lost CTA envelopes")
    if sum(item.step_count * count for item, count in result) != sum(
            item.step_count * count for item, count in class_counts):
        raise AssertionError("CTA class reduction lost KV-panel work")
    return tuple(result)


def _source_prefix_with_balanced_queue_tail(
        source_order_classes, resident_capacity):
    """Keep the measured resident prefix and balance only the queued tail.

    The SM80 admission probe establishes exact source-order admission for the
    first two CTAs on each SM, followed by a work-conserving queue without a
    stable CTA-to-SM identity.  Preserve that finite resident cohort and
    reduce only the queued suffix to its source-work moments.
    """

    if (isinstance(resident_capacity, bool)
            or not isinstance(resident_capacity, int)
            or resident_capacity <= 0):
        raise ValueError("FA2 resident capacity must be a positive integer")
    prefix = []
    queued = []
    remaining = resident_capacity
    for item, count in source_order_classes:
        prefix_count = min(remaining, count)
        if prefix_count:
            prefix.append((item, prefix_count))
            remaining -= prefix_count
        if count > prefix_count:
            queued.append((item, count - prefix_count))
    if remaining:
        return source_order_classes
    balanced_tail = _moment_preserving_cta_classes(tuple(queued))
    result = tuple(prefix) + balanced_tail
    if sum(count for _, count in result) != sum(
            count for _, count in source_order_classes):
        raise AssertionError("FA2 work-conserving queue lost CTA envelopes")
    if sum(item.step_count * count for item, count in result) != sum(
            item.step_count * count
            for item, count in source_order_classes):
        raise AssertionError("FA2 work-conserving queue lost KV-panel work")
    return result


def _balanced_prefix_with_source_suffix(
        source_order_classes, suffix_count):
    """Balance the queued prefix and retain one exact resident drain."""

    if (isinstance(suffix_count, bool)
            or not isinstance(suffix_count, int)
            or suffix_count <= 0):
        raise ValueError("FA2 source suffix count must be a positive integer")
    total_count = sum(count for _, count in source_order_classes)
    prefix_count = total_count - suffix_count
    if prefix_count <= 0:
        raise ValueError("FA2 source suffix requires a non-empty prefix")
    prefix = []
    suffix = []
    remaining_prefix = prefix_count
    for item, count in source_order_classes:
        current_prefix = min(remaining_prefix, count)
        if current_prefix:
            prefix.append((item, current_prefix))
            remaining_prefix -= current_prefix
        if count > current_prefix:
            suffix.append((item, count - current_prefix))
    if remaining_prefix:
        raise AssertionError("FA2 source suffix split lost its prefix")
    result = _moment_preserving_cta_classes(tuple(prefix)) + tuple(suffix)
    if sum(count for _, count in result) != total_count:
        raise AssertionError("FA2 source suffix split lost CTA envelopes")
    if sum(item.step_count * count for item, count in result) != sum(
            item.step_count * count
            for item, count in source_order_classes):
        raise AssertionError("FA2 source suffix split lost KV-panel work")
    return result


def _resident_fa2_source_order_classes(
        kernel: KernelSpec, tile_q: int, tile_kv: int):
    """Return the exact bounded source order for resident causal prefill.

    SM80 launches the query-tile axis first, followed by batch and head.  In
    the domain where every CTA fits in the two statically proven resident
    slots, retaining this short source period preserves which panel depths
    share one round-robin execution lane.  Larger and padded launches remain
    on the constant-size moment envelope.
    """

    batch, q_len, _, _, query_heads, _ = _shape(kernel)
    if (not _is_source_bound_fa2_pipeline(kernel)
            or q_len <= 1
            or not kernel.attributes["causal"]
            or _splits(kernel) != 1):
        return None
    q_lens, kv_lens = _request_lengths(kernel)
    query_tiles = math.ceil(q_len / tile_q)
    if (kernel.launch.grid != (query_tiles, batch, query_heads)
            or any(math.ceil(length / tile_q) != query_tiles
                   for length in q_lens)):
        return None
    configured_steps = kernel.attributes["fa_ksteps"]
    ordered = []
    for _ in range(query_heads):
        for query, key_value in zip(q_lens, kv_lens):
            context = key_value - query
            for query_tile in range(query_tiles):
                start = query_tile * tile_q
                end = min(query, start + tile_q)
                item = _AttentionCTAWork(
                    query_rows=end - start,
                    step_count=(
                        int(configured_steps)
                        if configured_steps is not None
                        else math.ceil((context + end) / tile_kv)),
                )
                if ordered and ordered[-1][0] == item:
                    previous, count = ordered[-1]
                    ordered[-1] = (previous, count + 1)
                else:
                    ordered.append((item, 1))
    return tuple(ordered)


def _fa2_varlen_single_tile_source_order_classes(
        kernel: KernelSpec, tile_q: int, tile_kv: int):
    """Return run-length CTA classes in the pinned varlen single-tile grid order.

    The pinned SM80 SASS body predicates its active query rows; a CTA whose
    request fits in one query tile issues the same fixed-length causal K/V
    loop whether that request is a decode step (q_len==1) or a short prefill
    continuation (1 < q_len <= tile_q) mixed into the same batch, because its
    causal window always ends at its own KV_LEN (context + end == KV_LEN
    whenever end == query == min(query, tile_q)).  Only ``query_rows`` -- the
    store/epilogue byte count -- and the launch's head-group multiplicity (via
    ``_pipeline_cta_class_counts``, unchanged) differ per request.  Requests
    needing more than one query tile fall through to the caller's other
    source-order builders.
    """

    batch, _, _, _, _, _ = _shape(kernel)
    if (not _is_source_bound_fa2_pipeline(kernel)
            or kernel.attributes["kv_layout"] != "paged_varlen"):
        return None
    q_lens, kv_lens = _request_lengths(kernel)
    if any(math.ceil(query / tile_q) != 1 for query in q_lens):
        return None
    splits = _splits(kernel)
    if splits > 1 and any(query != 1 for query in q_lens):
        # The split-KV combine reducer and its source-ordered K/V loop are
        # only pinned for one query row per request; a queued split launch
        # mixing longer requests has no independent source/microbench
        # evidence yet.
        return None
    _, head_groups = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=True)
    expected_grid = (
        (1, batch, head_groups)
        if splits == 1 else (1, splits, batch * head_groups))
    if kernel.launch.grid != expected_grid:
        raise ValueError(
            "FA2 varlen single-tile grid disagrees with its source order")
    configured_steps = kernel.attributes["fa_ksteps"]
    ordered = []

    def append_item(query_rows, step_count):
        item = _AttentionCTAWork(query_rows=query_rows, step_count=step_count)
        if ordered and ordered[-1][0] == item:
            previous, count = ordered[-1]
            ordered[-1] = (previous, count + 1)
        else:
            ordered.append((item, 1))

    if splits == 1:
        # CUDA linearizes (q_tile, batch, head_group), so batch is the
        # adjacent axis in the unsplit launch.
        for _ in range(head_groups):
            for query, key_value in zip(q_lens, kv_lens):
                append_item(
                    query,
                    int(configured_steps)
                    if configured_steps is not None
                    else math.ceil(key_value / tile_kv))
    else:
        # The split launch is (q_tile, split, batch*head_group).  Its split
        # axis is adjacent, and every request derives its own source KV range
        # before advancing to the next flattened batch/head coordinate.
        for key_value in kv_lens:
            step_counts = (
                (int(configured_steps),) * splits
                if configured_steps is not None
                else _split_tile_counts(
                    math.ceil(key_value / tile_kv), splits,
                    include_empty=True))
            for _ in range(head_groups):
                for step_count in step_counts:
                    append_item(1, step_count)
    return tuple(ordered)


def _fa2_chunked_prefill_source_order_period(
        kernel: KernelSpec, tile_q: int, tile_kv: int):
    """Return one head-group CTA period and its exact source repeat count.

    CUDA linearizes this launch as ``(q_tile, batch, head)``.  The source
    still launches the maximum query-tile extent for every request, but a CTA
    whose tile starts beyond its request-local query length exits before the
    attention body.  Keep the order of the active source programs without
    assigning any unproved duration to those early-exit CTAs.
    """

    batch, q_len, _, _, _, _ = _shape(kernel)
    if (not _is_source_bound_fa2_pipeline(kernel)
            or kernel.attributes["kv_layout"] not in (
                "chunked_prefill", "paged_varlen")
            or q_len <= 1
            or _splits(kernel) != 1):
        return None
    q_lens, kv_lens = _request_lengths(kernel)
    query_tiles = math.ceil(q_len / tile_q)
    _, head_groups = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=True)
    if kernel.launch.grid != (query_tiles, batch, head_groups):
        raise ValueError(
            "FA2 varlen prefill grid disagrees with its source order")
    configured_steps = kernel.attributes["fa_ksteps"]
    causal = kernel.attributes["causal"]
    if not isinstance(causal, bool):
        raise TypeError("FLASH_ATTN causal must be boolean")
    ordered = []
    for query, key_value in zip(q_lens, kv_lens):
        context = key_value - query
        for query_tile in range(math.ceil(query / tile_q)):
            start = query_tile * tile_q
            end = min(query, start + tile_q)
            attended = context + end if causal else key_value
            item = _AttentionCTAWork(
                query_rows=end - start,
                step_count=(
                    int(configured_steps)
                    if configured_steps is not None
                    else math.ceil(attended / tile_kv)),
            )
            if ordered and ordered[-1][0] == item:
                previous, count = ordered[-1]
                ordered[-1] = (previous, count + 1)
            else:
                ordered.append((item, 1))
    return tuple(ordered), head_groups


def _has_a800_fa2_paired_fifo_admission(
        kernel: KernelSpec, hardware, tile_q: int, tile_kv: int,
        *, class_counts=None, head_groups=None) -> bool:
    """Recognize the source/binary domain covered by the CUDA queue probe."""

    common = (
            hardware.num_compute_units == 108
            and hardware.register_reduction_dependency_cycles is not None
            and kernel.variant.lower() == "prefill"
            and kernel.attributes["causal"] is True
            and _splits(kernel) == 1
            and int(kernel.shape["head_dim"]) == 128
            and kernel.launch.block == (128, 1, 1))
    paged_program = (
        kernel.launch.algorithm == "vllm_flash_attn_splitkv"
        and kernel.attributes["kv_layout"] == "paged"
        and (tile_q, tile_kv) == (64, 128)
        and _attention_tile(kernel) == (64, 128)
        and kernel.launch.parameters[
            "page_table_bytes_per_kv_panel"] == 0.0)
    dense_grid256_program = (
        # Exact eight-class program pairing and FIFO suffix are bound by
        # fa2_dense_q128_kv64_program_admission_sm80.json (SHA256
        # 8161b39ed5f2c9e68d142d4d597a34b7707c667831f2be091317918392a82a89).
        _is_source_bound_fa2_dense_sass_program(kernel)
        and kernel.attributes["kv_layout"] == "contiguous"
        and (tile_q, tile_kv) == (128, 64)
        and _attention_tile(kernel) == (128, 64)
        and kernel.launch.grid_size == 256)
    if not common or not (paged_program or dense_grid256_program):
        return False
    if class_counts is None or head_groups is None:
        class_counts, head_groups = _pipeline_cta_class_counts(
            kernel, tile_q, tile_kv, grouped_mla=False,
            include_empty_splits=True)
    active_ctas = sum(count for _, count in class_counts)
    _, _, _, _, query_heads, _ = _shape(kernel)
    resident_capacity = (
        hardware.num_compute_units * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS)
    return (
        resident_capacity < active_ctas <= _A800_FA2_PAIRED_FIFO_MAX_CTAS
        and active_ctas == kernel.launch.grid_size
        and query_heads == head_groups
        and all(
            item.query_rows == tile_q and item.step_count > 0
            for item, _ in class_counts)
        and _resident_fa2_source_order_classes(
            kernel, tile_q, tile_kv) is not None)


def _has_a800_fa2_dense_bounded_fifo_drain(
        kernel: KernelSpec, hardware, tile_q: int, tile_kv: int,
        *, class_counts=None) -> bool:
    """Recognize the exact dense resource class covered by the CUDA probe."""

    if not (
            hardware.num_compute_units == 108
            and hardware.register_reduction_dependency_cycles is not None
            and _is_source_bound_fa2_dense_sass_program(kernel)
            and kernel.variant.lower() == "prefill"
            and kernel.attributes["causal"] is True
            and _splits(kernel) == 1
            and int(kernel.shape["head_dim"]) == 128
            and (tile_q, tile_kv) == (128, 64)
            and _attention_tile(kernel) == (128, 64)
            and kernel.launch.block == (128, 1, 1)
            and kernel.launch.grid_size == 512):
        return False
    if class_counts is None:
        class_counts, _ = _pipeline_cta_class_counts(
            kernel, tile_q, tile_kv, grouped_mla=False,
            include_empty_splits=True)
    return (
        sum(count for _, count in class_counts) == 512
        and all(item.query_rows == tile_q and item.step_count > 0
                for item, _ in class_counts)
        and _resident_fa2_source_order_classes(
            kernel, tile_q, tile_kv) is not None)


def _has_a800_fa2_split3_varlen_admission(
        kernel: KernelSpec, hardware, tile_q: int, tile_kv: int) -> bool:
    """Recognize the exact split-3 grid/resource admission observation."""

    # The resident topology and near-FIFO queued suffix are bound by
    # fa2_split3_varlen_program_admission_sm80.json (SHA256
    # 4f869920c02d46ac4f10fbeada4edea57cfc837209fc8d08018236a1d14d1e2d).
    return (
        hardware.num_compute_units == 108
        and hardware.register_reduction_dependency_cycles is not None
        and _is_source_bound_fa2_pipeline(kernel)
        and kernel.launch.algorithm == "vllm_flash_attn_splitkv"
        and kernel.variant.lower() == "decode"
        and kernel.attributes["kv_layout"] == "paged_varlen"
        and int(kernel.shape["q_len"]) == 1
        and int(kernel.shape["head_dim"]) == 128
        and _splits(kernel) == 3
        and (tile_q, tile_kv) == (64, 128)
        and _attention_tile(kernel) == (64, 128)
        and kernel.launch.grid == (1, 3, 128)
        and kernel.launch.block == (128, 1, 1)
    )


def _a800_split3_varlen_admission_order(source_order_classes):
    """Apply the measured resident-pair order and retain FIFO source tail.

    The exact 128-thread, 81920-byte SM80 admission observation records all
    first 216 CTAs on the 108 two-slot SMs.  CTAs 0..165 form adjacent pairs;
    the remaining resident cohort pairs 166..190 with 191..215.  The queued
    suffix remains in source order.  This is a discrete scheduler observation,
    not a duration-derived permutation.
    """

    expanded = tuple(
        item for item, count in source_order_classes for _ in range(count))
    if len(expanded) != 384:
        raise ValueError(
            "split-3 varlen admission requires exactly 384 source CTAs")
    resident_order = tuple(range(166)) + tuple(
        ordinal
        for lower in range(166, 191)
        for ordinal in (lower, lower + 25))
    order = resident_order + tuple(range(216, 384))
    if len(order) != len(expanded) or set(order) != set(range(len(expanded))):
        raise AssertionError("split-3 varlen admission lost source CTAs")
    reordered = []
    for ordinal in order:
        item = expanded[ordinal]
        if reordered and reordered[-1][0] == item:
            previous, count = reordered[-1]
            reordered[-1] = (previous, count + 1)
        else:
            reordered.append((item, 1))
    return tuple(reordered)


def _compact_exact_source_period(schedule):
    """Return one exact source period and its repeat count."""

    if not isinstance(schedule, (list, tuple)) or not schedule:
        raise TypeError("source schedule must be a non-empty sequence")
    schedule = tuple(schedule)
    prefix = [0] * len(schedule)
    for index in range(1, len(schedule)):
        length = prefix[index - 1]
        while length and schedule[index] != schedule[length]:
            length = prefix[length - 1]
        if schedule[index] == schedule[length]:
            length += 1
        prefix[index] = length
    period = len(schedule) - prefix[-1]
    if len(schedule) % period:
        return schedule, 1
    return schedule[:period], len(schedule) // period


def _balanced_cta_class_worker_groups(class_counts, worker_count):
    """Distribute analytical CTA-class counts over round-robin workers.

    Only per-class integer counts are retained.  The launch-sized CTA stream
    and worker-local CTA order are intentionally absent; quotient/remainder
    arithmetic gives the balanced class ownership directly.
    """

    if not isinstance(class_counts, tuple) or not class_counts:
        raise TypeError("CTA class counts must be a non-empty tuple")
    if (isinstance(worker_count, bool)
            or not isinstance(worker_count, int)
            or worker_count <= 0):
        raise ValueError("CTA class worker count must be positive")
    counts_by_worker = [[0] * len(class_counts) for _ in range(worker_count)]
    next_worker = 0
    for class_index, (_, count) in enumerate(class_counts):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("CTA class multiplicity must be positive")
        base, remainder = divmod(count, worker_count)
        if base:
            for counts in counts_by_worker:
                counts[class_index] = base
        for offset in range(remainder):
            counts_by_worker[(next_worker + offset) % worker_count][
                class_index] += 1
        next_worker = (next_worker + remainder) % worker_count

    grouped = {}
    for counts in counts_by_worker:
        key = tuple(counts)
        grouped[key] = grouped[key] + 1 if key in grouped else 1
    return tuple(grouped.items())


def _is_source_bound_fa2_pipeline(kernel: KernelSpec) -> bool:
    """Whether the launch selects one pinned SM80 FA2 worker program."""

    return (not _has_graph_padding(kernel)
            and kernel.launch.algorithm in {
        "flash_attn_dense",
        "flash_attn_splitkv",
        "vllm_flash_attn_splitkv",
    })


def _is_source_bound_fa2_dense_sass_program(
        kernel: KernelSpec) -> bool:
    """Recognize pinned FA2 2.8.3 SM80 dense source programs."""

    if _has_graph_padding(kernel):
        return False
    batch, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    expected_tile = {64: (128, 128), 128: (128, 64)}
    common = (
        kernel.launch.algorithm == "flash_attn_dense"
        and kernel.dtype.lower() == "bf16"
        and head_dim in expected_tile
        and _attention_tile(kernel) == expected_tile[head_dim]
        and kernel.launch.block == (128, 1, 1)
        and kernel.launch.parameters["unpadded_flops"] == 0.0
        and kernel.attributes["kv_layout"] == "contiguous"
        and _splits(kernel) == 1
        and len(q_lens) == batch
        and len(kv_lens) == batch
        and all(
            length > 0 and length % expected_tile[head_dim][1] == 0
            for length in kv_lens)
        and kernel.attributes["fa_ksteps"] is None
    )
    if not common:
        return False
    if kernel.variant.lower() == "decode":
        return (
            q_len == 1
            and kernel.attributes["causal"] is False
            and all(length == 1 for length in q_lens)
            and kernel.launch.grid == (1, batch, kv_heads)
        )
    if kernel.variant.lower() == "prefill":
        # The retained causal D128 specialization has an independently
        # checked affine SASS control-flow count only for complete Q128 rows.
        return (
            head_dim == 128
            and q_len > 1
            and kernel.attributes["causal"] is True
            and all(q == k and q % 128 == 0
                    for q, k in zip(q_lens, kv_lens))
            and kernel.launch.grid == (
                math.ceil(q_len / 128), batch, query_heads)
        )
    return False


def _is_source_bound_fa2_split_d128_stage_program(
        kernel: KernelSpec) -> bool:
    """Recognize the counter-closed SM80 D128 split-decode program."""

    if _has_graph_padding(kernel):
        return False
    batch, q_len, _, head_dim, _, _ = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    return (
        kernel.launch.algorithm == "flash_attn_splitkv"
        and kernel.variant.lower() == "decode"
        and kernel.dtype.lower() == "bf16"
        and head_dim == 128
        and _attention_tile(kernel) == (64, 128)
        and kernel.launch.block == (128, 1, 1)
        and kernel.launch.parameters["unpadded_flops"] == 0.0
        and kernel.attributes["causal"] is False
        and kernel.attributes["kv_layout"] == "contiguous"
        and _splits(kernel) > 1
        and len(q_lens) == batch
        and len(kv_lens) == batch
        and q_len == 1
        and all(length == 1 for length in q_lens)
        and all(length > 0 for length in kv_lens)
        and kernel.attributes["fa_ksteps"] is None
    )


def _has_a800_fa2_source_task_outer_envelope(kernel: KernelSpec, hardware) -> bool:
    """Whether a pinned D128/split2 FA2 program has the task outer bound."""

    model = hardware.fa2_source_task_outer_envelope
    if model is None:
        return False
    if kernel.launch.block_size != model["threads_per_cta"]:
        return False
    if not (model["minimum_grid_size"] <= kernel.launch.grid_size
            <= model["maximum_grid_size"]):
        return False
    if _is_source_bound_fa2_split_d128_stage_program(kernel):
        return _splits(kernel) == 2
    if not _is_source_bound_fa2_combine(kernel):
        return False
    _, _, _, head_dim, _, _ = _shape(kernel)
    return head_dim == 128 and _splits(kernel) == 2


def _is_source_bound_metax_tile16_program(kernel: KernelSpec) -> bool:
    """Whether this is the counter-closed MACA 3.7.2 tile16 decode program."""

    if _has_graph_padding(kernel):
        return False
    batch, q_len, _, head_dim, _, kv_heads = _shape(kernel)
    _, kv_lens = _request_lengths(kernel)
    matches_program = (
        kernel.launch.algorithm.startswith(_METAX_STATIC_TRAIT_DIRECT_PREFIX)
        and kernel.variant.lower() == "decode"
        and kernel.dtype.lower() == "bf16"
        and q_len == 1
        and head_dim in (64, 128)
        and _attention_tile(kernel) == (16, 16)
        and kernel.launch.block_size == 64
        and kernel.launch.parameters["unpadded_flops"] == 0.0
        and isinstance(kernel.attributes["causal"], bool)
        and kernel.attributes["kv_layout"] in (
            "contiguous", "paged", "paged_varlen")
    )
    if not matches_program:
        return False

    splits = _splits(kernel)
    layout = kernel.attributes["kv_layout"]
    if head_dim == 64:
        if layout != "contiguous":
            raise ValueError(
                "MetaX D64 tile16 decode requires contiguous KV layout")
        expected_grid = (
            (1, kv_heads, batch) if splits == 1
            else (batch * kv_heads, splits, 1))
    elif splits == 1:
        if layout not in ("paged", "paged_varlen"):
            raise ValueError(
                "MetaX D128 tile16 unsplit decode requires paged KV layout")
        expected_grid = (kv_heads, batch, 1)
    else:
        if layout not in ("contiguous", "paged", "paged_varlen"):
            raise ValueError(
                "MetaX D128 tile16 split decode requires a supported KV "
                "layout")
        expected_grid = (1, batch * kv_heads, splits)
    if len(kv_lens) != batch:
        raise ValueError(
            "MetaX tile16 request lengths disagree with launch batch")
    if kernel.launch.grid != expected_grid:
        raise ValueError(
            "MetaX tile16 grid axes disagree with the pinned source ABI: "
            f"launch={kernel.launch.grid}, source={expected_grid}")
    return True


def _fa2_main_sass_schedule_per_cta(
        kernel: KernelSpec, step_count: int) -> Tuple[float, float]:
    """Return pinned SM80 warp issues and fixed dependency cycles per CTA.

    The retained FA2 2.8.3 SM80 binary has four warps per CTA.  Static SASS
    control flow, checked against NCU's non-duration ``inst_executed`` counter,
    gives the affine instruction programs below.  The loop-independent prefix
    is also a serial four-cycle warp dependency chain; the tiled body remains
    priced by the existing Tensor/SFU/memory stages.  This adds no latency
    fitted from a production kernel.

    Binary SHA256:
    7c68f17400fac3f69c079842bd5304d3590e28dd5da850ff1465ff7ce86a1e93
    """

    is_fa2 = _is_source_bound_fa2_pipeline(kernel)
    is_dense = _is_source_bound_fa2_dense_sass_program(kernel)
    is_grouped_mla = _is_source_bound_grouped_mla_ldgsts(kernel)
    if ((not is_fa2 and not is_grouped_mla)
            or (_splits(kernel) <= 1 and not is_dense)):
        return 0.0, 0.0
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("FA2 SASS step count must be an integer")
    if step_count < 0:
        raise ValueError("FA2 SASS step count must be non-negative")
    if step_count == 0:
        return 716.0, 716.0

    if is_dense:
        # flash_fwd_kernel.h fixes one masking panel followed by
        # ``step_count - 1`` regular panels.  The retained SM80 cubin's CFG
        # contains the affine programs below across its four warps.  The
        # complete program is also a lower bound on a lone CTA's end-to-end
        # dependency span.
        _, _, _, head_dim, _, _ = _shape(kernel)
        if (kernel.variant.lower() == "prefill"
                and kernel.attributes["causal"] is True):
            if head_dim != 128 or step_count < 2:
                raise ValueError(
                    "causal dense FA2 SASS schedule requires complete D128 "
                    "Q128 tiles and at least two panels")
            # The causal D128 binary executes 5,304 fixed warp instructions
            # plus 3,600 per K/V panel.  A separate 28-instruction
            # request-head boundary adjustment is applied to the analytical
            # launch total below.  This static CFG count closes exactly
            # against all 31 canonical non-duration inst_executed
            # observations.
            warp_instructions = 5304.0 + 3600.0 * step_count
        elif head_dim == 64:
            warp_instructions = 3924.0 + 5200.0 * step_count
        elif head_dim == 128:
            warp_instructions = 3849.0 + 3968.0 * step_count
        else:
            raise AssertionError("dense FA2 SASS recognizer allowed bad D")
        return warp_instructions, warp_instructions

    if is_grouped_mla:
        fixed_instructions = 6576.0
        return (
            fixed_instructions + 3296.0 * step_count,
            fixed_instructions,
        )

    _, _, _, head_dim, _, _ = _shape(kernel)
    layout = kernel.attributes["kv_layout"]
    if layout == "contiguous":
        if head_dim == 128:
            fixed_instructions = 3104.0
            instructions_per_step = 3440.0
        elif head_dim == 64:
            fixed_instructions = 4084.0
            instructions_per_step = 4580.0
        else:
            raise ValueError(
                "pinned FA2 SASS schedule supports head dimensions 64/128")
        warp_instructions = (
            fixed_instructions + instructions_per_step * step_count)
    elif layout in ("paged", "paged_varlen"):
        if head_dim != 128:
            raise ValueError(
                "pinned paged FA2 SASS schedule requires head dimension 128")
        fixed_instructions = 3132.0
        if step_count == 1:
            warp_instructions = 6984.0
        else:
            warp_instructions = 3300.0 + 3852.0 * step_count
    else:
        raise ValueError(
            f"pinned split FA2 SASS schedule does not cover layout {layout!r}")
    return warp_instructions, fixed_instructions


def _grouped_mla_sass_stage_issue_per_cta(
        kernel: KernelSpec, step_count: int) -> _GroupedMLASASSStageIssue:
    """Split the pinned grouped-MLA SASS issue count by source stage.

    The frozen four-warp SM80 cubin places its recurrent program at
    ``0x33e0..0x6730``.  QK issue ends at ``0x3c60``, online-softmax issue at
    ``0x4450``, PV issue at ``0x5940``, and the next-panel LDGSTS/address
    program at ``0x6710``.  The two loop-control instructions plus the
    affine-CFG residual contribute sixteen warp issues per panel.  The
    loop-independent prefix and epilogue contain 3,320 and 3,252 warp issues,
    respectively; the four warps each execute one terminal ``EXIT`` issue.
    These non-duration counts sum exactly to the NCU-closed
    ``6576 + 3296 * panels`` program; they only place already-conserved issue
    work in the source DAG.

    Binary SHA256:
    7f2c5e8096a70832d68744d4fd574f267c9a18f7c889ad654c8039ef0ad99293
    """

    if not _is_source_bound_grouped_mla_ldgsts(kernel):
        raise ValueError(
            "grouped-MLA SASS stages require the pinned stage-1 program")
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("grouped-MLA SASS step count must be an integer")
    if step_count <= 0:
        raise ValueError(
            "grouped-MLA SASS stages require at least one source panel")
    issue = _GroupedMLASASSStageIssue(
        prologue=3320.0,
        qk=548.0,
        softmax=508.0,
        pv=1340.0,
        next_panel=900.0,
        epilogue=3252.0,
        cta_end=4.0,
    )
    expected, _ = _fa2_main_sass_schedule_per_cta(kernel, step_count)
    actual = (
        issue.prologue + issue.epilogue + issue.cta_end
        + step_count * (
            issue.qk + issue.softmax + issue.pv + issue.next_panel))
    if actual != expected:
        raise AssertionError(
            "grouped-MLA SASS stage partition does not conserve issue: "
            f"stages={actual}, aggregate={expected}")
    return issue


def _fa2_split_d128_sass_stage_issue_per_cta(
        kernel: KernelSpec, step_count: int) -> _FA2SplitD128SASSStageIssue:
    """Split the pinned D128 split-FA2 SASS issue by source stage.

    FlashAttention 2.8.3 commit ``060c9188`` and the retained SM80 cubin
    ``25a872527b9bec9c20fb79c2618b320d2da85715da15ef6b1d67b6c0b3ad32ae``
    place the recurrent regions in QK/V, softmax/PV, then next-K source
    order.  NCU SourceCounters' non-duration instruction counts close the
    exact one/two/four-panel programs at 6,544/9,984/16,864 warp issues.
    This partition only places that already-published whole-kernel demand;
    it does not add instructions or consume a production duration.
    """

    if not _is_source_bound_fa2_split_d128_stage_program(kernel):
        raise ValueError(
            "split-FA2 D128 SASS stages require the pinned source program")
    if isinstance(step_count, bool) or not isinstance(step_count, int):
        raise TypeError("split-FA2 D128 step count must be an integer")
    if step_count <= 0:
        raise ValueError(
            "split-FA2 D128 SASS stages require at least one source panel")
    issue = _FA2SplitD128SASSStageIssue(
        prologue=760.0,
        first_qk_v=2740.0,
        first_softmax_pv=2036.0,
        later_qk_v=980.0,
        later_softmax_pv=2280.0,
        next_panel=180.0,
        epilogue=1004.0,
        cta_end=4.0,
    )
    expected, _ = _fa2_main_sass_schedule_per_cta(kernel, step_count)
    actual = (
        issue.prologue + issue.first_qk_v + issue.first_softmax_pv
        + max(0, step_count - 1) * (
            issue.later_qk_v + issue.later_softmax_pv
            + issue.next_panel)
        + issue.epilogue + issue.cta_end)
    if actual != expected:
        raise AssertionError(
            "split-FA2 D128 SASS stage partition does not conserve issue: "
            f"stages={actual}, aggregate={expected}")
    return issue


def _fa2_main_sass_schedule(kernel: KernelSpec) -> Tuple[float, float]:
    """Return whole-launch SM80 issue and fixed dependency demand."""

    geometry = _pipeline_geometry(kernel)
    if geometry is None:
        return 0.0, 0.0
    tile_q, tile_kv, _, grouped_mla = geometry
    if (_splits(kernel) <= 1
            and not _is_source_bound_fa2_dense_sass_program(kernel)):
        return 0.0, 0.0
    class_counts, head_groups = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=grouped_mla,
        include_empty_splits=True)
    warp_instructions = float(math.fsum(
        count * _fa2_main_sass_schedule_per_cta(
            kernel, item.step_count)[0]
        for item, count in class_counts))
    fixed_dependency = float(math.fsum(
        count * _fa2_main_sass_schedule_per_cta(
            kernel, item.step_count)[1]
        for item, count in class_counts))
    if (kernel.launch.algorithm == "flash_attn_dense"
            and kernel.variant.lower() == "prefill"
            and kernel.attributes["causal"] is True):
        request_head_boundaries = len(_request_lengths(kernel)[0]) * head_groups
        boundary_instructions = 28.0 * request_head_boundaries
        warp_instructions -= boundary_instructions
        fixed_dependency -= boundary_instructions
    if warp_instructions <= 0 or fixed_dependency <= 0:
        raise ValueError("FA2 SASS launch schedule must remain positive")
    return warp_instructions, fixed_dependency


def _is_source_bound_grouped_mla_ldgsts(kernel: KernelSpec) -> bool:
    """Recognize the pinned vLLM Ampere grouped-MLA stage-1 loader."""

    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm != _A800_GROUPED_MLA_STAGE1_ALGORITHM):
        return False
    grouped = _grouped_mla_tile(kernel)
    batch, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    splits = _splits(kernel)
    expected_grid = (batch, math.ceil(query_heads / 16), splits)
    if not (
            kernel.dtype.lower() == "bf16"
            and kernel.variant.lower() == "decode"
            and q_len == 1
            and all(length == 1 for length in q_lens)
            and head_dim == 512
            and query_heads == 64
            and kv_heads == 1
            and kernel.attributes["causal"] is False
            and grouped == (16, 32, 576, 512)
            and kernel.launch.grid == expected_grid
            and kernel.launch.block == (128, 1, 1)):
        raise ValueError(
            "grouped-MLA LDGSTS marker disagrees with the pinned vLLM "
            "Ampere stage-1 source contract")
    return True


def _attach_gpu_ldgsts_frontend(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Bound a source-proven LDGSTS read path with independent 32-B rates."""

    if inp.mem_model != "gpu_mshr":
        raise ValueError("GPU LDGSTS frontend requires gpu_mshr memory")
    is_fa2 = _is_source_bound_fa2_pipeline(kernel)
    is_grouped_mla = _is_source_bound_grouped_mla_ldgsts(kernel)
    if not is_fa2 and not is_grouped_mla:
        return inp
    if is_fa2 and kernel.attributes["kv_layout"] == "mla":
        raise ValueError("FA2 LDGSTS source contract cannot describe MLA")
    tiling = kernel.launch.tiling
    if (is_fa2 and (tiling is None
            or not {"q", "kv"}.issubset(tiling.tile_shape))):
        raise ValueError("source-bound FA2 LDGSTS requires Q/KV tile geometry")
    single_sm = (
        hardware.ldgsts_initial_32b_transactions_single_sm_cycle)
    per_active_sm = (
        hardware.ldgsts_initial_32b_transactions_per_active_sm_cycle)
    full_chip = (
        hardware.ldgsts_initial_32b_transactions_full_chip_per_cycle)
    if single_sm is None or per_active_sm is None or full_chip is None:
        return inp
    if hardware.l2_transaction_bytes != 32:
        raise ValueError("GPU LDGSTS frontend requires 32-byte transactions")
    active_units = inp.active_units
    rate_per_active_unit_per_cycle = (
        float(single_sm) if active_units == 1
        else min(float(per_active_sm), float(full_chip) / active_units))
    rate_per_active_unit_s = (
        rate_per_active_unit_per_cycle * inp.sm_freq_hz)
    read_paths = tuple(
        path for path in inp.memory_paths if path.name == "global_read")
    if len(read_paths) != 1 or read_paths[0].access_kind != "read":
        raise ValueError("GPU LDGSTS read path must be unique")
    read_path = read_paths[0]
    finite_panel_bytes = hardware.ldgsts_finite_resident_panel_bytes
    if (finite_panel_bytes is not None
            and is_fa2
            and kernel.launch.algorithm == "vllm_flash_attn_splitkv"
            and kernel.variant.lower() == "prefill"
            # The finite probe fixes one uniform panel depth on every CTA.
            # Packed varlen launches mix request-local panel depths and remain
            # on the established aggregate frontend service until that
            # heterogeneous cohort is measured independently.
            and kernel.attributes["kv_layout"] == "paged"
            and int(kernel.shape["head_dim"]) == 128
            and _attention_tile(kernel) == (64, 128)
            and kernel.launch.block == (128, 1, 1)
            and kernel.launch.parameters[
                "page_table_bytes_per_kv_panel"] == 0.0
            and active_units == hardware.num_compute_units):
        tile_q, tile_kv = _attention_tile(kernel)
        class_counts, head_groups = _pipeline_cta_class_counts(
            kernel, tile_q, tile_kv, grouped_mla=False,
            include_empty_splits=True)
        active_ctas = sum(count for _, count in class_counts)
        _, _, _, _, query_heads, _ = _shape(kernel)
        full_query_tiles = all(
            item.query_rows == 64 and item.step_count > 0
            for item, _ in class_counts)
        if (active_ctas
                >= hardware.ldgsts_finite_resident_full_chip_ctas
                and query_heads == head_groups
                and full_query_tiles):
            transactions_per_panel = (
                finite_panel_bytes / hardware.l2_transaction_bytes)
            average_panels = math.fsum(
                count * item.step_count
                for item, count in class_counts) / active_ctas
            query_transactions_per_cta = (
                64 * int(kernel.shape["head_dim"])
                * int(kernel.shape["dtype_bytes"])
                / hardware.l2_transaction_bytes)
            expected_transactions = active_ctas * (
                query_transactions_per_cta
                + average_panels * transactions_per_panel)
            if not math.isclose(
                    read_path.initial_transactions, expected_transactions,
                    rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError(
                    "finite resident LDGSTS source bytes disagree with the "
                    "Q plus K/V panel decomposition")
            steady_minimum_grid = (
                hardware.ldgsts_queued_steady_minimum_grid_ctas)
            steady_maximum_panels = (
                hardware.ldgsts_queued_steady_maximum_panels_per_cta)
            steady_average_panels = (
                hardware.ldgsts_queued_steady_average_panels_per_cta)
            steady_full_chip_rate = (
                hardware
                .ldgsts_queued_steady_full_chip_initial_32b_transactions_per_cycle)
            steady_source_period = False
            if (steady_maximum_panels is not None
                    and tile_kv % tile_q == 0):
                steady_expected_period = tuple(
                    panel_depth
                    for panel_depth in range(
                        1, int(steady_maximum_panels) + 1)
                    for _ in range(tile_kv // tile_q))
                steady_source_period = (
                    all(
                        query == key_value
                        == int(steady_maximum_panels) * tile_kv
                        and query % tile_q == 0
                        and tuple(
                            math.ceil(end / tile_kv)
                            for end in range(tile_q, query + 1, tile_q)
                        ) == steady_expected_period
                        for query, key_value
                        in zip(*_request_lengths(kernel))))
            if (steady_minimum_grid is not None
                    and steady_average_panels is not None
                    and steady_full_chip_rate is not None
                    and active_ctas >= steady_minimum_grid
                    and kernel.attributes["causal"] is True
                    and steady_source_period
                    and math.isclose(
                        average_panels, steady_average_panels,
                        rel_tol=0.0, abs_tol=1e-12)):
                finite_panel_cycles = (
                    hardware.ldgsts_finite_resident_full_chip_ctas
                    * transactions_per_panel * average_panels
                    / steady_full_chip_rate)
            elif math.isclose(
                    average_panels, 1.0, rel_tol=0.0, abs_tol=1e-12):
                finite_panel_cycles = (
                    hardware.ldgsts_finite_resident_single_panel_cycles)
            elif (2.0 <= average_panels
                    <= hardware.ldgsts_finite_resident_maximum_panels):
                finite_panel_cycles = (
                    hardware.ldgsts_finite_resident_affine_startup_cycles
                    + average_panels
                    * hardware.ldgsts_finite_resident_affine_panel_cycles)
            else:
                finite_panel_cycles = None
            if finite_panel_cycles is not None:
                # The finite probe isolates only the recurrent 64-KiB K/V
                # rounds.  FA2 stages its one 16-KiB Q tile separately before
                # that loop, so retain the independently measured saturated
                # frontend service for Q instead of treating it as a
                # fractional recurrent panel.
                query_cycles = (
                    hardware.ldgsts_finite_resident_full_chip_ctas
                    * query_transactions_per_cta / float(full_chip))
                finite_cycles = finite_panel_cycles + query_cycles
                full_chip_rate = (
                    hardware.ldgsts_finite_resident_full_chip_ctas
                    * (query_transactions_per_cta
                       + transactions_per_panel * average_panels)
                    / finite_cycles)
                rate_per_active_unit_s = (
                    full_chip_rate / active_units * inp.sm_freq_hz)
    bounded_read = dataclasses.replace(
        read_path,
        source=dataclasses.replace(
            read_path.source,
            frontend_rate_per_unit_s=rate_per_active_unit_s,
        ),
    )
    paths = tuple(
        bounded_read if path is read_path else path
        for path in inp.memory_paths)
    limitations = tuple(
        limitation for limitation in inp.limitations
        if limitation != "request_frontend_capacity_evidence_unavailable")
    return dataclasses.replace(
        inp, memory_paths=tuple(paths), limitations=limitations)


def _a800_fa2_paged_period32_ldgsts_signature():
    """Return the independent two-resident K/V-round service signature."""

    return MemoryAccessClassSignature(
        access_class=_A800_FA2_PAGED_PERIOD32_LDGSTS_SERVICE,
        dtype="bf16",
        tile_m=64,
        tile_k=128,
        logical_row_bytes=65_536,
    )


def _has_a800_fa2_paged_period32_ldgsts_domain(kernel: KernelSpec,
                                                hardware) -> bool:
    """Whether one FA2 launch has the captured Q64/KV128 source period.

    The independent probe covers only the K/V rounds: every request-head has
    the exact causal period ``(1, 1, ..., 32, 32)`` at period offset zero.
    Q stays on the ordinary path because the probe deliberately did not price
    its separate 16-KiB prologue load.
    """

    if not hardware.has_memory_access_class_service(
            _A800_FA2_PAGED_PERIOD32_LDGSTS_SERVICE):
        return False
    if not (
            kernel.launch.algorithm == "vllm_flash_attn_splitkv"
            and kernel.variant.lower() == "prefill"
            and kernel.dtype.lower() == "bf16"
            and kernel.attributes["causal"] is True
            and kernel.attributes["kv_layout"] == "paged"
            and _splits(kernel) == 1
            and _attention_tile(kernel) == (64, 128)
            and kernel.launch.block == (128, 1, 1)
            and kernel.launch.parameters["page_table_bytes_per_kv_panel"]
            == 0.0):
        return False
    _, q_len, kv_len, head_dim, _, _ = _shape(kernel)
    if (q_len, kv_len, head_dim) != (4096, 4096, 128):
        return False
    q_lens, kv_lens = _request_lengths(kernel)
    if not all(query == key_value == 4096
               for query, key_value in zip(q_lens, kv_lens)):
        return False
    return kernel.launch.grid_size >= 2048


def _attach_a800_fa2_paged_period32_kv_service(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Split the captured K/V round stream from FA2's Q ingress path.

    The period-32 probe proves a full-chip source service for complete 64-KiB
    K/V LDGSTS rounds only. Q is compulsory traffic with a different 16-KiB
    prologue geometry, so it remains a separate generic path. This preserves
    the L2 and DRAM source-byte ledgers instead of applying the K/V rate to a
    whole Attention read path.
    """

    if not _has_a800_fa2_paged_period32_ldgsts_domain(kernel, hardware):
        return inp
    signature = _a800_fa2_paged_period32_ldgsts_signature()
    if not hardware.has_memory_access_class_service(
            _A800_FA2_PAGED_PERIOD32_LDGSTS_SERVICE):
        return inp
    service = hardware.memory_access_class_service(
        _A800_FA2_PAGED_PERIOD32_LDGSTS_SERVICE,
        "read",
        "l2_cache",
        signature,
    )
    paths_by_name = {path.name: path for path in inp.memory_paths}
    if set(paths_by_name) != {"global_read", "global_write"}:
        raise ValueError("FA2 period-32 K/V split requires global read/write paths")
    read_path = paths_by_name["global_read"]
    if read_path.access_class_demands:
        raise ValueError("FA2 period-32 K/V path is already typed")
    nodes_by_role = {node.resource_role: node for node in read_path.nodes}
    if set(nodes_by_role) != {"l2_cache", "dram"}:
        raise ValueError("FA2 period-32 K/V path requires L2 and DRAM nodes")
    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    q_lens, kv_lens = _request_lengths(kernel)
    q_bytes = float(sum(q_lens) * query_heads * head_dim * element_bytes)
    kv_dram_bytes = float(
        sum(kv_lens) * kv_heads * head_dim * element_bytes * 2)
    l2_bytes = read_path.nodes[0].bytes
    dram_bytes = nodes_by_role["dram"].bytes
    kv_l2_bytes = l2_bytes - q_bytes
    if kv_l2_bytes <= 0:
        raise ValueError("FA2 period-32 K/V source stream must be positive")
    if not math.isclose(dram_bytes, q_bytes + kv_dram_bytes,
                        rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "FA2 period-32 Q/KV compulsory traffic disagrees with the "
            "structured DRAM path")

    def _nodes(l2_path_bytes, dram_path_bytes):
        result = []
        for node in read_path.nodes:
            byte_count = (
                l2_path_bytes if node.resource_role == "l2_cache"
                else dram_path_bytes)
            if node.transaction_bytes is None:
                raise ValueError("FA2 period-32 path requires transactions")
            result.append(dataclasses.replace(
                node,
                transactions=byte_count / node.transaction_bytes,
                bytes=byte_count,
            ))
        return tuple(result)

    l2_transaction_bytes = nodes_by_role["l2_cache"].transaction_bytes
    if l2_transaction_bytes is None:
        raise ValueError("FA2 period-32 L2 path requires transactions")
    q_path = dataclasses.replace(
        read_path,
        nodes=_nodes(q_bytes, q_bytes),
        initial_transactions=q_bytes / l2_transaction_bytes,
    )
    kv_path = MemoryPathDemand(
        name="global_kv_read",
        access_kind="read",
        initial_transactions=kv_l2_bytes / l2_transaction_bytes,
        nodes=_nodes(kv_l2_bytes, kv_dram_bytes),
        source=read_path.source,
        queues=read_path.queues,
        access_class_demands=(MemoryAccessClassDemand(
            service=service.service,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=kv_l2_bytes,
            signature=service.signature,
            replaces_ingress_path_service=True,
        ),),
    )
    paths = tuple(
        item
        for path in inp.memory_paths
        for item in ((q_path, kv_path) if path is read_path else (path,))
    )
    return dataclasses.replace(inp, memory_paths=paths)


def _a800_fa2_chunked_q8192_s4096_ldgsts_signature():
    """Return the exact two-resident chunked K/V service signature."""

    return MemoryAccessClassSignature(
        access_class=_A800_FA2_CHUNKED_Q8192_S4096_LDGSTS_SERVICE,
        dtype="bf16",
        tile_m=64,
        tile_k=128,
        logical_row_bytes=65_536,
    )


def _has_a800_fa2_chunked_q8192_s4096_ldgsts_domain(
        kernel: KernelSpec, hardware) -> bool:
    """Whether a launch has one measured chunked C1 source program."""

    if not hardware.has_memory_access_class_service(
            _A800_FA2_CHUNKED_Q8192_S4096_LDGSTS_SERVICE):
        return False
    if not (
            kernel.launch.algorithm == "vllm_flash_attn_splitkv"
            and kernel.variant.lower() == "prefill"
            and kernel.dtype.lower() == "bf16"
            and kernel.attributes["causal"] is True
            and kernel.attributes["kv_layout"] == "chunked_prefill"
            and _splits(kernel) == 1
            and _attention_tile(kernel) == (64, 128)
            and kernel.launch.block == (128, 1, 1)
            and kernel.launch.grid_size == 4096
            and kernel.launch.parameters["page_table_bytes_per_kv_panel"]
            == 0.0):
        return False
    batch, q_len, kv_len, head_dim, query_heads, kv_heads = _shape(kernel)
    if (head_dim, query_heads, kv_heads) != (128, 32, 8):
        return False
    q_lens, kv_lens = _request_lengths(kernel)
    skew = ((1024, 4096), (1024, 4096))
    uniform = (
        (256, 365, 474, 583, 692, 801, 910, 1019),
        (1024, 1462, 1900, 2338, 2776, 3214, 3652, 4090),
    )
    if (q_lens, kv_lens) == skew:
        return (batch, q_len, kv_len) == (2, 4096, 4096)
    if (q_lens, kv_lens) == uniform:
        return (batch, q_len, kv_len) == (8, 1019, 4090)
    return False


def _attach_a800_fa2_chunked_q8192_s4096_kv_service(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Replace only the exact chunked source K/V ingress service."""

    if not _has_a800_fa2_chunked_q8192_s4096_ldgsts_domain(kernel, hardware):
        return inp
    signature = _a800_fa2_chunked_q8192_s4096_ldgsts_signature()
    service = hardware.memory_access_class_service(
        _A800_FA2_CHUNKED_Q8192_S4096_LDGSTS_SERVICE,
        "read",
        "l2_cache",
        signature,
    )
    paths_by_name = {path.name: path for path in inp.memory_paths}
    if set(paths_by_name) != {"global_read", "global_write"}:
        raise ValueError("FA2 chunked K/V split requires global read/write paths")
    read_path = paths_by_name["global_read"]
    if read_path.access_class_demands:
        raise ValueError("FA2 chunked K/V path is already typed")
    nodes_by_role = {node.resource_role: node for node in read_path.nodes}
    if set(nodes_by_role) != {"l2_cache", "dram"}:
        raise ValueError("FA2 chunked K/V path requires L2 and DRAM nodes")
    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    q_lens, kv_lens = _request_lengths(kernel)
    q_bytes = float(sum(q_lens) * query_heads * head_dim * element_bytes)
    kv_dram_bytes = float(
        sum(kv_lens) * kv_heads * head_dim * element_bytes * 2)
    l2_bytes = read_path.nodes[0].bytes
    dram_bytes = nodes_by_role["dram"].bytes
    kv_l2_bytes = l2_bytes - q_bytes
    if kv_l2_bytes <= 0:
        raise ValueError("FA2 chunked K/V source stream must be positive")
    if not math.isclose(dram_bytes, q_bytes + kv_dram_bytes,
                        rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "FA2 chunked Q/KV compulsory traffic disagrees with the "
            "structured DRAM path")

    def _nodes(l2_path_bytes, dram_path_bytes):
        result = []
        for node in read_path.nodes:
            byte_count = (
                l2_path_bytes if node.resource_role == "l2_cache"
                else dram_path_bytes)
            if node.transaction_bytes is None:
                raise ValueError("FA2 chunked path requires transactions")
            result.append(dataclasses.replace(
                node,
                transactions=byte_count / node.transaction_bytes,
                bytes=byte_count,
            ))
        return tuple(result)

    l2_transaction_bytes = nodes_by_role["l2_cache"].transaction_bytes
    if l2_transaction_bytes is None:
        raise ValueError("FA2 chunked L2 path requires transactions")
    q_path = dataclasses.replace(
        read_path,
        nodes=_nodes(q_bytes, q_bytes),
        initial_transactions=q_bytes / l2_transaction_bytes,
    )
    kv_path = MemoryPathDemand(
        name="global_kv_read",
        access_kind="read",
        initial_transactions=kv_l2_bytes / l2_transaction_bytes,
        nodes=_nodes(kv_l2_bytes, kv_dram_bytes),
        source=read_path.source,
        queues=read_path.queues,
        access_class_demands=(MemoryAccessClassDemand(
            service=service.service,
            ingress_resource_role=service.ingress_resource_role,
            logical_bytes=kv_l2_bytes,
            signature=service.signature,
            replaces_ingress_path_service=True,
        ),),
    )
    paths = tuple(
        item
        for path in inp.memory_paths
        for item in ((q_path, kv_path) if path is read_path else (path,))
    )
    return dataclasses.replace(inp, memory_paths=paths)


def _scheduled_workgroup_count(kernel: KernelSpec, tile_q: int,
                               grouped_mla: bool, useful: int) -> int:
    tiling = kernel.launch.tiling
    _, q_len, _, _, query_heads, _ = _shape(kernel)
    if (not grouped_mla and kernel.variant.lower() == "prefill"
            and tiling is not None
            and "query_heads_per_workgroup" in tiling.tile_shape):
        query_heads_per_workgroup = int(
            tiling.tile_shape["query_heads_per_workgroup"])
        scheduled = flash_attention_packed_varlen_scheduled_workgroups(
            batch=int(kernel.shape["batch"]),
            q_len=q_len,
            tile_q=tile_q,
            query_heads=query_heads,
            query_heads_per_workgroup=query_heads_per_workgroup,
            splits=_splits(kernel),
        )
    else:
        scheduled = useful
    return scheduled


def _semantic_workgroup_counts(kernel: KernelSpec, tile_q: int, tile_kv: int,
                               grouped_mla: bool):
    """Return useful and padded workgroups from analytical class counts."""

    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla)
    useful = sum(count for _, count in class_counts)
    scheduled = _scheduled_workgroup_count(
        kernel, tile_q, grouped_mla, useful)
    return useful, scheduled


def _validate_pipeline_class_work(
        kernel: KernelSpec, tile_q: int, tile_kv: int,
        grouped_mla: bool) -> None:
    """Validate launch coverage from analytical CTA-class counts."""

    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla)
    active_ctas = sum(count for _, count in class_counts)
    scheduled_ctas = _scheduled_workgroup_count(
        kernel, tile_q, grouped_mla, active_ctas)
    if active_ctas > scheduled_ctas:
        raise ValueError(
            "FLASH_ATTN useful workgroup count exceeds padded schedule: "
            f"useful={active_ctas}, scheduled={scheduled_ctas}")
    distribution = _work_distribution(kernel)
    if distribution == "direct_cta":
        if active_ctas > kernel.launch.grid_size:
            raise ValueError(
                "FLASH_ATTN semantic active CTA count exceeds launch grid: "
                f"active={active_ctas}, grid={kernel.launch.grid_size}")
    elif distribution in ("packed_varlen_queue", "persistent_queue"):
        if kernel.launch.grid_size <= 0:
            raise ValueError(
                "FLASH_ATTN queue launch requires a positive worker grid")
    else:
        raise AssertionError(
            f"unhandled FLASH_ATTN work distribution: {distribution!r}")


def _softmax_elements(
        kernel: KernelSpec, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET) -> float:
    """Return the semantic number of online-softmax exponent evaluations."""

    if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET:
        split_fuse_work = _cann_split_fuse_work(kernel)
    if split_fuse_work is not None:
        return float(split_fuse_work.attention_elements)
    query_lengths, key_value_lengths = _request_lengths(kernel)
    _, _, _, _, query_heads, _ = _shape(kernel)
    # Every query head owns an independent score row and online-softmax state.
    # Decode kernels may group several query heads under one KV-head worker,
    # but that launch grouping does not reduce the semantic Vector work.
    attention_elements = math.fsum(
        (
            query * key_value
            if not kernel.attributes["causal"] else
            query * (key_value - query) + query * (query + 1) / 2
        )
        for query, key_value in zip(query_lengths, key_value_lengths)
    )
    return float(query_heads * attention_elements)


def _gpu_softmax_exp2_operations(kernel: KernelSpec) -> float:
    """Return source-issued FA2 ``exp2f`` operations.

    The fixed Dao/vLLM FA2 forward source applies the mask to a complete
    ``BlockM x BlockN`` accumulator and then calls ``scale_apply_exp2`` over
    every accumulator element, including masked Q/KV padding.  Every KV step
    after the first also rescales each complete ``BlockM`` output row with one
    ``exp2f``.  Split-KV restarts that online-softmax state in each non-empty
    split, so the number of row-rescale operations is derived from the exact
    per-split step counts rather than from a whole-sequence approximation.

    A tile carrying ``query_heads_per_workgroup`` is a logical queue tile, not
    necessarily the physical accumulator M extent.  Until that separate
    physical extent is published, retain the semantic lower bound for those
    launches instead of treating the logical Q tile as ``BlockM``.
    """

    def _padded_source_exp2(tile_q, tile_kv, grouped_mla):
        class_counts, _ = _pipeline_cta_class_counts(
            kernel, tile_q, tile_kv, grouped_mla=grouped_mla)
        return float(math.fsum(
            count * (
                item.step_count * tile_q * tile_kv
                + max(0, item.step_count - 1) * tile_q)
            for item, count in class_counts))

    tile = _attention_tile(kernel)
    grouped = _grouped_mla_tile(kernel)
    if grouped is not None:
        head_tile, tile_kv, _, _ = grouped
        return _padded_source_exp2(head_tile, tile_kv, True)
    if _is_source_bound_metax_mla_main(kernel):
        work = _metax_mla_scheduler_work(kernel)
        return float(64 * sum(
            segment.kv_blocks * METAX_MLA_KV_BLOCK
            + max(0, segment.kv_blocks - 1)
            for segments in work.segments_by_partition
            for segment in segments))
    if tile is None:
        return _softmax_elements(kernel)
    tiling = kernel.launch.tiling
    if tiling is None:
        raise AssertionError("FLASH_ATTN tile disappeared from launch")
    if _has_metax_q64_prefill_pipeline(kernel):
        return _padded_source_exp2(64, 64, False)
    if _has_metax_q64_contiguous_decode_pipeline(kernel):
        return _padded_source_exp2(64, 64, False)
    if _has_metax_q128_contiguous_decode_pipeline(kernel):
        return _padded_source_exp2(128, 64, False)
    if kernel.launch.algorithm in {
            _METAX_372_Q128_PREFILL_ALGORITHM,
            _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM,
    }:
        return _padded_source_exp2(128, 64, False)
    if _is_source_bound_metax_tile16_program(kernel):
        return _padded_source_exp2(16, 16, False)
    if (kernel.launch.algorithm.startswith(
                _METAX_STATIC_TRAIT_DIRECT_PREFIX)
            or "query_heads_per_workgroup" in tiling.tile_shape
            or kernel.launch.parameters["unpadded_flops"] == 1.0):
        return _softmax_elements(kernel)
    if kernel.launch.parameters["unpadded_flops"] != 0.0:
        raise ValueError(
            "tiled FA2 softmax requires padded source-work accounting")

    tile_q, tile_kv = tile
    return _padded_source_exp2(tile_q, tile_kv, False)


def _fa2_scalar_stage_operations(
        tile_q: int, tile_kv: int, head_dim: int,
        step_index: int) -> Tuple[float, float, float, float]:
    """Return source-level scalar work for one non-empty FA2 KV step.

    Dao FlashAttention 2.8.3 ``softmax.h`` assigns each score row to a
    four-lane reduction group.  The counts below follow the C++ loops, not a
    compiled instruction listing: lane-local max/sum, two XOR reductions per
    lane, score scale/exp, online output rescale after the first panel, and the
    FP32-to-input-type score conversion before PV.

    The tuple is ``(fp32_max, fma_flops, sfu_ops, conversion_operands)``.
    FP32 FMA demand is expressed in FLOPs, consistent with ``compute_fma``.
    Max and conversion counts remain source-level checks rather than generic
    ALU demand.  A800 prices those fragments, and the score FMA/exp2/local-sum
    chain, with independently measured source-shaped dependency services.
    """

    if min(tile_q, tile_kv, head_dim) <= 0:
        raise ValueError("FA2 scalar stage dimensions must be positive")
    if step_index < 0:
        raise ValueError("FA2 scalar stage index must be non-negative")
    score_elements = float(tile_q * tile_kv)
    lane_local_rows = float(4 * tile_q)
    if step_index == 0:
        max_operations = score_elements + lane_local_rows
        # The source-shaped score FMA -> exp2 -> lane-local sum chain is
        # priced by the independent register-dependency mechanism.  Only the
        # row-max scale remains aggregate FMA work here.
        scalar_flops = lane_local_rows
        row_rescale_sfu = 0.0
    else:
        max_operations = score_elements + 2.0 * lane_local_rows
        # The score FMA -> exp2 -> lane-local sum chain is priced separately
        # by the dependency mechanism.  The online state and output rescale
        # remain aggregate FMA/SFU work.
        scalar_flops = float(
            4.0 * lane_local_rows
            + tile_q * head_dim)
        row_rescale_sfu = lane_local_rows
    return (
        max_operations,
        scalar_flops,
        row_rescale_sfu,
        score_elements,
    )


def _fa2_scalar_epilogue_operations(
        tile_q: int, head_dim: int) -> Tuple[float, float, float]:
    """Return source-level normalize/LSE/output-conversion work per CTA."""

    if min(tile_q, head_dim) <= 0:
        raise ValueError("FA2 scalar epilogue dimensions must be positive")
    # The two XOR-add steps are priced by the register-dependency mechanism.
    # The remaining two FP32 FLOPs per thread-local row form the LSE update;
    # every logical output element is then normalized and converted.
    lane_local_rows = float(4 * tile_q)
    fma_flops = float(2.0 * lane_local_rows + tile_q * head_dim)
    sfu_operations = 2.0 * lane_local_rows  # reciprocal and logarithm
    conversion_operations = float(tile_q * head_dim)
    return fma_flops, sfu_operations, conversion_operations


def _gpu_fa2_scalar_operations(
        kernel: KernelSpec) -> Tuple[float, float]:
    """Return priceable non-MMA scalar work from the pinned FA2 source."""

    if not _is_source_bound_fa2_pipeline(kernel):
        return 0.0, 0.0
    tile = _attention_tile(kernel)
    if tile is None:
        raise ValueError("source-bound FA2 scalar work requires its tile")
    tile_q, tile_kv = tile
    _, _, _, head_dim, _, _ = _shape(kernel)
    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=True)
    fma_flops = 0.0
    sfu_operations = 0.0
    epilogue = _fa2_scalar_epilogue_operations(tile_q, head_dim)
    _, first_fma, first_sfu, _ = _fa2_scalar_stage_operations(
        tile_q, tile_kv, head_dim, 0)
    _, repeated_fma, repeated_sfu, _ = _fa2_scalar_stage_operations(
        tile_q, tile_kv, head_dim, 1)
    for item, count in class_counts:
        if item.step_count == 0:
            continue
        repeated_steps = item.step_count - 1
        item_fma_flops = first_fma + repeated_steps * repeated_fma
        item_sfu_operations = first_sfu + repeated_steps * repeated_sfu
        fma_flops += count * (item_fma_flops + epilogue[0])
        sfu_operations += count * (item_sfu_operations + epilogue[1])
    return (
        float(fma_flops),
        float(sfu_operations),
    )


def _fa2_register_dependency_mechanism(kernel: KernelSpec, hardware):
    """Return the independently measured mechanism for this source fragment."""

    model = hardware.register_reduction_dependency_cycles
    if model is None or not _is_source_bound_fa2_pipeline(kernel):
        return None
    if kernel.launch.block_size != model["threads_per_cta"]:
        return None
    tile = _attention_tile(kernel)
    if tile is None:
        return None
    tile_q, tile_kv = tile
    if tile_q % 32 or tile_kv % model["quad_reduction_width"]:
        raise ValueError("FA2 source fragment geometry is not integral")
    identity = (
        tile_q // 32,
        tile_kv // model["quad_reduction_width"],
    )
    matches = tuple(
        entry for entry in model["entries"]
        if (entry["lane_local_rows_per_thread"],
            entry["score_values_per_lane_local_row"]) == identity)
    if len(matches) > 1:
        raise ValueError("FA2 register dependency geometry is ambiguous")
    return None if not matches else matches[0]


def _fa2_local_spill_dependency_cycles_per_cta(
        kernel: KernelSpec, hardware, step_count=None) -> float:
    """Return a source-matched SM80 FA2 local-stack dependency."""

    model = hardware.fa2_local_spill_dependency_cycles
    _, q_len, _, head_dim, _, _ = _shape(kernel)
    d64_chunked = (
        kernel.launch.algorithm == "vllm_flash_attn_splitkv"
        and kernel.variant.lower() == "prefill"
        and kernel.dtype.lower() == "bf16"
        and q_len > 1
        and head_dim == 64
        and _attention_tile(kernel) == (64, 256)
        and kernel.launch.block == (128, 1, 1)
        and kernel.launch.parameters["unpadded_flops"] == 0.0
        and kernel.attributes["causal"] is True
        and kernel.attributes["kv_layout"] == "chunked_prefill"
        and _splits(kernel) == 1
    )
    if d64_chunked:
        if model is None:
            return 0.0
        if model["threads_per_cta"] != kernel.launch.block_size:
            raise ValueError(
                "FA2 local spill evidence disagrees with the source block "
                "size")
        if (model["stack_bytes_per_thread"] != 56
                or model["scalar_store_sites"] != 13
                or model["scalar_load_sites"] != 13):
            raise ValueError(
                "FA2 local spill evidence changed its SASS contract")
        return float(model["spill_round_cycles"])

    dense_model = (
        hardware.fa2_dense_d128_causal_local_spill_dependency_cycles)
    dense_d128_causal = (
        _is_source_bound_fa2_dense_sass_program(kernel)
        and kernel.variant.lower() == "prefill"
        and head_dim == 128
        and kernel.attributes["causal"] is True
        # The existing resident-envelope dependency closes launches that fit
        # in the two compiler-proven CTA slots per SM.  This isolated service
        # represents the additional queued local-stack program only once the
        # launch exceeds that resident cohort.
        and kernel.launch.grid_size > (
            hardware.num_compute_units
            * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS)
    )
    if not dense_d128_causal or dense_model is None:
        return 0.0
    if step_count is None:
        raise ValueError(
            "dense causal D128 local spill requires the CTA panel count")
    if (not isinstance(step_count, int) or isinstance(step_count, bool)
            or step_count < 2):
        raise ValueError(
            "dense causal D128 local spill requires at least two panels")
    if dense_model["threads_per_cta"] != kernel.launch.block_size:
        raise ValueError(
            "dense causal D128 spill evidence disagrees with the source "
            "block size")
    if (dense_model["stack_bytes_per_thread"] != 176
            or dense_model["scalar_store_sites"] != 51
            or dense_model["scalar_load_sites"] != 53):
        raise ValueError(
            "dense causal D128 spill evidence changed its SASS contract")
    # Static CFG inspection gives 49 LDL and 45 STL sites in the K/V loop
    # plus ten one-time sites of each kind; those counts close the canonical
    # non-duration counters.  The independently measured service is the
    # complete CTA program's single local-stack dependency round, not a
    # duration multiplier for each dynamic loop visit.
    return float(dense_model["spill_round_cycles"])


def _fa2_local_spill_dependency_cycles(
        kernel: KernelSpec, hardware) -> float:
    """Return whole-launch local-spill dependency demand in SM cycles."""

    tile = _attention_tile(kernel)
    if tile is None:
        return 0.0
    tile_q, tile_kv = tile
    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=True)
    active_cta_count = sum(count for _, count in class_counts)
    schedule_class_counts = None
    source_repeat_count = 1
    if active_cta_count <= (
            hardware.num_compute_units
            * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS):
        schedule_class_counts = _resident_fa2_source_order_classes(
            kernel, tile_q, tile_kv)
    if schedule_class_counts is None:
        schedule_class_counts = _fa2_varlen_single_tile_source_order_classes(
            kernel, tile_q, tile_kv)
    if schedule_class_counts is None:
        source_period = _fa2_chunked_prefill_source_order_period(
            kernel, tile_q, tile_kv)
        if source_period is not None:
            schedule_class_counts, source_repeat_count = source_period
    if schedule_class_counts is None:
        schedule_class_counts = _moment_preserving_cta_classes(class_counts)
    cycles_by_step_count = {}

    def _cycles_per_cta(step_count):
        if step_count not in cycles_by_step_count:
            cycles_by_step_count[step_count] = (
                _fa2_local_spill_dependency_cycles_per_cta(
                    kernel, hardware, step_count))
        return cycles_by_step_count[step_count]

    total = source_repeat_count * math.fsum(
        count * _cycles_per_cta(item.step_count)
        for item, count in schedule_class_counts
        if item.step_count > 0)
    return float(total)


def _fa2_register_dependency_cycles_per_cta(
        kernel: KernelSpec, hardware, step_count: int) -> float:
    """Return source-ordered register reduction and barrier cycles per CTA."""

    mechanism = _fa2_register_dependency_mechanism(kernel, hardware)
    if mechanism is None or step_count == 0:
        return 0.0
    if step_count < 0:
        raise ValueError("FA2 dependency step count must be non-negative")
    factors = _fa2_resident_dependency_factors(
        kernel, hardware, mechanism)

    def _factor(name):
        return 1.0 if factors is None else factors[name]

    def _panel_cycles(step_index):
        predicate, maximum, scale_exp_sum, conversion, barriers = (
            _fa2_register_dependency_panel_cycles(mechanism, step_index))
        return math.fsum((
            predicate * _factor("predicate"),
            maximum * _factor("max"),
            scale_exp_sum * _factor("scale_exp_sum"),
            conversion * _factor("conversion"),
            barriers * _factor("barrier"),
        ))
    first_panel_cycles = math.fsum(
        (_panel_cycles(0),))
    repeated_panel_cycles = math.fsum(
        (_panel_cycles(1),))
    panel_cycles = (
        first_panel_cycles + (step_count - 1) * repeated_panel_cycles)
    _, _, _, head_dim, _, _ = _shape(kernel)
    _, tile_kv = _attention_tile(kernel)
    output_conversion_cycles = (
        mechanism["fp32_to_bf16_conversion_cycles"]
        * head_dim / tile_kv)
    epilogue_cycles = (
        mechanism["fp32_quad4_sum_cycles"]
        * _factor("quad_sum")
        + mechanism["cta_barrier_cycles"]
        * _factor("barrier")
        + output_conversion_cycles
        * _factor("conversion")
    )
    return float(
        panel_cycles + epilogue_cycles)


def _fa2_resident_dependency_factors(kernel, hardware, mechanism):
    if (mechanism is None
            or hardware.num_compute_units != 108
            or mechanism["lane_local_rows_per_thread"] != 4
            or mechanism["score_values_per_lane_local_row"] != 16
            or kernel.variant.lower() != "prefill"
            or kernel.attributes["causal"] is not True
            or kernel.launch.grid_size <= 2 * hardware.num_compute_units
            or "two_resident_demand_factors" not in mechanism):
        return None
    factors = mechanism["two_resident_demand_factors"]
    if factors["resident_ctas_per_sm"] != _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS:
        raise ValueError(
            "FA2 dependency factor residency disagrees with source model")
    return {
        "barrier": factors["cta_barrier"],
        "max": factors["fp32_max_quad4"],
        "scale_exp_sum": factors["fp32_scale_exp2_sum"],
        "predicate": factors["integer_column_predicate"],
        "conversion": factors["fp32_to_bf16_conversion"],
        "quad_sum": factors["fp32_quad4_sum"],
    }


def _fa2_register_dependency_panel_cycles(mechanism, step_index: int):
    """Return source-ordered fixed-cycle mechanisms for one FA2 panel."""

    if step_index < 0:
        raise ValueError("FA2 dependency step index must be non-negative")
    score_values = mechanism["score_values_per_lane_local_row"]
    max_cycles = mechanism["fp32_max_quad4_cycles"]
    scale_exp_sum_cycles = mechanism["fp32_scale_exp2_sum_cycles"]
    if step_index == 0:
        # zero_init assigns the first score instead of comparing/adding it.
        # The measured max chain has N local max steps plus two quad shuffles;
        # the measured sum chain has N local adds.
        max_cycles *= (score_values + 1.0) / (score_values + 2.0)
        # The measured source-shaped chain initializes its sum at zero, just
        # like the first FA2 panel, so unlike the standalone add-only probe it
        # needs no first-panel correction.
    return (
        float(mechanism["integer_column_predicate_cycles"]),
        float(max_cycles),
        float(scale_exp_sum_cycles),
        float(mechanism["fp32_to_bf16_conversion_cycles"]),
        float(2.0 * mechanism["cta_barrier_cycles"]),
    )


def _fa2_combine_dependency_stages(kernel: KernelSpec, hardware):
    """Return source-ordered, independently measured combine stage cycles."""

    if not _is_source_bound_fa2_combine(kernel):
        raise ValueError("FA2 combine dependency requires its pinned source")
    model = hardware.fa2_combine_dependency_cycles
    if model is None:
        raise ValueError(
            "A source-bound FA2 combine requires dependency-cycle evidence")
    if kernel.launch.block_size != model["threads_per_cta"]:
        raise ValueError(
            "FA2 combine block size disagrees with dependency evidence")
    block_m = _fa2_combine_block_m(kernel)
    head_dim = int(kernel.shape["head_dim"])
    max_splits = _fa2_combine_max_splits(kernel)
    allreduce_width = min(128 // block_m, max_splits)
    reduction_matches = tuple(
        entry for entry in model["reduction_entries"]
        if entry["allreduce_width"] == allreduce_width)
    if len(reduction_matches) != 1:
        raise ValueError(
            "FA2 combine dependency evidence has no unique allreduce width")
    conversion_matches = tuple(
        entry for entry in model["conversion_entries"]
        if (entry["block_m"], entry["head_dim"],
            entry["output_values_per_lane"])
        == (block_m, head_dim, block_m * head_dim // 128))
    if len(conversion_matches) != 1:
        raise ValueError(
            "FA2 combine dependency evidence has no unique conversion geometry")
    reduction = reduction_matches[0]
    conversion = conversion_matches[0]
    num_splits = _splits(kernel)
    predicate_class = "full" if num_splits == max_splits else "partial"

    # The pinned FA2 source instantiates one fully unrolled program for each
    # (kBlockM, head_dim, kMaxSplits).  Runtime num_splits only selects whether
    # the source predicates are full or partial; it does not instantiate a new
    # program.  Rows such as S=9 and S=11 at kMaxSplits=16 are therefore
    # repeated mechanism measurements of the same partial control-flow class,
    # not per-S latency samples.  Aggregate every independent row in that
    # source class instead of interpolating or requiring an exact runtime S.
    fragment_matches = tuple(
        entry for entry in model["source_fragment_entries"]
        if (entry["block_m"], entry["head_dim"], entry["max_splits"])
        == (block_m, head_dim, max_splits)
        and (
            "full" if entry["num_splits"] == entry["max_splits"]
            else "partial") == predicate_class)
    if not fragment_matches:
        raise ValueError(
            "FA2 combine dependency evidence has no source fragment class "
            f"for block_m={block_m}, head_dim={head_dim}, "
            f"max_splits={max_splits}, predicate_class={predicate_class}")

    def fragment_median(field):
        return float(statistics.median(
            entry[field] for entry in fragment_matches))

    return {
        "slse_publish_transpose": fragment_median(
            "slse_publish_transpose_cycles"),
        "max_allreduce": reduction["fp32_max_allreduce_cycles"],
        "post_max_exp_local_sum": fragment_median(
            "post_max_exp_local_sum_cycles"),
        "sum_allreduce": reduction["fp32_sum_allreduce_cycles"],
        "post_sum_log": fragment_median("post_sum_log_cycles"),
        "scale_exp_smem_barrier": fragment_median(
            "scale_exp_smem_barrier_cycles"),
        "output_conversion": conversion[
            "fp32_to_bf16_output_conversion_cycles"],
    }


def _metax_fa2_combine_dependency_stages(kernel: KernelSpec, hardware):
    """Return the independently measured C500 D128 reducer stages."""

    if not _is_source_bound_metax_fa2_combine(kernel):
        raise ValueError(
            "MetaX FA2 combine dependency requires its pinned source")
    model = hardware.metax_fa2_combine_dependency_cycles
    if model is None:
        raise ValueError(
            "Source-bound MetaX FA2 combine requires dependency evidence")
    if (kernel.launch.block_size != model["threads_per_cta"]
            or int(kernel.shape["head_dim"]) != 128):
        raise ValueError(
            "MetaX FA2 combine launch disagrees with dependency evidence")
    max_splits = _metax_fa2_combine_max_splits(kernel)
    predicate_class = (
        "full" if _splits(kernel) == max_splits else "partial")
    matches = tuple(
        entry for entry in model["entries"]
        if (entry["block_m"], entry["head_dim"], entry["max_splits"],
            entry["predicate_class"])
        == (8, 128, max_splits, predicate_class))
    if len(matches) != 1:
        raise ValueError(
            "MetaX FA2 combine has no unique source dependency program")
    return matches[0]["stage_cycles"]


def _fa2_oaccum_load_use_cycles(kernel: KernelSpec, hardware):
    """Return the source-loop LDG128 operand-readiness cycles per split."""

    if not _is_source_bound_fa2_combine(kernel):
        raise ValueError("FA2 Oaccum load-use requires its pinned source")
    model = hardware.fa2_oaccum_load_use_cycles
    if model is None:
        raise ValueError(
            "A source-bound FA2 combine requires Oaccum load-use evidence")
    if kernel.launch.block_size != model["threads_per_cta"]:
        raise ValueError(
            "FA2 combine block size disagrees with Oaccum load-use evidence")
    if model["load_bytes_per_lane"] != 16:
        raise ValueError("FA2 Oaccum source requires one LDG128 per lane")
    if (model["values_per_lane"] != 4
            or model["dependent_fp32_fma_per_lane"] != 4):
        raise ValueError(
            "FA2 Oaccum load-use evidence disagrees with source consumers")
    scratch_dram_width = _split_scratch_dram_width(kernel)
    if scratch_dram_width == 4.0:
        cache_tier = "dram_cg"
    elif scratch_dram_width == 0.0:
        cache_tier = "l2_warm_cg"
    else:
        raise ValueError(
            "FA2 Oaccum scratch has no source-closed cache-tier mapping")
    matches = tuple(
        entry for entry in model["cache_entries"]
        if entry["cache_tier"] == cache_tier)
    if len(matches) != 1:
        raise ValueError(
            "FA2 Oaccum evidence has no unique cache-tier entry")
    entry = matches[0]
    splits = _splits(kernel)
    if splits <= 0:
        raise ValueError("FA2 Oaccum source loop requires positive splits")
    return tuple(
        entry["first_load_ready_cycles"] if split == 0
        else entry["additional_load_interval_cycles"]
        for split in range(splits))


def _fa2_register_dependency_cycles(kernel: KernelSpec, hardware) -> float:
    """Return whole-launch dependency demand from source CTA enumeration."""

    mechanism = _fa2_register_dependency_mechanism(kernel, hardware)
    if mechanism is None:
        return 0.0
    tile_q, tile_kv = _attention_tile(kernel)
    class_counts, _ = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=True)
    return float(math.fsum(
        count * _fa2_register_dependency_cycles_per_cta(
            kernel, hardware, item.step_count)
        for item, count in class_counts))


def _fa2_source_smem_bytes(kernel: KernelSpec) -> float:
    """Return source-issued shared-memory reads and writes for FA2 main."""

    if not _is_source_bound_fa2_pipeline(kernel):
        return 0.0
    tile_q, tile_kv = _attention_tile(kernel)
    _, _, _, head_dim, query_heads, _ = _shape(kernel)
    class_counts, head_groups = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=False,
        include_empty_splits=True)
    if query_heads % head_groups:
        raise ValueError("FA2 CTA head grouping must divide query heads")
    query_heads_per_cta = query_heads // head_groups
    shared_query_groups_per_cta = _fa2_shared_query_groups_per_cta(
        kernel, query_heads_per_cta)
    element_bytes = int(kernel.shape["dtype_bytes"])
    output_element_bytes = 2 if _splits(kernel) == 1 else 4
    q_tile_bytes = float(
        tile_q * shared_query_groups_per_cta * head_dim * element_bytes)
    kv_panel_bytes = float(
        tile_kv * head_dim * element_bytes)
    output_tile_bytes = float(
        tile_q * shared_query_groups_per_cta
        * head_dim * output_element_bytes)
    total = 0.0
    for item, count in class_counts:
        if item.step_count == 0:
            continue
        q_write_bytes = float(
            item.query_rows * query_heads_per_cta
            * head_dim * element_bytes)
        # The source stages Q once, but with Is_Q_in_regs=false each QK panel
        # rereads the complete compile-time Q tile from shared memory.  K and V
        # each contribute one shared write and one shared read per panel.  The
        # converted output likewise traverses shared memory before its
        # predicated global store.
        total += count * (
            q_write_bytes
            + item.step_count * q_tile_bytes
            + item.step_count * 2.0 * (kv_panel_bytes + kv_panel_bytes)
            + 2.0 * output_tile_bytes)
    return float(total)


def _fa2_shared_query_groups_per_cta(
        kernel: KernelSpec, query_heads_per_cta: int) -> int:
    """Return the source-visible shared-memory query-group tile width.

    The pinned paged D128 vLLM split-KV main program has a four-warp shared
    tile even when its runtime GQA group is wider.  SourceCounters for the
    exact `(64, 128, 128-thread, split=4)` program at GQA 2, 4, and 8 close
    the `STS`, `LDS`, and `LDSM` warp and active-lane counts identically;
    only its direct `STG` output lanes grow.  Keep output traffic on the
    physical GQA group, but never price that direct-register expansion again
    as a larger shared-memory tile.
    """

    if (isinstance(query_heads_per_cta, bool)
            or not isinstance(query_heads_per_cta, int)
            or query_heads_per_cta <= 0):
        raise ValueError("FA2 shared query groups per CTA must be positive")
    shared_warp_groups = kernel.launch.block_size // 32
    if shared_warp_groups <= 0:
        raise ValueError("FA2 shared query tile requires at least one warp")
    if kernel.launch.algorithm in {
            "flash_attn_dense", "flash_attn_splitkv"}:
        return min(query_heads_per_cta, shared_warp_groups)
    if kernel.launch.algorithm == "vllm_flash_attn_splitkv":
        _, q_len, _, head_dim, _, _ = _shape(kernel)
        if (
                kernel.variant.lower() == "decode"
                and kernel.dtype.lower() == "bf16"
                and q_len == 1
                and head_dim == 128
                and kernel.attributes["kv_layout"] == "paged"
                and _attention_tile(kernel) == (64, 128)
                and _splits(kernel) == 4):
            return min(query_heads_per_cta, shared_warp_groups)
        return query_heads_per_cta
    raise ValueError("FA2 shared-query source algorithm is unsupported")


def _metax_372_fa_shared_issue_demand(
        kernel: KernelSpec, hardware,
        tensor_operations: float) -> float:
    """Return the exact shared MMA/MTE issue lower bound.

    C500's supplied architecture assigns MMA and Vector-ALU instructions to
    one common issue slot.  The fixed MACA 3.7.2 direct-decode binary exposes
    a source-level loop over complete 64-column KV tiles; exact-symbol
    non-duration counters close the MTE instruction count for that loop.  A
    second aggregate ``warp_issue`` demand counts one shared-slot issue for
    every mandatory MMA and MTE instruction.  Tensor execution service,
    Vector execution service, and this shared issue envelope are independent
    lower bounds and therefore combine by ``max``.

    A dedicated MACA 3.7.2 register-only mixed stream rejects the stronger
    assumption that one MMA holds the shared slot for its complete four-cycle
    saturated initiation interval: 4 MMA plus 12 independent Vector adds runs
    at the MMA-only envelope, while 4 MMA plus 16 adds follows the aggregate
    one-instruction/slot/cycle envelope.  Consequently this function must not
    add ``tensor_operations / tensor_capacity`` to the MTE issue count.

    This is intentionally narrower than the static trait itself.  Partial KV
    tails, split launches, other traits, causal attention and queue workers do
    not inherit the two-point full-tile instruction observation.
    """

    if _has_graph_padding(kernel):
        return 0.0
    tile = _attention_tile(kernel)
    batch, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    if _is_source_bound_metax_mla_main(kernel):
        work = _metax_mla_scheduler_work(kernel)
        mma_instructions = float(
            _METAX_MLA_MMA_PER_KV_BLOCK * work.kv_block_count)
        mte_instructions = float(
            _METAX_MLA_MTE_LAUNCH
            + _METAX_MLA_MTE_PER_KV_BLOCK * work.kv_block_count
            + _METAX_MLA_MTE_PER_SEGMENT * work.segment_count
            + _METAX_MLA_MTE_PER_SAME_WORKER_TRANSITION
            * work.same_worker_transitions
            + _METAX_MLA_MTE_PER_UNSPLIT_REQUEST
            * work.unsplit_requests)
        expected_tensor_operations = float(
            mma_instructions * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    elif kernel.launch.algorithm in {
            _METAX_372_Q64_PREFILL_ALGORITHM,
            _METAX_372_Q64_PACKED_PREFILL_ALGORITHM,
    }:
        is_direct_q64 = (
            kernel.launch.algorithm == _METAX_372_Q64_PREFILL_ALGORITHM)
        common_q64_domain = (
            kernel.variant.lower() == "prefill"
            and kernel.dtype.lower() == "bf16"
            and head_dim in _METAX_372_Q64_PREFILL_MMA_PER_KV_STEP_BY_HEAD_DIM
            and tile == (64, 64)
            and kernel.launch.block_size == 256
            and kernel.launch.parameters["unpadded_flops"] == 0.0
            and _splits(kernel) == 1
            and kernel.attributes["causal"] is True
            and kernel.attributes["fa_ksteps"] is None)
        direct_q64_domain = (
            is_direct_q64
            and kernel.attributes["kv_layout"] in ("paged", "paged_varlen")
            and q_len % 64 == 0
            and all(query % 64 == 0 and key_value % 64 == 0
                    for query, key_value in zip(q_lens, kv_lens))
            and all(query == key_value
                    for query, key_value in zip(q_lens, kv_lens))
            and kernel.launch.grid
            == (query_heads, q_len // 64, batch))
        packed_q64_domain = (
            not is_direct_q64
            and kernel.attributes["kv_layout"] == "chunked_prefill"
            and _work_distribution(kernel) in {
                "direct_cta", "packed_varlen_queue"})
        if not (common_q64_domain
                and (direct_q64_domain or packed_q64_domain)):
            raise ValueError(
                "MetaX MACA 3.7.2 Q64 prefill program identity disagrees "
                "with its exact counter domain")
        step_counts, head_groups = _metax_pipeline_step_counts(
            kernel, tile_q=64, tile_kv=64)
        if head_groups != query_heads:
            raise ValueError(
                "MetaX MACA 3.7.2 Q64 program requires one query head per "
                "logical workgroup")
        active_q_tiles = len(step_counts)
        causal_kv_steps = sum(step_counts)
        scheduled_q_tiles = batch * math.ceil(q_len / 64)
        no_prefix_requests = sum(
            query == key_value for query, key_value
            in zip(q_lens, kv_lens))
        mma_per_kv_step = (
            _METAX_372_Q64_PREFILL_MMA_PER_KV_STEP_BY_HEAD_DIM[head_dim])
        mma_instructions = float(
            query_heads * mma_per_kv_step * causal_kv_steps)
        if head_dim == 128:
            mte_instructions = float(query_heads * (
                _METAX_372_Q64_PREFILL_MTE_PER_KV_STEP * causal_kv_steps
                + _METAX_372_Q64_PREFILL_MTE_PER_ACTIVE_Q_TILE
                * active_q_tiles
                + _METAX_372_Q64_PREFILL_MTE_PER_SCHEDULED_CTA
                * scheduled_q_tiles
                + _METAX_372_Q64_PREFILL_MTE_PER_REQUEST_ADJUSTMENT
                * no_prefix_requests))
        elif head_dim == 64:
            # Full-coverage layer-1 observations close the Tensor instruction
            # count at 128 MMA instructions per causal KV step.  The D64 MTE
            # instruction decomposition is not yet closed, so it must not be
            # folded into the shared-issue lower bound.
            mte_instructions = 0.0
        else:
            raise AssertionError("MetaX Q64 head dimension escaped validation")
        expected_tensor_operations = float(
            mma_instructions * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    elif _has_metax_q64_contiguous_decode_pipeline(kernel):
        step_counts, head_groups = _metax_pipeline_step_counts(
            kernel, tile_q=64, tile_kv=64)
        active_ctas = len(step_counts) * head_groups
        if active_ctas != kernel.launch.grid_size:
            raise ValueError(
                "MetaX MACA 3.7.2 Q64 decode requires one source CTA per "
                f"workgroup: semantic={active_ctas}, "
                f"grid={kernel.launch.grid_size}")
        mma_instructions = float(
            head_groups * sum(step_counts)
            * _METAX_372_Q64_DECODE_MMA_PER_KV_STEP)
        # Only the exact MMA count is closed across this source program.  Do
        # not infer an MTE affine model from its single eight-panel capture.
        mte_instructions = 0.0
        expected_tensor_operations = float(
            mma_instructions * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    elif _is_source_bound_metax_tile16_program(kernel):
        step_counts, head_groups = _metax_pipeline_step_counts(
            kernel, tile_q=16, tile_kv=16)
        logical_ctas = len(step_counts) * head_groups
        distribution = _work_distribution(kernel)
        if (distribution == "direct_cta"
                and logical_ctas != kernel.launch.grid_size):
            raise ValueError(
                "MetaX MACA 3.7.2 tile16 FA MTE demand requires one "
                "source CTA per launch workgroup: "
                f"semantic={logical_ctas}, grid={kernel.launch.grid_size}")
        if (distribution == "packed_varlen_queue"
                and kernel.launch.grid_size > logical_ctas):
            raise ValueError(
                "MetaX MACA 3.7.2 tile16 packed queue has more physical "
                "workers than logical jobs: "
                f"logical={logical_ctas}, grid={kernel.launch.grid_size}")
        layout = kernel.attributes["kv_layout"]
        if head_dim == 64:
            if layout != "contiguous":
                raise AssertionError(
                    "source-bound MetaX D64 tile16 layout escaped validation")
            fixed_mte = (
                _METAX_372_TILE16_D64_UNSPLIT_MTE_FIXED
                if _splits(kernel) == 1
                else _METAX_372_TILE16_D64_SPLIT_MTE_FIXED)
            mte_per_step = _METAX_372_TILE16_D64_MTE_PER_KV_STEP
            mma_per_step = _METAX_372_TILE16_D64_MMA_PER_KV_STEP
        elif layout in ("paged", "paged_varlen"):
            fixed_mte = _METAX_372_TILE16_PAGED_MTE_FIXED
            mte_per_step = _METAX_372_TILE16_PAGED_MTE_PER_KV_STEP
            mma_per_step = _METAX_372_TILE16_MMA_PER_KV_STEP
        elif layout == "contiguous":
            fixed_mte = _METAX_372_TILE16_CONTIGUOUS_MTE_FIXED
            mte_per_step = _METAX_372_TILE16_CONTIGUOUS_MTE_PER_KV_STEP
            mma_per_step = _METAX_372_TILE16_MMA_PER_KV_STEP
        else:
            raise AssertionError(
                "source-bound MetaX tile16 layout escaped validation")
        mma_instructions = float(
            head_groups * sum(step_counts)
            * mma_per_step)
        # Empty SplitKV CTAs follow the source's early-exit store path before
        # its MMA/MTE loop.  They occupy a physical launch slot but do not
        # execute this counter-closed active-program instruction sequence.
        mte_instructions = float(head_groups * sum(
            fixed_mte + steps * mte_per_step
            for steps in step_counts if steps > 0))
        expected_tensor_operations = float(
            mma_instructions * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    elif kernel.launch.algorithm.startswith(
            _METAX_STATIC_TRAIT_DIRECT_PREFIX):
        if not (
                kernel.variant.lower() == "decode"
                and kernel.dtype.lower() == "bf16"
                and q_len == 1
                and all(length == 1 for length in q_lens)
                and head_dim == 128
                and tile == (128, 64)
                and kernel.launch.block_size == 256
                and kernel.launch.parameters["unpadded_flops"] == 0.0
                and _splits(kernel) == 1
                and kernel.attributes["causal"] is False
                and kernel.attributes["fa_ksteps"] is None
                and all(length % 64 == 0 for length in kv_lens)):
            return 0.0

        step_counts, head_groups = _metax_pipeline_step_counts(
            kernel, tile_q=128, tile_kv=64)
        expected_steps = tuple(length // 64 for length in kv_lens)
        if step_counts != expected_steps:
            raise ValueError(
                "MetaX MACA 3.7.2 exact FA MTE loop disagrees with KV tiles: "
                f"pipeline={step_counts}, lengths={expected_steps}")
        active_ctas = len(step_counts) * head_groups
        if active_ctas != kernel.launch.grid_size:
            raise ValueError(
                "MetaX MACA 3.7.2 exact FA MTE demand requires one complete "
                "source CTA per launch workgroup: "
                f"semantic={active_ctas}, grid={kernel.launch.grid_size}")

        expected_tensor_operations = float(
            head_groups
            * sum(step_counts)
            * _METAX_372_FA_MMA_PER_FULL_KV_TILE
            * _METAX_372_FA_TENSOR_OPS_PER_MMA)
        mte_instructions = float(head_groups * sum(
            _METAX_372_FA_MTE_FIXED_PER_CTA
            + steps * _METAX_372_FA_MTE_PER_FULL_KV_TILE
            for steps in step_counts))
        mma_instructions = (
            tensor_operations / _METAX_372_FA_TENSOR_OPS_PER_MMA)
    else:
        return 0.0
    if not math.isclose(
            tensor_operations, expected_tensor_operations,
            rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "MetaX MACA 3.7.2 exact FA Tensor work disagrees with the "
            "counter-proved full-tile program: "
            f"tensor={tensor_operations}, expected={expected_tensor_operations}")

    tensor_resource = hardware.resource_by_role("compute_tensor")
    issue_resource = hardware.resource_by_role("warp_issue")
    tensor_ops_per_ap_cycle = (
        tensor_resource.capacity
        / hardware.nominal_sm_freq_hz
        / hardware.num_compute_units)
    issue_per_ap_cycle = (
        issue_resource.capacity
        / hardware.nominal_sm_freq_hz
        / hardware.num_compute_units)
    if not math.isclose(
            tensor_ops_per_ap_cycle,
            _METAX_372_FA_TENSOR_OPS_PER_AP_CYCLE,
            rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "MetaX MACA 3.7.2 exact FA issue contract requires "
            f"2048 Tensor op/AP/cycle, got {tensor_ops_per_ap_cycle}")
    if not math.isclose(
            issue_per_ap_cycle,
            _METAX_372_FA_WARP_ISSUE_PER_AP_CYCLE,
            rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise ValueError(
            "MetaX MACA 3.7.2 exact FA issue contract requires "
            f"5 aggregate issue slots/AP/cycle, got {issue_per_ap_cycle}")

    # ``warp_issue`` is measured in occupied slot-cycles.  Each counter
    # instruction occupies one of the five independent AP issue slots, so one
    # instruction contributes one slot-cycle; the resource capacity already
    # supplies the factor of five slots/AP/cycle.  Multiplying the demand by
    # that capacity would incorrectly collapse the machine back to one
    # instruction/AP/cycle.
    return mma_instructions + mte_instructions


def _cann_workspace_lower_bound(
        kernel: KernelSpec, *,
        split_fuse_work=_CANN_SPLIT_FUSE_WORK_UNSET
        ) -> Optional[_CannWorkspaceLowerBound]:
    """Return source-mandatory, unpadded CANN FIA workspace traffic.

    The fixed CANN 8.5.1 AIC/AIV chain materializes MM1 FP32 scores, Vec1
    BF16 probabilities, and MM2 FP32 outputs in GM.  Per score element this
    is 4B write + 4B read + 2B write + 2B read; per MM2 output element it is
    4B write + 4B read.  Exact tiling payloads can only increase these totals
    through padding, repeated partial outputs, online-state replay, and Flash
    Decode partials, so the unpadded quantities below are an analytical lower
    bound rather than a workload-duration calibration.
    """

    algorithm = kernel.launch.algorithm
    if split_fuse_work is _CANN_SPLIT_FUSE_WORK_UNSET:
        split_fuse_work = _cann_split_fuse_work(kernel)
    if (algorithm not in _CANN_STANDARD_GQA_DECODE_ALGORITHMS
            and algorithm not in _CANN_MLA_DECODE_ALGORITHMS
            and split_fuse_work is None):
        return None
    if split_fuse_work is None:
        _, _, _, head_dim, query_heads, _ = _shape(kernel)
        q_lens, _ = _request_lengths(kernel)
        cann_mla = _cann_mla_dimensions(kernel)
        value_head_dim = head_dim if cann_mla is None else cann_mla[1]
        from src.kernel.memory_cann_mla import has_cann_mla_source,bound_cann_mla_source_work
        source=bound_cann_mla_source_work(kernel) if has_cann_mla_source(kernel) else bound_cann_bsh_workspace_work(kernel)
        if source is not None:
            score_elements=float(source['priced_score_elements']);output_elements=float(source['priced_output_elements'])
        else:
            score_elements = _softmax_elements(
                kernel, split_fuse_work=split_fuse_work)
            output_elements = float(
                query_heads * sum(q_lens) * value_head_dim)
    else:
        # SplitFuse materializes one FP32 MM2 result for every 512-column KV
        # stack, not merely one result per final output row.  The task/stack
        # enumeration is fixed by the source Q=128 and PRE_LAUNCH geometry;
        # padding, online-state replay, and residency can only add traffic.
        score_elements = float(split_fuse_work.attention_elements)
        output_elements = float(
            split_fuse_work.workspace_output_elements)
    read_bytes = 6.0 * score_elements + 4.0 * output_elements
    aic_write_bytes = 4.0 * score_elements + 4.0 * output_elements
    source_schedule = _cann_source_schedule_parameters(kernel)
    nupdate_write_bytes = (
        source_schedule["cann_source_nupdate_write_bytes"]
        if source_schedule is not None else 0.0)
    aiv_write_bytes = 2.0 * score_elements + nupdate_write_bytes
    if not math.isclose(
            read_bytes + nupdate_write_bytes,
            aic_write_bytes + aiv_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-9):
        raise AssertionError("CANN FIA base workspace traffic must balance")
    return _CannWorkspaceLowerBound(
        score_elements=score_elements,
        output_elements=output_elements,
        read_bytes=read_bytes,
        aic_write_bytes=aic_write_bytes,
        aiv_write_bytes=aiv_write_bytes,
    )


def _cann_standard_decode_vector_lower_bound(
        kernel: KernelSpec) -> Optional[_CannStandardDecodeVectorLowerBound]:
    """Return source-mandatory, unpadded standard-GQA AIV operations.

    The common analytical softmax demand already contributes three operations
    per score element.  Fixed CANN 8.5.1 ``FiaBlockVecNonQuant`` additionally
    scales every MM1 score and casts every Vec1 probability from FP32 to BF16.
    Its non-FlashDecode path then normalizes every final FP32 output element
    and casts it to BF16.  The FlashDecode path instead weights at least one
    partial output before the same final BF16 cast.  Therefore two additional
    operations per score and two per output are a source-closed lower bound
    shared by all five standard GQA variants.

    Padding, repeated online-state merges, multi-partial FlashDecode reduction,
    reduction instructions, and intrinsic setup remain unavailable.  They are
    deliberately not inferred from evaluation duration or shape residuals.
    """

    if _cann_standard_decode_grouped_rows(kernel) is None:
        return None
    workspace = _cann_workspace_lower_bound(kernel)
    if workspace is None:
        raise AssertionError(
            "CANN standard decode lost its workspace element contract")
    return _CannStandardDecodeVectorLowerBound(
        vec1_operations=5.0 * workspace.score_elements,
        vec2_operations=2.0 * workspace.output_elements,
    )


def _output_bytes(kernel: KernelSpec, hardware) -> float:
    """Return the source-defined output/scratch write volume."""
    from src.kernel.memory_cann_mla import has_cann_mla_source,current_cann_mla_source_work
    if has_cann_mla_source(kernel):return float(current_cann_mla_source_work(kernel,hardware)['write_bytes'])
    if has_cann_tnd_source(kernel):
        return float(current_cann_tnd_source_work(kernel,hardware)['write_bytes'])
    if has_metax_varlen_source(kernel):
        return float(current_metax_varlen_source_work(kernel,hardware)['priced_write_bytes'])
    if has_cann_attention_source(kernel):
        return float(current_cann_attention_source_work(kernel,hardware)['write_bytes'])
    if has_fa2_split_source(kernel):
        return float(current_fa2_split_source_work(kernel)['priced_write_bytes'])
    if has_grouped_mla_source(kernel):
        return float(current_grouped_mla_source_work(kernel)['priced_write_bytes'])
    if has_dense_flash_attention_source(kernel):
        from src.kernel.memory_flash_attention import has_dense_ragged_analytical_work
        if has_dense_ragged_analytical_work(kernel):
            return float(dense_attention_prediction_work(kernel,hardware)['output'])
        from src.kernel.memory_path_solver import dense_attention_output_projection
        return float(dense_attention_output_projection(dense_attention_memory_access_program(kernel,hardware)))
    from src.kernel.memory_access_builder import has_logical_interface_source,logical_interface_memory_access_program
    if has_logical_interface_source(kernel,hardware):
        from src.kernel.memory_access_builder import has_public_attention_source,public_attention_dimensions
        if has_public_attention_source(kernel):
            from src.kernel.memory_path_solver import public_attention_source_work
            work=public_attention_source_work(logical_interface_memory_access_program(kernel,hardware),expected_kernel=kernel,hardware=hardware)
            # The original MACA unknown-scratch policy prices only inputs.
            return 0.0 if public_attention_dimensions(kernel)[0]=='metax_mla' else float(work['write_bytes'])
        from src.kernel.memory_path_solver import logical_interface_source_bytes
        _,write=logical_interface_source_bytes(logical_interface_memory_access_program(kernel,hardware),
            expected_kernel=kernel,hardware=hardware)
        return float(write)

    variant = kernel.variant.lower()
    if variant in ("combine", "split_combine"):
        if _is_source_bound_metax_mla_combine(kernel):
            _, final_write = _metax_mla_combine_traffic_bytes(kernel)
            return final_write
        raise ValueError('Combine output requires its unified memory source')
    if _is_source_bound_metax_mla_main(kernel):
        return _metax_mla_main_output_bytes(kernel)
    if _is_unbound_metax_mla_scratch(kernel):
        return 0.0
    raise ValueError('Attention output demand requires its unified source')


def _compulsory_input_bytes(kernel: KernelSpec, hardware) -> float:
    """Return the unique Q/K/V input footprint established by the ABI."""
    if has_cann_tnd_source(kernel):
        return float(current_cann_tnd_source_work(kernel,hardware)['payload_read_bytes'])
    if has_metax_varlen_source(kernel):
        return float(current_metax_varlen_source_work(kernel,hardware)['payload_read_bytes'])
    if has_cann_attention_source(kernel):
        return float(current_cann_attention_source_work(kernel,hardware)['payload_read_bytes'])
    if has_fa2_split_source(kernel):
        return float(current_fa2_split_source_work(kernel)['compulsory_input_bytes'])
    if has_dense_flash_attention_source(kernel):
        source=dense_attention_prediction_work(kernel,hardware)
        return float(sum(source[role] for role in ('query','key','value')))

    from src.kernel.memory_cann_mla import has_cann_mla_source,current_cann_mla_source_work
    if has_cann_mla_source(kernel):
        work=current_cann_mla_source_work(kernel,hardware)
        return float(work['query_unique_bytes']+work['key_interface_bytes']+work['value_interface_bytes'])
    if has_grouped_mla_source(kernel):
        return float(current_grouped_mla_source_work(kernel)['legacy_nominal_compulsory_input_bytes'])
    if has_metax_mla_source(kernel):return float(current_metax_mla_source_work(kernel)['compulsory_input_bytes'])
    from src.kernel.memory_access_builder import has_public_attention_source,has_logical_interface_source,logical_interface_memory_access_program
    if has_public_attention_source(kernel):
        from src.kernel.memory_path_solver import public_attention_source_work
        return float(public_attention_source_work(logical_interface_memory_access_program(kernel,hardware),
            expected_kernel=kernel,hardware=hardware)['nominal_compulsory_input_bytes'])
    if has_logical_interface_source(kernel,hardware):
        from src.kernel.memory_path_solver import logical_interface_source_traffic
        return float(logical_interface_source_traffic(logical_interface_memory_access_program(kernel,hardware),
            expected_kernel=kernel,hardware=hardware)[1])
    raise ValueError('Attention compulsory demand requires its unified source')


def _compulsory_input_dram_bytes(kernel: KernelSpec, hardware) -> float:
    """Return the compulsory Q/K/V input footprint the GM node actually sees.

    Same ABI footprint as ``_compulsory_input_bytes``, with K and V divided by
    the adjacent-slice sharing their CANN ND2NZ copy geometry produces (see
    ``_cann_kv_fill_share``).  Q keeps its full compulsory account: the probe
    covers the K/V per-kv_head slice layout, and Q is under half a percent of
    this term for every decode shape in the dataset, so crediting it would be
    an extrapolation for no measurable gain.
    """

    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    mla = _mla_dimensions(kernel)
    if mla is None:
        qk_head_dim = head_dim
        value_head_dim = head_dim
    else:
        qk_head_dim, value_head_dim = mla
    from src.kernel.memory_cann_mla import has_cann_mla_source,current_cann_mla_source_work
    if has_cann_mla_source(kernel):
        work=current_cann_mla_source_work(kernel,hardware)
        query,key,value=(work[name] for name in ('query_unique_bytes','key_interface_bytes','value_interface_bytes'))
    elif has_cann_attention_source(kernel):
        work=current_cann_attention_source_work(kernel,hardware)
        query=work['query_unique_bytes'];key=value=work['kv_interface_bytes']/2
    elif has_cann_tnd_source(kernel):
        work=current_cann_tnd_source_work(kernel,hardware)
        query=work['query_unique_bytes'];key=value=work['kv_interface_bytes']/2
    else:
        from src.kernel.memory_access_builder import logical_interface_memory_access_program
        from src.kernel.memory_path_solver import public_attention_source_work
        work=public_attention_source_work(logical_interface_memory_access_program(kernel,hardware),expected_kernel=kernel,hardware=hardware)
        query,key,value=(work[name] for name in ('query_bytes','key_bytes','value_bytes'))
    kv_footprint=key if mla is not None else key+value
    key_share = _cann_kv_fill_share(
        hardware, kv_heads, qk_head_dim, element_bytes, kv_footprint)
    value_share = _cann_kv_fill_share(
        hardware, kv_heads, value_head_dim, element_bytes, kv_footprint)
    if mla is not None:
        # One latent per position: the value operand is that same object's
        # leading channels, so it adds no distinct GM footprint.
        return float(query + key / key_share)
    return float(query + key / key_share + value / value_share)


def _cann_kv_dram_footprint_bytes(
        kv_heads, kv_lens, qk_head_dim, value_head_dim, element_bytes,
        is_mla):
    """Return the distinct GM extent the K and V operands occupy.

    Ordinary attention keeps K and V in separate caches, so their footprints
    add.  MLA stores a single latent per position and the value operand is
    that latent's leading ``value_head_dim`` channels, so the two operands
    share one footprint and adding them counts the same bytes twice.

    The two operand streams are real and both are charged at L2, where the
    910B FlashAttention MLA cases predict within 0.906 to 0.960 of measured.
    It is only the compulsory GM footprint they share, which is what the
    measured DRAM side says: charging both puts the MLA read at 2.030 times
    the per-domain measured value, uniformly across batch 1 to 128 and KV 512
    to 8192, while every other FlashAttention group sits at 0.998 to 0.999.
    """

    width = (
        qk_head_dim if is_mla else qk_head_dim + value_head_dim)
    return float(kv_heads * sum(kv_lens) * width * element_bytes)


def _periodic_order_encoding(order):
    """Return an exact prefix-period, repeat count and tail for ``order``."""

    if not isinstance(order, tuple) or not order:
        raise TypeError("FLASH_ATTN program order must be a non-empty tuple")
    prefix_lengths = [0] * len(order)
    for index in range(1, len(order)):
        matched = prefix_lengths[index - 1]
        while matched and order[index] != order[matched]:
            matched = prefix_lengths[matched - 1]
        if order[index] == order[matched]:
            matched += 1
        prefix_lengths[index] = matched
    period = len(order) - prefix_lengths[-1]
    repeats, tail = divmod(len(order), period)
    if repeats <= 1:
        return order, 1, ()
    return order[:period], repeats, order[:tail]



def _require_periodic_cta_arguments(
        period_items, period_repetitions, worker_count) -> None:
    """Validate the arguments every periodic-CTA source program shares.

    A source period is a non-empty tuple and the repetition and worker counts
    are positive integers; these three facts gate both the materializing and
    the arithmetic expansion, so they are checked once.
    """

    if not isinstance(period_items, tuple) or not period_items:
        raise TypeError("periodic CTA source period must be non-empty")
    if (isinstance(period_repetitions, bool)
            or not isinstance(period_repetitions, int)
            or period_repetitions <= 0):
        raise ValueError("periodic CTA repetition count must be positive")
    if (isinstance(worker_count, bool)
            or not isinstance(worker_count, int)
            or worker_count <= 0):
        raise ValueError("periodic CTA worker count must be positive")


def _periodic_round_robin_worker_programs(
        period_items, period_repetitions, worker_count, *, start_ordinal=0):
    """Return exact strided worker programs without expanding all CTAs.

    A direct launch assigns source ordinal ``i`` to worker ``i % W``.  For a
    source period of length ``P``, one worker therefore observes another
    periodic sequence with length ``P / gcd(P, W)``.  This computes that
    sequence and its repeat/tail counts arithmetically, so storage is bounded
    by the source period and worker count rather than ``P * repetitions``.
    """

    _require_periodic_cta_arguments(
        period_items, period_repetitions, worker_count)
    if (isinstance(start_ordinal, bool)
            or not isinstance(start_ordinal, int)
            or not 0 <= start_ordinal < worker_count):
        raise ValueError(
            "periodic CTA start ordinal must be within the worker domain")

    period_length = len(period_items)
    total_items = period_length * period_repetitions
    sampled_period = period_length // math.gcd(period_length, worker_count)
    programs = []
    for worker in range(worker_count):
        first_offset = (worker - start_ordinal) % worker_count
        if first_offset >= total_items:
            programs.append(None)
            continue
        local_count = (total_items - 1 - first_offset) // worker_count + 1
        order_length = min(sampled_period, local_count)
        order = tuple(
            period_items[
                (first_offset + index * worker_count) % period_length]
            for index in range(order_length))
        if local_count > sampled_period:
            repeat_count, tail_length = divmod(
                local_count, sampled_period)
            tail_order = order[:tail_length]
        else:
            repeat_count = 1
            tail_order = ()
        programs.append((order, repeat_count, tail_order))
    return tuple(programs)


def _metax_bf16_mma_capacity_for_active_waves(active_waves_per_ap):
    """Return the register-only BF16 MMA capacity at active-wave occupancy.

    Fractional occupancies represent the chip-wide mixture of adjacent
    integer active-wave populations.  Linear averaging between the frozen
    independent sweep endpoints therefore preserves aggregate AP capacity;
    it is not fitted to a production kernel duration.
    """

    if (isinstance(active_waves_per_ap, bool)
            or not isinstance(active_waves_per_ap, (int, float))
            or not math.isfinite(active_waves_per_ap)
            or active_waves_per_ap <= 0):
        raise ValueError("MetaX active waves per AP must be positive")
    points = _METAX_372_BF16_MMA_FLOPS_PER_AP_CYCLE_BY_WAVES
    if active_waves_per_ap <= points[0][0]:
        return points[0][1]
    if active_waves_per_ap >= points[-1][0]:
        return points[-1][1]
    for (lower_waves, lower_capacity), (
            upper_waves, upper_capacity) in zip(points, points[1:]):
        if active_waves_per_ap <= upper_waves:
            fraction = (
                (active_waves_per_ap - lower_waves)
                / (upper_waves - lower_waves))
            return (
                lower_capacity
                + fraction * (upper_capacity - lower_capacity))
    raise AssertionError("MetaX BF16 MMA wave curve is not exhaustive")


def _is_metax_q64_direct_prefill_program(kernel: KernelSpec) -> bool:
    """Recognize the source-selected direct Q64 prefill trait.

    MACA 3.7.2 dispatches the same ``<D128, M64, N64, four-wave>`` direct
    trait for a ragged paged-varlen call as for a full-tile call.  The latter
    is a narrower counter-closure domain, not a different program.  This
    recognizer therefore relies solely on the source dispatch geometry and
    ABI-visible request lengths; it deliberately does not reuse the
    full-tile-only instruction-count contract.
    """

    if kernel.launch.algorithm not in {
            _METAX_372_Q64_PREFILL_ALGORITHM,
            "metax_flash_attention_trait_direct:prefill",
    }:
        return False
    batch, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    return (
        kernel.variant.lower() == "prefill"
        and kernel.dtype.lower() == "bf16"
        and head_dim in _METAX_372_Q64_PREFILL_MMA_PER_KV_STEP_BY_HEAD_DIM
        and max(q_lens) == q_len
        and _attention_tile(kernel) == (64, 64)
        and _splits(kernel) == 1
        and kernel.attributes["causal"] is True
        and kernel.attributes["kv_layout"] in ("paged", "paged_varlen")
        and kernel.launch.grid == (
            query_heads, math.ceil(q_len / 64), batch)
        and kernel.launch.block == (256, 1, 1)
        and all(query > 0 and query <= key_value
                for query, key_value in zip(q_lens, kv_lens))
    )


def _has_metax_q64_prefill_pipeline(kernel: KernelSpec) -> bool:
    """Whether the pinned Q64 source defines every active CTA's DAG."""

    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm not in {
                _METAX_372_Q64_PREFILL_ALGORITHM,
                "metax_flash_attention_trait_direct:prefill",
                _METAX_372_Q64_PACKED_PREFILL_ALGORITHM,
            }):
        return False
    _, q_len, _, head_dim, _, _ = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    common = (
        kernel.variant.lower() == "prefill"
        and kernel.dtype.lower() == "bf16"
        and head_dim in _METAX_372_Q64_PREFILL_MMA_PER_KV_STEP_BY_HEAD_DIM
        and max(q_lens) == q_len
        and _attention_tile(kernel) == (64, 64)
        and _splits(kernel) == 1
        and kernel.attributes["causal"] is True
        and kernel.launch.block == (256, 1, 1)
    )
    if not common:
        return False
    if kernel.launch.algorithm != _METAX_372_Q64_PACKED_PREFILL_ALGORITHM:
        return _is_metax_q64_direct_prefill_program(kernel)
    return (
        kernel.attributes["kv_layout"] == "chunked_prefill"
        and _work_distribution(kernel) == "direct_cta"
    )


def _metax_q64_prefill_barrier_cycles(kernel: KernelSpec, hardware) -> float:
    """Return source-counted D128 Q64 CTA-barrier cycles."""

    if not _has_metax_q64_prefill_pipeline(kernel):
        return 0.0
    _, _, _, head_dim, _, _ = _shape(kernel)
    if head_dim != 128:
        return 0.0
    mechanism = hardware.cta_barrier_service_cycles
    if mechanism is None:
        raise ValueError(
            "MetaX D128 Q64 pipeline requires CTA-barrier service evidence")
    if (kernel.launch.block_size != mechanism["threads_per_cta"]
            or mechanism["dynamic_shared_bytes"] != 32 * 1024
            or mechanism["resident_ctas_per_unit"]
            != _METAX_372_Q64_PREFILL_RESIDENT_CTA_SLOTS_PER_AP):
        raise ValueError(
            "MetaX Q64 launch does not match CTA-barrier evidence")
    class_counts, _ = _pipeline_cta_class_counts(
        kernel, 64, 64, grouped_mla=False)
    # The pinned true/true trait executes two Q/K shared-memory-reuse barriers
    # in its prologue, two wait/reuse barriers per KV panel, and two output
    # shared-memory-reuse barriers in its epilogue.
    barrier_count = sum(
        count * (2 * item.step_count + 4)
        for item, count in class_counts)
    return float(barrier_count * mechanism["barrier_cycles"])


def _has_metax_q64_contiguous_decode_pipeline(kernel: KernelSpec) -> bool:
    """Whether the pinned BlockM64/BlockN64 decode program is complete."""

    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm != _METAX_372_Q64_DECODE_ALGORITHM):
        return False
    batch, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    if query_heads % kv_heads:
        raise ValueError("MetaX Q64 decode GQA heads must divide exactly")
    return (
        kernel.variant.lower() == "decode"
        and kernel.dtype.lower() == "bf16"
        and q_len == 1
        and all(length == 1 for length in q_lens)
        and head_dim == 128
        and _attention_tile(kernel) == (64, 64)
        and _splits(kernel) == 1
        and kernel.attributes["causal"] is False
        and kernel.attributes["kv_layout"] == "contiguous"
        and all(length % 64 == 0 for length in kv_lens)
        and kernel.launch.grid == (kv_heads, batch, 1)
        and kernel.launch.block == (256, 1, 1)
    )


def _has_metax_q128_contiguous_decode_pipeline(kernel: KernelSpec) -> bool:
    """Whether the pinned BlockM128/BlockN64 decode program is complete."""

    if (_has_graph_padding(kernel)
            or kernel.launch.algorithm != _METAX_372_Q128_DECODE_ALGORITHM):
        return False
    batch, q_len, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    if query_heads % kv_heads:
        raise ValueError("MetaX Q128 decode GQA heads must divide exactly")
    return (
        kernel.variant.lower() == "decode"
        and kernel.dtype.lower() == "bf16"
        and q_len == 1
        and all(length == 1 for length in q_lens)
        and head_dim == 128
        and _attention_tile(kernel) == (128, 64)
        and _splits(kernel) == 1
        and kernel.attributes["causal"] is False
        and kernel.attributes["kv_layout"] == "contiguous"
        and all(length % 64 == 0 for length in kv_lens)
        and kernel.launch.grid == (kv_heads, batch, 1)
        and kernel.launch.block == (256, 1, 1)
    )


def _attach_metax_q64_prefill_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the source-defined MACA 3.7.2 Q64 prefill pipeline.

    Direct launches assign their source-linear CTA stream across active APs.
    Every AP remains an explicit worker and the exact compiler resource report
    permits two resident CTAs per AP.

    Ragged direct grids also contain padded query tiles.  The source rejects
    those tiles before the Q/K/V path, so they contribute no class to this
    active-work envelope.  Their mandatory instruction issue is retained by
    the independent whole-kernel ``warp_issue`` lower bound, whose scheduled
    CTA term covers the complete physical grid.

    The per-CTA DAG follows the pinned xcore1000 source: Q load/stage, K load,
    Q shared-memory read, V load, K stage, QK, V staging, next-K prefetch
    overlapped with online softmax/PV, and the output epilogue.  Aggregate
    MMA/MTE issue remains an
    independent whole-kernel lower bound because the non-duration counters do
    not place every issue instruction inside the source DAG.
    """

    PipelineStageDemand = bind_pipeline_stage_template
    if not _has_metax_q64_prefill_pipeline(kernel):
        raise ValueError(
            "MetaX Q64 source pipeline requires its exact causal trait")
    batch, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    if inp.active_units <= 0:
        raise ValueError("MetaX Q64 pipeline requires active APs")
    worker_count = inp.active_units

    scheduled_q_tiles = math.ceil(q_len / 64)
    class_counts, head_groups = _pipeline_cta_class_counts(
        kernel, 64, 64, grouped_mla=False)
    if head_groups != query_heads:
        raise ValueError(
            "MetaX Q64 source requires one CTA per query head")
    active_cta_count = sum(count for _, count in class_counts)
    scheduled_cta_count = batch * scheduled_q_tiles * query_heads
    if scheduled_cta_count != kernel.launch.grid_size:
        raise ValueError(
            "MetaX Q64 scheduled CTA count disagrees with the direct grid: "
            f"source={scheduled_cta_count}, grid={kernel.launch.grid_size}")
    _, kv_lens = _request_lengths(kernel)
    schedule_class_counts = []
    empty_cta_count = 0
    for request_q, request_kv in zip(q_lens, kv_lens):
        request_tiles = math.ceil(request_q / 64)
        context = request_kv - request_q
        for physical_q_tile in range(scheduled_q_tiles):
            source_q_tile = scheduled_q_tiles - physical_q_tile - 1
            if source_q_tile >= request_tiles:
                empty_cta_count += query_heads
                schedule_class_counts.append((None, query_heads))
                continue
            start = source_q_tile * 64
            end = min(request_q, start + 64)
            item = _AttentionCTAWork(
                query_rows=end - start,
                step_count=math.ceil((context + end) / 64),
            )
            if schedule_class_counts and schedule_class_counts[-1][0] == item:
                previous, count = schedule_class_counts[-1]
                schedule_class_counts[-1] = (
                    previous, count + query_heads)
            else:
                schedule_class_counts.append((item, query_heads))
    if active_cta_count + empty_cta_count != scheduled_cta_count:
        raise AssertionError("MetaX Q64 source schedule lost physical CTAs")
    if sum(count for _, count in schedule_class_counts) != scheduled_cta_count:
        raise AssertionError("MetaX Q64 source schedule lost physical CTAs")
    total_kv_steps = sum(
        item.step_count * count for item, count in class_counts)
    element_bytes = int(kernel.shape["dtype_bytes"])
    kv_bytes_per_step = float(64 * head_dim * element_bytes)
    read_bytes = float(
        inp.demand_per_resource["l2_cache"] - _output_bytes(kernel, hardware))
    write_bytes = _output_bytes(kernel, hardware)
    expected_read_bytes = math.fsum(
        count * (
            item.query_rows * head_dim * element_bytes
            + item.step_count * 2.0 * kv_bytes_per_step)
        for item, count in class_counts)
    expected_write_bytes = math.fsum(
        count * item.query_rows * (head_dim * element_bytes + 4)
        for item, count in class_counts)
    if not math.isclose(
            read_bytes, expected_read_bytes, rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 source DAG read bytes disagree with aggregate traffic: "
            f"dag={expected_read_bytes}, aggregate={read_bytes}")
    if not math.isclose(
            write_bytes, expected_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 source DAG write bytes disagree with aggregate traffic: "
            f"dag={expected_write_bytes}, aggregate={write_bytes}")
    smem_bytes = float(inp.demand_per_resource["smem"])
    expected_smem_bytes = 2.0 * (
        read_bytes + math.fsum(
            count * item.query_rows * head_dim * element_bytes
            for item, count in class_counts))
    if not math.isclose(
            smem_bytes, expected_smem_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 source DAG shared-memory bytes disagree with aggregate "
            f"traffic: dag={expected_smem_bytes}, aggregate={smem_bytes}")
    tensor_operations = float(
        inp.demand_per_resource["compute_tensor"])
    mma_per_kv_step = (
        _METAX_372_Q64_PREFILL_MMA_PER_KV_STEP_BY_HEAD_DIM[head_dim])
    expected_tensor_operations = float(
        total_kv_steps * mma_per_kv_step
        * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    if not math.isclose(
            tensor_operations, expected_tensor_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 source DAG Tensor work disagrees with aggregate demand")
    softmax_operations = float(inp.demand_per_resource["compute_sfu"])
    expected_softmax_operations = float(math.fsum(
        count * (
            item.step_count * 64 * 64
            + max(0, item.step_count - 1) * 64)
        for item, count in class_counts))
    if not math.isclose(
            softmax_operations, expected_softmax_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 source DAG softmax work disagrees with aggregate demand")
    if head_dim == 64:
        resident_cta_slots = (
            _METAX_372_D64_Q64_PREFILL_RESIDENT_CTA_SLOTS_PER_AP)
    elif head_dim == 128:
        resident_cta_slots = _METAX_372_Q64_PREFILL_RESIDENT_CTA_SLOTS_PER_AP
    else:
        raise ValueError("MetaX Q64 prefill head dimension is unsupported")
    waves_per_cta = kernel.launch.block_size / hardware.parallel_width
    active_waves_per_ap = min(
        kernel.launch.grid_size * waves_per_cta / inp.active_units,
        resident_cta_slots * waves_per_cta,
    )
    tensor_wave_efficiency = (
        _metax_bf16_mma_capacity_for_active_waves(active_waves_per_ap)
        / _METAX_372_FA_TENSOR_OPS_PER_AP_CYCLE)
    adjusted_compute_demands = []
    for demand in inp.compute_demands:
        if demand.resource_role == "compute_tensor":
            adjusted_compute_demands.append(dataclasses.replace(
                demand,
                schedule_efficiency=(
                    demand.schedule_efficiency * tensor_wave_efficiency),
            ))
        elif demand.resource_role in {
                "compute_vector", "compute_reduction", "compute_sfu",
                "warp_issue"}:
            adjusted_compute_demands.append(demand)
        else:
            raise ValueError(
                "MetaX Q64 prefill compute role is unsupported: "
                f"{demand.resource_role!r}")
    inp = dataclasses.replace(
        inp, compute_demands=tuple(adjusted_compute_demands))
    softmax_service = pipeline_compute_service(
        inp, hardware, roles=("compute_sfu",))
    barrier_service = None
    barrier_fraction = None
    if head_dim == 128:
        total_barrier_cycles = float(
            inp.demand_per_resource["compute_reduction"])
        mechanism = hardware.cta_barrier_service_cycles
        if mechanism is None or total_barrier_cycles <= 0:
            raise ValueError(
                "MetaX D128 Q64 pipeline requires positive CTA-barrier work")
        barrier_service = pipeline_compute_service(
            inp, hardware, roles=("compute_reduction",))
        barrier_fraction = (
            mechanism["barrier_cycles"] / total_barrier_cycles)
    tensor_step_fraction = (
        0.5 * mma_per_kv_step
        * _METAX_372_FA_TENSOR_OPS_PER_MMA
        / tensor_operations)

    sequence_templates = {}

    def _cta_phases(item):
        identity = (item.query_rows, item.step_count)
        if identity in sequence_templates:
            return sequence_templates[identity]
        step_count = item.step_count
        q_bytes_per_cta = float(
            item.query_rows * head_dim * element_bytes)
        output_bytes_per_cta = float(
            item.query_rows * (head_dim * element_bytes + 4))
        staged_output_bytes_per_cta = float(
            item.query_rows * head_dim * element_bytes)
        read_scale = 1.0 / read_bytes
        write_scale = 1.0 / write_bytes
        smem_scale = 1.0 / smem_bytes
        prologue = [
            PipelineStageDemand(
                "load_q", "memory:global_read",
                q_bytes_per_cta * read_scale,
                True, predecessors=()),
            PipelineStageDemand(
                "q_smem_write", "resource:smem",
                q_bytes_per_cta * smem_scale,
                predecessors=("load_q",)),
        ]
        # The Q64 source closes the first K and V copies before their shared
        # buffers are consumed.  Each CTA starts a new Q row tile, so neither
        # request can inherit an older CTA's in-flight panel stream.
        if barrier_service is not None:
            prologue.extend((
                PipelineStageDemand(
                    "q_ready_barrier", barrier_service, barrier_fraction,
                    predecessors=("q_smem_write",)),
                PipelineStageDemand(
                    "q_smem_read", "resource:smem",
                    q_bytes_per_cta * smem_scale,
                    predecessors=("q_ready_barrier",)),
                PipelineStageDemand(
                    "q_reuse_barrier", barrier_service, barrier_fraction,
                    predecessors=("q_smem_read",)),
                PipelineStageDemand(
                    "load_k_0", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    True,
                    predecessors=("q_reuse_barrier",)),
            ))
        else:
            prologue.extend((
            PipelineStageDemand(
                "load_k_0", "memory:global_read",
                kv_bytes_per_step * read_scale,
                predecessors=("q_smem_write",)),
            PipelineStageDemand(
                "q_smem_read", "resource:smem",
                q_bytes_per_cta * smem_scale,
                predecessors=("load_k_0",)),
            ))
        prologue.extend((
            PipelineStageDemand(
                "load_v_0", "memory:global_read",
                kv_bytes_per_step * read_scale,
                True,
                predecessors=("q_smem_read",)),
            PipelineStageDemand(
                "stage_k_0", "resource:smem",
                2.0 * kv_bytes_per_step * smem_scale,
                predecessors=("load_v_0",)),
        ))
        prologue = tuple(prologue)

        def _panel(*, phase_name, first, prefetch_next):
            stages = []
            qk = f"qk_{phase_name}"
            stage_v = f"stage_v_{phase_name}"
            softmax = f"softmax_{phase_name}"
            pv = f"pv_{phase_name}"
            qk_predecessors = ()
            qk_prior_resources = (
                "memory:global_read", "resource:smem")
            if not first:
                stages.append(PipelineStageDemand(
                    f"load_v_{phase_name}", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=(),
                    prior_resources=(
                        "resource:smem", "compute:compute_tensor")))
                qk_predecessors = (f"load_v_{phase_name}",)
                qk_prior_resources = ()
            if barrier_service is not None:
                ready = f"ready_{phase_name}"
                stages.append(PipelineStageDemand(
                    ready, barrier_service, barrier_fraction,
                    predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources))
                qk_predecessors = (ready,)
                qk_prior_resources = ()
            stages.extend((
                PipelineStageDemand(
                    qk, "compute:compute_tensor",
                    tensor_step_fraction,
                    predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources),
                PipelineStageDemand(
                    stage_v, "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=(qk,)),
            ))
            softmax_predecessors = (stage_v,)
            if barrier_service is not None:
                consumed = f"consumed_{phase_name}"
                stages.append(PipelineStageDemand(
                    consumed, barrier_service, barrier_fraction,
                    predecessors=(stage_v,)))
                softmax_predecessors = (consumed,)
            load_k_next = None
            if prefetch_next:
                load_k_next = "load_k_next"
                stages.append(PipelineStageDemand(
                    load_k_next, "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=(stage_v,)))
            softmax_step_operations = float(
                64 * 64 + (0 if first else 64))
            stages.extend((
                PipelineStageDemand(
                    softmax, softmax_service,
                    softmax_step_operations / softmax_operations,
                    predecessors=softmax_predecessors,
                    start_predecessors=(
                        () if load_k_next is None else (load_k_next,))),
                PipelineStageDemand(
                    pv, "compute:compute_tensor",
                    tensor_step_fraction,
                    predecessors=(softmax, stage_v)),
            ))
            if load_k_next is not None:
                stages.append(PipelineStageDemand(
                    "stage_k_next", "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=(load_k_next, pv)))
            return tuple(stages)

        phases = [AttentionPipelinePhaseDemand(
            name=(f"metax_q64_cta_rows{item.query_rows}_"
                  f"steps{item.step_count}"),
            stages=prologue)]
        if step_count == 1:
            phases.append(AttentionPipelinePhaseDemand(
                name="first_tail_panel",
                stages=_panel(
                    phase_name="0", first=True, prefetch_next=False)))
        else:
            phases.append(AttentionPipelinePhaseDemand(
                name="first_panel",
                stages=_panel(
                    phase_name="0", first=True, prefetch_next=True)))
            if step_count > 2:
                phases.append(AttentionPipelinePhaseDemand(
                    name="steady_panel",
                    stages=_panel(
                        phase_name="steady", first=False,
                        prefetch_next=True),
                    repeat_count=step_count - 2))
            phases.append(AttentionPipelinePhaseDemand(
                name="tail_panel",
                stages=_panel(
                    phase_name="tail", first=False,
                    prefetch_next=False)))
        epilogue = []
        output_predecessors = ()
        output_prior_resources = ("compute:compute_tensor",)
        if barrier_service is not None:
            epilogue.append(PipelineStageDemand(
                "output_ready_barrier", barrier_service, barrier_fraction,
                predecessors=(),
                prior_resources=("compute:compute_tensor",)))
            output_predecessors = ("output_ready_barrier",)
            output_prior_resources = ()
        epilogue.append(
            PipelineStageDemand(
                "output_smem", "resource:smem",
                2.0 * staged_output_bytes_per_cta * smem_scale,
                predecessors=output_predecessors,
                prior_resources=output_prior_resources))
        store_predecessors = ("output_smem",)
        if barrier_service is not None:
            epilogue.append(PipelineStageDemand(
                "output_visible_barrier", barrier_service, barrier_fraction,
                predecessors=("output_smem",)))
            store_predecessors = ("output_visible_barrier",)
        epilogue.append(PipelineStageDemand(
            "store", "memory:global_write",
            output_bytes_per_cta * write_scale,
            True, predecessors=store_predecessors))
        phases.append(AttentionPipelinePhaseDemand(
            name="epilogue", stages=tuple(epilogue)))
        sequence_templates[identity] = tuple(phases)
        return sequence_templates[identity]

    resident_capacity = worker_count * resident_cta_slots
    if active_cta_count > resident_capacity:
        # A fixed CTA-to-AP identity is an assumption this path declares
        # rather than a measurement (see the
        # ``metax_full_ap_round_robin_cta_assignment_assumed`` limitation
        # below), and for this launch shape it is not a harmless one.  Each
        # request contributes ``scheduled_q_tiles * query_heads`` consecutive
        # CTAs whose KV depth grows with the query tile.  When that block is
        # a whole multiple of the AP count, lane ``L`` receives the same tile
        # depths from every request instead of a rotating mix, so the lanes
        # holding the deepest tiles are charged the deepest tile of every
        # request: the modelled busiest lane runs 10x the median for the
        # HQ16/13-tile launches (208 = 2 x 104) and 4x for HQ32/13-tile
        # (416 = 4 x 104), while every launch whose block does not divide
        # the AP count stays within 1.1-1.8x.  The three aliasing launches
        # are exactly the three most over-predicted in this family.
        #
        # Once the active population exceeds the resident window the AP work
        # distributor must refill from a shared backlog, and a freed AP takes
        # whichever CTA is next rather than one fixed by a static mapping.
        # Keep the measured resident prefix in source order and reduce only
        # the queued remainder to its CTA-count- and KV-panel-work-preserving
        # balance -- the same treatment the SM80 varlen admission already
        # gets.  Padding tiles carry no phases and are retained so the CTA
        # envelope is conserved.
        active_entries = tuple(
            (item, count) for item, count in schedule_class_counts
            if item is not None)
        empty_entries = tuple(
            (item, count) for item, count in schedule_class_counts
            if item is None)
        schedule_class_counts = _source_prefix_with_balanced_queue_tail(
            active_entries, resident_capacity) + empty_entries
        source_repeat_count = 1
    else:
        schedule_class_counts, source_repeat_count = (
            _compact_exact_source_period(schedule_class_counts))
    classes = tuple(
        AttentionPipelineClassDemand(
            name=(
                f"run{run_index}_empty"
                if item is None else
                f"run{run_index}_rows{item.query_rows}_"
                f"steps{item.step_count}"),
            count=count,
            query_rows=(0 if item is None else item.query_rows),
            panel_count=(0 if item is None else item.step_count),
            phases=(() if item is None else _cta_phases(item)),
        )
        for run_index, (item, count) in enumerate(schedule_class_counts))
    plan = AttentionPipelineEnvelopeDemand(
        active_lane_count=min(active_cta_count, worker_count),
        classes=classes,
        source_repeat_count=source_repeat_count,
        buffers=(PipelineBuffer(
            name="resident_cta_lifecycle",
            producer_stage="load_q",
            consumer_stage="store",
            slots=(
                resident_cta_slots),
            scope="worker",
        ),),
    )
    limitations = tuple(
        value for value in inp.limitations
        if value not in {
            "gpu_flash_attention_cta_worker_mapping_unavailable",
            "metax_q64_empty_cta_issue_lifecycle_unavailable",
        })
    limitations += ("metax_full_ap_round_robin_cta_assignment_assumed",)
    # A padded query tile is a CTA that starts, reads the two integers
    # describing its tile, finds the tile beyond its own request, and returns
    # before the Q/K/V path.  The active-work envelope above gives it no
    # phases, i.e. exactly zero time, which is only correct if an AP retires
    # it for free.  The published probe measures that it does not: it is the
    # marginal launch-wide cost of one such CTA at this source's own
    # 256-thread / 32 KiB footprint.  Those CTAs occupy their AP's resident
    # slots in series with the real work, so the cost adds to the launch.
    early_exit_retire = hardware.early_exit_cta_retire
    early_exit_cta_retire_ns = None
    if early_exit_retire is not None and empty_cta_count:
        if kernel.launch.block_size != early_exit_retire["threads_per_cta"]:
            raise ValueError(
                "MetaX Q64 early-exit retire evidence was measured at a "
                "different block size than this launch")
        early_exit_cta_retire_ns = (
            empty_cta_count
            * early_exit_retire["launch_wide_ns_per_early_exit_cta"])
    else:
        limitations += ("metax_q64_padded_tile_retire_unpriced",)
    return attach_attention_pipeline_envelope(
        dataclasses.replace(
            inp,
            support_level="partial",
            limitations=limitations,
            early_exit_cta_retire_ns=early_exit_cta_retire_ns,
        ),
        hardware,
        plan,
    )


def _attach_metax_contiguous_decode_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the fixed BlockM64/128 by BlockN64 GQA decode CTA program.

    One direct CTA owns one ``(request, KV head)`` pair and therefore all GQA
    query heads mapped to that KV head.  The exact program executes complete
    64-column Tensor/softmax panels, while its global Q/O traffic remains
    predicated to the live GQA rows.  The source dependency order is the same
    Q/K/V -> QK -> online-softmax -> PV -> output chain used by the pinned
    MACA forward kernel.  Compiler resource reports fix the resident CTA slots;
    the common attention envelope charges panel first-request latency once at
    cold fill and retains only the analytically exposed resident tail.
    """

    PipelineStageDemand = bind_pipeline_stage_template
    is_q64 = _has_metax_q64_contiguous_decode_pipeline(kernel)
    is_q128 = _has_metax_q128_contiguous_decode_pipeline(kernel)
    if is_q64 == is_q128:
        raise ValueError(
            "MetaX decode pipeline requires exactly one pinned contiguous "
            "source trait")
    tile_q = 64 if is_q64 else 128
    mma_per_kv_step = (
        _METAX_372_Q64_DECODE_MMA_PER_KV_STEP
        if is_q64 else _METAX_372_FA_MMA_PER_FULL_KV_TILE)
    resident_cta_slots = (
        _METAX_372_Q64_DECODE_RESIDENT_CTA_SLOTS_PER_AP
        if is_q64 else _METAX_372_Q128_DECODE_RESIDENT_CTA_SLOTS_PER_AP)
    batch, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    if inp.active_units <= 0:
        raise ValueError("MetaX Q64 decode pipeline requires active APs")
    query_heads_per_cta = query_heads // kv_heads
    element_bytes = int(kernel.shape["dtype_bytes"])
    q_bytes_per_cta = float(
        query_heads_per_cta * head_dim * element_bytes)
    output_bytes_per_cta = q_bytes_per_cta
    staged_output_bytes_per_cta = q_bytes_per_cta
    kv_bytes_per_step = float(64 * head_dim * element_bytes)
    if len(set(kv_lens)) != 1:
        raise ValueError(
            "MetaX Q64 homogeneous decode envelope requires one KV length")
    step_count = math.ceil(kv_lens[0] / 64)
    active_cta_count = batch * kv_heads
    if active_cta_count != kernel.launch.grid_size:
        raise ValueError(
            "MetaX Q64 decode source launch disagrees with grid: "
            f"source={active_cta_count}, grid={kernel.launch.grid_size}")
    if step_count <= 0:
        raise ValueError("MetaX Q64 decode produced an empty CTA")

    total_steps = active_cta_count * step_count
    read_bytes = float(
        inp.demand_per_resource["l2_cache"] - _output_bytes(kernel, hardware))
    write_bytes = _output_bytes(kernel, hardware)
    expected_read_bytes = float(
        active_cta_count * q_bytes_per_cta
        + total_steps * 2.0 * kv_bytes_per_step)
    if has_dense_flash_attention_source(kernel):
        source = dense_attention_prediction_work(kernel,hardware)
        # The source owns exactly one (request, KV-head) CTA with equal GQA
        # rows in this uniform Q1 domain; these are lawful CTA projections.
        staged_output_bytes_per_cta = source['output']/active_cta_count
        output_bytes_per_cta = _output_bytes(kernel,hardware)/active_cta_count
    expected_write_bytes = float(active_cta_count * output_bytes_per_cta)
    if not math.isclose(
            read_bytes, expected_read_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 decode DAG read bytes disagree with aggregate "
            f"traffic: dag={expected_read_bytes}, aggregate={read_bytes}")
    if not math.isclose(
            write_bytes, expected_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 decode DAG write bytes disagree with aggregate "
            f"traffic: dag={expected_write_bytes}, aggregate={write_bytes}")
    smem_bytes = float(inp.demand_per_resource["smem"])
    expected_smem_bytes = 2.0 * (read_bytes + active_cta_count*staged_output_bytes_per_cta)
    if not math.isclose(
            smem_bytes, expected_smem_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 decode DAG shared-memory bytes disagree with "
            f"aggregate traffic: dag={expected_smem_bytes}, "
            f"aggregate={smem_bytes}")

    tensor_operations = float(inp.demand_per_resource["compute_tensor"])
    tensor_operations_per_step = float(
        mma_per_kv_step
        * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    expected_tensor_operations = total_steps * tensor_operations_per_step
    if not math.isclose(
            tensor_operations, expected_tensor_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 decode Tensor work disagrees with aggregate demand: "
            f"dag={expected_tensor_operations}, "
            f"aggregate={tensor_operations}")
    softmax_operations = float(inp.demand_per_resource["compute_sfu"])
    expected_softmax_operations = float(
        active_cta_count
        * (step_count * tile_q * 64
           + max(0, step_count - 1) * tile_q))
    if not math.isclose(
            softmax_operations, expected_softmax_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q64 decode softmax work disagrees with aggregate demand")

    # A 256-thread Q64 CTA contributes exactly four 64-thread waves.  When
    # there is at most one CTA per active AP, use the independently measured
    # four-wave MMA envelope instead of the 32-wave architectural peak.  Once
    # the launch has multiple resident CTAs/AP, this single endpoint no longer
    # identifies their mixed occupancy and the normal peak lower bound stays
    # in force rather than interpolating from production durations.
    if kernel.launch.grid_size <= hardware.num_compute_units:
        tensor_wave_efficiency = (
            _METAX_372_BF16_MMA_FLOPS_PER_AP_CYCLE_4_WAVES
            / _METAX_372_FA_TENSOR_OPS_PER_AP_CYCLE)
        adjusted_demands = []
        for demand in inp.compute_demands:
            if demand.resource_role == "compute_tensor":
                demand = dataclasses.replace(
                    demand,
                    schedule_efficiency=(
                        demand.schedule_efficiency
                        * tensor_wave_efficiency),
                )
            adjusted_demands.append(demand)
        inp = dataclasses.replace(
            inp,
            compute_demands=tuple(adjusted_demands),
        )

    read_scale = 1.0 / read_bytes
    write_scale = 1.0 / write_bytes
    smem_scale = 1.0 / smem_bytes
    tensor_step_fraction = (
        0.5 * tensor_operations_per_step / tensor_operations)
    softmax_service = pipeline_compute_service(
        inp, hardware, roles=("compute_sfu",))
    phase_templates = {}

    def _cta_phases(step_count, include_cold_start_latency):
        """Return the fixed source program, analytically repeated by panels.

        The MACA Q64/Q128 decode source has one prologue, a first panel, a
        repeated steady panel, a tail panel, and an epilogue.  Representing
        each panel as a newly named DAG stage replays source control flow in
        Python and makes prediction cost grow with KV length.  The shared
        attention-envelope solver instead composes this fixed template as
        exact max-plus transitions.  ``prior_resources`` carries the only
        cross-phase edges: the K/V shared-buffer handoff and the prior panel's
        Tensor completion.  Thus this is schedule-equivalent to the expanded
        DAG, not a throughput approximation.
        """

        identity = (step_count, include_cold_start_latency)
        if identity in phase_templates:
            return phase_templates[identity]
        prologue = AttentionPipelinePhaseDemand(
            name="prologue",
            stages=(
                PipelineStageDemand(
                    "load_q", "memory:global_read",
                    q_bytes_per_cta * read_scale,
                    include_cold_start_latency, predecessors=()),
                PipelineStageDemand(
                    "stage_q", "resource:smem",
                    q_bytes_per_cta * smem_scale,
                    predecessors=("load_q",)),
                PipelineStageDemand(
                    "load_k", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=("stage_q",)),
                PipelineStageDemand(
                    "read_q", "resource:smem",
                    q_bytes_per_cta * smem_scale,
                    predecessors=("load_k",)),
                PipelineStageDemand(
                    "load_v", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=("read_q",)),
                PipelineStageDemand(
                    "stage_k", "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=("load_v",)),
            ),
        )

        def _panel(*, name, first, prefetch_next, repeat_count=1):
            stages = []
            if not first:
                # The preceding panel's ``stage_k`` waits for both its
                # prefetched K load and its PV Tensor completion.  At a phase
                # boundary that complete source edge is the saved SMEM state.
                stages.append(PipelineStageDemand(
                    "load_v", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=(),
                    prior_resources=("resource:smem",)))
                qk_predecessors = ("load_v",)
            else:
                qk_predecessors = ()
            stages.extend((
                PipelineStageDemand(
                    "qk", "compute:compute_tensor",
                    tensor_step_fraction,
                    predecessors=qk_predecessors,
                    # The prologue publishes ``load_v`` and ``stage_k`` on
                    # these resource states.  Later panels retain ``stage_k``
                    # from the previous panel for the same reason.
                    prior_resources=(
                        "memory:global_read", "resource:smem")),
                PipelineStageDemand(
                    "stage_v", "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=("qk",)),
            ))
            if prefetch_next:
                stages.append(PipelineStageDemand(
                    "load_k_next", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=("stage_v",)))
            stages.extend((
                PipelineStageDemand(
                    "softmax", softmax_service,
                    float(tile_q * 64 + (0 if first else tile_q))
                    / softmax_operations,
                    predecessors=("stage_v",),
                    start_predecessors=(
                        ("load_k_next",) if prefetch_next else ())),
                PipelineStageDemand(
                    "pv", "compute:compute_tensor",
                    tensor_step_fraction,
                    predecessors=("softmax", "stage_v")),
            ))
            if prefetch_next:
                stages.append(PipelineStageDemand(
                    "stage_k_next", "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=("load_k_next", "pv")))
            return AttentionPipelinePhaseDemand(
                name=name,
                stages=tuple(stages),
                repeat_count=repeat_count,
            )

        if step_count == 1:
            panels = (_panel(
                name="first_tail_panel", first=True,
                prefetch_next=False),)
        else:
            panels = [_panel(
                name="first_panel", first=True, prefetch_next=True)]
            if step_count > 2:
                panels.append(_panel(
                    name="steady_panel", first=False, prefetch_next=True,
                    repeat_count=step_count - 2))
            panels.append(_panel(
                name="tail_panel", first=False, prefetch_next=False))
            panels = tuple(panels)
        epilogue = AttentionPipelinePhaseDemand(
            name="epilogue",
            stages=(
                PipelineStageDemand(
                    "stage_output", "resource:smem",
                    2.0 * staged_output_bytes_per_cta * smem_scale,
                    predecessors=(),
                    prior_resources=("compute:compute_tensor",)),
                PipelineStageDemand(
                    "store", "memory:global_write",
                    output_bytes_per_cta * write_scale,
                    True, predecessors=("stage_output",)),
            ),
        )
        phases = (prologue,) + panels + (epilogue,)
        phase_templates[identity] = phases
        return phases

    worker_count = min(active_cta_count, inp.active_units)
    cold_cta_count = worker_count
    warm_cta_count = active_cta_count - cold_cta_count
    classes = [AttentionPipelineClassDemand(
        name=f"cold_steps{step_count}",
        count=cold_cta_count,
        query_rows=query_heads_per_cta,
        panel_count=step_count,
        phases=_cta_phases(step_count, True),
    )]
    if warm_cta_count:
        classes.append(AttentionPipelineClassDemand(
            name=f"warm_steps{step_count}",
            count=warm_cta_count,
            query_rows=query_heads_per_cta,
            panel_count=step_count,
            phases=_cta_phases(step_count, False),
        ))
    plan = AttentionPipelineEnvelopeDemand(
        active_lane_count=worker_count,
        classes=tuple(classes),
        buffers=(PipelineBuffer(
            name="resident_cta_lifecycle",
            producer_stage="load_q",
            consumer_stage="store",
            slots=resident_cta_slots,
            scope="worker",
        ),),
        resident_tail_latency_service="memory:global_read",
        resident_tail_requests_per_panel=2,
    )
    limitations = tuple(
        value for value in inp.limitations
        if value != "gpu_flash_attention_cta_worker_mapping_unavailable")
    limitations += ("metax_full_ap_round_robin_cta_assignment_assumed",)
    return attach_attention_pipeline_envelope(
        dataclasses.replace(
            inp,
            support_level="partial",
            limitations=limitations,
        ),
        hardware,
        plan,
    )


def _attach_metax_tile16_decode_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the common Attention stage DAG to the C500 tile16 program.

    The pinned MACA 3.7.2 source/counter contract fixes one 64-thread CTA for
    each ``(split, request, KV head)`` tuple.  Unsplit paged launches encode
    those axes as ``(KV head, request, 1)``; split contiguous launches encode
    them as ``(1, request * KV head, split)``.  Hardware-specific work remains
    the tile16 MMA/MTE demand already computed above; this function only adds
    the source-mandatory ``Q/K/V load -> QK -> softmax -> PV -> output``
    dependencies and the finite one-wave CTA residency ceiling.
    """

    # The C500 source adapter contributes its program topology; its dynamic
    # fractions bind through the same cross-hardware template path as A800.
    PipelineStageDemand = bind_pipeline_stage_template

    if not _is_source_bound_metax_tile16_program(kernel):
        raise ValueError(
            "MetaX tile16 pipeline requires its pinned decode program")
    if _work_distribution(kernel) != "direct_cta":
        raise ValueError("MetaX tile16 pipeline requires a direct CTA launch")
    if inp.active_units <= 0:
        raise ValueError("MetaX tile16 pipeline requires active APs")

    _, _, _, head_dim, query_heads, kv_heads = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    splits = _splits(kernel)
    if any(length != 1 for length in q_lens):
        raise ValueError("MetaX tile16 decode requires one query per request")
    if query_heads % kv_heads:
        raise ValueError("MetaX tile16 GQA heads must divide exactly")
    query_heads_per_cta = query_heads // kv_heads
    element_bytes = int(kernel.shape["dtype_bytes"])
    q_bytes_per_cta = float(
        query_heads_per_cta * head_dim * element_bytes)
    output_width_bytes = (
        head_dim * element_bytes
        if splits == 1 else (head_dim + 1) * 4)
    output_bytes_per_cta = float(
        query_heads_per_cta * output_width_bytes)
    page_table_bytes_per_step = float(kernel.launch.parameters[
        "page_table_bytes_per_kv_panel"])

    # Build the source-linear launch order.  The exact grid-axis contract was
    # checked above; tail panels retain only in-range global bytes while their
    # MMA and softmax stages execute the complete 16x16 tile.
    request_split_work = []
    for kv_length in kv_lens:
        kv_tiles = math.ceil(kv_length / 16)
        tiles_per_split = math.ceil(kv_tiles / splits)
        split_counts = _split_tile_counts(
            kv_tiles, splits, include_empty=True)
        split_work = []
        for split, step_count in enumerate(split_counts):
            first_row = split * tiles_per_split * 16
            kv_rows = min(
                max(0, kv_length - first_row), step_count * 16)
            split_work.append((step_count, kv_rows))
        request_split_work.append(tuple(split_work))
    layout = kernel.attributes["kv_layout"]
    resident_cta_slots = (
        _METAX_372_TILE16_D64_RESIDENT_CTA_SLOTS_PER_AP
        if head_dim == 64
        else _METAX_372_TILE16_D128_RESIDENT_CTA_SLOTS_PER_AP)
    source_class_counts = {}
    cold_class_counts = {}
    resident_fill_class_counts = {}
    remaining_cold = min(kernel.launch.grid_size, inp.active_units)
    remaining_resident_fill = (
        max(
            0,
            min(
                kernel.launch.grid_size,
                inp.active_units * resident_cta_slots,
            ) - remaining_cold,
        )
        if (layout == "contiguous" and query_heads_per_cta > 1) else 0)
    for split in range(splits):
        for request in range(len(kv_lens)):
            identity = request_split_work[request][split]
            source_class_counts[identity] = (
                source_class_counts[identity] + kv_heads
                if identity in source_class_counts else kv_heads)
            cold_count = min(kv_heads, remaining_cold)
            if cold_count:
                cold_class_counts[identity] = (
                    cold_class_counts[identity] + cold_count
                    if identity in cold_class_counts else cold_count)
                remaining_cold -= cold_count
            resident_fill_count = min(
                kv_heads - cold_count, remaining_resident_fill)
            if resident_fill_count:
                resident_fill_class_counts[identity] = (
                    resident_fill_class_counts[identity]
                    + resident_fill_count
                    if identity in resident_fill_class_counts
                    else resident_fill_count)
                remaining_resident_fill -= resident_fill_count
    active_cta_count = sum(source_class_counts.values())
    if active_cta_count != kernel.launch.grid_size:
        raise ValueError(
            "MetaX tile16 source launch order disagrees with grid: "
            f"source={active_cta_count}, grid={kernel.launch.grid_size}")
    if any((step_count == 0 and kv_rows != 0)
           or (step_count > 0 and kv_rows <= 0)
           for step_count, kv_rows in source_class_counts):
        raise ValueError(
            "MetaX tile16 source split extent is inconsistent")

    total_steps = sum(
        count * step_count
        for (step_count, _), count in source_class_counts.items())
    read_bytes = float(
        inp.demand_per_resource["l2_cache"] - _output_bytes(kernel, hardware))
    write_bytes = _output_bytes(kernel, hardware)
    expected_read_bytes = math.fsum(
        count * (
            (q_bytes_per_cta if step_count > 0 else 0.0)
            + 2.0 * kv_rows * head_dim * element_bytes
            + step_count * page_table_bytes_per_step)
        for (step_count, kv_rows), count in source_class_counts.items())
    staged_output_bytes_per_cta = output_bytes_per_cta
    if has_dense_flash_attention_source(kernel):
        source = dense_attention_prediction_work(kernel,hardware)
        staged_output_bytes_per_cta = source['output']/active_cta_count
        output_bytes_per_cta = _output_bytes(kernel,hardware)/active_cta_count
    elif has_fa2_split_source(kernel):
        source = current_fa2_split_source_work(kernel)
        # Every (split, request, KV-head) CTA publishes the same logical
        # partial extent, including the source's empty-split constants.
        staged_output_bytes_per_cta = source['write_bytes']/active_cta_count
        output_bytes_per_cta = source['priced_write_bytes']/active_cta_count
    expected_write_bytes = active_cta_count * output_bytes_per_cta
    if not math.isclose(
            read_bytes, expected_read_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX tile16 source DAG read bytes disagree with aggregate "
            f"traffic: dag={expected_read_bytes}, aggregate={read_bytes}")
    if not math.isclose(
            write_bytes, expected_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX tile16 source DAG write bytes disagree with aggregate "
            f"traffic: dag={expected_write_bytes}, aggregate={write_bytes}")
    smem_bytes = float(inp.demand_per_resource["smem"])
    empty_cta_count = (
        source_class_counts[(0, 0)] if (0, 0) in source_class_counts else 0)
    page_table_bytes = total_steps * page_table_bytes_per_step
    expected_smem_bytes = 2.0 * (
        read_bytes + active_cta_count*staged_output_bytes_per_cta
        - page_table_bytes
        - empty_cta_count * staged_output_bytes_per_cta)
    if not math.isclose(
            smem_bytes, expected_smem_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX tile16 source DAG shared-memory bytes disagree with "
            f"aggregate traffic: dag={expected_smem_bytes}, "
            f"aggregate={smem_bytes}")

    tensor_operations = float(inp.demand_per_resource["compute_tensor"])
    mma_per_step = (
        _METAX_372_TILE16_D64_MMA_PER_KV_STEP
        if head_dim == 64 else _METAX_372_TILE16_MMA_PER_KV_STEP)
    expected_tensor_operations = float(
        total_steps * mma_per_step
        * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    if not math.isclose(
            tensor_operations, expected_tensor_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX tile16 source DAG Tensor work disagrees with aggregate "
            "demand")
    softmax_operations = float(inp.demand_per_resource["compute_sfu"])
    expected_softmax_operations = float(sum(
        count * (
            step_count * 16 * 16 + max(0, step_count - 1) * 16)
        for (step_count, _), count in source_class_counts.items()))
    if not math.isclose(
            softmax_operations, expected_softmax_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX tile16 source DAG softmax work disagrees with aggregate "
            "demand")
    # The tile16 block contains one 64-thread wave.  Price its Tensor work on
    # the independently measured register-only BF16 MMA active-wave curve,
    # rather than scaling the 32-wave architectural peak by an ideal lane-fill
    # ratio.  The curve uses the exact one-wave source block and contains no
    # production duration.
    peu_count_per_ap = (
        hardware.resource_by_role("compute_tensor").total_units
        // hardware.num_compute_units)
    if peu_count_per_ap != 4:
        raise ValueError(
            "MetaX tile16 pipeline requires four physical PEUs per AP")
    active_waves_per_ap = kernel.launch.grid_size / inp.active_units
    tensor_capacity_per_ap = _metax_bf16_mma_capacity_for_active_waves(
        active_waves_per_ap)
    architectural_tensor_capacity_per_ap = (
        _METAX_372_FA_TENSOR_OPS_PER_AP_CYCLE)
    tensor_wave_efficiency = (
        tensor_capacity_per_ap / architectural_tensor_capacity_per_ap)
    active_wave_slots = inp.active_units * peu_count_per_ap
    ideal_vector_wave_fill = (
        min(kernel.launch.grid_size, active_wave_slots) / active_wave_slots)
    efficiency_by_role = {
        "compute_tensor": tensor_wave_efficiency,
        "compute_sfu": ideal_vector_wave_fill,
        "warp_issue": ideal_vector_wave_fill,
    }
    adjusted_demands = []
    for demand in inp.compute_demands:
        if demand.resource_role not in efficiency_by_role:
            adjusted_demands.append(demand)
            continue
        adjusted_demands.append(dataclasses.replace(
            demand,
            schedule_efficiency=(
                demand.schedule_efficiency
                * efficiency_by_role[demand.resource_role]),
        ))
    inp = dataclasses.replace(
        inp,
        compute_demands=tuple(adjusted_demands),
    )
    tensor_step_fraction = (
        0.5 * mma_per_step
        * _METAX_372_FA_TENSOR_OPS_PER_MMA
        / tensor_operations)
    softmax_service = pipeline_compute_service(
        inp, hardware, roles=("compute_sfu",))
    read_scale = 1.0 / read_bytes
    write_scale = 1.0 / write_bytes
    smem_scale = 1.0 / smem_bytes
    sequence_templates = {}
    paged_page_templates = {}
    operand_completion_ns = None
    if (layout == "contiguous"
            and kernel.launch.grid_size == inp.active_units):
        completion_panel_counts = {
            step_count for step_count, _ in source_class_counts}
        if len(completion_panel_counts) == 1:
            completion_panel_count, = completion_panel_counts
            tile_shape = kernel.launch.tiling.tile_shape
            query_heads_per_workgroup = None
            if "query_heads_per_workgroup" in tile_shape:
                query_heads_per_workgroup = tile_shape[
                    "query_heads_per_workgroup"]
            operand_completion_ns = (
                hardware.metax_fa_tile16_operand_completion_time_ns(
                    inp.active_units, completion_panel_count, head_dim,
                    query_heads_per_workgroup))
    use_finite_operand_completion = operand_completion_ns is not None
    operand_completion_s = (
        operand_completion_ns * 1.0e-9
        if use_finite_operand_completion else None)
    operand_completion_service = (
        "mechanism:metax_tile16_operand_completion")

    def _cta_phases(
            step_count, kv_rows, include_cold_start_latency,
            include_panel_first_request_latency,
            include_resident_buffer_fill_latency):
        identity = (
            step_count, kv_rows, include_cold_start_latency,
            include_panel_first_request_latency,
            include_resident_buffer_fill_latency)
        if identity in sequence_templates:
            template = sequence_templates[identity]
        else:
            if step_count == 0:
                if kv_rows != 0:
                    raise ValueError(
                        "MetaX tile16 empty split has nonzero KV rows")
                # The source checks the split's K range before it constructs
                # Q/K/V fragments.  It directly writes the zeroed FP32
                # partial-output rectangle, then the FP32 -inf LSE scalar;
                # retain that order so this CTA holds a resident slot even
                # though it performs no SMEM or MMA work.
                partial_bytes = float(
                    query_heads_per_cta * head_dim * 4)
                lse_bytes = float(query_heads_per_cta * 4)
                if not math.isclose(
                        partial_bytes + lse_bytes,
                        output_bytes_per_cta, rel_tol=0.0, abs_tol=0.0):
                    raise AssertionError(
                        "MetaX tile16 empty split output does not close")
                template = (AttentionPipelinePhaseDemand(
                    name="empty_split_exit",
                    stages=(
                        PipelineStageDemand(
                            "cta_start", "memory:global_write",
                            partial_bytes * write_scale,
                            include_cold_start_latency, predecessors=()),
                        PipelineStageDemand(
                            "store", "memory:global_write",
                            lse_bytes * write_scale,
                            False, predecessors=("cta_start",)),
                    ),
                ),)
                sequence_templates[identity] = template
                return template
            tail_rows = kv_rows - 16 * (step_count - 1)
            if not 0 < tail_rows <= 16:
                raise ValueError(
                    "MetaX tile16 source DAG produced an invalid KV tail")
            full_panel_bytes = float(16 * head_dim * element_bytes)
            tail_panel_bytes = float(tail_rows * head_dim * element_bytes)
            first_panel_bytes = (
                tail_panel_bytes if step_count == 1 else full_panel_bytes)
            prologue = (
                PipelineStageDemand(
                    "cta_start", "memory:global_read",
                    q_bytes_per_cta * read_scale,
                    include_cold_start_latency, predecessors=()),
                PipelineStageDemand(
                    "stage_q", "resource:smem",
                    q_bytes_per_cta * smem_scale,
                    predecessors=("cta_start",)),
                *(() if page_table_bytes_per_step == 0.0 else (
                    PipelineStageDemand(
                        "load_page_0", "memory:global_read",
                        page_table_bytes_per_step * read_scale,
                        include_cold_start_latency, predecessors=()),
                )),
                PipelineStageDemand(
                    "load_k_0", "memory:global_read",
                    first_panel_bytes * read_scale,
                    (False if use_finite_operand_completion else (
                        include_cold_start_latency
                        or include_panel_first_request_latency
                        or include_resident_buffer_fill_latency)),
                    predecessors=(
                        (() if use_finite_operand_completion
                         else ("stage_q",))
                        + (() if page_table_bytes_per_step == 0.0
                           else ("load_page_0",)))),
                PipelineStageDemand(
                    "read_q", "resource:smem",
                    q_bytes_per_cta * smem_scale,
                    predecessors=(
                        "stage_q",) if use_finite_operand_completion
                    else ("load_k_0",)),
                PipelineStageDemand(
                    "load_v_0", "memory:global_read",
                    first_panel_bytes * read_scale,
                    (False if use_finite_operand_completion else (
                        include_cold_start_latency
                        or include_panel_first_request_latency
                        or include_resident_buffer_fill_latency)),
                    predecessors=(
                        () if use_finite_operand_completion
                        else ("read_q",))),
                PipelineStageDemand(
                    "stage_k_0", "resource:smem",
                    2.0 * first_panel_bytes * smem_scale,
                    predecessors=(
                        ("read_q",) if use_finite_operand_completion
                        else ("load_v_0",))),
            )

            def _panel(
                    *, phase_name, first, prefetch_next,
                    current_bytes, next_bytes,
                    include_first_v_request_latency=False,
                    include_next_k_request_latency=False):
                stages = []
                qk = f"qk_{phase_name}"
                stage_v = f"stage_v_{phase_name}"
                softmax = f"softmax_{phase_name}"
                pv = f"pv_{phase_name}"
                qk_predecessors = ()
                qk_prior_resources = (
                    "memory:global_read", "resource:smem")
                if not first:
                    stages.append(PipelineStageDemand(
                        f"load_v_{phase_name}", "memory:global_read",
                        current_bytes * read_scale,
                        (False if use_finite_operand_completion else (
                            include_panel_first_request_latency
                            or include_first_v_request_latency)),
                        predecessors=(),
                        prior_resources=(
                            () if use_finite_operand_completion else (
                                "resource:smem",
                                "compute:compute_tensor"))))
                    if not use_finite_operand_completion:
                        qk_predecessors = (f"load_v_{phase_name}",)
                        qk_prior_resources = ()
                if use_finite_operand_completion:
                    stages.extend((
                        PipelineStageDemand(
                            f"{qk}_capacity", "compute:compute_tensor",
                            tensor_step_fraction,
                            predecessors=()),
                        PipelineStageDemand(
                            qk, operand_completion_service, None,
                            predecessors=(),
                            prior_resources=("resource:smem",),
                            absolute_duration_s=operand_completion_s),
                    ))
                else:
                    stages.append(PipelineStageDemand(
                        qk, "compute:compute_tensor",
                        tensor_step_fraction,
                        predecessors=qk_predecessors,
                        prior_resources=qk_prior_resources))
                stages.append(
                    PipelineStageDemand(
                        stage_v, "resource:smem",
                        2.0 * current_bytes * smem_scale,
                        predecessors=(qk,)),
                )
                load_k_next = None
                if prefetch_next:
                    load_k_next = "load_k_next"
                    stages.append(PipelineStageDemand(
                        load_k_next, "memory:global_read",
                        next_bytes * read_scale,
                        (False if use_finite_operand_completion else (
                            include_panel_first_request_latency
                            or include_next_k_request_latency)),
                        predecessors=(
                            () if use_finite_operand_completion
                            else (stage_v,))))
                softmax_step_operations = float(
                    16 * 16 + (0 if first else 16))
                stages.append(
                    PipelineStageDemand(
                        softmax, softmax_service,
                        softmax_step_operations / softmax_operations,
                        predecessors=(stage_v,),
                        start_predecessors=(
                            () if (load_k_next is None
                                   or use_finite_operand_completion)
                            else (load_k_next,))),
                )
                if use_finite_operand_completion:
                    stages.extend((
                        PipelineStageDemand(
                            f"{pv}_capacity", "compute:compute_tensor",
                            tensor_step_fraction,
                            predecessors=()),
                        PipelineStageDemand(
                            pv, operand_completion_service, None,
                            predecessors=(softmax, stage_v),
                            absolute_duration_s=operand_completion_s),
                    ))
                else:
                    stages.append(
                    PipelineStageDemand(
                        pv, "compute:compute_tensor",
                        tensor_step_fraction,
                        predecessors=(softmax, stage_v)),
                    )
                if load_k_next is not None:
                    stages.append(PipelineStageDemand(
                        "stage_k_next", "resource:smem",
                        2.0 * next_bytes * smem_scale,
                        predecessors=(
                            (pv,) if use_finite_operand_completion
                            else (load_k_next, pv))))
                return tuple(stages)

            phases = [AttentionPipelinePhaseDemand(
                name="prologue", stages=prologue)]
            if layout in ("paged", "paged_varlen"):
                # The source loads a new physical block address every eight
                # 16-token panels.  K is prefetched behind the current panel,
                # while V completes before the next QK consumer.  The first
                # resident CTA on each worker has no older CTA request stream
                # to cover those V completions; later resident CTAs retain
                # only the first K/V completion of each 128-token page.  An
                # eight-panel phase preserves those source dependencies
                # without expanding CTAs or attaching workload-derived time.
                panels_per_page = 8

                def _paged_page(
                        phase_name, panel_start, panel_count,
                        *, first_page, final_page,
                        next_page_first_bytes=None):
                    template_key = (
                        phase_name,
                        panel_count,
                        first_page,
                        final_page,
                        next_page_first_bytes,
                        tail_panel_bytes,
                        include_cold_start_latency,
                    )
                    if template_key in paged_page_templates:
                        return paged_page_templates[template_key]
                    stages = []
                    for local_panel in range(panel_count):
                        panel = panel_start + local_panel
                        current_bytes = (
                            tail_panel_bytes
                            if panel == step_count - 1
                            else full_panel_bytes)
                        first = first_page and local_panel == 0
                        suffix = (
                            str(panel) if first_page
                            else f"{phase_name}_{local_panel}")
                        qk = f"qk_{suffix}"
                        stage_v = f"stage_v_{suffix}"
                        softmax = f"softmax_{suffix}"
                        pv = f"pv_{suffix}"
                        qk_predecessors = ()
                        qk_prior_resources = (
                            "memory:global_read", "resource:smem")
                        if not first:
                            load_v = f"load_v_{suffix}"
                            if local_panel == 0:
                                load_v_predecessors = ()
                                load_v_prior_resources = (
                                    "resource:smem",
                                    "compute:compute_tensor")
                            else:
                                load_v_predecessors = (
                                    f"stage_k_{phase_name}_{local_panel}",)
                                load_v_prior_resources = ()
                            stages.append(PipelineStageDemand(
                                load_v, "memory:global_read",
                                current_bytes * read_scale,
                                include_first_request_latency=(
                                    include_cold_start_latency
                                    or local_panel == 0),
                                predecessors=load_v_predecessors,
                                prior_resources=load_v_prior_resources))
                            qk_predecessors = (load_v,)
                            qk_prior_resources = ()
                        stages.extend((
                            PipelineStageDemand(
                                qk, "compute:compute_tensor",
                                tensor_step_fraction,
                                predecessors=qk_predecessors,
                                prior_resources=qk_prior_resources),
                            PipelineStageDemand(
                                stage_v, "resource:smem",
                                2.0 * current_bytes * smem_scale,
                                predecessors=(qk,)),
                        ))
                        has_next = panel + 1 < step_count
                        load_k_next = None
                        if has_next:
                            load_k_next = (
                                f"load_k_{phase_name}_{local_panel + 1}")
                            next_bytes = (
                                tail_panel_bytes
                                if panel + 1 == step_count - 1
                                else full_panel_bytes)
                            if (local_panel == panel_count - 1
                                    and not final_page
                                    and next_page_first_bytes is not None):
                                next_bytes = next_page_first_bytes
                            load_k_predecessors = (stage_v,)
                            if page_table_bytes_per_step > 0.0:
                                load_page_next = (
                                    f"load_page_{phase_name}_"
                                    f"{local_panel + 1}")
                                stages.append(PipelineStageDemand(
                                    load_page_next, "memory:global_read",
                                    page_table_bytes_per_step * read_scale,
                                    include_first_request_latency=(
                                        local_panel == panel_count - 1
                                        and not final_page),
                                    predecessors=(stage_v,)))
                                load_k_predecessors = (load_page_next,)
                            stages.append(PipelineStageDemand(
                                load_k_next, "memory:global_read",
                                next_bytes * read_scale,
                                include_first_request_latency=(
                                    local_panel == panel_count - 1
                                    and not final_page),
                                predecessors=load_k_predecessors))
                        stages.extend((
                            PipelineStageDemand(
                                softmax, softmax_service,
                                float(16 * 16 + (0 if panel == 0 else 16))
                                / softmax_operations,
                                predecessors=(stage_v,),
                                start_predecessors=(
                                    () if load_k_next is None
                                    else (load_k_next,))),
                            PipelineStageDemand(
                                pv, "compute:compute_tensor",
                                tensor_step_fraction,
                                predecessors=(softmax, stage_v)),
                        ))
                        if load_k_next is not None:
                            stages.append(PipelineStageDemand(
                                f"stage_k_{phase_name}_{local_panel + 1}",
                                "resource:smem",
                                2.0 * next_bytes * smem_scale,
                                predecessors=(load_k_next, pv)))
                    template = tuple(stages)
                    paged_page_templates[template_key] = template
                    return template

                page_count = math.ceil(step_count / panels_per_page)
                first_page_panels = min(step_count, panels_per_page)
                phases.append(AttentionPipelinePhaseDemand(
                    name="first_page",
                    stages=_paged_page(
                        "first_page", 0, first_page_panels,
                        first_page=True, final_page=page_count == 1,
                        next_page_first_bytes=(
                            tail_panel_bytes if step_count == 9
                            else full_panel_bytes))))
                if page_count > 3:
                    phases.append(AttentionPipelinePhaseDemand(
                        name="steady_page",
                        stages=_paged_page(
                            "steady_page", panels_per_page,
                            panels_per_page,
                            first_page=False, final_page=False),
                        repeat_count=page_count - 3))
                if page_count > 2:
                    final_page_panels = (
                        step_count
                        - (page_count - 1) * panels_per_page)
                    phases.append(AttentionPipelinePhaseDemand(
                        name="pre_final_page",
                        stages=_paged_page(
                            "pre_final_page",
                            (page_count - 2) * panels_per_page,
                            panels_per_page,
                            first_page=False, final_page=False,
                            next_page_first_bytes=(
                                tail_panel_bytes
                                if final_page_panels == 1
                                else full_panel_bytes))))
                if page_count > 1:
                    final_page_start = (
                        (page_count - 1) * panels_per_page)
                    phases.append(AttentionPipelinePhaseDemand(
                        name="final_page",
                        stages=_paged_page(
                            "final_page", final_page_start,
                            step_count - final_page_start,
                            first_page=False, final_page=True)))
            elif step_count == 1:
                phases.append(AttentionPipelinePhaseDemand(
                    name="first_tail_panel",
                    stages=_panel(
                        phase_name="0", first=True, prefetch_next=False,
                        current_bytes=tail_panel_bytes,
                        next_bytes=tail_panel_bytes)))
            else:
                # The source's two K/V shared-buffer slots have independent
                # cold request chains.  A worker's first resident CTA fills
                # both slots before the steady alternating-panel stream can
                # overlap their request latency.
                phases.append(AttentionPipelinePhaseDemand(
                    name="first_panel",
                    stages=_panel(
                        phase_name="0", first=True, prefetch_next=True,
                        current_bytes=full_panel_bytes,
                        next_bytes=(
                            tail_panel_bytes if step_count == 2
                            else full_panel_bytes),
                        include_next_k_request_latency=(
                            include_cold_start_latency
                            or include_resident_buffer_fill_latency))))
                if step_count > 2:
                    phases.append(AttentionPipelinePhaseDemand(
                        name="second_panel",
                        stages=_panel(
                            phase_name="second", first=False,
                            prefetch_next=True,
                            current_bytes=full_panel_bytes,
                            next_bytes=(
                                tail_panel_bytes if step_count == 3
                                else full_panel_bytes),
                            include_first_v_request_latency=(
                                include_cold_start_latency
                                or include_resident_buffer_fill_latency))))
                if step_count > 4:
                    phases.append(AttentionPipelinePhaseDemand(
                        name="steady_panel",
                        stages=_panel(
                            phase_name="steady", first=False,
                            prefetch_next=True,
                            current_bytes=full_panel_bytes,
                            next_bytes=full_panel_bytes),
                        repeat_count=step_count - 4))
                if step_count > 3:
                    phases.append(AttentionPipelinePhaseDemand(
                        name="pre_tail_panel",
                        stages=_panel(
                            phase_name="pre_tail", first=False,
                            prefetch_next=True,
                            current_bytes=full_panel_bytes,
                            next_bytes=tail_panel_bytes)))
                phases.append(AttentionPipelinePhaseDemand(
                    name="tail_panel",
                    stages=_panel(
                        phase_name="tail", first=False,
                        prefetch_next=False,
                        current_bytes=tail_panel_bytes,
                        next_bytes=tail_panel_bytes)))
            epilogue = (
                PipelineStageDemand(
                    "stage_output", "resource:smem",
                    2.0 * staged_output_bytes_per_cta * smem_scale,
                    predecessors=(),
                    prior_resources=(
                        ("compute:compute_tensor",
                         operand_completion_service)
                        if use_finite_operand_completion
                        else ("compute:compute_tensor",))),
                PipelineStageDemand(
                    "store", "memory:global_write",
                    output_bytes_per_cta * write_scale,
                    True, predecessors=("stage_output",)),
            )
            phases.append(AttentionPipelinePhaseDemand(
                name="epilogue", stages=epilogue))
            template = tuple(phases)
            sequence_templates[identity] = template
        return template

    worker_count = min(kernel.launch.grid_size, inp.active_units)

    def _exact_classes(identity_counts):
        result = []
        for (step_count, kv_rows), count in identity_counts.items():
            if count <= 0:
                continue
            if step_count == 0:
                if kv_rows != 0:
                    raise ValueError(
                        "MetaX tile16 empty split has nonzero KV rows")
                result.append((step_count, kv_rows, count))
                continue
            tail_rows = kv_rows - 16 * (step_count - 1)
            if not 0 < tail_rows <= 16:
                raise ValueError(
                    "MetaX tile16 class has an invalid tail extent")
            result.append((step_count, kv_rows, count))
        return tuple(result)

    use_resident_panel_round = (
        layout == "contiguous"
        and head_dim == 128
        and query_heads == kv_heads
        and len(source_class_counts) == 1
        and kernel.launch.grid_size > worker_count
        and kernel.launch.grid_size <= worker_count * resident_cta_slots)
    if use_resident_panel_round:
        (step_count, kv_rows), = source_class_counts

        def _cohort_phase(phase_name, programs):
            phases = []
            for program in programs:
                matches = tuple(
                    phase for phase in program if phase.name == phase_name)
                if len(matches) != 1:
                    raise ValueError(
                        "MetaX resident cohort lost a source phase")
                phases.append(matches[0])
            repeat_count = phases[0].repeat_count
            if any(phase.repeat_count != repeat_count for phase in phases):
                raise ValueError(
                    "MetaX resident cohort phase repeats disagree")
            merged = []
            for cta_index, phase in enumerate(phases):
                source_names = tuple(stage.name for stage in phase.stages)
                source_name_set = set(source_names)
                prefix = f"cta{cta_index}_"
                for stage_index, stage in enumerate(phase.stages):
                    predecessors = (
                        (() if stage_index == 0
                         else (phase.stages[stage_index - 1].name,))
                        if stage.predecessors is None else stage.predecessors)
                    if any(name not in source_name_set
                           for name in predecessors):
                        raise ValueError(
                            "MetaX resident phase has an external predecessor")
                    if any(name not in source_name_set
                           for name in stage.start_predecessors):
                        raise ValueError(
                            "MetaX resident phase has an external start edge")
                    prefixed_predecessors = tuple(
                        prefix + name for name in predecessors)
                    merged.append(dataclasses.replace(
                        stage,
                        name=prefix + stage.name,
                        predecessors=prefixed_predecessors,
                        start_predecessors=tuple(
                            prefix + name
                            for name in stage.start_predecessors),
                    ))
            return AttentionPipelinePhaseDemand(
                name=f"resident_round_{phase_name}",
                stages=tuple(merged),
                repeat_count=repeat_count,
            )

        def _cohort_class(cohort_size, lane_count):
            cold = _cta_phases(
                step_count, kv_rows, True, False, False)
            resident = _cta_phases(
                step_count, kv_rows, False, False, True)
            programs = (cold,) + (resident,) * (cohort_size - 1)
            phase_names = tuple(phase.name for phase in cold)
            if any(tuple(phase.name for phase in program) != phase_names
                   for program in programs):
                raise ValueError(
                    "MetaX resident CTA programs have different phases")
            return AttentionPipelineClassDemand(
                name=f"resident_cohort{cohort_size}_steps{step_count}",
                count=lane_count,
                query_rows=cohort_size * query_heads_per_cta,
                panel_count=cohort_size * step_count,
                phases=tuple(
                    _cohort_phase(name, programs) for name in phase_names),
            )

        cohort_floor, longer_lane_count = divmod(
            kernel.launch.grid_size, worker_count)
        cohort_histogram = []
        if longer_lane_count:
            cohort_histogram.append((
                cohort_floor + 1, longer_lane_count))
        if worker_count - longer_lane_count:
            cohort_histogram.append((
                cohort_floor, worker_count - longer_lane_count))
        if (sum(size * count for size, count in cohort_histogram)
                != kernel.launch.grid_size):
            raise AssertionError("MetaX resident cohort lost source CTAs")
        plan = AttentionPipelineEnvelopeDemand(
            active_lane_count=worker_count,
            classes=tuple(
                _cohort_class(size, count)
                for size, count in cohort_histogram),
            buffers=(),
        )
        limitations = tuple(
            value for value in inp.limitations
            if value != "gpu_flash_attention_cta_worker_mapping_unavailable")
        limitations += (
            "metax_static_resident_panel_round_robin_source_order",)
        return attach_attention_pipeline_envelope(
            dataclasses.replace(
                inp,
                support_level="partial",
                limitations=limitations,
            ),
            hardware,
            plan,
        )

    warm_class_counts = {}
    for identity, count in source_class_counts.items():
        cold_count = 0
        if identity in cold_class_counts:
            cold_count = cold_class_counts[identity]
        resident_fill_count = 0
        if identity in resident_fill_class_counts:
            resident_fill_count = resident_fill_class_counts[identity]
        warm_class_counts[identity] = (
            count - cold_count - resident_fill_count)
    schedule_classes = tuple(
        ("cold", step_count, kv_rows, count, True)
        for step_count, kv_rows, count in _exact_classes(cold_class_counts)
    ) + tuple(
        ("resident", step_count, kv_rows, count, False)
        for step_count, kv_rows, count in _exact_classes(
            resident_fill_class_counts)
    ) + tuple(
        ("warm", step_count, kv_rows, count, False)
        for step_count, kv_rows, count in _exact_classes(warm_class_counts)
    )
    if sum(item[3] for item in schedule_classes) != active_cta_count:
        raise AssertionError("MetaX tile16 classes lost source CTAs")
    classes = tuple(
        AttentionPipelineClassDemand(
            name=f"{kind}_steps{step_count}_rows{kv_rows}",
            count=count,
            query_rows=query_heads_per_cta,
            panel_count=step_count,
            phases=_cta_phases(
                step_count,
                kv_rows,
                first_request,
                layout in ("paged", "paged_varlen"),
                kind == "resident",
            ),
        )
        for (kind, step_count, kv_rows,
             count, first_request) in schedule_classes)
    if layout in ("paged", "paged_varlen"):
        tail_group_interval_panels = 8
        tail_requests_per_group = 1
    elif layout == "contiguous":
        tail_group_interval_panels = 0
        tail_requests_per_group = 0
    else:
        raise AssertionError(
            "source-bound MetaX tile16 layout escaped validation")
    plan = AttentionPipelineEnvelopeDemand(
        active_lane_count=worker_count,
        classes=classes,
        buffers=(PipelineBuffer(
            name="resident_cta_lifecycle",
            producer_stage="cta_start",
            consumer_stage="store",
            slots=(
                resident_cta_slots),
            scope="worker",
        ),),
        resident_tail_latency_service=(
            None if use_finite_operand_completion else "memory:global_read"),
        # Each panel has one dependent K and one dependent V fetch.  Paged KV
        # adds one dependent 32-bit block-table lookup per 128-token page,
        # which spans exactly eight 16-token panels.
        resident_tail_requests_per_panel=(
            0 if use_finite_operand_completion else 2),
        resident_tail_group_interval_panels=tail_group_interval_panels,
        resident_tail_requests_per_group=tail_requests_per_group,
    )
    limitations = tuple(
        value for value in inp.limitations
        if value != "gpu_flash_attention_cta_worker_mapping_unavailable")
    limitations += (
        "metax_full_ap_round_robin_cta_assignment_assumed",
    )
    return attach_attention_pipeline_envelope(
        dataclasses.replace(
            inp,
            support_level="partial",
            limitations=limitations,
        ),
        hardware,
        plan,
    )


def _attach_metax_mla_main_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the source scheduler and two-stage FlashMLA KV pipeline."""

    PipelineStageDemand = bind_pipeline_stage_template
    if not _is_source_bound_metax_mla_main(kernel):
        raise ValueError("MetaX FlashMLA main pipeline requires its source ABI")
    work = _metax_mla_scheduler_work(kernel)
    _, _, _, _, query_heads, _ = _shape(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    read_bytes = float(
        inp.demand_per_resource["l2_cache"] - _output_bytes(kernel, hardware))
    write_bytes = _output_bytes(kernel, hardware)
    tensor_operations = float(
        inp.demand_per_resource["compute_tensor"])
    softmax_operations = float(inp.demand_per_resource["compute_sfu"])
    shared_issue_operations = float(
        inp.demand_per_resource["warp_issue"])
    smem_bytes = float(inp.demand_per_resource["smem"])
    if min(
            read_bytes, write_bytes, tensor_operations,
            softmax_operations, shared_issue_operations,
            smem_bytes) <= 0:
        raise ValueError("MetaX FlashMLA main pipeline requires positive work")

    q_bytes = float(query_heads * 576 * element_bytes)
    # One shared qk_head_dim(576)-wide latent per KV position, not a
    # separate 512-wide value cache -- see _tensor_traffic's metax_mla
    # branch and _metax_mla_main_smem_bytes, whose k_stage_bytes/
    # v_stage_bytes below already price the on-chip QK/PV reuse of that
    # one global fetch as SMEM traffic.
    kv_bytes = float(METAX_MLA_KV_BLOCK * 576 * element_bytes)
    q_stage_bytes = float(2 * query_heads * 576 * element_bytes)
    k_stage_bytes = float(2 * METAX_MLA_KV_BLOCK * 576 * element_bytes)
    v_stage_bytes = float(METAX_MLA_KV_BLOCK * 512 * element_bytes)
    direct_output_bytes = float(query_heads * (512 * element_bytes + 4))
    partial_output_bytes = float(query_heads * (512 + 1) * 4)
    direct_output_stage_bytes = float(
        2 * query_heads * 512 * element_bytes)
    partial_output_stage_bytes = float(2 * query_heads * 512 * 4)
    tensor_operations_per_block = float(
        _METAX_MLA_MMA_PER_KV_BLOCK
        * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    issue_service = pipeline_compute_service(
        inp, hardware, roles=("warp_issue",))
    source_instruction_count = float(
        _METAX_MLA_MTE_LAUNCH
        + (_METAX_MLA_MTE_PER_KV_BLOCK + _METAX_MLA_MMA_PER_KV_BLOCK)
        * work.kv_block_count
        + _METAX_MLA_MTE_PER_SEGMENT * work.segment_count
        + _METAX_MLA_MTE_PER_SAME_WORKER_TRANSITION
        * work.same_worker_transitions
        + _METAX_MLA_MTE_PER_UNSPLIT_REQUEST * work.unsplit_requests)
    issue_weight = shared_issue_operations / source_instruction_count
    launch_instructions_per_worker = (
        _METAX_MLA_MTE_LAUNCH / METAX_MLA_SCHEDULER_PARTITIONS)

    def _issue_stage(
            name, instructions, predecessors, start_predecessors=()):
        if instructions <= 0:
            raise ValueError(
                "MetaX FlashMLA issue stage requires positive instructions")
        return PipelineStageDemand(
            name, issue_service,
            issue_weight * instructions / shared_issue_operations,
            predecessors=predecessors,
            start_predecessors=start_predecessors)

    def _segment_phases(kv_blocks, unsplit, same_worker_transition):
        segment_setup_instructions = (
            _METAX_MLA_MTE_PER_SEGMENT
            + (
                _METAX_MLA_MTE_PER_SAME_WORKER_TRANSITION
                if same_worker_transition else 0))
        prefix = "segment"
        prologue = (
            _issue_stage(
                f"{prefix}_issue_setup", segment_setup_instructions, ()),
            PipelineStageDemand(
                f"{prefix}_load_q", "memory:global_read", q_bytes / read_bytes,
                True, predecessors=()),
            PipelineStageDemand(
                f"{prefix}_stage_q", "resource:smem",
                q_stage_bytes / smem_bytes,
                predecessors=(f"{prefix}_load_q",)),
            PipelineStageDemand(
                f"{prefix}_load_kv_0", "memory:global_read",
                kv_bytes / read_bytes, True,
                predecessors=(f"{prefix}_stage_q",)),
            _issue_stage(
                f"{prefix}_mte_kv_0", _METAX_MLA_MTE_PER_KV_BLOCK,
                (f"{prefix}_issue_setup",)),
        )

        def _panel(*, phase_name, first, prefetch_next):
            qk = f"{prefix}_{phase_name}_qk"
            softmax = f"{prefix}_{phase_name}_softmax"
            pv = f"{prefix}_{phase_name}_pv"
            mma_issue = f"{prefix}_{phase_name}_mma_issue"
            qk_prior_resources = (
                "memory:global_read", issue_service)
            if not first:
                qk_prior_resources += ("compute:compute_tensor",)
            stage_k = f"{prefix}_{phase_name}_stage_k"
            stages = [
                PipelineStageDemand(
                    stage_k, "resource:smem",
                    k_stage_bytes / smem_bytes,
                    predecessors=(),
                    prior_resources=qk_prior_resources),
                PipelineStageDemand(
                    qk, "compute:compute_tensor",
                    0.5 * tensor_operations_per_block / tensor_operations,
                    predecessors=(stage_k,)),
            ]
            stages.append(_issue_stage(
                mma_issue, _METAX_MLA_MMA_PER_KV_BLOCK, (),
                start_predecessors=(qk,)))
            next_load = None
            if prefetch_next:
                next_load = f"{prefix}_{phase_name}_load_kv_next"
                stages.append(PipelineStageDemand(
                    next_load, "memory:global_read", kv_bytes / read_bytes,
                    True, predecessors=(qk,)))
                stages.append(_issue_stage(
                    f"{prefix}_{phase_name}_mte_kv_next",
                    _METAX_MLA_MTE_PER_KV_BLOCK, (qk,)))
            softmax_work = float(
                query_heads
                * (METAX_MLA_KV_BLOCK + (0 if first else 1)))
            stages.append(PipelineStageDemand(
                softmax, "compute:compute_sfu",
                softmax_work / softmax_operations,
                predecessors=(qk,),
                start_predecessors=(
                    () if next_load is None else (next_load,))))
            stage_v = f"{prefix}_{phase_name}_stage_v"
            stages.append(PipelineStageDemand(
                stage_v, "resource:smem", v_stage_bytes / smem_bytes,
                predecessors=(softmax,)))
            stages.append(PipelineStageDemand(
                pv, "compute:compute_tensor",
                0.5 * tensor_operations_per_block / tensor_operations,
                predecessors=(stage_v,)))
            return tuple(stages)

        phases = [AttentionPipelinePhaseDemand(
            name=f"{prefix}_prologue", stages=prologue)]
        if kv_blocks == 1:
            phases.append(AttentionPipelinePhaseDemand(
                name=f"{prefix}_first_tail",
                stages=_panel(
                    phase_name="first_tail", first=True,
                    prefetch_next=False)))
        else:
            phases.append(AttentionPipelinePhaseDemand(
                name=f"{prefix}_first",
                stages=_panel(
                    phase_name="first", first=True,
                    prefetch_next=True)))
            if kv_blocks > 2:
                phases.append(AttentionPipelinePhaseDemand(
                    name=f"{prefix}_steady",
                    stages=_panel(
                        phase_name="steady", first=False,
                        prefetch_next=True),
                    repeat_count=kv_blocks - 2))
            phases.append(AttentionPipelinePhaseDemand(
                name=f"{prefix}_tail",
                stages=_panel(
                    phase_name="tail", first=False,
                    prefetch_next=False)))
        output_bytes = (
            direct_output_bytes
            if unsplit
            else partial_output_bytes)
        output_stage_bytes = (
            direct_output_stage_bytes
            if unsplit
            else partial_output_stage_bytes)
        epilogue_stages = []
        if unsplit:
            issue_output = f"{prefix}_issue_unsplit_output"
            epilogue_stages.append(PipelineStageDemand(
                issue_output, issue_service,
                (issue_weight * _METAX_MLA_MTE_PER_UNSPLIT_REQUEST
                 / shared_issue_operations),
                predecessors=(),
                prior_resources=(
                    "compute:compute_tensor", issue_service)))
            output_stage_predecessors = (issue_output,)
            store_prior_resources = ()
        else:
            output_stage_predecessors = ()
            store_prior_resources = (
                "compute:compute_tensor", issue_service)
        stage_output = f"{prefix}_stage_output"
        epilogue_stages.append(PipelineStageDemand(
            stage_output, "resource:smem",
            output_stage_bytes / smem_bytes,
            predecessors=output_stage_predecessors,
            prior_resources=store_prior_resources))
        epilogue_stages.append(PipelineStageDemand(
            f"{prefix}_store", "memory:global_write",
            output_bytes / write_bytes, True,
            predecessors=(stage_output,)))
        phases.append(AttentionPipelinePhaseDemand(
            name=f"{prefix}_epilogue",
            stages=tuple(epilogue_stages)))
        return tuple(phases)

    segment_orders = tuple(
        tuple((
            segment.kv_blocks,
            work.splits_per_request[segment.request_index] == 1,
            ordinal > 0,
        ) for ordinal, segment in enumerate(segments))
        for segments in work.segments_by_partition)
    order_multiplicities = {}
    for order in segment_orders:
        order_multiplicities[order] = (
            order_multiplicities[order] + 1
            if order in order_multiplicities else 1)

    programs = []
    for worker, (segment_order, multiplicity) in enumerate(
            order_multiplicities.items()):
        identities = tuple(dict.fromkeys(segment_order))
        patterns = [PipelineTaskSequenceDemand(
            name=f"worker{worker}_launch",
            worker=worker,
            tasks=(PipelineTaskDemand(
                name="launch",
                worker=worker,
                stages=(_issue_stage(
                    "issue_launch", launch_instructions_per_worker, ()),),
            ),),
        )]
        pattern_by_identity = {}
        for identity in identities:
            kv_blocks, unsplit, transition = identity
            pattern_by_identity[identity] = len(patterns)
            phases = _segment_phases(kv_blocks, unsplit, transition)
            patterns.append(PipelineTaskSequenceDemand(
                name=(f"worker{worker}_"
                      f"{'direct' if unsplit else 'split'}_"
                      f"{'warm' if transition else 'first'}_"
                      f"blocks{kv_blocks}"),
                worker=worker,
                tasks=tuple(PipelineTaskDemand(
                    name=phase.name,
                    worker=worker,
                    stages=phase.stages,
                    repeat_count=phase.repeat_count,
                ) for phase in phases),
            ))
        programs.append(PipelineTaskProgramDemand(
            name=f"metax_mla_worker_program{worker}",
            worker=worker,
            patterns=tuple(patterns),
            order=(0,) + tuple(
                pattern_by_identity[identity]
                for identity in segment_order),
            worker_multiplicity=multiplicity,
        ))
    plan = PipelineWorkPlan(
        worker_count=METAX_MLA_SCHEDULER_PARTITIONS,
        tasks=tuple(programs),
        buffers=(),
    )
    limitations = tuple(
        value for value in inp.limitations
        if value not in {
            "gpu_flash_attention_tile_geometry_unavailable",
            "gpu_flash_attention_cta_worker_mapping_unavailable",
        })
    return attach_pipeline_work_plan(
        dataclasses.replace(
            inp,
            support_level="partial",
            limitations=limitations,
        ),
        hardware,
        plan,
    )


def _has_metax_q128_prefill_pipeline(kernel: KernelSpec) -> bool:
    """Whether a pinned Q128 direct prefill source closes every CTA DAG."""

    if _has_graph_padding(kernel):
        return False
    algorithm = kernel.launch.algorithm
    if algorithm not in {
            _METAX_372_Q128_PREFILL_ALGORITHM,
            _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM,
    }:
        return False
    batch, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    common = (
        kernel.variant.lower() == "prefill"
        and kernel.dtype.lower() == "bf16"
        and head_dim == 128
        and q_len % 128 == 0
        and all(
            query == q_len
            and query % 128 == 0
            and key_value >= query
            and key_value % 64 == 0
            for query, key_value in zip(q_lens, kv_lens))
        and _attention_tile(kernel) == (128, 64)
        and _splits(kernel) == 1
        and kernel.attributes["causal"] is True
        and kernel.launch.block == (256, 1, 1)
    )
    if not common:
        return False
    if algorithm == _METAX_372_Q128_PREFILL_ALGORITHM:
        return (
            kernel.attributes["kv_layout"] == "contiguous"
            and all(query == key_value
                    for query, key_value in zip(q_lens, kv_lens))
            and kernel.launch.grid == (query_heads, batch, q_len // 128)
        )
    return (
        kernel.attributes["kv_layout"] in (
            "paged", "paged_varlen", "chunked_prefill")
        and kernel.launch.grid == (query_heads, q_len // 128, batch)
    )


def _attach_metax_q128_prefill_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the source-defined MACA 3.7.2 Q128 prefill pipeline.

    The contiguous and split wrappers place the M block on different grid
    axes, but both reverse it before entering the same causal Q128/KV64 trait.
    Consequently the first physical CTAs execute the longest KV loops.  CTAs
    remain independent and occupy the two compiler-proven resident slots on
    each AP; only dependencies inside one CTA constrain Q/K/V, QK,
    online-softmax, PV, and the epilogue.
    """

    PipelineStageDemand = bind_pipeline_stage_template
    if not _has_metax_q128_prefill_pipeline(kernel):
        raise ValueError(
            "MetaX Q128 pipeline requires a pinned causal full-tile prefill "
            "program")
    batch, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, kv_lens = _request_lengths(kernel)
    if inp.active_units <= 0:
        raise ValueError("MetaX Q128 pipeline requires active APs")
    worker_count = inp.active_units

    scheduled_q_tiles = q_len // 128
    active_cta_count = batch * scheduled_q_tiles * query_heads
    if kernel.launch.algorithm == _METAX_372_Q128_PREFILL_ALGORITHM:
        schedule_class_counts = tuple(
            (_AttentionCTAWork(128, 2 * (tile + 1)), batch * query_heads)
            for tile in reversed(range(scheduled_q_tiles)))
    else:
        schedule = []
        for query, key_value in zip(q_lens, kv_lens):
            context = key_value - query
            for physical_q_tile in range(scheduled_q_tiles):
                source_q_tile = scheduled_q_tiles - physical_q_tile - 1
                attended = context + (source_q_tile + 1) * 128
                item = _AttentionCTAWork(
                    query_rows=128,
                    step_count=math.ceil(attended / 64),
                )
                if schedule and schedule[-1][0] == item:
                    previous, count = schedule[-1]
                    schedule[-1] = (previous, count + query_heads)
                else:
                    schedule.append((item, query_heads))
        schedule_class_counts = tuple(schedule)
    scheduled_cta_count = batch * scheduled_q_tiles * query_heads
    if active_cta_count != scheduled_cta_count:
        raise ValueError("MetaX Q128 full-tile launch contains an empty CTA")
    if scheduled_cta_count != kernel.launch.grid_size:
        raise ValueError(
            "MetaX Q128 scheduled CTA count disagrees with the direct grid: "
            f"source={scheduled_cta_count}, grid={kernel.launch.grid_size}")
    total_kv_steps = sum(
        item.step_count * count for item, count in schedule_class_counts)
    element_bytes = int(kernel.shape["dtype_bytes"])
    q_bytes_per_cta = float(128 * head_dim * element_bytes)
    kv_bytes_per_step = float(64 * head_dim * element_bytes)
    output_bytes_per_cta = float(128 * (head_dim * element_bytes + 4))
    staged_output_bytes_per_cta = float(128 * head_dim * element_bytes)
    read_bytes = float(
        inp.demand_per_resource["l2_cache"] - _output_bytes(kernel, hardware))
    write_bytes = _output_bytes(kernel, hardware)
    expected_read_bytes = float(
        active_cta_count * q_bytes_per_cta
        + total_kv_steps * 2.0 * kv_bytes_per_step)
    expected_write_bytes = float(active_cta_count * output_bytes_per_cta)
    if not math.isclose(
            read_bytes, expected_read_bytes, rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q128 source DAG read bytes disagree with aggregate "
            f"traffic: dag={expected_read_bytes}, aggregate={read_bytes}")
    if not math.isclose(
            write_bytes, expected_write_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q128 source DAG write bytes disagree with aggregate "
            f"traffic: dag={expected_write_bytes}, aggregate={write_bytes}")
    smem_bytes = float(inp.demand_per_resource["smem"])
    expected_smem_bytes = 2.0 * (
        read_bytes + active_cta_count * staged_output_bytes_per_cta)
    if not math.isclose(
            smem_bytes, expected_smem_bytes,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q128 source DAG shared-memory bytes disagree with "
            f"aggregate traffic: dag={expected_smem_bytes}, "
            f"aggregate={smem_bytes}")
    tensor_operations = float(inp.demand_per_resource["compute_tensor"])
    tensor_operations_per_step = float(
        _METAX_372_FA_MMA_PER_FULL_KV_TILE
        * _METAX_372_FA_TENSOR_OPS_PER_MMA)
    expected_tensor_operations = total_kv_steps * tensor_operations_per_step
    if not math.isclose(
            tensor_operations, expected_tensor_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q128 source DAG Tensor work disagrees with aggregate "
            "demand")
    softmax_operations = float(inp.demand_per_resource["compute_sfu"])
    expected_softmax_operations = float(math.fsum(
        count * (
            item.step_count * 128 * 64
            + max(0, item.step_count - 1) * 128)
        for item, count in schedule_class_counts))
    if not math.isclose(
            softmax_operations, expected_softmax_operations,
            rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise ValueError(
            "MetaX Q128 source DAG softmax work disagrees with aggregate "
            "demand")
    softmax_service = pipeline_compute_service(
        inp, hardware, roles=("compute_sfu",))
    tensor_step_fraction = (
        0.5 * tensor_operations_per_step / tensor_operations)

    sequence_templates = {}

    def _cta_phases(step_count):
        if step_count in sequence_templates:
            return sequence_templates[step_count]
        read_scale = 1.0 / read_bytes
        write_scale = 1.0 / write_bytes
        smem_scale = 1.0 / smem_bytes
        prologue = (
            PipelineStageDemand(
                "load_q", "memory:global_read",
                q_bytes_per_cta * read_scale,
                True, predecessors=()),
            PipelineStageDemand(
                "q_smem_write", "resource:smem",
                q_bytes_per_cta * smem_scale,
                predecessors=("load_q",)),
            PipelineStageDemand(
                "load_k_0", "memory:global_read",
                kv_bytes_per_step * read_scale,
                predecessors=("q_smem_write",)),
            PipelineStageDemand(
                "q_smem_read", "resource:smem",
                q_bytes_per_cta * smem_scale,
                predecessors=("load_k_0",)),
            PipelineStageDemand(
                "load_v_0", "memory:global_read",
                kv_bytes_per_step * read_scale,
                predecessors=("q_smem_read",)),
            PipelineStageDemand(
                "stage_k_0", "resource:smem",
                2.0 * kv_bytes_per_step * smem_scale,
                predecessors=("load_v_0",)),
        )

        def _panel(*, phase_name, first, prefetch_next):
            stages = []
            qk = f"qk_{phase_name}"
            stage_v = f"stage_v_{phase_name}"
            softmax = f"softmax_{phase_name}"
            pv = f"pv_{phase_name}"
            qk_predecessors = ()
            qk_prior_resources = (
                "memory:global_read", "resource:smem")
            if not first:
                stages.append(PipelineStageDemand(
                    f"load_v_{phase_name}", "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=(),
                    prior_resources=(
                        "resource:smem", "compute:compute_tensor")))
                qk_predecessors = (f"load_v_{phase_name}",)
                qk_prior_resources = ()
            stages.extend((
                PipelineStageDemand(
                    qk, "compute:compute_tensor",
                    tensor_step_fraction,
                    predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources),
                PipelineStageDemand(
                    stage_v, "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=(qk,)),
            ))
            load_k_next = None
            if prefetch_next:
                load_k_next = "load_k_next"
                stages.append(PipelineStageDemand(
                    load_k_next, "memory:global_read",
                    kv_bytes_per_step * read_scale,
                    False, predecessors=(stage_v,)))
            softmax_step_operations = float(
                128 * 64 + (0 if first else 128))
            stages.extend((
                PipelineStageDemand(
                    softmax, softmax_service,
                    softmax_step_operations / softmax_operations,
                    predecessors=(stage_v,),
                    start_predecessors=(
                        () if load_k_next is None else (load_k_next,))),
                PipelineStageDemand(
                    pv, "compute:compute_tensor",
                    tensor_step_fraction,
                    predecessors=(softmax, stage_v)),
            ))
            if load_k_next is not None:
                stages.append(PipelineStageDemand(
                    "stage_k_next", "resource:smem",
                    2.0 * kv_bytes_per_step * smem_scale,
                    predecessors=(load_k_next, pv)))
            return tuple(stages)

        phases = [AttentionPipelinePhaseDemand(
            name="prologue", stages=prologue)]
        if step_count == 1:
            phases.append(AttentionPipelinePhaseDemand(
                name="first_tail_panel",
                stages=_panel(
                    phase_name="0", first=True, prefetch_next=False)))
        else:
            phases.append(AttentionPipelinePhaseDemand(
                name="first_panel",
                stages=_panel(
                    phase_name="0", first=True, prefetch_next=True)))
            if step_count > 2:
                phases.append(AttentionPipelinePhaseDemand(
                    name="steady_panel",
                    stages=_panel(
                        phase_name="steady", first=False,
                        prefetch_next=True),
                    repeat_count=step_count - 2))
            phases.append(AttentionPipelinePhaseDemand(
                name="tail_panel",
                stages=_panel(
                    phase_name="tail", first=False,
                    prefetch_next=False)))
        epilogue = (
            PipelineStageDemand(
                "output_smem", "resource:smem",
                2.0 * staged_output_bytes_per_cta * smem_scale,
                predecessors=(),
                prior_resources=("compute:compute_tensor",)),
            PipelineStageDemand(
                "store", "memory:global_write",
                output_bytes_per_cta * write_scale,
                True, predecessors=("output_smem",)),
        )
        phases.append(AttentionPipelinePhaseDemand(
            name="epilogue", stages=epilogue))
        sequence_templates[step_count] = tuple(phases)
        return sequence_templates[step_count]

    schedule_class_counts, source_repeat_count = (
        _compact_exact_source_period(schedule_class_counts))
    classes = tuple(
        AttentionPipelineClassDemand(
            name=f"run{run_index}_rows128_steps{item.step_count}",
            count=count,
            query_rows=128,
            panel_count=item.step_count,
            phases=_cta_phases(item.step_count),
        )
        for run_index, (item, count) in enumerate(schedule_class_counts))
    plan = AttentionPipelineEnvelopeDemand(
        active_lane_count=min(active_cta_count, worker_count),
        classes=classes,
        source_repeat_count=source_repeat_count,
        buffers=(PipelineBuffer(
            name="resident_cta_lifecycle",
            producer_stage="load_q",
            consumer_stage="store",
            slots=_METAX_372_Q128_PREFILL_RESIDENT_CTA_SLOTS_PER_AP,
            scope="worker",
        ),),
    )
    limitations = tuple(
        value for value in inp.limitations
        if value != "gpu_flash_attention_cta_worker_mapping_unavailable")
    limitations += ("metax_full_ap_round_robin_cta_assignment_assumed",)
    return attach_attention_pipeline_envelope(
        dataclasses.replace(
            inp,
            support_level="partial",
            limitations=limitations,
        ),
        hardware,
        plan,
    )


def _attach_gpu_attention_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware,
        *, grouped_mla: bool, tile_q: int, tile_kv: int) -> PredictionInput:
    """Attach a closed-form SM80 attention pipeline envelope.

    Dao FA2 and vLLM's pinned Triton grouped-MLA source both execute a tiled
    ``load K -> QK`` and ``load V -> online softmax -> PV`` loop.  Their QK and
    value widths differ for grouped MLA, so stage bytes and Tensor work are
    derived separately instead of forcing the symmetric FA2 dimensions.  The
    envelope retains analytical CTA classes, their exact counts, and total
    panel work.  It does not create physical CTA or per-SM schedules.
    """

    is_fa2 = _is_source_bound_fa2_pipeline(kernel)
    is_grouped_mla = _is_source_bound_grouped_mla_ldgsts(kernel)
    # The A800 adapter supplies SM80 launch/source facts below.  Every stage
    # then binds through the same static-contract cache used by C500, leaving
    # only its physical tile bytes and work fractions dynamic per prediction.
    PipelineStageDemand = bind_pipeline_stage_template
    if grouped_mla != is_grouped_mla:
        raise ValueError("GPU attention pipeline/source-family mismatch")
    if not is_fa2 and not is_grouped_mla:
        raise ValueError(
            "FLASH_ATTN pipeline requires a pinned SM80 source family")
    class_counts, head_groups = _pipeline_cta_class_counts(
        kernel, tile_q, tile_kv, grouped_mla=grouped_mla,
        include_empty_splits=True)
    active_cta_count = sum(count for _, count in class_counts)
    if active_cta_count <= 0:
        raise ValueError("FLASH_ATTN pipeline has no active CTA")
    source_order_classes = None
    source_repeat_count = 1
    paired_fifo_admission = (
        is_fa2 and not grouped_mla
        and _has_a800_fa2_paired_fifo_admission(
            kernel, hardware, tile_q, tile_kv,
            class_counts=class_counts, head_groups=head_groups))
    bounded_fifo_drain = (
        is_fa2 and not grouped_mla
        and _has_a800_fa2_dense_bounded_fifo_drain(
            kernel, hardware, tile_q, tile_kv,
            class_counts=class_counts))
    split3_varlen_admission = (
        is_fa2 and not grouped_mla
        and _has_a800_fa2_split3_varlen_admission(
            kernel, hardware, tile_q, tile_kv))
    if (is_fa2 and (
            active_cta_count <= (
                inp.active_units * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS)
            or paired_fifo_admission or bounded_fifo_drain)):
        source_order_classes = _resident_fa2_source_order_classes(
            kernel, tile_q, tile_kv)
    if source_order_classes is None and is_fa2:
        source_order_classes = _fa2_varlen_single_tile_source_order_classes(
            kernel, tile_q, tile_kv)
    if source_order_classes is None and is_fa2:
        source_period = _fa2_chunked_prefill_source_order_period(
            kernel, tile_q, tile_kv)
        if source_period is not None:
            source_order_classes, source_repeat_count = source_period
    resident_capacity = (
        inp.active_units * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS)
    if bounded_fifo_drain:
        source_order_classes = _balanced_prefix_with_source_suffix(
            source_order_classes, resident_capacity)
        source_repeat_count = 1
    if split3_varlen_admission:
        if source_order_classes is None:
            raise AssertionError(
                "split-3 varlen admission requires exact source order")
        source_order_classes = _a800_split3_varlen_admission_order(
            source_order_classes)
        source_repeat_count = 1
    work_conserving_varlen_queue = (
        is_fa2
        and kernel.attributes["kv_layout"] == "paged_varlen"
        # A single global query tile (decode, or a short prefill
        # continuation mixed into the same batch) is what the GPU work
        # distributor's dynamic, work-conserving CTA admission was measured
        # on; it does not depend on any request's query length otherwise.
        and kernel.launch.grid[0] == 1
        and active_cta_count == kernel.launch.grid_size
        and active_cta_count > resident_capacity
        and source_order_classes is not None
        and len(class_counts) > 1)
    if work_conserving_varlen_queue and not split3_varlen_admission:
        source_order_classes = (
            _source_prefix_with_balanced_queue_tail(
                source_order_classes, resident_capacity)
            if _splits(kernel) == 1 else None)
        source_repeat_count = 1
    schedule_class_counts = (
        source_order_classes
        if source_order_classes is not None
        else _moment_preserving_cta_classes(class_counts))
    if source_repeat_count == 1:
        schedule_class_counts, source_repeat_count = (
            _compact_exact_source_period(schedule_class_counts))
    if (source_repeat_count
            * sum(count for _, count in schedule_class_counts)
            != active_cta_count):
        raise AssertionError("FA2 source schedule lost active CTAs")
    active_lane_count = min(kernel.launch.grid_size, inp.active_units)
    if (is_fa2
            and kernel.attributes["kv_layout"] in (
                "chunked_prefill", "paged_varlen")
            and int(kernel.shape["q_len"]) > 1):
        active_lane_count = min(
            inp.active_units,
            math.ceil(kernel.launch.grid_size
                      / _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS),
        )
    if active_lane_count <= 0:
        raise ValueError("FLASH_ATTN pipeline has no active execution lane")
    resident_frontend_overlap = (
        is_fa2
        and kernel.attributes["kv_layout"] == "contiguous"
        and active_cta_count > active_lane_count
        and active_cta_count <= (
            active_lane_count * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS)
    )
    # In the single resident cohort each SM can advance both FA2 CTAs through
    # their CTA-local LDGSTS dependencies.  Paged kernels stay excluded until
    # their block-table dependency is represented in this source envelope.
    distribution = _work_distribution(kernel)
    if distribution != "direct_cta":
        raise ValueError(
            "source-closed SM80 attention pipeline requires direct CTAs")
    padded_query_cta_count = kernel.launch.grid_size - active_cta_count
    if padded_query_cta_count < 0:
        raise ValueError("FA2 active CTA count exceeds its physical grid")
    if (padded_query_cta_count > 0
            and not (is_fa2
                     and kernel.attributes["kv_layout"] in (
                         "chunked_prefill", "paged_varlen")
                     and int(kernel.shape["q_len"]) > 1)):
        raise ValueError(
            "only source-bound FA2 chunked prefill may pad query CTAs")
    # FA2 2.8.3 constructs BlockInfo for every physical chunked-prefill CTA,
    # then returns before output stores when its query tile lies beyond the
    # request-local length.  Do not misrepresent those padded query CTAs as
    # the output-writing zero-KV-split source path.  The physical launch floor
    # already prices their admission/exit.  The active CTAs retain the same
    # source-ordered K/QK/V/softmax/PV program as the dense FA2 path, including
    # the independently established two resident CTA slots per SM, so keep
    # those active source programs in the common pipeline envelope below.
    total_steps = sum(
        item.step_count * count for item, count in class_counts)
    tensor_operations = float(inp.demand_per_resource["compute_tensor"])
    softmax_operations = float(inp.demand_per_resource["compute_sfu"])
    if tensor_operations <= 0 or softmax_operations <= 0:
        raise ValueError("SM80 FA2 source DAG requires Tensor and SFU demand")
    _, _, _, head_dim, query_heads, _ = _shape(kernel)
    if query_heads % head_groups:
        raise ValueError("FA2 CTA head grouping must divide query heads")
    query_heads_per_cta = query_heads // head_groups
    if kernel.launch.algorithm == _A800_GROUPED_MLA_STAGE1_ALGORITHM:
        shared_query_groups_per_cta = query_heads_per_cta
    else:
        shared_query_groups_per_cta = _fa2_shared_query_groups_per_cta(
            kernel, query_heads_per_cta)
    element_bytes = int(kernel.shape["dtype_bytes"])
    splits = _splits(kernel)
    has_source_scalar_program = is_fa2
    if grouped_mla:
        grouped = _grouped_mla_tile(kernel)
        if grouped is None:
            raise AssertionError("grouped MLA pipeline lost its tile")
        _, _, qk_head_dim, value_head_dim = grouped
    else:
        qk_head_dim = head_dim
        value_head_dim = head_dim
    k_stage_bytes = float(tile_kv * qk_head_dim * element_bytes)
    v_stage_bytes = float(tile_kv * value_head_dim * element_bytes)
    q_smem_tile_bytes = float(
        tile_q * shared_query_groups_per_cta
        * qk_head_dim * element_bytes)
    page_table_stage_bytes = kernel.launch.parameters[
        "page_table_bytes_per_kv_panel"]
    qk_step_operations = float(
        2 * tile_q * tile_kv * qk_head_dim)
    pv_step_operations = float(
        2 * tile_q * tile_kv * value_head_dim)
    tensor_qk_fraction = qk_step_operations / tensor_operations
    tensor_pv_fraction = pv_step_operations / tensor_operations
    output_width_bytes = (
        value_head_dim * element_bytes
        if splits == 1 else (value_head_dim + 1) * 4)
    if has_dense_flash_attention_source(kernel):
        output_width_bytes = _output_bytes(kernel,hardware)/(sum(kernel.attributes['q_lens'])*query_heads)
    output_smem_tile_bytes = float(
        tile_q * shared_query_groups_per_cta * value_head_dim
        * (element_bytes if splits == 1 else 4))

    def _cta_query_byte_count(item):
        return (
            float(item.query_rows * query_heads_per_cta
                  * qk_head_dim * element_bytes)
            if item.step_count > 0 else 0.0)

    def _cta_output_byte_count(item):
        return float(
            item.query_rows * query_heads_per_cta * output_width_bytes)

    read_bytes = float(math.fsum(
        count * (_cta_query_byte_count(item) + item.step_count * (
            page_table_stage_bytes + k_stage_bytes + v_stage_bytes)
        ) for item, count in class_counts))
    write_bytes = float(math.fsum(
        count * _cta_output_byte_count(item)
        for item, count in class_counts))
    expected_write_bytes = _output_bytes(kernel, hardware)
    expected_read_bytes = (
        float(inp.demand_per_resource["l2_cache"])
        - expected_write_bytes)
    if not math.isclose(
            read_bytes, expected_read_bytes, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "FA2 source DAG read bytes disagree with aggregate traffic: "
            f"dag={read_bytes}, aggregate={expected_read_bytes}")
    if not math.isclose(
            write_bytes, expected_write_bytes, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "FA2 source DAG write bytes disagree with aggregate traffic: "
            f"dag={write_bytes}, aggregate={expected_write_bytes}")
    paths_by_name = {path.name: path for path in inp.memory_paths}
    has_period32_kv_service = "global_kv_read" in paths_by_name
    if has_period32_kv_service:
        if set(paths_by_name) != {
                "global_read", "global_kv_read", "global_write"}:
            raise ValueError("FA2 period-32 pipeline has unexpected paths")
        if page_table_stage_bytes != 0.0:
            raise ValueError("FA2 period-32 K/V service cannot price page tables")
        q_read_bytes = paths_by_name["global_read"].nodes[0].bytes
        kv_read_bytes = paths_by_name["global_kv_read"].nodes[0].bytes
        expected_q_read_bytes = math.fsum(
            count * _cta_query_byte_count(item)
            for item, count in class_counts)
        expected_kv_read_bytes = math.fsum(
            count * item.step_count * (k_stage_bytes + v_stage_bytes)
            for item, count in class_counts)
        if not (
                math.isclose(q_read_bytes, expected_q_read_bytes,
                             rel_tol=1e-12, abs_tol=1e-6)
                and math.isclose(kv_read_bytes, expected_kv_read_bytes,
                                 rel_tol=1e-12, abs_tol=1e-6)
                and math.isclose(read_bytes, q_read_bytes + kv_read_bytes,
                                 rel_tol=1e-12, abs_tol=1e-6)):
            raise ValueError("FA2 period-32 Q/KV source bytes do not close")
    else:
        if set(paths_by_name) != {"global_read", "global_write"}:
            raise ValueError("FA2 source pipeline has unexpected memory paths")
        q_read_bytes = read_bytes
        kv_read_bytes = None
    if has_source_scalar_program:
        expected_fma, expected_sfu = (
            _gpu_fa2_scalar_operations(kernel))
        for role, expected in (
                ("compute_fma", expected_fma),
                ("compute_sfu", expected_sfu)):
            actual = float(inp.demand_per_resource[role])
            if not math.isclose(
                    actual, expected, rel_tol=1e-12, abs_tol=1e-6):
                raise ValueError(
                    "FA2 source DAG scalar work disagrees with aggregate "
                    f"demand for {role}: dag={expected}, aggregate={actual}")
        scalar_service = pipeline_compute_service(
            inp, hardware, roles=("compute_fma", "compute_sfu"))

        def _compute_service_time(role, operations):
            demand = next(
                value for value in inp.compute_demands
                if value.resource_role == role)
            resource = hardware.resource_by_role(role)
            capacity = resource.effective_capacity(
                inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
            schedule_efficiency = min(
                1.0, demand.schedule_efficiency / inp.wave_efficiency)
            return operations / (
                capacity * demand.shape_efficiency * schedule_efficiency)

        total_scalar_service_time = sum(
            _compute_service_time(
                role, float(inp.demand_per_resource[role]))
            for role in ("compute_fma", "compute_sfu"))
        if total_scalar_service_time <= 0:
            raise ValueError("FA2 scalar source work must be positive")
    else:
        expected_softmax_operations = math.fsum(
            count * (
                item.step_count * tile_q * tile_kv
                + max(0, item.step_count - 1) * tile_q)
            for item, count in class_counts)
        if not math.isclose(
                expected_softmax_operations, softmax_operations,
                rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(
                "attention source DAG SFU work disagrees with aggregate "
                f"demand: dag={expected_softmax_operations}, "
                f"aggregate={softmax_operations}")
        scalar_service = pipeline_compute_service(
            inp, hardware, roles=("compute_sfu",))
        total_scalar_service_time = None
    has_fa2_main_sass_program = (
        is_fa2
        and ((_splits(kernel) > 1
              and kernel.attributes["kv_layout"] == "contiguous")
             or _is_source_bound_fa2_dense_sass_program(kernel)))
    has_program_dependency = (
        has_fa2_main_sass_program
        and kernel.launch.grid_size <= hardware.num_compute_units)
    has_source_dependency = (
        is_fa2
        and inp.demand_per_resource["compute_reduction"] > 0
        and not has_program_dependency)
    has_sass_issue = inp.demand_per_resource["warp_issue"] > 0
    has_sass_dependency = (
        grouped_mla
        and inp.demand_per_resource["compute_reduction"] > 0)
    expected_sass_issue = (
        (grouped_mla and splits > 1)
        or has_fa2_main_sass_program)
    expected_sass_dependency = (
        grouped_mla and splits > 1
        and kernel.launch.grid_size <= hardware.num_compute_units)
    if (has_sass_issue != expected_sass_issue
            or has_sass_dependency != expected_sass_dependency):
        raise ValueError("FA2 main SASS schedule demand is incomplete")
    has_grouped_mla_sass_stages = grouped_mla and has_sass_issue
    has_fa2_split_d128_sass_stages = (
        has_sass_issue
        and _is_source_bound_fa2_split_d128_stage_program(kernel))
    expected_source_dependency = (
        is_fa2
        and not has_program_dependency
        and _fa2_register_dependency_mechanism(kernel, hardware) is not None)
    if has_source_dependency != expected_source_dependency:
        raise ValueError("FA2 source dependency demand is incomplete")
    if has_grouped_mla_sass_stages or has_fa2_split_d128_sass_stages:
        issue_service = pipeline_compute_service(
            inp, hardware, roles=("warp_issue",))
        total_issue = float(inp.demand_per_resource["warp_issue"])
    if (has_sass_dependency or has_source_dependency
            or has_program_dependency):
        dependency_service = pipeline_compute_service(
            inp, hardware, roles=("compute_reduction",))
        total_dependency = float(
            inp.demand_per_resource["compute_reduction"])
    source_dependency_mechanism = (
        _fa2_register_dependency_mechanism(kernel, hardware)
        if has_source_dependency else None)
    resident_dependency_factors = _fa2_resident_dependency_factors(
        kernel, hardware, source_dependency_mechanism)

    def _resident_dependency_factor(stage):
        return (
            1.0 if resident_dependency_factors is None
            else resident_dependency_factors[stage])
    has_local_spill_dependency = (
        _fa2_local_spill_dependency_cycles(kernel, hardware) > 0)
    if has_local_spill_dependency and not has_source_dependency:
        raise ValueError(
            "FA2 local spill requires the source dependency service")
    smem_bytes = float(inp.demand_per_resource["smem"])
    if (is_fa2 or grouped_mla) and smem_bytes <= 0:
        raise ValueError("SM80 attention source DAG requires shared-memory work")
    sequence_templates = {}

    def _cta_phases(item):
        include_panel_first_request_latency = (
            kernel.launch.grid_size <= hardware.num_compute_units
            or item.step_count <= 2)
        if item in sequence_templates:
            return sequence_templates[item]
        sass_issue, sass_dependency = _fa2_main_sass_schedule_per_cta(
            kernel, item.step_count)
        grouped_mla_issue = (
            _grouped_mla_sass_stage_issue_per_cta(kernel, item.step_count)
            if has_grouped_mla_sass_stages and item.step_count > 0
            else None)
        fa2_split_issue = (
            _fa2_split_d128_sass_stage_issue_per_cta(
                kernel, item.step_count)
            if has_fa2_split_d128_sass_stages and item.step_count > 0
            else None)
        sass_stages = []
        first_predecessors = ()
        if has_grouped_mla_sass_stages and item.step_count == 0:
            sass_stages.append(PipelineStageDemand(
                "instruction_issue", issue_service,
                sass_issue / total_issue, False,
                predecessors=(),
            ))
            first_predecessors = ("instruction_issue",)
        if has_sass_dependency:
            sass_stages.append(PipelineStageDemand(
                "fixed_dependency", dependency_service,
                sass_dependency / total_dependency, False,
                predecessors=first_predecessors,
            ))
            first_predecessors = ("fixed_dependency",)
        if item.step_count == 0:
            if has_program_dependency:
                sass_stages.append(PipelineStageDemand(
                    "program_dependency", dependency_service,
                    sass_issue / total_dependency, False,
                    predecessors=(),
                ))
            partial_output_bytes = float(
                item.query_rows * query_heads_per_cta * value_head_dim * 4)
            lse_output_bytes = float(
                item.query_rows * query_heads_per_cta * 4)
            if not math.isclose(
                    partial_output_bytes + lse_output_bytes,
                    _cta_output_byte_count(item),
                    rel_tol=1e-12, abs_tol=1e-6):
                raise AssertionError(
                    "FA2 empty split stores do not conserve output bytes")
            stages = tuple(sass_stages) + (
                PipelineStageDemand(
                    "cta_begin", "memory:global_write",
                    partial_output_bytes / write_bytes, True,
                    predecessors=first_predecessors,
                ),
                PipelineStageDemand(
                    "cta_end", "memory:global_write",
                    lse_output_bytes / write_bytes, False,
                    predecessors=(
                        ("cta_begin", "program_dependency")
                        if has_program_dependency else ("cta_begin",)),
                ),
            )
            template = (AttentionPipelinePhaseDemand(
                name=f"cta_empty_rows{item.query_rows}",
                stages=stages,
            ),)
            sequence_templates[item] = template
            return template

        prologue_stages = list(sass_stages)
        prologue_predecessors = first_predecessors
        prologue_issue = None
        if grouped_mla_issue is not None or fa2_split_issue is not None:
            prologue_issue = "prologue_issue"
            prologue_issue_instructions = (
                grouped_mla_issue.prologue
                if grouped_mla_issue is not None
                else fa2_split_issue.prologue)
            prologue_stages.append(PipelineStageDemand(
                prologue_issue, issue_service,
                prologue_issue_instructions / total_issue, False,
                predecessors=first_predecessors,
            ))
        if page_table_stage_bytes > 0:
            prologue_stages.append(PipelineStageDemand(
                "load_page", "memory:global_read",
                page_table_stage_bytes / read_bytes, True,
                predecessors=first_predecessors,
            ))
            prologue_predecessors = ("load_page",)
        if has_period32_kv_service:
            prologue_stages.extend((
                PipelineStageDemand(
                    "load_q", "memory:global_read",
                    _cta_query_byte_count(item) / q_read_bytes,
                    include_panel_first_request_latency,
                    predecessors=prologue_predecessors,
                ),
                PipelineStageDemand(
                    "load_k", "memory:global_kv_read",
                    k_stage_bytes / kv_read_bytes,
                    include_panel_first_request_latency,
                    predecessors=prologue_predecessors,
                ),
            ))
            prologue_load_starts = ("load_q", "load_k")
        else:
            prologue_stages.append(PipelineStageDemand(
                "cta_begin",
                "memory:global_read",
                (_cta_query_byte_count(item) + k_stage_bytes) / read_bytes,
                include_panel_first_request_latency,
                predecessors=prologue_predecessors,
            ))
            prologue_load_starts = ("cta_begin",)
        if grouped_mla:
            prologue_stages.append(PipelineStageDemand(
                "panel_stage_k", "resource:smem",
                2.0 * k_stage_bytes / smem_bytes, False,
                predecessors=(
                    ("cta_begin",)
                    if prologue_issue is None
                    else ("cta_begin", prologue_issue)),
            ))
            first_panel_prior_resources = ("resource:smem",)
            repeated_panel_prior_resources = first_panel_prior_resources
        else:
            # cp.async moves Q/K from global memory directly into shared
            # memory.  The source waits for both endpoints at the CTA barrier,
            # but the global transfer and shared-memory writes are one
            # streaming operation rather than two serial full-tile passes.
            prologue_stages.append(PipelineStageDemand(
                "panel_stage_k", "resource:smem",
                (_cta_query_byte_count(item) + k_stage_bytes)
                / smem_bytes,
                False,
                predecessors=prologue_predecessors,
                start_predecessors=prologue_load_starts,
            ))
            first_panel_prior_resources = (
                (("memory:global_read", "memory:global_kv_read",
                  "resource:smem") if has_period32_kv_service else
                 ("memory:global_read", "resource:smem")))
            repeated_panel_prior_resources = (
                (("memory:global_read", "memory:global_kv_read",
                  "resource:smem", "compute:compute_tensor")
                 if has_period32_kv_service else
                 ("memory:global_read", "resource:smem",
                  "compute:compute_tensor")))
        local_spill_cycles_per_cta = (
            _fa2_local_spill_dependency_cycles_per_cta(
                kernel, hardware, item.step_count))
        dense_d128_local_spill = (
            local_spill_cycles_per_cta > 0
            and _is_source_bound_fa2_dense_sass_program(kernel)
            and kernel.variant.lower() == "prefill"
            and head_dim == 128
            and kernel.attributes["causal"] is True)
        if local_spill_cycles_per_cta > 0 and not dense_d128_local_spill:
            prologue_stages.append(PipelineStageDemand(
                "local_spill", dependency_service,
                local_spill_cycles_per_cta / total_dependency,
                False, predecessors=("panel_stage_k",),
            ))

        dense_first_panel_spill_cycles = 0.0
        dense_repeated_panel_spill_cycles = 0.0
        if dense_d128_local_spill:
            dense_model = (
                hardware.
                fa2_dense_d128_causal_local_spill_dependency_cycles)
            # One source panel executes 49+45 recurrent warp instructions;
            # the ten fixed loads and stores join the first panel.  Split the
            # one measured CTA-program round by those static CFG weights so
            # the dependency remains inside the source panel DAG and can
            # overlap other resident CTAs.
            dense_first_panel_spill_cycles = (
                local_spill_cycles_per_cta
                * 114.0 / (94.0 * item.step_count + 20.0))
            dense_repeated_panel_spill_cycles = (
                (local_spill_cycles_per_cta
                 - dense_first_panel_spill_cycles)
                / (item.step_count - 1))
            if (dense_first_panel_spill_cycles <= 0
                    or dense_repeated_panel_spill_cycles <= 0):
                raise ValueError(
                    "dense causal D128 local spill panel service must be "
                    "positive")

        def _panel_stages(*, first_panel, prefetch_next):
            stages = []
            qk = "qk"
            load_v = "load_v"
            softmax = "softmax"
            pv = "pv"
            step = 0 if first_panel else 1
            dependency_panel_cycles = (
                None if source_dependency_mechanism is None
                else _fa2_register_dependency_panel_cycles(
                    source_dependency_mechanism, step))
            qk_predecessors = ()
            panel_prior_resources = (
                first_panel_prior_resources if first_panel
                else repeated_panel_prior_resources)
            qk_prior_resources = panel_prior_resources
            if source_dependency_mechanism is not None:
                pre_qk_barrier = "barrier_before_qk"
                stages.append(PipelineStageDemand(
                    pre_qk_barrier, dependency_service,
                    source_dependency_mechanism["cta_barrier_cycles"]
                    / total_dependency
                    * _resident_dependency_factor("barrier"),
                    False, predecessors=(),
                    prior_resources=panel_prior_resources,
                ))
                qk_predecessors = (pre_qk_barrier,)
                qk_prior_resources = ()
            qk_issue = None
            if grouped_mla_issue is not None or fa2_split_issue is not None:
                qk_issue = "qk_issue"
                qk_issue_instructions = (
                    grouped_mla_issue.qk
                    if grouped_mla_issue is not None
                    else (
                        fa2_split_issue.first_qk_v
                        if first_panel else fa2_split_issue.later_qk_v))
                stages.append(PipelineStageDemand(
                    qk_issue, issue_service,
                    qk_issue_instructions / total_issue, False,
                    predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources,
                ))
            if is_fa2:
                # ldmatrix/SMEM fragment loads are interleaved with the MMA
                # loop across the CTA's warps.  They consume the full SMEM
                # byte demand, while the coarse-grained Tensor stage may
                # start with the SMEM stream instead of after its last byte.
                q_smem_read = "read_q_k"
                stages.append(PipelineStageDemand(
                    q_smem_read, "resource:smem",
                    (q_smem_tile_bytes + k_stage_bytes) / smem_bytes,
                    False, predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources,
                ))
            stages.extend((
                PipelineStageDemand(
                    qk, "compute:compute_tensor",
                    tensor_qk_fraction, False,
                    predecessors=qk_predecessors,
                    start_predecessors=(
                        () if not is_fa2 else (q_smem_read,)),
                    prior_resources=qk_prior_resources,
                ),
                PipelineStageDemand(
                    load_v,
                    ("memory:global_kv_read" if has_period32_kv_service
                     else "memory:global_read"),
                    (v_stage_bytes / kv_read_bytes if has_period32_kv_service
                     else v_stage_bytes / read_bytes),
                    include_panel_first_request_latency,
                    predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources,
                ),
            ))
            stage_v = "stage_v"
            stages.append(PipelineStageDemand(
                stage_v, "resource:smem",
                (v_stage_bytes if is_fa2 else 2.0 * v_stage_bytes)
                / smem_bytes,
                False,
                predecessors=(
                    qk_predecessors if is_fa2 else (load_v,)),
                start_predecessors=(
                    (load_v,) if is_fa2 else ()),
                prior_resources=(
                    qk_prior_resources if is_fa2 else ()),
            ))
            local_spill_stage = None
            if dense_d128_local_spill:
                local_spill_stage = (
                    "local_spill_first"
                    if first_panel else "local_spill_repeated")
                stages.append(PipelineStageDemand(
                    local_spill_stage, dependency_service,
                    (
                        dense_first_panel_spill_cycles
                        if first_panel
                        else dense_repeated_panel_spill_cycles
                    ) / total_dependency,
                    False, predecessors=qk_predecessors,
                    prior_resources=qk_prior_resources,
                ))
            phase_join = (
                ((qk,) + (() if local_spill_stage is None else (
                    local_spill_stage,))
                 + (load_v, stage_v, q_smem_read))
                if is_fa2 else (qk, load_v))
            if qk_issue is not None:
                phase_join += (qk_issue,)
            max_predecessors = phase_join
            if source_dependency_mechanism is not None:
                (predicate_cycles, _, _, _, _) = dependency_panel_cycles
                predicate_stage = "mask_predicate"
                stages.append(PipelineStageDemand(
                    predicate_stage, dependency_service,
                    predicate_cycles / total_dependency
                    * _resident_dependency_factor("predicate"),
                    False, predecessors=(qk,),
                ))
                pre_softmax_barrier = "barrier_before_softmax"
                stages.append(PipelineStageDemand(
                    pre_softmax_barrier, dependency_service,
                    source_dependency_mechanism["cta_barrier_cycles"]
                    / total_dependency
                    * _resident_dependency_factor("barrier"),
                    False, predecessors=(predicate_stage,) + phase_join,
                ))
                phase_join = (pre_softmax_barrier,)
                max_predecessors = phase_join
            load_k_next = None
            stage_k_next = None
            if prefetch_next and grouped_mla_issue is None:
                load_k_next = "panel_k"
                load_k_predecessors = phase_join
                if page_table_stage_bytes > 0:
                    load_page_next = "load_page_next"
                    stages.append(PipelineStageDemand(
                        load_page_next, "memory:global_read",
                        page_table_stage_bytes / read_bytes, True,
                        predecessors=phase_join,
                    ))
                    load_k_predecessors = (load_page_next,)
                # Each source-loop iteration waits for a newly issued K/V
                # panel before its dependent MMA work.  Treating only the
                # prologue as a first request incorrectly turns a short-wave
                # CTA into one continuous full-bandwidth stream and drops the
                # fill latency of every later panel.
                stages.append(PipelineStageDemand(
                    load_k_next,
                    ("memory:global_kv_read" if has_period32_kv_service
                     else "memory:global_read"),
                    (k_stage_bytes / kv_read_bytes
                     if has_period32_kv_service
                     else k_stage_bytes / read_bytes),
                    include_panel_first_request_latency,
                    predecessors=load_k_predecessors,
                ))
                if fa2_split_issue is None:
                    stage_k_next = "panel_stage_k"
                    stages.append(PipelineStageDemand(
                        stage_k_next, "resource:smem",
                        (k_stage_bytes if is_fa2 else 2.0 * k_stage_bytes)
                        / smem_bytes,
                        False,
                        predecessors=(
                            load_k_predecessors if is_fa2
                            else (load_k_next,)),
                        start_predecessors=(
                            (load_k_next,) if is_fa2 else ()),
                    ))
            if has_source_scalar_program:
                _, fma, sfu, _ = (
                    _fa2_scalar_stage_operations(
                        tile_q, tile_kv, head_dim, step))
                softmax_predecessors = max_predecessors
                if source_dependency_mechanism is not None:
                    (_, max_cycles, scale_exp_sum_cycles,
                     conversion_cycles, _) = dependency_panel_cycles
                    max_stage = "max_reduce"
                    stages.append(PipelineStageDemand(
                        max_stage, dependency_service,
                        max_cycles / total_dependency
                        * _resident_dependency_factor("max"),
                        False, predecessors=max_predecessors,
                    ))
                    softmax_predecessors = (max_stage,)
                softmax_pv_issue = None
                if fa2_split_issue is not None:
                    softmax_pv_issue = "softmax_pv_issue"
                    stages.append(PipelineStageDemand(
                        softmax_pv_issue, issue_service,
                        (
                            fa2_split_issue.first_softmax_pv
                            if first_panel
                            else fa2_split_issue.later_softmax_pv
                        ) / total_issue,
                        False,
                        predecessors=phase_join,
                    ))
                stages.extend((
                    PipelineStageDemand(
                        softmax, scalar_service,
                        (
                            _compute_service_time("compute_fma", fma)
                            + _compute_service_time("compute_sfu", sfu)
                        ) / total_scalar_service_time,
                        False,
                        predecessors=softmax_predecessors,
                        start_predecessors=(
                            () if load_k_next is None else (load_k_next,)),
                    ),
                ))
                pv_predecessors = (softmax, stage_v)
                if source_dependency_mechanism is not None:
                    sum_stage = "scale_exp_sum"
                    stages.append(PipelineStageDemand(
                        sum_stage, dependency_service,
                        scale_exp_sum_cycles / total_dependency
                        * _resident_dependency_factor("scale_exp_sum"),
                        False, predecessors=(softmax,),
                    ))
                    conversion_stage = "score_conversion"
                    stages.append(PipelineStageDemand(
                        conversion_stage, dependency_service,
                        conversion_cycles / total_dependency
                        * _resident_dependency_factor("conversion"),
                        False, predecessors=(sum_stage,),
                    ))
                    pv_predecessors = (conversion_stage, stage_v)
                read_v = "read_v"
                stages.append(PipelineStageDemand(
                    read_v, "resource:smem",
                    v_stage_bytes / smem_bytes, False,
                    predecessors=pv_predecessors,
                ))
                stages.append(PipelineStageDemand(
                    pv, "compute:compute_tensor",
                    tensor_pv_fraction, False,
                    predecessors=pv_predecessors,
                    start_predecessors=(read_v,),
                ))
                if fa2_split_issue is not None and prefetch_next:
                    next_panel_issue = "next_panel_issue"
                    next_panel_predecessors = (
                        (pv, softmax_pv_issue)
                        if softmax_pv_issue is not None else (pv,))
                    stages.append(PipelineStageDemand(
                        next_panel_issue, issue_service,
                        fa2_split_issue.next_panel / total_issue,
                        False,
                        predecessors=next_panel_predecessors,
                    ))
                    stage_k_next = "panel_stage_k"
                    stages.append(PipelineStageDemand(
                        stage_k_next, "resource:smem",
                        k_stage_bytes / smem_bytes,
                        False,
                        predecessors=(load_k_next, next_panel_issue),
                    ))
            else:
                softmax_step_operations = float(
                    tile_q * tile_kv + (tile_q if not first_panel else 0))
                softmax_issue = None
                if grouped_mla_issue is not None:
                    softmax_issue = "softmax_issue"
                    stages.append(PipelineStageDemand(
                        softmax_issue, issue_service,
                        grouped_mla_issue.softmax / total_issue, False,
                        predecessors=phase_join,
                    ))
                stages.extend((
                    PipelineStageDemand(
                        softmax, scalar_service,
                        softmax_step_operations / softmax_operations, False,
                        predecessors=phase_join,
                        start_predecessors=(
                            () if load_k_next is None else (load_k_next,)),
                    ),
                ))
                pv_predecessors = (softmax, stage_v)
                if softmax_issue is not None:
                    pv_predecessors += (softmax_issue,)
                pv_issue = None
                if grouped_mla_issue is not None:
                    pv_issue = "pv_issue"
                    stages.append(PipelineStageDemand(
                        pv_issue, issue_service,
                        grouped_mla_issue.pv / total_issue, False,
                        predecessors=pv_predecessors,
                    ))
                stages.append(PipelineStageDemand(
                    pv, "compute:compute_tensor",
                    tensor_pv_fraction, False,
                    predecessors=pv_predecessors,
                ))
                if grouped_mla_issue is not None:
                    next_panel_predecessors = (pv, pv_issue)
                    next_panel_issue = "next_panel_issue"
                    stages.append(PipelineStageDemand(
                        next_panel_issue, issue_service,
                        grouped_mla_issue.next_panel / total_issue, False,
                        predecessors=next_panel_predecessors,
                    ))
                    if prefetch_next:
                        load_k_predecessors = next_panel_predecessors
                        if page_table_stage_bytes > 0:
                            load_page_next = "load_page_next"
                            stages.append(PipelineStageDemand(
                                load_page_next, "memory:global_read",
                                page_table_stage_bytes / read_bytes, True,
                                predecessors=next_panel_predecessors,
                            ))
                            load_k_predecessors = (load_page_next,)
                        load_k_next = "panel_k"
                        stages.append(PipelineStageDemand(
                            load_k_next, "memory:global_read",
                            k_stage_bytes / read_bytes,
                            include_panel_first_request_latency,
                            predecessors=load_k_predecessors,
                        ))
                        stage_k_next = "panel_stage_k"
                        stages.append(PipelineStageDemand(
                            stage_k_next, "resource:smem",
                            2.0 * k_stage_bytes / smem_bytes, False,
                            predecessors=(load_k_next, next_panel_issue),
                        ))
            return tuple(stages)

        tasks = [("prologue", tuple(prologue_stages), 1)]
        if item.step_count == 1:
            tasks.append((
                "first_tail_panel",
                _panel_stages(
                    first_panel=True,
                    prefetch_next=False),
                1,
            ))
        else:
            tasks.append((
                "first_panel",
                _panel_stages(
                    first_panel=True,
                    prefetch_next=True),
                1,
            ))
            if item.step_count > 2:
                tasks.append((
                    "steady_panel",
                    _panel_stages(
                        first_panel=False,
                        prefetch_next=True),
                    item.step_count - 2,
                ))
            tasks.append((
                "tail_panel",
                _panel_stages(
                    first_panel=False,
                    prefetch_next=False),
                1,
            ))

        epilogue_stages = []
        epilogue_issue = None
        if grouped_mla_issue is not None or fa2_split_issue is not None:
            epilogue_issue = "epilogue_issue"
            epilogue_issue_instructions = (
                grouped_mla_issue.epilogue
                if grouped_mla_issue is not None
                else fa2_split_issue.epilogue)
            epilogue_stages.append(PipelineStageDemand(
                epilogue_issue, issue_service,
                epilogue_issue_instructions / total_issue, False,
                predecessors=(),
            ))
        program_dependency = None
        if has_program_dependency:
            program_dependency = "program_dependency"
            epilogue_stages.append(PipelineStageDemand(
                program_dependency, dependency_service,
                sass_issue / total_dependency, False,
                predecessors=(),
            ))
        output_local_predecessors = ()
        output_prior_resources = (
            ("resource:smem",) if is_fa2
            else ("compute:compute_tensor",))
        if has_source_scalar_program:
            if source_dependency_mechanism is not None:
                quad_sum = "quad_sum_epilogue"
                epilogue_stages.append(PipelineStageDemand(
                    quad_sum, dependency_service,
                    source_dependency_mechanism["fp32_quad4_sum_cycles"]
                    / total_dependency
                    * _resident_dependency_factor("quad_sum"),
                    False, predecessors=(),
                    prior_resources=output_prior_resources,
                ))
                output_local_predecessors = (quad_sum,)
                output_prior_resources = ()
            epilogue_fma, epilogue_sfu, _ = (
                _fa2_scalar_epilogue_operations(tile_q, head_dim))
            epilogue_stages.append(PipelineStageDemand(
                "normalize_output", scalar_service,
                (
                    _compute_service_time(
                        "compute_fma", epilogue_fma)
                    + _compute_service_time(
                        "compute_sfu", epilogue_sfu)
                ) / total_scalar_service_time,
                False, predecessors=output_local_predecessors,
                prior_resources=output_prior_resources,
            ))
            output_local_predecessors = ("normalize_output",)
            output_prior_resources = ()
        if source_dependency_mechanism is not None:
            output_conversion_cycles = (
                source_dependency_mechanism[
                    "fp32_to_bf16_conversion_cycles"]
                * head_dim / tile_kv)
            output_conversion = "output_conversion"
            epilogue_stages.append(PipelineStageDemand(
                output_conversion, dependency_service,
                output_conversion_cycles / total_dependency
                * _resident_dependency_factor("conversion"),
                False, predecessors=output_local_predecessors,
                prior_resources=output_prior_resources,
            ))
            output_local_predecessors = (output_conversion,)
            output_prior_resources = ()
        if is_fa2:
            output_smem = "stage_output"
            epilogue_stages.append(PipelineStageDemand(
                output_smem, "resource:smem",
                2.0 * output_smem_tile_bytes / smem_bytes,
                False, predecessors=output_local_predecessors,
                prior_resources=output_prior_resources,
            ))
            output_local_predecessors = (output_smem,)
            output_prior_resources = ()
        if source_dependency_mechanism is not None:
            output_barrier = "barrier_before_output"
            epilogue_stages.append(PipelineStageDemand(
                output_barrier, dependency_service,
                source_dependency_mechanism["cta_barrier_cycles"]
                / total_dependency
                * _resident_dependency_factor("barrier"),
                False, predecessors=output_local_predecessors,
                prior_resources=output_prior_resources,
            ))
            output_local_predecessors = (output_barrier,)
            output_prior_resources = ()
        output_predecessors = (
            output_local_predecessors
            if program_dependency is None
            else output_local_predecessors + (program_dependency,))
        if epilogue_issue is None:
            epilogue_stages.append(PipelineStageDemand(
                "cta_end",
                "memory:global_write",
                _cta_output_byte_count(item) / write_bytes,
                True,
                predecessors=output_predecessors,
                prior_resources=output_prior_resources,
            ))
        else:
            epilogue_stages.extend((
                PipelineStageDemand(
                    "store_output",
                    "memory:global_write",
                    _cta_output_byte_count(item) / write_bytes,
                    True,
                    predecessors=output_predecessors,
                    prior_resources=output_prior_resources,
                ),
                PipelineStageDemand(
                    "cta_end", issue_service,
                    (
                        grouped_mla_issue.cta_end
                        if grouped_mla_issue is not None
                        else fa2_split_issue.cta_end
                    ) / total_issue,
                    False,
                    predecessors=("store_output", epilogue_issue),
                ),
            ))
        tasks.append(("epilogue", tuple(epilogue_stages), 1))
        template = tuple(AttentionPipelinePhaseDemand(
            name=name,
            stages=stages,
            repeat_count=repeat_count,
        ) for name, stages, repeat_count in tasks)
        sequence_templates[item] = template
        return template

    classes = tuple(
        AttentionPipelineClassDemand(
            name=(f"run{run_index}_rows{item.query_rows}_"
                  f"steps{item.step_count}"),
            count=count,
            query_rows=item.query_rows,
            panel_count=item.step_count,
            phases=_cta_phases(item),
        )
        for run_index, (item, count) in enumerate(schedule_class_counts))
    plan = AttentionPipelineEnvelopeDemand(
        active_lane_count=active_lane_count,
        classes=classes,
        source_repeat_count=source_repeat_count,
        cta_admission_policy=(
            "paired_resident_fifo"
            if (paired_fifo_admission or split3_varlen_admission) else (
                "bounded_resident_suffix_fifo"
                if bounded_fifo_drain else "round_robin")),
        buffers=(PipelineBuffer(
            name="resident_cta",
            producer_stage=(
                "load_q" if has_period32_kv_service else "cta_begin"),
            consumer_stage="cta_end",
            slots=(
                _A800_GROUPED_MLA_STAGE1_RESIDENT_CTA_SLOTS
                if grouped_mla else _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS),
            scope="worker",
        ),),
        resident_tail_latency_service=(
            "memory:global_read" if resident_frontend_overlap else None),
        resident_tail_requests_per_panel=(
            2 if resident_frontend_overlap else 0),
    )
    return attach_attention_pipeline_envelope(inp, hardware, plan)


def _attach_gpu_fa2_combine_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the pinned FA2 split-combine CTA program.

    Dao FA2 2.8.3 fixes one 128-thread CTA program: load the split LSE tile,
    reduce it, publish final LSE, form split scales, then synchronously repeat
    ``load partial O -> FMA accumulate`` before the final output store.  The
    CTA residency is intentionally unmodeled; no cross-CTA buffer credit is
    introduced.
    """

    PipelineStageDemand = bind_pipeline_stage_template
    if not _is_source_bound_fa2_combine(kernel):
        raise ValueError(
            "FA2 combine pipeline requires the pinned SM80 source family")
    _, _, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    positions = sum(q_lens) * query_heads
    block_m = _fa2_combine_block_m(kernel)
    cta_count = kernel.launch.grid_size
    if cta_count != math.ceil(positions / block_m):
        raise ValueError(
            "FA2 combine grid disagrees with its source-defined kBlockM")
    worker_count = min(cta_count, inp.active_units)
    if worker_count <= 0:
        raise ValueError("FA2 combine pipeline has no active worker")

    paths = {path.name: path for path in inp.memory_paths}
    if set(paths) != {"global_read", "global_write"}:
        raise ValueError(
            "FA2 combine requires separate global read/write paths")
    read_bytes = paths["global_read"].nodes[0].bytes
    write_bytes = paths["global_write"].nodes[0].bytes
    splits = _splits(kernel)
    max_splits = _fa2_combine_max_splits(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    expected_read = float(positions * splits * (head_dim + 1) * 4)
    expected_write = float(positions * (head_dim * element_bytes + 4))
    if not math.isclose(
            read_bytes, expected_read, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "FA2 combine read path disagrees with source scratch traffic")
    if not math.isclose(
            write_bytes, expected_write, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "FA2 combine write path disagrees with source output traffic")

    compute = {demand.resource_role: demand
               for demand in inp.compute_demands}
    expected_compute_roles = {
        "compute_fma", "compute_sfu", "compute_reduction"}
    if set(compute) != expected_compute_roles:
        raise ValueError(
            "FA2 combine requires FMA, SFU, and dependency demand")

    def _compute_time(role, operations):
        demand = compute[role]
        resource = hardware.resource_by_role(role)
        capacity = resource.effective_capacity(
            inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
        schedule_efficiency = min(
            1.0, demand.schedule_efficiency / inp.wave_efficiency)
        return operations / (
            capacity * demand.shape_efficiency * schedule_efficiency)

    padded_positions = cta_count * block_m
    expected_fma = float(cta_count * (
        2 * splits * head_dim * block_m
        + 2 * kernel.launch.block_size + splits * block_m))
    expected_sfu = float(cta_count * (
        2 * kernel.launch.block_size + splits * block_m))
    if not math.isclose(
            compute["compute_fma"].operations, expected_fma,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("FA2 combine FMA demand disagrees with its CTA tile")
    if not math.isclose(
            compute["compute_sfu"].operations, expected_sfu,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("FA2 combine SFU demand disagrees with its CTA tile")

    compute_service = pipeline_compute_service(
        inp, hardware, roles=("compute_fma", "compute_sfu"))
    dependency_service = pipeline_compute_service(
        inp, hardware, roles=("compute_reduction",))
    dependency_stages = _fa2_combine_dependency_stages(kernel, hardware)
    oaccum_load_use = _fa2_oaccum_load_use_cycles(kernel, hardware)
    dependency_per_cta = (
        math.fsum(dependency_stages.values())
        + math.fsum(oaccum_load_use))
    total_dependency = float(inp.demand_per_resource["compute_reduction"])
    if not math.isclose(
            total_dependency, dependency_per_cta * cta_count,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "FA2 combine dependency demand disagrees with source CTA count")
    total_compute_time = (
        _compute_time("compute_fma", expected_fma)
        + _compute_time("compute_sfu", expected_sfu))
    post_max_fraction = (
        _compute_time("compute_fma", float(kernel.launch.block_size))
        + _compute_time("compute_sfu", float(kernel.launch.block_size))
    ) / total_compute_time
    post_sum_fraction = (
        _compute_time("compute_fma", float(kernel.launch.block_size))
        + _compute_time("compute_sfu", float(kernel.launch.block_size))
    ) / total_compute_time
    scale_fraction = (
        _compute_time("compute_fma", float(splits * block_m))
        + _compute_time("compute_sfu", float(splits * block_m))
    ) / total_compute_time
    accumulate_fraction = _compute_time(
        "compute_fma", float(2 * head_dim * block_m)
    ) / total_compute_time

    def _cta_task(active_rows, worker):
        if active_rows <= 0 or active_rows > block_m:
            raise ValueError("FA2 combine produced an invalid CTA row class")
        stages = [PipelineStageDemand(
                "load_lse", "memory:global_read",
                active_rows * splits * 4 / read_bytes, True,
                predecessors=(),
            ), PipelineStageDemand(
                "slse_publish_transpose", dependency_service,
                dependency_stages["slse_publish_transpose"]
                / total_dependency, False,
                predecessors=("load_lse",),
            )]
        stages.extend((
            PipelineStageDemand(
                "max_allreduce", dependency_service,
                dependency_stages["max_allreduce"] / total_dependency, False,
                predecessors=("slse_publish_transpose",),
            ),
            PipelineStageDemand(
                "post_max_dependency", dependency_service,
                dependency_stages["post_max_exp_local_sum"]
                / total_dependency, False,
                predecessors=("max_allreduce",),
            ),
            PipelineStageDemand(
                "post_max_exp_local_sum", compute_service,
                post_max_fraction, False,
                predecessors=("max_allreduce",),
            ),
            PipelineStageDemand(
                "sum_allreduce", dependency_service,
                dependency_stages["sum_allreduce"] / total_dependency, False,
                predecessors=(
                    "post_max_dependency", "post_max_exp_local_sum"),
            ),
            PipelineStageDemand(
                "post_sum_dependency", dependency_service,
                dependency_stages["post_sum_log"] / total_dependency, False,
                predecessors=("sum_allreduce",),
            ),
            PipelineStageDemand(
                "post_sum_log", compute_service,
                post_sum_fraction, False,
                predecessors=("sum_allreduce",),
            ),
            PipelineStageDemand(
                "store_lse", "memory:global_write",
                active_rows * 4 / write_bytes, True,
                predecessors=("post_sum_dependency", "post_sum_log"),
            ),
            PipelineStageDemand(
                "scale_dependency", dependency_service,
                dependency_stages["scale_exp_smem_barrier"]
                / total_dependency, False,
                predecessors=("post_sum_dependency", "post_sum_log"),
                start_predecessors=("store_lse",),
            ),
            PipelineStageDemand(
                "scale_lse", compute_service, scale_fraction, False,
                predecessors=("post_sum_dependency", "post_sum_log"),
                start_predecessors=("store_lse",),
            ),
        ))
        predecessors = ("scale_dependency", "scale_lse")
        for split in range(splits):
            load = f"load_o_{split}"
            load_ready = f"load_o_ready_{split}"
            accumulate = f"accumulate_o_{split}"
            stages.extend((
                PipelineStageDemand(
                    load, "memory:global_read",
                    active_rows * head_dim * 4 / read_bytes, True,
                    predecessors=predecessors,
                ),
                PipelineStageDemand(
                    load_ready, dependency_service,
                    oaccum_load_use[split] / total_dependency, False,
                    predecessors=predecessors,
                ),
                PipelineStageDemand(
                    accumulate, compute_service,
                    accumulate_fraction, False,
                    predecessors=(load, load_ready),
                ),
            ))
            predecessors = (accumulate,)
        stages.extend((PipelineStageDemand(
            "output_conversion", dependency_service,
            dependency_stages["output_conversion"] / total_dependency, False,
            predecessors=predecessors,
        ), PipelineStageDemand(
            "store_output", "memory:global_write",
            active_rows * head_dim * element_bytes / write_bytes, True,
            predecessors=("output_conversion",),
        )))
        return PipelineTaskDemand(
            name=f"combine_rows{active_rows}",
            worker=worker,
            stages=tuple(stages),
        )

    full_ctas, tail_rows = divmod(positions, block_m)
    row_class_counts = []
    if full_ctas:
        row_class_counts.append((block_m, full_ctas))
    if tail_rows:
        row_class_counts.append((tail_rows, 1))
    worker_groups = _balanced_cta_class_worker_groups(
        tuple(row_class_counts), worker_count)
    tasks = []
    for worker, (worker_class_counts, multiplicity) in enumerate(
            worker_groups):
        active_classes = tuple(
            (class_index, count)
            for class_index, count in enumerate(worker_class_counts)
            if count > 0)
        patterns = tuple(
            PipelineTaskSequenceDemand(
                name=f"rows{row_class_counts[class_index][0]}",
                worker=worker,
                tasks=(_cta_task(
                    row_class_counts[class_index][0], worker),),
                repeat_count=count,
            )
            for class_index, count in active_classes)
        tasks.append(PipelineTaskProgramDemand(
            name=f"worker_class{worker}_combine_program",
            worker=worker,
            patterns=patterns,
            order=tuple(range(len(patterns))),
            repeat_count=1,
            tail_order=(),
            worker_multiplicity=multiplicity,
        ))
    plan = PipelineWorkPlan(
        worker_count=worker_count,
        tasks=tuple(tasks),
        buffers=(),
    )
    return attach_pipeline_work_plan(inp, hardware, plan)


def _attach_metax_fa2_combine_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach MACA 3.7.2's source-ordered D128 split reducer."""

    PipelineStageDemand = bind_pipeline_stage_template
    if not _is_source_bound_metax_fa2_combine(kernel):
        raise ValueError(
            "MetaX FA2 combine pipeline requires its pinned source family")
    _, _, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    positions = sum(q_lens) * query_heads
    block_m = 8
    cta_count = kernel.launch.grid_size
    if cta_count != math.ceil(positions / block_m):
        raise ValueError("MetaX FA2 combine grid disagrees with kBlockM=8")
    if inp.active_units <= 0:
        raise ValueError("MetaX FA2 combine has no active AP")

    paths = {path.name: path for path in inp.memory_paths}
    if set(paths) != {"global_read", "global_write"}:
        raise ValueError(
            "MetaX FA2 combine requires separate global read/write paths")
    read_bytes = paths["global_read"].nodes[0].bytes
    write_bytes = paths["global_write"].nodes[0].bytes
    splits = _splits(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    expected_read = float(positions * splits * (head_dim + 1) * 4)
    output_bytes_per_row = float(head_dim*element_bytes)
    if has_fa2_split_source(kernel):
        source = current_fa2_split_source_work(kernel)
        expected_read = float(source['partial_bytes'])
        output_bytes_per_row = source['priced_write_bytes']/positions
    expected_write = float(positions*output_bytes_per_row)
    if not math.isclose(
            read_bytes, expected_read, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "MetaX FA2 combine read path disagrees with scratch traffic")
    if not math.isclose(
            write_bytes, expected_write, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "MetaX FA2 combine write path disagrees with output traffic")

    compute = {demand.resource_role: demand
               for demand in inp.compute_demands}
    if set(compute) != {"compute_fma", "compute_reduction"}:
        raise ValueError(
            "MetaX FA2 combine requires Oaccum FMA and dependency demand")

    def _compute_time(role, operations):
        demand = compute[role]
        resource = hardware.resource_by_role(role)
        capacity = resource.effective_capacity(
            inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
        schedule_efficiency = min(
            1.0, demand.schedule_efficiency / inp.wave_efficiency)
        return operations / (
            capacity * demand.shape_efficiency * schedule_efficiency)

    total_fma = float(compute["compute_fma"].operations)
    accumulate_fma = float(2 * splits * head_dim * positions)
    if not math.isclose(
            total_fma, accumulate_fma, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError(
            "MetaX FA2 combine FMA demand must contain only Oaccum rounds")
    total_compute_time = _compute_time("compute_fma", total_fma)
    compute_service = pipeline_compute_service(
        inp, hardware, roles=("compute_fma",))
    dependency_service = pipeline_compute_service(
        inp, hardware, roles=("compute_reduction",))
    dependency_stages = _metax_fa2_combine_dependency_stages(
        kernel, hardware)
    dependency_per_cta = math.fsum(dependency_stages.values())
    total_dependency = float(inp.demand_per_resource["compute_reduction"])
    if not math.isclose(
            total_dependency, dependency_per_cta * cta_count,
            rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            "MetaX FA2 dependency demand disagrees with source CTA count")

    def _cta_stages(active_rows):
        if active_rows <= 0 or active_rows > block_m:
            raise ValueError("MetaX FA2 combine produced invalid active rows")
        accumulate_fraction = _compute_time(
            "compute_fma", float(2 * head_dim * active_rows)
        ) / total_compute_time
        stages = [
            PipelineStageDemand(
                "load_lse", "memory:global_read",
                active_rows * splits * 4 / read_bytes, True,
                predecessors=()),
            PipelineStageDemand(
                "lse_shared_transpose", dependency_service,
                dependency_stages["lse_shared_transpose"]
                / total_dependency, False,
                predecessors=("load_lse",)),
            PipelineStageDemand(
                "max_allreduce", dependency_service,
                dependency_stages["max_allreduce"]
                / total_dependency, False,
                predecessors=("lse_shared_transpose",)),
            PipelineStageDemand(
                "post_max_dependency", dependency_service,
                dependency_stages["post_max_exp_sum"]
                / total_dependency, False,
                predecessors=("max_allreduce",)),
            PipelineStageDemand(
                "sum_allreduce", dependency_service,
                dependency_stages["sum_allreduce"]
                / total_dependency, False,
                predecessors=("post_max_dependency",)),
            PipelineStageDemand(
                "post_sum_log", dependency_service,
                dependency_stages["post_sum_log"]
                / total_dependency, False,
                predecessors=("sum_allreduce",)),
            PipelineStageDemand(
                "scale_shared_barrier", dependency_service,
                dependency_stages["scale_shared_barrier"]
                / total_dependency, False,
                predecessors=("post_sum_log",)),
        ]
        predecessors = ("scale_shared_barrier",)
        for split in range(splits):
            load = f"load_o_{split}"
            accumulate = f"accumulate_o_{split}"
            stages.extend((
                PipelineStageDemand(
                    load, "memory:global_read",
                    active_rows * head_dim * 4 / read_bytes, True,
                    predecessors=predecessors),
                PipelineStageDemand(
                    accumulate, compute_service,
                    accumulate_fraction, False,
                    predecessors=(load,)),
            ))
            predecessors = (accumulate,)
        stages.extend((
            PipelineStageDemand(
                "output_conversion", dependency_service,
                dependency_stages["output_conversion"]
                / total_dependency, False,
                predecessors=predecessors),
            PipelineStageDemand(
                "store_output", "memory:global_write",
                active_rows * output_bytes_per_row / write_bytes, True,
                predecessors=("output_conversion",)),
        ))
        return tuple(stages)

    full_ctas, tail_rows = divmod(positions, block_m)
    classes = []
    if full_ctas:
        classes.append(AttentionPipelineClassDemand(
            name=f"metax_combine_rows{block_m}", count=full_ctas,
            query_rows=block_m, panel_count=0,
            phases=(AttentionPipelinePhaseDemand(
                name="combine", stages=_cta_stages(block_m)),),
        ))
    if tail_rows:
        classes.append(AttentionPipelineClassDemand(
            name=f"metax_combine_rows{tail_rows}", count=1,
            query_rows=tail_rows, panel_count=0,
            phases=(AttentionPipelinePhaseDemand(
                name="combine", stages=_cta_stages(tail_rows)),),
        ))
    # The runtime-query report for the exact kMaxSplits=8 reducer establishes
    # six 256-thread CTA contexts per C500 AP.  The 224-CTA frozen launch fits
    # entirely in that resident cohort; lane-local lifecycle credits model
    # the simultaneously live CTAs rather than serializing them one per AP.
    resident_cta_slots = 6
    return attach_attention_pipeline_envelope(
        inp, hardware,
        AttentionPipelineEnvelopeDemand(
            active_lane_count=inp.active_units,
            classes=tuple(classes),
            buffers=(PipelineBuffer(
                name="resident_cta_lifecycle", producer_stage="load_lse",
                consumer_stage="store_output", slots=resident_cta_slots,
                scope="worker"),),
        ))


def _attach_gpu_grouped_mla_combine_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach vLLM 0.11's source-defined Triton MLA stage-2 loop."""

    PipelineStageDemand = bind_pipeline_stage_template
    if not _is_source_bound_grouped_mla_combine(kernel):
        raise ValueError(
            "grouped MLA combine pipeline requires its pinned source family")
    _, _, _, head_dim, query_heads, _ = _shape(kernel)
    q_lens, _ = _request_lengths(kernel)
    positions = sum(q_lens) * query_heads
    if kernel.launch.grid_size != positions:
        raise ValueError("grouped MLA combine requires one CTA per output row")
    # One pipeline worker represents one independently progressing resident
    # CTA context, not one whole SM.  Stage-2 has no cross-CTA dependency and
    # its static launch/resources admit sixteen 128-thread CTAs per SM.  The
    # fractional stage service still conserves the whole-launch memory and
    # compute demand as worker_count grows; the extra contexts only allow their
    # dependent load/update chains to overlap.
    worker_count = min(
        positions,
        inp.active_units * _A800_GROUPED_MLA_STAGE2_RESIDENT_CTA_SLOTS,
    )
    splits = _splits(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])

    paths = {path.name: path for path in inp.memory_paths}
    if set(paths) != {"global_read", "global_write"}:
        raise ValueError(
            "grouped MLA combine requires separate global read/write paths")
    read_bytes = paths["global_read"].nodes[0].bytes
    write_bytes = paths["global_write"].nodes[0].bytes
    expected_read = float(positions * splits * (head_dim + 1) * 4)
    output_bytes_per_row = float(head_dim*element_bytes+4)
    if has_grouped_mla_source(kernel):
        output_bytes_per_row = current_grouped_mla_source_work(kernel)['priced_write_bytes']/positions
    expected_write = float(positions*output_bytes_per_row)
    if not math.isclose(read_bytes, expected_read, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("grouped MLA combine read traffic is inconsistent")
    if not math.isclose(
            write_bytes, expected_write, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("grouped MLA combine write traffic is inconsistent")

    compute = {demand.resource_role: demand for demand in inp.compute_demands}
    if set(compute) != {"compute_fma", "compute_sfu"}:
        raise ValueError(
            "grouped MLA combine requires exactly FMA and SFU demand")

    def _service_time(role, operations):
        demand = compute[role]
        resource = hardware.resource_by_role(role)
        capacity = resource.effective_capacity(
            inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
        schedule_efficiency = min(
            1.0, demand.schedule_efficiency / inp.wave_efficiency)
        return operations / (
            capacity * demand.shape_efficiency * schedule_efficiency)

    per_split_fma = float(3 * head_dim + 2)
    final_fma = float(head_dim)
    per_split_sfu = 2.0
    final_sfu = 2.0
    expected_fma = positions * (splits * per_split_fma + final_fma)
    expected_sfu = positions * (splits * per_split_sfu + final_sfu)
    if not math.isclose(
            compute["compute_fma"].operations, expected_fma,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("grouped MLA combine FMA work is inconsistent")
    if not math.isclose(
            compute["compute_sfu"].operations, expected_sfu,
            rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("grouped MLA combine SFU work is inconsistent")
    total_compute_time = (
        _service_time("compute_fma", expected_fma)
        + _service_time("compute_sfu", expected_sfu))
    split_compute_fraction = (
        _service_time("compute_fma", per_split_fma)
        + _service_time("compute_sfu", per_split_sfu)
    ) / total_compute_time
    final_compute_fraction = (
        _service_time("compute_fma", final_fma)
        + _service_time("compute_sfu", final_sfu)
    ) / total_compute_time
    compute_service = pipeline_compute_service(
        inp, hardware, roles=("compute_fma", "compute_sfu"))

    def _cta_task(worker):
        stages = []
        predecessor = None
        for split in range(splits):
            load = f"load_split_{split}"
            update = f"update_split_{split}"
            # The Triton loop consumes each partial row before advancing to
            # the next split, so every split begins a new dependent memory
            # round rather than sharing one launch-wide first request.
            stages.append(PipelineStageDemand(
                load, "memory:global_read",
                (head_dim + 1) * 4 / read_bytes,
                True,
                predecessors=(() if predecessor is None else (predecessor,)),
            ))
            stages.append(PipelineStageDemand(
                update, compute_service, split_compute_fraction, False,
                predecessors=(load,),
            ))
            predecessor = update
        stages.extend((
            PipelineStageDemand(
                "normalize", compute_service, final_compute_fraction, False,
                predecessors=(predecessor,),
            ),
            PipelineStageDemand(
                "store_output", "memory:global_write",
                output_bytes_per_row / write_bytes,
                True, predecessors=("normalize",),
            ),
        ))
        return PipelineTaskDemand(
            name="grouped_mla_combine_row", worker=worker,
            stages=tuple(stages))

    base_ctas, extra_workers = divmod(positions, worker_count)
    worker_groups = tuple(
        item for item in (
            (base_ctas + 1, extra_workers),
            (base_ctas, worker_count - extra_workers),
        ) if item[1] > 0)
    tasks = tuple(
        PipelineTaskProgramDemand(
            name=f"worker_class{worker}_grouped_mla_combine",
            worker=worker,
            patterns=(PipelineTaskSequenceDemand(
                name="row_ctas",
                worker=worker,
                tasks=(_cta_task(worker),),
                repeat_count=ctas_per_worker,
            ),),
            order=(0,),
            worker_multiplicity=multiplicity,
        )
        for worker, (ctas_per_worker, multiplicity) in enumerate(worker_groups)
    )
    return attach_pipeline_work_plan(
        inp, hardware,
        PipelineWorkPlan(worker_count=worker_count, tasks=tasks, buffers=()))


def _attach_metax_mla_combine_pipeline(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    """Attach the source-defined per-request FlashMLA reduction loop."""

    PipelineStageDemand = bind_pipeline_stage_template
    if not _is_source_bound_metax_mla_combine(kernel):
        raise ValueError(
            "MetaX FlashMLA combine pipeline requires its source ABI")
    work = _metax_mla_scheduler_work(kernel)
    batch, q_len, _, head_dim, query_heads, _ = _shape(kernel)
    block_m = _metax_mla_combine_block_m(kernel)
    worker_count = inp.active_units
    if worker_count <= 0:
        raise ValueError("MetaX FlashMLA combine requires active APs")

    paths = {path.name: path for path in inp.memory_paths}
    if set(paths) != {"global_read", "global_write"}:
        raise ValueError(
            "MetaX FlashMLA combine requires separate read/write paths")
    read_bytes = paths["global_read"].nodes[0].bytes
    write_bytes = paths["global_write"].nodes[0].bytes
    compute = {demand.resource_role: demand for demand in inp.compute_demands}
    if set(compute) != {
            "compute_fma", "compute_sfu", "compute_reduction"}:
        raise ValueError(
            "MetaX FlashMLA combine requires FMA, SFU, and barrier demand")

    def _service_time(role, operations):
        demand = compute[role]
        resource = hardware.resource_by_role(role)
        capacity = resource.effective_capacity(
            inp.active_units, inp.sm_freq_hz, inp.dram_freq_hz)
        schedule_efficiency = min(
            1.0, demand.schedule_efficiency / inp.wave_efficiency)
        return operations / (
            capacity * demand.shape_efficiency * schedule_efficiency)

    total_compute_time = sum(
        _service_time(role, compute[role].operations)
        for role in ("compute_fma", "compute_sfu"))
    compute_service = pipeline_compute_service(
        inp, hardware, roles=("compute_fma", "compute_sfu"))
    barrier_service = pipeline_compute_service(
        inp, hardware, roles=("compute_reduction",))
    barrier_mechanism = hardware.flash_mla_combine_barrier_service_cycles
    if barrier_mechanism is None:
        raise ValueError(
            "MetaX FlashMLA combine requires CTA-barrier evidence")
    total_barrier_cycles = compute["compute_reduction"].operations
    if total_barrier_cycles <= 0:
        raise ValueError(
            "MetaX FlashMLA combine requires positive CTA-barrier work")
    barrier_fraction = (
        barrier_mechanism["barrier_cycles"] / total_barrier_cycles)
    element_bytes = int(kernel.shape["dtype_bytes"])

    class_counts = {}
    rows_per_request = q_len * query_heads
    if rows_per_request % block_m:
        raise ValueError("MetaX FlashMLA combine rows are not CTA-aligned")
    ctas_per_request = rows_per_request // block_m
    for split_count in work.splits_per_request:
        if split_count > 1:
            class_counts[split_count] = (
                class_counts[split_count] + ctas_per_request
                if split_count in class_counts else ctas_per_request)
    scheduled_ctas = batch * ctas_per_request
    if scheduled_ctas != kernel.launch.grid_size:
        raise ValueError(
            "MetaX FlashMLA combine CTA enumeration disagrees with grid")

    def _cta_phases(split_count):
        lse_read = float(block_m * split_count * 4)
        partial_read = float(block_m * head_dim * 4)
        output_write = float(
            block_m * (head_dim * element_bytes + 4))
        lse_fma = float(block_m * (2 * 128 - 1))
        lse_sfu = float(block_m * (128 + split_count + 1))
        split_fma = float(2 * block_m * head_dim)
        lse_compute_fraction = (
            _service_time("compute_fma", lse_fma)
            + _service_time("compute_sfu", lse_sfu)
        ) / total_compute_time
        split_compute_fraction = (
            _service_time("compute_fma", split_fma)
            / total_compute_time)
        prologue = (
            PipelineStageDemand(
                "load_lse", "memory:global_read", lse_read / read_bytes,
                True, predecessors=()),
            PipelineStageDemand(
                "reduce_lse", compute_service, lse_compute_fraction,
                predecessors=("load_lse",)),
        )
        partial = (
            PipelineStageDemand(
                "load_partial", "memory:global_read",
                partial_read / read_bytes,
                True, predecessors=(),
                prior_resources=(compute_service,)),
            PipelineStageDemand(
                "partial_ready_barrier", barrier_service, barrier_fraction,
                predecessors=("load_partial",)),
            PipelineStageDemand(
                "update_partial", compute_service, split_compute_fraction,
                predecessors=("partial_ready_barrier",)),
        )
        epilogue = (PipelineStageDemand(
            "store", "memory:global_write", output_write / write_bytes,
            True, predecessors=(),
            prior_resources=(compute_service,)),)
        return (
            AttentionPipelinePhaseDemand(
                name="prologue", stages=prologue),
            AttentionPipelinePhaseDemand(
                name="partial", stages=partial,
                repeat_count=split_count),
            AttentionPipelinePhaseDemand(
                name="epilogue", stages=epilogue),
        )

    active_cta_count = sum(class_counts.values())
    if active_cta_count <= 0:
        raise ValueError("MetaX FlashMLA combine has no active split CTA")
    classes = tuple(
        AttentionPipelineClassDemand(
            name=f"split{split_count}",
            count=count,
            query_rows=block_m,
            panel_count=split_count,
            phases=_cta_phases(split_count),
        )
        for split_count, count in class_counts.items())
    limitations = tuple(
        value for value in inp.limitations
        if value != "gpu_flash_attention_combine_pipeline_evidence_unavailable")
    return attach_attention_pipeline_envelope(
        dataclasses.replace(
            inp,
            support_level="partial",
            limitations=limitations,
        ),
        hardware,
        AttentionPipelineEnvelopeDemand(
            active_lane_count=min(active_cta_count, worker_count),
            classes=classes,
            buffers=(),
        ),
    )


def _architecture_input(
        inp: PredictionInput, kernel: KernelSpec, hardware) -> PredictionInput:
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        return dataclasses.replace(inp, mem_model="npu_tile"), None
    elif memory_model == "gpu_mshr":
        return inp, None
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            "fused FLASH_ATTN is not modeled for PIM-bank or "
            "systolic-scratchpad execution; use EAGER_ATTN for the "
            "decomposed operator")
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")


def _target_mla_attention_elements(kernel: KernelSpec) -> int:
    q_lens = tuple(int(value) for value in kernel.attributes["q_lens"])
    kv_lens = tuple(int(value) for value in kernel.attributes["kv_lens"])
    causal = kernel.attributes["causal"]
    if not isinstance(causal, bool):
        raise TypeError("target MLA causal must be boolean")
    elements = 0
    for query, key_value in zip(q_lens, kv_lens):
        if query > key_value:
            raise ValueError("target MLA requires query length <= KV length")
        if causal:
            context = key_value - query
            elements += query * (2 * context + query + 1) // 2
        else:
            elements += query * key_value
    if elements <= 0:
        raise ValueError("target MLA requires positive attention work")
    return elements


def _target_mla_model(kernel: KernelSpec, hardware):
    """Price Kimi MLA from the pinned vLLM execution semantics.

    This is deliberately an aggregate analytical contract.  It uses the
    source-selected FA2 prefill or Triton grouped-decode geometry, semantic
    tensor work, and explicit tensor/scratch traffic; no production duration
    or workload-specific latency coefficient enters the model.
    """

    if hardware.kernel_memory_model != "gpu_mshr":
        raise NotImplementedError(
            "target MLA analytical execution is currently A800 GPU only")
    variant = kernel.variant.lower()
    if variant not in ("mla_prefill", "mla_decode"):
        raise ValueError(f"unknown target MLA variant: {variant!r}")
    batch, q_len, kv_len, query_heads, qk_head_dim, value_head_dim, kv_lora = (
        int(kernel.shape[name]) for name in (
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank"))
    del batch, q_len, kv_len
    q_lens = tuple(int(value) for value in kernel.attributes["q_lens"])
    kv_lens = tuple(int(value) for value in kernel.attributes["kv_lens"])
    query_rows = sum(q_lens)
    kv_rows = sum(kv_lens)
    attention_elements = _target_mla_attention_elements(kernel)
    element_bytes = int(kernel.shape["dtype_bytes"])
    splits = _splits(kernel)
    if variant == "mla_prefill":
        if ((qk_head_dim, value_head_dim, kv_lora)
                != (192, 128, 512)):
            raise ValueError("Kimi MLA prefill dimensions changed")
        if splits != 1:
            raise ValueError("A800 Kimi MLA prefill must not split KV")
        # SM80 FA2 pads the unequal V head from 128 to the 192-wide QK head.
        effective_value_dim = qk_head_dim
        limitations = ("a800_mla_prefill_v_head_padding_included", "a800_mla_prefill_fa2_worker_pipeline_projected")
        tensor_value_dim = qk_head_dim
    elif variant == "mla_decode":
        if ((qk_head_dim, value_head_dim, kv_lora)
                != (576, 512, 512)):
            raise ValueError("Kimi MLA decode dimensions changed")
        # The Triton MLA source loads one 576-wide compressed cache row and
        # reinterprets its first 512 channels as V; it does not issue a second
        # V-cache load.  Its grid carries one program row per query token
        # (q.shape[0]), so an MTP/EAGLE verify window (q_len > 1) launches
        # q_len rows per request; each row streams the request's compressed
        # cache through L2, while the co-resident window rows of one request
        # share the same DRAM lines.
        limitations = (
            "a800_mla_decode_stage1_program_projected" if splits>1 else "a800_mla_decode_unsplit_program_aggregated",
            "a800_mla_decode_worker_pipeline_projected",
        )
        tensor_value_dim = value_head_dim
    else:
        raise AssertionError(f"unhandled target MLA variant: {variant!r}")

    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    from src.kernel.memory_path_solver import logical_interface_source_traffic
    program=logical_interface_memory_access_program(kernel,hardware)
    source_l2,source_dram,source_write=logical_interface_source_traffic(program,expected_kernel=kernel,hardware=hardware)
    read_l2_bytes,read_bytes,write_bytes=map(float,(source_l2,source_dram,source_write))

    tensor_flops = float(
        2 * query_heads * attention_elements
        * (qk_head_dim + tensor_value_dim))
    softmax_elements = float(query_heads * attention_elements)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_reduction": 2.0 * softmax_elements,
        "compute_fma": 0.0,
        "compute_alu": 3.0 * softmax_elements,
        "compute_sfu": softmax_elements,
        "warp_issue": 0.0,
    }
    l2_total = read_l2_bytes + write_bytes
    dram_bytes = read_bytes + write_bytes
    demand = build_forward_demand(l2_total, dram_bytes, False, flops)
    validated_l2_miss_rate(dram_bytes, l2_total)
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
            *limitations,
            "production_duration_calibration_not_used",
        ),
    )
    normalized, transaction_bytes = _architecture_input(
        inp, kernel, hardware)
    producers, requests_per_producer = gpu_memory_source(hardware)
    structured = attach_structured_contract(
        normalized,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name="global_memory",
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
    )
    structured = split_global_memory_paths(
        structured,
        read_l2_bytes=read_l2_bytes,
        read_dram_bytes=read_bytes,
        write_l2_bytes=write_bytes,
        write_dram_bytes=write_bytes,
    )
    return structured, dram_bytes


def _model_uncached(
        kernel: KernelSpec, hardware, *, attach_pipeline: bool = True):
    _require_modeled_fa3_observation(kernel)
    if is_fa3_launch(kernel):
        del attach_pipeline
        return _fa3_resource_input(kernel, hardware)
    kernel = _metax_full_graph_execution_kernel(kernel)
    kernel = _fa2_full_graph_combine_execution_kernel(kernel)
    kernel = _cann_fia_appended_slot_execution_kernel(kernel)
    variant = kernel.variant.lower()
    if variant in ("mla_prefill", "mla_decode"):
        del attach_pipeline
        return _target_mla_model(kernel, hardware)
    memory_model = hardware.kernel_memory_model
    if memory_model in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            "fused FLASH_ATTN is not modeled for PIM-bank or "
            "systolic-scratchpad execution; use EAGER_ATTN for the "
            "decomposed operator")
    if variant in ("prefill", "decode"):
        is_combine = False
    elif variant in ("combine", "split_combine"):
        is_combine = True
    else:
        raise ValueError(f"unknown FLASH_ATTN variant: {kernel.variant!r}")
    batch, q_len, kv_len, head_dim, query_heads, kv_heads = _shape(kernel)
    splits = _splits(kernel)
    pipeline_geometry = None
    worker_program_evidence = False
    complete_direct_cta_program = False
    split_fuse_work = (
        _cann_split_fuse_work(kernel)
        if memory_model == "npu_tile" and not is_combine else None)
    if not is_combine:
        pipeline_geometry = _pipeline_geometry(kernel)
        if pipeline_geometry is not None:
            worker_program_evidence = _has_worker_program_evidence(kernel)
            complete_direct_cta_program = (
                _has_complete_direct_cta_program(kernel))
            if not _has_semantic_tile_evidence(kernel):
                raise ValueError(
                    "FLASH_ATTN evaluation-only launch classifier cannot "
                    "publish semantic tile geometry to production Forward")
            tile_q, tile_kv, _, grouped_mla = pipeline_geometry
            # Validate semantic work against the launch/queue contract before
            # demand construction.  This does not assign work to physical
            # workers and therefore does not create a pipeline plan.
            _validate_pipeline_class_work(
                kernel, tile_q, tile_kv, grouped_mla)
    if is_combine:
        l2_total, dram_bytes = _combine_traffic(kernel, hardware)
        flops = _combine_flops(kernel, hardware)
    else:
        l2_total, dram_bytes = _tensor_traffic(
            kernel, hardware, split_fuse_work=split_fuse_work)
        flops = _tensor_flops(
            kernel, hardware, split_fuse_work=split_fuse_work)
        source_schedule = _cann_source_schedule_parameters(kernel)
        if source_schedule is not None:
            if not math.isclose(
                    flops["compute_tensor"],
                    source_schedule[
                        "cann_source_tensor_logical_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "CANN source Tensor work disagrees with semantic work")
            flops["compute_tensor"] = source_schedule[
                "cann_source_tensor_logical_operations"]
        if memory_model == "gpu_mshr":
            if _is_source_bound_fa2_pipeline(kernel):
                fa2_fma, fa2_sfu = (
                    _gpu_fa2_scalar_operations(kernel))
                flops["compute_fma"] = fa2_fma
                flops["compute_sfu"] = fa2_sfu
            else:
                flops["compute_sfu"] = _gpu_softmax_exp2_operations(kernel)
            metax_issue = _metax_372_fa_shared_issue_demand(
                kernel, hardware, float(flops["compute_tensor"]))
            if metax_issue > 0:
                if flops["warp_issue"] > 0:
                    raise ValueError(
                        "FLASH_ATTN launch matched two warp-issue contracts")
                flops["warp_issue"] = metax_issue
        elif memory_model == "npu_tile":
            # Generic NPU semantics retain the three-operation softmax lower
            # bound.  Source-bound CANN variants replace it only with
            # statically enumerable instruction work; no duration evidence is
            # consumed here.
            flops["compute_vector"] = 3.0 * _softmax_elements(
                kernel, split_fuse_work=split_fuse_work)
            if split_fuse_work is not None:
                if source_schedule is not None:
                    raise ValueError(
                        "CANN SplitFuse cannot also publish a standard source "
                        "schedule")
                flops["compute_vector"] = float(
                    5 * split_fuse_work.attention_elements
                    + 2 * split_fuse_work.final_output_elements
                    + split_fuse_work.empty_dup_operations
                    + 2 * split_fuse_work.rescale_output_elements)
            else:
                cann_vector_lower_bound = (
                    _cann_standard_decode_vector_lower_bound(kernel))
                if cann_vector_lower_bound is not None:
                    flops["compute_vector"] = (
                        cann_vector_lower_bound.vec1_operations
                        + cann_vector_lower_bound.vec2_operations)
                if source_schedule is not None:
                    flops["compute_vector"] = source_schedule[
                        "cann_source_vector_operations"]
        else:
            raise ValueError(
                f"unknown FLASH_ATTN memory model: {memory_model!r}")
    demand = build_forward_demand(l2_total, dram_bytes, is_combine, flops)
    if not is_combine and _is_source_bound_metax_tile16_program(kernel):
        tile = _attention_tile(kernel)
        if tile != (16, 16):
            raise AssertionError("MetaX tile16 source marker lost its tile")
        class_counts, head_groups = _pipeline_cta_class_counts(
            kernel, 16, 16, grouped_mla=False,
            include_empty_splits=True)
        _, _, _, head_dim, query_heads, _ = _shape(kernel)
        if query_heads % head_groups:
            raise ValueError(
                "MetaX tile16 source CTA head groups must divide exactly")
        splits = _splits(kernel)
        if splits <= 1 and any(
                item.step_count == 0 for item, _ in class_counts):
            raise ValueError(
                "MetaX tile16 unsplit launch cannot contain an empty CTA")
        page_table_bytes = float(
            kernel.launch.parameters["page_table_bytes_per_kv_panel"])
        total_page_table_bytes = page_table_bytes * sum(
            item.step_count * count for item, count in class_counts)
        empty_ctas = sum(
            count for item, count in class_counts if item.step_count == 0)
        output_width_bytes = (
            head_dim * int(kernel.shape["dtype_bytes"])
            if splits == 1 else (head_dim + 1) * 4)
        empty_output_bytes = float(
            empty_ctas * (query_heads // head_groups) * output_width_bytes)
        demand["smem"] = 2.0 * (
            l2_total - total_page_table_bytes - empty_output_bytes)
        if demand["smem"] <= 0.0:
            raise ValueError(
                "MetaX tile16 source shared-memory demand must be positive")
    if not is_combine and _is_source_bound_grouped_mla_ldgsts(kernel):
        grouped = _grouped_mla_tile(kernel)
        if grouped is None:
            raise AssertionError("grouped MLA source marker lost its tile")
        head_tile, tile_kv, qk_head_dim, value_head_dim = grouped
        class_counts, _ = _pipeline_cta_class_counts(
            kernel, head_tile, tile_kv, grouped_mla=True,
            include_empty_splits=True)
        # The pinned TTGIR uses one shared K panel and one shared V panel.
        # Each byte is written once by LDGSTS and read once by the dot path;
        # Q and the split output remain register/global traffic.
        demand["smem"] = float(
            2 * sum(
                item.step_count * count
                for item, count in class_counts)
            * tile_kv * (qk_head_dim + value_head_dim)
            * int(kernel.shape["dtype_bytes"]))
    if (not is_combine
            and not _has_graph_padding(kernel)
            and (_has_metax_q64_prefill_pipeline(kernel)
                 or kernel.launch.algorithm in {
                     _METAX_372_Q128_PREFILL_ALGORITHM,
                     _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM,
                 })):
        # Q/K/V and BF16 O traverse shared memory.  The source writes FP32 LSE
        # directly from registers to global memory, so it must not inherit the
        # generic tensor-kernel two-pass SMEM traffic.
        q_lens, _ = _request_lengths(kernel)
        lse_bytes = float(sum(q_lens) * query_heads * 4)
        demand["smem"] -= 2.0 * lse_bytes
        if demand["smem"] <= 0:
            raise ValueError(
                "MetaX source-bound prefill SMEM demand must be positive")
    if has_dense_flash_attention_source(kernel) and kernel.launch.algorithm.startswith('metax_'):
        source = dense_attention_prediction_work(kernel,hardware)
        # The generic tensor demand stages its entire priced write volume.
        # Direct-global LSE must not acquire a fabricated GM->SMEM->GM copy.
        unstaged_write = _output_bytes(kernel,hardware)-source['output']
        if not 0 <= unstaged_write <= source['lse']:
            raise ValueError('C500 dense source store projection exceeds its actual O/LSE payload')
        demand['smem'] -= 2.0*unstaged_write
        if demand['smem'] <= 0:
            raise ValueError('C500 dense source requires positive output-payload staging')
    cann_workspace = (
        None if is_combine else _cann_workspace_lower_bound(
            kernel, split_fuse_work=split_fuse_work))
    npu_dram_write_bytes = None
    if memory_model == "npu_tile":
        output_bytes = _output_bytes(kernel, hardware)
        source_schedule = _cann_source_schedule_parameters(kernel)
        if source_schedule is not None:
            output_init_bytes = source_schedule[
                "cann_source_output_init_bytes"]
            external_write_bytes = source_schedule[
                "cann_source_external_write_bytes"]
            if not math.isclose(
                    external_write_bytes - output_init_bytes,
                    output_bytes, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "CANN source final output disagrees with the ABI")
            workspace_read = sum(source_schedule[name] for name in (
                "cann_source_vec1_read_bytes",
                "cann_source_mm2_score_read_bytes",
                "cann_source_vec2_mm2_read_bytes",
                "cann_source_vec2_state_read_bytes",
                "cann_source_flash_decode_terminal_read_bytes",
            ))
            workspace_aiv_write = sum(source_schedule[name] for name in (
                "cann_source_vec1_write_bytes",
                "cann_source_nupdate_write_bytes",
                "cann_source_vec2_write_bytes",
                "cann_source_flash_decode_terminal_write_bytes",
                "cann_source_output_init_bytes",
            ))
            source_input_l2 = math.fsum(
                _cann_source_task_input_bytes(kernel, source_schedule))
            demand["l2_cache"] = float(
                source_input_l2 + workspace_read)
            compulsory_input = _compulsory_input_dram_bytes(
                kernel, hardware)
            demand["dram"] = float(compulsory_input)
            demand["mte3"] = float(workspace_aiv_write)
            # MTE3 moves every byte, but DRAM sees only the distinct extent:
            # an independent probe shows a rewritten GM line costs one drain
            # however often it is rewritten, on both the MTE3 and the Fixpipe
            # engine (gm_write_back_residency_cann_8_5_1.json).  The launch
            # derives that extent from the source ring geometry.
            #
            # KNOWN GAP: only the AIV side is priced here.  The AIC domain
            # additionally shows a fixed write of about 33 KiB per AIC core
            # that no source construct accounts for -- workspaceParams carries
            # only mm1ResSize, mm2ResSize, fdAccumOutSize and fdLogSumExpSize,
            # and neither InitWorkspace allocates anything else -- so it stays
            # out rather than being attributed from the leftover difference.
            npu_dram_write_bytes = kernel.launch.parameters[
                "cann_source_aiv_unique_write_bytes"]
        else:
            if output_bytes > l2_total:
                raise ValueError(
                    "FLASH_ATTN output bytes exceed L2-path traffic")
            if output_bytes > dram_bytes:
                raise ValueError(
                    "NPU FLASH_ATTN output bytes exceed established GM traffic")
            workspace_read = (
                0.0 if cann_workspace is None else cann_workspace.read_bytes)
            workspace_aiv_write = (
                0.0 if cann_workspace is None
                else cann_workspace.aiv_write_bytes)
            # Fixed CANN source routes Q/K/V and AIC/AIV workspace consumers
            # through MTE2.  Vec1 probabilities and the final BF16 output leave
            # AIV through MTE3; MM1/MM2 FP32 workspace writes leave AIC through
            # Fixpipe.  Both write streams have a source-proven shared-L2 sink,
            # attached below, while Fixpipe service and workspace DRAM residency
            # remain explicitly unavailable.
            demand["l2_cache"] = float(
                l2_total - output_bytes + workspace_read)
            # GlobalTensor address-space semantics do not prove that a temporary
            # workspace reread misses L2.  Preserve only compulsory external
            # input bytes at the physical GM node.
            demand["dram"] = float(dram_bytes - output_bytes)
            demand["mte3"] = float(output_bytes + workspace_aiv_write)
            if split_fuse_work is None:
                npu_dram_write_bytes = output_bytes
            else:
                # SplitFuse prefill: DRAM sees the distinct extent the AIV
                # stores touch, which the source ring geometry fixes.  The
                # AIC Fixpipe workspace (gS, gOTmp) leaves through its own
                # port and is priced on the Fixpipe path below.
                npu_dram_write_bytes = (
                    _cann_split_fuse_aiv_unique_write_bytes(
                        split_fuse_work, head_dim, output_bytes,
                        workspace_aiv_write))
    mma_smem_bytes_per_flop = kernel.launch.parameters[
        "mma_smem_bytes_per_flop"]
    if not is_combine and _is_source_bound_metax_mla_main(kernel):
        demand["smem"] = _metax_mla_main_smem_bytes(kernel)
    elif (not is_combine
          and not _has_graph_padding(kernel)
          and mma_smem_bytes_per_flop > 0):
        if not _is_source_bound_fa2_pipeline(kernel):
            raise ValueError(
                "FLASH_ATTN SMEM staging requires a source-bound FA2 main "
                "kernel")
        # Count the complete source-level shared-memory path, not merely the
        # LDGSTS ingress bytes.  In these pinned traits Q is not retained in
        # registers, so every QK panel rereads the full compile-time Q tile;
        # the output also takes a shared-memory staging round before the
        # predicated global store.  Page-table metadata remains in registers.
        demand["smem"] = _fa2_source_smem_bytes(kernel)
    eta, wave_efficiency, active_units = (
        launch_wave_geometry(
            kernel, hardware)
    )
    if (not is_combine
            and _is_source_bound_fa2_pipeline(kernel)
            and kernel.attributes["kv_layout"] in ("paged", "paged_varlen")
            and q_len == 1 and splits > 1):
        # The independent SM80 admission trace shows consecutive source CTAs
        # filling both resident slots of one SM before moving to the next SM.
        # Both paged layouts dispatch the same pinned split-KV source program;
        # for a sub-216-CTA launch, active execution units are therefore the
        # number of occupied two-slot SMs, not the number of resident CTAs.
        active_units = min(
            hardware.num_compute_units,
            math.ceil(kernel.launch.grid_size
                      / _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS),
        )

    if not is_combine:
        if memory_model == "gpu_mshr":
            if pipeline_geometry is None:
                if kernel.launch.parameters["unpadded_flops"] != 1.0:
                    raise ValueError(
                        "GPU FLASH_ATTN without tile geometry must explicitly "
                        "use unpadded FLOP accounting")
            else:
                # The semantic-work/physical-launch contract was validated
                # before aggregate demand construction.  Worker assignment
                # and buffer scheduling remain separately gated below.
                pass
        elif memory_model == "npu_tile":
            # CANN exposes no stable Q/KV tile geometry in the canonical
            # contract.  Use analytical demand with the generic NPU memory
            # model; production AIC/AIV durations are evaluation evidence only.
            pass
        elif memory_model in ("pim_bank", "systolic_scratchpad"):
            raise AssertionError(
                "cacheless FLASH_ATTN should have been rejected above")
        else:
            raise ValueError(
                f"unknown FLASH_ATTN kernel_memory_model: {memory_model!r}")

    effective_fraction = float(kernel.attributes["fa_eff_mfrac"])
    if effective_fraction != 1.0:
        raise NotImplementedError(
            "FLASH_ATTN fa_eff_mfrac != 1 is unsupported: q_lens/kv_lens "
            "already encode the real per-request work, so applying another "
            "whole-kernel fraction would double-scale traffic and FLOPs")
    scratch_lifecycle_limitation = (
        ("flash_attention_split_scratch_dram_lifecycle_unavailable",)
        if splits > 1 and _split_scratch_dram_width(kernel) == 0 else ())
    split_fuse_limitations = (
        (
            "cann_split_fuse_replay_dram_residency_unavailable",
            "cann_split_fuse_workspace_lifecycle_unavailable",
            "cann_fia_fixpipe_service_evidence_unavailable",
            "cann_fia_workspace_dram_residency_unavailable",
        )
        if (not is_combine and _cann_split_fuse_tile(kernel) is not None)
        else ()
    )
    source_schedule = (
        None if is_combine else _cann_source_schedule_parameters(kernel))
    standard_decode_limitations = (
        (
            "cann_standard_decode_dynamic_partition_unavailable",
            "cann_standard_decode_output_init_partition_unavailable",
            "cann_standard_decode_workspace_padding_unavailable",
            "cann_standard_decode_online_state_workspace_unavailable",
            "cann_standard_decode_vector_padding_and_merge_unavailable",
            "cann_fia_fixpipe_service_evidence_unavailable",
            "cann_fia_workspace_dram_residency_unavailable",
        )
        if (not is_combine
            and kernel.launch.algorithm
            in _CANN_STANDARD_GQA_DECODE_ALGORITHMS
            and source_schedule is None)
        else (
            (
                "cann_fia_fixpipe_service_evidence_unavailable",
                "cann_fia_workspace_dram_residency_unavailable",
            )
            if source_schedule is not None else ()
        )
    )
    cann_mla_limitations = (
        (
            "cann_mla_dynamic_partition_unavailable",
            "cann_mla_workspace_lifecycle_unavailable",
            "cann_mla_nupdate_pipeline_unavailable",
            "cann_fia_fixpipe_service_evidence_unavailable",
            "cann_fia_workspace_dram_residency_unavailable",
        )
        if (not is_combine
            and _cann_mla_dimensions(kernel) is not None
            and source_schedule is None)
        else (
            ("cann_mla_nupdate_pipeline_unavailable",)
            if (not is_combine
                and _cann_mla_dimensions(kernel) is not None)
            else ()
        )
    )
    metax_mla_limitations = (
        (
            "metax_mla_3_2_1_device_source_identity_unavailable",
            "metax_mla_partial_scratch_layout_unavailable",
            "metax_mla_worker_and_pipeline_unavailable",
        )
        if _is_unbound_metax_mla_scratch(kernel)
        else ()
    )
    aiv_instruction_limitations = (
        ("aiv_generic_reduction_instruction_schedule_unavailable",)
        if source_schedule is not None else
        ("aiv_softmax_static_instruction_schedule_unavailable",)
    )
    if memory_model == "npu_tile":
        launch_task_kind = "aiv" if is_combine else "mixed"
        include_launch_floor, launch_limitations = (
            npu_launch_floor_contract(hardware, launch_task_kind))
        support_level = "aggregate_only"
        flash_decode_limitations = (
            (
                "cann_flash_decode_vector_demand_unavailable",
                "cann_flash_decode_partial_workspace_unavailable",
            )
            if (not is_combine and kernel.launch.algorithm
                == "cann_flash_attention_score:103000000010522221"
                and source_schedule is None)
            else ()
        )
        limitations = (
            scratch_lifecycle_limitation
            + split_fuse_limitations
            + standard_decode_limitations
            + cann_mla_limitations
            + metax_mla_limitations
            + (
                "cann_flash_attention_tile_pipeline_unavailable",
                "mte_transaction_geometry_unavailable",
                "memory_first_request_latency_unavailable",
                "mte2_descriptor_and_credit_evidence_unavailable",
                "workload_producer_independence_unavailable",
            )
            + aiv_instruction_limitations
            + flash_decode_limitations
            + launch_limitations
        )
    elif memory_model == "gpu_mshr":
        include_launch_floor = True
        if is_combine:
            combine_worker_evidence = (
                _has_complete_gpu_combine_program(kernel)
                and (
                    not _is_source_bound_metax_fa2_combine(kernel)
                    or hardware.metax_fa2_combine_dependency_cycles
                    is not None))
            if _is_source_bound_metax_mla_combine(kernel):
                support_level = "partial"
                limitations = gpu_partial_limitations(
                    hardware, *scratch_lifecycle_limitation)
            elif combine_worker_evidence:
                support_level = "partial"
                limitations = gpu_partial_limitations(
                    hardware,
                    *scratch_lifecycle_limitation,
                    *metax_mla_limitations,
                    *((
                        "gpu_flash_attention_round_robin_cta_assignment_assumed",
                    ) if not _has_uniform_fa2_combine_ctas(kernel) else ()),
                )
            else:
                support_level = "aggregate_only"
                limitations = gpu_partial_limitations(
                    hardware,
                    *scratch_lifecycle_limitation,
                    *metax_mla_limitations,
                    (
                        "gpu_flash_attention_cta_worker_mapping_unavailable"
                        if _is_source_bound_fa2_combine(kernel)
                        else
                        "gpu_flash_attention_combine_pipeline_evidence_unavailable"
                    ),
                )
        elif _is_source_bound_metax_mla_main(kernel):
            support_level = "partial"
            limitations = gpu_partial_limitations(
                hardware, *scratch_lifecycle_limitation)
        elif pipeline_geometry is None:
            support_level = "aggregate_only"
            limitations = gpu_partial_limitations(
                hardware,
                *scratch_lifecycle_limitation,
                *metax_mla_limitations,
                "gpu_flash_attention_tile_geometry_unavailable",
            )
        elif (not complete_direct_cta_program
              or not (
                  _is_source_bound_fa2_pipeline(kernel)
                  or _is_source_bound_grouped_mla_ldgsts(kernel)
                  or _is_source_bound_metax_tile16_program(kernel))):
            support_level = "aggregate_only"
            limitations = gpu_partial_limitations(
                hardware,
                *scratch_lifecycle_limitation,
                *metax_mla_limitations,
                "gpu_flash_attention_cta_worker_mapping_unavailable",
            )
        else:
            support_level = "partial"
            paired_fifo_admission = False
            bounded_fifo_drain = False
            split3_varlen_admission = False
            if pipeline_geometry is not None:
                tile_q, tile_kv, _, grouped_mla = pipeline_geometry
                paired_fifo_admission = (
                    not grouped_mla
                    and _is_source_bound_fa2_pipeline(kernel)
                    and _has_a800_fa2_paired_fifo_admission(
                        kernel, hardware, tile_q, tile_kv))
                bounded_fifo_drain = (
                    not grouped_mla
                    and _is_source_bound_fa2_pipeline(kernel)
                    and _has_a800_fa2_dense_bounded_fifo_drain(
                        kernel, hardware, tile_q, tile_kv))
                split3_varlen_admission = (
                    not grouped_mla
                    and _is_source_bound_fa2_pipeline(kernel)
                    and _has_a800_fa2_split3_varlen_admission(
                        kernel, hardware, tile_q, tile_kv))
            queued_varlen_decode = (
                _is_source_bound_fa2_pipeline(kernel)
                and kernel.attributes["kv_layout"] == "paged_varlen"
                and q_len == 1
                and kernel.launch.grid_size > (
                    hardware.num_compute_units
                    * _SM80_FA2_MAIN_RESIDENT_CTA_SLOTS)
                and len(set(_request_lengths(kernel)[1])) > 1)
            if (paired_fifo_admission or bounded_fifo_drain
                    or split3_varlen_admission):
                assignment_limitations = ()
            elif queued_varlen_decode:
                assignment_limitations = (
                    "gpu_flash_attention_work_conserving_queue_tail_balance",
                )
            elif kernel.attributes["kv_layout"] == "chunked_prefill":
                assignment_limitations = (
                    "gpu_flash_attention_chunked_class_balance_assumed",
                )
            elif (not worker_program_evidence
                  and not (
                      kernel.attributes["kv_layout"] == "paged_varlen"
                      and q_len == 1 and splits > 1)):
                assignment_limitations = (
                    "gpu_flash_attention_round_robin_cta_assignment_assumed",
                )
            else:
                assignment_limitations = ()
            limitations = gpu_partial_limitations(
                hardware,
                *scratch_lifecycle_limitation,
                *metax_mla_limitations,
                *assignment_limitations,
            )
        launch_task_kind = "generic"
    else:
        raise ValueError(
            f"unknown FLASH_ATTN memory model: {memory_model!r}")
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
        dram_capacity_multiplier=(
            hardware.dram_protocol_efficiency if not is_combine else 1.0),
        support_level=support_level,
        limitations=limitations,
    )
    if _is_source_bound_metax_mla_main(kernel):
        compulsory = float(current_metax_mla_source_work(kernel)['compulsory_input_bytes'])
    else:
        compulsory = _compulsory_input_bytes(kernel, hardware)
    normalized, transaction_bytes = _architecture_input(inp, kernel, hardware)
    if memory_model == "gpu_mshr":
        producers, requests_per_producer = gpu_memory_source(hardware)
        path_name = "global_memory"
    elif memory_model == "npu_tile":
        producers, requests_per_producer = 1.0, None
        path_name = "mte2_read"
    else:
        raise AssertionError(
            f"unhandled FLASH_ATTN memory model: {memory_model!r}")
    structured = attach_structured_contract(
        normalized,
        hardware,
        transaction_bytes=transaction_bytes,
        path_name=path_name,
        source_producers_per_unit=producers,
        independent_requests_per_producer=requests_per_producer,
        frontend_rate_per_unit_s=None,
    )
    if memory_model == "gpu_mshr":
        output_bytes = _output_bytes(kernel, hardware)
        if output_bytes > l2_total:
            raise ValueError("FLASH_ATTN output bytes exceed L2 traffic")
        if is_combine:
            write_dram = min(output_bytes, dram_bytes)
        else:
            write_dram = min(
                output_bytes, max(0.0, dram_bytes - compulsory))
        structured = split_global_memory_paths(
            structured,
            read_l2_bytes=l2_total - output_bytes,
            read_dram_bytes=dram_bytes - write_dram,
            write_l2_bytes=output_bytes,
            write_dram_bytes=write_dram,
        )
        if not is_combine:
            structured = _attach_gpu_ldgsts_frontend(
                structured, kernel, hardware)
            structured = _attach_a800_fa2_paged_period32_kv_service(
                structured, kernel, hardware)
            structured = _attach_a800_fa2_chunked_q8192_s4096_kv_service(
                structured, kernel, hardware)
        if (not is_combine and attach_pipeline
                and _has_metax_q64_prefill_pipeline(kernel)):
            structured = _attach_metax_q64_prefill_pipeline(
                structured, kernel, hardware)
        elif (not is_combine and attach_pipeline
                and (_has_metax_q64_contiguous_decode_pipeline(kernel)
                     or _has_metax_q128_contiguous_decode_pipeline(kernel))):
            structured = _attach_metax_contiguous_decode_pipeline(
                structured, kernel, hardware)
        elif (not is_combine and attach_pipeline
                and _is_source_bound_metax_tile16_program(kernel)):
            structured = _attach_metax_tile16_decode_pipeline(
                structured, kernel, hardware)
        elif (not is_combine and attach_pipeline
                and _is_source_bound_metax_mla_main(kernel)):
            structured = _attach_metax_mla_main_pipeline(
                structured, kernel, hardware)
        elif (not is_combine and attach_pipeline
                and _has_metax_q128_prefill_pipeline(kernel)):
            structured = _attach_metax_q128_prefill_pipeline(
                structured, kernel, hardware)
        elif (is_combine and attach_pipeline
                and _is_source_bound_metax_mla_combine(kernel)):
            structured = _attach_metax_mla_combine_pipeline(
                structured, kernel, hardware)
        elif (is_combine and attach_pipeline
                and _is_source_bound_metax_fa2_combine(kernel)):
            if hardware.metax_fa2_combine_dependency_cycles is not None:
                structured = _attach_metax_fa2_combine_pipeline(
                    structured, kernel, hardware)
        elif (is_combine and attach_pipeline
                and _has_complete_gpu_combine_program(kernel)):
            if _is_source_bound_grouped_mla_combine(kernel):
                structured = _attach_gpu_grouped_mla_combine_pipeline(
                    structured, kernel, hardware)
            else:
                structured = _attach_gpu_fa2_combine_pipeline(
                    structured, kernel, hardware)
        elif (not is_combine and pipeline_geometry is not None
                and complete_direct_cta_program and attach_pipeline
                and (
                    _is_source_bound_fa2_pipeline(kernel)
                    or _is_source_bound_grouped_mla_ldgsts(kernel))):
            tile_q, tile_kv, _, grouped_mla = pipeline_geometry
            structured = _attach_gpu_attention_pipeline(
                structured,
                kernel,
                hardware,
                grouped_mla=grouped_mla,
                tile_q=tile_q,
                tile_kv=tile_kv,
            )
    elif memory_model == "npu_tile":
        structured = _apply_cann_standard_decode_tensor_padding(
            structured, kernel)
        structured = _attach_cann_split_fuse_tensor_schedule(
            structured, kernel, split_fuse_work=split_fuse_work)
        structured = _attach_cann_split_fuse_vector_schedule(
            structured, kernel, split_fuse_work=split_fuse_work)
        cann_l2_write_bytes = (
            structured.demand_per_resource["mte3"]
            if cann_workspace is not None else None)
        structured = append_npu_write_path(
            structured,
            hardware,
            dram_write_bytes=npu_dram_write_bytes,
            l2_write_bytes=cann_l2_write_bytes,
        )
        structured = _attach_cann_fixpipe_l2_write(
            structured, kernel, split_fuse_work=split_fuse_work)
        structured = _attach_cann_kfc_workspace_clear(structured, kernel)
        structured = _attach_cann_standard_decode_phase_ledger(
            structured, kernel, hardware)
        structured = _attach_cann_split_fuse_pipeline(
            structured, kernel, hardware,
            attach_pipeline=attach_pipeline,
            split_fuse_work=split_fuse_work,
        )
        structured = _attach_cann_fia_cross_domain_plan(
            structured, kernel, hardware,
            attach_pipeline=attach_pipeline,
        )
    else:
        raise AssertionError(
            f"unhandled FLASH_ATTN memory model: {memory_model!r}")
    if (attach_pipeline
            and _has_a800_fa2_source_task_outer_envelope(kernel, hardware)):
        if structured.pipeline_plan is None:
            raise ValueError(
                "FA2 source task outer envelope requires a closed pipeline")
        structured = dataclasses.replace(
            structured,
            task_duration_floor_contract=GPU_SOURCE_PIPELINE_OUTER_ENVELOPE,
        )
    return structured, compulsory


# ---------------------------------------------------------------------------
# wp10 (H200 FA3 MLA): FlashAttention-3 Midlayer template.
#
# Source: vllm-flash-attention 803020a8fa15407871341d41eba4919ade2ee1ee
# (hopper/), the revision vLLM 0.24.0 pins.  Dispatched by the launch
# algorithm names of the H200 launch package (flash_attn_launch.py).
# Execution structure (FlashAttnFwdSm90, CollectiveMainloopFwdSm90):
#   * one producer warpgroup loads Q once per tile and K/V per KV block into
#     a two-stage ring; two MMA warpgroups consume it; the varlen dynamic
#     persistent scheduler hands tiles (m_block, head, split) to 132
#     persistent CTAs in source order (the first tile per CTA by blockIdx,
#     every later one to the first CTA that asks);
#   * prefill (hdim 192 / 128, kBlockM = kBlockN = 128, IntraWGOverlap,
#     no inter-warpgroup scheduler barrier because kHeadDim > 128): both MMA
#     warpgroups share the K/V two-slot ring. Per KV block each issues QK
#     (m64n128k16 SS x 12), rescales old O, then issues the previous block's
#     PV (m64n128k16 RS x 8). QK wait1 releases K; softmax and PV wait0
#     precede P reuse. Separate QK192 and V/O128 widths retain conversion,
#     shared output staging and actual request/split tails in the recurrence;
#   * decode (hdim 64 / 512, kBlockM = kBlockN = 64, LargeHeadDimV, no
#     intra-WG overlap, paged cp.async producer): the softmax warpgroup runs
#     QK+QV (m64n64k16 SS x 36) alone, one softmax step alone, then both
#     warpgroups run their m64n256k16 PV (x 4) concurrently; that consumer
#     occupancy was used as the hold of the measured two-stage paged ring,
#     which then gives the per-block initiation interval and first-block
#     latency by exact lookup at the active-CTA count;
#   * combine: FlashAttnFwdCombine (8 rows x 128 columns per CTA, four
#     cp.async stages over the splits). Normalization/shared weights and
#     weighted O share the GQA reducer services; global work/bytes compete
#     with the existing resident-slot local-chain floor.
# Every hardware number is an H200 mechanism measurement
# (scheduling.fa3_mla_mechanisms, memory_subsystem latencies, the L2 read
# capacity, CTA dispatch cost and span floor); no FlashAttention duration.
# ---------------------------------------------------------------------------
_FA3_PREFILL_ALGORITHM = "vllm_fa3_sm90_mla_prefill"
_FA3_PREFILL_SPLIT_ALGORITHM = "vllm_fa3_sm90_mla_prefill_split_partial"
_FA3_PREFILL_COMBINE_ALGORITHM = "vllm_fa3_sm90_mla_prefill_combine_partial"
_FA3_DECODE_ALGORITHM = "vllm_fa3_sm90_mla_decode"
_FA3_COMBINE_ALGORITHM = "vllm_fa3_sm90_combine"
_FA3_GQA_PREFILL_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_prefill_partial"
_FA3_GQA_OBSERVED_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_observed_partial"
_FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_observed_split_partial"
_FA3_GQA_COMBINE_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_combine_partial"
_FA3_GQA_TMA_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_tma_partial"
_FA3_GQA_TMA_SPLIT_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_tma_split_partial"
_FA3_GQA_TMA_COMBINE_030_ALGORITHM = "vllm_030_fa3_sm90_gqa_tma_combine_partial"
_FA3_SPARSE_MLA_030_ALGORITHM = "vllm_030_fa3_sm90_sparse_mla_partial"
_FA3_SPARSE_MLA_COMBINE_030_ALGORITHM = "vllm_030_fa3_sm90_sparse_mla_combine_partial"
_FLASHINFER_MLA_PREFILL_ALGORITHM = "flashinfer_mla_hopper_sm90_paged"
_FLASHINFER_MLA_THREADS = 256
_FA3_ALGORITHMS = frozenset((
    _FA3_PREFILL_ALGORITHM, _FA3_DECODE_ALGORITHM, _FA3_COMBINE_ALGORITHM,
    _FLASHINFER_MLA_PREFILL_ALGORITHM, _FA3_GQA_PREFILL_030_ALGORITHM,
    _FA3_GQA_OBSERVED_030_ALGORITHM, _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM,
    _FA3_GQA_COMBINE_030_ALGORITHM, _FA3_GQA_TMA_030_ALGORITHM,
    _FA3_GQA_TMA_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_COMBINE_030_ALGORITHM,
    _FA3_PREFILL_SPLIT_ALGORITHM, _FA3_PREFILL_COMBINE_ALGORITHM,
    _FA3_SPARSE_MLA_030_ALGORITHM, _FA3_SPARSE_MLA_COMBINE_030_ALGORITHM))
_FA3_ELEMENT_BYTES = 2
# The absorbed MLA decode tile (tile_size_fwd_sm90(64, 512)); the combine
# follows only that kernel, so its split derivation reuses this kBlockN.
_FA3_DECODE_BLOCK_N = 64
_FA3_COMBINE_STAGES = 4
_FA3_COMBINE_THREADS = 256
# FlashAttnFwdCombine registers per thread: a compiler fact recorded by the
# launch-geometry observation (vllm_0_24_0_dsv3_l32_tp4_launch_geometry_sm90
# .json, Reg/Trd 46); the source does not fix it. The canonical H200 FA3
# GQA observations independently record the same count for the four BF16
# Varlen templates (rows,columns,compiled splits) bound below. No other
# GQA compiler/template domain borrows the legacy resource fact.
_FA3_COMBINE_REGISTERS_PER_THREAD = 46
_FA3_PREPARE_MAX_BATCHES_1CTA = 992


# An observed H200 FlashAttention-3 launch outside the derived MLA decode and
# uniform fresh GQA prefill templates.  The FA2 (SM80) and MetaX templates
# describe different kernels, so no method prices it.
FA3_UNMODELED_OBSERVATION_ALGORITHM = "observed_vllm_fa3_sm90_unmodeled"


def is_fa3_launch(kernel: KernelSpec) -> bool:
    return kernel.launch.algorithm in _FA3_ALGORITHMS


def _require_modeled_fa3_observation(kernel: KernelSpec) -> None:
    if kernel.launch.algorithm == FA3_UNMODELED_OBSERVATION_ALGORITHM:
        raise NotImplementedError(
            "H200 FlashAttention-3 has derived templates for absorbed MLA "
            "decode and uniform fresh BF16 D128 GQA prefill; this "
            f"{kernel.variant} launch ({kernel.attributes['kv_layout']}) is "
            "unsupported")


def _fa3_mechanisms(hardware):
    mechanisms = hardware.fa3_mla_mechanisms
    if mechanisms is None:
        raise NotImplementedError(
            "FA3 MLA pricing requires scheduling.fa3_mla_mechanisms")
    return mechanisms


def _fa3_lengths(kernel: KernelSpec) -> tuple[tuple[int, ...], tuple[int, ...]]:
    full = ("source_program" in kernel.attributes
        and kernel.attributes["source_program"] == "vllm_0_30_fa3_gqa_full_decode")
    q_lens = tuple(int(value) for value in kernel.attributes["model_input_q_lens" if full else "q_lens"])
    kv_lens = tuple(int(value) for value in kernel.attributes["model_input_kv_lens" if full else "kv_lens"])
    if len(q_lens) != len(kv_lens) or not q_lens:
        raise ValueError("FA3 request lengths must pair one-to-one")
    return q_lens, kv_lens


def _fa3_decode_splits(q_lens, kv_lens, query_heads: int,
                       static_splits: int, num_sms: int) -> tuple[int, ...]:
    """prepare_varlen_num_blocks_kernel dynamic splits (PackGQA, one KV head)."""
    return _fa3_request_tile_splits(
        q_lens, kv_lens, static_splits, num_sms, block_m=64,
        block_n=_FA3_DECODE_BLOCK_N, ratio=query_heads, kv_heads=1)


def _fa3_request_tile_splits(q_lens, kv_lens, static_splits, num_sms, *,
                             block_m, block_n, ratio, kv_heads):
    """Pinned prepare's FP32 task count, after static source launch selection."""
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    if static_splits == 1 or len(q_lens) > _FA3_PREPARE_MAX_BATCHES_1CTA:
        return tuple(1 for _ in q_lens)
    m_blocks = [math.ceil(q * ratio / block_m) for q in q_lens]
    n_blocks = [math.ceil(k / block_n) for k in kv_lens]
    total = sum(m * n for m, n in zip(m_blocks, n_blocks))
    per_sm = math.ceil(f32(f32(f32(f32(float(total)) * f32(1.1))
                              * f32(float(kv_heads))) / f32(float(num_sms))))
    return tuple(max(min(math.ceil(n / per_sm), static_splits), 1) for n in n_blocks)


def fa3_decode_dynamic_splits(q_lens, kv_lens, query_heads: int,
                              static_splits: int,
                              num_sms: int) -> tuple[int, ...]:
    """Per-request dynamic splits the FA3 MLA decode template schedules."""

    return _fa3_decode_splits(
        tuple(q_lens), tuple(kv_lens), query_heads, static_splits, num_sms)


def _fa3_prefill_tiles(kernel: KernelSpec) -> list[int]:
    """KV blocks per tile, in VarlenDynamicPersistentTileScheduler order."""

    q_lens, kv_lens = _fa3_lengths(kernel)
    heads = int(kernel.shape["num_query_heads"])
    tile = kernel.launch.tiling.tile_shape
    block_m, block_n = tile["q"], tile["kv"]
    head_dim, head_dim_v = tile["qk_head"], tile["value_head"]
    # prepare_varlen: size_l2 = 32 MiB / 1 (qhead_per_khead == 1).
    max_kv_blocks_in_l2 = (32 * 1024 * 1024) // (
        block_n * (head_dim + head_dim_v) * _FA3_ELEMENT_BYTES)
    tiles = []
    for q_len, kv_len in zip(q_lens, kv_lens):
        num_m = math.ceil(q_len / block_m)
        num_n = math.ceil(kv_len / block_n)
        if not kernel.attributes["causal"]:
            # Noncausal context has no triangular mask or LPT head swizzle.
            tiles.extend([num_n] * (num_m * heads))
            continue
        blocks = []
        for m_block in range(num_m):
            causal_max = math.ceil(
                ((m_block + 1) * block_m + kv_len - q_len) / block_n)
            blocks.append(min(num_n, causal_max))
        n_for_l2 = max(num_n, 1)
        heads_in_l2 = 1
        for candidate in (16, 8, 4, 2):
            if n_for_l2 * candidate <= max_kv_blocks_in_l2:
                heads_in_l2 = candidate
                break
        heads_in_l2 = min(heads_in_l2, heads)
        section_start = 0
        while section_start < heads:
            section_heads = min(heads_in_l2, heads - section_start)
            for m_block in range(num_m - 1, -1, -1):
                tiles.extend([blocks[m_block]] * section_heads)
            section_start += section_heads
    return tiles


def _fa3_gqa_prefill_030_tile_blocks(kernel: KernelSpec) -> tuple[int, ...]:
    """Causal KV blocks for each packed-Q tile in the v0.30 varlen scheduler."""

    q_lens, kv_lens = _fa3_lengths(kernel)
    if (len(q_lens) != kernel.shape["batch"] or q_lens != kv_lens
            or any(length != kernel.shape["q_len"] for length in q_lens)):
        raise ValueError("vLLM 0.30 GQA prefill requires uniform fresh requests")
    ratio = kernel.shape["num_query_heads"] // kernel.shape["num_kv_heads"]
    block_m = kernel.launch.tiling.tile_shape["q"]
    block_n = kernel.launch.tiling.tile_shape["kv"]
    q_len = q_lens[0]
    per_head = tuple(
        ((min(q_len * ratio, (m_block + 1) * block_m) - 1)
         // ratio // block_n + 1)
        for m_block in range(math.ceil(q_len * ratio / block_m)))
    # flash_prepare_scheduler.cu reduces its 32-MiB head grouping budget by
    # the GQA ratio and chooses 16/8/4/2/1 heads per L2 section.  The causal
    # LPT scheduler walks M blocks in reverse within each section and cycles
    # through that section's KV heads at every M block.
    divisor = next((candidate for candidate in (1, 2, 4, 8, 16)
                    if ratio <= candidate), 16)
    max_kv_blocks_in_l2 = ((32 * 1024 * 1024) // divisor
                           // (block_n * (kernel.shape["head_dim"] * 2) * 2))
    n_blocks = math.ceil(q_len / block_n)
    heads_in_l2 = next((candidate for candidate in (16, 8, 4, 2)
                        if n_blocks * candidate <= max_kv_blocks_in_l2), 1)
    kv_heads = kernel.shape["num_kv_heads"]
    heads_in_l2 = min(heads_in_l2, kv_heads)
    per_request = tuple(
        blocks
        for section in range(0, kv_heads, heads_in_l2)
        for blocks in reversed(per_head)
        for _ in range(min(heads_in_l2, kv_heads - section)))
    return per_request * len(q_lens)


def _fa3_gqa_prefill_030_source_work(kernel: KernelSpec, hardware):
    """Source-bound GQA tile work and public Q/K/V extents for both APIs."""

    hardware.require_v030_fa3_gqa_prefill_source_launch(kernel)
    selected = kernel.launch
    from src.kernel.memory_access_builder import (
        logical_interface_memory_access_program)
    from src.kernel.memory_path_solver import public_attention_source_work
    public_work = public_attention_source_work(
        logical_interface_memory_access_program(kernel, hardware),
        expected_kernel=kernel, hardware=hardware)
    q_len = kernel.shape["q_len"]
    total_q = kernel.shape["batch"] * q_len
    if (public_work["query_bytes"] != total_q * kernel.shape["num_query_heads"] * 128 * 2
            or public_work["key_bytes"] != total_q * kernel.shape["num_kv_heads"] * 128 * 2
            or public_work["value_bytes"] != public_work["key_bytes"]):
        raise ValueError("vLLM 0.30 GQA public Q/K/V extents disagree with source")
    blocks = _fa3_gqa_prefill_030_tile_blocks(kernel)
    tile = selected.tiling.tile_shape
    tensor_flops = (4 * tile["q"] * tile["kv"] * tile["qk_head"]
                    * sum(blocks))
    logical_l2_read_bytes = (
        sum(blocks) * tile["kv"] * (tile["qk_head"] + tile["value_head"]) * 2
        + len(blocks) * tile["q"] * tile["qk_head"] * 2)
    return {
        "selected": selected, "public_work": public_work,
        "blocks": blocks, "tensor_flops": tensor_flops,
        "logical_l2_read_bytes": logical_l2_read_bytes,
    }


def _fa3_gqa_prefill_030_partial_prediction(kernel: KernelSpec, hardware):
    """Retain the public source binding and price the missing source stages."""

    _fa3_gqa_prefill_030_source_work(kernel, hardware)
    tile = kernel.launch.tiling
    execution = dataclasses.replace(
        kernel, launch=dataclasses.replace(
            kernel.launch, algorithm=_FA3_GQA_OBSERVED_030_ALGORITHM,
            tiling=dataclasses.replace(tile, tile_shape={
                "q": tile.tile_shape["q"], "kv": tile.tile_shape["kv"],
                "head_dim": kernel.shape["head_dim"]})))
    # The retained CUDA12.9 GQA4 prefill binary places wait0 before all EX2
    # and local-sum work in its steady loop (native PCs c1f0/c320). Only
    # the older explicit source-bound prefill family consumes this evidence;
    # normal source factories still need their actual compiler/caller binding.
    compiled_wait0 = (kernel.shape["num_query_heads"]
                      == 4 * kernel.shape["num_kv_heads"])
    prediction = _fa3_gqa_observed_prediction(
        execution, hardware, compiled_wait0_before_online=compiled_wait0)
    return dataclasses.replace(
        prediction, diagnostics=dataclasses.replace(
            prediction.diagnostics,
            limitations=prediction.diagnostics.limitations + (
                "vllm_030_fa3_gqa_one_int32_semaphore_fill_priced_separately",)))


def _fa3_gqa_partial_source_work(q_lens, kv_lens, *, heads, kv_heads,
                                  block_m, block_n, splits, causal):
    """Structural split-output work, including empty and masked ranges.

    A nonempty main M tile stores O for all its valid rows, including
    rows whose own causal prefix is empty in that split. An empty tile
    stores only -inf LSE. Combine prefetches its first three O stages;
    later loads stop at each row's last finite-LSE split. The owned
    combine copies assign one row per thread (M16/K64 or M8/K128).
    BlockMN clips each M tile's causal N extent before dividing it among
    splits. Consequently the split width can differ between M tiles of
    the same request. Counts assume finite unmasked attention operands;
    they do not bind
    physical scratch aliases/readiness or override native validity guards.
    """
    ratio = heads // kv_heads
    lse_rows, stored_o_rows, weighted_o_rows, loaded_o_rows = 0, 0, 0, 0
    max_valid_splits = 0

    for q, kv, count in zip(q_lens, kv_lens, splits):
        if count == 1:
            continue
        lse_rows += q * heads * count
        if causal:
            # Count packed-row intervals instead of individual output
            # elements or keys. A split is finite once its first key is
            # visible to that row; the first three O stages are prefetched
            # even when the row's corresponding LSE is negative infinity.
            prefetched = min(3, count)
            total_n = math.ceil(kv / block_n)
            for start in range(0, q * ratio, block_m):
                end = min(start + block_m, q * ratio)
                last_q = (end - 1) // ratio
                tile_n = min(total_n, max(0, math.ceil(
                    (kv - q + last_q + 1) / block_n)))
                panels_per_split = math.ceil(tile_n / count)
                live = math.ceil(tile_n / panels_per_split) if tile_n else 0
                max_valid_splits = max(max_valid_splits, live)
                stored_o_rows += (end - start) * kv_heads * live
                loaded_o_rows += (end - start) * kv_heads * prefetched
                for split_index in range(live):
                    first_key = split_index * panels_per_split * block_n
                    first_finite_packed_row = (first_key - kv + q) * ratio
                    finite = max(0, end - max(start, first_finite_packed_row))
                    weighted_o_rows += finite * kv_heads
                    if split_index >= prefetched:
                        loaded_o_rows += finite * kv_heads
        else:
            panels_per_split = math.ceil(math.ceil(kv / block_n) / count)
            live = math.ceil(kv / (panels_per_split * block_n)) if kv else 0
            max_valid_splits = max(max_valid_splits, live)
            stored_o_rows += q * heads * live
            weighted_o_rows += q * heads * live
            loaded_o_rows += q * heads * max(live, min(3, count))
    return {"partial_lse_rows": lse_rows,
            "materialized_partial_output_rows": stored_o_rows,
            "weighted_partial_output_rows": weighted_o_rows,
            "combine_partial_output_read_rows": loaded_o_rows,
            "max_valid_splits": max_valid_splits}


class _Fa3GqaSourceTask(NamedTuple):
    """One source task, without a detailed memory/control graph or a price.

    The two first-panel fields are absent for an empty split. Packed query
    tails remain live extents; no query or KV row is fabricated to fill a tile.
    """

    batch_index: int
    kv_head_index: int
    q_block_index: int
    split_index: int
    packed_q_start: int
    packed_q_rows: int
    kv_block_start: int
    kv_block_stop: int
    read_kv_rows: int
    first_panel_kv_rows: Optional[int]
    first_causal_all_masked_rows: Optional[int]

    @property
    def panel_count(self):
        return max(self.kv_block_stop - self.kv_block_start, 0)


def _fa3_gqa_source_tasks(q_lens, kv_lens, *, heads, kv_heads, block_m,
                          block_n, head_dim, splits, causal, value_head_dim=None):
    """Stream compact geometry in the retained model's head-section/LPT order.

    Masked-row counts describe mathematical eligibility in the first descending
    panel, not the compiled Check_inf flag. That flag still belongs to the
    hardware source program. Caller batch order is preserved; this is not the
    original GPU scheduler's global metadata sort or CTA ticket assignment.
    No observation duration or service is consumed.
    """
    value_head_dim = head_dim if value_head_dim is None else value_head_dim
    ratio = heads // kv_heads
    for batch, (q, kv, split) in enumerate(zip(q_lens, kv_lens, splits)):
        total_n = math.ceil(kv / block_n)
        per_head = []
        for m in range(math.ceil(q * ratio / block_m)):
            start = m * block_m
            rows = min(block_m, q * ratio - start)
            last_q = (start + rows - 1) // ratio
            causal_n = min(total_n, max(0, math.ceil(
                (last_q + 1 + kv - q) / block_n))) if causal else total_n
            # BlockMN::get_n_block_min_max applies the causal bound before
            # its runtime split quotient, not after a whole-request split.
            n_per_split = math.ceil(causal_n / split)
            row = []
            for s in range(split):
                lo = s * n_per_split
                hi = min((s + 1) * n_per_split, causal_n)
                if hi > lo:
                    live_kv = min(kv, hi * block_n) - lo * block_n
                    first_kv = min(block_n, kv - (hi - 1) * block_n)
                    first_masked = min(rows, max(
                        ((hi - 1) * block_n - kv + q) * ratio - start, 0)) if causal else 0
                else:
                    live_kv, first_kv, first_masked = 0, None, None
                row.append((m, s, start, rows, lo, hi, live_kv,
                            first_kv, first_masked))
            per_head.append(row)
        if causal and split == 1:
            divisor = next((v for v in (1, 2, 4, 8, 16) if ratio <= v), 16)
            budget = ((32 * 1024 * 1024) // divisor
                      // (block_n * (head_dim + value_head_dim) * 2))
            group = min(kv_heads, next(
                (v for v in (16, 8, 4, 2) if total_n * v <= budget), 1))
            for first_head in range(0, kv_heads, group):
                for row in reversed(per_head):
                    for head in range(first_head, min(first_head + group, kv_heads)):
                        yield _Fa3GqaSourceTask(batch, head, *row[0])
        else:
            m_order = tuple(reversed(per_head)) if causal else tuple(per_head)
            for head in range(kv_heads):
                for s in range(split):
                    for row in m_order:
                        yield _Fa3GqaSourceTask(batch, head, *row[s])


def _fa3_gqa_query_tma(kernel, block_m):
    ratio = kernel.shape["num_query_heads"] // kernel.shape["num_kv_heads"]
    # This owned build forces PackGQA even for ratio1. Its block-H1 path
    # loads Q through cp.async; only the explicitly enabled packed block-H
    # variants use Q TMA. MHA does not imply !PackGQA in this source.
    return (kernel.dtype.lower() == "bf16" and block_m > 64
        and ratio in (4, 8, 16) and kernel.launch.algorithm not in (
            _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM))


def _fa3_gqa_root_extra_rows(task, *, block_m, block_n, query_tma,
                            paged_tma, contiguous_kv, q_suffix, kv_suffix, ratio):
    """Source-valid TMA rows beyond this request, excluding descriptor OOB fill.

    Varlen descriptors are bounded by the flattened tensor, not a request.
    Only the first descending KV panel can overrun its logical sequence.
    These repeated requested bytes do not imply extra compulsory DRAM reads.
    """
    if not task.panel_count:
        return 0, 0
    q_extra = (min(block_m, q_suffix[task.batch_index] * ratio - task.packed_q_start)
               - task.packed_q_rows if query_tma else 0)
    k_extra = (min(block_n, kv_suffix[task.batch_index]
                  - (task.kv_block_stop - 1) * block_n) - task.first_panel_kv_rows
               if contiguous_kv and not paged_tma else 0)
    if q_extra < 0 or k_extra < 0:
        raise ValueError("FA3 TMA root cannot truncate a live source row")
    return q_extra, k_extra


def _fa3_gqa_observed_work(kernel: KernelSpec, hardware):
    """Packed GQA tile panels and partial O/LSE extents from pinned FA3.

    Work follows tile_scheduler.hpp and mainloop_fwd_sm90_tma_gmma_ws.hpp.
    Neither the NCU traffic counters nor any production duration is read.
    The physical cache reuse and within-CTA service remain explicit partial
    estimates; the variable per-request split and causal prefix are retained.
    """

    q_lens, kv_lens = _fa3_lengths(kernel)
    heads = kernel.shape["num_query_heads"]
    kv_heads = kernel.shape["num_kv_heads"]
    ratio, d = heads // kv_heads, kernel.shape["head_dim"]
    tile = kernel.launch.tiling.tile_shape
    # A combine observation retains its own tile, while its source input is
    # the same dynamic split schedule of the sibling main. For all recorded
    # GQA split calls the main uses the single-warpgroup decode tile or the
    # 128x128 causal/mixed tile.
    combine = kernel.launch.algorithm in (
        _FA3_GQA_COMBINE_030_ALGORITHM, _FA3_GQA_TMA_COMBINE_030_ALGORITHM,
        _FA3_PREFILL_COMBINE_ALGORITHM)
    paged_tma = kernel.launch.algorithm in (
        _FA3_GQA_TMA_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM,
        _FA3_GQA_TMA_COMBINE_030_ALGORITHM)
    if combine:
        one_wg = (max(q_lens) * ratio <= 64
                  and kernel.launch.algorithm != _FA3_PREFILL_COMBINE_ALGORITHM)
        block_m = 128 if d == 256 else (64 if one_wg else (192 if d == 64 else 128))
        paged = kernel.attributes["kv_layout"] in ("paged", "paged_varlen", "chunked_prefill")
        causal = kernel.attributes["causal"] and (max(q_lens) > 1 or paged and d == 128)
        paged = paged and not paged_tma
        if kernel.launch.algorithm == _FA3_PREFILL_COMBINE_ALGORITHM:
            block_n = 128
        elif d == 64:
            block_n = 192 if one_wg else (128 if causal else (160 if paged else 192))
        elif d == 128:
            block_n = 128 if causal or paged else (176 if one_wg else 160)
        else:
            block_n = 80
    else:
        block_m, block_n = tile["q"], tile["kv"]
    splits = hardware.fa3_gqa_observed_source_splits(
        kernel, block_m=block_m, block_n=block_n)
    output_rows = heads * sum(q_lens)
    # epilogue_fwd.hpp: Split+Varlen uses the per-request dynamic count.
    # A dynamic count of one writes final BF16 O/LSE directly even when
    # the compiled template is Split; combine returns before reading it.
    split_rows = heads * sum(q for q, s in zip(q_lens, splits) if s > 1)
    partial_rows = heads * sum(q * s for q, s in zip(q_lens, splits) if s > 1)
    partial_work = _fa3_gqa_partial_source_work(
        q_lens, kv_lens, heads=heads, kv_heads=kv_heads,
        block_m=block_m, block_n=block_n, splits=splits,
        causal=kernel.attributes["causal"] and max(q_lens) > 1)
    if combine:
        # Each value-column CTA reloads LSE, while only k_block0 writes the
        # final LSE. D256 spans two128-column combine tiles.
        value_blocks = math.ceil(d / tile["value_head"])
        partial_o_reads = partial_work["combine_partial_output_read_rows"] * d * 4
        partial_lse_reads = partial_rows * value_blocks * 4
        read_bytes = partial_o_reads + partial_lse_reads
        rows = tile["rows"]
        compiled_splits = 1 << max(4 if tile["value_head"] == 64 else 5,
            (kernel.launch.tiling.num_kv_splits - 1).bit_length())
        # The shared LSE transpose and its reductions execute across the
        # compiled padded split capacity, including masked tail rows.
        # Only dynamic-one requests bypass normalization. Every column CTA
        # normalizes its own copy; reciprocal/log run on every lane holding
        # the all-reduced row state, not one useful scalar per output row.
        normalization_rows = sum(math.ceil(q * heads / rows) * rows
            for q, s in zip(q_lens, splits) if s > 1) * value_blocks
        row_span = min(128, rows & -rows)
        reduction_lanes = kernel.launch.block_size // row_span
        # Both floating allreduces produce the row result on every lane.
        # The local-max seed and final log-sum add cancel one lane term;
        # the two butterfly trees add G*log2(G) operations each.
        normalization_flops = (4 * compiled_splits
            + 2 * reduction_lanes * (reduction_lanes.bit_length() - 1))
        return {"blocks": (), "tensor_flops": 0,
                **partial_work,
                "scalar_flops": (2 * d * partial_work["weighted_partial_output_rows"]
                                 + normalization_rows * normalization_flops),
                "sfu_ops": normalization_rows * (compiled_splits + 2 * reduction_lanes),
                "combine_normalization_rows": normalization_rows,
                "combine_compiled_splits": compiled_splits,
                "combine_reduction_lanes": reduction_lanes,
                "combine_normalization_flops_per_row": normalization_flops,
                "partial_output_read_bytes": partial_o_reads,
                "partial_lse_read_bytes": partial_lse_reads,
                "logical_l2_read_bytes": read_bytes,
                "read_bytes": read_bytes,
                "minimum_physical_read_bytes": read_bytes,
                "tma_kv_expected_transaction_bytes": 0,
                "tma_kv_padding_read_bytes": 0,
                # Keep the native O/LSE tensor union separate from the
                # one INT32 reset, which executes before split/row bypass.
                "write_bytes": split_rows * (d * 2 + 4),
                "semaphore_reset_bytes": 4,
                "scores": partial_rows, "output_rows": split_rows,
                "splits": splits, "block_m": tile["rows"],
                "block_n": tile["value_head"], "head_dim": d}
    causal = kernel.attributes["causal"] and max(q_lens) > 1
    query_tma = _fa3_gqa_query_tma(kernel, block_m)
    q_suffix = tuple(reversed(tuple(itertools.accumulate(reversed(q_lens)))))
    kv_suffix = tuple(reversed(tuple(itertools.accumulate(reversed(kv_lens)))))
    blocks = []
    logical_reads = issued_kv_bytes = live_kv_bytes = 0
    for task in _fa3_gqa_source_tasks(
            q_lens, kv_lens, heads=heads, kv_heads=kv_heads,
            block_m=block_m, block_n=block_n, head_dim=d,
            splits=splits, causal=causal):
        panels = task.panel_count
        blocks.append(panels)
        if panels:
            # A paged TMA descriptor is bounded by its physical page, not
            # the logical request length. An eligible page contains each
            # full N tile, including the masked logical KV tail. cp.async
            # instead skips K tails and zero-fills V tails without reading.
            rows_read = panels * block_n if paged_tma else task.read_kv_rows
            kv_bytes = rows_read * d * 4
            issued_kv_bytes += kv_bytes
            live_kv_bytes += task.read_kv_rows * d * 4
            logical_reads += task.packed_q_rows * d * 2 + kv_bytes
            q_extra, k_extra = _fa3_gqa_root_extra_rows(task,
                block_m=block_m, block_n=block_n, query_tma=query_tma,
                paged_tma=paged_tma, contiguous_kv=kernel.attributes["kv_layout"] == "contiguous",
                q_suffix=q_suffix, kv_suffix=kv_suffix, ratio=ratio)
            logical_reads += (q_extra * 2 + k_extra * 4) * d
    split_output = kernel.launch.algorithm in (
        _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM)
    partial_output_bytes = partial_work["materialized_partial_output_rows"] * d * 4 if split_output else 0
    partial_lse_bytes = partial_rows * 4 if split_output else 0
    partial_write_bytes = partial_output_bytes + partial_lse_bytes
    direct_write_bytes = (output_rows - split_rows if split_output else output_rows) * (d * 2 + 4)
    scores = sum(blocks) * block_m * block_n
    read_bytes = (heads * sum(q_lens) * d * 2
                  + kv_heads * sum(kv_lens) * d * 4)
    # Public operand payload remains logical for the caller ABI. The cold
    # read floor additionally includes every physically addressed KV-tail
    # row. Source-valid over-reads repeat rows already in the allocation;
    # their cache residency is unobserved and adds no compulsory DRAM bytes.
    minimum_physical_reads = (heads * sum(q_lens) * d * 2
        + kv_heads * sum(math.ceil(kv / block_n) * block_n
                         for q, kv in zip(q_lens, kv_lens) if q > 0) * d * 4
        if paged_tma else read_bytes)
    return {"blocks": tuple(blocks), "tensor_flops": 4 * d * scores,
            **partial_work,
            "logical_l2_read_bytes": logical_reads,
            "read_bytes": read_bytes,
            "minimum_physical_read_bytes": minimum_physical_reads,
            "tma_kv_expected_transaction_bytes": issued_kv_bytes if paged_tma else 0,
            "tma_kv_padding_read_bytes": issued_kv_bytes - live_kv_bytes if paged_tma else 0,
            "write_bytes": partial_write_bytes + direct_write_bytes,
            "partial_write_bytes": partial_write_bytes,
            "partial_output_bytes": partial_output_bytes,
            "partial_lse_bytes": partial_lse_bytes,
            "direct_write_bytes": direct_write_bytes,
            "semaphore_reset_bytes": 0,
            "scores": scores, "output_rows": output_rows,
            "splits": splits, "block_m": block_m,
            "block_n": block_n, "head_dim": d}


def _fa3_gqa_register_services(hardware, block_n=128):
    """Source-sized H200 register chains; no FA2 launch or CTA-overlap factors.

    Independent H200 probes measure max+quad shuffle and local dependent-add
    chains at exact two-row widths. These primitives match Softmax<2>'s
    register geometry. Composition remains an approximation to its compiler
    schedule, not a whole native-softmax rate.
    Conversion capacity is handled separately from these register reductions.
    Neither the old conversion loop nor the fused exp loop whose repeated
    inputs overflow is used as a FA3 service.
    """
    if not hardware.supports_source_contract("vllm_0_30_fa3_gqa_fresh_prefill"):
        raise ValueError("FA3 register services require H200 mechanism evidence")
    model = hardware.register_reduction_dependency_cycles
    if model is None:
        raise NotImplementedError("H200 register-chain evidence is required")
    if model["threads_per_cta"] != 128 or model["quad_reduction_width"] != 4:
        raise ValueError("FA3 register-chain thread/quad geometry differs")
    entries = tuple(entry for entry in model["entries"]
        if (entry["lane_local_rows_per_thread"],
            entry["score_values_per_lane_local_row"]) == (2, 32))
    if len(entries) != 1:
        raise ValueError("FA3 requires one measured two-row/32-value chain")
    entry = entries[0]
    row_values = block_n // 4
    if block_n != 128:
        measured = tuple(row for row in model["additional_register_chains"]
            if row["score_values_per_lane_local_row"] == row_values)
        if len(measured) != 1 or block_n % 4:
            raise NotImplementedError("FA3 register width requires exact independent evidence")
        # Quad sum still has two rows and two shuffle levels, independent
        # of the number of lane-local values reduced before finalization.
        entry = {**entry, **measured[0]}
    clock = hardware.nominal_sm_freq_hz
    # First reduce_max assigns the first element, then performs one fewer
    # local comparison than the probe, followed by the same two quad levels.
    first_max = entry["fp32_max_quad4_cycles"] * (row_values + 1) / (row_values + 2)
    first_sum = entry["fp32_local_sum_cycles"] * (row_values - 1) / row_values
    return {
        "first_online": (first_max + first_sum) / clock,
        "first_sum": first_sum / clock,
        "online": (entry["fp32_max_quad4_cycles"]
                   + entry["fp32_local_sum_cycles"]) / clock,
        "online_max": entry["fp32_max_quad4_cycles"] / clock,
        "online_sum": entry["fp32_local_sum_cycles"] / clock,
        "quad_sum": entry["fp32_quad4_sum_cycles"] / clock,
    }


def _fa3_final_row_services(value_head_dim, quad_sum_s, per_sm_capacity):
    """Final normalization and O scaling of one64-row consumer WG.

    Softmax<2> retains two row states per thread, replicated in quad4.
    Only the final quad sum uses an independent dependency service; RCP,
    log and O multiplication keep their existing capacity/issue estimates.
    """
    return {
        "finalize": (quad_sum_s
                     + 2 * 256 / per_sm_capacity("compute_sfu")
                     + 3 * 256 / per_sm_capacity("compute_fma")),
        "rescale": max(64 * value_head_dim / per_sm_capacity("compute_fma"),
                       2 * value_head_dim / per_sm_capacity("warp_issue")),
    }


def _fa3_ss_ex2_delay(start, work, intervals, ends, loss):
    """Run pure EX2 work through known SS busy intervals, in seconds."""
    if not work:
        return 0.0
    remaining, now = work, start
    for index in range(bisect_right(ends, start), len(intervals)):
        begin, end = intervals[index]
        if begin >= now + remaining:
            break
        if begin > now:
            remaining -= begin - now
            now = begin
        available = end - now
        if available <= 0:
            continue
        elapsed = min(available, remaining / (1 - loss))
        remaining = max(0.0, remaining - elapsed * (1 - loss))
        now += elapsed
        if not remaining:
            break
    return max(0.0, now + remaining - start - work)


def _fa3_gqa_capacity_task_cost(*args, ss_ex2_sharing=None, **kwargs):
    """Existing source schedule, optionally resolving SS/EX2 busy sharing.

    The optional service decomposition contains only EX2 work and its
    following local sum. Scalar dependencies and all other resources retain
    their existing services. Iteration updates the same source schedule;
    a stale uncoupled Tensor timeline is never used as the final result.
    """
    if ss_ex2_sharing is None:
        return _fa3_gqa_capacity_task_cost_once(*args, **kwargs)
    required = {"loss_fraction", "cycle_s", "first_sfu_s", "first_post_s",
                "online_sfu_s", "online_post_s", "prefix_sfu_s", "prefix_post_s"}
    if not isinstance(ss_ex2_sharing, dict) or set(ss_ex2_sharing) != required:
        raise ValueError("SS/EX2 sharing requires its complete service decomposition")
    for value in ss_ex2_sharing.values():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("SS/EX2 services must be finite nonnegative numbers")
    model = ss_ex2_sharing
    if (not 0 <= model["loss_fraction"] < 1 or model["cycle_s"] <= 0
            or model["prefix_sfu_s"] > model["online_sfu_s"]
            or model["prefix_post_s"] > model["online_post_s"]
            or (kwargs["warp_groups"], kwargs["head_dim"], kwargs["block_n"]) != (2, 128, 128)):
        raise ValueError("SS/EX2 sharing requires the bound two-WG N128/D128 domain")
    if model["loss_fraction"] == 0:
        return _fa3_gqa_capacity_task_cost_once(*args, **kwargs)
    extras = None
    previous = None
    iteration = 0
    while True:
        trace = {"ss": [], "scalar": []}
        result = _fa3_gqa_capacity_task_cost_once(*args, **kwargs,
            _ss_ex2_services=model, _ss_ex2_trace=trace, _scalar_extras=extras)
        if not trace["scalar"]:
            return result
        intervals = trace["ss"]
        ends = [end for _, end in intervals]
        updated = [_fa3_ss_ex2_delay(start, work, intervals, ends, model["loss_fraction"])
                   for start, work in trace["scalar"]]
        if extras is not None and len(extras) != len(updated):
            raise ValueError("SS/EX2 schedule changed its scalar dependency nodes")
        # One physical cycle over a complete task is the numerical budget,
        # distributed across its existing scalar dependency nodes.
        tolerance = model["cycle_s"] / (len(updated) + 1)
        if (extras is not None and max(abs(a-b) for a, b in zip(extras, updated)) <= tolerance
                and abs(result - previous) <= model["cycle_s"]):
            return result
        iteration += 1
        if iteration > 64 + 4 * len(updated):
            raise NotImplementedError("SS/EX2 source schedule did not converge")
        extras = updated if extras is None else [(a+b) / 2 for a, b in zip(extras, updated)]
        previous = result


def _fa3_gqa_capacity_task_cost_once(panels, q_rows, first_kv_rows, *, qk, pv,
                               first_online, online, conversion, rescale,
                               finalize, copy_bps, smem_bps, latency, page_lookup,
                               copy_issue, kv_tma, full_kv_tiles, warp_groups, output_dtype_bytes,
                               qk_completion, head_dim, block_n,
                               online_after_pv_wait=False,
                               q_copy_issue_per_four_rows=0.0,
                               q_copy_fixed_issue=0.0, value_head_dim=None,
                               kv_latency=None, first_k_ready_s=0.0, kv_copy_bps=None,
                               q_latency=None, q_copy_bps=None, first_q_ready_s=0.0,
                               pv_retirement_s=None,
                               kv_full_handoff_s=0.0, kv_empty_handoff_s=0.0,
                               qk_enqueue_s=0.0, pv_enqueue_s=0.0, pv_async_tail_s=None,
                               first_qk_cohort_s=0.0,
                               online_before_pv_wait_s=None,
                               first_mask_s=0.0, causal_mask_s=0.0,
                               causal_masked_panels=0, range_setup_s=0.0,
                               causal_loop_setup_s=0.0,
                               _ss_ex2_services=None,
                               _ss_ex2_trace=None, _scalar_extras=None,
                               q_extra_l2_rows=0, first_kv_extra_l2_rows=0):
    """Source K/V two-slot schedule with register or per-WG shared P.

    Services are seconds derived from existing HW capacities, not operator
    durations. Tensor and vector work each share one SM resource; Tensor
    enqueue/control instructions are explicitly idealized. Persistent tasks
    remain serial at their output boundary, so next-task prefetch is not priced.
    """
    if not math.isfinite(first_k_ready_s) or first_k_ready_s < 0:
        raise ValueError("FA3 initial K readiness must be finite and nonnegative")
    if (isinstance(range_setup_s, bool) or not isinstance(range_setup_s, (int, float))
            or not math.isfinite(range_setup_s) or range_setup_s < 0):
        raise ValueError("FA3 range setup must be finite nonnegative seconds")
    if (isinstance(causal_loop_setup_s, bool) or not isinstance(causal_loop_setup_s, (int, float))
            or not math.isfinite(causal_loop_setup_s) or causal_loop_setup_s < 0):
        raise ValueError("FA3 causal-loop setup must be finite nonnegative seconds")
    if pv_retirement_s is not None and (
            isinstance(pv_retirement_s, bool) or not isinstance(pv_retirement_s, (int, float))
            or not math.isfinite(pv_retirement_s) or pv_retirement_s <= 0):
        raise ValueError("FA3 PV retirement must be finite positive seconds")
    if pv_async_tail_s is not None and (
            isinstance(pv_async_tail_s, bool) or not isinstance(pv_async_tail_s, (int, float))
            or not math.isfinite(pv_async_tail_s) or pv_async_tail_s <= 0):
        raise ValueError("FA3 PV asynchronous suffix must be finite positive seconds")
    kv_copy_bps = copy_bps if kv_copy_bps is None else kv_copy_bps
    if (isinstance(kv_copy_bps, bool) or not isinstance(kv_copy_bps, (int, float))
            or not math.isfinite(kv_copy_bps) or kv_copy_bps <= 0):
        raise ValueError("FA3 KV path capacity must be finite positive bytes/s")
    value_head_dim = head_dim if value_head_dim is None else value_head_dim
    kv_latency = latency if kv_latency is None else kv_latency
    if (not math.isfinite(kv_latency) or kv_latency < latency):
        raise ValueError("FA3 KV ready latency must include its L2 prefix")
    q_copy_bps = copy_bps if q_copy_bps is None else q_copy_bps
    q_latency = latency if q_latency is None else q_latency
    if (isinstance(q_copy_bps, bool) or not isinstance(q_copy_bps, (int, float))
            or not math.isfinite(q_copy_bps) or q_copy_bps <= 0):
        raise ValueError("FA3 Q path capacity must be finite positive bytes/s")
    if (isinstance(q_latency, bool) or not isinstance(q_latency, (int, float))
            or not math.isfinite(q_latency) or q_latency < latency):
        raise ValueError("FA3 Q ready latency must include its L2 prefix")
    if ((warp_groups, head_dim, value_head_dim, block_n) not in (
            (1, 128, 128, 128), (2, 128, 128, 128), (3, 64, 64, 128), (2, 256, 256, 80),
            (2, 128, 128, 160), (1, 128, 128, 176), (1, 64, 64, 192),
            (3, 64, 64, 160), (3, 64, 64, 192), (2, 192, 128, 128))
            or output_dtype_bytes not in (2, 4)):
        raise ValueError("FA3 source schedule requires bound WG/head geometry and 16-bit/FP32 output")
    if type(online_after_pv_wait) is not bool:
        raise ValueError("FA3 compiled online/PV order must be explicit bool")
    if online_before_pv_wait_s is not None and (
            isinstance(online_before_pv_wait_s, bool)
            or not isinstance(online_before_pv_wait_s, (int, float))
            or not math.isfinite(online_before_pv_wait_s)
            or not 0 <= online_before_pv_wait_s <= online):
        raise ValueError("FA3 online prefix must be finite seconds within its full service")
    if type(kv_tma) is not bool:
        raise ValueError("FA3 KV TMA mode must be an explicit boolean")
    if type(full_kv_tiles) is not bool or full_kv_tiles and not kv_tma:
        raise ValueError("FA3 full physical KV tiles require explicit TMA mode")
    for rows in (q_extra_l2_rows, first_kv_extra_l2_rows):
        if type(rows) is not int or rows < 0:
            raise ValueError("FA3 extra TMA rows must be nonnegative integers")
    if (q_extra_l2_rows > 64 * warp_groups - q_rows
            or first_kv_extra_l2_rows and (not kv_tma or full_kv_tiles)
            or not panels and (q_extra_l2_rows or first_kv_extra_l2_rows)
            or panels and first_kv_extra_l2_rows > block_n - first_kv_rows):
        raise ValueError("FA3 extra TMA rows exceed their source box")
    if online_after_pv_wait and (warp_groups, head_dim, block_n) not in (
            (1, 128, 128), (2, 128, 128), (3, 64, 128), (1, 128, 176)):
        raise ValueError("FA3 compiled online/PV order requires bound native WG/head/N evidence")
    if not 0 < q_rows <= 64 * warp_groups:
        raise ValueError("FA3 live query rows exceed its consumer tile")
    for service in (q_copy_issue_per_four_rows, q_copy_fixed_issue,
                    kv_full_handoff_s, kv_empty_handoff_s, qk_enqueue_s, pv_enqueue_s,
                    first_mask_s, causal_mask_s, first_qk_cohort_s, first_q_ready_s):
        if (isinstance(service, bool) or not isinstance(service, (int, float))
                or not math.isfinite(service) or service < 0):
            raise ValueError("FA3 copy/notification/mask services must be finite nonnegative seconds")
    if first_qk_cohort_s and (warp_groups, head_dim, value_head_dim, block_n) != (2, 128, 128, 128):
        raise ValueError("FA3 coissued QK retirement requires the bound two-WG N128/D128 domain")
    if (type(causal_masked_panels) is not int
            or not 0 <= causal_masked_panels <= max(panels - 1, 0)):
        raise ValueError("FA3 causal mask count exceeds its remaining panels")
    if kv_tma and (kv_full_handoff_s or kv_empty_handoff_s):
        raise ValueError("FA3 cp.async notification evidence cannot price TMA barriers")
    if panels == 0:
        # mma's empty-range return still reaches epilogue.store_zero. Each
        # live packed row publishes -inf LSE; runtime-unsplit also zeros O.
        # No Q/KV read, shared staging or compute precedes these stores.
        # Use the same local L2 completion approximation as the nonempty
        # epilogue, competing with the caller's global byte floor by max.
        # Serial task output completion remains an analytical approximation;
        # store_zero does not explicitly wait for L2 retirement per tile.
        output_bytes = q_rows * (4 + (value_head_dim * 2 if output_dtype_bytes == 2 else 0))
        return range_setup_s + latency + output_bytes / copy_bps
    rescale_before_pv = head_dim > 128
    scheduler_token = head_dim <= 128 and warp_groups >= 2
    p_shared = head_dim == 64 and warp_groups == 3 and block_n in (160, 192)
    # These noncausal D64 tiles use SS PV. Each WG has its own single P
    # partition; LargeHeadDimV=false removes the PFull/PEmpty named barriers.
    # STSM/fence/syncwarp follows that WG's old PV wait0, not another WG.
    p_partition_bytes = 64 * block_n * 2 if p_shared else 0
    p_store_s = p_partition_bytes / smem_bps
    pv_completion = max(pv, p_store_s, 0.0 if pv_retirement_s is None else pv_retirement_s)
    # Producer and consumers compute BlockMN independently and concurrently.
    # Their runtime split quotient precedes K/Q requests and matrix work;
    # count one dependency latency, not one serialized service per warp.
    producer = memory = tensor = vector = range_setup_s
    k_ready, v_ready = [0.0, 0.0], [0.0, 0.0]
    k_release, v_release = [0.0, 0.0], [0.0, 0.0]
    p_ready = [0.0] * warp_groups

    def copy(byte_count, gate, issue, ready_latency, bandwidth=None, extra_l2_bytes=0):
        nonlocal producer, memory
        start = max(producer, gate)
        producer = start + issue
        rate = copy_bps if bandwidth is None else bandwidth
        # Extra source-valid rows may alias another request's input. Retain
        # their L2 byte service without inventing a DRAM miss fraction or a
        # second ready latency. Local port allocation remains approximate.
        memory = max(memory, start) + byte_count / rate + extra_l2_bytes / copy_bps
        return max(memory, producer) + ready_latency

    def mma(request, service, completion, prefix, suffix=None, *, ss=False):
        nonlocal tensor
        start = max(tensor, request)
        tensor = start + service
        if ss and _ss_ex2_trace is not None:
            _ss_ex2_trace["ss"].append((start, tensor))
        done = max(tensor, start + completion, request + prefix)
        # An asynchronous batch must finish enqueueing before its source
        # thread can release a buffer or run following scalar work. The
        # measured RS suffix limits how much execution can remain after
        # that point, without reserving Tensor for the whole retirement.
        enqueued = request + prefix
        if suffix is not None:
            enqueued = max(enqueued, done - suffix)
        return done, enqueued

    def scalar(request, service, *, ex2_kind=None):
        nonlocal vector
        if _ss_ex2_trace is None:
            vector = max(vector, request) + service
            return vector
        start = max(vector, request)
        work = post = 0.0
        if ex2_kind == "first":
            work, post = _ss_ex2_services["first_sfu_s"], _ss_ex2_services["first_post_s"]
        elif ex2_kind == "online":
            work, post = _ss_ex2_services["online_sfu_s"], _ss_ex2_services["online_post_s"]
        elif ex2_kind == "prefix":
            work, post = _ss_ex2_services["prefix_sfu_s"], _ss_ex2_services["prefix_post_s"]
        elif ex2_kind == "tail":
            work = _ss_ex2_services["online_sfu_s"] - _ss_ex2_services["prefix_sfu_s"]
            post = _ss_ex2_services["online_post_s"] - _ss_ex2_services["prefix_post_s"]
        elif ex2_kind is not None:
            raise ValueError("Unknown SS/EX2 scalar phase")
        if work + post > service * (1 + 1e-12):
            raise ValueError("EX2 and local-sum work exceed their source phase")
        index = len(_ss_ex2_trace["scalar"])
        extra = 0.0 if _scalar_extras is None else _scalar_extras[index]
        if not work and extra:
            raise ValueError("Non-EX2 scalar phase cannot consume sharing service")
        _ss_ex2_trace["scalar"].append((start + service - post - work, work))
        vector = start + service + extra
        return vector

    # The page-entry load feeds the pointer shuffle and cp.async issue.
    # Its cache tier is a declared L2-hit approximation, not an observation.
    producer += page_lookup
    row_bytes = head_dim * 2
    v_row_bytes = value_head_dim * 2
    # Paged TMA descriptors cover allocated physical pages. Their first
    # masked panel still transfers a full tile; cp.async skips/zero-fills
    # its logical tail. Contiguous TMA retains a live-byte lower bound,
    # since a root descriptor can read beyond a logical request boundary.
    # Keep the task's live geometry unchanged in all three cases.
    first_read_rows = block_n if full_kv_tiles else first_kv_rows
    first_k_start = producer
    k_ready[0] = copy(first_read_rows * row_bytes, 0.0, copy_issue, kv_latency,
                      bandwidth=kv_copy_bps,
                      extra_l2_bytes=first_kv_extra_l2_rows * row_bytes) + kv_full_handoff_s
    # The independent cold producer-completion floor already has its own
    # control path. Compete with it rather than serially adding another
    # notification to that measured first-burst envelope.
    k_ready[0] = max(k_ready[0], first_k_start + first_k_ready_s)
    # PackGQAManager uses eight lanes per row. A live four-row warp group
    # copies D/64 vectors. PackGQA's pointer SHFL precedes the live-row
    # predicate, so its full-tile issue joins the fixed producer control
    # budget. Only the copy issue and global payload shrink with the Q tail.
    # Charge this ingress once/task, rather than once per KV panel.
    q_issue = math.ceil(q_rows / 4) * q_copy_issue_per_four_rows + q_copy_fixed_issue
    q_start = producer
    q_ready = copy(q_rows * row_bytes, 0.0, q_issue, q_latency, bandwidth=q_copy_bps,
                   extra_l2_bytes=q_extra_l2_rows * row_bytes)
    q_ready = max(q_ready, q_start + first_q_ready_s)
    for wg in range(warp_groups):
        qk_done, _ = mma(max(q_ready, k_ready[0]), qk, qk_completion, qk_enqueue_s, ss=True)
        # Both consumers enter their first QK from the same Q/K readiness.
        # Independent coissued SS batches constrain the earliest completion;
        # this does not require every WG to await the last cohort member or
        # hold Tensor for the whole retirement interval. Steady QK/PV uses
        # the existing scheduler token and retains its current ordering.
        qk_done = max(qk_done, max(q_ready, k_ready[0]) + first_qk_cohort_s)
        k_release[0] = max(k_release[0], qk_done)
        first_done = scalar(qk_done, first_mask_s + first_online, ex2_kind="first")
        converted = scalar(first_done, conversion)
        # The prologue has no old O to rescale. For D<=128, O scaling
        # follows each subsequent online update and final normalization.
        # D>128 keeps its source rescale-before-PV path below, including
        # the first PV's explicitly initialized accumulator.
        p_ready[wg] = converted
        p_ready[wg] = max(p_ready[wg], converted + p_store_s)
        # After the first panel, the causal consumer computes its mask-loop
        # bound with a runtime quotient by cp_world_size, even when it is1.
        # WGs do this independently; copies and other consumers can progress.
        # This is separate from BlockMN's initial split quotient. A one-panel
        # task also reaches this control before its final PV; empty tasks do not.
        p_ready[wg] += causal_loop_setup_s
    k_release[0] += kv_empty_handoff_s

    token = 0.0
    for panel in range(1, panels):
        slot, previous = panel % 2, (panel - 1) % 2
        # Source loads the next page table before waiting for its K slot;
        # previous-V pointers are retained, rather than looked up again.
        producer += page_lookup
        k_ready[slot] = copy(block_n * row_bytes, k_release[slot], copy_issue, kv_latency,
                            bandwidth=kv_copy_bps) + kv_full_handoff_s
        previous_rows = first_read_rows if panel == 1 else block_n
        v_ready[previous] = copy(previous_rows * v_row_bytes, v_release[previous], copy_issue, kv_latency,
                                bandwidth=kv_copy_bps, extra_l2_bytes=(first_kv_extra_l2_rows
                                    * v_row_bytes if panel == 1 else 0)) + kv_full_handoff_s
        released_k = released_v = 0.0
        for wg in range(warp_groups):
            request = max(p_ready[wg], q_ready, k_ready[slot],
                          token if scheduler_token else 0.0)
            qk_done, qk_enqueued = mma(request, qk, qk_completion, qk_enqueue_s, ss=True)
            pv_enqueue = max(qk_enqueued, p_ready[wg], v_ready[previous])
            if rescale_before_pv:
                # QK dimensions above128 disable the WG scheduler token
                # and scale old O after QK enqueue, before the old-panel PV.
                pv_enqueue = max(pv_enqueue, scalar(qk_enqueued, rescale))
            pv_done, pv_enqueued = mma(pv_enqueue, pv, pv_completion, pv_enqueue_s, pv_async_tail_s)
            token = pv_enqueued
            wait1 = max(qk_done, pv_enqueued)
            released_k = max(released_k, wait1)
            if panel <= causal_masked_panels:
                # Mask predicates consume QK scores before max/exp/sum.
                # Keep the compiled early-PV-wait gate when it is bound;
                # this aggregate service does not establish a new compiler
                # placement or reserve Tensor while scalar work executes.
                wait1 = scalar(max(wait1, pv_done) if online_after_pv_wait else wait1,
                               causal_mask_s)
            prefix_done = wait1
            if online_before_pv_wait_s is None:
                online_done = scalar(max(wait1, pv_done) if online_after_pv_wait else wait1,
                                     online, ex2_kind="online")
            else:
                if online_before_pv_wait_s:
                    prefix_done = scalar(wait1, online_before_pv_wait_s, ex2_kind="prefix")
                tail = online - online_before_pv_wait_s
                gate = max(prefix_done, pv_done)
                # A zero-work gate does not reserve the shared scalar unit
                # while another warpgroup could execute independent work.
                online_done = scalar(gate, tail, ex2_kind="tail") if tail else gate
            # The compiled variant publishes V-empty directly after wait0,
            # before most online arithmetic. Keep that release distinct from
            # the conservative aggregate placement of its online work.
            released_v = max(released_v, max(prefix_done, pv_done) if online_after_pv_wait
                             else max(online_done, pv_done))
            converted = scalar(max(online_done, pv_done), conversion)
            p_ready[wg] = converted if rescale_before_pv else scalar(converted, rescale)
            # The previous PV has retired before conversion can overwrite
            # this WG's P. No global P-ready gate couples the other WGs.
            p_ready[wg] = max(p_ready[wg], converted + p_store_s)
        # All consumer WGs contribute to one empty event per buffer. Its
        # notification is neither per-WG nor per-byte work.
        k_release[slot] = released_k + kv_empty_handoff_s
        v_release[previous] = released_v + kv_empty_handoff_s

    last = (panels - 1) % 2
    last_rows = first_read_rows if panels == 1 else block_n
    v_ready[last] = copy(last_rows * v_row_bytes, v_release[last], copy_issue, kv_latency,
                        bandwidth=kv_copy_bps, extra_l2_bytes=(first_kv_extra_l2_rows
                            * v_row_bytes if panels == 1 else 0)) + kv_full_handoff_s
    done = 0.0
    for wg in range(warp_groups):
        request = max(p_ready[wg], v_ready[last])
        if rescale_before_pv:
            request = max(request, scalar(p_ready[wg], rescale))
        pv_done, pv_enqueued = mma(request, pv, pv_completion, pv_enqueue_s, pv_async_tail_s)
        finalized = scalar(pv_enqueued, finalize)
        rescaled = scalar(max(finalized, pv_done), rescale)
        # Varlen epilogue constructs/converts the16-bit shared staging before
        # its runtime split branch, even when global partial O remains FP32.
        # P has block_n/2 FP32 values/thread; O has value_head_dim/2.
        stored = scalar(rescaled, conversion * value_head_dim / block_n)
        done = max(done, stored)
    # Varlen epilogue_fwd.hpp first synchronizes every consumer and stages
    # the complete converted tile in shared memory, even for a packed-Q
    # tail or a runtime split output. Split partial O then comes directly
    # from FP32 registers; only final16-bit O is reloaded from shared.
    # These source stages use bandwidth lower bounds, not a borrowed
    # barrier latency or a measured whole-epilogue service.
    staging_bytes = 64 * warp_groups * value_head_dim * 2
    done += staging_bytes / smem_bps
    if output_dtype_bytes == 2:
        done += staging_bytes / smem_bps
    completed = copy(q_rows * (value_head_dim * output_dtype_bytes + 4), done, 0.0, latency)
    # P writes/SS reads share aggregate SM capacity, but prebooking another
    # WG's future store would invent an all-WG barrier. Keep local readiness
    # above and separately retain the total shared-byte capacity floor.
    shared_bytes = (2 * panels * warp_groups * p_partition_bytes
                    + staging_bytes * (2 if output_dtype_bytes == 2 else 1))
    return max(completed, range_setup_s + shared_bytes / smem_bps)


def _fa3_gqa_task_stage_cost(panels: int, qk: float, pv: float, softmax: float) -> float:
    """SM90 IntraWGOverlap prologue, steady iterations and PV drain.

    mainloop_fwd_sm90_tma_gmma_ws.hpp first computes QK+softmax without
    an earlier PV, then overlaps each following softmax with the previous
    PV, and finally waits for the last PV. An empty split does no MMA work.
    """
    if panels == 0:
        return 0.0
    return qk + softmax + (panels - 1) * (qk + max(pv, softmax)) + pv


def _validate_fa3_gqa_public_operands(kernel, hardware, work):
    """Consume actual public Q/cache/O operands without claiming scratch ABI."""
    if kernel.variant == "combine":
        return
    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    from src.kernel.memory_path_solver import public_attention_source_work
    public = public_attention_source_work(logical_interface_memory_access_program(kernel, hardware),
        expected_kernel=kernel, hardware=hardware)
    if public["read_bytes"] != work["read_bytes"]:
        raise ValueError("FA3 GQA public Q/K/V payload differs from the source call")


def _fa3_split_range_setup_s(hardware, clock_hz):
    """One BlockMN runtime signed32 quotient, shared by producer/consumers.

    Callers establish a compiled Split and positive pre-split N extent.
    Dynamic count1 still executes the quotient. This is not the initial
    scheduler's quotient or a serialized charge per participating warp.
    """
    integer = hardware.warp_integer_dependencies
    if integer is None:
        raise NotImplementedError(
            "FA3 split range requires independent signed32 division evidence")
    return integer["signed_divide_cycles"] / clock_hz


def _fa3_gqa_initial_metadata_s(batch_index, dynamic_splits, *, lpt, hardware,
                              split_template=False, clock_hz=None):
    """Prepared varlen scheduler -> optional head section -> sequence info.

    The source scans 31 batches per producer warp (lane31 is the prefix-sum
    sentinel). Initial work starts at batch0; each additional cohort needs
    another prepared-metadata load. LPT's unsplit head-section lookup uses
    the selected batch, and consumers then load that batch's sequence info.
    Each cohort requires a five-level warp scan. The selected work then
    divides by its M-block count; LPT sectioning or compiled Split adds a
    second runtime quotient, including dynamic-one Split. These are
    independent integer primitive services, not a full scheduler price.
    Small arrays remain explicitly assumed L2-resident.
    """
    levels = 1 + batch_index // 31 + int(lpt and dynamic_splits == 1) + 1
    integer = hardware.warp_integer_dependencies
    if integer is None:
        raise NotImplementedError("FA3 scheduler requires independent warp integer dependency evidence")
    clock = hardware.nominal_sm_freq_hz if clock_hz is None else clock_hz
    cycles = ((1 + batch_index // 31) * integer["inclusive_scan_cycles"]
              + (2 if lpt or split_template else 1) * integer["signed_divide_cycles"])
    return levels * hardware.l2_latency_ns * 1e-9 + cycles / clock


def _fa3_gqa_compiled_online_key(kernel, work):
    return (work["block_m"], work["block_n"], work["head_dim"],
        kernel.attributes["causal"],
        kernel.launch.algorithm in (_FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM),
        (kernel.attributes["kv_layout"] == "contiguous"
         or kernel.launch.algorithm in (_FA3_GQA_TMA_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM)),
        kernel.shape["num_query_heads"] > kernel.shape["num_kv_heads"])


def _fa3_gqa_compiled_online_wait(kernel, hardware, work):
    """A discrete native dependency projected by physical program fields."""
    if kernel.dtype.lower() != "bf16":
        return None
    profile = hardware.fa3_gqa_compiled_dependencies
    if profile is None:
        raise NotImplementedError("FA3 BF16 compiled order requires retained native dependency evidence")
    key = _fa3_gqa_compiled_online_key(kernel, work)
    return profile[key] if key in profile else None


def _fa3_gqa_parameter_read_service(kernel, hardware, work, active):
    """Aggregate incremental eager argument service, not per tile or SM sum."""
    if kernel.variant == "combine":
        # The retained SM90 Varlen combine programs have a separate, much
        # smaller argument struct than the main kernel. Reuse the same
        # independent region service, with its own static work count.
        if (kernel.dtype.lower() != "bf16" or graph_replayed_launch(kernel)
                or kernel.launch.algorithm not in (
                    _FA3_GQA_COMBINE_030_ALGORITHM, _FA3_GQA_TMA_COMBINE_030_ALGORITHM)):
            return None
        rows = kernel.launch.tiling.tile_shape["rows"]
        columns = kernel.launch.tiling.tile_shape["value_head"]
        compiled_splits = 1 << max(4 if columns == 64 else 5,
            (kernel.launch.tiling.num_kv_splits - 1).bit_length())
        key = rows, columns, compiled_splits, kernel.launch.block_size
        if key not in ((8, 128, 32, 256), (16, 64, 16, 256)):
            return None
        model = hardware.fa3_parameter_reads
        if model is None or key not in model["combine_parameter_regions"]:
            raise NotImplementedError("FA3 eager combine arguments require independent parameter-region evidence")
        q_lens, _ = _fa3_lengths(kernel)
        nonempty = sum(math.ceil(q * kernel.shape["num_query_heads"] / rows)
            for q, splits in zip(q_lens, work["splits"]) if splits > 1)
        resident_sms = min(active, nonempty)
        if not resident_sms:
            return 0.0
        return ((model["combine_parameter_regions"][key] - model["control_regions"])
                * (model["isolated_region_ns"] + (resident_sms - 1) * model["peer_region_ns"])
                * 1e-9)
    tile = (work["block_m"], work["block_n"], work["head_dim"])
    # These compiled GQA tiles have retained static parameter-region facts.
    # Keep the domain independent of whether its required profile is present.
    if (kernel.dtype.lower() != "bf16" or graph_replayed_launch(kernel)
            or tile not in ((64, 128, 128), (64, 176, 128), (64, 192, 64),
                            (128, 128, 128), (192, 128, 64))):
        return None
    model = hardware.fa3_parameter_reads
    if model is None or tile not in model["supported_main_tiles"]:
        raise NotImplementedError("FA3 eager argument reads require independent parameter-region evidence")
    resident_sms = sum(panels > 0 for panels in work["blocks"][:active])
    if not resident_sms:
        return 0.0
    # Each CTA reads the same launch parameters. The primitive measures the
    # completion period under concurrent SM requests. Charge the first cohort
    # once, without multiplying the critical-path delay by tiles or SMs.
    return ((model["main_parameter_regions"] - model["control_regions"])
            * (model["isolated_region_ns"] + (resident_sms - 1) * model["peer_region_ns"])
            * 1e-9)


def _fa3_varlen_combine_metadata_services(q_lens, rows, hardware):
    """Varlen combine's first-warp batch scan, then sequence metadata.

    STANDARD reads its batch directly from grid.z. The linearized source
    scans32 batches/cohort with five SHFL.UP/add levels, including CTAs
    that later return for dynamic split1. Retain the aggregate worst-cohort
    prefix rather than claim a per-CTA scheduling or cache observation.
    """
    latency = hardware.l2_latency_ns * 1e-9
    standard = (2 * math.ceil(sum(q_lens) / rows)
                >= len(q_lens) * math.ceil(max(q_lens) / rows))
    if standard:
        return latency, 0.0
    integer = hardware.warp_integer_dependencies
    if integer is None:
        raise NotImplementedError("FA3 combine batch scan requires independent warp scan evidence")
    cohorts = math.ceil(len(q_lens) / 32)
    scan = cohorts * integer["inclusive_scan_cycles"] / hardware.nominal_sm_freq_hz
    return (cohorts + 1) * latency + scan, scan


def _fa3_combine_resident_ctas(kernel, hardware, compiled_splits):
    """Resident ceiling for a caller-bound 46-register combine program."""
    tile = kernel.launch.tiling.tile_shape
    smem_bytes = 128 * math.ceil((
        tile["rows"] * tile["value_head"] * 4 * _FA3_COMBINE_STAGES
        + compiled_splits * tile["rows"] * 4 + tile["rows"] * 4 + 12) / 128)
    warps = _FA3_COMBINE_THREADS // 32
    unit = hardware.register_allocation_unit_per_warp
    registers = warps * (
        math.ceil(_FA3_COMBINE_REGISTERS_PER_THREAD * 32 / unit) * unit)
    return min(
        hardware.max_threads_per_unit // _FA3_COMBINE_THREADS,
        hardware.max_blocks_per_unit,
        hardware.max_regs_per_unit // registers,
        hardware.max_smem_per_unit_bytes // smem_bytes)


def _fa3_combine_metadata_residency_s(kernel, hardware, resident):
    """All launched CTA prefixes, including dynamic-one and padding exits.

    Retain the existing L2-hit/scan-transfer approximation. The total slot
    lifetime bounds device completion without imposing synchronized waves;
    the longest prefix separately protects small launches from dilution.
    Padding CTAs scan the batches but return before selected sequence loads.
    """
    q_lens = kernel.attributes["q_lens"]
    rows = kernel.launch.tiling.tile_shape["rows"]
    columns = kernel.launch.tiling.tile_shape["value_head"]
    heads = kernel.shape["num_query_heads"]
    value_blocks = math.ceil(kernel.shape["head_dim"] / columns)
    batch = len(q_lens)
    standard = 2 * math.ceil(sum(q_lens) / rows) >= batch * math.ceil(max(q_lens) / rows)
    expected = ((math.ceil(max(q_lens) * heads / rows), value_blocks, batch)
                if standard else
                (math.ceil(sum(q_lens) * heads / rows) + batch, value_blocks, 1))
    if tuple(kernel.launch.grid) != expected or resident <= 0:
        raise ValueError("FA3 combine metadata requires its source grid and positive residency")
    latency = hardware.l2_latency_ns * 1e-9
    if standard:
        return latency, kernel.launch.grid_size * latency / (hardware.num_compute_units * resident)
    integer = hardware.warp_integer_dependencies
    if integer is None:
        raise NotImplementedError("FA3 combine batch scan requires independent warp scan evidence")
    scan = integer["inclusive_scan_cycles"] / hardware.nominal_sm_freq_hz
    total = critical = 0.0
    coordinate_ctas = 0
    for index, length in enumerate(q_lens):
        ctas = math.ceil(length * heads / rows) * value_blocks
        cohorts = index // 32 + 1
        prefix = cohorts * (latency + scan) + latency
        coordinate_ctas += ctas
        total += ctas * prefix
        if ctas:
            critical = max(critical, prefix)
    padding = kernel.launch.grid_size - coordinate_ctas
    if padding:
        prefix = math.ceil(batch / 32) * (latency + scan)
        total += padding * prefix
        critical = max(critical, prefix)
    return critical, total / (hardware.num_compute_units * resident)


def _fa3_combine_local_services(*, rows, columns, compiled_splits, reduction_lanes,
                                valid_splits, latency_s, per_sm_capacity):
    """Pinned FA3 reducer stages at existing local resource capacities.

    LSE normalization publishes shared weights before partial-O accumulation.
    Three O panels are prefetched; each iteration issues i+3 before waiting
    for i, using four physical slots. Copy byte service and ready latency
    overlap normalization/accumulation. This is a local CTA approximation,
    not a measured reducer service or a grid-wide barrier.
    """
    if not valid_splits:
        return dict(partial_rounds=0.0, normalization=0.0,
            weight_shared_exchange=0.0, weighted_O=0.0,
            post_normalization=0.0, first_partial=0.0, output_ready=0.0)
    normalization_flops = 4 * compiled_splits + 2 * reduction_lanes * (reduction_lanes.bit_length() - 1)
    normalization_s = (
        rows * normalization_flops / per_sm_capacity("compute_fma")
        + rows * (compiled_splits + 2 * reduction_lanes) / per_sm_capacity("compute_sfu"))
    smem_bps = per_sm_capacity("smem")
    weight_exchange_s = 2 * rows * compiled_splits * 4 / smem_bps
    partial_values = rows * columns * valid_splits
    weighted_o_s = max(2 * partial_values / per_sm_capacity("compute_fma"),
        (partial_values + rows * valid_splits) * 4 / smem_bps)
    # The global L2 floor alone cannot constrain a dependent CTA's finite
    # copy queue. Reserve each finite O tile's bytes locally as well, then
    # compete the resulting chain with that global floor in the caller.
    # Extra initially prefetched invalid panels remain in the global byte
    # ledger; their local contention and uneven row tails stay partial.
    tile_copy_s = rows * columns * 4 / per_sm_capacity("l2_cache")
    panel_s = weighted_o_s / valid_splits
    ready = [0.0] * valid_splits
    memory = 0.0

    def load(index, request):
        nonlocal memory
        memory = max(memory, request) + tile_copy_s
        ready[index] = memory + latency_s

    for index in range(min(_FA3_COMBINE_STAGES - 1, valid_splits)):
        load(index, 0.0)
    normalized = latency_s + normalization_s + weight_exchange_s
    consumer = normalized
    for index in range(valid_splits):
        following = index + _FA3_COMBINE_STAGES - 1
        if following < valid_splits:
            load(following, consumer)
        consumer = max(consumer, ready[index]) + panel_s
    post_normalization_s = consumer - normalized
    # Retain the diagnostic key for the initial LSE wait plus O-copy stalls;
    # it is no longer a rounded number of whole-latency waves.
    copy_wait_s = max(0.0, post_normalization_s - weighted_o_s)
    output_s = latency_s + rows * (columns * 2 + 4) / per_sm_capacity("l2_cache")
    return dict(partial_rounds=latency_s + copy_wait_s, normalization=normalization_s,
        weight_shared_exchange=weight_exchange_s, weighted_O=weighted_o_s,
        post_normalization=post_normalization_s, first_partial=latency_s,
        output_ready=output_s)


def _fa3_gqa_observed_prediction(kernel: KernelSpec, hardware, *,
                               compiled_wait0_before_online=False):
    """Coarse source-stage critical path using independent hardware rates."""

    work = _fa3_gqa_observed_work(kernel, hardware)
    _validate_fa3_gqa_public_operands(kernel, hardware, work)
    clock, dram_clock = hardware.nominal_sm_freq_hz, hardware.nominal_dram_freq_hz
    tensor_clock = clock
    core_cycle_roles = ()
    measured_batch_domain = False
    ss_busy_capacity_domain = False
    notification_domain = False
    q_cp_async_issue_domain = False
    ss_ex2_sharing_domain = False
    online_prefix_s = None
    finite_tma_s = None
    def capacity(role, active):
        # The H200 Tensor rate is public FLOP/s normalized at the config's
        # nominal SM clock, not an architectural instruction/cycle count.
        # Keep that capacity basis. Only independently validated core-cycle
        # services use the active clock; SI memory services stay unchanged.
        return hardware.resource_by_role(role).effective_capacity(
            active, sm_freq_hz=(tensor_clock if role in core_cycle_roles else clock),
            dram_freq_hz=dram_clock)
    def per_sm_capacity(role):
        resource = hardware.resource_by_role(role)
        aggregate = capacity(role, 1)
        return (aggregate / hardware.num_compute_units
                if resource.scope == "shared" else aggregate)
    active = min(kernel.launch.grid_size, int(hardware.num_compute_units))
    combine = kernel.launch.algorithm in (
        _FA3_GQA_COMBINE_030_ALGORITHM, _FA3_GQA_TMA_COMBINE_030_ALGORITHM,
        _FA3_PREFILL_COMBINE_ALGORITHM)
    if type(compiled_wait0_before_online) is not bool:
        raise ValueError("FA3 compiler order must be explicit bool")
    if compiled_wait0_before_online and combine:
        raise ValueError("compiled PV/online order does not apply to combine")
    observed_wait0 = (None if combine else
        _fa3_gqa_compiled_online_wait(kernel, hardware, work))
    measured_tensor_clock = (not combine and kernel.dtype.lower() == "bf16"
        and (work["block_m"], work["block_n"], work["head_dim"]) in (
            (64, 128, 128), (128, 128, 128), (64, 176, 128), (64, 192, 64)))
    if measured_tensor_clock:
        model = hardware.bf16_wgmma_tensor_service_clock
        if model is None:
            raise NotImplementedError("BF16 Tensor service requires independent active-clock evidence")
        if work["block_m"] // 64 not in model["validated_active_warpgroups"]:
            raise NotImplementedError("BF16 Tensor clock lacks this WG validation")
        if work["block_n"] != 128 and hardware.fa3_gqa_wgmma_batches is None:
            raise NotImplementedError("BF16 non-N128 service requires independent batch evidence")
        tensor_clock = model["tensor_clock_hz"]
        core_cycle_roles = model["core_cycle_roles"]
    if combine:
        l2_s = (work["read_bytes"] + work["write_bytes"]
                + work["semaphore_reset_bytes"]) / capacity("l2_cache", active)
        scalar_s = (work["scalar_flops"] / capacity("compute_fma", active)
                    + work["sfu_ops"] / capacity("compute_sfu", active))
        latency = hardware.l2_latency_ns * 1e-9
        # Batch sorting is disabled in this source. STANDARD obtains the
        # batch directly from grid.z; LINEARIZE_M_AND_BATCH first reads
        # prepared per-batch Q extents to locate it. That lookup precedes
        # SeqlenInfo and dynamic-split loads; those latter loads can overlap.
        q_lens, _ = _fa3_lengths(kernel)
        rows = kernel.launch.tiling.tile_shape["rows"]
        metadata_s, batch_scan_s = _fa3_varlen_combine_metadata_services(q_lens, rows, hardware)
        local = _fa3_combine_local_services(
            rows=rows, columns=kernel.launch.tiling.tile_shape["value_head"],
            compiled_splits=work["combine_compiled_splits"],
            reduction_lanes=work["combine_reduction_lanes"],
            valid_splits=work["max_valid_splits"] if work["output_rows"] else 0,
            latency_s=latency, per_sm_capacity=per_sm_capacity)
        dependency_s, normalization_s = local["partial_rounds"], local["normalization"]
        weight_exchange_s, weighted_o_s = local["weight_shared_exchange"], local["weighted_O"]
        first_partial_s, post_normalization_s = local["first_partial"], local["post_normalization"]
        output_s = local["output_ready"]
        stage_chain_s = (metadata_s + first_partial_s + normalization_s
                         + weight_exchange_s + post_normalization_s + output_s)
        # Global byte/scalar capacity floors compete with a local CTA
        # dependency chain; no grid-wide barrier is imposed between stages.
        execution_s = max(l2_s, scalar_s, stage_chain_s)
        groups = {"fa3_gqa_combine_l2": l2_s,
                  "fa3_gqa_combine_scalar": scalar_s,
                  "fa3_gqa_combine_partial_rounds": dependency_s,
                  "fa3_gqa_combine_metadata": metadata_s,
                  "fa3_gqa_combine_batch_scan_dependency": batch_scan_s,
                  "fa3_gqa_combine_normalization": normalization_s,
                  "fa3_gqa_combine_weight_shared_exchange": weight_exchange_s,
                  "fa3_gqa_combine_weighted_O": weighted_o_s,
                  "fa3_gqa_combine_output_ready": output_s,
                  "fa3_gqa_combine_stage_chain": stage_chain_s}
        if (kernel.dtype.lower() == "bf16" and not graph_replayed_launch(kernel)
                and kernel.launch.algorithm in (
                    _FA3_GQA_COMBINE_030_ALGORITHM, _FA3_GQA_TMA_COMBINE_030_ALGORITHM)
                and (rows, kernel.launch.tiling.tile_shape["value_head"],
                     work["combine_compiled_splits"], kernel.launch.block_size)
                in ((8, 128, 32, 256), (16, 64, 16, 256),
                    (8, 128, 64, 256), (8, 128, 128, 256))):
            resident = _fa3_combine_resident_ctas(
                kernel, hardware, work["combine_compiled_splits"])
            metadata_critical_s, metadata_slot_s = _fa3_combine_metadata_residency_s(
                kernel, hardware, resident)
            # Dynamic-one and grid-padding CTAs still occupy their slots
            # while resolving metadata. Keep only that prefix lifetime;
            # the local body's output-completion approximation does not
            # establish when a CTA releases its physical resources.
            execution_s = max(execution_s, metadata_critical_s, metadata_slot_s)
            groups["fa3_gqa_combine_launched_metadata_chain"] = metadata_critical_s
            groups["fa3_gqa_combine_launched_metadata_slot_capacity"] = metadata_slot_s
    else:
        bm, bn, d = work["block_m"], work["block_n"], work["head_dim"]
        # Every owned D64/D128/D256 nonlocal tile has IntraWGOverlap=true,
        # including the64-row/single-consumer tile. Async PV overlaps the
        # next softmax, but the first softmax and last PV cannot disappear.
        qk = (2 * bm * bn * d) / per_sm_capacity("compute_tensor")
        pv = qk
        # FA3's quad Allreduce<4> uses floating max/sum and shuffles, not
        # the separate CUB block-reduction service measured for RMSNorm.
        # Charge the register arithmetic at the FP32 lane capacity; retain
        # instruction/shuffle dependency latency as an explicit limitation.
        softmax = (bm * bn / per_sm_capacity("compute_sfu")
                   + (4 * bm * bn + bm * d) / per_sm_capacity("compute_fma"))
        kv_tma = kernel.launch.algorithm in (
            _FA3_GQA_TMA_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM)
        full_kv_tiles = kv_tma
        kv_tma = kv_tma or kernel.attributes["kv_layout"] == "contiguous"
        capacity_schedule = (
            kernel.launch.algorithm in (_FA3_GQA_OBSERVED_030_ALGORITHM,
                                        _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM,
                                        _FA3_GQA_TMA_030_ALGORITHM,
                                        _FA3_GQA_TMA_SPLIT_030_ALGORITHM)
            and kernel.dtype.lower() in ("bf16", "fp16")
            and ((d == 128 and (bm, bn) in (
                    (64, 128), (128, 128), (128, 160), (64, 176)))
                 or (d == 64 and ((bm == 192 and bn == 128 and kernel.attributes["causal"])
                     or (bm, bn) in ((64, 192), (192, 160), (192, 192))))
                 or (d == 256 and bm == 128 and bn == 80))
            and kernel.attributes["kv_layout"] in (
                "contiguous", "paged", "paged_varlen", "chunked_prefill"))
        q_lens, kv_lens = _fa3_lengths(kernel)
        # With one packed-Q tile per request, each KV-head/split panel has
        # exactly one reader in this launch. Classify requests separately:
        # a long prefill does not give another request's decode KV a second
        # reader. Multi-M-tile cache reuse remains an explicit approximation.
        ratio = kernel.shape["num_query_heads"] // kernel.shape["num_kv_heads"]
        unique_kv_by_request = tuple(q * ratio <= bm for q in q_lens)
        unique_kv_panels = any(unique_kv_by_request)
        if compiled_wait0_before_online and not (
                capacity_schedule and not kv_tma
                and kernel.dtype.lower() == "bf16" and bm == bn == d == 128
                and kernel.attributes["causal"]
                and kernel.shape["num_query_heads"] == 4 * kernel.shape["num_kv_heads"]
                and kernel.launch.algorithm == _FA3_GQA_OBSERVED_030_ALGORITHM):
            raise ValueError("compiled PV/online order requires bound GQA4 unsplit prefill")
        if capacity_schedule:
            warp_groups = bm // 64
            # Packed TMA Q requires a block-H4/8/16 variant. The source
            # forces block-H1 for half, one consumer WG and compiled Split;
            # ratios such as2 also use block-H1, even with BF16/two WGs.
            q_tma = _fa3_gqa_query_tma(kernel, bm)
            q_cp_async_issue_domain = d == 128 and not q_tma
            if kernel.launch.block_size != (warp_groups + 1) * 128:
                raise ValueError("FA3 consumer geometry differs from its source block")
            q_lens, kv_lens = _fa3_lengths(kernel)
            scheduler_lpt = kernel.attributes["causal"] and (
                max(q_lens) > 1 or
                kernel.attributes["kv_layout"] != "contiguous" and 64 < d <= 128)
            memo, costs = {}, []
            first_wave_producers = sum(n > 0 for n in work["blocks"][:active])
            # K/V ingress writes shared memory and obtains a share of the
            # chip-wide L2 service. Empty resident CTAs do not consume that
            # share. These are capacity bounds, not measured copy periods;
            # local issue, readiness and two-slot release remain below.
            kv_ingress_bps = per_sm_capacity("l2_cache")
            cold_kv_ingress_bps = kv_ingress_bps
            if first_wave_producers:
                kv_ingress_bps = min(capacity("l2_cache", active) / first_wave_producers,
                                     per_sm_capacity("smem"))
                cold_kv_ingress_bps = min(kv_ingress_bps,
                    capacity("dram", active) / first_wave_producers)
            initial_k_ready_s = initial_q_ready_s = initial_metadata_s = 0.0
            # The register helper returns measured cycle services converted
            # at nominal clock. Recover that cycle basis and convert once
            # at the independently validated mixed-core clock. Fixed-SI
            # memory/parameter/launch services and normalized Tensor FLOP/s
            # never receive this ratio.
            registers = {key: seconds * clock / tensor_clock
                for key, seconds in _fa3_gqa_register_services(hardware, bn).items()}
            notification_domain = not kv_tma and d <= 128 and warp_groups in (1, 2)
            full_handoff_s = empty_handoff_s = 0.0
            if notification_domain:
                notification = hardware.cp_async_mbarrier_notification
                if notification is None:
                    raise NotImplementedError("FA3 cp.async K/V requires independent notification evidence")
                service = notification[128 * warp_groups]
                full_handoff_s = service["forward_cycles"] / tensor_clock
                empty_handoff_s = service["reverse_cycles"] / tensor_clock
            pv_retirement_s = pv_async_tail_s = None
            qk_enqueue_s = pv_enqueue_s = 0.0
            first_qk_cohort_s = 0.0
            measured_batch_domain = (kernel.dtype.lower() == "bf16"
                and (bm, bn, d) in ((64, 176, 128), (64, 192, 64),
                                   (64, 128, 128), (128, 128, 128)))
            if measured_batch_domain:
                batches = hardware.fa3_gqa_wgmma_batches
                if batches is None:
                    raise NotImplementedError("BF16 GQA requires independent typed WGMMA batch evidence")
                # The public SI Tensor capacity retains its original clock
                # basis. Independently backlogged SS instructions also bound
                # busy work at the existing active core clock. Take the
                # stronger capacity constraint; do not scale the public rate
                # or transfer this SS evidence to RS/PV or unmeasured opcodes.
                ss_capacity = batches["ss_busy_capacity"]
                if (bn in ss_capacity["instruction_n"]
                        and d // 16 == ss_capacity["k_steps"]
                        and warp_groups in ss_capacity["active_warpgroups"]):
                    ss_busy_capacity_domain = True
                    qk = max(qk, 2 * bm * bn * d
                             / (ss_capacity["flops_per_sm_cycle"] * tensor_clock))
                # RS P supplies four distinct register words/K16/thread.
                # SS PV and unmeasured tiles never borrow this RS service.
                pv_retirement_s = batches["retirement_s"][("rs", d, bn // 16)]
                queue = batches["enqueue"]
                qk_enqueue_s = queue["enqueue_cycles"][("ss", bn, d // 16)] / tensor_clock
                pv_enqueue_s = queue["enqueue_cycles"][("rs", d, bn // 16)] / tensor_clock
                pv_async_tail_s = queue["pv_retirement_tail_cycles"][d] / tensor_clock
                if (warp_groups, bn, d) == (2, 128, 128):
                    first_qk_cohort_s = queue["first_coissued_ss_completion_cycles"][bn, d // 16, 2] / tensor_clock
            if kernel.dtype.lower() == "bf16" and bn == 128:
                mechanisms = hardware.fa3_mla_mechanisms
                if mechanisms is None:
                    raise NotImplementedError("H200 BF16 QK instruction-batch evidence is required")
                # Same SS m64n128k16 instruction, d/16 K16 slices per WG.
                # The isolated fence/commit/wait0 batch is a retirement
                # approximation, not additional Tensor busy time.
                qk_completion = (mechanisms["wgmma_gemm_cycles"][("ss_n128_kmajor", d // 16, 1)]
                                 / tensor_clock)
            elif measured_batch_domain:
                qk_completion = batches["retirement_s"][("ss", bn, d // 16)]
            else:
                # Unmeasured instructions retain the explicit Tensor
                # capacity lower bound rather than borrow another opcode.
                qk_completion = qk / warp_groups
            # Source scale/exp work keeps its hardware-capacity estimate.
            # Max and local sum are replaced by the measured dependent chains,
            # so their former two scalar operations/score are removed here.
            scale_exp = (64 * bn / per_sm_capacity("compute_sfu")
                         + (2 * 64 * bn + 256) / per_sm_capacity("compute_fma"))
            online_s = (registers["online"] + scale_exp
                        + 256 / per_sm_capacity("compute_sfu")
                        + 3 * 256 / per_sm_capacity("compute_fma"))
            conversion_s = bn / per_sm_capacity("warp_issue")
            # mask.h applies a column predicate and two row selections per
            # column on the first panel, including full-size KV panels.
            # Causal predicates differ for the two rows. One WG has four
            # warps, two rows and N/4 columns per thread. Count source work
            # at existing integer/issue capacity; row-index/divmod setup and
            # predicate dependency latency remain explicit approximations.
            causal_mask = kernel.attributes["causal"] and max(q_lens) > 1
            comparisons = (2 if causal_mask else 1) * bn
            # The BF16 causal source checks runtime CP even for CP=1:
            # 4 warps * 2 rows * N/4 = 2N source guard instructions.
            # Retained programs have 62 repeated BRA and two guards lowered
            # to predicated work. Keep the coarse source projection rather
            # than claim an exact native issue count. Independent branch
            # packet latency did not validate composition across active warps;
            # neither that latency nor the skipped CP>1 body is priced here.
            mask_guard_issue_work = 0
            if (kernel.dtype.lower() == "bf16" and causal_mask
                    and _fa3_gqa_compiled_online_key(kernel, work)
                    in hardware.fa3_gqa_compiled_dependencies):
                mask_guard_issue_work = 2 * bn
            first_mask_s = max(32 * comparisons / per_sm_capacity("compute_alu"),
                              (comparisons + 2 * bn + mask_guard_issue_work)
                              / per_sm_capacity("warp_issue"))
            causal_mask_s = max(64 * bn / per_sm_capacity("compute_alu"),
                               (4 * bn + mask_guard_issue_work)
                               / per_sm_capacity("warp_issue"))
            if measured_batch_domain:
                conversion = hardware.packed_bf16_conversion
                if conversion is None:
                    raise NotImplementedError("BF16 GQA packed conversion requires independent capacity evidence")
                if (bn // 4 not in conversion["pairs_per_thread"]
                        or d // 4 not in conversion["pairs_per_thread"]
                        or warp_groups not in conversion["active_warpgroups"]):
                    raise NotImplementedError("BF16 GQA packed conversion geometry is unbound")
                # Each thread packs N/4 pairs; four warps issue N packed
                # instructions per WG. Integer feedback from the primitive
                # was removed by independent width/control differencing.
                # Retain the issue floor and use the existing core clock;
                # this is capacity work, not a full primitive/loop latency.
                conversion_s = max(conversion_s,
                    bn * conversion["cycles_per_warp_instruction"] / tensor_clock)
            ex2_prefix_fraction = sum_prefix_fraction = 0.0
            if kernel.dtype.lower() == "bf16":
                phases = hardware.fa3_gqa_compiled_online_prefix
                if phases is None:
                    raise NotImplementedError("FA3 BF16 online work requires static phase evidence")
                phase_key = _fa3_gqa_compiled_online_key(kernel, work)
                if phase_key in hardware.fa3_gqa_compiled_dependencies and phase_key not in phases:
                    raise NotImplementedError("FA3 bound online program lacks its static phase work")
                if phase_key in phases:
                    # Keep the same measured/capacity services and total
                    # work. Static counts only locate their guaranteed prefix
                    # before wait0. No duration or stall fraction is a weight.
                    services_by_op = {
                        "FMNMX": registers["online_max"],
                        "FADD": registers["online_sum"],
                        "EX2": (64 * bn + 256) / per_sm_capacity("compute_sfu"),
                        "FFMA": (2 * 64 * bn + 4 * 256) / per_sm_capacity("compute_fma")}
                    prefix_work = sum(services_by_op[op]
                        * (phases[phase_key][op][0] / phases[phase_key][op][1])
                        for op in services_by_op)
                    # Normalize the decomposition back to the unchanged full
                    # online service. Complete/empty prefixes remain exactly
                    # within [0,online],including floating-point endpoints.
                    online_prefix_s = online_s * (prefix_work / sum(services_by_op.values()))
                    if compiled_wait0_before_online:
                        online_prefix_s = 0.0
                    else:
                        ex2_prefix_fraction = phases[phase_key]["EX2"][0] / phases[phase_key]["EX2"][1]
                        sum_prefix_fraction = phases[phase_key]["FADD"][0] / phases[phase_key]["FADD"][1]
            ss_ex2_sharing = None
            causal_loop_setup_s = 0.0
            if causal_mask:
                integer = hardware.warp_integer_dependencies
                if integer is None:
                    raise NotImplementedError(
                        "FA3 causal-loop bound requires independent signed32 division evidence")
                causal_loop_setup_s = integer["signed_divide_cycles"] / tensor_clock
            ss_ex2_sharing_domain = (kernel.dtype.lower() == "bf16"
                and (warp_groups, d, bn) == (2, 128, 128))
            if ss_ex2_sharing_domain:
                sharing = hardware.ss_ex2_busy_sharing
                if sharing is None:
                    raise NotImplementedError("BF16 two-WG N128 requires independent SS/EX2 sharing evidence")
                # Only busy EX2 work loses capacity. Max/sum dependencies,
                # FFMA, RCP/log and conversion keep their existing services.
                # Static phase counts locate EX2 within the same aggregate
                # source phases; their fine interleaving remains approximate.
                online_ex2_s = (64 * bn + 256) / per_sm_capacity("compute_sfu")
                ss_ex2_sharing = dict(loss_fraction=sharing["loss_fraction"], cycle_s=1 / tensor_clock,
                    first_sfu_s=64 * bn / per_sm_capacity("compute_sfu"),
                    first_post_s=registers["first_sum"], online_sfu_s=online_ex2_s,
                    online_post_s=registers["online_sum"],
                    prefix_sfu_s=online_ex2_s * ex2_prefix_fraction,
                    prefix_post_s=registers["online_sum"] * sum_prefix_fraction)
            final_row = _fa3_final_row_services(d, registers["quad_sum"], per_sm_capacity)
            services = dict(qk=qk / warp_groups, pv=pv / warp_groups,
                ss_ex2_sharing=ss_ex2_sharing,
                causal_loop_setup_s=causal_loop_setup_s,
                kv_tma=kv_tma, full_kv_tiles=full_kv_tiles,
                warp_groups=warp_groups, head_dim=d, block_n=bn,
                online_after_pv_wait=(compiled_wait0_before_online or observed_wait0 is True),
                online_before_pv_wait_s=online_prefix_s,
                qk_completion=qk_completion,
                pv_retirement_s=pv_retirement_s,
                qk_enqueue_s=qk_enqueue_s, pv_enqueue_s=pv_enqueue_s, pv_async_tail_s=pv_async_tail_s,
                first_qk_cohort_s=first_qk_cohort_s,
                kv_full_handoff_s=full_handoff_s, kv_empty_handoff_s=empty_handoff_s,
                first_online=registers["first_online"] + scale_exp,
                # Four lanes each retain the two row states. Steady online
                # max rescaling adds exp2 plus three arithmetic operations
                # per lane-local row. No fused exp/local-sum service is added.
                online=online_s,
                conversion=conversion_s,
                first_mask_s=first_mask_s,
                causal_mask_s=causal_mask_s,
                # One FP32 FMUL per O value cannot consume the two-FLOP FMA
                # peak as if each multiply also performed an addition. Each
                # warp instruction updates32 values: retain its issue floor.
                rescale=final_row["rescale"],
                finalize=final_row["finalize"],
                copy_bps=per_sm_capacity("l2_cache"),
                # Requests with multiple M tiles retain KV's existing L2
                # reuse assumption. Q starts at the old L2 port ceiling;
                # its single-reader cold override is applied per task below.
                kv_copy_bps=kv_ingress_bps,
                smem_bps=per_sm_capacity("smem"),
                latency=hardware.l2_latency_ns * 1e-9,
                kv_latency=hardware.l2_latency_ns * 1e-9,
                # The bound packed D128 cp.async source uses128 producer
                # threads. Each live four-row group issues2LDGSTS.
                # The2SHFL per four tile rows
                # precede the live-row predicate even on a Q tail; all4
                # producer warps execute QueryEmpty sync, cpasync barrier
                # arrive and Q arrive.
                # This is an issue floor; async completion remains unbound.
                q_copy_issue_per_four_rows=(2 / per_sm_capacity("warp_issue")
                    if q_cp_async_issue_domain else 0.0),
                q_copy_fixed_issue=((bm // 2 + 12) / per_sm_capacity("warp_issue")
                    if q_cp_async_issue_domain else 0.0),
                # PagedKVManager uses eight threads/row,16 rows/producer
                # group and N/16 row pointers per thread. Each row needs
                # two pointer SHFL and d/64 vector LDGSTS instructions.
                # The bound CUDA12.9 GQA4 binary additionally forms each
                # of eight lane-column64 addresses with a low/high pair:
                # LEA+IMAD.X in first K, IADD3+IMAD.X in steady K/V and
                # last V. Page LDG below does not include those16 sites.
                # Other compiler/WG domains retain their partial budget.
                # Tails retain this full-panel instruction budget.
                # A single elected TMA producer issues one bulk load, not
                # the four-warp LDGSTS/pointer-shuffle sequence above.
                # Descriptor/barrier control remains a separate limitation.
                copy_issue=((1 if kv_tma else
                    ((2 + d // 64) * (bn // 16)
                     + (16 if compiled_wait0_before_online else 0)) * 4)
                    / per_sm_capacity("warp_issue")),
                page_lookup=(0.0 if kernel.attributes["kv_layout"] == "contiguous" else
                    hardware.l2_latency_ns * 1e-9
                    + (1 if kv_tma else 4) / per_sm_capacity("warp_issue")))
            cold_services = {**services,
                "kv_copy_bps": cold_kv_ingress_bps,
                "kv_latency": hardware.dram_latency_ns * 1e-9}
            cold_prefix_s = (hardware.dram_latency_ns - hardware.l2_latency_ns) * 1e-9
            if cold_prefix_s < 0:
                raise ValueError("FA3 cold DRAM latency must include its L2-ready prefix")
            first_read_prefix_costs = []
            q_suffix = tuple(reversed(tuple(itertools.accumulate(reversed(q_lens)))))
            kv_suffix = tuple(reversed(tuple(itertools.accumulate(reversed(kv_lens)))))
            source_tasks = _fa3_gqa_source_tasks(q_lens, kv_lens,
                    heads=kernel.shape["num_query_heads"], kv_heads=kernel.shape["num_kv_heads"],
                    block_m=bm, block_n=bn, head_dim=d, splits=work["splits"],
                    causal=kernel.attributes["causal"] and max(q_lens) > 1)
            first_wave_tasks = tuple(itertools.islice(source_tasks, active))
            first_wave_q_producers = sum(task.panel_count > 0 and work["splits"][task.batch_index] == 1
                                         for task in first_wave_tasks)
            for task_index, task in enumerate(itertools.chain(first_wave_tasks, source_tasks)):
                output_dtype_bytes = 4 if work["splits"][task.batch_index] > 1 else 2
                task_is_unique = unique_kv_by_request[task.batch_index]
                # Without a runtime K split this Q tile has one reader;
                # retain the existing cold-operand assumption on its own
                # ready dependency,not only in a launch-wide byte floor.
                q_is_unique = work["splits"][task.batch_index] == 1
                first_q_ready_s = 0.0
                if q_is_unique and task_index < active and task.panel_count:
                    first_q_ready_s = hardware.cold_async_copy_ready_s(
                        first_wave_q_producers, "tma" if q_tma else "cp_async",
                        task.packed_q_rows * d * 2)
                    initial_q_ready_s = max(initial_q_ready_s, first_q_ready_s)
                first_k_ready_s = 0.0
                if task_is_unique and task_index < active and task.panel_count:
                    # Only the first resident wave's initial K burst uses
                    # the finite-copy readiness service. Later panels use
                    # the existing stream capacity and two-slot overlap.
                    payload = (bn if full_kv_tiles else task.first_panel_kv_rows) * d * 2
                    first_k_ready_s = hardware.cold_async_copy_ready_s(
                        first_wave_producers, "tma" if kv_tma else "cp_async", payload)
                    initial_k_ready_s = max(initial_k_ready_s, first_k_ready_s)
                key = (task.panel_count, task.packed_q_rows, task.first_panel_kv_rows,
                       output_dtype_bytes, first_k_ready_s)
                # The first descending panel always has Seqlenk_mask=true.
                # Subsequent causal-mask panels stop at the first Q row's
                # boundary (mainloop_fwd_sm90_tma_gmma_ws.hpp). Empty splits
                # and noncausal requests do not acquire extra mask rounds.
                remaining_masks = 0
                if causal_mask and task.panel_count:
                    row_start = task.packed_q_start // ratio
                    mask_min = max(task.kv_block_start,
                        (row_start + kv_lens[task.batch_index]
                         - q_lens[task.batch_index]) // bn)
                    remaining_masks = max(0, task.kv_block_stop - 1 - mask_min)
                # BlockMN divides the tile-local N extent by the runtime
                # split count before checking whether this split is empty.
                # A compiled Split with dynamic count1 still executes it;
                # only a zero pre-split N extent bypasses the quotient.
                range_setup_s = 0.0
                if (kernel.launch.algorithm in (
                        _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM)
                        and task.kv_block_stop > 0):
                    range_setup_s = _fa3_split_range_setup_s(hardware, tensor_clock)
                q_extra, k_extra = _fa3_gqa_root_extra_rows(task,
                    block_m=bm, block_n=bn, query_tma=q_tma,
                    paged_tma=full_kv_tiles, contiguous_kv=kernel.attributes["kv_layout"] == "contiguous",
                    q_suffix=q_suffix, kv_suffix=kv_suffix, ratio=ratio)
                key += (remaining_masks, task_is_unique, range_setup_s, q_is_unique, first_q_ready_s,
                        q_extra, k_extra)
                if key not in memo:
                    task_services = cold_services if task_is_unique else services
                    if q_is_unique:
                        task_services = {**task_services,
                            "q_copy_bps": min(services["copy_bps"], cold_kv_ingress_bps),
                            "q_latency": hardware.dram_latency_ns * 1e-9}
                    memo[key] = _fa3_gqa_capacity_task_cost(*key[:3],
                        output_dtype_bytes=output_dtype_bytes,
                        first_k_ready_s=first_k_ready_s,
                        first_q_ready_s=first_q_ready_s,
                        causal_masked_panels=remaining_masks,
                        range_setup_s=range_setup_s,
                        q_extra_l2_rows=q_extra, first_kv_extra_l2_rows=k_extra,
                        **task_services)
                metadata_s = 0.0
                if task_index < active:
                    # Initial scheduler lookup precedes any Q/K request.
                    # Later work is prefetched by the producer while the
                    # current task runs; retain ideal overlap thereafter.
                    metadata_s = _fa3_gqa_initial_metadata_s(
                        task.batch_index, work["splits"][task.batch_index],
                        lpt=scheduler_lpt, hardware=hardware, clock_hz=tensor_clock,
                        split_template=kernel.launch.algorithm in (
                            _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM))
                    initial_metadata_s = max(initial_metadata_s, metadata_s)
                # A worker's first actual read carries one cold prefix for
                # a warm-reuse request. Unique-reader tasks already carry
                # DRAM readiness. Empty splits neither prime data nor consume
                # the worker's startup prefix; the scheduler attaches it to
                # the first subsequent nonempty task on that same worker.
                # K precedes Q in the same queue. Cold Q therefore covers
                # the first-K DRAM delta,except for any K notification not
                # hidden by Q's byte service. Keep that residual bounded by
                # the old delta;ignoring Q issue's extra hiding is conservative.
                prefix_s = cold_prefix_s
                if q_is_unique:
                    q_rate = min(services["copy_bps"], cold_kv_ingress_bps)
                    prefix_s = min(cold_prefix_s, max(0.0,
                        full_handoff_s - task.packed_q_rows * d * 2 / q_rate))
                first_read_prefix_costs.append(None if not task.panel_count else
                    0.0 if task_is_unique else prefix_s)
                costs.append(memo[key] + metadata_s)
        else:
            costs = [_fa3_gqa_task_stage_cost(panels, qk, pv, softmax)
                     for panels in work["blocks"]]
        cta_loads = _fa3_list_schedule(costs, active,
            first_read_prefix_costs=first_read_prefix_costs if capacity_schedule else None)
        chain_s = max(load for load, _ in cta_loads)
        l2_s = work["logical_l2_read_bytes"] / capacity("l2_cache", active)
        dram_s = (work["minimum_physical_read_bytes"] + work["write_bytes"]) / capacity("dram", active)
        # Capacity schedules already carry cold startup per initial worker
        # and DRAM dependencies for each unique-reader request. Byte-only
        # schedules retain the original launch-wide cold prefix.
        cold_ns = 0.0 if capacity_schedule else hardware.dram_latency_ns
        if cold_ns < 0:
            raise ValueError("FA3 cold DRAM latency must include its L2-ready prefix")
        # Only the CTA chain includes an L2-ready prefix already. Byte-only
        # bandwidth floors contain no ready latency: retain their full cold
        # DRAM prefix even when a new source schedule is available.
        execution_s = max(cold_ns * 1e-9 + chain_s,
                          hardware.dram_latency_ns * 1e-9 + max(l2_s, dram_s))
        groups = {"fa3_gqa_stage_chain": chain_s,
                  "fa3_gqa_l2_tile_reads": l2_s,
                  "fa3_gqa_compulsory_dram": dram_s}
        if kv_tma and not kernel.l2_execution_context.same_program_operand_replays:
            # Only compulsory live K/V (and physical paged tails), never Q or
            # outputs. Requests without Q issue no K/V reads. Normalize unique
            # bytes over active producers optimistically; no observed hit claim.
            kv_rows = sum((math.ceil(kv / bn) * bn if full_kv_tiles else kv)
                for q, kv in zip(q_lens, kv_lens) if q > 0)
            kv_read_bytes = kernel.shape["num_kv_heads"] * kv_rows * d * 4
            # A persistent launch includes idle/empty CTAs. Even an ideal
            # assignment cannot have more read producers than nonempty tasks;
            # use that optimistic bound,not the descriptor grid by itself.
            read_producers = min(active, sum(panels > 0 for panels in work["blocks"]))
            if kv_read_bytes and read_producers:
                finite_tma_s = hardware.tma_balanced_finite_read_completion_s(
                    threads_per_cta=kernel.launch.block_size, a_bytes=bn * d * 2,
                    b_bytes=bn * d * 2, slots_per_operand=2,
                    producers=read_producers, read_bytes=kv_read_bytes)
            if finite_tma_s is not None:
                execution_s = max(execution_s, finite_tma_s)
                groups["fa3_gqa_finite_tma_unique_read_completion"] = finite_tma_s
        if capacity_schedule:
            groups["fa3_gqa_initial_K_ready"] = initial_k_ready_s
            groups["fa3_gqa_initial_Q_ready"] = initial_q_ready_s
            groups["fa3_gqa_initial_metadata"] = initial_metadata_s
            if first_qk_cohort_s:
                groups["fa3_gqa_per_task_first_QK_cohort"] = first_qk_cohort_s
            groups["fa3_gqa_per_WG_first_panel_mask"] = first_mask_s
            groups["fa3_gqa_per_task_causal_loop_setup"] = causal_loop_setup_s
            if online_prefix_s is not None:
                groups["fa3_gqa_per_panel_online_before_PV_wait"] = online_prefix_s
            if notification_domain:
                groups["fa3_gqa_per_KV_full_notification"] = full_handoff_s
                groups["fa3_gqa_per_KV_empty_notification"] = empty_handoff_s
    parameter_limitations = ("fa3_gqa_combine_parameter_read_service_unbound",)
    if combine:
        parameter_s = _fa3_gqa_parameter_read_service(kernel, hardware, work, active)
        if parameter_s is not None:
            groups["fa3_gqa_parameter_read_service"] = parameter_s
            execution_s += parameter_s
            parameter_limitations = (
                "fa3_gqa_combine_static_argument_regions_at_independent_service",
                "fa3_gqa_combine_small_footprint_linear_transfer_and_overlap_approximation")
    else:
        parameter_s = _fa3_gqa_parameter_read_service(kernel, hardware, work, active)
        if parameter_s is None:
            parameter_limitations = ("fa3_gqa_parameter_reads_unbound_for_dtype_tile_or_graph_state",)
        else:
            groups["fa3_gqa_parameter_read_service"] = parameter_s
            execution_s += parameter_s
            parameter_limitations = (
                "fa3_gqa_parameter_reads_common_static_17_region_superset_approximation",
                "fa3_gqa_parameter_reads_increment_above_two_region_control_once_per_initial_cohort",
                "fa3_gqa_parameter_reads_blocking_control_aggregation_stage_overlap_unbound",
                "fa3_gqa_parameter_address_region_is_not_observed_cache_residency")
    dispatch_s, floor_s = _fa3_launch_timing(kernel, hardware)
    bottleneck = max(groups, key=groups.__getitem__)
    groups["cta_dispatch"] = dispatch_s
    groups["launch_span_floor"] = floor_s
    return KernelPrediction(
        t_total_s=max(dispatch_s + execution_s, floor_s),
        t_per_group=groups, t_launch_s=dispatch_s,
        bottleneck_group=bottleneck, t_per_resource=groups,
        diagnostics=PredictionDiagnostics(
            compute_time_s={}, memory=(), memory_access_classes=(),
            memory_arbitration=(), pipeline=None, critical_path=(bottleneck,),
            support_level="partial", limitations=(
                "fa3_gqa_source_stages_at_aggregate_hardware_capacity",
                "fa3_gqa_physical_l1_tma_reuse_unbound",
                "fa3_gqa_unique_operands_assumed_dram_cold",
                "fa3_gqa_cta_barrier_and_instruction_latency_unbound") + (
                    ("fa3_gqa_combine_all_launched_metadata_includes_dynamic_one_and_padding",
                     "fa3_gqa_combine_metadata_slot_bound_at_observed_compiled_resources",
                     "fa3_gqa_combine_metadata_L2_hit_scan_transfer_and_CTA_arbitration_approximate")
                    if combine and "fa3_gqa_combine_launched_metadata_slot_capacity" in groups
                    else ()) + (
                    ("fa3_gqa_source_valid_TMA_root_overreads_at_L2_lower_bound",
                     "fa3_gqa_TMA_extra_row_cache_residency_and_local_port_sharing_unbound")
                    if not combine and capacity_schedule else ()) + (
                    ("fa3_gqa_balanced_TMA_finite_unique_read_quiescent_start_approximation",
                     "fa3_gqa_finite_TMA_includes_fill_drain_composed_by_max",
                     "fa3_gqa_unique_KV_byte_cohort_normalization_address_order_partial")
                    if finite_tma_s is not None else ()) + (
                    ("fa3_gqa_finite_TMA_profile_unbound_for_source_cohort_or_packet",)
                    if not combine and kv_tma and finite_tma_s is None else ()) + parameter_limitations + (
                    ("fa3_gqa_prologue_first_coissued_WG_completion_from_independent_SS_batches",
                     "fa3_gqa_coarrival_and_conservative_marker_removal_transfer_is_partial",
                     "fa3_gqa_first_completion_is_not_a_full_cohort_barrier_or_Tensor_busy_cost")
                    if not combine and capacity_schedule and first_qk_cohort_s else ()) + (
                    ("fa3_gqa_causal_mask_loop_runtime_CP_quotient_after_first_panel",
                     "fa3_gqa_causal_loop_quotient_native_scheduling_and_warp_throughput_partial")
                    if not combine and capacity_schedule and causal_mask else ()) + (
                    ("fa3_gqa_online_work_split_at_static_PV_wait0_boundary",
                     "fa3_gqa_minimum_static_prefix_over_all_retained_regions",
                     "fa3_gqa_register_dependency_and_row_state_FMA_apportionment_is_aggregate")
                    if online_prefix_s is not None else ()) + (
                    ("fa3_gqa_SS_busy_competes_at_public_SI_and_independent_core_cycle_capacity",
                     "fa3_gqa_SS_capacity_positive_data_native_phase_and_arbitration_transfer_partial")
                    if ss_busy_capacity_domain else ()) + (
                    ("fa3_gqa_combine_shared_weights_then_weighted_O_at_local_capacity",
                     "fa3_gqa_combine_LSE_normalization_at_padded_split_and_row_capacity",
                     "fa3_gqa_combine_butterfly_work_and_replicated_RCP_log_lane_state_counted",
                     "fa3_gqa_combine_warp_reduction_dependency_latency_unbound",
                     "fa3_gqa_combine_weighted_O_shared_and_FMA_pipeline_is_an_aggregate_bound",
                     "fa3_gqa_combine_four_slot_O_copy_queue_at_existing_per_SM_L2_capacity",
                     "fa3_gqa_combine_invalid_prefetch_contention_and_uneven_row_tails_partial",
                     "fa3_gqa_combine_CTA_residency_and_other_integer_setup_remain_unbound")
                    if combine else ()) + (
                    ("fa3_gqa_combine_linearized_batch_scan_at_independent_warp_cycles",
                     "fa3_gqa_combine_last_batch_cohort_and_stage_composition_aggregate")
                    if combine and batch_scan_s else ()) + (
                    ("fa3_gqa_compiled_wait0_before_EX2_observed_online_prefix_aggregation_partial",
                     "fa3_gqa_compiled_dependency_transfer_is_not_a_new_PV_price")
                    if observed_wait0 is True else
                    ("fa3_gqa_compiled_mixed_or_post_EX2_wait_retains_source_overlap",)
                    if observed_wait0 is False else
                    ("fa3_gqa_compiler_wait_order_unbound",)) + (
                    ("fa3_gqa_public_Tensor_FLOPs_capacity_retains_its_original_normalization",
                     "fa3_gqa_core_cycle_services_at_independently_validated_mixed_clock",
                     "fa3_gqa_fixed_SI_services_keep_their_original_normalization",
                     "fa3_gqa_common_core_clock_is_not_a_whole_operator_time_multiplier")
                    if measured_tensor_clock else ()) + (
                    ("fa3_gqa_BF16_N128_QK_retirement_at_independent_active_clock",)
                    if measured_tensor_clock and work["block_n"] == 128 else ()) + (
                    ("fa3_gqa_typed_SS_RS_batch_retirement_isolated_approximation",
                     "fa3_gqa_packed_BF16_conversion_at_independent_capacity",
                     "fa3_gqa_conversion_feedback_removed_mixed_pipeline_arbitration_partial",
                     "fa3_gqa_finite_RS_enqueue_from_independent_retirement_suffix",
                     "fa3_gqa_SS_enqueue_floor_only_shared_queue_order_remains_partial")
                    if measured_batch_domain else ()) + (
                    ("fa3_gqa_cp_async_full_empty_notification_at_independent_cycles",
                     "fa3_gqa_isolated_notification_transfer_and_polling_phase_approximation")
                    if notification_domain else ()) + (
                    ("fa3_gqa_SS_N128_EX2_asymmetric_busy_capacity_from_independent_queued_primitives",
                     "fa3_gqa_EX2_only_sharing_RS_RCP_log_and_other_scalar_work_unchanged",
                     "fa3_gqa_SS_busy_and_EX2_source_phase_placement_is_a_coarse_approximation",
                     "fa3_gqa_delayed_arrival_transfer_and_native_instruction_interleaving_unverified")
                    if ss_ex2_sharing_domain else ()) + (
                    ("fa3_gqa_page_table_L2_hit_assumed_not_observed",
                     "fa3_gqa_tail_copy_uses_full_panel_issue_budget",
                     "fa3_gqa_full_tile_output_shared_staging_at_capacity",
                     "fa3_gqa_KV_L2_equal_share_uses_initial_nonempty_cohort_approximation",
                     "fa3_gqa_KV_shared_ingress_at_local_capacity_lower_bound",
                     "fa3_gqa_Q_and_global_output_keep_existing_per_SM_L2_port_ceiling",
                     "fa3_gqa_first_nonempty_read_carries_worker_cold_prefix_once",
                     "fa3_gqa_producer_consumer_warp_issue_contention_unbound",
                     "fa3_gqa_score_mask_compare_and_select_at_existing_ALU_and_issue_capacity",
                     "fa3_gqa_mask_then_reduction_is_a_coarse_source_stage_not_compiler_schedule",
                     "fa3_gqa_mask_row_index_setup_and_predicate_latency_unbound",
                     "fa3_gqa_initial_prepared_metadata_and_sequence_arrays_assumed_L2_hit",
                     "fa3_gqa_initial_scheduler_scan_and_divide_at_independent_warp_dependency_cycles",
                     "fa3_gqa_scheduler_primitive_composition_and_native_compiler_transfer_approximation",
                     "fa3_gqa_later_scheduler_metadata_ideally_prefetched_not_native_certified",
                     ("fa3_gqa_QK_and_PV_retirement_use_isolated_batch_approximations"
                      if measured_batch_domain else
                      "fa3_gqa_QK_retirement_uses_isolated_batch_approximation_PV_unbound"
                      if kernel.dtype.lower() == "bf16" and work["block_n"] == 128 else
                      "fa3_gqa_QK_ready_at_Tensor_capacity_lower_bound_PV_unbound"),
                     "fa3_gqa_producer_address_ALU_and_next_task_prefetch_unpriced")
                    if not combine and capacity_schedule else ()) + (
                    ("fa3_gqa_empty_tile_publishes_LSE_and_runtime_unsplit_zero_O_at_local_L2_capacity",
                     "fa3_gqa_empty_tile_scheduler_and_output_address_dependencies_partial",
                     "fa3_gqa_empty_tile_output_completion_at_serial_task_boundary_approximation")
                    if not combine and capacity_schedule and 0 in work["blocks"] else ()) + (
                    ("fa3_gqa_causal_CP1_forward_guards_at_generic_warp_issue_capacity",
                     "fa3_gqa_branch_dependency_and_concurrent_completion_latency_unbound")
                    if not combine and capacity_schedule and mask_guard_issue_work else ()) + (
                    ("fa3_gqa_unsplit_unique_Q_assumed_DRAM_cold_at_ready_dependency",
                     "fa3_gqa_Q_split_reuse_and_physical_cache_state_remain_assumptions",
                     "fa3_gqa_cold_Q_credits_first_K_startup_except_unhidden_notification",
                     "fa3_gqa_initial_unique_Q_cold_burst_at_independent_producer_capacity",
                     "fa3_gqa_Q_copy_live_byte_and_compiled_path_transfer_approximation")
                    if not combine and capacity_schedule and 1 in work["splits"] else ()) + (
                    ("fa3_gqa_compiled_Split_tile_range_quotient_at_independent_signed32_dependency_cycles",
                     "fa3_gqa_split_range_concurrent_producer_consumer_composition_partial")
                    if not combine and capacity_schedule and kernel.launch.algorithm in (
                        _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM)
                    else ()) + (
                    ("fa3_gqa_packed_D128_Q_cp_async_load_shuffle_and_control_issue_lower_bound",
                     "fa3_gqa_Q_cp_async_address_and_barrier_completion_latency_unbound")
                    if q_cp_async_issue_domain else ()) + (
                    ("fa3_gqa_unique_KV_panels_assumed_cold_at_each_ready_dependency",)
                    + ("fa3_gqa_unique_cold_KV_ready_path_reserves_DRAM_and_L2_capacity",
                       "fa3_gqa_unique_reader_KV_is_classified_per_request_in_mixed_batches",
                       "fa3_gqa_DRAM_equal_share_uses_initial_nonempty_cohort_approximation")
                    if not combine and capacity_schedule and unique_kv_panels else ()) + (
                    ("fa3_gqa_initial_K_cold_burst_at_independent_producer_capacity",
                     "fa3_gqa_initial_copy_address_order_and_cache_state_approximation")
                    if not combine and capacity_schedule and unique_kv_panels else ()) + (
                    ("fa3_gqa_nonN128_independent_register_chain_compiler_composition_approximation",
                     "fa3_gqa_nonN128_native_retirement_unbound")
                    if not combine and capacity_schedule and work["block_n"] != 128 else ()) + (
                    ("fa3_gqa_P_shared_single_partition_per_WG_waits_own_PV_retirement",
                     "fa3_gqa_P_shared_STSM_warp_fence_without_global_P_barrier",
                     "fa3_gqa_P_shared_read_write_at_local_and_aggregate_capacity_lower_bound")
                    if not combine and capacity_schedule and work["head_dim"] == 64
                        and work["block_m"] == 192 and work["block_n"] in (160, 192) else ()) + (
                    ("fa3_gqa_D256_N80_independent_register_chain_composition_approximation",
                     "fa3_gqa_D256_O_rescale_before_PV_without_WG_scheduler_token",
                     "fa3_gqa_D256_native_retirement_and_compiler_latency_unbound")
                    if not combine and capacity_schedule and work["head_dim"] == 256 else ()) + (
                    ("fa3_gqa_contiguous_TMA_KV_live_byte_ready_lower_bound",
                     "fa3_gqa_contiguous_TMA_root_extent_cross_request_reads_unbound")
                    if not combine and capacity_schedule
                        and kernel.attributes["kv_layout"] == "contiguous" else ()) + (
                    ("fa3_gqa_FP16_QK_ready_at_Tensor_capacity_lower_bound",
                     *( ("fa3_gqa_FP16_Q_cp_async_load_shuffle_and_control_issue_lower_bound",
                         "fa3_gqa_FP16_Q_address_and_barrier_completion_latency_unbound")
                        if q_cp_async_issue_domain else
                        ("fa3_gqa_FP16_Q_cp_async_issue_address_and_sync_unbound",)),
                     "fa3_gqa_FP16_packed_conversion_latency_and_numeric_validation_unbound",
                     "fa3_gqa_FP16_native_WGMMA_completion_and_compiler_schedule_unbound")
                    if not combine and capacity_schedule and kernel.dtype.lower() == "fp16" else ()) + (
                    ("fa3_gqa_cuda129_GQA4_prefill_wait0_before_exp_observed",
                     "fa3_gqa_cuda129_GQA4_KV_lane_address_issue_lower_bound",
                     "fa3_gqa_early_max_prefix_conservatively_aggregated_after_wait0",
                     "fa3_gqa_compiler_order_not_bound_to_normal_source_factory")
                    if compiled_wait0_before_online else ()) + (
                    ("fa3_gqa_paged_TMA_KV_reads_full_physical_tiles",
                     "fa3_gqa_TMA_elected_producer_descriptor_and_barrier_latency_unbound",
                     "fa3_gqa_TMA_Q_global_extent_and_transaction_bytes_unbound")
                    if not combine and kernel.launch.algorithm in (
                        _FA3_GQA_TMA_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM)
                    else ())))



def _fa3_list_schedule(tile_costs, workers: int, *, first_read_prefix_costs=None):
    """Hand tiles to the earliest-free CTA, retaining its first-read state.

    Optional prefix entries use None for a task without a read; zero means
    its existing cost already includes cold readiness. This is modeled
    progress state, not a cache-hit observation or a cross-request reuse claim.
    """

    if first_read_prefix_costs is not None:
        if (len(first_read_prefix_costs) != len(tile_costs)
                or any(value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0) for value in first_read_prefix_costs)):
            raise ValueError("FA3 first-read prefix costs must match tasks and be finite nonnegative")

    heap = []
    loads = [[0.0, 0] for _ in range(workers)]
    has_read = [False] * workers
    for index, cost in enumerate(tile_costs):
        if index < workers:
            worker = index
        else:
            _, worker = heapq.heappop(heap)
        if first_read_prefix_costs is not None:
            prefix = first_read_prefix_costs[index]
            if prefix is not None and not has_read[worker]:
                cost += prefix
                has_read[worker] = True
        loads[worker][0] += cost
        loads[worker][1] += 1
        heapq.heappush(heap, (loads[worker][0], worker))
    return loads


def _fa3_launch_timing(kernel: KernelSpec, hardware):
    """Order task submission before the body, with scheduler tail as a floor.

    This is the same hardware scheduler composition used by the common
    formula for source pipelines. The empty-task geometry tail can overlap
    the body; the independently measured submit prefix cannot. A CTA-count
    dispatch table does not replace that prefix when the table is absent.
    """
    dispatch_s = hardware.cta_dispatch_cost_ns(kernel.launch.grid_size) * 1e-9
    model = hardware.launch_latency_model
    if model is not None:
        if model["model"] not in ("isolated_empty_task_scheduler_v1",
                                  "isolated_nonempty_task_scheduler_v1"):
            raise ValueError("FA3 requires a decomposed task scheduler")
        dispatch_s += model["submit_ns"] * 1e-9
    floor_s = max(
        small_kernel_span_floor_s(hardware, kernel.launch.grid_size,
                                 kernel.variant, graph_replayed_launch(kernel)),
        hardware.empty_kernel_duration_for_launch(
            kernel.launch.grid_size, kernel.launch.block_size,
            kernel.launch.grid, kernel.launch.block) * 1e-9)
    return dispatch_s, floor_s


def _fa3_kernel_prediction(kernel: KernelSpec, hardware, execution_s: float,
                           groups: dict, bottleneck: str) -> KernelPrediction:
    dispatch_s, floor_s = _fa3_launch_timing(kernel, hardware)
    total_s = dispatch_s + execution_s
    if floor_s > total_s:
        total_s = floor_s
        bottleneck = "launch_span_floor"
    groups = dict(groups)
    groups["launch_span_floor"] = floor_s
    groups["cta_dispatch"] = dispatch_s
    # The FA3 template schedules source tiles directly; it does not build the
    # generic per-resource demand from which formula diagnostics are drawn.
    diagnostics = PredictionDiagnostics(
        compute_time_s={},
        memory=(),
        memory_access_classes=(),
        memory_arbitration=(),
        pipeline=None,
        critical_path=(bottleneck,),
        support_level="partial",
        limitations=("fa3_source_tile_schedule_without_resource_diagnostics",),
    )
    return KernelPrediction(
        t_total_s=total_s,
        t_per_group=groups,
        t_launch_s=dispatch_s,
        bottleneck_group=bottleneck,
        t_per_resource=groups,
        diagnostics=diagnostics,
    )


def _fa3_mla_prefill_capacity_costs(kernel, hardware, splits):
    """Expanded192/128 source tasks, with separate QK and V/O widths.

    Both source templates have two consumers, RS P and no WG scheduler
    token. Split forces PackGQA/kBlockH1 even for a dynamic-one request,
    so its Q ingress uses128-thread cp.async while K/V remain TMA. The
    unsplit equal-head template uses one elected TMA producer warp.
    """
    mechanisms = _fa3_mechanisms(hardware)
    clock = hardware.nominal_sm_freq_hz
    def per_sm(role):
        resource = hardware.resource_by_role(role)
        rate = resource.effective_capacity(1, sm_freq_hz=clock,
            dram_freq_hz=hardware.nominal_dram_freq_hz)
        return rate / hardware.num_compute_units if resource.scope == "shared" else rate
    registers = _fa3_gqa_register_services(hardware)
    split_template = kernel.launch.tiling.num_kv_splits > 1
    issue = per_sm("warp_issue")
    scale_exp = 64 * 128 / per_sm("compute_sfu") + (2 * 64 * 128 + 256) / per_sm("compute_fma")
    services = dict(qk=2 * 64 * 128 * 192 / per_sm("compute_tensor"),
        pv=2 * 64 * 128 * 128 / per_sm("compute_tensor"),
        first_online=registers["first_online"] + scale_exp,
        online=registers["online"] + scale_exp + 256 / per_sm("compute_sfu")
            + 3 * 256 / per_sm("compute_fma"),
        conversion=128 / issue,
        rescale=max(64 * 128 / per_sm("compute_fma"), 256 / issue),
        finalize=registers["quad_sum"] + 2 * 256 / per_sm("compute_sfu")
            + 3 * 256 / per_sm("compute_fma"),
        copy_bps=per_sm("l2_cache"), smem_bps=per_sm("smem"),
        latency=hardware.l2_latency_ns * 1e-9, page_lookup=0.,
        copy_issue=1 / issue, kv_tma=True, full_kv_tiles=False,
        warp_groups=2, head_dim=192, value_head_dim=128, block_n=128,
        qk_completion=mechanisms["wgmma_gemm_cycles"][("ss_n128_kmajor", 12, 1)] / clock,
        # Three16-byte copies/lane span192 BF16 columns. Pointer SHFL
        # runs over the full M128 tile; row predicates only skip copies.
        q_copy_issue_per_four_rows=3 / issue if split_template else 0.,
        q_copy_fixed_issue=(64 + 12) / issue if split_template else 1 / issue)
    q, kv = _fa3_lengths(kernel)
    heads = kernel.shape["num_query_heads"]
    tasks = _fa3_gqa_source_tasks(q, kv, heads=heads, kv_heads=heads,
        block_m=128, block_n=128, head_dim=192, value_head_dim=128,
        splits=splits, causal=kernel.attributes["causal"])
    memo, costs = {}, []
    for task in tasks:
        if not task.panel_count:
            costs.append(0.)
            continue
        output_width = 4 if splits[task.batch_index] > 1 else 2
        key = (task.panel_count, task.packed_q_rows, task.first_panel_kv_rows, output_width)
        if key not in memo:
            memo[key] = _fa3_gqa_capacity_task_cost(*key[:3],
                output_dtype_bytes=output_width, **services)
        costs.append(memo[key])
    return costs


def _fa3_mla_prefill_capacity_diagnostics(prediction):
    return dataclasses.replace(prediction, diagnostics=dataclasses.replace(
        prediction.diagnostics, limitations=prediction.diagnostics.limitations + (
            "fa3_mla_prefill_source_two_slot_QK192_VO128_capacity_schedule",
            "fa3_mla_prefill_O_rescale_before_PV_without_WG_scheduler_token",
            "fa3_mla_prefill_register_reduction_composition_and_QK_retirement_approximation",
            "fa3_mla_prefill_PV_ready_at_Tensor_capacity_lower_bound",
            "fa3_mla_prefill_contiguous_TMA_live_byte_tail_and_cold_input_assumptions",
            "fa3_mla_prefill_producer_control_barrier_and_native_compiler_latency_unbound")))


def _fa3_prefill_prediction(kernel: KernelSpec, hardware) -> KernelPrediction:
    tile = kernel.launch.tiling.tile_shape
    if (tile["q"], tile["kv"], tile["qk_head"], tile["value_head"]) != (
            128, 128, 192, 128):
        raise NotImplementedError("FA3 prefill template covers 128x128 hdim 192/128")
    work = current_fa3_prefill_source_work(kernel)
    if work is None:
        raise NotImplementedError("FA3 prefill requires its validated Q/K/V and O/LSE source boundary")
    q, _ = _fa3_lengths(kernel)
    costs = _fa3_mla_prefill_capacity_costs(kernel, hardware, (1,) * len(q))
    workers = kernel.launch.grid[0]
    active = min(workers, len(costs))
    def capacity(role):
        return hardware.resource_by_role(role).effective_capacity(active,
            sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz)
    critical_s = max(load for load, _ in _fa3_list_schedule(costs, workers))
    # Both O and FP32 LSE are unconditional outputs of this source kernel.
    # Returning LSE to a caller changes the interface, not its device work.
    # Their payload shares L2 with Q/K/V; the aggregate capacity check does
    # not serialize every CTA's stores after the last CTA's matrix work.
    payload_bytes = sum(work['totals'][role] for role in ('query', 'key', 'value', 'output', 'lse'))
    payload_s = payload_bytes / capacity("l2_cache")
    cold_s = (hardware.dram_latency_ns - hardware.l2_latency_ns) * 1e-9
    if cold_s < 0:
        raise ValueError("FA3 cold DRAM latency must include its L2-ready prefix")
    dram_s = (work['projected_read_bytes'] + work['totals']['output']
              + work['totals']['lse']) / capacity("dram")
    execution_s = max(cold_s + critical_s,
                      hardware.dram_latency_ns * 1e-9 + max(payload_s, dram_s))
    groups = {
        "fa3_prefill_stage_chain": critical_s,
        "fa3_prefill_payload_l2": payload_s,
        "fa3_prefill_compulsory_dram": dram_s,
    }
    bottleneck = max(groups, key=groups.__getitem__)
    return _fa3_mla_prefill_capacity_diagnostics(_fa3_kernel_prediction(
        kernel, hardware, execution_s, groups, bottleneck))


def _fa3_split_prefill_work(kernel: KernelSpec, hardware):
    """Expanded192/128 context: split-local tasks and actual O/LSE bytes."""
    tile = kernel.launch.tiling
    if (kernel.launch.algorithm != _FA3_PREFILL_SPLIT_ALGORITHM
            or kernel.attributes["causal"] or kernel.dtype != "bf16"
            or (kernel.shape["qk_head_dim"], kernel.shape["v_head_dim"],
                kernel.shape["kv_lora_rank"]) != (192, 128, 512)
            or dict(tile.tile_shape) != {"q": 128, "kv": 128, "qk_head": 192, "value_head": 128}
            or tile.num_kv_splits <= 1 or tile.pipeline_depth != 2
            or kernel.launch.block != (384, 1, 1)
            or kernel.launch.grid != (int(hardware.num_compute_units), 1, 1)):
        raise ValueError("FA3 split prefill differs from its owned192/128 context source")
    q_lens, kv_lens = _fa3_lengths(kernel)
    heads = kernel.shape["num_query_heads"]
    splits = _fa3_request_tile_splits(
        q_lens, kv_lens, tile.num_kv_splits, int(hardware.num_compute_units),
        block_m=128, block_n=128, ratio=1, kv_heads=heads)
    tasks, logical_reads = [], 0
    for q, kv, split in zip(q_lens, kv_lens, splits):
        num_m, num_n = math.ceil(q / 128), math.ceil(kv / 128)
        per_split = math.ceil(num_n / split)
        # Noncausal scheduler: M varies fastest, then split, then head.
        for _ in range(heads):
            for s in range(split):
                panels = max(0, min((s + 1) * per_split, num_n) - s * per_split)
                valid_kv = max(0, min(kv, (s + 1) * per_split * 128) - s * per_split * 128)
                for m in range(num_m):
                    tasks.append(panels)
                    if panels:
                        logical_reads += (min(128, q - m * 128) * 192 * 2
                                          + valid_kv * (192 + 128) * 2)
    partial_rows = heads * sum(q * s for q, s in zip(q_lens, splits) if s > 1)
    direct_rows = heads * sum(q for q, s in zip(q_lens, splits) if s == 1)
    partial_work = _fa3_gqa_partial_source_work(
        q_lens, kv_lens, heads=heads, kv_heads=heads,
        block_m=128, block_n=128, splits=splits, causal=False)
    partial_output_bytes = partial_work["materialized_partial_output_rows"] * 128 * 4
    partial_lse_bytes = partial_rows * 4
    partial_write_bytes = partial_output_bytes + partial_lse_bytes
    direct_write_bytes = direct_rows * (128 * 2 + 4)
    scores = sum(tasks) * 128 * 128
    return {"blocks": tuple(tasks), "splits": splits,
            **partial_work,
            "tensor_flops": 2 * scores * (192 + 128), "scores": scores,
            "read_bytes": heads * (sum(q_lens) * 192 + sum(kv_lens) * (192 + 128)) * 2,
            "logical_l2_read_bytes": logical_reads,
            "write_bytes": partial_write_bytes + direct_write_bytes,
            "partial_write_bytes": partial_write_bytes,
            "partial_output_bytes": partial_output_bytes,
            "partial_lse_bytes": partial_lse_bytes,
            "direct_write_bytes": direct_write_bytes}


def _fa3_split_prefill_prediction(kernel: KernelSpec, hardware) -> KernelPrediction:
    work = _fa3_split_prefill_work(kernel, hardware)
    costs = _fa3_mla_prefill_capacity_costs(kernel, hardware, work["splits"])
    active = min(kernel.launch.grid[0], len(work["blocks"]))
    def capacity(role):
        return hardware.resource_by_role(role).effective_capacity(
            active, sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz)
    critical_s = max(load for load, _ in _fa3_list_schedule(costs, kernel.launch.grid[0]))
    groups = {"fa3_split_prefill_critical_cta": critical_s,
              "fa3_split_prefill_l2": (work["logical_l2_read_bytes"] + work["write_bytes"]) / capacity("l2_cache"),
              "fa3_split_prefill_cold_dram": (work["read_bytes"] + work["write_bytes"]) / capacity("dram")}
    bottleneck = max(groups, key=groups.__getitem__)
    cold_s = (hardware.dram_latency_ns - hardware.l2_latency_ns) * 1e-9
    if cold_s < 0:
        raise ValueError("FA3 cold DRAM latency must include its L2-ready prefix")
    execution_s = max(cold_s + critical_s, hardware.dram_latency_ns * 1e-9
        + max(groups["fa3_split_prefill_l2"], groups["fa3_split_prefill_cold_dram"]))
    prediction = _fa3_mla_prefill_capacity_diagnostics(
        _fa3_kernel_prediction(kernel, hardware, execution_s, groups, bottleneck))
    return dataclasses.replace(prediction, diagnostics=dataclasses.replace(
        prediction.diagnostics, limitations=prediction.diagnostics.limitations + (
            "fa3_split_prefill_scratch_aggregate_without_shared_native_bindings",
            "fa3_split_prefill_cta_memory_and_barrier_overlap_unbound")))


def _fa3_decode_chain_cycles(mechanisms) -> float:
    wgmma = mechanisms["wgmma_gemm_cycles"]
    return (wgmma[("ss_n64_kmajor", 36, 1)]
            + mechanisms["softmax_step_cycles"][("16x64", 1)]
            + wgmma[("rs_n256_mnmajor", 4, 2)])


def _fa3_decode_source_tasks(kernel: KernelSpec, hardware, *, program=None):
    """Shared packed-Q geometry for the legacy absorbed-MLA scheduler.

    Public memory bindings still establish the request extents. The source
    scheduler then partitions each M tile and dynamic K split separately;
    a128-head request is two64-row tasks, not one double-width task.
    """
    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    from src.kernel.memory_flash_attention import logical_mla_request_extents
    from src.kernel.memory_path_solver import logical_interface_source_bytes

    if program is None:
        program = logical_interface_memory_access_program(kernel, hardware)
    logical_interface_source_bytes(program, expected_kernel=kernel)
    q_lens, kv_lens, heads = logical_mla_request_extents(program)
    tile = kernel.launch.tiling
    bm, bn = tile.tile_shape["q"], tile.tile_shape["kv"]
    splits = _fa3_decode_splits(q_lens, kv_lens, heads,
                               tile.num_kv_splits, int(hardware.num_compute_units))
    causal = kernel.attributes["causal"] and max(q_lens) > 1
    return tuple(_fa3_gqa_source_tasks(q_lens, kv_lens,
        heads=heads, kv_heads=1, block_m=bm, block_n=bn,
        head_dim=tile.tile_shape["qk_head"], value_head_dim=tile.tile_shape["value_head"],
        splits=splits, causal=causal))


def _fa3_decode_tile_blocks(kernel: KernelSpec, hardware, *, program=None):
    """KV panels per real packed-Q task, including multiple M tiles."""
    return tuple(task.panel_count for task in
        _fa3_decode_source_tasks(kernel, hardware, program=program))


def _fa3_sparse_mla_work(kernel: KernelSpec, hardware):
    """Page1 FA3: capacity selects static splits, used counts price work.

    Selected physical indices remain caller inputs. Read bytes below are
    an explicit cold-per-query projection, never observed unique/cache-hit
    bytes or a claim that the sparse cache was materialized densely.
    """
    combine = kernel.launch.algorithm == _FA3_SPARSE_MLA_COMBINE_030_ALGORITHM
    tile = kernel.launch.tiling
    dimensions = ({"rows", "value_head", "kv_page", "kv_capacity"} if combine else
                  {"q", "kv", "qk_head", "value_head", "kv_page", "kv_capacity"})
    if (not hardware.supports_source_contract("vllm_0_30_fa3_sparse_mla") or kernel.dtype != "bf16"
            or kernel.launch.algorithm not in (
                _FA3_SPARSE_MLA_030_ALGORITHM, _FA3_SPARSE_MLA_COMBINE_030_ALGORITHM)
            or set(tile.tile_shape) != dimensions or tile.tile_shape["kv_page"] != 1):
        raise ValueError("FA3 sparse MLA requires its page1 BF16 source program")
    q, used = _fa3_lengths(kernel)
    heads, batch = kernel.shape["num_query_heads"], kernel.shape["batch"]
    if (any(value != 1 for value in q) or len(q) != batch
            or min(used) < 0 or max(used) > tile.tile_shape["kv_capacity"]
            or kernel.attributes["causal"] is not True or kernel.attributes["kv_layout"] != "mla"
            or kernel.attributes["kv_dist"] not in ("uniform", "explicit")
            or kernel.attributes["kv_dist"] == "uniform" and len(set(used)) != 1
            or kernel.attributes["kv_dist"] == "explicit" and (
                "source_program" not in kernel.attributes
                or kernel.attributes["source_program"] != "vllm_0_30_fa3_sparse_mla")
            or any(name in kernel.attributes for name in (
                "graph_logical_tokens", "graph_model_input_tokens", "graph_runtime_mode",
                "model_input_shape", "model_input_q_lens", "model_input_kv_lens"))):
        raise ValueError("FA3 sparse MLA requires truthful nonnegative per-token used lengths")
    if (not combine and (kernel.attributes["full_cuda_graphs"] is not False
                         or kernel.attributes["max_cudagraph_capture_size"] != 0)):
        raise ValueError("FA3 sparse MLA graph/zero-row contract is unbound")
    static = tile.num_kv_splits
    if not any(used) and static > 1:
        raise NotImplementedError("FA3 sparse MLA all-zero dynamic prepare has a source zero divisor")
    if (kernel.launch.parameters != {
            "unpadded_flops": 0.0, "mma_smem_bytes_per_flop": 0.0,
            "split_scratch_dram_bytes_per_element": 4.0 if static > 1 else 0.0,
            "page_table_bytes_per_kv_panel": 0.0}
            or kernel.launch.source_program is not None):
        raise ValueError("FA3 sparse MLA source demand must own its index and scratch work")
    splits = _fa3_request_tile_splits(q, used, static, int(hardware.num_compute_units),
        block_m=64, block_n=64, ratio=heads, kv_heads=1)
    # Equal-width split ranges can leave trailing splits empty. store_zero
    # writes -inf LSE for those rows without initializing partial O.
    nonempty_splits = tuple(math.ceil(math.ceil(length / 64) /
        math.ceil(math.ceil(length / 64) / count)) if length else 0 for length, count in zip(used, splits))
    partial_rows = heads * sum(count for count in splits if count > 1)
    valid_partial_rows = heads * sum(live for live, count in zip(nonempty_splits, splits) if count > 1)
    split_rows = heads * sum(count > 1 for count in splits)
    if combine:
        expected_grid = ((math.ceil(heads / 8), 4, batch)
                         if 2 * math.ceil(batch / 8) >= batch else
                         (math.ceil(batch * heads / 8) + batch, 4, 1))
        if (kernel.variant != "combine" or kernel.shape["head_dim"] != 512
                or kernel.shape["num_kv_heads"] != 1 or static <= 1
                or tile.tile_shape["rows"] != 8 or tile.tile_shape["value_head"] != 128
                or tile.pipeline_depth != 4 or kernel.launch.block != (256, 1, 1)
                or kernel.launch.grid != expected_grid):
            raise ValueError("FA3 sparse MLA combine differs from source")
        output_bytes, lse_bytes = split_rows * 512 * 2, split_rows * 4
        # First three O stages are prefetched before LSE validity is known;
        # subsequent loads stop at the last finite LSE split.
        partial_output = heads * 512 * 4 * sum(
            max(live, min(count, 3)) for live, count in zip(nonempty_splits, splits) if count > 1)
        # Four value-column CTAs read LSE; only column0 writes final LSE.
        partial_lse_reads = partial_rows * 4 * 4
        return {"blocks": (), "splits": splits, "tensor_flops": 0,
                # Two arithmetic operations per weighted partial O value;
                # LSE scalar work is a mathematical projection, not SASS.
                "scalar_flops": 2 * valid_partial_rows * 512 + 4 * partial_rows,
                "sfu_ops": 4 * partial_rows,
                "scores": partial_rows, "read_bytes": partial_output + partial_lse_reads,
                "logical_l2_read_bytes": partial_output + partial_lse_reads,
                "write_bytes": output_bytes + lse_bytes + 4,
                "output_bytes": output_bytes, "lse_bytes": lse_bytes,
                "partial_output_bytes": partial_output, "partial_lse_read_bytes": partial_lse_reads,
                "valid_partial_output_bytes": valid_partial_rows * 512 * 4,
                "nonempty_splits": nonempty_splits,
                "semaphore_reset_bytes": 4,
                "active_ctas": math.ceil(heads / 8) * 4 * sum(count > 1 for count in splits)}
    if (kernel.variant != "mla_decode"
            or (kernel.shape["qk_head_dim"], kernel.shape["v_head_dim"],
                kernel.shape["kv_lora_rank"]) != (576, 512, 512)
            or any(tile.tile_shape[name] != value for name, value in (
                ("q", 64), ("kv", 64), ("qk_head", 64), ("value_head", 512)))
            or tile.pipeline_depth != 2 or kernel.launch.block != (384, 1, 1)
            or kernel.launch.grid != (int(hardware.num_compute_units), 1, 1)):
        raise ValueError("FA3 sparse MLA main differs from source")
    blocks, logical_reads, lookup_bytes = [], 0, 0
    for length, count in zip(used, splits):
        num_n = math.ceil(length / 64)
        per_split = math.ceil(num_n / count)
        for s in range(count):
            panels = max(0, min((s + 1) * per_split, num_n) - s * per_split)
            valid_k = max(0, min(length, (s + 1) * per_split * 64) - s * per_split * 64)
            for m in range(math.ceil(heads / 64)):
                blocks.append(panels)
                if panels:
                    logical_reads += min(64, heads - 64 * m) * 576 * 2 + valid_k * 576 * 2
                    lookup_bytes += valid_k * 4
    partial_write = valid_partial_rows * 512 * 4 + partial_rows * 4
    direct_write = (batch * heads - split_rows) * (512 * 2 + 4)
    scores = sum(blocks) * 64 * 64
    return {"blocks": tuple(blocks), "splits": splits,
            "tensor_flops": 2 * scores * (576 + 512), "scores": scores,
            "scalar_flops": 4 * scores, "sfu_ops": scores,
            "read_bytes": sum(length > 0 for length in used) * heads * 576 * 2 + sum(used) * (576 * 2 + 4),
            "logical_l2_read_bytes": logical_reads + lookup_bytes,
            "write_bytes": partial_write + direct_write,
            "partial_output_bytes": valid_partial_rows * 512 * 4,
            "partial_lse_bytes": partial_rows * 4,
            "partial_write_bytes": partial_write, "direct_write_bytes": direct_write,
            "page_lookup_bytes": lookup_bytes, "nonempty_splits": nonempty_splits,
            "task_count": len(blocks), "empty_tasks": blocks.count(0),
            "active_ctas": sum(value > 0 for value in blocks)}


def _fa3_sparse_mla_prediction(kernel: KernelSpec, hardware) -> KernelPrediction:
    work = _fa3_sparse_mla_work(kernel, hardware)
    clock = hardware.nominal_sm_freq_hz
    active = min(max(work["active_ctas"], 1), int(hardware.num_compute_units))
    def capacity(role):
        return hardware.resource_by_role(role).effective_capacity(
            active, sm_freq_hz=clock, dram_freq_hz=hardware.nominal_dram_freq_hz)
    groups = {"fa3_sparse_l2_projection": (work["logical_l2_read_bytes"] + work["write_bytes"]) / capacity("l2_cache")}
    if kernel.variant == "combine":
        rounds = (2 + math.ceil(max(max(work["nonempty_splits"]) - 3, 0) / 3)
                  if work["active_ctas"] else 0)
        waves = math.ceil(work["active_ctas"] / hardware.num_compute_units)
        groups["fa3_sparse_combine_rounds"] = waves * rounds * hardware.l2_latency_ns * 1e-9
        groups["fa3_sparse_combine_scalar"] = work["scalar_flops"] / capacity("compute_fma")
        groups["fa3_sparse_combine_sfu"] = work["sfu_ops"] / capacity("compute_sfu")
        latency = 0.0
    else:
        mechanisms = _fa3_mechanisms(hardware)
        # Matching matrix/softmax fragments only. In particular, do NOT
        # use the independently measured page16 decode_ring for page1.
        chain = _fa3_decode_chain_cycles(mechanisms) / mechanisms["measured_sm_clock_hz"]
        loads = _fa3_list_schedule([n * chain for n in work["blocks"]], kernel.launch.grid[0])
        groups["fa3_sparse_cta_chain"] = max(load for load, _ in loads)
        groups["fa3_sparse_cold_read_projection"] = (work["read_bytes"] + work["write_bytes"]) / capacity("dram")
        latency = hardware.dram_latency_ns * 1e-9 if work["read_bytes"] else 0.0
    bottleneck = max(groups, key=groups.__getitem__)
    prediction = _fa3_kernel_prediction(kernel, hardware, latency + groups[bottleneck], groups, bottleneck)
    return dataclasses.replace(prediction, diagnostics=dataclasses.replace(
        prediction.diagnostics, limitations=prediction.diagnostics.limitations + (
            "fa3_sparse_global_indices_alias_and_readiness_unbound",
            "fa3_sparse_page1_integer_and_memory_overlap_unbound",
            "fa3_sparse_selected_cache_reuse_unobserved",
            "fa3_sparse_cta_residency_and_overlap_unbound")))


def _fa3_decode_prediction(kernel: KernelSpec, hardware) -> KernelPrediction:
    mechanisms = _fa3_mechanisms(hardware)
    clock_hz = mechanisms["measured_sm_clock_hz"]
    tile = kernel.launch.tiling.tile_shape
    if (tile["q"], tile["kv"], tile["qk_head"], tile["value_head"]) != (
            64, 64, 64, 512):
        raise NotImplementedError("FA3 decode template covers 64x64 hdim 64/512")
    chain = _fa3_decode_chain_cycles(mechanisms)
    if round(chain) != mechanisms["decode_ring_hold_cycles"]:
        raise ValueError(
            "FA3 decode consumer chain no longer matches the hold at which "
            "the paged ring was measured")
    workers = kernel.launch.grid[0]
    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    program=logical_interface_memory_access_program(kernel,hardware)
    tasks = tuple(task for task in _fa3_decode_source_tasks(kernel, hardware, program=program)
                  if task.panel_count > 0)
    active = min(workers, len(tasks))
    ring = mechanisms["decode_ring_by_grid"]
    first_full, ii = ring[active]
    q_lens, kv_lens = _fa3_lengths(kernel)
    splits = _fa3_decode_splits(q_lens, kv_lens, kernel.shape["num_query_heads"],
        kernel.launch.tiling.num_kv_splits, int(hardware.num_compute_units))
    def capacity(role):
        return hardware.resource_by_role(role).effective_capacity(active,
            sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz)
    l2_share = capacity("l2_cache") / active
    shared_resource = hardware.resource_by_role("smem")
    shared_bps = shared_resource.effective_capacity(1,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz)
    if shared_resource.scope == "shared":
        shared_bps /= hardware.num_compute_units
    # The measured KV ring contains no Q/Qv, metadata or output. Keep its
    # first-full/II unchanged and add only these missing source boundaries.
    metadata_s = max(_fa3_gqa_initial_metadata_s(t.batch_index, splits[t.batch_index],
        lpt=kernel.attributes["causal"] and max(q_lens) > 1, hardware=hardware,
        split_template=kernel.launch.tiling.num_kv_splits > 1)
        for t in tasks[:active])
    # The legacy BlockMN uses the same runtime signed32 split quotient as
    # GQA. Its initial range follows sequence metadata and precedes Q/K
    # ingress. The independent KV ring contains neither metadata nor range
    # control. Producer
    # and consumers compute it concurrently, even for dynamic count1.
    # Later tasks retain ideal range/metadata prefetch overlap; do not
    # serialize a new quotient charge across every persistent task.
    initial_range_s = (_fa3_split_range_setup_s(hardware, hardware.nominal_sm_freq_hz)
        if kernel.launch.tiling.num_kv_splits > 1 else 0.0)
    query_bytes = max(t.packed_q_rows for t in tasks[:active]) * kernel.shape["qk_head_dim"] * 2
    kv_panel_bytes = tile["kv"] * kernel.shape["qk_head_dim"] * 2
    first_input_s = max(first_full / clock_hz,
        hardware.l2_latency_ns * 1e-9
        + (query_bytes + kv_panel_bytes) / min(l2_share, shared_bps))
    staging_bytes = tile["q"] * tile["value_head"] * 2
    # Varlen always stages BF16 O. Final O reloads it; split partials come
    # from FP32 registers. The ring's synthetic consumer does neither.
    staging = [staging_bytes / shared_bps * (2 if splits[t.batch_index] == 1 else 1)
               for t in tasks]
    # The ring's consumer hold covers the recurrent softmax/P conversion
    # and O rescale, not final row normalization or output conversion.
    # QK WG0 produces the row scales; both output WGs own256 value columns.
    # Keep a coarse serial output tail per task, without changing the hold
    # or transferring a new II from a differently composed consumer.
    final_row = _fa3_final_row_services(256,
        _fa3_gqa_register_services(hardware)["quad_sum"],
        lambda role: capacity(role) / active)
    final_output_s = (final_row["finalize"]
        + 2 * (final_row["rescale"] + 256 / (capacity("warp_issue") / active)))
    costs = [t.panel_count * ii / clock_hz + final_output_s + store
             for t, store in zip(tasks, staging)]
    critical_s = max(load for load, count in _fa3_list_schedule(costs, workers) if count > 0)
    output_bytes = max(t.packed_q_rows * (tile["value_head"]
        * (4 if splits[t.batch_index] > 1 else 2) + 4) for t in tasks)
    # Earlier global stores may overlap subsequent work. Retain one final
    # drain, rather than imposing a global-memory completion per task.
    output_s = output_bytes / l2_share + hardware.l2_latency_ns * 1e-9
    execution_s = (metadata_s + initial_range_s + first_input_s - ii / clock_hz
                   + chain / clock_hz + critical_s + output_s)
    critical_blocks = max(load / ii for load, count in _fa3_list_schedule(
        [t.panel_count * ii for t in tasks], workers) if count > 0)
    groups = {
        "fa3_decode_first_block": first_input_s,
        "fa3_decode_ring_ii": (critical_blocks - 1) * ii / clock_hz,
        "fa3_decode_consumer_chain": critical_blocks * chain / clock_hz,
        "fa3_decode_initial_metadata": metadata_s,
        "fa3_decode_initial_split_range": initial_range_s,
        "fa3_decode_output_finalize": max(load for load, _ in _fa3_list_schedule(
            [final_output_s] * len(tasks), workers)),
        "fa3_decode_output_staging": max(load for load, _ in _fa3_list_schedule(staging, workers)),
        "fa3_decode_output_drain": output_s,
        "fa3_decode_stage_chain": execution_s,
    }
    prediction = _fa3_kernel_prediction(kernel, hardware, execution_s, groups,
        "fa3_decode_stage_chain")
    return dataclasses.replace(prediction, diagnostics=dataclasses.replace(
        prediction.diagnostics, limitations=prediction.diagnostics.limitations + (
            "fa3_decode_metadata_and_Q_KV_startup_at_existing_capacity_bounds",
            "fa3_decode_per_task_output_staging_and_one_final_global_drain",
            "fa3_decode_final_row_quad_sum_at_existing_two_row_service",
            "fa3_decode_final_O_rescale_and_packed_conversion_at_capacity",
            "fa3_decode_output_tail_serial_composition_and_last_WG_overlap_partial",
            "fa3_decode_Q_and_output_L2_residency_and_initial_cohort_share_assumed",
            "fa3_decode_initial_split_range_at_independent_signed32_quotient_latency",
            "fa3_decode_later_split_range_prefetch_overlap_idealized",
            "fa3_decode_later_metadata_Q_prefetch_and_task_boundary_overlap_partial",
            "fa3_mla_parameter_pipe_composition_unbound_no_GQA_region_price_transfer")))


def _fa3_combine_prediction(kernel: KernelSpec, hardware) -> KernelPrediction:
    tile = kernel.launch.tiling.tile_shape
    work=current_fa3_combine_source_work(kernel)
    if work is None:
        work=fa3_combine_prediction_work(kernel,hardware)
    active,max_splits=work['active_ctas'],work['max_splits']
    static_splits = kernel.launch.tiling.num_kv_splits
    compiled_splits = 1 << max(5, (static_splits - 1).bit_length())
    resident = _fa3_combine_resident_ctas(kernel, hardware, compiled_splits)
    metadata_critical_s, metadata_capacity_s = _fa3_combine_metadata_residency_s(
        kernel, hardware, resident)
    metadata_groups = {"fa3_combine_launched_metadata_chain": metadata_critical_s,
                       "fa3_combine_launched_metadata_slot_capacity": metadata_capacity_s}
    metadata_limits = (
        "fa3_combine_all_launched_CTA_metadata_includes_dynamic_one_and_padding",
        "fa3_combine_metadata_slot_bound_keeps_existing_L2_hit_and_scan_transfer_assumptions")
    if active == 0:
        groups = {"fa3_combine_rounds": 0.0, **metadata_groups}
        bottleneck = max(metadata_groups, key=metadata_groups.__getitem__)
        prediction = _fa3_kernel_prediction(kernel, hardware, metadata_groups[bottleneck], groups, bottleneck)
        return dataclasses.replace(prediction, diagnostics=dataclasses.replace(
            prediction.diagnostics, limitations=prediction.diagnostics.limitations + metadata_limits))
    waves = math.ceil(active / (hardware.num_compute_units * resident))
    active_sms = min(active, int(hardware.num_compute_units))
    def capacity(role):
        return hardware.resource_by_role(role).effective_capacity(active_sms,
            sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz)
    def per_sm_capacity(role):
        resource = hardware.resource_by_role(role)
        rate = resource.effective_capacity(1, sm_freq_hz=hardware.nominal_sm_freq_hz,
            dram_freq_hz=hardware.nominal_dram_freq_hz)
        return rate / hardware.num_compute_units if resource.scope == "shared" else rate

    rows, columns = tile["rows"], tile["value_head"]
    row_span = min(128, rows & -rows)
    reduction_lanes = _FA3_COMBINE_THREADS // row_span
    latency = hardware.l2_latency_ns * 1e-9
    local = _fa3_combine_local_services(rows=rows, columns=columns,
        compiled_splits=compiled_splits, reduction_lanes=reduction_lanes,
        valid_splits=max_splits, latency_s=latency, per_sm_capacity=per_sm_capacity)
    # The legacy and GQA varlen combine headers are byte-identical. Reuse
    # the same coordinate-scan dependency; do not borrow GQA parameter-pipe
    # service or extend the empty-body envelope to unknown control work.
    metadata_s, batch_scan_s = _fa3_varlen_combine_metadata_services(
        kernel.attributes["q_lens"], rows, hardware)
    # Retain the existing slot-residency bound.
    # A slot remains occupied through normalization and the final O store;
    # the former byte-only rounds omitted those mandatory source stages.
    cta_s = (metadata_s + local["first_partial"] + local["normalization"]
        + local["weight_shared_exchange"] + local["post_normalization"] + local["output_ready"])
    totals = work["totals"]
    # O_tail is a declared full-split projection when max-valid READ values
    # are unknown. Do not turn that projection into observed valid scales.
    partial_o = totals["O_preload"] + totals["O_tail"]
    payload = (partial_o + totals["partial_lse"] + totals["output"]
        + totals["lse"] + totals["semaphore_reset"])
    normalization_rows = active * rows
    normalization_flops = 4 * compiled_splits + 2 * reduction_lanes * (reduction_lanes.bit_length() - 1)
    scalar_s = ((2 * (partial_o // 4) + normalization_rows * normalization_flops)
        / capacity("compute_fma")
        + normalization_rows * (compiled_splits + 2 * reduction_lanes) / capacity("compute_sfu"))
    groups = {**metadata_groups,
        "fa3_combine_rounds": waves * (metadata_s + local["partial_rounds"]),
        "fa3_combine_metadata": waves * metadata_s,
        "fa3_combine_batch_scan_dependency": waves * batch_scan_s,
        "fa3_combine_stage_chain": waves * cta_s,
        "fa3_combine_l2_payload": payload / capacity("l2_cache"),
        "fa3_combine_scalar": scalar_s}
    bottleneck = max(groups, key=groups.__getitem__)
    prediction = _fa3_kernel_prediction(kernel, hardware, groups[bottleneck], groups, bottleneck)
    return dataclasses.replace(prediction, diagnostics=dataclasses.replace(
        prediction.diagnostics, limitations=prediction.diagnostics.limitations + metadata_limits + (
            "fa3_combine_source_normalization_shared_weights_and_weighted_O_at_existing_capacity",
            "fa3_combine_padded_issued_rows_and_replicated_column_LSE_reads",
            "fa3_combine_full_split_partial_O_projection_not_observed_max_valid_scales",
            "fa3_combine_L2_reuse_and_resident_slot_lifetime_are_aggregate_assumptions",
            "fa3_combine_four_slot_O_copy_queue_at_existing_per_SM_L2_capacity",
            "fa3_combine_varlen_coordinate_metadata_at_shared_source_scan_dependency",
            "fa3_combine_invalid_prefetch_contention_and_uneven_row_tails_partial",
            "fa3_combine_warp_reduction_barrier_and_other_integer_dependency_latency_unbound")))



def _flashinfer_mla_prefill_prediction(
        kernel: KernelSpec, hardware) -> KernelPrediction:
    """flashinfer mla_hopper.cuh absorbed-MLA extend on the MLAPlan schedule.

    One 64 x 64 KV tile per CTA step runs the same absorbed-MLA wgmma mix as
    the FA3 decode tile (QK over 512 + 64 latent/rope dims, 16x64 softmax,
    PV into 512), whose measured chain is ``decode_ring.hold_cycles``; the
    KV stream through L2 bounds the step from below.
    """

    mechanisms = _fa3_mechanisms(hardware)
    clock_hz = mechanisms["measured_sm_clock_hz"]
    tile = kernel.launch.tiling.tile_shape
    if (tile["q"], tile["kv"], tile["qk_head"], tile["value_head"]) != (
            64, 64, 64, 512):
        raise NotImplementedError(
            "flashinfer MLA template covers 64x64 ckv 512 / kpe 64")
    chain = _fa3_decode_chain_cycles(mechanisms)
    workers = kernel.launch.grid[0]
    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    from src.kernel.memory_flash_attention import flashinfer_mla_source_plan
    program=logical_interface_memory_access_program(kernel,hardware)
    kv_tiles, items, _, merge_iterations = flashinfer_mla_source_plan(
        program,expected_kernel=kernel,num_sms=workers)
    active = workers
    from src.kernel.memory_path_solver import logical_mla_kv_step_bytes
    kv_block_bytes=logical_mla_kv_step_bytes(program,expected_kernel=kernel,kv_rows=tile["kv"])
    l2 = hardware.resource_by_role("l2_cache")
    l2_bytes_per_clk = l2.effective_capacity(
        active, sm_freq_hz=clock_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz) / clock_hz
    l2_ii = active * kv_block_bytes / l2_bytes_per_clk
    ii = max(chain, l2_ii)
    critical = max(kv_tiles)
    critical_cycles = critical * ii
    first_load_s = hardware.dram_latency_ns * 1e-9
    merge_s = _flashinfer_mla_merge_s(hardware, workers, merge_iterations)
    execution_s = critical_cycles / clock_hz + first_load_s + merge_s
    groups = {
        "flashinfer_mla_tile_chain": critical * chain / clock_hz,
        "flashinfer_mla_l2_kv_stream": critical * l2_ii / clock_hz,
        "flashinfer_mla_first_load": first_load_s,
        "flashinfer_mla_grid_sync_and_merge": merge_s,
    }
    bottleneck = ("flashinfer_mla_tile_chain" if chain >= l2_ii
                  else "flashinfer_mla_l2_kv_stream")
    return _fa3_kernel_prediction(
        kernel, hardware, execution_s, groups, bottleneck)


def _flashinfer_mla_merge_s(hardware, workers, merge_iterations) -> float:
    """grid.sync then DevicePersistentMergeStates when any KV tile is split.

    The merge loop (SASS of BatchMLAPageAttentionHopperKernel, flashinfer
    0.6.11) consumes each chunk's LSE and partial-O loads (FMNMX, FADD,
    MUFU.EX2, FMUL, FFMA) before issuing the next chunk's loads, so every
    chunk of a pass costs the measured serial merge-walk interval: loads of
    lines other SMs wrote before the grid barrier plus the rescale chain.
    """

    if not merge_iterations:
        return 0.0
    return (hardware.cooperative_grid_sync_ns(workers) * 1e-9
            + max(merge_iterations) * hardware.split_merge_chunk_s())


def _flashinfer_mla_decode_prediction(
        kernel: KernelSpec, hardware) -> KernelPrediction:
    """flashinfer Hopper MLA decode / verify on the MLAPlan schedule.

    Each work item streams its KV tiles through the same paged two-stage
    ring as the FA3 decode producer (64-row blocks of 576 BF16, 128
    producer threads, cp.async); the per-item cost is the measured ring
    first-full plus (tiles - 1) steady intervals at the active CTA count,
    then the absorbed-tile consumer chain.
    """

    mechanisms = _fa3_mechanisms(hardware)
    clock_hz = mechanisms["measured_sm_clock_hz"]
    tile = kernel.launch.tiling.tile_shape
    if (tile["q"], tile["kv"], tile["qk_head"], tile["value_head"]) != (
            64, 64, 64, 512):
        raise NotImplementedError(
            "flashinfer MLA template covers 64x64 ckv 512 / kpe 64")
    chain = _fa3_decode_chain_cycles(mechanisms)
    workers = kernel.launch.grid[0]
    from src.kernel.memory_access_builder import logical_interface_memory_access_program
    from src.kernel.memory_flash_attention import flashinfer_mla_source_plan
    program=logical_interface_memory_access_program(kernel,hardware)
    kv_tiles, items, _, merge_iterations = flashinfer_mla_source_plan(
        program,expected_kernel=kernel,num_sms=workers)
    active = sum(1 for count in items if count)
    ring = mechanisms["decode_ring_by_grid"]
    first_full, ii = ring[min(active, max(ring))]
    critical_cycles = max(
        count * (first_full + chain) + max(tiles - count, 0) * ii
        for tiles, count in zip(kv_tiles, items) if count)
    merge_s = _flashinfer_mla_merge_s(hardware, workers, merge_iterations)
    prologue_s = _flashinfer_mla_plan_prologue_s(hardware)
    execution_s = prologue_s + critical_cycles / clock_hz + merge_s
    groups = {
        "flashinfer_mla_plan_prologue": prologue_s,
        "flashinfer_mla_ring": critical_cycles / clock_hz,
        "flashinfer_mla_grid_sync_and_merge": merge_s,
    }
    return _fa3_kernel_prediction(
        kernel, hardware, execution_s, groups, "flashinfer_mla_ring")


def _flashinfer_mla_plan_prologue_s(hardware) -> float:
    """Dependent plan-metadata loads before a CTA's first KV tile.

    ``BatchMLAPageAttentionHopperKernel`` (mla_hopper.cuh) reads
    ``work_indptr[blockIdx.y]``, then ``get_block_coord`` reads the MLAPlan
    arrays at that work index; only then can ``prefetch_offset`` read the
    page table, whose load the measured paged-KV ring's first-full already
    contains.  In a decode step every layer streams its expert weights
    through L2, so the plan lines are DRAM-cold: two dependent loads at the
    measured DRAM first-load-to-use (fa2_oaccum_load_use_cycles, dram_cg).
    """

    model = hardware.fa2_oaccum_load_use_cycles
    if model is None:
        raise NotImplementedError(
            f"{hardware.hardware_id} publishes no DRAM load-to-use cycles")
    matches = tuple(entry for entry in model["cache_entries"]
                    if entry["cache_tier"] == "dram_cg")
    if len(matches) != 1:
        raise ValueError("load-to-use evidence has no unique dram_cg entry")
    mechanisms = _fa3_mechanisms(hardware)
    return (2 * matches[0]["first_load_ready_cycles"]
            / mechanisms["measured_sm_clock_hz"])


def _fa3_midlayer(kernel: KernelSpec, hardware) -> KernelPrediction:
    algorithm = kernel.launch.algorithm
    if algorithm in (_FA3_SPARSE_MLA_030_ALGORITHM, _FA3_SPARSE_MLA_COMBINE_030_ALGORITHM):
        return _fa3_sparse_mla_prediction(kernel, hardware)
    if algorithm in (_FA3_GQA_OBSERVED_030_ALGORITHM,
                     _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_COMBINE_030_ALGORITHM,
                     _FA3_GQA_TMA_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM,
                     _FA3_GQA_TMA_COMBINE_030_ALGORITHM, _FA3_PREFILL_COMBINE_ALGORITHM):
        return _fa3_gqa_observed_prediction(kernel, hardware)
    if algorithm == _FA3_GQA_PREFILL_030_ALGORITHM:
        return _fa3_gqa_prefill_030_partial_prediction(kernel, hardware)
    if algorithm == _FLASHINFER_MLA_PREFILL_ALGORITHM:
        if kernel.variant == "mla_decode":
            return _flashinfer_mla_decode_prediction(kernel, hardware)
        return _flashinfer_mla_prefill_prediction(kernel, hardware)
    if algorithm == _FA3_PREFILL_ALGORITHM:
        return _fa3_prefill_prediction(kernel, hardware)
    if algorithm == _FA3_PREFILL_SPLIT_ALGORITHM:
        return _fa3_split_prefill_prediction(kernel, hardware)
    if algorithm == _FA3_DECODE_ALGORITHM:
        return _fa3_decode_prediction(kernel, hardware)
    if algorithm == _FA3_COMBINE_ALGORITHM:
        return _fa3_combine_prediction(kernel, hardware)
    raise AssertionError(f"unhandled FA3 algorithm {algorithm!r}")


def _fa3_resource_input(kernel: KernelSpec, hardware):
    """Aggregate FA3 demand (Roofline/Oracle input); no schedule attached."""

    algorithm = kernel.launch.algorithm
    source_limits = ()
    q_lens, kv_lens = _fa3_lengths(kernel)
    heads = int(kernel.shape["num_query_heads"])
    tile = kernel.launch.tiling.tile_shape
    if algorithm in (_FA3_SPARSE_MLA_030_ALGORITHM, _FA3_SPARSE_MLA_COMBINE_030_ALGORITHM):
        work = _fa3_sparse_mla_work(kernel, hardware)
        source_limits = (
            "fa3_sparse_global_indices_alias_and_readiness_unbound",
            "fa3_sparse_selected_cache_reuse_unobserved",
        )
        read_bytes, write_bytes = float(work["read_bytes"]), float(work["write_bytes"])
        scores = float(work["scores"])
        flops = {"compute_tensor": float(work["tensor_flops"]), "compute_reduction": 0.0,
                 "compute_fma": float(work["scalar_flops"]), "compute_alu": 0.0,
                 "compute_sfu": float(work["sfu_ops"]), "warp_issue": 0.0}
    elif algorithm in (_FA3_GQA_OBSERVED_030_ALGORITHM,
                     _FA3_GQA_OBSERVED_SPLIT_030_ALGORITHM, _FA3_GQA_COMBINE_030_ALGORITHM,
                     _FA3_GQA_TMA_030_ALGORITHM, _FA3_GQA_TMA_SPLIT_030_ALGORITHM,
                     _FA3_GQA_TMA_COMBINE_030_ALGORITHM, _FA3_PREFILL_COMBINE_ALGORITHM):
        work = _fa3_gqa_observed_work(kernel, hardware)
        _validate_fa3_gqa_public_operands(kernel, hardware, work)
        read_bytes = float(work["read_bytes"])
        write_bytes = float(work["write_bytes"] + work["semaphore_reset_bytes"])
        scores = float(work["scores"])
        flops = {"compute_tensor": float(work["tensor_flops"]),
                 "compute_reduction": 0.0,
                 "compute_fma": (float(work["scalar_flops"]) if kernel.variant == "combine" else 4 * scores),
                 "compute_alu": 0.0,
                 "compute_sfu": (float(work["sfu_ops"]) if kernel.variant == "combine" else scores),
                 "warp_issue": 0.0}
    elif algorithm == _FA3_GQA_PREFILL_030_ALGORITHM:
        work = _fa3_gqa_prefill_030_source_work(kernel, hardware)
        public_work = work["public_work"]
        read_bytes = float(public_work["read_bytes"])
        write_bytes = float(public_work["write_bytes"])
        scores = float(sum(work["blocks"]) * tile["q"] * tile["kv"])
        flops = {"compute_tensor": float(work["tensor_flops"]),
                 "compute_reduction": 2 * scores, "compute_fma": 0.0,
                 "compute_alu": 3 * scores, "compute_sfu": scores,
                 "warp_issue": 0.0}
    elif algorithm == _FA3_COMBINE_ALGORITHM:
        work=current_fa3_combine_source_work(kernel)
        if work is None:
            work=fa3_combine_prediction_work(kernel,hardware)
        read_bytes=float(work['projected_read_bytes'])
        write_bytes=float(work['projected_write_bytes'])
        flops = {"compute_tensor": 0.0, "compute_reduction": 0.0,
                 "compute_fma": read_bytes / 4, "compute_alu": 0.0,
                 "compute_sfu": 0.0, "warp_issue": 0.0}
    elif algorithm == _FA3_PREFILL_SPLIT_ALGORITHM:
        work = _fa3_split_prefill_work(kernel, hardware)
        read_bytes, write_bytes = float(work["read_bytes"]), float(work["write_bytes"])
        scores = float(work["scores"])
        flops = {"compute_tensor": float(work["tensor_flops"]), "compute_reduction": 0.0,
                 "compute_fma": 4 * scores, "compute_alu": 0.0,
                 "compute_sfu": scores, "warp_issue": 0.0}
    elif algorithm == _FA3_PREFILL_ALGORITHM:
        blocks = sum(_fa3_prefill_tiles(kernel))
        issued = 2.0 * blocks * tile["q"] * tile["kv"] * (
            tile["qk_head"] + tile["value_head"])
        if has_fa3_prefill_source(kernel):
            work = current_fa3_prefill_source_work(kernel)
            if work is None:
                work = fa3_prefill_prediction_work(kernel,hardware)
            read_bytes = float(work['projected_read_bytes'])
            write_bytes = float(work['projected_write_bytes'] + work['unpriced_lse_write_bytes'])
        else:
            from src.kernel.memory_access_builder import logical_interface_memory_access_program
            from src.kernel.memory_path_solver import logical_interface_source_traffic
            program=logical_interface_memory_access_program(kernel,hardware)
            _,read_bytes,write_bytes=logical_interface_source_traffic(program,expected_kernel=kernel,hardware=hardware)
        if (not has_fa3_prefill_source(kernel)
                and "return_softmax_lse" in kernel.attributes and kernel.attributes["return_softmax_lse"]):
            # vLLM MLA context merges consume one FP32 LSE per live query/head.
            # This is an output of the same attention call, not a second launch.
            write_bytes += float(heads * sum(q_lens) * 4)
        scores = float(blocks * tile["q"] * tile["kv"])
        flops = {"compute_tensor": issued, "compute_reduction": 2 * scores,
                 "compute_fma": 0.0, "compute_alu": 3 * scores,
                 "compute_sfu": scores, "warp_issue": 0.0}
    elif algorithm == _FLASHINFER_MLA_PREFILL_ALGORITHM:
        from src.kernel.memory_access_builder import logical_interface_memory_access_program
        from src.kernel.memory_flash_attention import flashinfer_mla_source_plan
        from src.kernel.memory_path_solver import logical_interface_source_traffic
        program=logical_interface_memory_access_program(kernel,hardware)
        kv_tiles, _, cluster_size, _ = flashinfer_mla_source_plan(
            program,expected_kernel=kernel,num_sms=int(hardware.num_compute_units))
        blocks = sum(kv_tiles) * cluster_size
        issued = 2.0 * blocks * tile["q"] * tile["kv"] * (
            tile["qk_head"] + 2 * tile["value_head"])
        _, source_read, source_write=logical_interface_source_traffic(program,expected_kernel=kernel,hardware=hardware)
        read_bytes,write_bytes=float(source_read),float(source_write)
        scores = float(blocks * tile["q"] * tile["kv"])
        flops = {"compute_tensor": issued, "compute_reduction": 2 * scores,
                 "compute_fma": 0.0, "compute_alu": 3 * scores,
                 "compute_sfu": scores, "warp_issue": 0.0}
    elif algorithm == _FA3_DECODE_ALGORITHM:
        from src.kernel.memory_access_builder import logical_interface_memory_access_program
        from src.kernel.memory_path_solver import logical_interface_source_traffic
        program=logical_interface_memory_access_program(kernel,hardware)
        tiles = _fa3_decode_tile_blocks(kernel, hardware, program=program)
        blocks = sum(tiles)
        issued = 2.0 * blocks * tile["q"] * tile["kv"] * (
            tile["qk_head"] + 2 * tile["value_head"])
        _,source_read,source_write=logical_interface_source_traffic(program,expected_kernel=kernel,hardware=hardware)
        read_bytes,write_bytes=float(source_read),float(source_write)
        scores = float(blocks * tile["q"] * tile["kv"])
        flops = {"compute_tensor": issued, "compute_reduction": 2 * scores,
                 "compute_fma": 0.0, "compute_alu": 3 * scores,
                 "compute_sfu": scores, "warp_issue": 0.0}
    else:
        raise AssertionError(f"unhandled FA3 algorithm {algorithm!r}")
    dram_bytes = read_bytes + write_bytes
    demand = build_forward_demand(dram_bytes, dram_bytes, False, flops)
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
        variant=kernel.variant.lower(),
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
            "fa3_aggregate_demand_without_source_schedule",
            "production_duration_calibration_not_used",
        ) + source_limits,
    )
    return inp, dram_bytes
# --- end wp10 FA3 template --------------------------------------------------


_MODEL_CACHE_SIZE = 8
_model_cache = RuntimeCache()
_model_cache_lock = threading.Lock()


def prediction_equivalence_fields(kernel: KernelSpec):
    """Name the pinned FA2 reducer by its scratch/output ABI, not its KV.

    The split-combine launch consumes FP32 partial O and LSE, then writes O
    and final LSE. Its traffic, scalar/dependency work, and CTA program read
    query rows, head width, and the already-selected split count; they never
    read the original KV window. Keep the entire launch and every other
    field, including all padding and resource-policy controls, in the key.
    A graph with masked request slots retains its exact original contract.

    This describes prediction equivalence only. Public resource_input's
    historical Q/K/V compulsory-footprint return value is not canonicalized;
    the reducer's Roofline route uses its actual scratch/output demand.
    """

    if (type(kernel) is not KernelSpec
            or kernel.execution_support != "supported"
            or kernel.variant not in ("combine", "split_combine")
            or kernel.launch.algorithm != "flash_attn_splitkv_combine"):
        return None
    if (type(kernel.shape["kv_len"]) is not int
            or any(type(kernel.attributes[name]) is not tuple
                or any(type(value) is not int for value in kernel.attributes[name])
                for name in ("q_lens", "kv_lens", "model_input_q_lens", "model_input_kv_lens")
                if name in kernel.attributes)
            or _has_graph_padding(kernel)):
        return None
    q_lens, _ = _request_lengths(kernel)
    shape = dict(kernel.shape)
    attributes = dict(kernel.attributes)
    shape["kv_len"] = max(q_lens)
    attributes["kv_lens"] = q_lens
    if "model_input_kv_lens" in attributes:
        attributes["model_input_kv_lens"] = attributes["model_input_q_lens"]
    return {"shape": shape, "attributes": attributes}


def _model(kernel: KernelSpec, hardware, *, attach_pipeline: bool = True):
    # The scheduler-backed MLA ABI has no public semantic tile. Reject only
    # classifier-supplied tile geometry, as the ordinary construction does.
    if (has_metax_mla_source(kernel) and _pipeline_geometry(kernel) is not None
            and not _has_semantic_tile_evidence(kernel)):
        raise ValueError("FLASH_ATTN evaluation-only launch classifier cannot publish semantic tile geometry to production Forward")
    from src.kernel.memory_access_builder import logical_interface_priced_by_source
    if logical_interface_priced_by_source(kernel):
        # The public logical interfaces are cheap summaries. Price their
        # actual bound work each time instead of returning an unbound cache.
        return _model_uncached(kernel,hardware,attach_pipeline=attach_pipeline)
    if has_fa3_combine_source(kernel):
        work=fa3_combine_prediction_work(kernel,hardware)
        with fa3_combine_source_work(kernel,work):
            return _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
    if has_fa3_prefill_source(kernel):
        work = fa3_prefill_prediction_work(kernel,hardware)
        with fa3_prefill_source_work(kernel,work):
            return _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
    if has_cann_tnd_source(kernel):
        work = cann_tnd_source_demands(cann_tnd_memory_access_program(kernel,hardware))
        with cann_tnd_source_work(kernel,work):
            inp,compulsory = _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
        source_limits = (
            'cann_TND_public_payload_and_selected_INT8_mask_source_native_arguments_workspace_issued_mapping_pending',
            'cann_TND_host_lengths_not_tensor_reads_mask_DRAM_residency_unknown',)
        if work['kv_unique_bytes'] is None:
            source_limits += ('cann_TND_paged_payload_not_unique_physical_pages_metadata_native_rereads_unpriced',)
        return dataclasses.replace(inp,limitations=inp.limitations+source_limits),compulsory
    if has_metax_varlen_source(kernel):
        from src.kernel.memory_access_builder import metax_varlen_prediction_work
        work = metax_varlen_prediction_work(kernel,hardware)
        with metax_varlen_source_work(kernel,work):
            inp,compulsory = _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
        # New source admission must preserve the existing prediction API.
        # The source program itself exports its unknown address/workspace and
        # unpriced cumulative service boundaries. Keep the legacy diagnostic
        # annotation only on the domain that already published it.
        if kernel.launch.algorithm in (
                _METAX_372_Q128_SPLIT_PREFILL_ALGORITHM,
                _METAX_372_Q64_PREFILL_ALGORITHM,
                _METAX_372_Q64_PACKED_PREFILL_ALGORITHM,
                'metax_flash_attention_trait_direct:prefill'):
            return inp,compulsory
        if kernel.launch.algorithm != _METAX_372_Q128_PREFILL_ALGORITHM:
            raise AssertionError('C500 varlen source diagnostic dispatch is incomplete')
        return dataclasses.replace(inp,limitations=inp.limitations+(
            'metax_varlen_public_TND_cumulative_LSE_source_native_workspace_issued_rereads_pending',
            'metax_varlen_cumulative_reads_unpriced_legacy_service_projection',)),compulsory
    from src.kernel.memory_cann_mla import has_cann_mla_source,cann_mla_source_demands,cann_mla_source_work
    if has_cann_mla_source(kernel):
        from src.kernel.memory_access_builder import build_memory_access_program
        from src.kernel.memory_access import Granularity
        source=build_memory_access_program(kernel,hardware,None,Granularity.TILE,program_id='cann.mla')
        work=cann_mla_source_demands(source,expected_kernel=kernel,hardware=hardware)
        with cann_mla_source_work(kernel,work):
            return _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
    if has_cann_attention_source(kernel):
        work = cann_attention_source_demands(cann_attention_memory_access_program(kernel,hardware),expected_kernel=kernel)
        with cann_attention_source_work(kernel,work):
            inp,compulsory = _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
        source_limits = ('cann_attention_public_BSH_payload_and_argument_not_complete_native_workspace_or_issued_trace',)
        if work['kv_unique_bytes'] is None:
            source_limits += ('cann_paged_compulsory_projection_uses_logical_interface_bytes_physical_page_alias_unknown',)
        if work['argument_payload_bytes'] is None:
            source_limits += ('cann_attention_tiling_argument_source_unbound',)
        return dataclasses.replace(inp,limitations=inp.limitations+source_limits),compulsory
    if has_fa2_split_source(kernel):
        from src.kernel.memory_access_builder import fa2_split_prediction_work
        work = fa2_split_prediction_work(kernel,hardware)
        with fa2_split_source_work(kernel,work):
            inp,compulsory = _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
        source_limits = (
            'fa2_split_logical_payload_partial_lse_not_native_capacity_addresses_or_complete_side_effects',
            'fa2_split_legacy_replay_and_empty_split_price_projection_retained',)
        return dataclasses.replace(inp,limitations=inp.limitations+source_limits),compulsory
    if has_grouped_mla_source(kernel):
        source = grouped_mla_prediction_work(kernel,hardware)
        with grouped_mla_source_work(kernel,source):
            inp,compulsory = _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
        return dataclasses.replace(inp,limitations=inp.limitations+(
            'grouped_mla_packed_partial_and_cache_alias_source_native_page_and_other_side_effects_pending',
            'grouped_mla_length_reads_unpriced_and_legacy_full_split_scratch_projection',)),compulsory
    if has_metax_mla_source(kernel):
        source = metax_mla_prediction_work(kernel,hardware)
        with metax_mla_source_work(kernel,source):
            inp,compulsory = _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)
        return dataclasses.replace(inp,limitations=inp.limitations+(
            'metax_mla_payload_partial_lse_logical_source_not_native_metadata_page_or_issued_trace',)),compulsory
    return _cached_source_model(kernel,hardware,attach_pipeline=attach_pipeline)


def _cached_source_model(kernel: KernelSpec, hardware, *, attach_pipeline: bool = True):
    """Return an isolated copy of a repeated immutable kernel construction."""

    source_limits = ()
    if kernel.launch.algorithm in ('flash_attn_splitkv','flash_attn_splitkv_combine') and not has_fa2_split_source(kernel):
        source_limits = ('fa2_split_native_source_paged_ragged_graph_or_storage_abi_unbound',)
    if has_dense_flash_attention_source(kernel):
        dense_attention_prediction_work(kernel,hardware)
        # Source completeness belongs to the exported memory program. Keep
        # the existing public prediction diagnostics unchanged.

    key = (id(kernel), id(hardware), attach_pipeline)
    with _model_cache_lock:
        if key in _model_cache:
            cached_kernel, cached_hardware, inp, compulsory = _model_cache[key]
            if cached_kernel is kernel and cached_hardware is hardware:
                _model_cache.move_to_end(key)
                return dataclasses.replace(inp, limitations=inp.limitations + source_limits), compulsory
    inp, compulsory = _model_uncached(
        kernel, hardware, attach_pipeline=attach_pipeline)
    with _model_cache_lock:
        _model_cache[key] = (kernel, hardware, inp, compulsory)
        _model_cache.move_to_end(key)
        if len(_model_cache) > _MODEL_CACHE_SIZE:
            _model_cache.popitem(last=False)
    return dataclasses.replace(inp, limitations=inp.limitations + source_limits), compulsory


def kimi_k3_nope_context_attention_logical_work(kernel: KernelSpec):
    """Unmasked rectangular attention with a BF16 partial and FP32 LSE.

    The source caller fixes tensor math/interfaces. The noncausal backend
    tile/split/softmax program is not bound by the existing fresh-only FA3
    launch contract. Export logical work without assigning a complete price.
    """
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.variant != "kimi_k3_nope_context_attention"
            or kernel.execution_support != "unsupported" or kernel.launch is not None):
        raise ValueError("K3 context attention requires an unbound source KernelSpec")
    q, k, heads = (kernel.shape[name] for name in ("q_len", "kv_len", "num_query_heads"))
    operands = []
    for role, dtype, elements, access in (
            ("query", "bf16", q * heads * 192, "read"),
            ("context_key", "bf16", k * heads * 192, "read"),
            ("context_value_view", "bf16", k * heads * 128, "read"),
            ("context_partial", "bf16", q * heads * 128, "write"),
            ("context_lse", "fp32", q * heads, "write")):
        operands.append({"role": role, "dtype": dtype, "elements": elements,
                         "bytes": elements * (4 if dtype == "fp32" else 2), "access": access})
    return {"source": "vllm/models/kimi_k3/nvidia/mla.py:_compute_prefill_context/run_chunk; mla/prefill/flash_attn.py:run_prefill_context_chunk",
            "output_target": kernel.attributes["output_target"] if "output_target" in kernel.attributes else "chunk_partial",
            "operands": tuple(operands), "mathematical_tensor_flops": 2 * heads * q * k * (192 + 128),
            "unmasked_score_elements": heads * q * k, "lse_elements": heads * q,
            "demand": {"compute_tensor": float(2 * heads * q * k * (192 + 128)),
                       "compute_vector": 0.0, "compute_reduction": 0.0,
                       "compute_fma": float(4 * heads * q * k),
                       "compute_sfu": float(heads * q * k + 2 * heads * q), "compute_alu": 0.0},
            "source_launch": None, "in_place_roles": (),
            "limitations": ("BF16 tensor roles and FP32 LSE, no score/probability materialization assertion",
                            "ideal_stable_softmax_exp_reciprocal_and_log_not_native_tile_split_or_compiler_services",
                            "logical source work is not an actual provider or native latency acceptance")}


def _softmax_mathematical_work(pairs, rows, qk_dim, value_dim, payload):
    return {"valid_score_pairs": pairs, "query_head_rows": rows,
        "tensor_flops": 2 * (qk_dim + value_dim) * pairs, "scalar_flops": 4 * pairs - rows,
        "exp_operations": pairs, "reciprocal_operations": rows, "log_operations": 0,
        "max_comparisons": pairs - rows, "argument_payload_bytes": payload,
        "scope": "ideal mathematical FLOPs/bytes; no actual launch/provider/DRAM/scratch or scalar-instruction acceptance"}


def gqa_mathematical_work(kernel: KernelSpec):
    """Useful GQA math and unique argument payload, independent of tiles.

    QK^T and PV each cost 2*D FLOPs per valid query/key/head pair. Stable
    softmax is represented by scale, subtract-max, sum and normalization:
    4*P-R scalar FLOPs, P exponentials and R reciprocals. These are ideal
    mathematical demands, not source-issued scalar instructions or finite-
    precision equivalence to online softmax. Max comparisons are reported
    separately. No score matrix, LSE/split scratch or cache replay is invented.
    KV update is a separate producer in the caller graph.
    """
    if (kernel.kind is not KernelKind.FLASH_ATTN
            or kernel.variant not in ("prefill", "decode")
            or kernel.execution_support != "unsupported" or kernel.launch is not None
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] not in ("gqa_softmax_v1", "ascend_0_18_full_fia_softmax_v1")):
        raise ValueError("GQA mathematical work requires its explicit unbound program")
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    ascend_full = kernel.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1"
    if ascend_full:
        q_lens, kv_lens = kernel.attributes["model_input_q_lens"], kernel.attributes["model_input_kv_lens"]
    heads, kv_heads, dimension = (kernel.shape[name] for name in (
        "num_query_heads", "num_kv_heads", "head_dim"))
    pairs = sum(min(q, k) * k - (min(q, k) * (min(q, k) - 1) // 2 if kernel.attributes["causal"] else 0)
        for q, k in zip(q_lens, kv_lens, strict=True)) * heads
    rows = sum(min(q, k) for q, k in zip(q_lens, kv_lens, strict=True)) * heads
    # One unique read of Q/K/V and write of output, with GQA KV sharing.
    payload = (2 * sum(q_lens) * heads * dimension
        + 2 * sum(kv_lens) * kv_heads * dimension) * kernel.shape["dtype_bytes"]
    if ascend_full:
        # full_graph_fia passes one BF16 placeholder alongside the main .out
        # result. This is source-visible storage, not per-row mathematical LSE.
        payload += kernel.shape["dtype_bytes"]
    work = _softmax_mathematical_work(pairs, rows, dimension, dimension, payload)
    if ascend_full:
        return {**work, "physical_query_head_rows": sum(q_lens) * heads,
            "empty_query_head_rows": sum(q_lens) * heads - rows,
            "source_program": "ascend_0_18_full_fia",
            "empty_row_numeric_output_bound": False}
    return work


def mla_decode_mathematical_work(kernel: KernelSpec):
    """Dense absorbed MLA useful math, with K/V sharing latent cache rows.

    The cache's first512 BF16 channels are also V; no duplicate V-cache
    payload is invented. The upstream test_flashmla.py reference returns
    FP32 logsumexp as LSE, even under CP1. Reusing the ideal softmax's max
    and sum adds one logarithm and one addition per query/head row. This
    does not bind a compiler's LSE instruction program or library scratch.
    Cache insertion and Q/V absorption are separate caller operations.
    """
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.variant != "mla_decode"
            or kernel.execution_support != "unsupported" or kernel.launch is not None
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] not in ("bf16_mla_decode_softmax_v1",
                "bf16_mla_causal_empty_rows_softmax_v1")):
        raise ValueError("MLA mathematical work requires its explicit unbound decode program")
    return _absorbed_mla_mathematical_work(kernel, return_lse=True)


def mla_absorbed_prefill_mathematical_work(kernel: KernelSpec):
    """SGLang FlashInferMLA parent EXTEND returns O only, without public LSE.

    The useful causal math shares the absorbed latent-cache owner. Ragged
    query lengths have no FlashMLA uniform-query or128-token restriction.
    Caller metadata declares a nonempty prefix below its8192 threshold;
    backend page planning, scratch and physical services remain unbound.
    """
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.variant != "mla_absorbed_prefill"
            or kernel.execution_support != "unsupported" or kernel.launch is not None
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] != "bf16_mla_absorbed_prefill_softmax_v1"):
        raise ValueError("absorbed MLA prefill requires its explicit unbound no-LSE program")
    return _absorbed_mla_mathematical_work(kernel, return_lse=False)


def _absorbed_mla_mathematical_work(kernel, *, return_lse):
    heads = kernel.shape["num_query_heads"]
    qk_dim, value_dim = kernel.shape["qk_head_dim"], kernel.shape["v_head_dim"]
    query_rows, cache_rows = sum(kernel.attributes["q_lens"]), sum(kernel.attributes["kv_lens"])
    # Bottom-right causal alignment: when q > k, the leading q-k rows
    # have no visible key. They retain output/LSE storage but do not perform
    # a finite softmax or log(0). Keep this explicit mathematical scope
    # separate from any compiler's masked-tile or empty-row implementation.
    lengths = tuple(zip(kernel.attributes["q_lens"], kernel.attributes["kv_lens"], strict=True))
    pairs = heads * sum(min(q, k) * k - min(q, k) * (min(q, k) - 1) // 2 for q, k in lengths)
    rows = heads * sum(min(q, k) for q, k in lengths)
    empty_rows = heads * query_rows - rows
    payloads = {"prepared_query": query_rows * heads * qk_dim * 2,
        "shared_latent_rope_cache": cache_rows * qk_dim * 2,
        "latent_output": query_rows * heads * value_dim * 2}
    if return_lse:
        payloads["lse_output"] = heads * query_rows * 4
    work = _softmax_mathematical_work(pairs, rows, qk_dim, value_dim, sum(payloads.values()))
    return {**work, "scalar_flops": work["scalar_flops"] + (rows if return_lse else 0),
        "log_operations": rows if return_lse else 0,
        "payload_bytes": payloads, "lse_mathematical_work_included": return_lse,
        "source_instruction_program_bound": False,
        "empty_query_head_rows": empty_rows, "output_query_head_rows": heads * query_rows,
        "empty_state_output": "mathematical_zero_O_negative_infinity_LSE_not_native_output_acceptance" if empty_rows else None,
        "remaining_source_services": ("compiled_softmax_lse_program" if return_lse else "compiled_softmax_program",
            "page_table_and_schedule_metadata", "library_scratch_and_physical_layout", "actual_provider_and_launch")}


def mla_prefill_mathematical_work(kernel: KernelSpec):
    """Expanded BF16 MLA suffix/context Q/K/V and optional FP32 LSE.

    The caller already expanded latent KV and appended the cache. This
    counts its prepared arguments once, not latent cache again; it does not
    choose an online algorithm, source-issued comparisons, scratch or launch.
    """
    if (kernel.kind is not KernelKind.FLASH_ATTN or kernel.variant != "mla_prefill"
            or kernel.execution_support != "unsupported" or kernel.launch is not None
            or "mathematical_program" not in kernel.attributes
            or kernel.attributes["mathematical_program"] not in (
                "bf16_mla_prefill_softmax_v1", "bf16_mla_context_softmax_v1", "bf16_mla_context_empty_rows_softmax_v1", "bf16_mla_one_shot_prefill_softmax_v1")):
        raise ValueError("MLA prefill mathematical work requires its explicit unbound program")
    heads = kernel.shape["num_query_heads"]
    qk, value = kernel.shape["qk_head_dim"], kernel.shape["v_head_dim"]
    q_lens, kv_lens = kernel.attributes["q_lens"], kernel.attributes["kv_lens"]
    lse = kernel.attributes["return_softmax_lse"]
    if kernel.attributes["mathematical_program"] in ("bf16_mla_prefill_softmax_v1", "bf16_mla_one_shot_prefill_softmax_v1"):
        if kernel.attributes["causal"] is not True or (kernel.attributes["mathematical_program"] == "bf16_mla_prefill_softmax_v1" and q_lens != kv_lens):
            raise ValueError("MLA suffix mathematical program requires fresh causal sequences")
        pairs = heads * causal_attention_pairs(q_lens, kv_lens)
    else:
        if kernel.attributes["causal"] is not False or lse is not True:
            raise ValueError("MLA context mathematical program requires noncausal attention and LSE")
        pairs = heads * sum(q * k for q, k in zip(q_lens, kv_lens, strict=True))
    query_rows, cache_rows = sum(q_lens), sum(kv_lens)
    output_rows = heads * query_rows
    empty_rows = heads * sum(q for q, k in zip(q_lens, kv_lens, strict=True) if k == 0)
    rows = output_rows - empty_rows
    payloads = {"prepared_query": query_rows * heads * qk * 2,
        "prepared_key": cache_rows * heads * qk * 2,
        "prepared_value": cache_rows * heads * value * 2,
        "expanded_output": query_rows * heads * value * 2}
    if lse:
        payloads["lse_output"] = output_rows * 4
    work = _softmax_mathematical_work(pairs, rows, qk, value, sum(payloads.values()))
    # Stable LSE reuses the represented max and exponential sum, adding
    # log(sum) + max once per row, as in the absorbed MLA decode program.
    return {**work, "scalar_flops": work["scalar_flops"] + (rows if lse else 0),
        "log_operations": rows if lse else 0, "payload_bytes": payloads,
        "empty_query_head_rows": empty_rows,
        "empty_state_output": "zero_BF16_O_negative_infinity_FP32_LSE" if empty_rows else None,
        "lse_mathematical_work_included": lse, "source_instruction_program_bound": False}


class FlashAttentionPredictor(KernelPredictor):
    def analytical_without_launch(self, kernel: KernelSpec, hardware):
        if kernel.variant == "projected_attention_path":
            from src.kernel.predictors.projected_attention import source_prediction
            return source_prediction(kernel, hardware)
        if ("mathematical_program" in kernel.attributes
                or kernel.variant == "kimi_k3_nope_context_attention"):
            prediction = self.roofline_without_launch(kernel, hardware)
            return dataclasses.replace(prediction, bottleneck_group="analytical_attention")
        return super().analytical_without_launch(kernel, hardware)

    def roofline_without_launch(self, kernel: KernelSpec, hardware):
        if kernel.variant == "projected_attention_path":
            from src.kernel.predictors.projected_attention import source_prediction
            return source_prediction(kernel, hardware)
        if "mathematical_program" in kernel.attributes:
            program = kernel.attributes["mathematical_program"]
            if program in ("gqa_softmax_v1", "ascend_0_18_full_fia_softmax_v1"):
                work = gqa_mathematical_work(kernel)
            elif program in ("bf16_mla_decode_softmax_v1", "bf16_mla_causal_empty_rows_softmax_v1"):
                work = mla_decode_mathematical_work(kernel)
            elif program == "bf16_mla_absorbed_prefill_softmax_v1":
                work = mla_absorbed_prefill_mathematical_work(kernel)
            elif program in ("bf16_mla_prefill_softmax_v1", "bf16_mla_context_softmax_v1", "bf16_mla_context_empty_rows_softmax_v1", "bf16_mla_one_shot_prefill_softmax_v1"):
                work = mla_prefill_mathematical_work(kernel)
            else:
                raise ValueError("unknown mathematical attention program")
            from src.kernel.memory_access_builder import has_mathematical_memory_source, mathematical_prediction_memory_bytes
            if has_mathematical_memory_source(kernel):
                work = {**work, "argument_payload_bytes": mathematical_prediction_memory_bytes(kernel, hardware)}
            demand = {"compute_tensor": work["tensor_flops"],
                "compute_fma": work["scalar_flops"],
                "compute_sfu": work["exp_operations"] + work["reciprocal_operations"] + work["log_operations"],
                "compute_vector": 0.0, "compute_reduction": 0.0, "compute_alu": 0.0,
                "l2_cache": work["argument_payload_bytes"], "dram": work["argument_payload_bytes"]}
            return prediction_from_ns(roofline_ns(demand, hardware,
                work["argument_payload_bytes"]), "roofline")
        if kernel.variant != "kimi_k3_nope_context_attention":
            return super().roofline_without_launch(kernel, hardware)
        work = kimi_k3_nope_context_attention_logical_work(kernel)
        logical_bytes = sum(operand["bytes"] for operand in work["operands"])
        return prediction_from_ns(roofline_ns(work["demand"], hardware, logical_bytes), "roofline")

    kind = KernelKind.FLASH_ATTN

    def resource_input(self, kernel: KernelSpec, hardware):
        from src.kernel.memory_flash_attention import current_fa3_split_prefill_prediction_work
        current_fa3_split_prefill_prediction_work(kernel, hardware)
        from src.kernel.memory_flash_attention import current_fa3_sparse_mla_prediction_work
        sparse_source_work = current_fa3_sparse_mla_prediction_work(kernel, hardware)
        if "mathematical_program" in kernel.attributes:
            raise NotImplementedError("mathematical attention provides useful Roofline only, without a native resource schedule")
        from src.kernel.memory_access_builder import _standalone_resource_input_scope
        with _standalone_resource_input_scope(kernel):
            value = _model(kernel, hardware)
        if sparse_source_work is not None:
            inp, compulsory = value
            limits = tuple(item for item in inp.limitations
                if item != "fa3_sparse_global_indices_alias_and_readiness_unbound")
            value = dataclasses.replace(inp, limitations=limits + (
                "fa3_sparse_caller_snapshot_aliases_and_readiness_validated",
                "fa3_sparse_device_snapshot_truth_and_index_conversion_unbound",)), compulsory
        from src.kernel.prediction_cache import _retain_public_resource_input
        _retain_public_resource_input(kernel, hardware, value)
        return value

    def midlayer(self, kernel: KernelSpec, hardware):
        from src.kernel.memory_flash_attention import current_fa3_split_prefill_prediction_work
        split_source_work = current_fa3_split_prefill_prediction_work(kernel, hardware)
        from src.kernel.memory_flash_attention import current_fa3_sparse_mla_prediction_work
        sparse_source_work = current_fa3_sparse_mla_prediction_work(kernel, hardware)
        if ("return_softmax_lse" in kernel.attributes and kernel.attributes["return_softmax_lse"]
                and not (kernel.dtype == "bf16" and (has_fa3_prefill_source(kernel)
                    or kernel.variant == "combine" and split_source_work is not None))):
            raise NotImplementedError("explicit MLA prefill LSE currently has a Roofline source contract only")
        if split_source_work is not None:
            prediction = _fa3_midlayer(kernel, hardware)
            limitations = tuple(item for item in prediction.diagnostics.limitations
                if item != "fa3_split_prefill_scratch_aggregate_without_shared_native_bindings")
            return dataclasses.replace(prediction, diagnostics=dataclasses.replace(prediction.diagnostics,
                limitations=limitations + ("fa3_split_prefill_source_aliases_and_readiness_validated",)))
        if sparse_source_work is not None:
            prediction = _fa3_midlayer(kernel, hardware)
            limits = tuple(item for item in prediction.diagnostics.limitations
                if item != "fa3_sparse_global_indices_alias_and_readiness_unbound")
            return dataclasses.replace(prediction, diagnostics=dataclasses.replace(prediction.diagnostics,
                limitations=limits + ("fa3_sparse_caller_snapshot_aliases_and_readiness_validated",
                    "fa3_sparse_device_snapshot_truth_and_index_conversion_unbound",)))
        if is_fa3_launch(kernel):
            if has_fa3_combine_source(kernel):
                work=fa3_combine_prediction_work(kernel,hardware)
                with fa3_combine_source_work(kernel,work):
                    return _fa3_midlayer(kernel,hardware)
            if has_fa3_prefill_source(kernel):
                work = fa3_prefill_prediction_work(kernel,hardware)
                with fa3_prefill_source_work(kernel,work):
                    return _fa3_midlayer(kernel,hardware)
            return _fa3_midlayer(kernel, hardware)
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(
            _with_graph_regime(inp, kernel))

    def roofline(self, kernel: KernelSpec, hardware):
        # Within one evaluation case, Oracle and Forward already construct the
        # complete immutable source input for this exact spec.  Roofline
        # consumes only its aggregate demand and compulsory traffic, which are
        # identical to the no-pipeline construction; reuse the complete input
        # rather than reconstructing source work and dropping its schedule.
        inp, compulsory = self.resource_input(kernel, hardware)
        if kernel.variant.lower() in ("combine", "split_combine"):
            compulsory = inp.demand_per_resource["l2_cache"]
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory), "roofline")

    def pipeweave(self, kernel: KernelSpec, hardware, *, model_hardware=None):
        if kernel.variant.lower() in ("combine", "split_combine"):
            return super().pipeweave(
                kernel, hardware, model_hardware=model_hardware)
        from src.kernel.predictors.pipeweave_adapter import predict_flash_attn
        return predict_flash_attn(
            kernel, hardware, model_hardware=model_hardware)


# The memo is private to a synchronous resource-input construction.
class _GeometryLocal(threading.local):
    def __init__(self):
        self.frame = None

_geometry_local = _GeometryLocal()
_geometry_sources = tuple((name, globals()[name], globals()[name].__code__)
    for name in ("_is_source_bound_fa2_pipeline", "_splits",
        "_sum_ceiled_arithmetic_progression", "_split_tile_counts"))
_geometry_cta_work = _AttentionCTAWork

def _plain_geometry(value):
    if type(value) in (int, float, bool, str, type(None)):
        return True
    return type(value) is tuple and all(_plain_geometry(item) for item in value)

def _plain_geometry_map(values):
    return (type(values) is MappingProxyType
        and all(type(name) is str and _plain_geometry(item)
                for name, item in values.items()))


def _reuse_call_geometry(function):
    @wraps(function)
    def repeated(kernel, *args, **kwargs):
        frame = _geometry_local.frame
        if (frame is None or kernel is not frame[0]
                or (args and any(type(value) not in (int, bool) for value in args))
                or (kwargs and any(type(value) is not bool for value in kwargs.values()))):
            return function(kernel, *args, **kwargs)
        _, shape, attributes, launch, variant, dtype, tiling, tile, splits, algorithm, memo = frame
        name = function.__name__
        if name == "_pipeline_cta_class_counts" and (
                (len(args) > 2 and args[2] is not False)
                or ("grouped_mla" in kwargs and kwargs["grouped_mla"] is not False)):
            return function(kernel, *args, **kwargs)
        if ((name != "_request_lengths" and kernel.shape is not shape)
                or (name != "_shape" and kernel.attributes is not attributes)):
            return function(kernel, *args, **kwargs)
        if (name == "_pipeline_cta_class_counts" and (
                kernel.launch is not launch or kernel.variant is not variant
                or kernel.dtype is not dtype or launch.tiling is not tiling
                or tiling.tile_shape is not tile or tiling.num_kv_splits != splits
                or launch.algorithm is not algorithm)):
            return function(kernel, *args, **kwargs)
        key = (function, args, tuple(kwargs.items()))
        if key not in memo:
            memo[key] = function(kernel, *args, **kwargs)
        return memo[key]
    return repeated

_geometry_original_helpers = tuple((function, function.__code__)
    for function in (_shape, _request_lengths, _has_graph_padding,
        _pipeline_cta_class_counts))
_shape = _reuse_call_geometry(_shape)
_request_lengths = _reuse_call_geometry(_request_lengths)
_has_graph_padding = _reuse_call_geometry(_has_graph_padding)
_pipeline_cta_class_counts = _reuse_call_geometry(_pipeline_cta_class_counts)
_geometry_helpers = (_shape, _request_lengths, _has_graph_padding,
    _pipeline_cta_class_counts)
_geometry_model_uncached = _model_uncached

def _model_uncached(kernel, hardware, *, attach_pipeline=True):
    previous = _geometry_local.frame
    launch = kernel.launch
    if (type(kernel) is KernelSpec and type(hardware) is HWSpec
            and type(launch) is LaunchConfig and type(launch.tiling) is TilingSpec
            and type(kernel.shape) is MappingProxyType
            and type(kernel.attributes) is MappingProxyType
            and _plain_geometry_map(kernel.shape) and _plain_geometry_map(kernel.attributes)
            and _plain_geometry_map(launch.tiling.tile_shape)
            and type(launch.tiling.num_kv_splits) is int
            and type(kernel.variant) is str and type(kernel.dtype) is str
            and type(launch.algorithm) is str
            and launch.algorithm in ('flash_attn_dense', 'flash_attn_splitkv', 'vllm_flash_attn_splitkv')
            and _geometry_helpers == (_shape, _request_lengths,
                _has_graph_padding, _pipeline_cta_class_counts)
            and all(globals()[name] is function and function.__code__ is code
                for name, function, code in _geometry_sources)
            and all(function.__code__ is code for function, code in _geometry_original_helpers)
            and _AttentionCTAWork is _geometry_cta_work
            # This selects the known FA2 class-count branch, without calling
            # the unrelated grouped-MLA or vendor fallback classifiers.
            and not _has_graph_padding(kernel)):
        _geometry_local.frame = (kernel, kernel.shape, kernel.attributes, launch,
            kernel.variant, kernel.dtype, launch.tiling, launch.tiling.tile_shape,
            launch.tiling.num_kv_splits, launch.algorithm, {})
    else:
        _geometry_local.frame = None
    try:
        return _geometry_model_uncached(kernel, hardware, attach_pipeline=attach_pipeline)
    finally:
        _geometry_local.frame = previous
