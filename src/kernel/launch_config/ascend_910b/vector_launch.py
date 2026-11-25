"""Huawei 910B vector AIV launch geometry."""

from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.selection_primitives import (
    ceil_div, launch_policy, require_cann_vector_variant, require_dtype,
    require_non_empty_string, require_paged_kv_cache_contract,
    require_positive_integer, require_shape)
from src.kernel.model_contract import (
    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
    SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
    SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
    SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER,
)


_CANN_8_5_1_NORM_FULL_ROW_UB_ELEMENTS = 12_288
# CANN 8.5.1 RmsNorm host tiler, ``rms_norm_tiling.cpp``: the BF16 block
# (:372 with SetByDtype), the MERGE_N selection bound SMALL_REDUCE_NUM (:47),
# the UB reservation UB_USED (:36, :362), the MERGE_N row-buffer weights
# (:543), and the is_performance rank/extent bounds (:72-74, :504).
_CANN_8_5_1_NORM_BF16_ELEMENTS_PER_BLOCK = 16
_CANN_8_5_1_NORM_SMALL_REDUCE_NUM = 2_000
_CANN_8_5_1_NORM_UB_RESERVED_BYTES = 1_024
_CANN_8_5_1_NORM_MERGE_N_COL_ALIGN_WEIGHT = 16
_CANN_8_5_1_NORM_MERGE_N_ROW_OVERHEAD_BYTES = 260
_CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_COLUMNS = 5_120
_CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_LEADING = 512
_CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_SECOND = 8
_CANN_NORM_GROUPED_INPUT_MATERIALIZATION = "grouped_strided_to_contiguous"
_CANN_8_5_1_NORM_SPLIT_D_ROW_FACTOR = 64
_CANN_8_5_1_RMS_RSTD_VALUES_PER_TRANSACTION = 8
_CANN_8_5_1_RMS_RSTD_TRANSACTION_BYTES = 32
_CANN_8_5_1_ADD_RMS_KEY30_MIN_HIDDEN = 2_001
_CANN_8_5_1_RMSNORM_TILING_PAYLOAD_BYTES = 28
_CANN_8_5_1_ADD_RMSNORM_TILING_PAYLOAD_BYTES = 80
_CANN_8_5_1_ELEMENTWISE_TILING_PAYLOAD_BYTES = 56
_CANN_8_5_1_RESHAPE_CACHE_EQUAL_KEY = 200_000_000
_CANN_8_5_1_RESHAPE_CACHE_FALLBACK_KEY = 202_000_000
_CANN_8_5_1_RESHAPE_CACHE_FAST_TOKEN_LIMIT = 999
_CANN_8_5_1_RESHAPE_CACHE_BLOCK_BYTES = 32
_CANN_8_5_1_ROPE_LARGE_TOKEN_THRESHOLD = 64
_CANN_8_5_1_ROPE_BF16_SLICE_ELEMENTS = 4_096
_CANN_8_5_1_ROPE_BF16_ALIGN_SLICE_ELEMENTS = 2_048
_CANN_8_5_1_ROPE_ROTARY_COEFFICIENT = 2
_CANN_8_5_1_ROPE_COS_FORMAT = 0
_CANN_8_5_1_ROPE_VECTOR_REPEAT_ELEMENTS = 64
_CANN_8_5_1_GATHER_V3_TILING_PAYLOAD_BYTES = 192
_CANN_8_5_1_GATHER_V3_DIRECT_KEY = 10_330
_CANN_8_5_1_ELEMENTWISE_BLOCK_ALIGNMENT = 128
_CANN_8_5_1_ELEMENTWISE_MIN_BLOCK_ELEMENTS = 1_024
_CANN_8_5_1_ELEMENTWISE_MULTICORE_MIN_ELEMENTS = 1_536
_CANN_8_5_1_ADD_DOUBLE_BUFFER_BLOCK_MIN = 12_288
_CANN_8_5_1_ADD_DOUBLE_BUFFER_MAX_TILE = 6_016
_CANN_8_5_1_MUL_N_BUFFER_MAX_TILE = 9_728
_CANN_8_5_1_ELEMENTWISE_INPUT_PING_PONG_SLOTS = 2
_CANN_8_5_1_ELEMENTWISE_SOURCE_MAX_ELEMENTS = 16_777_216
_CANN_MOE_COMPUTE_EXPERT_TOKENS_SORTED_BOUND = 12_000
_CANN_MOE_COMPUTE_EXPERT_TOKENS_EXPERT_BOUND = 256
_CANN_MOE_COMPUTE_EXPERT_TOKENS_SPECIAL_SORTED = 65_536
_CANN_MOE_COMPUTE_EXPERT_TOKENS_SPECIAL_EXPERTS = 8
_CANN_MOE_COMPUTE_EXPERT_TOKENS_MAX_SORTED = 1 << 24
_CANN_MOE_COMPUTE_EXPERT_TOKENS_MAX_EXPERTS = 2_048
_CANN_MOE_FINALIZE_ROUTING_BF16_NO_BIAS_TILING_KEY = 20_036
_CANN_MOE_COMPUTE_EXPERT_TOKENS_EVIDENCE_REFERENCE = (
    "microbench/ascend_910b/references/"
    "moe_compute_expert_tokens_public_source.json")
_CANN_MOE_COMPUTE_EXPERT_TOKENS_EVIDENCE_SHA256 = (
    "d5973c750e4ca2df9640d487a774e6eb079b95c15add53cf439349f9ceea597a")
_CANN_MOE_CAST_ROLES = ("topk_ids", "expert_tokens")
_CANN_MOE_INIT_ROUTING_VARIANTS = (
    "moe_init_routing", "moe_init_routing_v2", "moe_init_routing_v3")
_CANN_8_5_0_SWIGLU_UB_BYTES = 196_608
_CANN_8_5_0_SWIGLU_BF16_BYTES = 2
_CANN_8_5_0_SWIGLU_UB_ALIGNMENT_BYTES = 32
_CANN_8_5_0_SWIGLU_L2_CACHE_LINE_BYTES = 512
_CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_COUNT = 4_095
_CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_LENGTH = 65_535
_CANN_8_5_0_SWIGLU_QUEUE_COUNT = 3
_CANN_8_5_0_SWIGLU_BF16_BUFFER_COUNT = 2
_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_UB_BYTES = 85 * 1024


def select_cann_public_rope_gather_v3_tiling(
        tokens: int, head_dim: int, dtype_bytes: int, vector_cores: int):
    """Return the CANN 8.5.1 GatherV3 program for public RoPE cache rows.

    ``torch_npu._npu_rotary_embedding`` calls IndexSelect twice on the public
    two-dimensional BF16 cos/sin cache.  CANN 8.5.1 launch captures at output
    token counts 1820, 1821, 2048, 2963, 3869 and 8192, including the Llama
    131072-row cache, all select the direct-row Tmpl0/10330 program with one
    output partition per AIV.  The full cache row count therefore does not
    select the all-cache or Robin programs for this ABI.  The returned fields
    are launch facts consumed identically by Dataset and Serving; no duration
    enters this selector.
    """

    for name, value in (
            ("tokens", tokens), ("head_dim", head_dim),
            ("dtype_bytes", dtype_bytes), ("vector_cores", vector_cores)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CANN GatherV3 {name} must be an integer")
        if value <= 0:
            raise ValueError(f"CANN GatherV3 {name} must be positive")
    if dtype_bytes != 2:
        raise ValueError("public CANN RoPE GatherV3 requires BF16 rows")

    return {
        "tiling_key": _CANN_8_5_1_GATHER_V3_DIRECT_KEY,
        "real_core_num": min(tokens, vector_cores),
        "selected_row_bytes": tokens * head_dim * dtype_bytes,
        "tiling_payload_bytes": _CANN_8_5_1_GATHER_V3_TILING_PAYLOAD_BYTES,
    }


def _aligned_elementwise_factor(value: int) -> int:
    return (ceil_div(value, _CANN_8_5_1_ELEMENTWISE_BLOCK_ALIGNMENT)
            * _CANN_8_5_1_ELEMENTWISE_BLOCK_ALIGNMENT)


def _cann_rmsnorm_rstd_transfer_bytes(
        rows: int, rows_per_core: int, work_items: int) -> int:
    """Return the source-defined physical key-0/1 rstd DMA payload.

    CANN RMSNorm writes one FP32 rstd value per logical row, but its
    ``DataCopyCustom`` output is issued in at least one 32-byte transaction
    (eight FP32 values) for every AIV-owned row extent.  This is transfer
    geometry, distinct from the public logical rstd tensor size.
    """

    if rows <= 0 or rows_per_core <= 0 or work_items <= 0:
        raise ValueError("CANN RMSNorm rstd transfer requires positive rows")
    last_rows = rows - (work_items - 1) * rows_per_core
    if last_rows <= 0 or last_rows > rows_per_core:
        raise ValueError(
            "CANN RMSNorm rstd transfer worker ownership is inconsistent")
    worker_rows = (rows_per_core,) * (work_items - 1) + (last_rows,)
    return sum(
        ceil_div(row_count, _CANN_8_5_1_RMS_RSTD_VALUES_PER_TRANSACTION)
        * _CANN_8_5_1_RMS_RSTD_TRANSACTION_BYTES
        for row_count in worker_rows)


def _cann_rmsnorm_num_col_align(hidden: int) -> int:
    """Return the CANN 8.5.1 RMSNorm ``numColAlign`` for a BF16 column count.

    ``rms_norm_tiling.cpp:372`` rounds ``numCol`` up to the dtype block, which
    ``SetByDtype`` fixes at sixteen elements for BF16.
    """

    if hidden <= 0:
        raise ValueError("CANN RMSNorm numColAlign requires positive columns")
    return ceil_div(
        hidden, _CANN_8_5_1_NORM_BF16_ELEMENTS_PER_BLOCK
    ) * _CANN_8_5_1_NORM_BF16_ELEMENTS_PER_BLOCK


def _cann_rmsnorm_merge_n_row_factor(hidden: int, hardware):
    """Return the MERGE_N row-group extent, or ``None`` outside that mode.

    ``rms_norm_tiling.cpp:524-526`` selects ``MODE_MERGE_N`` for every
    Ascend 910B RMSNorm whose ``numColAlign`` is at most
    ``SMALL_REDUCE_NUM`` (``:47``), after and overriding the Split-D test.
    ``:540-545`` then fixes the row group at
    ``rowFactor = ubSize / (numColAlign * 16 + 260)`` over the ``UB_USED``
    remainder of physical UB (``:36``, ``:362``).  All of these are discrete
    host-tiler facts.
    """

    num_col_align = _cann_rmsnorm_num_col_align(hidden)
    if num_col_align > _CANN_8_5_1_NORM_SMALL_REDUCE_NUM:
        return None
    unified_buffer_bytes = int(
        hardware.resource_by_role("ub").size_bytes)
    usable_bytes = unified_buffer_bytes - _CANN_8_5_1_NORM_UB_RESERVED_BYTES
    row_buffer_bytes = (
        num_col_align * _CANN_8_5_1_NORM_MERGE_N_COL_ALIGN_WEIGHT
        + _CANN_8_5_1_NORM_MERGE_N_ROW_OVERHEAD_BYTES)
    row_factor = usable_bytes // row_buffer_bytes
    if row_factor <= 0:
        raise ValueError(
            "CANN RMSNorm MERGE_N row factor must be positive")
    return row_factor


def _cann_rmsnorm_merge_n_row_groups(
        rows: int, rows_per_core: int, work_items: int, row_factor: int):
    """Return the per-AIV MERGE_N row-group extents.

    ``rms_norm_tiling.cpp:554-557`` derives ``rowLoop``/``rowTail`` from
    ``blockFactor`` and ``rowFactor`` separately for the tail AIV, and
    ``KernelRmsNormMergeN::Process`` (``:91-94``) runs ``rowLoop - 1`` full
    groups followed by the tail group.
    """

    if rows <= 0 or rows_per_core <= 0 or work_items <= 0 or row_factor <= 0:
        raise ValueError("CANN RMSNorm MERGE_N groups require positive rows")
    last_rows = rows - (work_items - 1) * rows_per_core
    if last_rows <= 0 or last_rows > rows_per_core:
        raise ValueError(
            "CANN RMSNorm MERGE_N worker ownership is inconsistent")
    groups = []
    for worker_rows in (
            (rows_per_core,) * (work_items - 1) + (last_rows,)):
        row_loop = ceil_div(worker_rows, row_factor)
        tail = worker_rows - (row_loop - 1) * row_factor
        groups.append((row_factor,) * (row_loop - 1) + (tail,))
    return tuple(groups)


def _cann_rmsnorm_merge_n_rstd_transfer_bytes(
        rows: int, rows_per_core: int, work_items: int,
        row_factor: int) -> int:
    """Return the physical MERGE_N rstd DataCopyCustom payload.

    ``KernelRmsNormMergeN::CopyOutRstd`` issues one transfer of the current
    group's row count, so the 32-byte minimum transaction applies once per row
    group rather than once per AIV extent.
    """

    return sum(
        ceil_div(group_rows, _CANN_8_5_1_RMS_RSTD_VALUES_PER_TRANSACTION)
        * _CANN_8_5_1_RMS_RSTD_TRANSACTION_BYTES
        for worker_groups in _cann_rmsnorm_merge_n_row_groups(
            rows, rows_per_core, work_items, row_factor)
        for group_rows in worker_groups)


def _cann_rmsnorm_merge_n_performance_flag(
        rows: int, hidden: int, attributes: Mapping[str, object]) -> int:
    """Return the host-tiler ``is_performance`` flag for a MERGE_N launch.

    ``rms_norm_tiling.cpp:503-507`` sets the flag from the *input tensor rank
    and leading extents*, never from a duration: a two- or three-dimensional x
    with a one-dimensional gamma, ``numCol <= 5120``, a leading dimension of at
    most 512 (and, when x is three-dimensional, a second dimension of at most
    8), and a BF16/FP16 non-mixed dtype.  ``KernelRmsNormMergeN`` reads it at
    ``:79`` and uses it to skip ``BroadCastGamma`` and to replace the two
    per-group ``BroadCast`` calls with ``Brcb``/``repeatByRow``.

    The NORM lowering publishes a grouped input materialization exactly when
    the caller passes the three-dimensional per-head view; every other NORM
    source is the two-dimensional token-major tensor.
    """

    if hidden > _CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_COLUMNS:
        return 0
    if "input_materialization" in attributes:
        if attributes["input_materialization"] != (
                _CANN_NORM_GROUPED_INPUT_MATERIALIZATION):
            raise ValueError(
                "unsupported Norm input materialization for CANN MERGE_N: "
                f"{attributes['input_materialization']!r}")
        outer_groups = require_positive_integer(
            attributes, "input_outer_groups")
        rows_per_group = require_positive_integer(
            attributes, "input_rows_per_group")
        if outer_groups * rows_per_group != rows:
            raise ValueError(
                "CANN RMSNorm grouped input extents disagree with rows")
        return int(
            outer_groups
            <= _CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_LEADING
            and rows_per_group
            <= _CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_SECOND)
    return int(rows <= _CANN_8_5_1_NORM_MERGE_N_PERFORMANCE_MAX_LEADING)


def select_cann_elementwise_tiling(        elements: int, operation: str, vector_cores: int):
    """Return CANN 8.5.1 TBE Add/Mul ownership and UB queue facts.

    The formulas reproduce the vendor AutoTiling result for the supported
    BF16 one-dimensional domain.  They use only the shipped TBE scheduler's
    alignment/capacity rules and decoded host tiling payload fields; no kernel
    duration or performance counter participates.  The source loop always
    selects input storage with ``tile_index & 1``: two physical ping/pong
    slots per input stream.  This is distinct from the Mul q4/q1 published
    steady-service admission depth, which remains four source tiles.
    """

    if isinstance(elements, bool) or not isinstance(elements, int):
        raise TypeError("CANN elementwise element count must be an integer")
    if elements <= 0:
        raise ValueError("CANN elementwise element count must be positive")
    if operation not in ("add", "mul"):
        raise ValueError(
            f"unsupported source-closed CANN elementwise operation: {operation!r}")
    if isinstance(vector_cores, bool) or not isinstance(vector_cores, int):
        raise TypeError("CANN elementwise Vector core count must be an integer")
    if vector_cores <= 0:
        raise ValueError("CANN elementwise Vector core count must be positive")
    if elements > _CANN_8_5_1_ELEMENTWISE_SOURCE_MAX_ELEMENTS:
        return None

    if elements < _CANN_8_5_1_ELEMENTWISE_MULTICORE_MIN_ELEMENTS:
        block_factor = elements
        block_dim = 1
    else:
        block_factor = max(
            _CANN_8_5_1_ELEMENTWISE_MIN_BLOCK_ELEMENTS,
            _aligned_elementwise_factor(ceil_div(elements, vector_cores)),
        )
        block_dim = ceil_div(elements, block_factor)

    if operation == "add":
        if block_factor >= _CANN_8_5_1_ADD_DOUBLE_BUFFER_BLOCK_MIN:
            tiling_key = 210_010_000
            input_queue_depth = 2
            # ``AddWithCastCompute`` is emitted through the LEVEL_2
            # ``BroadcastOneDimSch``.  Its two CopyIn nodes receive the
            # scheduler's ping/pong buffers, but the single CopyOut node does
            # not: ``GetBufferIdsImpl`` creates PongMte3 buffers only for
            # LEVEL_0.  Do not turn the tiler's multi-tile selection into a
            # fictitious second VECOUT credit.
            output_queue_depth = 1
            tile_count = ceil_div(
                block_factor, _CANN_8_5_1_ADD_DOUBLE_BUFFER_MAX_TILE)
            ub_factor = _aligned_elementwise_factor(
                ceil_div(block_factor, tile_count))
        else:
            tiling_key = 210_000_000
            input_queue_depth = 1
            output_queue_depth = 1
            ub_factor = block_factor
    elif operation == "mul":
        if block_dim == 1:
            tiling_key = 210_000_000
            input_queue_depth = 1
            output_queue_depth = 1
            ub_factor = block_factor
        else:
            tiling_key = 210_020_000
            input_queue_depth = 4
            output_queue_depth = 1
            tile_count = ceil_div(
                block_factor, _CANN_8_5_1_MUL_N_BUFFER_MAX_TILE)
            ub_factor = _aligned_elementwise_factor(
                ceil_div(block_factor, tile_count))
    else:
        raise AssertionError(f"unhandled CANN elementwise operation: {operation!r}")

    return {
        "block_dim": block_dim,
        "block_factor": block_factor,
        "ub_factor": ub_factor,
        "tiling_key": tiling_key,
        "input_queue_depth": input_queue_depth,
        "input_ping_pong_slots": (
            _CANN_8_5_1_ELEMENTWISE_INPUT_PING_PONG_SLOTS),
        "output_queue_depth": output_queue_depth,
    }


def select_cann_moe_abs_tiling(elements: int, vector_cores: int) -> dict[str, int]:
    """Return the public ``Abs<int32>`` AIV partition used by FusedMoE.

    ``aclnnAbs_AbsAiCore_Abs`` does not use the generic Add/Mul small-work
    branch.  Its host payload always establishes a 1024-element minimum
    worker extent, then balances the remaining work over the available AIVs
    on 128-element boundaries.  This is a launch-only fact from the public
    CANN task payloads: for example, 1024/1280/2560 elements launch
    1/2/3 AIVs, while 65536/163840 launch 47/48 AIVs.  It contains neither a
    duration nor an accuracy-derived threshold.
    """

    if isinstance(elements, bool) or not isinstance(elements, int):
        raise TypeError("CANN FusedMoE Abs element count must be an integer")
    if elements <= 0:
        raise ValueError("CANN FusedMoE Abs element count must be positive")
    if isinstance(vector_cores, bool) or not isinstance(vector_cores, int):
        raise TypeError("CANN FusedMoE Abs Vector core count must be an integer")
    if vector_cores <= 0:
        raise ValueError("CANN FusedMoE Abs Vector core count must be positive")

    block_factor = max(
        _CANN_8_5_1_ELEMENTWISE_MIN_BLOCK_ELEMENTS,
        _aligned_elementwise_factor(ceil_div(elements, vector_cores)),
    )
    return {
        "block_dim": ceil_div(elements, block_factor),
        "block_factor": block_factor,
    }


def select_moe_compute_expert_tokens_tiling_key(
        routed_items: int, experts: int) -> int:
    """Return the CANN host tiler's Int32SS/M/L program key."""

    for name, value in (("routed_items", routed_items), ("experts", experts)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CANN MoeComputeExpertTokens {name} must be an integer")
        if value <= 0:
            raise ValueError(f"CANN MoeComputeExpertTokens {name} must be positive")
    if routed_items > _CANN_MOE_COMPUTE_EXPERT_TOKENS_MAX_SORTED:
        raise ValueError(
            "CANN MoeComputeExpertTokens sorted input exceeds the source "
            "limit of 2**24 elements")
    if experts > _CANN_MOE_COMPUTE_EXPERT_TOKENS_MAX_EXPERTS:
        raise ValueError(
            "CANN MoeComputeExpertTokens expert count exceeds the source "
            "limit of 2048")
    if ((routed_items <= _CANN_MOE_COMPUTE_EXPERT_TOKENS_SORTED_BOUND
         and experts <= _CANN_MOE_COMPUTE_EXPERT_TOKENS_EXPERT_BOUND)
            or (routed_items
                == _CANN_MOE_COMPUTE_EXPERT_TOKENS_SPECIAL_SORTED
                and experts
                == _CANN_MOE_COMPUTE_EXPERT_TOKENS_SPECIAL_EXPERTS)):
        return 1001
    if experts <= _CANN_MOE_COMPUTE_EXPERT_TOKENS_EXPERT_BOUND:
        return 1002
    return 1003


def _align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _select_cann_swiglu_buffer_tiling(
        rows: int, columns: int, vector_cores: int, buffer_count: int):
    input_bytes = _CANN_8_5_0_SWIGLU_BF16_BYTES
    ub_min_block_len = (
        _CANN_8_5_0_SWIGLU_UB_ALIGNMENT_BYTES // input_bytes)
    cache_line_len = (
        _CANN_8_5_0_SWIGLU_L2_CACHE_LINE_BYTES // input_bytes)
    single_element_bytes = (
        buffer_count * _CANN_8_5_0_SWIGLU_QUEUE_COUNT * input_bytes
        + _CANN_8_5_0_SWIGLU_BF16_BUFFER_COUNT * 4)
    max_tile_len = _align_down(
        _CANN_8_5_0_SWIGLU_UB_BYTES // single_element_bytes,
        ub_min_block_len)
    is_32_byte_aligned = columns % ub_min_block_len == 0
    copy_col_limit = (
        _CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_LENGTH
        if is_32_byte_aligned
        else _CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_LENGTH // input_bytes)
    base_col_len = min(columns, max_tile_len, copy_col_limit)
    if base_col_len < columns and base_col_len > cache_line_len:
        base_col_len = _align_down(base_col_len, cache_line_len)
    max_base_row_len = min(
        rows, _CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_COUNT)

    def must_use_one_row(candidate_col_len: int) -> bool:
        stride = columns * 2 - candidate_col_len
        if is_32_byte_aligned:
            return stride > (
                _CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_LENGTH
                * ub_min_block_len)
        return stride * input_bytes > (
            _CANN_8_5_0_SWIGLU_COPY_MAX_BLOCK_LENGTH)

    best = None
    best_rank = None
    while True:
        base_row_len = min(
            max_tile_len // _align_up(base_col_len, ub_min_block_len),
            max_base_row_len)
        if (base_row_len < 1
                or (base_row_len > 1
                    and must_use_one_row(base_col_len))):
            if best is None:
                raise ValueError(
                    "CANN 8.5.0 SwiGLU source tiler found no legal base tile")
            break

        tail_col_len = columns % base_col_len
        if not (base_row_len > 1 and must_use_one_row(tail_col_len)):
            total_tiles = (
                ceil_div(rows, base_row_len)
                * ceil_div(columns, base_col_len))
            block_dim = min(total_tiles, vector_cores)
            base_size = base_row_len * base_col_len
            complete_base_tiles = (
                rows // base_row_len) * (columns // base_col_len)
            rank = (
                block_dim, -total_tiles, base_size, complete_base_tiles)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = {
                    "base_row_len": base_row_len,
                    "base_col_len": base_col_len,
                    "total_tiles": total_tiles,
                    "block_dim": block_dim,
                    "tiles_per_core": ceil_div(total_tiles, block_dim),
                }

        if (base_col_len <= cache_line_len
                or base_row_len >= max_base_row_len):
            break
        if base_col_len % cache_line_len == 0:
            base_col_len -= cache_line_len
        else:
            base_col_len = _align_down(base_col_len, cache_line_len)

    if best is None:
        raise ValueError(
            "CANN 8.5.0 SwiGLU source tiler found no legal base tile")
    return best


def select_cann_swiglu_tiling(
        rows: int, columns: int, vector_cores: int):
    """Return the CANN ops-nn 8.5.0 BF16 SwiGLU host tiling result.

    This is a direct translation of ``GluSingleTilingCalculator`` in
    ``cann/ops-nn@9f8a66e795ef1842c2118cf5bbadfe3624bdd1ef`` at
    ``activation/swi_glu/op_host/swi_glu_tiling.cpp``.  It uses the Ascend
    910B platform's 192 KiB UB and source constants only; no duration,
    profiler counter, or per-shape observation participates.
    """

    for name, value in (
            ("rows", rows), ("columns", columns),
            ("vector_cores", vector_cores)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"CANN SwiGLU {name} must be an integer")
        if value <= 0:
            raise ValueError(f"CANN SwiGLU {name} must be positive")

    double_buffer = _select_cann_swiglu_buffer_tiling(
        rows, columns, vector_cores, 2)
    selected = double_buffer
    queue_depth = 2
    if double_buffer["tiles_per_core"] == 2:
        single_buffer = _select_cann_swiglu_buffer_tiling(
            rows, columns, vector_cores, 1)
        if (single_buffer["tiles_per_core"] == 1
                and single_buffer["block_dim"]
                >= double_buffer["block_dim"]):
            selected = single_buffer
            queue_depth = 1
    return {
        "base_row_len": selected["base_row_len"],
        "base_col_len": selected["base_col_len"],
        "total_tiles": selected["total_tiles"],
        "block_dim": selected["block_dim"],
        "queue_depth": queue_depth,
    }


def _cann_swiglu_source_parameters(
        rows: int, columns: int, vector_cores: int):
    launch_fact = select_cann_swiglu_tiling(rows, columns, vector_cores)
    total_tiles = launch_fact["total_tiles"]
    exact_cores = launch_fact["block_dim"]
    serial_invocations = ceil_div(total_tiles, exact_cores)
    parameters = {
        "lowered_vector_operations": float(3 * rows * columns),
        "vector_exp_serial_invocations": float(serial_invocations),
        "vector_div_serial_invocations": float(serial_invocations),
        "swiglu_base_row_len": float(launch_fact["base_row_len"]),
        "swiglu_base_col_len": float(launch_fact["base_col_len"]),
        "swiglu_total_tiles": float(total_tiles),
        "swiglu_queue_depth": float(launch_fact["queue_depth"]),
    }
    return launch_fact, parameters


def validate_cann_moe_swiglu_launch(
        hardware_id: str, dtype: str, shape: Mapping[str, int],
        attributes: Mapping[str, object], launch: LaunchConfig) -> None:
    """Validate the exact CANN MoE SwiGLU source launch contract."""

    if hardware_id != "ascend_910b" or dtype != "bf16":
        raise ValueError(
            "VECTOR/moe_swiglu source launch requires ascend_910b BF16")
    if (launch.algorithm != "aclnn_moe_swiglu"
            or launch.execution_variant != "moe_swiglu"):
        raise ValueError(
            "VECTOR/moe_swiglu source launch requires the exact CANN "
            "algorithm identity")
    tokens, topk, intermediate = require_shape(
        shape, ("tokens", "topk", "intermediate"))
    routed_rows = require_positive_integer(attributes, "routed_rows")
    if routed_rows > tokens * topk:
        raise ValueError(
            "VECTOR/moe_swiglu routed_rows cannot exceed tokens * topk")
    launch_fact, expected_parameters = _cann_swiglu_source_parameters(
        routed_rows, intermediate, 48)
    expected_grid = (launch_fact["block_dim"], 1, 1)
    if launch.grid != expected_grid or launch.block != (1, 1, 1):
        raise ValueError(
            "VECTOR/moe_swiglu geometry disagrees with the CANN source "
            f"tiler: expected grid={expected_grid}, block=(1, 1, 1), "
            f"actual grid={launch.grid}, block={launch.block}")
    if dict(launch.parameters) != expected_parameters:
        raise ValueError(
            "VECTOR/moe_swiglu parameters disagree with the CANN source "
            "tiler")


def select_vector_launch(shape: Mapping[str, int], dtype: Optional[str],
                         variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    require_dtype(dtype, ("bf16",))
    execution_variant = require_cann_vector_variant(variant)
    policy = launch_policy(attributes)
    vector_cores = int(
        hardware.resource_by_role("compute_vector").total_units)
    parameters = {}
    elementwise_work_items = None
    elementwise_launch_facts = None
    if execution_variant == "elementwise":
        elementwise_work_items, = require_shape(shape, ("element_count",))
        operation = require_non_empty_string(attributes, "operation")
        if operation not in ("add", "mul", "copy", "fill", "read",
                             "notequal_int32_bool", "cast_bool_bf16"):
            raise ValueError(
                f"unsupported CANN elementwise operation: {operation!r}")
        if operation in ("notequal_int32_bool", "cast_bool_bf16"):
            # aclnnNeScalar_NotEqual and the dynamic Cast program are
            # pure-elementwise TBE graphs whose AutoTiling applies the same
            # 1024-element minimum block, 128-element aligned partition, and
            # 16,777,216-element source ceiling as the common schedule
            # (trace block_dims 1/2/27 reproduce exactly, moe_cast precedent).
            compare_tiling = select_cann_elementwise_tiling(
                elementwise_work_items, "add", vector_cores)
            if compare_tiling is None:
                raise NotImplementedError(
                    "Ascend 910B elementwise compare/cast exceeds the "
                    "source-backed CANN elementwise tiling domain")
            elementwise_launch_facts = compare_tiling
        if operation in ("add", "mul"):
            # CANN 8.5.1 AddWithCastCompute and MulXfp16Op both promote two
            # BF16 inputs to FP32 and round the result back to BF16.  The
            # common predictor already counts the arithmetic node, so publish
            # only those three source-level conversions as additional work.
            parameters["lowered_vector_operations"] = float(
                3 * elementwise_work_items)
        if operation in ("add", "mul"):
            elementwise_launch_facts = select_cann_elementwise_tiling(
                elementwise_work_items, operation, vector_cores)
            if elementwise_launch_facts is not None:
                parameters.update({
                    "cann_tiling_key": float(
                        elementwise_launch_facts["tiling_key"]),
                    "elementwise_block_factor": float(
                        elementwise_launch_facts["block_factor"]),
                    "elementwise_ub_factor": float(
                        elementwise_launch_facts["ub_factor"]),
                    "elementwise_input_queue_depth": float(
                        elementwise_launch_facts["input_queue_depth"]),
                    "elementwise_input_ping_pong_slots": float(
                        elementwise_launch_facts["input_ping_pong_slots"]),
                    "elementwise_output_queue_depth": float(
                        elementwise_launch_facts["output_queue_depth"]),
                    # ``BroadcastSch<201>`` deserializes its exact
                    # ``BroadcastOneDimTilingData`` before creating the
                    # scheduler. Its C++ layout is 56 B after the required
                    # int64 alignment; retain that source entry fact instead
                    # of silently reusing the unrelated 28-B RMSNorm model.
                    "cann_tiling_payload_bytes": float(
                        _CANN_8_5_1_ELEMENTWISE_TILING_PAYLOAD_BYTES),
                    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER: 1.0,
                })
    if policy not in ("analytical_launch", "injected_launch"):
        raise AssertionError(f"unhandled launch policy: {policy!r}")
    exact_cores = None
    source_program = None
    if execution_variant in ("rmsnorm", "add_rmsnorm"):
        rows, hidden = require_shape(shape, ("rows", "hidden"))
        rows_per_core = (rows + vector_cores - 1) // vector_cores
        work_items = (rows + rows_per_core - 1) // rows_per_core
        elements = rows * hidden
        # The fixed CANN 8.5.1 host tiler publishes ``blockFactor`` as the
        # number of contiguous rows assigned to every non-tail AIV.  Retained
        # decoded payloads cover the supported row range and agree with the
        # runtime blockDim selected below.  This is a discrete ownership fact,
        # not a duration-derived scheduling coefficient.
        parameters["norm_block_factor"] = float(rows_per_core)
        # The entry source materializes its typed tiling payload with
        # ``GET_TILING_DATA`` before dispatching an implementation key.  The
        # payload type belongs to the source entry, not to a shape or a
        # selected key: CANN 8.5.1 RmsNorm uses RmsNormTilingData (28 B),
        # while AddRmsNorm uses AddRmsNormTilingData (80 B).  Keep this
        # versioned launch fact explicit so Forward cannot price an
        # AddRMSNorm source task with the RMSNorm prologue mechanism.
        parameters["cann_tiling_payload_bytes"] = float(
            _CANN_8_5_1_RMSNORM_TILING_PAYLOAD_BYTES
            if execution_variant == "rmsnorm"
            else _CANN_8_5_1_ADD_RMSNORM_TILING_PAYLOAD_BYTES)
        # CANN 8.5.1 requires one FP32 rstd output per row for every selected
        # RMSNorm/AddRmsNorm key.  The Python caller may discard the returned
        # tensor, but the eager device kernel still writes it.
        parameters["rstd_output_bytes"] = float(rows * 4)
        # Every fixed CANN 8.5.1 RMSNorm/AddRmsNorm implementation adds
        # epsilon once after the row reduction.  Keep this scalar source work
        # explicit, as on the GPU launch packages, rather than hiding it in a
        # full-row lowering pass or in the reciprocal-sqrt service.
        parameters["norm_epsilon_add_flops"] = float(rows)
        # The CANN 8.5.1 host tiler keeps a complete BF16 row in UB through
        # 12,288 elements (RMS key 0; Add-RMS keys 30/33).  Above that
        # source-defined boundary it selects split-D key 1/31.  Those kernels
        # discard the reduction input and issue one second GM CopyIn for the
        # output phase; this is cache-path traffic, not compulsory DRAM work.
        if hidden > _CANN_8_5_1_NORM_FULL_ROW_UB_ELEMENTS:
            parameters["input_reread_count"] = 1.0
            # Split-D key 1 fixes row_factor to NUM_PER_REP_FP32=64; key 31
            # receives the same 64-row factor from the CANN 8.5.1 host tiler.
            # Gamma is copied once per row group and column tile, not once per
            # logical row.  Across all column tiles that is one complete gamma
            # vector per row group.
            full_core_groups = (
                (work_items - 1)
                * ceil_div(
                    rows_per_core,
                    _CANN_8_5_1_NORM_SPLIT_D_ROW_FACTOR))
            last_core_rows = rows - (work_items - 1) * rows_per_core
            weight_l2_copies = full_core_groups + ceil_div(
                last_core_rows, _CANN_8_5_1_NORM_SPLIT_D_ROW_FACTOR)
        else:
            # Normal key 0/30 loads gamma once in Process() and reuses it for
            # every row owned by that AIV.  Single-row key 33 naturally has
            # the same one-copy-per-launched-AIV result.
            weight_l2_copies = work_items
        parameters["norm_weight_l2_copies"] = float(weight_l2_copies)
        # CANN 8.5.1 lowers BF16 normalization to additional full-row Vector
        # passes that are absent from the semantic square/reduce/scale/mul
        # count in the common predictor.  Count logical element operations,
        # not profiler time or a fitted efficiency:
        #
        # * RMS key 0 has three extra casts; split-D key 1 has four because
        #   the output phase reloads and casts x from GM.
        # * Add-RMS key 33 (one row per AIV) has ten extra cast/mean/reduction
        #   passes.  Key 30 retains the FP32 sum and uses eight; split-D key
        #   31 restores a ninth pass after its GM reload.
        #
        # The 48-row key-33 boundary follows the fixed host tiler: one row per
        # physical AIV.  The 12,288-element split-D boundary is declared
        # above.  Retained aiv_vec_fops counters validate the resulting source
        # work but are not used to choose any coefficient.
        if execution_variant == "rmsnorm":
            merge_n_row_factor = _cann_rmsnorm_merge_n_row_factor(
                hidden, hardware)
            if merge_n_row_factor is not None:
                # ``rms_norm_tiling.cpp:524-526`` overrides every other mode on
                # this SoC once ``numColAlign <= SMALL_REDUCE_NUM``, and
                # ``:566-568`` publishes the mode itself as the tiling key.
                # ``KernelRmsNormMergeN`` then owns one ``rowFactor`` row group
                # per DataCopy instead of one row, so the rstd DataCopyCustom
                # boundary is the row group, not the whole AIV extent.
                tiling_key = 2
                parameters["cann_rmsnorm_rstd_transfer_bytes"] = float(
                    _cann_rmsnorm_merge_n_rstd_transfer_bytes(
                        rows, rows_per_core, work_items, merge_n_row_factor))
                parameters["norm_merge_n_row_factor"] = float(
                    merge_n_row_factor)
                parameters["norm_merge_n_performance_flag"] = float(
                    _cann_rmsnorm_merge_n_performance_flag(
                        rows, hidden, attributes))
                # ``rms_norm.cpp:34/54-55`` binds key 2 to
                # ``KernelRmsNormMergeN``.  Its row-group DataCopy, batched
                # ``ReduceSumMultiN`` and per-group depth-one queue lifetime
                # differ from every key-0 program, so publish a discrete
                # source identity rather than reusing a row-wise DAG.
                parameters["source_bound_rmsnorm_merge_n_dag"] = 1.0
            else:
                # ``KernelRmsNorm`` emits the rstd DataCopyCustom at one
                # 32-byte minimum transaction per AIV-owned row extent.  Keep
                # this physical DMA payload separate from the public logical
                # tensor.
                parameters["cann_rmsnorm_rstd_transfer_bytes"] = float(
                    _cann_rmsnorm_rstd_transfer_bytes(
                        rows, rows_per_core, work_items))
                tiling_key = (
                    1
                    if hidden > _CANN_8_5_1_NORM_FULL_ROW_UB_ELEMENTS
                    else 0)
            if tiling_key == 2:
                pass
            elif (tiling_key == 0 and rows <= vector_cores):
                # The normal RMSNorm source owns one complete row on each
                # launched AIV in this domain.  BUFFER_NUM=1 and the literal
                # Process/SubProcess call order close one gamma-load,
                # x-load, compute, y-store, rstd-store worker DAG.  Publish
                # the identity only for one-row workers; key 0 with more
                # rows per AIV has a different repeated-row lifetime.
                parameters[
                    "source_bound_rmsnorm_single_row_dag"] = 1.0
            elif tiling_key == 0:
                # The same fixed normal source keeps gamma for the lifetime
                # of one AIV and processes its rows in rowFactor=64 groups.
                # This marker is separate from the one-row identity so the
                # predictor must close the repeated queue program rather than
                # silently reusing the narrower DAG.
                parameters[
                    "source_bound_rmsnorm_row_group_dag"] = 1.0
            elif tiling_key == 1:
                # The fixed Split-D source completes each at-most-64-row
                # reduction group before its vector normalization and second
                # GM input/output pass.  Publish this discrete identity so
                # Forward can retain that phase order instead of using an
                # aggregate roofline.
                parameters[
                    "source_bound_rmsnorm_split_d_dag"] = 1.0
            if tiling_key == 2:
                # ``KernelRmsNormMergeN`` owns a different source inventory
                # from the row-wise keys, so publish it literally.
                # ``Process`` casts one gamma vector per AIV (``:135-143``)
                # and, when the host tiler leaves ``is_performance`` at zero,
                # expands it once over the whole ``rowFactor`` tile
                # (``:88-90``, ``:151-157``).  The group reduction then issues
                # ``Muls``/``Adds``/``Sqrt``/``Duplicate``/``Div`` per row
                # (``:254-261``); the semantic square/reduce/scale/mul
                # inventory covers two of those passes, so publish the
                # remainder as lowered reduction work.  The output phase runs
                # two full-length passes (``BroadCast`` then ``Mul``,
                # ``:274-282``) in the general branch and one
                # (``repeatByRow``, ``:269``) in the is_performance branch.
                performance_flag = int(
                    parameters["norm_merge_n_performance_flag"])
                num_col_align = _cann_rmsnorm_num_col_align(hidden)
                gamma_operations = work_items * hidden
                if not performance_flag:
                    gamma_operations += (
                        work_items * merge_n_row_factor * num_col_align)
                parameters["lowered_vector_input_operations"] = float(
                    elements + gamma_operations)
                parameters["lowered_vector_reduce_operations"] = float(
                    (1 if not performance_flag else 2) * rows)
                parameters["lowered_vector_output_operations"] = float(
                    (2 if not performance_flag else 1) * elements)
                input_passes = None
                reduce_passes = None
                output_passes = None
            else:
                input_passes = 1
                reduce_passes = 0
                output_passes = (
                    3 if hidden > _CANN_8_5_1_NORM_FULL_ROW_UB_ELEMENTS
                    else 2)
        elif execution_variant == "add_rmsnorm":
            # ``KernelAddRmsNormSplitD::SubProcess`` initializes its
            # depth-one rstd output tensor with ``Duplicate(0.0, rowCount)``
            # before issuing the Former loop.  It is one Vector operation per
            # logical row, independent of the 8,192-column tile count.  Keep
            # it in the reduction lowering so the aggregate contract exactly
            # conserves the separately staged accumulator initialization.
            rstd_accumulator_initialization_operations = 0
            if hidden > _CANN_8_5_1_NORM_FULL_ROW_UB_ELEMENTS:
                tiling_key = 31
                input_passes = 3
                reduce_passes = 1
                output_passes = 5
                rstd_accumulator_initialization_operations = rows
                # CANN 8.5.1 key 31 binds AddRmsNormSplitD.  Its fixed
                # 8,192-column-tile / at-most-64-row program has a distinct
                # former/reduce/latter queue lifetime from the full-row key
                # 30 implementation, so make the source identity explicit.
                parameters[
                    "source_bound_add_rmsnorm_split_d_dag"] = 1.0
            elif rows <= vector_cores:
                tiling_key = 33
                input_passes = 3
                reduce_passes = 3
                output_passes = 4
                # CANN 8.5.1 dispatch binds key 33 exclusively to
                # KernelAddRmsNormSingleN<bfloat16_t>.  That class fixes one
                # row per AIV and has a source-closed single-TBuf event DAG.
                # Publish a discrete source identity so the predictor cannot
                # borrow that DAG for another key or implementation.
                parameters[
                    "source_bound_add_rmsnorm_single_n_dag"] = 1.0
            else:
                if hidden < _CANN_8_5_1_ADD_RMS_KEY30_MIN_HIDDEN:
                    raise ValueError(
                        "CANN 8.5.1 AddRmsNorm with more than 48 rows and "
                        "hidden below 2001 selects an unsupported MergeN/"
                        "MultiN implementation")
                tiling_key = 30
                input_passes = 3
                reduce_passes = 1
                output_passes = 4
                # Key 30 binds the full-row BF16 KernelAddRmsNorm source.  It
                # reuses one gamma vector per AIV and processes rows in
                # source-defined groups of at most 64 with depth-one output
                # queues.  Publish only this discrete implementation identity;
                # the predictor reconstructs the source program analytically.
                parameters[
                    "source_bound_add_rmsnorm_row_group_dag"] = 1.0
        else:
            raise AssertionError(
                f"unexpected CANN normalization variant: "
                f"{execution_variant!r}")
        if input_passes is not None:
            parameters["lowered_vector_input_operations"] = float(
                input_passes * elements)
            if reduce_passes > 0:
                parameters["lowered_vector_reduce_operations"] = float(
                    reduce_passes * elements
                    + rstd_accumulator_initialization_operations)
            parameters["lowered_vector_output_operations"] = float(
                output_passes * elements)
        parameters["cann_tiling_key"] = float(tiling_key)
    elif execution_variant == "qkv_rmsnorm_rope":
        (seq_len, num_query_heads, num_kv_heads, head_dim,
         rotary_dim) = require_shape(
            shape,
            ("seq_len", "num_query_heads", "num_kv_heads", "head_dim",
             "rotary_dim"))
        if not attributes["is_neox"]:
            raise NotImplementedError(
                "vLLM Ascend split-QKV RMSNorm RoPE contract currently "
                "requires NeoX rotary layout")
        if rotary_dim > head_dim or rotary_dim % 2:
            raise ValueError(
                "vLLM Ascend split-QKV RMSNorm RoPE requires an even "
                "rotary_dim no larger than head_dim")
        q_hidden = num_query_heads * head_dim
        kv_hidden = num_kv_heads * head_dim
        # Qwen's Q/K-Norm fusion calls split_qkv_rmsnorm_rope, whose pinned
        # source has no Triton autotuner.  It launches every available AIV and
        # derives both row tiles directly from its fixed 85-KiB UB budget.
        # split_qkv_tp_rmsnorm_rope is a separate MiniMax-only operator with a
        # BLOCK autotuner and must not be substituted for this program.
        partial_rope = rotary_dim != head_dim
        if partial_rope:
            ub_factor = (
                5 * q_hidden + 3 * kv_hidden + 4 * rotary_dim
                + num_query_heads * rotary_dim)
        else:
            ub_factor = (
                5 * q_hidden + 3 * kv_hidden + 2 * rotary_dim
                + num_query_heads * rotary_dim // 2)
        qk_rows_per_tile = max(
            1,
            (_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_UB_BYTES // 2)
            // ub_factor,
        )
        v_rows_per_tile = max(
            1,
            (_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_UB_BYTES // 2)
            // (kv_hidden + 1),
        )
        parameters.update({
            SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER: 1.0,
            "qk_batch_rows_per_tile": float(qk_rows_per_tile),
            "v_batch_rows_per_tile": float(v_rows_per_tile),
            "source_ub_bytes": float(
                _VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_UB_BYTES),
        })
        # The vLLM-Ascend QKNormRope pattern-matching pass rewrites this
        # pattern at torchair compile time; capture-limit fallback (NONE)
        # steps execute the codegen'd fused fx graph rather than an eager
        # three-kernel sequence.  msprof serving traces on ascend_910b show
        # one split_qkv_rmsnorm_rope_kernel per layer in both lanes and
        # never a standalone [seq*heads, head_dim] RmsNorm launch, so the
        # fused parameters above are the only execution contract this
        # variant needs for both compiled and fallback steps.
        work_items = vector_cores
        exact_cores = vector_cores
    elif execution_variant in ("rope", "rotary"):
        seq_len, num_query_heads, num_kv_heads, head_dim = require_shape(
            shape,
            ("seq_len", "num_query_heads", "num_kv_heads", "head_dim"))
        tokens_per_core = (seq_len + vector_cores - 1) // vector_cores
        work_items = (seq_len + tokens_per_core - 1) // tokens_per_core
        # ATB's public BF16 RoPE source selects key 24 only for the aligned
        # >=64-token program; every other two-dimensional-cache BF16 case uses
        # key 22.  The vLLM caller supplies NeoX full-dimension RoPE through
        # rotaryCoeff=2 and cosFormat=0.  In the aligned domain used by the
        # production cases, key 22 owns 4096-element Q/K slices and key 24
        # halves that capacity to 2048 elements for its two ping-pong buffers.
        # These are static host-tiler/kernel-source facts from
        # ascend-transformer-boost f10dc20391d8af04c3f8024aca0e0dfa94d2a0a5;
        # no workload duration participates in this selection or work count.
        aligned_rotary = (
            head_dim % (_CANN_8_5_1_ROPE_ROTARY_COEFFICIENT * 16) == 0)
        tiling_key = (
            24
            if (seq_len >= _CANN_8_5_1_ROPE_LARGE_TOKEN_THRESHOLD
                and aligned_rotary)
            else 22)
        parameters["cann_tiling_key"] = float(tiling_key)
        full_dimension_neox = execution_variant == "rotary"
        if execution_variant == "rope":
            full_dimension_neox = (
                shape["rotary_dim"] == head_dim
                and attributes["is_neox"]
                and attributes["key_present"])
        if aligned_rotary and full_dimension_neox:
            nominal_slice = (
                _CANN_8_5_1_ROPE_BF16_ALIGN_SLICE_ELEMENTS
                if tiling_key == 24
                else _CANN_8_5_1_ROPE_BF16_SLICE_ELEMENTS)
            slice_elements = (nominal_slice // head_dim) * head_dim
            q_elements = num_query_heads * head_dim
            k_elements = num_kv_heads * head_dim
            if (slice_elements > 0
                    and q_elements % _CANN_8_5_1_ROPE_VECTOR_REPEAT_ELEMENTS == 0
                    and k_elements % _CANN_8_5_1_ROPE_VECTOR_REPEAT_ELEMENTS == 0):
                total_elements = seq_len * (q_elements + k_elements)
                cos_sin_cast_elements = (
                    2 * seq_len * min(q_elements, slice_elements))
                base_tokens_per_worker, worker_remainder = divmod(
                    seq_len, work_items)
                second_mask_workers = (
                    work_items
                    if tiling_key == 24 and base_tokens_per_worker > 1
                    else (
                        worker_remainder
                        if tiling_key == 24
                        and base_tokens_per_worker == 1
                        else 0))
                scalar_mask_writes = head_dim * (
                    work_items + second_mask_workers)
                query_heads_per_slice = min(
                    q_elements, slice_elements) // head_dim
                mask_copy_payload_bytes = head_dim * 4
                mask_copy_operations = (
                    (work_items + second_mask_workers)
                    * (query_heads_per_slice - 1)
                    * mask_copy_payload_bytes)
                cos_sin_copy_payload_bytes = head_dim * 2
                cos_sin_copy_operations = (
                    seq_len * 2 * (query_heads_per_slice - 1)
                    * cos_sin_copy_payload_bytes)
                # QkComm/QkCommBF16 issue two UB-to-UB half-rotation
                # DataCopy requests for every Q/K slice.  Across all slices,
                # each request copies one byte per BF16 source element.
                half_rotation_copy_operations = (
                    2 * seq_len * (q_elements + k_elements))
                local_copy_operations = (
                    mask_copy_operations
                    + cos_sin_copy_operations
                    + half_rotation_copy_operations)
                # The common semantic model already carries three arithmetic
                # operations per Q/K element.  ATB's aligned BF16 lowering is
                # 3 BF16->FP32 casts, 3 Mul + 1 Add, and one FP32->BF16 cast
                # per Q/K element, plus two cache-row casts per token.  Each
                # worker also builds one scalar mask, and key 24 builds its
                # second ping-pong mask exactly when that worker owns more
                # than one token.  Publish only source-counted work beyond
                # the three common arithmetic ops.
                parameters["lowered_vector_operations"] = float(
                    5 * total_elements + cos_sin_cast_elements
                    + scalar_mask_writes + local_copy_operations)
                parameters["source_bound_atb_bf16_rope_dag"] = 1.0
                parameters["atb_rope_slice_elements"] = float(
                    slice_elements)
                parameters["atb_rope_rotary_coefficient"] = float(
                    _CANN_8_5_1_ROPE_ROTARY_COEFFICIENT)
                parameters["atb_rope_cos_format"] = float(
                    _CANN_8_5_1_ROPE_COS_FORMAT)
                # torch_npu's public in-place RoPE entry selects the two
                # cache rows before invoking AtbRopeKernel. The two GatherV3
                # launches have identical [T, D] BF16 payloads; this is a
                # source-level call-graph fact, not a duration correction for
                # the ATB kernel.
                parameters[
                    SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER] = 2.0
                gather_tiling = select_cann_public_rope_gather_v3_tiling(
                    seq_len, head_dim, 2, vector_cores)
                parameters.update({
                    SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER: float(
                        gather_tiling["tiling_key"]),
                    "cann_rope_gather_v3_real_core_num": float(
                        gather_tiling["real_core_num"]),
                    "cann_rope_gather_v3_selected_row_bytes": float(
                        gather_tiling["selected_row_bytes"]),
                    "cann_rope_gather_v3_tiling_payload_bytes": float(
                        gather_tiling["tiling_payload_bytes"]),
                })
    elif execution_variant in ("kv_cache", "kvcache"):
        require_paged_kv_cache_contract(attributes)
        seq_len, num_kv_heads, head_dim = require_shape(
            shape, ("seq_len", "num_kv_heads", "head_dim"))
        # CANN 8.5.1 ReshapeAndCache host tiler:
        #
        # * key 200000000 is selected only for <=999 tokens when both the K
        #   and V token extents are 32-byte aligned and have equal head size;
        # * key 201000000 is the unequal-K/V-head-size specialization, which
        #   is unrepresentable by this contract's single ``head_dim``;
        # * every other supported shape selects key 202000000.
        #
        # The production contract is BF16-only here, so one token's K or V
        # payload is exactly num_kv_heads * head_dim * 2 bytes.  These are
        # source-level host-dispatch facts, not duration-derived thresholds.
        token_bytes = num_kv_heads * head_dim * 2
        if (seq_len <= _CANN_8_5_1_RESHAPE_CACHE_FAST_TOKEN_LIMIT
                and token_bytes
                % _CANN_8_5_1_RESHAPE_CACHE_BLOCK_BYTES == 0):
            tiling_key = _CANN_8_5_1_RESHAPE_CACHE_EQUAL_KEY
        else:
            tiling_key = _CANN_8_5_1_RESHAPE_CACHE_FALLBACK_KEY
        parameters["cann_tiling_key"] = float(tiling_key)
        work_items = min(seq_len, vector_cores)
    elif execution_variant in ("activation", "silu"):
        rows, columns, work_items = require_shape(
            shape, ("tok", "intermediate", "element_count"))
        if work_items != rows * columns:
            raise ValueError(
                "CANN SwiGLU element_count must equal tok * intermediate")
        # The fixed CANN 8.5.1 BF16 SwiGlu key 27 source executes two
        # BF16->FP32 input casts and one FP32->BF16 output cast in addition to
        # the common mathematical SiLU-and-mul demand.  Muls(-1) is already
        # one of the common predictor's three generic arithmetic operations,
        # so counting it again here would duplicate source work.  Publish
        # only the three source-proven cast passes.
        # The same selected-key chain proves one FP32 Exp and one FP32 Div
        # per element.  The public host tiler defines both the tile count and
        # blockDim, closing the critical-AIV intrinsic count consumed by the
        # typed service contract.  The larger retained aiv_vec_fops total
        # remains validation evidence, not a fitted coefficient for control
        # work.
        launch_fact, source_parameters = _cann_swiglu_source_parameters(
            rows, columns, vector_cores)
        parameters.update(source_parameters)
        total_tiles = launch_fact["total_tiles"]
        exact_cores = launch_fact["block_dim"]
        if ("injected_grid_size" in attributes
                and attributes["injected_grid_size"] != exact_cores):
            raise ValueError(
                "injected Ascend 910B SwiGLU grid size disagrees with "
                "the CANN 8.5.0 source tiler")
        work_items = total_tiles
        if exact_cores > vector_cores:
            raise ValueError(
                "source-derived Ascend 910B SwiGLU block_dim exceeds HWSpec "
                "AIVs")
    elif execution_variant == "elementwise":
        assert elementwise_work_items is not None
        work_items = elementwise_work_items
        if elementwise_launch_facts is not None:
            exact_cores = elementwise_launch_facts["block_dim"]
            if ("injected_grid_size" in attributes
                    and attributes["injected_grid_size"] != exact_cores):
                raise ValueError(
                    "injected Ascend 910B elementwise grid size disagrees "
                    "with the CANN 8.5.1 TBE tiler")
    elif execution_variant.startswith("moe_"):
        tokens, experts, topk, hidden, intermediate = require_shape(
            shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
        if execution_variant == "moe_swiglu":
                rows = require_positive_integer(attributes, "routed_rows")
                if rows > tokens * topk:
                    raise ValueError(
                        "Ascend 910B MoE SwiGLU routed_rows cannot exceed "
                        "tokens * topk")
                work_items = rows * intermediate
                # FusedMoe dispatches the same SwiGlu_3_high_performance_27 AIV
                # program as the standalone activation.  Its logical leading
                # dimension is the routed-row count rather than the input-token
                # count; the public host tiler otherwise consumes the identical
                # [rows, intermediate] matrix.
                launch_fact, source_parameters = _cann_swiglu_source_parameters(
                    rows, intermediate, vector_cores)
                parameters.update(source_parameters)
                total_tiles = launch_fact["total_tiles"]
                exact_cores = launch_fact["block_dim"]
                if ("injected_grid_size" in attributes
                        and attributes["injected_grid_size"] != exact_cores):
                    raise ValueError(
                        "injected Ascend 910B MoE SwiGLU grid size disagrees "
                        "with the CANN 8.5.0 source tiler")
        elif execution_variant in _CANN_MOE_INIT_ROUTING_VARIANTS:
            # CANN 8.5.1 MoeInitRouting host tiler assigns blockDim to the
            # platform AIV count unconditionally.  Its installed key-zero
            # manifest is a 0 AIC : 1 AIV mixed task, rather than an ordinary
            # AIV-only launch; preserve that source fact for the task-floor
            # contract selected by the predictor.
            work_items = tokens * topk
            exact_cores = vector_cores
            parameters["cann_moe_init_routing_mix_aiv_1_0"] = 1.0
        elif execution_variant == "moe_finalize_routing":
            # MoeFinalizeRoutingV2 host tiler partitions output rows using
            # normalCoreHandleNum=ceil(tokens/AIVs), then launches only the
            # cores which own at least one row.  FusedMoe's BF16 no-bias
            # entry point selects the installed CANN 8.5.1 key-20036
            # program; expose the program identity so the predictor can
            # validate its source-specific tensor ABI and arithmetic.
            work_items = tokens * topk
            rows_per_core = ceil_div(tokens, vector_cores)
            exact_cores = min(vector_cores, ceil_div(tokens, rows_per_core))
            parameters["cann_tiling_key"] = float(
                _CANN_MOE_FINALIZE_ROUTING_BF16_NO_BIAS_TILING_KEY)
        elif execution_variant == "moe_compute_expert_tokens":
            routed_items = tokens * topk
            tiling_key = select_moe_compute_expert_tokens_tiling_key(
                routed_items, experts)
            parameters["cann_tiling_key"] = float(tiling_key)
            # The installed MoeComputeExpertTokens entry wraps all Int32SS/M/L
            # programs in KERNEL_TYPE_MIX_AIV_1_0.  Preserve this launch fact
            # separately from its full-AIV blockDim so Forward prices the
            # matching complete mixed-task topology.
            parameters["cann_moe_compute_expert_tokens_mix_aiv_1_0"] = 1.0
            # The host tiler launches every physical AIV for all three
            # programs.  Individual phases may use fewer cores internally,
            # but that must not alter the runtime blockDim.
            exact_cores = vector_cores
            work_items = routed_items + experts
        elif execution_variant == "moe_range":
            work_items = tokens * topk + experts
            # The retained dynamic Range program takes running_core_num from
            # its CANN runtime tiling payload; CANN 8.5.1 supplies all AIVs.
            exact_cores = vector_cores
        elif execution_variant == "moe_transpose":
            work_items = tokens * topk
            # The retained dynamic Transpose program likewise launches the
            # platform core count and gates tail work inside the kernel.
            exact_cores = vector_cores
        elif execution_variant == "moe_cast":
            cast_role = require_non_empty_string(attributes, "cast_role")
            if cast_role not in _CANN_MOE_CAST_ROLES:
                raise ValueError(
                    "Ascend 910B moe_cast cast_role must be one of "
                    f"{list(_CANN_MOE_CAST_ROLES)}, got {cast_role!r}")
            if cast_role == "topk_ids":
                work_items = tokens * topk
            elif cast_role == "expert_tokens":
                work_items = experts
            else:
                raise AssertionError(f"unhandled moe_cast role: {cast_role!r}")
            # Dynamic Cast is a pure-elementwise TBE graph.  Its source
            # AutoTiling uses the same 1024-element minimum block and
            # 128-element aligned equal partition as the common elementwise
            # schedule already encoded above.
            cast_tiling = select_cann_elementwise_tiling(
                work_items, "add", vector_cores)
            if cast_tiling is None:
                raise NotImplementedError(
                    "Ascend 910B moe_cast exceeds the source-backed CANN "
                    "elementwise tiling domain")
            exact_cores = cast_tiling["block_dim"]
        elif execution_variant == "moe_gating_topk":
            # MoeGatingTopK's host tiler partitions the T input rows across
            # AIVs: per_core_rows=ceil(T/AIVs), blockDim=ceil(T/rows).
            rows_per_core = ceil_div(tokens, vector_cores)
            work_items = tokens * experts
            exact_cores = ceil_div(tokens, rows_per_core)
        elif execution_variant == "moe_abs":
            # CANN 8.5.1 Abs<int32> is an AIV-only ElementwiseSch program.
            # Its public host payload always applies the 1024-item minimum;
            # unlike generic Add it has no <1536 single-worker branch.
            work_items = tokens * topk
            abs_tiling = select_cann_moe_abs_tiling(
                work_items, vector_cores)
            exact_cores = abs_tiling["block_dim"]
        elif execution_variant == "moe_token_unpermute":
            # MoeTokenUnpermute host tiler sets usedCoreNum=min(T,AIVs).
            work_items = tokens * topk * hidden
            exact_cores = min(tokens, vector_cores)
        elif execution_variant in (
                "moe_topk", "moe_softmax", "moe_gating"):
            work_items = tokens * topk
        else:
            raise ValueError(
                f"unhandled CANN MoE auxiliary launch: {execution_variant!r}")
    else:
        work_items, = require_shape(shape, ("element_count",))
    if execution_variant in (
            "rmsnorm", "add_rmsnorm", "rope", "rotary",
            "kv_cache", "kvcache"):
        cores = work_items
    elif execution_variant in (
            "qkv_rmsnorm_rope",
            "elementwise", "activation", "silu",
            "reduce", "gather", "softmax", "gemm_reduce",
            "gemm_elementwise", "moe_swiglu", *_CANN_MOE_INIT_ROUTING_VARIANTS,
            "moe_finalize_routing", "moe_compute_expert_tokens",
            "moe_transpose", "moe_cast", "moe_range", "moe_topk",
            "moe_softmax", "moe_gating", "moe_gating_topk", "moe_abs",
            "moe_token_unpermute"):
        cores = (
            exact_cores
            if exact_cores is not None
            else min(vector_cores, work_items))
    else:
        raise ValueError(
            f"unhandled CANN Vector core mapping: {execution_variant!r}")
    return LaunchConfig(
        algorithm=(
            "vllm_ascend_split_qkv_rmsnorm_rope"
            if execution_variant == "qkv_rmsnorm_rope"
            else f"aclnn_{execution_variant}"),
        grid=(cores, 1, 1),
        block=(1, 1, 1),
        parameters=parameters,
        execution_variant=execution_variant,
        source_program=source_program,
    )
