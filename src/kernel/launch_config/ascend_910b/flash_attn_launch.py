"""Huawei 910B fused-attention AIC/AIV launch and source schedule."""

from dataclasses import dataclass
import math
import struct
from typing import Mapping, Optional

from src.kernel.launch_config import (
    FiaCrossDomainAicGroupProgram,
    FiaCrossDomainLaunchProgram,
    FiaCrossDomainTask,
    LaunchConfig,
    TilingSpec,
)
from src.kernel.launch_config.flash_attn_workload import (
    flash_attn_workload_key,
)
from src.kernel.launch_config.selection_primitives import (
    launch_policy, require_shape, require_variant)
from src.kernel.launch_config import hardware_skeleton as _skeleton


_SPLIT_FUSE_RUNTIME_VARIANTS = {
    "5000000000000200203",
    "5000000000010200203",
}
_SPLIT_FUSE_TILE_SHAPE = {"q": 128, "kv_stack_ceiling": 512}
_SOURCE_SCHEDULE_FIELDS = {"tile", "worker_endpoints", "flash_decode"}
_CANN_FIA_AIC_COUNT = 24
_CANN_FIA_AIV_PER_AIC = 2
_CANN_FIA_PAGE_SIZE = 128
_CANN_FIA_PAGED_LAYOUTS = {
    "paged", "paged_varlen", "chunked_prefill",
}
_CANN_FIA_REGULAR_LAYOUTS = {
    "contiguous", "paged", "paged_varlen",
}
_CANN_FIA_SPLIT_FUSE_LAYOUTS = {
    "contiguous", "paged", "paged_varlen", "chunked_prefill",
}
_MLA_VARIANTS = {
    "105000000000222220",
    "105000000000322220",
}
_STANDARD_GQA_SOFTMAX_WITH_BRC = {
    "103000000010022221": False,
    "103000000010222221": False,
    "103000000010422221": True,
    "103000000010522221": True,
    "103000000010622221": True,
    "103000000010722221": True,
    "105000000000222220": False,
    "105000000000322220": False,
}
_FP32_VECTOR_BUFFER_ELEMENTS = 8192
_FP32_SOFTMAX_WORKSPACE_BYTES = 2048
_FP32_BLOCK_ELEMENTS = 8
_FP32_REPEAT_ELEMENTS = 64
_AIV_INITIALIZATION_GENERIC_OPERATIONS = 2 * 512


# 910B forwards the public FlashAttention attributes unchanged.
normalize_flash_attn_execution_attributes = (
    _skeleton.normalize_flash_attn_execution_attributes)


# CANN FIA exposes no separate setup kernel in this contract.
select_flash_attn_setup_launches = _skeleton.select_flash_attn_setup_launches


# CANN FIA runs one fused call per layer in this contract.
select_flash_attn_forward_partition = (
    _skeleton.select_flash_attn_forward_partition)


def select_flash_attn_copy_launches(
        shape: Mapping[str, int], attributes: Mapping[str, object],
        hardware) -> tuple:
    """CANN FIA emits no separate framework copy in this contract."""

    del shape, attributes, hardware
    return ()


# CANN FIA emits no separate output-buffer fill in this contract.
select_flash_attn_output_fill_launches = (
    _skeleton.select_no_flash_attn_output_fill_launches)


def _require_non_negative_int(value, context, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"910B FlashAttention {context} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"910B FlashAttention {context} must be positive")
    if not positive and value < 0:
        raise ValueError(
            f"910B FlashAttention {context} must be non-negative")
    return value


def _require_int_list(value, context, *, length, positive=False):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(
            f"910B FlashAttention {context} must contain {length} integers")
    return tuple(
        _require_non_negative_int(
            item, f"{context}[{index}]", positive=positive)
        for index, item in enumerate(value)
    )


# CANN 8.5.1 fia_kernel_nonquant.h:91 `static constexpr uint32_t
# PRELOAD_NUM = 2`.  InitWorkspace gives every AIC core a ring this deep for
# vec1Res and vec2Res, indexed by `loop % PRELOAD_NUM`.
_CANN_FIA_PRELOAD_NUM = 2


def _align(value, alignment):
    if value <= 0 or alignment <= 0:
        raise ValueError("910B FlashAttention alignment inputs must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def _ceil_div(value, divisor):
    if value <= 0 or divisor <= 0:
        raise ValueError(
            "910B FlashAttention ceil-div inputs must be positive")
    return (value + divisor - 1) // divisor


def _vector_row_chunks(rows, columns, *, softmax_with_brc):
    """Return source-defined row chunks fitting the 32-KiB FP32 buffer."""

    rows_per_chunk = _FP32_VECTOR_BUFFER_ELEMENTS // columns
    if not softmax_with_brc:
        rows_per_chunk = (
            rows_per_chunk // _FP32_BLOCK_ELEMENTS * _FP32_BLOCK_ELEMENTS)
    if rows_per_chunk <= 0:
        raise ValueError(
            "910B FlashAttention Vector row chunk has no buffer capacity")
    rows_per_chunk = min(rows_per_chunk, rows)
    full_chunks, tail_rows = divmod(rows, rows_per_chunk)
    chunks = [rows_per_chunk] * full_chunks
    if tail_rows:
        chunks.append(tail_rows)
    if not chunks or sum(chunks) != rows:
        raise AssertionError(
            "910B FlashAttention Vector row chunks do not conserve rows")
    return tuple(chunks)


def _row_div_intrinsic_count(rows, actual_columns):
    """Return the literal ``vector_common.h::RowDivs`` Div call count."""

    full_repeats, tail_columns = divmod(
        actual_columns, _FP32_REPEAT_ELEMENTS)
    return min(full_repeats, rows) + int(tail_columns > 0)


def _brcb_physical_operations(rows):
    """Return elements written by the source Brcb repeat geometry."""

    if rows <= 0:
        raise ValueError("Brcb rows must be positive")
    return math.ceil(rows / 8) * 8 * 8


def _mla_vector_lane_rows(rows):
    """Return FiaBlockVecNonQuantMla's exact two-AIV M split."""

    if rows <= 0 or rows > 256:
        raise ValueError("910B MLA Vector rows must be in [1, 256]")
    if rows <= 16:
        return rows, 0
    first_lane = (((rows + 15) // 16 + 1) // 2) * 16
    first_lane = min(first_lane, rows)
    return first_lane, rows - first_lane


def _mla_vec1_lane_operations(rows, actual_s2, score_columns, *,
                              is_first, is_last):
    """Count MLA Vec1 operations executed by one source AIV lane."""

    if rows == 0:
        return (0, 0, 0, 0, 0)
    generic = 0
    exp = 0
    div = 0
    exp_invocations = 0
    div_invocations = 0
    for chunk_rows in _vector_row_chunks(
            rows, score_columns, softmax_with_brc=False):
        chunk_actual = chunk_rows * actual_s2
        chunk_padded = chunk_rows * score_columns
        # SoftmaxFlashV2: two actual/padded generic passes, one score Exp and
        # one row-state Exp.  AmlaVecCompute then performs one extra score
        # RowMuls, one row Exp, two row Divs, and the literal row-state
        # arithmetic/casts around them.
        generic += 2 * chunk_actual + 3 * chunk_padded
        row_generic_passes = 18 + int(is_first) + int(is_last)
        generic += row_generic_passes * chunk_rows
        exp += chunk_padded + 2 * chunk_rows
        div += 2 * chunk_rows
        exp_invocations += 3
        div_invocations += 2
    return generic, exp, div, exp_invocations, div_invocations


def _mla_vec2_lane_operations(rows, output_columns):
    """Count MLA final Vec2 Abs/Compare/Select/RowDiv/Cast work."""

    if rows == 0:
        return (0, 0, 0)
    output_padded = rows * output_columns
    output_actual = rows * 512
    div_invocations = int(math.fsum(
        _row_div_intrinsic_count(chunk_rows, 512)
        for chunk_rows in _vector_row_chunks(
            rows, output_columns, softmax_with_brc=False)))
    return 4 * output_padded, output_actual, div_invocations


@dataclass(frozen=True)
class _CannFiaSplitInput:
    batch: int
    kv_heads: int
    group_size: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    m_base_size: int
    s2_base_size: int


@dataclass
class _CannFiaSplitResult:
    used_core_count: int
    max_cost: int
    b_n2_ends: list[int]
    g_s1_ends: list[int]
    s2_ends: list[int]
    fd_b_n2: list[int]
    fd_g_s1: list[int]
    fd_s2_split_counts: list[int]
    fd_g_s1_split_counts: list[int]
    fd_g_s1_last_sizes: list[int]
    fd_s2_starts: list[int]
    fd_vector_head_ends: list[int]
    fd_vector_split_ends: list[int]
    fd_head_count: int
    fd_max_s2_split_count: int
    fd_used_vector_count: int


def _new_cann_fia_split_result():
    return _CannFiaSplitResult(
        used_core_count=0,
        max_cost=0,
        b_n2_ends=[0] * _CANN_FIA_AIC_COUNT,
        g_s1_ends=[0] * _CANN_FIA_AIC_COUNT,
        s2_ends=[0] * _CANN_FIA_AIC_COUNT,
        fd_b_n2=[0] * _CANN_FIA_AIC_COUNT,
        fd_g_s1=[0] * _CANN_FIA_AIC_COUNT,
        fd_s2_split_counts=[0] * _CANN_FIA_AIC_COUNT,
        fd_g_s1_split_counts=[0] * _CANN_FIA_AIC_COUNT,
        fd_g_s1_last_sizes=[0] * _CANN_FIA_AIC_COUNT,
        fd_s2_starts=[0] * _CANN_FIA_AIC_COUNT,
        fd_vector_head_ends=(
            [0] * (_CANN_FIA_AIC_COUNT * _CANN_FIA_AIV_PER_AIC)),
        fd_vector_split_ends=(
            [0] * (_CANN_FIA_AIC_COUNT * _CANN_FIA_AIV_PER_AIC)),
        fd_head_count=0,
        fd_max_s2_split_count=0,
        fd_used_vector_count=0,
    )


def _cann_fia_block_cost(m_size, s2_size):
    return 6 * _ceil_div(m_size, 16) + 10 * _ceil_div(s2_size, 64)


def _cann_fia_split_core(split_input):
    """Translate CANN ``attention/common/op_host/split_core.cpp``.

    The source is pinned at
    ``cann/ops-transformer@8fff8f9279086404073e2b38e51474792cae9e7b``.
    This path covers the non-empty, unmasked BSH/BNSD decode domain used by
    the 910B Serving integration, including zero-length KV entries that the
    FULL-graph replay ABI zero-fills for padded slots.  It is a discrete
    scheduling rule and does not consume kernel duration or profiler
    counters.
    """

    batch = split_input.batch
    kv_heads = split_input.kv_heads
    group_size = split_input.group_size
    q_lens = split_input.q_lens
    kv_lens = split_input.kv_lens
    m_base_size = split_input.m_base_size
    s2_base_size = split_input.s2_base_size
    if len(q_lens) != batch or len(kv_lens) != batch:
        raise ValueError(
            "910B FlashAttention sequence lengths disagree with batch")
    if kv_heads <= 0 or group_size <= 0:
        raise ValueError(
            "910B FlashAttention KV heads and group size must be positive")

    if any(q_len <= 0 for q_len in q_lens):
        raise ValueError(
            "910B FlashAttention source split requires non-empty Q lengths")
    if any(kv_len < 0 for kv_len in kv_lens):
        raise ValueError(
            "910B FlashAttention source split requires non-negative KV "
            "lengths")
    s1_block_counts = tuple(
        _ceil_div(q_len * group_size, m_base_size) for q_len in q_lens)
    s1_tail_sizes = tuple(
        q_len * group_size % m_base_size for q_len in q_lens)
    # Zero-length KV entries are the padded slots of the FULL-graph replay
    # tiler input: vllm-ascend zero-fills seq_lens past the real requests
    # (model_runner_v1.py:732-734) before update_graph_params re-derives the
    # FIA tiling (attention_v1.py:389), and the NPUGraph A/B ran
    # actual_seq_lengths_kv lists with zero-filled tails on hardware
    # (g3-fa-attribution A/B; artifacts since cleaned).  Such
    # entries contribute zero S2 blocks and zero cost, and the greedy core
    # loop absorbs them through its ``current_b_n2_cost == 0`` branch.
    s2_block_counts = tuple(
        0 if kv_len == 0 else _ceil_div(kv_len, s2_base_size)
        for kv_len in kv_lens)
    s2_tail_sizes = tuple(kv_len % s2_base_size for kv_len in kv_lens)

    def row_cache(batch_index, row_index):
        if row_index >= s1_block_counts[batch_index]:
            return {
                "start": 0,
                "end": 0,
                "blocks": 0,
                "cost": 0,
                "last_cost": 0,
                "normal_cost": 0,
            }
        start = 0
        end = s2_block_counts[batch_index]
        blocks = end - start
        is_m_tail = (
            row_index == s1_block_counts[batch_index] - 1
            and s1_tail_sizes[batch_index] != 0)
        m_size = (
            s1_tail_sizes[batch_index] if is_m_tail else m_base_size)
        normal_cost = _cann_fia_block_cost(m_size, s2_base_size)
        is_s2_tail = (
            s2_tail_sizes[batch_index] != 0
            and end == s2_block_counts[batch_index])
        last_cost = (
            _cann_fia_block_cost(m_size, s2_tail_sizes[batch_index])
            if is_s2_tail else normal_cost)
        return {
            "start": start,
            "end": end,
            "blocks": blocks,
            "cost": (
                normal_cost * (blocks - int(is_s2_tail))
                + (last_cost if is_s2_tail else 0)),
            "last_cost": last_cost,
            "normal_cost": normal_cost,
        }

    b_n2_costs = [0] * batch
    b_n2_blocks = [0] * batch
    b_n2_last_costs = [0] * batch
    for batch_index in range(batch):
        for row_index in range(s1_block_counts[batch_index]):
            cached_row = row_cache(batch_index, row_index)
            b_n2_costs[batch_index] += cached_row["cost"]
            b_n2_blocks[batch_index] += cached_row["blocks"]
            if cached_row["blocks"] > 0:
                b_n2_last_costs[batch_index] = cached_row["last_cost"]
    total_cost = sum(cost * kv_heads for cost in b_n2_costs)
    total_blocks = sum(blocks * kv_heads for blocks in b_n2_blocks)
    if total_blocks <= 0:
        raise AssertionError(
            "910B FlashAttention non-empty source domain has no blocks")

    def calculate_plan(core_count, cost_limit):
        result = _new_cann_fia_split_result()
        current_batch = 0
        current_b_n2 = 0
        current_row = 0
        current_cache = row_cache(current_batch, current_row)
        current_s2 = current_cache["start"]
        remaining_cost = total_cost
        current_b_n2_cost = b_n2_costs[current_batch]
        current_b_n2_blocks = b_n2_blocks[current_batch]
        kv_split_part = 1
        is_finished = False
        current_core_index = 0

        def within(limit, tolerance, value):
            return limit + tolerance >= value

        def update_cursor():
            nonlocal current_batch, current_b_n2, current_row, current_s2
            nonlocal current_cache, current_b_n2_cost, current_b_n2_blocks
            nonlocal is_finished
            update_s1 = False
            update_batch = False
            if current_s2 >= current_cache["end"]:
                current_s2 = 0
                current_row += 1
                update_s1 = True
            if current_row >= s1_block_counts[current_batch]:
                current_row = 0
                current_b_n2 += 1
            if current_b_n2 == batch * kv_heads:
                current_row = 0
                current_s2 = 0
                is_finished = True
                return
            if current_b_n2 // kv_heads != current_batch:
                current_batch = current_b_n2 // kv_heads
                current_row = 0
                update_batch = True
                update_s1 = True
            if update_batch:
                current_b_n2_cost = b_n2_costs[current_batch]
                current_b_n2_blocks = b_n2_blocks[current_batch]
            if update_s1:
                current_cache = row_cache(current_batch, current_row)
                current_s2 = current_cache["start"]

        for core_index in range(core_count):
            if result.max_cost > cost_limit:
                return result
            if is_finished or remaining_cost <= 0:
                break
            current_core_index = core_index
            result.fd_s2_starts[core_index] = kv_split_part - 1
            core_cost = 0
            core_blocks = 0
            core_limit = remaining_cost // (core_count - core_index)

            while (current_b_n2_cost == 0 or within(
                    core_limit, b_n2_last_costs[current_batch] // 2,
                    core_cost + current_b_n2_cost)):
                core_cost += current_b_n2_cost
                core_blocks += current_b_n2_blocks
                current_b_n2 += 1
                if current_b_n2 == batch * kv_heads:
                    current_row = 0
                    current_s2 = 0
                    is_finished = True
                    break
                if current_b_n2 // kv_heads != current_batch:
                    current_batch = current_b_n2 // kv_heads
                current_b_n2_cost = b_n2_costs[current_batch]
                current_b_n2_blocks = b_n2_blocks[current_batch]
                current_row = 0
                current_cache = row_cache(current_batch, current_row)
                current_s2 = current_cache["start"]

            while (not is_finished and within(
                    core_limit, current_cache["last_cost"] // 2,
                    core_cost + current_cache["cost"])):
                core_cost += current_cache["cost"]
                core_blocks += current_cache["blocks"]
                current_b_n2_cost = max(
                    current_b_n2_cost - current_cache["cost"], 0)
                current_b_n2_blocks = max(
                    current_b_n2_blocks - current_cache["blocks"], 0)
                while True:
                    current_row += 1
                    current_cache = row_cache(current_batch, current_row)
                    if current_cache["blocks"] != 0:
                        break
                current_s2 = current_cache["start"]

            if not is_finished:
                block_cost = current_cache["normal_cost"]
                if current_s2 == current_cache["end"] - 1:
                    block_cost = current_cache["last_cost"]
                while within(
                        core_limit, block_cost // 2,
                        core_cost + block_cost):
                    core_cost += block_cost
                    core_blocks += 1
                    current_s2 += 1
                    current_b_n2_cost -= block_cost
                    current_cache["cost"] -= block_cost
                    current_b_n2_blocks -= 1
                    current_cache["blocks"] -= 1

            if not is_finished and core_blocks == 0:
                block_cost = current_cache["normal_cost"]
                if current_s2 == current_cache["end"] - 1:
                    block_cost = current_cache["last_cost"]
                core_cost += block_cost
                core_blocks += 1
                current_s2 += 1
                current_b_n2_cost -= block_cost
                current_b_n2_blocks -= 1
                current_cache["cost"] -= block_cost
                current_cache["blocks"] -= 1
                update_cursor()

            result.b_n2_ends[core_index] = current_b_n2
            result.g_s1_ends[core_index] = current_row
            result.s2_ends[core_index] = current_s2
            result.max_cost = max(result.max_cost, core_cost)
            remaining_cost -= core_cost

            needs_flash_decode = (
                core_index != 0
                and kv_split_part > 1
                and not (
                    current_b_n2 == result.b_n2_ends[core_index - 1]
                    and current_row == result.g_s1_ends[core_index - 1]))
            if needs_flash_decode:
                split_batch = (
                    result.b_n2_ends[core_index - 1] // kv_heads)
                split_row = result.g_s1_ends[core_index - 1]
                fd_m = (
                    q_lens[split_batch] * group_size
                    - split_row * m_base_size
                    if split_row == s1_block_counts[split_batch] - 1
                    else m_base_size)
                fd_g_s1_splits = _ceil_div(fd_m, 8)
                fd_g_s1_last = fd_m - 8 * (fd_g_s1_splits - 1)
                fd_index = result.fd_head_count
                result.fd_max_s2_split_count = max(
                    result.fd_max_s2_split_count, kv_split_part)
                result.fd_b_n2[fd_index] = (
                    result.b_n2_ends[core_index - 1])
                result.fd_g_s1[fd_index] = (
                    result.g_s1_ends[core_index - 1])
                result.fd_s2_split_counts[fd_index] = kv_split_part
                result.fd_g_s1_split_counts[fd_index] = fd_g_s1_splits
                result.fd_g_s1_last_sizes[fd_index] = fd_g_s1_last
                result.fd_head_count += 1
                kv_split_part = 1
            if (current_s2 > current_cache["start"]
                    and current_s2 <= current_cache["end"]):
                kv_split_part += 1

        result.used_core_count = current_core_index + 1
        return result

    max_core_count = min(_CANN_FIA_AIC_COUNT, total_blocks)
    def float32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    float_block_count = float32(float(total_blocks))
    rounded_root = float32(math.sqrt(float32(float_block_count + 0.25)))
    min_core_count = int(float32(rounded_root + 0.5))
    min_core_count = min(min_core_count, max_core_count)
    selected = None
    best_max_cost = (1 << 63) - 1
    for core_count in range(min_core_count, max_core_count + 1):
        candidate = calculate_plan(core_count, best_max_cost)
        if candidate.max_cost < best_max_cost:
            selected = candidate
            best_max_cost = candidate.max_cost
    if selected is None:
        raise AssertionError(
            "910B FlashAttention source selector found no split plan")

    if selected.fd_head_count > 0:
        total_fd_load = sum(
            selected.fd_s2_split_counts[index]
            * selected.fd_g_s1_split_counts[index]
            for index in range(selected.fd_head_count))
        total_fd_head_splits = sum(
            selected.fd_g_s1_split_counts[index]
            for index in range(selected.fd_head_count))
        max_vector_count = min(
            total_fd_head_splits,
            selected.used_core_count * _CANN_FIA_AIV_PER_AIC)
        vector_threshold = total_fd_load / max_vector_count
        vector_load = 0
        vector_index = 0
        previous_head = 0
        previous_split = 0
        for head_index in range(selected.fd_head_count):
            kv_split_count = selected.fd_s2_split_counts[head_index]
            for g_s1_split_index in range(
                    selected.fd_g_s1_split_counts[head_index]):
                remaining_space = vector_threshold - vector_load
                if kv_split_count > remaining_space * 2:
                    selected.fd_vector_head_ends[vector_index] = previous_head
                    selected.fd_vector_split_ends[vector_index] = (
                        previous_split)
                    vector_index += 1
                    total_fd_load -= vector_load
                    vector_threshold = (
                        total_fd_load / (max_vector_count - vector_index))
                    vector_load = 0
                vector_load += kv_split_count
                previous_head = head_index
                previous_split = g_s1_split_index
        selected.fd_vector_head_ends[vector_index] = previous_head
        selected.fd_vector_split_ends[vector_index] = previous_split
        selected.fd_used_vector_count = vector_index + 1
    return selected


def _cann_fia_gqa_tile(kv_len, group_size, *, paged):
    initial_s2_size = (8192, 4096, 2048)[min(group_size // 5, 2)]
    physical_s2_extent = (
        _align(kv_len, _CANN_FIA_PAGE_SIZE) if paged else kv_len)
    s2_base_size = min(initial_s2_size, physical_s2_extent)
    aligned_s2_size = _align(s2_base_size, 16)
    if aligned_s2_size <= 512:
        m_base_size = 512
    elif aligned_s2_size <= 1024:
        m_base_size = 256
    elif aligned_s2_size <= 2048:
        m_base_size = 128
    elif aligned_s2_size <= 4096:
        m_base_size = 64
    else:
        m_base_size = 32
    return m_base_size, s2_base_size


def _cann_fia_gqa_runtime_variant(m_base_size, *, paged, flash_decode):
    template_flags = (
        (4 if m_base_size <= 128 else 0)
        + (2 if paged else 0)
        + int(flash_decode))
    packed_key = (
        1 + 10 * 2 + 100 * 2 + 1000 * 2 + 10000 * 2
        + 100000 * template_flags + 10000000)
    return str(100000000000000000 + 3 * 1000000000000000 + packed_key)


def _cann_fia_mla_runtime_variant(*, flash_decode):
    template_flags = 2 + int(flash_decode)
    packed_key = (
        10 * 2 + 100 * 2 + 1000 * 2 + 10000 * 2
        + 100000 * template_flags)
    return str(100000000000000000 + 5 * 1000000000000000 + packed_key)


def _cann_fia_replay_request_lengths(attributes, q_lens, kv_lens):
    """Return the request lengths the replay-time CANN tiler consumes.

    vLLM-Ascend re-derives the FIA launch at every FULL-graph replay through
    ``update_graph_params`` with the runtime ``seq_lens_list``
    (vllm_ascend/attention/attention_v1.py:389), and the padded slots are
    zero-filled in that list (vllm_ascend/worker/model_runner_v1.py:732-734).
    The greedy core split therefore consumes the logical request lengths
    plus a zero-cost tail, not the captured slot arrays: dummy entries with
    the uniform decode KV length carry positive cost, perturb
    ``core_limit = remaining_cost // (core_count - core_index)``, and
    re-group real requests into a lighter critical group, which prices a
    padded batch faster than its exact capture.  The captured slot arrays
    stay the task-walk and traffic ABI, so the workload key and the source
    schedule accounting are unchanged.
    """

    logical_fields = ("graph_logical_q_lens", "graph_logical_kv_lens")
    present = tuple(name for name in logical_fields if name in attributes)
    if not present:
        return q_lens, kv_lens
    if present != logical_fields:
        raise ValueError(
            "910B FlashAttention graph logical length fields must be "
            f"complete: present={sorted(present)}")
    logical_q_lens = tuple(attributes["graph_logical_q_lens"])
    logical_kv_lens = tuple(attributes["graph_logical_kv_lens"])
    if (len(logical_q_lens) != len(logical_kv_lens)
            or len(logical_q_lens) > len(q_lens)
            or len(logical_kv_lens) > len(kv_lens)):
        raise ValueError(
            "910B FlashAttention graph logical lengths disagree with the "
            "captured slot arrays")
    if (q_lens[:len(logical_q_lens)] != logical_q_lens
            or kv_lens[:len(logical_kv_lens)] != logical_kv_lens):
        raise ValueError(
            "910B FlashAttention graph padding is not the appending slot "
            "shape; the replay tiler input is unavailable")
    return (
        q_lens,
        logical_kv_lens + (0,) * (len(kv_lens) - len(logical_kv_lens)),
    )


def _cann_fia_regular_launch(shape, attributes, kv_layout):
    batch, q_len, kv_len, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    if q_len != 1:
        raise NotImplementedError(
            "Ascend 910B regular FIA source selector requires decode Q=1")
    if query_heads % kv_heads:
        raise ValueError("Ascend 910B FIA requires integral GQA groups")
    group_size = query_heads // kv_heads
    q_lens = tuple(attributes["q_lens"])
    kv_lens = tuple(attributes["kv_lens"])
    split_q_lens, split_kv_lens = _cann_fia_replay_request_lengths(
        attributes, q_lens, kv_lens)
    if kv_layout == "mla":
        if (head_dim, query_heads, kv_heads) != (512, 64, 1):
            raise NotImplementedError(
                "Ascend 910B FIA MLA source selector requires D512/HQ64/HKV1")
        m_base_size = 256
        s2_base_size = min(512, kv_len)
        is_mla = True
        paged = True
    elif kv_layout in _CANN_FIA_REGULAR_LAYOUTS:
        if head_dim not in (64, 128):
            raise NotImplementedError(
                "Ascend 910B FIA GQA source selector supports D64 or D128")
        paged = kv_layout in _CANN_FIA_PAGED_LAYOUTS
        m_base_size, s2_base_size = _cann_fia_gqa_tile(
            kv_len, group_size, paged=paged)
        is_mla = False
    else:
        raise NotImplementedError(
            "Ascend 910B FIA source selector does not support decode "
            f"KV layout {kv_layout!r}")
    result = _cann_fia_split_core(_CannFiaSplitInput(
        batch=batch,
        kv_heads=kv_heads,
        group_size=group_size,
        q_lens=split_q_lens,
        kv_lens=split_kv_lens,
        m_base_size=m_base_size,
        s2_base_size=s2_base_size,
    ))
    has_flash_decode = result.fd_head_count > 0
    runtime_variant = (
        _cann_fia_mla_runtime_variant(flash_decode=has_flash_decode)
        if is_mla else _cann_fia_gqa_runtime_variant(
            m_base_size,
            paged=paged,
            flash_decode=has_flash_decode,
        ))
    schedule = {
        "tile": [m_base_size, s2_base_size],
        "worker_endpoints": [
            [
                result.b_n2_ends[index],
                result.g_s1_ends[index],
                result.s2_ends[index],
            ]
            for index in range(result.used_core_count)
        ],
        "flash_decode": None,
    }
    if has_flash_decode:
        schedule["flash_decode"] = {
            "heads": [
                [
                    result.fd_b_n2[index],
                    result.fd_g_s1[index],
                    result.fd_s2_split_counts[index],
                    result.fd_g_s1_split_counts[index],
                    result.fd_g_s1_last_sizes[index],
                ]
                for index in range(result.fd_head_count)
            ],
            "vector_worker_ends": [
                [
                    result.fd_vector_head_ends[index],
                    result.fd_vector_split_ends[index],
                ]
                for index in range(result.fd_used_vector_count)
            ],
        }
    return {
        "runtime_variant": runtime_variant,
        "block_dim": result.used_core_count,
        "mix_block_dim": (
            _CANN_FIA_AIV_PER_AIC * result.used_core_count),
        "grid": [result.used_core_count, 1, 1],
        "block": [1, 1, 1],
        "source_schedule": schedule,
    }


def _cann_fia_split_fuse_launch(shape, attributes, kv_layout):
    _, _, _, head_dim, query_heads, kv_heads = require_shape(
        shape, (
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"))
    if attributes["causal"] is not True:
        raise NotImplementedError(
            "Ascend 910B SplitFuse source selector requires causal prefill")
    if kv_layout not in _CANN_FIA_SPLIT_FUSE_LAYOUTS:
        raise NotImplementedError(
            "Ascend 910B SplitFuse source selector does not support KV "
            f"layout {kv_layout!r}")
    if query_heads % kv_heads:
        raise ValueError("Ascend 910B SplitFuse requires integral GQA groups")
    if query_heads == kv_heads:
        raise NotImplementedError(
            "Ascend 910B BF16 SplitFuse source route requires GQA")
    if head_dim > 256 or head_dim % 16:
        raise NotImplementedError(
            "Ascend 910B SplitFuse source route requires 16-aligned D<=256")
    paged = kv_layout in _CANN_FIA_PAGED_LAYOUTS
    runtime_variant = str(
        5000000000000000000
        + (10000000 if paged else 0)
        + 200000
        + 3
        + 200)
    return {
        "runtime_variant": runtime_variant,
        "block_dim": _CANN_FIA_AIC_COUNT,
        "mix_block_dim": (
            _CANN_FIA_AIC_COUNT * _CANN_FIA_AIV_PER_AIC),
        "grid": [_CANN_FIA_AIC_COUNT, 1, 1],
        "block": [1, 1, 1],
        "source_schedule": None,
    }


def select_cann_fia_launch(shape, dtype, variant, attributes):
    """Return the CANN ops-transformer 8.5.0 source-selected FIA launch.

    The rules are translated from
    ``cann/ops-transformer@8fff8f9279086404073e2b38e51474792cae9e7b``:
    ``fia_tiling_nonquant.cpp``, ``fia_tiling_nonquant_mla.cpp``,
    ``split_core.cpp``, and ``flash_attention_infer_tiling.h``.  They select
    only discrete tiling keys, AIC/AIV counts, and source schedules.
    """

    execution_variant = require_variant(
        variant, ("prefill", "decode"), "FLASH_ATTN")
    full_split_fuse = ("source_program" in attributes
        and attributes["source_program"] == "ascend_0_18_full_fia_split_fuse")
    if full_split_fuse:
        q_lens, kv_lens = tuple(attributes["q_lens"]), tuple(attributes["kv_lens"])
        real_q, real_kv = tuple(attributes["graph_logical_q_lens"]), tuple(attributes["graph_logical_kv_lens"])
        count = len(real_q)
        fresh = attributes["ascend_fia_prefill_no_cache"]
        dummy = sum(q_lens) != count
        if (attributes["graph_runtime_mode"] != "FULL"
                or type(fresh) is not bool or fresh != (real_q == real_kv)
                or not count or any(type(q) is not int or q <= 0 for q in real_q)
                or len(q_lens) != shape["batch"] or len(kv_lens) != shape["batch"]
                or any(type(q) is not int or q < 0 for q in q_lens)
                or any(type(k) is not int or k < 0 for k in kv_lens)
                or len(q_lens) != count + int(dummy)
                or q_lens[:count] != real_q or kv_lens[:count] != real_kv
                or (dummy and kv_lens[-1] != (q_lens[-1] if fresh else 0))
                or (attributes["kv_layout"] != "contiguous" if fresh
                    else attributes["kv_layout"] not in ("paged", "paged_varlen"))):
            raise ValueError("FULL FIA SplitFuse must preserve real requests and its independent source tail")
        flash_attn_workload_key({**shape, "batch": count, "q_len": max(real_q), "kv_len": max(real_kv)}, dtype, execution_variant,
            {**attributes, "q_lens": real_q, "kv_lens": real_kv})
    else:
        if "source_program" in attributes:
            raise ValueError("unknown or retired CANN FIA source program")
        flash_attn_workload_key(shape, dtype, execution_variant, attributes)
    kv_layout = attributes["kv_layout"]
    if full_split_fuse:
        return _cann_fia_split_fuse_launch(shape, attributes, kv_layout)
    if execution_variant == "decode":
        return _cann_fia_regular_launch(shape, attributes, kv_layout)
    if execution_variant == "prefill":
        return _cann_fia_split_fuse_launch(shape, attributes, kv_layout)
    raise AssertionError(
        f"unhandled Ascend 910B FIA variant: {execution_variant!r}")


def _source_schedule_parameters(
        schedule, shape, attributes, entry, source_program=None):
    """Lower one reflected schedule into SI facts and a typed main program."""

    if not isinstance(schedule, dict) or set(schedule) != _SOURCE_SCHEDULE_FIELDS:
        raise ValueError("910B FlashAttention source-schedule schema changed")
    batch = int(shape["batch"])
    head_dim = int(shape["head_dim"])
    query_heads = int(shape["num_query_heads"])
    kv_heads = int(shape["num_kv_heads"])
    if query_heads % kv_heads:
        raise ValueError("910B FlashAttention requires integral GQA groups")
    group_size = query_heads // kv_heads
    runtime_variant = entry["runtime_variant"]
    is_mla = runtime_variant in _MLA_VARIANTS
    base = {
        "batch": batch,
        "kv_heads": kv_heads,
        "group_size": group_size,
        "q_len": int(shape["q_len"]),
        "kv_len": int(shape["kv_len"]),
        "head_dim": head_dim,
        "used_core_num": entry["block_dim"],
        "output_layout": 4 if is_mla else 0,
        "softmax_lse_flag": 0,
    }
    q_lens_value = attributes["q_lens"]
    kv_lens_value = attributes["kv_lens"]
    if not isinstance(q_lens_value, (list, tuple)):
        raise TypeError("910B FlashAttention q_lens must be a list or tuple")
    if not isinstance(kv_lens_value, (list, tuple)):
        raise TypeError("910B FlashAttention kv_lens must be a list or tuple")
    q_lens = tuple(
        _require_non_negative_int(
            value, f"q_lens[{index}]", positive=True)
        for index, value in enumerate(q_lens_value))
    kv_lens = tuple(
        _require_non_negative_int(
            value, f"kv_lens[{index}]", positive=True)
        for index, value in enumerate(kv_lens_value))
    if len(q_lens) != batch or len(kv_lens) != batch:
        raise ValueError(
            "910B FlashAttention source schedule requires one Q/KV length "
            "per batch item")
    if max(q_lens) > base["q_len"] or max(kv_lens) > base["kv_len"]:
        raise ValueError(
            "910B FlashAttention actual lengths exceed captured tensor extents")

    tile = schedule["tile"]
    if not isinstance(tile, list) or len(tile) != 2:
        raise ValueError(
            "910B FlashAttention source tile must contain M/S2")
    m_base_size = _require_non_negative_int(
        tile[0], "source_schedule.tile[0]", positive=True)
    s2_base_size = _require_non_negative_int(
        tile[1], "source_schedule.tile[1]", positive=True)
    # Every retained standard-GQA payload has one complete query-head group
    # in each S1 task.  This makes K/V a single traversal over the S2 task
    # sequence while Q is replayed once per S2 task.  Keep that source fact at
    # the launch boundary: a future payload that partitions one GQA group over
    # several S1 tasks needs an explicit K/V replay contract before Forward
    # can consume it.
    if m_base_size < group_size:
        raise NotImplementedError(
            "910B FlashAttention source schedule splits one GQA group over "
            "multiple S1 tasks; exact Q/K/V task traffic is unavailable")
    endpoint_values = schedule["worker_endpoints"]
    if (not isinstance(endpoint_values, list)
            or len(endpoint_values) != entry["block_dim"]):
        raise ValueError(
            "910B FlashAttention worker-end schema changed")
    used_core_num = base["used_core_num"]
    endpoints = []
    for worker, endpoint in enumerate(endpoint_values):
        values = _require_int_list(
            endpoint, f"source_schedule.worker_endpoints[{worker}]",
            length=3)
        endpoints.append(tuple(values))
    endpoints = tuple(endpoints)
    previous_endpoint = (0, 0, 0)
    for endpoint in endpoints:
        if endpoint <= previous_endpoint:
            raise ValueError(
                "910B FlashAttention worker endpoints must be strictly "
                "ordered")
        previous_endpoint = endpoint
    if endpoints[-1] != (batch * kv_heads, 0, 0):
        raise ValueError(
            "910B FlashAttention worker endpoints do not cover all heads")

    flash_decode = schedule["flash_decode"]
    is_flash_decode = flash_decode is not None
    flash_decode_head_keys = frozenset()
    expected_flash_decode_variant = (
        _cann_fia_mla_runtime_variant(flash_decode=True)
        if is_mla else _cann_fia_gqa_runtime_variant(
            m_base_size,
            paged=attributes["kv_layout"] in _CANN_FIA_PAGED_LAYOUTS,
            flash_decode=True,
        ))
    if is_flash_decode != (
            entry["runtime_variant"] == expected_flash_decode_variant):
        raise ValueError(
            "910B FlashAttention FlashDecode payload disagrees with variant")
    if is_flash_decode:
        if (not isinstance(flash_decode, dict)
                or set(flash_decode) != {"heads", "vector_worker_ends"}):
            raise ValueError(
                "910B FlashAttention FlashDecode source schema changed")
        heads = flash_decode["heads"]
        vector_worker_ends = flash_decode["vector_worker_ends"]
        if not isinstance(heads, list) or not heads:
            raise ValueError("910B FlashAttention FlashDecode heads are empty")
        if not isinstance(vector_worker_ends, list) or not vector_worker_ends:
            raise ValueError(
                "910B FlashAttention FlashDecode vector assignments are empty")
        decoded_heads = [
            _require_int_list(
                value, f"source_schedule.flash_decode.heads[{index}]",
                length=5)
            for index, value in enumerate(heads)
        ]
        decoded_worker_ends = [
            _require_int_list(
                value,
                f"source_schedule.flash_decode.vector_worker_ends[{index}]",
                length=2)
            for index, value in enumerate(vector_worker_ends)
        ]
        fd_head_count = len(decoded_heads)
        fd_used_vector = len(decoded_worker_ends)
        if fd_used_vector > 2 * used_core_num:
            raise ValueError(
                "910B FlashAttention FlashDecode exceeds AIV launch")
        flash_decode = {
            "used_vector_core_num": fd_used_vector,
            "head_task_count": fd_head_count,
            "g_s1_base_size": 8,
            "head_b_n2": tuple(value[0] for value in decoded_heads),
            "head_g_s1": tuple(value[1] for value in decoded_heads),
            "head_s2_split_count": tuple(value[2] for value in decoded_heads),
            "head_g_s1_split_count": tuple(value[3] for value in decoded_heads),
            "head_g_s1_last_size": tuple(value[4] for value in decoded_heads),
            "vector_worker_head_end": tuple(
                value[0] for value in decoded_worker_ends),
            "vector_worker_head_split_end": tuple(
                value[1] for value in decoded_worker_ends),
        }
        flash_decode_head_keys = frozenset(zip(
            flash_decode["head_b_n2"], flash_decode["head_g_s1"]))
        if len(flash_decode_head_keys) != fd_head_count:
            raise ValueError(
                "910B FlashAttention FlashDecode heads must be unique")

    def advance(coordinate):
        b_n2, g_s1, s2 = coordinate
        if b_n2 >= batch * kv_heads:
            raise ValueError(
                "910B FlashAttention task cursor advanced past BN2 domain")
        batch_index = b_n2 // kv_heads
        g_s1_count = math.ceil(
            q_lens[batch_index] * group_size / m_base_size)
        s2_count = math.ceil(kv_lens[batch_index] / s2_base_size)
        if g_s1 >= g_s1_count or s2 >= s2_count:
            raise ValueError(
                "910B FlashAttention task cursor is outside actual lengths")
        if s2 + 1 < s2_count:
            return b_n2, g_s1, s2 + 1
        if g_s1 + 1 < g_s1_count:
            return b_n2, g_s1 + 1, 0
        return b_n2 + 1, 0, 0

    tasks_by_worker = []
    cursor = (0, 0, 0)
    for worker, endpoint in enumerate(endpoints):
        tasks = []
        while cursor != endpoint:
            if cursor > endpoint:
                raise ValueError(
                    "910B FlashAttention endpoint bisects an invalid cursor")
            b_n2, g_s1, s2 = cursor
            batch_index = b_n2 // kv_heads
            actual_m = min(
                m_base_size,
                q_lens[batch_index] * group_size - g_s1 * m_base_size)
            actual_s2 = min(
                s2_base_size,
                kv_lens[batch_index] - s2 * s2_base_size)
            if actual_m <= 0 or actual_s2 <= 0:
                raise ValueError(
                    "910B FlashAttention source task has no actual work")
            next_cursor = advance(cursor)
            same_head_as_previous = bool(
                tasks and tasks[-1][0:2] == (b_n2, g_s1))
            same_head_as_next = (
                next_cursor != endpoint
                and next_cursor[0:2] == (b_n2, g_s1))
            tasks.append((
                b_n2, g_s1, s2, actual_m, actual_s2,
                not same_head_as_previous, not same_head_as_next,
                (b_n2, g_s1) in flash_decode_head_keys))
            cursor = next_cursor
        if not tasks:
            raise ValueError(
                f"910B FlashAttention worker {worker} has no source tasks")
        tasks_by_worker.append(tuple(tasks))
    reflected_program = FiaCrossDomainLaunchProgram(
        program_kind="cann_fia_1_2_three_handoff_v1",
        aic_groups=tuple(
            FiaCrossDomainAicGroupProgram(
                aic_group=worker,
                aiv_workers=(2 * worker, 2 * worker + 1),
                tasks=tuple(
                    FiaCrossDomainTask(
                        b_n2=task[0],
                        g_s1=task[1],
                        s2=task[2],
                        actual_m=task[3],
                        actual_s2=task[4],
                        is_head_first=task[5],
                        is_head_last=task[6],
                        is_s2_split_core=task[7],
                    )
                    for task in tasks),
            )
            for worker, tasks in enumerate(tasks_by_worker)),
    )
    if source_program is not None:
        if not isinstance(source_program, FiaCrossDomainLaunchProgram):
            raise TypeError(
                "910B FlashAttention source program must be typed")
        if len(source_program.aic_groups) != len(tasks_by_worker):
            raise ValueError(
                "910B FlashAttention source program worker count changed")
        for worker, expected_tasks in enumerate(tasks_by_worker):
            actual_group = source_program.aic_groups[worker]
            actual_tasks = tuple(
                (
                    task.b_n2, task.g_s1, task.s2,
                    task.actual_m, task.actual_s2,
                    task.is_head_first, task.is_head_last,
                    task.is_s2_split_core,
                )
                for task in actual_group.tasks)
            if actual_tasks != expected_tasks:
                raise ValueError(
                    "910B FlashAttention source program disagrees with the "
                    f"reflected cursor for AIC group {worker}")

    tensor_logical_by_worker = []
    tensor_physical_by_worker = []
    if runtime_variant not in _STANDARD_GQA_SOFTMAX_WITH_BRC:
        raise ValueError(
            "910B FlashAttention source schedule has an unsupported Vector "
            f"template {runtime_variant!r}")
    softmax_with_brc = _STANDARD_GQA_SOFTMAX_WITH_BRC[runtime_variant]
    physical_vector_workers = 2 * used_core_num
    generic_by_vector_worker = [
        _AIV_INITIALIZATION_GENERIC_OPERATIONS
    ] * physical_vector_workers
    exp_by_vector_worker = [0] * physical_vector_workers
    div_by_vector_worker = [0] * physical_vector_workers
    exp_invocations_by_vector_worker = [0] * physical_vector_workers
    div_invocations_by_vector_worker = [0] * physical_vector_workers
    vector_vec1_operations = 0
    vector_vec2_operations = 0
    vector_initialization_operations = (
        physical_vector_workers * _AIV_INITIALIZATION_GENERIC_OPERATIONS)
    task_count = 0
    vec1_read_bytes = 0
    mm2_score_read_bytes = 0
    vec2_mm2_read_bytes = 0
    vec2_state_read_bytes = 0
    vec1_write_bytes = 0
    nupdate_write_bytes = 0
    vec2_write_bytes = 0
    main_direct_output_bytes = 0
    flash_decode_partial_write_bytes = 0
    # Distinct AIV-written extent.  InitWorkspace gives each AIC core a
    # PRELOAD_NUM-deep ring for vec1Res and vec2Res and the tasks walk it as
    # loop % PRELOAD_NUM, so a slot's distinct extent is the largest write
    # that lands in it.  FlashDecode's accumOut/lse buffers are not rings and
    # the final output is written once, so both keep their full extent.
    vec1_ring_slot_bytes = []
    vec2_ring_slot_bytes = []
    # The same InitWorkspace ring holds the two AIC Fixpipe destinations,
    # mm1Res and mm2Res, addressed `mm1ResGm[loop % PRELOAD_NUM]`
    # (fia_block_cube_nonquant_gqa.h ComputeMm1) with FixpipeCToGM landing at
    # a sub-block offset inside the slot (:896).  They carry the FP32 score
    # and output tiles the Vector side reads back.
    mm1_ring_slot_bytes = []
    mm2_ring_slot_bytes = []
    partial_counts = {}
    qk_head_dim = head_dim + 64 if is_mla else head_dim
    value_head_dim = head_dim
    for task_worker, tasks in enumerate(tasks_by_worker):
        tensor_logical = 0
        tensor_physical = 0
        vector_worker = 2 * task_worker
        worker_vec1_slots = [0] * _CANN_FIA_PRELOAD_NUM
        worker_vec2_slots = [0] * _CANN_FIA_PRELOAD_NUM
        worker_mm1_slots = [0] * _CANN_FIA_PRELOAD_NUM
        worker_mm2_slots = [0] * _CANN_FIA_PRELOAD_NUM
        # `info.loop` advances once per (task, s2 chunk) pair while
        # `bn2IdxInCurCore` advances only on the first chunk of a head-task
        # (fia_kernel_nonquant_mla.h:498-561).  The MLA cube rotates mm2Res on
        # the head-task counter (fia_block_cube_nonquant_mla.h:999) while the
        # GQA cube rotates both rings on info.loop
        # (fia_block_cube_nonquant_gqa.h:936, :1066).
        worker_head_task_index = -1
        for task_index, (b_n2, g_s1, _s2, actual_m, actual_s2,
                         is_first, is_last, is_s2_split_core) in enumerate(
                             tasks):
            task_count += 1
            task_vec1_ring_bytes = 0
            task_vec2_ring_bytes = 0
            task_mm1_ring_bytes = 0
            task_mm2_ring_bytes = 0
            logical_tensor = (
                2 * actual_m * actual_s2
                * (qk_head_dim + value_head_dim))
            physical_tensor = (
                2 * _align(actual_m, 16) * actual_s2
                * (qk_head_dim + value_head_dim))
            tensor_logical += logical_tensor
            tensor_physical += physical_tensor

            score_actual = actual_m * actual_s2
            score_columns = _align(actual_s2, 32)
            score_padded = actual_m * score_columns
            output_columns = _align(value_head_dim, 32)
            output_actual = actual_m * value_head_dim
            output_padded = actual_m * output_columns
            if is_mla:
                lane_rows = _mla_vector_lane_rows(actual_m)
                for lane, rows in enumerate(lane_rows):
                    (vec1_generic, vec1_exp, vec1_div,
                     vec1_exp_invocations,
                     vec1_div_invocations) = _mla_vec1_lane_operations(
                         rows, actual_s2, score_columns,
                         is_first=is_first, is_last=is_last)
                    worker = vector_worker + lane
                    generic_by_vector_worker[worker] += vec1_generic
                    exp_by_vector_worker[worker] += vec1_exp
                    div_by_vector_worker[worker] += vec1_div
                    exp_invocations_by_vector_worker[worker] += (
                        vec1_exp_invocations)
                    div_invocations_by_vector_worker[worker] += (
                        vec1_div_invocations)
                    vector_vec1_operations += (
                        vec1_generic + vec1_exp + vec1_div)
                    if not is_first and rows:
                        # ProcessAmlaNupdate broadcasts one aligned Mx128
                        # panel and writes it four times into the 512-column
                        # FP32 MM2 workspace under atomic-add mode.
                        nupdate_generic = _align(rows, 16) * 128
                        generic_by_vector_worker[worker] += nupdate_generic
                        vector_vec1_operations += nupdate_generic
                        nupdate_write_bytes += _align(rows, 16) * 512 * 4
                        task_vec1_ring_bytes += (
                            _align(rows, 16) * 512 * 4)
            else:
                vec1_generic = 0
                vec1_exp = 0
                vec1_exp_invocations = 0
                for chunk_rows in _vector_row_chunks(
                        actual_m, score_columns,
                        softmax_with_brc=softmax_with_brc):
                    chunk_actual = chunk_rows * actual_s2
                    chunk_padded = chunk_rows * score_columns
                    # Two semantic online-softmax operations remain on the
                    # generic Vector lane. Source Muls/Cast each consume the
                    # padded score.
                    vec1_generic += 2 * chunk_actual + 2 * chunk_padded
                    # v220 isUpdate=true executes one row-state Exp and one
                    # padded score Exp in every source row chunk.
                    softmax_reduce_rows = (
                        _FP32_SOFTMAX_WORKSPACE_BYTES
                        // (_FP32_BLOCK_ELEMENTS + _FP32_REPEAT_ELEMENTS))
                    if chunk_rows > softmax_reduce_rows:
                        raise ValueError(
                            "910B FlashAttention SoftmaxFlashV2 row chunk "
                            "exceeds the fixed workspace schedule")
                    vec1_exp += chunk_padded + chunk_rows
                    vec1_exp_invocations += 2
                generic_by_vector_worker[vector_worker] += vec1_generic
                exp_by_vector_worker[vector_worker] += vec1_exp
                exp_invocations_by_vector_worker[vector_worker] += (
                    vec1_exp_invocations)
                vector_vec1_operations += vec1_generic + vec1_exp
            vec1_read_bytes += 4 * score_padded
            task_mm1_ring_bytes += 4 * score_padded
            mm2_score_read_bytes += 2 * score_padded
            vec1_write_bytes += 2 * score_padded
            task_vec1_ring_bytes += 2 * score_padded
            if is_mla:
                if is_last:
                    vec2_mm2_read_bytes += 4 * output_padded
                    task_mm2_ring_bytes += 4 * output_padded
                    for lane, rows in enumerate(lane_rows):
                        vec2_generic, vec2_div, div_invocations = (
                            _mla_vec2_lane_operations(rows, output_columns))
                        worker = vector_worker + lane
                        generic_by_vector_worker[worker] += vec2_generic
                        div_by_vector_worker[worker] += vec2_div
                        div_invocations_by_vector_worker[worker] += (
                            div_invocations)
                        vector_vec2_operations += vec2_generic + vec2_div
                    head_key = (b_n2, g_s1)
                    if is_s2_split_core:
                        vec2_write_bytes += actual_m * value_head_dim * 4
                        vec2_write_bytes += actual_m * 8 * 4 * 2
                        flash_decode_partial_write_bytes += (
                            actual_m * value_head_dim * 4
                            + actual_m * 8 * 4 * 2)
                        if head_key in partial_counts:
                            partial_counts[head_key] += 1
                        else:
                            partial_counts[head_key] = 1
                    else:
                        vec2_write_bytes += actual_m * value_head_dim * 2
                        main_direct_output_bytes += (
                            actual_m * value_head_dim * 2)
            else:
                vec2_mm2_read_bytes += 4 * output_padded
                task_mm2_ring_bytes += 4 * output_padded
            if not is_mla and not is_first:
                # RowMuls masks the actual D columns; Add covers the complete
                # padded workspace row.
                merge_generic = output_actual + output_padded
                generic_by_vector_worker[vector_worker] += merge_generic
                vector_vec2_operations += merge_generic
                vec2_state_read_bytes += 4 * output_padded
            if not is_mla and is_last:
                head_key = (b_n2, g_s1)
                if is_s2_split_core:
                    generic_by_vector_worker[vector_worker] += output_padded
                    vector_vec2_operations += output_padded
                    vec2_write_bytes += actual_m * value_head_dim * 4
                    vec2_write_bytes += actual_m * 8 * 4 * 2
                    flash_decode_partial_write_bytes += (
                        actual_m * value_head_dim * 4
                        + actual_m * 8 * 4 * 2)
                    if head_key in partial_counts:
                        partial_counts[head_key] += 1
                    else:
                        partial_counts[head_key] = 1
                else:
                    div_invocations = math.fsum(
                        _row_div_intrinsic_count(
                            chunk_rows, value_head_dim)
                        for chunk_rows in _vector_row_chunks(
                            actual_m, output_columns,
                            softmax_with_brc=softmax_with_brc))
                    if not float(div_invocations).is_integer():
                        raise AssertionError(
                            "910B FlashAttention Div invocation count is not "
                            "integral")
                    div_invocations = int(div_invocations)
                    generic_by_vector_worker[vector_worker] += output_padded
                    div_by_vector_worker[vector_worker] += output_actual
                    div_invocations_by_vector_worker[vector_worker] += (
                        div_invocations)
                    vector_vec2_operations += output_padded + output_actual
                    vec2_write_bytes += actual_m * value_head_dim * 2
                    main_direct_output_bytes += (
                        actual_m * value_head_dim * 2)
            elif not is_mla:
                vec2_write_bytes += 4 * output_padded
                task_vec2_ring_bytes += 4 * output_padded
            if not is_mla:
                brcb_operations = _brcb_physical_operations(actual_m)
                generic_by_vector_worker[vector_worker] += brcb_operations
                vector_vec2_operations += brcb_operations
            if is_first:
                worker_head_task_index += 1
            slot = task_index % _CANN_FIA_PRELOAD_NUM
            mm2_slot = (
                worker_head_task_index % _CANN_FIA_PRELOAD_NUM
                if is_mla else slot)
            if task_vec1_ring_bytes > worker_vec1_slots[slot]:
                worker_vec1_slots[slot] = task_vec1_ring_bytes
            if task_vec2_ring_bytes > worker_vec2_slots[slot]:
                worker_vec2_slots[slot] = task_vec2_ring_bytes
            if task_mm1_ring_bytes > worker_mm1_slots[slot]:
                worker_mm1_slots[slot] = task_mm1_ring_bytes
            if task_mm2_ring_bytes > worker_mm2_slots[mm2_slot]:
                worker_mm2_slots[mm2_slot] = task_mm2_ring_bytes
        vec1_ring_slot_bytes.extend(worker_vec1_slots)
        vec2_ring_slot_bytes.extend(worker_vec2_slots)
        mm1_ring_slot_bytes.extend(worker_mm1_slots)
        mm2_ring_slot_bytes.extend(worker_mm2_slots)
        tensor_logical_by_worker.append(tensor_logical)
        tensor_physical_by_worker.append(tensor_physical)

    terminal_by_worker = [0] * (0 if not is_flash_decode else fd_used_vector)
    flash_decode_terminal_read_bytes = 0
    flash_decode_terminal_write_bytes = 0
    if is_flash_decode:
        head_b_n2 = flash_decode["head_b_n2"]
        head_g_s1 = flash_decode["head_g_s1"]
        head_s2_split_count = flash_decode["head_s2_split_count"]
        head_g_s1_split_count = flash_decode["head_g_s1_split_count"]
        head_g_s1_last_size = flash_decode["head_g_s1_last_size"]
        vector_worker_head_end = flash_decode["vector_worker_head_end"]
        vector_worker_split_end = flash_decode[
            "vector_worker_head_split_end"]
        g_s1_base_size = flash_decode["g_s1_base_size"]
        if set(partial_counts) != flash_decode_head_keys:
            raise ValueError(
                "910B FlashAttention main-pipeline partial heads disagree "
                "with FlashDecode")
        for head_index in range(fd_head_count):
            head_key = (head_b_n2[head_index], head_g_s1[head_index])
            if head_key not in partial_counts:
                raise ValueError(
                    "910B FlashAttention FlashDecode head has no partials")
            if partial_counts[head_key] != head_s2_split_count[head_index]:
                raise ValueError(
                    "910B FlashAttention main-pipeline partial count "
                    "disagrees with FlashDecode")
        previous_head_end = 0
        previous_split_end = -1
        for worker in range(fd_used_vector):
            head_end = vector_worker_head_end[worker]
            split_end = vector_worker_split_end[worker]
            if head_end >= fd_head_count:
                raise ValueError(
                    "910B FlashAttention FlashDecode worker head is invalid")
            for head_index in range(previous_head_end, head_end + 1):
                start_split = (
                    previous_split_end + 1
                    if head_index == previous_head_end else 0)
                last_split = (
                    split_end if head_index == head_end
                    else head_g_s1_split_count[head_index] - 1)
                if start_split > last_split:
                    continue
                if last_split >= head_g_s1_split_count[head_index]:
                    raise ValueError(
                        "910B FlashAttention FlashDecode row split is invalid")
                for row_split in range(start_split, last_split + 1):
                    rows = (
                        head_g_s1_last_size[head_index]
                        if row_split + 1
                        == head_g_s1_split_count[head_index]
                        else g_s1_base_size)
                    partials = head_s2_split_count[head_index]
                    output_elements = rows * value_head_dim
                    terminal_by_worker[worker] += (
                        6 * partials * rows
                        + 2 * partials * output_elements)
                    flash_decode_terminal_read_bytes += (
                        partials * output_elements * 4
                        + partials * rows * 8 * 4 * 2)
                    flash_decode_terminal_write_bytes += (
                        output_elements * 2)
            previous_head_end = head_end
            previous_split_end = split_end

    vector_main = (
        vector_initialization_operations
        + vector_vec1_operations + vector_vec2_operations)
    worker_main = sum(generic_by_vector_worker) + sum(
        exp_by_vector_worker) + sum(div_by_vector_worker)
    if worker_main != vector_main:
        raise ValueError(
            "910B FlashAttention main Vector worker schedule does not "
            "conserve phase work")
    for worker, operations in enumerate(terminal_by_worker):
        generic_by_vector_worker[worker] += operations

    tensor_logical = sum(tensor_logical_by_worker)
    tensor_physical = sum(tensor_physical_by_worker)
    tensor_critical = max(tensor_physical_by_worker)
    vector_terminal = sum(terminal_by_worker)
    vector_by_worker = tuple(
        generic + exp + div
        for generic, exp, div in zip(
            generic_by_vector_worker,
            exp_by_vector_worker,
            div_by_vector_worker))
    vector_critical = max(vector_by_worker)
    if tensor_logical <= 0 or tensor_physical <= 0 or tensor_critical <= 0:
        raise ValueError(
            "910B FlashAttention source Tensor schedule is empty")
    if vector_main <= 0 or vector_critical <= 0:
        raise ValueError(
            "910B FlashAttention source Vector schedule is empty")
    tensor_shape_efficiency = tensor_logical / tensor_physical
    tensor_schedule_efficiency = (
        tensor_physical / (used_core_num * tensor_critical))
    vector_operations = sum(vector_by_worker)
    if vector_operations != vector_main + vector_terminal:
        raise ValueError(
            "910B FlashAttention Vector worker schedule does not conserve "
            "aggregate work")
    vector_schedule_efficiency = (
        vector_operations / (physical_vector_workers * vector_critical))
    for name, value in (
            ("tensor_shape_efficiency", tensor_shape_efficiency),
            ("tensor_schedule_efficiency", tensor_schedule_efficiency),
            ("vector_schedule_efficiency", vector_schedule_efficiency)):
        if not 0 < value <= 1:
            raise ValueError(
                f"910B FlashAttention source {name} is outside (0, 1]")

    # FIA initializes every physical output row first (InitOutputSingleCore),
    # but only nonempty KV requests reach the score/value task body. An empty
    # replay slot is complete after initialization; it has no second write.
    output_extent_bytes = sum(q_lens) * query_heads * value_head_dim * 2
    final_output_bytes = sum(q for q, k in zip(q_lens, kv_lens) if k > 0) * query_heads * value_head_dim * 2
    output_init_bytes = (
        0 if base["output_layout"] in (3, 5) else output_extent_bytes)
    if output_init_bytes == 0 and final_output_bytes != output_extent_bytes:
        raise ValueError("empty FIA slots require complete source output initialization")
    if (main_direct_output_bytes + flash_decode_terminal_write_bytes
            != final_output_bytes):
        raise ValueError(
            "910B FlashAttention direct and terminal final output is "
            "incomplete")
    external_write_bytes = output_init_bytes + final_output_bytes
    # DRAM sees the distinct extent, not the bytes MTE3 moves: an independent
    # probe shows a rewritten GM line costs one drain regardless of how often
    # it is rewritten (gm_write_back_residency_cann_8_5_1.json).  output_init
    # rewrites the output's own addresses, so it adds no distinct extent.
    vec1_unique_write_bytes = sum(vec1_ring_slot_bytes)
    vec2_unique_write_bytes = (
        sum(vec2_ring_slot_bytes) + flash_decode_partial_write_bytes)
    aiv_unique_write_bytes = (
        vec1_unique_write_bytes + vec2_unique_write_bytes
        + max(output_init_bytes, final_output_bytes))
    # The Fixpipe ring obeys the same rule: the probe found L0C->GM and
    # UB->GM drain identically, `write_hit = 1 - slots/passes` on both
    # (gm_write_back_residency_cann_8_5_1.json).
    mm1_unique_write_bytes = sum(mm1_ring_slot_bytes)
    mm2_unique_write_bytes = sum(mm2_ring_slot_bytes)
    aic_unique_write_bytes = (
        mm1_unique_write_bytes + mm2_unique_write_bytes)

    parameters = {
        "cann_source_mm1_unique_write_bytes": float(mm1_unique_write_bytes),
        "cann_source_mm2_unique_write_bytes": float(mm2_unique_write_bytes),
        "cann_source_aic_unique_write_bytes": float(aic_unique_write_bytes),
        "cann_source_m_base_size": float(m_base_size),
        "cann_source_s2_base_size": float(s2_base_size),
        "cann_source_task_count": float(task_count),
        "cann_source_tensor_logical_operations": float(tensor_logical),
        "cann_source_tensor_shape_efficiency": float(
            tensor_shape_efficiency),
        "cann_source_tensor_schedule_efficiency": float(
            tensor_schedule_efficiency),
        "cann_source_vector_operations": float(vector_operations),
        "cann_source_vector_main_operations": float(vector_main),
        "cann_source_vector_initialization_operations": float(
            vector_initialization_operations),
        "cann_source_vector_terminal_operations": float(vector_terminal),
        "cann_source_vector_vec1_operations": float(
            vector_vec1_operations),
        "cann_source_vector_vec2_operations": float(
            vector_vec2_operations),
        "cann_source_vector_worker_count": float(physical_vector_workers),
        "cann_source_vector_schedule_efficiency": float(
            vector_schedule_efficiency),
        "cann_source_vec1_read_bytes": float(vec1_read_bytes),
        "cann_source_mm2_score_read_bytes": float(mm2_score_read_bytes),
        "cann_source_vec2_mm2_read_bytes": float(vec2_mm2_read_bytes),
        "cann_source_vec2_state_read_bytes": float(vec2_state_read_bytes),
        "cann_source_flash_decode_terminal_read_bytes": float(
            flash_decode_terminal_read_bytes),
        "cann_source_vec1_write_bytes": float(vec1_write_bytes),
        "cann_source_nupdate_write_bytes": float(nupdate_write_bytes),
        "cann_source_vec2_write_bytes": float(vec2_write_bytes),
        "cann_source_flash_decode_terminal_write_bytes": float(
            flash_decode_terminal_write_bytes),
        "cann_source_output_init_bytes": float(output_init_bytes),
        "cann_source_external_write_bytes": float(external_write_bytes),
        "cann_source_vec1_unique_write_bytes": float(
            vec1_unique_write_bytes),
        "cann_source_vec2_unique_write_bytes": float(
            vec2_unique_write_bytes),
        "cann_source_aiv_unique_write_bytes": float(aiv_unique_write_bytes),
    }
    worker_fields = (
        ("generic_operations", generic_by_vector_worker),
        ("exp_operations", exp_by_vector_worker),
        ("div_operations", div_by_vector_worker),
        ("exp_invocations", exp_invocations_by_vector_worker),
        ("div_invocations", div_invocations_by_vector_worker),
    )
    for field, values in worker_fields:
        for worker, value in enumerate(values):
            parameters[
                f"cann_source_vector_worker_{worker}_{field}"] = float(value)
    return parameters, (
        reflected_program if source_program is None else source_program)


def _launch_fact(shape, dtype, variant, attributes):
    return select_cann_fia_launch(shape, dtype, variant, attributes)


def select_flash_attn_launch(shape: Mapping[str, int], dtype: Optional[str],
                             variant: Optional[str], attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    execution_variant = require_variant(
        variant, ("prefill", "decode"), "FLASH_ATTN")
    policy = launch_policy(attributes)
    if policy not in ("analytical_launch", "injected_launch"):
        raise AssertionError(f"unhandled launch policy: {policy!r}")
    entry = _launch_fact(shape, dtype, execution_variant, attributes)
    if entry["block_dim"] > int(hardware.num_compute_units):
        raise ValueError(
            "910B FlashAttention launch exceeds physical AIC capacity")
    vector_units = hardware.resource_by_role("compute_vector").total_units
    if entry["mix_block_dim"] > vector_units:
        raise ValueError(
            "910B FlashAttention launch exceeds physical AIV capacity")
    tile_shape = (
        _SPLIT_FUSE_TILE_SHAPE
        if entry["runtime_variant"] in _SPLIT_FUSE_RUNTIME_VARIANTS
        else {}
    )
    parameters = {
        "unpadded_flops": 1.0,
        "mma_smem_bytes_per_flop": 0.0,
        "split_scratch_dram_bytes_per_element": 0.0,
        "page_table_bytes_per_kv_panel": 0.0,
    }
    source_schedule = entry["source_schedule"]
    if source_schedule is None:
        source_program = None
    else:
        source_parameters, source_program = _source_schedule_parameters(
            source_schedule, shape, attributes, entry)
        parameters.update(source_parameters)
    return LaunchConfig(
        algorithm=(
            "cann_flash_attention_score:" + entry["runtime_variant"]),
        grid=tuple(entry["grid"]),
        block=tuple(entry["block"]),
        # The two fixed CANN 8.5.1 SplitFuse templates expose a source-defined
        # 128-row Q block and 512-position KV stack ceiling.  The marker is not
        # a generic Q/KV CTA tile: it activates only the source-issued
        # rectangular MMAD and replay accounting in the common predictor.  The
        # other seven templates still publish no semantic tile contract.
        tiling=TilingSpec(tile_shape=tile_shape, num_kv_splits=1),
        parameters=parameters,
        execution_variant=execution_variant,
        source_program=source_program,
    )
