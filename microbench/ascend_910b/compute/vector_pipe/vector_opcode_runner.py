#!/usr/bin/env python3
"""Run balanced raw Ascend 910B AIV Vector-opcode mechanism trials."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import struct
import sys


DEFAULT_CAMPAIGN = "exp_div_large"
NORM_STEADY_CALIBRATION_CAMPAIGN = "sqrt_div_norm_steady_calibration"
NORM_STEADY_VALIDATION_CAMPAIGN = "sqrt_div_norm_steady_validation"
NORM_FLOOR_CALIBRATION_CAMPAIGN = "sqrt_div_norm_floor_calibration"
NORM_FLOOR_VALIDATION_CAMPAIGN = "sqrt_div_norm_floor_validation"
SWIGLU_SERIALIZED_CAMPAIGN = "swiglu_serialized_services"
SERIALIZED_VECTOR_H8192_CAMPAIGN = "serialized_vector_h8192_validation"
AIV_WORKER_PROLOGUE_CAMPAIGN = "aiv_worker_prologue"
SINGLE_TBUF_INIT_CAMPAIGN = "single_tbuf_init"
SCALAR_INTEGER_DEPENDENCY_CAMPAIGN = "scalar_integer_dependency"
SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN = "scalar_u64_divrem_dependency"
SWIGLU_BUFFER_INIT_CAMPAIGN = "swiglu_buffer_init"
SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN = "swiglu_buffer_lifecycle"
SWIGLU_QUEUE_CONTROL_CAMPAIGN = "swiglu_queue_control"
RMSNORM_QUEUE_CONTROL_CAMPAIGN = "rmsnorm_queue_control"
RMSNORM_BUFFER_READY_CAMPAIGN = "rmsnorm_buffer_ready"
RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN = "rmsnorm_half_interval_reduce"
NORM_REQUEST_SERVICE_CAMPAIGN = "sqrt_div_norm_request_service_r2"
RMSNORM_LOCAL_SCALAR_CAMPAIGN = "rmsnorm_local_scalar_dependency"
ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN = "add_rmsnorm_local_scalar_handoff"
ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN = (
    "add_rmsnorm_split_d_local_scalar_handoff")
ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN = "add_rmsnorm_reduce_sum_fp32"
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN = (
    "add_rmsnorm_split_d_block_reduce_sum_fp32")
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN = (
    "add_rmsnorm_split_d_block_reduce_sum_composed")
VECTOR_TO_MTE2_FENCE_CAMPAIGN = "vector_to_mte2_fence"
LOCAL_SCALAR_FILL_SV_CAMPAIGN = "local_scalar_fill_s_v"
LOCAL_UB_COPY_CAMPAIGN = "local_ub_copy"
MERGE_N_ROW_GROUP_CAMPAIGN = "merge_n_row_group_primitives"
NORM_REQUEST_MERGE_N_CAMPAIGN = "sqrt_div_norm_request_service_merge_n"
FP32_PREDICATE_SELECT_CAMPAIGN = "fp32_predicate_select"
INT32_INT64_CAST_CAMPAIGN = "int32_int64_cast"
FP32_PREDICATE_SELECT_OPCODES = (
    (48, "compare_fp32_gt"),
    (49, "compare_fp32_eq"),
    (50, "compare_fp32_ordered_ne"),
    (51, "select_fp32"),
)
OPCODES = ((0, "exp"), (1, "div"), (2, "exp_div"))
NORM_OPCODES = ((3, "sqrt"), (1, "div"), (4, "sqrt_div"))
SWIGLU_SERIALIZED_OPCODES = (
    (5, "exp_serialized"),
    (6, "div_serialized"),
    (7, "cast_bf16_to_fp32_serialized"),
    (8, "cast_fp32_to_bf16_serialized"),
    (9, "muls_serialized"),
    (10, "adds_serialized"),
    (11, "mul_serialized"),
)
SERIALIZED_VECTOR_H8192_OPCODES = tuple(
    item for item in SWIGLU_SERIALIZED_OPCODES
    if item[1] not in ("exp_serialized", "div_serialized"))
AIV_WORKER_PROLOGUE_OPCODES = (
    (15, "two_tbuf_mask_prologue"),
)
SINGLE_TBUF_INIT_OPCODES = (
    (21, "single_tbuf_init"),
)
SCALAR_INTEGER_DEPENDENCY_OPCODES = (
    (22, "scalar_u32_divrem_dependency"),
    (23, "scalar_u32_mul_add_dependency"),
    (24, "scalar_u64_mul_dependency"),
)
SCALAR_U64_DIVREM_DEPENDENCY_OPCODES = (
    (32, "scalar_u64_divrem_dependency"),
)
SWIGLU_BUFFER_INIT_OPCODES = (
    (25, "swiglu_double_buffer_init"),
    (26, "swiglu_single_buffer_init"),
)
SWIGLU_BUFFER_LIFECYCLE_OPCODES = (
    (28, "swiglu_double_buffer_lifecycle"),
    (29, "swiglu_single_buffer_lifecycle"),
)
SWIGLU_QUEUE_CONTROL_OPCODES = (
    (30, "swiglu_double_buffer_queue_control"),
    (31, "swiglu_single_buffer_queue_control"),
)
RMSNORM_QUEUE_CONTROL_OPCODES = (
    (33, "rmsnorm_four_queue_three_tbuf_lifecycle"),
    (34, "rmsnorm_four_queue_control_chain"),
)
RMSNORM_BUFFER_READY_OPCODES = (
    (35, "rmsnorm_four_queue_three_tbuf_ready"),
)
RMSNORM_QUEUE_CONTROL_ELEMENTS = (2048, 8192, 12288)
RMSNORM_HALF_INTERVAL_REDUCE_OPCODES = (
    (27, "rmsnorm_half_interval_reduce"),
)
RMSNORM_HALF_INTERVAL_REDUCE_ELEMENTS = (
    2048, 2560, 4096, 5120, 8192, 12288)
NORM_REQUEST_OPCODES = (
    (12, "norm_sqrt_serialized"),
    (13, "norm_div_serialized"),
    (14, "norm_sqrt_duplicate_div_serialized"),
)
RMSNORM_LOCAL_SCALAR_OPCODES = (
    (16, "rmsnorm_local_scalar_dependency"),
)
ADD_RMSNORM_LOCAL_SCALAR_OPCODES = (
    (36, "add_rmsnorm_local_scalar_handoff"),
)
ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_OPCODES = (
    (39, "add_rmsnorm_split_d_local_scalar_handoff"),
    (36, "add_rmsnorm_local_scalar_handoff"),
)
ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_BLOCK_DIMS = (1, 32, 47)
ADD_RMSNORM_REDUCE_SUM_FP32_OPCODES = (
    (37, "add_rmsnorm_reduce_sum_fp32"),
)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_OPCODES = (
    (41, "add_rmsnorm_split_d_block_reduce_sum_fp32"),
)
VECTOR_TO_MTE2_FENCE_OPCODES = (
    (38, "vector_to_mte2_fence"),
)
ADD_RMSNORM_REDUCE_SUM_FP32_ELEMENTS = (
    2048, 2560, 4096, 5120, 8192, 12288)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_ELEMENTS = (
    8, 16, 32, 64, 128, 256, 512)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_BLOCK_DIMS = (1, 32, 47)
# CANN 8.5.1 ``BlockReduceSumFP32`` has two static source branches:
# full 64-FP32 repeats and one 8-FP32-aligned tail call.  These independent
# mechanism points calibrate and hold out the branches and their composition;
# they do not represent production operator shapes or durations.
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_ELEMENTS = (
    8, 16, 32, 56, 64, 72, 120, 128, 192, 256, 320, 448, 504, 512)
LOCAL_SCALAR_FILL_SV_OPCODES = (
    (17, "local_scalar_fill_s_v"),
)
MERGE_N_ROW_GROUP_OPCODES = (
    (42, "merge_n_reduce_sum_multi_n"),
    (43, "merge_n_broadcast_gamma"),
    (44, "merge_n_broadcast_rstd"),
    (45, "merge_n_performance_rstd_brcb"),
    (46, "merge_n_performance_mul_row_rstd"),
    (47, "merge_n_performance_mul_row_gamma"),
)
# CANN 8.5.1 RmsNorm MERGE_N row-group primitives are parameterized by the
# row count of one group.  rms_norm_tiling.cpp:543 fixes rowFactor = 84 at the
# production numColAlign of 128, so the sweep spans one row through that
# source bound and does not depend on any production shape.
MERGE_N_ROW_GROUP_ROWS = (1, 2, 4, 8, 16, 32, 64, 84)
MERGE_N_ROW_GROUP_ITERATIONS = 512
# One isolated AIV and the full production AIV population.
MERGE_N_ROW_GROUP_BLOCK_DIMS = (1, 48)
LOCAL_UB_COPY_OPCODES = (
    (18, "local_ub_copy_contiguous"),
    (19, "local_ub_copy_stride_64b"),
    (20, "local_ub_copy_stride_128b"),
)
# The retained key-27 launch payload proves an exact 6,144-element production
# tile.  Keep only its immediately preceding mechanism point and the exact
# production tile in this extension gate; smaller legacy points cannot answer
# whether the production-sized invocation has reached capacity.
ELEMENT_COUNTS = (4096, 6144)
BLOCK_DIMS = (1, 24, 48)
CELLS = tuple(
    (opcode, opcode_name, elements, block_dim)
    for opcode, opcode_name in OPCODES
    for elements in ELEMENT_COUNTS
    for block_dim in BLOCK_DIMS
)
TARGET_ELEMENTS_PER_INTERVAL = 8 * 1024 * 1024
# Ascend 910B SYS_CNT is a fixed 50 MHz platform counter.  Device-cycle
# intervals are converted with this hardware contract; host wall time remains
# raw provenance and is never used to price a Vector service.
SYSTEM_COUNTER_TICK_NS = 20.0
NORM_TARGET_ELEMENTS_PER_INTERVAL = 256 * 1024
NORM_REQUEST_ELEMENTS = tuple(range(1, 65))
# CANN 8.5.1 RmsNorm MERGE_N normalizes a whole row group, so the same
# Sqrt->Duplicate->Div request reaches rowFactor = 84 elements.  Extend the
# measured domain of the identical mechanism past the 64-element point.
NORM_REQUEST_MERGE_N_ELEMENTS = tuple(range(65, 85))
NORM_REQUEST_ITERATIONS = 8192
NORM_REQUEST_TRIALS_PER_RUN = 6
LOCAL_UB_COPY_ITERATIONS = 8192
LOCAL_UB_COPY_ELEMENTS = {
    "local_ub_copy_contiguous": (128, 256, 512),
    "local_ub_copy_stride_64b": (64, 2048, 4096),
    "local_ub_copy_stride_128b": (1024, 2048, 4096),
}
MAX_CELLS_PER_BATCH = 64
CAMPAIGNS = {
    DEFAULT_CAMPAIGN: (
        OPCODES, ELEMENT_COUNTS, TARGET_ELEMENTS_PER_INTERVAL),
    NORM_STEADY_CALIBRATION_CAMPAIGN: (
        NORM_OPCODES, (4096, 6144), TARGET_ELEMENTS_PER_INTERVAL),
    NORM_STEADY_VALIDATION_CAMPAIGN: (
        NORM_OPCODES, (2048,), TARGET_ELEMENTS_PER_INTERVAL),
    NORM_FLOOR_CALIBRATION_CAMPAIGN: (
        NORM_OPCODES, (1,), NORM_TARGET_ELEMENTS_PER_INTERVAL),
    NORM_FLOOR_VALIDATION_CAMPAIGN: (
        NORM_OPCODES, (32, 64), NORM_TARGET_ELEMENTS_PER_INTERVAL),
    SWIGLU_SERIALIZED_CAMPAIGN: (
        SWIGLU_SERIALIZED_OPCODES, (32, 2048, 6144),
        # These are steady affine-service cells, not the one-element NORM
        # floor campaign.  Keep the same interval work as the established
        # Exp/Div steady campaign so the host-wall/SYS_CNT difference is
        # large compared with host scheduling noise at every tile size.
        TARGET_ELEMENTS_PER_INTERVAL),
    SERIALIZED_VECTOR_H8192_CAMPAIGN: (
        SERIALIZED_VECTOR_H8192_OPCODES, (8192,),
        TARGET_ELEMENTS_PER_INTERVAL),
    AIV_WORKER_PROLOGUE_CAMPAIGN: (
        AIV_WORKER_PROLOGUE_OPCODES, (2048, 4096), None),
    SINGLE_TBUF_INIT_CAMPAIGN: (
        SINGLE_TBUF_INIT_OPCODES, (1024, 2048), None),
    SCALAR_INTEGER_DEPENDENCY_CAMPAIGN: (
        SCALAR_INTEGER_DEPENDENCY_OPCODES, (3, 47), None),
    SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN: (
        SCALAR_U64_DIVREM_DEPENDENCY_OPCODES, (3, 47), None),
    SWIGLU_BUFFER_INIT_CAMPAIGN: (
        SWIGLU_BUFFER_INIT_OPCODES, (6144, 9728, 13312), None),
    SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN: (
        SWIGLU_BUFFER_LIFECYCLE_OPCODES, (6144, 9728, 13312), None),
    SWIGLU_QUEUE_CONTROL_CAMPAIGN: (
        SWIGLU_QUEUE_CONTROL_OPCODES, (6144, 9728, 13312), None),
    RMSNORM_QUEUE_CONTROL_CAMPAIGN: (
        RMSNORM_QUEUE_CONTROL_OPCODES,
        RMSNORM_QUEUE_CONTROL_ELEMENTS,
        None),
    RMSNORM_BUFFER_READY_CAMPAIGN: (
        RMSNORM_BUFFER_READY_OPCODES,
        RMSNORM_QUEUE_CONTROL_ELEMENTS,
        None),
    RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN: (
        RMSNORM_HALF_INTERVAL_REDUCE_OPCODES,
        RMSNORM_HALF_INTERVAL_REDUCE_ELEMENTS,
        TARGET_ELEMENTS_PER_INTERVAL),
    NORM_REQUEST_SERVICE_CAMPAIGN: (
        NORM_REQUEST_OPCODES, NORM_REQUEST_ELEMENTS, None),
    RMSNORM_LOCAL_SCALAR_CAMPAIGN: (
        RMSNORM_LOCAL_SCALAR_OPCODES, (1,), None),
    ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN: (
        ADD_RMSNORM_LOCAL_SCALAR_OPCODES, (1,), None),
    ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN: (
        ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_OPCODES, (1,), None),
    ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN: (
        ADD_RMSNORM_REDUCE_SUM_FP32_OPCODES,
        ADD_RMSNORM_REDUCE_SUM_FP32_ELEMENTS,
        TARGET_ELEMENTS_PER_INTERVAL),
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN: (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_OPCODES,
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_ELEMENTS,
        NORM_TARGET_ELEMENTS_PER_INTERVAL),
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN: (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_OPCODES,
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_ELEMENTS,
        NORM_TARGET_ELEMENTS_PER_INTERVAL),
    VECTOR_TO_MTE2_FENCE_CAMPAIGN: (
        VECTOR_TO_MTE2_FENCE_OPCODES, (1,), None),
    LOCAL_SCALAR_FILL_SV_CAMPAIGN: (
        LOCAL_SCALAR_FILL_SV_OPCODES, (64, 128), None),
    LOCAL_UB_COPY_CAMPAIGN: (
        LOCAL_UB_COPY_OPCODES,
        tuple(sorted({
            elements
            for values in LOCAL_UB_COPY_ELEMENTS.values()
            for elements in values
        })),
        None),
    MERGE_N_ROW_GROUP_CAMPAIGN: (
        MERGE_N_ROW_GROUP_OPCODES, MERGE_N_ROW_GROUP_ROWS, None),
    NORM_REQUEST_MERGE_N_CAMPAIGN: (
        NORM_REQUEST_OPCODES, NORM_REQUEST_MERGE_N_ELEMENTS, None),
    FP32_PREDICATE_SELECT_CAMPAIGN: (
        FP32_PREDICATE_SELECT_OPCODES, (64, 512, 4096),
        TARGET_ELEMENTS_PER_INTERVAL),
    INT32_INT64_CAST_CAMPAIGN: (
        ((52, "cast_int32_to_int64"),), (64, 512, 4096),
        TARGET_ELEMENTS_PER_INTERVAL),
}
CAMPAIGN_BLOCK_DIMS = {
    DEFAULT_CAMPAIGN: BLOCK_DIMS,
    NORM_STEADY_CALIBRATION_CAMPAIGN: BLOCK_DIMS,
    NORM_STEADY_VALIDATION_CAMPAIGN: BLOCK_DIMS,
    NORM_FLOOR_CALIBRATION_CAMPAIGN: BLOCK_DIMS,
    NORM_FLOOR_VALIDATION_CAMPAIGN: BLOCK_DIMS,
    SWIGLU_SERIALIZED_CAMPAIGN: BLOCK_DIMS,
    SERIALIZED_VECTOR_H8192_CAMPAIGN: BLOCK_DIMS,
    AIV_WORKER_PROLOGUE_CAMPAIGN: BLOCK_DIMS,
    SINGLE_TBUF_INIT_CAMPAIGN: BLOCK_DIMS,
    SCALAR_INTEGER_DEPENDENCY_CAMPAIGN: BLOCK_DIMS,
    SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN: BLOCK_DIMS,
    SWIGLU_BUFFER_INIT_CAMPAIGN: BLOCK_DIMS,
    SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN: BLOCK_DIMS,
    SWIGLU_QUEUE_CONTROL_CAMPAIGN: BLOCK_DIMS,
    RMSNORM_QUEUE_CONTROL_CAMPAIGN: BLOCK_DIMS,
    RMSNORM_BUFFER_READY_CAMPAIGN: BLOCK_DIMS,
    RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN: BLOCK_DIMS,
    NORM_REQUEST_SERVICE_CAMPAIGN: BLOCK_DIMS,
    RMSNORM_LOCAL_SCALAR_CAMPAIGN: BLOCK_DIMS,
    ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN: BLOCK_DIMS,
    ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN: (
        ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_BLOCK_DIMS),
    ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN: BLOCK_DIMS,
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN: (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_BLOCK_DIMS),
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN: (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_BLOCK_DIMS),
    VECTOR_TO_MTE2_FENCE_CAMPAIGN: BLOCK_DIMS,
    LOCAL_SCALAR_FILL_SV_CAMPAIGN: BLOCK_DIMS,
    LOCAL_UB_COPY_CAMPAIGN: BLOCK_DIMS,
    MERGE_N_ROW_GROUP_CAMPAIGN: MERGE_N_ROW_GROUP_BLOCK_DIMS,
    NORM_REQUEST_MERGE_N_CAMPAIGN: BLOCK_DIMS,
    FP32_PREDICATE_SELECT_CAMPAIGN: (1, 48),
    INT32_INT64_CAST_CAMPAIGN: (1, 48),
}
CSV_FIELDS = 12
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
_ACTIVE_PROCESS = None


def _iterations(elements):
    return _iterations_for_campaign(DEFAULT_CAMPAIGN, elements)


def _iterations_for_campaign(campaign, elements):
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unsupported campaign: {campaign}")
    _opcodes, element_counts, target_elements = CAMPAIGNS[campaign]
    if elements not in element_counts:
        raise ValueError(f"unsupported element count: {elements}")
    if campaign in (
            NORM_REQUEST_SERVICE_CAMPAIGN,
            RMSNORM_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN,
            VECTOR_TO_MTE2_FENCE_CAMPAIGN,
            SCALAR_INTEGER_DEPENDENCY_CAMPAIGN,
            SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN,
            LOCAL_SCALAR_FILL_SV_CAMPAIGN,
            LOCAL_UB_COPY_CAMPAIGN,
            NORM_REQUEST_MERGE_N_CAMPAIGN):
        return NORM_REQUEST_ITERATIONS
    if campaign == MERGE_N_ROW_GROUP_CAMPAIGN:
        return MERGE_N_ROW_GROUP_ITERATIONS
    if campaign in (
            AIV_WORKER_PROLOGUE_CAMPAIGN, SINGLE_TBUF_INIT_CAMPAIGN,
            SWIGLU_BUFFER_INIT_CAMPAIGN,
            SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN,
            SWIGLU_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_BUFFER_READY_CAMPAIGN):
        return 1
    if target_elements is None:
        raise AssertionError(f"campaign {campaign} has no iteration policy")
    # Round the target work up to an even intrinsic count because the kernel
    # alternates two destinations per loop.  This is a workload-independent
    # measurement-duration choice, not a throughput-derived adjustment.
    iterations = 2 * (
        (target_elements + 2 * elements - 1)
        // (2 * elements))
    if iterations <= 0 or iterations % 2 != 0:
        raise AssertionError("opcode iterations must be positive and even")
    return iterations


def _cell_index(trial_index, order_position):
    return _cell_index_for_count(trial_index, order_position, len(CELLS))


def _cell_index_for_count(trial_index, order_position, cell_count):
    if cell_count <= 0:
        raise ValueError("cell_count must be positive")
    rotation = trial_index % cell_count
    phase = trial_index // cell_count
    if phase % 2 == 0:
        return (rotation + order_position) % cell_count
    return (rotation - order_position) % cell_count


def _norm_request_cell_index(trial_index, order_position, cell_count):
    if cell_count <= 0 or cell_count % 3 != 0:
        raise ValueError(
            "NORM request campaign requires a positive cell count "
            "divisible by three")
    if trial_index < 0 or trial_index >= NORM_REQUEST_TRIALS_PER_RUN:
        raise ValueError("NORM request trial index is out of range")
    if order_position < 0 or order_position >= cell_count:
        raise ValueError("NORM request order position is out of range")
    rotation = (trial_index // 2) * (cell_count // 3)
    if trial_index % 2 == 0:
        return (rotation + order_position) % cell_count
    return (rotation - order_position) % cell_count


def _ordered_campaign_cells(campaign, trial_index, cells):
    if campaign in (
            NORM_REQUEST_SERVICE_CAMPAIGN, NORM_REQUEST_MERGE_N_CAMPAIGN):
        index_for_position = _norm_request_cell_index
    elif campaign in CAMPAIGNS:
        index_for_position = _cell_index_for_count
    else:
        raise ValueError(f"unsupported campaign: {campaign}")
    return tuple(
        cells[index_for_position(trial_index, order_position, len(cells))]
        for order_position in range(len(cells))
    )


def _campaign_cells(campaign):
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unsupported campaign: {campaign}")
    opcodes, element_counts, _target_elements = CAMPAIGNS[campaign]
    block_dims = CAMPAIGN_BLOCK_DIMS[campaign]
    if campaign == LOCAL_UB_COPY_CAMPAIGN:
        return tuple(
            (opcode, opcode_name, elements, block_dim)
            for opcode, opcode_name in opcodes
            for elements in LOCAL_UB_COPY_ELEMENTS[opcode_name]
            for block_dim in block_dims
        )
    if campaign in (
            SWIGLU_BUFFER_INIT_CAMPAIGN,
            SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN,
            SWIGLU_QUEUE_CONTROL_CAMPAIGN):
        selected_opcodes = (
            SWIGLU_BUFFER_INIT_OPCODES
            if campaign == SWIGLU_BUFFER_INIT_CAMPAIGN
            else (
                SWIGLU_BUFFER_LIFECYCLE_OPCODES
                if campaign == SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN
                else SWIGLU_QUEUE_CONTROL_OPCODES))
        return tuple(
            (opcode, opcode_name, elements, block_dim)
            for opcode, opcode_name in selected_opcodes
            for elements in _swiglu_buffer_init_elements(opcode_name)
            for block_dim in CAMPAIGN_BLOCK_DIMS[campaign]
        )
    return tuple(
        (opcode, opcode_name, elements, block_dim)
        for opcode, opcode_name in opcodes
        for elements in element_counts
        for block_dim in block_dims
    )


def _swiglu_buffer_init_elements(opcode_name):
    if opcode_name in (
            "swiglu_double_buffer_init",
            "swiglu_double_buffer_lifecycle",
            "swiglu_double_buffer_queue_control"):
        return 6144, 9728
    elif opcode_name in (
            "swiglu_single_buffer_init",
            "swiglu_single_buffer_lifecycle",
            "swiglu_single_buffer_queue_control"):
        return (13312,)
    else:
        raise ValueError(f"unsupported SwiGLU buffer-init opcode: {opcode_name}")


def _validate_schedule(trials, cells=CELLS):
    required_multiple = 2 * len(cells)
    if trials <= 0 or trials % required_multiple != 0:
        raise ValueError(
            f"trials must be a positive multiple of {required_multiple}")
    position_counts = [[0 for _ in cells] for _ in cells]
    for trial_index in range(trials):
        seen = set()
        for order_position in range(len(cells)):
            cell_index = _cell_index_for_count(
                trial_index, order_position, len(cells))
            if cell_index in seen:
                raise AssertionError("one trial contains a duplicate cell")
            seen.add(cell_index)
            position_counts[order_position][cell_index] += 1
    expected = trials // len(cells)
    if any(
        count != expected
        for position in position_counts
        for count in position
    ):
        raise AssertionError("cell/order schedule is not balanced")


def _validate_norm_request_schedule(trials, cells):
    if trials != NORM_REQUEST_TRIALS_PER_RUN:
        raise ValueError(
            "NORM request campaign requires exactly "
            f"{NORM_REQUEST_TRIALS_PER_RUN} trials per independent run")
    for trial_index in range(trials):
        indices = {
            _norm_request_cell_index(
                trial_index, order_position, len(cells))
            for order_position in range(len(cells))
        }
        if indices != set(range(len(cells))):
            raise AssertionError(
                "one NORM request trial is not a complete permutation")


def _validate_campaign_schedule(campaign, trials, cells):
    if campaign in (
            NORM_REQUEST_SERVICE_CAMPAIGN, NORM_REQUEST_MERGE_N_CAMPAIGN):
        _validate_norm_request_schedule(trials, cells)
    elif campaign in CAMPAIGNS:
        _validate_schedule(trials, cells)
    else:
        raise ValueError(f"unsupported campaign: {campaign}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _terminate_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _handle_signal(signum, _frame):
    if _ACTIVE_PROCESS is not None:
        _terminate_process_group(_ACTIVE_PROCESS)
    raise SystemExit(128 + signum)


def main():
    global _ACTIVE_PROCESS

    if sys.argv[1:] == ["--integer-reference-only"]:
        pattern = (
            -(2 ** 31), -(2 ** 31) + 1, -65537, -65536, -1, 0, 1,
            65535, 65536, 65537, 2 ** 31 - 2, 2 ** 31 - 1, -2, 2,
            -(2 ** 30), 2 ** 30)
        checks = 0
        for elements in CAMPAIGNS[INT32_INT64_CAST_CAMPAIGN][1]:
            for element in range(elements):
                value = pattern[element % 16]
                packed_input = struct.pack("<i", value)
                widened = struct.pack("<q", struct.unpack("<i", packed_input)[0])
                if struct.unpack("<q", widened)[0] != value:
                    raise AssertionError("signed INT32->INT64 reference mismatch")
                if widened[4:] != (b"\xff" * 4 if value < 0 else b"\0" * 4):
                    raise AssertionError("signed extension high word mismatch")
                checks += 1
        print(json.dumps({
            "schema_version": 1,
            "scope": "cpu_synthetic_reference_only",
            "checks_passed": checks,
            "input_element_bytes": 4,
            "output_element_bytes": 8,
            "source_api_count_mode": "mask_count",
            "elements_per_api_invocation": list(
                CAMPAIGNS[INT32_INT64_CAST_CAMPAIGN][1]),
            "fixed_isa_repeat_count_claimed": False,
            "device_correctness_verified": False,
            "production_integer_cast_bound": False,
        }, sort_keys=True))
        return

    if sys.argv[1:] == ["--typed-reference-only"]:
        # CPU-only validation of the precise synthetic FP32 patterns and
        # masks hard-coded in the device validator.  Does not import Torch,
        # ACL or initialize/query any accelerator.
        lhs_bits = (
            0x3f800000, 0xbf800000, 0x00000000, 0x80000000,
            0x7f800000, 0xff800000, 0x7fc00001, 0x3f800000,
            0x7fc00001, 0x40000000, 0xc0000000, 0x7f800000,
            0xff800000, 0x3f800000, 0xbf800000, 0x00000000)
        rhs_bits = (
            0xbf800000, 0x3f800000, 0x80000000, 0x00000000,
            0x7f800000, 0xff800000, 0x3f800000, 0x7fc00001,
            0x7fc00001, 0x3f800000, 0xbf800000, 0x3f800000,
            0x3f800000, 0x3f800000, 0xbf800000, 0x3f800000)
        gt = (True, False, False, False, False, False, False, False,
              False, True, False, True, False, False, False, False)
        eq = (False, False, True, True, True, True, False, False,
              False, False, False, False, False, True, True, False)
        checks = 0
        for elements in CAMPAIGNS[FP32_PREDICATE_SELECT_CAMPAIGN][1]:
            for element in range(elements):
                lane = element % 16
                lhs = struct.unpack("<f", struct.pack("<I", lhs_bits[lane]))[0]
                rhs = struct.unpack("<f", struct.pack("<I", rhs_bits[lane]))[0]
                ordered_ne = (not math.isnan(lhs) and not math.isnan(rhs)
                              and lhs != rhs)
                comparisons = (lhs > rhs, lhs == rhs, ordered_ne, lhs > rhs)
                expected = (gt[lane], eq[lane],
                            lane not in (6, 7, 8) and not eq[lane], gt[lane])
                if comparisons != expected:
                    raise AssertionError("synthetic FP32 predicate mismatch")
                checks += len(comparisons)
        print(json.dumps({
            "schema_version": 1,
            "scope": "cpu_synthetic_reference_only",
            "checks_passed": checks,
            "includes": ["numeric_order", "ties", "signed_zero",
                         "infinities", "quiet_nan_lhs_rhs_both"],
            "device_correctness_verified": False,
            "production_argmax_bound": False,
            "opcode_50_semantics": "ordered_inequality",
            "quiet_nan_payload_tested": "0x7fc00001",
            "nan_detection_service_verified": False,
            "signaling_nan_semantics_verified": False,
        }, sort_keys=True))
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--device-id", required=True, type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--campaign", choices=tuple(CAMPAIGNS), default=DEFAULT_CAMPAIGN)
    parser.add_argument(
        "--trials", type=_positive_integer)
    parser.add_argument(
        "--child-timeout-seconds", type=_positive_integer, default=900)
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        parser.error("--run-id must match [A-Za-z0-9._-]{1,96}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        parser.error("--binary must name an executable file")
    cells = _campaign_cells(args.campaign)
    default_trials = (
        NORM_REQUEST_TRIALS_PER_RUN
        if args.campaign in (
            NORM_REQUEST_SERVICE_CAMPAIGN, NORM_REQUEST_MERGE_N_CAMPAIGN)
        else 2 * len(cells))
    trials = args.trials if args.trials is not None else default_trials
    try:
        _validate_campaign_schedule(args.campaign, trials, cells)
    except ValueError as error:
        parser.error(str(error))

    print(
        f"runner_pid={os.getpid()} device_id={args.device_id} "
        f"run_id={args.run_id} campaign={args.campaign} trials={trials} "
        f"binary={args.binary.resolve()} "
        f"binary_sha256={_sha256(args.binary)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "run_id,trial,order_position,process_pid,opcode,elements,block_dim,"
        "n_iters,short_block_cycles,short_wall_ns,long_block_cycles,"
        "long_wall_ns,tick_ns,per_iteration_ns,elements_per_ns,validation",
        flush=True,
    )
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    for trial_index in range(trials):
        ordered_cells = _ordered_campaign_cells(
            args.campaign, trial_index, cells)
        for batch_start in range(0, len(ordered_cells), MAX_CELLS_PER_BATCH):
            batch = ordered_cells[
                batch_start:batch_start + MAX_CELLS_PER_BATCH]
            command = [str(args.binary), "--batch"]
            for opcode, _opcode_name, elements, block_dim in batch:
                command.extend(
                    (str(opcode), str(elements), str(block_dim),
                     str(_iterations_for_campaign(args.campaign, elements)))
                )
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                start_new_session=True,
            )
            _ACTIVE_PROCESS = process
            try:
                stdout, stderr = process.communicate(
                    timeout=args.child_timeout_seconds)
            except subprocess.TimeoutExpired as error:
                _terminate_process_group(process)
                raise RuntimeError(
                    f"child PID {process.pid} timed out after "
                    f"{args.child_timeout_seconds} seconds"
                ) from error
            finally:
                if process.poll() is None:
                    _terminate_process_group(process)
                _ACTIVE_PROCESS = None
            if stderr:
                print(
                    f"child_pid={process.pid} trial={trial_index + 1} "
                    f"batch_start={batch_start} stderr={stderr.rstrip()}",
                    file=sys.stderr,
                    flush=True,
                )
            if process.returncode != 0:
                raise RuntimeError(
                    f"child PID {process.pid} failed with exit "
                    f"{process.returncode}: {' '.join(command)}"
                )
            output_lines = stdout.strip().splitlines()
            if len(output_lines) != len(batch):
                raise RuntimeError(
                    f"child PID {process.pid} emitted "
                    f"{len(output_lines)} data lines"
                )
            for batch_position, output_line in enumerate(output_lines):
                order_position = batch_start + batch_position
                fields = output_line.split(",")
                if len(fields) != CSV_FIELDS:
                    raise RuntimeError(
                        f"child PID {process.pid} emitted an invalid raw "
                        "schema")
                _opcode, opcode_name, elements, block_dim = batch[
                    batch_position]
                if (
                    fields[0] != opcode_name
                    or int(fields[1]) != elements
                    or int(fields[2]) != block_dim
                    or int(fields[3]) != _iterations_for_campaign(
                        args.campaign, elements)
                    or fields[-1] != "pass"
                ):
                    raise RuntimeError(
                        f"child PID {process.pid} changed the requested "
                        "order")
                print(
                    f"{args.run_id},{trial_index + 1},{order_position},"
                    f"{process.pid},{output_line}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
