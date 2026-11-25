#!/usr/bin/env python3
"""Audit raw AIV opcode trials with frozen, evaluation-independent gates."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from vector_opcode_runner import (
    ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN,
    ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_BLOCK_DIMS,
    ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN,
    ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN,
    ADD_RMSNORM_REDUCE_SUM_FP32_ELEMENTS,
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_BLOCK_DIMS,
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN,
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_ELEMENTS,
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN,
    ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_ELEMENTS,
    AIV_WORKER_PROLOGUE_CAMPAIGN,
    BLOCK_DIMS,
    CELLS,
    DEFAULT_CAMPAIGN,
    ELEMENT_COUNTS,
    FP32_PREDICATE_SELECT_CAMPAIGN,
    INT32_INT64_CAST_CAMPAIGN,
    LOCAL_SCALAR_FILL_SV_CAMPAIGN,
    LOCAL_UB_COPY_CAMPAIGN,
    LOCAL_UB_COPY_ELEMENTS,
    LOCAL_UB_COPY_OPCODES,
    NORM_FLOOR_CALIBRATION_CAMPAIGN,
    NORM_FLOOR_VALIDATION_CAMPAIGN,
    NORM_REQUEST_ELEMENTS,
    NORM_REQUEST_OPCODES,
    NORM_REQUEST_SERVICE_CAMPAIGN,
    NORM_REQUEST_TRIALS_PER_RUN,
    RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN,
    RMSNORM_HALF_INTERVAL_REDUCE_ELEMENTS,
    RMSNORM_BUFFER_READY_CAMPAIGN,
    RMSNORM_BUFFER_READY_OPCODES,
    RMSNORM_LOCAL_SCALAR_CAMPAIGN,
    RMSNORM_QUEUE_CONTROL_CAMPAIGN,
    RMSNORM_QUEUE_CONTROL_OPCODES,
    SCALAR_INTEGER_DEPENDENCY_CAMPAIGN,
    SCALAR_INTEGER_DEPENDENCY_OPCODES,
    SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN,
    SCALAR_U64_DIVREM_DEPENDENCY_OPCODES,
    SINGLE_TBUF_INIT_CAMPAIGN,
    NORM_STEADY_CALIBRATION_CAMPAIGN,
    NORM_STEADY_VALIDATION_CAMPAIGN,
    SYSTEM_COUNTER_TICK_NS,
    SERIALIZED_VECTOR_H8192_CAMPAIGN,
    SWIGLU_SERIALIZED_CAMPAIGN,
    SWIGLU_SERIALIZED_OPCODES,
    SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN,
    SWIGLU_BUFFER_LIFECYCLE_OPCODES,
    SWIGLU_QUEUE_CONTROL_CAMPAIGN,
    SWIGLU_QUEUE_CONTROL_OPCODES,
    TARGET_ELEMENTS_PER_INTERVAL,
    VECTOR_TO_MTE2_FENCE_CAMPAIGN,
    _campaign_cells,
    _cell_index,
    _cell_index_for_count,
    _iterations,
    _iterations_for_campaign,
    _norm_request_cell_index,
)


REQUIRED_FIELDS = (
    "run_id",
    "trial",
    "order_position",
    "process_pid",
    "opcode",
    "elements",
    "block_dim",
    "n_iters",
    "short_block_cycles",
    "short_wall_ns",
    "long_block_cycles",
    "long_wall_ns",
    "tick_ns",
    "per_iteration_ns",
    "elements_per_ns",
    "validation",
)
PLATEAU_ELEMENTS = (4096, 6144)
VALIDATION_ELEMENT = 2048
VALIDATION_CAMPAIGN_ELEMENTS = (512, 1024, 2048, 4096)
MINIMUM_INDEPENDENT_RUNS = 3
EXPECTED_TRIALS_PER_RUN = 2 * len(CELLS)
EXPECTED_VALIDATION_TRIALS_PER_RUN = 72
NORM_REPORT_CAMPAIGN = "sqrt_div_norm"
NORM_STEADY_CALIBRATION_ELEMENTS = (4096, 6144)
NORM_STEADY_VALIDATION_ELEMENT = 2048
NORM_FLOOR_CALIBRATION_ELEMENT = 1
NORM_FLOOR_VALIDATION_ELEMENTS = (32, 64)
NORM_OPCODE_NAMES = ("sqrt", "div", "sqrt_div")
FROZEN_DIV_SETUP_TIME_NS = 17.1644249580
FROZEN_DIV_ELEMENT_INTERVAL_NS = 1e9 / 28749679296.599228
SWIGLU_SERIALIZED_ELEMENTS = (32, 2048, 6144)
SWIGLU_SERIALIZED_CALIBRATION_ELEMENTS = (32, 6144)
SWIGLU_SERIALIZED_VALIDATION_ELEMENT = 2048
SWIGLU_SERIALIZED_OPCODE_NAMES = tuple(
    name for _opcode, name in SWIGLU_SERIALIZED_OPCODES)
SWIGLU_SERIALIZED_CALIBRATION_RUNS = 2
SWIGLU_SERIALIZED_MAX_VALIDATION_APE_PERCENT = 25.0
SERIALIZED_VECTOR_EXTENDED_ELEMENTS = (*SWIGLU_SERIALIZED_ELEMENTS, 8192)
SERIALIZED_VECTOR_EXTENDED_OPCODE_NAMES = tuple(
    name for name in SWIGLU_SERIALIZED_OPCODE_NAMES
    if name not in ("exp_serialized", "div_serialized"))
NORM_REQUEST_OPCODE_NAMES = tuple(
    name for _opcode, name in NORM_REQUEST_OPCODES)
NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD = 0.01
NORM_REQUEST_PROVENANCE_SCHEMA_VERSION = 2
RMSNORM_HALF_INTERVAL_CALIBRATION_ELEMENTS = (2048, 4096, 8192)
RMSNORM_HALF_INTERVAL_VALIDATION_ELEMENTS = (2560, 5120, 12288)
RMSNORM_HALF_INTERVAL_MAX_VALIDATION_APE_PERCENT = 10.0
ADD_RMSNORM_REDUCE_SUM_FP32_CALIBRATION_ELEMENTS = (2048, 4096, 8192)
ADD_RMSNORM_REDUCE_SUM_FP32_VALIDATION_ELEMENTS = (2560, 5120, 12288)
ADD_RMSNORM_REDUCE_SUM_FP32_MAX_VALIDATION_APE_PERCENT = 10.0
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CALIBRATION_ELEMENTS = (
    64, 128, 256)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_VALIDATION_ELEMENTS = (
    8, 16, 32, 512)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_MAX_VALIDATION_APE_PERCENT = 10.0
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_BODY_CALIBRATION_ELEMENTS = (
    64, 128, 256)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_BODY_VALIDATION_ELEMENTS = (
    192, 320, 448, 512)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_TAIL_CALIBRATION_ELEMENTS = (
    8, 16)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_TAIL_VALIDATION_ELEMENTS = (
    32, 56)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_COMPOSITE_VALIDATION_ELEMENTS = (
    72, 120, 504)
ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_MAX_VALIDATION_APE_PERCENT = 10.0
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
NORM_REQUEST_SOURCE_ARTIFACTS = (
    "source/CMakeLists.txt",
    "source/vector_opcode_kernel.cpp",
    "source/vector_opcode_main.cpp",
    "source/vector_opcode_report.py",
    "source/vector_opcode_runner.py",
)


def _quartiles(values):
    if len(values) < 4:
        raise ValueError("each opcode cell requires at least four samples")
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, statistics.median(values), q3


def _parse_cycles(text, block_dim, context):
    fields = text.split("|")
    if len(fields) != block_dim:
        raise ValueError(f"{context} does not match block_dim")
    cycles = tuple(int(field) for field in fields)
    if any(value <= 0 for value in cycles):
        raise ValueError(f"{context} contains non-positive cycles")
    return cycles


def _decode_row(raw, path, identities, expected_cells, expected_iterations):
    if raw["validation"] != "pass":
        raise ValueError(f"{path} contains failed validation")
    trial = int(raw["trial"])
    order_position = int(raw["order_position"])
    process_pid = int(raw["process_pid"])
    elements = int(raw["elements"])
    block_dim = int(raw["block_dim"])
    iterations = int(raw["n_iters"])
    short_wall_ns = int(raw["short_wall_ns"])
    long_wall_ns = int(raw["long_wall_ns"])
    if trial <= 0 or process_pid <= 0:
        raise ValueError(f"{path} contains invalid process identity")
    if order_position < 0 or order_position >= len(expected_cells):
        raise ValueError(f"{path} contains invalid order position")
    cell = (raw["opcode"], elements, block_dim)
    if cell not in expected_cells:
        raise ValueError(f"{path} contains unexpected cell {cell}")
    if iterations != expected_iterations(elements):
        raise ValueError(f"{path} changed the frozen iteration count")
    short_cycles = _parse_cycles(
        raw["short_block_cycles"], block_dim, "short cycles")
    long_cycles = _parse_cycles(
        raw["long_block_cycles"], block_dim, "long cycles")
    direct_task_sample = raw["opcode"] in (
        "two_tbuf_mask_prologue", "single_tbuf_init",
        "swiglu_double_buffer_lifecycle",
        "swiglu_single_buffer_lifecycle",
        "swiglu_double_buffer_queue_control",
        "swiglu_single_buffer_queue_control",
        "rmsnorm_four_queue_three_tbuf_lifecycle",
        "rmsnorm_four_queue_control_chain",
        "rmsnorm_four_queue_three_tbuf_ready",
    )
    if (not direct_task_sample and any(
        long <= short
        for short, long in zip(short_cycles, long_cycles)
    )):
        raise ValueError(f"{path} has non-increasing block cycles")
    if short_wall_ns <= 0 or long_wall_ns <= 0:
        raise ValueError(f"{path} has invalid wall intervals")
    identity = (raw["run_id"], trial, order_position)
    if identity in identities:
        raise ValueError(f"duplicate row identity {identity}")
    identities.add(identity)

    tick_ns = float(raw["tick_ns"])
    per_iteration_ns = float(raw["per_iteration_ns"])
    elements_per_ns = float(raw["elements_per_ns"])
    delta_cycles = max(long_cycles) - max(short_cycles)
    expected_tick_ns = SYSTEM_COUNTER_TICK_NS
    expected_per_iteration_ns = (
        0.5 * (max(short_cycles) + max(long_cycles))
        * SYSTEM_COUNTER_TICK_NS
        if direct_task_sample
        else delta_cycles * SYSTEM_COUNTER_TICK_NS / (3 * iterations))
    expected_elements_per_ns = elements / expected_per_iteration_ns
    checks = (
        (tick_ns, expected_tick_ns),
        (per_iteration_ns, expected_per_iteration_ns),
        (elements_per_ns, expected_elements_per_ns),
    )
    if any(
        not math.isfinite(observed) or observed <= 0
        or not math.isclose(
            observed, expected, rel_tol=1e-8, abs_tol=1e-8)
        for observed, expected in checks
    ):
        raise ValueError(f"{path} contains inconsistent derived values")
    return {
        "run_id": raw["run_id"],
        "trial": trial,
        "order_position": order_position,
        "cell": cell,
        "opcode": raw["opcode"],
        "elements": elements,
        "block_dim": block_dim,
        "tick_ns": tick_ns,
        "per_iteration_ns": per_iteration_ns,
        "elements_per_ns": elements_per_ns,
        "per_worker_sample_ns": tuple(
            0.5 * (short + long) * tick_ns
            for short, long in zip(short_cycles, long_cycles)),
    }


def _read_campaign(paths, expected_cells, expected_iterations):
    rows = []
    identities = set()
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
                raise ValueError(f"{path} has an invalid raw schema")
            for raw in reader:
                rows.append(_decode_row(
                    raw, path, identities, expected_cells,
                    expected_iterations))
    if not rows:
        raise ValueError("no opcode rows were provided")
    return rows


def _load_rows(paths):
    expected_cells = {
        (opcode_name, elements, block_dim)
        for _opcode, opcode_name, elements, block_dim in CELLS
    }
    return _read_campaign(paths, expected_cells, _iterations)


def _validation_iterations(elements):
    if elements not in VALIDATION_CAMPAIGN_ELEMENTS:
        raise ValueError(
            f"unsupported validation element count: {elements}")
    if TARGET_ELEMENTS_PER_INTERVAL % elements != 0:
        raise AssertionError(
            "validation campaign requires exact target work division")
    return TARGET_ELEMENTS_PER_INTERVAL // elements


def _load_validation_rows(paths):
    expected_cells = {
        (opcode, elements, block_dim)
        for opcode in ("exp", "div", "exp_div")
        for elements in VALIDATION_CAMPAIGN_ELEMENTS
        for block_dim in BLOCK_DIMS
    }
    rows = _read_campaign(paths, expected_cells, _validation_iterations)
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) < MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "validation opcode audit requires at least "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    by_trial = {}
    for row in rows:
        trial_key = (row["run_id"], row["trial"])
        if trial_key not in by_trial:
            by_trial[trial_key] = {}
        if row["cell"] in by_trial[trial_key]:
            raise ValueError(
                f"validation trial {trial_key} duplicates {row['cell']}")
        by_trial[trial_key][row["cell"]] = row["order_position"]
    for trial_key, trial_cells in by_trial.items():
        if set(trial_cells) != expected_cells:
            raise ValueError(f"validation trial {trial_key} is incomplete")
        if set(trial_cells.values()) != set(range(len(expected_cells))):
            raise ValueError(
                f"validation trial {trial_key} has incomplete positions")
    expected_trials = set(
        range(1, EXPECTED_VALIDATION_TRIALS_PER_RUN + 1))
    for run_id in run_ids:
        observed_trials = {
            row["trial"] for row in rows if row["run_id"] == run_id}
        if observed_trials != expected_trials:
            raise ValueError(
                f"validation run {run_id!r} must contain exactly "
                f"{EXPECTED_VALIDATION_TRIALS_PER_RUN} complete trials")
    return tuple(
        row for row in rows if row["elements"] == VALIDATION_ELEMENT)


def _load_balanced_campaign(paths, campaign):
    cells = _campaign_cells(campaign)
    expected_cells = {
        (opcode_name, elements, block_dim)
        for _opcode, opcode_name, elements, block_dim in cells
    }
    rows = _read_campaign(
        paths,
        expected_cells,
        lambda elements: _iterations_for_campaign(campaign, elements),
    )
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) < MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            f"{campaign} requires at least {MINIMUM_INDEPENDENT_RUNS} "
            "independent run IDs")
    by_trial = {}
    for row in rows:
        trial_key = (row["run_id"], row["trial"])
        if trial_key not in by_trial:
            by_trial[trial_key] = {}
        if row["cell"] in by_trial[trial_key]:
            raise ValueError(
                f"{campaign} trial {trial_key} duplicates {row['cell']}")
        by_trial[trial_key][row["cell"]] = row["order_position"]
    for trial_key, trial_cells in by_trial.items():
        if set(trial_cells) != expected_cells:
            raise ValueError(f"{campaign} trial {trial_key} is incomplete")
        if set(trial_cells.values()) != set(range(len(cells))):
            raise ValueError(
                f"{campaign} trial {trial_key} has incomplete positions")
        trial_index = trial_key[1] - 1
        for cell, order_position in trial_cells.items():
            expected_cell = cells[_cell_index_for_count(
                trial_index, order_position, len(cells))]
            expected_identity = (
                expected_cell[1], expected_cell[2], expected_cell[3])
            if cell != expected_identity:
                raise ValueError(
                    f"{campaign} trial {trial_key} position "
                    f"{order_position} changed the frozen balanced schedule")
    expected_trial_count = 2 * len(cells)
    expected_trials = set(range(1, expected_trial_count + 1))
    for run_id in run_ids:
        observed_trials = {
            row["trial"] for row in rows if row["run_id"] == run_id}
        if observed_trials != expected_trials:
            raise ValueError(
                f"{campaign} run {run_id!r} must contain exactly "
                f"{expected_trial_count} complete trials")
    return tuple(rows)


def _intersection(summaries):
    lower = max(summary[0] for summary in summaries)
    upper = min(summary[2] for summary in summaries)
    return lower, upper, lower <= upper


def _derive_affine_intrinsic_parameters(rows, run_ids, opcode):
    lower_elements, upper_elements = PLATEAU_ELEMENTS
    if upper_elements <= lower_elements:
        raise AssertionError("affine opcode points must be ordered")
    strata = []
    for run_id in run_ids:
        for block_dim in BLOCK_DIMS:
            medians = {}
            for elements in PLATEAU_ELEMENTS:
                values = [
                    row["per_iteration_ns"]
                    for row in rows
                    if row["run_id"] == run_id
                    and row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ]
                if not values:
                    raise ValueError(
                        f"missing affine stratum for {opcode}, {run_id}, "
                        f"block_dim={block_dim}, elements={elements}")
                medians[elements] = statistics.median(values)
            element_interval_ns = (
                (medians[upper_elements] - medians[lower_elements])
                / (upper_elements - lower_elements))
            setup_time_ns = (
                medians[lower_elements]
                - lower_elements * element_interval_ns)
            strata.append({
                "run_id": run_id,
                "block_dim": block_dim,
                "setup_time_ns": setup_time_ns,
                "element_interval_ns": element_interval_ns,
            })
    setup_values = [item["setup_time_ns"] for item in strata]
    interval_values = [item["element_interval_ns"] for item in strata]
    setup_summary = _quartiles(setup_values)
    interval_summary = _quartiles(interval_values)
    positive = all(value > 0.0 for value in setup_values + interval_values)
    interval_median = interval_summary[1]
    return {
        "strata": strata,
        "setup_time_ns_iqr_median": setup_summary,
        "element_interval_ns_iqr_median": interval_summary,
        "all_strata_positive": positive,
        "candidate_setup_time_s_per_intrinsic": (
            setup_summary[1] * 1e-9 if positive else None),
        "candidate_steady_rate_elements_per_s_per_aiv": (
            1.0 / (interval_median * 1e-9)
            if positive else None),
    }


def _audit_affine_validation(rows, affine):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) < MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "affine validation requires at least "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    expected_cells = {
        (opcode, VALIDATION_ELEMENT, block_dim)
        for opcode in ("exp", "div", "exp_div")
        for block_dim in BLOCK_DIMS
    }
    by_cell = {cell: [] for cell in expected_cells}
    for row in rows:
        if row["cell"] not in by_cell:
            raise ValueError(
                f"affine validation contains unexpected cell {row['cell']}")
        by_cell[row["cell"]].append(row["per_iteration_ns"])
    counts = {len(values) for values in by_cell.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError(
            "affine validation cells do not have equal non-zero samples")

    observed_by_block = {
        (opcode, block_dim): _quartiles(
            by_cell[(opcode, VALIDATION_ELEMENT, block_dim)])
        for opcode in ("exp", "div", "exp_div")
        for block_dim in BLOCK_DIMS
    }
    concurrency_common = {
        opcode: _intersection(tuple(
            observed_by_block[(opcode, block_dim)]
            for block_dim in BLOCK_DIMS))[2]
        for opcode in ("exp", "div", "exp_div")
    }
    observed = {
        opcode: _quartiles([
            row["per_iteration_ns"]
            for row in rows if row["opcode"] == opcode
        ])
        for opcode in ("exp", "div", "exp_div")
    }

    predicted_values = {}
    for opcode in ("exp", "div", "exp_div"):
        predicted_values[opcode] = [
            item["setup_time_ns"]
            + VALIDATION_ELEMENT * item["element_interval_ns"]
            for item in affine[opcode]["strata"]
        ]
    predicted = {
        opcode: _quartiles(predicted_values[opcode])
        for opcode in ("exp", "div", "exp_div")
    }
    intersections = {}
    for opcode in ("exp", "div", "exp_div"):
        lower = max(predicted[opcode][0], observed[opcode][0])
        upper = min(predicted[opcode][2], observed[opcode][2])
        intersections[opcode] = {
            "predicted_iqr_median_ns": predicted[opcode],
            "observed_iqr_median_ns": observed[opcode],
            "lower_ns": lower,
            "upper_ns": upper,
            "common": lower <= upper,
        }
    publishable = (
        all(concurrency_common.values())
        and all(item["common"] for item in intersections.values())
    )
    return {
        "element": VALIDATION_ELEMENT,
        "independent_runs": len(run_ids),
        "samples_per_cell": next(iter(counts)),
        "concurrency_common": concurrency_common,
        "intersections": intersections,
        "publishable": publishable,
    }


def _audit(rows, validation_rows=None):
    expected_cells = {
        (opcode_name, elements, block_dim)
        for _opcode, opcode_name, elements, block_dim in CELLS
    }
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) < MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            f"opcode audit requires at least {MINIMUM_INDEPENDENT_RUNS} "
            "independent run IDs")
    by_cell = {cell: [] for cell in expected_cells}
    by_trial = {}
    for row in rows:
        by_cell[row["cell"]].append(row)
        trial_key = (row["run_id"], row["trial"])
        if trial_key not in by_trial:
            by_trial[trial_key] = {}
        if row["cell"] in by_trial[trial_key]:
            raise ValueError(f"trial {trial_key} duplicates {row['cell']}")
        by_trial[trial_key][row["cell"]] = row["order_position"]
    for trial_key, trial_cells in by_trial.items():
        if set(trial_cells) != expected_cells:
            raise ValueError(f"trial {trial_key} is incomplete")
        if set(trial_cells.values()) != set(range(len(CELLS))):
            raise ValueError(f"trial {trial_key} has incomplete positions")
        trial_index = trial_key[1] - 1
        for cell, order_position in trial_cells.items():
            expected_cell = CELLS[_cell_index(trial_index, order_position)]
            expected_identity = (
                expected_cell[1], expected_cell[2], expected_cell[3])
            if cell != expected_identity:
                raise ValueError(
                    f"trial {trial_key} position {order_position} changed "
                    "the frozen balanced schedule")
    expected_trials = set(range(1, EXPECTED_TRIALS_PER_RUN + 1))
    for run_id in run_ids:
        observed_trials = {
            row["trial"] for row in rows if row["run_id"] == run_id}
        if observed_trials != expected_trials:
            raise ValueError(
                f"run {run_id!r} must contain exactly "
                f"{EXPECTED_TRIALS_PER_RUN} complete trials")
    counts = {len(cell_rows) for cell_rows in by_cell.values()}
    if len(counts) != 1 or 0 in counts:
        raise ValueError("opcode cells do not have equal non-zero samples")

    throughput_summary = {
        cell: _quartiles(
            [row["elements_per_ns"] for row in cell_rows])
        for cell, cell_rows in by_cell.items()
    }
    tick_summary = {
        cell: _quartiles([row["tick_ns"] for row in cell_rows])
        for cell, cell_rows in by_cell.items()
    }
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summary.values()))

    concurrency_common = {}
    for opcode in ("exp", "div", "exp_div"):
        for elements in ELEMENT_COUNTS:
            _lower, _upper, common = _intersection(tuple(
                throughput_summary[(opcode, elements, block_dim)]
                for block_dim in BLOCK_DIMS
            ))
            concurrency_common[(opcode, elements)] = common

    plateau = {}
    candidates = {}
    for opcode in ("exp", "div"):
        summaries = tuple(
            throughput_summary[(opcode, elements, block_dim)]
            for elements in PLATEAU_ELEMENTS
            for block_dim in BLOCK_DIMS
        )
        lower, upper, common = _intersection(summaries)
        plateau[opcode] = {
            "lower": lower,
            "upper": upper,
            "common": common,
        }
        values = [
            row["elements_per_ns"]
            for row in rows
            if row["opcode"] == opcode
            and row["elements"] in PLATEAU_ELEMENTS
        ]
        candidates[opcode] = statistics.median(values) if common else None

    pair_service_common = {}
    for elements in ELEMENT_COUNTS:
        for block_dim in BLOCK_DIMS:
            exp_summary = throughput_summary[("exp", elements, block_dim)]
            div_summary = throughput_summary[("div", elements, block_dim)]
            pair_summary = throughput_summary[
                ("exp_div", elements, block_dim)]
            expected_service_lower = 1.0 / exp_summary[2] + 1.0 / div_summary[2]
            expected_service_upper = 1.0 / exp_summary[0] + 1.0 / div_summary[0]
            pair_service_lower = 1.0 / pair_summary[2]
            pair_service_upper = 1.0 / pair_summary[0]
            overlap_lower = max(expected_service_lower, pair_service_lower)
            overlap_upper = min(expected_service_upper, pair_service_upper)
            pair_service_common[(elements, block_dim)] = (
                overlap_lower <= overlap_upper
                or math.isclose(
                    overlap_lower, overlap_upper,
                    rel_tol=1e-6, abs_tol=1e-12)
            )

    affine = {
        opcode: _derive_affine_intrinsic_parameters(rows, run_ids, opcode)
        for opcode in ("exp", "div", "exp_div")
    }
    affine_additive_common = {}
    for field in ("setup_time_ns", "element_interval_ns"):
        expected_values = [
            exp[field] + div[field]
            for exp, div in zip(
                affine["exp"]["strata"], affine["div"]["strata"])
        ]
        observed_values = [
            item[field] for item in affine["exp_div"]["strata"]]
        expected_summary = _quartiles(expected_values)
        observed_summary = _quartiles(observed_values)
        lower = max(expected_summary[0], observed_summary[0])
        upper = min(expected_summary[2], observed_summary[2])
        affine_additive_common[field] = {
            "expected_iqr_median": expected_summary,
            "observed_iqr_median": observed_summary,
            "common": lower <= upper,
        }

    capacity_plateau_publishable = (
        tick_common
        and all(concurrency_common.values())
        and all(item["common"] for item in plateau.values())
        and all(pair_service_common.values())
    )
    affine_publishable = (
        tick_common
        and all(concurrency_common.values())
        and all(item["all_strata_positive"] for item in affine.values())
        and all(pair_service_common.values())
        and all(item["common"] for item in affine_additive_common.values())
    )
    affine_validation = (
        _audit_affine_validation(validation_rows, affine)
        if validation_rows is not None else None)
    mechanism_publishable = (
        affine_publishable
        and affine_validation is not None
        and affine_validation["publishable"]
    )
    return {
        "independent_runs": len(run_ids),
        "samples_per_cell": next(iter(counts)),
        "tick_common": tick_common,
        "tick_intersection": (tick_lower, tick_upper),
        "concurrency_common": concurrency_common,
        "plateau": plateau,
        "pair_service_common": pair_service_common,
        "capacity_candidates_elements_per_ns_per_aiv": candidates,
        "capacity_plateau_publishable": capacity_plateau_publishable,
        "affine_intrinsic_parameters": affine,
        "affine_additive_common": affine_additive_common,
        "affine_calibration_publishable": affine_publishable,
        "affine_validation": affine_validation,
        "mechanism_publishable": mechanism_publishable,
    }


def _derive_norm_affine(rows, opcode):
    lower_elements, upper_elements = NORM_STEADY_CALIBRATION_ELEMENTS
    run_ids = sorted({row["run_id"] for row in rows})
    strata = []
    for run_id in run_ids:
        for block_dim in BLOCK_DIMS:
            medians = {}
            for elements in NORM_STEADY_CALIBRATION_ELEMENTS:
                values = [
                    row["per_iteration_ns"]
                    for row in rows
                    if row["run_id"] == run_id
                    and row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ]
                if not values:
                    raise ValueError(
                        f"missing NORM affine stratum for {opcode}, "
                        f"{run_id}, block_dim={block_dim}, "
                        f"elements={elements}")
                medians[elements] = statistics.median(values)
            element_interval_ns = (
                (medians[upper_elements] - medians[lower_elements])
                / (upper_elements - lower_elements))
            setup_time_ns = (
                medians[lower_elements]
                - lower_elements * element_interval_ns)
            strata.append({
                "run_id": run_id,
                "block_dim": block_dim,
                "setup_time_ns": setup_time_ns,
                "element_interval_ns": element_interval_ns,
            })
    setup_values = [item["setup_time_ns"] for item in strata]
    interval_values = [item["element_interval_ns"] for item in strata]
    setup_summary = _quartiles(setup_values)
    interval_summary = _quartiles(interval_values)
    positive = all(value > 0.0 for value in setup_values + interval_values)
    return {
        "strata": strata,
        "setup_time_ns_iqr_median": setup_summary,
        "element_interval_ns_iqr_median": interval_summary,
        "all_strata_positive": positive,
        "candidate_setup_time_s_per_intrinsic": (
            setup_summary[1] * 1e-9 if positive else None),
        "candidate_steady_rate_elements_per_s_per_aiv": (
            1.0 / (interval_summary[1] * 1e-9)
            if positive else None),
    }


def _interval_overlap(first, second):
    lower = max(first[0], second[0])
    upper = min(first[2], second[2])
    return {
        "first_iqr_median": first,
        "second_iqr_median": second,
        "lower": lower,
        "upper": upper,
        "common": lower <= upper or math.isclose(
            lower, upper, rel_tol=1e-4, abs_tol=1e-9),
    }


def _audit_norm_typed_service(
        steady_calibration_rows, steady_validation_rows,
        floor_calibration_rows, floor_validation_rows):
    run_counts = {
        "steady_calibration": len({
            row["run_id"] for row in steady_calibration_rows}),
        "steady_validation": len({
            row["run_id"] for row in steady_validation_rows}),
        "floor_calibration": len({
            row["run_id"] for row in floor_calibration_rows}),
        "floor_validation": len({
            row["run_id"] for row in floor_validation_rows}),
    }
    all_rows = (
        tuple(steady_calibration_rows) + tuple(steady_validation_rows)
        + tuple(floor_calibration_rows) + tuple(floor_validation_rows))
    all_elements = (
        NORM_FLOOR_CALIBRATION_ELEMENT,
        *NORM_FLOOR_VALIDATION_ELEMENTS,
        NORM_STEADY_VALIDATION_ELEMENT,
        *NORM_STEADY_CALIBRATION_ELEMENTS,
    )
    tick_summaries = {}
    for opcode in NORM_OPCODE_NAMES:
        for elements in all_elements:
            for block_dim in BLOCK_DIMS:
                values = [
                    row["tick_ns"] for row in all_rows
                    if row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ]
                tick_summaries[(opcode, elements, block_dim)] = _quartiles(
                    values)
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))

    concurrency_common = {}
    for opcode in NORM_OPCODE_NAMES:
        for elements in all_elements:
            summaries = tuple(
                _quartiles([
                    row["per_iteration_ns"] for row in all_rows
                    if row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ])
                for block_dim in BLOCK_DIMS
            )
            concurrency_common[(opcode, elements)] = _intersection(
                summaries)[2]

    affine = {
        opcode: _derive_norm_affine(steady_calibration_rows, opcode)
        for opcode in NORM_OPCODE_NAMES
    }
    pair_affine_common = {}
    for field in ("setup_time_ns", "element_interval_ns"):
        if field == "setup_time_ns":
            frozen_div_value = FROZEN_DIV_SETUP_TIME_NS
        elif field == "element_interval_ns":
            frozen_div_value = FROZEN_DIV_ELEMENT_INTERVAL_NS
        else:
            raise AssertionError(f"unhandled affine field: {field}")
        expected_values = [
            item[field] + frozen_div_value
            for item in affine["sqrt"]["strata"]
        ]
        observed_values = [
            item[field] for item in affine["sqrt_div"]["strata"]]
        pair_affine_common[field] = _interval_overlap(
            _quartiles(expected_values), _quartiles(observed_values))

    frozen_div_steady_consistency = {}
    for elements in (
            NORM_STEADY_VALIDATION_ELEMENT,
            *NORM_STEADY_CALIBRATION_ELEMENTS):
        observed = _quartiles([
            row["per_iteration_ns"] for row in all_rows
            if row["opcode"] == "div" and row["elements"] == elements
        ])
        predicted = (
            FROZEN_DIV_SETUP_TIME_NS
            + elements * FROZEN_DIV_ELEMENT_INTERVAL_NS)
        frozen_div_steady_consistency[elements] = {
            "predicted_ns": predicted,
            "observed_iqr_median_ns": observed,
            "common": observed[0] <= predicted <= observed[2],
        }

    steady_validation_predictions = {
        "sqrt": [
            item["setup_time_ns"]
            + NORM_STEADY_VALIDATION_ELEMENT
            * item["element_interval_ns"]
            for item in affine["sqrt"]["strata"]
        ],
        "div": [
            FROZEN_DIV_SETUP_TIME_NS
            + NORM_STEADY_VALIDATION_ELEMENT
            * FROZEN_DIV_ELEMENT_INTERVAL_NS
            for _item in affine["sqrt"]["strata"]
        ],
        "sqrt_div": [
            item["setup_time_ns"] + FROZEN_DIV_SETUP_TIME_NS
            + NORM_STEADY_VALIDATION_ELEMENT * (
                item["element_interval_ns"]
                + FROZEN_DIV_ELEMENT_INTERVAL_NS)
            for item in affine["sqrt"]["strata"]
        ],
    }
    steady_validation_intersections = {}
    for opcode in NORM_OPCODE_NAMES:
        predicted = _quartiles(steady_validation_predictions[opcode])
        observed = _quartiles([
            row["per_iteration_ns"] for row in steady_validation_rows
            if row["opcode"] == opcode
        ])
        steady_validation_intersections[opcode] = _interval_overlap(
            predicted, observed)

    floor_summaries = {
        opcode: _quartiles([
            row["per_iteration_ns"] for row in floor_calibration_rows
            if row["opcode"] == opcode
            and row["elements"] == NORM_FLOOR_CALIBRATION_ELEMENT
        ])
        for opcode in NORM_OPCODE_NAMES
    }
    expected_pair_floor = (
        floor_summaries["sqrt"][0] + floor_summaries["div"][0],
        floor_summaries["sqrt"][1] + floor_summaries["div"][1],
        floor_summaries["sqrt"][2] + floor_summaries["div"][2],
    )
    floor_pair_common = _interval_overlap(
        expected_pair_floor, floor_summaries["sqrt_div"])
    floor_candidates_s = {
        "sqrt": floor_summaries["sqrt"][1] * 1e-9,
        "div": floor_summaries["div"][1] * 1e-9,
    }

    small_validation = {}
    sqrt_setup_ns = affine["sqrt"][
        "setup_time_ns_iqr_median"][1]
    sqrt_interval_ns = affine["sqrt"][
        "element_interval_ns_iqr_median"][1]
    for elements in NORM_FLOOR_VALIDATION_ELEMENTS:
        sqrt_prediction = max(
            floor_summaries["sqrt"][1],
            sqrt_setup_ns + elements * sqrt_interval_ns)
        div_prediction = max(
            floor_summaries["div"][1],
            FROZEN_DIV_SETUP_TIME_NS
            + elements * FROZEN_DIV_ELEMENT_INTERVAL_NS)
        predicted_by_opcode = {
            "sqrt": sqrt_prediction,
            "div": div_prediction,
            "sqrt_div": sqrt_prediction + div_prediction,
        }
        for opcode in NORM_OPCODE_NAMES:
            observed = _quartiles([
                row["per_iteration_ns"] for row in floor_validation_rows
                if row["opcode"] == opcode
                and row["elements"] == elements
            ])
            predicted = predicted_by_opcode[opcode]
            small_validation[(opcode, elements)] = {
                "predicted_ns": predicted,
                "observed_iqr_median_ns": observed,
                "common": observed[0] <= predicted <= observed[2],
            }

    mechanism_publishable = (
        tick_common
        and all(concurrency_common.values())
        and affine["sqrt"]["all_strata_positive"]
        and all(item["common"] for item in pair_affine_common.values())
        and all(
            item["common"]
            for item in frozen_div_steady_consistency.values())
        and all(
            item["common"]
            for item in steady_validation_intersections.values())
        and floor_pair_common["common"]
        and all(item["common"] for item in small_validation.values())
    )
    return {
        "independent_run_counts": run_counts,
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "concurrency_common": concurrency_common,
        "affine_intrinsic_parameters": affine,
        "pair_affine_common": pair_affine_common,
        "frozen_div_steady_consistency": frozen_div_steady_consistency,
        "steady_validation_intersections": steady_validation_intersections,
        "minimum_service_time_s_per_intrinsic": floor_candidates_s,
        "floor_pair_common": floor_pair_common,
        "small_validation": small_validation,
        "mechanism_publishable": mechanism_publishable,
    }


def _derive_serialized_service_affine(rows, opcode, run_ids):
    if len(run_ids) < 2:
        raise ValueError(
            "serialized service affine requires at least two calibration "
            "run IDs")
    strata = []
    for run_id in run_ids:
        for block_dim in BLOCK_DIMS:
            points = []
            for elements in SWIGLU_SERIALIZED_ELEMENTS:
                values = [
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ]
                if not values:
                    raise ValueError(
                        "missing serialized service stratum: "
                        f"opcode={opcode}, run_id={run_id}, "
                        f"block_dim={block_dim}, elements={elements}")
                points.append((elements, statistics.median(values)))
            mean_elements = statistics.fmean(
                elements for elements, _duration_ns in points)
            mean_duration_ns = statistics.fmean(
                duration_ns for _elements, duration_ns in points)
            denominator = math.fsum(
                (elements - mean_elements) ** 2
                for elements, _duration_ns in points)
            if denominator <= 0.0:
                raise AssertionError(
                    "serialized service calibration sizes must differ")
            element_interval_ns = math.fsum(
                (elements - mean_elements)
                * (duration_ns - mean_duration_ns)
                for elements, duration_ns in points) / denominator
            setup_time_ns = (
                mean_duration_ns - mean_elements * element_interval_ns)
            strata.append({
                "run_id": run_id,
                "block_dim": block_dim,
                "setup_time_ns": setup_time_ns,
                "element_interval_ns": element_interval_ns,
            })
    setup_summary = _quartiles([
        item["setup_time_ns"] for item in strata])
    interval_summary = _quartiles([
        item["element_interval_ns"] for item in strata])
    positive = all(
        item["setup_time_ns"] > 0.0
        and item["element_interval_ns"] > 0.0
        for item in strata)
    return {
        "strata": strata,
        "setup_time_ns_iqr_median": setup_summary,
        "element_interval_ns_iqr_median": interval_summary,
        "all_strata_positive": positive,
        "candidate_setup_time_s_per_serialized_intrinsic": (
            setup_summary[1] * 1e-9 if positive else None),
        "candidate_steady_rate_elements_per_s_per_aiv": (
            1.0 / (interval_summary[1] * 1e-9)
            if positive else None),
    }


def _audit_swiglu_serialized_services(rows):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "serialized service audit requires exactly "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    calibration_run_ids = tuple(
        run_ids[:SWIGLU_SERIALIZED_CALIBRATION_RUNS])
    validation_run_ids = tuple(
        run_ids[SWIGLU_SERIALIZED_CALIBRATION_RUNS:])
    if not validation_run_ids:
        raise AssertionError(
            "serialized service audit requires an independent validation run")
    expected = {
        (opcode, elements, block_dim)
        for opcode in SWIGLU_SERIALIZED_OPCODE_NAMES
        for elements in SWIGLU_SERIALIZED_ELEMENTS
        for block_dim in BLOCK_DIMS
    }
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            "serialized service rows are incomplete or contain extra cells")

    tick_summaries = {
        cell: _quartiles([
            row["tick_ns"] for row in rows
            if (row["opcode"], row["elements"], row["block_dim"])
            == cell
        ])
        for cell in expected
    }
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))

    concurrency_common = {}
    for opcode in SWIGLU_SERIALIZED_OPCODE_NAMES:
        for elements in SWIGLU_SERIALIZED_ELEMENTS:
            summaries = tuple(
                _quartiles([
                    row["per_iteration_ns"] for row in rows
                    if row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ])
                for block_dim in BLOCK_DIMS
            )
            concurrency_common[(opcode, elements)] = _intersection(
                summaries)[2]

    affine = {
        opcode: _derive_serialized_service_affine(
            rows, opcode, calibration_run_ids)
        for opcode in SWIGLU_SERIALIZED_OPCODE_NAMES
    }
    validation = {}
    for opcode in SWIGLU_SERIALIZED_OPCODE_NAMES:
        setup_ns = affine[opcode][
            "candidate_setup_time_s_per_serialized_intrinsic"] * 1e9
        rate = affine[opcode][
            "candidate_steady_rate_elements_per_s_per_aiv"]
        by_element = {}
        for elements in SWIGLU_SERIALIZED_ELEMENTS:
            observed_ns = statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["run_id"] in validation_run_ids
                and row["opcode"] == opcode
                and row["elements"] == elements
            ])
            predicted_ns = setup_ns + elements / rate * 1e9
            ape_percent = abs(predicted_ns - observed_ns) / observed_ns * 100.0
            by_element[elements] = {
                "predicted_ns": predicted_ns,
                "observed_median_ns": observed_ns,
                "ape_percent": ape_percent,
            }
        maximum_ape_percent = max(
            item["ape_percent"] for item in by_element.values())
        validation[opcode] = {
            "by_element": by_element,
            "maximum_ape_percent": maximum_ape_percent,
            "passed": (
                maximum_ape_percent
                <= SWIGLU_SERIALIZED_MAX_VALIDATION_APE_PERCENT),
        }
    service_publishable = {
        opcode: (
            tick_common
            and all(
                concurrency_common[(opcode, elements)]
                for elements in SWIGLU_SERIALIZED_ELEMENTS)
            and affine[opcode]["all_strata_positive"]
            and validation[opcode]["passed"]
        )
        for opcode in SWIGLU_SERIALIZED_OPCODE_NAMES
    }
    return {
        "independent_runs": len(run_ids),
        "calibration_run_ids": calibration_run_ids,
        "validation_run_ids": validation_run_ids,
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "concurrency_common": concurrency_common,
        "affine_serialized_service_parameters": affine,
        "validation": validation,
        "service_publishable": service_publishable,
        "mechanism_publishable": all(service_publishable.values()),
    }


def _audit_extended_serialized_vector_services(base_rows, h8192_rows):
    """Fit and validate one affine service across the extended size domain."""

    base_run_ids = sorted({row["run_id"] for row in base_rows})
    h8192_run_ids = sorted({row["run_id"] for row in h8192_rows})
    if (len(base_run_ids) != MINIMUM_INDEPENDENT_RUNS
            or len(h8192_run_ids) != MINIMUM_INDEPENDENT_RUNS):
        raise ValueError(
            "extended serialized service audit requires exactly three "
            "independent runs from each campaign")
    expected_h8192 = {
        (opcode, 8192, block_dim)
        for opcode in SERIALIZED_VECTOR_EXTENDED_OPCODE_NAMES
        for block_dim in BLOCK_DIMS
    }
    observed_h8192 = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in h8192_rows
    }
    if observed_h8192 != expected_h8192:
        raise ValueError(
            "H8192 serialized service rows are incomplete or contain extras")

    calibration_pairs = tuple(zip(
        base_run_ids[:SWIGLU_SERIALIZED_CALIBRATION_RUNS],
        h8192_run_ids[:SWIGLU_SERIALIZED_CALIBRATION_RUNS]))
    validation_run_ids = (
        base_run_ids[SWIGLU_SERIALIZED_CALIBRATION_RUNS:],
        h8192_run_ids[SWIGLU_SERIALIZED_CALIBRATION_RUNS:],
    )
    if any(len(run_ids) != 1 for run_ids in validation_run_ids):
        raise AssertionError(
            "extended serialized service audit requires one independent "
            "validation run from each campaign")

    tick_summaries = {}
    concurrency_common = {}
    for opcode in SERIALIZED_VECTOR_EXTENDED_OPCODE_NAMES:
        for elements in SERIALIZED_VECTOR_EXTENDED_ELEMENTS:
            source_rows = h8192_rows if elements == 8192 else base_rows
            summaries = tuple(
                _quartiles([
                    row["per_iteration_ns"] for row in source_rows
                    if row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ])
                for block_dim in BLOCK_DIMS
            )
            concurrency_common[(opcode, elements)] = _intersection(
                summaries)[2]
            for block_dim in BLOCK_DIMS:
                tick_summaries[(opcode, elements, block_dim)] = _quartiles([
                    row["tick_ns"] for row in source_rows
                    if row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ])
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))

    affine = {}
    validation = {}
    service_publishable = {}
    for opcode in SERIALIZED_VECTOR_EXTENDED_OPCODE_NAMES:
        strata = []
        for base_run_id, h8192_run_id in calibration_pairs:
            for block_dim in BLOCK_DIMS:
                points = []
                for elements in SERIALIZED_VECTOR_EXTENDED_ELEMENTS:
                    source_rows = (
                        h8192_rows if elements == 8192 else base_rows)
                    run_id = (
                        h8192_run_id if elements == 8192 else base_run_id)
                    values = [
                        row["per_iteration_ns"] for row in source_rows
                        if row["run_id"] == run_id
                        and row["opcode"] == opcode
                        and row["elements"] == elements
                        and row["block_dim"] == block_dim
                    ]
                    if not values:
                        raise ValueError(
                            "missing extended serialized service stratum: "
                            f"opcode={opcode}, run_id={run_id}, "
                            f"block_dim={block_dim}, elements={elements}")
                    points.append((elements, statistics.median(values)))
                mean_elements = statistics.fmean(
                    elements for elements, _duration_ns in points)
                mean_duration_ns = statistics.fmean(
                    duration_ns for _elements, duration_ns in points)
                denominator = math.fsum(
                    (elements - mean_elements) ** 2
                    for elements, _duration_ns in points)
                element_interval_ns = math.fsum(
                    (elements - mean_elements)
                    * (duration_ns - mean_duration_ns)
                    for elements, duration_ns in points) / denominator
                setup_time_ns = (
                    mean_duration_ns - mean_elements * element_interval_ns)
                strata.append({
                    "base_run_id": base_run_id,
                    "h8192_run_id": h8192_run_id,
                    "block_dim": block_dim,
                    "setup_time_ns": setup_time_ns,
                    "element_interval_ns": element_interval_ns,
                })
        setup_summary = _quartiles([
            item["setup_time_ns"] for item in strata])
        interval_summary = _quartiles([
            item["element_interval_ns"] for item in strata])
        positive = all(
            item["setup_time_ns"] > 0.0
            and item["element_interval_ns"] > 0.0
            for item in strata)
        setup_time_s = setup_summary[1] * 1e-9 if positive else None
        steady_rate = (
            1.0 / (interval_summary[1] * 1e-9) if positive else None)
        affine[opcode] = {
            "strata": strata,
            "setup_time_ns_iqr_median": setup_summary,
            "element_interval_ns_iqr_median": interval_summary,
            "all_strata_positive": positive,
            "candidate_setup_time_s_per_serialized_intrinsic": setup_time_s,
            "candidate_steady_rate_elements_per_s_per_aiv": steady_rate,
        }

        by_element = {}
        for elements in SERIALIZED_VECTOR_EXTENDED_ELEMENTS:
            source_rows = h8192_rows if elements == 8192 else base_rows
            run_id = (
                validation_run_ids[1][0]
                if elements == 8192 else validation_run_ids[0][0])
            observed_ns = statistics.median([
                row["per_iteration_ns"] for row in source_rows
                if row["run_id"] == run_id
                and row["opcode"] == opcode
                and row["elements"] == elements
            ])
            predicted_ns = setup_time_s * 1e9 + elements / steady_rate * 1e9
            by_element[elements] = {
                "observed_ns": observed_ns,
                "predicted_ns": predicted_ns,
                "ape_percent": (
                    abs(predicted_ns - observed_ns) / observed_ns * 100.0),
            }
        maximum_ape = max(
            item["ape_percent"] for item in by_element.values())
        validation[opcode] = {
            "by_element": by_element,
            "maximum_ape_percent": maximum_ape,
        }
        service_publishable[opcode] = (
            tick_common
            and all(
                concurrency_common[(opcode, elements)]
                for elements in SERIALIZED_VECTOR_EXTENDED_ELEMENTS)
            and positive
            and maximum_ape
            <= SWIGLU_SERIALIZED_MAX_VALIDATION_APE_PERCENT)

    return {
        "base_run_ids": base_run_ids,
        "h8192_run_ids": h8192_run_ids,
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "concurrency_common": concurrency_common,
        "affine_serialized_service_parameters": affine,
        "validation": validation,
        "service_publishable": service_publishable,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": all(service_publishable.values()),
    }


def _audit_aiv_worker_prologue(rows):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "AIV worker prologue audit requires exactly "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    expected = {
        ("two_tbuf_mask_prologue", elements, block_dim)
        for elements in (2048, 4096)
        for block_dim in BLOCK_DIMS
    }
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            "AIV worker prologue rows are incomplete or contain extra cells")
    tick_summaries = {
        cell: _quartiles([
            row["tick_ns"] for row in rows
            if (row["opcode"], row["elements"], row["block_dim"]) == cell
        ])
        for cell in expected
    }
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    run_spreads = {}
    for elements in (2048, 4096):
        for block_dim in BLOCK_DIMS:
            medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ])
                for run_id in run_ids
            )
            run_spreads[(elements, block_dim)] = (
                max(medians) - min(medians)) / statistics.median(medians)
    concurrency_spreads = {}
    services = {}
    for elements in (2048, 4096):
        block_medians = tuple(
            statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["elements"] == elements
                and row["block_dim"] == block_dim
            ])
            for block_dim in BLOCK_DIMS
        )
        concurrency_spreads[elements] = (
            max(block_medians) - min(block_medians)
        ) / statistics.median(block_medians)
        services[elements] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["elements"] == elements
        ])
    publishable = (
        tick_common
        and max(run_spreads.values()) <= 0.10
        and max(concurrency_spreads.values()) <= 0.10
    )
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_worker_by_mask_elements": services,
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
    }


def _audit_single_tbuf_init(rows):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "single-TBuf-init audit requires exactly "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    expected = {
        ("single_tbuf_init", buffer_bytes, block_dim)
        for buffer_bytes in (1024, 2048)
        for block_dim in BLOCK_DIMS
    }
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            "single-TBuf-init rows are incomplete or contain extra cells")
    run_spreads = {}
    concurrency_spreads = {}
    services = {}
    for buffer_bytes in (1024, 2048):
        block_medians = []
        for block_dim in BLOCK_DIMS:
            medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["elements"] == buffer_bytes
                    and row["block_dim"] == block_dim
                ])
                for run_id in run_ids)
            run_spreads[(buffer_bytes, block_dim)] = (
                max(medians) - min(medians)) / statistics.median(medians)
            block_medians.append(statistics.median(medians))
        concurrency_spreads[buffer_bytes] = (
            max(block_medians) - min(block_medians)
        ) / statistics.median(block_medians)
        services[buffer_bytes] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["elements"] == buffer_bytes
        ])
    publishable = (
        max(run_spreads.values()) <= 0.10
        and max(concurrency_spreads.values()) <= 0.10)
    return {
        "independent_runs": len(run_ids),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_worker_by_buffer_bytes": services,
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
    }


def _audit_scalar_integer_dependency(
        rows, opcode_definitions=SCALAR_INTEGER_DEPENDENCY_OPCODES):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "scalar-integer-dependency audit requires exactly "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    opcode_names = tuple(
        opcode_name
        for _opcode, opcode_name in opcode_definitions)
    operands = (3, 47)
    expected = {
        (opcode_name, operand, block_dim)
        for opcode_name in opcode_names
        for operand in operands
        for block_dim in BLOCK_DIMS
    }
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            "scalar-integer-dependency rows are incomplete or contain "
            "extra cells")
    tick_summaries = {
        cell: _quartiles([
            row["tick_ns"] for row in rows
            if (row["opcode"], row["elements"], row["block_dim"])
            == cell
        ])
        for cell in expected
    }
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    run_spreads = {}
    concurrency_spreads = {}
    operand_services = {}
    operand_spreads = {}
    services = {}
    for opcode_name in opcode_names:
        for operand in operands:
            block_medians = []
            for block_dim in BLOCK_DIMS:
                medians = tuple(
                    statistics.median([
                        row["per_iteration_ns"] for row in rows
                        if row["run_id"] == run_id
                        and row["opcode"] == opcode_name
                        and row["elements"] == operand
                        and row["block_dim"] == block_dim
                    ])
                    for run_id in run_ids)
                run_spreads[(opcode_name, operand, block_dim)] = (
                    max(medians) - min(medians)
                ) / statistics.median(medians)
                block_medians.append(statistics.median(medians))
            concurrency_spreads[(opcode_name, operand)] = (
                max(block_medians) - min(block_medians)
            ) / statistics.median(block_medians)
            operand_services[(opcode_name, operand)] = statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["opcode"] == opcode_name
                and row["elements"] == operand
            ])
        opcode_operand_services = tuple(
            operand_services[(opcode_name, operand)]
            for operand in operands)
        operand_spreads[opcode_name] = (
            max(opcode_operand_services) - min(opcode_operand_services)
        ) / statistics.median(opcode_operand_services)
        services[opcode_name] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["opcode"] == opcode_name
        ])
    publishable = (
        tick_common
        and max(run_spreads.values()) <= 0.10
        and max(concurrency_spreads.values()) <= 0.10
        and max(operand_spreads.values()) <= 0.10)
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "maximum_operand_relative_median_spread": max(
            operand_spreads.values()),
        "service_time_ns_per_dependency_per_aiv": services,
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
    }


def _audit_swiglu_control_service(rows, opcode_definitions):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "SwiGLU control audit requires exactly "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    opcode_names = tuple(name for _opcode, name in opcode_definitions)
    tick_values = {row["tick_ns"] for row in rows}
    if tick_values != {SYSTEM_COUNTER_TICK_NS}:
        raise ValueError("SwiGLU control SYS_CNT tick changed")

    cell_medians = {}
    run_spreads_ns = {}
    concurrency_spreads_ns = {}
    element_spreads_ns = {}
    services = {}
    for opcode_name in opcode_names:
        elements = sorted({
            row["elements"] for row in rows
            if row["opcode"] == opcode_name})
        for element_count in elements:
            block_medians = []
            for block_dim in BLOCK_DIMS:
                run_medians = []
                for run_id in run_ids:
                    samples = [
                        sample
                        for row in rows
                        if row["run_id"] == run_id
                        and row["opcode"] == opcode_name
                        and row["elements"] == element_count
                        and row["block_dim"] == block_dim
                        for sample in row["per_worker_sample_ns"]
                    ]
                    run_medians.append(statistics.median(samples))
                run_spreads_ns[(opcode_name, element_count, block_dim)] = (
                    max(run_medians) - min(run_medians))
                block_medians.append(statistics.median(run_medians))
            concurrency_spreads_ns[(opcode_name, element_count)] = (
                max(block_medians) - min(block_medians))
            cell_medians[(opcode_name, element_count)] = statistics.median([
                sample
                for row in rows
                if row["opcode"] == opcode_name
                and row["elements"] == element_count
                for sample in row["per_worker_sample_ns"]
            ])
        opcode_cell_medians = tuple(
            cell_medians[(opcode_name, element_count)]
            for element_count in elements)
        element_spreads_ns[opcode_name] = (
            max(opcode_cell_medians) - min(opcode_cell_medians))
        services[opcode_name] = statistics.median([
            sample
            for row in rows
            if row["opcode"] == opcode_name
            for sample in row["per_worker_sample_ns"]
        ])
    maximum_quantized_spread_ns = max((
        *run_spreads_ns.values(),
        *concurrency_spreads_ns.values(),
        *element_spreads_ns.values(),
    ))
    publishable = maximum_quantized_spread_ns <= SYSTEM_COUNTER_TICK_NS
    return {
        "independent_runs": len(run_ids),
        "tick_ns": SYSTEM_COUNTER_TICK_NS,
        "maximum_quantized_spread_ns": maximum_quantized_spread_ns,
        "exact_service_time_ns_per_request_per_aiv": services,
        "mechanism_publishable": publishable,
        "production_or_evaluation_duration_consumed": False,
    }


def _audit_local_scalar_dependency(
        rows, opcode_name, mechanism_name, provenance_gate_passed):
    if not isinstance(provenance_gate_passed, bool):
        raise TypeError("provenance_gate_passed must be bool")
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            f"{mechanism_name} audit requires exactly three runs")
    expected = {
        (opcode_name, 1, block_dim)
        for block_dim in BLOCK_DIMS}
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            f"{mechanism_name} rows are incomplete or contain extras")
    run_spreads = {}
    for block_dim in BLOCK_DIMS:
        medians = tuple(
            statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["run_id"] == run_id
                and row["block_dim"] == block_dim])
            for run_id in run_ids)
        run_spreads[block_dim] = (
            max(medians) - min(medians)) / statistics.median(medians)
    block_medians = tuple(
        statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["block_dim"] == block_dim])
        for block_dim in BLOCK_DIMS)
    concurrency_spread = (
        max(block_medians) - min(block_medians)
    ) / statistics.median(block_medians)
    tick_summaries = {
        block_dim: _quartiles([
            row["tick_ns"] for row in rows
            if row["block_dim"] == block_dim])
        for block_dim in BLOCK_DIMS}
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    measurement_gate_passed = (
        tick_common
        and max(run_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and concurrency_spread <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD)
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "concurrency_relative_median_spread": concurrency_spread,
        "service_time_ns_per_request_per_aiv": statistics.median([
            row["per_iteration_ns"] for row in rows]),
        "measurement_gate_passed": measurement_gate_passed,
        "provenance_gate_passed": provenance_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": (
            measurement_gate_passed and provenance_gate_passed),
    }


def _audit_rmsnorm_local_scalar_dependency(rows, provenance_gate_passed):
    return _audit_local_scalar_dependency(
        rows, "rmsnorm_local_scalar_dependency", "RMSNorm local scalar",
        provenance_gate_passed)


def _audit_add_rmsnorm_local_scalar_handoff(
        rows, provenance_gate_passed):
    return _audit_local_scalar_dependency(
        rows, "add_rmsnorm_local_scalar_handoff",
        "AddRMSNorm local scalar handoff", provenance_gate_passed)


def _audit_add_rmsnorm_split_d_local_scalar_handoff(
        rows, provenance_gate_passed):
    """Audit key-31's GetValue handoff against the key-30 control."""

    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "AddRMSNorm Split-D handoff audit requires exactly three runs")
    target = "add_rmsnorm_split_d_local_scalar_handoff"
    control = "add_rmsnorm_local_scalar_handoff"
    expected = {
        (opcode, 1, block_dim)
        for opcode in (target, control)
        for block_dim in ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_BLOCK_DIMS}
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows}
    if observed != expected:
        raise ValueError(
            "AddRMSNorm Split-D handoff rows are incomplete or contain "
            "extras")
    run_spreads = {}
    concurrency_spreads = {}
    service_times = {}
    for opcode in (target, control):
        block_medians = []
        for block_dim in ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_BLOCK_DIMS:
            medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["opcode"] == opcode
                    and row["block_dim"] == block_dim])
                for run_id in run_ids)
            run_spreads[(opcode, block_dim)] = (
                max(medians) - min(medians)) / statistics.median(medians)
            block_medians.append(statistics.median(medians))
        concurrency_spreads[opcode] = (
            max(block_medians) - min(block_medians)
        ) / statistics.median(block_medians)
        service_times[opcode] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["opcode"] == opcode])
    tick_summaries = {
        (opcode, block_dim): _quartiles([
            row["tick_ns"] for row in rows
            if row["opcode"] == opcode
            and row["block_dim"] == block_dim])
        for opcode in (target, control)
        for block_dim in ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_BLOCK_DIMS}
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    measurement_gate_passed = (
        tick_common
        and max(run_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and max(concurrency_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD)
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_request_per_aiv": service_times[target],
        "control_time_ns_per_request_per_aiv": service_times[control],
        "measurement_gate_passed": measurement_gate_passed,
        "provenance_gate_passed": provenance_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": (
            measurement_gate_passed and provenance_gate_passed),
    }


def _audit_vector_to_mte2_fence(rows, provenance_gate_passed):
    return _audit_local_scalar_dependency(
        rows, "vector_to_mte2_fence", "Vector-to-MTE2 fence",
        provenance_gate_passed)


def _audit_source_affine_reduce(
        rows, *, opcode, label, element_counts, calibration_elements,
        validation_elements, maximum_validation_ape_percent, block_dims):
    maximum_allowed_validation_ape_percent = maximum_validation_ape_percent
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            f"{label} audit requires exactly three runs")
    expected = {
        (opcode, elements, block_dim)
        for elements in element_counts
        for block_dim in block_dims}
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows}
    if observed != expected:
        raise ValueError(
            f"{label} rows are incomplete or contain extras")
    run_spreads = {}
    for elements in element_counts:
        for block_dim in block_dims:
            medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim])
                for run_id in run_ids)
            run_spreads[(elements, block_dim)] = (
                max(medians) - min(medians)) / statistics.median(medians)
    concurrency_spreads = {}
    services = {}
    for elements in element_counts:
        block_medians = tuple(
            statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["elements"] == elements
                and row["block_dim"] == block_dim])
            for block_dim in block_dims)
        concurrency_spreads[elements] = (
            max(block_medians) - min(block_medians)
        ) / statistics.median(block_medians)
        services[elements] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["elements"] == elements])
    tick_summaries = {
        (elements, block_dim): _quartiles([
            row["tick_ns"] for row in rows
            if row["elements"] == elements
            and row["block_dim"] == block_dim])
        for elements in element_counts
        for block_dim in block_dims}
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    affine_strata = []
    for run_id in run_ids:
        for block_dim in block_dims:
            points = tuple(
                (elements, statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim]))
                for elements in calibration_elements)
            mean_elements = statistics.fmean(
                elements for elements, _duration_ns in points)
            mean_duration_ns = statistics.fmean(
                duration_ns for _elements, duration_ns in points)
            denominator = math.fsum(
                (elements - mean_elements) ** 2
                for elements, _duration_ns in points)
            if denominator <= 0.0:
                raise AssertionError(
                    f"{label} calibration sizes must differ")
            interval_ns = math.fsum(
                (elements - mean_elements)
                * (duration_ns - mean_duration_ns)
                for elements, duration_ns in points) / denominator
            setup_ns = mean_duration_ns - mean_elements * interval_ns
            affine_strata.append((setup_ns, interval_ns))
    setup_ns = statistics.median(
        value[0] for value in affine_strata)
    interval_ns = statistics.median(
        value[1] for value in affine_strata)
    affine_positive = setup_ns > 0.0 and interval_ns > 0.0
    validation_ape_percent = {}
    for elements in validation_elements:
        observed_ns = services[elements]
        predicted_ns = setup_ns + elements * interval_ns
        validation_ape_percent[elements] = (
            abs(predicted_ns - observed_ns) / observed_ns * 100.0)
    observed_maximum_validation_ape_percent = max(
        validation_ape_percent.values())
    measurement_gate_passed = (
        tick_common
        and max(run_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and max(concurrency_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and affine_positive
        and observed_maximum_validation_ape_percent
        <= maximum_allowed_validation_ape_percent)
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_request_per_aiv_by_elements": services,
        "candidate_setup_time_s_per_invocation": setup_ns * 1e-9,
        "candidate_service_rate_elements_per_s_per_aiv": (
            1.0 / (interval_ns * 1e-9)),
        "validation_ape_percent_by_elements": validation_ape_percent,
        "maximum_allowed_validation_ape_percent": (
            maximum_allowed_validation_ape_percent),
        "maximum_validation_ape_percent": (
            observed_maximum_validation_ape_percent),
        "measurement_gate_passed": measurement_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": measurement_gate_passed,
    }


def _audit_rmsnorm_half_interval_reduce(rows):
    return _audit_source_affine_reduce(
        rows,
        opcode="rmsnorm_half_interval_reduce",
        label="RMSNorm half-interval reduction",
        element_counts=RMSNORM_HALF_INTERVAL_REDUCE_ELEMENTS,
        calibration_elements=RMSNORM_HALF_INTERVAL_CALIBRATION_ELEMENTS,
        validation_elements=RMSNORM_HALF_INTERVAL_VALIDATION_ELEMENTS,
        maximum_validation_ape_percent=(
            RMSNORM_HALF_INTERVAL_MAX_VALIDATION_APE_PERCENT),
        block_dims=BLOCK_DIMS,
    )


def _audit_add_rmsnorm_reduce_sum_fp32(rows):
    return _audit_source_affine_reduce(
        rows,
        opcode="add_rmsnorm_reduce_sum_fp32",
        label="AddRMSNorm ReduceSumFP32",
        element_counts=ADD_RMSNORM_REDUCE_SUM_FP32_ELEMENTS,
        calibration_elements=(
            ADD_RMSNORM_REDUCE_SUM_FP32_CALIBRATION_ELEMENTS),
        validation_elements=(
            ADD_RMSNORM_REDUCE_SUM_FP32_VALIDATION_ELEMENTS),
        maximum_validation_ape_percent=(
            ADD_RMSNORM_REDUCE_SUM_FP32_MAX_VALIDATION_APE_PERCENT),
        block_dims=BLOCK_DIMS,
    )


def _audit_add_rmsnorm_split_d_block_reduce_sum_fp32(rows):
    return _audit_source_affine_reduce(
        rows,
        opcode="add_rmsnorm_split_d_block_reduce_sum_fp32",
        label="AddRMSNorm Split-D BlockReduceSumFP32",
        element_counts=ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_ELEMENTS,
        calibration_elements=(
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CALIBRATION_ELEMENTS),
        validation_elements=(
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_VALIDATION_ELEMENTS),
        maximum_validation_ape_percent=(
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_MAX_VALIDATION_APE_PERCENT),
        block_dims=(
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_BLOCK_DIMS),
    )


def _block_reduce_sum_source_branches(elements):
    if elements <= 0 or elements % 8 != 0:
        raise ValueError(
            "BlockReduceSumFP32 source count must be a positive multiple "
            "of eight")
    return elements // 64, elements % 64


def _audit_add_rmsnorm_split_d_block_reduce_sum_composed(rows):
    """Audit CANN's static full-repeat plus tail BlockReduceSum program."""

    label = "AddRMSNorm Split-D BlockReduceSumFP32 composed source service"
    opcode = "add_rmsnorm_split_d_block_reduce_sum_fp32"
    elements = ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_ELEMENTS
    block_dims = ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_BLOCK_DIMS
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(f"{label} audit requires exactly three runs")
    expected = {
        (opcode, element_count, block_dim)
        for element_count in elements
        for block_dim in block_dims}
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows}
    if observed != expected:
        raise ValueError(f"{label} rows are incomplete or contain extras")

    full_calibration = (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_BODY_CALIBRATION_ELEMENTS)
    full_validation = (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_BODY_VALIDATION_ELEMENTS)
    tail_calibration = (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_TAIL_CALIBRATION_ELEMENTS)
    tail_validation = (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_TAIL_VALIDATION_ELEMENTS)
    composite_validation = (
        ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_COMPOSITE_VALIDATION_ELEMENTS)
    if (any(_block_reduce_sum_source_branches(value)[1] != 0
            for value in full_calibration + full_validation)
            or any(_block_reduce_sum_source_branches(value)[0] != 0
                   for value in tail_calibration + tail_validation)
            or any(0 in _block_reduce_sum_source_branches(value)
                   for value in composite_validation)):
        raise AssertionError(
            "BlockReduceSumFP32 composed audit source branch sets are invalid")

    run_spreads = {}
    services = {}
    concurrency_spreads = {}
    tick_summaries = {}
    for element_count in elements:
        block_medians = []
        for block_dim in block_dims:
            run_medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["elements"] == element_count
                    and row["block_dim"] == block_dim])
                for run_id in run_ids)
            run_spreads[(element_count, block_dim)] = (
                (max(run_medians) - min(run_medians))
                / statistics.median(run_medians))
            block_medians.append(statistics.median(run_medians))
            tick_summaries[(element_count, block_dim)] = _quartiles([
                row["tick_ns"] for row in rows
                if row["elements"] == element_count
                and row["block_dim"] == block_dim])
        concurrency_spreads[element_count] = (
            (max(block_medians) - min(block_medians))
            / statistics.median(block_medians))
        services[element_count] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["elements"] == element_count])
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))

    body_strata = []
    for run_id in run_ids:
        for block_dim in block_dims:
            points = tuple(
                (_block_reduce_sum_source_branches(element_count)[0],
                 statistics.median([
                     row["per_iteration_ns"] for row in rows
                     if row["run_id"] == run_id
                     and row["elements"] == element_count
                     and row["block_dim"] == block_dim]))
                for element_count in full_calibration)
            mean_repeats = statistics.fmean(
                repeat_count for repeat_count, _duration_ns in points)
            mean_duration_ns = statistics.fmean(
                duration_ns for _repeat_count, duration_ns in points)
            denominator = math.fsum(
                (repeat_count - mean_repeats) ** 2
                for repeat_count, _duration_ns in points)
            if denominator <= 0.0:
                raise AssertionError(
                    "BlockReduceSumFP32 full-repeat calibration is degenerate")
            interval_ns = math.fsum(
                (repeat_count - mean_repeats)
                * (duration_ns - mean_duration_ns)
                for repeat_count, duration_ns in points) / denominator
            body_strata.append((
                mean_duration_ns - mean_repeats * interval_ns,
                interval_ns))
    full_setup_ns = statistics.median(
        setup_ns for setup_ns, _interval_ns in body_strata)
    full_repeat_interval_ns = statistics.median(
        interval_ns for _setup_ns, interval_ns in body_strata)
    tail_service_ns = statistics.median([
        row["per_iteration_ns"] for row in rows
        if row["elements"] in tail_calibration])
    if (full_setup_ns <= 0.0 or full_repeat_interval_ns <= 0.0
            or tail_service_ns <= 0.0):
        raise ValueError(
            "BlockReduceSumFP32 composed source service is not positive")

    def _predicted_ns(element_count):
        full_repeats, tail_elements = _block_reduce_sum_source_branches(
            element_count)
        full_ns = (
            0.0 if full_repeats == 0 else full_setup_ns
            + full_repeats * full_repeat_interval_ns)
        tail_ns = 0.0 if tail_elements == 0 else tail_service_ns
        return full_ns + tail_ns

    validation_ape_percent = {
        element_count: abs(
            _predicted_ns(element_count) - services[element_count])
        / services[element_count] * 100.0
        for element_count in (
            full_validation + tail_validation + composite_validation)}
    maximum_validation_ape_percent = max(validation_ape_percent.values())
    measurement_gate_passed = (
        tick_common
        and max(run_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and max(concurrency_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and maximum_validation_ape_percent
        <= ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_MAX_VALIDATION_APE_PERCENT)
    return {
        "source_branch_model": (
            "cann_8_5_1_block_reduce_sum_fp32_full_repeat_plus_tail_v1"),
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_request_per_aiv_by_elements": services,
        "full_repeat_setup_time_s_per_invocation": full_setup_ns * 1e-9,
        "full_repeat_interval_s_per_repeat": full_repeat_interval_ns * 1e-9,
        "tail_service_time_s_per_invocation": tail_service_ns * 1e-9,
        "validation_ape_percent_by_elements": validation_ape_percent,
        "maximum_allowed_validation_ape_percent": (
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_MAX_VALIDATION_APE_PERCENT),
        "maximum_validation_ape_percent": maximum_validation_ape_percent,
        "measurement_gate_passed": measurement_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": measurement_gate_passed,
    }


def _audit_local_scalar_fill_sv(rows):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError("local scalar fill audit requires exactly three runs")
    expected = {
        ("local_scalar_fill_s_v", writes, block_dim)
        for writes in (64, 128)
        for block_dim in BLOCK_DIMS}
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows}
    if observed != expected:
        raise ValueError(
            "local scalar fill rows are incomplete or contain extras")
    run_spreads = {}
    for writes in (64, 128):
        for block_dim in BLOCK_DIMS:
            medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["run_id"] == run_id
                    and row["elements"] == writes
                    and row["block_dim"] == block_dim])
                for run_id in run_ids)
            run_spreads[(writes, block_dim)] = (
                max(medians) - min(medians)) / statistics.median(medians)
    concurrency_spreads = {}
    services = {}
    for writes in (64, 128):
        block_medians = tuple(
            statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["elements"] == writes
                and row["block_dim"] == block_dim])
            for block_dim in BLOCK_DIMS)
        concurrency_spreads[writes] = (
            max(block_medians) - min(block_medians)
        ) / statistics.median(block_medians)
        services[writes] = statistics.median([
            row["per_iteration_ns"] for row in rows
            if row["elements"] == writes])
    tick_summaries = {
        (writes, block_dim): _quartiles([
            row["tick_ns"] for row in rows
            if row["elements"] == writes
            and row["block_dim"] == block_dim])
        for writes in (64, 128)
        for block_dim in BLOCK_DIMS}
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    publishable = (
        tick_common
        and max(run_spreads.values()) <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and max(concurrency_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD)
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_request_per_aiv_by_scalar_writes": services,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": publishable,
    }


def _audit_local_ub_copy(rows):
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError("local UB copy audit requires exactly three runs")
    expected = {
        (opcode, payload_bytes, block_dim)
        for _opcode_id, opcode in LOCAL_UB_COPY_OPCODES
        for payload_bytes in LOCAL_UB_COPY_ELEMENTS[opcode]
        for block_dim in BLOCK_DIMS
    }
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows}
    if observed != expected:
        raise ValueError(
            "local UB copy rows are incomplete or contain extras")
    run_spreads = {}
    for opcode, payload_bytes, block_dim in expected:
        medians = tuple(
            statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["run_id"] == run_id
                and row["opcode"] == opcode
                and row["elements"] == payload_bytes
                and row["block_dim"] == block_dim
            ])
            for run_id in run_ids)
        run_spreads[(opcode, payload_bytes, block_dim)] = (
            max(medians) - min(medians)) / statistics.median(medians)
    concurrency_spreads = {}
    services = {}
    for _opcode_id, opcode in LOCAL_UB_COPY_OPCODES:
        services[opcode] = {}
        for payload_bytes in LOCAL_UB_COPY_ELEMENTS[opcode]:
            block_medians = tuple(
                statistics.median([
                    row["per_iteration_ns"] for row in rows
                    if row["opcode"] == opcode
                    and row["elements"] == payload_bytes
                    and row["block_dim"] == block_dim
                ])
                for block_dim in BLOCK_DIMS)
            concurrency_spreads[(opcode, payload_bytes)] = (
                max(block_medians) - min(block_medians)
                ) / statistics.median(block_medians)
            services[opcode][payload_bytes] = statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["opcode"] == opcode
                and row["elements"] == payload_bytes
            ])
    tick_summaries = {
        cell: _quartiles([
            row["tick_ns"] for row in rows
            if (row["opcode"], row["elements"], row["block_dim"])
            == cell
        ])
        for cell in expected}
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))
    publishable = (
        tick_common
        and max(run_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD
        and max(concurrency_spreads.values())
        <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD)
    return {
        "independent_runs": len(run_ids),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "maximum_run_relative_median_spread": max(run_spreads.values()),
        "maximum_concurrency_relative_median_spread": max(
            concurrency_spreads.values()),
        "service_time_ns_per_request_per_aiv_by_opcode_and_payload_bytes": (
            services),
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": publishable,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_norm_request_rows(paths):
    cells = _campaign_cells(NORM_REQUEST_SERVICE_CAMPAIGN)
    expected_cells = {
        (opcode_name, elements, block_dim)
        for _opcode, opcode_name, elements, block_dim in cells
    }
    rows = _read_campaign(
        paths,
        expected_cells,
        lambda elements: _iterations_for_campaign(
            NORM_REQUEST_SERVICE_CAMPAIGN, elements),
    )
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "NORM request service audit requires exactly "
            f"{MINIMUM_INDEPENDENT_RUNS} independent run IDs")
    by_trial = {}
    for row in rows:
        trial_key = (row["run_id"], row["trial"])
        if trial_key not in by_trial:
            by_trial[trial_key] = {}
        if row["cell"] in by_trial[trial_key]:
            raise ValueError(
                f"NORM request trial {trial_key} duplicates {row['cell']}")
        by_trial[trial_key][row["cell"]] = row["order_position"]
    expected_trials = set(range(1, NORM_REQUEST_TRIALS_PER_RUN + 1))
    for run_id in run_ids:
        observed_trials = {
            row["trial"] for row in rows if row["run_id"] == run_id}
        if observed_trials != expected_trials:
            raise ValueError(
                f"NORM request run {run_id!r} must contain exactly "
                f"{NORM_REQUEST_TRIALS_PER_RUN} complete trials")
    for trial_key, trial_cells in by_trial.items():
        if set(trial_cells) != expected_cells:
            raise ValueError(f"NORM request trial {trial_key} is incomplete")
        if set(trial_cells.values()) != set(range(len(cells))):
            raise ValueError(
                f"NORM request trial {trial_key} has incomplete positions")
        trial_index = trial_key[1] - 1
        for cell, order_position in trial_cells.items():
            expected_cell = cells[_norm_request_cell_index(
                trial_index, order_position, len(cells))]
            expected_identity = (
                expected_cell[1], expected_cell[2], expected_cell[3])
            if cell != expected_identity:
                raise ValueError(
                    f"NORM request trial {trial_key} position "
                    f"{order_position} changed the frozen schedule")
    return tuple(rows)


def _audit_norm_request_services(rows, provenance_gate_passed):
    if not isinstance(provenance_gate_passed, bool):
        raise TypeError("provenance_gate_passed must be bool")
    run_ids = sorted({row["run_id"] for row in rows})
    if len(run_ids) != MINIMUM_INDEPENDENT_RUNS:
        raise ValueError(
            "NORM request service audit requires exactly three runs")
    expected = {
        (opcode, elements, block_dim)
        for opcode in NORM_REQUEST_OPCODE_NAMES
        for elements in NORM_REQUEST_ELEMENTS
        for block_dim in BLOCK_DIMS
    }
    observed = {
        (row["opcode"], row["elements"], row["block_dim"])
        for row in rows
    }
    if observed != expected:
        raise ValueError(
            "NORM request rows are incomplete or contain extra cells")

    tick_summaries = {
        cell: _quartiles([
            row["tick_ns"] for row in rows
            if (row["opcode"], row["elements"], row["block_dim"])
            == cell
        ])
        for cell in expected
    }
    tick_lower, tick_upper, tick_common = _intersection(
        tuple(tick_summaries.values()))

    run_consistency = {}
    run_relative_median_spread = {}
    for cell in expected:
        summaries = tuple(
            _quartiles([
                row["per_iteration_ns"] for row in rows
                if row["run_id"] == run_id
                and (row["opcode"], row["elements"], row["block_dim"])
                == cell
            ])
            for run_id in run_ids
        )
        medians = tuple(summary[1] for summary in summaries)
        relative_spread = (
            (max(medians) - min(medians)) / statistics.median(medians))
        run_relative_median_spread[cell] = relative_spread
        run_consistency[cell] = (
            relative_spread <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD)

    concurrency_common = {}
    concurrency_relative_median_spread = {}
    for opcode in NORM_REQUEST_OPCODE_NAMES:
        for elements in NORM_REQUEST_ELEMENTS:
            summaries = tuple(
                _quartiles([
                    row["per_iteration_ns"] for row in rows
                    if row["opcode"] == opcode
                    and row["elements"] == elements
                    and row["block_dim"] == block_dim
                ])
                for block_dim in BLOCK_DIMS
            )
            medians = tuple(summary[1] for summary in summaries)
            relative_spread = (
                (max(medians) - min(medians)) / statistics.median(medians))
            concurrency_relative_median_spread[(opcode, elements)] = (
                relative_spread)
            concurrency_common[(opcode, elements)] = (
                relative_spread <= NORM_REQUEST_MAX_RELATIVE_MEDIAN_SPREAD)

    service_ns = {
        opcode: {
            elements: statistics.median([
                row["per_iteration_ns"] for row in rows
                if row["opcode"] == opcode
                and row["elements"] == elements
            ])
            for elements in NORM_REQUEST_ELEMENTS
        }
        for opcode in NORM_REQUEST_OPCODE_NAMES
    }
    service_iqr_ns = {
        opcode: {
            elements: _quartiles([
                row["per_iteration_ns"] for row in rows
                if row["opcode"] == opcode
                and row["elements"] == elements
            ])
            for elements in NORM_REQUEST_ELEMENTS
        }
        for opcode in NORM_REQUEST_OPCODE_NAMES
    }
    measurement_gate_passed = (
        tick_common
        and all(run_consistency.values())
        and all(concurrency_common.values())
    )
    return {
        "independent_runs": len(run_ids),
        "samples_per_cell": len(rows) // len(expected),
        "tick_common": tick_common,
        "tick_intersection_ns": (tick_lower, tick_upper),
        "run_consistency": run_consistency,
        "run_relative_median_spread": run_relative_median_spread,
        "concurrency_common": concurrency_common,
        "concurrency_relative_median_spread": (
            concurrency_relative_median_spread),
        "exact_service_time_ns_per_request_per_aiv": service_ns,
        "exact_service_time_iqr_ns_per_request_per_aiv": service_iqr_ns,
        "measurement_gate_passed": measurement_gate_passed,
        "provenance_gate_passed": provenance_gate_passed,
        "production_or_evaluation_duration_consumed": False,
        "mechanism_publishable": (
            measurement_gate_passed and provenance_gate_passed),
    }


def _require_exact_keys(mapping, expected, context):
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ValueError(f"{context} has an invalid schema")


def _require_sha256(value, context):
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256")


def _resolve_artifact(artifact_root, relative_path, context):
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{context} path is invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context} path must stay under the artifact root")
    root = artifact_root.resolve()
    candidate = (root / relative).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise ValueError(f"{context} artifact is missing: {relative_path}")
    return candidate


def _parse_timestamp(value, context):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{context} timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} timestamp must include a timezone")
    return parsed


def _parse_npu_usage_artifact(path):
    text = path.read_text()
    patterns = {
        "device_id": r"^\s*NPU ID\s*:\s*(\d+)\s*$",
        "aicore_utilization_percent": (
            r"^\s*Aicore Usage Rate\(%\)\s*:\s*(\d+)\s*$"),
        "aivector_utilization_percent": (
            r"^\s*Aivector Usage Rate\(%\)\s*:\s*(\d+)\s*$"),
    }
    parsed = {}
    for field, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(
                f"telemetry artifact has invalid {field} evidence")
        parsed[field] = int(matches[0])
    return parsed


def _source_snapshot_sha256(artifact_root, source_files):
    if (not isinstance(source_files, dict)
            or set(source_files) != set(NORM_REQUEST_SOURCE_ARTIFACTS)):
        raise ValueError("NORM request source_files set is not frozen")
    digest = hashlib.sha256()
    for relative_path in NORM_REQUEST_SOURCE_ARTIFACTS:
        expected_sha256 = source_files[relative_path]
        _require_sha256(
            expected_sha256, f"source_files[{relative_path!r}]")
        source_path = _resolve_artifact(
            artifact_root, relative_path, f"source file {relative_path}")
        observed_sha256 = _sha256(source_path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"source file identity changed: {relative_path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(observed_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_norm_request_provenance(path, raw_paths, run_ids):
    with path.open() as stream:
        raw = json.load(stream)
    artifact_root = path.parent
    _require_exact_keys(raw, (
        "schema_version",
        "campaign",
        "hardware_id",
        "device_model",
        "device_id",
        "baseline_commit",
        "source_snapshot_sha256",
        "source_files",
        "binary_path",
        "binary_sha256",
        "compiler_path",
        "compiler_sha256",
        "cann_version",
        "shared_idle_context",
        "external_resident_processes",
        "pre_telemetry",
        "post_telemetry",
        "runs",
        "remaining_benchmark_processes",
        "production_or_evaluation_duration_consumed",
    ), "NORM request provenance")
    if raw["schema_version"] != NORM_REQUEST_PROVENANCE_SCHEMA_VERSION:
        raise ValueError("unsupported NORM request provenance version")
    if raw["campaign"] != NORM_REQUEST_SERVICE_CAMPAIGN:
        raise ValueError("NORM request provenance campaign is invalid")
    if raw["hardware_id"] != "ascend_910b":
        raise ValueError("NORM request provenance hardware_id is invalid")
    if raw["device_model"] != "Ascend910B2":
        raise ValueError("NORM request provenance device model is invalid")
    if not isinstance(raw["device_id"], int) or raw["device_id"] < 0:
        raise ValueError("NORM request provenance device_id is invalid")
    if (not isinstance(raw["baseline_commit"], str)
            or COMMIT_PATTERN.fullmatch(raw["baseline_commit"]) is None):
        raise ValueError("NORM request baseline_commit is invalid")
    _require_sha256(
        raw["source_snapshot_sha256"], "source_snapshot_sha256")
    observed_source_snapshot_sha256 = _source_snapshot_sha256(
        artifact_root, raw["source_files"])
    if observed_source_snapshot_sha256 != raw["source_snapshot_sha256"]:
        raise ValueError("NORM request source snapshot identity changed")
    binary_path = _resolve_artifact(
        artifact_root, raw["binary_path"], "benchmark binary")
    _require_sha256(raw["binary_sha256"], "binary_sha256")
    if _sha256(binary_path) != raw["binary_sha256"]:
        raise ValueError("NORM request binary identity changed")
    compiler_path = _resolve_artifact(
        artifact_root, raw["compiler_path"], "compiler binary")
    _require_sha256(raw["compiler_sha256"], "compiler_sha256")
    if _sha256(compiler_path) != raw["compiler_sha256"]:
        raise ValueError("NORM request compiler identity changed")
    if raw["cann_version"] != "8.5.1":
        raise ValueError("NORM request campaign requires CANN 8.5.1")
    if not isinstance(raw["shared_idle_context"], bool):
        raise ValueError("shared_idle_context must be bool")
    if not isinstance(raw["external_resident_processes"], list):
        raise ValueError("external_resident_processes must be a list")
    for process in raw["external_resident_processes"]:
        _require_exact_keys(
            process, ("pid", "used_memory_mib", "command"),
            "external resident process")
        if (not isinstance(process["pid"], int) or process["pid"] <= 0
                or not isinstance(process["used_memory_mib"], int)
                or process["used_memory_mib"] < 0
                or not isinstance(process["command"], str)
                or not process["command"]):
            raise ValueError("external resident process is invalid")

    telemetry_keys = (
        "timestamp",
        "device_id",
        "aicore_utilization_percent",
        "aivector_utilization_percent",
        "path",
        "raw_sha256",
    )
    telemetry_timestamps = {}
    for label in ("pre_telemetry", "post_telemetry"):
        telemetry = raw[label]
        _require_exact_keys(telemetry, telemetry_keys, label)
        telemetry_timestamps[label] = _parse_timestamp(
            telemetry["timestamp"], label)
        telemetry_path = _resolve_artifact(
            artifact_root, telemetry["path"], label)
        _require_sha256(telemetry["raw_sha256"], f"{label}.raw_sha256")
        if _sha256(telemetry_path) != telemetry["raw_sha256"]:
            raise ValueError(f"{label} artifact identity changed")
        observed_usage = _parse_npu_usage_artifact(telemetry_path)
        if (telemetry["device_id"] != raw["device_id"]
                or observed_usage["device_id"] != raw["device_id"]):
            raise ValueError(f"{label} device_id changed")
        for field in (
                "aicore_utilization_percent",
                "aivector_utilization_percent"):
            if telemetry[field] != observed_usage[field]:
                raise ValueError(f"{label} utilization evidence changed")
            if observed_usage[field] != 0:
                raise ValueError(f"{label} shows non-idle compute")

    raw_identity = {}
    for raw_path in raw_paths:
        with raw_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
                raise ValueError(f"{raw_path} has an invalid raw schema")
            file_run_ids = {row["run_id"] for row in reader}
        if len(file_run_ids) != 1:
            raise ValueError(
                f"{raw_path} must contain exactly one independent run")
        run_id, = file_run_ids
        if run_id in raw_identity:
            raise ValueError(f"duplicate raw file for run {run_id!r}")
        raw_identity[run_id] = (raw_path.resolve(), _sha256(raw_path))
    if set(raw_identity) != set(run_ids):
        raise ValueError("raw files and decoded run IDs do not match")

    run_keys = (
        "run_id",
        "argv",
        "runner_pid",
        "start_timestamp",
        "end_timestamp",
        "host_boot_id",
        "exit_code",
        "raw_csv_path",
        "raw_csv_sha256",
        "stderr_path",
        "stderr_sha256",
    )
    if not isinstance(raw["runs"], list) or len(raw["runs"]) != len(run_ids):
        raise ValueError("NORM request provenance runs are incomplete")
    provenance_run_ids = set()
    runner_pids = set()
    stderr_hashes = set()
    host_boot_ids = set()
    intervals = []
    for run in raw["runs"]:
        _require_exact_keys(run, run_keys, "NORM request run provenance")
        if run["run_id"] in provenance_run_ids:
            raise ValueError("NORM request provenance duplicates a run ID")
        provenance_run_ids.add(run["run_id"])
        argv = run["argv"]
        if (not isinstance(argv, list) or len(argv) != 12
                or any(not isinstance(value, str) for value in argv)):
            raise ValueError("NORM request run argv is invalid")
        if (not Path(argv[0]).is_absolute()
                or Path(argv[0]).name != "python3"
                or not Path(argv[1]).is_absolute()
                or Path(argv[1]).name != "vector_opcode_runner.py"
                or argv[2] != "--binary"
                or not Path(argv[3]).is_absolute()
                or argv[4:] != [
                    "--device-id", str(raw["device_id"]),
                    "--run-id", run["run_id"],
                    "--campaign", NORM_REQUEST_SERVICE_CAMPAIGN,
                    "--trials", str(NORM_REQUEST_TRIALS_PER_RUN),
                ]):
            raise ValueError("NORM request run argv changed")
        if not isinstance(run["runner_pid"], int) or run["runner_pid"] <= 0:
            raise ValueError("NORM request runner_pid is invalid")
        if run["runner_pid"] in runner_pids:
            raise ValueError("NORM request runner_pid is not independent")
        runner_pids.add(run["runner_pid"])
        start_timestamp = _parse_timestamp(
            run["start_timestamp"], "run start")
        end_timestamp = _parse_timestamp(run["end_timestamp"], "run end")
        if end_timestamp <= start_timestamp:
            raise ValueError("NORM request run interval is invalid")
        intervals.append((start_timestamp, end_timestamp))
        if (not isinstance(run["host_boot_id"], str)
                or not run["host_boot_id"]):
            raise ValueError("NORM request host_boot_id is invalid")
        host_boot_ids.add(run["host_boot_id"])
        if run["exit_code"] != 0:
            raise ValueError("NORM request run did not exit successfully")
        raw_csv_path = _resolve_artifact(
            artifact_root, run["raw_csv_path"], "run raw CSV")
        _require_sha256(run["raw_csv_sha256"], "raw_csv_sha256")
        stderr_path = _resolve_artifact(
            artifact_root, run["stderr_path"], "run stderr")
        _require_sha256(run["stderr_sha256"], "stderr_sha256")
        if (run["run_id"] not in raw_identity
                or raw_identity[run["run_id"]][0] != raw_csv_path
                or raw_identity[run["run_id"]][1]
                != run["raw_csv_sha256"]
                or _sha256(raw_csv_path) != run["raw_csv_sha256"]):
            raise ValueError("NORM request raw CSV identity changed")
        if _sha256(stderr_path) != run["stderr_sha256"]:
            raise ValueError("NORM request stderr identity changed")
        if run["stderr_sha256"] in stderr_hashes:
            raise ValueError("NORM request stderr is not independent")
        stderr_hashes.add(run["stderr_sha256"])
        stderr_text = stderr_path.read_text()
        required_stderr_fragments = (
            f"runner_pid={run['runner_pid']}",
            f"device_id={raw['device_id']}",
            f"run_id={run['run_id']}",
            f"campaign={NORM_REQUEST_SERVICE_CAMPAIGN}",
            f"binary_sha256={raw['binary_sha256']}",
        )
        if any(
            fragment not in stderr_text
            for fragment in required_stderr_fragments
        ):
            raise ValueError("NORM request stderr identity changed")
    if provenance_run_ids != set(run_ids):
        raise ValueError("provenance and raw run IDs do not match")
    if len(host_boot_ids) != 1:
        raise ValueError("NORM request runs changed host boot identity")
    ordered_intervals = sorted(intervals)
    if any(
        current[1] > following[0]
        for current, following in zip(
            ordered_intervals, ordered_intervals[1:])
    ):
        raise ValueError("NORM request independent runs overlap")
    if (telemetry_timestamps["pre_telemetry"]
            > ordered_intervals[0][0]
            or telemetry_timestamps["post_telemetry"]
            < ordered_intervals[-1][1]):
        raise ValueError("telemetry does not enclose all NORM request runs")
    if raw["remaining_benchmark_processes"] != []:
        raise ValueError("NORM request benchmark left residual processes")
    if raw["production_or_evaluation_duration_consumed"] is not False:
        raise ValueError(
            "NORM request provenance consumed forbidden duration evidence")
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign", choices=(
            DEFAULT_CAMPAIGN, NORM_REPORT_CAMPAIGN,
            SWIGLU_SERIALIZED_CAMPAIGN, SERIALIZED_VECTOR_H8192_CAMPAIGN,
            AIV_WORKER_PROLOGUE_CAMPAIGN,
            SINGLE_TBUF_INIT_CAMPAIGN,
            SCALAR_INTEGER_DEPENDENCY_CAMPAIGN,
            SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN,
            SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN,
            SWIGLU_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_BUFFER_READY_CAMPAIGN,
            NORM_REQUEST_SERVICE_CAMPAIGN,
            RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN,
            RMSNORM_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN,
            VECTOR_TO_MTE2_FENCE_CAMPAIGN,
            FP32_PREDICATE_SELECT_CAMPAIGN,
            INT32_INT64_CAST_CAMPAIGN,
            LOCAL_SCALAR_FILL_SV_CAMPAIGN, LOCAL_UB_COPY_CAMPAIGN),
        default=DEFAULT_CAMPAIGN)
    parser.add_argument("raw_csv", nargs="+", type=Path)
    parser.add_argument(
        "--validation-raw-csv", nargs="+", type=Path)
    parser.add_argument("--base-raw-csv", nargs="+", type=Path)
    parser.add_argument("--floor-raw-csv", nargs="+", type=Path)
    parser.add_argument("--floor-validation-raw-csv", nargs="+", type=Path)
    parser.add_argument("--provenance-json", type=Path)
    args = parser.parse_args()
    required_paths = tuple(args.raw_csv)
    if args.campaign == NORM_REPORT_CAMPAIGN:
        if (args.validation_raw_csv is None
                or args.floor_raw_csv is None
                or args.floor_validation_raw_csv is None):
            parser.error(
                "sqrt_div_norm requires --validation-raw-csv, "
                "--floor-raw-csv, and --floor-validation-raw-csv")
        required_paths += (
            tuple(args.validation_raw_csv)
            +
            tuple(args.floor_raw_csv)
            + tuple(args.floor_validation_raw_csv))
        if args.provenance_json is not None:
            parser.error(
                "provenance JSON is valid only for the NORM request "
                "service campaign")
        if args.base_raw_csv is not None:
            parser.error(
                "base raw inputs are valid only for the H8192 serialized "
                "Vector campaign")
    elif args.campaign == DEFAULT_CAMPAIGN:
        if args.validation_raw_csv is None:
            parser.error(
                "exp_div_large requires --validation-raw-csv")
        required_paths += tuple(args.validation_raw_csv)
        if (args.floor_raw_csv is not None
                or args.floor_validation_raw_csv is not None):
            parser.error(
                "floor raw inputs are valid only for sqrt_div_norm")
        if args.provenance_json is not None:
            parser.error(
                "provenance JSON is valid only for the NORM request "
                "service campaign")
        if args.base_raw_csv is not None:
            parser.error(
                "base raw inputs are valid only for the H8192 serialized "
                "Vector campaign")
    elif args.campaign == SERIALIZED_VECTOR_H8192_CAMPAIGN:
        if args.base_raw_csv is None:
            parser.error(
                "serialized_vector_h8192_validation requires "
                "--base-raw-csv")
        if (args.validation_raw_csv is not None
                or args.floor_raw_csv is not None
                or args.floor_validation_raw_csv is not None
                or args.provenance_json is not None):
            parser.error(
                "serialized_vector_h8192_validation accepts only its "
                "primary and base raw inputs")
        required_paths += tuple(args.base_raw_csv)
    elif args.campaign in (
            SWIGLU_SERIALIZED_CAMPAIGN, AIV_WORKER_PROLOGUE_CAMPAIGN,
            SINGLE_TBUF_INIT_CAMPAIGN,
            SCALAR_INTEGER_DEPENDENCY_CAMPAIGN,
            SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN,
            SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN,
            SWIGLU_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_BUFFER_READY_CAMPAIGN,
            RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN,
            RMSNORM_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN,
            ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN,
            ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN,
            VECTOR_TO_MTE2_FENCE_CAMPAIGN,
            LOCAL_SCALAR_FILL_SV_CAMPAIGN,
            LOCAL_UB_COPY_CAMPAIGN,
            FP32_PREDICATE_SELECT_CAMPAIGN,
            INT32_INT64_CAST_CAMPAIGN):
        if (args.validation_raw_csv is not None
                or args.floor_raw_csv is not None
                or args.floor_validation_raw_csv is not None):
            parser.error(
                f"{args.campaign} accepts only its primary raw inputs")
        if args.provenance_json is not None:
            parser.error(
                "provenance JSON is valid only for the NORM request "
                "service campaign")
        if args.base_raw_csv is not None:
            parser.error(
                "base raw inputs are valid only for the H8192 serialized "
                "Vector campaign")
    elif args.campaign == NORM_REQUEST_SERVICE_CAMPAIGN:
        if (args.validation_raw_csv is not None
                or args.floor_raw_csv is not None
                or args.floor_validation_raw_csv is not None):
            parser.error(
                "sqrt_div_norm_request_service_r2 accepts only its three "
                "primary raw CSV files")
        if args.provenance_json is None:
            parser.error(
                "sqrt_div_norm_request_service_r2 requires --provenance-json")
        required_paths += (args.provenance_json,)
        if args.base_raw_csv is not None:
            parser.error(
                "base raw inputs are valid only for the H8192 serialized "
                "Vector campaign")
    else:
        raise AssertionError(f"unhandled report campaign: {args.campaign}")
    for path in required_paths:
        if not path.is_file():
            parser.error(f"raw CSV does not exist: {path}")
    if args.campaign in (FP32_PREDICATE_SELECT_CAMPAIGN,
                         INT32_INT64_CAST_CAMPAIGN):
        rows = _load_balanced_campaign(tuple(args.raw_csv), args.campaign)
        cells = sorted({row["cell"] for row in rows})
        observations = []
        for cell in cells:
            values = [row["per_iteration_ns"] for row in rows
                      if row["cell"] == cell]
            opcode, elements, block_dim = cell
            observation = {
                "opcode": opcode,
                "elements_per_vector_invocation": elements,
                "block_dim": block_dim,
                "samples": len(values),
                "median_ns_per_vector_invocation_per_aiv":
                    statistics.median(values),
                "minimum_ns_per_vector_invocation_per_aiv": min(values),
                "maximum_ns_per_vector_invocation_per_aiv": max(values),
                "input_element_bytes": 4,
                "output_element_bytes": (
                    8 if args.campaign == INT32_INT64_CAST_CAMPAIGN else
                    (4 if opcode == "select_fp32" else None)),
                "output_packed_predicate_bits_per_element": (
                    1 if opcode.startswith("compare_") else None),
            }
            if args.campaign == INT32_INT64_CAST_CAMPAIGN:
                # Level-2 count Cast uses mask_count(N), followed by
                # vconv_s322s64 in count mode. N/64 is not an observed ISA
                # repeat count. This measurement prices N-element API
                # invocations, not an invented number of machine repeats.
                observation.update({
                    "source_api_count_mode": "mask_count",
                    "logical_input_bytes_per_invocation": 4 * elements,
                    "logical_output_bytes_per_invocation": 8 * elements,
                    "fixed_isa_repeat_count_claimed": False,
                })
            else:
                observation.update({
                    "active_elements_per_source_repeat": 64,
                    "repeats_per_invocation": elements // 64,
                })
                if opcode == "compare_fp32_ordered_ne":
                    observation.update({
                        "predicate_semantics": "ordered_inequality",
                        "quiet_nan_payload_tested": "0x7fc00001",
                        "nan_detection_service_verified": False,
                        "signaling_nan_semantics_verified": False,
                    })
            observations.append(observation)
        print(json.dumps({
            "schema_version": 1,
            "scope": "independent_typed_vector_invocations_only",
            "campaign": args.campaign,
            "independent_runs": len({row["run_id"] for row in rows}),
            "system_counter_tick_ns": SYSTEM_COUNTER_TICK_NS,
            "observations": observations,
            "production_duration_consumed": False,
            "single_isa_instruction_rate_verified": False,
            "int64_index_order_service_verified": False,
            "production_argmax_bound": False,
            "production_integer_cast_bound": False,
            "mechanism_publishable": False,
        }, indent=2, sort_keys=True))
        return
    if args.campaign == DEFAULT_CAMPAIGN:
        audit = _audit(
            _load_rows(tuple(args.raw_csv)),
            _load_validation_rows(tuple(args.validation_raw_csv)))
    elif args.campaign == NORM_REPORT_CAMPAIGN:
        audit = _audit_norm_typed_service(
            _load_balanced_campaign(
                tuple(args.raw_csv), NORM_STEADY_CALIBRATION_CAMPAIGN),
            _load_balanced_campaign(
                tuple(args.validation_raw_csv),
                NORM_STEADY_VALIDATION_CAMPAIGN),
            _load_balanced_campaign(
                tuple(args.floor_raw_csv), NORM_FLOOR_CALIBRATION_CAMPAIGN),
            _load_balanced_campaign(
                tuple(args.floor_validation_raw_csv),
                NORM_FLOOR_VALIDATION_CAMPAIGN),
        )
        print(f"independent_run_counts={audit['independent_run_counts']}")
        print(f"tick_common={audit['tick_common']}")
        print(f"tick_intersection_ns={audit['tick_intersection_ns']}")
        print(f"concurrency_common={audit['concurrency_common']}")
        print(
            "affine_intrinsic_parameters="
            f"{audit['affine_intrinsic_parameters']}")
        print(f"pair_affine_common={audit['pair_affine_common']}")
        print(
            "frozen_div_steady_consistency="
            f"{audit['frozen_div_steady_consistency']}")
        print(
            "steady_validation_intersections="
            f"{audit['steady_validation_intersections']}")
        print(
            "minimum_service_time_s_per_intrinsic="
            f"{audit['minimum_service_time_s_per_intrinsic']}")
        print(f"floor_pair_common={audit['floor_pair_common']}")
        print(f"small_validation={audit['small_validation']}")
        print(f"mechanism_publishable={audit['mechanism_publishable']}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == RMSNORM_LOCAL_SCALAR_CAMPAIGN:
        audit = _audit_rmsnorm_local_scalar_dependency(
            _load_balanced_campaign(
                tuple(args.raw_csv), RMSNORM_LOCAL_SCALAR_CAMPAIGN),
            provenance_gate_passed=False,
        )
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN:
        audit = _audit_add_rmsnorm_local_scalar_handoff(
            _load_balanced_campaign(
                tuple(args.raw_csv), ADD_RMSNORM_LOCAL_SCALAR_CAMPAIGN),
            provenance_gate_passed=False,
        )
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN:
        audit = _audit_add_rmsnorm_split_d_local_scalar_handoff(
            _load_balanced_campaign(
                tuple(args.raw_csv),
                ADD_RMSNORM_SPLIT_D_LOCAL_SCALAR_CAMPAIGN),
            provenance_gate_passed=False,
        )
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == VECTOR_TO_MTE2_FENCE_CAMPAIGN:
        audit = _audit_vector_to_mte2_fence(
            _load_balanced_campaign(
                tuple(args.raw_csv), VECTOR_TO_MTE2_FENCE_CAMPAIGN),
            provenance_gate_passed=False,
        )
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN:
        audit = _audit_rmsnorm_half_interval_reduce(
            _load_balanced_campaign(
                tuple(args.raw_csv),
                RMSNORM_HALF_INTERVAL_REDUCE_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN:
        audit = _audit_add_rmsnorm_reduce_sum_fp32(
            _load_balanced_campaign(
                tuple(args.raw_csv),
                ADD_RMSNORM_REDUCE_SUM_FP32_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN:
        audit = _audit_add_rmsnorm_split_d_block_reduce_sum_fp32(
            _load_balanced_campaign(
                tuple(args.raw_csv),
                ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_FP32_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN:
        audit = _audit_add_rmsnorm_split_d_block_reduce_sum_composed(
            _load_balanced_campaign(
                tuple(args.raw_csv),
                ADD_RMSNORM_SPLIT_D_BLOCK_REDUCE_SUM_COMPOSED_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["measurement_gate_passed"]:
            raise SystemExit(1)
        return
    elif args.campaign == LOCAL_SCALAR_FILL_SV_CAMPAIGN:
        audit = _audit_local_scalar_fill_sv(
            _load_balanced_campaign(
                tuple(args.raw_csv), LOCAL_SCALAR_FILL_SV_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == LOCAL_UB_COPY_CAMPAIGN:
        audit = _audit_local_ub_copy(
            _load_balanced_campaign(
                tuple(args.raw_csv), LOCAL_UB_COPY_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == SWIGLU_SERIALIZED_CAMPAIGN:
        audit = _audit_swiglu_serialized_services(
            _load_balanced_campaign(
                tuple(args.raw_csv), SWIGLU_SERIALIZED_CAMPAIGN))
        print(f"independent_runs={audit['independent_runs']}")
        print(f"tick_common={audit['tick_common']}")
        print(f"tick_intersection_ns={audit['tick_intersection_ns']}")
        print(f"concurrency_common={audit['concurrency_common']}")
        print(
            "affine_serialized_service_parameters="
            f"{audit['affine_serialized_service_parameters']}")
        print(
            "validation="
            f"{audit['validation']}")
        print(f"service_publishable={audit['service_publishable']}")
        print(f"mechanism_publishable={audit['mechanism_publishable']}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == SERIALIZED_VECTOR_H8192_CAMPAIGN:
        audit = _audit_extended_serialized_vector_services(
            _load_balanced_campaign(
                tuple(args.base_raw_csv), SWIGLU_SERIALIZED_CAMPAIGN),
            _load_balanced_campaign(
                tuple(args.raw_csv), SERIALIZED_VECTOR_H8192_CAMPAIGN),
        )
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == AIV_WORKER_PROLOGUE_CAMPAIGN:
        audit = _audit_aiv_worker_prologue(
            _load_balanced_campaign(
                tuple(args.raw_csv), AIV_WORKER_PROLOGUE_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == SINGLE_TBUF_INIT_CAMPAIGN:
        audit = _audit_single_tbuf_init(
            _load_balanced_campaign(
                tuple(args.raw_csv), SINGLE_TBUF_INIT_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == SCALAR_INTEGER_DEPENDENCY_CAMPAIGN:
        audit = _audit_scalar_integer_dependency(
            _load_balanced_campaign(
                tuple(args.raw_csv), SCALAR_INTEGER_DEPENDENCY_CAMPAIGN))
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN:
        audit = _audit_scalar_integer_dependency(
            _load_balanced_campaign(
                tuple(args.raw_csv),
                SCALAR_U64_DIVREM_DEPENDENCY_CAMPAIGN),
            SCALAR_U64_DIVREM_DEPENDENCY_OPCODES)
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign in (
            SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN,
            SWIGLU_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_QUEUE_CONTROL_CAMPAIGN,
            RMSNORM_BUFFER_READY_CAMPAIGN):
        opcode_definitions = (
            SWIGLU_BUFFER_LIFECYCLE_OPCODES
            if args.campaign == SWIGLU_BUFFER_LIFECYCLE_CAMPAIGN
            else (
                SWIGLU_QUEUE_CONTROL_OPCODES
                if args.campaign == SWIGLU_QUEUE_CONTROL_CAMPAIGN
                else (
                    RMSNORM_QUEUE_CONTROL_OPCODES
                    if args.campaign == RMSNORM_QUEUE_CONTROL_CAMPAIGN
                    else RMSNORM_BUFFER_READY_OPCODES)))
        audit = _audit_swiglu_control_service(
            _load_balanced_campaign(tuple(args.raw_csv), args.campaign),
            opcode_definitions)
        for name, value in audit.items():
            print(f"{name}={value}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    elif args.campaign == NORM_REQUEST_SERVICE_CAMPAIGN:
        rows = _load_norm_request_rows(tuple(args.raw_csv))
        run_ids = sorted({row["run_id"] for row in rows})
        _load_norm_request_provenance(
            args.provenance_json, tuple(args.raw_csv), run_ids)
        audit = _audit_norm_request_services(rows, True)
        print(f"independent_runs={audit['independent_runs']}")
        print(f"samples_per_cell={audit['samples_per_cell']}")
        print(f"tick_common={audit['tick_common']}")
        print(f"tick_intersection_ns={audit['tick_intersection_ns']}")
        print(
            "run_consistency_passed="
            f"{sum(audit['run_consistency'].values())}/"
            f"{len(audit['run_consistency'])}")
        print(
            "maximum_run_relative_median_spread="
            f"{max(audit['run_relative_median_spread'].values())}")
        print(
            "concurrency_common_passed="
            f"{sum(audit['concurrency_common'].values())}/"
            f"{len(audit['concurrency_common'])}")
        print(
            "maximum_concurrency_relative_median_spread="
            f"{max(audit['concurrency_relative_median_spread'].values())}")
        print(
            "exact_service_time_ns_per_request_per_aiv="
            f"{audit['exact_service_time_ns_per_request_per_aiv']}")
        print(
            "exact_service_time_iqr_ns_per_request_per_aiv="
            f"{audit['exact_service_time_iqr_ns_per_request_per_aiv']}")
        print(
            "production_or_evaluation_duration_consumed="
            f"{audit['production_or_evaluation_duration_consumed']}")
        print(
            f"measurement_gate_passed={audit['measurement_gate_passed']}")
        print(f"provenance_gate_passed={audit['provenance_gate_passed']}")
        print(f"mechanism_publishable={audit['mechanism_publishable']}")
        if not audit["mechanism_publishable"]:
            raise SystemExit(1)
        return
    else:
        raise AssertionError(f"unhandled report campaign: {args.campaign}")
    print(f"independent_runs={audit['independent_runs']}")
    print(f"samples_per_cell={audit['samples_per_cell']}")
    print(f"tick_common={audit['tick_common']}")
    print(f"tick_intersection_ns={audit['tick_intersection']}")
    print(f"concurrency_common={audit['concurrency_common']}")
    print(f"plateau={audit['plateau']}")
    print(f"pair_service_common={audit['pair_service_common']}")
    print(
        "capacity_candidates_elements_per_ns_per_aiv="
        f"{audit['capacity_candidates_elements_per_ns_per_aiv']}")
    print(
        "capacity_plateau_publishable="
        f"{audit['capacity_plateau_publishable']}")
    print(
        "affine_intrinsic_parameters="
        f"{audit['affine_intrinsic_parameters']}")
    print(f"affine_additive_common={audit['affine_additive_common']}")
    print(
        "affine_calibration_publishable="
        f"{audit['affine_calibration_publishable']}")
    print(f"affine_validation={audit['affine_validation']}")
    print(f"mechanism_publishable={audit['mechanism_publishable']}")
    if not audit["mechanism_publishable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
