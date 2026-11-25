"""
Resource model evaluation — all supported kernel operators.

Usage:
  python scripts/evaluation/kernel/run_kernel_evaluation.py                    # oracle + forward
  python scripts/evaluation/kernel/run_kernel_evaluation.py --mode oracle      # oracle only
  python scripts/evaluation/kernel/run_kernel_evaluation.py --mode forward     # forward only
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from evaluation.kernel.dataset import (
    ATTENTION_OP_TYPES,
    MOE_OP_TYPES,
    get_hardware_config_path,
    kernel_dataset_l2_execution_context,
    kernel_evaluation_dataset_location,
    load_kernel_candidate_root,
    load_kernel_evaluation_dataset,
    parse_kernel_evaluation_timing,
    require_canonical_hardware_id,
)
from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import KernelKind
from src.kernel.fused_moe_physical_specs import (
    build_fused_moe_physical_specs,
)
from evaluation.kernel.observation_parser import (
    is_non_tensor,
)
from evaluation.kernel.oracle_observation import normalize_oracle_observation
from src.kernel.kernel_base import create_predictor, predict as predict_production
from src.kernel.launch_config.gemm_launch_selection import (
    preload_gemm_launch_selection,
)
from src.kernel.dataset_storage import (
    KERNEL_DATASET_SUFFIX,
    load_kernel_dataset_document,
)
from evaluation.kernel.observation_normalizer import (
    build_dsv4_case_specs,
    dsv4_case_report_axes,
    normalize_kernel_spec,
    observed_flash_attention_splits,
    observed_gemm_split_k,
    observed_gemm_split_k_partial_bytes,
    observed_gemm_split_k_partial_dram_bytes,
    validate_kernel_observation_sequence,
)
from evaluation.kernel.fa_serving_coverage_report import (
    build_fa_serving_specs,
    validate_fa_serving_capture_record,
)
from evaluation.json_shape import file_sha256
from evaluation.json_shape import require_positive_metric as _positive_int


_MATRIX_LIKE_OP_TYPES = frozenset({"GEMM"}) | MOE_OP_TYPES

_ATTENTION_LAYOUTS = (
    "contiguous", "paged", "paged_varlen", "chunked_prefill", "mla",
)
_FORK_SHARED_GEMM_SELECTION_HARDWARE_IDS = frozenset({"ascend_910b"})

# The hardwares the kernel evaluator can run, taken from the canonical
# registry so adding a hardware is one registration rather than a second list.
_EVALUATION_HARDWARE_IDS = CANONICAL_HARDWARE_IDS

_OP_TYPE_PREDICTOR_KINDS = {
    "GEMM": (KernelKind.GEMM, KernelKind.VECTOR),
    "FUSEDMOE": (KernelKind.FUSED_MOE, KernelKind.VECTOR),
    "UNFUSEDMOE": (KernelKind.UNFUSED_MOE,),
    "FLASHATTN": (KernelKind.FLASH_ATTN,),
    "EAGERATTN": (KernelKind.EAGER_ATTN,),
    "NORM": (KernelKind.VECTOR,),
    "ROTARY": (KernelKind.VECTOR,),
    "ACTIVATION": (KernelKind.VECTOR,),
    "KV_CACHE": (KernelKind.KV_CACHE,),
    "ELEMENTWISE": (KernelKind.VECTOR,),
    "V4ATTN": (
        KernelKind.SPARSE_ATTN, KernelKind.KV_CACHE, KernelKind.VECTOR),
    "WQ_FUSEDMOE": (KernelKind.FUSED_MOE,),
}

# A case's main launch is the one carrying the operator's own arithmetic.  The
# primary kind of _OP_TYPE_PREDICTOR_KINDS already names it, and the VECTOR
# entries beside it are the surrounding glue -- the MoE routing/align/sort/sum
# launches and the split-K reduce.  Attention's split-KV combine is the same
# kind of thing: it reduces partial outputs the main launch produced, exactly
# as the split-K reduce does for GEMM, and only carries the FLASH_ATTN kind
# because it ships inside the attention op.  A baseline that cannot price the
# main launch is outside the case's domain; one that cannot price the glue is
# merely incomplete on it.
_AUXILIARY_LAUNCH_VARIANTS = {
    KernelKind.FLASH_ATTN: frozenset({"combine"}),
}


def is_main_launch(kernel_spec, op_type: str) -> bool:
    """Report whether this launch carries the case operator's own arithmetic."""

    kinds = _OP_TYPE_PREDICTOR_KINDS[op_type]
    if kernel_spec.kind is not kinds[0]:
        return False
    if kernel_spec.kind not in _AUXILIARY_LAUNCH_VARIANTS:
        return True
    return (kernel_spec.variant.lower()
            not in _AUXILIARY_LAUNCH_VARIANTS[kernel_spec.kind])


PIPEWEAVE_OUT_OF_DOMAIN = "OUT_OF_DOMAIN"

_FORWARD_TIMING_SCHEMA_VERSION = 1
_FORWARD_TIMING_SCOPE = "kernel_forward_evaluation"
_FORWARD_TIMED_REGION = "process_kernel_evaluation_cases"
_FORWARD_TIMING_EXCLUDED_PHASES = (
    "dataset_loading",
    "hardware_spec_loading",
    "case_packing",
    "metric_aggregation",
    "table_rendering",
    "counterfactual_evaluation",
    "plotting",
)
_CASE_TIMING_SCHEMA_VERSION = 1
_CASE_TIMING_SCOPE = "kernel_case_evaluation_timing"
_CASE_TIMED_REGION = "_process_op"
_CASE_TIMING_EXCLUDED_PHASES = (
    "dataset_loading",
    "hardware_spec_loading",
    "case_packing",
    "executor_queue_wait",
    "metric_aggregation",
    "table_rendering",
    "counterfactual_evaluation",
    "plotting",
)
_SEMANTIC_ARTIFACT_SCHEMA_VERSION = 1
_SEMANTIC_ARTIFACT_SCOPE = "kernel_evaluation_semantic_result"


def _non_negative_int_sequence(mapping, name, context):
    value = mapping[name]
    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError(f"{context}.{name} must be a non-empty sequence")
    invalid = [
        entry for entry in value
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0
    ]
    if invalid:
        raise ValueError(
            f"{context}.{name} entries must be non-negative integers, got "
            f"{invalid[:3]}")
    return tuple(value)


def _launch_task_kind_for_report(kernel_spec, hardware):
    """Return the static execution domain used only for launch-wave reports."""

    if not hardware.aic_aiv_separated:
        return "generic"
    if kernel_spec.kind in (
            KernelKind.VECTOR, KernelKind.KV_CACHE):
        return "aiv"
    elif kernel_spec.kind == KernelKind.FLASH_ATTN:
        if kernel_spec.variant.lower() in ("combine", "split_combine"):
            return "aiv"
        return "mixed"
    elif kernel_spec.kind == KernelKind.GEMM:
        aiv_mixed = kernel_spec.launch.parameters["aiv_mixed"]
        if aiv_mixed == 1.0:
            return "mixed"
        elif aiv_mixed == 0.0:
            return "aic"
        else:
            raise ValueError(
                "GEMM launch aiv_mixed must be exactly zero or one")
    elif kernel_spec.kind == KernelKind.FUSED_MOE:
        return "aic"
    else:
        raise ValueError(
            "kernel evaluation cannot derive a launch execution domain for "
            f"{kernel_spec.kind.value}")


def _launch_waves_for_report(kernel_spec, hardware):
    task_kind = _launch_task_kind_for_report(kernel_spec, hardware)
    execution_units = hardware.execution_units_for_task(task_kind)
    return math.ceil(kernel_spec.launch.grid_size / execution_units)


def _ascend_public_fused_moe_specs(op_attributes, hardware):
    """Lower the public vLLM-Ascend FusedMoE call from logical semantics.

    Ascend canonical FusedMoE timing is at logical-op granularity.  The
    retained historical ``kernels[]`` sequence is useful for diagnostics, but
    it is not the currently measured public call.  Forward therefore uses the
    shared production source/ABI lowerer rather than summing that legacy
    observation sequence.
    """

    if hardware.hardware_id != "ascend_910b":
        raise ValueError("public Ascend FusedMoE specs require ascend_910b")
    tokens = _positive_int(op_attributes, "TOK", "FUSEDMOE Attr")
    experts = _positive_int(op_attributes, "E", "FUSEDMOE Attr")
    topk = _positive_int(op_attributes, "TOPK", "FUSEDMOE Attr")
    hidden = _positive_int(op_attributes, "H", "FUSEDMOE Attr")
    intermediate = _positive_int(op_attributes, "INT", "FUSEDMOE Attr")
    counts = _non_negative_int_sequence(
        op_attributes, "tok_per_expert", "FUSEDMOE Attr")
    if len(counts) != experts:
        raise ValueError(
            "FUSEDMOE Attr.tok_per_expert must contain one count per expert")
    if sum(counts) != tokens * topk:
        raise ValueError(
            "FUSEDMOE Attr.tok_per_expert must sum to TOK*TOPK")
    return build_fused_moe_physical_specs(
        hardware_id="ascend_910b",
        shape={
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
            "hidden": hidden,
            "intermediate": intermediate,
        },
        dtype=hardware.kernel_dtype,
        prediction_attributes={
            "tokens_per_expert": counts,
            "local_tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(experts)),
            "expert_weight_tile_rows": None,
        },
        selection_attributes={
            "launch_policy": "analytical_launch",
            "expert_parallel_size": 1,
        },
        hardware=hardware,
    )


def _extract_bmkn(op_info):
    """Extract (B, M, K, N) from op_info, normalized per op type."""
    op_type = op_info["Type"]
    attrs = op_info["Attr"]
    if op_type == "GEMM":
        return tuple(
            _positive_int(attrs, name, "GEMM Attr")
            for name in ("B", "M", "K", "N")
        )
    elif op_type == "FLASHATTN":
        return tuple(
            _positive_int(attrs, name, "FLASHATTN Attr")
            for name in ("B", "S", "D", "HQ")
        )
    elif op_type == "EAGERATTN":
        # Eager (unfused) attention on PIM — same BMKN shape contract as FLASHATTN.
        # Ground truth is a decomposed unfused (qkt mm + softmax + pv mm)
        # latency from UniNDP, NOT a fused FA kernel.
        return tuple(
            _positive_int(attrs, name, "EAGERATTN Attr")
            for name in ("B", "S", "D", "HQ")
        )
    elif op_type in MOE_OP_TYPES:
        return (
            _positive_int(attrs, "TOK", f"{op_type} Attr"),
            1,
            _positive_int(attrs, "H", f"{op_type} Attr"),
            _positive_int(attrs, "INT", f"{op_type} Attr"),
        )
    else:
        raise ValueError(f"Unknown op_type '{op_type}' in _extract_bmkn")


def _process_fa_serving_capture_op(
        task_key, entry, hw, run_oracle, run_forward, run_roofline,
        roofline_launch_ns, run_pipeweave, pipeweave_model_hw,
        capture_mechanisms):
    """Evaluate one op-level FA campaign row without fabricating raw kernels."""

    if capture_mechanisms:
        raise NotImplementedError(
            "FA serving capture rows have op-level evidence and cannot provide "
            "legacy per-kernel mechanism captures")
    target = validate_fa_serving_capture_record(entry, hw.hardware_id)
    attr = entry["op_info"]["Attr"]
    specs = build_fa_serving_specs(attr, hw.hardware_id, hw)
    if not specs:
        raise RuntimeError(f"FA serving capture {task_key!r} produced no specs")

    forward_predictions = []
    roofline_predictions = []
    pipeweave_predictions = []
    forward_error = False
    roofline_error = False
    pipeweave_error = False
    pipeweave_main_unsupported = 0
    pipeweave_glue_roofline = 0
    kernel_results = []
    launch_waves = []
    for ordinal, kernel_spec in enumerate(specs):
        if kernel_spec.launch is not None:
            launch_waves.append(_launch_waves_for_report(kernel_spec, hw))
        row = {
            "task_key": task_key,
            "op_type": "FLASHATTN",
            "eqm": None,
            "actual_ns": None,
            "actual_source": entry["op_info"]["evaluation"]["timing"]["source"],
            "actual_granularity": "op",
            "kernel_ordinal": ordinal,
            "kernel_kind": kernel_spec.kind.value,
        }
        if run_oracle:
            row["oracle_unavailable_reason"] = (
                "serving-aligned FA records retain duration-free counter "
                "evidence rather than legacy Oracle profiler observations")
        if run_forward:
            try:
                prediction = predict_production(
                    kernel_spec, hw, method="midlayer")
            except NotImplementedError as error:
                forward_error = True
                row["forward_unavailable_reason"] = str(error)
            else:
                forward_predictions.append(prediction.t_total_ns)
                row["pred_forward_ns"] = prediction.t_total_ns
                row["forward_support_level"] = (
                    prediction.diagnostics.support_level
                    if prediction.diagnostics is not None else None)
                row["forward_limitations"] = (
                    prediction.diagnostics.limitations
                    if prediction.diagnostics is not None else None)
                row["forward_bottleneck"] = prediction.bottleneck_group
        if run_roofline:
            try:
                prediction_ns = predict_production(
                    kernel_spec, hw, method="roofline").t_total_ns
            except NotImplementedError as error:
                roofline_error = True
                row["roofline_unavailable_reason"] = str(error)
            else:
                if roofline_launch_ns > 0:
                    prediction_ns += roofline_launch_ns
                roofline_predictions.append(prediction_ns)
                row["pred_roofline_ns"] = prediction_ns
        if run_pipeweave:
            # Same rule as the case path: Roofline prices glue the baseline has
            # no model for, and a main launch it cannot price takes the case
            # out of its domain entirely.
            try:
                prediction_ns = predict_production(
                    kernel_spec, hw, method="pipeweave",
                    model_hardware=pipeweave_model_hw).t_total_ns
            except NotImplementedError:
                if is_main_launch(kernel_spec, "FLASHATTN"):
                    pipeweave_main_unsupported += 1
                    pipeweave_predictions.append(0.0)
                else:
                    try:
                        glue_ns = predict_production(
                            kernel_spec, hw, method="roofline").t_total_ns
                    except NotImplementedError:
                        pipeweave_main_unsupported += 1
                        pipeweave_predictions.append(0.0)
                    else:
                        pipeweave_glue_roofline += 1
                        pipeweave_predictions.append(glue_ns)
            else:
                pipeweave_predictions.append(prediction_ns)
        kernel_results.append(row)

    if run_forward and not forward_error and len(forward_predictions) != len(specs):
        raise RuntimeError(f"{task_key!r} produced an incomplete Forward result")
    if run_roofline and not roofline_error and len(roofline_predictions) != len(specs):
        raise RuntimeError(f"{task_key!r} produced an incomplete Roofline result")
    if run_pipeweave and not pipeweave_error and len(pipeweave_predictions) != len(specs):
        raise RuntimeError(f"{task_key!r} produced an incomplete PipeWeave result")

    main_launch = specs[0].launch
    if main_launch is None or main_launch.tiling is None:
        raise ValueError("FA serving capture main launch lacks tiling")
    executed_splits = (
        None if hw.hardware_id == "ascend_910b"
        else main_launch.tiling.num_kv_splits)
    if hw.hardware_id != "ascend_910b" and executed_splits is None:
        raise ValueError("FA serving capture GPU launch lacks split count")
    query_length = _positive_int(attr, "Q_LEN", "FLASHATTN Attr")
    kv_length = _positive_int(attr, "KV_LEN", "FLASHATTN Attr")
    batch = _positive_int(attr, "B", "FLASHATTN Attr")
    eqm = batch * kv_length
    meta = {
        "op_type": "FLASHATTN",
        "eqm": eqm,
        "range": (
            "tiny" if eqm <= 16 else "small" if eqm <= 64
            else "med" if eqm <= 512 else "prefill"),
        "split": executed_splits is not None and executed_splits > 1,
        "actual_source": entry["op_info"]["evaluation"]["timing"]["source"],
        "actual_granularity": "op",
        "actual_ns": target,
        "max_launch_waves": max(launch_waves, default=1),
        "query_length": query_length,
        "kv_length": kv_length,
        "executed_attention_splits": executed_splits,
        "attention_layout": attr["kv_layout"],
        "pipeweave_kernels": len(specs) if run_pipeweave else None,
        "pipeweave_main_unsupported_kernels": (
            pipeweave_main_unsupported if run_pipeweave else None),
        "pipeweave_glue_roofline_kernels": (
            pipeweave_glue_roofline if run_pipeweave else None),
    }
    oracle_pair = "ERR" if run_oracle else None
    forward_pair = (
        "ERR" if forward_error
        else ((math.fsum(forward_predictions), target) if run_forward else None))
    roofline_pair = (
        "ERR" if roofline_error
        else ((math.fsum(roofline_predictions), target)
              if run_roofline else None))
    if pipeweave_error:
        pipeweave_pair = "ERR"
    elif run_pipeweave and pipeweave_main_unsupported:
        pipeweave_pair = PIPEWEAVE_OUT_OF_DOMAIN
    elif run_pipeweave:
        pipeweave_pair = (math.fsum(pipeweave_predictions), target)
    else:
        pipeweave_pair = None
    return (
        task_key, meta, oracle_pair, forward_pair, roofline_pair,
        pipeweave_pair, kernel_results)


def _process_dsv4_case_op(task_key, entry, hw, run_oracle, run_forward,
                          run_roofline, roofline_launch_ns, run_pipeweave,
                          pipeweave_model_hw, capture_mechanisms):
    """Evaluate one DeepSeek-V4 H200 campaign case at op granularity.

    The case builder (observation_normalizer.build_dsv4_case_specs) maps the
    candidate's runner arguments and launch evidence to the production
    KernelSpec of the operator the case actually measured; unmodeled glue
    (combine/gather/align/activation/save kernels) stays inside the observed
    total and surfaces as prediction error rather than being silently priced.
    A case whose record does not map cleanly is explicitly unsupported.
    """

    if capture_mechanisms:
        raise NotImplementedError(
            "DeepSeek-V4 campaign cases map to op-level specs and cannot "
            "provide legacy per-kernel mechanism captures")
    oi = entry["op_info"]
    op_type = oi["Type"]
    attr = oi["Attr"]
    kernels_raw = entry["kernels"]
    if not isinstance(kernels_raw, list) or not kernels_raw:
        raise TypeError(
            f"dataset entry {task_key!r} kernels must be a non-empty list")
    validate_kernel_observation_sequence(
        kernels_raw, hw.hardware_id, op_info=oi, hardware=hw)
    timing_observation = parse_kernel_evaluation_timing(
        entry, kernels_raw, hw.hardware_id)
    total_actual = timing_observation.total_duration_ns
    if total_actual <= 0:
        raise ValueError(f"{task_key!r} has non-positive total observed duration")

    axes = None
    specs = None
    unsupported_reason = None
    try:
        axes = dsv4_case_report_axes(op_type, attr)
    except (ValueError, TypeError, NotImplementedError) as error:
        unsupported_reason = f"report axes: {error}"
    if axes is not None:
        try:
            specs = build_dsv4_case_specs(
                op_type, attr, kernels_raw, hw.hardware_id, hw)
        except (ValueError, TypeError, NotImplementedError) as error:
            unsupported_reason = str(error)

    if axes is None:
        axes = {
            "dsv4_stage": "unmapped",
            "dsv4_model": "unmapped",
            "dsv4_layer_type": None,
        }

    forward_pair = None
    roofline_pair = None
    kernel_results = []
    launch_waves = []
    eqm = None
    if specs is None:
        if unsupported_reason is None:
            raise RuntimeError(
                f"{task_key!r} produced no specs and no reason")
        row = {
            "task_key": task_key,
            "op_type": op_type,
            "eqm": None,
            "actual_ns": None,
            "actual_source": timing_observation.source,
            "actual_granularity": timing_observation.granularity,
            "observation_unsupported_reason": unsupported_reason,
        }
        if run_oracle:
            row["oracle_unavailable_reason"] = unsupported_reason
        if run_forward:
            row["forward_unavailable_reason"] = unsupported_reason
        if run_roofline:
            row["roofline_unavailable_reason"] = unsupported_reason
        kernel_results.append(row)
        if run_forward:
            forward_pair = "ERR"
        if run_roofline:
            roofline_pair = "ERR"
        launch_waves.append(1)
    else:
        main_shape = specs[0].shape
        if specs[0].kind is KernelKind.SPARSE_ATTN:
            eqm = main_shape["batch"] * main_shape["kv_len"]
        elif specs[0].kind is KernelKind.KV_CACHE:
            eqm = main_shape["seq_len"]
        elif specs[0].kind is KernelKind.VECTOR:
            eqm = main_shape["element_count"] // (
                main_shape["hc_mult"] * main_shape["hidden"])
        elif specs[0].kind is KernelKind.FUSED_MOE:
            eqm = main_shape["tokens"]
        else:
            raise AssertionError(
                f"unhandled DeepSeek-V4 case spec kind {specs[0].kind.value}")
        forward_predictions = []
        roofline_predictions = []
        forward_error = False
        roofline_error = False
        for ordinal, kernel_spec in enumerate(specs):
            launch_waves.append(_launch_waves_for_report(kernel_spec, hw))
            row = {
                "task_key": task_key,
                "op_type": op_type,
                "eqm": eqm,
                "actual_ns": None,
                "actual_source": timing_observation.source,
                "actual_granularity": "op",
                "kernel_ordinal": ordinal,
                "kernel_kind": kernel_spec.kind.value,
                "kernel_variant": kernel_spec.variant,
            }
            if run_oracle:
                row["oracle_unavailable_reason"] = (
                    "DeepSeek-V4 campaign cases carry op-level spec mappings; "
                    "no per-kernel Oracle observation is defined")
            if run_forward:
                try:
                    prediction = predict_production(
                        kernel_spec, hw, method="midlayer")
                except NotImplementedError as error:
                    forward_error = True
                    row["forward_unavailable_reason"] = str(error)
                else:
                    forward_predictions.append(prediction.t_total_ns)
                    row["pred_forward_ns"] = prediction.t_total_ns
                    if prediction.diagnostics is None:
                        raise RuntimeError(
                            "Forward prediction must expose structured "
                            "diagnostics")
                    row["forward_support_level"] = (
                        prediction.diagnostics.support_level)
                    row["forward_limitations"] = (
                        prediction.diagnostics.limitations)
                    row["forward_bottleneck"] = prediction.bottleneck_group
            if run_roofline:
                try:
                    prediction_ns = predict_production(
                        kernel_spec, hw, method="roofline").t_total_ns
                except NotImplementedError as error:
                    roofline_error = True
                    row["roofline_unavailable_reason"] = str(error)
                else:
                    if roofline_launch_ns > 0:
                        prediction_ns += roofline_launch_ns
                    roofline_predictions.append(prediction_ns)
                    row["pred_roofline_ns"] = prediction_ns
            kernel_results.append(row)
        if run_forward and not forward_error and (
                len(forward_predictions) != len(specs)):
            raise RuntimeError(
                f"{task_key!r} produced an incomplete Forward result")
        if run_roofline and not roofline_error and (
                len(roofline_predictions) != len(specs)):
            raise RuntimeError(
                f"{task_key!r} produced an incomplete Roofline result")
        forward_pair = (
            "ERR" if forward_error
            else ((math.fsum(forward_predictions), total_actual)
                  if run_forward else None))
        roofline_pair = (
            "ERR" if roofline_error
            else ((math.fsum(roofline_predictions), total_actual)
                  if run_roofline else None))

    meta = {
        "op_type": op_type,
        "eqm": eqm if eqm is not None else 1,
        "range": (
            "tiny" if (eqm or 1) <= 16 else "small" if (eqm or 1) <= 64
            else "med" if (eqm or 1) <= 512 else "prefill"),
        "split": False,
        "actual_source": timing_observation.source,
        "actual_granularity": timing_observation.granularity,
        "actual_ns": total_actual,
        "max_launch_waves": max(launch_waves),
        "pipeweave_kernels": len(specs) if (run_pipeweave and specs) else None,
        "pipeweave_main_unsupported_kernels": (
            len(specs) if (run_pipeweave and specs) else None),
        "pipeweave_glue_roofline_kernels": 0 if run_pipeweave else None,
        "dsv4_stage": axes["dsv4_stage"],
        "dsv4_model": axes["dsv4_model"],
        "dsv4_layer_type": axes["dsv4_layer_type"],
    }
    if op_type == "WQ_FUSEDMOE":
        if specs is not None:
            meta["moe_ep_size"] = specs[0].attributes["expert_parallel_size"]
            meta["moe_experts"] = specs[0].shape["experts"]
            meta["moe_topk"] = specs[0].shape["topk"]
        else:
            meta["moe_ep_size"] = -1
            meta["moe_experts"] = -1
            meta["moe_topk"] = -1
    oracle_pair = "ERR" if run_oracle else None
    pipeweave_pair = PIPEWEAVE_OUT_OF_DOMAIN if run_pipeweave else None
    return (
        task_key, meta, oracle_pair, forward_pair, roofline_pair,
        pipeweave_pair, kernel_results)


# ── Parallel per-op evaluation (module-level, self-contained workers) ──
#
# Per-op work combines the native swizzle-LRU simulator, which releases the
# GIL, with Python max-plus pipeline evaluation.  A bounded thread pool retains
# native parallelism without oversubscribing the Python-heavy pipeline path.
# Workers are module-level (not nested closures) and take an explicit hardware
# config path; HWSpec is built once and cached per interpreter.

import functools


@functools.lru_cache(maxsize=None)
def _get_hardware(hw_config_path):
    """Cached HWSpec for a hardware config path."""
    return HWSpec(hw_config_path)


def _unsupported_observation_case(
        task_key, kernel_list, hw, timing_observation, op_type, eqm,
        gemm_shape, reason, run_oracle, run_forward, run_roofline,
        run_pipeweave):
    """Return an explicitly unsupported GEMM case.

    Used only when the observed launch identity cannot be normalized into
    the production contract (e.g. a persistent nvjet split-K grid whose K
    split is not stated by its geometry).  Every method reports the case as
    unsupported with the reason; no prediction is fabricated.
    """

    if op_type != "GEMM" or hw.aic_aiv_separated:
        raise ValueError(
            "unsupported-observation cases are defined for GPU GEMM only")
    from evaluation.kernel.profiler_observation import extract_ncu_metrics

    execution_units = hw.execution_units_for_task("generic")
    launch_waves = []
    kernel_results = []
    for kdata in kernel_list:
        metrics = extract_ncu_metrics(kdata)
        launch_waves.append(math.ceil(metrics["grid_size"] / execution_units))
        row = {
            "task_key": task_key,
            "op_type": op_type,
            "eqm": eqm,
            "actual_ns": None,
            "actual_source": timing_observation.source,
            "actual_granularity": timing_observation.granularity,
            "observation_unsupported_reason": reason,
        }
        if run_oracle:
            row["oracle_unavailable_reason"] = reason
        if run_forward:
            row["forward_unavailable_reason"] = reason
        if run_roofline:
            row["roofline_unavailable_reason"] = reason
        kernel_results.append(row)
    gemm_batch, gemm_m, gemm_k, gemm_n = gemm_shape
    meta = {
        "op_type": op_type,
        "eqm": eqm,
        "range": ("tiny" if eqm <= 16 else "small" if eqm <= 64
                  else "med" if eqm <= 512 else "prefill"),
        "split": True,
        "actual_source": timing_observation.source,
        "actual_granularity": timing_observation.granularity,
        "actual_ns": timing_observation.total_duration_ns,
        "max_launch_waves": max(launch_waves),
        "pipeweave_kernels": len(kernel_list) if run_pipeweave else None,
        "pipeweave_main_unsupported_kernels": (
            len(kernel_list) if run_pipeweave else None),
        "pipeweave_glue_roofline_kernels": 0 if run_pipeweave else None,
        "gemm_m": gemm_batch * gemm_m,
        "gemm_n": gemm_n,
        "gemm_k": gemm_k,
    }
    return (
        task_key, meta,
        "ERR" if run_oracle else None,
        "ERR" if run_forward else None,
        "ERR" if run_roofline else None,
        PIPEWEAVE_OUT_OF_DOMAIN if run_pipeweave else None,
        kernel_results)


def _process_op(task_key, entry, hw_config_path, run_oracle, run_forward,
                run_roofline=False, roofline_launch_ns=0.0,
                run_pipeweave=False, pipeweave_model_hw=None,
                capture_mechanisms=False):
    """Evaluate one op and call every method through the src predictor API."""
    if not isinstance(entry, dict):
        raise TypeError(f"dataset entry {task_key!r} must be a mapping")
    oi = entry["op_info"]
    op_type = oi["Type"]
    if op_type not in (
            "GEMM", "FUSEDMOE", "UNFUSEDMOE", "FLASHATTN", "EAGERATTN",
                        "NORM", "ROTARY", "ACTIVATION", "KV_CACHE", "ELEMENTWISE",
                        "V4ATTN", "WQ_FUSEDMOE"):
        raise NotImplementedError(
            f"evaluation does not support dataset op type {op_type!r}")

    hw = _get_hardware(hw_config_path)

    if op_type in ("V4ATTN", "WQ_FUSEDMOE"):
        return _process_dsv4_case_op(
            task_key, entry, hw, run_oracle, run_forward, run_roofline,
            roofline_launch_ns, run_pipeweave, pipeweave_model_hw,
            capture_mechanisms)
    attr = oi["Attr"]

    if "capture_evidence" in entry:
        return _process_fa_serving_capture_op(
            task_key, entry, hw, run_oracle, run_forward, run_roofline,
            roofline_launch_ns, run_pipeweave, pipeweave_model_hw,
            capture_mechanisms)

    gemm_shape = None
    attention_layout = None
    moe_experts = None
    moe_topk = None
    moe_max_tokens_per_expert = None
    vector_tokens = None
    if op_type in _MATRIX_LIKE_OP_TYPES:
        bmkn = _extract_bmkn(oi)
        B, M, K, N = bmkn
        eqm = B * M
        if op_type == "GEMM":
            gemm_shape = (B, M, K, N)
        elif op_type in MOE_OP_TYPES:
            moe_experts = _positive_int(attr, "E", f"{op_type} Attr")
            moe_topk = _positive_int(attr, "TOPK", f"{op_type} Attr")
            tokens_per_expert = _non_negative_int_sequence(
                attr, "tok_per_expert", f"{op_type} Attr")
            if len(tokens_per_expert) != moe_experts:
                raise ValueError(
                    f"{op_type} Attr.tok_per_expert has "
                    f"{len(tokens_per_expert)} entries, expected {moe_experts}")
            expected_routes = B * moe_topk
            if sum(tokens_per_expert) != expected_routes:
                raise ValueError(
                    f"{op_type} Attr.tok_per_expert sums to "
                    f"{sum(tokens_per_expert)}, expected TOK*TOPK="
                    f"{expected_routes}")
            moe_max_tokens_per_expert = max(tokens_per_expert)
        else:
            raise ValueError(
                f"unexpected matrix-like dataset op type {op_type!r}")
    elif op_type in ATTENTION_OP_TYPES:
        eqm = None
        attention_layout = attr["kv_layout"]
        if (not isinstance(attention_layout, str)
                or not attention_layout.strip()):
            raise TypeError(
                f"{op_type} Attr.kv_layout must be a non-empty string")
        attention_layout = attention_layout.strip().lower()
        if attention_layout not in _ATTENTION_LAYOUTS:
            raise ValueError(
                f"{op_type} Attr.kv_layout must be one of "
                f"{_ATTENTION_LAYOUTS}, got {attention_layout!r}")
    elif op_type in ("NORM", "ROTARY", "ACTIVATION", "KV_CACHE", "ELEMENTWISE"):
        # vector/eltwise: bucket by tok (the primary size driver); element_count
        # carries the full work for per-shape inspection.
        if "tok" in attr:
            eqm = _positive_int(attr, "tok", f"{op_type} Attr")
            vector_tokens = eqm
        elif "element_count" in attr:
            eqm = _positive_int(
                attr, "element_count", f"{op_type} Attr")
            vector_tokens = eqm
        else:
            raise ValueError(
                f"{op_type} requires tok or element_count for bucketing")
    else:
        raise ValueError(f"Unknown op_type: '{op_type}'")

    kernels_raw = entry["kernels"]
    if isinstance(kernels_raw, dict):
        kernel_list = list(kernels_raw.values())
    elif isinstance(kernels_raw, list):
        kernel_list = kernels_raw
    else:
        raise TypeError(f"dataset entry {task_key!r} kernels must be a mapping or list")
    if not kernel_list:
        raise ValueError(f"dataset entry {task_key!r} has no kernel observations")
    if any(not isinstance(kernel, dict) for kernel in kernel_list):
        raise TypeError(
            f"dataset entry {task_key!r} contains a non-mapping kernel record")
    validate_kernel_observation_sequence(
        kernel_list, hw.hardware_id, op_info=oi, hardware=hw)
    timing_observation = parse_kernel_evaluation_timing(
        entry, kernel_list, hw.hardware_id)
    has_per_kernel_timing = bool(timing_observation.durations_ns)
    if (has_per_kernel_timing
            and len(timing_observation.durations_ns) != len(kernel_list)):
        raise ValueError(
            f"dataset entry {task_key!r} has incomplete per-kernel timing")
    if timing_observation.granularity == "kernel":
        mechanism_timing_source = timing_observation.source
        mechanism_timing_unavailable_reason = None
    elif timing_observation.granularity == "op":
        mechanism_timing_source = None
        mechanism_timing_unavailable_reason = "canonical_target_is_op_level"
    else:
        raise ValueError(
            "mechanism capture received unknown timing granularity: "
            f"{timing_observation.granularity!r}")
    public_ascend_fused_moe = (
        op_type == "FUSEDMOE"
        and hw.hardware_id == "ascend_910b"
        and timing_observation.granularity == "op")
    production_attr = attr
    executed_attention_splits = None
    executed_gemm_split_k = None
    executed_gemm_partial_bytes = None
    executed_gemm_partial_dram_bytes = None
    if op_type == "FLASHATTN" and hw.hardware_id != "ascend_910b":
        production_attr = dict(attr)
        executed_attention_splits = observed_flash_attention_splits(
            attr, kernel_list, hw.hardware_id)
        production_attr["SP"] = executed_attention_splits
    elif op_type == "GEMM":
        try:
            executed_gemm_split_k = observed_gemm_split_k(kernel_list, attr)
        except NotImplementedError as error:
            return _unsupported_observation_case(
                task_key, kernel_list, hw, timing_observation, op_type,
                eqm, gemm_shape, str(error), run_oracle, run_forward,
                run_roofline, run_pipeweave)
        executed_gemm_partial_bytes = (
            observed_gemm_split_k_partial_bytes(kernel_list, attr))
        executed_gemm_partial_dram_bytes = (
            observed_gemm_split_k_partial_dram_bytes(kernel_list, attr))
    l2_execution_context = kernel_dataset_l2_execution_context(
        oi, hw.hardware_id)

    oracle_preds = []
    oracle_coverage_states = []
    forward_preds = []
    roofline_preds = []
    pipeweave_preds = []
    pipeweave_err = False
    pipeweave_main_unsupported = 0
    pipeweave_glue_roofline = 0
    has_gz = False
    oracle_err = False  # set if predictor.oracle() raises NotImplementedError
    forward_err = False
    roofline_err = False
    kernel_results_local = []
    normalized_attention_shape = None
    launch_waves = []

    moe_tensor_ordinal = 0
    moe_reduce_ordinal = 0
    for kidx, kdata in enumerate(kernel_list):
        from evaluation.kernel.profiler_observation import (
            extract_ncu_metrics,
            require_hardware_kernel_record_kind,
        )
        metrics = extract_ncu_metrics(kdata)
        require_hardware_kernel_record_kind(metrics, hw.hardware_id)
        if hw.hardware_id == "metax_c500":
            if "MetaX" not in kdata or not isinstance(kdata["MetaX"], dict):
                raise ValueError(
                    f"dataset entry {task_key!r} kernel {kidx} is missing "
                    "MetaX timing provenance")
            duration_source = kdata["MetaX"]["dur_source"]
            if duration_source != "mcTracer":
                raise ValueError(
                    f"dataset entry {task_key!r} kernel {kidx} has an "
                    "invalid C500 raw timing provenance: raw C500 duration must "
                    f"come from mcTracer, got {duration_source!r}")
        actual_ns = (
            timing_observation.durations_ns[kidx]
            if has_per_kernel_timing else None)
        actual_source = timing_observation.source
        if actual_ns is not None:
            if actual_ns <= 0:
                raise ValueError(
                    f"dataset entry {task_key!r} kernel {kidx} has "
                    "non-positive duration")

        ordinal = moe_tensor_ordinal
        if (op_type == "FUSEDMOE"
                and not is_non_tensor(
                    op_type, kdata["Metadata"]["Kernel Name"],
                    ascend_op_type=metrics["op_type"])):
            moe_tensor_ordinal += 1
        current_moe_reduce_ordinal = None
        if (op_type == "FUSEDMOE"
                and hw.hardware_id == "nvidia_a800"
                and "reduce_kernel" in metrics["kernel_name"].lower()):
            current_moe_reduce_ordinal = moe_reduce_ordinal
            moe_reduce_ordinal += 1
        production_op = {"Type": op_type, "Attr": production_attr}
        kernel_spec = normalize_kernel_spec(
            production_op, kdata, hw.hardware_id, hw,
            tensor_ordinal=ordinal,
            kernel_ordinal=kidx,
            moe_reduce_ordinal=current_moe_reduce_ordinal,
            gemm_split_k=executed_gemm_split_k,
            gemm_split_k_partial_bytes=executed_gemm_partial_bytes,
            gemm_split_k_partial_dram_bytes=(
                executed_gemm_partial_dram_bytes),
            profiler_metrics=metrics,
            l2_execution_context=l2_execution_context)
        predictor = create_predictor(kernel_spec.kind)
        launch_waves.append(_launch_waves_for_report(kernel_spec, hw))
        if kernel_spec.kind in (
                KernelKind.FLASH_ATTN, KernelKind.EAGER_ATTN):
            current_attention_shape = tuple(
                kernel_spec.shape[name]
                for name in ("batch", "q_len", "kv_len"))
            if normalized_attention_shape is None:
                normalized_attention_shape = current_attention_shape
            elif normalized_attention_shape != current_attention_shape:
                raise ValueError(
                    f"dataset entry {task_key!r} kernels disagree on "
                    "normalized attention shape")

        inp_oracle = None
        pred_oracle = None
        oracle_observation = None
        oracle_unavailable_reason = None
        if run_oracle:
            try:
                oracle_observation = normalize_oracle_observation(
                    kernel_spec, kdata, hw, predictor=predictor)
                inp_oracle = oracle_observation.prediction_input
                pred_oracle = predict_production(
                    kernel_spec, hw, method="oracle",
                    oracle_input=inp_oracle)
            except NotImplementedError as error:
                oracle_err = True
                oracle_unavailable_reason = str(error)
            else:
                if inp_oracle is None:
                    raise RuntimeError(
                        "Oracle input normalization returned None for "
                        f"dataset entry {task_key!r} kernel {kidx}")
                oracle_preds.append(pred_oracle.t_total_ns)
                oracle_coverage_states.append(oracle_observation.coverage)

        pred_forward = None
        forward_unavailable_reason = None
        if run_forward:
            try:
                pred_forward = predict_production(
                    kernel_spec, hw, method="midlayer")
            except NotImplementedError as error:
                forward_err = True
                forward_unavailable_reason = str(error)
            else:
                forward_preds.append(pred_forward.t_total_ns)

        pred_roofline_ns = None
        roofline_unavailable_reason = None
        if run_roofline:
            try:
                pred_roofline_ns = predict_production(
                    kernel_spec, hw, method="roofline").t_total_ns
            except NotImplementedError as error:
                roofline_err = True
                roofline_unavailable_reason = str(error)
            else:
                if roofline_launch_ns > 0:
                    pred_roofline_ns += roofline_launch_ns
                roofline_preds.append(pred_roofline_ns)

        if run_pipeweave:
            # The baseline is scored on the cases whose main launch it can
            # price.  Glue it has no model for is priced by Roofline instead,
            # so the case is still a complete prediction rather than one with
            # a silent hole; a main launch it cannot price puts the whole case
            # outside its domain and the case is excluded, not scored as if
            # the operator took no time.
            try:
                pipeweave_preds.append(predict_production(
                    kernel_spec, hw, method="pipeweave",
                    model_hardware=pipeweave_model_hw).t_total_ns)
            except NotImplementedError:
                if is_main_launch(kernel_spec, op_type):
                    pipeweave_main_unsupported += 1
                    pipeweave_preds.append(0.0)
                else:
                    try:
                        glue_ns = predict_production(
                            kernel_spec, hw, method="roofline").t_total_ns
                    except NotImplementedError:
                        # Neither method describes this launch, so the case has
                        # no complete prediction to score.
                        pipeweave_main_unsupported += 1
                        pipeweave_preds.append(0.0)
                    else:
                        if roofline_launch_ns > 0:
                            glue_ns += roofline_launch_ns
                        pipeweave_glue_roofline += 1
                        pipeweave_preds.append(glue_ns)

        if metrics["grid_z"] > 1:
            has_gz = True
        if op_type == "GEMM" and executed_gemm_split_k is not None and executed_gemm_split_k > 1:
            # Hopper nvjet splits K without a grid-z axis.
            has_gz = True

        kr = {
            "task_key": task_key,
            "op_type": op_type,
            "eqm": eqm,
            "actual_ns": actual_ns,
            "actual_source": actual_source,
            "actual_granularity": timing_observation.granularity,
        }
        if run_oracle and pred_oracle is not None:
            if oracle_observation is None:
                raise RuntimeError(
                    "Oracle prediction has no provenance-bearing observation")
            kr["is_nt"] = is_non_tensor(
                op_type,
                kdata["Metadata"]["Kernel Name"],
                ascend_op_type=metrics["op_type"],
            )
            kr["pred_oracle_ns"] = pred_oracle.t_total_ns
            if actual_ns is not None:
                kr["err_oracle"] = (
                    abs(pred_oracle.t_total_ns - actual_ns) / actual_ns * 100)
            kr["bottleneck"] = pred_oracle.bottleneck_group
            kr["oracle_backend"] = oracle_observation.backend
            kr["oracle_unavailable_fields"] = dict(
                oracle_observation.unavailable_fields)
            kr["oracle_target_derived_paths"] = list(
                oracle_observation.target_derived_paths)
            kr["oracle_coverage"] = oracle_observation.coverage
        elif run_oracle:
            if oracle_unavailable_reason is None:
                raise RuntimeError(
                    "unsupported Oracle prediction has no reason")
            kr["oracle_unavailable_reason"] = oracle_unavailable_reason
        if run_forward and pred_forward is not None:
            kr["pred_forward_ns"] = pred_forward.t_total_ns
            if actual_ns is not None:
                kr["err_forward"] = (
                    abs(pred_forward.t_total_ns - actual_ns) / actual_ns * 100)
            if pred_forward.diagnostics is None:
                raise RuntimeError(
                    "Forward prediction must expose structured diagnostics")
            kr["forward_support_level"] = (
                pred_forward.diagnostics.support_level)
            kr["forward_limitations"] = (
                pred_forward.diagnostics.limitations)
            kr["forward_bottleneck"] = pred_forward.bottleneck_group
        elif run_forward:
            kr["forward_unavailable_reason"] = forward_unavailable_reason
        if capture_mechanisms:
            if pred_forward is None:
                raise NotImplementedError(
                    "mechanism capture requires a supported Forward "
                    f"prediction: {forward_unavailable_reason}")
            from evaluation.kernel.mechanism_validation_report import (
                capture_kernel_mechanisms,
            )
            kr["mechanisms"] = capture_kernel_mechanisms(
                kernel_spec=kernel_spec,
                profiler_record=kdata,
                profiler_metrics=metrics,
                hardware=hw,
                forward_prediction=pred_forward,
                oracle_observation=oracle_observation,
                oracle_unavailable_reason=oracle_unavailable_reason,
                timing_target_ns=(
                    actual_ns if mechanism_timing_source is not None else None),
                timing_target_source=mechanism_timing_source,
                timing_target_unavailable_reason=(
                    mechanism_timing_unavailable_reason),
                kernel_ordinal=kidx,
            )
        if run_roofline and pred_roofline_ns is not None:
            kr["pred_roofline_ns"] = pred_roofline_ns
            if actual_ns is not None:
                kr["err_roofline"] = (
                    abs(pred_roofline_ns - actual_ns) / actual_ns * 100)
        elif run_roofline:
            kr["roofline_unavailable_reason"] = roofline_unavailable_reason
        kernel_results_local.append(kr)

    if op_type == "FUSEDMOE" and moe_tensor_ordinal == 0:
        raise ValueError(
            f"dataset entry {task_key!r} has no fused-MoE tensor projection")
    if public_ascend_fused_moe and run_forward:
        # The legacy kernel rows above are intentionally retained for Oracle
        # and mechanism diagnostics.  Replace only their Forward aggregate by
        # the source-derived public-call recipe; no profiler duration, runtime
        # name, or counter enters these production specs.
        # Legacy helper prediction support must not determine whether this
        # distinct public physical call is Forward-supported.
        forward_err = False
        public_specs = _ascend_public_fused_moe_specs(attr, hw)
        public_forward_predictions = []
        for kernel_spec in public_specs:
            try:
                prediction = predict_production(
                    kernel_spec, hw, method="midlayer")
            except NotImplementedError:
                forward_err = True
                public_forward_predictions = []
                break
            public_forward_predictions.append(prediction.t_total_ns)
        if not forward_err:
            forward_preds = public_forward_predictions
            launch_waves.extend(
                _launch_waves_for_report(kernel_spec, hw)
                for kernel_spec in public_specs)
    if op_type in ATTENTION_OP_TYPES:
        if normalized_attention_shape is None:
            raise RuntimeError(
                f"dataset entry {task_key!r} produced no attention shape")
        attention_batch, query_length, kv_length = normalized_attention_shape
        eqm = attention_batch * kv_length

    if run_oracle and not oracle_preds and not oracle_err:
        raise RuntimeError(f"{task_key!r} produced no Oracle observations")
    if run_forward and not forward_preds and not forward_err:
        raise RuntimeError(f"{task_key!r} produced no Midlayer observations")
    if run_roofline and not roofline_preds and not roofline_err:
        raise RuntimeError(f"{task_key!r} produced no Roofline observations")

    total_actual = timing_observation.total_duration_ns
    total_actual_source = timing_observation.source
    if total_actual <= 0:
        raise ValueError(f"{task_key!r} has non-positive total observed duration")

    if run_pipeweave and pipeweave_err:
        pipeweave_pair = "ERR"
    elif run_pipeweave and pipeweave_main_unsupported:
        pipeweave_pair = PIPEWEAVE_OUT_OF_DOMAIN
    elif run_pipeweave:
        if len(pipeweave_preds) != len(kernel_list):
            raise RuntimeError(
                f"{task_key!r} produced an incomplete PipeWeave prediction")
        pipeweave_pair = (sum(pipeweave_preds), total_actual)
    else:
        pipeweave_pair = None

    rng = "tiny" if eqm <= 16 else "small" if eqm <= 64 else "med" if eqm <= 512 else "prefill"
    meta = {
        "op_type": op_type,
        "eqm": eqm,
        "range": rng,
        "split": has_gz,
        "actual_source": total_actual_source,
        "actual_granularity": timing_observation.granularity,
        "actual_ns": total_actual,
        "max_launch_waves": max(launch_waves),
        "pipeweave_kernels": len(kernel_list) if run_pipeweave else None,
        "pipeweave_main_unsupported_kernels": (
            pipeweave_main_unsupported if run_pipeweave else None),
        "pipeweave_glue_roofline_kernels": (
            pipeweave_glue_roofline if run_pipeweave else None),
    }
    if op_type == "GEMM":
        if gemm_shape is None:
            raise RuntimeError("GEMM report metadata is missing its shape")
        gemm_batch, gemm_m, gemm_k, gemm_n = gemm_shape
        meta["gemm_m"] = gemm_batch * gemm_m
        meta["gemm_n"] = gemm_n
        meta["gemm_k"] = gemm_k
    elif op_type in MOE_OP_TYPES:
        meta["moe_experts"] = moe_experts
        meta["moe_topk"] = moe_topk
        meta["moe_max_tokens_per_expert"] = moe_max_tokens_per_expert
    elif op_type in (
            "NORM", "ROTARY", "ACTIVATION", "KV_CACHE", "ELEMENTWISE"):
        meta["vector_tokens"] = vector_tokens
    elif op_type in ATTENTION_OP_TYPES:
        pass
    else:
        raise ValueError(
            f"unexpected report metadata op type {op_type!r}")
    if run_oracle and oracle_coverage_states:
        if "partial" in oracle_coverage_states:
            meta["oracle_coverage"] = "partial"
        else:
            meta["oracle_coverage"] = "complete"
        meta["oracle_coverage_kernel_counts"] = dict(
            Counter(oracle_coverage_states))
    if op_type in ATTENTION_OP_TYPES:
        meta["query_length"] = query_length
        meta["kv_length"] = kv_length
        # GPU FA exposes the executed split-KV count in its observed launch.
        # Ascend FusedInferAttentionScore and decomposed EAGERATTN do not expose
        # an equivalent external split contract, so their value remains None
        # and they are intentionally absent from split-range report buckets.
        meta["executed_attention_splits"] = executed_attention_splits
        meta["attention_layout"] = attention_layout
    oracle_pair = "ERR" if oracle_err else ((sum(oracle_preds), total_actual) if run_oracle else None)
    forward_pair = (
        "ERR" if forward_err
        else ((sum(forward_preds), total_actual) if run_forward else None))
    roofline_pair = (
        "ERR" if roofline_err
        else ((sum(roofline_preds), total_actual) if run_roofline else None))
    return task_key, meta, oracle_pair, forward_pair, roofline_pair, pipeweave_pair, kernel_results_local


def process_op_star(packed):
    """Unpack a single tuple arg for executor.map."""
    try:
        return _process_op(*packed)
    except Exception as error:
        _annotate_exception(
            error, f"while evaluating dataset entry {packed[0]!r}")
        raise


def _annotate_exception(error: BaseException, note: str) -> None:
    """Attach context without depending on Python 3.11 exception notes."""

    if sys.version_info >= (3, 11):
        error.add_note(note)
    else:
        error.args = (*error.args, note)


def _process_op_timed_star(packed):
    """Evaluate one case and return its worker-local execution evidence."""

    started = time.perf_counter()
    result = process_op_star(packed)
    elapsed_seconds = time.perf_counter() - started
    return result, {
        "case_id": result[0],
        "worker_pid": os.getpid(),
        "elapsed_seconds": elapsed_seconds,
        "kernel_prediction_count": len(result[6]),
    }


def _validate_case_executor_args(packed, jobs, hardware_id):
    if not isinstance(packed, list):
        raise TypeError("kernel evaluation packed cases must be a list")
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError("kernel evaluation jobs must be an integer")
    if jobs <= 0:
        raise ValueError("kernel evaluation jobs must be positive")
    if hardware_id not in _EVALUATION_HARDWARE_IDS:
        raise ValueError(
            f"unknown kernel evaluation hardware: {hardware_id!r}")


def _preload_case_worker_contracts(
        packed, hardware_id, hardware_config=None):
    """Materialize immutable launch data and dispatch modules before forking."""

    if (hardware_id in _FORK_SHARED_GEMM_SELECTION_HARDWARE_IDS
            and any(item[1]["op_info"]["Type"] == "GEMM" for item in packed)):
        if hardware_config is None:
            preload_gemm_launch_selection(hardware_id)
        else:
            preload_gemm_launch_selection(
                hardware_id, hardware_config=hardware_config)
    predictor_kinds = set()
    for _, entry, *_ in packed:
        op_type = entry["op_info"]["Type"]
        if op_type not in _OP_TYPE_PREDICTOR_KINDS:
            raise NotImplementedError(
                f"evaluation does not support dataset op type {op_type!r}")
        predictor_kinds.update(_OP_TYPE_PREDICTOR_KINDS[op_type])
    for kind in sorted(predictor_kinds, key=lambda value: value.value):
        create_predictor(kind)


def process_kernel_evaluation_cases(
        packed, jobs, hardware_id, hardware_config=None):
    """Evaluate independent cases with one executor policy on every hardware."""

    _validate_case_executor_args(packed, jobs, hardware_id)
    _preload_case_worker_contracts(packed, hardware_id, hardware_config)
    if jobs == 1 or len(packed) <= 1:
        return [process_op_star(item) for item in packed]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(process_op_star, packed))


def process_kernel_evaluation_cases_with_timing(
        packed, jobs, hardware_id, hardware_config=None):
    """Evaluate independent cases and preserve one record per executor task."""

    _validate_case_executor_args(packed, jobs, hardware_id)
    _preload_case_worker_contracts(packed, hardware_id, hardware_config)
    if jobs == 1 or len(packed) <= 1:
        timed = [_process_op_timed_star(item) for item in packed]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            timed = list(pool.map(_process_op_timed_star, packed))
    results = [result for result, _ in timed]
    case_timing = [timing for _, timing in timed]
    return results, case_timing


def _run_forward_timing(
        packed, jobs, hardware_id, warmups, repetitions,
        hardware_config=None):
    """Measure only the production Forward case-evaluation region."""

    if isinstance(warmups, bool) or not isinstance(warmups, int):
        raise TypeError("Forward timing warmups must be an integer")
    if warmups < 0:
        raise ValueError("Forward timing warmups must be non-negative")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise TypeError("Forward timing repetitions must be an integer")
    if repetitions <= 0:
        raise ValueError("Forward timing repetitions must be positive")

    def process_cases():
        if hardware_config is None:
            return process_kernel_evaluation_cases(packed, jobs, hardware_id)
        return process_kernel_evaluation_cases(
            packed, jobs, hardware_id, hardware_config)

    for _ in range(warmups):
        process_cases()

    elapsed_seconds = []
    results = None
    for _ in range(repetitions):
        start = time.perf_counter()
        results = process_cases()
        elapsed_seconds.append(time.perf_counter() - start)
    if results is None:
        raise RuntimeError("Forward timing produced no measured result")
    return results, elapsed_seconds


def _sha256_bytes(payload):
    if not isinstance(payload, bytes):
        raise TypeError("SHA256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _dataset_content_sha256(raw_data):
    if not isinstance(raw_data, dict):
        raise TypeError("kernel evaluation dataset must be a mapping")
    canonical = json.dumps(
        raw_data, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha256_bytes(canonical)


def _artifact_source_path(path):
    """Return a checkout-independent path when the source belongs to Midlayer."""

    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _forward_error_count(results):
    if not isinstance(results, list):
        raise TypeError("Forward timing results must be a list")
    error_count = 0
    for result in results:
        if not isinstance(result, tuple) or len(result) != 7:
            raise TypeError("Forward timing received an invalid evaluation result")
        forward_pair = result[3]
        if forward_pair == "ERR":
            error_count += 1
        elif not isinstance(forward_pair, tuple) or len(forward_pair) != 2:
            raise TypeError(
                "Forward timing requires every result to contain a Forward pair")
    return error_count


def _require_exact_fields(mapping, expected, context):
    if not isinstance(mapping, dict):
        raise TypeError(f"{context} must be a mapping")
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{context} fields are invalid: missing={missing}, unknown={unknown}")


def _load_forward_timing_baseline(path):
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        _annotate_exception(
            error, f"while reading Forward timing baseline {path}")
        raise
    _require_exact_fields(payload, {
        "schema_version", "scope", "hardware_id", "mode", "jobs",
        "case_count", "dataset", "hardware_config", "timing",
        "forward_error_count", "gate",
    }, "Forward timing baseline")
    _require_exact_fields(
        payload["dataset"], {"source", "content_sha256"},
        "Forward timing baseline dataset")
    _require_exact_fields(
        payload["hardware_config"], {"source", "content_sha256"},
        "Forward timing baseline hardware_config")
    _require_exact_fields(payload["timing"], {
        "warmup_runs", "measured_runs", "elapsed_seconds",
        "median_seconds", "timed_region", "excluded_phases",
    }, "Forward timing baseline timing")
    if payload["schema_version"] != _FORWARD_TIMING_SCHEMA_VERSION:
        raise ValueError(
            "Forward timing baseline schema_version must be "
            f"{_FORWARD_TIMING_SCHEMA_VERSION}")
    if payload["scope"] != _FORWARD_TIMING_SCOPE:
        raise ValueError(
            f"Forward timing baseline scope must be {_FORWARD_TIMING_SCOPE!r}")
    samples = payload["timing"]["elapsed_seconds"]
    if (not isinstance(samples, list)
            or len(samples) != payload["timing"]["measured_runs"]):
        raise ValueError(
            "Forward timing baseline elapsed_seconds count must equal "
            "measured_runs")
    if (isinstance(payload["timing"]["median_seconds"], bool)
            or not isinstance(payload["timing"]["median_seconds"], (int, float))
            or not math.isfinite(payload["timing"]["median_seconds"])
            or payload["timing"]["median_seconds"] <= 0.0):
        raise ValueError(
            "Forward timing baseline median_seconds must be finite and positive")
    return payload


def _forward_timing_gate(current, baseline, baseline_path, max_ratio):
    if isinstance(max_ratio, bool) or not isinstance(max_ratio, (int, float)):
        raise TypeError("Forward timing max ratio must be numeric")
    if not math.isfinite(max_ratio) or max_ratio <= 0.0:
        raise ValueError("Forward timing max ratio must be finite and positive")
    identity_fields = (
        ("hardware_id", current["hardware_id"], baseline["hardware_id"]),
        ("mode", current["mode"], baseline["mode"]),
        ("jobs", current["jobs"], baseline["jobs"]),
        ("case_count", current["case_count"], baseline["case_count"]),
        ("dataset.content_sha256", current["dataset"]["content_sha256"],
         baseline["dataset"]["content_sha256"]),
        ("timing.warmup_runs", current["timing"]["warmup_runs"],
         baseline["timing"]["warmup_runs"]),
        ("timing.measured_runs", current["timing"]["measured_runs"],
         baseline["timing"]["measured_runs"]),
        ("timing.timed_region", current["timing"]["timed_region"],
         baseline["timing"]["timed_region"]),
        ("timing.excluded_phases", current["timing"]["excluded_phases"],
         baseline["timing"]["excluded_phases"]),
    )
    mismatches = [
        f"{name}: current={current_value!r}, baseline={baseline_value!r}"
        for name, current_value, baseline_value in identity_fields
        if current_value != baseline_value
    ]
    if mismatches:
        raise ValueError(
            "Forward timing baseline identity mismatch: " + "; ".join(mismatches))
    baseline_median = baseline["timing"]["median_seconds"]
    ratio = current["timing"]["median_seconds"] / baseline_median
    return {
        "baseline_path": str(Path(baseline_path)),
        "baseline_median_seconds": baseline_median,
        "max_ratio": float(max_ratio),
        "ratio": ratio,
        "passed": ratio <= max_ratio,
    }


def _build_forward_timing_artifact(
        *, hardware_id, jobs, case_count, dataset_source, dataset_sha256,
        hardware_config_source, hardware_config_sha256, warmups, samples,
        forward_error_count):
    if not samples:
        raise ValueError("Forward timing samples must be non-empty")
    if any(
            isinstance(sample, bool) or not isinstance(sample, (int, float))
            or not math.isfinite(sample) or sample <= 0.0
            for sample in samples):
        raise ValueError("Forward timing samples must be finite and positive")
    return {
        "schema_version": _FORWARD_TIMING_SCHEMA_VERSION,
        "scope": _FORWARD_TIMING_SCOPE,
        "hardware_id": hardware_id,
        "mode": "forward",
        "jobs": jobs,
        "case_count": case_count,
        "dataset": {
            "source": str(dataset_source),
            "content_sha256": dataset_sha256,
        },
        "hardware_config": {
            "source": str(hardware_config_source),
            "content_sha256": hardware_config_sha256,
        },
        "timing": {
            "warmup_runs": warmups,
            "measured_runs": len(samples),
            "elapsed_seconds": list(samples),
            "median_seconds": statistics.median(samples),
            "timed_region": _FORWARD_TIMED_REGION,
            "excluded_phases": list(_FORWARD_TIMING_EXCLUDED_PHASES),
        },
        "forward_error_count": forward_error_count,
        "gate": None,
    }


def _write_forward_timing_artifact(path, artifact):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def _build_case_timing_artifact(
        *, hardware_id, mode, jobs, dataset_source, dataset_sha256,
        hardware_config_source, hardware_config_sha256, case_timing):
    if not isinstance(case_timing, list) or not case_timing:
        raise ValueError("kernel case timing must contain at least one case")
    required_case_fields = {
        "case_id", "worker_pid", "elapsed_seconds", "kernel_prediction_count",
    }
    case_ids = []
    worker_pids = []
    total_kernel_predictions = 0
    for index, record in enumerate(case_timing):
        _require_exact_fields(
            record, required_case_fields, f"kernel case timing[{index}]")
        case_id = record["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise TypeError(f"kernel case timing[{index}].case_id must be non-empty")
        elapsed_seconds = record["elapsed_seconds"]
        if (isinstance(elapsed_seconds, bool)
                or not isinstance(elapsed_seconds, (int, float))
                or not math.isfinite(elapsed_seconds)
                or elapsed_seconds <= 0.0):
            raise ValueError(
                f"kernel case timing[{index}].elapsed_seconds must be positive")
        worker_pid = record["worker_pid"]
        if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
            raise ValueError(
                f"kernel case timing[{index}].worker_pid must be positive")
        kernel_prediction_count = record["kernel_prediction_count"]
        if (isinstance(kernel_prediction_count, bool)
                or not isinstance(kernel_prediction_count, int)
                or kernel_prediction_count < 0):
            raise ValueError(
                "kernel case timing[%d].kernel_prediction_count must be "
                "non-negative" % index)
        case_ids.append(case_id)
        worker_pids.append(worker_pid)
        total_kernel_predictions += kernel_prediction_count
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("kernel case timing contains duplicate case IDs")
    return {
        "schema_version": _CASE_TIMING_SCHEMA_VERSION,
        "scope": _CASE_TIMING_SCOPE,
        "hardware_id": hardware_id,
        "mode": mode,
        "jobs": jobs,
        "case_count": len(case_timing),
        "dataset": {
            "source": str(dataset_source),
            "content_sha256": dataset_sha256,
        },
        "hardware_config": {
            "source": str(hardware_config_source),
            "content_sha256": hardware_config_sha256,
        },
        "timing": {
            "timed_region": _CASE_TIMED_REGION,
            "excluded_phases": list(_CASE_TIMING_EXCLUDED_PHASES),
            "worker_count": len(set(worker_pids)),
            "total_case_execution_seconds": sum(
                record["elapsed_seconds"] for record in case_timing),
        },
        "work_count": {
            "kernel_prediction_count": total_kernel_predictions,
        },
        "cases": case_timing,
    }


def _write_case_timing_artifact(path, artifact):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def _build_semantic_artifact(
        *, hardware_id, mode, jobs, dataset_source, dataset_sha256,
        hardware_config_source, hardware_config_sha256, results):
    """Serialize all semantic Kernel outputs without execution timing."""

    if not isinstance(results, list) or not results:
        raise ValueError("Kernel semantic artifact requires non-empty results")
    cases = []
    case_ids = set()
    for index, result in enumerate(results):
        if not isinstance(result, tuple) or len(result) != 7:
            raise TypeError(
                f"Kernel semantic result[{index}] must be a seven-item tuple")
        (case_id, report_meta, oracle, forward, roofline, pipeweave,
         kernel_results) = result
        if not isinstance(case_id, str) or not case_id:
            raise TypeError(
                f"Kernel semantic result[{index}] case ID must be non-empty")
        if case_id in case_ids:
            raise ValueError(f"Kernel semantic artifact duplicates {case_id!r}")
        if not isinstance(report_meta, dict):
            raise TypeError(
                f"Kernel semantic result[{index}] report metadata must be a mapping")
        if not isinstance(kernel_results, list):
            raise TypeError(
                f"Kernel semantic result[{index}] kernel results must be a list")
        case_ids.add(case_id)
        cases.append({
            "case_id": case_id,
            "report_meta": report_meta,
            "oracle": oracle,
            "forward": forward,
            "roofline": roofline,
            "pipeweave": pipeweave,
            "kernel_results": kernel_results,
        })
    artifact = {
        "schema_version": _SEMANTIC_ARTIFACT_SCHEMA_VERSION,
        "scope": _SEMANTIC_ARTIFACT_SCOPE,
        "hardware_id": hardware_id,
        "mode": mode,
        "jobs": jobs,
        "case_count": len(cases),
        "dataset": {
            "source": str(dataset_source),
            "content_sha256": dataset_sha256,
        },
        "hardware_config": {
            "source": str(hardware_config_source),
            "content_sha256": hardware_config_sha256,
        },
        "cases": cases,
    }
    json.dumps(artifact, sort_keys=True, allow_nan=False)
    return artifact


def _write_semantic_artifact(path, artifact):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def _validate_forward_timing_cli(mode, output_path, baseline_path, jobs):
    if baseline_path is not None and output_path is None:
        raise ValueError(
            "--forward-timing-baseline requires --forward-timing-output")
    if output_path is not None and mode != "forward":
        raise ValueError("--forward-timing-output requires --mode forward")
    if baseline_path is not None and jobs != 1:
        raise ValueError("Forward timing baseline gate requires --jobs 1")


def _validate_case_timing_cli(case_timing_output, forward_timing_output):
    if case_timing_output is not None and forward_timing_output is not None:
        raise ValueError(
            "--case-timing-output cannot combine with --forward-timing-output")


def _require_forward_timing_gate_passed(artifact):
    gate = artifact["gate"]
    if gate is not None and not gate["passed"]:
        raise RuntimeError(
            "Forward timing gate failed: current/baseline="
            f"{gate['ratio']:.6f} exceeds {gate['max_ratio']:.6f}")


def predict_forward_op_total(
        *, task_key: str, entry: dict, hardware_id: str) -> dict:
    """Predict one logical OP through the production Forward entry point.

    This is deliberately a small public adapter for candidate-only timing
    reports.  It reuses ``_process_op`` so the report cannot acquire a second
    predictor dispatcher or duplicate any production formula.
    """

    if not isinstance(task_key, str) or not task_key:
        raise TypeError("task_key must be a non-empty string")
    if not isinstance(entry, dict):
        raise TypeError("entry must be a mapping")
    require_canonical_hardware_id(hardware_id)
    result = _process_op(
        task_key, entry, get_hardware_config_path(hardware_id), False, True)
    forward_pair = result[3]
    kernel_results = result[6]
    if forward_pair == "ERR":
        reasons = sorted({
            kernel_result["forward_unavailable_reason"]
            for kernel_result in kernel_results
            if "forward_unavailable_reason" in kernel_result
        })
        if not reasons:
            raise RuntimeError(
                "Forward reported unsupported without a kernel-level reason")
        return {
            "status": "unsupported",
            "unsupported_reasons": reasons,
        }
    if (not isinstance(forward_pair, tuple)
            or len(forward_pair) != 2):
        raise RuntimeError("Forward did not return one OP prediction pair")
    prediction_ns, canonical_dataset_total_ns = forward_pair
    if (not isinstance(prediction_ns, (int, float))
            or isinstance(prediction_ns, bool)
            or not math.isfinite(prediction_ns) or prediction_ns <= 0):
        raise ValueError("Forward OP prediction must be finite and positive")
    if (not isinstance(canonical_dataset_total_ns, (int, float))
            or isinstance(canonical_dataset_total_ns, bool)
            or not math.isfinite(canonical_dataset_total_ns)
            or canonical_dataset_total_ns <= 0):
        raise ValueError("canonical OP total must be finite and positive")
    return {
        "status": "supported",
        "prediction_ns": prediction_ns,
        "canonical_dataset_total_ns": canonical_dataset_total_ns,
    }



def compute_metrics(pairs, exclude_unsupported=False):
    """Return aggregate and tail error for ``(pred_ns, actual_ns)`` pairs.

    A pair may be the string "ERR" (the op's predictor has no oracle path —
    NotImplementedError, see _process_op). If any pair in the subset is ERR,
    returns an explicit unsupported status so the table renders "Err" instead of
    crashing or fabricating a 100%-error number from a zero-sum pred.

    A pair may also be ``PIPEWEAVE_OUT_OF_DOMAIN``: the method cannot price the
    case's main launch, so the case is outside what it claims to model.  Those
    are excluded from the error rather than scored, and their count is carried
    alongside so a reader can see the subset the number was computed over.  A
    subset with nothing left to score reports no error at all, because an
    error over zero cases is not a measurement.

    ``exclude_unsupported`` applies the same subset rule to "ERR" pairs: the
    Forward and Roofline tables score the cases a method prices and carry
    the explicitly unsupported count (``unsupported``) beside ``scored``; a
    subset with no priced case stays ``unsupported``."""
    if not pairs:
        return None
    malformed = [pair for pair in pairs
                 if pair not in ("ERR", PIPEWEAVE_OUT_OF_DOMAIN)
                 and (not isinstance(pair, tuple) or len(pair) != 2)]
    if malformed:
        raise TypeError(
            f"evaluation metric pairs have invalid entries: {malformed[:3]}")
    err_count = sum(1 for p in pairs if p == "ERR")
    out_of_domain = sum(1 for p in pairs if p == PIPEWEAVE_OUT_OF_DOMAIN)
    valid = [p for p in pairs if isinstance(p, tuple)]
    if err_count and (not exclude_unsupported or not valid):
        return {
            "status": "unsupported",
            "N": len(pairs),
            "unsupported": err_count,
        }
    if not valid:
        return {
            "status": "out_of_domain",
            "N": len(pairs),
            "out_of_domain": out_of_domain,
        }
    preds = np.array([p for p, _ in valid])
    actuals = np.array([a for _, a in valid])
    errs = np.abs(preds - actuals) / actuals * 100
    weights = actuals / actuals.sum()
    return {
        "status": "ok",
        "N": len(pairs),
        "scored": len(valid),
        "unsupported": err_count,
        "out_of_domain": out_of_domain,
        "MAPE": errs.mean(),
        "MedAPE": np.median(errs),
        "P90APE": np.percentile(errs, 90),
        "MaxAPE": errs.max(),
        "WMAE": float(np.sum(weights * errs)),
    }


def compute_keyed_tail_metrics(pairs_by_key, exclude_unsupported=False):
    """Return tail metrics and the deterministic worst-case dataset key."""

    if not isinstance(pairs_by_key, dict):
        raise TypeError("keyed evaluation pairs must be a mapping")
    if any(not isinstance(key, str) or not key for key in pairs_by_key):
        raise TypeError("keyed evaluation pair names must be non-empty strings")
    metrics = compute_metrics(
        list(pairs_by_key.values()), exclude_unsupported=exclude_unsupported)
    if metrics is None or metrics["status"] == "unsupported":
        return metrics
    if metrics["status"] != "ok":
        raise ValueError(
            f"unknown evaluation metric status: {metrics['status']!r}")
    errors = {
        key: abs(pair[0] - pair[1]) / pair[1] * 100
        for key, pair in pairs_by_key.items()
        if isinstance(pair, tuple)
    }
    worst_key = max(sorted(errors), key=lambda key: errors[key])
    return {**metrics, "worst_key": worst_key}


def define_subsets(meta_dict):
    """Return ordered list of (label, [task_keys], indent, is_last) for standard subset breakdown.
    is_last indicates this is the last child at its indent level (for tree lines).
    """
    subsets = []

    subsets.append(("ALL", list(meta_dict.keys()), 0, True))

    # GEMM + sub-items.  Shape thresholds are fixed powers-of-two serving
    # ranges; they are reporting categories, never Forward parameters.
    g = [k for k, v in meta_dict.items() if v["op_type"] == "GEMM"]
    subsets.append(("GEMM", g, 1, False))
    gemm_children = []
    # Structural: non-split vs split-K (disjoint, sum to total)
    gemm_children.append(("non-split", [k for k in g if not meta_dict[k]["split"]], 2))
    gemm_children.append(("split-K", [k for k in g if meta_dict[k]["split"]], 2))
    # By-size: contiguous ranges (disjoint, sum to total)
    for lb, lo, hi, desc in [("tiny", 1, 16, "B×M≤16"),
                              ("small", 17, 64, "17≤B×M≤64"),
                              ("med", 65, 2047, "65≤B×M≤2047"),
                              ("prefill", 2048, int(1e9), "B×M≥2048")]:
        keys = [k for k in g if lo <= meta_dict[k]["gemm_m"] <= hi]
        gemm_children.append((f"{lb} ({desc})", keys, 2))
    for axis in ("n", "k"):
        for label, lower, upper in (
                (f"{axis.upper()}≤4096", 1, 4096),
                (f"4097≤{axis.upper()}≤8192", 4097, 8192),
                (f"8193≤{axis.upper()}≤16384", 8193, 16384),
                (f"{axis.upper()}≥16385", 16385, None)):
            keys = [
                k for k in g
                if lower <= meta_dict[k][f"gemm_{axis}"]
                and (upper is None
                     or meta_dict[k][f"gemm_{axis}"] <= upper)
            ]
            gemm_children.append((label, keys, 2))
    # split-K, M, N, and K each terminate one independent decomposition.
    gemm_is_last = {1, 5, 9, 13}
    for i, child in enumerate(gemm_children):
        subsets.append((*child, i in gemm_is_last))

    # Fused and decomposed MoE remain distinct internally and are combined in
    # the presentation table.  The per-variant row is intentionally omitted:
    # each hardware currently carries only one variant, so it merely repeats
    # the aggregate row without adding a useful cross-hardware comparison.
    m = [k for k, v in meta_dict.items()
         if v["op_type"] in MOE_OP_TYPES]
    subsets.append(("FUSEDMOE/UNFUSEDMOE", m, 1, False))
    moe_children = []
    for lb, lo, hi, desc in [("tiny", 1, 16, "TOK≤16"),
                              ("small", 17, 64, "17≤TOK≤64"),
                              ("med", 65, 2047, "65≤TOK≤2047"),
                              ("prefill", 2048, int(1e9), "TOK≥2048")]:
        keys = [k for k in m if lo <= meta_dict[k]["eqm"] <= hi]
        moe_children.append((f"{lb} ({desc})", keys, 2))
    for label, lower, upper in (
            ("experts≤32", 1, 32),
            ("33≤experts≤128", 33, 128),
            ("129≤experts≤256", 129, 256),
            ("experts≥257", 257, None)):
        keys = [
            k for k in m
            if lower <= meta_dict[k]["moe_experts"]
            and (upper is None or meta_dict[k]["moe_experts"] <= upper)
        ]
        moe_children.append((label, keys, 2))
    for label, lower, upper in (
            ("max tokens/expert≤16", 0, 16),
            ("17≤max tokens/expert≤64", 17, 64),
            ("65≤max tokens/expert≤256", 65, 256),
            ("max tokens/expert≥257", 257, None)):
        keys = [
            k for k in m
            if lower <= meta_dict[k]["moe_max_tokens_per_expert"]
            and (upper is None
                 or meta_dict[k]["moe_max_tokens_per_expert"] <= upper)
        ]
        moe_children.append((label, keys, 2))
    for label, lower, upper in (
            ("top-k≤4", 1, 4),
            ("5≤top-k≤8", 5, 8),
            ("top-k≥9", 9, None)):
        keys = [
            k for k in m
            if lower <= meta_dict[k]["moe_topk"]
            and (upper is None or meta_dict[k]["moe_topk"] <= upper)
        ]
        moe_children.append((label, keys, 2))
    moe_is_last = {3, 7, 11, 14}
    for i, child in enumerate(moe_children):
        subsets.append((*child, i in moe_is_last))

    # DeepSeek-V4 H200 campaign op types: grouped by model
    # (deepseek-v4-flash/pro), by stage (decode/prefill/compressor/mhc), and
    # by expert-parallel size for the MXFP4 routed experts.
    v4 = [k for k, v in meta_dict.items() if v["op_type"] == "V4ATTN"]
    if v4:
        subsets.append(("V4ATTN", v4, 1, False))
        v4_children = []
        for stage in ("decode", "prefill", "compressor", "mhc_pre",
                      "mhc_post_pre", "unmapped"):
            keys = [k for k in v4 if meta_dict[k]["dsv4_stage"] == stage]
            if keys:
                v4_children.append((f"stage={stage}", keys, 2))
        for model in ("deepseek-v4-flash", "deepseek-v4-pro", "unattributed",
                      "unmapped"):
            keys = [k for k in v4 if meta_dict[k]["dsv4_model"] == model]
            if keys:
                v4_children.append((f"model={model}", keys, 2))
        for i, child in enumerate(v4_children):
            subsets.append((*child, i == len(v4_children) - 1))

    wq = [k for k, v in meta_dict.items() if v["op_type"] == "WQ_FUSEDMOE"]
    if wq:
        subsets.append(("WQ_FUSEDMOE", wq, 1, False))
        wq_children = []
        for model in ("deepseek-v4-flash", "deepseek-v4-pro", "unmapped"):
            keys = [k for k in wq if meta_dict[k]["dsv4_model"] == model]
            if keys:
                wq_children.append((f"model={model}", keys, 2))
        ep_sizes = sorted({meta_dict[k]["moe_ep_size"] for k in wq})
        for ep_size in ep_sizes:
            keys = [k for k in wq if meta_dict[k]["moe_ep_size"] == ep_size]
            label = f"EP{ep_size}" if ep_size > 0 else "EP unmapped"
            wq_children.append((label, keys, 2))
        for i, child in enumerate(wq_children):
            subsets.append((*child, i == len(wq_children) - 1))

    # FLASHATTN and EAGERATTN remain distinct internally and are combined only
    # here, in the final presentation table.
    attn = [k for k, v in meta_dict.items()
            if v["op_type"] in ATTENTION_OP_TYPES]
    subsets.append(("FLASHATTN/EAGERATTN", attn, 1, False))
    attn_children = []
    # Source: static profiler dataset vs serving-aligned (vLLM serving shapes)
    import re as _re
    static_pattern = r'(?:FLASHATTN|EAGERATTN)_B\d'
    static_keys = [k for k in attn if _re.match(static_pattern, k)]
    serving_keys = [k for k in attn if not _re.match(static_pattern, k)]
    attn_children.append(("static (ncu)", static_keys, 2))
    attn_children.append(("serving-aligned", serving_keys, 2))
    # Phase: decode vs prefill (disjoint, sum to total)
    dec = [k for k in attn if meta_dict[k]["query_length"] == 1]
    pfl = [k for k in attn if meta_dict[k]["query_length"] > 1]
    attn_children.append(("decode (Q=1)", dec, 2))
    attn_children.append(("prefill (Q=S)", pfl, 2))
    query_group_indexes = []
    for label, lower, upper in (
            ("Q=1", 1, 1),
            ("short Q (2–512)", 2, 512),
            ("medium Q (513–2048)", 513, 2048),
            ("long Q (≥2049)", 2049, None)):
        keys = [
            k for k in attn
            if lower <= meta_dict[k]["query_length"]
            and (upper is None or meta_dict[k]["query_length"] <= upper)
        ]
        if keys:
            query_group_indexes.append(len(attn_children))
            attn_children.append((label, keys, 2))
    # Executed split-KV count. This is the observed launch result, not a
    # requested split hint. Hardware that does not expose an actual split axis
    # (Ascend CANN and decomposed eager attention) is omitted from this
    # decomposition rather than being mislabeled SP=1.
    split_ranges = (
        ("SP=1", 1, 1),
        ("SP=2–4", 2, 4),
        ("SP=5–8", 5, 8),
        ("SP≥9", 9, None),
    )
    split_group_indexes = []
    for label, lower, upper in split_ranges:
        keys = [
            k for k in attn
            if meta_dict[k]["executed_attention_splits"] is not None
            and lower <= meta_dict[k]["executed_attention_splits"]
            and (upper is None
                 or meta_dict[k]["executed_attention_splits"] <= upper)
        ]
        if keys:
            split_group_indexes.append(len(attn_children))
            attn_children.append((label, keys, 2))
    # Actual maximum KV length, grouped into stable serving ranges instead of
    # one row per exact value (uniform-length workloads can produce values such
    # as 1000, 1018, or 4093).
    kv_ranges = (
        ("short KV (≤512)", 1, 512),
        ("medium KV (513–2048)", 513, 2048),
        ("long KV (2049–8192)", 2049, 8192),
        ("very-long KV (≥8193)", 8193, None),
    )
    kv_group_indexes = []
    for label, lower, upper in kv_ranges:
        keys = [
            k for k in attn
            if lower <= meta_dict[k]["kv_length"]
            and (upper is None or meta_dict[k]["kv_length"] <= upper)
        ]
        if keys:
            kv_group_indexes.append(len(attn_children))
            attn_children.append((label, keys, 2))
    layout_group_indexes = []
    for layout in _ATTENTION_LAYOUTS:
        keys = [
            k for k in attn
            if meta_dict[k]["attention_layout"] == layout
        ]
        if keys:
            layout_group_indexes.append(len(attn_children))
            attn_children.append((f"layout={layout}", keys, 2))
    # is_last=True for last of each decomposition group:
    # serving-aligned (source), prefill (phase), and the final non-empty row of
    # the executed-split and KV-length decompositions.
    attn_is_last = {1, 3}
    if query_group_indexes:
        attn_is_last.add(query_group_indexes[-1])
    if split_group_indexes:
        attn_is_last.add(split_group_indexes[-1])
    if kv_group_indexes:
        attn_is_last.add(kv_group_indexes[-1])
    if layout_group_indexes:
        attn_is_last.add(layout_group_indexes[-1])
    for i, child in enumerate(attn_children):
        subsets.append((*child, i in attn_is_last))

    # Vector/eltwise ops (AIV-domain on 910B) — break out by op_type so each
    # predictor's MAPE is visible (goal §4: NORM/ROTARY/ACTIVATION/ELEMENTWISE/
    # KV_CACHE per-metric MAPE). Bucket by tok (the primary size driver).
    vec = [k for k, v in meta_dict.items()
           if v["op_type"] in ("NORM", "ROTARY", "ACTIVATION",
                               "KV_CACHE", "ELEMENTWISE")]
    if vec:
        subsets.append(("VECTOR/ELTWISE", vec, 1, False))
        vec_children = []
        for ot in ("NORM", "ROTARY", "ACTIVATION", "ELEMENTWISE", "KV_CACHE"):
            keys = [k for k in vec if meta_dict[k]["op_type"] == ot]
            if keys:
                vec_children.append((ot, keys, 2))
        vector_range_indexes = []
        for label, lower, upper in (
                ("tokens≤16", 1, 16),
                ("17≤tokens≤64", 17, 64),
                ("65≤tokens≤2047", 65, 2047),
                ("tokens≥2048", 2048, None)):
            keys = [
                k for k in vec
                if lower <= meta_dict[k]["vector_tokens"]
                and (upper is None
                     or meta_dict[k]["vector_tokens"] <= upper)
            ]
            if keys:
                vector_range_indexes.append(len(vec_children))
                vec_children.append((label, keys, 2))
        vector_is_last = {len(vec_children) - len(vector_range_indexes) - 1}
        if vector_range_indexes:
            vector_is_last.add(vector_range_indexes[-1])
        for i, child in enumerate(vec_children):
            subsets.append((*child, i in vector_is_last))

    # Launch waves are a cross-operator view over the maximum waves among the
    # kernels that implement one dataset operation. The execution-domain
    # denominator comes from HWSpec (SM/AP, AIC/mixed, or AIV), while the grid
    # is the normalized discrete launch fact. These report-only buckets do not
    # feed the Forward model.
    all_keys = list(meta_dict.keys())
    subsets.append(("MAX LAUNCH WAVES", all_keys, 1, True))
    wave_children = []
    for label, lower, upper in (
            ("1 wave", 1, 1),
            ("2–4 waves", 2, 4),
            ("5–16 waves", 5, 16),
            ("≥17 waves", 17, None)):
        keys = [
            k for k in all_keys
            if lower <= meta_dict[k]["max_launch_waves"]
            and (upper is None
                 or meta_dict[k]["max_launch_waves"] <= upper)
        ]
        if keys:
            wave_children.append((label, keys, 2))
    for i, child in enumerate(wave_children):
        subsets.append((*child, i == len(wave_children) - 1))

    return subsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "oracle", "forward", "roofline", "pipeweave", "both",
            "standard", "all",
        ],
        default="standard",
        help=(
            "Evaluation mode (default: standard=oracle+forward+roofline; "
            "PipeWeave runs only when explicitly selected by pipeweave/all)"),
    )
    parser.add_argument("--roofline-launch", action="store_true",
                        help="Add dispatch_overhead to roofline (version②). "
                             "Only affects roofline predictions.")
    parser.add_argument("--pipeweave-model-hw", default=None,
                        help="PipeWeave MLP model source hardware (default: same as "
                             "the canonical target hardware). Set to 'A100' for cross-hw eval.")
    parser.add_argument(
        "--hardware-id", required=True,
        help="Canonical hardware ID; resolves configs/hardware and kernel_dataset by convention.")
    parser.add_argument("--dataset", default=None,
                        help="Path to a merged dataset JSON, overriding the canonical "
                             "kernel_dataset/<hardware_id>/ directory.")
    parser.add_argument(
        "--candidate-root", default=None,
        help=("Candidate root (<plan>/<family>.json.zst + manifests) "
              "overriding configs/evaluation/kernel_candidate_roots.yaml."))
    parser.add_argument("--jobs", "-j", type=int, default=None,
                        help="Parallel worker threads (default: min(8, "
                             "os.cpu_count())). 1 = serial.")
    parser.add_argument(
        "--forward-timing-output", type=Path, default=None,
        help=(
            "Write structured pure-Forward timing JSON. The timed region "
            "contains only case evaluation, excluding loading and reports."))
    parser.add_argument(
        "--case-timing-output", type=Path, default=None,
        help=(
            "Write one worker-local timing and work-count record per Kernel "
            "case. The timed region is _process_op and excludes queue wait."))
    parser.add_argument("--report-output", type=Path, help="Offline Kernel HTML output (only with --interactive-report)")
    parser.add_argument("--interactive-report", action="store_true", help="Write the offline Kernel HTML report plus its projections")
    parser.add_argument("--no-report", dest="interactive_report", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--semantic-output", type=Path, default=None,
        help=(
            "Write the complete semantic result for every evaluated case, "
            "excluding execution timing."))
    parser.add_argument(
        "--forward-timing-baseline", type=Path, default=None,
        help=(
            "Compare pure-Forward median against a prior timing JSON. "
            "Requires --mode forward, --jobs 1, and --forward-timing-output."))
    parser.add_argument(
        "--forward-timing-max-ratio", type=float, default=1.10,
        help="Maximum current/baseline median ratio (default: 1.10).")
    parser.add_argument(
        "--forward-timing-repetitions", type=int, default=5,
        help="Measured pure-Forward repetitions (default: 5).")
    parser.add_argument(
        "--forward-timing-warmups", type=int, default=1,
        help="Unmeasured pure-Forward warmups (default: 1).")
    args = parser.parse_args()
    if args.report_output is not None and not args.interactive_report:
        parser.error("--report-output requires --interactive-report")
    from evaluation.reporting.kernel_overview import default_kernel_report_path, write_kernel_overview
    report_output = (args.report_output if args.report_output is not None else default_kernel_report_path()) if args.interactive_report else None
    if report_output is not None:
        if report_output.suffix != ".html":
            parser.error("--report-output must end in .html")
        report_files = {report_output.resolve(), report_output.with_suffix(".visualization.json").resolve(), report_output.with_suffix(".report.json").resolve(), report_output.with_suffix(".semantic.json").resolve()}
        for artifact_output in (args.semantic_output, args.forward_timing_output, args.case_timing_output):
            if artifact_output is not None and artifact_output.resolve() in report_files:
                parser.error("report and evaluation artifact paths must be separate")
    started = time.perf_counter()
    try:
        _run_evaluation(args, report_output, started)
    except Exception as error:
        if report_output is not None:
            receipt = write_kernel_overview(None, {}, vars(args), report_output,
                elapsed_s=time.perf_counter() - started,
                failure={"type": type(error).__name__, "message": str(error)})
            print("Kernel HTML report: " + receipt["path"])
        raise


def _run_evaluation(args, report_output, started):
    run_oracle = args.mode in ("oracle", "both", "standard", "all")
    run_forward = args.mode in ("forward", "both", "standard", "all")
    run_roofline = args.mode in ("roofline", "standard", "all")
    run_pipeweave = args.mode in ("pipeweave", "all")

    # Load: if --dataset points to a single file, dispatch by its explicit
    # storage suffix; otherwise merge canonical per-kernel JSON.zst files.
    hardware_id = require_canonical_hardware_id(args.hardware_id)
    if args.dataset is not None and args.candidate_root is not None:
        raise ValueError("--dataset and --candidate-root are exclusive")
    if args.candidate_root is not None:
        dataset_path = os.path.abspath(args.candidate_root)
        raw_data = load_kernel_candidate_root(hardware_id, dataset_path)
    elif args.dataset is None:
        dataset_kind, dataset_path = kernel_evaluation_dataset_location(
            hardware_id)
        raw_data = load_kernel_evaluation_dataset(hardware_id)
        if dataset_kind == "candidate":
            dataset_path = f"{dataset_path} (candidate root)"
    elif isinstance(args.dataset, str) and args.dataset:
        dataset_path = args.dataset
        if dataset_path.endswith(KERNEL_DATASET_SUFFIX):
            raw_data = load_kernel_dataset_document(dataset_path)
        elif dataset_path.endswith(".json"):
            with open(dataset_path) as stream:
                raw_data = json.load(stream)
        else:
            raise ValueError(
                "--dataset file must end with '.json' or "
                f"{KERNEL_DATASET_SUFFIX!r}")
    else:
        raise ValueError("--dataset must be a non-empty path when provided")

    hw_config_path = get_hardware_config_path(hardware_id)
    # Populate the same per-interpreter cache that case workers use before
    # ProcessPoolExecutor forks them.  HWSpec is immutable after construction,
    # so forked workers can reuse the parsed hardware contract instead of each
    # reparsing the YAML once before their first assigned case.
    hw = _get_hardware(hw_config_path)
    roofline_launch_ns = hw.dispatch_overhead_ns if (run_roofline and args.roofline_launch) else 0.0

    print(f"Dataset: {dataset_path}")
    print(f"Hardware: {hw}")
    print(f"t = max_groups(sum_within) + C_launch, C_launch = {hw.dispatch_overhead_ns:.0f} ns")

    # Evaluate
    kernel_results = []
    op_pairs_oracle = {}
    op_pairs_forward = {}
    op_pairs_roofline = {}
    op_pairs_pipeweave = {}
    op_meta = {}

    # Process ops with a bounded thread pool.  Native LRU simulation benefits
    # from several workers, while the Python max-plus evaluator contends on the
    # GIL; using every logical CPU makes the mixed workload slower and also
    # oversubscribes the cross-hardware full-evaluation matrix.
    items = sorted(raw_data.items())
    if args.jobs is not None:
        if args.jobs <= 0:
            raise ValueError("--jobs must be positive")
        n_workers = args.jobs
    else:
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("cannot determine CPU count; pass --jobs explicitly")
        n_workers = min(8, cpu_count)
    _validate_forward_timing_cli(
        args.mode, args.forward_timing_output,
        args.forward_timing_baseline, n_workers)
    _validate_case_timing_cli(
        args.case_timing_output, args.forward_timing_output)
    packed = [(k, v, hw_config_path, run_oracle, run_forward, run_roofline, roofline_launch_ns,
               run_pipeweave, args.pipeweave_model_hw)
              for k, v in items]
    if args.forward_timing_output is None:
        if args.case_timing_output is None:
            results = process_kernel_evaluation_cases(
                packed, n_workers, hardware_id, hw._source_config)
        else:
            results, case_timing = process_kernel_evaluation_cases_with_timing(
                packed, n_workers, hardware_id, hw._source_config)
            case_timing_artifact = _build_case_timing_artifact(
                hardware_id=hardware_id,
                mode=args.mode,
                jobs=n_workers,
                dataset_source=dataset_path,
                dataset_sha256=_dataset_content_sha256(raw_data),
                hardware_config_source=hw_config_path,
                hardware_config_sha256=file_sha256(Path(hw_config_path)),
                case_timing=case_timing,
            )
            _write_case_timing_artifact(
                args.case_timing_output, case_timing_artifact)
            print(
                "Case timing: "
                f"cases={case_timing_artifact['case_count']} "
                f"workers={case_timing_artifact['timing']['worker_count']} "
                f"jobs={n_workers}")
    else:
        results, elapsed_seconds = _run_forward_timing(
            packed, n_workers, hardware_id,
            args.forward_timing_warmups,
            args.forward_timing_repetitions,
            hw._source_config)
        timing_artifact = _build_forward_timing_artifact(
            hardware_id=hardware_id,
            jobs=n_workers,
            case_count=len(items),
            dataset_source=dataset_path,
            dataset_sha256=_dataset_content_sha256(raw_data),
            hardware_config_source=hw_config_path,
            hardware_config_sha256=file_sha256(Path(hw_config_path)),
            warmups=args.forward_timing_warmups,
            samples=elapsed_seconds,
            forward_error_count=_forward_error_count(results),
        )
        if args.forward_timing_baseline is not None:
            baseline = _load_forward_timing_baseline(
                args.forward_timing_baseline)
            timing_artifact["gate"] = _forward_timing_gate(
                timing_artifact, baseline,
                args.forward_timing_baseline,
                args.forward_timing_max_ratio)
        _write_forward_timing_artifact(
            args.forward_timing_output, timing_artifact)
        print(
            "Forward-only timing: "
            f"median={timing_artifact['timing']['median_seconds']:.6f}s "
            f"runs={timing_artifact['timing']['measured_runs']} "
            f"jobs={n_workers}")
        _require_forward_timing_gate_passed(timing_artifact)

    for result in results:
        task_key, meta, oracle_pair, forward_pair, roofline_pair, pipeweave_pair, krs = result
        op_meta[task_key] = meta
        if oracle_pair is not None:
            op_pairs_oracle[task_key] = oracle_pair
        if forward_pair is not None:
            op_pairs_forward[task_key] = forward_pair
        if roofline_pair is not None:
            op_pairs_roofline[task_key] = roofline_pair
        if pipeweave_pair is not None:
            op_pairs_pipeweave[task_key] = pipeweave_pair
        kernel_results.extend(krs)

    if args.semantic_output is not None or report_output is not None:
        semantic_artifact = _build_semantic_artifact(
            hardware_id=hardware_id,
            mode=args.mode,
            jobs=n_workers,
            dataset_source=dataset_path,
            dataset_sha256=_dataset_content_sha256(raw_data),
            hardware_config_source=_artifact_source_path(hw_config_path),
            hardware_config_sha256=file_sha256(Path(hw_config_path)),
            results=results,
        )
        if args.semantic_output is not None:
            _write_semantic_artifact(args.semantic_output, semantic_artifact)

    # Print table
    subsets = define_subsets(op_meta)

    # Determine column widths from data
    label_w = 4   # minimum "N" header
    for label, keys, indent, is_last in subsets:
        # Tree prefix: "├── " or "└── " for indent=2, "  " for indent=1
        if indent == 0:
            display = label
        elif indent == 1:
            display = "├─ " + label if not is_last else "└─ " + label
        elif indent >= 2:
            display = "│  ├─ " + label if not is_last else "│  └─ " + label
        else:
            raise ValueError(f"subset indent cannot be negative: {indent}")
        # For indent=2 under the last indent=1 item, use "   " instead of "│  "
        # We need to track parent is_last status — simplified: use fixed prefix
        label_w = max(label_w, len(display) + 1)

    # Build header matching data row widths exactly
    col_n = 5
    col_metric = 7  # e.g. "10.3" = 4 chars, right-aligned in 7

    # Each method group = 3 metrics with spaces.
    group_w = col_metric * 3 + 2  # MAPE Med WMAE with single-space gaps

    oracle_task_coverage = Counter()
    if run_oracle:
        for meta in op_meta.values():
            if "oracle_coverage" in meta:
                oracle_task_coverage[meta["oracle_coverage"]] += 1
        coverage_states = set(oracle_task_coverage)
        if len(coverage_states) == 1:
            only_state = next(iter(coverage_states))
            oracle_header = f"Oracle ({only_state})"
        elif coverage_states:
            oracle_header = "Oracle (mixed)"
        else:
            oracle_header = "Oracle"

    header1 = f"{'Dataset':<{label_w}}   {'N':>{col_n}}"
    header2 = f"{'':<{label_w}}   {'':>{col_n}}"
    if run_forward:
        header1 += f" │ {'Forward':^{group_w}}"
        header2 += f" │ {'MAPE':>{col_metric}} {'Med':>{col_metric}} {'WMAE':>{col_metric}}"
    if run_oracle:
        header1 += f" │ {oracle_header:^{group_w}}"
        header2 += f" │ {'MAPE':>{col_metric}} {'Med':>{col_metric}} {'WMAE':>{col_metric}}"
    if run_roofline:
        header1 += f" │ {'Roofline':^{group_w}}"
        header2 += f" │ {'MAPE':>{col_metric}} {'Med':>{col_metric}} {'WMAE':>{col_metric}}"
    if run_pipeweave:
        header1 += f" │ {'PipeWeave':^{group_w}}"
        header2 += f" │ {'MAPE':>{col_metric}} {'Med':>{col_metric}} {'WMAE':>{col_metric}}"

    sep_len = max(len(header1), len(header2))
    sep = "─" * sep_len

    print(f"{'═' * sep_len}")
    print(header1)
    print(header2)
    print(sep)

    # Track which indent=1 parents have been completed (for tree lines)
    # When a parent is the last among siblings, its children use "   " prefix
    # instead of "│  "
    parent_last = {}  # indent -> bool, whether parent at this indent was last
    pipeweave_row_subsets = []
    forward_row_subsets = []
    printed_rows = []

    # Track previous label for └─ group boundary markers
    _prev_label = ""
    for label, keys, indent, is_last in subsets:
        if not keys:
            continue

        _prev_label = label

        om = compute_metrics([op_pairs_oracle[k] for k in keys]) if run_oracle else None
        fm = compute_metrics(
            [op_pairs_forward[k] for k in keys],
            exclude_unsupported=True) if run_forward else None
        rm = compute_metrics(
            [op_pairs_roofline[k] for k in keys],
            exclude_unsupported=True) if run_roofline else None
        pm = compute_metrics(
            [op_pairs_pipeweave[k] for k in keys]) if run_pipeweave else None

        if om is None and fm is None and rm is None and pm is None:
            continue

        available_metrics = [
            metrics for metrics in (om, fm, rm, pm) if metrics is not None]
        counts = {metrics["N"] for metrics in available_metrics}
        if len(counts) != 1:
            raise ValueError(
                f"evaluation methods disagree on sample count for {label!r}: "
                f"{sorted(counts)}")
        n = available_metrics[0]["N"]

        # Build tree prefix
        if indent == 0:
            display_label = label
        elif indent == 1:
            connector = "└─" if is_last else "├─"
            display_label = connector + " " + label
        elif indent >= 2:
            # Check if parent indent=1 was last -> use "   " instead of "│  "
            parent_prefix = (
                "   " if 1 in parent_last and parent_last[1] else "│  ")
            connector = "└─" if is_last else "├─"
            display_label = parent_prefix + connector + " " + label
        else:
            raise ValueError(f"evaluation row indent cannot be negative: {indent}")

        if indent == 1:
            parent_last[indent] = is_last

        parts = []
        left = f"{display_label:<{label_w}}   {n:>{col_n}}"
        parts.append(left)

        def _format_metric_group(metrics):
            if metrics is None:
                return " " * group_w
            if metrics["status"] == "unsupported":
                return f"{'Err':>{group_w}}"
            elif metrics["status"] == "out_of_domain":
                return f"{'--':>{group_w}}"
            elif metrics["status"] == "ok":
                return (
                    f"{metrics['MAPE']:>{col_metric}.2f} "
                    f"{metrics['MedAPE']:>{col_metric}.2f} "
                    f"{metrics['WMAE']:>{col_metric}.2f}")
            else:
                raise ValueError(
                    f"unknown evaluation metric status: {metrics['status']!r}")

        if run_forward:
            parts.append(_format_metric_group(fm))
            if fm is not None and fm["status"] == "ok" and fm["unsupported"]:
                forward_row_subsets.append(
                    (len(printed_rows), display_label, fm["scored"], fm["N"]))

        if run_oracle:
            parts.append(_format_metric_group(om))

        if run_roofline:
            parts.append(_format_metric_group(rm))

        if run_pipeweave:
            parts.append(_format_metric_group(pm))
            # The N column is shared by every method, but PipeWeave may have
            # scored fewer cases than it names.  Record the difference so the
            # subset each number was computed over is printed, never implied.
            if pm is not None and pm["status"] == "ok" and pm["out_of_domain"]:
                pipeweave_row_subsets.append(
                    (label, pm["scored"], pm["N"]))
            elif pm is not None and pm["status"] == "out_of_domain":
                pipeweave_row_subsets.append((label, 0, pm["N"]))

        print(" │ ".join(parts))
        printed_rows.append(display_label)

    if run_forward and forward_row_subsets:
        print(f"{'═' * sep_len}")
        print("  Forward rows scored over a subset of their N "
              "(unsupported cases excluded)")
        print(f"{'═' * sep_len}")
        for row_index, label, scored, total in forward_row_subsets:
            print(f"  row {row_index:>3} scored {scored:>5} / {total:<5} "
                  f"{label.strip()}")

    if run_pipeweave and pipeweave_row_subsets:
        print(f"{'═' * sep_len}")
        print("  PipeWeave rows scored over a subset of their N")
        print(f"{'═' * sep_len}")
        for label, scored, total in pipeweave_row_subsets:
            print(f"  {label:<38} scored {scored:>5} / {total}")

    if run_pipeweave:
        scored_cases = 0
        glue_filled_cases = 0
        excluded_cases = 0
        total_kernels = 0
        glue_kernels = 0
        excluded_by_op = Counter()
        glue_by_op = Counter()
        for key, meta in op_meta.items():
            kernels = meta["pipeweave_kernels"]
            main_missing = meta["pipeweave_main_unsupported_kernels"]
            glue_filled = meta["pipeweave_glue_roofline_kernels"]
            if kernels is None or main_missing is None or glue_filled is None:
                raise RuntimeError(
                    f"{key!r} ran PipeWeave without recording its coverage")
            total_kernels += kernels
            if main_missing:
                excluded_cases += 1
                excluded_by_op[meta["op_type"]] += 1
                continue
            scored_cases += 1
            glue_kernels += glue_filled
            if glue_filled:
                glue_filled_cases += 1
                glue_by_op[meta["op_type"]] += glue_filled
        case_total = scored_cases + excluded_cases
        print(f"{'═' * sep_len}")
        print("  PipeWeave domain (glue priced by Roofline; cases whose main")
        print("  launch PipeWeave cannot price are excluded, not scored)")
        print(f"{'═' * sep_len}")
        print(f"  cases scored                  {scored_cases:>6} / {case_total}")
        print(f"    of which glue came from Roofline "
              f"{glue_filled_cases:>6}")
        print(f"  cases excluded (main launch)  {excluded_cases:>6} / {case_total}")
        for op_type, count in sorted(excluded_by_op.items()):
            print(f"    excluded {op_type:<28} {count:>6}")
        print(f"  glue launches from Roofline   {glue_kernels:>6} / {total_kernels}")
        for op_type, count in sorted(glue_by_op.items()):
            print(f"    glue in {op_type:<29} {count:>6}")

    timing_source_counts = Counter(
        meta["actual_source"] for meta in op_meta.values())
    print(f"{'═' * sep_len}")
    print("  Evaluation timing targets")
    print(f"{'═' * sep_len}")
    for source, count in sorted(timing_source_counts.items()):
        print(f"  {source:<30} tasks={count:>4}")

    if run_forward:
        print(f"{'═' * sep_len}")
        print("  Forward tail accuracy")
        print(f"{'═' * sep_len}")
        tail_labels = (
            "ALL", "GEMM", "FUSEDMOE/UNFUSEDMOE",
            "FLASHATTN/EAGERATTN", "VECTOR/ELTWISE",
            "NORM", "ROTARY", "ACTIVATION", "ELEMENTWISE", "KV_CACHE",
        )
        subset_keys = {
            label: keys for label, keys, _indent, _is_last in subsets
            if label in tail_labels
        }
        for label in tail_labels:
            if label not in subset_keys or not subset_keys[label]:
                continue
            metrics = compute_keyed_tail_metrics({
                key: op_pairs_forward[key] for key in subset_keys[label]
            }, exclude_unsupported=True)
            if metrics["status"] == "unsupported":
                print(
                    f"  {label:<28} unsupported="
                    f"{metrics['unsupported']}/{metrics['N']}")
            elif metrics["status"] == "ok":
                print(
                    f"  {label:<28} P90={metrics['P90APE']:>6.1f}% "
                    f"Max={metrics['MaxAPE']:>6.1f}%  "
                    f"worst={metrics['worst_key']}")
            else:
                raise ValueError(
                    "unknown evaluation metric status: "
                    f"{metrics['status']!r}")

        forward_support = Counter(
            result["forward_support_level"] for result in kernel_results
            if "forward_support_level" in result)
        forward_unsupported = sum(
            "forward_unavailable_reason" in result
            for result in kernel_results)
        print(f"{'═' * sep_len}")
        print("  Forward support coverage")
        print(f"{'═' * sep_len}")
        for state in (
                "complete", "partial", "aggregate_only", "compatibility"):
            count = forward_support[state]
            if count:
                print(f"  {state:<17} kernels={count:>4}")
        if forward_unsupported:
            print(f"  {'unsupported':<17} kernels={forward_unsupported:>4}")
        unsupported_cases = Counter(
            op_meta[key]["op_type"] for key, pair in op_pairs_forward.items()
            if pair == "ERR")
        if unsupported_cases:
            unsupported_reasons = Counter()
            for key, pair in op_pairs_forward.items():
                if pair != "ERR":
                    continue
                unsupported_reasons.update({
                    result["forward_unavailable_reason"]
                    for result in kernel_results
                    if result["task_key"] == key
                    and "forward_unavailable_reason" in result})
            print(f"{'═' * sep_len}")
            print("  Forward unsupported cases (explicit, excluded from scoring)")
            print(f"{'═' * sep_len}")
            print(f"  {'ALL':<28} cases={sum(unsupported_cases.values()):>5}"
                  f" / {len(op_pairs_forward)}")
            for op_type, count in sorted(unsupported_cases.items()):
                print(f"  {op_type:<28} cases={count:>5}")
            for reason, count in unsupported_reasons.most_common():
                print(f"    reason cases={count:>5}: {reason}")

    if run_oracle:
        kernel_coverage = Counter(
            result["oracle_coverage"] for result in kernel_results
            if "oracle_coverage" in result)
        print(f"{'═' * sep_len}")
        print("  Formula Ceiling evidence coverage")
        print(f"{'═' * sep_len}")
        for state in ("complete", "partial"):
            task_count = oracle_task_coverage[state]
            kernel_count = kernel_coverage[state]
            if task_count == 0 and kernel_count == 0:
                continue
            state_keys = [
                key for key, meta in op_meta.items()
                if ("oracle_coverage" in meta
                    and meta["oracle_coverage"] == state
                    and key in op_pairs_oracle)
            ]
            state_metrics = compute_metrics(
                [op_pairs_oracle[key] for key in state_keys])
            if state_metrics is None:
                metric_text = "no scored tasks"
            elif state_metrics["status"] == "unsupported":
                metric_text = "unsupported"
            elif state_metrics["status"] == "ok":
                metric_text = (
                    f"MAPE={state_metrics['MAPE']:.1f}% "
                    f"Med={state_metrics['MedAPE']:.1f}% "
                    f"WMAE={state_metrics['WMAE']:.1f}%")
            else:
                raise ValueError(
                    "unknown coverage metric status: "
                    f"{state_metrics['status']!r}")
            print(
                f"  {state:<17} tasks={task_count:>4} "
                f"kernels={kernel_count:>4}  {metric_text}")
        unavailable_counts = Counter()
        for result in kernel_results:
            if "oracle_unavailable_fields" not in result:
                continue
            for path in result["oracle_unavailable_fields"]:
                unavailable_counts[path] += 1
        if unavailable_counts:
            unavailable_text = ", ".join(
                f"{path}={count}"
                for path, count in unavailable_counts.most_common(8))
            print(f"  unavailable fields: {unavailable_text}")

    # Per-kernel bottleneck distribution
    if run_oracle:
        print(f"{'═' * sep_len}")
        print(f"  Per-KERNEL bottleneck distribution")
        print(f"{'═' * sep_len}")
        for ot in [
                "GEMM", "FUSEDMOE", "UNFUSEDMOE",
                "FLASHATTN", "EAGERATTN"]:
            for cat, filt in [("tensor", False), ("non-tensor", True)]:
                ks = [kr for kr in kernel_results if kr["op_type"] == ot
                      and "is_nt" in kr and kr["is_nt"] == filt]
                if not ks:
                    continue
                bn = Counter(kr["bottleneck"] for kr in ks)
                bn_items = bn.most_common()
                dist = "  ".join(f"{k}={v}" for k, v in bn_items)
                label = f"  {ot} {cat}"
                print(f"  {label:<24} n={len(ks):>4}  {dist}")

    if report_output is not None:
        from evaluation.reporting.kernel_overview import write_kernel_overview
        content = Path(hw_config_path).read_text()
        snapshot = {"kind": "hardware_config", "path": str(Path(hw_config_path).resolve()),
            "content": content, "sha256": hashlib.sha256(content.encode()).hexdigest()}
        pairs = {"oracle": op_pairs_oracle, "forward": op_pairs_forward,
            "roofline": op_pairs_roofline, "pipeweave": op_pairs_pipeweave}
        metrics = {method: compute_metrics(list(values.values()), exclude_unsupported=method in ("forward", "roofline"))
            for method, values in pairs.items()}
        receipt = write_kernel_overview(semantic_artifact, metrics, {**vars(args), "effective_jobs": n_workers},
            report_output, elapsed_s=time.perf_counter() - started, hardware_snapshot=snapshot)
        print("Kernel HTML report: " + receipt["path"])
        print("Kernel output cost receipt: " + receipt["cost_receipt_path"])
