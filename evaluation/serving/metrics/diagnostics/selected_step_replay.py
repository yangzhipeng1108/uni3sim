#!/usr/bin/env python3
"""Derive a full prediction fingerprint for one injected Scheduler step."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]

from evaluation.serving.metrics.diagnostics.selected_step_injection import (  # noqa: E402
    build_selected_step_injection_replay,
    write_json,
)
from evaluation.serving.metrics.diagnostics.ascend_910b_host_prepare import (  # noqa: E402
    ascend_910b_unquant_mlp_wrapper_events,
    ascend_910b_model_runner_outer_events,
    ascend_910b_rotary_cache_update_events,
    ascend_910b_moe_wrapper_control_events,
    ascend_910b_fia_source_wrapper_events,
    ascend_910b_qkv_triton_composite_events,
    ascend_910b_runtime_prepare_events,
)
from evaluation.serving.metrics.diagnostics.ascend_910b_compiled_fx import (  # noqa: E402
    ascend_910b_compiled_fx_backend_events,
    ascend_910b_compiled_fx_reshape_events,
)
from src.serving.host_timing import (  # noqa: E402
    HostRuntimePrepareProfile,
    load_host_launch_profile,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ascend_runtime_prepare_candidates(
    path: Path,
    collective_summary_dir: Path | None = None,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (payload["schema_version"] not in (4, 6, 7, 8, 9, 10, 11, 12)
            or payload["validation"] != "pass"):
        raise ValueError(
            "Ascend dynamic prepare summary must be a validated schema-v4 "
            "through schema-v12 run")
    generic_ns = payload["parameter"]["wall_value"]
    fit_by_kind = {
        fit["operation_kind"]: fit for fit in payload["fits"]
    }
    required_kinds = {
        "ascend_moe_gating_topk",
        "ascend_moe_init_routing",
        "ascend_reshape_and_cache",
        "routing_index_mask_mul",
    }
    if set(fit_by_kind) < required_kinds:
        raise ValueError("Ascend runtime prepare summary lacks required fits")
    service_ns_by_mechanism = {
        "moe_gating_prepare": (
            fit_by_kind["ascend_moe_gating_topk"]
            ["wall_slope_ns_per_call"] - generic_ns
        ),
        "moe_token_dispatch_prepare": (
            fit_by_kind["routing_index_mask_mul"]
            ["wall_slope_ns_per_call"] - 3.0 * generic_ns
            + fit_by_kind["ascend_moe_init_routing"]
            ["wall_slope_ns_per_call"] - generic_ns
        ),
        "reshape_cache_prepare": (
            fit_by_kind["ascend_reshape_and_cache"]
            ["wall_slope_ns_per_call"] - generic_ns
        ),
    }
    if payload["schema_version"] in (6, 7, 8, 9, 10, 11, 12):
        triton_candidate = payload[
            "triton_dynamic_argument_prepare_candidate"]
        if triton_candidate["holdout_dynamic_argument_count"] != 11:
            raise ValueError(
                "Ascend Triton argument holdout must cover 11 dynamic args")
        triton_total_ns = triton_candidate[
            "holdout_wall_predicted_ns_per_launch"]
        service_ns_by_mechanism["triton_dynamic_argument_prepare"] = (
            triton_total_ns - generic_ns
        )
    if payload["schema_version"] in (8, 9, 10, 11, 12):
        moe_mlp_candidates = payload["moe_mlp_runtime_prepare_candidates"]
        operation_by_mechanism = {
            "moe_grouped_gemm_prepare": "ascend_moe_grouped_matmul",
            "moe_swiglu_prepare": "ascend_moe_swiglu",
            "moe_token_combine_prepare": "ascend_moe_token_unpermute",
        }
        if set(moe_mlp_candidates) != set(operation_by_mechanism.values()):
            raise ValueError("Ascend MoE MLP prepare candidates are incomplete")
        for mechanism, operation_kind in operation_by_mechanism.items():
            candidate = moe_mlp_candidates[operation_kind]
            if (candidate["status"]
                    != "diagnostic_only_requires_source_occurrence_metadata"):
                raise ValueError("Ascend MoE MLP candidate status is invalid")
            if abs(candidate["holdout_wall_signed_error_percent"]) > 5.0:
                raise ValueError("Ascend MoE MLP holdout error exceeds 5 percent")
            increment_ns = candidate["wall_increment_over_generic_ns_per_call"]
            if (isinstance(increment_ns, bool)
                    or not isinstance(increment_ns, (int, float))
                    or increment_ns <= 0):
                raise ValueError("Ascend MoE MLP prepare increment is invalid")
            service_ns_by_mechanism[mechanism] = float(increment_ns)
    if payload["schema_version"] in (11, 12):
        candidate = payload["unquantized_gemm_runtime_prepare_candidate"]
        if (candidate["status"]
                != "diagnostic_only_requires_source_occurrence_metadata"):
            raise ValueError("Ascend unquantized-GEMM candidate status is invalid")
        if candidate["operation_kind"] != "ascend_unquantized_gemm":
            raise ValueError("Ascend unquantized-GEMM operation identity is invalid")
        if abs(candidate["holdout_wall_signed_error_percent"]) > 5.0:
            raise ValueError(
                "Ascend unquantized-GEMM holdout error exceeds 5 percent")
        increment_ns = candidate["wall_increment_over_generic_ns_per_call"]
        if (isinstance(increment_ns, bool)
                or not isinstance(increment_ns, (int, float))
                or increment_ns <= 0):
            raise ValueError("Ascend unquantized-GEMM prepare increment is invalid")
        service_ns_by_mechanism["unquantized_gemm_prepare"] = float(
            increment_ns)
    if payload["schema_version"] in (9, 10, 11, 12):
        stream_event_candidate = payload[
            "stream_event_runtime_prepare_candidate"]
        if (stream_event_candidate["status"]
                != "diagnostic_only_requires_source_occurrence_metadata"):
            raise ValueError("Ascend stream-event candidate status is invalid")
        if (stream_event_candidate["operation_kind"]
                != "ascend_stream_record_event"):
            raise ValueError("Ascend stream-event operation identity is invalid")
        if abs(stream_event_candidate["holdout_wall_signed_error_percent"]) > 5.0:
            raise ValueError("Ascend stream-event holdout error exceeds 5 percent")
        total_ns = stream_event_candidate["wall_total_ns_per_event"]
        if (isinstance(total_ns, bool)
                or not isinstance(total_ns, (int, float))
                or total_ns <= 0):
            raise ValueError("Ascend stream-event service is invalid")
        # Event record is an additional stream command, rather than the
        # planned kernel submit at the event's source anchor.  Retain its
        # total Host service instead of subtracting that anchor's C_submit.
        service_ns_by_mechanism["stream_record_event_prepare"] = float(
            total_ns)
    if payload["schema_version"] == 12:
        candidate = payload["rotary_cache_update_runtime_prepare_candidate"]
        if (candidate["status"]
                != "diagnostic_only_requires_source_occurrence_metadata"):
            raise ValueError("Ascend rotary-cache candidate status is invalid")
        if candidate["operation_kind"] != "ascend_rotary_cache_update":
            raise ValueError("Ascend rotary-cache operation identity is invalid")
        if abs(candidate["holdout_wall_signed_error_percent"]) > 5.0:
            raise ValueError("Ascend rotary-cache holdout error exceeds 5 percent")
        total_ns = candidate["wall_slope_ns_per_call"]
        if (isinstance(total_ns, bool)
                or not isinstance(total_ns, (int, float))
                or not math.isfinite(total_ns)
                or total_ns <= 0):
            raise ValueError("Ascend rotary-cache service is invalid")
        # The compact plan has no individual Device children for this runtime
        # cache update.  Retain the full independent helper service rather
        # than subtracting a submit that is not otherwise scheduled.
        service_ns_by_mechanism["rotary_cache_update_prepare"] = float(
            total_ns)
    collective_identity = None
    if collective_summary_dir is not None:
        collective_root = collective_summary_dir.resolve(strict=True)
        summary_paths = tuple(sorted(collective_root.glob("summary-rank-*.json")))
        if len(summary_paths) != 2:
            raise ValueError(
                "Ascend collective prepare input requires two rank summaries")
        collective_values = {
            "all_reduce": [],
            "all_gather": [],
        }
        collective_inputs = []
        observed_ranks = set()
        for summary_path in summary_paths:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary["schema_version"] != 2:
                raise ValueError(
                    "Ascend collective prepare summary must use schema-v2")
            if summary["call_stack"] != "vllm_tensor_parallel":
                raise ValueError(
                    "Ascend collective prepare summary must use the vLLM "
                    "tensor-parallel call stack")
            rank = summary["rank"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise TypeError("Ascend collective summary rank is invalid")
            if rank in observed_ranks:
                raise ValueError("Ascend collective summary ranks are duplicated")
            observed_ranks.add(rank)
            for fit in summary["fits"]:
                operation = fit["operation"]
                if operation not in collective_values:
                    raise ValueError(
                        f"unsupported collective operation: {operation!r}")
                value_ns = fit["wall_slope_ns_per_call"]
                if (isinstance(value_ns, bool)
                        or not isinstance(value_ns, (int, float))
                        or value_ns <= 0):
                    raise ValueError(
                        "collective Host-submit slope must be positive")
                collective_values[operation].append(float(value_ns))
            collective_inputs.append({
                "path": str(summary_path),
                "sha256": _sha256(summary_path),
                "rank": rank,
            })
        if observed_ranks != {0, 1}:
            raise ValueError(
                "Ascend collective summaries must cover ranks zero and one")
        collective_total_ns = {
            operation: statistics.median(values)
            for operation, values in collective_values.items()
        }
        service_ns_by_mechanism.update({
            "collective_all_reduce_prepare": (
                collective_total_ns["all_reduce"] - generic_ns),
            "collective_all_gather_prepare": (
                collective_total_ns["all_gather"] - generic_ns),
        })
        collective_identity = {
            "status": "diagnostic_only_not_production_eligible",
            "inputs": collective_inputs,
            "aggregation": (
                "median_across_rank_and_independent_message_size_sweeps"),
            "total_service_ns_by_operation": collective_total_ns,
        }
    if any(
            isinstance(value_ns, bool)
            or not isinstance(value_ns, (int, float))
            or value_ns <= 0
            for value_ns in service_ns_by_mechanism.values()):
        raise ValueError("Ascend runtime prepare increments must be positive")
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        tuple(
            (mechanism, float(value_ns) / 1e9)
            for mechanism, value_ns in service_ns_by_mechanism.items()
        ),
    )
    identity = {
        "status": "diagnostic_only_not_production_eligible",
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "mechanisms": {
            mechanism: {"service_time_ns": float(value_ns)}
            for mechanism, value_ns in service_ns_by_mechanism.items()
        },
        "measurement_boundary": (
            "source-localized wrapper work before the planned async submit; "
            "missing routing helper submits remain excluded"),
        "aggregation": "independent_count_sweep_by_source_mechanism",
    }
    if payload["schema_version"] in (6, 7, 8, 9, 10, 11):
        identity["triton_source_launch_descriptor"] = {
            "kernel": "split_qkv_rmsnorm_rope_kernel",
            "dynamic_pointer_arguments": 10,
            "dynamic_scalar_arguments": 1,
            "constexpr_arguments": 16,
            "modeled_dynamic_argument_count": 11,
        }
    if collective_identity is not None:
        identity["collective_prepare"] = collective_identity
    return profile, identity


def _load_ascend_compiled_fx_prepare_candidate(
    path: Path,
    host_launch_profile,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Bind a generic compiled-FX entry slope to its source descriptor."""

    resolved, payload, slope_ns, holdout_error = (
        _read_ascend_compiled_fx_measurement(path))
    total_service_s = slope_ns / 1e9
    incremental_service_s = (
        total_service_s - host_launch_profile.eager_enqueue_s)
    if incremental_service_s <= 0:
        raise ValueError(
            "compiled-FX service must exceed separately modeled eager submit")
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        (("compiled_fx_backend_entry", incremental_service_s),),
    )
    return profile, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "total_service_ns_per_entry": slope_ns,
        "subtracted_eager_submit_ns": (
            host_launch_profile.eager_enqueue_s * 1e9),
        "incremental_service_ns_per_entry": incremental_service_s * 1e9,
        "holdout_wall_signed_error_percent": holdout_error,
        "source_descriptor": (
            "ascend_910b_compiled_fx_backend_events"),
    }


def _read_ascend_compiled_fx_measurement(
    path: Path,
) -> tuple[Path, dict[str, object], float, float]:
    """Read one independently timed compiled-FX count sweep."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("compiled-FX summary must use schema-v1")
    if payload["hardware_id"] != "ascend_910b":
        raise ValueError("compiled-FX summary must target Ascend 910B")
    fit = payload["fit"]
    if (not isinstance(fit, dict)
            or set(fit) != {
                "wall_intercept_ns", "wall_slope_ns_per_node",
                "thread_cpu_intercept_ns", "thread_cpu_slope_ns_per_node",
                "holdout_wall_measured_ns", "holdout_wall_predicted_ns",
                "holdout_wall_signed_error_percent",
                "holdout_thread_cpu_measured_ns",
                "holdout_thread_cpu_predicted_ns",
                "holdout_thread_cpu_signed_error_percent",
            }):
        raise ValueError("compiled-FX summary fit fields are invalid")
    slope_ns = fit["wall_slope_ns_per_node"]
    holdout_error = fit["holdout_wall_signed_error_percent"]
    if (isinstance(slope_ns, bool)
            or not isinstance(slope_ns, (int, float))
            or slope_ns <= 0):
        raise ValueError("compiled-FX summary slope must be positive")
    if (isinstance(holdout_error, bool)
            or not isinstance(holdout_error, (int, float))
            or abs(holdout_error) > 5.0):
        raise ValueError("compiled-FX summary holdout error exceeds 5 percent")
    return resolved, payload, float(slope_ns), float(holdout_error)


def _load_ascend_compiled_fx_lane_prepare_candidate(
    lane_paths: tuple[tuple[int, Path], ...],
    host_launch_profile,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Bind a concurrent TP2 sweep to exact physical Host lanes.

    The count sweep owns no model/operator duration.  Its only purpose is to
    quantify the compiled-FX executor's per-entry Host service under two
    simultaneously active TP ranks.  A caller must provide both ranks so a
    partially measured lane cannot silently fall back to a generic value.
    """

    if len(lane_paths) != 2 or {rank for rank, _ in lane_paths} != {0, 1}:
        raise ValueError(
            "compiled-FX lane timing requires exactly rank-0 and rank-1 summaries")
    lane_entries = []
    lane_identity = []
    source_identity = None
    for rank, path in sorted(lane_paths):
        resolved, payload, slope_ns, holdout_error = (
            _read_ascend_compiled_fx_measurement(path))
        environment = payload["environment"]
        if (not isinstance(environment, dict)
                or environment["device_id"] != rank):
            raise ValueError(
                "compiled-FX lane summary device_id disagrees with its rank")
        source = payload["source"]
        if (not isinstance(source, dict)
                or set(source) != {"baseline_commit", "path", "sha256"}):
            raise ValueError("compiled-FX lane summary source identity is invalid")
        current_source_identity = (
            source["baseline_commit"], source["sha256"])
        if source_identity is None:
            source_identity = current_source_identity
        elif source_identity != current_source_identity:
            raise ValueError(
                "compiled-FX lane summaries must use one runner source")
        incremental_service_s = (
            slope_ns / 1e9 - host_launch_profile.eager_enqueue_s)
        if incremental_service_s <= 0:
            raise ValueError(
                "compiled-FX lane service must exceed eager submit")
        lane_entries.append((
            "main", rank, "compiled_fx_backend_entry", incremental_service_s))
        lane_identity.append({
            "pool_name": "main",
            "rank": rank,
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "total_service_ns_per_entry": slope_ns,
            "subtracted_eager_submit_ns": (
                host_launch_profile.eager_enqueue_s * 1e9),
            "incremental_service_ns_per_entry": incremental_service_s * 1e9,
            "holdout_wall_signed_error_percent": holdout_error,
        })
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        (),
        tuple(lane_entries),
    )
    return profile, {
        "measurement": "TP2-concurrent compiled-FX executor count sweep",
        "lane_services": lane_identity,
        "runner_source": {
            "baseline_commit": source_identity[0],
            "sha256": source_identity[1],
        },
        "source_descriptor": (
            "ascend_910b_compiled_fx_backend_events"),
    }


def _read_ascend_compiled_fx_reshape_measurement(
    path: Path,
    rank: int,
) -> tuple[Path, dict[str, object], float, float]:
    """Read one lane's independent ``aten.reshape`` metadata-node sweep."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if (payload["schema_version"] != 1
            or payload["kind"] != "ascend_compiled_fx_metadata_microbenchmark"
            or payload["hardware_id"] != "ascend_910b"):
        raise ValueError("compiled-FX reshape summary identity is invalid")
    environment = payload["environment"]
    if (not isinstance(environment, dict)
            or environment["device_id"] != rank):
        raise ValueError("compiled-FX reshape summary device disagrees with rank")
    fits = payload["fit_by_mechanism"]
    if not isinstance(fits, dict):
        raise TypeError("compiled-FX reshape summary fits must be an object")
    fit = fits["fx_aten_reshape_metadata"]
    expected_keys = {
        "wall_intercept_ns",
        "wall_slope_ns_per_node",
        "holdout_wall_measured_ns",
        "holdout_wall_predicted_ns",
        "holdout_wall_signed_error_percent",
    }
    if not isinstance(fit, dict) or set(fit) != expected_keys:
        raise ValueError("compiled-FX reshape fit fields are invalid")
    slope_ns = fit["wall_slope_ns_per_node"]
    holdout_error = fit["holdout_wall_signed_error_percent"]
    if (isinstance(slope_ns, bool)
            or not isinstance(slope_ns, (int, float))
            or not math.isfinite(slope_ns)
            or slope_ns <= 0):
        raise ValueError("compiled-FX reshape slope must be finite and positive")
    if (isinstance(holdout_error, bool)
            or not isinstance(holdout_error, (int, float))
            or not math.isfinite(holdout_error)
            or abs(holdout_error) > 5.0):
        raise ValueError("compiled-FX reshape holdout error exceeds 5 percent")
    return resolved, payload, float(slope_ns), float(holdout_error)


def _load_ascend_compiled_fx_reshape_lane_prepare_candidate(
    lane_paths: tuple[tuple[int, Path], ...],
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Bind measured reshape metadata to the two exact TP2 Host lanes.

    This is direct generated-Python work in the compiled FX GraphModule, not
    an additional Device submit, so its independently measured service is not
    reduced by ``C_submit``.
    """

    if len(lane_paths) != 2 or {rank for rank, _ in lane_paths} != {0, 1}:
        raise ValueError(
            "compiled-FX reshape timing requires rank-0 and rank-1 summaries")
    lane_entries = []
    lane_identity = []
    source_identity = None
    for rank, path in sorted(lane_paths):
        resolved, payload, slope_ns, holdout_error = (
            _read_ascend_compiled_fx_reshape_measurement(path, rank))
        runner = payload["runner"]
        if (not isinstance(runner, dict)
                or set(runner) != {"path", "sha256"}):
            raise ValueError("compiled-FX reshape runner identity is invalid")
        current_source_identity = (payload["baseline_commit"], runner["sha256"])
        if source_identity is None:
            source_identity = current_source_identity
        elif source_identity != current_source_identity:
            raise ValueError(
                "compiled-FX reshape lane summaries use different runners")
        lane_entries.append((
            "main", rank, "compiled_fx_aten_reshape_metadata", slope_ns / 1e9))
        lane_identity.append({
            "pool_name": "main",
            "rank": rank,
            "path": str(resolved),
            "sha256": _sha256(resolved),
            "service_ns_per_reshape": slope_ns,
            "holdout_wall_signed_error_percent": holdout_error,
        })
    profile = HostRuntimePrepareProfile("ascend_910b", (), tuple(lane_entries))
    return profile, {
        "measurement": "TP2-concurrent compiled-FX reshape metadata sweep",
        "lane_services": lane_identity,
        "runner_source": {
            "baseline_commit": source_identity[0],
            "sha256": source_identity[1],
        },
        "source_descriptor": "ascend_910b_compiled_fx_reshape_events",
        "submit_overlap": "none; direct generated-Python metadata work",
    }


def _load_ascend_moe_wrapper_control_candidate(
    path: Path,
    compiled_fx_identity: dict[str, object],
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Derive the non-overlapping MoE wrapper increment at an FX node."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1 or payload["status"] != "pass":
        raise ValueError("MoE wrapper composite summary must be a passing schema-v1 run")
    composite = payload["compiled_wrapper_composite"]
    if not isinstance(composite, dict):
        raise ValueError("MoE wrapper compiled composite must be an object")
    predeclared = composite["predeclared"]
    if predeclared != {
            "chain_lengths": [1, 8, 64, 512],
            "fit_chain_lengths": [1, 8, 64],
            "holdout_chain_length": 512,
    }:
        raise ValueError("MoE wrapper compiled composite count sweep differs")
    total_service_ns = composite["wall_slope_ns_per_composite_call"]
    holdout_error = composite["holdout_wall_signed_error_percent"]
    if (isinstance(total_service_ns, bool)
            or not isinstance(total_service_ns, (int, float))
            or total_service_ns <= 0):
        raise ValueError("MoE wrapper compiled composite slope must be positive")
    if (isinstance(holdout_error, bool)
            or not isinstance(holdout_error, (int, float))
            or abs(holdout_error) > 5.0):
        raise ValueError("MoE wrapper compiled composite holdout exceeds 5 percent")
    generic_service_ns = (
        compiled_fx_identity[
            "wrapper_control_reference_total_service_ns_per_entry"]
        if "wrapper_control_reference_total_service_ns_per_entry"
        in compiled_fx_identity
        else compiled_fx_identity["total_service_ns_per_entry"]
    )
    if (isinstance(generic_service_ns, bool)
            or not isinstance(generic_service_ns, (int, float))
            or generic_service_ns <= 0):
        raise ValueError("compiled-FX generic service is invalid")
    incremental_service_ns = float(total_service_ns) - float(generic_service_ns)
    if incremental_service_ns <= 0:
        raise ValueError("MoE wrapper adds no service over a generic FX entry")
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        (("compiled_fx_moe_wrapper_control", incremental_service_ns / 1e9),),
    )
    return profile, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "source_descriptor": "ascend_910b_moe_wrapper_control_events",
        "total_service_ns_per_moe_wrapper": float(total_service_ns),
        "subtracted_generic_fx_service_ns": float(generic_service_ns),
        "incremental_service_ns_per_moe_wrapper": incremental_service_ns,
        "holdout_wall_signed_error_percent": float(holdout_error),
    }


def _load_ascend_unquant_mlp_control_candidate(
    path: Path,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Load the source-only control time around unquantized MoE MLP children."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1 or payload["status"] != "pass":
        raise ValueError("MoE MLP control summary must be a passing schema-v1 run")
    fits = payload["fits"]
    if not isinstance(fits, list):
        raise TypeError("MoE MLP control summary fits must be a list")
    fits_by_measurement = {}
    for fit in fits:
        measurement = fit["measurement"]
        if measurement in fits_by_measurement:
            raise ValueError("MoE MLP control measurement is duplicated")
        fits_by_measurement[measurement] = fit
    if "unified_unquant_mlp_control" not in fits_by_measurement:
        raise ValueError("MoE MLP control summary lacks the unquantized path")
    fit = fits_by_measurement["unified_unquant_mlp_control"]
    expected_keys = {
        "cells",
        "holdout_wall_measured_ns",
        "holdout_wall_predicted_ns",
        "holdout_wall_signed_error_percent",
        "measurement",
        "wall_fixed_intercept_ns",
        "wall_slope_ns_per_call",
    }
    if set(fit) != expected_keys:
        raise ValueError("MoE MLP control fit fields are invalid")
    cells = fit["cells"]
    if (not isinstance(cells, list)
            or [cell["chain_length"] for cell in cells]
            != [1, 8, 64, 512, 2048]
            or any(cell["sample_count"] != 8 for cell in cells)):
        raise ValueError("MoE MLP control count sweep differs")
    total_service_ns = fit["wall_slope_ns_per_call"]
    holdout_error = fit["holdout_wall_signed_error_percent"]
    if (isinstance(total_service_ns, bool)
            or not isinstance(total_service_ns, (int, float))
            or not math.isfinite(total_service_ns)
            or total_service_ns <= 0):
        raise ValueError("MoE MLP control slope must be finite and positive")
    if (isinstance(holdout_error, bool)
            or not isinstance(holdout_error, (int, float))
            or not math.isfinite(holdout_error)
            or abs(holdout_error) > 5.0):
        raise ValueError("MoE MLP control holdout exceeds 5 percent")
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        (("moe_unquant_mlp_wrapper_prepare", total_service_ns / 1e9),),
    )
    return profile, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "source_descriptor": "ascend_910b_unquant_mlp_wrapper_events",
        "total_service_ns_per_moe_mlp": float(total_service_ns),
        "subtracted_child_service_ns": 0.0,
        "incremental_service_ns_per_moe_mlp": float(total_service_ns),
        "holdout_wall_signed_error_percent": float(holdout_error),
    }


def _load_ascend_fused_attention_prepare_candidate(
    path: Path,
    host_launch_profile,
    request_count: int,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Bind synthetic FIA descriptor work to a source-authored call site."""

    if (isinstance(request_count, bool)
            or not isinstance(request_count, int)
            or request_count < 1):
        raise ValueError("FIA request count must be a positive integer")
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    schema_version = payload["schema_version"]
    if schema_version not in (1, 3):
        raise ValueError("FIA prepare summary must use schema-v1 or schema-v3")
    if payload["hardware_id"] != "ascend_910b":
        raise ValueError("FIA prepare summary must target Ascend 910B")
    if schema_version == 1:
        fit = payload["request_metadata_fit"]
        mechanism = "python_argument_prepare"
        source_descriptor = "python_argument_prepare"
    else:
        fit = payload["source_static_wrapper_request_fit"]
        mechanism = "fused_attention_source_wrapper_prepare"
        source_descriptor = "ascend_910b_fia_source_wrapper_events"
    if not isinstance(fit, dict) or set(fit) != {
            "wall_intercept_ns", "wall_slope_ns_per_request",
            "thread_cpu_intercept_ns", "thread_cpu_slope_ns_per_request",
            "holdout_wall_measured_ns_per_call",
            "holdout_wall_predicted_ns_per_call",
            "holdout_wall_signed_error_percent",
            "holdout_thread_cpu_measured_ns_per_call",
            "holdout_thread_cpu_predicted_ns_per_call",
            "holdout_thread_cpu_signed_error_percent",
    }:
        raise ValueError("FIA prepare summary request fit fields are invalid")
    wall_intercept_ns = fit["wall_intercept_ns"]
    wall_slope_ns = fit["wall_slope_ns_per_request"]
    holdout_error = fit["holdout_wall_signed_error_percent"]
    for value, label in (
            (wall_intercept_ns, "FIA wall intercept"),
            (wall_slope_ns, "FIA wall request slope")):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0):
            raise ValueError(f"{label} must be finite and positive")
    if (isinstance(holdout_error, bool)
            or not isinstance(holdout_error, (int, float))
            or not math.isfinite(holdout_error)
            or abs(holdout_error) > 5.0):
        raise ValueError("FIA prepare summary holdout error exceeds 5 percent")
    total_service_ns = (
        float(wall_intercept_ns) + float(wall_slope_ns) * request_count)
    incremental_service_ns = (
        total_service_ns - host_launch_profile.eager_enqueue_s * 1e9)
    if incremental_service_ns <= 0:
        raise ValueError("FIA service must exceed separately modeled eager submit")
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        ((mechanism, incremental_service_ns / 1e9),),
    )
    return profile, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "source_descriptor": source_descriptor,
        "request_count": request_count,
        "total_service_ns_per_call": total_service_ns,
        "subtracted_eager_submit_ns": (
            host_launch_profile.eager_enqueue_s * 1e9),
        "incremental_service_ns_per_call": incremental_service_ns,
        "holdout_wall_signed_error_percent": float(holdout_error),
    }


def _load_ascend_qkv_triton_composite_candidate(
    path: Path,
    host_launch_profile,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Load a generic custom-op composite, excluding the separately modeled submit."""

    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload["schema_version"] != 2 or payload["validation"] != "pass":
        raise ValueError("QKV Triton composite summary must be validated schema-v2")
    if payload["status"] != "diagnostic_only_requires_source_static_descriptor":
        raise ValueError("QKV Triton composite summary status is invalid")
    descriptor = payload["source_static_descriptor"]
    expected_descriptor = {
        "source_kernel": "split_qkv_rmsnorm_rope_kernel",
        "dynamic_argument_count": 11,
        "constexpr_argument_count": 16,
    }
    if descriptor != expected_descriptor:
        raise ValueError("QKV Triton composite source descriptor is invalid")
    candidate = payload["custom_op_three_output_triton_composite_fit"]
    for key in (
            "wall_slope_ns_per_launch",
            "holdout_wall_measured_ns",
            "holdout_wall_predicted_ns",
    ):
        value = candidate[key]
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0):
            raise ValueError(f"QKV Triton composite {key} is invalid")
    holdout_error_percent = 100.0 * (
        candidate["holdout_wall_predicted_ns"]
        / candidate["holdout_wall_measured_ns"] - 1.0)
    if abs(holdout_error_percent) > 5.0:
        raise ValueError("QKV Triton composite holdout error exceeds 5 percent")
    total_service_ns = float(candidate["wall_slope_ns_per_launch"])
    incremental_service_ns = (
        total_service_ns - host_launch_profile.eager_enqueue_s * 1e9)
    if incremental_service_ns <= 0:
        raise ValueError("QKV composite must exceed separately modeled eager submit")
    profile = HostRuntimePrepareProfile(
        "ascend_910b",
        (("triton_three_output_composite_prepare",
          incremental_service_ns / 1e9),),
    )
    return profile, {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "source_descriptor": descriptor,
        "total_service_ns_per_call": total_service_ns,
        "subtracted_eager_submit_ns": host_launch_profile.eager_enqueue_s * 1e9,
        "incremental_service_ns_per_call": incremental_service_ns,
        "holdout_wall_signed_error_percent": holdout_error_percent,
    }


def _load_ascend_model_runner_outer_diagnostic(
    framework_timing_path: Path,
    event_boundary_path: Path,
    *,
    summarize_model_runner_outer_timing,
) -> tuple[HostRuntimePrepareProfile, dict[str, object]]:
    """Expose selected-step source timing only to an explicit diagnostic replay.

    This input is intentionally not an independent microbenchmark and hence
    never becomes a Forward parameter.  Its purpose is to quantify the
    maximum improvement worth pursuing before the corresponding mechanism
    sweeps are run on an available device.
    """

    summary = summarize_model_runner_outer_timing(
        framework_timing_path.resolve(strict=True),
        event_boundary_path.resolve(strict=True),
    )
    if summary["production_forward_eligible"] is not False:
        raise ValueError(
            "selected-step ModelRunner timing must remain diagnostic-only")
    components = summary["model_runner_outer_components_ns"]
    if not isinstance(components, dict) or set(components) != {
            "input_prepare_ns",
            "logits_prepare_ns",
            "forward_metadata_ns",
            "model_runner_uninstrumented_control_ns",
    }:
        raise ValueError("ModelRunner outer timing components are invalid")
    mechanism_names = {
        "model_runner_input_prepare": "input_prepare_ns",
        "model_runner_forward_metadata_prepare": "forward_metadata_ns",
        "model_runner_logits_prepare": "logits_prepare_ns",
    }
    services = []
    for mechanism, field in mechanism_names.items():
        value = components[field]
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0):
            raise ValueError(
                f"ModelRunner outer source timing is invalid: {field}")
        services.append((mechanism, float(value) / 1e9))
    trace_step_id = summary["trace_step_id"]
    if not isinstance(trace_step_id, str) or not trace_step_id:
        raise ValueError("ModelRunner outer timing trace step ID is invalid")
    worker_rank = summary["worker_rank"]
    if (isinstance(worker_rank, bool) or not isinstance(worker_rank, int)
            or worker_rank < 0):
        raise ValueError("ModelRunner outer timing worker rank is invalid")
    return HostRuntimePrepareProfile("ascend_910b", tuple(services)), {
        "status": "diagnostic_only_not_production_eligible",
        "source_descriptor": "ascend_910b_model_runner_outer_events",
        "trace_step_id": trace_step_id,
        "worker_rank": worker_rank,
        "source_timing": summary,
        "excluded_component": {
            "model_runner_uninstrumented_control_ns": components[
                "model_runner_uninstrumented_control_ns"],
            "reason": (
                "the aggregate source probe has no source-static position for "
                "this residual control interval"),
        },
    }


def _compose_runtime_prepare_profiles(
    profiles: tuple[HostRuntimePrepareProfile, ...],
) -> HostRuntimePrepareProfile:
    """Merge separately measured mechanisms without silently overwriting one."""

    if not profiles:
        raise ValueError("runtime prepare composition requires profiles")
    hardware_ids = {profile.hardware_id for profile in profiles}
    if len(hardware_ids) != 1:
        raise ValueError("runtime prepare composition spans hardware")
    entries = tuple(
        entry
        for profile in profiles
        for entry in profile.service_time_s_by_mechanism
    )
    lane_entries = tuple(
        entry
        for profile in profiles
        for entry in profile.lane_service_time_s_by_mechanism
    )
    names = tuple(mechanism for mechanism, _ in entries)
    if len(set(names)) != len(names):
        raise ValueError("runtime prepare composition repeats a mechanism")
    lane_keys = tuple(
        (pool_name, rank, mechanism)
        for pool_name, rank, mechanism, _ in lane_entries
    )
    if len(set(lane_keys)) != len(lane_keys):
        raise ValueError("runtime prepare composition repeats a lane mechanism")
    return HostRuntimePrepareProfile(
        profiles[0].hardware_id,
        entries,
        lane_entries,
    )


def _compose_runtime_prepare_event_builders(*builders):
    """Append independent event maps and fail closed on duplicate mechanisms."""

    if not builders:
        raise ValueError("runtime prepare composition requires event builders")

    def build(plan):
        events = {}
        for builder in builders:
            candidate_events = builder(plan)
            for key, candidate_mechanisms in candidate_events.items():
                combined = (
                    (*events[key], *candidate_mechanisms)
                    if key in events else tuple(candidate_mechanisms))
                names = tuple(event.mechanism for event in combined)
                if len(set(names)) != len(names):
                    raise ValueError(
                        "runtime prepare composition repeats an event "
                        f"mechanism at {key!r}")
                events[key] = combined
        return events

    return build


def _parse_compiled_fx_lane_summary(value: str) -> tuple[int, Path]:
    """Parse one exact ``rank:path`` compiled-FX timing input."""

    rank_text, separator, path_text = value.partition(":")
    if not separator or not rank_text or not path_text:
        raise argparse.ArgumentTypeError(
            "compiled-FX lane summary must be written as rank:path")
    try:
        rank = int(rank_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "compiled-FX lane summary rank must be an integer") from error
    if rank < 0:
        raise argparse.ArgumentTypeError(
            "compiled-FX lane summary rank must be non-negative")
    return rank, Path(path_text)


def main(*, summarize_model_runner_outer_timing) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-step-manifest", type=Path, required=True)
    parser.add_argument("--token-prefixes", type=Path, required=True)
    parser.add_argument("--scheduler-trace", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluation/experiments/sharegpt_phase1.json"),
    )
    parser.add_argument("--host-launch-profile", type=Path)
    parser.add_argument("--ascend-runtime-prepare-summary", type=Path)
    parser.add_argument("--ascend-qkv-triton-composite-summary", type=Path)
    parser.add_argument(
        "--ascend-runtime-prepare-mechanism",
        action="append",
        dest="ascend_runtime_prepare_mechanisms",
        help=("Enable only one measured Ascend prepare mechanism; repeat this "
              "option for an explicit source-event subset."),
    )
    parser.add_argument("--ascend-compiled-fx-summary", type=Path)
    parser.add_argument(
        "--ascend-compiled-fx-lane-summary",
        action="append",
        type=_parse_compiled_fx_lane_summary,
        help=("Exact concurrent compiled-FX count sweep as rank:path; "
              "provide rank 0 and rank 1 exactly once."),
    )
    parser.add_argument(
        "--ascend-compiled-fx-wrapper-reference-summary",
        type=Path,
        help=("Generic compiled-FX sweep used only to retain the separately "
              "measured MoE-wrapper increment when lane timing is enabled."),
    )
    parser.add_argument(
        "--ascend-compiled-fx-reshape-lane-summary",
        action="append",
        type=_parse_compiled_fx_lane_summary,
        help=("Exact concurrent compiled-FX reshape metadata sweep as "
              "rank:path; provide rank 0 and rank 1 exactly once."),
    )
    parser.add_argument("--ascend-moe-wrapper-control-summary", type=Path)
    parser.add_argument("--ascend-unquant-moe-mlp-control-summary", type=Path)
    parser.add_argument("--ascend-fused-attention-prepare-summary", type=Path)
    parser.add_argument("--ascend-collective-prepare-summary-dir", type=Path)
    parser.add_argument(
        "--ascend-model-runner-framework-timing",
        type=Path,
        help=("Exact selected-step ModelRunner source-timing sidecar; it is "
              "diagnostic-only and cannot enter production Forward."),
    )
    parser.add_argument(
        "--ascend-model-runner-event-boundary",
        type=Path,
        help=("Matching selected-step worker event-boundary sidecar for "
              "--ascend-model-runner-framework-timing."),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--independent-eager", action="store_true")
    arguments = parser.parse_args()
    selected = json.loads(
        arguments.selected_step_manifest.resolve(strict=True).read_text(
            encoding="utf-8"
        )
    )
    token_prefixes = json.loads(
        arguments.token_prefixes.resolve(strict=True).read_text(encoding="utf-8")
    )
    runtime_prepare_profile = None
    runtime_prepare_identity = None
    runtime_prepare_event_builder = None
    compiled_fx_identity = None
    model_runner_outer_identity = None
    host_launch_profile = (
        None if arguments.host_launch_profile is None
        else load_host_launch_profile(arguments.host_launch_profile.resolve())
    )
    if arguments.ascend_runtime_prepare_summary is not None:
        if arguments.host_launch_profile is None:
            parser.error(
                "--ascend-runtime-prepare-summary requires "
                "--host-launch-profile")
        runtime_prepare_profile, runtime_prepare_identity = (
            _load_ascend_runtime_prepare_candidates(
                arguments.ascend_runtime_prepare_summary,
                arguments.ascend_collective_prepare_summary_dir,
            ))
        if arguments.ascend_runtime_prepare_mechanisms is None:
            enabled_mechanisms = tuple(
                mechanism for mechanism, _ in
                runtime_prepare_profile.service_time_s_by_mechanism)
        else:
            enabled_mechanisms = tuple(
                arguments.ascend_runtime_prepare_mechanisms)
            if (not enabled_mechanisms
                    or any(not isinstance(value, str) or not value
                           for value in enabled_mechanisms)):
                parser.error(
                    "--ascend-runtime-prepare-mechanism must be non-empty")
            if len(set(enabled_mechanisms)) != len(enabled_mechanisms):
                parser.error(
                    "--ascend-runtime-prepare-mechanism cannot repeat")
            profile_mechanisms = {
                mechanism for mechanism, _ in
                runtime_prepare_profile.service_time_s_by_mechanism
            }
            unknown_mechanisms = set(enabled_mechanisms) - profile_mechanisms
            if unknown_mechanisms:
                parser.error(
                    "unmeasured Ascend runtime prepare mechanisms: "
                    f"{sorted(unknown_mechanisms)}")
        runtime_prepare_identity["enabled_mechanisms"] = list(
            enabled_mechanisms)
        rotary_cache_update_enabled = (
            "rotary_cache_update_prepare" in enabled_mechanisms)
        source_rule_mechanisms = tuple(
            mechanism for mechanism in enabled_mechanisms
            if mechanism != "rotary_cache_update_prepare")
        event_builders = []
        if source_rule_mechanisms:
            event_builders.append(partial(
                ascend_910b_runtime_prepare_events,
                enabled_mechanisms=source_rule_mechanisms,
            ))
        if rotary_cache_update_enabled:
            event_builders.append(ascend_910b_rotary_cache_update_events)
        if len(event_builders) == 1:
            runtime_prepare_event_builder = event_builders[0]
        elif len(event_builders) == 2:
            runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
                *event_builders)
        else:
            raise AssertionError("Ascend runtime prepare has no event builder")
    elif arguments.ascend_collective_prepare_summary_dir is not None:
        parser.error(
            "--ascend-collective-prepare-summary-dir requires "
            "--ascend-runtime-prepare-summary")
    if ((arguments.ascend_model_runner_framework_timing is None)
            != (arguments.ascend_model_runner_event_boundary is None)):
        parser.error(
            "--ascend-model-runner-framework-timing and "
            "--ascend-model-runner-event-boundary must be supplied together")
    if arguments.ascend_qkv_triton_composite_summary is not None:
        if host_launch_profile is None:
            parser.error(
                "--ascend-qkv-triton-composite-summary requires "
                "--host-launch-profile")
        if (arguments.ascend_runtime_prepare_mechanisms is None
                and arguments.ascend_runtime_prepare_summary is not None):
            parser.error(
                "--ascend-qkv-triton-composite-summary with a runtime "
                "prepare summary requires an explicit mechanism subset that "
                "omits triton_dynamic_argument_prepare")
        if (arguments.ascend_runtime_prepare_mechanisms is not None
                and "triton_dynamic_argument_prepare" in
                arguments.ascend_runtime_prepare_mechanisms):
            parser.error(
                "QKV composite cannot be combined with "
                "triton_dynamic_argument_prepare")
        qkv_profile, qkv_identity = _load_ascend_qkv_triton_composite_candidate(
            arguments.ascend_qkv_triton_composite_summary,
            host_launch_profile,
        )
        if runtime_prepare_profile is None:
            runtime_prepare_profile = qkv_profile
            runtime_prepare_identity = qkv_identity
            runtime_prepare_event_builder = ascend_910b_qkv_triton_composite_events
        else:
            runtime_prepare_profile = _compose_runtime_prepare_profiles(
                (runtime_prepare_profile, qkv_profile))
            runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
                runtime_prepare_event_builder,
                ascend_910b_qkv_triton_composite_events,
            )
            runtime_prepare_identity = {
                "status": "diagnostic_only_not_production_eligible",
                "components": (runtime_prepare_identity, qkv_identity),
                "composition": (
                    "source-authored events append only; duplicate mechanism "
                    "names fail closed"),
            }
    if (arguments.ascend_compiled_fx_summary is not None
            and arguments.ascend_compiled_fx_lane_summary is not None):
        parser.error(
            "choose either --ascend-compiled-fx-summary or exact "
            "--ascend-compiled-fx-lane-summary inputs")
    if (arguments.ascend_compiled_fx_wrapper_reference_summary is not None
            and arguments.ascend_compiled_fx_lane_summary is None):
        parser.error(
            "--ascend-compiled-fx-wrapper-reference-summary requires exact "
            "--ascend-compiled-fx-lane-summary inputs")
    if (arguments.ascend_compiled_fx_summary is not None
            or arguments.ascend_compiled_fx_lane_summary is not None):
        if host_launch_profile is None:
            parser.error(
                "compiled-FX timing requires "
                "--host-launch-profile")
        if arguments.ascend_compiled_fx_summary is not None:
            compiled_fx_profile, compiled_fx_identity = (
                _load_ascend_compiled_fx_prepare_candidate(
                    arguments.ascend_compiled_fx_summary,
                    host_launch_profile,
                ))
        else:
            compiled_fx_profile, compiled_fx_identity = (
                _load_ascend_compiled_fx_lane_prepare_candidate(
                    tuple(arguments.ascend_compiled_fx_lane_summary),
                    host_launch_profile,
                ))
            if arguments.ascend_compiled_fx_wrapper_reference_summary is not None:
                _, wrapper_reference_identity = (
                    _load_ascend_compiled_fx_prepare_candidate(
                        arguments.ascend_compiled_fx_wrapper_reference_summary,
                        host_launch_profile,
                    ))
                compiled_fx_identity[
                    "wrapper_control_reference_total_service_ns_per_entry"
                ] = wrapper_reference_identity["total_service_ns_per_entry"]
                compiled_fx_identity["wrapper_control_reference"] = (
                    wrapper_reference_identity)
        if runtime_prepare_profile is None:
            runtime_prepare_profile = compiled_fx_profile
            runtime_prepare_identity = compiled_fx_identity
            runtime_prepare_event_builder = ascend_910b_compiled_fx_backend_events
        else:
            runtime_prepare_profile = _compose_runtime_prepare_profiles(
                (runtime_prepare_profile, compiled_fx_profile))
            runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
                runtime_prepare_event_builder,
                ascend_910b_compiled_fx_backend_events,
            )
            runtime_prepare_identity = {
                "status": "diagnostic_only_not_production_eligible",
                "components": (
                    runtime_prepare_identity,
                    compiled_fx_identity,
                ),
                "composition": (
                    "source-authored events append only; duplicate mechanism "
                    "names fail closed"),
            }
    if arguments.ascend_moe_wrapper_control_summary is not None:
        if compiled_fx_identity is None:
            parser.error(
                "--ascend-moe-wrapper-control-summary requires "
                "compiled-FX timing")
        wrapper_profile, wrapper_identity = (
            _load_ascend_moe_wrapper_control_candidate(
                arguments.ascend_moe_wrapper_control_summary,
                compiled_fx_identity,
            ))
        if runtime_prepare_profile is None or runtime_prepare_event_builder is None:
            raise AssertionError("compiled-FX profile must precede MoE wrapper")
        runtime_prepare_profile = _compose_runtime_prepare_profiles(
            (runtime_prepare_profile, wrapper_profile))
        runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
            runtime_prepare_event_builder,
            ascend_910b_moe_wrapper_control_events,
        )
        runtime_prepare_identity = {
            "status": "diagnostic_only_not_production_eligible",
            "components": (
                runtime_prepare_identity,
                wrapper_identity,
            ),
            "composition": (
                "source-authored events append only; duplicate mechanism "
                "names fail closed"),
        }
    if arguments.ascend_compiled_fx_reshape_lane_summary is not None:
        if host_launch_profile is None:
            parser.error(
                "--ascend-compiled-fx-reshape-lane-summary requires "
                "--host-launch-profile")
        reshape_profile, reshape_identity = (
            _load_ascend_compiled_fx_reshape_lane_prepare_candidate(
                tuple(arguments.ascend_compiled_fx_reshape_lane_summary),
            ))
        if runtime_prepare_profile is None:
            runtime_prepare_profile = reshape_profile
            runtime_prepare_identity = reshape_identity
            runtime_prepare_event_builder = ascend_910b_compiled_fx_reshape_events
        else:
            runtime_prepare_profile = _compose_runtime_prepare_profiles(
                (runtime_prepare_profile, reshape_profile))
            runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
                runtime_prepare_event_builder,
                ascend_910b_compiled_fx_reshape_events,
            )
            runtime_prepare_identity = {
                "status": "diagnostic_only_not_production_eligible",
                "components": (
                    runtime_prepare_identity,
                    reshape_identity,
                ),
                "composition": (
                    "source-authored events append only; duplicate mechanism "
                    "names fail closed"),
            }
    if arguments.ascend_unquant_moe_mlp_control_summary is not None:
        mlp_profile, mlp_identity = _load_ascend_unquant_mlp_control_candidate(
            arguments.ascend_unquant_moe_mlp_control_summary,
        )
        if runtime_prepare_profile is None or runtime_prepare_event_builder is None:
            parser.error(
                "--ascend-unquant-moe-mlp-control-summary requires an "
                "existing Ascend runtime-prepare profile")
        runtime_prepare_profile = _compose_runtime_prepare_profiles(
            (runtime_prepare_profile, mlp_profile))
        runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
            runtime_prepare_event_builder,
            ascend_910b_unquant_mlp_wrapper_events,
        )
        runtime_prepare_identity = {
            "status": "diagnostic_only_not_production_eligible",
            "components": (
                runtime_prepare_identity,
                mlp_identity,
            ),
            "composition": (
                "source-authored events append only; duplicate mechanism "
                "names fail closed"),
        }
    if arguments.ascend_fused_attention_prepare_summary is not None:
        if host_launch_profile is None:
            parser.error(
                "--ascend-fused-attention-prepare-summary requires "
                "--host-launch-profile")
        expected_workload = selected["expected_fingerprint"]["request_workload"]
        if not isinstance(expected_workload, list) or not expected_workload:
            raise ValueError(
                "selected step lacks a non-empty expected request workload")
        attention_profile, attention_identity = (
            _load_ascend_fused_attention_prepare_candidate(
                arguments.ascend_fused_attention_prepare_summary,
                host_launch_profile,
                len(expected_workload),
            ))
        attention_event_builder = (
            ascend_910b_fia_source_wrapper_events
            if attention_identity["source_descriptor"]
            == "ascend_910b_fia_source_wrapper_events"
            else partial(
                ascend_910b_runtime_prepare_events,
                enabled_mechanisms=("python_argument_prepare",),
            )
        )
        if runtime_prepare_profile is None:
            runtime_prepare_profile = attention_profile
            runtime_prepare_identity = attention_identity
            runtime_prepare_event_builder = attention_event_builder
        else:
            runtime_prepare_profile = _compose_runtime_prepare_profiles(
                (runtime_prepare_profile, attention_profile))
            runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
                runtime_prepare_event_builder,
                attention_event_builder,
            )
            runtime_prepare_identity = {
                "status": "diagnostic_only_not_production_eligible",
                "components": (
                    runtime_prepare_identity,
                    attention_identity,
                ),
                "composition": (
                    "source-authored events append only; duplicate mechanism "
                    "names fail closed"),
            }
    if arguments.ascend_model_runner_framework_timing is not None:
        if host_launch_profile is None:
            parser.error(
                "--ascend-model-runner-framework-timing requires "
                "--host-launch-profile")
        model_runner_profile, model_runner_outer_identity = (
            _load_ascend_model_runner_outer_diagnostic(
                arguments.ascend_model_runner_framework_timing,
                arguments.ascend_model_runner_event_boundary,
                summarize_model_runner_outer_timing=summarize_model_runner_outer_timing,
            ))
        if runtime_prepare_profile is None:
            runtime_prepare_profile = model_runner_profile
            runtime_prepare_identity = model_runner_outer_identity
            runtime_prepare_event_builder = partial(
                ascend_910b_model_runner_outer_events,
                enabled_ranks=(model_runner_outer_identity["worker_rank"],),
            )
        else:
            runtime_prepare_profile = _compose_runtime_prepare_profiles(
                (runtime_prepare_profile, model_runner_profile))
            runtime_prepare_event_builder = _compose_runtime_prepare_event_builders(
                runtime_prepare_event_builder,
                partial(
                    ascend_910b_model_runner_outer_events,
                    enabled_ranks=(model_runner_outer_identity["worker_rank"],),
                ),
            )
            runtime_prepare_identity = {
                "status": "diagnostic_only_not_production_eligible",
                "components": (
                    runtime_prepare_identity,
                    model_runner_outer_identity,
                ),
                "composition": (
                    "source-authored events append only; duplicate mechanism "
                    "names fail closed"),
            }
    result = build_selected_step_injection_replay(
        REPOSITORY_ROOT,
        arguments.config.resolve(strict=True),
        selected,
        token_prefixes,
        arguments.scheduler_trace.resolve(strict=True),
        host_launch_profile=host_launch_profile,
        host_runtime_prepare_profile=runtime_prepare_profile,
        runtime_prepare_event_builder=runtime_prepare_event_builder,
        independent_eager=arguments.independent_eager,
    )
    if model_runner_outer_identity is not None:
        injected_trace_step_id = result["step_id_mapping"][
            "injected_trace_step_id"]
        if model_runner_outer_identity["trace_step_id"] != injected_trace_step_id:
            raise ValueError(
                "ModelRunner source timing trace step disagrees with replayed "
                "injected step")
    if runtime_prepare_identity is not None:
        result["runtime_prepare_candidate"] = runtime_prepare_identity
    write_json(arguments.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
