"""Single-case compact ShareGPT evaluation: simulation argv construction,
Forward execution checks, completion validation, and launch-gap
attribution."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

from src import _platform_compat as resource
from evaluation.serving.metrics.compact_step_alignment import align_compact_case
from evaluation.serving.metrics.scheduler_progress import (
    validate_scheduler_progress_manifest,
)
from evaluation.serving.simulation.execution_artifact import (
    DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
    performance_timing_status,
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
)
from src.serving.workload.expert_frequency import (
    load_expert_frequency_profile,
)
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput
from src.serving.host_timing import (
    SERVING_TIMING_MODES,
    load_host_launch_profile,
    load_host_request_path_profile,
    load_scheduler_host_profile,
    validate_serving_timing_mode,
)

from evaluation.json_shape import require_object_value as _mapping

from evaluation.serving.metrics.compact_shape_validators import (
    _canonical_json_sha256,
    _confined_path,
    _string,
    _sha256,
    _integer,
)
from evaluation.serving.metrics.compact_case_contract import (
    ValidatedCompactCase,
    scheduler_pipeline_depth_from_validated_case,
    scheduler_async_scheduling_from_validated_case,
    Phase1Config,
    Phase1Cell,
    compact_case_path,
    _validate_case_binding,
    _model_config_path,
)
from evaluation.serving.metrics.compact_run_environment import (
    _HOST_PROFILE_VARIANTS,
    resolve_repository_input,
    case_host_profile_paths,
)
from evaluation.serving.metrics.compact_execution_payload_validation import (
    _validate_midlayer_wall_time,
    _validate_debug_capacity,
    _validate_analytical_capacity,
    _validate_scheduler_runtime,
    _validate_execution_artifact,
    _validate_execution_cache_stats,
    _expand_runtime_memory_policy_path,
)
from evaluation.serving.metrics.compact_metric_comparison import (
    _timing_components,
    _validated_host_timing_payload,
    _host_error_attribution,
    _capture_runtime_comparison,
    _observed_metrics,
    _ttft_breakdown_unavailable_reason,
    _compare_prediction,
    _unavailable_prediction_metrics,
)
from evaluation.serving.metrics.compact_execution_payload_validation import validate_execution_repository_provenance


_KNOWN_A800_GEMM_LAUNCH_GAP = re.compile(
    r"^uncharacterized NVIDIA A800 GEMM launch key \(.+\)$"
)
_KNOWN_METAX_GEMM_LAUNCH_GAP = re.compile(
    r"^uncharacterized MetaX C500 GEMM launch key \(.+\)$"
)
_KNOWN_METAX_VECTOR_PREFIX = "MetaX C500 production Vector launch is unsupported for "
_KNOWN_METAX_VECTOR_SUFFIX = (
    ": no MACA 3.7.2 device source or complete versioned launch fact is retained"
)
_KNOWN_METAX_FLASH_ATTN_GAP = (
    "MetaX C500 MACA 3.7.2 production FlashAttention launch is "
    "evidence_pending: the installed library has no auditable "
    "default-solution export, while retained 3.2.1 observations only bind "
    "evaluation grid/block/split; NVIDIA FA2 launch rules cannot be substituted"
)
_KNOWN_A800_FA2_COMBINE_GAP = re.compile(
    r"^FA2 combine dependency evidence has no unique source fragment for "
    r"block_m=[1-9][0-9]*, head_dim=[1-9][0-9]*, "
    r"num_splits=[1-9][0-9]*, max_splits=[1-9][0-9]*$"
)
_KNOWN_A800_ADD_RMS_QUEUED_SOURCE_GAP = (
    "A800 block-256 AddRMSNorm queued CTA source service is unavailable"
)
_KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS = {
    "metax_c500": (
        "MetaX C500 FUSED_MOE expert-parallel launch is not source-backed"
    ),
    "ascend_910b": (
        "Ascend 910B FUSED_MOE expert-parallel launch is not source-backed"
    ),
}


_MOE_MODEL_IDS = {"qwen3-30b-a3b", "Qwen3-30B-A3B"}
_EXPERT_FREQUENCY_MANIFEST = Path(
    "serving_dataset/moe_expert_freq/manifest.json")


def _expected_expert_routing(
    repository_root: str | Path,
    cell: Phase1Cell,
) -> ExpertRoutingInput:
    if cell.model.dataset_model_id in _MOE_MODEL_IDS:
        model_path = _model_config_path(
            repository_root, cell.model.simulation_model_path)
        model = load_model_config(model_path)
        if model.model_id is None:
            executing_model_id = model_path.stem
        else:
            executing_model_id = model.model_id
        profile = load_expert_frequency_profile(
            Path(repository_root).resolve() / _EXPERT_FREQUENCY_MANIFEST,
            cell.model.dataset_model_id,
            expected_model_id=executing_model_id,
            expected_moe_layers=sum(
                model.layer_uses_moe(layer_index)
                for layer_index in range(model.num_hidden_layers)
            ),
            expected_num_experts=model.num_experts,
        )
        return ExpertRoutingInput.profile_observation(profile)
    return ExpertRoutingInput.not_applicable()


def _validated_precomputed_alignment(
    value: object,
) -> dict[str, object]:
    alignment = _mapping(value, "precomputed compact alignment")
    status = alignment["status"]
    if status not in ("matched", "mismatched", "unsupported"):
        raise ValueError(f"invalid precomputed alignment status: {status!r}")
    expected_success = status == "matched"
    expected_fields = {
        "alignment_stage_ran": True,
        "alignment_checked": expected_success,
        "formal_conclusion_allowed": expected_success,
    }
    for field, expected in expected_fields.items():
        if alignment[field] is not expected:
            raise ValueError(
                f"precomputed alignment {field} is inconsistent with {status}"
            )
    _mapping(alignment["phases"], "precomputed alignment phases")
    return dict(alignment)


def run_isolated_case(
    repository_root: str,
    config: Phase1Config,
    cell: Phase1Cell,
    source_commit: str,
    alignment_checked: bool | Mapping[str, object],
    execution_mode: str,
    *,
    load_validated_case: Callable[[Path], ValidatedCompactCase],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
    host_request_path_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Load and evaluate one isolated compact case in the default timing mode."""

    case = load_validated_case(compact_case_path(repository_root, config, cell))
    hardware_id = cell.hardware.hardware_id
    profile_path = None
    if (host_launch_profile_paths is not None
            and hardware_id in host_launch_profile_paths):
        profile_path = host_launch_profile_paths[hardware_id]
    common = {
        "source_commit": source_commit,
        "alignment_checked": alignment_checked,
        "execution_mode": execution_mode,
        "debug_num_kv_blocks": debug_num_kv_blocks,
        "debug_analytical_launch": debug_analytical_launch,
    }
    if profile_path is None:
        if scheduler_host_profile_paths is not None:
            raise ValueError(
                "device-only evaluation cannot consume Scheduler Host "
                "profiles")
        return evaluate_validated_case(
            repository_root,
            config,
            cell,
            case,
            run_simulation,
            serving_timing_mode="device_only",
            **common,
        )
    if (scheduler_host_profile_paths is None
            or hardware_id not in scheduler_host_profile_paths):
        raise ValueError(
            "full-timing evaluation requires a Scheduler Host profile for "
            f"{hardware_id}")
    request_path_override = None
    if (host_request_path_profile_paths is not None
            and hardware_id in host_request_path_profile_paths):
        request_path_override = host_request_path_profile_paths[hardware_id]
    return evaluate_validated_case(
        repository_root,
        config,
        cell,
        case,
        run_simulation,
        serving_timing_mode="device_plus_launch_plus_scheduler",
        host_launch_profile_path=profile_path,
        scheduler_host_profile_path=scheduler_host_profile_paths[hardware_id],
        host_request_path_profile_path=request_path_override,
        **common,
    )


def run_isolated_case_three_modes(
    repository_root: str,
    config: Phase1Config,
    cell: Phase1Cell,
    source_commit: str,
    alignment_checked: bool | Mapping[str, object],
    execution_mode: str,
    *,
    load_validated_case: Callable[[Path], ValidatedCompactCase],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Load and evaluate one isolated compact case in all timing modes."""

    case = load_validated_case(compact_case_path(repository_root, config, cell))
    hardware_id = cell.hardware.hardware_id
    if host_launch_profile_paths is None or hardware_id not in host_launch_profile_paths:
        raise ValueError(
            "three-mode evaluation requires a Host launch profile for "
            f"{hardware_id}"
        )
    if (scheduler_host_profile_paths is None
            or hardware_id not in scheduler_host_profile_paths):
        raise ValueError(
            "three-mode evaluation requires a Scheduler Host profile for "
            f"{hardware_id}"
        )
    return evaluate_validated_case_three_modes(
        repository_root,
        config,
        cell,
        case,
        run_simulation,
        source_commit=source_commit,
        alignment_checked=alignment_checked,
        execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        host_launch_profile_path=host_launch_profile_paths[hardware_id],
        scheduler_host_profile_path=scheduler_host_profile_paths[hardware_id],
    )


def build_simulation_argv(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    *,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    serving_timing_mode: str = "device_only",
    host_launch_profile_path: str | Path | None = None,
    scheduler_host_profile_path: str | Path | None = None,
    host_request_path_profile_path: str | Path | None = None,
    scheduler_pipeline_depth: int | None = None,
    scheduler_async_scheduling: bool | None = None,
) -> list[str]:
    """Build a fully explicit current-vLLM Forward invocation."""

    root = Path(repository_root).resolve()
    _validate_execution_mode(execution_mode)
    _validate_case_binding(config, cell, case)
    if scheduler_pipeline_depth is None:
        resolved_scheduler_pipeline_depth = (
            scheduler_pipeline_depth_from_validated_case(case))
    else:
        if (isinstance(scheduler_pipeline_depth, bool)
                or not isinstance(scheduler_pipeline_depth, int)
                or scheduler_pipeline_depth < 1):
            raise ValueError(
                "explicit scheduler_pipeline_depth must be a positive "
                "non-bool integer")
        resolved_scheduler_pipeline_depth = scheduler_pipeline_depth
    if scheduler_async_scheduling is None:
        async_scheduling = scheduler_async_scheduling_from_validated_case(case)
    elif isinstance(scheduler_async_scheduling, bool):
        async_scheduling = scheduler_async_scheduling
    else:
        raise TypeError("explicit scheduler_async_scheduling must be boolean")
    prediction = config.prediction.values
    model_path = _model_config_path(root, cell.model.simulation_model_path)
    hardware_path = _confined_path(
        root, cell.hardware.hardware_config_path, "hardware config"
    )
    if debug_num_kv_blocks is not None and (
            type(debug_num_kv_blocks) is not int or debug_num_kv_blocks < 1):
        raise ValueError("debug_num_kv_blocks must be a positive integer")
    if not isinstance(debug_analytical_launch, bool):
        raise TypeError("debug_analytical_launch must be boolean")
    validate_serving_timing_mode(serving_timing_mode)
    if serving_timing_mode == "device_only":
        if host_launch_profile_path is not None:
            raise ValueError("device_only cannot consume a Host launch profile")
    elif host_launch_profile_path is None:
        raise ValueError(f"{serving_timing_mode} requires a Host launch profile")
    if serving_timing_mode == "device_plus_launch_plus_scheduler":
        if scheduler_host_profile_path is None:
            raise ValueError(
                f"{serving_timing_mode} requires a Scheduler Host profile")
    elif scheduler_host_profile_path is not None:
        raise ValueError(
            f"{serving_timing_mode} cannot consume a Scheduler Host profile")
    variant_paths = None
    if (cell.hardware.hardware_id in _HOST_PROFILE_VARIANTS
            and serving_timing_mode != "device_only"):
        # Versioned Host profiles follow the runtime the case was captured on.
        case_runtime = _mapping(case.manifest["runtime"], "manifest runtime")
        variant_paths = case_host_profile_paths(
            root, cell.hardware.hardware_id,
            _string(case_runtime["runtime_version"], "case runtime version"))
        if host_request_path_profile_path is None and (
                serving_timing_mode == "device_plus_launch_plus_scheduler"):
            host_request_path_profile_path = variant_paths["host_request_path"]
    request_manifest = _mapping(case.manifest["requests"], "manifest requests")
    requests_path = Path(case.case_dir) / request_manifest["path"]
    argv = [
        "--canonical-requests", str(requests_path),
        "--model", str(model_path),
        "--hardware-config", str(hardware_path),
        "--deployment", str(prediction["deployment"]),
        "--backend", str(prediction["backend"]),
        "--vllm-scheduler-version", str(prediction["scheduler_version"]),
        "--prediction-method", str(prediction["prediction_method"]),
        "--execution-mode", execution_mode,
        "--serving-timing-mode", serving_timing_mode,
        "--collective-backend", str(
            prediction["tensor_collective_backend"]),
        "--launch-policy", (
            "analytical_launch"
            if debug_analytical_launch else str(prediction["launch_policy"])
        ),
        "--attention-kv-layout", str(prediction["attention_kv_layout"]),
        "--attention-kv-distribution", str(
            prediction["attention_kv_distribution"]
        ),
        "--graph-mode", cell.hardware.graph_execution.configured_mode,
        "--graph-uniform-decode-query-len", str(
            cell.hardware.graph_execution.uniform_decode_query_len
        ),
        "--max-batched-tokens", str(prediction["max_batched_tokens"]),
        "--max-requests", str(prediction["max_requests"]),
        "--scheduler-pipeline-depth", str(resolved_scheduler_pipeline_depth),
        (
            "--vllm-async-scheduling"
            if async_scheduling else "--no-vllm-async-scheduling"
        ),
        "--kv-block-size-tokens", str(prediction["kv_block_size_tokens"]),
        "--max-model-len", str(prediction["max_model_len"]),
        "--mtp-num-speculative-tokens", str(
            prediction["mtp_num_speculative_tokens"]
        ),
        "--mtp-acceptance-policy", str(prediction["mtp_acceptance_policy"]),
        "--mtp-fixed-accepted-tokens", str(
            prediction["mtp_fixed_accepted_tokens"]
        ),
        "--mtp-acceptance-source-id", str(
            prediction["mtp_acceptance_source_id"]
        ),
        "--prefill-tp-size", str(prediction["tensor_parallel_size"]),
        "--prefill-pp-size", str(prediction["pipeline_parallel_size"]),
        (
            "--enable-chunked-prefill"
            if prediction["enable_chunked_prefill"]
            else "--no-enable-chunked-prefill"
        ),
        (
            "--enable-prefix-caching"
            if prediction["enable_prefix_caching"]
            else "--no-enable-prefix-caching"
        ),
        "--router-resource-id", str(prediction["router_resource_id"]),
        "--router-dispatch-service-time-s", str(
            prediction["router_dispatch_service_time_s"]
        ),
    ]
    if cell.hardware.graph_execution.capture_sizes:
        argv.extend((
            "--graph-capture-sizes",
            *(str(size) for size in cell.hardware.graph_execution.capture_sizes),
        ))
    if cell.hardware.graph_execution.piecewise_eager_workload_kinds:
        argv.extend((
            "--graph-piecewise-eager-workload-kinds",
            *cell.hardware.graph_execution.piecewise_eager_workload_kinds,
        ))
    if host_launch_profile_path is not None:
        profile_path = resolve_repository_input(
            root, str(host_launch_profile_path), "Host launch profile")
        profile = load_host_launch_profile(profile_path)
        if profile.hardware_id != cell.hardware.hardware_id:
            raise ValueError(
                "Host launch profile hardware disagrees with evaluation cell")
        if (variant_paths is not None
                and profile_path.resolve() != variant_paths["host_launch"]):
            raise ValueError(
                "Host launch profile is not the case runtime version's profile")
        argv.extend(("--host-launch-profile", str(profile_path)))
    if scheduler_host_profile_path is not None:
        scheduler_path = resolve_repository_input(
            root, str(scheduler_host_profile_path), "Scheduler Host profile")
        scheduler_profile = load_scheduler_host_profile(scheduler_path)
        if scheduler_profile.hardware_id != cell.hardware.hardware_id:
            raise ValueError(
                "Scheduler Host profile hardware disagrees with evaluation "
                "cell")
        if (variant_paths is not None
                and scheduler_path.resolve() != variant_paths["scheduler_host"]):
            raise ValueError(
                "Scheduler Host profile is not the case runtime version's "
                "profile")
        argv.extend(("--scheduler-host-profile", str(scheduler_path)))
    if host_request_path_profile_path is not None:
        request_path = resolve_repository_input(
            root,
            str(host_request_path_profile_path),
            "Host request path profile")
        request_path_profile = load_host_request_path_profile(request_path)
        if request_path_profile.hardware_id != cell.hardware.hardware_id:
            raise ValueError(
                "Host request path profile hardware disagrees with evaluation "
                "cell")
        argv.extend(("--host-request-path-profile", str(request_path)))
    if variant_paths is not None:
        argv.extend((
            "--host-runtime-prepare-profile",
            str(variant_paths["host_runtime_prepare"]),
        ))
    if debug_num_kv_blocks is None:
        argv.extend((
            "--runtime-memory-policy",
            _expand_runtime_memory_policy_path(
                root, config, cell, cell.model.dataset_model_id),
        ))
    else:
        argv.extend(("--debug-num-kv-blocks", str(debug_num_kv_blocks)))
    if (prediction["tensor_collective_backend"] == "protobridge"
            and prediction["tensor_parallel_size"] > 1):
        argv.extend((
            "--tensor-all-reduce-profile-sha256",
            str(prediction["tensor_all_reduce_profile_sha256"]),
            "--tensor-all-gather-profile-sha256",
            str(prediction["tensor_all_gather_profile_sha256"]),
        ))
    if prediction["expert_parallel_size"] > 1:
        expert_routing = _expected_expert_routing(root, cell)
        if expert_routing.profile is None:
            raise RuntimeError(
                "profile expert routing lost its canonical profile")
        argv.extend((
            "--enable-expert-parallel",
            "--expert-routing-policy", "profile_observation",
            "--expert-routing-source-id", expert_routing.source_id,
            "--expert-routing-profile-manifest",
            str((root / _EXPERT_FREQUENCY_MANIFEST).resolve()),
        ))
    forbidden_flags = {"--workload", "--production-trace"}
    if any(argument in forbidden_flags for argument in argv):
        raise AssertionError("Forward argv contains an evaluation-only input")
    return argv


def evaluate_validated_case(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    *,
    source_commit: str,
    alignment_checked: bool | Mapping[str, object] = True,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    serving_timing_mode: str = "device_only",
    host_launch_profile_path: str | Path | None = None,
    scheduler_host_profile_path: str | Path | None = None,
    host_request_path_profile_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one current prediction and compare it with compact observations."""

    _validate_execution_mode(execution_mode)
    argv = build_simulation_argv(
        repository_root, config, cell, case, execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        serving_timing_mode=serving_timing_mode,
        host_launch_profile_path=host_launch_profile_path,
        scheduler_host_profile_path=scheduler_host_profile_path,
        host_request_path_profile_path=host_request_path_profile_path)
    if type(alignment_checked) is bool:
        alignment_result = (
            validate_compact_alignment(repository_root, config, cell, case)
            if alignment_checked
            else {
                "alignment_stage_ran": False,
                "alignment_checked": False,
                "status": "skipped_metrics_only_debug",
                "formal_conclusion_allowed": False,
                "phases": {},
            }
        )
    else:
        alignment_result = _validated_precomputed_alignment(alignment_checked)
    observed = _observed_metrics(case)
    common = _cell_context(config, cell, case, argv, alignment_result)
    started = time.perf_counter()
    try:
        full_payload = run_simulation(argv)
    except NotImplementedError as error:
        reason = _known_not_implemented_launch_gap_reason(
            error, cell.hardware.hardware_id
        )
        if reason is None:
            raise
        elapsed = time.perf_counter() - started
        return {
            **common,
            "execution_mode": execution_mode,
            "serving_timing_mode": serving_timing_mode,
            "evaluation_wall_time_seconds": elapsed,
            "prediction_status": "unsupported_launch_coverage",
            "prediction_reason": reason,
            "performance_timing_status": None,
            "accuracy_claim_status": "not_available",
            "forward_provenance": None,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "timing_components": _timing_components(False, elapsed),
            "metrics": _unavailable_prediction_metrics(observed),
        }
    except ValueError as error:
        reason = _known_value_error_launch_gap_reason(
            error, cell.hardware.hardware_id)
        if reason is None:
            raise
        elapsed = time.perf_counter() - started
        return {
            **common,
            "execution_mode": execution_mode,
            "serving_timing_mode": serving_timing_mode,
            "evaluation_wall_time_seconds": elapsed,
            "prediction_status": "unsupported_launch_coverage",
            "prediction_reason": reason,
            "performance_timing_status": None,
            "accuracy_claim_status": "not_available",
            "forward_provenance": None,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "timing_components": _timing_components(False, elapsed),
            "metrics": _unavailable_prediction_metrics(observed),
        }
    forward_provenance = _validate_completed_payload(
        repository_root,
        config,
        cell,
        case,
        full_payload,
        source_commit,
        execution_mode,
        debug_num_kv_blocks,
        debug_analytical_launch,
        serving_timing_mode,
    )
    comparison = _compare_prediction(observed, full_payload, len(case.requests))
    host_timing = _validated_host_timing_payload(
        full_payload, serving_timing_mode, cell.hardware.hardware_id)
    process_peak_rss_bytes = _integer(
        full_payload["process_peak_rss_bytes"], "process peak RSS bytes")
    if process_peak_rss_bytes <= 0:
        raise ValueError("process peak RSS bytes must be positive")
    del full_payload
    elapsed = time.perf_counter() - started
    return {
        **common,
        "execution_mode": execution_mode,
        "serving_timing_mode": serving_timing_mode,
        "evaluation_wall_time_seconds": elapsed,
        "prediction_status": "completed",
        "prediction_reason": None,
        "performance_timing_status": forward_provenance[
            "performance_timing_status"
        ],
        "accuracy_claim_status": (
            "provisional_observer_effect"
            if common["observation_quality"]["status"] == "provisional"
            else forward_provenance["accuracy_claim_status"]
        ),
        "forward_provenance": forward_provenance,
        "timing_components": _timing_components(True, elapsed),
        "process_peak_rss_bytes": process_peak_rss_bytes,
        "host_timing": host_timing,
        "metrics": comparison,
    }


def evaluate_validated_case_three_modes(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    *,
    source_commit: str,
    host_launch_profile_path: str | Path,
    scheduler_host_profile_path: str | Path,
    alignment_checked: bool | Mapping[str, object] = True,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
) -> dict[str, object]:
    """Run three independent simulations and attribute error reductions."""

    if type(alignment_checked) is bool:
        alignment_result = (
            validate_compact_alignment(repository_root, config, cell, case)
            if alignment_checked else {
                "alignment_stage_ran": False,
                "alignment_checked": False,
                "status": "skipped_metrics_only_debug",
                "formal_conclusion_allowed": False,
                "phases": {},
            }
        )
    else:
        alignment_result = _validated_precomputed_alignment(alignment_checked)
    repeated_alignment_input = (
        False if alignment_checked is False else alignment_result)
    results = {}
    for timing_mode in SERVING_TIMING_MODES:
        profile_path_by_mode = {
            "device_only": None,
            "device_plus_launch": host_launch_profile_path,
            "device_plus_launch_plus_scheduler": host_launch_profile_path,
        }
        scheduler_path_by_mode = {
            "device_only": None,
            "device_plus_launch": None,
            "device_plus_launch_plus_scheduler": scheduler_host_profile_path,
        }
        results[timing_mode] = evaluate_validated_case(
            repository_root,
            config,
            cell,
            case,
            run_simulation,
            source_commit=source_commit,
            alignment_checked=repeated_alignment_input,
            execution_mode=execution_mode,
            debug_num_kv_blocks=debug_num_kv_blocks,
            debug_analytical_launch=debug_analytical_launch,
            serving_timing_mode=timing_mode,
            host_launch_profile_path=profile_path_by_mode[timing_mode],
            scheduler_host_profile_path=scheduler_path_by_mode[timing_mode],
        )
    device_only = results["device_only"]
    mode_results = {
        mode: {
            field: result[field]
            for field in (
                "prediction_status", "prediction_reason",
                "performance_timing_status", "accuracy_claim_status",
                "forward_provenance", "timing_components", "metrics",
            )
        } | ({"host_timing": result["host_timing"]}
             if "host_timing" in result else {})
        for mode, result in results.items()
    }
    return {
        **device_only,
        "evaluation_wall_time_seconds": sum(
            result["evaluation_wall_time_seconds"] for result in results.values()),
        "timing_components": {
            "schema_version": 2,
            "independent_simulation_per_mode": True,
            "modes": tuple(SERVING_TIMING_MODES),
        },
        "timing_mode_results": mode_results,
        "host_error_attribution": _host_error_attribution(mode_results),
    }


def _cell_context(
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    argv: Sequence[str],
    alignment_result: Mapping[str, object],
) -> dict[str, object]:
    manifest = _mapping(case.manifest, "compact manifest")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    requests = _mapping(manifest["requests"], "manifest requests")
    alignment = _mapping(manifest["alignment_steps"], "manifest alignment_steps")
    context = {
        "cell_id": cell.cell_id,
        "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id,
        "case_id": cell.case_id,
        "target_qps": cell.qps,
        "request_count": len(case.requests),
        "ttft_breakdown_unavailable_reason": _ttft_breakdown_unavailable_reason(case),
        "manifest_sha256": cell.manifest_sha256,
        "requests_sha256": requests["sha256"],
        "capture_contract_status": (
            f"schema_v{manifest['schema_version']}_validated"
        ),
        "capture_runtime_comparison": _capture_runtime_comparison(
            runtime, config.prediction.values, cell.qps
        ),
        "prediction_contract": {
            **dict(config.prediction.values),
            "model_config_sha256": cell.model.model_config_sha256,
            "hardware_config_sha256": cell.hardware.hardware_config_sha256,
        },
        "forward_input": {
            "kind": "validated_manifest_binding_and_requests_csv_only",
            "argv_sha256": _canonical_json_sha256(list(argv)),
            "alignment_consumed_by_forward": False,
        },
        "alignment": dict(alignment_result),
        "observation_quality": _observation_quality(manifest),
        "alignment_files": {
            "alignment_steps": {
                "status": alignment["status"],
                "sha256": alignment["sha256"],
                "phases": alignment["phases"],
            },
        },
    }
    if case.scheduler_steps is not None:
        context["observed_scheduler_progress"] = (
            _observed_scheduler_progress(case.scheduler_steps, len(case.requests))
        )
    return context


def _observation_quality(manifest: Mapping[str, object]) -> dict[str, object]:
    """Label capture differences without excluding completed predictions."""

    if manifest["schema_version"] != 7:
        return {"status": "legacy_contract", "formal_allowed": True,
                "failed_metrics": [], "metric_delta_percent": None,
                "threshold_percent": None, "source_archive_id": None}
    capture = _mapping(manifest["capture"], "manifest capture")
    validation = _mapping(capture["observer_effect_validation"],
                          "capture observer effect validation")
    deltas = _mapping(validation["metric_delta_percent"],
                      "observer effect deltas")
    capture_failed = sorted(field for field, value in deltas.items()
                            if value > validation["threshold_percent"])
    if validation["passed"] != (not capture_failed):
        raise ValueError("validated observer quality is inconsistent")
    failed = sorted(field for field, value in deltas.items() if value > 5.0)
    return {
        "status": "accepted" if not failed else "provisional",
        "formal_allowed": True,
        "failed_metrics": failed,
        "metric_delta_percent": dict(deltas),
        "threshold_percent": 5.0,
        "source_archive_id": validation["source_archive_id"],
        "capture_validation_threshold_percent": validation["threshold_percent"],
        "capture_validation_passed": validation["passed"],
    }


def _observed_scheduler_progress(
    steps: Sequence[Mapping[str, object]], request_count: int
) -> Mapping[str, object]:
    if not steps:
        raise ValueError("observed scheduler steps cannot be empty")
    series = []
    for index, value in enumerate(steps):
        step = _mapping(value, f"observed scheduler step {index}")
        series.append({
            "step_index": step["step_index"],
            "step_id": step["step_id"],
            "start_time_s": step["step_start_ns"] / 1_000_000_000.0,
            "end_time_s": step["step_end_ns"] / 1_000_000_000.0,
            "request_count": step["scheduled_request_count"],
            "prefill_request_count": step["prefill_request_count"],
            "decode_request_count": step["decode_request_count"],
            "prefill_tokens": step["prefill_tokens"],
            "decode_tokens": step["decode_tokens"],
            "scheduled_tokens": step["scheduled_tokens"],
            "cumulative_scheduled_tokens": step[
                "cumulative_scheduled_tokens"
            ],
            "cumulative_completed_requests": step[
                "cumulative_completed_requests"
            ],
        })
    final = series[-1]
    return {
        "schema_version": 1,
        "time_axis": {
            "kind": "observed_serving_step_wall",
            "unit": "seconds",
            "start_boundary": "immediately_before_scheduler_schedule",
            "end_boundary": "immediately_after_scheduler_output_update",
        },
        "series": series,
        "summary": {
            "step_count": len(series),
            "request_count": request_count,
            "scheduled_tokens": final["cumulative_scheduled_tokens"],
            "completed_requests": final["cumulative_completed_requests"],
            "end_time_s": final["end_time_s"],
        },
    }


def validate_compact_alignment(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
) -> Mapping[str, object]:
    """Run real ScheduleStep-to-lowering structural alignment."""

    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    return align_compact_case(repository_root, config, cell, case, model)


def _validate_completed_payload(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    payload_value: object,
    source_commit: str,
    expected_execution_mode: str,
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    expected_serving_timing_mode: str = "device_only",
) -> Mapping[str, object]:
    payload = _mapping(payload_value, "simulation payload")
    validate_serving_timing_mode(expected_serving_timing_mode)
    if payload["serving_timing_mode"] != expected_serving_timing_mode:
        raise ValueError("simulation payload Serving timing mode mismatch")
    if payload["execution_mode"] != expected_execution_mode:
        raise ValueError("simulation payload execution_mode mismatch")
    midlayer_wall_time = _validate_midlayer_wall_time(payload)
    execution_cache_stats = _validate_execution_cache_stats(
        payload["execution_cache_stats"], expected_execution_mode)
    expected_launch_policy = (
        "analytical_launch"
        if debug_analytical_launch else config.prediction.values["launch_policy"]
    )
    expected_top_level = {
        "backend": config.prediction.values["backend"],
        "deployment": config.prediction.values["deployment"],
        "launch_policy": expected_launch_policy,
        "prediction_method": config.prediction.values["prediction_method"],
        "model_compute_dtype": config.prediction.values["compute_dtype"],
    }
    for field, expected in expected_top_level.items():
        if payload[field] != expected:
            raise ValueError(
                f"simulation payload {field} mismatch: "
                f"{payload[field]!r} != {expected!r}"
            )
    if payload["hardware_ids"] != [cell.hardware.hardware_id]:
        raise ValueError("simulation payload hardware_ids mismatch")

    request_manifest = _mapping(case.manifest["requests"], "manifest requests")
    request_source = _mapping(payload["request_source"], "request source")
    if request_source["kind"] != "canonical_requests":
        raise ValueError("simulation request source must be canonical_requests")
    request_expectations = {
        "source_sha256": cell.requests_sha256,
        "row_count": len(case.requests),
        "request_id_policy": "physical_row_request_id_v1",
        "equal_arrival_order": "physical_row_ascending",
    }
    for field, expected in request_expectations.items():
        if request_source[field] != expected:
            raise ValueError(f"simulation request source {field} mismatch")
    expected_requests_path = (
        Path(case.case_dir) / request_manifest["path"]
    ).resolve()
    if (Path(request_source["input_path"]).resolve() != expected_requests_path
            or Path(request_source["source_path"]).resolve()
            != expected_requests_path):
        raise ValueError("simulation request source path mismatch")
    if request_manifest["sha256"] != cell.requests_sha256:
        raise ValueError("validated manifest requests SHA256 changed from config")
    normalized_request_sha = _sha256(
        request_source["normalized_request_stream_sha256"],
        "normalized request stream SHA256",
    )
    workload_binding = _mapping(
        payload["workload_assumption_binding"], "workload assumption binding"
    )
    if workload_binding["request_stream_sha256"] != normalized_request_sha:
        raise ValueError("workload binding request stream SHA256 mismatch")
    expected_expert_routing = _expected_expert_routing(
        repository_root, cell)
    expected_routing_sha256 = expected_expert_routing.policy_descriptor_sha256
    binding_components = {
        "request_stream_sha256": normalized_request_sha,
        "expert_routing_policy_sha256": expected_routing_sha256,
        "mtp_acceptance_policy_sha256": None,
    }
    if workload_binding["mtp_acceptance_policy_sha256"] is not None:
        raise ValueError("disabled MTP workload binding must be null")
    if workload_binding["composite_sha256"] != _canonical_json_sha256(
        binding_components
    ):
        raise ValueError("workload binding composite SHA256 mismatch")
    arrival_process = _mapping(payload["arrival_process"], "arrival process")
    if (arrival_process["kind"] != "recorded_canonical_arrivals"
            or arrival_process["request_stream_sha256"] != normalized_request_sha):
        raise ValueError("arrival process is not bound to canonical requests")

    model = _mapping(payload["model_config_provenance"], "model provenance")
    if model["compute_dtype"] != config.prediction.values["compute_dtype"]:
        raise ValueError(
            "simulation model provenance compute_dtype differs from the "
            "prediction contract")
    reported_model_id = model["model_id"]
    if reported_model_id is not None and reported_model_id not in (
        cell.model.dataset_model_id,
        Path(cell.model.simulation_model_path).name,
    ):
        raise ValueError("simulation model identity disagrees with phase-1 mapping")

    execution = _mapping(payload["execution_provenance"], "execution provenance")
    expected_timing_status = performance_timing_status(
        expected_launch_policy,
        [cell.hardware.hardware_id],
        uses_development_protobridge=(
            (config.prediction.values["tensor_collective_backend"]
             == "protobridge"
             and config.prediction.values["tensor_parallel_size"] > 1)
            or config.prediction.values["deployment"] in (
                "af_disagg", "pd_disagg", "pd_af_disagg")),
    )
    if execution["performance_timing_status"] != expected_timing_status:
        raise ValueError("execution performance timing status mismatch")
    hardware_configs = _mapping(
        execution["hardware_configs"], "execution hardware configs"
    )
    prefill_hardware = _mapping(
        hardware_configs["prefill"], "prefill hardware provenance"
    )
    expected_hardware_path = (
        Path(repository_root).resolve() / cell.hardware.hardware_config_path
    ).resolve()
    if Path(prefill_hardware["path"]).resolve() != expected_hardware_path:
        raise ValueError("execution hardware config path mismatch")
    if hardware_configs["decode"] is not None:
        raise ValueError("colocated execution cannot carry decode hardware config")
    if tuple(hardware_configs["additional_colocated"]) != ():
        raise ValueError("phase-1 execution requires one colocated instance")
    if tuple(execution["loaded_hardware_ids"]) != (cell.hardware.hardware_id,):
        raise ValueError("execution provenance hardware IDs mismatch")
    if execution["prediction_method"] != config.prediction.values[
        "prediction_method"
    ]:
        raise ValueError("execution provenance prediction method mismatch")
    collective_backend = config.prediction.values[
        "tensor_collective_backend"]
    if (collective_backend == "protobridge"
            and config.prediction.values["tensor_parallel_size"] > 1):
        expected_external_predictor_inputs = {
            "protobridge": {
                "backend": "protobridge",
                "source_kind": "development_only",
                "profile_sha256_by_operation": {
                    "all_reduce": config.prediction.values[
                        "tensor_all_reduce_profile_sha256"],
                    "all_gather": config.prediction.values[
                        "tensor_all_gather_profile_sha256"],
                },
            },
        }
    elif collective_backend in (
            "analytical_roofline_v1", "protobridge"):
        expected_external_predictor_inputs = None
    else:
        raise AssertionError(
            "validated tensor collective backend became unreachable")
    if (execution["external_predictor_inputs"]
            != expected_external_predictor_inputs):
        raise ValueError(
            "execution ProtoBridge backend/profile provenance mismatch")
    launch_inputs = _mapping(execution["launch_inputs"], "launch inputs")
    if launch_inputs["launch_policy"] != expected_launch_policy:
        raise ValueError("execution provenance launch policy mismatch")
    runtime_inputs = _mapping(execution["runtime_inputs"], "runtime inputs")
    expert_routing = _mapping(
        runtime_inputs["expert_routing"], "expert routing input"
    )
    runtime_expectations = {
        "expert_routing": expected_expert_routing.manifest(),
        "attention_kv_layout": config.prediction.values["attention_kv_layout"],
        "attention_kv_distribution": config.prediction.values[
            "attention_kv_distribution"
        ],
        "graph_execution": cell.hardware.graph_execution.manifest(),
    }
    for field, expected in runtime_expectations.items():
        if runtime_inputs[field] != expected:
            raise ValueError(f"execution runtime input {field} mismatch")
    if config.prediction.values["expert_parallel_size"] > 1:
        routing_expectations = expected_expert_routing.manifest()
        for field, expected in routing_expectations.items():
            if expert_routing[field] != expected:
                raise ValueError(
                    f"simulation expert routing {field} mismatch"
                )
        routing_sha = _sha256(
            expert_routing["policy_descriptor_sha256"],
            "expert routing policy descriptor SHA256",
        )
        if workload_binding["expert_routing_policy_sha256"] != routing_sha:
            raise ValueError("workload binding expert routing SHA256 mismatch")
    else:
        if expert_routing["source_kind"] != "not_applicable":
            raise ValueError("dense compact-suite execution requires no routing")
        if workload_binding["expert_routing_policy_sha256"] is not None:
            raise ValueError("dense workload binding expert routing must be null")

    mtp = _mapping(payload["mtp_scenario"], "MTP scenario")
    mtp_expectations = {
        "num_speculative_tokens": 0,
        "acceptance_policy": "disabled",
        "fixed_accepted_tokens": 0,
        "acceptance_source_id": "disabled",
        "acceptance_replay": None,
        "validation_status": "disabled",
    }
    for field, expected in mtp_expectations.items():
        if type(mtp[field]) is not type(expected) or mtp[field] != expected:
            raise ValueError(f"simulation MTP scenario {field} mismatch")

    repository = validate_execution_repository_provenance(execution, require_unchanged=True)
    if repository["git_commit"] != source_commit:
        raise ValueError(
            "execution repository commit does not match evaluation source commit"
        )

    scheduler_map = _mapping(payload["scheduler_runtime"], "scheduler runtime")
    if set(scheduler_map) != {"main"}:
        raise ValueError("phase-1 scheduler runtime must contain only main")
    capacity = (
        _validate_analytical_capacity(
            repository_root, payload["kv_capacity"], config, cell)
        if debug_num_kv_blocks is None
        else _validate_debug_capacity(
            repository_root, payload["kv_capacity"], config, cell,
            debug_num_kv_blocks)
    )
    scheduler, scheduler_sha = _validate_scheduler_runtime(
        repository_root, scheduler_map["main"], config, cell,
        capacity["num_kv_blocks"])

    artifact_summary = _validate_execution_artifact(
        payload, expected_timing_status
    )
    scheduler_progress = validate_scheduler_progress_manifest(
        payload["scheduler_progress"])
    expected_time_axis = {
        "device_only": "predicted_device_only_virtual_time",
        "device_plus_launch": "predicted_device_plus_launch_virtual_time",
        "device_plus_launch_plus_scheduler": (
            "predicted_device_plus_launch_plus_scheduler_virtual_time"),
    }[expected_serving_timing_mode]
    if scheduler_progress["time_axis"] != expected_time_axis:
        raise ValueError("Scheduler progress time axis disagrees with timing mode")
    if scheduler_progress["summary"]["step_count"] != artifact_summary[
            "counts"]["schedule_steps"]:
        raise ValueError(
            "scheduler progress step count disagrees with execution artifact")
    if debug_num_kv_blocks is not None:
        accuracy_claim_status = "debug_capacity_not_for_accuracy"
    elif expected_timing_status == PROJECTED_PERFORMANCE_TIMING_STATUS:
        accuracy_claim_status = "supported_for_accuracy_interpretation"
    elif expected_timing_status in (
            UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
            DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS):
        accuracy_claim_status = "unsupported_development_only"
    else:
        raise AssertionError("canonical timing status classification is incomplete")

    return {
        "validation_status": "validated_current_forward",
        "model_identity": {
            "dataset_model_id": cell.model.dataset_model_id,
            "simulation_model_basename": Path(
                cell.model.simulation_model_path
            ).name,
            "reported_model_id": reported_model_id,
            "config_sha256": cell.model.model_config_sha256,
            "binding": "explicit_model_path_alias_and_config_sha256",
        },
        "hardware_config_sha256": cell.hardware.hardware_config_sha256,
        "normalized_request_stream_sha256": normalized_request_sha,
        "runtime_inputs": runtime_expectations,
        "expert_routing": dict(expert_routing),
        "scheduler_runtime": scheduler,
        "kv_capacity": capacity,
        "main_chain_status": "executed",
        "performance_timing_status": expected_timing_status,
        "accuracy_claim_status": accuracy_claim_status,
        "execution_artifact": artifact_summary,
        "scheduler_progress": scheduler_progress,
        "midlayer_wall_time": midlayer_wall_time,
        "execution_mode": expected_execution_mode,
        "execution_cache_stats": execution_cache_stats,
        "execution_provenance_sha256": _canonical_json_sha256(execution),
        "workload_assumption_binding_sha256": _canonical_json_sha256(
            workload_binding
        ),
        "scheduler_runtime_sha256": scheduler_sha,
    }


# Each hardware declares which error messages its frozen phase-1 contract
# accepts as a *known* launch gap.  The two functions below differ only in
# which error class carries the message, so one engine reads this table and
# each caller keeps its own error class and unknown-hardware message.
#
def _is_known_metax_vector_gap(message: str) -> bool:
    """Return whether ``message`` is the MetaX Vector unsupported-for-'<op>' gap."""

    if not (message.startswith(_KNOWN_METAX_VECTOR_PREFIX)
            and message.endswith(_KNOWN_METAX_VECTOR_SUFFIX)):
        return False
    operation = message[
        len(_KNOWN_METAX_VECTOR_PREFIX):-len(_KNOWN_METAX_VECTOR_SUFFIX)
    ]
    return (len(operation) >= 3 and operation[0] == "'"
            and operation[-1] == "'")


# The selector-gap message each hardware's launch selector is allowed to raise
# on a case outside its characterized corpus.  A hardware absent from this map
# has no permitted gap, so its exception stays a failure.  The pattern must
# match the selector's own wording exactly; it is evidence consumed at the
# evaluation boundary, never a fallback for a missing selection.
# ``matchers`` are evaluated in order; the first one that accepts the message
# returns it.  A matcher is the message itself (exact equality), a compiled
# pattern (full match), or a callable returning whether the message is the
# declared gap.
_LAUNCH_GAP_CONTRACT = {
    "nvidia_a800": {
        "not_implemented": {
            "matchers": (_KNOWN_A800_GEMM_LAUNCH_GAP, _KNOWN_A800_ADD_RMS_QUEUED_SOURCE_GAP),
        },
        "value_error": {"matchers": (_KNOWN_A800_FA2_COMBINE_GAP,)},
    },
    "nvidia_h200": {
        "not_implemented": None,
        "value_error": {"matchers": ()},
    },
    "metax_c500": {
        "not_implemented": {
            "matchers": (
                _KNOWN_METAX_GEMM_LAUNCH_GAP,
                _KNOWN_METAX_FLASH_ATTN_GAP,
                _is_known_metax_vector_gap,
                _KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS["metax_c500"],
            ),
        },
        "value_error": {"matchers": ()},
    },
    "ascend_910b": {
        "not_implemented": {
            "matchers": (
                _KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS["ascend_910b"],
            ),
        },
        "value_error": {"matchers": ()},
    },
}


def _known_not_implemented_launch_gap_reason(
    error: NotImplementedError,
    hardware_id: str,
) -> str | None:
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    return _declared_launch_gap(
        error.args[0], hardware_id, "not_implemented",
        unknown_message=f"unknown compact-suite hardware: {hardware_id}")


def _known_value_error_launch_gap_reason(
    error: ValueError,
    hardware_id: str,
) -> str | None:
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    return _declared_launch_gap(
        error.args[0], hardware_id, "value_error",
        unknown_message=f"unknown compact-suite hardware: {hardware_id}")


def _declared_launch_gap(
    message: str, hardware_id: str, scope: str, *, unknown_message: str,
) -> str | None:
    """Return ``message`` when the declared contract names it as a gap.

    ``None`` in the row means the hardware has no gaps of this error class:
    the message is not a declared gap.  A hardware id outside the table is a
    contract violation, not an absent gap.
    """

    if hardware_id not in _LAUNCH_GAP_CONTRACT:
        raise ValueError(unknown_message)
    entry = _LAUNCH_GAP_CONTRACT[hardware_id][scope]
    if entry is None:
        return None
    for matcher in entry["matchers"]:
        if callable(matcher):
            if matcher(message):
                return message
        elif hasattr(matcher, "fullmatch"):
            if matcher.fullmatch(message) is not None:
                return message
        elif message == matcher:
            return message
    return None


def _process_peak_rss_bytes() -> int:
    return max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    ) * 1024


def _validate_execution_mode(execution_mode: object) -> str:
    if execution_mode not in ("accelerated", "reference", "fast"):
        raise ValueError(
            "execution_mode must be 'accelerated', 'reference', or 'fast'")
    return str(execution_mode)
