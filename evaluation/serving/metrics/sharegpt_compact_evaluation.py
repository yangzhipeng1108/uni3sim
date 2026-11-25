"""First-phase ShareGPT evaluation against validated compact cases.

The compact schema remains owned by :mod:`dataset_pipeline`.  This module only
accepts objects that the CLI has already validated and never imports that
owner.  Observations are consumed only after the current production Forward
has returned. Persisted request-level values are limited to synthetic request
indexes and paired latency scalars needed for evaluation plots.

This module is the compatibility re-export layer for the historical
import surface; the implementation lives in the ``compact_*`` sibling
modules of this package.
"""

from evaluation.json_shape import require_object_value as _mapping

from evaluation.serving.metrics.compact_shape_validators import (
    _SHA256,
    _unique_json_object,
    _canonical_json_sha256,
    _confined_path,
    _relative_path,
    _exact_fields,
    _sequence,
    _string,
    _sha256,
    _integer,
    _finite_number,
)

from evaluation.serving.metrics.compact_case_contract import (
    _GIT_COMMIT,
    _SUITE_ID,
    _CASE_ID,
    _MODEL_FIELDS,
    _HARDWARE_FIELDS,
    _GRAPH_HARDWARE_FIELDS,
    _GRAPH_EXECUTION_FIELDS,
    _PREDICTION_FIELDS,
    _CONFIG_FIELDS,
    _CELL_FIELDS,
    _CANONICAL_MODEL_SPECS,
    _MODEL_CONFIG_ALTERNATES,
    _LEGACY_MODEL_IDS,
    _EXPECTED_MODELS,
    _EXPECTED_HARDWARE,
    _OFFICIAL_HARDWARE,
    _EXPECTED_PREDICTION,
    OFFICIAL_METRICS_SUITE_SELECTIONS,
    NVIDIA_H200_SUITE_SELECTIONS,
    ValidatedCompactCase,
    scheduler_pipeline_depth_from_validated_case,
    scheduler_async_scheduling_from_validated_case,
    ModelEntry,
    HardwareEntry,
    PredictionContract,
    Phase1Config,
    Phase1Cell,
    load_phase1_config,
    phase1_cells,
    official_metrics_suite_runs,
    _official_metrics_case_keys,
    _matrix_case_keys,
    _canonical_cell_id,
    select_phase1_cells,
    compact_case_path,
    _validate_case_binding,
    _canonical_dataset_model_id,
    _validate_prediction_contract,
    _validate_case_id_contract,
    _model_config_path,
)

from evaluation.serving.metrics.compact_run_environment import (
    _RUN_ID,
    _HOST_PROFILE_VARIANTS,
    _HOST_PROFILE_DIRECTORIES,
    create_run_directory,
    resolve_repository_input,
    resolve_host_launch_profile_paths,
    resolve_scheduler_host_profile_paths,
    host_profile_variant_hardware,
    case_host_profile_paths,
    repository_source_commit,
)

from evaluation.serving.metrics.compact_execution_payload_validation import (
    EXECUTION_ARTIFACT_FIELDS,
    _STREAMING_EXECUTION_ARTIFACT_FIELDS,
    _EXECUTION_ARTIFACT_COUNT_FIELDS,
    _EXECUTION_ARTIFACT_HASH_METHOD,
    STREAMING_EXECUTION_ARTIFACT_HASH_METHODS,
    _MAIN_CHAIN,
    _validate_midlayer_wall_time,
    _validate_debug_capacity,
    _validate_analytical_capacity,
    _validate_scheduler_runtime,
    _runtime_value_matches_type,
    _validate_execution_artifact,
    _validate_streaming_execution_artifact,
    _artifact_tuple,
    _artifact_call_count,
    _streaming_canonical_json_sha256,
    _validate_execution_cache_stats,
    _expand_runtime_memory_policy_path,
)

from evaluation.serving.metrics.compact_metric_comparison import (
    _LATENCY_FIELDS,
    _THROUGHPUT_FIELDS,
    _timing_components,
    _validated_host_timing_payload,
    _host_error_attribution,
    _capture_runtime_comparison,
    _observed_ttft_breakdown,
    _observed_metrics,
    _compare_prediction,
    _unavailable_prediction_metrics,
    _canonical_latency_quantile_projection,
    _paired_error_summary,
    _distribution,
    _comparison,
)

from evaluation.serving.metrics.compact_case_evaluation import (
    _KNOWN_A800_GEMM_LAUNCH_GAP,
    _KNOWN_METAX_GEMM_LAUNCH_GAP,
    _KNOWN_METAX_VECTOR_PREFIX,
    _KNOWN_METAX_VECTOR_SUFFIX,
    _KNOWN_METAX_FLASH_ATTN_GAP,
    _KNOWN_A800_FA2_COMBINE_GAP,
    _KNOWN_A800_ADD_RMS_QUEUED_SOURCE_GAP,
    _KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS,
    _MOE_MODEL_IDS,
    _EXPERT_FREQUENCY_MANIFEST,
    _expected_expert_routing,
    _validated_precomputed_alignment,
    run_isolated_case,
    run_isolated_case_three_modes,
    build_simulation_argv,
    evaluate_validated_case,
    evaluate_validated_case_three_modes,
    _cell_context,
    _observed_scheduler_progress,
    validate_compact_alignment,
    _validate_completed_payload,
    _is_known_metax_vector_gap,
    _LAUNCH_GAP_CONTRACT,
    _known_not_implemented_launch_gap_reason,
    _known_value_error_launch_gap_reason,
    _declared_launch_gap,
    _process_peak_rss_bytes,
    _validate_execution_mode,
)

from evaluation.serving.metrics.compact_suite_driver import (
    run_official_metrics_evaluation,
    _official_alignment_inputs,
    _case_alignment_inputs,
    run_phase1_evaluation,
    _prepare_phase1_evaluation,
    _persist_phase1_evaluation,
    _evaluate_one_case,
    _run_parallel_cases,
    _run_parallel_case_jobs,
    write_phase1_results,
    _write_csv,
    _csv_row,
    _paired_csv_row,
    _render_markdown,
    _write_json,
    _validate_jobs,
)
from evaluation.serving.metrics.compact_suite_driver import run_virtual_forward_report
from evaluation.serving.metrics.compact_execution_payload_validation import validate_execution_repository_provenance
