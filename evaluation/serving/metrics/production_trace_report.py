"""Deterministic Markdown reporting for production-Trace sweeps.

This module is the compatibility re-export layer for the historical
import surface; the implementation lives in the sibling modules of
this package.
"""

from evaluation.serving.metrics.production_trace_report_shape import (
    _SWEEP_FIELDS,
    _AGGREGATE_SWEEP_FIELDS,
    _AGGREGATE_RESULT_FIELDS,
    _POINT_FIELDS,
    _SUMMARY_METRIC_FIELDS,
    _SOURCE_TRACE_FIELDS,
    _REPLAY_FIELDS,
    _EXECUTION_PROVENANCE_FIELDS,
    _EXPECTED_FIXED_STATUS,
    _ARRIVAL_TRANSFORM,
    _sequence,
    _exact_fields,
    _required_fields,
    _sha256,
    _string,
    _positive_number,
    _nonnegative_number,
    _nonnegative_integer,
    _nondecreasing_nonnegative,
    _format,
    _inline_json,
    _canonical_json_bytes,
    _code,
    _markdown_cell,
    _table_code,
)

from evaluation.serving.metrics.production_trace_execution_validation import (
    _validate_execution_and_host_timing,
    _validate_host_launch_profile_manifest,
    _validate_host_runtime_prepare_profile_manifest,
    _validate_host_timing_step,
    _validate_source_trace,
    _validate_execution_provenance,
    _validate_execution_artifact,
    _validate_modeled_kv_offload_artifact,
    _validate_modeled_kv_offload_v4,
)

from evaluation.serving.metrics.production_trace_result_validation import (
    _validate_latency_distributions,
    _validate_slo_goodput,
    _slo_contract,
    _validate_system_resources,
    _validate_host_kv_link_resources,
    _validate_host_kv_resource_binding,
    _validate_critical_path,
    _validate_contributions,
    _validate_sweep_analysis,
    _interactivity_tgs_contract,
)

from evaluation.serving.metrics.production_trace_report_rendering import (
    _render_aggregate_point,
    _render_timing_scope,
    _render_mtp_scenario,
    _render_execution_provenance,
    _render_sweep_analysis,
    _render_point,
    _render_latency_distributions,
    _render_slo,
    _render_resources,
    _render_critical_path,
)

from evaluation.serving.metrics.production_trace_sweep_report import (
    render_production_trace_sweep_report,
    write_production_trace_sweep_plots,
    _render_aggregate_production_trace_sweep_report,
    _validate_aggregate_point,
    _validate_point,
)
