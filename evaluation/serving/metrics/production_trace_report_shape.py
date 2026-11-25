"""Payload shape for Production Trace sweep reports: field contracts
and the scalar / Markdown format helpers shared by validators and
renderers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from numbers import Real
import re


_SWEEP_FIELDS = {
    "schema_version",
    "kind",
    "interpretation",
    "interactivity_tgs",
    "peak_selection",
    "saturation_analysis",
    "scale_order",
    "arrival_transform",
    "point_count",
    "arrival_rate_scales",
    "source_trace",
    "execution_provenance",
    "points",
}
_AGGREGATE_SWEEP_FIELDS = _SWEEP_FIELDS | {"persistence"}
_AGGREGATE_RESULT_FIELDS = {
    "schema_version",
    "kind",
    "persistence",
    "version",
    "backend",
    "deployment_id",
    "hardware_ids",
    "model_compute_dtype",
    "model_config_provenance",
    "deployment",
    "deployment_plan",
    "routing_policy",
    "kv_handoff_mode",
    "kv_capacity",
    "kv_lifecycle_replay",
    "launch_policy",
    "prediction_method",
    "execution_mode",
    "execution_cache_stats",
    "execution_parallelism",
    "serving_timing_mode",
    "host_timing",
    "performance_timing_status",
    "execution_provenance",
    "workload_assumption_binding",
    "main_chain_status",
    "execution_artifact",
    "scheduler_progress",
    "scheduler_runtime",
    "mtp_scenario",
    "mtp_acceptance_outcomes",
    "mtp_pd_handoff_contract",
    "arrival_process",
    "num_steps",
    "num_transfers",
    "num_kernel_calls",
    "schedule_end_time_s",
    "simulation_wall_time_s",
    "evaluation_wall_time_breakdown_s",
    "process_peak_rss_bytes",
    "metrics",
    "request_source",
}
_AGGREGATE_RESULT_TIMING_FIELDS = frozenset({
    "simulation_started_monotonic_s",
    "simulation_finished_monotonic_s",
})
_POINT_FIELDS = {
    "arrival_rate_scale",
    "request_stream_sha256",
    "validation_status",
    "metrics",
    "result",
}
_SUMMARY_METRIC_FIELDS = {
    "request_throughput_per_s",
    "output_token_throughput_per_s",
    "token_throughput_per_s",
    "latency_distributions",
    "slo_goodput",
}
_SOURCE_TRACE_FIELDS = {
    "schema_version",
    "successful_only",
    "source_path",
    "source_sha256",
    "source_format",
    "field_mapping",
    "input_time_unit",
    "normalized_time_unit",
    "request_id_policy",
    "equal_arrival_order",
    "invalid_row_policy",
    "unmapped_field_policy",
    "prefix_oracle",
    "scheduler_prefix_alignment_owner",
    "success_filter",
    "statistics",
    "normalized_request_stream_sha256",
}
_REPLAY_FIELDS = {
    "schema_version",
    "kind",
    "arrival_rate_scale",
    "arrival_transform",
    "arrival_anchor_s",
    "request_count",
    "arrival_time_total_s",
    "arrival_time_min_s",
    "arrival_time_max_s",
    "request_distribution_policy",
    "interpretation",
    "request_stream_sha256",
}
_EXECUTION_PROVENANCE_FIELDS = {
    "schema_version",
    "repository",
    "hardware_configs",
    "loaded_hardware_ids",
    "performance_timing_status",
    "runtime_inputs",
    "launch_inputs",
    "microbenchmark_inputs",
    "prediction_method",
    "external_predictor_inputs",
}
_EXPECTED_FIXED_STATUS = {
    "trace_replay": "validated_conservation",
    "scheduler_control_plane": "native_pinned_vllm",
}
_ARRIVAL_TRANSFORM = "t_anchor + (t - t_anchor) / arrival_rate_scale"


def _sequence(value: object, label: str) -> Sequence[object]:
    if (not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        raise TypeError(f"{label} must be a sequence")
    return value


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    supplied = set(value)
    if supplied != expected:
        raise ValueError(
            f"{label} fields disagree with schema: "
            f"missing={sorted(expected - supplied)}, "
            f"unexpected={sorted(supplied - expected)}")


def _required_fields(
    value: Mapping[str, object],
    required: set[str],
    label: str,
) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")


def _sha256(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be lowercase SHA256 hex")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_number(value: object, label: str) -> float:
    normalized = _nonnegative_number(value, label)
    if normalized <= 0:
        raise ValueError(f"{label} must be positive")
    return normalized


def _nonnegative_number(value: object, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nondecreasing_nonnegative(values: Sequence[object], label: str) -> None:
    normalized = tuple(_nonnegative_number(value, label) for value in values)
    if tuple(sorted(normalized)) != normalized:
        raise ValueError(f"{label} must be nondecreasing")


def _format(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ValueError("report cannot format a non-finite number")
        return format(value, ".12g")
    return str(value)


def _inline_json(value: object) -> str:
    try:
        return _code(json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ))
    except (TypeError, ValueError) as error:
        raise ValueError("report field is not canonical JSON") from error


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("report field is not canonical JSON") from error


def _code(value: object) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    backtick_run_lengths = tuple(
        len(match.group(0)) for match in re.finditer(r"`+", text)
    )
    longest_backtick_run = (
        max(backtick_run_lengths) if backtick_run_lengths else 0)
    delimiter = "`" * (longest_backtick_run + 1)
    needs_padding = (
        text.startswith(("`", " ")) or text.endswith(("`", " ")))
    padding = " " if needs_padding else ""
    return f"{delimiter}{padding}{text}{padding}{delimiter}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table_code(value: object) -> str:
    if value is None:
        return "N/A"
    return _markdown_cell(_code(value))
