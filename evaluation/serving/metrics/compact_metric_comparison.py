"""Observed-metric extraction and prediction comparison for validated
compact ShareGPT cases."""

from __future__ import annotations

from dataclasses import fields
import math
import statistics
from typing import Mapping, Sequence

from evaluation.serving.metrics.latency_distribution import (
    latency_distribution_manifest,
    linear_percentile,
)
from evaluation.serving.metrics.sharegpt_evaluation_report import STATISTICS
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
)
from src.serving.metrics.serving_metrics import RequestMetrics
from src.serving.host_timing import SERVING_TIMING_MODES

from evaluation.json_shape import require_object_value as _mapping

from evaluation.serving.metrics.compact_shape_validators import (
    _exact_fields,
    _sequence,
    _integer,
    _finite_number,
)
from evaluation.serving.metrics.compact_case_contract import (
    ValidatedCompactCase,
)


_LATENCY_FIELDS = {
    "ttft_ms": "ttft_s",
    "tpot_ms": "tpot_s",
    "e2e_latency_ms": "latency_s",
}
_THROUGHPUT_FIELDS = {
    "request_throughput_per_s": "request_throughput_per_s",
    "output_token_throughput_per_s": "output_token_throughput_per_s",
    "total_token_throughput_per_s": "token_throughput_per_s",
}


def _timing_components(
    device_only_available: bool,
    simulator_wall_time_seconds: float,
) -> dict[str, object]:
    """Describe prediction time axes without treating simulator CPU as latency."""

    if not isinstance(device_only_available, bool):
        raise TypeError("device-only availability must be boolean")
    wall_time = _finite_number(
        simulator_wall_time_seconds, "simulator wall time")
    if wall_time < 0.0:
        raise ValueError("simulator wall time must be non-negative")
    return {
        "schema_version": 1,
        "device_only": {
            "availability": (
                "available" if device_only_available else "unavailable"),
            "metrics_field": "metrics" if device_only_available else None,
            "time_axis": "predicted_device_only_virtual_time",
        },
        "cpu_plus_device": {
            "availability": "unavailable",
            "metrics_field": None,
            "cpu_source": None,
            "overlap_rule": None,
            "reason": (
                "no validated production critical-path CPU observation was "
                "supplied; simulator Python wall time is diagnostic only"
            ),
        },
        "simulator_wall_time_seconds": wall_time,
    }


def _validated_host_timing_payload(
    payload: Mapping[str, object],
    expected_mode: str,
    expected_hardware_id: str,
) -> dict[str, object]:
    if payload["serving_timing_mode"] != expected_mode:
        raise ValueError("simulation payload Serving timing mode mismatch")
    timing = _mapping(payload["host_timing"], "Host timing payload")
    _exact_fields(timing, {
        "schema_version", "timing_mode", "scheduler_measurement_boundary",
        "launch_measurement_boundary", "profiles", "runtime_prepare_profiles",
        "aggregate", "steps",
    }, "Host timing payload")
    if timing["schema_version"] != 1 or timing["timing_mode"] != expected_mode:
        raise ValueError("Host timing schema or mode mismatch")
    if timing["scheduler_measurement_boundary"] != (
            "in_scheduler_process_schedule_and_output_update_only"):
        raise ValueError("Host timing Scheduler boundary mismatch")
    if timing["launch_measurement_boundary"] != (
            "host_submit_only_no_device_sync"):
        raise ValueError("Host launch measurement boundary mismatch")
    profiles = _sequence(timing["profiles"], "Host launch profiles")
    expected_profile_count = {
        "device_only": 0,
        "device_plus_launch": 1,
        "device_plus_launch_plus_scheduler": 1,
    }[expected_mode]
    if len(profiles) != expected_profile_count:
        raise ValueError(
            "Host timing profile count disagrees with timing mode")
    if profiles:
        profile = _mapping(profiles[0], "Host launch profile")
        if profile["hardware_id"] != expected_hardware_id:
            raise ValueError("Host timing profile hardware mismatch")
    runtime_prepare_profiles = _sequence(
        timing["runtime_prepare_profiles"], "Host runtime prepare profiles")
    if len(runtime_prepare_profiles) > expected_profile_count:
        raise ValueError("Host runtime prepare profile count disagrees with timing mode")
    if runtime_prepare_profiles:
        profile = _mapping(
            runtime_prepare_profiles[0], "Host runtime prepare profile")
        _exact_fields(profile, {
            "hardware_id", "service_time_s_by_mechanism",
            "lane_service_time_s_by_mechanism", "profile_path", "profile_sha256",
        }, "Host runtime prepare profile")
        if profile["hardware_id"] != expected_hardware_id:
            raise ValueError("Host runtime prepare profile hardware mismatch")
    aggregate = _mapping(timing["aggregate"], "Host timing aggregate")
    expected_aggregate_fields = {
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
        "eager_dispatch_count", "graph_replay_count", "scheduler_cpu_s",
        "schedule_makespan_s", "step_count",
    }
    _exact_fields(aggregate, expected_aggregate_fields, "Host timing aggregate")
    for field in expected_aggregate_fields - {
            "eager_dispatch_count", "graph_replay_count", "step_count"}:
        if _finite_number(aggregate[field], field) < 0:
            raise ValueError(f"Host timing aggregate {field} must be non-negative")
    for field in ("eager_dispatch_count", "graph_replay_count", "step_count"):
        if _integer(aggregate[field], field) < 0:
            raise ValueError(f"Host timing aggregate {field} must be non-negative")
    if not math.isclose(
            aggregate["scheduler_cpu_s"],
            aggregate["scheduler_plan_cpu_s"]
            + aggregate["scheduler_update_cpu_s"],
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError("Host timing Scheduler aggregate is inconsistent")
    steps = _sequence(timing["steps"], "Host timing steps")
    if len(steps) != aggregate["step_count"] or not steps:
        raise ValueError("Host timing step count is inconsistent")
    return dict(timing)


def _host_error_attribution(
    mode_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Quantify each added Host component without treating it as fitted."""

    if set(mode_results) != set(SERVING_TIMING_MODES):
        raise ValueError("Host attribution requires all three timing modes")
    if any(
            mode_results[mode]["prediction_status"] != "completed"
            for mode in SERVING_TIMING_MODES):
        return {
            "schema_version": 1,
            "status": "unavailable_incomplete_prediction",
            "metrics": {},
        }

    def attribution(rows):
        device = _mapping(rows["device_only"], "device-only comparison")
        launch = _mapping(rows["device_plus_launch"], "launch comparison")
        scheduler = _mapping(
            rows["device_plus_launch_plus_scheduler"],
            "launch-plus-Scheduler comparison",
        )
        observed = _finite_number(device["observed"], "observed metric")
        if launch["observed"] != observed or scheduler["observed"] != observed:
            raise ValueError("timing-mode comparisons use different truth")
        predicted = {
            mode: _finite_number(rows[mode]["predicted"], f"{mode} prediction")
            for mode in SERVING_TIMING_MODES
        }
        errors = {mode: predicted[mode] - observed for mode in SERVING_TIMING_MODES}
        return {
            "observed": observed,
            "device_only": predicted["device_only"],
            "device_plus_launch": predicted["device_plus_launch"],
            "device_plus_launch_plus_scheduler": predicted[
                "device_plus_launch_plus_scheduler"],
            "launch_prediction_delta": (
                predicted["device_plus_launch"] - predicted["device_only"]),
            "scheduler_prediction_delta": (
                predicted["device_plus_launch_plus_scheduler"]
                - predicted["device_plus_launch"]),
            "launch_absolute_error_reduction": (
                abs(errors["device_only"]) - abs(errors["device_plus_launch"])),
            "scheduler_absolute_error_reduction": (
                abs(errors["device_plus_launch"])
                - abs(errors["device_plus_launch_plus_scheduler"])),
            "final_residual": errors["device_plus_launch_plus_scheduler"],
            "final_absolute_residual": abs(
                errors["device_plus_launch_plus_scheduler"]),
        }

    mode_metrics = {
        mode: _mapping(mode_results[mode]["metrics"], f"{mode} metrics")
        for mode in SERVING_TIMING_MODES
    }
    latency = {}
    for metric in _LATENCY_FIELDS:
        latency[metric] = {}
        for statistic in STATISTICS:
            latency[metric][statistic] = attribution({
                mode: _mapping(
                    _mapping(mode_metrics[mode]["latency"], "latency")[metric],
                    metric,
                )[statistic]
                for mode in SERVING_TIMING_MODES
            })
    throughput = {
        metric: attribution({
            mode: _mapping(
                mode_metrics[mode]["throughput"], "throughput")[metric]
            for mode in SERVING_TIMING_MODES
        })
        for metric in _THROUGHPUT_FIELDS
    }
    return {
        "schema_version": 1,
        "status": "available",
        "units": {"latency": "ms", "throughput": "per_second"},
        "latency": latency,
        "throughput": throughput,
    }


def _capture_runtime_comparison(
    capture: Mapping[str, object],
    prediction: Mapping[str, object],
    target_qps: int,
) -> Mapping[str, object]:
    comparison_specs = (
        ("dtype", "bfloat16", True),
        ("deployment", prediction["deployment"], True),
        ("tensor_parallel_size", prediction["tensor_parallel_size"], True),
        ("pipeline_parallel_size", prediction["pipeline_parallel_size"], True),
        ("expert_parallel_size", prediction["expert_parallel_size"], True),
        ("target_qps", float(target_qps), True),
        ("seed", None, False),
        ("scheduler_config_sha256", None, False),
        ("runtime_version", VLLM_SCHEDULER_VERSION, True),
        ("source_commit", VLLM_SCHEDULER_SOURCE_COMMIT, True),
        (
            "capture_enable_prefix_caching",
            prediction["enable_prefix_caching"],
            True,
        ),
    )
    fields = {}
    for field, prediction_value, comparable in comparison_specs:
        capture_value = capture[field]
        if capture_value is None:
            field_status = "unknown"
        elif not comparable:
            field_status = "known"
        elif capture_value == prediction_value:
            field_status = "equal"
        elif field == "dtype" and {
            capture_value, prediction_value
        } == {"float16", "bfloat16"}:
            field_status = "performance_equivalent"
        else:
            field_status = "mismatch"
        fields[field] = {
            "status": field_status,
            "capture": capture_value,
            "prediction": prediction_value,
        }
    if any(entry["status"] == "mismatch" for entry in fields.values()):
        status = "mismatch"
    else:
        status = "incomplete_capture_contract"
    return {
        "status": status,
        "fields": fields,
        "interpretation": "current_prediction_not_capture_replay",
    }


def _observed_ttft_breakdown(
    case: ValidatedCompactCase,
    by_request: Mapping[int, Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Current compact observations do not retain a common TTFT clock.

    Keep the historical evaluation import surface; request membership and
    relative Scheduler timestamps alone cannot identify either stage.
    """
    return None


def _ttft_breakdown_unavailable_reason(case: ValidatedCompactCase) -> str:
    """Explain the evidence missing from the current compact clock contract.

    Scheduler timestamps start at the first schedule; canonical arrivals use
    the client's planned benchmark clock. Neither the absolute clock offset
    nor the client's actual per-request TTFT start survives normalization.
    Request membership alone therefore cannot make a measured stage split.
    """
    if case.scheduler_steps is None:
        return "canonical case has no observed first-Scheduler-step evidence"
    return (
        "canonical Scheduler timestamps start at the first schedule; absolute "
        "alignment to the client clock and actual per-request TTFT starts "
        "are unavailable, so measured Wait/Processing cannot be separated"
    )


def _observed_metrics(case: ValidatedCompactCase) -> Mapping[str, object]:
    if case.metrics is None:
        raise ValueError("compact case has no metrics")
    metrics = _mapping(case.metrics, "alignment metrics")
    per_request = _sequence(metrics["per_request"], "observed per_request")
    summary = _mapping(metrics["summary"], "observed summary")
    expected_indices = set(range(len(case.requests)))
    actual_indices = {
        _integer(_mapping(item, "observed request")["request_idx"], "request_idx")
        for item in per_request
    }
    if actual_indices != expected_indices or len(per_request) != len(case.requests):
        raise ValueError("observed request_idx set disagrees with requests.csv")
    latencies = {}
    by_request = {}
    for item in per_request:
        record = _mapping(item, "observed request")
        request_idx = _integer(record["request_idx"], "request_idx")
        by_request[request_idx] = {
            name: record[name] for name in _LATENCY_FIELDS
        }
    for observed_name in _LATENCY_FIELDS:
        values = []
        for item in per_request:
            record = _mapping(item, "observed request")
            value = record[observed_name]
            if value is None:
                raise ValueError(
                    f"first-phase observed metric {observed_name} cannot be null"
                )
            values.append(_finite_number(value, observed_name))
        latencies[observed_name] = _distribution(values)
    throughputs = {
        name: _finite_number(summary[name], name) for name in _THROUGHPUT_FIELDS
    }
    result = {
        "latencies": latencies,
        "throughputs": throughputs,
        "by_request": by_request,
    }
    breakdown = _observed_ttft_breakdown(case, by_request)
    if breakdown is not None:
        result["ttft_breakdown"] = breakdown
    return result


def _compare_prediction(
    observed: Mapping[str, object],
    payload_value: object,
    request_count: int,
) -> Mapping[str, object]:
    payload = _mapping(payload_value, "simulation payload")
    if payload["backend"] != "vllm" or payload["deployment"] != "colocated":
        raise ValueError("phase-1 prediction returned the wrong runtime contract")
    metrics = _mapping(payload["metrics"], "simulation metrics")
    predicted_requests = _mapping(metrics["requests"], "predicted requests")
    expected_ids = {f"request-{idx:012d}" for idx in range(request_count)}
    if set(predicted_requests) != expected_ids:
        raise ValueError("predicted request IDs disagree with canonical request_idx")
    predicted_distributions = {}
    paired_errors = {}
    paired_latency = {}
    observed_by_request = _mapping(
        observed["by_request"], "observed requests by index"
    )
    for observed_name, predicted_name in _LATENCY_FIELDS.items():
        values = []
        absolute_errors = []
        percentage_errors = []
        paired_values = []
        for request_idx in range(request_count):
            request_id = f"request-{request_idx:012d}"
            record = _mapping(predicted_requests[request_id], "predicted request")
            value = record[predicted_name]
            if value is None:
                raise ValueError(
                    f"first-phase predicted metric {predicted_name} cannot be null"
                )
            predicted_ms = 1000.0 * _finite_number(value, predicted_name)
            values.append(predicted_ms)
            observed_record = _mapping(
                observed_by_request[request_idx], "observed request metric"
            )
            observed_ms = _finite_number(
                observed_record[observed_name], observed_name
            )
            paired_values.append({
                "request_idx": request_idx,
                "measured_ms": observed_ms,
                "predicted_ms": predicted_ms,
            })
            absolute_error = abs(predicted_ms - observed_ms)
            absolute_errors.append(absolute_error)
            if observed_ms != 0:
                percentage_errors.append(absolute_error / abs(observed_ms) * 100.0)
        predicted_distributions[observed_name] = _distribution(values)
        paired_errors[observed_name] = _paired_error_summary(
            absolute_errors,
            percentage_errors,
            request_count,
        )
        paired_latency[observed_name] = paired_values

    observed_latency = _mapping(observed["latencies"], "observed latencies")
    latency_comparisons = {}
    for metric_name in _LATENCY_FIELDS:
        observed_distribution = _mapping(
            observed_latency[metric_name], f"observed {metric_name}"
        )
        predicted_distribution = predicted_distributions[metric_name]
        latency_comparisons[metric_name] = {
            statistic: _comparison(
                observed_distribution[statistic],
                predicted_distribution[statistic],
            )
            for statistic in STATISTICS
        }

    observed_throughputs = _mapping(
        observed["throughputs"], "observed throughputs"
    )
    throughput_comparisons = {}
    for observed_name, predicted_name in _THROUGHPUT_FIELDS.items():
        throughput_comparisons[observed_name] = _comparison(
            observed_throughputs[observed_name],
            _finite_number(metrics[predicted_name], predicted_name),
        )
    result = {
        "latency": latency_comparisons,
        "latency_quantiles_p0_p100": {
            "comparison_kind": "measured_vs_predicted",
            "measured": _canonical_latency_quantile_projection(
                observed_by_request,
                request_count,
                predicted=False,
            ),
            "predicted": _canonical_latency_quantile_projection(
                predicted_requests,
                request_count,
                predicted=True,
            ),
        },
        "paired_request_error": paired_errors,
        "paired_request_latency": {
            "schema_version": 1,
            "comparison_kind": "measured_vs_predicted_by_request_idx",
            "request_count": request_count,
            "metrics": paired_latency,
        },
        "throughput": throughput_comparisons,
        "ttft_breakdown": None,
    }
    if "ttft_breakdown" in observed:
        arrivals = _sequence(
            payload["request_arrivals_s"], "predicted request arrivals"
        )
        arrival_by_id = {}
        for arrival_value in arrivals:
            arrival = _mapping(arrival_value, "predicted request arrival")
            request_id = str(arrival["request_id"])
            if request_id in arrival_by_id:
                raise ValueError("predicted request arrival IDs must be unique")
            arrival_by_id[request_id] = _finite_number(
                arrival["arrival_time_s"], "predicted request arrival time"
            )
        if set(arrival_by_id) != expected_ids:
            raise ValueError("predicted request arrivals disagree with requests")
        observed_breakdown_manifest = _mapping(
            observed["ttft_breakdown"], "observed TTFT breakdown"
        )
        observed_breakdown = _mapping(
            observed_breakdown_manifest["records"],
            "observed TTFT breakdown records",
        )
        records = []
        for request_idx in range(request_count):
            request_id = f"request-{request_idx:012d}"
            observed_record = _mapping(
                observed_breakdown[request_idx], "observed TTFT breakdown record"
            )
            predicted_record = _mapping(
                predicted_requests[request_id], "predicted request"
            )
            predicted_first_schedule_s = _finite_number(
                predicted_record["first_schedule_start_s"],
                "predicted first schedule start",
            )
            predicted_wait_ms = 1000.0 * (
                predicted_first_schedule_s - arrival_by_id[request_id]
            )
            if predicted_wait_ms < -1e-6:
                raise ValueError(
                    "predicted first schedule starts before request arrival"
                )
            predicted_wait_ms = max(0.0, predicted_wait_ms)
            predicted_ttft_ms = 1000.0 * _finite_number(
                predicted_record["ttft_s"], "predicted TTFT"
            )
            predicted_processing_ms = predicted_ttft_ms - predicted_wait_ms
            if predicted_processing_ms < -1e-6:
                raise ValueError(
                    "predicted first-schedule wait exceeds predicted TTFT"
                )
            measured_wait_ms = _finite_number(
                observed_record["wait_ms"], "measured first-schedule wait"
            )
            measured_processing_ms = _finite_number(
                observed_record["processing_ms"], "measured TTFT processing"
            )
            records.append({
                "request_idx": request_idx,
                "measured_wait_ms": measured_wait_ms,
                "measured_processing_ms": measured_processing_ms,
                "predicted_wait_ms": predicted_wait_ms,
                "predicted_processing_ms": max(0.0, predicted_processing_ms),
            })
        result["ttft_breakdown"] = {
            "schema_version": 1,
            "boundary": "request_arrival_to_first_schedule_to_first_token",
            "processing_scope": (
                "first_schedule_to_client_visible_first_token"
            ),
            "measured_arrival_basis": observed_breakdown_manifest[
                "arrival_basis"
            ],
            "measured_negative_wait_clamp_count": _integer(
                observed_breakdown_manifest["negative_wait_clamp_count"],
                "measured negative wait clamp count",
            ),
            "measured_minimum_raw_wait_ms": _finite_number(
                observed_breakdown_manifest["minimum_raw_wait_ms"],
                "measured minimum raw wait",
            ),
            "request_count": request_count,
            "records": records,
        }
    return result


def _unavailable_prediction_metrics(
    observed: Mapping[str, object],
) -> Mapping[str, object]:
    observed_latency = _mapping(observed["latencies"], "observed latencies")
    latency = {}
    for metric_name in _LATENCY_FIELDS:
        distribution = _mapping(observed_latency[metric_name], metric_name)
        latency[metric_name] = {
            statistic: _comparison(distribution[statistic], None)
            for statistic in STATISTICS
        }
    observed_throughput = _mapping(
        observed["throughputs"], "observed throughputs"
    )
    throughput = {
        metric_name: _comparison(observed_throughput[metric_name], None)
        for metric_name in _THROUGHPUT_FIELDS
    }
    paired = {
        metric_name: {
            "request_count": len(_mapping(
                observed["by_request"], "observed requests by index"
            )),
            "absolute_percentage_error_eligible_count": None,
            "mae_ms": None,
            "mape_percent": None,
            "absolute_error_ms": {"p50": None, "p90": None, "p99": None},
            "absolute_percentage_error": {
                "p50": None,
                "p90": None,
                "p99": None,
            },
        }
        for metric_name in _LATENCY_FIELDS
    }
    observed_by_request = _mapping(
        observed["by_request"], "observed requests by index"
    )
    return {
        "latency": latency,
        "latency_quantiles_p0_p100": {
            "comparison_kind": "measured_vs_predicted",
            "measured": _canonical_latency_quantile_projection(
                observed_by_request,
                len(observed_by_request),
                predicted=False,
            ),
            "predicted": None,
        },
        "paired_request_error": paired,
        "paired_request_latency": None,
        "ttft_breakdown": None,
        "throughput": throughput,
    }


def _canonical_latency_quantile_projection(
    records_value: object,
    request_count: int,
    *,
    predicted: bool,
) -> Mapping[str, object]:
    records = _mapping(records_value, "request latency records")
    if len(records) != request_count:
        raise ValueError("request latency records do not cover every request")
    request_metrics = {}
    for request_idx in range(request_count):
        request_id = f"request-{request_idx:012d}"
        key = request_id if predicted else request_idx
        record = _mapping(records[key], "request latency record")
        if predicted:
            ttft_s = _finite_number(record["ttft_s"], "predicted ttft_s")
            tpot_s = _finite_number(record["tpot_s"], "predicted tpot_s")
            latency_s = _finite_number(record["latency_s"], "predicted latency_s")
        else:
            ttft_s = _finite_number(record["ttft_ms"], "measured ttft_ms") / 1000.0
            tpot_s = _finite_number(record["tpot_ms"], "measured tpot_ms") / 1000.0
            latency_s = (
                _finite_number(
                    record["e2e_latency_ms"], "measured e2e_latency_ms"
                )
                / 1000.0
            )
        request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            latency_s=latency_s,
        )
    manifest = latency_distribution_manifest(request_metrics)
    return {
        "schema_version": 1,
        "source_schema_version": manifest["schema_version"],
        "quantile_representation": manifest["quantile_representation"],
        "percentile_method": manifest["percentile_method"],
        **{
            field: {
                "availability": manifest[field]["availability"],
                "eligible_request_count": manifest[field][
                    "eligible_request_count"
                ],
                "excluded_request_count": manifest[field]["excluded_request_count"],
                "exclusion_reason": manifest[field]["exclusion_reason"],
                "quantiles": manifest[field]["quantiles"],
            }
            for field in ("ttft_s", "tpot_s", "latency_s")
        },
    }


def _paired_error_summary(
    absolute_errors: Sequence[float],
    percentage_errors: Sequence[float],
    request_count: int,
) -> Mapping[str, object]:
    if len(absolute_errors) != request_count:
        raise ValueError("paired latency errors do not cover every request")
    absolute_distribution = _distribution(absolute_errors)
    if percentage_errors:
        percentage_distribution = _distribution(percentage_errors)
        mape = percentage_distribution["mean"]
        percentage_quantiles = {
            statistic: percentage_distribution[statistic]
            for statistic in ("p50", "p90", "p99")
        }
    else:
        mape = None
        percentage_quantiles = {"p50": None, "p90": None, "p99": None}
    return {
        "request_count": request_count,
        "absolute_percentage_error_eligible_count": len(percentage_errors),
        "mae_ms": absolute_distribution["mean"],
        "mape_percent": mape,
        "absolute_error_ms": {
            statistic: absolute_distribution[statistic]
            for statistic in ("p50", "p90", "p99")
        },
        "absolute_percentage_error": percentage_quantiles,
    }


def _distribution(values: Sequence[float]) -> Mapping[str, float]:
    if not values:
        raise ValueError("metric distribution cannot be empty")
    sorted_values = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "p50": linear_percentile(sorted_values, 50.0),
        "p90": linear_percentile(sorted_values, 90.0),
        "p99": linear_percentile(sorted_values, 99.0),
    }


def _comparison(observed: object, predicted: object | None) -> Mapping[str, object]:
    observed_value = _finite_number(observed, "observed metric")
    if predicted is None:
        return {
            "observed": observed_value,
            "predicted": None,
            "error": None,
            "absolute_percentage_error": None,
        }
    predicted_value = _finite_number(predicted, "predicted metric")
    error = predicted_value - observed_value
    ape = None if observed_value == 0 else abs(error) / abs(observed_value) * 100.0
    return {
        "observed": observed_value,
        "predicted": predicted_value,
        "error": error,
        "absolute_percentage_error": ape,
    }
