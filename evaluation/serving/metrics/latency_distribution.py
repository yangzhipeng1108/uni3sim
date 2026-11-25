"""Canonical empirical latency CDF and P0-P100 quantile reporting."""

from __future__ import annotations

import math
from numbers import Real
from typing import TYPE_CHECKING, Mapping, Sequence

if TYPE_CHECKING:
    from src.serving.metrics.serving_metrics import RequestMetrics


_PERCENTILES = tuple(range(101))
_PERCENTILE_METHOD = "linear_interpolation_rank_n_minus_1_v1"
_CDF_METHOD = "empirical_right_continuous_less_than_or_equal_v1"


def linear_percentile(
    sorted_values: Sequence[float],
    percentile: float,
) -> float:
    """Return the linearly interpolated percentile of sorted finite values."""

    if not sorted_values:
        raise ValueError("cannot compute a percentile for empty values")
    normalized = []
    for value in sorted_values:
        if (isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)):
            raise ValueError("percentile values must be finite numbers")
        normalized.append(float(value))
    if normalized != sorted(normalized):
        raise ValueError("percentile values must be sorted")
    if (isinstance(percentile, bool)
            or not isinstance(percentile, Real)
            or not math.isfinite(percentile)
            or percentile < 0
            or percentile > 100):
        raise ValueError("percentile must be finite and within [0, 100]")
    if len(normalized) == 1:
        return normalized[0]
    rank = (len(normalized) - 1) * float(percentile) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(normalized) - 1)
    weight = rank - lower
    lower_value = normalized[lower]
    upper_value = normalized[upper]
    if lower_value == upper_value:
        return lower_value
    delta = upper_value - lower_value
    if math.isfinite(delta):
        interpolated = lower_value + delta * weight
    else:
        interpolated = (
            lower_value * (1.0 - weight) + upper_value * weight)
    return min(upper_value, max(lower_value, interpolated))


def latency_distribution_manifest(
    request_metrics: Mapping[str, RequestMetrics],
) -> dict[str, object]:
    """Build TTFT/TPOT/E2E empirical CDFs and P0-P100 quantile points."""

    from src.serving.metrics.serving_metrics import RequestMetrics

    if not request_metrics:
        raise ValueError("latency CDF requires at least one request")
    ttft_values = []
    tpot_values = []
    latency_values = []
    for request_id in sorted(request_metrics):
        metric = request_metrics[request_id]
        if not isinstance(metric, RequestMetrics):
            raise TypeError(
                "latency CDF values must be RequestMetrics objects")
        if metric.request_id != request_id:
            raise ValueError(
                "latency distribution request key and RequestMetrics ID "
                "disagree")
        if ((metric.ttft_s is None and metric.tpot_s is not None)
                or (metric.ttft_s is not None and metric.tpot_s is None)):
            raise ValueError(
                "latency distribution requires TTFT and TPOT to be "
                "available or unavailable together")
        ttft_values.append(metric.ttft_s)
        tpot_values.append(metric.tpot_s)
        latency_values.append(metric.latency_s)
    return {
        "schema_version": 2,
        "quantile_representation": (
            "linear_interpolated_quantile_points_p0_p100"),
        "percentile_method": _PERCENTILE_METHOD,
        "cdf_representation": "empirical_right_continuous_step_points",
        "cdf_method": _CDF_METHOD,
        "ttft_s": _optional_latency_distribution(
            ttft_values, "requests_without_output_tokens"),
        "tpot_s": _optional_latency_distribution(
            tpot_values, "requests_without_output_tokens"),
        "latency_s": _optional_latency_distribution(
            latency_values, "none_all_successful_requests_are_eligible"),
    }


def _optional_latency_distribution(
    values: Sequence[float | None],
    exclusion_reason: str,
) -> dict[str, object]:
    if not isinstance(exclusion_reason, str) or not exclusion_reason:
        raise ValueError("latency exclusion_reason must be non-empty")
    eligible = []
    excluded_count = 0
    for value in values:
        if value is None:
            excluded_count += 1
        elif (isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0):
            raise ValueError(
                "latency distribution values must be finite and non-negative")
        else:
            eligible.append(float(value))
    eligible.sort()
    if eligible:
        availability = "available"
        percentiles = _PERCENTILES
        percentile_values = tuple(
            linear_percentile(eligible, percentile)
            for percentile in percentiles
        )
        cdf_values, cumulative_probabilities = _empirical_cdf(eligible)
    else:
        availability = "unavailable_no_eligible_requests"
        percentiles = ()
        percentile_values = ()
        cdf_values = ()
        cumulative_probabilities = ()
    return {
        "availability": availability,
        "eligible_request_count": len(eligible),
        "excluded_request_count": excluded_count,
        "exclusion_reason": exclusion_reason,
        "quantiles": {
            "percentiles": percentiles,
            "values_s": percentile_values,
        },
        "empirical_cdf": {
            "values_s": cdf_values,
            "cumulative_probabilities": cumulative_probabilities,
        },
    }


def _empirical_cdf(
    sorted_values: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    cdf_values = []
    cumulative_probabilities = []
    total = len(sorted_values)
    for index, value in enumerate(sorted_values):
        is_last = index == total - 1
        if is_last or sorted_values[index + 1] != value:
            cdf_values.append(value)
            cumulative_probabilities.append((index + 1) / total)
    return tuple(cdf_values), tuple(cumulative_probabilities)
