"""Versioned Interactivity--TGS and arrival-sweep analysis."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence


_CONFIG_KEYS = {
    "schema_version",
    "contract_id",
    "interactivity_metric",
    "interactivity_percentile",
    "interactivity_unit",
    "interactivity_aggregation",
    "interactivity_eligibility",
    "tgs_metric",
    "tgs_unit",
    "tgs_aggregation",
    "peak_selection",
    "saturation_rule",
    "saturation_marginal_gain_ratio_threshold",
    "saturation_minimum_consecutive_intervals",
}

_INTERACTIVITY_ELIGIBILITY = {
    "ttft_s": "successful_requests_with_positive_output_tokens",
    "tpot_s": "successful_requests_with_positive_output_tokens",
    "latency_s": "all_successful_requests",
}

_TGS_UNITS = {
    "output_token_throughput_per_s": "output_tokens_per_second",
    "token_throughput_per_s": "input_plus_output_tokens_per_second",
}


@dataclass(frozen=True)
class InteractivityTGSContract:
    """Frozen reporting choices for one arrival-rate sweep."""

    schema_version: int
    contract_id: str
    interactivity_metric: str
    interactivity_percentile: int
    interactivity_unit: str
    interactivity_aggregation: str
    interactivity_eligibility: str
    tgs_metric: str
    tgs_unit: str
    tgs_aggregation: str
    peak_selection: str
    saturation_rule: str
    saturation_marginal_gain_ratio_threshold: float
    saturation_minimum_consecutive_intervals: int
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (isinstance(self.schema_version, bool)
                or not isinstance(self.schema_version, int)
                or self.schema_version != 1):
            raise ValueError(
                "Interactivity--TGS contract schema_version must be 1")
        if not isinstance(self.contract_id, str) or not self.contract_id:
            raise ValueError(
                "Interactivity--TGS contract_id must be non-empty")
        if self.interactivity_metric not in _INTERACTIVITY_ELIGIBILITY:
            raise ValueError(
                "interactivity_metric must be 'ttft_s', 'tpot_s', or "
                "'latency_s'")
        if (isinstance(self.interactivity_percentile, bool)
                or not isinstance(self.interactivity_percentile, int)
                or not 0 <= self.interactivity_percentile <= 100):
            raise ValueError(
                "interactivity_percentile must be an integer from 0 to 100")
        if self.interactivity_unit != "seconds":
            raise ValueError("interactivity_unit must be 'seconds'")
        if (self.interactivity_aggregation
                != "linear_percentile_over_eligible_requests"):
            raise ValueError(
                "interactivity_aggregation must be "
                "'linear_percentile_over_eligible_requests'")
        expected_eligibility = _INTERACTIVITY_ELIGIBILITY[
            self.interactivity_metric]
        if self.interactivity_eligibility != expected_eligibility:
            raise ValueError(
                "interactivity_eligibility disagrees with "
                f"{self.interactivity_metric!r}: expected "
                f"{expected_eligibility!r}")
        if self.tgs_metric not in _TGS_UNITS:
            raise ValueError(
                "tgs_metric must be 'output_token_throughput_per_s' or "
                "'token_throughput_per_s'")
        expected_tgs_unit = _TGS_UNITS[self.tgs_metric]
        if self.tgs_unit != expected_tgs_unit:
            raise ValueError(
                "tgs_unit disagrees with tgs_metric: expected "
                f"{expected_tgs_unit!r}")
        if (self.tgs_aggregation
                != "completed_tokens_over_simulation_origin_to_last_completion"):
            raise ValueError(
                "tgs_aggregation must be "
                "'completed_tokens_over_simulation_origin_to_last_completion'")
        if (self.peak_selection
                != "maximum_tgs_then_lowest_arrival_rate_scale"):
            raise ValueError(
                "peak_selection must be "
                "'maximum_tgs_then_lowest_arrival_rate_scale'")
        if (self.saturation_rule
                != "first_consecutive_marginal_tgs_gain_ratio_at_or_below_threshold"):
            raise ValueError(
                "saturation_rule must be "
                "'first_consecutive_marginal_tgs_gain_ratio_at_or_below_threshold'")
        threshold = self.saturation_marginal_gain_ratio_threshold
        if (isinstance(threshold, bool)
                or not isinstance(threshold, Real)
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1):
            raise ValueError(
                "saturation_marginal_gain_ratio_threshold must be finite "
                "and between 0 and 1")
        intervals = self.saturation_minimum_consecutive_intervals
        if (isinstance(intervals, bool)
                or not isinstance(intervals, int)
                or intervals < 1):
            raise ValueError(
                "saturation_minimum_consecutive_intervals must be positive")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("contract source_path must be non-empty")
        if (not isinstance(self.source_sha256, str)
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.source_sha256)):
            raise ValueError(
                "contract source_sha256 must be lowercase SHA256 hex")


@dataclass(frozen=True)
class SweepMetricPoint:
    """Metrics needed for one point of the versioned sweep analysis."""

    arrival_rate_scale: float
    request_throughput_per_s: float
    output_token_throughput_per_s: float
    token_throughput_per_s: float
    interactivity_value_s: float | None
    interactivity_eligible_request_count: int
    interactivity_excluded_request_count: int
    interactivity_percentile_method: str

    def __post_init__(self) -> None:
        _finite_positive("arrival_rate_scale", self.arrival_rate_scale)
        _finite_non_negative(
            "request_throughput_per_s", self.request_throughput_per_s)
        _finite_non_negative(
            "output_token_throughput_per_s",
            self.output_token_throughput_per_s,
        )
        _finite_non_negative(
            "token_throughput_per_s", self.token_throughput_per_s)
        if self.interactivity_value_s is not None:
            _finite_non_negative(
                "interactivity_value_s", self.interactivity_value_s)
        count = self.interactivity_eligible_request_count
        if (isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0):
            raise ValueError(
                "interactivity_eligible_request_count must be non-negative")
        if (self.interactivity_value_s is None) != (count == 0):
            raise ValueError(
                "interactivity value availability disagrees with eligible "
                "request count")
        excluded = self.interactivity_excluded_request_count
        if (isinstance(excluded, bool)
                or not isinstance(excluded, int)
                or excluded < 0):
            raise ValueError(
                "interactivity_excluded_request_count must be non-negative")
        if (not isinstance(self.interactivity_percentile_method, str)
                or not self.interactivity_percentile_method):
            raise ValueError(
                "interactivity_percentile_method must be non-empty")


def load_interactivity_tgs_contract(
    path: str | Path,
) -> InteractivityTGSContract:
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid Interactivity--TGS contract JSON: {source_path}") \
            from error
    if not isinstance(payload, dict):
        raise ValueError("Interactivity--TGS contract must be a JSON object")
    supplied = set(payload)
    if supplied != _CONFIG_KEYS:
        raise ValueError(
            "Interactivity--TGS contract fields disagree with schema: "
            f"missing={sorted(_CONFIG_KEYS - supplied)}, "
            f"unexpected={sorted(supplied - _CONFIG_KEYS)}")
    return InteractivityTGSContract(
        schema_version=payload["schema_version"],
        contract_id=payload["contract_id"],
        interactivity_metric=payload["interactivity_metric"],
        interactivity_percentile=payload["interactivity_percentile"],
        interactivity_unit=payload["interactivity_unit"],
        interactivity_aggregation=payload["interactivity_aggregation"],
        interactivity_eligibility=payload["interactivity_eligibility"],
        tgs_metric=payload["tgs_metric"],
        tgs_unit=payload["tgs_unit"],
        tgs_aggregation=payload["tgs_aggregation"],
        peak_selection=payload["peak_selection"],
        saturation_rule=payload["saturation_rule"],
        saturation_marginal_gain_ratio_threshold=payload[
            "saturation_marginal_gain_ratio_threshold"],
        saturation_minimum_consecutive_intervals=payload[
            "saturation_minimum_consecutive_intervals"],
        source_path=str(source_path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def sweep_metric_point_from_manifest(
    arrival_rate_scale: float,
    metrics: Mapping[str, object],
    contract: InteractivityTGSContract,
) -> SweepMetricPoint:
    """Select the contract's interactivity quantile from one run manifest."""

    if not isinstance(metrics, Mapping):
        raise TypeError("sweep point metrics must be a mapping")
    if not isinstance(contract, InteractivityTGSContract):
        raise TypeError(
            "sweep point contract must be InteractivityTGSContract")
    required = {
        "request_throughput_per_s",
        "output_token_throughput_per_s",
        "token_throughput_per_s",
        "latency_distributions",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(
            f"sweep point metrics are missing fields: {sorted(missing)}")
    distributions = metrics["latency_distributions"]
    if not isinstance(distributions, Mapping):
        raise TypeError("latency_distributions must be a mapping")
    if "percentile_method" not in distributions:
        raise ValueError(
            "latency_distributions is missing percentile_method")
    if contract.interactivity_metric not in distributions:
        raise ValueError(
            "latency_distributions does not contain contract metric "
            f"{contract.interactivity_metric!r}")
    distribution = distributions[contract.interactivity_metric]
    if not isinstance(distribution, Mapping):
        raise TypeError("selected latency distribution must be a mapping")
    required_distribution = {
        "availability",
        "eligible_request_count",
        "excluded_request_count",
        "quantiles",
    }
    missing_distribution = required_distribution - set(distribution)
    if missing_distribution:
        raise ValueError(
            "selected latency distribution is missing fields: "
            f"{sorted(missing_distribution)}")
    availability = distribution["availability"]
    eligible_count = distribution["eligible_request_count"]
    excluded_count = distribution["excluded_request_count"]
    percentile_method = distributions["percentile_method"]
    if percentile_method != "linear_interpolation_rank_n_minus_1_v1":
        raise ValueError(
            "latency distribution percentile_method disagrees with the "
            "contract aggregation")
    quantiles = distribution["quantiles"]
    if not isinstance(quantiles, Mapping):
        raise TypeError("selected latency quantiles must be a mapping")
    required_quantiles = {"percentiles", "values_s"}
    missing_quantiles = required_quantiles - set(quantiles)
    if missing_quantiles:
        raise ValueError(
            "selected latency quantiles are missing fields: "
            f"{sorted(missing_quantiles)}")
    percentiles = tuple(quantiles["percentiles"])
    values_s = tuple(quantiles["values_s"])
    if availability == "available":
        if len(percentiles) != len(values_s):
            raise ValueError(
                "selected latency percentiles and values have different "
                "lengths")
        if len(percentiles) != len(set(percentiles)):
            raise ValueError("selected latency percentiles are not unique")
        if contract.interactivity_percentile not in percentiles:
            raise ValueError(
                "selected latency distribution does not contain contract "
                f"percentile P{contract.interactivity_percentile}")
        index = percentiles.index(contract.interactivity_percentile)
        interactivity_value_s = values_s[index]
    elif availability == "unavailable_no_eligible_requests":
        if percentiles or values_s or eligible_count != 0:
            raise ValueError(
                "unavailable latency distribution contains quantile data")
        interactivity_value_s = None
    else:
        raise ValueError(
            f"unknown latency distribution availability: {availability!r}")
    return SweepMetricPoint(
        arrival_rate_scale=arrival_rate_scale,
        request_throughput_per_s=metrics["request_throughput_per_s"],
        output_token_throughput_per_s=metrics[
            "output_token_throughput_per_s"],
        token_throughput_per_s=metrics["token_throughput_per_s"],
        interactivity_value_s=interactivity_value_s,
        interactivity_eligible_request_count=eligible_count,
        interactivity_excluded_request_count=excluded_count,
        interactivity_percentile_method=percentile_method,
    )


def arrival_rate_sweep_analysis_manifest(
    points: Sequence[SweepMetricPoint] | None,
    contract: InteractivityTGSContract | None,
) -> dict[str, object]:
    """Build the trade-off, observed peak, and explicit saturation result."""

    if contract is None:
        if points is not None:
            raise ValueError(
                "sweep metric points require an Interactivity--TGS contract")
        unavailable = {
            "availability": "unavailable_no_metric_contract",
            "contract": None,
        }
        return {
            "interactivity_tgs": {
                **unavailable,
                "points": (),
                "pareto_frontier": (),
            },
            "peak_selection": unavailable,
            "saturation_analysis": {
                **unavailable,
                "intervals": (),
            },
        }
    if not isinstance(contract, InteractivityTGSContract):
        raise TypeError(
            "sweep analysis contract must be InteractivityTGSContract or None")
    if points is None or not isinstance(points, Sequence) or not points:
        raise ValueError(
            "versioned sweep analysis requires at least one metric point")
    if any(not isinstance(point, SweepMetricPoint) for point in points):
        raise TypeError("sweep analysis points must be SweepMetricPoint")
    ordered = tuple(sorted(points, key=lambda point: point.arrival_rate_scale))
    scales = tuple(point.arrival_rate_scale for point in ordered)
    if len(scales) != len(set(scales)):
        raise ValueError("sweep analysis arrival-rate scales must be unique")
    contract_manifest = _contract_manifest(contract)
    point_manifests = tuple(
        _point_manifest(point, contract) for point in ordered)
    available_points = tuple(
        point for point in ordered
        if point.interactivity_value_s is not None)
    if not available_points:
        interactivity_availability = "unavailable_no_eligible_requests"
    elif len(available_points) == len(ordered):
        interactivity_availability = "available"
    else:
        interactivity_availability = "partially_available"
    pareto = tuple(
        _point_manifest(point, contract)
        for point in _pareto_frontier(available_points, contract)
    )
    peak = min(
        ordered,
        key=lambda point: (
            -_tgs_value(point, contract),
            point.arrival_rate_scale,
        ),
    )
    peak_tgs = _tgs_value(peak, contract)
    saturation = _saturation_manifest(ordered, contract, contract_manifest)
    return {
        "interactivity_tgs": {
            "availability": interactivity_availability,
            "contract": contract_manifest,
            "point_order": "arrival_rate_scale_ascending",
            "dominance": "lower_interactivity_and_higher_tgs",
            "points": point_manifests,
            "pareto_frontier": pareto,
        },
        "peak_selection": (
            {
                "availability": "unavailable_no_positive_tgs",
                "contract": contract_manifest,
                "selection": contract.peak_selection,
                "selection_scope": "sampled_arrival_rate_scales_only",
                "point": None,
            }
            if peak_tgs == 0 else
            {
                "availability": "available_observed_sampled_range",
                "contract": contract_manifest,
                "selection": contract.peak_selection,
                "selection_scope": "sampled_arrival_rate_scales_only",
                "point": _point_manifest(peak, contract),
            }
        ),
        "saturation_analysis": saturation,
    }


def _contract_manifest(
    contract: InteractivityTGSContract,
) -> dict[str, object]:
    return {
        "schema_version": contract.schema_version,
        "contract_id": contract.contract_id,
        "interactivity_metric": contract.interactivity_metric,
        "interactivity_percentile": contract.interactivity_percentile,
        "interactivity_unit": contract.interactivity_unit,
        "interactivity_aggregation": contract.interactivity_aggregation,
        "interactivity_eligibility": contract.interactivity_eligibility,
        "tgs_metric": contract.tgs_metric,
        "tgs_unit": contract.tgs_unit,
        "tgs_aggregation": contract.tgs_aggregation,
        "peak_selection": contract.peak_selection,
        "saturation_rule": contract.saturation_rule,
        "saturation_marginal_gain_ratio_threshold": (
            contract.saturation_marginal_gain_ratio_threshold),
        "saturation_minimum_consecutive_intervals": (
            contract.saturation_minimum_consecutive_intervals),
        "source_path": contract.source_path,
        "source_sha256": contract.source_sha256,
    }


def _point_manifest(
    point: SweepMetricPoint,
    contract: InteractivityTGSContract,
) -> dict[str, object]:
    return {
        "arrival_rate_scale": point.arrival_rate_scale,
        "interactivity_value_s": point.interactivity_value_s,
        "interactivity_eligible_request_count": (
            point.interactivity_eligible_request_count),
        "interactivity_excluded_request_count": (
            point.interactivity_excluded_request_count),
        "interactivity_distribution_availability": (
            "available" if point.interactivity_value_s is not None
            else "unavailable_no_eligible_requests"),
        "interactivity_percentile_method": (
            point.interactivity_percentile_method),
        "tgs_value": _tgs_value(point, contract),
        "request_throughput_per_s": point.request_throughput_per_s,
    }


def _tgs_value(
    point: SweepMetricPoint,
    contract: InteractivityTGSContract,
) -> float:
    if contract.tgs_metric == "output_token_throughput_per_s":
        return point.output_token_throughput_per_s
    elif contract.tgs_metric == "token_throughput_per_s":
        return point.token_throughput_per_s
    else:
        raise AssertionError(
            f"validated unknown TGS metric: {contract.tgs_metric!r}")


def _pareto_frontier(
    points: Sequence[SweepMetricPoint],
    contract: InteractivityTGSContract,
) -> tuple[SweepMetricPoint, ...]:
    frontier = []
    for candidate in points:
        candidate_interactivity = candidate.interactivity_value_s
        if candidate_interactivity is None:
            raise AssertionError("unavailable point entered Pareto analysis")
        candidate_tgs = _tgs_value(candidate, contract)
        dominated = False
        for other in points:
            if other is candidate:
                continue
            other_interactivity = other.interactivity_value_s
            if other_interactivity is None:
                raise AssertionError(
                    "unavailable point entered Pareto analysis")
            other_tgs = _tgs_value(other, contract)
            weakly_better = (
                other_interactivity <= candidate_interactivity
                and other_tgs >= candidate_tgs)
            strictly_better = (
                other_interactivity < candidate_interactivity
                or other_tgs > candidate_tgs)
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(
        frontier,
        key=lambda point: (
            point.interactivity_value_s,
            -_tgs_value(point, contract),
            point.arrival_rate_scale,
        ),
    ))


def _saturation_manifest(
    points: Sequence[SweepMetricPoint],
    contract: InteractivityTGSContract,
    contract_manifest: Mapping[str, object],
) -> dict[str, object]:
    if len(points) < 2:
        return {
            "availability": "unavailable_fewer_than_two_points",
            "observation_scope": "sampled_arrival_rate_scales_only",
            "observation_rule": "sampled_elasticity_threshold",
            "contains_throughput_regression": False,
            "contract": contract_manifest,
            "intervals": (),
            "saturation_onset_point": None,
            "saturation_confirmed_at_point": None,
        }
    intervals = []
    consecutive = 0
    current_run_onset = None
    selected_onset = None
    selected_confirmation = None
    comparable_interval_count = 0
    contains_throughput_regression = False
    threshold = contract.saturation_marginal_gain_ratio_threshold
    required = contract.saturation_minimum_consecutive_intervals
    for previous, current in zip(points, points[1:]):
        offered_gain = (
            (current.arrival_rate_scale - previous.arrival_rate_scale)
            / previous.arrival_rate_scale)
        if offered_gain <= 0:
            raise ValueError(
                "saturation analysis requires strictly increasing scales")
        previous_tgs = _tgs_value(previous, contract)
        current_tgs = _tgs_value(current, contract)
        if previous_tgs == 0:
            tgs_relative_gain = None
            marginal_gain_ratio = None
            at_or_below_threshold = False
            interval_classification = "incomparable_zero_tgs_baseline"
            consecutive = 0
            current_run_onset = None
        else:
            comparable_interval_count += 1
            tgs_relative_gain = (current_tgs - previous_tgs) / previous_tgs
            marginal_gain_ratio = tgs_relative_gain / offered_gain
            at_or_below_threshold = marginal_gain_ratio <= threshold
            if tgs_relative_gain < 0:
                interval_classification = "throughput_regression"
                contains_throughput_regression = True
            elif at_or_below_threshold:
                interval_classification = "plateau_candidate"
            else:
                interval_classification = "growth"
            if at_or_below_threshold:
                if consecutive == 0:
                    current_run_onset = current
                consecutive += 1
                if consecutive >= required and selected_onset is None:
                    selected_onset = current_run_onset
                    selected_confirmation = current
            else:
                consecutive = 0
                current_run_onset = None
        intervals.append({
            "from_arrival_rate_scale": previous.arrival_rate_scale,
            "to_arrival_rate_scale": current.arrival_rate_scale,
            "offered_load_relative_gain": offered_gain,
            "tgs_relative_gain": tgs_relative_gain,
            "marginal_tgs_gain_ratio": marginal_gain_ratio,
            "at_or_below_threshold": at_or_below_threshold,
            "classification": interval_classification,
        })
    if comparable_interval_count == 0:
        availability = "unavailable_no_positive_tgs_baseline"
    elif selected_onset is None:
        availability = "not_observed_in_sampled_range"
    else:
        availability = "observed_sampled_elasticity_threshold"
    return {
        "availability": availability,
        "observation_scope": "sampled_arrival_rate_scales_only",
        "observation_rule": "sampled_elasticity_threshold",
        "contains_throughput_regression": contains_throughput_regression,
        "contract": contract_manifest,
        "intervals": tuple(intervals),
        "saturation_onset_point": (
            None if selected_onset is None
            else _point_manifest(selected_onset, contract)),
        "saturation_confirmed_at_point": (
            None if selected_confirmation is None
            else _point_manifest(selected_confirmation, contract)),
    }


def _unique_json_object(pairs) -> dict[str, object]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(
                f"duplicate Interactivity--TGS contract field: {key!r}")
        payload[key] = value
    return payload


def _finite_positive(name: str, value: object) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0):
        raise ValueError(f"{name} must be finite and positive")


def _finite_non_negative(name: str, value: object) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0):
        raise ValueError(f"{name} must be finite and non-negative")
