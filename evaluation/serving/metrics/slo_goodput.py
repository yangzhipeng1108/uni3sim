"""Versioned SLO contracts and request/token goodput reporting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from src.serving.metrics.serving_metrics import RequestMetrics, ServingMetrics
from src.serving.request_gen.request_stream import RequestStream


_CONFIG_KEYS = {
    "schema_version",
    "contract_id",
    "ttft_threshold_s",
    "tpot_threshold_s",
    "comparison",
    "threshold_composition",
    "request_eligibility",
    "single_output_tpot",
    "token_basis",
    "duration_basis",
}


@dataclass(frozen=True)
class SLOContract:
    schema_version: int
    contract_id: str
    ttft_threshold_s: float
    tpot_threshold_s: float
    comparison: str
    threshold_composition: str
    request_eligibility: str
    single_output_tpot: str
    token_basis: str
    duration_basis: str
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        if (isinstance(self.schema_version, bool)
                or not isinstance(self.schema_version, int)
                or self.schema_version != 1):
            raise ValueError("SLO contract schema_version must be 1")
        if not isinstance(self.contract_id, str) or not self.contract_id:
            raise ValueError("SLO contract_id must be a non-empty string")
        for name, value in (
                ("ttft_threshold_s", self.ttft_threshold_s),
                ("tpot_threshold_s", self.tpot_threshold_s)):
            if (isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(value)
                    or value < 0):
                raise ValueError(
                    f"SLO {name} must be finite and non-negative")
        if self.comparison != "less_than_or_equal":
            raise ValueError(
                "SLO comparison must be 'less_than_or_equal'")
        if self.threshold_composition != "ttft_and_tpot":
            raise ValueError(
                "SLO threshold_composition must be 'ttft_and_tpot'")
        if self.request_eligibility != "positive_output_tokens":
            raise ValueError(
                "SLO request_eligibility must be "
                "'positive_output_tokens'")
        if self.single_output_tpot != "zero_seconds":
            raise ValueError(
                "SLO single_output_tpot must be 'zero_seconds'")
        if self.token_basis not in (
                "output_tokens", "input_plus_output_tokens"):
            raise ValueError(
                "SLO token_basis must be 'output_tokens' or "
                "'input_plus_output_tokens'")
        if self.duration_basis != "simulation_origin_to_last_completion":
            raise ValueError(
                "SLO duration_basis must be "
                "'simulation_origin_to_last_completion'")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("SLO source_path must be a non-empty string")
        if (not isinstance(self.source_sha256, str)
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.source_sha256)):
            raise ValueError("SLO source_sha256 must be lowercase SHA256 hex")


def load_slo_contract(path: str | Path) -> SLOContract:
    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid SLO contract JSON: {source_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("SLO contract must be a JSON object")
    supplied = set(payload)
    if supplied != _CONFIG_KEYS:
        raise ValueError(
            "SLO contract fields disagree with schema: "
            f"missing={sorted(_CONFIG_KEYS - supplied)}, "
            f"unexpected={sorted(supplied - _CONFIG_KEYS)}")
    return SLOContract(
        schema_version=payload["schema_version"],
        contract_id=payload["contract_id"],
        ttft_threshold_s=payload["ttft_threshold_s"],
        tpot_threshold_s=payload["tpot_threshold_s"],
        comparison=payload["comparison"],
        threshold_composition=payload["threshold_composition"],
        request_eligibility=payload["request_eligibility"],
        single_output_tpot=payload["single_output_tpot"],
        token_basis=payload["token_basis"],
        duration_basis=payload["duration_basis"],
        source_path=str(source_path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _unique_json_object(pairs) -> dict[str, object]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate SLO contract field: {key!r}")
        payload[key] = value
    return payload


def slo_goodput_manifest(
    requests: RequestStream,
    metrics: ServingMetrics,
    duration_s: float,
    contract: SLOContract | None,
) -> dict[str, object]:
    if not isinstance(requests, RequestStream):
        raise TypeError("SLO goodput requests must be a RequestStream")
    if not isinstance(metrics, ServingMetrics):
        raise TypeError("SLO goodput metrics must be ServingMetrics")
    if (isinstance(duration_s, bool)
            or not isinstance(duration_s, Real)
            or not math.isfinite(duration_s)
            or duration_s <= 0):
        raise ValueError("SLO goodput duration must be finite and positive")
    request_by_id = {
        request.request_id: request for request in requests
    }
    if set(metrics.requests) != set(request_by_id):
        raise ValueError(
            "SLO goodput request IDs disagree between requests and metrics")
    eligible = []
    excluded_count = 0
    for request_id in sorted(request_by_id):
        request = request_by_id[request_id]
        metric = metrics.requests[request_id]
        _validate_request_metric(request_id, request.output_tokens, metric)
        if request.output_tokens == 0:
            excluded_count += 1
        elif request.output_tokens > 0:
            eligible.append((request, metric))
        else:
            raise AssertionError("RequestSpec validates output token counts")
    population = {
        "total_request_count": len(request_by_id),
        "eligible_request_count": len(eligible),
        "excluded_request_count": excluded_count,
        "slo_met_request_count": None,
        "violated_request_count": None,
        "eligible_token_count": None,
        "slo_met_token_count": None,
    }
    if contract is None:
        return {
            "schema_version": 1,
            "availability": "unavailable_no_slo_contract",
            "contract": None,
            "population": population,
            "violation_counts": None,
            "request_goodput_per_s": None,
            "token_goodput_per_s": None,
        }
    if not isinstance(contract, SLOContract):
        raise TypeError("SLO goodput contract must be SLOContract or None")
    met_request_count = 0
    met_token_count = 0
    eligible_token_count = 0
    ttft_only = 0
    tpot_only = 0
    both = 0
    for request, metric in eligible:
        token_count = _request_token_count(request, contract.token_basis)
        eligible_token_count += token_count
        ttft_violated = metric.ttft_s > contract.ttft_threshold_s
        tpot_violated = metric.tpot_s > contract.tpot_threshold_s
        if not ttft_violated and not tpot_violated:
            met_request_count += 1
            met_token_count += token_count
        elif ttft_violated and not tpot_violated:
            ttft_only += 1
        elif not ttft_violated and tpot_violated:
            tpot_only += 1
        elif ttft_violated and tpot_violated:
            both += 1
        else:
            raise AssertionError("SLO threshold comparisons are exhaustive")
    violated_count = ttft_only + tpot_only + both
    population["slo_met_request_count"] = met_request_count
    population["violated_request_count"] = violated_count
    population["eligible_token_count"] = eligible_token_count
    population["slo_met_token_count"] = met_token_count
    return {
        "schema_version": 1,
        "availability": "available",
        "contract": _contract_manifest(contract),
        "population": population,
        "violation_counts": {
            "ttft_only": ttft_only,
            "tpot_only": tpot_only,
            "ttft_and_tpot": both,
        },
        "request_goodput_per_s": met_request_count / float(duration_s),
        "token_goodput_per_s": met_token_count / float(duration_s),
    }


def _validate_request_metric(
    request_id: str,
    output_tokens: int,
    metric: RequestMetrics,
) -> None:
    if not isinstance(metric, RequestMetrics):
        raise TypeError("SLO goodput request metrics must be RequestMetrics")
    if metric.request_id != request_id:
        raise ValueError(
            "SLO goodput request key and RequestMetrics ID disagree")
    if output_tokens == 0:
        if metric.ttft_s is not None or metric.tpot_s is not None:
            raise ValueError(
                "zero-output SLO request must have unavailable TTFT/TPOT")
        return
    for name, value in (("TTFT", metric.ttft_s), ("TPOT", metric.tpot_s)):
        if (isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0):
            raise ValueError(
                f"positive-output SLO request {name} must be finite and "
                "non-negative")
    if output_tokens == 1 and metric.tpot_s != 0:
        raise ValueError(
            "single-output SLO request TPOT must be zero seconds")


def _request_token_count(request, token_basis: str) -> int:
    if token_basis == "output_tokens":
        return request.output_tokens
    elif token_basis == "input_plus_output_tokens":
        return request.input_tokens + request.output_tokens
    else:
        raise ValueError(f"unknown SLO token basis: {token_basis!r}")


def _contract_manifest(contract: SLOContract) -> dict[str, object]:
    return {
        "schema_version": contract.schema_version,
        "contract_id": contract.contract_id,
        "ttft_threshold_s": float(contract.ttft_threshold_s),
        "tpot_threshold_s": float(contract.tpot_threshold_s),
        "comparison": contract.comparison,
        "threshold_composition": contract.threshold_composition,
        "request_eligibility": contract.request_eligibility,
        "single_output_tpot": contract.single_output_tpot,
        "token_basis": contract.token_basis,
        "duration_basis": contract.duration_basis,
        "source_path": contract.source_path,
        "source_sha256": contract.source_sha256,
    }
