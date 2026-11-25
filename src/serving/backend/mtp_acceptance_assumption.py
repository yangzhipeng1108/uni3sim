"""Deterministic maximum-entropy MTP accepted-count assumptions."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
import hashlib
import json
from typing import Iterable, Tuple


_ALGORITHM = "finite_support_maximum_entropy_sha256_inverse_cdf_v1"
_PAIRED_ALGORITHM = "finite_support_maximum_entropy_request_output_prefix_v2"
_SAMPLE_SPACE = 1 << 256


@dataclass(frozen=True)
class MTPAcceptanceAssumption:
    source_id: str
    proposal_depth: int
    mean_acceptance_rate: str
    root_seed: int
    request_stream_sha256: str
    sampling_key: str = "step_stream_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("MTP acceptance source_id must be non-empty")
        if (isinstance(self.proposal_depth, bool)
                or not isinstance(self.proposal_depth, int)):
            raise TypeError("MTP acceptance proposal_depth must be an integer")
        if self.proposal_depth < 1:
            raise ValueError("MTP acceptance proposal_depth must be positive")
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("MTP acceptance root_seed must be an integer")
        if self.root_seed < 0:
            raise ValueError("MTP acceptance root_seed cannot be negative")
        _require_sha256(
            "MTP acceptance request_stream_sha256",
            self.request_stream_sha256,
        )
        normalized_rate = _canonical_decimal(
            self.mean_acceptance_rate, "MTP mean_acceptance_rate")
        rate = Decimal(normalized_rate)
        if rate < 0 or rate > 1:
            raise ValueError("MTP mean_acceptance_rate must be within [0, 1]")
        object.__setattr__(self, "mean_acceptance_rate", normalized_rate)
        if self.sampling_key not in ("step_stream_v1", "request_output_prefix_v1"):
            raise ValueError("unknown MTP acceptance sampling_key")

    @property
    def algorithm(self) -> str:
        return (_PAIRED_ALGORITHM if self.sampling_key == "request_output_prefix_v1"
                else _ALGORITHM)

    @property
    def policy_descriptor_sha256(self) -> str:
        descriptor = {
            "schema_version": 1,
            "source_kind": "explicit_assumption",
            "source_id": self.source_id,
            "algorithm": self.algorithm,
            "proposal_depth": self.proposal_depth,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "root_seed": self.root_seed,
            "request_stream_sha256": self.request_stream_sha256,
        }
        if self.sampling_key == "request_output_prefix_v1":
            descriptor["sampling_key"] = self.sampling_key
        return _canonical_json_sha256(descriptor)

    def probabilities(
        self,
        proposed_tokens: int | None = None,
    ) -> Tuple[Decimal, ...]:
        """Return max-entropy P(K=k) with E[K]=rate*proposed_tokens."""

        if proposed_tokens is None:
            proposed_tokens = self.proposal_depth
        _validate_proposed_tokens(proposed_tokens, self.proposal_depth)
        return _maximum_entropy_probabilities(
            proposed_tokens,
            Decimal(self.mean_acceptance_rate),
        )

    def accepted_count(
        self,
        *,
        step_id: str,
        request_id: str,
        proposed_tokens: int,
        output_tokens_before: int | None = None,
    ) -> int:
        """Select one stable K without pretending it is observed routing.

        The draw is a quantile of a deterministic hash evaluated in
        *ascending cumulative probability* order: threshold[i] is the
        boundary between outcome i and outcome i+1, so a uniform hash
        sample lands in outcome i with probability exactly
        ``probabilities[i]``.  (Selecting on the descending limb — the
        probability of accepting at least k tokens — would double-count
        and push the realized mean to depth+1 minus the target.)
        """

        for name, value in (("step_id", step_id), ("request_id", request_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"MTP acceptance {name} must be non-empty")
        _validate_proposed_tokens(proposed_tokens, self.proposal_depth)
        if self.sampling_key == "request_output_prefix_v1":
            if type(output_tokens_before) is not int or output_tokens_before < 0:
                raise ValueError("paired MTP requires a nonnegative generated output prefix")
            # The request's generated prefix advances only when an output is
            # produced. Padding/no-proposal/recomputation steps do not consume
            # a draw. Arrival, prefix hit, routing and global batching must not
            # change a paired experiment's quantile. Stream/source identities
            # remain bound by the manifest and native admission validation.
            entropy = _canonical_json_sha256({
                "algorithm": self.algorithm,
                "root_seed": self.root_seed,
                "proposal_depth": self.proposal_depth,
                "mean_acceptance_rate": self.mean_acceptance_rate,
                "request_id": request_id,
                "output_tokens_before": output_tokens_before,
            })
        else:
            if output_tokens_before is not None:
                raise ValueError("step-stream MTP does not accept an output-prefix key")
            entropy = f"{self.policy_descriptor_sha256}:{step_id}:{request_id}:{proposed_tokens}"
        probabilities = self.probabilities(proposed_tokens)
        thresholds = []
        cumulative = Decimal(0)
        for probability in probabilities[:-1]:
            cumulative += probability
            thresholds.append(int(cumulative * _SAMPLE_SPACE))
        thresholds.append(_SAMPLE_SPACE)
        digest = hashlib.sha256(
            (
                f"midlayer:{self.algorithm}:{entropy}"
            ).encode("utf-8")
        ).digest()
        sample = int.from_bytes(digest, "big")
        accepted = bisect_right(thresholds, sample)
        if not 0 <= accepted <= proposed_tokens:
            raise RuntimeError("MTP maximum-entropy sampler escaped its support")
        return accepted

    def manifest(self) -> dict[str, object]:
        probabilities = self.probabilities()
        expected = Decimal(self.mean_acceptance_rate) * self.proposal_depth
        manifest = {
            "schema_version": 1,
            "source_kind": "explicit_assumption",
            "source_id": self.source_id,
            "algorithm": self.algorithm,
            "support": {
                "minimum_accepted_tokens": 0,
                "maximum_accepted_tokens": self.proposal_depth,
            },
            "proposal_depth": self.proposal_depth,
            "mean_acceptance_rate": self.mean_acceptance_rate,
            "expected_accepted_tokens": _decimal_text(expected),
            "accepted_count_probabilities": tuple(
                _decimal_text(probability) for probability in probabilities),
            "root_seed": self.root_seed,
            "request_stream_sha256": self.request_stream_sha256,
            "policy_descriptor_sha256": self.policy_descriptor_sha256,
            "duration_observation_source": None,
            "interpretation": "projected_explicit_assumption",
        }
        if self.sampling_key == "request_output_prefix_v1":
            manifest["sampling_key"] = self.sampling_key
        return manifest


def build_mtp_acceptance_rate_sweep(
    *,
    source_id: str,
    proposal_depth: int,
    mean_acceptance_rates: Iterable[str],
    root_seed: int,
    request_stream_sha256: str,
) -> Tuple[MTPAcceptanceAssumption, ...]:
    """Build distinct, hash-bound acceptance-rate sensitivity points."""

    rates = tuple(mean_acceptance_rates)
    if not rates:
        raise ValueError("MTP acceptance-rate sweep requires at least one rate")
    assumptions = tuple(
        MTPAcceptanceAssumption(
            source_id,
            proposal_depth,
            rate,
            root_seed,
            request_stream_sha256,
        )
        for rate in rates
    )
    normalized_rates = tuple(
        assumption.mean_acceptance_rate for assumption in assumptions)
    if len(normalized_rates) != len(set(normalized_rates)):
        raise ValueError("MTP acceptance-rate sweep rates must be unique")
    return assumptions


def normalize_mtp_acceptance_assumption_manifest(
    value: object,
) -> dict[str, object]:
    """Validate and canonicalize an external assumption manifest."""

    if not isinstance(value, Mapping):
        raise TypeError("MTP acceptance assumption must be a mapping")
    raw = dict(value)
    fields = {
        "schema_version",
        "source_kind",
        "source_id",
        "algorithm",
        "support",
        "proposal_depth",
        "mean_acceptance_rate",
        "expected_accepted_tokens",
        "accepted_count_probabilities",
        "root_seed",
        "request_stream_sha256",
        "policy_descriptor_sha256",
        "duration_observation_source",
        "interpretation",
    }
    if "sampling_key" in raw:
        fields.add("sampling_key")
    if set(raw) != fields:
        raise ValueError(
            "MTP acceptance assumption fields must be exactly "
            f"{sorted(fields)}, got {sorted(raw)}")
    assumption = MTPAcceptanceAssumption(
        raw["source_id"],
        raw["proposal_depth"],
        raw["mean_acceptance_rate"],
        raw["root_seed"],
        raw["request_stream_sha256"],
        raw["sampling_key"] if "sampling_key" in raw else "step_stream_v1",
    )
    normalized = assumption.manifest()
    probabilities = raw["accepted_count_probabilities"]
    if not isinstance(probabilities, (list, tuple)):
        raise TypeError(
            "MTP accepted_count_probabilities must be a sequence")
    comparable = dict(raw)
    comparable["accepted_count_probabilities"] = tuple(probabilities)
    if comparable != normalized:
        raise ValueError(
            "MTP acceptance assumption disagrees with its canonical "
            "maximum-entropy contract")
    return normalized


@lru_cache(maxsize=256)
def _maximum_entropy_probabilities(
    proposal_depth: int,
    rate: Decimal,
) -> Tuple[Decimal, ...]:
    if proposal_depth == 0:
        return (Decimal(1),)
    if rate == 0:
        return (Decimal(1),) + (Decimal(0),) * proposal_depth
    if rate == 1:
        return (Decimal(0),) * proposal_depth + (Decimal(1),)
    target = rate * proposal_depth
    with localcontext() as context:
        context.prec = 50
        lower = Decimal(-1)
        upper = Decimal(1)
        while _exponential_family_mean(lower, proposal_depth) > target:
            lower *= 2
        while _exponential_family_mean(upper, proposal_depth) < target:
            upper *= 2
        for _ in range(160):
            midpoint = (lower + upper) / 2
            if _exponential_family_mean(midpoint, proposal_depth) < target:
                lower = midpoint
            else:
                upper = midpoint
        theta = (lower + upper) / 2
        offset = proposal_depth if theta > 0 else 0
        weights = tuple(
            (theta * (accepted - offset)).exp()
            for accepted in range(proposal_depth + 1)
        )
        normalization = sum(weights)
        probabilities = tuple(weight / normalization for weight in weights)
        probability_sum = sum(probabilities)
        if probability_sum != 1:
            probabilities = probabilities[:-1] + (
                probabilities[-1] + (1 - probability_sum),)
        return probabilities


def _exponential_family_mean(theta: Decimal, proposal_depth: int) -> Decimal:
    offset = proposal_depth if theta > 0 else 0
    weights = tuple(
        (theta * (accepted - offset)).exp()
        for accepted in range(proposal_depth + 1)
    )
    normalization = sum(weights)
    return sum(
        accepted * weight for accepted, weight in enumerate(weights)
    ) / normalization


def _validate_proposed_tokens(value: object, proposal_depth: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("MTP proposed_tokens must be an integer")
    if not 0 <= value <= proposal_depth:
        raise ValueError(
            "MTP proposed_tokens must be between zero and proposal_depth")


def _canonical_decimal(value: object, label: str) -> str:
    if isinstance(value, bool) or not isinstance(
            value, (str, int, float, Decimal)):
        raise TypeError(f"{label} must be a decimal string or number")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} must be a finite decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"{label} must be a finite decimal")
    return _decimal_text(parsed)


def _decimal_text(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _require_sha256(label: str, value: object) -> str:
    if (not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
