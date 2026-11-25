"""The MetaX C500 OP Device Event dataset's frozen protocol contract.

The MetaX C500 event modules consume the same MetaX C500 Device Event
evidence.  They must agree on the case families, the selection groups, the
expected canonical counts and digests, and the two Event measurement
protocols, so those facts are declared here once.  A per-module copy could
drift and the modules would then disagree about what "the audited selection"
means.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset.config import file_sha256


FAMILY_COUNTS = {
    "vector": 296, "flash_attn": 294, "gemm": 221, "fused_moe": 176,
}

GROUPS = (
    "vector_warning", "flash_attn", "gemm", "fused_moe_unstable",
    "fused_moe_missing",
)

GROUP_FAMILIES = {
    "vector_warning": "vector", "flash_attn": "flash_attn", "gemm": "gemm",
    "fused_moe_unstable": "fused_moe", "fused_moe_missing": "fused_moe",
}

GROUP_COUNTS = {
    "vector_warning": 149, "flash_attn": 237, "gemm": 13,
    "fused_moe_unstable": 10, "fused_moe_missing": 84,
}

EXPECTED_CANONICAL_SHA256 = {
    "vector": "5ed91693686fa7f24f9c576a8459271e6eb00b1a1c5610b1910ec019d9047d3c",
    "flash_attn": "504499cbd833dfe5288412c23b1517fc7bac2d1fe40d6a07ad919541bc6464ad",
    "gemm": "b1e8fbe39fafc8dfe914f1411caeb0f96b70c40576a82e85f2b2ec1fa6da1ba0",
    "fused_moe": "65490cc7296487d776b5d50ca84b5dc73d9880642385430a2ccfbcdfbc614c58",
}

HISTORICAL_VECTOR_SELECTION_PROTOCOL = {
    "measurement_mode": (
        "paired_differential_events_around_cuda_graph_replay"),
    "warmup_iterations": 5,
    "sample_count_per_repetition": 50,
    "unique_buffers_per_replay": True,
    "buffer_policy": "unique_buffer_per_selected_replay_call",
    "graph_repetitions": (
        "derived_per_case_from_64_MiB_minimum_distinct_input"),
    "hardware_l2_mib": 8,
    "minimum_distinct_input_l2_multiple": 8,
    "minimum_distinct_input_mib": 64,
    "cache_eviction_mib": 256,
}

VECTOR_RECAPTURE_PROTOCOL = {
    "measurement_mode": (
        "paired_differential_events_around_cuda_graph_replay"),
    "warmup_iterations": 5,
    "sample_count_per_repetition": 50,
    "unique_buffers_per_replay": True,
    "buffer_policy": "unique_buffer_per_selected_replay_call",
    "graph_repetitions": (
        "derived_per_case_from_512_MiB_minimum_distinct_input"),
    "hardware_l2_mib": 8,
    "minimum_distinct_input_l2_multiple": 64,
    "minimum_distinct_input_mib": 512,
    "cache_eviction_mib": 256,
    "rationale": (
        "global mechanism-based Event interval extension for short-Graph "
        "lower/high linearity; no per-shape or measured-duration fitting"),
}


def require_positive_integer(value: Any, label: str) -> int:
    """Require a positive integer (bool is not an integer here)."""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    """Load one JSON artifact, requiring an existing file holding an object."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def event_sample_percentile(
    sorted_values: Sequence[float], percentile: float
) -> float:
    """Linear-interpolated percentile over an already-sorted sample."""

    position = percentile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def summarize_event_samples(samples_ns: list[float]) -> Mapping[str, float]:
    """Summarize independent positive device-Event samples."""

    if not isinstance(samples_ns, list) or not samples_ns:
        raise TypeError("samples_ns must be a non-empty list")
    normalized = []
    for index, value in enumerate(samples_ns):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"event sample {index} must be numeric")
        sample = float(value)
        if not math.isfinite(sample) or sample <= 0.0:
            raise ValueError(
                f"event sample {index} must be finite and positive")
        normalized.append(sample)
    ordered = tuple(sorted(normalized))
    median_ns = statistics.median(ordered)
    standard_deviation_ns = statistics.pstdev(ordered)
    return {
        "median_ns": median_ns,
        "p10_ns": event_sample_percentile(ordered, 0.10),
        "p90_ns": event_sample_percentile(ordered, 0.90),
        "standard_deviation_ns": standard_deviation_ns,
        "coefficient_of_variation": standard_deviation_ns / median_ns,
    }
