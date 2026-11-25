"""Deterministic sampling and OP-level Event comparison for kernel datasets."""

from __future__ import annotations

from evaluation.json_shape import file_sha256

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from evaluation.kernel.dataset import (
    get_kernel_dataset_path,
    parse_kernel_evaluation_timing,
)
from evaluation.kernel.device_event import (
    BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
    CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
    DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE,
    DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE,
    GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION,
    LEGACY_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
    LEGACY_GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION,
    SYNCHRONIZED_SWEEP_GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION,
)
from src.kernel.dataset_storage import (
    kernel_dataset_content_sha256,
    load_kernel_dataset_document,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


SAMPLE_MANIFEST_SCHEMA_VERSION = 1
SAMPLE_BUCKET_SCHEMA = "kernel_golden_primary_v1"
HARDWARE_IDS = EVALUATED_HARDWARE_IDS
DATASET_FAMILIES = ("gemm", "flash_attn", "fused_moe", "vector")
SAMPLE_POSITIONS = ("minimum", "median", "maximum")
EVENT_CASE_REQUIRED_KEYS = frozenset({
    "schema_version", "status", "unsupported_reason", "hardware_id",
    "family", "case_id", "baseline_commit", "source_snapshot_sha256",
    "runner_path", "runner_sha256", "timer_sanity_sha256",
    "canonical_attributes_sha256", "timer", "stream_contract",
    "allocation_contract", "workload", "environment", "measurement",
})


def _range_label(value: int, ranges: tuple[tuple[str, int, int | None], ...],
                 context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    matches = [
        label for label, lower, upper in ranges
        if lower <= value and (upper is None or value <= upper)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{context} must match exactly one primary bucket, got {matches}")
    return matches[0]


def primary_bucket(family: str, entry: Mapping[str, Any]) -> str:
    """Return the non-overlapping primary bucket used for first-pass sampling.

    Labels and boundaries are the existing single-kernel evaluation report's
    primary workload-size/operator groups.  Its orthogonal diagnostic views
    (N/K, split count, KV length/layout, and launch waves) intentionally remain
    report dimensions rather than additional sampling partitions.
    """

    op_info = entry["op_info"]
    attributes = op_info["Attr"]
    if family == "gemm":
        if op_info["Type"] != "GEMM":
            raise ValueError("gemm dataset entry must have Type='GEMM'")
        size = attributes["B"] * attributes["M"]
        return _range_label(size, (
            ("tiny (B×M≤16)", 1, 16),
            ("small (17≤B×M≤64)", 17, 64),
            ("med (65≤B×M≤2047)", 65, 2047),
            ("prefill (B×M≥2048)", 2048, None),
        ), "GEMM B×M")
    elif family == "fused_moe":
        if op_info["Type"] != "FUSEDMOE":
            raise ValueError("fused_moe dataset entry must have Type='FUSEDMOE'")
        return _range_label(attributes["TOK"], (
            ("tiny (TOK≤16)", 1, 16),
            ("small (17≤TOK≤64)", 17, 64),
            ("med (65≤TOK≤2047)", 65, 2047),
            ("prefill (TOK≥2048)", 2048, None),
        ), "FusedMoE TOK")
    elif family == "flash_attn":
        if op_info["Type"] != "FLASHATTN":
            raise ValueError(
                "flash_attn dataset entry must have Type='FLASHATTN'")
        return _range_label(attributes["Q_LEN"], (
            ("Q=1", 1, 1),
            ("short Q (2–512)", 2, 512),
            ("medium Q (513–2048)", 513, 2048),
            ("long Q (≥2049)", 2049, None),
        ), "FlashAttention Q_LEN")
    elif family == "vector":
        op_type = op_info["Type"]
        labels = {
            "NORM": "NORM",
            "ROTARY": "ROTARY",
            "ACTIVATION": "ACTIVATION",
            "ELEMENTWISE": "ELEMENTWISE",
            "KV_CACHE": "KV_CACHE",
        }
        if op_type not in labels:
            raise ValueError(f"unsupported vector op type: {op_type!r}")
        return labels[op_type]
    else:
        raise ValueError(f"unsupported kernel dataset family: {family!r}")


def workload_units(family: str, entry: Mapping[str, Any]) -> int:
    """Return a duration-independent integer used only to rank samples."""

    attributes = entry["op_info"]["Attr"]
    if family == "gemm":
        value = (
            attributes["B"] * attributes["M"]
            * attributes["K"] * attributes["N"]
        )
    elif family == "fused_moe":
        value = (
            attributes["TOK"] * attributes["TOPK"]
            * attributes["H"] * attributes["INT"]
        )
    elif family == "flash_attn":
        query_lengths = attributes["Q_LENS"]
        key_value_lengths = attributes["KV_LENS"]
        if (not isinstance(query_lengths, list)
                or not isinstance(key_value_lengths, list)
                or len(query_lengths) != len(key_value_lengths)
                or not query_lengths):
            raise ValueError(
                "FlashAttention Q_LENS/KV_LENS must be equal non-empty lists")
        causal = attributes["causal"]
        if not isinstance(causal, bool):
            raise TypeError("FlashAttention causal must be boolean")
        if any(query > key_value for query, key_value in zip(
                query_lengths, key_value_lengths, strict=True)):
            raise ValueError(
                "FlashAttention query length cannot exceed KV length")
        if causal:
            attended_pairs = sum(
                query * (key_value - query) + query * (query + 1) // 2
                for query, key_value in zip(
                    query_lengths, key_value_lengths, strict=True)
            )
        else:
            attended_pairs = sum(
                query * key_value for query, key_value in zip(
                    query_lengths, key_value_lengths, strict=True)
            )
        value = 4 * attributes["HQ"] * attributes["D"] * attended_pairs
    elif family == "vector":
        value = attributes["element_count"]
    else:
        raise ValueError(f"unsupported kernel dataset family: {family!r}")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{family} workload size must be a positive integer")
    return value


def select_representative_cases(
        family: str, dataset: Mapping[str, Any], samples_per_bucket: int = 3
        ) -> Mapping[str, list[Mapping[str, Any]]]:
    """Select deterministic minimum/middle/maximum workloads per bucket."""

    if (isinstance(samples_per_bucket, bool)
            or not isinstance(samples_per_bucket, int)):
        raise TypeError("samples_per_bucket must be an integer")
    if samples_per_bucket != 3:
        raise ValueError(
            "kernel-golden v1 fixes samples_per_bucket at 3")
    if not isinstance(dataset, Mapping) or not dataset:
        raise TypeError("dataset must be a non-empty mapping")

    buckets: dict[str, list[tuple[int, str]]] = {}
    for case_id, entry in dataset.items():
        if not isinstance(case_id, str) or not case_id:
            raise TypeError("dataset case IDs must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise TypeError(f"dataset entry {case_id!r} must be a mapping")
        bucket = primary_bucket(family, entry)
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append((workload_units(family, entry), case_id))

    result = {}
    for bucket, candidates in sorted(buckets.items()):
        ordered = sorted(candidates)
        indexes = sorted({0, (len(ordered) - 1) // 2, len(ordered) - 1})
        selected = []
        for output_index, candidate_index in enumerate(indexes):
            if len(indexes) == 1:
                position = "minimum"
            elif len(indexes) == 2:
                position = ("minimum", "maximum")[output_index]
            elif len(indexes) == 3:
                position = SAMPLE_POSITIONS[output_index]
            else:
                raise AssertionError("three-point selection produced >3 indexes")
            units, case_id = ordered[candidate_index]
            selected.append({
                "position": position,
                "case_id": case_id,
                "workload_units": units,
            })
        result[bucket] = selected
    return result


def _load_family_dataset(hardware_id: str, family: str) -> Mapping[str, Any]:
    path = Path(get_kernel_dataset_path(hardware_id, family))
    payload = load_kernel_dataset_document(path)
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"kernel dataset must be a non-empty mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    return file_sha256(path)


def build_sample_manifest(baseline_commit: str) -> Mapping[str, Any]:
    """Build the reproducible first-pass sample manifest."""

    if (not isinstance(baseline_commit, str)
            or len(baseline_commit) != 40
            or any(character not in "0123456789abcdef"
                   for character in baseline_commit)):
        raise ValueError("baseline_commit must be a lowercase 40-digit SHA-1")
    hardware = {}
    for hardware_id in HARDWARE_IDS:
        families = {}
        for family in DATASET_FAMILIES:
            path = Path(get_kernel_dataset_path(hardware_id, family))
            dataset = _load_family_dataset(hardware_id, family)
            families[family] = {
                "dataset_sha256": kernel_dataset_content_sha256(path),
                "dataset_case_count": len(dataset),
                "buckets": select_representative_cases(family, dataset),
            }
        hardware[hardware_id] = {"families": families}
    return {
        "schema_version": SAMPLE_MANIFEST_SCHEMA_VERSION,
        "baseline_commit": baseline_commit,
        "bucket_schema": SAMPLE_BUCKET_SCHEMA,
        "samples_per_bucket": 3,
        "selection_positions": list(SAMPLE_POSITIONS),
        "selection_order": (
            "duration_independent_logical_work_then_canonical_case_id"),
        "scope": {
            "primary_partition": {
                "gemm": "B_times_M_range",
                "flash_attn": "executed_Q_LEN_range",
                "fused_moe": "TOK_range",
                "vector": "op_type",
            },
            "orthogonal_report_dimensions_not_sampling_partitions": [
                "gemm_split_and_N_and_K",
                "flash_attn_source_and_phase_and_split_and_KV_and_layout",
                "fused_moe_experts_and_max_tokens_per_expert_and_top_k",
                "vector_token_range",
                "maximum_launch_waves",
            ],
        },
        "hardware": hardware,
    }


def load_sample_manifest(path: Path) -> Mapping[str, Any]:
    """Load the committed manifest and prove it matches current datasets."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("kernel-golden sample manifest must be a mapping")
    expected = build_sample_manifest(payload["baseline_commit"])
    if payload != expected:
        raise ValueError(
            "kernel-golden sample manifest does not match current canonical "
            "datasets and deterministic selection contract")
    return payload


def compare_case_measurement(
        *, hardware_id: str, family: str, case_id: str,
        event_samples_ns: list[float]) -> Mapping[str, Any]:
    """Compare a complete-OP Event sample distribution with canonical truth."""

    from evaluation.kernel.device_event import (
        summarize_event_samples,
    )

    dataset = _load_family_dataset(hardware_id, family)
    if case_id not in dataset:
        raise KeyError(
            f"case {case_id!r} is absent from {hardware_id}/{family}")
    entry = dataset[case_id]
    kernels = entry["kernels"]
    if isinstance(kernels, dict):
        kernel_records = list(kernels.values())
    elif isinstance(kernels, list):
        kernel_records = kernels
    else:
        raise TypeError(f"dataset entry {case_id!r} kernels must be list/mapping")
    target = parse_kernel_evaluation_timing(entry, kernel_records, hardware_id)
    statistics = summarize_event_samples(event_samples_ns)
    event_median_ns = statistics["median_ns"]
    signed_error_ns = target.total_duration_ns - event_median_ns
    absolute_error_ns = abs(signed_error_ns)
    return {
        "hardware_id": hardware_id,
        "family": family,
        "case_id": case_id,
        "bucket": primary_bucket(family, entry),
        "dataset_timing_source": target.source,
        "dataset_total_duration_ns": target.total_duration_ns,
        "event_statistics": statistics,
        "absolute_error_ns": absolute_error_ns,
        "relative_error": absolute_error_ns / event_median_ns,
        "signed_relative_error": signed_error_ns / event_median_ns,
    }


def _canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selected_case_index(
        manifest: Mapping[str, Any], hardware_id: str
        ) -> Mapping[str, str]:
    families = manifest["hardware"][hardware_id]["families"]
    result = {}
    for family in DATASET_FAMILIES:
        buckets = families[family]["buckets"]
        for rows in buckets.values():
            for row in rows:
                case_id = row["case_id"]
                if case_id in result:
                    raise ValueError(
                        f"sample manifest repeats case {case_id!r} for "
                        f"{hardware_id}")
                result[case_id] = family
    return result


def _load_json_mapping(path: Path, context: str) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON mapping: {path}")
    return payload


def _validate_timer_sanity(
        *, path: Path, hardware_id: str, baseline_commit: str
        ) -> tuple[Mapping[str, Any], str]:
    payload = _load_json_mapping(path, "timer sanity artifact")
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported timer sanity schema: {path}")
    if payload["artifact_kind"] != "device_event_timer_sanity":
        raise ValueError(f"invalid timer sanity artifact kind: {path}")
    if payload["hardware_id"] != hardware_id:
        raise ValueError(
            f"timer sanity hardware mismatch: expected={hardware_id}, "
            f"actual={payload['hardware_id']}")
    if payload["baseline_commit"] != baseline_commit:
        raise ValueError(f"timer sanity baseline mismatch: {path}")
    if payload["profiler_started"] is not False:
        raise ValueError(f"timer sanity must prove profiler_started=false: {path}")
    checks = payload["checks"]
    if not isinstance(checks, dict) or not checks:
        raise TypeError(f"timer sanity checks must be a non-empty mapping: {path}")
    failed_checks = [name for name, passed in checks.items() if passed is not True]
    if failed_checks:
        raise ValueError(
            f"timer sanity has failed checks {failed_checks}: {path}")
    return payload, _sha256(path)


def _validate_npu_rotating_buffer_contract(
        contract: Any, case_id: str) -> None:
    expected_keys = {
        "graph_backend", "hardware_l2_bytes", "cache_eviction_bytes",
        "cache_eviction_l2_multiple",
        "cache_eviction_l2_multiple_rounded_2dp",
        "lower_operation_count", "higher_operation_count",
        "operation_count_delta", "preallocated_operation_count",
        "preallocated_distinct_bytes", "graph_output_distinct_bytes",
        "shared_nonoverlapping_output_bytes", "rotation_total_distinct_bytes",
        "addresses_unique_within_each_graph",
        "rotating_set_may_be_smaller_than_l2", "selection_rule",
        "cache_policy",
    }
    if not isinstance(contract, dict) or set(contract) != expected_keys:
        raise ValueError(f"invalid NPUGraph rotating-buffer contract for {case_id}")
    if contract["graph_backend"] != "torch.npu.NPUGraph":
        raise ValueError(f"invalid NPUGraph backend for {case_id}")
    if contract["hardware_l2_bytes"] != 192 * 1024 * 1024:
        raise ValueError(f"invalid Ascend 910B L2 size for {case_id}")
    if contract["cache_eviction_bytes"] != 2 * 1024 * 1024 * 1024:
        raise ValueError(f"NPUGraph sweep is not 2 GiB for {case_id}")
    expected_multiple = (
        contract["cache_eviction_bytes"] / contract["hardware_l2_bytes"])
    if (not math.isclose(
            contract["cache_eviction_l2_multiple"], expected_multiple,
            rel_tol=1e-12, abs_tol=1e-12)
            or contract["cache_eviction_l2_multiple_rounded_2dp"] != 10.67):
        raise ValueError(f"invalid NPUGraph sweep/L2 multiple for {case_id}")
    lower_count = contract["lower_operation_count"]
    higher_count = contract["higher_operation_count"]
    operation_delta = contract["operation_count_delta"]
    preallocated_count = contract["preallocated_operation_count"]
    if (any(isinstance(value, bool) or not isinstance(value, int)
            for value in (
                lower_count, higher_count, operation_delta,
                preallocated_count))
            or lower_count <= 0 or lower_count >= higher_count
            or operation_delta != higher_count - lower_count
            or preallocated_count != higher_count):
        raise ValueError(f"invalid NPUGraph operation counts for {case_id}")
    for name in (
            "preallocated_distinct_bytes", "rotation_total_distinct_bytes"):
        value = contract[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"invalid NPUGraph {name} for {case_id}")
    for name in (
            "graph_output_distinct_bytes",
            "shared_nonoverlapping_output_bytes"):
        value = contract[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid NPUGraph {name} for {case_id}")
    if contract["rotation_total_distinct_bytes"] != (
            contract["preallocated_distinct_bytes"]
            + contract["graph_output_distinct_bytes"]
            + contract["shared_nonoverlapping_output_bytes"]):
        raise ValueError(f"invalid NPUGraph total distinct bytes for {case_id}")
    if (contract["addresses_unique_within_each_graph"] is not True
            or contract["rotating_set_may_be_smaller_than_l2"] is not True):
        raise ValueError(f"invalid NPUGraph address contract for {case_id}")
    if contract["selection_rule"] not in {
            "npu_graph_if_static_256_call_rotation_estimate_lte_l2",
            "fixed_16_256_unique_preallocated_calls",
            "memory_bounded_unique_preallocated_calls_v1"}:
        raise ValueError(f"invalid NPUGraph selection rule for {case_id}")
    if contract["cache_policy"] != (
            "same_stream_2_gib_bf16_sweep_before_every_graph_replay"):
        raise ValueError(f"invalid NPUGraph cache policy for {case_id}")


def _validate_event_measurement(
        measurement: Mapping[str, Any], case_id: str
        ) -> list[float]:
    if not isinstance(measurement, dict):
        raise TypeError(f"measured case {case_id} measurement must be a mapping")
    schema_version = measurement["schema_version"]
    if (isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version not in {1, 2, 3,
                                      BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION}):
        raise ValueError(f"unsupported Event sample schema for {case_id}")
    is_graph_measurement = "mode" in measurement
    if is_graph_measurement:
        graph_mode = measurement["mode"]
        if graph_mode not in {
                "external_events_around_cuda_graph_replay",
                DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE,
                DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE}:
            raise ValueError(f"unsupported graph Event mode for {case_id}")
        if (graph_mode in {
                DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE,
                DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE}
                and schema_version != GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION):
            raise ValueError(
                "differential graph Event mode requires queued schema 3 for "
                f"{case_id}")
        if "event_prelude_iterations" in measurement:
            raise ValueError(
                "graph Event samples must not claim an Event prelude for "
                f"{case_id}")
        if schema_version == LEGACY_GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION:
            if ("cache_eviction_bytes_before_replay" in measurement
                    or "cache_eviction_requested_bytes" in measurement):
                raise ValueError(
                    "graph Event schema 1 must not claim cache eviction for "
                    f"{case_id}")
            if "event_prelude_iterations_per_repetition" in measurement:
                raise ValueError(
                    "graph Event schema 1 must not claim a replay prelude for "
                    f"{case_id}")
        elif schema_version == (
                SYNCHRONIZED_SWEEP_GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION):
            if "cache_eviction_bytes_before_replay" not in measurement:
                raise ValueError(
                    "graph Event schema 2 requires "
                    f"cache_eviction_bytes_before_replay for {case_id}")
            cache_eviction_bytes = measurement[
                "cache_eviction_bytes_before_replay"]
            if (isinstance(cache_eviction_bytes, bool)
                    or not isinstance(cache_eviction_bytes, int)
                    or cache_eviction_bytes < 0):
                raise ValueError(
                    "graph Event schema 2 requires a nonnegative integer "
                    f"cache_eviction_bytes_before_replay for {case_id}")
            if ("cache_eviction_requested_bytes" in measurement
                    or "event_prelude_iterations_per_repetition" in measurement):
                raise ValueError(
                    "graph Event schema 2 must not claim queued-sweep fields "
                    f"for {case_id}")
        elif schema_version == GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION:
            if "cache_eviction_bytes_before_replay" not in measurement:
                raise ValueError(
                    "graph Event schema 3 requires "
                    f"cache_eviction_bytes_before_replay for {case_id}")
            cache_eviction_bytes = measurement[
                "cache_eviction_bytes_before_replay"]
            if (isinstance(cache_eviction_bytes, bool)
                    or not isinstance(cache_eviction_bytes, int)
                    or cache_eviction_bytes < 0):
                raise ValueError(
                    "graph Event schema 3 requires a nonnegative integer "
                    f"cache_eviction_bytes_before_replay for {case_id}")
            if "cache_eviction_requested_bytes" not in measurement:
                raise ValueError(
                    "graph Event schema 3 requires "
                    f"cache_eviction_requested_bytes for {case_id}")
            requested_bytes = measurement["cache_eviction_requested_bytes"]
            if (isinstance(requested_bytes, bool)
                    or not isinstance(requested_bytes, int)
                    or requested_bytes != cache_eviction_bytes):
                raise ValueError(
                    f"graph Event cache request/actual mismatch for {case_id}")
            if "event_prelude_iterations_per_repetition" not in measurement:
                raise ValueError(
                    "graph Event schema 3 requires "
                    "event_prelude_iterations_per_repetition for "
                    f"{case_id}")
            event_prelude_iterations = measurement[
                "event_prelude_iterations_per_repetition"]
            if (isinstance(event_prelude_iterations, bool)
                    or not isinstance(event_prelude_iterations, int)
                    or event_prelude_iterations != 1):
                raise ValueError(
                    "graph Event schema 3 requires "
                    "event_prelude_iterations_per_repetition=1 for "
                    f"{case_id}")
        else:
            raise ValueError(f"unsupported graph Event schema for {case_id}")
    elif schema_version in {
            LEGACY_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
            CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
            BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
            }:
        if "event_prelude_iterations" not in measurement:
            raise ValueError(
                f"direct Event schema {schema_version} requires "
                "event_prelude_iterations "
                f"for {case_id}")
        event_prelude_iterations = measurement["event_prelude_iterations"]
        if (isinstance(event_prelude_iterations, bool)
                or not isinstance(event_prelude_iterations, int)
                or event_prelude_iterations != 1):
            raise ValueError(
                f"direct Event schema {schema_version} requires "
                f"event_prelude_iterations=1 for {case_id}")
        if "cache_eviction_bytes_before_sample" not in measurement:
            raise ValueError(
                f"direct Event schema {schema_version} requires "
                f"cache_eviction_bytes_before_sample for {case_id}")
        cache_eviction_bytes = measurement[
            "cache_eviction_bytes_before_sample"]
        if (isinstance(cache_eviction_bytes, bool)
                or not isinstance(cache_eviction_bytes, int)
                or cache_eviction_bytes < 0):
            raise ValueError(
                f"direct Event schema {schema_version} requires a "
                "nonnegative integer "
                f"cache_eviction_bytes_before_sample for {case_id}")
        if schema_version == LEGACY_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION:
            if "cache_eviction_requested_bytes" in measurement:
                raise ValueError(
                    "direct Event schema 2 must not claim queued-sweep fields "
                    f"for {case_id}")
        elif schema_version == CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION:
            if "cache_eviction_requested_bytes" not in measurement:
                raise ValueError(
                    "direct Event schema 3 requires "
                    f"cache_eviction_requested_bytes for {case_id}")
            requested_bytes = measurement["cache_eviction_requested_bytes"]
            if (isinstance(requested_bytes, bool)
                    or not isinstance(requested_bytes, int)
                    or requested_bytes != cache_eviction_bytes):
                raise ValueError(
                    f"direct Event cache request/actual mismatch for {case_id}")
            if "operation_repetitions_per_sample" in measurement:
                raise ValueError(
                    "direct Event schema 3 must not claim batched operations "
                    f"for {case_id}")
        elif schema_version == BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION:
            if "cache_eviction_requested_bytes" not in measurement:
                raise ValueError(
                    "batched direct Event schema 4 requires "
                    f"cache_eviction_requested_bytes for {case_id}")
            requested_bytes = measurement["cache_eviction_requested_bytes"]
            if (isinstance(requested_bytes, bool)
                    or not isinstance(requested_bytes, int)
                    or requested_bytes != cache_eviction_bytes):
                raise ValueError(
                    f"batched direct Event cache request/actual mismatch for {case_id}")
            repetitions = measurement["operation_repetitions_per_sample"]
            if (isinstance(repetitions, bool)
                    or not isinstance(repetitions, int)
                    or repetitions <= 1):
                raise ValueError(
                    "batched direct Event operation repetitions must exceed one "
                    f"for {case_id}")
        else:
            raise AssertionError("validated direct schema was not dispatched")
    elif schema_version == 1:
        if ("event_prelude_iterations" in measurement
                or "event_prelude_iterations_per_repetition" in measurement):
            raise ValueError(
                "Event sample schema 1 must not claim an Event prelude for "
                f"{case_id}")
        if ("cache_eviction_bytes_before_sample" in measurement
                or "cache_eviction_requested_bytes" in measurement):
            raise ValueError(
                f"direct Event schema 1 must not claim cache eviction for {case_id}")
    else:
        raise AssertionError("validated Event sample schema was not dispatched")
    sample_count = measurement["sample_count"]
    warmup_iterations = measurement["warmup_iterations"]
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count <= 0):
        raise ValueError(f"invalid sample_count for {case_id}")
    if (isinstance(warmup_iterations, bool)
            or not isinstance(warmup_iterations, int)
            or warmup_iterations <= 0):
        raise ValueError(f"invalid warmup_iterations for {case_id}")
    if is_graph_measurement:
        repetition_counts = measurement["repetition_counts"]
        if (not isinstance(repetition_counts, list)
                or len(repetition_counts) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in repetition_counts)
                or repetition_counts[0] >= repetition_counts[1]):
            raise ValueError(f"invalid graph repetition_counts for {case_id}")
        operation_cycle_length = measurement["operation_cycle_length"]
        if (isinstance(operation_cycle_length, bool)
                or not isinstance(operation_cycle_length, int)
                or operation_cycle_length <= 0):
            raise ValueError(
                f"invalid graph operation_cycle_length for {case_id}")
        repetition_runs = measurement["repetition_runs"]
        if not isinstance(repetition_runs, list) or len(repetition_runs) != 2:
            raise ValueError(f"invalid graph repetition_runs for {case_id}")
        run_durations = []
        for run_index, (run, repetition_count) in enumerate(zip(
                repetition_runs, repetition_counts, strict=True)):
            if not isinstance(run, dict):
                raise TypeError(
                    f"graph run {run_index} for {case_id} must be a mapping")
            if run["repetition_count"] != repetition_count:
                raise ValueError(
                    f"graph run repetition mismatch for {case_id}")
            if run["sample_count"] != sample_count:
                raise ValueError(f"graph run sample count mismatch for {case_id}")
            durations = _validate_event_samples(
                run["samples"], sample_count, case_id,
                f"graph run {run_index}")
            _validate_event_statistics(
                durations, run["statistics"], case_id,
                f"graph run {run_index}")
            for sample in run["samples"]:
                if sample["repetition_count"] != repetition_count:
                    raise ValueError(
                        f"graph sample repetition mismatch for {case_id}")
                replay_duration = sample["replay_duration_ns"]
                if (isinstance(replay_duration, bool)
                        or not isinstance(replay_duration, (int, float))
                        or not math.isfinite(replay_duration)
                        or replay_duration <= 0.0
                        or not math.isclose(
                            float(sample["event_duration_ns"]),
                            float(replay_duration) / repetition_count,
                            rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError(
                        f"invalid graph replay duration for {case_id}")
            run_durations.append(durations)
        if graph_mode == "external_events_around_cuda_graph_replay":
            if measurement["samples"] != repetition_runs[1]["samples"]:
                raise ValueError(
                    "graph selected samples are not the higher run for "
                    f"{case_id}")
            if measurement["statistics"] != repetition_runs[1]["statistics"]:
                raise ValueError(
                    "graph selected statistics are not the higher run for "
                    f"{case_id}")
            lower_median_ns = repetition_runs[0]["statistics"]["median_ns"]
            higher_median_ns = repetition_runs[1]["statistics"]["median_ns"]
            disagreement_ns = abs(higher_median_ns - lower_median_ns)
            tolerance_ns = max(
                500.0, 0.05 * max(lower_median_ns, higher_median_ns))
            expected_convergence = {
                "status": (
                    "converged"
                    if disagreement_ns <= tolerance_ns else "nonconverged"),
                "absolute_difference_ns": disagreement_ns,
                "tolerance_ns": tolerance_ns,
                "relative_difference": (
                    disagreement_ns / max(
                        lower_median_ns, higher_median_ns)),
            }
            if measurement["convergence"] != expected_convergence:
                raise ValueError(f"graph convergence mismatch for {case_id}")
            return run_durations[1]

        if "convergence" in measurement:
            raise ValueError(
                f"differential graph samples must not claim convergence: "
                f"{case_id}")
        if measurement["estimator"] != "(T_high-T_low)/(N_high-N_low)":
            raise ValueError(
                f"invalid differential graph estimator for {case_id}")
        if measurement["sample_pairing"] != (
                "alternating_lower_then_higher_and_higher_then_lower"):
            raise ValueError(
                f"invalid differential graph pairing for {case_id}")
        prelude_durations = measurement[
            "event_prelude_replay_durations_ns"]
        if (not isinstance(prelude_durations, list)
                or len(prelude_durations) != 2
                or any(isinstance(value, bool)
                       or not isinstance(value, (int, float))
                       or not math.isfinite(value) or value <= 0.0
                       for value in prelude_durations)):
            raise ValueError(
                f"invalid differential graph preludes for {case_id}")
        if graph_mode == DIFFERENTIAL_NPU_GRAPH_REPLAY_EVENT_MODE:
            repetition_delta = repetition_counts[1] - repetition_counts[0]
            if (operation_cycle_length != repetition_counts[1]
                    or measurement["cache_eviction_requested_bytes"]
                    != 2 * 1024 * 1024 * 1024
                    or measurement["cache_eviction_bytes_before_replay"]
                    != 2 * 1024 * 1024 * 1024):
                raise ValueError(
                    f"invalid fixed NPUGraph measurement contract for {case_id}")
            _validate_npu_rotating_buffer_contract(
                measurement["rotating_buffer_contract"], case_id)
            rotating_contract = measurement["rotating_buffer_contract"]
            if (rotating_contract["lower_operation_count"]
                    != repetition_counts[0]
                    or rotating_contract["higher_operation_count"]
                    != repetition_counts[1]
                    or rotating_contract["operation_count_delta"]
                    != repetition_delta
                    or rotating_contract["preallocated_operation_count"]
                    != operation_cycle_length):
                raise ValueError(
                    f"NPUGraph measurement/rotation counts disagree for "
                    f"{case_id}")
            differential_durations = _validate_event_samples(
                measurement["samples"], sample_count, case_id,
                "differential NPUGraph")
            _validate_event_statistics(
                differential_durations, measurement["statistics"], case_id,
                "differential NPUGraph")
            for sample_index, sample in enumerate(measurement["samples"]):
                expected_order = (
                    "lower_then_higher" if sample_index % 2 == 0
                    else "higher_then_lower")
                lower_sample = repetition_runs[0]["samples"][sample_index]
                higher_sample = repetition_runs[1]["samples"][sample_index]
                lower_replay_ns = float(lower_sample["replay_duration_ns"])
                higher_replay_ns = float(higher_sample["replay_duration_ns"])
                expected_positions = (
                    (0, 1) if sample_index % 2 == 0 else (1, 0))
                if (sample["replay_order"] != expected_order
                        or sample["repetition_delta"] != repetition_delta
                        or lower_sample["replay_order_position"]
                        != expected_positions[0]
                        or higher_sample["replay_order_position"]
                        != expected_positions[1]
                        or not math.isclose(
                            float(sample["lower_replay_duration_ns"]),
                            lower_replay_ns, rel_tol=1e-12, abs_tol=1e-12)
                        or not math.isclose(
                            float(sample["higher_replay_duration_ns"]),
                            higher_replay_ns, rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError(
                        f"differential NPUGraph raw replay mismatch for {case_id}")
                expected_duration_ns = (
                    (higher_replay_ns - lower_replay_ns) / repetition_delta)
                if (expected_duration_ns <= 0.0 or not math.isclose(
                        differential_durations[sample_index],
                        expected_duration_ns, rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError(
                        f"differential NPUGraph sample mismatch for {case_id}")
            return differential_durations
        distinct_input = measurement["distinct_input_contract"]
        expected_distinct_keys = {
            "hardware_l2_bytes", "required_minimum_l2_multiple",
            "configured_minimum_l2_multiple",
            "configured_minimum_bytes", "per_call_bytes",
            "lower_replay_bytes", "higher_replay_bytes",
            "delta_replay_bytes", "repetition_selection_rule",
        }
        if (not isinstance(distinct_input, dict)
                or set(distinct_input) != expected_distinct_keys):
            raise ValueError(
                f"invalid differential distinct-input contract for {case_id}")
        for name in expected_distinct_keys - {"repetition_selection_rule"}:
            value = distinct_input[name]
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ValueError(
                    f"invalid differential distinct-input {name} for "
                    f"{case_id}")
        if distinct_input["required_minimum_l2_multiple"] != 8:
            raise ValueError(
                f"differential input threshold is not 8x L2 for {case_id}")
        configured_multiple = distinct_input[
            "configured_minimum_l2_multiple"]
        if (configured_multiple < 8
                or distinct_input["configured_minimum_bytes"]
                != configured_multiple * distinct_input["hardware_l2_bytes"]):
            raise ValueError(
                f"differential configured L2 multiple mismatch for {case_id}")
        if distinct_input["configured_minimum_bytes"] < (
                8 * distinct_input["hardware_l2_bytes"]):
            raise ValueError(
                f"differential input threshold is below 8x L2 for {case_id}")
        expected_rule = (
            "lower=ceil(configured_minimum_bytes/per_call_bytes);"
            "higher=2*lower")
        if distinct_input["repetition_selection_rule"] != expected_rule:
            raise ValueError(
                f"invalid differential repetition rule for {case_id}")
        lower_count, higher_count = repetition_counts
        per_call_bytes = distinct_input["per_call_bytes"]
        expected_lower_count = math.ceil(
            distinct_input["configured_minimum_bytes"] / per_call_bytes)
        if (lower_count != expected_lower_count
                or higher_count != 2 * lower_count
                or operation_cycle_length != higher_count):
            raise ValueError(
                f"differential graph repetition selection mismatch for "
                f"{case_id}")
        expected_input_bytes = {
            "lower_replay_bytes": lower_count * per_call_bytes,
            "higher_replay_bytes": higher_count * per_call_bytes,
            "delta_replay_bytes": (
                higher_count - lower_count) * per_call_bytes,
        }
        for name, expected in expected_input_bytes.items():
            if distinct_input[name] != expected:
                raise ValueError(
                    f"differential graph {name} mismatch for {case_id}")
        if (distinct_input["lower_replay_bytes"]
                < distinct_input["configured_minimum_bytes"]
                or distinct_input["delta_replay_bytes"]
                < distinct_input["configured_minimum_bytes"]):
            raise ValueError(
                f"differential graph input threshold was not met for {case_id}")

        differential_durations = _validate_event_samples(
            measurement["samples"], sample_count, case_id,
            "differential graph")
        _validate_event_statistics(
            differential_durations, measurement["statistics"], case_id,
            "differential graph")
        repetition_delta = higher_count - lower_count
        for sample_index, sample in enumerate(measurement["samples"]):
            expected_order = (
                "lower_then_higher" if sample_index % 2 == 0
                else "higher_then_lower")
            if (sample["replay_order"] != expected_order
                    or sample["repetition_delta"] != repetition_delta):
                raise ValueError(
                    f"differential graph pair ordering mismatch for {case_id}")
            lower_sample = repetition_runs[0]["samples"][sample_index]
            higher_sample = repetition_runs[1]["samples"][sample_index]
            lower_replay_ns = float(lower_sample["replay_duration_ns"])
            higher_replay_ns = float(higher_sample["replay_duration_ns"])
            expected_positions = (
                (0, 1) if sample_index % 2 == 0 else (1, 0))
            if (lower_sample["replay_order_position"] != expected_positions[0]
                    or higher_sample["replay_order_position"]
                    != expected_positions[1]
                    or not math.isclose(
                        float(sample["lower_replay_duration_ns"]),
                        lower_replay_ns, rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(
                        float(sample["higher_replay_duration_ns"]),
                        higher_replay_ns, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(
                    f"differential graph raw replay mismatch for {case_id}")
            expected_duration_ns = (
                (higher_replay_ns - lower_replay_ns) / repetition_delta)
            if (expected_duration_ns <= 0.0 or not math.isclose(
                    differential_durations[sample_index],
                    expected_duration_ns, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(
                    f"differential graph sample mismatch for {case_id}")
        return differential_durations
    samples = measurement["samples"]
    durations = _validate_event_samples(
        samples, sample_count, case_id, "direct Event")
    _validate_event_statistics(
        durations, measurement["statistics"], case_id, "direct Event")
    if schema_version == BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION:
        repetitions = measurement["operation_repetitions_per_sample"]
        for index, sample in enumerate(samples):
            sample_repetitions = sample["operation_repetition_count"]
            aggregate_duration = sample["aggregate_event_duration_ns"]
            if sample_repetitions != repetitions:
                raise ValueError(
                    "batched direct Event repetition count mismatch for "
                    f"{case_id} sample {index}")
            if (isinstance(aggregate_duration, bool)
                    or not isinstance(aggregate_duration, (int, float))
                    or not math.isfinite(aggregate_duration)
                    or aggregate_duration <= 0.0
                    or not math.isclose(
                        durations[index],
                        float(aggregate_duration) / repetitions,
                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(
                    "batched direct Event aggregate normalization mismatch for "
                    f"{case_id} sample {index}")
    return durations


def _validate_event_samples(
        samples: Any, sample_count: int, case_id: str,
        context: str) -> list[float]:
    if not isinstance(samples, list) or len(samples) != sample_count:
        raise ValueError(f"{context} sample count mismatch for {case_id}")
    durations = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(
                f"{context} sample {index} for {case_id} must be a mapping")
        if sample["sample_index"] != index:
            raise ValueError(
                f"{context} sample indexes are not contiguous for {case_id}")
        duration = sample["event_duration_ns"]
        if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0.0):
            raise ValueError(
                f"invalid {context} duration for {case_id} sample {index}")
        durations.append(float(duration))
    return durations


def _validate_event_statistics(
        durations: list[float], recorded_statistics: Any,
        case_id: str, context: str) -> None:
    from evaluation.kernel.device_event import summarize_event_samples
    expected_statistics = summarize_event_samples(durations)
    if (not isinstance(recorded_statistics, dict)
            or set(recorded_statistics) != set(expected_statistics)):
        raise ValueError(
            f"{context} statistics disagree with raw samples: {case_id}")
    for name, expected in expected_statistics.items():
        recorded = recorded_statistics[name]
        if (isinstance(recorded, bool)
                or not isinstance(recorded, (int, float))
                or not math.isfinite(recorded)
                or not math.isclose(
                    float(recorded), expected, rel_tol=1e-12,
                    abs_tol=1e-12)):
            raise ValueError(
                f"{context} statistic {name} disagrees with raw samples: "
                f"{case_id}")


def _validate_ascend_vector_input_ring(
        *, record: Mapping[str, Any], case_id: str,
        canonical_attributes: Mapping[str, Any]) -> None:
    """Validate the no-reuse dense-input condition for Ascend Vector Event data."""

    workload = record["workload"]
    input_ring = workload["input_ring"]
    common_keys = {
        "mode", "requested_bytes", "allocated_bytes", "input_bytes_per_call",
        "slot_count", "slot_stride", "selection", "storage_offset",
        "initialization", "pre_target_sweep",
    }
    if not isinstance(input_ring, dict) or "mode" not in input_ring:
        raise ValueError(f"invalid Ascend Vector input-ring contract for {case_id}")
    input_ring_mode = input_ring["mode"]
    if input_ring_mode == "rotating_dense_input_backing_ring":
        expected_keys = common_keys
        expected_selection = (
            "coprime_stride_permutation_of_disjoint_contiguous_backing_slots")
        expected_storage_offset = "slot_index_times_input_element_count"
        expected_initialization = "torch.empty_then_normal_before_timing"
    elif input_ring_mode == "rotating_paired_dense_input_backing_ring":
        expected_keys = common_keys | {
            "input_bytes_per_operand", "dynamic_input_operands",
            "static_input_operands",
        }
        expected_selection = (
            "coprime_stride_permutation_of_disjoint_contiguous_paired_backing_slots")
        expected_storage_offset = "slot_index_times_paired_input_element_count"
        expected_initialization = "torch.empty_then_normal_before_timing"
    elif input_ring_mode == "rotating_asymmetric_paired_dense_input_backing_ring":
        expected_keys = common_keys | {
            "input_bytes_per_operand", "dynamic_input_operands",
            "static_input_operands",
        }
        expected_selection = (
            "coprime_stride_permutation_of_disjoint_asymmetric_dense_input_slots")
        expected_storage_offset = (
            "slot_index_times_per_operand_element_count_in_separate_backings")
        expected_initialization = "torch.empty_then_normal_before_timing"
    elif input_ring_mode == "rotating_kvcache_input_and_destination_ring":
        expected_keys = common_keys | {
            "input_bytes_per_operand", "dynamic_input_operands",
            "static_input_operands",
        }
        expected_selection = (
            "coprime_stride_permutation_of_disjoint_kv_inputs_and_cache_destinations")
        expected_storage_offset = (
            "slot_index_times_per_operand_element_count_in_separate_backings")
        expected_initialization = "random_inputs_zeroed_cache_before_timing"
    else:
        raise ValueError(f"invalid Ascend Vector input-ring mode for {case_id}")
    if set(input_ring) != expected_keys:
        raise ValueError(f"invalid Ascend Vector input-ring contract for {case_id}")
    if input_ring["selection"] != expected_selection:
        raise ValueError(f"invalid Ascend Vector input-ring selection for {case_id}")
    if input_ring["storage_offset"] != expected_storage_offset:
        raise ValueError(f"invalid Ascend Vector input-ring storage offset for {case_id}")
    if input_ring["initialization"] != expected_initialization:
        raise ValueError(
            f"invalid Ascend Vector input-ring initialization for {case_id}")
    if input_ring["pre_target_sweep"] != "none":
        raise ValueError(f"Ascend Vector input ring mixes a sweep: {case_id}")
    if workload["event_measurement_mode"] != "direct":
        raise ValueError(f"Ascend Vector input ring is not directly timed: {case_id}")
    measurement = record["measurement"]
    if measurement["schema_version"] == (
            BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION):
        if input_ring_mode == "rotating_paired_dense_input_backing_ring":
            expected_input_selection = (
                "advance_coprime_paired_dense_input_ring_once_per_public_call")
        elif input_ring_mode == "rotating_dense_input_backing_ring":
            expected_input_selection = (
                "advance_coprime_dense_input_ring_once_per_public_call")
        elif input_ring_mode == (
                "rotating_asymmetric_paired_dense_input_backing_ring"):
            expected_input_selection = (
                "advance_coprime_asymmetric_paired_dense_input_ring_once_per_public_call")
        elif input_ring_mode == "rotating_kvcache_input_and_destination_ring":
            expected_input_selection = (
                "advance_coprime_kvcache_input_and_destination_ring_once_per_public_call")
        else:
            raise AssertionError(
                f"validated input-ring mode was not dispatched: {case_id}")
        repetitions = measurement["operation_repetitions_per_sample"]
        if repetitions not in {32, 512, 2048, 8192}:
            raise ValueError(
                f"invalid Ascend Vector Event batch repetitions for {case_id}")
        if input_ring["slot_count"] < repetitions:
            raise ValueError(
                "Ascend Vector input ring reuses a slot within one Event "
                f"interval: {case_id}")
        expected_batch = {
            "operation_repetitions_per_sample": repetitions,
            "input_selection": expected_input_selection,
            "duration_reporting": (
                "aggregate_current_stream_event_duration_divided_by_public_call_count"),
        }
        if workload["event_operation_batch"] != expected_batch:
            raise ValueError(
                f"invalid Ascend Vector Event batching contract for {case_id}")
        if record["timer"]["mode"] != (
                "fixed_count_distinct_public_ops_per_event_interval_reported_per_call"):
            raise ValueError(
                f"invalid Ascend Vector batched timer mode for {case_id}")
        persistent_batch = workload["persistent_batch"]
        expected_batch_keys = {
            "mode", "manifest_sha256", "case_ordinal", "case_count"}
        if (not isinstance(persistent_batch, dict)
                or set(persistent_batch) != expected_batch_keys
                or persistent_batch["mode"]
                != "one_process_serial_vector_event_capture"):
            raise ValueError(
                f"invalid Ascend Vector persistent batch contract for {case_id}")
        manifest_sha256 = persistent_batch["manifest_sha256"]
        if (not isinstance(manifest_sha256, str)
                or len(manifest_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in manifest_sha256)):
            raise ValueError(
                f"invalid Ascend Vector persistent manifest digest for {case_id}")
        case_ordinal = persistent_batch["case_ordinal"]
        case_count = persistent_batch["case_count"]
        if (isinstance(case_ordinal, bool) or isinstance(case_count, bool)
                or not isinstance(case_ordinal, int)
                or not isinstance(case_count, int)
                or case_count <= 0 or not 0 <= case_ordinal < case_count):
            raise ValueError(
                f"invalid Ascend Vector persistent batch ordinal for {case_id}")
    elif measurement["schema_version"] == CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION:
        if "event_operation_batch" in workload:
            raise ValueError(
                f"unbatched Ascend Vector Event claims batching for {case_id}")
        if record["timer"]["mode"] != "one_exact_public_op_per_event_interval":
            raise ValueError(
                f"invalid Ascend Vector direct timer mode for {case_id}")
    else:
        raise ValueError(
            f"unsupported Ascend Vector input-ring Event schema for {case_id}")
    if "power_warmup" in workload:
        power_warmup = workload["power_warmup"]
        legacy_power_warmup = {
            "kind": "premeasurement_bf16_gemm_device_power_warmup",
            "matrix_dimension": 4096,
            "gemm_iterations": 1000,
            "synchronized_before_event_samples": True,
            "included_in_event_interval": False,
        }
        maintained_power_warmup = {
            "kind": "bf16_gemm_power_warmup_and_per_event_maintenance",
            "matrix_dimension": 4096,
            "initial_gemm_iterations": 1000,
            "per_event_gemm_iterations": 1,
            "per_event_placement": (
                "same_current_stream_immediately_before_start_event"),
            "synchronized_before_event_samples": True,
            "included_in_event_interval": False,
        }
        if power_warmup == legacy_power_warmup:
            expected_power_state_condition = (
                "sustained_bf16_gemm_before_event_samples")
        elif power_warmup == maintained_power_warmup:
            expected_power_state_condition = (
                "sustained_bf16_gemm_before_each_event_interval")
        else:
            raise ValueError(
                f"invalid Ascend Vector device power warmup for {case_id}")
        if record["timer"]["power_state_condition"] != (
                expected_power_state_condition):
            raise ValueError(
                f"invalid Ascend Vector power-state condition for {case_id}")
    elif ("power_state_condition" in record["timer"]
          and record["timer"]["power_state_condition"] != "ordinary_warm_state"):
        raise ValueError(
            f"unexpected Ascend Vector power-state condition for {case_id}")
    op = canonical_attributes["op"]
    expected_dynamic_input_operands = None
    expected_static_input_operands = None
    expected_input_bytes_per_operand = None
    if op == "rmsnorm":
        input_elements = canonical_attributes["tok"] * canonical_attributes[
            "hidden"]
        expected_ring_mode = "rotating_dense_input_backing_ring"
    elif op == "silu":
        input_elements = (
            canonical_attributes["tok"]
            * 2 * canonical_attributes["intermediate"])
        expected_ring_mode = "rotating_dense_input_backing_ring"
    elif op == "add_rmsnorm":
        input_elements = (
            2 * canonical_attributes["tok"] * canonical_attributes["hidden"])
        expected_ring_mode = "rotating_paired_dense_input_backing_ring"
        expected_dynamic_input_operands = ["x", "residual"]
        expected_static_input_operands = ["weight"]
    elif op == "elementwise":
        input_elements = (
            2 * canonical_attributes["tok"] * canonical_attributes["hidden"])
        expected_ring_mode = "rotating_paired_dense_input_backing_ring"
        expected_dynamic_input_operands = ["left", "right"]
        expected_static_input_operands = []
    elif op == "rope":
        q_elements = (
            canonical_attributes["tok"] * canonical_attributes["num_heads"]
            * canonical_attributes["head_dim"])
        k_elements = (
            canonical_attributes["tok"] * canonical_attributes["num_kv_heads"]
            * canonical_attributes["head_dim"])
        input_elements = q_elements + k_elements
        expected_ring_mode = "rotating_asymmetric_paired_dense_input_backing_ring"
        expected_dynamic_input_operands = ["q_flat", "k_flat"]
        expected_static_input_operands = ["positions", "cos_sin_cache"]
        expected_input_bytes_per_operand = {
            "q_flat": q_elements * canonical_attributes["dtype_bytes"],
            "k_flat": k_elements * canonical_attributes["dtype_bytes"],
        }
    elif op == "kvcache":
        key_elements = (
            canonical_attributes["tok"]
            * canonical_attributes["num_kv_heads"]
            * canonical_attributes["head_dim"])
        cache_block_size = 128
        cache_elements = (
            math.ceil(canonical_attributes["tok"] / cache_block_size)
            * cache_block_size * canonical_attributes["num_kv_heads"]
            * canonical_attributes["head_dim"])
        input_elements = 2 * (key_elements + cache_elements)
        expected_ring_mode = "rotating_kvcache_input_and_destination_ring"
        expected_dynamic_input_operands = [
            "key", "value", "key_cache", "value_cache"]
        expected_static_input_operands = ["slot_indices"]
        expected_input_bytes_per_operand = {
            "key": key_elements * canonical_attributes["dtype_bytes"],
            "value": key_elements * canonical_attributes["dtype_bytes"],
            "key_cache": cache_elements * canonical_attributes["dtype_bytes"],
            "value_cache": cache_elements * canonical_attributes["dtype_bytes"],
        }
    else:
        raise ValueError(
            "Ascend rotating input ring is only valid for RMSNorm, "
            "AddRMSNorm, RoPE, SiLU, Elementwise, or KV-cache: "
            f"{case_id}")
    if input_ring_mode != expected_ring_mode:
        raise ValueError(f"Ascend Vector input-ring op mode mismatch: {case_id}")
    if (expected_dynamic_input_operands is not None
            and (input_ring["dynamic_input_operands"]
                 != expected_dynamic_input_operands
                 or input_ring["static_input_operands"]
                 != expected_static_input_operands)):
        raise ValueError(
            f"invalid {op} paired-input identities for {case_id}")
    input_bytes = input_elements * canonical_attributes["dtype_bytes"]
    requested_bytes = input_ring["requested_bytes"]
    allocated_bytes = input_ring["allocated_bytes"]
    slot_count = input_ring["slot_count"]
    slot_stride = input_ring["slot_stride"]
    for name, value in (
            ("requested_bytes", requested_bytes),
            ("allocated_bytes", allocated_bytes),
            ("input_bytes_per_call", input_ring["input_bytes_per_call"]),
            ("slot_count", slot_count), ("slot_stride", slot_stride)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"invalid Ascend Vector input-ring {name} for {case_id}")
    if input_ring_mode == "rotating_paired_dense_input_backing_ring":
        minimum_requested_bytes = 2048 * 1024 * 1024
    elif input_ring_mode in {
            "rotating_dense_input_backing_ring",
            "rotating_asymmetric_paired_dense_input_backing_ring"}:
        minimum_requested_bytes = 1536 * 1024 * 1024
    elif input_ring_mode == "rotating_kvcache_input_and_destination_ring":
        minimum_requested_bytes = 2048 * 1024 * 1024
    else:
        raise AssertionError(
            f"validated input-ring mode was not dispatched: {case_id}")
    if requested_bytes < minimum_requested_bytes:
        raise ValueError(
            f"Ascend Vector input ring is not clearly larger than L2: {case_id}")
    if input_ring["input_bytes_per_call"] != input_bytes:
        raise ValueError(f"Ascend Vector input-ring shape mismatch: {case_id}")
    if (input_ring_mode == "rotating_paired_dense_input_backing_ring"
            and input_ring["input_bytes_per_operand"] * 2 != input_bytes):
        raise ValueError(
            f"Ascend Vector paired input-ring operand size mismatch: {case_id}")
    if (expected_input_bytes_per_operand is not None
            and input_ring["input_bytes_per_operand"]
            != expected_input_bytes_per_operand):
        raise ValueError(
            f"Ascend Vector input-ring operand sizes mismatch: {case_id}")
    expected_slot_count = math.ceil(requested_bytes / input_bytes)
    if slot_count != expected_slot_count:
        raise ValueError(f"Ascend Vector input-ring slot count mismatch: {case_id}")
    if math.gcd(slot_stride, slot_count) != 1:
        raise ValueError(f"Ascend Vector input-ring stride is not coprime: {case_id}")
    if allocated_bytes != slot_count * input_bytes:
        raise ValueError(f"Ascend Vector input-ring allocation mismatch: {case_id}")


def _validate_a800_vector_input_ring(
        *, record: Mapping[str, Any], case_id: str,
        canonical_attributes: Mapping[str, Any]) -> None:
    """Validate an explicit A800 batched-Event rotating-buffer protocol."""

    workload = record["workload"]
    input_ring = workload["input_ring"]
    v1_expected_keys = {
        "mode", "hardware_l2_bytes", "minimum_input_ring_bytes",
        "requested_input_ring_bytes", "allocated_input_ring_bytes",
        "input_bytes_per_public_call", "dynamic_input_bytes_per_operand",
        "preallocated_output_bytes_per_operand",
        "allocated_output_ring_bytes", "output_padding_bytes",
        "in_place_output_operands", "public_call_count", "slot_count",
        "slot_stride", "selection", "initialization", "pre_target_sweep",
    }
    v2_expected_keys = v1_expected_keys | {
        "protocol_version", "reuse_contract"}
    v3_expected_keys = (v1_expected_keys - {"public_call_count"}) | {
        "protocol_version", "reuse_contract",
        "graph_cycle_public_call_count", "graph_repetition_counts"}
    differential_expected_keys = (v1_expected_keys - {"public_call_count"}) | {
        "protocol_version", "reuse_contract",
        "minimum_distinct_input_bytes",
        "lower_graph_public_call_count", "higher_graph_public_call_count",
        "graph_repetition_counts"}
    if not isinstance(input_ring, dict):
        raise ValueError(f"invalid A800 Vector input ring: {case_id}")
    if set(input_ring) == v1_expected_keys:
        protocol_version = 1
        expected_mode = "rotating_distinct_preallocated_vector_buffers"
        expected_selection = (
            "coprime_stride_without_reuse_within_each_event_batch")
        allowed_public_call_counts = {32}
        expected_timer_mode = (
            "fixed_count_distinct_public_ops_per_event_interval_"
            "reported_per_call")
    elif set(input_ring) == v2_expected_keys:
        protocol_version = input_ring["protocol_version"]
        if (isinstance(protocol_version, bool)
                or not isinstance(protocol_version, int)
                or protocol_version != 2):
            raise ValueError(
                f"invalid A800 Vector input-ring protocol: {case_id}")
        if input_ring["reuse_contract"] != (
                "reuse_only_after_full_input_ring_cycle"):
            raise ValueError(
                f"invalid A800 Vector input-ring reuse contract: {case_id}")
        expected_mode = (
            "rotating_preallocated_vector_buffers_full_ring_cycle_reuse")
        expected_selection = (
            "coprime_stride_reuse_only_after_full_input_ring_cycle")
        allowed_public_call_counts = {32, 128, 512, 2048}
        expected_timer_mode = (
            "fixed_count_public_ops_over_full_ring_cycle_per_event_interval_"
            "reported_per_call")
    elif set(input_ring) == v3_expected_keys:
        protocol_version = input_ring["protocol_version"]
        if (isinstance(protocol_version, bool)
                or not isinstance(protocol_version, int)
                or protocol_version != 3):
            raise ValueError(
                f"invalid A800 Vector CUDA-Graph protocol: {case_id}")
        if input_ring["reuse_contract"] != (
                "reuse_only_after_full_input_ring_cycle_and_between_graph_"
                "replays"):
            raise ValueError(
                f"invalid A800 Vector CUDA-Graph reuse contract: {case_id}")
        expected_mode = (
            "rotating_preallocated_vector_buffers_full_ring_cuda_graph_replay")
        expected_selection = (
            "fixed_coprime_full_ring_cycle_inside_each_cuda_graph_replay")
        expected_timer_mode = (
            "external_cuda_events_around_full_ring_cuda_graph_replay_"
            "reported_per_public_call")
    elif set(input_ring) == differential_expected_keys:
        protocol_version = input_ring["protocol_version"]
        if (isinstance(protocol_version, bool)
                or not isinstance(protocol_version, int)
                or protocol_version not in {4, 5}):
            raise ValueError(
                f"invalid A800 Vector differential CUDA-Graph protocol: "
                f"{case_id}")
        if input_ring["reuse_contract"] != (
                "no_dynamic_input_slot_reuse_within_either_graph"):
            raise ValueError(
                "invalid A800 Vector differential CUDA-Graph reuse "
                f"contract: {case_id}")
        expected_mode = (
            "unique_preallocated_vector_buffers_paired_differential_"
            "cuda_graph")
        expected_selection = (
            "coprime_unique_lower_prefix_and_equal_unique_higher_suffix")
        expected_timer_mode = (
            "paired_differential_events_around_cuda_graph_replay")
    else:
        raise ValueError(f"invalid A800 Vector input ring: {case_id}")
    if input_ring["mode"] != expected_mode:
        raise ValueError(f"invalid A800 Vector input-ring mode: {case_id}")
    if input_ring["hardware_l2_bytes"] != 40 * 1024 * 1024:
        raise ValueError(f"invalid A800 Vector L2 size: {case_id}")
    if input_ring["minimum_input_ring_bytes"] != 320 * 1024 * 1024:
        raise ValueError(f"invalid A800 Vector minimum input ring: {case_id}")
    if input_ring["selection"] != expected_selection:
        raise ValueError(f"invalid A800 Vector input selection: {case_id}")
    if input_ring["initialization"] != "torch.randn_or_zeros_before_timing":
        raise ValueError(f"invalid A800 Vector input initialization: {case_id}")
    if input_ring["pre_target_sweep"] != "none":
        raise ValueError(f"invalid A800 Vector pre-target sweep: {case_id}")
    if protocol_version in {1, 2}:
        public_call_count = input_ring["public_call_count"]
        if (isinstance(public_call_count, bool)
                or not isinstance(public_call_count, int)
                or public_call_count not in allowed_public_call_counts):
            raise ValueError(f"invalid A800 Vector public call count: {case_id}")

    def positive_attribute(name: str) -> int:
        value = canonical_attributes[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"invalid A800 Vector canonical {name}: {case_id}")
        return value

    op = canonical_attributes["op"]
    dtype_bytes = positive_attribute("dtype_bytes")
    tok = positive_attribute("tok")
    if op in {"rmsnorm", "add_rmsnorm", "elementwise"}:
        tensor_bytes = tok * positive_attribute("hidden") * dtype_bytes
        if op == "rmsnorm":
            dynamic_inputs = {"x": tensor_bytes}
            outputs = {"output": tensor_bytes}
            in_place_outputs = []
        elif op == "add_rmsnorm":
            dynamic_inputs = {"x": tensor_bytes, "residual": tensor_bytes}
            outputs = {}
            in_place_outputs = ["x", "residual"]
        elif op == "elementwise":
            dynamic_inputs = {"left": tensor_bytes, "right": tensor_bytes}
            outputs = {"output": tensor_bytes}
            in_place_outputs = []
        else:
            raise AssertionError("validated A800 Vector op was not dispatched")
        padded_cache_output = False
    elif op == "silu":
        intermediate = positive_attribute("intermediate")
        dynamic_inputs = {"input_tensor": tok * 2 * intermediate * dtype_bytes}
        outputs = {"output": tok * intermediate * dtype_bytes}
        in_place_outputs = []
        padded_cache_output = False
    elif op == "rope":
        head_dim = positive_attribute("head_dim")
        dynamic_inputs = {
            "q": tok * positive_attribute("num_heads") * head_dim * dtype_bytes,
            "k": tok * positive_attribute("num_kv_heads") * head_dim * dtype_bytes,
        }
        outputs = {}
        in_place_outputs = ["q", "k"]
        padded_cache_output = False
    elif op == "kvcache":
        tensor_bytes = (
            tok * positive_attribute("num_kv_heads")
            * positive_attribute("head_dim") * dtype_bytes)
        dynamic_inputs = {"key": tensor_bytes, "value": tensor_bytes}
        outputs = {
            "key_cache_destination": tensor_bytes,
            "value_cache_destination": tensor_bytes,
        }
        in_place_outputs = []
        padded_cache_output = True
    else:
        raise ValueError(f"unsupported A800 Vector op: {op!r}")

    requested_bytes = input_ring["requested_input_ring_bytes"]
    if (isinstance(requested_bytes, bool)
            or not isinstance(requested_bytes, int)
            or requested_bytes < 320 * 1024 * 1024
            or requested_bytes % (40 * 1024 * 1024) != 0):
        raise ValueError(f"invalid A800 Vector input-ring size: {case_id}")
    input_bytes = sum(dynamic_inputs.values())
    requested_slot_count = math.ceil(requested_bytes / input_bytes)
    if protocol_version == 1:
        expected_slot_count = max(32, requested_slot_count)
    elif protocol_version in {2, 3}:
        expected_slot_count = requested_slot_count
    elif protocol_version in {4, 5}:
        minimum_distinct_input_bytes = input_ring[
            "minimum_distinct_input_bytes"]
        if (isinstance(minimum_distinct_input_bytes, bool)
                or not isinstance(minimum_distinct_input_bytes, int)
                or minimum_distinct_input_bytes < 320 * 1024 * 1024
                or minimum_distinct_input_bytes % (40 * 1024 * 1024) != 0
                or requested_bytes != 2 * minimum_distinct_input_bytes):
            raise ValueError(
                "invalid A800 Vector differential distinct-input contract: "
                f"{case_id}")
        expected_lower_count = math.ceil(
            minimum_distinct_input_bytes / input_bytes)
        expected_slot_count = 2 * expected_lower_count
    else:
        raise AssertionError("validated A800 Vector protocol was not dispatched")
    expected_stride = (
        1 if expected_slot_count == 1 else expected_slot_count // 2 + 1)
    if expected_stride >= expected_slot_count and expected_slot_count != 1:
        expected_stride = 1
    while math.gcd(expected_stride, expected_slot_count) != 1:
        expected_stride += 1
        if expected_stride >= expected_slot_count:
            expected_stride = 1
    if input_ring["dynamic_input_bytes_per_operand"] != dynamic_inputs:
        raise ValueError(f"A800 Vector input operands mismatch: {case_id}")
    if input_ring["preallocated_output_bytes_per_operand"] != outputs:
        raise ValueError(f"A800 Vector output operands mismatch: {case_id}")
    if input_ring["in_place_output_operands"] != in_place_outputs:
        raise ValueError(f"A800 Vector in-place outputs mismatch: {case_id}")
    if input_ring["input_bytes_per_public_call"] != input_bytes:
        raise ValueError(f"A800 Vector input byte count mismatch: {case_id}")
    if input_ring["slot_count"] != expected_slot_count:
        raise ValueError(f"A800 Vector input-ring slot count mismatch: {case_id}")
    if input_ring["slot_stride"] != expected_stride:
        raise ValueError(f"A800 Vector input-ring stride mismatch: {case_id}")
    if input_ring["allocated_input_ring_bytes"] != (
            expected_slot_count * input_bytes):
        raise ValueError(f"A800 Vector input allocation mismatch: {case_id}")
    logical_output_bytes = expected_slot_count * sum(outputs.values())
    if padded_cache_output:
        cache_slot_count = math.ceil(expected_slot_count * tok / 16) * 16
        expected_output_bytes = (
            cache_slot_count * positive_attribute("num_kv_heads")
            * positive_attribute("head_dim") * dtype_bytes * 2)
    else:
        expected_output_bytes = logical_output_bytes
    if input_ring["allocated_output_ring_bytes"] != expected_output_bytes:
        raise ValueError(f"A800 Vector output allocation mismatch: {case_id}")
    if input_ring["output_padding_bytes"] != (
            expected_output_bytes - logical_output_bytes):
        raise ValueError(f"A800 Vector output padding mismatch: {case_id}")
    batch = workload["event_operation_batch"]
    if protocol_version == 1:
        expected_batch = {
            "public_call_count": 32,
            "input_selection": (
                "advance_coprime_ring_once_per_public_call_without_"
                "within_batch_reuse"),
            "duration_reporting": (
                "aggregate_current_stream_event_duration_divided_by_"
                "public_call_count"),
        }
    elif protocol_version == 2:
        expected_batch = {
            "protocol_version": 2,
            "public_call_count": public_call_count,
            "public_call_count_selection": (
                "explicit_cli_value_not_duration_based"),
            "input_selection": (
                "advance_coprime_ring_once_per_public_call_reuse_only_after_"
                "full_input_ring_cycle"),
            "duration_reporting": (
                "aggregate_current_stream_event_duration_divided_by_"
                "public_call_count"),
        }
    elif protocol_version == 3:
        expected_batch = {
            "protocol_version": 3,
            "graph_backend": "torch.cuda.CUDAGraph",
            "input_selection": (
                "fixed_coprime_full_ring_cycle_inside_each_cuda_graph_replay"),
            "graph_cycle_public_call_count": expected_slot_count,
            "graph_repetition_counts": [
                expected_slot_count, 2 * expected_slot_count],
            "duration_reporting": (
                "external_cuda_event_around_full_ring_graph_replay_divided_by_"
                "graph_public_call_count"),
        }
        if input_ring["graph_cycle_public_call_count"] != expected_slot_count:
            raise ValueError(
                f"invalid A800 Vector CUDA-Graph cycle count: {case_id}")
        if input_ring["graph_repetition_counts"] != [
                expected_slot_count, 2 * expected_slot_count]:
            raise ValueError(
                f"invalid A800 Vector CUDA-Graph repetitions: {case_id}")
    elif protocol_version in {4, 5}:
        expected_batch = {
            "protocol_version": protocol_version,
            "graph_backend": "torch.cuda.CUDAGraph",
            "input_selection": (
                "shared_unique_lower_prefix_plus_equal_unique_higher_suffix"),
            "lower_graph_public_call_count": expected_lower_count,
            "higher_graph_public_call_count": expected_slot_count,
            "graph_repetition_counts": [
                expected_lower_count, expected_slot_count],
            "sample_pairing": (
                "alternating_lower_then_higher_and_higher_then_lower"),
            "duration_reporting": "(T_high-T_low)/(N_high-N_low)",
        }
        if (input_ring["lower_graph_public_call_count"]
                != expected_lower_count
                or input_ring["higher_graph_public_call_count"]
                != expected_slot_count
                or input_ring["graph_repetition_counts"] != [
                    expected_lower_count, expected_slot_count]):
            raise ValueError(
                "invalid A800 Vector differential CUDA-Graph repetitions: "
                f"{case_id}")
    else:
        raise AssertionError("validated A800 Vector protocol was not dispatched")
    if batch != expected_batch:
        raise ValueError(f"invalid A800 Vector Event batch contract: {case_id}")
    measurement = record["measurement"]
    if protocol_version in {4, 5}:
        if workload["event_measurement_mode"] != "cuda_graph_differential":
            raise ValueError(
                "invalid A800 Vector differential CUDA-Graph workload mode: "
                f"{case_id}")
        if (measurement["schema_version"] != 3
                or measurement["mode"]
                != "paired_differential_events_around_cuda_graph_replay"
                or measurement["operation_cycle_length"]
                != expected_slot_count
                or measurement["repetition_counts"] != [
                    expected_lower_count, expected_slot_count]):
            raise ValueError(
                "invalid A800 Vector differential CUDA-Graph measurement: "
                f"{case_id}")
        distinct_input = measurement["distinct_input_contract"]
        if (distinct_input["hardware_l2_bytes"] != 40 * 1024 * 1024
                or distinct_input["configured_minimum_bytes"]
                != minimum_distinct_input_bytes
                or distinct_input["per_call_bytes"] != input_bytes
                or distinct_input["lower_replay_bytes"]
                != expected_lower_count * input_bytes
                or distinct_input["higher_replay_bytes"]
                != expected_slot_count * input_bytes
                or distinct_input["delta_replay_bytes"]
                != expected_lower_count * input_bytes):
            raise ValueError(
                "invalid A800 Vector differential distinct-input "
                f"measurement: {case_id}")
        if (measurement["cache_eviction_requested_bytes"] != 0
                or measurement["cache_eviction_bytes_before_replay"] != 0
                or workload["cache_eviction_bytes_before_replay"] != 0):
            raise ValueError(
                "A800 Vector differential CUDA-Graph ring cannot mix cache "
                f"eviction: {case_id}")
        if protocol_version == 5:
            expected_power_warmup = {
                "kind": (
                    "bf16_gemm_power_warmup_and_per_graph_replay_maintenance"),
                "matrix_dimension": 4096,
                "initial_gemm_iterations": 1000,
                "per_replay_gemm_iterations": 1,
                "per_replay_placement": (
                    "same_current_stream_immediately_before_start_event"),
                "synchronized_before_graph_measurement": True,
                "included_in_event_interval": False,
            }
            if workload["power_warmup"] != expected_power_warmup:
                raise ValueError(
                    f"invalid A800 Vector power warmup: {case_id}")
            if record["timer"]["power_state_condition"] != (
                    "sustained_bf16_gemm_before_each_graph_replay"):
                raise ValueError(
                    f"invalid A800 Vector power state: {case_id}")
            if measurement[
                    "pre_event_operation_executions_per_replay"] != 1:
                raise ValueError(
                    f"invalid A800 Vector replay power maintenance: {case_id}")
        elif "power_warmup" in workload:
            raise ValueError(
                f"legacy A800 Vector protocol claims power warmup: {case_id}")
    elif protocol_version == 3:
        if workload["event_measurement_mode"] != "cuda_graph_full_ring_replay":
            raise ValueError(
                f"invalid A800 Vector CUDA-Graph workload mode: {case_id}")
        if (measurement["schema_version"] != 3
                or measurement["mode"]
                != "external_events_around_cuda_graph_replay"
                or measurement["operation_cycle_length"]
                != expected_slot_count
                or measurement["repetition_counts"] != [
                    expected_slot_count, 2 * expected_slot_count]):
            raise ValueError(
                f"invalid A800 Vector CUDA-Graph measurement: {case_id}")
        if (measurement["cache_eviction_requested_bytes"] != 0
                or measurement["cache_eviction_bytes_before_replay"] != 0
                or workload["cache_eviction_bytes_before_replay"] != 0):
            raise ValueError(
                "A800 Vector CUDA-Graph ring cannot mix cache eviction: "
                f"{case_id}")
    else:
        if (measurement["schema_version"]
                != BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION
                or measurement["operation_repetitions_per_sample"]
                != public_call_count
                or "mode" in measurement):
            raise ValueError(f"invalid A800 Vector Event measurement: {case_id}")
        if (measurement["cache_eviction_requested_bytes"] != 0
                or measurement["cache_eviction_bytes_before_sample"] != 0):
            raise ValueError(
                f"A800 Vector input ring cannot mix cache eviction: {case_id}")
    if record["timer"]["mode"] != expected_timer_mode:
        raise ValueError(f"invalid A800 Vector timer mode: {case_id}")


# The rotating-input-ring protocol each hardware's Vector Event data uses.
# A hardware absent from this map has no such protocol, so an observed ring is
# a validation failure rather than something to accept unchecked.
_INPUT_RING_VALIDATORS = {
    "nvidia_a800": _validate_a800_vector_input_ring,
    "ascend_910b": _validate_ascend_vector_input_ring,
}


def _validate_case_record(
        *, record: Mapping[str, Any], path: Path, hardware_id: str,
        family: str, case_id: str, baseline_commit: str,
        source_snapshot_sha256: str, timer_sanity_sha256: str,
        canonical_attributes: Mapping[str, Any],
        runner_root: Path | None = None) -> list[float] | None:
    if set(record) != EVENT_CASE_REQUIRED_KEYS:
        raise ValueError(
            f"Event case record has unexpected schema keys for {case_id}: "
            f"{sorted(set(record) ^ EVENT_CASE_REQUIRED_KEYS)}")
    expected_scalars = {
        "schema_version": 1,
        "hardware_id": hardware_id,
        "family": family,
        "case_id": case_id,
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
        "canonical_attributes_sha256": _canonical_mapping_sha256(
            canonical_attributes),
    }
    for name, expected in expected_scalars.items():
        if record[name] != expected:
            raise ValueError(
                f"Event case {case_id} field {name} mismatch: "
                f"expected={expected!r}, actual={record[name]!r}")
    if path.stem != case_id:
        raise ValueError(
            f"Event case filename must equal case_id: {path.name}")
    runner_path = Path(record["runner_path"])
    if runner_path.is_absolute() or ".." in runner_path.parts:
        raise ValueError(f"Event runner path must be repository-relative: {case_id}")
    repository_root = (
        Path(__file__).resolve().parents[2]
        if runner_root is None else runner_root.resolve())
    resolved_runner = repository_root / runner_path
    if not resolved_runner.is_file():
        raise FileNotFoundError(f"recorded Event runner is absent: {resolved_runner}")
    if record["runner_sha256"] != _sha256(resolved_runner):
        raise ValueError(f"recorded Event runner SHA-256 mismatch: {case_id}")
    for name in ("workload", "environment"):
        if not isinstance(record[name], dict) or not record[name]:
            raise TypeError(f"Event case {case_id} {name} must be non-empty")
    status = record["status"]
    if status == "measured":
        if record["unsupported_reason"] is not None:
            raise ValueError(f"measured case has unsupported reason: {case_id}")
        for name in ("timer", "stream_contract", "allocation_contract"):
            if not isinstance(record[name], dict) or not record[name]:
                raise TypeError(
                    f"measured case {case_id} {name} must be non-empty")
        durations = _validate_event_measurement(
            record["measurement"], case_id)
        measurement = record["measurement"]
        has_vector_input_ring = "input_ring" in record["workload"]
        if has_vector_input_ring:
            if family != "vector":
                raise ValueError(
                    "rotating input ring is only valid for Vector: "
                    f"{case_id}")
            is_a800_validated_graph = (
                hardware_id == "nvidia_a800"
                and measurement["schema_version"]
                == GRAPH_REPLAY_EVENT_SAMPLE_SCHEMA_VERSION
                and measurement["mode"] in {
                    "external_events_around_cuda_graph_replay",
                    DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE,
                })
            if (not is_a800_validated_graph
                    and (measurement["schema_version"] not in {
                        CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
                        BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION}
                         or "mode" in measurement)):
                raise ValueError(
                    "rotating input ring requires direct queued Event schema "
                    "3 or 4, or the validated A800 CUDA-Graph replay "
                    f"schema: {case_id}")
            if hardware_id not in _INPUT_RING_VALIDATORS:
                raise ValueError(
                    "rotating input ring is unsupported on this hardware: "
                    f"{case_id}")
            _INPUT_RING_VALIDATORS[hardware_id](
                record=record, case_id=case_id,
                canonical_attributes=canonical_attributes)
            if hardware_id == "ascend_910b":
                if (measurement["cache_eviction_requested_bytes"] != 0
                        or measurement[
                            "cache_eviction_bytes_before_sample"] != 0):
                    raise ValueError(
                        "Ascend rotating input ring cannot mix cache eviction: "
                        f"{case_id}")
        if measurement["schema_version"] in {
                LEGACY_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
                CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION,
                BATCHED_CURRENT_STREAM_EVENT_SAMPLE_SCHEMA_VERSION}:
            if "mode" in measurement:
                cache_key = "cache_eviction_bytes_before_replay"
                other_cache_key = "cache_eviction_bytes_before_sample"
                if measurement["mode"] == DIFFERENTIAL_GRAPH_REPLAY_EVENT_MODE:
                    swept_condition = (
                        "queued_pre_replay_device_sweep_without_intra_replay_"
                        "buffer_reuse")
                else:
                    swept_condition = (
                        "queued_pre_replay_device_sweep_with_intra_replay_reuse"
                        if measurement["schema_version"] == 3
                        else "pre_replay_device_sweep_with_intra_replay_reuse")
            else:
                cache_key = "cache_eviction_bytes_before_sample"
                other_cache_key = "cache_eviction_bytes_before_replay"
                swept_condition = (
                    "queued_pre_sample_device_sweep"
                    if measurement["schema_version"] == 3
                    else "pre_sample_device_sweep")
            cache_eviction_bytes = measurement[cache_key]
            if record["workload"][cache_key] != cache_eviction_bytes:
                raise ValueError(
                    f"Event case {case_id} workload {cache_key} mismatch")
            if other_cache_key in record["workload"]:
                raise ValueError(
                    f"Event case {case_id} workload mixes cache conditions")
            expected_cache_condition = (
                record["workload"]["input_ring"]["mode"]
                if has_vector_input_ring else (
                    swept_condition
                    if cache_eviction_bytes else "ordinary_warm_state"))
            if record["timer"][
                    "cache_condition"] != expected_cache_condition:
                raise ValueError(
                    f"Event case {case_id} timer cache condition mismatch")
        return durations
    elif status == "unsupported":
        reason = record["unsupported_reason"]
        if not isinstance(reason, str) or not reason:
            raise TypeError(f"unsupported case needs a reason: {case_id}")
        for name in ("timer", "stream_contract", "allocation_contract",
                     "measurement"):
            if record[name] is not None:
                raise ValueError(
                    f"unsupported case {case_id} must normalize {name} to null")
        return None
    else:
        raise ValueError(f"unknown Event case status for {case_id}: {status!r}")


def _group_summary(comparisons: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    measured = [row for row in comparisons if row["status"] == "measured"]
    unsupported = [
        row for row in comparisons if row["status"] == "unsupported"]
    result: dict[str, Any] = {
        "selected_case_count": len(comparisons),
        "measured_case_count": len(measured),
        "unsupported_case_count": len(unsupported),
        "unsupported_reasons": sorted({
            row["unsupported_reason"] for row in unsupported}),
    }
    if measured:
        absolute = [row["absolute_error_ns"] for row in measured]
        relative = [row["relative_error"] for row in measured]
        signed = [row["signed_relative_error"] for row in measured]
        cvs = [
            row["event_statistics"]["coefficient_of_variation"]
            for row in measured]
        result["error_statistics"] = {
            "mean_absolute_error_ns": statistics.fmean(absolute),
            "median_absolute_error_ns": statistics.median(absolute),
            "mean_relative_error": statistics.fmean(relative),
            "median_relative_error": statistics.median(relative),
            "maximum_relative_error": max(relative),
            "mean_signed_relative_error": statistics.fmean(signed),
            "median_event_coefficient_of_variation": statistics.median(cvs),
            "maximum_event_coefficient_of_variation": max(cvs),
        }
    else:
        result["error_statistics"] = None
    return result


def _summaries_by_key(
        comparisons: list[Mapping[str, Any]], key_names: tuple[str, ...]
        ) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in comparisons:
        key = tuple(row[name] for name in key_names)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)
    result = []
    for key, rows in sorted(grouped.items()):
        identity = dict(zip(key_names, key, strict=True))
        result.append({**identity, **_group_summary(rows)})
    return result


OP_CANDIDATE_SELECTION_SCHEMA_VERSION = 1
OP_CANDIDATE_SELECTION_KIND = "kernel_golden_op_candidate_selection"
OP_CANDIDATE_REPORT_KIND = "kernel_golden_op_time_candidate_report"
_EVENT_SOURCE_KEYS = frozenset({
    "kind", "case_directory", "timer_sanity_path",
    "source_snapshot_archive", "source_snapshot_root", "runner_root",
})
_TIMELINE_SOURCE_KEYS = frozenset({
    "kind", "candidate_path", "source_snapshot_archive",
    "source_snapshot_root", "runner_root",
})
_SELECTION_MANIFEST_KEYS = frozenset({
    "schema_version", "artifact_kind", "hardware_id", "families",
    "sources", "selections", "stability_policy",
})
_SELECTION_KEYS = frozenset({"family", "case_id", "source_id"})
_STABILITY_POLICY_KEYS = frozenset({
    "event_coefficient_of_variation_maximum",
    "event_maximum_relative_deviation_maximum",
    "timeline_relative_mad_maximum",
    "timeline_maximum_relative_deviation_maximum",
})


def _require_exact_keys(value: Any, expected: frozenset[str], context: str
                        ) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    if set(value) != expected:
        raise ValueError(
            f"{context} keys mismatch: "
            f"{sorted(set(value) ^ expected)}")
    return value


def _resolve_selection_path(value: Any, manifest_path: Path,
                            context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context} must be a non-empty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _require_existing_directory(path: Path, context: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{context} is not a directory: {path}")
    return path


def _require_existing_file(path: Path, context: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is not a file: {path}")
    return path


def _load_op_candidate_selection(
        selection_manifest_path: Path
        ) -> tuple[Mapping[str, Any], Path]:
    manifest_path = selection_manifest_path.resolve()
    payload = _require_exact_keys(
        _load_json_mapping(manifest_path, "OP candidate selection manifest"),
        _SELECTION_MANIFEST_KEYS, "OP candidate selection manifest")
    if payload["schema_version"] != OP_CANDIDATE_SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported OP candidate selection manifest schema")
    if payload["artifact_kind"] != OP_CANDIDATE_SELECTION_KIND:
        raise ValueError("invalid OP candidate selection manifest kind")
    hardware_id = payload["hardware_id"]
    if hardware_id not in HARDWARE_IDS:
        raise ValueError(f"unknown OP candidate hardware: {hardware_id!r}")
    families = payload["families"]
    if (not isinstance(families, list) or not families
            or any(not isinstance(family, str) for family in families)):
        raise TypeError("OP candidate families must be a non-empty string list")
    if len(set(families)) != len(families):
        raise ValueError("OP candidate families cannot repeat")
    unknown_families = set(families) - set(DATASET_FAMILIES)
    if unknown_families:
        raise ValueError(f"unknown OP candidate families: {sorted(unknown_families)}")
    sources = payload["sources"]
    if not isinstance(sources, dict) or not sources:
        raise TypeError("OP candidate sources must be a non-empty mapping")
    for source_id, source in sources.items():
        if not isinstance(source_id, str) or not source_id:
            raise TypeError("OP candidate source IDs must be non-empty strings")
        if not isinstance(source, dict):
            raise TypeError(f"OP candidate source {source_id!r} must be a mapping")
        kind = source["kind"]
        if kind == "event_cases":
            _require_exact_keys(
                source, _EVENT_SOURCE_KEYS,
                f"OP candidate Event source {source_id!r}")
        elif kind == "ascend_timeline_candidate":
            _require_exact_keys(
                source, _TIMELINE_SOURCE_KEYS,
                f"OP candidate Timeline source {source_id!r}")
        else:
            raise ValueError(
                f"unsupported OP candidate source kind for {source_id!r}: "
                f"{kind!r}")
    selections = payload["selections"]
    if not isinstance(selections, list) or not selections:
        raise TypeError("OP candidate selections must be a non-empty list")
    seen_cases = set()
    for selection in selections:
        _require_exact_keys(selection, _SELECTION_KEYS, "OP candidate selection")
        family = selection["family"]
        case_id = selection["case_id"]
        source_id = selection["source_id"]
        if family not in families:
            raise ValueError(
                f"OP candidate selection family is outside scope: {family!r}")
        if not isinstance(case_id, str) or not case_id:
            raise TypeError("OP candidate case IDs must be non-empty strings")
        if source_id not in sources:
            raise KeyError(
                f"OP candidate selection references unknown source {source_id!r}")
        identity = (family, case_id)
        if identity in seen_cases:
            raise ValueError(f"OP candidate selection repeats {identity!r}")
        seen_cases.add(identity)
    policy = _require_exact_keys(
        payload["stability_policy"], _STABILITY_POLICY_KEYS,
        "OP candidate stability policy")
    for name, value in policy.items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(
                f"OP candidate stability policy {name} must be in (0, 1]")
    return payload, manifest_path


def _source_provenance(
        *, source: Mapping[str, Any], manifest_path: Path,
        source_id: str) -> Mapping[str, Any]:
    snapshot_archive = _require_existing_file(
        _resolve_selection_path(
            source["source_snapshot_archive"], manifest_path,
            f"source archive for {source_id}"),
        f"source archive for {source_id}")
    archive_sha256 = _sha256(snapshot_archive)
    snapshot_root = _require_existing_directory(
        _resolve_selection_path(
            source["source_snapshot_root"], manifest_path,
            f"source snapshot root for {source_id}"),
        f"source snapshot root for {source_id}")
    snapshot_digest_path = snapshot_root / "SOURCE_SNAPSHOT_SHA256"
    if snapshot_digest_path.exists():
        snapshot_digest = _require_existing_file(
            snapshot_digest_path,
            f"source snapshot digest for {source_id}").read_text(
                encoding="utf-8").strip()
        if snapshot_digest != archive_sha256:
            raise ValueError(
                f"source archive does not match snapshot provenance for "
                f"{source_id}")
        snapshot_digest_path_text: str | None = str(snapshot_digest_path)
    else:
        # Early frozen snapshots predate the sidecar.  Their Event records and
        # timer sanity still carry the archive digest, and the runner is hashed
        # below from the source tree under this snapshot root.
        snapshot_digest_path_text = None
    runner_root = _require_existing_directory(
        _resolve_selection_path(
            source["runner_root"], manifest_path,
            f"runner root for {source_id}"),
        f"runner root for {source_id}")
    if not runner_root.is_relative_to(snapshot_root):
        raise ValueError(
            f"runner root is outside source snapshot root for {source_id}")
    return {
        "source_id": source_id,
        "source_kind": source["kind"],
        "source_snapshot_archive": str(snapshot_archive),
        "source_snapshot_sha256": archive_sha256,
        "source_snapshot_root": str(snapshot_root),
        "source_snapshot_digest_path": snapshot_digest_path_text,
        "runner_root": str(runner_root),
    }


def _event_candidate_case(
        *, source: Mapping[str, Any], source_id: str,
        manifest_path: Path, hardware_id: str, family: str, case_id: str,
        entry: Mapping[str, Any], policy: Mapping[str, Any]
        ) -> Mapping[str, Any]:
    provenance = _source_provenance(
        source=source, manifest_path=manifest_path, source_id=source_id)
    case_directory = _require_existing_directory(
        _resolve_selection_path(
            source["case_directory"], manifest_path,
            f"Event case directory for {source_id}"),
        f"Event case directory for {source_id}")
    case_path = _require_existing_file(
        case_directory / f"{case_id}.json", f"Event case record for {case_id}")
    timer_path = _require_existing_file(
        _resolve_selection_path(
            source["timer_sanity_path"], manifest_path,
            f"timer sanity for {source_id}"),
        f"timer sanity for {source_id}")
    timer_payload = _load_json_mapping(timer_path, "timer sanity artifact")
    timer_sanity, timer_sanity_sha256 = _validate_timer_sanity(
        path=timer_path, hardware_id=hardware_id,
        baseline_commit=timer_payload["baseline_commit"])
    if timer_sanity["source_snapshot_sha256"] != (
            provenance["source_snapshot_sha256"]):
        raise ValueError(f"timer sanity source snapshot mismatch: {case_id}")
    record = _load_json_mapping(case_path, "Event case record")
    durations = _validate_case_record(
        record=record, path=case_path, hardware_id=hardware_id,
        family=family, case_id=case_id,
        baseline_commit=timer_sanity["baseline_commit"],
        source_snapshot_sha256=provenance["source_snapshot_sha256"],
        timer_sanity_sha256=timer_sanity_sha256,
        canonical_attributes=entry["op_info"]["Attr"],
        runner_root=Path(provenance["runner_root"]))
    if durations is None:
        return {
            "status": "unsupported",
            "unsupported_reason": record["unsupported_reason"],
            "observed_op_gpu_time_ns": None,
            "candidate_op_gpu_time_ns": None,
            "stability": None,
            "timing": {
                "source_kind": "event_cases",
                "scope": "logical_public_op",
                "measurement_mode": None,
                "timer": None,
            },
            "provenance": {
                **provenance,
                "capture_baseline_commit": timer_sanity["baseline_commit"],
                "runner_path": record["runner_path"],
                "runner_sha256": record["runner_sha256"],
                "raw_case_path": str(case_path),
                "raw_case_sha256": _sha256(case_path),
                "timer_sanity_path": str(timer_path),
                "timer_sanity_sha256": timer_sanity_sha256,
            },
        }
    statistics_payload = record["measurement"]["statistics"]
    median_ns = statistics_payload["median_ns"]
    coefficient_of_variation = statistics_payload["coefficient_of_variation"]
    if (not isinstance(median_ns, (int, float)) or isinstance(median_ns, bool)
            or not math.isfinite(median_ns) or median_ns <= 0):
        raise ValueError(f"Event median must be finite and positive: {case_id}")
    if (not isinstance(coefficient_of_variation, (int, float))
            or isinstance(coefficient_of_variation, bool)
            or not math.isfinite(coefficient_of_variation)
            or coefficient_of_variation < 0):
        raise ValueError(f"Event CV must be finite and non-negative: {case_id}")
    maximum_relative_deviation = max(
        abs(duration - median_ns) / median_ns for duration in durations)
    stable = (
        coefficient_of_variation
        <= policy["event_coefficient_of_variation_maximum"]
        and maximum_relative_deviation
        <= policy["event_maximum_relative_deviation_maximum"])
    return {
        "status": "accepted" if stable else "unstable",
        "unsupported_reason": None,
        "observed_op_gpu_time_ns": median_ns,
        "candidate_op_gpu_time_ns": median_ns if stable else None,
        "stability": {
            "event_coefficient_of_variation": coefficient_of_variation,
            "event_coefficient_of_variation_maximum": policy[
                "event_coefficient_of_variation_maximum"],
            "maximum_relative_deviation": maximum_relative_deviation,
            "maximum_relative_deviation_maximum": policy[
                "event_maximum_relative_deviation_maximum"],
        },
        "timing": {
            "source_kind": "event_cases",
            "scope": "logical_public_op",
            "measurement_mode": record["measurement"]["schema_version"],
            "timer": record["timer"],
        },
        "provenance": {
            **provenance,
            "capture_baseline_commit": timer_sanity["baseline_commit"],
            "runner_path": record["runner_path"],
            "runner_sha256": record["runner_sha256"],
            "raw_case_path": str(case_path),
            "raw_case_sha256": _sha256(case_path),
            "timer_sanity_path": str(timer_path),
            "timer_sanity_sha256": timer_sanity_sha256,
        },
    }


def _timeline_candidate_case(
        *, source: Mapping[str, Any], source_id: str,
        manifest_path: Path, hardware_id: str, family: str, case_id: str,
        entry: Mapping[str, Any], policy: Mapping[str, Any]
        ) -> Mapping[str, Any]:
    if hardware_id != "ascend_910b":
        raise ValueError("Timeline candidates are only supported for Ascend 910B")
    provenance = _source_provenance(
        source=source, manifest_path=manifest_path, source_id=source_id)
    candidate_path = _require_existing_file(
        _resolve_selection_path(
            source["candidate_path"], manifest_path,
            f"Timeline candidate for {source_id}"),
        f"Timeline candidate for {source_id}")
    candidate = _load_json_mapping(candidate_path, "Timeline candidate")
    if candidate["schema_version"] != 3:
        raise ValueError(f"unsupported Timeline candidate schema: {candidate_path}")
    if candidate["hardware_id"] != hardware_id:
        raise ValueError(f"Timeline candidate hardware mismatch: {case_id}")
    if candidate["status"] != "timing_only_candidate":
        raise ValueError(f"Timeline candidate status mismatch: {case_id}")
    if candidate["profile_mode"] != "timeline":
        raise ValueError(f"Timeline candidate mode mismatch: {case_id}")
    if candidate["timing_source"] != "msprof_timeline_task_start_and_duration":
        raise ValueError(f"Timeline candidate timing source mismatch: {case_id}")
    if candidate["aggregation"] != "median_of_per_call_op_elapsed_duration":
        raise ValueError(f"Timeline candidate aggregation mismatch: {case_id}")
    if candidate["cache_condition"] != (
            "2_gib_bf16_fill_before_every_public_call"):
        raise ValueError(f"Timeline candidate cache condition mismatch: {case_id}")
    source_snapshot = candidate["source_snapshot"]
    if source_snapshot["archive_sha256"] != provenance["source_snapshot_sha256"]:
        raise ValueError(f"Timeline candidate archive mismatch: {case_id}")
    family_payload = candidate["families"][family]
    current_dataset_path = Path(get_kernel_dataset_path(hardware_id, family))
    if family_payload["canonical_dataset"]["sha256"] != kernel_dataset_content_sha256(
            current_dataset_path):
        raise ValueError(f"Timeline candidate dataset mismatch: {case_id}")
    if family_payload["canonical_dataset"]["case_count"] != len(
            _load_family_dataset(hardware_id, family)):
        raise ValueError(f"Timeline candidate dataset count mismatch: {case_id}")
    runner_source = family_payload["runner_source"]
    runner_path = Path(runner_source["path"])
    if runner_path.is_absolute() or ".." in runner_path.parts:
        raise ValueError(f"invalid Timeline runner path: {case_id}")
    runner = _require_existing_file(
        Path(provenance["runner_root"]) / runner_path,
        f"Timeline runner for {case_id}")
    if _sha256(runner) != runner_source["sha256"]:
        raise ValueError(f"Timeline runner hash mismatch: {case_id}")
    update = family_payload["case_updates"][case_id]
    if update["status"] != "timing_candidate":
        raise ValueError(f"Timeline case is not a timing candidate: {case_id}")
    if update["canonical_entry_sha256"] != _canonical_mapping_sha256(entry):
        raise ValueError(f"Timeline canonical entry mismatch: {case_id}")
    if update["canonical_attributes_sha256"] != _canonical_mapping_sha256(
            entry["op_info"]["Attr"]):
        raise ValueError(f"Timeline canonical attributes mismatch: {case_id}")
    if update["runner_sha256"] != runner_source["sha256"]:
        raise ValueError(f"Timeline case runner mismatch: {case_id}")
    replacement_timing = update["replacement_timing"]
    if replacement_timing["granularity"] != "public_call":
        raise ValueError(f"invalid Timeline timing semantics: {case_id}")
    timing_source = replacement_timing["source"]
    timing_aggregation = replacement_timing["aggregation"]
    if timing_source == "msprof_timeline_task_duration":
        expected_per_call_aggregation = "single_task_direct_duration"
        expected_timing_aggregation = (
            "median_of_per_call_single_task_duration")
    elif timing_source == "msprof_timeline_task_schedule":
        expected_per_call_aggregation = "same_stream_task_schedule_envelope"
        expected_timing_aggregation = (
            "median_of_per_call_same_stream_task_schedule_envelope")
    else:
        raise ValueError(f"invalid Timeline timing source: {case_id}")
    if timing_aggregation != expected_timing_aggregation:
        raise ValueError(f"invalid Timeline timing aggregation: {case_id}")
    measurement = update["measurement"]
    if measurement["per_call_timing_aggregation"] != (
            expected_per_call_aggregation):
        raise ValueError(f"Timeline per-call aggregation mismatch: {case_id}")
    if measurement["cross_call_timing_aggregation"] != "median":
        raise ValueError(f"Timeline cross-call aggregation mismatch: {case_id}")
    calls = measurement["calls"]
    if not isinstance(calls, list) or not calls:
        raise TypeError(f"Timeline calls must be a non-empty list: {case_id}")
    durations = []
    for call in calls:
        duration = call["op_elapsed_duration_ns"]
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError(f"invalid Timeline call duration: {case_id}")
        durations.append(duration)
    if measurement["sample_count"] != len(durations):
        raise ValueError(f"Timeline sample count mismatch: {case_id}")
    median_ns = statistics.median(durations)
    if measurement["median_op_elapsed_duration_ns"] != median_ns:
        raise ValueError(f"Timeline median mismatch: {case_id}")
    if replacement_timing["total_duration_ns"] != median_ns:
        raise ValueError(f"Timeline replacement timing mismatch: {case_id}")
    median_absolute_deviation = statistics.median(
        abs(duration - median_ns) for duration in durations)
    if measurement["median_absolute_deviation_ns"] != median_absolute_deviation:
        raise ValueError(f"Timeline MAD mismatch: {case_id}")
    relative_mad = median_absolute_deviation / median_ns
    maximum_relative_deviation = max(
        abs(duration - median_ns) / median_ns for duration in durations)
    stable = (
        relative_mad <= policy["timeline_relative_mad_maximum"]
        and maximum_relative_deviation
        <= policy["timeline_maximum_relative_deviation_maximum"])
    timeline_summary = update["timeline_summary"]
    summary_path = Path(timeline_summary["path"])
    if summary_path.is_absolute() or ".." in summary_path.parts:
        raise ValueError(f"invalid Timeline summary path: {case_id}")
    resolved_summary_path = _require_existing_file(
        Path(provenance["runner_root"]) / summary_path,
        f"Timeline summary for {case_id}")
    if _sha256(resolved_summary_path) != timeline_summary["sha256"]:
        raise ValueError(f"Timeline summary hash mismatch: {case_id}")
    return {
        "status": "accepted" if stable else "unstable",
        "unsupported_reason": None,
        "observed_op_gpu_time_ns": median_ns,
        "candidate_op_gpu_time_ns": median_ns if stable else None,
        "stability": {
            "relative_mad": relative_mad,
            "relative_mad_maximum": policy["timeline_relative_mad_maximum"],
            "maximum_relative_deviation": maximum_relative_deviation,
            "maximum_relative_deviation_maximum": policy[
                "timeline_maximum_relative_deviation_maximum"],
        },
        "timing": {
            "source_kind": "ascend_timeline_candidate",
            "scope": "logical_public_op",
            "measurement_mode": candidate["profile_mode"],
            "timer": None,
        },
        "provenance": {
            **provenance,
            "capture_baseline_commit": candidate["capture_baseline_commit"],
            "runner_path": runner_source["path"],
            "runner_sha256": runner_source["sha256"],
            "raw_case_path": str(resolved_summary_path),
            "raw_case_sha256": timeline_summary["sha256"],
            "timer_sanity_path": None,
            "timer_sanity_sha256": None,
            "timeline_candidate_path": str(candidate_path),
            "timeline_candidate_sha256": _sha256(candidate_path),
        },
    }


def _candidate_group_summary(cases: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    counts = {"accepted": 0, "unstable": 0, "unsupported": 0}
    errors = []
    for case in cases:
        status = case["status"]
        if status not in counts:
            raise ValueError(f"unknown OP candidate status: {status!r}")
        counts[status] += 1
        forward = case["forward"]
        if status == "accepted" and forward["status"] == "supported":
            errors.append(forward["absolute_percentage_error"])
    return {
        "case_count": len(cases),
        **counts,
        "forward_supported_accepted_case_count": len(errors),
        "accepted_only_op_mape_percent": (
            statistics.fmean(errors) if errors else None),
    }


def build_op_candidate_report(
        *, selection_manifest_path: Path, allow_partial: bool = False
        ) -> Mapping[str, Any]:
    """Build an archive-bound, explicit-source logical-OP timing candidate."""

    if not isinstance(allow_partial, bool):
        raise TypeError("allow_partial must be boolean")
    selection_manifest, manifest_path = _load_op_candidate_selection(
        selection_manifest_path)
    hardware_id = selection_manifest["hardware_id"]
    families = selection_manifest["families"]
    sources = selection_manifest["sources"]
    policy = selection_manifest["stability_policy"]
    datasets = {
        family: _load_family_dataset(hardware_id, family)
        for family in families
    }
    selected_by_family = {family: set() for family in families}
    cases = []
    for selection in selection_manifest["selections"]:
        family = selection["family"]
        case_id = selection["case_id"]
        source_id = selection["source_id"]
        dataset = datasets[family]
        if case_id not in dataset:
            raise KeyError(
                f"OP candidate case is absent from current dataset: "
                f"{hardware_id}/{family}/{case_id}")
        selected_by_family[family].add(case_id)
        source = sources[source_id]
        if source["kind"] == "event_cases":
            candidate = dict(_event_candidate_case(
                source=source, source_id=source_id, manifest_path=manifest_path,
                hardware_id=hardware_id, family=family, case_id=case_id,
                entry=dataset[case_id], policy=policy))
        elif source["kind"] == "ascend_timeline_candidate":
            candidate = dict(_timeline_candidate_case(
                source=source, source_id=source_id, manifest_path=manifest_path,
                hardware_id=hardware_id, family=family, case_id=case_id,
                entry=dataset[case_id], policy=policy))
        else:
            raise AssertionError(f"validated source kind was not dispatched: {source_id}")
        candidate["hardware_id"] = hardware_id
        candidate["family"] = family
        candidate["case_id"] = case_id
        candidate["canonical_attributes_sha256"] = _canonical_mapping_sha256(
            dataset[case_id]["op_info"]["Attr"])
        if candidate["status"] == "accepted":
            from evaluation.kernel.evaluation_runner import predict_forward_op_total
            forward = predict_forward_op_total(
                task_key=case_id, entry=dataset[case_id], hardware_id=hardware_id)
            if forward["status"] == "supported":
                observed_ns = candidate["candidate_op_gpu_time_ns"]
                prediction_ns = forward["prediction_ns"]
                forward["absolute_percentage_error"] = (
                    abs(prediction_ns - observed_ns) / observed_ns * 100)
            elif forward["status"] == "unsupported":
                pass
            else:
                raise ValueError(
                    f"unknown Forward candidate status: {forward['status']!r}")
        elif candidate["status"] in {"unstable", "unsupported"}:
            forward = {
                "status": "not_evaluated",
                "reason": "timing_candidate_not_accepted",
            }
        else:
            raise ValueError(
                f"unknown OP timing candidate status: "
                f"{candidate['status']!r}")
        candidate["forward"] = forward
        cases.append(candidate)
    for family, dataset in datasets.items():
        if not allow_partial and selected_by_family[family] != set(dataset):
            raise ValueError(
                f"strict OP candidate coverage mismatch for {family}: "
                f"missing={sorted(set(dataset) - selected_by_family[family])}")
    cases.sort(key=lambda row: (row["family"], row["case_id"]))
    if not allow_partial:
        nonaccepted = [
            f"{row['family']}/{row['case_id']}:{row['status']}"
            for row in cases if row["status"] != "accepted"]
        if nonaccepted:
            raise ValueError(
                "strict OP candidate contains non-accepted cases: "
                f"{nonaccepted}")
    family_summaries = []
    mode_groups: dict[tuple[str, str, Any], list[Mapping[str, Any]]] = {}
    for family in families:
        family_cases = [row for row in cases if row["family"] == family]
        family_summaries.append({
            "family": family,
            "canonical_case_count": len(datasets[family]),
            "selected_case_count": len(family_cases),
            "missing_canonical_case_count": len(datasets[family]) - len(family_cases),
            "canonical_dataset_sha256": kernel_dataset_content_sha256(
                Path(get_kernel_dataset_path(hardware_id, family))),
            **_candidate_group_summary(family_cases),
        })
        for row in family_cases:
            timing = row["timing"]
            key = (
                family, timing["source_kind"],
                timing["measurement_mode"])
            if key not in mode_groups:
                mode_groups[key] = []
            mode_groups[key].append(row)
    mode_summaries = []
    for key, group in sorted(mode_groups.items()):
        family, source_kind, measurement_mode = key
        mode_summaries.append({
            "family": family,
            "source_kind": source_kind,
            "measurement_mode": measurement_mode,
            **_candidate_group_summary(group),
        })
    source_summaries = {}
    for source_id, source in sources.items():
        source_summaries[source_id] = _source_provenance(
            source=source, manifest_path=manifest_path, source_id=source_id)
    return {
        "schema_version": 1,
        "artifact_kind": OP_CANDIDATE_REPORT_KIND,
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": _sha256(manifest_path),
        "allow_partial": allow_partial,
        "hardware_id": hardware_id,
        "families": families,
        "stability_policy": policy,
        "sources": source_summaries,
        "case_count": len(cases),
        "cases": cases,
        "summaries": {
            "overall": _candidate_group_summary(cases),
            "family": family_summaries,
            "family_timing_mode": mode_summaries,
        },
    }


def build_comparison_report(
        *, manifest_path: Path, run_directories: Mapping[str, Path]
        ) -> Mapping[str, Any]:
    """Validate complete fixed samples and build per-case/group comparisons."""

    if set(run_directories) != set(HARDWARE_IDS):
        raise ValueError(
            "comparison report requires exactly one run directory for each "
            f"hardware: expected={sorted(HARDWARE_IDS)}, "
            f"actual={sorted(run_directories)}")
    manifest = load_sample_manifest(manifest_path)
    baseline_commit = manifest["baseline_commit"]
    comparisons = []
    runs = {}
    for hardware_id in HARDWARE_IDS:
        run_directory = Path(run_directories[hardware_id]).resolve()
        timer_sanity, timer_sanity_sha256 = _validate_timer_sanity(
            path=run_directory / "timer_sanity.json",
            hardware_id=hardware_id,
            baseline_commit=baseline_commit,
        )
        source_snapshot_sha256 = timer_sanity["source_snapshot_sha256"]
        selected = _selected_case_index(manifest, hardware_id)
        case_directory = run_directory / "cases"
        case_paths = sorted(
            path for path in case_directory.glob("*.json")
            if not path.name.endswith(".workload.json")
        )
        actual_case_ids = {path.stem for path in case_paths}
        if actual_case_ids != set(selected):
            raise ValueError(
                f"Event case set mismatch for {hardware_id}: "
                f"missing={sorted(set(selected) - actual_case_ids)}, "
                f"extra={sorted(actual_case_ids - set(selected))}")
        status_counts = {"measured": 0, "unsupported": 0}
        for path in case_paths:
            case_id = path.stem
            family = selected[case_id]
            dataset = _load_family_dataset(hardware_id, family)
            record = _load_json_mapping(path, "Event case record")
            durations = _validate_case_record(
                record=record,
                path=path,
                hardware_id=hardware_id,
                family=family,
                case_id=case_id,
                baseline_commit=baseline_commit,
                source_snapshot_sha256=source_snapshot_sha256,
                timer_sanity_sha256=timer_sanity_sha256,
                canonical_attributes=dataset[case_id]["op_info"]["Attr"],
            )
            status = record["status"]
            status_counts[status] += 1
            if durations is None:
                comparison = {
                    "hardware_id": hardware_id,
                    "family": family,
                    "case_id": case_id,
                    "bucket": primary_bucket(family, dataset[case_id]),
                    "status": "unsupported",
                    "unsupported_reason": record["unsupported_reason"],
                    "dataset_timing_source": dataset[case_id]["op_info"][
                        "evaluation"]["timing"]["source"],
                    "dataset_total_duration_ns": dataset[case_id]["op_info"][
                        "evaluation"]["timing"]["total_duration_ns"],
                    "event_statistics": None,
                    "absolute_error_ns": None,
                    "relative_error": None,
                    "signed_relative_error": None,
                }
            else:
                comparison = dict(compare_case_measurement(
                    hardware_id=hardware_id,
                    family=family,
                    case_id=case_id,
                    event_samples_ns=durations,
                ))
                comparison["status"] = "measured"
                comparison["unsupported_reason"] = None
            comparisons.append(comparison)
        runs[hardware_id] = {
            "run_directory": str(run_directory),
            "source_snapshot_sha256": source_snapshot_sha256,
            "timer_sanity_sha256": timer_sanity_sha256,
            "selected_case_count": len(selected),
            "measured_case_count": status_counts["measured"],
            "unsupported_case_count": status_counts["unsupported"],
        }
    comparisons.sort(
        key=lambda row: (row["hardware_id"], row["family"], row["bucket"],
                         row["case_id"]))
    return {
        "schema_version": 1,
        "artifact_kind": "kernel_golden_device_event_comparison",
        "baseline_commit": baseline_commit,
        "sample_manifest": str(manifest_path.resolve()),
        "sample_manifest_sha256": _sha256(manifest_path),
        "runs": runs,
        "case_count": len(comparisons),
        "cases": comparisons,
        "summaries": {
            "hardware": _summaries_by_key(comparisons, ("hardware_id",)),
            "hardware_family": _summaries_by_key(
                comparisons, ("hardware_id", "family")),
            "hardware_family_bucket": _summaries_by_key(
                comparisons, ("hardware_id", "family", "bucket")),
        },
    }


def _parse_run_directories(values: list[str]) -> Mapping[str, Path]:
    result = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(
                f"run directory must use HARDWARE_ID=PATH syntax: {value!r}")
        hardware_id, raw_path = value.split("=", 1)
        if hardware_id not in HARDWARE_IDS:
            raise ValueError(f"unknown run directory hardware: {hardware_id!r}")
        if hardware_id in result:
            raise ValueError(f"duplicate run directory for {hardware_id}")
        if not raw_path:
            raise ValueError(f"run directory path is empty for {hardware_id}")
        result[hardware_id] = Path(raw_path)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and report kernel-golden Event validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample_parser = subparsers.add_parser(
        "build-sample-manifest",
        help="select fixed minimum/middle/maximum cases per primary bucket",
    )
    sample_parser.add_argument("--baseline-commit", required=True)
    sample_parser.add_argument("--output", required=True, type=Path)
    report_parser = subparsers.add_parser(
        "build-comparison-report",
        help="validate fixed Event artifacts and summarize OP timing errors",
    )
    report_parser.add_argument("--sample-manifest", required=True, type=Path)
    report_parser.add_argument(
        "--run-dir", required=True, action="append",
        help="repeat HARDWARE_ID=PATH once for each target hardware",
    )
    report_parser.add_argument("--output", required=True, type=Path)
    candidate_parser = subparsers.add_parser(
        "build-op-candidate-report",
        help=("validate explicit archive-bound OP timing sources and report "
              "accepted-only Forward MAPE"),
    )
    candidate_parser.add_argument(
        "--selection-manifest", required=True, type=Path,
        help="explicit per-case source selection manifest",
    )
    candidate_parser.add_argument(
        "--allow-partial", action="store_true",
        help="allow incomplete canonical-family coverage or unstable cases",
    )
    candidate_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "build-sample-manifest":
        _write_json(
            args.output,
            build_sample_manifest(args.baseline_commit),
        )
        return 0
    elif args.command == "build-comparison-report":
        _write_json(
            args.output,
            build_comparison_report(
                manifest_path=args.sample_manifest,
                run_directories=_parse_run_directories(args.run_dir),
            ),
        )
        return 0
    elif args.command == "build-op-candidate-report":
        _write_json(
            args.output,
            build_op_candidate_report(
                selection_manifest_path=args.selection_manifest,
                allow_partial=args.allow_partial,
            ),
        )
        return 0
    else:
        raise AssertionError(
            f"argparse accepted an unknown command: {args.command!r}")
