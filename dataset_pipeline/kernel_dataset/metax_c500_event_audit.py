"""Fail-closed audit of complete MetaX C500 OP Device Event evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tarfile
from typing import Any, Mapping, Sequence

from dataset_pipeline.kernel_dataset.metax_c500_event_capture import (
    build_vector_buffer_contract,
    build_vector_distinct_input_contract,
)

from dataset_pipeline.kernel_dataset.metax_c500_event_protocol import (
    load_json_mapping,
    file_sha256,
    require_positive_integer,
    EXPECTED_CANONICAL_SHA256,
    FAMILY_COUNTS,
    GROUPS,
    GROUP_COUNTS,
    GROUP_FAMILIES,
    HISTORICAL_VECTOR_SELECTION_PROTOCOL,
    VECTOR_RECAPTURE_PROTOCOL,
    summarize_event_samples,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "experiment_logs" / "kernel_golden"
FAMILIES = ("vector", "flash_attn", "gemm", "fused_moe")
EVENT_CASE_REQUIRED_KEYS = frozenset({
    "schema_version", "status", "unsupported_reason", "hardware_id",
    "family", "case_id", "baseline_commit", "source_snapshot_sha256",
    "runner_path", "runner_sha256", "timer_sanity_sha256",
    "canonical_attributes_sha256", "timer", "stream_contract",
    "allocation_contract", "workload", "environment", "measurement",
})
GRAPH_REPETITIONS = {
    "vector": [50, 200],
    "flash_attn": [50, 200],
    "gemm": [50, 200],
    "fused_moe": [5, 20],
}
SELECTED_FAMILY_COUNTS = {
    "vector": 149, "flash_attn": 237, "gemm": 13, "fused_moe": 94,
}
EXPECTED_COMPLETION_SHA256 = (
    "c89df8137a2dec4917eab329f00e3c21073f35fa7c3916a0a9268d776699077e"
)
EXPECTED_FLASH_LEGACY_AUDIT_SHA256 = (
    "568bd17490324301cd7972f001559dfdf6d74cfb5f7ea8c5f394fd82ba096940"
)
EXPECTED_REMAINING_LEGACY_AUDIT_SHA256 = (
    "6c50baf0e7d8c5c0665947136de1f15204c12aec8944cc64b701b6b762eb637c"
)
EXPECTED_VECTOR_LEGACY_TREE_SHA256 = (
    "f5dd67cfee1501333d91cf5df15ffc7329e3ad8dcf23de7fe276502a6fa484a1"
)
EXPECTED_NEW_BASELINE = "41d29da9cd124dfaac51a72003860d97a7de6d91"
EXPECTED_NEW_ARCHIVE_SHA256 = (
    "a6a4429980c65513d5faa87a22d049478f9b5d0e95d744285cc05707deddd279"
)
EXPECTED_NEW_DESCRIPTOR_SHA256 = (
    "8b5cf5e53fe17d69c6efdfe689120177accb261d10f805e73f62ba28f1df50b3"
)
EXPECTED_NEW_TIMER_SHA256 = (
    "092df0e687688e304a30b9e2e06103ca2f6cd5a6cb0feec80b87ce61658d37d3"
)
EXPECTED_NEW_ENVIRONMENTS = {
    "gemm": {
        "torch_version": "2.4.0+metax3.7.2.0",
        "torch_cuda_version_field": "11.6",
        "device_index": 0,
        "device_name": "MetaX C500",
        "device_total_memory_bytes": 68283269120,
        "input_seed": 20260826,
    },
    "flash_attn": {
        "torch_version": "2.4.0+metax3.7.2.0",
        "torch_cuda_version_field": "11.6",
        "flash_attn_version": "2.6.3+metax3.7.2.0torch2.4",
        "device_index": 0,
        "device_name": "MetaX C500",
        "device_total_memory_bytes": 68283269120,
        "input_seed": 20260826,
    },
    "fused_moe": {
        "torch_version": "2.4.0+metax3.7.2.0",
        "torch_cuda_version_field": "11.6",
        "triton_version": "3.0.0",
        "device_index": 0,
        "device_name": "MetaX C500",
        "device_total_memory_bytes": 68283269120,
        "input_seed": 42,
    },
    "vector": {
        "torch_version": "2.4.0+metax3.7.2.0",
        "torch_cuda_version_field": "11.6",
        "device_index": 0,
        "device_name": "MetaX C500",
    },
}
RUNNER_PATHS = {
    family: f"scripts/kernel_dataset/metax_c500/{family}_runner.py"
    for family in FAMILIES
}
CV_MAXIMUM = 0.05
MAXIMUM_RELATIVE_DEVIATION = 0.05
CONVERGENCE_ABSOLUTE_NS = 500.0
CONVERGENCE_RELATIVE = 0.05
SELECTION_RULE = "latest_started_attempt_must_pass_v1"
FORBIDDEN_PROFILER_MARKERS = (
    "mctracer", "mcprofiler", "perf_exec", "tracer_out", "event_batch",
)

SELECTION_KEYS = frozenset({
    "schema_version", "artifact_kind", "hardware_id", "selected_case_count",
    "existing_strict_case_count", "target_case_count", "stability_policy",
    "vector_protocols", "vector_stability_audit", "groups", "family_case_ids",
    "existing_strict_family_case_ids", "provenance",
})
CAMPAIGN_KEYS = frozenset({
    "schema_version", "artifact_kind", "hardware_id", "campaign_root",
    "selection_manifest", "runtime_contract", "sources", "attempts",
    "selection_rule",
})
SOURCE_KEYS = frozenset({
    "role", "archive_status", "baseline_commit", "snapshot_root", "source_archive",
    "source_archive_sha256", "source_descriptor", "source_descriptor_sha256",
    "source_files_manifest", "source_files_manifest_sha256", "timer_sanity",
    "timer_sanity_sha256",
})
ATTEMPT_KEYS = frozenset({
    "attempt_id", "ordinal", "cohort", "source_id", "family",
    "physical_device", "root", "command_argv", "pid", "process_group_id",
    "start_wall_time_ns", "end_wall_time_ns", "planned_case_ids",
    "case_attempts", "terminal_status", "error", "superseded",
    "preflight_snapshot_id", "during_snapshot_ids", "postflight_snapshot_id",
    "process_isolation", "root_inventory", "wrapper", "execution_module",
    "launcher_pid",
    "launcher_process_group_id", "launcher_command_argv", "host_uid",
    "device_gid",
})
CASE_ATTEMPT_KEYS = frozenset({
    "case_id", "status", "artifact_path", "artifact_sha256", "error_path",
    "error_sha256", "superseded_by_attempt_id",
})
PROCESS_MANIFEST_KEYS = frozenset({
    "schema_version", "artifact_kind", "hardware_id", "host",
    "collector_path", "collector_sha256", "snapshots",
})
PROCESS_SNAPSHOT_KEYS = frozenset({
    "schema_version", "artifact_kind", "scope", "collection_complete",
    "host", "timestamp_wall_time_ns", "processes", "device_processes",
})
PROCESS_KEYS = frozenset({
    "pid", "ppid", "process_group_id", "argv", "physical_devices",
})
DEVICE_PROCESS_KEYS = frozenset({"physical_device", "pid"})
TABLE_KEYS = frozenset({
    "schema_version", "hardware_id", "family", "method", "baseline_commit",
    "source_snapshot_sha256", "timer_sanity_sha256",
    "canonical_dataset_sha256", "expected_case_count",
    "unsharded_case_count", "shard_count", "shard_index",
    "sample_count_per_repetition", "warmup_iterations", "convergence_rule",
    "differential_estimator", "buffer_policy", "buffer_count",
    "cache_eviction_mib", "minimum_distinct_input_mib", "rows",
})
TABLE_ROW_KEYS = frozenset({
    "family", "case_id", "attributes_json", "executed_workload_json", "op",
    "op2", "tok", "hidden", "intermediate", "num_heads", "num_kv_heads",
    "head_dim", "batch", "B", "M", "K", "N", "S", "Q_LEN", "HQ",
    "HKV", "D", "SP_REQUESTED", "kv_layout", "causal", "TOK", "E",
    "TOPK", "H", "INT", "canonical_duration_us", "lower_repetitions",
    "higher_repetitions", "lower_median_us", "lower_p10_us", "lower_p90_us",
    "lower_cv", "higher_median_us", "higher_p10_us", "higher_p90_us",
    "higher_cv", "selected_duration_us", "absolute_error_us", "relative_error",
    "signed_relative_error", "convergence_status", "status", "error",
})
SNAPSHOT_DESCRIPTOR_KEYS = frozenset({
    "schema_version", "source_commit", "source_archive_sha256",
    "source_files_manifest", "source_files_manifest_sha256",
    "source_file_count", "excluded_top_level_payload_roots",
})
TIMER_SANITY_KEYS = frozenset({
    "schema_version", "artifact_kind", "hardware_id", "baseline_commit",
    "source_snapshot_sha256", "runner_sha256", "profiler_started", "device",
    "software", "timer", "backend_evidence", "sanity_workload", "measurement",
    "checks",
})




def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return _bytes_sha256(encoded)




def _exact_keys(value: Mapping[str, Any], expected: frozenset[str],
                label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys mismatch: {sorted(set(value) ^ expected)}")


def _path_record(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = Path(value["path"]).resolve()
    expected = value["sha256"]
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} SHA-256 is invalid")
    if file_sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, expected


def _validate_legacy_root_inventory(
        value: Any, root: Path) -> tuple[Path, set[Path]]:
    manifest, _ = _path_record(value, "legacy root inventory")
    if manifest.is_relative_to(root):
        raise ValueError("legacy root inventory must be external to frozen root")
    try:
        lines = manifest.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("legacy root inventory must be ASCII") from error
    entries = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("legacy root inventory line is malformed")
        digest = line[:64]
        relative = Path(line[66:])
        if (any(character not in "0123456789abcdef" for character in digest)
                or not relative.as_posix() or relative.is_absolute()
                or ".." in relative.parts):
            raise ValueError("legacy root inventory entry is invalid")
        relative_text = relative.as_posix()
        if relative_text in entries:
            raise ValueError("legacy root inventory repeats a path")
        entries[relative_text] = digest
    if list(entries) != sorted(entries):
        raise ValueError("legacy root inventory is not ASCII sorted")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("legacy root must not contain symlinks")
    actual = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in root.rglob("*") if path.is_file()}
    if entries != actual:
        raise ValueError("legacy root inventory does not match frozen root")
    return manifest, {manifest, *(
        root / relative for relative in entries)}




def _sha256sum_manifest(paths: Sequence[Path], root: Path) -> str:
    lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}\n".encode("ascii")
        for path in paths
    ]
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line)
    return digest.hexdigest()


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
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0.0):
            raise ValueError(
                f"invalid {context} duration for {case_id} sample {index}")
        durations.append(float(duration))
    return durations


def _validate_event_statistics(
        durations: list[float], recorded: Any,
        case_id: str, context: str) -> None:
    expected = summarize_event_samples(durations)
    if not isinstance(recorded, dict) or set(recorded) != set(expected):
        raise ValueError(
            f"{context} statistics disagree with raw samples: {case_id}")
    for name, expected_value in expected.items():
        value = recorded[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(
                    float(value), expected_value, rel_tol=1e-12,
                    abs_tol=1e-12)):
            raise ValueError(
                f"{context} statistic {name} disagrees with raw samples: "
                f"{case_id}")


def _validate_event_measurement(
        measurement: Mapping[str, Any], case_id: str) -> list[float]:
    """Validate the schema-3 Event forms accepted by the C500 campaign."""
    if not isinstance(measurement, dict):
        raise TypeError(f"measured case {case_id} measurement must be a mapping")
    if measurement["schema_version"] != 3:
        raise ValueError(f"unsupported C500 Event sample schema for {case_id}")
    sample_count = require_positive_integer(
        measurement["sample_count"], f"sample_count for {case_id}")
    require_positive_integer(
        measurement["warmup_iterations"],
        f"warmup_iterations for {case_id}")
    if "mode" not in measurement:
        if (measurement["event_prelude_iterations"] != 1
                or measurement["cache_eviction_requested_bytes"] !=
                measurement["cache_eviction_bytes_before_sample"]):
            raise ValueError(f"invalid direct Event contract for {case_id}")
        durations = _validate_event_samples(
            measurement["samples"], sample_count, case_id, "direct Event")
        _validate_event_statistics(
            durations, measurement["statistics"], case_id, "direct Event")
        return durations

    mode = measurement["mode"]
    if mode not in {
            "external_events_around_cuda_graph_replay",
            "paired_differential_events_around_cuda_graph_replay"}:
        raise ValueError(f"unsupported C500 graph Event mode for {case_id}")
    if (measurement["event_prelude_iterations_per_repetition"] != 1
            or measurement["cache_eviction_requested_bytes"] !=
            measurement["cache_eviction_bytes_before_replay"]):
        raise ValueError(f"invalid graph Event contract for {case_id}")
    repetition_counts = measurement["repetition_counts"]
    if (not isinstance(repetition_counts, list)
            or len(repetition_counts) != 2
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in repetition_counts)
            or repetition_counts[0] >= repetition_counts[1]):
        raise ValueError(f"invalid graph repetition_counts for {case_id}")
    require_positive_integer(
        measurement["operation_cycle_length"],
        f"operation_cycle_length for {case_id}")
    repetition_runs = measurement["repetition_runs"]
    if not isinstance(repetition_runs, list) or len(repetition_runs) != 2:
        raise ValueError(f"invalid graph repetition_runs for {case_id}")
    run_durations = []
    for run_index, (run, repetition_count) in enumerate(zip(
            repetition_runs, repetition_counts, strict=True)):
        if (not isinstance(run, dict)
                or run["repetition_count"] != repetition_count
                or run["sample_count"] != sample_count):
            raise ValueError(f"graph run contract mismatch for {case_id}")
        durations = _validate_event_samples(
            run["samples"], sample_count, case_id,
            f"graph run {run_index}")
        _validate_event_statistics(
            durations, run["statistics"], case_id,
            f"graph run {run_index}")
        for sample in run["samples"]:
            replay_duration = sample["replay_duration_ns"]
            if (sample["repetition_count"] != repetition_count
                    or isinstance(replay_duration, bool)
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
    if mode == "external_events_around_cuda_graph_replay":
        if (measurement["samples"] != repetition_runs[1]["samples"]
                or measurement["statistics"] !=
                repetition_runs[1]["statistics"]):
            raise ValueError(
                f"graph selected samples are not the higher run for {case_id}")
        lower_median = repetition_runs[0]["statistics"]["median_ns"]
        higher_median = repetition_runs[1]["statistics"]["median_ns"]
        difference = abs(higher_median - lower_median)
        tolerance = max(500.0, 0.05 * max(lower_median, higher_median))
        expected_status = {
            True: "converged", False: "nonconverged",
        }[difference <= tolerance]
        if measurement["convergence"] != {
                "status": expected_status,
                "absolute_difference_ns": difference,
                "tolerance_ns": tolerance,
                "relative_difference": difference / max(
                    lower_median, higher_median)}:
            raise ValueError(f"graph convergence mismatch for {case_id}")
        return run_durations[1]
    elif mode == "paired_differential_events_around_cuda_graph_replay":
        if (measurement["estimator"] != "(T_high-T_low)/(N_high-N_low)"
                or measurement["sample_pairing"] !=
                "alternating_lower_then_higher_and_higher_then_lower"):
            raise ValueError(f"invalid differential graph contract for {case_id}")
        durations = _validate_event_samples(
            measurement["samples"], sample_count, case_id,
            "differential graph")
        _validate_event_statistics(
            durations, measurement["statistics"], case_id,
            "differential graph")
        repetition_delta = repetition_counts[1] - repetition_counts[0]
        for index, sample in enumerate(measurement["samples"]):
            expected_order = {
                True: "lower_then_higher", False: "higher_then_lower",
            }[index % 2 == 0]
            lower_sample = repetition_runs[0]["samples"][index]
            higher_sample = repetition_runs[1]["samples"][index]
            lower_replay = float(lower_sample["replay_duration_ns"])
            higher_replay = float(higher_sample["replay_duration_ns"])
            expected_positions = {
                True: (0, 1), False: (1, 0),
            }[index % 2 == 0]
            expected_duration = (
                higher_replay - lower_replay) / repetition_delta
            if (sample["replay_order"] != expected_order
                    or sample["repetition_delta"] != repetition_delta
                    or lower_sample["replay_order_position"] !=
                    expected_positions[0]
                    or higher_sample["replay_order_position"] !=
                    expected_positions[1]
                    or not math.isclose(
                        float(sample["lower_replay_duration_ns"]),
                        lower_replay, rel_tol=1e-12, abs_tol=1e-12)
                    or not math.isclose(
                        float(sample["higher_replay_duration_ns"]),
                        higher_replay, rel_tol=1e-12, abs_tol=1e-12)
                    or expected_duration <= 0.0
                    or not math.isclose(
                        durations[index], expected_duration,
                        rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"differential graph mismatch for {case_id}")
        return durations
    else:
        raise ValueError(f"unsupported C500 graph Event mode for {case_id}")


def _validate_timer_sanity(
        *, path: Path, hardware_id: str,
        baseline_commit: str) -> tuple[Mapping[str, Any], str]:
    payload = load_json_mapping(path, "timer sanity artifact")
    if (payload["schema_version"] != 1
            or payload["artifact_kind"] != "device_event_timer_sanity"
            or payload["hardware_id"] != hardware_id
            or payload["baseline_commit"] != baseline_commit
            or payload["profiler_started"] is not False):
        raise ValueError(f"timer sanity identity is invalid: {path}")
    checks = payload["checks"]
    if not isinstance(checks, dict) or not checks:
        raise TypeError(f"timer sanity checks must be non-empty: {path}")
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise ValueError(f"timer sanity has failed checks {failed}: {path}")
    return payload, file_sha256(path)


def _stability(measurement: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    durations = _validate_event_measurement(measurement, case_id)
    median_ns = statistics.median(durations)
    population_cv = statistics.pstdev(durations) / median_ns
    maximum_relative_deviation = max(
        abs(duration - median_ns) / median_ns for duration in durations)
    repetition_medians = [
        statistics.median([
            float(sample["event_duration_ns"]) for sample in run["samples"]
        ]) for run in measurement["repetition_runs"]
    ]
    difference_ns = abs(repetition_medians[0] - repetition_medians[1])
    limit_ns = max(
        CONVERGENCE_ABSOLUTE_NS,
        CONVERGENCE_RELATIVE * max(repetition_medians))
    failures = []
    if population_cv > CV_MAXIMUM:
        failures.append("population_cv")
    if maximum_relative_deviation > MAXIMUM_RELATIVE_DEVIATION:
        failures.append("maximum_relative_deviation")
    if difference_ns > limit_ns:
        failures.append("cross_repetition_convergence")
    return {
        "stable": not failures,
        "failures": failures,
        "median_ns": median_ns,
        "population_cv": population_cv,
        "maximum_relative_deviation": maximum_relative_deviation,
        "lower_repetition_median_ns": repetition_medians[0],
        "higher_repetition_median_ns": repetition_medians[1],
        "cross_repetition_difference_ns": difference_ns,
        "cross_repetition_limit_ns": limit_ns,
    }


def _validate_completion_selection(
        selection: Mapping[str, Any], completion: Mapping[str, Any]) -> None:
    if (completion["schema_version"] != 1
            or completion["artifact_kind"] !=
            "metax_c500_op_golden_completion_manifest"
            or completion["hardware_id"] != "metax_c500"):
        raise ValueError("completion manifest identity is invalid")
    expected_policy = {
        "convergence": (
            "absolute difference <= max(500 ns, 5% of larger median)"),
        "coefficient_of_variation_maximum": CV_MAXIMUM,
        "maximum_relative_deviation_maximum": MAXIMUM_RELATIVE_DEVIATION,
    }
    if completion["stability_policy"] != expected_policy:
        raise ValueError("completion stability policy is not frozen")
    completion_families = completion["families"]
    if set(completion_families) != {
            "flash_attn", "gemm", "fused_moe_unstable",
            "fused_moe_missing"}:
        raise ValueError("completion family groups are invalid")
    for group in (
            "flash_attn", "gemm", "fused_moe_unstable",
            "fused_moe_missing"):
        entry = completion_families[group]
        selected = selection["groups"][group]
        if entry["case_ids"] != selected["case_ids"]:
            raise ValueError(f"completion/selection case mismatch: {group}")
        family = GROUP_FAMILIES[group]
        if entry["canonical_dataset_sha256"] != EXPECTED_CANONICAL_SHA256[family]:
            raise ValueError(f"completion canonical SHA mismatch: {group}")
        expected_repetitions = GRAPH_REPETITIONS[family]
        if entry["graph_repetitions"] != expected_repetitions:
            raise ValueError(f"completion repetitions mismatch: {group}")


def _validate_vector_selection_raw(
        selection: Mapping[str, Any], canonical_vector: Mapping[str, Any]) -> None:
    provenance = selection["provenance"]["vector_historical_raw"]
    required = {
        "path", "baseline_commit", "source_snapshot_sha256", "runner_path",
        "runner_sha256", "timer_sanity_sha256",
        "historical_canonical_dataset_sha256", "case_artifact_count",
        "case_artifact_manifest_format", "case_artifact_manifest_sha256",
        "shards",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        raise ValueError("Vector historical provenance schema is invalid")
    root = Path(provenance["path"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Vector historical raw root is absent: {root}")
    shards = provenance["shards"]
    if not isinstance(shards, list) or len(shards) != 4:
        raise ValueError("Vector historical provenance must contain four shards")
    case_paths = []
    warning_ids = []
    strict_ids = []
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict) or set(shard) != {
                "shard_index", "case_count", "table_path", "table_sha256",
                "case_artifact_manifest_sha256"}:
            raise ValueError("Vector historical shard provenance is invalid")
        if shard["shard_index"] != shard_index:
            raise ValueError("Vector historical shard indexes are not contiguous")
        shard_root = root / f"shard_{shard_index}"
        table_path = Path(shard["table_path"]).resolve()
        if table_path != (shard_root / "vector_table.json").resolve():
            raise ValueError("Vector historical table path is outside its shard")
        if file_sha256(table_path) != shard["table_sha256"]:
            raise ValueError("Vector historical table SHA-256 mismatch")
        paths = sorted((shard_root / "cases").glob("VECTOR_*.json"))
        if len(paths) != shard["case_count"]:
            raise ValueError("Vector historical shard case count mismatch")
        if _sha256sum_manifest(paths, root) != shard[
                "case_artifact_manifest_sha256"]:
            raise ValueError("Vector historical shard artifact hash mismatch")
        case_paths.extend(paths)
        for path in paths:
            record = load_json_mapping(path, "Vector historical Event case")
            case_id = record["case_id"]
            if (path.stem != case_id or record["hardware_id"] != "metax_c500"
                    or record["family"] != "vector"
                    or record["status"] != "measured"):
                raise ValueError(f"invalid historical Vector case: {path}")
            for name in (
                    "baseline_commit", "source_snapshot_sha256", "runner_path",
                    "runner_sha256", "timer_sanity_sha256"):
                if record[name] != provenance[name]:
                    raise ValueError(
                        f"historical Vector {name} mismatch: {case_id}")
            if case_id not in canonical_vector:
                raise ValueError(f"non-canonical historical Vector case: {case_id}")
            expected_attributes = _canonical_json_sha256(
                canonical_vector[case_id]["op_info"]["Attr"])
            if record["canonical_attributes_sha256"] != expected_attributes:
                raise ValueError(
                    f"historical Vector canonical attributes mismatch: {case_id}")
            result = _stability(record["measurement"], case_id)
            if result["stable"] is True:
                strict_ids.append(case_id)
            elif result["stable"] is False:
                warning_ids.append(case_id)
            else:
                raise TypeError("Vector stability flag must be boolean")
    if (len(case_paths) != 296 or len({path.stem for path in case_paths}) != 296
            or set(path.stem for path in case_paths) != set(canonical_vector)):
        raise ValueError("historical Vector raw does not cover canonical 296")
    if provenance["case_artifact_count"] != 296:
        raise ValueError("historical Vector aggregate count mismatch")
    if _sha256sum_manifest(case_paths, root) != provenance[
            "case_artifact_manifest_sha256"]:
        raise ValueError("historical Vector aggregate artifact hash mismatch")
    canonical_order = list(canonical_vector)
    ordered_warning = [case_id for case_id in canonical_order
                       if case_id in set(warning_ids)]
    ordered_strict = [case_id for case_id in canonical_order
                      if case_id in set(strict_ids)]
    if ordered_warning != selection["groups"]["vector_warning"]["case_ids"]:
        raise ValueError("historical Vector warning selection does not replay")
    if ordered_strict != selection["existing_strict_family_case_ids"]["vector"]:
        raise ValueError("historical Vector strict selection does not replay")


def _validate_frozen_selection(
        selection_path: Path,
        ) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]],
                   Mapping[tuple[str, str], Mapping[str, Any]], set[Path]]:
    resolved = selection_path.resolve()
    selection = load_json_mapping(resolved, "full Event selection manifest")
    _exact_keys(selection, SELECTION_KEYS, "full Event selection")
    expected_scalars = {
        "schema_version": 2,
        "artifact_kind": "metax_c500_full_op_event_selection",
        "hardware_id": "metax_c500",
        "selected_case_count": 493,
        "existing_strict_case_count": 494,
        "target_case_count": 987,
    }
    for name, expected in expected_scalars.items():
        if selection[name] != expected:
            raise ValueError(f"selection {name} is not frozen")
    expected_policy = {
        "sample_population_cv_maximum": CV_MAXIMUM,
        "sample_maximum_relative_deviation_from_median_maximum":
            MAXIMUM_RELATIVE_DEVIATION,
        "cross_repetition_convergence": {
            "absolute_floor_ns": CONVERGENCE_ABSOLUTE_NS,
            "relative_to_larger_median_maximum": CONVERGENCE_RELATIVE,
        },
    }
    if selection["stability_policy"] != expected_policy:
        raise ValueError("selection stability policy is not frozen")
    expected_vector_protocols = {
        "historical_selection_protocol": HISTORICAL_VECTOR_SELECTION_PROTOCOL,
        "recapture_protocol": VECTOR_RECAPTURE_PROTOCOL,
    }
    if selection["vector_protocols"] != expected_vector_protocols:
        raise ValueError("selection Vector protocols are not frozen")
    groups = selection["groups"]
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError("selection groups are invalid")
    selected_ids = set()
    for group in GROUPS:
        entry = groups[group]
        if group == "vector_warning":
            expected_group_keys = {
                "family", "case_count", "case_ids",
                "historical_selection_protocol_ref", "recapture_protocol_ref",
            }
        elif group in {
                "flash_attn", "gemm", "fused_moe_unstable",
                "fused_moe_missing"}:
            expected_group_keys = {
                "family", "case_count", "case_ids", "runner_parameters",
            }
        else:
            raise ValueError(f"unknown selection group: {group}")
        if set(entry) != expected_group_keys:
            raise ValueError(f"selection group schema is invalid: {group}")
        if (entry["family"] != GROUP_FAMILIES[group]
                or entry["case_count"] != GROUP_COUNTS[group]
                or len(entry["case_ids"]) != GROUP_COUNTS[group]
                or len(set(entry["case_ids"])) != GROUP_COUNTS[group]):
            raise ValueError(f"selection group is invalid: {group}")
        if group == "vector_warning":
            if (entry["historical_selection_protocol_ref"] !=
                    "vector_protocols.historical_selection_protocol"
                    or entry["recapture_protocol_ref"] !=
                    "vector_protocols.recapture_protocol"):
                raise ValueError("selection Vector protocol refs are invalid")
        elif group in {
                "flash_attn", "gemm", "fused_moe_unstable",
                "fused_moe_missing"}:
            repetitions = GRAPH_REPETITIONS[entry["family"]]
            if entry["runner_parameters"] != {
                    "graph_repetitions": repetitions}:
                raise ValueError(
                    f"selection group runner parameters are invalid: {group}")
        else:
            raise ValueError(f"unknown selection group: {group}")
        for case_id in entry["case_ids"]:
            identity = (entry["family"], case_id)
            if identity in selected_ids:
                raise ValueError(f"selection repeats case: {identity}")
            selected_ids.add(identity)
    if len(selected_ids) != 493:
        raise ValueError("selection does not contain exactly 493 cases")

    provenance = selection["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
            "completion_manifest", "vector_historical_raw",
            "canonical_datasets"}:
        raise ValueError("selection provenance is invalid")
    canonical_provenance = provenance["canonical_datasets"]
    if set(canonical_provenance) != set(FAMILIES):
        raise ValueError("selection canonical families are invalid")
    canonical_datasets = {}
    protected_paths = {resolved}
    for family in FAMILIES:
        record = canonical_provenance[family]
        if not isinstance(record, dict) or set(record) != {
                "path", "sha256", "case_count"}:
            raise ValueError(f"canonical provenance is invalid: {family}")
        path = Path(record["path"]).resolve()
        protected_paths.add(path)
        if (file_sha256(path) != record["sha256"]
                or record["sha256"] != EXPECTED_CANONICAL_SHA256[family]):
            raise ValueError(f"canonical SHA-256 mismatch: {family}")
        dataset = load_json_mapping(path, f"canonical {family}")
        if (len(dataset) != FAMILY_COUNTS[family]
                or record["case_count"] != FAMILY_COUNTS[family]):
            raise ValueError(f"canonical count mismatch: {family}")
        canonical_datasets[family] = dataset
    completion_record = provenance["completion_manifest"]
    completion_path, completion_sha256 = _path_record(
        completion_record, "completion manifest")
    protected_paths.add(completion_path)
    if completion_sha256 != EXPECTED_COMPLETION_SHA256:
        raise ValueError("completion manifest is not the frozen artifact")
    completion = load_json_mapping(completion_path, "completion manifest")
    _validate_completion_selection(selection, completion)
    _validate_vector_selection_raw(selection, canonical_datasets["vector"])
    vector_root = Path(provenance["vector_historical_raw"]["path"]).resolve()
    protected_paths.add(vector_root)

    legacy_evidence = {}
    existing_vector_ids = set(
        selection["existing_strict_family_case_ids"]["vector"])
    for path in sorted(vector_root.glob("shard_*/cases/VECTOR_*.json")):
        if path.stem in existing_vector_ids:
            legacy_evidence[("vector", path.stem)] = {
                "path": str(path.resolve()), "sha256": file_sha256(path)}
    evidence_root = completion_path.parent.parent
    legacy_audits = {
        "flash_attn": (
            evidence_root / "metax_c500_flash_graph_partial" / "audit.json",
            EXPECTED_FLASH_LEGACY_AUDIT_SHA256),
        "remaining": (
            evidence_root / "metax_c500_full_graph_v2" /
            "c500_remaining_audit.json",
            EXPECTED_REMAINING_LEGACY_AUDIT_SHA256),
    }
    for label, (audit_path, expected_sha) in legacy_audits.items():
        if file_sha256(audit_path) != expected_sha:
            raise ValueError(f"frozen legacy audit SHA-256 mismatch: {label}")
        protected_paths.add(audit_path)
        payload = load_json_mapping(audit_path, f"legacy {label} audit")
        if label == "flash_attn":
            family_records = {"flash_attn": payload["cases"]}
            prefix_name = "metax_c500_flash_graph_partial"
        elif label == "remaining":
            family_records = {
                family: payload["families"][family]["cases"]
                for family in ("gemm", "fused_moe")
            }
            prefix_name = "metax_c500_full_graph_v2"
        else:
            raise ValueError(f"unknown legacy audit label: {label}")
        prefix = Path("experiment_logs/kernel_golden") / prefix_name
        for family, records in family_records.items():
            for record in records:
                if (label == "remaining" and record["status"] != "stable"):
                    continue
                relative = Path(record["raw_case_path"])
                try:
                    suffix = relative.relative_to(prefix)
                except ValueError as error:
                    raise ValueError(
                        f"legacy raw path is outside frozen audit: {relative}") from error
                raw_path = audit_path.parent / suffix
                if (file_sha256(raw_path) != record["raw_case_sha256"]
                        or raw_path.stem != record["case_id"]):
                    raise ValueError(
                        f"legacy raw evidence mismatch: {family}/{record['case_id']}")
                legacy_evidence[(family, record["case_id"])] = {
                    "path": str(raw_path.resolve()),
                    "sha256": record["raw_case_sha256"],
                }

    family_case_ids = selection["family_case_ids"]
    existing_ids = selection["existing_strict_family_case_ids"]
    if set(family_case_ids) != set(FAMILIES) or set(existing_ids) != set(FAMILIES):
        raise ValueError("selection family membership maps are invalid")
    existing_count = 0
    for family in FAMILIES:
        selected = family_case_ids[family]
        existing = existing_ids[family]
        if (len(selected) != SELECTED_FAMILY_COUNTS[family]
                or len(selected) != len(set(selected))
                or len(existing) != len(set(existing))
                or set(selected) & set(existing)
                or set(selected) | set(existing) != set(canonical_datasets[family])):
            raise ValueError(f"selection exact complement failed: {family}")
        expected_flat = [
            case_id for group in GROUPS
            if GROUP_FAMILIES[group] == family
            for case_id in groups[group]["case_ids"]]
        if selected != expected_flat:
            raise ValueError(f"selection family/group ordering mismatch: {family}")
        existing_count += len(existing)
    if existing_count != 494:
        raise ValueError("existing strict selection does not contain 494 cases")
    expected_legacy = {
        (family, case_id) for family in FAMILIES
        for case_id in existing_ids[family]}
    if set(legacy_evidence) != expected_legacy:
        raise ValueError("frozen legacy raw audits do not cover exact existing 494")
    return selection, canonical_datasets, legacy_evidence, protected_paths


def _parse_source_files_manifest(path: Path) -> Mapping[str, str]:
    entries = {}
    for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError(
                f"invalid source-files manifest line {line_number}: {path}")
        digest = line[:64]
        relative = line[66:]
        if (len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or not relative.startswith("./")
                or Path(relative[2:]).is_absolute()
                or ".." in Path(relative[2:]).parts):
            raise ValueError(
                f"invalid source-files manifest entry {line_number}: {path}")
        normalized = relative[2:]
        if normalized in entries:
            raise ValueError(f"duplicate source-files entry: {normalized}")
        entries[normalized] = digest
    if not entries:
        raise ValueError("source-files manifest is empty")
    return entries




def _tar_manifest_hashes(
        archive: Path, expected_paths: set[str]) -> Mapping[str, str]:
    hashes = {}
    with tarfile.open(archive, mode="r:*") as stream:
        for member in stream.getmembers():
            relative = member.name.removeprefix("./")
            if relative not in expected_paths:
                continue
            if relative in hashes or not member.isfile():
                raise ValueError(
                    f"source archive member is duplicate/non-regular: {relative}")
            extracted = stream.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read source archive member: {relative}")
            hashes[relative] = _bytes_sha256(extracted.read())
    if set(hashes) != expected_paths:
        raise ValueError("source archive is missing manifest entries")
    return hashes


def _nullable_path_and_sha(
        path_value: Any, sha_value: Any,
        label: str) -> tuple[Path | None, str | None]:
    if path_value is None:
        if sha_value is None:
            return None, None
        if not isinstance(sha_value, str) or len(sha_value) != 64:
            raise ValueError(f"{label} SHA-256 is invalid")
        return None, sha_value
    if not isinstance(sha_value, str) or len(sha_value) != 64:
        raise ValueError(f"{label} SHA-256 is invalid")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is invalid")
    path = Path(path_value).resolve()
    if file_sha256(path) != sha_value:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path, sha_value


def _validate_source(source_id: str, source: Mapping[str, Any]
                     ) -> tuple[Mapping[str, Any], set[Path]]:
    if not isinstance(source, dict):
        raise TypeError(f"campaign source must be an object: {source_id}")
    _exact_keys(source, SOURCE_KEYS, f"campaign source {source_id}")
    role = source["role"]
    if role not in {"new_capture", "legacy_frozen"}:
        raise ValueError(f"invalid source role: {source_id}")
    status = source["archive_status"]
    if status not in {
            "complete", "legacy_archive_without_manifest",
            "legacy_archive_unavailable"}:
        raise ValueError(f"invalid source archive status: {source_id}")
    if role == "new_capture" and status != "complete":
        raise ValueError("new capture source requires a complete archive")
    baseline = source["baseline_commit"]
    if (not isinstance(baseline, str) or len(baseline) != 40
            or any(character not in "0123456789abcdef" for character in baseline)):
        raise ValueError(f"invalid source baseline commit: {source_id}")
    archive, archive_sha = _nullable_path_and_sha(
        source["source_archive"], source["source_archive_sha256"],
        f"source archive {source_id}")
    descriptor, descriptor_sha = _nullable_path_and_sha(
        source["source_descriptor"], source["source_descriptor_sha256"],
        f"source descriptor {source_id}")
    files_manifest, files_sha = _nullable_path_and_sha(
        source["source_files_manifest"], source["source_files_manifest_sha256"],
        f"source files manifest {source_id}")
    timer_path, timer_sha = _nullable_path_and_sha(
        source["timer_sanity"], source["timer_sanity_sha256"],
        f"timer sanity {source_id}")
    if timer_path is None:
        raise ValueError(f"timer sanity cannot be unavailable: {source_id}")
    if archive_sha is None or timer_sha is None:
        raise ValueError(f"source archive/timer hashes are required: {source_id}")
    timer, validated_timer_sha = _validate_timer_sanity(
        path=timer_path, hardware_id="metax_c500", baseline_commit=baseline)
    if validated_timer_sha != timer_sha:
        raise AssertionError("timer sanity digest changed during validation")
    _validate_event_measurement(timer["measurement"], f"timer/{source_id}")
    if timer["source_snapshot_sha256"] != archive_sha:
        raise ValueError(f"timer/source archive mismatch: {source_id}")
    if timer["timer"] != {
            "api": "torch.cuda.Event",
            "implementation": "wcudaEvent_to_maca_mcEvent",
            "returned_unit": "millisecond",
            "nanoseconds_per_unit": 1_000_000}:
        raise ValueError(f"invalid Device Event timer: {source_id}")
    protected = {timer_path}
    runner_sha256 = {}
    snapshot_root_value = source["snapshot_root"]
    if status == "complete":
        if (archive is None or descriptor is None or files_manifest is None
                or not isinstance(snapshot_root_value, str)
                or not snapshot_root_value):
            raise ValueError(f"complete source paths are missing: {source_id}")
        snapshot_root = Path(snapshot_root_value).resolve()
        if not snapshot_root.is_dir():
            raise FileNotFoundError(f"source snapshot root is absent: {source_id}")
        if descriptor != (snapshot_root / "SOURCE_SNAPSHOT.json").resolve():
            raise ValueError(f"source descriptor path mismatch: {source_id}")
        descriptor_payload = load_json_mapping(descriptor, "source descriptor")
        _exact_keys(
            descriptor_payload, SNAPSHOT_DESCRIPTOR_KEYS, "source descriptor")
        if (descriptor_payload["schema_version"] != 1
                or descriptor_payload["source_commit"] != baseline
                or descriptor_payload["source_archive_sha256"] != archive_sha
                or descriptor_payload["source_files_manifest_sha256"] != files_sha
                or (snapshot_root / descriptor_payload["source_files_manifest"])
                .resolve() != files_manifest):
            raise ValueError(f"source descriptor contents mismatch: {source_id}")
        manifest_entries = _parse_source_files_manifest(files_manifest)
        if len(manifest_entries) != descriptor_payload["source_file_count"]:
            raise ValueError(f"source-files count mismatch: {source_id}")
        archive_hashes = _tar_manifest_hashes(
            archive, set(manifest_entries))
        for relative, expected in manifest_entries.items():
            snapshot_file = snapshot_root / relative
            if (not snapshot_file.is_file()
                    or file_sha256(snapshot_file) != expected):
                raise ValueError(f"snapshot source file mismatch: {relative}")
            if archive_hashes[relative] != expected:
                raise ValueError(f"archive source file mismatch: {relative}")
        for family, relative in RUNNER_PATHS.items():
            if relative not in manifest_entries:
                raise ValueError(f"runner absent from source manifest: {relative}")
            expected = manifest_entries[relative]
            runner_sha256[family] = expected
        protected.update({archive, descriptor, files_manifest, snapshot_root})
    elif status == "legacy_archive_without_manifest":
        if (role != "legacy_frozen" or archive is None
                or not isinstance(snapshot_root_value, str)
                or not snapshot_root_value or descriptor is not None
                or files_manifest is not None):
            raise ValueError(f"legacy archive-only source is malformed: {source_id}")
        snapshot_root = Path(snapshot_root_value).resolve()
        if not snapshot_root.is_dir():
            raise FileNotFoundError(f"legacy snapshot root is absent: {source_id}")
        protected.update({archive, snapshot_root})
    elif status == "legacy_archive_unavailable":
        if (role != "legacy_frozen" or archive is not None
                or descriptor is not None or files_manifest is not None
                or snapshot_root_value is not None):
            raise ValueError(f"unavailable legacy source must use null paths: {source_id}")
        snapshot_root = None
    else:
        raise ValueError(f"invalid source archive status: {source_id}")
    if role == "new_capture":
        if (baseline != EXPECTED_NEW_BASELINE
                or archive_sha != EXPECTED_NEW_ARCHIVE_SHA256
                or descriptor_sha != EXPECTED_NEW_DESCRIPTOR_SHA256
                or timer_sha != EXPECTED_NEW_TIMER_SHA256):
            raise ValueError("new capture source is not the frozen r2 source")
    elif role == "legacy_frozen":
        pass
    else:
        raise ValueError(f"invalid source role: {source_id}")
    return {
        "source_id": source_id,
        "role": role,
        "archive_status": status,
        "baseline_commit": baseline,
        "source_snapshot_sha256": archive_sha,
        "snapshot_root": str(snapshot_root) if snapshot_root is not None else None,
        "timer_sanity_path": str(timer_path),
        "timer_sanity_sha256": timer_sha,
        "runner_sha256": runner_sha256,
    }, protected


def _validate_process_isolation(
        process_isolation: Mapping[str, Any],
        ) -> tuple[Mapping[str, Mapping[str, Any]], set[Path], Mapping[str, Any]]:
    isolation_path, isolation_sha = _path_record(
        process_isolation, "process-isolation manifest")
    isolation = load_json_mapping(isolation_path, "process-isolation manifest")
    _exact_keys(isolation, PROCESS_MANIFEST_KEYS, "process-isolation manifest")
    if (isolation["schema_version"] != 1
            or isolation["artifact_kind"] !=
            "metax_c500_full_system_process_isolation"
            or isolation["hardware_id"] != "metax_c500"):
        raise ValueError("process-isolation manifest identity is invalid")
    collector = Path(isolation["collector_path"]).resolve()
    if file_sha256(collector) != isolation["collector_sha256"]:
        raise ValueError("process-isolation collector SHA-256 mismatch")
    snapshots = isolation["snapshots"]
    if not isinstance(snapshots, list):
        raise ValueError("process-isolation snapshots must be a list")
    indexed = {}
    protected = {isolation_path, collector}
    for record in snapshots:
        if not isinstance(record, dict) or set(record) != {
                "snapshot_id", "path", "sha256"}:
            raise ValueError("process-isolation snapshot record is invalid")
        snapshot_id = record["snapshot_id"]
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("process-isolation snapshot ID is invalid")
        if snapshot_id in indexed:
            raise ValueError(f"duplicate process snapshot ID: {snapshot_id}")
        path, sha = _path_record(
            {"path": record["path"], "sha256": record["sha256"]},
            f"process snapshot {snapshot_id}")
        snapshot = load_json_mapping(path, f"process snapshot {snapshot_id}")
        _exact_keys(snapshot, PROCESS_SNAPSHOT_KEYS, "process snapshot")
        if (snapshot["schema_version"] != 1
                or snapshot["artifact_kind"] !=
                "metax_c500_full_system_process_snapshot"
                or snapshot["scope"] != "host_all_processes"
                or snapshot["collection_complete"] is not True
                or snapshot["host"] != isolation["host"]):
            raise ValueError(f"process snapshot contract mismatch: {snapshot_id}")
        timestamp = require_positive_integer(
            snapshot["timestamp_wall_time_ns"], "process snapshot timestamp")
        processes = snapshot["processes"]
        device_processes = snapshot["device_processes"]
        if not isinstance(processes, list) or not isinstance(device_processes, list):
            raise TypeError("process snapshot process lists are invalid")
        process_index = {}
        for process in processes:
            if not isinstance(process, dict):
                raise TypeError("process snapshot process must be an object")
            _exact_keys(process, PROCESS_KEYS, "process snapshot process")
            pid = require_positive_integer(process["pid"], "process PID")
            if pid in process_index:
                raise ValueError(f"process snapshot repeats PID {pid}")
            if (isinstance(process["ppid"], bool)
                    or not isinstance(process["ppid"], int)
                    or process["ppid"] < 0):
                raise ValueError("process PPID must be a nonnegative integer")
            if (isinstance(process["process_group_id"], bool)
                    or not isinstance(process["process_group_id"], int)
                    or process["process_group_id"] < 0):
                raise ValueError(
                    "snapshot process group ID must be a nonnegative integer")
            argv = process["argv"]
            devices = process["physical_devices"]
            if (not isinstance(argv, list) or not argv
                    or any(not isinstance(value, str) for value in argv)
                    or not isinstance(devices, list)
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           or not 0 <= value <= 3 for value in devices)):
                raise ValueError(f"invalid process record for PID {pid}")
            command = "\0".join(argv).lower()
            if any(marker in command for marker in FORBIDDEN_PROFILER_MARKERS):
                raise ValueError(
                    f"profiler process present in snapshot {snapshot_id}: {pid}")
            process_index[pid] = process
        device_index = set()
        for process in device_processes:
            if not isinstance(process, dict):
                raise TypeError("device process must be an object")
            _exact_keys(process, DEVICE_PROCESS_KEYS, "device process")
            device = process["physical_device"]
            pid = process["pid"]
            if (isinstance(device, bool) or not isinstance(device, int)
                    or not 0 <= device <= 3):
                raise ValueError("device process has invalid physical device")
            require_positive_integer(pid, "device process PID")
            identity = (device, pid)
            if identity in device_index:
                raise ValueError(f"duplicate device process: {identity}")
            if pid not in process_index or device not in process_index[pid][
                    "physical_devices"]:
                raise ValueError(f"device/process snapshot disagreement: {identity}")
            device_index.add(identity)
        indexed[snapshot_id] = {
            "path": str(path), "sha256": sha, "timestamp_wall_time_ns": timestamp,
            "processes": process_index, "device_processes": device_index,
        }
        protected.add(path)
    snapshot_root = isolation_path.parent / "process_isolation"
    actual_snapshot_paths = {
        path.resolve() for path in snapshot_root.glob("*.json")}
    declared_snapshot_paths = {
        Path(record["path"]).resolve() for record in snapshots}
    if actual_snapshot_paths != declared_snapshot_paths:
        raise ValueError("process-isolation snapshot inventory mismatch")
    return indexed, protected, {
        "path": str(isolation_path), "sha256": isolation_sha,
        "host": isolation["host"], "snapshot_count": len(indexed),
        "collector_path": str(collector),
        "collector_sha256": isolation["collector_sha256"],
    }


def _validate_attempt_isolation(
        attempt: Mapping[str, Any], snapshots: Mapping[str, Mapping[str, Any]]) -> None:
    pre_id = attempt["preflight_snapshot_id"]
    during_ids = attempt["during_snapshot_ids"]
    post_id = attempt["postflight_snapshot_id"]
    completed = attempt["terminal_status"] == "completed"
    if (not isinstance(during_ids, list)
            or any(not isinstance(value, str) or value not in snapshots
                   for value in during_ids)
            or (pre_id is not None
                and (not isinstance(pre_id, str) or pre_id not in snapshots))
            or (post_id is not None
                and (not isinstance(post_id, str) or post_id not in snapshots))
            or (completed and (
                pre_id is None or post_id is None or not during_ids))):
        raise ValueError(
            f"attempt process-isolation references are incomplete: "
            f"{attempt['attempt_id']}")
    start = attempt["start_wall_time_ns"]
    end = attempt["end_wall_time_ns"]
    pre = snapshots[pre_id] if pre_id is not None else None
    post = snapshots[post_id] if post_id is not None else None
    if pre is not None and pre["timestamp_wall_time_ns"] > start:
        raise ValueError(f"preflight snapshot is after attempt start: {attempt['attempt_id']}")
    if post is not None and post["timestamp_wall_time_ns"] < end:
        raise ValueError(f"postflight snapshot is before attempt end: {attempt['attempt_id']}")
    pid = attempt["pid"]
    pgid = attempt["process_group_id"]
    device = attempt["physical_device"]
    boundary_snapshots = [
        snapshot for snapshot in (pre, post) if snapshot is not None]
    if any(
            owner_device == device
            for snapshot in boundary_snapshots
            for owner_device, _ in snapshot["device_processes"]):
        raise ValueError(
            f"attempt device is not exclusive outside its window: "
            f"{attempt['attempt_id']}")
    if pgid is not None and any(
            process["process_group_id"] == pgid
            for snapshot in boundary_snapshots
            for process in snapshot["processes"].values()):
        raise ValueError(
            f"attempt process group exists outside its process window: "
            f"{attempt['attempt_id']}")
    launcher_pgid = attempt["launcher_process_group_id"]
    if launcher_pgid is not None and any(
            process["process_group_id"] == launcher_pgid
            for snapshot in boundary_snapshots
            for process in snapshot["processes"].values()):
        raise ValueError(
            f"outer docker process group exists outside attempt window: "
            f"{attempt['attempt_id']}")
    boundary_ids = {value for value in (pre_id, post_id) if value is not None}
    interval_snapshot_ids = [
        snapshot_id for snapshot_id, snapshot in snapshots.items()
        if snapshot_id not in boundary_ids
        and start <= snapshot["timestamp_wall_time_ns"] <= end]
    if set(snapshots) != boundary_ids | set(interval_snapshot_ids):
        raise ValueError(
            f"process snapshot is outside attempt evidence window: "
            f"{attempt['attempt_id']}")
    if not set(during_ids).issubset(interval_snapshot_ids):
        raise ValueError(
            f"during snapshot is outside attempt window: {attempt['attempt_id']}")
    launcher_seen = False
    for snapshot_id in interval_snapshot_ids:
        snapshot = snapshots[snapshot_id]
        launcher_pid = attempt["launcher_pid"]
        target_members = [
            member_pid for member_device, member_pid in
            snapshot["device_processes"] if member_device == device]
        device_members = [] if pgid is None else [
            member_pid for member_pid in target_members
            if snapshot["processes"][member_pid]["process_group_id"] == pgid]
        foreign_members = [
            member_pid for member_pid in target_members
            if pgid is None or
            snapshot["processes"][member_pid]["process_group_id"] != pgid]
        if foreign_members:
            raise ValueError(
                f"attempt process/device identity mismatch: "
                f"{attempt['attempt_id']}")
        runner_present = pid is not None and pid in snapshot["processes"]
        if runner_present:
            process = snapshot["processes"][pid]
            if (process["process_group_id"] != pgid
                    or process["argv"] != attempt["command_argv"]):
                raise ValueError(
                    f"attempt process/device identity mismatch: "
                    f"{attempt['attempt_id']}")
        if launcher_pid is not None and launcher_pid in snapshot["processes"]:
            launcher_seen = True
            launcher = snapshot["processes"][launcher_pid]
            if (launcher["process_group_id"] != launcher_pgid
                    or launcher["argv"] != attempt["launcher_command_argv"]):
                raise ValueError(
                    f"outer docker process identity mismatch: "
                    f"{attempt['attempt_id']}")
        if snapshot_id in during_ids and (
                not runner_present or not device_members):
            raise ValueError(
                f"active during snapshot lacks runner/device owner: "
                f"{attempt['attempt_id']}")
    if completed and not launcher_seen:
        raise ValueError(
            f"completed attempt never observes outer docker launcher: "
            f"{attempt['attempt_id']}")


def _require_workload_matches(
        record: Mapping[str, Any], attributes: Mapping[str, Any]) -> None:
    family = record["family"]
    case_id = record["case_id"]
    workload = record["workload"]
    if workload["dtype"] != "bf16":
        raise ValueError(f"capture workload dtype is not bf16: {case_id}")
    family_fields = {
        "gemm": ("B", "M", "K", "N"),
        "flash_attn": (
            "B", "Q_LEN", "HQ", "HKV", "D", "SP_REQUESTED",
            "Q_LENS", "KV_LENS", "kv_layout", "causal"),
        "fused_moe": ("TOK", "E", "TOPK", "H", "INT", "tok_per_expert"),
    }
    if family in family_fields:
        for name in family_fields[family]:
            if workload[name] != attributes[name]:
                raise ValueError(
                    f"workload/canonical mismatch: {case_id}/{name}")
        if family == "flash_attn" and workload["S"] != attributes["KV_LEN"]:
            raise ValueError(
                f"workload/canonical mismatch: {case_id}/S-to-KV_LEN")
        return
    vector_fields = {
        "rmsnorm": ("op", "tok", "hidden", "dtype_bytes"),
        "add_rmsnorm": ("op", "tok", "hidden", "dtype_bytes"),
        "rope": (
            "op", "tok", "num_heads", "num_kv_heads", "head_dim",
            "dtype_bytes"),
        "silu": ("op", "tok", "intermediate", "dtype_bytes"),
        "elementwise": ("op", "op2", "tok", "hidden", "dtype_bytes"),
        "kvcache": ("op", "tok", "num_kv_heads", "head_dim", "dtype_bytes"),
    }
    op = attributes["op"]
    if family != "vector" or op not in vector_fields:
        raise ValueError(f"unsupported workload family/op: {family}/{op}")
    for name in vector_fields[op]:
        if workload[name] != attributes[name]:
            raise ValueError(f"workload/canonical mismatch: {case_id}/{name}")
    for name, expected in build_vector_buffer_contract(attributes).items():
        if workload[name] != expected:
            raise ValueError(f"Vector buffer contract mismatch: {case_id}/{name}")


def _validate_vector_unique_buffers(
        record: Mapping[str, Any], attributes: Mapping[str, Any],
        minimum_distinct_input_mib: int, *,
        require_pre_event_operation_count: bool) -> None:
    case_id = record["case_id"]
    measurement = record["measurement"]
    expected = build_vector_distinct_input_contract(
        attributes, minimum_distinct_input_mib)
    repetitions = list(expected["repetition_counts"])
    expected = dict(expected)
    del expected["repetition_counts"]
    workload = record["workload"]
    buffers = build_vector_buffer_contract(attributes)
    higher = repetitions[1]
    pre_event_key = "pre_event_operation_executions_per_replay"
    if require_pre_event_operation_count:
        pre_event_contract_valid = (
            pre_event_key in measurement and measurement[pre_event_key] == 0)
    else:
        pre_event_contract_valid = pre_event_key not in measurement
    if (measurement["mode"] !=
            "paired_differential_events_around_cuda_graph_replay"
            or measurement["repetition_counts"] != repetitions
            or measurement["operation_cycle_length"] != higher
            or measurement["distinct_input_contract"] != expected
            or measurement["cache_eviction_requested_bytes"]
            != 256 * 1024 * 1024
            or measurement["cache_eviction_bytes_before_replay"]
            != 256 * 1024 * 1024
            or not pre_event_contract_valid
            or workload["buffer_count"] != higher
            or workload["rotation_mode"] !=
            "unique_buffer_per_selected_replay_call"
            or workload["selected_replay_distinct_buffer_count"] != higher
            or workload["distinct_input_contract"] != expected
            or workload["rotating_working_set_bytes"] !=
            buffers["per_call_distinct_working_set_bytes"] * higher
            or workload["selected_replay_distinct_working_set_bytes"] !=
            buffers["per_call_distinct_working_set_bytes"] * higher
            or workload["cache_eviction_bytes_before_replay"]
            != 256 * 1024 * 1024):
        raise ValueError(f"Vector unique-buffer contract mismatch: {case_id}")


def _validate_record_without_runner_tree(
        *, record: Mapping[str, Any], path: Path, source: Mapping[str, Any],
        family: str, case_id: str, attributes: Mapping[str, Any]) -> None:
    _exact_keys(record, EVENT_CASE_REQUIRED_KEYS, f"Event case {case_id}")
    expected = {
        "schema_version": 1, "status": "measured", "unsupported_reason": None,
        "hardware_id": "metax_c500", "family": family, "case_id": case_id,
        "baseline_commit": source["baseline_commit"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "timer_sanity_sha256": source["timer_sanity_sha256"],
        "canonical_attributes_sha256": _canonical_json_sha256(attributes),
    }
    for name, value in expected.items():
        if record[name] != value:
            raise ValueError(f"legacy Event field mismatch: {case_id}/{name}")
    if path.stem != case_id or record["runner_path"] != RUNNER_PATHS[family]:
        raise ValueError(f"legacy Event path/runner mismatch: {case_id}")
    if (not isinstance(record["runner_sha256"], str)
            or len(record["runner_sha256"]) != 64):
        raise ValueError(f"legacy Event runner SHA is invalid: {case_id}")
    _validate_event_measurement(record["measurement"], case_id)


def _validate_case_record(
        *, record: Mapping[str, Any], path: Path, hardware_id: str,
        family: str, case_id: str, baseline_commit: str,
        source_snapshot_sha256: str, timer_sanity_sha256: str,
        canonical_attributes: Mapping[str, Any], runner_root: Path,
        ) -> list[float]:
    source = {
        "baseline_commit": baseline_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "timer_sanity_sha256": timer_sanity_sha256,
    }
    if hardware_id != "metax_c500":
        raise ValueError("C500 Event audit received another hardware ID")
    _validate_record_without_runner_tree(
        record=record, path=path, source=source, family=family,
        case_id=case_id, attributes=canonical_attributes)
    runner_path = Path(record["runner_path"])
    if runner_path.is_absolute() or ".." in runner_path.parts:
        raise ValueError(f"Event runner path must be repository-relative: {case_id}")
    resolved_runner = runner_root.resolve() / runner_path
    if not resolved_runner.is_file():
        raise FileNotFoundError(f"recorded Event runner is absent: {resolved_runner}")
    if record["runner_sha256"] != file_sha256(resolved_runner):
        raise ValueError(f"recorded Event runner SHA-256 mismatch: {case_id}")
    for name in (
            "timer", "stream_contract", "allocation_contract", "workload",
            "environment"):
        if not isinstance(record[name], dict) or not record[name]:
            raise TypeError(f"Event case {case_id} {name} must be non-empty")
    return _validate_event_measurement(record["measurement"], case_id)


def _validate_case(
        *, record: Mapping[str, Any], path: Path,
        canonical_entry: Mapping[str, Any], source: Mapping[str, Any],
        runtime_environment: Mapping[str, Any], require_new_contract: bool,
        ) -> Mapping[str, Any]:
    family = record["family"]
    case_id = record["case_id"]
    attributes = canonical_entry["op_info"]["Attr"]
    snapshot_root = source["snapshot_root"]
    if snapshot_root is None:
        _validate_record_without_runner_tree(
            record=record, path=path, source=source, family=family,
            case_id=case_id, attributes=attributes)
    else:
        durations = _validate_case_record(
            record=record, path=path, hardware_id="metax_c500",
            family=family, case_id=case_id,
            baseline_commit=source["baseline_commit"],
            source_snapshot_sha256=source["source_snapshot_sha256"],
            timer_sanity_sha256=source["timer_sanity_sha256"],
            canonical_attributes=attributes, runner_root=Path(snapshot_root))
        if durations is None:
            raise ValueError(f"Event case is unsupported: {case_id}")
    if record["runner_path"] != RUNNER_PATHS[family]:
        raise ValueError(f"Event runner path mismatch: {case_id}")
    if source["archive_status"] == "complete" and record["runner_sha256"] != (
            source["runner_sha256"][family]):
        raise ValueError(f"Event runner is not archive-bound: {case_id}")
    if record["environment"] != runtime_environment:
        raise ValueError(f"runtime environment mismatch: {case_id}")
    if (record["environment"]["device_name"] != "MetaX C500"
            or record["environment"]["device_index"] != 0
            or record["environment"]["torch_version"] !=
            "2.4.0+metax3.7.2.0"
            or record["environment"]["torch_cuda_version_field"] != "11.6"):
        raise ValueError(f"runtime identity mismatch: {case_id}")
    if (record["stream_contract"]["event_stream"] !=
            "torch.cuda.current_stream"
            or record["stream_contract"]["all_op_streams_joined"] is not True
            or record["allocation_contract"]["inputs_preallocated"] is not True):
        raise ValueError(f"OP Event boundary contract mismatch: {case_id}")
    _require_workload_matches(record, attributes)
    measurement = record["measurement"]
    if (measurement["schema_version"] != 3
            or measurement["warmup_iterations"] != 5
            or measurement["mode"] not in {
                "external_events_around_cuda_graph_replay",
                "paired_differential_events_around_cuda_graph_replay"}):
        raise ValueError(f"Event Graph measurement contract mismatch: {case_id}")
    if family == "vector":
        if require_new_contract:
            minimum_distinct_input_mib = VECTOR_RECAPTURE_PROTOCOL[
                "minimum_distinct_input_mib"]
        else:
            minimum_distinct_input_mib = HISTORICAL_VECTOR_SELECTION_PROTOCOL[
                "minimum_distinct_input_mib"]
        _validate_vector_unique_buffers(
            record, attributes, minimum_distinct_input_mib,
            require_pre_event_operation_count=require_new_contract)
        expected_timer_mode = (
            "paired_differential_events_around_rotating_cuda_graph_replay")
        expected_cache_condition = (
            "queued_pre_replay_device_sweep_without_intra_replay_buffer_reuse")
        if require_new_contract and measurement["sample_count"] != 50:
            raise ValueError(f"new Vector sample count mismatch: {case_id}")
    elif family in {"flash_attn", "gemm", "fused_moe"}:
        expected_timer_mode = "external_events_around_cuda_graph_replay"
        expected_cache_condition = (
            "queued_pre_replay_device_sweep_with_intra_replay_reuse")
        if (measurement["operation_cycle_length"] != 1
                or measurement["cache_eviction_requested_bytes"]
                != 2048 * 1024 * 1024
                or measurement["cache_eviction_bytes_before_replay"]
                != 2048 * 1024 * 1024
                or record["workload"]["cache_eviction_bytes_before_replay"]
                != 2048 * 1024 * 1024):
            raise ValueError(f"Event cache contract mismatch: {case_id}")
        if require_new_contract:
            expected_repetitions = GRAPH_REPETITIONS[family]
            if (measurement["sample_count"] != 20
                    or measurement["repetition_counts"] != expected_repetitions):
                raise ValueError(f"new Event sampling contract mismatch: {case_id}")
    else:
        raise ValueError(f"unknown Event family: {family}")
    expected_timer = {
        "api": "torch.cuda.Event",
        "implementation": "wcudaEvent_to_maca_mcEvent",
        "returned_unit": "millisecond",
        "nanoseconds_per_unit": 1_000_000,
        "cupti_identity": "not_proven",
        "mode": expected_timer_mode,
        "cache_condition": expected_cache_condition,
    }
    if record["timer"] != expected_timer:
        raise ValueError(f"Device Event timer mismatch: {case_id}")
    result = dict(_stability(measurement, case_id))
    result.update({
        "family": family, "case_id": case_id, "path": str(path),
        "sha256": file_sha256(path),
        "canonical_attributes_sha256": _canonical_json_sha256(attributes),
    })
    return result


def _expected_table_row(record: Mapping[str, Any], canonical_entry: Mapping[str, Any]
                        ) -> Mapping[str, Any]:
    attributes = canonical_entry["op_info"]["Attr"]
    measurement = record["measurement"]
    repetitions = measurement["repetition_counts"]
    runs = measurement["repetition_runs"]
    lower = runs[0]["statistics"]
    higher = runs[1]["statistics"]
    if measurement["mode"] == "paired_differential_events_around_cuda_graph_replay":
        selected_ns = measurement["statistics"]["median_ns"]
        convergence_status = "paired_differential"
    elif measurement["mode"] == "external_events_around_cuda_graph_replay":
        selected_ns = higher["median_ns"]
        convergence_status = measurement["convergence"]["status"]
    else:
        raise ValueError("unsupported Event measurement mode")
    canonical_ns = canonical_entry["op_info"]["evaluation"]["timing"][
        "total_duration_ns"]
    signed_error_ns = selected_ns - canonical_ns
    optional = (
        "op", "op2", "tok", "hidden", "intermediate", "num_heads",
        "num_kv_heads", "head_dim", "batch", "B", "M", "K", "N", "S",
        "Q_LEN", "HQ", "HKV", "D", "SP_REQUESTED", "kv_layout", "causal",
        "TOK", "E", "TOPK", "H", "INT")
    row = dict.fromkeys(TABLE_ROW_KEYS)
    row.update({
        "family": record["family"], "case_id": record["case_id"],
        "attributes_json": json.dumps(
            attributes, separators=(",", ":"), sort_keys=True),
        "executed_workload_json": json.dumps(
            record["workload"], separators=(",", ":"), sort_keys=True),
        "canonical_duration_us": canonical_ns / 1000.0,
        "lower_repetitions": repetitions[0],
        "higher_repetitions": repetitions[1],
        "lower_median_us": lower["median_ns"] / 1000.0,
        "lower_p10_us": lower["p10_ns"] / 1000.0,
        "lower_p90_us": lower["p90_ns"] / 1000.0,
        "lower_cv": lower["coefficient_of_variation"],
        "higher_median_us": higher["median_ns"] / 1000.0,
        "higher_p10_us": higher["p10_ns"] / 1000.0,
        "higher_p90_us": higher["p90_ns"] / 1000.0,
        "higher_cv": higher["coefficient_of_variation"],
        "selected_duration_us": selected_ns / 1000.0,
        "absolute_error_us": abs(signed_error_ns) / 1000.0,
        "relative_error": abs(signed_error_ns) / canonical_ns,
        "signed_relative_error": signed_error_ns / canonical_ns,
        "convergence_status": convergence_status,
        "status": "measured", "error": None,
    })
    for name in optional:
        if name in attributes:
            row[name] = attributes[name]
    return row


def _validate_attempt_table(
        *, attempt: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]],
        canonical: Mapping[str, Any], source: Mapping[str, Any],
        canonical_sha256: str) -> Mapping[str, Any] | None:
    root = Path(attempt["root"]).resolve()
    family = attempt["family"]
    table_path = root / f"{family}_table.json"
    if attempt["cohort"] == "existing_strict":
        return None
    if not records:
        if table_path.exists():
            raise ValueError(
                f"attempt with no measured cases has a table: "
                f"{attempt['attempt_id']}")
        return None
    table = load_json_mapping(table_path, f"attempt table {attempt['attempt_id']}")
    _exact_keys(table, TABLE_KEYS, f"attempt table {attempt['attempt_id']}")
    expected = {
        "schema_version": 1, "hardware_id": "metax_c500", "family": family,
        "baseline_commit": source["baseline_commit"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "timer_sanity_sha256": source["timer_sanity_sha256"],
        "canonical_dataset_sha256": canonical_sha256,
        "expected_case_count": len(attempt["planned_case_ids"]),
    }
    for name, value in expected.items():
        if table[name] != value:
            raise ValueError(
                f"attempt table field mismatch: {attempt['attempt_id']}/{name}")
    sample_counts = {record["measurement"]["sample_count"]
                     for record in records.values()}
    warmup_counts = {record["measurement"]["warmup_iterations"]
                     for record in records.values()}
    modes = {record["measurement"]["mode"] for record in records.values()}
    if (len(sample_counts) != 1 or table["sample_count_per_repetition"]
            != next(iter(sample_counts))
            or len(warmup_counts) != 1 or table["warmup_iterations"]
            != next(iter(warmup_counts))
            or len(modes) != 1 or table["method"] != next(iter(modes))):
        raise ValueError(f"attempt table sampling metadata mismatch: {attempt['attempt_id']}")
    if family == "vector":
        if attempt["cohort"] == "new":
            minimum_distinct_input_mib = VECTOR_RECAPTURE_PROTOCOL[
                "minimum_distinct_input_mib"]
        elif attempt["cohort"] == "existing_strict":
            minimum_distinct_input_mib = HISTORICAL_VECTOR_SELECTION_PROTOCOL[
                "minimum_distinct_input_mib"]
        else:
            raise ValueError("unknown Event attempt cohort")
        expected_metadata = {
            "convergence_rule": None,
            "differential_estimator": "(T_high-T_low)/(N_high-N_low)",
            "buffer_policy": "unique_buffer_per_selected_replay_call",
            "buffer_count": None, "cache_eviction_mib": 256,
            "minimum_distinct_input_mib": minimum_distinct_input_mib,
        }
    elif family in {"flash_attn", "gemm", "fused_moe"}:
        expected_metadata = {
            "convergence_rule": (
                "abs(delta) <= max(500 ns, 5% of larger median)"),
            "differential_estimator": None, "buffer_policy": "fixed",
            "buffer_count": 1, "cache_eviction_mib": 2048,
            "minimum_distinct_input_mib": 64,
        }
    else:
        raise ValueError(f"unknown Event family: {family}")
    for name, value in expected_metadata.items():
        if table[name] != value:
            raise ValueError(
                f"attempt table metadata mismatch: {attempt['attempt_id']}/{name}")
    for name in ("unsharded_case_count", "shard_count", "shard_index"):
        if isinstance(table[name], bool) or not isinstance(table[name], int):
            raise TypeError(f"attempt table {name} must be an integer")
    if (table["unsharded_case_count"] < len(attempt["planned_case_ids"])
            or table["shard_count"] <= 0
            or not 0 <= table["shard_index"] < table["shard_count"]):
        raise ValueError(f"attempt table shard metadata is invalid: {attempt['attempt_id']}")
    rows = table["rows"]
    if not isinstance(rows, list) or len(rows) != len(records):
        raise ValueError(f"attempt table rows are incomplete: {attempt['attempt_id']}")
    row_index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("attempt table row must be an object")
        _exact_keys(row, TABLE_ROW_KEYS, "attempt table row")
        case_id = row["case_id"]
        if case_id in row_index:
            raise ValueError(f"attempt table repeats case: {case_id}")
        row_index[case_id] = row
    if set(row_index) != set(records):
        raise ValueError(f"attempt table/raw case mismatch: {attempt['attempt_id']}")
    for case_id, record in records.items():
        expected_row = _expected_table_row(record, canonical[case_id])
        if row_index[case_id] != expected_row:
            raise ValueError(
                f"attempt table row disagrees with raw case: "
                f"{attempt['attempt_id']}/{case_id}")
    return {
        "path": str(table_path), "sha256": file_sha256(table_path),
        "case_count": len(rows),
    }


def _validate_runtime_contract(
        campaign: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]]) -> None:
    contract = campaign["runtime_contract"]
    if not isinstance(contract, dict) or set(contract) != set(sources):
        raise ValueError("runtime contract must cover every source exactly")
    for source_id, family_contracts in contract.items():
        if (not isinstance(family_contracts, dict) or not family_contracts
                or not set(family_contracts).issubset(FAMILIES)):
            raise ValueError(f"runtime contract families are invalid: {source_id}")
        for family, environment in family_contracts.items():
            if not isinstance(environment, dict) or not environment:
                raise ValueError(
                    f"runtime environment must be non-empty: {source_id}/{family}")
            required = {
                "torch_version": "2.4.0+metax3.7.2.0",
                "torch_cuda_version_field": "11.6",
                "device_index": 0,
                "device_name": "MetaX C500",
            }
            for name, value in required.items():
                if environment[name] != value:
                    raise ValueError(
                        f"runtime identity is not frozen: {source_id}/{family}/{name}")
            if sources[source_id]["role"] == "new_capture" and environment != (
                    EXPECTED_NEW_ENVIRONMENTS[family]):
                raise ValueError(
                    f"new capture environment is not frozen: {source_id}/{family}")


def _validate_attempt_header(
        attempt: Mapping[str, Any], campaign_root: Path,
        sources: Mapping[str, Mapping[str, Any]]) -> None:
    _exact_keys(attempt, ATTEMPT_KEYS, "campaign attempt")
    attempt_id = attempt["attempt_id"]
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("attempt ID must be non-empty")
    if attempt["cohort"] not in {"new", "existing_strict"}:
        raise ValueError(f"invalid attempt cohort: {attempt_id}")
    if attempt["cohort"] == "new":
        require_positive_integer(attempt["ordinal"], f"attempt ordinal {attempt_id}")
    elif attempt["ordinal"] is not None:
        raise ValueError(f"legacy attempt ordinal must be null: {attempt_id}")
    source_id = attempt["source_id"]
    if source_id not in sources:
        raise ValueError(f"attempt references unknown source: {attempt_id}")
    if attempt["cohort"] == "new":
        expected_role = "new_capture"
    elif attempt["cohort"] == "existing_strict":
        expected_role = "legacy_frozen"
    else:
        raise ValueError(f"invalid attempt cohort: {attempt_id}")
    if sources[source_id]["role"] != expected_role:
        raise ValueError(f"attempt/source role mismatch: {attempt_id}")
    family = attempt["family"]
    if family not in FAMILIES:
        raise ValueError(f"attempt family is invalid: {attempt_id}")
    device = attempt["physical_device"]
    if (attempt["cohort"] == "new" and (
            isinstance(device, bool) or not isinstance(device, int)
            or not 0 <= device <= 3)):
        raise ValueError(f"attempt physical device is invalid: {attempt_id}")
    if attempt["cohort"] == "existing_strict" and device is not None:
        raise ValueError(f"legacy attempt device must be null: {attempt_id}")
    root = Path(attempt["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"attempt root is absent: {root}")
    if attempt["cohort"] == "new" and root.parent != campaign_root:
        raise ValueError(f"new attempt root is outside campaign root: {attempt_id}")
    if attempt["cohort"] == "new":
        start = require_positive_integer(
            attempt["start_wall_time_ns"], f"attempt start {attempt_id}")
        end = require_positive_integer(
            attempt["end_wall_time_ns"], f"attempt end {attempt_id}")
        if start >= end:
            raise ValueError(f"attempt time interval is invalid: {attempt_id}")
    elif (attempt["start_wall_time_ns"] is not None
          or attempt["end_wall_time_ns"] is not None):
        raise ValueError(f"legacy attempt times must be null: {attempt_id}")
    planned = attempt["planned_case_ids"]
    case_attempts = attempt["case_attempts"]
    if (not isinstance(planned, list) or not planned
            or len(planned) != len(set(planned))
            or any(not isinstance(case_id, str) or not case_id
                   for case_id in planned)
            or not isinstance(case_attempts, list) or not case_attempts):
        raise ValueError(f"attempt case plan is invalid: {attempt_id}")
    terminal = attempt["terminal_status"]
    if terminal not in {"completed", "failed", "interrupted"}:
        raise ValueError(f"attempt terminal status is invalid: {attempt_id}")
    error = attempt["error"]
    if ((terminal == "completed" and error is not None)
            or (terminal != "completed"
                and (not isinstance(error, str) or not error))):
        raise ValueError(f"attempt terminal error is invalid: {attempt_id}")
    if attempt["superseded"] is not False:
        raise ValueError(
            f"immutable attempt must not embed derived supersession: "
            f"{attempt_id}")
    if attempt["cohort"] == "existing_strict" and (
            attempt["ordinal"] is not None
            or attempt["command_argv"] is not None
            or attempt["pid"] is not None
            or attempt["process_group_id"] is not None
            or attempt["preflight_snapshot_id"] is not None
            or attempt["during_snapshot_ids"] != []
            or attempt["postflight_snapshot_id"] is not None
            or attempt["process_isolation"] is not None
            or attempt["wrapper"] is not None
            or attempt["execution_module"] is not None
            or attempt["launcher_pid"] is not None
            or attempt["launcher_process_group_id"] is not None
            or attempt["launcher_command_argv"] is not None
            or attempt["host_uid"] is not None
            or attempt["device_gid"] is not None):
        raise ValueError(
            f"legacy attempt must not invent process snapshots: {attempt_id}")
    if (attempt["cohort"] == "existing_strict"
            and (terminal != "completed"
                 or not isinstance(attempt["root_inventory"], dict))):
        raise ValueError(f"legacy attempt inventory/status is invalid: {attempt_id}")
    if attempt["cohort"] == "new" and not isinstance(
            attempt["process_isolation"], dict):
        raise ValueError(f"new attempt lacks process isolation: {attempt_id}")
    if attempt["cohort"] == "new":
        if attempt["root_inventory"] is not None:
            raise ValueError(f"new attempt must not use legacy inventory: {attempt_id}")
        wrapper_path, _ = _path_record(
            attempt["wrapper"], f"attempt wrapper {attempt_id}")
        module_path, _ = _path_record(
            attempt["execution_module"],
            f"attempt execution module {attempt_id}")
        valid_wrapper_suffixes = (
            "/scripts/kernel_dataset/metax_c500/"
            "run_event_campaign_attempt.py",
            "/scripts/kernel_dataset/metax_c500/"
            "run_event_campaign_attempt_runner.py",
        )
        if not wrapper_path.as_posix().endswith(valid_wrapper_suffixes):
            raise ValueError(f"attempt wrapper path is invalid: {attempt_id}")
        if not module_path.as_posix().endswith(
                "/dataset_pipeline/kernel_dataset/"
                "metax_c500_event_campaign.py"):
            raise ValueError(
                f"attempt execution module path is invalid: {attempt_id}")
        if (isinstance(attempt["host_uid"], bool)
                or not isinstance(attempt["host_uid"], int)
                or attempt["host_uid"] < 0
                or isinstance(attempt["device_gid"], bool)
                or not isinstance(attempt["device_gid"], int)
                or attempt["device_gid"] < 0):
            raise ValueError(f"attempt host identity is invalid: {attempt_id}")
        launcher = attempt["launcher_command_argv"]
        if (not isinstance(launcher, list) or len(launcher) < 10
                or any(not isinstance(value, str) or not value
                       for value in launcher)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", launcher[8]) is None):
            raise ValueError(f"attempt launcher command is invalid: {attempt_id}")
        # The configured target is retained in the content-bound launcher argv;
        # process-isolation checks match the complete command to host snapshots.
        container = launcher[8]
        expected_prefix = [
            "exec", "--env", f"CUDA_VISIBLE_DEVICES={device}",
            "--env", f"MACA_VISIBLE_DEVICES={device}",
            "--user", f"{attempt['host_uid']}:{attempt['device_gid']}",
            container,
        ]
        if (Path(launcher[0]).name != "docker"
                or launcher[1:9] != expected_prefix):
            raise ValueError(f"attempt launcher command is invalid: {attempt_id}")
        inner_identity = (
            attempt["command_argv"], attempt["pid"],
            attempt["process_group_id"])
        if all(value is None for value in inner_identity):
            if terminal == "completed":
                raise ValueError(
                    f"completed attempt lacks runner identity: {attempt_id}")
        elif any(value is None for value in inner_identity):
            raise ValueError(f"attempt runner identity is partial: {attempt_id}")
        else:
            argv = attempt["command_argv"]
            if (not isinstance(argv, list) or not argv
                    or any(not isinstance(value, str) for value in argv)):
                raise ValueError(f"attempt command argv is invalid: {attempt_id}")
            command = "\0".join(argv).lower()
            if any(marker in command for marker in FORBIDDEN_PROFILER_MARKERS):
                raise ValueError(
                    f"attempt command invokes a profiler: {attempt_id}")
            expected_runner = str(
                Path(sources[source_id]["snapshot_root"]) /
                "scripts/kernel_dataset/metax_c500/kernel_golden_graph_runner.py"
            ) if sources[source_id]["snapshot_root"] is not None else (
                "scripts/kernel_dataset/metax_c500/kernel_golden_graph_runner.py")
            if not any(value == expected_runner or value.endswith(
                    "/scripts/kernel_dataset/metax_c500/"
                    "kernel_golden_graph_runner.py") for value in argv):
                raise ValueError(
                    f"attempt command does not use graph runner: {attempt_id}")
            require_positive_integer(attempt["pid"], f"attempt PID {attempt_id}")
            require_positive_integer(
                attempt["process_group_id"], f"attempt PGID {attempt_id}")
        launcher_identity = (
            attempt["launcher_pid"], attempt["launcher_process_group_id"])
        if all(value is None for value in launcher_identity):
            if terminal == "completed":
                raise ValueError(
                    f"completed attempt lacks launcher identity: {attempt_id}")
        elif any(value is None for value in launcher_identity):
            raise ValueError(f"attempt launcher identity is partial: {attempt_id}")
        else:
            require_positive_integer(attempt["launcher_pid"], "launcher PID")
            require_positive_integer(
                attempt["launcher_process_group_id"], "launcher PGID")


def _validate_case_attempt_record(
        *, attempt: Mapping[str, Any], case_attempt: Mapping[str, Any],
        source: Mapping[str, Any], canonical: Mapping[str, Any],
        runtime_environment: Mapping[str, Any],
        ) -> tuple[Mapping[str, Any] | None, Path | None, set[Path]]:
    attempt_id = attempt["attempt_id"]
    if not isinstance(case_attempt, dict):
        raise TypeError(f"case attempt must be an object: {attempt_id}")
    _exact_keys(case_attempt, CASE_ATTEMPT_KEYS, "case attempt")
    case_id = case_attempt["case_id"]
    if case_id not in attempt["planned_case_ids"] or case_id not in canonical:
        raise ValueError(f"case attempt is outside its plan/canonical set: {case_id}")
    status = case_attempt["status"]
    if status not in {"measured", "failed"}:
        raise ValueError(f"case attempt status is invalid: {attempt_id}/{case_id}")
    if case_attempt["superseded_by_attempt_id"] is not None:
        raise ValueError(
            f"immutable case attempt must not embed derived supersession: "
            f"{attempt_id}/{case_id}")
    protected = set()
    if status == "failed":
        if (case_attempt["artifact_path"] is not None
                or case_attempt["artifact_sha256"] is not None
                or not isinstance(case_attempt["error_path"], str)
                or not isinstance(case_attempt["error_sha256"], str)):
            raise ValueError(f"failed case evidence is incomplete: {attempt_id}/{case_id}")
        error_path = Path(case_attempt["error_path"]).resolve()
        if (file_sha256(error_path) != case_attempt["error_sha256"]
                or not error_path.is_relative_to(Path(attempt["root"]).resolve())):
            raise ValueError(f"failed case error evidence mismatch: {attempt_id}/{case_id}")
        protected.add(error_path)
        return None, None, protected
    if (case_attempt["error_path"] is not None
            or case_attempt["error_sha256"] is not None
            or not isinstance(case_attempt["artifact_path"], str)
            or not isinstance(case_attempt["artifact_sha256"], str)):
        raise ValueError(f"measured case evidence is malformed: {attempt_id}/{case_id}")
    artifact_path = Path(case_attempt["artifact_path"]).resolve()
    expected_path = Path(attempt["root"]).resolve() / "cases" / f"{case_id}.json"
    if artifact_path != expected_path or file_sha256(artifact_path) != case_attempt[
            "artifact_sha256"]:
        raise ValueError(f"measured case artifact mismatch: {attempt_id}/{case_id}")
    record = load_json_mapping(artifact_path, "Event case artifact")
    if record["family"] != attempt["family"] or record["case_id"] != case_id:
        raise ValueError(f"case attempt identity mismatch: {attempt_id}/{case_id}")
    result = _validate_case(
        record=record, path=artifact_path, canonical_entry=canonical[case_id],
        source=source, runtime_environment=runtime_environment,
        require_new_contract=attempt["cohort"] == "new")
    protected.add(artifact_path)
    return result, artifact_path, protected


def _validate_output_path(output_path: Path, protected: set[Path]) -> Path:
    output = output_path.resolve()
    root = OUTPUT_ROOT.resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError(
            "audit output must be a file under experiment_logs/kernel_golden")
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    for path in protected:
        resolved = path.resolve()
        if output == resolved or (resolved.is_dir() and output.is_relative_to(resolved)):
            raise ValueError(f"audit output overlaps protected input: {resolved}")
    return output


def audit_full_event_campaign(
        campaign_manifest_path: Path, output_path: Path) -> Mapping[str, Any]:
    campaign_path = campaign_manifest_path.resolve()
    campaign = load_json_mapping(campaign_path, "full Event campaign manifest")
    _exact_keys(campaign, CAMPAIGN_KEYS, "full Event campaign manifest")
    if (campaign["schema_version"] != 1
            or campaign["artifact_kind"] != "metax_c500_full_event_campaign"
            or campaign["hardware_id"] != "metax_c500"
            or campaign["selection_rule"] != SELECTION_RULE):
        raise ValueError("full Event campaign identity/policy is invalid")
    selection_path, selection_sha = _path_record(
        campaign["selection_manifest"], "campaign selection manifest")
    selection, canonical_datasets, legacy_evidence, protected = (
        _validate_frozen_selection(
        selection_path)
    )
    protected.update({campaign_path, selection_path})
    if selection_sha != file_sha256(selection_path):
        raise AssertionError("selection digest changed during audit")

    sources_payload = campaign["sources"]
    if not isinstance(sources_payload, dict) or not sources_payload:
        raise ValueError("campaign sources must be a non-empty object")
    sources = {}
    for source_id, source_payload in sources_payload.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("campaign source ID is invalid")
        source, source_paths = _validate_source(source_id, source_payload)
        sources[source_id] = source
        protected.update(source_paths)
    if sum(source["role"] == "new_capture" for source in sources.values()) != 1:
        raise ValueError("campaign must contain exactly one new-capture source")
    _validate_runtime_contract(campaign, sources)

    campaign_root = Path(campaign["campaign_root"]).resolve()
    if not campaign_root.is_dir():
        raise FileNotFoundError(f"campaign root is absent: {campaign_root}")
    protected.add(campaign_root)
    attempt_references = campaign["attempts"]
    if not isinstance(attempt_references, list) or not attempt_references:
        raise ValueError("campaign attempts must be a non-empty list")
    attempts = []
    for reference in attempt_references:
        attempt_path, _ = _path_record(reference, "campaign attempt")
        attempt = load_json_mapping(attempt_path, "campaign attempt")
        root_attempt_path = (
            Path(attempt["root"]).resolve() / "attempt.json").resolve()
        if (attempt["cohort"] == "new" and attempt_path != root_attempt_path):
            raise ValueError("campaign attempt path/root mismatch")
        if (attempt["cohort"] == "existing_strict"
                and attempt_path.is_relative_to(Path(attempt["root"]).resolve())):
            raise ValueError("legacy attempt descriptor must be external")
        protected.add(attempt_path)
        attempts.append(attempt)
    attempt_index = {}
    roots = set()
    new_roots = set()
    case_history: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    attempt_reports = []
    isolation_summaries = []
    used_snapshot_ids = set()
    used_snapshot_paths = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise TypeError("campaign attempt must be an object")
        _validate_attempt_header(attempt, campaign_root, sources)
        attempt_id = attempt["attempt_id"]
        ordinal = attempt["ordinal"]
        root = Path(attempt["root"]).resolve()
        if (attempt_id in attempt_index
                or (attempt["cohort"] == "new" and root in roots)):
            raise ValueError("attempt IDs and new roots must be unique")
        attempt_index[attempt_id] = attempt
        roots.add(root)
        protected.add(root)
        if attempt["cohort"] == "new":
            new_roots.add(root)
            snapshots, isolation_paths, isolation_summary = (
                _validate_process_isolation(attempt["process_isolation"]))
            snapshot_paths = {
                Path(snapshot["path"]).resolve()
                for snapshot in snapshots.values()}
            if (used_snapshot_ids & set(snapshots)
                    or used_snapshot_paths & snapshot_paths):
                raise ValueError("process-isolation evidence is reused")
            used_snapshot_ids.update(snapshots)
            used_snapshot_paths.update(snapshot_paths)
            protected.update(isolation_paths)
            isolation_summaries.append(isolation_summary)
            _validate_attempt_isolation(attempt, snapshots)
        elif attempt["cohort"] == "existing_strict":
            inventory_path, inventory_paths = _validate_legacy_root_inventory(
                attempt["root_inventory"], root)
            protected.update(inventory_paths)
            protected.add(inventory_path)
        else:
            raise ValueError(f"invalid attempt cohort: {attempt_id}")
        profiler_artifacts = sorted(
            str(path.relative_to(root)) for path in root.rglob("*")
            if any(marker in path.name.lower()
                   for marker in FORBIDDEN_PROFILER_MARKERS))
        if attempt["cohort"] == "new" and profiler_artifacts:
            raise ValueError(
                f"attempt root contains profiler artifacts: "
                f"{attempt_id}/{profiler_artifacts}")
        family = attempt["family"]
        canonical = canonical_datasets[family]
        source = sources[attempt["source_id"]]
        runtime = campaign["runtime_contract"][attempt["source_id"]][family]
        case_attempt_ids = set()
        measured_records = {}
        measured_results = {}
        attempt_protected = set()
        for case_attempt in attempt["case_attempts"]:
            case_id = case_attempt["case_id"]
            if case_id in case_attempt_ids:
                raise ValueError(f"attempt repeats case: {attempt_id}/{case_id}")
            case_attempt_ids.add(case_id)
            result, artifact_path, case_paths = _validate_case_attempt_record(
                attempt=attempt, case_attempt=case_attempt, source=source,
                canonical=canonical, runtime_environment=runtime)
            attempt_protected.update(case_paths)
            history_record = {
                "attempt_id": attempt_id, "ordinal": ordinal,
                "case_attempt_start_wall_time_ns":
                    attempt["start_wall_time_ns"],
                "terminal_status": attempt["terminal_status"],
                "cohort": attempt["cohort"], "case_attempt": case_attempt,
                "result": result,
            }
            identity = (family, case_id)
            if identity not in case_history:
                case_history[identity] = []
            case_history[identity].append(history_record)
            if artifact_path is not None:
                measured_records[case_id] = load_json_mapping(
                    artifact_path, "Event case artifact")
                measured_results[case_id] = result
        protected.update(attempt_protected)
        actual_case_paths = sorted((root / "cases").glob("*.json"))
        declared_artifacts = sorted(
            Path(case_attempt["artifact_path"]).resolve()
            for case_attempt in attempt["case_attempts"]
            if case_attempt["artifact_path"] is not None)
        if (attempt["cohort"] == "new"
                and actual_case_paths != declared_artifacts):
            raise ValueError(f"attempt omits or invents raw cases: {attempt_id}")
        if (attempt["cohort"] == "existing_strict"
                and not set(declared_artifacts).issubset(actual_case_paths)):
            raise ValueError(
                f"legacy attempt declares raw outside root inventory: "
                f"{attempt_id}")
        if attempt["terminal_status"] == "completed" and (
                case_attempt_ids != set(attempt["planned_case_ids"])
                or any(case_attempt["status"] != "measured"
                       for case_attempt in attempt["case_attempts"])):
            raise ValueError(f"completed attempt is incomplete: {attempt_id}")
        table = _validate_attempt_table(
            attempt=attempt, records=measured_records, canonical=canonical,
            source=source,
            canonical_sha256=selection["provenance"]["canonical_datasets"][
                family]["sha256"])
        if table is not None:
            protected.add(Path(table["path"]))
        attempt_reports.append({
            "attempt_id": attempt_id, "ordinal": ordinal,
            "cohort": attempt["cohort"], "family": family,
            "root": str(root), "terminal_status": attempt["terminal_status"],
            "superseded": attempt["superseded"],
            "planned_case_count": len(attempt["planned_case_ids"]),
            "started_case_count": len(case_attempt_ids),
            "measured_case_count": len(measured_records),
            "strict_case_count": sum(
                result["stable"] for result in measured_results.values()),
            "table": table,
        })

    actual_new_dirs = {path.resolve() for path in campaign_root.iterdir()
                       if path.is_dir()}
    if actual_new_dirs != new_roots:
        raise ValueError(
            "campaign attempt directory inventory mismatch: "
            f"undeclared={sorted(str(path) for path in actual_new_dirs-new_roots)}, "
            f"missing={sorted(str(path) for path in new_roots-actual_new_dirs)}")
    canonical_identities = {
        (family, case_id) for family in FAMILIES
        for case_id in canonical_datasets[family]}
    for device in range(4):
        device_attempts = [
            attempt for attempt in attempts if attempt["cohort"] == "new"
            and attempt["physical_device"] == device]
        device_attempts.sort(key=lambda value: value["start_wall_time_ns"])
        if any(device_attempts[index]["end_wall_time_ns"] >
               device_attempts[index + 1]["start_wall_time_ns"]
               for index in range(len(device_attempts) - 1)):
            raise ValueError(f"new attempts overlap on physical device {device}")

    selected_new = {
        (family, case_id) for family in FAMILIES
        for case_id in selection["family_case_ids"][family]}
    selected_rows = []
    selected_attempt_ids = set()
    for identity in sorted(canonical_identities):
        family, case_id = identity
        if identity not in case_history:
            raise ValueError(f"no raw attempt exists for canonical case: {identity}")
        history = sorted(
            case_history[identity],
            key=lambda row: row["case_attempt_start_wall_time_ns"])
        if (len(history) > 1 and history[-1][
                "case_attempt_start_wall_time_ns"] == history[-2][
                "case_attempt_start_wall_time_ns"]):
            raise ValueError(f"attempt start time is ambiguous: {identity}")
        latest = history[-1]
        selected_attempt_id = latest["attempt_id"]
        if (latest["terminal_status"] != "completed"
                or latest["case_attempt"]["status"] != "measured"
                or latest["result"] is None or not latest["result"]["stable"]):
            raise ValueError(f"latest attempt is not strict passing: {identity}")
        if identity in selected_new:
            expected_cohort = "new"
        else:
            expected_cohort = "existing_strict"
        if latest["cohort"] != expected_cohort:
            raise ValueError(f"selected attempt cohort mismatch: {identity}")
        selected_attempt_ids.add(selected_attempt_id)
        canonical_timing = canonical_datasets[family][case_id]["op_info"][
            "evaluation"]["timing"]
        if expected_cohort == "existing_strict" and (
                canonical_timing["granularity"] != "op"
                or canonical_timing["source"] != "device_event_op_median"
                or not math.isclose(
                    float(canonical_timing["total_duration_ns"]),
                    float(latest["result"]["median_ns"]),
                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"existing strict timing does not replay: {identity}")
        if expected_cohort == "existing_strict":
            frozen = legacy_evidence[identity]
            expected_tree_sha_by_family = {
                "vector": EXPECTED_VECTOR_LEGACY_TREE_SHA256,
                "flash_attn": EXPECTED_FLASH_LEGACY_AUDIT_SHA256,
                "gemm": EXPECTED_REMAINING_LEGACY_AUDIT_SHA256,
                "fused_moe": EXPECTED_REMAINING_LEGACY_AUDIT_SHA256,
            }
            expected_tree_sha = expected_tree_sha_by_family[family]
            timing_provenance = canonical_timing["provenance"]
            if (latest["result"]["path"] != frozen["path"]
                    or latest["result"]["sha256"] != frozen["sha256"]
                    or timing_provenance["source_artifact_sha256"] !=
                    frozen["sha256"]
                    or timing_provenance["source_artifact_tree_sha256"] !=
                    expected_tree_sha
                    or timing_provenance["profiler"] != "none"
                    or timing_provenance["scope"] != "end_to_end_case"):
                raise ValueError(
                    f"existing strict raw is not frozen evidence: {identity}")
        selected_rows.append({
            **latest["result"], "attempt_id": selected_attempt_id,
            "cohort": expected_cohort,
        })
    for report in attempt_reports:
        report["superseded"] = (
            report["attempt_id"] not in selected_attempt_ids)

    output = _validate_output_path(output_path, protected)
    coverage = []
    for family in FAMILIES:
        rows = [row for row in selected_rows if row["family"] == family]
        new_count = sum(row["cohort"] == "new" for row in rows)
        existing_count = sum(row["cohort"] == "existing_strict" for row in rows)
        if (len(rows) != FAMILY_COUNTS[family]
                or new_count != SELECTED_FAMILY_COUNTS[family]):
            raise AssertionError(f"final strict coverage mismatch: {family}")
        coverage.append({
            "family": family, "canonical_case_count": FAMILY_COUNTS[family],
            "existing_strict_case_count": existing_count,
            "new_strict_case_count": new_count,
            "strict_event_case_count": len(rows),
        })
    new_event_paths = [
        Path(row["path"]) for row in selected_rows if row["cohort"] == "new"]
    if (len(new_event_paths) != sum(SELECTED_FAMILY_COUNTS.values())
            or any(not path.is_relative_to(campaign_root)
                   for path in new_event_paths)):
        raise AssertionError("accepted new Event paths are outside campaign root")
    return {
        "schema_version": 2,
        "artifact_kind": "metax_c500_full_op_event_capture_audit",
        "status": "accepted",
        "hardware_id": "metax_c500",
        "output": str(output),
        "campaign_manifest": str(campaign_path),
        "campaign_manifest_sha256": file_sha256(campaign_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": selection_sha,
        "selection_rule": SELECTION_RULE,
        "process_isolation": isolation_summaries,
        "existing_strict_case_count": 494,
        "new_strict_case_count": 493,
        "new_event_artifact_tree_sha256": _sha256sum_manifest(
            new_event_paths, campaign_root),
        "strict_event_case_count": 987,
        "coverage": coverage,
        "sources": sources,
        "attempts": attempt_reports,
        "selected_cases": selected_rows,
    }


def write_audit_report(output_path: Path, payload: Mapping[str, Any]) -> None:
    resolved = output_path.resolve()
    if payload["output"] != str(resolved):
        raise ValueError("audit report output path does not match validated path")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(resolved)
