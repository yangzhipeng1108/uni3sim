"""Strict validation for versioned compact ShareGPT Serving cases.

The schema owns only the compact on-disk contract. It deliberately does not import the Serving simulator, profiler readers, or evaluation code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS

from .compressed_artifact import (
    COMPRESSION_ALGORITHM,
    read_compressed_canonical_bytes,
)

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
    compact_sequence as _sequence,
    require_exact_keys,
    require_nonempty_string,
)


SCHEMA_VERSION = 2
SCHEDULER_CAPTURE_SCHEMA_VERSION = 3
RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION = 4
SCHEDULER_PIPELINE_DEPTH_SCHEMA_VERSION = 5
PD_DISAGGREGATED_SCHEMA_VERSION = 6
OBSERVER_QUALITY_SCHEMA_VERSION = 7
LATEST_SCHEMA_VERSION = SCHEDULER_PIPELINE_DEPTH_SCHEMA_VERSION
SUPPORTED_SCHEMA_VERSIONS = (
    SCHEMA_VERSION,
    SCHEDULER_CAPTURE_SCHEMA_VERSION,
    RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
    SCHEDULER_PIPELINE_DEPTH_SCHEMA_VERSION,
    OBSERVER_QUALITY_SCHEMA_VERSION,
)
_SCHEDULER_CAPTURE_SCHEMA_VERSIONS = (
    SCHEDULER_CAPTURE_SCHEMA_VERSION,
    RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
    SCHEDULER_PIPELINE_DEPTH_SCHEMA_VERSION,
    OBSERVER_QUALITY_SCHEMA_VERSION,
)
_DEPTH_SCHEMA_VERSIONS = (
    SCHEDULER_PIPELINE_DEPTH_SCHEMA_VERSION,
    OBSERVER_QUALITY_SCHEMA_VERSION,
)
CANONICAL_REQUEST_COLUMNS = (
    "Query Time Stamp [ms]",
    "In Token [tokens]",
    "Out Token [tokens]",
    "Cached hit [tokens]",
)
PHASES = ("prefill", "decode", "mixed")
ALIGNMENT_NAMES = ("alignment_steps",)
ALIGNMENT_FILENAME = "alignment_steps.json.zst"
SCHEDULER_STEPS_FILENAME = "scheduler_steps.jsonl.zst"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_MODEL_ID = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_INTEGER_TEXT = re.compile(r"[+]?[0-9]+")
_FORBIDDEN_ALIGNMENT_TIME_TOKENS = {
    "clock", "duration", "elapsed", "latency", "time", "timestamp", "ts",
}
_ALIGNMENT_TIME_BOUNDARY_TOKENS = {"end", "finish", "start", "wall"}
_ALIGNMENT_TIME_UNIT_TOKENS = {"ms", "ns", "s", "us"}
_SCHEDULER_FIELD_PROVENANCE = {
    "step_id": "normalized",
    "pool_name": "normalized_constant",
    "requests[].request_id": "normalized",
    "requests[].phase": "derived",
    "requests[].input_tokens": "derived",
    "requests[].output_tokens": "derived",
    "requests[].computed_tokens": "derived",
    "requests[].recomputed_tokens": "unobserved_zero",
    "requests[].prefix_replay_tokens": "unobserved_zero",
    "requests[].resumed_from_preemption": "unobserved_false",
    "requests[].trace_prefix_hit_tokens": "unobserved_zero",
    "requests[].scheduler_prefix_hit_tokens": "unobserved_zero",
    "requests[].prefix_scheduler_delta_tokens": "derived",
    "requests[].raw_model_output_tokens": "unobserved_zero",
    "requests[].mtp_proposed_tokens": "unobserved_zero",
    "requests[].mtp_verified_tokens": "unobserved_zero",
    "requests[].mtp_accepted_tokens": "unobserved_zero",
    "requests[].mtp_rejected_tokens": "derived",
    "requests[].mtp_proposer_tokens": "unobserved_zero",
    "requests[].mtp_next_proposed_tokens": "unobserved_zero",
    "requests[].mtp_proposer_mode": "normalized_constant",
    "requests[].mtp_acceptance_source_kind": "normalized_constant",
    "requests[].mtp_acceptance_source_id": "normalized_constant",
    "finished_request_ids": "observed",
    "preempted_request_ids": "unobserved_empty",
    "attention_kv_layout": "unobserved_null",
}
_SCHEDULE_FIELDS = (
    "step_id", "pool_name", "requests", "finished_request_ids",
    "preempted_request_ids", "attention_kv_layout",
)
_SCHEDULED_REQUEST_FIELDS = (
    "request_id", "phase", "input_tokens", "output_tokens", "computed_tokens",
    "recomputed_tokens", "prefix_replay_tokens", "resumed_from_preemption",
    "trace_prefix_hit_tokens", "scheduler_prefix_hit_tokens",
    "prefix_scheduler_delta_tokens", "raw_model_output_tokens",
    "mtp_proposed_tokens", "mtp_verified_tokens", "mtp_accepted_tokens",
    "mtp_rejected_tokens", "mtp_proposer_tokens", "mtp_next_proposed_tokens",
    "mtp_proposer_mode", "mtp_acceptance_source_kind",
    "mtp_acceptance_source_id",
)


@dataclass(frozen=True)
class CanonicalRequestRow:
    """One physical request row used by Serving Forward."""

    arrival_time_ms: float
    input_tokens: int
    output_tokens: int
    cached_hit_tokens: int

    def __post_init__(self) -> None:
        _finite_non_negative(self.arrival_time_ms, "request arrival_time_ms")
        _integer(self.input_tokens, "request input_tokens", minimum=1)
        _integer(self.output_tokens, "request output_tokens", minimum=0)
        _integer(self.cached_hit_tokens, "request cached_hit_tokens", minimum=0)
        if self.cached_hit_tokens > self.input_tokens:
            raise ValueError("request cached_hit_tokens cannot exceed input_tokens")


@dataclass(frozen=True)
class AlignmentStepSample:
    """One complete, representative Scheduler step for a phase."""

    source_step_id: str
    source_execution_order: int
    rank_ids: tuple[int, ...]
    scheduler: Mapping[str, object]
    operators: tuple[Mapping[str, object], ...]
    launches: tuple[Mapping[str, object], ...]
    association: str
    coverage: str
    provenance: Mapping[str, object]
    sequence_sha256: Mapping[str, object]


@dataclass(frozen=True)
class CompactCase:
    """A fully validated compact Serving case."""

    case_dir: Path
    manifest: Mapping[str, object]
    requests: tuple[CanonicalRequestRow, ...]
    metrics: Mapping[str, object] | None
    scheduler_steps: tuple[Mapping[str, object], ...] | None
    alignment_steps: Mapping[str, AlignmentStepSample | None]
    operator_signatures: Mapping[str, Mapping[str, object]]
    launch_signatures: Mapping[str, Mapping[str, object]]
    warmup_requests: tuple[CanonicalRequestRow, ...] = ()

    @property
    def scheduler_pipeline_depth(self) -> int:
        """Return capacity normalized by the canonical dataset validator."""

        return scheduler_pipeline_depth_from_compact_case(self)


def scheduler_pipeline_depth_from_compact_case(case: CompactCase) -> int:
    """Resolve the configured runtime capacity from a strict compact case."""

    if not isinstance(case, CompactCase):
        raise TypeError("scheduler pipeline depth requires a CompactCase")
    manifest = _mapping(case.manifest, "compact case manifest")
    if manifest["schema_version"] not in _DEPTH_SCHEMA_VERSIONS:
        raise ValueError(
            "runtime.scheduler_pipeline_depth is required; this compact "
            "case predates configured Scheduler depth evidence")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    if runtime["deployment"] in ("pd_disagg", "pd_af_disagg"):
        raise ValueError(
            "PD compact cases require per-pool Scheduler depth resolution")
    depth = runtime["scheduler_pipeline_depth"]
    _integer(depth, "runtime scheduler_pipeline_depth", minimum=1)
    if isinstance(depth, bool):
        raise TypeError("runtime scheduler_pipeline_depth must not be boolean")
    return depth


def scheduler_pipeline_depths_from_compact_case(
    case: CompactCase,
    pool_roles: Mapping[str, object],
) -> dict[str, int]:
    """Resolve every pool's configured Scheduler capacity without a fallback."""

    if not isinstance(case, CompactCase):
        raise TypeError("scheduler pipeline depths require a CompactCase")
    if not isinstance(pool_roles, Mapping) or not pool_roles:
        raise TypeError("scheduler pipeline pool roles must be a non-empty mapping")
    manifest = _mapping(case.manifest, "compact case manifest")
    if manifest["schema_version"] not in _DEPTH_SCHEMA_VERSIONS:
        raise ValueError(
            "runtime Scheduler pipeline depth evidence is required; this "
            "compact case predates configured depth metadata")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    deployment = runtime["deployment"]
    if deployment in ("pd_disagg", "pd_af_disagg"):
        instances = _mapping(runtime["instances"], "runtime instances")
        result = {}
        for pool_name, role in pool_roles.items():
            if not isinstance(pool_name, str) or not pool_name:
                raise ValueError("scheduler pool names must be non-empty strings")
            if role not in ("prefill", "decode"):
                raise ValueError(
                    f"PD scheduler pool {pool_name!r} has invalid role {role!r}")
            instance = _mapping(instances[role], f"runtime instance {role}")
            depth = instance["scheduler_pipeline_depth"]
            _integer(
                depth,
                f"runtime instance {role} scheduler_pipeline_depth",
                minimum=1,
            )
            if isinstance(depth, bool):
                raise TypeError(
                    f"runtime instance {role} scheduler_pipeline_depth must "
                    "not be boolean")
            result[pool_name] = depth
        return result
    depth = scheduler_pipeline_depth_from_compact_case(case)
    result = {}
    for pool_name in pool_roles:
        if not isinstance(pool_name, str) or not pool_name:
            raise ValueError("scheduler pool names must be non-empty strings")
        result[pool_name] = depth
    return result


def load_compact_case(case_dir: str | Path) -> CompactCase:
    """Load and fail loudly on one versioned compact ShareGPT case."""

    root = Path(case_dir)
    if not root.is_dir():
        raise ValueError(f"compact Serving case is not a directory: {root}")
    manifest = _read_json_object(root / "manifest.json")
    identity = _mapping(manifest["identity"], "manifest identity")
    if identity["source_kind"] == "synthetic":
        return _load_synthetic_case(root, manifest)
    if identity["source_kind"] == "sharegpt_pd_disaggregated":
        from .pd_compact import load_pd_disaggregated_case

        return load_pd_disaggregated_case(root)
    _validate_manifest(manifest)
    schema_version = manifest["schema_version"]
    requests = read_canonical_request_rows(root / "requests.csv")
    request_bytes = (root / "requests.csv").read_bytes()
    if request_bytes != canonical_requests_csv_bytes(requests):
        raise ValueError("requests.csv is valid but not in canonical schema-v2 serialization")
    request_entry = _mapping(manifest["requests"], "manifest requests")
    if request_entry["row_count"] != len(requests):
        raise ValueError("manifest request row_count disagrees with requests.csv")
    _verify_sha(root / "requests.csv", request_entry["sha256"], "requests.csv")
    _validate_cached_hit_rows(requests, request_entry)
    metrics_entry = _mapping(manifest["metrics"], "manifest metrics")
    source_kind = _mapping(manifest["identity"], "manifest identity")["source_kind"]
    metrics = _load_metrics(
        root,
        metrics_entry,
        request_entry["sha256"],
        len(requests),
        source_kind,
        schema_version,
    )
    if schema_version in _SCHEDULER_CAPTURE_SCHEMA_VERSIONS:
        _validate_same_run_bindings(manifest, metrics)
    _validate_metrics_raw_binding(metrics, requests, manifest)
    alignment_entry = _mapping(manifest["alignment_steps"], "manifest alignment_steps")
    alignment_value = _load_alignment(
        root, alignment_entry, manifest, len(requests), schema_version
    )
    scheduler_steps = (
        None
        if schema_version == SCHEMA_VERSION
        else _load_scheduler_steps(
            root,
            _mapping(manifest["scheduler_steps"], "manifest scheduler_steps"),
            manifest,
            requests,
        )
    )
    _validate_case_files(root, metrics_entry, schema_version)
    return CompactCase(
        case_dir=root,
        manifest=manifest,
        requests=requests,
        metrics=metrics,
        scheduler_steps=scheduler_steps,
        alignment_steps=alignment_value["steps"],
        operator_signatures=alignment_value["operator_signatures"],
        launch_signatures=alignment_value["launch_signatures"],
    )


def validate_compact_case(case_dir: str | Path) -> None:
    """Validate a compact case without returning it."""

    load_compact_case(case_dir)


def _load_synthetic_case(root: Path, manifest: Mapping[str, object]) -> CompactCase:
    """Validate input-only synthetic cases in the unified Serving schema.

    Unlike captured cases, generated requests have no measured metrics,
    alignment or Scheduler trace. Framework settings live in the experiment
    suite so multiple frameworks can consume the same immutable input.
    """

    _exact_keys(manifest, (
        "schema_version", "identity", "runtime", "requests", "warmup_requests",
        "runtime_memory_policies", "metrics", "alignment_steps", "scheduler_steps",
        "generation",
    ), "synthetic manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 5:
        raise ValueError("synthetic cases require unified Serving schema version 5")
    identity = _mapping(manifest["identity"], "synthetic identity")
    _exact_keys(identity, ("source_kind", "case_id", "hardware_id", "model_id"),
                "synthetic identity")
    if identity["source_kind"] != "synthetic":
        raise ValueError("synthetic source_kind mismatch")
    _nonempty_string(identity["case_id"], "synthetic case_id")
    _model_id(identity["model_id"], "synthetic model_id")
    if identity["hardware_id"] not in CANONICAL_HARDWARE_IDS:
        raise ValueError("synthetic hardware_id must be canonical")
    runtime = _mapping(manifest["runtime"], "synthetic runtime")
    _exact_keys(runtime, (
        "tensor_parallel_size", "pipeline_parallel_size",
        "expert_parallel_size", "client_concurrency", "requests_per_client",
        "warmup_requests_per_client", "arrival_semantics", "prefix_caching",
    ), "synthetic runtime")
    for field in ("tensor_parallel_size", "pipeline_parallel_size",
                  "expert_parallel_size", "client_concurrency", "requests_per_client",
                  "warmup_requests_per_client"):
        _integer(runtime[field], f"synthetic {field}", minimum=1)
    if runtime["arrival_semantics"] != "zero_think_time_closed_loop":
        raise ValueError("synthetic arrival semantics must be explicit closed loop")
    if runtime["prefix_caching"] is not False:
        raise ValueError("synthetic zero cache-hit inputs require disabled prefix caching")
    for field in ("metrics", "alignment_steps", "scheduler_steps"):
        entry = _mapping(manifest[field], field)
        _exact_keys(entry, ("status", "reason", "path", "sha256"), field)
        if (entry["status"] != "not_observed" or entry["path"] is not None
                or entry["sha256"] is not None):
            raise ValueError(f"synthetic {field} must declare not_observed")
        _nonempty_string(entry["reason"], f"synthetic {field} reason")
    generation = _mapping(manifest["generation"], "synthetic generation")
    _exact_keys(generation, ("algorithm", "seed", "input_tokens", "output_tokens",
                            "cached_hit_provenance"), "synthetic generation")
    if generation["algorithm"] != "numpy_randomstate_randint_inclusive":
        raise ValueError("unsupported synthetic generation algorithm")
    _integer(generation["seed"], "synthetic generation seed", minimum=0)
    bounds = {}
    for field in ("input_tokens", "output_tokens"):
        values = _sequence(generation[field], field)
        if len(values) != 2:
            raise ValueError("synthetic token bounds require two integers")
        for value in values:
            _integer(value, field, minimum=1)
        if values[0] > values[1]:
            raise ValueError("synthetic token bounds are reversed")
        bounds[field] = values
    if generation["cached_hit_provenance"] != "assumed_zero_prefix_cache_disabled":
        raise ValueError("synthetic cache hits must be declared assumed, not observed")
    rows_by_field = {}
    for field, filename, count_field in (
            ("requests", "requests.csv", "requests_per_client"),
            ("warmup_requests", "warmup_requests.csv", "warmup_requests_per_client")):
        entry = _mapping(manifest[field], field)
        _exact_keys(entry, ("path", "row_count", "sha256"), field)
        if entry["path"] != filename:
            raise ValueError(f"synthetic {field} path must be {filename}")
        _integer(entry["row_count"], f"{field} row_count", minimum=1)
        _sha256(entry["sha256"], f"{field} sha256")
        path = root / filename
        _verify_sha(path, entry["sha256"], filename)
        rows = read_canonical_request_rows(path)
        if path.read_bytes() != canonical_requests_csv_bytes(rows):
            raise ValueError(f"synthetic {filename} serialization is not canonical")
        if len(rows) != entry["row_count"] or len(rows) != (
                runtime["client_concurrency"] * runtime[count_field]):
            raise ValueError(f"synthetic {field} row_count disagrees with protocol")
        for row in rows:
            if row.arrival_time_ms != 0 or row.cached_hit_tokens != 0:
                raise ValueError("synthetic closed-loop rows require zero arrival/cache fields")
            for name in bounds:
                if not bounds[name][0] <= getattr(row, name) <= bounds[name][1]:
                    raise ValueError(f"synthetic {name} is outside declared bounds")
        rows_by_field[field] = rows
    policies = _mapping(manifest["runtime_memory_policies"], "synthetic memory policies")
    _exact_keys(policies, ("colocated", "pd_disagg"), "synthetic memory policies")
    expected = {"manifest.json", "requests.csv", "warmup_requests.csv"}
    for deployment, value in policies.items():
        policy = _mapping(value, "synthetic memory policy")
        _exact_keys(policy, ("path", "sha256"), "synthetic memory policy")
        filename = f"runtime_memory_policy_{deployment}.json"
        if policy["path"] != filename:
            raise ValueError("synthetic memory policy filename mismatch")
        _sha256(policy["sha256"], "synthetic memory policy sha256")
        _verify_sha(root / filename, policy["sha256"], filename)
        _read_json_object(root / filename)
        expected.add(filename)
    if {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} != expected:
        raise ValueError("synthetic case file set mismatch")
    return CompactCase(
        case_dir=root, manifest=manifest, requests=rows_by_field["requests"],
        warmup_requests=rows_by_field["warmup_requests"], metrics=None,
        scheduler_steps=None, alignment_steps={phase: None for phase in PHASES},
        operator_signatures={}, launch_signatures={})


def read_canonical_request_rows(path: str | Path) -> tuple[CanonicalRequestRow, ...]:
    """Read the exact four-column request input without reordering it."""

    source_path = Path(path)
    if not source_path.is_file():
        raise ValueError(f"canonical requests source is not a file: {source_path}")
    try:
        text = source_path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("canonical requests CSV must be UTF-8") from error
    reader = csv.reader(io.StringIO(text), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("canonical requests CSV is empty") from None
    except csv.Error as error:
        raise ValueError(f"invalid canonical requests CSV header: {error}") from error
    if tuple(header) != CANONICAL_REQUEST_COLUMNS:
        raise ValueError("canonical requests CSV header does not match the four-column schema")
    rows = []
    try:
        for request_idx, values in enumerate(reader):
            if len(values) != len(CANONICAL_REQUEST_COLUMNS):
                raise ValueError(f"canonical requests row {request_idx} has {len(values)} columns; expected 4")
            rows.append(CanonicalRequestRow(
                arrival_time_ms=_decimal_number(values[0], "arrival time", request_idx),
                input_tokens=_decimal_integer(values[1], "input tokens", request_idx),
                output_tokens=_decimal_integer(values[2], "output tokens", request_idx),
                cached_hit_tokens=_decimal_integer(values[3], "cached hit tokens", request_idx),
            ))
    except csv.Error as error:
        raise ValueError(f"invalid canonical requests CSV near row {len(rows)}: {error}") from error
    if not rows:
        raise ValueError("canonical requests CSV requires at least one data row")
    return tuple(rows)


def canonical_requests_csv_bytes(rows: Sequence[CanonicalRequestRow]) -> bytes:
    """Return the only allowed physical serialization of request rows."""

    if not rows:
        raise ValueError("canonical requests serialization requires at least one row")
    if any(not isinstance(row, CanonicalRequestRow) for row in rows):
        raise TypeError("canonical requests serialization requires CanonicalRequestRow")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CANONICAL_REQUEST_COLUMNS)
    for row in rows:
        writer.writerow((_format_number(row.arrival_time_ms), row.input_tokens, row.output_tokens, row.cached_hit_tokens))
    return buffer.getvalue().encode("utf-8")


def raw_source_set_sha256(raw_sources: Sequence[Mapping[str, object]]) -> str:
    """Hash the precise raw-source identities consumed by a case."""

    normalized = []
    archive_ids = set()
    for index, value in enumerate(raw_sources):
        source = _mapping(value, f"raw source {index}")
        _exact_keys(source, ("archive_id", "role", "relative_path", "bytes", "sha256", "tool", "tool_version"), f"raw source {index}")
        _nonempty_string(source["archive_id"], f"raw source {index} archive_id")
        if source["archive_id"] in archive_ids:
            raise ValueError("raw source set contains duplicate archive_id")
        archive_ids.add(source["archive_id"])
        _nonempty_string(source["role"], f"raw source {index} role")
        _relative_path(source["relative_path"], f"raw source {index} relative_path")
        _integer(source["bytes"], f"raw source {index} bytes", minimum=0)
        _sha256(source["sha256"], f"raw source {index} sha256")
        _optional_nonempty_string(source["tool"], f"raw source {index} tool")
        _optional_nonempty_string(source["tool_version"], f"raw source {index} tool_version")
        normalized.append(dict(source))
    if not normalized:
        raise ValueError("raw source set cannot be empty")
    normalized.sort(key=lambda source: source["archive_id"])
    return _canonical_json_sha256(normalized)


def signature_id(payload: Mapping[str, object]) -> str:
    """Return a content-addressed signature ID for one canonical JSON object."""

    _json_value(payload, "signature payload")
    return _canonical_json_sha256(payload)


def _validate_manifest(manifest: Mapping[str, object]) -> None:
    if "schema_version" not in manifest:
        raise ValueError("manifest is missing schema_version")
    _schema_version(manifest["schema_version"], "manifest")
    schema_version = manifest["schema_version"]
    common_keys = (
        "schema_version", "identity", "runtime", "requests", "metrics",
        "alignment_steps", "raw_sources", "provenance", "converter",
    )
    if schema_version == SCHEMA_VERSION:
        _exact_keys(manifest, common_keys, "manifest")
    elif schema_version in _SCHEDULER_CAPTURE_SCHEMA_VERSIONS:
        _exact_keys(
            manifest,
            common_keys + ("scheduler_steps", "capture"),
            "manifest",
        )
    else:
        raise AssertionError("manifest schema version was validated")
    identity = _mapping(manifest["identity"], "manifest identity")
    _exact_keys(identity, ("source_kind", "case_id", "hardware_id", "model_id", "raw_model_id", "model_revision", "tokenizer_revision"), "manifest identity")
    _enum(identity["source_kind"], ("sharegpt", "production"), "identity source_kind")
    _nonempty_string(identity["case_id"], "identity case_id")
    if identity["source_kind"] == "sharegpt":
        if identity["hardware_id"] not in CANONICAL_HARDWARE_IDS:
            raise ValueError("ShareGPT hardware_id must be canonical")
    elif identity["hardware_id"] is not None:
        raise ValueError("Production hardware_id must be null")
    _model_id(identity["model_id"], "identity model_id")
    _optional_nonempty_string(identity["raw_model_id"], "identity raw_model_id")
    _optional_nonempty_string(identity["model_revision"], "identity model_revision")
    _optional_nonempty_string(identity["tokenizer_revision"], "identity tokenizer_revision")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    runtime_keys = (
        "dtype", "deployment", "tensor_parallel_size", "expert_parallel_size",
        "pipeline_parallel_size", "target_qps", "seed",
        "scheduler_config_sha256", "runtime_version", "source_commit",
        "capture_enable_prefix_caching",
    )
    pd_runtime = runtime["deployment"] in ("pd_disagg", "pd_af_disagg")
    if schema_version in _DEPTH_SCHEMA_VERSIONS:
        runtime_keys += (
            ("instances",)
            if pd_runtime
            else (
                "scheduler_pipeline_depth",
                "scheduler_pipeline_depth_evidence",
            )
        )
    _exact_keys(runtime, runtime_keys, "manifest runtime")
    _optional_nonempty_string(runtime["dtype"], "runtime dtype")
    _optional_nonempty_string(runtime["deployment"], "runtime deployment")
    if identity["source_kind"] == "sharegpt":
        for key in ("tensor_parallel_size", "expert_parallel_size", "pipeline_parallel_size"):
            _integer(runtime[key], f"runtime {key}", minimum=1)
    elif identity["source_kind"] == "production":
        for key in ("tensor_parallel_size", "expert_parallel_size", "pipeline_parallel_size"):
            _optional_integer(runtime[key], f"runtime {key}", minimum=1)
    else:
        raise AssertionError("identity source_kind was validated")
    if identity["source_kind"] == "sharegpt":
        _finite_positive(runtime["target_qps"], "runtime target_qps")
    elif runtime["target_qps"] is not None:
        _finite_positive(runtime["target_qps"], "runtime target_qps")
    _optional_integer(runtime["seed"], "runtime seed", minimum=0)
    _optional_sha256(runtime["scheduler_config_sha256"], "runtime scheduler_config_sha256")
    _optional_nonempty_string(runtime["runtime_version"], "runtime runtime_version")
    _optional_nonempty_string(runtime["source_commit"], "runtime source_commit")
    if runtime["capture_enable_prefix_caching"] is not None and not isinstance(runtime["capture_enable_prefix_caching"], bool):
        raise TypeError("runtime capture_enable_prefix_caching must be boolean or null")
    if schema_version in _DEPTH_SCHEMA_VERSIONS:
        if pd_runtime:
            _validate_scheduler_pipeline_depth_instances(runtime)
        else:
            _validate_scheduler_pipeline_depth_runtime(runtime)
    requests = _mapping(manifest["requests"], "manifest requests")
    _exact_keys(requests, ("path", "row_count", "sha256", "field_provenance", "cached_hit_value_evidence"), "manifest requests")
    if requests["path"] != "requests.csv":
        raise ValueError("manifest requests path must be requests.csv")
    _integer(requests["row_count"], "requests row_count", minimum=1)
    _sha256(requests["sha256"], "requests sha256")
    field_provenance = _mapping(requests["field_provenance"], "requests field_provenance")
    _exact_keys(field_provenance, CANONICAL_REQUEST_COLUMNS, "requests field_provenance")
    for field in CANONICAL_REQUEST_COLUMNS:
        value = _mapping(field_provenance[field], f"request provenance {field}")
        _exact_keys(value, ("kind", "source_archive_id", "reason"), f"request provenance {field}")
        _enum(value["kind"], ("requested", "observed_actual", "scheduler_derived", "assumed_zero", "scheduler_equivalent_representative"), f"request provenance {field} kind")
        _optional_nonempty_string(value["source_archive_id"], f"request provenance {field} source_archive_id")
        _optional_nonempty_string(value["reason"], f"request provenance {field} reason")
    _validate_metrics_entry(manifest["metrics"])
    _validate_alignment_entry(manifest["alignment_steps"])
    if schema_version in _SCHEDULER_CAPTURE_SCHEMA_VERSIONS:
        if identity["source_kind"] != "sharegpt":
            raise ValueError("scheduler-capture schemas require ShareGPT source_kind")
        _validate_scheduler_entry(manifest["scheduler_steps"])
    raw_sources = _sequence(manifest["raw_sources"], "manifest raw_sources")
    expected_raw_sha = raw_source_set_sha256(tuple(_mapping(value, "manifest raw source") for value in raw_sources))
    historical_raw_source_shas = {
        raw_source_set_sha256(tuple(
            _mapping(value, "manifest raw source")
            for value in raw_sources[:prefix_length]
        ))
        for prefix_length in range(1, len(raw_sources) + 1)
    }
    raw_archive_ids = {
        _mapping(value, "manifest raw source")["archive_id"] for value in raw_sources
    }
    if schema_version in _SCHEDULER_CAPTURE_SCHEMA_VERSIONS:
        _validate_capture(
            manifest["capture"],
            manifest["scheduler_steps"],
            manifest["metrics"],
            raw_sources,
            schema_version,
            identity,
        )
    _validate_request_provenance_bindings(field_provenance, raw_archive_ids)
    _validate_cached_hit_value_evidence(
        requests["cached_hit_value_evidence"],
        field_provenance["Cached hit [tokens]"],
        raw_archive_ids,
        historical_raw_source_shas,
    )
    provenance = _mapping(manifest["provenance"], "manifest provenance")
    _exact_keys(provenance, ("raw_source_set_sha256", "gaps"), "manifest provenance")
    if provenance["raw_source_set_sha256"] != expected_raw_sha:
        raise ValueError("manifest provenance raw_source_set_sha256 disagrees with raw_sources")
    gaps = _sequence(provenance["gaps"], "manifest provenance gaps")
    for index, gap in enumerate(gaps):
        _nonempty_string(gap, f"manifest provenance gap {index}")
    converter = _mapping(manifest["converter"], "manifest converter")
    _exact_keys(converter, ("implementation_id", "source_commit", "selection_rule"), "manifest converter")
    _nonempty_string(converter["implementation_id"], "converter implementation_id")
    validate_converter_source_commit(converter["source_commit"])
    if identity["source_kind"] == "sharegpt":
        expected_selection_rule = "middle_complete_work_bearing_step_v1"
    elif identity["source_kind"] == "production":
        expected_selection_rule = "physical_row_order_v1"
    else:
        raise AssertionError("identity source_kind was validated")
    if converter["selection_rule"] != expected_selection_rule:
        raise ValueError(f"converter selection_rule must be {expected_selection_rule}")


def _validate_cached_hit_value_evidence(
    value: object,
    field_provenance: object,
    raw_archive_ids: set[object],
    historical_raw_source_shas: set[str],
) -> None:
    provenance = _mapping(field_provenance, "cached-hit request provenance")
    kind = provenance["kind"]
    if kind == "observed_actual":
        if value is not None:
            raise ValueError("observed cached-hit provenance requires null value evidence")
        return
    if kind not in ("assumed_zero", "scheduler_equivalent_representative"):
        raise ValueError("cached-hit provenance must be observed_actual, assumed_zero, or scheduler_equivalent_representative")
    evidence = _mapping(value, "cached-hit value evidence")
    _exact_keys(
        evidence,
        ("kind", "logical_prefix_hit_observability", "representative_value", "net_prefix_work_reduction_tokens", "derivation", "source_archive_ids", "raw_source_set_sha256"),
        "cached-hit value evidence",
    )
    if evidence["kind"] != kind:
        raise ValueError("cached-hit evidence kind disagrees with field provenance")
    _integer(evidence["representative_value"], "cached-hit representative_value", minimum=0)
    if evidence["representative_value"] != 0:
        raise ValueError("cached-hit representative evidence must encode zero")
    _integer(evidence["net_prefix_work_reduction_tokens"], "cached-hit net_prefix_work_reduction_tokens", minimum=0)
    if evidence["net_prefix_work_reduction_tokens"] != 0:
        raise ValueError("cached-hit representative evidence requires zero net work reduction")
    source_ids = _sequence(evidence["source_archive_ids"], "cached-hit source_archive_ids")
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError("cached-hit evidence requires unique, non-empty source_archive_ids")
    for source_id in source_ids:
        if source_id not in raw_archive_ids:
            raise ValueError("cached-hit evidence references unknown raw archive")
    if evidence["raw_source_set_sha256"] not in historical_raw_source_shas:
        raise ValueError("cached-hit evidence raw source set SHA disagrees with manifest")
    if kind == "assumed_zero":
        if evidence["logical_prefix_hit_observability"] != "observed" or evidence["derivation"] != "explicit_h_zero_proof":
            raise ValueError("assumed_zero cached-hit evidence requires explicit observed H=0 proof")
    elif (
        evidence["logical_prefix_hit_observability"] != "unavailable"
        or evidence["derivation"] != "full_scheduler_token_conservation"
    ):
        raise ValueError("scheduler-equivalent cached-hit evidence requires unavailable H and full Scheduler conservation")


def _validate_request_provenance_bindings(
    field_provenance: Mapping[str, object], raw_archive_ids: set[object]
) -> None:
    for field in CANONICAL_REQUEST_COLUMNS:
        provenance = _mapping(field_provenance[field], f"request provenance {field}")
        if provenance["kind"] in ("assumed_zero", "scheduler_equivalent_representative"):
            if provenance["source_archive_id"] is not None:
                raise ValueError("representative cached-hit provenance must cite structured evidence only")
        elif provenance["source_archive_id"] not in raw_archive_ids:
            raise ValueError(f"request provenance for {field} references unknown raw archive")


def _validate_cached_hit_rows(
    requests: Sequence[CanonicalRequestRow],
    request_entry: Mapping[str, object],
) -> None:
    field_provenance = _mapping(request_entry["field_provenance"], "requests field_provenance")
    cached_hit = _mapping(
        field_provenance["Cached hit [tokens]"], "cached-hit request provenance"
    )
    if cached_hit["kind"] in ("assumed_zero", "scheduler_equivalent_representative") and any(
        request.cached_hit_tokens != 0 for request in requests
    ):
        raise ValueError("zero-valued cached-hit evidence requires every requests.csv value to be zero")


def _validate_metrics_entry(value: object) -> None:
    entry = _mapping(value, "manifest metrics")
    _exact_keys(entry, ("status", "reason", "path", "sha256"), "manifest metrics")
    _enum(entry["status"], ("available", "unavailable"), "manifest metrics status")
    _optional_nonempty_string(entry["reason"], "manifest metrics reason")
    if entry["status"] == "available":
        if entry["reason"] is not None or entry["path"] != "metrics.json":
            raise ValueError("available metrics require null reason and metrics.json path")
        _sha256(entry["sha256"], "manifest metrics sha256")
    elif entry["reason"] is None or entry["path"] is not None or entry["sha256"] is not None:
        raise ValueError("unavailable metrics require reason and null path/SHA")


def _validate_alignment_entry(value: object) -> None:
    entry = _mapping(value, "manifest alignment")
    _exact_keys(
        entry,
        ("status", "path", "sha256", "compressed_sha256", "compression", "phases"),
        "manifest alignment",
    )
    if entry["status"] != "available":
        raise ValueError("schema-v2 compact cases require available alignment_steps")
    if entry["path"] != ALIGNMENT_FILENAME:
        raise ValueError(f"manifest alignment path must be {ALIGNMENT_FILENAME}")
    _sha256(entry["sha256"], "manifest alignment sha256")
    _sha256(entry["compressed_sha256"], "manifest alignment compressed_sha256")
    if entry["compression"] != COMPRESSION_ALGORITHM:
        raise ValueError("manifest alignment compression must be zstd")
    phases = _mapping(entry["phases"], "manifest alignment phases")
    _exact_keys(phases, PHASES, "manifest alignment phases")
    for phase in PHASES:
        availability = _mapping(phases[phase], f"manifest phase {phase}")
        _exact_keys(availability, ("status", "rank_ids"), f"manifest phase {phase}")
        _enum(availability["status"], ("available", "not_observed"), f"manifest phase {phase} status")
        ranks = _rank_ids(availability["rank_ids"], f"manifest phase {phase} rank_ids")
        if availability["status"] == "available" and not ranks:
            raise ValueError(f"available phase {phase} requires ranks")
        if availability["status"] == "not_observed" and ranks:
            raise ValueError(f"not_observed phase {phase} requires empty ranks")


def _validate_scheduler_entry(value: object) -> None:
    entry = _mapping(value, "manifest scheduler_steps")
    _exact_keys(
        entry,
        (
            "path", "sha256", "compressed_sha256", "compression",
            "row_count", "timing_boundary", "provenance",
        ),
        "manifest scheduler_steps",
    )
    if entry["path"] != SCHEDULER_STEPS_FILENAME:
        raise ValueError(
            f"manifest scheduler_steps path must be {SCHEDULER_STEPS_FILENAME}")
    _sha256(entry["sha256"], "manifest scheduler_steps sha256")
    _sha256(
        entry["compressed_sha256"],
        "manifest scheduler_steps compressed_sha256",
    )
    if entry["compression"] != COMPRESSION_ALGORITHM:
        raise ValueError("manifest scheduler_steps compression must be zstd")
    _integer(entry["row_count"], "manifest scheduler_steps row_count", minimum=1)
    boundary = _mapping(
        entry["timing_boundary"], "manifest scheduler_steps timing_boundary"
    )
    _exact_keys(
        boundary,
        (
            "clock", "origin", "step_start", "schedule_end",
            "update_start", "step_end",
        ),
        "manifest scheduler_steps timing_boundary",
    )
    expected_boundary = {
        "clock": "time.perf_counter_ns",
        "origin": "first_scheduler_step_start",
        "step_start": "immediately_before_scheduler_schedule",
        "schedule_end": "immediately_after_scheduler_schedule",
        "update_start": "immediately_before_scheduler_output_update",
        "step_end": "immediately_after_scheduler_output_update",
    }
    if dict(boundary) != expected_boundary:
        raise ValueError("manifest scheduler timing boundary is invalid")
    provenance = _mapping(
        entry["provenance"], "manifest scheduler_steps provenance"
    )
    _exact_keys(
        provenance,
        (
            "source_archive_id", "source_sha256", "capture_run_id",
            "instrumentation_id", "instrumentation_version", "buffering",
            "profiler_attached",
        ),
        "manifest scheduler_steps provenance",
    )
    _nonempty_string(
        provenance["source_archive_id"], "scheduler source_archive_id"
    )
    _sha256(provenance["source_sha256"], "scheduler source_sha256")
    _nonempty_string(provenance["capture_run_id"], "scheduler capture_run_id")
    _nonempty_string(
        provenance["instrumentation_id"], "scheduler instrumentation_id"
    )
    _integer(
        provenance["instrumentation_version"],
        "scheduler instrumentation_version",
        minimum=1,
    )
    if provenance["buffering"] != "in_memory_append_then_single_write":
        raise ValueError("scheduler trace buffering contract is invalid")
    if provenance["profiler_attached"] is not False:
        raise ValueError("canonical scheduler trace must be captured without profiler")


def _validate_capture(
    value: object,
    scheduler_entry_value: object,
    metrics_entry_value: object,
    raw_sources: Sequence[object],
    schema_version: int,
    identity: Mapping[str, object],
) -> None:
    raw_by_id = {
        _mapping(source, "manifest raw source")["archive_id"]:
        _mapping(source, "manifest raw source") for source in raw_sources
    }
    raw_archive_ids = set(raw_by_id)
    capture = _mapping(value, "manifest capture")
    _exact_keys(
        capture,
        (
            "run_id", "requests_source_archive_ids", "metrics_source_archive_id",
            "scheduler_source_archive_id", "scheduler_trace_enabled",
            "profiler_attached", "observer_effect_validation",
        ),
        "manifest capture",
    )
    _nonempty_string(capture["run_id"], "capture run_id")
    request_source_ids = _string_id_sequence(
        capture["requests_source_archive_ids"],
        "capture requests_source_archive_ids",
    )
    if not request_source_ids:
        raise ValueError("capture requests_source_archive_ids cannot be empty")
    for archive_id in request_source_ids:
        if archive_id not in raw_archive_ids:
            raise ValueError("capture requests source references unknown raw archive")
    for field in ("metrics_source_archive_id", "scheduler_source_archive_id"):
        _nonempty_string(capture[field], f"capture {field}")
        if capture[field] not in raw_archive_ids:
            raise ValueError(f"capture {field} references unknown raw archive")
    if capture["scheduler_trace_enabled"] is not True:
        raise ValueError("schema-v3 canonical capture requires scheduler trace")
    if capture["profiler_attached"] is not False:
        raise ValueError("schema-v3 canonical capture must not attach a profiler")
    scheduler_entry = _mapping(scheduler_entry_value, "manifest scheduler_steps")
    scheduler_provenance = _mapping(
        scheduler_entry["provenance"], "scheduler provenance"
    )
    if scheduler_provenance["capture_run_id"] != capture["run_id"]:
        raise ValueError("scheduler capture_run_id disagrees with manifest capture")
    if scheduler_provenance["source_archive_id"] != capture[
        "scheduler_source_archive_id"
    ]:
        raise ValueError("scheduler source archive disagrees with manifest capture")
    metrics_entry = _mapping(metrics_entry_value, "manifest metrics")
    if metrics_entry["status"] != "available":
        raise ValueError("schema-v3 canonical capture requires available metrics")
    validation = _mapping(
        capture["observer_effect_validation"],
        "capture observer_effect_validation",
    )
    validation_fields = (
        "status", "validation_case_id", "baseline_run_id",
        "instrumented_run_id", "threshold_percent", "metric_delta_percent",
        "passed",
    )
    if schema_version == OBSERVER_QUALITY_SCHEMA_VERSION:
        validation_fields += ("source_archive_id",)
    _exact_keys(validation, validation_fields, "capture observer_effect_validation")
    _enum(
        validation["status"],
        ("measured_for_this_case", "inherited_from_representative_case"),
        "observer effect validation status",
    )
    for field in (
        "validation_case_id", "baseline_run_id", "instrumented_run_id"
    ):
        _nonempty_string(validation[field], f"observer effect {field}")
    threshold = validation["threshold_percent"]
    _finite_non_negative(threshold, "observer effect threshold_percent")
    if threshold != 3.0:
        raise ValueError("observer effect threshold_percent must equal 3.0")
    deltas = _mapping(
        validation["metric_delta_percent"], "observer effect metric deltas"
    )
    legacy_delta_fields = {"request_throughput", "ttft", "tpot", "e2e"}
    current_delta_fields = legacy_delta_fields | {"output_token_throughput"}
    allowed_delta_sets = (
        {frozenset(current_delta_fields)}
        if schema_version == OBSERVER_QUALITY_SCHEMA_VERSION else
        {frozenset(legacy_delta_fields), frozenset(current_delta_fields)}
    )
    if frozenset(deltas) not in allowed_delta_sets:
        raise ValueError("observer effect metric deltas keys mismatch")
    for field in deltas:
        delta = deltas[field]
        _finite_non_negative(delta, f"observer effect {field} delta")
        if schema_version != OBSERVER_QUALITY_SCHEMA_VERSION and delta > threshold:
            raise ValueError("accepted scheduler instrumentation exceeds 3% gate")
    if schema_version == OBSERVER_QUALITY_SCHEMA_VERSION:
        if identity["hardware_id"] != "nvidia_h200":
            raise ValueError("schema-v7 observer quality is H200-only")
        if validation["status"] != "measured_for_this_case":
            raise ValueError("schema-v7 requires own-case observer measurement")
        expected_case = "/".join(str(identity[field]) for field in
                                 ("hardware_id", "model_id", "case_id"))
        if validation["validation_case_id"] != expected_case:
            raise ValueError("observer validation case differs from captured case")
        if validation["instrumented_run_id"] != capture["run_id"]:
            raise ValueError("own observer instrumented run differs from capture")
        if type(validation["passed"]) is not bool or validation["passed"] != all(
                delta <= threshold for delta in deltas.values()):
            raise ValueError("observer passed disagrees with all five metric deltas")
        source_id = validation["source_archive_id"]
        if source_id not in raw_by_id:
            raise ValueError("observer source archive is absent")
        source = raw_by_id[source_id]
        run_parts = capture["run_id"].split("/")
        if (len(run_parts) != 2
                or re.fullmatch(r"trace_on(?:_[A-Za-z0-9_]+)?", run_parts[0]) is None
                or run_parts[1] != "ground_truth.json"):
            raise ValueError("own observer capture run identity is invalid")
        if (source["role"] != "scheduler-observer-effect"
                or source["relative_path"] !=
                f"{run_parts[0]}/scheduler_trace_observer_effect.json"):
            raise ValueError("observer source archive has wrong role or path")
    elif validation["passed"] is not True:
        raise ValueError("canonical scheduler instrumentation must pass A/B gate")


def _validate_same_run_bindings(
    manifest: Mapping[str, object],
    metrics: Mapping[str, object] | None,
) -> None:
    if metrics is None:
        raise ValueError("schema-v3 same-run binding requires metrics")
    capture = _mapping(manifest["capture"], "manifest capture")
    request_source_ids = set(_sequence(
        capture["requests_source_archive_ids"],
        "capture requests_source_archive_ids",
    ))
    requests = _mapping(manifest["requests"], "manifest requests")
    field_provenance = _mapping(
        requests["field_provenance"], "requests field_provenance"
    )
    for field in CANONICAL_REQUEST_COLUMNS:
        provenance = _mapping(
            field_provenance[field], f"request provenance {field}"
        )
        source_id = provenance["source_archive_id"]
        if source_id is not None and source_id not in request_source_ids:
            raise ValueError(
                f"request provenance for {field} is not bound to capture run"
            )
    metric_provenance = _mapping(
        metrics["metric_provenance"], "metrics metric_provenance"
    )
    if metric_provenance["source_archive_id"] != capture[
        "metrics_source_archive_id"
    ]:
        raise ValueError("metrics source archive disagrees with manifest capture")
    validation = _mapping(
        capture["observer_effect_validation"],
        "capture observer_effect_validation",
    )
    if (
        validation["status"] == "measured_for_this_case"
        and validation["instrumented_run_id"] != capture["run_id"]
    ):
        raise ValueError(
            "measured observer-effect instrumented run must be the accepted run"
        )


def _load_metrics(
    root: Path,
    entry: Mapping[str, object],
    request_sha: str,
    request_count: int,
    source_kind: object,
    schema_version: int,
) -> Mapping[str, object] | None:
    if entry["status"] == "unavailable":
        return None
    path = _safe_case_path(root, entry["path"])
    _verify_sha(path, entry["sha256"], "metrics.json")
    metrics = _read_json_object(path)
    _validate_metrics(
        metrics, request_sha, request_count, source_kind, schema_version
    )
    return metrics


def _validate_metrics_raw_binding(
    metrics: Mapping[str, object] | None,
    requests: Sequence[CanonicalRequestRow],
    manifest: Mapping[str, object],
) -> None:
    if metrics is None:
        return
    raw_sources = _sequence(manifest["raw_sources"], "manifest raw_sources")
    source_sha_by_id = {}
    for value in raw_sources:
        source = _mapping(value, "manifest raw source")
        source_sha_by_id[source["archive_id"]] = source["sha256"]
    provenance = _mapping(metrics["metric_provenance"], "metrics metric_provenance")
    if provenance["source_archive_id"] not in source_sha_by_id:
        raise ValueError("metrics provenance references unknown raw archive")
    if source_sha_by_id[provenance["source_archive_id"]] != provenance["source_sha256"]:
        raise ValueError("metrics provenance source SHA disagrees with raw source")
    source_kind = _mapping(manifest["identity"], "manifest identity")["source_kind"]
    for value in _sequence(metrics["per_request"], "metrics per_request"):
        entry = _mapping(value, "metrics per_request entry")
        request_idx = entry["request_idx"]
        if (
            source_kind == "sharegpt"
            and entry["actual_output_tokens"] != requests[request_idx].output_tokens
        ):
            raise ValueError("metrics actual_output_tokens disagrees with requests.csv")


def _validate_metrics(
    metrics: Mapping[str, object],
    request_sha: str | None = None,
    request_count: int | None = None,
    source_kind: object = "sharegpt",
    schema_version: int | None = None,
) -> None:
    """Validate source-kind-specific metrics within the shared envelope."""

    common_keys = (
        "schema_version", "requests_sha256", "metric_provenance",
        "per_request", "summary",
    )
    if source_kind == "sharegpt":
        _exact_keys(metrics, common_keys, "metrics")
    elif source_kind == "production":
        _exact_keys(metrics, common_keys + ("field_coverage",), "metrics")
    else:
        raise ValueError(f"unsupported metrics source_kind: {source_kind!r}")
    if schema_version is None:
        _schema_version(metrics["schema_version"], "metrics")
    else:
        _schema_version(
            metrics["schema_version"], "metrics", expected=schema_version
        )
    _sha256(metrics["requests_sha256"], "metrics requests_sha256")
    if request_sha is not None and metrics["requests_sha256"] != request_sha:
        raise ValueError("metrics requests_sha256 disagrees with requests.csv")
    provenance = _mapping(metrics["metric_provenance"], "metrics metric_provenance")
    _exact_keys(provenance, ("source_kind", "source_archive_id", "source_sha256", "profiler_attached"), "metrics metric_provenance")
    if source_kind == "sharegpt":
        expected_provenance_kind = "serving_runtime"
    elif source_kind == "production":
        expected_provenance_kind = "production_csv"
    else:
        raise AssertionError("metrics source_kind was validated")
    if provenance["source_kind"] != expected_provenance_kind:
        raise ValueError(
            f"schema-v2 {source_kind} metrics require "
            f"{expected_provenance_kind} provenance"
        )
    _nonempty_string(provenance["source_archive_id"], "metrics source_archive_id")
    _sha256(provenance["source_sha256"], "metrics source_sha256")
    if provenance["profiler_attached"] is not False:
        raise ValueError("published metrics must be explicitly non-profiler")
    if source_kind == "production":
        _validate_production_metrics(metrics, request_count)
        return
    entries = _sequence(metrics["per_request"], "metrics per_request")
    seen = set()
    for position, value in enumerate(entries):
        entry = _mapping(value, f"metrics per_request {position}")
        _exact_keys(entry, ("request_idx", "actual_output_tokens", "ttft_ms", "tpot_ms", "e2e_latency_ms"), f"metrics per_request {position}")
        _integer(entry["request_idx"], f"metrics request {position} request_idx", minimum=0)
        if entry["request_idx"] in seen:
            raise ValueError("metrics per_request contains duplicate request_idx")
        seen.add(entry["request_idx"])
        _integer(entry["actual_output_tokens"], f"metrics request {position} actual_output_tokens", minimum=0)
        _finite_non_negative(entry["e2e_latency_ms"], f"metrics request {position} e2e_latency_ms")
        if entry["actual_output_tokens"] == 0:
            if entry["ttft_ms"] is not None or entry["tpot_ms"] is not None:
                raise ValueError("zero-output metrics require null TTFT and TPOT")
        elif entry["actual_output_tokens"] == 1:
            _finite_non_negative(entry["ttft_ms"], f"metrics request {position} ttft_ms")
            if entry["tpot_ms"] != 0.0:
                raise ValueError("one-output metrics require zero TPOT")
        else:
            _finite_non_negative(entry["ttft_ms"], f"metrics request {position} ttft_ms")
            _finite_non_negative(entry["tpot_ms"], f"metrics request {position} tpot_ms")
    if request_count is not None and seen != set(range(request_count)):
        raise ValueError("metrics must contain every requests.csv row exactly once")
    summary = _mapping(metrics["summary"], "metrics summary")
    _exact_keys(summary, ("request_count", "request_throughput_per_s", "output_token_throughput_per_s", "total_token_throughput_per_s"), "metrics summary")
    if summary["request_count"] != len(entries):
        raise ValueError("metrics summary request_count disagrees with per_request")
    for key in ("request_throughput_per_s", "output_token_throughput_per_s", "total_token_throughput_per_s"):
        _finite_non_negative(summary[key], f"metrics summary {key}")


def _validate_production_metrics(
    metrics: Mapping[str, object], request_count: int | None
) -> None:
    _exact_keys(
        metrics,
        (
            "schema_version", "requests_sha256", "metric_provenance",
            "field_coverage", "per_request", "summary",
        ),
        "Production metrics",
    )
    field_coverage = _mapping(
        metrics["field_coverage"], "Production metrics field_coverage"
    )
    fields = (
        "inference_time_ms", "ttft_ms", "generation_speed_tokens_per_s",
    )
    _exact_keys(field_coverage, fields, "Production metrics field_coverage")
    entries = _sequence(metrics["per_request"], "Production metrics per_request")
    seen = set()
    unavailable_counts = {field: 0 for field in fields}
    for position, value in enumerate(entries):
        entry = _mapping(value, f"Production metrics per_request {position}")
        _exact_keys(
            entry,
            (
                "request_idx", "actual_output_tokens", "ttft_ms", "tpot_ms",
                "e2e_latency_ms", "inference_time_ms",
                "generation_speed_tokens_per_s",
            ),
            f"Production metrics per_request {position}",
        )
        _integer(entry["request_idx"], "Production metrics request_idx", minimum=0)
        if entry["request_idx"] in seen:
            raise ValueError("Production metrics contain duplicate request_idx")
        seen.add(entry["request_idx"])
        if entry["actual_output_tokens"] is not None:
            raise ValueError("Production CSV metrics do not observe actual_output_tokens")
        if entry["tpot_ms"] is not None or entry["e2e_latency_ms"] is not None:
            raise ValueError("Production CSV metrics cannot derive TPOT or E2E latency")
        _finite_non_negative(entry["inference_time_ms"], "Production inference_time_ms")
        for field in ("ttft_ms", "generation_speed_tokens_per_s"):
            if entry[field] is None:
                unavailable_counts[field] += 1
            else:
                _finite_non_negative(entry[field], f"Production {field}")
    if request_count is not None and seen != set(range(request_count)):
        raise ValueError("Production metrics must contain every requests.csv row exactly once")
    unavailable_counts["inference_time_ms"] = 0
    for field in fields:
        coverage = _mapping(
            field_coverage[field], f"Production metrics coverage {field}"
        )
        _exact_keys(
            coverage,
            ("coverage", "unavailable_request_count", "reason"),
            f"Production metrics coverage {field}",
        )
        expected_count = unavailable_counts[field]
        if coverage["unavailable_request_count"] != expected_count:
            raise ValueError(f"Production {field} unavailable count is inconsistent")
        expected_coverage = "complete" if expected_count == 0 else "partial"
        if coverage["coverage"] != expected_coverage:
            raise ValueError(f"Production {field} coverage is inconsistent")
        if (expected_count == 0) != (coverage["reason"] is None):
            raise ValueError(f"Production {field} coverage reason is inconsistent")
        if coverage["reason"] is not None:
            _nonempty_string(coverage["reason"], f"Production {field} reason")
    summary = _mapping(metrics["summary"], "Production metrics summary")
    _exact_keys(
        summary,
        (
            "request_count", "request_throughput_per_s",
            "output_token_throughput_per_s", "total_token_throughput_per_s",
        ),
        "Production metrics summary",
    )
    if summary["request_count"] != len(entries):
        raise ValueError("Production metrics summary request_count is inconsistent")
    for key in (
        "request_throughput_per_s", "output_token_throughput_per_s",
        "total_token_throughput_per_s",
    ):
        if summary[key] is not None:
            raise ValueError(f"Production metrics summary {key} must be null")


def _load_alignment(
    root: Path,
    entry: Mapping[str, object],
    manifest: Mapping[str, object],
    request_count: int,
    schema_version: int,
) -> Mapping[str, object]:
    path = _safe_case_path(root, entry["path"])
    content = read_compressed_canonical_bytes(
        path,
        compressed_sha256=entry["compressed_sha256"],
        content_sha256=entry["sha256"],
        label=ALIGNMENT_FILENAME,
    )
    alignment = _mapping(
        _read_json_bytes(content, ALIGNMENT_FILENAME),
        f"JSON file {path}",
    )
    identity = _mapping(manifest["identity"], "manifest identity")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    if identity["source_kind"] == "sharegpt":
        expected_ranks = tuple(
            range(max(runtime["tensor_parallel_size"], runtime["expert_parallel_size"]))
        )
    elif identity["source_kind"] == "production":
        expected_ranks = ()
    else:
        raise AssertionError("manifest source_kind was validated")
    raw_sources = _sequence(manifest["raw_sources"], "manifest raw_sources")
    raw_archive_ids = {_mapping(value, "manifest raw source")["archive_id"] for value in raw_sources}
    historical_raw_source_shas = {
        raw_source_set_sha256(tuple(
            _mapping(value, "manifest raw source")
            for value in raw_sources[:prefix_length]
        ))
        for prefix_length in range(1, len(raw_sources) + 1)
    }
    return _validate_alignment(
        alignment,
        entry,
        identity["hardware_id"],
        historical_raw_source_shas,
        raw_archive_ids,
        request_count,
        expected_ranks,
        schema_version,
    )


def canonical_scheduler_steps_jsonl_bytes(
    steps: Sequence[Mapping[str, object]],
) -> bytes:
    """Return the only allowed physical serialization of scheduler steps."""

    if not steps:
        raise ValueError("scheduler_steps serialization requires at least one row")
    lines = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise TypeError(f"scheduler_steps row {index} must be an object")
        try:
            lines.append(json.dumps(
                step,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ))
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"scheduler_steps row {index} is not finite JSON data: {error}"
            ) from error
    return ("\n".join(lines) + "\n").encode("utf-8")


def _load_scheduler_steps(
    root: Path,
    entry: Mapping[str, object],
    manifest: Mapping[str, object],
    requests: Sequence[CanonicalRequestRow],
) -> tuple[Mapping[str, object], ...]:
    path = _safe_case_path(root, entry["path"])
    content = read_compressed_canonical_bytes(
        path,
        compressed_sha256=entry["compressed_sha256"],
        content_sha256=entry["sha256"],
        label=SCHEDULER_STEPS_FILENAME,
    )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("scheduler_steps.jsonl must be UTF-8") from error
    if not text.endswith("\n"):
        raise ValueError("scheduler_steps.jsonl must end with one newline")
    lines = text.splitlines()
    if not lines or any(not line for line in lines):
        raise ValueError("scheduler_steps.jsonl cannot contain empty rows")
    steps = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(
                line,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(
                f"invalid scheduler_steps.jsonl row {index}: {error}"
            ) from error
        steps.append(_mapping(value, f"scheduler_steps row {index}"))
    if content != canonical_scheduler_steps_jsonl_bytes(steps):
        raise ValueError("scheduler_steps.jsonl is not in canonical serialization")
    if entry["row_count"] != len(steps):
        raise ValueError("manifest scheduler row_count disagrees with JSONL")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    expected_rank_ids = tuple(
        range(max(runtime["tensor_parallel_size"], runtime["expert_parallel_size"]))
    )
    configured_depth = (
        runtime["scheduler_pipeline_depth"]
        if (manifest["schema_version"]
            in _DEPTH_SCHEMA_VERSIONS
            and runtime["deployment"] not in ("pd_disagg", "pd_af_disagg"))
        else None
    )
    _validate_scheduler_steps(
        steps, requests, expected_rank_ids, configured_depth)
    scheduler_provenance = _mapping(
        entry["provenance"], "manifest scheduler_steps provenance"
    )
    raw_sources = _sequence(manifest["raw_sources"], "manifest raw_sources")
    source_sha_by_id = {
        _mapping(value, "manifest raw source")["archive_id"]:
        _mapping(value, "manifest raw source")["sha256"]
        for value in raw_sources
    }
    source_id = scheduler_provenance["source_archive_id"]
    if source_id not in source_sha_by_id:
        raise ValueError("scheduler provenance references unknown raw archive")
    if source_sha_by_id[source_id] != scheduler_provenance["source_sha256"]:
        raise ValueError("scheduler provenance source SHA disagrees with raw source")
    return tuple(steps)


def _validate_scheduler_steps(
    steps: Sequence[Mapping[str, object]],
    requests: Sequence[CanonicalRequestRow],
    expected_rank_ids: tuple[int, ...],
    configured_depth: int | None,
) -> None:
    first_step = _mapping(steps[0], "scheduler step 0")
    schema_version = first_step["schema_version"]
    _schema_version(schema_version, "scheduler step 0")
    if schema_version == SCHEDULER_CAPTURE_SCHEMA_VERSION:
        _validate_scheduler_steps_v3(
            steps,
            requests,
            include_terminal_output_token=False,
        )
        return
    if schema_version in (
            RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
            SCHEDULER_PIPELINE_DEPTH_SCHEMA_VERSION):
        _validate_scheduler_steps_v4(
            steps,
            requests,
            expected_rank_ids,
            schema_version=schema_version,
            configured_depth=configured_depth,
        )
        return
    raise AssertionError("scheduler step schema version was validated")


def _validate_scheduler_pipeline_depth_runtime(
    runtime: Mapping[str, object],
) -> None:
    """Validate configured capacity evidence independently of trace occupancy."""

    _integer(
        runtime["scheduler_pipeline_depth"],
        "runtime scheduler_pipeline_depth",
        minimum=1,
    )
    evidence = _mapping(
        runtime["scheduler_pipeline_depth_evidence"],
        "runtime scheduler_pipeline_depth_evidence",
    )
    source_pair = (evidence["source_kind"], evidence["source_field"])
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        _exact_keys(
            evidence,
            (
                "source_kind", "source_field", "executor_class",
                "pipeline_parallel_size", "runtime_version", "source_identity",
            ),
            "runtime scheduler_pipeline_depth_evidence",
        )
        source_identity = evidence["source_identity"]
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError("scheduler pipeline depth evidence source identity is invalid")
        if source_identity != runtime["source_commit"]:
            raise ValueError(
                "scheduler pipeline depth evidence source identity disagrees "
                "with runtime")
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        _exact_keys(
            evidence,
            (
                "source_kind", "source_field", "configured_depth",
                "executor_class", "pipeline_parallel_size", "runtime_version",
                "source_identity",
            ),
            "runtime scheduler_pipeline_depth_evidence",
        )
        if evidence["configured_depth"] != runtime["scheduler_pipeline_depth"]:
            raise ValueError(
                "scheduler pipeline depth evidence configured depth disagrees "
                "with runtime")
        source_identity = _mapping(
            evidence["source_identity"],
            "scheduler pipeline depth evidence source identity",
        )
        _exact_keys(
            source_identity, ("module_file", "module_name"),
            "scheduler pipeline depth evidence source identity",
        )
        _nonempty_string(
            source_identity["module_file"],
            "scheduler pipeline depth evidence module file",
        )
        _nonempty_string(
            source_identity["module_name"],
            "scheduler pipeline depth evidence module name",
        )
    else:
        raise ValueError("scheduler pipeline depth evidence source is invalid")
    _nonempty_string(
        evidence["executor_class"],
        "scheduler pipeline depth evidence executor_class",
    )
    _integer(
        evidence["pipeline_parallel_size"],
        "scheduler pipeline depth evidence pipeline_parallel_size",
        minimum=1,
    )
    if evidence["pipeline_parallel_size"] != runtime["pipeline_parallel_size"]:
        raise ValueError(
            "scheduler pipeline depth evidence PP disagrees with runtime")
    _nonempty_string(
        evidence["runtime_version"],
        "scheduler pipeline depth evidence runtime_version",
    )
    if evidence["runtime_version"] != runtime["runtime_version"]:
        raise ValueError(
            "scheduler pipeline depth evidence version disagrees with runtime")


def _validate_scheduler_pipeline_depth_instances(
    runtime: Mapping[str, object],
) -> None:
    """Validate independent Prefill and Decode Executor evidence for PD."""

    instances = _mapping(runtime["instances"], "runtime instances")
    _exact_keys(instances, ("prefill", "decode"), "runtime instances")
    for role in ("prefill", "decode"):
        instance = _mapping(instances[role], f"runtime instance {role}")
        _exact_keys(
            instance,
            ("scheduler_pipeline_depth", "scheduler_pipeline_depth_evidence"),
            f"runtime instance {role}",
        )
        _integer(
            instance["scheduler_pipeline_depth"],
            f"runtime instance {role} scheduler_pipeline_depth",
            minimum=1,
        )
        evidence = _mapping(
            instance["scheduler_pipeline_depth_evidence"],
            f"runtime instance {role} scheduler_pipeline_depth_evidence",
        )
        _exact_keys(
            evidence,
            (
                "source_kind", "source_field", "executor_class",
                "pipeline_parallel_size", "runtime_version", "source_identity",
            ),
            f"runtime instance {role} scheduler_pipeline_depth_evidence",
        )
        if evidence["source_kind"] != "runtime_executor_property":
            raise ValueError(
                f"runtime instance {role} Scheduler depth source_kind is invalid")
        if evidence["source_field"] != "max_concurrent_batches":
            raise ValueError(
                f"runtime instance {role} Scheduler depth source_field is invalid")
        _nonempty_string(
            evidence["executor_class"],
            f"runtime instance {role} Scheduler depth executor_class",
        )
        _integer(
            evidence["pipeline_parallel_size"],
            f"runtime instance {role} Scheduler depth pipeline_parallel_size",
            minimum=1,
        )
        if evidence["pipeline_parallel_size"] != runtime[
                "pipeline_parallel_size"]:
            raise ValueError(
                f"runtime instance {role} Scheduler depth PP disagrees with "
                "runtime")
        _nonempty_string(
            evidence["runtime_version"],
            f"runtime instance {role} Scheduler depth runtime_version",
        )
        if evidence["runtime_version"] != runtime["runtime_version"]:
            raise ValueError(
                f"runtime instance {role} Scheduler depth version disagrees "
                "with runtime")
        _nonempty_string(
            evidence["source_identity"],
            f"runtime instance {role} Scheduler depth source_identity",
        )
        if evidence["source_identity"] != runtime["source_commit"]:
            raise ValueError(
                f"runtime instance {role} Scheduler depth source identity "
                "disagrees with runtime")


def _validate_scheduler_steps_v3(
    steps: Sequence[Mapping[str, object]],
    requests: Sequence[CanonicalRequestRow],
    *,
    include_terminal_output_token: bool,
) -> None:
    base_fields = (
        "schema_version", "step_index", "step_id", "source_iteration_id",
        "step_start_ns", "schedule_end_ns", "update_start_ns",
        "step_end_ns", "scheduled_requests",
        "scheduled_request_count", "prefill_request_count",
        "decode_request_count", "prefill_tokens", "decode_tokens",
        "scheduled_tokens", "finished_request_indexes",
        "cumulative_scheduled_tokens", "cumulative_completed_requests",
    )
    graph_runtime_fields = (
        "configured_cudagraph_mode", "runtime_cudagraph_mode",
        "num_unpadded_tokens", "capture_bucket_tokens",
        "capture_limit_tokens", "padded_total_tokens", "padding_tokens",
        "graph_fallback_reason", "schedule_start_ns",
        "device_execution_start_ns", "device_execution_end_ns",
        "device_execution_clock", "device_execution_source",
        "scheduler_update_start_ns", "scheduler_update_end_ns",
        "in_flight_step_count",
    )
    has_graph_runtime = "configured_cudagraph_mode" in steps[0]
    expected_fields = (
        base_fields + graph_runtime_fields if has_graph_runtime else base_fields
    )
    request_work = [0 for _ in requests]
    finished = set()
    cumulative_tokens = 0
    previous_timestamps = (0, 0, 0, 0)
    previous_iteration = -1
    for expected_index, value in enumerate(steps):
        step = _mapping(value, f"scheduler step {expected_index}")
        _exact_keys(step, expected_fields, f"scheduler step {expected_index}")
        _schema_version(
            step["schema_version"],
            f"scheduler step {expected_index}",
            expected=SCHEDULER_CAPTURE_SCHEMA_VERSION,
        )
        _integer(step["step_index"], "scheduler step_index", minimum=0)
        if step["step_index"] != expected_index:
            raise ValueError("scheduler step_index must be contiguous from zero")
        _nonempty_string(step["step_id"], "scheduler step_id")
        _integer(
            step["source_iteration_id"],
            "scheduler source_iteration_id",
            minimum=0,
        )
        if step["source_iteration_id"] <= previous_iteration:
            raise ValueError("scheduler source_iteration_id must strictly increase")
        previous_iteration = step["source_iteration_id"]
        timestamp_fields = (
            "step_start_ns", "schedule_end_ns", "update_start_ns",
            "step_end_ns",
        )
        for field in timestamp_fields:
            _integer(step[field], f"scheduler {field}", minimum=0)
        timestamps = tuple(step[field] for field in timestamp_fields)
        if expected_index == 0 and step["step_start_ns"] != 0:
            raise ValueError("first scheduler step_start_ns must define zero origin")
        if timestamps != tuple(sorted(timestamps)):
            raise ValueError("scheduler timing boundaries must be nondecreasing")
        if any(
            current < previous
            for current, previous in zip(
                timestamps, previous_timestamps, strict=True
            )
        ):
            raise ValueError(
                "scheduler timing boundaries must advance monotonically"
            )
        previous_timestamps = timestamps
        scheduled = _sequence(
            step["scheduled_requests"], "scheduler scheduled_requests"
        )
        request_indexes = set()
        prefill_request_count = 0
        decode_request_count = 0
        prefill_tokens = 0
        decode_tokens = 0
        for work_index, raw_work in enumerate(scheduled):
            work = _mapping(
                raw_work,
                f"scheduler step {expected_index} request work {work_index}",
            )
            _exact_keys(
                work,
                (
                    "request_idx", "prefill_tokens", "decode_tokens",
                    "scheduled_tokens",
                ),
                "scheduler request work",
            )
            _integer(work["request_idx"], "scheduler request_idx", minimum=0)
            if work["request_idx"] >= len(requests):
                raise ValueError("scheduler request_idx exceeds requests.csv")
            if work["request_idx"] in request_indexes:
                raise ValueError("scheduler step schedules one request twice")
            request_indexes.add(work["request_idx"])
            for field in ("prefill_tokens", "decode_tokens"):
                _integer(
                    work[field], f"scheduler request {field}", minimum=0
                )
            _integer(work["scheduled_tokens"], "scheduler request scheduled_tokens", minimum=1)
            if work["scheduled_tokens"] != (
                work["prefill_tokens"] + work["decode_tokens"]
            ):
                raise ValueError(
                    "scheduler request token split disagrees with scheduled_tokens"
                )
            request_work[work["request_idx"]] += work["scheduled_tokens"]
            if work["prefill_tokens"]:
                prefill_request_count += 1
                prefill_tokens += work["prefill_tokens"]
            if work["decode_tokens"]:
                decode_request_count += 1
                decode_tokens += work["decode_tokens"]
        derived = {
            "scheduled_request_count": len(scheduled),
            "prefill_request_count": prefill_request_count,
            "decode_request_count": decode_request_count,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "scheduled_tokens": prefill_tokens + decode_tokens,
        }
        for field, expected in derived.items():
            _integer(step[field], f"scheduler {field}", minimum=0)
            if step[field] != expected:
                raise ValueError(f"scheduler {field} disagrees with request work")
        cumulative_tokens += derived["scheduled_tokens"]
        if step["cumulative_scheduled_tokens"] != cumulative_tokens:
            raise ValueError("scheduler cumulative token count is inconsistent")
        finished_indexes = _sequence(
            step["finished_request_indexes"],
            "scheduler finished_request_indexes",
        )
        for request_idx in finished_indexes:
            _integer(request_idx, "scheduler finished request_idx", minimum=0)
            if request_idx >= len(requests):
                raise ValueError("finished request_idx exceeds requests.csv")
            if request_idx in finished:
                raise ValueError("scheduler request is finished more than once")
            finished.add(request_idx)
        _integer(
            step["cumulative_completed_requests"],
            "scheduler cumulative_completed_requests",
            minimum=0,
        )
        if step["cumulative_completed_requests"] != len(finished):
            raise ValueError("scheduler cumulative completion count is inconsistent")
        if has_graph_runtime:
            _validate_scheduler_graph_runtime(step, expected_index)
    if finished != set(range(len(requests))):
        raise ValueError("scheduler steps must finish every request exactly once")
    for request_idx, request in enumerate(requests):
        expected_output_tokens = (
            request.output_tokens
            if include_terminal_output_token
            else max(request.output_tokens - 1, 0)
        )
        expected_work = (
            request.input_tokens - request.cached_hit_tokens + expected_output_tokens
        )
        if request_work[request_idx] != expected_work:
            raise ValueError(
                f"scheduler token conservation failed for request {request_idx}"
            )


def _validate_scheduler_graph_runtime(
    step: Mapping[str, object], step_index: int
) -> None:
    _enum(
        step["configured_cudagraph_mode"],
        ("FULL", "PIECEWISE", "FULL_AND_PIECEWISE", "NONE"),
        f"scheduler step {step_index} configured_cudagraph_mode",
    )
    runtime_mode = step["runtime_cudagraph_mode"]
    _enum(
        runtime_mode,
        ("FULL", "PIECEWISE", "NONE"),
        f"scheduler step {step_index} runtime_cudagraph_mode",
    )
    for field in (
        "num_unpadded_tokens", "capture_limit_tokens",
        "padded_total_tokens", "padding_tokens", "in_flight_step_count",
    ):
        _integer(step[field], f"scheduler step {step_index} {field}", minimum=0)
    if step["capture_limit_tokens"] < 1:
        raise ValueError("scheduler capture_limit_tokens must be positive")
    if step["num_unpadded_tokens"] != step["scheduled_tokens"]:
        raise ValueError("scheduler num_unpadded_tokens is inconsistent")
    if step["padded_total_tokens"] != (
        step["scheduled_tokens"] + step["padding_tokens"]
    ):
        raise ValueError("scheduler padded_total_tokens is inconsistent")
    if not 1 <= step["in_flight_step_count"] <= 2:
        raise ValueError("scheduler in-flight depth is outside depth=2")
    fallback_reason = step["graph_fallback_reason"]
    if runtime_mode == "NONE":
        if step["capture_bucket_tokens"] is not None:
            raise ValueError("NONE scheduler runtime row has a capture bucket")
        _enum(
            fallback_reason,
            ("above_capture_limit", "runtime_dispatch_no_matching_graph"),
            f"scheduler step {step_index} graph_fallback_reason",
        )
        above_limit = step["scheduled_tokens"] > step["capture_limit_tokens"]
        if above_limit != (fallback_reason == "above_capture_limit"):
            raise ValueError(
                "scheduler Graph fallback reason disagrees with capture limit"
            )
    elif runtime_mode in ("FULL", "PIECEWISE"):
        _integer(
            step["capture_bucket_tokens"],
            f"scheduler step {step_index} capture_bucket_tokens",
            minimum=1,
        )
        if step["capture_bucket_tokens"] != step["padded_total_tokens"]:
            raise ValueError("scheduler runtime capture bucket is inconsistent")
        if step["capture_bucket_tokens"] > step["capture_limit_tokens"]:
            raise ValueError("scheduler Graph capture bucket exceeds capture limit")
        if fallback_reason is not None:
            raise ValueError("Graph scheduler runtime row has a fallback reason")
    else:
        raise ValueError(f"unsupported Scheduler runtime Graph mode: {runtime_mode!r}")
    for field in (
        "schedule_start_ns", "schedule_end_ns", "device_execution_start_ns",
        "device_execution_end_ns", "scheduler_update_start_ns",
        "scheduler_update_end_ns",
    ):
        _integer(step[field], f"scheduler step {step_index} {field}", minimum=0)
    if not (
        step["schedule_start_ns"] <= step["schedule_end_ns"]
        <= step["device_execution_start_ns"]
        <= step["device_execution_end_ns"]
        <= step["scheduler_update_start_ns"]
        <= step["scheduler_update_end_ns"]
    ):
        raise ValueError("scheduler execution timing boundaries are not causal")
    if (
        step["schedule_start_ns"] != step["step_start_ns"]
        or step["scheduler_update_start_ns"] != step["update_start_ns"]
        or step["scheduler_update_end_ns"] != step["step_end_ns"]
    ):
        raise ValueError("scheduler runtime timing aliases are inconsistent")
    if step["device_execution_clock"] != "time.perf_counter_ns":
        raise ValueError("scheduler device execution clock is invalid")
    if step["device_execution_source"] != "host_monotonic_executor_rpc_envelope":
        raise ValueError("scheduler device execution source is invalid")


def _validate_scheduler_steps_v4(
    steps: Sequence[Mapping[str, object]],
    requests: Sequence[CanonicalRequestRow],
    expected_rank_ids: tuple[int, ...],
    *,
    schema_version: int,
    configured_depth: int | None,
) -> None:
    """Validate runtime Graph observations while reusing v3 token semantics."""

    v3_fields = (
        "schema_version", "step_index", "step_id", "source_iteration_id",
        "step_start_ns", "schedule_end_ns", "update_start_ns",
        "step_end_ns", "scheduled_requests",
        "scheduled_request_count", "prefill_request_count",
        "decode_request_count", "prefill_tokens", "decode_tokens",
        "scheduled_tokens", "finished_request_indexes",
        "cumulative_scheduled_tokens", "cumulative_completed_requests",
    )
    v4_fields = v3_fields + (
        "schedule_start_ns", "scheduler_update_start_ns",
        "scheduler_update_end_ns", "in_flight_step_count",
        "device_execution_clock", "device_execution_start_ns",
        "device_execution_end_ns", "device_execution_source",
        "rank_observations",
    )
    projected_v3_steps = []
    max_in_flight = 0
    for expected_index, value in enumerate(steps):
        step = _mapping(value, f"scheduler step {expected_index}")
        _exact_keys(step, v4_fields, f"scheduler step {expected_index}")
        _schema_version(
            step["schema_version"],
            f"scheduler step {expected_index}",
            expected=schema_version,
        )
        for field in (
            "schedule_start_ns", "scheduler_update_start_ns",
            "scheduler_update_end_ns", "device_execution_start_ns",
            "device_execution_end_ns",
        ):
            _integer(step[field], f"scheduler {field}", minimum=0)
        if step["schedule_start_ns"] != step["step_start_ns"]:
            raise ValueError("scheduler schedule_start_ns disagrees with step_start_ns")
        if step["scheduler_update_start_ns"] != step["update_start_ns"]:
            raise ValueError(
                "scheduler scheduler_update_start_ns disagrees with update_start_ns"
            )
        if step["scheduler_update_end_ns"] != step["step_end_ns"]:
            raise ValueError(
                "scheduler scheduler_update_end_ns disagrees with step_end_ns"
            )
        if step["device_execution_start_ns"] > step["device_execution_end_ns"]:
            raise ValueError("scheduler device execution envelope is inverted")
        if step["device_execution_clock"] != "time.perf_counter_ns":
            raise ValueError("scheduler device execution clock is invalid")
        if step["device_execution_source"] != "host_monotonic_executor_rpc_envelope":
            raise ValueError("scheduler device execution source is invalid")
        _integer(
            step["in_flight_step_count"],
            "scheduler in_flight_step_count",
            minimum=1,
        )
        if (configured_depth is not None
                and step["in_flight_step_count"] > configured_depth):
            raise ValueError(
                "scheduler in_flight_step_count exceeds configured depth")
        max_in_flight = max(max_in_flight, step["in_flight_step_count"])
        observations = _sequence(
            step["rank_observations"],
            f"scheduler step {expected_index} rank_observations",
        )
        observed_ranks = []
        for position, value in enumerate(observations):
            observation = _mapping(
                value,
                f"scheduler step {expected_index} rank observation {position}",
            )
            _validate_runtime_graph_observation(
                observation,
                expected_index,
                step["step_id"],
                step["scheduled_tokens"],
            )
            observed_ranks.append(observation["rank"])
        if tuple(observed_ranks) != expected_rank_ids:
            raise ValueError(
                "scheduler rank observations must cover every runtime rank in order"
            )
        projected = {field: step[field] for field in v3_fields}
        projected["schema_version"] = SCHEDULER_CAPTURE_SCHEMA_VERSION
        projected_v3_steps.append(projected)
    _validate_scheduler_steps_v3(
        projected_v3_steps,
        requests,
        include_terminal_output_token=True,
    )


def _validate_runtime_graph_observation(
    observation: Mapping[str, object],
    step_index: int,
    step_id: object,
    scheduled_tokens: object,
) -> None:
    _exact_keys(
        observation,
        (
            "schema_version", "rank", "trace_step_id",
            "configured_cudagraph_mode", "runtime_cudagraph_mode",
            "dispatch_input_key", "selected_graph_key",
            "scheduled_tokens", "num_unpadded_tokens", "padded_total_tokens",
            "padding_tokens",
            "graph_fallback_reason", "worker_execution_clock",
            "worker_execution_start_ns", "worker_execution_end_ns",
            "worker_execution_source", "source",
        ),
        f"scheduler step {step_index} rank observation",
    )
    _schema_version(
        observation["schema_version"],
        f"scheduler step {step_index} rank observation",
        expected=1,
    )
    _integer(
        observation["rank"],
        f"scheduler step {step_index} rank observation rank",
        minimum=0,
    )
    if observation["trace_step_id"] != step_id:
        raise ValueError("scheduler rank observation trace_step_id disagrees with step")
    if observation["configured_cudagraph_mode"] != "FULL_AND_PIECEWISE":
        raise ValueError("scheduler rank observation configured CUDAGraph mode is invalid")
    _enum(
        observation["runtime_cudagraph_mode"],
        ("FULL", "PIECEWISE", "NONE"),
        "scheduler rank observation runtime CUDAGraph mode",
    )
    _validate_dispatch_key(
        observation["dispatch_input_key"],
        "scheduler rank observation dispatch_input_key",
    )
    for field in (
        "scheduled_tokens", "num_unpadded_tokens", "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns", "worker_execution_end_ns",
    ):
        _integer(
            observation[field],
            f"scheduler rank observation {field}",
            minimum=0,
        )
    if observation["scheduled_tokens"] != scheduled_tokens:
        raise ValueError("scheduler rank observation tokens disagree with step")
    if observation["num_unpadded_tokens"] != scheduled_tokens:
        raise ValueError("scheduler rank observation unpadded tokens disagree with step")
    if observation["padded_total_tokens"] != (
        observation["num_unpadded_tokens"] + observation["padding_tokens"]
    ):
        raise ValueError("scheduler rank observation padding conservation failed")
    mode = observation["runtime_cudagraph_mode"]
    if mode == "NONE":
        if observation["selected_graph_key"] is not None:
            raise ValueError("NONE runtime CUDAGraph mode requires null selected_graph_key")
        _nonempty_string(
            observation["graph_fallback_reason"],
            "NONE runtime CUDAGraph mode fallback reason",
        )
    elif mode in ("FULL", "PIECEWISE"):
        _validate_dispatch_key(
            observation["selected_graph_key"],
            "scheduler rank observation selected_graph_key",
        )
        if observation["graph_fallback_reason"] is not None:
            raise ValueError("selected runtime CUDAGraph mode requires null fallback")
    else:
        raise ValueError(f"unsupported runtime CUDAGraph mode: {mode!r}")
    if observation["worker_execution_clock"] != "time.perf_counter_ns":
        raise ValueError("scheduler worker execution clock is invalid")
    if observation["worker_execution_start_ns"] > observation["worker_execution_end_ns"]:
        raise ValueError("scheduler worker execution envelope is inverted")
    if observation["worker_execution_source"] != "vllm_v1_worker_execute_model_envelope":
        raise ValueError("scheduler worker execution source is invalid")
    if observation["source"] != "vllm_v1_worker_cudagraph_dispatch":
        raise ValueError("scheduler runtime Graph source is invalid")


def _validate_dispatch_key(value: object, description: str) -> None:
    key = _mapping(value, description)
    _exact_keys(key, ("num_tokens", "uniform_decode"), description)
    _integer(key["num_tokens"], f"{description} num_tokens", minimum=1)
    if not isinstance(key["uniform_decode"], bool):
        raise TypeError(f"{description} uniform_decode must be boolean")


def _validate_alignment(
    alignment: Mapping[str, object],
    manifest_entry: Mapping[str, object],
    hardware_id: object,
    historical_raw_source_shas: set[str],
    raw_archive_ids: set[object],
    request_count: int,
    expected_ranks: tuple[int, ...],
    schema_version: int,
) -> Mapping[str, object]:
    _exact_keys(alignment, ("schema_version", "kind", "operator_signatures", "launch_signatures", "phases"), "alignment_steps")
    _schema_version(
        alignment["schema_version"],
        "alignment_steps",
        expected=schema_version,
    )
    if alignment["kind"] != "sharegpt_alignment_steps":
        raise ValueError("alignment_steps kind must be sharegpt_alignment_steps")
    _reject_timing(alignment, "alignment_steps")
    allow_empty = not any(
        _mapping(value, "alignment phase")["status"] == "available"
        for value in _mapping(alignment["phases"], "alignment phases").values()
    )
    operator_signatures = _signature_table(
        alignment["operator_signatures"], "operator", allow_empty=allow_empty
    )
    launch_signatures = _signature_table(
        alignment["launch_signatures"], "launch", allow_empty=allow_empty
    )
    for value in operator_signatures.values():
        _exact_keys(value, ("operation", "attrs"), "operator signature")
        _nonempty_string(value["operation"], "operator signature operation")
        _mapping(value["attrs"], "operator signature attrs")
    for value in launch_signatures.values():
        _validate_launch_signature(value, hardware_id)
    phases = _mapping(alignment["phases"], "alignment_steps phases")
    _exact_keys(phases, PHASES, "alignment_steps phases")
    manifest_phases = _mapping(manifest_entry["phases"], "manifest alignment phases")
    steps: dict[str, AlignmentStepSample | None] = {}
    for phase in PHASES:
        value = _mapping(phases[phase], f"alignment phase {phase}")
        _exact_keys(value, ("status", "sample"), f"alignment phase {phase}")
        declared = _mapping(manifest_phases[phase], f"manifest phase {phase}")
        if value["status"] != declared["status"]:
            raise ValueError(f"alignment phase {phase} status disagrees with manifest")
        if value["status"] == "not_observed":
            if value["sample"] is not None:
                raise ValueError(f"not_observed phase {phase} requires null sample")
            steps[phase] = None
            continue
        if value["status"] != "available":
            raise ValueError(f"alignment phase {phase} has unknown status")
        sample = _validate_alignment_sample(_mapping(value["sample"], f"alignment phase {phase} sample"), phase, operator_signatures, launch_signatures, historical_raw_source_shas, raw_archive_ids, request_count)
        if sample.rank_ids != expected_ranks:
            raise ValueError(f"alignment phase {phase} ranks disagree with runtime topology")
        declared_ranks = _rank_ids(declared["rank_ids"], f"manifest phase {phase} rank_ids")
        if sample.rank_ids != declared_ranks:
            raise ValueError(f"alignment phase {phase} rank_ids disagree with manifest")
        steps[phase] = sample
    return {"steps": steps, "operator_signatures": operator_signatures, "launch_signatures": launch_signatures}


def _signature_table(
    value: object, description: str, *, allow_empty: bool
) -> Mapping[str, Mapping[str, object]]:
    table = _mapping(value, f"{description} signatures")
    if not table and not allow_empty:
        raise ValueError(f"{description} signatures cannot be empty")
    normalized = {}
    for identifier, payload in table.items():
        _sha256(identifier, f"{description} signature ID")
        signature = _mapping(payload, f"{description} signature {identifier}")
        if signature_id(signature) != identifier:
            raise ValueError(f"{description} signature ID disagrees with canonical content")
        normalized[identifier] = signature
    return normalized


# The launch-geometry ABI each hardware's profiler records, and whether its
# legacy launch projections may leave the geometry unavailable (only the C500
# mcTracer projection could not recover it).
_LAUNCH_GEOMETRY_ABIS = {
    "nvidia_a800": ("cuda_dim3", False),
    "nvidia_h200": ("cuda_dim3", False),
    "metax_c500": ("cuda_dim3", True),
    "ascend_910b": ("ascend_block_dim", False),
}


def _validate_launch_signature(signature: Mapping[str, object], hardware_id: object) -> None:
    _exact_keys(signature, ("kernel_name", "geometry", "stream", "device", "op_family"), "launch signature")
    _nonempty_string(signature["kernel_name"], "launch signature kernel_name")
    _optional_scalar(signature["stream"], "launch signature stream")
    _optional_integer(signature["device"], "launch signature device", minimum=0)
    _optional_nonempty_string(signature["op_family"], "launch signature op_family")
    geometry = _mapping(signature["geometry"], "launch signature geometry")
    abi = (_LAUNCH_GEOMETRY_ABIS[hardware_id]
           if isinstance(hardware_id, str) and hardware_id in _LAUNCH_GEOMETRY_ABIS
           else None)
    if geometry["kind"] == "unavailable":
        if abi is None or not abi[1]:
            raise ValueError(
                "unavailable launch geometry is reserved for C500 legacy projections"
            )
        _exact_keys(geometry, ("kind",), "unavailable launch geometry")
        return
    if abi is not None and abi[0] == "cuda_dim3":
        _exact_keys(geometry, ("kind", "grid", "block", "shared_memory_bytes", "registers_per_thread"), "cuda launch geometry")
        if geometry["kind"] != "cuda_dim3":
            raise ValueError("NVIDIA and MetaX launch signatures require cuda_dim3")
        _dim3(geometry["grid"], "cuda launch grid")
        _dim3(geometry["block"], "cuda launch block")
        _optional_integer(geometry["shared_memory_bytes"], "cuda launch shared_memory_bytes", minimum=0)
        _optional_integer(geometry["registers_per_thread"], "cuda launch registers_per_thread", minimum=0)
    elif abi is not None and abi[0] == "ascend_block_dim":
        _exact_keys(geometry, ("kind", "block_dim", "mix_block_dim"), "Ascend launch geometry")
        if geometry["kind"] != "ascend_block_dim":
            raise ValueError("Ascend launch signatures require ascend_block_dim")
        block_dim = _integer(
            geometry["block_dim"], "Ascend launch block_dim", minimum=0
        )
        if block_dim == 0 and not (
            signature["kernel_name"]
            == "aclnnBincount_BincountAiCpu_Bincount"
            and signature["op_family"] == "AI_CPU/Bincount"
        ):
            raise ValueError(
                "Ascend block_dim=0 is reserved for the proven AICPU Bincount task"
            )
        _optional_integer(geometry["mix_block_dim"], "Ascend launch mix_block_dim", minimum=0)
    else:
        raise ValueError("unsupported hardware for launch signature")


def _validate_alignment_sample(sample: Mapping[str, object], phase: str, operator_signatures: Mapping[str, Mapping[str, object]], launch_signatures: Mapping[str, Mapping[str, object]], historical_raw_source_shas: set[str], raw_archive_ids: set[object], request_count: int) -> AlignmentStepSample:
    _exact_keys(sample, ("source_step_id", "source_execution_order", "scheduler", "scheduler_field_provenance", "rank_ids", "operators", "launches", "association", "coverage", "provenance", "sequence_sha256"), f"alignment {phase} sample")
    _nonempty_string(sample["source_step_id"], f"alignment {phase} source_step_id")
    _integer(sample["source_execution_order"], f"alignment {phase} source_execution_order", minimum=0)
    ranks = _rank_ids(sample["rank_ids"], f"alignment {phase} rank_ids")
    if not ranks:
        raise ValueError(f"alignment {phase} requires at least one rank")
    if sample["association"] not in ("step_order_only", "explicit"):
        raise ValueError("alignment association must be step_order_only or explicit")
    if sample["coverage"] != "complete":
        raise ValueError("published alignment samples must have complete coverage")
    scheduler = _mapping(sample["scheduler"], f"alignment {phase} scheduler")
    _validate_scheduler(scheduler, phase, request_count)
    scheduler_provenance = _mapping(
        sample["scheduler_field_provenance"],
        f"alignment {phase} scheduler_field_provenance",
    )
    if dict(scheduler_provenance) != _SCHEDULER_FIELD_PROVENANCE:
        raise ValueError(
            "alignment scheduler_field_provenance must use the canonical raw/derived boundary"
        )
    operators = _rank_event_groups(sample["operators"], ranks, "operator", operator_signatures)
    launches = _rank_event_groups(sample["launches"], ranks, "launch", launch_signatures)
    provenance = _mapping(sample["provenance"], f"alignment {phase} provenance")
    _exact_keys(provenance, ("source_archive_ids", "raw_source_set_sha256"), f"alignment {phase} provenance")
    archive_ids = _sequence(provenance["source_archive_ids"], f"alignment {phase} source_archive_ids")
    if not archive_ids or len(archive_ids) != len(set(archive_ids)):
        raise ValueError("alignment provenance requires unique, non-empty source_archive_ids")
    for archive_id in archive_ids:
        if archive_id not in raw_archive_ids:
            raise ValueError("alignment provenance references unknown raw archive")
    if provenance["raw_source_set_sha256"] not in historical_raw_source_shas:
        raise ValueError("alignment provenance raw_source_set_sha256 disagrees with manifest")
    sequence = _mapping(sample["sequence_sha256"], f"alignment {phase} sequence_sha256")
    _exact_keys(sequence, ("operators", "launches"), f"alignment {phase} sequence_sha256")
    for kind, groups in (("operators", operators), ("launches", launches)):
        hashes = _mapping(sequence[kind], f"alignment {phase} {kind} sequence_sha256")
        if set(hashes) != {str(rank) for rank in ranks}:
            raise ValueError(f"alignment {phase} {kind} sequence hashes must cover every rank")
        for group in groups:
            rank = group["rank"]
            if hashes[str(rank)] != _canonical_json_sha256(group["events"]):
                raise ValueError(f"alignment {phase} {kind} sequence SHA disagrees for rank {rank}")
    return AlignmentStepSample(sample["source_step_id"], sample["source_execution_order"], ranks, scheduler, tuple(operators), tuple(launches), sample["association"], sample["coverage"], provenance, sequence)


def _validate_scheduler(value: Mapping[str, object], phase: str, request_count: int) -> None:
    _exact_keys(value, _SCHEDULE_FIELDS, "alignment scheduler")
    _nonempty_string(value["step_id"], "alignment scheduler step_id")
    _nonempty_string(value["pool_name"], "alignment scheduler pool_name")
    scheduled = _sequence(value["requests"], "alignment scheduler requests")
    if not scheduled:
        raise ValueError("representative alignment step must be work-bearing")
    normalized_requests = []
    for index, item in enumerate(scheduled):
        request = _mapping(item, f"alignment scheduled request {index}")
        _exact_keys(request, _SCHEDULED_REQUEST_FIELDS, f"alignment scheduled request {index}")
        request_id = request["request_id"]
        _nonempty_string(request_id, "alignment request_id")
        match = re.fullmatch(r"request-(?P<index>[0-9]+)", request_id)
        if match is None or int(match.group("index")) >= request_count:
            raise ValueError("alignment request_id is outside requests.csv")
        _enum(request["phase"], ("prefill", "decode"), "alignment request phase")
        token_fields = (
            "input_tokens", "output_tokens", "computed_tokens",
            "recomputed_tokens", "prefix_replay_tokens",
            "trace_prefix_hit_tokens", "scheduler_prefix_hit_tokens",
            "raw_model_output_tokens", "mtp_proposed_tokens",
            "mtp_verified_tokens", "mtp_accepted_tokens",
            "mtp_proposer_tokens", "mtp_next_proposed_tokens",
        )
        for field in token_fields:
            _integer(request[field], f"alignment request {field}", minimum=0)
        scheduled_tokens = request["input_tokens"] + request["output_tokens"]
        if scheduled_tokens == 0:
            raise ValueError("alignment request must execute at least one token")
        if request["recomputed_tokens"] + request["prefix_replay_tokens"] > scheduled_tokens:
            raise ValueError("alignment recompute and prefix replay exceed scheduled work")
        if request["scheduler_prefix_hit_tokens"] > request["trace_prefix_hit_tokens"]:
            raise ValueError("alignment scheduler prefix hit exceeds trace prefix hit")
        if request["scheduler_prefix_hit_tokens"] > request["computed_tokens"]:
            raise ValueError("alignment scheduler prefix hit exceeds computed tokens")
        if request["prefix_scheduler_delta_tokens"] != (
            request["trace_prefix_hit_tokens"]
            - request["scheduler_prefix_hit_tokens"]
        ):
            raise ValueError("alignment prefix_scheduler_delta_tokens is inconsistent")
        if request["mtp_verified_tokens"] > request["mtp_proposed_tokens"]:
            raise ValueError("alignment MTP verified tokens exceed proposed tokens")
        if request["mtp_accepted_tokens"] > request["mtp_verified_tokens"]:
            raise ValueError("alignment MTP accepted tokens exceed verified tokens")
        if request["mtp_rejected_tokens"] != (
            request["mtp_verified_tokens"] - request["mtp_accepted_tokens"]
        ):
            raise ValueError("alignment mtp_rejected_tokens is inconsistent")
        if not isinstance(request["resumed_from_preemption"], bool):
            raise TypeError("alignment resumed_from_preemption must be boolean")
        for field in (
            "mtp_proposer_mode", "mtp_acceptance_source_kind",
            "mtp_acceptance_source_id",
        ):
            _nonempty_string(request[field], f"alignment request {field}")
        if request["mtp_acceptance_source_kind"] == "disabled":
            if request["mtp_proposer_mode"] != "disabled" or request[
                "mtp_acceptance_source_id"
            ] != "disabled":
                raise ValueError("disabled alignment MTP requires disabled mode and source")
            if any(request[field] for field in (
                "mtp_proposed_tokens", "mtp_verified_tokens",
                "mtp_accepted_tokens", "mtp_proposer_tokens",
                "mtp_next_proposed_tokens",
            )):
                raise ValueError("disabled alignment MTP cannot carry speculative tokens")
        elif request["mtp_acceptance_source_kind"] in (
            "explicit_assumption", "replay_observation",
        ):
            if request["mtp_proposer_mode"] != "padded":
                raise ValueError("enabled alignment MTP requires padded proposer mode")
        else:
            raise ValueError("unknown alignment MTP acceptance source kind")
        normalized_requests.append(request)
    request_ids = [request["request_id"] for request in normalized_requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("alignment Scheduler request IDs must be unique")
    finished = _string_id_sequence(
        value["finished_request_ids"], "alignment finished_request_ids"
    )
    preempted = _string_id_sequence(
        value["preempted_request_ids"], "alignment preempted_request_ids"
    )
    if not set(finished).issubset(request_ids):
        raise ValueError("alignment finished requests must be scheduled")
    if set(preempted) & (set(request_ids) | set(finished)):
        raise ValueError("alignment preempted requests must be disjoint")
    _enum(
        value["attention_kv_layout"],
        (None, "contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"),
        "alignment attention_kv_layout",
    )
    request_phases = {request["phase"] for request in normalized_requests}
    expected_phase = "mixed" if len(request_phases) > 1 else next(iter(request_phases))
    if expected_phase != phase:
        raise ValueError("representative phase disagrees with scheduled request phases")


def _string_id_sequence(value: object, description: str) -> tuple[str, ...]:
    values = _sequence(value, description)
    for item in values:
        _nonempty_string(item, description)
    if len(values) != len(set(values)):
        raise ValueError(f"{description} must not contain duplicates")
    return tuple(values)


def _rank_event_groups(value: object, ranks: tuple[int, ...], kind: str, signatures: Mapping[str, Mapping[str, object]]) -> list[Mapping[str, object]]:
    groups = _sequence(value, f"alignment {kind} groups")
    if len(groups) != len(ranks):
        raise ValueError(f"alignment {kind} groups must cover every rank once")
    result = []
    for position, value in enumerate(groups):
        group = _mapping(value, f"alignment {kind} group {position}")
        _exact_keys(group, ("rank", "events"), f"alignment {kind} group {position}")
        if group["rank"] != ranks[position]:
            raise ValueError(f"alignment {kind} groups must be ordered by rank_ids")
        events = _sequence(group["events"], f"alignment {kind} events rank {group['rank']}")
        if not events:
            raise ValueError(f"alignment {kind} sequence cannot be empty")
        for sequence_idx, value in enumerate(events):
            event = _mapping(value, f"alignment {kind} event {sequence_idx}")
            if kind == "operator":
                expected = ("order", "signature_id", "layer")
            elif kind == "launch":
                expected = ("order", "signature_id")
            else:
                raise AssertionError(f"unsupported alignment event kind: {kind!r}")
            _exact_keys(event, expected, f"alignment {kind} event {sequence_idx}")
            if event["order"] != sequence_idx:
                raise ValueError(f"alignment {kind} event order must be contiguous from zero")
            if event["signature_id"] not in signatures:
                raise ValueError(f"alignment {kind} event references unknown signature")
            if kind == "operator":
                _optional_integer(event["layer"], "alignment operator event layer", minimum=0)
        result.append({"rank": group["rank"], "events": list(events)})
    return result


def _validate_case_files(
    root: Path,
    metrics_entry: Mapping[str, object],
    schema_version: int,
) -> None:
    expected = {
        Path("manifest.json"), Path("requests.csv"), Path(ALIGNMENT_FILENAME),
    }
    if metrics_entry["status"] == "available":
        expected.add(Path("metrics.json"))
    if schema_version in _SCHEDULER_CAPTURE_SCHEMA_VERSIONS:
        expected.add(Path(SCHEDULER_STEPS_FILENAME))
    actual = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        extra = sorted(str(path) for path in actual - expected)
        raise ValueError(f"compact case file set mismatch; missing={missing}, extra={extra}")


def _read_json_object(path: Path) -> Mapping[str, object]:
    return _mapping(_read_json_value(path), f"JSON file {path}")


def _read_json_value(path: Path) -> object:
    if not path.is_file():
        raise ValueError(f"required JSON file does not exist: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
    except UnicodeDecodeError as error:
        raise ValueError(f"JSON file must be UTF-8: {path}") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON file {path}: {error}") from error


def _read_json_bytes(content: bytes, label: str) -> object:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} content must be UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {label} content: {error}") from error


def _safe_case_path(root: Path, relative_path: object) -> Path:
    _relative_path(relative_path, "case relative path")
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"declared compact case file does not exist: {relative_path}")
    return path


def _verify_sha(path: Path, expected: object, description: str) -> None:
    _sha256(expected, f"{description} SHA256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{description} SHA256 disagrees with file content")


def _rank_ids(value: object, description: str) -> tuple[int, ...]:
    values = _sequence(value, description)
    result = []
    for rank in values:
        _integer(rank, description, minimum=0)
        result.append(rank)
    if result != sorted(set(result)):
        raise ValueError(f"{description} must be unique and ascending")
    return tuple(result)


def _dim3(value: object, description: str) -> None:
    values = _sequence(value, description)
    if len(values) != 3:
        raise ValueError(f"{description} must have exactly three dimensions")
    for dimension in values:
        _integer(dimension, description, minimum=1)


def _reject_timing(value: object, description: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} object key must be a string")
            separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
            tokens = tuple(
                token for token in re.split(r"[^a-z0-9]+", separated.lower())
                if token
            )
            compact = "".join(tokens)
            has_boundary_time_unit = (
                any(token in _ALIGNMENT_TIME_BOUNDARY_TOKENS for token in tokens)
                and any(token in _ALIGNMENT_TIME_UNIT_TOKENS for token in tokens)
            )
            if (
                any(token in _FORBIDDEN_ALIGNMENT_TIME_TOKENS for token in tokens)
                or any(part in compact for part in ("timestamp", "duration", "latency", "elapsed"))
                or has_boundary_time_unit
            ):
                raise ValueError(f"{description} must not contain timing field {key!r}")
            _reject_timing(nested, description)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_timing(nested, description)


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError(f"canonical JSON is invalid: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")






def _exact_keys(value: Mapping[str, object], expected: Sequence[str], description: str) -> None:
    require_exact_keys(value, expected, description)


def _json_value(value: object, description: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{description} must contain finite JSON numbers")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} keys must be strings")
            _json_value(nested, description)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _json_value(nested, description)
        return
    raise TypeError(f"{description} is not JSON data")


def _schema_version(
    value: object, description: str, *, expected: int | None = None
) -> None:
    allowed = SUPPORTED_SCHEMA_VERSIONS if expected is None else (expected,)
    if value not in allowed:
        if expected is None:
            raise ValueError(
                f"{description} schema_version must be one of "
                f"{SUPPORTED_SCHEMA_VERSIONS}"
            )
        raise ValueError(f"{description} schema_version must equal {expected}")


def _sha256(value: object, description: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{description} must be a lowercase SHA256")


def _optional_sha256(value: object, description: str) -> None:
    if value is not None:
        _sha256(value, description)


def validate_converter_source_commit(value: object) -> str:
    """Return one exact Git commit digest used to identify converter code."""

    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise ValueError(
            "converter_source_commit must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def _model_id(value: object, description: str) -> None:
    if not isinstance(value, str) or _MODEL_ID.fullmatch(value) is None:
        raise ValueError(f"{description} must be lowercase hyphenated canonical model ID")


def _nonempty_string(value: object, description: str) -> None:
    require_nonempty_string(value, description)


def _optional_nonempty_string(value: object, description: str) -> None:
    if value is not None:
        _nonempty_string(value, description)


def _relative_path(value: object, description: str) -> None:
    _nonempty_string(value, description)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) in (".", ""):
        raise ValueError(f"{description} must be a safe relative path")


def _integer(value: object, description: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")


def _optional_integer(value: object, description: str, *, minimum: int) -> None:
    if value is not None:
        _integer(value, description, minimum=minimum)


def _finite_non_negative(value: object, description: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{description} must be a finite number >= 0")


def _finite_positive(value: object, description: str) -> None:
    _finite_non_negative(value, description)
    if value <= 0:
        raise ValueError(f"{description} must be positive")


def _optional_scalar(value: object, description: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{description} must be string, integer, or null")


def _enum(value: object, choices: Sequence[object], description: str) -> None:
    if value not in choices:
        raise ValueError(f"{description} must be one of {tuple(choices)}")


def _decimal_number(value: str, field: str, request_idx: int) -> float:
    try:
        decimal = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid {field} in request row {request_idx}") from error
    if not decimal.is_finite() or decimal < 0:
        raise ValueError(f"invalid {field} in request row {request_idx}")
    result = float(decimal)
    if not math.isfinite(result):
        raise ValueError(f"invalid {field} in request row {request_idx}")
    return result


def _decimal_integer(value: str, field: str, request_idx: int) -> int:
    if _INTEGER_TEXT.fullmatch(value) is None:
        raise ValueError(f"invalid {field} in request row {request_idx}")
    result = int(value)
    if result < 0:
        raise ValueError(f"invalid {field} in request row {request_idx}")
    return result


def _format_number(value: float) -> str:
    return "0" if value == 0 else repr(float(value))
