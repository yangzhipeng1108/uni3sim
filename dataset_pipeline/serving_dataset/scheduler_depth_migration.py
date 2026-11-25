"""Metadata-only compact-case migration for recorded Scheduler capacity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .compact_candidate_writer import (
    AvailabilityPayload,
    CompactCasePayload,
    write_compact_case_candidate,
)
from .compact_case import (
    ALIGNMENT_FILENAME,
    LATEST_SCHEMA_VERSION,
    RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION,
    SCHEDULER_CAPTURE_SCHEMA_VERSION,
    load_compact_case,
    validate_converter_source_commit,
)
from .compressed_artifact import read_compressed_canonical_bytes

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
    file_sha256,
)


@dataclass(frozen=True)
class SchedulerDepthMigrationResult:
    """One strictly reloaded metadata-only candidate and its source ledger."""

    candidate_dir: Path
    canonical_case_dir: Path
    raw_runtime_path: Path
    scheduler_pipeline_depth: int
    source_file_sha256: Mapping[str, str]
    candidate_file_sha256: Mapping[str, str]


def build_scheduler_depth_metadata_candidates(
    *,
    canonical_root: str | Path,
    raw_root: str | Path,
    raw_result_name: str,
    raw_mapping_path: str | Path | None,
    hardware_id: str,
    executor_class: str,
    source_identity: str,
    candidate_root: str | Path,
    converter_source_commit: str,
    expected_case_count: int,
) -> dict[str, object]:
    """Build one hardware's complete Scheduler-depth migration candidate."""

    canonical_base = Path(canonical_root).resolve(strict=True)
    raw_base = Path(raw_root).resolve(strict=True)
    candidate_base = Path(candidate_root).resolve()
    if expected_case_count < 1:
        raise ValueError("expected case count must be positive")
    if candidate_base.exists() and not candidate_base.is_dir():
        raise ValueError("candidate root must name a directory")
    candidate_base.mkdir(parents=True, exist_ok=True)
    if raw_mapping_path is None:
        raw_index = _raw_depth_index(raw_base, raw_result_name, hardware_id)
        raw_mapping = None
        raw_mapping_sha256 = None
    else:
        raw_index, raw_mapping_sha256 = _raw_depth_mapping_index(
            Path(raw_mapping_path), raw_base, hardware_id
        )
        raw_mapping = str(Path(raw_mapping_path).resolve(strict=True))
    canonical_hardware_root = canonical_base / hardware_id
    case_dirs = sorted(
        path.parent for path in canonical_hardware_root.glob("*/*/manifest.json")
    )
    if len(case_dirs) != expected_case_count:
        raise ValueError(
            f"expected {expected_case_count} canonical cases, found {len(case_dirs)}"
        )
    canonical_keys = set()
    for case_dir in case_dirs:
        manifest = _json_object(case_dir / "manifest.json", "canonical manifest")
        identity = _mapping(manifest["identity"], "canonical identity")
        canonical_keys.add((identity["model_id"], identity["case_id"]))
    if raw_mapping_path is not None and set(raw_index) != canonical_keys:
        raise ValueError("raw mapping must cover exactly the canonical hardware cases")
    report_cases = []
    for case_dir in case_dirs:
        manifest = _json_object(case_dir / "manifest.json", "canonical manifest")
        identity = _mapping(manifest["identity"], "canonical identity")
        key = (identity["model_id"], identity["case_id"])
        if key not in raw_index:
            raise ValueError(
                f"raw Executor capacity is missing for {hardware_id}/{key[0]}/{key[1]}"
            )
        result = migrate_scheduler_depth_metadata_case(
            canonical_case_dir=case_dir,
            raw_runtime_path=raw_index[key],
            executor_class=executor_class,
            source_identity=source_identity,
            candidate_root=candidate_base,
            converter_source_commit=converter_source_commit,
        )
        report_cases.append({
            "hardware_id": hardware_id,
            "model_id": key[0],
            "case_id": key[1],
            "raw_runtime_path": str(result.raw_runtime_path),
            "raw_runtime_bytes": result.raw_runtime_path.stat().st_size,
            "raw_runtime_sha256": _sha256(result.raw_runtime_path),
            "candidate_dir": str(result.candidate_dir),
            "scheduler_pipeline_depth": result.scheduler_pipeline_depth,
            "canonical_file_sha256": dict(result.source_file_sha256),
            "candidate_file_sha256": dict(result.candidate_file_sha256),
        })
    report = {
        "schema_version": 1,
        "kind": "sharegpt_scheduler_depth_metadata_migration",
        "hardware_id": hardware_id,
        "case_count": len(report_cases),
        "executor_class": executor_class,
        "source_identity": source_identity,
        "converter_source_commit": converter_source_commit,
        "raw_mapping_path": raw_mapping,
        "raw_mapping_sha256": raw_mapping_sha256,
        "cases": report_cases,
    }
    report_path = candidate_base / "sharegpt_scheduler_depth_metadata" / (
        f"migration_report_{hardware_id}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite migration report: {report_path}")
    report_path.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    return {**report, "report_path": str(report_path)}


def _raw_depth_index(
    raw_root: Path, result_name: str, hardware_id: str
) -> dict[tuple[str, str], Path]:
    index = {}
    for raw_path in sorted(raw_root.rglob(result_name)):
        raw = _json_object(raw_path, "raw result")
        if "server" not in raw or "workload" not in raw:
            continue
        server = _mapping(raw["server"], "raw server")
        workload = _mapping(raw["workload"], "raw workload")
        if "identity" not in workload:
            continue
        identity = _mapping(workload["identity"], "raw workload identity")
        if identity["hardware_id"] != hardware_id:
            continue
        engine = _mapping(server["engine"], "raw server engine")
        if "max_concurrent_batches" not in engine:
            continue
        model_id = identity["model_id"]
        case_id = identity["case_id"]
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"raw model ID is invalid: {raw_path}")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"raw case ID is invalid: {raw_path}")
        key = (model_id, case_id)
        if key in index:
            raise ValueError(
                f"raw depth evidence is ambiguous for {hardware_id}/{model_id}/{case_id}"
            )
        index[key] = raw_path
    return index


def _raw_depth_mapping_index(
    mapping_path: Path,
    raw_root: Path,
    hardware_id: str,
) -> tuple[dict[tuple[str, str], Path], str]:
    mapping_path = mapping_path.resolve(strict=True)
    mapping = _json_object(mapping_path, "raw mapping")
    if set(mapping) != {"schema_version", "kind", "hardware_id", "cases"}:
        raise ValueError("raw mapping fields are invalid")
    if mapping["schema_version"] != 1:
        raise ValueError("raw mapping schema_version must be 1")
    if mapping["kind"] != "sharegpt_scheduler_depth_raw_mapping":
        raise ValueError("raw mapping kind is invalid")
    if mapping["hardware_id"] != hardware_id:
        raise ValueError("raw mapping hardware identity disagrees")
    cases = _sequence(mapping["cases"], "raw mapping cases")
    if not cases:
        raise ValueError("raw mapping cases must be non-empty")
    index = {}
    for case_index, value in enumerate(cases):
        case = _mapping(value, f"raw mapping case {case_index}")
        if set(case) != {
            "model_id", "case_id", "raw_runtime_path", "raw_runtime_sha256"
        }:
            raise ValueError(f"raw mapping case {case_index} fields are invalid")
        model_id = case["model_id"]
        case_id = case["case_id"]
        if (not isinstance(model_id, str) or not model_id
                or not isinstance(case_id, str) or not case_id):
            raise ValueError(f"raw mapping case {case_index} identity is invalid")
        raw_path = Path(case["raw_runtime_path"]).resolve(strict=True)
        try:
            raw_path.relative_to(raw_root)
        except ValueError as error:
            raise ValueError(
                f"raw mapping case {model_id}/{case_id} escapes raw root"
            ) from error
        expected_sha256 = case["raw_runtime_sha256"]
        if (not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in expected_sha256)):
            raise ValueError(
                f"raw mapping case {model_id}/{case_id} SHA256 is invalid"
            )
        if _sha256(raw_path) != expected_sha256:
            raise ValueError(
                f"raw mapping case {model_id}/{case_id} SHA256 changed"
            )
        raw = _json_object(raw_path, "mapped raw result")
        server = _mapping(raw["server"], "mapped raw server")
        workload = _mapping(raw["workload"], "mapped raw workload")
        identity = _mapping(workload["identity"], "mapped raw identity")
        if (identity["hardware_id"] != hardware_id
                or identity["model_id"] != model_id
                or identity["case_id"] != case_id):
            raise ValueError(
                f"mapped raw identity disagrees for {model_id}/{case_id}"
            )
        engine = _mapping(server["engine"], "mapped raw server engine")
        if "max_concurrent_batches" not in engine:
            raise ValueError(f"mapped raw has no Executor capacity: {raw_path}")
        key = (model_id, case_id)
        if key in index:
            raise ValueError(f"raw mapping repeats {hardware_id}/{model_id}/{case_id}")
        index[key] = raw_path
    return index, _sha256(mapping_path)


def audit_scheduler_depth_metadata_candidates(
    *,
    canonical_root: str | Path,
    candidate_root: str | Path,
    migration_reports: list[str | Path],
) -> dict[str, object]:
    """Strictly reload candidates and prove raw evidence and payload equality."""

    canonical_base = Path(canonical_root).resolve(strict=True)
    candidate_base = Path(candidate_root).resolve(strict=True)
    cases = []
    seen = set()
    for report_value in migration_reports:
        report_path = Path(report_value).resolve(strict=True)
        report = _json_object(report_path, "Scheduler depth migration report")
        report_cases = _sequence(report["cases"], "migration report cases")
        for value in report_cases:
            case = _mapping(value, "migration report case")
            identity = (
                case["hardware_id"], case["model_id"], case["case_id"])
            if identity in seen:
                raise ValueError(f"duplicate migration case {identity}")
            seen.add(identity)
            canonical_dir = canonical_base.joinpath(*identity).resolve(strict=True)
            candidate_dir = candidate_base.joinpath(*identity).resolve(strict=True)
            raw_path = Path(case["raw_runtime_path"]).resolve(strict=True)
            if raw_path.stat().st_size != case["raw_runtime_bytes"]:
                raise ValueError(f"raw runtime byte count changed for {identity}")
            if _sha256(raw_path) != case["raw_runtime_sha256"]:
                raise ValueError(f"raw runtime SHA256 changed for {identity}")
            raw = _json_object(raw_path, "raw Scheduler runtime record")
            server = _mapping(raw["server"], "raw server")
            engine = _mapping(server["engine"], "raw server engine")
            raw_fields = {
                "server.engine.max_concurrent_batches": engine[
                    "max_concurrent_batches"],
                "server.engine.pipeline_parallel_size": engine[
                    "pipeline_parallel_size"],
                "server.client_vllm_version": server["client_vllm_version"],
            }
            loaded = load_compact_case(candidate_dir)
            runtime = _mapping(loaded.manifest["runtime"], "candidate runtime")
            evidence = _mapping(
                runtime["scheduler_pipeline_depth_evidence"],
                "candidate Scheduler depth evidence",
            )
            if runtime["scheduler_pipeline_depth"] != raw_fields[
                    "server.engine.max_concurrent_batches"]:
                raise ValueError(f"candidate depth disagrees with raw for {identity}")
            if evidence["pipeline_parallel_size"] != raw_fields[
                    "server.engine.pipeline_parallel_size"]:
                raise ValueError(f"candidate PP disagrees with raw for {identity}")
            if evidence["runtime_version"] != raw_fields[
                    "server.client_vllm_version"]:
                raise ValueError(
                    f"candidate runtime version disagrees with raw for {identity}")
            canonical_payload = _nonmetadata_payload_hashes(canonical_dir)
            candidate_payload = _nonmetadata_payload_hashes(candidate_dir)
            if canonical_payload != candidate_payload:
                raise ValueError(
                    f"metadata migration changed non-metadata payload for {identity}")
            cases.append({
                "hardware_id": identity[0],
                "model_id": identity[1],
                "case_id": identity[2],
                "candidate_dir": str(candidate_dir),
                "reload_status": "passed",
                "raw_runtime_path": str(raw_path),
                "raw_runtime_bytes": raw_path.stat().st_size,
                "raw_runtime_sha256": _sha256(raw_path),
                "raw_fields_read": raw_fields,
                "executor_class": evidence["executor_class"],
                "source_identity": evidence["source_identity"],
                "nonmetadata_payload_sha256": canonical_payload,
                "payload_equivalence_status": "byte_or_canonical_json_equal",
            })
    if len(cases) != 45:
        raise ValueError(f"Scheduler depth migration audit requires 45 cases, got {len(cases)}")
    return {
        "schema_version": 1,
        "kind": "sharegpt_scheduler_depth_metadata_migration_audit",
        "case_count": len(cases),
        "reload_passed_count": len(cases),
        "raw_evidence_verified_count": len(cases),
        "nonmetadata_payload_equivalent_count": len(cases),
        "failure_count": 0,
        "cases": cases,
    }


def migrate_scheduler_depth_metadata_case(
    *,
    canonical_case_dir: str | Path,
    raw_runtime_path: str | Path,
    executor_class: str,
    source_identity: str,
    candidate_root: str | Path,
    converter_source_commit: str,
) -> SchedulerDepthMigrationResult:
    """Write one v5 candidate from a raw Executor capacity record.

    The raw record supplies only configured capacity and runtime identity.  No
    duration or per-shape timing observation is consumed by this migration.
    """

    validate_converter_source_commit(converter_source_commit)
    source_dir = Path(canonical_case_dir).resolve(strict=True)
    raw_path = Path(raw_runtime_path).resolve(strict=True)
    if not isinstance(executor_class, str) or not executor_class:
        raise ValueError("Executor class must be non-empty text")
    if not isinstance(source_identity, str) or not source_identity:
        raise ValueError("Scheduler source identity must be non-empty text")
    source = load_compact_case(source_dir)
    manifest = _json_object(source_dir / "manifest.json", "canonical manifest")
    if manifest["schema_version"] not in (
            SCHEDULER_CAPTURE_SCHEMA_VERSION,
            RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION):
        raise ValueError(
            "metadata migration requires a schema-v3 or schema-v4 canonical "
            "compact case")
    runtime = _mapping(manifest["runtime"], "canonical runtime")
    raw = _json_object(raw_path, "raw Scheduler runtime record")
    server = _mapping(raw["server"], "raw server")
    engine = _mapping(server["engine"], "raw server engine")
    depth = engine["max_concurrent_batches"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError(
            "raw server engine max_concurrent_batches must be a positive "
            "non-bool integer")
    runtime_version = server["client_vllm_version"]
    if not isinstance(runtime_version, str) or not runtime_version:
        raise ValueError("raw server client_vllm_version must be non-empty text")
    if runtime_version != runtime["runtime_version"]:
        raise ValueError("raw server runtime version disagrees with compact case")
    pipeline_parallel_size = engine["pipeline_parallel_size"]
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)
            or pipeline_parallel_size < 1):
        raise ValueError("raw Executor PP must be a positive non-bool integer")
    if pipeline_parallel_size != runtime["pipeline_parallel_size"]:
        raise ValueError("raw Executor PP disagrees with compact case")

    migrated_runtime = {
        **runtime,
        "source_commit": source_identity,
        "scheduler_pipeline_depth": depth,
        "scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_executor_property",
            "source_field": "max_concurrent_batches",
            "executor_class": executor_class,
            "pipeline_parallel_size": pipeline_parallel_size,
            "runtime_version": runtime_version,
            "source_identity": source_identity,
        },
    }
    raw_sources = [
        dict(_mapping(value, "canonical raw source"))
        for value in _sequence(manifest["raw_sources"], "canonical raw sources")
    ]
    raw_source_id = "scheduler-pipeline-depth-runtime-v1"
    if any(source_entry["archive_id"] == raw_source_id
           for source_entry in raw_sources):
        raise ValueError("canonical raw sources already contain depth migration")
    raw_sources.append({
        "archive_id": raw_source_id,
        "role": "scheduler-executor-capacity",
        "relative_path": (
            f"external_scheduler_depth/{hashlib.sha256(str(raw_path).encode()).hexdigest()}/"
            f"{raw_path.name}"),
        "bytes": raw_path.stat().st_size,
        "sha256": _sha256(raw_path),
        "tool": "vllm",
        "tool_version": runtime_version,
    })

    metrics_entry = _mapping(manifest["metrics"], "canonical metrics entry")
    if metrics_entry["status"] == "available":
        metrics = _json_object(
            source_dir / metrics_entry["path"], "canonical metrics")
        del metrics["schema_version"]
        del metrics["requests_sha256"]
        metrics_payload = AvailabilityPayload("available", None, metrics)
    elif metrics_entry["status"] == "unavailable":
        metrics_payload = AvailabilityPayload(
            "unavailable", metrics_entry["reason"], None)
    else:
        raise ValueError("canonical metrics status is invalid")
    alignment_entry = _mapping(
        manifest["alignment_steps"], "canonical alignment entry")
    alignment = _json_bytes_object(
        read_compressed_canonical_bytes(
            source_dir / alignment_entry["path"],
            compressed_sha256=alignment_entry["compressed_sha256"],
            content_sha256=alignment_entry["sha256"],
            label=ALIGNMENT_FILENAME,
        ),
        "canonical alignment",
    )
    del alignment["schema_version"]
    del alignment["kind"]
    scheduler_entry = _mapping(
        manifest["scheduler_steps"], "canonical scheduler steps entry")
    scheduler_rows = _jsonl_bytes_objects(
        read_compressed_canonical_bytes(
            source_dir / scheduler_entry["path"],
            compressed_sha256=scheduler_entry["compressed_sha256"],
            content_sha256=scheduler_entry["sha256"],
            label="canonical scheduler steps",
        ),
        "canonical scheduler steps",
    )
    scheduler_step_schema_version = scheduler_rows[0]["schema_version"]
    if scheduler_step_schema_version not in (
            SCHEDULER_CAPTURE_SCHEMA_VERSION,
            RUNTIME_GRAPH_SCHEDULER_SCHEMA_VERSION):
        raise ValueError("canonical Scheduler step schema is unsupported")
    for row in scheduler_rows:
        if row["schema_version"] != scheduler_step_schema_version:
            raise ValueError("canonical Scheduler step schemas disagree")
        del row["schema_version"]
    requests_entry = _mapping(manifest["requests"], "canonical requests entry")
    candidate_path = (
        Path(candidate_root).resolve()
        / "sharegpt_scheduler_depth_metadata"
        / manifest["identity"]["hardware_id"]
        / manifest["identity"]["model_id"]
        / manifest["identity"]["case_id"]
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    payload = CompactCasePayload(
        identity=_mapping(manifest["identity"], "canonical identity"),
        runtime=migrated_runtime,
        requests=tuple(source.requests),
        request_field_provenance=requests_entry["field_provenance"],
        cached_hit_value_evidence=requests_entry["cached_hit_value_evidence"],
        metrics=metrics_payload,
        alignment_steps=alignment,
        raw_sources=raw_sources,
        provenance_gaps=_mapping(manifest["provenance"], "canonical provenance")[
            "gaps"],
        converter={
            "implementation_id": "midlayer.scheduler_depth_metadata_migration.v1",
            "source_commit": converter_source_commit,
            "selection_rule": _mapping(
                manifest["converter"], "canonical converter")["selection_rule"],
        },
        schema_version=LATEST_SCHEMA_VERSION,
        scheduler_steps=scheduler_rows,
        scheduler_step_schema_version=scheduler_step_schema_version,
        scheduler_timing_boundary=scheduler_entry["timing_boundary"],
        scheduler_provenance=scheduler_entry["provenance"],
        capture=_mapping(manifest["capture"], "canonical capture"),
    )
    written = write_compact_case_candidate(
        candidate_path, payload, candidate_root=candidate_root)
    reloaded = load_compact_case(written)
    if reloaded.manifest["runtime"]["scheduler_pipeline_depth"] != depth:
        raise AssertionError("reloaded Scheduler depth disagrees with raw record")
    source_hashes = _case_file_hashes(source_dir)
    candidate_hashes = _case_file_hashes(written)
    return SchedulerDepthMigrationResult(
        candidate_dir=written,
        canonical_case_dir=source_dir,
        raw_runtime_path=raw_path,
        scheduler_pipeline_depth=depth,
        source_file_sha256=source_hashes,
        candidate_file_sha256=candidate_hashes,
    )


def _case_file_hashes(case_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(case_dir.iterdir())
        if path.is_file()
    }


def _nonmetadata_payload_hashes(case_dir: Path) -> dict[str, str]:
    manifest = _json_object(case_dir / "manifest.json", "manifest")
    scheduler_entry = _mapping(
        manifest["scheduler_steps"], "manifest scheduler steps")
    scheduler_rows = _jsonl_bytes_objects(
        read_compressed_canonical_bytes(
            case_dir / scheduler_entry["path"],
            compressed_sha256=scheduler_entry["compressed_sha256"],
            content_sha256=scheduler_entry["sha256"],
            label="Scheduler steps",
        ),
        "Scheduler steps",
    )
    for row in scheduler_rows:
        del row["schema_version"]
    metrics = _json_object(case_dir / "metrics.json", "metrics")
    del metrics["schema_version"]
    del metrics["requests_sha256"]
    alignment_entry = _mapping(
        manifest["alignment_steps"], "manifest alignment")
    alignment = _json_bytes_object(
        read_compressed_canonical_bytes(
            case_dir / alignment_entry["path"],
            compressed_sha256=alignment_entry["compressed_sha256"],
            content_sha256=alignment_entry["sha256"],
            label=ALIGNMENT_FILENAME,
        ),
        "alignment",
    )
    del alignment["schema_version"]
    del alignment["kind"]
    return {
        "requests.csv": _sha256(case_dir / "requests.csv"),
        "scheduler_steps_without_schema_version": _object_sha256(scheduler_rows),
        "metrics_without_schema_metadata": _object_sha256(metrics),
        "alignment_without_schema_metadata": _object_sha256(alignment),
    }


def _json_object(path: Path, description: str) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be an object")
    return value


def _json_bytes_object(content: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be an object")
    return value


def _jsonl_objects(path: Path, description: str) -> list[dict[str, object]]:
    return _jsonl_bytes_objects(path.read_bytes(), description)


def _jsonl_bytes_objects(
    content: bytes, description: str,
) -> list[dict[str, object]]:
    rows = []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} must be UTF-8") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{description} row {line_number} must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{description} cannot be empty")
    return rows




def _sequence(value: object, description: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{description} must be a list")
    return value


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _object_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
