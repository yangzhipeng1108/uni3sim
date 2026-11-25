"""Batch construction of frozen ShareGPT compact candidates."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Mapping, Sequence

from .compact_candidate_writer import AvailabilityPayload
from .compact_case import (
    SCHEMA_VERSION,
    _validate_metrics,
    canonical_requests_csv_bytes,
    raw_source_set_sha256,
    read_canonical_request_rows,
    validate_converter_source_commit,
)

from .compact_candidate_writer import write_compact_case_candidate
from .sharegpt_raw_adapter import build_sharegpt_case_compact_payload
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from dataset_pipeline.serving_dataset.json_shape import (
    compact_mapping as _mapping,
    compact_sequence as _sequence,
    require_exact_keys,
    reject_duplicate_json_keys,
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW_ROOT = _REPOSITORY_ROOT / "profiler_output" / "serving"
_CANDIDATE_ROOT = (
    _REPOSITORY_ROOT / "profiler_output" / "serving_dataset_compaction"
)
_CANONICAL_DATASET_ROOT = _REPOSITORY_ROOT / "serving_dataset" / "sharegpt"
_OFFICIAL_TRUTH_SNAPSHOT_FILENAME = "official_truth_snapshot.json"
_OFFICIAL_TRUTH_SNAPSHOT_KIND = "sharegpt_official_truth_snapshot"
_OFFICIAL_TRUTH_SOURCE_ID = "production-ground-truth"


@dataclass(frozen=True)
class ShareGPTCaseSpec:
    hardware_id: str
    model_id: str
    raw_model_id: str
    target_qps: int
    tensor_parallel_size: int
    expert_parallel_size: int
    raw_directory_name: str
    case_id: str
    publish_metrics: bool


def _case_specs() -> tuple[ShareGPTCaseSpec, ...]:
    specs = []
    hardware_ids = EVALUATED_HARDWARE_IDS
    for hardware_id in hardware_ids:
        for model_id, raw_model_id in (
            ("qwen3-4b", "Qwen3-4B"),
            ("qwen3-8b", "Qwen3-8B"),
            ("llama-3.1-8b", "Llama-3.1-8B"),
        ):
            for qps in (4, 16, 64):
                specs.append(ShareGPTCaseSpec(hardware_id, model_id, raw_model_id, qps, 1, 1, f"ShareGPT-{raw_model_id}_TP1_n128_qps{qps}", f"tp1_qps{qps}", True))
        for model_id, raw_model_id, ep in (
            ("qwen3-30b-a3b", "Qwen3-30B-A3B", 2),
            ("qwen3-32b", "Qwen3-32B", 1),
        ):
            for qps in (1, 4, 16):
                parallel_tag = "EP2" if ep == 2 else "TP2"
                specs.append(ShareGPTCaseSpec(hardware_id, model_id, raw_model_id, qps, 2, ep, f"ShareGPT-{raw_model_id}_{parallel_tag}_n128_qps{qps}", f"tp2_ep{ep}_qps{qps}", True))
    for qps in (1, 4, 16):
        specs.append(ShareGPTCaseSpec("nvidia_a800", "llama-3-70b", "Llama-3-70B", qps, 4, 1, f"ShareGPT-Llama-3-70B_TP4_n128_qps{qps}", f"tp4_qps{qps}", False))
        specs.append(ShareGPTCaseSpec("metax_c500", "llama-3.1-70b", "Llama-3.1-70B", qps, 4, 1, f"ShareGPT-Llama-3.1-70B_TP4_n128_qps{qps}", f"tp4_qps{qps}", False))
    result = tuple(specs)
    if len(result) != 51:
        raise AssertionError("frozen ShareGPT matrix must contain exactly 51 cases")
    return result


SHAREGPT_CASE_SPECS = _case_specs()


def build_official_truth_snapshot(
    *, source_dataset_root: str | Path, snapshot_root: str | Path
) -> Path:
    """Build the exact 45-case, small official-truth transfer snapshot."""

    source_root = Path(source_dataset_root).resolve(strict=True)
    destination = Path(snapshot_root).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"official-truth snapshot already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.snapshot-", dir=destination.parent)
    )
    try:
        entries = []
        for spec in (value for value in SHAREGPT_CASE_SPECS if value.publish_metrics):
            source_case = (
                source_root / spec.hardware_id / spec.raw_model_id / spec.case_id
            )
            entry = _copy_official_truth_case(source_case, temporary, spec)
            entries.append(entry)
        if len(entries) != 45:
            raise AssertionError("official-truth snapshot must contain exactly 45 cases")
        inventory = {
            "schema_version": 1,
            "kind": _OFFICIAL_TRUTH_SNAPSHOT_KIND,
            "case_count": len(entries),
            "cases": entries,
        }
        inventory["inventory_sha256"] = _canonical_json_sha256(entries)
        (temporary / _OFFICIAL_TRUTH_SNAPSHOT_FILENAME).write_bytes(
            _json_bytes(inventory)
        )
        validate_official_truth_snapshot(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def validate_official_truth_snapshot(
    snapshot_root: str | Path,
) -> Mapping[str, Mapping[str, object]]:
    """Validate the complete 45-case snapshot and return its identity index."""

    root = Path(snapshot_root)
    if not root.is_dir():
        raise ValueError(f"official-truth snapshot is not a directory: {root}")
    manifest_path = root / _OFFICIAL_TRUTH_SNAPSHOT_FILENAME
    manifest = _json_object(manifest_path, "official-truth snapshot manifest")
    _exact_keys(
        manifest,
        ("schema_version", "kind", "case_count", "cases", "inventory_sha256"),
        "official-truth snapshot manifest",
    )
    if manifest["schema_version"] != 1:
        raise ValueError("official-truth snapshot schema_version must equal 1")
    if manifest["kind"] != _OFFICIAL_TRUTH_SNAPSHOT_KIND:
        raise ValueError("official-truth snapshot kind is invalid")
    cases = _sequence(manifest["cases"], "official-truth snapshot cases")
    if manifest["case_count"] != 45 or len(cases) != 45:
        raise ValueError("official-truth snapshot must contain exactly 45 cases")
    if manifest["inventory_sha256"] != _canonical_json_sha256(cases):
        raise ValueError("official-truth snapshot inventory SHA256 is inconsistent")
    expected = {
        (spec.hardware_id, spec.model_id, spec.case_id)
        for spec in SHAREGPT_CASE_SPECS if spec.publish_metrics
    }
    index = {}
    actual = set()
    expected_files = {_OFFICIAL_TRUTH_SNAPSHOT_FILENAME}
    for position, value in enumerate(cases):
        entry = _mapping(value, f"official-truth snapshot case {position}")
        _exact_keys(
            entry,
            (
                "hardware_id", "model_id", "case_id", "raw_model_id",
                "requests", "metrics", "official_source",
            ),
            f"official-truth snapshot case {position}",
        )
        identity = (entry["hardware_id"], entry["model_id"], entry["case_id"])
        if identity in actual:
            raise ValueError(f"duplicate official-truth snapshot case: {identity!r}")
        actual.add(identity)
        for name in ("requests", "metrics"):
            file_entry = _mapping(entry[name], f"snapshot {name} entry")
            _exact_keys(
                file_entry,
                ("path", "bytes", "sha256"),
                f"snapshot {name} entry",
            )
            relative = _safe_relative_path(file_entry["path"], f"snapshot {name} path")
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"snapshot {name} file is missing or a symlink: {relative}")
            content = path.read_bytes()
            if file_entry["bytes"] != len(content):
                raise ValueError(f"snapshot {name} byte count is inconsistent")
            if file_entry["sha256"] != hashlib.sha256(content).hexdigest():
                raise ValueError(f"snapshot {name} SHA256 is inconsistent")
            expected_files.add(relative.as_posix())
        source = _mapping(entry["official_source"], "snapshot official source")
        _validate_official_source(source)
        requests = read_canonical_request_rows(root / entry["requests"]["path"])
        request_bytes = canonical_requests_csv_bytes(requests)
        if request_bytes != (root / entry["requests"]["path"]).read_bytes():
            raise ValueError("snapshot requests.csv is not canonical")
        metrics = _json_object(root / entry["metrics"]["path"], "snapshot metrics")
        _validate_snapshot_metrics(metrics, entry["requests"]["sha256"], len(requests), source)
        index["/".join(identity)] = entry
    if actual != expected:
        raise ValueError(
            "official-truth snapshot case matrix mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "official-truth snapshot file set mismatch; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    return index


def build_all_sharegpt_candidates(
    *, raw_root: str | Path, converter_source_commit: str,
    candidate_root: str | Path = _CANDIDATE_ROOT,
    canonical_dataset_root: str | Path = _CANONICAL_DATASET_ROOT,
    hardware_ids: Sequence[str] | None = None,
    model_ids: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
    jobs: int = 1,
) -> tuple[Path, ...]:
    """Build a selected subset of the frozen 51-case matrix."""

    validate_converter_source_commit(converter_source_commit)
    destination_root = Path(candidate_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    if not isinstance(jobs, int) or isinstance(jobs, bool) or not 1 <= jobs <= 32:
        raise ValueError("candidate build jobs must be an integer in [1, 32]")
    selected = _select_case_specs(hardware_ids, model_ids, case_ids)
    snapshot_root = Path(canonical_dataset_root)
    snapshot_index = validate_official_truth_snapshot(snapshot_root)
    outputs: list[Path | None] = [None] * len(selected)
    failures = []
    arguments = tuple(
        (
            Path(raw_root), converter_source_commit, destination_root,
            snapshot_root,
            None if not spec.publish_metrics else snapshot_index[
                f"{spec.hardware_id}/{spec.model_id}/{spec.case_id}"
            ],
            spec,
        )
        for spec in selected
    )
    if jobs == 1:
        for index, argument in enumerate(arguments):
            try:
                outputs[index] = _build_one_sharegpt_candidate(argument)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                failures.append(_candidate_failure(selected[index], error))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_index = {
                executor.submit(_build_one_sharegpt_candidate, argument): index
                for index, argument in enumerate(arguments)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    outputs[index] = future.result()
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    failures.append(_candidate_failure(selected[index], error))
    if failures:
        summary = {
            "successful_count": sum(output is not None for output in outputs),
            "failure_count": len(failures),
            "failures": failures,
        }
        raise RuntimeError(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return tuple(output for output in outputs if output is not None)


def _build_one_sharegpt_candidate(arguments: tuple[object, ...]) -> Path:
    (
        raw_root, converter_source_commit, destination_root,
        official_truth_snapshot_root, official_truth_entry, spec,
    ) = arguments
    if not isinstance(spec, ShareGPTCaseSpec):
        raise TypeError("candidate worker requires ShareGPTCaseSpec")
    raw_case = _resolve_raw_case(Path(raw_root), spec)
    payload = build_sharegpt_case_compact_payload(
        raw_case,
        hardware_id=spec.hardware_id,
        raw_model_id=spec.raw_model_id,
        target_qps=spec.target_qps,
        tensor_parallel_size=spec.tensor_parallel_size,
        expert_parallel_size=spec.expert_parallel_size,
        case_id=spec.case_id,
        publish_metrics=spec.publish_metrics,
        converter_source_commit=str(converter_source_commit),
    )
    if spec.publish_metrics:
        payload = _bind_existing_official_truth(
            payload,
            spec,
            snapshot_root=Path(official_truth_snapshot_root),
            snapshot_entry=_mapping(
                official_truth_entry, "official-truth snapshot case"
            ),
        )
    destination = Path(destination_root) / spec.hardware_id / spec.model_id / spec.case_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    return write_compact_case_candidate(
        destination,
        payload,
        candidate_root=Path(destination_root),
    )


def _candidate_failure(
    spec: ShareGPTCaseSpec, error: Exception
) -> Mapping[str, object]:
    return {
        "cell_id": f"{spec.hardware_id}/{spec.model_id}/{spec.case_id}",
        "error_type": type(error).__name__,
        "message": str(error),
    }


def _select_case_specs(
    hardware_ids: Sequence[str] | None,
    model_ids: Sequence[str] | None,
    case_ids: Sequence[str] | None,
) -> tuple[ShareGPTCaseSpec, ...]:
    """Select a non-empty, duplicate-free subset of the frozen case matrix."""

    selections = (
        (hardware_ids, {spec.hardware_id for spec in SHAREGPT_CASE_SPECS}, "hardware"),
        (model_ids, {spec.model_id for spec in SHAREGPT_CASE_SPECS}, "model"),
        (case_ids, {spec.case_id for spec in SHAREGPT_CASE_SPECS}, "case"),
    )
    for values, allowed, description in selections:
        if values is None:
            continue
        if not values:
            raise ValueError(f"{description} selection cannot be empty")
        if len(set(values)) != len(values):
            raise ValueError(f"{description} selection contains duplicates")
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"{description} selection contains unsupported values: {unknown}")
    selected = tuple(
        spec for spec in SHAREGPT_CASE_SPECS
        if (hardware_ids is None or spec.hardware_id in hardware_ids)
        and (model_ids is None or spec.model_id in model_ids)
        and (case_ids is None or spec.case_id in case_ids)
    )
    if not selected:
        raise ValueError("ShareGPT selection matches no frozen case")
    return selected


def _bind_existing_official_truth(
    payload,
    spec: ShareGPTCaseSpec,
    *,
    snapshot_root: Path,
    snapshot_entry: Mapping[str, object],
):
    """Preserve the existing 45-case requests and no-profiler metric truth."""

    cell_id = f"{spec.hardware_id}/{spec.model_id}/{spec.case_id}"
    snapshot_identity = "/".join(
        str(snapshot_entry[key])
        for key in ("hardware_id", "model_id", "case_id")
    )
    if snapshot_identity != cell_id:
        raise ValueError("official-truth snapshot entry identity is inconsistent")
    requests_path = snapshot_root / snapshot_entry["requests"]["path"]
    metrics_path = snapshot_root / snapshot_entry["metrics"]["path"]
    official_source = dict(
        _mapping(snapshot_entry["official_source"], "snapshot official source")
    )
    if any(
        source["archive_id"] == _OFFICIAL_TRUTH_SOURCE_ID
        for source in payload.raw_sources
    ):
        raise ValueError("raw payload already contains production-ground-truth")
    raw_sources = tuple(payload.raw_sources) + (official_source,)
    source_set_sha = raw_source_set_sha256(raw_sources)
    evidence = dict(payload.cached_hit_value_evidence)
    evidence["raw_source_set_sha256"] = source_set_sha
    alignment = dict(payload.alignment_steps)
    phases = dict(alignment["phases"])
    for phase, value in phases.items():
        if value["status"] != "available":
            continue
        sample = dict(value["sample"])
        provenance = dict(sample["provenance"])
        provenance["raw_source_set_sha256"] = source_set_sha
        sample["provenance"] = provenance
        phases[phase] = {"status": "available", "sample": sample}
    alignment["phases"] = phases
    snapshot_metrics = _json_object(metrics_path, "snapshot metrics")
    metrics = {
        key: snapshot_metrics[key]
        for key in ("metric_provenance", "per_request", "summary")
    }
    return replace(
        payload,
        requests=read_canonical_request_rows(requests_path),
        metrics=AvailabilityPayload("available", None, metrics),
        cached_hit_value_evidence=evidence,
        alignment_steps=alignment,
        raw_sources=raw_sources,
    )


def _copy_official_truth_case(
    source_case: Path,
    snapshot_root: Path,
    spec: ShareGPTCaseSpec,
) -> Mapping[str, object]:
    manifest_path = source_case / "manifest.json"
    requests_path = source_case / "requests.csv"
    if not manifest_path.is_file() or not requests_path.is_file():
        raise FileNotFoundError(
            "missing canonical official truth for "
            f"{spec.hardware_id}/{spec.raw_model_id}/{spec.case_id}"
        )
    manifest = _json_object(manifest_path, "canonical case manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("official-truth snapshot source must use legacy schema version 1")
    identity = _mapping(manifest["identity"], "canonical case identity")
    for key, expected in (
        ("source_kind", "sharegpt"),
        ("hardware_id", spec.hardware_id),
        ("model_id", spec.raw_model_id),
        ("case_id", spec.case_id),
    ):
        if identity[key] != expected:
            raise ValueError(f"canonical case identity {key} mismatch")
    request_entry = _mapping(manifest["requests"], "canonical requests entry")
    if request_entry["path"] != "requests.csv":
        raise ValueError("canonical requests path must equal requests.csv")
    request_bytes = requests_path.read_bytes()
    request_sha = hashlib.sha256(request_bytes).hexdigest()
    if request_entry["requests_sha256"] != request_sha:
        raise ValueError("canonical requests SHA256 is inconsistent")
    requests = read_canonical_request_rows(requests_path)
    if request_bytes != canonical_requests_csv_bytes(requests):
        raise ValueError("canonical requests.csv serialization is not stable")
    if request_entry["row_count"] != len(requests):
        raise ValueError("canonical requests row_count is inconsistent")
    alignment = _mapping(manifest["alignment"], "canonical alignment")
    metric_entry = _mapping(alignment["metrics"], "canonical metrics entry")
    if metric_entry["status"] != "available":
        raise ValueError("official metrics must be available")
    source_metrics_path = source_case / _safe_relative_path(
        metric_entry["path"], "canonical metrics path"
    )
    if not source_metrics_path.is_file() or source_metrics_path.is_symlink():
        raise ValueError("canonical metrics path is missing or a symlink")
    source_metrics_bytes = source_metrics_path.read_bytes()
    if metric_entry["sha256"] != hashlib.sha256(source_metrics_bytes).hexdigest():
        raise ValueError("canonical metrics SHA256 is inconsistent")
    source_metrics = _json_object(source_metrics_path, "canonical metrics")
    raw_sources = _sequence(manifest["raw_sources"], "canonical raw_sources")
    official_sources = tuple(
        _mapping(value, "canonical raw source") for value in raw_sources
        if _mapping(value, "canonical raw source")["archive_id"]
        == _OFFICIAL_TRUTH_SOURCE_ID
    )
    if len(official_sources) != 1:
        raise ValueError(
            "official metrics require exactly one production-ground-truth source"
        )
    official_source = dict(official_sources[0])
    _validate_official_source(official_source)
    metrics = _normalize_snapshot_metrics(
        source_metrics, request_sha, len(requests), official_source
    )
    relative_case = Path(spec.hardware_id, spec.model_id, spec.case_id)
    destination = snapshot_root / relative_case
    destination.mkdir(parents=True, exist_ok=False)
    destination_requests = destination / "requests.csv"
    destination_metrics = destination / "metrics.json"
    destination_requests.write_bytes(request_bytes)
    destination_metrics.write_bytes(_json_bytes(metrics))
    return {
        "hardware_id": spec.hardware_id,
        "model_id": spec.model_id,
        "case_id": spec.case_id,
        "raw_model_id": spec.raw_model_id,
        "requests": _snapshot_file_entry(
            destination_requests, snapshot_root
        ),
        "metrics": _snapshot_file_entry(destination_metrics, snapshot_root),
        "official_source": official_source,
    }


def _normalize_snapshot_metrics(
    metrics: Mapping[str, object],
    request_sha: str,
    request_count: int,
    official_source: Mapping[str, object],
) -> Mapping[str, object]:
    if metrics["schema_version"] != 1:
        raise ValueError("canonical official metrics must use schema version 1")
    if metrics["requests_sha256"] != request_sha:
        raise ValueError("canonical metrics requests SHA256 is inconsistent")
    legacy_provenance = _mapping(
        metrics["metric_provenance"], "canonical metric provenance"
    )
    if (
        legacy_provenance["source_kind"] != "serving_runtime"
        or legacy_provenance["source_id"] != _OFFICIAL_TRUTH_SOURCE_ID
        or legacy_provenance["source_sha256"] != official_source["sha256"]
    ):
        raise ValueError("canonical metrics provenance is inconsistent")
    records = _sequence(metrics["per_request"], "canonical metrics per_request")
    per_request = []
    for position, value in enumerate(records):
        row = _mapping(value, f"canonical metric row {position}")
        per_request.append({
            "request_idx": row["request_idx"],
            "actual_output_tokens": row["actual_output_tokens"],
            "ttft_ms": row["ttft_ms"],
            "tpot_ms": row["tpot_ms"],
            "e2e_latency_ms": row["e2e_latency_ms"],
        })
    result = {
        "schema_version": SCHEMA_VERSION,
        "requests_sha256": request_sha,
        "metric_provenance": {
            "source_kind": "serving_runtime",
            "source_archive_id": _OFFICIAL_TRUTH_SOURCE_ID,
            "source_sha256": official_source["sha256"],
            "profiler_attached": False,
        },
        "per_request": per_request,
        "summary": metrics["summary"],
    }
    _validate_snapshot_metrics(result, request_sha, request_count, official_source)
    return result


def _validate_snapshot_metrics(
    metrics: Mapping[str, object],
    request_sha: object,
    request_count: int,
    official_source: Mapping[str, object],
) -> None:
    _exact_keys(
        metrics,
        (
            "schema_version", "requests_sha256", "metric_provenance",
            "per_request", "summary",
        ),
        "snapshot metrics",
    )
    if metrics["schema_version"] != SCHEMA_VERSION:
        raise ValueError("snapshot metrics schema_version is inconsistent")
    if metrics["requests_sha256"] != request_sha:
        raise ValueError("snapshot metrics requests SHA256 is inconsistent")
    provenance = _mapping(metrics["metric_provenance"], "snapshot metric provenance")
    _exact_keys(
        provenance,
        ("source_kind", "source_archive_id", "source_sha256", "profiler_attached"),
        "snapshot metric provenance",
    )
    if provenance != {
        "source_kind": "serving_runtime",
        "source_archive_id": _OFFICIAL_TRUTH_SOURCE_ID,
        "source_sha256": official_source["sha256"],
        "profiler_attached": False,
    }:
        raise ValueError("snapshot metrics provenance is inconsistent")
    per_request = _sequence(metrics["per_request"], "snapshot metrics per_request")
    if len(per_request) != request_count:
        raise ValueError("snapshot metrics request count is inconsistent")
    for position, value in enumerate(per_request):
        row = _mapping(value, f"snapshot metric row {position}")
        _exact_keys(
            row,
            (
                "request_idx", "actual_output_tokens", "ttft_ms", "tpot_ms",
                "e2e_latency_ms",
            ),
            f"snapshot metric row {position}",
        )
        if row["request_idx"] != position:
            raise ValueError("snapshot metrics must preserve physical request order")
    summary = _mapping(metrics["summary"], "snapshot metrics summary")
    if summary["request_count"] != request_count:
        raise ValueError("snapshot metrics summary request count is inconsistent")
    _validate_metrics(metrics, str(request_sha), request_count, "sharegpt")


def _validate_official_source(source: Mapping[str, object]) -> None:
    raw_source_set_sha256((source,))
    if source["archive_id"] != _OFFICIAL_TRUTH_SOURCE_ID:
        raise ValueError("official source archive_id is inconsistent")
    if not isinstance(source["bytes"], int) or isinstance(source["bytes"], bool):
        raise TypeError("official source bytes must be an integer")
    if source["bytes"] < 0:
        raise ValueError("official source bytes must be non-negative")
    if not _is_sha256(source["sha256"]):
        raise ValueError("official source SHA256 is invalid")


def _snapshot_file_entry(path: Path, root: Path) -> Mapping[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _safe_relative_path(value: object, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) in (".", ""):
        raise ValueError(f"{description} must be a safe relative path")
    return path


def _json_object(path: Path, description: str) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{description} contains invalid constant {constant}")
            ),
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{description} must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {description}: {error}") from error
    return _mapping(value, description)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    return reject_duplicate_json_keys(pairs)






def _exact_keys(
    value: Mapping[str, object], expected: Sequence[str], description: str
) -> None:
    require_exact_keys(value, expected, description)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _resolve_raw_case(raw_root: Path, spec: ShareGPTCaseSpec) -> Path:
    # The captures live under profiler_output/serving/<hardware_id>/
    # sharegpt_raw/<case>; the other shapes keep older roots loadable.
    candidates = (
        raw_root / spec.hardware_id / "sharegpt_raw" / spec.raw_directory_name,
        raw_root / spec.hardware_id / "sharegpt" / spec.raw_directory_name,
        raw_root / spec.raw_directory_name,
        raw_root / spec.hardware_id / spec.raw_directory_name,
    )
    existing = tuple(path for path in candidates if path.is_dir())
    if len(existing) != 1:
        raise ValueError(
            f"expected exactly one raw directory for {spec.hardware_id}/{spec.model_id}/{spec.case_id}; found {existing}"
        )
    return existing[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or inspect frozen ShareGPT compact Serving candidates"
    )
    parser.add_argument("--raw-root", type=Path, default=_DEFAULT_RAW_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=_CANDIDATE_ROOT)
    parser.add_argument(
        "--canonical-dataset-root",
        type=Path,
        default=None,
        help=(
            "explicit official-truth snapshot root for all-case candidate builds; "
            "legacy canonical ShareGPT root only when building a snapshot"
        ),
    )
    parser.add_argument(
        "--build-official-truth-snapshot",
        action="store_true",
        help="build and validate the exact 45-case lightweight truth snapshot",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        help="new output directory for --build-official-truth-snapshot",
    )
    parser.add_argument(
        "--hardware-id",
        action="append",
        choices=EVALUATED_HARDWARE_IDS,
        dest="hardware_ids",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
    )
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--converter-source-commit")
    parser.add_argument("--jobs", type=int, choices=range(1, 33), default=1)
    args = parser.parse_args(argv)
    if args.build_official_truth_snapshot:
        if args.snapshot_root is None:
            parser.error("--build-official-truth-snapshot requires --snapshot-root")
        if args.canonical_dataset_root is None:
            parser.error("--build-official-truth-snapshot requires --canonical-dataset-root")
        started = time.perf_counter()
        try:
            output = build_official_truth_snapshot(
                source_dataset_root=args.canonical_dataset_root,
                snapshot_root=args.snapshot_root,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            print(json.dumps({
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            }, sort_keys=True, separators=(",", ":")))
            return 1
        files = tuple(sorted(path for path in output.rglob("*") if path.is_file()))
        print(json.dumps({
            "status": "ok",
            "case_count": 45,
            "elapsed_seconds": time.perf_counter() - started,
            "total_bytes": sum(path.stat().st_size for path in files),
            "files": {
                path.relative_to(output).as_posix(): path.stat().st_size
                for path in files
            },
        }, sort_keys=True, separators=(",", ":")))
        return 0
    if args.snapshot_root is not None:
        parser.error("--snapshot-root requires --build-official-truth-snapshot")
    if args.converter_source_commit is None:
        parser.error("candidate construction requires --converter-source-commit")
    if args.canonical_dataset_root is None:
        parser.error("candidate construction requires --canonical-dataset-root snapshot")
    started = time.perf_counter()
    try:
        outputs = build_all_sharegpt_candidates(
            raw_root=args.raw_root,
            converter_source_commit=args.converter_source_commit,
            candidate_root=args.candidate_root,
            canonical_dataset_root=args.canonical_dataset_root,
            hardware_ids=args.hardware_ids,
            model_ids=args.model_ids,
            case_ids=args.case_ids,
            jobs=args.jobs,
        )
    except RuntimeError as error:
        payload = json.loads(str(error))
        payload["status"] = "failed"
        payload["elapsed_seconds"] = time.perf_counter() - started
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 1
    except (OSError, TypeError, ValueError) as error:
        print(json.dumps({
            "status": "failed",
            "successful_count": 0,
            "failure_count": 1,
            "failures": [{
                "cell_id": None,
                "error_type": type(error).__name__,
                "message": str(error),
            }],
            "elapsed_seconds": time.perf_counter() - started,
        }, sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps({
        "status": "ok",
        "successful_count": len(outputs),
        "failure_count": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "cases": [_candidate_file_record(path) for path in outputs],
    }, sort_keys=True, separators=(",", ":")))
    return 0


def _candidate_file_record(path: Path) -> Mapping[str, object]:
    files = tuple(sorted(candidate for candidate in path.iterdir() if candidate.is_file()))
    return {
        "path": str(path),
        "total_bytes": sum(candidate.stat().st_size for candidate in files),
        "files": {
            candidate.name: candidate.stat().st_size for candidate in files
        },
    }
