"""Strict conversion of production ground truth into metrics candidates.

The converter reads immutable ``ground_truth.json`` results and canonical
ShareGPT case inputs, validates their binding, and writes candidate metrics.
It never updates the canonical Serving dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Mapping, Sequence

from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS

from .compact_case import (
    ALIGNMENT_FILENAME,
    SCHEMA_VERSION,
    CanonicalRequestRow,
    load_compact_case,
    raw_source_set_sha256,
    read_canonical_request_rows,
    validate_compact_case,
)
from .compressed_artifact import (
    COMPRESSION_ALGORITHM,
    read_compressed_canonical_bytes,
    write_compressed_canonical_bytes,
)


_GROUND_TRUTH_IDENTITY_KEYS = (
    "source_kind",
    "hardware_id",
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "case_id",
)
_COMPACT_IDENTITY_KEYS = (
    "source_kind",
    "case_id",
    "hardware_id",
    "model_id",
    "raw_model_id",
    "model_revision",
    "tokenizer_revision",
)
_PARALLEL_FIELDS = (
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
)
_GROUND_TRUTH_FILENAME = "ground_truth.json"
_METRICS_RELATIVE_PATH = Path("metrics.json")
_ALIGNMENT_STEPS_RELATIVE_PATH = Path(ALIGNMENT_FILENAME)
_REQUEST_ID_PREFIX = "request-"
_GROUND_TRUTH_ARCHIVE_ID = "production-ground-truth"
_EXPECTED_SCHEDULER_POLICY = "fcfs"
_GROUND_TRUTH_CONVERTER_ID = "midlayer.ground_truth_metrics_converter.v2"
_CANDIDATE_DATASET_PATH = Path("serving_dataset/sharegpt")
_CANDIDATE_ATTESTATION_PATH = Path(
    "provenance/scheduler_policy_attestation.json"
)
_CANDIDATE_SUITE_DIRECTORY = Path("configs/evaluation/experiments")
_CANDIDATE_SOURCE_SUITE_DIRECTORY = Path("provenance/source_suite")
_CANDIDATE_CAPTURE_SUITES_PATH = Path("provenance/capture_suites")
_CANDIDATE_SOURCE_SNAPSHOTS_PATH = Path("provenance/capture_source_snapshots")


class GroundTruthMetricsConversionError(ValueError):
    """Raised when a measured result is not bound to its canonical case."""


def build_ground_truth_metrics_candidates(
    ground_truth_inputs: Sequence[str | Path],
    *,
    dataset_root: str | Path,
    suite_manifest_path: str | Path,
    capture_suite_manifest_paths: Sequence[str | Path],
    scheduler_policy_attestation_path: str | Path | None,
    capture_source_snapshot_manifest_paths: Sequence[str | Path],
    candidate_root: str | Path,
    capture_dataset_root: str | Path | None = None,
    template_hardware_id: str | None = None,
    converter_source_commit: str | None = None,
) -> tuple[Path, ...]:
    """Validate and convert ground truth files below a new candidate root.

    ``candidate_root`` is mandatory, must not overlap the canonical dataset,
    and must not already exist.  Every input is validated before any output is
    published.  ``template_hardware_id`` explicitly selects create mode for
    absent targets; create mode rejects an already-existing canonical target.
    """

    source_paths = _discover_ground_truth_paths(ground_truth_inputs)
    if template_hardware_id is not None:
        _nonempty_string(template_hardware_id, "template hardware ID")
        if template_hardware_id not in CANONICAL_HARDWARE_IDS:
            raise GroundTruthMetricsConversionError(
                f"template hardware ID is not canonical: {template_hardware_id!r}"
            )
        if converter_source_commit is None:
            raise GroundTruthMetricsConversionError(
                "create mode requires converter_source_commit"
            )
        converter_source_commit = _git_commit(
            converter_source_commit, "converter source commit"
        )
    elif converter_source_commit is not None:
        raise GroundTruthMetricsConversionError(
            "converter_source_commit is only valid in create mode"
        )
    canonical_root = Path(dataset_root).resolve(strict=True)
    if not canonical_root.is_dir():
        raise GroundTruthMetricsConversionError(
            f"Serving dataset root is not a directory: {canonical_root}"
        )
    suite_path = Path(suite_manifest_path).resolve(strict=True)
    suite_bytes = suite_path.read_bytes()
    suite = _json_object(suite_bytes, f"suite manifest {suite_path}")
    capture_root = (
        canonical_root
        if capture_dataset_root is None
        else Path(capture_dataset_root).resolve(strict=True)
    )
    if not capture_root.is_dir():
        raise GroundTruthMetricsConversionError(
            f"capture dataset root is not a directory: {capture_root}"
        )
    _validate_suite(suite, dataset_root=capture_root)
    suite_sha256 = _sha256_bytes(suite_bytes)
    capture_suites = _load_capture_suites(
        capture_suite_manifest_paths,
        dataset_root=capture_root,
    )
    scheduler_attestation = _load_scheduler_policy_attestation(
        scheduler_policy_attestation_path
    )
    capture_source_snapshots = _load_capture_source_snapshots(
        capture_source_snapshot_manifest_paths
    )
    destination = Path(candidate_root).resolve()
    _validate_candidate_destination(destination, canonical_root)

    converted = []
    seen_identities = set()
    for source_path in source_paths:
        result = _convert_one(
            source_path,
            dataset_root=canonical_root,
            capture_dataset_root=capture_root,
            suite=suite,
            current_suite_sha256=suite_sha256,
            capture_suites=capture_suites,
            scheduler_attestation=scheduler_attestation,
            capture_source_snapshots=capture_source_snapshots,
            template_hardware_id=template_hardware_id,
            converter_source_commit=converter_source_commit,
        )
        identity_tuple = tuple(result["identity"][key] for key in (
            "hardware_id", "model_id", "case_id"
        ))
        if identity_tuple in seen_identities:
            raise GroundTruthMetricsConversionError(
                f"duplicate ground-truth identity: {identity_tuple!r}"
            )
        seen_identities.add(identity_tuple)
        converted.append(result)

    _validate_attestation_coverage(converted, scheduler_attestation)
    _validate_capture_source_snapshot_coverage(
        converted, capture_source_snapshots
    )

    suite_identities = {
        tuple(cell[key] for key in ("hardware_id", "model_id", "case_id"))
        for cell in (
            _mapping(value, "suite cell")
            for value in _sequence(suite["cells"], "suite cells")
        )
    }
    if seen_identities != suite_identities:
        missing = sorted(suite_identities - seen_identities)
        unexpected = sorted(seen_identities - suite_identities)
        raise GroundTruthMetricsConversionError(
            "ground-truth inputs must exactly cover the suite cells: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.candidate-", dir=destination.parent)
    )
    validation_temporary = Path(tempfile.mkdtemp(
        prefix=".ground-truth-validation-",
        dir=canonical_root.parent,
    ))
    try:
        case_records = []
        output_paths = []
        manifest_replacements = []
        for result in sorted(
            converted,
            key=lambda item: tuple(item["identity"][key] for key in (
                "hardware_id", "model_id", "case_id"
            )),
        ):
            identity = result["identity"]
            relative_case = Path(
                identity["hardware_id"], identity["model_id"], identity["case_id"]
            )
            candidate_case_dir = temporary / _CANDIDATE_DATASET_PATH / relative_case
            candidate_case_dir.mkdir(parents=True, exist_ok=False)
            metrics_relative_path = (
                _CANDIDATE_DATASET_PATH / relative_case / _METRICS_RELATIVE_PATH
            )
            metrics_path = temporary / metrics_relative_path
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_bytes = _canonical_json_bytes(result["metrics"])
            metrics_path.write_bytes(metrics_bytes)
            metrics_sha256 = _sha256_bytes(metrics_bytes)
            replacement_paths = {
                Path("manifest.json"),
                _METRICS_RELATIVE_PATH,
                _ALIGNMENT_STEPS_RELATIVE_PATH,
            }
            if result["target"]["operation"] == "create":
                request_bytes = result["request_bytes"]
                if not isinstance(request_bytes, bytes):
                    raise TypeError("converted request_bytes must be bytes")
                (candidate_case_dir / "requests.csv").write_bytes(request_bytes)
            alignment_bytes = result["alignment_bytes"]
            if not isinstance(alignment_bytes, bytes):
                raise TypeError("converted alignment_bytes must be bytes")
            updated_manifest = _updated_case_manifest(
                result["manifest"],
                metrics_sha256=metrics_sha256,
                ground_truth_source=result["raw_source"],
            )
            updated_alignment_bytes = _updated_alignment_steps_bytes(
                alignment_bytes,
                raw_source_set_sha256(updated_manifest["raw_sources"]),
            )
            compressed_alignment = write_compressed_canonical_bytes(
                candidate_case_dir / _ALIGNMENT_STEPS_RELATIVE_PATH,
                updated_alignment_bytes,
            )
            updated_alignment_entry = dict(_mapping(
                updated_manifest["alignment_steps"], "manifest alignment_steps"
            ))
            updated_alignment_entry["sha256"] = _sha256_bytes(
                updated_alignment_bytes
            )
            updated_alignment_entry["compressed_sha256"] = _sha256_bytes(
                compressed_alignment
            )
            updated_alignment_entry["compression"] = COMPRESSION_ALGORITHM
            updated_manifest["alignment_steps"] = updated_alignment_entry
            manifest_bytes = _canonical_json_bytes(updated_manifest)
            (candidate_case_dir / "manifest.json").write_bytes(manifest_bytes)
            if result["target"]["operation"] == "create":
                validate_compact_case(candidate_case_dir)
            elif result["target"]["operation"] == "replace":
                _validate_case_overlay(
                    canonical_root / relative_case,
                    candidate_case_dir,
                    validation_root=validation_temporary / relative_case,
                    replacement_paths=tuple(replacement_paths),
                )
            else:
                raise AssertionError(
                    f"unknown candidate target operation: {result['target']['operation']!r}"
                )
            manifest_sha256 = _sha256_bytes(manifest_bytes)
            manifest_replacements.append((
                result["target"]["capture_case_manifest_sha256"],
                manifest_sha256,
            ))
            output_paths.append(destination / metrics_relative_path)
            case_records.append({
                "identity": identity,
                "target": result["target"],
                "source": result["source"],
                "metrics": {
                    "path": metrics_relative_path.as_posix(),
                    "sha256": metrics_sha256,
                },
                "manifest": {
                    "path": (
                        _CANDIDATE_DATASET_PATH / relative_case / "manifest.json"
                    ).as_posix(),
                    "sha256": manifest_sha256,
                },
                "removed_paths": [],
            })
        candidate_suite_bytes = _rewrite_suite_manifest_bytes(
            suite_bytes,
            manifest_replacements,
            source_dataset_root=suite["dataset_root"],
        )
        candidate_suite_relative_path = (
            _CANDIDATE_SUITE_DIRECTORY / suite_path.name
        )
        candidate_suite_path = temporary / candidate_suite_relative_path
        candidate_suite_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_suite_path.write_bytes(candidate_suite_bytes)
        candidate_suite = _json_object(
            candidate_suite_bytes, "candidate suite manifest"
        )
        _validate_suite(
            candidate_suite,
            dataset_root=temporary / _CANDIDATE_DATASET_PATH,
        )
        _validate_candidate_suite_bindings(
            candidate_suite,
            temporary / _CANDIDATE_DATASET_PATH,
            canonical_root,
        )
        source_suite_relative_path = (
            _CANDIDATE_SOURCE_SUITE_DIRECTORY / suite_path.name
        )
        source_suite_path = temporary / source_suite_relative_path
        source_suite_path.parent.mkdir(parents=True, exist_ok=True)
        source_suite_path.write_bytes(suite_bytes)
        capture_suite_entries = _write_capture_suite_snapshots(
            temporary, capture_suites
        )
        capture_source_snapshot_entries = _write_capture_source_snapshots(
            temporary, capture_source_snapshots
        )
        if scheduler_attestation is None:
            attestation_entry = None
        else:
            attestation_path = temporary / _CANDIDATE_ATTESTATION_PATH
            attestation_path.parent.mkdir(parents=True, exist_ok=True)
            attestation_bytes = scheduler_attestation["source_bytes"]
            if not isinstance(attestation_bytes, bytes):
                raise TypeError("scheduler-policy attestation source_bytes must be bytes")
            attestation_path.write_bytes(attestation_bytes)
            attestation_entry = {
                "path": _CANDIDATE_ATTESTATION_PATH.as_posix(),
                "sha256": scheduler_attestation["sha256"],
            }
        candidate_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "sharegpt_ground_truth_metrics_candidate_set",
            "canonical_dataset_root": _CANDIDATE_DATASET_PATH.as_posix(),
            "source_suite_manifest": {
                "path": source_suite_relative_path.as_posix(),
                "sha256": suite_sha256,
            },
            "capture_suite_manifests": capture_suite_entries,
            "candidate_suite_manifest": {
                "path": candidate_suite_relative_path.as_posix(),
                "sha256": _sha256_bytes(candidate_suite_bytes),
            },
            "scheduler_policy_attestation": attestation_entry,
            "capture_source_snapshots": capture_source_snapshot_entries,
            "case_count": len(case_records),
            "cases": case_records,
        }
        (temporary / "candidate_manifest.json").write_bytes(
            _canonical_json_bytes(candidate_manifest)
        )
        os.replace(temporary, destination)
        return tuple(output_paths)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if validation_temporary.exists():
            shutil.rmtree(validation_temporary)


def _convert_one(
    source_path: Path,
    *,
    dataset_root: Path,
    capture_dataset_root: Path,
    suite: Mapping[str, object],
    current_suite_sha256: str,
    capture_suites: Mapping[str, Mapping[str, object]],
    scheduler_attestation: Mapping[str, object] | None,
    capture_source_snapshots: Mapping[str, Mapping[str, object]],
    template_hardware_id: str | None,
    converter_source_commit: str | None,
) -> dict[str, object]:
    source_bytes = source_path.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    payload = _json_object(source_bytes, f"ground truth {source_path}")
    if payload["schema_version"] != 1:
        raise GroundTruthMetricsConversionError(
            f"ground truth schema_version must equal 1: {source_path}"
        )
    if payload["status"] != "success":
        raise GroundTruthMetricsConversionError(
            f"ground truth status must equal success: {source_path}"
        )
    workload = _mapping(payload["workload"], "ground truth workload")
    if workload["kind"] != "sharegpt_canonical_schedule":
        raise GroundTruthMetricsConversionError(
            f"unsupported ground truth workload kind: {source_path}"
        )
    identity = _mapping(workload["identity"], "ground truth identity")
    _exact_keys(identity, _GROUND_TRUTH_IDENTITY_KEYS, "ground truth identity")
    if identity["source_kind"] != "sharegpt":
        raise GroundTruthMetricsConversionError(
            f"ground truth source_kind must equal sharegpt: {source_path}"
        )
    for key in ("hardware_id", "model_id", "case_id"):
        _nonempty_string(identity[key], f"ground truth identity {key}")
    recorded_suite_sha256 = _sha256(
        workload["suite_manifest_sha256"],
        "ground truth recorded suite manifest SHA256",
    )
    if recorded_suite_sha256 not in capture_suites:
        raise GroundTruthMetricsConversionError(
            "ground truth is not bound to any supplied capture suite manifest: "
            f"{recorded_suite_sha256}"
        )
    capture_suite_record = capture_suites[recorded_suite_sha256]
    capture_suite = _mapping(
        capture_suite_record["payload"], "capture suite manifest"
    )

    target_case_dir = (
        dataset_root
        / identity["hardware_id"]
        / identity["model_id"]
        / identity["case_id"]
    ).resolve()
    if dataset_root != target_case_dir and dataset_root not in target_case_dir.parents:
        raise GroundTruthMetricsConversionError(
            f"ground-truth identity escapes the canonical dataset root: {source_path}"
        )
    if template_hardware_id is None:
        if not target_case_dir.is_dir():
            raise GroundTruthMetricsConversionError(
                "canonical target case does not exist; an explicit "
                f"template_hardware_id is required: {target_case_dir}"
            )
        case_dir = target_case_dir
        target_operation = "replace"
    else:
        if target_case_dir.exists() or target_case_dir.is_symlink():
            raise GroundTruthMetricsConversionError(
                "template_hardware_id selects create mode, but the canonical "
                f"target case already exists: {target_case_dir}"
            )
        template_case_dir = (
            dataset_root
            / template_hardware_id
            / identity["model_id"]
            / identity["case_id"]
        ).resolve()
        if (
            dataset_root != template_case_dir
            and dataset_root not in template_case_dir.parents
        ):
            raise GroundTruthMetricsConversionError(
                "template hardware identity escapes the canonical dataset root"
            )
        if not template_case_dir.is_dir():
            raise GroundTruthMetricsConversionError(
                f"canonical template case does not exist: {template_case_dir}"
            )
        case_dir = template_case_dir
        target_operation = "create"
    try:
        target_manifest = _json_object(
            (case_dir / "manifest.json").read_bytes(),
            "canonical target/template manifest",
        )
        if target_manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version must equal {SCHEMA_VERSION}"
            )
        load_compact_case(case_dir)
    except (TypeError, ValueError, OSError) as error:
        raise GroundTruthMetricsConversionError(
            "canonical target/template must be a valid compact schema-v2 case: "
            f"{case_dir}: {error}"
        ) from error
    manifest_path = (case_dir / "manifest.json").resolve(strict=True)
    manifest_bytes = manifest_path.read_bytes()
    manifest = _json_object(manifest_bytes, f"case manifest {manifest_path}")
    manifest_identity = _mapping(manifest["identity"], "case manifest identity")
    _exact_keys(manifest_identity, _COMPACT_IDENTITY_KEYS, "case manifest identity")
    if target_operation == "replace":
        if any(
            identity[key] != manifest_identity[key]
            for key in _GROUND_TRUTH_IDENTITY_KEYS
        ):
            raise GroundTruthMetricsConversionError(
                f"ground-truth identity disagrees with case manifest: {source_path}"
            )
        template_binding = None
    elif target_operation == "create":
        if manifest_identity["hardware_id"] != template_hardware_id:
            raise GroundTruthMetricsConversionError(
                "canonical template manifest hardware identity disagrees with "
                "template_hardware_id"
            )
        identity_fields = tuple(
            key for key in _GROUND_TRUTH_IDENTITY_KEYS if key != "hardware_id"
        )
        if any(manifest_identity[key] != identity[key] for key in identity_fields):
            raise GroundTruthMetricsConversionError(
                "canonical template identity disagrees with the ground-truth "
                "model or case identity"
            )
        template_binding = {
            "identity": dict(manifest_identity),
            "converter": dict(_mapping(
                manifest["converter"], "template case manifest converter"
            )),
        }
    else:
        raise AssertionError(f"unknown target operation: {target_operation!r}")
    manifest_sha256 = _sha256_bytes(manifest_bytes)

    capture_case_dir = (
        capture_dataset_root
        / identity["hardware_id"]
        / identity["model_id"]
        / identity["case_id"]
    ).resolve(strict=True)
    if (
        capture_dataset_root != capture_case_dir
        and capture_dataset_root not in capture_case_dir.parents
    ):
        raise GroundTruthMetricsConversionError(
            f"ground-truth identity escapes the capture dataset root: {source_path}"
        )
    try:
        load_compact_case(capture_case_dir)
    except (TypeError, ValueError, OSError) as error:
        raise GroundTruthMetricsConversionError(
            f"capture target must be a valid compact schema-v2 case: {capture_case_dir}: {error}"
        ) from error
    capture_manifest_path = (
        capture_case_dir / "manifest.json"
    ).resolve(strict=True)
    capture_manifest_bytes = capture_manifest_path.read_bytes()
    capture_manifest = _json_object(
        capture_manifest_bytes,
        f"capture case manifest {capture_manifest_path}",
    )
    capture_manifest_identity = _mapping(
        capture_manifest["identity"], "capture case manifest identity"
    )
    _exact_keys(
        capture_manifest_identity,
        _COMPACT_IDENTITY_KEYS,
        "capture case manifest identity",
    )
    if any(
        identity[key] != capture_manifest_identity[key]
        for key in _GROUND_TRUTH_IDENTITY_KEYS
    ):
        raise GroundTruthMetricsConversionError(
            f"ground-truth identity disagrees with capture case manifest: {source_path}"
        )
    capture_manifest_sha256 = _sha256_bytes(capture_manifest_bytes)
    _require_matching_sha(
        workload["case_manifest_sha256"], capture_manifest_sha256,
        "ground truth capture case manifest SHA256",
    )

    manifest_requests = _mapping(manifest["requests"], "case manifest requests")
    requests_path = (case_dir / manifest_requests["path"]).resolve(strict=True)
    if case_dir not in requests_path.parents:
        raise GroundTruthMetricsConversionError(
            f"case requests path escapes its case directory: {requests_path}"
        )
    request_bytes = requests_path.read_bytes()
    requests_sha256 = _sha256_bytes(request_bytes)
    _require_matching_sha(
        manifest_requests["sha256"], requests_sha256,
        "case manifest requests SHA256",
    )
    capture_manifest_requests = _mapping(
        capture_manifest["requests"], "capture case manifest requests"
    )
    capture_requests_path = (
        capture_case_dir / capture_manifest_requests["path"]
    ).resolve(strict=True)
    if capture_case_dir not in capture_requests_path.parents:
        raise GroundTruthMetricsConversionError(
            f"capture requests path escapes its case directory: {capture_requests_path}"
        )
    capture_request_bytes = capture_requests_path.read_bytes()
    capture_requests_sha256 = _sha256_bytes(capture_request_bytes)
    _require_matching_sha(
        capture_manifest_requests["sha256"],
        capture_requests_sha256,
        "capture case manifest requests SHA256",
    )
    _require_matching_sha(
        capture_requests_sha256,
        requests_sha256,
        "capture and canonical requests SHA256",
    )
    if capture_request_bytes != request_bytes:
        raise GroundTruthMetricsConversionError(
            "capture and canonical requests bytes disagree"
        )
    _require_matching_sha(
        workload["requests_sha256"], capture_requests_sha256,
        "ground truth requests SHA256",
    )
    if template_binding is not None:
        template_binding["case_manifest_sha256"] = manifest_sha256
        template_binding["requests_sha256"] = requests_sha256
    rows = read_canonical_request_rows(requests_path)
    if manifest_requests["row_count"] != len(rows):
        raise GroundTruthMetricsConversionError(
            f"case manifest request count disagrees with requests.csv: {source_path}"
        )
    if capture_manifest_requests["row_count"] != len(rows):
        raise GroundTruthMetricsConversionError(
            "capture case manifest request count disagrees with requests.csv: "
            f"{source_path}"
        )
    if workload["request_count"] != len(rows):
        raise GroundTruthMetricsConversionError(
            f"ground truth request count disagrees with requests.csv: {source_path}"
        )
    request_stream_sha256 = _request_stream_sha256(rows)
    _require_matching_sha(
        workload["request_stream_sha256"], request_stream_sha256,
        "ground truth request-stream SHA256",
    )

    suite_cell = _unique_suite_entry(
        suite["cells"],
        lambda cell: all(cell[key] == identity[key] for key in (
            "hardware_id", "model_id", "case_id"
        )),
        "suite cell",
    )
    _require_matching_sha(
        suite_cell["manifest_sha256"], capture_manifest_sha256,
        "suite cell manifest SHA256",
    )
    _require_matching_sha(
        suite_cell["requests_sha256"], requests_sha256,
        "suite cell requests SHA256",
    )
    capture_suite_cell = _unique_suite_entry(
        capture_suite["cells"],
        lambda cell: all(cell[key] == identity[key] for key in (
            "hardware_id", "model_id", "case_id"
        )),
        "capture suite cell",
    )
    _require_matching_sha(
        capture_suite_cell["manifest_sha256"], capture_manifest_sha256,
        "capture suite cell manifest SHA256",
    )
    _require_matching_sha(
        capture_suite_cell["requests_sha256"], requests_sha256,
        "capture suite cell requests SHA256",
    )
    suite_model = _unique_suite_entry(
        suite["models"],
        lambda model: model["dataset_model_id"] == identity["model_id"],
        "suite model",
    )
    model_config_sha256 = _sha256(
        workload["model_config_sha256"], "ground truth model config SHA256"
    )
    _require_matching_sha(
        suite_model["model_config_sha256"], model_config_sha256,
        "suite model config SHA256",
    )
    capture_suite_model = _unique_suite_entry(
        capture_suite["models"],
        lambda model: model["dataset_model_id"] == identity["model_id"],
        "capture suite model",
    )
    _require_matching_sha(
        capture_suite_model["model_config_sha256"], model_config_sha256,
        "capture suite model config SHA256",
    )

    runtime = _mapping(
        capture_manifest["runtime"], "capture case manifest runtime"
    )
    server = _mapping(payload["server"], "ground truth server")
    if server["hardware_id"] != identity["hardware_id"]:
        raise GroundTruthMetricsConversionError(
            f"ground truth server hardware disagrees with identity: {source_path}"
        )
    engine = _mapping(server["engine"], "ground truth server engine")
    tool_version = _nonempty_string(
        server["client_vllm_version"], "ground truth client vLLM version"
    )
    for field in _PARALLEL_FIELDS:
        _integer(runtime[field], f"case manifest {field}", minimum=1)
        _integer(engine[field], f"ground truth engine {field}", minimum=1)
        if engine[field] != runtime[field]:
            raise GroundTruthMetricsConversionError(
                f"ground truth engine {field} disagrees with case manifest: {source_path}"
            )
    _nonempty_string(runtime["dtype"], "case manifest dtype")
    if engine["dtype"] != runtime["dtype"]:
        raise GroundTruthMetricsConversionError(
            f"ground truth engine dtype disagrees with case manifest: {source_path}"
        )
    target_qps = _positive_number(runtime["target_qps"], "case target_qps")
    expected_qps = _positive_number(
        workload["expected_qps"], "ground truth expected_qps"
    )
    if expected_qps != target_qps:
        raise GroundTruthMetricsConversionError(
            f"ground truth QPS disagrees with case manifest: {source_path}"
        )
    _validate_capture_runtime_contract(
        engine,
        capture_suite,
        contract_name="capture suite",
        source_path=source_path,
    )
    _validate_capture_runtime_contract(
        engine,
        suite,
        contract_name="current suite",
        source_path=source_path,
    )
    uses_external_capture_manifest = capture_manifest_path != manifest_path
    if uses_external_capture_manifest:
        _validate_external_capture_contract(
            workload,
            capture_runtime=runtime,
            engine=engine,
            client_vllm_version=tool_version,
            rows=rows,
            source_path=source_path,
        )
    scheduler_policy_binding = _scheduler_policy_binding(
        engine,
        identity=identity,
        ground_truth_sha256=source_sha256,
        attestation=scheduler_attestation,
        source_path=source_path,
    )

    metrics = _convert_metrics(
        payload["metrics"],
        rows=rows,
        requests_sha256=requests_sha256,
        source_sha256=source_sha256,
        source_path=source_path,
    )
    capture_provenance = _mapping(
        payload["provenance"], "ground truth capture provenance"
    )
    capture_code_identity = _capture_code_identity(capture_provenance)
    capture_source_snapshot_binding = _capture_source_snapshot_binding(
        capture_code_identity,
        identity=identity,
        ground_truth_sha256=source_sha256,
        snapshots=capture_source_snapshots,
        source_path=source_path,
    )
    candidate_manifest = capture_manifest if uses_external_capture_manifest else manifest
    if target_operation == "create":
        if converter_source_commit is None:
            raise AssertionError("create mode requires converter source commit")
        candidate_manifest = json.loads(json.dumps(capture_manifest, allow_nan=False))
        candidate_manifest["converter"] = {
            "implementation_id": _GROUND_TRUTH_CONVERTER_ID,
            "source_commit": converter_source_commit,
            "selection_rule": "middle_complete_work_bearing_step_v1",
        }
    alignment_entry = _mapping(
        capture_manifest["alignment_steps"],
        "capture case manifest alignment_steps",
    )
    alignment_path = (capture_case_dir / alignment_entry["path"]).resolve(
        strict=True)
    if capture_case_dir not in alignment_path.parents:
        raise GroundTruthMetricsConversionError(
            f"capture alignment_steps path escapes its case directory: {alignment_path}"
        )
    alignment_bytes = read_compressed_canonical_bytes(
        alignment_path,
        compressed_sha256=alignment_entry["compressed_sha256"],
        content_sha256=alignment_entry["sha256"],
        label=ALIGNMENT_FILENAME,
    )
    return {
        "identity": dict(identity),
        "target": {
            "operation": target_operation,
            "canonical_target_exists": target_operation == "replace",
            "template": template_binding,
            "case_manifest_sha256": manifest_sha256,
            "canonical_source_case_manifest_sha256": manifest_sha256,
            "capture_case_manifest_sha256": capture_manifest_sha256,
            "requests_sha256": requests_sha256,
            "request_stream_sha256": request_stream_sha256,
            "model_config_sha256": model_config_sha256,
            "recorded_capture_suite_manifest_sha256": recorded_suite_sha256,
            "current_suite_manifest_sha256": current_suite_sha256,
            "scheduler_policy_binding": scheduler_policy_binding,
            "capture_source_snapshot_binding": capture_source_snapshot_binding,
        },
        "source": {
            "relative_path": source_path.name,
            "bytes": len(source_bytes),
            "sha256": source_sha256,
            "capture_provenance_sha256": _canonical_json_sha256(
                capture_provenance
            ),
            "capture_code_identity": capture_code_identity,
            "tool": "vllm-serving-runtime",
            "tool_version": tool_version,
        },
        "raw_source": {
            "archive_id": _GROUND_TRUTH_ARCHIVE_ID,
            "role": "production-ground-truth",
            "relative_path": source_path.name,
            "bytes": len(source_bytes),
            "sha256": _sha256_bytes(source_bytes),
            "tool": "vllm-serving-runtime",
            "tool_version": tool_version,
        },
        "manifest": candidate_manifest,
        "capture_manifest": capture_manifest,
        "alignment_bytes": alignment_bytes,
        "metrics": metrics,
        "request_bytes": request_bytes,
    }


def _validate_case_overlay(
    canonical_case: Path,
    candidate_case: Path,
    *,
    validation_root: Path,
    replacement_paths: Sequence[Path],
) -> None:
    """Validate a delta through a disposable view of unchanged canonical files."""

    if not canonical_case.is_dir():
        raise GroundTruthMetricsConversionError(
            f"canonical case directory does not exist: {canonical_case}"
        )
    validation_root.mkdir(parents=True, exist_ok=False)
    replacements = {
        _relative_path(path.as_posix(), "candidate replacement path")
        for path in replacement_paths
    }
    missing_replacements = sorted(
        path.as_posix()
        for path in replacements
        if not (candidate_case / path).is_file()
    )
    if missing_replacements:
        raise GroundTruthMetricsConversionError(
            "candidate replacement paths do not exist: "
            f"{missing_replacements!r}"
        )
    for source_path in sorted(canonical_case.rglob("*")):
        relative_path = source_path.relative_to(canonical_case)
        target_path = validation_root / relative_path
        if source_path.is_symlink():
            raise GroundTruthMetricsConversionError(
                f"canonical case payload must not contain symlinks: {source_path}"
            )
        if source_path.is_dir():
            target_path.mkdir(exist_ok=True)
        elif source_path.is_file():
            if relative_path not in replacements:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                os.link(source_path, target_path)
        else:
            raise GroundTruthMetricsConversionError(
                f"canonical case payload contains an unsupported file type: {source_path}"
            )
    for relative_path in replacements:
        target_path = validation_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(candidate_case / relative_path, target_path)
    try:
        validate_compact_case(validation_root)
    finally:
        shutil.rmtree(validation_root)


def _updated_case_manifest(
    original: Mapping[str, object],
    *,
    metrics_sha256: str,
    ground_truth_source: Mapping[str, object],
) -> dict[str, object]:
    manifest = json.loads(json.dumps(original, allow_nan=False))
    raw_sources = _sequence(manifest["raw_sources"], "manifest raw_sources")
    archive_ids = [
        _mapping(source, "manifest raw source")["archive_id"]
        for source in raw_sources
    ]
    if _GROUND_TRUTH_ARCHIVE_ID in archive_ids:
        raise GroundTruthMetricsConversionError(
            "canonical manifest already contains production-ground-truth raw source"
        )
    updated_raw_sources = [dict(_mapping(source, "manifest raw source")) for source in raw_sources]
    updated_raw_sources.append(dict(ground_truth_source))
    manifest["raw_sources"] = updated_raw_sources

    manifest["metrics"] = {
        "status": "available",
        "reason": None,
        "path": _METRICS_RELATIVE_PATH.as_posix(),
        "sha256": metrics_sha256,
    }

    requests = dict(_mapping(manifest["requests"], "manifest requests"))
    cached_hit_evidence = requests["cached_hit_value_evidence"]
    if cached_hit_evidence is not None:
        updated_evidence = dict(
            _mapping(cached_hit_evidence, "cached-hit value evidence")
        )
        updated_evidence["raw_source_set_sha256"] = raw_source_set_sha256(
            updated_raw_sources
        )
        requests["cached_hit_value_evidence"] = updated_evidence
    manifest["requests"] = requests
    provenance = dict(_mapping(manifest["provenance"], "manifest provenance"))
    provenance["raw_source_set_sha256"] = raw_source_set_sha256(
        updated_raw_sources
    )
    manifest["provenance"] = provenance
    return manifest


def _updated_alignment_steps_bytes(
    source_bytes: bytes,
    raw_source_sha256: str,
) -> bytes:
    """Rebind unchanged structural samples to the case's expanded raw set."""

    alignment = json.loads(json.dumps(
        _json_object(source_bytes, "capture alignment_steps"),
        allow_nan=False,
    ))
    phases = _mapping(alignment["phases"], "capture alignment phases")
    for phase in ("prefill", "decode", "mixed"):
        entry = _mapping(phases[phase], f"capture alignment phase {phase}")
        if entry["status"] == "not_observed":
            continue
        if entry["status"] != "available":
            raise GroundTruthMetricsConversionError(
                f"unsupported capture alignment phase status: {entry['status']!r}"
            )
        sample = _mapping(entry["sample"], f"capture alignment phase {phase} sample")
        provenance = dict(_mapping(
            sample["provenance"], f"capture alignment phase {phase} provenance"
        ))
        provenance["raw_source_set_sha256"] = raw_source_sha256
        sample["provenance"] = provenance
    return _canonical_json_bytes(alignment)


def _validate_external_capture_contract(
    workload: Mapping[str, object],
    *,
    capture_runtime: Mapping[str, object],
    engine: Mapping[str, object],
    client_vllm_version: str,
    rows: Sequence[CanonicalRequestRow],
    source_path: Path,
) -> None:
    """Bind an external prefix-enabled capture to its zero-hit proof."""

    prefix_evidence = _mapping(
        workload["prefix_cache_reuse_evidence"],
        "ground truth prefix-cache reuse evidence",
    )
    _exact_keys(
        prefix_evidence,
        (
            "kind",
            "logical_prefix_hit_observability",
            "enable_prefix_caching",
            "kv_block_size_tokens",
            "request_count",
            "prompt_token_stream_sha256",
            "shared_prefix_block_pair_count",
            "max_shared_prefix_blocks",
            "effective_reuse_representative_tokens",
            "representative_provenance",
        ),
        "ground truth prefix-cache reuse evidence",
    )
    if prefix_evidence["kind"] != "static_prompt_token_prefix_analysis_v1":
        raise GroundTruthMetricsConversionError(
            "external capture requires static prompt-token prefix analysis"
        )
    if prefix_evidence["logical_prefix_hit_observability"] != (
        "unavailable_without_instrumentation"
    ):
        raise GroundTruthMetricsConversionError(
            "external capture prefix-hit observability is unsupported"
        )
    if prefix_evidence["enable_prefix_caching"] is not True:
        raise GroundTruthMetricsConversionError(
            "external capture prefix evidence must record enabled prefix caching"
        )
    if prefix_evidence["representative_provenance"] != (
        "scheduler_equivalent_representative"
    ):
        raise GroundTruthMetricsConversionError(
            "external capture uses unsupported prefix representative provenance"
        )
    evidence_request_count = _integer(
        prefix_evidence["request_count"],
        "ground truth prefix evidence request_count",
        minimum=1,
    )
    if evidence_request_count != len(rows):
        raise GroundTruthMetricsConversionError(
            "ground truth prefix evidence request_count disagrees with requests.csv: "
            f"{source_path}"
        )
    _sha256(
        prefix_evidence["prompt_token_stream_sha256"],
        "ground truth prefix evidence prompt token stream SHA256",
    )
    evidence_block_size = _integer(
        prefix_evidence["kv_block_size_tokens"],
        "ground truth prefix evidence kv_block_size_tokens",
        minimum=1,
    )
    engine_block_size = _integer(
        engine["kv_block_size_tokens"],
        "ground truth engine kv_block_size_tokens",
        minimum=1,
    )
    if evidence_block_size != engine_block_size:
        raise GroundTruthMetricsConversionError(
            "ground truth prefix evidence KV block size disagrees with engine: "
            f"{source_path}"
        )
    for field in (
        "shared_prefix_block_pair_count",
        "max_shared_prefix_blocks",
        "effective_reuse_representative_tokens",
    ):
        value = _integer(
            prefix_evidence[field],
            f"ground truth prefix evidence {field}",
            minimum=0,
        )
        if value != 0:
            raise GroundTruthMetricsConversionError(
                f"external capture requires zero {field}"
            )
    if any(row.cached_hit_tokens != 0 for row in rows):
        raise GroundTruthMetricsConversionError(
            "external capture zero-prefix proof requires zero cached-hit CSV values"
        )
    if engine["enable_prefix_caching"] is not True:
        raise GroundTruthMetricsConversionError(
            "external capture engine must enable prefix caching"
        )
    if capture_runtime["capture_enable_prefix_caching"] is not True:
        raise GroundTruthMetricsConversionError(
            "capture runtime must record enabled prefix caching"
        )
    capture_seed = _integer(
        capture_runtime["seed"], "capture runtime seed", minimum=0
    )
    engine_seed = _integer(
        engine["seed"], "ground truth engine seed", minimum=0
    )
    workload_seed = _integer(
        workload["seed"], "ground truth workload seed", minimum=0
    )
    if engine_seed != capture_seed or workload_seed != capture_seed:
        raise GroundTruthMetricsConversionError(
            "ground truth engine/workload seed disagrees with capture runtime: "
            f"{source_path}"
        )
    capture_runtime_version = _nonempty_string(
        capture_runtime["runtime_version"], "capture runtime version"
    )
    if client_vllm_version != capture_runtime_version:
        raise GroundTruthMetricsConversionError(
            "ground truth client vLLM version disagrees with capture runtime: "
            f"{source_path}"
        )


def _rewrite_suite_manifest_bytes(
    source_bytes: bytes,
    manifest_replacements: Sequence[tuple[object, str]],
    *,
    source_dataset_root: object,
) -> bytes:
    """Retarget a capture suite to the canonical candidate overlay."""

    updated = source_bytes
    source_dataset_root_value = _nonempty_string(
        source_dataset_root, "suite source dataset_root"
    )
    old_dataset_fragment = (
        f'"dataset_root": "{source_dataset_root_value}"'.encode("utf-8")
    )
    new_dataset_fragment = (
        f'"dataset_root": "{_CANDIDATE_DATASET_PATH.as_posix()}"'.encode(
            "utf-8"
        )
    )
    occurrence_count = updated.count(old_dataset_fragment)
    if occurrence_count != 1:
        raise GroundTruthMetricsConversionError(
            "suite source dataset_root must occur exactly once; "
            f"found {occurrence_count} occurrences"
        )
    updated = updated.replace(old_dataset_fragment, new_dataset_fragment, 1)
    for old_value, new_value in manifest_replacements:
        old_sha256 = _sha256(old_value, "suite source cell manifest SHA256")
        new_sha256 = _sha256(new_value, "suite candidate cell manifest SHA256")
        old_fragment = f'"manifest_sha256": "{old_sha256}"'.encode("utf-8")
        new_fragment = f'"manifest_sha256": "{new_sha256}"'.encode("utf-8")
        occurrence_count = updated.count(old_fragment)
        if occurrence_count != 1:
            raise GroundTruthMetricsConversionError(
                "suite source cell manifest SHA256 must occur exactly once; "
                f"found {occurrence_count} occurrences for {old_sha256}"
            )
        updated = updated.replace(old_fragment, new_fragment, 1)
    return updated


def _validate_candidate_suite_bindings(
    suite: Mapping[str, object],
    candidate_dataset_root: Path,
    canonical_dataset_root: Path,
) -> None:
    for value in _sequence(suite["cells"], "candidate suite cells"):
        cell = _mapping(value, "candidate suite cell")
        case_dir = (
            candidate_dataset_root
            / cell["hardware_id"]
            / cell["model_id"]
            / cell["case_id"]
        )
        manifest_path = case_dir / "manifest.json"
        candidate_requests_path = case_dir / "requests.csv"
        if candidate_requests_path.is_file():
            requests_path = candidate_requests_path
        else:
            requests_path = (
                canonical_dataset_root
                / cell["hardware_id"]
                / cell["model_id"]
                / cell["case_id"]
                / "requests.csv"
            )
        if _sha256_bytes(manifest_path.read_bytes()) != cell["manifest_sha256"]:
            raise GroundTruthMetricsConversionError(
                f"candidate suite manifest SHA256 disagrees with {case_dir}"
            )
        if _sha256_bytes(requests_path.read_bytes()) != cell["requests_sha256"]:
            raise GroundTruthMetricsConversionError(
                f"candidate suite requests SHA256 disagrees with {case_dir}"
            )


def _load_capture_suites(
    paths: Sequence[str | Path],
    *,
    dataset_root: Path,
) -> dict[str, Mapping[str, object]]:
    if not paths:
        raise GroundTruthMetricsConversionError(
            "at least one capture suite manifest is required"
        )
    suites = {}
    for value in paths:
        path = Path(value).resolve(strict=True)
        source_bytes = path.read_bytes()
        digest = _sha256_bytes(source_bytes)
        if digest in suites:
            raise GroundTruthMetricsConversionError(
                f"duplicate capture suite manifest SHA256: {digest}"
            )
        suite = _json_object(source_bytes, f"capture suite manifest {path}")
        _validate_suite(suite, dataset_root=dataset_root)
        suites[digest] = {
            "payload": suite,
            "source_bytes": source_bytes,
        }
    return suites


def _write_capture_suite_snapshots(
    root: Path,
    suites: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    entries = []
    for digest in sorted(suites):
        record = suites[digest]
        source_bytes = record["source_bytes"]
        if not isinstance(source_bytes, bytes):
            raise TypeError("capture suite source_bytes must be bytes")
        relative_path = _CANDIDATE_CAPTURE_SUITES_PATH / f"{digest}.json"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_bytes)
        entries.append({"path": relative_path.as_posix(), "sha256": digest})
    return entries


def _load_scheduler_policy_attestation(
    path: str | Path | None,
) -> Mapping[str, object] | None:
    if path is None:
        return None
    source_path = Path(path).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    payload = _json_object(source_bytes, "scheduler-policy attestation")
    _exact_keys(
        payload,
        ("schema_version", "kind", "attestor", "basis", "cases"),
        "scheduler-policy attestation",
    )
    if payload["schema_version"] != 1:
        raise GroundTruthMetricsConversionError(
            "scheduler-policy attestation schema_version must equal 1"
        )
    if payload["kind"] != "sharegpt_ground_truth_scheduler_policy_attestation":
        raise GroundTruthMetricsConversionError(
            "scheduler-policy attestation kind is unsupported"
        )
    _nonempty_string(payload["attestor"], "scheduler-policy attestor")
    if payload["basis"] != "capture_operator_confirmation_v1":
        raise GroundTruthMetricsConversionError(
            "scheduler-policy attestation basis must equal "
            "capture_operator_confirmation_v1"
        )
    cases = _sequence(payload["cases"], "scheduler-policy attestation cases")
    if not cases:
        raise GroundTruthMetricsConversionError(
            "scheduler-policy attestation requires at least one case"
        )
    seen = set()
    for index, value in enumerate(cases):
        case = _mapping(value, f"scheduler-policy attestation case {index}")
        _exact_keys(
            case,
            (
                "identity",
                "ground_truth_sha256",
                "scheduler_policy",
                "evidence_kind",
            ),
            f"scheduler-policy attestation case {index}",
        )
        identity = _mapping(
            case["identity"], f"scheduler-policy attestation case {index} identity"
        )
        _exact_keys(
            identity,
            _GROUND_TRUTH_IDENTITY_KEYS,
            f"scheduler-policy attestation case {index} identity",
        )
        identity_tuple = tuple(
            identity[key] for key in _GROUND_TRUTH_IDENTITY_KEYS
        )
        if identity_tuple in seen:
            raise GroundTruthMetricsConversionError(
                "scheduler-policy attestation contains duplicate identity"
            )
        seen.add(identity_tuple)
        _sha256(
            case["ground_truth_sha256"],
            f"scheduler-policy attestation case {index} ground truth SHA256",
        )
        if case["scheduler_policy"] != _EXPECTED_SCHEDULER_POLICY:
            raise GroundTruthMetricsConversionError(
                "scheduler-policy attestation must assert fcfs"
            )
        if case["evidence_kind"] != "ground_truth_runner_configuration_review_v1":
            raise GroundTruthMetricsConversionError(
                "scheduler-policy attestation evidence_kind is unsupported"
            )
    return {
        "payload": payload,
        "sha256": _sha256_bytes(source_bytes),
        "relative_path": _CANDIDATE_ATTESTATION_PATH.as_posix(),
        "source_bytes": source_bytes,
    }


def _load_capture_source_snapshots(
    manifest_paths: Sequence[str | Path],
) -> dict[str, Mapping[str, object]]:
    by_ground_truth_sha256 = {}
    seen_manifest_sha256 = set()
    for value in manifest_paths:
        manifest_path = Path(value).resolve(strict=True)
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        if manifest_sha256 in seen_manifest_sha256:
            raise GroundTruthMetricsConversionError(
                f"duplicate capture source snapshot manifest: {manifest_sha256}"
            )
        seen_manifest_sha256.add(manifest_sha256)
        payload = _json_object(
            manifest_bytes, f"capture source snapshot manifest {manifest_path}"
        )
        _exact_keys(
            payload,
            ("schema_version", "kind", "source_snapshot", "code_identity", "ground_truths"),
            "capture source snapshot manifest",
        )
        if payload["schema_version"] != 1:
            raise GroundTruthMetricsConversionError(
                "capture source snapshot manifest schema_version must equal 1"
            )
        if payload["kind"] != "sharegpt_ground_truth_capture_source_snapshot":
            raise GroundTruthMetricsConversionError(
                "capture source snapshot manifest kind is unsupported"
            )
        source_snapshot = _mapping(
            payload["source_snapshot"], "capture source snapshot"
        )
        _exact_keys(
            source_snapshot,
            (
                "hash_method",
                "sha256",
                "base_commit",
                "runner_source_commit",
                "changed_paths",
            ),
            "capture source snapshot",
        )
        if source_snapshot["hash_method"] != "midlayer-git-worktree-content-v1":
            raise GroundTruthMetricsConversionError(
                "capture source snapshot hash_method must equal "
                "midlayer-git-worktree-content-v1"
            )
        base_commit = _git_commit(
            source_snapshot["base_commit"], "capture source snapshot base_commit"
        )
        runner_source_commit = _git_commit(
            source_snapshot["runner_source_commit"],
            "capture source snapshot runner_source_commit",
        )
        changed_paths = _validate_snapshot_changed_paths(
            source_snapshot["changed_paths"],
            runner_source_commit=runner_source_commit,
        )
        snapshot_identity = {
            "hash_method": source_snapshot["hash_method"],
            "base_commit": base_commit,
            "runner_source_commit": runner_source_commit,
            "changed_paths": changed_paths,
        }
        snapshot_sha256 = _canonical_json_sha256(snapshot_identity)
        _require_matching_sha(
            source_snapshot["sha256"],
            snapshot_sha256,
            "capture source snapshot SHA256",
        )
        code_identity = _validate_snapshot_code_identity(payload["code_identity"])
        if code_identity["git_commit"] != base_commit:
            raise GroundTruthMetricsConversionError(
                "capture source snapshot base_commit disagrees with code identity"
            )
        if len(changed_paths) != code_identity["changed_path_count"]:
            raise GroundTruthMetricsConversionError(
                "capture source snapshot changed paths disagree with code identity count"
            )
        ground_truths = _sequence(
            payload["ground_truths"], "capture source snapshot ground truths"
        )
        if not ground_truths:
            raise GroundTruthMetricsConversionError(
                "capture source snapshot must bind at least one ground truth"
            )
        for index, item in enumerate(ground_truths):
            ground_truth = _mapping(
                item, f"capture source snapshot ground truth {index}"
            )
            _exact_keys(
                ground_truth,
                ("identity", "ground_truth_sha256"),
                f"capture source snapshot ground truth {index}",
            )
            identity = _mapping(
                ground_truth["identity"],
                f"capture source snapshot ground truth {index} identity",
            )
            _exact_keys(
                identity,
                _GROUND_TRUTH_IDENTITY_KEYS,
                f"capture source snapshot ground truth {index} identity",
            )
            ground_truth_sha256 = _sha256(
                ground_truth["ground_truth_sha256"],
                f"capture source snapshot ground truth {index} SHA256",
            )
            if ground_truth_sha256 in by_ground_truth_sha256:
                raise GroundTruthMetricsConversionError(
                    "capture source snapshots bind one ground truth more than once"
                )
            output_directory = _CANDIDATE_SOURCE_SNAPSHOTS_PATH / manifest_sha256
            by_ground_truth_sha256[ground_truth_sha256] = {
                "identity": dict(identity),
                "code_identity": code_identity,
                "manifest_source_bytes": manifest_bytes,
                "manifest_sha256": manifest_sha256,
                "manifest_candidate_path": (
                    output_directory / "manifest.json"
                ).as_posix(),
                "snapshot_sha256": snapshot_sha256,
                "changed_paths": changed_paths,
            }
    return by_ground_truth_sha256


def _validate_snapshot_changed_paths(
    value: object,
    *,
    runner_source_commit: str,
) -> list[dict[str, object]]:
    entries = _sequence(value, "capture source snapshot changed_paths")
    if not entries:
        raise GroundTruthMetricsConversionError(
            "capture source snapshot changed_paths cannot be empty"
        )
    normalized = []
    seen_paths = set()
    runner_paths = 0
    for index, item in enumerate(entries):
        entry = _mapping(item, f"capture source snapshot changed path {index}")
        _exact_keys(
            entry,
            (
                "relative_path",
                "classification",
                "content_sha256",
                "source_commit",
                "lfs_oid_sha256",
            ),
            f"capture source snapshot changed path {index}",
        )
        relative_path = _relative_path(
            entry["relative_path"],
            f"capture source snapshot changed path {index} relative_path",
        ).as_posix()
        if relative_path in seen_paths:
            raise GroundTruthMetricsConversionError(
                "capture source snapshot changed paths contain a duplicate path"
            )
        seen_paths.add(relative_path)
        classification = entry["classification"]
        content_sha256 = _sha256(
            entry["content_sha256"],
            f"capture source snapshot changed path {index} content SHA256",
        )
        if classification == "runner_source_from_commit":
            runner_paths += 1
            if relative_path != "evaluation/serving/experiments/sharegpt_ground_truth.py":
                raise GroundTruthMetricsConversionError(
                    "runner source snapshot path must identify sharegpt_ground_truth.py"
                )
            if entry["source_commit"] != runner_source_commit:
                raise GroundTruthMetricsConversionError(
                    "runner source path must bind runner_source_commit"
                )
            if entry["lfs_oid_sha256"] is not None:
                raise GroundTruthMetricsConversionError(
                    "runner source path cannot declare an LFS OID"
                )
            source_commit = runner_source_commit
            lfs_oid_sha256 = None
        elif classification == "canonical_lfs_payload_without_clean_filter":
            if not relative_path.startswith("serving_dataset/sharegpt/ascend_910b/"):
                raise GroundTruthMetricsConversionError(
                    "LFS materialization path must be an Ascend ShareGPT case file"
                )
            if not (
                relative_path.endswith("/manifest.json")
                or relative_path.endswith("/requests.csv")
            ):
                raise GroundTruthMetricsConversionError(
                    "LFS materialization path must be manifest.json or requests.csv"
                )
            if entry["source_commit"] is not None:
                raise GroundTruthMetricsConversionError(
                    "LFS materialization path cannot declare source_commit"
                )
            lfs_oid_sha256 = _sha256(
                entry["lfs_oid_sha256"],
                f"capture source snapshot changed path {index} LFS OID",
            )
            if lfs_oid_sha256 != content_sha256:
                raise GroundTruthMetricsConversionError(
                    "canonical LFS payload content SHA256 must equal its LFS OID"
                )
            source_commit = None
        else:
            raise GroundTruthMetricsConversionError(
                "capture source snapshot changed path classification is unsupported"
            )
        normalized.append({
            "relative_path": relative_path,
            "classification": classification,
            "content_sha256": content_sha256,
            "source_commit": source_commit,
            "lfs_oid_sha256": lfs_oid_sha256,
        })
    if runner_paths != 1:
        raise GroundTruthMetricsConversionError(
            "capture source snapshot must contain exactly one runner source path"
        )
    return sorted(normalized, key=lambda item: item["relative_path"])


def _validate_snapshot_code_identity(value: object) -> dict[str, object]:
    identity = _mapping(value, "capture source snapshot code identity")
    _exact_keys(
        identity,
        (
            "git_commit",
            "worktree_state",
            "changed_path_count",
            "git_diff_hash",
            "status_porcelain_sha256",
        ),
        "capture source snapshot code identity",
    )
    if identity["worktree_state"] != "dirty_not_reproducible_by_commit_alone":
        raise GroundTruthMetricsConversionError(
            "capture source snapshot is only valid for dirty ground-truth runs"
        )
    git_commit = identity["git_commit"]
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise GroundTruthMetricsConversionError(
            "capture source snapshot git_commit must be a lowercase 40-character digest"
        )
    return {
        "git_commit": git_commit,
        "worktree_state": identity["worktree_state"],
        "changed_path_count": _integer(
            identity["changed_path_count"],
            "capture source snapshot changed_path_count",
            minimum=1,
        ),
        "git_diff_hash": _sha256(
            identity["git_diff_hash"], "capture source snapshot git_diff_hash"
        ),
        "status_porcelain_sha256": _sha256(
            identity["status_porcelain_sha256"],
            "capture source snapshot status_porcelain_sha256",
        ),
    }


def _capture_source_snapshot_binding(
    code_identity: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    ground_truth_sha256: str,
    snapshots: Mapping[str, Mapping[str, object]],
    source_path: Path,
) -> Mapping[str, object] | None:
    state = code_identity["worktree_state"]
    if state == "clean_reproducible_by_commit":
        if ground_truth_sha256 in snapshots:
            raise GroundTruthMetricsConversionError(
                f"clean ground truth must not use a capture source snapshot: {source_path}"
            )
        return None
    if state != "dirty_not_reproducible_by_commit_alone":
        raise AssertionError(f"unknown validated worktree state: {state!r}")
    if ground_truth_sha256 not in snapshots:
        raise GroundTruthMetricsConversionError(
            f"dirty ground truth requires a capture source snapshot: {source_path}"
        )
    snapshot = snapshots[ground_truth_sha256]
    if dict(_mapping(snapshot["identity"], "snapshot identity")) != dict(identity):
        raise GroundTruthMetricsConversionError(
            f"capture source snapshot identity mismatch: {source_path}"
        )
    snapshot_code_identity = _mapping(
        snapshot["code_identity"], "snapshot code identity"
    )
    comparable_code_identity = {
        key: code_identity[key]
        for key in (
            "git_commit",
            "worktree_state",
            "changed_path_count",
            "git_diff_hash",
            "status_porcelain_sha256",
        )
    }
    if dict(snapshot_code_identity) != comparable_code_identity:
        raise GroundTruthMetricsConversionError(
            f"capture source snapshot code identity mismatch: {source_path}"
        )
    snapshot_paths = tuple(
        entry["relative_path"]
        for entry in _sequence(snapshot["changed_paths"], "snapshot changed paths")
    )
    if tuple(code_identity["changed_paths"]) != snapshot_paths:
        raise GroundTruthMetricsConversionError(
            f"capture source snapshot changed paths mismatch: {source_path}"
        )
    return {
        "manifest_path": snapshot["manifest_candidate_path"],
        "manifest_sha256": snapshot["manifest_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }


def _validate_capture_source_snapshot_coverage(
    converted: Sequence[Mapping[str, object]],
    snapshots: Mapping[str, Mapping[str, object]],
) -> None:
    required = {
        _mapping(result["source"], "converted source")["sha256"]
        for result in converted
        if _mapping(
            _mapping(result["source"], "converted source")["capture_code_identity"],
            "capture code identity",
        )["worktree_state"] == "dirty_not_reproducible_by_commit_alone"
    }
    supplied = set(snapshots)
    if supplied != required:
        raise GroundTruthMetricsConversionError(
            "capture source snapshot coverage must exactly equal dirty ground truths: "
            f"missing={sorted(required - supplied)!r}, "
            f"unexpected={sorted(supplied - required)!r}"
        )


def _write_capture_source_snapshots(
    root: Path,
    snapshots: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    written = {}
    for ground_truth_sha256 in sorted(snapshots):
        snapshot = snapshots[ground_truth_sha256]
        manifest_sha256 = snapshot["manifest_sha256"]
        if manifest_sha256 in written:
            continue
        manifest_path = root / snapshot["manifest_candidate_path"]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_bytes = snapshot["manifest_source_bytes"]
        if not isinstance(manifest_bytes, bytes):
            raise TypeError("capture source snapshot manifest bytes must be bytes")
        manifest_path.write_bytes(manifest_bytes)
        written[manifest_sha256] = {
            "manifest_path": snapshot["manifest_candidate_path"],
            "manifest_sha256": manifest_sha256,
            "snapshot_sha256": snapshot["snapshot_sha256"],
        }
    return [written[key] for key in sorted(written)]


def _validate_capture_runtime_contract(
    engine: Mapping[str, object],
    suite: Mapping[str, object],
    *,
    contract_name: str,
    source_path: Path,
) -> None:
    prediction = _mapping(suite["prediction_contract"], "suite prediction contract")
    exact_fields = (
        ("tensor_parallel_size", "tensor_parallel_size"),
        ("pipeline_parallel_size", "pipeline_parallel_size"),
        ("expert_parallel_size", "expert_parallel_size"),
        ("enable_chunked_prefill", "enable_chunked_prefill"),
        ("enable_prefix_caching", "enable_prefix_caching"),
        ("max_num_batched_tokens", "max_batched_tokens"),
        ("max_num_seqs", "max_requests"),
        ("max_model_len", "max_model_len"),
        ("kv_block_size_tokens", "kv_block_size_tokens"),
    )
    for engine_field, prediction_field in exact_fields:
        if engine[engine_field] != prediction[prediction_field]:
            raise GroundTruthMetricsConversionError(
                f"ground truth engine configuration disagrees with {contract_name} "
                f"prediction contract at {engine_field}: {source_path}"
            )
    if engine["dtype"] != "bfloat16" or prediction["compute_dtype"] != "bf16":
        raise GroundTruthMetricsConversionError(
            f"ground truth dtype disagrees with {contract_name} prediction contract: "
            f"{source_path}"
        )


def _scheduler_policy_binding(
    engine: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    ground_truth_sha256: str,
    attestation: Mapping[str, object] | None,
    source_path: Path,
) -> dict[str, object]:
    if "scheduler_policy" in engine:
        if engine["scheduler_policy"] != _EXPECTED_SCHEDULER_POLICY:
            raise GroundTruthMetricsConversionError(
                f"ground truth scheduler policy must equal fcfs: {source_path}"
            )
        return {
            "scheduler_policy": _EXPECTED_SCHEDULER_POLICY,
            "provenance_kind": "ground_truth_payload",
            "attestation_relative_path": None,
            "attestation_sha256": None,
        }
    if attestation is None:
        raise GroundTruthMetricsConversionError(
            "ground truth does not record scheduler_policy; a structured "
            f"scheduler-policy attestation is required: {source_path}"
        )
    payload = _mapping(attestation["payload"], "scheduler-policy attestation")
    matches = []
    for value in _sequence(payload["cases"], "scheduler-policy attestation cases"):
        case = _mapping(value, "scheduler-policy attestation case")
        case_identity = _mapping(case["identity"], "attestation case identity")
        if all(
            case_identity[key] == identity[key]
            for key in _GROUND_TRUTH_IDENTITY_KEYS
        ):
            matches.append(case)
    if len(matches) != 1:
        raise GroundTruthMetricsConversionError(
            "scheduler-policy attestation must contain exactly one matching case: "
            f"{source_path}"
        )
    case = matches[0]
    _require_matching_sha(
        case["ground_truth_sha256"],
        ground_truth_sha256,
        "scheduler-policy attestation ground truth SHA256",
    )
    return {
        "scheduler_policy": _EXPECTED_SCHEDULER_POLICY,
        "provenance_kind": "capture_operator_attestation",
        "attestation_relative_path": attestation["relative_path"],
        "attestation_sha256": attestation["sha256"],
    }


def _validate_attestation_coverage(
    converted: Sequence[Mapping[str, object]],
    attestation: Mapping[str, object] | None,
) -> None:
    required = {
        tuple(
            _mapping(result["identity"], "converted identity")[key]
            for key in _GROUND_TRUTH_IDENTITY_KEYS
        )
        for result in converted
        if _mapping(
            _mapping(result["target"], "converted target")["scheduler_policy_binding"],
            "scheduler policy binding",
        )["provenance_kind"] == "capture_operator_attestation"
    }
    if not required:
        if attestation is not None:
            raise GroundTruthMetricsConversionError(
                "scheduler-policy attestation was supplied but no result requires it"
            )
        return
    if attestation is None:
        raise AssertionError("required scheduler-policy attestation is missing")
    payload = _mapping(attestation["payload"], "scheduler-policy attestation")
    supplied = set()
    for value in _sequence(payload["cases"], "scheduler-policy attestation cases"):
        case = _mapping(value, "scheduler-policy attestation case")
        identity = _mapping(case["identity"], "scheduler-policy attestation identity")
        supplied.add(
            tuple(identity[key] for key in _GROUND_TRUTH_IDENTITY_KEYS)
        )
    if supplied != required:
        raise GroundTruthMetricsConversionError(
            "scheduler-policy attestation identities must exactly equal the cases "
            f"that lack recorded scheduler_policy: missing={sorted(required - supplied)!r}, "
            f"unexpected={sorted(supplied - required)!r}"
        )


def _capture_code_identity(provenance: Mapping[str, object]) -> dict[str, object]:
    _exact_keys(
        provenance,
        (
            "repository_root",
            "git_commit",
            "worktree_state",
            "changed_path_count",
            "git_diff_hash",
            "status_porcelain_sha256",
            "git_status_short",
            "argv",
        ),
        "ground truth capture provenance",
    )
    git_commit = provenance["git_commit"]
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise GroundTruthMetricsConversionError(
            "ground truth capture git_commit must be a lowercase 40-character digest"
        )
    state = provenance["worktree_state"]
    if state not in (
        "clean_reproducible_by_commit",
        "dirty_not_reproducible_by_commit_alone",
    ):
        raise GroundTruthMetricsConversionError(
            "ground truth capture worktree_state is unsupported"
        )
    changed_path_count = _integer(
        provenance["changed_path_count"],
        "ground truth capture changed_path_count",
        minimum=0,
    )
    status = _sequence(
        provenance["git_status_short"], "ground truth capture git_status_short"
    )
    if len(status) != changed_path_count or any(
        not isinstance(value, str) or not value for value in status
    ):
        raise GroundTruthMetricsConversionError(
            "ground truth capture changed_path_count disagrees with git_status_short"
        )
    if state == "clean_reproducible_by_commit" and changed_path_count != 0:
        raise GroundTruthMetricsConversionError(
            "clean ground truth capture cannot declare changed paths"
        )
    if state == "dirty_not_reproducible_by_commit_alone" and changed_path_count == 0:
        raise GroundTruthMetricsConversionError(
            "dirty ground truth capture must declare changed paths"
        )
    changed_paths = sorted(
        _git_status_relative_path(value, index)
        for index, value in enumerate(status)
    )
    return {
        "git_commit": git_commit,
        "worktree_state": state,
        "changed_path_count": changed_path_count,
        "git_diff_hash": _sha256(
            provenance["git_diff_hash"], "ground truth capture git_diff_hash"
        ),
        "status_porcelain_sha256": _sha256(
            provenance["status_porcelain_sha256"],
            "ground truth capture status_porcelain_sha256",
        ),
        "changed_paths": changed_paths,
    }


def _git_status_relative_path(value: object, index: int) -> str:
    if not isinstance(value, str) or len(value) < 4 or value[2] != " ":
        raise GroundTruthMetricsConversionError(
            f"ground truth git_status_short entry {index} is not porcelain-v1 text"
        )
    if " -> " in value:
        raise GroundTruthMetricsConversionError(
            "renamed paths are unsupported in capture source snapshot binding"
        )
    return _relative_path(
        value[3:], f"ground truth git_status_short entry {index} path"
    ).as_posix()


def _convert_metrics(
    value: object,
    *,
    rows: Sequence[CanonicalRequestRow],
    requests_sha256: str,
    source_sha256: str,
    source_path: Path,
) -> dict[str, object]:
    raw = _mapping(value, "ground truth metrics")
    duration_s = _positive_number(raw["duration_s"], "ground truth duration_s")
    request_count = len(rows)
    if raw["completed_requests"] != request_count:
        raise GroundTruthMetricsConversionError(
            f"completed request count disagrees with requests.csv: {source_path}"
        )
    records = _sequence(raw["per_request"], "ground truth per_request")
    if len(records) != request_count:
        raise GroundTruthMetricsConversionError(
            f"per-request metric count disagrees with requests.csv: {source_path}"
        )
    converted = []
    total_input_tokens = 0
    total_output_tokens = 0
    for request_idx, (value, row) in enumerate(zip(records, rows)):
        record = _mapping(value, f"ground truth request {request_idx}")
        expected_request_id = f"{_REQUEST_ID_PREFIX}{request_idx:012d}"
        if record["request_id"] != expected_request_id:
            raise GroundTruthMetricsConversionError(
                "ground-truth request IDs must preserve canonical physical-row order; "
                f"expected {expected_request_id!r} at row {request_idx}: {source_path}"
            )
        input_tokens = _integer(
            record["input_tokens"], f"ground truth request {request_idx} input_tokens",
            minimum=1,
        )
        output_tokens = _integer(
            record["output_tokens"], f"ground truth request {request_idx} output_tokens",
            minimum=1,
        )
        if input_tokens != row.input_tokens or output_tokens != row.output_tokens:
            raise GroundTruthMetricsConversionError(
                f"ground-truth token counts disagree with requests.csv row {request_idx}: "
                f"{source_path}"
            )
        scheduled_arrival_s = _nonnegative_number(
            record["scheduled_arrival_s"],
            f"ground truth request {request_idx} scheduled_arrival_s",
        )
        if scheduled_arrival_s != row.arrival_time_ms * 0.001:
            raise GroundTruthMetricsConversionError(
                f"ground-truth arrival disagrees with requests.csv row {request_idx}: "
                f"{source_path}"
            )
        ttft_ms = _nonnegative_number(
            record["ttft_ms"], f"ground truth request {request_idx} ttft_ms"
        )
        tpot_ms = _nonnegative_number(
            record["tpot_ms"], f"ground truth request {request_idx} tpot_ms"
        )
        e2e_ms = _nonnegative_number(
            record["e2e_ms"], f"ground truth request {request_idx} e2e_ms"
        )
        if e2e_ms < ttft_ms:
            raise GroundTruthMetricsConversionError(
                f"ground-truth E2E latency is smaller than TTFT at row {request_idx}: "
                f"{source_path}"
            )
        if output_tokens == 1 and tpot_ms != 0.0:
            raise GroundTruthMetricsConversionError(
                f"one-token request must report zero TPOT at row {request_idx}: "
                f"{source_path}"
            )
        converted.append({
            "request_idx": request_idx,
            "actual_output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_latency_ms": e2e_ms,
        })
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
    if raw["total_input_tokens"] != total_input_tokens:
        raise GroundTruthMetricsConversionError(
            f"ground-truth total input tokens disagree with requests.csv: {source_path}"
        )
    if raw["total_output_tokens"] != total_output_tokens:
        raise GroundTruthMetricsConversionError(
            f"ground-truth total output tokens disagree with requests.csv: {source_path}"
        )
    request_throughput = _positive_number(
        raw["request_throughput_per_s"],
        "ground truth request throughput",
    )
    output_throughput = _positive_number(
        raw["output_token_throughput_per_s"],
        "ground truth output token throughput",
    )
    _require_close(
        request_throughput, request_count / duration_s,
        "ground truth request throughput",
    )
    _require_close(
        output_throughput, total_output_tokens / duration_s,
        "ground truth output token throughput",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "requests_sha256": requests_sha256,
        "metric_provenance": {
            "source_kind": "serving_runtime",
            "source_archive_id": _GROUND_TRUTH_ARCHIVE_ID,
            "source_sha256": source_sha256,
            "profiler_attached": False,
        },
        "per_request": converted,
        "summary": {
            "request_count": request_count,
            "request_throughput_per_s": request_throughput,
            "output_token_throughput_per_s": output_throughput,
            "total_token_throughput_per_s": (
                total_input_tokens + total_output_tokens
            ) / duration_s,
        },
    }


def _request_stream_sha256(rows: Sequence[CanonicalRequestRow]) -> str:
    payload = [
        {
            "arrival_time_s": row.arrival_time_ms * 0.001,
            "request_id": f"{_REQUEST_ID_PREFIX}{request_idx:012d}",
            "input_tokens": row.input_tokens,
            "output_tokens": row.output_tokens,
            "prefix_hit_tokens": row.cached_hit_tokens,
        }
        for request_idx, row in enumerate(rows)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _discover_ground_truth_paths(
    inputs: Sequence[str | Path],
) -> tuple[Path, ...]:
    if not inputs:
        raise GroundTruthMetricsConversionError(
            "at least one ground-truth input is required"
        )
    paths = []
    for value in inputs:
        path = Path(value)
        if path.is_file():
            if path.name != _GROUND_TRUTH_FILENAME:
                raise GroundTruthMetricsConversionError(
                    f"ground-truth file must be named {_GROUND_TRUTH_FILENAME}: {path}"
                )
            paths.append(path.resolve())
        elif path.is_dir():
            paths.extend(
                candidate.resolve()
                for candidate in path.rglob(_GROUND_TRUTH_FILENAME)
            )
        else:
            raise GroundTruthMetricsConversionError(
                f"ground-truth input does not exist: {path}"
            )
    unique = sorted(set(paths))
    if not unique:
        raise GroundTruthMetricsConversionError(
            "no ground_truth.json files were discovered"
        )
    if len(unique) != len(paths):
        raise GroundTruthMetricsConversionError(
            "duplicate ground-truth input paths were supplied"
        )
    return tuple(unique)


def _validate_candidate_destination(destination: Path, dataset_root: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"ground-truth metrics candidate root already exists: {destination}"
        )
    if destination == dataset_root or dataset_root in destination.parents:
        raise GroundTruthMetricsConversionError(
            "candidate root must not be the canonical Serving dataset or one of "
            "its descendants"
        )
    if destination in dataset_root.parents:
        raise GroundTruthMetricsConversionError(
            "candidate root must not contain the canonical Serving dataset"
        )


def _validate_suite(
    suite: Mapping[str, object],
    *,
    dataset_root: Path,
) -> None:
    if suite["schema_version"] != 1:
        raise GroundTruthMetricsConversionError(
            "ShareGPT suite manifest schema_version must equal 1"
        )
    declared_dataset_root = _relative_path(
        suite["dataset_root"], "ShareGPT suite dataset_root"
    )
    declared_parts = declared_dataset_root.parts
    actual_parts = dataset_root.resolve(strict=True).parts
    if (
        declared_parts[-2:] != ("serving_dataset", "sharegpt")
        or len(actual_parts) < len(declared_parts)
        or actual_parts[-len(declared_parts):] != declared_parts
    ):
        raise GroundTruthMetricsConversionError(
            "ShareGPT suite dataset_root disagrees with the supplied dataset root"
        )
    models = _sequence(suite["models"], "suite models")
    cells = _sequence(suite["cells"], "suite cells")
    if not models or not cells:
        raise GroundTruthMetricsConversionError(
            "ShareGPT suite requires non-empty models and cells"
        )
    for index, value in enumerate(models):
        model = _mapping(value, f"suite model {index}")
        _nonempty_string(model["dataset_model_id"], f"suite model {index} id")
        _sha256(model["model_config_sha256"], f"suite model {index} config SHA256")
    for index, value in enumerate(cells):
        cell = _mapping(value, f"suite cell {index}")
        for key in ("hardware_id", "model_id", "case_id"):
            _nonempty_string(cell[key], f"suite cell {index} {key}")
        _sha256(cell["manifest_sha256"], f"suite cell {index} manifest SHA256")
        _sha256(cell["requests_sha256"], f"suite cell {index} requests SHA256")


def _unique_suite_entry(
    values: object,
    predicate: object,
    description: str,
) -> Mapping[str, object]:
    records = [_mapping(value, description) for value in _sequence(values, description)]
    matches = [record for record in records if predicate(record)]
    if len(matches) != 1:
        raise GroundTruthMetricsConversionError(
            f"expected exactly one matching {description}; found {len(matches)}"
        )
    return matches[0]


def _json_object(payload: bytes, description: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GroundTruthMetricsConversionError(
            f"invalid JSON for {description}: {error}"
        ) from error
    return _mapping(value, description)


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GroundTruthMetricsConversionError(f"{description} must be an object")
    return value


def _sequence(value: object, description: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise GroundTruthMetricsConversionError(f"{description} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: Sequence[str], description: str
) -> None:
    if set(value) != set(expected):
        raise GroundTruthMetricsConversionError(
            f"{description} keys must exactly equal {tuple(expected)!r}"
        )


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundTruthMetricsConversionError(
            f"{description} must be a non-empty string"
        )
    return value


def _integer(value: object, description: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GroundTruthMetricsConversionError(
            f"{description} must be an integer >= {minimum}"
        )
    return value


def _nonnegative_number(value: object, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise GroundTruthMetricsConversionError(
            f"{description} must be finite and non-negative"
        )
    return float(value)


def _positive_number(value: object, description: str) -> float:
    result = _nonnegative_number(value, description)
    if result <= 0:
        raise GroundTruthMetricsConversionError(f"{description} must be positive")
    return result


def _sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GroundTruthMetricsConversionError(
            f"{description} must be a lowercase SHA256 digest"
        )
    return value


def _git_commit(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GroundTruthMetricsConversionError(
            f"{description} must be a lowercase 40-character digest"
        )
    return value


def _require_matching_sha(value: object, expected: str, description: str) -> None:
    actual = _sha256(value, description)
    if actual != expected:
        raise GroundTruthMetricsConversionError(
            f"{description} mismatch: {actual} != {expected}"
        )


def _require_close(actual: float, expected: float, description: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise GroundTruthMetricsConversionError(
            f"{description} disagrees with duration and token totals"
        )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative_path(value: object, description: str) -> Path:
    text = _nonempty_string(value, description)
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != text:
        raise GroundTruthMetricsConversionError(
            f"{description} must be a normalized relative POSIX path"
        )
    return Path(*pure.parts)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
