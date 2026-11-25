"""Publish and merge evidence-only FA serving campaign records.

The release format preserves exact Event/Timeline golden targets, duration-free counter candidates, launch provenance, and the explicit C500 counter-unavailable state.  Merging retains this schema beside the historical detailed profiler records in ``flash_attn.json.zst`` and deliberately does not synthesize missing profiler sections.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from src.kernel.dataset_storage import (
    compress_kernel_dataset_content,
    load_kernel_dataset_document,
)

from dataset_pipeline.kernel_dataset.fa_serving_capture import (
    build_capture_plan,
    fa_serving_timing_source,
    load_capture_manifest,
)
from dataset_pipeline.kernel_dataset.fa_serving_capture_report import (
    _counter_records,
    _require_complete_pair,
    _sha256,
    _timing_records,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


_HARDWARE_IDS = EVALUATED_HARDWARE_IDS


def _write_compressed_json(payload: dict, output_path: Path, prefix: str) -> Path:
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    compressed = compress_kernel_dataset_content(encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=output_path.parent, prefix=prefix,
                delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(compressed)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _selected_duration_ns(hardware_id: str, timing: dict) -> float:
    source = fa_serving_timing_source(hardware_id)
    return float(timing[source.container][source.duration_field])


def _attributes(case: dict, hardware_id: str) -> dict:
    workload = case["capture_metadata"]["workload"]
    requested_splits = case["capture_metadata"]["requested_num_splits"]
    if hardware_id == "ascend_910b":
        executed_splits = None
    elif requested_splits is None:
        raise ValueError(f"{hardware_id} capture has no requested split")
    else:
        executed_splits = requested_splits
    return {
        "B": workload["B"],
        "S": workload["KV_LEN"],
        "Q_LEN": workload["Q_LEN"],
        "Q_REQUESTED": workload["Q_LEN"],
        "KV_LEN": workload["KV_LEN"],
        "KV_REQUESTED": workload["KV_LEN"],
        "HQ": workload["HQ"],
        "HKV": workload["HKV"],
        "D": workload["D"],
        "SP": executed_splits,
        "SP_REQUESTED": requested_splits,
        "Q_LENS": workload["Q_LENS"],
        "KV_LENS": workload["KV_LENS"],
        "split_policy": workload["split_policy"],
        "kv_layout": workload["kv_layout"],
        "kv_dist": workload["kv_dist"],
        "causal": workload["causal"],
        "dtype": workload["dtype"],
        "layer": f"GQA_{workload['HQ']}_{workload['HKV']}_D{workload['D']}",
    }


def _repository_relative_paths(value, repository_root: Path):
    """Normalize retained evidence paths without weakening provenance."""
    if isinstance(value, dict):
        return {
            key: _repository_relative_paths(item, repository_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _repository_relative_paths(item, repository_root)
            for item in value
        ]
    if not isinstance(value, str) or not Path(value).is_absolute():
        return value
    path = Path(value).resolve()
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"FA serving evidence path escapes repository root: {path}") from error


def _record(case: dict, hardware_id: str, evidence: dict,
            repository_root: Path) -> dict:
    counter = evidence["counter"]
    if counter is None:
        counter_payload = {
            "status": "unavailable",
            "reason": evidence["counter_unavailable_reason"],
            "candidate_path": None,
            "candidate_sha256": None,
        }
    else:
        counter_payload = {
            "status": "available",
            "candidate": counter,
            "candidate_path": str(evidence["counter_path"]),
            "candidate_sha256": _sha256(evidence["counter_path"]),
        }
    timing = evidence["timing"]
    return _repository_relative_paths({
        "op_info": {
            "Type": "FLASHATTN",
            "Attr": _attributes(case, hardware_id),
            "evaluation": {
                "timing": {
                    "granularity": "op",
                    "source": fa_serving_timing_source(
                        hardware_id).source_label,
                    "total_duration_ns": _selected_duration_ns(hardware_id, timing),
                },
            },
        },
        "capture_evidence": {
            "logical_identity_sha256": case["capture_metadata"]["logical_identity_sha256"],
            "workload_digest": case["workload_digest"],
            "phase": case["capture_metadata"]["phase"],
            "timing": {
                "audit_path": str(evidence["timing_path"]),
                "audit_sha256": _sha256(evidence["timing_path"]),
                "source_snapshot_sha256": timing["source_snapshot"]["sha256"],
            },
            "counter": counter_payload,
        },
    }, repository_root)


def build_release_dataset(*, capture_manifest: Path, timing_root: Path,
                          candidate_root: Path, hardware_id: str,
                          output_path: Path) -> Path:
    """Publish one hardware's complete campaign evidence as a zstd dataset."""
    if hardware_id not in _HARDWARE_IDS:
        raise ValueError(f"unsupported hardware: {hardware_id!r}")
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"release dataset already exists: {output_path}")
    loaded = load_capture_manifest(capture_manifest)
    repository_root = loaded["path"].parents[3]
    plan = build_capture_plan(loaded, hardware_id=hardware_id)
    expected = {item["case_id"]: item for item in plan["plans"]}
    timings = _timing_records(timing_root, hardware_id, expected)
    counters = _counter_records(candidate_root, hardware_id, expected, timings)
    evidence = _require_complete_pair(
        hardware_id=hardware_id, expected=expected, timings=timings,
        counters=counters)
    records = {
        case_id: _record(
            expected[case_id], hardware_id, evidence[case_id], repository_root)
        for case_id in sorted(expected)
    }
    if len(records) != 40:
        raise AssertionError("FA serving release must contain exactly 40 cases")
    payload = {
        "schema_version": 1,
        "kind": "fa_serving_capture_release",
        "hardware_id": hardware_id,
        "capture_manifest": _repository_relative_paths(
            str(capture_manifest.resolve()), repository_root),
        "capture_manifest_sha256": _sha256(capture_manifest),
        "case_count": len(records),
        "counter_available_case_count": sum(
            record["capture_evidence"]["counter"]["status"] == "available"
            for record in records.values()),
        "counter_unavailable_case_count": sum(
            record["capture_evidence"]["counter"]["status"] == "unavailable"
            for record in records.values()),
        "records": records,
    }
    return _write_compressed_json(payload, output_path, ".fa-release-")


def merge_release_into_canonical(
        *, canonical_path: Path, release_path: Path,
        hardware_id: str) -> Path:
    """Merge one complete 40-case FA release into its canonical family file."""

    if hardware_id not in _HARDWARE_IDS:
        raise ValueError(f"unsupported hardware: {hardware_id!r}")
    canonical_path = canonical_path.resolve()
    release_path = release_path.resolve()
    canonical = load_kernel_dataset_document(canonical_path)
    release = load_kernel_dataset_document(release_path)
    if not isinstance(canonical, dict):
        raise TypeError("canonical FA dataset must be a mapping")
    if not isinstance(release, dict):
        raise TypeError("FA serving release must be a mapping")
    required_release_fields = {
        "schema_version", "kind", "hardware_id", "capture_manifest",
        "capture_manifest_sha256", "case_count",
        "counter_available_case_count", "counter_unavailable_case_count",
        "records",
    }
    if set(release) != required_release_fields:
        raise ValueError("FA serving release fields are invalid")
    if (release["schema_version"] != 1
            or release["kind"] != "fa_serving_capture_release"
            or release["hardware_id"] != hardware_id):
        raise ValueError("FA serving release identity is invalid")
    records = release["records"]
    if (not isinstance(records, dict) or release["case_count"] != 40
            or len(records) != 40):
        raise ValueError("FA serving release must contain exactly 40 records")
    available = sum(
        record["capture_evidence"]["counter"]["status"] == "available"
        for record in records.values())
    unavailable = sum(
        record["capture_evidence"]["counter"]["status"] == "unavailable"
        for record in records.values())
    if available != release["counter_available_case_count"]:
        raise ValueError("FA serving release available-counter count is invalid")
    if unavailable != release["counter_unavailable_case_count"]:
        raise ValueError("FA serving release unavailable-counter count is invalid")
    if available + unavailable != 40:
        raise ValueError("FA serving release contains an invalid counter status")
    collisions = sorted(set(canonical).intersection(records))
    if collisions:
        raise ValueError(
            "FA serving release collides with canonical cases: "
            f"{collisions[:3]}")

    release_identity = {
        "schema_version": release["schema_version"],
        "kind": release["kind"],
        "hardware_id": release["hardware_id"],
        "capture_manifest": release["capture_manifest"],
        "capture_manifest_sha256": release["capture_manifest_sha256"],
    }
    merged = dict(canonical)
    for case_id, record in records.items():
        if not isinstance(record, dict):
            raise TypeError(f"FA serving release record {case_id!r} is invalid")
        merged_record = json.loads(json.dumps(record, allow_nan=False))
        merged_record["capture_evidence"]["release"] = release_identity
        merged[case_id] = merged_record
    if len(merged) != len(canonical) + 40:
        raise AssertionError("canonical FA merge did not add exactly 40 records")
    ordered = {case_id: merged[case_id] for case_id in sorted(merged)}
    return _write_compressed_json(
        ordered, canonical_path, ".fa-canonical-merge-")
