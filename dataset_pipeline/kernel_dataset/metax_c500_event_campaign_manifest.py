"""Assemble an immutable MetaX C500 full-Event campaign manifest."""

from __future__ import annotations
from dataset_pipeline.kernel_dataset.metax_c500_event_protocol import (
    load_json_mapping,
    file_sha256,
    require_positive_integer,
)

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from dataset_pipeline.kernel_dataset import metax_c500_event_audit as audit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_OWNER = REPOSITORY_ROOT / "experiment_logs/kernel_golden"
SOURCE_REGISTRY_KEYS = frozenset({
    "schema_version", "artifact_kind", "hardware_id", "sources",
})
SOURCE_REGISTRY_KIND = "metax_c500_full_event_source_registry"
FAMILIES = ("vector", "flash_attn", "gemm", "fused_moe")
NEW_CASE_COUNT = 493
EXISTING_CASE_COUNT = 494






def _exact_keys(
        value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys mismatch: {sorted(set(value) ^ expected)}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")


def _source_registry(
        path: Path,
        ) -> tuple[Mapping[str, Mapping[str, Any]],
                   Mapping[str, Mapping[str, Any]]]:
    registry = load_json_mapping(path, "Event source registry")
    _exact_keys(registry, SOURCE_REGISTRY_KEYS, "Event source registry")
    if (registry["schema_version"] != 1
            or registry["artifact_kind"] != SOURCE_REGISTRY_KIND
            or registry["hardware_id"] != "metax_c500"):
        raise ValueError("Event source registry identity is invalid")
    payloads = registry["sources"]
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError("Event source registry sources must be non-empty")
    validated = {}
    for source_id, payload in payloads.items():
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Event source ID must be a non-empty string")
        source, _ = audit._validate_source(source_id, payload)
        validated[source_id] = source
    if sum(source["role"] == "new_capture"
           for source in validated.values()) != 1:
        raise ValueError("source registry must contain one new-capture source")
    if not any(source["role"] == "legacy_frozen"
               for source in validated.values()):
        raise ValueError("source registry must contain legacy evidence")
    return payloads, validated




def _source_for_record(
        record: Mapping[str, Any], family: str,
        sources: Mapping[str, Mapping[str, Any]], expected_role: str) -> str:
    matches = []
    for source_id, source in sources.items():
        if (source["role"] == expected_role
                and source["baseline_commit"] == record["baseline_commit"]
                and source["source_snapshot_sha256"] ==
                record["source_snapshot_sha256"]
                and source["timer_sanity_sha256"] ==
                record["timer_sanity_sha256"]):
            runner_hashes = source["runner_sha256"]
            if (source["archive_status"] != "complete"
                    or runner_hashes[family] == record["runner_sha256"]):
                matches.append(source_id)
    if len(matches) != 1:
        raise ValueError(
            f"Event record must match exactly one {expected_role} source: "
            f"{family}/{record['case_id']}")
    return matches[0]


def _record_runtime(
        runtime: dict[str, dict[str, Mapping[str, Any]]], source_id: str,
        family: str, record: Mapping[str, Any]) -> None:
    environment = record["environment"]
    if not isinstance(environment, dict) or not environment:
        raise ValueError(
            f"Event environment is invalid: {family}/{record['case_id']}")
    if source_id not in runtime:
        runtime[source_id] = {}
    family_runtime = runtime[source_id]
    if family not in family_runtime:
        family_runtime[family] = environment
    elif family_runtime[family] != environment:
        raise ValueError(
            f"Event runtime is inconsistent: {source_id}/{family}")


def _validate_case_attempt(
        attempt: Mapping[str, Any], case_attempt: Mapping[str, Any],
        selected: set[str], sources: Mapping[str, Mapping[str, Any]],
        runtime: dict[str, dict[str, Mapping[str, Any]]],
        histories: dict[tuple[str, str], list[tuple[int, str, str]]]) -> None:
    _exact_keys(case_attempt, audit.CASE_ATTEMPT_KEYS, "new case attempt")
    case_id = case_attempt["case_id"]
    family = attempt["family"]
    if case_id not in selected:
        raise ValueError(f"new attempt includes unselected case: {family}/{case_id}")
    status = case_attempt["status"]
    if status not in {"measured", "failed"}:
        raise ValueError(f"new case attempt status is invalid: {case_id}")
    if case_attempt["superseded_by_attempt_id"] is not None:
        raise ValueError(f"immutable case embeds supersession: {case_id}")
    root = Path(attempt["root"]).resolve()
    if status == "measured":
        artifact_path = Path(case_attempt["artifact_path"]).resolve()
        expected_path = root / "cases" / f"{case_id}.json"
        if (artifact_path != expected_path
                or file_sha256(artifact_path) != case_attempt["artifact_sha256"]
                or case_attempt["error_path"] is not None
                or case_attempt["error_sha256"] is not None):
            raise ValueError(f"new Event artifact binding is invalid: {case_id}")
        record = load_json_mapping(artifact_path, "new Event artifact")
        if record["family"] != family or record["case_id"] != case_id:
            raise ValueError(f"new Event artifact identity is invalid: {case_id}")
        expected_source = _source_for_record(
            record, family, sources, "new_capture")
        if expected_source != attempt["source_id"]:
            raise ValueError(f"new Event source mismatch: {case_id}")
        _record_runtime(runtime, expected_source, family, record)
    else:
        if (case_attempt["artifact_path"] is not None
                or case_attempt["artifact_sha256"] is not None
                or not isinstance(case_attempt["error_path"], str)
                or not isinstance(case_attempt["error_sha256"], str)
                or file_sha256(Path(case_attempt["error_path"]).resolve()) !=
                case_attempt["error_sha256"]):
            raise ValueError(f"failed Event evidence is invalid: {case_id}")
    identity = (family, case_id)
    if identity not in histories:
        histories[identity] = []
    histories[identity].append((
        attempt["start_wall_time_ns"], attempt["terminal_status"], status))


def _new_attempts(
        campaign_root: Path, selected_by_family: Mapping[str, list[str]],
        sources: Mapping[str, Mapping[str, Any]],
        runtime: dict[str, dict[str, Mapping[str, Any]]],
        ) -> list[tuple[Path, Mapping[str, Any], str]]:
    attempt_dirs = sorted(path.resolve() for path in campaign_root.iterdir()
                          if path.is_dir())
    if not attempt_dirs:
        raise ValueError("formal campaign root contains no attempt directories")
    attempts = []
    attempt_ids = set()
    starts = set()
    histories: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
    selected_sets = {
        family: set(case_ids) for family, case_ids in selected_by_family.items()}
    for root in attempt_dirs:
        attempt_path = root / "attempt.json"
        if (attempt_path.is_symlink()
                or any(path.is_symlink() for path in root.rglob("*"))):
            raise ValueError(f"new attempt contains a symlink: {root}")
        attempt_sha256 = file_sha256(attempt_path)
        attempt = load_json_mapping(attempt_path, "immutable new attempt")
        _exact_keys(attempt, audit.ATTEMPT_KEYS, "immutable new attempt")
        attempt_id = attempt["attempt_id"]
        family = attempt["family"]
        if (attempt["cohort"] != "new" or family not in FAMILIES
                or Path(attempt["root"]).resolve() != root
                or attempt["source_id"] not in sources
                or sources[attempt["source_id"]]["role"] != "new_capture"):
            raise ValueError(f"new attempt identity is invalid: {attempt_id}")
        if attempt["superseded"] is not False:
            raise ValueError(
                f"immutable attempt embeds supersession: {attempt_id}")
        audit._validate_attempt_header(attempt, campaign_root, sources)
        if "root_inventory" in audit.ATTEMPT_KEYS and attempt[
                "root_inventory"] is not None:
            raise ValueError(f"new attempt must not use legacy inventory: {attempt_id}")
        require_positive_integer(attempt["ordinal"], "new attempt ordinal")
        start = require_positive_integer(
            attempt["start_wall_time_ns"], "new attempt start")
        if attempt_id in attempt_ids or start in starts:
            raise ValueError("new attempt IDs and starts must be unique")
        attempt_ids.add(attempt_id)
        starts.add(start)
        planned = attempt["planned_case_ids"]
        case_attempts = attempt["case_attempts"]
        if (not isinstance(planned, list) or not planned
                or len(planned) != len(set(planned))
                or not isinstance(case_attempts, list)
                or len(case_attempts) != len(planned)):
            raise ValueError(f"new attempt plan is invalid: {attempt_id}")
        observed = set()
        for case_attempt in case_attempts:
            case_id = case_attempt["case_id"]
            if case_id in observed:
                raise ValueError(f"new attempt repeats case: {attempt_id}/{case_id}")
            observed.add(case_id)
            _validate_case_attempt(
                attempt, case_attempt, selected_sets[family], sources,
                runtime, histories)
        if observed != set(planned):
            raise ValueError(f"new attempt case inventory mismatch: {attempt_id}")
        if (attempt["terminal_status"] == "completed"
                and any(case_attempt["status"] != "measured"
                        for case_attempt in case_attempts)):
            raise ValueError(f"completed attempt contains a failure: {attempt_id}")
        actual_cases = {
            path.resolve() for path in (root / "cases").glob("*.json")}
        declared_cases = {
            Path(case_attempt["artifact_path"]).resolve()
            for case_attempt in case_attempts
            if case_attempt["artifact_path"] is not None}
        if actual_cases != declared_cases:
            raise ValueError(f"new attempt raw inventory mismatch: {attempt_id}")
        if file_sha256(attempt_path) != attempt_sha256:
            raise ValueError(f"new attempt changed during assembly: {attempt_id}")
        attempts.append((attempt_path, attempt, attempt_sha256))
    attempts.sort(key=lambda item: item[1]["start_wall_time_ns"])
    expected = {
        (family, case_id) for family in FAMILIES
        for case_id in selected_by_family[family]}
    if len(expected) != NEW_CASE_COUNT:
        raise ValueError("selection does not contain the required new case count")
    if set(histories) != expected:
        raise ValueError("new attempt inventory does not cover exact 493 selection")
    for identity, history in histories.items():
        history.sort()
        if history[-1][1:] != ("completed", "measured"):
            raise ValueError(f"latest new attempt failed: {identity}")
    return attempts


def _legacy_groups(
        legacy_evidence: Mapping[tuple[str, str], Mapping[str, Any]],
        existing_by_family: Mapping[str, list[str]],
        sources: Mapping[str, Mapping[str, Any]],
        runtime: dict[str, dict[str, Mapping[str, Any]]],
        ) -> list[Mapping[str, Any]]:
    groups: dict[tuple[str, Path, str], list[tuple[str, Path, str]]] = {}
    for family in FAMILIES:
        for case_id in existing_by_family[family]:
            evidence = legacy_evidence[(family, case_id)]
            path = Path(evidence["path"]).resolve()
            if file_sha256(path) != evidence["sha256"]:
                raise ValueError(f"legacy Event SHA-256 mismatch: {family}/{case_id}")
            if path.parent.name != "cases" or path.stem != case_id:
                raise ValueError(f"legacy Event path is invalid: {family}/{case_id}")
            record = load_json_mapping(path, "legacy Event artifact")
            if record["family"] != family or record["case_id"] != case_id:
                raise ValueError(f"legacy Event identity is invalid: {family}/{case_id}")
            source_id = _source_for_record(
                record, family, sources, "legacy_frozen")
            _record_runtime(runtime, source_id, family, record)
            key = (family, path.parent.parent.resolve(), source_id)
            if key not in groups:
                groups[key] = []
            groups[key].append((case_id, path, evidence["sha256"]))
    if sum(len(records) for records in groups.values()) != EXISTING_CASE_COUNT:
        raise ValueError(
            "legacy Event inventory does not contain the required case count")
    return [{
        "family": key[0], "root": key[1], "source_id": key[2],
        "records": records,
    } for key, records in sorted(
        groups.items(), key=lambda item: (item[0][0], str(item[0][1])))]


def _inventory_bytes(root: Path) -> bytes:
    entries = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"legacy root contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            entries.append((relative.encode("ascii"), file_sha256(path)))
    if not entries:
        raise ValueError(f"legacy root contains no regular files: {root}")
    return b"".join(
        f"{digest}  {relative.decode('ascii')}\n".encode("ascii")
        for relative, digest in sorted(entries))


def _legacy_descriptor(
        *, group: Mapping[str, Any], index: int,
        inventory_reference: Mapping[str, Any]) -> Mapping[str, Any]:
    family = group["family"]
    root = group["root"]
    records = group["records"]
    descriptor = {
        "attempt_id": f"legacy_{family}_{index:03d}",
        "ordinal": None,
        "cohort": "existing_strict",
        "source_id": group["source_id"],
        "family": family,
        "physical_device": None,
        "root": str(root),
        "command_argv": None,
        "pid": None,
        "process_group_id": None,
        "start_wall_time_ns": None,
        "end_wall_time_ns": None,
        "planned_case_ids": [record[0] for record in records],
        "case_attempts": [{
            "case_id": case_id,
            "status": "measured",
            "artifact_path": str(path),
            "artifact_sha256": sha256,
            "error_path": None,
            "error_sha256": None,
            "superseded_by_attempt_id": None,
        } for case_id, path, sha256 in records],
        "terminal_status": "completed",
        "error": None,
        "superseded": False,
        "preflight_snapshot_id": None,
        "during_snapshot_ids": [],
        "postflight_snapshot_id": None,
        "process_isolation": None,
        "wrapper": None,
        "execution_module": None,
        "launcher_pid": None,
        "launcher_process_group_id": None,
        "launcher_command_argv": None,
        "host_uid": None,
        "device_gid": None,
    }
    if "root_inventory" in audit.ATTEMPT_KEYS:
        descriptor["root_inventory"] = inventory_reference
    _exact_keys(descriptor, audit.ATTEMPT_KEYS, "synthetic legacy attempt")
    return descriptor


def _output_root(output_root: Path, protected: set[Path]) -> Path:
    absolute = Path(os.path.abspath(output_root))
    resolved = absolute.resolve()
    owner = OUTPUT_OWNER.resolve()
    if absolute != resolved or output_root.is_symlink():
        raise ValueError("campaign manifest output root must not contain symlinks")
    try:
        relative = resolved.relative_to(owner)
    except ValueError as error:
        raise ValueError(f"campaign manifest output must stay under {owner}") from error
    if not relative.parts:
        raise ValueError("campaign manifest output must be a child of its owner")
    if resolved.exists():
        raise FileExistsError(f"campaign manifest output already exists: {resolved}")
    if any(resolved == path or resolved in path.parents for path in protected):
        raise ValueError("campaign manifest output overlaps an immutable input")
    return resolved


def build_full_event_campaign_manifest(
        selection_path: Path, campaign_root: Path, source_registry_path: Path,
        output_root: Path) -> Mapping[str, Any]:
    """Build one non-overwriting manifest bundle for audit consumption."""
    selection_path = selection_path.resolve()
    campaign_root = campaign_root.resolve()
    source_registry_path = source_registry_path.resolve()
    if not campaign_root.is_dir():
        raise FileNotFoundError(f"formal campaign root is absent: {campaign_root}")
    selection, _, legacy_evidence, protected = (
        audit._validate_frozen_selection(selection_path))
    protected.update({campaign_root, source_registry_path})
    source_payloads, sources = _source_registry(source_registry_path)
    runtime: dict[str, dict[str, Mapping[str, Any]]] = {}
    new_attempts = _new_attempts(
        campaign_root, selection["family_case_ids"], sources, runtime)
    legacy_groups = _legacy_groups(
        legacy_evidence, selection["existing_strict_family_case_ids"],
        sources, runtime)
    legacy_ids = {
        f"legacy_{group['family']}_{index:03d}"
        for index, group in enumerate(legacy_groups, start=1)}
    new_ids = {attempt["attempt_id"] for _, attempt, _ in new_attempts}
    if legacy_ids & new_ids:
        raise ValueError("synthetic legacy attempt ID collides with new attempt")
    if set(runtime) != set(sources):
        raise ValueError("source registry contains unused sources")
    audit._validate_runtime_contract(
        {"runtime_contract": runtime}, sources)
    output_root = _output_root(output_root, {path.resolve() for path in protected})
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        legacy_root = temporary / "legacy_attempts"
        legacy_root.mkdir()
        final_legacy_root = output_root / "legacy_attempts"
        legacy_references = []
        for index, group in enumerate(legacy_groups, start=1):
            inventory_bytes = _inventory_bytes(group["root"])
            stem = f"{index:03d}_{group['family']}"
            inventory_path = legacy_root / f"{stem}.sha256"
            inventory_path.write_bytes(inventory_bytes)
            final_inventory_path = final_legacy_root / inventory_path.name
            inventory_reference = {
                "path": str(final_inventory_path),
                "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            }
            descriptor = _legacy_descriptor(
                group=group, index=index,
                inventory_reference=inventory_reference)
            descriptor_path = legacy_root / f"{stem}.json"
            _write_json(descriptor_path, descriptor)
            legacy_references.append({
                "path": str(final_legacy_root / descriptor_path.name),
                "sha256": file_sha256(descriptor_path),
            })
        new_references = []
        for path, attempt, sha256 in new_attempts:
            if file_sha256(path) != sha256:
                raise ValueError(
                    f"new attempt changed during assembly: "
                    f"{attempt['attempt_id']}")
            new_references.append({"path": str(path), "sha256": sha256})
        manifest = {
            "schema_version": 1,
            "artifact_kind": "metax_c500_full_event_campaign",
            "hardware_id": "metax_c500",
            "campaign_root": str(campaign_root),
            "selection_manifest": {
                "path": str(selection_path), "sha256": file_sha256(selection_path),
            },
            "runtime_contract": runtime,
            "sources": source_payloads,
            "attempts": legacy_references + new_references,
            "selection_rule": audit.SELECTION_RULE,
        }
        _exact_keys(manifest, audit.CAMPAIGN_KEYS, "full Event campaign")
        _write_json(temporary / "campaign_manifest.json", manifest)
        temporary.rename(output_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest
