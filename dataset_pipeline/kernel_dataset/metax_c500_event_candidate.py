"""Build a candidate-only MetaX C500 dataset from accepted OP Event evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

from dataset_pipeline.kernel_dataset.metax_c500_event_protocol import (
    load_json_mapping,
    file_sha256,
    EXPECTED_CANONICAL_SHA256,
    FAMILY_COUNTS,
    GROUPS,
    GROUP_COUNTS,
    GROUP_FAMILIES,
    HISTORICAL_VECTOR_SELECTION_PROTOCOL,
    VECTOR_RECAPTURE_PROTOCOL,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_OWNER = (
    REPOSITORY_ROOT / "profiler_output/kernel_dataset/metax_c500/candidate")
FAMILIES = ("vector", "flash_attn", "gemm", "fused_moe")
GRAPH_REPETITIONS = {
    "vector": [50, 200],
    "flash_attn": [50, 200],
    "gemm": [50, 200],
    "fused_moe": [5, 20],
}
SELECTED_COUNTS = {
    "vector": 149, "flash_attn": 237, "gemm": 13, "fused_moe": 94,
}
CV_MAXIMUM = 0.05
MAXIMUM_RELATIVE_DEVIATION = 0.05
CONVERGENCE_ABSOLUTE_NS = 500.0
CONVERGENCE_RELATIVE = 0.05
AUDIT_KEYS = frozenset({
    "schema_version", "artifact_kind", "status", "hardware_id", "output",
    "campaign_manifest", "campaign_manifest_sha256", "selection_manifest",
    "selection_manifest_sha256", "selection_rule", "process_isolation",
    "existing_strict_case_count", "new_strict_case_count",
    "strict_event_case_count", "coverage", "sources", "attempts",
    "selected_cases", "new_event_artifact_tree_sha256",
})
AUDIT_CASE_KEYS = frozenset({
    "stable", "failures", "median_ns", "population_cv",
    "maximum_relative_deviation", "lower_repetition_median_ns",
    "higher_repetition_median_ns", "cross_repetition_difference_ns",
    "cross_repetition_limit_ns", "family", "case_id", "path", "sha256",
    "canonical_attributes_sha256", "attempt_id", "cohort",
})
EXPECTED_OUTPUT_FILES = frozenset(
    {f"{family}.json" for family in FAMILIES} | {"source_manifest.json"})
EVENT_CANDIDATE_MANIFEST_REVISION = 1
TRACE_BINDING_SCHEMA_VERSION = 1
TRACE_BINDING_ARTIFACT_KIND = (
    "metax_c500_mctracer_trace_binding_manifest")




def _mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), separators=(",", ":"), sort_keys=True,
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _event_artifact_tree_sha256(
        rows: list[Mapping[str, Any]], campaign_root: Path) -> str:
    lines = []
    for row in rows:
        if row["cohort"] != "new":
            continue
        path = Path(row["path"]).resolve()
        try:
            relative = path.relative_to(campaign_root)
        except ValueError as error:
            raise ValueError(
                f"new Event artifact is outside campaign root: {path}") from error
        line = f"{_require_sha256(row['sha256'], 'Event artifact SHA-256')}  "
        line += f"{relative.as_posix()}\n"
        lines.append(line.encode("ascii"))
    if len(lines) != 493:
        raise ValueError("new Event artifact tree must contain 493 cases")
    return hashlib.sha256(b"".join(sorted(lines))).hexdigest()




def _exact_keys(value: Mapping[str, Any], expected: frozenset[str],
                label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys mismatch: {sorted(set(value) ^ expected)}")


def _canonical_inputs(canonical_root: Path
                      ) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    datasets = {}
    provenance = {}
    for family in FAMILIES:
        path = (canonical_root / f"{family}.json").resolve()
        sha256 = file_sha256(path)
        if sha256 != EXPECTED_CANONICAL_SHA256[family]:
            raise ValueError(f"canonical {family} SHA-256 mismatch")
        dataset = load_json_mapping(path, f"canonical {family} dataset")
        if len(dataset) != FAMILY_COUNTS[family]:
            raise ValueError(f"canonical {family} case count mismatch")
        datasets[family] = dataset
        provenance[family] = {
            "path": str(path), "sha256": sha256,
            "case_count": len(dataset),
        }
    return datasets, provenance


def _selection_membership(
        selection_path: Path, canonical_datasets: Mapping[str, Mapping[str, Any]],
        canonical_provenance: Mapping[str, Any],
        ) -> tuple[Mapping[str, Any], dict[str, set[str]], dict[str, set[str]]]:
    selection = load_json_mapping(selection_path, "selection manifest")
    required = frozenset({
        "schema_version", "artifact_kind", "hardware_id",
        "selected_case_count", "existing_strict_case_count",
        "target_case_count", "stability_policy", "vector_protocols",
        "vector_stability_audit", "groups", "family_case_ids",
        "existing_strict_family_case_ids", "provenance",
    })
    _exact_keys(selection, required, "selection manifest")
    if (selection["schema_version"] != 2
            or selection["artifact_kind"] !=
            "metax_c500_full_op_event_selection"
            or selection["hardware_id"] != "metax_c500"
            or selection["selected_case_count"] != 493
            or selection["existing_strict_case_count"] != 494
            or selection["target_case_count"] != 987):
        raise ValueError("selection manifest identity/counts are invalid")
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
        raise ValueError("selection stability policy is invalid")
    expected_protocols = {
        "historical_selection_protocol": HISTORICAL_VECTOR_SELECTION_PROTOCOL,
        "recapture_protocol": VECTOR_RECAPTURE_PROTOCOL,
    }
    if selection["vector_protocols"] != expected_protocols:
        raise ValueError("selection Vector protocols are invalid")
    selected_map = selection["family_case_ids"]
    existing_map = selection["existing_strict_family_case_ids"]
    if (not isinstance(selected_map, dict) or not isinstance(existing_map, dict)
            or set(selected_map) != set(FAMILIES)
            or set(existing_map) != set(FAMILIES)):
        raise ValueError("selection family membership maps are invalid")
    selected_sets = {}
    existing_sets = {}
    for family in FAMILIES:
        selected = selected_map[family]
        existing = existing_map[family]
        if (not isinstance(selected, list) or not isinstance(existing, list)
                or len(selected) != len(set(selected))
                or len(existing) != len(set(existing))
                or len(selected) != SELECTED_COUNTS[family]):
            raise ValueError(f"selection membership is invalid: {family}")
        selected_set = set(selected)
        existing_set = set(existing)
        if (selected_set & existing_set
                or selected_set | existing_set != set(
                    canonical_datasets[family])):
            raise ValueError(f"selection complement is invalid: {family}")
        selected_sets[family] = selected_set
        existing_sets[family] = existing_set
    if sum(len(values) for values in existing_sets.values()) != 494:
        raise ValueError("selection existing strict count is invalid")
    groups = selection["groups"]
    if not isinstance(groups, dict) or set(groups) != set(GROUPS):
        raise ValueError("selection groups are invalid")
    flattened = {family: [] for family in FAMILIES}
    for group in GROUPS:
        entry = groups[group]
        if group == "vector_warning":
            expected_keys = {
                "family", "case_count", "case_ids",
                "historical_selection_protocol_ref", "recapture_protocol_ref",
            }
        elif group in {
                "flash_attn", "gemm", "fused_moe_unstable",
                "fused_moe_missing"}:
            expected_keys = {
                "family", "case_count", "case_ids", "runner_parameters",
            }
        else:
            raise ValueError(f"unknown selection group: {group}")
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            raise ValueError(f"selection group schema is invalid: {group}")
        family = GROUP_FAMILIES[group]
        if (entry["family"] != family
                or entry["case_count"] != GROUP_COUNTS[group]
                or len(entry["case_ids"]) != GROUP_COUNTS[group]
                or len(entry["case_ids"]) != len(set(entry["case_ids"]))):
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
            expected_repetitions = GRAPH_REPETITIONS[family]
            if entry["runner_parameters"] != {
                    "graph_repetitions": expected_repetitions}:
                raise ValueError(
                    f"selection runner parameters are invalid: {group}")
        else:
            raise ValueError(f"unknown selection group: {group}")
        flattened[family].extend(entry["case_ids"])
    for family in FAMILIES:
        if flattened[family] != selected_map[family]:
            raise ValueError(
                f"selection group/family order mismatch: {family}")
    canonical_records = selection["provenance"]["canonical_datasets"]
    if set(canonical_records) != set(FAMILIES):
        raise ValueError("selection canonical provenance families are invalid")
    for family in FAMILIES:
        if canonical_records[family] != canonical_provenance[family]:
            raise ValueError(
                f"selection canonical provenance mismatch: {family}")
    return selection, selected_sets, existing_sets


def _positive_event_samples(measurement: Mapping[str, Any], case_id: str
                            ) -> list[float]:
    if "samples" not in measurement or not isinstance(
            measurement["samples"], list) or not measurement["samples"]:
        raise ValueError(f"Event measurement samples are invalid: {case_id}")
    samples = []
    for index, row in enumerate(measurement["samples"]):
        if not isinstance(row, dict) or "event_duration_ns" not in row:
            raise ValueError(f"Event sample is invalid: {case_id}/{index}")
        duration = row["event_duration_ns"]
        if (isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or duration <= 0):
            raise ValueError(f"Event duration is invalid: {case_id}/{index}")
        samples.append(float(duration))
    return samples


def _event_result(record: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    measurement = record["measurement"]
    if not isinstance(measurement, dict):
        raise TypeError(f"Event measurement must be an object: {case_id}")
    mode = measurement["mode"]
    if mode not in {
            "external_events_around_cuda_graph_replay",
            "paired_differential_events_around_cuda_graph_replay"}:
        raise ValueError(f"non-Event timing evidence is forbidden: {case_id}")
    samples = _positive_event_samples(measurement, case_id)
    median_ns = statistics.median(samples)
    population_cv = statistics.pstdev(samples) / median_ns
    maximum_relative_deviation = max(
        abs(duration - median_ns) / median_ns for duration in samples)
    runs = measurement["repetition_runs"]
    if not isinstance(runs, list) or len(runs) != 2:
        raise ValueError(f"Event repetition runs are invalid: {case_id}")
    repetition_medians = [
        statistics.median(_positive_event_samples(run, case_id))
        for run in runs]
    difference_ns = abs(repetition_medians[0] - repetition_medians[1])
    limit_ns = max(
        CONVERGENCE_ABSOLUTE_NS,
        CONVERGENCE_RELATIVE * max(repetition_medians))
    stable = (
        population_cv <= CV_MAXIMUM
        and maximum_relative_deviation <= MAXIMUM_RELATIVE_DEVIATION
        and difference_ns <= limit_ns)
    return {
        "stable": stable,
        "median_ns": median_ns,
        "population_cv": population_cv,
        "maximum_relative_deviation": maximum_relative_deviation,
        "lower_repetition_median_ns": repetition_medians[0],
        "higher_repetition_median_ns": repetition_medians[1],
        "cross_repetition_difference_ns": difference_ns,
        "cross_repetition_limit_ns": limit_ns,
    }


def _event_artifact(
        audit_row: Mapping[str, Any], canonical_entry: Mapping[str, Any],
        family: str, case_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    _exact_keys(audit_row, AUDIT_CASE_KEYS, f"audit selected case {case_id}")
    if (audit_row["family"] != family or audit_row["case_id"] != case_id
            or audit_row["stable"] is not True
            or audit_row["failures"] != []):
        raise ValueError(f"audit selected case is not strict: {family}/{case_id}")
    path = Path(audit_row["path"]).resolve()
    if file_sha256(path) != audit_row["sha256"]:
        raise ValueError(f"Event artifact SHA-256 mismatch: {family}/{case_id}")
    record = load_json_mapping(path, f"Event artifact {family}/{case_id}")
    required = {
        "status", "hardware_id", "family", "case_id", "baseline_commit",
        "source_snapshot_sha256", "runner_path", "runner_sha256",
        "timer_sanity_sha256", "canonical_attributes_sha256", "timer",
        "measurement",
    }
    if not required.issubset(record):
        raise ValueError(f"Event artifact keys are incomplete: {family}/{case_id}")
    if (record["status"] != "measured"
            or record["hardware_id"] != "metax_c500"
            or record["family"] != family or record["case_id"] != case_id):
        raise ValueError(f"Event artifact identity is invalid: {family}/{case_id}")
    attributes_sha256 = _mapping_sha256(canonical_entry["op_info"]["Attr"])
    if (record["canonical_attributes_sha256"] != attributes_sha256
            or audit_row["canonical_attributes_sha256"] != attributes_sha256):
        raise ValueError(
            f"Event artifact canonical attributes mismatch: {family}/{case_id}")
    timer = record["timer"]
    if (not isinstance(timer, dict) or timer["api"] != "torch.cuda.Event"
            or timer["implementation"] != "wcudaEvent_to_maca_mcEvent"):
        raise ValueError(f"Event timer identity is invalid: {family}/{case_id}")
    result = _event_result(record, case_id)
    for key in (
            "stable", "median_ns", "population_cv",
            "maximum_relative_deviation", "lower_repetition_median_ns",
            "higher_repetition_median_ns", "cross_repetition_difference_ns",
            "cross_repetition_limit_ns"):
        expected = result[key]
        actual = audit_row[key]
        if isinstance(expected, bool):
            equal = actual is expected
        else:
            equal = (isinstance(actual, (int, float))
                     and not isinstance(actual, bool)
                     and math.isclose(
                         float(actual), float(expected),
                         rel_tol=1e-12, abs_tol=1e-12))
        if not equal:
            raise ValueError(
                f"audit/Event statistic mismatch: {family}/{case_id}/{key}")
    return record, result


def _validate_existing_timing(
        timing: Mapping[str, Any], record: Mapping[str, Any],
        result: Mapping[str, Any], artifact_sha256: str,
        family: str, case_id: str) -> None:
    if (timing["granularity"] != "op"
            or timing["source"] != "device_event_op_median"
            or not math.isclose(
                float(timing["total_duration_ns"]),
                float(result["median_ns"]), rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(
            f"existing strict Event timing mismatch: {family}/{case_id}")
    provenance = timing["provenance"]
    expected = {
        "baseline_commit": record["baseline_commit"],
        "measurement_mode": record["measurement"]["mode"],
        "profiler": "none",
        "runner_sha256": record["runner_sha256"],
        "scope": "end_to_end_case",
        "source_artifact_sha256": artifact_sha256,
        "source_snapshot_sha256": record["source_snapshot_sha256"],
        "statistic": "median_ns",
        "timer_api": record["timer"]["api"],
        "timer_implementation": record["timer"]["implementation"],
        "timer_sanity_sha256": record["timer_sanity_sha256"],
    }
    for key, value in expected.items():
        if provenance[key] != value:
            raise ValueError(
                f"existing strict provenance mismatch: "
                f"{family}/{case_id}/{key}")


def _replacement_timing(
        record: Mapping[str, Any], result: Mapping[str, Any],
        artifact_sha256: str, artifact_tree_sha256: str) -> Mapping[str, Any]:
    return {
        "granularity": "op",
        "source": "device_event_op_median",
        "total_duration_ns": result["median_ns"],
        "provenance": {
            "baseline_commit": record["baseline_commit"],
            "measurement_mode": record["measurement"]["mode"],
            "profiler": "none",
            "runner_sha256": record["runner_sha256"],
            "scope": "end_to_end_case",
            "source_artifact_sha256": artifact_sha256,
            "source_artifact_tree_sha256": artifact_tree_sha256,
            "source_snapshot_sha256": record["source_snapshot_sha256"],
            "statistic": "median_ns",
            "timer_api": record["timer"]["api"],
            "timer_implementation": record["timer"]["implementation"],
            "timer_sanity_sha256": record["timer_sanity_sha256"],
        },
    }


def _validate_audit(
        audit_path: Path, selection_path: Path, selection: Mapping[str, Any],
        selected_sets: Mapping[str, set[str]],
        existing_sets: Mapping[str, set[str]],
        canonical_datasets: Mapping[str, Mapping[str, Any]],
        ) -> tuple[
            dict[str, Mapping[str, Any]], list[Mapping[str, Any]], str]:
    audit = load_json_mapping(audit_path, "full Event audit report")
    _exact_keys(audit, AUDIT_KEYS, "full Event audit report")
    if (audit["schema_version"] != 2
            or audit["artifact_kind"] !=
            "metax_c500_full_op_event_capture_audit"
            or audit["status"] != "accepted"
            or audit["hardware_id"] != "metax_c500"
            or audit["existing_strict_case_count"] != 494
            or audit["new_strict_case_count"] != 493
            or audit["strict_event_case_count"] != 987):
        raise ValueError("full Event audit identity/status/counts are invalid")
    if (Path(audit["output"]).resolve() != audit_path
            or Path(audit["selection_manifest"]).resolve() != selection_path
            or audit["selection_manifest_sha256"] != file_sha256(selection_path)):
        raise ValueError("full Event audit input binding is invalid")
    campaign_path = Path(audit["campaign_manifest"]).resolve()
    if file_sha256(campaign_path) != audit["campaign_manifest_sha256"]:
        raise ValueError("full Event audit campaign binding is invalid")
    campaign = load_json_mapping(campaign_path, "full Event campaign manifest")
    campaign_root = Path(campaign["campaign_root"]).resolve()
    rows = audit["selected_cases"]
    if not isinstance(rows, list) or len(rows) != 987:
        raise ValueError("full Event audit must contain 987 selected cases")
    by_identity = {}
    sources = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("audit selected case must be an object")
        family = row["family"]
        case_id = row["case_id"]
        if family not in FAMILIES or case_id not in canonical_datasets[family]:
            raise ValueError(f"audit selected case is non-canonical: {row}")
        identity = (family, case_id)
        if identity in by_identity:
            raise ValueError(f"audit repeats selected case: {identity}")
        expected_cohort = (
            "new" if case_id in selected_sets[family] else "existing_strict")
        if (row["cohort"] != expected_cohort
                or (expected_cohort == "existing_strict"
                    and case_id not in existing_sets[family])):
            raise ValueError(f"audit selected cohort mismatch: {identity}")
        record, result = _event_artifact(
            row, canonical_datasets[family][case_id], family, case_id)
        by_identity[identity] = {
            "audit_row": row, "artifact": record, "result": result,
        }
        sources.append({
            "family": family,
            "case_id": case_id,
            "cohort": expected_cohort,
            "attempt_id": row["attempt_id"],
            "event_artifact": {
                "path": str(Path(row["path"]).resolve()),
                "sha256": row["sha256"],
            },
            "median_ns": result["median_ns"],
        })
    expected_identities = {
        (family, case_id) for family in FAMILIES
        for case_id in canonical_datasets[family]}
    if set(by_identity) != expected_identities:
        raise ValueError("full Event audit selected membership is incomplete")
    artifact_tree_sha256 = _event_artifact_tree_sha256(rows, campaign_root)
    if artifact_tree_sha256 != _require_sha256(
            audit["new_event_artifact_tree_sha256"],
            "audit new Event artifact tree SHA-256"):
        raise ValueError("new Event artifact tree SHA-256 mismatch")
    return by_identity, sources, artifact_tree_sha256


def _candidate_output_root(output_root: Path, protected: set[Path]) -> Path:
    absolute = Path(os.path.abspath(output_root))
    resolved = absolute.resolve()
    owner = CANDIDATE_OWNER.resolve()
    if absolute != resolved or output_root.is_symlink():
        raise ValueError("candidate output root must not contain symlinks")
    try:
        relative = resolved.relative_to(owner)
    except ValueError as error:
        raise ValueError(
            f"candidate output must stay under {owner}") from error
    if not relative.parts:
        raise ValueError("candidate output must be a child of candidate owner")
    if any(resolved == path or resolved in path.parents for path in protected):
        raise ValueError("candidate output overlaps a protected input")
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError("candidate output root must be a directory")
        unexpected = sorted(
            path.name for path in resolved.iterdir()
            if path.name not in EXPECTED_OUTPUT_FILES or not path.is_file())
        if unexpected:
            raise ValueError(
                f"candidate output contains unexpected entries: {unexpected}")
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def build_full_event_candidate(
        selection_path: Path, audit_path: Path, canonical_root: Path,
        output_root: Path) -> Mapping[str, Any]:
    """Build four Event-only candidate datasets and their source manifest."""

    selection_path = selection_path.resolve()
    audit_path = audit_path.resolve()
    canonical_root = canonical_root.resolve()
    canonical_datasets, canonical_provenance = _canonical_inputs(canonical_root)
    selection, selected_sets, existing_sets = _selection_membership(
        selection_path, canonical_datasets, canonical_provenance)
    evidence, case_sources, new_event_artifact_tree_sha256 = _validate_audit(
        audit_path, selection_path, selection, selected_sets, existing_sets,
        canonical_datasets)
    protected = {selection_path, audit_path, canonical_root}
    protected.update(
        Path(record["path"]).resolve()
        for record in canonical_provenance.values())
    protected.update(
        Path(source["event_artifact"]["path"]).resolve()
        for source in case_sources)
    output_root = _candidate_output_root(output_root, protected)

    candidates = {}
    replacements = []
    for family in FAMILIES:
        candidate = copy.deepcopy(canonical_datasets[family])
        for case_id in candidate:
            identity = (family, case_id)
            source = evidence[identity]
            artifact = source["artifact"]
            result = source["result"]
            canonical_entry = canonical_datasets[family][case_id]
            candidate_entry = candidate[case_id]
            if case_id in selected_sets[family]:
                before = copy.deepcopy(candidate_entry)
                candidate_entry["op_info"]["evaluation"]["timing"] = (
                    _replacement_timing(
                        artifact, result, source["audit_row"]["sha256"],
                        new_event_artifact_tree_sha256))
                before["op_info"]["evaluation"] = copy.deepcopy(
                    candidate_entry["op_info"]["evaluation"])
                if before != candidate_entry:
                    raise AssertionError(
                        f"candidate changed non-evaluation semantics: {identity}")
                replacements.append(identity)
            else:
                timing = candidate_entry["op_info"]["evaluation"]["timing"]
                _validate_existing_timing(
                    timing, artifact, result,
                    source["audit_row"]["sha256"], family, case_id)
                if candidate_entry != canonical_entry:
                    raise AssertionError(
                        f"existing strict case changed: {identity}")
        candidates[family] = candidate
    if len(replacements) != 493:
        raise AssertionError("candidate replacement count is not 493")
    event_only_count = sum(
        entry["op_info"]["evaluation"]["timing"]["source"] ==
        "device_event_op_median"
        for dataset in candidates.values() for entry in dataset.values())
    if event_only_count != 987:
        raise AssertionError("candidate is not 987-case Event-only timing")

    output_root.mkdir(parents=True, exist_ok=True)
    candidate_records = {}
    for family in FAMILIES:
        path = output_root / f"{family}.json"
        _write_json(path, candidates[family])
        candidate_records[family] = {
            "path": str(path), "sha256": file_sha256(path),
            "case_count": len(candidates[family]),
        }
    missing_trace_ids = selection["groups"]["fused_moe_missing"]["case_ids"]
    source_manifest = {
        "schema_version": 1,
        "manifest_revision": EVENT_CANDIDATE_MANIFEST_REVISION,
        "artifact_kind": "metax_c500_event_only_candidate_manifest",
        "status": "event_timing_complete_trace_binding_incomplete",
        "publication_ready": False,
        "hardware_id": "metax_c500",
        "event_timing": {
            "granularity": "op",
            "source": "device_event_op_median",
            "profiler": "none",
            "strict_case_count": 987,
            "new_case_count": 493,
            "reused_case_count": 494,
        },
        "trace_binding": {
            "status": "incomplete",
            "required_group": "fused_moe_missing",
            "required_case_count": 84,
            "bound_case_count": 0,
            "case_ids": list(missing_trace_ids),
            "duration_use": "forbidden",
            "reason": "mcTracer Graph launch binding was not provided",
            "binding_contract": {
                "storage": "external_manifest",
                "artifact_kind": TRACE_BINDING_ARTIFACT_KIND,
                "schema_version": TRACE_BINDING_SCHEMA_VERSION,
                "candidate_manifest_revision": (
                    EVENT_CANDIDATE_MANIFEST_REVISION),
                "required_event_candidate_dataset_sha256": {
                    family: candidate_records[family]["sha256"]
                    for family in FAMILIES
                },
                "required_full_event_audit_sha256": file_sha256(audit_path),
                "required_new_event_artifact_tree_sha256": (
                    new_event_artifact_tree_sha256),
            },
        },
        "inputs": {
            "selection_manifest": {
                "path": str(selection_path), "sha256": file_sha256(selection_path),
            },
            "full_event_audit": {
                "path": str(audit_path), "sha256": file_sha256(audit_path),
            },
            "canonical_datasets": canonical_provenance,
        },
        "candidate_datasets": candidate_records,
        "case_sources": case_sources,
    }
    _write_json(output_root / "source_manifest.json", source_manifest)
    return source_manifest
