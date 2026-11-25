"""Build the audited 493-case MetaX C500 Device Event selection."""

import hashlib
import json
import math
import statistics

from dataset_pipeline.kernel_dataset.metax_c500_event_protocol import (
    file_sha256,
    EXPECTED_CANONICAL_SHA256,
    HISTORICAL_VECTOR_SELECTION_PROTOCOL,
    VECTOR_RECAPTURE_PROTOCOL,
    summarize_event_samples,
)


EXPECTED_COMPLETION_MANIFEST_SHA256 = (
    "c89df8137a2dec4917eab329f00e3c21073f35fa7c3916a0a9268d776699077e"
)
EXPECTED_VECTOR_RAW_PROVENANCE = {
    "root_name": "20260827_vector_cold_diff_8x_full_d494ee96",
    "baseline_commit": "d494ee96aee889235f41439f919a6033aa39a9aa",
    "source_snapshot_sha256": (
        "66aa945e7732c147ec9da111bbbae7511851e5f0e871054b8b02f23997374bec"),
    "runner_path": "scripts/kernel_dataset/metax_c500/vector_runner.py",
    "runner_sha256": (
        "d651a067f5e7d15f127484dddd896b2525de884c11dbc70a1062b7b56ac548f3"),
    "timer_sanity_sha256": (
        "d37191d49d55aa8a79078eb22917da6f5da45f2e2b7f0addfbf62f8fc727b48b"),
    "historical_canonical_dataset_sha256": (
        "0dcbe26e0aee02058b0188ce1ec74b8a311cbce4c47f0533279be5d4ddcb42ae"),
    "shard_table_sha256": [
        "2d5947aff156694de089d40acfbbb123dd414f27989e387c3822e70a28dd2431",
        "6e2c89c698629ccfc48b135cf0e8e06f1ebd8bf084939315bd9fa1900f0a0825",
        "4ebb4ecfe267b6ef2507e10d8781fb489783be92bc1706505dea7055cab0a9c5",
        "0cbe7c828206c8ceaf7a8b007ac75e94ebb728dee8d03fc80f5eecfa6b9474ce",
    ],
    "case_artifact_manifest_sha256": (
        "28ed78446b92bb64807a3fdf51379070dc30378b4eda75035863e60872be02f8"),
}
EXPECTED_CANONICAL_COUNTS = {
    "vector": 296,
    "flash_attn": 294,
    "gemm": 221,
    "fused_moe": 176,
}
EXPECTED_GROUP_COUNTS = {
    "vector_warning": 149,
    "flash_attn": 237,
    "gemm": 13,
    "fused_moe_unstable": 10,
    "fused_moe_missing": 84,
}
COMPLETION_GROUP_FAMILIES = {
    "flash_attn": "flash_attn",
    "gemm": "gemm",
    "fused_moe_unstable": "fused_moe",
    "fused_moe_missing": "fused_moe",
}
CV_MAXIMUM = 0.05
MAXIMUM_RELATIVE_DEVIATION = 0.05
CONVERGENCE_ABSOLUTE_NS = 500.0
CONVERGENCE_RELATIVE = 0.05




def _load_json(path):
    if not path.is_file():
        raise ValueError(f"required JSON file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _require_exact_keys(container, required, label):
    missing = sorted(set(required).difference(container))
    if missing:
        raise ValueError(f"{label} is missing required keys: {missing}")


def _require_unique_case_ids(case_ids, label):
    if not isinstance(case_ids, list) or not all(
            isinstance(case_id, str) and case_id for case_id in case_ids):
        raise ValueError(f"{label} case_ids must be a list of non-empty strings")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{label} case_ids contain duplicates")


def _validate_expected_sha(path, expected, label):
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _canonical_inputs(canonical_root):
    datasets = {}
    provenance = {}
    for family in ("vector", "flash_attn", "gemm", "fused_moe"):
        path = canonical_root / f"{family}.json"
        sha256 = _validate_expected_sha(
            path, EXPECTED_CANONICAL_SHA256[family],
            f"canonical {family} dataset")
        payload = _load_json(path)
        if len(payload) != EXPECTED_CANONICAL_COUNTS[family]:
            raise ValueError(
                f"canonical {family} dataset must contain "
                f"{EXPECTED_CANONICAL_COUNTS[family]} cases, got "
                f"{len(payload)}")
        datasets[family] = payload
        provenance[family] = {
            "path": str(path.resolve()),
            "sha256": sha256,
            "case_count": len(payload),
        }
    return datasets, provenance


def _completion_groups(completion_manifest, canonical_datasets):
    _require_exact_keys(
        completion_manifest,
        ("schema_version", "artifact_kind", "hardware_id",
         "stability_policy", "families"),
        "completion manifest")
    if completion_manifest["schema_version"] != 1:
        raise ValueError("completion manifest schema_version must be 1")
    if completion_manifest["artifact_kind"] != (
            "metax_c500_op_golden_completion_manifest"):
        raise ValueError("completion manifest artifact_kind is invalid")
    if completion_manifest["hardware_id"] != "metax_c500":
        raise ValueError("completion manifest hardware_id must be metax_c500")
    stability_policy = completion_manifest["stability_policy"]
    _require_exact_keys(
        stability_policy,
        ("convergence", "coefficient_of_variation_maximum",
         "maximum_relative_deviation_maximum"),
        "completion manifest stability_policy")
    if (stability_policy["convergence"] !=
            "absolute difference <= max(500 ns, 5% of larger median)"
            or stability_policy["coefficient_of_variation_maximum"] != 0.05
            or stability_policy[
                "maximum_relative_deviation_maximum"] != 0.05):
        raise ValueError("completion manifest stability_policy is not frozen")
    families = completion_manifest["families"]
    if not isinstance(families, dict):
        raise ValueError("completion manifest families must be an object")
    if set(families) != set(COMPLETION_GROUP_FAMILIES):
        raise ValueError(
            "completion manifest must contain exactly flash_attn, gemm, "
            "fused_moe_unstable, and fused_moe_missing")

    result = {}
    for group, family in COMPLETION_GROUP_FAMILIES.items():
        entry = families[group]
        _require_exact_keys(
            entry,
            ("canonical_dataset_sha256", "mode", "graph_repetitions",
             "case_ids"),
            f"completion group {group}")
        if entry["canonical_dataset_sha256"] != (
                EXPECTED_CANONICAL_SHA256[family]):
            raise ValueError(
                f"completion group {group} canonical SHA-256 is invalid")
        expected_repetitions_by_family = {
            "flash_attn": [50, 200],
            "gemm": [50, 200],
            "fused_moe": [5, 20],
        }
        expected_repetitions = expected_repetitions_by_family[family]
        if entry["graph_repetitions"] != expected_repetitions:
            raise ValueError(
                f"completion group {group} graph_repetitions must be "
                f"{expected_repetitions}")
        case_ids = entry["case_ids"]
        _require_unique_case_ids(case_ids, f"completion group {group}")
        if len(case_ids) != EXPECTED_GROUP_COUNTS[group]:
            raise ValueError(
                f"completion group {group} must contain "
                f"{EXPECTED_GROUP_COUNTS[group]} cases, got {len(case_ids)}")
        unknown = sorted(set(case_ids).difference(canonical_datasets[family]))
        if unknown:
            raise ValueError(
                f"completion group {group} contains non-canonical cases: "
                f"{unknown[:5]}")
        result[group] = list(case_ids)
    return result


def _sha256sum_manifest_sha256(paths, root):
    lines = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        lines.append(
            f"{file_sha256(path)}  {relative}\n".encode("ascii"))
    digest = hashlib.sha256()
    for line in sorted(lines):
        digest.update(line)
    return digest.hexdigest()


def _positive_finite_samples(samples, label):
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{label} must be a non-empty list")
    values = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict) or "event_duration_ns" not in sample:
            raise ValueError(
                f"{label}[{sample_index}] lacks event_duration_ns")
        value = sample["event_duration_ns"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(
                f"{label}[{sample_index}].event_duration_ns must be "
                "positive and finite")
        values.append(float(value))
    return values


def _vector_stability(case_path):
    artifact = _load_json(case_path)
    _require_exact_keys(
        artifact, ("schema_version", "status", "hardware_id", "family",
                   "case_id", "baseline_commit", "source_snapshot_sha256",
                   "runner_path", "runner_sha256", "timer_sanity_sha256",
                   "measurement"),
        f"Vector artifact {case_path}")
    if artifact["status"] != "measured":
        raise ValueError(f"Vector artifact is not measured: {case_path}")
    if artifact["hardware_id"] != "metax_c500":
        raise ValueError(f"Vector artifact hardware_id is invalid: {case_path}")
    if artifact["family"] != "vector":
        raise ValueError(f"Vector artifact family is invalid: {case_path}")
    if artifact["case_id"] != case_path.stem:
        raise ValueError(f"Vector artifact case_id/path mismatch: {case_path}")
    for key in ("baseline_commit", "source_snapshot_sha256", "runner_path",
                "runner_sha256", "timer_sanity_sha256"):
        if artifact[key] != EXPECTED_VECTOR_RAW_PROVENANCE[key]:
            raise ValueError(
                f"Vector artifact {key} does not match frozen provenance: "
                f"{case_path}")

    measurement = artifact["measurement"]
    _require_exact_keys(
        measurement,
        ("mode", "sample_count", "repetition_counts", "repetition_runs",
         "samples"),
        f"Vector measurement {case_path}")
    if measurement["mode"] != (
            "paired_differential_events_around_cuda_graph_replay"):
        raise ValueError(f"Vector measurement mode is invalid: {case_path}")
    samples = _positive_finite_samples(
        measurement["samples"], f"Vector samples {case_path}")
    if measurement["sample_count"] != len(samples):
        raise ValueError(f"Vector sample_count mismatch: {case_path}")
    if len(samples) != 50:
        raise ValueError(f"Vector paired sample count must be 50: {case_path}")

    sample_summary = summarize_event_samples(samples)
    median_ns = sample_summary["median_ns"]
    population_cv = sample_summary["coefficient_of_variation"]
    maximum_relative_deviation = max(
        abs(value - median_ns) / median_ns for value in samples)

    repetition_counts = measurement["repetition_counts"]
    repetition_runs = measurement["repetition_runs"]
    if (not isinstance(repetition_counts, list)
            or len(repetition_counts) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool)
                       and value > 0 for value in repetition_counts)
            or repetition_counts[0] >= repetition_counts[1]):
        raise ValueError(
            f"Vector repetition_counts must be two increasing integers: "
            f"{case_path}")
    if not isinstance(repetition_runs, list) or len(repetition_runs) != 2:
        raise ValueError(f"Vector measurement must have two runs: {case_path}")
    repetition_medians = []
    for index, run in enumerate(repetition_runs):
        _require_exact_keys(
            run, ("repetition_count", "sample_count", "samples"),
            f"Vector repetition run {index} in {case_path}")
        if run["repetition_count"] != repetition_counts[index]:
            raise ValueError(
                f"Vector repetition run/count mismatch: {case_path}")
        values = _positive_finite_samples(
            run["samples"],
            f"Vector repetition run {index} samples in {case_path}")
        if run["sample_count"] != len(values):
            raise ValueError(
                f"Vector repetition run sample_count mismatch: {case_path}")
        repetition_medians.append(statistics.median(values))
    convergence_difference_ns = abs(
        repetition_medians[0] - repetition_medians[1])
    convergence_limit_ns = max(
        CONVERGENCE_ABSOLUTE_NS,
        CONVERGENCE_RELATIVE * max(repetition_medians))
    failed = []
    if population_cv > CV_MAXIMUM:
        failed.append("population_cv")
    if maximum_relative_deviation > MAXIMUM_RELATIVE_DEVIATION:
        failed.append("maximum_relative_deviation")
    if convergence_difference_ns > convergence_limit_ns:
        failed.append("cross_repetition_convergence")
    return artifact["case_id"], failed


def _vector_groups(vector_raw_root, canonical_vector):
    if vector_raw_root.name != EXPECTED_VECTOR_RAW_PROVENANCE["root_name"]:
        raise ValueError(
            "Vector raw root name does not match frozen provenance: "
            f"{vector_raw_root.name}")
    shard_dirs = sorted(
        path for path in vector_raw_root.iterdir()
        if path.is_dir() and path.name.startswith("shard_"))
    expected_shards = [f"shard_{index}" for index in range(4)]
    if [path.name for path in shard_dirs] != expected_shards:
        raise ValueError(
            f"Vector raw root must contain exactly {expected_shards}")

    all_case_paths = []
    shard_provenance = []
    warnings = []
    strict = []
    failure_counts = {
        "population_cv": 0,
        "maximum_relative_deviation": 0,
        "cross_repetition_convergence": 0,
    }
    for shard_index, shard_dir in enumerate(shard_dirs):
        case_paths = sorted((shard_dir / "cases").glob("VECTOR_*.json"))
        vector_table_path = shard_dir / "vector_table.json"
        table_sha256 = _validate_expected_sha(
            vector_table_path,
            EXPECTED_VECTOR_RAW_PROVENANCE["shard_table_sha256"][
                shard_index],
            f"Vector shard {shard_index} table")
        metadata = _load_json(vector_table_path)
        _require_exact_keys(
            metadata,
            ("hardware_id", "family", "method", "baseline_commit",
             "source_snapshot_sha256", "timer_sanity_sha256",
             "canonical_dataset_sha256", "expected_case_count",
             "unsharded_case_count", "shard_count", "shard_index",
             "sample_count_per_repetition", "warmup_iterations",
             "differential_estimator", "buffer_policy", "buffer_count",
             "cache_eviction_mib", "minimum_distinct_input_mib", "rows"),
            f"Vector shard metadata {vector_table_path}")
        if (metadata["hardware_id"] != "metax_c500"
                or metadata["family"] != "vector"
                or metadata["method"] != (
                    "paired_differential_events_around_cuda_graph_replay")
                or metadata["baseline_commit"] !=
                EXPECTED_VECTOR_RAW_PROVENANCE["baseline_commit"]
                or metadata["source_snapshot_sha256"] !=
                EXPECTED_VECTOR_RAW_PROVENANCE["source_snapshot_sha256"]
                or metadata["timer_sanity_sha256"] !=
                EXPECTED_VECTOR_RAW_PROVENANCE["timer_sanity_sha256"]
                or metadata["canonical_dataset_sha256"] !=
                EXPECTED_VECTOR_RAW_PROVENANCE[
                    "historical_canonical_dataset_sha256"]
                or metadata["shard_count"] != 4
                or metadata["shard_index"] != shard_index
                or metadata["unsharded_case_count"] != 296
                or metadata["expected_case_count"] != len(case_paths)
                or metadata["sample_count_per_repetition"] != 50
                or metadata["warmup_iterations"] != 5
                or metadata["differential_estimator"] !=
                "(T_high-T_low)/(N_high-N_low)"
                or metadata["buffer_policy"] !=
                "unique_buffer_per_selected_replay_call"
                or metadata["buffer_count"] is not None
                or metadata["cache_eviction_mib"] != 256
                or metadata["minimum_distinct_input_mib"] != 64):
            raise ValueError(
                f"Vector shard metadata is inconsistent: {vector_table_path}")
        rows = metadata["rows"]
        if not isinstance(rows, list):
            raise ValueError(
                f"Vector shard table rows must be a list: {vector_table_path}")
        for row in rows:
            _require_exact_keys(
                row, ("case_id", "status"),
                f"Vector shard table row in {vector_table_path}")
        row_case_ids = [row["case_id"] for row in rows]
        if (len(row_case_ids) != len(set(row_case_ids))
                or set(row_case_ids) != {path.stem for path in case_paths}
                or any(row["status"] != "measured" for row in rows)):
            raise ValueError(
                f"Vector shard table rows are inconsistent: "
                f"{vector_table_path}")
        all_case_paths.extend(case_paths)
        shard_provenance.append({
            "shard_index": shard_index,
            "case_count": len(case_paths),
            "table_path": str(vector_table_path.resolve()),
            "table_sha256": table_sha256,
            "case_artifact_manifest_sha256": _sha256sum_manifest_sha256(
                case_paths, vector_raw_root),
        })
        for case_path in case_paths:
            case_id, failed = _vector_stability(case_path)
            if failed:
                warnings.append(case_id)
                for criterion in failed:
                    failure_counts[criterion] += 1
            else:
                strict.append(case_id)

    if len(all_case_paths) != EXPECTED_CANONICAL_COUNTS["vector"]:
        raise ValueError(
            "Vector raw root must contain exactly 296 case artifacts, got "
            f"{len(all_case_paths)}")
    case_artifact_manifest_sha256 = _sha256sum_manifest_sha256(
        all_case_paths, vector_raw_root)
    if case_artifact_manifest_sha256 != EXPECTED_VECTOR_RAW_PROVENANCE[
            "case_artifact_manifest_sha256"]:
        raise ValueError(
            "Vector case artifact manifest SHA-256 mismatch: expected "
            f"{EXPECTED_VECTOR_RAW_PROVENANCE['case_artifact_manifest_sha256']}, "
            f"got {case_artifact_manifest_sha256}")
    all_case_ids = [path.stem for path in all_case_paths]
    if len(all_case_ids) != len(set(all_case_ids)):
        raise ValueError("Vector raw root contains duplicate case artifacts")
    if set(all_case_ids) != set(canonical_vector):
        missing = sorted(set(canonical_vector).difference(all_case_ids))
        extra = sorted(set(all_case_ids).difference(canonical_vector))
        raise ValueError(
            f"Vector raw/canonical membership mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}")
    if (len(warnings) != EXPECTED_GROUP_COUNTS["vector_warning"]
            or len(strict) != 147):
        raise ValueError(
            "frozen Vector stability policy must reproduce 149 warning and "
            f"147 strict cases, got {len(warnings)} warning and "
            f"{len(strict)} strict")
    warning_set = set(warnings)
    strict_set = set(strict)
    canonical_order = list(canonical_vector)
    ordered_warnings = [case_id for case_id in canonical_order
                        if case_id in warning_set]
    ordered_strict = [case_id for case_id in canonical_order
                      if case_id in strict_set]
    provenance = {
        "path": str(vector_raw_root.resolve()),
        "baseline_commit": EXPECTED_VECTOR_RAW_PROVENANCE["baseline_commit"],
        "source_snapshot_sha256": EXPECTED_VECTOR_RAW_PROVENANCE[
            "source_snapshot_sha256"],
        "runner_path": EXPECTED_VECTOR_RAW_PROVENANCE["runner_path"],
        "runner_sha256": EXPECTED_VECTOR_RAW_PROVENANCE["runner_sha256"],
        "timer_sanity_sha256": EXPECTED_VECTOR_RAW_PROVENANCE[
            "timer_sanity_sha256"],
        "historical_canonical_dataset_sha256": (
            EXPECTED_VECTOR_RAW_PROVENANCE[
                "historical_canonical_dataset_sha256"]),
        "case_artifact_count": len(all_case_paths),
        "case_artifact_manifest_format": (
            "LC_ALL=C sorted '<sha256>  <relative-posix-path>\\n' lines"),
        "case_artifact_manifest_sha256": case_artifact_manifest_sha256,
        "shards": shard_provenance,
    }
    audit = {
        "evaluated_case_count": len(all_case_paths),
        "strict_case_count": len(ordered_strict),
        "warning_case_count": len(ordered_warnings),
        "criterion_failure_counts": failure_counts,
    }
    return ordered_warnings, ordered_strict, provenance, audit


def _validate_selection(groups, canonical_datasets):
    selected_owner = {}
    for group, entry in groups.items():
        case_ids = entry["case_ids"]
        _require_unique_case_ids(case_ids, f"selection group {group}")
        family = entry["family"]
        unknown = sorted(set(case_ids).difference(canonical_datasets[family]))
        if unknown:
            raise ValueError(
                f"selection group {group} contains non-canonical cases: "
                f"{unknown[:5]}")
        for case_id in case_ids:
            if case_id in selected_owner:
                raise ValueError(
                    f"selection groups {selected_owner[case_id]} and {group} "
                    f"overlap at {case_id}")
            selected_owner[case_id] = group
    if len(selected_owner) != 493:
        raise ValueError(
            f"full Event selection must contain 493 cases, got "
            f"{len(selected_owner)}")


def build_full_event_selection(
        completion_manifest_path, vector_raw_root, canonical_root):
    completion_manifest_path = completion_manifest_path.resolve()
    vector_raw_root = vector_raw_root.resolve()
    canonical_root = canonical_root.resolve()
    if not vector_raw_root.is_dir():
        raise ValueError(f"Vector raw root does not exist: {vector_raw_root}")
    completion_sha256 = _validate_expected_sha(
        completion_manifest_path, EXPECTED_COMPLETION_MANIFEST_SHA256,
        "completion manifest")
    completion_manifest = _load_json(completion_manifest_path)
    canonical_datasets, canonical_provenance = _canonical_inputs(
        canonical_root)
    completion_groups = _completion_groups(
        completion_manifest, canonical_datasets)
    vector_warning, vector_strict, vector_provenance, vector_audit = (
        _vector_groups(vector_raw_root, canonical_datasets["vector"]))

    groups = {
        "vector_warning": {
            "family": "vector",
            "case_count": len(vector_warning),
            "case_ids": vector_warning,
            "historical_selection_protocol_ref": (
                "vector_protocols.historical_selection_protocol"),
            "recapture_protocol_ref": (
                "vector_protocols.recapture_protocol"),
        },
        "flash_attn": {
            "family": "flash_attn",
            "case_count": len(completion_groups["flash_attn"]),
            "case_ids": completion_groups["flash_attn"],
            "runner_parameters": {"graph_repetitions": [50, 200]},
        },
        "gemm": {
            "family": "gemm",
            "case_count": len(completion_groups["gemm"]),
            "case_ids": completion_groups["gemm"],
            "runner_parameters": {"graph_repetitions": [50, 200]},
        },
        "fused_moe_unstable": {
            "family": "fused_moe",
            "case_count": len(completion_groups["fused_moe_unstable"]),
            "case_ids": completion_groups["fused_moe_unstable"],
            "runner_parameters": {"graph_repetitions": [5, 20]},
        },
        "fused_moe_missing": {
            "family": "fused_moe",
            "case_count": len(completion_groups["fused_moe_missing"]),
            "case_ids": completion_groups["fused_moe_missing"],
            "runner_parameters": {"graph_repetitions": [5, 20]},
        },
    }
    _validate_selection(groups, canonical_datasets)
    family_case_ids = {
        "vector": list(groups["vector_warning"]["case_ids"]),
        "flash_attn": list(groups["flash_attn"]["case_ids"]),
        "gemm": list(groups["gemm"]["case_ids"]),
        "fused_moe": (
            list(groups["fused_moe_unstable"]["case_ids"])
            + list(groups["fused_moe_missing"]["case_ids"])),
    }
    if {family: len(case_ids) for family, case_ids in family_case_ids.items()} != {
            "vector": 149, "flash_attn": 237, "gemm": 13,
            "fused_moe": 94}:
        raise AssertionError("flattened family counts are inconsistent")
    selected_sets = {
        family: set(case_ids) for family, case_ids in family_case_ids.items()
    }
    existing_strict_family_case_ids = {
        "vector": vector_strict,
        "flash_attn": [
            case_id for case_id in canonical_datasets["flash_attn"]
            if case_id not in selected_sets["flash_attn"]],
        "gemm": [
            case_id for case_id in canonical_datasets["gemm"]
            if case_id not in selected_sets["gemm"]],
        "fused_moe": [
            case_id for case_id in canonical_datasets["fused_moe"]
            if case_id not in selected_sets["fused_moe"]],
    }
    existing_strict_case_count = sum(
        len(case_ids)
        for case_ids in existing_strict_family_case_ids.values())
    if existing_strict_case_count != 494:
        raise AssertionError(
            "canonical complement must contain 494 existing strict cases")

    return {
        "schema_version": 2,
        "artifact_kind": "metax_c500_full_op_event_selection",
        "hardware_id": "metax_c500",
        "selected_case_count": 493,
        "existing_strict_case_count": existing_strict_case_count,
        "target_case_count": 987,
        "stability_policy": {
            "sample_population_cv_maximum": CV_MAXIMUM,
            "sample_maximum_relative_deviation_from_median_maximum": (
                MAXIMUM_RELATIVE_DEVIATION),
            "cross_repetition_convergence": {
                "absolute_floor_ns": CONVERGENCE_ABSOLUTE_NS,
                "relative_to_larger_median_maximum": CONVERGENCE_RELATIVE,
            },
        },
        "vector_protocols": {
            "historical_selection_protocol": dict(
                HISTORICAL_VECTOR_SELECTION_PROTOCOL),
            "recapture_protocol": dict(VECTOR_RECAPTURE_PROTOCOL),
        },
        "vector_stability_audit": vector_audit,
        "groups": groups,
        "family_case_ids": family_case_ids,
        "existing_strict_family_case_ids": existing_strict_family_case_ids,
        "provenance": {
            "completion_manifest": {
                "path": str(completion_manifest_path),
                "sha256": completion_sha256,
            },
            "vector_historical_raw": vector_provenance,
            "canonical_datasets": canonical_provenance,
        },
    }


def _write_json(path, payload):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)
