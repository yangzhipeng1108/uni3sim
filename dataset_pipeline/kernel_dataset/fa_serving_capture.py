"""Strict manifest-native FlashAttention serving-capture inputs.

This module owns only candidate-capture input validation and emission.  It
does not read a canonical kernel dataset, select a launch, or derive any
timing.  In particular, every hardware plan is constructed from the frozen
per-request arrays in the capture manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple


_HARDWARE_IDS = (
    "nvidia_a800",
    "metax_c500",
    "ascend_910b",
)


class FaServingTimingSource(NamedTuple):
    """Where one backend's accepted timing record keeps its selected duration.

    Each profiler names its own artifact: mcTracer records a Graph device
    ``event``, nsys and msprof record an ``aggregation`` over windows.  The
    duration is published under ``source_label`` as the evaluation timing
    source.
    """

    container: str
    duration_field: str
    source_label: str


FA_SERVING_TIMING_SOURCES = {
    "nvidia_a800": FaServingTimingSource(
        "aggregation", "median_kernel_sum_ns", "nsys_nvtx_kernel_median"),
    "metax_c500": FaServingTimingSource(
        "event", "median_duration_ns", "device_event_op_median"),
    "ascend_910b": FaServingTimingSource(
        "aggregation", "median_duration_ns", "msprof_timeline_op_median"),
}


class FaServingAuditContract(NamedTuple):
    """Which audit allowances one backend's accepted evidence is entitled to.

    ``pooled_median_timing``: accepted timing may be a user-approved pooled
    median over windows (a backend whose single-shot capture cannot meet the
    stability gate).
    ``retry_disambiguation``: several accepted candidates for one case are
    resolved by comparing window hashes rather than rejected as ambiguous.
    ``counter_absent_cases``: case ids whose counter evidence is legitimately
    unavailable, so a missing counter is permitted for exactly those.
    """

    pooled_median_timing: bool
    retry_disambiguation: bool
    counter_absent_cases: tuple = ()


# The one C500 case whose counter evidence is legitimately unavailable.
_METAX_C500_COUNTER_UNAVAILABLE_CASE = (
    "FA_SERVING_HQ32_HKV8_D128_DECODE_B8_QSUM8_QMAX1_KVMAX8192_61df13118f3a")

FA_SERVING_AUDIT_CONTRACTS = {
    "nvidia_a800": FaServingAuditContract(False, False),
    "metax_c500": FaServingAuditContract(
        False, False, (_METAX_C500_COUNTER_UNAVAILABLE_CASE,)),
    "ascend_910b": FaServingAuditContract(True, True),
}


def fa_serving_audit_contract(hardware_id: str) -> FaServingAuditContract:
    if hardware_id not in FA_SERVING_AUDIT_CONTRACTS:
        raise AssertionError(f"unhandled hardware: {hardware_id!r}")
    return FA_SERVING_AUDIT_CONTRACTS[hardware_id]


def fa_serving_timing_source(hardware_id: str) -> FaServingTimingSource:
    if hardware_id not in FA_SERVING_TIMING_SOURCES:
        raise AssertionError(f"unhandled hardware: {hardware_id!r}")
    return FA_SERVING_TIMING_SOURCES[hardware_id]
_ROOT_FIELDS = frozenset({
    "baseline_commit",
    "canonical_dataset_mutation_authorized",
    "cases",
    "expected_hardware_case_count",
    "hardware_ids",
    "kind",
    "logical_case_count",
    "schema_version",
    "scope_contract",
})
_SCOPE_FIELDS = frozenset({
    "causal_by_phase",
    "dtype",
    "hardware_public_api_contract",
    "head_dim",
    "kv_dist_semantics",
    "kv_layout",
    "logical_identity_fields",
    "observed_not_prescribed_fields",
    "same_logical_input_on_every_hardware",
    "source_case_count",
    "synthetic_case_count",
    "topology_phase_counts",
})
_CASE_FIELDS = frozenset({
    "case_id",
    "logical_identity_sha256",
    "phase",
    "publication_action",
    "purpose",
    "source",
    "workload",
})
_WORKLOAD_FIELDS = frozenset({
    "B",
    "D",
    "HKV",
    "HQ",
    "KV_LEN",
    "KV_LENS",
    "Q_LEN",
    "Q_LENS",
    "causal",
    "dtype",
    "kv_dist",
    "kv_layout",
    "split_policy",
})
_IDENTITY_FIELDS = (
    "phase",
    "dtype",
    "kv_layout",
    "causal",
    "split_policy",
    "B",
    "Q_LEN",
    "KV_LEN",
    "HQ",
    "HKV",
    "D",
    "Q_LENS",
    "KV_LENS",
)
_TOPOLOGY_PHASE_COUNTS = {
    "HQ32_HKV8": {"prefill": 3, "decode": 9, "mixed": 6},
    "HQ32_HKV4": {"prefill": 3, "decode": 4, "mixed": 4},
    "HQ16_HKV2": {"prefill": 3, "decode": 4, "mixed": 4},
}
_HARDWARE_CAPTURE_CONTRACT = {
    "nvidia_a800": {
        "kv_page_size_tokens": 16,
        "requested_num_splits": 1,
        "runner": "scripts/kernel_dataset/nvidia_a800/flash_attn_runner.py",
    },
    "metax_c500": {
        "kv_page_size_tokens": 16,
        "requested_num_splits": 1,
        "runner": "scripts/kernel_dataset/metax_c500/flash_attn_runner.py",
    },
    "ascend_910b": {
        "kv_page_size_tokens": 128,
        "requested_num_splits": None,
        "runner": "scripts/kernel_dataset/ascend_910b/flash_attn_runner.py",
    },
}
_CAPTURE_METADATA_FIELDS = frozenset({
    "capture_kind",
    "capture_manifest_sha256",
    "case_id",
    "hardware_id",
    "kv_page_size_tokens",
    "logical_identity_sha256",
    "phase",
    "requested_num_splits",
    "schema_version",
    "workload",
    "workload_digest",
})


def _require_mapping(value, context):
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _require_exact_fields(value, fields, context):
    mapping = _require_mapping(value, context)
    actual = frozenset(mapping)
    if actual != fields:
        raise ValueError(
            f"{context} fields are invalid: "
            f"missing={sorted(fields - actual)}, "
            f"unexpected={sorted(actual - fields)}")
    return mapping


def _require_positive_int(value, context):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    if value <= 0:
        raise ValueError(f"{context} must be positive")
    return value


def _require_lower_hex(value, length, context):
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    if len(value) != length or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{context} must be a {length}-character lower hex string")
    return value


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repository_root(manifest_path):
    resolved = manifest_path.resolve()
    if len(resolved.parents) < 4:
        raise ValueError(f"capture manifest path is too shallow: {resolved}")
    return resolved.parents[3]


def _identity_payload(case):
    workload = case["workload"]
    payload = {}
    for field in _IDENTITY_FIELDS:
        payload[field] = case["phase"] if field == "phase" else workload[field]
    return payload


def logical_identity_sha256(case):
    """Return the stable logical identity SHA-256 for one manifest case."""
    return _sha256_text(_canonical_json(_identity_payload(case)))


def workload_digest(case):
    """Digest the complete manifest-owned executable input, including provenance."""
    return _sha256_text(_canonical_json({
        "case_id": case["case_id"],
        "logical_identity_sha256": case["logical_identity_sha256"],
        "phase": case["phase"],
        "workload": case["workload"],
    }))


def _validate_source(case, repository_root):
    source = _require_mapping(case["source"], f"case {case['case_id']} source")
    source_kind = source["kind"]
    workload = case["workload"]
    if source_kind == "canonical_serving_alignment_step":
        _require_exact_fields(
            source,
            frozenset({
                "alignment_steps_zst_sha256", "case_path", "kind", "phase",
                "request_phases", "source_execution_order", "source_step_id",
            }),
            f"case {case['case_id']} alignment source")
        if source["phase"] != case["phase"]:
            raise ValueError(f"case {case['case_id']} source phase disagrees")
        if not isinstance(source["case_path"], str) or not source["case_path"]:
            raise ValueError(f"case {case['case_id']} source case_path is invalid")
        source_path = (repository_root / source["case_path"] / "alignment_steps.json.zst").resolve()
        expected_root = (repository_root / "serving_dataset").resolve()
        if expected_root not in source_path.parents:
            raise ValueError(f"case {case['case_id']} source path escapes serving_dataset")
        if not source_path.is_file():
            raise FileNotFoundError(f"case {case['case_id']} source is absent: {source_path}")
        expected_sha256 = _require_lower_hex(
            source["alignment_steps_zst_sha256"], 64,
            f"case {case['case_id']} alignment source SHA-256")
        if _sha256_file(source_path) != expected_sha256:
            raise ValueError(f"case {case['case_id']} alignment source SHA-256 changed")
        _require_positive_int(
            source["source_execution_order"] + 1,
            f"case {case['case_id']} source execution order")
        if not isinstance(source["source_step_id"], str) or not source["source_step_id"]:
            raise ValueError(f"case {case['case_id']} source step ID is invalid")
        request_phases = source["request_phases"]
        if not isinstance(request_phases, list) or len(request_phases) != workload["B"]:
            raise ValueError(f"case {case['case_id']} source request phases are invalid")
        if any(phase not in ("prefill", "decode") for phase in request_phases):
            raise ValueError(f"case {case['case_id']} source request phase is invalid")
        return source_kind
    if source_kind == "synthetic_scope_point":
        _require_exact_fields(
            source, frozenset({"derivation", "kind", "sequence_length"}),
            f"case {case['case_id']} synthetic source")
        sequence_length = _require_positive_int(
            source["sequence_length"], f"case {case['case_id']} sequence length")
        if (any(value != sequence_length for value in workload["KV_LENS"])
                or (case["phase"] == "prefill"
                    and workload["Q_LENS"] != [sequence_length])
                or (case["phase"] == "decode"
                    and any(value != 1 for value in workload["Q_LENS"]))):
            raise ValueError(f"case {case['case_id']} synthetic source disagrees with workload")
        if not isinstance(source["derivation"], str) or not source["derivation"]:
            raise ValueError(f"case {case['case_id']} synthetic derivation is invalid")
        return source_kind
    raise ValueError(f"case {case['case_id']} source kind is unsupported: {source_kind!r}")


def _validate_case(case, repository_root):
    _require_exact_fields(case, _CASE_FIELDS, "capture manifest case")
    case_id = case["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("capture manifest case_id is invalid")
    phase = case["phase"]
    if phase not in ("prefill", "decode", "mixed"):
        raise ValueError(f"case {case_id} phase is invalid")
    if case["publication_action"] != "pending_exact_semantic_comparison":
        raise ValueError(f"case {case_id} publication action is invalid")
    if not isinstance(case["purpose"], str) or not case["purpose"]:
        raise ValueError(f"case {case_id} purpose is invalid")
    workload = _require_exact_fields(
        case["workload"], _WORKLOAD_FIELDS, f"case {case_id} workload")
    if workload["dtype"] != "bf16" or workload["kv_layout"] != "paged_varlen":
        raise ValueError(f"case {case_id} does not use the frozen FA contract")
    if workload["split_policy"] != "serving":
        raise ValueError(f"case {case_id} has a non-serving split policy")
    if workload["kv_dist"] not in ("explicit", "uniform"):
        raise ValueError(f"case {case_id} KV distribution is invalid")
    if not isinstance(workload["causal"], bool):
        raise TypeError(f"case {case_id} causal must be boolean")
    expected_causal = {"prefill": True, "decode": False, "mixed": True}[phase]
    if workload["causal"] is not expected_causal:
        raise ValueError(f"case {case_id} causal flag does not match phase")
    for field in ("B", "Q_LEN", "KV_LEN", "HQ", "HKV", "D"):
        _require_positive_int(workload[field], f"case {case_id} workload.{field}")
    if workload["D"] != 128 or (workload["HQ"], workload["HKV"]) not in {
            (32, 8), (32, 4), (16, 2)}:
        raise ValueError(f"case {case_id} topology is outside frozen scope")
    for field, maximum in (("Q_LENS", workload["Q_LEN"]), ("KV_LENS", workload["KV_LEN"])):
        values = workload[field]
        if not isinstance(values, list) or len(values) != workload["B"]:
            raise ValueError(f"case {case_id} workload.{field} has the wrong request count")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError(f"case {case_id} workload.{field} has an invalid length")
        if max(values) != maximum:
            raise ValueError(f"case {case_id} workload.{field} maximum disagrees")
    if any(query_length > key_value_length for query_length, key_value_length in zip(
            workload["Q_LENS"], workload["KV_LENS"])):
        raise ValueError(f"case {case_id} has a request with Q_LEN > KV_LEN")
    if phase == "decode" and any(value != 1 for value in workload["Q_LENS"]):
        raise ValueError(f"case {case_id} decode must contain only one-token queries")
    if phase == "prefill" and any(value <= 1 for value in workload["Q_LENS"]):
        raise ValueError(f"case {case_id} prefill must contain only multi-token queries")
    if phase == "mixed" and (not any(value == 1 for value in workload["Q_LENS"])
                             or not any(value > 1 for value in workload["Q_LENS"])):
        raise ValueError(f"case {case_id} mixed phase must contain decode and prefill requests")
    identity = logical_identity_sha256(case)
    if case["logical_identity_sha256"] != identity:
        raise ValueError(f"case {case_id} logical identity SHA-256 disagrees")
    expected_case_id = (
        f"FA_SERVING_HQ{workload['HQ']}_HKV{workload['HKV']}_D{workload['D']}_"
        f"{phase.upper()}_B{workload['B']}_QSUM{sum(workload['Q_LENS'])}_"
        f"QMAX{workload['Q_LEN']}_KVMAX{workload['KV_LEN']}_{identity[:12]}")
    if case_id != expected_case_id:
        raise ValueError(f"case ID disagrees with its complete logical identity: {case_id}")
    return _validate_source(case, repository_root)


def load_capture_manifest(path):
    """Load and fail closed on the frozen 40-case serving capture manifest."""
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    _require_exact_fields(manifest, _ROOT_FIELDS, "FA serving capture manifest")
    if manifest["schema_version"] != 1:
        raise ValueError("FA serving capture manifest schema is unsupported")
    if manifest["kind"] != "flash_attn_cross_hardware_capture_scope":
        raise ValueError("FA serving capture manifest kind is invalid")
    _require_lower_hex(manifest["baseline_commit"], 40, "capture manifest baseline commit")
    if manifest["canonical_dataset_mutation_authorized"] is not False:
        raise ValueError("candidate capture must not authorize canonical mutation")
    if manifest["hardware_ids"] != list(_HARDWARE_IDS):
        raise ValueError("capture manifest hardware IDs are not the frozen three-card scope")
    if manifest["logical_case_count"] != 40 or manifest["expected_hardware_case_count"] != 120:
        raise ValueError("capture manifest case-count contract is invalid")
    scope = _require_exact_fields(manifest["scope_contract"], _SCOPE_FIELDS, "capture scope")
    if (scope["dtype"] != "bf16" or scope["kv_layout"] != "paged_varlen"
            or scope["head_dim"] != 128 or scope["same_logical_input_on_every_hardware"] is not True):
        raise ValueError("capture scope fixed workload contract is invalid")
    if scope["causal_by_phase"] != {"prefill": True, "decode": False, "mixed": True}:
        raise ValueError("capture scope causal contract is invalid")
    if scope["logical_identity_fields"] != list(_IDENTITY_FIELDS):
        raise ValueError("capture scope logical identity fields are invalid")
    if scope["topology_phase_counts"] != _TOPOLOGY_PHASE_COUNTS:
        raise ValueError("capture scope topology/phase counts are invalid")
    if scope["source_case_count"] != 32 or scope["synthetic_case_count"] != 8:
        raise ValueError("capture scope source counts are invalid")
    if scope["observed_not_prescribed_fields"] != [
            "program_or_variant", "grid_or_block_dim", "body_and_combine_sequence",
            "executed_split_count"]:
        raise ValueError("capture scope observed-field contract is invalid")
    if scope["kv_dist_semantics"] != (
            "capture provenance only; exact Q_LENS/KV_LENS are authoritative and "
            "kv_dist is excluded from logical identity"):
        raise ValueError("capture scope KV-distribution contract is invalid")
    api_contract = scope["hardware_public_api_contract"]
    _require_exact_fields(api_contract, frozenset(_HARDWARE_IDS), "hardware public API contract")
    for hardware_id in _HARDWARE_IDS:
        contract = _require_exact_fields(
            api_contract[hardware_id],
            frozenset({"kv_block_size_tokens", "meaning", "requested_num_splits"}),
            f"{hardware_id} public API contract")
        expected = _HARDWARE_CAPTURE_CONTRACT[hardware_id]
        if (contract["kv_block_size_tokens"] != expected["kv_page_size_tokens"]
                or contract["requested_num_splits"] != expected["requested_num_splits"]
                or not isinstance(contract["meaning"], str) or not contract["meaning"]):
            raise ValueError(f"{hardware_id} public API contract is invalid")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != manifest["logical_case_count"]:
        raise ValueError("capture manifest cases are invalid")
    repository_root = _repository_root(manifest_path)
    case_ids = set()
    identities = set()
    source_counts = {"canonical_serving_alignment_step": 0, "synthetic_scope_point": 0}
    topology_phase_counts = {
        topology: {"prefill": 0, "decode": 0, "mixed": 0}
        for topology in _TOPOLOGY_PHASE_COUNTS
    }
    for case in cases:
        source_kind = _validate_case(case, repository_root)
        case_id = case["case_id"]
        identity = case["logical_identity_sha256"]
        if case_id in case_ids or identity in identities:
            raise ValueError("capture manifest contains a duplicate logical case")
        case_ids.add(case_id)
        identities.add(identity)
        source_counts[source_kind] += 1
        workload = case["workload"]
        topology = f"HQ{workload['HQ']}_HKV{workload['HKV']}"
        topology_phase_counts[topology][case["phase"]] += 1
    if source_counts != {
            "canonical_serving_alignment_step": scope["source_case_count"],
            "synthetic_scope_point": scope["synthetic_case_count"]}:
        raise ValueError("capture manifest source census is invalid")
    if topology_phase_counts != scope["topology_phase_counts"]:
        raise ValueError("capture manifest topology/phase census is invalid")
    return {
        "path": manifest_path,
        "sha256": _sha256_file(manifest_path),
        "manifest": manifest,
    }


def capture_metadata(case, *, manifest_sha256, hardware_id):
    """Build the immutable metadata that every raw runner must echo."""
    if hardware_id not in _HARDWARE_CAPTURE_CONTRACT:
        raise ValueError(f"unsupported hardware ID: {hardware_id!r}")
    contract = _HARDWARE_CAPTURE_CONTRACT[hardware_id]
    return {
        "schema_version": 1,
        "capture_kind": "fa_serving_coverage_candidate",
        "capture_manifest_sha256": manifest_sha256,
        "case_id": case["case_id"],
        "logical_identity_sha256": case["logical_identity_sha256"],
        "workload_digest": workload_digest(case),
        "hardware_id": hardware_id,
        "phase": case["phase"],
        "kv_page_size_tokens": contract["kv_page_size_tokens"],
        "requested_num_splits": contract["requested_num_splits"],
        "workload": case["workload"],
    }


def validate_runner_capture_metadata(path, *, hardware_id, actual_workload):
    """Validate a runner sidecar against the exact CLI workload it received."""
    with Path(path).open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    _require_exact_fields(metadata, _CAPTURE_METADATA_FIELDS, "FA runner capture metadata")
    if metadata["schema_version"] != 1 or metadata["capture_kind"] != "fa_serving_coverage_candidate":
        raise ValueError("FA runner capture metadata schema is invalid")
    _require_lower_hex(metadata["capture_manifest_sha256"], 64, "capture manifest SHA-256")
    _require_lower_hex(metadata["logical_identity_sha256"], 64, "logical identity SHA-256")
    _require_lower_hex(metadata["workload_digest"], 64, "workload digest")
    if metadata["hardware_id"] != hardware_id:
        raise ValueError("runner capture metadata hardware disagrees")
    if metadata["phase"] not in ("prefill", "decode", "mixed"):
        raise ValueError("runner capture metadata phase is invalid")
    expected_contract = _HARDWARE_CAPTURE_CONTRACT[hardware_id]
    if (metadata["kv_page_size_tokens"] != expected_contract["kv_page_size_tokens"]
            or metadata["requested_num_splits"] != expected_contract["requested_num_splits"]):
        raise ValueError("runner capture metadata public API contract disagrees")
    synthetic_case = {
        "case_id": metadata["case_id"],
        "logical_identity_sha256": metadata["logical_identity_sha256"],
        "phase": metadata["phase"],
        "workload": metadata["workload"],
    }
    if metadata["workload"] != actual_workload:
        raise ValueError("runner CLI workload differs from frozen capture metadata")
    if logical_identity_sha256(synthetic_case) != metadata["logical_identity_sha256"]:
        raise ValueError("runner capture metadata identity disagrees")
    if workload_digest(synthetic_case) != metadata["workload_digest"]:
        raise ValueError("runner capture metadata digest disagrees")
    workload = metadata["workload"]
    expected_case_id = (
        f"FA_SERVING_HQ{workload['HQ']}_HKV{workload['HKV']}_D{workload['D']}_"
        f"{metadata['phase'].upper()}_B{workload['B']}_QSUM{sum(workload['Q_LENS'])}_"
        f"QMAX{workload['Q_LEN']}_KVMAX{workload['KV_LEN']}_"
        f"{metadata['logical_identity_sha256'][:12]}")
    if metadata["case_id"] != expected_case_id:
        raise ValueError("runner capture metadata case ID disagrees")
    return metadata


def build_capture_plan(loaded_manifest, *, hardware_id, case_ids=None):
    """Emit deterministic runner inputs without executing or selecting a launch."""
    if hardware_id not in _HARDWARE_CAPTURE_CONTRACT:
        raise ValueError(f"unsupported hardware ID: {hardware_id!r}")
    manifest = loaded_manifest["manifest"]
    if case_ids is None:
        selected_cases = manifest["cases"]
    else:
        if not isinstance(case_ids, list) or not case_ids:
            raise ValueError("case_ids must be a non-empty list when supplied")
        by_case_id = {case["case_id"]: case for case in manifest["cases"]}
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_ids must not contain duplicates")
        selected_cases = []
        for case_id in case_ids:
            if case_id not in by_case_id:
                raise ValueError(f"case_id is outside the frozen manifest: {case_id}")
            selected_cases.append(by_case_id[case_id])
    contract = _HARDWARE_CAPTURE_CONTRACT[hardware_id]
    plans = []
    for case in selected_cases:
        workload = case["workload"]
        metadata = capture_metadata(
            case, manifest_sha256=loaded_manifest["sha256"], hardware_id=hardware_id)
        common = [
            "--batch", str(workload["B"]),
            "--seq_len_cache", str(workload["KV_LEN"]),
            "--num_heads_q", str(workload["HQ"]),
            "--num_heads_kv", str(workload["HKV"]),
            "--head_dim", str(workload["D"]),
            "--dtype", workload["dtype"],
            "--seq_len_q", str(workload["Q_LEN"]),
            "--kv_layout", workload["kv_layout"],
            "--kv_dist", workload["kv_dist"],
            "--causal", str(int(workload["causal"])),
            "--capture-metadata-json", "{capture_metadata_json}",
        ]
        if hardware_id == "ascend_910b":
            runner_arguments = [
                *common,
                "--workload-json", "{runner_workload_json}",
            ]
        else:
            runner_arguments = [
                *common,
                "--q-lens-json", _canonical_json(workload["Q_LENS"]),
                "--kv-lens-json", _canonical_json(workload["KV_LENS"]),
                "--num_splits", str(contract["requested_num_splits"]),
            ]
            if hardware_id == "nvidia_a800":
                runner_arguments.extend(("--nvtx_range_name", case["case_id"]))
        plans.append({
            "hardware_id": hardware_id,
            "runner": contract["runner"],
            "case_id": case["case_id"],
            "workload_digest": metadata["workload_digest"],
            "capture_metadata": metadata,
            "runner_workload": {
                "q_lens": workload["Q_LENS"],
                "kv_lens": workload["KV_LENS"],
            },
            "runner_arguments": runner_arguments,
        })
    return {
        "schema_version": 1,
        "capture_kind": "fa_serving_coverage_candidate_plan",
        "manifest_path": str(loaded_manifest["path"]),
        "manifest_sha256": loaded_manifest["sha256"],
        "hardware_id": hardware_id,
        "case_count": len(plans),
        "plans": plans,
    }
