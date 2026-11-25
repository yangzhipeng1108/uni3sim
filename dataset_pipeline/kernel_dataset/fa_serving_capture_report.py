"""Fail-closed publication audit for frozen FA serving capture candidates.

This module consumes only candidate timing/counter evidence.  It never writes a
canonical dataset and it intentionally refuses incomplete capture sets or an
ambiguous retry selection.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import zstandard

from dataset_pipeline.kernel_dataset.fa_serving_capture import (
    build_capture_plan,
    fa_serving_audit_contract,
    fa_serving_timing_source,
    load_capture_manifest,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


_HARDWARE_IDS = EVALUATED_HARDWARE_IDS
_CANONICAL_IDENTITY_FIELDS = (
    "B", "Q_LEN", "KV_LEN", "HQ", "HKV", "D", "Q_LENS", "KV_LENS",
    "split_policy", "kv_layout", "causal",
)



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, context: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{context} is absent: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object: {path}")
    return value


def _require_fields(value: dict, fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{context} is missing fields: {missing}")


def _require_sha256(value: object, context: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{context} must be a lower SHA256")
    return value


def _write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_csv_new(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _load_canonical(path: Path, hardware_id: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"canonical {hardware_id} dataset is absent: {path}")
    try:
        with path.open("rb") as stream:
            decoded = zstandard.ZstdDecompressor().decompress(stream.read())
    except zstandard.ZstdError as error:
        raise ValueError(f"canonical {hardware_id} dataset is not zstd") from error
    value = json.loads(decoded)
    if not isinstance(value, dict) or len(value) != 294:
        raise ValueError(
            f"canonical {hardware_id} dataset must contain exactly 294 cases")
    return value


def _parse_hardware_paths(values: list[str], label: str) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if value.count("=") != 1:
            raise ValueError(f"{label} must be HARDWARE=PATH: {value!r}")
        hardware_id, raw_path = value.split("=", 1)
        if hardware_id not in _HARDWARE_IDS or not raw_path:
            raise ValueError(f"{label} has an invalid hardware path: {value!r}")
        if hardware_id in parsed:
            raise ValueError(f"{label} repeats {hardware_id}")
        parsed[hardware_id] = Path(raw_path).resolve()
    if set(parsed) != set(_HARDWARE_IDS):
        raise ValueError(f"{label} must supply exactly {_HARDWARE_IDS}")
    return parsed


def _timing_records(root: Path, hardware_id: str, expected: dict) -> dict[str, tuple[Path, dict]]:
    records: dict[str, list[tuple[Path, dict]]] = {}
    for path in sorted(root.rglob("*.timing.json")):
        value = _load_json(path, "timing audit")
        _require_fields(value, ("status", "hardware", "case_id", "workload_digest", "source_snapshot"), "timing audit")
        hardware = value["hardware"]
        if not isinstance(hardware, dict) or hardware["hardware_id"] != hardware_id:
            raise ValueError(f"timing audit has unexpected hardware: {path}")
        case_id = value["case_id"]
        if case_id not in expected:
            raise ValueError(f"timing audit case is outside frozen manifest: {path}")
        if value["workload_digest"] != expected[case_id]["workload_digest"]:
            raise ValueError(f"timing audit digest disagrees with frozen manifest: {path}")
        if value["status"] == "pass":
            aggregation = value["aggregation"] if "aggregation" in value else None
            if (isinstance(aggregation, dict)
                    and aggregation["selection"].startswith("user-approved pooled median")):
                if not fa_serving_audit_contract(
                        hardware_id).pooled_median_timing:
                    raise ValueError(
                        f"pooled timing is only authorized for Ascend: {path}")
                pooled_windows = value["pooled_windows"]
                if not isinstance(pooled_windows, list) or len(pooled_windows) < 3:
                    raise ValueError(f"pooled timing has insufficient windows: {path}")
                if aggregation["pass"] is not True:
                    raise ValueError(f"pooled timing was not explicitly approved: {path}")
            if case_id not in records:
                records[case_id] = []
            records[case_id].append((path, value))
    selected = {}
    for case_id, candidates in records.items():
        if len(candidates) != 1:
            fingerprints = {
                _timing_equivalence_fingerprint(hardware_id, value)
                for _, value in candidates
            }
            if len(fingerprints) != 1:
                raise ValueError(
                    f"{hardware_id} has ambiguous accepted timing retries for {case_id}")
            # The same raw Timeline CSVs were copied to a later counter-only
            # attempt.  Their hashes and selected timing are identical, so the
            # lexical-last audit is merely a canonical provenance location,
            # not a retry-selection rule.
            candidates = [sorted(candidates, key=lambda item: str(item[0]))[-1]]
        selected[case_id] = candidates[0]
    return selected


def _timing_equivalence_fingerprint(hardware_id: str, value: dict) -> str:
    if hardware_id == "metax_c500":
        event = value["event"]
        payload = {
            "snapshot": value["source_snapshot"]["sha256"],
            "selected_duration_ns": event["median_duration_ns"],
            "event_sha256": event["sha256"],
            "trace_sha256": [item["sha256"] for item in value["mctracer"]["shards"]],
        }
    elif hardware_id == "nvidia_a800":
        payload = {
            "snapshot": value["source_snapshot"]["sha256"],
            "selected_duration_ns": value["aggregation"]["median_kernel_sum_ns"],
            "windows": [item["sqlite_sha256"] for item in value["windows"]],
        }
    elif hardware_id == "ascend_910b":
        payload = {
            "snapshot": value["source_snapshot"]["sha256"],
            "selected_duration_ns": value["aggregation"]["median_duration_ns"],
            "windows": [
                (item["duration_ns"], item["op_summary_sha256"])
                for item in value["windows"]
            ],
        }
    else:
        raise AssertionError(f"unhandled hardware: {hardware_id!r}")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _counter_records(root: Path, hardware_id: str, expected: dict,
                     timings: dict) -> dict[str, tuple[Path, dict]]:
    records: dict[str, list[tuple[Path, dict]]] = {}
    for path in sorted(root.rglob("*.counter.json")):
        value = _load_json(path, "counter candidate")
        _require_fields(value, (
            "hardware_id", "case_id", "workload_digest", "source_snapshot_sha256",
            "counter_contract"), "counter candidate")
        if value["hardware_id"] != hardware_id:
            raise ValueError(f"counter candidate has unexpected hardware: {path}")
        case_id = value["case_id"]
        if case_id not in expected:
            raise ValueError(f"counter candidate case is outside frozen manifest: {path}")
        if value["workload_digest"] != expected[case_id]["workload_digest"]:
            raise ValueError(f"counter candidate digest disagrees with frozen manifest: {path}")
        counter_contract = value["counter_contract"]
        if not isinstance(counter_contract, dict) or counter_contract["duration_consumed"] is not False:
            raise ValueError(f"counter candidate consumes duration: {path}")
        if case_id not in timings:
            continue
        timing_snapshot = timings[case_id][1]["source_snapshot"]
        if value["source_snapshot_sha256"] != timing_snapshot["sha256"]:
            continue
        if case_id not in records:
            records[case_id] = []
        records[case_id].append((path, value))
    selected = {}
    for case_id, candidates in records.items():
        if len(candidates) != 1:
            if not fa_serving_audit_contract(
                    hardware_id).retry_disambiguation:
                raise ValueError(
                    f"{hardware_id} has ambiguous accepted counter retries for {case_id}")
            timing_hashes = [
                window["op_summary_sha256"]
                for window in timings[case_id][1]["windows"]
            ]
            for _, candidate in candidates:
                join_hashes = [
                    window["op_summary_sha256"]
                    for window in candidate["timing_launch_join"]["windows"]
                ]
                if join_hashes != timing_hashes:
                    raise ValueError(
                        f"Ascend counter retry does not join selected timing: {case_id}")
            # These are separate duration-free counter collections for the
            # identical frozen Timeline launch set.  A later capture is chosen
            # deterministically without consuming duration or a counter value.
            candidates = [sorted(candidates, key=lambda item: str(item[0]))[-1]]
        selected[case_id] = candidates[0]
    return selected


def _require_complete_pair(*, hardware_id: str, expected: dict,
                           timings: dict, counters: dict) -> dict[str, dict]:
    missing_timing = sorted(set(expected) - set(timings))
    missing_counter = sorted(set(expected) - set(counters))
    permitted_missing_counter = []
    counter_absent = list(
        fa_serving_audit_contract(hardware_id).counter_absent_cases)
    if counter_absent and missing_counter == counter_absent:
        permitted_missing_counter = missing_counter
        missing_counter = []
    if missing_timing or missing_counter:
        raise ValueError(
            f"{hardware_id} capture is incomplete: missing_timing={missing_timing}, "
            f"missing_counter={missing_counter}")
    joined = {}
    for case_id in expected:
        timing_path, timing = timings[case_id]
        if case_id in permitted_missing_counter:
            joined[case_id] = {
                "timing_path": timing_path,
                "timing": timing,
                "counter_path": None,
                "counter": None,
                "counter_status": "unavailable",
                "counter_unavailable_reason": (
                    "mcProfiler did not emit a per-invocation kernel report after "
                    "15 s, 60 s, and 120 s attach waits; aggregate reports are "
                    "diagnostic-only and cannot substitute for per-invocation counters"),
            }
            continue
        counter_path, counter = counters[case_id]
        timing_source = timing["source_snapshot"]
        if (not isinstance(timing_source, dict)
                or timing_source["sha256"] != counter["source_snapshot_sha256"]):
            raise ValueError(f"{hardware_id} timing/counter snapshot mismatch: {case_id}")
        joined[case_id] = {
            "timing_path": timing_path,
            "timing": timing,
            "counter_path": counter_path,
            "counter": counter,
            "counter_status": "available",
        }
    return joined


def _launch_rows(*, hardware_id: str, case: dict, evidence: dict) -> list[dict]:
    timing = evidence["timing"]
    counter = evidence["counter"]
    metadata = case["capture_metadata"]
    common = {
        "hardware_id": hardware_id,
        "case_id": case["case_id"],
        "logical_identity_sha256": metadata["logical_identity_sha256"],
        "workload_digest": case["workload_digest"],
        "phase": metadata["phase"],
        "kv_page_size_tokens": metadata["kv_page_size_tokens"],
        "requested_num_splits": metadata["requested_num_splits"],
        "timing_sha256": _sha256(evidence["timing_path"]),
        "counter_sha256": (
            _sha256(evidence["counter_path"])
            if evidence["counter_path"] is not None else ""),
    }
    rows = []
    if hardware_id == "nvidia_a800":
        kernels = counter["kernels"]
        if not isinstance(kernels, list) or not kernels:
            raise ValueError(f"A800 counter has no physical kernels: {case['case_id']}")
        for ordinal, kernel in enumerate(kernels):
            if kernel["ordinal"] != ordinal:
                raise ValueError(f"A800 counter ordinals are not contiguous: {case['case_id']}")
            program = kernel["identity"]
            rows.append({
                **common, "physical_ordinal": ordinal, "program": program,
                "grid": json.dumps(kernel["grid"]), "block": json.dumps(kernel["block"]),
                "launch_role": _launch_role(program), "observed_execution": "symbol+NSYS launch",
            })
    elif hardware_id == "metax_c500":
        observed = timing["observed_execution"]
        if observed["requested_num_splits"] != 1:
            raise ValueError(f"C500 requested split changed: {case['case_id']}")
        if evidence["counter_status"] == "unavailable":
            return [{
                **common, "physical_ordinal": -1, "program": "",
                "grid": "", "block": "", "launch_role": "unavailable",
                "observed_execution": json.dumps(observed, sort_keys=True),
            }]
        kernels = counter["kernels"]
        if not isinstance(kernels, list) or not kernels:
            raise ValueError(f"C500 counter has no physical kernels: {case['case_id']}")
        body_count = 0
        combine_count = 0
        for ordinal, kernel in enumerate(kernels):
            if kernel["ordinal"] != ordinal:
                raise ValueError(f"C500 counter ordinals are not contiguous: {case['case_id']}")
            program = kernel["program"]
            role = _launch_role(program)
            if role == "splitkv_body":
                body_count += 1
            if role == "splitkv_combine":
                combine_count += 1
            rows.append({
                **common, "physical_ordinal": ordinal, "program": program,
                "grid": json.dumps(kernel["grid"]), "block": str(kernel["block_size"]),
                "launch_role": role,
                "observed_execution": json.dumps(observed, sort_keys=True),
            })
        if body_count != 1:
            raise ValueError(f"C500 SplitKV body count is invalid: {case['case_id']}")
        if observed["separate_combine_kernel_required"] is False and combine_count != 0:
            raise ValueError(f"C500 split=1 emitted an unexpected combine: {case['case_id']}")
        if observed["separate_combine_kernel_required"] is True and combine_count != 1:
            raise ValueError(f"C500 missing required combine: {case['case_id']}")
    elif hardware_id == "ascend_910b":
        kernel = counter["kernel"]
        windows = timing["windows"]
        if not isinstance(windows, list) or len(windows) != 3:
            raise ValueError(f"Ascend timing requires three windows: {case['case_id']}")
        first = windows[0]
        signature = (first["block_dim"], first["mix_block_dim"], first["device_id"])
        for window in windows[1:]:
            if (window["block_dim"], window["mix_block_dim"], window["device_id"]) != signature:
                raise ValueError(f"Ascend timing launch changes between windows: {case['case_id']}")
        rows.append({
            **common, "physical_ordinal": kernel["ordinal"], "program": kernel["program"],
            "grid": str(first["block_dim"]), "block": str(first["mix_block_dim"]),
            "launch_role": "public_op_mix_aic", "observed_execution": "Timeline block_dim/mix_block_dim",
        })
    else:
        raise AssertionError(f"unhandled hardware: {hardware_id!r}")
    return rows


def _launch_role(program: object) -> str:
    if not isinstance(program, str) or not program:
        raise ValueError("physical kernel program must be non-empty")
    if "flash_fwd_splitkv_combine_kernel" in program:
        return "splitkv_combine"
    if "flash_fwd_splitkv_kernel" in program:
        return "splitkv_body"
    return "auxiliary"


def _stability_record(hardware_id: str, case_id: str, timing: dict) -> dict:
    if hardware_id == "metax_c500":
        if timing["status"] != "pass":
            raise ValueError(f"accepted timing lacks a passing stability gate: {case_id}")
        event = timing["event"]
        return {
            "case_id": case_id, "hardware_id": hardware_id,
            "method": "Graph Device Event 50/200 replay agreement",
            "pass": True, "selected_duration_ns": event["median_duration_ns"],
            "spread": event["median_difference_ns"],
            "spread_gate": event["median_difference_tolerance_ns"],
            "coefficient_of_variation": event["selected_coefficient_of_variation"],
        }
    aggregation = timing["aggregation"]
    if aggregation["pass"] is not True or timing["status"] != "pass":
        raise ValueError(f"accepted timing lacks a passing stability gate: {case_id}")
    duration_field = fa_serving_timing_source(hardware_id).duration_field
    pooled = aggregation["selection"].startswith("user-approved pooled median")
    return {
        "case_id": case_id, "hardware_id": hardware_id,
        "method": aggregation["selection"], "pass": True,
        "selected_duration_ns": aggregation[duration_field],
        "spread": aggregation["max_relative_spread"] if not pooled else None,
        "spread_gate": aggregation["max_relative_spread_gate"] if not pooled else None,
        "coefficient_of_variation": None,
    }


def _canonical_matches(case: dict, canonical: dict) -> list[str]:
    workload = case["capture_metadata"]["workload"]
    matches = []
    for key, value in canonical.items():
        op_info = value["op_info"]
        attributes = op_info["Attr"]
        if all(attributes[field] == workload[field] for field in _CANONICAL_IDENTITY_FIELDS):
            matches.append(key)
    return matches


def _add_replace_rows(*, hardware_id: str, cases: dict, canonical: dict,
                      canonical_path: Path) -> list[dict]:
    rows = []
    for case_id, case in cases.items():
        matches = _canonical_matches(case, canonical)
        if not matches:
            action = "add"
            replaced_id = ""
            explanation = "no canonical match on all explicit comparable logical fields"
        elif len(matches) == 1:
            # A partial Attr match cannot legally be a replace: the frozen
            # candidate is BF16, whereas the canonical record has no dtype
            # evidence at all.  Preserve that old record and propose an add
            # rather than guessing that its unrecorded dtype was BF16.
            action = "add"
            replaced_id = ""
            explanation = (
                "canonical Attr matches the non-dtype fields but has no dtype "
                "evidence; preserving it and adding the explicitly BF16 candidate")
        else:
            raise ValueError(
                f"{hardware_id} canonical identity is ambiguous for {case_id}: {matches}")
        rows.append({
            "hardware_id": hardware_id,
            "case_id": case_id,
            "logical_identity_sha256": case["capture_metadata"]["logical_identity_sha256"],
            "action": action,
            "replaced_canonical_id": replaced_id,
            "canonical_dataset": str(canonical_path),
            "canonical_dataset_sha256": _sha256(canonical_path),
            "explanation": explanation,
        })
    return rows


def build_publication_audit(*, capture_manifest: Path, timing_roots: dict,
                            candidate_roots: dict, canonical_paths: dict,
                            output_root: Path) -> dict:
    """Build all final candidate-only artifacts, refusing an incomplete capture."""
    if output_root.exists():
        raise FileExistsError(f"publication audit output already exists: {output_root}")
    loaded = load_capture_manifest(capture_manifest)
    all_rows = []
    stability = []
    completion_records = []
    add_replace = []
    for hardware_id in _HARDWARE_IDS:
        plan = build_capture_plan(loaded, hardware_id=hardware_id)
        expected = {item["case_id"]: item for item in plan["plans"]}
        timings = _timing_records(timing_roots[hardware_id], hardware_id, expected)
        counters = _counter_records(
            candidate_roots[hardware_id], hardware_id, expected, timings)
        evidence = _require_complete_pair(
            hardware_id=hardware_id, expected=expected, timings=timings, counters=counters)
        canonical = _load_canonical(canonical_paths[hardware_id], hardware_id)
        add_replace.extend(_add_replace_rows(
            hardware_id=hardware_id, cases=expected, canonical=canonical,
            canonical_path=canonical_paths[hardware_id]))
        for case_id, case in expected.items():
            item = evidence[case_id]
            all_rows.extend(_launch_rows(
                hardware_id=hardware_id, case=case, evidence=item))
            stability.append(_stability_record(hardware_id, case_id, item["timing"]))
            completion_records.append({
                "hardware_id": hardware_id,
                "case_id": case_id,
                "workload_digest": case["workload_digest"],
                "timing_path": str(item["timing_path"]),
                "timing_sha256": _sha256(item["timing_path"]),
                "counter_status": item["counter_status"],
                "counter_path": (
                    str(item["counter_path"])
                    if item["counter_path"] is not None else None),
                "counter_sha256": (
                    _sha256(item["counter_path"])
                    if item["counter_path"] is not None else None),
                "counter_unavailable_reason": (
                    item["counter_unavailable_reason"]
                    if item["counter_status"] == "unavailable" else None),
                "source_snapshot_sha256": (
                    item["counter"]["source_snapshot_sha256"]
                    if item["counter"] is not None
                    else item["timing"]["source_snapshot"]["sha256"]),
            })
    if len(completion_records) != 120:
        raise AssertionError("publication audit did not produce 120 hardware cases")
    completion_records.sort(key=lambda value: (value["hardware_id"], value["case_id"]))
    all_rows.sort(key=lambda value: (value["hardware_id"], value["case_id"], value["physical_ordinal"]))
    stability.sort(key=lambda value: (value["hardware_id"], value["case_id"]))
    add_replace.sort(key=lambda value: (value["hardware_id"], value["case_id"]))
    output_root.mkdir(parents=True)
    completion_path = output_root / "completion_manifest.json"
    launch_path = output_root / "cross_hardware_launch_contract.csv"
    stability_path = output_root / "stability.json"
    add_replace_path = output_root / "add_replace.csv"
    diff_path = output_root / "candidate_diff_manifest.json"
    summary_path = output_root / "summary.md"
    _write_json_new(completion_path, {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_completion",
        "capture_manifest": str(capture_manifest.resolve()),
        "capture_manifest_sha256": _sha256(capture_manifest),
        "hardware_case_count": len(completion_records),
        "counter_available_case_count": sum(
            record["counter_status"] == "available" for record in completion_records),
        "counter_unavailable_case_count": sum(
            record["counter_status"] == "unavailable" for record in completion_records),
        "records": completion_records,
    })
    launch_columns = (
        "hardware_id", "case_id", "logical_identity_sha256", "workload_digest", "phase",
        "kv_page_size_tokens", "requested_num_splits", "physical_ordinal", "program",
        "grid", "block", "launch_role", "observed_execution", "timing_sha256", "counter_sha256",
    )
    _write_csv_new(launch_path, all_rows, launch_columns)
    _write_json_new(stability_path, {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_stability",
        "records": stability,
    })
    add_columns = (
        "hardware_id", "case_id", "logical_identity_sha256", "action",
        "replaced_canonical_id", "canonical_dataset", "canonical_dataset_sha256", "explanation",
    )
    _write_csv_new(add_replace_path, add_replace, add_columns)
    add_count = sum(row["action"] == "add" for row in add_replace)
    replace_count = sum(row["action"] == "replace" for row in add_replace)
    _write_json_new(diff_path, {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_diff",
        "before_case_count_by_hardware": {hardware_id: 294 for hardware_id in _HARDWARE_IDS},
        "candidate_case_count_by_hardware": {hardware_id: 40 for hardware_id in _HARDWARE_IDS},
        "add_count": add_count,
        "replace_count": replace_count,
        "after_case_count_by_hardware": {
            hardware_id: 294 + sum(
                row["hardware_id"] == hardware_id and row["action"] == "add"
                for row in add_replace) - sum(
                    row["hardware_id"] == hardware_id and row["action"] == "replace"
                    for row in add_replace)
            for hardware_id in _HARDWARE_IDS
        },
        "add_replace_csv_sha256": _sha256(add_replace_path),
    })
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write("# FA serving coverage candidate publication audit\n\n")
        stream.write("All 120 hardware cases passed timing validation; 119 have paired per-invocation counters.\n\n")
        stream.write("| Hardware | Cases | Physical launch rows | Add | Replace |\n")
        stream.write("| --- | ---: | ---: | ---: | ---: |\n")
        for hardware_id in _HARDWARE_IDS:
            stream.write(
                f"| {hardware_id} | 40 | "
                f"{sum(row['hardware_id'] == hardware_id for row in all_rows)} | "
                f"{sum(row['hardware_id'] == hardware_id and row['action'] == 'add' for row in add_replace)} | "
                f"{sum(row['hardware_id'] == hardware_id and row['action'] == 'replace' for row in add_replace)} |\n")
        stream.write("\nC500 split=1 records a SplitKV body and records no separate combine only when the observed public-op evidence says a reduction stage is unnecessary.\n")
        stream.write("\nThe C500 B8/KV8192 case has a validated Event golden and trace, but its counter is explicitly unavailable after three attach-wait attempts; aggregate reports remain diagnostic-only.\n")
    return {
        "completion_manifest": completion_path,
        "cross_hardware_launch_contract": launch_path,
        "stability": stability_path,
        "add_replace": add_replace_path,
        "candidate_diff_manifest": diff_path,
        "summary": summary_path,
    }
