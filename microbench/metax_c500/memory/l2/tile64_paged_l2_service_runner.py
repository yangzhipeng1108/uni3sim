#!/usr/bin/env python3
"""Freeze and validate the C500 tile64 GQA L2-residency experiment."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 3
STDOUT_SCHEMA_VERSION = 2
HARDWARE_ID = "metax_c500"
MECHANISM = "paged_tile64_gqa_l2_residency"
# This is the first revision carrying the paged-tile mechanism contract.  The
# exact instrumented source is identified by SOURCE_SHA256 and the deployed
# snapshot revision is retained separately in provenance.
BASELINE_COMMIT = "a029e071f6d1c0e0e7f52e95f2c26e5288ea74a4"
SOURCE_RELATIVE_PATH = (
    "microbench/metax_c500/memory/l2/tile64_paged_l2_service.cpp")
SOURCE_SHA256 = (
    "aae68a2c8e7de1a7652d866cc9640a220b5f84b1e0e86d62626573cf744257f6")
BLOCK_THREADS = 256
WAVE_SIZE = 64
DYNAMIC_SHARED_BYTES = 32 * 1024
QUERY_HEADS = 32
QUERY_TILES = 64
BATCH = 128
QUERY_TILE_TOKENS = 64
KV_TILE_TOKENS = 64
HEAD_DIM = 128
ELEMENT_BYTES = 2
TILE_BYTES_PER_OPERAND = KV_TILE_TOKENS * HEAD_DIM * ELEMENT_BYTES
EVENT_TRIALS = 3
ATTACH_WAIT_SECONDS = 15
L2_CONTRACT_RELATIVE_ERROR_LIMIT = 0.10
EVENT_RELATIVE_SPAN_LIMIT = 0.05
CAUSAL_KV_STEPS_RULE = "q_tile_index_plus_one"
CASES = (
    ("reuse1_pool224", "c500fa_gqa_r1_p224", 1, 224),
    ("reuse4_pool224", "c500fa_gqa_r4_p224", 4, 224),
    ("reuse1_pool288", "c500fa_gqa_r1_p288", 1, 288),
    ("reuse4_pool288", "c500fa_gqa_r4_p288", 4, 288),
)
STDOUT_FIELDS = (
    "mode", "case_id", "kernel_identity",
    "grid_x", "grid_y", "grid_z", "workgroups", "block",
    "dynamic_shared_bytes", "query_heads", "q_tiles", "batch",
    "q_tile_tokens", "kv_tile_tokens", "head_dim",
    "consumer_reuse_group_size", "page_owner_count",
    "pool_pages_per_operand", "combined_kv_working_set_bytes",
    "causal_kv_steps_rule", "logical_k_read_bytes",
    "logical_v_read_bytes", "sink_write_bytes", "event_trials",
    "event_ms_min", "event_ms_median", "event_ms_max",
    "first_sink", "last_sink",
)
INTEGER_STDOUT_FIELDS = (
    "grid_x", "grid_y", "grid_z", "workgroups", "block",
    "dynamic_shared_bytes", "query_heads", "q_tiles", "batch",
    "q_tile_tokens", "kv_tile_tokens", "head_dim",
    "consumer_reuse_group_size", "page_owner_count",
    "pool_pages_per_operand", "combined_kv_working_set_bytes",
    "logical_k_read_bytes", "logical_v_read_bytes", "sink_write_bytes",
    "event_trials",
)
PROVENANCE_KEYS = {
    "schema_version", "hardware_id", "mechanism", "baseline_commit",
    "software_revision", "container_image", "physical_device_id",
    "production_or_evaluation_duration_consumed", "plan_path",
    "plan_sha256", "source_path", "source_sha256", "runner_path",
    "runner_sha256", "binary_path", "binary_sha256",
    "device_preflight_path", "device_preflight_sha256",
    "raw_stdout_policy",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_root():
    return Path(__file__).resolve().parents[4]


def audit_source():
    path = _repository_root() / SOURCE_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"tile64 source is missing: {path}")
    if _sha256(path) != SOURCE_SHA256:
        raise ValueError("tile64 source differs from the frozen source")
    source = path.read_text(encoding="utf-8")
    required = (
        "extern __shared__ std::uint32_t occupancy_control[]",
        "kDynamicSharedBytes = 32 * 1024",
        "kQueryHeads = 32",
        "kQueryTiles = 64",
        "kBatch = 128",
        "step <= q_tile",
        "q_head / ConsumerReuseGroupSize",
        "% PoolPagesPerOperand",
        "CHECK(mcProfilerStart());",
        "CHECK(mcProfilerStop());",
        "O_WRONLY | O_CREAT | O_EXCL",
        "open_profile_output",
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("tile64 source contract changed")
    forbidden = (
        "const std::uint32_t* page_table", "mma_sync(",
        "__builtin_mma", "expf(", "flash_attn(",
    )
    lowered = source.lower()
    if any(fragment in lowered for fragment in forbidden):
        raise ValueError("tile64 source contains excluded production work")
    marker = source[
        source.index("CHECK(mcProfilerStart());"):
        source.index("CHECK(mcProfilerStop());")
    ]
    if marker.count("launch_case(") != 1:
        raise ValueError("tile64 profile marker must contain one target launch")
    return path


def _sample(case_id, kernel_identity, reuse_group, pool_pages):
    workgroups = QUERY_HEADS * QUERY_TILES * BATCH
    causal_steps = QUERY_TILES * (QUERY_TILES + 1) // 2
    operand_tile_loads = BATCH * QUERY_HEADS * causal_steps
    operand_bytes = operand_tile_loads * TILE_BYTES_PER_OPERAND
    sink_write_bytes = workgroups * BLOCK_THREADS * 4
    return {
        "case_id": case_id,
        "kernel_identity": kernel_identity,
        "evidence_role": "counter_mechanism",
        "grid_x": QUERY_HEADS,
        "grid_y": QUERY_TILES,
        "grid_z": BATCH,
        "workgroups": workgroups,
        "waves": workgroups * BLOCK_THREADS // WAVE_SIZE,
        "block": BLOCK_THREADS,
        "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
        "query_heads": QUERY_HEADS,
        "q_tiles": QUERY_TILES,
        "batch": BATCH,
        "q_tile_tokens": QUERY_TILE_TOKENS,
        "kv_tile_tokens": KV_TILE_TOKENS,
        "head_dim": HEAD_DIM,
        "consumer_reuse_group_size": reuse_group,
        "page_owner_count": QUERY_HEADS // reuse_group,
        "pool_pages_per_operand": pool_pages,
        "combined_kv_working_set_bytes": (
            2 * pool_pages * TILE_BYTES_PER_OPERAND),
        "causal_kv_steps_rule": CAUSAL_KV_STEPS_RULE,
        "operand_tile_loads": operand_tile_loads,
        "logical_k_read_bytes": operand_bytes,
        "logical_v_read_bytes": operand_bytes,
        "sink_write_bytes": sink_write_bytes,
        "expected_l2_bytes": 2 * operand_bytes + sink_write_bytes,
        "event_trials": EVENT_TRIALS,
        "event_binary_arguments": ["--event", case_id],
        "profile_binary_arguments": [
            "--profile-once", case_id, str(ATTACH_WAIT_SECONDS),
            "<exclusive_profile_stdout.csv>"],
    }


def build_plan():
    audit_source()
    samples = [_sample(*case) for case in CASES]
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "baseline_commit": BASELINE_COMMIT,
        "production_or_evaluation_duration_consumed": False,
        "source": {
            "path": SOURCE_RELATIVE_PATH,
            "sha256": SOURCE_SHA256,
        },
        "fixed_contract": {
            "dtype": "bf16_width_bitwise_load",
            "query_heads": QUERY_HEADS,
            "query_tiles": QUERY_TILES,
            "batch": BATCH,
            "q_tile_tokens": QUERY_TILE_TOKENS,
            "kv_tile_tokens": KV_TILE_TOKENS,
            "head_dim": HEAD_DIM,
            "block": BLOCK_THREADS,
            "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
            "causal_kv_steps_rule": CAUSAL_KV_STEPS_RULE,
            "page_mapping": (
                "(8191*linear(batch,owner,q_tile,kv_step)+17)%pool_pages"),
            "consumer_mapping": {
                "control": "owner=q_head",
                "gqa4": "owner=q_head//4",
            },
            "excluded_work": [
                "page_table_load", "mma", "softmax", "reduction",
                "output_kernel", "production_operator",
            ],
        },
        "duration_contract": {
            "independent_microbenchmark_device_event_consumed": False,
            "mcprofiler_instrumented_duration_consumed": False,
            "production_or_evaluation_duration_consumed": False,
        },
        "mcprofiler_contract": {
            "mode": "custom_per_kernel",
            "marker_kernel_invocations": 1,
            "isolated_report_ordinal": 1,
            "attach_wait_seconds": ATTACH_WAIT_SECONDS,
            "duration_fields_consumed": False,
            "profile_stdout_delivery": (
                "target writes an exclusive CSV after mcProfilerStop"),
            "required_non_duration_evidence": [
                "WORKGROUPS", "WAVES", "l2c_memacs", "all_memacs",
                "Global Memory Read bytes", "Global Memory Write bytes",
                "Dnoc Read Req", "Dnoc Write Req",
            ],
        },
        "release_contract": {
            "counter_only": True,
            "l2_contract_relative_error_limit": (
                L2_CONTRACT_RELATIVE_ERROR_LIMIT),
            "within_pool_rule": (
                "downstream_read(reuse4)<=downstream_read(reuse1)"),
            "capacity_boundary_rule": (
                "downstream_read(pool288)>downstream_read(pool224) "
                "for reuse1 and reuse4"),
            "production_duration_consumed": False,
        },
        "question": (
            "Do four adjacent query-head consumers reuse the same paged K/V "
            "stream, and does that reuse change when the combined K/V pool "
            "crosses the documented 8-MiB L2 capacity?"),
        "samples": samples,
    }


def write_plan(path):
    if path.exists():
        raise FileExistsError(f"plan path already exists: {path}")
    payload = build_plan()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return payload


def read_plan(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = build_plan()
    if payload != expected:
        raise ValueError("tile64 plan differs from the frozen generator")
    return expected


def _parse_integer(text, label):
    if not text or not text.isascii() or not text.isdigit():
        raise ValueError(f"{label} must be a canonical non-negative integer")
    if len(text) > 1 and text.startswith("0"):
        raise ValueError(f"{label} must not contain leading zeroes")
    return int(text)


def _parse_float(text, label, *, positive):
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} has an invalid value")
    return value


def _parse_hex(text, label):
    if (len(text) != 8 or not text.isascii()
            or any(character not in "0123456789abcdef" for character in text)):
        raise ValueError(f"{label} must be eight lowercase hexadecimal digits")
    return int(text, 16)


def parse_stdout(path, planned, expected_mode):
    if expected_mode == "event":
        expected_event_trials = planned["event_trials"]
    elif expected_mode == "profile":
        expected_event_trials = 0
    else:
        raise ValueError(f"unknown stdout mode: {expected_mode!r}")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 3:
        raise ValueError(f"{path} must contain metadata, header, and sample")
    if tuple(rows[0]) != (
            "metadata", str(STDOUT_SCHEMA_VERSION), HARDWARE_ID, MECHANISM,
            "production_or_evaluation_duration_consumed=0",
            "mcprofiler_instrumented_duration_consumed=0"):
        raise ValueError(f"{path} metadata is invalid")
    if tuple(rows[1]) != ("sample_header",) + STDOUT_FIELDS:
        raise ValueError(f"{path} sample header is invalid")
    if len(rows[2]) != len(STDOUT_FIELDS) + 1 or rows[2][0] != "sample":
        raise ValueError(f"{path} sample row is invalid")
    values = dict(zip(STDOUT_FIELDS, rows[2][1:]))
    parsed = {
        "mode": values["mode"],
        "case_id": values["case_id"],
        "kernel_identity": values["kernel_identity"],
        "causal_kv_steps_rule": values["causal_kv_steps_rule"],
        "event_ms_min": _parse_float(
            values["event_ms_min"], "event_ms_min",
            positive=expected_mode == "event"),
        "event_ms_median": _parse_float(
            values["event_ms_median"], "event_ms_median",
            positive=expected_mode == "event"),
        "event_ms_max": _parse_float(
            values["event_ms_max"], "event_ms_max",
            positive=expected_mode == "event"),
        "first_sink": _parse_hex(values["first_sink"], "first_sink"),
        "last_sink": _parse_hex(values["last_sink"], "last_sink"),
    }
    for field in INTEGER_STDOUT_FIELDS:
        parsed[field] = _parse_integer(values[field], field)
    expected = {
        "mode": expected_mode,
        "case_id": planned["case_id"],
        "kernel_identity": planned["kernel_identity"],
        "grid_x": planned["grid_x"],
        "grid_y": planned["grid_y"],
        "grid_z": planned["grid_z"],
        "workgroups": planned["workgroups"],
        "block": planned["block"],
        "dynamic_shared_bytes": planned["dynamic_shared_bytes"],
        "query_heads": planned["query_heads"],
        "q_tiles": planned["q_tiles"],
        "batch": planned["batch"],
        "q_tile_tokens": planned["q_tile_tokens"],
        "kv_tile_tokens": planned["kv_tile_tokens"],
        "head_dim": planned["head_dim"],
        "consumer_reuse_group_size": planned[
            "consumer_reuse_group_size"],
        "page_owner_count": planned["page_owner_count"],
        "pool_pages_per_operand": planned["pool_pages_per_operand"],
        "combined_kv_working_set_bytes": planned[
            "combined_kv_working_set_bytes"],
        "causal_kv_steps_rule": planned["causal_kv_steps_rule"],
        "logical_k_read_bytes": planned["logical_k_read_bytes"],
        "logical_v_read_bytes": planned["logical_v_read_bytes"],
        "sink_write_bytes": planned["sink_write_bytes"],
        "event_trials": expected_event_trials,
    }
    for field, expected_value in expected.items():
        if parsed[field] != expected_value:
            raise ValueError(
                f"{path} field {field!r} differs from the frozen plan")
    if parsed["first_sink"] == 0 or parsed["last_sink"] == 0:
        raise ValueError("tile64 sink values must be non-zero")
    event_values = (
        parsed["event_ms_min"], parsed["event_ms_median"],
        parsed["event_ms_max"])
    if expected_mode == "event":
        if event_values != tuple(sorted(event_values)):
            raise ValueError("event durations must satisfy min <= median <= max")
    elif any(value != 0 for value in event_values):
        raise ValueError("profile stdout must not expose instrumented duration")
    return parsed


def normalize_profile_stdout(raw_path, output_path, planned):
    if output_path.exists():
        raise FileExistsError(f"normalized stdout already exists: {output_path}")
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    prefixes = ("metadata,", "sample_header,", "sample,")
    positions = []
    selected = []
    for prefix in prefixes:
        matches = [
            (index, line) for index, line in enumerate(lines)
            if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"{raw_path} must contain exactly one {prefix[:-1]!r} row")
        positions.append(matches[0][0])
        selected.append(matches[0][1])
    if positions != list(range(positions[0], positions[0] + 3)):
        raise ValueError("profile CSV rows must be consecutive and ordered")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return parse_stdout(output_path, planned, "profile")


def _mcprofiler_number(value, label):
    if isinstance(value, bool):
        raise TypeError(f"{label} cannot be boolean")
    if isinstance(value, (int, float)):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip().replace(",", "")
        for suffix in ("%", "byte", "bytes"):
            if text.endswith(suffix):
                text = text[:-len(suffix)]
        text = text.split("(", 1)[0].strip()
        try:
            result = float(text)
        except ValueError as error:
            raise ValueError(f"{label} is not numeric: {value!r}") from error
    else:
        raise TypeError(f"{label} has an invalid numeric type")
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _profiler_scalars(payload, path):
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"{path} must contain a non-empty JSON object")
    scalars = {}
    for section, entries in payload.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError(f"{path} section {section!r} has a non-object")
            name = entry["name"]
            value = entry["value"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"{path} contains an invalid metric name")
            if not isinstance(value, dict):
                scalars[name] = value
    internal = payload["Internal"]
    if not isinstance(internal, list):
        raise TypeError(f"{path} Internal section must be a list")
    rooflines = [
        entry for entry in internal
        if isinstance(entry, dict) and entry["name"] == "RoofLine"]
    if len(rooflines) != 1:
        raise ValueError(f"{path} must contain exactly one RoofLine entry")
    value = rooflines[0]["value"]
    if not isinstance(value, dict):
        raise TypeError(f"{path} RoofLine value must be an object")
    data = value["data"]
    processed = value["processed_data"]
    if not isinstance(data, dict) or not isinstance(processed, dict):
        raise TypeError(f"{path} RoofLine payload must contain two objects")
    return scalars, {**processed, **data}


def _required_number(mapping, name, path):
    if name not in mapping:
        raise KeyError(f"{path} is missing mcProfiler metric {name!r}")
    return _mcprofiler_number(mapping[name], f"{path}:{name}")


def parse_profiler_report(path, planned):
    payload = json.loads(path.read_text(encoding="utf-8"))
    scalars, roofline = _profiler_scalars(payload, path)
    metrics = {
        "workgroups": _required_number(scalars, "WORKGROUPS", path),
        "waves": _required_number(scalars, "WAVES", path),
        "global_read_bytes": _required_number(
            scalars, "Global Memory Read bytes", path),
        "global_write_bytes": _required_number(
            scalars, "Global Memory Write bytes", path),
        "dnoc_read_requests": _required_number(
            scalars, "Dnoc Read Req", path),
        "dnoc_write_requests": _required_number(
            scalars, "Dnoc Write Req", path),
        "l2_hit_rate_percent": _required_number(
            scalars, "L2C Hit Rate", path),
        "l2_bytes": _required_number(roofline, "l2c_memacs", path),
        "downstream_bytes": _required_number(roofline, "all_memacs", path),
    }
    if not math.isclose(
            metrics["workgroups"], planned["workgroups"], abs_tol=0.1):
        raise ValueError("mcProfiler WORKGROUPS differs from the frozen grid")
    if not math.isclose(metrics["waves"], planned["waves"], abs_tol=0.1):
        raise ValueError("mcProfiler WAVES differs from the frozen launch")
    for name in (
            "global_read_bytes", "global_write_bytes", "dnoc_read_requests",
            "dnoc_write_requests", "l2_bytes", "downstream_bytes"):
        if metrics[name] < 0:
            raise ValueError(f"mcProfiler {name} cannot be negative")
    if not 0 <= metrics["l2_hit_rate_percent"] <= 100:
        raise ValueError("mcProfiler L2C Hit Rate must be within [0, 100]")
    if (metrics["l2_bytes"] <= 0 or metrics["global_read_bytes"] <= 0
            or metrics["global_write_bytes"] <= 0):
        raise ValueError("tile64 read and write traffic must be positive")
    metrics["l2_contract_relative_error"] = abs(
        metrics["l2_bytes"] - planned["expected_l2_bytes"]
    ) / planned["expected_l2_bytes"]
    metrics["l2_contract_pass"] = (
        metrics["l2_contract_relative_error"]
        <= L2_CONTRACT_RELATIVE_ERROR_LIMIT)
    if not math.isclose(
            metrics["downstream_bytes"],
            metrics["global_read_bytes"] + metrics["global_write_bytes"],
            rel_tol=1e-9, abs_tol=256.0):
        raise ValueError("mcProfiler all_memacs disagrees with global bytes")
    for byte_name, request_name, direction in (
            ("global_read_bytes", "dnoc_read_requests", "read"),
            ("global_write_bytes", "dnoc_write_requests", "write")):
        byte_count = metrics[byte_name]
        requests = metrics[request_name]
        if requests <= 0:
            raise ValueError(f"mcProfiler {direction} requests must be positive")
        if byte_count / requests > 256.0 + 1e-9:
            raise ValueError(
                f"mcProfiler exceeds two 128-byte lines per {direction} request")
    return metrics


def _safe_artifact_path(root, relative, label):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} must be a non-empty relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} must stay under the artifact root")
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def read_provenance(artifact_root, plan_path):
    path = artifact_root / "provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != PROVENANCE_KEYS:
        raise ValueError("tile64 provenance schema is invalid")
    if (payload["schema_version"] != SCHEMA_VERSION
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != MECHANISM
            or payload["baseline_commit"] != BASELINE_COMMIT
            or payload["production_or_evaluation_duration_consumed"] is not False):
        raise ValueError("tile64 provenance identity is invalid")
    if payload["plan_sha256"] != _sha256(plan_path):
        raise ValueError("tile64 plan SHA-256 does not match provenance")
    checks = (
        ("plan_path", "plan_sha256", "plan snapshot"),
        ("source_path", "source_sha256", "source snapshot"),
        ("runner_path", "runner_sha256", "runner snapshot"),
        ("binary_path", "binary_sha256", "compiled binary"),
        (
            "device_preflight_path", "device_preflight_sha256",
            "device preflight",
        ),
    )
    for path_key, hash_key, label in checks:
        artifact = _safe_artifact_path(artifact_root, payload[path_key], label)
        if _sha256(artifact) != payload[hash_key]:
            raise ValueError(f"tile64 {label} SHA-256 mismatch")
    if payload["source_sha256"] != SOURCE_SHA256:
        raise ValueError("tile64 provenance source is not frozen")
    for field in ("software_revision", "container_image", "physical_device_id"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"tile64 provenance {field} is invalid")
    if payload["raw_stdout_policy"] != (
            "preserve_raw_then_extract_one_consecutive_csv_triplet"):
        raise ValueError("tile64 provenance stdout policy is invalid")
    return payload


def _isolated_report(sample_dir):
    allk = sample_dir / "mcprofiler_out" / "_allk"
    if not allk.is_dir():
        raise FileNotFoundError(f"mcProfiler _allk directory is missing: {allk}")
    candidates = []
    for path in allk.glob("*.txt.json"):
        if path.name == "report.txt.json" or "period" in path.name:
            continue
        prefix, separator, _remainder = path.name.partition("_")
        if separator and prefix.isdigit():
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("sample must contain exactly one isolated report")
    if candidates[0].name.partition("_")[0] != "1":
        raise ValueError("isolated report ordinal must be one")
    return candidates[0]


def _relative_span(minimum, median, maximum):
    if minimum <= 0 or median <= 0 or maximum <= 0:
        raise ValueError("event stability requires positive durations")
    if not minimum <= median <= maximum:
        raise ValueError("event stability requires min <= median <= max")
    return (maximum - minimum) / median


def build_report(plan_path, artifact_root):
    plan = read_plan(plan_path)
    provenance = read_provenance(artifact_root, plan_path)
    samples_root = artifact_root / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"sample directory is missing: {samples_root}")
    expected_ids = {sample["case_id"] for sample in plan["samples"]}
    actual_ids = {path.name for path in samples_root.iterdir() if path.is_dir()}
    if actual_ids != expected_ids:
        raise ValueError("artifact sample set differs from the frozen plan")
    rows = []
    for planned in plan["samples"]:
        sample_dir = samples_root / planned["case_id"]
        event = parse_stdout(sample_dir / "event_stdout.csv", planned, "event")
        profile = parse_stdout(
            sample_dir / "profile_stdout.csv", planned, "profile")
        counters = parse_profiler_report(_isolated_report(sample_dir), planned)
        relative_span = _relative_span(
            event["event_ms_min"], event["event_ms_median"],
            event["event_ms_max"])
        row = {
            "case_id": planned["case_id"],
            "consumer_reuse_group_size": planned[
                "consumer_reuse_group_size"],
            "pool_pages_per_operand": planned["pool_pages_per_operand"],
            "combined_kv_working_set_bytes": planned[
                "combined_kv_working_set_bytes"],
            "event_ms_min": event["event_ms_min"],
            "event_ms_median": event["event_ms_median"],
            "event_ms_max": event["event_ms_max"],
            "event_relative_span": relative_span,
            "event_stability_diagnostic": (
                relative_span <= EVENT_RELATIVE_SPAN_LIMIT),
            "l2_bytes": counters["l2_bytes"],
            "l2_contract_relative_error": counters[
                "l2_contract_relative_error"],
            "l2_contract_pass": counters["l2_contract_pass"],
            "downstream_read_bytes": counters["global_read_bytes"],
            "downstream_write_bytes": counters["global_write_bytes"],
            "dnoc_read_requests": counters["dnoc_read_requests"],
            "dnoc_write_requests": counters["dnoc_write_requests"],
            "l2_hit_rate_percent": counters["l2_hit_rate_percent"],
            "profile_sink_matches_event": (
                profile["first_sink"] == event["first_sink"]
                and profile["last_sink"] == event["last_sink"]),
        }
        if not row["profile_sink_matches_event"]:
            raise ValueError(
                "event and profile executions produced different sinks")
        rows.append(row)
    by_cell = {
        (row["consumer_reuse_group_size"], row["pool_pages_per_operand"]): row
        for row in rows}
    expected_cells = {(1, 224), (4, 224), (1, 288), (4, 288)}
    if set(by_cell) != expected_cells:
        raise ValueError("residency report is missing a frozen 2x2 cell")
    within_pool = {
        str(pool): (
            by_cell[(4, pool)]["downstream_read_bytes"]
            <= by_cell[(1, pool)]["downstream_read_bytes"])
        for pool in (224, 288)}
    capacity_boundary = {
        str(reuse): (
            by_cell[(reuse, 288)]["downstream_read_bytes"]
            > by_cell[(reuse, 224)]["downstream_read_bytes"])
        for reuse in (1, 4)}
    reuse_reduction_percent = {
        str(pool): (
            100.0 * (
                by_cell[(1, pool)]["downstream_read_bytes"]
                - by_cell[(4, pool)]["downstream_read_bytes"])
            / by_cell[(1, pool)]["downstream_read_bytes"])
        for pool in (224, 288)}
    pool_increase_percent = {
        str(reuse): (
            100.0 * (
                by_cell[(reuse, 288)]["downstream_read_bytes"]
                - by_cell[(reuse, 224)]["downstream_read_bytes"])
            / by_cell[(reuse, 224)]["downstream_read_bytes"])
        for reuse in (1, 4)}
    l2_contract_pass = all(row["l2_contract_pass"] for row in rows)
    reuse_rule_pass = all(within_pool.values())
    capacity_boundary_pass = all(capacity_boundary.values())
    eligible = (
        l2_contract_pass and reuse_rule_pass and capacity_boundary_pass)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "baseline_commit": BASELINE_COMMIT,
        "production_or_evaluation_duration_consumed": False,
        "mcprofiler_instrumented_duration_consumed": False,
        "release_gate": {
            "status": (
                "counter_mechanism_pass"
                if eligible else "counter_mechanism_fail"),
            "all_l2_contract_pass": l2_contract_pass,
            "within_pool_reuse_rule": within_pool,
            "capacity_boundary_rule": capacity_boundary,
            "eligible_for_residency_formula_derivation": eligible,
            "duration_used_for_gate": False,
        },
        "effect_sizes": {
            "reuse4_downstream_read_reduction_percent": (
                reuse_reduction_percent),
            "pool224_to_pool288_downstream_read_increase_percent": (
                pool_increase_percent),
        },
        "provenance": provenance,
        "samples": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", type=Path, required=True)
    normalize_parser = subparsers.add_parser("normalize-profile-stdout")
    normalize_parser.add_argument("--plan", type=Path, required=True)
    normalize_parser.add_argument("--case-id", required=True)
    normalize_parser.add_argument("--raw", type=Path, required=True)
    normalize_parser.add_argument("--output", type=Path, required=True)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--plan", type=Path, required=True)
    report_parser.add_argument("--artifact-root", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        write_plan(args.output)
    elif args.command == "normalize-profile-stdout":
        plan = read_plan(args.plan)
        matches = [
            sample for sample in plan["samples"]
            if sample["case_id"] == args.case_id]
        if len(matches) != 1:
            raise ValueError(f"unknown frozen case: {args.case_id!r}")
        normalize_profile_stdout(args.raw, args.output, matches[0])
    elif args.command == "report":
        if args.output.exists():
            raise FileExistsError(f"report path already exists: {args.output}")
        report = build_report(args.plan, args.artifact_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    else:
        raise AssertionError(f"unhandled command: {args.command!r}")


if __name__ == "__main__":
    main()
