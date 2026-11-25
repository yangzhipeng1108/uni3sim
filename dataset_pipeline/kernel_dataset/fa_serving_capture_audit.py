"""Fail-closed candidate audits for frozen FlashAttention serving capture."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

from dataset_pipeline.kernel_dataset.nvidia_a800 import load_nsys_timing_sqlite


_A800_RAW_COUNTER_COLUMNS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "smsp__inst_executed.sum",
    "launch__registers_per_thread",
)
_A800_BYTE_COUNTER_UNIT_SCALES = {
    "byte": 1,
    "Kbyte": 1024,
    "Mbyte": 1024 ** 2,
    "Gbyte": 1024 ** 3,
}
_A800_FIXED_COUNTER_UNITS = {
    "smsp__inst_executed.sum": "inst",
    "launch__registers_per_thread": "register/thread",
}
_A800_REQUIRED_RAW_COLUMNS = frozenset((
    "ID",
    "Kernel Name",
    "Grid Size",
    "Block Size",
    "Device",
    *_A800_RAW_COUNTER_COLUMNS,
))
_ASCEND_OP_BASIC_COLUMNS = frozenset((
    "Op Name", "Op Type", "Block Dim", "Mix Block Dim", "Device Id",
))
_ASCEND_PIPE_COLUMNS = frozenset((
    "block_id", "sub_block_id", "aic_total_cycles", "aiv_total_cycles",
    "aic_mte2_ratio", "aiv_mte2_ratio",
))
_ASCEND_MEMORY_COLUMNS = frozenset((
    "block_id", "sub_block_id", "read_main_memory_datas(KB)",
    "write_main_memory_datas(KB)",
))
_ASCEND_TIMELINE_COLUMNS = frozenset((
    "Device_id", "Task ID", "Stream ID", "Op Name", "OP Type",
    "Task Type", "Block Dim", "Mix Block Dim",
))


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_lower_hex(value, context):
    if (not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None):
        raise ValueError(f"{context} must be a lower SHA256 hex string")
    return value


def _require_nonempty_text(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-empty text")
    return value.strip()


def _normalize_kernel_identity(value):
    return re.sub(r"\s+", " ", _require_nonempty_text(
        value, "A800 NCU kernel name")).strip()


def _parse_triplet(value, context):
    text = _require_nonempty_text(value, context)
    match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if match is None:
        raise ValueError(f"{context} must be a positive integer triplet: {value!r}")
    dimensions = tuple(int(component) for component in match.groups())
    if any(component <= 0 for component in dimensions):
        raise ValueError(f"{context} dimensions must be positive: {value!r}")
    return dimensions


def _parse_nonnegative_number(value, context):
    text = _require_nonempty_text(value, context)
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"{context} must be numeric: {value!r}") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context} must be finite and non-negative: {value!r}")
    return number


def _parse_positive_integer(value, context):
    number = _parse_nonnegative_number(value, context)
    if not number.is_integer() or number <= 0:
        raise ValueError(f"{context} must be a positive integer: {value!r}")
    return int(number)


def _load_csv_rows(path, required_columns, context):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"{context} has no header: {path}")
        has_empty_trailer = fieldnames[-1] == "" and "" not in fieldnames[:-1]
        normalized_fieldnames = fieldnames[:-1] if has_empty_trailer else fieldnames
        if any(not isinstance(field, str) or not field for field in normalized_fieldnames):
            raise ValueError(f"{context} has an empty header: {path}")
        if len(normalized_fieldnames) != len(set(normalized_fieldnames)):
            raise ValueError(f"{context} has duplicate headers: {path}")
        missing = sorted(required_columns - set(normalized_fieldnames))
        if missing:
            raise ValueError(f"{context} is missing required columns: {missing}")
        rows = []
        for row in reader:
            if has_empty_trailer:
                if row[""] not in (None, ""):
                    raise ValueError(f"{context} has data in its empty trailing column")
                del row[""]
            rows.append(row)
    if not rows:
        raise ValueError(f"{context} has no data rows: {path}")
    for row in rows:
        if None in row:
            raise ValueError(f"{context} row has excess fields: {path}")
        missing_values = [
            field for field in normalized_fieldnames if row[field] is None]
        if missing_values:
            raise ValueError(
                f"{context} row is missing values for {missing_values}: {path}")
    return rows


def _load_a800_ncu_raw(raw_csv_path):
    path = Path(raw_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"A800 NCU raw CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"A800 NCU raw CSV has no header: {path}")
        if any(not isinstance(field, str) or not field for field in fieldnames):
            raise ValueError(f"A800 NCU raw CSV has an empty header: {path}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"A800 NCU raw CSV has duplicate headers: {path}")
        missing = sorted(_A800_REQUIRED_RAW_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"A800 NCU raw CSV is missing required columns: {missing}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"A800 NCU raw CSV lacks a unit row and kernel rows: {path}")
    unit_row = rows[0]
    if unit_row["ID"] != "" or unit_row["Kernel Name"] != "":
        raise ValueError("A800 NCU raw CSV has no leading unit row")
    for name in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
        if unit_row[name] not in _A800_BYTE_COUNTER_UNIT_SCALES:
            raise ValueError(
                "A800 NCU raw CSV counter unit is not a binary byte unit: "
                f"metric={name!r}, actual={unit_row[name]!r}")
    for name, unit in _A800_FIXED_COUNTER_UNITS.items():
        if unit_row[name] != unit:
            raise ValueError(
                "A800 NCU raw CSV counter unit changed: "
                f"metric={name!r}, expected={unit!r}, actual={unit_row[name]!r}")

    records = []
    for expected_ordinal, row in enumerate(rows[1:]):
        if None in row:
            raise ValueError(f"A800 NCU raw CSV row has excess fields: {path}")
        missing_values = [field for field in fieldnames if row[field] is None]
        if missing_values:
            raise ValueError(
                f"A800 NCU raw CSV row is missing values for {missing_values}: {path}")
        ordinal_text = row["ID"]
        if ordinal_text != str(expected_ordinal):
            raise ValueError(
                "A800 NCU raw CSV kernel ordinal is not contiguous: "
                f"expected={expected_ordinal}, actual={ordinal_text!r}")
        records.append({
            "ordinal": expected_ordinal,
            "identity": _normalize_kernel_identity(row["Kernel Name"]),
            "grid": _parse_triplet(row["Grid Size"], "A800 NCU grid"),
            "block": _parse_triplet(row["Block Size"], "A800 NCU block"),
            "device": _parse_nonnegative_number(row["Device"], "A800 NCU device"),
            "counters": {
                name: (
                    _parse_nonnegative_number(row[name], f"A800 NCU {name}")
                    * _A800_BYTE_COUNTER_UNIT_SCALES[unit_row[name]]
                    if name in ("dram__bytes_read.sum", "dram__bytes_write.sum")
                    else _parse_nonnegative_number(row[name], f"A800 NCU {name}"))
                for name in _A800_RAW_COUNTER_COLUMNS
            },
        })
    if not records:
        raise ValueError(f"A800 NCU raw CSV has no kernel records: {path}")
    devices = {record["device"] for record in records}
    if len(devices) != 1:
        raise ValueError(f"A800 NCU raw CSV spans devices: {sorted(devices)}")
    return records


def _mean_nonempty_msprof_values(rows, column, context):
    values = []
    for row in rows:
        value = row[column].strip()
        if value in ("", "NA", "N/A"):
            continue
        values.append(_parse_nonnegative_number(value, f"{context} {column}"))
    if not values:
        raise ValueError(f"{context} has no measured values for {column!r}")
    return math.fsum(values) / len(values)


def _sum_nonempty_msprof_values(rows, column, context):
    values = []
    for row in rows:
        value = row[column].strip()
        if value in ("", "NA", "N/A"):
            continue
        values.append(_parse_nonnegative_number(value, f"{context} {column}"))
    if not values:
        raise ValueError(f"{context} has no measured values for {column!r}")
    return math.fsum(values)


def _msprof_execution_keys(rows, context):
    keys = []
    for row in rows:
        block_id = _require_nonempty_text(row["block_id"], f"{context} block_id")
        sub_block_id = _require_nonempty_text(
            row["sub_block_id"], f"{context} sub_block_id")
        keys.append((block_id, sub_block_id))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} has duplicate execution keys")
    return frozenset(keys)


def _load_ascend_counter_sample(op_basic_path, pipe_path, memory_path):
    basic_rows = _load_csv_rows(
        op_basic_path, _ASCEND_OP_BASIC_COLUMNS, "Ascend msprof OpBasicInfo")
    if len(basic_rows) != 1:
        raise ValueError("Ascend msprof OpBasicInfo must contain exactly one kernel")
    basic = basic_rows[0]
    program = _require_nonempty_text(basic["Op Name"], "Ascend msprof program")
    if not program.startswith("FusedInferAttentionScore_"):
        raise ValueError(f"unexpected Ascend counter program: {program!r}")
    if _require_nonempty_text(basic["Op Type"], "Ascend msprof op type") != "mix":
        raise ValueError("Ascend counter program must be a mixed AIC/AIV kernel")
    block_dim = _parse_positive_integer(
        basic["Block Dim"], "Ascend msprof Block Dim")
    mix_block_dim = _parse_positive_integer(
        basic["Mix Block Dim"], "Ascend msprof Mix Block Dim")
    device_id = _parse_positive_integer(
        str(_parse_nonnegative_number(
            basic["Device Id"], "Ascend msprof Device Id") + 1),
        "Ascend msprof device") - 1

    pipe_rows = _load_csv_rows(
        pipe_path, _ASCEND_PIPE_COLUMNS, "Ascend msprof PipeUtilization")
    memory_rows = _load_csv_rows(
        memory_path, _ASCEND_MEMORY_COLUMNS, "Ascend msprof Memory")
    if _msprof_execution_keys(pipe_rows, "Ascend msprof PipeUtilization") != (
            _msprof_execution_keys(memory_rows, "Ascend msprof Memory")):
        raise ValueError("Ascend msprof PipeUtilization and Memory execution keys differ")
    return {
        "program": program,
        "block_dim": block_dim,
        "mix_block_dim": mix_block_dim,
        "device_id": device_id,
        "counters": {
            "aic_total_cycles_mean": _mean_nonempty_msprof_values(
                pipe_rows, "aic_total_cycles", "Ascend msprof PipeUtilization"),
            "aiv_total_cycles_mean": _mean_nonempty_msprof_values(
                pipe_rows, "aiv_total_cycles", "Ascend msprof PipeUtilization"),
            "aic_mte2_ratio_mean": _mean_nonempty_msprof_values(
                pipe_rows, "aic_mte2_ratio", "Ascend msprof PipeUtilization"),
            "aiv_mte2_ratio_mean": _mean_nonempty_msprof_values(
                pipe_rows, "aiv_mte2_ratio", "Ascend msprof PipeUtilization"),
            "main_memory_read_kb_sum": _sum_nonempty_msprof_values(
                memory_rows, "read_main_memory_datas(KB)", "Ascend msprof Memory"),
            "main_memory_write_kb_sum": _sum_nonempty_msprof_values(
                memory_rows, "write_main_memory_datas(KB)", "Ascend msprof Memory"),
        },
    }


def _load_ascend_timeline_launch(op_summary_path, program_hash_path,
                                 expected_task_id):
    timeline_rows = _load_csv_rows(
        op_summary_path, _ASCEND_TIMELINE_COLUMNS, "Ascend Timeline op_summary")
    matches = []
    for row in timeline_rows:
        if row["Op Name"].strip() != "FusedInferAttentionScore":
            continue
        _parse_positive_integer(row["Task ID"], "Ascend Timeline task ID")
        matches.append(row)
    if expected_task_id is None:
        if not matches:
            raise ValueError("Ascend Timeline has no FusedInferAttentionScore task")
        expected_task_id = max(
            _parse_positive_integer(row["Task ID"], "Ascend Timeline task ID")
            for row in matches)
        matches = [
            row for row in matches
            if _parse_positive_integer(row["Task ID"], "Ascend Timeline task ID")
            == expected_task_id]
    else:
        expected_task_id = _parse_positive_integer(
            str(expected_task_id), "expected Ascend Timeline task ID")
        matches = [
            row for row in matches
            if _parse_positive_integer(row["Task ID"], "Ascend Timeline task ID")
            == expected_task_id]
    if len(matches) != 1:
        raise ValueError(
            "Ascend Timeline must have exactly one selected FusedInferAttentionScore "
            f"task: expected_task_id={expected_task_id}, actual={len(matches)}")
    row = matches[0]
    if _require_nonempty_text(row["OP Type"], "Ascend Timeline OP Type") != "FusedInferAttentionScore":
        raise ValueError("Ascend Timeline selected task has unexpected OP Type")
    if _require_nonempty_text(row["Task Type"], "Ascend Timeline task type") != "MIX_AIC":
        raise ValueError("Ascend Timeline selected task is not a mixed AIC task")
    launch = {
        "task_id": expected_task_id,
        "stream_id": _parse_positive_integer(row["Stream ID"], "Ascend Timeline stream ID"),
        "block_dim": _parse_positive_integer(row["Block Dim"], "Ascend Timeline Block Dim"),
        "mix_block_dim": _parse_positive_integer(
            row["Mix Block Dim"], "Ascend Timeline Mix Block Dim"),
        "device_id": _parse_positive_integer(
            str(_parse_nonnegative_number(row["Device_id"], "Ascend Timeline device") + 1),
            "Ascend Timeline device") - 1,
    }
    program_hash_path = Path(program_hash_path)
    if not program_hash_path.is_file():
        raise FileNotFoundError(
            f"Ascend Timeline program hash data does not exist: {program_hash_path}")
    return launch, program_hash_path.read_text(encoding="utf-8")


def build_ascend_counter_candidate(*, case_id, workload_digest,
                                   source_snapshot_sha256,
                                   counter_op_basic_paths, counter_pipe_paths,
                                   counter_memory_paths,
                                   timeline_op_summary_paths,
                                   timeline_program_hash_paths):
    """Build an Ascend counter candidate without consuming profiler timing fields."""
    case_id = _require_nonempty_text(case_id, "case_id")
    workload_digest = _require_lower_hex(workload_digest, "workload_digest")
    source_snapshot_sha256 = _require_lower_hex(
        source_snapshot_sha256, "source_snapshot_sha256")
    path_groups = (
        list(counter_op_basic_paths), list(counter_pipe_paths),
        list(counter_memory_paths), list(timeline_op_summary_paths),
        list(timeline_program_hash_paths),
    )
    if any(len(paths) != 3 for paths in path_groups):
        raise ValueError("Ascend counter candidate requires exactly three raw windows")
    counter_samples = []
    timing_joins = []
    program = None
    for index, paths in enumerate(zip(*path_groups, strict=True), start=1):
        basic_path, pipe_path, memory_path, summary_path, hash_path = paths
        sample = _load_ascend_counter_sample(basic_path, pipe_path, memory_path)
        launch, hash_text = _load_ascend_timeline_launch(
            summary_path, hash_path, expected_task_id=None)
        if sample["device_id"] != launch["device_id"]:
            raise ValueError(f"Ascend counter/Timeline device mismatch in window {index}")
        if (sample["block_dim"], sample["mix_block_dim"]) != (
                launch["block_dim"], launch["mix_block_dim"]):
            raise ValueError(f"Ascend counter/Timeline launch mismatch in window {index}")
        program_stem = sample["program"].removesuffix("_mix_aic")
        if program_stem == sample["program"] or program_stem not in hash_text:
            raise ValueError(
                "Ascend Timeline source hash does not prove the counter program: "
                f"window={index}, program={sample['program']!r}")
        if program is None:
            program = sample["program"]
        elif sample["program"] != program:
            raise ValueError("Ascend counter repetitions have different program identities")
        counter_samples.append({
            "counter_window": index,
            "op_basic_path": str(basic_path),
            "op_basic_sha256": _file_sha256(Path(basic_path)),
            "pipe_path": str(pipe_path),
            "pipe_sha256": _file_sha256(Path(pipe_path)),
            "memory_path": str(memory_path),
            "memory_sha256": _file_sha256(Path(memory_path)),
            "counters": sample["counters"],
        })
        timing_joins.append({
            "timing_window": index,
            "op_summary_path": str(summary_path),
            "op_summary_sha256": _file_sha256(Path(summary_path)),
            "program_hash_path": str(hash_path),
            "program_hash_sha256": _file_sha256(Path(hash_path)),
            "task_id": launch["task_id"],
            "stream_id": launch["stream_id"],
            "device_id": launch["device_id"],
            "block_dim": launch["block_dim"],
            "mix_block_dim": launch["mix_block_dim"],
        })
    if program is None:
        raise AssertionError("Ascend counter candidate lost all counter samples")
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_ascend_910b_counter",
        "hardware_id": "ascend_910b",
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot_sha256": source_snapshot_sha256,
        "counter_contract": {
            "raw_tool": "msprof op",
            "duration_consumed": False,
            "retained_metrics": [
                "aic_total_cycles_mean", "aiv_total_cycles_mean",
                "aic_mte2_ratio_mean", "aiv_mte2_ratio_mean",
                "main_memory_read_kb_sum", "main_memory_write_kb_sum",
            ],
            "excluded_metric_classes": [
                "task_duration", "aic_time", "aiv_time", "frequency",
                "bandwidth_rate",
            ],
        },
        "kernel": {
            "ordinal": 0,
            "program": program,
            "counter_samples": counter_samples,
        },
        "timing_launch_join": {
            "join_key": "workload digest + source snapshot + program + device + block_dim + mix_block_dim + final Timeline task ordinal",
            "windows": timing_joins,
        },
    }


def _load_json_mapping(path, context):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{context} does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON mapping: {path}")
    return value


def build_canary_closure(*, hardware_id, case_id, workload_digest,
                         timing_audit_path, counter_candidate_path):
    """Close one canary only when its raw timing and counters agree exactly."""
    if hardware_id not in ("nvidia_a800", "ascend_910b"):
        raise ValueError(f"unsupported canary closure hardware: {hardware_id!r}")
    case_id = _require_nonempty_text(case_id, "case_id")
    workload_digest = _require_lower_hex(workload_digest, "workload_digest")
    timing_path = Path(timing_audit_path)
    candidate_path = Path(counter_candidate_path)
    timing = _load_json_mapping(timing_path, "canary timing audit")
    candidate = _load_json_mapping(candidate_path, "canary counter candidate")
    if timing["status"] != "pass" or timing["aggregation"]["pass"] is not True:
        raise ValueError("canary timing audit did not pass its stability gate")
    if timing["hardware"]["hardware_id"] != hardware_id:
        raise ValueError("canary timing audit hardware does not match")
    if candidate["hardware_id"] != hardware_id:
        raise ValueError("canary counter candidate hardware does not match")
    for field, expected in (("case_id", case_id), ("workload_digest", workload_digest)):
        if timing[field] != expected or candidate[field] != expected:
            raise ValueError(f"canary {field} does not match the requested closure")
    timing_source = timing["source_snapshot"]["sha256"]
    candidate_source = candidate["source_snapshot_sha256"]
    if timing_source != candidate_source:
        raise ValueError("canary timing and counter source snapshots differ")
    if hardware_id == "nvidia_a800":
        join_windows = candidate["timing_launch_join"]["windows"]
        expected_kernel_count = len(candidate["kernels"])
        if expected_kernel_count <= 0 or any(
                window["launch_count"] != expected_kernel_count
                for window in join_windows):
            raise ValueError("A800 canary counter launch join is incomplete")
    elif hardware_id == "ascend_910b":
        join_windows = candidate["timing_launch_join"]["windows"]
        if any(window["task_id"] != 42 or window["stream_id"] != 2
               for window in join_windows):
            raise ValueError("Ascend canary counter launch join is incomplete")
    else:
        raise AssertionError(f"unhandled closure hardware: {hardware_id!r}")
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_canary_closure",
        "status": "pass",
        "hardware_id": hardware_id,
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot_sha256": timing_source,
        "timing": {
            "path": str(timing_path),
            "sha256": _file_sha256(timing_path),
            "stability_pass": True,
        },
        "counter_candidate": {
            "path": str(candidate_path),
            "sha256": _file_sha256(candidate_path),
            "duration_consumed": candidate["counter_contract"]["duration_consumed"],
        },
        "gate": {
            "runner_validation": True,
            "golden_timing": True,
            "non_duration_counter": True,
            "ordinal_launch_join": True,
            "provenance": True,
            "candidate_aggregation": True,
            "stability": True,
        },
    }


def build_a800_timing_audit(*, case_id, workload_digest,
                            nsys_sqlite_paths, source_snapshot_sha256):
    """Summarize three complete A800 NVTX kernel windows without counters."""
    case_id = _require_nonempty_text(case_id, "case_id")
    workload_digest = _require_lower_hex(workload_digest, "workload_digest")
    source_snapshot_sha256 = _require_lower_hex(
        source_snapshot_sha256, "source_snapshot_sha256")
    paths = [Path(path) for path in nsys_sqlite_paths]
    if len(paths) != 3:
        raise ValueError("A800 timing audit requires exactly three NSYS windows")
    if len(set(paths)) != len(paths):
        raise ValueError("A800 timing audit NSYS paths must be unique")

    expected_sequence = None
    windows = []
    totals = []
    for ordinal, path in enumerate(paths, start=1):
        launches = load_nsys_timing_sqlite(path, case_id)
        sequence = tuple(
            (launch["identity"], launch["grid"], launch["block"])
            for launch in launches)
        if expected_sequence is None:
            expected_sequence = sequence
        elif sequence != expected_sequence:
            raise ValueError(
                "A800 NSYS timing windows have different launch sequences: "
                f"window={ordinal}")
        total_duration_ns = math.fsum(
            launch["duration_ns"] for launch in launches)
        if total_duration_ns <= 0:
            raise ValueError(f"A800 NSYS timing window has no positive duration: {path}")
        totals.append(total_duration_ns)
        windows.append({
            "window": ordinal,
            "sqlite_path": str(path),
            "sqlite_sha256": _file_sha256(path),
            "kernel_count": len(launches),
            "kernel_sum_ns": total_duration_ns,
            "launches": [
                {
                    "ordinal": index,
                    "identity": launch["identity"],
                    "grid": list(launch["grid"]),
                    "block": list(launch["block"]),
                    "duration_ns": launch["duration_ns"],
                }
                for index, launch in enumerate(launches)
            ],
        })
    median_duration_ns = sorted(totals)[1]
    relative_spread = (max(totals) - min(totals)) / max(totals)
    stable = relative_spread <= 0.05
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_a800_nsys_timing",
        "status": "pass" if stable else "fail",
        "hardware": {"hardware_id": "nvidia_a800"},
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot": {"sha256": source_snapshot_sha256},
        "windows": windows,
        "aggregation": {
            "selection": "median kernel sum of three independent complete processes",
            "median_kernel_sum_ns": median_duration_ns,
            "max_relative_spread": relative_spread,
            "max_relative_spread_gate": 0.05,
            "pass": stable,
        },
    }


def build_a800_counter_candidate(*, case_id, workload_digest, raw_csv_path,
                                 nsys_sqlite_paths, source_snapshot_sha256):
    """Build one duration-free NCU candidate after strict NSYS ordinal joins."""
    case_id = _require_nonempty_text(case_id, "case_id")
    workload_digest = _require_lower_hex(workload_digest, "workload_digest")
    source_snapshot_sha256 = _require_lower_hex(
        source_snapshot_sha256, "source_snapshot_sha256")
    raw_path = Path(raw_csv_path)
    nsys_paths = [Path(path) for path in nsys_sqlite_paths]
    if len(nsys_paths) != 3:
        raise ValueError("A800 counter candidate requires exactly three NSYS windows")
    if len(set(nsys_paths)) != len(nsys_paths):
        raise ValueError("A800 NSYS window paths must be unique")

    ncu_records = _load_a800_ncu_raw(raw_path)
    ncu_sequence = tuple(
        (record["identity"], record["grid"], record["block"])
        for record in ncu_records)
    timing_windows = []
    for ordinal, nsys_path in enumerate(nsys_paths, start=1):
        launches = load_nsys_timing_sqlite(nsys_path, case_id)
        nsys_sequence = tuple(
            (launch["identity"], launch["grid"], launch["block"])
            for launch in launches)
        if nsys_sequence != ncu_sequence:
            raise ValueError(
                "A800 NCU/NSYS launch sequence mismatch: "
                f"window={ordinal}, ncu={ncu_sequence!r}, nsys={nsys_sequence!r}")
        timing_windows.append({
            "window": ordinal,
            "sqlite_path": str(nsys_path),
            "sqlite_sha256": _file_sha256(nsys_path),
            "launch_count": len(nsys_sequence),
        })
    return {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_a800_counter",
        "hardware_id": "nvidia_a800",
        "case_id": case_id,
        "workload_digest": workload_digest,
        "source_snapshot_sha256": source_snapshot_sha256,
        "counter_contract": {
            "raw_tool": "Nsight Compute raw page",
            "duration_consumed": False,
            "retained_metric_units": {
                "dram__bytes_read.sum": "byte",
                "dram__bytes_write.sum": "byte",
                "smsp__inst_executed.sum": "inst",
                "launch__registers_per_thread": "register/thread",
            },
            "retained_metrics": list(_A800_RAW_COUNTER_COLUMNS),
            "excluded_metric_classes": [
                "duration", "elapsed_cycles", "frequency", "throughput",
                "per_second", "active_cycles",
            ],
        },
        "raw_counter": {
            "path": str(raw_path),
            "sha256": _file_sha256(raw_path),
            "kernel_count": len(ncu_records),
        },
        "timing_launch_join": {
            "join_key": "ordinal + full normalized symbol + grid + block",
            "windows": timing_windows,
        },
        "kernels": [
            {
                "ordinal": record["ordinal"],
                "identity": record["identity"],
                "grid": list(record["grid"]),
                "block": list(record["block"]),
                "counters": record["counters"],
            }
            for record in ncu_records
        ],
    }


def write_json_new(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
