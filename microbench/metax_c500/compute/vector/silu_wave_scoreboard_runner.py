#!/usr/bin/env python3
"""Freeze and reduce the C500 active-wave SiLU-like scoreboard campaign."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


SCHEMA_VERSION = 1
STATIC_AUDIT_SCHEMA_VERSION = 1
HARDWARE_ID = "metax_c500"
SOFTWARE_REVISION = "maca_3.7.2"
BASELINE_COMMIT = "8778cc80b2e589063180deb5f18e74ba850a235c"
MECHANISM = "active_wave_silu_like_dependency_scoreboard"
AP_COUNT = 104
WAVE_SIZE = 64
DAG_ROUNDS = 40
ATTACH_WAIT_SECONDS = 5
FULL_TRIALS = (1, 2, 3)
TRAINING_TRIALS = (1, 2)
VALIDATION_TRIALS = (3,)
MAX_VALIDATION_MAPE_PERCENT = 20.0
MAX_VALIDATION_MAX_APE_PERCENT = 40.0
MAX_COUNTER_MAPE_PERCENT = 20.0
MAX_COUNTER_MAX_APE_PERCENT = 40.0
CONFIGURATIONS = (
    {
        "configuration": "register_q1",
        "kernel_symbol": "silu_wave_reg_q1",
        "cache_state": "register_only",
        "grid": 104,
        "block": 64,
        "waves_per_cta": 1,
        "active_waves_per_ap": 1,
        "working_set_bytes": 0,
    },
    {
        "configuration": "register_q8",
        "kernel_symbol": "silu_wave_reg_q8",
        "cache_state": "register_only",
        "grid": 104,
        "block": 512,
        "waves_per_cta": 8,
        "active_waves_per_ap": 8,
        "working_set_bytes": 0,
    },
    {
        "configuration": "register_q32",
        "kernel_symbol": "silu_wave_reg_q32",
        "cache_state": "register_only",
        "grid": 416,
        "block": 512,
        "waves_per_cta": 8,
        "active_waves_per_ap": 32,
        "working_set_bytes": 0,
    },
    {
        "configuration": "l2_q32",
        "kernel_symbol": "silu_wave_l2_q32",
        "cache_state": "l2_resident_4mib",
        "grid": 416,
        "block": 512,
        "waves_per_cta": 8,
        "active_waves_per_ap": 32,
        "working_set_bytes": 4 * 1024 * 1024,
    },
    {
        "configuration": "streaming_q32",
        "kernel_symbol": "silu_wave_dram_q32",
        "cache_state": "streaming_64mib",
        "grid": 416,
        "block": 512,
        "waves_per_cta": 8,
        "active_waves_per_ap": 32,
        "working_set_bytes": 64 * 1024 * 1024,
    },
)
CONFIGURATION_BY_NAME = {
    row["configuration"]: row for row in CONFIGURATIONS}
STATIC_AUDIT_KEYS = {
    "schema_version", "status", "hardware_id", "mechanism",
    "software_revision", "baseline_commit",
    "production_or_evaluation_duration_consumed", "source", "binary",
    "disassembly", "kernels",
}
RUN_REPORT_KEYS = {
    "schema_version", "hardware_id", "mechanism", "software_revision",
    "baseline_commit", "production_or_evaluation_duration_consumed",
    "instrumented_profiler_cycles_or_duration_consumed",
    "trial", "physical_device_id", "reported_clock_khz", "plan",
    "static_audit", "source_sha256", "binary_sha256", "samples",
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path, label):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object")
    return payload


def _canonical_nonnegative_integer(text, label):
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise ValueError(f"{label} must be a canonical non-negative integer")
    return int(text)


def _positive_finite(text, label):
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def build_plan(trial):
    if trial not in FULL_TRIALS:
        raise ValueError("scoreboard trial must be 1, 2, or 3")
    order = list(CONFIGURATIONS)
    random.Random(500729 + 7919 * trial).shuffle(order)
    samples = []
    for position, configuration in enumerate(order):
        samples.append({
            "sample_id": (
                f"t{trial}_p{position}_{configuration['configuration']}"),
            "trial": trial,
            "order_position": position,
            **configuration,
            "dag_rounds": DAG_ROUNDS,
            "expected_workgroups": configuration["grid"],
            "expected_live_wave_count": (
                configuration["grid"] * configuration["waves_per_cta"]),
            "profiler_attach_wait_seconds": ATTACH_WAIT_SECONDS,
            "timing_binary_arguments_with_device_placeholder": [
                "--timing", "--device-id", "DEVICE_ID",
                "--trial", str(trial),
                "--order-position", str(position),
                "--configuration", configuration["configuration"],
            ],
            "profile_binary_arguments_with_device_placeholder": [
                "--profile-once", "--device-id", "DEVICE_ID",
                "--trial", str(trial),
                "--order-position", str(position),
                "--configuration", configuration["configuration"],
                "--attach-wait-seconds", str(ATTACH_WAIT_SECONDS),
            ],
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "software_revision": SOFTWARE_REVISION,
        "baseline_commit": BASELINE_COMMIT,
        "production_or_evaluation_duration_consumed": False,
        "instrumented_profiler_cycles_or_duration_consumed": False,
        "trial": trial,
        "randomization_seed": 500729 + 7919 * trial,
        "samples": samples,
    }


def _load_plan(path, expected_trial):
    payload = _read_json_object(path, "scoreboard plan")
    if payload != build_plan(expected_trial):
        raise ValueError("scoreboard plan is not the frozen randomized plan")
    return payload


def _function_instructions(path, symbol_fragment):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = re.compile(r"^([0-9a-f]+) <(.+)>:$")
    starts = []
    for index, line in enumerate(lines):
        match = header.match(line)
        if match is not None and symbol_fragment in match.group(2):
            starts.append(index)
    if len(starts) != 1:
        raise ValueError(
            f"disassembly must contain one {symbol_fragment!r} function")
    start = starts[0]
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if header.match(lines[index]) is not None:
            end = index
            break
    instruction_with_bytes = re.compile(
        r"^\s*([0-9a-f]+):(?: [0-9a-f]{2}){8}\s+"
        r"([A-Za-z0-9_]+)(?:\s+(.*))?$")
    instruction_without_bytes = re.compile(
        r"^\s*([0-9a-f]+):\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"(?:\s+(.*))?$")
    instructions = []
    for line in lines[start + 1:end]:
        match = instruction_with_bytes.match(line)
        if match is None:
            match = instruction_without_bytes.match(line)
        if match is not None:
            instructions.append({
                "address": int(match.group(1), 16),
                "opcode": match.group(2),
                "operands": "" if match.group(3) is None else match.group(3),
            })
    if not instructions:
        raise ValueError(f"{symbol_fragment!r} has no decoded instructions")
    return tuple(instructions)


def _timer_region(instructions, configuration):
    timers = tuple(
        index for index, instruction in enumerate(instructions)
        if instruction["opcode"] == "get_time")
    if len(timers) != 2:
        raise ValueError(
            f"{configuration} must contain exactly two clock64 get_time ops")
    region = instructions[timers[0] + 1:timers[1]]
    opcodes = tuple(instruction["opcode"] for instruction in region)
    if "exp_f32" not in opcodes or "rcp_f32" not in opcodes:
        raise ValueError(f"{configuration} is missing the locked exp/rcp DAG")
    if opcodes.index("exp_f32") >= opcodes.index("rcp_f32"):
        raise ValueError(f"{configuration} exp/rcp dependency order changed")
    global_loads = sum(opcode.startswith("ldg_") for opcode in opcodes)
    global_stores = sum(opcode.startswith("stg_") for opcode in opcodes)
    expected_memory = CONFIGURATION_BY_NAME[configuration][
        "working_set_bytes"] != 0
    if expected_memory and (global_loads < 2 or global_stores < 1):
        raise ValueError(
            f"{configuration} must retain two loads and one store")
    if (expected_memory
            and (sum(opcode.startswith("ldg_")
                     for opcode in opcodes[:opcodes.index("exp_f32")]) < 2
                 or not any(opcode.startswith("stg_")
                            for opcode in opcodes[
                                opcodes.index("rcp_f32") + 1:]))):
        raise ValueError(
            f"{configuration} load/DAG/store dependency order changed")
    if not expected_memory and (global_loads != 0 or global_stores != 0):
        raise ValueError(
            f"{configuration} register timed region touches global memory")
    return {
        "timed_static_instruction_count": len(region),
        "timed_opcode_sequence": list(opcodes),
        "exp_f32_static_count": opcodes.count("exp_f32"),
        "rcp_f32_static_count": opcodes.count("rcp_f32"),
        "global_load_static_count": global_loads,
        "global_store_static_count": global_stores,
    }


def build_static_audit(
        source_path, binary_path, disassembly_path, baseline_commit):
    if baseline_commit != BASELINE_COMMIT:
        raise ValueError("scoreboard static audit baseline commit changed")
    kernels = []
    for configuration in CONFIGURATIONS:
        instructions = _function_instructions(
            disassembly_path, configuration["kernel_symbol"])
        kernels.append({
            "configuration": configuration["configuration"],
            "kernel_symbol": configuration["kernel_symbol"],
            **_timer_region(instructions, configuration["configuration"]),
        })
    return {
        "schema_version": STATIC_AUDIT_SCHEMA_VERSION,
        "status": "pass",
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "software_revision": SOFTWARE_REVISION,
        "baseline_commit": baseline_commit,
        "production_or_evaluation_duration_consumed": False,
        "source": {"path": str(source_path), "sha256": _sha256(source_path)},
        "binary": {"path": str(binary_path), "sha256": _sha256(binary_path)},
        "disassembly": {
            "path": str(disassembly_path),
            "sha256": _sha256(disassembly_path),
        },
        "kernels": kernels,
    }


def _load_static_audit(path):
    payload = _read_json_object(path, "scoreboard static audit")
    if set(payload) != STATIC_AUDIT_KEYS:
        raise ValueError("scoreboard static audit schema is invalid")
    if (payload["schema_version"] != STATIC_AUDIT_SCHEMA_VERSION
            or payload["status"] != "pass"
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != MECHANISM
            or payload["software_revision"] != SOFTWARE_REVISION
            or payload["baseline_commit"] != BASELINE_COMMIT
            or payload["production_or_evaluation_duration_consumed"]
            is not False):
        raise ValueError("scoreboard static audit contract is invalid")
    if ({row["configuration"] for row in payload["kernels"]}
            != set(CONFIGURATION_BY_NAME)):
        raise ValueError("scoreboard static audit kernel set is incomplete")
    for label in ("source", "binary", "disassembly"):
        reference = payload[label]
        if (not isinstance(reference, dict)
                or set(reference) != {"path", "sha256"}
                or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
                is None):
            raise ValueError(
                f"scoreboard static audit {label} reference is invalid")
        artifact_path = Path(reference["path"])
        if _sha256(artifact_path) != reference["sha256"]:
            raise ValueError(
                f"scoreboard static audit {label} hash no longer matches")
    return payload


def parse_sample_stdout(path, plan_row, expected_device_id, expected_mode):
    if expected_mode == "timing":
        expected_row_kind = "timing_sample"
    elif expected_mode == "profile_once":
        expected_row_kind = "profile_sample"
    else:
        raise ValueError("scoreboard stdout mode is unknown")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 2:
        raise ValueError("scoreboard sample stdout must contain two rows")
    metadata = rows[0]
    if len(metadata) != 10 or tuple(metadata[:4]) != (
            "metadata", "2", expected_mode, "MetaX C500"):
        raise ValueError("scoreboard sample metadata schema is invalid")
    metadata_values = tuple(
        _canonical_nonnegative_integer(value, "metadata field")
        for value in metadata[4:])
    expected_metadata = (
        expected_device_id, AP_COUNT, WAVE_SIZE, metadata_values[3],
        plan_row["trial"], len(CONFIGURATIONS),
    )
    if metadata_values != expected_metadata or metadata_values[3] <= 0:
        raise ValueError("scoreboard sample metadata contract changed")
    row = rows[1]
    if len(row) != 17 or row[0] != expected_row_kind:
        raise ValueError("scoreboard sample row schema is invalid")
    integer_values = tuple(
        _canonical_nonnegative_integer(value, "sample field")
        for value in row[1:2] + row[4:14] + row[15:17])
    (order_position, grid, block, waves_per_cta, active_waves_per_ap,
     dag_rounds, working_set_bytes, live_wave_count, median_cycles,
     minimum_cycles, maximum_cycles, registers_per_thread,
     shared_bytes_per_cta) = integer_values
    checksum = _positive_finite(row[14], "sample checksum")
    expected = (
        plan_row["order_position"], plan_row["grid"], plan_row["block"],
        plan_row["waves_per_cta"], plan_row["active_waves_per_ap"],
        plan_row["dag_rounds"], plan_row["working_set_bytes"],
        plan_row["expected_live_wave_count"],
    )
    if ((order_position, grid, block, waves_per_cta, active_waves_per_ap,
            dag_rounds, working_set_bytes, live_wave_count) != expected
            or row[2] != plan_row["configuration"]
            or row[3] != plan_row["cache_state"]):
        raise ValueError("scoreboard sample does not match the frozen plan")
    if (not 0 < minimum_cycles <= median_cycles <= maximum_cycles
            or registers_per_thread <= 0):
        raise ValueError("scoreboard sample cycle/resource values are invalid")
    return {
        "configuration": row[2],
        "cache_state": row[3],
        "order_position": order_position,
        "grid": grid,
        "block": block,
        "active_ap_count": AP_COUNT,
        "active_ap_count_source": (
            "validated_C500_topology_plus_one_or_four_CTAs_per_AP_launch"),
        "active_waves_per_ap": active_waves_per_ap,
        "live_wave_count_clock64": live_wave_count,
        "dag_rounds": dag_rounds,
        "working_set_bytes": working_set_bytes,
        "median_wave_elapsed_xcore_cycles": median_cycles,
        "minimum_wave_elapsed_xcore_cycles": minimum_cycles,
        "maximum_wave_elapsed_xcore_cycles": maximum_cycles,
        "c_round_xcore_cycles": median_cycles / dag_rounds,
        "validation_checksum": checksum,
        "registers_per_thread": registers_per_thread,
        "shared_bytes_per_cta": shared_bytes_per_cta,
        "reported_clock_khz": metadata_values[3],
    }


def _profiler_number(value, label):
    if isinstance(value, bool):
        raise TypeError(f"{label} cannot be boolean")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        match = re.fullmatch(
            r"\s*([0-9][0-9,]*(?:\.[0-9]+)?)"
            r"(?:\([A-Za-z]+\)|[A-Za-z/]+|%)?\s*",
            value)
        if match is None:
            raise ValueError(f"{label} is not a supported profiler number")
        parsed = float(match.group(1).replace(",", ""))
    else:
        raise TypeError(f"{label} has an unsupported profiler value type")
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return parsed


def _metric_map(payload):
    metrics = {}
    for section, entries in payload.items():
        if section == "Internal":
            continue
        if not isinstance(entries, list):
            raise TypeError(f"profiler section {section!r} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {
                    "name", "description", "value"}:
                raise ValueError("profiler scalar entry schema is invalid")
            name = entry["name"]
            if name in metrics:
                raise ValueError(f"profiler metric {name!r} is duplicated")
            metrics[name] = entry["value"]
    return metrics


def parse_counter_report(path, plan_row):
    if (path.name == "report.txt.json" or "period" in path.name
            or re.fullmatch(r"[1-9][0-9]*_.+\.txt\.json", path.name)
            is None):
        raise ValueError(
            "mcProfiler evidence must be a numbered isolated report")
    payload = _read_json_object(path, "mcProfiler isolated report")
    metrics = _metric_map(payload)
    required = (
        "Total Instructions", "WORKGROUPS", "WAVES", "Achieved waves",
        "Dispatched waves", "Global Read Instructions",
        "Global Write Instructions", "Global Memory Read bytes",
        "Global Memory Write bytes",
    )
    if any(name not in metrics for name in required):
        raise ValueError("mcProfiler isolated report lacks a required metric")
    internal = payload["Internal"]
    if not isinstance(internal, list):
        raise TypeError("mcProfiler Internal section must be a list")
    rooflines = tuple(
        entry for entry in internal
        if isinstance(entry, dict) and entry["name"] == "RoofLine")
    if len(rooflines) != 1:
        raise ValueError("mcProfiler report must contain one RoofLine entry")
    processed = rooflines[0]["value"]["processed_data"]
    if not isinstance(processed, dict):
        raise TypeError("mcProfiler RoofLine processed_data is invalid")
    workgroups = _profiler_number(metrics["WORKGROUPS"], "WORKGROUPS")
    waves = _profiler_number(metrics["WAVES"], "WAVES")
    achieved = _profiler_number(metrics["Achieved waves"], "Achieved waves")
    dispatched = _profiler_number(
        metrics["Dispatched waves"], "Dispatched waves")
    expected_waves = plan_row["expected_live_wave_count"]
    if (workgroups != plan_row["expected_workgroups"]
            or waves != expected_waves or achieved != expected_waves
            or dispatched != expected_waves):
        raise ValueError("mcProfiler live workgroup/wave count changed")
    dram_read = _profiler_number(
        metrics["Global Memory Read bytes"], "Global Memory Read bytes")
    dram_write = _profiler_number(
        metrics["Global Memory Write bytes"], "Global Memory Write bytes")
    l2_bytes = _profiler_number(processed["l2c_memacs"], "l2c_memacs")
    all_memacs = _profiler_number(processed["all_memacs"], "all_memacs")
    if abs(all_memacs - (dram_read + dram_write)) > 1.0:
        raise ValueError("mcProfiler DRAM byte counters are inconsistent")
    dynamic_instructions = _profiler_number(
        metrics["Total Instructions"], "Total Instructions")
    if dynamic_instructions <= 0 or l2_bytes <= 0:
        raise ValueError("mcProfiler instruction/L2 counter must be positive")
    return {
        "workgroups": int(workgroups),
        "live_wave_count_mcprofiler": int(waves),
        "achieved_wave_count": int(achieved),
        "dispatched_wave_count": int(dispatched),
        "dynamic_instruction_count": dynamic_instructions,
        "global_read_instruction_count": _profiler_number(
            metrics["Global Read Instructions"],
            "Global Read Instructions"),
        "global_write_instruction_count": _profiler_number(
            metrics["Global Write Instructions"],
            "Global Write Instructions"),
        "l2_request_bytes": l2_bytes,
        "dram_read_bytes": dram_read,
        "dram_write_bytes": dram_write,
        "dram_total_bytes": all_memacs,
    }


def build_run_report(
        plan_path, static_audit_path, timing_stdout_paths,
        profile_stdout_paths, counter_paths,
        trial, physical_device_id):
    if trial not in FULL_TRIALS:
        raise ValueError("scoreboard trial must be 1, 2, or 3")
    if (isinstance(physical_device_id, bool)
            or not isinstance(physical_device_id, int)):
        raise TypeError("scoreboard device ID must be an integer")
    if physical_device_id < 0:
        raise ValueError("scoreboard device ID cannot be negative")
    plan = _load_plan(plan_path, trial)
    static_audit = _load_static_audit(static_audit_path)
    if set(timing_stdout_paths) != set(CONFIGURATION_BY_NAME):
        raise ValueError("scoreboard timing stdout set is incomplete")
    if set(profile_stdout_paths) != set(CONFIGURATION_BY_NAME):
        raise ValueError("scoreboard profile stdout set is incomplete")
    if set(counter_paths) != set(CONFIGURATION_BY_NAME):
        raise ValueError("scoreboard counter set is incomplete")
    static_by_name = {
        row["configuration"]: row for row in static_audit["kernels"]}
    samples = []
    reported_clocks = set()
    for plan_row in plan["samples"]:
        name = plan_row["configuration"]
        timing_stdout_path = timing_stdout_paths[name]
        profile_stdout_path = profile_stdout_paths[name]
        counter_path = counter_paths[name]
        timing_sample = parse_sample_stdout(
            timing_stdout_path, plan_row, physical_device_id, "timing")
        profile_sample = parse_sample_stdout(
            profile_stdout_path, plan_row, physical_device_id,
            "profile_once")
        bound_fields = (
            "configuration", "cache_state", "order_position", "grid",
            "block", "active_ap_count", "active_ap_count_source",
            "active_waves_per_ap", "live_wave_count_clock64", "dag_rounds",
            "working_set_bytes", "validation_checksum",
            "registers_per_thread", "shared_bytes_per_cta",
            "reported_clock_khz",
        )
        if any(timing_sample[field] != profile_sample[field]
               for field in bound_fields):
            raise ValueError(
                "timing and profile executions do not bind to one config")
        sample = timing_sample
        sample["c_round_source"] = "uninstrumented_timing_stdout"
        sample["instrumented_profile_cycles_consumed"] = False
        sample["instrumented_profile_median_wave_elapsed_xcore_cycles"] = (
            profile_sample["median_wave_elapsed_xcore_cycles"])
        sample["instrumented_profile_minimum_wave_elapsed_xcore_cycles"] = (
            profile_sample["minimum_wave_elapsed_xcore_cycles"])
        sample["instrumented_profile_maximum_wave_elapsed_xcore_cycles"] = (
            profile_sample["maximum_wave_elapsed_xcore_cycles"])
        sample.update(parse_counter_report(counter_path, plan_row))
        sample["timed_static_instruction_count"] = static_by_name[name][
            "timed_static_instruction_count"]
        sample["exp_f32_static_count"] = static_by_name[name][
            "exp_f32_static_count"]
        sample["rcp_f32_static_count"] = static_by_name[name][
            "rcp_f32_static_count"]
        sample["timing_stdout"] = {
            "path": str(timing_stdout_path),
            "sha256": _sha256(timing_stdout_path),
        }
        sample["profile_stdout"] = {
            "path": str(profile_stdout_path),
            "sha256": _sha256(profile_stdout_path),
        }
        sample["mcprofiler_isolated_report"] = {
            "path": str(counter_path), "sha256": _sha256(counter_path)}
        reported_clocks.add(sample.pop("reported_clock_khz"))
        samples.append(sample)
    if len(reported_clocks) != 1:
        raise ValueError("scoreboard samples report inconsistent XCORE clocks")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "software_revision": SOFTWARE_REVISION,
        "baseline_commit": BASELINE_COMMIT,
        "production_or_evaluation_duration_consumed": False,
        "instrumented_profiler_cycles_or_duration_consumed": False,
        "trial": trial,
        "physical_device_id": physical_device_id,
        "reported_clock_khz": next(iter(reported_clocks)),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "static_audit": {
            "path": str(static_audit_path),
            "sha256": _sha256(static_audit_path),
        },
        "source_sha256": static_audit["source"]["sha256"],
        "binary_sha256": static_audit["binary"]["sha256"],
        "samples": samples,
    }


def _load_run_report(path):
    payload = _read_json_object(path, "scoreboard run report")
    if set(payload) != RUN_REPORT_KEYS:
        raise ValueError("scoreboard run report schema is invalid")
    if (payload["schema_version"] != SCHEMA_VERSION
            or payload["hardware_id"] != HARDWARE_ID
            or payload["mechanism"] != MECHANISM
            or payload["software_revision"] != SOFTWARE_REVISION
            or payload["baseline_commit"] != BASELINE_COMMIT
            or payload["production_or_evaluation_duration_consumed"]
            is not False
            or payload[
                "instrumented_profiler_cycles_or_duration_consumed"]
            is not False):
        raise ValueError("scoreboard run report contract is invalid")
    if ({sample["configuration"] for sample in payload["samples"]}
            != set(CONFIGURATION_BY_NAME)):
        raise ValueError("scoreboard run report configuration set is incomplete")
    if any(
            sample["c_round_source"] != "uninstrumented_timing_stdout"
            or sample["instrumented_profile_cycles_consumed"] is not False
            for sample in payload["samples"]):
        raise ValueError("scoreboard run report consumes instrumented cycles")
    return payload


def _prediction_metrics(training_by_name, validation_samples, field):
    absolute_percentages = []
    signed_percentages = []
    for sample in validation_samples:
        observed = sample[field]
        predicted = training_by_name[sample["configuration"]][field]
        if observed <= 0:
            raise ValueError(f"scoreboard observed {field} must be positive")
        error = predicted - observed
        absolute_percentages.append(abs(error) / observed)
        signed_percentages.append(error / observed)
    return {
        "sample_count": len(validation_samples),
        "mape_percent": statistics.fmean(absolute_percentages) * 100.0,
        "signed_mean_percentage_error": (
            statistics.fmean(signed_percentages) * 100.0),
        "maximum_absolute_percentage_error": (
            max(absolute_percentages) * 100.0),
    }


def aggregate_reports(paths):
    reports = tuple(_load_run_report(path) for path in paths)
    trials = tuple(sorted(report["trial"] for report in reports))
    if trials != FULL_TRIALS or len(reports) != len(FULL_TRIALS):
        raise ValueError("scoreboard aggregate requires trials 1, 2, and 3")
    device_ids = {report["physical_device_id"] for report in reports}
    source_hashes = {report["source_sha256"] for report in reports}
    binary_hashes = {report["binary_sha256"] for report in reports}
    static_hashes = {report["static_audit"]["sha256"] for report in reports}
    if (len(device_ids) != 1 or len(source_hashes) != 1
            or len(binary_hashes) != 1 or len(static_hashes) != 1):
        raise ValueError("scoreboard aggregate mixes device or build provenance")
    training_samples = tuple(
        sample for report in reports if report["trial"] in TRAINING_TRIALS
        for sample in report["samples"])
    validation_samples = tuple(
        sample for report in reports if report["trial"] in VALIDATION_TRIALS
        for sample in report["samples"])
    parameter_fields = (
        "c_round_xcore_cycles", "dynamic_instruction_count",
        "l2_request_bytes", "dram_total_bytes",
    )
    training_by_name = {}
    signature = []
    for configuration in CONFIGURATIONS:
        name = configuration["configuration"]
        selected = tuple(
            sample for sample in training_samples
            if sample["configuration"] == name)
        if len(selected) != len(TRAINING_TRIALS):
            raise ValueError("scoreboard training cell is incomplete")
        medians = {
            field: statistics.median(sample[field] for sample in selected)
            for field in parameter_fields
        }
        training_by_name[name] = medians
        signature.append({
            "active_waves_per_ap": configuration["active_waves_per_ap"],
            "cache_state": configuration["cache_state"],
            "c_round_xcore_cycles": medians["c_round_xcore_cycles"],
            "dag_rounds": DAG_ROUNDS,
            "evidence": "training_trials_1_and_2_median",
        })
    diagnostics = {
        field: _prediction_metrics(
            training_by_name, validation_samples, field)
        for field in parameter_fields
    }
    l2_dram = training_by_name["l2_q32"]["dram_total_bytes"]
    streaming_dram = training_by_name["streaming_q32"]["dram_total_bytes"]
    cache_state_separated = streaming_dram > l2_dram
    cycle_gate = diagnostics["c_round_xcore_cycles"]
    counter_gates = tuple(
        diagnostics[field] for field in parameter_fields[1:])
    publishable = (
        cycle_gate["mape_percent"] <= MAX_VALIDATION_MAPE_PERCENT
        and cycle_gate["maximum_absolute_percentage_error"]
        <= MAX_VALIDATION_MAX_APE_PERCENT
        and all(
            metric["mape_percent"] <= MAX_COUNTER_MAPE_PERCENT
            and metric["maximum_absolute_percentage_error"]
            <= MAX_COUNTER_MAX_APE_PERCENT
            for metric in counter_gates)
        and cache_state_separated)
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "software_revision": SOFTWARE_REVISION,
        "baseline_commit": BASELINE_COMMIT,
        "production_or_evaluation_duration_consumed": False,
        "instrumented_profiler_cycles_or_duration_consumed": False,
        "physical_device_id": next(iter(device_ids)),
        "source_sha256": next(iter(source_hashes)),
        "binary_sha256": next(iter(binary_hashes)),
        "static_audit_sha256": next(iter(static_hashes)),
        "run_reports": [
            {"path": str(path), "sha256": _sha256(path)} for path in paths],
        "formula": "C_round(q, cache_state) in AP-local XCORE cycles",
        "scope": (
            "full active-wave signature for the fixed SiLU-like dependency "
            "DAG; not a per-element or production-shape correction"),
        "signature": signature,
        "frozen_validation": diagnostics,
        "cache_state_separation": {
            "l2_resident_dram_bytes": l2_dram,
            "streaming_dram_bytes": streaming_dram,
            "streaming_exceeds_l2_resident": cache_state_separated,
        },
        "publication_gate": {
            "maximum_validation_mape_percent": (
                MAX_VALIDATION_MAPE_PERCENT),
            "maximum_validation_max_ape_percent": (
                MAX_VALIDATION_MAX_APE_PERCENT),
            "maximum_counter_mape_percent": MAX_COUNTER_MAPE_PERCENT,
            "maximum_counter_max_ape_percent": (
                MAX_COUNTER_MAX_APE_PERCENT),
            "streaming_dram_must_exceed_l2_resident": True,
        },
        "publishable": publishable,
        "disposition": (
            "eligible_active_wave_scoreboard_signature" if publishable
            else "not_publishable_without_a_new_predeclared_campaign"),
    }


def _parse_labeled_paths(values, label):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use CONFIGURATION=PATH")
        configuration, raw_path = value.split("=", 1)
        if configuration not in CONFIGURATION_BY_NAME:
            raise ValueError(f"{label} configuration is unknown")
        if configuration in result:
            raise ValueError(f"{label} configuration is duplicated")
        result[configuration] = Path(raw_path)
    return result


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="C500 active-wave SiLU-like scoreboard mechanism")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--trial", type=int, required=True)
    plan.add_argument("--output", type=Path, required=True)
    static = subparsers.add_parser("static-audit")
    static.add_argument("--source", type=Path, required=True)
    static.add_argument("--binary", type=Path, required=True)
    static.add_argument("--disassembly", type=Path, required=True)
    static.add_argument("--baseline-commit", required=True)
    static.add_argument("--output", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--static-audit", type=Path, required=True)
    analyze.add_argument("--timing-stdout", action="append", required=True)
    analyze.add_argument("--profile-stdout", action="append", required=True)
    analyze.add_argument("--counter", action="append", required=True)
    analyze.add_argument("--trial", type=int, required=True)
    analyze.add_argument("--device-id", type=int, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--report", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        payload = build_plan(args.trial)
    elif args.command == "static-audit":
        payload = build_static_audit(
            args.source, args.binary, args.disassembly,
            args.baseline_commit)
    elif args.command == "analyze":
        payload = build_run_report(
            args.plan, args.static_audit,
            _parse_labeled_paths(args.timing_stdout, "timing stdout"),
            _parse_labeled_paths(args.profile_stdout, "profile stdout"),
            _parse_labeled_paths(args.counter, "counter"),
            args.trial, args.device_id)
    elif args.command == "aggregate":
        payload = aggregate_reports(args.report)
    else:
        raise AssertionError(f"unknown scoreboard command: {args.command!r}")
    _write_json(args.output, payload)


if __name__ == "__main__":
    main()
