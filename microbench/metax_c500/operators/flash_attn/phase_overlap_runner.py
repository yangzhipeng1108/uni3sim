#!/usr/bin/env python3
"""Capture and validate one C500 operator-independent cross-slot cell."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


SCHEMA_VERSION = 11
STDOUT_SCHEMA_VERSION = 3
HARDWARE_ID = "metax_c500"
MECHANISM = "cross_slot_smoke"
CELLS = ("mma_gvm", "mte_gvm", "mma_bsm", "mte_bsm")
MODES = ("slot2_isolated", "cross_isolated", "independent", "dependent")
KERNEL_IDENTITIES = {
    cell: {
        "slot2_isolated": f"c500_cross_{cell}_slot2",
        "cross_isolated": f"c500_cross_{cell}_cross",
        "independent": f"c500_cross_{cell}_independent",
        "dependent": f"c500_cross_{cell}_dependent",
    } for cell in CELLS
}
AP_COUNT = 104
WAVE_SIZE = 64
L2_BYTES = 8 * 1024 * 1024
GRID_BLOCKS = AP_COUNT
BLOCK_THREADS = 128
WAVES_PER_BLOCK = 2
ITERATIONS = 4093
STATIC_OPS_PER_ITERATION = 4
DYNAMIC_SHARED_BYTES = 64 * 1024
EVENT_TRIALS = 7
CLOCK_POSITIONS = ("before", "after")
ATTACH_WAIT_SECONDS = 15
TARGET_XCORE_CLOCK_MHZ = 1125
EXPECTED_MX_SMI_VERSION = "2.3.1"
EXPECTED_MACA_VERSION = "3.7.2"
# This mx-smi build reports a quantized 1 MBytes/s floor on every otherwise
# idle C500 (0% GPU utilization and no process).  Preserve the observed value
# and reject anything above that device-management floor.
MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND = 1.0
EXPECTED_STREAM_INSTRUCTIONS = (
    GRID_BLOCKS * ITERATIONS * STATIC_OPS_PER_ITERATION)
EXPECTED_CROSS_INSTRUCTIONS = EXPECTED_STREAM_INSTRUCTIONS
EXPECTED_SINK_GVM_INSTRUCTIONS = GRID_BLOCKS
INSTRUCTION_CLASSES = (
    "MTE", "MMA", "STE", "GVM", "BSM", "LDU", "MISC")
PROFILE_METRICS = (
    "WORKGROUPS", "WAVES", "Instructions Comparison",
    "Private Read Instructions", "Private Write Instructions")
MCPROFILER_VERSION_FIELDS = ("version", "time", "hash", "distro")


def _expected_instructions(cell, mode):
    if cell not in CELLS or mode not in MODES:
        raise ValueError("unknown cross-slot cell/mode")
    slot2_instruction = "MMA" if cell.startswith("mma_") else "MTE"
    cross_instruction = "GVM" if cell.endswith("_gvm") else "BSM"
    expected = {instruction: 0 for instruction in INSTRUCTION_CLASSES}
    if mode == "slot2_isolated":
        expected[slot2_instruction] = EXPECTED_STREAM_INSTRUCTIONS
        expected["GVM"] = EXPECTED_SINK_GVM_INSTRUCTIONS
    elif mode == "cross_isolated":
        expected[cross_instruction] = EXPECTED_CROSS_INSTRUCTIONS
    elif mode in ("independent", "dependent"):
        expected[slot2_instruction] = EXPECTED_STREAM_INSTRUCTIONS
        expected[cross_instruction] = EXPECTED_CROSS_INSTRUCTIONS
        expected["GVM"] = EXPECTED_SINK_GVM_INSTRUCTIONS
        if cross_instruction == "GVM":
            expected["GVM"] = (
                EXPECTED_CROSS_INSTRUCTIONS + EXPECTED_SINK_GVM_INSTRUCTIONS)
    else:
        raise AssertionError(f"unhandled cross-slot mode: {mode}")
    return expected
SOURCE_RELATIVE_PATH = (
    "microbench/metax_c500/operators/flash_attn/phase_overlap.cpp")
RUNNER_RELATIVE_PATH = (
    "microbench/metax_c500/operators/flash_attn/phase_overlap_runner.py")
ALLOWED_OUTPUT_RELATIVE_PATH = "profiler_output/microbench/metax_c500"
MACA_PATH = Path("/opt/maca")
TARGET_LIBSTDCXX_PATH = Path("/lib/x86_64-linux-gnu/libstdc++.so.6")
MXCC_PATH = MACA_PATH / "mxgpu_llvm/bin/mxcc"
MX_SMI_PATH = Path("/opt/mxdriver/bin/mx-smi")
def _host_global_lock_path():
    if "MIDLAYER_C500_MCPROFILER_LOCK" not in os.environ:
        raise ValueError(
            "MIDLAYER_C500_MCPROFILER_LOCK is required for mcProfiler execution"
        )
    return Path(os.environ["MIDLAYER_C500_MCPROFILER_LOCK"])
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
IMAGE_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NUMBERED_REPORT_PATTERN = re.compile(r"([1-9][0-9]*)_(.+)\.txt\.json")
MX_SMI_DEVICE_PATTERN = re.compile(
    r"^\s*GPU#(?P<device>\d+)\s+MXC500\s+"
    r"(?P<bus>[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_XCORE_PATTERN = re.compile(
    r"^\s*\|?\s*XCORE\s*(?:\||:)\s*(\d+)\s*MHz\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_GPU_USAGE_PATTERN = re.compile(
    r"^\s*\|?\s*GPU\s*(?:\||:)\s*(\d+(?:\.\d+)?)\s*%\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_HBM_PATTERN = re.compile(
    r"^\s*\|?\s*throughput\s*(?:\||:)\s*"
    r"(\d+(?:\.\d+)?)\s*MBytes/s\s*\|?\s*$",
    re.IGNORECASE | re.MULTILINE)
MX_SMI_NO_PROCESS_PATTERN = re.compile(
    r"^[ \t]*\|?[ \t]*"
    r"(?:No process(?: found)?|No running processes(?: found)?)\.?"
    r"[ \t]*\|?[ \t]*$", re.IGNORECASE | re.MULTILINE)


def _repository_root():
    return Path(__file__).resolve().parents[4]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(mapping, expected, label):
    if not isinstance(mapping, dict) or set(mapping) != set(expected):
        raise ValueError(f"{label} schema is not exact")


def _parse_mcprofiler_version(text):
    """Return the exact build identity printed by ``mcProfiler version``."""

    fields = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key, separator, value = stripped.partition(":")
        if (separator != ":" or key not in MCPROFILER_VERSION_FIELDS
                or key in fields or not value):
            raise ValueError("mcProfiler version output is invalid")
        fields[key] = value
    if tuple(fields) != MCPROFILER_VERSION_FIELDS:
        raise ValueError("mcProfiler version output is incomplete or reordered")
    return "|".join(f"{key}:{fields[key]}" for key in MCPROFILER_VERSION_FIELDS)


def _write_report_anchor(report_path, anchor_path):
    with anchor_path.open("x", encoding="ascii") as stream:
        stream.write(f"{_sha256(report_path)}  {report_path.name}\n")


def _verify_report_anchor(report_path, anchor_path):
    fields = anchor_path.read_text(encoding="ascii").splitlines()
    expected = f"{_sha256(report_path)}  {report_path.name}"
    if fields != [expected]:
        raise ValueError("final report hash anchor does not match report bytes")


def _capture_contract_payload(
        mcprofiler_path, mcprofiler_sha256, python_executable, port):
    return {
        "schema_version": SCHEMA_VERSION,
        "mcprofiler_runtime_path": str(mcprofiler_path),
        "mcprofiler_executable_sha256": mcprofiler_sha256,
        "python_executable": str(python_executable),
        "port": port,
        "host_global_lock_path": str(_host_global_lock_path()),
        "holder_pid": os.getpid(),
    }


def _verify_capture_contract(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_exact_keys(payload, {
        "schema_version", "mcprofiler_runtime_path",
        "mcprofiler_executable_sha256", "python_executable", "port",
        "host_global_lock_path", "holder_pid"}, "capture contract")
    runtime_value = payload["mcprofiler_runtime_path"]
    python_value = payload["python_executable"]
    executable_sha256 = payload["mcprofiler_executable_sha256"]
    port = payload["port"]
    holder_pid = payload["holder_pid"]
    if (payload["schema_version"] != SCHEMA_VERSION
            or not isinstance(runtime_value, str)
            or not isinstance(executable_sha256, str)
            or SHA256_PATTERN.fullmatch(executable_sha256) is None
            or not isinstance(python_value, str)
            or not isinstance(port, int) or isinstance(port, bool)
            or port <= 0 or port > 65535
            or not isinstance(payload["host_global_lock_path"], str)
            or payload["host_global_lock_path"]
            != str(_host_global_lock_path())
            or not isinstance(holder_pid, int) or isinstance(holder_pid, bool)
            or holder_pid <= 0):
        raise ValueError("capture contract identity is invalid")
    runtime_path = Path(runtime_value)
    python_executable = Path(python_value)
    if (not runtime_path.is_absolute()
            or runtime_path != runtime_path.resolve(strict=False)
            or not python_executable.is_absolute()
            or python_executable != python_executable.resolve(strict=False)):
        raise ValueError("capture contract path identity is invalid")
    return payload


def _write_json_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _copy_exclusive(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_stream, destination.open("xb") as output:
        for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
            output.write(chunk)


def _require_contained(path, root, label):
    resolved = path.resolve()
    allowed = root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"{label} must stay below {allowed}")
    return resolved


def _git_commit(repository_root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, check=False)
    commit = result.stdout.strip()
    if result.returncode != 0 or COMMIT_PATTERN.fullmatch(commit) is None:
        raise RuntimeError("cannot resolve an exact deployed source commit")
    return commit


def audit_source(path=None):
    if path is None:
        path = _repository_root() / SOURCE_RELATIVE_PATH
    source = path.read_text(encoding="utf-8")
    required = (
        "kGridBlocks = kExpectedApCount",
        "kBlockThreads = 2 * kExpectedWaveSize",
        "kIterations = 4093",
        "__builtin_mxc_mma_16x16x16bf16",
        'asm volatile("add_f32',
        "c500_cross_mma_gvm_slot2",
        "c500_cross_mma_gvm_cross",
        "c500_cross_mma_gvm_independent",
        "c500_cross_mma_gvm_dependent",
        "c500_cross_mte_gvm_independent",
        "c500_cross_mma_bsm_independent",
        "c500_cross_mte_bsm_independent",
        "const float token = slot2_stream<slot2>(seed);",
        "gvm_store_stream(",
        "bsm_store_stream(",
        "attributes.sharedSizeBytes != expected_static_shared_bytes",
        "kDynamicSharedBytes = 64 * 1024",
        "resident_blocks_per_ap != 1",
        "raise(SIGSTOP)",
        "CHECK(mcProfilerStart());",
        "CHECK(mcProfilerStop());",
    )
    if any(fragment not in source for fragment in required):
        raise ValueError("cross-slot source contract changed")
    forbidden = (
        "flash_attn(", "paged_attention(", "kernel_dataset/",
        "evaluation/", "execute_phase(", "__syncthreads()",
        "kPipelineCycles", "q64", "Q64",
    )
    if any(fragment in source for fragment in forbidden):
        raise ValueError("cross-slot source contains excluded workload logic")
    marker = source[
        source.index("CHECK(mcProfilerStart());"):
        source.index("CHECK(mcProfilerStop());")]
    if marker.count("launch(spec") != 1:
        raise ValueError("profile marker must contain one target launch")
    return path


def build_plan(cell):
    if cell not in CELLS:
        raise ValueError("unknown cross-slot cell")
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "cell": cell,
        "production_or_evaluation_duration_consumed": False,
        "mcprofiler_instrumented_duration_consumed": False,
        "launch": {
            "grid_blocks": GRID_BLOCKS,
            "block_threads": BLOCK_THREADS,
            "waves_per_block": WAVES_PER_BLOCK,
            "required_resident_blocks_per_ap": 1,
            "static_shared_bytes": 0,
            "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
            "placement_rationale": (
                "device-wide grid equals AP count; every mode launches with "
                "the full 64-KiB per-AP shared-memory capacity and the "
                "runtime occupancy query must therefore report one resident "
                "two-wave CTA per AP"),
        },
        "fixed_work": {
            "iterations": ITERATIONS,
            "static_ops_per_iteration": STATIC_OPS_PER_ITERATION,
            "q64_proportion_reproduced": False,
            "instruction_class_ledger": list(INSTRUCTION_CLASSES),
            "auxiliary_lowering_policy": (
                "STE/LDU/MISC are pre-registered as zero; any compiler "
                "lowering that emits them is retained raw and rejected "
                "until independently audited"),
        },
        "samples": [{
            "mode": mode,
            "kernel_identity": KERNEL_IDENTITIES[cell][mode],
            "expected_instruction_counts": _expected_instructions(cell, mode),
        } for mode in MODES],
        "decision_rule": {
            "independent_expected": "max(isolated_slot2, isolated_cross)",
            "dependent_expected": "sum(isolated_slot2, isolated_cross)",
            "classification": (
                "all seven candidate trials must fall in exactly one "
                "pre-registered envelope expanded only by the isolated "
                "event-timer repeatability measured in this experiment; "
                "trial dispersion must remain smaller than one quarter of "
                "the separation between the max and sum envelopes"),
            "clock_evidence": (
                "one synchronous mx-smi query immediately before and one "
                "immediately after each device-event trial must both report "
                "1125 MHz; the child remains SIGSTOPed during every query so "
                "clock observation cannot inflate device-event duration"),
        },
    }


def _parse_uint(text, label):
    if re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise ValueError(f"{label} must be a canonical unsigned integer")
    return int(text)


def _parse_positive_float(text, label):
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


def parse_inspection(path, device_id):
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 1 or len(rows[0]) != 11:
        raise ValueError("device inspection schema is invalid")
    row = rows[0]
    expected = (
        "inspect", str(STDOUT_SCHEMA_VERSION), HARDWARE_ID, str(device_id),
        row[4], str(AP_COUNT), str(WAVE_SIZE), str(L2_BYTES),
        str(GRID_BLOCKS), str(BLOCK_THREADS), str(ITERATIONS))
    if tuple(row) != expected or "C500" not in row[4]:
        raise ValueError("device inspection violates the C500 contract")
    return {"device_name": row[4], "physical_device_id": device_id}


def parse_stdout(path, cell, mode, capture_kind, device_id):
    if cell not in CELLS or mode not in MODES:
        raise ValueError("unknown cross-slot cell/mode")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if capture_kind == "event":
        expected_rows = 2 + EVENT_TRIALS
        expected_trials = EVENT_TRIALS
    elif capture_kind == "profile":
        expected_rows = 2
        expected_trials = 0
    else:
        raise ValueError("capture kind must be event or profile")
    if len(rows) != expected_rows:
        raise ValueError("cross-slot stdout row count is invalid")
    metadata = (
        "metadata", str(STDOUT_SCHEMA_VERSION), capture_kind, HARDWARE_ID,
        MECHANISM, "0", "0", str(device_id), str(AP_COUNT), str(WAVE_SIZE),
        str(L2_BYTES))
    if tuple(rows[0]) != metadata:
        raise ValueError("cross-slot stdout metadata is invalid")
    row = rows[1]
    if len(row) != 18 or tuple(row[:4]) != (
            "sample", cell, mode, KERNEL_IDENTITIES[cell][mode]):
        raise ValueError("cross-slot sample identity is invalid")
    integers = tuple(_parse_uint(value, "sample integer") for value in row[4:10])
    if integers != (
            GRID_BLOCKS, BLOCK_THREADS, WAVES_PER_BLOCK, ITERATIONS,
            STATIC_OPS_PER_ITERATION, expected_trials):
        raise ValueError("cross-slot sample launch/work contract changed")
    durations = tuple(float(value) for value in row[10:13])
    if capture_kind == "event":
        if (not all(math.isfinite(value) and value > 0 for value in durations)
                or tuple(sorted(durations)) != durations):
            raise ValueError("cross-slot event summary is invalid")
    elif durations != (0.0, 0.0, 0.0):
        raise ValueError("profile stdout exposes instrumented duration")
    checksum = float(row[13])
    if not math.isfinite(checksum):
        raise ValueError("cross-slot checksum is invalid")
    resources = tuple(_parse_uint(value, "resource field") for value in row[14:])
    if (resources[0] <= 0 or resources[1] != 0
            or resources[2] != DYNAMIC_SHARED_BYTES or resources[3] != 1):
        raise ValueError("cross-slot resource metadata is invalid")
    trials = []
    previous_end = None
    if capture_kind == "event":
        for index, trial_row in enumerate(rows[2:], start=1):
            if len(trial_row) != 5 or trial_row[0] != "trial":
                raise ValueError("cross-slot event trial row is invalid")
            trial_index = _parse_uint(trial_row[1], "trial index")
            started = _parse_uint(trial_row[2], "trial start")
            ended = _parse_uint(trial_row[3], "trial end")
            elapsed_ms = _parse_positive_float(trial_row[4], "trial duration")
            if (trial_index != index or ended < started
                    or (previous_end is not None and started <= previous_end)):
                raise ValueError("cross-slot event intervals are invalid")
            if ended - started < elapsed_ms * 1_000_000.0:
                raise ValueError("host interval is shorter than device event")
            previous_end = ended
            trials.append({
                "trial_index": trial_index,
                "started_monotonic_raw_ns": started,
                "ended_monotonic_raw_ns": ended,
                "event_ms": elapsed_ms,
            })
        sorted_trials = sorted(trial["event_ms"] for trial in trials)
        expected_summary = (
            sorted_trials[0], sorted_trials[len(sorted_trials) // 2],
            sorted_trials[-1])
        if any(not math.isclose(value, expected, abs_tol=5e-10, rel_tol=0)
               for value, expected in zip(durations, expected_summary)):
            raise ValueError("event summary disagrees with trial rows")
    return {
        "mode": mode,
        "kernel_identity": KERNEL_IDENTITIES[cell][mode],
        "event_ms_min": durations[0],
        "event_ms_median": durations[1],
        "event_ms_max": durations[2],
        "checksum": checksum,
        "registers_per_thread": resources[0],
        "static_shared_bytes": resources[1],
        "dynamic_shared_bytes": resources[2],
        "resident_blocks_per_ap": resources[3],
        "trials": trials,
    }


def _trial_values(sample):
    trials = sample["trials"]
    if (not isinstance(trials, list) or len(trials) != EVENT_TRIALS
            or [trial["trial_index"] for trial in trials]
            != list(range(1, EVENT_TRIALS + 1))):
        raise ValueError("decision requires seven ordered raw event trials")
    values = [trial["event_ms"] for trial in trials]
    if not all(isinstance(value, float) and value > 0 for value in values):
        raise ValueError("decision event trials are invalid")
    return values


def _classify_trials(observed, isolated_slot2, isolated_cross):
    slot2_interval = (min(isolated_slot2), max(isolated_slot2))
    cross_interval = (min(isolated_cross), max(isolated_cross))
    timer_repeatability_ms = max(
        slot2_interval[1] - slot2_interval[0],
        cross_interval[1] - cross_interval[0])
    raw_max_interval = (
        max(slot2_interval[0], cross_interval[0]),
        max(slot2_interval[1], cross_interval[1]))
    raw_sum_interval = (
        slot2_interval[0] + cross_interval[0],
        slot2_interval[1] + cross_interval[1])
    envelope_separation_ms = raw_sum_interval[0] - raw_max_interval[1]
    dispersion_limit_ms = envelope_separation_ms / 4.0
    dispersions = {
        "slot2_isolated": slot2_interval[1] - slot2_interval[0],
        "cross_isolated": cross_interval[1] - cross_interval[0],
        "candidate": max(observed) - min(observed),
    }
    max_interval = (
        max(0.0, raw_max_interval[0] - timer_repeatability_ms),
        raw_max_interval[1] + timer_repeatability_ms)
    sum_interval = (
        max(0.0, raw_sum_interval[0] - timer_repeatability_ms),
        raw_sum_interval[1] + timer_repeatability_ms)
    classification = "inconclusive"
    trial_matches = []
    if (envelope_separation_ms > 0
            and all(value <= dispersion_limit_ms
                    for value in dispersions.values())
            and max_interval[1] < sum_interval[0]):
        for value in observed:
            matches = []
            if max_interval[0] <= value <= max_interval[1]:
                matches.append("max")
            if sum_interval[0] <= value <= sum_interval[1]:
                matches.append("sum")
            trial_matches.append(matches)
        if all(matches == ["max"] for matches in trial_matches):
            classification = "max"
        elif all(matches == ["sum"] for matches in trial_matches):
            classification = "sum"
    return {
        "classification": classification,
        "observed_interval_ms": [min(observed), max(observed)],
        "max_envelope_ms": list(max_interval),
        "sum_envelope_ms": list(sum_interval),
        "timer_repeatability_ms": timer_repeatability_ms,
        "envelope_separation_ms": envelope_separation_ms,
        "dispersion_limit_ms": dispersion_limit_ms,
        "dispersions_ms": dispersions,
        "trial_matches": trial_matches,
    }


def _validate_mode_resources(by_mode):
    resource_fields = (
        "registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes",
        "resident_blocks_per_ap")
    for mode, sample in by_mode.items():
        if any(field not in sample for field in resource_fields):
            raise ValueError(f"{mode} is missing launch resource evidence")
    resident = {
        sample["resident_blocks_per_ap"] for sample in by_mode.values()}
    shared = {sample["static_shared_bytes"] for sample in by_mode.values()}
    dynamic_shared = {
        sample["dynamic_shared_bytes"] for sample in by_mode.values()}
    registers = {
        sample["registers_per_thread"] for sample in by_mode.values()}
    if (resident != {1} or shared != {0}
            or dynamic_shared != {DYNAMIC_SHARED_BYTES}
            or min(registers) <= 0):
        raise ValueError("mode launch resources are not exactly closed")
    return {
        field: {mode: by_mode[mode][field] for mode in MODES}
        for field in resource_fields
    }


def decide(samples):
    by_mode = {sample["mode"]: sample for sample in samples}
    if len(samples) != len(MODES) or set(by_mode) != set(MODES):
        raise ValueError("one-cell event evidence is incomplete")
    resources = _validate_mode_resources(by_mode)
    slot2 = _trial_values(by_mode["slot2_isolated"])
    cross = _trial_values(by_mode["cross_isolated"])
    independent = _classify_trials(
        _trial_values(by_mode["independent"]), slot2, cross)
    dependent = _classify_trials(
        _trial_values(by_mode["dependent"]), slot2, cross)
    if (independent["classification"] == "max"
            and dependent["classification"] == "sum"):
        status = "cross_slot_concurrent_dependency_serial"
    elif independent["classification"] == "sum":
        status = "cross_slot_serial"
    else:
        status = "inconclusive"
    return {
        "status": status,
        "independent": independent,
        "dependent": dependent,
        "resource_equivalence": resources,
        "placement_scope": (
            "device-wide fixed 104-CTA grid with equal resident-block "
            "capacity; no per-AP CTA mapping is inferred"),
        "continuous_fit_performed": False,
    }


def _profiler_sections(payload, path):
    if not isinstance(payload, dict) or not payload:
        raise TypeError(f"{path} must contain a non-empty JSON object")
    scalars = {}
    charts = {}
    for section, entries in payload.items():
        if not isinstance(entries, list):
            raise TypeError(f"{path} profiler section must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) < {"name", "value"}:
                raise ValueError(f"{path} profiler entry is invalid")
            name = entry["name"]
            value = entry["value"]
            if isinstance(value, dict):
                if section != "Internal":
                    if name in charts:
                        raise ValueError(f"{path} duplicates chart {name}")
                    wrapped_chart = set(value) == {"data", "filename"}
                    if wrapped_chart:
                        if (not isinstance(value["data"], dict)
                                or not isinstance(value["filename"], str)
                                or not value["filename"]):
                            raise TypeError(
                                f"{path} wrapped chart is invalid")
                        charts[name] = value["data"]
                    else:
                        charts[name] = value
            else:
                if name in scalars:
                    raise ValueError(f"{path} duplicates scalar {name}")
                scalars[name] = value
    return scalars, charts


def _profiler_integer(value, label):
    if isinstance(value, bool):
        raise TypeError(f"{label} cannot be boolean")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", ""))
        except ValueError as error:
            raise ValueError(f"{label} must be numeric") from error
    else:
        raise TypeError(f"{label} has an invalid type")
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return int(number)


def parse_counter_report(path, cell, mode):
    expected_instructions = _expected_instructions(cell, mode)
    match = NUMBERED_REPORT_PATTERN.fullmatch(path.name)
    if (match is None or int(match.group(1)) != 1
            or match.group(2) != KERNEL_IDENTITIES[cell][mode]
            or "period" in path.name.lower()):
        raise ValueError("mcProfiler report identity is invalid")
    scalars, charts = _profiler_sections(
        json.loads(path.read_text(encoding="utf-8")), path)
    for name in ("WORKGROUPS", "WAVES"):
        if name not in scalars:
            raise KeyError(f"{path} is missing {name}")
    if (_profiler_integer(scalars["WORKGROUPS"], "WORKGROUPS") != GRID_BLOCKS
            or _profiler_integer(scalars["WAVES"], "WAVES")
            != GRID_BLOCKS * WAVES_PER_BLOCK):
        raise ValueError("mcProfiler launch counts disagree with the cell")
    if "Instructions Comparison" not in charts:
        raise KeyError(f"{path} is missing Instructions Comparison")
    instructions = charts["Instructions Comparison"]
    if not isinstance(instructions, dict):
        raise TypeError("Instructions Comparison must be an object")
    if not set(INSTRUCTION_CLASSES).issubset(instructions):
        raise KeyError(f"{path} is missing a required instruction class")
    observed = {
        instruction: _profiler_integer(instructions[instruction], instruction)
        for instruction in INSTRUCTION_CLASSES
    }
    for instruction, value in instructions.items():
        if instruction not in INSTRUCTION_CLASSES and _profiler_integer(
                value, instruction) != 0:
            raise ValueError(
                f"{mode} has an unregistered nonzero instruction class")
    for instruction in expected_instructions:
        if instruction not in instructions:
            raise KeyError(f"{path} is missing {instruction}")
    exact_classes = ("MMA", "GVM", "BSM")
    exact_mismatch = any(
        observed[instruction] != expected_instructions[instruction]
        for instruction in exact_classes)
    # MTE also contains compiler-generated loop/control traffic, while the
    # injected MMA/GVM/BSM instructions have exact static counts.
    mte_minimum = expected_instructions["MTE"]
    if exact_mismatch or observed["MTE"] < mte_minimum:
        raise ValueError(
            f"{mode} instruction counts differ from the frozen source contract")
    spill_fields = ("private_kernel_rd", "private_kernel_wr")
    if "Memory Data Flow Chart" in charts:
        memory = charts["Memory Data Flow Chart"]
        if any(field not in memory for field in spill_fields):
            raise KeyError(f"{path} is missing private-memory spill evidence")
        spill_traffic = {
            field: _profiler_integer(memory[field], field)
            for field in spill_fields
        }
    else:
        private_instruction_fields = {
            "private_kernel_rd": "Private Read Instructions",
            "private_kernel_wr": "Private Write Instructions",
        }
        if any(
                metric not in scalars
                for metric in private_instruction_fields.values()):
            raise KeyError(f"{path} is missing private-memory spill evidence")
        private_instructions = {
            field: _profiler_integer(scalars[metric], metric)
            for field, metric in private_instruction_fields.items()
        }
        if any(private_instructions.values()):
            raise ValueError(f"{mode} has private-memory spill instructions")
        spill_traffic = {field: 0 for field in spill_fields}
    if any(spill_traffic.values()):
        raise ValueError(f"{mode} has nonzero private-memory spill traffic")
    return {
        "mode": mode,
        "kernel_identity": KERNEL_IDENTITIES[cell][mode],
        "instruction_counts": observed,
        "private_memory_spill_bytes": spill_traffic,
        "report_path": str(path),
        "report_sha256": _sha256(path),
        "instrumented_duration_consumed": False,
    }


def _run_checked(argv, stdout_path, stderr_path, cwd, environment):
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        result = subprocess.run(
            argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; "
            f"stdout={stdout_path}; stderr={stderr_path}")
    return result.returncode


def _manifest_payload(root, excluded_names):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact inventory contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact inventory contains special file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in excluded_names:
            continue
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {"schema_version": SCHEMA_VERSION, "files": files}


def _verify_manifest(root, manifest_path, allowed_unlisted, exact_inventory):
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "files"}:
        raise ValueError("artifact manifest schema is invalid")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("artifact manifest version changed")
    entries = payload["files"]
    if not isinstance(entries, list):
        raise TypeError("artifact manifest files must be a list")
    expected_paths = set()
    for entry in entries:
        if (not isinstance(entry, dict)
                or set(entry) != {"path", "size_bytes", "sha256"}):
            raise ValueError("artifact manifest entry schema is invalid")
        relative = Path(entry["path"])
        if (relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() in expected_paths):
            raise ValueError("artifact manifest path is invalid")
        expected_paths.add(relative.as_posix())
        path = _require_contained(
            root / relative, root, "artifact manifest entry")
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != entry["size_bytes"]
                or _sha256(path) != entry["sha256"]):
            raise ValueError(f"artifact file differs from manifest: {relative}")
    if exact_inventory:
        actual_paths = set()
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("artifact inventory contains a symlink")
            if path.is_file():
                actual_paths.add(path.relative_to(root).as_posix())
        if actual_paths != expected_paths | set(allowed_unlisted):
            raise ValueError("artifact file inventory differs from manifest")
    return payload


def _unique_device_bus(text, device_id, label):
    devices = list(MX_SMI_DEVICE_PATTERN.finditer(text))
    if len(devices) != 1 or int(devices[0].group("device")) != device_id:
        raise ValueError(f"mx-smi {label} output has no unique target C500")
    return devices[0].group("bus").lower()


def _unique_float(pattern, text, label):
    values = pattern.findall(text)
    if len(values) != 1:
        raise ValueError(f"mx-smi output has invalid {label}")
    value = float(values[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"mx-smi output has invalid {label}")
    return value


def _capture_idle_preflight(
        output_dir, device_id, expected_bus, repository_root, environment):
    output_dir.mkdir(parents=True, exist_ok=False)
    queries = (
        ("usage", "--show-usage"),
        ("hbm", "--show-hbm-bandwidth"),
        ("process", "--show-process"),
    )
    evidence = {}
    bus_id = None
    gpu_utilization = None
    hbm_throughput = None
    for name, option in queries:
        stdout_path = output_dir / f"{name}.stdout.txt"
        stderr_path = output_dir / f"{name}.stderr.txt"
        argv = [str(MX_SMI_PATH), option, "-i", str(device_id)]
        started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        _run_checked(
            argv, stdout_path, stderr_path, repository_root, environment)
        ended = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        text = stdout_path.read_text(encoding="utf-8")
        if name != "process":
            observed_bus = _unique_device_bus(text, device_id, name)
            if bus_id is None:
                bus_id = observed_bus
            elif observed_bus != bus_id:
                raise ValueError("preflight device bus identity changed")
        evidence[name] = {
            "argv": argv,
            "started_monotonic_raw_ns": started,
            "ended_monotonic_raw_ns": ended,
            "stdout_path": stdout_path.relative_to(output_dir).as_posix(),
            "stdout_sha256": _sha256(stdout_path),
            "stderr_path": stderr_path.relative_to(output_dir).as_posix(),
            "stderr_sha256": _sha256(stderr_path),
        }
        if name == "usage":
            gpu_utilization = _unique_float(
                MX_SMI_GPU_USAGE_PATTERN, text, "GPU utilization")
            if gpu_utilization != 0.0:
                raise RuntimeError(
                    "C500 preflight found nonzero GPU utilization")
        if name == "hbm":
            hbm_throughput = _unique_float(
                MX_SMI_HBM_PATTERN, text, "HBM throughput")
            if hbm_throughput > MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND:
                raise RuntimeError(
                    "C500 preflight found HBM throughput above the idle "
                    "mx-smi reporting floor")
        if name == "process":
            markers = MX_SMI_NO_PROCESS_PATTERN.findall(text)
            if len(markers) != 1:
                raise RuntimeError("C500 preflight found an active GPU process")
    if expected_bus is not None and bus_id != expected_bus:
        raise ValueError("preflight device differs from captured C500")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "device_id": device_id,
        "device_bus_id": bus_id,
        "idle": True,
        "gpu_utilization_pct": gpu_utilization,
        "hbm_throughput_mbytes_per_second": hbm_throughput,
        "process_host_pids": [],
        "queries": evidence,
    }
    _write_json_exclusive(output_dir / "preflight.json", payload)
    return _verify_idle_preflight(
        output_dir, device_id, bus_id)


def _verify_idle_preflight(output_dir, device_id, expected_bus):
    payload_path = output_dir / "preflight.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    _require_exact_keys(payload, {
        "schema_version", "device_id", "device_bus_id", "idle",
        "gpu_utilization_pct", "hbm_throughput_mbytes_per_second",
        "process_host_pids", "queries"}, "idle preflight")
    _require_exact_keys(payload["queries"], {"usage", "hbm", "process"},
                        "idle preflight queries")
    expected_files = {"preflight.json"}
    recomputed_queries = {}
    bus_id = None
    gpu_utilization = None
    hbm_throughput = None
    for name, option in (
            ("usage", "--show-usage"),
            ("hbm", "--show-hbm-bandwidth"),
            ("process", "--show-process")):
        query = payload["queries"][name]
        _require_exact_keys(query, {
            "argv", "started_monotonic_raw_ns", "ended_monotonic_raw_ns",
            "stdout_path", "stdout_sha256", "stderr_path",
            "stderr_sha256"}, "idle preflight query")
        if query["argv"] != [
                str(MX_SMI_PATH), option, "-i", str(device_id)]:
            raise ValueError("idle preflight query identity changed")
        started = query["started_monotonic_raw_ns"]
        ended = query["ended_monotonic_raw_ns"]
        if (not isinstance(started, int) or not isinstance(ended, int)
                or started >= ended):
            raise ValueError("idle preflight query interval is invalid")
        paths = []
        for kind in ("stdout", "stderr"):
            relative = Path(query[f"{kind}_path"])
            expected_relative = Path(f"{name}.{kind}.txt")
            if relative != expected_relative:
                raise ValueError("idle preflight raw path changed")
            path = output_dir / relative
            if (not path.is_file()
                    or _sha256(path) != query[f"{kind}_sha256"]):
                raise ValueError("idle preflight raw hash changed")
            expected_files.add(relative.as_posix())
            paths.append(path)
        text = paths[0].read_text(encoding="utf-8")
        if name != "process":
            observed_bus = _unique_device_bus(text, device_id, name)
            if bus_id is None:
                bus_id = observed_bus
            elif observed_bus != bus_id:
                raise ValueError("idle preflight bus identity changed")
        if name == "usage":
            gpu_utilization = _unique_float(
                MX_SMI_GPU_USAGE_PATTERN, text, "GPU utilization")
            if gpu_utilization != 0.0:
                raise RuntimeError(
                    "C500 preflight found nonzero GPU utilization")
        if name == "hbm":
            hbm_throughput = _unique_float(
                MX_SMI_HBM_PATTERN, text, "HBM throughput")
            if hbm_throughput > MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND:
                raise RuntimeError(
                    "C500 preflight found HBM throughput above the idle "
                    "mx-smi reporting floor")
        if (name == "process"
                and len(MX_SMI_NO_PROCESS_PATTERN.findall(text)) != 1):
            raise RuntimeError("C500 preflight found an active GPU process")
        recomputed_queries[name] = query
    actual_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("idle preflight raw file inventory changed")
    recomputed = {
        "schema_version": SCHEMA_VERSION,
        "device_id": device_id,
        "device_bus_id": bus_id,
        "idle": True,
        "gpu_utilization_pct": gpu_utilization,
        "hbm_throughput_mbytes_per_second": hbm_throughput,
        "process_host_pids": [],
        "queries": recomputed_queries,
    }
    if bus_id != expected_bus or payload != recomputed:
        raise ValueError("idle preflight derived payload changed")
    return recomputed


def _run_profile_checked(argv, stdout_path, stderr_path, cwd, environment):
    started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    process = None
    cleanup = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            process = subprocess.Popen(
                argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True)
            pid = process.pid
            pgid = pid
            if os.getpgid(pid) != pgid:
                process.kill()
                process.wait(timeout=30)
                raise RuntimeError(
                    "mcProfiler did not enter its own process group")
            returncode = process.wait(timeout=2400)
        except BaseException:
            raise
        finally:
            if process is not None:
                cleanup = _terminate_process_group(
                    process, process.pid, process.pid)
    ended = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if returncode != 0:
        raise RuntimeError(f"mcProfiler exited with code {returncode}")
    return {
        "pid": process.pid,
        "pgid": pgid,
        "process_group_cleanup": cleanup,
        "started_monotonic_raw_ns": started,
        "ended_monotonic_raw_ns": ended,
    }


def _parse_clock(path, device_id):
    text = path.read_text(encoding="utf-8")
    clocks = MX_SMI_XCORE_PATTERN.findall(text)
    if len(clocks) != 1:
        raise ValueError("active mx-smi output has no unique XCORE clock")
    return _unique_device_bus(text, device_id, "clock"), int(clocks[0])


def _process_group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(process, expected_pid, expected_pgid):
    if process.pid != expected_pid or expected_pgid != expected_pid:
        raise RuntimeError("refusing inexact process-group cleanup")
    term_sent = False
    kill_sent = False
    if _process_group_exists(expected_pgid):
        try:
            observed_pgid = os.getpgid(expected_pid)
        except ProcessLookupError:
            observed_pgid = expected_pgid
        if observed_pgid != expected_pgid:
            raise RuntimeError("process group identity changed before cleanup")
        try:
            os.killpg(expected_pgid, signal.SIGTERM)
        except ProcessLookupError as error:
            if _process_group_exists(expected_pgid):
                raise RuntimeError(
                    "process group disappeared before cleanup") from error
        else:
            term_sent = True
    deadline = time.monotonic() + 5.0
    while _process_group_exists(expected_pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_group_exists(expected_pgid):
        try:
            os.killpg(expected_pgid, signal.SIGKILL)
        except ProcessLookupError as error:
            if _process_group_exists(expected_pgid):
                raise RuntimeError(
                    "process group disappeared before SIGKILL") from error
        else:
            kill_sent = True
    process.wait(timeout=30)
    deadline = time.monotonic() + 5.0
    while _process_group_exists(expected_pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_group_exists(expected_pgid):
        raise RuntimeError("profile process group is not quiescent")
    return {
        "pid": expected_pid,
        "pgid": expected_pgid,
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
        "pgid_quiescent": True,
    }


def _capture_mode(
        binary, cell, mode, device_id, sample_dir, repository_root, env):
    stdout_path = sample_dir / "event.stdout.csv"
    stderr_path = sample_dir / "event.stderr.txt"
    raw_clock = sample_dir / "active_clock"
    raw_clock.mkdir(parents=True)
    argv = [
        str(binary), "--event-controlled", "--device-id", str(device_id),
        "--cell", cell, "--mode", mode]
    clock_rows = []
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        target = subprocess.Popen(
            argv, cwd=repository_root, env=env, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True)
        target_pid = target.pid
        target_pgid = os.getpgid(target_pid)
        if target_pgid != target_pid:
            target.kill()
            target.wait(timeout=30)
            raise RuntimeError("event target did not enter its own process group")
        try:
            waited_pid, status = os.waitpid(target.pid, os.WUNTRACED)
            if (waited_pid != target.pid or not os.WIFSTOPPED(status)
                    or os.WSTOPSIG(status) != signal.SIGSTOP):
                raise RuntimeError("event target did not stop before trial")
            for trial_index in range(1, EVENT_TRIALS + 1):
                clock_rows.append(_capture_clock_query(
                    raw_clock, trial_index, "before", device_id,
                    repository_root, env))
                os.kill(target_pid, signal.SIGCONT)
                if trial_index < EVENT_TRIALS:
                    waited_pid, status = os.waitpid(target.pid, os.WUNTRACED)
                    if (waited_pid != target.pid or not os.WIFSTOPPED(status)
                            or os.WSTOPSIG(status) != signal.SIGSTOP):
                        raise RuntimeError(
                            "event target did not stop after trial")
                else:
                    returncode = target.wait(timeout=60)
                clock_rows.append(_capture_clock_query(
                    raw_clock, trial_index, "after", device_id,
                    repository_root, env))
        except BaseException:
            _terminate_process_group(target, target_pid, target_pgid)
            raise
    if returncode != 0:
        raise RuntimeError(f"event target failed for {mode}")
    sample = parse_stdout(stdout_path, cell, mode, "event", device_id)
    clock_manifest = {"schema_version": SCHEMA_VERSION, "queries": []}
    for query in clock_rows:
        clock_manifest["queries"].append({
            "trial_index": query["trial_index"],
            "position": query["position"],
            "query_argv": query["query_argv"],
            "query_started_monotonic_raw_ns": query[
                "query_started_monotonic_raw_ns"],
            "query_ended_monotonic_raw_ns": query[
                "query_ended_monotonic_raw_ns"],
            "stdout_path": query["stdout_path"].relative_to(
                sample_dir).as_posix(),
            "stdout_sha256": _sha256(query["stdout_path"]),
            "stderr_path": query["stderr_path"].relative_to(
                sample_dir).as_posix(),
            "stderr_sha256": _sha256(query["stderr_path"]),
        })
    _write_json_exclusive(raw_clock / "manifest.json", clock_manifest)
    clock_evidence, bus_id = _verify_clock_evidence(
        sample_dir, sample, device_id)
    sample["stdout_path"] = stdout_path.name
    sample["stdout_sha256"] = _sha256(stdout_path)
    sample["stderr_path"] = stderr_path.name
    sample["stderr_sha256"] = _sha256(stderr_path)
    sample["active_clock"] = clock_evidence
    return sample, bus_id


def _capture_clock_query(
        raw_clock, trial_index, position, device_id, repository_root, env):
    if position not in CLOCK_POSITIONS:
        raise ValueError("clock position is invalid")
    query_stdout = raw_clock / (
        f"trial-{trial_index:02d}-{position}.stdout.txt")
    query_stderr = raw_clock / (
        f"trial-{trial_index:02d}-{position}.stderr.txt")
    query_argv = [
        str(MX_SMI_PATH), "--show-clock", "-i", str(device_id)]
    query_started = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    _run_checked(
        query_argv, query_stdout, query_stderr, repository_root, env)
    query_ended = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return {
        "trial_index": trial_index,
        "position": position,
        "query_argv": query_argv,
        "query_started_monotonic_raw_ns": query_started,
        "query_ended_monotonic_raw_ns": query_ended,
        "stdout_path": query_stdout,
        "stderr_path": query_stderr,
    }


def _verify_clock_evidence(sample_dir, sample, device_id):
    raw_clock = sample_dir / "active_clock"
    manifest_path = raw_clock / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (set(manifest) != {"schema_version", "queries"}
            or manifest["schema_version"] != SCHEMA_VERSION
            or not isinstance(manifest["queries"], list)
            or len(manifest["queries"])
            != EVENT_TRIALS * len(CLOCK_POSITIONS)):
        raise ValueError("active-clock manifest schema is invalid")
    expected_files = {"manifest.json"}
    evidence = []
    bus_id = None
    by_trial = {trial["trial_index"]: trial for trial in sample["trials"]}
    for ordinal, query in enumerate(manifest["queries"]):
        trial_index = ordinal // len(CLOCK_POSITIONS) + 1
        position = CLOCK_POSITIONS[ordinal % len(CLOCK_POSITIONS)]
        required = {
            "trial_index", "position", "query_argv",
            "query_started_monotonic_raw_ns", "query_ended_monotonic_raw_ns",
            "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
        }
        if (not isinstance(query, dict) or set(query) != required
                or query["trial_index"] != trial_index
                or query["position"] != position
                or query["query_argv"] != [
                    str(MX_SMI_PATH), "--show-clock", "-i", str(device_id)]):
            raise ValueError("active-clock query identity is invalid")
        started = query["query_started_monotonic_raw_ns"]
        ended = query["query_ended_monotonic_raw_ns"]
        if (not isinstance(started, int) or not isinstance(ended, int)
                or started >= ended):
            raise ValueError("active-clock query interval is invalid")
        paths = []
        for kind in ("stdout", "stderr"):
            relative = Path(query[f"{kind}_path"])
            if (relative.is_absolute() or ".." in relative.parts
                    or relative.parts[0] != "active_clock"):
                raise ValueError("active-clock raw path is invalid")
            path = sample_dir / relative
            if (not path.is_file()
                    or _sha256(path) != query[f"{kind}_sha256"]):
                raise ValueError("active-clock raw hash changed")
            expected_files.add(Path(*relative.parts[1:]).as_posix())
            paths.append(path)
        observed_bus, clock_mhz = _parse_clock(paths[0], device_id)
        if bus_id is None:
            bus_id = observed_bus
        elif bus_id != observed_bus:
            raise ValueError("active-clock bus identity changed")
        trial = by_trial[trial_index]
        if position == "before":
            interval_valid = ended <= trial["started_monotonic_raw_ns"]
        elif position == "after":
            interval_valid = started >= trial["ended_monotonic_raw_ns"]
        else:
            raise AssertionError(f"unhandled clock position: {position}")
        if not interval_valid:
            raise ValueError("active-clock query does not bracket event trial")
        if clock_mhz != TARGET_XCORE_CLOCK_MHZ:
            raise ValueError("bracketing active-clock sample is not 1125 MHz")
        evidence.append({
            **query,
            "query_interval_relation_to_event": position,
            "xcore_clock_mhz": clock_mhz,
        })
    actual_files = {
        path.relative_to(raw_clock).as_posix()
        for path in raw_clock.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("active-clock raw file inventory changed")
    return evidence, bus_id


def _validate_environment(args, repository_root):
    if COMMIT_PATTERN.fullmatch(args.expected_commit) is None:
        raise ValueError("expected commit must be a full lowercase object ID")
    if _git_commit(repository_root) != args.expected_commit:
        raise ValueError("deployed source commit differs from expected commit")
    if IMAGE_PATTERN.fullmatch(args.container_image_id) is None:
        raise ValueError("container image ID must be an exact SHA-256")
    for value, label in (
            (args.expected_mxcc_sha256, "mxcc"),
            (args.expected_mx_smi_sha256, "mx-smi")):
        if SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"expected {label} SHA-256 is invalid")
    if "MACA_PATH" not in os.environ or os.environ["MACA_PATH"] != str(MACA_PATH):
        raise ValueError("MACA_PATH must be exactly /opt/maca")
    for path, expected, label in (
            (MXCC_PATH, args.expected_mxcc_sha256, "mxcc"),
            (MX_SMI_PATH, args.expected_mx_smi_sha256, "mx-smi")):
        if (not path.is_file() or not os.access(path, os.X_OK)
                or _sha256(path) != expected):
            raise ValueError(f"{label} executable identity is invalid")
    return audit_source(repository_root / SOURCE_RELATIVE_PATH)


def smoke(args):
    if args.cell not in CELLS or args.device_id < 0:
        raise ValueError("cross-slot cell/device is invalid")
    repository_root = _repository_root()
    source = _validate_environment(args, repository_root)
    allowed = (repository_root / ALLOWED_OUTPUT_RELATIVE_PATH).resolve()
    output_root = _require_contained(args.output_root, allowed, "output root")
    if RUN_ID_PATTERN.fullmatch(args.run_id) is None:
        raise ValueError("run ID is invalid")
    artifact_root = output_root / args.run_id
    artifact_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    plan = build_plan(args.cell)
    _write_json_exclusive(artifact_root / "plan.json", plan)
    source_snapshot = artifact_root / "snapshots/phase_overlap.cpp"
    runner_snapshot = artifact_root / "snapshots/phase_overlap_runner.py"
    mx_smi_snapshot = artifact_root / "snapshots/mx-smi"
    mxcc_snapshot = artifact_root / "snapshots/mxcc"
    _copy_exclusive(source, source_snapshot)
    _copy_exclusive(repository_root / RUNNER_RELATIVE_PATH, runner_snapshot)
    _copy_exclusive(MX_SMI_PATH, mx_smi_snapshot)
    _copy_exclusive(MXCC_PATH, mxcc_snapshot)
    mx_smi_snapshot.chmod(MX_SMI_PATH.stat().st_mode & 0o777)
    mxcc_snapshot.chmod(MXCC_PATH.stat().st_mode & 0o777)
    version_stdout = artifact_root / "mx-smi-version.stdout.txt"
    version_stderr = artifact_root / "mx-smi-version.stderr.txt"
    _run_checked(
        [str(MX_SMI_PATH), "--version"], version_stdout, version_stderr,
        repository_root, env)
    matches = re.findall(
        r"(?<![0-9.])2\.3\.1(?![0-9.])",
        version_stdout.read_text(encoding="utf-8"))
    if matches != [EXPECTED_MX_SMI_VERSION]:
        raise ValueError("mx-smi version is not exactly 2.3.1")
    compiler_stdout = artifact_root / "mxcc-version.stdout.txt"
    compiler_stderr = artifact_root / "mxcc-version.stderr.txt"
    _run_checked(
        [str(MXCC_PATH), "--version"], compiler_stdout, compiler_stderr,
        repository_root, env)
    if re.findall(
            r"(?<![0-9.])3\.7\.2(?![0-9.])",
            compiler_stdout.read_text(encoding="utf-8")) != [
                EXPECTED_MACA_VERSION]:
        raise ValueError("mxcc version is not exactly MACA 3.7.2")
    binary = artifact_root / "phase_overlap"
    _run_checked([
        str(MXCC_PATH), "-x", "maca", "-offload-arch", "native",
        "--maca-path=/opt/maca", "-O3", "-std=c++17", "-o",
        str(binary), str(source)], artifact_root / "build.stdout.txt",
        artifact_root / "build.stderr.txt", repository_root, env)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError("mxcc did not produce an executable")
    preflight = _capture_idle_preflight(
        artifact_root / "preflight-smoke", args.device_id, None,
        repository_root, env)
    inspect_stdout = artifact_root / "inspect.stdout.csv"
    _run_checked(
        [str(binary), "--inspect", "--device-id", str(args.device_id)],
        inspect_stdout, artifact_root / "inspect.stderr.txt",
        repository_root, env)
    device = parse_inspection(inspect_stdout, args.device_id)
    samples = []
    bus_id = preflight["device_bus_id"]
    for mode in MODES:
        sample_dir = artifact_root / "samples" / mode
        sample_dir.mkdir(parents=True)
        sample, observed_bus = _capture_mode(
            binary, args.cell, mode, args.device_id, sample_dir,
            repository_root, env)
        if bus_id is None:
            bus_id = observed_bus
        elif observed_bus != bus_id:
            raise ValueError("device bus identity changed across modes")
        samples.append(sample)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "cell": args.cell,
        "source_commit": args.expected_commit,
        "container_image_id": args.container_image_id,
        "physical_device_id": args.device_id,
        "device_bus_id": bus_id,
        "device_inspection": device,
        "production_or_evaluation_duration_consumed": False,
        "mcprofiler_instrumented_duration_consumed": False,
        "plan_sha256": _sha256(artifact_root / "plan.json"),
        "source_sha256": _sha256(source_snapshot),
        "runner_sha256": _sha256(runner_snapshot),
        "binary_sha256": _sha256(binary),
        "compiler_sha256": args.expected_mxcc_sha256,
        "mx_smi_sha256": args.expected_mx_smi_sha256,
        "smoke_preflight": preflight,
        "active_xcore_clock_mhz": TARGET_XCORE_CLOCK_MHZ,
        "samples": samples,
        "event_only_diagnostic": decide(samples),
        "release_status": "counter_validation_pending",
    }
    _write_json_exclusive(artifact_root / "event_candidate.json", payload)
    event_manifest_path = artifact_root / "event_manifest.json"
    _write_json_exclusive(
        event_manifest_path,
        _manifest_payload(artifact_root, {event_manifest_path.name}))
    _verify_event_artifacts(artifact_root, exact_inventory=True)
    print(str(artifact_root))


def _verify_event_artifacts(artifact_root, exact_inventory):
    event_path = artifact_root / "event_candidate.json"
    manifest_path = artifact_root / "event_manifest.json"
    _verify_manifest(
        artifact_root, manifest_path, {manifest_path.name}, exact_inventory)
    event = json.loads(event_path.read_text(encoding="utf-8"))
    _require_exact_keys(event, {
        "schema_version", "hardware_id", "mechanism", "cell",
        "source_commit", "container_image_id", "physical_device_id",
        "device_bus_id", "device_inspection",
        "production_or_evaluation_duration_consumed",
        "mcprofiler_instrumented_duration_consumed", "plan_sha256",
        "source_sha256", "runner_sha256", "binary_sha256",
        "compiler_sha256", "mx_smi_sha256", "smoke_preflight",
        "active_xcore_clock_mhz", "samples", "event_only_diagnostic",
        "release_status"}, "event candidate")
    cell = event["cell"]
    if (event["schema_version"] != SCHEMA_VERSION or cell not in CELLS
            or event["hardware_id"] != HARDWARE_ID
            or event["mechanism"] != MECHANISM
            or IMAGE_PATTERN.fullmatch(event["container_image_id"]) is None
            or event["production_or_evaluation_duration_consumed"] is not False
            or event["mcprofiler_instrumented_duration_consumed"] is not False
            or event["release_status"] != "counter_validation_pending"):
        raise ValueError("event candidate identity is invalid")
    repository_root = _repository_root()
    if (_git_commit(repository_root) != event["source_commit"]
            or COMMIT_PATTERN.fullmatch(event["source_commit"]) is None):
        raise ValueError("event source commit differs from deployed checkout")
    plan_path = artifact_root / "plan.json"
    if (json.loads(plan_path.read_text(encoding="utf-8")) != build_plan(cell)
            or _sha256(plan_path) != event["plan_sha256"]):
        raise ValueError("event plan differs from the frozen contract")
    identity_paths = (
        (artifact_root / "snapshots/phase_overlap.cpp",
         repository_root / SOURCE_RELATIVE_PATH, event["source_sha256"],
         "source"),
        (artifact_root / "snapshots/phase_overlap_runner.py",
         repository_root / RUNNER_RELATIVE_PATH, event["runner_sha256"],
         "runner"),
        (artifact_root / "snapshots/mxcc", MXCC_PATH,
         event["compiler_sha256"], "mxcc"),
        (artifact_root / "snapshots/mx-smi", MX_SMI_PATH,
         event["mx_smi_sha256"], "mx-smi"),
        (artifact_root / "phase_overlap", artifact_root / "phase_overlap",
         event["binary_sha256"], "binary"),
    )
    for snapshot, deployed, expected_hash, label in identity_paths:
        if (SHA256_PATTERN.fullmatch(expected_hash) is None
                or not snapshot.is_file() or not deployed.is_file()
                or _sha256(snapshot) != expected_hash
                or _sha256(deployed) != expected_hash):
            raise ValueError(f"event {label} identity changed")
    audit_source(artifact_root / "snapshots/phase_overlap.cpp")
    mx_smi_versions = re.findall(
        r"(?<![0-9.])2\.3\.1(?![0-9.])",
        (artifact_root / "mx-smi-version.stdout.txt").read_text(
            encoding="utf-8"))
    mxcc_versions = re.findall(
        r"(?<![0-9.])3\.7\.2(?![0-9.])",
        (artifact_root / "mxcc-version.stdout.txt").read_text(
            encoding="utf-8"))
    if (mx_smi_versions != [EXPECTED_MX_SMI_VERSION]
            or mxcc_versions != [EXPECTED_MACA_VERSION]):
        raise ValueError("event tool version evidence changed")
    inspection = parse_inspection(
        artifact_root / "inspect.stdout.csv", event["physical_device_id"])
    if inspection != event["device_inspection"]:
        raise ValueError("event device inspection evidence changed")
    preflight = _verify_idle_preflight(
        artifact_root / "preflight-smoke", event["physical_device_id"],
        event["device_bus_id"])
    if (preflight != event["smoke_preflight"] or not preflight["idle"]
            or preflight["device_id"] != event["physical_device_id"]
            or preflight["device_bus_id"] != event["device_bus_id"]
            or preflight["gpu_utilization_pct"] != 0.0
            or preflight["hbm_throughput_mbytes_per_second"]
            > MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND
            or preflight["process_host_pids"] != []):
        raise ValueError("smoke idle preflight evidence changed")
    samples = []
    bus_id = None
    for mode in MODES:
        sample_dir = artifact_root / "samples" / mode
        stdout_path = sample_dir / "event.stdout.csv"
        sample = parse_stdout(
            stdout_path, cell, mode, "event", event["physical_device_id"])
        active_clock, observed_bus = _verify_clock_evidence(
            sample_dir, sample, event["physical_device_id"])
        if bus_id is None:
            bus_id = observed_bus
        elif bus_id != observed_bus:
            raise ValueError("event device bus differs across raw samples")
        stderr_path = sample_dir / "event.stderr.txt"
        sample["stdout_path"] = stdout_path.name
        sample["stdout_sha256"] = _sha256(stdout_path)
        sample["stderr_path"] = stderr_path.name
        sample["stderr_sha256"] = _sha256(stderr_path)
        sample["active_clock"] = active_clock
        samples.append(sample)
    decision = decide(samples)
    if (bus_id != event["device_bus_id"] or samples != event["samples"]
            or decision != event["event_only_diagnostic"]):
        raise ValueError("derived event candidate differs from raw evidence")
    return {"event": event, "samples": samples, "decision": decision}


def _write_replay_stable_profile_csv(output, raw_stdout):
    csv_lines = [
        line for line in raw_stdout.splitlines(keepends=True)
        if line.startswith((b"metadata,", b"sample,"))]
    if len(csv_lines) != 2:
        raise ValueError("profile target emitted an invalid canonical CSV")
    canonical_output = b"".join(csv_lines)
    if output.exists():
        if output.read_bytes() != canonical_output:
            raise ValueError("profile replay changed canonical stdout")
    else:
        with output.open("xb") as stdout:
            stdout.write(canonical_output)


def profile_target(args):
    repository_root = _repository_root()
    allowed = (repository_root / ALLOWED_OUTPUT_RELATIVE_PATH).resolve()
    binary = _require_contained(args.binary, allowed, "binary")
    output = _require_contained(args.output, allowed, "profile stdout")
    if (args.cell not in CELLS or args.mode not in MODES or args.device_id < 0
            or SHA256_PATTERN.fullmatch(args.expected_binary_sha256) is None
            or not binary.is_file() or not os.access(binary, os.X_OK)
            or _sha256(binary) != args.expected_binary_sha256):
        raise ValueError("profile target identity is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    target_environment = dict(os.environ)
    inherited_preload = ""
    if "LD_PRELOAD" in target_environment:
        inherited_preload = target_environment["LD_PRELOAD"]
    target_environment["LD_PRELOAD"] = str(TARGET_LIBSTDCXX_PATH)
    if inherited_preload:
        target_environment["LD_PRELOAD"] += f":{inherited_preload}"
    result = subprocess.run([
        str(binary), "--profile-once", "--device-id", str(args.device_id),
        "--cell", args.cell, "--mode", args.mode, "--attach-wait-seconds",
        str(ATTACH_WAIT_SECONDS)], cwd=repository_root,
        env=target_environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    sys.stderr.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    sys.stderr.buffer.flush()
    if result.returncode != 0:
        raise RuntimeError("profile target failed")
    _write_replay_stable_profile_csv(output, result.stdout)
    parse_stdout(output, args.cell, args.mode, "profile", args.device_id)


def _find_report(allk, cell, mode):
    candidates = []
    for path in sorted(allk.glob("*.txt.json")):
        if (path.name != "report.txt.json"
                and "period" not in path.name.lower()
                and NUMBERED_REPORT_PATTERN.fullmatch(path.name) is not None):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("profile must produce exactly one numbered report")
    parse_counter_report(candidates[0], cell, mode)
    return candidates[0]


def _profile_command_argv(
        contract, repository_root, artifact_root, binary, event, cell, mode):
    sample_dir = artifact_root / "samples" / mode
    profile_stdout = sample_dir / "profile.stdout.csv"
    target = [
        contract["python_executable"],
        str(repository_root / RUNNER_RELATIVE_PATH),
        "profile-target", "--binary", str(binary),
        "--expected-binary-sha256", event["binary_sha256"],
        "--output", str(profile_stdout), "--device-id",
        str(event["physical_device_id"]), "--cell", cell,
        "--mode", mode]
    return [
        contract["mcprofiler_runtime_path"], "perf_exec", "--cmdline",
        (f"export LD_PRELOAD={TARGET_LIBSTDCXX_PATH}"
         "${LD_PRELOAD:+:$LD_PRELOAD}; export MACA_PATH=/opt/maca; export "
         "LD_LIBRARY_PATH=/opt/maca/lib/:$LD_LIBRARY_PATH; "
         + shlex.join(target)), "--casename", f"{cell}_{mode}",
        "--output", str(sample_dir / "mcprofiler_out/_allk"),
        "--host", "127.0.0.1", "--port", str(contract["port"]),
        "--metrics", *PROFILE_METRICS, "--custom", "--per-kernel"]


def capture_profile(args):
    repository_root = _repository_root()
    allowed = (repository_root / ALLOWED_OUTPUT_RELATIVE_PATH).resolve()
    artifact_root = _require_contained(
        args.artifact_root, allowed, "artifact root")
    event_path = artifact_root / "event_candidate.json"
    verified_event = _verify_event_artifacts(
        artifact_root, exact_inventory=True)
    event = verified_event["event"]
    cell = event["cell"]
    binary = artifact_root / "phase_overlap"
    if (not args.mcprofiler.is_absolute() or not args.mcprofiler.is_file()
            or not os.access(args.mcprofiler, os.X_OK)
            or SHA256_PATTERN.fullmatch(args.expected_mcprofiler_sha256) is None
            or _sha256(args.mcprofiler) != args.expected_mcprofiler_sha256):
        raise ValueError("mcProfiler executable identity is invalid")
    if (not args.python_executable.is_absolute()
            or not args.python_executable.is_file()
            or not os.access(args.python_executable, os.X_OK)):
        raise ValueError("Python executable identity is invalid")
    if ("MACA_PATH" not in os.environ
            or os.environ["MACA_PATH"] != str(MACA_PATH)
            or "MACA_ETC_PATH" not in os.environ
            or not Path(os.environ["MACA_ETC_PATH"]).is_dir()):
        raise ValueError("mcProfiler environment is invalid")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("mcProfiler port is invalid")
    mcprofiler_runtime = args.mcprofiler.resolve(strict=True)
    python_executable = args.python_executable.resolve(strict=True)
    capture_contract_path = artifact_root / "capture_contract.json"
    _write_json_exclusive(
        capture_contract_path,
        _capture_contract_payload(
            mcprofiler_runtime, args.expected_mcprofiler_sha256,
            python_executable, args.port))
    capture_contract = _verify_capture_contract(capture_contract_path)
    mcprofiler_snapshot = artifact_root / "snapshots/mcProfiler"
    _copy_exclusive(mcprofiler_runtime, mcprofiler_snapshot)
    mcprofiler_snapshot.chmod(mcprofiler_runtime.stat().st_mode & 0o777)
    version_stdout = artifact_root / "mcprofiler-version.stdout.txt"
    version_stderr = artifact_root / "mcprofiler-version.stderr.txt"
    _run_checked(
        [str(mcprofiler_runtime), "version"], version_stdout, version_stderr,
        repository_root, dict(os.environ))
    mcprofiler_version = _parse_mcprofiler_version(
        version_stdout.read_text(encoding="utf-8"))
    lock_path = _host_global_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("host-global mcProfiler lock is held") from error
        acquired = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        profile_preflight = _capture_idle_preflight(
            artifact_root / "preflight-profile",
            event["physical_device_id"], event["device_bus_id"],
            repository_root, dict(os.environ))
        previous_end = None
        for mode in MODES:
            sample_dir = artifact_root / "samples" / mode
            profile_stdout = sample_dir / "profile.stdout.csv"
            allk = sample_dir / "mcprofiler_out/_allk"
            outer = _profile_command_argv(
                capture_contract, repository_root, artifact_root, binary,
                event, cell, mode)
            execution = _run_profile_checked(
                outer, sample_dir / "mcprofiler.stdout.txt",
                sample_dir / "mcprofiler.stderr.txt", repository_root,
                dict(os.environ))
            started = execution["started_monotonic_raw_ns"]
            ended = execution["ended_monotonic_raw_ns"]
            if previous_end is not None and started <= previous_end:
                raise ValueError("mcProfiler invocations overlap")
            previous_end = ended
            report = _find_report(allk, cell, mode)
            parse_stdout(
                profile_stdout, cell, mode, "profile",
                event["physical_device_id"])
            records.append({
                **parse_counter_report(report, cell, mode),
                "report_path": report.relative_to(artifact_root).as_posix(),
                "profile_stdout_path": profile_stdout.relative_to(
                    artifact_root).as_posix(),
                "profile_stdout_sha256": _sha256(profile_stdout),
                "mcprofiler_stdout_path": (
                    sample_dir / "mcprofiler.stdout.txt").relative_to(
                        artifact_root).as_posix(),
                "mcprofiler_stdout_sha256": _sha256(
                    sample_dir / "mcprofiler.stdout.txt"),
                "mcprofiler_stderr_path": (
                    sample_dir / "mcprofiler.stderr.txt").relative_to(
                        artifact_root).as_posix(),
                "mcprofiler_stderr_sha256": _sha256(
                    sample_dir / "mcprofiler.stderr.txt"),
                "command_argv": outer,
                "command_pid": execution["pid"],
                "command_pgid": execution["pgid"],
                "process_group_cleanup": execution[
                    "process_group_cleanup"],
                "started_monotonic_raw_ns": started,
                "ended_monotonic_raw_ns": ended,
            })
        released = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    if acquired >= released or len(records) != len(MODES):
        raise ValueError("host-global profile interval is invalid")
    if any(
            record["started_monotonic_raw_ns"] < acquired
            or record["ended_monotonic_raw_ns"] > released
            for record in records):
        raise ValueError("mcProfiler invocation escaped the host-global lock")
    final = {
        "schema_version": SCHEMA_VERSION,
        "hardware_id": HARDWARE_ID,
        "mechanism": MECHANISM,
        "cell": cell,
        "event_candidate_sha256": _sha256(event_path),
        "capture_contract_sha256": _sha256(capture_contract_path),
        "production_or_evaluation_duration_consumed": False,
        "mcprofiler_instrumented_duration_consumed": False,
        "mcprofiler": {
            "runtime_path": capture_contract["mcprofiler_runtime_path"],
            "executable_sha256": args.expected_mcprofiler_sha256,
            "snapshot_path": mcprofiler_snapshot.relative_to(
                artifact_root).as_posix(),
            "snapshot_sha256": _sha256(mcprofiler_snapshot),
            "version": mcprofiler_version,
            "version_stdout_path": version_stdout.relative_to(
                artifact_root).as_posix(),
            "version_stdout_sha256": _sha256(version_stdout),
            "version_stderr_path": version_stderr.relative_to(
                artifact_root).as_posix(),
            "version_stderr_sha256": _sha256(version_stderr),
        },
        "profile_preflight": profile_preflight,
        "host_global_lock": {
            "path": str(_host_global_lock_path()),
            "method": "fcntl.flock(LOCK_EX|LOCK_NB)",
            "holder_pid": capture_contract["holder_pid"],
            "acquired_monotonic_raw_ns": acquired,
            "released_monotonic_raw_ns": released,
        },
        "counter_validation": records,
        "decision": verified_event["decision"],
        "release_status": "candidate_only_not_canonical",
    }
    evidence_manifest_path = artifact_root / "evidence_manifest.json"
    report_path = artifact_root / "report.json"
    _write_json_exclusive(
        evidence_manifest_path,
        _manifest_payload(
            artifact_root,
            {evidence_manifest_path.name, report_path.name}))
    final["artifact_inventory"] = {
        "manifest_path": evidence_manifest_path.name,
        "manifest_sha256": _sha256(evidence_manifest_path),
        "report_anchor_path": "report.sha256",
        "detached_final_manifest_path": "final_manifest.json",
        "exact_inventory_required": True,
    }
    _write_json_exclusive(report_path, final)
    _write_report_anchor(report_path, artifact_root / "report.sha256")
    final_manifest_path = artifact_root / "final_manifest.json"
    _write_json_exclusive(
        final_manifest_path,
        _manifest_payload(artifact_root, {final_manifest_path.name}))
    final_manifest_sha256 = _sha256(final_manifest_path)
    _verify_final_artifacts(artifact_root, final_manifest_sha256)
    print(str(artifact_root / "report.json"))
    print(f"final_manifest_sha256={final_manifest_sha256}")


def _verify_final_artifacts(artifact_root, expected_final_manifest_sha256):
    if (not isinstance(expected_final_manifest_sha256, str)
            or SHA256_PATTERN.fullmatch(
                expected_final_manifest_sha256) is None):
        raise ValueError("expected detached final manifest SHA-256 is invalid")
    final_manifest_path = artifact_root / "final_manifest.json"
    if _sha256(final_manifest_path) != expected_final_manifest_sha256:
        raise ValueError("detached final manifest identity changed")
    _verify_manifest(
        artifact_root, final_manifest_path, {final_manifest_path.name}, True)
    report_path = artifact_root / "report.json"
    anchor_path = artifact_root / "report.sha256"
    _verify_report_anchor(report_path, anchor_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require_exact_keys(report, {
        "schema_version", "hardware_id", "mechanism", "cell",
        "event_candidate_sha256", "capture_contract_sha256",
        "production_or_evaluation_duration_consumed",
        "mcprofiler_instrumented_duration_consumed", "mcprofiler",
        "profile_preflight", "host_global_lock", "counter_validation",
        "decision", "release_status", "artifact_inventory"},
        "final report")
    _require_exact_keys(report["artifact_inventory"], {
        "manifest_path", "manifest_sha256", "report_anchor_path",
        "detached_final_manifest_path", "exact_inventory_required"},
        "artifact inventory")
    if (report["artifact_inventory"]["report_anchor_path"] != anchor_path.name
            or report["artifact_inventory"][
                "detached_final_manifest_path"] != final_manifest_path.name):
        raise ValueError("final report anchor path changed")
    manifest_path = _require_contained(
        artifact_root / report["artifact_inventory"]["manifest_path"],
        artifact_root, "final manifest")
    if (report["artifact_inventory"]["exact_inventory_required"] is not True
            or _sha256(manifest_path)
            != report["artifact_inventory"]["manifest_sha256"]):
        raise ValueError("final artifact inventory identity is invalid")
    _verify_manifest(
        artifact_root, manifest_path,
        {manifest_path.name, report_path.name, "report.sha256",
         final_manifest_path.name}, True)
    capture_contract_path = artifact_root / "capture_contract.json"
    if (not isinstance(report["capture_contract_sha256"], str)
            or SHA256_PATTERN.fullmatch(
                report["capture_contract_sha256"]) is None
            or _sha256(capture_contract_path)
            != report["capture_contract_sha256"]):
        raise ValueError("capture contract identity changed")
    capture_contract = _verify_capture_contract(capture_contract_path)
    verified_event = _verify_event_artifacts(
        artifact_root, exact_inventory=False)
    event = verified_event["event"]
    if (report["schema_version"] != SCHEMA_VERSION
            or report["hardware_id"] != HARDWARE_ID
            or report["mechanism"] != MECHANISM
            or report["cell"] != event["cell"]
            or report["production_or_evaluation_duration_consumed"] is not False
            or report["mcprofiler_instrumented_duration_consumed"] is not False
            or report["event_candidate_sha256"]
            != _sha256(artifact_root / "event_candidate.json")
            or report["decision"] != verified_event["decision"]
            or report["release_status"] != "candidate_only_not_canonical"):
        raise ValueError("final report differs from recomputed raw evidence")
    profiler = report["mcprofiler"]
    _require_exact_keys(profiler, {
        "runtime_path", "executable_sha256", "snapshot_path",
        "snapshot_sha256", "version", "version_stdout_path",
        "version_stdout_sha256", "version_stderr_path",
        "version_stderr_sha256"}, "mcProfiler provenance")
    if any(not isinstance(profiler[field], str) for field in profiler):
        raise ValueError("mcProfiler provenance value schema is invalid")
    profiler_snapshot = _require_contained(
        artifact_root / profiler["snapshot_path"], artifact_root,
        "mcProfiler snapshot")
    if (profiler["runtime_path"]
            != capture_contract["mcprofiler_runtime_path"]
            or profiler["executable_sha256"]
            != capture_contract["mcprofiler_executable_sha256"]
            or profiler["snapshot_path"] != "snapshots/mcProfiler"
            or profiler["version_stdout_path"]
            != "mcprofiler-version.stdout.txt"
            or profiler["version_stderr_path"]
            != "mcprofiler-version.stderr.txt"
            or SHA256_PATTERN.fullmatch(profiler["snapshot_sha256"]) is None
            or SHA256_PATTERN.fullmatch(
                profiler["version_stdout_sha256"]) is None
            or SHA256_PATTERN.fullmatch(
                profiler["version_stderr_sha256"]) is None
            or profiler["executable_sha256"] != profiler["snapshot_sha256"]
            or _sha256(profiler_snapshot) != profiler["snapshot_sha256"]):
        raise ValueError("preserved mcProfiler executable identity changed")
    version_stdout = _require_contained(
        artifact_root / profiler["version_stdout_path"], artifact_root,
        "mcProfiler version stdout")
    version_stderr = _require_contained(
        artifact_root / profiler["version_stderr_path"], artifact_root,
        "mcProfiler version stderr")
    if (_sha256(version_stdout) != profiler["version_stdout_sha256"]
            or _sha256(version_stderr) != profiler["version_stderr_sha256"]
            or _parse_mcprofiler_version(
                version_stdout.read_text(encoding="utf-8"))
            != profiler["version"]):
        raise ValueError("preserved mcProfiler version evidence changed")
    preflight = _verify_idle_preflight(
        artifact_root / "preflight-profile", event["physical_device_id"],
        event["device_bus_id"])
    if (not preflight["idle"]
            or preflight["device_id"] != event["physical_device_id"]
            or preflight["device_bus_id"] != event["device_bus_id"]
            or preflight["gpu_utilization_pct"] != 0.0
            or preflight["hbm_throughput_mbytes_per_second"]
            > MAX_IDLE_HBM_THROUGHPUT_MBYTES_PER_SECOND
            or preflight["process_host_pids"] != []
            or preflight != report["profile_preflight"]):
        raise ValueError("profile idle preflight evidence is invalid")
    lock = report["host_global_lock"]
    _require_exact_keys(lock, {
        "path", "method", "holder_pid", "acquired_monotonic_raw_ns",
        "released_monotonic_raw_ns"}, "host-global lock")
    if (not isinstance(lock["path"], str)
            or not isinstance(lock["method"], str)
            or not isinstance(lock["holder_pid"], int)
            or isinstance(lock["holder_pid"], bool)
            or not isinstance(lock["acquired_monotonic_raw_ns"], int)
            or isinstance(lock["acquired_monotonic_raw_ns"], bool)
            or not isinstance(lock["released_monotonic_raw_ns"], int)
            or isinstance(lock["released_monotonic_raw_ns"], bool)
            or lock["path"] != str(_host_global_lock_path())
            or lock["method"] != "fcntl.flock(LOCK_EX|LOCK_NB)"
            or lock["holder_pid"] != capture_contract["holder_pid"]
            or lock["acquired_monotonic_raw_ns"]
            >= lock["released_monotonic_raw_ns"]):
        raise ValueError("host-global lock evidence is invalid")
    for query in preflight["queries"].values():
        if (query["started_monotonic_raw_ns"]
                < lock["acquired_monotonic_raw_ns"]
                or query["ended_monotonic_raw_ns"]
                > lock["released_monotonic_raw_ns"]):
            raise ValueError("profile preflight escaped the host-global lock")
    records = report["counter_validation"]
    if (not isinstance(records, list) or len(records) != len(MODES)):
        raise ValueError("final counter evidence is incomplete")
    previous_end = None
    for mode, record in zip(MODES, records):
        _require_exact_keys(record, {
            "mode", "kernel_identity", "instruction_counts",
            "private_memory_spill_bytes", "report_path", "report_sha256",
            "instrumented_duration_consumed", "profile_stdout_path",
            "profile_stdout_sha256", "mcprofiler_stdout_path",
            "mcprofiler_stdout_sha256", "mcprofiler_stderr_path",
            "mcprofiler_stderr_sha256", "command_argv", "command_pid",
            "command_pgid", "process_group_cleanup",
            "started_monotonic_raw_ns", "ended_monotonic_raw_ns"},
            "counter validation record")
        sample_dir = artifact_root / "samples" / mode
        expected_command_argv = _profile_command_argv(
            capture_contract, _repository_root(), artifact_root,
            artifact_root / "phase_overlap", event, event["cell"], mode)
        if (not isinstance(record["command_argv"], list)
                or any(not isinstance(field, str)
                       for field in record["command_argv"])
                or record["command_argv"] != expected_command_argv):
            raise ValueError("mcProfiler command argv differs from contract")
        report_file = _find_report(
            sample_dir / "mcprofiler_out/_allk", event["cell"], mode)
        if report_file.relative_to(artifact_root).as_posix() != record[
                "report_path"]:
            raise ValueError("numbered report path changed")
        parsed = parse_counter_report(report_file, event["cell"], mode)
        for field in (
                "mode", "kernel_identity", "instruction_counts",
                "private_memory_spill_bytes", "report_sha256",
                "instrumented_duration_consumed"):
            if record[field] != parsed[field]:
                raise ValueError("final counter record differs from raw report")
        for prefix in ("profile_stdout", "mcprofiler_stdout",
                       "mcprofiler_stderr"):
            path = _require_contained(
                artifact_root / record[f"{prefix}_path"], artifact_root,
                f"{prefix} evidence")
            if _sha256(path) != record[f"{prefix}_sha256"]:
                raise ValueError("preserved profiler log hash changed")
        parse_stdout(
            artifact_root / record["profile_stdout_path"], event["cell"],
            mode, "profile", event["physical_device_id"])
        cleanup = record["process_group_cleanup"]
        _require_exact_keys(cleanup, {
            "pid", "pgid", "sigterm_sent", "sigkill_sent",
            "pgid_quiescent"}, "process-group cleanup")
        if (not isinstance(record["command_pid"], int)
                or isinstance(record["command_pid"], bool)
                or record["command_pid"] <= 0
                or not isinstance(record["command_pgid"], int)
                or isinstance(record["command_pgid"], bool)
                or not isinstance(record["started_monotonic_raw_ns"], int)
                or isinstance(record["started_monotonic_raw_ns"], bool)
                or not isinstance(record["ended_monotonic_raw_ns"], int)
                or isinstance(record["ended_monotonic_raw_ns"], bool)
                or not isinstance(cleanup["pid"], int)
                or isinstance(cleanup["pid"], bool)
                or not isinstance(cleanup["pgid"], int)
                or isinstance(cleanup["pgid"], bool)
                or record["command_pid"] != record["command_pgid"]
                or cleanup["pid"] != record["command_pid"]
                or cleanup["pgid"] != record["command_pgid"]
                or not isinstance(cleanup["sigterm_sent"], bool)
                or not isinstance(cleanup["sigkill_sent"], bool)
                or cleanup["pgid_quiescent"] is not True
                or record["started_monotonic_raw_ns"]
                >= record["ended_monotonic_raw_ns"]
                or record["started_monotonic_raw_ns"]
                < lock["acquired_monotonic_raw_ns"]
                or record["ended_monotonic_raw_ns"]
                > lock["released_monotonic_raw_ns"]
                or (previous_end is not None
                    and record["started_monotonic_raw_ns"] <= previous_end)):
            raise ValueError("profile process provenance is invalid")
        previous_end = record["ended_monotonic_raw_ns"]
    return {"report": report, **verified_event}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output-root", type=Path, required=True)
    smoke_parser.add_argument("--run-id", required=True)
    smoke_parser.add_argument("--cell", choices=CELLS, required=True)
    smoke_parser.add_argument("--device-id", type=int, required=True)
    smoke_parser.add_argument("--expected-commit", required=True)
    smoke_parser.add_argument("--expected-mxcc-sha256", required=True)
    smoke_parser.add_argument("--expected-mx-smi-sha256", required=True)
    smoke_parser.add_argument("--container-image-id", required=True)
    target_parser = subparsers.add_parser("profile-target")
    target_parser.add_argument("--binary", type=Path, required=True)
    target_parser.add_argument("--expected-binary-sha256", required=True)
    target_parser.add_argument("--output", type=Path, required=True)
    target_parser.add_argument("--device-id", type=int, required=True)
    target_parser.add_argument("--cell", choices=CELLS, required=True)
    target_parser.add_argument("--mode", choices=MODES, required=True)
    profile_parser = subparsers.add_parser("capture-profile")
    profile_parser.add_argument("--artifact-root", type=Path, required=True)
    profile_parser.add_argument("--mcprofiler", type=Path, required=True)
    profile_parser.add_argument(
        "--expected-mcprofiler-sha256", required=True)
    profile_parser.add_argument(
        "--python-executable", type=Path, required=True)
    profile_parser.add_argument("--port", type=int, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--artifact-root", type=Path, required=True)
    verify_parser.add_argument(
        "--expected-final-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args)
    elif args.command == "profile-target":
        profile_target(args)
    elif args.command == "capture-profile":
        capture_profile(args)
    elif args.command == "verify":
        repository_root = _repository_root()
        allowed = (repository_root / ALLOWED_OUTPUT_RELATIVE_PATH).resolve()
        artifact_root = _require_contained(
            args.artifact_root, allowed, "artifact root")
        _verify_final_artifacts(
            artifact_root, args.expected_final_manifest_sha256)
        print(str(artifact_root / "report.json"))
    else:
        raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
