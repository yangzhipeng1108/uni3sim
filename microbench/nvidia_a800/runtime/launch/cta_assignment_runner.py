#!/usr/bin/env python3
"""Capture and validate the A800 CTA-assignment mechanism probe."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


HARDWARE_ID = "nvidia_a800"
EXPECTED_DEVICE_NAME = "NVIDIA A800-SXM4-80GB"
EXPECTED_SM_COUNT = 108
EXPECTED_DYNAMIC_SHARED_BYTES = 96 * 1024
EXPECTED_BLOCK_THREADS = 128
EXPECTED_WARMUP_LAUNCHES = 3
SCENARIO = "uniform_control"
HETEROGENEOUS_SCENARIOS = (
    "alternating_8to1",
    "head_stragglers_16to1",
)
HETEROGENEOUS_GRID_SIZES = (216, 432, 864)
TAIL_RESIDUALS = (1, 27, 53, 107)
GRID_SIZES = (
    1, 27, 54, 81, 107, 108,
    109, 135, 161, 163, 189, 215, 216,
    217, 243, 269, 323, 324,
    431, 432, 433, 864,
)
BASE_WORK_ITERATIONS = 2048
SMOKE_CELL = (SCENARIO, 109)
SOURCE_NAME = "cta_assignment.cu"
RUNNER_NAME = "cta_assignment_runner.py"
NVIDIA_SMI_PATH = Path("/usr/bin/nvidia-smi")
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,96}")
RAW_HEADER = (
    "raw", "scenario", "grid_size", "trial", "block_index",
    "sm_id", "start_event_ticket", "end_event_ticket", "start_cycle",
    "end_cycle", "work_iterations", "work_cycles", "checksum",
)
REQUIRED_META = {
    "schema", "device_ordinal", "device_name", "device_uuid",
    "pci_bus_id", "compute_capability", "sm_count",
    "dynamic_shared_bytes_per_cta", "resident_ctas_per_sm",
    "block_threads", "warmup_launches", "warmup_complete", "capture_mode",
    "timer_domain",
    "cross_sm_clocks_comparable", "event_order_domain",
    "uniform_scenario_cta_program_identical",
    "production_or_evaluation_duration_consumed",
    "production_parameter_published",
}
DEVICE_QUERY_FIELDS = (
    "uuid,name,pci.bus_id,driver_version,pstate,memory.used,memory.total,"
    "utilization.gpu,utilization.memory,clocks.sm,clocks.mem"
)
EXPECTED_CAPTURE_ARTIFACTS = frozenset({
    "baseline_commit.txt",
    "build/compile.stderr",
    "build/compile.stdout",
    "build/cta_assignment",
    "build/cta_assignment.sass",
    "build/cuobjdump.stderr",
    "build/nvidia_smi_version.stderr",
    "build/nvidia_smi_version.txt",
    "build/nvcc_version.stderr",
    "build/nvcc_version.txt",
    "build/sass_audit.json",
    "candidate.json",
    "capture_identity.json",
    "device/post_native_device.csv",
    "device/post_native_device.stderr",
    "device/post_native_processes.csv",
    "device/post_native_processes.stderr",
    "device/pre_native_device.csv",
    "device/pre_native_device.stderr",
    "device/pre_native_processes.csv",
    "device/pre_native_processes.stderr",
    "native/native.stderr",
    "native/raw.csv",
    "postflight_device.json",
    "preflight_device.json",
    "provenance.json",
    f"source_snapshot/{RUNNER_NAME}",
    f"source_snapshot/{SOURCE_NAME}",
    "tracked_worktree_status.txt",
})


def _require_nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _parse_uint(text, label):
    normalized = _require_nonempty(text, label)
    if re.fullmatch(r"[0-9]+", normalized) is None:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(normalized)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, *, cwd, env=None, stdout_path=None, stderr_path=None):
    if not isinstance(command, list) or not command:
        raise TypeError("command must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in command):
        raise ValueError("command entries must be non-empty strings")
    stdout_stream = subprocess.PIPE
    stderr_stream = subprocess.PIPE
    opened = []
    if stdout_path is not None:
        stdout_stream = stdout_path.open("w", encoding="utf-8")
        opened.append(stdout_stream)
    if stderr_path is not None:
        stderr_stream = stderr_path.open("w", encoding="utf-8")
        opened.append(stderr_stream)
    try:
        return subprocess.run(
            command, cwd=cwd, env=env, text=True,
            stdout=stdout_stream, stderr=stderr_stream,
            check=True)
    finally:
        for stream in opened:
            stream.close()


def _git_output(repo_root, *arguments):
    return _run(
        ["git", *arguments], cwd=repo_root).stdout.strip()


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_manifest(output_dir):
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_dir)
        if relative in (Path("manifest.sha256"), Path("manifest_anchor.json")):
            continue
        entries.append(f"{_sha256(path)}  {relative}")
    manifest = output_dir / "manifest.sha256"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    anchor = {
        "schema": "sha256_manifest_anchor_v1",
        "manifest_path": "manifest.sha256",
        "manifest_sha256": _sha256(manifest),
        "entry_count": len(entries),
    }
    _write_json(output_dir / "manifest_anchor.json", anchor)
    return anchor


def _normalize_pci_bus_id(text):
    value = _require_nonempty(text, "PCI bus ID")
    match = re.fullmatch(
        r"([0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):"
        r"([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])",
        value)
    if match is None:
        raise ValueError(f"PCI bus ID has invalid syntax: {value!r}")
    domain = int(match.group(1), 16)
    bus = int(match.group(2), 16)
    device = int(match.group(3), 16)
    function = int(match.group(4), 16)
    if domain > 0xffff or device > 0x1f:
        raise ValueError(f"PCI bus ID is outside PCI address bounds: {value!r}")
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def _expected_work_iterations(scenario, block_index):
    if scenario == SCENARIO:
        return BASE_WORK_ITERATIONS
    if scenario == "alternating_8to1":
        return BASE_WORK_ITERATIONS * (1 if block_index % 2 == 0 else 8)
    if scenario == "head_stragglers_16to1":
        is_head_straggler = (
            block_index < EXPECTED_SM_COUNT and block_index % 4 == 0)
        return BASE_WORK_ITERATIONS * (16 if is_head_straggler else 1)
    raise ValueError(f"unknown scenario {scenario!r}")


def _expected_cells():
    return (
        tuple((SCENARIO, grid_size) for grid_size in GRID_SIZES)
        + tuple(
            (scenario, grid_size)
            for scenario in HETEROGENEOUS_SCENARIOS
            for grid_size in HETEROGENEOUS_GRID_SIZES))


def _validate_meta(meta, expected_mode):
    if set(meta) != REQUIRED_META:
        raise ValueError(
            "native meta schema mismatch: "
            f"missing={sorted(REQUIRED_META-set(meta))}, "
            f"extra={sorted(set(meta)-REQUIRED_META)}")
    expected_strings = {
        "schema": "nvidia_a800_cta_assignment_native_v2",
        "device_ordinal": "0",
        "device_name": EXPECTED_DEVICE_NAME,
        "compute_capability": "sm_80",
        "sm_count": str(EXPECTED_SM_COUNT),
        "dynamic_shared_bytes_per_cta": str(
            EXPECTED_DYNAMIC_SHARED_BYTES),
        "resident_ctas_per_sm": "1",
        "block_threads": str(EXPECTED_BLOCK_THREADS),
        "warmup_launches": str(EXPECTED_WARMUP_LAUNCHES),
        "warmup_complete": "true",
        "capture_mode": expected_mode,
        "timer_domain": "clock64_sm_local_cycles",
        "cross_sm_clocks_comparable": "false",
        "event_order_domain": "global_atomic_ticket",
        "uniform_scenario_cta_program_identical": "true",
        "production_or_evaluation_duration_consumed": "false",
        "production_parameter_published": "false",
    }
    for name, expected in expected_strings.items():
        if meta[name] != expected:
            raise ValueError(
                f"native meta {name!r} is {meta[name]!r}, expected "
                f"{expected!r}")
    if re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
            meta["device_uuid"]) is None:
        raise ValueError("native device UUID has invalid syntax")
    _normalize_pci_bus_id(meta["pci_bus_id"])


def parse_native_output(text, expected_trials=7, expected_mode="full_matrix"):
    """Strictly parse the one-process native matrix and raw CTA records."""

    if not isinstance(text, str) or not text:
        raise ValueError("native output must be non-empty text")
    if (isinstance(expected_trials, bool)
            or not isinstance(expected_trials, int)
            or expected_trials <= 0):
        raise ValueError("expected_trials must be a positive integer")
    if expected_mode not in ("full_matrix", "smoke_cell"):
        raise ValueError(f"unknown expected capture mode {expected_mode!r}")
    meta = {}
    records = {}
    raw_header = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        columns = tuple(next(csv.reader([line])))
        if columns[0] == "meta":
            if len(columns) != 3:
                raise ValueError(f"native meta row {line_number} is malformed")
            key = _require_nonempty(columns[1], "native meta key")
            if key in meta:
                raise ValueError(f"native meta repeats {key!r}")
            meta[key] = columns[2]
        elif columns[0] == "raw" and columns[1:] == RAW_HEADER[1:]:
            if raw_header is not None:
                raise ValueError("native output repeats the raw header")
            raw_header = columns
        elif columns[0] == "raw":
            if len(columns) != len(RAW_HEADER):
                raise ValueError(f"native raw row {line_number} is malformed")
            scenario = columns[1]
            grid_size = _parse_uint(columns[2], "grid size")
            trial = _parse_uint(columns[3], "trial")
            block_index = _parse_uint(columns[4], "block index")
            key = (scenario, grid_size, trial, block_index)
            if key in records:
                raise ValueError(f"native raw repeats {key!r}")
            records[key] = {
                "scenario": scenario,
                "grid_size": grid_size,
                "trial": trial,
                "block_index": block_index,
                "sm_id": _parse_uint(columns[5], "SM ID"),
                "start_event_ticket": _parse_uint(
                    columns[6], "start event ticket"),
                "end_event_ticket": _parse_uint(
                    columns[7], "end event ticket"),
                "start_cycle": _parse_uint(columns[8], "start cycle"),
                "end_cycle": _parse_uint(columns[9], "end cycle"),
                "work_iterations": _parse_uint(
                    columns[10], "work iterations"),
                "work_cycles": _parse_uint(columns[11], "work cycles"),
                "checksum": _parse_uint(columns[12], "checksum"),
            }
        else:
            raise ValueError(
                f"native output has unknown row at line {line_number}")
    _validate_meta(meta, expected_mode)
    if raw_header != RAW_HEADER:
        raise ValueError("native output lacks the exact raw header")

    if expected_mode == "full_matrix":
        expected_cells = set(_expected_cells())
    elif expected_mode == "smoke_cell":
        expected_cells = {SMOKE_CELL}
        if expected_trials != 1:
            raise ValueError("smoke-cell output requires exactly one trial")
    else:
        raise AssertionError(expected_mode)
    observed_cells = {
        (key[0], key[1]) for key in records
    }
    if observed_cells != expected_cells:
        raise ValueError(
            "native cell matrix mismatch: "
            f"missing={sorted(expected_cells-observed_cells)}, "
            f"extra={sorted(observed_cells-expected_cells)}")
    validated_cells = {}
    for scenario, grid_size in sorted(expected_cells):
        grid = grid_size
        for trial in range(expected_trials):
            cell_key = (scenario, grid_size, trial)
            cell_records = {
                block: records[(scenario, grid_size, trial, block)]
                for block in range(grid)
                if (scenario, grid_size, trial, block) in records
            }
            if set(cell_records) != set(range(grid)):
                raise ValueError(
                    f"native raw block matrix is incomplete for {cell_key!r}")
            _validate_cell_records(cell_key, cell_records, grid)
            validated_cells[cell_key] = tuple(
                cell_records[block] for block in range(grid))
    expected_record_count = sum(
        grid_size * expected_trials for _, grid_size in expected_cells)
    if len(records) != expected_record_count:
        raise ValueError(
            f"native returned {len(records)} raw rows, expected "
            f"{expected_record_count}")
    return {"meta": meta, "cells": validated_cells}


def _validate_cell_records(cell_key, records, grid):
    scenario, _, _ = cell_key
    events = {}
    for block, record in records.items():
        if record["sm_id"] >= EXPECTED_SM_COUNT:
            raise ValueError(f"SM ID is out of range for {cell_key!r}")
        if record["start_cycle"] >= record["end_cycle"]:
            raise ValueError(f"clock64 interval is not positive for {cell_key!r}")
        if record["work_cycles"] != (
                record["end_cycle"] - record["start_cycle"]):
            raise ValueError(f"work-cycle identity failed for {cell_key!r}")
        expected_work = _expected_work_iterations(scenario, block)
        if record["work_iterations"] != expected_work:
            raise ValueError(f"source work changed for {cell_key!r}")
        for kind in ("start", "end"):
            ticket = record[f"{kind}_event_ticket"]
            if ticket >= 2 * grid:
                raise ValueError(f"event ticket is out of range for {cell_key!r}")
            if ticket in events:
                raise ValueError(f"event ticket repeats for {cell_key!r}")
            events[ticket] = (kind, block)
        if record["start_event_ticket"] >= record["end_event_ticket"]:
            raise ValueError(f"CTA ends before it starts for {cell_key!r}")
    if set(events) != set(range(2 * grid)):
        raise ValueError(f"event ordering is incomplete for {cell_key!r}")

    ordered_starts = [
        records[block]
        for kind, block in (events[ticket] for ticket in range(2 * grid))
        if kind == "start"
    ]
    initial_count = min(grid, EXPECTED_SM_COUNT)
    initial_sm_ids = {
        record["sm_id"] for record in ordered_starts[:initial_count]
    }
    if len(initial_sm_ids) != initial_count:
        raise ValueError(
            f"first {initial_count} CTA start tickets must cover "
            f"{initial_count} unique SMs for "
            f"{cell_key!r}, observed {len(initial_sm_ids)}")

    active_blocks = set()
    active_sms = set()
    maximum_active = 0
    for ticket in range(2 * grid):
        kind, block = events[ticket]
        sm_id = records[block]["sm_id"]
        if kind == "start":
            if block in active_blocks or sm_id in active_sms:
                raise ValueError(
                    f"more than one CTA is resident on an SM for {cell_key!r}")
            active_blocks.add(block)
            active_sms.add(sm_id)
            maximum_active = max(maximum_active, len(active_blocks))
        elif kind == "end":
            if block not in active_blocks or sm_id not in active_sms:
                raise ValueError(f"event ordering is invalid for {cell_key!r}")
            active_blocks.remove(block)
            active_sms.remove(sm_id)
        else:
            raise AssertionError(kind)
    if active_blocks or active_sms or maximum_active > EXPECTED_SM_COUNT:
        raise ValueError(f"resident CTA accounting failed for {cell_key!r}")

    by_sm = {}
    for record in records.values():
        sm_id = record["sm_id"]
        if sm_id not in by_sm:
            by_sm[sm_id] = []
        by_sm[sm_id].append(record)
    for sm_id, local_records in by_sm.items():
        local_records.sort(key=lambda record: record["start_cycle"])
        for previous, following in zip(local_records, local_records[1:]):
            if previous["end_cycle"] >= following["start_cycle"]:
                raise ValueError(
                    f"same-SM clock64 intervals overlap on SM {sm_id} "
                    f"for {cell_key!r}")
            if (previous["end_event_ticket"]
                    >= following["start_event_ticket"]):
                raise ValueError(
                    f"same-SM event order disagrees with clock64 for "
                    f"{cell_key!r}")


def _nearest_rank_percentile(values, percentile):
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def _analyze_cell(records, uniform_assignment):
    grid = len(records)
    starts = sorted(records, key=lambda record: record["start_event_ticket"])
    initial_count = min(grid, EXPECTED_SM_COUNT)
    initial_cohort = starts[:initial_count]
    initial_fill_ticket = max(
        record["start_event_ticket"] for record in initial_cohort)
    first_initial_end_ticket = min(
        record["end_event_ticket"] for record in initial_cohort)
    initial_cohort_resident_together = (
        initial_fill_ticket < first_initial_end_ticket)
    start_tickets = sorted(record["start_event_ticket"] for record in records)
    by_sm = {}
    for record in records:
        sm_id = record["sm_id"]
        if sm_id not in by_sm:
            by_sm[sm_id] = []
        by_sm[sm_id].append(record)
    cta_counts = tuple(len(by_sm[sm_id]) if sm_id in by_sm else 0
                       for sm_id in range(EXPECTED_SM_COUNT))
    minimum_ctas_per_sm = min(cta_counts)
    maximum_ctas_per_sm = max(cta_counts)
    expected_minimum = grid // EXPECTED_SM_COUNT
    expected_maximum = (
        grid + EXPECTED_SM_COUNT - 1) // EXPECTED_SM_COUNT
    balanced_assignment = (
        maximum_ctas_per_sm - minimum_ctas_per_sm <= 1
        and minimum_ctas_per_sm == expected_minimum
        and maximum_ctas_per_sm == expected_maximum)

    events = {}
    for record in records:
        events[record["start_event_ticket"]] = ("start", record)
        events[record["end_event_ticket"]] = ("end", record)
    active_sms = set()
    started_count = 0
    pending_end_count = 0
    minimum_active_after_pending_end = None
    whole_wave_drain_count = 0
    for ticket in range(2 * grid):
        kind, record = events[ticket]
        sm_id = record["sm_id"]
        if kind == "start":
            active_sms.add(sm_id)
            started_count += 1
        elif kind == "end":
            active_sms.remove(sm_id)
            unstarted = grid - started_count
            if ticket > initial_fill_ticket and unstarted > 0:
                pending_end_count += 1
                active_after_end = len(active_sms)
                if minimum_active_after_pending_end is None:
                    minimum_active_after_pending_end = active_after_end
                else:
                    minimum_active_after_pending_end = min(
                        minimum_active_after_pending_end, active_after_end)
                if active_after_end == 0:
                    whole_wave_drain_count += 1
        else:
            raise AssertionError(kind)

    queue_exists = grid > EXPECTED_SM_COUNT
    no_whole_wave_barrier = (
        whole_wave_drain_count == 0 if queue_exists else None)
    gaps = []
    guaranteed_gaps = []
    guaranteed_without_successor = 0
    for local_records in by_sm.values():
        local_records.sort(key=lambda record: record["start_cycle"])
        for index, previous in enumerate(local_records):
            starts_before_end = sum(
                ticket < previous["end_event_ticket"]
                for ticket in start_tickets)
            unstarted_at_end_event = grid - starts_before_end
            successor = (
                local_records[index + 1]
                if index + 1 < len(local_records) else None)
            if successor is not None:
                gap = successor["start_cycle"] - previous["end_cycle"]
                gaps.append(gap)
            if unstarted_at_end_event >= EXPECTED_SM_COUNT:
                if successor is None:
                    guaranteed_without_successor += 1
                else:
                    guaranteed_gaps.append(gap)
    observed_sms = len(by_sm)
    expected_sms = min(grid, EXPECTED_SM_COUNT)
    full_wave_queue_transition_expected = grid >= 2 * EXPECTED_SM_COUNT
    full_wave_queue_assignment_pass = (
        guaranteed_without_successor == 0
        and (not full_wave_queue_transition_expected or bool(guaranteed_gaps)))
    uniform_structural_pass = (
        observed_sms == expected_sms
        and initial_cohort_resident_together
        and balanced_assignment
        and full_wave_queue_assignment_pass)
    heterogeneous_barrier_pass = (
        observed_sms == expected_sms
        and initial_cohort_resident_together
        and (not queue_exists or no_whole_wave_barrier)
        and full_wave_queue_assignment_pass)
    gap_p50 = _nearest_rank_percentile(gaps, 0.50) if gaps else None
    gap_p95 = _nearest_rank_percentile(gaps, 0.95) if gaps else None
    gap_max = max(gaps) if gaps else None
    guaranteed_p95 = (
        _nearest_rank_percentile(guaranteed_gaps, 0.95)
        if guaranteed_gaps else None)
    guaranteed_max = max(guaranteed_gaps) if guaranteed_gaps else None
    return {
        "observed_sm_count": observed_sms,
        "expected_active_sm_count": expected_sms,
        "physical_sm_count": EXPECTED_SM_COUNT,
        "active_sm_count_pass": observed_sms == expected_sms,
        "initial_cohort_resident_together": (
            initial_cohort_resident_together),
        "minimum_ctas_per_physical_sm": minimum_ctas_per_sm,
        "maximum_ctas_per_physical_sm": maximum_ctas_per_sm,
        "cta_count_spread": maximum_ctas_per_sm - minimum_ctas_per_sm,
        "expected_minimum_ctas_per_sm": expected_minimum,
        "expected_maximum_ctas_per_sm": expected_maximum,
        "balanced_assignment_pass": balanced_assignment,
        "assignment_invariant_work_waves": expected_maximum,
        "tail_cta_count": grid % EXPECTED_SM_COUNT,
        "queue_exists_after_initial_fill": queue_exists,
        "no_whole_wave_drain_while_queue_nonempty_in_event_order": (
            no_whole_wave_barrier),
        "queue_pending_end_event_count": pending_end_count,
        "minimum_active_ctas_after_pending_end_event": (
            minimum_active_after_pending_end),
        "whole_wave_drain_count": whole_wave_drain_count,
        "same_sm_boundary_gap_count": len(gaps),
        "same_sm_boundary_gap_p50_cycles": gap_p50,
        "same_sm_boundary_gap_p95_cycles": gap_p95,
        "same_sm_boundary_gap_max_cycles": gap_max,
        "guaranteed_pending_transition_count": len(guaranteed_gaps),
        "full_wave_queue_transition_expected": (
            full_wave_queue_transition_expected),
        "guaranteed_pending_without_local_successor": (
            guaranteed_without_successor),
        "no_structurally_unassigned_sm_while_full_wave_queued": (
            full_wave_queue_assignment_pass),
        "guaranteed_pending_boundary_gap_p95_cycles": guaranteed_p95,
        "guaranteed_pending_boundary_gap_max_cycles": guaranteed_max,
        "uniform_tail_assignment_pass": (
            uniform_structural_pass if uniform_assignment else None),
        "heterogeneous_barrier_control_pass": (
            None if uniform_assignment else heterogeneous_barrier_pass),
    }


def analyze_assignment(native, raw_sha256):
    """Create a mechanism-only candidate; never publish a Forward parameter."""

    if re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None:
        raise ValueError("raw SHA256 must be 64 lowercase hexadecimal digits")
    cell_analysis = {}
    uniform_assignment_pass = True
    barrier_control_pass = True
    for cell_key, records in native["cells"].items():
        scenario, grid_size, trial = cell_key
        is_uniform = scenario == SCENARIO
        analysis = _analyze_cell(records, is_uniform)
        name = f"{scenario}:grid{grid_size}:t{trial}"
        cell_analysis[name] = analysis
        if is_uniform:
            uniform_assignment_pass = (
                uniform_assignment_pass
                and analysis["uniform_tail_assignment_pass"])
        elif scenario in HETEROGENEOUS_SCENARIOS:
            barrier_control_pass = (
                barrier_control_pass
                and analysis["heterogeneous_barrier_control_pass"])
        else:
            raise ValueError(f"unknown parsed scenario {scenario!r}")

    if native["meta"]["capture_mode"] == "full_matrix":
        full_matrix = True
    elif native["meta"]["capture_mode"] == "smoke_cell":
        full_matrix = False
    else:
        raise ValueError(
            f"unknown native capture mode {native['meta']['capture_mode']!r}")
    passed = full_matrix and uniform_assignment_pass and barrier_control_pass
    observed_gaps = [
        analysis["guaranteed_pending_boundary_gap_max_cycles"]
        for analysis in cell_analysis.values()
        if analysis["guaranteed_pending_boundary_gap_max_cycles"] is not None]
    instrumented_boundary_diagnostic = {
        "name": "queued_cta_instrumented_boundary_gap",
        "maximum_observed_value": max(observed_gaps) if observed_gaps else None,
        "unit": "SM_cycle",
        "statistic": (
            "maximum observed same-SM boundary gap over every transition "
            "whose end event had at least 108 not-yet-started CTAs"),
        "instrumentation_bound": (
            "The measured boundary begins before the predecessor end-event "
            "atomic and record store and ends after the successor CTA entry "
            "and start-event atomic. All probe metadata, predecessor teardown, "
            "hardware assignment, and successor entry are included, so the "
            "un-instrumented scheduler-only handoff is no greater than this "
            "observed value for each sampled transition."),
        "allowed_use": "diagnostic only; not a Forward parameter",
        "forbidden_interpretation": (
            "bare kernel launch latency, bare CTA reissue latency, or a "
            "cross-SM timestamp"),
    }
    return {
        "schema": "nvidia_a800_cta_assignment_candidate_v3",
        "evidence_status": (
            "mechanism_observation" if passed else "evidence_pending"),
        "capture_mode": native["meta"]["capture_mode"],
        "mechanism": "uniform_one_resident_cta_tail_assignment",
        "raw_sha256": raw_sha256,
        "timer_domain": "clock64_sm_local_cycles",
        "cross_sm_clock_values_compared": False,
        "event_order_domain": "global_atomic_ticket",
        "event_order_interpretation": (
            "detects whether every active CTA records an end before any "
            "queued CTA records a start; it is not an instantaneous hardware "
            "occupancy, utilization, or latency measurement"),
        "interval_interpretation": (
            "work window including probe synchronization; not kernel or "
            "production duration"),
        "boundary_gap_interpretation": (
            "same-SM clock64 distance including end metadata, CTA teardown, "
            "scheduler assignment, CTA entry, and start metadata; not bare "
            "reissue latency"),
        "matrix_contract": {
            "physical_sm_count": EXPECTED_SM_COUNT,
            "exact_grid_sizes": list(GRID_SIZES),
            "tail_residuals": list(TAIL_RESIDUALS),
            "heterogeneous_barrier_control_grid_sizes": list(
                HETEROGENEOUS_GRID_SIZES),
            "work_iterations_per_cta": BASE_WORK_ITERATIONS,
            "resident_ctas_per_sm": 1,
        },
        "candidate_scope": (
            "observed assignment-invariant work-wave count for this frozen "
            "dummy CTA program at one resident CTA per SM: enabled-SM count "
            "and balanced CTA counts, with separate heterogeneous dummy-work "
            "controls used only to detect a scheduler-wide whole-wave drain"),
        "supported_conclusion": (
            "for this exact dummy CTA program and resource contract, the "
            "longest per-SM assignment contains ceil(grid/108) CTAs; this is "
            "a work-count observation, not an elapsed-time model"),
        "forward_admissibility": "evidence_pending",
        "cross_program_transfer_supported": False,
        "missing_minimum_mechanism_evidence": (
            "an independent authoritative hardware/compiler source proving "
            "that A800 block assignment and whole-wave-barrier behavior depend "
            "only on launch resource/residency state and not on the CTA "
            "instruction or data-flow program; without that evidence this "
            "dummy-program observation cannot be consumed by FlashAttention "
            "Forward"),
        "unsupported_conclusions": (
            "different CTA programs or resource footprints, bare launch or "
            "reissue latency, cross-SM timestamp ordering, any FlashAttention "
            "duration, any FlashAttention Forward schedule, or any empirical "
            "Forward coefficient"),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
        "uniform_assignment_matrix_pass": uniform_assignment_pass,
        "heterogeneous_barrier_control_matrix_pass": (
            barrier_control_pass if full_matrix else None),
        "instrumented_boundary_diagnostic": instrumented_boundary_diagnostic,
        "cell_analysis": cell_analysis,
    }


def validate_sass(text):
    if not isinstance(text, str) or not text:
        raise ValueError("SASS must be non-empty text")
    match = re.search(
        r"Function\s*:\s*cta_assignment_probe\b(?P<body>.*?)(?=\n\s*Function\s*:|\Z)",
        text, flags=re.DOTALL)
    if match is None:
        raise ValueError("SASS lacks cta_assignment_probe")
    body = match.group("body")
    clock_reads = len(re.findall(r"SR_CLOCKLO\b", body))
    global_timer_reads = len(re.findall(r"SR_GLOBALTIMER", body))
    local_memory_ops = len(re.findall(r"\b(?:LDL|STL)(?:\.|\s)", body))
    if clock_reads != 2:
        raise ValueError(
            "cta_assignment_probe must contain exactly two clock64 reads")
    if global_timer_reads != 0:
        raise ValueError("cta_assignment_probe must not read globaltimer")
    if local_memory_ops != 0:
        raise ValueError("cta_assignment_probe spills into local memory")
    return {
        "clock64_read_count": clock_reads,
        "globaltimer_read_count": global_timer_reads,
        "local_memory_instruction_count": local_memory_ops,
    }


def validate_ptxas(text):
    if not isinstance(text, str) or not text:
        raise ValueError("PTXAS output must be non-empty text")
    if "Function properties for cta_assignment_probe" not in text:
        raise ValueError("PTXAS output lacks cta_assignment_probe")
    spills = re.findall(
        r"([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads", text)
    if len(spills) != 1:
        raise ValueError("PTXAS output must contain one probe spill record")
    if spills[0] != ("0", "0"):
        raise ValueError("cta_assignment_probe has register spills")


def _parse_device_row(text, physical_identity):
    rows = tuple(csv.reader(text.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 11:
        raise ValueError("nvidia-smi returned an unexpected device row")
    row = tuple(value.strip() for value in rows[0])
    integers = {
        "memory_used_mib": _parse_uint(row[5], "memory used"),
        "memory_total_mib": _parse_uint(row[6], "memory total"),
        "utilization_gpu_pct": _parse_uint(row[7], "GPU utilization"),
        "utilization_memory_pct": _parse_uint(
            row[8], "memory utilization"),
        "sm_clock_mhz": _parse_uint(row[9], "SM clock"),
        "memory_clock_mhz": _parse_uint(row[10], "memory clock"),
    }
    if row[1] != EXPECTED_DEVICE_NAME:
        raise RuntimeError(
            f"selected device is {row[1]!r}, expected {EXPECTED_DEVICE_NAME!r}")
    return {
        "physical_identity": physical_identity,
        "uuid": row[0],
        "name": row[1],
        "pci_bus_id": row[2],
        "driver_version": row[3],
        "pstate": row[4],
        **integers,
    }


def _parse_nvidia_smi_version(text):
    match = re.search(
        r"^NVIDIA System Management Interface -- v([0-9]+(?:\.[0-9]+)+)$",
        text, flags=re.MULTILINE)
    if match is None:
        raise ValueError("nvidia-smi help does not expose a version")
    return match.group(1)


def _run_preserved(command, *, cwd, stdout_path, stderr_path):
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            "command failed: "
            f"command={command!r}, returncode={result.returncode}, "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}")
    return result.stdout


def _query_device(
        physical_identity, cwd, *, stdout_path, stderr_path):
    text = _run_preserved([
        str(NVIDIA_SMI_PATH), f"--id={physical_identity}",
        f"--query-gpu={DEVICE_QUERY_FIELDS}",
        "--format=csv,noheader,nounits",
    ], cwd=cwd, stdout_path=stdout_path, stderr_path=stderr_path)
    return _parse_device_row(text, physical_identity)


def _query_compute_processes(cwd, *, stdout_path, stderr_path):
    return _run_preserved([
        str(NVIDIA_SMI_PATH), "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ], cwd=cwd, stdout_path=stdout_path, stderr_path=stderr_path)


def _require_device_identity(device, expected_uuid, expected_pci):
    if device["uuid"] != expected_uuid:
        raise ValueError("nvidia-smi UUID disagrees with the requested device")
    if (_normalize_pci_bus_id(device["pci_bus_id"])
            != _normalize_pci_bus_id(expected_pci)):
        raise ValueError("nvidia-smi PCI bus ID disagrees with the request")


def _active_compute_pids(processes, expected_uuid):
    active_pids = []
    for line in processes.splitlines():
        if not line.strip() or line.startswith("No running processes found"):
            continue
        columns = tuple(value.strip() for value in line.split(","))
        if len(columns) != 2:
            raise ValueError("nvidia-smi compute-process row is malformed")
        if columns[1] == expected_uuid:
            active_pids.append(columns[0])
    return active_pids


def _require_idle_device(device, expected_uuid, expected_pci, processes):
    _require_device_identity(device, expected_uuid, expected_pci)
    if (device["memory_used_mib"] != 0
            or device["utilization_gpu_pct"] != 0):
        raise RuntimeError(
            "requested GPU is not idle: "
            f"memory={device['memory_used_mib']} MiB, "
            f"utilization={device['utilization_gpu_pct']}%")
    active_pids = _active_compute_pids(processes, expected_uuid)
    if active_pids:
        raise RuntimeError(
            f"requested GPU has active compute processes: {active_pids}")


def _require_released_device(device, expected_uuid, expected_pci, processes):
    """Require capture teardown without rejecting rolling utilization.

    ``nvidia-smi`` utilization is a recent sampling window, so the probe's own
    kernel can legitimately remain visible immediately after its process has
    exited.  Postflight therefore proves release with zero allocated memory
    and no compute PID; only the preflight query is an idle admission check.
    """

    _require_device_identity(device, expected_uuid, expected_pci)
    if device["memory_used_mib"] != 0:
        raise RuntimeError(
            "requested GPU was not released after capture: "
            f"memory={device['memory_used_mib']} MiB")
    active_pids = _active_compute_pids(processes, expected_uuid)
    if active_pids:
        raise RuntimeError(
            "requested GPU retains compute processes after capture: "
            f"{active_pids}")


def _normalized_capture_argv(arguments, repo_root, output_dir):
    command = [
        RUNNER_NAME,
        "capture",
        "--output-dir", str(output_dir),
        "--run-id", arguments.run_id,
        "--device-uuid", arguments.device_uuid,
        "--pci-bus-id", _normalize_pci_bus_id(arguments.pci_bus_id),
        "--repo-root", str(repo_root),
        "--nvcc", str(arguments.nvcc.resolve()),
        "--cuobjdump", str(arguments.cuobjdump.resolve()),
    ]
    if arguments.smoke_cell:
        command.append("--smoke-cell")
    else:
        command.extend(("--trials", str(arguments.trials)))
    return command


def capture(arguments, invocation_argv):
    repo_root = arguments.repo_root.resolve()
    output_dir = arguments.output_dir.resolve()
    allowed_root = (
        repo_root / "profiler_output" / "microbench" / HARDWARE_ID).resolve()
    if allowed_root not in output_dir.parents:
        raise ValueError(
            f"output directory must be a child of {allowed_root}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = (
        repo_root / "microbench" / HARDWARE_ID / "runtime" / "launch"
        / SOURCE_NAME)
    runner = source.with_name(RUNNER_NAME)
    nvcc_path = arguments.nvcc.resolve()
    cuobjdump_path = arguments.cuobjdump.resolve()
    tool_paths = (
        source, runner, nvcc_path, cuobjdump_path,
        NVIDIA_SMI_PATH)
    for path in tool_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = _git_output(repo_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", baseline) is None:
        raise ValueError("git HEAD is not a full commit identity")
    for path in (source, runner):
        relative = str(path.relative_to(repo_root))
        if _git_output(repo_root, "ls-files", "--error-unmatch", relative) != relative:
            raise RuntimeError(f"capture source is not tracked: {relative}")
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=repo_root, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"capture source differs from baseline {baseline}: {relative}")

    output_dir.mkdir(parents=True)
    build_dir = output_dir / "build"
    native_dir = output_dir / "native"
    snapshot_dir = output_dir / "source_snapshot"
    device_dir = output_dir / "device"
    for directory in (build_dir, native_dir, snapshot_dir, device_dir):
        directory.mkdir()
    snapshot_source = snapshot_dir / SOURCE_NAME
    snapshot_runner = snapshot_dir / RUNNER_NAME
    shutil.copy2(source, snapshot_source)
    shutil.copy2(runner, snapshot_runner)
    normalized_capture_argv = _normalized_capture_argv(
        arguments, repo_root, output_dir)
    _write_json(output_dir / "capture_identity.json", {
        "schema": "nvidia_a800_cta_assignment_capture_identity_v1",
        "hardware_id": HARDWARE_ID,
        "baseline_commit": baseline,
        "run_id": arguments.run_id,
        "device_uuid": arguments.device_uuid,
        "pci_bus_id": _normalize_pci_bus_id(arguments.pci_bus_id),
        "invocation_argv": invocation_argv,
        "normalized_capture_argv": normalized_capture_argv,
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "output_path": str(output_dir),
        "capture_mode": (
            "smoke_cell" if arguments.smoke_cell else "full_matrix"),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    (output_dir / "baseline_commit.txt").write_text(
        baseline + "\n", encoding="utf-8")
    (output_dir / "tracked_worktree_status.txt").write_text(
        _git_output(repo_root, "status", "--short") + "\n",
        encoding="utf-8")

    nvcc_version = build_dir / "nvcc_version.txt"
    _run(
        [str(nvcc_path), "--version"], cwd=repo_root,
        stdout_path=nvcc_version,
        stderr_path=build_dir / "nvcc_version.stderr")
    nvidia_smi_version = build_dir / "nvidia_smi_version.txt"
    _run(
        [str(NVIDIA_SMI_PATH), "-h"], cwd=repo_root,
        stdout_path=nvidia_smi_version,
        stderr_path=build_dir / "nvidia_smi_version.stderr")
    nvidia_smi_version_text = _parse_nvidia_smi_version(
        nvidia_smi_version.read_text(encoding="utf-8"))
    binary = build_dir / "cta_assignment"
    compile_argv = [
        str(nvcc_path), "-O3", "-std=c++17", "-arch=sm_80",
        "-Xptxas=-v", str(snapshot_source), "-o", str(binary),
    ]
    ptxas_stderr = build_dir / "compile.stderr"
    _run(
        compile_argv, cwd=repo_root,
        stdout_path=build_dir / "compile.stdout",
        stderr_path=ptxas_stderr)
    validate_ptxas(ptxas_stderr.read_text(encoding="utf-8"))
    sass_path = build_dir / "cta_assignment.sass"
    cuobjdump_argv = [
        str(cuobjdump_path), "--dump-sass", str(binary)]
    _run(
        cuobjdump_argv, cwd=repo_root,
        stdout_path=sass_path,
        stderr_path=build_dir / "cuobjdump.stderr")
    sass_audit = validate_sass(sass_path.read_text(encoding="utf-8"))
    _write_json(build_dir / "sass_audit.json", sass_audit)

    execution_env = os.environ.copy()
    execution_env["CUDA_VISIBLE_DEVICES"] = arguments.device_uuid
    raw_path = native_dir / "raw.csv"
    capture_mode = "smoke_cell" if arguments.smoke_cell else "full_matrix"
    trials = 1 if arguments.smoke_cell else arguments.trials
    native_argv = [str(binary), "--trials", str(trials)]
    if arguments.smoke_cell:
        native_argv.append("--smoke-cell")
    pre_device_raw = device_dir / "pre_native_device.csv"
    pre_device_stderr = device_dir / "pre_native_device.stderr"
    pre_process_raw = device_dir / "pre_native_processes.csv"
    pre_process_stderr = device_dir / "pre_native_processes.stderr"
    preflight = _query_device(
        arguments.device_uuid, repo_root,
        stdout_path=pre_device_raw, stderr_path=pre_device_stderr)
    pre_processes = _query_compute_processes(
        repo_root, stdout_path=pre_process_raw,
        stderr_path=pre_process_stderr)
    _require_idle_device(
        preflight, arguments.device_uuid, arguments.pci_bus_id,
        pre_processes)
    _run(
        native_argv, cwd=repo_root, env=execution_env,
        stdout_path=raw_path, stderr_path=native_dir / "native.stderr")
    post_device_raw = device_dir / "post_native_device.csv"
    post_device_stderr = device_dir / "post_native_device.stderr"
    post_process_raw = device_dir / "post_native_processes.csv"
    post_process_stderr = device_dir / "post_native_processes.stderr"
    postflight = _query_device(
        arguments.device_uuid, repo_root,
        stdout_path=post_device_raw, stderr_path=post_device_stderr)
    post_processes = _query_compute_processes(
        repo_root, stdout_path=post_process_raw,
        stderr_path=post_process_stderr)
    _require_released_device(
        postflight, arguments.device_uuid, arguments.pci_bus_id,
        post_processes)
    raw_sha256 = _sha256(raw_path)
    native = parse_native_output(
        raw_path.read_text(encoding="utf-8"), trials, capture_mode)
    if native["meta"]["device_uuid"] != arguments.device_uuid:
        raise ValueError("native CUDA UUID disagrees with preflight")
    if (_normalize_pci_bus_id(native["meta"]["pci_bus_id"])
            != _normalize_pci_bus_id(arguments.pci_bus_id)):
        raise ValueError("native CUDA PCI bus ID disagrees with preflight")
    candidate = analyze_assignment(native, raw_sha256)
    _write_json(output_dir / "candidate.json", candidate)

    if (postflight["uuid"] != preflight["uuid"]
            or _normalize_pci_bus_id(postflight["pci_bus_id"])
            != _normalize_pci_bus_id(preflight["pci_bus_id"])):
        raise RuntimeError("postflight device identity changed")
    _write_json(output_dir / "preflight_device.json", preflight)
    _write_json(output_dir / "postflight_device.json", postflight)
    _write_json(output_dir / "provenance.json", {
        "schema": "nvidia_a800_cta_assignment_provenance_v3",
        "hardware_id": HARDWARE_ID,
        "baseline_commit": baseline,
        "capture_mode": capture_mode,
        "trials": trials,
        "source": {
            "path": str(snapshot_source.relative_to(output_dir)),
            "sha256": _sha256(snapshot_source),
        },
        "runner": {
            "path": str(snapshot_runner.relative_to(output_dir)),
            "sha256": _sha256(snapshot_runner),
        },
        "binary": {
            "path": str(binary.relative_to(output_dir)),
            "sha256": _sha256(binary),
        },
        "compiler": {
            "path": str(nvcc_path),
            "sha256": _sha256(nvcc_path),
            "version_output_path": str(nvcc_version.relative_to(output_dir)),
            "version_output_sha256": _sha256(nvcc_version),
        },
        "nvidia_smi": {
            "path": str(NVIDIA_SMI_PATH),
            "sha256": _sha256(NVIDIA_SMI_PATH.resolve()),
            "version": nvidia_smi_version_text,
            "version_output_path": str(
                nvidia_smi_version.relative_to(output_dir)),
            "version_output_sha256": _sha256(nvidia_smi_version),
        },
        "binary_audit": {
            "cuobjdump_path": str(cuobjdump_path),
            "cuobjdump_sha256": _sha256(cuobjdump_path),
            "sass_path": str(sass_path.relative_to(output_dir)),
            "sass_sha256": _sha256(sass_path),
            "sass_audit": sass_audit,
        },
        "device_preflight": preflight,
        "device_postflight": postflight,
        "device_evidence": {
            "pre_native_device_raw": {
                "path": str(pre_device_raw.relative_to(output_dir)),
                "sha256": _sha256(pre_device_raw),
            },
            "pre_native_device_stderr": {
                "path": str(pre_device_stderr.relative_to(output_dir)),
                "sha256": _sha256(pre_device_stderr),
            },
            "pre_native_processes_raw": {
                "path": str(pre_process_raw.relative_to(output_dir)),
                "sha256": _sha256(pre_process_raw),
            },
            "pre_native_processes_stderr": {
                "path": str(pre_process_stderr.relative_to(output_dir)),
                "sha256": _sha256(pre_process_stderr),
            },
            "post_native_device_raw": {
                "path": str(post_device_raw.relative_to(output_dir)),
                "sha256": _sha256(post_device_raw),
            },
            "post_native_device_stderr": {
                "path": str(post_device_stderr.relative_to(output_dir)),
                "sha256": _sha256(post_device_stderr),
            },
            "post_native_processes_raw": {
                "path": str(post_process_raw.relative_to(output_dir)),
                "sha256": _sha256(post_process_raw),
            },
            "post_native_processes_stderr": {
                "path": str(post_process_stderr.relative_to(output_dir)),
                "sha256": _sha256(post_process_stderr),
            },
        },
        "capture_argv": normalized_capture_argv,
        "compile_argv": compile_argv,
        "cuobjdump_argv": cuobjdump_argv,
        "native_argv": native_argv,
        "raw": {
            "path": str(raw_path.relative_to(output_dir)),
            "sha256": raw_sha256,
        },
        "candidate_evidence_status": candidate["evidence_status"],
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    return _write_manifest(output_dir)


def _read_json_object(path, label):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _safe_artifact_path(capture_dir, relative_text):
    relative = Path(_require_nonempty(relative_text, "artifact path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe capture artifact path: {relative}")
    path = (capture_dir / relative).resolve()
    if capture_dir != path and capture_dir not in path.parents:
        raise ValueError(f"capture artifact escapes root: {relative}")
    return relative, path


def _parse_manifest(capture_dir):
    manifest_path = capture_dir / "manifest.sha256"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("manifest must not be empty")
    entries = {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"manifest line {line_number} is malformed")
        relative, path = _safe_artifact_path(capture_dir, match.group(2))
        relative_text = str(relative)
        if relative_text in entries:
            raise ValueError(f"manifest repeats {relative_text!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != match.group(1):
            raise ValueError(f"manifest hash mismatch for {relative_text}")
        entries[relative_text] = match.group(1)
    actual = set()
    for path in capture_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(capture_dir)
        if relative in (Path("manifest.sha256"), Path("manifest_anchor.json")):
            continue
        actual.add(str(relative))
    if set(entries) != actual:
        raise ValueError(
            "manifest file set mismatch: "
            f"missing={sorted(actual-set(entries))}, "
            f"extra={sorted(set(entries)-actual)}")
    if set(entries) != EXPECTED_CAPTURE_ARTIFACTS:
        raise ValueError(
            "capture artifact inventory mismatch: "
            f"missing={sorted(EXPECTED_CAPTURE_ARTIFACTS-set(entries))}, "
            f"extra={sorted(set(entries)-EXPECTED_CAPTURE_ARTIFACTS)}")
    return entries


def _verify_artifact_reference(capture_dir, manifest, reference, label):
    if set(reference) != {"path", "sha256"}:
        raise ValueError(f"{label} reference schema changed")
    relative, path = _safe_artifact_path(capture_dir, reference["path"])
    relative_text = str(relative)
    if relative_text not in manifest:
        raise ValueError(f"{label} is absent from manifest")
    if reference["sha256"] != manifest[relative_text]:
        raise ValueError(f"{label} hash disagrees with manifest")
    if _sha256(path) != reference["sha256"]:
        raise ValueError(f"{label} file hash changed")


def _require_exact_keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        observed = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} schema mismatch: "
            f"missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}")


def _git_blob(repo_root, commit, relative_path):
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False)
    if result.returncode != 0:
        raise ValueError(
            f"baseline {commit} does not contain {relative_path}: "
            f"{result.stderr.decode(errors='replace')}")
    return result.stdout


def _require_executable_identity(path_text, expected_sha256, label):
    path = Path(_require_nonempty(path_text, f"{label} path"))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} executable path is not an absolute file")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError(f"{label} executable hash is malformed")
    if _sha256(path.resolve()) != expected_sha256:
        raise ValueError(f"{label} executable differs from capture")
    return path


def _run_recomputed(command, cwd, label):
    result = subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise ValueError(
            f"{label} failed during verification: "
            f"returncode={result.returncode}, stdout={result.stdout!r}, "
            f"stderr={result.stderr!r}")
    return result


def _verify_source_and_build_chain(capture_dir, repo_root, provenance):
    baseline = provenance["baseline_commit"]
    if re.fullmatch(r"[0-9a-f]{40}", baseline) is None:
        raise ValueError("provenance baseline commit is malformed")
    baseline_text = (capture_dir / "baseline_commit.txt").read_text(
        encoding="utf-8")
    if baseline_text != baseline + "\n":
        raise ValueError("baseline_commit.txt disagrees with provenance")
    source_relative = (
        f"microbench/{HARDWARE_ID}/runtime/launch/{SOURCE_NAME}")
    runner_relative = (
        f"microbench/{HARDWARE_ID}/runtime/launch/{RUNNER_NAME}")
    expected_snapshot_paths = {
        "source": f"source_snapshot/{SOURCE_NAME}",
        "runner": f"source_snapshot/{RUNNER_NAME}",
    }
    for label, repository_relative in (
            ("source", source_relative), ("runner", runner_relative)):
        reference = provenance[label]
        if reference["path"] != expected_snapshot_paths[label]:
            raise ValueError(f"{label} snapshot path is not canonical")
        snapshot = capture_dir / reference["path"]
        if snapshot.read_bytes() != _git_blob(
                repo_root, baseline, repository_relative):
            raise ValueError(
                f"{label} snapshot differs from baseline {baseline}")

    compiler = _require_executable_identity(
        provenance["compiler"]["path"],
        provenance["compiler"]["sha256"], "compiler")
    cuobjdump = _require_executable_identity(
        provenance["binary_audit"]["cuobjdump_path"],
        provenance["binary_audit"]["cuobjdump_sha256"], "cuobjdump")
    _require_executable_identity(
        provenance["nvidia_smi"]["path"],
        provenance["nvidia_smi"]["sha256"], "nvidia-smi")
    source_path = capture_dir / provenance["source"]["path"]
    binary_path = capture_dir / provenance["binary"]["path"]
    sass_path = capture_dir / provenance["binary_audit"]["sass_path"]
    if not os.access(binary_path, os.X_OK):
        raise ValueError("captured binary is not executable")
    expected_compile_argv = [
        str(compiler), "-O3", "-std=c++17", "-arch=sm_80",
        "-Xptxas=-v", str(source_path), "-o", str(binary_path),
    ]
    if provenance["compile_argv"] != expected_compile_argv:
        raise ValueError("compile argv does not match the frozen build contract")
    expected_cuobjdump_argv = [
        str(cuobjdump), "--dump-sass", str(binary_path)]
    if provenance["cuobjdump_argv"] != expected_cuobjdump_argv:
        raise ValueError("cuobjdump argv does not match the frozen audit contract")
    expected_native_argv = [
        str(binary_path), "--trials", str(provenance["trials"])]
    if provenance["capture_mode"] == "smoke_cell":
        expected_native_argv.append("--smoke-cell")
    elif provenance["capture_mode"] == "full_matrix":
        pass
    else:
        raise ValueError("provenance capture mode is unknown")
    if provenance["native_argv"] != expected_native_argv:
        raise ValueError("native argv does not match capture mode and trials")

    preserved_sass = sass_path.read_text(encoding="utf-8")
    validate_ptxas((capture_dir / "build/compile.stderr").read_text(
        encoding="utf-8"))
    preserved_audit = validate_sass(preserved_sass)
    audit_json = _read_json_object(
        capture_dir / "build/sass_audit.json", "SASS audit")
    if (provenance["binary_audit"]["sass_audit"] != preserved_audit
            or audit_json != preserved_audit):
        raise ValueError("preserved SASS audit does not recompute")
    binary_sass = _run_recomputed(
        expected_cuobjdump_argv, repo_root,
        "captured-binary SASS extraction").stdout
    if binary_sass != preserved_sass:
        raise ValueError("captured binary does not reproduce preserved SASS")

    with tempfile.TemporaryDirectory(prefix="cta_assignment_verify_") as path:
        rebuild_binary = Path(path) / "cta_assignment"
        rebuild_argv = [*expected_compile_argv[:-1], str(rebuild_binary)]
        rebuilt = _run_recomputed(
            rebuild_argv, repo_root, "source rebuild")
        validate_ptxas(rebuilt.stderr)
        rebuilt_sass = _run_recomputed(
            [str(cuobjdump), "--dump-sass", str(rebuild_binary)],
            repo_root, "rebuilt-binary SASS extraction").stdout
    validate_sass(rebuilt_sass)
    if rebuilt_sass != preserved_sass:
        raise ValueError(
            "captured binary SASS does not reproduce from baseline source")


def verify_capture(
        capture_directory, expected_manifest_sha256, repo_root=Path.cwd()):
    """Verify a capture against an out-of-tree final-manifest digest."""

    capture_dir = capture_directory.resolve()
    repository = repo_root.resolve()
    if not capture_dir.is_dir():
        raise NotADirectoryError(capture_dir)
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is None:
        raise ValueError(
            "expected detached manifest SHA256 must be 64 lowercase "
            "hexadecimal digits")
    observed_manifest_sha256 = _sha256(capture_dir / "manifest.sha256")
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("detached manifest SHA256 mismatch")
    manifest = _parse_manifest(capture_dir)
    anchor = _read_json_object(
        capture_dir / "manifest_anchor.json", "manifest anchor")
    expected_anchor_keys = {
        "schema", "manifest_path", "manifest_sha256", "entry_count"}
    if set(anchor) != expected_anchor_keys:
        raise ValueError("manifest anchor schema changed")
    if (anchor["schema"] != "sha256_manifest_anchor_v1"
            or anchor["manifest_path"] != "manifest.sha256"
            or anchor["entry_count"] != len(manifest)
            or anchor["manifest_sha256"]
               != _sha256(capture_dir / "manifest.sha256")):
        raise ValueError("manifest anchor does not bind the manifest")

    provenance = _read_json_object(
        capture_dir / "provenance.json", "provenance")
    candidate = _read_json_object(
        capture_dir / "candidate.json", "candidate")
    capture_identity = _read_json_object(
        capture_dir / "capture_identity.json", "capture identity")
    _require_exact_keys(provenance, {
        "schema", "hardware_id", "baseline_commit", "capture_mode",
        "trials", "source", "runner", "binary", "compiler",
        "nvidia_smi", "binary_audit", "device_preflight",
        "device_postflight", "device_evidence", "capture_argv",
        "compile_argv", "cuobjdump_argv", "native_argv", "raw",
        "candidate_evidence_status",
        "production_or_evaluation_duration_consumed",
        "production_parameter_published",
    }, "provenance")
    _require_exact_keys(capture_identity, {
        "schema", "hardware_id", "baseline_commit", "run_id",
        "device_uuid", "pci_bus_id", "invocation_argv",
        "normalized_capture_argv", "pid", "started_unix_s", "output_path",
        "capture_mode", "production_or_evaluation_duration_consumed",
        "production_parameter_published",
    }, "capture identity")
    for label in ("source", "runner", "binary", "raw"):
        _require_exact_keys(provenance[label], {"path", "sha256"}, label)
    _require_exact_keys(provenance["compiler"], {
        "path", "sha256", "version_output_path", "version_output_sha256",
    }, "compiler")
    _require_exact_keys(provenance["nvidia_smi"], {
        "path", "sha256", "version", "version_output_path",
        "version_output_sha256",
    }, "nvidia-smi")
    _require_exact_keys(provenance["binary_audit"], {
        "cuobjdump_path", "cuobjdump_sha256", "sass_path", "sass_sha256",
        "sass_audit",
    }, "binary audit")
    expected_device_evidence = {
        "pre_native_device_raw", "pre_native_device_stderr",
        "pre_native_processes_raw", "pre_native_processes_stderr",
        "post_native_device_raw", "post_native_device_stderr",
        "post_native_processes_raw", "post_native_processes_stderr",
    }
    _require_exact_keys(
        provenance["device_evidence"], expected_device_evidence,
        "device evidence")
    for label in expected_device_evidence:
        _require_exact_keys(
            provenance["device_evidence"][label], {"path", "sha256"},
            label)
    if provenance["schema"] != "nvidia_a800_cta_assignment_provenance_v3":
        raise ValueError("unknown provenance schema")
    if candidate["schema"] != "nvidia_a800_cta_assignment_candidate_v3":
        raise ValueError("unknown candidate schema")
    if (capture_identity["schema"]
            != "nvidia_a800_cta_assignment_capture_identity_v1"):
        raise ValueError("unknown capture identity schema")
    if (provenance["hardware_id"] != HARDWARE_ID
            or capture_identity["hardware_id"] != HARDWARE_ID):
        raise ValueError("capture hardware ID changed")
    if (capture_identity["baseline_commit"]
            != provenance["baseline_commit"]):
        raise ValueError("capture identity baseline disagrees with provenance")
    if capture_identity["capture_mode"] != provenance["capture_mode"]:
        raise ValueError("capture identity mode disagrees with provenance")
    if Path(capture_identity["output_path"]).resolve() != capture_dir:
        raise ValueError("capture identity output path changed")
    if (not isinstance(capture_identity["pid"], int)
            or isinstance(capture_identity["pid"], bool)
            or capture_identity["pid"] <= 0
            or not isinstance(capture_identity["started_unix_s"], (int, float))
            or isinstance(capture_identity["started_unix_s"], bool)):
        raise ValueError("capture process identity is malformed")
    if (RUN_ID_PATTERN.fullmatch(
            _require_nonempty(capture_identity["run_id"], "run ID")) is None):
        raise ValueError("capture identity run ID is malformed")
    if (not isinstance(capture_identity["invocation_argv"], list)
            or not capture_identity["invocation_argv"]
            or any(not isinstance(value, str) or not value
                   for value in capture_identity["invocation_argv"])):
        raise ValueError("capture invocation argv is malformed")
    expected_capture_argv = [
        RUNNER_NAME,
        "capture",
        "--output-dir", str(capture_dir),
        "--run-id", capture_identity["run_id"],
        "--device-uuid", capture_identity["device_uuid"],
        "--pci-bus-id", _normalize_pci_bus_id(
            capture_identity["pci_bus_id"]),
        "--repo-root", str(repository),
        "--nvcc", provenance["compiler"]["path"],
        "--cuobjdump", provenance["binary_audit"]["cuobjdump_path"],
    ]
    if provenance["capture_mode"] == "smoke_cell":
        expected_capture_argv.append("--smoke-cell")
    elif provenance["capture_mode"] == "full_matrix":
        expected_capture_argv.extend(("--trials", str(provenance["trials"])))
    else:
        raise ValueError("provenance capture mode is unknown")
    if (capture_identity["normalized_capture_argv"] != expected_capture_argv
            or provenance["capture_argv"] != expected_capture_argv):
        raise ValueError("normalized capture argv does not recompute")
    if (capture_identity["production_or_evaluation_duration_consumed"]
            is not False
            or capture_identity["production_parameter_published"] is not False):
        raise ValueError("capture identity violates zero-fit evidence boundary")
    expected_reference_paths = {
        "source": f"source_snapshot/{SOURCE_NAME}",
        "runner": f"source_snapshot/{RUNNER_NAME}",
        "binary": "build/cta_assignment",
        "raw": "native/raw.csv",
    }
    for label in ("source", "runner", "binary", "raw"):
        if provenance[label]["path"] != expected_reference_paths[label]:
            raise ValueError(f"{label} artifact path is not canonical")
        _verify_artifact_reference(
            capture_dir, manifest, provenance[label], label)
    expected_device_paths = {
        "pre_native_device_raw": "device/pre_native_device.csv",
        "pre_native_device_stderr": "device/pre_native_device.stderr",
        "pre_native_processes_raw": "device/pre_native_processes.csv",
        "pre_native_processes_stderr": "device/pre_native_processes.stderr",
        "post_native_device_raw": "device/post_native_device.csv",
        "post_native_device_stderr": "device/post_native_device.stderr",
        "post_native_processes_raw": "device/post_native_processes.csv",
        "post_native_processes_stderr": "device/post_native_processes.stderr",
    }
    for label in (
            "pre_native_device_raw", "pre_native_device_stderr",
            "pre_native_processes_raw", "pre_native_processes_stderr",
            "post_native_device_raw", "post_native_device_stderr",
            "post_native_processes_raw", "post_native_processes_stderr"):
        if (provenance["device_evidence"][label]["path"]
                != expected_device_paths[label]):
            raise ValueError(f"{label} artifact path is not canonical")
        _verify_artifact_reference(
            capture_dir, manifest, provenance["device_evidence"][label],
            label)
    compiler_version_reference = {
        "path": provenance["compiler"]["version_output_path"],
        "sha256": provenance["compiler"]["version_output_sha256"],
    }
    if compiler_version_reference["path"] != "build/nvcc_version.txt":
        raise ValueError("compiler version artifact path is not canonical")
    _verify_artifact_reference(
        capture_dir, manifest, compiler_version_reference,
        "compiler version output")
    nvidia_smi_version_reference = {
        "path": provenance["nvidia_smi"]["version_output_path"],
        "sha256": provenance["nvidia_smi"]["version_output_sha256"],
    }
    if (nvidia_smi_version_reference["path"]
            != "build/nvidia_smi_version.txt"):
        raise ValueError("nvidia-smi version artifact path is not canonical")
    _verify_artifact_reference(
        capture_dir, manifest, nvidia_smi_version_reference,
        "nvidia-smi version output")
    if provenance["nvidia_smi"]["version"] != _parse_nvidia_smi_version(
            (capture_dir / nvidia_smi_version_reference["path"]).read_text(
                encoding="utf-8")):
        raise ValueError("nvidia-smi version disagrees with preserved output")
    sass_reference = {
        "path": provenance["binary_audit"]["sass_path"],
        "sha256": provenance["binary_audit"]["sass_sha256"],
    }
    if sass_reference["path"] != "build/cta_assignment.sass":
        raise ValueError("SASS artifact path is not canonical")
    _verify_artifact_reference(
        capture_dir, manifest, sass_reference, "SASS")
    if provenance["nvidia_smi"]["path"] != str(NVIDIA_SMI_PATH):
        raise ValueError("nvidia-smi path is not pinned to /usr/bin/nvidia-smi")
    _verify_source_and_build_chain(capture_dir, repository, provenance)

    raw_path = capture_dir / provenance["raw"]["path"]
    raw_sha256 = provenance["raw"]["sha256"]
    if (not isinstance(provenance["trials"], int)
            or isinstance(provenance["trials"], bool)
            or provenance["trials"] <= 0 or provenance["trials"] > 100):
        raise ValueError("provenance trials are outside [1, 100]")
    native = parse_native_output(
        raw_path.read_text(encoding="utf-8"), provenance["trials"],
        provenance["capture_mode"])
    if (native["meta"]["device_uuid"] != capture_identity["device_uuid"]
            or _normalize_pci_bus_id(native["meta"]["pci_bus_id"])
            != _normalize_pci_bus_id(capture_identity["pci_bus_id"])):
        raise ValueError("capture identity disagrees with native device")
    pre_device_path = (
        capture_dir
        / provenance["device_evidence"]["pre_native_device_raw"]["path"])
    post_device_path = (
        capture_dir
        / provenance["device_evidence"]["post_native_device_raw"]["path"])
    pre_process_path = (
        capture_dir
        / provenance["device_evidence"]["pre_native_processes_raw"]["path"])
    post_process_path = (
        capture_dir
        / provenance["device_evidence"]["post_native_processes_raw"]["path"])
    preflight = _parse_device_row(
        pre_device_path.read_text(encoding="utf-8"),
        provenance["device_preflight"]["physical_identity"])
    postflight = _parse_device_row(
        post_device_path.read_text(encoding="utf-8"),
        provenance["device_postflight"]["physical_identity"])
    if preflight != provenance["device_preflight"]:
        raise ValueError("parsed preflight device disagrees with provenance")
    if postflight != provenance["device_postflight"]:
        raise ValueError("parsed postflight device disagrees with provenance")
    _require_idle_device(
        preflight, native["meta"]["device_uuid"],
        native["meta"]["pci_bus_id"],
        pre_process_path.read_text(encoding="utf-8"))
    _require_released_device(
        postflight, native["meta"]["device_uuid"],
        native["meta"]["pci_bus_id"],
        post_process_path.read_text(encoding="utf-8"))
    reconstructed = analyze_assignment(native, raw_sha256)
    if candidate != reconstructed:
        raise ValueError("candidate does not reproduce from raw evidence")
    if candidate["raw_sha256"] != raw_sha256:
        raise ValueError("candidate raw hash disagrees with provenance")
    if (candidate["evidence_status"]
            != provenance["candidate_evidence_status"]):
        raise ValueError("candidate status disagrees with provenance")
    if (candidate["forward_admissibility"] != "evidence_pending"
            or candidate["cross_program_transfer_supported"] is not False):
        raise ValueError(
            "candidate illegally claims cross-program Forward admissibility")
    if (provenance["production_or_evaluation_duration_consumed"] is not False
            or provenance["production_parameter_published"] is not False):
        raise ValueError("provenance violates the zero-fit evidence boundary")
    return {
        "schema": "nvidia_a800_cta_assignment_verification_v2",
        "status": "pass",
        "capture_mode": provenance["capture_mode"],
        "evidence_status": candidate["evidence_status"],
        "manifest_sha256": anchor["manifest_sha256"],
        "manifest_entry_count": len(manifest),
        "raw_sha256": raw_sha256,
    }


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument("--run-id", required=True)
    capture_parser.add_argument("--device-uuid", required=True)
    capture_parser.add_argument("--pci-bus-id", required=True)
    capture_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    capture_parser.add_argument(
        "--nvcc", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    capture_parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    capture_parser.add_argument("--trials", type=int)
    capture_parser.add_argument("--smoke-cell", action="store_true")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--capture-dir", type=Path, required=True)
    verify_parser.add_argument(
        "--expected-manifest-sha256", required=True)
    verify_parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    if arguments.action == "verify":
        return arguments
    if arguments.action != "capture":
        raise ValueError(f"unknown action {arguments.action!r}")
    arguments.run_id = _require_nonempty(arguments.run_id, "run ID")
    if RUN_ID_PATTERN.fullmatch(arguments.run_id) is None:
        raise ValueError("run ID contains unsupported characters")
    arguments.device_uuid = _require_nonempty(
        arguments.device_uuid, "device UUID")
    arguments.pci_bus_id = _require_nonempty(
        arguments.pci_bus_id, "PCI bus ID")
    _normalize_pci_bus_id(arguments.pci_bus_id)
    if arguments.smoke_cell:
        if arguments.trials is not None:
            raise ValueError(
                "--smoke-cell uses its fixed single trial; do not pass "
                "--trials")
        arguments.trials = 1
    elif arguments.trials is None:
        arguments.trials = 7
    if (isinstance(arguments.trials, bool)
            or not isinstance(arguments.trials, int)
            or arguments.trials <= 0 or arguments.trials > 100):
        raise ValueError("trials must be in [1, 100]")
    return arguments


def main(argv=None):
    effective_argv = sys.argv if argv is None else [str(RUNNER_NAME), *argv]
    arguments = _arguments(argv)
    if arguments.action == "capture":
        result = capture(arguments, effective_argv)
    elif arguments.action == "verify":
        result = verify_capture(
            arguments.capture_dir, arguments.expected_manifest_sha256,
            arguments.repo_root)
    else:
        raise ValueError(f"unknown action {arguments.action!r}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
