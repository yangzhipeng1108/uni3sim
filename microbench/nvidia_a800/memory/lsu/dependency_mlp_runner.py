#!/usr/bin/env python3
"""Capture and audit the A800 finite-source dependent-load MLP sweep."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


HARDWARE_ID = "nvidia_a800"
EXPECTED_SM_COUNT = 108
EXPECTED_L2_BYTES = 40 * 1024 * 1024
EXPECTED_L2_CACHELINE_BYTES = 128
EXPECTED_DYNAMIC_SHARED_BYTES = 96 * 1024
EXPECTED_DRAM_RING_BYTES = 256 * 1024 * 1024
EXPECTED_EVICTION_BYTES = 80 * 1024 * 1024
SECTOR_BYTES = 32
WARP_POINTS = (1, 4, 8, 16, 32)
CHAIN_POINTS = (1, 2, 4, 8)
TIERS = ("l1_warm_default", "l2_warm_cg", "dram_cg")
UNTIMED_LOAD_ROUNDS_BY_TIER = {
    "l1_warm_default": 2,
    "l2_warm_cg": 2,
    "dram_cg": 1,
}
SASS_TARGET_LOAD_SITES_BY_TIER = {
    "l1_warm_default": 2,
    "l2_warm_cg": 2,
    "dram_cg": 1,
}
SASS_CACHE_SCOPES_BY_TIER = {
    "l1_warm_default": ("SM", "GPU"),
    "l2_warm_cg": ("GPU", "SM"),
    "dram_cg": ("GPU", "SM"),
}
BOUNDARY_POINTS = ((1, 1), (1, 8), (32, 1), (32, 8))
_MAXIMUM_PRODUCER_SLOTS = (
    EXPECTED_SM_COUNT * max(WARP_POINTS) * max(CHAIN_POINTS))
_L2_TIER_DEVICE_ALLOCATIONS = (
    _MAXIMUM_PRODUCER_SLOTS * SECTOR_BYTES,
    _MAXIMUM_PRODUCER_SLOTS * 4,
    EXPECTED_SM_COUNT * max(WARP_POINTS) * 4,
    EXPECTED_SM_COUNT * SECTOR_BYTES,
)
MAXIMUM_L2_TIER_COMPULSORY_DRAM_SECTORS = sum(
    math.ceil(size / EXPECTED_L2_CACHELINE_BYTES)
    * (EXPECTED_L2_CACHELINE_BYTES // SECTOR_BYTES)
    for size in _L2_TIER_DEVICE_ALLOCATIONS
)
NCU_METRICS = (
    "smsp__sass_inst_executed_op_global_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "lts__t_sectors_aperture_device_op_read.sum",
    "dram__sectors_read.sum",
)
NCU_METRIC_UNITS = {
    "smsp__sass_inst_executed_op_global_ld.sum": "inst",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "sector",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit.sum": "byte",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum": "byte",
    "lts__t_sectors_aperture_device_op_read.sum": "sector",
    "dram__sectors_read.sum": "sector",
}
RATE_UNIT = "32B_dependent_load_requests_per_active_SM_per_SM_cycle"
EXPECTED_HEADER = (
    "expected", "tier", "kernel", "grid", "producer_warps_per_sm",
    "independent_chains_per_warp", "steps_per_chain",
    "target_chain_advances", "target_request_bytes",
    "counter_load_warp_instructions", "counter_load_sectors",
)
RAW_HEADER = (
    "raw", "tier", "kernel", "grid", "producer_warps_per_sm",
    "independent_chains_per_warp", "steps_per_chain", "trial",
    "target_chain_advances", "sum_cta_sm_cycles", "min_cta_sm_cycles",
    "max_cta_sm_cycles", "requests_per_active_sm_cycle",
)
SOURCE_NAME = "dependency_mlp.cu"
RUNNER_NAME = "dependency_mlp_runner.py"


def _require_nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _run(command, *, cwd, env=None, stdout_path=None, stderr_path=None):
    if not isinstance(command, list) or not command:
        raise TypeError("command must be a non-empty list")
    if any(not isinstance(value, str) or not value for value in command):
        raise ValueError("command entries must be non-empty strings")
    stdout_stream = subprocess.PIPE
    stderr_stream = subprocess.PIPE
    opened = []
    if stdout_path is not None:
        stdout_stream = open(stdout_path, "w", encoding="utf-8")
        opened.append(stdout_stream)
    if stderr_path is not None:
        stderr_stream = open(stderr_path, "w", encoding="utf-8")
        opened.append(stderr_stream)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            check=True,
        )
    finally:
        for stream in opened:
            stream.close()


def _git_output(repo_root, *arguments):
    result = _run(
        ["git", *arguments], cwd=repo_root)
    return result.stdout.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _kernel_name(tier, chains):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if chains not in CHAIN_POINTS:
        raise ValueError(f"unknown chain point {chains!r}")
    return f"dependency_mlp_{tier}_c{chains}"


def _expected_cells():
    return tuple(
        (tier, chains, warps)
        for tier in TIERS
        for chains in CHAIN_POINTS
        for warps in WARP_POINTS
    )


def _parse_int(text, label):
    normalized = _require_nonempty(text, label).replace(",", "")
    if not re.fullmatch(r"[0-9]+", normalized):
        raise ValueError(f"{label} must be a non-negative integer")
    return int(normalized)


def _parse_float(text, label):
    normalized = _require_nonempty(text, label).replace(",", "")
    value = float(normalized)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


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


def parse_native_output(text, expected_trials=9):
    """Validate the complete native matrix and return structured rows."""

    if not isinstance(text, str) or not text:
        raise ValueError("native output must be a non-empty string")
    if (isinstance(expected_trials, bool)
            or not isinstance(expected_trials, int)
            or expected_trials <= 0):
        raise ValueError("expected_trials must be a positive integer")
    meta = {}
    expected = {}
    raw = {}
    summary = {}
    expected_columns = None
    raw_columns = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        columns = next(csv.reader([line]))
        kind = columns[0]
        if kind == "meta":
            if len(columns) != 3:
                raise ValueError(f"native meta row {line_number} is malformed")
            key = _require_nonempty(columns[1], "native meta key")
            if key in meta:
                raise ValueError(f"native meta repeats {key!r}")
            meta[key] = columns[2]
        elif kind == "expected" and columns[1] == "tier":
            if expected_columns is not None:
                raise ValueError("native output repeats expected header")
            expected_columns = tuple(columns)
        elif kind == "raw" and columns[1] == "tier":
            if raw_columns is not None:
                raise ValueError("native output repeats raw header")
            raw_columns = tuple(columns)
        elif kind == "expected":
            if len(columns) != 11:
                raise ValueError(
                    f"native expected row {line_number} is malformed")
            tier = columns[1]
            kernel = columns[2]
            grid = _parse_int(columns[3], "expected grid")
            warps = _parse_int(columns[4], "expected warps")
            chains = _parse_int(columns[5], "expected chains")
            key = (tier, chains, warps)
            if key in expected:
                raise ValueError(f"native expected repeats {key!r}")
            record = {
                "tier": tier,
                "kernel": kernel,
                "grid": grid,
                "warps": warps,
                "chains": chains,
                "steps": _parse_int(columns[6], "expected steps"),
                "target_chain_advances": _parse_int(
                    columns[7], "target chain advances"),
                "target_request_bytes": _parse_int(
                    columns[8], "target request bytes"),
                "counter_load_warp_instructions": _parse_int(
                    columns[9], "counter load warp instructions"),
                "counter_load_sectors": _parse_int(
                    columns[10], "counter load sectors"),
            }
            expected[key] = record
        elif kind == "raw":
            if len(columns) != 13:
                raise ValueError(f"native raw row {line_number} is malformed")
            tier = columns[1]
            kernel = columns[2]
            grid = _parse_int(columns[3], "raw grid")
            warps = _parse_int(columns[4], "raw warps")
            chains = _parse_int(columns[5], "raw chains")
            trial = _parse_int(columns[7], "raw trial")
            cell = (tier, chains, warps)
            key = (tier, chains, warps, trial)
            if key in raw:
                raise ValueError(f"native raw repeats {key!r}")
            raw[key] = {
                "tier": tier,
                "kernel": kernel,
                "grid": grid,
                "warps": warps,
                "chains": chains,
                "steps": _parse_int(columns[6], "raw steps"),
                "trial": trial,
                "target_chain_advances": _parse_int(
                    columns[8], "raw target chain advances"),
                "sum_cta_sm_cycles": _parse_int(
                    columns[9], "sum CTA SM cycles"),
                "min_cta_sm_cycles": _parse_int(
                    columns[10], "minimum CTA SM cycles"),
                "max_cta_sm_cycles": _parse_int(
                    columns[11], "maximum CTA SM cycles"),
                "requests_per_active_sm_cycle": _parse_float(
                    columns[12], "request rate"),
                "cell": cell,
            }
        elif kind == "summary":
            if len(columns) != 8:
                raise ValueError(
                    f"native summary row {line_number} is malformed")
            tier = columns[1]
            kernel = columns[2]
            grid = _parse_int(columns[3], "summary grid")
            warps = _parse_int(columns[4], "summary warps")
            chains = _parse_int(columns[5], "summary chains")
            key = (tier, chains, warps)
            if key in summary:
                raise ValueError(f"native summary repeats {key!r}")
            summary[key] = {
                "tier": tier,
                "kernel": kernel,
                "grid": grid,
                "warps": warps,
                "chains": chains,
                "steps": _parse_int(columns[6], "summary steps"),
                "median_requests_per_active_sm_cycle": _parse_float(
                    columns[7], "summary request rate"),
            }
        else:
            raise ValueError(
                f"native output has unknown row kind at line {line_number}: "
                f"{kind!r}")

    required_meta = {
        "device_uuid", "pci_bus_id", "compute_capability", "sm_count",
        "l2_bytes", "dynamic_shared_bytes_per_cta",
        "producer_warps_per_sm_points",
        "independent_chains_per_warp_points", "cache_tiers", "sector_bytes",
        "dram_ring_bytes", "eviction_bytes", "rate_unit",
        "production_or_evaluation_duration_consumed",
        "production_parameter_published",
    }
    missing_meta = required_meta - set(meta)
    if missing_meta:
        raise ValueError(f"native output lacks meta fields {sorted(missing_meta)}")
    if set(meta) - (required_meta | {"device_index", "device_name"}):
        raise ValueError(
            "native output has unknown meta fields "
            f"{sorted(set(meta) - required_meta - {'device_index', 'device_name'})}")
    if meta["compute_capability"] != "sm_80":
        raise ValueError("native output is not sm_80")
    if _parse_int(meta["sm_count"], "SM count") != EXPECTED_SM_COUNT:
        raise ValueError("native output does not cover 108 A800 SMs")
    if _parse_int(meta["l2_bytes"], "L2 bytes") != EXPECTED_L2_BYTES:
        raise ValueError("native output does not report the 40-MiB A800 L2")
    if (_parse_int(
            meta["dynamic_shared_bytes_per_cta"],
            "dynamic shared bytes per CTA")
            != EXPECTED_DYNAMIC_SHARED_BYTES):
        raise ValueError("native output does not enforce one 96-KiB CTA per SM")
    if meta["producer_warps_per_sm_points"] != "|".join(
            str(value) for value in WARP_POINTS):
        raise ValueError("native output changed the producer-warp matrix")
    if meta["independent_chains_per_warp_points"] != "|".join(
            str(value) for value in CHAIN_POINTS):
        raise ValueError("native output changed the independent-chain matrix")
    if meta["cache_tiers"] != "|".join(TIERS):
        raise ValueError("native output changed the cache-tier matrix")
    if _parse_int(meta["sector_bytes"], "sector bytes") != SECTOR_BYTES:
        raise ValueError("native output transaction size is not 32 bytes")
    if (_parse_int(meta["dram_ring_bytes"], "DRAM ring bytes")
            != EXPECTED_DRAM_RING_BYTES):
        raise ValueError("native output changed the 256-MiB DRAM ring")
    if (_parse_int(meta["eviction_bytes"], "eviction bytes")
            != EXPECTED_EVICTION_BYTES):
        raise ValueError("native output changed the 80-MiB L2 eviction")
    if meta["rate_unit"] != RATE_UNIT:
        raise ValueError("native output changed the physical rate unit")
    if meta["production_or_evaluation_duration_consumed"] != "false":
        raise ValueError("native output consumed a forbidden duration")
    if meta["production_parameter_published"] != "false":
        raise ValueError("native sweep cannot publish a production parameter")
    if expected_columns is None or raw_columns is None:
        raise ValueError("native output lacks CSV headers")
    if expected_columns != EXPECTED_HEADER or raw_columns != RAW_HEADER:
        raise ValueError("native output CSV schema changed")

    cells = set(_expected_cells())
    if set(expected) != cells:
        raise ValueError(
            "native expected matrix mismatch: "
            f"missing={sorted(cells - set(expected))}, "
            f"extra={sorted(set(expected) - cells)}")
    if set(summary) != cells:
        raise ValueError(
            "native summary matrix mismatch: "
            f"missing={sorted(cells - set(summary))}, "
            f"extra={sorted(set(summary) - cells)}")
    expected_raw_keys = {
        (tier, chains, warps, trial)
        for tier, chains, warps in cells
        for trial in range(expected_trials)
    }
    if set(raw) != expected_raw_keys:
        raise ValueError(
            "native raw matrix mismatch: "
            f"missing={sorted(expected_raw_keys - set(raw))}, "
            f"extra={sorted(set(raw) - expected_raw_keys)}")

    for cell in sorted(cells):
        tier, chains, warps = cell
        expected_record = expected[cell]
        summary_record = summary[cell]
        if expected_record["kernel"] != _kernel_name(tier, chains):
            raise ValueError(f"native kernel identity mismatch for {cell!r}")
        if expected_record["grid"] != EXPECTED_SM_COUNT:
            raise ValueError(f"native expected grid mismatch for {cell!r}")
        if expected_record["steps"] <= 0:
            raise ValueError(f"native step count is not positive for {cell!r}")
        expected_advances = (
            EXPECTED_SM_COUNT * warps * chains * expected_record["steps"])
        if expected_record["target_chain_advances"] != expected_advances:
            raise ValueError(f"native target advances do not conserve {cell!r}")
        if expected_record["target_request_bytes"] != (
                expected_record["target_chain_advances"] * SECTOR_BYTES):
            raise ValueError(f"native target bytes do not conserve {cell!r}")
        untimed_rounds = UNTIMED_LOAD_ROUNDS_BY_TIER[tier]
        expected_counter_loads = (
            expected_advances
            + EXPECTED_SM_COUNT * warps * chains * untimed_rounds)
        if (expected_record["counter_load_warp_instructions"]
                != expected_counter_loads):
            raise ValueError(f"native counter load count changed for {cell!r}")
        if (expected_record["counter_load_warp_instructions"]
                != expected_record["counter_load_sectors"]):
            raise ValueError(f"native load request geometry changed for {cell!r}")
        if any(
                summary_record[field] != expected_record[field]
                for field in ("kernel", "grid", "warps", "chains", "steps")):
            raise ValueError(f"native summary/expected mismatch for {cell!r}")
        if summary_record["median_requests_per_active_sm_cycle"] <= 0:
            raise ValueError(f"native summary rate is not positive for {cell!r}")
        rates = []
        for trial in range(expected_trials):
            record = raw[(tier, chains, warps, trial)]
            if (record["kernel"] != expected_record["kernel"]
                    or record["grid"] != expected_record["grid"]
                    or record["steps"] != expected_record["steps"]
                    or record["target_chain_advances"]
                       != expected_record["target_chain_advances"]):
                raise ValueError(f"native raw/expected mismatch for {cell!r}")
            if not (0 < record["min_cta_sm_cycles"]
                    <= record["max_cta_sm_cycles"]
                    <= record["sum_cta_sm_cycles"]):
                raise ValueError(f"native cycle interval is invalid for {cell!r}")
            if record["requests_per_active_sm_cycle"] <= 0:
                raise ValueError(f"native request rate is not positive for {cell!r}")
            recomputed = (
                record["target_chain_advances"]
                / record["sum_cta_sm_cycles"])
            if not math.isclose(
                    recomputed, record["requests_per_active_sm_cycle"],
                    rel_tol=5e-12, abs_tol=5e-13):
                raise ValueError(f"native request rate is inconsistent for {cell!r}")
            rates.append(record["requests_per_active_sm_cycle"])
        median = sorted(rates)[len(rates) // 2]
        if not math.isclose(
                median, summary_record["median_requests_per_active_sm_cycle"],
                rel_tol=5e-12, abs_tol=5e-13):
            raise ValueError(f"native summary median is inconsistent for {cell!r}")
    return {
        "meta": meta,
        "expected": expected,
        "raw": raw,
        "summary": summary,
    }


def _sass_functions(text):
    functions = {}
    current = None
    lines = []
    for line in text.splitlines():
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match is not None:
            if current is not None:
                functions[current] = "\n".join(lines)
            current = match.group(1)
            lines = [line]
        elif current is not None:
            lines.append(line)
    if current is not None:
        functions[current] = "\n".join(lines)
    return functions


def validate_sass(text):
    """Validate frozen ABI, cache policy, timer count, and spill absence."""

    functions = _sass_functions(text)
    expected_names = {
        _kernel_name(tier, chains)
        for tier in TIERS for chains in CHAIN_POINTS
    }
    missing = expected_names - set(functions)
    if missing:
        raise ValueError(f"SASS lacks dependency-MLP kernels {sorted(missing)}")
    result = {}
    for tier in TIERS:
        for chains in CHAIN_POINTS:
            name = _kernel_name(tier, chains)
            body = functions[name]
            if re.search(r"\b(?:LDL|STL)(?:\.|\s)", body):
                raise ValueError(f"{name} spills chain state to local memory")
            clock_reads = len(re.findall(r"CS2R\s+R\d+,\s*SR_CLOCKLO", body))
            if clock_reads != 2:
                raise ValueError(f"{name} must contain exactly two clock64 reads")
            head_loads = len(re.findall(r"\sLDG\.E\s+R", body))
            if head_loads != chains:
                raise ValueError(
                    f"{name} must contain {chains} unscoped head loads, "
                    f"got {head_loads}")
            target_scope, wrong_scope = SASS_CACHE_SCOPES_BY_TIER[tier]
            target_loads = len(re.findall(
                rf"LDG\.E\.STRONG\.{target_scope}\s+R", body))
            wrong_loads = len(re.findall(
                rf"LDG\.E\.STRONG\.{wrong_scope}\s+R", body))
            expected_target_loads = (
                chains * SASS_TARGET_LOAD_SITES_BY_TIER[tier])
            if target_loads != expected_target_loads or wrong_loads != 0:
                raise ValueError(
                    f"{name} cache policy/chain count mismatch: "
                    f"target={target_loads}, expected={expected_target_loads}, "
                    f"wrong={wrong_loads}")
            result[name] = {
                "chains": chains,
                "clock_reads": clock_reads,
                "head_load_instructions": head_loads,
                "target_load_instructions": target_loads,
                "local_memory_instructions": 0,
            }
    return result


def validate_ptxas(text):
    expected_names = {
        _kernel_name(tier, chains)
        for tier in TIERS for chains in CHAIN_POINTS
    }
    seen = set(re.findall(r"Function properties for (dependency_mlp_\S+)", text))
    if seen != expected_names:
        raise ValueError(
            f"PTXAS kernel matrix mismatch: missing={sorted(expected_names-seen)}, "
            f"extra={sorted(seen-expected_names)}")
    spills = re.findall(
        r"(\d+) bytes spill stores, (\d+) bytes spill loads", text)
    if len(spills) != len(expected_names):
        raise ValueError("PTXAS output lacks one spill record per kernel")
    if any(int(stores) != 0 or int(loads) != 0 for stores, loads in spills):
        raise ValueError("PTXAS reports chain-state spills")


def _parse_ncu_dimension(text, label):
    value = _require_nonempty(text, label)
    match = re.fullmatch(
        r"\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*\)",
        value)
    if match is None:
        raise ValueError(f"{label} has invalid syntax: {value!r}")
    return tuple(int(match.group(index)) for index in range(1, 4))


def _validate_ncu_launch(
        row, expected_kernel, expected_grid, expected_block_threads):
    if row["Kernel Name"] != expected_kernel:
        raise ValueError(
            f"NCU selected {row['Kernel Name']!r}, expected "
            f"{expected_kernel!r}")
    grid = _parse_ncu_dimension(row["Grid Size"], "NCU grid size")
    block = _parse_ncu_dimension(row["Block Size"], "NCU block size")
    if grid != (expected_grid, 1, 1):
        raise ValueError(
            f"NCU grid {grid!r} does not match {(expected_grid, 1, 1)!r}")
    if block != (expected_block_threads, 1, 1):
        raise ValueError(
            "NCU block does not match the selected producer warps: "
            f"{block!r} != {(expected_block_threads, 1, 1)!r}")


def parse_ncu_csv(
        text, expected_kernel, expected_grid, expected_block_threads):
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith('"ID","Process ID"'):
            header_index = index
            break
    if header_index is None:
        raise ValueError("NCU CSV lacks its metric header")
    reader = csv.DictReader(lines[header_index:])
    if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)):
        raise ValueError("NCU CSV header is absent or contains duplicates")
    fields = set(reader.fieldnames)
    rows = tuple(reader)
    if not rows:
        raise ValueError("NCU CSV contains no metric rows")
    launch_fields = {"Kernel Name", "Grid Size", "Block Size"}
    wide_fields = launch_fields | set(NCU_METRICS)
    tall_fields = launch_fields | {
        "Metric Name", "Metric Unit", "Metric Value"}
    if wide_fields <= fields:
        unit_rows = tuple(row for row in rows if not row["Kernel Name"])
        data_rows = tuple(row for row in rows if row["Kernel Name"])
        if len(unit_rows) != 1 or len(data_rows) != 1:
            raise ValueError(
                "wide NCU CSV requires exactly one unit row and one data row")
        unit_row = unit_rows[0]
        data_row = data_rows[0]
        _validate_ncu_launch(
            data_row, expected_kernel, expected_grid, expected_block_threads)
        for metric in NCU_METRICS:
            if unit_row[metric] != NCU_METRIC_UNITS[metric]:
                raise ValueError(
                    f"NCU {metric} unit changed: {unit_row[metric]!r}")
        metrics = {
            metric: _parse_int(data_row[metric], f"NCU {metric}")
            for metric in NCU_METRICS
        }
    elif tall_fields <= fields:
        metrics = {}
        for row in rows:
            _validate_ncu_launch(
                row, expected_kernel, expected_grid, expected_block_threads)
            metric = row["Metric Name"]
            if metric in metrics:
                raise ValueError(f"NCU CSV repeats metric {metric!r}")
            if metric not in NCU_METRIC_UNITS:
                raise ValueError(f"NCU CSV has unknown metric {metric!r}")
            if row["Metric Unit"] != NCU_METRIC_UNITS[metric]:
                raise ValueError(
                    f"NCU {metric} unit changed: {row['Metric Unit']!r}")
            metrics[metric] = _parse_int(
                row["Metric Value"], f"NCU {metric}")
        if set(metrics) != set(NCU_METRICS):
            raise ValueError(
                "NCU metric schema mismatch: "
                f"missing={sorted(set(NCU_METRICS)-set(metrics))}, "
                f"extra={sorted(set(metrics)-set(NCU_METRICS))}")
    else:
        raise ValueError(
            "NCU CSV lacks either the wide or tall metric schema")
    return metrics


def validate_ncu_contract(tier, expected, metrics):
    expected_loads = expected["counter_load_warp_instructions"]
    instruction_metric = "smsp__sass_inst_executed_op_global_ld.sum"
    sector_metric = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
    hit_metric = "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit.sum"
    miss_metric = "l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
    dram_metric = "dram__sectors_read.sum"
    if metrics[instruction_metric] != expected_loads:
        raise ValueError(
            f"NCU load instruction count {metrics[instruction_metric]} "
            f"does not equal source count {expected_loads}")
    if metrics[sector_metric] != expected["counter_load_sectors"]:
        raise ValueError("NCU load sectors do not equal one sector per instruction")
    target_bytes = expected["target_request_bytes"]
    hit_bytes = metrics[hit_metric]
    miss_bytes = metrics[miss_metric]
    if hit_bytes + miss_bytes != metrics[sector_metric] * SECTOR_BYTES:
        raise ValueError("NCU L1 hit/miss bytes do not conserve load sectors")
    if tier == "l1_warm_default":
        if hit_bytes < target_bytes:
            raise ValueError("L1 tier target recurrence is not fully L1-hit")
    elif tier == "l2_warm_cg":
        if miss_bytes < target_bytes:
            raise ValueError("L2 tier target recurrence did not bypass L1")
        # The timed target is L2-warm.  Bound compulsory DRAM by one complete
        # 128-byte-line fill of every allocation reachable by this tier:
        # nodes, heads, sink, and result.  Their 0.97-MiB total is far below
        # the 40-MiB L2.  This allocation-footprint bound admits cache-line
        # promotion and write allocation without allowing target-loop DRAM
        # traffic to scale with the 113M repeated recurrence advances.
        if (metrics[dram_metric]
                > MAXIMUM_L2_TIER_COMPULSORY_DRAM_SECTORS):
            raise ValueError("L2 tier target recurrence is not L2-resident")
    elif tier == "dram_cg":
        if miss_bytes < target_bytes:
            raise ValueError("DRAM tier target recurrence did not bypass L1")
        # A 256-MiB unique traversal is 6.4x the 40-MiB L2.  Even allowing
        # replay/cache transients, at least 75% of target sectors must reach
        # DRAM; this gate is derived from capacity, not evaluation error.
        minimum_dram_sectors = math.floor(
            expected["target_chain_advances"] * 0.75)
        if metrics[dram_metric] < minimum_dram_sectors:
            raise ValueError("DRAM tier target recurrence remained cache-resident")
    else:
        raise ValueError(f"unknown NCU tier {tier!r}")


def _query_device(nvidia_smi, device_uuid, cwd):
    result = _run([
        nvidia_smi,
        f"--id={device_uuid}",
        "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,"
        "utilization.gpu,compute_mode",
        "--format=csv,noheader,nounits",
    ], cwd=cwd)
    rows = tuple(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise ValueError("nvidia-smi returned an unexpected device row")
    row = tuple(value.strip() for value in rows[0])
    return {
        "index": _parse_int(row[0], "GPU index"),
        "uuid": row[1],
        "pci_bus_id": row[2],
        "memory_used_mib": _parse_int(row[3], "GPU memory used"),
        "memory_total_mib": _parse_int(row[4], "GPU memory total"),
        "utilization_gpu_pct": _parse_int(row[5], "GPU utilization"),
        "compute_mode": row[6],
    }


def _require_idle(device, expected_uuid, expected_pci):
    if device["uuid"] != expected_uuid:
        raise ValueError("nvidia-smi UUID disagrees with the requested device")
    if (_normalize_pci_bus_id(device["pci_bus_id"])
            != _normalize_pci_bus_id(expected_pci)):
        raise ValueError("nvidia-smi PCI bus ID disagrees with the request")
    if device["memory_used_mib"] != 0 or device["utilization_gpu_pct"] != 0:
        raise RuntimeError(
            "requested GPU is not idle: "
            f"memory={device['memory_used_mib']} MiB, "
            f"util={device['utilization_gpu_pct']}%")


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_manifest(output_dir):
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.sha256":
            continue
        entries.append(f"{_sha256(path)}  {path.relative_to(output_dir)}")
    manifest = output_dir / "manifest.sha256"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")


def capture(arguments):
    repo_root = arguments.repo_root.resolve()
    output_dir = arguments.output_dir.resolve()
    source = repo_root / "microbench/nvidia_a800/memory/lsu" / SOURCE_NAME
    runner = repo_root / "microbench/nvidia_a800/memory/lsu" / RUNNER_NAME
    for path in (source, runner, arguments.nvcc, arguments.cuobjdump,
                 arguments.ncu, arguments.nvidia_smi):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    baseline = _git_output(repo_root, "rev-parse", "HEAD")
    for path in (source, runner):
        relative = str(path.relative_to(repo_root))
        diff = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=repo_root)
        if diff.returncode != 0:
            raise RuntimeError(
                f"capture source {relative} differs from baseline {baseline}")
        tracked = _git_output(repo_root, "ls-files", "--error-unmatch", relative)
        if tracked != relative:
            raise RuntimeError(f"capture source is not tracked: {relative}")

    preflight = _query_device(
        str(arguments.nvidia_smi), arguments.device_uuid, repo_root)
    _require_idle(preflight, arguments.device_uuid, arguments.pci_bus_id)
    output_dir.mkdir(parents=True)
    build_dir = output_dir / "build"
    native_dir = output_dir / "native"
    ncu_dir = output_dir / "ncu"
    snapshot_dir = output_dir / "source_snapshot"
    for directory in (build_dir, native_dir, ncu_dir, snapshot_dir):
        directory.mkdir()
    shutil.copy2(source, snapshot_dir / SOURCE_NAME)
    shutil.copy2(runner, snapshot_dir / RUNNER_NAME)
    (output_dir / "baseline_commit.txt").write_text(
        baseline + "\n", encoding="utf-8")
    (output_dir / "tracked_worktree_status.txt").write_text(
        _git_output(repo_root, "status", "--short") + "\n",
        encoding="utf-8")
    _write_json(output_dir / "preflight_device.json", preflight)
    _write_json(output_dir / "capture_identity.json", {
        "hardware_id": HARDWARE_ID,
        "owner": "root/main_agent",
        "baseline_commit": baseline,
        "run_id": arguments.run_id,
        "device_uuid": arguments.device_uuid,
        "pci_bus_id": arguments.pci_bus_id,
        "command": sys.argv,
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "output_path": str(output_dir),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })

    binary = build_dir / "dependency_mlp"
    ptxas_stderr = build_dir / "ptxas.stderr"
    _run([
        str(arguments.nvcc), "-O3", "-std=c++14", "-arch=sm_80",
        "-Xptxas=-v", str(snapshot_dir / SOURCE_NAME), "-o", str(binary),
    ], cwd=repo_root, stdout_path=build_dir / "compile.stdout",
       stderr_path=ptxas_stderr)
    validate_ptxas(ptxas_stderr.read_text(encoding="utf-8"))
    sass_path = build_dir / "dependency_mlp.sass"
    _run([
        str(arguments.cuobjdump), "--dump-sass", str(binary),
    ], cwd=repo_root, stdout_path=sass_path,
       stderr_path=build_dir / "cuobjdump.stderr")
    sass_audit = validate_sass(sass_path.read_text(encoding="utf-8"))
    _write_json(build_dir / "sass_audit.json", sass_audit)

    execution_env = os.environ.copy()
    execution_env["CUDA_VISIBLE_DEVICES"] = arguments.device_uuid
    native_output = native_dir / "sweep.csv"
    _run([
        str(binary), "--trials", str(arguments.trials),
    ], cwd=repo_root, env=execution_env, stdout_path=native_output,
       stderr_path=native_dir / "sweep.stderr")
    native = parse_native_output(
        native_output.read_text(encoding="utf-8"), arguments.trials)
    if native["meta"]["device_uuid"] != arguments.device_uuid:
        raise ValueError("native CUDA UUID disagrees with preflight")
    if (_normalize_pci_bus_id(native["meta"]["pci_bus_id"])
            != _normalize_pci_bus_id(arguments.pci_bus_id)):
        raise ValueError("native CUDA PCI bus ID disagrees with preflight")

    ncu_audit = {}
    metric_argument = ",".join(NCU_METRICS)
    for tier in TIERS:
        for warps, chains in BOUNDARY_POINTS:
            kernel = _kernel_name(tier, chains)
            cell_name = f"{tier}_w{warps}_c{chains}"
            csv_path = ncu_dir / f"{cell_name}.csv"
            report_path = ncu_dir / cell_name
            _run([
                str(arguments.ncu),
                "--kernel-name-base", "function",
                "--kernel-name", f"regex:^{kernel}$",
                "--launch-count", "1",
                "--metrics", metric_argument,
                "--page", "raw", "--csv", "--print-units", "base",
                "--log-file", str(csv_path),
                "--export", str(report_path), "--force-overwrite",
                str(binary), "--profile", "--tier", tier,
                "--chains", str(chains), "--warps", str(warps),
            ], cwd=repo_root, env=execution_env,
               stdout_path=ncu_dir / f"{cell_name}.stdout",
               stderr_path=ncu_dir / f"{cell_name}.stderr")
            expected = native["expected"][(tier, chains, warps)]
            metrics = parse_ncu_csv(
                csv_path.read_text(encoding="utf-8"), kernel,
                expected["grid"], expected["warps"] * 32)
            validate_ncu_contract(tier, expected, metrics)
            ncu_audit[cell_name] = {
                "tier": tier,
                "producer_warps_per_sm": warps,
                "independent_chains_per_warp": chains,
                "kernel": kernel,
                "expected": expected,
                "metrics": metrics,
            }
    _write_json(ncu_dir / "audit.json", ncu_audit)

    postflight = _query_device(
        str(arguments.nvidia_smi), arguments.device_uuid, repo_root)
    if postflight["uuid"] != preflight["uuid"]:
        raise ValueError("postflight device UUID changed")
    _write_json(output_dir / "postflight_device.json", postflight)
    _write_json(output_dir / "report.json", {
        "schema": "nvidia_a800_dependency_mlp_capture_v1",
        "mechanism": "finite_source_dependent_global_load_recurrence",
        "baseline_commit": baseline,
        "source_sha256": _sha256(snapshot_dir / SOURCE_NAME),
        "runner_sha256": _sha256(snapshot_dir / RUNNER_NAME),
        "binary_sha256": _sha256(binary),
        "sass_sha256": _sha256(sass_path),
        "native_sha256": _sha256(native_output),
        "device": preflight,
        "matrix": {
            "cache_tiers": TIERS,
            "producer_warps_per_sm": WARP_POINTS,
            "independent_chains_per_warp": CHAIN_POINTS,
            "native_trials_per_cell": arguments.trials,
            "ncu_boundary_points": BOUNDARY_POINTS,
        },
        "rate_unit": (
            RATE_UNIT),
        "production_or_evaluation_duration_consumed": False,
        "production_parameter_published": False,
    })
    _write_manifest(output_dir)


def _arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device-uuid", required=True)
    parser.add_argument("--pci-bus-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--nvcc", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/nvcc"))
    parser.add_argument(
        "--cuobjdump", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/cuobjdump"))
    parser.add_argument(
        "--ncu", type=Path,
        default=Path("/usr/local/cuda-12.2/bin/ncu"))
    parser.add_argument(
        "--nvidia-smi", type=Path,
        default=Path("/usr/bin/nvidia-smi"))
    parser.add_argument("--trials", type=int, default=9)
    arguments = parser.parse_args(argv)
    arguments.run_id = _require_nonempty(arguments.run_id, "run_id")
    arguments.device_uuid = _require_nonempty(
        arguments.device_uuid, "device_uuid")
    arguments.pci_bus_id = _require_nonempty(
        arguments.pci_bus_id, "pci_bus_id")
    if (isinstance(arguments.trials, bool)
            or not isinstance(arguments.trials, int)
            or arguments.trials <= 0):
        raise ValueError("trials must be a positive integer")
    return arguments


def main(argv=None):
    capture(_arguments(argv))


if __name__ == "__main__":
    main()
